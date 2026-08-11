"""FastAPI application wiring.

The app is constructed in ``create_app`` so it is testable without import side
effects, and so the registry/services are created once per process. Error
handlers translate ``GatewayError`` subclasses into the consistent JSON envelope
defined in ``core/exceptions``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from mm_gateway.config import Settings
from mm_gateway.core.exceptions import GatewayError
from mm_gateway.observability.httplog import frontend_request_log, frontend_response_log
from mm_gateway.observability.logging import (
    bind_context,
    clear_context,
    configure_logging,
    get_logger,
    new_request_id,
)
from mm_gateway.registry import Registry
from mm_gateway.services import ImageService, MusicService, VideoService
from mm_gateway.tasks.store import TaskStore

log = get_logger("app")


def _problem_response(
    request: Request,
    *,
    status: int,
    code: str,
    detail: str,
    errors: list[dict] | None = None,
) -> JSONResponse:
    payload = {
        "type": f"urn:mm-gateway:problem:{code}",
        "title": code.replace("_", " ").title(),
        "status": status,
        "detail": detail,
        "instance": request.url.path,
        "code": code,
        "request_id": getattr(request.state, "request_id", None),
    }
    if errors:
        payload["errors"] = errors
    return JSONResponse(
        status_code=status,
        content=payload,
        media_type="application/problem+json",
    )


def create_app(
    settings: Settings | None = None,
    *,
    task_store: TaskStore | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    configure_logging(level=settings.log_level, fmt=settings.log_format)

    registry = Registry(settings)
    image_service = ImageService(
        registry, max_sync_wait=settings.max_sync_wait,
        poll_interval=settings.poll_interval, sync_default=settings.image_sync_default,
    )
    video_service = VideoService(
        registry, max_sync_wait=settings.max_sync_wait,
        poll_interval=settings.poll_interval, sync_default=settings.video_sync_default,
    )
    music_service = MusicService(
        registry, max_sync_wait=settings.max_sync_wait,
        poll_interval=settings.poll_interval, sync_default=settings.music_sync_default,
    )
    task_store = task_store or TaskStore()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log.info("gateway_starting", host=settings.host, port=settings.port,
                 providers=list(registry.providers.keys()))
        yield
        log.info("gateway_stopping")

    app = FastAPI(
        title="mm-gateway",
        description="Unified image, video, and music gateway with separate REST APIs.",
        version="0.1.0", lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.registry = registry
    app.state.image_service = image_service
    app.state.video_service = video_service
    app.state.music_service = music_service
    app.state.task_store = task_store

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or new_request_id()
        request.state.request_id = request_id
        bind_context(request_id=request_id)
        # Inbound request log: curl format (masked sensitive headers) + body.
        # Reading ``request.body()`` caches it in Starlette's
        # ``BaseHTTPMiddleware`` request wrapper, which re-serves the cached
        # body to downstream handlers automatically — no manual re-injection.
        try:
            raw_body = await request.body()
        except Exception:  # noqa: BLE001
            raw_body = None
        frontend_request_log(
            request.method, str(request.url), request.headers, raw_body,
        )
        # If the downstream raises, the response never streams, so clear the
        # context now — the streaming wrapper below is never installed.
        try:
            response = await call_next(request)
        except Exception:
            clear_context()
            raise
        # Outbound response body log. Wrap the streaming body iterator so chunks
        # flow through to the client unchanged — eagerly buffering the body
        # would deadlock streaming endpoints (the downstream runs in a task
        # group that only produces chunks while the response streams out, e.g.
        # the MCP SSE transport). Capture chunks as they pass and log once the
        # stream completes. ``clear_context`` is deferred to the end of the
        # stream (and re-binds ``request_id`` on the consumer task, which may
        # differ from this one) so the response log stays correlated to its
        # request.
        captured = bytearray()
        original_iter = response.body_iterator

        async def logged_iter():
            bind_context(request_id=request_id)
            try:
                async for chunk in original_iter:
                    if isinstance(chunk, (bytes, bytearray)):
                        captured.extend(chunk)
                    elif isinstance(chunk, str):
                        captured.extend(chunk.encode("utf-8"))
                    yield chunk
                frontend_response_log(response.status_code, response.headers, bytes(captured))
            finally:
                clear_context()

        response.body_iterator = logged_iter()
        response.headers["x-request-id"] = request_id
        return response

    @app.exception_handler(GatewayError)
    async def gateway_error_handler(request: Request, exc: GatewayError):
        log.warning("request_error", code=exc.code, provider=exc.provider, message=exc.message)
        return _problem_response(
            request,
            status=exc.status_code,
            code=exc.public_code,
            detail=exc.public_message,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        return _problem_response(
            request,
            status=422,
            code="validation_error",
            detail="Request validation failed.",
            errors=jsonable_encoder(exc.errors()),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, exc: StarletteHTTPException):
        code = "not_found" if exc.status_code == 404 else "method_not_allowed" \
            if exc.status_code == 405 else "http_error"
        response = _problem_response(
            request,
            status=exc.status_code,
            code=code,
            detail=str(exc.detail),
        )
        response.headers.update(exc.headers or {})
        return response

    # Register routes.
    from mm_gateway.server.routes import (
        image_routes,
        meta_routes,
        music_routes,
        video_routes,
    )
    app.include_router(meta_routes.router, tags=["meta"])
    app.include_router(image_routes.router)
    app.include_router(video_routes.router)
    app.include_router(music_routes.router)

    # Optionally mount the HTTP MCP endpoint (no-op when mcp_enabled is false).
    from mm_gateway.server.mcp import mount_mcp
    mount_mcp(app, settings)

    _install_openapi_customization(app)
    return app


# Paths that require a front-end API key (everything except health/metrics).
_OPEN_PATHS = {"/health", "/metrics"}

# RFC 9457 response blocks, keyed by HTTP status. Every protected operation
# uses the same ProblemDetail schema and ``application/problem+json`` media type.
_ERROR_RESPONSES: dict[str, dict] = {
    "400": {
        "description": "Invalid request (invalid_request_error / unsupported_feature).",
    },
    "401": {
        "description": "Missing or unknown API key (unauthorized).",
    },
    "403": {
        "description": "Key not allowed to perform the request (forbidden).",
    },
    "409": {
        "description": "Idempotency key conflicts with an earlier request.",
    },
    "404": {
        "description": "Model or task not found.",
    },
    "422": {
        "description": "Request validation failed or a task failed.",
    },
    "502": {
        "description": "Generation service returned an error.",
    },
    "503": {
        "description": "No usable generation service is configured.",
    },
    "504": {
        "description": "Generation service timed out.",
    },
}

# Concrete Problem Details examples (one per status) so the published spec shows
# clients the exact failure body, not just a $ref.
_ERROR_EXAMPLES: dict[str, dict] = {
    "400": {"summary": "Invalid request", "code": "invalid_request_error", "detail": "The routing profile is unavailable."},
    "401": {"summary": "Missing API key", "code": "unauthorized", "detail": "Missing API key. Send an 'Authorization: Bearer <token>' header."},
    "403": {"summary": "Forbidden", "code": "forbidden", "detail": "The API key is not allowed to perform this request."},
    "409": {"summary": "Idempotency conflict", "code": "idempotency_conflict", "detail": "The Idempotency-Key was already used with a different request."},
    "404": {"summary": "Unknown task", "code": "task_not_found", "detail": "image task img_unknown not found"},
    "422": {
        "summary": "Validation failed",
        "code": "validation_error",
        "detail": "Request validation failed.",
        "errors": [{"type": "missing", "loc": ["body", "model"], "msg": "Field required"}],
    },
    "502": {"summary": "Generation service error", "code": "generation_service_error", "detail": "The generation service returned an error."},
    "503": {"summary": "Generation service unavailable", "code": "generation_service_unavailable", "detail": "No generation service is available for this request."},
    "504": {"summary": "Generation service timeout", "code": "generation_service_timeout", "detail": "The generation service timed out."},
}

# Concrete example values for each 2xx success-response schema, keyed by the
# schema name (the last path segment of a "#/components/schemas/<name>" $ref).
# Injected into each operation's 200 media block so the published spec shows
# clients a worked response body, not just a bare $ref.
_SUCCESS_EXAMPLES: dict[str, dict] = {
    "HealthResponse": {"status": "ok"},
    "ModelListResponse": {
        "object": "list",
        "data": [
            {"id": "gateway-image-pro", "object": "model", "modality": "image"},
            {"id": "gateway-video-pro", "object": "model", "modality": "video"},
            {"id": "gateway-music-lyria", "object": "model", "modality": "music"},
        ],
    },
}

_RESOURCE_EXAMPLES: dict[tuple[str, str, str], dict] = {
    ("/v1/images", "post", "202"): {
        "id": "img_01HZX4J3K7NQ8X2V9Y6R5W4T3P",
        "object": "image",
        "model": "gateway-image-pro",
        "status": "pending",
        "outputs": [],
        "metadata": {"requester": "design-tool"},
        "created_at": "2026-08-09T12:00:00Z",
        "links": {"self": "https://gateway.example.test/v1/images/img_01HZX4J3K7NQ8X2V9Y6R5W4T3P"},
    },
    ("/v1/images/{image_id}", "get", "200"): {
        "id": "img_01HZX4J3K7NQ8X2V9Y6R5W4T3P",
        "object": "image",
        "model": "gateway-image-pro",
        "status": "succeeded",
        "outputs": [{
            "uri": "https://cdn.example.test/img/abc.png",
            "mime_type": "image/png",
            "revised_prompt": "a cyberpunk cat in the rain",
        }],
        "usage": {"cost": 0.02, "output_count": 1},
        "metadata": {"requester": "design-tool"},
        "created_at": "2026-08-09T12:00:00Z",
        "completed_at": "2026-08-09T12:00:04Z",
        "links": {"self": "https://gateway.example.test/v1/images/img_01HZX4J3K7NQ8X2V9Y6R5W4T3P"},
    },
    ("/v1/videos", "post", "202"): {
        "id": "vid_01HZX4J3K7NQ8X2V9Y6R5W4T3P",
        "object": "video",
        "model": "gateway-video-pro",
        "status": "pending",
        "outputs": [],
        "metadata": {},
        "created_at": "2026-08-09T12:00:00Z",
        "links": {"self": "https://gateway.example.test/v1/videos/vid_01HZX4J3K7NQ8X2V9Y6R5W4T3P"},
    },
    ("/v1/videos/{video_id}", "get", "200"): {
        "id": "vid_01HZX4J3K7NQ8X2V9Y6R5W4T3P",
        "object": "video",
        "model": "gateway-video-pro",
        "status": "succeeded",
        "outputs": [{
            "uri": "https://cdn.example.test/vid/abc.mp4",
            "cover_uri": "https://cdn.example.test/vid/abc-last.png",
            "mime_type": "video/mp4",
        }],
        "usage": {"cost": 0.08, "output_count": 1, "duration_seconds": 5},
        "metadata": {},
        "created_at": "2026-08-09T12:00:00Z",
        "completed_at": "2026-08-09T12:00:20Z",
        "links": {"self": "https://gateway.example.test/v1/videos/vid_01HZX4J3K7NQ8X2V9Y6R5W4T3P"},
    },
    ("/v1/music", "post", "202"): {
        "id": "mus_01HZX4J3K7NQ8X2V9Y6R5W4T3P",
        "object": "music",
        "model": "gateway-music-lyria",
        "status": "pending",
        "outputs": [],
        "metadata": {},
        "created_at": "2026-08-09T12:00:00Z",
        "links": {"self": "https://gateway.example.test/v1/music/mus_01HZX4J3K7NQ8X2V9Y6R5W4T3P"},
    },
    ("/v1/music/{music_id}", "get", "200"): {
        "id": "mus_01HZX4J3K7NQ8X2V9Y6R5W4T3P",
        "object": "music",
        "model": "gateway-music-lyria",
        "status": "succeeded",
        "outputs": [{
            "uri": "https://cdn.example.test/mus/abc.wav",
            "mime_type": "audio/wav",
        }],
        "lyrics": "[verse]\nWalking down the street...",
        "usage": {"cost": 0.05, "output_count": 1, "duration_seconds": 30},
        "metadata": {},
        "created_at": "2026-08-09T12:00:00Z",
        "completed_at": "2026-08-09T12:00:30Z",
        "links": {"self": "https://gateway.example.test/v1/music/mus_01HZX4J3K7NQ8X2V9Y6R5W4T3P"},
    },
}


def _install_openapi_customization(app: FastAPI) -> None:
    """Override ``app.openapi`` to add the Bearer security scheme, mark protected
    routes as requiring it, and attach the gateway Problem Details responses.

    Done as a post-process of FastAPI's generated schema so the route decorators
    stay the single source of truth for request/response shapes; this layer only
    adds cross-cutting security + error metadata.
    """
    from mm_gateway.schemas.api import ProblemDetail

    def custom_openapi() -> dict:
        if app.openapi_schema:
            return app.openapi_schema
        spec = FastAPI.openapi(app)

        # Register the Problem Details component referenced by error responses.
        # Use the app's own schema generator so nested refs resolve against
        # #/components/schemas consistently with the rest of the spec, then hoist
        # any $defs it emits into the top-level components block.
        schemas = spec.setdefault("components", {}).setdefault("schemas", {})
        if "ProblemDetail" not in schemas:
            problem_schema = ProblemDetail.model_json_schema(
                ref_template="#/components/schemas/{model}"
            )
            for name, sub in problem_schema.pop("$defs", {}).items():
                schemas.setdefault(name, sub)
            schemas["ProblemDetail"] = problem_schema

        # Bearer auth scheme.
        spec.setdefault("components", {}).setdefault("securitySchemes", {})["BearerAuth"] = {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "API key",
            "description": 'Front-end API key sent as "Authorization: Bearer <token>".',
        }

        problem_ref = {"$ref": "#/components/schemas/ProblemDetail"}
        for path, item in spec.get("paths", {}).items():
            for method, op in item.items():
                if method not in ("get", "post", "put", "patch", "delete"):
                    continue
                if path not in _OPEN_PATHS:
                    op.setdefault("security", [{"BearerAuth": []}])
                    responses = op.setdefault("responses", {})
                    # Every protected op can return these problem statuses.
                    # Ensure each carries both the shared schema and an example;
                    # route decorators may
                    # declare a bare {"description": ...} for some codes, so
                    # set the schema/example on the media block directly rather
                    # than setdefault-ing the whole response block.
                    error_codes = ["400", "401", "403", "404", "422", "502", "503", "504"]
                    if method == "post" and path in {"/v1/images", "/v1/videos", "/v1/music"}:
                        error_codes.append("409")
                    for code in error_codes:
                        resp = responses.setdefault(
                            code, {"description": _ERROR_RESPONSES[code]["description"]}
                        )
                        resp.setdefault("description", _ERROR_RESPONSES[code]["description"])
                        content = resp.setdefault("content", {})
                        content.pop("application/json", None)
                        media = content.setdefault("application/problem+json", {})
                        media["schema"] = problem_ref
                        example = _ERROR_EXAMPLES[code]
                        value = {
                            "type": f"urn:mm-gateway:problem:{example['code']}",
                            "title": example["code"].replace("_", " ").title(),
                            "status": int(code),
                            "detail": example["detail"],
                            "instance": path,
                            "code": example["code"],
                            "request_id": "req_01HZX4J3K7NQ8X2V9Y6R5W4T3P",
                        }
                        if example.get("errors"):
                            value["errors"] = example["errors"]
                        media.setdefault("examples", {}).setdefault(
                            "error",
                            {"summary": example["summary"], "value": value},
                        )

                # Attach a worked example to each JSON success-response media
                # block, keyed off the schema $ref so clients see a concrete
                # response body in the spec, not just a bare $ref.
                for code, resp in op.get("responses", {}).items():
                    if not str(code).startswith("2"):
                        continue
                    media = resp.get("content", {}).get("application/json")
                    if not media:
                        continue
                    ref = media.get("schema", {}).get("$ref", "")
                    schema_name = ref.rsplit("/", 1)[-1] if ref else ""
                    ex = _RESOURCE_EXAMPLES.get((path, method, str(code)))
                    if ex is None:
                        ex = _SUCCESS_EXAMPLES.get(schema_name)
                    if ex and "examples" not in media and "example" not in media:
                        media.setdefault("example", ex)

        # FastAPI generates these for its default 422 envelope. Every public
        # operation uses ProblemDetail instead, so leaving them in components
        # would publish an unreachable second validation format.
        schemas.pop("HTTPValidationError", None)
        schemas.pop("ValidationError", None)

        # Make the required/optional split explicit: Pydantic omits the
        # `required` key when every field is optional, which reads ambiguously
        # in a published spec. Emit an explicit `required: []` for every object
        # schema that has properties but no required list, so consumers can see
        # at a glance that all fields are optional.
        for sch in schemas.values():
            if "properties" in sch and "required" not in sch:
                sch["required"] = []

        app.openapi_schema = spec
        return spec

    app.openapi = custom_openapi  # type: ignore[assignment]


app = create_app()


def run() -> None:
    settings = Settings.from_env()
    uvicorn.run("mm_gateway.server.app:app", host=settings.host, port=settings.port,
                log_level=settings.log_level.lower())


if __name__ == "__main__":
    run()
