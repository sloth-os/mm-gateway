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
from fastapi.responses import JSONResponse

from mm_gateway.config import Settings
from mm_gateway.core.exceptions import GatewayError
from mm_gateway.observability.httplog import frontend_request_log, frontend_response_log
from mm_gateway.observability.logging import bind_context, clear_context, configure_logging, get_logger, new_request_id
from mm_gateway.observability.metrics import render_prometheus
from mm_gateway.registry import Registry
from mm_gateway.services import ImageService, MusicService, VideoService
from mm_gateway.tasks.store import TaskStore

log = get_logger("app")


def create_app(settings: Settings | None = None) -> FastAPI:
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
    task_store = TaskStore()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log.info("gateway_starting", host=settings.host, port=settings.port,
                 providers=list(registry.providers.keys()))
        yield
        log.info("gateway_stopping")

    app = FastAPI(
        title="mm-gateway",
        description="Unified image / video / AI gateway. OpenAI- and OpenRouter-compatible.",
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
    async def gateway_error_handler(_: Request, exc: GatewayError):
        log.warning("request_error", code=exc.code, provider=exc.provider, message=exc.message)
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

    # Register routes.
    from mm_gateway.server.routes import image_routes, video_routes, music_routes, meta_routes
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

# Error-envelope response blocks, keyed by HTTP status. Every protected
# operation can return any of these; the body is the gateway's consistent
# {"error": {"code", "message", [provider], [details]}} envelope.
_ERROR_RESPONSES: dict[str, dict] = {
    "400": {
        "description": "Invalid request (invalid_request_error / unsupported_feature).",
        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorEnvelope"}}},
    },
    "401": {
        "description": "Missing or unknown API key (unauthorized).",
        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorEnvelope"}}},
    },
    "403": {
        "description": "Key not allowed to use the requested backend (forbidden).",
        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorEnvelope"}}},
    },
    "404": {
        "description": "Model, provider, or task not found.",
        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorEnvelope"}}},
    },
    "502": {
        "description": "Backend returned an error (provider_error).",
        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorEnvelope"}}},
    },
    "504": {
        "description": "Backend timed out (provider_timeout).",
        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorEnvelope"}}},
    },
}

# Concrete error-envelope examples (one per status) so the published spec shows
# clients the exact failure body, not just a $ref.
_ERROR_EXAMPLES: dict[str, dict] = {
    "400": {"summary": "Invalid request", "value": {"error": {"code": "invalid_request_error", "message": "`model` is required for image generation."}}},
    "401": {"summary": "Missing API key", "value": {"error": {"code": "unauthorized", "message": "Missing API key. Send an 'Authorization: Bearer <token>' header."}}},
    "403": {"summary": "Forbidden backend", "value": {"error": {"code": "forbidden", "message": "API key 'bob' is not allowed to use backend 'vid-a'.", "provider": "vid-a"}}},
    "404": {"summary": "Unknown task", "value": {"error": {"code": "task_not_found", "message": "image task img-999 not found", "provider": "fake"}}},
    "502": {"summary": "Provider error", "value": {"error": {"code": "provider_error", "message": "upstream returned 500", "provider": "openai"}}},
    "504": {"summary": "Provider timeout", "value": {"error": {"code": "provider_timeout", "message": "upstream timed out", "provider": "volcengine"}}},
}

# Concrete example values for each 2xx success-response schema, keyed by the
# schema name (the last path segment of a "#/components/schemas/<name>" $ref).
# Injected into each operation's 200 media block so the published spec shows
# clients a worked response body, not just a bare $ref.
_SUCCESS_EXAMPLES: dict[str, dict] = {
    "HealthResponse": {"status": "ok"},
    "CreateResponse": {"id": "img-01HZX4J3K7NQ8X2V9Y6R5W4T3P"},
    "ImageTaskResponse": {
        "id": "img-01HZX4J3K7NQ8X2V9Y6R5W4T3P",
        "model": "gateway-image-pro",
        "status": "succeeded",
        "steps": [
            {
                "type": "model_output",
                "content": [
                    {"type": "image", "url": "https://cdn.example.test/img/abc.png", "mime_type": "image/png"},
                ],
            }
        ],
        "output_image_url": "https://cdn.example.test/img/abc.png",
        "usage": {"cost": 0.02},
        "created_at": "2026-08-09T12:00:00Z",
        "completed_at": "2026-08-09T12:00:04Z",
    },
    "VideoTaskResponse": {
        "id": "vid-01HZX4J3K7NQ8X2V9Y6R5W4T3P",
        "model": "gateway-video-pro",
        "status": "succeeded",
        "content": {
            "video_url": "https://cdn.example.test/vid/abc.mp4",
            "last_frame_url": "https://cdn.example.test/vid/abc-last.png",
        },
        "usage": {"cost": 0.08},
    },
    "MusicTaskResponse": {
        "id": "mus-01HZX4J3K7NQ8X2V9Y6R5W4T3P",
        "model": "gateway-music-lyria",
        "status": "succeeded",
        "steps": [
            {
                "type": "model_output",
                "content": [
                    {"type": "audio", "url": "https://cdn.example.test/mus/abc.wav", "mime_type": "audio/wav"},
                    {"type": "text", "text": "[verse]\nWalking down the street..."},
                ],
            }
        ],
        "output_audio_url": "https://cdn.example.test/mus/abc.wav",
        "output_text": "[verse]\nWalking down the street...",
        "usage": {"cost": 0.05},
    },
    "ModelListResponse": {
        "object": "list",
        "data": [
            {"id": "gateway-image-pro", "modality": "image", "type": "alias", "underlying": "dall-e-3"},
            {"id": "gateway-video-pro", "modality": "video", "type": "alias", "underlying": "seedance-1-0"},
            {"id": "gateway-music-lyria", "modality": "music", "type": "alias", "underlying": "lyria-2"},
        ],
    },
}


def _install_openapi_customization(app: FastAPI) -> None:
    """Override ``app.openapi`` to add the Bearer security scheme, mark protected
    routes as requiring it, and attach the gateway error-envelope responses.

    Done as a post-process of FastAPI's generated schema so the route decorators
    stay the single source of truth for request/response shapes; this layer only
    adds cross-cutting security + error metadata.
    """
    from mm_gateway.schemas.api import ErrorEnvelope

    def custom_openapi() -> dict:
        if app.openapi_schema:
            return app.openapi_schema
        spec = FastAPI.openapi(app)

        # Register the error envelope component (referenced by _ERROR_RESPONSES).
        # Use the app's own schema generator so nested refs resolve against
        # #/components/schemas consistently with the rest of the spec, then hoist
        # any $defs it emits into the top-level components block.
        schemas = spec.setdefault("components", {}).setdefault("schemas", {})
        if "ErrorEnvelope" not in schemas:
            env_schema = ErrorEnvelope.model_json_schema(ref_template="#/components/schemas/{model}")
            for name, sub in env_schema.pop("$defs", {}).items():
                schemas.setdefault(name, sub)
            schemas["ErrorEnvelope"] = env_schema

        # Bearer auth scheme.
        spec.setdefault("components", {}).setdefault("securitySchemes", {})["BearerAuth"] = {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "API key",
            "description": 'Front-end API key sent as "Authorization: Bearer <token>".',
        }

        _ERR_REF = {"$ref": "#/components/schemas/ErrorEnvelope"}
        for path, item in spec.get("paths", {}).items():
            for method, op in item.items():
                if method not in ("get", "post", "put", "patch", "delete"):
                    continue
                if path not in _OPEN_PATHS:
                    op.setdefault("security", [{"BearerAuth": []}])
                    responses = op.setdefault("responses", {})
                    # Every protected op can return any of these gateway-error
                    # statuses. Ensure each carries BOTH the ErrorEnvelope
                    # schema and a concrete example -- route decorators may
                    # declare a bare {"description": ...} for some codes, so
                    # set the schema/example on the media block directly rather
                    # than setdefault-ing the whole response block.
                    for code in ("400", "401", "403", "404", "502", "504"):
                        resp = responses.setdefault(
                            code, {"description": _ERROR_RESPONSES[code]["description"]}
                        )
                        resp.setdefault("description", _ERROR_RESPONSES[code]["description"])
                        media = resp.setdefault("content", {}).setdefault(
                            "application/json", {}
                        )
                        media.setdefault("schema", _ERR_REF)
                        media.setdefault("examples", {}).setdefault(
                            "error", _ERROR_EXAMPLES[code]
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
                    ex = _SUCCESS_EXAMPLES.get(schema_name)
                    if ex and "examples" not in media and "example" not in media:
                        media.setdefault("example", ex)

                # FastAPI auto-attaches a 422 HTTPValidationError response.
                # Route decorators may declare a bare {"description": ...} for
                # 422 (which strips the auto-generated schema), so ensure the
                # HTTPValidationError schema ref is present, then attach a
                # concrete example body. 422 uses FastAPI's {"detail": [...]}
                # shape (distinct from the gateway's {"error": {...}} envelope)
                # because it is produced by FastAPI's request-body validation
                # before the route runs.
                err422 = op.get("responses", {}).get("422")
                if err422:
                    em = err422.setdefault("content", {}).setdefault("application/json", {})
                    em.setdefault(
                        "schema", {"$ref": "#/components/schemas/HTTPValidationError"}
                    )
                    if "examples" not in em and "example" not in em:
                        em.setdefault("example", {
                            "detail": [
                                {
                                    "type": "missing",
                                    "loc": ["body", "model"],
                                    "msg": "Field required",
                                    "input": {"input": "a prompt"},
                                }
                            ]
                        })

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
