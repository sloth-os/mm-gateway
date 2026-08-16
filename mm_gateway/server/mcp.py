"""HTTP MCP server — exposes the gateway as a Streamable-HTTP MCP endpoint.

When ``Settings.mcp_enabled`` is true, ``mount_mcp(app, settings)`` mounts a
``StreamableHTTPSessionManager``-backed route at ``Settings.mcp_path`` (default
``/mcp``). The route is a streaming ASGI passthrough — it hands the request's
``scope``/``receive``/``send`` straight to the session manager rather than
buffering its body into a single ``Response``. The session manager's lifespan
is tied to the app lifespan so the MCP server starts/stops with the gateway.

The MCP server registers eight tools that mirror the provider-neutral HTTP
resources: model listing (with and without limits) plus create/get pairs for
images, videos, and music.
Create tools accept the same typed ``input`` and ``parameters`` values as REST,
always return immediately, and expose gateway-owned task ids.

Every tool authenticates the caller through the same bearer-token resolution as
the HTTP routes: the ``Authorization`` header carried on the MCP ``Context`` is
resolved via ``auth.resolve_key`` against the gateway's ``Settings`` (with the
open-key fallback), so a single set of keys governs both surfaces. A tool whose
underlying call raises a ``GatewayError`` returns an MCP error result carrying
that error's code/message rather than an opaque 500.
"""

from __future__ import annotations

import functools
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

from fastapi import FastAPI, Request
from fastapi.responses import Response
from pydantic import Field

from mm_gateway.config import Settings
from mm_gateway.core.exceptions import GatewayError, TaskNotFoundError
from mm_gateway.observability.logging import get_logger
from mm_gateway.schemas.api import (
    ImageInputList,
    ImageParameters,
    ImageRequest,
    ImageTaskResponse,
    MusicInputList,
    MusicParameters,
    MusicRequest,
    MusicTaskResponse,
    RoutingDirective,
    VideoInputList,
    VideoParameters,
    VideoRequest,
    VideoTaskResponse,
)
from mm_gateway.server.auth import authorize_task_access, resolve_key
from mm_gateway.server.routes._resources import (
    find_idempotent_record,
    new_record,
    remember_create_response,
    replay_resource,
    request_fingerprint,
    stamped_model,
)
from mm_gateway.translators.rest import (
    from_image_request,
    from_music_request,
    from_video_request,
    to_image_response,
    to_music_response,
    to_video_response,
)

log = get_logger("mcp")

# The session manager / route are only constructible when the ``mcp`` package
# is importable; the import is deferred so the gateway still starts without it
# when MCP is disabled.
_mcp_import_error: Exception | None = None
try:  # pragma: no cover - import guard exercised only when mcp is absent
    from mcp.server import MCPServer
    from mcp.server.mcpserver import Context
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from mcp.shared.exceptions import MCPError
except Exception as exc:  # noqa: BLE001
    _mcp_import_error = exc

# JSON-RPC reserves -32000..-32099 for server-defined errors; gateway errors
# land in that band so clients can distinguish them from MCP protocol errors.
_GATEWAY_ERROR_CODE = -32000
IdempotencyKey = Annotated[
    str | None,
    Field(
        min_length=1,
        max_length=255,
        description="Client-generated key used to safely retry this create call.",
    ),
]


def _gateway_to_mcp_error(exc: GatewayError) -> MCPError:
    """Translate a GatewayError into a structured MCP (JSON-RPC) error."""
    payload = exc.to_public_dict()["error"]
    payload["status_code"] = exc.status_code
    return MCPError(_GATEWAY_ERROR_CODE, exc.message, data=payload)


class _StreamPassthrough(Response):
    """A Starlette ``Response`` that is a pure ASGI passthrough to the MCP
    session manager.

    Streaming matters for the long-lived GET SSE channel: the session manager
    emits ``http.response.start`` followed by a series of ``http.response.body``
    messages with ``more_body=True`` that must reach the client as they are
    produced (server-initiated JSON-RPC notifications/requests ride this
    stream). A normal ``Response`` buffers the whole body and re-emits it once
    the handler returns — for the SSE stream that is only when the client
    disconnects, so the channel never delivers anything live and the client
    reconnects in a tight loop. By handing ``send`` straight to
    ``handle_request``, every chunk is forwarded immediately, exactly as if the
    session manager were mounted as a raw ASGI app.
    """

    def __init__(self, sm: Any) -> None:
        # No body to serialise — ``__call__`` short-circuits ``Response``'s own
        # send path entirely and delegates straight to the session manager.
        super().__init__(content=b"")
        self._sm = sm

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        await self._sm.handle_request(scope, receive, send)


def _tool(fn):
    """Decorator: surface GatewayError as a structured MCP error, pass others through.

    Without this the SDK wraps every exception in a bare ``is_error`` text result
    (``"Error executing tool <name>: <str(exc)>"``), losing the gateway's stable
    ``code``/``status_code``. Translating to ``MCPError`` keeps the envelope so an
    MCP client can branch on it exactly like an HTTP client branches on status.
    """

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await fn(*args, **kwargs)
        except GatewayError as exc:
            raise _gateway_to_mcp_error(exc) from exc

    return wrapper


def _bearer_from_ctx(ctx: Context) -> str | None:
    """Pull the ``Bearer <token>`` value off an MCP tool context, or None."""
    headers = ctx.headers if ctx.headers is not None else {}
    header = headers.get("authorization") or headers.get("Authorization")
    if not header:
        return None
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token.strip()


def _build_mcp_server(app: FastAPI) -> MCPServer:
    """Construct the MCPServer with gateway tools, closing over ``app.state``."""
    if _mcp_import_error is not None:
        raise _mcp_import_error

    settings: Settings = app.state.settings
    registry = app.state.registry
    image_service = app.state.image_service
    video_service = app.state.video_service
    music_service = app.state.music_service
    task_store = app.state.task_store
    mcp = MCPServer(name="mm-gateway", version="0.1.0")

    def _key(ctx: Context):
        return resolve_key(settings, _bearer_from_ctx(ctx))

    @mcp.tool()
    @_tool
    async def list_models(
        ctx: Context,
        modality: Literal["image", "video", "music"] | None = None,
    ) -> str:
        """List usable models, optionally filtered by image, video, or music."""
        import json

        key = _key(ctx)
        models = registry.list_public_models(key)
        if modality is not None:
            models = [model for model in models if model["modality"] == modality]
        return json.dumps({"object": "list", "data": models})

    @mcp.tool()
    @_tool
    async def list_model_limits(
        ctx: Context,
        modality: Literal["image", "video", "music"] | None = None,
    ) -> str:
        """List usable models with their documented input/output limits.

        Each entry's ``limits`` carries the neutral limits the auto-router
        uses and that a client can consult when crafting a prompt for a
        specific model (input modalities, max prompt length, max output
        count, supported sizes/durations, per-role support flags).
        """
        import json

        key = _key(ctx)
        models = registry.list_model_limits(key)
        if modality is not None:
            models = [model for model in models if model["modality"] == modality]
        return json.dumps({"object": "list", "data": models})

    @mcp.tool()
    @_tool
    async def create_image(
        ctx: Context,
        input: ImageInputList,
        parameters: ImageParameters | None = None,
        model: str | None = None,
        routing: RoutingDirective | None = None,
        metadata: dict[str, Any] | None = None,
        idempotency_key: IdempotencyKey = None,
    ) -> str:
        """Create an asynchronous image task from text and image inputs.

        Omit ``model`` (or set ``auto``) to let the gateway auto-route to a
        backend whose limits fit the request's input.
        """

        key = _key(ctx)
        body = ImageRequest(
            model=model,
            input=input,
            parameters=parameters or ImageParameters(),
            routing=routing,
            metadata=metadata or {},
        )
        fingerprint = request_fingerprint(body)
        async with task_store.idempotency_guard(key.id, "image", idempotency_key):
            record = await find_idempotent_record(
                task_store,
                owner_key_id=key.id,
                modality="image",
                idempotency_key=idempotency_key,
                fingerprint=fingerprint,
            )
            if record is not None:
                resource = replay_resource(
                    record,
                    ImageTaskResponse,
                    resource_url=f"/v1/images/{record.task_id}",
                )
                return resource.model_dump_json(by_alias=True, exclude_none=True)

            task = await image_service.create(
                from_image_request(body),
                key=key,
                tag=routing.profile if routing else None,
                wait=False,
            )
            record = new_record(
                "img",
                task,
                model=stamped_model(model, task.model),
                modality="image",
                metadata=body.metadata,
                owner_key_id=key.id,
                idempotency_key=idempotency_key,
                request_fingerprint=fingerprint if idempotency_key else None,
            )
            resource = to_image_response(
                task, record, self_url=f"/v1/images/{record.task_id}"
            )
            remember_create_response(record, resource)
            await task_store.put(record)
            return resource.model_dump_json(by_alias=True, exclude_none=True)

    @mcp.tool()
    @_tool
    async def get_image(ctx: Context, id: str) -> str:
        """Retrieve the latest state of an image task."""

        key = _key(ctx)
        record = await task_store.get(id)
        if record is None or record.modality != "image":
            raise TaskNotFoundError(f"image task {id} not found")
        authorize_task_access(registry, key, record)
        task = await image_service.get(
            record.provider_task_id or record.task_id,
            backend_name=record.provider,
        )
        resource = to_image_response(task, record, self_url=f"/v1/images/{id}")
        return resource.model_dump_json(by_alias=True, exclude_none=True)

    @mcp.tool()
    @_tool
    async def create_video(
        ctx: Context,
        input: VideoInputList,
        parameters: VideoParameters | None = None,
        model: str | None = None,
        routing: RoutingDirective | None = None,
        metadata: dict[str, Any] | None = None,
        idempotency_key: IdempotencyKey = None,
    ) -> str:
        """Create an asynchronous video task from multimodal inputs.

        Omit ``model`` (or set ``auto``) to let the gateway auto-route.
        """

        key = _key(ctx)
        body = VideoRequest(
            model=model,
            input=input,
            parameters=parameters or VideoParameters(),
            routing=routing,
            metadata=metadata or {},
        )
        fingerprint = request_fingerprint(body)
        async with task_store.idempotency_guard(key.id, "video", idempotency_key):
            record = await find_idempotent_record(
                task_store,
                owner_key_id=key.id,
                modality="video",
                idempotency_key=idempotency_key,
                fingerprint=fingerprint,
            )
            if record is not None:
                resource = replay_resource(
                    record,
                    VideoTaskResponse,
                    resource_url=f"/v1/videos/{record.task_id}",
                )
                return resource.model_dump_json(by_alias=True, exclude_none=True)

            task = await video_service.create(
                from_video_request(body),
                key=key,
                tag=routing.profile if routing else None,
                wait=False,
            )
            record = new_record(
                "vid",
                task,
                model=stamped_model(model, task.model),
                modality="video",
                metadata=body.metadata,
                owner_key_id=key.id,
                idempotency_key=idempotency_key,
                request_fingerprint=fingerprint if idempotency_key else None,
            )
            resource = to_video_response(
                task, record, self_url=f"/v1/videos/{record.task_id}"
            )
            remember_create_response(record, resource)
            await task_store.put(record)
            return resource.model_dump_json(by_alias=True, exclude_none=True)

    @mcp.tool()
    @_tool
    async def get_video(ctx: Context, id: str) -> str:
        """Retrieve the latest state of a video task."""

        key = _key(ctx)
        record = await task_store.get(id)
        if record is None or record.modality != "video":
            raise TaskNotFoundError(f"video task {id} not found")
        authorize_task_access(registry, key, record)
        task = await video_service.get(
            record.provider_task_id or record.task_id,
            backend_name=record.provider,
        )
        resource = to_video_response(task, record, self_url=f"/v1/videos/{id}")
        return resource.model_dump_json(by_alias=True, exclude_none=True)

    @mcp.tool()
    @_tool
    async def create_music(
        ctx: Context,
        input: MusicInputList,
        parameters: MusicParameters | None = None,
        model: str | None = None,
        routing: RoutingDirective | None = None,
        metadata: dict[str, Any] | None = None,
        idempotency_key: IdempotencyKey = None,
    ) -> str:
        """Create an asynchronous music task from multimodal inputs."""

        key = _key(ctx)
        body = MusicRequest(
            model=model,
            input=input,
            parameters=parameters or MusicParameters(),
            routing=routing,
            metadata=metadata or {},
        )
        fingerprint = request_fingerprint(body)
        async with task_store.idempotency_guard(key.id, "music", idempotency_key):
            record = await find_idempotent_record(
                task_store,
                owner_key_id=key.id,
                modality="music",
                idempotency_key=idempotency_key,
                fingerprint=fingerprint,
            )
            if record is not None:
                resource = replay_resource(
                    record,
                    MusicTaskResponse,
                    resource_url=f"/v1/music/{record.task_id}",
                )
                return resource.model_dump_json(by_alias=True, exclude_none=True)

            task = await music_service.create(
                from_music_request(body),
                key=key,
                tag=routing.profile if routing else None,
                wait=False,
            )
            record = new_record(
                "mus",
                task,
                model=stamped_model(model, task.model),
                modality="music",
                metadata=body.metadata,
                owner_key_id=key.id,
                idempotency_key=idempotency_key,
                request_fingerprint=fingerprint if idempotency_key else None,
            )
            resource = to_music_response(
                task, record, self_url=f"/v1/music/{record.task_id}"
            )
            remember_create_response(record, resource)
            await task_store.put(record)
            return resource.model_dump_json(by_alias=True, exclude_none=True)

    @mcp.tool()
    @_tool
    async def get_music(ctx: Context, id: str) -> str:
        """Retrieve the latest state of a music task."""

        key = _key(ctx)
        record = await task_store.get(id)
        if record is None or record.modality != "music":
            raise TaskNotFoundError(f"music task {id} not found")
        authorize_task_access(registry, key, record)
        task = await music_service.get(
            record.provider_task_id or record.task_id,
            backend_name=record.provider,
        )
        resource = to_music_response(task, record, self_url=f"/v1/music/{id}")
        return resource.model_dump_json(by_alias=True, exclude_none=True)

    return mcp


def mount_mcp(app: FastAPI, settings: Settings) -> None:
    """Mount the HTTP MCP endpoint onto ``app`` if MCP is enabled.

    No-op when ``settings.mcp_enabled`` is false (the default), so existing
    deployments are unaffected. When enabled but the ``mcp`` package is not
    importable, a warning is logged and nothing is mounted.
    """
    if not settings.mcp_enabled:
        return
    if _mcp_import_error is not None:
        log.warning("mcp_disabled_no_package", error=str(_mcp_import_error))
        return

    mcp = _build_mcp_server(app)
    session_manager = StreamableHTTPSessionManager(
        app=mcp._lowlevel_server, json_response=False, stateless=False,
        session_idle_timeout=settings.mcp_session_idle_timeout,
    )
    # Exposed for runtime introspection (logs/metrics) and tests; the endpoint
    # closes over the same instance.
    app.state.mcp_session_manager = session_manager

    @asynccontextmanager
    async def _mcp_lifespan(_app: FastAPI):
        async with session_manager.run():
            log.info("mcp_server_started", path=settings.mcp_path)
            yield
            log.info("mcp_server_stopped")

    # Compose the MCP lifespan with the app's existing lifespan so both run.
    _previous_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def _combined_lifespan(_app: FastAPI):
        async with _mcp_lifespan(_app), _previous_lifespan(_app):
            yield

    app.router.lifespan_context = _combined_lifespan

    @app.api_route(settings.mcp_path, methods=["GET", "POST", "DELETE"],
                   include_in_schema=False)
    async def mcp_endpoint(request: Request) -> Response:
        return _StreamPassthrough(session_manager)
