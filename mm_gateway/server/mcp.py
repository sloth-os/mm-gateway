"""HTTP MCP server — exposes the gateway as a Streamable-HTTP MCP endpoint.

When ``Settings.mcp_enabled`` is true, ``mount_mcp(app, settings)`` mounts a
``StreamableHTTPSessionManager``-backed route at ``Settings.mcp_path`` (default
``/mcp``). The route is a streaming ASGI passthrough — it hands the request's
``scope``/``receive``/``send`` straight to the session manager rather than
buffering its body into a single ``Response``. The session manager's lifespan
is tied to the app lifespan so the MCP server starts/stops with the gateway.

The MCP server registers seven tools that mirror the HTTP API:

* ``list_models``        — list usable models for the calling key (image+video+music).
* ``create_image``       — submit an image task (Gemini shape: ``model`` + ``input``
                           string/parts), returning a task id; honours ``wait``
                           for sync-style.
* ``get_image``          — poll an image task by id; returns the Gemini
                           steps/content envelope (image base64/URL blocks).
* ``create_video``       — submit a video task (Seedance content-array shape),
                           returning a task id; honours ``wait`` for sync-style.
* ``get_video``          — poll a video task by id.
* ``create_music``       — submit a music task (Gemini Lyria 3 shape: ``model``
                           + ``input`` string/parts), returning an interaction id;
                           honours ``wait`` for sync-style.
* ``get_music``          — poll a music task by id; returns the Lyria steps/content
                           envelope (audio base64 + lyrics blocks).

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
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import Response

from mm_gateway.config import Settings
from mm_gateway.core.exceptions import GatewayError
from mm_gateway.observability.logging import get_logger
from mm_gateway.server.auth import resolve_key
from mm_gateway.translators.image import gemini_compat
from mm_gateway.translators.music import lyria_compat
from mm_gateway.translators.video import seedance_compat

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


def _gateway_to_mcp_error(exc: GatewayError) -> "MCPError":
    """Translate a GatewayError into a structured MCP (JSON-RPC) error."""
    payload = exc.to_dict()["error"]
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


def _build_mcp_server(app: FastAPI) -> "MCPServer":
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
    async def list_models(ctx: Context) -> str:
        """List the image and video models usable by the calling API key."""
        import json
        key = _key(ctx)
        models = registry.list_models(key)
        return json.dumps({"object": "list", "data": models})

    @mcp.tool()
    @_tool
    async def create_image(
        ctx: Context,
        model: str,
        input: str | list[dict[str, Any]],
        wait: bool = True,
        tag: str | None = None,
        backend: str | None = None,
    ) -> str:
        """Submit an image generation task (Gemini shape).

        ``input`` is either a string prompt or a parts array
        (``[{"type":"text","text":...}, {"type":"image","url":...}, ...]``).
        Returns ``{"id": "<task_id>"}``; when ``wait`` is true the call blocks
        until the task reaches a terminal state (up to the sync wait limit).
        """
        import json
        from mm_gateway.tasks.store import TaskRecord
        key = _key(ctx)
        unified = gemini_compat.from_gemini({"model": model, "input": input})
        task = await image_service.create(
            unified, key=key, tag=tag, backend_name=backend, wait=wait,
        )
        await task_store.put(TaskRecord(
            task_id=task.task_id, provider=task.provider, model=task.model,
            modality="image",
        ))
        return json.dumps(gemini_compat.to_gemini_create(task))

    @mcp.tool()
    @_tool
    async def get_image(ctx: Context, id: str) -> str:
        """Poll an image task by id; returns the Gemini steps/content envelope."""
        import json
        key = _key(ctx)
        record = await task_store.get(id)
        # Same cross-tenant guard as get_video/get_music: the calling key must be
        # authorised for the backend that owns the task.
        owner = record.provider if record else None
        if owner and owner not in registry.usable_backends(key):
            from mm_gateway.core.exceptions import ForbiddenError
            raise ForbiddenError(
                f"API key '{key.id}' is not allowed to use backend '{owner}'."
            )
        task = await image_service.get(id, backend_name=owner)
        return json.dumps(gemini_compat.to_gemini_task(task))

    @mcp.tool()
    @_tool
    async def create_video(
        ctx: Context,
        model: str,
        content: list[dict[str, Any]],
        wait: bool = True,
        tag: str | None = None,
        backend: str | None = None,
    ) -> str:
        """Submit a video generation task (Seedance content-array shape).

        Returns ``{"id": "<task_id>"}``; when ``wait`` is true the call blocks
        until the task reaches a terminal state (up to the sync wait limit).
        """
        import json
        from mm_gateway.tasks.store import TaskRecord
        key = _key(ctx)
        unified = seedance_compat.from_seedance({"model": model, "content": content})
        task = await video_service.create(
            unified, key=key, tag=tag, backend_name=backend, wait=wait,
        )
        await task_store.put(TaskRecord(
            task_id=task.task_id, provider=task.provider, model=task.model,
        ))
        return json.dumps(seedance_compat.to_seedance_create(task))

    @mcp.tool()
    @_tool
    async def get_video(ctx: Context, id: str) -> str:
        """Poll a video task by id; returns the Seedance-shape task envelope."""
        import json
        key = _key(ctx)
        record = await task_store.get(id)
        # Polling bypasses the create-path usable-backends check, so enforce it
        # here: the calling key must be authorised for the backend that owns the
        # task, else an authenticated key could read another tenant's task/artefacts.
        owner = record.provider if record else None
        if owner and owner not in registry.usable_backends(key):
            from mm_gateway.core.exceptions import ForbiddenError
            raise ForbiddenError(
                f"API key '{key.id}' is not allowed to use backend '{owner}'."
            )
        task = await video_service.get(id, backend_name=owner)
        return json.dumps(seedance_compat.to_seedance_task(task))

    @mcp.tool()
    @_tool
    async def create_music(
        ctx: Context,
        model: str,
        input: str | list[dict[str, Any]],
        wait: bool = True,
        tag: str | None = None,
        backend: str | None = None,
    ) -> str:
        """Submit a music generation task (Gemini Lyria 3 shape).

        ``input`` is either a string prompt or a Lyria parts array
        (``[{"type":"text","text":...}, ...]``). Returns ``{"id": "<task_id>"}``;
        when ``wait`` is true the call blocks until the task reaches a terminal
        state (up to the sync wait limit).
        """
        import json
        from mm_gateway.tasks.store import TaskRecord
        key = _key(ctx)
        unified = lyria_compat.from_lyria({"model": model, "input": input})
        task = await music_service.create(
            unified, key=key, tag=tag, backend_name=backend, wait=wait,
        )
        await task_store.put(TaskRecord(
            task_id=task.task_id, provider=task.provider, model=task.model,
            modality="music",
        ))
        return json.dumps(lyria_compat.to_lyria_create(task))

    @mcp.tool()
    @_tool
    async def get_music(ctx: Context, id: str) -> str:
        """Poll a music task by id; returns the Lyria steps/content envelope."""
        import json
        key = _key(ctx)
        record = await task_store.get(id)
        # Same cross-tenant guard as get_video: the calling key must be authorised
        # for the backend that owns the task.
        owner = record.provider if record else None
        if owner and owner not in registry.usable_backends(key):
            from mm_gateway.core.exceptions import ForbiddenError
            raise ForbiddenError(
                f"API key '{key.id}' is not allowed to use backend '{owner}'."
            )
        task = await music_service.get(id, backend_name=owner)
        return json.dumps(lyria_compat.to_lyria_task(task))

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
        async with _mcp_lifespan(_app):
            async with _previous_lifespan(_app):
                yield

    app.router.lifespan_context = _combined_lifespan

    @app.api_route(settings.mcp_path, methods=["GET", "POST", "DELETE"],
                   include_in_schema=False)
    async def mcp_endpoint(request: Request) -> Response:
        return _StreamPassthrough(session_manager)
