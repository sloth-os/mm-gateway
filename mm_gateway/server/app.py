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
from mm_gateway.observability.logging import bind_context, clear_context, configure_logging, get_logger, new_request_id
from mm_gateway.observability.metrics import render_prometheus
from mm_gateway.registry import Registry
from mm_gateway.services import ImageService, VideoService
from mm_gateway.tasks.store import TaskStore

log = get_logger("app")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    configure_logging(level=settings.log_level, fmt=settings.log_format)

    registry = Registry(settings)
    image_service = ImageService(registry)
    video_service = VideoService(
        registry, max_sync_wait=settings.max_sync_wait,
        poll_interval=settings.poll_interval, sync_default=settings.video_sync_default,
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
    app.state.task_store = task_store

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or new_request_id()
        bind_context(request_id=request_id)
        try:
            response = await call_next(request)
        finally:
            clear_context()
        response.headers["x-request-id"] = request_id
        return response

    @app.exception_handler(GatewayError)
    async def gateway_error_handler(_: Request, exc: GatewayError):
        log.warning("request_error", code=exc.code, provider=exc.provider, message=exc.message)
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

    # Register routes.
    from mm_gateway.server.routes import image_routes, video_routes, meta_routes
    app.include_router(meta_routes.router, tags=["meta"])
    app.include_router(image_routes.router)
    app.include_router(video_routes.router)

    return app


app = create_app()


def run() -> None:
    settings = Settings.from_env()
    uvicorn.run("mm_gateway.server.app:app", host=settings.host, port=settings.port,
                log_level=settings.log_level.lower())


if __name__ == "__main__":
    run()
