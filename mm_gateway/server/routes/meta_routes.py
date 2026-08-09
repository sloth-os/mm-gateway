"""Meta routes — health, model listing, metrics."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import PlainTextResponse

from mm_gateway.config import KeyConfig
from mm_gateway.observability.metrics import render_prometheus
from mm_gateway.schemas.api import HealthResponse, ModelListResponse
from mm_gateway.server.auth import get_api_key

router = APIRouter()


@router.get(
    "/health",
    tags=["meta"],
    response_model=HealthResponse,
    responses={200: {"description": "Gateway is healthy"}},
)
async def health() -> dict:
    return {"status": "ok"}


@router.get(
    "/v1/models",
    tags=["meta"],
    response_model=ModelListResponse,
    responses={
        401: {"description": "Missing or unknown API key"},
        403: {"description": "Key not allowed to use any backend"},
    },
)
@router.get(
    "/api/v1/models",
    tags=["meta"],
    response_model=ModelListResponse,
    include_in_schema=False,
)
async def list_models(
    request: Request,
    key: KeyConfig = Depends(get_api_key),
    authorization: str | None = Header(
        None, alias="Authorization",
        description='Bearer token: "Bearer <api-key>".',
        examples={"default": {"value": "Bearer sk-gateway-demo"}},
    ),
    x_request_id: str | None = Header(
        None, alias="X-Request-Id", description="Client-supplied request id (echoed back).",
        examples={"default": {"value": "req-01HZX4J3K7NQ8X2V9Y6R5W4T3P"}},
    ),
    x_backend_tag: str | None = Header(
        None, alias="X-Backend-Tag", description="Pin to a backend by tag label.",
        examples={"default": {"value": "prod"}},
    ),
    x_backend: str | None = Header(
        None, alias="X-Backend", description="Pin to a backend by name.",
        examples={"default": {"value": "openai"}},
    ),
) -> dict:
    registry = request.app.state.registry
    return {"object": "list", "data": registry.list_models(key)}


@router.get(
    "/metrics",
    tags=["meta"],
    response_class=PlainTextResponse,
    responses={200: {"description": "Prometheus exposition"}},
)
async def metrics() -> str:
    return render_prometheus()
