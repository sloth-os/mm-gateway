"""Meta routes — health, model listing, metrics."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse

from mm_gateway.observability.metrics import render_prometheus

router = APIRouter()


@router.get("/health", tags=["meta"])
async def health() -> dict:
    return {"status": "ok"}


@router.get("/v1/models", tags=["meta"])
@router.get("/api/v1/models", tags=["meta"])
async def list_models(request: Request) -> dict:
    registry = request.app.state.registry
    return {"object": "list", "data": registry.list_models()}


@router.get("/metrics", tags=["meta"], response_class=PlainTextResponse)
async def metrics() -> str:
    return render_prometheus()
