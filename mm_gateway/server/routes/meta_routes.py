"""Meta routes — health, model listing, metrics."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Header, Query, Request
from fastapi.responses import PlainTextResponse

from mm_gateway.config import KeyConfig
from mm_gateway.observability.metrics import render_prometheus
from mm_gateway.observability.selection import STORE as SELECTION_STORE
from mm_gateway.schemas.api import HealthResponse, ModelLimitsListResponse, ModelListResponse
from mm_gateway.server.auth import get_api_key
from mm_gateway.server.routes._resources import (
    IfNoneMatchHeader,
    render_conditional_json,
)

router = APIRouter()


@router.get(
    "/health",
    tags=["meta"],
    operation_id="getHealth",
    response_model=HealthResponse,
    responses={200: {"description": "Gateway is healthy"}},
)
async def health() -> dict:
    return {"status": "ok"}


@router.get(
    "/v1/models",
    tags=["meta"],
    operation_id="listModels",
    response_model=ModelListResponse,
    responses={
        200: {
            "description": "Models available to the authenticated client.",
            "headers": {
                "ETag": {
                    "description": "Version identifier for conditional retrieval.",
                    "schema": {"type": "string"},
                }
            },
        },
        304: {"description": "The model catalogue has not changed."},
        401: {"description": "Missing or unknown API key"},
        403: {"description": "Key not allowed to use any generation service"},
    },
)
async def list_models(
    request: Request,
    key: Annotated[KeyConfig, Depends(get_api_key)],
    modality: Annotated[
        Literal["image", "video", "music"] | None,
        Query(description="Filter models by output modality."),
    ] = None,
    authorization: Annotated[
        str | None,
        Header(
            alias="Authorization",
            description='Bearer token: "Bearer <api-key>".',
            examples=["Bearer sk-gateway-demo"],
        ),
    ] = None,
    x_request_id: Annotated[
        str | None,
        Header(
            alias="X-Request-Id",
            description="Client-supplied request id (echoed back).",
            examples=["req-01HZX4J3K7NQ8X2V9Y6R5W4T3P"],
        ),
    ] = None,
    if_none_match: IfNoneMatchHeader = None,
) -> dict:
    registry = request.app.state.registry
    models = registry.list_public_models(key)
    if modality is not None:
        models = [model for model in models if model.get("modality") == modality]
    return render_conditional_json(
        {"object": "list", "data": models},
        request,
        cache_control="private, max-age=60",
    )


@router.get(
    "/v1/models/limits",
    tags=["meta"],
    operation_id="listModelLimits",
    response_model=ModelLimitsListResponse,
    responses={
        200: {
            "description": "Models available to the authenticated client with their input/output limits.",
            "headers": {
                "ETag": {
                    "description": "Version identifier for conditional retrieval.",
                    "schema": {"type": "string"},
                }
            },
        },
        304: {"description": "The model catalogue has not changed."},
        401: {"description": "Missing or unknown API key"},
        403: {"description": "Key not allowed to use any generation service"},
    },
)
async def list_model_limits(
    request: Request,
    key: Annotated[KeyConfig, Depends(get_api_key)],
    modality: Annotated[
        Literal["image", "video", "music"] | None,
        Query(description="Filter models by output modality."),
    ] = None,
    authorization: Annotated[
        str | None,
        Header(
            alias="Authorization",
            description='Bearer token: "Bearer <api-key>".',
            examples=["Bearer sk-gateway-demo"],
        ),
    ] = None,
    x_request_id: Annotated[
        str | None,
        Header(
            alias="X-Request-Id",
            description="Client-supplied request id (echoed back).",
            examples=["req-01HZX4J3K7NQ8X2V9Y6R5W4T3P"],
        ),
    ] = None,
    if_none_match: IfNoneMatchHeader = None,
) -> dict:
    """List usable models with their documented input/output limits.

    Use this to pick a model and craft a prompt that fits: each entry's
    ``limits`` carries the input modalities accepted, the max prompt length,
    the max output count, supported sizes/durations, and per-role support
    flags (image-to-image, first/last frame, reference audio, lyrics, ...).
    The same catalogue drives auto-routing when a request omits ``model``.
    """
    registry = request.app.state.registry
    models = registry.list_model_limits(key)
    if modality is not None:
        models = [model for model in models if model.get("modality") == modality]
    return render_conditional_json(
        {"object": "list", "data": models},
        request,
        cache_control="private, max-age=60",
    )


@router.get(
    "/metrics",
    tags=["meta"],
    operation_id="getMetrics",
    response_class=PlainTextResponse,
    responses={200: {"description": "Prometheus exposition"}},
)
async def metrics() -> str:
    # Request counters/histograms first, then the per-backend selection health
    # (success rate, latency EWMA, rate-limit cooldown, attempts) that drives
    # auto-routing — the two share the same Prometheus exposition.
    parts = [render_prometheus(), SELECTION_STORE.render_prometheus()]
    return "\n".join(p for p in parts if p)
