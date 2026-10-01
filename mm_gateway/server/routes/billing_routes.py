"""Cost estimates and usage (docs/design/auto-mode.md#prices-and-estimates).

``POST /v1/{images,videos,music}/estimate`` takes the same body as the create
call and runs auto mode's planning stages without creating a task: which model
a create would try first, what it would cost, and why other candidates are not
admissible. ``GET /v1/usage`` reports the authenticated key's spend,
reservations and budgets from the ledger.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import Response

from mm_gateway.auto_mode import estimate_route, resolve_policy
from mm_gateway.config import KeyConfig
from mm_gateway.schemas.api import (
    EstimateResponse,
    ImageRequest,
    MusicRequest,
    UsageResponse,
    VideoRequest,
)
from mm_gateway.server.auth import get_api_key
from mm_gateway.server.routes._resources import render_conditional_json
from mm_gateway.translators.rest import from_image_request, from_music_request, from_video_request

router = APIRouter()


def _estimate(request: Request, unified: Any, routing: Any, key: KeyConfig, modality: str) -> Response:
    state = request.app.state
    body = estimate_route(
        state.registry, state.ledger, unified, key=key, modality=modality,
        policy=resolve_policy(state.settings, routing),
    )
    payload = jsonable_encoder(EstimateResponse.model_validate(body), exclude_none=True)
    # Estimates depend on live health and spend: never reuse one.
    return render_conditional_json(payload, request, cache_control="no-store", conditional=False)


_ESTIMATE = {
    "response_model": EstimateResponse,
    "response_model_exclude_none": True,
    "responses": {200: {"description": "How auto mode would route the request, and its estimated cost."}},
}


@router.post("/v1/images/estimate", name="estimate_image", operation_id="estimateImage",
             summary="Estimate an image request", tags=["images"], **_ESTIMATE)
async def estimate_image(
    request: Request,
    body: Annotated[ImageRequest, Body()],
    key: Annotated[KeyConfig, Depends(get_api_key)],
) -> Response:
    return _estimate(request, from_image_request(body), body.routing, key, "image")


@router.post("/v1/videos/estimate", name="estimate_video", operation_id="estimateVideo",
             summary="Estimate a video request", tags=["videos"], **_ESTIMATE)
async def estimate_video(
    request: Request,
    body: Annotated[VideoRequest, Body()],
    key: Annotated[KeyConfig, Depends(get_api_key)],
) -> Response:
    return _estimate(request, from_video_request(body), body.routing, key, "video")


@router.post("/v1/music/estimate", name="estimate_music", operation_id="estimateMusic",
             summary="Estimate a music request", tags=["music"], **_ESTIMATE)
async def estimate_music(
    request: Request,
    body: Annotated[MusicRequest, Body()],
    key: Annotated[KeyConfig, Depends(get_api_key)],
) -> Response:
    return _estimate(request, from_music_request(body), body.routing, key, "music")


@router.get(
    "/v1/usage",
    name="get_usage",
    operation_id="getUsage",
    summary="Spend and budgets of the authenticated key",
    tags=["usage"],
    response_model=UsageResponse,
    response_model_exclude_none=True,
    responses={200: {"description": "Spend, reservations and budgets for the current period."}},
)
async def get_usage(
    request: Request,
    key: Annotated[KeyConfig, Depends(get_api_key)],
    scope: Annotated[
        str | None,
        Query(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:/-]+$",
              description="Report only this budget scope."),
    ] = None,
) -> Response:
    usage = request.app.state.ledger.usage(key, scope)
    payload = jsonable_encoder(UsageResponse.model_validate(usage), exclude_none=True)
    return render_conditional_json(payload, request, cache_control="private, no-cache")


__all__ = ["router"]
