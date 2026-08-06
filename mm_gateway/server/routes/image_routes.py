"""Image routes — OpenAI-compatible and OpenRouter-compatible.

- ``POST /v1/images/generations``  (OpenAI shape)
- ``POST /api/v1/images``          (OpenRouter unified shape)

Both translate to the unified schema, dispatch through ``ImageService``, and
translate the unified response back to the requested shape. The OpenRouter
shape is detected by path; a client can also force the response shape with the
``X-Response-Format: openai|openrouter`` header. Both routes require a valid
front-end API key and honour ``X-Backend-Tag`` / ``provider.tag`` /
``provider.backend`` routing overrides.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import JSONResponse

from mm_gateway.config import KeyConfig
from mm_gateway.observability.logging import get_logger
from mm_gateway.server.auth import get_api_key, routing_overrides
from mm_gateway.translators.image import openai_compat, openrouter_compat

log = get_logger("route.image")
router = APIRouter()


@router.post("/v1/images/generations", tags=["image"])
async def openai_images(
    request: Request,
    key: KeyConfig = Depends(get_api_key),
    x_response_format: str | None = Header(default=None, alias="X-Response-Format"),
) -> Any:
    body = await request.json()
    unified = openai_compat.from_openai(body)
    tag, backend_name = routing_overrides(request, body)
    service = request.app.state.image_service
    resp = await service.generate(unified, key=key, tag=tag, backend_name=backend_name)
    out = openai_compat.to_openai(resp) if x_response_format != "openrouter" else openrouter_compat.to_openrouter(resp)
    return JSONResponse(content=out)


@router.post("/api/v1/images", tags=["image"])
async def openrouter_images(
    request: Request,
    key: KeyConfig = Depends(get_api_key),
    x_response_format: str | None = Header(default=None, alias="X-Response-Format"),
) -> Any:
    body = await request.json()
    unified = openrouter_compat.from_openrouter(body)
    tag, backend_name = routing_overrides(request, body)
    service = request.app.state.image_service
    resp = await service.generate(unified, key=key, tag=tag, backend_name=backend_name)
    out = openrouter_compat.to_openrouter(resp) if x_response_format != "openai" else openai_compat.to_openai(resp)
    return JSONResponse(content=out)
