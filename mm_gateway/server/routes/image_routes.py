"""Image routes — Gemini-compatible.

Gemini image shape (``POST /v1/images``):

- ``POST /v1/images``        -> create task, returns ``{"id": ...}``
- ``POST /v1/images/async``  -> same, but always async (``wait=False``); the URL
                                 itself encodes the intent, so ``?wait`` /
                                 ``Prefer`` are ignored on this path.
- ``GET  /v1/images/{id}``   -> poll task; response carries ``steps[].content[]``
                                 image blocks (inline base64 or URL) plus
                                 ``output_image`` / ``output_image_url`` helpers.

On the base ``/v1/images`` path, sync vs async is controlled by the
``Prefer: respond-async`` header or the ``?wait=true`` query param. Without
either, the ``image_sync_default`` setting governs. The ``/async`` sibling is
the explicit async URL: clients that always want a task id back (and will poll
themselves) hit it directly and never block. The backend that owns a task id is
recorded in the task store at creation time so subsequent polls route correctly
even though task ids are opaque to the gateway. All routes require a valid
front-end API key.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends, Header, Path, Query, Request
from fastapi.responses import JSONResponse

from mm_gateway.config import KeyConfig
from mm_gateway.observability.logging import get_logger
from mm_gateway.schemas.api import CreateResponse, ImageRequest, ImageTaskResponse
from mm_gateway.schemas.image import UnifiedImageTask
from mm_gateway.server.auth import authorize_task, get_api_key, routing_overrides
from mm_gateway.tasks.store import TaskRecord
from mm_gateway.translators.image import gemini_compat

log = get_logger("route.image")
router = APIRouter()


def _is_async(request: Request, wait: bool | None) -> bool:
    if wait is not None:
        return not wait
    prefer = request.headers.get("prefer", "")
    return "respond-async" in prefer.lower()


async def _remember(store, task: UnifiedImageTask) -> None:
    await store.put(TaskRecord(
        task_id=task.task_id, provider=task.provider, model=task.model,
        modality="image",
    ))


async def _create(request: Request, key: KeyConfig, *, wait: bool) -> Any:
    body = await request.json()
    unified = gemini_compat.from_gemini(body)
    tag, backend_name = routing_overrides(request, body)
    service = request.app.state.image_service
    task = await service.create(
        unified, key=key, tag=tag, backend_name=backend_name, wait=wait,
    )
    await _remember(request.app.state.task_store, task)
    out = gemini_compat.to_gemini_create(task)
    return JSONResponse(content=out)


@router.post(
    "/v1/images",
    tags=["image"],
    response_model=CreateResponse,
    response_model_exclude_none=False,
    responses={422: {"description": "Validation Error"}},
)
async def image_create(
    request: Request,
    key: KeyConfig = Depends(get_api_key),
    wait: bool | None = Query(
        default=None,
        description="true blocks until completion; false returns a task id to poll.",
        examples={"default": {"value": True}},
    ),
    body: ImageRequest = Body(
        ...,
        openapi_examples={
            "text_prompt": {
                "summary": "Text-to-image (string prompt)",
                "value": {
                    "model": "gateway-image-pro",
                    "input": "a cyberpunk cat in the rain, neon",
                    "config": {"n": 1, "size": "1024x1024", "quality": "high"},
                },
            },
            "parts": {
                "summary": "Image-to-image (parts array)",
                "value": {
                    "model": "gateway-image-pro",
                    "input": [
                        {"type": "image", "url": "https://example.test/ref.png"},
                        {"type": "text", "text": "stylise as watercolour"},
                    ],
                    "config": {"strength": 0.6},
                },
            },
        },
    ),
    prefer: str | None = Header(
        None, alias="Prefer",
        description='Set to "respond-async" to return a task id instead of blocking.',
        examples={"default": {"value": "respond-async"}},
    ),
    x_backend_tag: str | None = Header(
        None, alias="X-Backend-Tag", description="Pin to a backend by tag label.",
        examples={"default": {"value": "prod"}},
    ),
    x_backend: str | None = Header(
        None, alias="X-Backend", description="Pin to a backend by name.",
        examples={"default": {"value": "openai"}},
    ),
    x_request_id: str | None = Header(
        None, alias="X-Request-Id", description="Client-supplied request id (echoed back).",
        examples={"default": {"value": "req-01HZX4J3K7NQ8X2V9Y6R5W4T3P"}},
    ),
) -> Any:
    return await _create(request, key, wait=not _is_async(request, wait))


@router.post(
    "/v1/images/async",
    tags=["image"],
    response_model=CreateResponse,
    responses={422: {"description": "Validation Error"}},
)
async def image_create_async(
    request: Request,
    key: KeyConfig = Depends(get_api_key),
    body: ImageRequest = Body(
        ...,
        openapi_examples={
            "text_prompt": {
                "summary": "Text-to-image (string prompt)",
                "value": {"model": "gateway-image-pro", "input": "a neon cityscape"},
            },
        },
    ),
    prefer: str | None = Header(
        None, alias="Prefer", description="Ignored on the /async path (always async).",
        examples={"default": {"value": "respond-async"}},
    ),
    x_backend_tag: str | None = Header(
        None, alias="X-Backend-Tag", description="Pin to a backend by tag label.",
        examples={"default": {"value": "prod"}},
    ),
    x_backend: str | None = Header(
        None, alias="X-Backend", description="Pin to a backend by name.",
        examples={"default": {"value": "openai"}},
    ),
    x_request_id: str | None = Header(
        None, alias="X-Request-Id", description="Client-supplied request id (echoed back).",
        examples={"default": {"value": "req-01HZX4J3K7NQ8X2V9Y6R5W4T3P"}},
    ),
) -> Any:
    # The /async URL is unconditionally async — ?wait / Prefer are ignored.
    return await _create(request, key, wait=False)


@router.get(
    "/v1/images/{task_id}",
    tags=["image"],
    response_model=ImageTaskResponse,
    responses={
        404: {"description": "Unknown task id"},
        403: {"description": "Key not authorised for the task's backend"},
    },
)
async def image_get(
    request: Request,
    task_id: str = Path(..., description="Opaque task id returned by create.", examples={"default": {"value": "img-01HZX4J3K7NQ8X2V9Y6R5W4T3P"}}),
    key: KeyConfig = Depends(get_api_key),
    x_request_id: str | None = Header(
        None, alias="X-Request-Id", description="Client-supplied request id (echoed back).",
        examples={"default": {"value": "req-01HZX4J3K7NQ8X2V9Y6R5W4T3P"}},
    ),
) -> Any:
    service = request.app.state.image_service
    record = await request.app.state.task_store.get(task_id)
    authorize_task(request, key, record.provider if record else None)
    task = await service.get(task_id, backend_name=record.provider if record else None)
    out = gemini_compat.to_gemini_task(task)
    return JSONResponse(content=out)

