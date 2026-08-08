"""Image routes — Gemini-compatible.

Gemini image shape (``POST /v1/images``):

- ``POST /v1/images``       -> create task, returns ``{"id": ...}``
- ``GET  /v1/images/{id}``  -> poll task; response carries ``steps[].content[]``
                                image blocks (inline base64 or URL) plus
                                ``output_image`` / ``output_image_url`` helpers.

Sync vs async is controlled by the ``Prefer: respond-async`` header or the
``?wait=true`` query param. Without either, the ``image_sync_default`` setting
governs. The backend that owns a task id is recorded in the task store at
creation time so subsequent polls route correctly even though task ids are
opaque to the gateway. All routes require a valid front-end API key.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse

from mm_gateway.config import KeyConfig
from mm_gateway.observability.logging import get_logger
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


@router.post("/v1/images", tags=["image"])
async def image_create(
    request: Request,
    key: KeyConfig = Depends(get_api_key),
    wait: bool | None = Query(default=None),
) -> Any:
    body = await request.json()
    unified = gemini_compat.from_gemini(body)
    tag, backend_name = routing_overrides(request, body)
    service = request.app.state.image_service
    task = await service.create(
        unified, key=key, tag=tag, backend_name=backend_name,
        wait=not _is_async(request, wait),
    )
    await _remember(request.app.state.task_store, task)
    out = gemini_compat.to_gemini_create(task)
    return JSONResponse(content=out)


@router.get("/v1/images/{task_id}", tags=["image"])
async def image_get(
    task_id: str,
    request: Request,
    key: KeyConfig = Depends(get_api_key),
) -> Any:
    service = request.app.state.image_service
    record = await request.app.state.task_store.get(task_id)
    authorize_task(request, key, record.provider if record else None)
    task = await service.get(task_id, backend_name=record.provider if record else None)
    out = gemini_compat.to_gemini_task(task)
    return JSONResponse(content=out)
