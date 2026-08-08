"""Video routes — Seedance-compatible.

Seedance shape (Volcengine Ark ``/contents/generations/tasks``):

- ``POST /v1/videos``         -> create task, returns ``{"id": ...}``
- ``POST /v1/videos/async``   -> same, but always async (``wait=False``); the URL
                                  itself encodes the intent, so ``?wait`` /
                                  ``Prefer`` are ignored on this path.
- ``GET  /v1/videos/{id}``    -> poll task

On the base ``/v1/videos`` path, sync vs async is controlled by the
``Prefer: respond-async`` header or the ``?wait=true`` query param. Without
either, the ``video_sync_default`` setting governs. The ``/async`` sibling is
the explicit async URL: clients that always want a task id back (and will poll
themselves) hit it directly and never block. The backend that owns a task id is
recorded in the task store at creation time so subsequent polls route correctly
even though task ids are opaque to the gateway. All routes require a valid
front-end API key.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse

from mm_gateway.config import KeyConfig
from mm_gateway.observability.logging import get_logger
from mm_gateway.schemas.video import UnifiedVideoTask
from mm_gateway.server.auth import authorize_task, get_api_key, routing_overrides
from mm_gateway.tasks.store import TaskRecord
from mm_gateway.translators.video import seedance_compat

log = get_logger("route.video")
router = APIRouter()


def _is_async(request: Request, wait: bool | None) -> bool:
    if wait is not None:
        return not wait
    prefer = request.headers.get("prefer", "")
    return "respond-async" in prefer.lower()


async def _remember(store, task: UnifiedVideoTask) -> None:
    await store.put(TaskRecord(
        task_id=task.task_id, provider=task.provider, model=task.model,
    ))


async def _create(request: Request, key: KeyConfig, *, wait: bool) -> Any:
    body = await request.json()
    unified = seedance_compat.from_seedance(body)
    tag, backend_name = routing_overrides(request, body)
    service = request.app.state.video_service
    task = await service.create(
        unified, key=key, tag=tag, backend_name=backend_name, wait=wait,
    )
    await _remember(request.app.state.task_store, task)
    out = seedance_compat.to_seedance_create(task)
    return JSONResponse(content=out)


@router.post("/v1/videos", tags=["video"])
async def seedance_create(
    request: Request,
    key: KeyConfig = Depends(get_api_key),
    wait: bool | None = Query(default=None),
) -> Any:
    return await _create(request, key, wait=not _is_async(request, wait))


@router.post("/v1/videos/async", tags=["video"])
async def seedance_create_async(
    request: Request,
    key: KeyConfig = Depends(get_api_key),
) -> Any:
    # The /async URL is unconditionally async — ?wait / Prefer are ignored.
    return await _create(request, key, wait=False)


@router.get("/v1/videos/{task_id}", tags=["video"])
async def seedance_get(
    task_id: str,
    request: Request,
    key: KeyConfig = Depends(get_api_key),
) -> Any:
    service = request.app.state.video_service
    record = await request.app.state.task_store.get(task_id)
    authorize_task(request, key, record.provider if record else None)
    task = await service.get(task_id, backend_name=record.provider if record else None)
    out = seedance_compat.to_seedance_task(task)
    return JSONResponse(content=out)

