"""Video routes — Seedance-compatible and OpenRouter-compatible.

Seedance shape (Volcengine Ark ``/contents/generations/tasks``):

- ``POST /v1/videos``         -> create task, returns ``{"id": ...}``
- ``GET  /v1/videos/{id}``     -> poll task

OpenRouter unified shape:

- ``POST /api/v1/videos``               -> create task, returns polling handle
- ``GET  /api/v1/videos/{id}``          -> poll task
- ``GET  /api/v1/videos/{id}/content``   -> fetch completed video url(s)

Sync vs async is controlled by the ``Prefer: respond-async`` header or the
``?wait=true`` query param. Without either, the ``video_sync_default`` setting
governs. The backend that owns a task id is recorded in the task store at
creation time so subsequent polls route correctly even though task ids are
opaque to the gateway. All routes require a valid front-end API key.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Header, Query, Request
from fastapi.responses import JSONResponse

from mm_gateway.config import KeyConfig
from mm_gateway.observability.logging import get_logger
from mm_gateway.schemas.video import UnifiedVideoTask
from mm_gateway.server.auth import authorize_task, get_api_key, routing_overrides
from mm_gateway.tasks.store import TaskRecord
from mm_gateway.translators.video import openrouter_compat, seedance_compat

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


# --------------------------------------------------------------------------- #
# Seedance-compatible
# --------------------------------------------------------------------------- #

@router.post("/v1/videos", tags=["video"])
async def seedance_create(
    request: Request,
    key: KeyConfig = Depends(get_api_key),
    wait: bool | None = Query(default=None),
) -> Any:
    body = await request.json()
    unified = seedance_compat.from_seedance(body)
    tag, backend_name = routing_overrides(request, body)
    service = request.app.state.video_service
    task = await service.create(
        unified, key=key, tag=tag, backend_name=backend_name,
        wait=not _is_async(request, wait),
    )
    await _remember(request.app.state.task_store, task)
    out = seedance_compat.to_seedance_create(task)
    return JSONResponse(content=out)


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


# --------------------------------------------------------------------------- #
# OpenRouter-compatible
# --------------------------------------------------------------------------- #

def _base_url(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-host")
    host = fwd or request.headers.get("host", "")
    if host and not host.startswith("http"):
        host = f"http://{host}" if request.url.scheme == "http" else f"https://{host}"
    return host


@router.post("/api/v1/videos", tags=["video"])
async def openrouter_create(
    request: Request,
    key: KeyConfig = Depends(get_api_key),
    wait: bool | None = Query(default=None),
    x_response_format: str | None = Header(default=None, alias="X-Response-Format"),
) -> Any:
    body = await request.json()
    unified = openrouter_compat.from_openrouter(body)
    tag, backend_name = routing_overrides(request, body)
    service = request.app.state.video_service
    task = await service.create(
        unified, key=key, tag=tag, backend_name=backend_name,
        wait=not _is_async(request, wait),
    )
    await _remember(request.app.state.task_store, task)
    if x_response_format == "seedance":
        out = seedance_compat.to_seedance_create(task)
    else:
        out = openrouter_compat.to_openrouter(task, base_url=_base_url(request))
    return JSONResponse(content=out)


@router.get("/api/v1/videos/{task_id}", tags=["video"])
async def openrouter_get(
    task_id: str,
    request: Request,
    key: KeyConfig = Depends(get_api_key),
) -> Any:
    service = request.app.state.video_service
    record = await request.app.state.task_store.get(task_id)
    authorize_task(request, key, record.provider if record else None)
    task = await service.get(task_id, backend_name=record.provider if record else None)
    out = openrouter_compat.to_openrouter(task, base_url=_base_url(request))
    return JSONResponse(content=out)


@router.get("/api/v1/videos/{task_id}/content", tags=["video"])
async def openrouter_content(
    task_id: str,
    request: Request,
    key: KeyConfig = Depends(get_api_key),
) -> Any:
    service = request.app.state.video_service
    record = await request.app.state.task_store.get(task_id)
    authorize_task(request, key, record.provider if record else None)
    task = await service.get(task_id, backend_name=record.provider if record else None)
    if task.status != "succeeded":
        return JSONResponse(status_code=409, content={
            "id": task.task_id, "status": task.status,
            "error": task.error or f"task is {task.status}",
        })
    return JSONResponse(content={
        "id": task.task_id, "status": task.status,
        "unsigned_urls": task.video_urls,
        "cover_url": task.cover_url,
    })
