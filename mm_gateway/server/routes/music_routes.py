"""Music routes — Gemini Lyria 3-compatible.

Lyria 3 shape (``POST /v1beta/interactions`` mirrored at ``/v1/music``):

- ``POST /v1/music``         -> create task, returns ``{"id": ...}``
- ``POST /v1/music/async``   -> same, but always async (``wait=False``); the URL
                                  itself encodes the intent, so ``?wait`` /
                                  ``Prefer`` are ignored on this path.
- ``GET  /v1/music/{id}``    -> poll task; response carries ``steps[].content[]``
                                  audio/text blocks plus ``output_audio`` /
                                  ``output_text`` / ``output_audio_url`` helpers.

On the base ``/v1/music`` path, sync vs async is controlled by the
``Prefer: respond-async`` header or the ``?wait=true`` query param. Without
either, the ``music_sync_default`` setting governs. The ``/async`` sibling is
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
from mm_gateway.schemas.music import UnifiedMusicTask
from mm_gateway.server.auth import authorize_task, get_api_key, routing_overrides
from mm_gateway.tasks.store import TaskRecord
from mm_gateway.translators.music import lyria_compat

log = get_logger("route.music")
router = APIRouter()


def _is_async(request: Request, wait: bool | None) -> bool:
    if wait is not None:
        return not wait
    prefer = request.headers.get("prefer", "")
    return "respond-async" in prefer.lower()


async def _remember(store, task: UnifiedMusicTask) -> None:
    await store.put(TaskRecord(
        task_id=task.task_id, provider=task.provider, model=task.model,
        modality="music",
    ))


async def _create(request: Request, key: KeyConfig, *, wait: bool) -> Any:
    body = await request.json()
    unified = lyria_compat.from_lyria(body)
    tag, backend_name = routing_overrides(request, body)
    service = request.app.state.music_service
    task = await service.create(
        unified, key=key, tag=tag, backend_name=backend_name, wait=wait,
    )
    await _remember(request.app.state.task_store, task)
    out = lyria_compat.to_lyria_create(task)
    return JSONResponse(content=out)


@router.post("/v1/music", tags=["music"])
async def music_create(
    request: Request,
    key: KeyConfig = Depends(get_api_key),
    wait: bool | None = Query(default=None),
) -> Any:
    return await _create(request, key, wait=not _is_async(request, wait))


@router.post("/v1/music/async", tags=["music"])
async def music_create_async(
    request: Request,
    key: KeyConfig = Depends(get_api_key),
) -> Any:
    # The /async URL is unconditionally async — ?wait / Prefer are ignored.
    return await _create(request, key, wait=False)


@router.get("/v1/music/{task_id}", tags=["music"])
async def music_get(
    task_id: str,
    request: Request,
    key: KeyConfig = Depends(get_api_key),
) -> Any:
    service = request.app.state.music_service
    record = await request.app.state.task_store.get(task_id)
    authorize_task(request, key, record.provider if record else None)
    task = await service.get(task_id, backend_name=record.provider if record else None)
    out = lyria_compat.to_lyria_task(task)
    return JSONResponse(content=out)

