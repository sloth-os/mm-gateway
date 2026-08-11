"""REST endpoints for music generation tasks."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, Depends, Path, Request

from mm_gateway.config import KeyConfig
from mm_gateway.core.exceptions import TaskNotFoundError
from mm_gateway.schemas.api import MusicRequest, MusicTaskResponse
from mm_gateway.server.auth import authorize_task, get_api_key
from mm_gateway.server.routes._resources import (
    POLL_HEADERS,
    RESOURCE_HEADERS,
    IdempotencyKeyHeader,
    IfNoneMatchHeader,
    find_idempotent_record,
    new_record,
    remember_create_response,
    render_resource,
    replay_resource,
    request_fingerprint,
)
from mm_gateway.translators.rest import from_music_request, to_music_response

router = APIRouter(tags=["music"])


@router.post(
    "/v1/music",
    name="create_music",
    operation_id="createMusic",
    summary="Create a music task",
    status_code=202,
    response_model=MusicTaskResponse,
    response_model_exclude_none=True,
    responses={
        202: {
            "description": "The music task was accepted.",
            "headers": RESOURCE_HEADERS,
        }
    },
)
async def create_music(
    request: Request,
    body: Annotated[MusicRequest, Body()],
    key: Annotated[KeyConfig, Depends(get_api_key)],
    idempotency_key: IdempotencyKeyHeader = None,
) -> MusicTaskResponse:
    routing = body.routing
    store = request.app.state.task_store
    fingerprint = request_fingerprint(body)
    async with store.idempotency_guard(key.id, "music", idempotency_key):
        record = await find_idempotent_record(
            store,
            owner_key_id=key.id,
            modality="music",
            idempotency_key=idempotency_key,
            fingerprint=fingerprint,
        )
        if record is not None:
            resource_url = str(request.url_for("get_music", music_id=record.task_id))
            resource = replay_resource(
                record, MusicTaskResponse, resource_url=resource_url
            )
            return render_resource(
                resource,
                request,
                resource_url=resource_url,
                created=True,
                replayed=True,
            )

        task = await request.app.state.music_service.create(
            from_music_request(body),
            key=key,
            tag=routing.profile if routing else None,
            wait=False,
        )
        record = new_record(
            "mus",
            task,
            model=body.model,
            modality="music",
            metadata=body.metadata,
            owner_key_id=key.id,
            idempotency_key=idempotency_key,
            request_fingerprint=fingerprint if idempotency_key else None,
        )
        resource_url = str(request.url_for("get_music", music_id=record.task_id))
        resource = to_music_response(task, record, self_url=resource_url)
        remember_create_response(record, resource)
        await store.put(record)
        return render_resource(
            resource, request, resource_url=resource_url, created=True
        )


@router.get(
    "/v1/music/{music_id}",
    name="get_music",
    operation_id="getMusic",
    summary="Retrieve a music task",
    response_model=MusicTaskResponse,
    response_model_exclude_none=True,
    responses={
        200: {
            "description": "The latest music task state.",
            "headers": {
                **POLL_HEADERS,
            },
        },
        304: {"description": "The task representation has not changed."},
        404: {"description": "Unknown music task id."},
    },
)
async def get_music(
    request: Request,
    music_id: Annotated[str, Path(description="Opaque music task id.")],
    key: Annotated[KeyConfig, Depends(get_api_key)],
    if_none_match: IfNoneMatchHeader = None,
) -> MusicTaskResponse:
    record = await request.app.state.task_store.get(music_id)
    if record is None or record.modality != "music":
        raise TaskNotFoundError(f"music task {music_id} not found")
    authorize_task(request, key, record)
    task = await request.app.state.music_service.get(
        record.provider_task_id or record.task_id,
        backend_name=record.provider,
    )
    resource_url = str(request.url_for("get_music", music_id=music_id))
    resource = to_music_response(task, record, self_url=resource_url)
    return render_resource(resource, request, resource_url=resource_url)


__all__ = ["router"]
