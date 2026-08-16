"""REST endpoints for video generation tasks."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, Depends, Path, Request

from mm_gateway.config import KeyConfig
from mm_gateway.core.exceptions import TaskNotFoundError
from mm_gateway.schemas.api import VideoRequest, VideoTaskResponse
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
    stamped_model,
)
from mm_gateway.translators.rest import from_video_request, to_video_response

router = APIRouter(tags=["videos"])


@router.post(
    "/v1/videos",
    name="create_video",
    operation_id="createVideo",
    summary="Create a video task",
    status_code=202,
    response_model=VideoTaskResponse,
    response_model_exclude_none=True,
    responses={
        202: {
            "description": "The video task was accepted.",
            "headers": RESOURCE_HEADERS,
        }
    },
)
async def create_video(
    request: Request,
    body: Annotated[VideoRequest, Body()],
    key: Annotated[KeyConfig, Depends(get_api_key)],
    idempotency_key: IdempotencyKeyHeader = None,
) -> VideoTaskResponse:
    routing = body.routing
    store = request.app.state.task_store
    fingerprint = request_fingerprint(body)
    async with store.idempotency_guard(key.id, "video", idempotency_key):
        record = await find_idempotent_record(
            store,
            owner_key_id=key.id,
            modality="video",
            idempotency_key=idempotency_key,
            fingerprint=fingerprint,
        )
        if record is not None:
            resource_url = str(request.url_for("get_video", video_id=record.task_id))
            resource = replay_resource(
                record, VideoTaskResponse, resource_url=resource_url
            )
            return render_resource(
                resource,
                request,
                resource_url=resource_url,
                created=True,
                replayed=True,
            )

        task = await request.app.state.video_service.create(
            from_video_request(body),
            key=key,
            tag=routing.profile if routing else None,
            wait=False,
        )
        record = new_record(
            "vid",
            task,
            model=stamped_model(body.model, task.model),
            modality="video",
            metadata=body.metadata,
            owner_key_id=key.id,
            idempotency_key=idempotency_key,
            request_fingerprint=fingerprint if idempotency_key else None,
        )
        resource_url = str(request.url_for("get_video", video_id=record.task_id))
        resource = to_video_response(task, record, self_url=resource_url)
        remember_create_response(record, resource)
        await store.put(record)
        return render_resource(
            resource, request, resource_url=resource_url, created=True
        )


@router.get(
    "/v1/videos/{video_id}",
    name="get_video",
    operation_id="getVideo",
    summary="Retrieve a video task",
    response_model=VideoTaskResponse,
    response_model_exclude_none=True,
    responses={
        200: {
            "description": "The latest video task state.",
            "headers": {
                **POLL_HEADERS,
            },
        },
        304: {"description": "The task representation has not changed."},
        404: {"description": "Unknown video task id."},
    },
)
async def get_video(
    request: Request,
    video_id: Annotated[str, Path(description="Opaque video task id.")],
    key: Annotated[KeyConfig, Depends(get_api_key)],
    if_none_match: IfNoneMatchHeader = None,
) -> VideoTaskResponse:
    record = await request.app.state.task_store.get(video_id)
    if record is None or record.modality != "video":
        raise TaskNotFoundError(f"video task {video_id} not found")
    authorize_task(request, key, record)
    task = await request.app.state.video_service.get(
        record.provider_task_id or record.task_id,
        backend_name=record.provider,
    )
    resource_url = str(request.url_for("get_video", video_id=video_id))
    resource = to_video_response(task, record, self_url=resource_url)
    return render_resource(resource, request, resource_url=resource_url)


__all__ = ["router"]
