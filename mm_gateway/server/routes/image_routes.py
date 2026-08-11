"""REST endpoints for image generation tasks."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, Depends, Path, Request

from mm_gateway.config import KeyConfig
from mm_gateway.core.exceptions import TaskNotFoundError
from mm_gateway.schemas.api import ImageRequest, ImageTaskResponse
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
from mm_gateway.translators.rest import from_image_request, to_image_response

router = APIRouter(tags=["images"])


@router.post(
    "/v1/images",
    name="create_image",
    operation_id="createImage",
    summary="Create an image task",
    status_code=202,
    response_model=ImageTaskResponse,
    response_model_exclude_none=True,
    responses={
        202: {
            "description": "The image task was accepted.",
            "headers": RESOURCE_HEADERS,
        }
    },
)
async def create_image(
    request: Request,
    body: Annotated[ImageRequest, Body()],
    key: Annotated[KeyConfig, Depends(get_api_key)],
    idempotency_key: IdempotencyKeyHeader = None,
) -> ImageTaskResponse:
    routing = body.routing
    store = request.app.state.task_store
    fingerprint = request_fingerprint(body)
    async with store.idempotency_guard(key.id, "image", idempotency_key):
        record = await find_idempotent_record(
            store,
            owner_key_id=key.id,
            modality="image",
            idempotency_key=idempotency_key,
            fingerprint=fingerprint,
        )
        if record is not None:
            resource_url = str(request.url_for("get_image", image_id=record.task_id))
            resource = replay_resource(
                record, ImageTaskResponse, resource_url=resource_url
            )
            return render_resource(
                resource,
                request,
                resource_url=resource_url,
                created=True,
                replayed=True,
            )

        task = await request.app.state.image_service.create(
            from_image_request(body),
            key=key,
            tag=routing.profile if routing else None,
            wait=False,
        )
        record = new_record(
            "img",
            task,
            model=body.model,
            modality="image",
            metadata=body.metadata,
            owner_key_id=key.id,
            idempotency_key=idempotency_key,
            request_fingerprint=fingerprint if idempotency_key else None,
        )
        resource_url = str(request.url_for("get_image", image_id=record.task_id))
        resource = to_image_response(task, record, self_url=resource_url)
        remember_create_response(record, resource)
        await store.put(record)
        return render_resource(
            resource, request, resource_url=resource_url, created=True
        )


@router.get(
    "/v1/images/{image_id}",
    name="get_image",
    operation_id="getImage",
    summary="Retrieve an image task",
    response_model=ImageTaskResponse,
    response_model_exclude_none=True,
    responses={
        200: {
            "description": "The latest image task state.",
            "headers": {
                **POLL_HEADERS,
            },
        },
        304: {"description": "The task representation has not changed."},
        404: {"description": "Unknown image task id."},
    },
)
async def get_image(
    request: Request,
    image_id: Annotated[str, Path(description="Opaque image task id.")],
    key: Annotated[KeyConfig, Depends(get_api_key)],
    if_none_match: IfNoneMatchHeader = None,
) -> ImageTaskResponse:
    record = await request.app.state.task_store.get(image_id)
    if record is None or record.modality != "image":
        raise TaskNotFoundError(f"image task {image_id} not found")
    authorize_task(request, key, record)
    task = await request.app.state.image_service.get(
        record.provider_task_id or record.task_id,
        backend_name=record.provider,
    )
    resource_url = str(request.url_for("get_image", image_id=image_id))
    resource = to_image_response(task, record, self_url=resource_url)
    return render_resource(resource, request, resource_url=resource_url)


__all__ = ["router"]
