"""Shared HTTP mechanics for the three modality resource collections."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Annotated, Any, Literal, TypeVar
from uuid import uuid4

from fastapi import Header, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from mm_gateway.core.exceptions import ConflictError
from mm_gateway.tasks.store import TaskRecord

TERMINAL_STATUSES = {"succeeded", "failed", "cancelled", "expired"}
ResponseModel = TypeVar("ResponseModel", bound=BaseModel)

IdempotencyKeyHeader = Annotated[
    str | None,
    Header(
        alias="Idempotency-Key",
        min_length=1,
        max_length=255,
        description=(
            "Client-generated key used to safely retry this create request. "
            "Reuse with a different body returns 409."
        ),
    ),
]
IfNoneMatchHeader = Annotated[
    str | None,
    Header(
        alias="If-None-Match",
        description="Previously returned ETag; unchanged resources return 304.",
    ),
]

RESOURCE_HEADERS = {
    "Location": {
        "description": "Canonical URL of the created task resource.",
        "schema": {"type": "string", "format": "uri"},
    },
    "Link": {
        "description": 'Canonical task URL with rel="self".',
        "schema": {"type": "string"},
    },
    "Retry-After": {
        "description": "Suggested number of seconds before polling again.",
        "schema": {"type": "integer", "minimum": 1},
    },
    "ETag": {
        "description": "Version identifier for conditional polling.",
        "schema": {"type": "string"},
    },
    "Idempotency-Replayed": {
        "description": "true when the response replays an earlier create request.",
        "schema": {"type": "boolean"},
    },
}

POLL_HEADERS = {
    "Link": RESOURCE_HEADERS["Link"],
    "Retry-After": RESOURCE_HEADERS["Retry-After"],
    "ETag": RESOURCE_HEADERS["ETag"],
}


def new_record(
    prefix: str,
    task: Any,
    *,
    model: str,
    modality: Literal["image", "video", "music"],
    metadata: dict[str, Any],
    owner_key_id: str,
    idempotency_key: str | None = None,
    request_fingerprint: str | None = None,
) -> TaskRecord:
    return TaskRecord(
        task_id=f"{prefix}_{uuid4().hex}",
        provider=task.provider,
        provider_task_id=task.task_id,
        model=model,
        modality=modality,
        metadata=dict(metadata),
        owner_key_id=owner_key_id,
        idempotency_key=idempotency_key,
        request_fingerprint=request_fingerprint,
    )


def request_fingerprint(body: BaseModel) -> str:
    """Hash the normalized public request for idempotency conflict detection."""
    payload = body.model_dump(mode="json", by_alias=True, exclude_none=True)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


async def find_idempotent_record(
    store: Any,
    *,
    owner_key_id: str,
    modality: Literal["image", "video", "music"],
    idempotency_key: str | None,
    fingerprint: str,
) -> TaskRecord | None:
    if idempotency_key is None:
        return None
    record = await store.get_by_idempotency(
        owner_key_id, modality, idempotency_key
    )
    if record and record.request_fingerprint != fingerprint:
        raise ConflictError(
            "The Idempotency-Key was already used with a different request.",
            code="idempotency_conflict",
        )
    return record


def remember_create_response(record: TaskRecord, resource: BaseModel) -> None:
    record.create_response = resource.model_dump(
        mode="json", by_alias=True, exclude_none=True
    )


def replay_resource(
    record: TaskRecord,
    response_model: type[ResponseModel],
    *,
    resource_url: str,
) -> ResponseModel:
    if record.create_response is None:
        raise RuntimeError("idempotent task has no stored create response")
    payload = dict(record.create_response)
    payload["links"] = {"self": resource_url}
    return response_model.model_validate(payload)


def _etag(content: dict[str, Any]) -> str:
    encoded = json.dumps(
        content,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return f'"{hashlib.sha256(encoded).hexdigest()}"'


def _etag_matches(value: str | None, etag: str) -> bool:
    if not value:
        return False
    expected = etag.removeprefix("W/")
    return any(
        candidate == "*" or candidate.removeprefix("W/") == expected
        for candidate in (part.strip() for part in value.split(","))
    )


def render_conditional_json(
    content: dict[str, Any],
    request: Request,
    *,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
    cache_control: str = "private, no-cache",
    conditional: bool = True,
) -> Response:
    """Render authenticated JSON with ETag and optional 304 revalidation."""
    etag = _etag(content)
    response_headers = dict(headers or {})
    response_headers.update({
        "ETag": etag,
        "Cache-Control": cache_control,
        "Vary": "Authorization",
    })
    if conditional and _etag_matches(request.headers.get("if-none-match"), etag):
        return Response(status_code=304, headers=response_headers)
    return JSONResponse(
        status_code=status_code,
        content=content,
        headers=response_headers,
    )


def render_resource(
    resource: BaseModel,
    request: Request,
    *,
    resource_url: str,
    created: bool = False,
    replayed: bool = False,
) -> Response:
    status_value = resource.status
    content = resource.model_dump(mode="json", by_alias=True, exclude_none=True)
    headers = {
        "Link": f'<{resource_url}>; rel="self"',
    }
    if created:
        headers["Location"] = resource_url
    if replayed:
        headers["Idempotency-Replayed"] = "true"
    if status_value not in TERMINAL_STATUSES:
        headers["Retry-After"] = str(max(
            1, math.ceil(request.app.state.settings.poll_interval)
        ))
    return render_conditional_json(
        content,
        request,
        status_code=202 if created else 200,
        headers=headers,
        conditional=not created,
    )


__all__ = [
    "POLL_HEADERS",
    "RESOURCE_HEADERS",
    "IdempotencyKeyHeader",
    "IfNoneMatchHeader",
    "find_idempotent_record",
    "new_record",
    "remember_create_response",
    "render_conditional_json",
    "render_resource",
    "replay_resource",
    "request_fingerprint",
]
