"""Seedance-compatible video translator.

The unified video schema **is** the Seedance (Volcengine Ark
``/contents/generations/tasks``) shape: a ``content`` array of typed parts
(text / image_url with a role / video_url / audio_url / draft_task) plus a flat
set of generation knobs. So the request translator is essentially a passthrough
into ``UnifiedVideoRequest(content=[...], ratio=..., ...)``, and the response
translators map the unified task back to the Seedance create/poll envelopes.
"""

from __future__ import annotations

from typing import Any

from mm_gateway.core.exceptions import ValidationError
from mm_gateway.schemas.video import (
    UnifiedVideoRequest,
    UnifiedVideoTask,
    audio_part,
    draft_part,
    image_part,
    text_part,
    video_part,
)

# Top-level fields the unified model owns; everything else -> extra.
_KNOWN = {
    "model", "content", "negative_prompt", "duration", "ratio", "resolution",
    "size", "width", "height", "fps", "seed", "generate_audio", "camera_fixed",
    "watermark", "prompt_extend", "callback_url", "return_last_frame",
}


def from_seedance(body: dict[str, Any]) -> UnifiedVideoRequest:
    if "model" not in body:
        raise ValidationError("`model` is required for video generation.")
    kwargs: dict[str, Any] = {"model": body["model"]}

    # Build typed content parts from the raw content array (if present).
    parts = []
    for part in body.get("content") or []:
        ptype = part.get("type")
        if ptype == "text":
            parts.append(text_part(part.get("text") or ""))
        elif ptype == "image_url":
            url = (part.get("image_url") or {}).get("url")
            if url:
                parts.append(image_part(url, part.get("role") or "first_frame"))
        elif ptype == "video_url":
            url = (part.get("video_url") or {}).get("url")
            if url:
                parts.append(video_part(url, part.get("role") or "reference_video"))
        elif ptype == "audio_url":
            url = (part.get("audio_url") or {}).get("url")
            if url:
                parts.append(audio_part(url, part.get("role") or "reference_audio"))
        elif ptype == "draft_task":
            if isinstance(part.get("draft_task"), dict):
                parts.append(draft_part(part["draft_task"]))
    if parts:
        kwargs["content"] = parts

    # Flat generation knobs.
    for k in ("negative_prompt", "duration", "ratio", "resolution", "size",
              "width", "height", "fps", "seed", "generate_audio", "camera_fixed",
              "watermark", "prompt_extend", "callback_url", "return_last_frame"):
        if (v := body.get(k)) is not None:
            kwargs[k] = v

    # Seedance accepts ``aspect_ratio`` as a synonym for ``ratio``.
    if "ratio" not in kwargs and (v := body.get("aspect_ratio")) is not None:
        kwargs["ratio"] = v

    extra: dict[str, Any] = {}
    for k, v in body.items():
        if k not in _KNOWN and k != "aspect_ratio":
            extra[k] = v
    if extra:
        kwargs["extra"] = extra
    return UnifiedVideoRequest(**kwargs)


def to_seedance_create(task: UnifiedVideoTask) -> dict[str, Any]:
    """The create endpoint returns only the task id (per Volcengine's contract)."""
    return {"id": task.task_id}


def to_seedance_task(task: UnifiedVideoTask) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": task.task_id,
        "model": task.model,
        "status": task.status,
    }
    content: dict[str, Any] = {}
    if task.video_urls:
        content["video_url"] = task.video_urls[0]
    if task.cover_url:
        content["last_frame_url"] = task.cover_url
    if content:
        out["content"] = content
    if task.error:
        out["error"] = {"code": "failed", "message": task.error}
    if task.usage:
        usage: dict[str, Any] = {}
        if task.usage.cost is not None:
            usage["cost"] = task.usage.cost
        if usage:
            out["usage"] = usage
    return out


__all__ = ["from_seedance", "to_seedance_create", "to_seedance_task"]
