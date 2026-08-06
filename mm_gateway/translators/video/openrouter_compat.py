"""OpenRouter-compatible video translator.

Maps OpenRouter's unified ``POST /api/v1/videos`` shape to/from the unified
video schema. OpenRouter carries the prompt as a flat ``prompt`` string and
input images as ``frame_images``; the unified schema stores both in a
``content`` array of typed parts, so the request translator builds that array
and the response translator rebuilds ``frame_images`` from it on the way out.
"""

from __future__ import annotations

from typing import Any

from mm_gateway.core.exceptions import ValidationError
from mm_gateway.schemas.video import (
    UnifiedVideoRequest,
    UnifiedVideoTask,
    image_part,
    text_part,
)

_KNOWN = {
    "model", "prompt", "aspect_ratio", "resolution", "size", "duration",
    "seed", "generate_audio", "callback_url",
}


def from_openrouter(body: dict[str, Any]) -> UnifiedVideoRequest:
    if "model" not in body:
        raise ValidationError("`model` is required for video generation.")
    kwargs: dict[str, Any] = {"model": body["model"]}
    parts = []
    if body.get("prompt"):
        parts.append(text_part(body["prompt"]))
    for frame in body.get("frame_images") or []:
        url = (frame.get("image_url") or {}).get("url") if isinstance(frame, dict) else None
        if not url:
            continue
        ftype = frame.get("frame_type")
        role = "last_frame" if ftype == "last_frame" else "first_frame"
        parts.append(image_part(url, role))
    for ref in body.get("input_references") or []:
        url = (ref.get("image_url") or {}).get("url") if isinstance(ref, dict) else None
        if url:
            parts.append(image_part(url, "reference_image"))
    if parts:
        kwargs["content"] = parts

    extra: dict[str, Any] = {}
    for k, v in body.items():
        if k in ("frame_images", "input_references", "provider"):
            continue
        if k == "model":
            continue
        if k in _KNOWN:
            if v is not None:
                # OpenRouter uses ``aspect_ratio``; the unified schema names it ``ratio``.
                kwargs["ratio" if k == "aspect_ratio" else k] = v
        else:
            extra[k] = v
    if extra:
        kwargs["extra"] = extra
    return UnifiedVideoRequest(**kwargs)


def to_openrouter(task: UnifiedVideoTask, *, base_url: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": task.task_id,
        "status": task.status,
        "polling_url": f"{base_url}/api/v1/videos/{task.task_id}".strip("/"),
    }
    if task.video_urls:
        out["unsigned_urls"] = task.video_urls
    if task.usage:
        if task.usage.cost is not None:
            out["usage"] = {"cost": task.usage.cost}
    if task.error:
        out["error"] = task.error
    return out


__all__ = ["from_openrouter", "to_openrouter"]
