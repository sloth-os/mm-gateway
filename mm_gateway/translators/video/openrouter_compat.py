"""OpenRouter-compatible video translator.

Maps OpenRouter's unified ``POST /api/v1/videos`` shape to/from the unified
video schema. The response carries a ``polling_url`` and, on completion,
``unsigned_urls``.
"""

from __future__ import annotations

from typing import Any

from mm_gateway.core.exceptions import ValidationError
from mm_gateway.schemas.video import UnifiedVideoRequest, UnifiedVideoTask

_KNOWN = {
    "model", "prompt", "aspect_ratio", "resolution", "size", "duration",
    "seed", "generate_audio", "callback_url",
}


def from_openrouter(body: dict[str, Any]) -> UnifiedVideoRequest:
    if "model" not in body:
        raise ValidationError("`model` is required for video generation.")
    kwargs: dict[str, Any] = {"model": body["model"]}
    extra: dict[str, Any] = {}
    for k, v in body.items():
        if k in _KNOWN and v is not None:
            kwargs[k] = v
        elif k == "frame_images":
            for frame in v or []:
                url = (frame.get("image_url") or {}).get("url")
                ftype = frame.get("frame_type")
                if not url:
                    continue
                if ftype == "last_frame":
                    kwargs["last_frame_image"] = url
                else:
                    kwargs["image"] = url
        elif k in ("input_references", "provider"):
            continue
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
