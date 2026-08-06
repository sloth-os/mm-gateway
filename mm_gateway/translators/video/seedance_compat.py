"""Seedance-compatible video translator.

Maps the Volcengine Ark ``/contents/generations/tasks`` (Seedance) shape to/from
the unified video schema. The create body uses a ``content`` array of typed
parts (text / image_url with a role), and a flat ``parameters``-ish set of
fields (duration, resolution, ratio, camera_fixed, seed, watermark).
"""

from __future__ import annotations

from typing import Any

from mm_gateway.core.exceptions import ValidationError
from mm_gateway.schemas.video import UnifiedVideoRequest, UnifiedVideoTask


def from_seedance(body: dict[str, Any]) -> UnifiedVideoRequest:
    if "model" not in body:
        raise ValidationError("`model` is required for video generation.")
    prompt = None
    image = None
    last_frame = None
    reference_images: list[str] = []
    reference_videos: list[str] = []
    reference_audios: list[str] = []
    for part in body.get("content") or []:
        ptype = part.get("type")
        if ptype == "text":
            prompt = part.get("text")
        elif ptype == "image_url":
            url = (part.get("image_url") or {}).get("url")
            role = part.get("role")
            if role == "last_frame":
                last_frame = url
            elif role == "reference_image" and url:
                reference_images.append(url)
            else:
                # "first_frame" (or unspecified) — the leading input frame.
                image = url
        elif ptype == "video_url":
            url = (part.get("video_url") or {}).get("url")
            if url:
                reference_videos.append(url)
        elif ptype == "audio_url":
            url = (part.get("audio_url") or {}).get("url")
            if url:
                reference_audios.append(url)

    kwargs: dict[str, Any] = {"model": body["model"]}
    if prompt:
        kwargs["prompt"] = prompt
    if image:
        kwargs["image"] = image
    if last_frame:
        kwargs["last_frame_image"] = last_frame
    if reference_images:
        kwargs["reference_images"] = reference_images
    if (v := body.get("duration")) is not None:
        kwargs["duration"] = v
    if (v := body.get("resolution")) is not None:
        kwargs["resolution"] = v
    if (v := body.get("ratio")) is not None:
        kwargs["aspect_ratio"] = v
    if (v := body.get("camera_fixed")) is not None:
        kwargs["camera_fixed"] = v
    if (v := body.get("seed")) is not None:
        kwargs["seed"] = v
    if (v := body.get("watermark")) is not None:
        kwargs["watermark"] = v
    if (v := body.get("generate_audio")) is not None:
        kwargs["generate_audio"] = v
    if (v := body.get("callback_url")) is not None:
        kwargs["callback_url"] = v

    extra: dict[str, Any] = {}
    if reference_videos:
        extra["reference_videos"] = reference_videos
    if reference_audios:
        extra["reference_audios"] = reference_audios
    if (v := body.get("return_last_frame")) is not None:
        extra["return_last_frame"] = v
    for k, v in body.items():
        if k not in {"model", "content", "duration", "resolution", "ratio",
                     "camera_fixed", "seed", "watermark", "generate_audio",
                     "callback_url", "return_last_frame"}:
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
