"""Volcengine Ark provider — Seedream (image) and Seedance (video).

Both modalities are served by the same ``AsyncArk`` client:

* **Image** (Seedream) via ``client.images.generate`` — OpenAI-compatible.
* **Video** (Seedance 1.0 and 2.0) via ``client.content_generation.tasks`` — the
  typed wrapper around ``POST/GET /contents/generations/tasks``. A single model
  id (e.g. ``doubao-seedance-2-0-260128``) serves text-to-video,
  image-to-video and reference-to-video; the *content parts* decide the mode
  (a ``text`` part for the prompt, ``image_url`` parts with roles like
  ``first_frame`` / ``last_frame`` / ``reference_image``, and ``video_url`` /
  ``audio_url`` parts for Seedance 2.0 multi-modal references).
"""

from __future__ import annotations

import time
from typing import Any

from volcenginesdkarkruntime import AsyncArk

from mm_gateway.config import ProviderCredentials
from mm_gateway.core.base import ImageProvider, VideoProvider
from mm_gateway.core.exceptions import ProviderNotConfiguredError, ProviderRequestError
from mm_gateway.observability.logging import get_logger
from mm_gateway.schemas.image import ImageData, ImageUsage, UnifiedImageRequest, UnifiedImageResponse
from mm_gateway.schemas.video import UnifiedVideoRequest, UnifiedVideoTask, VideoUsage

log = get_logger("provider.volcengine")

_BASE = "https://ark.cn-beijing.volces.com/api/v3"

# Ark task statuses -> unified lifecycle.
# ("queued" is the Ark pre-run state; "running" once generation starts.)
_STATUS_MAP = {
    "queued": "pending",
    "running": "running",
    "succeeded": "succeeded",
    "failed": "failed",
    "cancelled": "cancelled",
}


class VolcengineProvider(ImageProvider, VideoProvider):
    name = "volcengine"
    image_models = ["doubao-seedream-3-0-t2i-250415", "doubao-seedream-4-0-t2i-250828"]
    video_models = [
        # Seedance 1.0 — distinct t2v / i2v model ids.
        "doubao-seedance-1-0-pro-250528",
        "doubao-seedance-1-0-lite-i2v-250428",
        "doubao-seedance-1-0-pro-i2v-250528",
        # Seedance 2.0 — one omni model; the content parts pick t2v / i2v / r2v.
        "doubao-seedance-2-0-260128",
    ]

    def __init__(self, credentials: ProviderCredentials):
        super().__init__(credentials)
        if not credentials.api_key:
            raise ProviderNotConfiguredError("volcengine")
        base = credentials.base_url or _BASE
        self._ark = AsyncArk(api_key=credentials.api_key, base_url=base)

    async def generate_image(self, request: UnifiedImageRequest) -> UnifiedImageResponse:
        kwargs: dict[str, Any] = {"model": request.model, "prompt": request.prompt}
        if request.size:
            kwargs["size"] = request.size
        if request.response_format:
            kwargs["response_format"] = request.response_format
        if request.seed is not None:
            kwargs["seed"] = request.seed
        if request.guidance_scale is not None:
            kwargs["guidance_scale"] = request.guidance_scale
        if request.watermark is not None:
            kwargs["watermark"] = request.watermark
        if request.input_images:
            kwargs["image"] = [i.url for i in request.input_images if i.url]
        kwargs.update(request.extra)

        try:
            resp = await self._ark.images.generate(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise ProviderRequestError(f"volcengine image failed: {exc}", provider="volcengine") from exc

        data = [ImageData(url=getattr(d, "url", None), b64_json=getattr(d, "b64_json", None))
                for d in (resp.data or [])]
        usage = None
        if getattr(resp, "usage", None):
            usage = ImageUsage(total_tokens=getattr(resp.usage, "total_tokens", None))
        return UnifiedImageResponse(
            created=getattr(resp, "created_at", None) or int(time.time()),
            data=data, model=request.model, provider=self.name, usage=usage,
        )

    # -- Seedance video --------------------------------------------------- #

    def _build_content(self, request: UnifiedVideoRequest) -> list[dict[str, Any]]:
        """Translate the unified request into Ark content parts.

        Seedance is content-driven: a ``text`` part is the prompt, ``image_url``
        parts carry first/last frame and reference images (by role), and
        ``video_url`` / ``audio_url`` parts carry Seedance 2.0 multi-modal
        references (read from ``request.extra`` since the unified model has no
        first-class field for them).
        """
        content: list[dict[str, Any]] = []
        if request.prompt:
            content.append({"type": "text", "text": request.prompt})
        if request.image:
            content.append({"type": "image_url", "image_url": {"url": request.image}, "role": "first_frame"})
        if request.last_frame_image:
            content.append({"type": "image_url", "image_url": {"url": request.last_frame_image}, "role": "last_frame"})
        for img in request.reference_images or []:
            content.append({"type": "image_url", "image_url": {"url": img}, "role": "reference_image"})
        for vid in (request.extra.get("reference_videos") or []):
            content.append({"type": "video_url", "video_url": {"url": vid}, "role": "reference_video"})
        for aud in (request.extra.get("reference_audios") or []):
            content.append({"type": "audio_url", "audio_url": {"url": aud}, "role": "reference_audio"})
        return content

    async def create_video_task(self, request: UnifiedVideoRequest) -> UnifiedVideoTask:
        kwargs: dict[str, Any] = {"model": request.model, "content": self._build_content(request)}
        if request.duration is not None:
            kwargs["duration"] = int(request.duration)
        if request.resolution:
            kwargs["resolution"] = request.resolution
        if request.aspect_ratio:
            kwargs["ratio"] = request.aspect_ratio
        if request.camera_fixed is not None:
            kwargs["camera_fixed"] = request.camera_fixed
        if request.seed is not None:
            kwargs["seed"] = request.seed
        if request.watermark is not None:
            kwargs["watermark"] = request.watermark
        if request.generate_audio is not None:
            kwargs["generate_audio"] = request.generate_audio
        if request.callback_url:
            kwargs["callback_url"] = request.callback_url
        # Seedance-specific knobs the unified model doesn't name.
        if (v := request.extra.get("return_last_frame")) is not None:
            kwargs["return_last_frame"] = v
        if (v := request.extra.get("frames")) is not None:
            kwargs["frames"] = v
        if (v := request.extra.get("draft")) is not None:
            kwargs["draft"] = v
        if (v := request.extra.get("service_tier")) is not None:
            kwargs["service_tier"] = v
        if (v := request.extra.get("priority")) is not None:
            kwargs["priority"] = v

        try:
            result = await self._ark.content_generation.tasks.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise ProviderRequestError(f"volcengine video create failed: {exc}", provider="volcengine") from exc

        task_id = getattr(result, "id", None) or ""
        if not task_id:
            raise ProviderRequestError("volcengine video create returned no task id", provider="volcengine")
        return UnifiedVideoTask(task_id=task_id, provider=self.name, model=request.model, status="pending")

    async def get_video_task(self, task_id: str) -> UnifiedVideoTask:
        try:
            result = await self._ark.content_generation.tasks.get(task_id=task_id)
        except Exception as exc:  # noqa: BLE001
            raise ProviderRequestError(f"volcengine video poll failed: {exc}", provider="volcengine") from exc

        status = _STATUS_MAP.get(getattr(result, "status", "") or "", "running")
        task = UnifiedVideoTask(
            task_id=task_id, provider=self.name,
            model=getattr(result, "model", "") or "",
            status=status,  # type: ignore[arg-type]
            created_at=_to_epoch(getattr(result, "created_at", None)),
            completed_at=_to_epoch(getattr(result, "updated_at", None)),
        )
        content = getattr(result, "content", None)
        if content is not None:
            video_url = getattr(content, "video_url", None)
            if video_url:
                task.video_urls = [video_url]
            last_frame = getattr(content, "last_frame_url", None)
            if last_frame:
                task.cover_url = last_frame
        usage = getattr(result, "usage", None)
        if usage is not None:
            task.usage = VideoUsage(extra={
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            })
        err = getattr(result, "error", None)
        if err is not None and status == "failed":
            code = getattr(err, "code", "") or ""
            message = getattr(err, "message", "") or ""
            task.error = f"{code}: {message}".strip(": ")
        return task


def _to_epoch(value: Any) -> int | None:
    """Ark timestamps are Unix-epoch seconds (int); pass through unchanged."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    # Tolerate a numeric/float or ISO string defensively.
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
