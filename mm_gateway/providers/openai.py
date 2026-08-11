"""OpenAI provider — DALL·E / GPT-Image (image) and Sora (video)."""

from __future__ import annotations

import base64
import time
from typing import Any, ClassVar

import httpx
from openai import AsyncOpenAI

from mm_gateway.core.base import ImageProvider, VideoProvider
from mm_gateway.core.exceptions import ProviderNotConfiguredError, ProviderRequestError
from mm_gateway.observability.httplog import backend_event_hooks
from mm_gateway.observability.logging import get_logger
from mm_gateway.providers._dimensions import pixel_size
from mm_gateway.providers._sync_image import SyncImageTaskMixin
from mm_gateway.schemas.image import (
    ImageData,
    ImageUsage,
    UnifiedImageRequest,
    UnifiedImageResponse,
)
from mm_gateway.schemas.video import UnifiedVideoRequest, UnifiedVideoTask

log = get_logger("provider.openai")


def _logged_httpx() -> httpx.AsyncClient:
    return httpx.AsyncClient(event_hooks=backend_event_hooks())


class OpenAIProvider(SyncImageTaskMixin, ImageProvider, VideoProvider):
    name = "openai"
    image_models: ClassVar[list[str]] = [
        "gpt-image-1",
        "gpt-image-1-mini",
        "gpt-image-2",
        "dall-e-2",
        "dall-e-3",
    ]
    video_models: ClassVar[list[str]] = ["sora-2", "sora-2-pro"]

    def __init__(self, backend):
        super().__init__(backend)
        if not backend.api_key:
            raise ProviderNotConfiguredError("openai")
        # Per-modality clients honor the sync/async URL split resolved by
        # ``config.py``: image (DALL·E/GPT-Image) uses ``base_url`` (the
        # ``*_IMAGE_BASE_URL`` sync endpoint); video (Sora) uses
        # ``extra["video_base_url"]`` (the ``*_VIDEO_BASE_URL`` async endpoint)
        # when it differs from the image one. The real api.openai.com serves
        # both at one host, so the two clients collapse unless an operator pins
        # them apart.
        image_base = backend.base_url or None
        video_base = backend.extra.get("video_base_url") or image_base
        self._client = AsyncOpenAI(api_key=backend.api_key, base_url=image_base, http_client=_logged_httpx())
        self._client_video = AsyncOpenAI(api_key=backend.api_key, base_url=video_base, http_client=_logged_httpx())

    async def _generate_image(self, request: UnifiedImageRequest) -> UnifiedImageResponse:
        kwargs: dict[str, Any] = {"model": request.model, "prompt": request.prompt() or ""}
        if request.n and request.n > 1:
            kwargs["n"] = request.n
        if size := pixel_size(request):
            kwargs["size"] = size
        if request.quality:
            kwargs["quality"] = request.quality
        if request.style:
            kwargs["style"] = request.style
        if request.response_format:
            kwargs["response_format"] = request.response_format
        if request.output_format:
            kwargs["output_format"] = request.output_format
        if request.background:
            kwargs["background"] = request.background
        if request.user:
            kwargs["user"] = request.user
        kwargs.update(request.extra)

        try:
            resp = await self._client.images.generate(**kwargs)
        except Exception as exc:
            raise ProviderRequestError(f"openai image failed: {exc}", provider="openai") from exc

        data = [
            ImageData(url=d.url, b64_json=d.b64_json, revised_prompt=getattr(d, "revised_prompt", None))
            for d in (resp.data or [])
        ]
        usage = None
        if getattr(resp, "usage", None):
            u = resp.usage
            usage = ImageUsage(
                input_tokens=getattr(u, "input_tokens", None),
                output_tokens=getattr(u, "output_tokens", None),
                total_tokens=getattr(u, "total_tokens", None),
            )
        return UnifiedImageResponse(
            created=resp.created or int(time.time()),
            data=data,
            model=request.model,
            provider=self.name,
            usage=usage,
        )

    async def create_video_task(self, request: UnifiedVideoRequest) -> UnifiedVideoTask:
        kwargs: dict[str, Any] = {"model": request.model, "prompt": request.prompt() or ""}
        if request.duration is not None:
            # Sora accepts "4" | "8" | "12"
            kwargs["seconds"] = str(int(request.duration))
        if size := pixel_size(request):
            kwargs["size"] = size
        if request.first_image():
            kwargs["input_reference"] = {"image_url": request.first_image()}
        kwargs.update(request.extra)

        try:
            video = await self._client_video.videos.create(**kwargs)
        except Exception as exc:
            raise ProviderRequestError(f"openai video create failed: {exc}", provider="openai") from exc
        return self._to_task(video)

    async def get_video_task(self, task_id: str) -> UnifiedVideoTask:
        try:
            video = await self._client_video.videos.retrieve(task_id)
        except Exception as exc:
            raise ProviderRequestError(f"openai video poll failed: {exc}", provider="openai") from exc
        task = self._to_task(video)
        # Sora does not put the URL on the job object; fetch on completion.
        if task.status == "succeeded" and not task.video_urls:
            try:
                content = await self._client_video.videos.download_content(task_id)
                blob = await content.aread()
                data_url = "data:video/mp4;base64," + base64.b64encode(blob).decode()
                task.video_urls = [data_url]
            except Exception as exc:  # noqa: BLE001
                log.warning("openai_video_download_failed", task_id=task_id, error=str(exc))
        return task

    _STATUS_MAP: ClassVar[dict[str, str]] = {
        "queued": "pending", "in_progress": "running",
        "completed": "succeeded", "failed": "failed",
    }

    def _to_task(self, video: Any) -> UnifiedVideoTask:
        status = self._STATUS_MAP.get(video.status, "running")
        err = None
        if getattr(video, "error", None):
            err = f"{video.error.code}: {video.error.message}"
        return UnifiedVideoTask(
            task_id=video.id,
            provider=self.name,
            model=video.model or "",
            status=status,  # type: ignore[arg-type]
            error=err,
            created_at=getattr(video, "created_at", None),
            completed_at=getattr(video, "completed_at", None),
            raw={"progress": getattr(video, "progress", None)},
        )
