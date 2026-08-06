"""xAI provider — Grok Imagine image and video via the xai_sdk gRPC client."""

from __future__ import annotations

import time
from typing import Any

from xai_sdk import AsyncClient

from mm_gateway.core.base import ImageProvider, VideoProvider
from mm_gateway.core.exceptions import ProviderNotConfiguredError, ProviderRequestError
from mm_gateway.observability.logging import get_logger
from mm_gateway.schemas.image import ImageData, ImageUsage, UnifiedImageRequest, UnifiedImageResponse
from mm_gateway.schemas.video import UnifiedVideoRequest, UnifiedVideoTask

log = get_logger("provider.xai")


class XAIProvider(ImageProvider, VideoProvider):
    name = "xai"
    image_models = ["grok-imagine-image", "grok-imagine-image-pro", "grok-imagine-image-quality"]
    video_models = ["grok-imagine-video", "grok-imagine-video-1.5-preview"]

    def __init__(self, backend):
        super().__init__(backend)
        if not backend.api_key:
            raise ProviderNotConfiguredError("xai")
        # The SDK reads XAI_API_KEY; we pass it explicitly for determinism.
        self._client = AsyncClient(api_key=backend.api_key)

    async def generate_image(self, request: UnifiedImageRequest) -> UnifiedImageResponse:
        kwargs: dict[str, Any] = {"prompt": request.prompt, "model": request.model}
        if request.response_format == "b64_json":
            kwargs["image_format"] = "base64"
        else:
            kwargs["image_format"] = "url"
        if request.aspect_ratio:
            kwargs["aspect_ratio"] = request.aspect_ratio
        if request.resolution:
            kwargs["resolution"] = request.resolution
        if request.user:
            kwargs["user"] = request.user
        if request.input_images:
            kwargs["image_urls"] = [i.url for i in request.input_images if i.url]
        kwargs.update(request.extra)

        try:
            resp = await self._client.image.sample(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise ProviderRequestError(f"xai image failed: {exc}", provider="xai") from exc

        data = [ImageData(url=resp.url or None, b64_json=resp.base64 or None)]
        return UnifiedImageResponse(
            created=int(time.time()), data=data, model=request.model,
            provider=self.name, usage=ImageUsage(cost=getattr(resp, "cost_usd", None)),
        )

    async def create_video_task(self, request: UnifiedVideoRequest) -> UnifiedVideoTask:
        kwargs: dict[str, Any] = {"prompt": request.prompt() or "", "model": request.model}
        if request.duration is not None:
            kwargs["duration"] = int(request.duration)
        if request.ratio:
            kwargs["aspect_ratio"] = request.ratio
        if request.resolution:
            kwargs["resolution"] = request.resolution
        if request.first_image():
            kwargs["image_url"] = request.first_image()
        if request.reference_images():
            kwargs["reference_image_urls"] = request.reference_images()
        kwargs.update(request.extra)

        try:
            start = await self._client.video.start(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise ProviderRequestError(f"xai video create failed: {exc}", provider="xai") from exc
        return UnifiedVideoTask(
            task_id=start.request_id, provider=self.name, model=request.model, status="pending",
        )

    _STATUS_MAP = {"PENDING": "pending", "DONE": "succeeded", "EXPIRED": "expired", "FAILED": "failed"}

    async def get_video_task(self, task_id: str) -> UnifiedVideoTask:
        try:
            result = await self._client.video.get(task_id)
        except Exception as exc:  # noqa: BLE001
            raise ProviderRequestError(f"xai video poll failed: {exc}", provider="xai") from exc
        status = self._STATUS_MAP.get(str(result.status), "running")
        task = UnifiedVideoTask(task_id=task_id, provider=self.name, model="", status=status)  # type: ignore[arg-type]
        if status == "succeeded" and result.response and getattr(result.response, "video", None):
            task.video_urls = [result.response.video.url]
        return task
