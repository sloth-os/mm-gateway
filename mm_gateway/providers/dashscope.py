"""DashScope provider — Wanx (image) and Wan (video) via the async AIO classes."""

from __future__ import annotations

import time
from typing import Any

import dashscope
from dashscope.aigc.image_synthesis import AioImageSynthesis
from dashscope.aigc.video_synthesis import AioVideoSynthesis

from mm_gateway.core.base import ImageProvider, VideoProvider
from mm_gateway.core.exceptions import ProviderNotConfiguredError, ProviderRequestError, TaskFailedError
from mm_gateway.observability.logging import get_logger
from mm_gateway.schemas.image import ImageData, ImageUsage, UnifiedImageRequest, UnifiedImageResponse
from mm_gateway.schemas.video import UnifiedVideoRequest, UnifiedVideoTask

log = get_logger("provider.dashscope")

_STATUS_MAP = {
    "PENDING": "pending", "RUNNING": "running", "SUSPENDED": "running",
    "SUCCEEDED": "succeeded", "FAILED": "failed", "CANCELED": "cancelled", "UNKNOWN": "failed",
}


class DashScopeProvider(ImageProvider, VideoProvider):
    name = "dashscope"
    image_models = ["wanx2.1-t2i-turbo", "wanx2.1-t2i-plus", "wanx2.1-t2i-flash"]
    video_models = ["wanx2.1-t2v-turbo", "wanx2.1-i2v-turbo", "wanx2.1-t2v-plus", "wanx2.1-i2v-plus"]

    def __init__(self, backend):
        super().__init__(backend)
        if not backend.api_key:
            raise ProviderNotConfiguredError("dashscope")
        dashscope.api_key = backend.api_key
        if backend.base_url:
            dashscope.base_http_api_url = backend.base_url

    async def generate_image(self, request: UnifiedImageRequest) -> UnifiedImageResponse:
        kwargs: dict[str, Any] = {"model": request.model, "prompt": request.prompt}
        if request.n:
            kwargs["n"] = request.n
        if request.size:
            kwargs["size"] = request.size.replace("x", "*")
        if request.seed is not None:
            kwargs["seed"] = request.seed
        if request.negative_prompt:
            kwargs["negative_prompt"] = request.negative_prompt
        kwargs.update(request.extra)

        try:
            resp = await AioImageSynthesis.async_call(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise ProviderRequestError(f"dashscope image submit failed: {exc}", provider="dashscope") from exc

        task_id = resp.output.task_id
        final = await AioImageSynthesis.wait(resp, wait_timeout=300)
        if final.status_code != 200 or final.output.task_status != "SUCCEEDED":
            raise TaskFailedError(
                f"dashscope image task {task_id}: {final.output.task_status}", provider="dashscope"
            )
        data = [ImageData(url=getattr(r, "url", None)) for r in (final.output.results or [])]
        return UnifiedImageResponse(
            created=int(time.time()), data=data, model=request.model, provider=self.name,
            usage=ImageUsage(),
        )

    async def create_video_task(self, request: UnifiedVideoRequest) -> UnifiedVideoTask:
        kwargs: dict[str, Any] = {"model": request.model, "prompt": request.prompt() or ""}
        if request.size:
            kwargs["size"] = request.size.replace("x", "*")
        if request.ratio:
            kwargs["ratio"] = request.ratio
        if request.duration is not None:
            kwargs["duration"] = int(request.duration)
        if request.seed is not None:
            kwargs["seed"] = request.seed
        if request.prompt_extend is not None:
            kwargs["prompt_extend"] = request.prompt_extend
        if request.first_image():
            kwargs["img_url"] = request.first_image()
        kwargs.update(request.extra)

        try:
            resp = await AioVideoSynthesis.async_call(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise ProviderRequestError(f"dashscope video submit failed: {exc}", provider="dashscope") from exc

        task_id = resp.output.task_id
        return UnifiedVideoTask(task_id=task_id, provider=self.name, model=request.model, status="pending")

    async def get_video_task(self, task_id: str) -> UnifiedVideoTask:
        try:
            status = await AioVideoSynthesis.fetch(task_id)
        except Exception as exc:  # noqa: BLE001
            raise ProviderRequestError(f"dashscope video poll failed: {exc}", provider="dashscope") from exc
        st = _STATUS_MAP.get(status.output.task_status, "running")
        model = getattr(status.output, "model", "") or ""
        task = UnifiedVideoTask(task_id=task_id, provider=self.name, model=model, status=st)  # type: ignore[arg-type]
        if st == "succeeded":
            url = getattr(status.output, "video_url", None)
            if url:
                task.video_urls = [url]
            if getattr(status, "usage", None):
                task.usage = None  # populated if present
        elif st in ("failed", "cancelled", "expired"):
            task.error = getattr(status.output, "message", None) or status.output.task_status
        return task
