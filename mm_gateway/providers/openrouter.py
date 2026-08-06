"""OpenRouter provider — the unified image/video REST API as a backend.

OpenRouter is itself a normalising router, so this adapter is mostly a
passthrough: it forwards the unified request to OpenRouter's ``/images`` and
``/videos`` endpoints and maps the response back. Useful when the gateway
should delegate routing/billing to OpenRouter instead of picking a provider.
"""

from __future__ import annotations

import time
from typing import Any

from mm_gateway.core.base import ImageProvider, VideoProvider
from mm_gateway.core.exceptions import ProviderNotConfiguredError, ProviderRequestError
from mm_gateway.observability.logging import get_logger
from mm_gateway.providers._http import make_client, request_json
from mm_gateway.schemas.image import ImageData, ImageUsage, UnifiedImageRequest, UnifiedImageResponse
from mm_gateway.schemas.video import UnifiedVideoRequest, UnifiedVideoTask

log = get_logger("provider.openrouter")

_BASE = "https://openrouter.ai/api/v1"
_STATUS_MAP = {
    "pending": "pending", "in_progress": "running",
    "completed": "succeeded", "failed": "failed",
    "cancelled": "cancelled", "expired": "expired",
}


class OpenRouterProvider(ImageProvider, VideoProvider):
    name = "openrouter"
    image_models = []  # OpenRouter's catalogue is dynamic; resolved at request time.
    video_models = []

    def __init__(self, backend):
        super().__init__(backend)
        if not backend.api_key:
            raise ProviderNotConfiguredError("openrouter")
        self._client = make_client(
            backend.base_url or _BASE, timeout=120,
            headers={"Authorization": f"Bearer {backend.api_key}"},
        )

    async def generate_image(self, request: UnifiedImageRequest) -> UnifiedImageResponse:
        body = _image_body(request)
        result = await request_json(self._client, "POST", "/images", provider="openrouter", json=body)
        data = [ImageData(b64_json=d.get("b64_json"), media_type=d.get("media_type"))
                for d in (result.get("data") or [])]
        usage = None
        if result.get("usage"):
            usage = ImageUsage(cost=result["usage"].get("cost"))
        return UnifiedImageResponse(
            created=result.get("created", int(time.time())), data=data,
            model=request.model, provider=self.name, usage=usage,
        )

    async def create_video_task(self, request: UnifiedVideoRequest) -> UnifiedVideoTask:
        body = _video_body(request)
        result = await request_json(self._client, "POST", "/videos", provider="openrouter", json=body)
        task_id = result.get("id", "")
        if not task_id:
            raise ProviderRequestError("openrouter video create returned no id", provider="openrouter")
        status = _STATUS_MAP.get(result.get("status", "pending"), "pending")
        task = UnifiedVideoTask(task_id=task_id, provider=self.name, model=request.model, status=status)  # type: ignore[arg-type]
        if result.get("unsigned_urls"):
            task.video_urls = result["unsigned_urls"]
        return task

    async def get_video_task(self, task_id: str) -> UnifiedVideoTask:
        result = await request_json(self._client, "GET", f"/videos/{task_id}", provider="openrouter")
        status = _STATUS_MAP.get(result.get("status", ""), "running")
        task = UnifiedVideoTask(
            task_id=task_id, provider=self.name,
            model=result.get("model", ""), status=status,  # type: ignore[arg-type]
            raw=result,
        )
        if result.get("unsigned_urls"):
            task.video_urls = result["unsigned_urls"]
        if result.get("error"):
            task.error = str(result["error"])
        return task


def _image_body(request: UnifiedImageRequest) -> dict[str, Any]:
    body: dict[str, Any] = {"model": request.model, "prompt": request.prompt}
    if request.n:
        body["n"] = request.n
    if request.aspect_ratio:
        body["aspect_ratio"] = request.aspect_ratio
    if request.resolution:
        body["resolution"] = request.resolution
    if request.size:
        body["size"] = request.size
    if request.quality:
        body["quality"] = request.quality
    if request.seed is not None:
        body["seed"] = request.seed
    if request.background:
        body["background"] = request.background
    if request.output_format:
        body["output_format"] = request.output_format
    if request.output_compression is not None:
        body["output_compression"] = request.output_compression
    if request.input_images:
        body["input_references"] = [
            {"type": "image_url", "image_url": {"url": i.url or f"data:image/png;base64,{i.b64_json}"}}
            for i in request.input_images
        ]
    body.update(request.extra)
    return body


def _video_body(request: UnifiedVideoRequest) -> dict[str, Any]:
    body: dict[str, Any] = {"model": request.model}
    prompt = request.prompt()
    if prompt:
        body["prompt"] = prompt
    if request.ratio:
        body["aspect_ratio"] = request.ratio
    if request.resolution:
        body["resolution"] = request.resolution
    if request.size:
        body["size"] = request.size
    if request.duration is not None:
        body["duration"] = int(request.duration)
    if request.seed is not None:
        body["seed"] = request.seed
    if request.generate_audio is not None:
        body["generate_audio"] = request.generate_audio
    if request.callback_url:
        body["callback_url"] = request.callback_url
    frames: list[dict[str, Any]] = []
    first = request.first_image()
    if first:
        frames.append({"type": "image_url", "image_url": {"url": first}, "frame_type": "first_frame"})
    last = request.last_image()
    if last:
        frames.append({"type": "image_url", "image_url": {"url": last}, "frame_type": "last_frame"})
    if frames:
        body["frame_images"] = frames
    refs = request.reference_images()
    if refs:
        body["input_references"] = [
            {"type": "image_url", "image_url": {"url": u}} for u in refs
        ]
    body.update(request.extra)
    return body
