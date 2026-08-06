"""Stability AI provider — Stable Image (SD3/SDXL/Core) and SVD (video).

The installed ``stability-sdk`` is a legacy gRPC client for SD v1.5 only; the
modern API is REST on ``api.stability.ai``. This adapter uses the REST API via
httpx so it can reach SD3/3.5, SDXL, Core, and Stable Video Diffusion.
Image gen is synchronous (one long HTTP call); video gen is also synchronous
on v2beta but we expose it through the task model so the gateway can offer a
poll surface — the task "completes" once the single blocking call returns.
"""

from __future__ import annotations

import base64
import time
import uuid
from typing import Any

import httpx

from mm_gateway.core.base import ImageProvider, VideoProvider
from mm_gateway.core.exceptions import ProviderNotConfiguredError, ProviderRequestError, TaskFailedError
from mm_gateway.schemas.image import ImageData, ImageUsage, UnifiedImageRequest, UnifiedImageResponse
from mm_gateway.schemas.video import UnifiedVideoRequest, UnifiedVideoTask

_BASE = "https://api.stability.ai/v2beta"

_IMAGE_PATHS = {
    "sd3": "/stable-image/generate/sd3",
    "sdxl": "/stable-image/generate/sdxl",
    "stable-image-core": "/stable-image/generate/core",
    "stable-image-ultra": "/stable-image/generate/ultra",
}

# In-memory store for the synchronous-video "tasks". SVD has no job id, so we
# mint a gateway id and run the blocking call on poll. (Single-process only;
# a real deployment would use a durable task store — see tasks/store.py.)
_VIDEO_TASKS: dict[str, dict[str, Any]] = {}


class StabilityProvider(ImageProvider, VideoProvider):
    name = "stability"
    image_models = ["sd3.5-large", "sd3.5-medium", "sdxl", "stable-image-core", "stable-image-ultra"]
    video_models = ["stable-video-1-1", "stable-video-1-0", "stable-video-diffusion"]

    def __init__(self, backend):
        super().__init__(backend)
        if not backend.api_key:
            raise ProviderNotConfiguredError("stability")
        self._api_key = backend.api_key
        self._client = httpx.AsyncClient(
            base_url=backend.base_url or _BASE, timeout=300,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )

    def _image_path(self, model: str) -> str:
        if model in _IMAGE_PATHS:
            return _IMAGE_PATHS[model]
        if model.startswith("sd3"):
            return _IMAGE_PATHS["sd3"]
        if "core" in model:
            return _IMAGE_PATHS["stable-image-core"]
        if "ultra" in model:
            return _IMAGE_PATHS["stable-image-ultra"]
        return _IMAGE_PATHS["sdxl"]

    async def generate_image(self, request: UnifiedImageRequest) -> UnifiedImageResponse:
        data: dict[str, Any] = {"prompt": request.prompt}
        if request.model and request.model.startswith("sd3"):
            data["model"] = request.model
        if request.seed is not None:
            data["seed"] = request.seed
        if request.aspect_ratio:
            data["aspect_ratio"] = request.aspect_ratio
        if request.output_format:
            data["output_format"] = request.output_format
        if request.guidance_scale is not None:
            data["cfg_scale"] = request.guidance_scale
        if request.num_inference_steps is not None:
            data["steps"] = request.num_inference_steps
        if request.style:
            data["style"] = request.style
        if request.negative_prompt:
            data["negative_prompt"] = request.negative_prompt
        data.update(request.extra)

        accept = "application/json" if request.response_format == "b64_json" else "image/*"
        headers = {"Accept": accept}
        try:
            resp = await self._client.post(self._image_path(request.model), data=data, headers=headers)
        except httpx.HTTPError as exc:
            raise ProviderRequestError(f"stability transport error: {exc}", provider="stability") from exc
        if resp.status_code >= 400:
            raise ProviderRequestError(
                f"stability returned HTTP {resp.status_code}", provider="stability",
                status_code=502, details={"upstream_body": resp.text[:1000]},
            )

        if accept == "application/json":
            body = resp.json()
            b64 = body.get("image")
        else:
            b64 = base64.b64encode(resp.content).decode()
            body = {}
        finish = resp.headers.get("finish-reason", body.get("finish_reason", ""))
        if finish == "CONTENT_FILTERED":
            raise TaskFailedError("stability content filtered", provider="stability")
        data_out = [ImageData(b64_json=b64, media_type=f"image/{request.output_format or 'png'}")]
        return UnifiedImageResponse(
            created=int(time.time()), data=data_out, model=request.model,
            provider=self.name, usage=ImageUsage(),
        )

    async def create_video_task(self, request: UnifiedVideoRequest) -> UnifiedVideoTask:
        # SVD requires an init image; the unified request may carry a data: URI.
        first_image = request.first_image()
        if not first_image:
            raise ProviderRequestError("stability SVD requires an input image", provider="stability",
                                       status_code=400)
        image_bytes, mime = _decode_image_input(first_image)
        task_id = f"svd-{uuid.uuid4().hex}"
        _VIDEO_TASKS[task_id] = {
            "model": request.model, "prompt": request.prompt() or "",
            "image_bytes": image_bytes, "mime": mime,
            "fps": request.fps, "seed": request.seed,
            "motion_bucket_id": request.extra.get("motion_bucket_id", 127),
            "cfg_scale": request.extra.get("cfg_scale", 1.0),
            "output_format": request.extra.get("output_format", "mp4"),
            "status": "pending",
            "created_at": int(time.time()),
        }
        return UnifiedVideoTask(task_id=task_id, provider=self.name, model=request.model, status="pending")

    async def get_video_task(self, task_id: str) -> UnifiedVideoTask:
        rec = _VIDEO_TASKS.get(task_id)
        if rec is None:
            raise ProviderRequestError(f"stability task {task_id} not found", provider="stability", status_code=404)
        if rec["status"] in ("succeeded", "failed"):
            return UnifiedVideoTask(
                task_id=task_id, provider=self.name, model=rec["model"], status=rec["status"],
                video_urls=rec.get("video_urls", []), error=rec.get("error"),
                created_at=rec["created_at"], completed_at=rec.get("completed_at"),
            )
        # Run the blocking SVD call now.
        rec["status"] = "running"
        data = {
            "seed": rec["seed"] or 0,
            "motion_bucket_id": rec["motion_bucket_id"],
            "cfg_scale": rec["cfg_scale"],
            "output_format": rec["output_format"],
            "model": rec["model"],
        }
        files = {"image": ("init.png", rec["image_bytes"], rec["mime"])}
        try:
            resp = await self._client.post(
                "/stable-video-generation/image-to-video",
                data=data, files=files, headers={"Accept": "application/json"},
            )
        except httpx.HTTPError as exc:
            rec["status"] = "failed"
            rec["error"] = str(exc)
            raise ProviderRequestError(f"stability video transport error: {exc}", provider="stability") from exc
        if resp.status_code >= 400:
            rec["status"] = "failed"
            rec["error"] = resp.text[:500]
            raise ProviderRequestError(f"stability video HTTP {resp.status_code}", provider="stability",
                                       status_code=502, details={"upstream_body": resp.text[:1000]})
        body = resp.json()
        b64 = body.get("video")
        if not b64:
            rec["status"] = "failed"
            rec["error"] = "no video in response"
            raise TaskFailedError("stability video returned no data", provider="stability")
        rec["status"] = "succeeded"
        rec["completed_at"] = int(time.time())
        rec["video_urls"] = ["data:video/mp4;base64," + b64]
        return UnifiedVideoTask(
            task_id=task_id, provider=self.name, model=rec["model"], status="succeeded",
            video_urls=rec["video_urls"], created_at=rec["created_at"], completed_at=rec["completed_at"],
        )


def _decode_image_input(image: str) -> tuple[bytes, str]:
    """Accept a data: URI or fetch an http(s) URL to raw bytes."""
    if image.startswith("data:"):
        header, _, b64 = image.partition(",")
        mime = header.split(";")[0].split(":")[1] if ":" in header else "image/png"
        return base64.b64decode(b64), mime
    import io
    # lazy to avoid a module-level httpx for a pure helper
    with httpx.Client(timeout=60) as c:
        r = c.get(image)
        r.raise_for_status()
        mime = r.headers.get("content-type", "image/png")
        return r.content, mime
