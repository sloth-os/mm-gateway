"""Shared ``google-genai`` image/video surface for the AI Studio and Vertex backends.

Both the ``google`` (AI Studio / Generative Language) and ``vertex`` (Vertex AI
Enterprise Agent Platform) backends talk to the *same* ``google-genai`` SDK —
Imagen for image, Veo (a long-running operation) for video. Only the
``genai.Client`` construction differs (AI Studio: ``api_key`` + the
``generativelanguage`` host; Vertex: ``vertexai=True`` + api-key or ADC + the
``aiplatform`` host). The request/response shapes the SDK exposes are identical
in both modes, so the generation logic is shared here and mixed into both
provider classes.

The mixin holds no state of its own; it relies on the subclass setting
``self._client`` and ``self._client_video`` (two SDK clients, one per modality
so an operator can pin image and video at different endpoints) and
``self.name`` in ``__init__``.
"""

from __future__ import annotations

import base64
import time
from typing import Any

from google.genai import types

from mm_gateway.core.exceptions import ProviderRequestError
from mm_gateway.providers._dimensions import aspect_ratio, video_resolution
from mm_gateway.schemas.image import (
    ImageData,
    ImageUsage,
    UnifiedImageRequest,
    UnifiedImageResponse,
)
from mm_gateway.schemas.video import UnifiedVideoRequest, UnifiedVideoTask


class GenaiImageVideoMixin:
    """Imagen image + Veo video generation over the ``google-genai`` SDK.

    Subclasses must set ``self._client`` / ``self._client_video`` (the image and
    video SDK clients) and ``self.name`` in ``__init__``, and call
    ``super().__init__(backend)``.
    """

    name: str  # set by the concrete provider subclass

    async def _generate_image(self, request: UnifiedImageRequest) -> UnifiedImageResponse:
        model = request.model
        try:
            if model.startswith("gemini"):
                return await self._generate_content_image(request)
            return await self._generate_imagen(request)
        except Exception as exc:
            raise ProviderRequestError(f"{self.name} image failed: {exc}", provider=self.name) from exc

    async def _generate_imagen(self, request: UnifiedImageRequest) -> UnifiedImageResponse:
        cfg_fields: dict[str, Any] = {}
        if request.n:
            cfg_fields["number_of_images"] = request.n
        if ratio := aspect_ratio(request):
            cfg_fields["aspect_ratio"] = ratio
        if request.negative_prompt:
            cfg_fields["negative_prompt"] = request.negative_prompt
        if request.seed is not None:
            cfg_fields["seed"] = request.seed
        if request.guidance_scale is not None:
            cfg_fields["guidance_scale"] = request.guidance_scale
        if request.output_format:
            cfg_fields["output_mime_type"] = f"image/{request.output_format}"
        cfg_fields.update(request.extra)
        config = types.GenerateImagesConfig(**cfg_fields)

        resp = await self._client.aio.models.generate_images(
            model=request.model, prompt=request.prompt() or "", config=config
        )
        data: list[ImageData] = []
        for gen in resp.generated_images or []:
            img = gen.image
            b64 = base64.b64encode(img.image_bytes).decode() if img.image_bytes else None
            data.append(ImageData(b64_json=b64, media_type=getattr(img, "mime_type", None) or "image/png"))
        return UnifiedImageResponse(
            created=int(time.time()), data=data, model=request.model,
            provider=self.name, usage=ImageUsage(),
        )

    async def _generate_content_image(self, request: UnifiedImageRequest) -> UnifiedImageResponse:
        image_config = types.ImageConfig()
        if ratio := aspect_ratio(request):
            image_config.aspect_ratio = ratio
        config = types.GenerateContentConfig(response_modalities=["IMAGE"], image_config=image_config)
        resp = await self._client.aio.models.generate_content(
            model=request.model, contents=request.prompt() or "", config=config
        )
        data: list[ImageData] = []
        for cand in resp.candidates or []:
            for part in (cand.content.parts or []):
                if part.inline_data and part.inline_data.data:
                    b64 = base64.b64encode(part.inline_data.data).decode()
                    data.append(ImageData(b64_json=b64, media_type=part.inline_data.mime_type or "image/png"))
        return UnifiedImageResponse(
            created=int(time.time()), data=data, model=request.model,
            provider=self.name, usage=ImageUsage(),
        )

    async def create_video_task(self, request: UnifiedVideoRequest) -> UnifiedVideoTask:
        source_fields: dict[str, Any] = {}
        if request.prompt():
            source_fields["prompt"] = request.prompt()
        if request.first_image():
            source_fields["image"] = types.Image(image_uri=request.first_image())
        source = types.GenerateVideosSource(**source_fields)

        cfg: dict[str, Any] = {}
        if ratio := aspect_ratio(request):
            cfg["aspect_ratio"] = ratio
        if resolution := video_resolution(request):
            cfg["resolution"] = resolution
        if request.duration is not None:
            cfg["duration_seconds"] = int(request.duration)
        if request.seed is not None:
            cfg["seed"] = request.seed
        if request.generate_audio is not None:
            cfg["generate_audio"] = request.generate_audio
        if request.negative_prompt:
            cfg["negative_prompt"] = request.negative_prompt
        if request.last_image():
            cfg["last_frame"] = types.Image(image_uri=request.last_image())
        cfg.update(request.extra)
        config = types.GenerateVideosConfig(**cfg)

        try:
            op = await self._client_video.aio.models.generate_videos(
                model=request.model, source=source, config=config
            )
        except Exception as exc:
            raise ProviderRequestError(f"{self.name} video create failed: {exc}", provider=self.name) from exc
        return UnifiedVideoTask(
            task_id=op.name or op.response and "unknown",
            provider=self.name, model=request.model, status="pending",
            raw={"operation_name": op.name},
        )

    async def get_video_task(self, task_id: str) -> UnifiedVideoTask:
        # Reconstruct a handle for the LRO so we can poll by name.
        op = types.Operation(name=task_id)  # type: ignore[arg-type]
        try:
            op = await self._client_video.aio.operations.get(op)
        except Exception as exc:
            raise ProviderRequestError(f"{self.name} video poll failed: {exc}", provider=self.name) from exc
        status: str = "running" if not op.done else "succeeded"
        task = UnifiedVideoTask(task_id=task_id, provider=self.name, model="", status=status)  # type: ignore[arg-type]
        if op.done and op.response and getattr(op.response, "generated_videos", None):
            urls: list[str] = []
            for v in op.response.generated_videos:
                if v.video and v.video.uri:
                    urls.append(v.video.uri)
            task.video_urls = urls
        elif op.done and getattr(op, "error", None):
            task.status = "failed"
            task.error = str(op.error)
        return task
