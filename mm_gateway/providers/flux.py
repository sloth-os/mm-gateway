"""FLUX.2 provider (runapi-flux-2) — text-to-image and remix (image-to-image).

The runapi SDK is synchronous and blocking, so calls are offloaded to a thread
via ``asyncio.to_thread`` to keep the gateway's event loop responsive. FLUX.2
has no video model, so this provider only implements ``ImageProvider``.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from runapi.flux_2 import Flux2Client

from mm_gateway.core.base import ImageProvider
from mm_gateway.core.exceptions import ProviderNotConfiguredError, ProviderRequestError, TaskFailedError
from mm_gateway.observability.logging import get_logger
from mm_gateway.schemas.image import ImageData, ImageUsage, UnifiedImageRequest, UnifiedImageResponse

log = get_logger("provider.flux")

_T2I = ("flux-2-flex-text-to-image", "flux-2-max-text-to-image", "flux-2-pro-text-to-image")


class FluxProvider(ImageProvider):
    name = "flux"
    image_models = list(_T2I) + (
        ["flux-2-flex-remix-image", "flux-2-max-remix-image", "flux-2-pro-remix-image"]
    )

    def __init__(self, backend):
        super().__init__(backend)
        if not backend.api_key:
            raise ProviderNotConfiguredError("flux")
        kwargs: dict[str, Any] = {"api_key": backend.api_key}
        if backend.base_url:
            # runapi uses a global configure; set via core if a base url override is provided.
            try:
                from runapi.core import configure
                configure(base_url=backend.base_url)
            except Exception:  # noqa: BLE001
                pass
        self._client = Flux2Client(**kwargs)

    async def generate_image(self, request: UnifiedImageRequest) -> UnifiedImageResponse:
        is_remix = request.model.endswith("remix-image") or bool(request.input_images)
        params: dict[str, Any] = {"model": request.model, "prompt": request.prompt}
        if request.aspect_ratio:
            params["aspect_ratio"] = request.aspect_ratio
        if request.resolution:
            params["output_resolution"] = request.resolution
        if is_remix and request.input_images:
            params["source_image_urls"] = [i.url for i in request.input_images if i.url]
        params.update(request.extra)

        try:
            if is_remix:
                task = await asyncio.to_thread(self._client.remix_image.create, **params)
            else:
                task = await asyncio.to_thread(self._client.text_to_image.create, **params)
        except Exception as exc:  # noqa: BLE001
            raise ProviderRequestError(f"flux create failed: {exc}", provider="flux") from exc

        task_id = getattr(task, "id", None)
        if not task_id:
            raise ProviderRequestError("flux create returned no task id", provider="flux")

        # Poll to completion (the runapi sync client polls internally with .run(),
        # but we manage our own loop so the gateway stays responsive and observable).
        resource = self._client.remix_image if is_remix else self._client.text_to_image
        for _ in range(150):  # ~5 min at 2s
            await asyncio.sleep(2)
            try:
                status = await asyncio.to_thread(resource.get, task_id)
            except Exception as exc:  # noqa: BLE001
                raise ProviderRequestError(f"flux poll failed: {exc}", provider="flux") from exc
            st = getattr(status, "status", "") or ""
            if st == "completed":
                images = getattr(status, "images", None) or []
                data = [ImageData(url=getattr(im, "url", None)) for im in images]
                return UnifiedImageResponse(
                    created=int(time.time()), data=data, model=request.model,
                    provider=self.name, usage=ImageUsage(),
                )
            if st == "failed":
                err = getattr(status, "error", None) or "unknown"
                raise TaskFailedError(f"flux task failed: {err}", provider="flux")
        raise ProviderRequestError(f"flux task {task_id} timed out", provider="flux", status_code=504)
