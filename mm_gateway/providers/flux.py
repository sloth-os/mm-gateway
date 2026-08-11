"""FLUX.2 provider (runapi-flux-2) — text-to-image and remix (image-to-image).

The runapi SDK is synchronous and blocking, so calls are offloaded to a thread
via ``asyncio.to_thread`` to keep the gateway's event loop responsive. FLUX.2
has no video model, so this provider only implements ``ImageProvider``.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from runapi.core.http_client import HttpClient
from runapi.core.options import ClientOptions
from runapi.flux_2 import Flux2Client

from mm_gateway.core.base import ImageProvider
from mm_gateway.core.exceptions import (
    ProviderNotConfiguredError,
    ProviderRequestError,
    TaskFailedError,
)
from mm_gateway.observability.httplog import backend_sync_event_hooks
from mm_gateway.observability.logging import get_logger
from mm_gateway.providers._dimensions import aspect_ratio, image_resolution
from mm_gateway.providers._sync_image import SyncImageTaskMixin
from mm_gateway.schemas.image import (
    ImageData,
    ImageUsage,
    UnifiedImageRequest,
    UnifiedImageResponse,
)

log = get_logger("provider.flux")

_T2I = ("flux-2-flex-text-to-image", "flux-2-max-text-to-image", "flux-2-pro-text-to-image")


class _LoggedHttpClient(HttpClient):
    """runapi ``HttpClient`` subclass that registers backend logging event
    hooks on the SDK's internal ``httpx.Client`` instances.

    The runapi SDK builds its ``httpx.Client`` instances inside
    ``HttpClient.__init__`` (one for API calls, one for pre-authorized
    uploads). ``httpx`` exposes
    ``client.event_hooks`` as a mutable mapping, so rather than rebuild the
    clients we just extend the request/response hook lists after the parent
    constructor has wired up auth headers, base url, timeout and retries. The
    SDK's retry/multipart logic in ``request()`` / ``upload()`` is left
    untouched.
    """

    def __init__(self, options: ClientOptions, *, transport=None) -> None:
        super().__init__(options, transport=transport)
        hooks = backend_sync_event_hooks()
        self._client.event_hooks["request"].extend(hooks["request"])
        self._client.event_hooks["response"].extend(hooks["response"])
        self._upload_client.event_hooks["request"].extend(hooks["request"])
        self._upload_client.event_hooks["response"].extend(hooks["response"])


class FluxProvider(SyncImageTaskMixin, ImageProvider):
    name = "flux"
    image_models = list(_T2I) + (
        ["flux-2-flex-remix-image", "flux-2-max-remix-image", "flux-2-pro-remix-image"]
    )

    def __init__(self, backend):
        super().__init__(backend)
        if not backend.api_key:
            raise ProviderNotConfiguredError("flux")
        if backend.base_url:
            # runapi uses a global configure; set via core if a base url override is provided.
            try:
                from runapi.core import configure
                configure(base_url=backend.base_url)
            except Exception:  # noqa: BLE001, S110
                pass
        # Inject an HttpClient whose httpx.Clients carry backend logging
        # hooks. ``ClientOptions`` resolves its own base_url fallback from the
        # global config set above, so passing base_url only when the operator
        # pinned one mirrors the SDK's own defaulting.
        options = ClientOptions(api_key=backend.api_key, base_url=backend.base_url or None)
        self._client = Flux2Client(api_key=backend.api_key, http_client=_LoggedHttpClient(options))

    async def _generate_image(self, request: UnifiedImageRequest) -> UnifiedImageResponse:
        is_remix = request.model.endswith("remix-image") or bool(request.input_images())
        params: dict[str, Any] = {"model": request.model, "prompt": request.prompt() or ""}
        if ratio := aspect_ratio(request):
            params["aspect_ratio"] = ratio
        if resolution := image_resolution(request):
            params["output_resolution"] = resolution
        if is_remix and request.input_images():
            params["source_image_urls"] = [i.url for i in request.input_images() if i.url]
        params.update(request.extra)

        try:
            if is_remix:
                task = await asyncio.to_thread(self._client.remix_image.create, **params)
            else:
                task = await asyncio.to_thread(self._client.text_to_image.create, **params)
        except Exception as exc:
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
            except Exception as exc:
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
