"""Service layer — orchestrates a request end-to-end.

This is the seam between the HTTP front-end and the provider back-end. Routes
hand it a unified request (already translated from whichever front-end shape
arrived); the service resolves the provider via the registry, calls it, and
returns a unified response. Keeping this logic out of the route handler keeps
the route testable (just translator + service) and lets a CLI or worker reuse
the same path.
"""

from __future__ import annotations

import asyncio
import time

from mm_gateway.core.base import ImageProvider, VideoProvider
from mm_gateway.core.exceptions import GatewayError, TaskFailedError
from mm_gateway.observability.logging import get_logger
from mm_gateway.observability.metrics import timed
from mm_gateway.registry import Registry
from mm_gateway.schemas.image import UnifiedImageRequest, UnifiedImageResponse
from mm_gateway.schemas.video import UnifiedVideoRequest, UnifiedVideoTask

log = get_logger("service")


class ImageService:
    def __init__(self, registry: Registry):
        self.registry = registry

    async def generate(self, request: UnifiedImageRequest) -> UnifiedImageResponse:
        provider_obj, real_model = self.registry.resolve(
            request.model, request.provider, modality="image"
        )
        if not isinstance(provider_obj, ImageProvider):
            raise GatewayError(
                f"Provider '{provider_obj.name}' does not support image generation.",
                code="unsupported_feature", status_code=400,
            )
        routed = request.model_copy(update={"model": real_model, "provider": provider_obj.name})
        with timed(provider_obj.name, "image"):
            try:
                resp = await provider_obj.generate_image(routed)
            except GatewayError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise GatewayError(f"image generation failed: {exc}", provider=provider_obj.name,
                                   code="provider_error", status_code=502) from exc
        if not resp.data:
            raise GatewayError("provider returned no images", provider=provider_obj.name,
                               code="provider_error", status_code=502)
        return resp


class VideoService:
    def __init__(self, registry: Registry, *, max_sync_wait: float, poll_interval: float,
                 sync_default: bool):
        self.registry = registry
        self.max_sync_wait = max_sync_wait
        self.poll_interval = poll_interval
        self.sync_default = sync_default

    async def create(self, request: UnifiedVideoRequest, *, wait: bool | None = None) -> UnifiedVideoTask:
        provider_obj, real_model = self.registry.resolve(
            request.model, request.provider, modality="video"
        )
        if not isinstance(provider_obj, VideoProvider):
            raise GatewayError(
                f"Provider '{provider_obj.name}' does not support video generation.",
                code="unsupported_feature", status_code=400,
            )
        routed = request.model_copy(update={"model": real_model, "provider": provider_obj.name})
        with timed(provider_obj.name, "video"):
            try:
                task = await provider_obj.create_video_task(routed)
            except GatewayError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise GatewayError(f"video create failed: {exc}", provider=provider_obj.name,
                                   code="provider_error", status_code=502) from exc
        if (wait if wait is not None else self.sync_default):
            task = await self._await_or_timeout(provider_obj, task)
        return task

    async def get(self, task_id: str, provider_name: str | None = None) -> UnifiedVideoTask:
        provider_obj = self._find_provider_for(task_id, provider_name)
        with timed(provider_obj.name, "video"):
            try:
                task = await provider_obj.get_video_task(task_id)
            except GatewayError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise GatewayError(f"video poll failed: {exc}", provider=provider_obj.name,
                                   code="provider_error", status_code=502) from exc
        return task

    async def _await_or_timeout(self, provider: VideoProvider, task: UnifiedVideoTask) -> UnifiedVideoTask:
        deadline = time.monotonic() + self.max_sync_wait
        while time.monotonic() < deadline:
            task = await provider.get_video_task(task.task_id)
            if task.status in ("succeeded", "failed", "cancelled", "expired"):
                if task.status in ("failed", "cancelled", "expired") and not task.error:
                    task.error = task.status
                return task
            await asyncio.sleep(self.poll_interval)
        # Timed out waiting — return the latest non-terminal task so the client can keep polling.
        log.info("video_sync_wait_timeout", task_id=task.task_id, provider=provider.name)
        return task

    def _find_provider_for(self, task_id: str, provider_name: str | None) -> VideoProvider:
        if provider_name:
            return self.registry.video_provider(provider_name)
        # Without a hint, try each configured video provider. Task ids are
        # opaque, so a real deployment would record the provider when the task
        # is created (see tasks/store.py). For the single-process case we accept
        # the provider explicitly via the ?provider= query or the X-Provider header.
        providers = [p for p in self.registry.providers.values() if isinstance(p, VideoProvider)]
        if len(providers) == 1:
            return providers[0]
        # Fall back to the default video provider if exactly one is configured.
        return self.registry.video_provider(self.registry.settings.default_video_provider)
