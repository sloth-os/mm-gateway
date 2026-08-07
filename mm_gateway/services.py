"""Service layer — orchestrates a request end-to-end.

This is the seam between the HTTP front-end and the provider back-end. Routes
hand it a unified request (already translated from whichever front-end shape
arrived); the service resolves the backend via the registry (scoped to the
authenticated key), calls it, and returns a unified response. Keeping this
logic out of the route handler keeps the route testable and lets a CLI or
worker reuse the same path.
"""

from __future__ import annotations

import asyncio
import time

from mm_gateway.config import KeyConfig
from mm_gateway.core.base import ImageProvider, MusicProvider, VideoProvider
from mm_gateway.core.exceptions import GatewayError, TaskFailedError
from mm_gateway.observability.logging import get_logger
from mm_gateway.observability.metrics import timed
from mm_gateway.registry import Registry
from mm_gateway.schemas.image import UnifiedImageRequest, UnifiedImageResponse
from mm_gateway.schemas.music import UnifiedMusicRequest, UnifiedMusicTask
from mm_gateway.schemas.video import UnifiedVideoRequest, UnifiedVideoTask

log = get_logger("service")


class ImageService:
    def __init__(self, registry: Registry):
        self.registry = registry

    async def generate(
        self,
        request: UnifiedImageRequest,
        *,
        key: KeyConfig | None = None,
        tag: str | None = None,
        backend_name: str | None = None,
    ) -> UnifiedImageResponse:
        provider_obj, real_model, backend = self.registry.resolve(
            request.model, key, modality="image", tag=tag, backend_name=backend_name
        )
        if not isinstance(provider_obj, ImageProvider):
            raise GatewayError(
                f"Backend '{backend}' does not support image generation.",
                code="unsupported_feature", status_code=400,
            )
        routed = request.model_copy(update={"model": real_model, "provider": backend})
        with timed(backend, "image"):
            try:
                resp = await provider_obj.generate_image(routed)
            except GatewayError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise GatewayError(f"image generation failed: {exc}", provider=backend,
                                   code="provider_error", status_code=502) from exc
        if not resp.data:
            raise GatewayError("provider returned no images", provider=backend,
                               code="provider_error", status_code=502)
        return resp


class VideoService:
    def __init__(self, registry: Registry, *, max_sync_wait: float, poll_interval: float,
                 sync_default: bool):
        self.registry = registry
        self.max_sync_wait = max_sync_wait
        self.poll_interval = poll_interval
        self.sync_default = sync_default

    async def create(
        self,
        request: UnifiedVideoRequest,
        *,
        wait: bool | None = None,
        key: KeyConfig | None = None,
        tag: str | None = None,
        backend_name: str | None = None,
    ) -> UnifiedVideoTask:
        provider_obj, real_model, backend = self.registry.resolve(
            request.model, key, modality="video", tag=tag, backend_name=backend_name
        )
        if not isinstance(provider_obj, VideoProvider):
            raise GatewayError(
                f"Backend '{backend}' does not support video generation.",
                code="unsupported_feature", status_code=400,
            )
        routed = request.model_copy(update={"model": real_model, "provider": backend})
        with timed(backend, "video"):
            try:
                task = await provider_obj.create_video_task(routed)
            except GatewayError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise GatewayError(f"video create failed: {exc}", provider=backend,
                                   code="provider_error", status_code=502) from exc
        # Stamp the owning backend so the poll route can route correctly.
        task.provider = backend
        if (wait if wait is not None else self.sync_default):
            task = await self._await_or_timeout(provider_obj, task)
        return task

    async def get(self, task_id: str, backend_name: str | None = None) -> UnifiedVideoTask:
        provider_obj = self._find_provider_for(task_id, backend_name)
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

    def _find_provider_for(self, task_id: str, backend_name: str | None) -> VideoProvider:
        if backend_name:
            return self.registry.video_provider(backend_name)
        providers = [p for p in self.registry.providers.values() if isinstance(p, VideoProvider)]
        if len(providers) == 1:
            return providers[0]
        return self.registry.video_provider(self.registry.settings.default_video_provider)


class MusicService:
    """Orchestrates a music request end-to-end, mirroring ``VideoService``.

    Music generation is task-based on every provider (ElevenLabs' synchronous
    stream is wrapped as a synthetic in-memory task by its adapter), so the
    flow is identical: resolve → create → optionally block → return a handle.
    """

    def __init__(self, registry: Registry, *, max_sync_wait: float, poll_interval: float,
                 sync_default: bool):
        self.registry = registry
        self.max_sync_wait = max_sync_wait
        self.poll_interval = poll_interval
        self.sync_default = sync_default

    async def create(
        self,
        request: UnifiedMusicRequest,
        *,
        wait: bool | None = None,
        key: KeyConfig | None = None,
        tag: str | None = None,
        backend_name: str | None = None,
    ) -> UnifiedMusicTask:
        provider_obj, real_model, backend = self.registry.resolve(
            request.model, key, modality="music", tag=tag, backend_name=backend_name
        )
        if not isinstance(provider_obj, MusicProvider):
            raise GatewayError(
                f"Backend '{backend}' does not support music generation.",
                code="unsupported_feature", status_code=400,
            )
        routed = request.model_copy(update={"model": real_model, "provider": backend})
        with timed(backend, "music"):
            try:
                task = await provider_obj.create_music_task(routed)
            except GatewayError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise GatewayError(f"music create failed: {exc}", provider=backend,
                                   code="provider_error", status_code=502) from exc
        # Stamp the owning backend so the poll route can route correctly.
        task.provider = backend
        if (wait if wait is not None else self.sync_default):
            task = await self._await_or_timeout(provider_obj, task)
        return task

    async def get(self, task_id: str, backend_name: str | None = None) -> UnifiedMusicTask:
        provider_obj = self._find_provider_for(task_id, backend_name)
        with timed(provider_obj.name, "music"):
            try:
                task = await provider_obj.get_music_task(task_id)
            except GatewayError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise GatewayError(f"music poll failed: {exc}", provider=provider_obj.name,
                                   code="provider_error", status_code=502) from exc
        return task

    async def _await_or_timeout(self, provider: MusicProvider, task: UnifiedMusicTask) -> UnifiedMusicTask:
        deadline = time.monotonic() + self.max_sync_wait
        while time.monotonic() < deadline:
            task = await provider.get_music_task(task.task_id)
            if task.status in ("succeeded", "failed", "cancelled", "expired"):
                if task.status in ("failed", "cancelled", "expired") and not task.error:
                    task.error = task.status
                return task
            await asyncio.sleep(self.poll_interval)
        # Timed out waiting — return the latest non-terminal task so the client can keep polling.
        log.info("music_sync_wait_timeout", task_id=task.task_id, provider=provider.name)
        return task

    def _find_provider_for(self, task_id: str, backend_name: str | None) -> MusicProvider:
        if backend_name:
            return self.registry.music_provider(backend_name)
        providers = [p for p in self.registry.providers.values() if isinstance(p, MusicProvider)]
        if len(providers) == 1:
            return providers[0]
        return self.registry.music_provider(self.registry.settings.default_music_provider)
