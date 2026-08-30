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
from mm_gateway.schemas.image import UnifiedImageRequest, UnifiedImageTask
from mm_gateway.schemas.music import UnifiedMusicRequest, UnifiedMusicTask
from mm_gateway.schemas.video import UnifiedVideoRequest, UnifiedVideoTask
from mm_gateway.services_selection import retry_across_backends
from mm_gateway.tasks.supervisor import AsyncTaskSupervisor

log = get_logger("service")

# Model id values (None or "auto") that trigger auto-routing + retry-across.
_AUTO = {None, "", "auto"}


def _is_auto(model: str | None) -> bool:
    """True when the caller left model selection to the gateway (auto-routing)."""
    return model is None or model.lower() in _AUTO


class ImageService:
    """Orchestrates an image request end-to-end, mirroring ``VideoService``.

    Image generation is task-based on every provider: the synchronous backends
    (OpenAI, Imagen, Stability, xAI, Volcengine, OpenRouter, FLUX) wrap their
    blocking call as a synthetic in-memory task (the first poll runs it), and
    DashScope Wanx is natively async. The flow is identical to video/music:
    resolve → create → optionally block → return a handle.
    """

    def __init__(self, registry: Registry, *, max_sync_wait: float, poll_interval: float,
                 sync_default: bool):
        self.registry = registry
        self.max_sync_wait = max_sync_wait
        self.poll_interval = poll_interval
        self.sync_default = sync_default
        self._supervisor = AsyncTaskSupervisor[UnifiedImageTask](
            "image", poll_interval=poll_interval,
        )

    async def create(
        self,
        request: UnifiedImageRequest,
        *,
        wait: bool | None = None,
        key: KeyConfig | None = None,
        tag: str | None = None,
        backend_name: str | None = None,
    ) -> UnifiedImageTask:
        want_sync = wait if wait is not None else self.sync_default
        model = request.model
        if _is_auto(model):
            # Auto mode: the registry ranks every fitting (backend, account,
            # model) candidate by limits *and* live health; we try them
            # best-first, retrying the next on a retryable failure so one flaky
            # backend (or one exhausted key) doesn't surface to the client.
            candidates = self.registry.enumerate_auto_candidates(
                request, key, modality="image", tag=tag, backend_name=backend_name,
            )
            if not candidates:
                # enumerate raises a 422 when nothing fits; reaching here means
                # the candidates were dropped after ranking (e.g. all in
                # cooldown) — still surface a validation error, not a 500.
                raise GatewayError(
                    "No configured model can serve this auto-routed request.",
                    code="validation_error", status_code=422,
                )
            routed_root = request

            async def attempt(prov, account_id, real_model, backend):
                if not isinstance(prov, ImageProvider):
                    raise GatewayError(
                        f"Backend '{backend}' does not support image generation.",
                        code="unsupported_feature", status_code=400,
                    )
                routed = routed_root.model_copy(
                    update={"model": real_model, "provider": backend}
                )
                with timed(backend, "image"):
                    try:
                        task = await prov.create_image_task(routed, sync=want_sync)
                    except GatewayError:
                        raise
                    except Exception as exc:  # noqa: BLE001
                        raise GatewayError(f"image create failed: {exc}", provider=backend,
                                           code="provider_error", status_code=502) from exc
                task.provider = backend
                # Sync-wait on the per-account provider that owns the task, so
                # polling uses the correct credential.
                if want_sync:
                    task = await self._await_or_timeout(prov, task)
                self._start_monitor(prov, backend, task)
                return task

            return await retry_across_backends(
                candidates=candidates, attempt=attempt, modality="image", key=key,
            )
        provider_obj, real_model, backend = self.registry.resolve(
            model, key, modality="image", tag=tag, backend_name=backend_name,
            request=request,
        )
        if not isinstance(provider_obj, ImageProvider):
            raise GatewayError(
                f"Backend '{backend}' does not support image generation.",
                code="unsupported_feature", status_code=400,
            )
        routed = request.model_copy(update={"model": real_model, "provider": backend})
        with timed(backend, "image"):
            try:
                task = await provider_obj.create_image_task(routed, sync=want_sync)
            except GatewayError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise GatewayError(f"image create failed: {exc}", provider=backend,
                                   code="provider_error", status_code=502) from exc
        # Stamp the owning backend so the poll route can route correctly.
        task.provider = backend
        if want_sync:
            task = await self._await_or_timeout(provider_obj, task)
        self._start_monitor(provider_obj, backend, task)
        return task

    async def get(self, task_id: str, backend_name: str | None = None) -> UnifiedImageTask:
        cached = self._supervisor.snapshot(task_id, provider=backend_name)
        if cached is not None:
            return cached
        provider_obj = self._find_provider_for(task_id, backend_name)
        return await self._poll(provider_obj, task_id)

    async def _poll(self, provider_obj: ImageProvider, task_id: str) -> UnifiedImageTask:
        with timed(provider_obj.name, "image"):
            try:
                task = await provider_obj.get_image_task(task_id)
            except GatewayError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise GatewayError(f"image poll failed: {exc}", provider=provider_obj.name,
                                   code="provider_error", status_code=502) from exc
        return task

    def _start_monitor(
        self, provider_obj: ImageProvider, backend: str, task: UnifiedImageTask,
    ) -> None:
        self._supervisor.start(
            provider=backend,
            task=task,
            poll=lambda: self._poll(provider_obj, task.task_id),
        )

    async def aclose(self) -> None:
        await self._supervisor.aclose()

    async def _await_or_timeout(self, provider: ImageProvider, task: UnifiedImageTask) -> UnifiedImageTask:
        deadline = time.monotonic() + self.max_sync_wait
        while time.monotonic() < deadline:
            task = await provider.get_image_task(task.task_id)
            if task.status in ("succeeded", "failed", "cancelled", "expired"):
                if task.status in ("failed", "cancelled", "expired") and not task.error:
                    task.error = task.status
                return task
            await asyncio.sleep(self.poll_interval)
        # Timed out waiting — return the latest non-terminal task so the client can keep polling.
        log.info("image_sync_wait_timeout", task_id=task.task_id, provider=provider.name)
        return task

    def _find_provider_for(self, task_id: str, backend_name: str | None) -> ImageProvider:
        if backend_name:
            return self.registry.image_provider(backend_name)
        providers = [p for p in self.registry.providers.values() if isinstance(p, ImageProvider)]
        if len(providers) == 1:
            return providers[0]
        return self.registry.image_provider(self.registry.settings.default_image_provider)


class VideoService:
    def __init__(self, registry: Registry, *, max_sync_wait: float, poll_interval: float,
                 sync_default: bool):
        self.registry = registry
        self.max_sync_wait = max_sync_wait
        self.poll_interval = poll_interval
        self.sync_default = sync_default
        self._supervisor = AsyncTaskSupervisor[UnifiedVideoTask](
            "video", poll_interval=poll_interval,
        )

    async def create(
        self,
        request: UnifiedVideoRequest,
        *,
        wait: bool | None = None,
        key: KeyConfig | None = None,
        tag: str | None = None,
        backend_name: str | None = None,
    ) -> UnifiedVideoTask:
        want_sync = wait if wait is not None else self.sync_default
        if _is_auto(request.model):
            # Auto mode: rank every fitting (backend, account, model) candidate
            # by limits *and* live health, then try them best-first, retrying on
            # a retryable failure so one flaky backend / exhausted key is hidden.
            candidates = self.registry.enumerate_auto_candidates(
                request, key, modality="video", tag=tag, backend_name=backend_name,
            )
            if not candidates:
                raise GatewayError(
                    "No configured model can serve this auto-routed request.",
                    code="validation_error", status_code=422,
                )
            routed_root = request

            async def attempt(prov, account_id, real_model, backend):
                if not isinstance(prov, VideoProvider):
                    raise GatewayError(
                        f"Backend '{backend}' does not support video generation.",
                        code="unsupported_feature", status_code=400,
                    )
                routed = routed_root.model_copy(
                    update={"model": real_model, "provider": backend}
                )
                with timed(backend, "video"):
                    try:
                        task = await prov.create_video_task(routed)
                    except GatewayError:
                        raise
                    except Exception as exc:  # noqa: BLE001
                        raise GatewayError(f"video create failed: {exc}", provider=backend,
                                           code="provider_error", status_code=502) from exc
                task.provider = backend
                # Sync-wait here, on the per-account provider that actually owns
                # the task, so polling uses the correct credential.
                if want_sync:
                    task = await self._await_or_timeout(prov, task)
                self._start_monitor(prov, backend, task)
                return task

            return await retry_across_backends(
                candidates=candidates, attempt=attempt, modality="video", key=key,
            )
        provider_obj, real_model, backend = self.registry.resolve(
            request.model, key, modality="video", tag=tag, backend_name=backend_name,
            request=request,
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
        if want_sync:
            task = await self._await_or_timeout(provider_obj, task)
        self._start_monitor(provider_obj, backend, task)
        return task

    async def get(self, task_id: str, backend_name: str | None = None) -> UnifiedVideoTask:
        cached = self._supervisor.snapshot(task_id, provider=backend_name)
        if cached is not None:
            return cached
        provider_obj = self._find_provider_for(task_id, backend_name)
        return await self._poll(provider_obj, task_id)

    async def _poll(self, provider_obj: VideoProvider, task_id: str) -> UnifiedVideoTask:
        with timed(provider_obj.name, "video"):
            try:
                task = await provider_obj.get_video_task(task_id)
            except GatewayError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise GatewayError(f"video poll failed: {exc}", provider=provider_obj.name,
                                   code="provider_error", status_code=502) from exc
        return task

    def _start_monitor(
        self, provider_obj: VideoProvider, backend: str, task: UnifiedVideoTask,
    ) -> None:
        self._supervisor.start(
            provider=backend,
            task=task,
            poll=lambda: self._poll(provider_obj, task.task_id),
        )

    async def aclose(self) -> None:
        await self._supervisor.aclose()

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
        self._supervisor = AsyncTaskSupervisor[UnifiedMusicTask](
            "music", poll_interval=poll_interval,
        )

    async def create(
        self,
        request: UnifiedMusicRequest,
        *,
        wait: bool | None = None,
        key: KeyConfig | None = None,
        tag: str | None = None,
        backend_name: str | None = None,
    ) -> UnifiedMusicTask:
        want_sync = wait if wait is not None else self.sync_default
        if _is_auto(request.model):
            # Auto mode: rank every fitting (backend, account, model) candidate
            # by limits *and* live health, then try them best-first, retrying on
            # a retryable failure so one flaky backend / exhausted key is hidden.
            candidates = self.registry.enumerate_auto_candidates(
                request, key, modality="music", tag=tag, backend_name=backend_name,
            )
            if not candidates:
                raise GatewayError(
                    "No configured model can serve this auto-routed request.",
                    code="validation_error", status_code=422,
                )
            routed_root = request

            async def attempt(prov, account_id, real_model, backend):
                if not isinstance(prov, MusicProvider):
                    raise GatewayError(
                        f"Backend '{backend}' does not support music generation.",
                        code="unsupported_feature", status_code=400,
                    )
                routed = routed_root.model_copy(
                    update={"model": real_model, "provider": backend}
                )
                with timed(backend, "music"):
                    try:
                        task = await prov.create_music_task(routed)
                    except GatewayError:
                        raise
                    except Exception as exc:  # noqa: BLE001
                        raise GatewayError(f"music create failed: {exc}", provider=backend,
                                           code="provider_error", status_code=502) from exc
                task.provider = backend
                # Sync-wait on the per-account provider that owns the task, so
                # polling uses the correct credential.
                if want_sync:
                    task = await self._await_or_timeout(prov, task)
                self._start_monitor(prov, backend, task)
                return task

            return await retry_across_backends(
                candidates=candidates, attempt=attempt, modality="music", key=key,
            )
        provider_obj, real_model, backend = self.registry.resolve(
            request.model, key, modality="music", tag=tag, backend_name=backend_name,
            request=request,
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
        if want_sync:
            task = await self._await_or_timeout(provider_obj, task)
        self._start_monitor(provider_obj, backend, task)
        return task

    async def get(self, task_id: str, backend_name: str | None = None) -> UnifiedMusicTask:
        cached = self._supervisor.snapshot(task_id, provider=backend_name)
        if cached is not None:
            return cached
        provider_obj = self._find_provider_for(task_id, backend_name)
        return await self._poll(provider_obj, task_id)

    async def _poll(self, provider_obj: MusicProvider, task_id: str) -> UnifiedMusicTask:
        with timed(provider_obj.name, "music"):
            try:
                task = await provider_obj.get_music_task(task_id)
            except GatewayError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise GatewayError(f"music poll failed: {exc}", provider=provider_obj.name,
                                   code="provider_error", status_code=502) from exc
        return task

    def _start_monitor(
        self, provider_obj: MusicProvider, backend: str, task: UnifiedMusicTask,
    ) -> None:
        self._supervisor.start(
            provider=backend,
            task=task,
            poll=lambda: self._poll(provider_obj, task.task_id),
        )

    async def aclose(self) -> None:
        await self._supervisor.aclose()

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
