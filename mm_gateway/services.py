"""Service layer — orchestrates a request end-to-end.

This is the seam between the HTTP front-end and the provider back-end. Routes
hand it a unified request (already translated from whichever front-end shape
arrived); the service plans the route with auto mode (candidates, fit,
lifecycle, price, budget, ranking — :mod:`mm_gateway.auto_mode`), attempts the
candidates best-first, and hands the accepted task to the background monitor,
which settles its cost in the ledger when it finishes. Keeping this logic out
of the route handler keeps the route testable and lets a CLI or worker reuse
the same path.

Image, video and music share one flow (:class:`_GenerationService`); the
subclasses only name their provider capability and native create/get calls.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Generic, TypeVar

from mm_gateway.auto_mode import (
    Candidate,
    execute_plan,
    is_auto,
    plan_route,
    resolve_policy,
    settlement,
)
from mm_gateway.billing import CostLedger
from mm_gateway.config import KeyConfig
from mm_gateway.core.base import ImageProvider, MusicProvider, VideoProvider
from mm_gateway.core.exceptions import GatewayError
from mm_gateway.observability.logging import get_logger
from mm_gateway.observability.metrics import timed
from mm_gateway.registry import Registry
from mm_gateway.schemas.image import UnifiedImageRequest, UnifiedImageTask
from mm_gateway.schemas.music import UnifiedMusicRequest, UnifiedMusicTask
from mm_gateway.schemas.video import UnifiedVideoRequest, UnifiedVideoTask
from mm_gateway.tasks.supervisor import AsyncTaskSupervisor

log = get_logger("service")

TaskT = TypeVar("TaskT", UnifiedImageTask, UnifiedVideoTask, UnifiedMusicTask)

_TERMINAL = ("succeeded", "failed", "cancelled", "expired")


def _is_auto(model: str | None) -> bool:
    """True when the caller left model selection to the gateway (auto-routing)."""
    return is_auto(model)


class _GenerationService(Generic[TaskT]):
    """Plan → attempt → monitor → settle, shared by the three modalities."""

    modality: str = ""
    provider_type: type = object

    def __init__(self, registry: Registry, *, max_sync_wait: float, poll_interval: float,
                 sync_default: bool, ledger: CostLedger | None = None):
        self.registry = registry
        self.max_sync_wait = max_sync_wait
        self.poll_interval = poll_interval
        self.sync_default = sync_default
        # One ledger per gateway: create_app shares it across the modalities.
        self.ledger = ledger if ledger is not None else CostLedger()
        self._supervisor = AsyncTaskSupervisor[TaskT](self.modality, poll_interval=poll_interval)

    # -- modality hooks ------------------------------------------------------ #

    async def _create_task(self, provider: Any, request: Any, *, sync: bool) -> TaskT:
        raise NotImplementedError

    async def _get_task(self, provider: Any, task_id: str) -> TaskT:
        raise NotImplementedError

    # -- create -------------------------------------------------------------- #

    async def create(
        self,
        request: Any,
        *,
        wait: bool | None = None,
        key: KeyConfig | None = None,
        tag: str | None = None,
        backend_name: str | None = None,
        routing: Any = None,
    ) -> TaskT:
        """Route and create a task (docs/design/auto-mode.md#pipeline).

        ``routing`` is the public :class:`~mm_gateway.schemas.api.RoutingDirective`;
        ``tag`` is the legacy spelling of ``routing.profile``. A pinned model
        without ``routing.fallback`` keeps its single-attempt behaviour.
        """
        want_sync = wait if wait is not None else self.sync_default
        policy = resolve_policy(self.registry.settings, routing, tag=tag)
        plan = plan_route(self.registry, self.ledger, request, key=key, modality=self.modality,
                          policy=policy, backend_name=backend_name)

        async def attempt(cand: Candidate, accept) -> TaskT:
            prov = cand.provider
            if not isinstance(prov, self.provider_type):
                raise GatewayError(
                    f"Backend '{cand.backend}' does not support {self.modality} generation.",
                    code="unsupported_feature", status_code=400,
                )
            routed = request.model_copy(update={"model": cand.model, "provider": cand.backend})
            with timed(cand.backend, self.modality):
                try:
                    task = await self._create_task(prov, routed, sync=want_sync)
                except GatewayError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    raise GatewayError(f"{self.modality} create failed: {exc}", provider=cand.backend,
                                       code="provider_error", status_code=502) from exc
            # Stamp the owning backend so the poll route can route correctly.
            task.provider = cand.backend
            # Bind the ledger reservation before the task can finish and settle.
            accept(task)
            # Sync-wait on the per-account provider that owns the task, so
            # polling uses the correct credential.
            if want_sync:
                task = await self._await_or_timeout(prov, task)
            self._start_monitor(prov, cand.backend, task)
            return task

        task, info = await execute_plan(plan, attempt, ledger=self.ledger, key=key)
        task.routing = info
        return task

    # -- observe ------------------------------------------------------------- #

    async def get(self, task_id: str, backend_name: str | None = None) -> TaskT:
        cached = self._supervisor.snapshot(task_id, provider=backend_name)
        if cached is not None:
            return cached
        provider_obj = self._find_provider_for(task_id, backend_name)
        return await self._poll(provider_obj, task_id)

    async def _poll(self, provider_obj: Any, task_id: str) -> TaskT:
        with timed(provider_obj.name, self.modality):
            try:
                task = await self._get_task(provider_obj, task_id)
            except GatewayError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise GatewayError(f"{self.modality} poll failed: {exc}", provider=provider_obj.name,
                                   code="provider_error", status_code=502) from exc
        return task

    def _start_monitor(self, provider_obj: Any, backend: str, task: TaskT) -> None:
        self._supervisor.start(
            provider=backend,
            task=task,
            poll=lambda: self._poll(provider_obj, task.task_id),
            on_terminal=settlement(self.ledger, self.modality, backend),
        )

    async def aclose(self) -> None:
        await self._supervisor.aclose()

    async def _await_or_timeout(self, provider: Any, task: TaskT) -> TaskT:
        deadline = time.monotonic() + self.max_sync_wait
        while time.monotonic() < deadline:
            task = await self._get_task(provider, task.task_id)
            if task.status in _TERMINAL:
                if task.status in ("failed", "cancelled", "expired") and not task.error:
                    task.error = task.status
                return task
            await asyncio.sleep(self.poll_interval)
        # Timed out waiting — return the latest non-terminal task so the client can keep polling.
        log.info(f"{self.modality}_sync_wait_timeout", task_id=task.task_id, provider=provider.name)
        return task

    def _find_provider_for(self, task_id: str, backend_name: str | None) -> Any:
        lookup = getattr(self.registry, f"{self.modality}_provider")
        if backend_name:
            return lookup(backend_name)
        providers = [p for p in self.registry.providers.values() if isinstance(p, self.provider_type)]
        if len(providers) == 1:
            return providers[0]
        return lookup(getattr(self.registry.settings, f"default_{self.modality}_provider"))


class ImageService(_GenerationService[UnifiedImageTask]):
    """Image requests. Synchronous upstreams (OpenAI, Imagen, Stability, xAI,
    Volcengine, OpenRouter, FLUX) wrap their blocking call as a synthetic
    in-memory task (the first poll runs it); DashScope Wanx is natively async."""

    modality = "image"
    provider_type = ImageProvider

    async def create(self, request: UnifiedImageRequest, **kwargs: Any) -> UnifiedImageTask:
        return await super().create(request, **kwargs)

    async def _create_task(self, provider: ImageProvider, request: UnifiedImageRequest, *,
                           sync: bool) -> UnifiedImageTask:
        return await provider.create_image_task(request, sync=sync)

    async def _get_task(self, provider: ImageProvider, task_id: str) -> UnifiedImageTask:
        return await provider.get_image_task(task_id)


class VideoService(_GenerationService[UnifiedVideoTask]):
    modality = "video"
    provider_type = VideoProvider

    async def create(self, request: UnifiedVideoRequest, **kwargs: Any) -> UnifiedVideoTask:
        return await super().create(request, **kwargs)

    async def _create_task(self, provider: VideoProvider, request: UnifiedVideoRequest, *,
                           sync: bool) -> UnifiedVideoTask:
        return await provider.create_video_task(request)

    async def _get_task(self, provider: VideoProvider, task_id: str) -> UnifiedVideoTask:
        return await provider.get_video_task(task_id)


class MusicService(_GenerationService[UnifiedMusicTask]):
    """Music requests. ElevenLabs' synchronous stream is wrapped as a synthetic
    in-memory task by its adapter, so the flow is identical to video."""

    modality = "music"
    provider_type = MusicProvider

    async def create(self, request: UnifiedMusicRequest, **kwargs: Any) -> UnifiedMusicTask:
        return await super().create(request, **kwargs)

    async def _create_task(self, provider: MusicProvider, request: UnifiedMusicRequest, *,
                           sync: bool) -> UnifiedMusicTask:
        return await provider.create_music_task(request)

    async def _get_task(self, provider: MusicProvider, task_id: str) -> UnifiedMusicTask:
        return await provider.get_music_task(task_id)


__all__ = ["ImageService", "MusicService", "VideoService"]
