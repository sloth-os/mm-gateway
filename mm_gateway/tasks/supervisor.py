"""Background supervision for provider task polling.

The public REST API is asynchronous even when an upstream provider is not.
Services register each accepted provider task here, and the supervisor owns all
subsequent provider polls. HTTP/MCP reads only take an immutable snapshot from
the supervisor, so a slow synchronous generation call or a stalled upstream
status endpoint can never hold a task-status request open.

The supervisor is intentionally provider-neutral and process-local, matching
the default :mod:`mm_gateway.tasks.store`. A durable deployment can replace
both with a queue/worker plus persistent task state without changing routes.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from mm_gateway.observability.logging import get_logger
from mm_gateway.observability.metrics import (
    record_async_task_finished,
    record_async_task_poll_error,
    record_async_task_submitted,
)
from mm_gateway.schemas.image import UnifiedImageTask
from mm_gateway.schemas.music import UnifiedMusicTask
from mm_gateway.schemas.video import UnifiedVideoTask

log = get_logger("task.supervisor")

TaskT = TypeVar(
    "TaskT",
    UnifiedImageTask,
    UnifiedVideoTask,
    UnifiedMusicTask,
)
TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled", "expired"})
_MAX_CONSECUTIVE_POLL_ERRORS = 3


@dataclass
class _Entry(Generic[TaskT]):
    snapshot: TaskT
    poll: Callable[[], Awaitable[TaskT]]
    monitor: asyncio.Task[None] | None = None
    consecutive_errors: int = 0


class AsyncTaskSupervisor(Generic[TaskT]):
    """Own provider polling and expose non-blocking, latest-state snapshots."""

    def __init__(self, modality: str, *, poll_interval: float) -> None:
        self.modality = modality
        self.poll_interval = max(0.0, poll_interval)
        self._entries: dict[tuple[str, str], _Entry[TaskT]] = {}
        self._closed = False

    def start(
        self,
        *,
        provider: str,
        task: TaskT,
        poll: Callable[[], Awaitable[TaskT]],
    ) -> None:
        """Cache ``task`` and start exactly one monitor for its provider id."""
        if self._closed:
            raise RuntimeError("task supervisor is closed")
        key = (provider, task.task_id)
        existing = self._entries.get(key)
        if existing is not None:
            # Idempotent registration is useful for replay-safe service code;
            # never replace a live monitor with a duplicate poller.
            return
        entry = _Entry(snapshot=task.model_copy(deep=True), poll=poll)
        self._entries[key] = entry
        record_async_task_submitted(provider, self.modality)
        log.info(
            "async_task_submitted",
            provider=provider,
            modality=self.modality,
            task_id=task.task_id,
            status=task.status,
        )
        if task.status in TERMINAL_STATUSES:
            self._finish(provider, task.task_id, task.status, time.monotonic())
            return
        entry.monitor = asyncio.create_task(
            self._monitor(key, entry),
            name=f"mm-gateway:{self.modality}:{provider}:{task.task_id}",
        )

    def snapshot(self, task_id: str, *, provider: str | None = None) -> TaskT | None:
        """Return a detached latest snapshot without awaiting provider I/O."""
        entry: _Entry[TaskT] | None = None
        if provider is not None:
            entry = self._entries.get((provider, task_id))
        else:
            matches = [candidate for (name, tid), candidate in self._entries.items()
                       if tid == task_id]
            if len(matches) == 1:
                entry = matches[0]
        return entry.snapshot.model_copy(deep=True) if entry is not None else None

    async def wait_for_terminal(
        self,
        task_id: str,
        *,
        provider: str,
        timeout: float,
    ) -> TaskT | None:
        """Wait on cached state only; the monitor remains the sole poll owner."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            task = self.snapshot(task_id, provider=provider)
            if task is None or task.status in TERMINAL_STATUSES:
                return task
            await asyncio.sleep(self.poll_interval)
        return self.snapshot(task_id, provider=provider)

    async def aclose(self) -> None:
        """Cancel and join live monitors during graceful gateway shutdown."""
        self._closed = True
        monitors = [entry.monitor for entry in self._entries.values()
                    if entry.monitor is not None and not entry.monitor.done()]
        for monitor in monitors:
            monitor.cancel()
        if monitors:
            await asyncio.gather(*monitors, return_exceptions=True)

    async def _monitor(
        self,
        key: tuple[str, str],
        entry: _Entry[TaskT],
    ) -> None:
        provider, task_id = key
        started = time.monotonic()
        previous = entry.snapshot.status
        log.info(
            "async_task_monitor_started",
            provider=provider,
            modality=self.modality,
            task_id=task_id,
            status=previous,
        )
        try:
            while True:
                try:
                    latest = await entry.poll()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - background boundary
                    entry.consecutive_errors += 1
                    record_async_task_poll_error(provider, self.modality)
                    log.warning(
                        "async_task_poll_failed",
                        provider=provider,
                        modality=self.modality,
                        task_id=task_id,
                        consecutive_errors=entry.consecutive_errors,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    if entry.consecutive_errors >= _MAX_CONSECUTIVE_POLL_ERRORS:
                        entry.snapshot = entry.snapshot.model_copy(update={
                            "status": "failed",
                            "error": "Provider status polling failed repeatedly.",
                            "completed_at": int(time.time()),
                        }, deep=True)
                        self._finish(provider, task_id, entry.snapshot.status, started)
                        return
                else:
                    entry.consecutive_errors = 0
                    entry.snapshot = latest.model_copy(deep=True)
                    if latest.status != previous:
                        log.info(
                            "async_task_state_changed",
                            provider=provider,
                            modality=self.modality,
                            task_id=task_id,
                            previous_status=previous,
                            status=latest.status,
                        )
                        previous = latest.status
                    if latest.status in TERMINAL_STATUSES:
                        self._finish(provider, task_id, latest.status, started)
                        return
                await asyncio.sleep(self.poll_interval)
        except asyncio.CancelledError:
            duration = time.monotonic() - started
            record_async_task_finished(provider, self.modality, "cancelled", duration)
            log.info(
                "async_task_monitor_cancelled",
                provider=provider,
                modality=self.modality,
                task_id=task_id,
                duration_s=round(duration, 3),
            )
            raise

    def _finish(self, provider: str, task_id: str, status: str, started: float) -> None:
        duration = time.monotonic() - started
        record_async_task_finished(provider, self.modality, status, duration)
        log.info(
            "async_task_monitor_finished",
            provider=provider,
            modality=self.modality,
            task_id=task_id,
            status=status,
            duration_s=round(duration, 3),
        )


__all__ = ["AsyncTaskSupervisor", "TERMINAL_STATUSES"]
