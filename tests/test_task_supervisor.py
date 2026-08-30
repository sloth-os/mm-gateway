"""Unit tests for non-blocking background provider-task supervision."""

from __future__ import annotations

import asyncio

from mm_gateway.observability.metrics import render_prometheus
from mm_gateway.schemas.music import UnifiedMusicTask
from mm_gateway.tasks.supervisor import AsyncTaskSupervisor


async def test_snapshot_returns_while_provider_poll_is_still_running() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def slow_poll() -> UnifiedMusicTask:
        entered.set()
        await release.wait()
        return UnifiedMusicTask(
            task_id="provider-1", provider="slow", model="music-1",
            status="succeeded", audio_b64="AAAA",
        )

    supervisor = AsyncTaskSupervisor[UnifiedMusicTask]("music", poll_interval=0.001)
    supervisor.start(
        provider="slow",
        task=UnifiedMusicTask(
            task_id="provider-1", provider="slow", model="music-1", status="pending",
        ),
        poll=slow_poll,
    )
    await asyncio.wait_for(entered.wait(), timeout=0.1)

    # snapshot() is observation-only: the provider call above is deliberately
    # blocked, yet the latest state is available synchronously.
    current = supervisor.snapshot("provider-1", provider="slow")
    assert current is not None
    assert current.status == "pending"

    release.set()
    completed = await supervisor.wait_for_terminal(
        "provider-1", provider="slow", timeout=0.1,
    )
    assert completed is not None
    assert completed.status == "succeeded"
    assert completed.audio_b64 == "AAAA"
    metrics = render_prometheus()
    assert 'gateway_async_tasks_submitted_total{modality="music",provider="slow"}' in metrics
    assert ('gateway_async_tasks_finished_total{modality="music",provider="slow",'
            'status="succeeded"}') in metrics
    assert ('gateway_async_task_duration_seconds_count{modality="music",provider="slow",'
            'status="succeeded"}') in metrics
    await supervisor.aclose()


async def test_three_poll_errors_become_a_sanitized_failed_task() -> None:
    calls = 0

    async def broken_poll() -> UnifiedMusicTask:
        nonlocal calls
        calls += 1
        raise RuntimeError("private provider credential detail")

    supervisor = AsyncTaskSupervisor[UnifiedMusicTask]("music", poll_interval=0)
    supervisor.start(
        provider="broken",
        task=UnifiedMusicTask(
            task_id="provider-2", provider="broken", model="music-2", status="running",
        ),
        poll=broken_poll,
    )
    failed = await supervisor.wait_for_terminal(
        "provider-2", provider="broken", timeout=0.1,
    )

    assert failed is not None
    assert failed.status == "failed"
    assert failed.error == "Provider status polling failed repeatedly."
    assert "credential" not in failed.error
    assert calls == 3
    await supervisor.aclose()


async def test_shutdown_cancels_and_joins_a_live_monitor() -> None:
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def never_finishes() -> UnifiedMusicTask:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        raise AssertionError("unreachable")

    supervisor = AsyncTaskSupervisor[UnifiedMusicTask]("music", poll_interval=0.001)
    supervisor.start(
        provider="slow",
        task=UnifiedMusicTask(
            task_id="provider-3", provider="slow", model="music-3", status="running",
        ),
        poll=never_finishes,
    )
    await asyncio.wait_for(entered.wait(), timeout=0.1)
    await supervisor.aclose()
    await asyncio.wait_for(cancelled.wait(), timeout=0.1)
