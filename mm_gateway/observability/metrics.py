"""Lightweight in-process metrics.

A small counter/histogram registry with a Prometheus-text exposition endpoint.
Kept dependency-free so the gateway has zero external metrics infra to adopt;
swap for prometheus_client if you need pushgateway/wasm integration.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class _MetricStore:
    counters: dict[str, dict[tuple, float]] = field(default_factory=lambda: defaultdict(dict))
    histograms: dict[str, dict[tuple, list[float]]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(list)))
    _lock = threading.Lock()

    def inc_counter(self, name: str, value: float = 1.0, **labels: str) -> None:
        key = tuple(sorted(labels.items()))
        with self._lock:
            d = self.counters[name]
            d[key] = d.get(key, 0.0) + value

    def observe(self, name: str, value: float, **labels: str) -> None:
        key = tuple(sorted(labels.items()))
        with self._lock:
            self.histograms[name][key].append(value)

    def render_prometheus(self) -> str:
        lines: list[str] = []
        with self._lock:
            for name, d in self.counters.items():
                for key, val in d.items():
                    label_str = ",".join(f'{k}="{v}"' for k, v in key)
                    lines.append(f'{name}{{{label_str}}} {val}')
            for name, h in self.histograms.items():
                for key, samples in h.items():
                    label_str = ",".join(f'{k}="{v}"' for k, v in key)
                    if not samples:
                        continue
                    n = len(samples)
                    total = sum(samples)
                    lines.append(f'{name}_count{{{label_str}}} {n}')
                    lines.append(f'{name}_sum{{{label_str}}} {total}')
        return "\n".join(lines) + ("\n" if lines else "")


STORE = _MetricStore()


def record_request(provider: str, modality: str, status: str, duration_s: float) -> None:
    STORE.inc_counter("gateway_requests_total", provider=provider, modality=modality, status=status)
    STORE.observe("gateway_request_duration_seconds", duration_s, provider=provider, modality=modality)


def record_async_task_submitted(provider: str, modality: str) -> None:
    """Count a provider task handed to the background polling supervisor."""
    STORE.inc_counter(
        "gateway_async_tasks_submitted_total", provider=provider, modality=modality,
    )


def record_async_task_poll_error(provider: str, modality: str) -> None:
    """Count a provider poll that failed before yielding a task snapshot."""
    STORE.inc_counter(
        "gateway_async_task_poll_errors_total", provider=provider, modality=modality,
    )


def record_async_task_finished(
    provider: str, modality: str, status: str, duration_s: float,
) -> None:
    """Record the terminal outcome and monitor lifetime of an async task."""
    STORE.inc_counter(
        "gateway_async_tasks_finished_total",
        provider=provider,
        modality=modality,
        status=status,
    )
    STORE.observe(
        "gateway_async_task_duration_seconds",
        duration_s,
        provider=provider,
        modality=modality,
        status=status,
    )


def render_prometheus() -> str:
    return STORE.render_prometheus()


class timed:
    """Context manager that records wall-clock duration for a provider call."""

    def __init__(self, provider: str, modality: str):
        self.provider = provider
        self.modality = modality
        self.status = "ok"
        self._start = 0.0

    def __enter__(self) -> "timed":
        self._start = time.monotonic()
        return self

    def __exit__(self, exc_type, exc, _tb) -> None:
        duration = time.monotonic() - self._start
        if exc_type is not None:
            self.status = "error"
        record_request(self.provider, self.modality, self.status, duration)
