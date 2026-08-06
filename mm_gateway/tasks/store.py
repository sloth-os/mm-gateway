"""In-memory task store.

When a video request runs in async (non-blocking) mode the gateway must remember
which provider owns a task id so the poll endpoint can route correctly. The
Stability adapter also needs to persist its synthetic task state. This module
is a tiny pluggable interface; the default is process-local, but a real
deployment can drop in a Redis/DB-backed implementation.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskRecord:
    task_id: str
    provider: str
    model: str
    backend: str | None = None
    modality: str = "video"
    created_at: int = field(default_factory=lambda: int(time.time()))
    metadata: dict[str, Any] = field(default_factory=dict)


class TaskStore:
    """Process-local task store. Thread-safe enough for a single async event loop."""

    def __init__(self) -> None:
        self._records: dict[str, TaskRecord] = {}
        self._lock = asyncio.Lock()

    async def put(self, record: TaskRecord) -> None:
        async with self._lock:
            self._records[record.task_id] = record

    async def get(self, task_id: str) -> TaskRecord | None:
        async with self._lock:
            return self._records.get(task_id)

    async def delete(self, task_id: str) -> None:
        async with self._lock:
            self._records.pop(task_id, None)
