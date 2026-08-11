"""In-memory task store.

The public REST APIs expose gateway-owned task ids, so the gateway must retain
the owning provider and its native task id for subsequent polls. This module is
a tiny pluggable interface; the default is process-local, but a deployment can
replace it with a Redis/DB-backed implementation.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class TaskRecord:
    # Public id exposed by the gateway. The provider id is deliberately kept
    # separate so public ids remain globally unique and provider-neutral.
    task_id: str
    provider: str
    model: str
    owner_key_id: str
    modality: Literal["image", "video", "music"]
    provider_task_id: str | None = None
    created_at: int = field(default_factory=lambda: int(time.time()))
    metadata: dict[str, Any] = field(default_factory=dict)
    idempotency_key: str | None = None
    request_fingerprint: str | None = None
    create_response: dict[str, Any] | None = None


class TaskStore:
    """Process-local task store. Thread-safe enough for a single async event loop."""

    def __init__(self) -> None:
        self._records: dict[str, TaskRecord] = {}
        self._idempotency: dict[tuple[str, str, str], str] = {}
        self._idempotency_locks: dict[tuple[str, str, str], asyncio.Lock] = {}
        self._lock = asyncio.Lock()

    async def put(self, record: TaskRecord) -> None:
        async with self._lock:
            self._records[record.task_id] = record
            if record.idempotency_key:
                scope = (
                    record.owner_key_id,
                    record.modality,
                    record.idempotency_key,
                )
                self._idempotency[scope] = record.task_id

    async def get(self, task_id: str) -> TaskRecord | None:
        async with self._lock:
            return self._records.get(task_id)

    async def get_by_idempotency(
        self,
        owner_key_id: str,
        modality: str,
        idempotency_key: str,
    ) -> TaskRecord | None:
        async with self._lock:
            task_id = self._idempotency.get(
                (owner_key_id, modality, idempotency_key)
            )
            return self._records.get(task_id) if task_id else None

    @asynccontextmanager
    async def idempotency_guard(
        self,
        owner_key_id: str,
        modality: str,
        idempotency_key: str | None,
    ) -> AsyncIterator[None]:
        """Serialize creates sharing an idempotency scope within this process."""
        if idempotency_key is None:
            yield
            return
        scope = (owner_key_id, modality, idempotency_key)
        async with self._lock:
            guard = self._idempotency_locks.setdefault(scope, asyncio.Lock())
        async with guard:
            yield

    async def delete(self, task_id: str) -> None:
        async with self._lock:
            record = self._records.pop(task_id, None)
            if record and record.idempotency_key:
                scope = (
                    record.owner_key_id,
                    record.modality,
                    record.idempotency_key,
                )
                self._idempotency.pop(scope, None)
                self._idempotency_locks.pop(scope, None)
