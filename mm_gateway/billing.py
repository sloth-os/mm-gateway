"""The cost ledger: spend, reservations and budgets (docs/design/auto-mode.md#budgets-and-the-ledger).

Auto mode estimates a task's cost before calling a provider and **holds** that
amount against every applicable budget; the hold becomes a **reservation** once
the provider accepts the task, and the background monitor **settles** it when
the task finishes (succeeded: actual cost — provider-reported, else the estimate
— moves to spent; failed/cancelled/expired: released at no cost). Holding before
the call makes concurrent creates safe: ten parallel requests cannot all see the
same remaining budget.

Budgets: a key's operator budget (``keys[].budget``, per UTC day, month or
total) and client scopes (``routing.budget.scope``, lifetime totals, capped by
the client's ``limit_usd`` and the operator's ``scopes_limit_usd``).

Like the bundled task store this ledger is process-local; a multi-instance
deployment injects a shared implementation via ``create_app(..., ledger=...)``.
"""

from __future__ import annotations

import itertools
import threading
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from mm_gateway.config import KeyConfig
from mm_gateway.observability.logging import get_logger
from mm_gateway.observability.metrics import STORE as METRICS, prometheus_labels

log = get_logger("ledger")

_EPS = 1e-9


def utc_now() -> datetime:
    return datetime.now(UTC)


def period_window(kind: str, now: datetime) -> tuple[str, datetime | None, datetime | None]:
    """``(period id, start, end)`` of the budget period containing ``now`` (UTC)."""
    if kind == "day":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return start.strftime("%Y-%m-%d"), start, start + timedelta(days=1)
    if kind == "month":
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end = start.replace(year=start.year + 1, month=1) if start.month == 12 \
            else start.replace(month=start.month + 1)
        return start.strftime("%Y-%m"), start, end
    return "total", None, None


@dataclass
class _Hold:
    key_id: str
    period: str
    scope: str | None
    model: str
    modality: str
    amount: float
    priced: bool = True
    task: tuple[str, str, str] | None = None


@dataclass
class BudgetRejection:
    """Why a hold was refused (the binding budget and its state)."""

    budget: str  # "key" | "scope"
    scope: str | None
    limit_usd: float
    spent_usd: float
    reserved_usd: float
    estimate_usd: float

    def to_dict(self) -> dict:
        return {
            "budget": self.budget,
            "scope": self.scope,
            "limit_usd": round(self.limit_usd, 6),
            "spent_usd": round(self.spent_usd, 6),
            "reserved_usd": round(self.reserved_usd, 6),
            "remaining_usd": round(max(0.0, self.limit_usd - self.spent_usd - self.reserved_usd), 6),
            "estimated_cost": round(self.estimate_usd, 6),
        }


@dataclass
class CostLedger:
    """Thread-safe, process-local spend ledger keyed by API key, period and scope."""

    clock: Callable[[], datetime] = utc_now
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _ids: itertools.count = field(default_factory=lambda: itertools.count(1))
    _holds: dict[int, _Hold] = field(default_factory=dict)
    _by_task: dict[tuple[str, str, str], int] = field(default_factory=dict)
    _key_spent: dict[tuple[str, str], float] = field(default_factory=lambda: defaultdict(float))
    _key_tasks: dict[tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))
    _scope_spent: dict[tuple[str, str], float] = field(default_factory=lambda: defaultdict(float))
    _scope_tasks: dict[tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))
    _model_spent: dict[tuple[str, str, str, str], list] = field(
        default_factory=lambda: defaultdict(lambda: [0.0, 0])
    )
    _scope_limits: dict[tuple[str, str], float] = field(default_factory=dict)

    # -- budgets ------------------------------------------------------------- #

    @staticmethod
    def key_limit(key: KeyConfig | None) -> float | None:
        return key.budget.limit_usd if key is not None and key.budget is not None else None

    @staticmethod
    def scope_limit(key: KeyConfig | None, client_limit: float | None) -> float | None:
        operator = key.budget.scopes_limit_usd if key is not None and key.budget is not None else None
        limits = [v for v in (client_limit, operator) if v is not None]
        return min(limits) if limits else None

    def applies(self, key: KeyConfig | None, scope: str | None, client_limit: float | None) -> bool:
        """True iff any cap constrains a task of ``key`` in ``scope``."""
        if self.key_limit(key) is not None:
            return True
        return scope is not None and self.scope_limit(key, client_limit) is not None

    def _period(self, key: KeyConfig | None) -> str:
        kind = key.budget.period if key is not None and key.budget is not None else "month"
        return period_window(kind, self.clock())[0]

    def _reserved(self, key_id: str, *, period: str | None = None, scope: str | None = None) -> float:
        total = 0.0
        for hold in self._holds.values():
            if hold.key_id != key_id:
                continue
            if period is not None and hold.period != period:
                continue
            if scope is not None and hold.scope != scope:
                continue
            total += hold.amount
        return total

    def _check(self, key: KeyConfig | None, scope: str | None, client_limit: float | None,
               amount: float) -> BudgetRejection | None:
        key_id = key.id if key is not None else "anonymous"
        limit = self.key_limit(key)
        if limit is not None:
            period = self._period(key)
            spent = self._key_spent[(key_id, period)]
            reserved = self._reserved(key_id, period=period)
            if spent + reserved + amount > limit + _EPS:
                return BudgetRejection("key", None, limit, spent, reserved, amount)
        if scope is not None:
            slimit = self.scope_limit(key, client_limit)
            if slimit is not None:
                spent = self._scope_spent[(key_id, scope)]
                reserved = self._reserved(key_id, scope=scope)
                if spent + reserved + amount > slimit + _EPS:
                    return BudgetRejection("scope", scope, slimit, spent, reserved, amount)
        return None

    def check(self, key: KeyConfig | None, scope: str | None, client_limit: float | None,
              amount: float) -> BudgetRejection | None:
        """Would a task costing ``amount`` fit every applicable budget right now?"""
        with self._lock:
            return self._check(key, scope, client_limit, amount)

    # -- holds, reservations, settlement ------------------------------------- #

    def hold(self, key: KeyConfig | None, *, scope: str | None, client_limit: float | None,
             model: str, modality: str, amount: float | None) -> int | BudgetRejection:
        """Atomically check and hold ``amount`` (``None`` = unpriced, holds 0)."""
        key_id = key.id if key is not None else "anonymous"
        value = max(0.0, amount or 0.0)
        with self._lock:
            rejection = self._check(key, scope, client_limit, value)
            if rejection is not None:
                return rejection
            hold_id = next(self._ids)
            self._holds[hold_id] = _Hold(key_id=key_id, period=self._period(key), scope=scope,
                                         model=model, modality=modality, amount=value,
                                         priced=amount is not None)
            if scope is not None:
                limit = self.scope_limit(key, client_limit)
                if limit is not None:
                    self._scope_limits[(key_id, scope)] = limit
            return hold_id

    def bind(self, hold_id: int, *, modality: str, backend: str, task_id: str) -> None:
        """The provider accepted the task: the hold becomes its reservation."""
        token = (modality, backend, task_id)
        with self._lock:
            hold = self._holds.get(hold_id)
            if hold is None:
                return
            hold.task = token
            self._by_task[token] = hold_id

    def release(self, hold_id: int) -> None:
        with self._lock:
            hold = self._holds.pop(hold_id, None)
            if hold is not None and hold.task is not None:
                self._by_task.pop(hold.task, None)

    def settle(self, *, modality: str, backend: str, task_id: str,
               cost: float | None, source: str) -> float | None:
        """Finish a task's reservation: record ``cost`` as spent, or release it (``None``)."""
        token = (modality, backend, task_id)
        with self._lock:
            hold_id = self._by_task.pop(token, None)
            hold = self._holds.pop(hold_id, None) if hold_id is not None else None
            if hold is None:
                return None
            if cost is None:
                return None
            amount = max(0.0, cost)
            self._key_spent[(hold.key_id, hold.period)] += amount
            self._key_tasks[(hold.key_id, hold.period)] += 1
            if hold.scope is not None:
                self._scope_spent[(hold.key_id, hold.scope)] += amount
                self._scope_tasks[(hold.key_id, hold.scope)] += 1
            entry = self._model_spent[(hold.key_id, hold.period, hold.model, hold.modality)]
            entry[0] += amount
            entry[1] += 1
        METRICS.inc_counter("gateway_cost_usd_total", amount, key=hold.key_id,
                            modality=hold.modality, model=hold.model, source=source)
        log.info("ledger_settled", task_id=task_id, backend=backend, modality=modality,
                 model=hold.model, cost=round(amount, 6), source=source, scope=hold.scope)
        return amount

    def reservation(self, *, modality: str, backend: str, task_id: str) -> tuple[float, bool] | None:
        """``(reserved amount, priced)`` of a task's reservation, or ``None``."""
        with self._lock:
            hold_id = self._by_task.get((modality, backend, task_id))
            hold = self._holds.get(hold_id) if hold_id is not None else None
            return (hold.amount, hold.priced) if hold is not None else None

    # -- reporting ------------------------------------------------------------ #

    def scope_state(self, key: KeyConfig | None, scope: str, client_limit: float | None = None) -> dict:
        key_id = key.id if key is not None else "anonymous"
        with self._lock:
            limit = self.scope_limit(key, client_limit)
            if limit is None:
                limit = self._scope_limits.get((key_id, scope))
            spent = self._scope_spent.get((key_id, scope), 0.0)
            reserved = self._reserved(key_id, scope=scope)
            return {
                "scope": scope,
                "limit_usd": limit,
                "spent_usd": round(spent, 6),
                "reserved_usd": round(reserved, 6),
                "remaining_usd": None if limit is None else round(max(0.0, limit - spent - reserved), 6),
                "tasks": self._scope_tasks.get((key_id, scope), 0),
            }

    def usage(self, key: KeyConfig | None, scope: str | None = None) -> dict:
        """The ``GET /v1/usage`` body for ``key``."""
        key_id = key.id if key is not None else "anonymous"
        kind = key.budget.period if key is not None and key.budget is not None else "month"
        period, start, end = period_window(kind, self.clock())
        limit = self.key_limit(key)
        with self._lock:
            spent = self._key_spent.get((key_id, period), 0.0)
            reserved = self._reserved(key_id, period=period)
            tasks = self._key_tasks.get((key_id, period), 0)
            scopes = sorted({s for (k, s) in self._scope_spent if k == key_id}
                            | {h.scope for h in self._holds.values() if h.key_id == key_id and h.scope}
                            | {s for (k, s) in self._scope_limits if k == key_id})
            models = [
                {"model": model, "modality": modality, "spent_usd": round(v[0], 6), "tasks": v[1]}
                for (k, p, model, modality), v in self._model_spent.items()
                if k == key_id and p == period
            ]
        if scope is not None:
            scopes = [scope]
        return {
            "object": "usage",
            "currency": "USD",
            "period": {"kind": kind, "start": start, "end": end},
            "key": {
                "limit_usd": limit,
                "spent_usd": round(spent, 6),
                "reserved_usd": round(reserved, 6),
                "remaining_usd": None if limit is None else round(max(0.0, limit - spent - reserved), 6),
                "tasks": tasks,
            },
            "scopes": [self.scope_state(key, s) for s in scopes],
            "models": sorted(models, key=lambda m: (-m["spent_usd"], m["model"])),
        }

    def render_prometheus(self, keys: list[KeyConfig]) -> str:
        """Key budget gauges for the current period (scopes stay out of labels)."""
        lines: list[str] = []
        for key in keys:
            limit = self.key_limit(key)
            if limit is None:
                continue
            period = self._period(key)
            with self._lock:
                spent = self._key_spent.get((key.id, period), 0.0)
            labels = prometheus_labels((("key", key.id),))
            lines.append(f'gateway_budget_spent_usd{{{labels}}} {round(spent, 6)}')
            lines.append(f'gateway_budget_limit_usd{{{labels}}} {limit}')
        return "\n".join(lines) + ("\n" if lines else "")

    def clear(self) -> None:
        with self._lock:
            self._holds.clear()
            self._by_task.clear()
            self._key_spent.clear()
            self._key_tasks.clear()
            self._scope_spent.clear()
            self._scope_tasks.clear()
            self._model_spent.clear()
            self._scope_limits.clear()


__all__ = ["BudgetRejection", "CostLedger", "period_window", "utc_now"]
