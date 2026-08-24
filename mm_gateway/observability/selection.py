"""Backend selection metrics — history-driven provider/account ranking.

This store records per ``(backend, account, model, modality)`` outcome data so
the auto-router can prefer backends that are *currently* fast and healthy, and
steer around ones that are rate-limited or failing — in addition to the static
limits fit the router already computes.

It is intentionally dependency-free and process-local (mirroring
:mod:`mm_gateway.observability.metrics`): a single module-level ``STORE``
holds rolling counts and a latency EWMA keyed by outcome tuple. A time-decayed
window lets "success rate at a specific time point" recover after a backend
recovers, and a rate-limit cooldown gates a candidate out until its window
expires. The same numbers are rendered to Prometheus via the existing
``/metrics`` exposition, and (like the rest of the metrics module) they can be
swapped for an external Prometheus by replacing this store.

Keys are strings (``backend``/``account``/``model``/``modality``); ``account``
identifies which credential a multi-account backend used (``"default"`` for
single-account backends, so a backend that never opts into multi-account
behaves exactly as before).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

# How long a rate-limit (429) or an explicit cooldown keeps a candidate out of
# selection. Picked to match typical provider rate-limit windows; operators with
# longer windows rely on the EWMA decay rather than this hard gate.
DEFAULT_RATE_LIMIT_COOLDOWN_S = 60.0
# Half-life of the latency EWMA in seconds. A backend that was slow 10 minutes
# ago but fast now should rank well now; one that was fast then but slow now
# should rank poorly. ~5 min half-life balances "history matters" with
# "current state matters".
DEFAULT_LATENCY_HALF_LIFE_S = 300.0
# Half-life of success/failure counting, so "success rate at a specific time
# point" reflects recent behaviour, not the process lifetime average.
DEFAULT_OUTCOME_HALF_LIFE_S = 300.0
# Rolling cap on the per-key event ring (purely to bound memory under load).
_MAX_EVENTS = 4096


@dataclass
class _Outcome:
    """Mutable accumulator for one ``(backend, account, model, modality)`` key.

    Mass values (``success_mass`` etc.) are *time-decayed* counts: they are not
    raw totals, they are folds over a half-life so recent behaviour dominates.
    Decay is applied lazily on each touch via the record's ``_last`` timestamp
    (stored as an attribute to keep the schema stable for pickling/printing).
    """

    success_mass: float = 0.0
    failure_mass: float = 0.0
    rate_limited_mass: float = 0.0
    latency_ewma: float | None = None
    # When a hard cooldown (rate limit / explicit) expires, in monotonic seconds.
    cooldown_until: float = 0.0
    _last: float = 0.0
    events: deque = field(default_factory=lambda: deque(maxlen=_MAX_EVENTS))


@dataclass
class _SelectionStore:
    """Thread-safe, in-process store of selection metrics.

    All access goes through the lock; reads are cheap (a few dict lookups + a
    decay fold) so holding it briefly is fine even under concurrent fan-out.

    Time is monotonic (``time.monotonic``) so cooldowns survive wall-clock
    skew; the ``_tick`` hook lets tests advance the clock without sleeping.
    """

    _records: dict[tuple, _Outcome] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _rate_limit_cooldown_s: float = DEFAULT_RATE_LIMIT_COOLDOWN_S
    _latency_half_life_s: float = DEFAULT_LATENCY_HALF_LIFE_S
    _outcome_half_life_s: float = DEFAULT_OUTCOME_HALF_LIFE_S

    # -- configuration ------------------------------------------------------- #

    def configure(
        self,
        *,
        rate_limit_cooldown_s: float | None = None,
        latency_half_life_s: float | None = None,
        outcome_half_life_s: float | None = None,
    ) -> None:
        """Tune decay/cooldown windows. Safe to call at any time."""
        with self._lock:
            if rate_limit_cooldown_s is not None:
                self._rate_limit_cooldown_s = max(0.0, rate_limit_cooldown_s)
            if latency_half_life_s is not None:
                self._latency_half_life_s = max(1e-3, latency_half_life_s)
            if outcome_half_life_s is not None:
                self._outcome_half_life_s = max(1e-3, outcome_half_life_s)

    # -- internal ------------------------------------------------------------ #

    def _now(self) -> float:
        return time.monotonic()

    def _record(self, key: tuple) -> _Outcome:
        rec = self._records.get(key)
        if rec is None:
            rec = _Outcome(_last=self._now())
            self._records[key] = rec
        return rec

    def _decay_factor(self, *, since: float, half_life: float) -> float:
        """Multiplicative decay for mass that elapsed ``since`` seconds ago."""
        if half_life <= 0 or since <= 0:
            return 1.0
        return 0.5 ** (since / half_life)

    def _decay_into(self, rec: _Outcome, now: float) -> None:
        """Fold accumulated mass forward to ``now`` so it reflects recency."""
        last = rec._last or now
        delta = max(0.0, now - last)
        if delta > 0:
            o_decay = self._decay_factor(since=delta, half_life=self._outcome_half_life_s)
            rec.success_mass *= o_decay
            rec.failure_mass *= o_decay
            rec.rate_limited_mass *= o_decay
        rec._last = now

    def _fold_latency(self, rec: _Outcome, latency_s: float) -> None:
        """EWMA-update the latency estimate toward the new sample."""
        # alpha derived from the half-life: the per-step "new" weight.
        alpha = 1.0 - 0.5 ** (1.0 / max(1.0, self._latency_half_life_s))
        if rec.latency_ewma is None:
            rec.latency_ewma = latency_s
        else:
            rec.latency_ewma = (1.0 - alpha) * rec.latency_ewma + alpha * latency_s

    # -- public API ---------------------------------------------------------- #

    def observe(
        self,
        *,
        backend: str,
        account: str = "default",
        model: str | None = None,
        modality: str = "image",
        outcome: str,
        latency_s: float | None = None,
        rate_limited: bool = False,
        cooldown_s: float | None = None,
    ) -> None:
        """Record one attempt's outcome.

        ``outcome`` is ``"success"`` or ``"failure"``. ``rate_limited`` (True on
        a 429 or an explicit cooldown signal) additionally accrues to the
        rate-limit mass and arms the hard cooldown gate. ``latency_s`` (wall
        clock of the attempt) folds into the latency EWMA. ``cooldown_s``
        overrides the default rate-limit cooldown window for this observation.
        """
        key = (backend, account, model or "", modality)
        now = self._now()
        with self._lock:
            rec = self._record(key)
            self._decay_into(rec, now)
            if outcome == "success":
                rec.success_mass += 1.0
            else:
                rec.failure_mass += 1.0
            if rate_limited:
                rec.rate_limited_mass += 1.0
                window = cooldown_s if cooldown_s is not None else self._rate_limit_cooldown_s
                rec.cooldown_until = max(rec.cooldown_until, now + max(0.0, window))
            if latency_s is not None and latency_s >= 0:
                self._fold_latency(rec, latency_s)
            rec.events.append((now, outcome, latency_s))

    def is_rate_limited(
        self, *, backend: str, account: str = "default", model: str | None = None,
        modality: str = "image",
    ) -> bool:
        """True iff the key is inside a hard cooldown window right now."""
        key = (backend, account, model or "", modality)
        now = self._now()
        with self._lock:
            rec = self._records.get(key)
            if rec is None:
                return False
            self._decay_into(rec, now)
            return now < rec.cooldown_until

    def health(
        self, *, backend: str, account: str = "default", model: str | None = None,
        modality: str = "image",
    ) -> dict:
        """Snapshot the key's current health: success rate, latency, cooldown.

        ``success_rate`` is the time-decayed success share
        (``success / (success + failure)``); ``None`` when there is no history
        (an untried backend is *not* penalised — it ranks as neutral). A backend
        in a hard cooldown reports ``rate_limited=True`` and ``success_rate=0``.
        """
        key = (backend, account, model or "", modality)
        now = self._now()
        with self._lock:
            rec = self._records.get(key)
            if rec is None:
                return {"success_rate": None, "latency_s": None, "rate_limited": False,
                        "attempts": 0}
            self._decay_into(rec, now)
            total = rec.success_mass + rec.failure_mass
            rate_limited = now < rec.cooldown_until
            if rate_limited:
                success_rate = 0.0
            elif total <= 0:
                success_rate = None
            else:
                success_rate = rec.success_mass / total
            return {
                "success_rate": success_rate,
                "latency_s": rec.latency_ewma,
                "rate_limited": rate_limited,
                "attempts": int(round(total)),
            }

    def score(
        self, *, backend: str, account: str = "default", model: str | None = None,
        modality: str = "image",
    ) -> float:
        """A single comparable health score in ``[0, 1]`` for ranking.

        Higher is better. ``None`` history → neutral ``0.5``. Cooldown → ``0``.
        Otherwise a blend of success rate (dominant) and latency (secondary):
        ``0.7 * success_rate + 0.3 * latency_share``, where a fast backend gets
        a higher latency_share. Latency share maps a 0–10 s latency band to
        1.0–0.0 so typical (sub-second) latencies cluster near the top and only
        genuinely slow backends are pushed down.
        """
        h = self.health(backend=backend, account=account, model=model, modality=modality)
        if h["rate_limited"]:
            return 0.0
        if h["success_rate"] is None:
            return 0.5
        latency = h["latency_s"]
        if latency is None or latency <= 0:
            latency_share = 1.0
        else:
            latency_share = max(0.0, 1.0 - min(1.0, latency / 10.0))
        return 0.7 * h["success_rate"] + 0.3 * latency_share

    def cooldown(
        self, *, backend: str, account: str = "default", model: str | None = None,
        modality: str = "image", cooldown_s: float | None = None,
    ) -> None:
        """Explicitly arm a cooldown (e.g. a backend self-reported rate limit)."""
        key = (backend, account, model or "", modality)
        now = self._now()
        window = cooldown_s if cooldown_s is not None else self._rate_limit_cooldown_s
        with self._lock:
            rec = self._record(key)
            self._decay_into(rec, now)
            rec.rate_limited_mass += 1.0
            rec.cooldown_until = max(rec.cooldown_until, now + max(0.0, window))

    def snapshot(self) -> dict:
        """All keys → health snapshot (for tests / debugging / Prometheus)."""
        out: dict = {}
        with self._lock:
            now = self._now()
            for key, rec in self._records.items():
                self._decay_into(rec, now)
                backend, account, model, modality = key
                total = rec.success_mass + rec.failure_mass
                rate_limited = now < rec.cooldown_until
                out[f"{backend}/{account}/{model or '*'}/{modality}"] = {
                    "backend": backend,
                    "account": account,
                    "model": model or None,
                    "modality": modality,
                    "success_rate": (0.0 if rate_limited
                                     else (rec.success_mass / total if total > 0 else None)),
                    "latency_s": rec.latency_ewma,
                    "rate_limited": rate_limited,
                    "cooldown_remaining_s": max(0.0, rec.cooldown_until - now),
                    "attempts": int(round(total)),
                }
        return out

    def clear(self) -> None:
        """Reset all history (tests)."""
        with self._lock:
            self._records.clear()

    def render_prometheus(self) -> str:
        """Expose selection metrics on ``/metrics`` alongside the request stats."""
        lines: list[str] = []
        for key, snap in self.snapshot().items():
            labels = (
                f'backend="{snap["backend"]}",account="{snap["account"]}",'
                f'model="{snap["model"] or ""}",modality="{snap["modality"]}"'
            )
            sr = snap["success_rate"]
            lat = snap["latency_s"]
            lines.append(
                f'gateway_selection_success_rate{{{labels}}} {sr if sr is not None else ""}'
            )
            if lat is not None:
                lines.append(f'gateway_selection_latency_seconds{{{labels}}} {lat}')
            lines.append(
                f'gateway_selection_rate_limited{{{labels}}} '
                f'{"1" if snap["rate_limited"] else "0"}'
            )
            lines.append(f'gateway_selection_attempts{{{labels}}} {snap["attempts"]}')
        return "\n".join(lines) + ("\n" if lines else "")

    # -- test hooks ---------------------------------------------------------- #

    def _tick(self, delta_s: float) -> None:
        """Advance every record's clock by ``delta_s`` (tests only).

        Since the store reads ``time.monotonic()``, simulating elapsed time
        means subtracting from each record's ``_last``/``cooldown_until``
        references so the delta appears to have passed. This keeps tests fast
        without sleeping while exercising the same decay/cooldown math.
        """
        with self._lock:
            for rec in self._records.values():
                rec._last = max(0.0, rec._last - delta_s)
                rec.cooldown_until = max(0.0, rec.cooldown_until - delta_s)


STORE = _SelectionStore()


def reset_for_tests() -> None:
    """Clear the module store (unit tests only)."""
    STORE.clear()


__all__ = [
    "STORE",
    "reset_for_tests",
    "_SelectionStore",
]
