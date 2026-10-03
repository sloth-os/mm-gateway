"""Auto mode: plan, attempt and account for one generation request.

Implements the pipeline of docs/design/auto-mode.md:

1. **Candidates** — every usable (backend, account, model) for the modality, or
   the backends serving a pinned model.
2. **Policy** — the ``routing`` directive merged over the operator's profile.
3. **Fit** — the limits catalogue's hard constraints (incl. native audio).
4. **Lifecycle** — retired models are dropped; deprecated ones rank last.
5. **Price** — each candidate's estimate; ``max_cost_usd`` and unpriced filters.
6. **Budget** — the key and scope budgets must absorb the estimate.
7. **Rank** — by the policy's ``optimize`` mode, then live health.
8. **Attempt** — best-first, holding the estimate in the ledger; retryable
   failures move to the next candidate.
9. **Settle** — done by the services' task monitors via :func:`settlement`.

The planner never contacts a provider: when nothing is admissible the request
fails with the most specific reason (budget → 402, cost cap → 422, limits or
retirement → 422; a pinned retired model → 410).
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from mm_gateway.billing import BudgetRejection, CostLedger
from mm_gateway.config import KeyConfig, Settings
from mm_gateway.core.base import AudioProvider, Provider, VoiceCloneProvider
from mm_gateway.core.exceptions import GatewayError, ProviderTimeoutError, ValidationError
from mm_gateway.models.pricing import estimate_cost
from mm_gateway.observability.logging import get_logger
from mm_gateway.observability.metrics import STORE as METRICS
from mm_gateway.observability.selection import STORE as SELECTION_STORE
from mm_gateway.services_selection import _is_rate_limited, _is_retryable
import mm_gateway.router as router

log = get_logger("auto_mode")

_AUTO = {None, "", "auto"}
_EPS = 1e-9
# Latency assumed for an untried candidate in ``optimize: latency`` (the middle
# of the selection store's 0–10 s latency band), so new backends get a chance.
_NEUTRAL_LATENCY_S = 5.0


def is_auto(model: str | None) -> bool:
    return model is None or model.lower() in _AUTO


class ModelRetiredError(GatewayError):
    status_code = 410
    code = "model_retired"


class BudgetExceededError(GatewayError):
    status_code = 402
    code = "budget_exceeded"


class CostLimitExceededError(GatewayError):
    status_code = 422
    code = "cost_limit_exceeded"


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RoutingPolicy:
    """The effective routing policy of one request."""

    optimize: str = "balanced"
    profile: str | None = None
    # Backends must carry one of these (a configured profile's ``tags``).
    tags: tuple[str, ...] = ()
    # An unconfigured ``profile`` name is a backend tag (the original behaviour).
    legacy_tag: str | None = None
    max_cost_usd: float | None = None
    fallback: str = "none"
    budget_scope: str | None = None
    budget_limit_usd: float | None = None


def resolve_policy(settings: Settings, directive: Any = None, *, tag: str | None = None) -> RoutingPolicy:
    """Merge the request's ``routing`` directive over the named profile (member by member)."""
    name = getattr(directive, "profile", None) or tag
    profile = settings.routing_profiles.get(name) if name else None
    budget = getattr(directive, "budget", None)

    def pick(member: str, default: Any) -> Any:
        value = getattr(directive, member, None)
        if value is not None:
            return value
        if profile is not None and getattr(profile, member, None) is not None:
            return getattr(profile, member)
        return default

    return RoutingPolicy(
        optimize=pick("optimize", settings.routing_default_optimize),
        profile=name,
        tags=tuple(profile.tags) if profile is not None else (),
        legacy_tag=name if name and profile is None else None,
        max_cost_usd=pick("max_cost_usd", None),
        fallback=pick("fallback", "none"),
        budget_scope=budget.scope if budget is not None else None,
        budget_limit_usd=budget.limit_usd if budget is not None else None,
    )


# --------------------------------------------------------------------------- #
# Plan
# --------------------------------------------------------------------------- #


@dataclass
class Candidate:
    provider: Provider
    account_id: str
    model: str
    backend: str
    kind: str  # "pinned" | "replacement" | "auto"
    estimate: float | None
    lifecycle: str
    sort_key: tuple = ()


@dataclass
class Exclusion:
    model: str
    reason: str  # limits | retired | max_cost | unpriced | budget
    estimate: float | None
    lifecycle: str


@dataclass
class RoutePlan:
    modality: str
    mode: str  # "auto" | "pinned"
    requested_model: str
    policy: RoutingPolicy
    candidates: list[Candidate] = field(default_factory=list)
    excluded: list[Exclusion] = field(default_factory=list)
    rejections: list[BudgetRejection] = field(default_factory=list)
    # Why a pinned request cannot use its model (``model_retired``/``over_budget``).
    pinned_unavailable: str | None = None


class _Planner:
    def __init__(self, registry: Any, ledger: CostLedger, request: Any, *, key: KeyConfig | None,
                 modality: str, policy: RoutingPolicy):
        self.registry = registry
        self.ledger = ledger
        self.key = key
        self.modality = modality
        self.policy = policy
        self.catalog = registry.catalog
        self.today = self.catalog.today()
        self.profile = router.profile_for(request)
        self.request = request
        self.side = router.longest_side(self.profile)
        self.budgeted = ledger.applies(key, policy.budget_scope, policy.budget_limit_usd)
        self.allow_unpriced = registry.settings.budget_allow_unpriced
        self.excluded: dict[tuple[str, str], Exclusion] = {}
        self.rejections: list[BudgetRejection] = []

    def estimate(self, model: str, limits: Any) -> float | None:
        return estimate_cost(
            self.catalog.price_for(model), modality=self.modality, limits=limits,
            duration_seconds=self.profile.duration_seconds,
            output_count=self.profile.output_count, longest_side=self.side,
            input_characters=self.profile.prompt_chars, voice_clone=self.profile.wants_voice_clone,
        )

    def audio_candidate_ok(self, backend: str, account: str, prov: Provider, model: str) -> bool:
        if self.modality != "audio":
            return True
        if not isinstance(prov, AudioProvider):
            return False
        if self.profile.wants_voice_clone and not isinstance(prov, VoiceCloneProvider):
            return False
        if getattr(self.request, "voice_backend", None) not in (None, backend):
            return False
        if getattr(self.request, "voice_account", None) not in (None, account):
            return False
        if prov.audio_request_error(self.request, model):
            self.exclude(model, "limits", None, self.catalog.lifecycle(model, "audio"))
            return False
        return True

    def exclude(self, model: str, reason: str, estimate: float | None, lifecycle: str) -> None:
        self.excluded.setdefault((model, reason), Exclusion(model, reason, estimate, lifecycle))

    def admit(self, model: str, limits: Any) -> tuple[bool, float | None, str | None]:
        """Stages 5–6: the estimate must fit ``max_cost_usd`` and the budgets."""
        est = self.estimate(model, limits)
        cap = self.policy.max_cost_usd
        if cap is not None:
            if est is None:
                return False, est, "unpriced"
            if est > cap + _EPS:
                return False, est, "max_cost"
        if self.budgeted:
            if est is None and not self.allow_unpriced:
                return False, est, "unpriced"
            rejection = self.ledger.check(self.key, self.policy.budget_scope,
                                          self.policy.budget_limit_usd, est or 0.0)
            if rejection is not None:
                self.rejections.append(rejection)
                return False, est, "budget"
        return True, est, None

    def tag_ok(self, backend: str) -> bool:
        tags = self.registry.tags_of(backend)
        if self.policy.legacy_tag and self.policy.legacy_tag not in tags:
            return False
        if self.policy.tags and not set(self.policy.tags) & set(tags):
            return False
        return True

    def _health(self, backend: str, account: str, model: str) -> tuple[bool, float, float | None]:
        keys = dict(backend=backend, account=account, model=model, modality=self.modality)
        return (
            SELECTION_STORE.is_rate_limited(**keys),
            SELECTION_STORE.score(**keys),
            SELECTION_STORE.health(**keys)["latency_s"],
        )

    def order_key(self, static: tuple, est: float | None, lifecycle: str,
                  rate_limited: bool, health: float, latency: float | None) -> tuple:
        deprecated = 1 if lifecycle == "deprecated" else 0
        if self.policy.optimize == "cost":
            return (rate_limited, est is None, est or 0.0, deprecated, -health, static)
        if self.policy.optimize == "latency":
            return (rate_limited, latency if latency is not None else _NEUTRAL_LATENCY_S, -health, static)
        return (rate_limited, -health, static)

    def auto_candidates(self, backend_name: str | None, exclude_model: str | None) -> list[Candidate]:
        default_backend, default_tag = self.registry.key_defaults(self.key, self.modality)
        out: list[Candidate] = []
        for b_index, name in enumerate(self.registry.usable_for_modality(self.key, self.modality, backend_name)):
            if not self.tag_ok(name):
                continue
            tags = self.registry.tags_of(name)
            for m_index, model in enumerate(self.registry.models_of(name, self.modality)):
                if exclude_model is not None and model == exclude_model:
                    continue
                limits = self.catalog.limits_for(model, self.modality)
                fit = router.score(self.profile, limits, backend_index=b_index, model_index=m_index)
                lifecycle = limits.lifecycle(self.today)
                if not fit.fits:
                    self.exclude(model, "limits", None, lifecycle)
                    continue
                if lifecycle == "retired":
                    self.exclude(model, "retired", self.estimate(model, limits), lifecycle)
                    continue
                ok, est, reason = self.admit(model, limits)
                if not ok:
                    self.exclude(model, reason or "limits", est, lifecycle)
                    continue
                static = (
                    0 if name == default_backend else 1,
                    0 if default_tag and default_tag in tags else 1,
                    -fit.optional_hits,
                    1 if lifecycle == "deprecated" else 0,
                    b_index,
                    m_index,
                )
                for account_id, prov in self.registry.accounts_of(name):
                    if not self.audio_candidate_ok(name, account_id, prov, model):
                        continue
                    rate_limited, health, latency = self._health(name, account_id, model)
                    out.append(Candidate(
                        provider=prov, account_id=account_id, model=model, backend=name,
                        kind="auto", estimate=est, lifecycle=lifecycle,
                        sort_key=self.order_key(static, est, lifecycle, rate_limited, health, latency),
                    ))
        out.sort(key=lambda c: c.sort_key)
        return out


def plan_route(
    registry: Any,
    ledger: CostLedger,
    request: Any,
    *,
    key: KeyConfig | None,
    modality: str,
    policy: RoutingPolicy,
    backend_name: str | None = None,
    raise_errors: bool = True,
) -> RoutePlan:
    """Stages 1–7. ``raise_errors=False`` (estimates) returns an empty plan instead of failing."""
    planner = _Planner(registry, ledger, request, key=key, modality=modality, policy=policy)
    mode = "auto" if is_auto(request.model) else "pinned"
    plan = RoutePlan(modality=modality, mode=mode,
                     requested_model="auto" if mode == "auto" else request.model, policy=policy)

    if mode == "auto":
        plan.candidates = planner.auto_candidates(backend_name, None)
    else:
        real_model, backends = registry.backends_serving(
            request.model, key, modality=modality, backend_name=backend_name)
        if modality == "audio":
            backends = [name for name in backends if any(
                planner.audio_candidate_ok(name, account, prov, real_model)
                for account, prov in registry.accounts_of(name)
            )]
        tagged = [b for b in backends if planner.tag_ok(b)]
        if (policy.legacy_tag or policy.tags) and not tagged:
            raise ValidationError(
                f"Routing profile '{policy.profile}' is unavailable for model '{request.model}'."
            )
        limits = planner.catalog.limits_for(real_model, modality)
        lifecycle = limits.lifecycle(planner.today)
        pinned: list[Candidate] = []
        if lifecycle == "retired":
            planner.exclude(request.model, "retired", None, lifecycle)
            plan.pinned_unavailable = "model_retired"
            if policy.fallback != "any" and raise_errors:
                replacement = f"; replacement: {limits.replacement}" if limits.replacement else ""
                raise ModelRetiredError(
                    f"Model '{request.model}' was retired on {limits.retired_on}{replacement}. "
                    "Pin another model, or allow routing.fallback='any'."
                )
        elif modality == "audio" and not router.score(planner.profile, limits, backend_index=0, model_index=0).fits:
            planner.exclude(request.model, "limits", None, lifecycle)
        else:
            ok, est, reason = planner.admit(real_model, limits)
            if not ok:
                planner.exclude(request.model, reason or "budget", est, lifecycle)
                plan.pinned_unavailable = "over_budget"
            else:
                chosen = tagged if policy.fallback in ("same_model", "any") else tagged[:1]
                for b_order, name in enumerate(chosen):
                    accounts = registry.accounts_of(name)
                    if modality == "audio":
                        accounts = [(account, prov) for account, prov in accounts
                                    if planner.audio_candidate_ok(name, account, prov, real_model)]
                    if policy.fallback == "none":
                        accounts = accounts[:1]
                    for account_id, prov in accounts:
                        rate_limited, health, latency = planner._health(name, account_id, real_model)
                        pinned.append(Candidate(
                            provider=prov, account_id=account_id, model=real_model, backend=name,
                            kind="pinned", estimate=est, lifecycle=lifecycle,
                            sort_key=(rate_limited, -health, b_order) if policy.fallback != "none" else (),
                        ))
                if policy.fallback != "none":
                    pinned.sort(key=lambda c: c.sort_key)
        plan.candidates = pinned
        if policy.fallback == "any":
            others = planner.auto_candidates(backend_name, real_model)
            successor = limits.replacement
            if successor:
                for c in others:
                    if c.model == successor:
                        c.kind = "replacement"
                others.sort(key=lambda c: (c.kind != "replacement",))
            plan.candidates = pinned + others

    plan.excluded = list(planner.excluded.values())
    plan.rejections = planner.rejections
    METRICS.inc_counter("gateway_routing_decisions_total", modality=modality, mode=mode,
                        optimize=policy.optimize)
    for ex in plan.excluded:
        METRICS.inc_counter("gateway_routing_excluded_total", modality=modality, reason=ex.reason)
    first = plan.candidates[0] if plan.candidates else None
    log.info("auto_route_plan", modality=modality, mode=mode, optimize=policy.optimize,
             candidates=len(plan.candidates),
             excluded={r: sum(1 for e in plan.excluded if e.reason == r)
                       for r in {e.reason for e in plan.excluded}},
             first_model=first.model if first else None,
             first_estimate=first.estimate if first else None,
             budget_scope=policy.budget_scope)
    if not plan.candidates and raise_errors:
        raise _no_candidate_error(plan, key)
    return plan


def _no_candidate_error(plan: RoutePlan, key: KeyConfig | None) -> GatewayError:
    reasons = {e.reason for e in plan.excluded}
    if "budget" in reasons and plan.rejections:
        cheapest = min(plan.rejections, key=lambda r: r.estimate_usd)
        METRICS.inc_counter("gateway_budget_rejections_total",
                            key=key.id if key is not None else "anonymous", budget=cheapest.budget)
        what = f"scope '{cheapest.scope}'" if cheapest.budget == "scope" else "the key's budget"
        return BudgetExceededError(
            f"Every model that fits this request would exceed {what}: "
            f"{cheapest.spent_usd + cheapest.reserved_usd:.4f} of {cheapest.limit_usd:.4f} USD used, "
            f"the cheapest candidate costs {cheapest.estimate_usd:.4f} USD.",
            details={"budget": cheapest.to_dict()},
        )
    if reasons & {"max_cost", "unpriced"}:
        cap = plan.policy.max_cost_usd
        priced = [e.estimate for e in plan.excluded if e.estimate is not None]
        cheapest = f" The cheapest candidate costs {min(priced):.4f} USD." if priced else ""
        limit = f"max_cost_usd={cap}" if cap is not None else "the budget"
        return CostLimitExceededError(
            f"No model that fits this request has a known price within {limit}.{cheapest}",
            details={"max_cost_usd": cap, "cheapest_estimate": min(priced) if priced else None},
        )
    if "retired" in reasons and not reasons - {"retired"}:
        return GatewayError(
            "Every model that fits this request is retired; set an explicit model or relax the input.",
            code="validation_error", status_code=422,
        )
    return GatewayError(
        "No configured model can serve this auto-routed request; relax "
        "the input (fewer images, smaller dimensions, shorter duration) "
        "or set an explicit model.",
        code="validation_error", status_code=422,
    )


# --------------------------------------------------------------------------- #
# Attempt
# --------------------------------------------------------------------------- #


def _failure_reason(exc: BaseException) -> str:
    if isinstance(exc, ProviderTimeoutError):
        return "timeout"
    if _is_rate_limited(exc):
        return "rate_limited"
    return "provider_unavailable"


Accept = Callable[[Any], None]


async def execute_plan(
    plan: RoutePlan,
    attempt: Callable[[Candidate, Accept], Awaitable[Any]],
    *,
    ledger: CostLedger,
    key: KeyConfig | None,
) -> tuple[Any, dict[str, Any]]:
    """Stage 8: try candidates best-first; returns ``(task, routing info)``.

    ``attempt(candidate, accept)`` creates the provider task and must call
    ``accept(task)`` as soon as the provider accepted it (before any sync wait
    or monitor start), so the ledger reservation is bound before settlement can
    happen. A retryable failure releases the hold and moves to the next
    candidate; a client error stops at once.
    """
    if not plan.candidates:
        raise _no_candidate_error(plan, key)
    policy = plan.policy
    reason: str | None = plan.pinned_unavailable
    last_exc: BaseException | None = None
    rejection: BudgetRejection | None = None
    attempts = 0
    for index, cand in enumerate(plan.candidates):
        hold = ledger.hold(key, scope=policy.budget_scope, client_limit=policy.budget_limit_usd,
                           model=cand.model, modality=plan.modality, amount=cand.estimate)
        if isinstance(hold, BudgetRejection):
            # The budget filled up since planning (concurrent creates): try a cheaper candidate.
            rejection = hold
            reason = "over_budget"
            continue
        attempts += 1
        start = time.monotonic()
        accepted = False

        def accept(task: Any, _hold: int = hold, _cand: Candidate = cand) -> None:
            nonlocal accepted
            accepted = True
            ledger.bind(_hold, modality=plan.modality, backend=_cand.backend, task_id=task.task_id)

        try:
            task = await attempt(cand, accept)
        except Exception as exc:  # noqa: BLE001
            ledger.release(hold)
            latency = time.monotonic() - start
            retryable = _is_retryable(exc)
            rate_limited = _is_rate_limited(exc)
            SELECTION_STORE.observe(backend=cand.backend, account=cand.account_id, model=cand.model,
                                    modality=plan.modality, outcome="failure", latency_s=latency,
                                    rate_limited=rate_limited)
            log.info("auto_route_attempt_failed", backend=cand.backend, account=cand.account_id,
                     model=cand.model, modality=plan.modality, latency_s=round(latency, 3),
                     rate_limited=rate_limited, retryable=retryable, error=str(exc))
            last_exc = exc
            # A pinned request without fallback keeps its single-attempt contract.
            if not retryable or (plan.mode == "pinned" and policy.fallback == "none"):
                raise
            reason = _failure_reason(exc)
            continue
        if not accepted:
            accept(task)
        latency = time.monotonic() - start
        SELECTION_STORE.observe(backend=cand.backend, account=cand.account_id, model=cand.model,
                                modality=plan.modality, outcome="success", latency_s=latency)
        log.info("auto_route_attempt_ok", backend=cand.backend, account=cand.account_id,
                 model=cand.model, modality=plan.modality, latency_s=round(latency, 3),
                 attempts=attempts, estimate=cand.estimate)
        fallback = index > 0 or plan.pinned_unavailable is not None \
            or (plan.mode == "pinned" and cand.kind != "pinned")
        if fallback:
            METRICS.inc_counter("gateway_routing_fallbacks_total", modality=plan.modality,
                                reason=reason or "provider_unavailable")
        info: dict[str, Any] = {
            "requested_model": plan.requested_model,
            "fallback": fallback,
            "fallback_reason": (reason or "provider_unavailable") if fallback else None,
            "attempts": attempts,
            "optimize": policy.optimize,
            "estimated_cost": cand.estimate,
        }
        if policy.budget_scope:
            info["budget"] = ledger.scope_state(key, policy.budget_scope, policy.budget_limit_usd)
        elif ledger.key_limit(key) is not None:
            info["budget"] = {k: v for k, v in ledger.usage(key)["key"].items()}
        return task, info
    if last_exc is not None:
        raise last_exc
    if rejection is not None:
        plan.rejections.append(rejection)
        plan.excluded.append(Exclusion(plan.candidates[-1].model, "budget",
                                       rejection.estimate_usd, plan.candidates[-1].lifecycle))
    raise _no_candidate_error(plan, key)


def settlement(ledger: CostLedger, modality: str, backend: str) -> Callable[[Any], None]:
    """Stage 9: the monitor's terminal callback that settles a task's reservation."""

    def on_terminal(task: Any) -> None:
        usage = getattr(task, "usage", None)
        provider_cost = getattr(usage, "cost", None) if usage is not None else None
        if task.status != "succeeded":
            ledger.settle(modality=modality, backend=backend, task_id=task.task_id,
                          cost=None, source="none")
            return
        if provider_cost is not None:
            ledger.settle(modality=modality, backend=backend, task_id=task.task_id,
                          cost=float(provider_cost), source="provider")
            return
        reservation = ledger.reservation(modality=modality, backend=backend, task_id=task.task_id)
        if reservation is None:
            return
        amount, priced = reservation
        # An unpriced model's task is counted, but its unknown cost is not invented.
        ledger.settle(modality=modality, backend=backend, task_id=task.task_id,
                      cost=amount if priced else 0.0, source="estimate" if priced else "unpriced")

    return on_terminal


# --------------------------------------------------------------------------- #
# Estimate
# --------------------------------------------------------------------------- #


def estimate_route(
    registry: Any,
    ledger: CostLedger,
    request: Any,
    *,
    key: KeyConfig | None,
    modality: str,
    policy: RoutingPolicy,
) -> dict[str, Any]:
    """The ``POST /v1/{modality}/estimate`` body: stages 1–7 without creating a task."""
    plan = plan_route(registry, ledger, request, key=key, modality=modality, policy=policy,
                      raise_errors=False)
    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []
    for c in plan.candidates:
        if c.model in seen:
            continue
        seen.add(c.model)
        candidates.append({"model": c.model, "estimated_cost": c.estimate,
                           "lifecycle": c.lifecycle, "admissible": True})
    for e in plan.excluded:
        if e.model in seen:
            continue
        seen.add(e.model)
        candidates.append({"model": e.model, "estimated_cost": e.estimate, "lifecycle": e.lifecycle,
                           "admissible": False, "reason": e.reason})
    first = plan.candidates[0] if plan.candidates else None
    budget = None
    if policy.budget_scope:
        budget = ledger.scope_state(key, policy.budget_scope, policy.budget_limit_usd)
    elif ledger.key_limit(key) is not None:
        budget = ledger.usage(key)["key"]
    return {
        "object": "estimate",
        "modality": modality,
        "currency": "USD",
        "model": first.model if first else None,
        "estimated_cost": first.estimate if first else None,
        "candidates": candidates,
        "budget": budget,
    }


__all__ = [
    "BudgetExceededError",
    "Candidate",
    "CostLimitExceededError",
    "ModelRetiredError",
    "RoutePlan",
    "RoutingPolicy",
    "estimate_route",
    "execute_plan",
    "is_auto",
    "plan_route",
    "resolve_policy",
    "settlement",
]
