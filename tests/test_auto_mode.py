"""Auto mode: native-audio routing, model lifecycle, fallbacks, cost ordering,
cost caps, budgets and the ledger, estimates and usage (docs/design/auto-mode.md).

Scripted video providers serve real catalogue ids (``veo-2.0-generate-001`` is
silent, ``veo-3.1-generate-preview`` renders audio and is priced,
``sora-2`` was retired on 2026-09-24) so the built-in limits, lifecycle and
price catalogues drive the decisions under test.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mm_gateway.auto_mode import (
    BudgetExceededError,
    CostLimitExceededError,
    ModelRetiredError,
    plan_route,
    resolve_policy,
)
from mm_gateway.billing import CostLedger, period_window
from mm_gateway.config import BackendConfig, KeyBudget, KeyConfig, RoutingProfile, Settings
from mm_gateway.core.base import VideoProvider
from mm_gateway.core.exceptions import ConfigError, GatewayError, ProviderRequestError
from mm_gateway.models.catalog import Catalog, parse_overrides
from mm_gateway.models.limits import limits_for
from mm_gateway.models.pricing import ModelPrice, PriceTier, estimate_cost
from mm_gateway.observability.metrics import STORE as METRICS
from mm_gateway.observability.selection import STORE as SELECTION_STORE
from mm_gateway.registry import Registry
from mm_gateway.schemas.api import BudgetDirective, RoutingDirective
from mm_gateway.schemas.video import UnifiedVideoRequest, UnifiedVideoTask, VideoUsage, text_part
from mm_gateway.server.app import create_app
from mm_gateway.services import VideoService

TODAY = date(2026, 10, 1)


@pytest.fixture(autouse=True)
def _clean_stores():
    SELECTION_STORE.clear()
    yield
    SELECTION_STORE.clear()


class Scripted(VideoProvider):
    """A video backend serving ``models``; ``behaviour`` is popped per create.

    Items: an Exception to raise, or ``"ok"`` (a pending task that succeeds on
    the first poll, with ``cost`` as provider-reported cost when set), or
    ``"fail"`` (a task that fails on its first poll).
    """

    def __init__(self, name: str, models: list[str], behaviour: list[Any] | None = None,
                 cost: float | None = None):
        super().__init__(BackendConfig(name=name, type="scripted", api_key="k"))
        self.video_models = list(models)
        self.behaviour = list(behaviour or [])
        self.cost = cost
        self.calls: list[UnifiedVideoRequest] = []
        self.outcome: dict[str, str] = {}

    @property
    def name(self) -> str:  # type: ignore[override]
        return self.backend.name

    async def create_video_task(self, request: UnifiedVideoRequest) -> UnifiedVideoTask:
        self.calls.append(request)
        item = self.behaviour.pop(0) if self.behaviour else "ok"
        if isinstance(item, BaseException):
            raise item
        task_id = f"{self.name}-{len(self.calls)}"
        self.outcome[task_id] = item
        return UnifiedVideoTask(task_id=task_id, provider=self.name, model=request.model,
                                status="pending")

    async def get_video_task(self, task_id: str) -> UnifiedVideoTask:
        if self.outcome.get(task_id) == "fail":
            return UnifiedVideoTask(task_id=task_id, provider=self.name, model="m", status="failed",
                                    error="upstream rejected the prompt")
        task = UnifiedVideoTask(task_id=task_id, provider=self.name, model="m", status="succeeded",
                                video_urls=["https://example.test/out.mp4"])
        if self.cost is not None:
            task.usage = VideoUsage(cost=self.cost, video_count=1)
        return task


def _registry(providers: list[Scripted], *, settings: Settings | None = None,
              today: date = TODAY) -> Registry:
    settings = settings or Settings(keys=[KeyConfig(id="t", key="")])
    cfgs = [BackendConfig(name=p.name, type="scripted", api_key="k", tags=list(p.backend.tags))
            for p in providers]
    settings = Settings(**{**settings.__dict__, "backends": cfgs})
    reg = Registry(settings)
    for p, cfg in zip(providers, cfgs, strict=True):
        reg._backends[p.name] = p
        reg._configs[p.name] = cfg
        reg._backend_accounts[p.name] = ["default"]
        reg._accounts[(p.name, "default")] = p
    reg.catalog = Catalog(parse_overrides(settings.catalog_models), today=lambda: today)
    return reg


def _video(model: str = "auto", *, audio: bool | None = None, duration: float | None = 8,
           **extra: Any) -> UnifiedVideoRequest:
    return UnifiedVideoRequest(model=model, content=[text_part("a lighthouse at dusk")],
                               duration=duration, generate_audio=audio, **extra)


def _service(reg: Registry, ledger: CostLedger | None = None) -> VideoService:
    return VideoService(reg, max_sync_wait=1.0, poll_interval=0.01, sync_default=False,
                        ledger=ledger or CostLedger())


def _plan(reg, request, *, key=None, routing=None, ledger=None, **kw):
    return plan_route(reg, ledger or CostLedger(), request, key=key or KeyConfig(id="t", key=""),
                      modality="video", policy=resolve_policy(reg.settings, routing), **kw)


# --------------------------------------------------------------------------- #
# Catalogue: lifecycle, native audio, prices
# --------------------------------------------------------------------------- #


def test_sora_2_is_retired_from_2026_09_24():
    sora = limits_for("sora-2", "video")
    assert sora.lifecycle(date(2026, 1, 1)) == "active"
    assert sora.lifecycle(date(2026, 3, 24)) == "deprecated"
    assert sora.lifecycle(date(2026, 9, 23)) == "deprecated"
    assert sora.lifecycle(date(2026, 9, 24)) == "retired"
    assert limits_for("sora-2-pro", "video").lifecycle(TODAY) == "retired"
    public = sora.to_public_dict(TODAY)
    assert public["lifecycle"] == "retired" and public["retired_on"] == "2026-09-24"


def test_native_audio_flags_follow_the_providers_documentation():
    assert limits_for("veo-2.0-generate-001", "video").supports_audio_output is False
    assert limits_for("veo-3.1-generate-preview", "video").supports_audio_output is True
    assert limits_for("doubao-seedance-2-0-260128", "video").supports_audio_output is True
    assert limits_for("wanx2.1-t2v-turbo", "video").supports_audio_output is False


def test_estimates_use_size_tiers_duration_and_counts():
    veo = Catalog().price_for("veo-3.1-generate-preview")
    limits = limits_for("veo-3.1-generate-preview", "video")
    assert estimate_cost(veo, modality="video", limits=limits, duration_seconds=8,
                         longest_side=1280) == pytest.approx(3.2)
    assert estimate_cost(veo, modality="video", limits=limits, duration_seconds=8,
                         longest_side=3840) == pytest.approx(4.8)
    # An open duration uses the model's minimum (4 s for Veo 3.1).
    assert estimate_cost(veo, modality="video", limits=limits) == pytest.approx(1.6)
    image = ModelPrice(per_image=0.04, per_request=0.01)
    assert estimate_cost(image, modality="image", limits=limits_for("x", "image"),
                         output_count=3) == pytest.approx(0.13)
    music = ModelPrice(per_second=0.002)
    assert estimate_cost(music, modality="music", limits=limits_for("x", "music")) == pytest.approx(0.06)
    assert estimate_cost(None, modality="video", limits=limits) is None
    assert ModelPrice(per_second_tiers=(PriceTier(1280, 0.1), PriceTier(1920, 0.12))).rate_for(1920) == 0.12


def test_catalog_overrides_retire_price_and_flag_models():
    catalog = Catalog(parse_overrides({
        "veo-3.1-generate-preview": {"retired_on": "2026-09-30", "replacement": "veo-3.1-fast-generate-preview"},
        "my-ltx": {"modality": "video", "supports_audio_output": True, "price": {"per_second": 0.01}},
    }), today=lambda: TODAY)
    assert catalog.lifecycle("veo-3.1-generate-preview", "video") == "retired"
    assert catalog.limits_for("veo-3.1-generate-preview", "video").replacement == "veo-3.1-fast-generate-preview"
    assert catalog.limits_for("my-ltx", "video").supports_audio_output is True
    assert catalog.price_for("my-ltx").per_second == 0.01
    assert catalog.public_limits("my-ltx", "video")["price"] == {"currency": "USD", "per_second": 0.01}


def test_catalog_declares_multi_shot_and_enhancement_models():
    catalog = Catalog(parse_overrides({
        "my-multishot": {"modality": "video", "max_shots": 4, "max_duration_seconds": 20},
        "my-upscaler": {"modality": "video", "supports_upscale": True, "supports_frame_interpolation": True,
                        "max_fps": 60},
    }), today=lambda: TODAY)
    assert catalog.limits_for("my-multishot", "video").max_shots == 4
    public = catalog.public_limits("my-multishot", "video")
    assert public["max_shots"] == 4 and public["max_duration_seconds"] == 20
    upscaler = catalog.public_limits("my-upscaler", "video")
    assert upscaler["supports_upscale"] is True
    assert upscaler["supports_frame_interpolation"] is True
    assert upscaler["max_fps"] == 60
    # undocumented fields stay out of the public object
    assert "max_shots" not in catalog.public_limits("veo-3.1-generate-preview", "video")


def test_catalog_declares_segmentation_models():
    catalog = Catalog(parse_overrides({
        "my-matting": {"modality": "video", "supports_reference_video": True, "supports_segmentation": True},
    }), today=lambda: TODAY)
    assert catalog.limits_for("my-matting", "video").supports_segmentation is True
    public = catalog.public_limits("my-matting", "video")
    assert public["supports_segmentation"] is True and public["supports_reference_video"] is True
    assert "supports_segmentation" not in catalog.public_limits("veo-3.1-generate-preview", "video")


def test_catalog_declares_performance_models():
    catalog = Catalog(parse_overrides({
        "my-act": {"modality": "video", "supports_first_frame": True, "supports_reference_video": True,
                   "supports_performance": True, "max_duration_seconds": 30},
    }), today=lambda: TODAY)
    assert catalog.limits_for("my-act", "video").supports_performance is True
    public = catalog.public_limits("my-act", "video")
    assert public["supports_performance"] is True and public["supports_first_frame"] is True
    assert "supports_performance" not in catalog.public_limits("veo-3.1-generate-preview", "video")


@pytest.mark.parametrize("raw", [
    {"m": {"retired_on": "next tuesday"}},
    {"m": {"unknown_field": 1}},
    {"m": {"price": {"per_second": -1}}},
    {"m": {"price": {"currency": "EUR", "per_second": 1}}},
    {"m": {"modality": "speech"}},
])
def test_catalog_overrides_are_validated(raw):
    with pytest.raises(ConfigError):
        parse_overrides(raw)


def test_settings_parse_routing_catalog_budget_and_key_budgets():
    settings = Settings._from_yaml("""
routing:
  default_optimize: latency
  profiles:
    cheap: {optimize: cost}
    capped: {optimize: cost, max_cost_usd: 2, fallback: any, tags: [prod]}
catalog:
  models:
    sora-2: {retired_on: 2026-09-24}
budget: {allow_unpriced: true}
keys:
  - id: app
    key: k
    budget: {limit_usd: 500, period: month, scopes_limit_usd: 50}
""")
    assert settings.routing_default_optimize == "latency"
    assert settings.routing_profiles["capped"] == RoutingProfile(
        name="capped", tags=["prod"], optimize="cost", max_cost_usd=2.0, fallback="any")
    assert settings.budget_allow_unpriced is True
    assert settings.keys[0].budget == KeyBudget(limit_usd=500.0, period="month", scopes_limit_usd=50.0)
    assert parse_overrides(settings.catalog_models)["sora-2"].fields == {"retired_on": "2026-09-24"}
    for bad in ("routing: {default_optimize: cheapest}",
                "routing: {profiles: {x: {fallback: sometimes}}}",
                "keys: [{id: a, key: k, budget: {period: weekly}}]"):
        with pytest.raises(ValueError):
            Settings._from_yaml(bad)


# --------------------------------------------------------------------------- #
# Fit and lifecycle
# --------------------------------------------------------------------------- #


def test_include_audio_never_routes_to_a_silent_model():
    silent = Scripted("silent", ["veo-2.0-generate-001"])
    loud = Scripted("loud", ["veo-3.1-generate-preview"])
    reg = _registry([silent, loud])
    plan = _plan(reg, _video(audio=True))
    assert [c.model for c in plan.candidates] == ["veo-3.1-generate-preview"]
    assert {(e.model, e.reason) for e in plan.excluded} == {("veo-2.0-generate-001", "limits")}
    # Without native audio the configuration order wins again.
    assert _plan(reg, _video(audio=None)).candidates[0].model == "veo-2.0-generate-001"


def test_retired_models_are_skipped_and_deprecated_ones_rank_last():
    sora = Scripted("openai-video", ["sora-2"])
    veo = Scripted("google-video", ["veo-3.1-generate-preview"])
    plan = _plan(_registry([sora, veo]), _video())
    assert [c.model for c in plan.candidates] == ["veo-3.1-generate-preview"]
    assert [(e.model, e.reason, e.lifecycle) for e in plan.excluded] == [("sora-2", "retired", "retired")]
    # In May 2026 Sora was deprecated: still routable, but after the active model.
    plan = _plan(_registry([sora, veo], today=date(2026, 5, 1)), _video())
    assert [c.model for c in plan.candidates] == ["veo-3.1-generate-preview", "sora-2"]


def test_only_retired_fits_fails_before_any_provider_call():
    sora = Scripted("openai-video", ["sora-2"])
    with pytest.raises(GatewayError) as err:
        asyncio.run(_service(_registry([sora])).create(_video(), key=KeyConfig(id="t", key="")))
    assert err.value.status_code == 422 and "retired" in err.value.message
    assert sora.calls == []


def test_pinned_retired_model_is_gone_unless_fallback_any():
    sora = Scripted("openai-video", ["sora-2"])
    veo = Scripted("google-video", ["veo-3.1-generate-preview"])
    reg = _registry([sora, veo], settings=Settings(
        keys=[KeyConfig(id="t", key="")],
        catalog_models={"sora-2": {"replacement": "veo-3.1-generate-preview"}}))
    svc = _service(reg)
    key = KeyConfig(id="t", key="")
    with pytest.raises(ModelRetiredError) as err:
        asyncio.run(svc.create(_video("sora-2"), key=key))
    assert err.value.status_code == 410
    assert "2026-09-24" in err.value.message and "veo-3.1-generate-preview" in err.value.message

    task = asyncio.run(svc.create(_video("sora-2"), key=key,
                                  routing=RoutingDirective(fallback="any")))
    assert task.provider == "google-video" and sora.calls == []
    assert veo.calls[0].model == "veo-3.1-generate-preview"
    assert task.routing == {
        "requested_model": "sora-2", "fallback": True, "fallback_reason": "model_retired",
        "attempts": 1, "optimize": "balanced", "estimated_cost": pytest.approx(3.2),
    }


# --------------------------------------------------------------------------- #
# Fallbacks
# --------------------------------------------------------------------------- #


def test_pinned_without_fallback_keeps_its_single_attempt():
    a = Scripted("a", ["veo-3.1-generate-preview"],
                 [ProviderRequestError("429", status_code=429, provider="a")])
    b = Scripted("b", ["veo-3.1-generate-preview"])
    with pytest.raises(ProviderRequestError):
        asyncio.run(_service(_registry([a, b])).create(_video("veo-3.1-generate-preview"),
                                                       key=KeyConfig(id="t", key="")))
    assert b.calls == []


def test_same_model_fallback_tries_other_backends_of_the_pinned_model():
    a = Scripted("a", ["veo-3.1-generate-preview"],
                 [ProviderRequestError("429", status_code=429, provider="a")])
    b = Scripted("b", ["veo-3.1-generate-preview"])
    c = Scripted("c", ["veo-2.0-generate-001"])
    task = asyncio.run(_service(_registry([a, b, c])).create(
        _video("veo-3.1-generate-preview"), key=KeyConfig(id="t", key=""),
        routing=RoutingDirective(fallback="same_model")))
    assert task.provider == "b" and c.calls == []
    assert task.routing["fallback"] is True
    assert task.routing["fallback_reason"] == "rate_limited"
    assert task.routing["attempts"] == 2


def test_any_fallback_moves_to_another_model_after_the_pinned_one_fails():
    a = Scripted("a", ["veo-3.1-generate-preview"],
                 [ProviderRequestError("503", status_code=503, provider="a")])
    b = Scripted("b", ["veo-2.0-generate-001"])
    task = asyncio.run(_service(_registry([a, b])).create(
        _video("veo-3.1-generate-preview"), key=KeyConfig(id="t", key=""),
        routing=RoutingDirective(fallback="any")))
    assert task.provider == "b"
    assert task.routing["fallback_reason"] == "provider_unavailable"


def test_client_errors_never_fall_back():
    a = Scripted("a", ["veo-3.1-generate-preview"],
                 [GatewayError("bad prompt", code="invalid_request_error", status_code=400)])
    b = Scripted("b", ["veo-2.0-generate-001"])
    with pytest.raises(GatewayError):
        asyncio.run(_service(_registry([a, b])).create(_video(), key=KeyConfig(id="t", key="")))
    assert b.calls == []


# --------------------------------------------------------------------------- #
# Cost: ordering and caps
# --------------------------------------------------------------------------- #


def _priced_registry(**settings_kw):
    fast = Scripted("fast", ["veo-3.1-fast-generate-preview"])
    std = Scripted("std", ["veo-3.1-generate-preview"])
    free = Scripted("unpriced", ["mystery-video"])
    settings = Settings(keys=[KeyConfig(id="t", key="")], **settings_kw)
    return _registry([std, free, fast], settings=settings), fast, std, free


def test_optimize_cost_orders_by_estimate_and_unpriced_last():
    reg, *_ = _priced_registry()
    request = _video(width=1280, height=720)
    balanced = _plan(reg, request)
    assert [c.model for c in balanced.candidates] == [
        "veo-3.1-generate-preview", "mystery-video", "veo-3.1-fast-generate-preview"]
    cost = _plan(reg, request, routing=RoutingDirective(optimize="cost"))
    assert [(c.model, c.estimate) for c in cost.candidates] == [
        ("veo-3.1-fast-generate-preview", pytest.approx(0.8)),
        ("veo-3.1-generate-preview", pytest.approx(3.2)),
        ("mystery-video", None),
    ]


def test_a_routing_profile_supplies_the_optimize_mode():
    reg, *_ = _priced_registry(routing_profiles={"cheap": RoutingProfile(name="cheap", optimize="cost")})
    plan = _plan(reg, _video(width=1280, height=720), routing=RoutingDirective(profile="cheap"))
    assert plan.candidates[0].model == "veo-3.1-fast-generate-preview"
    assert plan.policy.legacy_tag is None  # a configured profile is not a backend tag


def test_latency_mode_prefers_the_fastest_observed_candidate():
    reg, fast, std, free = _priced_registry()
    SELECTION_STORE.observe(backend="std", model="veo-3.1-generate-preview", modality="video",
                            outcome="success", latency_s=9.0)
    SELECTION_STORE.observe(backend="fast", model="veo-3.1-fast-generate-preview", modality="video",
                            outcome="success", latency_s=0.5)
    plan = _plan(reg, _video(), routing=RoutingDirective(optimize="latency"))
    assert plan.candidates[0].model == "veo-3.1-fast-generate-preview"


def test_max_cost_excludes_expensive_and_unpriced_models():
    reg, fast, std, free = _priced_registry()
    plan = _plan(reg, _video(width=1280, height=720), routing=RoutingDirective(max_cost_usd=1.0))
    assert [c.model for c in plan.candidates] == ["veo-3.1-fast-generate-preview"]
    assert {(e.model, e.reason) for e in plan.excluded} == {
        ("veo-3.1-generate-preview", "max_cost"), ("mystery-video", "unpriced")}
    with pytest.raises(CostLimitExceededError) as err:
        _plan(reg, _video(width=1280, height=720), routing=RoutingDirective(max_cost_usd=0.5))
    assert err.value.status_code == 422
    assert err.value.details["cheapest_estimate"] == pytest.approx(0.8)


# --------------------------------------------------------------------------- #
# Budgets and the ledger
# --------------------------------------------------------------------------- #


def test_period_windows_are_utc_calendar_periods():
    now = datetime(2026, 12, 31, 23, 59, tzinfo=UTC)
    assert period_window("day", now)[0] == "2026-12-31"
    pid, start, end = period_window("month", now)
    assert (pid, start.month, end.year, end.month) == ("2026-12", 12, 2027, 1)
    assert period_window("total", now) == ("total", None, None)


def test_ledger_holds_bind_settle_and_release():
    ledger = CostLedger()
    key = KeyConfig(id="k", key="", budget=KeyBudget(limit_usd=1.0, period="total"))
    h1 = ledger.hold(key, scope="s", client_limit=None, model="m", modality="video", amount=0.6)
    assert isinstance(h1, int)
    # A second 0.6 would overrun the key budget while the first is reserved.
    rejected = ledger.hold(key, scope="s", client_limit=None, model="m", modality="video", amount=0.6)
    assert rejected.budget == "key" and rejected.reserved_usd == pytest.approx(0.6)
    ledger.bind(h1, modality="video", backend="b", task_id="t1")
    assert ledger.settle(modality="video", backend="b", task_id="t1", cost=0.5, source="provider") == 0.5
    usage = ledger.usage(key)
    assert usage["key"] == {"limit_usd": 1.0, "spent_usd": 0.5, "reserved_usd": 0.0,
                            "remaining_usd": 0.5, "tasks": 1}
    assert usage["models"] == [{"model": "m", "modality": "video", "spent_usd": 0.5, "tasks": 1}]
    h2 = ledger.hold(key, scope="s", client_limit=None, model="m", modality="video", amount=0.4)
    ledger.release(h2)
    assert ledger.usage(key)["key"]["reserved_usd"] == 0.0


def test_scope_cap_is_the_smaller_of_client_and_operator_limits():
    key = KeyConfig(id="k", key="", budget=KeyBudget(scopes_limit_usd=2.0))
    ledger = CostLedger()
    assert ledger.scope_limit(key, 5.0) == 2.0
    assert ledger.scope_limit(key, 1.0) == 1.0
    assert ledger.scope_limit(KeyConfig(id="o", key=""), None) is None
    assert ledger.applies(key, "s", None) is True
    assert ledger.applies(KeyConfig(id="o", key=""), "s", None) is False


def test_key_budget_reserves_estimates_and_refuses_with_402():
    reg, fast, std, free = _priced_registry()
    ledger = CostLedger()
    svc = _service(reg, ledger)
    key = KeyConfig(id="t", key="", budget=KeyBudget(limit_usd=1.0, period="month"))
    routing = RoutingDirective(optimize="cost")

    async def scenario():
        first = await svc.create(_video(width=1280, height=720), key=key, routing=routing)
        assert first.routing["estimated_cost"] == pytest.approx(0.8)
        assert ledger.usage(key)["key"]["reserved_usd"] == pytest.approx(0.8)
        with pytest.raises(BudgetExceededError) as err:
            await svc.create(_video(width=1280, height=720), key=key, routing=routing)
        assert err.value.status_code == 402
        assert err.value.details["budget"]["budget"] == "key"
        assert err.value.details["budget"]["remaining_usd"] == pytest.approx(0.2)
        # The first task finishes: its estimate becomes spend, nothing is reserved.
        await svc._supervisor.wait_for_terminal(first.task_id, provider="fast", timeout=2)
        usage = ledger.usage(key)["key"]
        assert usage["spent_usd"] == pytest.approx(0.8) and usage["reserved_usd"] == 0.0

    asyncio.run(scenario())
    # Unpriced models never serve a budgeted key (their cost cannot be proven to fit).
    assert free.calls == []


def test_failed_tasks_release_their_reservation_without_cost():
    std = Scripted("std", ["veo-3.1-generate-preview"], ["fail"])
    reg = _registry([std])
    ledger = CostLedger()
    svc = _service(reg, ledger)
    key = KeyConfig(id="t", key="")
    scope = RoutingDirective(budget=BudgetDirective(scope="rideo:prj_1", limit_usd=10))

    async def scenario():
        task = await svc.create(_video(), key=key, routing=scope)
        assert ledger.scope_state(key, "rideo:prj_1")["reserved_usd"] == pytest.approx(3.2)
        await svc._supervisor.wait_for_terminal(task.task_id, provider="std", timeout=2)
        state = ledger.scope_state(key, "rideo:prj_1")
        assert (state["spent_usd"], state["reserved_usd"], state["tasks"]) == (0.0, 0.0, 0)

    asyncio.run(scenario())


def test_provider_reported_cost_wins_over_the_estimate_in_the_ledger():
    std = Scripted("std", ["veo-3.1-generate-preview"], cost=2.5)
    ledger = CostLedger()
    svc = _service(_registry([std]), ledger)
    key = KeyConfig(id="t", key="")
    scope = RoutingDirective(budget=BudgetDirective(scope="s1"))

    async def scenario():
        task = await svc.create(_video(), key=key, routing=scope)
        await svc._supervisor.wait_for_terminal(task.task_id, provider="std", timeout=2)

    before = METRICS.counters.get("gateway_cost_usd_total", {}).copy()
    asyncio.run(scenario())
    assert ledger.scope_state(key, "s1")["spent_usd"] == pytest.approx(2.5)
    after = METRICS.counters["gateway_cost_usd_total"]
    label = (("key", "t"), ("modality", "video"), ("model", "veo-3.1-generate-preview"),
             ("source", "provider"))
    assert after[label] - before.get(label, 0.0) == pytest.approx(2.5)


def test_scope_budget_refuses_when_the_scope_is_spent():
    reg, *_ = _priced_registry()
    ledger = CostLedger()
    key = KeyConfig(id="t", key="")
    directive = RoutingDirective(optimize="cost", budget=BudgetDirective(scope="p", limit_usd=1.0))
    hold = ledger.hold(key, scope="p", client_limit=1.0, model="x", modality="video", amount=0.9)
    assert isinstance(hold, int)
    with pytest.raises(BudgetExceededError) as err:
        _plan(reg, _video(width=1280, height=720), key=key, routing=directive, ledger=ledger)
    assert err.value.details["budget"]["budget"] == "scope"
    assert err.value.details["budget"]["scope"] == "p"


def test_unpriced_models_can_serve_budgets_when_the_operator_allows_it():
    free = Scripted("unpriced", ["mystery-video"])
    key = KeyConfig(id="t", key="", budget=KeyBudget(limit_usd=1.0))
    with pytest.raises(CostLimitExceededError):
        _plan(_registry([free]), _video(), key=key)
    reg = _registry([free], settings=Settings(keys=[key], budget_allow_unpriced=True))
    assert [c.model for c in _plan(reg, _video(), key=key).candidates] == ["mystery-video"]


# --------------------------------------------------------------------------- #
# HTTP: task routing info, estimates, usage, listings, problems, metrics
# --------------------------------------------------------------------------- #


def _app(providers: list[Scripted], **settings_kw):
    keys = settings_kw.pop("keys", [KeyConfig(id="test", key="")])
    cfgs = [BackendConfig(name=p.name, type="scripted", api_key="k") for p in providers]
    settings = Settings(backends=cfgs, keys=keys, poll_interval=0.01, max_sync_wait=2.0,
                        video_sync_default=False, **settings_kw)
    app = create_app(settings)
    reg = app.state.registry
    for p, cfg in zip(providers, cfgs, strict=True):
        reg._backends[p.name] = p
        reg._configs[p.name] = cfg
        reg._backend_accounts[p.name] = ["default"]
        reg._accounts[(p.name, "default")] = p
    reg.catalog = Catalog(parse_overrides(settings.catalog_models), today=lambda: TODAY)
    return app


_BODY = {"input": [{"type": "text", "text": "a lighthouse at dusk"}],
         "parameters": {"duration_seconds": 8, "dimensions": {"width": 1280, "height": 720},
                        "include_audio": True}}


def _poll(client: TestClient, url: str) -> dict:
    for _ in range(200):
        body = client.get(url).json()
        if body["status"] in ("succeeded", "failed", "cancelled", "expired"):
            return body
    raise AssertionError("task did not finish")


def test_task_resource_reports_routing_and_the_estimated_cost():
    app = _app([Scripted("silent", ["veo-2.0-generate-001"]),
                Scripted("loud", ["veo-3.1-generate-preview"])])
    with TestClient(app) as client:
        created = client.post("/v1/videos", json={
            **_BODY, "routing": {"budget": {"scope": "rideo:prj_1", "limit_usd": 50}}})
        assert created.status_code == 202
        body = created.json()
        assert body["model"] == "veo-3.1-generate-preview"
        assert body["routing"]["requested_model"] == "auto"
        assert body["routing"]["fallback"] is False
        assert body["routing"]["estimated_cost"] == pytest.approx(3.2)
        assert body["routing"]["budget"]["scope"] == "rideo:prj_1"
        assert body["routing"]["budget"]["reserved_usd"] == pytest.approx(3.2)
        done = _poll(client, body["links"]["self"])
        assert done["usage"]["cost"] == pytest.approx(3.2)
        assert done["usage"]["cost_source"] == "estimate"
        assert done["usage"]["currency"] == "USD"
        usage = client.get("/v1/usage").json()
        assert usage["object"] == "usage" and usage["period"]["kind"] == "month"
        assert usage["key"]["spent_usd"] == pytest.approx(3.2)
        assert usage["scopes"] == [{"scope": "rideo:prj_1", "limit_usd": 50.0, "spent_usd": 3.2,
                                    "reserved_usd": 0.0, "remaining_usd": 46.8, "tasks": 1}]
        assert usage["models"] == [{"model": "veo-3.1-generate-preview", "modality": "video",
                                    "spent_usd": 3.2, "tasks": 1}]
        assert client.get("/v1/usage", params={"scope": "nope"}).json()["scopes"][0]["spent_usd"] == 0.0


def test_fallback_stamps_the_served_model():
    app = _app([Scripted("a", ["veo-3.1-generate-preview"],
                         [ProviderRequestError("429", status_code=429, provider="a")]),
                Scripted("b", ["veo-3.1-fast-generate-preview"])])
    with TestClient(app) as client:
        body = client.post("/v1/videos", json={
            **_BODY, "model": "veo-3.1-generate-preview", "routing": {"fallback": "any"}}).json()
    assert body["model"] == "veo-3.1-fast-generate-preview"
    assert body["routing"]["requested_model"] == "veo-3.1-generate-preview"
    assert body["routing"]["fallback_reason"] == "rate_limited"


def test_estimate_endpoint_explains_the_route_without_creating_a_task():
    loud = Scripted("loud", ["veo-3.1-generate-preview"])
    fast = Scripted("fast", ["veo-3.1-fast-generate-preview"])
    app = _app([Scripted("openai-video", ["sora-2"]), Scripted("silent", ["veo-2.0-generate-001"]),
                loud, fast])
    with TestClient(app) as client:
        res = client.post("/v1/videos/estimate", json={
            **_BODY, "routing": {"optimize": "cost", "max_cost_usd": 2}})
        assert res.status_code == 200
        assert res.headers["cache-control"] == "no-store"
        body = res.json()
    assert body["object"] == "estimate" and body["currency"] == "USD"
    assert body["model"] == "veo-3.1-fast-generate-preview"
    assert body["estimated_cost"] == pytest.approx(0.8)
    by_model = {c["model"]: c for c in body["candidates"]}
    assert by_model["veo-3.1-fast-generate-preview"]["admissible"] is True
    assert by_model["veo-3.1-generate-preview"]["reason"] == "max_cost"
    assert by_model["sora-2"] == {"model": "sora-2", "lifecycle": "retired", "admissible": False,
                                  "reason": "retired"}
    assert by_model["veo-2.0-generate-001"]["reason"] == "limits"
    assert loud.calls == [] and fast.calls == []


def test_budget_refusals_are_402_problems_with_the_binding_budget():
    app = _app([Scripted("loud", ["veo-3.1-generate-preview"])],
               keys=[KeyConfig(id="test", key="", budget=KeyBudget(limit_usd=1.0, period="day"))])
    with TestClient(app) as client:
        res = client.post("/v1/videos", json=_BODY)
        assert res.status_code == 402
        problem = res.json()
        assert problem["code"] == "budget_exceeded"
        assert problem["budget"]["limit_usd"] == 1.0
        assert problem["budget"]["estimated_cost"] == pytest.approx(3.2)
        assert "veo" not in problem["detail"] and "loud" not in problem["detail"]
        assert client.post("/v1/videos", json={**_BODY, "routing": {"max_cost_usd": 1}}).json()["code"] \
            == "cost_limit_exceeded"
        assert client.post("/v1/videos", json={**_BODY, "model": "sora-2"}).status_code == 404
        metrics = client.get("/metrics").text
    assert 'gateway_budget_limit_usd{key="test"} 1.0' in metrics
    assert "gateway_routing_decisions_total" in metrics
    assert 'gateway_budget_rejections_total{budget="key",key="test"}' in metrics


def test_retired_models_leave_the_catalogue_but_stay_in_limits():
    app = _app([Scripted("openai-video", ["sora-2"]), Scripted("loud", ["veo-3.1-generate-preview"])])
    with TestClient(app) as client:
        models = {m["id"] for m in client.get("/v1/models", params={"modality": "video"}).json()["data"]}
        limits = {m["id"]: m["limits"] for m in
                  client.get("/v1/models/limits", params={"modality": "video"}).json()["data"]}
        pinned = client.post("/v1/videos", json={**_BODY, "model": "sora-2"})
    assert "sora-2" not in models and "veo-3.1-generate-preview" in models
    assert limits["sora-2"]["lifecycle"] == "retired"
    assert limits["veo-3.1-generate-preview"]["supports_audio_output"] is True
    assert limits["veo-3.1-generate-preview"]["price"]["per_second_tiers"][0] == {
        "max_longest_side": 1920, "per_second": 0.4}
    assert pinned.status_code == 410 and pinned.json()["code"] == "model_retired"
