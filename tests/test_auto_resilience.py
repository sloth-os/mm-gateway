"""Auto-mode resilience: selection metrics, multi-credential backends,
registry candidate enumeration, retry-across-backends, and /metrics exposure.

These exercise the end-to-end behaviour the goal described:

* one backend fronts multiple keys/accounts (BackendConfig.credentials);
* auto mode ranks candidates by live history (latency, success rate,
  rate-limit cooldown) kept in-process (selection store) and rendered to
  Prometheus on /metrics;
* auto mode retries a request on the next backend/account when the chosen
  one fails in a retryable way (rate limit / timeout / 5xx), so an unstable
  backend doesn't surface to the client;
* a non-retryable failure (client 4xx / unsupported feature) propagates at
  once instead of wasting the other backends.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mm_gateway.config import BackendConfig, KeyConfig, Settings
from mm_gateway.core.base import ImageProvider
from mm_gateway.core.exceptions import (
    GatewayError,
    ProviderRequestError,
    ProviderTimeoutError,
)
from mm_gateway.observability.selection import STORE as SELECTION_STORE
from mm_gateway.registry import Registry
from mm_gateway.schemas.image import UnifiedImageRequest, UnifiedImageTask, text_part
from mm_gateway.services import ImageService
from mm_gateway.services_selection import retry_across_backends


# --------------------------------------------------------------------------- #
# Selection store
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _clean_selection_store():
    """Each test starts with an empty, freshly-configured store."""
    SELECTION_STORE.clear()
    SELECTION_STORE.configure(
        rate_limit_cooldown_s=60.0,
        latency_half_life_s=300.0,
        outcome_half_life_s=300.0,
    )
    _PerAccountScripted.behaviours = {}
    _PerAccountScripted.tried_order = []
    yield
    SELECTION_STORE.clear()
    _PerAccountScripted.behaviours = {}
    _PerAccountScripted.tried_order = []


def test_score_is_neutral_with_no_history():
    # An untried backend is NOT penalised: it ranks as neutral 0.5 so a fresh
    # candidate can be picked ahead of a failing one.
    assert SELECTION_STORE.score(backend="b", account="a", model="m", modality="image") == 0.5


def test_observe_success_raises_score():
    SELECTION_STORE.observe(backend="b", account="a", model="m", modality="image",
                            outcome="success", latency_s=0.5)
    assert SELECTION_STORE.score(backend="b", account="a", model="m", modality="image") > 0.5


def test_observe_failures_sink_score_below_neutral():
    for _ in range(5):
        SELECTION_STORE.observe(backend="b", account="a", model="m", modality="image",
                                outcome="failure", latency_s=1.0)
    assert SELECTION_STORE.score(backend="b", account="a", model="m", modality="image") < 0.5


def test_rate_limit_arms_cooldown_and_zeros_score():
    SELECTION_STORE.observe(backend="b", account="a", model="m", modality="image",
                            outcome="failure", latency_s=0.1, rate_limited=True)
    assert SELECTION_STORE.is_rate_limited(backend="b", account="a", model="m",
                                           modality="image") is True
    # A rate-limited candidate is gated out of selection -> score 0.
    assert SELECTION_STORE.score(backend="b", account="a", model="m", modality="image") == 0.0


def test_cooldown_expires_after_window():
    SELECTION_STORE.observe(backend="b", account="a", model="m", modality="image",
                            outcome="failure", rate_limited=True)
    assert SELECTION_STORE.is_rate_limited(backend="b", account="a", model="m",
                                           modality="image") is True
    # Advance the store's clock past the cooldown window.
    SELECTION_STORE._tick(61.0)
    assert SELECTION_STORE.is_rate_limited(backend="b", account="a", model="m",
                                           modality="image") is False


def test_latency_share_pushes_slow_backends_down():
    # Two healthy backends; the slow one should score lower than the fast one
    # thanks to the latency component.
    SELECTION_STORE.observe(backend="fast", account="default", model="m",
                            modality="image", outcome="success", latency_s=0.2)
    SELECTION_STORE.observe(backend="slow", account="default", model="m",
                            modality="image", outcome="success", latency_s=8.0)
    f = SELECTION_STORE.score(backend="fast", account="default", model="m", modality="image")
    s = SELECTION_STORE.score(backend="slow", account="default", model="m", modality="image")
    assert f > s


def test_render_prometheus_exposes_health_metrics():
    SELECTION_STORE.observe(backend="b", account="a", model="m", modality="image",
                            outcome="success", latency_s=0.4)
    out = SELECTION_STORE.render_prometheus()
    assert "gateway_selection_success_rate" in out
    assert "gateway_selection_latency_seconds" in out
    assert "gateway_selection_rate_limited" in out
    assert "gateway_selection_attempts" in out
    assert 'backend="b"' in out
    assert 'account="a"' in out


# --------------------------------------------------------------------------- #
# Multi-credential config
# --------------------------------------------------------------------------- #


def test_backend_with_no_credentials_has_single_default_account():
    cfg = BackendConfig(name="b", type="openai", api_key="k", extra={"x": 1})
    assert cfg.accounts() == [("default", "k", None, {"x": 1})]


def test_backend_with_credentials_enumerates_accounts():
    cfg = BackendConfig(
        name="b", type="openai", api_key="ignored", extra={"shared": 1},
        credentials=[
            {"id": "acct-a", "api_key": "ka", "base_url": "https://a"},
            {"api_key": "kb"},
        ],
    )
    accounts = cfg.accounts()
    assert accounts[0] == ("acct-a", "ka", "https://a", {"shared": 1})
    # Second account has a synthetic id and inherits the backend-level extra.
    aid, key, base, extra = accounts[1]
    assert aid == "account-1"
    assert key == "kb"
    assert base is None
    assert extra == {"shared": 1}


def test_credential_extra_merges_over_backend_extra():
    cfg = BackendConfig(
        name="b", type="openai", api_key="k", extra={"shared": 1, "over": "base"},
        credentials=[{"api_key": "k1", "extra": {"over": "acct", "only": 2}}],
    )
    _aid, _key, _base, extra = cfg.accounts()[0]
    assert extra == {"shared": 1, "over": "acct", "only": 2}


def test_configured_gate_uses_credentials_when_present():
    # A backend whose top-level api_key is empty but carries credentials with a
    # real key is still configured.
    cfg = BackendConfig(name="b", type="openai", api_key=None,
                        credentials=[{"api_key": "k1"}, {"api_key": "k2"}])
    assert cfg.configured is True
    # And one with only empty credential keys is not.
    cfg_empty = BackendConfig(name="b", type="openai", api_key=None,
                              credentials=[{"api_key": ""}, {"api_key": None}])
    assert cfg_empty.configured is False


def test_credential_list_parses_comma_and_newline_blob():
    from mm_gateway.config import _credential_list
    assert _credential_list("a,b \nc") == [
        {"api_key": "a"}, {"api_key": "b"}, {"api_key": "c"},
    ]
    assert _credential_list("  ") == []


def test_legacy_env_plural_keys_build_multi_account_backend(monkeypatch):
    monkeypatch.setenv("OPENAI_IMAGE_API_KEYS", "k1,k2,k3")
    monkeypatch.delenv("OPENAI_IMAGE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings.from_env()
    oai = settings.backend("openai")
    assert oai is not None
    accounts = oai.accounts()
    assert [a[1] for a in accounts] == ["k1", "k2", "k3"]
    assert [a[0] for a in accounts] == ["account-0", "account-1", "account-2"]


# --------------------------------------------------------------------------- #
# Registry: credential-aware build + candidate enumeration
# --------------------------------------------------------------------------- #


class _RecordingImage(ImageProvider):
    """An image provider that records which credential it was built with."""

    image_models = ["gpt-image-1"]
    name = "openai"

    def __init__(self, backend: Any):
        super().__init__(backend)
        self.calls: list[UnifiedImageRequest] = []
        # Account id is stashed on extra by the registry.
        self.account_id = backend.extra.get("__account_id", "default")

    async def create_image_task(self, request, *, sync=None):
        self.calls.append(request)
        return UnifiedImageTask(task_id=f"t-{self.account_id}-{len(self.calls)}",
                                provider=self.name, model=request.model, status="succeeded")

    async def get_image_task(self, task_id):
        return UnifiedImageTask(task_id=task_id, provider=self.name,
                                model="gpt-image-1", status="succeeded")


def _build_registry_with_accounts(monkeypatch):
    monkeypatch.setattr(
        "mm_gateway.registry._PROVIDER_CLASSES", {"openai": "_RecordingImage"},
    )
    # _RecordingImage lives in this test module; importlib.import_module(
    # "mm_gateway.providers.openai") would fail, so patch the import to return
    # this module. Restored by monkeypatch at teardown.
    import sys
    monkeypatch.setitem(sys.modules, "mm_gateway.providers.openai", sys.modules[__name__])
    cfg = BackendConfig(
        name="oai", type="openai", extra={},
        credentials=[{"id": "prod", "api_key": "k-prod"},
                     {"id": "staging", "api_key": "k-stg"}],
    )
    settings = Settings(backends=[cfg], keys=[KeyConfig(id="t", key="")])
    return Registry(settings), cfg


def test_registry_builds_one_provider_per_account(monkeypatch):
    reg, _cfg = _build_registry_with_accounts(monkeypatch)
    assert reg._backend_accounts["oai"] == ["prod", "staging"]
    assert reg._accounts[("oai", "prod")].backend.api_key == "k-prod"
    assert reg._accounts[("oai", "staging")].backend.api_key == "k-stg"
    # The default (first) account backs the legacy accessor.
    assert reg._backends["oai"].backend.api_key == "k-prod"


def test_enumerate_auto_candidates_lists_one_entry_per_account(monkeypatch):
    reg, _cfg = _build_registry_with_accounts(monkeypatch)
    req = UnifiedImageRequest(model="auto", content=[text_part("hi")])
    cands = reg.enumerate_auto_candidates(req, KeyConfig(id="t", key=""),
                                          modality="image")
    # Two accounts -> two candidates for the one fitting model.
    assert len(cands) == 2
    accounts = {aid for _p, aid, _m, _b in cands}
    assert accounts == {"prod", "staging"}
    # Both advertise the same fitting model.
    assert {m for _p, _a, m, _b in cands} == {"gpt-image-1"}


def test_enumerate_ranks_healthy_candidate_above_rate_limited(monkeypatch):
    reg, _cfg = _build_registry_with_accounts(monkeypatch)
    req = UnifiedImageRequest(model="auto", content=[text_part("hi")])
    key = KeyConfig(id="t", key="")
    # Drive the prod account into a rate-limit cooldown.
    SELECTION_STORE.observe(backend="oai", account="prod", model="gpt-image-1",
                            modality="image", outcome="failure", rate_limited=True)
    cands = reg.enumerate_auto_candidates(req, key, modality="image")
    # The healthy staging account must rank first; prod sinks to the bottom.
    assert cands[0][1] == "staging"
    assert cands[-1][1] == "prod"


# --------------------------------------------------------------------------- #
# retry_across_backends
# --------------------------------------------------------------------------- #


class _ScriptedImage(ImageProvider):
    """An image provider whose create_image_task follows a scripted behaviour list.

    Each entry is either a UnifiedImageTask (success) or an Exception to raise.
    Records the call order so a test can assert which accounts were tried.
    """

    image_models = ["gpt-image-1"]
    name = "openai"

    def __init__(self, backend: Any, behaviours):
        super().__init__(backend)
        self._behaviours = list(behaviours)
        self.account_id = backend.extra.get("__account_id", "default")
        self.tried: list[str] = []

    async def create_image_task(self, request, *, sync=None):
        self.tried.append(self.account_id)
        if not self._behaviours:
            raise RuntimeError("no more scripted behaviours")
        item = self._behaviours.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    async def get_image_task(self, task_id):  # pragma: no cover - sync path off
        raise RuntimeError("not used")


def _candidate(prov, account_id, model="gpt-image-1", backend="oai"):
    return (prov, account_id, model, backend)


def test_retry_succeeds_on_second_candidate_after_retryable_failure():
    failing = _ScriptedImage(BackendConfig(name="oai", type="openai", api_key="k",
                                            extra={"__account_id": "a"}),
                              [ProviderRequestError("429", status_code=429, provider="oai")])
    ok = _ScriptedImage(BackendConfig(name="oai", type="openai", api_key="k",
                                       extra={"__account_id": "b"}),
                         [UnifiedImageTask(task_id="ok-1", provider="oai",
                                           model="gpt-image-1", status="succeeded")])
    cands = [_candidate(failing, "a"), _candidate(ok, "b")]

    async def attempt(prov, account_id, model, backend):
        return await prov.create_image_task(UnifiedImageRequest(model=model, content=[]))

    result = asyncio.run(retry_across_backends(candidates=cands, attempt=attempt,
                                                modality="image", key=None))
    assert result.task_id == "ok-1"
    assert failing.tried == ["a"]
    assert ok.tried == ["b"]


def test_retry_stops_immediately_on_non_retryable_client_error():
    # A validation_error / 4xx is terminal — the other candidate must NOT be tried.
    terminal = _ScriptedImage(BackendConfig(name="oai", type="openai", api_key="k",
                                             extra={"__account_id": "a"}),
                               [GatewayError("bad prompt", code="validation_error",
                                             status_code=422)])
    ok = _ScriptedImage(BackendConfig(name="oai", type="openai", api_key="k",
                                       extra={"__account_id": "b"}),
                         [UnifiedImageTask(task_id="ok", provider="oai",
                                           model="gpt-image-1", status="succeeded")])
    cands = [_candidate(terminal, "a"), _candidate(ok, "b")]

    async def attempt(prov, account_id, model, backend):
        return await prov.create_image_task(UnifiedImageRequest(model=model, content=[]))

    with pytest.raises(GatewayError) as exc_info:
        asyncio.run(retry_across_backends(candidates=cands, attempt=attempt,
                                          modality="image", key=None))
    assert exc_info.value.code == "validation_error"
    assert ok.tried == []  # never reached


def test_retry_re_raises_last_error_when_all_candidates_exhausted():
    a = _ScriptedImage(BackendConfig(name="oai", type="openai", api_key="k",
                                      extra={"__account_id": "a"}),
                       [ProviderTimeoutError("slow", provider="oai")])
    b = _ScriptedImage(BackendConfig(name="oai", type="openai", api_key="k",
                                      extra={"__account_id": "b"}),
                       [ProviderRequestError("502", status_code=502, provider="oai")])
    cands = [_candidate(a, "a"), _candidate(b, "b")]

    async def attempt(prov, account_id, model, backend):
        return await prov.create_image_task(UnifiedImageRequest(model=model, content=[]))

    with pytest.raises(ProviderRequestError):
        asyncio.run(retry_across_backends(candidates=cands, attempt=attempt,
                                          modality="image", key=None))
    assert a.tried == ["a"]
    assert b.tried == ["b"]


def test_retry_records_outcomes_in_selection_store():
    failing = _ScriptedImage(BackendConfig(name="oai", type="openai", api_key="k",
                                            extra={"__account_id": "a"}),
                              [ProviderRequestError("429", status_code=429, provider="oai")])
    ok = _ScriptedImage(BackendConfig(name="oai", type="openai", api_key="k",
                                       extra={"__account_id": "b"}),
                         [UnifiedImageTask(task_id="ok", provider="oai",
                                           model="gpt-image-1", status="succeeded")])
    cands = [_candidate(failing, "a"), _candidate(ok, "b")]

    async def attempt(prov, account_id, model, backend):
        return await prov.create_image_task(UnifiedImageRequest(model=model, content=[]))

    asyncio.run(retry_across_backends(candidates=cands, attempt=attempt,
                                      modality="image", key=None))
    # The failed (rate-limited) account is now in cooldown; the ok one is healthy.
    assert SELECTION_STORE.is_rate_limited(backend="oai", account="a", model="gpt-image-1",
                                           modality="image") is True
    assert SELECTION_STORE.score(backend="oai", account="b", model="gpt-image-1",
                                  modality="image") > 0.5


def test_retry_empty_candidates_raises_validation_error():
    async def attempt(prov, account_id, model, backend):  # pragma: no cover
        raise AssertionError("should not be called")

    with pytest.raises(GatewayError) as exc_info:
        asyncio.run(retry_across_backends(candidates=[], attempt=attempt,
                                          modality="image", key=None))
    assert exc_info.value.status_code == 422


# --------------------------------------------------------------------------- #
# Services: auto mode retries across backends end-to-end
# --------------------------------------------------------------------------- #


class _PerAccountScripted(ImageProvider):
    """A shared scripted provider keyed by the account it was built for.

    ``behaviours`` maps account_id -> list of outcomes (Exception to raise or
    task to return), popped in order; ``tried_order`` records the call sequence
    across all instances so a test can assert retry order across accounts.
    """

    image_models = ["gpt-image-1"]
    name = "openai"
    behaviours: dict[str, list[Any]] = {}
    tried_order: list[str] = []

    def __init__(self, backend: Any):
        super().__init__(backend)
        self.account_id = backend.extra.get("__account_id", "default")

    async def create_image_task(self, request, *, sync=None):
        _PerAccountScripted.tried_order.append(self.account_id)
        seq = _PerAccountScripted.behaviours.get(self.account_id, [])
        if not seq:
            raise RuntimeError(f"no scripted behaviour for {self.account_id}")
        item = seq.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    async def get_image_task(self, task_id):  # pragma: no cover - sync off
        raise RuntimeError("not used")


class _FlakyThenOKImage(ImageProvider):
    """A registry-wide image provider whose first N calls fail then succeed.

    Used to back two *backends* (not just accounts) so we can assert the service
    tries backend B when backend A fails — i.e. retry across *providers*, not
    only across accounts of one backend.
    """

    image_models = ["gpt-image-1"]
    name = "openai"

    def __init__(self, backend: Any, behaviour):
        super().__init__(backend)
        self.behaviour = behaviour
        self.account_id = backend.extra.get("__account_id", "default")
        self.calls = 0

    async def create_image_task(self, request, *, sync=None):
        self.calls += 1
        item = self.behaviour
        if isinstance(item, BaseException):
            raise item
        return item

    async def get_image_task(self, task_id):  # pragma: no cover - sync off
        raise RuntimeError("not used")


def _registry_two_backends(prov_a, prov_b, monkeypatch):
    """Inject two single-account image backends into a registry, bypassing build.

    The registry's ``_build`` would try to import the real provider module, so
    we construct it with a no-op build (the providers are injected directly,
    mirroring the conftest ``app`` fixture).
    """
    cfg_a = BackendConfig(name="aaa", type="openai", api_key="ka")
    cfg_b = BackendConfig(name="bbb", type="openai", api_key="kb")
    settings = Settings(backends=[cfg_a, cfg_b], keys=[KeyConfig(id="t", key="")])
    reg = Registry.__new__(Registry)
    reg.settings = settings
    reg._backends = {}
    reg._configs = {}
    reg._accounts = {}
    reg._backend_accounts = {}
    reg._aliases = {}
    reg._backends["aaa"] = prov_a
    reg._backends["bbb"] = prov_b
    reg._configs["aaa"] = cfg_a
    reg._configs["bbb"] = cfg_b
    reg._backend_accounts["aaa"] = ["default"]
    reg._backend_accounts["bbb"] = ["default"]
    reg._accounts[("aaa", "default")] = prov_a
    reg._accounts[("bbb", "default")] = prov_b
    return reg


def test_image_service_auto_retries_next_backend_on_retryable_failure(monkeypatch):
    fail = _FlakyThenOKImage(
        BackendConfig(name="aaa", type="openai", api_key="k", extra={"__account_id": "default"}),
        ProviderRequestError("502", status_code=502, provider="aaa"),
    )
    ok = _FlakyThenOKImage(
        BackendConfig(name="bbb", type="openai", api_key="k", extra={"__account_id": "default"}),
        UnifiedImageTask(task_id="ok-1", provider="bbb", model="gpt-image-1", status="succeeded"),
    )
    reg = _registry_two_backends(fail, ok, monkeypatch)
    svc = ImageService(reg, max_sync_wait=1.0, poll_interval=0.01, sync_default=False)
    req = UnifiedImageRequest(model="auto", content=[text_part("hi")])

    task = asyncio.run(svc.create(req, key=KeyConfig(id="t", key="")))
    assert task.task_id == "ok-1"
    assert task.provider == "bbb"
    assert fail.calls == 1
    assert ok.calls == 1


def test_image_service_auto_surfaces_terminal_error_without_trying_others(monkeypatch):
    terminal = _FlakyThenOKImage(
        BackendConfig(name="aaa", type="openai", api_key="k", extra={"__account_id": "default"}),
        GatewayError("unsupported", code="unsupported_feature", status_code=400),
    )
    ok = _FlakyThenOKImage(
        BackendConfig(name="bbb", type="openai", api_key="k", extra={"__account_id": "default"}),
        UnifiedImageTask(task_id="ok-1", provider="bbb", model="gpt-image-1", status="succeeded"),
    )
    reg = _registry_two_backends(terminal, ok, monkeypatch)
    svc = ImageService(reg, max_sync_wait=1.0, poll_interval=0.01, sync_default=False)
    req = UnifiedImageRequest(model="auto", content=[text_part("hi")])

    with pytest.raises(GatewayError) as exc_info:
        asyncio.run(svc.create(req, key=KeyConfig(id="t", key="")))
    assert exc_info.value.code == "unsupported_feature"
    assert ok.calls == 0


def test_image_service_auto_retries_across_accounts_of_one_backend(monkeypatch):
    """One backend, two accounts: the prod key rate-limits, staging succeeds."""
    cfg = BackendConfig(
        name="oai", type="openai", extra={},
        credentials=[{"id": "prod", "api_key": "kp"},
                     {"id": "staging", "api_key": "ks"}],
    )
    # Replace the build so the registry constructs our scripted providers.
    import sys
    monkeypatch.setitem(sys.modules, "mm_gateway.providers.openai", sys.modules[__name__])
    monkeypatch.setattr("mm_gateway.registry._PROVIDER_CLASSES",
                        {"openai": "_PerAccountScripted"})
    settings = Settings(backends=[cfg], keys=[KeyConfig(id="t", key="")])
    # Pre-seed the scripted outcomes for the two accounts.
    _PerAccountScripted.behaviours = {
        "prod": [ProviderRequestError("429", status_code=429, provider="oai")],
        "staging": [UnifiedImageTask(task_id="ok-stg", provider="oai",
                                     model="gpt-image-1", status="succeeded")],
    }
    reg = Registry(settings)
    svc = ImageService(reg, max_sync_wait=1.0, poll_interval=0.01, sync_default=False)
    req = UnifiedImageRequest(model="auto", content=[text_part("hi")])

    task = asyncio.run(svc.create(req, key=KeyConfig(id="t", key="")))
    assert task.task_id == "ok-stg"
    assert task.provider == "oai"
    # The prod account was tried first (and failed), staging second (and won).
    assert _PerAccountScripted.tried_order == ["prod", "staging"]


# --------------------------------------------------------------------------- #
# /metrics exposure of selection health
# --------------------------------------------------------------------------- #


def test_metrics_endpoint_exposes_selection_health(settings, app):
    SELECTION_STORE.observe(backend="fake", account="default", model="fake-image-1",
                            modality="image", outcome="success", latency_s=0.3)
    client = TestClient(app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text
    assert "gateway_selection_success_rate" in body
    assert "gateway_selection_latency_seconds" in body
    assert 'backend="fake"' in body
    # The request counters are still emitted too.
    assert "gateway_requests_total" in body
