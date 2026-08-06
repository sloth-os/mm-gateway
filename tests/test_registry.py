"""Tests for the registry's provider/model resolution."""

from __future__ import annotations

import importlib
import types

import pytest

from mm_gateway.config import BackendConfig, KeyConfig, Settings
from mm_gateway.core.base import ImageProvider
from mm_gateway.core.exceptions import (
    ForbiddenError,
    ModelNotFoundError,
)
from mm_gateway.registry import Registry


class _FakeProv:
    def __init__(self, cfg):
        self.name = "fake"
        self.image_models = ["fake-image-1"]
        self.video_models = ["fake-video-1"]
        self.backend = cfg

    supports_image = True
    supports_video = True


@pytest.fixture
def key() -> KeyConfig:
    return KeyConfig(id="test", key="")


@pytest.fixture
def settings() -> Settings:
    return Settings(backends=[BackendConfig(name="fake", type="fake", api_key="k")],
                    keys=[KeyConfig(id="test", key="")])


@pytest.fixture
def registry(settings):
    reg = Registry(settings)
    cfg = settings.backends[0]
    reg._backends["fake"] = _FakeProv(cfg)
    reg._configs["fake"] = cfg
    return reg


def test_resolve_alias_to_unconfigured_provider_raises(registry, key):
    # gateway-image-pro aliases to the openai provider, which we did not
    # register, so resolution must surface a not-found error.
    with pytest.raises(ModelNotFoundError):
        registry.resolve("gateway-image-pro", key)


def test_resolve_by_backend_name_passes_through(registry, key):
    prov, model, backend = registry.resolve("whatever-model", key, backend_name="fake")
    assert prov.name == "fake"
    assert model == "whatever-model"
    assert backend == "fake"


def test_resolve_unknown_backend_name_falls_through(registry, key):
    # "nope" is not a registered backend, so it is not in candidates; resolution
    # falls back to the only usable backend that serves the model.
    prov, model, backend = registry.resolve("fake-image-1", key, backend_name="nope")
    assert backend == "fake"


def test_resolve_known_model_id(registry, key):
    prov, model, backend = registry.resolve("fake-image-1", key)
    assert prov.name == "fake"
    assert model == "fake-image-1"
    assert backend == "fake"


def test_resolve_unknown_model_raises(registry, key):
    with pytest.raises(ModelNotFoundError):
        registry.resolve("does-not-exist", key)


def test_list_models(registry, key):
    models = registry.list_models(key)
    ids = [m["id"] for m in models]
    assert "fake-image-1" in ids
    assert "fake-video-1" in ids


def test_key_with_no_usable_backend_is_forbidden(settings):
    # A key that denies the only backend may not resolve anything.
    deny_key = KeyConfig(id="deny", key="d", deny_tags=["fake"])
    reg = Registry(settings)
    cfg = settings.backends[0]
    reg._backends["fake"] = _FakeProv(cfg)
    reg._configs["fake"] = cfg
    with pytest.raises(ForbiddenError):
        reg.resolve("fake-image-1", deny_key)


def test_build_appends_pinned_image_model(monkeypatch):
    # The legacy env layout records a *_MODEL as BackendConfig.extra[
    # "image_model"]; _build must append it to that backend's image_models so an
    # operator can reach a model id not in the provider's hardcoded catalogue.
    class StubImageProvider(ImageProvider):
        name = "stub"
        image_models = ["stub-image-1"]
        video_models = []

        def __init__(self, backend):
            self.backend = backend

        async def generate_image(self, request):  # pragma: no cover - not called
            raise NotImplementedError

    stub_module = types.ModuleType("mm_gateway.providers.stub")
    stub_module.StubProvider = StubImageProvider

    def fake_import(name):
        return stub_module if name == "mm_gateway.providers.stub" else _real_import_module(name)
    monkeypatch.setattr(importlib, "import_module", fake_import)
    import mm_gateway.registry as registry_mod
    monkeypatch.setattr(registry_mod, "_PROVIDER_CLASSES", {"stub": "StubProvider"})

    settings = Settings(
        backends=[BackendConfig(name="stub", type="stub", api_key="k",
                                 extra={"image_model": "brand-new-model"})],
        keys=[KeyConfig(id="test", key="")],
    )
    reg = Registry(settings)
    prov = reg.get("stub")
    # The pinned model is appended without disturbing the hardcoded list, and
    # video_models is untouched.
    assert prov.image_models == ["stub-image-1", "brand-new-model"]
    assert prov.video_models == []
    # The pinned model is now resolvable and listed.
    p, real_model, backend = reg.resolve("brand-new-model", KeyConfig(id="test", key=""))
    assert backend == "stub" and real_model == "brand-new-model"
    assert "brand-new-model" in {m["id"] for m in reg.list_models()}


def test_build_no_extra_model_leaves_list_unchanged(monkeypatch):
    # Without extra["image_model"], _build must not touch the provider's list.
    class StubImageProvider(ImageProvider):
        name = "stub"
        image_models = ["stub-image-1"]
        video_models = []

        def __init__(self, backend):
            self.backend = backend

        async def generate_image(self, request):  # pragma: no cover - not called
            raise NotImplementedError

    stub_module = types.ModuleType("mm_gateway.providers.stub")
    stub_module.StubProvider = StubImageProvider

    def fake_import(name):
        return stub_module if name == "mm_gateway.providers.stub" else _real_import_module(name)
    monkeypatch.setattr(importlib, "import_module", fake_import)
    import mm_gateway.registry as registry_mod
    monkeypatch.setattr(registry_mod, "_PROVIDER_CLASSES", {"stub": "StubProvider"})

    settings = Settings(
        backends=[BackendConfig(name="stub", type="stub", api_key="k")],
        keys=[KeyConfig(id="test", key="")],
    )
    reg = Registry(settings)
    assert reg.get("stub").image_models == ["stub-image-1"]


# Capture the real import so the monkeypatch only reroutes the stub path and
# leaves every other import (pytest internals, etc.) intact.
_real_import_module = importlib.import_module
