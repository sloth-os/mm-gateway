"""Tests for the registry's provider/model resolution."""

from __future__ import annotations

import pytest

from mm_gateway.config import BackendConfig, KeyConfig, Settings
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
