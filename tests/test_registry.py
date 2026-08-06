"""Tests for the registry's provider/model resolution."""

from __future__ import annotations

import pytest

from mm_gateway.config import ProviderCredentials, Settings
from mm_gateway.core.exceptions import (
    ModelNotFoundError,
    ProviderNotConfiguredError,
    ProviderNotFoundError,
)
from mm_gateway.registry import Registry


class _FakeProv:
    def __init__(self, creds):
        self.name = "fake"
        self.image_models = ["fake-image-1"]
        self.video_models = ["fake-video-1"]

    supports_image = True
    supports_video = True


@pytest.fixture
def registry(settings):
    reg = Registry(settings)
    reg._providers["fake"] = _FakeProv(ProviderCredentials(name="fake", api_key="k"))
    return reg


def test_resolve_alias_to_unconfigured_provider_raises(registry):
    # gateway-image-pro aliases to the openai provider, which we did not
    # register, so resolution must surface a not-configured error.
    with pytest.raises(ProviderNotConfiguredError):
        registry.resolve("gateway-image-pro")


def test_resolve_by_provider_hint_and_model_passes_through(registry):
    prov, model = registry.resolve("whatever-model", provider="fake")
    assert prov.name == "fake"
    assert model == "whatever-model"


def test_resolve_unknown_provider_raises(registry):
    with pytest.raises(ProviderNotConfiguredError):
        registry.resolve("m", provider="nope")


def test_resolve_known_model_id(registry):
    prov, model = registry.resolve("fake-image-1")
    assert prov.name == "fake"
    assert model == "fake-image-1"


def test_resolve_unknown_model_raises(registry):
    with pytest.raises(ModelNotFoundError):
        registry.resolve("does-not-exist")


def test_list_models(registry):
    models = registry.list_models()
    ids = [m["id"] for m in models]
    assert "fake-image-1" in ids
    assert "fake-video-1" in ids
