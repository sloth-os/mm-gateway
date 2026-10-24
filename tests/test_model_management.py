"""Model allowlists, alias routing and draft upstream discovery through the admin API."""

import copy
import dataclasses
import json

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from mm_gateway.config import Settings
from mm_gateway.core.exceptions import ModelNotFoundError
from mm_gateway.server.app import create_app
from tests.test_management import ADMIN, ROOT, USER, admin_client, config, management_app


def save_backend(client, **changes):
    body, headers = config(client)
    backend = body["config"]["backends"][0]
    backend.update(changes)
    response = client.put(ROOT + "/backends/fake", headers=headers, json=backend)
    assert response.status_code == 200, response.text
    return response


def public_ids(client):
    return {entry["id"] for entry in client.get("/v1/models", headers=USER).json()["data"]}


def test_allowlist_alias_and_future_model_reach_actual_upstream(admin_client, management_app):
    save_backend(admin_client, models={"image": ["future-image-2027"], "video": ["fake-video-1"]},
                 model_aliases={"image-stable": "future-image-2027", "video-stable": "fake-video-1"})
    assert public_ids(admin_client) == {"future-image-2027", "image-stable", "fake-video-1", "video-stable"}
    provider = management_app.state.registry.get("fake")
    for model in ("future-image-2027", "image-stable"):
        response = admin_client.post("/v1/images", headers=USER,
                                     json={"model": model, "input": [{"type": "text", "text": "example"}]})
        assert response.status_code == 202, response.text
        assert provider.image_calls[-1].model == "future-image-2027"
    video = admin_client.post("/v1/videos", headers=USER,
                             json={"model": "video-stable", "input": [{"type": "text", "text": "example"}]})
    assert video.status_code == 202, video.text
    assert provider.video_calls[-1].model == "fake-video-1"
    for model in ("fake-image-1", "fake/fake-image-1", "video-stable"):
        blocked = admin_client.post("/v1/images", headers=USER,
                                   json={"model": model, "input": [{"type": "text", "text": "blocked"}]})
        assert blocked.status_code == 404, blocked.text
    # Both the legacy resolver and the public route planner must enforce the allowlist.
    with pytest.raises(ModelNotFoundError):
        management_app.state.registry.resolve("fake-image-1", backend_name="fake")
    estimate = admin_client.post("/v1/images/estimate", headers=USER,
                                json={"model": "image-stable", "input": [{"type": "text", "text": "estimate"}]})
    assert estimate.status_code == 200, estimate.text
    auto = admin_client.post("/v1/images", headers=USER,
                            json={"model": "auto", "input": [{"type": "text", "text": "auto"}]})
    assert auto.status_code == 202
    assert provider.image_calls[-1].model == "future-image-2027"
    assert management_app.state.settings.backends[0].api_key == "provider-secret"


def test_empty_allowlist_blocks_explicit_and_auto_requests(admin_client, management_app):
    save_backend(admin_client, models={})
    assert public_ids(admin_client) == set()
    for model in ("fake-image-1", "fake/not-allowed", "auto"):
        response = admin_client.post("/v1/images", headers=USER,
                                     json={"model": model, "input": [{"type": "text", "text": "blocked"}]})
        assert response.status_code == (422 if model == "auto" else 404), response.text
        with pytest.raises(ModelNotFoundError):
            management_app.state.registry.backends_serving("fake-image-1", None, modality="image", backend_name="fake")
    assert not management_app.state.registry.get("fake").image_calls


def test_builtin_alias_cannot_bypass_allowlist(admin_client, management_app, monkeypatch):
    from mm_gateway.registry import _MODEL_ALIASES
    monkeypatch.setitem(_MODEL_ALIASES, "builtin-image", ("fake", "fake-image-1"))
    save_backend(admin_client, models={"image": ["future-image"]})
    assert "builtin-image" not in public_ids(admin_client)
    for resolve in (management_app.state.registry.resolve, management_app.state.registry.backends_serving):
        with pytest.raises(ModelNotFoundError):
            resolve("builtin-image", None, modality="image", backend_name="fake")
    save_backend(admin_client, models={"image": ["fake-image-1"]})
    assert "builtin-image" in public_ids(admin_client)


def test_alias_is_scoped_to_backend_and_key_permissions(admin_client, management_app):
    save_backend(admin_client, models={"image": ["fake-image-1"]}, model_aliases={"private-model": "fake-image-1"})
    _, headers = config(admin_client)
    assert admin_client.put(ROOT + "/backends/other", headers=headers,
                            json={"name": "other", "type": "fake", "api_key": "other-secret"}).status_code == 200
    _, headers = config(admin_client)
    assert admin_client.put(ROOT + "/keys/restricted", headers=headers,
                            json={"id": "restricted", "key": "restricted-token", "allow_backends": ["other"]}).status_code == 200
    restricted = {"Authorization": "Bearer restricted-token"}
    ids = {entry["id"] for entry in admin_client.get("/v1/models", headers=restricted).json()["data"]}
    assert "private-model" not in ids
    response = admin_client.post("/v1/images", headers=restricted,
                                 json={"model": "private-model", "input": [{"type": "text", "text": "blocked"}]})
    assert response.status_code == 404
    save_backend(admin_client, enabled=False)
    assert "private-model" not in public_ids(admin_client)
    with pytest.raises(ModelNotFoundError):
        management_app.state.registry.resolve("private-model")


def test_alias_target_changes_live_and_is_removed_on_delete(admin_client, management_app):
    save_backend(admin_client, models={"image": ["first", "second"]}, model_aliases={"stable": "first"})
    assert management_app.state.registry.resolve("stable")[1:] == ("first", "fake")
    save_backend(admin_client, model_aliases={"stable": "second"})
    assert management_app.state.registry.resolve("stable")[1:] == ("second", "fake")
    save_backend(admin_client, model_aliases={})
    assert "stable" not in public_ids(admin_client)
    with pytest.raises(ModelNotFoundError):
        management_app.state.registry.resolve("stable")


def test_native_namespaced_model_ids_are_preserved(admin_client, management_app):
    save_backend(admin_client, models={"image": ["fake/native-image"]}, model_aliases={"stable": "fake/native-image"})
    registry = management_app.state.registry
    assert registry.resolve("fake/native-image")[1:] == ("fake/native-image", "fake")
    assert registry.resolve("stable")[1:] == ("fake/native-image", "fake")
    assert registry.backends_serving("fake/native-image", None, modality="image")[0] == "fake/native-image"


def test_alias_names_are_public_identifiers_not_credential_field_names(admin_client, management_app):
    aliases = {"token-image": "fake-image-1", "private-image": "fake-image-1"}
    save_backend(admin_client, models={"image": ["fake-image-1"]}, model_aliases=aliases)
    body, _ = config(admin_client)
    backend = body["config"]["backends"][0]
    assert backend["model_aliases"] == aliases
    assert backend["api_key"] == "[redacted]"
    assert management_app.state.registry.resolve("token-image")[1:] == ("fake-image-1", "fake")


def test_alias_limits_estimates_and_retirement_use_target(admin_client):
    save_backend(admin_client, models={"image": ["future-image"]}, model_aliases={"stable": "future-image"})
    body, headers = config(admin_client)
    body["config"]["catalog_models"] = {"future-image": {
        "modality": "image", "max_prompt_chars": 100, "price": {"per_image": 0.25}}}
    assert admin_client.put(ROOT + "/config", headers=headers, json=body["config"]).status_code == 200
    limits = admin_client.get("/v1/models/limits", headers=USER).json()["data"]
    alias = next(entry for entry in limits if entry["id"] == "stable")
    assert alias["limits"]["max_prompt_chars"] == 100
    assert alias["limits"]["price"]["per_image"] == 0.25
    estimate = admin_client.post("/v1/images/estimate", headers=USER,
                                json={"model": "stable", "input": [{"type": "text", "text": "estimate"}]})
    assert estimate.status_code == 200, estimate.text
    assert estimate.json()["estimated_cost"] == 0.25
    body, headers = config(admin_client)
    body["config"]["catalog_models"]["future-image"]["retired_on"] = "2000-01-01"
    assert admin_client.put(ROOT + "/config", headers=headers, json=body["config"]).status_code == 200
    assert "stable" not in public_ids(admin_client)
    response = admin_client.post("/v1/images", headers=USER,
                                 json={"model": "stable", "input": [{"type": "text", "text": "blocked"}]})
    assert response.status_code == 410, response.text


@pytest.mark.parametrize("changes", [
    {"models": {"image": ["duplicate", "duplicate"]}},
    {"models": {"image": ["auto"]}},
    {"models": {"image": ["gateway-image-pro"]}},
    {"models": {"text": ["text-model"]}},
    {"models": {"audio": ["unsupported-audio"]}},
    {"model_aliases": {"auto": "fake-image-1"}},
    {"model_aliases": {"fake-image-1": "fake-video-1"}},
    {"model_aliases": {"gateway-image-pro": "fake-image-1"}},
    {"model_aliases": {"one": "two", "two": "one"}},
    {"models": {"image": ["active"]}, "model_aliases": {"stable": "inactive"}},
])
def test_invalid_model_changes_are_atomic(admin_client, changes):
    body, headers = config(admin_client)
    backend = {**body["config"]["backends"][0], **changes}
    response = admin_client.put(ROOT + "/backends/fake", headers=headers, json=backend)
    assert response.status_code == 422, response.text
    assert config(admin_client)[0] == body


def test_models_and_aliases_survive_restart_and_yaml(management_app, tmp_path):
    settings = dataclasses.replace(management_app.state.settings, management_config_path=str(tmp_path / "managed.json"))
    with TestClient(create_app(settings)) as client:
        save_backend(client, models={"image": ["future-model"]}, model_aliases={"stable": "future-model"})
    restored = create_app(settings)
    assert restored.state.registry.resolve("stable")[1:] == ("future-model", "fake")
    persisted = json.loads((tmp_path / "managed.json").read_text())
    assert persisted["backends"][0]["models"] == {"image": ["future-model"]}
    assert persisted["backends"][0]["api_key"].startswith("enc:v1:")
    yaml = Settings._from_yaml("""
backends:
  - name: configured
    type: fake
    api_key: test
    models:
      image: [future-model]
    model_aliases:
      stable: future-model
""")
    assert create_app(yaml).state.registry.resolve("stable")[1:] == ("future-model", "configured")


def test_builtin_catalog_does_not_construct_provider(admin_client, monkeypatch):
    from tests.conftest import FakeProvider
    def fail(*args):
        raise AssertionError("Built-in model metadata must not construct a provider")
    monkeypatch.setattr(FakeProvider, "__init__", fail)
    response = admin_client.get(ROOT + "/backend-types/fake/models", headers=ADMIN)
    assert response.status_code == 200
    assert response.json() == {"models": {"image": ["fake-image-1"], "video": ["fake-video-1"], "music": ["fake-music-1"]}}
    assert admin_client.get(ROOT + "/backend-types/fake/models", headers=USER).status_code == 401


@respx.mock
def test_discovery_uses_draft_or_saved_secret_and_pagination_without_saving(admin_client):
    before, _ = config(admin_client)
    body = copy.deepcopy(before["config"]["backends"][0])
    body["base_url"] = "https://models.example/v1/"
    first_page = httpx.Response(200, json={"data": [{"id": "fake-image-1"}], "has_more": True})
    second_page = httpx.Response(200, json={"data": [{"id": "future-model"}]})
    upstream = respx.get("https://models.example/v1/models").mock(
        side_effect=[first_page, second_page, first_page, second_page])
    response = admin_client.post(ROOT + "/backends/discover-models", headers=ADMIN, json=body)
    assert response.status_code == 200, response.text
    assert response.json() == {"data": [{"id": "fake-image-1", "modalities": ["image"]}, {"id": "future-model", "modalities": []}]}
    assert upstream.calls[0].request.headers["Authorization"] == "Bearer provider-secret"
    assert upstream.calls[1].request.url.params["after"] == "fake-image-1"
    assert "provider-secret" not in response.text
    assert response.headers["Cache-Control"] == "no-store"
    assert config(admin_client)[0] == before
    body.update(name="unsaved", api_key="draft-key", credentials=[])
    assert admin_client.post(ROOT + "/backends/discover-models", headers=ADMIN, json=body).status_code == 200
    assert upstream.calls[-1].request.headers["Authorization"] == "Bearer draft-key"
    assert admin_client.post(ROOT + "/backends/discover-models", headers=USER, json=body).status_code == 401


@respx.mock
def test_discovery_pool_auth_custom_path_and_sanitized_error(admin_client):
    body = {"name": "draft", "type": "fake", "base_url": "https://models.example/api",
            "credentials": [{"id": "primary", "api_key": "pool-secret"}],
            "extra": {"models_path": "available?api-version=test", "models_headers": {"X-Project": "project-secret"}}}
    route = respx.get("https://models.example/api/available").mock(
        return_value=httpx.Response(200, json=["fake-image-1", "future-model"]))
    response = admin_client.post(ROOT + "/backends/discover-models", headers=ADMIN, json=body)
    assert response.status_code == 200, response.text
    assert route.calls[0].request.headers["Authorization"] == "Bearer pool-secret"
    assert route.calls[0].request.headers["X-Project"] == "project-secret"
    assert route.calls[0].request.url.params["api-version"] == "test"
    route.mock(return_value=httpx.Response(401, text="private pool-secret upstream failure"))
    response = admin_client.post(ROOT + "/backends/discover-models", headers=ADMIN, json=body)
    assert response.status_code == 502
    assert "HTTP 401" in response.json()["detail"]
    assert "pool-secret" not in response.text
    route.mock(return_value=httpx.Response(200, json={"unexpected": []}))
    assert admin_client.post(ROOT + "/backends/discover-models", headers=ADMIN, json=body).status_code == 502


@respx.mock
@pytest.mark.parametrize("provider_type,url,header,model,payload", [
    ("google", "https://generativelanguage.googleapis.com/v1beta/models", "x-goog-api-key", "imagen-future",
     {"models": [{"name": "models/imagen-future"}]}),
    ("elevenlabs", "https://api.elevenlabs.io/v1/models", "xi-api-key", "speech-future",
     [{"model_id": "speech-future"}]),
])
def test_discovery_provider_auth_and_response_formats(admin_client, provider_type, url, header, model, payload):
    upstream = respx.get(url).mock(return_value=httpx.Response(200, json=payload))
    response = admin_client.post(ROOT + "/backends/discover-models", headers=ADMIN,
                                 json={"name": "draft", "type": provider_type, "api_key": "draft-secret"})
    assert response.status_code == 200, response.text
    assert response.json()["data"][0]["id"] == model
    assert upstream.calls[0].request.headers[header] == "draft-secret"
