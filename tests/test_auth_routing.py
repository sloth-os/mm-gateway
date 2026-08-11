"""Tests for API-key authentication and backend routing through the HTTP API.

These cover the bearer-token resolution (``server.auth.resolve_key`` /
``get_api_key``) and the hybrid backend routing (``registry.resolve`` plus the
typed, provider-neutral ``routing.profile`` request directive) over a
multi-backend fake fixture — i.e. the same routing matrix the
MCP tools exercise, but driven through the FastAPI routes so Problem Details
and HTTP statuses are asserted directly.

A shared multi-backend app is built once per test via the ``multi`` fixture so
each test starts from a clean provider call log.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mm_gateway.config import BackendConfig, KeyConfig, Settings
from mm_gateway.server.app import create_app
from tests.conftest import FakeProvider

# --------------------------------------------------------------------------- #
# Multi-backend fixture
# --------------------------------------------------------------------------- #


def _backends() -> list[BackendConfig]:
    return [
        BackendConfig(name="img-a", type="fake", api_key="k", tags=["image-primary", "shared"]),
        BackendConfig(name="img-b", type="fake", api_key="k", tags=["image-secondary"]),
        BackendConfig(name="vid-a", type="fake", api_key="k", tags=["video-primary", "shared"]),
        BackendConfig(name="vid-b", type="fake", api_key="k", tags=["video-secondary"]),
    ]


def _keys() -> list[KeyConfig]:
    return [
        # May use every backend (allow_tags intersect each).
        KeyConfig(id="alice", key="alice-token",
                  allow_tags=["image-primary", "image-secondary",
                              "video-primary", "video-secondary", "shared"]),
        # Pinned to a single image backend by name.
        KeyConfig(id="bob", key="bob-token", allow_backends=["img-a"]),
        # May use both image backends, but denies img-a by *name*. (deny_tags
        # matches backend NAME or TYPE, not a backend's tag labels.) With img-a
        # removed, an image call must route to the remaining image backend img-b
        # — proving deny_tags is selective (removes one backend, not all).
        KeyConfig(id="carol", key="carol-token",
                  allow_tags=["image-primary", "image-secondary"],
                  deny_tags=["img-a"]),
        # Allow_tags match no configured backend -> every call forbidden.
        KeyConfig(id="dave", key="dave-token", allow_tags=["restricted"]),
        # May use the shared tag, but denies every backend of type "fake" ->
        # also forbidden (deny_tags by type).
        KeyConfig(id="erin", key="erin-token", allow_tags=["shared"],
                  deny_tags=["fake"]),
        # Open key (empty token) -> admits any caller, no header required.
        KeyConfig(id="open", key=""),
    ]


def _no_open_keys() -> list[KeyConfig]:
    """The key set above minus the open key — for tests that need auth enforced."""
    return [k for k in _keys() if k.id != "open"]


@pytest.fixture
def multi():
    """A multi-backend app with a clean fake-provider call log per test."""
    settings = Settings(
        backends=_backends(), keys=_keys(),
        video_sync_default=True, max_sync_wait=5.0, poll_interval=0.01,
    )
    app = create_app(settings)
    for cfg in settings.backends:
        app.state.registry._backends[cfg.name] = FakeProvider(cfg)
        app.state.registry._configs[cfg.name] = cfg
    return app


@pytest.fixture
def client(multi) -> TestClient:
    return TestClient(multi)


def _providers(multi) -> dict[str, FakeProvider]:
    return {n: p for n, p in multi.state.registry._backends.items()}  # type: ignore[attr-defined]


def _image_landed(provs: dict[str, FakeProvider], name: str) -> bool:
    """True iff backend ``name`` recorded an image call."""
    p = provs.get(name)
    return bool(p and p.image_calls)


# --------------------------------------------------------------------------- #
# Auth: 401 / open-key / unknown
# --------------------------------------------------------------------------- #


def _build(backends, keys):
    """Construct a multi-backend app with the fake provider injected into every slot."""
    s = Settings(backends=backends, keys=keys,
                 video_sync_default=True, max_sync_wait=5.0, poll_interval=0.01)
    app = create_app(s)
    for cfg in s.backends:
        app.state.registry._backends[cfg.name] = FakeProvider(cfg)
        app.state.registry._configs[cfg.name] = cfg
    return app


def test_missing_token_401_when_no_open_key():
    # No open key configured -> a missing token is a real 401.
    app = _build(_backends(), _no_open_keys())
    r = TestClient(app).get("/v1/models")
    assert r.status_code == 401
    assert r.json()["code"] == "unauthorized"
    assert "Missing API key" in r.json()["detail"]


def test_unknown_token_401():
    # An open key admits any caller (incl. an unknown token), so "unknown token
    # rejected" only holds when no open key is configured.
    app = _build(_backends(), _no_open_keys())
    r = TestClient(app).get("/v1/models", headers={"authorization": "Bearer not-real"})
    assert r.status_code == 401
    assert r.json()["code"] == "unauthorized"
    assert "Unknown" in r.json()["detail"]


def test_open_key_admits_any_caller_no_header(client):
    # No Authorization header at all, yet the open key matches.
    r = client.get("/v1/models")
    assert r.status_code == 200
    assert r.json()["object"] == "list"


def test_valid_token_lists_usable_models_without_backend_details(client):
    # The catalogue is scoped by the key but never exposes deployment details.
    r = client.get("/v1/models", headers={"authorization": "Bearer bob-token"})
    assert r.status_code == 200
    models = r.json()["data"]
    assert models
    assert all(set(model) == {"id", "object", "modality"} for model in models)


# --------------------------------------------------------------------------- #
# Forbidden: 403 when the key has no usable backend
# --------------------------------------------------------------------------- #


def test_key_with_no_usable_backend_403(client):
    r = client.post("/v1/images",
                    headers={"authorization": "Bearer dave-token"},
                    json={"model": "fake-image-1", "input": "x"})
    assert r.status_code == 403
    body = r.json()
    assert body["code"] == "forbidden"
    assert "not allowed" in body["detail"]


def test_carol_denied_backend_routes_away(client, multi):
    # Carol may use both image backends but denies img-a by name. The image call
    # must therefore land on the remaining usable image backend, img-b — proving
    # deny_tags removes the named backend from routing rather than rejecting the
    # whole key (which would be a 403).
    r = client.post("/v1/images",
                    headers={"authorization": "Bearer carol-token"},
                    json={"model": "fake-image-1", "input": "x"})
    assert r.status_code == 202, r.text
    provs = _providers(multi)
    assert _image_landed(provs, "img-b")
    assert not _image_landed(provs, "img-a")


def test_erin_denied_by_type_403(client):
    # Erin denies the 'fake' type entirely -> no backend usable -> forbidden.
    r = client.post("/v1/images",
                    headers={"authorization": "Bearer erin-token"},
                    json={"model": "fake-image-1", "input": "x"})
    assert r.status_code == 403
    assert r.json()["code"] == "forbidden"


# --------------------------------------------------------------------------- #
# Tag routing
# --------------------------------------------------------------------------- #


def test_profile_routing_via_request_body(client, multi):
    r = client.post("/v1/images",
                    headers={"authorization": "Bearer alice-token"},
                    json={"model": "fake-image-1", "input": "x",
                          "routing": {"profile": "image-secondary"}})
    assert r.status_code == 202, r.text
    provs = _providers(multi)
    assert _image_landed(provs, "img-b")


# --------------------------------------------------------------------------- #
# Routing profiles never expose backend instance names
# --------------------------------------------------------------------------- #


def test_profile_can_select_a_uniquely_tagged_backend(client, multi):
    r = client.post("/v1/images",
                    headers={"authorization": "Bearer alice-token"},
                    json={"model": "fake-image-1", "input": "x",
                          "routing": {"profile": "image-secondary"}})
    assert r.status_code == 202, r.text
    provs = _providers(multi)
    assert _image_landed(provs, "img-b")


def test_unknown_routing_profile_is_rejected(client):
    response = client.post(
        "/v1/images",
        headers={"authorization": "Bearer alice-token"},
        json={
            "model": "fake-image-1",
            "input": "x",
            "routing": {"profile": "does-not-exist"},
        },
    )
    assert response.status_code == 400
    assert response.json()["code"] == "invalid_request_error"


def test_bob_pinned_to_img_a_short_circuits(client, multi):
    # Bob's only allowed backend is img-a; the call lands there even though the
    # model is also served by img-b. (Registry.usable_backends scopes to img-a,
    # so the pin need not even be named on the request.)
    r = client.post("/v1/images",
                    headers={"authorization": "Bearer bob-token"},
                    json={"model": "fake-image-1", "input": "x"})
    assert r.status_code == 202, r.text
    provs = _providers(multi)
    assert _image_landed(provs, "img-a")
    assert not _image_landed(provs, "img-b")


# --------------------------------------------------------------------------- #
# deny_tags exclusion
# --------------------------------------------------------------------------- #
# ``test_carol_denied_backend_routes_away`` (above) covers this: carol denies
# img-a by name yet her image call still succeeds by routing to img-b — i.e.
# deny_tags removes a backend from the usable set rather than rejecting the
# whole key. No separate test needed here.


# --------------------------------------------------------------------------- #
# default_image_backend / default_video_backend key selection
# --------------------------------------------------------------------------- #


def test_default_image_backend_is_picked_without_override(client, multi):
    # A key whose default_image_backend names a backend it is also authorised to
    # use must route there absent any explicit override. zoe may use both image
    # backends (img-a, img-b) and defaults to img-b — so img-b, not the
    # first-listed img-a, must handle the call.
    s = Settings(
        backends=_backends(),
        keys=[KeyConfig(id="zoe", key="zoe-token",
                        allow_tags=["image-primary", "image-secondary"],
                        default_image_backend="img-b")],
        video_sync_default=True, max_sync_wait=5.0, poll_interval=0.01,
    )
    app = create_app(s)
    for cfg in s.backends:
        app.state.registry._backends[cfg.name] = FakeProvider(cfg)
        app.state.registry._configs[cfg.name] = cfg
    zoe_client = TestClient(app)
    r = zoe_client.post("/v1/images",
                        headers={"authorization": "Bearer zoe-token"},
                        json={"model": "fake-image-1", "input": "x"})
    assert r.status_code == 202, r.text
    provs = {n: p for n, p in app.state.registry._backends.items()}
    assert _image_landed(provs, "img-b")
    assert not _image_landed(provs, "img-a")


def test_default_video_tag_is_picked(client, multi):
    # A key whose default_video_tag points at the secondary video backend should
    # route a video there absent any explicit override.
    s = Settings(
        backends=_backends(),
        keys=[KeyConfig(id="yara", key="yara-token",
                        allow_tags=["video-primary", "video-secondary"],
                        default_video_tag="video-secondary")],
        video_sync_default=True, max_sync_wait=5.0, poll_interval=0.01,
    )
    app = create_app(s)
    for cfg in s.backends:
        app.state.registry._backends[cfg.name] = FakeProvider(cfg)
        app.state.registry._configs[cfg.name] = cfg
    yara_client = TestClient(app)
    r = yara_client.post("/v1/videos",
                         headers={"authorization": "Bearer yara-token"},
                         json={"model": "fake-video-1", "input": "x"})
    assert r.status_code == 202, r.text
    provs = {n: p for n, p in app.state.registry._backends.items()}
    assert provs["vid-b"].video_calls
    assert not provs["vid-a"].video_calls


# --------------------------------------------------------------------------- #
# Cross-tenant poll authorisation
# --------------------------------------------------------------------------- #


def test_poll_denied_when_key_not_authorised_for_tasks_backend(client, multi):
    # Alice creates a video through the primary profile. Bob is authorised
    # for img-a only, so he is NOT authorised for vid-a — polling alice's task
    # must be a 403, not a cross-tenant leak. (The pin is what makes the task
    # land on a backend bob can't use; without it the first usable backend
    # would handle it, which the FakeProvider serves on every slot.)
    created = client.post("/v1/videos",
                          headers={"authorization": "Bearer alice-token"},
                          json={"model": "fake-video-1",
                                "input": "x",
                                "routing": {"profile": "video-primary"}})
    assert created.status_code == 202, created.text
    task_id = created.json()["id"]

    poll = client.get(f"/v1/videos/{task_id}",
                      headers={"authorization": "Bearer bob-token"})
    assert poll.status_code == 403
    assert poll.json()["code"] == "forbidden"


def test_poll_allowed_when_key_authorised_for_tasks_backend(client, multi):
    # Same-tenant round-trip: alice creates through a routing profile and polls it
    # herself, so the poll succeeds (the authz guard admits the owner's own key).
    created = client.post("/v1/videos",
                          headers={"authorization": "Bearer alice-token"},
                          json={"model": "fake-video-1",
                                "input": "x",
                                "routing": {"profile": "video-primary"}})
    task_id = created.json()["id"]
    poll = client.get(f"/v1/videos/{task_id}",
                      headers={"authorization": "Bearer alice-token"})
    assert poll.status_code == 200, poll.text
    assert poll.json()["id"] == task_id


def test_poll_denied_to_another_key_even_on_the_same_backend(client):
    created = client.post(
        "/v1/images",
        headers={"authorization": "Bearer alice-token"},
        json={
            "model": "fake-image-1",
            "input": "x",
            "routing": {"profile": "image-primary"},
        },
    )
    assert created.status_code == 202

    poll = client.get(
        created.headers["location"],
        headers={"authorization": "Bearer bob-token"},
    )
    assert poll.status_code == 403
    assert poll.json()["code"] == "forbidden"
    assert poll.json()["detail"] == (
        "The API key is not allowed to perform this request."
    )
