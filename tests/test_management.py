"""Exercise administrative auth, live edits, persistence and observable runtime state."""

import dataclasses
import asyncio
from datetime import date
import json
import stat
import sys
import threading
import time

import pytest
import httpx
from fastapi.testclient import TestClient

from mm_gateway.config import BackendConfig, KeyConfig, Settings
from mm_gateway.observability.metrics import _MetricStore
from mm_gateway.observability.selection import _SelectionStore
from mm_gateway.registry import _PROVIDER_CLASSES
from mm_gateway.server.app import create_app
from mm_gateway.schemas.management import ManagementConfig
from tests.conftest import FakeProvider

ADMIN = {"Authorization": "Bearer management-test"}
USER = {"Authorization": "Bearer generation-test"}
ROOT = "/v1/management"


@pytest.fixture
def management_app(monkeypatch, tmp_path):
    monkeypatch.setitem(_PROVIDER_CLASSES, "fake", "FakeProvider")
    monkeypatch.setitem(sys.modules, "mm_gateway.providers.fake", sys.modules["tests.conftest"])
    settings = Settings(management_api_key="management-test", poll_interval=0.01,
                        keys=[KeyConfig(id="user", key="generation-test")],
                        backends=[BackendConfig(name="fake", type="fake", api_key="provider-secret")])
    return create_app(settings)


@pytest.fixture
def admin_client(management_app):
    with TestClient(management_app) as client:
        yield client


def config(client):
    response = client.get(ROOT + "/config", headers=ADMIN)
    assert response.status_code == 200
    return response.json(), {**ADMIN, "If-Match": response.headers["ETag"]}


def test_admin_auth_never_uses_generation_or_open_keys(admin_client):
    for headers in ({}, USER, {"Authorization": "Bearer unknown"}):
        for path in ("config", "status", "metrics", "tasks", "usage"):
            assert admin_client.get(f"{ROOT}/{path}", headers=headers).status_code == 401
    app = create_app(Settings(keys=[KeyConfig(id="open", key="")]))
    with TestClient(app) as client:
        assert client.get(ROOT + "/status", headers=ADMIN).status_code == 401
        assert client.put(ROOT + "/config", json={}, headers=ADMIN).status_code == 401


def test_secrets_are_redacted_and_preserved_during_live_edit(admin_client, management_app):
    body, headers = config(admin_client)
    serialized = json.dumps(body)
    assert "provider-secret" not in serialized
    assert "generation-test" not in serialized
    assert body["config"]["keys"][0]["key"] == "[redacted]"
    body["config"]["keys"][0]["budget"] = {"period": "day", "limit_usd": 20}
    response = admin_client.put(ROOT + "/config", json=body["config"], headers=headers)
    assert response.status_code == 200, response.text
    assert response.headers["Cache-Control"] == "no-store"
    assert management_app.state.settings.keys[0].key == "generation-test"
    assert management_app.state.settings.backends[0].api_key == "provider-secret"
    assert admin_client.get("/v1/usage", headers=USER).json()["key"]["limit_usd"] == 20


def test_revision_prevents_lost_updates_and_new_redacted_secrets(admin_client):
    body, headers = config(admin_client)
    assert admin_client.put(ROOT + "/config", json=body["config"], headers=ADMIN).status_code == 428
    assert admin_client.put(ROOT + "/config", json=body["config"], headers=headers).status_code == 200
    assert admin_client.put(ROOT + "/config", json=body["config"], headers=headers).status_code == 412
    _, current = config(admin_client)
    response = admin_client.put(ROOT + "/keys/new", headers=current, json={"id": "new", "key": "[redacted]"})
    assert response.status_code == 422


def test_key_create_disable_rotate_delete_and_admin_separation(admin_client):
    _, headers = config(admin_client)
    response = admin_client.put(ROOT + "/keys/new", headers=headers, json={"id": "new", "key": "new-key", "allow_backends": ["fake"]})
    assert response.status_code == 200
    assert "new-key" not in response.text
    token = {"Authorization": "Bearer new-key"}
    assert admin_client.get("/v1/models", headers=token).status_code == 200
    assert admin_client.get(ROOT + "/status", headers=token).status_code == 401
    key = response.json()["config"]["keys"][-1]
    key["enabled"] = False
    response = admin_client.put(ROOT + "/keys/new", headers={**ADMIN, "If-Match": response.headers["ETag"]}, json=key)
    assert response.status_code == 200
    assert admin_client.get("/v1/models", headers=token).status_code == 401
    key.update(enabled=True, key="rotated")
    _, headers = config(admin_client)
    assert admin_client.put(ROOT + "/keys/new", headers=headers, json=key).status_code == 200
    assert admin_client.get("/v1/models", headers={"Authorization": "Bearer rotated"}).status_code == 200
    _, headers = config(admin_client)
    assert admin_client.delete(ROOT + "/keys/new", headers=headers).status_code == 200
    assert admin_client.get("/v1/models", headers={"Authorization": "Bearer rotated"}).status_code == 401


def test_backend_create_disable_and_delete_updates_model_routing(admin_client):
    _, headers = config(admin_client)
    response = admin_client.put(ROOT + "/backends/second", headers=headers,
                                json={"name": "second", "type": "fake", "api_key": "another-secret"})
    assert response.status_code == 200, response.text
    status = admin_client.get(ROOT + "/status", headers=ADMIN).json()
    assert all(b["active"] for b in status["backends"])
    backend = response.json()["config"]["backends"][-1]
    backend["enabled"] = False
    _, headers = config(admin_client)
    assert admin_client.put(ROOT + "/backends/second", json=backend, headers=headers).status_code == 200
    status = admin_client.get(ROOT + "/status", headers=ADMIN).json()
    assert status["backends"][-1]["active"] is False
    _, headers = config(admin_client)
    assert admin_client.delete(ROOT + "/backends/second", headers=headers).status_code == 200


def test_invalid_updates_leave_registry_and_revision_unchanged(admin_client, management_app):
    body, headers = config(admin_client)
    original = management_app.state.registry.providers["fake"]
    for invalid in (
        {"name": "bad", "type": "unknown", "api_key": "secret"},
        {"name": "bad", "type": "fake", "credentials": [{"id": "dup"}, {"id": "dup"}]},
    ):
        assert admin_client.put(ROOT + "/backends/bad", headers=headers, json=invalid).status_code == 422
        assert config(admin_client)[0]["revision"] == body["revision"]
        assert management_app.state.registry.providers["fake"] is original
    response = admin_client.put(ROOT + "/keys/user", headers=headers,
                                json={"id": "user", "key": "new-secret", "budget": {"limit_usd": -1}})
    assert response.status_code == 422
    assert "new-secret" not in response.text
    assert '"input"' not in response.text


def test_proxy_accounts_preserve_secrets_when_reordered(admin_client, management_app):
    _, headers = config(admin_client)
    response = admin_client.put(ROOT + "/proxies/example.test", headers=headers,
                                json={"base_url": "https://example.test/v1", "headers": {"X-Custom": "static-secret"},
                                      "accounts": [{"id": "a", "headers": {"Authorization": "Bearer alpha"}},
                                                   {"id": "b", "headers": {"Authorization": "Bearer beta"}}]})
    assert response.status_code == 200
    assert "alpha" not in response.text and "beta" not in response.text and "static-secret" not in response.text
    proxy = response.json()["config"]["proxies"][0]
    proxy["accounts"].reverse()
    _, headers = config(admin_client)
    assert admin_client.put(ROOT + "/proxies/example.test", headers=headers, json=proxy).status_code == 200
    stored = management_app.state.settings.proxies[0]
    assert stored.accounts[0]["headers"]["Authorization"] == "Bearer beta"
    _, headers = config(admin_client)
    assert admin_client.delete(ROOT + "/proxies/example.test", headers=headers).status_code == 200


def test_live_task_snapshots_filters_pagination_usage_and_metrics(admin_client, management_app):
    created = admin_client.post("/v1/images", headers=USER,
                                 json={"input": [{"type": "text", "text": "private prompt"}]}).json()
    for _ in range(50):
        tasks = admin_client.get(ROOT + "/tasks?modality=image&key_id=user&limit=1", headers=ADMIN).json()
        if tasks["data"][0]["status"] == "succeeded":
            break
        time.sleep(0.01)
    assert tasks["data"][0]["id"] == created["id"]
    assert tasks["data"][0]["status"] == "succeeded"
    assert "private prompt" not in json.dumps(tasks)
    assert admin_client.get(ROOT + "/tasks?offset=1", headers=ADMIN).json()["data"] == []
    assert admin_client.get(ROOT + "/tasks?status=failed", headers=ADMIN).json()["total"] == 0
    assert admin_client.get(ROOT + "/tasks?limit=201", headers=ADMIN).status_code == 422
    metrics = admin_client.get(ROOT + "/metrics", headers=ADMIN).json()
    assert any(c["name"] == "gateway_http_requests_total" for c in metrics["counters"])
    assert any(c["name"] == "gateway_request_duration_seconds" and c["count"] > 0 for c in metrics["histograms"])
    assert admin_client.get(ROOT + "/usage", headers=ADMIN).json()["data"][0]["key_id"] == "user"
    status = admin_client.get(ROOT + "/status", headers=ADMIN).json()
    assert status["tasks_by_status"]["succeeded"] == 1


def test_persistent_overlay_restores_on_restart_and_failed_write_is_atomic(management_app, tmp_path, monkeypatch):
    settings = dataclasses.replace(management_app.state.settings, management_config_path=str(tmp_path / "managed.json"))
    app = create_app(settings)
    with TestClient(app) as client:
        body, headers = config(client)
        body["config"]["routing_default_optimize"] = "cost"
        assert client.put(ROOT + "/config", headers=headers, json=body["config"]).status_code == 200
        file = tmp_path / "managed.json"
        assert stat.S_IMODE(file.stat().st_mode) == 0o600
        restarted = create_app(settings)
        assert restarted.state.settings.routing_default_optimize == "cost"
        assert restarted.state.settings.keys[0].key == "generation-test"
        body, headers = config(client)
        def fail_write(*args):
            raise OSError("private path and credential")
        monkeypatch.setattr("mm_gateway.management._persist", fail_write)
        body["config"]["routing_default_optimize"] = "latency"
        response = client.put(ROOT + "/config", headers=headers, json=body["config"])
        assert response.status_code == 503
        assert app.state.settings.routing_default_optimize == "cost"
        assert config(client)[0]["revision"] == body["revision"]


def test_disabled_metrics_and_yaml_configuration():
    settings = Settings._from_yaml("""
management:
  api_key: management-test
backends:
  - name: example
    type: openai
    enabled: false
    api_key: example
keys:
  - id: disabled
    key: user
    enabled: false
defaults:
  enable_metrics: false
""")
    with TestClient(create_app(settings)) as client:
        assert client.get("/metrics").text == ""
        assert client.get(ROOT + "/metrics", headers=ADMIN).json()["counters"] == []
        assert client.get("/v1/models", headers={"Authorization": "Bearer user"}).status_code == 401
        assert client.get(ROOT + "/status", headers=ADMIN).json()["backends"][0]["active"] is False


def test_metrics_aggregate_memory_and_escape_labels():
    store = _MetricStore()
    for value in range(10000):
        store.observe("latency", value, model='quote"\\\n')
    sample = store.snapshot()["histograms"][0]
    assert sample == {"name": "latency", "labels": {"model": 'quote"\\\n'},
                      "count": 10000, "sum": 49995000, "min": 0, "max": 9999, "mean": 4999.5}
    assert 'model="quote\\"\\\\\\n"' in store.render_prometheus()
    selection = _SelectionStore()
    selection.cooldown(backend='quote"', cooldown_s=0)
    assert "NaN" in selection.render_prometheus()


def test_management_openapi_declares_admin_auth_and_revision_errors(admin_client):
    spec = admin_client.get("/openapi.json").json()
    for path, methods in spec["paths"].items():
        if path.startswith(ROOT):
            for operation in methods.values():
                assert operation["security"] == [{"ManagementAuth": []}]
                for code in ("412", "428"):
                    if code in operation["responses"]:
                        assert set(operation["responses"][code]["content"]) == {"application/problem+json"}


@pytest.mark.asyncio
async def test_concurrent_updates_only_publish_one_revision(management_app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=management_app), base_url="http://test") as client:
        old = await client.get(ROOT + "/config", headers=ADMIN)
        headers = {**ADMIN, "If-Match": old.headers["ETag"]}
        results = await asyncio.gather(*[
            client.put(ROOT + "/config", headers=headers, json=old.json()["config"]) for _ in range(2)
        ])
        assert sorted(result.status_code for result in results) == [200, 412]


def test_credentials_in_urls_and_nested_options_survive_redacted_round_trip(admin_client, management_app):
    body, headers = config(admin_client)
    body["config"]["proxies"] = [
        {"base_url": "https://user:private-password@one.example.test/", "accounts": [{"id": "a", "headers": {"X-Key": "private"}}]},
        {"base_url": "https://user:other-password@two.example.test/", "accounts": [{"id": "a", "headers": {"X-Key": "private"}}]},
    ]
    body["config"]["backends"][0]["extra"] = {"credentials_json": {"private_key": "nested-private-key"}}
    response = admin_client.put(ROOT + "/config", headers=headers, json=body["config"])
    assert response.status_code == 200, response.text
    assert "private-password" not in response.text and "nested-private-key" not in response.text
    returned = response.json()["config"]
    assert returned["proxies"][0]["domain"] == "one.example.test"
    returned["proxies"].reverse()
    headers = {**ADMIN, "If-Match": response.headers["ETag"]}
    assert admin_client.put(ROOT + "/config", headers=headers, json=returned).status_code == 200
    assert management_app.state.settings.proxies[0].base_url == "https://user:other-password@two.example.test/"


def test_parent_options_and_headers_remain_inherited_after_round_trip(admin_client, management_app):
    body, headers = config(admin_client)
    body["config"]["backends"][0].update(extra={"image_model": "custom-one"},
                                             credentials=[{"id": "a", "api_key": "account-secret", "extra": {}}])
    body["config"]["proxies"] = [{"base_url": "https://example.test/", "headers": {"X-Version": "one"},
                                 "accounts": [{"id": "a", "headers": {"X-Key": "secret"}}]}]
    response = admin_client.put(ROOT + "/config", headers=headers, json=body["config"])
    assert response.status_code == 200
    returned = response.json()["config"]
    assert returned["backends"][0]["credentials"][0]["extra"] == {}
    assert "X-Version" not in returned["proxies"][0]["accounts"][0]["headers"]
    returned["backends"][0]["extra"]["image_model"] = "custom-two"
    returned["proxies"][0]["headers"]["X-Version"] = "two"
    headers = {**ADMIN, "If-Match": response.headers["ETag"]}
    assert admin_client.put(ROOT + "/config", headers=headers, json=returned).status_code == 200
    assert management_app.state.settings.backends[0].accounts()[0][3]["image_model"] == "custom-two"
    assert management_app.state.settings.proxies[0].enumerate_accounts()[0][1]["X-Version"] == "two"


def test_invalid_catalog_and_proxy_collisions_are_atomic(admin_client):
    body, headers = config(admin_client)
    body["config"]["catalog_models"] = {"fake-image-1": {"unknown": "private-value"}}
    response = admin_client.put(ROOT + "/config", headers=headers, json=body["config"])
    assert response.status_code == 422 and "private-value" not in response.text
    assert config(admin_client)[0]["revision"] == body["revision"]
    response = admin_client.put(ROOT + "/proxies/fake", headers=headers,
                                json={"base_url": "https://fake/", "accounts": [{"id": "a", "headers": {"X-Key": "private"}}]})
    assert response.status_code == 422
    assert config(admin_client)[0]["revision"] == body["revision"]


def test_management_bodies_are_excluded_from_http_logging(admin_client, monkeypatch):
    logged = []
    monkeypatch.setattr("mm_gateway.server.app.frontend_request_log", lambda method, url, headers, body: logged.append(body))
    _, headers = config(admin_client)
    assert admin_client.put(ROOT + "/keys/secret", headers=headers,
                            json={"id": "secret", "key": "must-not-be-logged"}).status_code == 200
    assert logged == [None, None]


@pytest.mark.parametrize("root_path", ["/nested/gateway", "/v1"])
def test_prefixed_docs_and_management_keep_secrets_private(management_app, monkeypatch, root_path):
    settings = dataclasses.replace(management_app.state.settings, root_path=root_path)
    app = create_app(settings)
    with TestClient(app) as client:
        docs = client.get(root_path + "/docs")
        assert docs.status_code == 200
        assert root_path + "/openapi.json" in docs.text
        spec = client.get(root_path + "/openapi.json").json()
        assert {"url": root_path} in spec["servers"]

        logged = []
        monkeypatch.setattr("mm_gateway.server.app.frontend_request_log",
                            lambda method, url, headers, body: logged.append(body))
        response = client.get(root_path + ROOT + "/config", headers=ADMIN)
        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "no-store"
        headers = {**ADMIN, "If-Match": response.headers["ETag"]}
        invalid = client.put(root_path + ROOT + "/keys/prefixed", headers=headers,
                             json={"id": "prefixed", "key": {"secret": "private-validation-input"}})
        assert invalid.status_code == 422
        assert "private-validation-input" not in invalid.text
        assert invalid.headers["Cache-Control"] == "no-store"
        saved = client.put(root_path + ROOT + "/keys/prefixed", headers=headers,
                           json={"id": "prefixed", "key": "private-prefixed-token"})
        assert saved.status_code == 200
        assert "private-prefixed-token" not in saved.text
        assert logged == [None, None, None]


@pytest.mark.asyncio
async def test_cancelled_writer_finishes_atomic_publication(management_app, tmp_path, monkeypatch):
    from mm_gateway.management import _persist

    path = tmp_path / "management.json"
    management_app.state.settings = dataclasses.replace(management_app.state.settings, management_config_path=str(path))
    service = management_app.state.management
    body = service.configuration()
    body["config"]["routing_default_optimize"] = "latency"
    started, release = threading.Event(), threading.Event()

    def slow_persist(*args):
        started.set()
        assert release.wait(timeout=5)
        _persist(*args)

    monkeypatch.setattr("mm_gateway.management._persist", slow_persist)
    task = asyncio.create_task(service.update(ManagementConfig.model_validate(body["config"]), body["revision"]))
    assert await asyncio.to_thread(started.wait, 5)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert service.revision != body["revision"]
    assert management_app.state.settings.routing_default_optimize == "latency"
    assert json.loads(path.read_text())["routing_default_optimize"] == "latency"


def test_yaml_catalog_dates_do_not_block_unrelated_key_edits(management_app):
    app = create_app(dataclasses.replace(management_app.state.settings,
                     catalog_models={"fake-image-1": {"retired_on": date(2027, 1, 1)}}))
    with TestClient(app) as client:
        body, headers = config(client)
        assert body["config"]["catalog_models"]["fake-image-1"]["retired_on"] == "2027-01-01"
        response = client.put(ROOT + "/keys/new", headers=headers, json={"id": "new", "key": "date-test-key"})
        assert response.status_code == 200, response.text
