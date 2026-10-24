"""Plugin lifecycle, HTTP transformations, safe reloads and real Git fetching."""

import asyncio
import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated

import httpx
import pytest
import respx
from fastapi import Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

from mm_gateway.config import BackendConfig, KeyConfig, ProxyConfig, Settings
from mm_gateway.plugins.store import PluginManager
from mm_gateway.schemas.plugins import PluginOptions
from mm_gateway.server.app import create_app
from mm_gateway.server.auth import get_api_key
from tests.conftest import FakeProvider

ROOT = "/v1/management/plugins"
ADMIN = {"Authorization": "Bearer admin-test"}
USER = {"Authorization": "Bearer user-test"}


def package(root, plugin_id="transform", source=None, **manifest):
    directory = root / plugin_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "plugin.json").write_text(
        json.dumps(
            {
                "id": plugin_id,
                "name": plugin_id.title(),
                "version": "1.0",
                "api_version": "1",
                "hooks": ["on_response"],
                **manifest,
            }
        )
    )
    (directory / "plugin.py").write_text(
        source
        or """
class Plugin:
    def __init__(self, config):
        self.config = config
    async def on_response(self, response, context):
        response.headers["x-test-plugin"] = self.config.get("label", "original")
def create_plugin(config):
    return Plugin(config)
"""
    )
    return directory


@pytest.fixture
def settings(tmp_path):
    return Settings(
        management_api_key="admin-test",
        keys=[KeyConfig(id="user", key="user-test")],
        plugins_dir=str(tmp_path / "sources"),
        plugins_state_path=str(tmp_path / "state" / "plugins.json"),
        plugins_poll_interval=0.1,
        log_level="WARNING",
        poll_interval=0.01,
    )


def app_for(settings):
    app = create_app(settings)

    @app.api_route("/v1/echo", methods=["GET", "POST"])
    async def echo(request: Request, key: Annotated[KeyConfig, Depends(get_api_key)]):
        body = await request.body()
        return JSONResponse(
            {
                "body": json.loads(body) if body else None,
                "query": request.url.query,
                "header": request.headers.get("x-in"),
                "key_id": key.id,
            },
            headers={"etag": '"original"'},
        )

    return app


def headers(client):
    result = client.get(ROOT, headers=ADMIN)
    assert result.status_code == 200, result.text
    return {**ADMIN, "If-Match": result.headers["ETag"]}


def install(client, plugin_id="transform", **options):
    result = client.put(ROOT + "/" + plugin_id, headers=headers(client), json=options)
    assert result.status_code == 200, result.text
    return result


def entry(body, plugin_id="transform"):
    return next(row for row in body["data"] if row["id"] == plugin_id)


def test_discovery_does_not_execute_code_and_admin_boundary(settings, tmp_path):
    sentinel = tmp_path / "executed"
    package(
        Path(settings.plugins_dir),
        source=f"from pathlib import Path\nPath({str(sentinel)!r}).touch()\n",
    )
    with TestClient(app_for(settings)) as client:
        for auth in ({}, USER):
            assert client.get(ROOT, headers=auth).status_code == 401
            assert (
                client.post(
                    ROOT + "/install",
                    headers=auth,
                    json={"repository": "https://github.com/test/plugins"},
                ).status_code
                == 401
            )
        body = client.get(ROOT, headers=ADMIN).json()
        assert not entry(body)["installed"]
        assert (
            entry(body, "request-monitor")["manifest"]["author"]
            == "mm-gateway contributors"
        )
        assert not sentinel.exists()
        assert (
            client.put(
                ROOT + "/transform", headers=headers(client), json={}
            ).status_code
            == 422
        )
        assert not entry(client.get(ROOT, headers=ADMIN).json())["active"]


def test_request_response_transforms_are_ordered_and_repair_http_metadata(settings):
    source = """
class Plugin:
    def __init__(self, config):
        self.label = config["label"]
    async def on_request(self, request, context):
        payload = request.json()
        payload["trace"] += self.label
        request.set_json(payload)
        request.headers["x-in"] = "modified"
        request.query_string = b"plugin=yes"
        context.state["seen"] = self.label
    async def on_response(self, response, context):
        payload = response.json()
        payload.setdefault("trace", "")
        payload["trace"] += context.state["seen"]
        response.set_json(payload)
        response.headers["x-out"] = "changed"
def create_plugin(config):
    return Plugin(config)
"""
    for plugin_id in ("first", "second"):
        package(
            Path(settings.plugins_dir),
            plugin_id,
            source,
            hooks=["on_request", "on_response"],
            response_body="json",
        )
    with TestClient(app_for(settings)) as client:
        install(client, "first", config={"label": "A"}, order=20)
        install(client, "second", config={"label": "B"}, order=10)
        response = client.post("/v1/echo?before=1", headers=USER, json={"trace": ""})
        assert response.status_code == 200, response.text
        assert response.json() == {
            "body": {"trace": "BA"},
            "trace": "AB",
            "query": "plugin=yes",
            "header": "modified",
            "key_id": "user",
        }
        assert response.headers["x-out"] == "changed"
        assert int(response.headers["content-length"]) == len(response.content)
        assert "etag" not in response.headers


def test_original_identity_is_used_and_control_plane_is_excluded(settings):
    package(
        Path(settings.plugins_dir),
        source="""
from mm_gateway.plugins import PluginResponse
class Plugin:
    async def on_request(self, request, context):
        request.headers["authorization"] = "Bearer admin-test"
    async def on_response(self, response, context):
        response.headers["x-plugin-key"] = context.key_id
def create_plugin(config):
    return Plugin()
""",
        hooks=["on_request", "on_response"],
    )
    with TestClient(app_for(settings)) as client:
        install(client)
        assert client.get("/v1/echo").status_code == 401
        response = client.get("/v1/echo", headers=USER)
        assert response.json()["key_id"] == "user"
        assert response.headers["x-plugin-key"] == "user"
        for path in (ROOT, "/v1/management/status", "/health", "/metrics"):
            response = client.get(path, headers=ADMIN)
            assert response.status_code == 200
            assert "x-plugin-key" not in response.headers


def test_short_circuit_does_not_contact_provider_and_invalid_json_still_validates(
    settings,
):
    package(
        Path(settings.plugins_dir),
        source="""
from mm_gateway.plugins import PluginResponse
class Plugin:
    async def on_request(self, request, context):
        if request.headers.get("x-block"):
            return PluginResponse.from_json({"blocked": True}, 403)
        request.set_json({"input": []})
def create_plugin(config):
    return Plugin()
""",
        hooks=["on_request"],
        paths=["/v1/images"],
    )
    backend = BackendConfig(name="fake", type="fake", api_key="test")
    app = create_app(dataclasses.replace(settings, backends=[backend]))
    fake = FakeProvider(backend)
    app.state.registry._backends["fake"] = fake
    app.state.registry._configs["fake"] = backend
    with TestClient(app) as client:
        install(client)
        response = client.post("/v1/images", json={}, headers={**USER, "x-block": "1"})
        assert response.status_code == 403 and response.json() == {"blocked": True}
        assert client.post("/v1/images", json={}, headers=USER).status_code == 422
        assert not fake.image_calls


def test_generation_payload_is_modified_before_provider_submission(settings):
    package(
        Path(settings.plugins_dir),
        source="""
class Plugin:
    async def on_request(self, request, context):
        payload = request.json()
        payload["input"][0]["text"] = "rewritten by plugin"
        request.set_json(payload)
def create_plugin(config):
    return Plugin()
""",
        hooks=["on_request"],
        paths=["/v1/images"],
        methods=["POST"],
    )
    backend = BackendConfig(name="fake", type="fake", api_key="test")
    app = create_app(dataclasses.replace(settings, backends=[backend]))
    fake = FakeProvider(backend)
    app.state.registry._backends["fake"] = fake
    app.state.registry._configs["fake"] = backend
    with TestClient(app) as client:
        install(client)
        response = client.post(
            "/v1/images",
            headers=USER,
            json={"input": [{"type": "text", "text": "original"}]},
        )
        assert response.status_code == 202, response.text
        for _ in range(100):
            if fake.image_calls:
                break
            time.sleep(0.01)
        assert fake.image_calls[0].prompt() == "rewritten by plugin"


def test_persistence_redaction_revision_and_disable_uninstall(settings):
    package(Path(settings.plugins_dir))
    with TestClient(app_for(settings)) as client:
        old = headers(client)
        assert (
            client.put(ROOT + "/transform", headers=ADMIN, json={}).status_code == 428
        )
        result = install(
            client, config={"api_key": "private-plugin-secret", "label": "saved"}
        )
        assert "private-plugin-secret" not in result.text
        assert client.put(ROOT + "/transform", headers=old, json={}).status_code == 412
        assert (
            "private-plugin-secret" not in Path(settings.plugins_state_path).read_text()
        )
        assert (Path(settings.plugins_state_path).stat().st_mode & 0o777) == 0o600
        options = entry(result.json())["options"]
        result = install(client, **options)
        assert (
            client.app.state.plugins._records["transform"].options.config["api_key"]
            == "private-plugin-secret"
        )
    with TestClient(app_for(settings)) as client:
        assert client.get("/v1/echo", headers=USER).headers["x-test-plugin"] == "saved"
        install(client, enabled=False)
        assert "x-test-plugin" not in client.get("/v1/echo", headers=USER).headers
        install(client, config={"label": "again"})
        assert (
            client.delete(ROOT + "/transform", headers=headers(client)).status_code
            == 200
        )
        assert "x-test-plugin" not in client.get("/v1/echo", headers=USER).headers
        assert Path(settings.plugins_dir, "transform", "plugin.py").exists()


def test_hot_reload_helpers_failed_reload_and_recovery(settings):
    directory = package(
        Path(settings.plugins_dir),
        source="""
from .helper import LABEL
class Plugin:
    async def on_response(self, response, context):
        response.headers["x-test-plugin"] = LABEL
def create_plugin(config):
    return Plugin()
""",
    )
    helper = directory / "helper.py"
    helper.write_text('LABEL = "one"\n')
    with TestClient(app_for(settings)) as client:
        install(client)
        assert client.get("/v1/echo", headers=USER).headers["x-test-plugin"] == "one"
        # Same length and timestamp: reload must not reuse stale .pyc helpers.
        import os

        original_stat = helper.stat()
        helper.write_text('LABEL = "two"\n')
        os.utime(helper, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        deadline = time.monotonic() + 5
        while client.get("/v1/echo", headers=USER).headers["x-test-plugin"] != "two":
            assert time.monotonic() < deadline
            time.sleep(0.03)
        (directory / "plugin.py").write_text("broken syntax(")
        response = client.post(ROOT + "/transform/reload", headers=headers(client))
        assert response.status_code == 422
        assert client.get("/v1/echo", headers=USER).headers["x-test-plugin"] == "two"
        assert entry(client.get(ROOT, headers=ADMIN).json())["error"]
        # A broken manifest still permits disabling the last good generation.
        (directory / "plugin.json").write_text("{}")
        install(client, enabled=False)
        assert "x-test-plugin" not in client.get("/v1/echo", headers=USER).headers


@pytest.mark.parametrize(
    "failure",
    [
        "raise RuntimeError('private-plugin-secret')",
        "await asyncio.sleep(2)",
        "response.headers['x-bad'] = 'bad\\r\\nheader'",
    ],
)
def test_hook_failures_timeout_and_rollback(settings, failure):
    package(
        Path(settings.plugins_dir),
        source=f"""
import asyncio
class Plugin:
    async def on_response(self, response, context):
        response.headers["x-partial"] = "must rollback"
        {failure}
def create_plugin(config):
    return Plugin()
""",
    )
    with TestClient(app_for(settings)) as client:
        install(client, failure_mode="open", timeout_ms=20)
        response = client.get("/v1/echo", headers=USER)
        assert response.status_code == 200 and "x-partial" not in response.headers
        install(client, failure_mode="closed", timeout_ms=20)
        response = client.get("/v1/echo", headers=USER)
        assert response.status_code == 500 and response.json()["code"] == "plugin_error"
        assert "private-plugin-secret" not in response.text
        assert entry(client.get(ROOT, headers=ADMIN).json())["failures"] == 1


def test_body_limits_and_non_json_responses(settings):
    package(
        Path(settings.plugins_dir),
        source="""
class Plugin:
    async def on_request(self, request, context):
        pass
    async def on_response(self, response, context):
        response.headers["x-buffered"] = str(response.body is not None)
def create_plugin(config):
    return Plugin()
""",
        hooks=["on_request", "on_response"],
        response_body="json",
    )
    app = app_for(dataclasses.replace(settings, plugins_max_body_bytes=256))

    @app.get("/v1/large")
    async def large():
        return {"content": "a" * 1000}

    @app.get("/v1/bytes")
    async def binary():
        from fastapi.responses import Response

        return Response(b"large binary" * 1000, media_type="image/png")

    with TestClient(app) as client:
        install(client)
        assert (
            client.post("/v1/echo", headers=USER, content=b"x" * 257).status_code == 413
        )
        assert client.get("/v1/large", headers=USER).status_code == 502
        response = client.get("/v1/bytes", headers=USER)
        assert response.status_code == 200 and response.headers["x-buffered"] == "False"
        assert len(response.content) == 12000


def test_proxy_transform_and_stream_chunks(settings):
    package(
        Path(settings.plugins_dir),
        source="""
class Plugin:
    async def on_request(self, request, context):
        request.set_json({"upstream": "changed"})
    async def on_response(self, response, context):
        response.headers["x-plugin-head"] = "present"
    async def on_response_chunk(self, chunk, context, more_body):
        return chunk.replace(b"original", b"transformed")
def create_plugin(config):
    return Plugin()
""",
        hooks=["on_request", "on_response", "on_response_chunk"],
        response_body="json",
        paths=["/proxy/"],
    )
    proxy = ProxyConfig(
        base_url="https://plugin-upstream.test",
        accounts=[
            {"id": "one", "headers": {"Authorization": "Bearer provider-secret"}}
        ],
    )
    settings = dataclasses.replace(settings, proxies=[proxy])
    with respx.mock, TestClient(app_for(settings)) as client:
        seen = []

        def upstream(request):
            seen.append(request)
            return httpx.Response(
                200, json={"text": "original"}, headers={"etag": '"old"'}
            )

        respx.post("https://plugin-upstream.test/json").mock(side_effect=upstream)
        respx.post("https://plugin-upstream.test/stream").mock(
            return_value=httpx.Response(
                200,
                content=b"data: original\n\n",
                headers={"content-type": "text/event-stream", "content-length": "16"},
            )
        )
        install(client)
        response = client.post(
            "/proxy/plugin-upstream.test/json", headers=USER, json={}
        )
        assert response.json() == {"text": "transformed"}
        assert json.loads(seen[0].content) == {"upstream": "changed"}
        assert seen[0].headers["authorization"] == "Bearer provider-secret"
        assert "etag" not in response.headers
        response = client.post(
            "/proxy/plugin-upstream.test/stream", headers=USER, json={}
        )
        assert response.content == b"data: transformed\n\n"
        assert response.headers["x-plugin-head"] == "present"
        assert "content-length" not in response.headers


async def test_streaming_is_incremental_and_inflight_generation_survives_reload(
    settings,
):
    directory = package(
        Path(settings.plugins_dir),
        source="""
from pathlib import Path
class Plugin:
    def __init__(self, config):
        self.closed_file = config["closed_file"]
    async def on_response_chunk(self, chunk, context, more_body):
        return chunk.replace(b"original", b"first")
    async def aclose(self):
        Path(self.closed_file).touch()
def create_plugin(config):
    return Plugin(config)
""",
        hooks=["on_response_chunk"],
    )
    manager = PluginManager(dataclasses.replace(settings, plugins_watch=False))
    await manager.start()
    closed_file = directory / ".closed"
    await manager.put(
        "transform",
        PluginOptions(config={"closed_file": str(closed_file)}),
        manager.revision,
    )
    old = manager._active["transform"]
    try:
        async with manager.lease("/v1/stream", "GET") as leased:
            (directory / "plugin.py").write_text(
                (directory / "plugin.py").read_text().replace('b"first"', 'b"second"')
            )
            await manager.reload("transform", manager.revision)
            assert leased == (old,)
            assert not old.closed and not closed_file.exists()
            assert (
                await old.instance.on_response_chunk(b"original", None, True)
                == b"first"
            )
            assert (
                await manager._active["transform"].instance.on_response_chunk(
                    b"original", None, True
                )
                == b"second"
            )
        assert old.closed and closed_file.exists()
        assert old.namespace not in sys.modules
    finally:
        await manager.aclose()


async def test_sse_first_chunk_reaches_client_before_stream_finishes(settings):
    directory = package(
        Path(settings.plugins_dir),
        source="""
class Plugin:
    async def on_response(self, response, context):
        assert response.body is None
    async def on_response_chunk(self, chunk, context, more_body):
        return chunk.upper()
def create_plugin(config):
    return Plugin()
""",
        hooks=["on_response", "on_response_chunk"],
        response_body="json",
    )
    app = app_for(settings)
    first_sent = asyncio.Event()
    finish = asyncio.Event()

    @app.get("/v1/stream")
    async def stream():
        async def chunks():
            yield b"data: first\n\n"
            await finish.wait()
            yield b"data: last\n\n"

        return StreamingResponse(chunks(), media_type="text/event-stream")

    async with app.router.lifespan_context(app):
        await app.state.plugins.put(
            "transform", PluginOptions(), app.state.plugins.revision
        )
        request_sent = False

        async def receive():
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {"type": "http.request", "body": b"", "more_body": False}
            await asyncio.Event().wait()

        messages = []

        async def send(message):
            messages.append(message)
            if (
                message["type"] == "http.response.body"
                and message.get("body") == b"DATA: FIRST\n\n"
            ):
                first_sent.set()

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/v1/stream",
            "raw_path": b"/v1/stream",
            "query_string": b"",
            "headers": [(b"authorization", b"Bearer user-test")],
            "client": ("127.0.0.1", 1),
            "server": ("test", 80),
        }
        task = asyncio.create_task(app(scope, receive, send))
        try:
            await asyncio.wait_for(first_sent.wait(), 2)
            assert not task.done()
            old = app.state.plugins._active["transform"]
            (directory / "plugin.py").write_text(
                (directory / "plugin.py")
                .read_text()
                .replace("chunk.upper()", "chunk.lower()")
            )
            await app.state.plugins.reload("transform", app.state.plugins.revision)
            assert not old.closed
            finish.set()
            await asyncio.wait_for(task, 2)
            assert (
                b"".join(m.get("body", b"") for m in messages)
                == b"DATA: FIRST\n\nDATA: LAST\n\n"
            )
            assert old.closed
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.fixture
def github_repo(tmp_path, monkeypatch):
    """Exercise the real git fetch/checkout path using a deterministic local origin."""
    from mm_gateway.plugins import remote

    repo = tmp_path / "origin"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Plugin test"], check=True
    )
    directory = package(repo / "packages", "remote-plugin")

    def commit():
        subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(repo), "commit", "-qm", "Plugin revision"], check=True
        )
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()

    original_git = remote._git

    async def local_git(directory, *args):
        if args[:3] == ("remote", "add", "origin"):
            args = (*args[:3], str(repo))
        return await original_git(directory, "-c", "protocol.file.allow=always", *args)

    monkeypatch.setattr(remote, "_git", local_git)
    return directory, commit


def test_github_install_update_failure_rollback_restart_and_uninstall(
    settings, github_repo, monkeypatch
):
    directory, commit = github_repo
    first_commit = commit()
    spec = {
        "repository": "https://github.com/contributor/gateway-plugins",
        "subdirectory": "packages/remote-plugin",
    }
    with TestClient(app_for(settings)) as client:
        result = client.post(ROOT + "/install", headers=headers(client), json=spec)
        assert result.status_code == 200, result.text
        row = entry(result.json(), "remote-plugin")
        assert row["source"]["commit"] == first_commit
        assert (
            client.get("/v1/echo", headers=USER).headers["x-test-plugin"] == "original"
        )
        (directory / "plugin.py").write_text(
            (directory / "plugin.py").read_text().replace('"original"', '"updated"')
        )
        second_commit = commit()
        result = client.post(ROOT + "/remote-plugin/update", headers=headers(client))
        assert result.status_code == 200, result.text
        assert (
            entry(result.json(), "remote-plugin")["source"]["commit"] == second_commit
        )
        assert (
            client.get("/v1/echo", headers=USER).headers["x-test-plugin"] == "updated"
        )
        (directory / "plugin.py").write_text("not valid Python(")
        commit()
        assert (
            client.post(
                ROOT + "/remote-plugin/update", headers=headers(client)
            ).status_code
            == 422
        )
        assert (
            client.get("/v1/echo", headers=USER).headers["x-test-plugin"] == "updated"
        )
        # Persistence failure also restores both old package and active generation.
        installed_source = (
            client.app.state.plugins.install_directory / "remote-plugin" / "plugin.py"
        ).read_text()
        (directory / "plugin.py").write_text(
            installed_source.replace('"updated"', '"third"')
        )
        commit()

        def fail_persist(records):
            raise OSError("simulated full disk")

        with monkeypatch.context() as patch:
            patch.setattr(client.app.state.plugins, "_persist", fail_persist)
            assert (
                client.post(
                    ROOT + "/remote-plugin/update", headers=headers(client)
                ).status_code
                == 422
            )
        assert (
            client.app.state.plugins.install_directory / "remote-plugin" / "plugin.py"
        ).read_text() == installed_source
        assert (
            client.get("/v1/echo", headers=USER).headers["x-test-plugin"] == "updated"
        )
    with TestClient(app_for(settings)) as client:
        assert (
            client.get("/v1/echo", headers=USER).headers["x-test-plugin"] == "updated"
        )
        assert (
            entry(client.get(ROOT, headers=ADMIN).json(), "remote-plugin")["source"][
                "commit"
            ]
            == second_commit
        )
        target = client.app.state.plugins.install_directory / "remote-plugin"
        assert (
            client.delete(ROOT + "/remote-plugin", headers=headers(client)).status_code
            == 200
        )
        assert not target.exists()
        assert "x-test-plugin" not in client.get("/v1/echo", headers=USER).headers


@pytest.mark.parametrize(
    "payload",
    [
        {"repository": "file:///etc"},
        {"repository": "https://user:pass@github.com/owner/repo"},
        {"repository": "https://github.com/owner/repo", "ref": "--upload-pack=evil"},
        {"repository": "https://github.com/owner/repo", "subdirectory": "../escape"},
    ],
)
def test_remote_install_rejects_non_repository_inputs(settings, payload):
    with TestClient(app_for(settings)) as client:
        assert (
            client.post(
                ROOT + "/install", headers=headers(client), json=payload
            ).status_code
            == 422
        )


def test_plugin_package_symlinks_and_api_version_are_rejected(settings, tmp_path):
    directory = package(Path(settings.plugins_dir))
    (directory / "outside.py").symlink_to(tmp_path / "private.py")
    with TestClient(app_for(settings)) as client:
        body = client.get(ROOT, headers=ADMIN).json()
        assert body["errors"] and not any(
            row["id"] == "transform" for row in body["data"]
        )
        (directory / "outside.py").unlink()
        manifest = json.loads((directory / "plugin.json").read_text())
        manifest["api_version"] = "99"
        (directory / "plugin.json").write_text(json.dumps(manifest))
        assert client.get(ROOT, headers=ADMIN).json()["errors"]


def poll_task(client, created):
    assert created.status_code == 202, created.text
    deadline = time.monotonic() + 5
    while True:
        result = client.get(created.headers["Location"], headers=USER)
        assert result.status_code == 200, result.text
        if result.json()["status"] in {"succeeded", "failed"}:
            return result.json()
        assert time.monotonic() < deadline
        time.sleep(0.01)


def put_backend(client, name, provider_type, **values):
    config = client.get("/v1/management/config", headers=ADMIN)
    result = client.put(
        f"/v1/management/backends/{name}",
        headers={**ADMIN, "If-Match": config.headers["ETag"]},
        json={"name": name, "type": provider_type, **values},
    )
    assert result.status_code == 200, result.text
    return result


def test_contributor_provider_registers_models_and_generates_through_public_api(
    settings,
):
    import shutil

    settings = dataclasses.replace(
        settings,
        management_config_path=str(
            Path(settings.plugins_state_path).with_name("management.json")
        ),
    )
    source = Path(__file__).parents[1] / "examples" / "plugins" / "example_provider"
    target = Path(settings.plugins_dir) / "example"
    target.parent.mkdir(parents=True)
    shutil.copytree(source, target)
    with respx.mock, TestClient(app_for(settings)) as client:
        install(client, "example-provider")
        assert (
            "example-image"
            in client.get("/v1/management/status", headers=ADMIN).json()[
                "backend_types"
            ]
        )
        catalog = client.get(
            "/v1/management/backend-types/example-image/models", headers=ADMIN
        )
        assert catalog.json() == {"models": {"image": ["example-image-v1"]}}
        put_backend(
            client,
            "contributed",
            "example-image",
            base_url="https://example-image.test/",
        )
        assert {
            "id": "example-image-v1",
            "object": "model",
            "modality": "image",
        } in client.get("/v1/models", headers=USER).json()["data"]
        upstream = respx.post("https://example-image.test/generate").mock(
            return_value=httpx.Response(
                200, json={"images": ["https://example.test/plugin.png"]}
            )
        )
        result = poll_task(
            client,
            client.post(
                "/v1/images",
                headers=USER,
                json={"input": [{"type": "text", "text": "plugin provider"}]},
            ),
        )
        assert result["status"] == "succeeded", result
        assert result["outputs"][0]["uri"] == "https://example.test/plugin.png"
        assert (
            json.loads(upstream.calls[0].request.content)["prompt"] == "plugin provider"
        )
        original_provider = client.app.state.registry.backends["contributed"]
        install(client, "response-headers", config={"headers": {}})
        assert client.app.state.registry.backends["contributed"] is original_provider
        install(client, "example-provider", enabled=False)
        assert (
            "example-image"
            not in client.get("/v1/management/status", headers=ADMIN).json()[
                "backend_types"
            ]
        )
        assert not client.get("/v1/models", headers=USER).json()["data"]
        install(client, "example-provider")
        assert client.get("/v1/models", headers=USER).json()["data"]
    with TestClient(app_for(settings)) as client:
        assert "contributed" in client.app.state.registry.backends


def test_backend_auth_prepares_sdk_and_signs_wire_requests_without_persisting_derived_credentials(
    settings,
):
    package(
        Path(settings.plugins_dir),
        "sdk-auth",
        """
from dataclasses import replace
class Plugin:
    def __init__(self, config):
        self.token = config["token"]
    async def prepare_backend(self, backend, context):
        return replace(backend, api_key="constructor-" + self.token)
    async def on_backend_request(self, request, context):
        request.headers["authorization"] = "Bearer wire-" + self.token
        request.headers["x-signed-account"] = context.account_id
def create_plugin(config):
    return Plugin(config)
""",
        hooks=["prepare_backend", "on_backend_request"],
    )
    with respx.mock, TestClient(app_for(settings)) as client:
        installed = install(client, "sdk-auth", config={"token": "private-auth-token"})
        assert "private-auth-token" not in installed.text
        config = put_backend(
            client,
            "authenticated",
            "openai",
            base_url="https://openai-plugin.test/v1",
            extra={"auth_plugin": "sdk-auth"},
        )
        assert config.json()["config"]["backends"][0]["api_key"] is None
        assert (
            config.json()["config"]["backends"][0]["extra"]["auth_plugin"] == "sdk-auth"
        )
        assert (
            client.app.state.registry.backends["authenticated"].backend.api_key
            == "constructor-private-auth-token"
        )
        upstream = respx.post("https://openai-plugin.test/v1/images/generations").mock(
            return_value=httpx.Response(
                200,
                json={
                    "created": 1,
                    "data": [{"url": "https://example.test/authenticated.png"}],
                },
            )
        )
        result = poll_task(
            client,
            client.post(
                "/v1/images",
                headers=USER,
                json={
                    "model": "gpt-image-1",
                    "input": [{"type": "text", "text": "signed"}],
                },
            ),
        )
        assert result["status"] == "succeeded", result
        assert (
            upstream.calls[0].request.headers["authorization"]
            == "Bearer wire-private-auth-token"
        )
        assert upstream.calls[0].request.headers["x-signed-account"] == "default"
        listing = respx.get("https://openai-plugin.test/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": "gpt-image-1"}]})
        )
        discovered = client.post(
            "/v1/management/backends/discover-models",
            headers=ADMIN,
            json={
                "name": "authenticated",
                "type": "openai",
                "base_url": "https://openai-plugin.test/v1",
                "extra": {"auth_plugin": "sdk-auth"},
            },
        )
        assert discovered.status_code == 200, discovered.text
        assert (
            listing.calls[0].request.headers["authorization"]
            == "Bearer wire-private-auth-token"
        )
        assert (
            "private-auth-token"
            not in client.get("/v1/management/config", headers=ADMIN).text
        )
        install(client, "sdk-auth", enabled=False)
        assert "authenticated" not in client.app.state.registry.backends
        install(client, "sdk-auth", config={"token": "rotated-token"})
        assert (
            client.app.state.registry.backends["authenticated"].backend.api_key
            == "constructor-rotated-token"
        )


def test_route_plugin_agrees_with_estimates_and_cannot_select_forbidden_or_over_budget_candidates(
    settings, monkeypatch
):
    from mm_gateway.registry import _PROVIDER_CLASSES

    monkeypatch.setitem(_PROVIDER_CLASSES, "fake", "FakeProvider")
    monkeypatch.setitem(
        sys.modules, "mm_gateway.providers.fake", sys.modules["tests.conftest"]
    )
    with TestClient(app_for(settings)) as client:
        put_backend(client, "first", "fake", api_key="test")
        put_backend(client, "preferred", "fake", api_key="test")
        install(
            client, "route-preference", config={"preferred_backends": ["preferred"]}
        )
        request = {"input": [{"type": "text", "text": "custom route"}]}
        estimate = client.post("/v1/images/estimate", headers=USER, json=request)
        assert estimate.status_code == 200
        created = client.post("/v1/images", headers=USER, json=request)
        result = poll_task(client, created)
        assert result["model"] == estimate.json()["model"]
        assert client.app.state.registry.backends["preferred"].image_calls
        assert not client.app.state.registry.backends["first"].image_calls
        # A restricted key only exposes permitted candidates to the plugin.
        config = client.get("/v1/management/config", headers=ADMIN)
        restricted = client.put(
            "/v1/management/keys/user",
            headers={**ADMIN, "If-Match": config.headers["ETag"]},
            json={"id": "user", "key": "user-test", "allow_backends": ["first"]},
        )
        assert restricted.status_code == 200, restricted.text
        poll_task(client, client.post("/v1/images", headers=USER, json=request))
        assert client.app.state.registry.backends["first"].image_calls
        # Candidates excluded by a cost cap never reach a route hook.
        limited = {**request, "routing": {"max_cost_usd": 0.00000001}}
        assert client.post("/v1/images", headers=USER, json=limited).status_code == 422
        assert (
            not client.post("/v1/images/estimate", headers=USER, json=limited)
            .json()
            .get("model")
        )


def test_invalid_route_choice_fails_closed_and_provider_type_collisions_rollback(
    settings, monkeypatch
):
    package(
        Path(settings.plugins_dir),
        "route-invalid",
        """
class Plugin:
    async def route(self, routing, context):
        return ["not-an-admissible-id"]
def create_plugin(config):
    return Plugin()
""",
        hooks=["route"],
    )
    from mm_gateway.registry import _PROVIDER_CLASSES

    monkeypatch.setitem(_PROVIDER_CLASSES, "fake", "FakeProvider")
    monkeypatch.setitem(
        sys.modules, "mm_gateway.providers.fake", sys.modules["tests.conftest"]
    )
    with TestClient(app_for(settings)) as client:
        put_backend(client, "backend", "fake", api_key="test")
        install(client, "route-invalid")
        result = client.post(
            "/v1/images",
            headers=USER,
            json={"input": [{"type": "text", "text": "route"}]},
        )
        assert (
            result.status_code == 503 and result.json()["code"] == "plugin_route_error"
        )
        package(
            Path(settings.plugins_dir),
            "colliding",
            """
from tests.conftest import FakeProvider
class Plugin:
    providers = {"openai": FakeProvider}
def create_plugin(config):
    return Plugin()
""",
            hooks=[],
            providers=["openai"],
        )
        result = client.put(ROOT + "/colliding", headers=headers(client), json={})
        assert result.status_code == 422
        assert not entry(client.get(ROOT, headers=ADMIN).json(), "colliding")[
            "installed"
        ]


def test_provider_reload_keeps_accepted_tasks_on_their_owning_generation(
    settings, tmp_path
):
    source = """
from pathlib import Path
from uuid import uuid4
from mm_gateway.core.base import ImageProvider
from mm_gateway.schemas.image import UnifiedImageTask, ImageData
class StickyProvider(ImageProvider):
    image_models = ["sticky-image"]
    label = "version-one"
    def __init__(self, backend):
        super().__init__(backend)
        self.name = backend.name
    async def create_image_task(self, request, *, sync=None):
        return UnifiedImageTask(task_id=uuid4().hex, provider=self.name, model=request.model, status="pending")
    async def get_image_task(self, task_id):
        complete = Path(self.backend.extra["release_file"]).exists()
        return UnifiedImageTask(task_id=task_id, provider=self.name, model="sticky-image",
            status="succeeded" if complete else "pending",
            images=[ImageData(url="https://example.test/" + self.label + ".png")] if complete else [])
    async def aclose(self):
        Path(self.backend.extra["closed_directory"], self.label).touch()
class Plugin:
    providers = {"sticky-image": StickyProvider}
def create_plugin(config):
    return Plugin()
"""
    directory = package(
        Path(settings.plugins_dir),
        "sticky-plugin",
        source,
        hooks=[],
        providers=["sticky-image"],
    )
    release = tmp_path / "release"
    with TestClient(app_for(settings)) as client:
        install(client, "sticky-plugin")
        put_backend(
            client,
            "sticky",
            "sticky-image",
            extra={"release_file": str(release), "closed_directory": str(tmp_path)},
        )
        old = client.app.state.registry.backends["sticky"]._gateway_plugin_owner
        created = client.post(
            "/v1/images",
            headers=USER,
            json={"input": [{"type": "text", "text": "held task"}]},
        )
        assert created.status_code == 202
        deadline = time.monotonic() + 2
        while not client.app.state.image_service._plugin_pins:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        (directory / "plugin.py").write_text(
            source.replace('"version-one"', '"version-two"')
        )
        result = client.post(ROOT + "/sticky-plugin/reload", headers=headers(client))
        assert result.status_code == 200, result.text
        assert not old.closed
        release.touch()
        assert (
            poll_task(client, created)["outputs"][0]["uri"]
            == "https://example.test/version-one.png"
        )
        newer = poll_task(
            client,
            client.post(
                "/v1/images",
                headers=USER,
                json={"input": [{"type": "text", "text": "new task"}]},
            ),
        )
        assert newer["outputs"][0]["uri"] == "https://example.test/version-two.png"
        deadline = time.monotonic() + 2
        while not old.closed:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert (tmp_path / "version-one").exists()


def test_backend_auth_applies_to_proxy_accounts(settings):
    proxy = ProxyConfig(
        base_url="https://dynamic-proxy.test", auth_plugin="backend-bearer"
    )
    with (
        respx.mock,
        TestClient(app_for(dataclasses.replace(settings, proxies=[proxy]))) as client,
    ):
        install(client, "backend-bearer", config={"token": "dynamic-proxy-token"})
        upstream = respx.get("https://dynamic-proxy.test/ok").mock(
            return_value=httpx.Response(200, text="signed")
        )
        result = client.get("/proxy/dynamic-proxy.test/ok", headers=USER)
        assert result.status_code == 200, result.text
        assert (
            upstream.calls[0].request.headers["authorization"]
            == "Bearer dynamic-proxy-token"
        )
        install(client, "backend-bearer", enabled=False)
        assert (
            client.get("/proxy/dynamic-proxy.test/ok", headers=USER).status_code == 503
        )


def test_github_source_provenance_survives_a_stateless_restart(
    settings, github_repo, monkeypatch, tmp_path
):
    _, commit = github_repo
    commit_id = commit()
    monkeypatch.chdir(tmp_path)
    settings = dataclasses.replace(
        settings, plugins_state_path=None, management_config_path=None
    )
    with TestClient(app_for(settings)) as client:
        result = client.post(
            ROOT + "/install",
            headers=headers(client),
            json={
                "repository": "https://github.com/contributor/gateway-plugins",
                "ref": commit_id,
                "subdirectory": "packages/remote-plugin",
            },
        )
        assert result.status_code == 200, result.text
    with TestClient(app_for(settings)) as client:
        row = entry(client.get(ROOT, headers=ADMIN).json(), "remote-plugin")
        assert not row["installed"]
        assert (
            row["source"]["repository"]
            == "https://github.com/contributor/gateway-plugins"
        )
        assert row["source"]["commit"] == commit_id
        install(client, "remote-plugin")
        assert (
            client.post(
                ROOT + "/remote-plugin/update", headers=headers(client)
            ).status_code
            == 200
        )


def test_route_plugin_cannot_override_a_live_spend_budget(settings, monkeypatch):
    from mm_gateway.registry import _PROVIDER_CLASSES

    monkeypatch.setitem(_PROVIDER_CLASSES, "fake", "FakeProvider")
    monkeypatch.setitem(
        sys.modules, "mm_gateway.providers.fake", sys.modules["tests.conftest"]
    )
    with TestClient(app_for(settings)) as client:
        put_backend(client, "preferred", "fake", api_key="test")
        install(
            client, "route-preference", config={"preferred_backends": ["preferred"]}
        )
        current = client.get("/v1/management/config", headers=ADMIN)
        config = current.json()["config"]
        config["catalog_models"] = {
            "fake-image-1": {"modality": "image", "price": {"per_image": 1}}
        }
        config["keys"][0]["budget"] = {"period": "day", "limit_usd": 0.5}
        saved = client.put(
            "/v1/management/config",
            headers={**ADMIN, "If-Match": current.headers["ETag"]},
            json=config,
        )
        assert saved.status_code == 200, saved.text
        result = client.post(
            "/v1/images",
            headers=USER,
            json={"input": [{"type": "text", "text": "budget checked"}]},
        )
        assert result.status_code == 402 and result.json()["code"] == "budget_exceeded"
        assert not client.app.state.registry.backends["preferred"].image_calls
