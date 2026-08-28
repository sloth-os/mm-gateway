"""Tests for the general pass-through proxy surface.

The proxy forwards raw HTTP to an upstream root URL through its own
``httpx.AsyncClient`` (owned by ``ProxyRunner`` and created lazily), so we use
``respx`` to intercept those clients by URL — no real network. The selection
store is reset per test so account ranking starts deterministic.

A proxy is matched by its upstream **domain** (the host of ``base_url``): the
public URL is ``/proxy/{domain}/{path}``. The upstream credential is not a
proxy field — it lives in each account's ``headers`` (whatever header a vendor
expects, e.g. ``authorization: Bearer ...`` or ``x-goog-api-key: ...``).
"""

from __future__ import annotations

import httpx
import pytest
import respx

from mm_gateway.config import BackendConfig, KeyConfig, ProxyConfig, Settings
from mm_gateway.observability.selection import STORE as SELECTION_STORE
from mm_gateway.server.app import create_app

# A shared upstream root the proxies below point at. respx routes are matched
# against the full URLs the ProxyRunner builds via _join_url(base_url, path).
_UPSTREAM = "https://upstream.example.test"
# The proxy's routing identity is the host of base_url (the {domain} segment).
_DOMAIN = "upstream.example.test"

_AUTH = {"Authorization": "Bearer t0pSecret"}


def _proxy(
    *,
    accounts: list | None = None,
    tags: list[str] | None = None,
    headers: dict | None = None,
    base_url: str = _UPSTREAM,
) -> ProxyConfig:
    return ProxyConfig(
        base_url=base_url,
        tags=tags or [],
        headers=headers or {},
        accounts=accounts or [],
    )


def _settings(proxies: list[ProxyConfig], *, allow_tags: list[str] | None = None) -> Settings:
    # A key that either is open (no allow_tags) or restricted to given tags.
    key = KeyConfig(id="test", key="t0pSecret", allow_tags=allow_tags or [])
    # A fake backend keeps create_app's registry happy without importing a real
    # provider module; the proxies are what we actually exercise.
    return Settings(
        backends=[BackendConfig(name="fake", type="fake", api_key="test")],
        keys=[key],
        max_sync_wait=5.0,
        poll_interval=0.01,
        proxies=proxies,
    )


@pytest.fixture(autouse=True)
def _clean_selection_store():
    """The selection store is a process-global singleton; account outcomes from
    one test (e.g. a 429 arming a 60s cooldown) would otherwise rank accounts in
    the next. Each proxy test starts — and ends — with an empty store."""
    SELECTION_STORE.clear()
    yield
    SELECTION_STORE.clear()


@pytest.fixture
def app_with_proxy():
    """An app + proxy whose upstream is the respx-intercepted _UPSTREAM root.

    Yields (app, proxy) so a test can poke state if needed; the respx router is
    started here so test routes add themselves on top.
    """
    SELECTION_STORE.clear()
    proxy = _proxy(accounts=[
        {"id": "acct-a", "headers": {"Authorization": "Bearer upstream-key-a"}},
    ])
    settings = _settings([proxy])
    app = create_app(settings)
    # The fake backend's provider never imports (type "fake" is unknown), so it
    # is silently skipped by the registry — that is fine for proxy tests.
    yield app, proxy
    SELECTION_STORE.clear()


@pytest.fixture
def client(app_with_proxy):
    from fastapi.testclient import TestClient

    app, _ = app_with_proxy
    with TestClient(app) as c:
        yield c


# -- Auth + authorization -------------------------------------------------- #


def test_missing_api_key_is_401(app_with_proxy):
    app, _ = app_with_proxy
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        resp = c.get(f"/proxy/{_DOMAIN}/v1/anything")
    assert resp.status_code == 401
    assert resp.headers["content-type"] == "application/problem+json"
    assert resp.json()["code"] == "unauthorized"


def test_unknown_proxy_domain_is_404(client):
    resp = client.get("/proxy/nope.example/v1/x", headers=_AUTH)
    assert resp.status_code == 404
    assert resp.json()["code"] == "generation_service_not_found"


def test_authorized_by_tag(client):
    # Rebuild the app with a tag-restricted key that DOES allow the proxy.
    proxy = _proxy(tags=["team-a"], accounts=[{"id": "a", "headers": {"Authorization": "Bearer key-a"}}])
    settings = _settings([proxy], allow_tags=["team-a"])
    from fastapi.testclient import TestClient

    with respx.mock:
        respx.get(f"{_UPSTREAM}/ok").mock(return_value=httpx.Response(200, text="hi"))
        with TestClient(create_app(settings)) as c:
            resp = c.get(f"/proxy/{_DOMAIN}/ok", headers=_AUTH)
    assert resp.status_code == 200
    assert resp.text == "hi"


# -- Forwarding ------------------------------------------------------------- #


def test_forwards_path_query_method_and_body(client):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = request.content
        return httpx.Response(201, json={"ok": True}, headers={"x-trace": "abc"})

    with respx.mock:
        respx.post(f"{_UPSTREAM}/v1/things").mock(side_effect=handler)
        resp = client.post(
            f"/proxy/{_DOMAIN}/v1/things?limit=5&tag=blue",
            headers={**_AUTH, "content-type": "application/json", "x-client-hdr": "c1"},
            json={"hello": "world"},
        )
    assert resp.status_code == 201
    assert resp.json() == {"ok": True}
    # Path + query forwarded verbatim to the upstream root.
    assert seen["method"] == "POST"
    assert seen["url"] == f"{_UPSTREAM}/v1/things?limit=5&tag=blue"
    assert seen["body"] == b'{"hello":"world"}'  # httpx re-serialises compactly
    # The configured account credential is injected; the client's is dropped.
    assert seen["headers"]["authorization"] == "Bearer upstream-key-a"
    assert seen["headers"]["x-client-hdr"] == "c1"
    # Account attribution header is present for observability.
    assert seen["headers"]["x-mm-gateway-proxy-account"] == "acct-a"
    # Upstream response headers flow back.
    assert resp.headers["x-trace"] == "abc"
    assert resp.headers["x-request-id"]  # gateway-injected


def test_get_with_path_param_segments_forwarded(client):
    with respx.mock:
        respx.get(f"{_UPSTREAM}/api/users/42/orders").mock(
            return_value=httpx.Response(200, text="ok")
        )
        resp = client.get(f"/proxy/{_DOMAIN}/api/users/42/orders", headers=_AUTH)
    assert resp.status_code == 200
    assert resp.text == "ok"


def test_client_credential_never_reaches_upstream(client):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = dict(request.headers)
        return httpx.Response(200, text="ok")

    with respx.mock:
        respx.get(f"{_UPSTREAM}/x").mock(side_effect=handler)
        # A caller-supplied Authorization and x-goog-api-key must be overwritten /
        # dropped: the proxy injects the configured account key instead.
        client.get(f"/proxy/{_DOMAIN}/x", headers={
            **_AUTH, "authorization": "Bearer evil-client-token",
            "x-goog-api-key": "evil-goog",
        })
    assert seen["headers"]["authorization"] == "Bearer upstream-key-a"
    assert "x-goog-api-key" not in seen["headers"]


def test_account_header_is_whatever_the_vendor_expects():
    # The credential is just a header on the account: x-goog-api-key (Google's
    # raw-key style) is injected verbatim, and no Authorization is added.
    proxy = _proxy(
        accounts=[{"id": "g", "headers": {"x-goog-api-key": "goog-raw-key"}}],
    )
    settings = _settings([proxy])
    from fastapi.testclient import TestClient

    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = dict(request.headers)
        return httpx.Response(200, text="ok")

    with respx.mock:
        respx.get(f"{_UPSTREAM}/v1/models").mock(side_effect=handler)
        with TestClient(create_app(settings)) as c:
            resp = c.get(f"/proxy/{_DOMAIN}/v1/models", headers=_AUTH)
    assert resp.status_code == 200
    # Raw key, no scheme prefix; the client's Authorization was dropped and the
    # account injects no Authorization of its own.
    assert seen["headers"]["x-goog-api-key"] == "goog-raw-key"
    assert "authorization" not in seen["headers"]


# -- Retry across accounts -------------------------------------------------- #


def test_retries_429_to_next_account():
    proxy = _proxy(accounts=[
        {"id": "a", "headers": {"Authorization": "Bearer key-a"}},
        {"id": "b", "headers": {"Authorization": "Bearer key-b"}},
    ])
    settings = _settings([proxy])
    from fastapi.testclient import TestClient

    tried = []

    def handler(request: httpx.Request) -> httpx.Response:
        acct = request.headers.get("x-mm-gateway-proxy-account")
        tried.append(acct)
        if acct == "a":
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, text="ok-from-b")

    with respx.mock:
        respx.get(f"{_UPSTREAM}/v1/x").mock(side_effect=handler)
        with TestClient(create_app(settings)) as c:
            resp = c.get(f"/proxy/{_DOMAIN}/v1/x", headers=_AUTH)
    assert resp.status_code == 200
    assert resp.text == "ok-from-b"
    # Account a was tried (and got 429), then b.
    assert tried == ["a", "b"]


def test_retries_5xx_to_next_account():
    proxy = _proxy(accounts=[
        {"id": "a", "headers": {"Authorization": "Bearer key-a"}},
        {"id": "b", "headers": {"Authorization": "Bearer key-b"}},
    ])
    settings = _settings([proxy])
    from fastapi.testclient import TestClient

    tried = []

    def handler(request: httpx.Request) -> httpx.Response:
        acct = request.headers.get("x-mm-gateway-proxy-account")
        tried.append(acct)
        if acct == "a":
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, text="ok-from-b")

    with respx.mock:
        respx.get(f"{_UPSTREAM}/v1/x").mock(side_effect=handler)
        with TestClient(create_app(settings)) as c:
            resp = c.get(f"/proxy/{_DOMAIN}/v1/x", headers=_AUTH)
    assert resp.status_code == 200
    assert resp.text == "ok-from-b"
    assert tried == ["a", "b"]


def test_client_4xx_not_retried_surfaces_verbatim():
    # A 400 is the caller's bad request — it must not burn another account.
    proxy = _proxy(accounts=[
        {"id": "a", "headers": {"Authorization": "Bearer key-a"}},
        {"id": "b", "headers": {"Authorization": "Bearer key-b"}},
    ])
    settings = _settings([proxy])
    from fastapi.testclient import TestClient

    tried = []

    def handler(request: httpx.Request) -> httpx.Response:
        tried.append(request.headers.get("x-mm-gateway-proxy-account"))
        return httpx.Response(400, json={"error": "bad request"})

    with respx.mock:
        respx.get(f"{_UPSTREAM}/v1/x").mock(side_effect=handler)
        with TestClient(create_app(settings)) as c:
            resp = c.get(f"/proxy/{_DOMAIN}/v1/x", headers=_AUTH)
    assert resp.status_code == 400
    assert resp.json() == {"error": "bad request"}
    # Only the first account was contacted.
    assert tried == ["a"]


def test_all_accounts_retryable_fail_surfaces_last_status():
    proxy = _proxy(accounts=[
        {"id": "a", "headers": {"Authorization": "Bearer key-a"}},
        {"id": "b", "headers": {"Authorization": "Bearer key-b"}},
    ])
    settings = _settings([proxy])
    from fastapi.testclient import TestClient

    tried = []

    def handler(request: httpx.Request) -> httpx.Response:
        tried.append(request.headers.get("x-mm-gateway-proxy-account"))
        return httpx.Response(500, text="boom")

    with respx.mock:
        respx.get(f"{_UPSTREAM}/v1/x").mock(side_effect=handler)
        with TestClient(create_app(settings)) as c:
            resp = c.get(f"/proxy/{_DOMAIN}/v1/x", headers=_AUTH)
    # On the last account the real upstream 500 is streamed back verbatim
    # rather than being wrapped in a 502.
    assert resp.status_code == 500
    assert resp.text == "boom"
    assert tried == ["a", "b"]


def test_proxy_with_no_configured_account_is_not_registered():
    # A proxy whose accounts carry no headers is unusable (configured is False),
    # so the registry never registers it: the route 404s rather than 503-ing.
    proxy = _proxy(accounts=[{"id": "a", "headers": {}}])
    settings = _settings([proxy])
    from fastapi.testclient import TestClient

    with TestClient(create_app(settings)) as c:
        resp = c.get(f"/proxy/{_DOMAIN}/v1/x", headers=_AUTH)
    assert resp.status_code == 404
    assert resp.json()["code"] == "generation_service_not_found"


# -- Event-stream (SSE) passthrough ---------------------------------------- #


def test_sse_response_streamed_verbatim():
    from fastapi.testclient import TestClient

    def handler(request: httpx.Request) -> httpx.Response:
        # An SSE stream: content-type preserved, body flows chunk by chunk.
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream", "cache-control": "no-cache"},
            content=b"data: chunk1\n\ndata: chunk2\n\n",
        )

    with respx.mock:
        respx.get(f"{_UPSTREAM}/v1/stream").mock(side_effect=handler)
        with TestClient(create_app(settings=_settings([_proxy(
            accounts=[{"id": "a", "headers": {"Authorization": "Bearer key-a"}}])]))) as c:
            resp = c.get(f"/proxy/{_DOMAIN}/v1/stream", headers=_AUTH)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "text/event-stream"
    assert resp.content == b"data: chunk1\n\ndata: chunk2\n\n"


def test_forbidden_when_tag_not_allowed(client):
    # Build an app whose key only allows the "team-a" tag but the proxy carries
    # "team-b": configured (so registered) but not usable by this key -> 403.
    proxy = _proxy(tags=["team-b"], accounts=[{"id": "a", "headers": {"Authorization": "Bearer key-a"}}])
    settings = _settings([proxy], allow_tags=["team-a"])
    from fastapi.testclient import TestClient

    with TestClient(create_app(settings)) as c:
        resp = c.get(f"/proxy/{_DOMAIN}/v1/x", headers=_AUTH)
    assert resp.status_code == 403
    assert resp.json()["code"] == "forbidden"


# -- Pure helper unit tests ------------------------------------------------ #


def test_enumerate_accounts_merges_proxy_headers_under_account_headers():
    # Proxy-level headers are a base; per-account headers override them. The
    # merged set (which carries the credential) is what the forwarder injects.
    proxy = _proxy(
        headers={"x-goog-user-project": "base", "x-shared": "from-proxy"},
        accounts=[{"id": "a", "headers": {
            "Authorization": "Bearer key-a", "x-shared": "from-account",
        }}],
    )
    accounts = proxy.enumerate_accounts()
    assert accounts == [(
        "a",
        {"x-goog-user-project": "base", "x-shared": "from-account",
         "Authorization": "Bearer key-a"},
    )]


def test_host_is_the_routing_identity_lowercased():
    assert ProxyConfig(base_url="https://API.Example.COM/v1").host == "api.example.com"
    # No parseable host falls back to the raw base_url (lowercased).
    assert ProxyConfig(base_url="not-a-url").host == "not-a-url"


def test_configured_is_false_when_no_account_has_headers():
    assert _proxy(accounts=[]).configured is False
    assert _proxy(accounts=[{"id": "a", "headers": {}}]).configured is False
    # Proxy-level headers alone make an account usable (an account inherits them).
    assert _proxy(
        headers={"x-goog-api-key": "k"},
        accounts=[{"id": "a", "headers": {}}],
    ).configured is True


def test_join_url_handles_leading_slash_and_empty_path():
    from mm_gateway.proxy import _join_url

    assert _join_url("https://up.test/", "v1/foo") == "https://up.test/v1/foo"
    assert _join_url("https://up.test", "/v1/foo") == "https://up.test/v1/foo"
    # An empty captured path forwards to the root.
    assert _join_url("https://up.test/", "") == "https://up.test"


def test_ws_url_rewrites_http_schemes_to_ws():
    from mm_gateway.proxy import _ws_url

    assert _ws_url("https://up.test", "ws") == "wss://up.test/ws"
    assert _ws_url("http://up.test", "ws") == "ws://up.test/ws"
    assert _ws_url("wss://up.test", "ws") == "wss://up.test/ws"
    assert _ws_url("ws://up.test", "ws") == "ws://up.test/ws"


# -- WebSocket bridge ------------------------------------------------------ #


def test_websocket_rejects_unknown_token():
    # A WS upgrade with an unknown bearer is closed before bridging (browsers
    # can't set headers on an upgrade, so the token also comes via query).
    from fastapi.testclient import TestClient

    proxy = _proxy(accounts=[{"id": "a", "headers": {"Authorization": "Bearer key-a"}}])
    settings = _settings([proxy])
    with TestClient(create_app(settings)) as c:
        with pytest.raises(Exception):
            with c.websocket_connect(
                f"/proxy/{_DOMAIN}/v1/rt?access_token=not-a-real-key"
            ) as ws:
                ws.receive()


async def test_websocket_bridge_pumps_both_directions_to_a_live_upstream():
    # Call bridge_websocket directly against a real upstream echo server on the
    # SAME event loop. Running it through Starlette's TestClient would put the
    # app on a worker thread/loop that cannot share a live asyncio server, so
    # the upstream connect would hang for the whole timeout. A fake client_ws
    # stands in for the gateway's WebSocket: it sends one frame, the upstream
    # echoes it, then closes — proving both directions pump and the credential
    # is injected on the upstream handshake.
    pytest.importorskip("websockets")
    import asyncio
    from websockets.asyncio.server import serve

    from mm_gateway.proxy import bridge_websocket

    upstream_received: list[str] = []
    upstream_headers: dict[str, str] = {}

    async def upstream_handler(ws):
        upstream_headers.update(dict(ws.request.headers))
        async for msg in ws:
            upstream_received.append(msg)
            await ws.send(f"echo:{msg}")
            return  # close after one echo -> the bridge's pump finishes

    client_sent: list[str] = []

    class FakeClientWS:
        """Minimal WebSocket surface bridge_websocket reads + pumps through."""
        def __init__(self, token: str):
            self._token = token
            self._incoming: asyncio.Queue = asyncio.Queue()
            self._sent: list[str] = []
            self.subprotocol = None

        # auth/route code reads these off the live WS before bridging:
        @property
        def headers(self):
            return {"authorization": f"Bearer {self._token}"}

        @property
        def url(self):
            class _U:
                query = ""
            return _U()

        async def accept(self, subprotocol=None):
            self.subprotocol = subprotocol

        async def receive(self):
            # Mirror Starlette's raw ASGI message shape: one websocket.receive
            # frame per queued client message, then a websocket.disconnect to
            # stop the client->upstream pump. bridge_websocket pumps text via
            # ``msg["text"]`` and bytes via ``msg["bytes"]`` (Starlette 1.4 has
            # no iter_data, so the pump reads the raw message, not a typed
            # async iterator).
            while True:
                msg = await self._incoming.get()
                if msg is None:
                    return {"type": "websocket.disconnect", "code": 1000}
                client_sent.append(msg)
                if isinstance(msg, bytes):
                    return {"type": "websocket.receive", "bytes": msg}
                return {"type": "websocket.receive", "text": msg}

        async def send_text(self, data):
            self._sent.append(data)

        async def send_bytes(self, data):
            self._sent.append(data)

        async def close(self, code=1000, reason=""):
            pass

    fake = FakeClientWS(token="t0pSecret")
    fake._incoming.put_nowait("ping")

    async with serve(upstream_handler, "127.0.0.1", 0) as server:
        port = server.socks[0].getsockname()[1] if hasattr(server, "socks") \
            else server.sockets[0].getsockname()[1]
        proxy = ProxyConfig(
            base_url=f"ws://127.0.0.1:{port}",
            accounts=[{"id": "a", "headers": {"Authorization": "Bearer upstream-key-a"}}],
        )
        await bridge_websocket(proxy, "rt", fake.url.query, fake.headers, fake)

    # The upstream saw the client's frame, and the echo came back to the client.
    assert upstream_received == ["ping"]
    assert fake._sent == ["echo:ping"]
    # The account credential was injected on the upstream handshake; the
    # client's bearer token did not leak upstream.
    assert upstream_headers.get("authorization") == "Bearer upstream-key-a"
    assert upstream_headers.get("x-mm-gateway-proxy-account") == "a"
