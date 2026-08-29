"""Tests for configurable HTTP / SOCKS5 outbound proxying.

Two surfaces must route their *outbound* traffic through a configured proxy:

* every backend provider adapter builds its upstream client (httpx, the
  genai/volcengine/openai SDKs' injected httpx client, runapi's transport, or
  the dashscope SDK's aiohttp session) from ``backend.extra["outbound_proxy"]``;
* the pass-through proxy layer (``ProxyRunner._client`` httpx forwarder +
  ``bridge_websocket``'s websockets connection) reads ``proxy.outbound_proxy``.

The registry folds the global ``Settings.outbound_proxy`` onto both when no
per-target override is set (backend / proxy override wins). These tests prove
the URL reaches the right client without any network: httpx exposes the proxy
URL on the connection pool, the SDK clients expose their injected httpx client,
the aiohttp connector carries its proxy type, and websockets records the
``proxy`` argument it was given. respx bypasses httpx proxy mounts, so the
proxy-routing *behaviour* (a request actually leaving through the proxy) is
proved separately with a live local HTTP proxy server.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from mm_gateway.config import (
    BackendConfig,
    KeyConfig,
    ProxyConfig,
    Settings,
    _is_socks_proxy,
    _resolve_outbound_proxy,
    _with_outbound_proxy,
)
from mm_gateway.proxy import ProxyRunner
from mm_gateway.registry import Registry


# -- introspection helpers ------------------------------------------------- #


def _httpx_proxy_url(client: httpx.AsyncClient) -> str | None:
    """The proxy URL an httpx client was built with, or None when direct.

    httpx installs an ``AsyncHTTPProxy`` connection pool (carrying
    ``_proxy_url``) on the proxy's mount; a direct client's pool has no such
    attribute. Used instead of respx (which bypasses proxy mounts).
    """
    for transport in client._mounts.values():  # noqa: SLF001 — introspection
        pool = getattr(transport, "_pool", None)  # noqa: SLF001
        url = getattr(pool, "_proxy_url", None)  # noqa: SLF001
        if url is not None:
            return f"{url.scheme.decode()}://{url.host.decode()}:{url.port}"
    return None


def _backend(name: str = "openai", type_: str = "openai", *,
             extra: dict[str, Any] | None = None,
             api_key: str = "sk-test",
             base_url: str | None = None) -> BackendConfig:
    kw: dict[str, Any] = {"name": name, "type": type_, "api_key": api_key}
    if base_url is not None:
        kw["base_url"] = base_url
    if extra:
        kw["extra"] = extra
    return BackendConfig(**kw)


# -- config: helpers + global / per-backend / per-proxy resolution -------- #


def test_is_socks_proxy_classifies_schemes() -> None:
    assert _is_socks_proxy("socks5://127.0.0.1:1080")
    assert _is_socks_proxy("socks5h://127.0.0.1:1080")
    assert _is_socks_proxy("socks4://127.0.0.1:1080")
    assert _is_socks_proxy("socks4a://127.0.0.1:1080")
    assert not _is_socks_proxy("http://127.0.0.1:3128")
    assert not _is_socks_proxy("https://proxy:443")
    assert not _is_socks_proxy(None)
    assert not _is_socks_proxy("")


def test_resolve_outbound_proxy_normalises() -> None:
    assert _resolve_outbound_proxy("  http://proxy:3128  ") == "http://proxy:3128"
    assert _resolve_outbound_proxy("socks5://proxy:1080") == "socks5://proxy:1080"
    # Empty / None -> not-set here (no env fallback unless asked).
    assert _resolve_outbound_proxy("") is None
    assert _resolve_outbound_proxy(None) is None


def test_resolve_outbound_proxy_env_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OUTBOUND_PROXY", "http://from-env:3128")
    assert _resolve_outbound_proxy(None, env_fallback=True) == "http://from-env:3128"
    # An explicit value still wins over the env var.
    assert _resolve_outbound_proxy("socks5://explicit:1080",
                                  env_fallback=True) == "socks5://explicit:1080"
    # Without env_fallback the env var is NOT consulted.
    assert _resolve_outbound_proxy(None) is None


def test_with_outbound_proxy_folds_into_extra() -> None:
    extra = _with_outbound_proxy({"region": "cn"}, "http://proxy:3128")
    assert extra == {"region": "cn", "outbound_proxy": "http://proxy:3128"}

    # An explicit extra value wins over the top-level shorthand.
    extra = _with_outbound_proxy({"outbound_proxy": "socks5://a:1080"},
                                 "http://b:3128")
    assert extra["outbound_proxy"] == "socks5://a:1080"

    # None drops the key (no literal "None" string leaks in).
    extra = _with_outbound_proxy({"region": "cn"}, None)
    assert "outbound_proxy" not in extra


def test_settings_yaml_reads_global_and_per_target(
        tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OUTBOUND_PROXY", raising=False)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "outbound_proxy: http://global:3128\n"
        "backends:\n"
        "  - name: oa\n"
        "    type: openai\n"
        "    api_key: sk-x\n"
        "    outbound_proxy: socks5://per-backend:1080\n"
        "  - name: oa2\n"
        "    type: openai\n"
        "    api_key: sk-y\n"
        "    extra:\n"
        "      outbound_proxy: socks5://extra-wins:1080\n"
        "    outbound_proxy: http://shorthand-loses:3128\n"
        "proxies:\n"
        "  - base_url: https://up.example\n"
        "    outbound_proxy: http://per-proxy:3128\n"
        "keys:\n"
        "  - id: alice\n"
        "    key: tok\n"
    )
    s = Settings.from_file(cfg)
    assert s.outbound_proxy == "http://global:3128"
    assert s.backends[0].extra["outbound_proxy"] == "socks5://per-backend:1080"
    assert s.backends[1].extra["outbound_proxy"] == "socks5://extra-wins:1080"
    assert s.proxies[0].outbound_proxy == "http://per-proxy:3128"


def test_legacy_env_global_outbound_proxy(monkeypatch: pytest.MonkeyPatch,
                                          tmp_path) -> None:
    """The env-var fallback layout reads OUTBOUND_PROXY + per-backend stems."""
    monkeypatch.chdir(tmp_path)  # no ./mm-gateway.{yaml,yml} in scope
    monkeypatch.delenv("MM_GATEWAY_CONFIG", raising=False)
    monkeypatch.setenv("OUTBOUND_PROXY", "http://global-env:3128")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setenv("OPENAI_OUTBOUND_PROXY", "socks5://oa-env:1080")
    monkeypatch.delenv("GATEWAY_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("VERTEX_*", raising=False)
    s = Settings.from_env()
    assert s.outbound_proxy == "http://global-env:3128"
    oa = next(b for b in s.backends if b.name == "openai")
    assert oa.extra["outbound_proxy"] == "socks5://oa-env:1080"


# -- registry: global folded in, per-target override wins ------------------ #


def test_registry_folds_global_outbound_proxy_onto_backend() -> None:
    """``_per_account_config`` sets ``outbound_proxy`` to the global when the
    backend itself did not pin one, so the provider reads one resolved value."""
    r = Registry(Settings(outbound_proxy="http://global:3128"))
    cfg = _backend("oa", "openai", api_key="sk-x")
    per = r._per_account_config(cfg, "default", "sk-x", None, {})
    assert per.extra["outbound_proxy"] == "http://global:3128"
    # The account id is still stashed.
    assert per.extra["__account_id"] == "default"


def test_registry_backend_override_wins_over_global() -> None:
    """A backend-level override is already in ``extra``; setdefault leaves it."""
    r = Registry(Settings(outbound_proxy="http://global:3128"))
    cfg = _backend("oa", "openai", api_key="sk-x")
    per = r._per_account_config(
        cfg, "default", "sk-x", None, {"outbound_proxy": "socks5://override:1080"},
    )
    assert per.extra["outbound_proxy"] == "socks5://override:1080"


def test_registry_bakes_global_outbound_proxy_onto_proxy() -> None:
    proxy = ProxyConfig(
        base_url="https://up.example",
        accounts=[{"headers": {"authorization": "Bearer k"}}],
    )
    r = Registry(Settings(
        proxies=[proxy],
        keys=[KeyConfig(id="alice", key="tok", allow_backends=["up.example"])],
        outbound_proxy="http://global:3128",
    ))
    assert r.proxy("up.example").outbound_proxy == "http://global:3128"


def test_registry_per_proxy_override_wins_over_global() -> None:
    proxy = ProxyConfig(
        base_url="https://up.example",
        accounts=[{"headers": {"authorization": "Bearer k"}}],
        outbound_proxy="socks5://per-proxy:1080",
    )
    r = Registry(Settings(
        proxies=[proxy],
        keys=[KeyConfig(id="alice", key="tok", allow_backends=["up.example"])],
        outbound_proxy="http://global:3128",
    ))
    assert r.proxy("up.example").outbound_proxy == "socks5://per-proxy:1080"


# -- provider adapters: proxy reaches the real httpx/aiohttp client -------- #
#
# Each adapter builds its upstream client from ``backend.extra["outbound_proxy"]``
# in ``__init__``. Rather than mock the SDK (and risk asserting on a mock
# rather than the real wiring), we construct the real provider and introspect
# the httpx client it built (the proxy URL lives on the connection pool).
# No network: client construction never sends a request.


def test_openai_provider_threads_proxy_into_sdk_http_client() -> None:
    from mm_gateway.providers.openai import OpenAIProvider
    p = OpenAIProvider(_backend(
        "https://image.test",
        extra={"outbound_proxy": "http://proxy:3128", "video_base_url": "https://video.test"},
    ))
    # AsyncOpenAI stores the injected httpx client on ``_client``.
    assert _httpx_proxy_url(p._client._client) == "http://proxy:3128"  # noqa: SLF001
    assert _httpx_proxy_url(p._client_video._client) == "http://proxy:3128"  # noqa: SLF001


def test_openai_provider_socks_proxy() -> None:
    from mm_gateway.providers.openai import OpenAIProvider
    p = OpenAIProvider(_backend(
        "https://image.test",
        extra={"outbound_proxy": "socks5://proxy:1080"},
    ))
    assert _httpx_proxy_url(p._client._client) == "socks5://proxy:1080"  # noqa: SLF001


def test_openai_provider_no_proxy_is_direct() -> None:
    """Without ``outbound_proxy`` the client is direct (env-honoring)."""
    from mm_gateway.providers.openai import OpenAIProvider
    p = OpenAIProvider(_backend("https://image.test"))
    assert _httpx_proxy_url(p._client._client) is None  # noqa: SLF001


def test_volcengine_provider_threads_proxy_into_ark_client() -> None:
    from mm_gateway.providers.volcengine import VolcengineProvider
    p = VolcengineProvider(_backend(
        "https://ark.test", type_="volcengine",
        extra={"outbound_proxy": "socks5://proxy:1080"},
    ))
    assert _httpx_proxy_url(p._ark._client) == "socks5://proxy:1080"  # noqa: SLF001
    assert _httpx_proxy_url(p._ark_video._client) == "socks5://proxy:1080"  # noqa: SLF001


def test_google_provider_threads_proxy_into_genai_and_lyria_clients() -> None:
    from mm_gateway.providers.google import GoogleProvider
    p = GoogleProvider(_backend(
        "https://genai.test", type_="google",
        extra={"outbound_proxy": "http://proxy:3128",
               "video_base_url": "https://video.test",
               "music_base_url": "https://music.test"},
    ))
    # genai SDK stores the injected httpx client at
    # ``_aio._api_client._async_httpx_client``.
    img = p._client._aio._api_client._async_httpx_client  # noqa: SLF001
    vid = p._client_video._aio._api_client._async_httpx_client  # noqa: SLF001
    assert _httpx_proxy_url(img) == "http://proxy:3128"
    assert _httpx_proxy_url(vid) == "http://proxy:3128"


def test_vertex_provider_threads_proxy_into_genai_clients(
        monkeypatch: pytest.MonkeyPatch) -> None:
    # Vertex resolves real ADC in __init__; we only care that the proxy reaches
    # the genai client, so stub credential discovery with a dummy credential +
    # project. The genai client does not validate credentials at construction.
    from mm_gateway.providers import vertex as vertex_mod
    from mm_gateway.providers.vertex import VertexProvider
    monkeypatch.setattr(VertexProvider, "_resolve_credentials",
                        staticmethod(lambda backend: (object(), "proj", "us-central1")))
    p = VertexProvider(_backend(
        type_="vertex", api_key="", base_url="https://vertex.test",
        extra={"outbound_proxy": "http://proxy:3128"},
    ))
    img = p._client._aio._api_client._async_httpx_client  # noqa: SLF001
    assert _httpx_proxy_url(img) == "http://proxy:3128"
    vid = p._client_video._aio._api_client._async_httpx_client  # noqa: SLF001
    assert _httpx_proxy_url(vid) == "http://proxy:3128"


def test_flux_provider_threads_proxy_into_runapi_transport() -> None:
    from mm_gateway.providers.flux import FluxProvider
    p = FluxProvider(_backend(
        "https://flux.test", type_="flux",
        extra={"outbound_proxy": "socks5://proxy:1080"},
    ))
    # runapi: Flux2Client._http (HttpClient) -> _client/_upload_client (httpx
    # sync clients) -> _transport._pool (an HTTPProxy/SOCKSProxy pool).
    def _pool_proxy(http_client) -> str | None:
        pool = http_client._transport._pool  # noqa: SLF001
        u = getattr(pool, "_proxy_url", None)  # noqa: SLF001
        if u is None:
            return None
        return f"{u.scheme.decode()}://{u.host.decode()}:{u.port}"
    assert _pool_proxy(p._client._http._client) == "socks5://proxy:1080"  # noqa: SLF001
    assert _pool_proxy(p._client._http._upload_client) == "socks5://proxy:1080"  # noqa: SLF001


def test_dashscope_provider_aiohttp_session_uses_proxy() -> None:
    """The aiohttp SDK session carries a ProxyConnector for the proxy URL, and
    the httpx *poll* clients carry the same proxy (submit + poll same egress)."""
    from mm_gateway.providers.dashscope import DashScopeProvider
    p = DashScopeProvider(_backend(
        "https://dashscope.test", type_="dashscope",
        extra={"outbound_proxy": "socks5://proxy:1080"},
    ))
    asyncio.run(_assert_dashscope_proxy(p))


async def _assert_dashscope_proxy(p) -> None:
    sess = await p._aio_session()  # noqa: SLF001
    assert sess is not None
    con = sess.connector
    # aiohttp_socks ProxyConnector resolves the proxy into _proxy_host/port and
    # a _proxy_type enum (SOCKS5/HTTP...).
    from python_socks import ProxyType
    assert con._proxy_host == "proxy"  # noqa: SLF001
    assert con._proxy_port == 1080  # noqa: SLF001
    assert con._proxy_type == ProxyType.SOCKS5  # noqa: SLF001
    assert sess.trust_env is False  # explicit connector is authoritative
    await sess.close()
    # The httpx poll clients share the proxy.
    assert _httpx_proxy_url(p._client_image) == "socks5://proxy:1080"  # noqa: SLF001
    assert _httpx_proxy_url(p._client_video) == "socks5://proxy:1080"  # noqa: SLF001


def test_dashscope_provider_no_proxy_returns_none_session() -> None:
    """Without a proxy the SDK falls back to its shared (env-honoring) session."""
    from mm_gateway.providers.dashscope import DashScopeProvider
    p = DashScopeProvider(_backend("https://dashscope.test", type_="dashscope"))
    assert asyncio.run(p._aio_session()) is None  # noqa: SLF001
    assert _httpx_proxy_url(p._client_image) is None  # noqa: SLF001


def test_dashscope_proxy_without_aiohttp_socks_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured proxy on the dashscope (aiohttp) path needs aiohttp-socks —
    for *any* scheme, since the dedicated ProxyConnector session is built for HTTP
    too. A missing extra must raise a clear ProviderRequestError naming the
    package, not an opaque ImportError at first use."""
    import builtins
    from mm_gateway.core.exceptions import ProviderRequestError
    from mm_gateway.providers import dashscope as ds

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "aiohttp_socks":
            raise ImportError("simulated: aiohttp-socks not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(ProviderRequestError) as exc_info:
        # HTTP scheme: aiohttp path builds the connector regardless of scheme,
        # so the missing-extra guard fires for HTTP proxies, not only SOCKS.
        ds._aiohttp_proxy_connector("http://proxy:3128")
    assert "aiohttp-socks" in exc_info.value.message
    assert "mm-gateway[socks]" in exc_info.value.message


# -- stability input-image fetch honours the outbound proxy --------------- #


async def test_stability_input_image_routes_through_outbound_proxy() -> None:
    """A caller-supplied ``http://`` input image is fetched through the backend's
    outbound proxy (the same egress as the upstream calls), so a proxy-only
    network can pull the image. Proved live: the fetch leaves through the local
    forward proxy to reach the bare upstream that serves the image bytes."""
    from mm_gateway.providers.stability import StabilityProvider, _decode_image_input

    proxy_saw: list[str] = []
    upstream_saw: list[tuple[str, str, dict[str, str]]] = []
    proxy_url, upstream_url, close = await _serve_local_forward_proxy(
        proxy_saw, upstream_saw)
    image_url = upstream_url + "/img.png"
    try:
        # ``_decode_image_input`` is a *sync* httpx fetch; run it in a thread so
        # the asyncio forward proxy (same loop) can serve the relayed connection
        # instead of deadlocking behind the blocking call.
        bytes_png, mime = await asyncio.to_thread(_decode_image_input, image_url, proxy_url)
        assert mime.startswith("text/plain")  # bare upstream sends text/plain
        assert bytes_png == b"ok"
        assert proxy_saw and proxy_saw[0].startswith("GET http://")
        assert upstream_saw[0][:2] == ("GET", "/img.png")

        # no proxy / unset: a direct fetch reaches the upstream without the relay.
        proxy_saw.clear(); upstream_saw.clear()
        await asyncio.to_thread(_decode_image_input, image_url, None)
        assert not proxy_saw
        assert upstream_saw[0][:2] == ("GET", "/img.png")

        # The provider wires its resolved proxy onto the helper. Build a provider
        # instance and confirm the stored proxy matches its config.
        p = StabilityProvider(_backend(
            "https://api.stability.test", type_="stability",
            extra={"outbound_proxy": proxy_url},
        ))
        assert p._proxy_url == proxy_url  # noqa: SLF001
    finally:
        await close()


# -- live local HTTP forward proxy: the forwarder actually routes through it #
#
# respx bypasses httpx proxy mounts (a dead-proxy request returns the mocked
# 200), so it cannot prove a request *leaves* through the proxy. Instead we
# stand up a real local forward proxy + a real local upstream and drive the
# ProxyRunner's own httpx client (``_client(proxy)``, the exact client
# ``forward()`` uses) through them. The upstream records what it received; the
# proxy records the relayed request line. A dead-proxy URL failing proves the
# wiring is a real route, not a happy-path coincidence.


async def _serve_local_forward_proxy(proxy_saw: list[str],
                                      upstream_saw: list[tuple[str, str, dict[str, str]]]):
    """Start a forward proxy (absolute-form) + a bare upstream; return URLs.

    httpx routes plain-HTTP through a proxy via the absolute-form forward path
    (``GET http://host/path``), so the proxy parses that, opens a raw TCP
    connection to the upstream, rewrites to origin-form, and relays both
    directions. Returns ``(proxy_url, upstream_url, closer)``.
    """
    from urllib.parse import urlparse

    async def upstream_handler(reader, writer):
        try:
            line = await reader.readline()
            parts = line.decode().split()
            method, path_qs = parts[0], parts[1]
            headers: dict[str, str] = {}
            while True:
                h = await reader.readline()
                if h in (b"\r\n", b"\n", b""):
                    break
                k, _, v = h.decode().partition(":")
                headers[k.strip().lower()] = v.strip()
            upstream_saw.append((method, path_qs, headers))
            writer.write(b'HTTP/1.1 200 OK\r\ncontent-type: text/plain\r\n'
                         b'content-length: 2\r\n\r\nok')
            await writer.drain()
        finally:
            try:
                writer.close()
            except Exception:  # noqa: BLE001
                pass

    async def proxy_handler(reader, writer):
        try:
            line = await reader.readline()
            proxy_saw.append(line.decode().strip())
            parts = line.decode().split()
            method, abs_url = parts[0], parts[1]
            headers: dict[str, str] = {}
            while True:
                h = await reader.readline()
                if h in (b"\r\n", b"\n", b""):
                    break
                k, _, v = h.decode().partition(":")
                headers[k.strip().lower()] = v.strip()
            pu = urlparse(abs_url)
            up_r, up_w = await asyncio.open_connection(pu.hostname, pu.port or 80)
            path_qs = pu.path or "/"
            if pu.query:
                path_qs += "?" + pu.query
            up_w.write(f"{method} {path_qs} HTTP/1.1\r\n".encode())
            up_w.write(f"host: {pu.netloc}\r\n".encode())
            for k, v in headers.items():
                if k == "host":
                    continue
                up_w.write(f"{k}: {v}\r\n".encode())
            up_w.write(b"\r\n")

            async def pump(src, dst):
                try:
                    while data := await src.read(4096):
                        dst.write(data)
                        await dst.drain()
                except Exception:  # noqa: BLE001
                    pass
                try:
                    dst.close()
                except Exception:  # noqa: BLE001
                    pass

            await asyncio.gather(pump(reader, up_w), pump(up_r, writer))
        except Exception:  # noqa: BLE001 — client gone / partial read
            pass
        finally:
            try:
                writer.close()
            except Exception:  # noqa: BLE001
                pass

    up_srv = await asyncio.start_server(upstream_handler, "127.0.0.1", 0)
    px_srv = await asyncio.start_server(proxy_handler, "127.0.0.1", 0)
    up_port = up_srv.sockets[0].getsockname()[1]
    px_port = px_srv.sockets[0].getsockname()[1]
    upstream_url = f"http://127.0.0.1:{up_port}"
    proxy_url = f"http://127.0.0.1:{px_port}"

    async def close() -> None:
        up_srv.close()
        px_srv.close()
        await up_srv.wait_closed()
        await px_srv.wait_closed()

    return proxy_url, upstream_url, close


async def test_proxy_forwarder_routes_through_live_proxy():
    """The ProxyRunner httpx client actually tunnels through a real forward
    proxy: the proxy saw the relayed request, and the upstream received it."""
    proxy_saw: list[str] = []
    upstream_saw: list[tuple[str, str, dict[str, str]]] = []
    proxy_url, upstream_url, close = await _serve_local_forward_proxy(
        proxy_saw, upstream_saw)
    runner = ProxyRunner()
    proxy = ProxyConfig(
        base_url=upstream_url,
        accounts=[{"headers": {"x-echo": "acct"}}],
        outbound_proxy=proxy_url,
    )
    try:
        client = runner._client(proxy)  # noqa: SLF001 — the exact client forward() uses
        resp = await client.get(upstream_url + "/v1/hello")
        assert resp.status_code == 200
        assert resp.text == "ok"
        # The proxy saw the absolute-form forward request line.
        assert proxy_saw, "forward proxy never saw the tunneled request"
        assert proxy_saw[0].startswith("GET http://")
        # The upstream received the rewritten origin-form request.
        assert upstream_saw, "upstream never received the request"
        method, path_qs, _headers = upstream_saw[0]
        assert (method, path_qs) == ("GET", "/v1/hello")
        await client.aclose()
    finally:
        await runner.aclose()
        await close()


async def test_proxy_forwarder_dead_proxy_fails():
    """A dead (unreachable) proxy URL must make the request fail — proving the
    client really routes through it rather than going direct. This is the
    behaviour respx could not prove (it bypasses mounts)."""
    runner = ProxyRunner()
    proxy = ProxyConfig(
        base_url="http://127.0.0.1:1",
        accounts=[{"headers": {}}],
        outbound_proxy="http://127.0.0.1:1",  # nothing listening on port 1
    )
    try:
        client = runner._client(proxy)  # noqa: SLF001
        with pytest.raises(httpx.HTTPError):
            await client.get("http://127.0.0.1:1/x")
    finally:
        await runner.aclose()


async def test_proxy_forwarder_no_proxy_is_direct():
    """Without ``outbound_proxy`` the ProxyRunner client is direct: the request
    reaches the upstream with no relay hop and the proxy-saw list stays empty."""
    proxy_saw: list[str] = []
    upstream_saw: list[tuple[str, str, dict[str, str]]] = []
    _, upstream_url, close = await _serve_local_forward_proxy(
        proxy_saw, upstream_saw)
    runner = ProxyRunner()
    proxy = ProxyConfig(
        base_url=upstream_url,
        accounts=[{"headers": {"x-echo": "acct"}}],
        outbound_proxy=None,  # direct
    )
    try:
        client = runner._client(proxy)  # noqa: SLF001
        resp = await client.get(upstream_url + "/v1/direct")
        assert resp.status_code == 200
        assert not proxy_saw, "direct request should not touch the proxy"
        assert upstream_saw[0][1] == "/v1/direct"
        await client.aclose()
    finally:
        await runner.aclose()
        await close()


# -- WebSocket bridge: outbound-proxy translation + SOCKS-guard ------------ #


def test_ws_connect_proxy_translates_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_ws_connect_proxy`` maps the resolved outbound proxy to a value
    ``websockets.connect(proxy=...)`` accepts: ``True`` (env-honoring default)
    when unset, the URL straight through for HTTP/SOCKS."""
    from mm_gateway.proxy import _ws_connect_proxy
    assert _ws_connect_proxy(None) is True  # no override -> ambient env vars
    assert _ws_connect_proxy("http://proxy:3128") == "http://proxy:3128"
    assert _ws_connect_proxy("socks5://proxy:1080") == "socks5://proxy:1080"


def test_ws_connect_proxy_socks_without_extra_raises(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A SOCKS URL whose ``python-socks`` extra is missing raises a clear
    ``ProviderRequestError`` rather than an opaque handshake-time ImportError."""
    import builtins
    from mm_gateway.core.exceptions import ProviderRequestError
    from mm_gateway.proxy import _ws_connect_proxy

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "python_socks":
            raise ImportError("simulated: python-socks not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(ProviderRequestError) as exc_info:
        _ws_connect_proxy("socks5://proxy:1080")
    assert "python-socks" in exc_info.value.message


async def test_bridge_websocket_rejects_when_socks_proxy_misconfigured(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """When the resolved SOCKS outbound proxy cannot be loaded, the bridge
    rejects the client upgrade (4503) before attempting any upstream connect."""
    import builtins
    from mm_gateway.proxy import bridge_websocket

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "python_socks":
            raise ImportError("simulated: python-socks not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    class _RecordingClientWS:
        def __init__(self) -> None:
            self.subprotocol = None
            self.closed_calls: list[tuple[int, str]] = []

        @property
        def headers(self):
            return {"authorization": "Bearer t0pSecret"}

        @property
        def url(self):
            class _U:
                query = ""
            return _U()

        async def accept(self, subprotocol=None):
            self.subprotocol = subprotocol

        async def receive(self):
            await asyncio.Event().wait()
            return {"type": "websocket.disconnect", "code": 1000}

        async def send_text(self, data):
            pass

        async def send_bytes(self, data):
            pass

        async def close(self, code=1000, reason=""):
            self.closed_calls.append((int(code), str(reason or "")))

    proxy = ProxyConfig(
        base_url="ws://127.0.0.1:1",  # never reached
        accounts=[{"id": "a", "headers": {"Authorization": "Bearer k"}}],
        outbound_proxy="socks5://proxy:1080",
    )
    fake = _RecordingClientWS()
    await bridge_websocket(proxy, "rt", "", fake.headers, fake)
    # The client was rejected before any upstream connect attempt.
    assert fake.closed_calls, "bridge did not reject the misconfigured client"
    code, reason = fake.closed_calls[0]
    assert code == 4503
    assert "proxy" in reason.lower()
