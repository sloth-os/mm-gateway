"""Tests for the HTTP MCP server (``/mcp``).

These exercise the full stack — FastAPI app lifespan, the
``StreamableHTTPSessionManager``, the delegating ``/mcp`` route, and the four
gateway tools (``list_models``, ``generate_image``, ``create_video``,
``get_video``) — over an in-process httpx ASGI transport talking the real MCP
Streamable-HTTP client protocol. No sockets, no network.

The fake provider is a multi-backend fixture so auth and routing can be
exercised through the MCP surface exactly as they are through the HTTP routes.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import MCPError

from mm_gateway.config import BackendConfig, KeyConfig, Settings
from mm_gateway.server.app import create_app
from tests.conftest import FakeProvider

# --------------------------------------------------------------------------- #
# Multi-backend fixture
# --------------------------------------------------------------------------- #


def _backends() -> list[BackendConfig]:
    return [
        BackendConfig(name="fake-img", type="fake", api_key="k", tags=["image-primary"]),
        BackendConfig(name="fake-vid", type="fake", api_key="k", tags=["video-primary"]),
        BackendConfig(name="fake-mus", type="fake", api_key="k", tags=["music-primary"]),
        BackendConfig(name="fake-other", type="fake", api_key="k", tags=["other"]),
    ]


def _keys() -> list[KeyConfig]:
    return [
        # A real-token key that may use every backend.
        KeyConfig(id="alice", key="alice-token",
                  allow_tags=["image-primary", "video-primary", "music-primary", "other"]),
        # A key pinned to one backend by name.
        KeyConfig(id="bob", key="bob-token", allow_backends=["fake-img"]),
        # A key whose allow_tags match no configured backend, so every call is
        # forbidden — used to assert GatewayError surfaces as an MCPError.
        KeyConfig(id="dave", key="dave-token", allow_tags=["restricted"]),
        # An open key (empty token) — admits any caller, no header required.
        KeyConfig(id="open", key=""),
    ]


def _keys_no_open() -> list[KeyConfig]:
    """The key set above minus the open key — for tests that need auth enforced."""
    return [k for k in _keys() if k.id != "open"]


def _make_app(backends, keys):
    settings = Settings(
        backends=backends, keys=keys,
        mcp_enabled=True, mcp_path="/mcp",
        video_sync_default=True, max_sync_wait=5.0, poll_interval=0.01,
    )
    app = create_app(settings)
    # Inject the fake provider into each backend slot, bypassing the registry's
    # import-based construction (there is no real "fake" provider module).
    for cfg in settings.backends:
        app.state.registry._backends[cfg.name] = FakeProvider(cfg)
        app.state.registry._configs[cfg.name] = cfg
    return app


@pytest.fixture
def mcp_app():
    return _make_app(_backends(), _keys())


@asynccontextmanager
async def _lifespan(app):
    """Enter the app lifespan once — this starts the MCP session manager's
    ``run()`` context, which may be entered only once per instance.

    Tests that need to act as more than one API key must share a single
    ``_lifespan`` block and open a separate ``_session`` per identity inside it,
    rather than re-entering the lifespan (which would try to start the session
    manager a second time and raise ``RuntimeError``).
    """
    async with app.router.lifespan_context(app):
        yield


@asynccontextmanager
async def _session(app, token: str | None):
    """Yield an initialised MCP ``ClientSession`` over an in-process ASGI
    transport for ``token``. Does not manage the app lifespan — the caller must
    hold a ``_lifespan`` block open around it. A fresh ``httpx.AsyncClient``
    (and thus a fresh MCP session id) is created per call, so different tokens
    are isolated even within one lifespan.
    """
    headers = {"authorization": f"Bearer {token}"} if token else {}
    transport = httpx.ASGITransport(app=app)
    http_client = httpx.AsyncClient(transport=transport, headers=headers)
    async with streamable_http_client(
        "http://testserver/mcp", http_client=http_client,
    ) as (read, write), ClientSession(read, write) as sess:
        await sess.initialize()
        yield sess


@asynccontextmanager
async def _client(app, token: str | None):
    """Yield a single ``ClientSession`` for ``token`` with the app lifespan running.

    Convenience for tests that act as one identity: opens one lifespan and one
    session inside it. Tests needing more than one identity should use
    ``_lifespan`` + ``_session`` directly (see
    ``test_poll_denied_when_key_not_authorised_for_tasks_backend``).
    """
    async with _lifespan(app), _session(app, token) as sess:
        yield sess


def _first_leaf(exc: BaseException, leaf_type: type) -> BaseException | None:
    """Walk a (possibly nested) ``BaseExceptionGroup`` and return the first leaf
    that is an instance of ``leaf_type``, or None.

    The MCP Streamable-HTTP client and session manager run their read/write loops
    in anyio ``TaskGroup``s; a protocol error raised by ``call_tool`` (and any
    task cancelled during teardown) is therefore delivered to the caller wrapped
    in nested ``BaseExceptionGroup``s. ``pytest.raises(MCPError)`` cannot match a
    group, so ``_call`` unwraps to the real ``MCPError`` leaf.
    """
    if isinstance(exc, BaseExceptionGroup):
        for sub in exc.exceptions:
            leaf = _first_leaf(sub, leaf_type)
            if leaf is not None:
                return leaf
        return None
    return exc if isinstance(exc, leaf_type) else None


async def _call(app, token: str | None, tool: str, args: dict | None = None):
    """Open a session, call ``tool``, return ``(is_error, text)``.

    A protocol-level error (e.g. an auth failure surfaced as ``MCPError``)
    propagates to the caller — unwrapped from its TaskGroup envelope — so tests
    can assert on it with ``pytest.raises(MCPError)``.
    """
    try:
        async with _client(app, token) as sess:
            res = await sess.call_tool(tool, args or {})
            text = res.content[0].text if res.content else None
            return res.is_error, text
    except BaseException as exc:  # unwrap TaskGroup envelope — see _first_leaf
        leaf = _first_leaf(exc, MCPError)
        if leaf is not None:
            raise leaf from None
        raise


# --------------------------------------------------------------------------- #
# Tool listing
# --------------------------------------------------------------------------- #


async def test_lists_all_gateway_tools(mcp_app):
    async with _client(mcp_app, "alice-token") as sess:
        tools = await sess.list_tools()
    names = sorted(t.name for t in tools.tools)
    assert names == ["create_music", "create_video", "generate_image",
                     "get_music", "get_video", "list_models"]


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #


async def test_missing_token_is_rejected_when_no_open_key():
    # Build an app whose only key requires a real token, so a missing token is a
    # genuine 401 rather than an open-key admit. (The MCP server closes over the
    # settings object captured at app-construction time, so the no-open-key case
    # must be built from its own settings — mutating app.state.settings after the
    # fact would not reach the already-built tools.)
    app = _make_app(_backends(), [k for k in _keys() if k.id != "open"])
    with pytest.raises(MCPError) as exc:
        await _call(app, None, "list_models")
    assert "Missing API key" in str(exc.value)


async def test_unknown_token_is_rejected():
    # An open key (empty token) admits any caller, including an unknown one, so
    # "unknown token rejected" only holds when no open key is configured. Build
    # the app from its own (open-key-free) settings for the same closure reason
    # as test_missing_token_is_rejected_when_no_open_key.
    app = _make_app(_backends(), _keys_no_open())
    with pytest.raises(MCPError) as exc:
        await _call(app, "not-a-real-token", "list_models")
    assert "Unknown API key" in str(exc.value)


async def test_open_key_admits_any_caller(mcp_app):
    # No token at all, yet the call succeeds because the open key matches.
    is_error, text = await _call(mcp_app, None, "list_models")
    assert not is_error
    assert "fake-image-1" in text


async def test_valid_token_succeeds(mcp_app):
    is_error, text = await _call(mcp_app, "alice-token", "list_models")
    assert not is_error
    data = json.loads(text)
    ids = [m["id"] for m in data["data"]]
    assert "fake-image-1" in ids and "fake-video-1" in ids and "fake-music-1" in ids


# --------------------------------------------------------------------------- #
# Image generation
# --------------------------------------------------------------------------- #


async def test_generate_image_returns_openai_shape(mcp_app):
    is_error, text = await _call(mcp_app, "alice-token", "generate_image",
                                 {"model": "fake-image-1", "prompt": "a cat"})
    assert not is_error, text
    body = json.loads(text)
    assert body["data"][0]["url"] == "https://example.test/out.png"
    assert body["data"][0]["revised_prompt"] == "a cat"


async def test_generate_image_routes_to_pinned_backend(mcp_app):
    # Bob is pinned to fake-img; verify the request landed there.
    is_error, text = await _call(mcp_app, "bob-token", "generate_image",
                                 {"model": "fake-image-1", "prompt": "x"})
    assert not is_error, text
    prov = mcp_app.state.registry._backends["fake-img"]
    assert prov.image_calls and prov.image_calls[0].provider == "fake-img"


async def test_generate_image_tag_routing_via_tool_arg(mcp_app):
    # Alice may use any backend; pin via the ``backend`` arg to fake-other.
    is_error, text = await _call(mcp_app, "alice-token", "generate_image",
                                 {"model": "fake-image-1", "prompt": "x", "backend": "fake-other"})
    assert not is_error, text
    prov = mcp_app.state.registry._backends["fake-other"]
    assert prov.image_calls and prov.image_calls[0].provider == "fake-other"


# --------------------------------------------------------------------------- #
# Video create + poll
# --------------------------------------------------------------------------- #


async def test_create_video_returns_task_id(mcp_app):
    is_error, text = await _call(mcp_app, "alice-token", "create_video", {
        "model": "fake-video-1",
        "content": [{"type": "text", "text": "a cat playing"}],
        "wait": True,
    })
    assert not is_error, text
    assert json.loads(text) == {"id": "task-1"}


async def test_get_video_polls_to_succeeded(mcp_app):
    # The fake provider transitions pending -> running -> succeeded across
    # polls. create_video with wait=True blocks until terminal (succeeded) via
    # VideoService._await_or_timeout; a follow-up get_video re-confirms the
    # terminal state and the final artefact url. Both calls ride one
    # lifespan/session — the session manager is single-use per instance.
    async with _client(mcp_app, "alice-token") as sess:
        res = await sess.call_tool("create_video", {
            "model": "fake-video-1",
            "content": [{"type": "text", "text": "a cat playing"}],
            "wait": True,
        })
        assert not res.is_error, res.content
        task_id = json.loads(res.content[0].text)["id"]

        res = await sess.call_tool("get_video", {"id": task_id})
        assert not res.is_error, res.content
        body = json.loads(res.content[0].text)
    assert body["id"] == task_id
    assert body["status"] == "succeeded"
    assert body["content"]["video_url"] == "https://example.test/out.mp4"


# --------------------------------------------------------------------------- #
# Music create + poll (Gemini Lyria 3 shape)
# --------------------------------------------------------------------------- #


async def test_create_music_returns_interaction_id(mcp_app):
    is_error, text = await _call(mcp_app, "alice-token", "create_music", {
        "model": "fake-music-1",
        "input": "an upbeat pop song",
        "wait": True,
        # Pin to the music backend so we can assert the call landed there; without
        # this, resolution picks the first usable backend that serves the model
        # (all fake backends do), which would be fake-img.
        "backend": "fake-mus",
    })
    assert not is_error, text
    assert json.loads(text) == {"id": "music-1"}
    prov = mcp_app.state.registry._backends["fake-mus"]
    assert prov.music_calls and prov.music_calls[0].prompt() == "an upbeat pop song"


async def test_create_music_accepts_parts_input(mcp_app):
    # ``input`` as a Lyria parts array is accepted; text parts concatenate.
    is_error, text = await _call(mcp_app, "alice-token", "create_music", {
        "model": "fake-music-1",
        "input": [{"type": "text", "text": "verse"}, {"type": "text", "text": "chorus"}],
        "wait": True,
        "backend": "fake-mus",
    })
    assert not is_error, text
    prov = mcp_app.state.registry._backends["fake-mus"]
    assert prov.music_calls[0].prompt() == "verse\nchorus"


async def test_get_music_polls_to_succeeded(mcp_app):
    # Async create returns immediately; a follow-up get_music re-polls to the
    # terminal state and the Lyria steps/content envelope (audio + lyrics blocks).
    async with _client(mcp_app, "alice-token") as sess:
        res = await sess.call_tool("create_music", {
            "model": "fake-music-1",
            "input": "a sad ballad",
            "wait": False,  # async, so we get the id back immediately
            "backend": "fake-mus",
        })
        assert not res.is_error, res.content
        task_id = json.loads(res.content[0].text)["id"]

        # Poll until terminal — the fake provider moves pending->running->succeeded
        # across polls, so a couple of get_music calls converge.
        body = None
        for _ in range(10):
            res = await sess.call_tool("get_music", {"id": task_id})
            assert not res.is_error, res.content
            body = json.loads(res.content[0].text)
            if body["status"] == "succeeded":
                break
    assert body is not None
    assert body["id"] == task_id
    assert body["status"] == "succeeded"
    step = body["steps"][0]
    assert step["type"] == "model_output"
    blocks = step["content"]
    assert any(b["type"] == "audio" and b["data"] == "AAAA" for b in blocks)
    assert any(b["type"] == "text" and b["text"] == "la la la" for b in blocks)
    assert body["output_audio"] == "AAAA"
    assert body["output_text"] == "la la la"


# --------------------------------------------------------------------------- #
# Forbidden key surfaces as a structured MCP error
# --------------------------------------------------------------------------- #


async def test_key_with_no_usable_backend_gets_forbidden(mcp_app):
    # Dave's allow_tags match no configured backend, so every call is a 403.
    # The ``_tool`` decorator must surface that ``ForbiddenError`` as an
    # ``MCPError`` carrying the gateway's code/message — not a bare 500.
    with pytest.raises(MCPError) as exc:
        await _call(mcp_app, "dave-token", "create_video", {
            "model": "fake-video-1",
            "content": [{"type": "text", "text": "x"}],
            "wait": True,
        })
    msg = str(exc.value)
    assert "not allowed" in msg or "forbidden" in msg.lower()


async def test_poll_denied_when_key_not_authorised_for_tasks_backend(mcp_app):
    # Alice creates a video pinned (via the `backend` arg) to fake-vid. Bob is
    # pinned to fake-img, so he is not authorised for fake-vid — polling alice's
    # task must surface as an MCPError, not a cross-tenant leak.
    #
    # Two identities ride one ``_lifespan`` (the session manager's ``run()`` is
    # single-use per instance), each on its own ``_session`` so the bearer token
    # is isolated. Bob's forbidden call raises an ``MCPError`` that anyio wraps
    # in a ``BaseExceptionGroup`` on the way out of the client task group, so the
    # ``_first_leaf`` unwrapper recovers it before asserting.
    async with _lifespan(mcp_app):
        async with _session(mcp_app, "alice-token") as sess:
            res = await sess.call_tool("create_video", {
                "model": "fake-video-1",
                "content": [{"type": "text", "text": "x"}],
                "backend": "fake-vid",
                "wait": False,  # async, so we get the id back immediately
            })
            assert not res.is_error, res.content
            task_id = json.loads(res.content[0].text)["id"]

        raised: MCPError | None = None
        try:
            async with _session(mcp_app, "bob-token") as sess:
                await sess.call_tool("get_video", {"id": task_id})
        except BaseException as exc:  # unwrap TaskGroup envelope — see _first_leaf
            raised = _first_leaf(exc, MCPError)  # type: ignore[assignment]
            if raised is None:
                raise
    assert raised is not None
    msg = str(raised)
    assert "not allowed" in msg or "forbidden" in msg.lower()


# --------------------------------------------------------------------------- #
# MCP disabled by default
# --------------------------------------------------------------------------- #


async def test_mcp_disabled_when_not_enabled():
    settings = Settings(
        backends=_backends(), keys=_keys(), mcp_enabled=False, mcp_path="/mcp",
    )
    app = create_app(settings)
    paths = [getattr(r, "path", None) for r in app.router.routes]
    assert "/mcp" not in paths


async def test_mcp_session_idle_timeout_is_wired_through(mcp_app):
    # A stateful session manager must be constructed with a finite
    # session_idle_timeout, else a client that initialises and then vanishes
    # keeps its session (and serve-loop task) alive for the gateway's whole
    # lifetime — unbounded growth. The timeout defaults to 1800s in Settings;
    # here we verify the configured value reaches the manager.
    sm = mcp_app.state.mcp_session_manager
    assert sm.session_idle_timeout == mcp_app.state.settings.mcp_session_idle_timeout
    assert sm.session_idle_timeout == 1800
    assert not sm.stateless  # stateful — the timeout only applies in this mode


async def test_mcp_session_idle_timeout_is_configurable():
    # A non-default timeout must flow from Settings through to the manager.
    settings = Settings(
        backends=_backends(), keys=_keys(),
        mcp_enabled=True, mcp_path="/mcp", mcp_session_idle_timeout=42,
    )
    app = create_app(settings)
    for cfg in settings.backends:
        app.state.registry._backends[cfg.name] = FakeProvider(cfg)
        app.state.registry._configs[cfg.name] = cfg
    assert app.state.mcp_session_manager.session_idle_timeout == 42


# --------------------------------------------------------------------------- #
# The /mcp route is a streaming ASGI passthrough
# --------------------------------------------------------------------------- #


async def test_mcp_endpoint_is_streaming_passthrough():
    # The /mcp route must hand the raw ASGI ``send`` straight to the session
    # manager rather than buffering the body into a single Response — the
    # long-lived GET SSE stream only delivers live if chunks flush as they are
    # produced. Drive ``_StreamPassthrough`` with a fake session manager that
    # emits a streaming sequence (start, body with more_body=True, final body)
    # and assert every message — including the ``more_body=True`` marker — is
    # forwarded to ``send`` verbatim. A buffering implementation would collapse
    # this into a single ``http.response.body`` with no ``more_body`` flag.
    from mm_gateway.server.mcp import _StreamPassthrough

    class _FakeSM:
        def __init__(self):
            self.received_scope = None

        async def handle_request(self, scope, receive, send):
            self.received_scope = scope
            await send({"type": "http.response.start", "status": 200,
                        "headers": [(b"content-type", b"text/event-stream")]})
            await send({"type": "http.response.body", "body": b"data: ping\n\n",
                        "more_body": True})
            await send({"type": "http.response.body", "body": b"data: pong\n\n",
                        "more_body": False})

    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    fake = _FakeSM()
    passthrough = _StreamPassthrough(fake)
    await passthrough({"type": "http", "method": "GET"}, receive, send)

    assert fake.received_scope == {"type": "http", "method": "GET"}
    assert sent[0] == {"type": "http.response.start", "status": 200,
                       "headers": [(b"content-type", b"text/event-stream")]}
    # The intermediate chunk MUST retain more_body=True — this is the marker a
    # buffering route would drop, and the one SSE delivery depends on.
    assert sent[1] == {"type": "http.response.body", "body": b"data: ping\n\n",
                       "more_body": True}
    assert sent[2] == {"type": "http.response.body", "body": b"data: pong\n\n",
                       "more_body": False}
    # Three distinct messages, not one buffered amalgam.
    assert len(sent) == 3
