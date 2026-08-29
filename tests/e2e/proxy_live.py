#!/usr/bin/env python3
"""Real Gemini Live (AI Studio) WebSocket e2e for the general proxy.

A companion to ``tests/e2e/smoke.py`` (which drives the generation contract
against real image/video/music providers). This script drives the gateway's
**general pass-through proxy** over a real upstream Gemini Live (bidirectional
realtime) WebSocket call, using the google-genai SDK's ``client.aio.live`` API —
the same SDK a real client uses — pointed at ``/proxy/<domain>/...`` so the
gateway's WebSocket bridge, front-end auth, and account injection are all
exercised end-to-end against Google's Generative Language API.

This is the AI Studio / Generative Language surface, NOT Vertex: the Live 3.x
models (``gemini-3.1-flash-live-preview``) are AI-Studio-only, so the SDK must
take its mldev path (an ``api_key`` client). That path is what forces
``wss://`` and injects ``x-goog-api-key`` — both of which the gateway handles:

* the SDK always upgrades to ``wss`` (``_websocket_base_url`` does
  ``urlparse(base_url)._replace(scheme='wss')`` whenever an api_key is set), and
  the CI gateway serves plain ``http``/``ws``, so the SDK cannot connect to it
  directly. A tiny in-process TLS terminator relays the SDK's ``wss`` bytes to
  the gateway's plain ``ws`` listener — a protocol-agnostic TCP pump that adds
  no WebSocket framing of its own.
* the SDK authenticates upstream with an ``x-goog-api-key`` header, which the
  gateway **drops** and replaces with the configured account key; the SDK also
  sends our ``Authorization: Bearer`` front-end key (set via
  ``http_options.headers``), which the gateway reads off the WS upgrade to
  authenticate the caller. So the test proves the gateway injects the upstream
  credential and never leaks it to the client.

When to run
-----------
Runs only when ``PROXY_API_KEY`` + ``PROXY_MODEL`` are both set — it does **not**
require ``GATEWAY_API_KEY``. The front-end key is optional: when set, the
gateway is protected and the script authenticates with it; when unset, the
gateway's env key has an empty token (an "open" key that admits any caller, per
``resolve_key``), so the script sends no ``Authorization`` header and still
authenticates. So the proxy e2e can run with just the Gemini upstream key +
model, independent of the front-end key the provider modalities share. When the
pair is unset it exits 0 (skips), so the workflow is green before secrets are
wired and only spends a real Live call once the proxy is fully configured.

Exit codes: 0 ok / skipped, 1 the gateway proxy path is broken (auth,
forwarding, or the real upstream call failed), 2 misconfigured.
"""

from __future__ import annotations

import asyncio
import ipaddress
import os
import ssl
import tempfile
import time
from pathlib import Path

import httpx

import google.genai as genai
from google.genai import types

# The google-genai SDK is a transitive dep of the gateway (google-genai>=2.19);
# cryptography (self-signed cert generation) and websockets (SDK transport) come
# with it, so no extra runner deps are needed beyond `pip install google-genai`.
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

BASE = os.environ.get("MM_GATEWAY", "http://127.0.0.1:8000").rstrip("/")
TOKEN = os.environ.get("GATEWAY_API_KEY", "")
PROXY_API_KEY = os.environ.get("PROXY_API_KEY", "")
PROXY_MODEL = os.environ.get("PROXY_MODEL", "")
# The proxy is matched by its upstream domain (the host of base_url). The
# gateway's default proxy (from PROXY_API_KEY) targets the AI Studio Generative
# Language host; PROXY_BASE_URL overrides it, and the routing domain follows.
from urllib.parse import urlparse
_PROXY_BASE_URL = os.environ.get(
    "PROXY_BASE_URL", "https://generativelanguage.googleapis.com"
)
PROXY_DOMAIN = urlparse(_PROXY_BASE_URL).hostname or _PROXY_BASE_URL
LIVE_TURN_TIMEOUT = float(os.environ.get("PROXY_LIVE_TIMEOUT", "60"))


def log(msg: str) -> None:
    print(f"[e2e-proxy] {msg}", flush=True)


def configured() -> bool:
    """Run iff the proxy pair is set: PROXY_API_KEY (Gemini upstream key) +
    PROXY_MODEL (the Live model id). GATEWAY_API_KEY is *optional*: when set
    the gateway is token-protected and the script authenticates with it; when
    unset the gateway's env key has an empty token (an "open" key that admits
    any caller), so the script needs no front-end key at all. So the proxy
    e2e stands alone on the Gemini key + model, independent of the front-end
    key the provider modalities share."""
    return bool(PROXY_API_KEY and PROXY_MODEL)


def auth_headers() -> dict[str, str]:
    # No front-end key when GATEWAY_API_KEY is unset: the gateway's env key has
    # an empty token, which resolve_key treats as an open key admitting any
    # caller. Sending no Authorization header still authenticates there.
    return {"authorization": f"Bearer {TOKEN}"} if TOKEN else {}


def wait_for_health(client: httpx.Client) -> None:
    deadline = time.time() + 60
    last = ""
    while time.time() < deadline:
        try:
            r = client.get("/health", timeout=5)
            if r.status_code == 200:
                return
            last = f"status {r.status_code}"
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
        time.sleep(1)
    raise RuntimeError(f"gateway did not become healthy within 60s (last: {last})")


# -- TLS terminator --------------------------------------------------------- #
# The SDK forces wss; the CI gateway is plain ws. A pure asyncio TLS->plaintext
# TCP relay bridges them without touching WebSocket framing: the SDK's TLS bytes
# are terminated here, the decrypted stream is pumped to the gateway's plain
# listener, and replies are pumped back. Protocol-agnostic, so it cannot
# introduce WS framing bugs.


def _make_self_signed_cert() -> tuple[str, str]:
    """Return ``(cert_path, key_path)`` for a self-signed localhost cert."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    import datetime as _dt  # local: only needed for cert validity
    now = _dt.datetime.now(_dt.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + _dt.timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
            ]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    d = Path(tempfile.mkdtemp(prefix="mm-proxy-live-"))
    cert_path = str(d / "cert.pem")
    key_path = str(d / "key.pem")
    Path(cert_path).write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    Path(key_path).write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    return cert_path, key_path


def _server_ssl_ctx(cert_path: str, key_path: str) -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=cert_path, keyfile=key_path)
    return ctx


def _client_unverified_ssl_ctx() -> ssl.SSLContext:
    """Trust our self-signed terminator without pinning the cert."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


async def _pipe(src: asyncio.StreamReader, dst: asyncio.StreamWriter) -> None:
    try:
        while True:
            data = await src.read(65536)
            if not data:
                break
            dst.write(data)
            await dst.drain()
    except (ConnectionError, asyncio.IncompleteReadError, OSError):
        pass
    finally:
        try:
            dst.close()
        except OSError:
            pass


async def _handle_relay(client_reader, client_writer, up_host: str, up_port: int) -> None:
    try:
        up_reader, up_writer = await asyncio.open_connection(up_host, up_port)
    except OSError:
        client_writer.close()
        return
    await asyncio.gather(
        _pipe(client_reader, up_writer),
        _pipe(up_reader, client_writer),
        return_exceptions=True,
    )
    try:
        client_writer.close()
    except OSError:
        pass


async def _start_tls_relay(
    cert_path: str, key_path: str, up_host: str, up_port: int
) -> asyncio.base_events.Server:
    """Start a TLS listener that relays decrypted bytes to the plain gateway."""
    server = await asyncio.start_server(
        lambda r, w: _handle_relay(r, w, up_host, up_port),
        host="127.0.0.1",
        port=0,
        ssl=_server_ssl_ctx(cert_path, key_path),
    )
    return server


def _relay_port(server: asyncio.base_events.Server) -> int:
    return server.sockets[0].getsockname()[1]


# -- Live turn + main ------------------------------------------------------- #


async def _live_turn_via_proxy(relay_port: int) -> str:
    """Drive one text-in/text-out Gemini Live turn through the gateway proxy.

    The SDK client is pointed at the TLS relay so its forced-``wss`` connect
    hits ``wss://127.0.0.1:<relay>/proxy/<PROXY_DOMAIN>``; the relay terminates
    TLS and pumps the bytes to the plain-HTTP gateway, which bridges the WS
    upgrade to ``wss://generativelanguage.googleapis.com/...`` (the AI Studio
    upstream) with the configured account's ``x-goog-api-key``. When a front-end
    ``GATEWAY_API_KEY`` is set, the SDK passes our ``Authorization: Bearer``
    front-end key on the handshake (via ``http_options.headers``) and the
    gateway authenticates the caller with it; when it is unset the gateway's
    env key is open and the upgrade carries no Authorization header, which
    still authenticates. Either way the front-end header is dropped before the
    upstream call.
    """
    base_url = f"https://127.0.0.1:{relay_port}/proxy/{PROXY_DOMAIN}"
    # Attach the front-end bearer only when a key is set; an open gateway needs
    # none. The SDK puts http_options.headers on the WS handshake, so the
    # gateway reads the bearer off the upgrade to authenticate the caller.
    ws_headers = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}
    client = genai.Client(
        api_key=PROXY_API_KEY,
        http_options=types.HttpOptions(
            base_url=base_url,
            api_version="v1beta",
            headers=ws_headers,
            # Trust our self-signed terminator; the SDK passes this ssl context
            # to ws_connect via _websocket_ssl_ctx (only filled in if unset).
            async_client_args={"ssl": _client_unverified_ssl_ctx()},
        ),
    )

    prompt = os.environ.get("PROXY_LIVE_PROMPT", "Say 'guest pass' and nothing else.")
    text: list[str] = []
    async with client.aio.live.connect(
        model=PROXY_MODEL,
        config={"response_modalities": ["TEXT"]},
    ) as session:
        await session.send_client_content(
            turns=types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            ),
            turn_complete=True,
        )
        async for message in session.receive():
            # The Live turn is text; a single modelTurn carries the reply. Stop
            # once we have it so we never hang waiting for more frames than the
            # model emits. ``message.text`` is None/sentinel until parts exist.
            if getattr(message, "text", None):
                text.append(message.text)
            # server_content with a turn_complete flag ends the turn; the SDK
            # surfaces it via .server_content when present.
            sc = getattr(message, "server_content", None)
            if sc is not None and getattr(sc, "turn_complete", False):
                break
    reply = "".join(text).strip()
    if not reply:
        raise RuntimeError("Live turn completed but the model returned no text")
    return reply


async def _run() -> int:
    log(f"target gateway: {BASE}")
    log(f"proxy domain: {PROXY_DOMAIN}  model: {PROXY_MODEL}")

    with httpx.Client(base_url=BASE, timeout=30) as client:
        wait_for_health(client)
        log("gateway healthy")
        # Sanity: the proxy must be visible to this caller's key. A 404 here
        # means PROXY_API_KEY never registered the proxy (env not threaded into
        # the container); a 403 means the key isn't authorised for it. Both are
        # caught up front rather than surfacing as an opaque WS close.
        proxy_probe = client.get(
            f"/proxy/{PROXY_DOMAIN}/_probe", headers=auth_headers(), timeout=15,
        )
        # The gateway forwards the probe verbatim; Google returns 404/400 for an
        # unknown path, but a *gateway* 401/403/404 distinguishes front-end/auth
        # problems from an upstream one. Only the gateway-level failures fail.
        if proxy_probe.status_code in (401, 403):
            raise RuntimeError(
                f"gateway rejected proxy access: {proxy_probe.status_code} "
                f"{proxy_probe.text[:200]} (is GATEWAY_API_KEY / PROXY_API_KEY set "
                f"in the gateway env?)"
            )
        if proxy_probe.status_code == 404:
            raise RuntimeError(
                f"proxy domain {PROXY_DOMAIN!r} is not configured on the gateway "
                f"(PROXY_API_KEY not threaded into the container?)"
            )
        log(f"proxy reachable (probe -> {proxy_probe.status_code}; upstream "
            f"rejection of /_probe is expected and fine)")

    cert_path, key_path = _make_self_signed_cert()
    # Relay to the gateway's plain-HTTP listener. Parse host/port from BASE.
    parsed = urlparse(BASE)
    up_host = parsed.hostname or "127.0.0.1"
    up_port = parsed.port or 8000
    server = await _start_tls_relay(cert_path, key_path, up_host, up_port)
    relay_port = _relay_port(server)
    # Keep the relay's accept loop alive for the duration of the turn.
    try:
        log(f"TLS relay on 127.0.0.1:{relay_port} -> plain {up_host}:{up_port}")
        reply = await asyncio.wait_for(
            _live_turn_via_proxy(relay_port), timeout=LIVE_TURN_TIMEOUT
        )
    finally:
        server.close()
        await server.wait_closed()

    assert reply, "empty reply after successful turn"
    log(f"Live reply through proxy: {reply[:200]!r}")
    log("REAL upstream Gemini Live call succeeded through the gateway proxy")
    return 0


def main() -> int:
    if not configured():
        log(
            "PROXY_API_KEY + PROXY_MODEL not set — skipping Gemini Live proxy e2e "
            "(GATEWAY_API_KEY is optional and not required)"
        )
        return 0
    try:
        return asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001
        log(f"FAILED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

