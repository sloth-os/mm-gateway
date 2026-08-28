"""General pass-through proxy: forward HTTP / WebSocket / event-stream
requests to an upstream service with multi-account scheduling and retry.

The public URL is ``/proxy/{name}/{path:path}``. The gateway authenticates the
front-end caller with the usual bearer key and authorizes it against the
proxy's tags (the same hybrid rule as provider backends), then forwards the
request verbatim — method, the captured ``path``, query string, body, and
client headers — to the upstream ``base_url`` root, injecting one configured
account's credential as a header.

A proxy fronts a *pool* of upstream accounts (:meth:`ProxyConfig.enumerate_accounts`).
The gateway ranks them by live health (success rate + latency + rate-limit
cooldown — the same selection store auto-routing uses) and retries the next
account on a rate-limit (429), timeout, or upstream 5xx, so one proxy name
fronts a pool of upstream keys. Client-side 4xx is not retried across
accounts — the caller's request was rejected, not the account.

HTTP responses (including SSE / event streams) are streamed back chunk by
chunk; a WebSocket upgrade on the same path is bridged to an upstream
ws/wss connection in both directions. The account credential never reaches
the client: it is injected only on the request to the upstream.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
from fastapi import Request
from fastapi.responses import Response, StreamingResponse

from mm_gateway.config import ProxyConfig
from mm_gateway.core.exceptions import (
    GatewayError,
    ProviderRequestError,
    ProviderTimeoutError,
)
from mm_gateway.observability.httplog import backend_event_hooks
from mm_gateway.observability.logging import get_logger
from mm_gateway.observability.selection import STORE as SELECTION_STORE

log = get_logger("proxy")


# Header names that carry the *upstream* account credential, hop-by-hop
# transport headers, and other values the gateway owns for a forwarded
# request. The client's copy (if any) is never forwarded: the proxy injects
# the configured account's value instead, and hop-by-hop headers are invalid
# to relay. Host is derived from the upstream URL.
_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}

# Client headers that must not be copied onto the upstream request.
_DROP_CLIENT_HEADERS = _HOP_BY_HOP | {"authorization", "x-goog-api-key", "api-key", "apikey"}


def proxy_auth_header(proxy: ProxyConfig, api_key: str | None) -> dict[str, str]:
    """Render the credential header for an account, per the proxy config.

    ``auth_header="Authorization"`` (default) with ``auth_scheme="Bearer"``
    produces ``{"Authorization": "Bearer <key>"}``; set
    ``auth_header="x-goog-api-key"`` and ``auth_scheme=None`` for Google's
    raw-key header. An account with no key contributes no auth header (its
    request may still carry static ``headers`` from the config).
    """
    if not api_key:
        return {}
    if proxy.auth_scheme:
        return {proxy.auth_header: f"{proxy.auth_scheme} {api_key}"}
    return {proxy.auth_header: api_key}


def _selection_keys(proxy: ProxyConfig, account_id: str) -> dict[str, Any]:
    """Keyword args for the selection store keyed on the proxy + account."""
    return {
        "backend": proxy.name,
        "account": account_id,
        "model": "*",
        "modality": "proxy",
    }


def _ranked_accounts(proxy: ProxyConfig) -> list[tuple[str, str | None, str | None, dict[str, str], float]]:
    """Return ``(account_id, api_key, base_url, headers, score)`` best-first.

    A non-rate-limited, healthy, fast account ranks first; one in a
    rate-limit cooldown sinks to the bottom but is still returned (the cooldown
    may be stale). Returns one entry per credential.
    """
    ranked: list[tuple[float, float, str, str | None, str | None, dict[str, str]]] = []
    for account_id, api_key, base_url, headers in proxy.enumerate_accounts():
        keys = _selection_keys(proxy, account_id)
        score = SELECTION_STORE.score(**keys)
        rate_limited = SELECTION_STORE.is_rate_limited(**keys)
        # rate_limited (0/1) sorts first so healthy accounts win; negate score
        # so a higher health score ranks earlier; account_id is the stable tie-break.
        ranked.append((1 if rate_limited else 0, -score, account_id, api_key, base_url, headers))
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    return [(aid, ak, bu, h, -neg_score) for (_, neg_score, aid, ak, bu, h) in ranked]


def _is_retryable_status(status: int) -> bool:
    """True iff an upstream status warrants trying the next account.

    429 (rate-limit) and 5xx (unstable upstream) are retried across accounts;
    a 4xx is the caller's request being rejected, so the response is surfaced
    verbatim rather than burning another account on the same bad request.
    """
    return status == 429 or status >= 500


class ProxyRunner:
    """Builds and owns the per-proxy httpx clients and forwards requests.

    One shared :class:`httpx.AsyncClient` per proxy keeps TLS/connection pooling
    across requests; account-specific headers and (rare) per-account
    ``base_url`` overrides are applied per request. Clients are created lazily
    and live for the process, so the runner is a singleton on ``app.state``.
    """

    def __init__(self) -> None:
        self._clients: dict[str, httpx.AsyncClient] = {}

    def _client(self, proxy: ProxyConfig) -> httpx.AsyncClient:
        client = self._clients.get(proxy.name)
        if client is None:
            client = httpx.AsyncClient(
                timeout=httpx.Timeout(proxy.timeout, connect=10.0),
                # Keep the *request* hook (logs the masked upstream credential +
                # the account-attribution header) but NOT the response hook: the
                # response hook materialises the body via aread(), which consumes
                # the stream the proxy must hand to the client via aiter_raw().
                event_hooks=backend_event_hooks(log_response=False),
                follow_redirects=False,
            )
            self._clients[proxy.name] = client
        return client

    @staticmethod
    def _forward_headers(client_headers: Any, proxy: ProxyConfig,
                         account_id: str, api_key: str | None,
                         account_headers: dict[str, str]) -> dict[str, str]:
        """Merge client + proxy + account headers for the upstream request.

        The client's hop-by-hop and credential headers are dropped (the proxy
        injects the configured account's credential); proxy-level and account
        headers are applied over the forwarded client headers; ``host`` is left
        for httpx to derive from the upstream URL.
        """
        out: dict[str, str] = {}
        # ``request.headers`` is a Starlette ``Headers`` object (not a dict):
        # iterating it (or ``list(...)``) yields just the header *names*, so use
        # ``items()`` to get the (name, value) pairs. dict, Starlette ``Headers``,
        # and httpx ``Headers`` all expose ``items()``.
        if hasattr(client_headers, "items"):
            items = list(client_headers.items())
        else:
            items = list(client_headers)
        for name, value in items:
            if name.lower() in _DROP_CLIENT_HEADERS:
                continue
            out[name] = value
        # Proxy-level static headers, then per-account headers (account wins),
        # then the credential header injected last so it always overrides any
        # client-supplied value with the configured account's key.
        out.update(proxy.headers)
        out.update(account_headers)
        out.update(proxy_auth_header(proxy, api_key))
        out["x-mm-gateway-proxy-account"] = account_id
        return out

    async def aclose(self) -> None:
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()

    async def forward(
        self,
        proxy: ProxyConfig,
        method: str,
        path: str,
        request: Request,
    ) -> Response:
        """Forward an HTTP request, retrying across ranked accounts.

        Reads the client body once, then tries each account best-first: it
        sends the request and waits for the upstream response *headers*. A
        retryable failure (timeout, transport error, 429, 5xx) closes that
        response and tries the next account; the first non-retryable response
        (2xx / 4xx) is streamed back to the client verbatim, so event-stream
        (SSE) and large bodies flow through without buffering.

        Every attempt is timed and recorded in the selection store so the next
        request's ranking reflects what just happened. The last failure is
        surfaced when every account is exhausted.
        """
        body = await request.body()
        query_params = list(request.query_params.multi_items())
        accounts = _ranked_accounts(proxy)
        if not accounts:
            raise GatewayError(
                f"Proxy '{proxy.name}' has no configured account.",
                code="provider_not_configured", status_code=503, provider=proxy.name,
            )

        client = self._client(proxy)
        last_exc: BaseException | None = None
        for index, (account_id, api_key, base_url, account_headers, _score) in enumerate(accounts):
            upstream_url = _join_url(base_url or proxy.base_url, path)
            headers = self._forward_headers(
                request.headers, proxy, account_id, api_key, account_headers,
            )
            start = time.monotonic()
            try:
                upstream_req = client.build_request(
                    method, upstream_url, params=query_params, content=body, headers=headers,
                )
                resp = await client.send(upstream_req, stream=True)
            except httpx.TimeoutException as exc:
                latency = time.monotonic() - start
                self._observe(proxy, account_id, "failure", latency, rate_limited=False)
                log.info("proxy_attempt_failed", proxy=proxy.name, account=account_id,
                         method=method, url=upstream_url, latency_s=round(latency, 3),
                         error=f"timeout: {exc}", retryable=True)
                last_exc = ProviderTimeoutError(
                    f"proxy '{proxy.name}' upstream timed out: {exc}", provider=proxy.name,
                )
                continue
            except httpx.HTTPError as exc:
                latency = time.monotonic() - start
                self._observe(proxy, account_id, "failure", latency, rate_limited=False)
                log.info("proxy_attempt_failed", proxy=proxy.name, account=account_id,
                         method=method, url=upstream_url, latency_s=round(latency, 3),
                         error=f"transport: {exc}", retryable=True)
                last_exc = ProviderRequestError(
                    f"proxy '{proxy.name}' transport error: {exc}", provider=proxy.name,
                )
                continue

            latency = time.monotonic() - start
            if _is_retryable_status(resp.status_code) and index + 1 < len(accounts):
                # Read+close so the connection is released back to the pool
                # before we try the next account.
                await resp.aclose()
                rate_limited = resp.status_code == 429
                self._observe(proxy, account_id, "failure", latency, rate_limited=rate_limited)
                log.info("proxy_attempt_failed", proxy=proxy.name, account=account_id,
                         method=method, url=upstream_url, latency_s=round(latency, 3),
                         status=resp.status_code, rate_limited=rate_limited, retryable=True)
                last_exc = ProviderRequestError(
                    f"proxy '{proxy.name}' upstream returned HTTP {resp.status_code}",
                    provider=proxy.name, status_code=resp.status_code,
                )
                continue

            # Non-retryable (2xx / 4xx): stream the upstream response back. A
            # 429/5xx on the *last* account is also streamed verbatim here so
            # the client sees the real upstream status rather than a wrapped 502.
            outcome = "failure" if resp.status_code >= 500 or resp.status_code == 429 else "success"
            self._observe(proxy, account_id, outcome, latency,
                          rate_limited=resp.status_code == 429)
            log.info("proxy_attempt_ok" if outcome == "success" else "proxy_attempt_final",
                     proxy=proxy.name, account=account_id, method=method, url=upstream_url,
                     latency_s=round(latency, 3), status=resp.status_code,
                     attempts=index + 1)
            return _stream_upstream(resp)

        if last_exc is not None:
            raise last_exc
        raise GatewayError(
            f"No account could serve proxy '{proxy.name}'.",
            code="provider_error", status_code=502, provider=proxy.name,
        )

    @staticmethod
    def _observe(proxy: ProxyConfig, account_id: str, outcome: str,
                 latency: float, *, rate_limited: bool) -> None:
        SELECTION_STORE.observe(
            backend=proxy.name, account=account_id, model="*", modality="proxy",
            outcome=outcome, latency_s=latency, rate_limited=rate_limited,
        )


def _join_url(base_url: str, path: str) -> str:
    """Join a proxy root and the captured request path into an upstream URL."""
    root = base_url.rstrip("/")
    tail = path.lstrip("/")
    return f"{root}/{tail}" if tail else root


def _stream_upstream(resp: httpx.Response) -> StreamingResponse:
    """Stream an upstream response back to the client with its headers/status.

    Hop-by-hop response headers are dropped (they describe the gateway↔upstream
    hop, not the gateway↔client hop); everything else (``content-type``,
    ``content-encoding``, ``cache-control``, SSE ``x-accel-buffering``…) is
    forwarded. ``resp`` is closed once the stream completes so the connection
    returns to the pool.
    """
    headers: dict[str, str] = {}
    for name, value in resp.headers.multi_items():
        if name.lower() in _HOP_BY_HOP:
            continue
        headers[name] = value

    async def body_iter():
        try:
            async for chunk in resp.aiter_raw():
                if chunk:
                    yield chunk
        finally:
            await resp.aclose()

    # Marked so the app's response-logging middleware skips buffering a body
    # that may be a long-lived event stream (it still flows through to the
    # client unchanged). The middleware reads this attribute.
    response = StreamingResponse(
        body_iter(), status_code=resp.status_code, headers=headers,
    )
    setattr(response, "_mm_proxy_stream", True)
    return response


# -- WebSocket bridge ------------------------------------------------------- #


def _ws_url(base_url: str, path: str) -> str:
    """Rewrite an http(s) proxy root to ws(s) and append the captured path."""
    root = base_url.rstrip("/")
    if root.startswith("https://"):
        scheme, rest = "wss", root[len("https://"):]
    elif root.startswith("http://"):
        scheme, rest = "ws", root[len("http://"):]
    elif root.startswith("wss://"):
        scheme, rest = "wss", root[len("wss://"):]
    elif root.startswith("ws://"):
        scheme, rest = "ws", root[len("ws://"):]
    else:
        scheme, rest = "ws", root
    tail = path.lstrip("/")
    host = rest
    return f"{scheme}://{host}/{tail}" if tail else f"{scheme}://{host}"


async def bridge_websocket(
    proxy: ProxyConfig,
    path: str,
    request_query: str,
    request_headers,
    client_ws,
) -> None:
    """Bridge a client WebSocket to an upstream ws/wss connection.

    Tries each ranked account on the upstream handshake (a connect failure or
    a 429/401 during the handshake is retried on the next account); once an
    upstream connection is established, both directions are pumped until either
    side closes. The client's query string and Sec-WebSocket-Protocol are
    forwarded; the account credential is injected as a header on the upstream
    handshake and never reaches the client.

    The route passes the raw query string and client headers directly rather
    than a ``Request`` wrapper: a WebSocket scope is not an HTTP scope, so the
    Starlette ``Request(websocket.scope, ...)`` constructor asserts and raises.
    Headers are user-supplied lookups; the workspace-sensitive hop headers are
    filtered by :func:`_forward_headers_drop_auth`.
    """
    from websockets.asyncio.client import connect  # local import: optional dep
    from websockets.exceptions import InvalidStatus

    query = request_query
    client_subprotocol = (request_headers.get("sec-websocket-protocol")
                          if hasattr(request_headers, "get")
                          else None)
    subprotocols = [p.strip() for p in client_subprotocol.split(",")] if client_subprotocol else []
    accounts = _ranked_accounts(proxy)
    if not accounts:
        await _ws_reject(client_ws, 4503, "proxy has no configured account")
        return

    upstream = None
    chosen_account: str | None = None
    for index, (account_id, api_key, base_url, account_headers, _score) in enumerate(accounts):
        upstream_url = _ws_url(base_url or proxy.base_url, path)
        if query:
            upstream_url = f"{upstream_url}?{query}"
        headers = _forward_headers_drop_auth(request_headers, proxy, account_id,
                                             api_key, account_headers)
        start = time.monotonic()
        try:
            upstream = await connect(
                upstream_url,
                additional_headers=headers,
                subprotocols=subprotocols or None,
                open_timeout=proxy.timeout,
            )
        except Exception as exc:  # noqa: BLE001
            latency = time.monotonic() - start
            # websockets raises InvalidStatus when the server replies with an
            # HTTP status (e.g. 401/429) instead of completing the upgrade.
            retryable = True
            status = _ws_handshake_status(exc)
            if status is not None and status < 500 and status not in (429, 401, 403):
                retryable = False
            ProxyRunner._observe(proxy, account_id, "failure", latency,
                                 rate_limited=status == 429)
            log.info("proxy_ws_connect_failed", proxy=proxy.name, account=account_id,
                     url=upstream_url, latency_s=round(latency, 3),
                     status=status, error=str(exc), retryable=retryable)
            if not retryable and index == 0:
                # Hard client error from the very first account — surface it
                # rather than burning the pool. The upstream HTTP status rides
                # in the close reason; the close code itself is a valid 4xxx
                # code (a raw 404/401/429 would be an invalid WebSocket close
                # code per RFC 6455 §7.4).
                await _ws_reject(client_ws, 4502,
                                 f"upstream refused: {status or 'error'}")
                return
            continue
        latency = time.monotonic() - start
        ProxyRunner._observe(proxy, account_id, "success", latency, rate_limited=False)
        chosen_account = account_id
        log.info("proxy_ws_connect_ok", proxy=proxy.name, account=account_id,
                 url=upstream_url, latency_s=round(latency, 3),
                 subprotocol=getattr(upstream, "subprotocol", None))
        break

    if upstream is None:
        await _ws_reject(client_ws, 4502, "all upstream accounts failed to connect")
        return

    await client_ws.accept(
        subprotocol=_negotiated_subprotocol(upstream, subprotocols)
    )
    log.info("proxy_ws_bridge_open", proxy=proxy.name, account=chosen_account)
    try:
        await _pump_ws(client_ws, upstream)
    finally:
        try:
            await upstream.close()
        except Exception:  # noqa: BLE001
            pass
        log.info("proxy_ws_bridge_closed", proxy=proxy.name, account=chosen_account)


def _forward_headers_drop_auth(client_headers, proxy: ProxyConfig, account_id: str,
                              api_key: str | None, account_headers: dict[str, str]) -> dict[str, str]:
    """Forward client headers minus credentials, then inject the account key."""
    out: dict[str, str] = {}
    if hasattr(client_headers, "items"):
        items = list(client_headers.items())
    else:
        items = list(client_headers)
    for name, value in items:
        if name.lower() in _DROP_CLIENT_HEADERS:
            continue
        # Sec-WebSocket-* is negotiated by the websockets client itself from
        # the subprotocols arg; relaying the client's copy would confuse it.
        if name.lower().startswith("sec-websocket-"):
            continue
        out[name] = value
    out.update(proxy.headers)
    out.update(account_headers)
    out.update(proxy_auth_header(proxy, api_key))
    out["x-mm-gateway-proxy-account"] = account_id
    return out


def _ws_handshake_status(exc: BaseException) -> int | None:
    """Extract the HTTP status a websocket handshake was rejected with, if any."""
    # ``InvalidStatus`` carries a ``.response.status`` (websockets >= 12).
    if isinstance(exc, OSError):  # connection refused / DNS — treat as 502
        return None
    status = getattr(exc, "response", None)
    if status is not None:
        got = getattr(status, "status_code", None)
        if got is not None:
            return got
        return getattr(status, "status", None)
    return None


def _negotiated_subprotocol(upstream, requested: list[str]) -> str | None:
    chosen = getattr(upstream, "subprotocol", None)
    if chosen:
        return chosen
    return requested[0] if requested else None


async def _ws_reject(client_ws, status: int, reason: str) -> None:
    """Reject a client WebSocket before it is accepted."""
    try:
        await client_ws.close(code=status, reason=reason)
    except Exception:  # noqa: BLE001
        # Some Starlette versions require accept before close on a never-opened
        # socket; close() still terminates the upgrade so the client sees it.
        pass


async def _pump_ws(client_ws, upstream) -> None:
    """Pump messages both directions until either side closes."""
    import asyncio

    async def client_to_upstream():
        # Starlette's WebSocket exposes no ``iter_data``: ``iter_text`` and
        # ``iter_bytes`` each block on one frame type, so neither alone can
        # drain a connection carrying mixed text/bytes frames. ``receive()``
        # returns the raw ASGI message carrying whichever the client sent, so
        # forward both via ``upstream.send`` (which accepts str or bytes).
        try:
            while True:
                msg = await client_ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("text") is not None:
                    await upstream.send(msg["text"])
                elif msg.get("bytes") is not None:
                    await upstream.send(msg["bytes"])
        except Exception:  # noqa: BLE001
            pass

    async def upstream_to_client():
        try:
            async for msg in upstream:
                if isinstance(msg, str):
                    await client_ws.send_text(msg)
                else:
                    await client_ws.send_bytes(msg)
        except Exception:  # noqa: BLE001
            pass

    tasks = [asyncio.create_task(client_to_upstream()),
             asyncio.create_task(upstream_to_client())]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for t in pending:
        t.cancel()


__all__ = [
    "ProxyRunner",
    "bridge_websocket",
    "proxy_auth_header",
]
