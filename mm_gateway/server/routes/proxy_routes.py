"""General pass-through proxy routes.

Mounts a single catch-all path ``/proxy/{domain}/{path:path}`` that forwards
any HTTP method to an upstream service (:class:`ProxyConfig`) and, on a
WebSocket upgrade, bridges the connection to an upstream ws/wss endpoint. The
front-end caller authenticates with the usual bearer key; the same hybrid tag
authorization governs which proxies a key may use. The upstream account's
credential is injected by the gateway and never reaches the client.

A proxy is identified by its upstream **domain** (the host of its configured
``base_url``): the request's ``domain`` path segment selects it. A WebSocket
upgrade is auto-detected from the client's ``Upgrade: websocket`` header on
the same path (Starlette routes an upgrade to the ``websocket`` operation, a
plain request to the HTTP one) — there is no per-proxy ``websocket`` toggle.

See :mod:`mm_gateway.proxy` for the forwarding/selection/retry mechanics.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Path, Request, WebSocket
from fastapi.responses import Response

from mm_gateway.config import KeyConfig
from mm_gateway.core.exceptions import ForbiddenError, UnauthorizedError
from mm_gateway.proxy import ProxyRunner, bridge_websocket
from mm_gateway.server.auth import get_api_key, resolve_key

router = APIRouter(tags=["proxy"])

# Every standard HTTP method is forwarded verbatim. CONNECT is intentionally
# excluded — it tunnels a raw TCP stream and is not an HTTP proxy concern.
_HTTP_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]

_PROXY_RESPONSES = {
    200: {"description": "The upstream response, streamed back verbatim (any media type)."},
    401: {"description": "Missing or unknown API key."},
    403: {"description": "Key not allowed to use this proxy."},
    404: {"description": "No proxy configured for this domain."},
    502: {"description": "Every upstream account failed."},
    503: {"description": "The proxy has no configured account."},
}


def _resolve_proxy(request: Request, domain: str, key: KeyConfig) -> "object":
    registry = request.app.state.registry
    if domain not in registry.usable_proxies(key):
        # Distinguish "no such proxy" (404) from "not allowed" (403): the proxy
        # may be configured but out of the key's tag set, which is a permissions
        # gap, while an unknown domain is a client typo.
        if domain in registry.proxy_names():
            raise ForbiddenError(f"API key '{key.id}' is not allowed to use proxy '{domain}'.")
        from mm_gateway.core.exceptions import ProviderNotFoundError
        raise ProviderNotFoundError(f"Proxy '{domain}' is not configured.")
    return registry.proxy(domain)


async def proxy_request(
    request: Request,
    domain: Annotated[str, Path(description="Configured proxy domain (the upstream host segment selecting the proxy).")],
    path: Annotated[str, Path(description="Path forwarded to the upstream root URL.")],
    key: Annotated[KeyConfig, Depends(get_api_key)],
) -> Response:
    """Forward an HTTP request through a domain-matched proxy to its upstream.

    The path, query string, body, and most client headers are forwarded
    verbatim to ``{base_url}/{path}``; the configured account's credential is
    injected from the account's ``headers`` and the upstream response (including
    event streams) is streamed back. Retries across accounts on a rate-limit /
    timeout / 5xx.
    """
    proxy = _resolve_proxy(request, domain, key)
    runner: ProxyRunner = request.app.state.proxy_runner
    # Signal the response-logging middleware: this response may be a long-lived
    # event stream, so it must flow through without buffering.
    request.state.proxy_streaming = True
    return await runner.forward(proxy, request.method, path, request)


# Register one path operation per HTTP method so each carries a unique OpenAPI
# operationId (FastAPI collapses a multi-method ``api_route`` onto one id, which
# violates the spec's uniqueness requirement). They share the handler above.
for _method in _HTTP_METHODS:
    router.api_route(
        "/proxy/{domain}/{path:path}",
        methods=[_method],
        operation_id=f"proxyRequest{_method.title()}",
        summary="Forward a request through a domain-matched proxy",
        response_class=Response,
        responses=_PROXY_RESPONSES,
    )(proxy_request)


@router.websocket("/proxy/{domain}/{path:path}")
async def proxy_websocket(
    websocket: WebSocket,
    domain: Annotated[str, Path(description="Configured proxy domain (the upstream host segment).")],
    path: Annotated[str, Path(description="Path forwarded to the upstream root URL.")],
) -> None:
    """Bridge a WebSocket upgrade on the same path to an upstream ws/wss URL.

    The upgrade is auto-detected from the client's ``Upgrade: websocket``
    header. The front-end caller authenticates with the usual bearer key, read
    off the upgrade handshake's ``Authorization`` header (or an
    ``access_token`` / ``token`` query param, since browsers cannot set headers
    on a WS upgrade).
    """
    app = websocket.app
    settings = app.state.settings
    token = _ws_token(websocket)
    try:
        key = resolve_key(settings, token)
    except UnauthorizedError:
        await websocket.close(code=4401, reason="unauthorized")
        return
    registry = app.state.registry
    if domain not in registry.usable_proxies(key):
        await websocket.close(code=4403, reason="forbidden" if domain in registry.proxy_names()
                              else "not_configured")
        return
    proxy = registry.proxy(domain)
    # The WebSocket exposes its own headers/query_params, so pass them straight
    # to the bridge. Wrapping a websocket scope in a Starlette ``Request`` is
    # not possible: its constructor asserts ``scope["type"] == "http"``.
    await bridge_websocket(
        proxy, path, websocket.url.query, websocket.headers, websocket,
    )


def _ws_token(websocket: WebSocket) -> str | None:
    header = websocket.headers.get("authorization") or websocket.headers.get("Authorization")
    if header:
        scheme, _, token = header.partition(" ")
        if scheme.lower() == "bearer" and token:
            return token.strip()
    # Browsers cannot set headers on a WebSocket upgrade; accept a query token
    # so a browser client can still authenticate. The value never reaches the
    # upstream (it is the front-end key, dropped before forwarding).
    return (websocket.query_params.get("access_token")
            or websocket.query_params.get("token"))


__all__ = ["router"]
