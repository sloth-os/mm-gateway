"""Shared async HTTP helpers for REST-based provider adapters.

Most providers either have a sync-only SDK (runapi-flux-2, dashscope sync) or no
ergonomic video method (Volcengine Seedance). Rather than depend on each SDK's
event-loop quirks, those adapters use plain httpx here. A shared client factory
keeps TLS/connection pooling and timeout behaviour identical across providers.
"""

from __future__ import annotations

import httpx

from mm_gateway.config import _is_socks_proxy
from mm_gateway.core.exceptions import ProviderRequestError, ProviderTimeoutError
from mm_gateway.observability.httplog import backend_event_hooks


def proxy_kwargs(proxy_url: str | None) -> dict:
    """Build httpx client kwargs that route traffic through ``proxy_url``.

    httpx accepts an explicit ``proxy=<url>`` for both HTTP CONNECT and SOCKS
    proxies (SOCKS needs the optional ``socksio`` extra). An explicit proxy
    **wins** over the ambient ``HTTP_PROXY`` / ``HTTPS_PROXY`` env vars httpx
    otherwise honours via ``trust_env``. When ``proxy_url`` is ``None`` we set
    nothing, so the client keeps its default ``trust_env=True`` and an operator
    can still steer traffic through env vars alone.

    The resolved URL lives in ``backend.extra["outbound_proxy"]`` (the registry
    folds the global ``Settings.outbound_proxy`` in when no per-backend override
    is set), so callers pass that value straight through.
    """
    if not proxy_url:
        return {}
    if _is_socks_proxy(proxy_url):
        try:
            import socksio  # noqa: F401  — registering the httpx SOCKS extension
        except ImportError as exc:  # pragma: no cover - config-time guard
            raise ProviderRequestError(
                f"SOCKS outbound proxy {proxy_url!r} requires the 'socksio' "
                "package (install httpx[socks]); HTTP proxies need no extra dep."
            ) from exc
    return {"proxy": proxy_url}


def make_client(base_url: str, *, timeout: float,
                headers: dict[str, str] | None = None,
                proxy_url: str | None = None) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=base_url, timeout=timeout, headers=headers or {},
        event_hooks=backend_event_hooks(), **proxy_kwargs(proxy_url),
    )


async def request_json(client: httpx.AsyncClient, method: str, url: str, *,
                       provider: str, **kwargs) -> dict:
    """Issue a request and return JSON, mapping transport errors to gateway errors."""
    try:
        resp = await client.request(method, url, **kwargs)
    except httpx.TimeoutException as exc:
        raise ProviderTimeoutError(f"{provider} request timed out: {exc}", provider=provider) from exc
    except httpx.HTTPError as exc:
        raise ProviderRequestError(f"{provider} transport error: {exc}", provider=provider) from exc

    if resp.status_code >= 400:
        body = resp.text
        raise ProviderRequestError(
            f"{provider} returned HTTP {resp.status_code}",
            provider=provider,
            status_code=_map_status(resp.status_code),
            details={"upstream_status": resp.status_code, "upstream_body": body[:1000]},
        )
    return resp.json()


def _map_status(upstream: int) -> int:
    # Surfaced as the gateway status code so clients see provider auth/quota issues.
    if upstream == 401 or upstream == 403:
        return 502  # upstream auth problem — our config issue, not the client's
    if upstream == 429:
        return 429
    if upstream >= 500:
        return 502
    return 502
