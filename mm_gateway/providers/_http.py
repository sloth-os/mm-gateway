"""Shared async HTTP helpers for REST-based provider adapters.

Most providers either have a sync-only SDK (runapi-flux-2, dashscope sync) or no
ergonomic video method (Volcengine Seedance). Rather than depend on each SDK's
event-loop quirks, those adapters use plain httpx here. A shared client factory
keeps TLS/connection pooling and timeout behaviour identical across providers.
"""

from __future__ import annotations

import httpx

from mm_gateway.core.exceptions import ProviderRequestError, ProviderTimeoutError
from mm_gateway.observability.httplog import backend_event_hooks


def make_client(base_url: str, *, timeout: float, headers: dict[str, str] | None = None) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=base_url, timeout=timeout, headers=headers or {},
        event_hooks=backend_event_hooks(),
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
