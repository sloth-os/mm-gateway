"""Retry-across-backends for auto-routed requests.

When a request lets the gateway pick the backend (``model=auto``) and the chosen
backend fails in a way that suggests instability — a rate limit, a timeout, a
provider 5xx, or a transport error — the gateway retries the *next* candidate
returned by :meth:`mm_gateway.registry.Registry.enumerate_auto_candidates`,
rather than surfacing the failure to the client. This makes auto mode resilient
to a single flaky backend, and (for multi-account backends) to a single
exhausted key.

Non-retryable failures — a client-side 4xx (bad request, unsupported feature,
forbidden, validation) — propagate immediately: retrying them on another
backend would either repeat the same client error or mask a genuine client
mistake. An explicit-model request (the caller pinned a backend/model) keeps
its single-attempt behaviour: the gateway honours the pin and does not
second-guess it.

Every attempt records its outcome (success/failure, latency, rate-limit) in the
selection store so the next request's ranking reflects what just happened.
"""

from __future__ import annotations

import time
from typing import Any, Awaitable, Callable

from mm_gateway.config import KeyConfig
from mm_gateway.core.exceptions import (
    GatewayError,
    ProviderRequestError,
    ProviderTimeoutError,
)
from mm_gateway.observability.logging import get_logger
from mm_gateway.observability.selection import STORE as SELECTION_STORE

log = get_logger("selection")


def _is_retryable(exc: BaseException) -> bool:
    """True iff a backend failure is worth retrying on another candidate.

    Rate limits (429) and timeouts are always retryable. Provider errors are
    retryable only when the upstream returned a 5xx / 429 / transport error
    (i.e. an unstable backend), *not* a client 4xx — those are surfaced so the
    caller fixes their request. ``GatewayError`` with no upstream status (the
    generic "provider_error" the services wrap unknown exceptions in) is treated
    as retryable, since an unrecognised exception usually means instability.
    """
    if isinstance(exc, ProviderTimeoutError):
        return True
    if isinstance(exc, ProviderRequestError):
        # ``ProviderRequestError.status_code`` is the *mapped* gateway status
        # (see providers._http._map_status): 429 or 502 means upstream-side /
        # rate-limit → retry; a 4xx otherwise means the client's request was
        # rejected upstream, which we do not retry across backends.
        return exc.status_code in (429, 502, 503, 504)
    if isinstance(exc, GatewayError):
        code = getattr(exc, "code", "")
        status = getattr(exc, "status_code", 500)
        # Client-side / config errors are terminal for this request.
        if code in ("invalid_request_error", "unsupported_feature", "forbidden",
                    "unauthorized", "validation_error", "conflict",
                    "model_not_found", "task_not_found", "task_failed"):
            return False
        if 400 <= status < 500:
            return False
        # Everything else (5xx, generic provider_error) is treated as instability.
        return True
    # A bare Exception surfaced by the service (the ``except Exception`` wrap)
    # reads as ``provider_error`` (502) — retryable by default.
    return True


def _is_rate_limited(exc: BaseException) -> bool:
    """True iff the failure was specifically a rate-limit (429)."""
    if isinstance(exc, ProviderRequestError):
        return exc.status_code == 429
    return False


async def retry_across_backends(
    *,
    candidates: list[tuple[Any, str, str, str]],
    attempt: Callable[[Any, str, str, str], Awaitable[Any]],
    modality: str,
    key: KeyConfig | None,
    max_attempts: int | None = None,
) -> Any:
    """Try candidates best-first; on a retryable failure, try the next.

    ``candidates`` is the ordered list from ``enumerate_auto_candidates``:
    ``(provider, account_id, model, backend_name)``. ``attempt`` runs one
    attempt against a candidate and returns the task/result (or raises).
    Each attempt is timed and recorded in the selection store before its
    result/exception is propagated, so the store reflects reality even when an
    attempt raises.

    The last failure is re-raised if every candidate is exhausted, so the
    client sees the most informative error from the last backend tried.
    """
    if not candidates:
        raise GatewayError(
            "No candidate backend is available to serve this auto-routed request.",
            code="validation_error", status_code=422,
        )
    # Cap attempts at the candidate count unless an explicit ceiling is given.
    limit = max_attempts if max_attempts is not None else len(candidates)
    last_exc: BaseException | None = None
    for index, (provider, account_id, model, backend_name) in enumerate(candidates):
        if index >= limit:
            break
        start = time.monotonic()
        try:
            result = await attempt(provider, account_id, model, backend_name)
        except Exception as exc:  # noqa: BLE001
            latency = time.monotonic() - start
            retryable = _is_retryable(exc)
            rate_limited = _is_rate_limited(exc)
            SELECTION_STORE.observe(
                backend=backend_name, account=account_id, model=model,
                modality=modality, outcome="failure", latency_s=latency,
                rate_limited=rate_limited,
            )
            log.info("auto_route_attempt_failed", backend=backend_name,
                     account=account_id, model=model, modality=modality,
                     latency_s=round(latency, 3), rate_limited=rate_limited,
                     retryable=retryable, error=str(exc))
            last_exc = exc
            if not retryable:
                # Terminal (client) error — don't waste other backends on it.
                raise
            continue
        latency = time.monotonic() - start
        SELECTION_STORE.observe(
            backend=backend_name, account=account_id, model=model,
            modality=modality, outcome="success", latency_s=latency,
        )
        log.info("auto_route_attempt_ok", backend=backend_name, account=account_id,
                 model=model, modality=modality, latency_s=round(latency, 3),
                 attempts=index + 1)
        return result
    # Exhausted every candidate.
    if last_exc is not None:
        raise last_exc
    raise GatewayError(
        "No candidate backend could serve this auto-routed request.",
        code="provider_error", status_code=502,
    )


__all__ = ["retry_across_backends"]
