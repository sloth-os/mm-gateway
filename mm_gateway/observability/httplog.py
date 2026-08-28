"""HTTP request/response logging with curl-format request dumps and masked
sensitive header values.

Two surfaces use this:

* The FastAPI middleware in ``server/app`` logs every *inbound* request (curl
  format, masked headers) and the *outbound* response body / status.
* The httpx event hooks attached to every provider HTTP client log the
  *backend* request (curl format, masked headers, body) and the *backend*
  response (status, masked headers, body).

"Header key need mask" means the *value* of any sensitive header is replaced
with a masked form so secrets never reach the log stream; the header *names*
are kept so the curl dump is still useful for replay/debugging.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

import httpx

from mm_gateway.observability.logging import get_logger

log = get_logger("http")

# Header names whose values carry secrets. Compared case-insensitively. A name
# is sensitive if it is one of these, or contains a telltale substring.
_SENSITIVE_HEADERS = {
    "authorization",
    "proxy-authorization",
    "api-key",
    "apikey",
    "x-api-key",
    "cookie",
    "set-cookie",
    "x-auth-token",
    "x-request-id",  # not secret, but excluded implicitly — kept verbatim
}
_SENSITIVE_SUBSTRINGS = ("key", "token", "secret", "auth", "password", "passwd", "credential")

# Bodies larger than this (bytes) are truncated in the log to avoid dumping
# inline base64 media payloads wholesale.
_MAX_BODY_LOG = 8192

# curl-format request log in curl format — only this one is rendered as a curl
# command; response logs use the structured form.


def _is_sensitive(name: str) -> bool:
    low = name.lower()
    if low in _SENSITIVE_HEADERS:
        return True
    return any(s in low for s in _SENSITIVE_SUBSTRINGS)


def mask_authorization(value: str) -> str:
    """Mask a sensitive header value, keeping a short prefix/suffix hint.

    ``Bearer sk-abc...wxyz`` style values keep the scheme and a few characters
    on each end so an operator can confirm *which* credential was used without
    the credential itself leaking.
    """
    if not value:
        return ""
    scheme, sep, rest = value.partition(" ")
    if sep:
        return f"{scheme} {_mask_token(rest)}"
    return _mask_token(value)


def _mask_token(value: str) -> str:
    if len(value) <= 8:
        return "****"
    return f"{value[:4]}...{value[-4:]}"


def mask_headers(headers: Mapping[str, str] | Iterable[tuple[str, str]]) -> dict[str, str]:
    """Return a header mapping with sensitive values masked."""
    out: dict[str, str] = {}
    items = headers.items() if isinstance(headers, Mapping) else headers
    for name, value in items:
        if _is_sensitive(name):
            out[name] = mask_authorization(value)
        else:
            out[name] = value
    return out


def _truncate(body: str) -> str:
    if len(body) > _MAX_BODY_LOG:
        return body[:_MAX_BODY_LOG] + f"...<{len(body)} bytes truncated>"
    return body


def _decode_body(content: bytes | str | None, content_type: str | None) -> str:
    """Render a body as text for logging, best-effort JSON pretty-print."""
    if content is None:
        return ""
    if isinstance(content, str):
        raw = content
    else:
        try:
            raw = content.decode("utf-8")
        except UnicodeDecodeError:
            return f"<{len(content)} bytes binary>"
    ct = (content_type or "").lower()
    if "json" in ct or (raw and raw.lstrip()[:1] in "{["):
        try:
            return _truncate(json.dumps(json.loads(raw), ensure_ascii=False))
        except (ValueError, TypeError):
            pass
    return _truncate(raw)


def to_curl(method: str, url: str, headers: Mapping[str, str] | Iterable[tuple[str, str]],
            content: bytes | str | None) -> str:
    """Render an HTTP request as a ``curl`` command with masked sensitive headers."""
    masked = mask_headers(headers)
    parts = ["curl", "-X", method, shell_quote(url)]
    for name, value in masked.items():
        parts.extend(["-H", shell_quote(f"{name}: {value}")])
    if content:
        if isinstance(content, bytes):
            try:
                body = content.decode("utf-8")
            except UnicodeDecodeError:
                body = repr(content)
        else:
            body = content
        if body:
            parts.extend(["--data", shell_quote(_truncate(body))])
    return " ".join(parts)


def shell_quote(s: str) -> str:
    """Single-quote a string for shell safety (curl -H / --data values)."""
    return "'" + s.replace("'", "'\"'\"'") + "'"


# -- Backend (httpx) hooks -------------------------------------------------- #

async def _backend_request_hook(request: httpx.Request) -> None:
    # The body stream may already be consumed by httpx; aread() materialises it
    # and (httpx re-uses the same bytes) keeps the request sendable.
    content: bytes | None = None
    if request.content is not None:
        content = request.content
    curl = to_curl(request.method, str(request.url), request.headers, content)
    body_text = _decode_body(content, request.headers.get("content-type"))
    log.info("backend_request", curl=curl, method=request.method, url=str(request.url),
             headers=mask_headers(request.headers), body=body_text)


async def _backend_response_hook(response: httpx.Response) -> None:
    # aread() caches the body so the provider can still read it afterwards.
    try:
        await response.aread()
    except Exception:  # noqa: BLE001
        pass
    content = response.content
    body_text = _decode_body(content, response.headers.get("content-type"))
    log.info("backend_response", status=response.status_code, url=str(response.request.url),
             headers=mask_headers(response.headers), body=body_text)


def backend_event_hooks(*, log_response: bool = True) -> dict[str, list]:
    """Event hooks to attach to an ``httpx.AsyncClient`` for backend logging.

    The response hook (:func:`_backend_response_hook`) materialises the response
    body via ``aread()`` so it can be logged. That *consumes* the response byte
    stream, which is fine for providers that then read ``response.content`` —
    but a pass-through proxy streams the upstream response back to the client via
    ``aiter_raw()`` and cannot tolerate the stream being consumed first. Such a
    client should pass ``log_response=False``: the inbound upstream request is
    still logged (masked credential + account attribution) and the response
    status is logged by the proxy's own ``proxy_attempt_*`` line, so the response
    hook is redundant there as well as incompatible.
    """
    hooks: dict[str, list] = {"request": [_backend_request_hook]}
    if log_response:
        hooks["response"] = [_backend_response_hook]
    return hooks


# Sync variants for providers whose SDK uses a synchronous ``httpx.Client``
# (runapi-flux-2). Same shape as the async hooks, but ``response.read()``
# instead of ``await response.aread()``.


def _backend_request_hook_sync(request: httpx.Request) -> None:
    content: bytes | None = None
    if request.content is not None:
        content = request.content
    curl = to_curl(request.method, str(request.url), request.headers, content)
    body_text = _decode_body(content, request.headers.get("content-type"))
    log.info("backend_request", curl=curl, method=request.method, url=str(request.url),
             headers=mask_headers(request.headers), body=body_text)


def _backend_response_hook_sync(response: httpx.Response) -> None:
    try:
        response.read()
    except Exception:  # noqa: BLE001
        pass
    content = response.content
    body_text = _decode_body(content, response.headers.get("content-type"))
    log.info("backend_response", status=response.status_code, url=str(response.request.url),
             headers=mask_headers(response.headers), body=body_text)


def backend_sync_event_hooks() -> dict[str, list]:
    """Event hooks to attach to a sync ``httpx.Client`` for backend logging."""
    return {"request": [_backend_request_hook_sync], "response": [_backend_response_hook_sync]}


# -- Backend (manual / SDK-backed) ------------------------------------------- #
# Providers whose SDK manages its own transport (DashScope uses aiohttp) can't
# attach httpx event hooks, so they call these directly with the request they
# are about to issue and the response they got back.


def log_backend_request(method: str, url: str, headers, content: bytes | str | None) -> None:
    """Log a backend request as a curl command (masked headers) + body."""
    curl = to_curl(method, url, headers, content)
    body_text = _decode_body(content, _content_type(headers))
    log.info("backend_request", curl=curl, method=method, url=url,
             headers=mask_headers(headers), body=body_text)


def log_backend_response(status: Any, url: str, headers, content: bytes | str | None) -> None:
    """Log a backend response: status, masked headers, body."""
    body_text = _decode_body(content, _content_type(headers))
    log.info("backend_response", status=status, url=url, headers=mask_headers(headers), body=body_text)


def _content_type(headers) -> str | None:
    return headers.get("content-type") if isinstance(headers, Mapping) else None


# -- Frontend (FastAPI) helpers --------------------------------------------- #

def frontend_request_log(method: str, url: str, headers, body_bytes: bytes | None) -> None:
    curl = to_curl(method, url, headers, body_bytes)
    body_text = _decode_body(body_bytes, headers.get("content-type") if isinstance(headers, Mapping) else None)
    log.info("frontend_request", curl=curl, method=method, url=url, body=body_text)


def frontend_response_log(status: int, headers, body_bytes: bytes | None) -> None:
    ct = headers.get("content-type") if isinstance(headers, Mapping) else None
    body_text = _decode_body(body_bytes, ct)
    log.info("frontend_response", status=status, headers=mask_headers(headers), body=body_text)
