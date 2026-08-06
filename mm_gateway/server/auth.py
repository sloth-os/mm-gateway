"""API key authentication.

A request to any generation or model-listing endpoint must carry an
``Authorization: Bearer <token>`` header whose token matches a configured
``KeyConfig``. Unknown or absent tokens raise ``UnauthorizedError`` (401). The
``X-Backend-Tag`` header, and the ``provider.tag``/``provider.backend`` fields
of a JSON request body, are read here so the routes can pass them down to the
service for routing.

The token-to-``KeyConfig`` lookup lives in ``resolve_key`` so it can be shared
by both the HTTP dependency (``get_api_key``) and the MCP tools, which read the
``Authorization`` header off the MCP ``Context`` rather than a FastAPI
``Request``.
"""

from __future__ import annotations

from typing import Any

from fastapi import Request

from mm_gateway.config import KeyConfig, Settings
from mm_gateway.core.exceptions import ForbiddenError, UnauthorizedError


def _extract_token(request: Request) -> str | None:
    header = request.headers.get("authorization") or request.headers.get("Authorization")
    if not header:
        return None
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token.strip()


def resolve_key(settings: Settings, token: str | None) -> KeyConfig:
    """Resolve a bearer token to a KeyConfig, or raise ``UnauthorizedError``.

    ``token`` may be ``None`` (no header at all) or an unknown string. A
    matched token resolves directly; otherwise an "open" gateway (a key with an
    empty token, e.g. the legacy env key without ``GATEWAY_API_KEY`` set)
    admits any caller. If neither holds, this is an auth failure.
    """
    if token:
        key = settings.key_for(token)
        if key is not None:
            return key
    # No token, or an unknown one: fall back to an open key if configured.
    for k in settings.keys:
        if not k.key:
            return k
    if token:
        raise UnauthorizedError("Unknown API key.")
    raise UnauthorizedError("Missing API key. Send an 'Authorization: Bearer <token>' header.")


def get_api_key(request: Request) -> KeyConfig:
    """FastAPI dependency: resolve the bearer token on the request to a KeyConfig."""
    return resolve_key(request.app.state.settings, _extract_token(request))


def authorize_task(
    request: Request, key: KeyConfig, provider_name: str | None
) -> None:
    """Ensure ``key`` may access a task owned by ``provider_name``.

    Polling a task id never re-runs the create-path routing, so the hybrid
    usable check there is bypassed on read. Without this guard, any
    authenticated key could poll (and read the resulting artefacts of) a task
    owned by a backend it is not authorised to use — a cross-tenant leak. The
    owning backend comes from the task store record written at creation time.
    """
    if not provider_name:
        return  # no record -> let the service surface the not-found error
    if provider_name not in request.app.state.registry.usable_backends(key):
        raise ForbiddenError(
            f"API key '{key.id}' is not allowed to use backend '{provider_name}'."
        )


def routing_overrides(request: Request, body: dict[str, Any] | None = None
                      ) -> tuple[str | None, str | None]:
    """Return ``(tag, backend_name)`` overrides from headers and/or body."""
    tag = request.headers.get("x-backend-tag")
    backend = request.headers.get("x-backend")
    if body and isinstance(body.get("provider"), dict):
        pref = body["provider"]
        if tag is None and pref.get("tag"):
            tag = pref["tag"]
        if backend is None and pref.get("backend"):
            backend = pref["backend"]
    return tag, backend
