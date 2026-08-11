"""API key authentication.

A request to any generation or model-listing endpoint must carry an
``Authorization: Bearer <token>`` header whose token matches a configured
``KeyConfig``. Unknown or absent tokens raise ``UnauthorizedError`` (401).

The token-to-``KeyConfig`` lookup lives in ``resolve_key`` so it can be shared
by both the HTTP dependency (``get_api_key``) and the MCP tools, which read the
``Authorization`` header off the MCP ``Context`` rather than a FastAPI
``Request``.
"""

from __future__ import annotations

from fastapi import Request

from mm_gateway.config import KeyConfig, Settings
from mm_gateway.core.exceptions import ForbiddenError, UnauthorizedError
from mm_gateway.tasks.store import TaskRecord


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
    request: Request, key: KeyConfig, record: TaskRecord
) -> None:
    """Ensure ``key`` owns the task and may still use its routed backend.

    Polling a task id never re-runs the create-path routing, so the hybrid
    usable check there is bypassed on read. Without this guard, any
    authenticated key could poll (and read the resulting artefacts of) a task
    owned by a backend it is not authorised to use — a cross-tenant leak. The
    owning backend comes from the task store record written at creation time.
    """
    authorize_task_access(request.app.state.registry, key, record)


def authorize_task_access(registry, key: KeyConfig, record: TaskRecord) -> None:
    """Pure task authorization shared by HTTP routes and MCP tools."""
    if record.owner_key_id != key.id:
        raise ForbiddenError("The API key is not allowed to access this task.")
    if record.provider not in registry.usable_backends(key):
        raise ForbiddenError("The API key is not allowed to access this task.")
