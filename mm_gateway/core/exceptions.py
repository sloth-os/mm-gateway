"""Exception hierarchy for mm-gateway.

A single base class lets the HTTP layer translate every provider error into a
consistent JSON error envelope without ``except`` ladders for each provider.
"""

from __future__ import annotations

from typing import Any


class GatewayError(Exception):
    """Base class for all gateway errors.

    ``status_code`` is the HTTP status the server layer should emit.
    ``code`` is a stable machine-readable string surfaced to API clients.
    """

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, *, code: str | None = None, status_code: int | None = None,
                 provider: str | None = None, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.details = details or {}
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "error": {
                "code": self.code,
                "message": self.message,
            }
        }
        if self.provider:
            payload["error"]["provider"] = self.provider
        if self.details:
            payload["error"]["details"] = self.details
        return payload


class ConfigError(GatewayError):
    status_code = 500
    code = "config_error"


class ProviderNotConfiguredError(GatewayError):
    status_code = 503
    code = "provider_not_configured"

    def __init__(self, provider: str, message: str | None = None):
        super().__init__(
            message or f"Provider '{provider}' is not configured (missing API key/credentials).",
            provider=provider,
        )


class ProviderNotFoundError(GatewayError):
    status_code = 404
    code = "provider_not_found"


class ModelNotFoundError(GatewayError):
    status_code = 404
    code = "model_not_found"


class ValidationError(GatewayError):
    status_code = 400
    code = "invalid_request_error"


class UnsupportedFeatureError(GatewayError):
    """A provider does not support the requested capability (e.g. video on a provider)."""
    status_code = 400
    code = "unsupported_feature"


class ProviderRequestError(GatewayError):
    """A provider returned a non-2xx / raised an error while fulfilling the request."""

    status_code = 502
    code = "provider_error"


class ProviderTimeoutError(GatewayError):
    status_code = 504
    code = "provider_timeout"


class TaskNotFoundError(GatewayError):
    status_code = 404
    code = "task_not_found"


class TaskFailedError(GatewayError):
    status_code = 422
    code = "task_failed"


class UnauthorizedError(GatewayError):
    """No or unknown API key presented on a request that requires auth."""

    status_code = 401
    code = "unauthorized"


class ForbiddenError(GatewayError):
    """The authenticated key is not allowed to use the requested backend/model."""

    status_code = 403
    code = "forbidden"
