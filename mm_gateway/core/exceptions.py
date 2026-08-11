"""Exception hierarchy for mm-gateway.

A single base class lets the HTTP layer translate every provider error into a
consistent JSON error envelope without ``except`` ladders for each provider.
"""

from __future__ import annotations

from typing import Any

_PUBLIC_ERROR_CODES = {
    "provider_not_configured": "generation_service_unavailable",
    "provider_not_found": "generation_service_not_found",
    "provider_error": "generation_service_error",
    "provider_timeout": "generation_service_timeout",
    "config_error": "internal_error",
}

_PUBLIC_ERROR_MESSAGES = {
    "generation_service_unavailable": "No generation service is available for this request.",
    "generation_service_not_found": "The requested generation service was not found.",
    "generation_service_error": "The generation service returned an error.",
    "generation_service_timeout": "The generation service timed out.",
    "internal_error": "The gateway could not complete the request.",
    "forbidden": "The API key is not allowed to perform this request.",
}


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

    def to_public_dict(self) -> dict[str, Any]:
        """Return the provider-neutral error representation exposed to clients."""
        payload: dict[str, Any] = {
            "error": {
                "code": self.public_code,
                "message": self.public_message,
            }
        }
        # Upstream details can contain provider wire payloads or credentials.
        # Structured client-error details remain useful and safe to expose.
        if self.details and self.status_code < 500:
            payload["error"]["details"] = self.details
        return payload

    @property
    def public_code(self) -> str:
        return _PUBLIC_ERROR_CODES.get(self.code, self.code)

    @property
    def public_message(self) -> str:
        return _PUBLIC_ERROR_MESSAGES.get(self.public_code, self.message)


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


class ConflictError(GatewayError):
    status_code = 409
    code = "conflict"
