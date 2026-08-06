"""Configuration for mm-gateway.

All configuration is environment-driven so the gateway runs the same way in dev,
container, and serverless. Settings are loaded once into a frozen ``Settings``
object and injected via FastAPI dependency injection.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


def _env(name: str, default: str | None = None) -> str | None:
    val = os.environ.get(name)
    return val if val not in (None, "") else default


@dataclass(frozen=True)
class ProviderCredentials:
    """Credentials and endpoint overrides for one provider.

    Every provider is optional; the registry only registers providers that have
    an ``api_key`` set. This lets the gateway degrade gracefully when only some
    providers are configured.
    """

    name: str
    api_key: str | None = None
    base_url: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


@dataclass(frozen=True)
class Settings:
    host: str = field(default_factory=lambda: _env("HOST", "0.0.0.0") or "0.0.0.0")
    port: int = field(default_factory=lambda: int(_env("PORT", "8000") or "8000"))
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO") or "INFO")
    log_format: str = field(default_factory=lambda: _env("LOG_FORMAT", "json") or "json")
    request_timeout: float = field(default_factory=lambda: float(_env("REQUEST_TIMEOUT", "120") or "120"))
    # When True, video endpoints block until the task completes (sync-style) up to
    # max_sync_wait seconds, then fall back to returning a task id.
    video_sync_default: bool = field(default_factory=lambda: _env("VIDEO_SYNC_DEFAULT", "true").lower() == "true")
    max_sync_wait: float = field(default_factory=lambda: float(_env("MAX_SYNC_WAIT", "300") or "300"))
    poll_interval: float = field(default_factory=lambda: float(_env("POLL_INTERVAL", "2.0") or "2.0"))
    # Default routing: which provider backs the gateway's own model aliases.
    default_image_provider: str = field(default_factory=lambda: _env("DEFAULT_IMAGE_PROVIDER", "openai") or "openai")
    default_video_provider: str = field(default_factory=lambda: _env("DEFAULT_VIDEO_PROVIDER", "volcengine") or "volcengine")
    enable_metrics: bool = field(default_factory=lambda: _env("ENABLE_METRICS", "true").lower() == "true")

    providers: dict[str, ProviderCredentials] = field(default_factory=dict)

    def provider(self, name: str) -> ProviderCredentials:
        return self.providers.get(name) or ProviderCredentials(name=name)

    @classmethod
    def from_env(cls) -> "Settings":
        providers: dict[str, ProviderCredentials] = {}

        def add(name: str, key_env: str, url_env: str | None = None, **extra: Any) -> None:
            providers[name] = ProviderCredentials(
                name=name,
                api_key=_env(key_env),
                base_url=_env(url_env) if url_env else None,
                extra=dict(extra),
            )

        add("openai", "OPENAI_API_KEY", "OPENAI_BASE_URL")
        add("google", "GOOGLE_API_KEY", "GOOGLE_BASE_URL")
        add("xai", "XAI_API_KEY", "XAI_BASE_URL")
        add("volcengine", "ARK_API_KEY", "ARK_BASE_URL")
        # runapi-flux-2 reads RUNAPI_API_KEY and defaults to https://runapi.ai
        add("flux", "RUNAPI_API_KEY", "FLUX_BASE_URL")
        add("openrouter", "OPENROUTER_API_KEY", "OPENROUTER_BASE_URL")
        add("dashscope", "DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL")
        add("stability", "STABILITY_API_KEY", "STABILITY_BASE_URL")

        return cls(providers=providers)
