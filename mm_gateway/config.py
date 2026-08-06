"""Configuration for mm-gateway.

Configuration is YAML-driven. The canonical source is a ``mm-gateway.yaml``
file (located via the ``MM_GATEWAY_CONFIG`` env var, or ``./mm-gateway.yaml``
by default); environment-variable interpolation (``${ENV}`` and
``${ENV:default}``) lets secrets stay out of the file. For backward
compatibility, if no YAML file is found the settings fall back to a
legacy env-var layout that registers one backend per provider by its
``<NAME>_API_KEY``.

The model distinguishes **backends** (named, typed provider instances, each
carrying its own credentials and zero or more routing tags) from **keys**
(front-end API keys with allow/deny rules over tags and backends). This lets
the gateway run several instances of the same provider type (e.g. a prod and a
staging Volcengine account) and route a given front-end key to a chosen subset.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _env(name: str, default: str | None = None) -> str | None:
    val = os.environ.get(name)
    return val if val not in (None, "") else default


@dataclass(frozen=True)
class BackendConfig:
    """One named, typed provider instance with credentials and tags.

    ``name`` identifies the instance (it is what a front-end key's
    ``allow_backends``/``default_*_backend`` refers to, and what is recorded on
    task records for poll routing). ``type`` selects the provider adapter
    class (e.g. ``volcengine``, ``openai``). ``tags`` are free-form routing
    labels a key can allow/deny.
    """

    name: str
    type: str
    api_key: str | None = None
    base_url: str | None = None
    tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


@dataclass(frozen=True)
class KeyConfig:
    """A front-end API key with routing rules.

    A request authenticated with this key may use any backend whose tags
    intersect ``allow_tags`` OR whose name is in ``allow_backends`` (the hybrid
    usable rule), and never any backend named in ``deny_tags``. Routing then
    picks among the usable backends by an explicit override, the per-modality
    default tag/backend, or the first usable backend.
    """

    id: str
    key: str
    allow_tags: list[str] = field(default_factory=list)
    deny_tags: list[str] = field(default_factory=list)
    allow_backends: list[str] = field(default_factory=list)
    default_image_tag: str | None = None
    default_video_tag: str | None = None
    default_image_backend: str | None = None
    default_video_backend: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


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
    enable_metrics: bool = field(default_factory=lambda: _env("ENABLE_METRICS", "true").lower() == "true")

    # Whether the HTTP MCP server (POST /<mcp_path>) is mounted, and the path
    # it is served on. The MCP server exposes the gateway's image/video/model
    # capabilities as MCP tools that clients can call over Streamable HTTP.
    mcp_enabled: bool = field(default_factory=lambda: _env("MCP_ENABLED", "false").lower() == "true")
    mcp_path: str = field(default_factory=lambda: _env("MCP_PATH", "/mcp") or "/mcp")
    # Idle-session reaping for the stateful MCP session manager: a client that
    # initialises and then vanishes without sending DELETE would otherwise keep
    # its session (and serve-loop task) alive for the gateway's whole lifetime.
    # The SDK recommends a finite value (default 1800s / 30 min).
    mcp_session_idle_timeout: float = field(
        default_factory=lambda: float(_env("MCP_SESSION_IDLE_TIMEOUT", "1800") or "1800")
    )

    backends: list[BackendConfig] = field(default_factory=list)
    keys: list[KeyConfig] = field(default_factory=list)

    # -- accessors ------------------------------------------------------------ #

    def backend(self, name: str) -> BackendConfig | None:
        for b in self.backends:
            if b.name == name:
                return b
        return None

    def backends_of_type(self, type_: str) -> list[BackendConfig]:
        return [b for b in self.backends if b.type == type_]

    def key_for(self, token: str) -> KeyConfig | None:
        for k in self.keys:
            if k.key == token:
                return k
        return None

    @property
    def default_image_provider(self) -> str:
        """Legacy-compat name for code that still expects a single default."""
        for k in self.keys:
            if k.default_image_backend:
                return k.default_image_backend
        return self.backends[0].name if self.backends else ""

    @property
    def default_video_provider(self) -> str:
        for k in self.keys:
            if k.default_video_backend:
                return k.default_video_backend
        return self.backends[0].name if self.backends else ""

    # -- loaders -------------------------------------------------------------- #

    @classmethod
    def from_file(cls, path: str | Path) -> "Settings":
        text = Path(path).read_text(encoding="utf-8")
        return cls._from_yaml(text)

    @classmethod
    def from_env(cls) -> "Settings":
        """Locate a YAML config file, else fall back to legacy env vars."""
        path = _env("MM_GATEWAY_CONFIG") or _find_config_file()
        if path:
            return cls.from_file(path)
        return cls._from_legacy_env()

    @classmethod
    def _from_yaml(cls, text: str) -> "Settings":
        import yaml  # local import so the dep is optional for env-only use

        raw = yaml.safe_load(_interpolate(text)) or {}
        backends = [
            BackendConfig(
                name=b["name"],
                type=b["type"],
                api_key=b.get("api_key"),
                base_url=b.get("base_url"),
                tags=list(b.get("tags") or []),
                extra=dict(b.get("extra") or {}),
            )
            for b in (raw.get("backends") or [])
        ]
        keys = [
            KeyConfig(
                id=k["id"],
                key=k["key"],
                allow_tags=list(k.get("allow_tags") or []),
                deny_tags=list(k.get("deny_tags") or []),
                allow_backends=list(k.get("allow_backends") or []),
                default_image_tag=k.get("default_image_tag"),
                default_video_tag=k.get("default_video_tag"),
                default_image_backend=k.get("default_image_backend"),
                default_video_backend=k.get("default_video_backend"),
                extra=dict(k.get("extra") or {}),
            )
            for k in (raw.get("keys") or [])
        ]
        def _section(name: str) -> dict:
            # A scalar like `mcp: true` (a plausible "enable mcp" shorthand)
            # would otherwise crash the later `.get(...)` calls with an opaque
            # AttributeError; coerce any non-mapping to an empty section.
            val = raw.get(name)
            return val if isinstance(val, dict) else {}

        server = _section("server")
        video = _section("video")
        defaults = _section("defaults")
        mcp = _section("mcp")
        return cls(
            host=str(server.get("host", _env("HOST", "0.0.0.0") or "0.0.0.0")),
            port=int(server.get("port", _env("PORT", "8000") or "8000")),
            log_level=str(server.get("log_level", _env("LOG_LEVEL", "INFO") or "INFO")),
            log_format=str(server.get("log_format", _env("LOG_FORMAT", "json") or "json")),
            request_timeout=float(server.get("request_timeout", _env("REQUEST_TIMEOUT", "120") or "120")),
            video_sync_default=_bool(video.get("sync_default", _env("VIDEO_SYNC_DEFAULT", "true"))),
            max_sync_wait=float(video.get("max_sync_wait", _env("MAX_SYNC_WAIT", "300") or "300")),
            poll_interval=float(video.get("poll_interval", _env("POLL_INTERVAL", "2.0") or "2.0")),
            enable_metrics=_bool(defaults.get("enable_metrics", _env("ENABLE_METRICS", "true"))),
            mcp_enabled=_bool(mcp.get("enabled", _env("MCP_ENABLED", "false"))),
            mcp_path=str(mcp.get("path", _env("MCP_PATH", "/mcp") or "/mcp")),
            mcp_session_idle_timeout=float(
                mcp.get("session_idle_timeout", _env("MCP_SESSION_IDLE_TIMEOUT", "1800") or "1800")
            ),
            backends=backends,
            keys=keys,
        )

    @classmethod
    def _from_legacy_env(cls) -> "Settings":
        """Build backends+keys from the pre-YAML env-var layout.

        One backend per known provider type, enabled iff its ``*_API_KEY`` is
        set; a single implicit key (``env``) allows every configured backend.
        """
        # (type, api_key env, base_url env, image_model env). The model env lets
        # an operator pin/extend a backend's served image model (e.g. a brand-new
        # id not yet in the provider's hardcoded list) without editing code; the
        # registry appends it to the backend's image_models at build time.
        specs = [
            ("openai", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"),
            ("google", "GOOGLE_API_KEY", "GOOGLE_BASE_URL", "GOOGLE_MODEL"),
            ("xai", "XAI_API_KEY", "XAI_BASE_URL", "XAI_MODEL"),
            ("volcengine", "ARK_API_KEY", "ARK_BASE_URL", "ARK_MODEL"),
            ("flux", "RUNAPI_API_KEY", "FLUX_BASE_URL", "FLUX_MODEL"),
            ("openrouter", "OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "OPENROUTER_MODEL"),
            ("dashscope", "DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL", "DASHSCOPE_MODEL"),
            ("stability", "STABILITY_API_KEY", "STABILITY_BASE_URL", "STABILITY_MODEL"),
        ]
        backends: list[BackendConfig] = []
        for type_, key_env, url_env, model_env in specs:
            api_key = _env(key_env)
            if not api_key:
                continue
            model = _env(model_env)
            backends.append(BackendConfig(
                name=type_, type=type_, api_key=api_key,
                base_url=_env(url_env), tags=[],
                extra={"image_model": model} if model else {},
            ))
        # An implicit key authorises all configured backends. If the operator
        # sets GATEWAY_API_KEY, that becomes the required token; otherwise the
        # gateway is open (token matches anything, incl. absent).
        token = _env("GATEWAY_API_KEY") or ""
        keys = [KeyConfig(
            id="env", key=token,
            allow_backends=[b.name for b in backends],
            default_image_backend=_env("DEFAULT_IMAGE_PROVIDER"),
            default_video_backend=_env("DEFAULT_VIDEO_PROVIDER"),
        )]
        return cls(backends=backends, keys=keys)


def _find_config_file() -> str | None:
    for candidate in ("./mm-gateway.yaml", "./mm-gateway.yml", "/etc/mm-gateway/config.yaml"):
        if Path(candidate).exists():
            return candidate
    return None


_INTERP = re.compile(r"\$\{([A-Z0-9_]+)(?::([^}]*))?\}")


def _interpolate(text: str) -> str:
    """Replace ``${ENV}`` and ``${ENV:default}`` with environment values."""
    def repl(m: re.Match) -> str:
        name, default = m.group(1), m.group(2)
        return os.environ.get(name, default if default is not None else m.group(0))

    return _INTERP.sub(repl, text)


def _bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() == "true"
