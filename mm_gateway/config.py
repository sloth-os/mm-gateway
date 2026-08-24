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

    Multiple accounts/keys on one backend: a backend may carry a ``credentials``
    list (each entry a mapping with ``api_key``/``base_url``/``extra``) in
    addition to — or instead of — the single ``api_key``/``base_url``/``extra``
    triple. When ``credentials`` is set the registry builds one provider instance
    per credential and the auto-router/selection layer rotates among them (and
    retries the next account on a rate-limit), so one backend name fronts a
    pool of upstream keys. A backend with no ``credentials`` behaves exactly as
    before — a single (default) account.
    """

    name: str
    type: str
    api_key: str | None = None
    base_url: str | None = None
    tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    # Per-account credentials. Each entry is a mapping with optional
    # ``api_key``/``base_url``/``extra`` keys; an entry may be a bare string
    # (treated as an ``api_key`` with inherited base_url/extra). When non-empty
    # the single ``api_key``/``base_url``/``extra`` fields are ignored for
    # account enumeration (but still gate ``configured`` below).
    credentials: list[dict[str, Any]] = field(default_factory=list)

    @property
    def configured(self) -> bool:
        # Most backends are usable when they have an API key. Vertex is the
        # exception: it authenticates with a service-account JSON key
        # (credentials_json / credentials_file) or ambient ADC, never an API
        # key — so it is configured once any Vertex signal is present
        # (credentials, project, or an explicit location). Credentials resolve
        # at build time, and location defaults to the global endpoint when
        # unset, so neither is individually required.
        if self.type == "vertex":
            return any(self.extra.get(k) for k in
                       ("credentials_json", "credentials_file", "project", "location"))
        if self.credentials:
            return any(_cred_api_key(c) for c in self.credentials)
        return bool(self.api_key)

    def accounts(self) -> list[tuple[str, str | None, str | None, dict[str, Any]]]:
        """Enumerate ``(account_id, api_key, base_url, extra)`` per credential.

        ``extra`` is the merged account-over-base dict (account-extra shadows the
        backend-level extra), so a per-account credential can override an
        endpoint or model pin while inheriting the rest. ``base_url`` is the
        account's own override (``None`` means inherit the backend-level one —
        the registry resolves it when building the per-account BackendConfig).
        ``account_id`` is ``id``/``name``/``account`` from the credential entry,
        or a synthetic ``account-<i>`` when omitted. A backend with no
        ``credentials`` yields one ``default`` account built from its single
        api_key/base_url/extra.
        """
        if not self.credentials:
            return [("default", self.api_key, None, dict(self.extra))]
        out: list[tuple[str, str | None, str | None, dict[str, Any]]] = []
        for i, cred in enumerate(self.credentials):
            account_id = _cred_id(cred, i)
            api_key = _cred_api_key(cred)
            base_url = _cred_base_url(cred)
            merged = dict(self.extra)
            cextra = cred.get("extra") if isinstance(cred, dict) else None
            if cextra:
                merged.update(cextra)
            out.append((account_id, api_key, base_url, merged))
        return out


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
    default_music_tag: str | None = None
    default_image_backend: str | None = None
    default_video_backend: str | None = None
    default_music_backend: str | None = None
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
    # Same semantics for music endpoints.
    music_sync_default: bool = field(default_factory=lambda: _env("MUSIC_SYNC_DEFAULT", "true").lower() == "true")
    # Same semantics for image endpoints. Most image backends are synchronous
    # (the blocking call runs on the first poll of the synthetic task), so this
    # only controls whether create blocks for that first poll to finish.
    image_sync_default: bool = field(default_factory=lambda: _env("IMAGE_SYNC_DEFAULT", "true").lower() == "true")
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

    @property
    def default_music_provider(self) -> str:
        for k in self.keys:
            if k.default_music_backend:
                return k.default_music_backend
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
                credentials=list(b.get("credentials") or []),
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
                default_music_tag=k.get("default_music_tag"),
                default_image_backend=k.get("default_image_backend"),
                default_video_backend=k.get("default_video_backend"),
                default_music_backend=k.get("default_music_backend"),
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
        music = _section("music")
        image = _section("image")
        defaults = _section("defaults")
        mcp = _section("mcp")
        return cls(
            host=str(server.get("host", _env("HOST", "0.0.0.0") or "0.0.0.0")),
            port=int(server.get("port", _env("PORT", "8000") or "8000")),
            log_level=str(server.get("log_level", _env("LOG_LEVEL", "INFO") or "INFO")),
            log_format=str(server.get("log_format", _env("LOG_FORMAT", "json") or "json")),
            request_timeout=float(server.get("request_timeout", _env("REQUEST_TIMEOUT", "120") or "120")),
            video_sync_default=_bool(video.get("sync_default", _env("VIDEO_SYNC_DEFAULT", "true"))),
            music_sync_default=_bool(music.get("sync_default", _env("MUSIC_SYNC_DEFAULT", "true"))),
            image_sync_default=_bool(image.get("sync_default", _env("IMAGE_SYNC_DEFAULT", "true"))),
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

        The env layout splits credentials and endpoints by modality —
        ``*_IMAGE_API_KEY`` / ``*_IMAGE_BASE_URL`` / ``*_IMAGE_MODEL`` and the
        matching ``*_VIDEO_*`` triple — so an operator can wire a provider for
        image, video, or both independently. The legacy un-split ``*_API_KEY`` /
        ``*_BASE_URL`` / ``*_MODEL`` names are still honoured as a fallback for
        each modality, so an existing deployment keeps working until it migrates.
        """
        # (type, key env pair, base_url env pair, model env pair, legacy triple,
        #  music triple). Each pair is (image_env, video_env); the image triple
        # registers/pins the served image model and the video triple the video
        # model. The music triple is (music_key_env, music_url_env,
        # music_model_env). The model env lets an operator pin/extend a backend's
        # served models (e.g. a brand-new id not yet in the provider's hardcoded
        # list) without editing code; the registry appends them to the backend's
        # image/video/music model lists at build time.
        _NO_PAIR = ("", "")   # placeholder for modality pairs a provider doesn't serve
        _NO_TRIPLE = ("", "", "")  # placeholder for modality triples a provider doesn't serve
        specs = [
            ("openai", ("OPENAI_IMAGE_API_KEY", "OPENAI_VIDEO_API_KEY"),
             ("OPENAI_IMAGE_BASE_URL", "OPENAI_VIDEO_BASE_URL"),
             ("OPENAI_IMAGE_MODEL", "OPENAI_VIDEO_MODEL"),
             ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"), _NO_TRIPLE),
            ("google", ("GOOGLE_IMAGE_API_KEY", "GOOGLE_VIDEO_API_KEY"),
             ("GOOGLE_IMAGE_BASE_URL", "GOOGLE_VIDEO_BASE_URL"),
             ("GOOGLE_IMAGE_MODEL", "GOOGLE_VIDEO_MODEL"),
             ("GOOGLE_API_KEY", "GOOGLE_BASE_URL", "GOOGLE_MODEL"),
             ("GOOGLE_MUSIC_API_KEY", "GOOGLE_MUSIC_BASE_URL", "GOOGLE_MUSIC_MODEL")),
            # Vertex AI (Gemini Enterprise Agent Platform) — same Imagen/Veo/
            # Lyria models as the AI Studio surface, reached via the
            # google-genai SDK in vertexai mode. Auth is **Application Default
            # Credentials only** (a service-account JSON key), never an API
            # key: the per-modality key/base_url pairs and the legacy un-split
            # triple are unused, so they're placeholders. The dedicated vertex
            # branch below reads VERTEX_CREDENTIALS_JSON /
            # VERTEX_CREDENTIALS_FILE / VERTEX_PROJECT / VERTEX_LOCATION and
            # registers when any of them is set — a region is optional
            # (defaults to the global endpoint). The music triple carries the
            # VERTEX_MUSIC_MODEL / VERTEX_MUSIC_BASE_URL pins (its key env is a
            # placeholder — Vertex never uses an API key).
            ("vertex", _NO_PAIR, _NO_PAIR,
             ("VERTEX_IMAGE_MODEL", "VERTEX_VIDEO_MODEL"),
             _NO_TRIPLE,
             ("VERTEX_MUSIC_API_KEY", "VERTEX_MUSIC_BASE_URL", "VERTEX_MUSIC_MODEL")),
            ("xai", ("XAI_IMAGE_API_KEY", "XAI_VIDEO_API_KEY"),
             ("XAI_IMAGE_BASE_URL", "XAI_VIDEO_BASE_URL"),
             ("XAI_IMAGE_MODEL", "XAI_VIDEO_MODEL"),
             ("XAI_API_KEY", "XAI_BASE_URL", "XAI_MODEL"), _NO_TRIPLE),
            ("volcengine", ("ARK_IMAGE_API_KEY", "ARK_VIDEO_API_KEY"),
             ("ARK_IMAGE_BASE_URL", "ARK_VIDEO_BASE_URL"),
             ("ARK_IMAGE_MODEL", "ARK_VIDEO_MODEL"),
             ("ARK_API_KEY", "ARK_BASE_URL", "ARK_MODEL"), _NO_TRIPLE),
            ("flux", ("FLUX_IMAGE_API_KEY", "FLUX_VIDEO_API_KEY"),
             ("FLUX_IMAGE_BASE_URL", "FLUX_VIDEO_BASE_URL"),
             ("FLUX_IMAGE_MODEL", "FLUX_VIDEO_MODEL"),
             ("RUNAPI_API_KEY", "FLUX_BASE_URL", "FLUX_MODEL"), _NO_TRIPLE),
            ("openrouter", ("OPENROUTER_IMAGE_API_KEY", "OPENROUTER_VIDEO_API_KEY"),
             ("OPENROUTER_IMAGE_BASE_URL", "OPENROUTER_VIDEO_BASE_URL"),
             ("OPENROUTER_IMAGE_MODEL", "OPENROUTER_VIDEO_MODEL"),
             ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "OPENROUTER_MODEL"), _NO_TRIPLE),
            ("dashscope", ("DASHSCOPE_IMAGE_API_KEY", "DASHSCOPE_VIDEO_API_KEY"),
             ("DASHSCOPE_IMAGE_BASE_URL", "DASHSCOPE_VIDEO_BASE_URL"),
             ("DASHSCOPE_IMAGE_MODEL", "DASHSCOPE_VIDEO_MODEL"),
             ("DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL", "DASHSCOPE_MODEL"), _NO_TRIPLE),
            ("stability", ("STABILITY_IMAGE_API_KEY", "STABILITY_VIDEO_API_KEY"),
             ("STABILITY_IMAGE_BASE_URL", "STABILITY_VIDEO_BASE_URL"),
             ("STABILITY_IMAGE_MODEL", "STABILITY_VIDEO_MODEL"),
             ("STABILITY_API_KEY", "STABILITY_BASE_URL", "STABILITY_MODEL"), _NO_TRIPLE),
            # Music-only providers. They carry no image/video triple.
            ("elevenlabs", _NO_PAIR, _NO_PAIR, _NO_PAIR,
             ("ELEVENLABS_API_KEY", "ELEVENLABS_BASE_URL", "ELEVENLABS_MODEL"),
             ("ELEVENLABS_MUSIC_API_KEY", "ELEVENLABS_MUSIC_BASE_URL", "ELEVENLABS_MUSIC_MODEL")),
            ("minimax", ("", "MINIMAX_VIDEO_API_KEY"),
             ("", "MINIMAX_VIDEO_BASE_URL"),
             ("", "MINIMAX_VIDEO_MODEL"),
             ("MINIMAX_API_KEY", "MINIMAX_BASE_URL", "MINIMAX_MODEL"),
             ("MINIMAX_MUSIC_API_KEY", "MINIMAX_MUSIC_BASE_URL", "MINIMAX_MUSIC_MODEL")),
            ("udioapi", _NO_PAIR, _NO_PAIR, _NO_PAIR,
             ("UDIOAPI_API_KEY", "UDIOAPI_BASE_URL", "UDIOAPI_MODEL"),
             ("UDIOAPI_MUSIC_API_KEY", "UDIOAPI_MUSIC_BASE_URL", "UDIOAPI_MUSIC_MODEL")),
            ("mureka", _NO_PAIR, _NO_PAIR, _NO_PAIR,
             ("MUREKA_API_KEY", "MUREKA_BASE_URL", "MUREKA_MODEL"),
             ("MUREKA_MUSIC_API_KEY", "MUREKA_MUSIC_BASE_URL", "MUREKA_MUSIC_MODEL")),
            ("acestep", _NO_PAIR, _NO_PAIR, _NO_PAIR,
             ("ACESTEP_API_KEY", "ACESTEP_BASE_URL", "ACESTEP_MODEL"),
             ("ACESTEP_MUSIC_API_KEY", "ACESTEP_MUSIC_BASE_URL", "ACESTEP_MUSIC_MODEL")),
        ]
        backends: list[BackendConfig] = []
        for (type_, (img_key_env, vid_key_env), (img_url_env, vid_url_env),
             (img_model_env, vid_model_env), (legacy_key, legacy_url, legacy_model),
             (mus_key_env, mus_url_env, mus_model_env)) in specs:
            # Resolve each modality's key/url/model, preferring the split
            # *_IMAGE_* / *_VIDEO_* / *_MUSIC_* env over the legacy un-split *_* name.
            img_key = _env(img_key_env) or _env(legacy_key)
            vid_key = _env(vid_key_env) or _env(legacy_key)
            mus_key = _env(mus_key_env) or _env(legacy_key)
            # Vertex AI authenticates with Application Default Credentials (a
            # service-account JSON key), NOT an API key. The key content comes
            # from VERTEX_CREDENTIALS_JSON (raw JSON, e.g. a CI secret) or
            # VERTEX_CREDENTIALS_FILE (a path, e.g. a YAML deployment); absent
            # both, the SDK falls back to ambient ADC (GOOGLE_APPLICATION_
            # CREDENTIALS / metadata server). A region (VERTEX_LOCATION) is
            # optional — unset, the provider defaults to the "global" endpoint
            # (the one Lyria 3 requires). No other backend type has an ADC path,
            # so for them a key is still required.
            if type_ == "vertex":
                project = _env("VERTEX_PROJECT")
                location = _env("VERTEX_LOCATION")
                cred_json = _env("VERTEX_CREDENTIALS_JSON")
                cred_file = _env("VERTEX_CREDENTIALS_FILE")
                # Registered when any Vertex signal is set. Credentials resolve
                # at build time (the adapter raises ProviderNotConfiguredError if
                # none of the key sources nor ambient ADC are available), so an
                # operator who only sets VERTEX_PROJECT still gets a build
                # attempt that may succeed via ambient ADC.
                if not (cred_json or cred_file or project or location):
                    continue
                extra: dict[str, Any] = {}
                if cred_json:
                    extra["credentials_json"] = cred_json
                if cred_file:
                    extra["credentials_file"] = cred_file
                if project:
                    extra["project"] = project
                if location:
                    extra["location"] = location
                img_model = _env(img_model_env)
                vid_model = _env(vid_model_env)
                mus_model = _env(mus_model_env)
                if img_model:
                    extra["image_model"] = img_model
                if vid_model:
                    extra["video_model"] = vid_model
                if mus_model:
                    extra["music_model"] = mus_model
                # Operator-pinned regional hosts. The provider reads the image
                # base from ``backend.base_url`` and the video/music bases from
                # ``extra["video_base_url"]`` / ``extra["music_base_url"]``
                # (each falling back to the image base), mirroring the google
                # adapter. So ``base_url`` gets the image pin (if any) and the
                # video/music pins are recorded only when they differ from it.
                img_base = _env("VERTEX_IMAGE_BASE_URL") or _env("VERTEX_BASE_URL")
                vid_base = _env("VERTEX_VIDEO_BASE_URL")
                mus_base = _env("VERTEX_MUSIC_BASE_URL")
                base_url = img_base
                if vid_base and vid_base != img_base:
                    extra["video_base_url"] = vid_base
                if mus_base and mus_base != img_base:
                    extra["music_base_url"] = mus_base
                backends.append(BackendConfig(
                    name=type_, type=type_, api_key=None,
                    base_url=base_url, tags=[], extra=extra,
                ))
                continue
            # A backend is registered iff at least one modality carries a key.
            # The plural *_API_KEYS variants also count: an operator may set only
            # the multi-account list (no singular key), so the plural blob is the
            # configured gate in that case and the first listed key becomes the
            # default account.
            api_key = img_key or vid_key or mus_key
            creds = _credential_list(
                _env_plural(img_key_env)
                or _env_plural(vid_key_env)
                or _env_plural(mus_key_env)
                or _env_plural(legacy_key)
            )
            if not api_key:
                if creds:
                    # Only the plural list is set — the first key is the default.
                    api_key = _cred_api_key(creds[0]) or None
                else:
                    continue
            base_url = _env(img_url_env) or _env(legacy_url) or _env(vid_url_env) or _env(mus_url_env)
            extra: dict[str, Any] = {}
            img_model = _env(img_model_env) or _env(legacy_model)
            vid_model = _env(vid_model_env)
            mus_model = _env(mus_model_env) or (_env(legacy_model) if not (img_model or vid_model) else None)
            if img_model:
                extra["image_model"] = img_model
            if vid_model:
                extra["video_model"] = vid_model
            if mus_model:
                extra["music_model"] = mus_model
            if _env(vid_url_env) and _env(vid_url_env) != base_url:
                extra["video_base_url"] = _env(vid_url_env)
            if _env(mus_url_env) and _env(mus_url_env) != base_url:
                extra["music_base_url"] = _env(mus_url_env)
            # creds (the multi-account list) was resolved above as part of the
            # configured gate; pass it through so the registry builds one
            # provider per credential.
            backends.append(BackendConfig(
                name=type_, type=type_, api_key=api_key,
                base_url=base_url, tags=[], extra=extra,
                credentials=creds,
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
            default_music_backend=_env("DEFAULT_MUSIC_PROVIDER"),
        )]
        return cls(backends=backends, keys=keys)


def _find_config_file() -> str | None:
    for candidate in ("./mm-gateway.yaml", "./mm-gateway.yml", "/etc/mm-gateway/config.yaml"):
        if Path(candidate).exists():
            return candidate
    return None


# -- credential-entry helpers (multi-account backends) ---------------------- #


def _cred_id(cred: Any, index: int) -> str:
    """A credential entry's account id (``id``/``name`` or ``account-<i>``)."""
    if isinstance(cred, dict):
        for k in ("id", "name", "account"):
            v = cred.get(k)
            if v:
                return str(v)
    return f"account-{index}"


def _cred_api_key(cred: Any) -> str | None:
    """Extract an API key from a credential entry (dict or bare string)."""
    if isinstance(cred, str):
        return cred or None
    if isinstance(cred, dict):
        v = cred.get("api_key") or cred.get("key")
        return v or None
    return None


def _cred_base_url(cred: Any) -> str | None:
    """Extract a base_url override from a credential entry."""
    if isinstance(cred, dict):
        return cred.get("base_url")
    return None


def _env_plural(name: str | None) -> str | None:
    """Read the plural ``<NAME>S`` env var (multi-account key list)."""
    if not name:
        return None
    return _env(f"{name}S")


def _credential_list(raw: str | None) -> list[dict[str, Any]]:
    """Parse a comma/newline-separated multi-account key blob into credentials.

    Whitespace-only entries are dropped; the remaining tokens each become a
    bare ``{"api_key": token}`` credential entry (account ids auto-assigned by
    :meth:`BackendConfig.accounts`). Returns ``[]`` (single-account default)
    when the blob is empty — preserving the legacy single-key behaviour.
    """
    if not raw:
        return []
    keys = [t.strip() for t in raw.replace("\n", ",").split(",") if t.strip()]
    return [{"api_key": k} for k in keys]


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
