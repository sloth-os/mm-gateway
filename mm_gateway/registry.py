"""Provider registry.

A single place where providers are constructed from settings and looked up by
name or by model alias. One ``Provider`` instance is built **per named
backend** (so two Volcengine accounts, say ``volcengine-prod`` and
``volcengine-staging``, are two distinct instances). The registry only
instantiates backends that have valid credentials, so unconfigured backends are
simply absent rather than erroring at request time.

Routing is *hybrid*: a front-end key allows a backend if the backend's tags
intersect the key's ``allow_tags`` OR the backend's name is in the key's
``allow_backends``, and the backend's name is not in the key's ``deny_tags``.
Within the usable set, the choice follows: an explicit ``backend_name`` (from
the request) > an explicit ``tag`` (from the request) > the key's per-modality
default tag/backend > the first usable backend.
"""

from __future__ import annotations

import importlib
from typing import Any

from mm_gateway.config import BackendConfig, KeyConfig, Settings
from mm_gateway.core.base import ImageProvider, MusicProvider, Provider, VideoProvider
from mm_gateway.core.exceptions import (
    ForbiddenError,
    ModelNotFoundError,
    ProviderNotConfiguredError,
    ProviderNotFoundError,
)
from mm_gateway.observability.logging import get_logger

log = get_logger("registry")

# provider type -> class name (the provider module is mm_gateway.providers.<type>)
_PROVIDER_CLASSES: dict[str, str] = {
    "openai": "OpenAIProvider",
    "google": "GoogleProvider",
    "xai": "XAIProvider",
    "volcengine": "VolcengineProvider",
    "flux": "FluxProvider",
    "openrouter": "OpenRouterProvider",
    "dashscope": "DashScopeProvider",
    "stability": "StabilityProvider",
    "elevenlabs": "ElevenLabsProvider",
    "minimax": "MiniMaxProvider",
    "udioapi": "UdioApiProvider",
    "mureka": "MurekaProvider",
    "acestep": "AceStepProvider",
}

# Gateway-friendly model aliases -> (backend type, real_model). Lets clients call
# a stable name like "gateway-image-pro" without pinning a backend-specific id.
_MODEL_ALIASES: dict[str, tuple[str, str]] = {
    "gateway-image-pro": ("openai", "gpt-image-1"),
    "gateway-image-fast": ("openai", "gpt-image-1-mini"),
    "gateway-image-flux": ("flux", "flux-2-pro-text-to-image"),
    "gateway-image-imagen": ("google", "imagen-4.0-generate-001"),
    "gateway-image-grok": ("xai", "grok-imagine-image"),
    "gateway-image-seedream": ("volcengine", "doubao-seedream-3-0-t2i-250415"),
    "gateway-image-wanx": ("dashscope", "wanx2.1-t2i-turbo"),
    "gateway-image-sd": ("stability", "stable-image-core"),
    "gateway-video-pro": ("volcengine", "doubao-seedance-1-0-pro-250528"),
    "gateway-video-veo": ("google", "veo-2.0-generate-001"),
    "gateway-video-sora": ("openai", "sora-2"),
    "gateway-video-grok": ("xai", "grok-imagine-video"),
    "gateway-video-wan": ("dashscope", "wanx2.1-t2v-turbo"),
    "gateway-video-svd": ("stability", "stable-video-diffusion"),
    # Seedance 2.0 is a single Ark model; the content parts pick t2v / i2v, so
    # both aliases resolve to the same omni model id.
    "gateway-video-seedance-2": ("volcengine", "doubao-seedance-2-0-260128"),
    "gateway-video-seedance-2-i2v": ("volcengine", "doubao-seedance-2-0-260128"),
    # Music aliases (Gemini Lyria 3 is the front-end shape; each backend serves a
    # stable id under a friendlier name).
    "gateway-music-lyria": ("google", "lyria-3"),
    "gateway-music-elevenlabs": ("elevenlabs", "music_v2"),
    "gateway-music-minimax": ("minimax", "music-3.0"),
    "gateway-music-udio": ("udioapi", "udio-v2"),
    "gateway-music-mureka": ("mureka", "mureka-song-1"),
    "gateway-music-acestep": ("acestep", "ace-step-1.5"),
}


class Registry:
    def __init__(self, settings: Settings):
        self.settings = settings
        # backend name -> Provider instance
        self._backends: dict[str, Provider] = {}
        # backend name -> BackendConfig
        self._configs: dict[str, BackendConfig] = {}
        self._aliases = dict(_MODEL_ALIASES)
        self._build()

    def _build(self) -> None:
        for cfg in self.settings.backends:
            if not cfg.configured:
                continue
            cls_name = _PROVIDER_CLASSES.get(cfg.type)
            if cls_name is None:
                log.warning("unknown_backend_type", backend=cfg.name, type=cfg.type)
                continue
            try:
                module = importlib.import_module(f"mm_gateway.providers.{cfg.type}")
                cls = getattr(module, cls_name)
                provider = cls(cfg)
                # Honor an operator-pinned image model (BackendConfig.extra[
                # "image_model"], set by the legacy env-var layout's *_MODEL).
                # Append it to this instance's served list so resolve() and
                # /v1/models accept it even if it isn't in the provider's
                # hardcoded catalogue (e.g. a freshly released model id). The
                # instance attribute shadows the ClassVar, so other backends of
                # the same type are unaffected.
                extra_model = cfg.extra.get("image_model")
                if extra_model and provider.supports_image and extra_model not in provider.image_models:
                    provider.image_models = [*provider.image_models, extra_model]
                # Same treatment for a video model pinned via the legacy env
                # layout's *_VIDEO_MODEL (extra["video_model"]).
                extra_video = cfg.extra.get("video_model")
                if extra_video and provider.supports_video and extra_video not in provider.video_models:
                    provider.video_models = [*provider.video_models, extra_video]
                # Same treatment for a music model pinned via the legacy env
                # layout's *_MUSIC_MODEL (extra["music_model"]).
                extra_music = cfg.extra.get("music_model")
                if extra_music and provider.supports_music and extra_music not in provider.music_models:
                    provider.music_models = [*provider.music_models, extra_music]
                self._backends[cfg.name] = provider
                self._configs[cfg.name] = cfg
                log.info("backend_registered", backend=cfg.name, type=cfg.type,
                         tags=cfg.tags, image=provider.supports_image,
                         video=provider.supports_video, music=provider.supports_music)
            except ProviderNotConfiguredError as exc:
                log.info("backend_skipped", backend=cfg.name, reason=exc.message)
            except Exception as exc:  # noqa: BLE001
                log.error("backend_init_failed", backend=cfg.name, error=str(exc))

    @property
    def providers(self) -> dict[str, Provider]:
        """Backwards-compatible alias keyed by backend name."""
        return dict(self._backends)

    @property
    def backends(self) -> dict[str, Provider]:
        return dict(self._backends)

    def backend_names(self) -> list[str]:
        return list(self._backends)

    def usable_backends(self, key: KeyConfig) -> list[str]:
        """Backend names the key may use, per the hybrid allow/deny rule."""
        allowed: list[str] = []
        for name, prov in self._backends.items():
            cfg = self._configs[name]
            if name in key.deny_tags or cfg.type in key.deny_tags:
                continue
            tag_ok = bool(set(cfg.tags) & set(key.allow_tags))
            name_ok = name in key.allow_backends
            # An open key (no allow_tags/allow_backends) may use every backend.
            open_key = not key.allow_tags and not key.allow_backends
            if tag_ok or name_ok or open_key:
                allowed.append(name)
        return allowed

    def _modality_models(self, prov: Provider, modality: str) -> list[str]:
        if modality == "image":
            return prov.image_models
        if modality == "music":
            return prov.music_models
        return prov.video_models

    def list_models(self, key: KeyConfig | None = None) -> list[dict[str, Any]]:
        models: list[dict[str, Any]] = []
        usable = self.usable_backends(key) if key else list(self._backends)
        usable_set = set(usable)
        for alias, (btype, real) in self._aliases.items():
            # Include an alias iff at least one usable backend is of its type.
            if any(self._configs[n].type == btype for n in usable if n in self._configs):
                if "video" in alias:
                    modality = "video"
                elif "music" in alias:
                    modality = "music"
                else:
                    modality = "image"
                models.append({"id": alias, "type": btype, "underlying": real, "modality": modality})
        for name in usable:
            prov = self._backends[name]
            for m in prov.image_models:
                models.append({"id": m, "provider": name, "modality": "image"})
            for m in prov.video_models:
                models.append({"id": m, "provider": name, "modality": "video"})
            for m in prov.music_models:
                models.append({"id": m, "provider": name, "modality": "music"})
        return models

    def resolve(
        self,
        model: str,
        key: KeyConfig | None = None,
        *,
        modality: str = "image",
        tag: str | None = None,
        backend_name: str | None = None,
    ) -> tuple[Provider, str, str]:
        """Return ``(provider, real_model, backend_name)`` for a request.

        ``key`` scopes which backends are usable (raises ``ForbiddenError`` if
        none). Within the usable set, ``backend_name`` > ``tag`` > the key's
        per-modality defaults > first usable backend that serves the model.
        """
        usable = self.usable_backends(key) if key else list(self._backends)
        if key and not usable:
            raise ForbiddenError(f"API key '{key.id}' is not allowed to use any backend.")

        # Resolve the model alias (if any) to (type, real_model). Aliases pin a
        # backend *type*, not a name — any usable backend of that type serves it.
        alias_type: str | None = None
        real_model = model
        if model in self._aliases:
            alias_type, real_model = self._aliases[model]
        elif "/" in model:
            prefix, rest = model.split("/", 1)
            if prefix in (self._configs[n].type for n in usable):
                alias_type, real_model = prefix, rest

        def serves(name: str) -> bool:
            prov = self._backends[name]
            # An explicit backend pin trusts the caller's choice entirely; the
            # service layer validates modality support afterwards. This lets a
            # pinned provider with a dynamic catalogue (or a brand-new model id
            # not yet in its advertised list) still be reached.
            if backend_name and name == backend_name:
                return True
            # A backend that doesn't implement the requested modality cannot
            # serve it, regardless of its advertised model list.
            if modality == "image" and not prov.supports_image:
                return False
            if modality == "video" and not prov.supports_video:
                return False
            if modality == "music" and not prov.supports_music:
                return False
            known = self._modality_models(prov, modality)
            # A backend serves the request if it knows the real model, or if the
            # model is an alias of its type, or if it advertises no fixed list
            # (OpenRouter's catalogue is dynamic).
            return real_model in known or alias_type == self._configs[name].type or not known

        candidates = [n for n in usable if serves(n)]
        if not candidates:
            raise ModelNotFoundError(f"No backend configured for model '{model}'.")

        chosen: str | None = None
        if backend_name and backend_name in candidates:
            chosen = backend_name
        elif tag:
            tagged = [n for n in candidates if tag in self._configs[n].tags]
            if tagged:
                chosen = tagged[0]
        if chosen is None and key:
            if modality == "image":
                default = key.default_image_backend
                default_tag = key.default_image_tag
            elif modality == "music":
                default = key.default_music_backend
                default_tag = key.default_music_tag
            else:
                default = key.default_video_backend
                default_tag = key.default_video_tag
            if default and default in candidates:
                chosen = default
            elif default_tag:
                tagged = [n for n in candidates if default_tag in self._configs[n].tags]
                if tagged:
                    chosen = tagged[0]
        if chosen is None:
            chosen = candidates[0]

        return self._backends[chosen], real_model, chosen

    def get(self, name: str) -> Provider:
        prov = self._backends.get(name)
        if prov is None:
            raise ProviderNotFoundError(f"Backend '{name}' is not available.")
        return prov

    def image_provider(self, name: str) -> ImageProvider:
        prov = self.get(name)
        if not isinstance(prov, ImageProvider):
            raise ProviderNotFoundError(f"Backend '{name}' does not support image generation.")
        return prov  # type: ignore[return-value]

    def video_provider(self, name: str) -> VideoProvider:
        prov = self.get(name)
        if not isinstance(prov, VideoProvider):
            raise ProviderNotFoundError(f"Backend '{name}' does not support video generation.")
        return prov  # type: ignore[return-value]

    def music_provider(self, name: str) -> MusicProvider:
        prov = self.get(name)
        if not isinstance(prov, MusicProvider):
            raise ProviderNotFoundError(f"Backend '{name}' does not support music generation.")
        return prov  # type: ignore[return-value]
