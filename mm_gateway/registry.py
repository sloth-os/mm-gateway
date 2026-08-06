"""Provider registry.

A single place where providers are constructed from settings and looked up by
name or by model alias. The registry only instantiates providers that have
valid credentials, so unconfigured providers are simply absent rather than
erroring at request time.
"""

from __future__ import annotations

import importlib
from typing import Any

from mm_gateway.config import Settings
from mm_gateway.core.base import ImageProvider, Provider, VideoProvider
from mm_gateway.core.exceptions import ModelNotFoundError, ProviderNotFoundError, ProviderNotConfiguredError
from mm_gateway.observability.logging import get_logger

log = get_logger("registry")

# provider module path -> class name
_PROVIDER_CLASSES: dict[str, str] = {
    "openai": "OpenAIProvider",
    "google": "GoogleProvider",
    "xai": "XAIProvider",
    "volcengine": "VolcengineProvider",
    "flux": "FluxProvider",
    "openrouter": "OpenRouterProvider",
    "dashscope": "DashScopeProvider",
    "stability": "StabilityProvider",
}

# Gateway-friendly model aliases -> (provider, real_model). Lets clients call a
# stable name like "gateway-image-pro" without pinning a provider-specific id.
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
}


class Registry:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._providers: dict[str, Provider] = {}
        self._aliases = dict(_MODEL_ALIASES)
        self._build()

    def _build(self) -> None:
        for name, cls_name in _PROVIDER_CLASSES.items():
            creds = self.settings.provider(name)
            if not creds.configured:
                continue
            try:
                module = importlib.import_module(f"mm_gateway.providers.{name}")
                cls = getattr(module, cls_name)
                provider = cls(creds)
                self._providers[name] = provider
                log.info("provider_registered", provider=name,
                         image=provider.supports_image, video=provider.supports_video)
            except ProviderNotConfiguredError as exc:
                log.info("provider_skipped", provider=name, reason=exc.message)
            except Exception as exc:  # noqa: BLE001
                log.error("provider_init_failed", provider=name, error=str(exc))

    @property
    def providers(self) -> dict[str, Provider]:
        return dict(self._providers)

    def list_models(self) -> list[dict[str, Any]]:
        models: list[dict[str, Any]] = []
        for alias, (prov, real) in self._aliases.items():
            modality = "video" if "video" in alias else "image"
            models.append({"id": alias, "provider": prov, "underlying": real, "modality": modality})
        for name, prov in self._providers.items():
            for m in prov.image_models:
                models.append({"id": m, "provider": name, "modality": "image"})
            for m in prov.video_models:
                models.append({"id": m, "provider": name, "modality": "video"})
        return models

    def resolve(self, model: str, provider: str | None = None, modality: str = "image") -> tuple[Provider, str]:
        """Return (provider, real_model) for a request's model + optional provider hint."""
        if provider:
            prov = self._providers.get(provider)
            if prov is None:
                raise ProviderNotConfiguredError(provider)
            return prov, model

        if model in self._aliases:
            prov_name, real_model = self._aliases[model]
            prov = self._providers.get(prov_name)
            if prov is None:
                raise ProviderNotConfiguredError(prov_name)
            return prov, real_model

        # Heuristic: a provider prefix like "openai/gpt-image-1" or a known
        # provider-specific model id declared on the provider class.
        if "/" in model:
            prefix, rest = model.split("/", 1)
            prov = self._providers.get(prefix)
            if prov is not None:
                return prov, rest

        for prov in self._providers.values():
            known = prov.image_models if modality == "image" else prov.video_models
            if model in known:
                return prov, model

        raise ModelNotFoundError(f"No provider configured for model '{model}'.")

    def get(self, name: str) -> Provider:
        prov = self._providers.get(name)
        if prov is None:
            raise ProviderNotFoundError(f"Provider '{name}' is not available.")
        return prov

    def image_provider(self, name: str) -> ImageProvider:
        prov = self.get(name)
        if not isinstance(prov, ImageProvider):
            raise ProviderNotFoundError(f"Provider '{name}' does not support image generation.")
        return prov  # type: ignore[return-value]

    def video_provider(self, name: str) -> VideoProvider:
        prov = self.get(name)
        if not isinstance(prov, VideoProvider):
            raise ProviderNotFoundError(f"Provider '{name}' does not support video generation.")
        return prov  # type: ignore[return-value]
