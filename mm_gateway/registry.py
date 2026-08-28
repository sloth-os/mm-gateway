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

from mm_gateway.config import BackendConfig, KeyConfig, ProxyConfig, Settings
from mm_gateway.core.base import ImageProvider, MusicProvider, Provider, VideoProvider
from mm_gateway.core.exceptions import (
    ForbiddenError,
    GatewayError,
    ModelNotFoundError,
    ProviderNotConfiguredError,
    ProviderNotFoundError,
    ValidationError,
)
from mm_gateway.models.limits import limits_for
from mm_gateway.observability.logging import get_logger
from mm_gateway.observability.selection import STORE as SELECTION_STORE
import mm_gateway.router as router

log = get_logger("registry")

# provider type -> class name (the provider module is mm_gateway.providers.<type>)
_PROVIDER_CLASSES: dict[str, str] = {
    "openai": "OpenAIProvider",
    "google": "GoogleProvider",
    "vertex": "VertexProvider",
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
    # Vertex AI serves the same Imagen/Veo ids as the AI Studio surface; distinct
    # aliases let a client target the Enterprise platform host explicitly.
    "gateway-image-vertex": ("vertex", "imagen-4.0-generate-001"),
    "gateway-image-grok": ("xai", "grok-imagine-image"),
    "gateway-image-seedream": ("volcengine", "doubao-seedream-3-0-t2i-250415"),
    "gateway-image-wanx": ("dashscope", "wanx2.1-t2i-turbo"),
    "gateway-image-sd": ("stability", "stable-image-core"),
    "gateway-video-pro": ("volcengine", "doubao-seedance-1-0-pro-250528"),
    "gateway-video-veo": ("google", "veo-2.0-generate-001"),
    "gateway-video-vertex": ("vertex", "veo-2.0-generate-001"),
    "gateway-video-sora": ("openai", "sora-2"),
    "gateway-video-grok": ("xai", "grok-imagine-video"),
    "gateway-video-wan": ("dashscope", "wanx2.1-t2v-turbo"),
    "gateway-video-svd": ("stability", "stable-video-diffusion"),
    # MiniMax H3 — one omni model; the content parts pick t2v / i2v / r2v.
    "gateway-video-minimax": ("minimax", "MiniMax-H3"),
    # Seedance 2.0 is a single Ark model; the content parts pick t2v / i2v, so
    # both aliases resolve to the same omni model id.
    "gateway-video-seedance-2": ("volcengine", "doubao-seedance-2-0-260128"),
    "gateway-video-seedance-2-i2v": ("volcengine", "doubao-seedance-2-0-260128"),
    # Music aliases (Gemini Lyria 3 is the front-end shape; each backend serves a
    # stable id under a friendlier name).
    "gateway-music-lyria": ("google", "lyria-3"),
    "gateway-music-vertex": ("vertex", "lyria-3"),
    "gateway-music-elevenlabs": ("elevenlabs", "music_v2"),
    "gateway-music-minimax": ("minimax", "music-3.0"),
    "gateway-music-udio": ("udioapi", "chirp-v5"),
    "gateway-music-mureka": ("mureka", "mureka-song-1"),
    "gateway-music-acestep": ("acestep", "ace-step-1.5"),
}


class Registry:
    def __init__(self, settings: Settings):
        self.settings = settings
        # backend name -> Provider instance (the *default* account's instance;
        # kept for backward compatibility with ``providers``/``backends``).
        self._backends: dict[str, Provider] = {}
        # backend name -> BackendConfig
        self._configs: dict[str, BackendConfig] = {}
        # (backend name, account id) -> Provider instance (one per credential).
        self._accounts: dict[tuple[str, str], Provider] = {}
        # backend name -> ordered account ids (at least one: "default").
        self._backend_accounts: dict[str, list[str]] = {}
        self._aliases = dict(_MODEL_ALIASES)
        # proxy domain -> ProxyConfig (the configured, pass-through proxies).
        self._proxies: dict[str, ProxyConfig] = {}
        self._build()

    def _build(self) -> None:
        for cfg in self.settings.backends:
            if not cfg.configured:
                continue
            cls_name = _PROVIDER_CLASSES.get(cfg.type)
            if cls_name is None:
                log.warning("unknown_backend_type", backend=cfg.name, type=cfg.type)
                continue
            module = importlib.import_module(f"mm_gateway.providers.{cfg.type}")
            cls = getattr(module, cls_name)
            accounts = cfg.accounts()
            account_ids: list[str] = []
            for account_id, api_key, base_url, extra in accounts:
                per_cfg = self._per_account_config(cfg, account_id, api_key,
                                                   base_url, extra)
                try:
                    provider = cls(per_cfg)
                except ProviderNotConfiguredError as exc:
                    log.info("backend_account_skipped", backend=cfg.name,
                             account=account_id, reason=exc.message)
                    continue
                except Exception as exc:  # noqa: BLE001
                    log.error("backend_account_init_failed", backend=cfg.name,
                              account=account_id, error=str(exc))
                    continue
                self._apply_pinned_models(provider, per_cfg)
                self._accounts[(cfg.name, account_id)] = provider
                account_ids.append(account_id)
            if not account_ids:
                # Every account failed to construct → the backend is unusable.
                log.info("backend_skipped", backend=cfg.name, reason="no account built")
                continue
            self._configs[cfg.name] = cfg
            self._backend_accounts[cfg.name] = account_ids
            # The default account (first one) backs the legacy single-instance
            # accessors and explicit (non-auto) resolution.
            default = self._accounts[(cfg.name, account_ids[0])]
            self._backends[cfg.name] = default
            log.info("backend_registered", backend=cfg.name, type=cfg.type,
                     tags=cfg.tags, accounts=account_ids, image=default.supports_image,
                     video=default.supports_video, music=default.supports_music)

        # General pass-through proxies. Unlike provider backends they carry no
        # SDK instance — the proxy runner forwards raw HTTP/WebSocket — so only
        # the configured ProxyConfig is retained, keyed by upstream domain (the
        # proxy's routing identity) for the routes.
        for proxy in self.settings.proxies:
            if not proxy.configured:
                continue
            if proxy.host in self._proxies or proxy.host in self._backends:
                log.warning("proxy_domain_collision", domain=proxy.host)
                continue
            self._proxies[proxy.host] = proxy
            log.info("proxy_registered", proxy=proxy.host,
                     base_url=proxy.base_url, tags=proxy.tags,
                     accounts=[aid for aid, _ in proxy.enumerate_accounts()])

    def _per_account_config(self, cfg: BackendConfig, account_id: str,
                            api_key: str | None, base_url: str | None,
                            extra: dict[str, Any]) -> BackendConfig:
        """Build a per-account BackendConfig so the provider sees the right creds.

        A provider constructor reads ``backend.api_key`` / ``backend.base_url`` /
        ``backend.extra``; for a multi-account backend each account carries its
        own key (and optional base_url). The account id is stashed on ``extra``
        so adapters/selection can attribute outcomes to the right account.
        """
        merged = dict(extra)
        merged.setdefault("__account_id", account_id)
        return BackendConfig(
            name=cfg.name, type=cfg.type, api_key=api_key or cfg.api_key,
            base_url=base_url or cfg.base_url, tags=list(cfg.tags), extra=merged,
        )

    def _apply_pinned_models(self, provider: Provider, cfg: BackendConfig) -> None:
        # Honor an operator-pinned image model (BackendConfig.extra[
        # "image_model"], set by the legacy env-var layout's *_MODEL). Append it
        # to this instance's served list so resolve() and /v1/models accept it
        # even if it isn't in the provider's hardcoded catalogue (e.g. a freshly
        # released model id). The instance attribute shadows the ClassVar, so
        # other backends of the same type are unaffected.
        extra_model = cfg.extra.get("image_model")
        if extra_model and provider.supports_image and extra_model not in provider.image_models:
            provider.image_models = [*provider.image_models, extra_model]
        extra_video = cfg.extra.get("video_model")
        if extra_video and provider.supports_video and extra_video not in provider.video_models:
            provider.video_models = [*provider.video_models, extra_video]
        extra_music = cfg.extra.get("music_model")
        if extra_music and provider.supports_music and extra_music not in provider.music_models:
            provider.music_models = [*provider.music_models, extra_music]

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
        for name in self._backends:
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

    def proxy_names(self) -> list[str]:
        """Proxy domains (the routing identity), for diagnostics/listing."""
        return list(self._proxies)

    def proxy(self, domain: str) -> ProxyConfig:
        """Return the configured :class:`ProxyConfig` for ``domain``.

        ``domain`` is the upstream host (the proxy's routing identity, matched
        case-insensitively against the request's ``/proxy/{domain}/...`` path
        segment). Raises ``ProviderNotFoundError`` (404) when no such proxy is
        configured; ``ForbiddenError`` (403) when it exists but the
        authenticated key is not allowed to use it — both surface so a caller
        can tell a typo from a permissions gap.
        """
        proxy = self._proxies.get(domain)
        if proxy is None:
            raise ProviderNotFoundError(f"Proxy '{domain}' is not configured.")
        return proxy

    def usable_proxies(self, key: KeyConfig | None) -> list[str]:
        """Proxy domains a key may use, via the same hybrid allow/deny rule.

        A key is authorised for a proxy by the proxy's domain in
        ``allow_backends``, by ``tags`` intersection, or as an open key (no
        allow_tags/allow_backends). Domains the key denies (in ``deny_tags``)
        are excluded.
        """
        if key is None:
            return list(self._proxies)
        allowed: list[str] = []
        for domain, proxy in self._proxies.items():
            if domain in key.deny_tags:
                continue
            tag_ok = bool(set(proxy.tags) & set(key.allow_tags))
            domain_ok = domain in key.allow_backends
            open_key = not key.allow_tags and not key.allow_backends
            if tag_ok or domain_ok or open_key:
                allowed.append(domain)
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

    def list_public_models(self, key: KeyConfig | None = None) -> list[dict[str, str]]:
        """Return the provider-neutral model catalogue exposed to clients."""
        public: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for model in self.list_models(key):
            identity = (model["id"], model["modality"])
            if identity in seen:
                continue
            seen.add(identity)
            public.append({
                "id": model["id"],
                "object": "model",
                "modality": model["modality"],
            })
        return public

    def list_model_limits(self, key: KeyConfig | None = None) -> list[dict[str, Any]]:
        """Return the catalogue with each model's documented input/output limits.

        Each entry is the public model object plus a ``limits`` member holding
        the neutral limits (input modalities, max prompt length, max output
        count, supported sizes/durations, per-role support flags, ...). An
        alias resolves to its underlying model's limits so a client crafting a
        prompt for ``gateway-image-pro`` sees ``gpt-image-1``'s real limits.
        Models with no documented entry get a permissive ``limits`` object.
        """
        entries: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for model in self.list_models(key):
            identity = (model["id"], model["modality"])
            if identity in seen:
                continue
            seen.add(identity)
            modality = model["modality"]
            # An alias pins a backend *type*; its limits are the underlying
            # model's. A raw underlying id is looked up directly.
            underlying = model["id"]
            if underlying in self._aliases:
                _btype, underlying = self._aliases[underlying]
            limits = limits_for(underlying, modality).to_public_dict()
            entries.append({
                "id": model["id"],
                "object": "model",
                "modality": modality,
                "limits": limits,
            })
        return entries

    # Model id values that mean "the gateway should pick a backend+model for me".
    # A request that omits ``model`` (None) or sets it to ``auto`` triggers
    # auto-routing via :mod:`mm_gateway.router` + the limits catalogue.
    AUTO_MODEL: frozenset[str] = frozenset({"", "auto"})

    def resolve(
        self,
        model: str | None,
        key: KeyConfig | None = None,
        *,
        modality: str = "image",
        tag: str | None = None,
        backend_name: str | None = None,
        request: Any = None,
    ) -> tuple[Provider, str, str]:
        """Return ``(provider, real_model, backend_name)`` for a request.

        ``key`` scopes which backends are usable (raises ``ForbiddenError`` if
        none). Within the usable set, ``backend_name`` > ``tag`` > the key's
        per-modality defaults > first usable backend that serves the model.

        When ``model`` is ``None`` or ``"auto"`` (auto-routing), ``request``
        must carry the unified request so the router can score each candidate
        model's limits against the request's input profile. The explicit
        ``backend_name``/``tag`` overrides still apply as preferences.
        """
        if model is None or model.lower() in self.AUTO_MODEL:
            if request is None:
                raise ValidationError(
                    "Auto-routing requires the request payload; pass the unified "
                    "request to resolve()."
                )
            return self.resolve_auto(request, key, modality=modality, tag=tag,
                                    backend_name=backend_name)

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
            if not tagged:
                raise ValidationError(
                    f"Routing profile '{tag}' is unavailable for model '{model}'."
                )
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

    def resolve_auto(
        self,
        request: Any,
        key: KeyConfig | None = None,
        *,
        modality: str = "image",
        tag: str | None = None,
        backend_name: str | None = None,
    ) -> tuple[Provider, str, str]:
        """Pick a backend+model whose documented limits fit the request.

        Enumerates every model the usable backends serve for ``modality``,
        scores each against the request's input profile (modality, roles,
        prompt length, output count, dimensions/duration/fps) using the limits
        catalogue, and prefers — among the fitting candidates — the key's
        per-modality default backend, then its default tag, then the candidate
        that satisfies the most optional controls, then a stable config order.
        Raises ``ModelNotFoundError`` when no configured model can serve the
        request (every candidate failed a hard limit or modality check).
        """
        usable = self.usable_backends(key) if key else list(self._backends)
        if key and not usable:
            raise ForbiddenError(f"API key '{key.id}' is not allowed to use any backend.")
        # An explicit backend pin scopes auto-routing to that backend only.
        if backend_name and backend_name in usable:
            usable = [backend_name]

        profile = router.profile_for(request)
        default_backend = self._key_default_backend(key, modality)
        default_tag = self._key_default_tag(key, modality)

        # (sort_key, name, model). The sort key encodes the preference order:
        # key default backend > key default tag > most optional controls > the
        # stable backend/model index. (All ascending except optional_hits.)
        scored: list[tuple[tuple[int, int, int, int, int], str, str]] = []
        for b_index, name in enumerate(usable):
            prov = self._backends[name]
            if not self._serves_modality(prov, modality):
                continue
            cfg_tags = self._configs[name].tags if name in self._configs else []
            # A routing-profile tag override scopes candidates to tagged backends.
            if tag and tag not in cfg_tags:
                continue
            default_backend_rank = 0 if name == default_backend else 1
            default_tag_rank = 0 if (default_tag and default_tag in cfg_tags) else 1
            for m_index, model in enumerate(self._modality_models(prov, modality)):
                limits = limits_for(model, modality)
                s = router.score(profile, limits, backend_index=b_index,
                                 model_index=m_index)
                if not s.fits:
                    continue
                sort_key = (default_backend_rank, default_tag_rank,
                            -s.optional_hits, b_index, m_index)
                scored.append((sort_key, name, model))

        if not scored:
            # Auto-route failure is a 422 validation error (the request's input
            # is incompatible with every configured model), not a 404: no
            # specific model id was requested. The explicit-model "no backend
            # serves this id" path above keeps raising ModelNotFoundError (404).
            raise GatewayError(
                "No configured model can serve this auto-routed request; relax "
                "the input (fewer images, smaller dimensions, shorter duration) "
                "or set an explicit model.",
                code="validation_error",
                status_code=422,
            )
        scored.sort(key=lambda item: item[0])
        candidates = self._rank_auto_candidates(scored, modality)
        if not candidates:
            # Auto-route failure is a 422 validation error (the request's input
            # is incompatible with every configured model), not a 404: no
            # specific model id was requested. The explicit-model "no backend
            # serves this id" path above keeps raising ModelNotFoundError (404).
            raise GatewayError(
                "No configured model can serve this auto-routed request; relax "
                "the input (fewer images, smaller dimensions, shorter duration) "
                "or set an explicit model.",
                code="validation_error",
                status_code=422,
            )
        prov, account_id, model, name = candidates[0]
        # Stash the account on the provider instance so the service's
        # outcome-recording (latency / rate-limit) attributes to the right
        # credential. Set defensively — some test fakes may not allow setattr.
        try:
            prov.account_id = account_id  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
        return prov, model, name

    def enumerate_auto_candidates(
        self,
        request: Any,
        key: KeyConfig | None = None,
        *,
        modality: str = "image",
        tag: str | None = None,
        backend_name: str | None = None,
    ) -> list[tuple[Provider, str, str, str]]:
        """Return ``(provider, account_id, model, backend_name)`` ranked best-first.

        Same fit/limits logic as :meth:`resolve_auto`, but every fitting candidate
        is returned in health-ranked order rather than just the top one. The
        ranking blends the router's static preference (key defaults > most
        optional controls > stable order) with the selection store's live health
        score (success rate + latency + rate-limit cooldown): a candidate in a
        rate-limit cooldown sinks to the bottom, a healthy fast one rises. Each
        backend's accounts are interleaved so a multi-account backend contributes
        one candidate per credential — the retry layer then tries the next
        account (same backend) or the next backend in turn.

        Used by the services' retry-across-backends loop in auto mode. Raises
        the same 422 :class:`GatewayError` as :meth:`resolve_auto` when nothing
        fits.
        """
        usable = self.usable_backends(key) if key else list(self._backends)
        if key and not usable:
            raise ForbiddenError(f"API key '{key.id}' is not allowed to use any backend.")
        if backend_name and backend_name in usable:
            usable = [backend_name]

        profile = router.profile_for(request)
        default_backend = self._key_default_backend(key, modality)
        default_tag = self._key_default_tag(key, modality)

        scored: list[tuple[tuple[int, int, int, int, int], str, str]] = []
        for b_index, name in enumerate(usable):
            prov = self._backends[name]
            if not self._serves_modality(prov, modality):
                continue
            cfg_tags = self._configs[name].tags if name in self._configs else []
            if tag and tag not in cfg_tags:
                continue
            default_backend_rank = 0 if name == default_backend else 1
            default_tag_rank = 0 if (default_tag and default_tag in cfg_tags) else 1
            for m_index, model in enumerate(self._modality_models(prov, modality)):
                limits = limits_for(model, modality)
                s = router.score(profile, limits, backend_index=b_index,
                                 model_index=m_index)
                if not s.fits:
                    continue
                sort_key = (default_backend_rank, default_tag_rank,
                            -s.optional_hits, b_index, m_index)
                scored.append((sort_key, name, model))
        scored.sort(key=lambda item: item[0])
        return self._rank_auto_candidates(scored, modality)

    def _rank_auto_candidates(
        self, scored: list[tuple[tuple, str, str]], modality: str,
    ) -> list[tuple[Provider, str, str, str]]:
        """Expand a static-ranked list into per-account, health-ranked candidates.

        ``scored`` is the router's static ordering (key defaults → optional
        hits → stable index). For each ``(backend, model)`` entry we emit one
        candidate per account (credential) the backend carries, attribute it a
        live health score from the selection store, and reorder so healthy,
        fast, non-rate-limited candidates come first while keeping the static
        preference as a tie-breaker. A candidate in a rate-limit cooldown sinks
        to the very bottom (the retry layer still tries it last rather than
        dropping it, in case the cooldown is stale). Returns ``[]`` when no
        candidate fits.
        """
        if not scored:
            return []
        expanded: list[tuple[tuple, Provider, str, str, str]] = []
        for rank, name, model in scored:
            account_ids = self._backend_accounts.get(name, ["default"])
            for account_id in account_ids:
                prov = self._accounts.get((name, account_id)) or self._backends.get(name)
                if prov is None:
                    continue
                health = SELECTION_STORE.score(backend=name, account=account_id,
                                               model=model, modality=modality)
                rate_limited = SELECTION_STORE.is_rate_limited(
                    backend=name, account=account_id, model=model, modality=modality,
                )
                # Sort key: (rate_limited_flag desc→so healthy first, then
                # health desc, then the static rank asc). ``rate_limited`` is
                # 0/1 so healthy (0) sorts before rate-limited (1).
                health_key = (1 if rate_limited else 0, -health, rank, name, account_id)
                expanded.append((health_key, prov, account_id, model, name))
        expanded.sort(key=lambda item: (item[0][0], item[0][1], item[0][2]))
        return [(p, aid, m, n) for (_, p, aid, m, n) in expanded]

    # -- helpers ------------------------------------------------------------ #

    def _serves_modality(self, prov: Provider, modality: str) -> bool:
        if modality == "image":
            return prov.supports_image
        if modality == "video":
            return prov.supports_video
        return prov.supports_music

    def _key_default_backend(self, key: KeyConfig | None, modality: str) -> str | None:
        if not key:
            return None
        if modality == "image":
            return key.default_image_backend
        if modality == "video":
            return key.default_video_backend
        return key.default_music_backend

    def _key_default_tag(self, key: KeyConfig | None, modality: str) -> str | None:
        if not key:
            return None
        if modality == "image":
            return key.default_image_tag
        if modality == "video":
            return key.default_video_tag
        return key.default_music_tag

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
