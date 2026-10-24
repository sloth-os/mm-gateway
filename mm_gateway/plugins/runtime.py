"""Provider registration, backend credentials and routing extension points."""

from __future__ import annotations

import asyncio
import dataclasses
from contextlib import asynccontextmanager
from contextvars import ContextVar
from copy import deepcopy

import httpx
from starlette.datastructures import MutableHeaders
from structlog.contextvars import get_contextvars

from mm_gateway.config import BackendConfig
from mm_gateway.core.exceptions import GatewayError
from mm_gateway.plugins.api import (
    BackendContext,
    BackendRequest,
    HookContext,
    RouteCandidate,
    RouteContext,
)
from mm_gateway.plugins.loader import PluginLoadError
from mm_gateway.plugins.middleware import (
    FAILED,
    PipelineError,
    _invoke,
    _validate_headers,
)

OUTBOUND = ContextVar("gateway_plugin_outbound", default=None)
LEASED = ContextVar("gateway_plugin_extensions", default=None)
EXTENSION_HOOKS = {"prepare_backend", "on_backend_request"}


def request_id():
    return get_contextvars().get("request_id", "")


async def authenticate_outgoing(request: httpx.Request):
    scope = OUTBOUND.get()
    if scope:
        manager, backend, _ = scope
        await manager.authenticate_request(backend, request)


def authenticate_outgoing_sync(request: httpx.Request):
    scope = OUTBOUND.get()
    if scope:
        manager, backend, loop = scope
        hooks = manager._auth_plugins(backend, manager._active)
        if not any("on_backend_request" in p.manifest.hooks for p in hooks) and not (
            backend.extra.get("auth_plugin") and not hooks
        ):
            return
        # Synchronous SDKs run in the provider's worker thread. Never block the
        # event loop waiting on itself if an adapter calls a sync client inline.
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            raise RuntimeError(
                "Run synchronous SDK calls with asyncio.to_thread for backend authentication hooks."
            )
        asyncio.run_coroutine_threadsafe(
            manager.authenticate_request(backend, request), loop
        ).result()


class PluginExtensions:
    """Mixin used by PluginManager; one registry and one mutation lock per app."""

    def provider_factories(self, plugins=None):
        from mm_gateway.registry import _PROVIDER_CLASSES

        factories = {}
        for loaded in (self._active if plugins is None else plugins).values():
            for name, provider_type in getattr(
                loaded.instance, "providers", {}
            ).items():
                if name in factories or name in _PROVIDER_CLASSES:
                    raise PluginLoadError(
                        f"Provider type '{name}' is already registered."
                    )

                def factory(config, cls=provider_type, owner=loaded):
                    provider = cls(config)
                    provider._gateway_plugin_owner = owner
                    owner.resources.append(provider)
                    return provider

                factory.provider_class = provider_type
                factories[name] = factory
        return factories

    def _auth_plugins(self, backend, plugins):
        selected = backend.extra.get("auth_plugin")
        return [
            p
            for p in sorted(
                plugins.values(), key=lambda p: (p.options.order, p.manifest.id)
            )
            if (
                p.manifest.id == selected
                if selected
                else backend.type in p.manifest.backend_types
                or "*" in p.manifest.backend_types
            )
            and set(p.manifest.hooks) & {"prepare_backend", "on_backend_request"}
        ]

    async def prepare_backends(self, settings, plugins):
        prepared, unavailable = {}, set()
        for cfg in settings.backends:
            if not cfg.enabled or not cfg.configured:
                continue
            hooks = self._auth_plugins(cfg, plugins)
            if cfg.extra.get("auth_plugin") and not hooks:
                unavailable.add(cfg.name)
                continue
            for account_id, api_key, base_url, extra in cfg.accounts():
                if not hooks:
                    continue
                extra = {
                    **extra,
                    "__account_id": account_id,
                    "outbound_proxy": extra.get(
                        "outbound_proxy", settings.outbound_proxy
                    ),
                }
                backend = dataclasses.replace(
                    cfg,
                    api_key=api_key or cfg.api_key,
                    base_url=base_url or cfg.base_url,
                    extra=extra,
                    credentials=[],
                )
                for plugin in hooks:
                    if "prepare_backend" not in plugin.manifest.hooks:
                        continue
                    draft = deepcopy(backend)

                    def validate(result, draft=draft, backend=backend):
                        value = draft if result is None else result
                        if not isinstance(value, BackendConfig):
                            raise TypeError(
                                "prepare_backend must return BackendConfig or None"
                            )
                        for field in (
                            "name",
                            "type",
                            "tags",
                            "models",
                            "model_aliases",
                            "enabled",
                            "credentials",
                        ):
                            if getattr(value, field) != getattr(backend, field):
                                raise ValueError(
                                    "Authentication cannot change backend identity or routing configuration"
                                )

                    try:
                        result = await _invoke(
                            plugin,
                            "prepare_backend",
                            (
                                draft,
                                BackendContext(deepcopy(backend), plugin.manifest.id),
                            ),
                            validate,
                        )
                    except PipelineError:
                        raise GatewayError(
                            "A backend authentication plugin failed.",
                            code="plugin_auth_error",
                            status_code=503,
                        ) from None
                    if result is not FAILED:
                        backend = draft if result is None else result
                # Existing SDKs that require a constructor key can delegate the
                # actual wire credential to on_backend_request on every attempt.
                if (
                    not backend.api_key
                    and any("on_backend_request" in p.manifest.hooks for p in hooks)
                    and cfg.type != "vertex"
                ):
                    backend = dataclasses.replace(backend, api_key="plugin-managed")
                backend.extra["__account_id"] = account_id
                if cfg.extra.get("auth_plugin"):
                    backend.extra["auth_plugin"] = cfg.extra["auth_plugin"]
                prepared[(cfg.name, account_id)] = backend
        return prepared, unavailable

    async def prepare_runtime(self, active):
        factories = self.provider_factories(active)
        if self.app is None:
            return None
        # HTTP-only changes leave all provider clients intact.
        changed = {
            name
            for name in self._active.keys() | active.keys()
            if self._active.get(name) is not active.get(name)
        }
        extension_plugins = [
            p
            for p in (*self._active.values(), *active.values())
            if p.manifest.id in changed
            and (p.manifest.providers or set(p.manifest.hooks) & EXTENSION_HOOKS)
        ]
        if not extension_plugins:
            return None
        settings = self.app.state.settings
        prepared, unavailable = await self.prepare_backends(settings, active)
        types = {name for p in extension_plugins for name in p.manifest.providers}
        affected = {
            cfg.name
            for cfg in settings.backends
            if cfg.type in types
            or any(
                p.manifest.id in changed for p in self._auth_plugins(cfg, self._active)
            )
            or any(p.manifest.id in changed for p in self._auth_plugins(cfg, active))
            or cfg.extra.get("auth_plugin") in changed
        }
        return await asyncio.to_thread(
            self.app.state.management._prepare_registry,
            settings,
            factories,
            prepared,
            unavailable,
            affected,
            types,
        )

    def publish_runtime(self, candidate):
        if candidate is None:
            return
        registry = self.app.state.registry
        for name in (
            "_backends",
            "_configs",
            "_accounts",
            "_backend_accounts",
            "_proxies",
            "_catalog",
            "_aliases",
            "_alias_backends",
            "provider_factories",
            "prepared_backends",
        ):
            setattr(registry, name, getattr(candidate, name))
        # Client configuration and plugin revisions are independent; still
        # invalidate editors when the available provider types/models change.
        from uuid import uuid4

        self.app.state.management.revision = uuid4().hex

    @asynccontextmanager
    async def lease_extensions(self):
        plugins = tuple(self._active.values())
        for plugin in plugins:
            plugin.users += 1
        token = LEASED.set((self, plugins))
        try:
            yield plugins
        finally:
            LEASED.reset(token)
            for plugin in plugins:
                await self.release(plugin)

    async def release(self, plugin):
        plugin.users -= 1
        if plugin.retired and not plugin.users:
            await plugin.aclose()
            self._retired.discard(plugin)

    @asynccontextmanager
    async def backend_scope(self, backend):
        token = OUTBOUND.set((self, backend, asyncio.get_running_loop()))
        try:
            yield
        finally:
            OUTBOUND.reset(token)

    async def authenticate_request(self, backend, request):
        async with self.lease_extensions() as plugins:
            active = {p.manifest.id: p for p in plugins}
            hooks = self._auth_plugins(backend, active)
            if backend.extra.get("auth_plugin") and not hooks:
                raise GatewayError(
                    "The backend authentication plugin is unavailable.",
                    code="plugin_auth_error",
                    status_code=503,
                )
            if not any("on_backend_request" in p.manifest.hooks for p in hooks):
                return
            try:
                body = request.content
            except httpx.RequestNotRead:
                body = None
            draft = BackendRequest(
                request.method,
                str(request.url),
                MutableHeaders(raw=[(k.lower(), v) for k, v in request.headers.raw]),
                body,
            )
            for plugin in hooks:
                if "on_backend_request" not in plugin.manifest.hooks:
                    continue
                modified = deepcopy(draft)

                def validate(result, modified=modified):
                    if (
                        result is not None
                        or modified.method != request.method
                        or modified.body != body
                    ):
                        raise ValueError(
                            "Backend auth hooks may change only URL and headers"
                        )
                    _validate_headers(modified.headers)
                    url = httpx.URL(modified.url)
                    if url.scheme not in {"http", "https"} or not url.host:
                        raise ValueError("Backend auth requires an HTTP(S) URL")

                context = BackendContext(
                    deepcopy(backend), plugin.manifest.id, request_id()
                )
                try:
                    result = await _invoke(
                        plugin, "on_backend_request", (modified, context), validate
                    )
                except PipelineError:
                    raise GatewayError(
                        "A backend authentication plugin failed.",
                        code="plugin_auth_error",
                        status_code=503,
                    ) from None
                if result is not FAILED:
                    draft = modified
            request.headers = httpx.Headers(draft.headers.raw)
            request.url = httpx.URL(draft.url)

    async def route_plan(self, plan, request, key, *, raise_errors=True):
        leased = LEASED.get()
        if leased is None or leased[0] is not self:
            async with self.lease_extensions():
                return await self.route_plan(
                    plan, request, key, raise_errors=raise_errors
                )
        plugins = sorted(leased[1], key=lambda p: (p.options.order, p.manifest.id))
        for plugin in plugins:
            if "route" not in plugin.manifest.hooks:
                continue
            by_id = {str(i): c for i, c in enumerate(plan.candidates)}
            candidates = tuple(
                RouteCandidate(
                    i,
                    c.backend,
                    c.account_id,
                    c.model,
                    c.estimate,
                    tuple(self.app.state.registry.tags_of(c.backend)),
                )
                for i, c in by_id.items()
            )
            context = RouteContext(
                plan.modality,
                key.id if key else "",
                request.model_dump(mode="json"),
                candidates,
            )

            def validate(result, by_id=by_id):
                if result is not None and (
                    not isinstance(result, list)
                    or any(not isinstance(i, str) or i not in by_id for i in result)
                    or len(result) != len(set(result))
                ):
                    raise ValueError(
                        "route must return a unique ordered subset of candidate ids or None"
                    )

            try:
                result = await _invoke(
                    plugin,
                    "route",
                    (
                        context,
                        HookContext(plugin.manifest.id, request_id(), context.key_id),
                    ),
                    validate,
                )
            except PipelineError:
                raise GatewayError(
                    "A routing plugin failed.",
                    code="plugin_route_error",
                    status_code=503,
                ) from None
            if result is not FAILED and result is not None:
                from mm_gateway.auto_mode import Exclusion

                plan.excluded.extend(
                    Exclusion(c.model, "plugin_policy", c.estimate, c.lifecycle)
                    for i, c in by_id.items()
                    if i not in result
                )
                plan.candidates = [by_id[i] for i in result]
        if not plan.candidates and raise_errors:
            raise GatewayError(
                "No admissible backend was selected by the routing policy.",
                code="plugin_route_rejected",
                status_code=422,
            )
        return plan
