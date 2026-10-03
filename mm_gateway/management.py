"""Atomic administrative updates to the running gateway and optional disk overlay."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import re
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from pydantic import ValidationError as PydanticValidationError

from mm_gateway.config import (BackendConfig, KeyBudget, KeyConfig, ProxyConfig, RoutingProfile, Settings,
                               _cred_api_key, _cred_base_url, _cred_id)
from mm_gateway.core.exceptions import GatewayError
from mm_gateway.observability.logging import get_logger
from mm_gateway.observability.metrics import STORE as METRICS
from mm_gateway.observability.selection import STORE as SELECTION
from mm_gateway.registry import Registry, _PROVIDER_CLASSES
from mm_gateway.schemas.management import ManagementConfig, REDACTED

log = get_logger("management")
_SECRET = re.compile(r"(^key$|api.?key|token|secret|password|passwd|authorization|cookie|credential|private.?key)", re.I)


def _redact(value: Any, name: str = "", *, header: bool = False) -> Any:
    if value is None or value == "":
        return value
    if header or (_SECRET.search(name) and name not in {"keys", "credentials"}):
        return REDACTED
    if isinstance(value, dict):
        return {k: _redact(v, k, header=name == "headers") for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v, name) for v in value]
    if isinstance(value, str) and "://" in value:
        try:
            parsed = urlsplit(value)
        except ValueError:
            return REDACTED
        if parsed.username or parsed.password or _SECRET.search(parsed.query):
            return REDACTED
    return value


def _identity(value: dict) -> str | None:
    return value.get("name") or value.get("id") or value.get("domain") or value.get("base_url")


def _restore(value: Any, previous: Any) -> Any:
    """Resolve redaction markers by identity, including reordered account pools."""
    if value == REDACTED:
        if previous is None:
            raise GatewayError("A redacted value has no existing secret to preserve.",
                               code="invalid_config", status_code=422)
        return previous
    if isinstance(value, dict):
        old = previous if isinstance(previous, dict) else {}
        return {k: _restore(v, old.get(k)) for k, v in value.items()}
    if isinstance(value, list):
        old_list = previous if isinstance(previous, list) else []
        old_by_id = {_identity(v): v for v in old_list if isinstance(v, dict) and _identity(v)}
        return [_restore(v, old_by_id.get(_identity(v)))
                if isinstance(v, dict) and _identity(v) else _restore(v, None) for v in value]
    return value


def config_from_settings(settings: Settings) -> ManagementConfig:
    raw = dataclasses.asdict(settings)
    values = {name: raw[name] for name in ManagementConfig.model_fields}
    # Legacy YAML allows shorthand credentials and unnamed accounts. Expose stable
    # ids so preserving secrets never depends on the browser's array order.
    for backend, cfg in zip(values["backends"], settings.backends):
        backend["credentials"] = [
            {"id": _cred_id(entry, index), "api_key": _cred_api_key(entry),
             "base_url": _cred_base_url(entry),
             "extra": entry.get("extra", {}) if isinstance(entry, dict) else {}}
            for index, entry in enumerate(cfg.credentials)
        ] if cfg.credentials else []
    for proxy, cfg in zip(values["proxies"], settings.proxies):
        proxy["domain"] = cfg.host
        proxy["accounts"] = [{"id": _cred_id(entry, index),
                              "headers": {str(k): str(v) for k, v in entry.get("headers", {}).items() if v is not None}
                              if isinstance(entry, dict) else {}}
                             for index, entry in enumerate(cfg.accounts)]
    for profile in values["routing_profiles"].values():
        profile.pop("name", None)
    # YAML may parse catalogue dates into date objects. Canonical JSON values
    # also make untouched YAML configuration safe to persist during a key edit.
    validated = ManagementConfig.model_validate(values)
    return ManagementConfig.model_validate(validated.model_dump(mode="json"))


def settings_from_config(settings: Settings, config: ManagementConfig) -> Settings:
    values = config.model_dump()
    values["backends"] = [BackendConfig(**entry) for entry in values["backends"]]
    values["proxies"] = [ProxyConfig(**{k: v for k, v in entry.items() if k != "domain"})
                          for entry in values["proxies"]]
    values["keys"] = [KeyConfig(**{**entry, "budget": KeyBudget(**entry["budget"])
                                if entry["budget"] else None}) for entry in values["keys"]]
    values["routing_profiles"] = {name: RoutingProfile(name=name, **entry)
                                  for name, entry in values["routing_profiles"].items()}
    return dataclasses.replace(settings, **values)


def load_management_config(settings: Settings) -> Settings:
    if settings.management_config_path and Path(settings.management_config_path).exists():
        raw = json.loads(Path(settings.management_config_path).read_text(encoding="utf-8"))
        config = ManagementConfig.model_validate(raw)
        # A persisted overlay contains actual credentials, never UI placeholders.
        _restore(config.model_dump(), None)
        return settings_from_config(settings, config)
    return settings


def _persist(path: str, config: ManagementConfig) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(config.model_dump(), handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class ManagementService:
    def __init__(self, app):
        self.app = app
        self.started_at = time.monotonic()
        self.revision = uuid4().hex
        self._lock = asyncio.Lock()

    def configuration(self) -> dict:
        settings = self.app.state.settings
        return {"revision": self.revision,
                "persistent": bool(settings.management_config_path),
                "config": _redact(config_from_settings(settings).model_dump())}

    def _check_revision(self, if_match: str | None) -> None:
        if not if_match:
            raise GatewayError("Send the current configuration revision in If-Match.",
                               code="precondition_required", status_code=428)
        if if_match.strip('"') != self.revision:
            raise GatewayError("Configuration changed. Reload it before saving.",
                               code="precondition_failed", status_code=412)

    async def update(self, config: ManagementConfig, if_match: str | None) -> dict:
        async with self._lock:
            self._check_revision(if_match)
            return await self._apply(config)

    async def update_resource(self, section: str, identity: str, value, if_match: str | None) -> dict:
        async with self._lock:
            self._check_revision(if_match)
            config = config_from_settings(self.app.state.settings).model_dump()
            def matches(entry):
                if section == "proxies":
                    return urlsplit(entry["base_url"]).hostname == identity.lower()
                return entry["name" if section == "backends" else "id"] == identity
            entries = config[section]
            existing = next((entry for entry in entries if matches(entry)), None)
            if value is None and existing is None:
                raise GatewayError("Configuration resource not found.", code="not_found", status_code=404)
            if value is not None:
                item = value.model_dump()
                if not matches(item):
                    raise GatewayError("Resource identity must match the URL.", code="invalid_config", status_code=422)
                config[section] = [item if matches(entry) else entry for entry in entries]
                if existing is None:
                    config[section].append(item)
            else:
                config[section] = [entry for entry in entries if not matches(entry)]
            return await self._apply(ManagementConfig.model_validate(config))

    async def _apply(self, incoming: ManagementConfig) -> dict:
        settings = self.app.state.settings
        try:
            merged = _restore(incoming.model_dump(), config_from_settings(settings).model_dump())
            config = ManagementConfig.model_validate(merged)
            json.dumps(config.model_dump(), allow_nan=False)
            new_settings = settings_from_config(settings, config)
            candidate = await asyncio.to_thread(self._prepare_registry, new_settings)
        except (PydanticValidationError, ValueError, TypeError, ImportError) as exc:
            raise GatewayError("Configuration is invalid; no changes were applied.",
                               code="invalid_config", status_code=422) from exc
        except GatewayError as exc:
            if exc.code == "config_error":
                raise GatewayError("Model catalog configuration is invalid; no changes were applied.",
                                   code="invalid_config", status_code=422) from exc
            raise
        commit = asyncio.create_task(self._publish(candidate, config, settings))
        try:
            return await asyncio.shield(commit)
        except asyncio.CancelledError:
            # Once a durable write begins, finish publication under the mutation
            # lock even if the caller disconnects. Never leave disk and runtime
            # on different revisions or let a later edit race the disk thread.
            await commit
            raise

    async def _publish(self, candidate: Registry, config: ManagementConfig, settings: Settings) -> dict:
        if settings.management_config_path:
            try:
                await asyncio.to_thread(_persist, settings.management_config_path, config)
            except OSError as exc:
                raise GatewayError("Could not persist configuration; no changes were applied.",
                                   code="config_write_failed", status_code=503) from exc
        # Publish in one event-loop step, retaining the registry object shared by
        # HTTP and MCP services. Existing task monitors keep their owning client.
        registry = self.app.state.registry
        for name in ("_backends", "_configs", "_accounts", "_backend_accounts", "_proxies", "_catalog"):
            setattr(registry, name, getattr(candidate, name))
        registry.settings = candidate.settings
        self.app.state.settings = candidate.settings
        self.revision = uuid4().hex
        log.info("management_configuration_updated", revision=self.revision,
                 persistent=bool(settings.management_config_path))
        return self.configuration()

    def _prepare_registry(self, settings: Settings) -> Registry:
        previous = self.app.state.registry
        canonical_previous = settings_from_config(previous.settings, config_from_settings(previous.settings))
        reused = {cfg.name for cfg in settings.backends
                  if cfg.enabled and canonical_previous.backend(cfg.name) == cfg
                  and settings.outbound_proxy == previous.settings.outbound_proxy}
        changed = [cfg for cfg in settings.backends if cfg.name not in reused]
        if {cfg.name for cfg in settings.backends} & {proxy.host for proxy in settings.proxies}:
            raise GatewayError("Proxy domains must not collide with backend names.", code="invalid_config", status_code=422)
        for cfg in changed:
            if cfg.enabled and cfg.type not in _PROVIDER_CLASSES:
                raise GatewayError(f"Unknown backend type: {cfg.type}.", code="invalid_config", status_code=422)
        candidate = Registry(dataclasses.replace(settings, backends=changed))
        for cfg in changed:
            if cfg.enabled and cfg.configured:
                actual = candidate._backend_accounts.get(cfg.name, [])
                if set(actual) != {account[0] for account in cfg.accounts()}:
                    raise GatewayError(f"Could not initialize every account for backend {cfg.name}; no changes were applied.",
                                       code="backend_init_failed", status_code=422)
        for name in reused:
            if name in previous._backends:
                candidate._backends[name] = previous._backends[name]
                candidate._configs[name] = previous._configs[name]
            if name in previous._backend_accounts:
                candidate._backend_accounts[name] = previous._backend_accounts[name]
            candidate._accounts.update({key: val for key, val in previous._accounts.items() if key[0] == name})
        collisions = set(candidate._backends) & set(candidate._proxies)
        if collisions:
            raise GatewayError("Proxy domains must not collide with backend names.", code="invalid_config", status_code=422)
        candidate.settings = settings
        # Retain configured order: it also defines the initial routing preference.
        candidate._backends = {cfg.name: candidate._backends[cfg.name]
                               for cfg in settings.backends if cfg.name in candidate._backends}
        return candidate

    async def tasks(self) -> list[dict]:
        entries = []
        for record in await self.app.state.task_store.list_all():
            service = getattr(self.app.state, f"{record.modality}_service")
            snapshot = service._supervisor.summary(record.provider_task_id or record.task_id,
                                                    provider=record.provider)
            entries.append({"id": record.task_id, "owner_key_id": record.owner_key_id,
                            "backend": record.provider, "model": record.model,
                            "modality": record.modality, "status": snapshot["status"] if snapshot else "pending",
                            "created_at": record.created_at,
                            "completed_at": snapshot["completed_at"] if snapshot else None})
        return sorted(entries, key=lambda task: (task["created_at"], task["id"]), reverse=True)

    async def status(self) -> dict:
        settings = self.app.state.settings
        registry = self.app.state.registry
        backends = []
        for cfg in settings.backends:
            provider = registry.providers.get(cfg.name)
            backends.append({"name": cfg.name, "type": cfg.type, "enabled": cfg.enabled,
                             "configured": cfg.configured, "active": provider is not None,
                             "tags": cfg.tags, "accounts": registry._backend_accounts.get(cfg.name, []),
                             "models": {m: list(getattr(provider, f"{m}_models", []))
                                        for m in ("image", "video", "music", "audio")}})
        return {"status": "ok", "version": self.app.version,
                "uptime_seconds": time.monotonic() - self.started_at, "revision": self.revision,
                "persistent": bool(settings.management_config_path), "metrics_enabled": settings.enable_metrics,
                "backend_types": sorted(_PROVIDER_CLASSES), "backends": backends,
                "proxies": [{"domain": p.host, "enabled": p.enabled, "active": p.host in registry._proxies,
                             "accounts": [aid for aid, _ in p.enumerate_accounts()]} for p in settings.proxies],
                "keys_count": len(settings.keys), "enabled_keys_count": sum(k.enabled for k in settings.keys),
                "tasks_by_status": dict(Counter(task["status"] for task in await self.tasks()))}

    def metrics(self) -> dict:
        enabled = self.app.state.settings.enable_metrics
        values = METRICS.snapshot() if enabled else {"counters": [], "histograms": []}
        return {"collected_at": datetime.now(timezone.utc).isoformat(), "enabled": enabled,
                **values, "selection": list(SELECTION.snapshot().values()) if enabled else []}

    def usage(self, key_id: str | None = None) -> dict:
        keys = self.app.state.settings.keys
        if key_id is not None:
            keys = [key for key in keys if key.id == key_id]
        return {"object": "list", "data": [{"key_id": key.id, "enabled": key.enabled,
                                            "usage": self.app.state.ledger.usage(key)} for key in keys]}
