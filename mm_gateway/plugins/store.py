"""Administrative plugin catalog, durable installs and atomic runtime swaps."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from mm_gateway.config import Settings
from mm_gateway.core.exceptions import GatewayError
from mm_gateway.management import _redact, _restore
from mm_gateway.observability.logging import get_logger
from mm_gateway.plugins.loader import (
    Candidate,
    LoadedPlugin,
    PluginLoadError,
    discover,
    inspect_package,
    load_plugin,
)
from mm_gateway.plugins.remote import fetch_github
from mm_gateway.plugins.runtime import PluginExtensions
from mm_gateway.schemas.plugins import (
    ID_PATTERN,
    GitHubPluginInstall,
    PluginOptions,
    PluginRecord,
)
from mm_gateway.secrets import SecretStore, atomic_write

log = get_logger("plugins")


def _load_error(exc: Exception) -> str:
    return (
        str(exc)
        if isinstance(exc, PluginLoadError)
        else f"Plugin load failed ({type(exc).__name__})."
    )


class PluginManager(PluginExtensions):
    def __init__(self, settings: Settings):
        self.settings = settings
        if settings.plugins_max_body_bytes <= 0 or settings.plugins_poll_interval <= 0:
            raise ValueError("Plugin body limit and poll interval must be positive.")
        self.directory = Path(settings.plugins_dir).absolute()
        # Share the operator's existing durable volume by default.
        self.state_path = (
            Path(settings.plugins_state_path)
            if settings.plugins_state_path
            else Path(settings.management_config_path).with_name("plugins.json")
            if settings.management_config_path
            else None
        )
        self.install_directory = (
            (self.state_path.parent if self.state_path else Path(".mm-gateway"))
            / "installed-plugins"
        ).absolute()
        self.revision = uuid4().hex
        self._lock = asyncio.Lock()
        self.runtime_lock = asyncio.Lock()
        self.app = None
        self._records: dict[str, PluginRecord] = {}
        self._catalog: dict[str, Candidate] = {}
        self._discovery_errors: list[dict] = []
        self._errors: dict[str, str] = {}
        self._attempted: dict[str, str] = {}
        self._active: dict[str, LoadedPlugin] = {}
        self._retired: set[LoadedPlugin] = set()
        self._watcher: asyncio.Task | None = None
        if self.state_path and self.state_path.exists():
            raw = SecretStore(str(self.state_path)).decrypt(
                json.loads(self.state_path.read_text())
            )
            if not isinstance(raw, dict) or not isinstance(raw.get("plugins"), dict):
                raise ValueError("Invalid plugin state file.")
            for plugin_id, record in raw["plugins"].items():
                if not re.fullmatch(ID_PATTERN, plugin_id):
                    raise ValueError("Invalid plugin id in state file.")
                self._records[plugin_id] = PluginRecord.model_validate(record)

    async def _scan(self):
        roots = [
            (Path(__file__).parent / "builtin", "builtin"),
            (self.directory, "local"),
            (self.install_directory, "github"),
        ]
        self._catalog, self._discovery_errors = await asyncio.to_thread(discover, roots)

    def _candidate(self, plugin_id: str) -> Candidate:
        candidate = self._catalog.get(plugin_id)
        record = self._records.get(plugin_id)
        if not candidate or (record and Path(record.source.path) != candidate.path):
            raise PluginLoadError(
                "Plugin package is missing, invalid, or has a conflicting id."
            )
        return candidate

    async def start(self):
        async with self._lock, self.runtime_lock:
            await self._scan()
            for plugin_id, record in self._records.items():
                if record.options.enabled:
                    await self._reload(plugin_id)
        if self.settings.plugins_watch:
            self._watcher = asyncio.create_task(
                self._watch(), name="gateway-plugin-watcher"
            )

    async def _watch(self):
        while True:
            await asyncio.sleep(max(0.1, self.settings.plugins_poll_interval))
            try:
                async with self._lock, self.runtime_lock:
                    await self._scan()
                    for plugin_id, record in self._records.items():
                        if not record.options.enabled:
                            continue
                        try:
                            candidate = self._candidate(plugin_id)
                        except PluginLoadError as exc:
                            self._errors[plugin_id] = str(exc)
                            continue
                        if self._attempted.get(plugin_id) != candidate.fingerprint:
                            await self._reload(plugin_id)
                        elif (
                            plugin_id in self._active
                            and self._active[plugin_id].fingerprint
                            == candidate.fingerprint
                        ):
                            self._errors.pop(plugin_id, None)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - one bad package must not stop the watcher
                log.warning("plugin_watch_failed", error_type=type(exc).__name__)

    async def _retire(self, previous: LoadedPlugin | None):
        if previous is None:
            return
        previous.retired = True
        if previous.users:
            self._retired.add(previous)
        else:
            await previous.aclose()

    async def _prepare_publication(self, plugin_id, loaded):
        active = {k: v for k, v in self._active.items() if k != plugin_id}
        if loaded:
            active[plugin_id] = loaded
        return await self.prepare_runtime(active)

    async def _publish(self, plugin_id: str, loaded: LoadedPlugin | None, runtime=None):
        previous = self._active.get(plugin_id)
        if loaded:
            self._active = {**self._active, plugin_id: loaded}
        else:
            self._active = {k: v for k, v in self._active.items() if k != plugin_id}
        self.publish_runtime(runtime)
        await self._retire(previous)

    async def _reload(self, plugin_id: str) -> bool:
        loaded = None
        try:
            candidate = self._candidate(plugin_id)
            self._attempted[plugin_id] = candidate.fingerprint
            loaded = await load_plugin(candidate, self._records[plugin_id].options)
            runtime = await self._prepare_publication(plugin_id, loaded)
        except Exception as exc:  # noqa: BLE001 - retain the previous generation for any contributor failure
            if loaded:
                await loaded.aclose()
            self._errors[plugin_id] = _load_error(exc)
            log.warning(
                "plugin_reload_failed",
                plugin_id=plugin_id,
                error_type=type(exc).__name__,
            )
            return False
        await self._publish(plugin_id, loaded, runtime)
        self._errors.pop(plugin_id, None)
        log.info("plugin_loaded", plugin_id=plugin_id, generation=loaded.generation)
        return True

    @asynccontextmanager
    async def lease(self, path: str, method: str):
        # No await between reading the snapshot and acquiring every reference.
        plugins = tuple(
            sorted(
                (p for p in self._active.values() if p.matches(path, method)),
                key=lambda p: (p.options.order, p.manifest.id),
            )
        )
        for plugin in plugins:
            plugin.users += 1
        try:
            yield plugins
        finally:
            for plugin in plugins:
                plugin.users -= 1
                if plugin.retired and not plugin.users:
                    await plugin.aclose()
                    self._retired.discard(plugin)

    async def aclose(self):
        if self._watcher:
            self._watcher.cancel()
            try:
                await self._watcher
            except asyncio.CancelledError:
                pass
            self._watcher = None
        for plugin in (*self._active.values(), *self._retired):
            await plugin.aclose()
        self._active = {}
        self._retired.clear()

    def _check_revision(self, revision: str | None):
        if not revision:
            raise GatewayError(
                "Send the current plugin store revision in If-Match.",
                code="precondition_required",
                status_code=428,
            )
        if revision.strip('"') != self.revision:
            raise GatewayError(
                "Plugin store changed. Refresh it before saving.",
                code="precondition_failed",
                status_code=412,
            )

    def _persist(self, records: dict[str, PluginRecord]):
        if self.state_path:
            raw = {
                "plugins": {k: v.model_dump(mode="json") for k, v in records.items()}
            }
            encrypted = SecretStore(str(self.state_path)).encrypt(raw)
            atomic_write(
                str(self.state_path),
                json.dumps(encrypted, indent=2, allow_nan=False) + "\n",
            )

    def _snapshot(self):
        data = []
        for plugin_id in sorted(self._catalog.keys() | self._records.keys()):
            candidate = self._catalog.get(plugin_id)
            record = self._records.get(plugin_id)
            loaded = self._active.get(plugin_id)
            manifest = (
                candidate.manifest if candidate else loaded.manifest if loaded else None
            )
            manifest_data = manifest.model_dump() if manifest else None
            if manifest_data:
                manifest_data["default_config"] = _redact(
                    manifest_data["default_config"]
                )
            # default_config can carry credentials just like installed options.
            data.append(
                {
                    "id": plugin_id,
                    "manifest": manifest_data,
                    "source": record.source if record else candidate.source,
                    "installed": record is not None,
                    "options": _redact(record.options.model_dump()) if record else None,
                    "active": loaded is not None,
                    "generation": loaded.generation if loaded else None,
                    "loaded_at": loaded.loaded_at if loaded else None,
                    "error": self._errors.get(plugin_id),
                    "calls": loaded.calls if loaded else 0,
                    "failures": loaded.failures if loaded else 0,
                }
            )
        return {
            "revision": self.revision,
            "persistent": self.state_path is not None,
            "watching": self._watcher is not None and not self._watcher.done(),
            "directory": str(self.directory),
            "data": data,
            "errors": self._discovery_errors,
        }

    async def snapshot(self):
        async with self._lock:
            await self._scan()
            return self._snapshot()

    async def put(self, plugin_id: str, options: PluginOptions, revision: str | None):
        async with self._lock, self.runtime_lock:
            self._check_revision(revision)
            await self._scan()
            previous = self._records.get(plugin_id)
            candidate = None
            try:
                candidate = self._candidate(plugin_id)
            except PluginLoadError as exc:
                if options.enabled or previous is None:
                    raise GatewayError(
                        str(exc), code="plugin_not_found", status_code=404
                    ) from None
            restored = _restore(
                options.model_dump(),
                previous.options.model_dump() if previous else None,
            )
            options = PluginOptions.model_validate(restored)
            record = PluginRecord(
                options=options,
                source=previous.source if previous else candidate.source,
            )
            loaded = None
            try:
                if options.enabled:
                    loaded = await load_plugin(candidate, options)
                runtime = await self._prepare_publication(plugin_id, loaded)
                records = {**self._records, plugin_id: record}
                self._persist(records)
            except Exception as exc:  # noqa: BLE001 - preserve the old installation on load or persistence failure
                if loaded:
                    await loaded.aclose()
                raise GatewayError(
                    _load_error(exc), code="plugin_install_failed", status_code=422
                ) from None
            self._records = records
            self.revision = uuid4().hex
            self._errors.pop(plugin_id, None)
            if candidate:
                self._attempted[plugin_id] = candidate.fingerprint
            await self._publish(plugin_id, loaded, runtime)
            return self._snapshot()

    async def delete(self, plugin_id: str, revision: str | None):
        async with self._lock, self.runtime_lock:
            self._check_revision(revision)
            previous = self._records.get(plugin_id)
            if not previous:
                raise GatewayError(
                    "Plugin is not installed.", code="plugin_not_found", status_code=404
                )
            records = {k: v for k, v in self._records.items() if k != plugin_id}
            runtime = await self._prepare_publication(plugin_id, None)
            self._persist(records)
            self._records = records
            self.revision = uuid4().hex
            await self._publish(plugin_id, None, runtime)
            self._errors.pop(plugin_id, None)
            self._attempted.pop(plugin_id, None)
            if previous.source.kind == "github":
                # Only remove our managed package directory, never local sources.
                target = self.install_directory / plugin_id
                if target.exists():
                    shutil.rmtree(target)
            await self._scan()
            return self._snapshot()

    async def reload(self, plugin_id: str, revision: str | None):
        async with self._lock, self.runtime_lock:
            self._check_revision(revision)
            if plugin_id not in self._records:
                raise GatewayError(
                    "Plugin is not installed.", code="plugin_not_found", status_code=404
                )
            await self._scan()
            if self._records[plugin_id].options.enabled and not await self._reload(
                plugin_id
            ):
                raise GatewayError(
                    self._errors[plugin_id],
                    code="plugin_reload_failed",
                    status_code=422,
                )
            self.revision = uuid4().hex
            return self._snapshot()

    async def install_github(
        self,
        spec: GitHubPluginInstall,
        revision: str | None,
        *,
        update_id: str | None = None,
    ):
        async with self._lock, self.runtime_lock:
            self._check_revision(revision)
            # Staging is on the same filesystem for atomic directory replacement.
            self.install_directory.mkdir(parents=True, exist_ok=True)
            staged = self.install_directory / (".staged-" + uuid4().hex)
            loaded = None
            backup = None
            target = None
            committed = False
            target_created = False
            try:
                source, manifest = await fetch_github(spec, staged)
                await self._scan()
                plugin_id = manifest.id
                if update_id and plugin_id != update_id:
                    raise PluginLoadError(
                        "Updated package must retain the installed plugin id."
                    )
                if not update_id and (
                    plugin_id in self._records or plugin_id in self._catalog
                ):
                    raise PluginLoadError(
                        "This plugin id already exists. Update or uninstall it first."
                    )
                target = self.install_directory / plugin_id
                source.path = str(target)
                (staged / ".gateway-source.json").write_text(
                    source.model_dump_json(indent=2) + "\n"
                )
                candidate = Candidate(
                    staged, manifest, inspect_package(staged)[1], source
                )
                if spec.options.enabled:
                    loaded = await load_plugin(candidate, spec.options)
                runtime = await self._prepare_publication(plugin_id, loaded)
                if target.exists():
                    backup = self.install_directory / (".backup-" + uuid4().hex)
                    target.rename(backup)
                staged.rename(target)
                target_created = True
                source.path = str(target)
                records = {
                    **self._records,
                    plugin_id: PluginRecord(options=spec.options, source=source),
                }
                self._persist(records)
                committed = True
                self._records = records
                self.revision = uuid4().hex
                active = loaded
                loaded = None
                await self._publish(plugin_id, active, runtime)
                self._errors.pop(plugin_id, None)
                self._attempted[plugin_id] = candidate.fingerprint
                await self._scan()
                return self._snapshot()
            except Exception as exc:  # noqa: BLE001 - roll back packages and redact contributor exception details
                raise GatewayError(
                    _load_error(exc), code="plugin_install_failed", status_code=422
                ) from None
            finally:
                if loaded:
                    await loaded.aclose()
                if staged.exists():
                    shutil.rmtree(staged)
                if not committed:
                    if target_created and target and target.exists():
                        shutil.rmtree(target)
                    if backup and backup.exists():
                        backup.rename(target)
                        backup = None
                if backup and backup.exists():
                    shutil.rmtree(backup)

    async def update_github(self, plugin_id: str, revision: str | None):
        # install_github takes the lock and rechecks revision after this lookup.
        record = self._records.get(plugin_id)
        if (
            not record
            or record.source.kind != "github"
            or not record.source.repository
            or not record.source.ref
        ):
            raise GatewayError(
                "Plugin was not installed from GitHub.",
                code="plugin_not_found",
                status_code=404,
            )
        spec = GitHubPluginInstall(
            repository=record.source.repository,
            ref=record.source.ref,
            subdirectory=record.source.subdirectory,
            options=record.options,
        )
        return await self.install_github(spec, revision, update_id=plugin_id)
