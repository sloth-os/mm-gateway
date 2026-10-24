"""Discover and load isolated generations of trusted Python plugin packages."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import inspect
import json
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any
from uuid import uuid4

from mm_gateway.core.base import Provider
from mm_gateway.observability.logging import get_logger
from mm_gateway.schemas.plugins import (
    GitHubPluginInstall,
    PluginManifest,
    PluginOptions,
    PluginSource,
)

log = get_logger("plugins")
IGNORED = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache"}
MAX_PACKAGE_BYTES = 16 * 1024 * 1024
MAX_PACKAGE_FILES = 512


class PluginLoadError(ValueError):
    pass


def package_files(directory: Path) -> list[tuple[str, bytes]]:
    """Bound package size and reject links, including links outside the repo."""
    files: list[tuple[str, bytes]] = []
    total = 0
    directories = 0

    def walk(path: Path):
        nonlocal total, directories
        directories += 1
        if directories > 1024 or len(path.relative_to(directory).parts) > 32:
            raise PluginLoadError(
                "Plugin package directory structure exceeds its limits."
            )
        if path.is_symlink():
            raise PluginLoadError("Plugin packages cannot contain symbolic links.")
        for item in sorted(path.iterdir()):
            if item.name in IGNORED or item.suffix in {".pyc", ".pyo"}:
                continue
            if item.is_symlink():
                raise PluginLoadError("Plugin packages cannot contain symbolic links.")
            if item.is_dir():
                walk(item)
            elif item.is_file():
                if (
                    item.stat().st_size + total > MAX_PACKAGE_BYTES
                    or len(files) >= MAX_PACKAGE_FILES
                ):
                    raise PluginLoadError("Plugin package exceeds 512 files or 16 MiB.")
                content = item.read_bytes()
                total += len(content)
                if total > MAX_PACKAGE_BYTES:
                    raise PluginLoadError("Plugin package exceeds 16 MiB.")
                files.append((item.relative_to(directory).as_posix(), content))
            else:
                raise PluginLoadError("Plugin packages must contain regular files.")

    walk(directory)
    return files


def inspect_package(
    directory: Path,
) -> tuple[PluginManifest, str, list[tuple[str, bytes]]]:
    files = package_files(directory)
    contents = dict(files)
    if "plugin.json" not in contents or "plugin.py" not in contents:
        raise PluginLoadError("A plugin package requires plugin.json and plugin.py.")
    try:
        manifest = PluginManifest.model_validate_json(contents["plugin.json"])
    except ValueError:
        # Validation inputs and exception strings may include config secrets.
        raise PluginLoadError(
            "Invalid plugin.json; check the plugin manifest contract."
        ) from None
    digest = hashlib.sha256()
    for name, content in files:
        digest.update(name.encode() + b"\0" + content + b"\0")
    return manifest, digest.hexdigest(), files


@dataclass
class Candidate:
    path: Path
    manifest: PluginManifest
    fingerprint: str
    source: PluginSource


def discover(roots: list[tuple[Path, str]]) -> tuple[dict[str, Candidate], list[dict]]:
    catalog: dict[str, Candidate] = {}
    errors: list[dict] = []
    for root, kind in roots:
        if not root.exists():
            continue
        try:
            directories = sorted(root.iterdir())
        except OSError:
            errors.append({"path": str(root), "error": "Cannot read plugin directory."})
            continue
        for directory in directories:
            if directory.name.startswith(".") or not directory.is_dir():
                continue
            try:
                manifest, fingerprint, _ = inspect_package(directory)
                if manifest.id in catalog:
                    raise PluginLoadError(f"Duplicate plugin id '{manifest.id}'.")
                source = PluginSource(kind=kind, path=str(directory))
                metadata = directory / ".gateway-source.json"
                if kind == "github" and metadata.exists():
                    source = PluginSource.model_validate_json(metadata.read_bytes())
                    GitHubPluginInstall(
                        repository=source.repository,
                        ref=source.ref,
                        subdirectory=source.subdirectory,
                    )
                    source.kind, source.path = "github", str(directory)
                catalog[manifest.id] = Candidate(
                    directory, manifest, fingerprint, source
                )
            except (OSError, ValueError) as exc:
                error = (
                    str(exc)
                    if isinstance(exc, PluginLoadError)
                    else "Cannot read plugin package."
                )
                errors.append({"path": str(directory), "error": error})
    return catalog, errors


@dataclass(eq=False)
class LoadedPlugin:
    manifest: PluginManifest
    options: PluginOptions
    fingerprint: str
    instance: Any
    namespace: str
    directory: tempfile.TemporaryDirectory
    generation: str = field(default_factory=lambda: uuid4().hex)
    loaded_at: float = field(default_factory=time.time)
    users: int = 0
    retired: bool = False
    closed: bool = False
    calls: int = 0
    failures: int = 0
    resources: list[Any] = field(default_factory=list)

    def matches(self, path: str, method: str) -> bool:
        return any(path.startswith(prefix) for prefix in self.manifest.paths) and (
            not self.manifest.methods or method in self.manifest.methods
        )

    async def aclose(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            for resource in self.resources:
                close_resource = getattr(resource, "aclose", None)
                if close_resource:
                    try:
                        async with asyncio.timeout(self.options.timeout_ms / 1000):
                            await close_resource()
                    except Exception as exc:  # noqa: BLE001 - resource failures must not prevent module cleanup
                        log.warning(
                            "plugin_resource_close_failed",
                            plugin_id=self.manifest.id,
                            error_type=type(exc).__name__,
                        )
            close = getattr(self.instance, "aclose", None)
            if close:
                async with asyncio.timeout(self.options.timeout_ms / 1000):
                    await close()
        except Exception as exc:  # noqa: BLE001 - always remove the namespace and source snapshot
            log.warning(
                "plugin_close_failed",
                plugin_id=self.manifest.id,
                error_type=type(exc).__name__,
            )
        finally:
            for name in list(sys.modules):
                if name == self.namespace or name.startswith(self.namespace + "."):
                    sys.modules.pop(name, None)
            self.directory.cleanup()


async def load_plugin(candidate: Candidate, options: PluginOptions) -> LoadedPlugin:
    # Copy every source/resource file into a fresh package. Unique namespaces
    # and a fresh directory also invalidate helper-module .pyc caches on reload.
    manifest, fingerprint, files = await asyncio.to_thread(
        inspect_package, candidate.path
    )
    if manifest.id != candidate.manifest.id:
        raise PluginLoadError("Plugin identity changed while loading.")
    directory = tempfile.TemporaryDirectory(prefix="mm-gateway-plugin-")
    namespace = "_mm_gateway_plugin_" + uuid4().hex
    instance = None

    def construct():
        for name, content in files:
            target = Path(directory.name) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        package = ModuleType(namespace)
        package.__path__ = [directory.name]
        sys.modules[namespace] = package
        spec = importlib.util.spec_from_file_location(
            namespace + ".plugin", Path(directory.name) / "plugin.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        factory = getattr(module, "create_plugin", None)
        if not callable(factory):
            raise PluginLoadError("plugin.py must export create_plugin(config).")
        # A fresh JSON copy prevents a plugin mutating store configuration.
        config = json.loads(
            json.dumps({**manifest.default_config, **options.config}, allow_nan=False)
        )
        return factory(config)

    try:
        construction = asyncio.create_task(asyncio.to_thread(construct))
        try:
            instance = await asyncio.shield(construction)
        except asyncio.CancelledError:
            # A Python import in a thread cannot be interrupted. Finish it
            # before removing its namespace and temporary source directory.
            try:
                instance = await construction
                if inspect.iscoroutine(instance):
                    instance.close()
                    instance = None
            except Exception as exc:  # noqa: BLE001 - discard a failed constructor after cancellation
                log.warning(
                    "plugin_constructor_failed_after_cancellation",
                    plugin_id=manifest.id,
                    error_type=type(exc).__name__,
                )
            raise
        if inspect.isawaitable(instance):
            async with asyncio.timeout(options.timeout_ms / 1000):
                instance = await instance
        for hook in manifest.hooks:
            if not inspect.iscoroutinefunction(getattr(instance, hook, None)):
                raise PluginLoadError(f"Declared hook {hook} must be an async method.")
        providers = getattr(instance, "providers", {})
        if not isinstance(providers, dict) or set(providers) != set(manifest.providers):
            raise PluginLoadError(
                "instance.providers must export exactly the provider types declared in plugin.json."
            )
        for provider in providers.values():
            if (
                not inspect.isclass(provider)
                or not issubclass(provider, Provider)
                or inspect.isabstract(provider)
            ):
                raise PluginLoadError(
                    "Exported providers must be concrete subclasses of the gateway Provider interfaces."
                )
        close = getattr(instance, "aclose", None)
        if close is not None and not inspect.iscoroutinefunction(close):
            raise PluginLoadError("aclose must be an async method.")
        return LoadedPlugin(
            manifest, options, fingerprint, instance, namespace, directory
        )
    except BaseException:
        loaded = LoadedPlugin(
            manifest, options, fingerprint, instance, namespace, directory
        )
        await loaded.aclose()
        raise
