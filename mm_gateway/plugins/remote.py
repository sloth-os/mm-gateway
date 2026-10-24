"""Fetch GitHub source without executing installers, hooks or dependency scripts."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

from mm_gateway.plugins.loader import PluginLoadError, inspect_package
from mm_gateway.schemas.plugins import GitHubPluginInstall, PluginManifest, PluginSource


async def _git(directory: Path, *args: str) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(
        GIT_TERMINAL_PROMPT="0",
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_LFS_SKIP_SMUDGE="1",
    )
    try:
        process = await asyncio.create_subprocess_exec(
            "git",
            "-c",
            "core.hooksPath=" + os.devnull,
            "-c",
            "protocol.file.allow=never",
            "-c",
            "protocol.ext.allow=never",
            *args,
            cwd=directory,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        raise PluginLoadError("Git is required to install GitHub plugins.") from None
    try:
        async with asyncio.timeout(60):
            stdout, _ = await process.communicate()
    except BaseException:
        if process.returncode is None:
            process.kill()
        await process.communicate()
        raise
    if process.returncode:
        raise PluginLoadError(
            "GitHub fetch failed; check the public repository URL and ref."
        )
    return stdout.decode().strip()


async def fetch_github(
    spec: GitHubPluginInstall, target: Path
) -> tuple[PluginSource, PluginManifest]:
    """Export a validated package to target and return its exact provenance."""
    with tempfile.TemporaryDirectory(prefix="mm-gateway-github-") as temporary:
        repo = Path(temporary)
        await _git(repo, "init", "--quiet")
        await _git(repo, "remote", "add", "origin", spec.repository)
        await _git(
            repo, "fetch", "--quiet", "--depth=1", "--no-tags", "origin", spec.ref
        )
        commit = await _git(repo, "rev-parse", "FETCH_HEAD")
        # Do not initialize submodules or run package managers.
        await _git(repo, "checkout", "--quiet", "--detach", "FETCH_HEAD")
        package = repo / spec.subdirectory
        if not package.resolve().is_relative_to(repo.resolve()):
            raise PluginLoadError(
                "Plugin subdirectory must stay inside the repository."
            )
        manifest, _, files = await asyncio.to_thread(inspect_package, package)
        target.mkdir(parents=True, exist_ok=False)
        for name, content in files:
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        return (
            PluginSource(
                kind="github",
                path=str(target),
                repository=spec.repository,
                ref=spec.ref,
                commit=commit,
                subdirectory=spec.subdirectory,
            ),
            manifest,
        )
