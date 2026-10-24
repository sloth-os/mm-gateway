"""Contributor scaffolding and package checks: mm-gateway-plugin new/check."""

import argparse
import asyncio
import json
import os
import re
from pathlib import Path

from mm_gateway.plugins.loader import Candidate, inspect_package, load_plugin
from mm_gateway.schemas.plugins import ID_PATTERN, PluginOptions, PluginSource

TEMPLATE = """from mm_gateway.plugins import HookContext, PluginRequest, PluginResponse


class Plugin:
    def __init__(self, config):
        self.config = config

    async def on_request(self, request: PluginRequest, context: HookContext):
        context.state["path"] = request.path
        # Inspect or change headers, query_string, and body. For JSON:
        # payload = request.json()
        # request.set_json(payload)
        # Return PluginResponse.from_json({...}, 403) to reject a request.

    async def on_response(self, response: PluginResponse, context: HookContext):
        response.headers["x-plugin"] = self.config.get("label", context.plugin_id)
        if response.body is not None:
            payload = response.json()
            # Edit the JSON response before it reaches the client.
            response.set_json(payload)


def create_plugin(config):
    return Plugin(config)
"""


async def check(directory: Path):
    manifest, digest, _ = inspect_package(directory)
    candidate = Candidate(
        directory, manifest, digest, PluginSource(kind="local", path=str(directory))
    )
    loaded = await load_plugin(candidate, PluginOptions())
    await loaded.aclose()
    print(f"OK: {manifest.id} {manifest.version} (API {manifest.api_version})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    new = commands.add_parser("new", help="Create a contributor plugin package")
    new.add_argument("id")
    new.add_argument("--directory", default=os.environ.get("PLUGINS_DIR", "plugins"))
    validate = commands.add_parser(
        "check", help="Load a trusted package and verify its manifest and async hooks"
    )
    validate.add_argument("path", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "check":
            asyncio.run(check(args.path))
            return
        if not re.fullmatch(ID_PATTERN, args.id):
            parser.error(
                "id must start with a lowercase letter and contain up to 64 lowercase letters, digits, _ or -"
            )
        directory = Path(args.directory) / args.id
        directory.mkdir(parents=True, exist_ok=False)
        manifest = {
            "id": args.id,
            "name": args.id.replace("-", " ").title(),
            "version": "0.1.0",
            "api_version": "1",
            "description": "",
            "hooks": ["on_request", "on_response"],
            "response_body": "json",
            "default_config": {"label": args.id},
        }
        (directory / "plugin.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (directory / "plugin.py").write_text(TEMPLATE)
        print(
            f"Created {directory}. Run mm-gateway-plugin check {directory}, then install it in /admin/ → Plugin Store."
        )
    except Exception as exc:  # noqa: BLE001 - contributor exception values may contain secrets
        # Package exceptions can contain credentials; report only their type.
        parser.exit(
            1,
            f"Plugin command failed ({type(exc).__name__}). Check the package and destination.\n",
        )


if __name__ == "__main__":
    main()
