#!/usr/bin/env python3
"""Generate the mm-gateway OpenAPI (Swagger) spec as a static ``openapi.json``.

The gateway's FastAPI app auto-generates an OpenAPI 3.x schema from its routes
and Pydantic models. This script dumps that schema to a file so it can be
published as static API docs (e.g. via the ``openapi.yml`` GitHub workflow to
GitHub Pages) without running the server.

The schema is *independent of provider configuration*: the routes are registered
unconditionally in ``create_app``, and provider credentials only affect runtime
behaviour (which models ``GET /v1/models`` lists), not the route/response shapes
that make up the spec. So we build the app with an empty ``Settings()`` and no
network or secrets — the resulting ``openapi.json`` is identical to one produced
by a fully-configured gateway.

Usage::

    python scripts/generate_openapi.py [output_path]   # default: docs/openapi.json

Exits non-zero if the app fails to build or the schema is missing the expected
paths, so the publish workflow fails fast rather than shipping a broken spec.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from mm_gateway.config import Settings
from mm_gateway.server.app import create_app

# A subset of paths the gateway must expose — used as a sanity check so a
# silently-broken registry (e.g. a route module that fails to import) can never
# publish an empty or partial spec.
EXPECTED_PATHS = {
    "/health",
    "/v1/models",
    "/v1/images",
    "/v1/images/async",
    "/v1/videos",
    "/v1/music",
    "/metrics",
}


def main(argv: list[str]) -> int:
    out = Path(argv[1]) if len(argv) > 1 else Path("docs/openapi.json")

    # Empty Settings: no backends, no keys. The OpenAPI schema is the same
    # regardless of provider creds (see module docstring), so this avoids any
    # need for secrets or network when generating the spec.
    app = create_app(Settings())
    spec = app.openapi()

    paths = set(spec.get("paths", {}))
    missing = EXPECTED_PATHS - paths
    if missing:
        print(f"openapi spec is missing expected paths: {sorted(missing)}", file=sys.stderr)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote {out} ({out.stat().st_size} bytes, {len(paths)} paths, "
          f"openapi {spec.get('openapi')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
