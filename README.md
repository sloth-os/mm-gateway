# mm-gateway

A unified **image / video / AI gateway** written in Python 3.11+. It exposes
OpenAI- and OpenRouter-compatible front-ends and routes to multiple provider
back-ends, translating each provider's native SDK shape into one canonical
internal schema.

```
            ┌─────────────────────────── mm-gateway ───────────────────────────┐
 OpenAI     │  POST /v1/images/generations          POST /v1/videos            │
 shape  ───▶│  POST /api/v1/images          GET  /v1/videos/{id}               │
            │  POST /api/v1/videos          GET  /api/v1/videos/{id}           │
 OpenRouter │                               GET  /api/v1/videos/{id}/content   │
 shape  ───▶│       translators → unified schema → services → registry         │
            │                          → provider adapters → SDKs              │
            └──────────────────────────────────────────────────────────────────┘
   providers: openai · google · xai · volcengine(seedream+seedance) · flux · openrouter · dashscope · stability
```

## Why

Every provider SDK speaks a different dialect. `mm-gateway` collapses them into
two stable, well-documented front-end shapes so clients don't learn eight APIs,
and adding a ninth provider never touches the HTTP layer. The design rationale
(commonalities, differences, the unified spine) is in
[`docs/design/unification.md`](docs/design/unification.md); per-provider
code-verified facts are in [`docs/providers/reference.md`](docs/providers/reference.md).

## Front-end shapes

| Modality | OpenAI-compatible | OpenRouter-compatible |
|----------|-------------------|-----------------------|
| Image    | `POST /v1/images/generations` | `POST /api/v1/images` |
| Video    | `POST /v1/videos` + `GET /v1/videos/{id}` (Seedance shape) | `POST /api/v1/videos` + `GET /api/v1/videos/{id}` + `GET /api/v1/videos/{id}/content` |

Plus `GET /health`, `GET /v1/models` (and `/api/v1/models`), `GET /metrics`
(Prometheus text). Override the response shape of any endpoint with
`X-Response-Format: openai|openrouter|seedance`.

Video sync vs async: by default the create call blocks until the task completes
(`VIDEO_SYNC_DEFAULT=true`). Send `Prefer: respond-async` or `?wait=false` to
get a polling handle back immediately.

Every generation or model-listing endpoint requires a front-end API key, sent as
`Authorization: Bearer <token>`. Keys are configured in `mm-gateway.yaml` (or,
for backward compatibility, the gateway is open if no key is set).

## MCP

The gateway also exposes an **MCP** (Model Context Protocol) server over the
Streamable-HTTP transport. When the `mcp` config section is enabled, four tools
mirror the HTTP API — `list_models`, `generate_image`, `create_video`,
`get_video` — so any MCP client (an IDE, agent, or the `mcp` CLI) can drive the
gateway with the same bearer-token auth and backend routing as the HTTP routes.
A `GatewayError` raised by a tool surfaces as a structured MCP/JSON-RPC error
(code `-32000`, the gateway's `code`/`status_code` in `data`), so an MCP client
can branch on it exactly like an HTTP client branches on status.

```yaml
mcp:
  enabled: true        # default false — opt in
  path: /mcp           # default /mcp; mount the server elsewhere if you like
  session_idle_timeout: 1800  # seconds; reap abandoned stateful sessions
```

MCP depends on the `mcp` package (a core dependency of the gateway). If
`enabled: true` but the package fails to import for some reason, the gateway
starts normally and logs a warning instead of mounting the endpoint.

The `/mcp` route is a streaming ASGI passthrough to the session manager — the
long-lived GET SSE channel (server-to-client JSON-RPC notifications/requests)
flushes live as produced rather than being buffered until disconnect. Stateful
sessions are reaped after `session_idle_timeout` seconds of inactivity (default
1800s / 30 min) so a client that initialises and vanishes without sending
`DELETE` cannot accumulate sessions indefinitely.

## Configuration

Config is **YAML-driven**. The gateway looks for `mm-gateway.yaml` (or `.yml`)
in the working directory, or `/etc/mm-gateway/config.yaml`, or the path in
`MM_GATEWAY_CONFIG`. Environment-variable interpolation (`${ENV}`,
`${ENV:default}`) lets secrets stay out of the file. If no YAML file is found,
the gateway falls back to a legacy env-var layout (one backend per provider,
enabled iff its `*_API_KEY` is set, and a single open key).

```yaml
# mm-gateway.yaml
server:
  host: 0.0.0.0
  port: 8000
  log_level: INFO
video:
  sync_default: true
  max_sync_wait: 300
mcp:
  enabled: true        # expose the gateway as an MCP Streamable-HTTP server
  path: /mcp           # default /mcp
  session_idle_timeout: 1800  # seconds; reap abandoned stateful sessions

backends:
  - name: volcengine-prod
    type: volcengine
    api_key: ${ARK_API_KEY}          # interpolate from env
    tags: [prod, video-primary]
  - name: openai-default
    type: openai
    api_key: ${OPENAI_API_KEY}
    tags: [prod, image-primary]

keys:
  - id: alice
    key: ${FRONTEND_KEY_ALICE}       # the Bearer token this client sends
    allow_tags: [prod]              # may use any backend tagged "prod"
    default_video_backend: volcengine-prod
    default_image_backend: openai-default
  - id: bob
    key: ${FRONTEND_KEY_BOB}
    allow_backends: [openai-default] # pinned to one backend by name
```

### Backend / key model

* **Backend** — a named, typed provider instance (`name`, `type`, `api_key`,
  `base_url`, `tags`, `extra`). Two backends of the same `type` (e.g. a prod
  and a staging Volcengine account) are distinct instances.
* **Key** — a front-end API key with routing rules: `allow_tags` /
  `allow_backends` (hybrid: a backend is usable if its tags intersect
  `allow_tags` **or** its name is in `allow_backends`), `deny_tags` (never),
  and per-modality `default_*_tag` / `default_*_backend` for routing.

A request routes, in priority order: an explicit `X-Backend` header or
`provider.backend` body field → an explicit `X-Backend-Tag` header or
`provider.tag` → the key's per-modality default → the first usable backend.

### Supported providers

| Provider    | `type`       | Image | Video |
|-------------|--------------|:-----:|:-----:|
| OpenAI      | `openai`     | ✅ | ✅ (sora) |
| Google      | `google`     | ✅ (imagen) | ✅ (veo) |
| xAI         | `xai`        | ✅ (grok-imagine) | ✅ |
| Volcengine  | `volcengine` | ✅ (seedream) | ✅ (seedance 1.0 + 2.0) |
| FLUX        | `flux`       | ✅ | — |
| OpenRouter  | `openrouter` | ✅ | ✅ |
| DashScope   | `dashscope`  | ✅ (wanx) | ✅ (wan) |
| Stability   | `stability`  | ✅ (sd) | ✅ (svd) |

Gateway model aliases (stable across providers): `gateway-image-pro`,
`gateway-image-flux`, `gateway-image-imagen`, `gateway-video-pro` (seedance),
`gateway-video-seedance-2` (seedance 2.0), `gateway-video-veo`,
`gateway-video-sora`, … — see `mm_gateway/registry.py`.

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

# write mm-gateway.yaml (see above), then:
export OPENAI_API_KEY=sk-...
export ARK_API_KEY=...            # volcengine seedance
export FRONTEND_KEY_ALICE=...     # the token you will send as Bearer

# run
mm-gateway                       # or: python -m mm_gateway.server.app
```

### Image (OpenAI shape)

```bash
curl -s localhost:8000/v1/images/generations \
  -H 'authorization: Bearer $FRONTEND_KEY_ALICE' \
  -H 'content-type: application/json' \
  -d '{"model":"gateway-image-pro","prompt":"a cat in a spacesuit","size":"1024x1024"}'
```

### Video (Seedance shape, async then poll)

```bash
T=$(curl -s localhost:8000/v1/videos \
  -H "authorization: Bearer $FRONTEND_KEY_ALICE" \
  -H 'content-type: application/json' \
  -H 'prefer: respond-async' \
  -d '{"model":"gateway-video-pro","content":[{"type":"text","text":"a cat playing"}]}' \
  | jq -r .id)
curl -s -H "authorization: Bearer $FRONTEND_KEY_ALICE" localhost:8000/v1/videos/$T
```

### Video (OpenRouter shape)

```bash
curl -s localhost:8000/api/v1/videos \
  -H "authorization: Bearer $FRONTEND_KEY_ALICE" \
  -H 'content-type: application/json' \
  -H 'prefer: respond-async' \
  -d '{"model":"gateway-video-pro","prompt":"a cat playing"}'
```

See [`examples/client.py`](examples/client.py) for a Python client.

### MCP

With `mcp.enabled: true`, point any MCP client at the gateway's `/mcp` endpoint
and send the same `Authorization: Bearer <token>` as the HTTP routes:

```python
import httpx
from mcp.client.streamable_http import streamable_http_client
from mcp import ClientSession

# The bearer token travels on the httpx client (the SDK's transport), since
# `streamable_http_client` takes an `http_client=`, not a `headers=` kwarg.
http_client = httpx.AsyncClient(
    headers={"authorization": f"Bearer {FRONTEND_KEY_ALICE}"},
)
async with streamable_http_client(
    "http://localhost:8000/mcp", http_client=http_client,
) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()                 # the four gateway tools
        img = await session.call_tool("generate_image",
                                      {"model": "gateway-image-pro",
                                       "prompt": "a cat in a spacesuit"})
```

## Architecture

```
mm_gateway/
  schemas/        # unified canonical request/response models
  translators/    # front-end shape <-> unified schema (image, video)
  providers/      # one adapter per upstream SDK
  core/           # Provider/ImageProvider/VideoProvider ABCs + exceptions
  registry.py     # build providers from settings, resolve model -> provider
  services.py     # ImageService / VideoService: resolve, call, map errors
  tasks/          # opaque task-id -> provider store (for polling)
  observability/  # structlog logging + prometheus-text metrics
  server/         # FastAPI app + routes (image, video, meta) + auth
  config.py       # YAML-driven Settings (backends + keys)
```

The layering is **routes → translators → services → registry → providers**.
Routes never touch SDKs; providers never touch HTTP. Translators are
`O(formats + providers)` rather than `O(formats × providers)` because every
shape funnels through the one unified schema.

## Tests

```bash
pytest -q        # no network calls — a fake in-memory provider
```

A real-provider end-to-end smoke test lives at `tests/e2e/smoke.py`. It talks to
a **live** gateway and a **real** upstream provider, so unlike `pytest` it does
spend real API calls. A backend is exercised only when it is **fully configured**
— all three of its `*_API_KEY`, `*_BASE_URL`, and `*_MODEL` are set. The
`*_BASE_URL` proves the operator pointed at a real endpoint and flows into the
gateway container; `*_MODEL` pins the exact upstream model id to request (rather
than a hard-coded alias) and is also published to the gateway so the registry
serves it. The script collects every fully-configured backend, confirms the
chosen model is actually served by `GET /v1/models`, then generates one image
through the OpenAI-shape front-end and asserts real image data comes back. When
*no* backend has all three set it **exits 0** (skips), so it is safe to wire
into CI before secrets exist.

```bash
mm-gateway &                          # or: docker run -p 8000:8000 ghcr.io/<owner>/<repo>:latest
OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://api.openai.com OPENAI_MODEL=gpt-image-1 \
  GATEWAY_API_KEY=... python tests/e2e/smoke.py
# pin a single backend / model instead of collecting all configured ones:
E2E_BACKEND=volcengine E2E_IMAGE_MODEL=gateway-image-seedream python tests/e2e/smoke.py
```

## CI / Docker

`.github/workflows/ci.yml` runs on every push, PR, and `v*` tag, and on manual
dispatch:

1. **Lint & unit tests** — installs the package with dev deps, import-checks
   every provider SDK, then runs `pytest`.
2. **Build & publish Docker image** — multi-stage build from `Dockerfile`,
   pushes to **GHCR** (`ghcr.io/<owner>/<repo>`) and Docker Hub
   (`<owner>/<repo>`). Tags: `latest` (on `main`), branch name, semver
   `{{version}}/{{major}}/{{major}}.{{minor}}` (on `v*` tags), and `sha-<short>`.
   The built image is also saved as an artifact for the e2e job.
3. **E2E (real provider)** — loads the *just-built* image, starts it with the
   provider `*_API_KEY`, `*_BASE_URL`, and `*_MODEL` secrets (plus
   `GATEWAY_API_KEY`) passed straight through to the container — whatever
   secrets you configure in GitHub flow into the image 1:1, no hard-coding —
   and runs `tests/e2e/smoke.py`. This job **only runs** when at least one
   provider is fully configured (all three of `*_API_KEY` + `*_BASE_URL` +
   `*_MODEL` set) *or* the manual `run-e2e` flag is checked, so the workflow
   stays green before secrets exist.

Secrets to set in **Settings → Secrets and variables → Actions** (all optional;
the e2e runs for a provider only when its `*_API_KEY` + `*_BASE_URL` + `*_MODEL`
are all set):

| Secret | Provider | Notes |
|--------|----------|-------|
| `OPENAI_API_KEY` | OpenAI | image (gpt-image) + video (sora) |
| `OPENAI_BASE_URL` | OpenAI | upstream endpoint, flows into the container |
| `OPENAI_MODEL` | OpenAI | model id to request + serve via the registry |
| `ARK_API_KEY` | Volcengine | seedream image + seedance video |
| `ARK_BASE_URL` / `ARK_MODEL` | Volcengine | endpoint + model id |
| `GOOGLE_API_KEY` | Google | imagen + veo |
| `GOOGLE_BASE_URL` / `GOOGLE_MODEL` | Google | endpoint + model id |
| `XAI_API_KEY` | xAI | grok-imagine |
| `XAI_BASE_URL` / `XAI_MODEL` | xAI | endpoint + model id |
| `RUNAPI_API_KEY` | FLUX | image only (key env is `RUNAPI_*`) |
| `FLUX_BASE_URL` / `FLUX_MODEL` | FLUX | endpoint + model id (env is `FLUX_*`) |
| `DASHSCOPE_API_KEY` | DashScope | wanx + wan |
| `DASHSCOPE_BASE_URL` / `DASHSCOPE_MODEL` | DashScope | endpoint + model id |
| `STABILITY_API_KEY` | Stability | sd + svd |
| `STABILITY_BASE_URL` / `STABILITY_MODEL` | Stability | endpoint + model id |
| `OPENROUTER_API_KEY` | OpenRouter | router (no first-class image alias) |
| `OPENROUTER_BASE_URL` / `OPENROUTER_MODEL` | OpenRouter | endpoint + model id |
| `GATEWAY_API_KEY` | — | front-end Bearer token; if unset the gateway is open | |

`GITHUB_TOKEN` (for pushing to GHCR) is provided automatically — no setup needed.

## License

Apache-2.0.
