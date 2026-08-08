# mm-gateway

A unified **image / video / music / AI gateway** written in Python 3.11+. It
exposes Gemini-compatible image, Seedance-compatible video, and Gemini Lyria
3-compatible music front-ends and routes to multiple provider back-ends,
translating each provider's native SDK shape into one canonical internal schema.

```
            ┌─────────────────────────── mm-gateway ───────────────────────────┐
 Gemini     │  POST /v1/images              GET  /v1/images/{id}               │
 image  ───▶│                                                               │
            │  POST /v1/videos              GET  /v1/videos/{id}               │
 Seedance   │                                                               │
 video  ───▶│  POST /v1/music               GET  /v1/music/{id}               │
            │       translators → unified schema → services → registry         │
 Gemini     │                          → provider adapters → SDKs              │
 Lyria 3    └──────────────────────────────────────────────────────────────────┘
 music  ───▶│   providers: openai · google · xai · volcengine(seedream+seedance) · flux · openrouter · dashscope · stability
            │              elevenlabs · minimax · udioapi · mureka · acestep · google(lyria)
```

## Why

Every provider SDK speaks a different dialect. `mm-gateway` collapses them into
three stable, well-documented front-end shapes so clients don't learn thirteen
APIs, and adding a fourteenth provider never touches the HTTP layer. The design
rationale (commonalities, differences, the unified spine) is in
[`docs/design/unification.md`](docs/design/unification.md); per-provider
code-verified facts are in [`docs/providers/reference.md`](docs/providers/reference.md).

## Front-end shapes

| Modality | Front-end shape |
|----------|-----------------|
| Image    | `POST /v1/images` + `GET /v1/images/{id}` (Gemini shape: `{model, input}` → `{id}`, poll `steps[].content[]` image blocks) |
| Video    | `POST /v1/videos` + `GET /v1/videos/{id}` (Seedance shape: `{model, content[]}` → `{id}`, poll `{status, content}`) |
| Music    | `POST /v1/music` + `GET /v1/music/{id}` (Gemini Lyria 3 shape: `{model, input}` → `{id}`, poll `steps[].content[]` audio/lyrics blocks) |

Plus `GET /health`, `GET /v1/models` (and `/api/v1/models`), `GET /metrics`
(Prometheus text).

Image sync vs async: by default the create call blocks until the task completes
(`IMAGE_SYNC_DEFAULT=true`). Send `Prefer: respond-async` or `?wait=false` to
get a polling handle back immediately.

Video sync vs async: the same trade-off, governed by
`VIDEO_SYNC_DEFAULT=true`. Send `Prefer: respond-async` or `?wait=false` to
get a polling handle back immediately and poll `GET /v1/videos/{id}`;
`?wait=true` forces blocking.

Music sync vs async: the same trade-off, governed by `MUSIC_SYNC_DEFAULT=true`.
Send `Prefer: respond-async` or `?wait=false` to get the interaction id back
immediately and poll `GET /v1/music/{id}`; `?wait=true` forces blocking.

Music responses: `GET /v1/music/{id}` returns the Gemini Lyria `steps[].content[]`
envelope — `{type:"audio",data|url,mime_type}` blocks plus `{type:"text",text}`
lyrics — with `output_audio` / `output_audio_url` / `output_text` helpers (full
shape in [`docs/design/unification.md`](docs/design/unification.md)). Audio
arrives inline as base64 (`audio_b64`) or as a fetchable URL (`audio_urls`); each
provider normalizes its upstream form (streamed bytes, hex, file-path bytes, or
URL). Three providers (ElevenLabs, MiniMax, Google Lyria) are synchronous
upstream and wrapped as single-process synthetic in-memory tasks, so in-flight
task state is lost on a restart; udioapi / Mureka / ACE-Step are genuine
two-phase async.

Every generation or model-listing endpoint requires a front-end API key, sent as
`Authorization: Bearer <token>`. Keys are configured in `mm-gateway.yaml` (or,
for backward compatibility, the gateway is open if no key is set).

## MCP

The gateway also exposes an **MCP** (Model Context Protocol) server over the
Streamable-HTTP transport. When the `mcp` config section is enabled, seven tools
mirror the HTTP API — `list_models`, `create_image`, `get_image`,
`create_video`, `get_video`, `create_music`, `get_music` — so any MCP client (an
IDE, agent, or the `mcp` CLI) can drive the gateway with the same bearer-token
auth and backend routing as the HTTP routes.
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
music:
  sync_default: true        # block on create until the task completes (up to max_sync_wait)
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
  - name: mureka-prod
    type: mureka
    api_key: ${MUREKA_MUSIC_API_KEY} # music-only provider
    tags: [prod, music-primary]

keys:
  - id: alice
    key: ${FRONTEND_KEY_ALICE}       # the Bearer token this client sends
    allow_tags: [prod]              # may use any backend tagged "prod"
    default_video_backend: volcengine-prod
    default_image_backend: openai-default
    default_music_backend: mureka-prod
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

| Provider    | `type`       | Image | Video | Music |
|-------------|--------------|:-----:|:-----:|:-----:|
| OpenAI      | `openai`     | ✅ | ✅ (sora) | — |
| Google      | `google`     | ✅ (imagen) | ✅ (veo) | ✅ (lyria) |
| xAI         | `xai`        | ✅ (grok-imagine) | ✅ | — |
| Volcengine  | `volcengine` | ✅ (seedream) | ✅ (seedance 1.0 + 2.0) | — |
| FLUX        | `flux`       | ✅ | — | — |
| OpenRouter  | `openrouter` | ✅ | ✅ | — |
| DashScope   | `dashscope`  | ✅ (wanx) | ✅ (wan) | — |
| Stability   | `stability`  | ✅ (sd) | ✅ (svd) | — |
| ElevenLabs  | `elevenlabs` | — | — | ✅ (music_v1, music_v2) |
| MiniMax     | `minimax`    | — | — | ✅ (music-3.0, music-2.6, music-cover) |
| udioapi.pro | `udioapi`    | — | — | ✅ (chirp-v4-0, chirp-v4-5, chirp-v4-5-plus, chirp-v5, chirp-v5-5) |
| Mureka      | `mureka`     | — | — | ✅ (mureka-song-1, mureka-song-1.5) |
| ACE-Step    | `acestep`    | — | — | ✅ (acestep-v15-turbo, acestep-v15-base, ace-step-1.5) |

Gateway model aliases (stable across providers): `gateway-image-pro`,
`gateway-image-flux`, `gateway-image-imagen`, `gateway-video-pro` (seedance),
`gateway-video-seedance-2` (seedance 2.0), `gateway-video-veo`,
`gateway-video-sora`, … — and the music aliases `gateway-music-lyria`,
`gateway-music-elevenlabs`, `gateway-music-minimax`, `gateway-music-udio`,
`gateway-music-mureka`, `gateway-music-acestep` — see `mm_gateway/registry.py`.

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

# write mm-gateway.yaml (see above), then:
export OPENAI_API_KEY=sk-...
export ARK_API_KEY=...            # volcengine seedance
export MUREKA_MUSIC_API_KEY=...   # mureka music
export FRONTEND_KEY_ALICE=...     # the token you will send as Bearer

# run
mm-gateway                       # or: python -m mm_gateway.server.app
```

### Image (Gemini shape, async then poll)

```bash
T=$(curl -s localhost:8000/v1/images \
  -H "authorization: Bearer $FRONTEND_KEY_ALICE" \
  -H 'content-type: application/json' \
  -H 'prefer: respond-async' \
  -d '{"model":"gateway-image-pro","input":"a cat in a spacesuit"}' \
  | jq -r .id)
curl -s -H "authorization: Bearer $FRONTEND_KEY_ALICE" localhost:8000/v1/images/$T
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

### Music (Lyria shape, async then poll)

```bash
T=$(curl -s localhost:8000/v1/music \
  -H "authorization: Bearer $FRONTEND_KEY_ALICE" \
  -H 'content-type: application/json' \
  -H 'prefer: respond-async' \
  -d '{"model":"gateway-music-mureka","input":"a cat playing piano, jazz"}' \
  | jq -r .id)
curl -s -H "authorization: Bearer $FRONTEND_KEY_ALICE" localhost:8000/v1/music/$T
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
        tools = await session.list_tools()                 # the seven gateway tools
        img = await session.call_tool("create_image",
                                      {"model": "gateway-image-pro",
                                       "input": "a cat in a spacesuit"})
        mus = await session.call_tool("create_music",
                                      {"model": "gateway-music-mureka",
                                       "input": "a cat playing piano, jazz",
                                       "wait": True})
```

## Architecture

```
mm_gateway/
  schemas/        # unified canonical request/response models (image, video, music)
  translators/    # front-end shape <-> unified schema (image, video, music)
  providers/      # one adapter per upstream SDK
  core/           # Provider/ImageProvider/VideoProvider/MusicProvider ABCs + exceptions
  registry.py     # build providers from settings, resolve model -> provider
  services.py     # ImageService / VideoService / MusicService: resolve, call, map errors
  tasks/          # opaque task-id -> provider store (for polling)
  observability/  # structlog logging + prometheus-text metrics
  server/         # FastAPI app + routes (image, video, music, meta) + auth
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
spend real API calls. A backend modality is exercised only when it is **fully
configured** — all three of its `*_IMAGE_API_KEY` + `*_IMAGE_BASE_URL` +
`*_IMAGE_MODEL` (image) or `*_VIDEO_*` triple (video) are set. (Music backends
register from the matching `*_MUSIC_*` triple the same way, but the e2e script
currently drives only the image and video modalities.) The `*_BASE_URL`
proves the operator pointed at a real endpoint and flows into the gateway
container; `*_MODEL` pins the exact upstream model id to request (rather than a
hard-coded alias) and is also published to the gateway so the registry serves
it. The script collects every fully-configured (backend, modality), confirms
the chosen model is actually served by `GET /v1/models`, then generates one
image through the Gemini-shape front-end (image) and one video through the
Seedance-shape front-end (create + poll), asserting real data comes back. When
*no* modality of any backend has all three set it **exits 0** (skips), so it is
safe to wire into CI before secrets exist.

```bash
mm-gateway &                          # or: docker run -p 8000:8000 ghcr.io/<owner>/<repo>:latest
OPENAI_IMAGE_API_KEY=sk-... OPENAI_IMAGE_BASE_URL=https://api.openai.com/v1 OPENAI_IMAGE_MODEL=gpt-image-1 \
OPENAI_VIDEO_API_KEY=sk-... OPENAI_VIDEO_BASE_URL=https://api.openai.com/v1 OPENAI_VIDEO_MODEL=sora-2 \
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
   pushes to **GHCR** (`ghcr.io/<owner>/<repo>`) always, and to **Docker Hub**
   (`<username>/<repo>`) only when `DOCKERHUB_USERNAME` + `DOCKERHUB_TOKEN`
   secrets are set. Tags: `latest` (on `main`), branch name, semver
   `{{version}}/{{major}}/{{major}}.{{minor}}` (on `v*` tags), and `sha-<short>`.
   The built image is also saved as an artifact for the e2e job.
3. **E2E (real provider)** — loads the *just-built* image, starts it with the
   provider `*_IMAGE_API_KEY` / `*_VIDEO_API_KEY` / `*_MUSIC_API_KEY` (secrets)
   and the matching `*_IMAGE_BASE_URL` / `*_IMAGE_MODEL` / `*_VIDEO_BASE_URL` /
   `*_VIDEO_MODEL` / `*_MUSIC_BASE_URL` / `*_MUSIC_MODEL` (variables), plus
   `GATEWAY_API_KEY`, passed straight through to the container — whatever you
   configure in GitHub flows into the image 1:1, no hard-coding — and runs
   `tests/e2e/smoke.py`. This job **only runs** when at least one provider
   modality is fully configured (all three of an `*_IMAGE_*` or `*_VIDEO_*`
   triple set) *or* the manual `run-e2e` flag is checked, so the workflow stays
   green before secrets exist. Music backends register from their `*_MUSIC_*`
   triple the same way, so a wired music provider is live in the container even
   though the e2e script currently drives image and video.

Configure these in **Settings → Secrets and variables → Actions**, under the
`ci` environment. Each provider's image, video, and music credentials are
**split**: `*_IMAGE_API_KEY` + `*_VIDEO_API_KEY` + `*_MUSIC_API_KEY` are
sensitive, so store them as **secrets**; `*_IMAGE_BASE_URL` / `*_IMAGE_MODEL` /
`*_VIDEO_BASE_URL` / `*_VIDEO_MODEL` / `*_MUSIC_BASE_URL` / `*_MUSIC_MODEL` are
not sensitive, so store them as **variables** — the workflow reads them from the
`vars.*` context, not `secrets.*`, so a `*_BASE_URL`/`*_MODEL` stored as a
*secret* is invisible to the gate and the e2e silently skips. A provider
modality runs only when **all three** of its `*_IMAGE_*` (or `*_VIDEO_*`)
triple — `*_API_KEY` (secret) + `*_BASE_URL` (variable) + `*_MODEL` (variable)
— are set. The legacy un-split `*_API_KEY` / `*_BASE_URL` / `*_MODEL` names
still work as a per-modality fallback until you migrate to the split names.
ACE-Step is self-hosted, so `ACESTEP_MUSIC_BASE_URL` is **required** (there is
no default cloud host).

**Secrets** (`*_IMAGE_API_KEY` + `*_VIDEO_API_KEY` + `*_MUSIC_API_KEY` + gateway/publish creds):

| Secret | Provider | Notes |
|--------|----------|-------|
| `OPENAI_IMAGE_API_KEY` | OpenAI | image (gpt-image) |
| `OPENAI_VIDEO_API_KEY` | OpenAI | video (sora) |
| `ARK_IMAGE_API_KEY` | Volcengine | seedream image |
| `ARK_VIDEO_API_KEY` | Volcengine | seedance video |
| `GOOGLE_IMAGE_API_KEY` | Google | imagen |
| `GOOGLE_VIDEO_API_KEY` | Google | veo |
| `XAI_IMAGE_API_KEY` | xAI | grok-imagine image |
| `XAI_VIDEO_API_KEY` | xAI | grok-imagine video |
| `FLUX_IMAGE_API_KEY` | FLUX | image only (FLUX has no video) |
| `DASHSCOPE_IMAGE_API_KEY` | DashScope | wanx image |
| `DASHSCOPE_VIDEO_API_KEY` | DashScope | wan video |
| `STABILITY_IMAGE_API_KEY` | Stability | sd image |
| `STABILITY_VIDEO_API_KEY` | Stability | svd video |
| `OPENROUTER_IMAGE_API_KEY` | OpenRouter | router image (no first-class alias) |
| `OPENROUTER_VIDEO_API_KEY` | OpenRouter | router video (no first-class alias) |
| `GOOGLE_MUSIC_API_KEY` | Google | lyria music (falls back to the un-split `GOOGLE_API_KEY`) |
| `ELEVENLABS_MUSIC_API_KEY` | ElevenLabs | music_v2 |
| `MINIMAX_MUSIC_API_KEY` | MiniMax | music-3.0 |
| `UDIOAPI_MUSIC_API_KEY` | udioapi.pro | chirp-v5 |
| `MUREKA_MUSIC_API_KEY` | Mureka | mureka-song-1 |
| `ACESTEP_MUSIC_API_KEY` | ACE-Step | ace-step-1.5 (optional at the adapter, but a backend registers only when a key is set, so omitting it disables ACE-Step) |
| `GATEWAY_API_KEY` | — | front-end Bearer token; if unset the gateway is open |
| `DOCKERHUB_USERNAME` | — | opt-in: also publish to Docker Hub as `<username>/<repo>` |
| `DOCKERHUB_TOKEN` | — | opt-in: Docker Hub access token (paired with `DOCKERHUB_USERNAME`) |

**Variables** (`*_IMAGE_*` + `*_VIDEO_*` + `*_MUSIC_*`, one triple per provider modality you wire up):

| Variable | Provider | Notes |
|----------|----------|-------|
| `OPENAI_IMAGE_BASE_URL` / `OPENAI_IMAGE_MODEL` | OpenAI | image endpoint + model id |
| `OPENAI_VIDEO_BASE_URL` / `OPENAI_VIDEO_MODEL` | OpenAI | video endpoint + model id |
| `ARK_IMAGE_BASE_URL` / `ARK_IMAGE_MODEL` | Volcengine | seedream endpoint + model id |
| `ARK_VIDEO_BASE_URL` / `ARK_VIDEO_MODEL` | Volcengine | seedance endpoint + model id |
| `GOOGLE_IMAGE_BASE_URL` / `GOOGLE_IMAGE_MODEL` | Google | imagen endpoint + model id |
| `GOOGLE_VIDEO_BASE_URL` / `GOOGLE_VIDEO_MODEL` | Google | veo endpoint + model id |
| `XAI_IMAGE_BASE_URL` / `XAI_IMAGE_MODEL` | xAI | image endpoint (with or without `/v1` — the adapter normalises) + model id |
| `XAI_VIDEO_BASE_URL` / `XAI_VIDEO_MODEL` | xAI | video endpoint + model id |
| `FLUX_IMAGE_BASE_URL` / `FLUX_IMAGE_MODEL` | FLUX | image endpoint + model id (FLUX is image-only) |
| `DASHSCOPE_IMAGE_BASE_URL` / `DASHSCOPE_IMAGE_MODEL` | DashScope | wanx endpoint + model id |
| `DASHSCOPE_VIDEO_BASE_URL` / `DASHSCOPE_VIDEO_MODEL` | DashScope | wan endpoint + model id |
| `STABILITY_IMAGE_BASE_URL` / `STABILITY_IMAGE_MODEL` | Stability | sd endpoint + model id |
| `STABILITY_VIDEO_BASE_URL` / `STABILITY_VIDEO_MODEL` | Stability | svd endpoint + model id |
| `OPENROUTER_IMAGE_BASE_URL` / `OPENROUTER_IMAGE_MODEL` | OpenRouter | image endpoint + model id |
| `OPENROUTER_VIDEO_BASE_URL` / `OPENROUTER_VIDEO_MODEL` | OpenRouter | video endpoint + model id |
| `GOOGLE_MUSIC_BASE_URL` / `GOOGLE_MUSIC_MODEL` | Google | lyria endpoint (defaults to `generativelanguage.googleapis.com`) + model id |
| `ELEVENLABS_MUSIC_BASE_URL` / `ELEVENLABS_MUSIC_MODEL` | ElevenLabs | music endpoint (SDK default if unset) + model id |
| `MINIMAX_MUSIC_BASE_URL` / `MINIMAX_MUSIC_MODEL` | MiniMax | music endpoint (defaults to `api.minimax.io`) + model id |
| `UDIOAPI_MUSIC_BASE_URL` / `UDIOAPI_MUSIC_MODEL` | udioapi.pro | music endpoint (defaults to `udioapi.pro`) + model id |
| `MUREKA_MUSIC_BASE_URL` / `MUREKA_MUSIC_MODEL` | Mureka | music endpoint (defaults to `platform.mureka.ai`) + model id |
| `ACESTEP_MUSIC_BASE_URL` / `ACESTEP_MUSIC_MODEL` | ACE-Step | **required** — self-hosted, no default host + model id |

For OpenAI (and OpenAI-compatible endpoints), the base URL should **include
`/v1`** (the SDK uses it verbatim), e.g. `https://api.openai.com/v1`.

`GITHUB_TOKEN` (for pushing to GHCR) is provided automatically — no setup needed.
Without `DOCKERHUB_*` the image is published to GHCR only, so the workflow stays
green before any secrets are configured.

## License

Apache-2.0.
