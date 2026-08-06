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

## Provider configuration

All config is environment-driven; a provider is enabled iff its API key is set.

| Provider    | Key env var        | Image | Video |
|-------------|--------------------|:-----:|:-----:|
| OpenAI      | `OPENAI_API_KEY`   | ✅ | ✅ (sora) |
| Google      | `GOOGLE_API_KEY`   | ✅ (imagen) | ✅ (veo) |
| xAI         | `XAI_API_KEY`      | ✅ (grok-imagine) | ✅ |
| Volcengine  | `ARK_API_KEY`      | ✅ (seedream) | ✅ (seedance 1.0 + 2.0) |
| FLUX        | `RUNAPI_API_KEY`   | ✅ | — |
| OpenRouter  | `OPENROUTER_API_KEY` | ✅ | ✅ |
| DashScope   | `DASHSCOPE_API_KEY`| ✅ (wanx) | ✅ (wan) |
| Stability   | `STABILITY_API_KEY`| ✅ (sd) | ✅ (svd) |

Gateway model aliases (stable across providers): `gateway-image-pro`,
`gateway-image-flux`, `gateway-image-imagen`, `gateway-video-pro` (seedance),
`gateway-video-seedance-2` (seedance 2.0), `gateway-video-veo`,
`gateway-video-sora`, … — see `mm_gateway/registry.py`.

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

# enable the providers you have keys for
export OPENAI_API_KEY=sk-...
export ARK_API_KEY=...            # volcengine seedance

# run
mm-gateway                       # or: python -m mm_gateway.server.app
```

### Image (OpenAI shape)

```bash
curl -s localhost:8000/v1/images/generations \
  -H 'content-type: application/json' \
  -d '{"model":"gateway-image-pro","prompt":"a cat in a spacesuit","size":"1024x1024"}'
```

### Video (Seedance shape, async then poll)

```bash
T=$(curl -s localhost:8000/v1/videos \
  -H 'content-type: application/json' \
  -H 'prefer: respond-async' \
  -d '{"model":"gateway-video-pro","content":[{"type":"text","text":"a cat playing"}]}' \
  | jq -r .id)
curl -s localhost:8000/v1/videos/$T
```

### Video (OpenRouter shape)

```bash
curl -s localhost:8000/api/v1/videos \
  -H 'content-type: application/json' \
  -H 'prefer: respond-async' \
  -d '{"model":"gateway-video-pro","prompt":"a cat playing"}'
```

See [`examples/client.py`](examples/client.py) for a Python client.

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
  server/         # FastAPI app + routes (image, video, meta)
  config.py       # env-driven Settings
```

The layering is **routes → translators → services → registry → providers**.
Routes never touch SDKs; providers never touch HTTP. Translators are
`O(formats + providers)` rather than `O(formats × providers)` because every
shape funnels through the one unified schema.

## Tests

```bash
pytest -q        # no network calls — a fake in-memory provider
```

## License

Apache-2.0.
