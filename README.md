# mm-gateway

`mm-gateway` is a Python 3.11+ gateway for image, video, and music generation.
Each output modality has its own REST API, while all clients use the same
provider-neutral request and task conventions.

Provider SDK request names never appear in the public generation contract. The
gateway validates a strict set of media concepts and generation controls, then
each backend adapter translates those concepts to its native SDK or REST shape.

## HTTP API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/images` | Create an image task |
| `GET` | `/v1/images/{image_id}` | Retrieve an image task |
| `POST` | `/v1/videos` | Create a video task |
| `GET` | `/v1/videos/{video_id}` | Retrieve a video task |
| `POST` | `/v1/music` | Create a music task |
| `GET` | `/v1/music/{music_id}` | Retrieve a music task |
| `GET` | `/v1/models?modality=image\|video\|music` | List usable models |
| `GET` | `/v1/models/limits?modality=image\|video\|music` | List usable models with documented input/output limits |
| `GET` | `/health` | Liveness check |
| `GET` | `/metrics` | Prometheus metrics |

The `/proxy/{domain}/{path}` surface (documented below) forwards any HTTP
method — and WebSocket upgrades — verbatim to a configured upstream root URL,
matched by its upstream domain.

Generation is always asynchronous. A successful `POST` returns `202 Accepted`
with the complete current task representation. The `Location` and
`Link: <...>; rel="self"` headers identify the canonical task URL. Poll that URL
until `status` is `succeeded`, `failed`, `cancelled`, or `expired`; non-terminal
responses include `Retry-After`.

Create requests accept an optional `Idempotency-Key` header. Retrying the same
body with the same key returns the original task and does not start another
generation; reusing the key with a different body returns `409 Conflict`.
Task responses include an `ETag`. Send it back as `If-None-Match` while polling;
an unchanged representation returns `304 Not Modified` with no response body.

There are no `/async` variants, `wait` query parameters, or provider-specific
request fields.

Every create body uses this strict envelope:

```json
{
  "model": "gateway-image-pro",
  "input": [{"type": "text", "text": "a prompt"}],
  "parameters": {},
  "routing": {"profile": "quality"},
  "metadata": {"job_id": "job-123"}
}
```

- `model` is optional. Omit it (or set `"auto"`) and the gateway auto-routes to
  a usable backend whose documented limits fit the request's input — the text
  prompt length, input modalities (e.g. image-to-image), requested output count,
  size, and duration. When set, it is an id returned by `GET /v1/models`. If no
  configured backend's limits accommodate the request, the call fails with a
  `422` validation error before any provider is contacted.
- `input` is always a non-empty ordered list of typed parts. There is no string
  shorthand.
- `parameters` contains only provider-neutral generation controls.
- `routing.profile` optionally selects a server-defined policy such as
  `quality`, `fast`, or `eu`; provider and backend names are never accepted.
- `metadata` is client-owned JSON returned unchanged with the task.

`GET /v1/models` is scoped to the authenticated key, privately cacheable for 60
seconds, and supports `ETag` / `If-None-Match` revalidation. `GET
/v1/models/limits` returns the same model list, each augmented with a `limits`
object — the neutral input/output caps the auto-router reasons about and that a
client can consult when crafting a prompt for a specific model (accepted input
modalities, max prompt length, max output count, supported sizes/durations, and
per-role support flags such as image-to-image, first-frame, or lyrics). Unknown
models fall back to a permissive entry with no documented constraint.

Unknown envelope and parameter fields return a normalized `422` error. This is
intentional: adding a backend does not silently add its private wire options to
the public API.

Request objects are strict; response objects are additive. Clients should ignore
unknown response members so the gateway can add optional links, usage data, or
problem-detail extensions without forcing a new API version.

### Image

Image input supports one or more interleaved text and image parts. Every media
part uses one required `uri`; inline bytes use a base64 data URI.

```bash
curl -i http://localhost:8000/v1/images \
  -H "authorization: Bearer $GATEWAY_API_KEY" \
  -H 'idempotency-key: design-job-123' \
  -H 'content-type: application/json' \
  -d '{
    "model": "gateway-image-pro",
    "input": [
      {"type":"image","uri":"https://assets.example/subject.png"},
      {"type":"text","text":"place the subject in a rainy city"},
      {"type":"image","uri":"data:image/png;base64,BASE64"}
    ],
    "parameters": {
      "output_count": 2,
      "dimensions": {"width": 1280, "height": 720},
      "quality": "high",
      "delivery": "remote",
      "file_format": "png"
    }
  }'
```

Image parameters include exact pixel dimensions, quality, style, seed, guidance,
inference steps, edit strength, watermarking, delivery form, file format,
compression, and background.

### Video

Video input supports ordered text plus multiple images, audio clips, and video
clips. Roles express semantics without exposing an upstream wire format.

```json
{
  "model": "gateway-video-pro",
  "input": [
    {"type":"text","text":"cut between these references"},
    {"type":"image","uri":"https://assets.example/first.png","role":"first_frame"},
    {"type":"image","uri":"https://assets.example/style.png","role":"reference_image"},
    {"type":"audio","uri":"data:audio/wav;base64,BASE64","role":"reference_audio"},
    {"type":"video","uri":"https://assets.example/motion.mp4","role":"reference_video"}
  ],
  "parameters": {
    "duration_seconds": 8,
    "dimensions": {"width": 1280, "height": 720},
    "include_audio": true,
    "camera_motion": "auto",
    "enhance_prompt": true
  }
}
```

Images support `first_frame`, `last_frame`, and `reference_image`. Audio and
video use `reference_audio` and `reference_video`. Remote and inline media use
the same `uri` field.

### Music

Music input supports ordered descriptive text, structured lyrics, reference
images, and reference or continuation audio.

```json
{
  "model": "gateway-music-lyria",
  "input": [
    {"type":"text","text":"cinematic pop with a warm vocal"},
    {"type":"lyrics","text":"[Verse]\nUnder the city lights"},
    {"type":"image","uri":"https://assets.example/mood.jpg"},
    {"type":"audio","uri":"https://assets.example/theme.wav","role":"reference_audio"}
  ],
  "parameters": {
    "title": "City Lights",
    "duration_seconds": 30,
    "bpm": 118,
    "key": "C",
    "scale": "minor",
    "file_format": "wav",
    "sample_rate_hz": 44100,
    "bitrate_kbps": 192,
    "instrumental": false
  }
}
```

The music vocabulary also includes style, vocal language, output count,
guidance, lyric enhancement, voice, vocal gender, style strength, novelty,
reference-audio strength, inference steps, section-duration adherence, and
provenance signing. Adapters map these concepts only where the selected backend
supports them.

### Task resource

All three APIs use the same lifecycle and field names. The `object` value and
the output schema are modality-specific.

```json
{
  "id": "vid_01HZX4J3K7NQ8X2V9Y6R5W4T3P",
  "object": "video",
  "model": "gateway-video-pro",
  "status": "succeeded",
  "outputs": [
    {"uri": "https://cdn.example/video.mp4", "mime_type": "video/mp4"}
  ],
  "usage": {"output_count": 1, "duration_seconds": 8},
  "metadata": {"job_id": "job-123"},
  "created_at": "2026-08-11T12:00:00Z",
  "completed_at": "2026-08-11T12:00:20Z",
  "links": {"self": "https://gateway.example/v1/videos/vid_01HZX4J3K7NQ8X2V9Y6R5W4T3P"}
}
```

Every output contains one `uri`; inline results are base64 data URIs. Image
outputs may also contain `mime_type` and `revised_prompt`, video outputs may
contain `cover_uri` and `mime_type`, and music outputs may contain `mime_type`.
Generated lyrics are returned in the music task's `lyrics` field.

HTTP errors use RFC 9457 Problem Details with the
`application/problem+json` media type:

```json
{
  "type": "urn:mm-gateway:problem:validation_error",
  "title": "Validation Error",
  "status": 422,
  "detail": "Request validation failed.",
  "instance": "/v1/images",
  "code": "validation_error",
  "request_id": "req_01HZX4J3K7NQ8X2V9Y6R5W4T3P",
  "errors": []
}
```

`code` is the stable machine-readable extension. Provider identity and raw
upstream payloads are available in gateway logs, not public error responses.

## General pass-through proxy

`/proxy/{domain}/{path}` forwards a raw request verbatim — method, path, query
string, body, and most client headers — to an upstream root URL. A proxy is
matched by its upstream **domain** (the host of `base_url`), not a name: one
proxy per upstream service, no bespoke names to remember. It exists for upstreams
the gateway does not (and should not) translate through the media-generation
contract: a model's own REST API, a vendor's non-media endpoints, or anything
else that just needs the gateway's auth, pooling, and retry layered on top of a
raw HTTP/SSE/WebSocket hop.

```bash
curl http://localhost:8000/proxy/api.openai.com/v1/chat/completions \
  -H "authorization: Bearer $GATEWAY_API_KEY" \
  -H 'content-type: application/json' \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}'
```

The front-end caller authenticates with the same `Authorization: Bearer` key as
the generation endpoints; the key must be allowed by the proxy's `tags` (the
same hybrid tag rule as backends). The upstream account's credential is injected
by the gateway and **never** reaches the client — a caller-supplied
`Authorization` or `x-goog-api-key` is overwritten or dropped. A 401 means the
front-end key was missing/unknown, a 403 means the key is not allowed on this
proxy, and a 404 means no proxy is configured for that domain.

Every standard HTTP method is forwarded (`GET`, `POST`, `PUT`, `PATCH`, `DELETE`,
`HEAD`, `OPTIONS`). `CONNECT` is not. Event-stream (SSE) and any other responses
are streamed back to the client unchanged — they are not buffered or wrapped.

WebSocket upgrades on the same path are bridged to an upstream `ws`/`wss` URL
automatically: the upgrade is detected from the client's `Upgrade: websocket`
header, so there is no per-proxy `websocket` toggle to set. Browsers cannot set
headers on a WS upgrade, so the front-end bearer key may instead be sent as an
`access_token` (or `token`) query parameter; it authenticates the upgrade and is
dropped before forwarding to the upstream.

A proxy fronts a **pool** of upstream accounts, just like a backend with
`credentials`: the request layer ranks them by live health (success rate,
latency, rate-limit cooldown — the same store auto-routing uses) and retries
the next account on a rate-limit, timeout, or upstream 5xx. A caller's 4xx is
their own bad request and is not retried against another account. When every
account fails, the last upstream response is surfaced verbatim rather than
wrapped in a 502.

The upstream credential is not a special field — it lives directly on each
account's `headers`, exactly the header line the vendor expects. A Bearer
provider puts `authorization: Bearer <key>` there; a Google-style raw-key
provider puts `x-goog-api-key: <key>` there. One provider-agnostic mechanism
covers any upstream.

```yaml
proxies:
  - base_url: https://api.openai.com
    tags: [prod]
    timeout: 120
    # Static headers applied to every forwarded request. Per-account headers
    # shadow these.
    headers: {}
    accounts:
      - id: primary
        headers:
          authorization: Bearer ${OPENAI_PROXY_KEY_PRIMARY}
      - id: overflow
        headers:
          authorization: Bearer ${OPENAI_PROXY_KEY_OVERFLOW:}
        # Optional per-account headers override (merged over the proxy-level
        # set, account wins).
        # headers: {x-account: overflow}
```

A proxy with no credential at all is unusable and is not registered — its path
404s rather than 503-ing. A header whose `${ENV:}` is unset is dropped, so an
absent credential leaves the account empty rather than injecting a literal
`"None"` upstream.

## Authentication

Generation and model-listing endpoints use `Authorization: Bearer <token>`.
Keys control which configured deployment targets are usable and can set a
default target or profile per modality. Task resources are owned by the key
that created them; another key cannot retrieve a task even when both keys can
use the same underlying generation service. Health and metrics are open.

## MCP

Set `mcp.enabled: true` to expose the same contract through eight MCP tools:
`list_models`, `list_model_limits`, `create_image`, `get_image`, `create_video`,
`get_video`, `create_music`, and `get_music`. `model` is optional on the create
tools; omit it (or pass `auto`) to auto-route to a fitting backend.

Create tools take `model`, typed `input`, a modality-specific `parameters`
object, optional `routing`, optional `metadata`, and an optional
`idempotency_key`. They are asynchronous and return the same normalized task
resources as REST. Provider wire fields are not accepted by MCP either.

```yaml
mcp:
  enabled: true
  path: /mcp
  session_idle_timeout: 1800
```

## Configuration

The gateway loads `mm-gateway.yaml`, `/etc/mm-gateway/config.yaml`, or the path
in `MM_GATEWAY_CONFIG`. Environment interpolation supports `${ENV}` and
`${ENV:default}`.

```yaml
server:
  host: 0.0.0.0
  port: 8000

backends:
  - name: image-primary
    type: openai
    api_key: ${OPENAI_API_KEY}
    tags: [production, image, quality]
  - name: media-primary
    type: volcengine
    api_key: ${ARK_API_KEY}
    tags: [production, video, quality]
  - name: music-primary
    type: mureka
    api_key: ${MUREKA_MUSIC_API_KEY}
    tags: [production, music, quality]

keys:
  - id: application
    key: ${GATEWAY_API_KEY}
    allow_tags: [production]
    default_image_backend: image-primary
    default_video_backend: media-primary
    default_music_backend: music-primary
```

Provider credentials, endpoints, model pins, and adapter-only options belong in
operator configuration. They do not change the public request schemas.

The `proxies:` section configures the general pass-through proxy surface
described under [General pass-through proxy](#general-pass-through-proxy) above.

The bundled task store is process-local. Production deployments with multiple
workers or replicas must use a shared durable task store so gateway task IDs,
ownership checks, and idempotency keys remain valid across instances. A custom
store can be injected with `create_app(settings, task_store=...)`.

Supported adapter types are `openai`, `google`, `vertex`, `xai`, `volcengine`,
`flux`, `openrouter`, `dashscope`, `stability`, `elevenlabs`, `minimax`,
`udioapi`, `mureka`, and `acestep`. See
[`docs/providers/reference.md`](docs/providers/reference.md) for backend wire
details and [`examples/mm-gateway.yaml`](examples/mm-gateway.yaml) for a larger
configuration.

If no YAML file exists, environment-based backend configuration is also
available. The split variables are `<PROVIDER>_IMAGE_*`,
`<PROVIDER>_VIDEO_*`, and `<PROVIDER>_MUSIC_*`, each with `API_KEY`, `BASE_URL`,
and `MODEL` variants. `vertex` is the exception: it authenticates with
Application Default Credentials (a service-account JSON key), so instead of an
`API_KEY` it reads `VERTEX_CREDENTIALS_JSON` (raw key content, e.g. a CI secret)
or `VERTEX_CREDENTIALS_FILE` (key path, e.g. a YAML deployment), falling back to
ambient ADC (`GOOGLE_APPLICATION_CREDENTIALS` / metadata server) when neither is
set. The `VERTEX_PROJECT` and optional `VERTEX_LOCATION` (region) select the
endpoint — when no location is pinned the client defaults to the `global`
endpoint, which is the one Lyria 3 requires. Model and base-URL pins follow the
same split as the other providers: `VERTEX_IMAGE_MODEL` / `VERTEX_VIDEO_MODEL` /
`VERTEX_MUSIC_MODEL` and `VERTEX_IMAGE_BASE_URL` (or `VERTEX_BASE_URL`) /
`VERTEX_VIDEO_BASE_URL` / `VERTEX_MUSIC_BASE_URL`. Vertex supports all three
modalities — Imagen (image), Veo (video), and Lyria (music); the music modality
goes through the same Interactions surface as the AI Studio adapter, just
authenticated with the ADC bearer token instead of an `x-goog-api-key`.

## Development

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
pytest -q
python scripts/generate_openapi.py
mm-gateway
```

The unit suite uses in-memory providers and makes no network calls. The live
smoke client in `tests/e2e/smoke.py` exercises every fully configured modality.
See [`examples/client.py`](examples/client.py) for a runnable Python client and
[`docs/openapi.json`](docs/openapi.json) for the generated API specification.

The implementation path is:

```text
REST or MCP -> public schema -> canonical translator -> service/registry
            -> selected adapter -> provider SDK or REST API
```

The design rules and capability mapping are documented in
[`docs/design/unification.md`](docs/design/unification.md).
