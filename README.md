# mm-gateway

`mm-gateway` is a Python 3.11+ gateway for image, video, music, speech synthesis, and voice cloning.
Each output modality has its own REST API, while all clients use the same
provider-neutral request and task conventions.

Provider SDK request names never appear in the public generation contract. The
gateway validates a strict set of media concepts and generation controls, then
each backend adapter translates those concepts to its native SDK or REST shape.

## Run with Docker Compose

With Docker and Docker Compose v2.24+ installed, run from the repository root:

```bash
docker compose up --build -d
```

This builds the gateway and management console, then starts the API at
`http://localhost:8000`. Open `http://localhost:8000/admin/` and sign in with
the default admin token `mm-gateway-admin`. The default generation API token
is `mm-gateway-local`; send it as `Authorization: Bearer mm-gateway-local`.
The health endpoint is `http://localhost:8000/health`.

No configuration file or provider credentials are needed to start. Add a backend
and its credentials in the console to enable generation. Console configuration
is stored in the `mm-gateway-data` Docker volume and survives container recreation
and `docker compose down`. Task snapshots remain process-local.

To customize the defaults, copy [`.env.example`](.env.example) to `.env`.
`PORT` changes the published host port; the container always listens on port
8000. The service binds to `127.0.0.1` by default. For access from other hosts,
set `BIND_ADDRESS=0.0.0.0` and replace both default tokens with private values.
Provider environment settings such as `OPENAI_API_KEY`, `GOOGLE_API_KEY`, and
`ARK_API_KEY` can also go in `.env`. Run the same startup command after editing it.

Use `docker compose logs -f` to follow logs and `docker compose down` to stop.

## HTTP API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/images` | Create an image task |
| `GET` | `/v1/images/{image_id}` | Retrieve an image task |
| `POST` | `/v1/videos` | Create a video task |
| `GET` | `/v1/videos/{video_id}` | Retrieve a video task |
| `POST` | `/v1/music` | Create a music task |
| `GET` | `/v1/music/{music_id}` | Retrieve a music task |
| `POST` | `/v1/audio` | Create a speech synthesis task |
| `GET` | `/v1/audio/{audio_id}` | Retrieve a speech task |
| `POST` | `/v1/voices` | Create a reusable voice clone |
| `GET` | `/v1/voices` | List voice presets and owned clones |
| `GET` | `/v1/voices/{voice_id}` | Retrieve a voice or clone task |
| `POST` | `/v1/audio/estimate` | Estimate speech cost and routing |
| `POST` | `/v1/voices/estimate` | Estimate cloning cost and routing |
| `GET` | `/v1/models?modality=image\|video\|music\|audio` | List usable models |
| `GET` | `/v1/models/limits?modality=image\|video\|music\|audio` | List usable models with documented input/output limits |
| `POST` | `/v1/images/estimate` | Estimate the cost and routing of an image request without creating a task |
| `POST` | `/v1/videos/estimate` | Estimate the cost and routing of a video request without creating a task |
| `POST` | `/v1/music/estimate` | Estimate the cost and routing of a music request without creating a task |
| `GET` | `/v1/usage` | Spend, reservations and budgets of the authenticated key |
| `GET` | `/health` | Liveness check |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/v1/management/status` | Admin runtime status and backend inventory |
| `GET`, `PUT` | `/v1/management/config` | Read redacted configuration or atomically replace it |
| `PUT`, `DELETE` | `/v1/management/backends/{name}` | Configure or remove a backend |
| `PUT`, `DELETE` | `/v1/management/keys/{key_id}` | Configure, rotate, disable, or remove a generation key |
| `PUT`, `DELETE` | `/v1/management/proxies/{domain}` | Configure or remove a proxy |
| `GET` | `/v1/management/metrics` | Structured counters, durations, and account selection health |
| `GET` | `/v1/management/tasks` | Filtered and paginated task snapshots |
| `GET` | `/v1/management/usage` | Spend and budgets across all generation keys |

### Management console

Set a separate `MANAGEMENT_API_KEY`, start the gateway, and open `/admin/`.
Docker images include the console. For a source checkout, first build it:

```bash
cd web
npm ci
npm run build
cd ..
export MANAGEMENT_API_KEY='your-admin-token'
export MANAGEMENT_CONFIG_PATH='./data/management.json'
mm-gateway
```

For Caddy subpaths such as `/gateway/admin/`, set `ROOT_PATH=/gateway` and use
`handle_path /gateway/*`. The frontend detects the prefix at runtime; see the
[Caddy deployment example](docs/management.md#caddy-and-subpath-deployments).

The console uses [`sloth-os/mm-gateway-ts`](https://github.com/sloth-os/mm-gateway-ts)
with typed management bindings generated from this gateway's OpenAPI contract.
It displays real runtime metrics, task status and key usage, and manages backend
credentials, account pools, API-key permissions and budgets, proxies and routing.
The admin token stays in browser-tab memory. Generation keys never grant admin
access, even when the generation API is open.

`MANAGEMENT_CONFIG_PATH` enables a private, atomically written configuration
overlay that is loaded on restart. Without it, changes last for the process
lifetime. Mount this path on persistent storage when using Docker. Live
management uses one gateway process; multiple workers or replicas require a
shared control plane. See [management setup and API details](docs/management.md)
for revision checks, credential preservation, and development commands.

The `/proxy/{domain}/{path}` surface (documented below) forwards any HTTP
method — and WebSocket upgrades — verbatim to a configured upstream root URL,
matched by its upstream domain.

Generation is always asynchronous. A successful `POST` returns `202 Accepted`
with the complete current task representation. The `Location` and
`Link: <...>; rel="self"` headers identify the canonical task URL. Poll that URL
until `status` is `succeeded`, `failed`, `cancelled`, or `expired`; non-terminal
responses include `Retry-After`.

Task observation is non-blocking. After a create has been accepted, the gateway
owns provider polling in a background monitor and caches the newest task
snapshot. A `GET` of the task URL reads that snapshot; it never starts
generation and never waits for a provider generation or status request to
finish. This remains true for providers that only offer a synchronous
generation call: the adapter call runs inside the monitor while clients see a
prompt `pending` or `running` response.

The monitor applies one provider-neutral lifecycle mapping:

| Provider observation | Cached public state | Monitor action |
|---|---|---|
| `pending` or `running` | same non-terminal state | poll again after the configured interval |
| `succeeded`, `failed`, `cancelled`, or `expired` | same terminal state | stop polling |
| transient poll error | last non-terminal snapshot | log/measure the error and retry |
| three consecutive poll errors | `failed` with a sanitized error | stop polling |

Background task submission, state transitions, poll failures, completion, and
shutdown cancellation are emitted as structured logs and Prometheus counters /
duration observations (`gateway_async_tasks_submitted_total`,
`gateway_async_task_poll_errors_total`, `gateway_async_tasks_finished_total`,
and `gateway_async_task_duration_seconds`). Background monitors are cancelled
during graceful gateway shutdown; the default task state remains process-local,
so deployments that require restart persistence should replace it with a
durable worker and task store.

Create requests accept an optional `Idempotency-Key` header. Retrying the same
body with the same key returns the original task and does not start another
generation; reusing the key with a different body returns `409 Conflict`.
Task responses include an `ETag`. Send it back as `If-None-Match` while polling;
an unchanged representation returns `304 Not Modified` with no response body.

There are no `/async` variants, `wait` query parameters, or provider-specific
request fields: the canonical endpoints themselves are asynchronous.

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
- `routing` optionally steers auto mode: `profile` selects a server-defined
  policy such as `quality`, `fast`, or `eu` (provider and backend names are
  never accepted), `optimize` orders candidates by `balanced`, `cost`, or
  `latency`, `max_cost_usd` caps the estimated cost of the task, `fallback`
  (`none`, `same_model`, `any`) lets a pinned model fall back when it is
  retired or unavailable, and `budget` (`{scope, limit_usd}`) tracks and caps
  spend per client-chosen scope. See
  [`docs/design/auto-mode.md`](docs/design/auto-mode.md).
- `metadata` is client-owned JSON returned unchanged with the task.

`GET /v1/models` is scoped to the authenticated key, privately cacheable for 60
seconds, and supports `ETag` / `If-None-Match` revalidation. `GET
/v1/models/limits` returns the same model list, each augmented with a `limits`
object — the neutral input/output caps the auto-router reasons about and that a
client can consult when crafting a prompt for a specific model (accepted input
modalities, max prompt length, max output count, supported sizes/durations, and
per-role support flags such as image-to-image, first-frame, or lyrics). Unknown
models fall back to a permissive entry with no documented constraint. Limits
also carry `supports_audio_output` (video models that render sound), `max_shots`
(multi-shot models), `supports_upscale` / `supports_frame_interpolation` (video
enhancement models), `supports_segmentation` (models returning a subject matte),
`supports_performance` (models animating a character from a driving performance video), and the
model lifecycle (`lifecycle`, `deprecated_on`, `retired_on`, `replacement`):
retired models are omitted from `GET /v1/models` and never auto-routed, but
stay listed here so clients can migrate pinned ids.

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

### Speech and voice cloning

`POST /v1/audio` synthesizes the text parts in order. Omit `model` to route
by the speech limits. OpenAI, ElevenLabs, MiniMax and Azure speech adapters are
available; the existing music endpoints keep their music-generation semantics.

```json
{
  "input": [{"type": "text", "text": "Welcome to our application."}],
  "parameters": {"voice": "default", "file_format": "mp3", "speed": 1.0}
}
```

`GET /v1/voices` lists gateway voice ids. `default` is a portable preset;
operators can add named presets in each backend's `extra.voice_presets`
mapping. A preset selects a voice on the chosen backend. Speech parameters also
include `instructions`, ISO `language`, `sample_rate_hz`, `bitrate_kbps`, `seed`,
and `delivery`. Unsupported controls and encoding combinations exclude a
candidate before any upstream generation. `inline` is the default delivery;
MiniMax also supports `remote`. Instructions are separate from the spoken text.

Azure Text to Speech uses the official `azure-cognitiveservices-speech` SDK.
Configure a Speech resource key and region, then select `azure-tts` (or the
`gateway-audio-azure` alias):

```yaml
backends:
  - name: azure-speech
    type: azure
    api_key: ${AZURE_AUDIO_API_KEY}
    tags: [speech]
    extra:
      region: ${AZURE_AUDIO_REGION}
      voice_presets:
        default: en-US-AvaMultilingualNeural
        narrator: en-US-JennyNeural
```

Azure supports inline MP3, WAV, raw 16-bit mono PCM, and Ogg Opus. Speed is
0.5–2; `language` accepts locales such as `fr-FR`. Switching a voice to another
locale requires multilingual support.
For a custom endpoint, set `base_url` to its full SDK endpoint URL instead of
`extra.region`. Existing named credential accounts can each override the region
or endpoint. Azure speech does not expose cloning through `/v1/voices`.
Environment-only configuration accepts `AZURE_AUDIO_API_KEY` and
`AZURE_AUDIO_REGION`, or the SDK quickstart's `SPEECH_KEY` and `SPEECH_REGION`.
Set `DEFAULT_AUDIO_PROVIDER=azure-audio` to make it the default backend.

Clone an authorized speaker with `POST /v1/voices`:

```json
{
  "input": [{"type": "audio", "uri": "https://assets.example/speaker.wav"}],
  "parameters": {"name": "Application narrator"},
  "consent": {"granted": true},
  "metadata": {"speaker": "narrator"}
}
```

Poll the returned `Location`. Once the clone succeeds and
`verification_required` is false, pass its `voice_...` id as
`parameters.voice` in `/v1/audio`. The clone belongs to its creating gateway
key. Reuse and estimates check ownership and bind routing to the backend and
credential account that created it, including when fallback is requested.
Preset aliases may select different speakers across providers; a cloned voice
always selects the recorded speaker on its owning account.

OpenAI custom voices require an eligible upstream account and
`extra.voice_cloning_enabled: true`. For that route, also supply
`consent.recording_uri` and `consent.language` (for example `en`); the recording
must use the provider's published consent phrase. The adapter creates the
upstream consent resource and then the voice. ElevenLabs supports multiple
samples and MiniMax one sample; neither accepts a separate consent recording
through this clone operation. The caller's `granted: true` attestation is
required for every provider. Provider sample duration, format and account
requirements are documented in [the audio design and research](docs/design/audio.md).

Both creates support `Idempotency-Key`, cached non-blocking reads, and ETags.
Audio and clone resources share the configured process-local task store and
monitor; they do not survive a gateway restart with the default implementation.
Speech estimates use configured `price.per_character` and/or `per_request`;
cloning uses `price.per_clone`. Unknown prices remain unknown, and budget or
cost-capped requests exclude unpriced operations unless the configured unpriced
budget policy permits them. See [the audio design](docs/design/audio.md)
for configuration and provider mappings.

### Task resource

All generation APIs use the same lifecycle and field names. The `object` value and
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
  "usage": {"output_count": 1, "duration_seconds": 8, "cost": 3.2, "cost_source": "estimate", "currency": "USD"},
  "routing": {"requested_model": "gateway-video-pro", "fallback": false, "attempts": 1, "optimize": "balanced", "estimated_cost": 3.2},
  "metadata": {"job_id": "job-123"},
  "created_at": "2026-08-11T12:00:00Z",
  "completed_at": "2026-08-11T12:00:20Z",
  "links": {"self": "https://gateway.example/v1/videos/vid_01HZX4J3K7NQ8X2V9Y6R5W4T3P"}
}
```

`usage.cost` is the provider-reported cost when the provider returns one, else
the gateway's estimate from its price catalogue (`cost_source` says which).
`routing` reports how auto mode served the task: the requested model, whether a
fallback happened and why, the number of attempts, the estimate, and the budget
scope's state.

Every output contains one `uri`; inline results are base64 data URIs. Image
outputs may also contain `mime_type` and `revised_prompt`, video outputs may
contain `cover_uri` and `mime_type`, and music outputs may contain `mime_type`. Speech outputs also carry available
sample rate, channel count, and duration in seconds.
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

The proxy's HTTP forwarder and WebSocket bridge route their **outbound** traffic
to the upstream through the same `outbound_proxy` knob as the backend providers
(global default, overridable per proxy); see
[Outbound proxy](#outbound-proxy). A misconfigured SOCKS proxy (e.g. the
`python-socks` extra missing for a WS bridge) rejects the upgrade with `4503`
before any upstream connect.

## Authentication

Generation and model-listing endpoints use `Authorization: Bearer <token>`.
Keys control which configured deployment targets are usable and can set a
default target or profile per modality. Task resources are owned by the key
that created them; another key cannot retrieve a task even when both keys can
use the same underlying generation service. Health and metrics are open.

## MCP

Set `mcp.enabled: true` to expose the same contract through fifteen MCP tools:
`list_models`, `list_model_limits`, `create_image`, `get_image`, `create_video`,
`get_video`, `create_music`, `get_music`, `create_audio`, `get_audio`,
`create_voice`, `get_voice`, `list_voices`, `estimate_cost`, and `get_usage`. `model` is optional on the create
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

Auto mode is configured in three optional sections: `routing` (the default
`optimize` mode and named `profiles`), `catalog.models` (per-model overrides of
lifecycle dates, `supports_audio_output`, and prices), and `budget`
(`allow_unpriced`). A key may carry `budget: {limit_usd, period, scopes_limit_usd}`
(`period` is `day`, `month`, or `total`). Over-budget creates fail with `402
budget_exceeded`; requests above `max_cost_usd` with `422 cost_limit_exceeded`;
retired pinned models with `410 model_retired`. See
[`docs/design/auto-mode.md`](docs/design/auto-mode.md) for the full pipeline.

```yaml
routing:
  default_optimize: balanced
  profiles:
    cheap: {optimize: cost}
catalog:
  models:
    my-self-hosted-model: {supports_audio_output: true, price: {per_second: 0.01}}
keys:
  - id: application
    key: ${GATEWAY_API_KEY}
    budget: {limit_usd: 500, period: month}
```

The `proxies:` section configures the general pass-through proxy surface
described under [General pass-through proxy](#general-pass-through-proxy) above.

The bundled task store is process-local. Production deployments with multiple
workers or replicas must use a shared durable task store so gateway task IDs,
ownership checks, and idempotency keys remain valid across instances. A custom
store can be injected with `create_app(settings, task_store=...)`.

Supported adapter types are `openai`, `google`, `vertex`, `xai`, `volcengine`,
`flux`, `openrouter`, `dashscope`, `stability`, `elevenlabs`, `minimax`, `azure`,
`udioapi`, `mureka`, and `acestep`. See
[`docs/providers/reference.md`](docs/providers/reference.md) for backend wire
details and [`examples/mm-gateway.yaml`](examples/mm-gateway.yaml) for a larger
configuration.

If no YAML file exists, environment-based backend configuration is also
available. The split variables are `<PROVIDER>_IMAGE_*`,
`<PROVIDER>_VIDEO_*`, `<PROVIDER>_MUSIC_*`, and `<PROVIDER>_AUDIO_*`, each with `API_KEY`, `BASE_URL`,
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

### Outbound proxy

Every backend provider's outbound SDK/httpx traffic **and** every pass-through
proxy's HTTP forwarder and WebSocket bridge can be routed through an HTTP or
SOCKS5 proxy. Azure's native Speech SDK supports HTTP proxies only (on Linux
and Windows); unsupported proxy schemes are rejected when its backend is built.
Set the global default once, then override it per backend or per
proxy:

```yaml
outbound_proxy: ${OUTBOUND_PROXY:}        # global default for all backends + proxies

backends:
  - name: openai-default
    type: openai
    api_key: ${OPENAI_API_KEY}
    outbound_proxy: http://proxy:3128    # overrides the global for this backend only

proxies:
  - base_url: https://api.openai.com
    outbound_proxy: socks5://proxy:1080  # overrides the global for this proxy's
                                          # forwarder + WebSocket bridge
    accounts: [...]
```

A backend may also nest the value under `extra.outbound_proxy`; a top-level
`outbound_proxy` and `extra.outbound_proxy` are equivalent, and an explicit
`extra.outbound_proxy` wins when both are set. Resolution order is **per-target
override (backend `outbound_proxy` / proxy `outbound_proxy`) → global
`outbound_proxy`**, and the registry bakes the effective URL onto each backend
and proxy at startup, so the adapters and the proxy layer read one authoritative
value.

When `outbound_proxy` is unset (and no per-target override is set), providers and
the proxy layer connect directly: httpx-based clients keep httpx's default
`trust_env=True`, so ambient `HTTP_PROXY`/`HTTPS_PROXY` still apply; the
WebSocket bridge honours ambient `ALL_PROXY`/`HTTPS_PROXY`; and the dashscope
SDK falls back to its shared env-honouring session. An explicit `outbound_proxy`
always wins where it is set.

HTTP (CONNECT) proxies need no extra dependency on the httpx-based providers, the
HTTP forwarder, or the WebSocket bridge (built in). SOCKS5 needs a transport shim
per client. The dashscope SDK speaks aiohttp internally, and an explicit
`outbound_proxy` on a dashscope backend — of **any** scheme, HTTP or SOCKS — is
routed through an `aiohttp_socks` `ProxyConnector`, so it needs `aiohttp-socks`
even for an HTTP proxy. All three shims are bundled in the optional `[socks]`
extra:

```bash
pip install -e ".[socks]"
```

Without it, a proxy that needs a missing shim raises a clear
`ProviderRequestError` naming the package — `socksio` for the httpx-based
providers and the HTTP forwarder (SOCKS only), `python-socks` for the WebSocket
bridge (SOCKS only), and `aiohttp-socks` for the dashscope async path (any
explicit proxy) — rather than failing opaquely at handshake time. The `[socks]`
extra is not in the base dependencies; a Docker deployment that uses SOCKS (or a
dashscope HTTP proxy) must add it (the base image installs only
`[project.dependencies]`).

In the env-var layout (no YAML), set the global `OUTBOUND_PROXY`, a per-backend
`<PROVIDER>_OUTBOUND_PROXY` override (e.g. `OPENAI_OUTBOUND_PROXY`,
`VERTEX_OUTBOUND_PROXY`), and `PROXY_OUTBOUND_PROXY` for the pass-through
proxy — the last falls back to `OUTBOUND_PROXY` when unset.

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
[`docs/openapi.json`](docs/openapi.json) for the generated API specification. The
same spec drives three generated client SDKs, republished automatically whenever
routes or schemas change:

| Language | Repo | Install |
|---|---|---|
| Go | [`sloth-os/mm-gateway-go`](https://github.com/sloth-os/mm-gateway-go) | `go get github.com/sloth-os/mm-gateway-go` |
| Python | [`sloth-os/mm-gateway-py`](https://github.com/sloth-os/mm-gateway-py) | `pip install git+https://github.com/sloth-os/mm-gateway-py.git` |
| TypeScript | [`sloth-os/mm-gateway-ts`](https://github.com/sloth-os/mm-gateway-ts) | `npm install @sloth-os/mm-gateway-ts` |

Each SDK is regenerated from the published OpenAPI spec by the
`openapi.yml` workflow (via `openapi-generator-cli`), verified to compile, and
pushed to its repo on every merge to `main`.

The implementation path is:

```text
REST or MCP -> public schema -> canonical translator -> service/registry
            -> selected adapter -> provider SDK or REST API
```

The design rules and capability mapping are documented in
[`docs/design/unification.md`](docs/design/unification.md).
