# Provider API reference

Concrete, code-verified facts for each provider SDK installed in this repo's
venv. This is the source of truth the gateway adapters were written against.
Models/params marked *(docs)* are from official docs, not the installed SDK.

Everything below is adapter-side implementation detail. These native field
names are not accepted by the public REST or MCP parameter schemas; see
`docs/design/unification.md` for the provider-neutral contract.

Every adapter routes its outbound SDK/httpx traffic through the configured
`outbound_proxy` (HTTP or SOCKS5): a global default, overridable per backend via
`outbound_proxy` / `extra["outbound_proxy"]`, resolved at startup. HTTP proxies
need no extra dep on the httpx paths; SOCKS5 needs the optional `[socks]` extra
(`socksio` for httpx-based providers), and the dashscope (aiohttp) path routes
any explicit proxy — HTTP or SOCKS — through `aiohttp-socks`. See the README
*Outbound proxy* section.

---

## openai — `openai` v2.53.0

- **Client**: `from openai import AsyncOpenAI; AsyncOpenAI(api_key=, base_url=)` (env `OPENAI_API_KEY`, `OPENAI_BASE_URL`).
- **Sync vs async URL**: two `AsyncOpenAI` instances — image (DALL·E/GPT-Image)
  on `base_url` (the `OPENAI_IMAGE_BASE_URL` sync endpoint) and video (Sora) on
  `backend.extra["video_base_url"]` (the `OPENAI_VIDEO_BASE_URL` async
  endpoint, set by `config.py` when it differs from the image one). For the
  real api.openai.com both share one host, so the two clients collapse to the
  same base (or `None`, the SDK default) unless an operator pins them apart.
- **Image**: `client.images.generate(model, prompt, n, size, quality, response_format, style, background, output_format, user, ...)`.
  - Models: `gpt-image-1`, `gpt-image-1-mini`, `gpt-image-2`, `dall-e-2`, `dall-e-3`.
  - GPT-image models **always** return `data[].b64_json`; `dall-e-2/3` return `data[].url` (default) or `b64_json`.
  - `response.data[i].b64_json | .url`; `usage` only for gpt-image.
- **Edit**: `client.images.edit(image=<file>, mask=, model, prompt, ...)` — multipart.
- **Video (Sora)**: `client.videos.create(model, prompt, seconds="4|8|12", size, input_reference=)` → `Video{id,status,progress}`. Poll `client.videos.retrieve(id)` until `completed`/`failed`, then `client.videos.download_content(id)` → MP4 bytes.
  - Models: `sora-2`, `sora-2-pro`.

## google-genai — `google-genai` v2.19.0

- **Client**: `from google import genai; client = genai.Client(api_key=)` (env `GOOGLE_API_KEY`/`GEMINI_API_KEY`). Async via `client.aio.models.*`.
- **Sync vs async URL**: two `genai.Client` instances — image (Imagen /
  generate_content) on `base_url` (the `GOOGLE_IMAGE_BASE_URL` sync endpoint)
  and video (Veo) on `backend.extra["video_base_url"]` (the
  `GOOGLE_VIDEO_BASE_URL` async endpoint, set by `config.py` when it differs
  from the image one). For the real generativelanguage.googleapis.com both
  share one host, so the two clients collapse unless an operator pins them
  apart. Music (Lyria) stays on its own REST base `_music_base`
  (`music_base_url` → `base_url` → SDK default), independent of both.
- **Image (Imagen)**: `client.aio.models.generate_images(model, prompt, config=GenerateImagesConfig(...))`.
  - Models: `imagen-4.0-generate-001`, `imagen-3.0-generate-001`.
  - Response: `resp.generated_images[i].image.image_bytes` (bytes; **no URL** on dev API).
- **Image (Gemini native)**: `generate_content(model="gemini-2.5-flash-image", contents=, config=GenerateContentConfig(response_modalities=["IMAGE"]))`. Image bytes in `part.inline_data.data`.
- **Video (Veo)**: `client.aio.models.generate_videos(model, source=GenerateVideosSource(prompt=|image=), config=GenerateVideosConfig(...))` → long-running op. Poll `client.aio.operations.get(op)` until `op.done`. Result: `op.result.generated_videos[0].video.uri`; fetch bytes via `client.aio.files.download(file=video)`.
  - Models: `veo-2.0-generate-001`, `veo-3.0-generate-001`, `veo-3.1-generate-preview`.
- **Music (Lyria 3)**: REST `POST {music_base}/v1beta/interactions` over a per-request `httpx.AsyncClient(timeout=240.0)` — the Interactions surface. Auth = api_key in the `x-goog-api-key` header (matching the google-genai SDK). `music_base` = `backend.extra["music_base_url"]` or `backend.base_url` or `https://generativelanguage.googleapis.com`.
  - Models: `lyria-3`.
  - Body: `{model, input: parts[], response_format?, generation_config?}` (top-level fields; there is no `config` wrapper on this surface). Canonical text, image, and audio parts become native input parts. Data URLs are decoded into inline `{type, mime_type, data}` blocks; ordinary references remain URL blocks. `response_format={"type":"audio","mime_type":...}` selects the container when `audio_format` is set. `generation_config` carries `seed` plus the translated negative prompt, guidance, and output count (best-effort).
  - Synchronous — a single call returns audio inline. Wrapped as a synthetic in-memory task (id `lyria-{uuid4.hex}`) that moves `pending -> running -> succeeded` on the first poll, like ElevenLabs/MiniMax.
  - Response: `steps[].content[]` (or `model_output[].content[]`) blocks — `{type:"audio", data}` (base64) and `{type:"text", text}` (lyrics); top-level `output_audio`/`audio` and `output_text`/`text` fallbacks. `audio_media_type` derived from the requested format (`audio/wav` / `audio/mpeg` / `audio/{format}`), defaulting to `audio/wav`. `usage = MusicUsage(duration=int(request.duration))` when duration set.
  - Errors: `httpx.HTTPError` or HTTP ≥400 → `ProviderRequestError(502)`; no audio → `TaskFailedError`; unknown task id → 404.

## vertex — `google-genai` v2.19.0 (Vertex mode)

Vertex AI is the Gemini Enterprise Agent Platform surface: it exposes the
**same** Imagen/Veo/Lyria models as the AI Studio (google) adapter above,
reached through the **same** `google-genai` SDK, only the client is built with
`vertexai=True`. So this adapter reuses the google adapter's Imagen/Veo
request/response logic (`GenaiImageVideoMixin`) and the shared Lyria helpers
(`mm_gateway.providers._lyria`) verbatim; only `genai.Client` construction
differs.

- **Auth: Application Default Credentials only (no API key).** A
  service-account JSON key is resolved in this order:
  - `backend.extra["credentials_json"]` — raw JSON content (env
    `VERTEX_CREDENTIALS_JSON`, used by CI where a GitHub secret holds the file
    contents verbatim). Loaded with `google.auth.load_credentials_from_dict`.
  - `backend.extra["credentials_file"]` — filesystem path to a key JSON (env
    `VERTEX_CREDENTIALS_FILE`, used by YAML deployments). Loaded with
    `google.auth.load_credentials_from_file`.
  - Else ambient ADC — `google.auth.default(scopes=[cloud-platform])` — so a
    host that has run `gcloud auth application-default login` or runs on
    GCE/GKE/Workload Identity (or sets `GOOGLE_APPLICATION_CREDENTIALS`) works
    with no explicit key.
  - The resolved `Credentials` are passed straight to
    `genai.Client(credentials=...)`. **`api_key` is never set.** No key and no
    ambient ADC raises `ProviderNotConfiguredError("vertex")`.
- **Client**: `from google import genai; genai.Client(vertexai=True,
  credentials=, project=, location=, http_options=types.HttpOptions(...))`.
  Three instances — image (Imagen/generate_content) on `base_url` (an operator
  `VERTEX_*_BASE_URL` pin, usually unset), video (Veo) on
  `backend.extra["video_base_url"]` when it differs, and music (Lyria
  Interactions) on `backend.extra["music_base_url"]` when it differs. With
  none pinned the SDK derives the regional endpoint
  `https://{location}-aiplatform.googleapis.com` from the location.
  - `project` = `VERTEX_PROJECT` (env, in `extra["project"]`) or the SA key's
    own `project_id`.
  - `location` = `VERTEX_LOCATION` (env, in `extra["location"]`) —
    **optional**. When unset the provider defaults to `"global"`, which
    selects the `https://aiplatform.googleapis.com` endpoint (no region
    prefix). The global endpoint is the one **Lyria 3 requires** on Vertex —
    Lyria 3 only serves from the global location (regional requests return an
    internal error, see google-genai issue #2533), so defaulting to `global`
    makes the music modality work out of the box; the image/video models accept
    the global endpoint too. An operator who prefers a region (e.g.
    `us-central1`) can still pin it.
  - The injected `httpxAsyncClient` carries the backend-logging event hooks
    (curl format + masked sensitive headers), matching the google adapter.
- **Image (Imagen)** & **Video (Veo)**: identical to the google adapter
  above — `client.aio.models.generate_images(...)` /
  `client.aio.models.generate_videos(...)` + operation poll.
  - Image models: `imagen-4.0-generate-001`, `imagen-3.0-generate-001`,
    `gemini-2.5-flash-image`.
  - Video models: `veo-2.0-generate-001`, `veo-3.0-generate-001`,
    `veo-3.1-generate-preview`.
- **Music (Lyria 3)**: the SDK's Interactions surface
  (`client.aio.interactions.create(...)` → REST `POST /v1beta/interactions`),
  the same wire shape the google adapter uses, authenticated with the ADC
  bearer token the SDK injects (not `x-goog-api-key`). Synchronous — a single
  call returns the audio inline — so, like the google adapter, it is wrapped
  as a synthetic in-memory task (id `lyria-{uuid4.hex}`) that moves
  `pending -> running -> succeeded` on the first poll. The body is built by
  `lyria_body()` and the output extracted by `extract_lyria_output()` (both
  accept either the raw JSON dict or the pydantic `Interaction` model).
  - Models: `lyria-3`. Alias `gateway-music-vertex`.
  - Default audio output is MP3; `audio_format` `mp3`/`wav`/`ogg_opus` map to
    `audio/mpeg`/`audio/wav`/`audio/ogg` (the gateway convention every other
    music provider follows).

## xai-sdk — `xai_sdk` v1.17.0 (gRPC)

- **Client**: `from xai_sdk import AsyncClient` (env `XAI_API_KEY`).
- **Adapter note**: the adapter no longer imports `xai_sdk`; it speaks the xAI
  REST API (`POST /v1/images/generations`, `POST /v1/videos/generations` +
  `GET /v1/videos/{id}`) over httpx, so an operator can point the gateway at any
  xAI-compatible HTTP endpoint (`XAI_BASE_URL`) — the same 1:1 env contract every
  other REST adapter honours.
- **Sync vs async URL**: two `httpx` clients — image (Grok Imagine image) on
  `base_url` (the `XAI_IMAGE_BASE_URL` sync endpoint) and video (Grok Imagine
  video) on `backend.extra["video_base_url"]` (the `XAI_VIDEO_BASE_URL` async
  endpoint, set by `config.py` when it differs from the image one). For the real
  api.x.ai both share one host, so the two clients collapse unless an operator
  pins them apart.
- **Image**: `await client.image.sample(prompt, model, image_format="url|base64", aspect_ratio, resolution)` → `.url` / `.base64` / `.image` (bytes).
  - Models: `grok-imagine-image`, `-pro`, `-quality`. Legacy `grok-2-image-1212` via REST `POST https://api.x.ai/v1/images/generations` only.
- **Video**: `client.video.start(prompt, model, ...)` → `request_id`; poll `client.video.get(request_id)` until `DONE`. Models: `grok-imagine-video`, `grok-imagine-video-1.5-preview`. Result `.url` (24h TTL). Params: `duration`, `aspect_ratio`, `resolution`, `image_url` (i2v).

## volcengine-ark — `volcenginesdkarkruntime` (env `ARK_API_KEY`)

- **Client**: `from volcenginesdkarkruntime import AsyncArk; AsyncArk(api_key=, base_url=)`; base default `https://ark.cn-beijing.volces.com/api/v3`.
- **Image** (Seedream, OpenAI-compat): `client.images.generate(model, prompt, size, response_format, seed, guidance_scale, watermark, ...)`. Models: `doubao-seedream-3-0-t2i-250415`, `doubao-seedream-4-0-t2i-250828`. Response `data[].url|b64_json`.
- **Video (Seedance 1.0 & 2.0)** — the typed `content_generation.tasks` resource (a wrapper around the same REST API used to require raw httpx):
  - **Create**: `await client.content_generation.tasks.create(*, model, content, safety_identifier=, callback_url=, return_last_frame=, service_tier=, execution_expires_after=, priority=, generate_audio=, draft=, camera_fixed=, watermark=, seed=, resolution=, ratio=, duration=, frames=, tools=)` → `ContentGenerationTaskID{id, safety_identifier}`. POSTs `/contents/generations/tasks`.
    - `content` is a list of typed parts (the parts — not the model id — choose t2v / i2v / r2v):
      - `{type:"text", text}` — the prompt (required for t2v).
      - `{type:"image_url", image_url:{url}, role}` — `role` ∈ `first_frame` | `last_frame` | `reference_image`.
      - `{type:"video_url", video_url:{url}, role}` — Seedance 2.0 reference video (`role:"reference_video"`).
      - `{type:"audio_url", audio_url:{url}, role}` — Seedance 2.0 reference audio (`role:"reference_audio"`).
      - `{type:"draft_task", draft_task:{id}}` — resume a draft task.
    - 2.0 example model: `doubao-seedance-2-0-260128`. Typical 2.0 params: `generate_audio=True, ratio="16:9", duration=11, watermark=True`.
  - **Poll**: `await client.content_generation.tasks.get(*, task_id)` → `ContentGenerationTask`. GETs `/contents/generations/tasks/{task_id}`.
    - Fields: `id, model, safety_identifier, status, error, content, usage, created_at, updated_at, generate_audio, duration, ratio, resolution, seed, revised_prompt, service_tier, draft, draft_task_id, tools, frames, framespersecond, subdivisionlevel, fileformat, priority, execution_expires_after`.
    - `status` ∈ `running | failed | queued | succeeded | cancelled`.
    - `content` (`Content`): `video_url`, `last_frame_url`, `file_url`.
    - `error` (`ContentGenerationError`): `message`, `code`.
    - `usage` (`Usage`): `completion_tokens`, `total_tokens`.
  - `tasks` also exposes `.list(page_num=, page_size=, status=, task_ids=, model=, service_tier=)` and `.delete(task_id=)`; there is **no** `.cancel`.
- **Sync vs async URL**: two `AsyncArk` clients — image (Seedream) on
  `base_url` (the `*_IMAGE_BASE_URL` sync endpoint) and video (Seedance) on
  `backend.extra["video_base_url"]` (the `*_VIDEO_BASE_URL` async endpoint,
  set by `config.py` when it differs from the image one). For the real Ark
  API both share one host, so the two clients collapse to the same base
  unless an operator pins them apart.

## runapi-flux-2 — `runapi.flux_2.Flux2Client` v0.3.1 (sync)

- **Client**: `Flux2Client(api_key=)` (env `RUNAPI_API_KEY`). **Sync SDK** — wrapped via `asyncio.to_thread` in the adapter.
- **Text-to-image**: `client.text_to_image.create(model, prompt, aspect_ratio, output_resolution)` → `{id}`; poll `client.text_to_image.get(id)` until `completed`; result `images[].url` (temporary).
  - Models: `flux-2-flex-text-to-image`, `flux-2-max-text-to-image`, `flux-2-pro-text-to-image`.
- **Remix (i2i)**: `client.remix_image.create(model, prompt, source_image_urls=[...])`. Models: `flux-2-*-remix-image`.
- **No video.**

## openrouter — `openrouter` v1.1.30

- **Client**: `OpenRouter(api_key=, server_url=)` (env `OPENROUTER_API_KEY`). Headers `HTTP-Referer`, `X-OpenRouter-Title`.
- **Unified image API**: `POST /api/v1/images` body `{model, prompt, aspect_ratio, n, output_format, quality, resolution, size, seed, background, input_references:[{type:"image_url",image_url:{url}}], provider:{only,options}}` → `{created, data:[{b64_json, media_type}], usage}` (always base64).
- **Unified video API**: `POST /api/v1/videos` body `{model, prompt, aspect_ratio, resolution, size, duration, generate_audio, seed, frame_images:[{type:"image_url",image_url:{url},frame_type}], input_references, callback_url}` → `{id, polling_url, status:"pending|in_progress|completed|failed|cancelled|expired", unsigned_urls:[...], usage}`. Poll `GET /api/v1/videos/{id}`; download `GET /api/v1/videos/{id}/content`.
- **Sync vs async URL**: two `httpx` clients — image on `base_url` (the
  `OPENROUTER_IMAGE_BASE_URL` sync endpoint) and video on
  `backend.extra["video_base_url"]` (the `OPENROUTER_VIDEO_BASE_URL` async
  endpoint, set by `config.py` when it differs from the image one). For the
  real openrouter.ai both share one host, so the two clients collapse unless an
  operator pins them apart.

## dashscope — `dashscope` v1.26.5 (env `DASHSCOPE_API_KEY`)

- **Image (Wanx/Qwen-Image)**: native async task path first —
  `AioImageSynthesis.async_call(model, prompt, n, size="W*H", seed, negative_prompt)` →
  task, then `AioImageSynthesis.wait(task_id)` blocks until
  `{output:{task_status:"SUCCEEDED", results:[{url}]}}`. This is the
  unrestricted route serving every model incl. `qwen-image-2.0-pro`. Falls
  back to `AioImageSynthesis.sync_call(...)` (the headerless inline path via
  `BaseAioApi.call`, no `X-DashScope-Async: enable`) when the backend rejects
  async task submission (403 / `AccessDenied` / "does not support asynchronous
  calls"); the gateway wraps both as a synthetic in-memory task.
  - Models: `wanx2.1-t2i-turbo`, `wanx2.1-t2i-plus`, `wanx2.1-t2i-flash`,
    `qwen-image-2.0-pro`.
- **Video (Wan)**: `AioVideoSynthesis.async_call(model, prompt, size, duration, img_url, prompt_extend)` → task; poll `AioVideoSynthesis.fetch(task_id)`; result `output.video_url` (single temp OSS URL).
  - Models: `wanx2.1-t2v-turbo`, `wanx2.1-i2v-turbo`, `wanx2.1-t2v-plus`, `wanx2.1-i2v-plus`.
- **Sync vs async URL**: image calls carry `base_address=backend.base_url`
  (the `*_IMAGE_BASE_URL` sync endpoint); video calls carry
  `base_address=backend.extra["video_base_url"]` (the `*_VIDEO_BASE_URL` async
  endpoint, set by `config.py` when it differs from the image one). Per-call
  `base_address=` replaces the old module-global
  `dashscope.base_http_api_url` mutation.

## stability-sdk — `stability-sdk` v0.2.2 (legacy gRPC)

- **Client**: prefer REST `https://api.stability.ai/v2beta/...` via httpx (env `STABILITY_API_KEY`).
- **Sync vs async URL**: two `httpx.AsyncClient` instances — image (SD3/SDXL/Core/Ultra) on `base_url` (the `STABILITY_IMAGE_BASE_URL` sync endpoint) and video (SVD) on `backend.extra["video_base_url"]` (the `STABILITY_VIDEO_BASE_URL` async endpoint, set by `config.py` when it differs from the image one). For the real api.stability.ai both share one host, so the two clients collapse unless an operator pins them apart.
- **Image**: `POST /v2beta/stable-image/generate/sd3` (multipart: `prompt`, `model`, `seed`, `aspect_ratio`, `output_format`). Models: `sd3.5-large`, `sd3.5-medium`, `stable-image-core`, `sdxl`. Returns `image` (base64) + `finish_reason`.
- **Video (SVD)**: `POST /v2beta/stable-video-generation` (multipart: `image`, `motion_bucket_id`, `fps`, `seed`). Async seed → poll for result (raw MP4 bytes).

## elevenlabs — `elevenlabs` v2.62.0 (SDK)

- **Client**: `from elevenlabs import AsyncElevenLabs; AsyncElevenLabs(api_key=, base_url=|None, timeout=240.0)` (env `ELEVENLABS_MUSIC_API_KEY`/`ELEVENLABS_MUSIC_BASE_URL`, legacy `ELEVENLABS_API_KEY`/`ELEVENLABS_BASE_URL`). Required: `backend.api_key` else `ProviderNotConfiguredError("elevenlabs")`. No default base_url — `None` falls back to the SDK default.
- **Music**: `client.music.compose(...)` — an async generator that streams audio bytes from a single `POST /v1/music`. No task id to poll; the adapter mints a synthetic in-memory task (id `el-{uuid4.hex}`) that moves `pending -> running -> succeeded` as the stream completes on the first poll.
  - Models: `music_v1`, `music_v2` (default `music_v2`).
  - `compose` kwargs: `prompt`, `model_id`, `output_format`, `music_length_ms` (= public duration seconds converted to ms and clamped to `[3000, 600000]`), `seed`, `force_instrumental`, section-duration adherence, and provenance signing.
  - `output_format` combines the neutral file format, sample rate, and bitrate as the SDK's `{codec}_{sample_rate}_{bitrate}` string. A bare file format uses the SDK's `auto` choice.
  - Forwarded extra knobs: `finetune_id`, `respect_sections_durations`, `store_for_inpainting`, `sign_with_c_2_pa`.
  - Audio delivery = streamed bytes (`async for chunk in client.music.compose(**kwargs)` joined with `b"".join`), base64-encoded into `audio_b64`. `_media_type` maps `audio_format` → MIME (`mp3`→`audio/mpeg`, `wav`→`audio/wav`, `ogg`→`audio/ogg`, `aac`→`audio/aac`, default `audio/mpeg`). `usage = MusicUsage(duration=request.duration)`.
  - Errors: any `Exception` from the stream → `ProviderRequestError(502)`; no audio bytes → `TaskFailedError`; unknown task id → 404; missing prompt (no text part) → `ProviderRequestError(400)`.

## minimax — REST over `https://api.minimax.io`

- **Client**: two `httpx.AsyncClient`s (timeout 300, headers `{Authorization: Bearer {api_key}, Content-Type: application/json}`). The music client (`_client`) uses `backend.base_url or "https://api.minimax.io"` (env `MINIMAX_MUSIC_API_KEY`/`MINIMAX_MUSIC_BASE_URL`, legacy `MINIMAX_API_KEY`/`MINIMAX_BASE_URL`); the video client (`_client_video`) uses `backend.extra["video_base_url"]` when it differs (env `MINIMAX_VIDEO_API_KEY`/`MINIMAX_VIDEO_BASE_URL`), else collapses onto the music client. The real `api.minimax.io` serves both at one host. Required: `backend.api_key` else `ProviderNotConfiguredError("minimax")`.
- **Music**: `POST /v1/music_generation` (synchronous — a single blocking call returns `data.status` 1 = in progress or 2 = completed with audio inline). No job id; the adapter mints a synthetic in-memory task (id `mm-{uuid4.hex}`) and runs the POST on the first poll — `pending -> running -> succeeded`/`failed`. A `data.status` of 1 leaves the task `running` so a later poll re-issues the call.
  - Models: `music-3.0` (default), `music-2.6`, `music-cover`.
  - Body (`_build_body`): `model`, descriptive `prompt`, structured `lyrics`, `is_instrumental`, URL output, and `audio_setting`. Neutral sample-rate Hz and bitrate kbps become native `sample_rate` and bits-per-second. Reference audio becomes `audio_url` or `audio_base64`; lyric enhancement becomes `lyrics_optimizer`.
  - Forwarded extra knobs: `stream`, `lyrics_optimizer`, `audio_base64`, `cover_feature_id`.
  - Status mapping: `data.status` 1 → `running`, 2 → `succeeded`; `base_resp.status_code != 0` → `failed` (error from `base_resp.status_msg`). `extra_info.music_duration` → `MusicUsage(duration)`.
  - Audio delivery: `output_format == "url"` and `data.audio` starts with `http` → `audio_urls=[audio]`; else hex-decoded via `bytes.fromhex` and re-base64 into `audio_b64`. `_MIME_BY_FORMAT = {mp3:audio/mpeg, wav:audio/wav, pcm:audio/pcm}` (default `audio/mpeg`).
  - Errors: `httpx.HTTPError` → `ProviderRequestError(502)`; HTTP ≥400 → `ProviderRequestError` via `_map_status`; `base_resp.status_code != 0` → `TaskFailedError`; no audio → `TaskFailedError`; unknown task id → 404.
- **Video (H3)**: genuine two-phase async REST — `POST /v2/video_generation` → `{task_id}`, then `GET /v2/query/video_generation/{task_id}` → `{task:{status, content:{url, cover_url}}}`. The H3 `content[]` shape (typed `text` / `image_url`-with-role / `video_url` / `audio_url` parts) *is* the unified video schema's content shape, so the parts pass straight through.
  - Model: `MiniMax-H3` (one omni model; the content parts pick t2v / i2v / first-frame / reference / audio).
  - Body (`_build_video_body`): `model`, `content` = `[p.model_dump(exclude_none=True) for p in request.content]`, then optional `duration` (int seconds), `resolution` (verbatim H3 token such as `768P`/`2K` — not derived from width/height), `ratio` (from `aspect_ratio(request)`), and a pass-through of provider-specific knobs the caller stashed in `request.extra` (e.g. `prompt_optimizer`, `watermark`).
  - Create: `POST /v2/video_generation` → reads `task_id`; honours a `base_resp.status_code != 0` error envelope when present; no `task_id` → `ProviderRequestError("minimax video create returned no task_id", 502)`. Returns a `pending` `UnifiedVideoTask`.
  - Poll: `GET /v2/query/video_generation/{task_id}` → `task`. Status mapping (`_VIDEO_STATUS_MAP`): `queued` → `pending`, `running`/`processing` → `running`, `succeeded` → `succeeded`, `failed` → `failed`, `cancelled` → `cancelled` (unknown status defaults to `running`). On `succeeded` reads `task.content.url` → `video_urls=[url]` and `task.content.cover_url` → `cover_url`; `succeeded` with no url → `failed` (defensive). On `failed`/`cancelled` the error comes from `task.error.message`/`task.error.code` (or the raw status). `task.model` → `model`.
  - Errors: `httpx.HTTPError` → `ProviderRequestError(502)`; HTTP ≥400 → `ProviderRequestError` via `_map_status`; `base_resp.status_code != 0` → `TaskFailedError`; no `task_id` → `ProviderRequestError`; no `task` in poll → `ProviderRequestError`.

## udioapi — REST over `https://udioapi.pro`

- **Client**: `make_client(backend.base_url or "https://udioapi.pro", timeout=180.0, headers={Authorization: Bearer {api_key}})` (env `UDIOAPI_MUSIC_API_KEY`/`UDIOAPI_MUSIC_BASE_URL`, legacy `UDIOAPI_API_KEY`/`UDIOAPI_BASE_URL`). Required: `backend.api_key` else `ProviderNotConfiguredError("udioapi")`.
- **Music**: Genuine two-phase async REST mirroring Suno's shape.
  - **Create**: `POST /api/v2/generate` body from `_build_body` → `{workId}` (or `data.task_id`); the upstream workId *is* the gateway task id (no synthetic store). Missing → `ProviderRequestError("udioapi create returned no workId")`.
  - **Poll**: `GET /api/v2/feed?workId=...` → `data.response_data[]` tracks; picks the first `complete` track else the last track. No tracks yet → `running` (queued).
  - Models: `chirp-v4-0`, `chirp-v4-5`, `chirp-v4-5-plus`, `chirp-v5`, `chirp-v5-5`.
  - Body (`_build_body`): custom mode (when style, title, or negative prompt is present) uses native `prompt`, `style`, `title`, and `tags`; otherwise inspiration mode uses `gpt_description_prompt`. Neutral instrumental, vocal gender, style strength, novelty, and reference-audio strength map to the corresponding native controls.
  - Status mapping (`_STAGE_TO_STATUS`): track `status` `text`/`first` → `running`, `complete` → `succeeded`; `data.type` upper `FAIL` → `failed`; `track.fail_message`/`error_message` → `failed`.
  - Audio delivery = URL; on `complete` reads `track.audio_url` → `audio_urls=[url]`, `audio_media_type="audio/mpeg"`; `complete` with no URL → `failed` (moderation). `track.duration` → `MusicUsage(duration)`.
  - Errors: `httpx.HTTPError` → `ProviderRequestError(502)`; HTTP ≥400 → `ProviderRequestError` via `_map_status`; create with no workId → `ProviderRequestError`.

## mureka — REST over `https://platform.mureka.ai`

- **Client**: `make_client(backend.base_url or "https://platform.mureka.ai", timeout=180.0, headers={Authorization: Bearer {api_key}})` (env `MUREKA_MUSIC_API_KEY`/`MUREKA_MUSIC_BASE_URL`, legacy `MUREKA_API_KEY`/`MUREKA_BASE_URL`). Required: `backend.api_key` else `ProviderNotConfiguredError("mureka")`.
- **Music**: Genuine two-phase async REST. The official docs render schemas client-side, so the adapter is tolerant of response field names via candidate fallbacks.
  - **Create**: `POST /v1/song/generate` body from `_build_body` → a task id read via `_first(data, _TASK_ID_FIELDS) or _first(data.data, _TASK_ID_FIELDS)` where `_TASK_ID_FIELDS = ("task_id", "taskId", "id")`; the upstream id is the gateway task id. Missing → `ProviderRequestError("mureka create returned no task_id")`.
  - **Poll**: `GET /v1/song/query/{task_id}` → `{status, audio_url, ...}`. `status` read from `data.status` or `data.task_status` (default `running`).
  - Models: `mureka-song-1`, `mureka-song-1.5`.
  - Body (`_build_body`): `model`, descriptive `prompt`, structured `lyrics`, `title`, style/negative tags, `instrumental`, `bpm`, integer duration, and seed. Neutral voice maps to `voice_id`; file format, sample rate, and bitrate form native `audio_config`.
  - Status mapping (`_STATUS_MAP`): `queued`/`pending` → `pending`, `running`/`processing` → `running`, `succeeded`/`success`/`completed` → `succeeded`, `failed`/`error` → `failed`.
  - Audio delivery = URL; on `succeeded` reads audio via `_first(data, _AUDIO_FIELDS)` where `_AUDIO_FIELDS = ("audio_url", "audioUrl", "url")` → `audio_urls=[audio]`, `audio_media_type="audio/mpeg"`; no URL → `failed` (error from `fail_message` or `"mureka task completed with no audio URL"`). `duration` from `data.duration` or `data.extra.duration` → `MusicUsage(duration)`. On `failed`, error from `fail_message`/`error`/`"mureka task failed"`.
  - Errors: `httpx.HTTPError` → `ProviderRequestError(502)`; HTTP ≥400 → `ProviderRequestError` via `_map_status`; create with no task_id → `ProviderRequestError`.

## acestep — REST, dual-mode (native self-hosted + completion cloud)

- **Client**: `make_client(backend.base_url, timeout=300.0, headers={Authorization: Bearer {api_key}} if api_key else {})` (env `ACESTEP_MUSIC_API_KEY`/`ACESTEP_MUSIC_BASE_URL`, legacy `ACESTEP_API_KEY`/`ACESTEP_BASE_URL`). **Self-hosted — no default cloud host**: requires `ACESTEP_BASE_URL` (i.e. `backend.base_url`) else `ProviderNotConfiguredError("acestep", "ACE-Step requires ACESTEP_BASE_URL to be set.")`. `api_key` is optional (header sent only when set).
- **Mode selection** (`_resolve_mode`): the same server speaks two modes (matching the official `acestep.sh` `api_mode`). `auto` (the default) picks **completion** when the base URL host ends with `acemusic.ai` — the cloud host fronts native `/release_task` with Cloudflare and its origin is chronically unreachable (CI sees `504 Gateway Timeout`, `retry-after: 120`, even with a valid key and the create-retry) — and **native** otherwise (self-hosted servers like `http://127.0.0.1:8001`). Force either with `backend.extra["acestep_api_mode"]` = `native` | `completion`. The official script and the sibling `speak` repo both use completion mode against `api.acemusic.ai`.
- **Native music**: Genuine two-phase async REST plus a binary download.
  - **Create**: `POST /release_task` body from `_build_body` (with a bounded retry — `_create_max_attempts=3`, backoff `0.5*2^n` — on transient `502`/`503`/`504`/timeouts; 4xx is raised immediately) → `data.task_id`; the upstream id is the gateway task id. Missing → `ProviderRequestError("acestep create returned no task_id")`.
  - **Poll**: `POST /query_result` body `{"task_id_list": [task_id]}` → `data[]`; finds the entry matching `task_id` (else `items[0]`), maps its integer `status` via `_STATUS_MAP = {0: running, 1: succeeded, 2: failed}` (default `running`). No entry → `running`.
  - **Fetch audio**: on `succeeded` parses `entry.result` (a JSON string, or a list/dict) → `parsed[0].file` (a path like `/v1/audio?path=...`); `GET` that path → raw bytes → base64 into `audio_b64`. If the fetch raises `ProviderRequestError`, falls back to `audio_urls=[absolute path]`.
  - **Native body (`_build_body`)**: `prompt`, structured `lyrics`, `thinking`/`use_format`/`use_cot_caption`/`use_cot_language`/`use_random_seed` (all `True` by default; `use_random_seed` flips to `False` when `seed` is set), vocal language, translated file format, `audio_duration`, `bpm`, combined `key_scale`, `time_signature`, `guidance_scale`, `seed`, `model`, `inference_steps`, and `batch_size` (from `n`). A continuation or reference audio part maps to `src_audio_path` (official field name) and defaults `task_type="cover"`; `audio_cover_strength`/`repainting_start`/`repainting_end` forward from `extra`.
  - **Completion music**: OpenAI-style single-phase — `POST /v1/chat/completions` returns the final result inline. Because there is no upstream job id to poll, the adapter mints a gateway-local task id (`acestep-<uuid>`) at create and runs the blocking call on first poll (the synthetic-task pattern shared with the MiniMax adapter; in-memory `_COMPLETION_TASKS`).
  - **Create (completion)**: `_create_completion_task` stores `{model, request, status:"pending"}` and returns a `pending` `UnifiedMusicTask`.
  - **Poll (completion)**: `_get_completion_task` — on first poll builds the body (`_build_completion_body`) and `POST /v1/chat/completions`; terminal results are cached so subsequent polls return the cached task. `choices[0].finish_reason == "error"` → `failed` (error from `_completion_error`: `detail`/`error.message`). Audio is read from `choices[0].message.audio[].audio_url.url` (a `data:audio/mpeg;base64,...` data URL → `audio_b64` via `_data_url_b64`); no audio → `failed`. `message.content` (string) → `task.lyrics`. `usage.duration`/`metas.duration` → `MusicUsage(duration)`. HTTP ≥400 → `ProviderRequestError`; non-JSON → `TaskFailedError`.
  - **Completion body (`_build_completion_body`)**: `model` (prefixed `acemusic/` when it has no `/`; default `acemusic/acestep-v15-turbo`), `messages=[{role:"user", content}]` where content is the string `"<prompt>{p}</prompt>\n<lyrics>{l}</lyrics>"` for text-only or a parts array `[{type:text,text},{type:input_audio,input_audio:{data,format}}]` when source audio is given (via `_completion_message`/`_decode_audio_input`), `stream:false`, `thinking`/`use_format`/`use_cot_caption`/`use_cot_language` `True`, and `audio_config={format, vocal_language?, duration?, bpm?, key_scale?, time_signature?}`; optional `guidance_scale`/`seed`/`batch_size`; `sample_mode`/`task_type`/`audio_cover_strength`/`repainting_start`/`repainting_end` forward from `extra`.
  - Models: `acestep-v15-turbo`, `acestep-v15-xl-turbo`, `acestep-v15-base`, `acestep-v15-turbo-shift3`, `ace-step-1.5`.
  - Audio delivery = inline base64 (native: bytes fetched via `_fetch_audio`; completion: data-URL payload via `_data_url_b64`); native fallback to `audio_urls=[absolute path]` on fetch failure. `_media_type_for` maps `flac`→`audio/flac`, `mp3`→`audio/mpeg`, `opus`→`audio/ogg`, `aac`→`audio/aac`, `wav`/`wav32`→`audio/wav`, default `audio/mpeg`. Native `parsed[0].metas.duration` → `MusicUsage(duration)`; `parsed[0].lyrics` → `task.lyrics`. No result → `failed`; no file/audio → `failed`.
  - Errors: `httpx.HTTPError`/transport error → `ProviderRequestError(502)`; HTTP ≥400 → `ProviderRequestError` via `_map_status`; create with no task_id → `ProviderRequestError`; audio-fetch HTTP ≥400 → `ProviderRequestError`; non-JSON completion response → `TaskFailedError`.

