# Provider API reference

Concrete, code-verified facts for each provider SDK installed in this repo's
venv. This is the source of truth the gateway adapters were written against.
Models/params marked *(docs)* are from official docs, not the installed SDK.

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

## google-genai — `google-genai` v2.16.0

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
- **Music (Lyria 3)**: REST `POST {music_base}/v1beta/models/{model}:predictInteractions?key={api_key}` over a per-request `httpx.AsyncClient(timeout=240.0)` — the SDK does not yet expose Lyria ergonomically. Auth = api_key as a `?key=` query param. `music_base` = `backend.extra["music_base_url"]` or `backend.base_url` or `https://generativelanguage.googleapis.com`.
  - Models: `lyria-3`.
  - Body: `{model, input: parts[], config}`. `input` parts are `{type:"text", text}` (from request text parts) and `{type:"image", mime_type (default "image/jpeg"), data}` (from `extra["images"]`); if no parts, a single text part from `prompt()`. `config` = `{response_modalities:["AUDIO"]}` plus `response_format:{type:audio_format}` (if set), `negative_prompt`, `seed`, `guidance_scale`, `number_of_outputs` (= `n`), then `extra["lyria_config"]` merged in.
  - Synchronous — a single call returns audio inline. Wrapped as a synthetic in-memory task (id `lyria-{uuid4.hex}`) that moves `pending -> running -> succeeded` on the first poll, like ElevenLabs/MiniMax.
  - Response: `steps[].content[]` (or `model_output[].content[]`) blocks — `{type:"audio", data}` (base64) and `{type:"text", text}` (lyrics); top-level `output_audio`/`audio` and `output_text`/`text` fallbacks. `audio_media_type` hard-coded `audio/wav`. `usage = MusicUsage(duration=int(request.duration))` when duration set.
  - Errors: `httpx.HTTPError` or HTTP ≥400 → `ProviderRequestError(502)`; no audio → `TaskFailedError`; unknown task id → 404.

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
  - `compose` kwargs: `prompt` (from `request.prompt()`), `model_id`, `output_format` (built from `audio_quality`/`audio_format` — see below; omit if neither set), `music_length_ms` (= `duration * 1000` seconds→ms, clamped to `[3000, 600000]`), `seed`, `force_instrumental` (= `is_instrumental`).
  - `output_format` mapping: if `audio_quality` is set → `{codec}_{audio_quality}` where codec comes from `_CODEC_BY_FORMAT` keyed by `audio_format` (default `mp3`); bare `audio_format` with no quality → `"auto"`; nothing → `None` (omitted).
  - Forwarded extra knobs: `finetune_id`, `respect_sections_durations`, `store_for_inpainting`, `sign_with_c_2_pa`.
  - Audio delivery = streamed bytes (`async for chunk in client.music.compose(**kwargs)` joined with `b"".join`), base64-encoded into `audio_b64`. `_media_type` maps `audio_format` → MIME (`mp3`→`audio/mpeg`, `wav`→`audio/wav`, `ogg`→`audio/ogg`, `aac`→`audio/aac`, default `audio/mpeg`). `usage = MusicUsage(duration=request.duration)`.
  - Errors: any `Exception` from the stream → `ProviderRequestError(502)`; no audio bytes → `TaskFailedError`; unknown task id → 404; missing prompt (no text part) → `ProviderRequestError(400)`.

## minimax — REST over `https://api.minimax.io`

- **Client**: `httpx.AsyncClient(base_url=backend.base_url or "https://api.minimax.io", timeout=300, headers={Authorization: Bearer {api_key}, Content-Type: application/json})` (env `MINIMAX_MUSIC_API_KEY`/`MINIMAX_MUSIC_BASE_URL`, legacy `MINIMAX_API_KEY`/`MINIMAX_BASE_URL`). Required: `backend.api_key` else `ProviderNotConfiguredError("minimax")`.
- **Music**: `POST /v1/music_generation` (synchronous — a single blocking call returns `data.status` 1 = in progress or 2 = completed with audio inline). No job id; the adapter mints a synthetic in-memory task (id `mm-{uuid4.hex}`) and runs the POST on the first poll — `pending -> running -> succeeded`/`failed`. A `data.status` of 1 leaves the task `running` so a later poll re-issues the call.
  - Models: `music-3.0` (default), `music-2.6`, `music-cover`.
  - Body (`_build_body`): `model`, `prompt` (= `request.prompt()`, when present), `lyrics` (= `extra["lyrics"]`, or the prompt itself when not instrumental and no separate lyrics — in which case `prompt` is popped), `is_instrumental`, `output_format` (`"url"` unless `audio_format == "hex"`), `audio_setting` (`sample_rate`/`bitrate` from `audio_quality` split on `_`, `format` from `audio_format` if `mp3`/`wav`/`pcm`), `audio_url` (= `request.reference_audios()[0]` if present).
  - Forwarded extra knobs: `stream`, `lyrics_optimizer`, `audio_base64`, `cover_feature_id`.
  - Status mapping: `data.status` 1 → `running`, 2 → `succeeded`; `base_resp.status_code != 0` → `failed` (error from `base_resp.status_msg`). `extra_info.music_duration` → `MusicUsage(duration)`.
  - Audio delivery: `output_format == "url"` and `data.audio` starts with `http` → `audio_urls=[audio]`; else hex-decoded via `bytes.fromhex` and re-base64 into `audio_b64`. `_MIME_BY_FORMAT = {mp3:audio/mpeg, wav:audio/wav, pcm:audio/pcm}` (default `audio/mpeg`).
  - Errors: `httpx.HTTPError` → `ProviderRequestError(502)`; HTTP ≥400 → `ProviderRequestError` via `_map_status`; `base_resp.status_code != 0` → `TaskFailedError`; no audio → `TaskFailedError`; unknown task id → 404.

## udioapi — REST over `https://udioapi.pro`

- **Client**: `make_client(backend.base_url or "https://udioapi.pro", timeout=180.0, headers={Authorization: Bearer {api_key}})` (env `UDIOAPI_MUSIC_API_KEY`/`UDIOAPI_MUSIC_BASE_URL`, legacy `UDIOAPI_API_KEY`/`UDIOAPI_BASE_URL`). Required: `backend.api_key` else `ProviderNotConfiguredError("udioapi")`.
- **Music**: Genuine two-phase async REST mirroring Suno's shape.
  - **Create**: `POST /api/v2/generate` body from `_build_body` → `{workId}` (or `data.task_id`); the upstream workId *is* the gateway task id (no synthetic store). Missing → `ProviderRequestError("udioapi create returned no workId")`.
  - **Poll**: `GET /api/v2/feed?workId=...` → `data.response_data[]` tracks; picks the first `complete` track else the last track. No tracks yet → `running` (queued).
  - Models: `chirp-v4-0`, `chirp-v4-5`, `chirp-v4-5-plus`, `chirp-v5`, `chirp-v5-5`.
  - Body (`_build_body`): *custom mode* (when `extra["style"]`/`extra["title"]` or `negative_prompt` are present) → `prompt`, `style`, `title`, `tags` (= `negative_prompt`); else *inspiration mode* → `gpt_description_prompt` = prompt. Also `model`, `make_instrumental` (= `is_instrumental`).
  - Forwarded extra knobs: `gender`, `style_weight`, `weirdness_constraint`, `audio_weight`.
  - Status mapping (`_STAGE_TO_STATUS`): track `status` `text`/`first` → `running`, `complete` → `succeeded`; `data.type` upper `FAIL` → `failed`; `track.fail_message`/`error_message` → `failed`.
  - Audio delivery = URL; on `complete` reads `track.audio_url` → `audio_urls=[url]`, `audio_media_type="audio/mpeg"`; `complete` with no URL → `failed` (moderation). `track.duration` → `MusicUsage(duration)`.
  - Errors: `httpx.HTTPError` → `ProviderRequestError(502)`; HTTP ≥400 → `ProviderRequestError` via `_map_status`; create with no workId → `ProviderRequestError`.

## mureka — REST over `https://platform.mureka.ai`

- **Client**: `make_client(backend.base_url or "https://platform.mureka.ai", timeout=180.0, headers={Authorization: Bearer {api_key}})` (env `MUREKA_MUSIC_API_KEY`/`MUREKA_MUSIC_BASE_URL`, legacy `MUREKA_API_KEY`/`MUREKA_BASE_URL`). Required: `backend.api_key` else `ProviderNotConfiguredError("mureka")`.
- **Music**: Genuine two-phase async REST. The official docs render schemas client-side, so the adapter is tolerant of response field names via candidate fallbacks.
  - **Create**: `POST /v1/song/generate` body from `_build_body` → a task id read via `_first(data, _TASK_ID_FIELDS) or _first(data.data, _TASK_ID_FIELDS)` where `_TASK_ID_FIELDS = ("task_id", "taskId", "id")`; the upstream id is the gateway task id. Missing → `ProviderRequestError("mureka create returned no task_id")`.
  - **Poll**: `GET /v1/song/query/{task_id}` → `{status, audio_url, ...}`. `status` read from `data.status` or `data.task_status` (default `running`).
  - Models: `mureka-song-1`, `mureka-song-1.5`.
  - Body (`_build_body`): `model`, `lyrics` (= `extra["lyrics"]` and `prompt` together; or the prompt itself when it contains a newline or starts with `[`) else `prompt`, `title` (= `extra["title"]`), `tags` (= `negative_prompt` elif `extra["style"]`), `instrumental` (= `is_instrumental`), `bpm`, `duration` (= `int(request.duration)`).
  - Forwarded extra knobs: `model_name`, `audio_config`, `voice_id`, `seed`.
  - Status mapping (`_STATUS_MAP`): `queued`/`pending` → `pending`, `running`/`processing` → `running`, `succeeded`/`success`/`completed` → `succeeded`, `failed`/`error` → `failed`.
  - Audio delivery = URL; on `succeeded` reads audio via `_first(data, _AUDIO_FIELDS)` where `_AUDIO_FIELDS = ("audio_url", "audioUrl", "url")` → `audio_urls=[audio]`, `audio_media_type="audio/mpeg"`; no URL → `failed` (error from `fail_message` or `"mureka task completed with no audio URL"`). `duration` from `data.duration` or `data.extra.duration` → `MusicUsage(duration)`. On `failed`, error from `fail_message`/`error`/`"mureka task failed"`.
  - Errors: `httpx.HTTPError` → `ProviderRequestError(502)`; HTTP ≥400 → `ProviderRequestError` via `_map_status`; create with no task_id → `ProviderRequestError`.

## acestep — REST (self-hosted, no default host)

- **Client**: `make_client(backend.base_url, timeout=300.0, headers={Authorization: Bearer {api_key}} if api_key else {})` (env `ACESTEP_MUSIC_API_KEY`/`ACESTEP_MUSIC_BASE_URL`, legacy `ACESTEP_API_KEY`/`ACESTEP_BASE_URL`). **Self-hosted — no default cloud host**: requires `ACESTEP_BASE_URL` (i.e. `backend.base_url`) else `ProviderNotConfiguredError("acestep", "ACE-Step requires ACESTEP_BASE_URL to be set.")`. `api_key` is optional (header sent only when set).
- **Music**: Genuine two-phase async REST plus a binary download.
  - **Create**: `POST /release_task` body from `_build_body` → `data.task_id`; the upstream id is the gateway task id. Missing → `ProviderRequestError("acestep create returned no task_id")`.
  - **Poll**: `POST /query_result` body `{"task_id_list": [task_id]}` → `data[]`; finds the entry matching `task_id` (else `items[0]`), maps its integer `status` via `_STATUS_MAP = {0: running, 1: succeeded, 2: failed}` (default `running`). No entry → `running`.
  - **Fetch audio**: on `succeeded` parses `entry.result` (a JSON string, or a list/dict) → `parsed[0].file` (a path like `/v1/audio?path=...`); `GET` that path → raw bytes → base64 into `audio_b64`. If the fetch raises `ProviderRequestError`, falls back to `audio_urls=[absolute path]`.
  - Models: `acestep-v15-turbo`, `acestep-v15-base`, `ace-step-1.5`.
  - Body (`_build_body`): `prompt`, `lyrics` (= `extra["lyrics"]`), `vocal_language`, `audio_format` (if in `flac`/`mp3`/`opus`/`aac`/`wav`/`wav32`), `audio_duration` (= `float(duration)`), `bpm`, `key_scale`, `time_signature`, `guidance_scale`, `seed` (and `use_random_seed=False`), `model`.
  - Forwarded extra knobs: `thinking`, `sample_mode`, `sample_query`, `use_format`, `inference_steps`, `batch_size`, `task_type`, `reference_audio_path`.
  - Audio delivery = inline base64 (bytes fetched via `_fetch_audio`, base64-encoded); fallback to `audio_urls=[absolute path]` on fetch failure. `_media_type` maps `flac`→`audio/flac`, `mp3`→`audio/mpeg`, `opus`→`audio/ogg`, `aac`→`audio/aac`, `wav`/`wav32`→`audio/wav`, default `audio/mpeg`. `parsed[0].metas.duration` → `MusicUsage(duration)`; `parsed[0].lyrics` → `task.lyrics`. No result → `failed`; no file → `failed`.
  - Errors: `httpx.HTTPError` → `ProviderRequestError(502)`; HTTP ≥400 → `ProviderRequestError` via `_map_status`; create with no task_id → `ProviderRequestError`; audio-fetch HTTP ≥400 → `ProviderRequestError`.

