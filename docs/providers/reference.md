# Provider API reference

Concrete, code-verified facts for each provider SDK installed in this repo's
venv. This is the source of truth the gateway adapters were written against.
Models/params marked *(docs)* are from official docs, not the installed SDK.

---

## openai — `openai` v2.53.0

- **Client**: `from openai import AsyncOpenAI; AsyncOpenAI(api_key=, base_url=)` (env `OPENAI_API_KEY`, `OPENAI_BASE_URL`).
- **Image**: `client.images.generate(model, prompt, n, size, quality, response_format, style, background, output_format, user, ...)`.
  - Models: `gpt-image-1`, `gpt-image-1-mini`, `gpt-image-2`, `dall-e-2`, `dall-e-3`.
  - GPT-image models **always** return `data[].b64_json`; `dall-e-2/3` return `data[].url` (default) or `b64_json`.
  - `response.data[i].b64_json | .url`; `usage` only for gpt-image.
- **Edit**: `client.images.edit(image=<file>, mask=, model, prompt, ...)` — multipart.
- **Video (Sora)**: `client.videos.create(model, prompt, seconds="4|8|12", size, input_reference=)` → `Video{id,status,progress}`. Poll `client.videos.retrieve(id)` until `completed`/`failed`, then `client.videos.download_content(id)` → MP4 bytes.
  - Models: `sora-2`, `sora-2-pro`.

## google-genai — `google-genai` v2.16.0

- **Client**: `from google import genai; client = genai.Client(api_key=)` (env `GOOGLE_API_KEY`/`GEMINI_API_KEY`). Async via `client.aio.models.*`.
- **Image (Imagen)**: `client.aio.models.generate_images(model, prompt, config=GenerateImagesConfig(...))`.
  - Models: `imagen-4.0-generate-001`, `imagen-3.0-generate-001`.
  - Response: `resp.generated_images[i].image.image_bytes` (bytes; **no URL** on dev API).
- **Image (Gemini native)**: `generate_content(model="gemini-2.5-flash-image", contents=, config=GenerateContentConfig(response_modalities=["IMAGE"]))`. Image bytes in `part.inline_data.data`.
- **Video (Veo)**: `client.aio.models.generate_videos(model, source=GenerateVideosSource(prompt=|image=), config=GenerateVideosConfig(...))` → long-running op. Poll `client.aio.operations.get(op)` until `op.done`. Result: `op.result.generated_videos[0].video.uri`; fetch bytes via `client.aio.files.download(file=video)`.
  - Models: `veo-2.0-generate-001`, `veo-3.0-generate-001`, `veo-3.1-generate-preview`.

## xai-sdk — `xai_sdk` v1.17.0 (gRPC)

- **Client**: `from xai_sdk import AsyncClient` (env `XAI_API_KEY`).
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

## dashscope — `dashscope` v1.26.5 (env `DASHSCOPE_API_KEY`)

- **Image (Wanx)**: `dashscope.ImageSynthesis.async_call(model, prompt, n, size="W*H", seed, negative_prompt, prompt_extend)` → `{output:{task_id, task_status:"PENDING"}}`. Poll `fetch(task_id)`/`wait()`; result `output.results[].url` (temp OSS URL).
  - Models: `wanx2.1-t2i-turbo`, `wanx2.1-t2i-plus`.
- **Video (Wan)**: `dashscope.VideoSynthesis.async_call(model, prompt, size, duration, img_url, prompt_extend)` → task; result `output.video_url` (single temp OSS URL).
  - Models: `wanx2.1-t2v-turbo`, `wanx2.1-i2v-turbo`. Use `AioVideoSynthesis` for async.

## stability-sdk — `stability-sdk` v0.2.2 (legacy gRPC)

- **Client**: prefer REST `https://api.stability.ai/v2beta/...` via httpx (env `STABILITY_API_KEY`).
- **Image**: `POST /v2beta/stable-image/generate/sd3` (multipart: `prompt`, `model`, `seed`, `aspect_ratio`, `output_format`). Models: `sd3.5-large`, `sd3.5-medium`, `stable-image-core`, `sdxl`. Returns `image` (base64) + `finish_reason`.
- **Video (SVD)**: `POST /v2beta/stable-video-generation` (multipart: `image`, `motion_bucket_id`, `fps`, `seed`). Async seed → poll for result (raw MP4 bytes).
