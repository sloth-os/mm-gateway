# Unification design — API commonalities & differences

This document records how the provider APIs differ, the common spine we
extracted, and the front-end shapes the gateway exposes. Music joins image and
video as a first-class modality: the gateway serves **Gemini Lyria 3-compatible**
music endpoints (`POST /v1/music`, `GET /v1/music/{id}`) backed by six music
providers (Google Lyria, ElevenLabs, MiniMax, udioapi.pro, Mureka, ACE-Step),
unified through a single `UnifiedMusicRequest`/`UnifiedMusicTask` schema and the
`translators/music/lyria_compat.py` translator. It is the rationale behind
`mm_gateway/schemas/` and `mm_gateway/translators/`.

## Provider matrix

| Provider | Image | Video | Music | Async video? | Image delivery | Auth env |
|---|:--:|:--:|:--:|:--:|---|---|
| OpenAI (Sora/DALL·E) | ✅ | ✅ | ❌ | task+poll (retrieve/download) | `data[].b64_json` (gpt-image) / `url` (dall-e) | `OPENAI_API_KEY` |
| Google (Imagen/Veo/Lyria) | ✅ | ✅ | ✅ (Lyria 3) | LRO op+poll | bytes (`image_bytes`) | `GOOGLE_API_KEY` |
| xAI (Grok Imagine) | ✅ | ✅ | ❌ | deferred+poll | `url`/`base64`/bytes | `XAI_API_KEY` |
| Volcengine Ark (Seedream/Seedance) | ✅ | ✅ | ❌ | task+poll | `data[].url|b64_json` | `ARK_API_KEY` |
| FLUX.2 (runapi) | ✅ | ❌ | ❌ | n/a | `images[].url` | `RUNAPI_API_KEY` |
| OpenRouter (unified) | ✅ | ✅ | ❌ | job+poll | `data[].b64_json` | `OPENROUTER_API_KEY` |
| DashScope (Wanx/Qwen-Image/Wan) | ✅ | ✅ | ❌ | task+poll | native async `results[].url` (sync_call fallback) / `video_url` | `DASHSCOPE_API_KEY` |
| Stability (SD/SVD) | ✅ | ✅ | ❌ | sync image / async video | base64 / MP4 bytes | `STABILITY_API_KEY` |
| ElevenLabs (music) | ❌ | ❌ | ✅ | n/a | streamed bytes → base64 (`audio/mpeg`) | `ELEVENLABS_API_KEY` |
| MiniMax (music) | ❌ | ❌ | ✅ | n/a | 24h URL (default) → `audio_urls`, or hex bytes → base64 | `MINIMAX_API_KEY` |
| udioapi.pro (music) | ❌ | ❌ | ✅ | task+poll | `audio_url` (URL) | `UDIOAPI_API_KEY` |
| Mureka (music) | ❌ | ❌ | ✅ | task+poll | `audio_url` (URL) | `MUREKA_API_KEY` |
| ACE-Step 1.5 (music) | ❌ | ❌ | ✅ | task+poll + binary download | file-path bytes → base64 (`audio/mpeg`) | `ACESTEP_API_KEY` (optional; needed to register) |

## Common spine (what every provider can do)

Every image call: `model + prompt -> [image]`. Every video call:
`model + prompt (+ optional input image) -> async task -> [video_url]`. Everything
else — `n`, `size`, `seed`, `quality`, `negative_prompt`, `aspect_ratio`,
`duration`, `fps`, `style`, `guidance_scale`, `steps`, `watermark`,
`camera_fixed`, `generate_audio` — is a *knob* that some providers honour and
others ignore. The unified schema is the union of these knobs; the gateway
passes each knob only to providers that accept it (adapter's responsibility),
so a client never has to learn provider-specific shapes.

## Key differences the gateway must paper over

1. **Delivery form** — base64 (OpenAI gpt-image, OpenRouter, Stability, Google
   bytes), URL (DALL·E, FLUX, DashScope, Volcengine, xAI). The adapter
   normalises to whichever `response_format` the client asked for: a URL result
   is downloaded+base64-encoded for `b64_json`; a b64 result can stay as-is.
2. **Sync vs async video** — image gen is synchronous on most providers;
   DashScope is the exception (native async task API for image, with a
   `sync_call` fallback when the backend rejects async submission — see the
   DashScope adapter docstring). Video gen is *always* async (task + poll) on
   every provider. The unified video API therefore exposes a task lifecycle
   (`pending → running → succeeded | failed`) and the gateway optionally blocks
   (sync-style) up to `MAX_SYNC_WAIT`.
3. **Polling shape** — OpenAI `videos.retrieve`, Google `operations.get`,
   Volcengine `GET .../tasks/{id}`, DashScope `fetch`, xAI `video.get`,
   OpenRouter `GET /videos/{id}`, FLUX `text_to_image.get`. The adapter hides
   this behind `VideoProvider.get_video_task(task_id)`.
4. **First-frame image input** — Volcengine wants
   `{type:"image_url",image_url:{url},role:"first_frame"}`, OpenRouter wants
   `frame_images:[{...,frame_type:"first_frame"}]`, DashScope wants `img_url`,
   Google wants `source=GenerateVideosSource(image=...)`, xAI wants
   `image_url`. The unified `image` field (URL or data: URI) is mapped by the
   adapter to whichever the provider expects.
5. **Size representation** — `size="1024x1024"` (OpenAI/Volcengine),
   `"1024*1024"` (DashScope), `aspect_ratio="16:9"` (FLUX/Google/OpenRouter),
   `resolution="1k"|"2k"` (FLUX/OpenRouter), explicit `width`/`height`
   (Stability). The schema carries all of `size`, `width`, `height`,
   `aspect_ratio`, `resolution`; each adapter picks the field its provider
   accepts and derives the rest when only one is given.
6. **Auth headers** — `Authorization: Bearer` everywhere except Google
   (`x-goog-api-key`) and OpenRouter's extra attribution headers. Handled in
   each adapter's HTTP client.

## Front-end shapes exposed

The gateway serves **one** request/response shape per modality, all routed
through the same unified internal model so a request in one shape can be
fulfilled by any provider:

| Modality | Front-end shape |
|---|---|
| Image | **Gemini-compatible** `POST /v1/images` (`{model,input:string|parts[]}` → `{id}`) + `GET /v1/images/{id}` (`{id,model,status,steps:[{type:"model_output",content:[{type:"image",data,url,mime_type}|{type:"text",text}]}],output_image|output_image_url,error,usage}`) |
| Video | **Seedance-compatible** `POST /v1/videos` (`{model,content:[{type:text|image_url}],duration,resolution,ratio,camera_fixed,seed,watermark}` → `{id,status,...}`) + `GET /v1/videos/{id}` |
| Music | **Gemini Lyria 3-compatible** `POST /v1/music` (`{model,input:string|parts[],negative_prompt,duration,bpm,key_scale,key,scale,time_signature,vocal_language,audio_format,audio_quality,is_instrumental,generate_audio,seed,guidance_scale,n,callback_url,response_format}` → `{id}`) + `GET /v1/music/{id}` (`{id,model,status,steps:[{type:"model_output",content:[{type:"audio",data,url,mime_type}|{type:"text",text}]}],output_audio|output_audio_url,output_text,error,usage}`) |

Translators are pure functions (no I/O) living under `translators/`, one file
per (modality × shape) direction. Adding a second front-end shape (e.g. a
Replicate-style surface) is a new translator pair, not a provider change.

## Best-effort translation policy

When a field from the incoming shape has no exact home in the unified model or
the target provider, the gateway (in order): maps it to the closest knob,
else drops it into `extra` and passes it through to providers that accept
arbitrary keys, else silently drops it. Dropped fields are logged at DEBUG so
nothing disappears without a trace. We never reject a request for carrying an
unrecognised field.

## Music unification

Music mirrors the video design — a unified schema, a single front-end shape,
and per-provider adapters that hide wildly different upstream surfaces behind
one `create_music_task` / `get_music_task` poll interface.

### Unified music schema (`schemas/music.py`)

`UnifiedMusicRequest` carries `content: list[MusicContentPart]` (a
discriminated union over `type`) plus a flat set of generation knobs:

- **Text parts** (`{type:"text", text}`) — the prompt / lyrics. `prompt()`
  joins them with `\n`.
- **Audio parts** (`{type:"audio_url", audio_url:{url}, role}`) — reference or
  continuation audio. `role` is `reference_audio` (default) or
  `continuation_audio`; `reference_audios()` / `continuation_audio()` accessors
  filter by role.
- **Image parts** (`{type:"image_url", image_url:{url}, role:"reference_image"}`)
  — reference image for image-to-music.

Flat knobs: `negative_prompt`, `duration` (seconds), `bpm`, `key_scale`,
`key`, `scale`, `time_signature`, `vocal_language`, `audio_format`,
`audio_quality`, `is_instrumental`, `generate_audio`, `seed`,
`guidance_scale`, `n`, `callback_url`, `provider`, `extra`. Each provider reads
the subset it supports and ignores the rest.

`UnifiedMusicTask` holds the result as either `audio_urls: list[str]` (a
provider-returned URL) or `audio_b64: str` (inline base64 bytes), plus
`audio_media_type`, `lyrics`, `error`, `usage: MusicUsage` (carrying
`cost`/`duration`/`extra`), `raw`, and timestamps. `TaskStatus` has six values:
`pending`, `running`, `succeeded`, `failed`, `cancelled`, `expired`.

### Gemini Lyria 3 front-end (`translators/music/lyria_compat.py`)

The gateway exposes one music shape — the Gemini Lyria 3 Interactions surface —
so any backend is reachable through a Lyria-shaped request. `from_lyria` maps
the request; `to_lyria_create` / `to_lyria_task` map the response.

- **Request**: `{model, input, ...}`. `input` is a string (→ one text part) or
  a parts array. Lyria `text` parts → `text_part`; `image` parts (inline
  `{mime_type, data}`) are stashed into `extra["images"]` (not a content part —
  a gateway extension for providers that consume inline reference images);
  `audio_url`/`image_url` parts map to the native content parts. Flat knobs are
  copied when present. `response_format: {type:"audio"}` sets `audio_format="wav"`;
  any other string `type` sets `audio_format` to it; `response_format.quality`
  sets `audio_quality`. Any body key not in the known set and not `input` /
  `response_format` is funneled into `extra`.
- **Create response**: `to_lyria_create` returns `{"id": task.task_id}`.
- **Task response**: `to_lyria_task` builds `steps[0].content[]` blocks —
  `{type:"audio", data, mime_type}` for inline base64, `{type:"audio", url,
  mime_type}` for each URL, `{type:"text", text}` for lyrics — plus top-level
  `output_audio` / `output_audio_url` / `output_text` helpers, an `error` block
  when `task.error` is set, and a cost-only `usage` block.

### Sync vs async — the synthetic-task pattern

Three providers are synchronous upstream but the gateway's surface is
task-based, so they are wrapped as **synthetic in-memory tasks** (a module-level
`_MUSIC_TASKS` dict, single-process only). `create_music_task` mints a
gateway-local id and stores the request as `pending`; the first
`get_music_task` poll flips it to `running`, runs the blocking call, and moves
it to `succeeded`/`failed`:

| Provider | Upstream flow | Task id | Why synthetic |
|---|---|---|---|
| ElevenLabs | SDK `music.compose` streams bytes from one `POST /v1/music` | `el-{hex}` | no upstream task id |
| MiniMax | `POST /v1/music_generation` returns audio inline (`data.status` 1/2) | `mm-{hex}` | no upstream task id |
| Google Lyria | `POST …/models/{model}:predictInteractions?key=` returns audio inline | `lyria-{hex}` | no upstream task id |

The remaining three are **genuine two-phase async** — `create` returns an
upstream task id, `get` polls it:

| Provider | Create | Poll | Task id source |
|---|---|---|---|
| udioapi.pro | `POST /api/v2/generate` | `GET /api/v2/feed?workId=` | `workId` |
| Mureka | `POST /v1/song/generate` | `GET /v1/song/query/{task_id}` | `task_id`/`taskId`/`id` |
| ACE-Step 1.5 | `POST /release_task` | `POST /query_result` + `GET /v1/audio?path=` | `data.task_id` |

Sync-vs-async for the *client* is governed by `Prefer: respond-async`, the
`?wait` query param, or the `MUSIC_SYNC_DEFAULT` setting (default `true`). When
blocking, `MusicService._await_or_timeout` polls up to `MAX_SYNC_WAIT` (default
300s) at `POLL_INTERVAL` (default 2.0s).

### Audio delivery normalization

Providers return audio in four forms; each adapter normalizes to
`audio_b64` (inline base64) or `audio_urls` (a URL the client fetches):

| Provider | Upstream form | Normalized to |
|---|---|---|
| ElevenLabs | streamed bytes (`b''.join` of SDK chunks) | `audio_b64` |
| MiniMax | 24h URL (default, `output_format:"url"`) or hex bytes (`output_format:"hex"`, opt-in via `audio_format:"hex"`) | `audio_urls` (URL) or `audio_b64` (hex decoded → base64) |
| udioapi.pro | `track.audio_url` | `audio_urls` |
| Mureka | `audio_url`/`audioUrl`/`url` | `audio_urls` |
| ACE-Step 1.5 | file path → `GET /v1/audio?path=` bytes | `audio_b64` (falls back to `audio_urls` on fetch failure) |
| Google Lyria | inline base64 in `steps[].content[]` audio blocks | `audio_b64` |

`audio_media_type` is set from the requested `audio_format` (e.g. `mp3` →
`audio/mpeg`, `wav` → `audio/wav`) by ElevenLabs and MiniMax; udioapi.pro,
Mureka, and ACE-Step hard-code `audio/mpeg`, and Google Lyria hard-codes
`audio/wav`. (ACE-Step forwards `audio_format` to the upstream body but, since
the request is not retained across the async poll, reports `audio/mpeg` back to
the client regardless.)

### Best-effort knob mapping

Each adapter maps the unified knobs onto its provider's native fields and
forwards a curated set of provider-specific extras verbatim:

- **ElevenLabs**: `duration` → `music_length_ms` (clamped to
  [3000, 600000]); `is_instrumental` → `force_instrumental`; `audio_quality` →
  `output_format` (`{codec}_{quality}`); forwards `finetune_id`,
  `respect_sections_durations`, `store_for_inpainting`, `sign_with_c_2_pa`.
- **MiniMax**: prompt or `extra["lyrics"]` → `lyrics`/`prompt`;
  `is_instrumental`; `audio_quality` → `audio_setting{sample_rate,bitrate}`;
  `reference_audios()[0]` → `audio_url`; forwards `stream`,
  `lyrics_optimizer`, `audio_base64`, `cover_feature_id`.
- **udioapi.pro**: custom mode (`style`/`title`/`negative_prompt` present) vs
  inspiration mode (`gpt_description_prompt`); `is_instrumental` →
  `make_instrumental`; forwards `gender`, `style_weight`,
  `weirdness_constraint`, `audio_weight`.
- **Mureka**: lyrics vs prompt heuristic (multi-line or `[`-prefixed → lyrics);
  `negative_prompt` or `extra["style"]` → `tags`; `bpm`, `duration`;
  forwards `model_name`, `audio_config`, `voice_id`, `seed`.
- **ACE-Step 1.5**: `duration` → `audio_duration`; `bpm`; `audio_format`
  (`flac`/`mp3`/`opus`/`aac`/`wav`/`wav32`); `key_scale`, `time_signature`,
  `guidance_scale`, `seed` (sets `use_random_seed=False`), `vocal_language`;
  forwards `thinking`, `sample_mode`, `sample_query`, `use_format`,
  `inference_steps`, `batch_size`, `task_type`, `reference_audio_path`.
- **Google Lyria**: `content[]` text → input parts; `extra["images"]` →
  inline image parts; `negative_prompt`, `seed`, `guidance_scale`, `n` →
  config; `audio_format` → `response_format.type`; merges
  `extra["lyria_config"]`.

### Routing, env, and MCP

- **Routes** (`server/routes/music_routes.py`): `POST /v1/music` →
  `{"id":...}`; `GET /v1/music/{task_id}` → Lyria steps/content. Both require
  `get_api_key` auth; `GET` enforces `authorize_task`'s cross-tenant guard.
  The owning backend is recorded in the task store at create time so polls
  route correctly.
- **Config** (`config.py`): `Settings.music_sync_default` (env
  `MUSIC_SYNC_DEFAULT`, default `true`); `KeyConfig.default_music_tag` /
  `default_music_backend` (legacy env `DEFAULT_MUSIC_PROVIDER` sets the
  backend). Music providers register `*_MUSIC_*` env triples
  (`*_MUSIC_API_KEY` / `*_MUSIC_BASE_URL` / `*_MUSIC_MODEL`), falling back to
  the legacy un-split `*_API_KEY` / `*_BASE_URL` / `*_MODEL`. The shared
  `base_url` is resolved modality-first (`*_IMAGE_BASE_URL` → legacy
  `*_BASE_URL` → `*_VIDEO_BASE_URL` → `*_MUSIC_BASE_URL`, preferring image as
  the primary surface); the split `*_VIDEO_BASE_URL` and `*_MUSIC_BASE_URL`,
  when they differ from `base_url`, are separately recorded as
  `backend.extra["video_base_url"]` (consumed by the DashScope and Volcengine
  adapters as the async task endpoint) and `backend.extra["music_base_url"]`
  (consumed by the Google provider's `_music_base`). ACE-Step requires
  `ACESTEP_BASE_URL` (self-hosted, no default host); its API key is optional
  at the adapter level, but a backend only registers when at least one
  modality carries a key, so an unset `ACESTEP_API_KEY` (and no image/video
  key) means the ACE-Step backend is skipped entirely.
- **Registry** (`registry.py`): six `gateway-music-*` aliases —
  `gateway-music-lyria` → `(google, lyria-3)`,
  `gateway-music-elevenlabs` → `(elevenlabs, music_v2)`,
  `gateway-music-minimax` → `(minimax, music-3.0)`,
  `gateway-music-udio` → `(udioapi, chirp-v5)`,
  `gateway-music-mureka` → `(mureka, mureka-song-1)`,
  `gateway-music-acestep` → `(acestep, ace-step-1.5)`. An operator-pinned
  `extra["music_model"]` is appended to a provider's `music_models` at build
  time. `resolve()` gates music routing on `supports_music` and
  `key.default_music_backend` / `default_music_tag`.
- **MCP** (`server/mcp.py`): part of the gateway's **seven-tool** MCP surface —
  `create_music(model, input, wait=True, tag, backend)` and `get_music(id)`
  (the other five — `list_models`, `create_image`, `get_image`, `create_video`,
  `get_video` — mirror image/video/models) — using the same bearer-token auth
  and cross-tenant guard as the HTTP routes.
