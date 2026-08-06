# Unification design — API commonalities & differences

This document records how the eight provider APIs differ, the common spine we
extracted, and the two front-end shapes the gateway exposes. It is the
rationale behind `mm_gateway/schemas/` and `mm_gateway/translators/`.

## Provider matrix

| Provider | Image | Video | Async video? | Image delivery | Auth env |
|---|:--:|:--:|:--:|---|---|
| OpenAI (Sora/DALL·E) | ✅ | ✅ | task+poll (retrieve/download) | `data[].b64_json` (gpt-image) / `url` (dall-e) | `OPENAI_API_KEY` |
| Google (Imagen/Veo) | ✅ | ✅ | LRO op+poll | bytes (`image_bytes`) | `GOOGLE_API_KEY` |
| xAI (Grok Imagine) | ✅ | ✅ (gRPC) | deferred+poll | `url`/`base64`/bytes | `XAI_API_KEY` |
| Volcengine Ark (Seedream/Seedance) | ✅ | ✅ | task+poll | `data[].url|b64_json` | `ARK_API_KEY` |
| FLUX.2 (runapi) | ✅ | ❌ | n/a | `images[].url` | `RUNAPI_API_KEY` |
| OpenRouter (unified) | ✅ | ✅ | job+poll | `data[].b64_json` | `OPENROUTER_API_KEY` |
| DashScope (Wanx/Wan) | ✅ | ✅ | task+poll | `results[].url` / `video_url` | `DASHSCOPE_API_KEY` |
| Stability (SD/SVD) | ✅ | ✅ | sync image / async video | base64 / MP4 bytes | `STABILITY_API_KEY` |

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
2. **Sync vs async video** — image gen is synchronous everywhere; video gen is
   *always* async (task + poll) on every provider. The unified video API
   therefore exposes a task lifecycle (`pending → running → succeeded | failed`)
   and the gateway optionally blocks (sync-style) up to `MAX_SYNC_WAIT`.
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

The gateway serves **two** request/response shapes for each modality, mapped
through the same unified internal model so a request in one shape can be
fulfilled by any provider:

| Modality | Shape A | Shape B |
|---|---|---|
| Image | **OpenAI-compatible** `POST /v1/images/generations` (`{model,prompt,n,size,response_format,...}` → `{created,data:[{url|b64_json,revised_prompt}]}`) | **OpenRouter-compatible** `POST /api/v1/images` (`{model,prompt,aspect_ratio,n,output_format,quality,resolution,size,seed,background,input_references,provider}` → `{created,data:[{b64_json,media_type}],usage}`) |
| Video | **Seedance-compatible** `POST /v1/videos` (`{model,content:[{type:text|image_url}],duration,resolution,ratio,camera_fixed,seed,watermark}` → `{id,status,...}`) + `GET /v1/videos/{id}` | **OpenRouter-compatible** `POST /api/v1/videos` (`{model,prompt,aspect_ratio,resolution,duration,frame_images,seed,callback_url}` → `{id,status,unsigned_urls}`) + `GET /api/v1/videos/{id}` |

Translators are pure functions (no I/O) living under `translators/`, one file
per (modality × shape) direction. Adding a third front-end shape (e.g. a
Replicate-style surface) is a new translator pair, not a provider change.

## Best-effort translation policy

When a field from the incoming shape has no exact home in the unified model or
the target provider, the gateway (in order): maps it to the closest knob,
else drops it into `extra` and passes it through to providers that accept
arbitrary keys, else silently drops it. Dropped fields are logged at DEBUG so
nothing disappears without a trace. We never reject a request for carrying an
unrecognised field.
