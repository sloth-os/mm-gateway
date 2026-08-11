# Public API unification

This document defines the boundary between mm-gateway clients and provider
adapters. The public contract is intentionally not compatible with any one
provider. Image, video, and music remain separate resources because their
inputs, controls, outputs, and capability discovery evolve differently.

## Invariants

1. Image, video, and music use separate collection and item endpoints.
2. Creation is always asynchronous and returns `202 Accepted`.
3. Every request uses `{model, input, parameters, routing, metadata}`.
4. Every public request and response field has one canonical JSON shape.
5. `input` is a non-empty ordered typed-parts array; repeated media types are valid.
6. Public request envelopes, input parts, and parameter objects reject unknown
   fields; response objects permit additive extension members.
7. A backend wire field is never accepted just because an adapter supports it.
8. Adapters translate neutral concepts to native names, units, and structures.
9. Responses never use a provider's task id or result envelope.
10. Task ownership is bound to the authenticated API key.
11. Create retries are idempotent when the client supplies `Idempotency-Key`.

The public contract lives in `mm_gateway/schemas/api.py`. Translation to the
canonical internal models lives in `mm_gateway/translators/rest.py`. Provider
wire translation belongs in `mm_gateway/providers/`.

## Resources and lifecycle

The API contains three independent resource collections:

| Resource | Create | Retrieve | Public id prefix |
|---|---|---|---|
| Image | `POST /v1/images` | `GET /v1/images/{image_id}` | `img_` |
| Video | `POST /v1/videos` | `GET /v1/videos/{video_id}` | `vid_` |
| Music | `POST /v1/music` | `GET /v1/music/{music_id}` | `mus_` |

The create response is the initial task representation, not a second response
type. `Location` is its canonical URL, `Link` repeats that URL with `rel="self"`,
and `Retry-After` is present while work is non-terminal. Both create and get use
`Cache-Control: private, no-cache` and return an `ETag`. Poll clients may send
`If-None-Match`; an unchanged representation returns `304 Not Modified` without
a body. This keeps authenticated task data out of shared caches while preserving
normal HTTP conditional-request semantics.

`POST` accepts an optional `Idempotency-Key` (1-255 characters), scoped to the
authenticated key and modality collection. The first accepted request stores
its normalized request fingerprint and create representation. A retry with the
same body replays that `202` resource and sets `Idempotency-Replayed: true`; a
different body under the same key returns `409 idempotency_conflict`. Concurrent
creates sharing the same scope are serialized so only one reaches an adapter.
The bundled task store is process-local for development. A multi-worker or
multi-instance deployment must provide a shared durable implementation of the
same task/idempotency operations through `create_app(..., task_store=...)` so
these guarantees span every instance.

Status is one of `pending`, `running`, `succeeded`, `failed`, `cancelled`, or
`expired`. A task store maps the public id to the selected deployment target and
native task id, so provider identity and native ids never become API state.

There is no synchronous mode. Upstreams that respond synchronously are wrapped
as internal tasks and still follow the same public lifecycle.

## Request envelope

```json
{
  "model": "gateway-video-pro",
  "input": [{"type": "text", "text": "a prompt"}],
  "parameters": {},
  "routing": {"profile": "quality"},
  "metadata": {}
}
```

- `model` selects an id from `GET /v1/models`.
- `input` is a non-empty ordered typed-parts array containing semantic content
  consumed by the model.
- `parameters` controls generation and output using neutral names.
- `routing` is deployment policy, not a place for provider SDK options.
- `metadata` is opaque client state echoed in the task.

`routing` accepts one required `profile` when present. Profiles are
operator-defined, provider-neutral policies such as `quality`, `fast`, or `eu`.
The gateway maps a profile to internal backend tags; provider types and backend
instance names are never valid public selectors. An unavailable explicit
profile fails with `400` instead of silently falling back to a different route.

## Ordered multimodal input

`input` always uses a non-empty array, including the one-prompt case. This gives
the field one stable shape while preserving order, roles, repeated prompts, and
multiple media references. A text-only request is
`[{"type":"text","text":"..."}]`; a bare string is invalid.

### Images

Allowed parts are `text` and `image`. Every image has exactly one source:
`url` or base64 `data`; `mime_type` accompanies inline data.

The canonical image request preserves all parts. Adapters that support multiple
references receive them in order. Adapters with a smaller capability select the
supported subset at their own boundary.

### Videos

Allowed parts are:

| Type | Sources | Roles |
|---|---|---|
| `text` | `text` | none |
| `image` | `url` or base64 `data` | `first_frame`, `last_frame`, `reference_image` |
| `audio` | `url` or base64 `data` | `reference_audio` |
| `video` | `url` or base64 `data` | `reference_video` |

The REST translator represents inline media as data URLs internally. This keeps
the canonical content list ordered while allowing an adapter to translate the
same value to a native inline block, upload, URL field, or base64 field.

### Music

Allowed parts are:

| Type | Sources | Roles |
|---|---|---|
| `text` | `text` | none |
| `lyrics` | `text` | none |
| `image` | `url` or base64 `data` | `reference_image` |
| `audio` | `url` or base64 `data` | `reference_audio`, `continuation_audio` |

Multiple text and lyrics parts remain ordered. Text parts form the descriptive
prompt; lyrics parts are joined with newlines into the canonical lyrics field.

## Neutral parameter vocabulary

The public API describes intent. Internal models retain a few historical field
names, but those are not wire contracts.

### Image mapping

| Public concept | Canonical field | Adapter examples |
|---|---|---|
| `output_count` | `n` | image count, number of images, number of outputs |
| `delivery: base64` | `response_format: b64_json` | Accept header or native response mode |
| `file_format` | `output_format` | MIME type, output codec, multipart field |
| `compression` | `output_compression` | native image compression control |
| `inference_steps` | `num_inference_steps` | sampling steps |
| `guidance_scale` | `guidance_scale` | guidance or CFG scale |

Dimensions, aspect ratio, resolution, quality, style, seed, edit strength,
watermarking, and background retain their semantic names. Adapters convert
formats such as `WxH` versus `W*H` where necessary.

### Video mapping

| Public concept | Canonical field | Adapter examples |
|---|---|---|
| `duration_seconds` | `duration` | seconds, integer enum, or native duration unit |
| `aspect_ratio` | `ratio` | aspect ratio or ratio |
| `include_audio` | `generate_audio` | native audio-generation switch |
| `camera_motion: fixed` | `camera_fixed: true` | fixed-camera switch |
| `enhance_prompt` | `prompt_extend` | prompt expansion switch |
| `include_last_frame` | `return_last_frame` | last-frame result switch |
| `motion_intensity` | `motion_intensity` | Stability motion bucket |
| `frame_count` | `frame_count` | native frame count |
| `file_format` | `output_format` | video output format |
| `guidance_scale` | `guidance_scale` | Stability CFG scale |

Duration is exposed as `duration_seconds`. Adapters convert it to integer seconds,
string enum values, or other native units. Image roles are translated to a
source image, frame image list, first/last frame object, or typed content part.
Providers with multimodal reference support receive all audio, image, and video
parts; narrower providers consume the applicable roles.

### Music mapping

| Public concept | Canonical field | Adapter examples |
|---|---|---|
| `duration_seconds` | `duration` | seconds or a provider-native duration unit |
| `file_format` | `audio_format` | codec, container, response format |
| `sample_rate_hz` | same | composite output format or audio settings |
| `bitrate_kbps` | same | composite output format or bits-per-second setting |
| `instrumental` | `is_instrumental` | instrumental/force-instrumental switch |
| `output_count` | `n` | number of outputs or batch size |
| `enhance_lyrics` | same | native lyrics optimizer |
| `voice` | same | native voice id where supported |
| `vocal_gender` | same | native vocal gender control |
| `style_strength` | same | native style weight |
| `novelty` | same | native variation/weirdness control |
| `reference_audio_strength` | same | native audio influence weight |
| `inference_steps` | same | sampling/inference steps |
| `respect_section_durations` | same | section timing adherence |
| `provenance` | same | content provenance signing |

`key` and `scale` are separate concepts publicly. The ACE-Step adapter joins
them for its combined native field. Reference audio data is translated to a URL,
inline base64, or reference path according to the backend. Title, style, lyrics,
and voice semantics use native fields where available and are folded into a
descriptive prompt only where no native field exists.

Provider-only controls with no stable semantic abstraction remain unavailable
to clients. They may exist in operator configuration or internal compatibility
models, but are never accepted through `parameters`.

Request strictness catches misspellings and prevents adapter options from
leaking into the public contract. Response schemas intentionally remain open to
additive members, including RFC 9457 extensions. Clients must ignore response
fields they do not understand; removing or changing an existing field still
requires a new API version.

## Response normalization

Every task has:

```text
id, object, model, status, outputs, error, usage, metadata,
created_at, completed_at, links.self
```

Output types are intentionally separate:

- Image: `url` or base64 `data`, `mime_type`, optional `revised_prompt`.
- Video: `url`, optional `cover_url`, optional `mime_type`.
- Music: `url` or base64 `data`, optional `mime_type`; generated lyrics are a
  task-level field.

Usage exposes only shared concepts: cost, token counts, output count, and output
duration. Provider raw payloads and arbitrary usage dictionaries remain inside
the adapter layer.

Validation, authentication, authorization, conflict, routing, not-found,
timeout, and upstream HTTP failures use RFC 9457 Problem Details with
`application/problem+json`. The standard fields are `type`, `title`, `status`,
`detail`, and `instance`; stable `code`, `request_id`, and optional field-level
`errors` are gateway extensions. A task that reaches a failed terminal state
uses `{code, message}` inside the task resource. Provider identity and raw
upstream error details remain in server logs and are not fields in the public
problem schema.

## Model catalogue

`GET /v1/models` returns only `{id, object: "model", modality}` and may be
filtered by modality. Backend instance names, provider types, and underlying
alias targets are not returned. The authenticated key still scopes which
models are usable. The catalogue is privately cacheable for 60 seconds and
supports `ETag` / `If-None-Match` revalidation.

## MCP parity

MCP create tools use the same typed input models and parameter classes as REST.
They do not expose `wait`, raw `content`, provider config dictionaries, or
individual provider arguments. They accept `idempotency_key` with the same
scope and conflict behavior as HTTP creates. Create and get tools return the
same normalized task JSON and gateway-owned ids, and enforce the same task
ownership checks.

## Adding a capability

A new backend capability enters the public API only after all of these steps:

1. Name the user intent independently of the provider's wire field.
2. Add a typed, constrained field to the correct public parameter model.
3. Map it in `translators/rest.py` to a canonical internal field.
4. Translate that field in every adapter that can support it.
5. Add REST contract, adapter, and OpenAPI assertions.
6. Document unsupported or lossy mappings explicitly.

Unknown public fields stay forbidden. This prevents the public contract from
becoming the accidental union of thirteen provider SDKs.
