# Provider-neutral speech and voice cloning

Research checked against official documentation on 2026-10-03. Speech is the
`audio` output modality. Music continues to use `/v1/music`. Voice cloning
creates reusable speaker identities through `/v1/voices`; a clone's selected
audio model identifies its hosting speech service.

## Provider research and mappings

| Adapter | Speech call and result | Clone operation | Controls and constraints |
|---|---|---|---|
| OpenAI | `POST /v1/audio/speech`; binary audio | Upload a consent recording to `/v1/audio/voice_consents`, then submit `audio_sample` and its consent id to `/v1/audio/voices` | 4,096 text characters; speed 0.25–4; MP3, Opus, AAC, FLAC, WAV, PCM. Separate instructions on GPT speech models. PCM is fixed 24 kHz mono S16LE. |
| ElevenLabs | `POST /v1/text-to-speech/{voice_id}`; binary audio | Multipart `/v1/voices/add`, with one or more `files`, returns a voice id and verification flag | Multilingual v2: 10,000 characters; Flash/Turbo v2.5: 40,000; v3: 5,000. Speed 0.7–1.2; language and seed where supported. Encoding combines codec, sample rate and bitrate. |
| MiniMax | `POST /v1/t2a_v2`; completed hex audio or a temporary URL | Upload a file with purpose `voice_clone`, then call `/v1/voice_clone` using the uploaded file id and a gateway-generated native voice id | Under 10,000 characters; speed 0.5–2; MP3, PCM, FLAC, WAV, Opus. Rates 8/16/22.05/24/32/44.1 kHz; MP3 bitrates 32/64/128/256 kbps. |
| Azure | Official Speech SDK `SpeechSynthesizer.speak_text_async` / `speak_ssml_async`; in-memory `audio_data` | Not exposed by this adapter | 64 KiB escaped UTF-8 synthesis message; speed 0.5–2; MP3, WAV, mono S16LE PCM, Ogg Opus. Locale control via SSML on compatible multilingual voices. |

OpenAI custom voices require approved account access, a matching speaker sample
and consent recording, samples no longer than 30 seconds, and uploads no larger
than 10 MiB. Its consent recording must use a published phrase in the declared
language. Gateway cloning is disabled for OpenAI until the operator sets
`voice_cloning_enabled: true`. This flag declares account eligibility; upstream
access and recording checks still apply. Legacy `tts-1`/`tts-1-hd` are served
for built-in speech but are excluded from cloning and cloned-voice reuse.
See the official [speech API](https://developers.openai.com/api/reference/resources/audio/subresources/speech/methods/create)
and [custom voice guide](https://developers.openai.com/api/docs/guides/custom-voices).

ElevenLabs Instant Voice Cloning accepts multiple uploaded samples. The gateway
exposes the verification flag and keeps the clone running until its background
monitor observes that verification is complete. Clients complete any required
verification through the upstream account; public polling stays non-blocking.
The gateway does not implement professional voice training. High bitrate and
sample-rate options may require a higher upstream plan. WAV delivery uses native
PCM samples wrapped in a mono 16-bit WAV header, preserving the requested rate.
Language control is rejected on Multilingual v2 because the upstream does not
support that control. See [TTS](https://elevenlabs.io/docs/api-reference/text-to-speech/convert),
[instant cloning](https://elevenlabs.io/docs/api-reference/voices/ivc/create),
[voice verification observations](https://elevenlabs.io/docs/api-reference/voices/get),
[model limits](https://elevenlabs.io/docs/overview/models), and
[speed control](https://elevenlabs.io/docs/help-center/product/core-capabilities/text-to-speech/can-i-change-the-pace-of-the-voice).
Turbo v2.5 is documented as deprecated in favor of Flash v2.5. Its 40,000
character ceiling here is inferred from the provider's stated functional
equivalence to Flash v2.5; Flash has an explicit published limit. The gateway's
fast alias targets Flash, and Turbo remains available for explicit requests.

MiniMax accepts one MP3/M4A/WAV cloning sample, 10–300 seconds long, at most
20 MB. Unused clones expire after seven days. Voice creation fees are incurred
on the first speech use; operator prices should account for that fee. Audio
URLs expire after 24 hours. A non-streaming incomplete response has no usable
provider job id and is treated as failure, never as permission to submit the
same synthesis again. See [speech](https://platform.minimax.io/docs/api-reference/speech-t2a-http),
[cloning](https://platform.minimax.io/docs/api-reference/voice-cloning-clone),
[file uploads](https://platform.minimax.io/docs/api-reference/file-management-upload),
and the machine-readable [speech](https://platform.minimax.io/docs/api-reference/speech/t2a/api/openapi.json)
and [cloning](https://platform.minimax.io/docs/api-reference/speech/voice-cloning/api/openapi.json) schemas.

Azure uses `type: azure` and the gateway model id `azure-tts`; the
`gateway-audio-azure` alias resolves to it. Authentication uses `api_key` plus
`extra.region`, or `api_key` plus a full SDK endpoint in `base_url`. The SDK
wait runs in a worker thread, and each request has its own config and
synthesizer with `audio_config=None`. Synthesis runs once through the shared
background monitor; cancellation, SDK exceptions and empty audio become
cached failures. Output includes mono channel count, the chosen sample rate,
audio duration, and input character usage.

The default voice is `en-US-AvaMultilingualNeural`, with operator aliases in
`extra.voice_presets`. `language` accepts locales such as `fr-FR`; changing a
voice's language requires multilingual support. Speed uses escaped SSML prosody. Spoken text remains text,
including angle brackets and ampersands. MP3 defaults to 24 kHz / 48 kbps and
supports 16 kHz at 32/64/128 kbps, 24 kHz at 48/96/160 kbps, or 48 kHz at
96/192 kbps. WAV/PCM support 8/16/22.05/24/44.1/48 kHz. Ogg Opus supports
16/24/48 kHz with fixed codec bitrate. WAV/PCM/Opus default to 24 kHz.
Separate instructions, seeds, remote delivery and gateway cloning are rejected
before synthesis. A deployed custom voice can be configured as a preset with
`extra.endpoint_id` and `extra.language` where required by its endpoint.
The native SDK supports HTTP outbound proxies on Linux and Windows via
`SpeechConfig.set_proxy`; other proxy schemes are rejected. For local Linux
installations, install the SDK's required OpenSSL and ALSA libraries.
See Microsoft's [synthesis guide](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-speech-synthesis),
[SSML controls](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-synthesis-markup-voice),
[output formats](https://learn.microsoft.com/en-us/python/api/azure-cognitiveservices-speech/azure.cognitiveservices.speech.speechsynthesisoutputformat),
[quotas](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-services-quotas-and-limits),
and [SDK installation](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/quickstarts/setup-platform?pivots=programming-language-python).

## Contract

Speech follows the existing strict `{model, input, parameters, routing,
metadata}` envelope. `input` is a nonempty ordered list of text parts joined
with newlines. `parameters.voice` defaults to the portable `default` preset.
Other neutral controls are `instructions`, `language` (ISO code), `speed`,
`file_format`, `sample_rate_hz`, `bitrate_kbps`, `delivery`, and `seed`. Every
requested control must fit the selected model/adapter. Unknown fields and
unsupported encoding combinations do not silently become upstream options.

Cloning uses the same envelope, plus required `consent`:

```json
{
  "input": [{"type": "audio", "uri": "data:audio/wav;base64,BASE64"}],
  "parameters": {"name": "Narrator"},
  "consent": {"granted": true},
  "metadata": {"speaker": "narrator"}
}
```

`parameters` can also include `description` and `remove_background_noise`
where supported. A consent-recording route includes `consent.recording_uri`
and `consent.language`. The attestation does not fabricate an upstream consent
id or bypass the provider's checks. Remote audio uses HTTP(S); inline audio
uses a validated base64 audio data URI. Local files and URI credentials are
rejected. Downloads have byte and redirect limits, resolve only public IPs,
pin the validated IP for the connection, preserve TLS verification/SNI, and
send no provider credentials. Providers enforce encoded sample-duration and
speaker requirements; the gateway checks transport size and format limits.

| Method | Path | Resource |
|---|---|---|
| POST | `/v1/audio` | Accept speech, return 202 and `aud_...` |
| GET | `/v1/audio/{audio_id}` | Read cached speech task |
| POST | `/v1/voices` | Accept cloning, return 202 and `voice_...` |
| GET | `/v1/voices/{voice_id}` | Read preset or cached clone task |
| GET | `/v1/voices` | List configured presets and this key's clones |
| POST | `/v1/audio/estimate` | Plan speech without upstream generation |
| POST | `/v1/voices/estimate` | Plan cloning without upstream uploads |

Successful clone ids can be used directly as `parameters.voice` once
`verification_required` is false. Native voice ids, consent ids, provider
names, and account ids stay internal. Retrieval, listing, reuse and estimates
enforce ownership and current backend authorization. A reusable clone binds
speech selection to its creating backend **and credential account**. Fallback
may choose another compatible model on that account; it cannot select another
speaker or credential. Presets can have different native speakers on each
backend and are deliberately distinct from owned clones.

Both creates support idempotency keys, including across REST and MCP. Replays
return the original create representation; changed bodies return 409. Reads
support ETags and 304. The monitor owns all generation and verification I/O,
and task GETs read cached snapshots. A synchronous adapter accepts a synthetic
task and executes the paid operation once in the background; failure is cached
terminally. Automatic submission fallback does not re-run a failed synthesis
after an asynchronous task has already been accepted.

MCP adds `create_audio`, `get_audio`, `create_voice`, `get_voice`, and
`list_voices`. `estimate_cost` supports `audio` and `voice`; cloning requires
the same `consent` and name as REST. `/v1/models` and `/v1/models/limits` accept
`modality=audio`, including documented cloning capability, formats, speeds,
text/sample limits and lifecycle.

## Operator configuration

```yaml
backends:
  - name: speech-primary
    type: openai
    api_key: ${OPENAI_AUDIO_API_KEY}
    tags: [speech]
    extra:
      audio_only: true
      voice_cloning_enabled: true # eligible accounts only
      voice_presets:
        narrator: alloy
        warm: coral
  - name: speech-fallback
    type: elevenlabs
    api_key: ${ELEVENLABS_AUDIO_API_KEY}
    tags: [speech]
    extra:
      audio_only: true
      voice_presets:
        narrator: JBFqnCBsd6RMkjVDRZzb
keys:
  - id: application
    key: ${GATEWAY_API_KEY}
    allow_tags: [speech]
    default_audio_backend: speech-primary
```

Named accounts use existing `credentials` configuration; their voice clones
remain private to that credential. `extra.audio_model` extends the served
catalogue. REST speech adapters accept `extra.audio_base_url` to override the
shared endpoint; Azure uses `base_url` for its SDK endpoint. The existing
outbound proxy applies to speech calls and sample downloads. Environment-only
deployments support `OPENAI_AUDIO_*`, `ELEVENLABS_AUDIO_*`, and `MINIMAX_AUDIO_*`
(`API_KEY`/`API_KEYS`, `BASE_URL`, `MODEL`, `OUTBOUND_PROXY`). A split audio key
creates a separate `<provider>-audio` backend; legacy shared provider keys also
serve audio. Set `DEFAULT_AUDIO_PROVIDER` to select the default backend.

Azure accepts `AZURE_AUDIO_API_KEY`/`AZURE_AUDIO_API_KEYS`,
`AZURE_AUDIO_REGION`, `AZURE_AUDIO_BASE_URL`, `AZURE_AUDIO_MODEL`, and
`AZURE_AUDIO_OUTBOUND_PROXY`, creating `azure-audio`. Shared `AZURE_API_KEY` /
`AZURE_API_KEYS`, `AZURE_REGION`, `AZURE_BASE_URL`, and `AZURE_OUTBOUND_PROXY`
are fallbacks. The official SDK quickstart variables `SPEECH_KEY`,
`SPEECH_REGION`, and `SPEECH_ENDPOINT` are also accepted. Named credentials can
override `extra.region` and `base_url` per account. Azure prices remain unknown
until the operator configures the applicable `azure-tts` character/request rate.

Prices vary by model and contract. Operators set `catalog.models.<id>.price`
using `per_character` and/or `per_request` for speech and `per_clone` for voice
creation, all in USD. `per_clone` is the complete clone price and does not add
the speech `per_request` charge. Clone estimates do not reuse speech character rates;
unknown-length speech is not priced using a fabricated duration. Estimates,
reservations, budget scopes, and terminal settlement use the existing ledger.
No unverified provider list prices are built in. Missing operation prices stay
unknown and are excluded by budgets/cost caps unless the existing unpriced
budget policy is enabled.

The default resource store, adapter tasks, monitors and voice mappings are
process-local. A durable production worker/store must persist speaker ownership,
native voice ids and credential affinity as well as generation task state.
Upstream voice/consent deletion and revocation remain upstream account
management operations; the current gateway exposes creation, observation,
listing and reuse. Real-time speech streaming, transcription, voice conversion,
multi-speaker dialogue and professional voice training are separate operations
that this implementation does not define.

## Verification

The automated suite exercises strict validation, REST/MCP parity, binary/hex/URL
delivery, WAV headers, multipart upload ordering, verification state, account
affinity, isolation between keys, idempotency, cached reads, failures that do not
repeat generation, capability routing, cost estimates and budget settlement.
Provider tests use mocked HTTP responses derived from the documented contracts;
they do not establish production account eligibility or audio quality. The live
speech smoke path can exercise configured credentials without creating a clone.
