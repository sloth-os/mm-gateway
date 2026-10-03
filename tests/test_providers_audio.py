"""Documented speech and cloning wire contracts, with no live generation calls."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import wave

import httpx
import pytest

from mm_gateway.config import BackendConfig, KeyConfig, Settings
from mm_gateway.core.base import Provider
from mm_gateway.core.exceptions import ValidationError
from mm_gateway.providers.elevenlabs import ElevenLabsProvider
from mm_gateway.providers.minimax import MiniMaxProvider
from mm_gateway.providers.openai import OpenAIProvider
from mm_gateway.schemas.api import AudioRequest, VoiceCloneRequest
from mm_gateway.translators.rest import from_audio_request, from_voice_request


def provider(cls, **extra):
    # Speech uses its own REST transport; the image/music SDKs are unrelated.
    instance = cls.__new__(cls)
    Provider.__init__(instance, BackendConfig(name=cls.name, type=cls.name, api_key="provider-secret", extra=extra))
    return instance


def speech(model, **parameters):
    return from_audio_request(AudioRequest(model=model, input=[{"type": "text", "text": "hello"}], parameters=parameters))


def clone(model, *, consent_recording=False, **parameters):
    consent = {"granted": True}
    if consent_recording:
        consent.update(recording_uri="data:audio/wav;base64,Y29uc2VudA==", language="en")
    return from_voice_request(VoiceCloneRequest(
        model=model, input=[{"type": "audio", "uri": "data:audio/wav;base64,c2FtcGxl"}],
        parameters={"name": "Speaker", **parameters}, consent=consent,
    ))


async def test_openai_speech_binary_format_and_custom_voice_object(respx_mock):
    p = provider(OpenAIProvider)
    req = speech("gpt-4o-mini-tts", instructions="Speak warmly", speed=1.1, file_format="wav")
    req.native_voice_id = "private_voice"
    route = respx_mock.post("https://api.openai.com/v1/audio/speech").mock(
        return_value=httpx.Response(200, content=b"wave-bytes", headers={"content-type": "audio/wav"}))
    task = await p.create_audio_task(req)
    assert task.status == "pending" and not route.called
    results = await asyncio.gather(p.get_audio_task(task.task_id), p.get_audio_task(task.task_id))
    assert route.call_count == 1 and all(t.status == "succeeded" for t in results)
    assert json.loads(route.calls[0].request.content) == {
        "model": "gpt-4o-mini-tts", "input": "hello", "voice": {"id": "private_voice"},
        "response_format": "wav", "instructions": "Speak warmly", "speed": 1.1,
    }
    assert route.calls[0].request.headers["authorization"] == "Bearer provider-secret"
    assert results[0].outputs[0].mime_type == "audio/wav"
    assert base64.b64decode(results[0].outputs[0].uri.partition(",")[2]) == b"wave-bytes"


async def test_elevenlabs_pcm_wrapped_as_real_wav_and_neutral_controls(respx_mock):
    p = provider(ElevenLabsProvider, voice_presets={"narrator": "native/narrator"})
    route = respx_mock.post("https://api.elevenlabs.io/v1/text-to-speech/native%2Fnarrator").mock(
        return_value=httpx.Response(200, content=b"\x00\x00\x01\x00"))
    task = await p.create_audio_task(speech("eleven_flash_v2_5", voice="narrator", file_format="wav",
                                            sample_rate_hz=24000, speed=0.9, language="fr", seed=4))
    result = await p.get_audio_task(task.task_id)
    assert result.status == "succeeded", result.error
    request = route.calls[0].request
    assert request.url.params["output_format"] == "pcm_24000"
    assert request.headers["xi-api-key"] == "provider-secret" and "authorization" not in request.headers
    assert json.loads(request.content) == {"model_id": "eleven_flash_v2_5", "text": "hello",
                                           "language_code": "fr", "voice_settings": {"speed": 0.9}, "seed": 4}
    blob = base64.b64decode(result.outputs[0].uri.partition(",")[2])
    with wave.open(io.BytesIO(blob)) as audio:
        assert audio.getframerate() == 24000 and audio.getnchannels() == 1
        assert audio.readframes(2) == b"\x00\x00\x01\x00"


@pytest.mark.parametrize("delivery", ["inline", "remote"])
async def test_minimax_units_output_and_usage(respx_mock, delivery):
    p = provider(MiniMaxProvider)
    output = "https://cdn.example/audio.mp3" if delivery == "remote" else b"audio".hex()
    route = respx_mock.post("https://api.minimax.io/v1/t2a_v2").mock(return_value=httpx.Response(200, json={
        "data": {"audio": output, "status": 2}, "base_resp": {"status_code": 0},
        "extra_info": {"audio_length": 1500, "usage_characters": 5, "audio_sample_rate": 32000, "audio_channel": 1},
    }))
    task = await p.create_audio_task(speech("speech-2.8-hd", delivery=delivery, language="zh-yue", speed=1.5,
                                            file_format="mp3", sample_rate_hz=32000, bitrate_kbps=128))
    result = await p.get_audio_task(task.task_id)
    assert result.status == "succeeded"
    body = json.loads(route.calls[0].request.content)
    assert body["audio_setting"] == {"format": "mp3", "sample_rate": 32000, "bitrate": 128000}
    assert body["language_boost"] == "Chinese,Yue" and body["voice_setting"]["speed"] == 1.5
    assert body["stream"] is False and body["output_format"] == ("url" if delivery == "remote" else "hex")
    assert result.usage.duration_seconds == result.outputs[0].duration_seconds == 1.5
    assert result.outputs[0].uri == (output if delivery == "remote" else "data:audio/mpeg;base64,YXVkaW8=")


async def test_openai_clone_uploads_real_consent_and_sample(respx_mock):
    p = provider(OpenAIProvider, voice_cloning_enabled=True, __account_id="second")
    consent = respx_mock.post("https://api.openai.com/v1/audio/voice_consents").mock(
        return_value=httpx.Response(200, json={"id": "consent-private"}))
    voice = respx_mock.post("https://api.openai.com/v1/audio/voices").mock(
        return_value=httpx.Response(200, json={"id": "voice-private"}))
    task = await p.create_voice_task(clone("gpt-4o-mini-tts", consent_recording=True))
    assert not consent.called and not voice.called
    result = await p.get_voice_task(task.task_id)
    assert result.status == "succeeded" and result.native_voice_id == "voice-private" and result.account_id == "second"
    assert b'name="recording"' in consent.calls[0].request.content
    assert b'consent\r\n' in consent.calls[0].request.content
    assert b'name="language"\r\n\r\nen' in consent.calls[0].request.content
    assert b'name="audio_sample"' in voice.calls[0].request.content
    assert b'consent-private' in voice.calls[0].request.content
    assert b'sample\r\n' in voice.calls[0].request.content


async def test_elevenlabs_instant_clone_uses_multipart_multiple_files_and_verification(respx_mock):
    p = provider(ElevenLabsProvider)
    route = respx_mock.post("https://api.elevenlabs.io/v1/voices/add").mock(
        return_value=httpx.Response(200, json={"voice_id": "private", "requires_verification": True}))
    req = clone("eleven_multilingual_v2", description="Narrator", remove_background_noise=True)
    req.samples.append(req.samples[0].model_copy())
    task = await p.create_voice_task(req)
    result = await p.get_voice_task(task.task_id)
    assert result.status == "running" and result.verification_required is True and result.completed_at is None
    body = route.calls[0].request.content
    assert body.count(b'name="files"') == 2
    assert b'name="remove_background_noise"\r\n\r\ntrue' in body
    assert b'name="description"\r\n\r\nNarrator' in body
    checked = respx_mock.get("https://api.elevenlabs.io/v1/voices/private").mock(
        return_value=httpx.Response(200, json={"voice_verification": {"requires_verification": False, "is_verified": True}}))
    verified = await p.get_voice_task(task.task_id)
    assert verified.status == "succeeded" and verified.verification_required is False
    assert verified.completed_at is not None
    await p.get_voice_task(task.task_id)
    assert checked.call_count == route.call_count == 1


async def test_minimax_clone_upload_then_clone_has_private_valid_native_id(respx_mock):
    p = provider(MiniMaxProvider)
    upload = respx_mock.post("https://api.minimax.io/v1/files/upload").mock(return_value=httpx.Response(200, json={
        "file": {"file_id": "123"}, "base_resp": {"status_code": 0},
    }))
    route = respx_mock.post("https://api.minimax.io/v1/voice_clone").mock(return_value=httpx.Response(200, json={
        "base_resp": {"status_code": 0}, "input_sensitive": {"type": 0},
    }))
    task = await p.create_voice_task(clone("speech-2.8-hd", remove_background_noise=True))
    result = await p.get_voice_task(task.task_id)
    assert result.status == "succeeded" and result.native_voice_id.startswith("Gateway")
    assert b'name="purpose"\r\n\r\nvoice_clone' in upload.calls[0].request.content
    body = json.loads(route.calls[0].request.content)
    assert body == {"file_id": 123, "voice_id": result.native_voice_id, "need_noise_reduction": True}


@pytest.mark.parametrize("data", [
    {"data": {"status": 1, "audio": "deadbeef"}},
    {"data": None, "base_resp": {"status_code": 1002}},
    {"data": {"status": 2, "audio": "not-hex"}},
])
async def test_incomplete_or_error_response_is_terminal_and_never_resynthesized(respx_mock, data):
    p = provider(MiniMaxProvider)
    route = respx_mock.post("https://api.minimax.io/v1/t2a_v2").mock(return_value=httpx.Response(200, json=data))
    task = await p.create_audio_task(speech("speech-2.8-hd"))
    for _ in range(4):
        assert (await p.get_audio_task(task.task_id)).status == "failed"
    assert route.call_count == 1


async def test_partial_openai_clone_failure_does_not_reupload_consent_on_poll(respx_mock):
    p = provider(OpenAIProvider, voice_cloning_enabled=True)
    consent = respx_mock.post("https://api.openai.com/v1/audio/voice_consents").mock(
        return_value=httpx.Response(200, json={"id": "consent-private"}))
    voice = respx_mock.post("https://api.openai.com/v1/audio/voices").mock(return_value=httpx.Response(500))
    task = await p.create_voice_task(clone("gpt-4o-mini-tts", consent_recording=True))
    for _ in range(4):
        result = await p.get_voice_task(task.task_id)
        assert result.status == "failed" and "private" not in result.error
    assert consent.call_count == voice.call_count == 1


@pytest.mark.parametrize("cls,model,parameters", [
    (OpenAIProvider, "tts-1", {"instructions": "calm"}),
    (OpenAIProvider, "gpt-4o-mini-tts", {"delivery": "remote"}),
    (OpenAIProvider, "gpt-4o-mini-tts", {"sample_rate_hz": 44100}),
    (ElevenLabsProvider, "eleven_multilingual_v2", {"language": "en"}),
    (ElevenLabsProvider, "eleven_flash_v2_5", {"speed": 2}),
    (ElevenLabsProvider, "eleven_flash_v2_5", {"file_format": "aac"}),
    (MiniMaxProvider, "speech-2.8-hd", {"file_format": "pcm", "bitrate_kbps": 128}),
    (MiniMaxProvider, "speech-2.8-hd", {"sample_rate_hz": 32001}),
])
async def test_unsupported_controls_rejected_before_provider_calls(cls, model, parameters):
    p = provider(cls)
    with pytest.raises(ValidationError):
        await p.create_audio_task(speech(model, **parameters))


async def test_openai_clone_requires_account_opt_in_and_consent():
    with pytest.raises(ValidationError):
        await provider(OpenAIProvider).create_voice_task(clone("gpt-4o-mini-tts", consent_recording=True))
    with pytest.raises(ValidationError):
        await provider(OpenAIProvider, voice_cloning_enabled=True).create_voice_task(clone("gpt-4o-mini-tts"))


def test_speech_and_cloning_require_operation_specific_prices():
    from mm_gateway.models.limits import ModelLimits
    from mm_gateway.models.pricing import ModelPrice, estimate_cost
    limits = ModelLimits(modality="audio")
    assert estimate_cost(ModelPrice(per_second=0.1), modality="audio", limits=limits, input_characters=5) is None
    assert estimate_cost(ModelPrice(per_character=0.1, per_request=1), modality="audio", limits=limits,
                         voice_clone=True) is None
    assert estimate_cost(ModelPrice(per_character=0.1, per_request=1, per_clone=2), modality="audio",
                         limits=limits, input_characters=5) == 1.5
    assert estimate_cost(ModelPrice(per_character=0.1, per_request=1, per_clone=2), modality="audio",
                         limits=limits, voice_clone=True) == 2


def test_audio_env_config_and_model_catalogue(monkeypatch):
    monkeypatch.setenv("OPENAI_AUDIO_API_KEY", "audio-secret")
    monkeypatch.setenv("OPENAI_AUDIO_BASE_URL", "https://speech.example/v1")
    monkeypatch.setenv("OPENAI_AUDIO_MODEL", "new-speech")
    monkeypatch.setenv("DEFAULT_AUDIO_PROVIDER", "openai-audio")
    settings = Settings._from_legacy_env()
    config = next(b for b in settings.backends if b.name == "openai-audio")
    assert config.api_key == "audio-secret" and config.base_url == "https://speech.example/v1"
    assert config.extra["audio_model"] == "new-speech"
    assert settings.keys[0].default_audio_backend == "openai-audio"
    from mm_gateway.registry import Registry
    registry = Registry(Settings(backends=[config]))
    assert all(model["modality"] == "audio" for model in registry.list_public_models())
    assert registry.resolve("new-speech", modality="audio")[1] == "new-speech"


async def test_routing_and_estimates_use_model_limits_for_pinned_and_auto_audio(respx_mock):
    from mm_gateway.server.app import create_app
    app = create_app(Settings(backends=[BackendConfig(name="oa", type="openai", api_key="test"),
                                       BackendConfig(name="el", type="elevenlabs", api_key="test")],
                              keys=[KeyConfig(id="key", key="")]))
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway",
    ) as client:
        too_long = {"model": "gpt-4o-mini-tts", "input": [{"type": "text", "text": "x" * 4097}]}
        response = await client.post("/v1/audio", json=too_long)
        assert response.status_code == 422
        estimate = await client.post("/v1/audio/estimate", json=too_long)
        assert all(not item["admissible"] for item in estimate.json()["candidates"])
        auto = await client.post("/v1/audio/estimate", json={"input": too_long["input"]})
        assert auto.json()["model"].startswith("eleven_")
        assert respx_mock.calls.call_count == 0
