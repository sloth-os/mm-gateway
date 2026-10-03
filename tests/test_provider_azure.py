"""Azure SDK synthesis and gateway integration, without live service calls."""

from __future__ import annotations

import asyncio
import base64
import json
import threading
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock
from xml.etree import ElementTree as ET

import azure.cognitiveservices.speech as speechsdk
import httpx
import pytest

from mm_gateway.config import BackendConfig, Settings
from mm_gateway.core.base import AudioProvider, VoiceCloneProvider
from mm_gateway.core.exceptions import ProviderNotConfiguredError, ValidationError
from mm_gateway.providers.azure import AzureProvider
from mm_gateway.registry import Registry
from mm_gateway.schemas.api import AudioRequest
from mm_gateway.server.app import create_app
from mm_gateway.translators.rest import from_audio_request


@pytest.fixture
def sdk_mock(monkeypatch):
    # Use real SDK config, enums and native format mapping; replace only the
    # synthesizer's network operation and spy on the config constructor.
    result = SimpleNamespace(reason=speechsdk.ResultReason.SynthesizingAudioCompleted,
                             audio_data=b"azure-audio", audio_duration=timedelta(seconds=1.5))
    future = SimpleNamespace(get=Mock(return_value=result))
    synth = SimpleNamespace(speak_text_async=Mock(return_value=future),
                            speak_ssml_async=Mock(return_value=future))
    constructor = Mock(return_value=synth)
    config = Mock(wraps=speechsdk.SpeechConfig)
    monkeypatch.setattr(speechsdk, "SpeechSynthesizer", constructor)
    monkeypatch.setattr(speechsdk, "SpeechConfig", config)
    return SimpleNamespace(result=result, future=future, synth=synth,
                           constructor=constructor, config=config)


def backend(**kwargs):
    return BackendConfig(name="azure-speech", type="azure", api_key="azure-secret",
                         extra={"region": "eastus", **kwargs})


def speech(text="hello", **parameters):
    return from_audio_request(AudioRequest(model="azure-tts", input=[{"type": "text", "text": text}],
                                          parameters=parameters))


async def test_sdk_plain_text_and_concurrent_polls_execute_once(sdk_mock):
    provider = AzureProvider(backend())
    assert isinstance(provider, AudioProvider) and not isinstance(provider, VoiceCloneProvider)
    assert provider.supports_audio and not any([provider.supports_image, provider.supports_video, provider.supports_music])
    text = '<voice name="injected">Hello & goodbye</voice>'
    task = await provider.create_audio_task(speech(text))
    assert task.status == "pending" and not sdk_mock.constructor.called
    results = await asyncio.gather(provider.get_audio_task(task.task_id), provider.get_audio_task(task.task_id))
    assert all(result.status == "succeeded" for result in results)
    sdk_mock.config.assert_called_once_with(subscription="azure-secret", region="eastus")
    sdk_mock.synth.speak_text_async.assert_called_once_with(text)
    sdk_mock.synth.speak_ssml_async.assert_not_called()
    sdk_mock.future.get.assert_called_once()
    config = sdk_mock.constructor.call_args.kwargs["speech_config"]
    assert config.speech_synthesis_voice_name == "en-US-AvaMultilingualNeural"
    assert config.speech_synthesis_output_format_string == "audio-24khz-48kbitrate-mono-mp3"
    assert sdk_mock.constructor.call_args.kwargs["audio_config"] is None
    output = results[0].outputs[0]
    assert output.mime_type == "audio/mpeg" and output.sample_rate_hz == 24000 and output.channels == 1
    assert output.duration_seconds == results[0].usage.duration_seconds == 1.5
    assert results[0].usage.input_characters == len(text)
    assert base64.b64decode(output.uri.partition(",")[2]) == b"azure-audio"
    assert await provider.get_audio_task(task.task_id) == results[0]
    assert sdk_mock.constructor.call_count == 1


async def test_ssml_locale_speed_and_voice_attributes_are_escaped(sdk_mock):
    voice_id = 'en-US-AvaMultilingualNeural" & <invalid>'
    provider = AzureProvider(backend(voice_presets={"narrator": voice_id}))
    text = 'こんにちは & <audio src="https://example.test/secret"/>\n"Bonjour"'
    task = await provider.create_audio_task(speech(text, voice="narrator", language="fr-FR", speed=1.25))
    assert (await provider.get_audio_task(task.task_id)).status == "succeeded"
    sdk_mock.synth.speak_text_async.assert_not_called()
    xml = sdk_mock.synth.speak_ssml_async.call_args.args[0]
    root = ET.fromstring(xml)
    ns = {"s": "http://www.w3.org/2001/10/synthesis"}
    voice = root.find("s:voice", ns)
    assert voice.attrib["name"] == voice_id
    assert root.attrib["{http://www.w3.org/XML/1998/namespace}lang"] == "en-US"
    assert voice.find("s:lang", ns).attrib["{http://www.w3.org/XML/1998/namespace}lang"] == "fr-FR"
    prosody = voice.find("s:lang/s:prosody", ns)
    assert prosody.attrib == {"rate": "1.25"} and prosody.text == text
    assert root.find(".//s:audio", ns) is None


async def test_native_locale_does_not_wrap_monolingual_voice_in_lang(sdk_mock):
    provider = AzureProvider(backend(voice_presets={"default": "fr-FR-DeniseNeural"}))
    task = await provider.create_audio_task(speech(language="fr-FR", speed=0.5))
    assert (await provider.get_audio_task(task.task_id)).status == "succeeded"
    xml = sdk_mock.synth.speak_ssml_async.call_args.args[0]
    assert 'xml:lang="fr-FR"' in xml and "<lang" not in xml and 'rate="0.5"' in xml


async def test_multilingual_voice_honors_explicit_native_locale(sdk_mock):
    provider = AzureProvider(backend())
    task = await provider.create_audio_task(speech(language="en-US"))
    assert (await provider.get_audio_task(task.task_id)).status == "succeeded"
    assert '<lang xml:lang="en-US">' in sdk_mock.synth.speak_ssml_async.call_args.args[0]


async def test_monolingual_voice_rejects_different_language_before_sdk_calls(sdk_mock):
    provider = AzureProvider(backend(voice_presets={"default": "en-US-JennyNeural"}))
    with pytest.raises(ValidationError, match="language locale"):
        await provider.create_audio_task(speech(language="fr-FR"))
    sdk_mock.constructor.assert_not_called()


@pytest.mark.parametrize(("parameters", "expected"), [
    ({"file_format": "mp3", "sample_rate_hz": 16000, "bitrate_kbps": 128}, "audio-16khz-128kbitrate-mono-mp3"),
    ({"file_format": "mp3", "bitrate_kbps": 192}, "audio-48khz-192kbitrate-mono-mp3"),
    ({"file_format": "mp3", "sample_rate_hz": 48000}, "audio-48khz-96kbitrate-mono-mp3"),
    ({"file_format": "wav", "sample_rate_hz": 22050}, "riff-22050hz-16bit-mono-pcm"),
    ({"file_format": "wav", "sample_rate_hz": 44100}, "riff-44100hz-16bit-mono-pcm"),
    ({"file_format": "pcm", "sample_rate_hz": 8000}, "raw-8khz-16bit-mono-pcm"),
    ({"file_format": "pcm", "sample_rate_hz": 48000}, "raw-48khz-16bit-mono-pcm"),
    ({"file_format": "opus", "sample_rate_hz": 16000}, "ogg-16khz-16bit-mono-opus"),
    ({"file_format": "opus", "sample_rate_hz": 48000}, "ogg-48khz-16bit-mono-opus"),
])
async def test_sdk_encoding_mapping(sdk_mock, parameters, expected):
    provider = AzureProvider(backend())
    task = await provider.create_audio_task(speech(**parameters))
    result = await provider.get_audio_task(task.task_id)
    assert result.status == "succeeded", result.error
    config = sdk_mock.constructor.call_args.kwargs["speech_config"]
    assert config.speech_synthesis_output_format_string == expected
    assert result.outputs[0].sample_rate_hz == (parameters.get("sample_rate_hz") or 48000)
    assert result.outputs[0].mime_type == {"mp3": "audio/mpeg", "wav": "audio/wav", "pcm": "audio/pcm", "opus": "audio/ogg"}[parameters["file_format"]]


@pytest.mark.parametrize("parameters", [
    {"file_format": "flac"}, {"file_format": "aac"}, {"sample_rate_hz": 32000},
    {"file_format": "opus", "sample_rate_hz": 44100}, {"file_format": "opus", "bitrate_kbps": 64},
    {"file_format": "wav", "bitrate_kbps": 128}, {"sample_rate_hz": 24000, "bitrate_kbps": 128},
    {"delivery": "remote"}, {"instructions": "speak warmly"}, {"seed": 0},
    {"speed": 0.25}, {"speed": 4}, {"language": "fr"}, {"voice": "missing"},
])
async def test_unsupported_controls_fail_before_sdk_calls(sdk_mock, parameters):
    with pytest.raises(ValidationError):
        await AzureProvider(backend()).create_audio_task(speech(**parameters))
    sdk_mock.config.assert_not_called()
    sdk_mock.constructor.assert_not_called()


async def test_utf8_and_xml_escaping_count_toward_message_limit(sdk_mock):
    provider = AzureProvider(backend())
    assert provider.audio_request_error(speech("a" * 64000), "azure-tts") is None
    for text in ("你" * 22000, "&" * 14000, "a" * 65536):
        with pytest.raises(ValidationError, match="64 KiB"):
            await provider.create_audio_task(speech(text, speed=1))
    sdk_mock.constructor.assert_not_called()


@pytest.mark.parametrize("failure", ["canceled", "unexpected", "empty", "exception"])
async def test_failures_are_terminal_and_never_resubmit(sdk_mock, failure):
    if failure == "canceled":
        sdk_mock.result.reason = speechsdk.ResultReason.Canceled
        sdk_mock.result.cancellation_details = "secret upstream credential"
    elif failure == "unexpected":
        sdk_mock.result.reason = speechsdk.ResultReason.SynthesizingAudioStarted
    elif failure == "empty":
        sdk_mock.result.audio_data = b""
    else:
        sdk_mock.future.get.side_effect = RuntimeError("secret upstream credential")
    provider = AzureProvider(backend())
    task = await provider.create_audio_task(speech())
    result = await provider.get_audio_task(task.task_id)
    assert result.status == "failed" and result.error == "Speech generation failed."
    assert not result.outputs and result.completed_at is not None
    assert await provider.get_audio_task(task.task_id) == result
    sdk_mock.synth.speak_text_async.assert_called_once()


async def test_sdk_wait_runs_outside_event_loop(sdk_mock):
    started, release = threading.Event(), threading.Event()
    main_thread = threading.get_ident()

    def blocking_get():
        assert threading.get_ident() != main_thread
        started.set()
        assert release.wait(timeout=5)
        return sdk_mock.result

    sdk_mock.future.get.side_effect = blocking_get
    provider = AzureProvider(backend())
    task = await provider.create_audio_task(speech())
    running = asyncio.create_task(provider.get_audio_task(task.task_id))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        # A synchronous callback on this loop must still run while SDK .get()
        # is blocked in its worker thread.
        tick = asyncio.get_running_loop().create_future()
        asyncio.get_running_loop().call_soon(tick.set_result, True)
        assert await asyncio.wait_for(tick, timeout=0.5)
        assert not running.done()
    finally:
        release.set()
        result = await running
    assert result.status == "succeeded"


async def test_custom_endpoint_and_http_proxy_use_sdk_config(sdk_mock, monkeypatch):
    proxy = Mock()
    monkeypatch.setattr(speechsdk.SpeechConfig._mock_wraps, "set_proxy", proxy)
    cfg = BackendConfig(name="custom", type="azure", api_key="custom-key",
                        base_url="wss://custom.example/tts/cognitiveservices/websocket/v1",
                        extra={"endpoint_id": "deployment-id", "language": "de-DE",
                               "voice_presets": {"default": "de-DE-KatjaNeural"},
                               "outbound_proxy": "http://user%40corp:p%3Ass@proxy.example:3128"})
    provider = AzureProvider(cfg)
    task = await provider.create_audio_task(speech(speed=1))
    assert (await provider.get_audio_task(task.task_id)).status == "succeeded"
    sdk_mock.config.assert_called_once_with(subscription="custom-key", endpoint=cfg.base_url)
    config = sdk_mock.constructor.call_args.kwargs["speech_config"]
    assert config.endpoint_id == "deployment-id" and config.speech_synthesis_language == "de-DE"
    assert config.speech_synthesis_voice_name == "de-DE-KatjaNeural"
    proxy.assert_called_once_with("proxy.example", 3128, "user@corp", "p:ss")


@pytest.mark.parametrize("proxy", ["socks5://proxy:1080", "https://proxy:3128", "http://proxy/path", "http://proxy:invalid"])
def test_unsupported_proxy_config_is_rejected(proxy):
    with pytest.raises(ValueError):
        AzureProvider(backend(outbound_proxy=proxy))


@pytest.mark.parametrize("cfg", [
    BackendConfig(name="missing", type="azure", extra={"region": "eastus"}),
    BackendConfig(name="missing", type="azure", api_key="key"),
])
def test_missing_credentials_or_location_skip_backend(cfg):
    with pytest.raises(ProviderNotConfiguredError):
        AzureProvider(cfg)
    assert Registry(Settings(backends=[cfg])).backends == {}


def test_explicit_proxy_on_macos_is_rejected(monkeypatch):
    monkeypatch.setattr("mm_gateway.providers.azure.sys.platform", "darwin")
    with pytest.raises(ValueError, match="macOS"):
        AzureProvider(backend(outbound_proxy="http://proxy:3128"))


async def test_named_accounts_keep_credentials_endpoints_and_configs_separate(sdk_mock):
    cfg = BackendConfig(name="azure-pool", type="azure", extra={"region": "eastus"}, credentials=[
        {"id": "first", "api_key": "first-key"},
        {"id": "second", "api_key": "second-key", "base_url": "wss://second.example/tts",
         "extra": {"voice_presets": {"default": "fr-FR-DeniseNeural"}}},
    ])
    registry = Registry(Settings(backends=[cfg]))
    providers = dict(registry.accounts_of("azure-pool"))
    for provider in providers.values():
        task = await provider.create_audio_task(speech())
        assert (await provider.get_audio_task(task.task_id)).status == "succeeded"
    assert sdk_mock.config.call_args_list[0].kwargs == {"subscription": "first-key", "region": "eastus"}
    assert sdk_mock.config.call_args_list[1].kwargs == {"subscription": "second-key", "endpoint": "wss://second.example/tts"}
    configs = [call.kwargs["speech_config"] for call in sdk_mock.constructor.call_args_list]
    assert configs[0] is not configs[1]
    assert [config.speech_synthesis_voice_name for config in configs] == ["en-US-AvaMultilingualNeural", "fr-FR-DeniseNeural"]


@pytest.fixture
def azure_env(monkeypatch):
    for prefix in ("AZURE_AUDIO_", "AZURE_", "SPEECH_"):
        for suffix in ("API_KEY", "API_KEYS", "KEY", "REGION", "BASE_URL", "ENDPOINT", "MODEL", "OUTBOUND_PROXY"):
            monkeypatch.delenv(prefix + suffix, raising=False)
    return monkeypatch


@pytest.mark.parametrize("env", [
    {"AZURE_AUDIO_API_KEY": "key", "AZURE_AUDIO_REGION": "eastus"},
    {"AZURE_API_KEY": "key", "AZURE_REGION": "eastus"},
    {"SPEECH_KEY": "key", "SPEECH_REGION": "eastus"},
    {"SPEECH_KEY": "key", "SPEECH_ENDPOINT": "wss://custom.example/tts"},
    {"AZURE_AUDIO_API_KEYS": "key,second-key", "AZURE_AUDIO_REGION": "eastus"},
    {"AZURE_API_KEYS": "key,second-key", "AZURE_REGION": "eastus"},
])
def test_env_configuration_builds_resolvable_audio_backend(azure_env, env):
    for key, value in env.items():
        azure_env.setenv(key, value)
    azure_env.setenv("DEFAULT_AUDIO_PROVIDER", "azure-audio")
    settings = Settings._from_legacy_env()
    cfg = next(b for b in settings.backends if b.name == "azure-audio")
    registry = Registry(Settings(backends=[cfg], keys=settings.keys))
    assert registry.resolve("gateway-audio-azure", modality="audio")[1] == "azure-tts"
    assert settings.keys[0].default_audio_backend == "azure-audio"
    assert {model["modality"] for model in registry.list_public_models()} == {"audio"}


def test_split_azure_env_takes_precedence_over_shared_credentials(azure_env):
    azure_env.setenv("AZURE_AUDIO_API_KEY", "audio-key")
    azure_env.setenv("AZURE_AUDIO_REGION", "eastus")
    azure_env.setenv("AZURE_AUDIO_BASE_URL", "wss://audio.example/tts")
    azure_env.setenv("AZURE_AUDIO_MODEL", "new-azure-speech")
    azure_env.setenv("AZURE_AUDIO_OUTBOUND_PROXY", "http://audio-proxy:3128")
    azure_env.setenv("AZURE_API_KEYS", "shared-key,shared-key-2")
    azure_env.setenv("AZURE_REGION", "westus")
    azure_env.setenv("AZURE_BASE_URL", "wss://shared.example/tts")
    azure_env.setenv("AZURE_OUTBOUND_PROXY", "http://shared-proxy:3128")
    cfg = next(b for b in Settings._from_legacy_env().backends if b.name == "azure-audio")
    assert cfg.api_key == "audio-key" and not cfg.credentials
    assert cfg.extra["region"] == "eastus" and cfg.base_url == "wss://audio.example/tts"
    assert cfg.extra["outbound_proxy"] == "http://audio-proxy:3128"
    registry = Registry(Settings(backends=[cfg]))
    assert registry.resolve("new-azure-speech", modality="audio")[1] == "new-azure-speech"


async def test_rest_create_estimate_presets_and_limits_use_azure_sdk(sdk_mock):
    settings = Settings._from_yaml("""
backends:
  - name: azure-speech
    type: azure
    api_key: azure-secret
    tags: [speech]
    extra:
      region: eastus
      voice_presets:
        narrator: en-US-JennyNeural
keys:
  - id: app
    key: gateway-key
    allow_tags: [speech]
    default_audio_backend: azure-speech
catalog:
  models:
    azure-tts:
      price: {per_character: 0.001}
""")
    settings = replace(settings, poll_interval=0.001)
    app = create_app(settings)
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway",
        headers={"authorization": "Bearer gateway-key"},
    ) as client:
        models = (await client.get("/v1/models/limits?modality=audio")).json()["data"]
        limits = next(model["limits"] for model in models if model["id"] == "azure-tts")
        assert limits["supports_voice_cloning"] is False and limits["supported_file_formats"] == ["mp3", "wav", "pcm", "opus"]
        voices = (await client.get("/v1/voices")).json()["data"]
        assert {voice["id"] for voice in voices} == {"default", "narrator"}
        assert "en-US" not in json.dumps(voices)
        body = {"model": "gateway-audio-azure", "input": [{"type": "text", "text": "hello"}],
                "parameters": {"voice": "narrator", "file_format": "wav", "sample_rate_hz": 16000}}
        estimate = await client.post("/v1/audio/estimate", json=body)
        assert estimate.status_code == 200 and estimate.json()["model"] == "azure-tts"
        sdk_mock.constructor.assert_not_called()
        created = await client.post("/v1/audio", json=body, headers={"idempotency-key": "azure-request"})
        assert created.status_code == 202, created.text
        from tests.test_audio_api import terminal
        completed = (await terminal(client, created.headers["location"])).json()
        assert completed["status"] == "succeeded", completed
        assert completed["outputs"][0]["mime_type"] == "audio/wav"
        assert completed["usage"]["cost"] == 0.005 and completed["usage"]["input_characters"] == 5
        replay = await client.post("/v1/audio", json=body, headers={"idempotency-key": "azure-request"})
        assert replay.json() == created.json()
        assert sdk_mock.constructor.call_count == 1
        assert sdk_mock.constructor.call_args.kwargs["speech_config"].speech_synthesis_voice_name == "en-US-JennyNeural"
        unsupported = {**body, "parameters": {"file_format": "wav", "bitrate_kbps": 128}}
        assert (await client.post("/v1/audio", json=unsupported)).status_code == 422
        invalid_estimate = (await client.post("/v1/audio/estimate", json=unsupported)).json()
        assert not any(candidate["admissible"] for candidate in invalid_estimate["candidates"])
        clone = {"model": "azure-tts", "input": [{"type": "audio", "uri": "data:audio/wav;base64,AAAA"}],
                 "parameters": {"name": "Speaker"}, "consent": {"granted": True}}
        assert (await client.post("/v1/voices", json=clone)).status_code == 422
        assert sdk_mock.constructor.call_count == 1
