"""Tests for the music provider adapters' request/response mapping.

The REST adapters (MiniMax, udioapi, Mureka, ACE-Step) talk to their upstreams
over ``httpx``, so we mount an ``httpx.MockTransport`` onto each provider's
client and assert the request bodies they build + the responses they map back —
no network. ElevenLabs is SDK-driven (its ``_compose`` stream drain is covered by
the streaming fix in the adapter itself) and Google Lyria's output extraction is
unit-tested directly.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

import httpx
import pytest

from mm_gateway.config import BackendConfig
from mm_gateway.core.exceptions import ProviderNotConfiguredError
from mm_gateway.schemas.music import UnifiedMusicRequest, audio_part, text_part

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _backend(name: str, *, api_key: str = "k", base_url: str | None = None) -> BackendConfig:
    kw: dict[str, Any] = {"name": name, "type": name, "api_key": api_key}
    if base_url is not None:
        kw["base_url"] = base_url
    return BackendConfig(**kw)


def _mount(provider, handler, *, base_url: str) -> None:
    """Replace the provider's httpx client with a MockTransport-driven one."""
    provider._client = httpx.AsyncClient(
        base_url=base_url, transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer k"},
    )


def _body(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.content) if request.content else {}


def _req(prompt: str = "a happy song", **kw) -> UnifiedMusicRequest:
    return UnifiedMusicRequest(model=kw.pop("model", "m"), content=[text_part(prompt)], **kw)


def test_elevenlabs_translates_neutral_audio_quality() -> None:
    from mm_gateway.providers.elevenlabs import ElevenLabsProvider

    request = _req(audio_format="wav", sample_rate_hz=48000, bitrate_kbps=192)
    assert ElevenLabsProvider._output_format(request) == "wav_48000_192"


# --------------------------------------------------------------------------- #
# MiniMax — synchronous, hex/url audio, in-memory synthetic task
# --------------------------------------------------------------------------- #


def test_minimax_requires_api_key() -> None:
    from mm_gateway.providers.minimax import MiniMaxProvider
    with pytest.raises(ProviderNotConfiguredError):
        MiniMaxProvider(BackendConfig(name="minimax", type="minimax"))


def test_minimax_builds_body_and_decodes_hex_audio() -> None:
    from mm_gateway.providers.minimax import _MUSIC_TASKS, MiniMaxProvider
    _MUSIC_TASKS.clear()
    p = MiniMaxProvider(_backend("minimax"))

    captured: dict[str, Any] = {}

    audio_hex = b"\x49\x44\x33\x03".hex()  # 'ID3' + 0x03

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = _body(request)
        return httpx.Response(200, json={
            "base_resp": {"status_code": 0},
            "data": {"status": 2, "audio": audio_hex},
            "extra_info": {"music_duration": 30},
        })

    # MiniMax builds its own httpx client; swap its transport.
    p._client = httpx.AsyncClient(
        base_url="https://api.minimax.io", transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer k", "Content-Type": "application/json"},
    )

    task = asyncio.run(p.create_music_task(_req(model="music-3.0")))
    assert task.status == "pending"
    # The blocking call runs on the first poll.
    result = asyncio.run(p.get_music_task(task.task_id))
    assert result.status == "succeeded"
    assert captured["path"] == "/v1/music_generation"
    b = captured["body"]
    assert b["model"] == "music-3.0"
    # A bare prompt with no separate lyrics and not instrumental -> treated as lyrics.
    assert b.get("lyrics") == "a happy song"
    # Default output_format is 'url' (avoids a large hex round-trip).
    assert b["output_format"] == "url"
    # The hex audio was decoded to bytes then re-base64 for the unified task.
    assert result.audio_b64 == base64.b64encode(bytes.fromhex(audio_hex)).decode()
    assert result.usage and result.usage.duration == 30


def test_minimax_in_progress_status_keeps_task_running() -> None:
    from mm_gateway.providers.minimax import _MUSIC_TASKS, MiniMaxProvider
    _MUSIC_TASKS.clear()
    p = MiniMaxProvider(_backend("minimax"))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"base_resp": {"status_code": 0},
                                         "data": {"status": 1}})  # still in progress

    p._client = httpx.AsyncClient(
        base_url="https://api.minimax.io", transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer k", "Content-Type": "application/json"},
    )
    task = asyncio.run(p.create_music_task(_req()))
    result = asyncio.run(p.get_music_task(task.task_id))
    assert result.status == "running"


def test_minimax_translates_quality_inline_reference_and_lyrics_control() -> None:
    from mm_gateway.providers.minimax import MiniMaxProvider

    p = MiniMaxProvider(_backend("minimax"))
    request = UnifiedMusicRequest(
        model="music-3.0",
        content=[text_part("a happy song"), audio_part("data:audio/wav;base64,AAAA")],
        sample_rate_hz=44100,
        bitrate_kbps=192,
        enhance_lyrics=True,
    )
    body = p._build_body(request, "music-3.0")
    assert body["audio_setting"] == {"sample_rate": 44100, "bitrate": 192000}
    assert body["audio_base64"] == "AAAA"
    assert body["lyrics_optimizer"] is True


def test_minimax_base_resp_error_fails_task() -> None:
    from mm_gateway.core.exceptions import TaskFailedError
    from mm_gateway.providers.minimax import _MUSIC_TASKS, MiniMaxProvider
    _MUSIC_TASKS.clear()
    p = MiniMaxProvider(_backend("minimax"))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"base_resp": {"status_code": 1004, "status_msg": "bad"}})

    p._client = httpx.AsyncClient(
        base_url="https://api.minimax.io", transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer k", "Content-Type": "application/json"},
    )
    task = asyncio.run(p.create_music_task(_req()))
    with pytest.raises(TaskFailedError):
        asyncio.run(p.get_music_task(task.task_id))


# --------------------------------------------------------------------------- #
# udioapi — two-phase, feed status -> unified lifecycle
# --------------------------------------------------------------------------- #


def test_udioapi_requires_api_key() -> None:
    from mm_gateway.providers.udioapi import UdioApiProvider
    with pytest.raises(ProviderNotConfiguredError):
        UdioApiProvider(BackendConfig(name="udioapi", type="udioapi"))


def test_udioapi_create_returns_workid_and_poll_completes() -> None:
    from mm_gateway.providers.udioapi import UdioApiProvider
    p = UdioApiProvider(_backend("udioapi"))

    create_path = ["/api/v2/generate"]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v2/generate":
            create_path.append(_body(request))
            return httpx.Response(200, json={"workId": "w-1"})
        # feed poll
        return httpx.Response(200, json={"data": {"response_data": [
            {"status": "complete", "audio_url": "https://x.test/song.mp3", "duration": 45},
        ]}})

    _mount(p, handler, base_url="https://udioapi.pro")

    task = asyncio.run(p.create_music_task(_req(model="chirp-v5", style="edm")))
    assert task.task_id == "w-1" and task.status == "pending"
    # Custom mode is selected because style is present.
    assert create_path[1]["prompt"] == "a happy song"
    assert create_path[1]["style"] == "edm"

    result = asyncio.run(p.get_music_task("w-1"))
    assert result.status == "succeeded"
    assert result.audio_urls == ["https://x.test/song.mp3"]
    assert result.usage and result.usage.duration == 45


def test_udioapi_fail_message_fails_task() -> None:
    from mm_gateway.providers.udioapi import UdioApiProvider
    p = UdioApiProvider(_backend("udioapi"))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v2/generate":
            return httpx.Response(200, json={"workId": "w-2"})
        return httpx.Response(200, json={"data": {"response_data": [
            {"status": "text", "fail_message": "moderation blocked"},
        ]}})

    _mount(p, handler, base_url="https://udioapi.pro")
    asyncio.run(p.create_music_task(_req()))
    result = asyncio.run(p.get_music_task("w-2"))
    assert result.status == "failed"
    assert "moderation" in (result.error or "")


def test_udioapi_translates_neutral_influence_controls() -> None:
    from mm_gateway.providers.udioapi import UdioApiProvider

    p = UdioApiProvider(_backend("udioapi"))
    body = p._build_body(_req(
        vocal_gender="female",
        style_strength=0.8,
        novelty=0.25,
        reference_audio_strength=0.6,
    ))
    assert body["gender"] == "female"
    assert body["style_weight"] == 0.8
    assert body["weirdness_constraint"] == 0.25
    assert body["audio_weight"] == 0.6


# --------------------------------------------------------------------------- #
# Mureka — two-phase, candidate field names + status map
# --------------------------------------------------------------------------- #


def test_mureka_requires_api_key() -> None:
    from mm_gateway.providers.mureka import MurekaProvider
    with pytest.raises(ProviderNotConfiguredError):
        MurekaProvider(BackendConfig(name="mureka", type="mureka"))


def test_mureka_create_and_poll_succeeded() -> None:
    from mm_gateway.providers.mureka import MurekaProvider
    p = MurekaProvider(_backend("mureka"))

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/song/generate":
            captured["body"] = _body(request)
            return httpx.Response(200, json={"task_id": "mk-1"})
        return httpx.Response(200, json={
            "status": "succeeded", "audio_url": "https://x.test/mk.mp3", "duration": 20,
        })

    _mount(p, handler, base_url="https://platform.mureka.ai")

    task = asyncio.run(p.create_music_task(_req(model="mureka-song-1", style="pop")))
    assert task.task_id == "mk-1"
    # A single-line prompt with no lyrics -> "prompt" field, style -> tags.
    assert captured["body"]["prompt"] == "a happy song"
    assert captured["body"]["tags"] == "pop"

    result = asyncio.run(p.get_music_task("mk-1"))
    assert result.status == "succeeded"
    assert result.audio_urls == ["https://x.test/mk.mp3"]
    assert result.usage and result.usage.duration == 20


def test_mureka_lyrics_input_routes_to_lyrics_field() -> None:
    from mm_gateway.providers.mureka import MurekaProvider
    p = MurekaProvider(_backend("mureka"))

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/song/generate":
            captured["body"] = _body(request)
            return httpx.Response(200, json={"task_id": "mk-2"})
        return httpx.Response(200, json={"status": "running"})

    _mount(p, handler, base_url="https://platform.mureka.ai")
    # Multi-line text looks like lyrics -> sent as `lyrics`.
    asyncio.run(p.create_music_task(_req(prompt="verse one\nchorus here")))
    assert "lyrics" in captured["body"]
    assert captured["body"]["lyrics"] == "verse one\nchorus here"


def test_mureka_failed_status_surfaces_error() -> None:
    from mm_gateway.providers.mureka import MurekaProvider
    p = MurekaProvider(_backend("mureka"))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/song/generate":
            return httpx.Response(200, json={"task_id": "mk-3"})
        return httpx.Response(200, json={"status": "failed", "fail_message": "nope"})

    _mount(p, handler, base_url="https://platform.mureka.ai")
    asyncio.run(p.create_music_task(_req()))
    result = asyncio.run(p.get_music_task("mk-3"))
    assert result.status == "failed"
    assert result.error == "nope"


def test_mureka_translates_voice_and_audio_quality() -> None:
    from mm_gateway.providers.mureka import MurekaProvider

    p = MurekaProvider(_backend("mureka"))
    body = p._build_body(_req(
        voice="warm-alto",
        audio_format="wav",
        sample_rate_hz=44100,
        bitrate_kbps=192,
        seed=17,
    ))
    assert body["voice_id"] == "warm-alto"
    assert body["audio_config"] == {
        "format": "wav",
        "sample_rate": 44100,
        "bitrate": 192000,
    }
    assert body["seed"] == 17


# --------------------------------------------------------------------------- #
# ACE-Step — two-phase, result JSON string + binary audio fetch
# --------------------------------------------------------------------------- #


def test_acestep_requires_base_url() -> None:
    from mm_gateway.providers.acestep import AceStepProvider
    with pytest.raises(ProviderNotConfiguredError):
        AceStepProvider(BackendConfig(name="acestep", type="acestep", api_key="k"))


def test_acestep_create_retries_transient_504_then_succeeds() -> None:
    from mm_gateway.providers.acestep import AceStepProvider
    p = AceStepProvider(_backend("acestep", base_url="https://ace.test"))
    p._create_backoff_base = 0.0  # no real sleeping in the test

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/release_task":
            calls["n"] += 1
            if calls["n"] == 1:
                # Cloudflare-style transient gateway timeout (observed in CI).
                return httpx.Response(504, text="<html>504 Gateway Timeout</html>")
            return httpx.Response(200, json={"data": {"task_id": "ac-r"}})
        return httpx.Response(404)

    _mount(p, handler, base_url="https://ace.test")
    task = asyncio.run(p.create_music_task(_req(model="acestep-v15-xl-turbo")))
    assert task.task_id == "ac-r"
    assert calls["n"] == 2  # one transient 504, then success


def test_acestep_create_does_not_retry_4xx() -> None:
    from mm_gateway.core.exceptions import ProviderRequestError
    from mm_gateway.providers.acestep import AceStepProvider
    p = AceStepProvider(_backend("acestep", base_url="https://ace.test"))
    p._create_backoff_base = 0.0

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/release_task":
            calls["n"] += 1
            return httpx.Response(400, json={"error": "bad prompt"})
        return httpx.Response(404)

    _mount(p, handler, base_url="https://ace.test")
    with pytest.raises(ProviderRequestError):
        asyncio.run(p.create_music_task(_req()))
    assert calls["n"] == 1  # 4xx is not retried


def test_acestep_create_surfaces_error_after_exhausting_retries() -> None:
    from mm_gateway.core.exceptions import ProviderRequestError
    from mm_gateway.providers.acestep import AceStepProvider
    p = AceStepProvider(_backend("acestep", base_url="https://ace.test"))
    p._create_backoff_base = 0.0

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/release_task":
            calls["n"] += 1
            return httpx.Response(504, text="<html>504</html>")
        return httpx.Response(404)

    _mount(p, handler, base_url="https://ace.test")
    with pytest.raises(ProviderRequestError):
        asyncio.run(p.create_music_task(_req()))
    assert calls["n"] == p._create_max_attempts


def test_acestep_create_and_poll_fetches_audio() -> None:
    from mm_gateway.providers.acestep import AceStepProvider
    p = AceStepProvider(_backend("acestep", base_url="https://ace.test"))

    captured: dict[str, Any] = {}
    audio_bytes = b"FAKEAUDIO"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/release_task":
            captured["body"] = _body(request)
            return httpx.Response(200, json={"data": {"task_id": "ac-1", "status": "queued"}})
        if request.url.path == "/query_result":
            return httpx.Response(200, json={"data": [{
                "task_id": "ac-1", "status": 1,
                "result": json.dumps([{"file": "/v1/audio?path=xyz", "metas": {"duration": 12}}]),
            }]})
        if request.url.path == "/v1/audio":
            captured["audio_path"] = request.url.path + "?" + request.url.query.decode()
            return httpx.Response(200, content=audio_bytes)
        return httpx.Response(404)

    _mount(p, handler, base_url="https://ace.test")

    task = asyncio.run(p.create_music_task(_req(model="acestep-v15-turbo")))
    assert task.task_id == "ac-1"
    assert captured["body"]["prompt"] == "a happy song"

    result = asyncio.run(p.get_music_task("ac-1"))
    assert result.status == "succeeded"
    # The audio file was fetched and inlined as base64.
    assert result.audio_b64 == base64.b64encode(audio_bytes).decode()
    assert result.usage and result.usage.duration == 12


def test_acestep_failed_status_surfaces_error() -> None:
    from mm_gateway.providers.acestep import AceStepProvider
    p = AceStepProvider(_backend("acestep", base_url="https://ace.test"))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/release_task":
            return httpx.Response(200, json={"data": {"task_id": "ac-2"}})
        return httpx.Response(200, json={"data": [{"task_id": "ac-2", "status": 2}]})

    _mount(p, handler, base_url="https://ace.test")
    asyncio.run(p.create_music_task(_req()))
    result = asyncio.run(p.get_music_task("ac-2"))
    assert result.status == "failed"


def test_acestep_translates_key_batch_steps_and_continuation_audio() -> None:
    from mm_gateway.providers.acestep import AceStepProvider

    p = AceStepProvider(_backend("acestep", base_url="https://ace.test"))
    request = UnifiedMusicRequest(
        model="ace-step-1.5",
        content=[
            text_part("a happy song"),
            audio_part("https://assets.test/continue.wav", "continuation_audio"),
        ],
        key="A",
        scale="minor",
        inference_steps=32,
        n=3,
    )
    body = p._build_body(request)
    assert body["key_scale"] == "A minor"
    assert body["inference_steps"] == 32
    assert body["batch_size"] == 3
    assert body["src_audio_path"] == "https://assets.test/continue.wav"
    assert body["task_type"] == "cover"
    # Official acestep.sh native defaults.
    assert body["thinking"] is True
    assert body["use_format"] is True
    assert body["use_cot_caption"] is True
    assert body["use_cot_language"] is True
    assert body["use_random_seed"] is True


# --------------------------------------------------------------------------- #
# ACE-Step — completion mode (POST /v1/chat/completions, synthetic task)
# --------------------------------------------------------------------------- #


def test_acestep_auto_mode_picks_completion_for_acemusic_host() -> None:
    from mm_gateway.providers.acestep import AceStepProvider
    assert AceStepProvider._resolve_mode(_backend("acestep", base_url="https://api.acemusic.ai")) == "completion"
    assert AceStepProvider._resolve_mode(_backend("acestep", base_url="http://127.0.0.1:8001")) == "native"


def test_acestep_explicit_mode_override() -> None:
    from mm_gateway.config import BackendConfig
    from mm_gateway.providers.acestep import AceStepProvider
    b = BackendConfig(name="acestep", type="acestep", base_url="https://api.acemusic.ai",
                      extra={"acestep_api_mode": "native"})
    assert AceStepProvider._resolve_mode(b) == "native"


def test_acestep_completion_create_mints_synthetic_task() -> None:
    from mm_gateway.providers.acestep import _COMPLETION_TASKS, AceStepProvider
    _COMPLETION_TASKS.clear()
    p = AceStepProvider(_backend("acestep", base_url="https://api.acemusic.ai"))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)  # create must NOT hit the network

    _mount(p, handler, base_url="https://api.acemusic.ai")
    task = asyncio.run(p.create_music_task(_req(model="acestep-v15-turbo")))
    assert task.status == "pending"
    assert task.task_id.startswith("acestep-")
    assert task.task_id in _COMPLETION_TASKS


def test_acestep_completion_body_matches_official_spec() -> None:
    from mm_gateway.providers.acestep import AceStepProvider
    p = AceStepProvider(_backend("acestep", base_url="https://api.acemusic.ai"))
    request = UnifiedMusicRequest(
        model="acestep-v15-turbo",
        content=[text_part("soft rock")],
        lyrics="[Verse]\nhello",
        duration=180, bpm=120, key="A", scale="minor",
        vocal_language="en", audio_format="mp3",
    )
    body = p._build_completion_body(request, request.model)
    assert body["model"] == "acemusic/acestep-v15-turbo"  # prefix added
    assert body["stream"] is False
    assert body["thinking"] is True
    assert body["use_format"] is True
    assert body["use_cot_caption"] is True
    assert body["use_cot_language"] is True
    msg = body["messages"][0]
    assert msg["role"] == "user"
    assert "<prompt>soft rock</prompt>" in msg["content"]
    assert "<lyrics>[Verse]\nhello</lyrics>" in msg["content"]
    assert body["audio_config"] == {
        "format": "mp3", "vocal_language": "en", "duration": 180.0,
        "bpm": 120, "key_scale": "A minor",
    }


def test_acestep_completion_poll_inlines_data_url_audio() -> None:
    from mm_gateway.providers.acestep import _COMPLETION_TASKS, AceStepProvider
    _COMPLETION_TASKS.clear()
    p = AceStepProvider(_backend("acestep", base_url="https://api.acemusic.ai"))
    audio_b64 = base64.b64encode(b"FAKEAUDIO").decode()
    data_url = f"data:audio/mpeg;base64,{audio_b64}"
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            captured["body"] = _body(request)
            return httpx.Response(200, json={
                "id": "chatcmpl-1",
                "choices": [{"finish_reason": "stop", "message": {
                    "content": "la la la",
                    "audio": [{"type": "audio_url", "audio_url": {"url": data_url}}],
                }}],
            })
        return httpx.Response(404)

    _mount(p, handler, base_url="https://api.acemusic.ai")
    task = asyncio.run(p.create_music_task(_req(model="acestep-v15-turbo")))
    result = asyncio.run(p.get_music_task(task.task_id))
    assert result.status == "succeeded"
    assert result.audio_b64 == audio_b64
    assert result.audio_media_type == "audio/mpeg"
    assert result.lyrics == "la la la"
    assert captured["body"]["model"] == "acemusic/acestep-v15-turbo"


def test_acestep_completion_finish_reason_error_fails() -> None:
    from mm_gateway.providers.acestep import _COMPLETION_TASKS, AceStepProvider
    _COMPLETION_TASKS.clear()
    p = AceStepProvider(_backend("acestep", base_url="https://api.acemusic.ai"))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json={
                "id": "chatcmpl-err",
                "choices": [{"finish_reason": "error", "message": {"content": ""}}],
                "detail": "invalid lyrics",
            })
        return httpx.Response(404)

    _mount(p, handler, base_url="https://api.acemusic.ai")
    task = asyncio.run(p.create_music_task(_req()))
    result = asyncio.run(p.get_music_task(task.task_id))
    assert result.status == "failed"
    assert "invalid lyrics" in (result.error or "")


def test_acestep_completion_no_audio_fails() -> None:
    from mm_gateway.providers.acestep import _COMPLETION_TASKS, AceStepProvider
    _COMPLETION_TASKS.clear()
    p = AceStepProvider(_backend("acestep", base_url="https://api.acemusic.ai"))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json={"id": "x", "choices": [{"finish_reason": "stop", "message": {}}]})
        return httpx.Response(404)

    _mount(p, handler, base_url="https://api.acemusic.ai")
    task = asyncio.run(p.create_music_task(_req()))
    result = asyncio.run(p.get_music_task(task.task_id))
    assert result.status == "failed"


def test_acestep_native_use_random_seed_false_when_seed_set() -> None:
    from mm_gateway.providers.acestep import AceStepProvider
    p = AceStepProvider(_backend("acestep", base_url="http://ace.local:8001"))
    body = p._build_body(_req(seed=42))
    assert body["seed"] == 42
    assert body["use_random_seed"] is False


def test_acestep_native_cover_and_repaint_fields_from_extra() -> None:
    from mm_gateway.providers.acestep import AceStepProvider
    p = AceStepProvider(_backend("acestep", base_url="http://ace.local:8001"))
    request = UnifiedMusicRequest(
        model="ace-step-1.5",
        content=[text_part("cover"), audio_part("https://assets.test/src.wav", "continuation_audio")],
        extra={"audio_cover_strength": 0.7, "repainting_start": 10, "repainting_end": 50},
    )
    body = p._build_body(request)
    assert body["src_audio_path"] == "https://assets.test/src.wav"
    assert body["task_type"] == "cover"
    assert body["audio_cover_strength"] == 0.7
    assert body["repainting_start"] == 10
    assert body["repainting_end"] == 50


# --------------------------------------------------------------------------- #
# Google Lyria — output extraction (steps[].content[] blocks)
# --------------------------------------------------------------------------- #


def test_lyria_extract_audio_and_lyrics_from_steps() -> None:
    from mm_gateway.providers.google import _extract_lyria_output
    data = {"steps": [{"type": "model_output", "content": [
        {"type": "audio", "data": "UklGRiQAAABXQVZFZmV"},
        {"type": "text", "text": "la la la"},
    ]}]}
    audio, lyrics = _extract_lyria_output(data)
    assert audio == "UklGRiQAAABXQVZFZmV"
    assert lyrics == "la la la"


def test_lyria_extract_falls_back_to_top_level_fields() -> None:
    from mm_gateway.providers.google import _extract_lyria_output
    audio, lyrics = _extract_lyria_output({"output_audio": "AAA", "output_text": "do re mi"})
    assert audio == "AAA"
    assert lyrics == "do re mi"


def test_lyria_extract_returns_none_when_empty() -> None:
    from mm_gateway.providers.google import _extract_lyria_output
    audio, lyrics = _extract_lyria_output({})
    assert audio is None and lyrics is None


def test_lyria_translates_inline_reference_media() -> None:
    from mm_gateway.providers.google import GoogleProvider
    from mm_gateway.schemas.music import image_part

    p = GoogleProvider(_backend("google"))
    request = UnifiedMusicRequest(
        model="lyria-3",
        content=[
            text_part("a happy song"),
            image_part("data:image/png;base64,BBBB"),
            audio_part("data:audio/wav;base64,AAAA"),
        ],
    )
    parts = p._lyria_body(request)["input"]
    assert {"type": "image", "mime_type": "image/png", "data": "BBBB"} in parts
    assert {
        "type": "audio",
        "mime_type": "audio/wav",
        "data": "AAAA",
        "role": "reference_audio",
    } in parts


def test_lyria_body_uses_interactions_shape_not_predict_path() -> None:
    """The Interactions surface takes top-level response_format /
    generation_config — no `config` wrapper, no response_modalities."""
    from mm_gateway.providers.google import GoogleProvider

    p = GoogleProvider(_backend("google"))
    request = UnifiedMusicRequest(
        model="lyria-3-pro-preview",
        content=[text_part("a happy song")],
        seed=7, negative_prompt="vocals", guidance_scale=2.0, n=3,
        audio_format="mp3",
    )
    body = p._lyria_body(request)
    # model and input are the only required top-level fields; the path the
    # adapter posts to is /v1beta/interactions, so model is NOT in the URL.
    assert body["model"] == "lyria-3-pro-preview"
    assert body["input"] == [{"type": "text", "text": "a happy song"}]
    assert "config" not in body
    assert "response_modalities" not in body
    # response_format.mime_type uses the SDK output enum: audio/mp3 (NOT
    # audio/mpeg, which only the input-parts enum accepts).
    assert body["response_format"] == {"type": "audio", "mime_type": "audio/mp3"}
    gc = body["generation_config"]
    assert gc["seed"] == 7
    assert gc["negative_prompt"] == "vocals"
    assert gc["guidance_scale"] == 2.0
    assert gc["number_of_outputs"] == 3


def test_lyria_request_mime_uses_sdk_output_enum_and_omits_default() -> None:
    """response_format.mime_type is the SDK output enum (audio/mp3 not
    audio/mpeg); no audio_format => omit response_format (Lyria default MP3)."""
    from mm_gateway.providers.google import _lyria_request_mime

    assert _lyria_request_mime(None) is None
    assert _lyria_request_mime("") is None
    assert _lyria_request_mime("mp3") == "audio/mp3"
    assert _lyria_request_mime("wav") == "audio/wav"
    assert _lyria_request_mime("ogg_opus") == "audio/ogg_opus"


def test_lyria_body_omits_response_format_when_no_audio_format() -> None:
    from mm_gateway.providers.google import GoogleProvider

    p = GoogleProvider(_backend("google"))
    body = p._lyria_body(UnifiedMusicRequest(
        model="lyria-3-pro-preview", content=[text_part("a happy song")]))
    # No audio_format => no response_format sent; Lyria's default MP3 applies.
    assert "response_format" not in body
    assert "generation_config" not in body  # no knobs set either


def test_lyria_media_type_defaults_to_mp3_and_maps_formats() -> None:
    from mm_gateway.providers.google import _lyria_media_type

    # Default + mp3 -> audio/mpeg (the IANA/gateway-standard MIME for MP3),
    # NOT audio/wav (Lyria's default output is MP3, not WAV).
    assert _lyria_media_type(None) == "audio/mpeg"
    assert _lyria_media_type("") == "audio/mpeg"
    assert _lyria_media_type("mp3") == "audio/mpeg"
    assert _lyria_media_type("wav") == "audio/wav"
    assert _lyria_media_type("ogg_opus") == "audio/ogg"


def test_lyria_poll_hits_interactions_endpoint_with_api_key_header() -> None:
    """End-to-end of the Lyria synthetic-task poll: the first poll performs the
    blocking upstream POST. Assert the URL path and auth header the adapter now
    uses (no :predictInteractions, no ?key=)."""
    from mm_gateway.providers.google import GoogleProvider

    p = GoogleProvider(_backend("google", api_key="SECRET"))

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={
            "steps": [{"type": "model_output", "content": [
                {"type": "audio", "data": "UklGRiQAAABXQVZFZmV"},
            ]}],
        })

    # Swap the per-poll httpx.AsyncClient for a MockTransport one so the
    # event-hook-wrapped client in get_music_task is exercised in-process.
    import mm_gateway.providers.google as gmod
    real_async_client = httpx.AsyncClient

    class _MockClient(httpx.AsyncClient):
        def __init__(self, *a, **kw):
            kw.pop("timeout", None)
            kw["transport"] = httpx.MockTransport(handler)
            kw["base_url"] = "https://generativelanguage.googleapis.com"
            super().__init__(*a, **kw)

    gmod.httpx.AsyncClient = _MockClient  # type: ignore[attr-defined]
    try:
        create = asyncio.run(p.create_music_task(
            UnifiedMusicRequest(model="lyria-3-pro-preview", content=[text_part("x")])
        ))
        task = asyncio.run(p.get_music_task(create.task_id))
    finally:
        gmod.httpx.AsyncClient = real_async_client  # type: ignore[attr-defined]

    assert len(seen) == 1
    req = seen[0]
    # /v1beta/interactions — NOT /v1beta/models/...:predictInteractions?key=
    assert req.url.path == "/v1beta/interactions"
    assert "key=" not in str(req.url)
    assert req.headers["x-goog-api-key"] == "SECRET"
    assert task.status == "succeeded"
    assert task.audio_b64 == "UklGRiQAAABXQVZFZmV"
    # No audio_format => Lyria default MP3 => reported as audio/mpeg.
    assert task.audio_media_type == "audio/mpeg"
