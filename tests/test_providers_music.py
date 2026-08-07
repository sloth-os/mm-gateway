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
from mm_gateway.schemas.music import UnifiedMusicRequest, text_part

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

    task = asyncio.run(p.create_music_task(_req(model="chirp-v5", extra={"style": "edm"})))
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

    task = asyncio.run(p.create_music_task(_req(model="mureka-song-1", extra={"style": "pop"})))
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


# --------------------------------------------------------------------------- #
# ACE-Step — two-phase, result JSON string + binary audio fetch
# --------------------------------------------------------------------------- #


def test_acestep_requires_base_url() -> None:
    from mm_gateway.providers.acestep import AceStepProvider
    with pytest.raises(ProviderNotConfiguredError):
        AceStepProvider(BackendConfig(name="acestep", type="acestep", api_key="k"))


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
