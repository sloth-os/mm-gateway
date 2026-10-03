"""Speech/clone integration across routing, ownership, billing, REST and MCP."""

from __future__ import annotations

import asyncio
import json
import sys
import types

import httpx
import pytest

from mm_gateway.config import BackendConfig, KeyBudget, KeyConfig, Settings
from mm_gateway.providers._speech import SpeechTaskMixin, inline_audio
from mm_gateway.schemas.audio import AudioUsage
from mm_gateway.server.app import create_app
from tests.test_mcp_server import _client


class FakeSpeech(SpeechTaskMixin):
    name = "fake_speech"
    audio_models = ["fake-audio"]
    default_voice = "private-default"

    def __init__(self, backend):
        super().__init__(backend)
        self.audio_calls = []
        self.clone_calls = []
        self.audio_started = asyncio.Event()
        self.audio_release = asyncio.Event()
        self.clone_release = asyncio.Event()

    async def _synthesize(self, request):
        self.audio_calls.append(request)
        self.audio_started.set()
        if self.backend.extra.get("delay_audio"):
            await self.audio_release.wait()
        if self.backend.extra.get("fail_audio"):
            raise RuntimeError("secret upstream credential should never appear")
        return [inline_audio(b"audio", "mp3")], AudioUsage(input_characters=len(request.text))

    async def _clone(self, request):
        self.clone_calls.append(request)
        if self.backend.extra.get("delay_clone"):
            await self.clone_release.wait()
        return f"private-clone-{self.backend.extra.get('__account_id', 'default')}", self.backend.extra.get("verify", False)

    async def _verification_required(self, native_voice_id):
        return self.backend.extra.get("verify", False)


@pytest.fixture
def speech_app(monkeypatch):
    from mm_gateway import registry
    monkeypatch.setitem(registry._PROVIDER_CLASSES, "fake_speech", "FakeSpeech")
    monkeypatch.setitem(sys.modules, "mm_gateway.providers.fake_speech", types.SimpleNamespace(FakeSpeech=FakeSpeech))

    def make(*, extra=None, pooled=False, budget=None):
        backend = BackendConfig(name="speech", type="fake_speech", api_key="test", extra=extra or {},
                                credentials=[{"id": "first", "api_key": "a"}, {"id": "second", "api_key": "b"}]
                                if pooled else [])
        return create_app(Settings(
            backends=[backend], keys=[KeyConfig(id="alice", key="alice", budget=budget), KeyConfig(id="bob", key="bob")],
            poll_interval=0.001, mcp_enabled=True,
            catalog_models={"fake-audio": {"modality": "audio", "supports_voice_cloning": True,
                                           "price": {"per_character": 0.001, "per_clone": 0.2}}},
        ))
    return make


def speech(voice="default", text="hello", **parameters):
    return {"input": [{"type": "text", "text": text}], "parameters": {"voice": voice, **parameters}}


def clone():
    return {"input": [{"type": "audio", "uri": "data:audio/wav;base64,AAAA"}],
            "parameters": {"name": "My voice"}, "consent": {"granted": True}, "metadata": {"job": 1}}


async def terminal(client, location):
    for _ in range(100):
        response = await client.get(location)
        response.raise_for_status()
        if response.json()["status"] in {"succeeded", "failed"}:
            return response
        await asyncio.sleep(0.002)
    raise AssertionError("task did not finish")


async def test_clone_to_speech_and_owned_catalogue(speech_app):
    app = speech_app()
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway", headers={"authorization": "Bearer alice"},
    ) as client:
        created = await client.post("/v1/voices", json=clone(), headers={"idempotency-key": "clone-1"})
        assert created.status_code == 202
        id = created.json()["id"]
        assert id.startswith("voice_")
        completed = await terminal(client, created.headers["location"])
        voice = completed.json()
        assert voice["status"] == "succeeded" and voice["metadata"] == {"job": 1}
        assert "private-clone" not in completed.text and "account_id" not in completed.text
        assert voice["usage"]["cost"] == 0.2
        replay = await client.post("/v1/voices", json=clone(), headers={"idempotency-key": "clone-1"})
        assert replay.json() == created.json()
        assert replay.headers["idempotency-replayed"] == "true"
        conflict_body = clone(); conflict_body["parameters"]["name"] = "Different"
        conflict = await client.post("/v1/voices", json=conflict_body, headers={"idempotency-key": "clone-1"})
        assert conflict.status_code == 409
        made_audio = await client.post("/v1/audio", json=speech(id), headers={"idempotency-key": "speech-1"})
        assert made_audio.status_code == 202
        audio = (await terminal(client, made_audio.headers["location"])).json()
        assert audio["object"] == "audio" and audio["status"] == "succeeded"
        assert audio["usage"]["input_characters"] == 5
        assert audio["usage"]["cost"] == 0.005
        assert audio["outputs"][0]["uri"] == "data:audio/mpeg;base64,YXVkaW8="
        replay_audio = await client.post("/v1/audio", json=speech(id), headers={"idempotency-key": "speech-1"})
        assert replay_audio.json() == made_audio.json()
        listed = await client.get("/v1/voices")
        assert {v["id"] for v in listed.json()["data"]} == {"default", id}
        assert (await client.get("/v1/voices", headers={"if-none-match": listed.headers["etag"]})).status_code == 304
        assert (await client.get(created.headers["location"], headers={"if-none-match": completed.headers["etag"]})).status_code == 304
        bob = {"authorization": "Bearer bob"}
        for path, body in [("/v1/audio", speech(id)), ("/v1/audio/estimate", speech(id))]:
            assert (await client.post(path, json=body, headers=bob)).status_code == 403
        assert (await client.get(created.headers["location"], headers=bob)).status_code == 403
        assert (await client.get(made_audio.headers["location"], headers=bob)).status_code == 403
        assert {v["id"] for v in (await client.get("/v1/voices", headers=bob)).json()["data"]} == {"default"}
        usage = (await client.get("/v1/usage")).json()
        assert usage["key"]["spent_usd"] == pytest.approx(0.205)
        provider = app.state.registry.audio_provider("speech")
        assert len(provider.clone_calls) == len(provider.audio_calls) == 1
        assert provider.audio_calls[0].native_voice_id == "private-clone-default"


async def test_cloned_voice_stays_on_second_account_and_fallback_never_changes_it(speech_app):
    app = speech_app(pooled=True)
    from mm_gateway.observability.selection import STORE
    STORE.clear()
    STORE.observe(backend="speech", account="first", model="fake-audio", modality="audio",
                  outcome="failure", latency_s=1, rate_limited=True)
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway", headers={"authorization": "Bearer alice"},
    ) as client:
        created = await client.post("/v1/voices", json=clone())
        id = (await terminal(client, created.headers["location"])).json()["id"]
        STORE.clear()  # The default account is preferred again for normal speech.
        req = speech(id); req["model"] = "fake-audio"; req["routing"] = {"fallback": "any"}
        est = await client.post("/v1/audio/estimate", json=req)
        assert est.status_code == 200 and est.json()["model"] == "fake-audio"
        audio = await client.post("/v1/audio", json=req)
        assert audio.status_code == 202
        assert (await terminal(client, audio.headers["location"])).json()["status"] == "succeeded"
        accounts = dict(app.state.registry.accounts_of("speech"))
        assert accounts["first"].audio_calls == []
        assert len(accounts["second"].audio_calls) == 1
        assert accounts["second"].audio_calls[0].native_voice_id == "private-clone-second"
    STORE.clear()


async def test_create_and_get_remain_prompt_while_speech_is_running(speech_app):
    app = speech_app(extra={"delay_audio": True})
    provider = app.state.registry.audio_provider("speech")
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway", headers={"authorization": "Bearer alice"},
    ) as client:
        created = await asyncio.wait_for(client.post("/v1/audio", json=speech()), 0.5)
        assert created.status_code == 202
        await asyncio.wait_for(provider.audio_started.wait(), 0.5)
        current = await asyncio.wait_for(client.get(created.headers["location"]), 0.5)
        assert current.json()["status"] in {"pending", "running"}
        provider.audio_release.set()
        assert (await terminal(client, created.headers["location"])).json()["status"] == "succeeded"


@pytest.mark.parametrize("extra", [{"delay_clone": True}, {"verify": True}])
async def test_unready_or_unverified_voice_cannot_generate_speech(speech_app, extra):
    app = speech_app(extra=extra)
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway", headers={"authorization": "Bearer alice"},
    ) as client:
        created = await client.post("/v1/voices", json=clone())
        if extra.get("verify"):
            for _ in range(100):
                voice = (await client.get(created.headers["location"])).json()
                if voice["verification_required"]:
                    break
                await asyncio.sleep(0.002)
            assert voice["verification_required"] is True
        rejected = await client.post("/v1/audio", json=speech(created.json()["id"]))
        assert rejected.status_code == 409 and rejected.json()["code"] == "voice_not_ready"
        provider = app.state.registry.audio_provider("speech")
        assert provider.audio_calls == []
        if extra.get("verify"):
            provider.backend.extra["verify"] = False
            ready = (await terminal(client, created.headers["location"])).json()
            assert ready["status"] == "succeeded" and ready["verification_required"] is False
            audio = await client.post("/v1/audio", json=speech(created.json()["id"]))
            assert (await terminal(client, audio.headers["location"])).json()["status"] == "succeeded"
            assert len(provider.clone_calls) == 1


async def test_failed_speech_is_cached_and_releases_budget(speech_app):
    app = speech_app(extra={"fail_audio": True}, budget=KeyBudget(limit_usd=1))
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway", headers={"authorization": "Bearer alice"},
    ) as client:
        created = await client.post("/v1/audio", json=speech())
        failure = await terminal(client, created.headers["location"])
        assert failure.json()["status"] == "failed" and "secret" not in failure.text
        for _ in range(3):
            assert (await client.get(created.headers["location"])).json()["status"] == "failed"
        assert len(app.state.registry.audio_provider("speech").audio_calls) == 1
        usage = (await client.get("/v1/usage")).json()["key"]
        assert usage["spent_usd"] == usage["reserved_usd"] == 0


async def test_estimates_respect_clone_price_and_budgets_without_provider_calls(speech_app):
    app = speech_app(budget=KeyBudget(limit_usd=0.1))
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway", headers={"authorization": "Bearer alice"},
    ) as client:
        estimate = await client.post("/v1/audio/estimate", json=speech(text="abc"))
        assert estimate.json()["estimated_cost"] == 0.003
        clone_est = await client.post("/v1/voices/estimate", json=clone())
        assert clone_est.json()["candidates"][0]["reason"] == "budget"
        assert (await client.post("/v1/voices", json=clone())).status_code == 402
        assert (await client.get("/v1/usage")).json()["key"]["reserved_usd"] == 0
        assert app.state.registry.audio_provider("speech").clone_calls == []


@pytest.mark.parametrize("path,body", [
    ("/v1/audio", {"input": "text"}),
    ("/v1/audio", {"input": []}),
    ("/v1/audio", speech(text="   ")),
    ("/v1/audio", speech(response_format="wav")),
    ("/v1/voices", {**clone(), "consent": {"granted": False}}),
    ("/v1/voices", {**clone(), "consent": {"granted": 1}}),
    ("/v1/voices", {key: value for key, value in clone().items() if key != "consent"}),
    ("/v1/voices", {**clone(), "input": [{"type": "audio", "uri": "data:audio/wav;base64,!invalid"}]}),
    ("/v1/voices", {**clone(), "input": [{"type": "audio", "uri": "file:///etc/passwd"}]}),
])
async def test_strict_contract_rejects_invalid_inputs(speech_app, path, body):
    app = speech_app()
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway", headers={"authorization": "Bearer alice"},
    ) as client:
        response = await client.post(path, json=body)
        assert response.status_code == 422 and response.headers["content-type"] == "application/problem+json"


async def test_mcp_clone_then_speech_has_same_resources_and_idempotency(speech_app):
    app = speech_app()
    async with _client(app, "alice") as session:
        args = {**clone(), "idempotency_key": "mcp-clone"}
        result = await session.call_tool("create_voice", args)
        assert not result.is_error, result.content
        made_voice = json.loads(result.content[0].text)
        replay = await session.call_tool("create_voice", args)
        assert json.loads(replay.content[0].text) == made_voice
        for _ in range(100):
            voice = await session.call_tool("get_voice", {"id": made_voice["id"]})
            if json.loads(voice.content[0].text)["status"] == "succeeded":
                break
            await asyncio.sleep(0.002)
        listed = await session.call_tool("list_voices", {})
        assert made_voice["id"] in {v["id"] for v in json.loads(listed.content[0].text)["data"]}
        estimated = await session.call_tool("estimate_cost", {"modality": "audio", **speech(made_voice["id"])})
        assert json.loads(estimated.content[0].text)["estimated_cost"] == 0.005
        result = await session.call_tool("create_audio", {**speech(made_voice["id"]), "idempotency_key": "mcp-speech"})
        assert not result.is_error, result.content
        id = json.loads(result.content[0].text)["id"]
        for _ in range(100):
            audio = await session.call_tool("get_audio", {"id": id})
            if json.loads(audio.content[0].text)["status"] == "succeeded":
                break
            await asyncio.sleep(0.002)
        assert json.loads(audio.content[0].text)["outputs"][0]["mime_type"] == "audio/mpeg"
