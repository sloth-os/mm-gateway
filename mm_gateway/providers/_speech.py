"""Single-execution speech tasks and authenticated REST transport for adapters."""

from __future__ import annotations

import asyncio
import base64
import re
import time
import uuid
from typing import Any

import httpx

from mm_gateway.core.base import VoiceCloneProvider
from mm_gateway.core.exceptions import ProviderRequestError, ValidationError
from mm_gateway.observability.logging import get_logger
from mm_gateway.providers._http import _map_status, make_client
from mm_gateway.schemas.api import AudioOutput
from mm_gateway.schemas.audio import (
    AudioUsage, UnifiedAudioRequest, UnifiedAudioTask, UnifiedVoiceRequest, UnifiedVoiceTask,
)

log = get_logger("provider.speech")
MIME_TYPES = {"mp3": "audio/mpeg", "wav": "audio/wav", "pcm": "audio/pcm",
              "flac": "audio/flac", "opus": "audio/ogg", "aac": "audio/aac"}


def inline_audio(blob: bytes, fmt: str, **details: Any) -> AudioOutput:
    if not blob:
        raise ProviderRequestError("Speech provider returned empty audio.")
    mime = MIME_TYPES[fmt]
    return AudioOutput(uri=f"data:{mime};base64,{base64.b64encode(blob).decode()}",
                       mime_type=mime, **details)


class SpeechTaskMixin(VoiceCloneProvider):
    """Each adapter/account owns its tasks. Failed synthesis is never re-issued."""

    default_voice = ""
    default_speech_base = ""

    def voice_presets(self) -> dict[str, str]:
        configured = self.backend.extra.get("voice_presets", {})
        if not isinstance(configured, dict) or any(
            not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", key)
            or key.startswith("voice_")
            or not isinstance(value, str) or not value for key, value in configured.items()
        ):
            raise ValueError("voice_presets must map gateway preset names to native voice ids; voice_ is reserved")
        return {"default": self.default_voice, **configured}

    def _voice_id(self, request: UnifiedAudioRequest) -> str:
        return request.native_voice_id or self.voice_presets()[request.parameters.voice]

    def _state(self) -> dict[str, dict[str, Any]]:
        if "_speech_tasks" not in self.__dict__:
            self._speech_tasks: dict[str, dict[str, Any]] = {}
        return self._speech_tasks

    async def _speech_request(self, path: str, *, method: str = "POST", **kwargs: Any) -> httpx.Response:
        base = self.backend.extra.get("audio_base_url") or self.backend.base_url or self.default_speech_base
        headers = {"xi-api-key": self.backend.api_key} if self.name == "elevenlabs" else {
            "Authorization": f"Bearer {self.backend.api_key}"
        }
        try:
            async with make_client(base, headers=headers, timeout=240,
                                   proxy_url=self.backend.extra.get("outbound_proxy")) as client:
                request = client.build_request(method, path, **kwargs)
                # httpx multipart bodies are streaming even when their source
                # is bytes. Materialize before the shared logging hook reads
                # request.content (sample uploads are already bounded).
                await request.aread()
                response = await client.send(request)
        except httpx.HTTPError as exc:
            raise ProviderRequestError(f"Speech transport failed: {exc}", provider=self.name) from exc
        if response.status_code >= 400:
            raise ProviderRequestError(f"Speech upstream returned {response.status_code}", provider=self.name,
                                       status_code=_map_status(response.status_code))
        return response

    def _validate_request(self, request: UnifiedAudioRequest | UnifiedVoiceRequest) -> None:
        if reason := self.audio_request_error(request, request.model):
            raise ValidationError(reason)

    async def _verification_required(self, native_voice_id: str) -> bool:
        """Adapters that require upstream verification observe it without re-cloning."""
        return True

    async def create_audio_task(self, request: UnifiedAudioRequest) -> UnifiedAudioTask:
        self._validate_request(request)
        task = UnifiedAudioTask(task_id=f"speech-{uuid.uuid4().hex}", provider=self.name,
                                model=request.model, status="pending", created_at=int(time.time()))
        self._state()[task.task_id] = {"request": request.model_copy(deep=True), "task": task, "lock": asyncio.Lock()}
        return task.model_copy(deep=True)

    async def create_voice_task(self, request: UnifiedVoiceRequest) -> UnifiedVoiceTask:
        self._validate_request(request)
        task = UnifiedVoiceTask(task_id=f"clone-{uuid.uuid4().hex}", provider=self.name,
                                model=request.model, name=request.parameters.name, status="pending",
                                account_id=self.backend.extra.get("__account_id", "default"),
                                created_at=int(time.time()))
        self._state()[task.task_id] = {"request": request.model_copy(deep=True), "task": task, "lock": asyncio.Lock()}
        return task.model_copy(deep=True)

    async def _observe(self, task_id: str, kind: type) -> UnifiedAudioTask | UnifiedVoiceTask:
        record = self._state().get(task_id)
        if record is None or not isinstance(record["task"], kind):
            raise ProviderRequestError("Speech task not found", provider=self.name, status_code=404)
        async with record["lock"]:
            task = record["task"]
            if task.status in {"succeeded", "failed"}:
                return task.model_copy(deep=True)
            if isinstance(task, UnifiedVoiceTask) and task.native_voice_id and task.verification_required:
                task.verification_required = await self._verification_required(task.native_voice_id)
                if not task.verification_required:
                    task.status = "succeeded"
                    task.completed_at = int(time.time())
                return task.model_copy(deep=True)
            task.status = "running"
            try:
                if isinstance(task, UnifiedAudioTask):
                    task.outputs, task.usage = await self._synthesize(record["request"])
                    if not task.outputs:
                        raise ProviderRequestError("Speech provider returned no outputs", provider=self.name)
                else:
                    task.native_voice_id, task.verification_required = await self._clone(record["request"])
                    if not task.native_voice_id:
                        raise ProviderRequestError("Cloning returned no voice id", provider=self.name)
                task.status = "running" if isinstance(task, UnifiedVoiceTask) and task.verification_required else "succeeded"
            except Exception as exc:
                log.warning("speech_task_failed", provider=self.name, task_id=task_id, error=str(exc))
                task.status = "failed"
                task.error = "Voice cloning failed." if isinstance(task, UnifiedVoiceTask) else "Speech generation failed."
            if task.status in {"succeeded", "failed"}:
                task.completed_at = int(time.time())
            record.pop("request", None)
            return task.model_copy(deep=True)

    async def get_audio_task(self, task_id: str) -> UnifiedAudioTask:
        return await self._observe(task_id, UnifiedAudioTask)

    async def get_voice_task(self, task_id: str) -> UnifiedVoiceTask:
        return await self._observe(task_id, UnifiedVoiceTask)
