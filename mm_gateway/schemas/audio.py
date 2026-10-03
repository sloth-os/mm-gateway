"""Canonical speech and voice-clone requests; private voice affinity stays internal."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from mm_gateway.schemas.api import (
    AudioOutput, AudioParameters, TaskStatus, VoiceConsent, VoiceParameters, VoiceSampleInput,
)


class UnifiedAudioRequest(BaseModel):
    model: str
    text: str
    parameters: AudioParameters = Field(default_factory=AudioParameters)
    provider: str | None = None
    # Resolved only by the gateway after authorizing the voice resource.
    native_voice_id: str | None = None
    voice_backend: str | None = None
    voice_account: str | None = None


class UnifiedVoiceRequest(BaseModel):
    model: str
    samples: list[VoiceSampleInput]
    parameters: VoiceParameters
    consent: VoiceConsent
    provider: str | None = None


class AudioUsage(BaseModel):
    cost: float | None = None
    input_characters: int | None = None
    duration_seconds: float | None = None


class UnifiedAudioTask(BaseModel):
    task_id: str
    provider: str
    model: str
    status: TaskStatus
    outputs: list[AudioOutput] = Field(default_factory=list)
    error: str | None = None
    usage: AudioUsage | None = None
    routing: dict[str, Any] | None = None
    created_at: int | None = None
    completed_at: int | None = None


class UnifiedVoiceTask(BaseModel):
    task_id: str
    provider: str
    model: str
    name: str
    status: TaskStatus
    # Never serialized into the public response.
    native_voice_id: str | None = None
    account_id: str = "default"
    verification_required: bool = False
    error: str | None = None
    usage: AudioUsage | None = None
    routing: dict[str, Any] | None = None
    created_at: int | None = None
    completed_at: int | None = None
