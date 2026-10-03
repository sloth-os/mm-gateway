"""OpenAI speech and consent-backed custom voices over the documented REST API."""

from __future__ import annotations

from mm_gateway.providers._speech import SpeechTaskMixin, inline_audio
from mm_gateway.providers._speech_media import load_voice_sample
from mm_gateway.schemas.audio import AudioUsage, UnifiedAudioRequest, UnifiedVoiceRequest


class OpenAISpeechMixin(SpeechTaskMixin):
    audio_models = ["gpt-4o-mini-tts", "gpt-4o-mini-tts-2025-12-15", "tts-1", "tts-1-hd"]
    default_voice = "alloy"
    default_speech_base = "https://api.openai.com/v1"

    def audio_request_error(self, request, model):
        if isinstance(request, UnifiedVoiceRequest):
            if not self.backend.extra.get("voice_cloning_enabled", False):
                return "Voice cloning is not enabled for this account."
            if model in {"tts-1", "tts-1-hd"}:
                return "This speech model does not support cloned voices."
            if len(request.samples) != 1 or not request.consent.recording_uri:
                return "This cloning model requires one sample and a consent recording."
            if request.parameters.description is not None or request.parameters.remove_background_noise is not None:
                return "This cloning model does not support description or noise removal."
            return None
        p = request.parameters
        if request.native_voice_id and model in {"tts-1", "tts-1-hd"}:
            return "This speech model does not support cloned voices."
        if not request.native_voice_id and p.voice not in self.voice_presets():
            return "Voice preset is unavailable for this model."
        if p.delivery == "remote" or p.bitrate_kbps is not None or p.seed is not None or p.language is not None:
            return "This speech model does not support the requested delivery, bitrate, seed or language control."
        if p.sample_rate_hz is not None and not (p.file_format == "pcm" and p.sample_rate_hz == 24000):
            return "This speech model only exposes a fixed 24000 Hz PCM sample rate."
        if p.instructions and model in {"tts-1", "tts-1-hd"}:
            return "This speech model does not support instructions."
        return None

    async def _synthesize(self, request: UnifiedAudioRequest):
        p = request.parameters
        fmt = p.file_format or "mp3"
        voice = {"id": request.native_voice_id} if request.native_voice_id else self._voice_id(request)
        body = {"model": request.model, "input": request.text, "voice": voice, "response_format": fmt}
        if p.speed is not None:
            body["speed"] = p.speed
        if p.instructions is not None:
            body["instructions"] = p.instructions
        response = await self._speech_request("audio/speech", json=body)
        details = {"sample_rate_hz": 24000, "channels": 1} if fmt == "pcm" else {}
        return [inline_audio(response.content, fmt, **details)], AudioUsage(input_characters=len(request.text))

    async def _clone(self, request: UnifiedVoiceRequest):
        proxy = self.backend.extra.get("outbound_proxy")
        sample = await load_voice_sample(request.samples[0].uri, proxy_url=proxy)
        recording = await load_voice_sample(request.consent.recording_uri, proxy_url=proxy)
        if max(len(sample[1]), len(recording[1])) > 10 * 1024 * 1024:
            raise ValueError("Custom voice uploads must be at most 10 MiB")
        consent = (await self._speech_request("audio/voice_consents",
                   data={"name": request.parameters.name, "language": request.consent.language},
                   files={"recording": recording})).json()
        result = (await self._speech_request("audio/voices", data={"name": request.parameters.name,
                                   "consent": consent["id"]}, files={"audio_sample": sample})).json()
        return result.get("id"), False
