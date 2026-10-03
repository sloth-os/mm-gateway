"""ElevenLabs text-to-speech and Instant Voice Cloning."""

from __future__ import annotations

import io
import wave
from urllib.parse import quote

from mm_gateway.providers._speech import SpeechTaskMixin, inline_audio
from mm_gateway.providers._speech_media import load_voice_sample
from mm_gateway.schemas.audio import AudioUsage, UnifiedAudioRequest, UnifiedVoiceRequest

_FORMATS = {
    "mp3_22050_32", "mp3_24000_48", "mp3_44100_32", "mp3_44100_64", "mp3_44100_96",
    "mp3_44100_128", "mp3_44100_192", "opus_48000_32", "opus_48000_64", "opus_48000_96",
    "opus_48000_128", "opus_48000_192",
    *(f"pcm_{rate}" for rate in (8000, 16000, 22050, 24000, 32000, 44100, 48000)),
}


class ElevenLabsSpeechMixin(SpeechTaskMixin):
    audio_models = ["eleven_multilingual_v2", "eleven_flash_v2_5", "eleven_turbo_v2_5", "eleven_v3"]
    default_voice = "JBFqnCBsd6RMkjVDRZzb"
    default_speech_base = "https://api.elevenlabs.io"

    @staticmethod
    def _speech_format(p):
        fmt = p.file_format or "mp3"
        rate = p.sample_rate_hz or (48000 if fmt == "opus" else 44100)
        codec = "pcm" if fmt == "wav" else fmt
        bitrate = p.bitrate_kbps or ({22050: 32, 24000: 48}.get(rate, 128) if codec == "mp3" else 128)
        return (f"{codec}_{rate}" if codec == "pcm" else f"{codec}_{rate}_{bitrate}"), rate

    def audio_request_error(self, request, model):
        if isinstance(request, UnifiedVoiceRequest):
            if request.consent.recording_uri:
                return "This cloning model does not accept a separate consent recording."
            return None
        p = request.parameters
        if not request.native_voice_id and p.voice not in self.voice_presets():
            return "Voice preset is unavailable for this model."
        fmt, _ = self._speech_format(p)
        if fmt not in _FORMATS or (p.file_format in {"wav", "pcm"} and p.bitrate_kbps is not None):
            return "This model does not support the requested audio encoding combination."
        if p.delivery == "remote" or p.instructions is not None:
            return "This model does not support remote delivery or separate instructions."
        if p.language and model == "eleven_multilingual_v2":
            return "This speech model detects language from text and does not support a language control."
        if p.language and ("-" in p.language or len(p.language) != 2):
            return "This model accepts an ISO language code without a region."
        if p.speed is not None and not 0.7 <= p.speed <= 1.2:
            return "This model supports speed from 0.7 to 1.2."
        return None

    async def _synthesize(self, request: UnifiedAudioRequest):
        p = request.parameters
        fmt, rate = self._speech_format(p)
        body = {"model_id": request.model, "text": request.text}
        if p.language is not None:
            body["language_code"] = p.language
        if p.speed is not None:
            body["voice_settings"] = {"speed": p.speed}
        if p.seed is not None:
            body["seed"] = p.seed
        response = await self._speech_request(f"v1/text-to-speech/{quote(self._voice_id(request), safe='')}",
                                             params={"output_format": fmt}, json=body)
        blob = response.content
        if not blob:
            raise ValueError("Speech provider returned empty audio")
        if p.file_format == "wav":
            if len(blob) % 2:
                raise ValueError("Invalid PCM byte count")
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(rate)
                writer.writeframes(blob)
            blob = buffer.getvalue()
        output = inline_audio(blob, p.file_format or "mp3", sample_rate_hz=rate, channels=1)
        return [output], AudioUsage(input_characters=len(request.text))

    async def _clone(self, request: UnifiedVoiceRequest):
        samples = []
        total_bytes = 0
        for sample in request.samples:
            loaded = await load_voice_sample(sample.uri, proxy_url=self.backend.extra.get("outbound_proxy"))
            total_bytes += len(loaded[1])
            if total_bytes > 20 * 1024 * 1024:
                raise ValueError("Combined voice samples exceed 20 MiB")
            samples.append(loaded)
        data = {"name": request.parameters.name}
        if request.parameters.description is not None:
            data["description"] = request.parameters.description
        if request.parameters.remove_background_noise is not None:
            data["remove_background_noise"] = str(request.parameters.remove_background_noise).lower()
        response = await self._speech_request("v1/voices/add", data=data,
                                             files=[("files", sample) for sample in samples])
        result = response.json()
        return result.get("voice_id"), bool(result.get("requires_verification", False))

    async def _verification_required(self, native_voice_id: str) -> bool:
        result = (await self._speech_request(f"v1/voices/{quote(native_voice_id, safe='')}", method="GET")).json()
        verification = result.get("voice_verification") or {}
        # Absence of a verification observation must not unlock a gated clone.
        return not (verification.get("is_verified") is True or verification.get("requires_verification") is False)
