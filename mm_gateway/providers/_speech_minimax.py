"""MiniMax T2A v2 and file-upload/voice-clone translation."""

from __future__ import annotations

import uuid

from mm_gateway.providers._speech import SpeechTaskMixin, inline_audio
from mm_gateway.providers._speech_media import load_voice_sample
from mm_gateway.schemas.api import AudioOutput
from mm_gateway.schemas.audio import AudioUsage, UnifiedAudioRequest, UnifiedVoiceRequest

_LANGUAGES = dict(zip(
    "zh en ar ru es fr pt de tr nl uk vi id ja it ko th pl ro el cs fi hi bg da he ms fa sk sv hr tl hu no sl ca nn ta af".split(),
    "Chinese English Arabic Russian Spanish French Portuguese German Turkish Dutch Ukrainian Vietnamese Indonesian Japanese Italian Korean Thai Polish Romanian Greek Czech Finnish Hindi Bulgarian Danish Hebrew Malay Persian Slovak Swedish Croatian Filipino Hungarian Norwegian Slovenian Catalan Nynorsk Tamil Afrikaans".split(),
    strict=True,
))


def _result(response):
    data = response.json()
    if (data.get("base_resp") or {}).get("status_code", 0) != 0:
        raise ValueError("MiniMax rejected the speech request")
    return data


class MiniMaxSpeechMixin(SpeechTaskMixin):
    audio_models = ["speech-2.8-hd", "speech-2.8-turbo", "speech-2.6-hd", "speech-2.6-turbo",
                    "speech-02-hd", "speech-02-turbo", "speech-01-hd", "speech-01-turbo"]
    default_voice = "English_expressive_narrator"
    default_speech_base = "https://api.minimax.io"

    def audio_request_error(self, request, model):
        if isinstance(request, UnifiedVoiceRequest):
            if len(request.samples) != 1:
                return "This cloning model requires one sample."
            if request.consent.recording_uri or request.parameters.description is not None:
                return "This cloning model does not accept a separate consent recording or description."
            return None
        p = request.parameters
        if not request.native_voice_id and p.voice not in self.voice_presets():
            return "Voice preset is unavailable for this model."
        if p.instructions is not None or p.seed is not None:
            return "This model does not support separate instructions or a seed."
        if p.file_format == "aac":
            return "This model does not support AAC output."
        if p.sample_rate_hz is not None and p.sample_rate_hz not in {8000, 16000, 22050, 24000, 32000, 44100}:
            return "This model does not support the requested sample rate."
        if p.speed is not None and not 0.5 <= p.speed <= 2:
            return "This model supports speed from 0.5 to 2."
        if p.bitrate_kbps is not None and (p.bitrate_kbps not in {32, 64, 128, 256}
                                         or p.file_format not in {None, "mp3"}):
            return "This model only supports 32, 64, 128 or 256 kbps MP3 bitrate."
        if p.language and p.language not in _LANGUAGES and p.language != "zh-yue":
            return "This model does not support the requested language code."
        if p.language in {"fa", "tl", "ta"} and model.startswith(("speech-01", "speech-02")):
            return "This speech model does not support the requested language."
        return None

    async def _synthesize(self, request: UnifiedAudioRequest):
        p = request.parameters
        fmt = p.file_format or "mp3"
        voice = {"voice_id": self._voice_id(request)}
        if p.speed is not None:
            voice["speed"] = p.speed
        audio = {"format": fmt}
        if p.sample_rate_hz is not None:
            audio["sample_rate"] = p.sample_rate_hz
        if p.bitrate_kbps is not None:
            audio["bitrate"] = p.bitrate_kbps * 1000
        body = {"model": request.model, "text": request.text, "stream": False,
                "voice_setting": voice, "audio_setting": audio,
                "output_format": "url" if p.delivery == "remote" else "hex"}
        if p.language:
            body["language_boost"] = "Chinese,Yue" if p.language == "zh-yue" else _LANGUAGES[p.language]
        result = _result(await self._speech_request("v1/t2a_v2", json=body))
        data = result.get("data") or {}
        if data.get("status") != 2 or not data.get("audio"):
            # The non-streaming endpoint has no job handle to poll. Never issue
            # the synthesis again in response to an incomplete result.
            raise ValueError("MiniMax returned incomplete audio")
        extra = result.get("extra_info") or {}
        duration = extra["audio_length"] / 1000 if extra.get("audio_length") is not None else None
        details = {"duration_seconds": duration, "sample_rate_hz": extra.get("audio_sample_rate"),
                   "channels": extra.get("audio_channel")}
        if p.delivery == "remote":
            if not data["audio"].startswith(("http://", "https://")):
                raise ValueError("MiniMax returned no remote audio URL")
            from mm_gateway.providers._speech import MIME_TYPES
            output = AudioOutput(uri=data["audio"], mime_type=MIME_TYPES[fmt], **details)
        else:
            output = inline_audio(bytes.fromhex(data["audio"]), fmt, **details)
        usage = AudioUsage(input_characters=extra.get("usage_characters", len(request.text)), duration_seconds=duration)
        return [output], usage

    async def _clone(self, request: UnifiedVoiceRequest):
        sample = await load_voice_sample(request.samples[0].uri, proxy_url=self.backend.extra.get("outbound_proxy"))
        if sample[2] not in {"audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/mp4", "audio/x-m4a"}:
            raise ValueError("MiniMax cloning requires MP3, M4A or WAV")
        if len(sample[1]) > 20_000_000:
            raise ValueError("MiniMax cloning sample exceeds 20 MB")
        upload = _result(await self._speech_request("v1/files/upload", data={"purpose": "voice_clone"},
                                                  files={"file": sample}))
        native_id = f"Gateway{uuid.uuid4().hex}"
        body = {"file_id": int(upload["file"]["file_id"]), "voice_id": native_id}
        if request.parameters.remove_background_noise is not None:
            body["need_noise_reduction"] = request.parameters.remove_background_noise
        result = _result(await self._speech_request("v1/voice_clone", json=body))
        sensitive = result.get("input_sensitive")
        if sensitive is True or (isinstance(sensitive, dict) and sensitive.get("type", 0) != 0):
            raise ValueError("MiniMax rejected the cloning sample")
        return native_id, False
