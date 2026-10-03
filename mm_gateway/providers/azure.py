"""Azure Text to Speech through the official Cognitive Services Speech SDK."""

from __future__ import annotations

import asyncio
import re
import sys
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree as ET

from mm_gateway.core.exceptions import ProviderNotConfiguredError, ProviderRequestError
from mm_gateway.providers._speech import SpeechAudioTaskMixin, inline_audio
from mm_gateway.schemas.api import AudioParameters
from mm_gateway.schemas.audio import AudioUsage, UnifiedAudioRequest, UnifiedVoiceRequest

_MP3_FORMATS = {
    (16000, 32): "Audio16Khz32KBitRateMonoMp3",
    (16000, 64): "Audio16Khz64KBitRateMonoMp3",
    (16000, 128): "Audio16Khz128KBitRateMonoMp3",
    (24000, 48): "Audio24Khz48KBitRateMonoMp3",
    (24000, 96): "Audio24Khz96KBitRateMonoMp3",
    (24000, 160): "Audio24Khz160KBitRateMonoMp3",
    (48000, 96): "Audio48Khz96KBitRateMonoMp3",
    (48000, 192): "Audio48Khz192KBitRateMonoMp3",
}
_PCM_RATES = {8000: "8Khz", 16000: "16Khz", 22050: "22050Hz",
              24000: "24Khz", 44100: "44100Hz", 48000: "48Khz"}
_OPUS_RATES = {16000: "16Khz", 24000: "24Khz", 48000: "48Khz"}
_SSML_MAX_BYTES = 64 * 1024


class AzureProvider(SpeechAudioTaskMixin):
    name = "azure"
    # Azure selects a voice rather than accepting a model id on the SDK call.
    audio_models = ["azure-tts"]
    default_voice = "en-US-AvaMultilingualNeural"

    def __init__(self, backend):
        super().__init__(backend)
        region = backend.extra.get("region")
        if not backend.api_key or not (backend.base_url or region):
            raise ProviderNotConfiguredError(self.name, "Azure speech requires an API key and a region or base_url endpoint.")
        if region is not None and (not isinstance(region, str) or not region.strip()):
            raise ValueError("Azure speech region must be a nonempty string.")
        # Import only for configured Azure backends; no service call at startup.
        import azure.cognitiveservices.speech as speechsdk

        self._sdk = speechsdk
        self._proxy = self._proxy_settings(backend.extra.get("outbound_proxy"))

    @staticmethod
    def _proxy_settings(url: str | None) -> tuple[str, int, str | None, str | None] | None:
        if not url:
            return None
        if sys.platform == "darwin":
            raise ValueError("Azure Speech SDK outbound proxies are unavailable on macOS.")
        parsed = urlsplit(url)
        if parsed.scheme != "http" or not parsed.hostname or parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise ValueError("Azure Speech SDK outbound_proxy requires an HTTP proxy URL.")
        return (parsed.hostname, parsed.port or 80,
                unquote(parsed.username) if parsed.username is not None else None,
                unquote(parsed.password) if parsed.password is not None else None)

    @staticmethod
    def _output_format(p: AudioParameters) -> tuple[str, int]:
        fmt = p.file_format or "mp3"
        rate = p.sample_rate_hz or 24000
        if fmt == "mp3":
            if p.sample_rate_hz is None and p.bitrate_kbps is not None:
                rate = next((r for r in (24000, 16000, 48000) if (r, p.bitrate_kbps) in _MP3_FORMATS), rate)
            bitrate = p.bitrate_kbps or {16000: 32, 24000: 48, 48000: 96}.get(rate)
            return _MP3_FORMATS[(rate, bitrate)], rate
        if p.bitrate_kbps is not None:
            raise ValueError("Bitrate is configurable only for Azure MP3 output.")
        if fmt in {"pcm", "wav"}:
            prefix = "Raw" if fmt == "pcm" else "Riff"
            return f"{prefix}{_PCM_RATES[rate]}16BitMonoPcm", rate
        if fmt == "opus":
            return f"Ogg{_OPUS_RATES[rate]}16BitMonoOpus", rate
        raise ValueError("Unsupported Azure speech output format.")

    def _ssml(self, request: UnifiedAudioRequest) -> str:
        voice_id = self._voice_id(request)
        # Standard voice names begin with their native locale. Operators using
        # a custom voice name can configure its locale via extra.language.
        locale = re.match(r"^[a-z]{2,3}-[A-Za-z0-9]{2,8}(?=-)", voice_id)
        default_language = self.backend.extra.get("language") or (locale.group() if locale else "en-US")
        root = ET.Element("speak", {"version": "1.0", "xmlns": "http://www.w3.org/2001/10/synthesis",
                                    "xml:lang": default_language})
        node = ET.SubElement(root, "voice", {"name": voice_id})
        if request.parameters.language is not None and (
            "Multilingual" in voice_id or request.parameters.language != default_language
        ):
            node = ET.SubElement(node, "lang", {"xml:lang": request.parameters.language})
        if request.parameters.speed is not None:
            node = ET.SubElement(node, "prosody", {"rate": f"{request.parameters.speed:g}"})
        node.text = request.text
        # Text and attribute values are escaped; input never becomes SSML code.
        return ET.tostring(root, encoding="unicode")

    def audio_request_error(self, request, model):
        if isinstance(request, UnifiedVoiceRequest) or request.native_voice_id:
            return "Azure speech does not support gateway voice cloning."
        p = request.parameters
        if p.voice not in self.voice_presets():
            return "Voice preset is unavailable for this model."
        if p.delivery == "remote" or p.instructions is not None or p.seed is not None:
            return "Azure speech does not support remote delivery, separate instructions or seed."
        if p.speed is not None and not 0.5 <= p.speed <= 2:
            return "Azure speech supports speed from 0.5 to 2."
        if p.language is not None and "-" not in p.language:
            return "Azure speech requires a language locale such as en-US or fr-FR."
        if p.language is not None:
            voice_id = self._voice_id(request)
            locale = re.match(r"^([a-z]{2,3}-[A-Za-z0-9]{2,8})-[A-Za-z]+Neural$", voice_id)
            if locale and "Multilingual" not in voice_id and p.language != locale.group(1):
                return "The selected Azure voice does not support the requested language locale."
        try:
            self._output_format(p)
        except (KeyError, ValueError):
            return "Azure speech does not support the requested audio encoding combination."
        # The SDK sends a 64 KiB WebSocket message. Measure the complete escaped
        # document in bytes, including non-ASCII text and preset voice names.
        if len(self._ssml(request).encode("utf-8")) > _SSML_MAX_BYTES:
            return "Azure speech input exceeds the 64 KiB synthesis message limit."
        return None

    async def _synthesize(self, request: UnifiedAudioRequest):
        # SDK ResultFuture.get() blocks; config, synthesis and native cleanup
        # all run off the gateway's event loop, with a fresh config per request.
        return await asyncio.to_thread(self._synthesize_sync, request)

    def _synthesize_sync(self, request: UnifiedAudioRequest):
        sdk = self._sdk
        endpoint = self.backend.base_url
        kwargs = {"endpoint": endpoint} if endpoint else {"region": self.backend.extra["region"]}
        config = sdk.SpeechConfig(subscription=self.backend.api_key, **kwargs)
        config.speech_synthesis_voice_name = self._voice_id(request)
        if endpoint_id := self.backend.extra.get("endpoint_id"):
            config.endpoint_id = endpoint_id
        if language := self.backend.extra.get("language"):
            config.speech_synthesis_language = language
        if self._proxy is not None:
            config.set_proxy(*self._proxy)
        output_format, rate = self._output_format(request.parameters)
        config.set_speech_synthesis_output_format(getattr(sdk.SpeechSynthesisOutputFormat, output_format))
        # None returns in-memory audio and avoids using a server audio device.
        synthesizer = sdk.SpeechSynthesizer(speech_config=config, audio_config=None)
        if request.parameters.speed is not None or request.parameters.language is not None:
            result = synthesizer.speak_ssml_async(self._ssml(request)).get()
        else:
            result = synthesizer.speak_text_async(request.text).get()
        if result.reason != sdk.ResultReason.SynthesizingAudioCompleted:
            # Cancellation details can include upstream credentials or content.
            raise ProviderRequestError("Azure speech synthesis did not complete.", provider=self.name)
        duration = result.audio_duration.total_seconds()
        output = inline_audio(result.audio_data, request.parameters.file_format or "mp3",
                              sample_rate_hz=rate, channels=1, duration_seconds=duration)
        return [output], AudioUsage(input_characters=len(request.text), duration_seconds=duration)
