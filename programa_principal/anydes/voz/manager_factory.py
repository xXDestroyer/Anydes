from __future__ import annotations

from anydes.cognition.voice import VoiceManager
from anydes.cognition.live_runtime.interfaces import MetricsCollector
from anydes.nucleo.config import Config
from anydes.voz.identity import VoiceProfile, load_voice_identity
from anydes.voz.tts import build_tts


def build_voice_manager(
    config: Config,
    metrics: MetricsCollector | None = None,
) -> VoiceManager:
    """Composition root that keeps concrete TTS engines behind VoiceManager."""

    if config.new_voice_enabled:
        from anydes.voz.nueva.factory import build_new_voice_manager

        return build_new_voice_manager(config)

    return VoiceManager.from_backend(
        build_tts(config),
        profile=VoiceProfile(identity=load_voice_identity(config)),
        provider_name=config.tts_provider,
        # A backend with a different speaker would break Anydes' VoiceIdentity.
        # Voice failures therefore degrade to text instead of changing voices.
        fallback_backends=(),
        metrics=metrics,
    )
