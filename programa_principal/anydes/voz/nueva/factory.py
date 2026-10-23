from __future__ import annotations

import os
from dataclasses import replace

from anydes.nucleo.config import Config, PROJECT_DIR
from anydes.voz.identity import load_voice_identity
from anydes.voz.nueva.audio_pipeline import AudioPipeline
from anydes.voz.nueva.common_phrase_cache import CommonPhraseAudioCache
from anydes.voz.nueva.engines import (
    EmergencyFallbackProvider,
    FasterQwen3TTSProvider,
    KokoroProvider,
    PiperProvider,
    Qwen3TTSBaseProvider,
    TTSEngine,
    XTTSProvider,
)
from anydes.voz.nueva.manager import NewVoiceManager


def build_new_voice_manager(config: Config) -> NewVoiceManager:
    if config.vram_safety_enabled:
        os.environ["ANYDES_VRAM_TTS_METRICS_PATH"] = str(config.vram_tts_metrics_path)
    identity = load_voice_identity(config)
    if config.new_voice_reference_audio.is_file():
        identity = replace(
            identity,
            reference_audio_path=str(config.new_voice_reference_audio),
        )
    engine = build_engine(config)
    phrase_cache = None
    if config.new_voice_common_phrase_cache_enabled:
        phrase_cache = CommonPhraseAudioCache(config.new_voice_common_phrase_cache_manifest)
    emergency = None
    if config.new_voice_allow_emergency_fallback and engine.engine_id != "piper":
        emergency = EmergencyFallbackProvider(
            config.piper_model_path,
            config.piper_config_path,
            voice_id="anydes-emergency-authorized",
        )
    manager = NewVoiceManager(
        identity=identity,
        engine=engine,
        emergency_engine=emergency,
        audio=AudioPipeline(
            sample_rate=config.new_voice_sample_rate,
            channels=config.new_voice_channels,
        ),
        delivery_mode=config.new_voice_delivery_mode,
        segmentation_strategy=config.new_voice_segmentation_strategy,
        phrase_cache=phrase_cache,
        idle_gpu_cache_release_seconds=config.vram_tts_idle_release_seconds,
    )
    if config.new_voice_warmup_on_start:
        manager.warmup_async()
    return manager


def build_engine(config: Config) -> TTSEngine:
    selected = config.new_voice_engine.strip().lower()
    worker_script = PROJECT_DIR / "scripts" / "voice_engine_worker.py"
    common = {
        "worker_python": config.new_voice_worker_python,
        "worker_script": worker_script,
        "device": config.new_voice_device,
    }
    if selected == "xtts":
        return XTTSProvider(model=config.new_voice_xtts_model, **common)
    if selected == "kokoro":
        return KokoroProvider(voice_id=config.new_voice_kokoro_voice, **common)
    if selected == "piper":
        return PiperProvider(
            config.piper_model_path,
            config.piper_config_path,
            voice_id=config.piper_model_path.stem,
        )
    if selected in {"qwen3", "qwen3-base", "qwen3-tts-base"}:
        return Qwen3TTSBaseProvider(
            model=config.new_voice_qwen_model,
            manifest_path=config.new_voice_reference_manifest,
            reference_audio=config.new_voice_reference_audio,
            **common,
        )
    if selected in {"qwen3-faster", "qwen3-tts-faster"}:
        return FasterQwen3TTSProvider(
            model=config.new_voice_qwen_model,
            manifest_path=config.new_voice_reference_manifest,
            reference_audio=config.new_voice_reference_audio,
            **common,
        )
    raise ValueError(f"Unsupported new voice engine: {selected}")
