from __future__ import annotations

import os
import site
import sys
from dataclasses import dataclass
from pathlib import Path

PROGRAM_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = PROGRAM_DIR.parent


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file(PROJECT_DIR / ".env")
_load_env_file(PROGRAM_DIR / ".env")


def _ensure_user_site_packages() -> None:
    try:
        user_site = site.getusersitepackages()
    except Exception:
        return
    if user_site and user_site not in sys.path:
        sys.path.append(user_site)


_ensure_user_site_packages()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default))
    return value if value.is_absolute() else PROGRAM_DIR / value


@dataclass(frozen=True)
class Config:
    profile_name: str = os.getenv("ANYDES_PROFILE_NAME", "Anydes")
    user_name: str = os.getenv("ANYDES_USER_NAME", "DemoUser")
    data_dir: Path = _env_path("ANYDES_DATA_DIR", "datos")
    avatar_model_dir: Path = _env_path(
        "ANYDES_AVATAR_MODEL_DIR",
        "../Anydes 3D Modelo",
    )
    avatar_adaptive_camera_enabled: bool = _env_bool(
        "ANYDES_AVATAR_ADAPTIVE_CAMERA_ENABLED",
        False,
    )
    personality_prompt_path: Path = _env_path(
        "ANYDES_PERSONALITY_PROMPT",
        "anydes/personalidad/prompt_principal.md",
    )
    require_confirmation: bool = _env_bool("ANYDES_REQUIRE_CONFIRMATION", True)
    network_enabled: bool = _env_bool("ANYDES_NETWORK_ENABLED", False)
    external_ai_allowed: bool = _env_bool("ANYDES_ALLOW_EXTERNAL_AI", False)
    external_vision_allowed: bool = _env_bool("ANYDES_ALLOW_EXTERNAL_VISION", False)
    weather_allowed: bool = _env_bool("ANYDES_ALLOW_WEATHER", False)
    external_tts_allowed: bool = _env_bool("ANYDES_ALLOW_EXTERNAL_TTS", False)
    external_stt_allowed: bool = _env_bool("ANYDES_ALLOW_EXTERNAL_STT", False)
    ai_provider: str = os.getenv("ANYDES_AI_PROVIDER", "local")
    ai_model: str = os.getenv("ANYDES_AI_MODEL", "gpt-5-mini")
    ollama_url: str = os.getenv("ANYDES_OLLAMA_URL", "http://127.0.0.1:11434")
    ollama_turn_timeout_seconds: float = float(os.getenv("ANYDES_OLLAMA_TURN_TIMEOUT_SECONDS", "45"))
    ollama_optional_timeout_seconds: float = float(os.getenv("ANYDES_OLLAMA_OPTIONAL_TIMEOUT_SECONDS", "7"))
    ollama_expression_revisions: int = int(os.getenv("ANYDES_OLLAMA_EXPRESSION_REVISIONS", "2"))
    ollama_grounded_think: bool = _env_bool("ANYDES_OLLAMA_GROUNDED_THINK", False)
    groq_model: str = os.getenv("ANYDES_GROQ_MODEL", "llama-3.3-70b-versatile")
    groq_base_url: str = os.getenv("ANYDES_GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    relationship_reasoning_enabled: bool = _env_bool(
        "ANYDES_RELATIONSHIP_REASONING_ENABLED",
        True,
    )
    self_reasoning_enabled: bool = _env_bool("ANYDES_SELF_REASONING_ENABLED", True)
    reasoning_mode: str = os.getenv("ANYDES_REASONING_MODE", "native")
    unified_inference_mode: str = os.getenv("ANYDES_UNIFIED_INFERENCE_MODE", "off")
    conversation_mode: str = os.getenv("ANYDES_CONVERSATION_MODE", "stable")
    realtime_model: str = os.getenv("ANYDES_REALTIME_MODEL", "llama3.2:latest")
    realtime_context_token_budget: int = int(
        os.getenv("ANYDES_REALTIME_CONTEXT_TOKEN_BUDGET", "800")
    )
    realtime_max_output_tokens: int = int(
        os.getenv("ANYDES_REALTIME_MAX_OUTPUT_TOKENS", "56")
    )
    realtime_max_response_words: int = int(
        os.getenv("ANYDES_REALTIME_MAX_RESPONSE_WORDS", "72")
    )
    realtime_timeout_seconds: float = float(
        os.getenv("ANYDES_REALTIME_TIMEOUT_SECONDS", "2.2")
    )
    realtime_keep_alive: str = os.getenv("ANYDES_REALTIME_KEEP_ALIVE", "30m")
    realtime_num_ctx: int = int(os.getenv("ANYDES_REALTIME_NUM_CTX", "2048"))
    vision_keep_alive: str = os.getenv("ANYDES_VISION_KEEP_ALIVE", "")
    vision_coordinate_gpu_residency: bool = _env_bool(
        "ANYDES_VISION_COORDINATE_GPU_RESIDENCY",
        False,
    )
    vision_elastic_residency_enabled: bool = _env_bool(
        "ANYDES_VISION_ELASTIC_RESIDENCY_ENABLED",
        False,
    )
    vision_grace_period_seconds: float = float(
        os.getenv("ANYDES_VISION_GRACE_PERIOD_SECONDS", "120")
    )
    vram_safety_enabled: bool = _env_bool("ANYDES_VRAM_SAFETY_ENABLED", False)
    vram_warning_free_mb: float = float(os.getenv("ANYDES_VRAM_WARNING_FREE_MB", "500"))
    vram_critical_free_mb: float = float(os.getenv("ANYDES_VRAM_CRITICAL_FREE_MB", "250"))
    vram_emergency_floor_mb: float = float(os.getenv("ANYDES_VRAM_EMERGENCY_FLOOR_MB", "96"))
    vram_poll_interval_seconds: float = float(os.getenv("ANYDES_VRAM_POLL_INTERVAL_SECONDS", "2"))
    vram_action_cooldown_seconds: float = float(os.getenv("ANYDES_VRAM_ACTION_COOLDOWN_SECONDS", "8"))
    vram_tts_release_pressure: str = os.getenv(
        "ANYDES_VRAM_TTS_RELEASE_PRESSURE",
        "warning",
    ).strip().lower()
    vram_tts_idle_release_seconds: float = float(
        os.getenv("ANYDES_VRAM_TTS_IDLE_RELEASE_SECONDS", "0")
    )
    vram_tts_metrics_path: Path = _env_path(
        "ANYDES_VRAM_TTS_METRICS_PATH",
        "datos/telemetria/vram_tts_runtime.json",
    )
    vram_event_log_path: Path = _env_path(
        "ANYDES_VRAM_EVENT_LOG_PATH",
        "datos/telemetria/vram_safety.jsonl",
    )
    realtime_frame_max_age_seconds: float = float(
        os.getenv("ANYDES_REALTIME_FRAME_MAX_AGE_SECONDS", "12")
    )
    realtime_phase1_enabled: bool = _env_bool(
        "ANYDES_REALTIME_PHASE1_ENABLED",
        False,
    )
    realtime_phase2_enabled: bool = _env_bool(
        "ANYDES_REALTIME_PHASE2_ENABLED",
        False,
    )
    realtime_phase3_enabled: bool = _env_bool(
        "ANYDES_REALTIME_PHASE3_ENABLED",
        False,
    )
    realtime_phase4_enabled: bool = _env_bool(
        "ANYDES_REALTIME_PHASE4_ENABLED",
        False,
    )
    realtime_self_visual_enabled: bool = _env_bool(
        "ANYDES_REALTIME_SELF_VISUAL_ENABLED",
        False,
    )
    realtime_audio_stabilization_enabled: bool = _env_bool(
        "ANYDES_REALTIME_AUDIO_STABILIZATION_ENABLED",
        False,
    )
    new_voice_enabled: bool = _env_bool("ANYDES_NEW_VOICE_ENABLED", False)
    new_voice_engine: str = os.getenv("ANYDES_NEW_VOICE_ENGINE", "piper")
    new_voice_device: str = os.getenv("ANYDES_NEW_VOICE_DEVICE", "cpu")
    new_voice_sample_rate: int = int(os.getenv("ANYDES_NEW_VOICE_SAMPLE_RATE", "24000"))
    new_voice_channels: int = int(os.getenv("ANYDES_NEW_VOICE_CHANNELS", "1"))
    new_voice_delivery_mode: str = os.getenv(
        "ANYDES_NEW_VOICE_DELIVERY_MODE",
        "monolithic",
    )
    new_voice_warmup_on_start: bool = _env_bool(
        "ANYDES_NEW_VOICE_WARMUP_ON_START",
        False,
    )
    new_voice_segmentation_strategy: str = os.getenv(
        "ANYDES_NEW_VOICE_SEGMENTATION_STRATEGY",
        "adaptive",
    )
    new_voice_common_phrase_cache_enabled: bool = _env_bool(
        "ANYDES_NEW_VOICE_COMMON_PHRASE_CACHE_ENABLED",
        False,
    )
    new_voice_common_phrase_cache_manifest: Path = _env_path(
        "ANYDES_NEW_VOICE_COMMON_PHRASE_CACHE_MANIFEST",
        "datos/voz/canonical/anydes_canonical_voice_v1/common_phrases/manifest.json",
    )
    new_voice_allow_emergency_fallback: bool = _env_bool(
        "ANYDES_NEW_VOICE_ALLOW_EMERGENCY_FALLBACK",
        False,
    )
    new_voice_worker_python: Path = _env_path(
        "ANYDES_NEW_VOICE_WORKER_PYTHON",
        "../.venv-voice/Scripts/python.exe",
    )
    new_voice_reference_audio: Path = _env_path(
        "ANYDES_NEW_VOICE_REFERENCE_AUDIO",
        "datos/voz/referencia_anydes.mp3",
    )
    new_voice_reference_manifest: Path = _env_path(
        "ANYDES_NEW_VOICE_REFERENCE_MANIFEST",
        "datos/voz/canonical/anydes_canonical_voice_v1/manifest.json",
    )
    new_voice_qwen_model: Path = _env_path(
        "ANYDES_NEW_VOICE_QWEN_MODEL",
        (
            "modelos/qwen3-tts-cache/"
            "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/"
            "snapshots/fd4b254389122332181a7c3db7f27e918eec64e3"
        ),
    )
    new_voice_kokoro_voice: str = os.getenv("ANYDES_NEW_VOICE_KOKORO_VOICE", "ef_dora")
    new_voice_xtts_model: str = os.getenv(
        "ANYDES_NEW_VOICE_XTTS_MODEL",
        "tts_models/multilingual/multi-dataset/xtts_v2",
    )
    stt_provider: str = os.getenv("ANYDES_STT_PROVIDER", "windows")
    stt_culture: str = os.getenv("ANYDES_STT_CULTURE", "es-MX")
    stt_model: str = os.getenv("ANYDES_STT_MODEL", "gpt-4o-mini-transcribe")
    vosk_model_dir: Path = _env_path("ANYDES_VOSK_MODEL_DIR", "modelos/vosk/vosk-model-small-es-0.42")
    tts_provider: str = os.getenv("ANYDES_TTS_PROVIDER", "windows")
    tts_model: str = os.getenv("ANYDES_TTS_MODEL", "gpt-4o-mini-tts")
    tts_voice: str = os.getenv("ANYDES_TTS_VOICE", "nova")
    edge_tts_voice: str = os.getenv("ANYDES_EDGE_TTS_VOICE", "es-MX-DaliaNeural")
    edge_tts_rate: str = os.getenv("ANYDES_EDGE_TTS_RATE", "-15%")
    edge_tts_pitch: str = os.getenv("ANYDES_EDGE_TTS_PITCH", "+10Hz")
    piper_model_path: Path = _env_path("ANYDES_PIPER_MODEL", "modelos/piper/es_AR-daniela-high.onnx")
    piper_config_path: Path = _env_path(
        "ANYDES_PIPER_CONFIG",
        "modelos/piper/es_AR-daniela-high.onnx.json",
    )
    piper_length_scale: float = float(os.getenv("ANYDES_PIPER_LENGTH_SCALE", "1.00"))
    piper_sentence_silence: float = float(os.getenv("ANYDES_PIPER_SENTENCE_SILENCE", "0.18"))
    piper_pitch_factor: float = float(os.getenv("ANYDES_PIPER_PITCH_FACTOR", "1.16"))
    voice_seconds: int = int(os.getenv("ANYDES_VOICE_SECONDS", "6"))
    continuous_idle_seconds: float = float(os.getenv("ANYDES_CONTINUOUS_IDLE_SECONDS", "0.6"))
    tts_rate: int = int(os.getenv("ANYDES_TTS_RATE", "-2"))
    wake_word: str = os.getenv("ANYDES_WAKE_WORD", "anydes")
    voice_activation: str = os.getenv("ANYDES_VOICE_ACTIVATION", "button")
    proactive_enabled: bool = _env_bool("ANYDES_PROACTIVE_ENABLED", False)
    proactive_interval_seconds: int = int(os.getenv("ANYDES_PROACTIVE_INTERVAL_SECONDS", "1800"))
    proactive_minute_marks: str = os.getenv("ANYDES_PROACTIVE_MINUTE_MARKS", "0,30")
    speaker_verification_mode: str = os.getenv("ANYDES_SPEAKER_VERIFICATION", "passphrase")
    speaker_verification_phrase: str = os.getenv("ANYDES_SPEAKER_PHRASE", "soy DemoUser")
    speaker_match_threshold: float = float(os.getenv("ANYDES_SPEAKER_MATCH_THRESHOLD", "0.24"))
    typed_input_is_owner: bool = _env_bool("ANYDES_TYPED_INPUT_IS_OWNER", True)
    sensitive_action_verification_mode: str = os.getenv(
        "ANYDES_SENSITIVE_ACTION_VERIFICATION",
        "passphrase",
    )
    vision_enabled: bool = _env_bool("ANYDES_VISION_ENABLED", False)
    vision_provider: str = os.getenv("ANYDES_VISION_PROVIDER", "auto")
    vision_model: str = os.getenv("ANYDES_VISION_MODEL", "")
    vision_auto_pull: bool = _env_bool("ANYDES_VISION_AUTO_PULL", False)
    vision_reactivity_level: float = float(os.getenv("ANYDES_VISION_REACTIVITY_LEVEL", "0.7"))
    screen_watch_interval_seconds: int = int(os.getenv("ANYDES_SCREEN_INTERVAL_SECONDS", "5"))
    screen_live_mode: bool = _env_bool("ANYDES_SCREEN_LIVE_MODE", False)
    screen_live_interval_seconds: float = float(os.getenv("ANYDES_SCREEN_LIVE_INTERVAL_SECONDS", "1.5"))
    screen_fresh_on_analyze: bool = _env_bool("ANYDES_SCREEN_FRESH_ON_ANALYZE", True)
    screen_monitor: int = int(os.getenv("ANYDES_SCREEN_MONITOR", "0"))
    screen_max_width: int = int(os.getenv("ANYDES_SCREEN_MAX_WIDTH", "1280"))
    screen_max_height: int = int(os.getenv("ANYDES_SCREEN_MAX_HEIGHT", "720"))
    screen_jpeg_quality: int = int(os.getenv("ANYDES_SCREEN_JPEG_QUALITY", "85"))
    live_runtime_enabled: bool = _env_bool("ANYDES_LIVE_RUNTIME_ENABLED", True)
    live_runtime_simulated: bool = _env_bool("ANYDES_LIVE_RUNTIME_SIMULATED", False)
    live_context_mode: str = os.getenv("ANYDES_LIVE_CONTEXT_MODE", "shadow")
    developer_mode: bool = _env_bool("ANYDES_DEVELOPER_MODE", False)
    slow_interaction_threshold_ms: float = float(
        os.getenv("ANYDES_SLOW_INTERACTION_THRESHOLD_MS", "5000")
    )
    live_runtime_workers: int = int(os.getenv("ANYDES_LIVE_RUNTIME_WORKERS", "4"))
    conversation_retention_days: int = int(os.getenv("ANYDES_CONVERSATION_RETENTION_DAYS", "0"))
    conversation_retention_max_turns: int = int(os.getenv("ANYDES_CONVERSATION_RETENTION_MAX_TURNS", "0"))

    def resolved_live_context_mode(self) -> str:
        mode = self.live_context_mode.strip().lower()
        return mode if mode in {"legacy", "shadow", "live"} else "shadow"

    def resolved_reasoning_mode(self) -> str:
        mode = self.reasoning_mode.strip().lower()
        return mode if mode in {"legacy", "shadow", "native"} else "native"

    def resolved_unified_inference_mode(self) -> str:
        mode = self.unified_inference_mode.strip().lower()
        return mode if mode in {"off", "shadow"} else "off"

    def resolved_conversation_mode(self) -> str:
        mode = self.conversation_mode.strip().lower()
        return mode if mode in {"stable", "realtime"} else "stable"

    def resolved_screen_watch_interval_seconds(self) -> float:
        if self.screen_live_mode:
            return max(0.5, float(self.screen_live_interval_seconds))
        return max(1.0, float(self.screen_watch_interval_seconds))

    def resolved_vision_provider(self) -> str:
        explicit = self.vision_provider.strip().lower()
        if explicit and explicit != "auto":
            return explicit

        model = self.vision_model.strip().lower()
        groq_hints = (
            "llama-3.2",
            "llama-4",
            "meta-llama/",
            "gpt-4o",
            "vision-preview",
        )
        ollama_hints = (
            "moondream",
            "llava",
            "bakllava",
            "minicpm-v",
            "gemma3",
            "qwen2.5vl",
            "qwen3-vl",
            "llama3.2-vision",
        )
        if any(hint in model for hint in groq_hints):
            return "groq"
        if any(hint in model for hint in ollama_hints):
            return "ollama"

        provider = self.ai_provider.lower()
        if provider in {"groq", "openai", "ollama"}:
            return provider
        return "local"

    def resolved_vision_model(self) -> str:
        if self.vision_model.strip():
            return self.vision_model.strip()

        provider = self.resolved_vision_provider()
        if provider == "ollama":
            return "moondream"
        if provider == "openai":
            return "gpt-4o-mini"
        if provider == "groq":
            return "meta-llama/llama-4-scout-17b-16e-instruct"
        return ""

    @property
    def audit_log_path(self) -> Path:
        return self.data_dir / "audit.log"

    @property
    def audit_migration_backup_dir(self) -> Path:
        return self.data_dir / "audit_backups"

    @property
    def interaction_trace_path(self) -> Path:
        return self.data_dir / "developer" / "interaction_latency.jsonl"

    @property
    def slow_interaction_trace_path(self) -> Path:
        return self.data_dir / "developer" / "slow_interactions.jsonl"

    @property
    def realtime_trace_path(self) -> Path:
        return self.data_dir / "developer" / "realtime_conversation.jsonl"

    @property
    def realtime_journal_path(self) -> Path:
        return self.data_dir / "realtime" / "conversation_journal.v1.jsonl"

    @property
    def realtime_tasks_path(self) -> Path:
        return self.data_dir / "realtime" / "tasks.v1.json"

    @property
    def realtime_identity_profile_path(self) -> Path:
        return self.data_dir / "realtime" / "identity_profile.v1.json"

    @property
    def realtime_derived_insights_path(self) -> Path:
        return self.data_dir / "realtime" / "derived_insights.v1.json"

    @property
    def realtime_calendar_events_path(self) -> Path:
        return self.data_dir / "realtime" / "calendar_events.v1.json"

    @property
    def memory_path(self) -> Path:
        return self.permanent_memory_path

    @property
    def permanent_memory_path(self) -> Path:
        return self.data_dir / "memoria" / "permanente.json"

    @property
    def temporary_memory_path(self) -> Path:
        return self.data_dir / "memoria" / "temporal.json"

    @property
    def protected_memory_path(self) -> Path:
        return self.data_dir / "memoria" / "protegida.json"

    @property
    def conversation_log_path(self) -> Path:
        return self.data_dir / "memoria" / "conversaciones.jsonl"

    @property
    def classified_memory_path(self) -> Path:
        return self.data_dir / "memoria" / "clasificada.json"

    @property
    def conversation_archive_path(self) -> Path:
        return self.data_dir / "memoria" / "archivo_conversaciones.v2.jsonl"

    @property
    def conversation_cognitive_path(self) -> Path:
        return self.data_dir / "memoria" / "contexto_conversaciones.v2.jsonl"

    @property
    def conversation_categories_path(self) -> Path:
        return self.data_dir / "memoria" / "categorias_conversaciones.v2.json"

    @property
    def conversation_migration_marker_path(self) -> Path:
        return self.data_dir / "memoria" / "migracion_conversaciones.v2.json"

    @property
    def conversation_migration_backup_dir(self) -> Path:
        return self.data_dir / "memoria" / "respaldos_migracion"

    @property
    def relationship_state_path(self) -> Path:
        return self.data_dir / "relacion" / "estado.json"

    @property
    def self_state_path(self) -> Path:
        return self.data_dir / "self" / "estado.json"

    @property
    def curiosity_state_path(self) -> Path:
        return self.data_dir / "cognicion" / "curiosidad.v1.json"

    @property
    def speaker_profiles_path(self) -> Path:
        return self.data_dir / "voz" / "perfiles_voces.json"

    @property
    def voice_dir(self) -> Path:
        return self.data_dir / "voz"

    @property
    def piper_output_path(self) -> Path:
        return self.voice_dir / "respuesta_piper.wav"

    @property
    def avatar_state_path(self) -> Path:
        return self.data_dir / "avatar" / "estado.json"


config = Config()
