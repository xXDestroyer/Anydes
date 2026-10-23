from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from anydes.nucleo.config import Config, PROGRAM_DIR
from anydes.primitives import JsonValue, clamp01


@dataclass(frozen=True)
class VoiceIdentity:
    """Identidad vocal permanente de Anydes, independiente del motor TTS.

    La procedencia y el uso permitido de la referencia se declaran en el
    manifest canónico. El contrato conserva timbre, ritmo, calidez, pausas y
    naturalidad con independencia del motor que la reproduzca.
    """

    name: str = "Anydes"
    identity_id: str = "anydes-voice-v2"
    status: str = "unresolved"
    canonical_name: str = ""
    language: str = "es"
    locale: str = "es-MX"
    accent: str = "espanol latino neutro con cercania mexicana"
    timbre: str = "femenino juvenil, dulce, claro y suave"
    apparent_age: str = "aproximadamente 18 a 20 anos, juvenil sin sonar infantil"
    rhythm: str = "pausado, cercano, con respiracion natural"
    base_speed: float = 0.82
    base_pitch: float = 0.62
    warmth: float = 0.84
    expressiveness: float = 0.52
    breathiness: float = 0.22
    pause_style: str = "micro pausas antes de ideas reflexivas"
    intonation: str = "curiosa, tranquila, con sonrisa sutil"
    base_energy: float = 0.38
    voice_smile: float = 0.28
    naturalness: float = 0.9
    reference_audio_path: str = ""
    reference_usage: str = "acoustic_reference_only_non_cloning"
    pronunciation_lexicon: dict[str, str] = field(default_factory=dict)
    variation_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "speed": (0.74, 1.02),
            "pitch": (0.48, 0.72),
            "energy": (0.22, 0.68),
            "warmth": (0.68, 0.96),
            "expressiveness": (0.26, 0.7),
        }
    )
    notes: tuple[str, ...] = (
        "Toda referencia necesita autorizacion y procedencia registradas.",
        "Los cambios de timbre requieren una decision explicita de DemoUser.",
        "Los cambios expresivos deben ser sutiles y nunca exagerados.",
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_speed", max(0.6, min(1.25, float(self.base_speed))))
        object.__setattr__(self, "base_pitch", clamp01(self.base_pitch))
        object.__setattr__(self, "warmth", clamp01(self.warmth))
        object.__setattr__(self, "expressiveness", clamp01(self.expressiveness))
        object.__setattr__(self, "breathiness", clamp01(self.breathiness))
        object.__setattr__(self, "base_energy", clamp01(self.base_energy))
        object.__setattr__(self, "voice_smile", clamp01(self.voice_smile))
        object.__setattr__(self, "naturalness", clamp01(self.naturalness))
        object.__setattr__(self, "notes", tuple(self.notes))
        object.__setattr__(
            self,
            "pronunciation_lexicon",
            {
                str(key).strip(): str(value).strip()
                for key, value in self.pronunciation_lexicon.items()
                if str(key).strip() and str(value).strip()
            },
        )
        object.__setattr__(
            self,
            "variation_limits",
            {
                str(key): (float(value[0]), float(value[1]))
                for key, value in self.variation_limits.items()
            },
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "name": self.name,
            "identity_id": self.identity_id,
            "status": self.status,
            "canonical_name": self.canonical_name,
            "language": self.language,
            "locale": self.locale,
            "accent": self.accent,
            "timbre": self.timbre,
            "apparent_age": self.apparent_age,
            "rhythm": self.rhythm,
            "base_speed": self.base_speed,
            "base_pitch": self.base_pitch,
            "warmth": self.warmth,
            "expressiveness": self.expressiveness,
            "breathiness": self.breathiness,
            "pause_style": self.pause_style,
            "intonation": self.intonation,
            "base_energy": self.base_energy,
            "voice_smile": self.voice_smile,
            "naturalness": self.naturalness,
            "reference_audio_path": self.reference_audio_path,
            "reference_usage": self.reference_usage,
            "pronunciation_lexicon": dict(self.pronunciation_lexicon),
            "variation_limits": {
                key: list(value) for key, value in self.variation_limits.items()
            },
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class VoiceProfile:
    """Forma natural de hablar de Anydes, portable entre backends."""

    identity: VoiceIdentity = field(default_factory=VoiceIdentity)
    traits: tuple[str, ...] = (
        "joven",
        "tranquila",
        "dulce",
        "curiosa",
        "cercana",
        "pausada",
        "natural",
    )
    constraints: tuple[str, ...] = (
        "nunca exagerada",
        "nunca robotica",
        "nunca extremadamente energetica",
        "nunca sustituir silenciosamente la identidad vocal",
    )
    preferred_local_backends: tuple[str, ...] = (
        "qwen3-tts-faster",
        "qwen3-tts-base",
        "piper",
    )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "identity": self.identity.to_dict(),
            "traits": list(self.traits),
            "constraints": list(self.constraints),
            "preferred_local_backends": list(self.preferred_local_backends),
        }


def load_voice_identity(config: Config) -> VoiceIdentity:
    identity_path = PROGRAM_DIR / "configuracion" / "voice_identity_anydes.json"
    values: dict[str, object] = {}
    try:
        payload = json.loads(identity_path.read_text(encoding="utf-8"))
        base = payload.get("base") if isinstance(payload.get("base"), dict) else {}
        limits = payload.get("variation_limits")
        values = {
            "name": str(payload.get("display_name") or config.profile_name),
            "identity_id": str(payload.get("identity_id") or "anydes-voice-v2"),
            "status": str(
                payload.get("voice_identity_status")
                or payload.get("identity_selection_status")
                or "unresolved"
            ),
            "canonical_name": str(payload.get("canonical_name") or ""),
            "language": str(payload.get("language") or "es"),
            "locale": str(payload.get("locale") or "es-MX"),
            "accent": str(payload.get("accent") or VoiceIdentity.accent),
            "timbre": str(payload.get("timbre") or VoiceIdentity.timbre),
            "apparent_age": str(payload.get("apparent_age") or VoiceIdentity.apparent_age),
            "base_speed": float(base.get("speed", VoiceIdentity.base_speed)),
            "base_pitch": float(base.get("pitch", VoiceIdentity.base_pitch)),
            "base_energy": float(base.get("energy", VoiceIdentity.base_energy)),
            "warmth": float(base.get("warmth", VoiceIdentity.warmth)),
            "expressiveness": float(base.get("expressiveness", VoiceIdentity.expressiveness)),
            "breathiness": float(base.get("breathiness", VoiceIdentity.breathiness)),
        }
        pronunciations = payload.get("pronunciation_lexicon")
        if isinstance(pronunciations, dict):
            values["pronunciation_lexicon"] = {
                str(key): str(value) for key, value in pronunciations.items()
            }
        reference_payload = payload.get("reference")
        if isinstance(reference_payload, dict):
            values["reference_usage"] = str(
                reference_payload.get("usage") or VoiceIdentity.reference_usage
            )
        if isinstance(limits, dict):
            values["variation_limits"] = {
                str(key): (float(value[0]), float(value[1]))
                for key, value in limits.items()
                if isinstance(value, list) and len(value) == 2
            }
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        values = {"name": config.profile_name}
    reference = config.voice_dir / "referencia_anydes.mp3"
    values["reference_audio_path"] = str(reference) if reference.exists() else ""
    return VoiceIdentity(**values)


def voice_reference_drop_folder(config: Config) -> Path:
    path = config.voice_dir / "referencias"
    path.mkdir(parents=True, exist_ok=True)
    return path
