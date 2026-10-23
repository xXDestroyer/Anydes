from __future__ import annotations

import json
import math
import struct
import wave
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class VoiceFeatures:
    pitch_hz: float
    rms: float
    zero_crossing_rate: float
    duration_seconds: float


@dataclass(frozen=True)
class SpeakerMatch:
    name: str | None
    matched: bool
    confidence: float
    features: VoiceFeatures
    distance: float | None = None

    def label(self) -> str:
        if self.matched and self.name:
            return f"{self.name} ({self.confidence:.0%})"
        return f"voz desconocida ({self.confidence:.0%})"

    def prompt_context(self) -> str:
        if self.matched and self.name:
            return (
                f"Interlocutor detectado por voz: {self.name}. "
                f"Confianza local aproximada: {self.confidence:.0%}. "
                "No reveles memoria protegida solo por esta senal."
            )
        return (
            "Interlocutor por voz: desconocido o confianza insuficiente. "
            "Usa solo memoria no sensible y pide verificacion si hace falta."
        )


class SpeakerRegistry:
    def __init__(self, path: Path, threshold: float = 0.24) -> None:
        self.path = path
        self.threshold = threshold

    def enroll(self, name: str, audio_path: Path) -> SpeakerMatch:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Indica un nombre para registrar la voz.")

        features = analyze_wav(audio_path)
        data = self._load()
        speakers = data.setdefault("speakers", {})
        current = speakers.get(clean_name)
        if current:
            count = int(current.get("samples", 1))
            old = current["features"]
            averaged = {
                key: ((float(old[key]) * count) + float(getattr(features, key))) / (count + 1)
                for key in asdict(features)
            }
            current["features"] = averaged
            current["samples"] = count + 1
            current["updated_at"] = _now()
        else:
            speakers[clean_name] = {
                "features": asdict(features),
                "samples": 1,
                "created_at": _now(),
                "updated_at": _now(),
            }
        self._save(data)
        return SpeakerMatch(clean_name, True, 1.0, features, 0.0)

    def identify(self, audio_path: Path) -> SpeakerMatch:
        features = analyze_wav(audio_path)
        data = self._load()
        speakers = data.get("speakers", {})
        best_name: str | None = None
        best_distance: float | None = None

        for name, payload in speakers.items():
            distance = _feature_distance(features, _features_from_dict(payload["features"]))
            if best_distance is None or distance < best_distance:
                best_name = name
                best_distance = distance

        if best_name is None or best_distance is None:
            return SpeakerMatch(None, False, 0.0, features, None)

        confidence = max(0.0, min(1.0, 1.0 - (best_distance / (self.threshold * 2.0))))
        matched = best_distance <= self.threshold
        return SpeakerMatch(best_name if matched else None, matched, confidence, features, best_distance)

    def list_profiles(self) -> str:
        speakers = self._load().get("speakers", {})
        if not speakers:
            return "No hay voces registradas todavia."
        rows = []
        for name, payload in sorted(speakers.items()):
            features = _features_from_dict(payload["features"])
            rows.append(
                f"{name}: {payload.get('samples', 1)} muestra(s), "
                f"pitch {features.pitch_hz:.1f} Hz, energia {features.rms:.3f}"
            )
        return "\n".join(rows)

    def _load(self) -> dict:
        if not self.path.exists():
            return {"version": 1, "speakers": {}}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"version": 1, "speakers": {}}

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")


def analyze_wav(audio_path: Path) -> VoiceFeatures:
    sample_rate, samples = _read_mono_samples(audio_path)
    if not samples:
        raise ValueError("El audio no contiene muestras utiles.")

    max_samples = min(len(samples), int(sample_rate * 1.5))
    window = samples[:max_samples]
    rms = math.sqrt(sum(sample * sample for sample in window) / len(window))
    zero_crossings = sum(
        1
        for left, right in zip(window, window[1:])
        if (left < 0.0 <= right) or (left >= 0.0 > right)
    )
    zcr = zero_crossings / max(1, len(window) - 1)
    pitch = _estimate_pitch(window, sample_rate)
    return VoiceFeatures(
        pitch_hz=pitch,
        rms=rms,
        zero_crossing_rate=zcr,
        duration_seconds=len(samples) / sample_rate,
    )


def _read_mono_samples(audio_path: Path) -> tuple[int, list[float]]:
    with wave.open(str(audio_path), "rb") as audio:
        sample_rate = audio.getframerate()
        channels = audio.getnchannels()
        sample_width = audio.getsampwidth()
        frames = audio.readframes(audio.getnframes())

    if sample_width != 2:
        raise ValueError("Por ahora el reconocimiento de voz espera WAV PCM de 16 bits.")

    raw = struct.unpack("<" + "h" * (len(frames) // 2), frames)
    if channels <= 1:
        return sample_rate, [sample / 32768.0 for sample in raw]

    mono = []
    for index in range(0, len(raw), channels):
        mono.append(sum(raw[index : index + channels]) / (32768.0 * channels))
    return sample_rate, mono


def _estimate_pitch(samples: list[float], sample_rate: int) -> float:
    if len(samples) < sample_rate // 10:
        return 0.0

    frame_size = min(len(samples), int(sample_rate * 0.8))
    start = max(0, (len(samples) - frame_size) // 2)
    frame = samples[start : start + frame_size]
    frame_rms = math.sqrt(sum(sample * sample for sample in frame) / len(frame))
    if frame_rms < 0.01:
        return 0.0

    min_lag = max(1, int(sample_rate / 350))
    max_lag = min(len(frame) - 1, int(sample_rate / 70))
    best_lag = min_lag
    best_corr = -1.0

    for lag in range(min_lag, max_lag + 1):
        corr = 0.0
        for index in range(0, len(frame) - lag, 2):
            corr += frame[index] * frame[index + lag]
        if corr > best_corr:
            best_corr = corr
            best_lag = lag

    return sample_rate / best_lag if best_lag else 0.0


def _feature_distance(left: VoiceFeatures, right: VoiceFeatures) -> float:
    pitch_distance = abs(left.pitch_hz - right.pitch_hz) / max(1.0, right.pitch_hz)
    rms_distance = abs(left.rms - right.rms) / max(0.02, right.rms)
    zcr_distance = abs(left.zero_crossing_rate - right.zero_crossing_rate) / max(
        0.01,
        right.zero_crossing_rate,
    )
    return (pitch_distance * 0.55) + (rms_distance * 0.20) + (zcr_distance * 0.25)


def _features_from_dict(data: dict) -> VoiceFeatures:
    return VoiceFeatures(
        pitch_hz=float(data.get("pitch_hz", 0.0)),
        rms=float(data.get("rms", 0.0)),
        zero_crossing_rate=float(data.get("zero_crossing_rate", 0.0)),
        duration_seconds=float(data.get("duration_seconds", 0.0)),
    )


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
