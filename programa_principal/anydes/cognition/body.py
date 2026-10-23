from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from anydes.avatar.state import write_avatar_payload
from anydes.avatar.vrm import inspect_vrm, preferred_vrm_model
from anydes.cognition.contracts import BodyStateSnapshot, BodyUpdateResult, Intention, JsonValue
from anydes.nucleo.config import Config
from anydes.voz.identity import VoiceProfile, load_voice_identity
from anydes.voz.interpreter import VoiceInterpreter


class EventPublisher(Protocol):
    def publish(self, event: Any) -> None: ...


class _NullEventPublisher:
    def publish(self, event: Any) -> None:
        return None


@dataclass(frozen=True)
class BodyStateProvider:
    """Reads the current body state without deciding or mutating anything."""

    avatar_state_path: Path
    model_3d_root: Path

    @classmethod
    def from_config(cls, config: Config) -> "BodyStateProvider":
        return cls(
            avatar_state_path=config.avatar_state_path,
            model_3d_root=config.avatar_model_dir,
        )

    def snapshot(self) -> BodyStateSnapshot:
        if not self.avatar_state_path.exists():
            return BodyStateSnapshot(
                model_3d=_model_3d_payload(self.model_3d_root),
                warnings=("No body state has been published yet.",),
            )
        try:
            data = json.loads(self.avatar_state_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Body state root must be an object.")
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            return BodyStateSnapshot(
                model_3d=_model_3d_payload(self.model_3d_root),
                warnings=("Published body state is unavailable or invalid.",),
            )
        state = str(data.get("state") or "neutral")
        return BodyStateSnapshot(
            available=True,
            state=state,
            expression=str(data.get("expression") or "neutral"),
            gesture=str(data.get("gesture") or "idle"),
            gaze=str(data.get("gaze") or "user"),
            posture=str(data.get("posture") or "relaxed"),
            attention=str(data.get("attention") or "user"),
            expression_intensity=float(data.get("expression_intensity", 0.5)),
            speaking=state == "speaking",
            idle=bool(data.get("idle", state == "neutral")),
            lip_sync=data.get("lip_sync") if isinstance(data.get("lip_sync"), dict) else {},
            model_3d=data.get("model_3d") if isinstance(data.get("model_3d"), dict) else _model_3d_payload(self.model_3d_root),
            updated_at=str(data.get("updated_at") or ""),
        )


@dataclass(frozen=True)
class BodyController:
    """Applies an already reasoned Intention to Anydes' virtual body.

    The controller receives only Intention. It does not inspect context, user text,
    memory, goals, emotions or tools, and it never decides actions.
    """

    avatar_state_path: Path
    model_3d_root: Path
    voice_interpreter: VoiceInterpreter
    events: EventPublisher

    @classmethod
    def from_config(
        cls,
        config: Config,
        *,
        events: EventPublisher | None = None,
    ) -> "BodyController":
        return cls(
            avatar_state_path=config.avatar_state_path,
            model_3d_root=config.avatar_model_dir,
            voice_interpreter=VoiceInterpreter(
                VoiceProfile(identity=load_voice_identity(config))
            ),
            events=events or _NullEventPublisher(),
        )

    def apply(self, intention: Intention) -> BodyUpdateResult:
        try:
            payload = self.payload_from_intention(intention)
            write_avatar_payload(self.avatar_state_path, payload)
            from anydes.cognition.live_runtime.events import BODY_CHANGED, DomainEvent

            self.events.publish(
                DomainEvent(
                    event_type=BODY_CHANGED,
                    source="BodyController",
                    aggregate_id="body",
                    payload={"intention_id": intention.id},
                )
            )
            return BodyUpdateResult(
                intention_id=intention.id,
                status="updated",
                state=str(payload["state"]),
                expression=str(payload["expression"]),
                gesture=str(payload["gesture"]),
                gaze=str(payload["gaze"]),
                message=str(payload["message"]),
                output_path=str(self.avatar_state_path),
            )
        except Exception as exc:
            return BodyUpdateResult(
                intention_id=intention.id,
                status="failed",
                error=str(exc),
                output_path=str(self.avatar_state_path),
            )

    def payload_from_intention(self, intention: Intention) -> dict[str, JsonValue]:
        avatar = intention.avatar
        speech = intention.speech_intention
        voice_plan = self.voice_interpreter.interpret(speech)
        expression = _safe_avatar_value(avatar, "expression", "neutral")
        gesture = _safe_avatar_value(avatar, "gesture", "idle")
        gaze = _safe_avatar_value(avatar, "gaze", "user")
        posture = _safe_avatar_value(avatar, "posture", "relaxed")
        attention = _safe_avatar_value(avatar, "attention", "user")
        expression_intensity = _safe_avatar_number(avatar, "expression_intensity", 0.5)
        state = _state_for(expression=expression, gesture=gesture, has_speech=bool(intention.speech.strip()))
        message = _safe_avatar_value(avatar, "message", intention.speech[:160])
        lip_sync = avatar.get("lip_sync")
        if not isinstance(lip_sync, dict):
            lip_sync = _lip_sync_from_speech(
                intention.speech,
                speech_rate=voice_plan.speech_rate,
            )

        return {
            "state": state,
            "message": message,
            "expression": expression,
            "gesture": gesture,
            "gaze": gaze,
            "posture": posture,
            "attention": attention,
            "expression_intensity": expression_intensity,
            "idle": not bool(intention.speech.strip()),
            "lip_sync": lip_sync,
            "speech": speech.to_dict(),
            "speaking_motion": voice_plan.synchronization.to_dict(),
            "voice_performance": voice_plan.to_dict(),
            "intention_id": intention.id,
            "model_3d": _model_3d_payload(self.model_3d_root),
        }

    def _preferred_vrm(self) -> Path | None:
        return preferred_vrm_model(self.model_3d_root)


class BodyProvider(BodyController):
    """Deprecated output alias retained temporarily for import compatibility."""


def _safe_avatar_value(avatar: dict[str, JsonValue], key: str, default: str) -> str:
    value = avatar.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _model_3d_payload(model_3d_root: Path) -> dict[str, JsonValue]:
    preferred = preferred_vrm_model(model_3d_root)
    if preferred is None:
        return {
            "available": False,
            "asset_collection": model_3d_root.name if model_3d_root.exists() else "",
            "preferred_vrm": "",
        }
    return {
        **inspect_vrm(preferred).public_payload(),
        "asset_collection": model_3d_root.name,
    }


def _safe_avatar_number(
    avatar: dict[str, JsonValue],
    key: str,
    default: float,
) -> float:
    value = avatar.get(key)
    if isinstance(value, bool):
        return default
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _state_for(*, expression: str, gesture: str, has_speech: bool) -> str:
    if gesture in {"listen", "listening"}:
        return "listening"
    if gesture in {"think", "thinking"} or expression == "thinking":
        return "thinking"
    if has_speech:
        return "speaking"
    if expression in {"happy", "smile", "excited"}:
        return "happy"
    return "neutral"


def _lip_sync_from_speech(speech: str, *, speech_rate: float = 0.9) -> dict[str, JsonValue]:
    text = speech.strip()
    if not text:
        return {"enabled": False, "visemes": []}
    vowels = {
        "A": sum(text.lower().count(char) for char in "aá"),
        "E": sum(text.lower().count(char) for char in "eé"),
        "I": sum(text.lower().count(char) for char in "ií"),
        "O": sum(text.lower().count(char) for char in "oó"),
        "U": sum(text.lower().count(char) for char in "uú"),
    }
    total = max(1, sum(vowels.values()))
    visemes = [
        {"name": name, "weight": round(count / total, 3)}
        for name, count in vowels.items()
        if count > 0
    ]
    return {
        "enabled": True,
        "duration_hint_ms": min(8000, max(600, int(len(text) * 45 / max(0.65, speech_rate)))),
        "visemes": visemes,
    }
