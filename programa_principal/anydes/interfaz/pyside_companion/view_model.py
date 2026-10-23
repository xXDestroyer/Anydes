from __future__ import annotations

from dataclasses import dataclass, field

from anydes.cognition.contracts import InternalContext, Intention


@dataclass(frozen=True)
class InnerWorldCard:
    title: str
    value: str
    description: str


@dataclass(frozen=True)
class CompanionState:
    status: str = "😊 Tranquila"
    expression: str = "neutral"
    gesture: str = "idle"
    gaze: str = "user"
    posture: str = "relaxed"
    attention: str = "user"
    expression_intensity: float = 0.5
    thought: str = "Estoy aqui, tranquila."
    speech: dict[str, object] = field(default_factory=dict)
    lip_sync: dict[str, object] = field(default_factory=dict)


def public_status(raw: str) -> str:
    raw = raw.lower().strip()
    if "pens" in raw or raw == "thinking":
        return "💭 Pensando..."
    if "observ" in raw or raw in {"watching", "vision"}:
        return "👀 Observando..."
    if "escuch" in raw or raw == "listening":
        return "🎤 Escuchando..."
    return "😊 Tranquila"


def camera_mode_for_state(state: CompanionState) -> str:
    """Select framing from observable conversation activity only."""

    status = state.status.casefold()
    gesture = state.gesture.casefold()
    speech_text = str(state.speech.get("text") or "").strip()
    lip_sync_active = bool(state.lip_sync.get("enabled"))
    active_status = any(token in status for token in ("pens", "escuch", "habl"))
    active_gesture = gesture in {"talk", "speaking", "thinking", "listening"}
    return (
        "conversation"
        if speech_text or lip_sync_active or active_status or active_gesture
        else "environment"
    )


def state_from_intention(intention: Intention | None) -> CompanionState:
    if intention is None:
        return CompanionState()
    avatar = intention.avatar
    expression = str(avatar.get("expression", "speaking" if intention.speech.strip() else "neutral"))
    gesture = str(avatar.get("gesture", "talk" if intention.speech.strip() else "idle"))
    gaze = str(avatar.get("gaze", "user"))
    posture = str(avatar.get("posture", "relaxed"))
    attention = str(avatar.get("attention", "user"))
    expression_intensity = _clamp(avatar.get("expression_intensity", 0.5))
    status = "thinking" if expression == "thinking" else ("speaking" if intention.speech.strip() else "idle")
    return CompanionState(
        status=public_status(status),
        expression=expression,
        gesture=gesture,
        gaze=gaze,
        posture=posture,
        attention=attention,
        expression_intensity=expression_intensity,
        thought=str(avatar.get("message", "") or _summarize_speech(intention.speech)),
        speech=intention.speech_intention.to_dict(),
        lip_sync=(
            dict(avatar["lip_sync"])
            if isinstance(avatar.get("lip_sync"), dict)
            else _lip_sync_hint(intention)
        ),
    )


def inner_world_cards(context: InternalContext | None, intention: Intention | None) -> tuple[InnerWorldCard, ...]:
    if context is None:
        return (
            InnerWorldCard("🧠 Curiosidad", "Suave", "Estoy conociendo el ritmo de esta sesion."),
            InnerWorldCard("⚡ Energia", "Tranquila", "Mi presencia se mantiene calmada y atenta."),
            InnerWorldCard("💙 Estado emocional", "Serena", "Estoy lista para acompañarte sin ruido."),
            InnerWorldCard("🎯 Objetivo actual", "Estar contigo", "Mantener una experiencia cercana y clara."),
        )

    expression = "neutral"
    if intention is not None:
        expression = str(intention.avatar.get("expression", "neutral"))

    return (
        InnerWorldCard("🧠 Curiosidad", _level(_curiosity(context)), _curiosity_text(context)),
        InnerWorldCard("⚡ Energia", _level(context.emotion.energy), _energy_text(context.emotion.energy)),
        InnerWorldCard("💙 Estado emocional", _emotion(context.emotion.dominant_label), context.emotion.summary or "Me siento estable y cercana."),
        InnerWorldCard("🎯 Objetivo actual", "Presente", _goal_text(context)),
        InnerWorldCard("👀 Atencion", _attention(context), _attention_text(context)),
        InnerWorldCard("💭 Pensamiento resumido", "Suave", state_from_intention(intention).thought),
        InnerWorldCard("📚 Recuerdos recientes", _memory_value(context), _memory_text(context)),
        InnerWorldCard("🧍 Expresion", _expression(expression), "Mi cuerpo virtual acompaña mi estado actual."),
    )


def _summarize_speech(text: str) -> str:
    text = " ".join(text.split())
    if not text:
        return "Estoy escuchando el ambiente con calma."
    return text[:118] + ("..." if len(text) > 118 else "")


def _level(value: float) -> str:
    if value >= 0.78:
        return "Muy alta"
    if value >= 0.58:
        return "Alta"
    if value >= 0.36:
        return "Suave"
    return "Baja"


def _curiosity(context: InternalContext) -> float:
    if not context.curiosity:
        return 0.34
    return max(item.importance for item in context.curiosity)


def _curiosity_text(context: InternalContext) -> str:
    active = [item for item in context.curiosity if not getattr(item, "satisfied", False)]
    if not active:
        return "No necesito interrumpirte con preguntas ahora."
    curiosity = max(active, key=lambda item: item.importance)
    topic = getattr(curiosity, "topic", "") or getattr(curiosity, "missing_information", "")
    return f"Me intriga {topic}. {curiosity.reason}"


def _energy_text(value: float) -> str:
    if value >= 0.7:
        return "Estoy despierta y presente."
    if value >= 0.4:
        return "Tengo una energia tranquila."
    return "Estoy en un modo lento y cuidadoso."


def _emotion(label: str) -> str:
    labels = {
        "neutral": "Serena",
        "happy": "Contenta",
        "feliz": "Contenta",
        "curiosa": "Curiosa",
        "thinking": "Pensativa",
        "tensa": "Atenta",
    }
    return labels.get(label.lower(), label.capitalize() or "Serena")


def _goal_text(context: InternalContext) -> str:
    goals = tuple(context.goals.temporary) + tuple(context.goals.permanent)
    if not goals:
        return "Acompañarte de forma util y calida."
    return max(goals, key=lambda item: item.priority).description


def _attention(context: InternalContext) -> str:
    if any(item.source == "vision" and item.salience > 0 for item in context.observations):
        return "En tu pantalla"
    if any(item.source in {"user_text", "voice"} for item in context.observations):
        return "En ti"
    return "En calma"


def _attention_text(context: InternalContext) -> str:
    if _attention(context) == "En tu pantalla":
        return "Puedo observar señales visuales sin convertirlas en ruido."
    return "Estoy siguiendo la conversacion con suavidad."


def _memory_value(context: InternalContext) -> str:
    count = len(context.memory.short) + len(context.memory.long)
    if count >= 8:
        return "Muy presentes"
    if count >= 3:
        return "Presentes"
    if count:
        return "Pocos"
    return "En blanco"


def _memory_text(context: InternalContext) -> str:
    memories = tuple(context.memory.short) + tuple(context.memory.long)
    if not memories:
        return "No hay recuerdos recientes para mostrar."
    return memories[0].content.splitlines()[0][:120]


def _expression(expression: str) -> str:
    labels = {
        "neutral": "Relajada",
        "speaking": "Hablando",
        "happy": "Sonriendo",
        "thinking": "Pensativa",
        "listening": "Escuchando",
        "shy": "Dulce",
        "surprised": "Sorprendida",
    }
    return labels.get(expression.lower(), expression.capitalize() or "Relajada")


def _clamp(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


def _lip_sync_hint(intention: Intention) -> dict[str, object]:
    text = intention.speech_intention.text.strip()
    if not text:
        return {"enabled": False, "duration_hint_ms": 0}
    rate = max(0.65, intention.speech_intention.speech_rate)
    return {
        "enabled": True,
        "duration_hint_ms": min(12000, max(700, int(len(text) * 48 / rate))),
    }
