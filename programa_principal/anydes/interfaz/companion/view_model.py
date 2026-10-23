from __future__ import annotations

from dataclasses import dataclass

from anydes.cognition.contracts import InternalContext, Intention


@dataclass(frozen=True)
class InnerWorldCardData:
    """Datos naturales para una tarjeta del Mundo Interior."""

    title: str
    value: str
    description: str
    accent: str = "soft"


@dataclass(frozen=True)
class CompanionViewState:
    """Estado minimo que necesita la interfaz para dibujar a Anydes."""

    status: str = "✨ Lista"
    expression: str = "neutral"
    gesture: str = "idle"
    gaze: str = "user"
    current_thought: str = "Estoy aqui contigo."


STATUS_LABELS = {
    "thinking": "💭 Pensando...",
    "watching": "👀 Observando la pantalla...",
    "listening": "🎤 Escuchando...",
    "remembering": "🧠 Recordando...",
    "learning": "✨ Aprendiendo...",
    "speaking": "💬 Hablando...",
    "idle": "✨ Tranquila",
}


def status_label(status: str) -> str:
    """Convierte estados internos en etiquetas breves y no tecnicas."""

    return STATUS_LABELS.get(status, STATUS_LABELS["idle"])


def view_state_from_intention(intention: Intention | None) -> CompanionViewState:
    """Deriva expresion, gesto y pensamiento visible desde una Intention."""

    if intention is None:
        return CompanionViewState()
    avatar = intention.avatar
    expression = str(avatar.get("expression", "speaking" if intention.speech.strip() else "neutral"))
    gesture = str(avatar.get("gesture", "talk" if intention.speech.strip() else "idle"))
    gaze = str(avatar.get("gaze", "user"))
    thought = str(avatar.get("message", "") or _soft_thought_from_intention(intention))
    status = "speaking" if intention.speech.strip() else "idle"
    return CompanionViewState(
        status=status_label(status),
        expression=expression,
        gesture=gesture,
        gaze=gaze,
        current_thought=thought,
    )


def build_inner_world_cards(context: InternalContext | None, intention: Intention | None) -> tuple[InnerWorldCardData, ...]:
    """Crea tarjetas descriptivas del Mundo Interior sin mostrar porcentajes."""

    if context is None:
        return (
            InnerWorldCardData("🧠 Curiosidad", "Despierta", "Aun estoy reuniendo contexto de esta sesion."),
            InnerWorldCardData("💙 Estado emocional", "Serena", "Mantengo un tono calido y tranquilo."),
            InnerWorldCardData("🎯 Objetivo principal", "Acompañarte", "Estoy lista para ayudarte sin invadir tu espacio."),
        )

    emotion = context.emotion
    uncertainty = context.uncertainty
    memory = context.memory
    goal = _main_goal(context)
    attention = _attention_from_context(context)
    thought = _soft_thought_from_intention(intention)
    expression = "neutral"
    if intention is not None:
        expression = str(intention.avatar.get("expression", expression))

    return (
        InnerWorldCardData("🧠 Curiosidad", _level_label(_curiosity_level(context)), _curiosity_description(context)),
        InnerWorldCardData("⚡ Energia", _level_label(emotion.energy), _energy_description(emotion.energy)),
        InnerWorldCardData("💙 Estado emocional", _emotion_label(emotion.dominant_label), emotion.summary or "Me siento estable y cercana."),
        InnerWorldCardData("🎯 Objetivo principal", "Presente", goal),
        InnerWorldCardData("👀 Atencion", attention, _attention_description(context)),
        InnerWorldCardData("💭 Pensamiento actual", "Suave", thought),
        InnerWorldCardData("📚 Recuerdos recientes", _memory_label(memory.short, memory.long), _memory_description(memory.short, memory.long)),
        InnerWorldCardData("🧍 Expresion actual", _expression_label(expression), "Mi cuerpo virtual acompaña lo que estoy diciendo."),
        InnerWorldCardData("🌫️ Incertidumbre", _level_label(1.0 - uncertainty.reasoning), _uncertainty_description(uncertainty.reasoning)),
    )


def _soft_thought_from_intention(intention: Intention | None) -> str:
    if intention is None:
        return "Estoy esperando con calma."
    if intention.speech.strip():
        text = " ".join(intention.speech.strip().split())
        return text[:120] + ("..." if len(text) > 120 else "")
    if intention.actions:
        return "Estoy preparando algo util para ti."
    return "Estoy ordenando lo que percibo."


def _level_label(value: float) -> str:
    if value >= 0.78:
        return "Muy alta"
    if value >= 0.58:
        return "Alta"
    if value >= 0.38:
        return "Suave"
    if value >= 0.18:
        return "Baja"
    return "Muy baja"


def _curiosity_level(context: InternalContext) -> float:
    prompts = context.curiosity
    if not prompts:
        return 0.32
    return max(prompt.importance for prompt in prompts)


def _curiosity_description(context: InternalContext) -> str:
    active = [item for item in context.curiosity if not getattr(item, "satisfied", False)]
    if not active:
        return "No necesito preguntarte nada urgente ahora."
    curiosity = max(active, key=lambda item: item.importance)
    topic = getattr(curiosity, "topic", "") or getattr(curiosity, "missing_information", "")
    return f"Me intriga {topic}. {curiosity.reason}"


def _energy_description(value: float) -> str:
    if value >= 0.7:
        return "Me siento despierta y atenta."
    if value >= 0.4:
        return "Tengo una energia tranquila."
    return "Estoy en un ritmo bajo y cuidadoso."


def _emotion_label(label: str) -> str:
    labels = {
        "neutral": "Serena",
        "curiosa": "Curiosa",
        "alegre": "Dulce",
        "feliz": "Contenta",
        "tensa": "Atenta",
        "triste": "Suave",
    }
    return labels.get(label.lower(), label.capitalize() or "Serena")


def _main_goal(context: InternalContext) -> str:
    goals = tuple(context.goals.temporary) + tuple(context.goals.permanent)
    if not goals:
        return "Acompañarte de forma util y cercana."
    goal = max(goals, key=lambda item: item.priority)
    return goal.description


def _attention_from_context(context: InternalContext) -> str:
    if any(observation.source == "vision" and observation.salience > 0 for observation in context.observations):
        return "En tu pantalla"
    if any(observation.source in {"user_text", "voice"} for observation in context.observations):
        return "En ti"
    return "En calma"


def _attention_description(context: InternalContext) -> str:
    if _attention_from_context(context) == "En tu pantalla":
        return "Tengo señales visuales disponibles, pero solo respondo si aportan algo."
    return "Estoy siguiendo la conversacion sin saturarte."


def _memory_label(short: tuple, long: tuple) -> str:
    total = len(short) + len(long)
    if total >= 8:
        return "Muy presentes"
    if total >= 3:
        return "Presentes"
    if total:
        return "Pocos"
    return "En blanco"


def _memory_description(short: tuple, long: tuple) -> str:
    if short:
        return str(short[0].content).splitlines()[0][:120]
    if long:
        return str(long[0].content).splitlines()[0][:120]
    return "Todavia no hay recuerdos recientes para esta vista."


def _expression_label(expression: str) -> str:
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


def _uncertainty_description(value: float) -> str:
    if value >= 0.65:
        return "Estoy cuidando mis conclusiones porque falta contexto."
    if value >= 0.35:
        return "Tengo una idea razonable, aunque puedo ajustar si me das mas contexto."
    return "El contexto se siente bastante claro."
