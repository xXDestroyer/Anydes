from __future__ import annotations

from hashlib import sha256

from anydes.cognition.contracts import InternalContext, Intention, SpeechIntention


class StartupReasoningModel:
    """Context-only reasoning model for Anydes' first presence in a session."""

    def __init__(self, context: InternalContext, user_name: str) -> None:
        self.context = context
        self.user_name = user_name

    def generate(self, prompt: str) -> str:
        _ = prompt
        text = self._select_greeting()
        emotion = self.context.emotion
        uncertainty = self.context.uncertainty
        curiosity = self._curiosity_signal()
        confidence = max(0.45, 1.0 - uncertainty.reasoning * 0.45 - uncertainty.memory * 0.2)
        speech = SpeechIntention(
            text=text,
            emotion=emotion.dominant_label,
            energy=emotion.energy * 0.72 + 0.12,
            warmth=emotion.affection * 0.58 + emotion.interest * 0.27 + 0.12,
            confidence=confidence,
            thinking=uncertainty.reasoning * 0.36,
            curiosity=emotion.interest * 0.56 + curiosity * 0.44,
            pause_before=0.12 + uncertainty.reasoning * 0.08,
            pause_after=0.16,
            speech_rate=0.92 - uncertainty.reasoning * 0.03,
            pitch_variation=0.13 + emotion.interest * 0.04,
            volume=0.86 + emotion.energy * 0.05,
            smile=emotion.affection * 0.28 + max(0.0, emotion.mood) * 0.12,
            metadata={"origin": "startup_reasoning"},
        )
        return Intention(
            id=self._intention_id(text),
            intent_type="session_greeting",
            speech=speech.text,
            speech_intention=speech,
            avatar={
                "expression": "soft_smile",
                "gesture": "idle",
                "gaze": "user",
            },
            confidence=confidence,
            reasoning_summary=(
                "Session greeting selected from time, recent continuity and current internal state."
            ),
        ).to_json()

    def _select_greeting(self) -> str:
        period = self._time_of_day()
        name = self.user_name
        candidates = {
            "dawn": (
                f"Buenos dias, {name}.",
                f"Hola, {name}. Que gusto verte tan temprano.",
                "Buenos dias. Como empieza tu dia?",
            ),
            "day": (
                f"Hola, {name}.",
                "Que bueno verte.",
                "Hola. En que estas trabajando hoy?",
                "Hola. Como va tu dia?",
            ),
            "sunset": (
                f"Hola, {name}. Como va tu tarde?",
                "Que bueno verte. Como estuvo tu dia?",
                "Hola. En que andas a esta hora?",
            ),
            "night": (
                f"Buenas noches, {name}.",
                f"Hola, {name}. Que bueno verte.",
                "Buenas noches. Como termino tu dia?",
                "Hola. Seguimos con algo tranquilo?",
            ),
        }.get(period, (f"Hola, {name}.", "Que bueno verte."))

        if self.context.memory.short:
            candidates = (
                *candidates,
                f"Hola, {name}. Retomamos lo que estabamos haciendo?",
                "Que bueno verte otra vez. Seguimos desde donde quedamos?",
            )
        continuity = self.context.memory.short[-1].content if self.context.memory.short else "new-session"
        material = (
            f"{self.context.timestamp}|{period}|{continuity}|"
            f"{self.context.emotion.dominant_label}|{self.context.emotion.energy:.2f}"
        )
        index = int(sha256(material.encode("utf-8")).hexdigest()[:8], 16) % len(candidates)
        return candidates[index]

    def _time_of_day(self) -> str:
        for observation in self.context.observations:
            if observation.source != "environment":
                continue
            value = observation.data.get("time_of_day")
            if isinstance(value, str) and value:
                return value
        return "day"

    def _curiosity_signal(self) -> float:
        active = [
            item.importance
            for item in self.context.curiosity
            if getattr(item, "is_active", True)
        ]
        if not active:
            return 0.0
        return sum(active) / len(active)

    def _intention_id(self, text: str) -> str:
        digest = sha256(f"{self.context.timestamp}|{text}".encode("utf-8")).hexdigest()[:12]
        return f"intention:session_greeting:{digest}"
