from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Optional

from anydes.nucleo.config import Config

if TYPE_CHECKING:
    from anydes.vision.memoria_visual import MemoriaVisual


class BehaviorOrchestrator:
    """Orquesta el comportamiento de Anydes: visión, proactividad y conversación."""

    def __init__(
        self,
        config: Config,
        user_name: str = "DemoUser",
        profile_name: str = "Anydes",
        memory: Optional["MemoriaVisual"] = None,
    ) -> None:
        self.config = config
        self.user_name = user_name
        self.profile_name = profile_name
        if memory is None:
            from anydes.vision.memoria_visual import MemoriaVisual

            memory = MemoriaVisual(max_registros=150)
        self.memory = memory
        self.last_user_message = ""
        self.last_assistant_response = ""
        self.last_activity = time.monotonic()
        self.focus_history: dict[str, float] = {}
        self.profile_manager: Any | None = None

        self.attention_weights = {
            "juego": 0.95,
            "media": 0.9,
            "problema": 0.88,
            "general": 0.65,
            "navegacion": 0.35,
            "codigo": 0.25,
        }
        self.ignored_categories = {"codigo", "navegacion"}

    def notify_user_activity(self) -> None:
        self.last_activity = time.monotonic()

    def on_user_message(self, text: str) -> None:
        self.last_user_message = text
        self.notify_user_activity()
        try:
            self.infer_preferences_from_text(text)
        except Exception:
            # inference is best-effort
            pass

    def on_assistant_response(self, text: str) -> None:
        self.last_assistant_response = text

    def register_vision_observation(
        self,
        descripcion: str,
        categoria: str,
        sentimiento: str = "neutra",
        similitud_anterior: float = 0.0,
    ) -> None:
        self.memory.agregar(
            descripcion=descripcion,
            categoria=categoria,
            sentimiento=sentimiento,
            similitud_anterior=similitud_anterior,
        )
        self.focus_history[categoria] = time.time()

    def get_emotional_context(self) -> str:
        recientes = list(self.memory.registros)[-12:]
        if not recientes:
            return ""

        positivos = sum(1 for r in recientes if r.sentimiento == "positivo")
        negativos = sum(1 for r in recientes if r.sentimiento == "negativo")
        neutros = sum(1 for r in recientes if r.sentimiento == "neutra")

        if positivos > negativos * 1.2 and positivos > neutros:
            return (
                "El ambiente visual se siente positivo y animado. "
                "Responde con entusiasmo, ánimo y curiosidad amistosa."
            )
        if negativos > positivos * 1.2 and negativos > neutros:
            return (
                "El ambiente visual se siente tenso o frustrado. "
                "Responde con calma, empatía y apoyo práctico."
            )
        return (
            "El ambiente visual es mayormente neutro. "
            "Responde con curiosidad humana, manteniendo variedad y simpatía."
        )

    def get_attention_directive(self) -> str:
        return (
            "Atención selectiva: prioriza cambios inesperados, emociones visibles y situaciones nuevas. "
            "Ignora lo repetido o monótono a menos que haya algo emocionalmente relevante. "
            "Si detectas que el usuario está estancado o frustrado, ofrece apoyo breve y amable."
        )

    def enrich_vision_prompt(
        self,
        prompt: str,
        cambio_brusco: bool = False,
    ) -> str:
        resultado = [prompt]
        resultado.append(self.get_attention_directive())
        contexto_emocional = self.get_emotional_context()
        if contexto_emocional:
            resultado.append(f"Contexto emocional reciente: {contexto_emocional}")
        if cambio_brusco:
            resultado.append(
                "El cambio fue brusco o inesperado. Reacciona como si realmente te sorprendieras."
            )
        return "\n\n".join(resultado)

    def should_comment(
        self,
        category: str,
        similarity: float,
        sentiment: str,
    ) -> bool:
        if category in self.ignored_categories and similarity >= 0.45 and sentiment == "neutra":
            return False

        if category == "general" and similarity >= 0.65 and sentiment == "neutra":
            return False

        if sentiment in {"positivo", "negativo"}:
            return True

        importance = self.attention_weights.get(category, 0.5)
        if importance < 0.4 and similarity > 0.45:
            return False

        return True

    def get_proactive_hint(self) -> str:
        ultima = self.memory.ultima_descripcion()
        if ultima:
            return f"En pantalla, parece que {ultima[0].lower() + ultima[1:]}"
        contexto = self.memory.contexto_reciente(segundos=120.0)
        if contexto:
            return f"Resumen visual reciente: {contexto}"
        return ""

    def get_attention_summary(self) -> str:
        if not self.memory.registros:
            return ""
        recientes = [r for r in list(self.memory.registros)[-6:]]
        lineas = [f"{r.categoria}: {r.descripcion}" for r in recientes if r.descripcion]
        return "; ".join(lineas)

    # Per-person helpers
    def set_active_person(self, name: str, preferences: dict | None = None) -> None:
        """Set the active person profile and apply simple preference hints."""
        self.active_person = name
        if preferences:
            # Example: allow overriding attention weights per person
            prefs = preferences or {}
            aw = prefs.get("attention_weights")
            if isinstance(aw, dict):
                for k, v in aw.items():
                    try:
                        self.attention_weights[k] = float(v)
                    except Exception:
                        pass
            ignored = prefs.get("ignored_categories")
            if isinstance(ignored, list):
                self.ignored_categories = set(ignored)

    def set_profile_manager(self, profile_manager: Any) -> None:
        """Retain the legacy attachment point without granting write authority."""
        self.profile_manager = profile_manager

    def infer_preferences_from_text(self, text: str) -> None:
        """Deprecated no-op; preferences must emerge through Reasoning and Writers."""
        _ = text
        return None

    def persist_preference(self, key: str, value: object, confidence: float) -> bool:
        """Deprecated compatibility boundary; direct profile writes are forbidden."""
        _ = key, value, confidence
        return False

    def get_active_person(self) -> str | None:
        return getattr(self, "active_person", None)
