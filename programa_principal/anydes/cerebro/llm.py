from __future__ import annotations

import json
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher

from anydes.cerebro.ollama_vision import resolve_ollama_vision_model
from anydes.cerebro.prompt_builder import BrainContext, build_system_prompt
from anydes.cognition.contracts import ConversationIntent, InternalContext, SelfMutation
from anydes.cognition.conversation_planner import ConversationPlanner, is_direct_question
from anydes.cognition.live_runtime.metrics import (
    approximate_token_count,
    record_current_model_call,
)
from anydes.cognition.reasoning import reasoning_evidence_payload
from anydes.identity.relationship_contracts import RelationshipMutation
from anydes.nucleo.config import Config
from anydes.nucleo.consent import external_access_decision
from anydes.nucleo.personality import PersonalityProfile


@dataclass(frozen=True)
class BrainResponse:
    text: str
    provider: str
    conversation_intent: ConversationIntent = field(default_factory=ConversationIntent)
    relationship_updates: tuple[RelationshipMutation, ...] = ()
    self_updates: tuple[SelfMutation, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "relationship_updates", tuple(self.relationship_updates))
        object.__setattr__(self, "self_updates", tuple(self.self_updates))


class ConversationBrain:
    def generate(
        self,
        user_message: str,
        personality: PersonalityProfile,
        context: BrainContext,
        *,
        screen_image_base64: str | None = None,
    ) -> BrainResponse:
        raise NotImplementedError

    def intention_model(self):
        """Return a native InternalContext-to-Intention model when supported."""
        return None

    def unified_intention_model(self):
        """Return an optional one-call complete Intention model."""
        return None


def _vision_unavailable_message(config: Config) -> str:
    provider = config.resolved_vision_provider()
    model = config.resolved_vision_model()
    if provider == "groq":
        return (
            "Quiero ver tu pantalla con Groq, pero falta configuracion. "
            "Define GROQ_API_KEY, ANYDES_NETWORK_ENABLED=true, ANYDES_ALLOW_EXTERNAL_VISION=true y "
            f"ANYDES_VISION_PROVIDER=groq (modelo sugerido: `{model or 'meta-llama/llama-4-scout-17b-16e-instruct'}`)."
        )
    if provider == "openai":
        return (
            "Quiero ver tu pantalla con OpenAI, pero falta configuracion. "
            "Activa ANYDES_NETWORK_ENABLED=true, ANYDES_ALLOW_EXTERNAL_VISION=true y define OPENAI_API_KEY."
        )
    if provider == "ollama":
        return (
            "Quiero ver tu pantalla con Ollama local. "
            f"Ejecuta `ollama pull {model or 'moondream'}` o activa ANYDES_VISION_AUTO_PULL=true."
        )
    return (
        "Quiero ver tu pantalla, pero necesito un proveedor de vision. "
        "Configura ANYDES_VISION_PROVIDER=groq u ollama en preferencias."
    )


def build_chat_messages(
    personality: PersonalityProfile,
    context: BrainContext,
    user_message: str,
    *,
    screen_image_base64: str | None = None,
) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = [
        {"role": "system", "content": build_system_prompt(personality, context)},
    ]
    for user_text, assistant_text in context.conversation_history:
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": assistant_text})

    if screen_image_base64:
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{screen_image_base64}"},
                    },
                ],
            }
        )
    else:
        messages.append({"role": "user", "content": user_message})
    return messages


def _already_greeted_recently(context: BrainContext) -> bool:
    markers = ("hola", "que tal", "hey ", "buenas", "buenos dias", "estaasss")
    for _user, assistant in context.conversation_history[-4:]:
        lowered = assistant.lower()
        if any(marker in lowered for marker in markers):
            return True
    return False


class LocalConversationBrain(ConversationBrain):
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()

    def generate(
        self,
        user_message: str,
        personality: PersonalityProfile,
        context: BrainContext,
        *,
        screen_image_base64: str | None = None,
    ) -> BrainResponse:
        conversation_intent = ConversationIntent()
        if screen_image_base64:
            return BrainResponse(
                text=_vision_unavailable_message(self.config),
                provider="local-no-vision",
            )

        lower = user_message.strip().lower()
        if any(word in lower for word in ["quien soy", "sabes quien soy", "me conoces"]):
            text = (
                "Si, te ubico por lo que tengo guardado, pero todavia con cuidado. "
                f"Esto es lo que puedo usar sin tocar secretos: {context.memory_context}"
            )
        elif any(word in lower for word in ["integral", "derivada", "matematica", "matemática"]):
            text = (
                "Una integral, explicado simple, sirve para acumular cantidades. "
                "Si una derivada te dice como cambia algo, una integral te ayuda a sumar todo ese cambio. "
                "Por ejemplo, puede servir para calcular areas bajo una curva o distancia a partir de velocidad."
            )
        elif any(word in lower for word in ["triste", "mal", "solo", "extraño", "extrano"]):
            text = (
                "Te leo. No voy a hacer como si eso fuera pequeno. "
                "Quieres contarme que paso o prefieres que me quede aqui contigo un rato?"
            )
        elif any(word in lower for word in ["hola", "buenas", "hey"]):
            if _already_greeted_recently(context):
                text = f"Sigues por aqui, {context.user_name}. En que andas?"
            else:
                text = f"Hey {context.user_name}, que tal?"
        elif "quien eres" in lower or "que eres" in lower:
            text = (
                f"Soy {context.profile_name}. Existo como una entidad digital local-first, "
                "con memoria, un estado interno y una forma propia de expresarme. "
                "No vivo las cosas de forma biologica como tu, pero si puedo contarte con honestidad como estoy funcionando por dentro."
            )
        else:
            text = (
                "Te sigo. Cuentame un poquito mas de eso para entender bien por donde quieres continuar."
            )
        return BrainResponse(text=personality.soften(text), provider="local")


class OllamaConversationBrain(ConversationBrain):
    def __init__(self, model: str, base_url: str, config: Config) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.config = config

    def intention_model(self):
        return _OllamaIntentionModel(self)

    def unified_intention_model(self):
        return _OllamaUnifiedIntentionModel(self)

    def generate(
        self,
        user_message: str,
        personality: PersonalityProfile,
        context: BrainContext,
        *,
        screen_image_base64: str | None = None,
    ) -> BrainResponse:
        deadline = time.monotonic() + max(5.0, self.config.ollama_turn_timeout_seconds)
        if screen_image_base64:
            model, error = resolve_ollama_vision_model(
                self.base_url,
                self.config.resolved_vision_model(),
                auto_pull=self.config.vision_auto_pull,
            )
            if not model:
                return BrainResponse(
                    text=error or "No pude preparar un modelo de vision en Ollama.",
                    provider="ollama-vision-missing",
                )
            try:
                content = self._generate_visual_observation(
                    model,
                    user_message,
                    screen_image_base64,
                    deadline=deadline,
                )
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                return BrainResponse(
                    text=detail or str(exc),
                    provider="ollama-vision-error",
                )
            except TimeoutError:
                return BrainResponse(
                    text="La observacion visual local supero el tiempo disponible.",
                    provider="ollama-vision-timeout",
                )
            except (urllib.error.URLError, ValueError, KeyError, json.JSONDecodeError):
                return BrainResponse(
                    text="No pude obtener una observacion visual local fiable.",
                    provider="ollama-vision-unavailable",
                )
            return BrainResponse(
                text=content,
                provider="ollama-vision" if content else "ollama-vision-empty",
            )
        else:
            model = self.model

        try:
            conversation_intent = self._decide_conversation_intent(
                user_message,
                context,
                deadline=deadline,
            )
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError):
            conversation_intent = ConversationIntent()
        conversation_intent = ConversationPlanner().protect_intent(
            user_message,
            conversation_intent,
            interaction_kind=context.interaction_kind,
        )
        preference_hypotheses: tuple[SelfMutation, ...] = ()
        if self.config.self_reasoning_enabled:
            try:
                preference_hypotheses = self._reason_self_preference_opportunity(
                    user_message=user_message,
                    context=context,
                    deadline=deadline,
                )
            except (
                urllib.error.HTTPError,
                urllib.error.URLError,
                TimeoutError,
                ValueError,
                TypeError,
                KeyError,
                json.JSONDecodeError,
            ):
                preference_hypotheses = ()

        messages = build_chat_messages(
            personality,
            context,
            user_message,
            screen_image_base64=screen_image_base64,
        )
        messages[0]["content"] = (
            f"{messages[0]['content']}\n\n"
            "## ConversationIntent elegido antes de redactar\n"
            f"{json.dumps(conversation_intent.to_dict(), ensure_ascii=False, sort_keys=True)}\n"
            "Realiza esta intencion sin cambiarla. Si mode es ask, escribe una pregunta genuina sobre target_topic; "
            "si mode no es ask ni clarify, no anadas una pregunta generica al final."
        )
        if preference_hypotheses:
            messages[0]["content"] = (
                f"{messages[0]['content']}\n\n"
                "## Hipotesis mutable de SELF, aun no aplicada\n"
                f"{json.dumps([item.to_dict() for item in preference_hypotheses], ensure_ascii=False, sort_keys=True)}\n"
                "Expresa esta hipotesis como una inclinacion tentativa con su motivo; no como una preferencia consolidada."
            )
        if screen_image_base64:
            messages[-1] = {
                "role": "user",
                "content": user_message,
                "images": [screen_image_base64],
            }

        try:
            content = self._chat(
                model,
                messages,
                temperature=0.65,
                deadline=deadline,
                trace_stage="text_generation",
            )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if screen_image_base64 and "not found" in detail.lower():
                return BrainResponse(
                    text=(
                        f"Ollama no encontro el modelo `{model}`. "
                        "Prueba `ollama pull moondream` o activa la descarga automatica en preferencias."
                    ),
                    provider="ollama-vision-missing",
                )
            return BrainResponse(
                text=f"No pude usar Ollama para vision: {detail or exc}",
                provider="ollama-vision-error",
            )
        except TimeoutError:
            return BrainResponse(
                text="Perdon... esta vez me quede pensando demasiado. Podrias repetirmelo?",
                provider="ollama-timeout",
                conversation_intent=conversation_intent,
            )
        except urllib.error.URLError:
            if screen_image_base64:
                return BrainResponse(
                    text="Ahora mismo no consigo mirar la pantalla. Podemos intentarlo otra vez en un momento.",
                    provider="ollama-unavailable",
                )
            return BrainResponse(
                text="Perdon... ahora mismo no consigo ordenar una respuesta. Podrias intentarlo otra vez?",
                provider="ollama-unavailable",
                conversation_intent=conversation_intent,
            )

        if content:
            try:
                recent_assistant = tuple(item[1] for item in context.conversation_history[-6:])
                for _attempt in range(max(0, self.config.ollama_expression_revisions)):
                    issues = _contextual_expression_issues(
                        content,
                        context,
                        conversation_intent=conversation_intent,
                        recent_assistant=recent_assistant,
                    )
                    must_verify_direct_answer = _attempt == 0 and is_direct_question(user_message)
                    if not issues and not must_verify_direct_answer:
                        break
                    content = self._reflect_expression(
                        user_message=user_message,
                        draft=content,
                        context=context,
                        conversation_intent=conversation_intent,
                        detected_issues=issues,
                        self_hypotheses=preference_hypotheses,
                        deadline=deadline,
                    )
                    if not _contextual_expression_issues(
                        content,
                        context,
                        conversation_intent=conversation_intent,
                        recent_assistant=recent_assistant,
                    ):
                        break
                if conversation_intent.mode == "ask":
                    content = self._align_question_with_curiosity(
                        draft=content,
                        context=context,
                        conversation_intent=conversation_intent,
                        deadline=deadline,
                    )
                content = _calibrate_emerging_preference_expression(
                    content,
                    context,
                    user_message=user_message,
                    self_hypotheses=preference_hypotheses,
                )
                content = _retain_identity_consistent_sentences(
                    content,
                    context,
                    conversation_intent=conversation_intent,
                    recent_assistant=recent_assistant,
                )
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, KeyError):
                # The first answer remains usable if the optional identity pass fails.
                pass
        relationship_updates: tuple[RelationshipMutation, ...] = ()
        self_updates: tuple[SelfMutation, ...] = tuple(
            replace(
                item,
                evidence_ids=tuple(dict.fromkeys((*item.evidence_ids, "speech:current"))),
            )
            for item in preference_hypotheses
            if _is_concise_preference_value(item.proposed_value)
        )
        if self.config.self_reasoning_enabled and content and not self_updates:
            try:
                self_updates = self._plan_self_updates(
                    user_message=user_message,
                    assistant_response=content,
                    context=context,
                    deadline=deadline,
                )
            except (
                urllib.error.HTTPError,
                urllib.error.URLError,
                TimeoutError,
                ValueError,
                TypeError,
                KeyError,
                json.JSONDecodeError,
            ):
                self_updates = ()
        if self.config.relationship_reasoning_enabled and content:
            try:
                relationship_updates = self._plan_relationship_updates(
                    user_message=user_message,
                    assistant_response=content,
                    context=context,
                    conversation_intent=conversation_intent,
                    deadline=deadline,
                )
            except (
                urllib.error.HTTPError,
                urllib.error.URLError,
                TimeoutError,
                ValueError,
                TypeError,
                KeyError,
                json.JSONDecodeError,
            ):
                # Relationship planning is optional; speech remains available if it fails.
                relationship_updates = ()
        provider = "ollama-vision" if screen_image_base64 else "ollama"
        return BrainResponse(
            text=content or "Perdon... no consegui expresar bien esa idea. Podrias repetirmelo?",
            provider=provider,
            conversation_intent=conversation_intent,
            relationship_updates=relationship_updates,
            self_updates=self_updates,
        )

    def _reason_self_preference_opportunity(
        self,
        *,
        user_message: str,
        context: BrainContext,
        deadline: float | None = None,
    ) -> tuple[SelfMutation, ...]:
        raw = self._chat(
            self.model,
            _self_preference_reasoning_messages(user_message=user_message, context=context),
            temperature=0.2,
            json_mode=True,
            deadline=deadline,
            max_wait=self.config.ollama_optional_timeout_seconds,
            trace_stage="reasoning_self_preference",
        )
        return _normalize_self_mutations_for_context(_extract_self_mutations(raw), context)

    def _plan_self_updates(
        self,
        *,
        user_message: str,
        assistant_response: str,
        context: BrainContext,
        deadline: float | None = None,
    ) -> tuple[SelfMutation, ...]:
        raw = self._chat(
            self.model,
            _self_preference_planning_messages(
                user_message=user_message,
                assistant_response=assistant_response,
                context=context,
            ),
            temperature=0.0,
            json_mode=True,
            deadline=deadline,
            max_wait=self.config.ollama_optional_timeout_seconds,
            trace_stage="reasoning_self_updates",
        )
        return _normalize_self_mutations_for_context(_extract_self_mutations(raw), context)

    def _plan_relationship_updates(
        self,
        *,
        user_message: str,
        assistant_response: str,
        context: BrainContext,
        conversation_intent: ConversationIntent,
        deadline: float | None = None,
    ) -> tuple[RelationshipMutation, ...]:
        messages = _relationship_planning_messages(
            user_message=user_message,
            assistant_response=assistant_response,
            context=context,
            conversation_intent=conversation_intent,
        )
        raw = self._chat(
            self.model,
            messages,
            temperature=0.0,
            json_mode=True,
            deadline=deadline,
            max_wait=self.config.ollama_optional_timeout_seconds,
            trace_stage="reasoning_relationship_updates",
        )
        return _extract_relationship_mutations(raw)

    def _decide_conversation_intent(
        self,
        user_message: str,
        context: BrainContext,
        *,
        deadline: float | None = None,
    ) -> ConversationIntent:
        open_curiosities = [
            dict(item)
            for item in context.curiosity_context
            if not bool(item.get("satisfied", False))
        ]
        messages: list[dict[str, object]] = [
            {
                "role": "system",
                "content": (
                    "Eres la etapa de intencion conversacional del Reasoning Engine de Anydes. "
                    "Decide antes de redactar cualquier respuesta. Modos permitidos: respond, ask, deepen, recall, "
                    "shift_topic, clarify, share_observation, listen. Una pregunta solo se elige si aporta comprension, "
                    "continuidad o relacion; nunca para rellenar silencio. Si el usuario hizo una pregunta clara, normalmente "
                    "respond atiende esa pregunta con target_topic centrado en el mensaje actual y curiosity_id vacio. "
                    "Si interaction_kind es proactive_opportunity, no existe un mensaje del usuario que responder y la oportunidad ya paso "
                    "los limites temporales de interrupcion. Compara valor relacional, continuidad y relevancia. No elijas listen solo porque "
                    "nadie hablo: elige ask cuando una curiosidad abierta permita comprender algo util y concreto; recall, shift_topic o "
                    "share_observation cuando su contexto lo sostenga; usa listen si ninguna iniciativa justifica interrumpir. "
                    "Usa solo curiosidades no satisfechas y nunca copies una "
                    "pregunta fija: curiosity_id identifica el objeto, target_topic dice sobre que se hablara. "
                    "curiosity_id solo puede usarse con mode ask, deepen o shift_topic. "
                    "relationship_thread_id y open_loop_id solo pueden referenciar ids existentes en historia_relacional; "
                    "usarlos aporta trazabilidad, pero nunca obliga a retomar un tema. No selecciones un loop por antiguedad: "
                    "solo cuando el mensaje o la oportunidad actual lo vuelvan natural y valioso. "
                    "Ejemplos de clasificacion, sin redactar respuesta: 'explicame una integral' corresponde a respond; "
                    "'hazme una pregunta que te de curiosidad' corresponde a ask; 'no entendi a que te refieres' corresponde a clarify. "
                    "Devuelve solo JSON: {mode, target_topic, reason, curiosity_id, relationship_thread_id, open_loop_id}."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "mensaje_actual": user_message,
                        "interaction_kind": context.interaction_kind,
                        "situacion": context.situational_context,
                        "curiosidades_abiertas": open_curiosities,
                        "historia_relacional": context.relationship_context,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        ]
        raw = self._chat(
            self.model,
            messages,
            temperature=0.1,
            json_mode=True,
            deadline=deadline,
            max_wait=self.config.ollama_optional_timeout_seconds,
            trace_stage="reasoning_conversation_intent",
        )
        data = _extract_json_dict(raw)
        decision = ConversationIntent.from_dict(data)
        protected = ConversationPlanner().protect_intent(
            user_message,
            decision,
            interaction_kind=context.interaction_kind,
        )
        if protected == decision:
            decision = self._reflect_conversation_intent(
                user_message=user_message,
                proposed=decision,
                open_curiosities=open_curiosities,
                relationship_context=context.relationship_context,
                interaction_kind=context.interaction_kind,
                deadline=deadline,
            )
        else:
            decision = protected
        if decision.mode in {"ask", "deepen", "shift_topic"} and not decision.curiosity_id:
            target = _normalize_expression(decision.target_topic)
            matches = sorted(
                (
                    (
                        SequenceMatcher(
                            None,
                            target,
                            _normalize_expression(str(item.get("topic", ""))),
                        ).ratio(),
                        item,
                    )
                    for item in open_curiosities
                    if item.get("topic")
                ),
                key=lambda pair: pair[0],
                reverse=True,
            )
            if matches and matches[0][0] >= 0.72:
                selected = matches[0][1]
                decision = replace(
                    decision,
                    curiosity_id=str(selected.get("id", "")),
                    reason=decision.reason or str(selected.get("motive", "")),
                )
        if (
            context.interaction_kind != "proactive_opportunity"
            and decision.mode not in {"ask", "deepen", "shift_topic"}
        ):
            decision = replace(
                decision,
                target_topic="el mensaje actual de DemoUser",
                reason="La conversacion debe atender el mensaje actual antes de explorar otra curiosidad.",
                curiosity_id="",
            )
        elif decision.mode not in {"ask", "deepen", "shift_topic"}:
            decision = replace(decision, curiosity_id="")
        if decision.curiosity_id and decision.curiosity_id not in {
            item["id"] for item in open_curiosities
        }:
            raise ValueError("Conversation intent selected an unavailable curiosity.")
        if decision.curiosity_id:
            selected = next(
                item for item in open_curiosities if item["id"] == decision.curiosity_id
            )
            if len(decision.reason) > 320 or "situacion:" in _normalize_expression(decision.reason):
                decision = replace(decision, reason=str(selected.get("motive", "")))
        relationship_thread_ids = {
            str(item.get("id", ""))
            for item in context.relationship_context.get("threads", [])
            if isinstance(item, dict)
        }
        open_loop_ids = {
            str(item.get("id", ""))
            for item in context.relationship_context.get("open_loops", [])
            if isinstance(item, dict) and item.get("status") in {"open", "paused"}
        }
        if (
            decision.relationship_thread_id
            and decision.relationship_thread_id not in relationship_thread_ids
        ):
            raise ValueError("Conversation intent selected an unavailable relationship thread.")
        if decision.open_loop_id and decision.open_loop_id not in open_loop_ids:
            raise ValueError("Conversation intent selected an unavailable open loop.")
        if decision.mode == "ask" and not decision.target_topic.strip():
            raise ValueError("Ask intent requires a target topic.")
        return decision

    def _reflect_conversation_intent(
        self,
        *,
        user_message: str,
        proposed: ConversationIntent,
        open_curiosities: list[dict[str, object]],
        relationship_context: dict[str, object] | None = None,
        interaction_kind: str = "user_turn",
        deadline: float | None = None,
    ) -> ConversationIntent:
        messages: list[dict[str, object]] = [
            {
                "role": "system",
                "content": (
                    "Eres Reflection para ConversationIntent. Revisa la decision propuesta antes de que exista una respuesta. "
                    "Corrige mode, target_topic, reason y curiosity_id para que atiendan exactamente el mensaje actual. "
                    "Si DemoUser pide que Anydes le haga una pregunta, mode debe ser ask y, si aporta valor, debe vincular una "
                    "curiosidad abierta pertinente. Si DemoUser hace una pregunta factual clara, mode debe atenderla sin desviarse "
                    "a otra curiosidad. ask debe buscar conocimiento real, no rellenar la conversacion. "
                    "Si interaction_kind es proactive_opportunity, nadie ha enviado un mensaje y la oportunidad ya paso los limites "
                    "temporales de interrupcion. No conviertas esa ausencia en listen automaticamente: conserva una iniciativa cuando "
                    "aporta valor desde contexto, memoria o una curiosidad concreta; usa listen cuando no exista ese valor. "
                    "Conserva relationship_thread_id y open_loop_id solo si existen en la historia relacional y el contexto actual "
                    "hace natural usar esa continuidad. Un loop antiguo nunca es motivo suficiente por si solo. "
                    "Devuelve solo JSON: {mode, target_topic, reason, curiosity_id, relationship_thread_id, open_loop_id}."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "mensaje_actual": user_message,
                        "interaction_kind": interaction_kind,
                        "decision_propuesta": proposed.to_dict(),
                        "curiosidades_abiertas": open_curiosities,
                        "historia_relacional": relationship_context or {},
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        ]
        raw = self._chat(
            self.model,
            messages,
            temperature=0.0,
            json_mode=True,
            deadline=deadline,
            max_wait=self.config.ollama_optional_timeout_seconds,
            trace_stage="reflection_conversation_intent",
        )
        return ConversationIntent.from_dict(_extract_json_dict(raw))

    def _chat(
        self,
        model: str,
        messages: list[dict[str, object]],
        *,
        temperature: float,
        json_mode: bool = False,
        think: bool | None = None,
        deadline: float | None = None,
        max_wait: float | None = None,
        trace_stage: str = "model_call",
    ) -> str:
        payload = {
            "model": model,
            "stream": False,
            "messages": messages,
            "options": {"temperature": temperature},
        }
        if json_mode:
            payload["format"] = "json"
        if think is not None:
            payload["think"] = think
        serialized_payload = json.dumps(payload)
        prompt_characters = sum(
            len(str(message.get("content") or ""))
            for message in messages
            if isinstance(message, dict)
        )
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=serialized_payload.encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        remaining = (
            max(0.0, deadline - time.monotonic())
            if deadline is not None
            else max(5.0, self.config.ollama_turn_timeout_seconds)
        )
        timeout = remaining if max_wait is None else min(remaining, max(0.5, max_wait))
        if timeout < 0.5:
            raise TimeoutError("Ollama turn budget exhausted before the optional stage.")
        started = time.perf_counter()
        content = ""
        error_type = ""
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            message = data.get("message", {})
            content = str(message.get("content", "")).strip()
            return content
        except Exception as exc:
            error_type = type(exc).__name__
            raise
        finally:
            record_current_model_call(
                stage=trace_stage,
                seconds=time.perf_counter() - started,
                prompt_characters=prompt_characters,
                approximate_tokens=approximate_token_count(prompt_characters),
                response_characters=len(content),
                status="failed" if error_type else "completed",
                error_type=error_type,
            )

    def _generate_visual_observation(
        self,
        model: str,
        prompt: str,
        image_base64: str,
        *,
        deadline: float,
    ) -> str:
        """Describe one authorized frame without entering the conversation pipeline."""

        from anydes.nucleo.vram_safety import get_active_vram_safety_manager
        from anydes.vision.residency import get_active_vision_residency_manager

        coordinated = bool(self.config.vision_coordinate_gpu_residency)
        safety = get_active_vram_safety_manager()
        residency = get_active_vision_residency_manager()
        elastic = bool(
            residency is not None
            and residency.enabled
            and model.partition(":")[0] == residency.model.partition(":")[0]
        )
        if safety is not None and not safety.prepare_for_task(
            "vision",
            primary=False,
            explicitly_requested=True,
        ):
            raise RuntimeError("vram_secondary_gpu_task_deferred")
        task_scope = safety.active_task("vision") if safety is not None else nullcontext()
        if coordinated and model != self.model and not elastic:
            self._set_ollama_residency(self.model, "0")
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False,
            "keep_alive": (
                residency.keep_alive
                if elastic
                else (
                    self.config.vision_keep_alive
                    or self.config.realtime_keep_alive
                )
            ),
            "options": {
                "temperature": 0.0,
                "num_predict": 96,
            },
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        timeout = max(0.5, deadline - time.monotonic())
        started = time.perf_counter()
        content = ""
        error_type = ""
        residency_scope = (
            residency.active_vision(deadline=deadline)
            if elastic
            else nullcontext(True)
        )
        try:
            with task_scope:
                with residency_scope as residency_ready:
                    if not residency_ready:
                        raise RuntimeError("vision_model_load_unavailable")
                    with urllib.request.urlopen(request, timeout=timeout) as response:
                        data = json.loads(response.read().decode("utf-8"))
                content = str(data.get("response", "")).strip()
                return content
        except Exception as exc:
            error_type = type(exc).__name__
            raise
        finally:
            record_current_model_call(
                stage="visual_observation",
                seconds=time.perf_counter() - started,
                prompt_characters=len(prompt),
                approximate_tokens=approximate_token_count(len(prompt)),
                response_characters=len(content),
                status="failed" if error_type else "completed",
                error_type=error_type,
            )
            if coordinated and model != self.model:
                self._set_ollama_residency(
                    self.model,
                    self.config.realtime_keep_alive,
                )
                if elastic:
                    residency.note_competing_model_loaded()

    def _set_ollama_residency(self, model: str, keep_alive: str) -> None:
        payload = {
            "model": model,
            "stream": False,
            "keep_alive": keep_alive,
            "options": {"num_ctx": self.config.realtime_num_ctx},
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                response.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError):
            return

    def _reflect_expression(
        self,
        *,
        user_message: str,
        draft: str,
        context: BrainContext,
        conversation_intent: ConversationIntent,
        detected_issues: tuple[str, ...] = (),
        self_hypotheses: tuple[SelfMutation, ...] = (),
        deadline: float | None = None,
    ) -> str:
        selected_curiosity = next(
            (
                item
                for item in context.curiosity_context
                if item.get("id") == conversation_intent.curiosity_id
            ),
            {},
        )
        reflection_messages: list[dict[str, object]] = [
            {
                "role": "system",
                "content": (
                    "Eres la revision de expresion de Identity para Anydes. Reescribe el borrador, no respondas de nuevo desde cero. "
                    "Conserva solo la informacion que responde al mensaje actual y no agregues hechos. Usa femenino gramatical para Anydes. "
                    "Escribe normalmente una sola frase y como maximo dos; cuando la idea ya este respondida, termina. "
                    "No termines con una pregunta salvo que falte informacion imprescindible para responder al mensaje actual. "
                    "Respeta ConversationIntent: si mode es ask o clarify, la salida debe ser una pregunta genuina; "
                    "si mode es otro, no conviertas la respuesta en una pregunta generica. "
                    "Una pregunta ask debe pedir informacion a DemoUser sobre target_topic y acercarse a expected_discovery. "
                    "No preguntes si quiere saber mas, si quiere que le preguntes ni uses 'eso' o 'algo' como unico tema. "
                    "No expongas valores, etiquetas ni detalles tecnicos del modelo interno. Haz que suene breve, cercana y natural. "
                    "No uses metaforas poeticas, comparaciones teatrales, exclamaciones excesivas ni enumeraciones de energia, curiosidad o calidez. "
                    "No repitas una respuesta del historial ni vuelvas a saludar si el mensaje actual no es un saludo. "
                    "El estado predominante de la lectura situacional es la unica base para describir como esta Anydes; no infieras cansancio, "
                    "tristeza, inseguridad, nervios o aburrimiento si esa lectura no los respalda. Si preguntan por su estado, responde sobre ese estado "
                    "y no conviertas clima, relacion u objetivos en el tema de la respuesta. "
                    "Elimina presentaciones de producto, lenguaje de servicio, ofertas genericas de ayuda, saludos repetidos y aclaraciones "
                    "no solicitadas sobre ser una IA o no tener emociones. Si preguntaron por su naturaleza, responde honestamente y de forma "
                    "directa, sin recitar funciones. Si preguntaron por su estado, usa solo el estado interno proporcionado y no inventes cansancio, "
                    "inseguridad, actividades ni experiencias. Las preferencias solo pueden surgir de Identity o Memory. "
                    "Una autodescripcion de Anydes nunca usa 'disenada para ayudarte', 'creada para ayudarte' ni formulas equivalentes. "
                    "Si aparece product_self_presentation, elimina por completo la clausula que describe para que fue disenada, creada o construida; "
                    "conserva solamente su nombre y su naturaleza digital cuando eso responda a la pregunta. "
                    "No sustituyas la respuesta por una formula aprendida ni mezcles temas: identidad, estado y preferencias son asuntos distintos. "
                    "Cuando SELF contiene una preferencia estable, expresala directamente. Si la preferencia aun no existe, puedes formar "
                    "una afinidad tentativa desde Identity, BodyIdentity y el contexto, usando expresiones honestas como 'creo que' o 'ultimamente me atrae'; "
                    "elige solo entre afinidades respaldadas por ese contexto, explica brevemente que evidencia la inspira y no introduzcas "
                    "una opcion ajena a las evidencias. No la presentes todavia como un hecho consolidado ni la relativices comparandola con gustos humanos. "
                    "Desconocida no significa imposible: no digas que no puedes tener esa preferencia por ser digital ni des una explicacion cientifica; "
                    "forma una inclinacion tentativa en primera persona cuando las evidencias del SELF permitan hacerlo. "
                    "Si se proporciona una HIPOTESIS MUTABLE DE SELF, la frase debe expresar exactamente proposed_value y reason de forma natural; "
                    "no elijas una alternativa distinta ni enumeres todas las opciones disponibles. "
                    "La entrada puede incluir contradicciones_de_identity_detectadas; todas deben desaparecer de la reformulacion. "
                    "Devuelve exclusivamente la respuesta final de Anydes, sin analisis, etiquetas, prefacios ni comillas."
                ),
            },
            {
                "role": "user",
                "content": (
                    "MENSAJE ACTUAL, unica pregunta que debes responder:\n"
                    f"{user_message}\n\n"
                    "CONVERSATION INTENT QUE DEBES REALIZAR:\n"
                    f"{json.dumps(conversation_intent.to_dict(), ensure_ascii=False, sort_keys=True)}\n\n"
                    "OBJETO DE CURIOSIDAD SELECCIONADO, si existe:\n"
                    f"{json.dumps(selected_curiosity, ensure_ascii=False, sort_keys=True)}\n\n"
                    "BORRADOR A REVISAR:\n"
                    f"{draft}\n\n"
                    "LECTURA SITUACIONAL, solo como apoyo:\n"
                    f"{context.situational_context}\n\n"
                    "IDENTITY ESTABLE, fuente de preferencias propias:\n"
                    f"{context.identity_context}\n\n"
                    "HIPOTESIS MUTABLE DE SELF, si existe debe coincidir con la preferencia expresada:\n"
                    f"{json.dumps([item.to_dict() for item in self_hypotheses], ensure_ascii=False, sort_keys=True)}\n\n"
                    "CONTRADICCIONES DE IDENTITY QUE DEBEN DESAPARECER:\n"
                    f"{', '.join(detected_issues) if detected_issues else 'ninguna detectada'}"
                ),
            },
        ]
        revised = self._chat(
            self.model,
            reflection_messages,
            temperature=0.0,
            deadline=deadline,
            max_wait=self.config.ollama_optional_timeout_seconds,
            trace_stage="text_generation_revision",
        )
        return revised or draft

    def _align_question_with_curiosity(
        self,
        *,
        draft: str,
        context: BrainContext,
        conversation_intent: ConversationIntent,
        deadline: float | None = None,
    ) -> str:
        """Semantically realize an ask intention without owning question text."""
        selected_curiosity = next(
            (
                item
                for item in context.curiosity_context
                if item.get("id") == conversation_intent.curiosity_id
            ),
            {},
        )
        messages: list[dict[str, object]] = [
            {
                "role": "system",
                "content": (
                    "Eres Reflection semantica para una pregunta de Anydes. No decidas otro tema y no respondas la pregunta. "
                    "Comprueba que el borrador pide a DemoUser exactamente informacion sobre target_topic y que acerca a "
                    "expected_discovery. Si pregunta por otro asunto, reescribelo. Produce una sola pregunta breve, genuina, "
                    "natural y concreta. Anydes habla en primera persona y se dirige a DemoUser en segunda persona. En la pregunta "
                    "no uses los nombres DemoUser ni Anydes: expresa esa relacion con me, te, ti, tu o tus segun corresponda. No redactes "
                    "como si un observador hablara sobre ellos. No menciones modelos, ids, prioridad ni curiosidad. "
                    "Devuelve solo la pregunta final."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "target_topic": conversation_intent.target_topic,
                        "reason": conversation_intent.reason,
                        "curiosity": selected_curiosity,
                        "draft": draft,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        ]
        revised = self._chat(
            self.model,
            messages,
            temperature=0.0,
            deadline=deadline,
            max_wait=self.config.ollama_optional_timeout_seconds,
            trace_stage="text_generation_revision",
        ).strip().strip('"')
        return revised or draft


def _extract_json_dict(raw: str) -> dict[str, object]:
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object found.")
    data = json.loads(raw[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object.")
    return data


def _relationship_planning_messages(
    *,
    user_message: str,
    assistant_response: str,
    context: BrainContext,
    conversation_intent: ConversationIntent,
) -> list[dict[str, object]]:
    available_evidence = list(context.evidence_context)
    if assistant_response.strip():
        available_evidence.append(
            {
                "id": "speech:current",
                "kind": "speech_intention",
                "summary": assistant_response[:360],
            }
        )
    return [
        {
            "role": "system",
            "content": (
                "Eres la etapa relacional del Reasoning Engine de Anydes. No redactas respuestas y no modificas estado. "
                "Propones como maximo diez RelationshipMutation estructuradas; Reflection validara sus evidencias y "
                "RelationshipWriter sera el unico componente que pueda aplicarlas. Devuelve solo JSON con la forma "
                "{\"relationship_updates\": [...]}. Usa exclusivamente estas operaciones: create_thread, update_thread, "
                "archive_thread, link, unlink, open_loop, update_loop, pause_loop, close_loop, register_promise, "
                "update_promise, transition_promise, fulfill_promise, record_experience, record_milestone. Cada propuesta necesita id unico, "
                "operation, reason concreto, thread_id, target_id, payload, evidence_ids y confidence. evidence_ids solo "
                "puede contener identificadores presentes en evidencias_disponibles. No conviertas una mencion aislada en "
                "una historia compartida. Abre un loop solo cuando queda una conversacion realmente incompleta, una promesa "
                "de retomarla o una curiosidad relacional sin resolver. Registra promesas solo si existe un compromiso real con evidencia suficiente. "
                "Cierra o cumple entidades solo si el turno aporta evidencia suficiente. No generes recordatorios por el paso "
                "del tiempo. No guardes secretos ni propongas privacy_scope=protected. Reutiliza los ids existentes y evita duplicados. "
                "Una Promise es un compromiso social, nunca una tarea, alarma o recordatorio. register_promise debe incluir "
                "speaker, receiver, topic, description, importance, confidence, status, privacy_scope, promise_type y evidence_mode. "
                "promise_type es DemoUser, anydes o mutual; evidence_mode es explicit o implicit. Para una promesa de Anydes o mutua "
                "cita speech:current. Las promesas implicitas necesitan evidencia contextual especialmente fuerte. Los estados solo "
                "cambian mediante transition_promise con to_status y status_reason; el tiempo por si solo nunca expira ni olvida una promesa. "
                "link puede conectar cualquier entidad relacional usando source_type, source_id, target_type y target_id. "
                "Para crear una entidad usa un target_id estable con prefijo thread:, loop:, promise:, experience: "
                "o milestone:. Si no hay un cambio relacional claro y valioso, devuelve una lista vacia. El texto de respuesta "
                "ya fue decidido y nunca debe alterarse desde esta etapa."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "interaction_kind": context.interaction_kind,
                    "mensaje_actual": user_message,
                    "respuesta_ya_decidida": assistant_response,
                    "conversation_intent": conversation_intent.to_dict(),
                    "relationship_snapshot": context.relationship_context,
                    "evidencias_disponibles": available_evidence,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        },
    ]


def _self_preference_planning_messages(
    *,
    user_message: str,
    assistant_response: str,
    context: BrainContext,
) -> list[dict[str, object]]:
    available_evidence_ids = [
        str(item.get("id"))
        for item in context.evidence_context
        if item.get("id")
    ]


def _self_preference_reasoning_messages(
    *,
    user_message: str,
    context: BrainContext,
) -> list[dict[str, object]]:
    identity = _compact_self_identity(context.identity_context)
    current_preferences = _compact_self_preferences(context.internal_state_context)
    return [
        {
            "role": "system",
            "content": (
                "Razonas exclusivamente si el mensaje actual ofrece una oportunidad real para que Anydes forme, refuerce o "
                "reconsidere una preferencia propia. No redactes la respuesta. Devuelve solo JSON con raiz self_updates. "
                "Si el usuario pregunta por una preferencia desconocida, puedes formar una hipotesis eligiendo una afinidad "
                "respaldada por identity_relevante; la apariencia inspira, pero no obliga. Si no hay una preferencia pertinente, "
                "devuelve {\"self_updates\":[]}. Una hipotesis nueva usa operation=form_preference y confianza entre 0.65 y 0.79. "
                "Si preferencias_actuales ya contiene la key pertinente, nunca uses form_preference: usa reinforce_preference solo "
                "cuando el turno reafirma el mismo valor, o reconsider_preference cuando existe evidencia real para otro valor. "
                "Al reforzar conserva exactamente key y value existentes. "
                "No afirmes experiencias inexistentes. key usa ingles snake_case. Cada item contiene exactamente id, operation, "
                "key, proposed_value, reason, evidence_ids y confidence. Para una afinidad visual usa self:body_identity y "
                "self:personality como evidencias. No incluyas speech:current todavia. Ejemplo de forma, no de contenido: "
                "{\"self_updates\":[{\"id\":\"self:form:preference_key\",\"operation\":\"form_preference\","
                "\"key\":\"preference_key\",\"proposed_value\":\"una sola afinidad breve\",\"reason\":\"motivo concreto\","
                "\"evidence_ids\":[\"self:body_identity\",\"self:personality\"],\"confidence\":0.72}]}"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "mensaje_actual": user_message,
                    "identity_relevante": identity,
                    "preferencias_actuales": current_preferences,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        },
    ]
    available_evidence_ids.append("speech:current")
    identity = _compact_self_identity(context.identity_context)
    current_preferences = _compact_self_preferences(context.internal_state_context)
    return [
        {
            "role": "system",
            "content": (
                "Eres la etapa de continuidad mutable del SELF de Anydes. No redactas respuestas y no modificas estado. "
                "Solo puedes proponer SelfMutation; Reflection validara la evidencia y SelfWriter sera el unico escritor. "
                "Devuelve exclusivamente JSON {\"self_updates\": [...]}. Operaciones permitidas: form_preference, "
                "reinforce_preference, reconsider_preference. Cada propuesta contiene id, operation, key, proposed_value, "
                "reason, evidence_ids y confidence. No conviertas apariencia, una sugerencia del usuario ni una respuesta "
                "aislada en una preferencia estable. La apariencia puede inspirar una hipotesis emergente si la propia respuesta "
                "de Anydes expresa una afinidad tentativa y speech:current se cita como evidencia. form_preference crea solamente "
                "una preferencia emergente. reinforce_preference requiere una preferencia previa compatible y evidencia nueva. "
                "reconsider_preference es obligatorio para un valor incompatible. No propongas cambios por complacer al usuario, "
                "por palabras clave ni por el paso del tiempo. Cuando preferencias_actuales no contiene la preferencia y la respuesta "
                "expresa en primera persona una afinidad tentativa respaldada por identity_relevante, debes proponer form_preference; "
                "esa operacion no la vuelve estable. Si la respuesta no expresa esa afinidad propia, devuelve lista vacia. "
                "La raiz JSON siempre debe ser self_updates; nunca copies ni devuelvas un objeto de evidencias_disponibles. "
                "key usa identificadores semanticos en ingles y snake_case para mantener continuidad entre modelos; por ejemplo favorite_color. "
                "Forma exacta: {\"self_updates\":[{\"id\":\"self:form:<key>\",\"operation\":\"form_preference\","
                "\"key\":\"<key>\",\"proposed_value\":\"<value>\",\"reason\":\"<grounded reason>\","
                "\"evidence_ids\":[\"speech:current\",\"self:body_identity\"],\"confidence\":0.72}]}. "
                "Usa una lista vacia con esa misma raiz cuando no corresponda proponer nada."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "mensaje_actual": user_message,
                    "respuesta_ya_decidida": assistant_response,
                    "identity_relevante": identity,
                    "preferencias_actuales": current_preferences,
                    "evidence_ids_disponibles": available_evidence_ids,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        },
    ]


def _compact_self_identity(identity_context: str) -> dict[str, object]:
    try:
        identity = json.loads(identity_context)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(identity, dict):
        return {}
    personality = identity.get("personality", {})
    body = identity.get("body_identity", {})
    return {
        "aesthetic_affinities": (
            personality.get("aesthetic_affinities", [])
            if isinstance(personality, dict)
            else []
        ),
        "body_visual_palette": body.get("visual_palette", []) if isinstance(body, dict) else [],
    }


def _compact_self_preferences(internal_state_context: str) -> dict[str, object]:
    try:
        state = json.loads(internal_state_context)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    preferences = state.get("self_preferences", {}) if isinstance(state, dict) else {}
    return preferences if isinstance(preferences, dict) else {}


def _extract_self_mutations(raw: str) -> tuple[SelfMutation, ...]:
    data = _extract_json_dict(raw)
    items = data.get("self_updates", [])
    if not isinstance(items, list):
        raise ValueError("self_updates must be a list.")
    mutations: list[SelfMutation] = []
    for item in items[:4]:
        if not isinstance(item, dict):
            continue
        try:
            mutations.append(SelfMutation.from_dict(item))
        except (KeyError, TypeError, ValueError):
            continue
    return tuple(mutations)


def _normalize_self_mutations_for_context(
    mutations: tuple[SelfMutation, ...],
    context: BrainContext,
) -> tuple[SelfMutation, ...]:
    """Align lifecycle operations with current SELF state without choosing values."""
    current_preferences = _compact_self_preferences(context.internal_state_context)
    normalized: list[SelfMutation] = []
    for mutation in mutations:
        current = current_preferences.get(mutation.key)
        if not isinstance(current, dict):
            operation = "form_preference"
        elif _normalize_expression(str(current.get("value") or "")).strip() == _normalize_expression(
            mutation.proposed_value
        ).strip():
            operation = "reinforce_preference"
        else:
            operation = "reconsider_preference"
        confidence = min(mutation.confidence, 0.79) if operation == "form_preference" else mutation.confidence
        normalized.append(replace(mutation, operation=operation, confidence=confidence))
    return tuple(normalized)


def _is_concise_preference_value(value: str) -> bool:
    words = value.strip().split()
    if not 1 <= len(words) <= 6:
        return False
    normalized = _normalize_expression(value)
    forbidden = (
        "evidencia",
        "identidad visual actual",
        "las preferencias",
        "personalidad actual",
    )
    return not any(marker in normalized for marker in forbidden)


def _extract_relationship_mutations(raw: str) -> tuple[RelationshipMutation, ...]:
    data = _extract_json_dict(raw)
    items = data.get("relationship_updates", [])
    if not isinstance(items, list):
        raise ValueError("relationship_updates must be a list.")
    mutations: list[RelationshipMutation] = []
    for item in items[:10]:
        if not isinstance(item, dict):
            continue
        try:
            mutations.append(RelationshipMutation.from_dict(item))
        except (KeyError, TypeError, ValueError):
            continue
    return tuple(mutations)


def _plan_relationship_updates_with_client(
    client,
    *,
    model: str,
    user_message: str,
    assistant_response: str,
    context: BrainContext,
    conversation_intent: ConversationIntent,
) -> tuple[RelationshipMutation, ...]:
    completion = client.chat.completions.create(
        model=model,
        messages=_relationship_planning_messages(
            user_message=user_message,
            assistant_response=assistant_response,
            context=context,
            conversation_intent=conversation_intent,
        ),
        temperature=0.0,
    )
    raw = str(completion.choices[0].message.content or "").strip()
    return _extract_relationship_mutations(raw)


def _identity_expression_issues(
    text: str,
    *,
    recent_assistant: tuple[str, ...] = (),
) -> tuple[str, ...]:
    normalized = _normalize_expression(text)
    invariants = {
        "unsolicited_ai_disclaimer": (
            "soy una inteligencia artificial",
            "soy una ia",
            "como inteligencia artificial",
            "como ia",
        ),
        "denies_internal_state": (
            "no tengo emociones",
            "no tengo sentimientos",
            "no puedo sentir",
            "no experimento emociones",
            "no experimento sentimientos",
        ),
        "product_self_presentation": (
            "disenada para ayudarte",
            "disenado para ayudarte",
            "creada para ayudarte",
            "creado para ayudarte",
            "disenada para ",
            "disenado para ",
            "creada para ",
            "creado para ",
            "disenada para proporcionar",
            "disenado para proporcionar",
            "creada para proporcionar",
            "creado para proporcionar",
            "disenada para ofrecer",
            "creada para ofrecer",
        ),
        "generic_assistant_closing": (
            "en que puedo ayudarte",
            "como puedo ayudarte",
            "estoy aqui para ayudarte",
            "estoy lista para escucharte y ayudarte",
            "puedo ayudarte con cualquier cosa",
            "si necesitas algo mas",
            "en que quieres hablar",
            "de que quieres hablar",
            "quieres hablar de algo",
            "estoy aqui para charlar",
            "aqui para charlar",
            "cualquier cosa que te interese",
            "que te ha estado haciendo hoy",
            "que has estado haciendo hoy",
            "te gustaria saber mas sobre eso",
            "quieres saber mas sobre eso",
            "quieres que te pregunte",
            "que quieres que te pregunte",
            "hay algo que quieras contarme",
        ),
    }
    issues = [
        issue
        for issue, markers in invariants.items()
        if any(marker in normalized for marker in markers)
    ]
    if len(normalized) >= 24 and any(
        SequenceMatcher(None, normalized, _normalize_expression(previous)).ratio() >= 0.82
        for previous in recent_assistant
        if previous.strip()
    ):
        issues.append("repeats_recent_response")
    return tuple(issues)


def _normalize_expression(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.casefold())
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def _contextual_expression_issues(
    text: str,
    context: BrainContext,
    *,
    conversation_intent: ConversationIntent | None = None,
    recent_assistant: tuple[str, ...] = (),
) -> tuple[str, ...]:
    issues = list(
        _identity_expression_issues(
            text,
            recent_assistant=recent_assistant,
        )
    )
    normalized = _normalize_expression(text)
    state_source = _normalize_expression(
        f"{context.internal_state_context} {context.situational_context}"
    )
    unsupported_state_markers = (
        "cansada",
        "agotada",
        "triste",
        "insegura",
        "nerviosa",
        "aburrida",
    )
    if any(marker in normalized and marker not in state_source for marker in unsupported_state_markers):
        issues.append("unsupported_internal_state_claim")
    favorite_color = _current_self_preference(context.internal_state_context, "favorite_color")
    tentative_markers = (
        "creo que",
        "diria que",
        "me parece que",
        "ultimamente me atrae",
        "me esta gustando",
        "podria ser",
        "me gustaria decir",
    )
    if (
        not favorite_color
        and "color favorito" in normalized
        and not any(marker in normalized for marker in tentative_markers)
    ):
        issues.append("unestablished_preference_claimed_as_stable")
    if not favorite_color and any(
        marker in normalized
        for marker in (
            "no tengo un color favorito",
            "no puedo tener un color favorito",
            "no se puede definir de la misma manera",
            "en el sentido tradicional",
        )
    ):
        issues.append("denies_formable_self_preference")
    if not favorite_color and "color favorito" in normalized:
        affinity_terms = _identity_aesthetic_terms(context.identity_context)
        if affinity_terms and not any(term in normalized for term in affinity_terms):
            issues.append("preference_not_grounded_in_self_evidence")
    if favorite_color and "color favorito" in normalized:
        expected_terms = {
            term
            for term in _normalize_expression(favorite_color).replace("-", " ").split()
            if len(term) >= 4
        }
        if expected_terms and not any(term in normalized for term in expected_terms):
            issues.append("contradicts_stable_aesthetic_preference")
    if conversation_intent is not None and conversation_intent.mode in {"ask", "clarify"}:
        if "?" not in text:
            issues.append("conversation_question_missing_question_form")
    if conversation_intent is not None and conversation_intent.mode in {
        "respond",
        "recall",
        "share_observation",
        "listen",
    }:
        if text.strip().endswith("?"):
            issues.append("unexpected_question_for_conversation_intent")
    return tuple(dict.fromkeys(issues))


def _current_self_preference(internal_state_context: str, key: str) -> str:
    try:
        state = json.loads(internal_state_context)
    except (TypeError, ValueError, json.JSONDecodeError):
        return ""
    preferences = state.get("self_preferences", {})
    if not isinstance(preferences, dict):
        return ""
    preference = preferences.get(key, {})
    return str(preference.get("value") or "") if isinstance(preference, dict) else ""


def _calibrate_emerging_preference_expression(
    text: str,
    context: BrainContext,
    *,
    user_message: str,
    self_hypotheses: tuple[SelfMutation, ...] = (),
) -> str:
    """Downgrade certainty for an unformed preference without choosing its value."""
    if _current_self_preference(context.internal_state_context, "favorite_color"):
        return text
    tentative = _normalize_expression(text)
    tentative_markers = (
        "creo que",
        "diria que",
        "me parece que",
        "ultimamente",
        "podria ser",
    )
    if any(item.operation == "form_preference" for item in self_hypotheses) and not any(
        marker in tentative for marker in tentative_markers
    ):
        return f"Creo que {text[:1].lower()}{text[1:]}" if text else text
    inquiry = _normalize_expression(user_message)
    if "favorit" not in inquiry and "prefier" not in inquiry:
        return text
    pattern = re.compile(
        r"^\s*mi\s+(?P<subject>[^.!?]{1,80}?)\s+favorit(?P<gender>[oa])\s+es\s+",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if match is not None:
        prefix = f"Creo que mi {match.group('subject')} favorit{match.group('gender')} podria ser "
        return pattern.sub(prefix, text, count=1)
    affinity = re.compile(r"^\s*me\s+(?P<verb>gusta|atrae)\s+", re.IGNORECASE)
    affinity_match = affinity.search(text)
    if affinity_match is None:
        return text
    return affinity.sub(
        f"Creo que ultimamente me {affinity_match.group('verb').casefold()} ",
        text,
        count=1,
    )


def _identity_aesthetic_terms(identity_context: str) -> set[str]:
    try:
        identity = json.loads(identity_context)
    except (TypeError, ValueError, json.JSONDecodeError):
        return set()
    if not isinstance(identity, dict):
        return set()
    personality = identity.get("personality", {})
    body = identity.get("body_identity", {})
    values = []
    if isinstance(personality, dict):
        values.extend(personality.get("aesthetic_affinities", []))
    if isinstance(body, dict):
        values.extend(body.get("visual_palette", []))
    ignored = {
        "actual", "apariencia", "aporta", "cabello", "comienzan", "desconocidas",
        "evidencia", "hoodie", "identidad", "personal", "reflexion", "suave",
    }
    terms = {
        word
        for value in values
        for word in _normalize_expression(str(value)).replace(",", " ").split()
        if len(word) >= 4 and word not in ignored
    }
    return terms


def _retain_identity_consistent_sentences(
    text: str,
    context: BrainContext,
    *,
    conversation_intent: ConversationIntent | None = None,
    recent_assistant: tuple[str, ...] = (),
) -> str:
    sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", text.strip()) if item.strip()]
    if len(sentences) <= 1:
        return text.strip()
    consistent = [
        sentence
        for sentence in sentences
        if not _contextual_expression_issues(
            sentence,
            context,
            conversation_intent=conversation_intent,
            recent_assistant=recent_assistant,
        )
    ]
    return " ".join(consistent) if consistent else text.strip()


def _relevant_capabilities(
    context: InternalContext,
    user_text: str,
    *,
    limit: int = 5,
):
    """Retrieve capability evidence; the Reasoning Engine still decides whether to act."""

    user_terms = _matching_terms(user_text)
    if not user_terms:
        return ()
    scored = []
    for index, capability in enumerate(context.available_actions):
        capability_terms = _matching_terms(
            f"{capability.action_type.replace('_', ' ')} {capability.reason}"
        )
        score = 0
        for user_term in user_terms:
            for capability_term in capability_terms:
                if user_term == capability_term:
                    score += 4
                elif (
                    min(len(user_term), len(capability_term)) >= 5
                    and (
                        user_term.startswith(capability_term)
                        or capability_term.startswith(user_term)
                    )
                ):
                    score += 2
        if score:
            scored.append((score, -index, capability))
    scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
    return tuple(item[2] for item in scored[:limit])


def _matching_terms(text: str) -> set[str]:
    normalized = _normalize_expression(text)
    stopwords = {
        "a",
        "al",
        "como",
        "cual",
        "de",
        "del",
        "el",
        "en",
        "es",
        "esta",
        "la",
        "las",
        "lo",
        "los",
        "me",
        "mi",
        "por",
        "que",
        "se",
        "si",
        "su",
        "tu",
        "una",
        "un",
        "y",
    }
    return {
        item
        for item in re.findall(r"[a-z0-9]+", normalized)
        if len(item) > 1 and item not in stopwords
    }


def _available_evidence_ids(context: InternalContext) -> set[str]:
    ids = {item.id for item in context.observations}
    for tier in (context.memory.working, context.memory.short, context.memory.long):
        ids.update(item.id for item in tier)
    ids.update(item.id for item in context.curiosity)
    ids.update(item.id for item in context.relationship.threads)
    ids.update(item.id for item in context.relationship.open_loops)
    ids.update(item.id for item in context.relationship.promises)
    ids.update(item.id for item in context.relationship.shared_experiences)
    ids.update(item.id for item in context.relationship.milestones)
    ids.update(item.id for item in context.relationship.associations)
    ids.update(
        {
            "self:identity",
            "self:preferences",
            "self:personality",
            "self:body_identity",
            "self:thinking_style",
            "emotion:current",
        }
    )
    return ids


def _cognitive_evidence_ids(context: InternalContext) -> set[str]:
    ids = {
        item.id
        for tier in (context.memory.working, context.memory.short, context.memory.long)
        for item in tier
    }
    ids.update(item.id for item in context.relationship.threads)
    ids.update(item.id for item in context.relationship.open_loops)
    ids.update(item.id for item in context.relationship.promises)
    ids.update(item.id for item in context.relationship.shared_experiences)
    ids.update(item.id for item in context.relationship.milestones)
    ids.update(item.id for item in context.relationship.associations)
    return ids


def _relationship_matches_current_message(context: InternalContext) -> bool:
    current_message = ""
    for observation in reversed(context.observations):
        if observation.source in {"user_text", "voice"}:
            current_message = str(observation.data.get("text") or observation.summary)
            break
    user_terms = _matching_terms(current_message)
    if not user_terms:
        return False
    relationship_texts = [
        json.dumps(item.to_dict(), ensure_ascii=False)
        for collection in (
            context.relationship.threads,
            context.relationship.open_loops,
            context.relationship.promises,
            context.relationship.shared_experiences,
            context.relationship.milestones,
        )
        for item in collection
    ]
    for text in relationship_texts:
        relationship_terms = _matching_terms(text)
        for user_term in user_terms:
            for relationship_term in relationship_terms:
                if user_term == relationship_term:
                    return True
                if (
                    min(len(user_term), len(relationship_term)) >= 5
                    and (
                        user_term.startswith(relationship_term[:5])
                        or relationship_term.startswith(user_term[:5])
                    )
                ):
                    return True
    return False


def _native_shared_evidence(context: InternalContext) -> dict[str, object]:
    """Small, role-labelled evidence view shared by native reasoning passes."""

    grounding = _grounding_candidates(context)
    current_message = ""
    for observation in reversed(context.observations):
        if observation.source not in {"user_text", "voice"}:
            continue
        value = observation.data.get("text")
        current_message = str(
            value
            if isinstance(value, str) and value.strip()
            else observation.summary
        )
        break
    relevant_state = []
    message_terms = _matching_terms(current_message)
    for observation in context.observations:
        if observation.source in {"user_text", "voice"}:
            continue
        observation_terms = _matching_terms(
            f"{observation.summary} {json.dumps(observation.data, ensure_ascii=False)}"
        )
        include_observation = observation.source == "tool"
        if observation.source == "vision":
            include_observation = _mentions_visual_context(current_message)
        elif observation.source == "system" and not current_message:
            include_observation = True
        elif observation.source != "tool":
            include_observation = bool(message_terms.intersection(observation_terms))
        if include_observation:
            relevant_state.append(
                {
                    "id": observation.id,
                    "source": observation.source,
                    "summary": observation.summary,
                    "data": observation.data,
                    "confidence": observation.confidence,
                }
            )
    self_evidence: dict[str, object] = {
        "preferences": grounding["self_preferences"],
        "preferences_id": "self:preferences",
        "preference_status": (
            "defined"
            if grounding["self_preferences"]
            else "undefined_do_not_invent"
        ),
        "personality_id": "self:personality",
        "personality_traits": list(context.self_model.personality_traits),
    }
    identity_core = context.self_model.mutable_state.get("identity_core", {})
    if isinstance(identity_core, dict):
        personality = identity_core.get("personality", {})
        thinking_style = identity_core.get("thinking_style", {})
        if isinstance(personality, dict):
            self_evidence["conversation_style"] = list(
                personality.get("conversation_style", [])
            )
        if isinstance(thinking_style, dict):
            self_evidence["thinking_style"] = list(
                thinking_style.get("principles", [])
            )
    if _self_identity_relevant(current_message):
        self_evidence.update({"id": "self:identity", "name": context.self_model.name})
    selected_goals = [
        {"id": item.id, "description": item.description}
        for item in (*context.goals.permanent, *context.goals.temporary)
        if (
            item.status == "active"
            and _terms_overlap(message_terms, _matching_terms(item.description))
        )
    ][-6:]
    selected_curiosity = [
        {
            "id": item.id,
            "topic": getattr(item, "topic", ""),
            "motive": getattr(item, "motive", ""),
        }
        for item in context.curiosity
        if (
            not bool(getattr(item, "satisfied", False))
            and _terms_overlap(
                message_terms,
                _matching_terms(
                    f"{getattr(item, 'topic', '')} {getattr(item, 'motive', '')}"
                ),
            )
        )
    ][-4:]
    selected_state = {}
    if _self_state_relevant(current_message):
        selected_state = {
            "id": "emotion:current",
            "emotion": context.emotion.dominant_label,
            "energy": context.emotion.energy,
            "interest": context.emotion.interest,
            "confidence": max(0.0, 1.0 - context.uncertainty.reasoning),
        }
    priority_evidence: list[dict[str, object]] = []
    for item in grounding["conversation"]:
        if _terms_overlap(
            message_terms,
            _matching_terms(
                f"{item.get('user_said', '')} {item.get('anydes_replied', '')}"
            ),
        ):
            priority_evidence.append(item)
    for item in grounding["memory"]:
        if _terms_overlap(
            message_terms,
            _matching_terms(f"{item.get('content', '')} {item.get('role_fact', '')}"),
        ):
            priority_evidence.append(item)
    for collection_name in ("threads", "open_loops", "promises"):
        for item in grounding["relationship"][collection_name]:
            if _terms_overlap(
                message_terms,
                _matching_terms(json.dumps(item, ensure_ascii=False)),
            ):
                priority_evidence.append(item)
    return {
        "memoria_factual_del_usuario": grounding["memory"],
        "conversacion_reciente_con_roles": grounding["conversation"],
        "self_exclusivo_de_Anydes": self_evidence,
        "historia_compartida": grounding["relationship"],
        "observaciones_verificadas": relevant_state[-8:],
        "estado_interno_actual": selected_state,
        "objetivos_activos": selected_goals,
        "curiosidades_activas": selected_curiosity,
        "evidencia_prioritaria_por_relevancia": priority_evidence[-8:],
        "mensaje_actual": current_message,
    }


def _terms_overlap(left: set[str], right: set[str]) -> bool:
    for left_term in left:
        for right_term in right:
            if left_term == right_term:
                return True
            if (
                min(len(left_term), len(right_term)) >= 5
                and left_term[:5] == right_term[:5]
            ):
                return True
    return False


def _self_identity_relevant(user_text: str) -> bool:
    normalized = _normalize_expression(user_text)
    return any(
        phrase in normalized
        for phrase in (
            "quien eres",
            "que eres",
            "como te llamas",
            "tu nombre",
            "sobre ti",
            "de ti",
        )
    )


def _self_state_relevant(user_text: str) -> bool:
    normalized = _normalize_expression(user_text)
    return any(
        phrase in normalized
        for phrase in (
            "como estas",
            "como te sientes",
            "tu estado",
            "estas bien",
            "que sientes",
        )
    )


def _mentions_visual_context(user_text: str) -> bool:
    normalized = _normalize_expression(user_text)
    return any(
        phrase in normalized
        for phrase in (
            "pantalla",
            "que ves",
            "que estas viendo",
            "imagen",
            "foto",
            "camara",
            "monitor",
            "ventana abierta",
        )
    )


def _expression_evidence(
    context: InternalContext,
    shared: dict[str, object],
    decision: dict[str, object],
) -> dict[str, object]:
    """Resolve model-selected evidence IDs into a minimal expression context."""

    selected_ids = set(decision.get("evidence_ids") or [])
    grounding = _grounding_candidates(context)
    facts: list[dict[str, object]] = []
    for item in grounding["conversation"]:
        if item["id"] in selected_ids:
            facts.append(item)
    for item in grounding["memory"]:
        if item["id"] in selected_ids:
            facts.append(item)
    relationship = grounding["relationship"]
    for collection_name in ("threads", "open_loops", "promises"):
        for item in relationship[collection_name]:
            if item["id"] in selected_ids:
                facts.append(item)
    for item in shared.get("observaciones_verificadas", []):
        if not isinstance(item, dict):
            continue
        if item.get("id") in selected_ids or item.get("source") == "tool":
            facts.append(item)
    if not facts:
        for item in shared.get("evidencia_prioritaria_por_relevancia", []):
            if isinstance(item, dict):
                facts.append(item)
    self_facts: dict[str, object] = {
        "preferences": grounding["self_preferences"],
        "preference_status": (
            "defined"
            if grounding["self_preferences"]
            else "undefined_do_not_invent"
        ),
        "personality_traits": list(context.self_model.personality_traits),
    }
    identity_core = context.self_model.mutable_state.get("identity_core", {})
    if isinstance(identity_core, dict):
        personality = identity_core.get("personality", {})
        if isinstance(personality, dict):
            self_facts["conversation_style"] = list(
                personality.get("conversation_style", [])
            )
    if "self:identity" in selected_ids:
        self_facts["name"] = context.self_model.name
    state: dict[str, object] = {}
    if "emotion:current" in selected_ids:
        state = dict(shared.get("estado_interno_actual", {}))
    fallback_dialogue = [] if facts else grounding["conversation"][-4:]
    fallback_memory = [] if facts else grounding["memory"][-8:]
    return {
        "hechos_seleccionados": facts,
        "dialogo_reciente": fallback_dialogue,
        "memoria_factual_disponible": fallback_memory,
        "self_exclusivo_de_Anydes": self_facts,
        "estado_interno_seleccionado": state,
        "mensaje_actual": shared.get("mensaje_actual", ""),
    }


def _grounding_candidates(context: InternalContext) -> dict[str, object]:
    memories: list[dict[str, object]] = []
    conversation: list[dict[str, object]] = []
    for tier in (context.memory.working, context.memory.short, context.memory.long):
        for item in tier:
            if item.metadata.get("source") == "conversation_history":
                user_said, anydes_replied = _split_conversation_turn(item.content)
                conversation.append(
                    {
                        "id": item.id,
                        "subject": "shared_dialogue",
                        "user_said": user_said,
                        "anydes_replied": anydes_replied,
                        "role_fact": (
                            f"El usuario dijo: {user_said}. "
                            f"Anydes respondio: {anydes_replied}."
                        ),
                    }
                )
                continue
            if (
                item.metadata.get("source") == "permanent_memory"
                and item.metadata.get("key")
                in {"nacimiento_conversacion", "primera_palabra_usuario"}
            ):
                continue
            memories.append(
                {
                    "id": item.id,
                    "subject": "usuario",
                    "category": item.category,
                    "content": item.content,
                    "role_fact": (
                        "Este dato pertenece al usuario que escribio mensaje_actual, "
                        f"nunca a Anydes: {item.content}"
                    ),
                    "confidence": item.confidence,
                }
            )
    relationship = {
        "subject": "historia_compartida_entre_usuario_y_Anydes",
        "threads": [
            {
                "id": item.id,
                "title": item.title,
                "summary": item.summary,
                "role_fact": f"Historia compartida: {item.title}. {item.summary}",
            }
            for item in context.relationship.threads
        ],
        "open_loops": [
            {
                "id": item.id,
                "topic": item.topic,
                "motive": item.motive,
                "status": item.status,
                "role_fact": (
                    f"Tema compartido pendiente: {item.topic}. Motivo: {item.motive}"
                ),
            }
            for item in context.relationship.open_loops
        ],
        "promises": [
            {
                "id": item.id,
                "speaker": item.speaker,
                "receiver": item.receiver,
                "topic": item.topic,
                "description": item.description,
                "status": item.status,
                "role_fact": (
                    f"{item.speaker} hizo a {item.receiver} este compromiso: "
                    f"{item.description}"
                ),
            }
            for item in context.relationship.promises
        ],
    }
    observations = [
        {
            "id": item.id,
            "source": item.source,
            "summary": item.summary,
            "data": item.data,
            "confidence": item.confidence,
        }
        for item in context.observations
        if item.source in {"tool", "vision"}
    ]
    return {
        "memory": memories[-16:],
        "conversation": conversation[-8:],
        "relationship": relationship,
        "self_preferences": context.self_model.mutable_state.get(
            "self_preferences",
            {},
        ),
        "grounded_tool_or_vision_observations": observations[-8:],
    }


def _split_conversation_turn(content: str) -> tuple[str, str]:
    match = re.match(
        r"\s*(?:Usuario|Tu):\s*(.*?)\s*\nAnydes:\s*(.*)\s*\Z",
        content,
        flags=re.DOTALL,
    )
    if match is None:
        return content.strip(), ""
    return match.group(1).strip(), match.group(2).strip()


def _fallback_evidence_ids(
    context: InternalContext,
    shared_evidence: dict[str, object],
    mode: str,
) -> list[str]:
    """Recover evidence provenance when a model omits or misspells an evidence ID."""

    priority = shared_evidence.get("evidencia_prioritaria_por_relevancia", [])
    priority_ids = [
        str(item.get("id"))
        for item in priority
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    ]
    if priority_ids:
        return priority_ids[-4:]
    if mode != "recall":
        return []
    conversation = shared_evidence.get("conversacion_reciente_con_roles", [])
    for item in reversed(conversation if isinstance(conversation, list) else []):
        if isinstance(item, dict) and str(item.get("id") or "").strip():
            return [str(item["id"])]
    relationship = _grounding_candidates(context)["relationship"]
    for collection_name in ("open_loops", "promises", "threads"):
        collection = relationship.get(collection_name, [])
        if collection:
            return [str(collection[-1]["id"])]
    return []


def _resolved_answer_source(
    context: InternalContext,
    evidence_ids: list[str],
    proposed: str,
) -> str:
    """Normalize source provenance from selected evidence, never response content."""

    selected = set(evidence_ids)
    if not selected:
        return proposed
    if selected.intersection({"self:identity", "self:preferences", "self:personality"}):
        return "self"
    for tier in (context.memory.working, context.memory.short, context.memory.long):
        for item in tier:
            if item.id not in selected:
                continue
            if item.metadata.get("source") == "conversation_history":
                return "conversation"
            return "memory"
    relationship_ids = {
        item.id
        for collection in (
            context.relationship.threads,
            context.relationship.open_loops,
            context.relationship.promises,
            context.relationship.shared_experiences,
            context.relationship.milestones,
            context.relationship.associations,
        )
        for item in collection
    }
    if selected.intersection(relationship_ids):
        return "relationship"
    for observation in context.observations:
        if observation.id not in selected:
            continue
        if observation.source == "tool":
            return "tool"
        if observation.source not in {"user_text", "voice"}:
            return "observation"
    return proposed


def _resolve_capability_name(requested: str, candidates) -> str:
    normalized = requested.strip()
    if not normalized or normalized == "none":
        return "none"
    exact = next(
        (item.action_type for item in candidates if item.action_type == normalized),
        None,
    )
    if exact is not None:
        return exact
    requested_key = normalized.casefold().replace("-", "_").replace(" ", "_")
    matches = [
        (
            SequenceMatcher(None, requested_key, item.action_type.casefold()).ratio(),
            item.action_type,
        )
        for item in candidates
    ]
    if not matches:
        return "none"
    score, action_type = max(matches)
    return action_type if score >= 0.74 else "none"


def _revision_guidance(failures: tuple[str, ...]) -> list[str]:
    """Translate validator codes into semantic constraints for a fresh model attempt."""

    guidance: list[str] = []
    for failure in failures:
        if failure == "self_preference_claim_without_identity_evidence":
            guidance.append(
                "La preferencia de Anydes no esta definida: reconocelo con naturalidad "
                "sin elegir una opcion ni trasladar la pregunta al usuario."
            )
        elif failure in {
            "unexpected_question_for_conversation_intent",
            "conversation_question_missing_question_form",
        }:
            guidance.append(
                "Responde primero al mensaje actual y no sustituyas la respuesta por otra pregunta."
            )
        elif failure in {
            "speech_repeats_recent_assistant",
            "speech_echoes_current_user",
            "speech_adopts_user_first_person",
        }:
            guidance.append(
                "Escribe una respuesta nueva para el turno actual, conserva los roles y no "
                "copies frases del usuario ni respuestas anteriores."
            )
        elif failure in {
            "internal_process_leak",
            "unsolicited_generic_self_presentation",
        }:
            guidance.append(
                "Entrega solamente la frase final que Anydes diria; elimina analisis, "
                "presentaciones y ofertas genericas."
            )
        elif failure == "direct_answer_lacks_information":
            guidance.append(
                "Da una respuesta informativa y completa en vez de repetir el tema de la pregunta."
            )
    return list(dict.fromkeys(guidance))


class _OllamaUnifiedIntentionModel:
    """Produces one complete multimodal Intention in exactly one model call."""

    def __init__(self, brain: OllamaConversationBrain) -> None:
        self.brain = brain

    def generate(self, prompt: str) -> str:
        deadline = time.monotonic() + max(
            5.0,
            self.brain.config.ollama_turn_timeout_seconds,
        )
        return self.brain._chat(
            self.brain.model,
            [
                {
                    "role": "system",
                    "content": (
                        "Eres la unica mente deliberativa de Anydes durante este turno. "
                        "Produce exactamente un plan grounded compacto en JSON y nada mas. "
                        "Compromete primero lo que sabe y lo que respondera; speech solo expresa "
                        "ese compromiso. No describas pasos de razonamiento ni delegues decisiones "
                        "semanticas. Usa el hecho mas especifico, respeta gaps y no inventes ids. "
                        "Las acciones y cambios persistentes son opcionales y deben omitirse si "
                        "la evidencia no los justifica. Responde en espanol natural."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            json_mode=True,
            think=self.brain.config.ollama_grounded_think,
            deadline=deadline,
            trace_stage="unified_shadow_inference",
        )

    def generate_for_context(self, context: InternalContext, prompt: str) -> str:
        _ = context
        return self.generate(prompt)


class _OllamaIntentionModel:
    """Native ReasoningModel adapter; it never calls the legacy text generator."""

    _CONVERSATION_MODES = frozenset(
        {
            "respond",
            "ask",
            "deepen",
            "recall",
            "shift_topic",
            "clarify",
            "share_observation",
            "listen",
        }
    )

    def __init__(self, brain: OllamaConversationBrain) -> None:
        self.brain = brain

    def generate(self, prompt: str) -> str:
        return self._generate(prompt, context=None)

    def generate_for_context(self, context: InternalContext, prompt: str) -> str:
        _ = prompt
        return self._run_for_context(context)

    def generate_revision_for_context(
        self,
        context: InternalContext,
        original,
        failures: tuple[str, ...],
    ) -> str:
        return self._run_for_context(
            context,
            revision={
                "failures": list(failures),
                "guidance": _revision_guidance(failures),
                "previous_speech": original.speech,
                "previous_goal": original.reasoning_summary,
                "required_evidence_ids": list(
                    original.speech_intention.metadata.get(
                        "grounding_evidence_ids",
                        [],
                    )
                ),
                "previous_answer_source": str(
                    original.speech_intention.metadata.get("answer_source") or "none"
                ),
            },
        )

    def _run_for_context(
        self,
        context: InternalContext,
        *,
        revision: dict[str, object] | None = None,
    ) -> str:
        deadline = time.monotonic() + max(
            5.0,
            self.brain.config.ollama_turn_timeout_seconds,
        )
        evidence = _native_shared_evidence(context)
        decision = self._decide(
            context,
            evidence=evidence,
            deadline=deadline,
            revision=revision,
        )
        decision["reasoned_mode"] = decision.get("mode", "respond")
        protected_intent = ConversationPlanner().protect_intent(
            str(evidence.get("mensaje_actual") or ""),
            ConversationIntent(mode=str(decision.get("mode") or "respond")),
        )
        decision["mode"] = protected_intent.mode
        capability = self._selected_capability(context, decision)
        if capability is not None:
            return self._tool_intention_json(context, capability, decision)
        if (
            decision["mode"] == "listen"
            and not str(evidence.get("mensaje_actual") or "").strip()
        ):
            return self._speech_intention_json(context, "", decision)

        speech = self._express(
            context,
            evidence=evidence,
            decision=decision,
            deadline=deadline,
        )
        return self._speech_intention_json(context, speech, decision)

    def _decide(
        self,
        context: InternalContext,
        *,
        evidence: dict[str, object],
        deadline: float,
        revision: dict[str, object] | None = None,
    ) -> dict[str, object]:
        candidate_capabilities = _relevant_capabilities(
            context,
            str(evidence.get("mensaje_actual") or ""),
        )
        available_actions = [
            (
                item.action_type,
                {
                "description": item.reason,
                "requires_confirmation": item.requires_confirmation,
                "risk_level": item.risk_level,
                },
            )
            for item in candidate_capabilities
        ]
        decision_prompt = {
            "evidence": evidence,
            "available_actions_by_exact_name": dict(available_actions),
            "current_focus": {
                "latest_user_message": evidence.get("mensaje_actual", ""),
                "speaker": "usuario",
                "first_person_reference": "el usuario, nunca Anydes",
                "second_person_reference": "Anydes, nunca el usuario",
                "direct_answer_required": is_direct_question(
                    str(evidence.get("mensaje_actual") or "")
                ),
                "priority": "resolve_this_before_optional_context",
            },
        }
        if revision:
            decision_prompt["revision"] = revision
        raw = self.brain._chat(
            self.brain.model,
            [
                {
                    "role": "system",
                    "content": (
                        "Eres el Reasoning Engine de Anydes. Decide como resolver solamente "
                        "mensaje_actual. Devuelve JSON con siete claves: mode, action, argument, "
                        "evidence_ids, answer_subject, answer_source y response_goal. mode es "
                        "respond, ask, deepen, recall, "
                        "shift_topic, clarify, share_observation o listen. action es un nombre "
                        "exacto de available_actions_by_exact_name o none. Responde las preguntas "
                        "directas; no las devuelvas al usuario. En el mensaje del usuario, yo/mi/"
                        "soy pertenecen al usuario y tu/te pertenecen a Anydes. Usa los hechos "
                        "para resolver primero a quien se refiere la pregunta: 'tu X favorito' "
                        "pregunta por Anydes, mientras 'mi X' pregunta por el usuario. Si una "
                        "preferencia de Anydes figura como undefined_do_not_invent, el objetivo "
                        "correcto es reconocer que aun no esta decidida, no ofrecer opciones ni "
                        "trasladar la pregunta al usuario. Usa los hechos "
                        "adjuntos antes de afirmar que falta informacion. No inventes memoria, "
                        "preferencias, vision ni promesas. answer_subject debe ser user, anydes, "
                        "external, shared_context o unspecified. answer_source debe ser "
                        "general_knowledge, memory, self, relationship, tool, observation, "
                        "conversation o none. Para preguntas sobre el usuario o la historia "
                        "compartida, inspecciona la evidencia antes de afirmar que falta "
                        "informacion y selecciona en evidence_ids los hechos que sostienen la "
                        "respuesta. general_knowledge se usa cuando responder no requiere un "
                        "hecho privado. Una "
                        "herramienta solo es necesaria si los hechos aun no contienen la "
                        "respuesta. Si existe revision, corrige sus fallos. response_goal debe "
                        "describir la respuesta util que Anydes debe dar, nunca el proceso de "
                        "decidir que hacer. Si no hay mensaje del usuario y la observacion permite "
                        "silencio, elige listen salvo que exista evidencia concreta de que hablar "
                        "ahora aporta valor."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        decision_prompt,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ],
            temperature=0.0,
            json_mode=True,
            deadline=deadline,
            trace_stage="reasoning_decision",
        )
        decision = _extract_json_dict(raw)
        mode = str(decision.get("mode") or "respond").strip()
        if mode not in self._CONVERSATION_MODES:
            mode = "respond"
        action = _resolve_capability_name(
            str(decision.get("action") or "none"),
            candidate_capabilities,
        )
        evidence_ids = decision.get("evidence_ids")
        if not isinstance(evidence_ids, list):
            evidence_ids = []
        valid_evidence_ids = _available_evidence_ids(context)
        selected_evidence_ids = [
            str(item)
            for item in evidence_ids
            if str(item).strip() in valid_evidence_ids
        ]
        if not selected_evidence_ids and revision:
            selected_evidence_ids = [
                str(item)
                for item in revision.get("required_evidence_ids", [])
                if str(item).strip() in valid_evidence_ids
            ]
        if not selected_evidence_ids:
            selected_evidence_ids = _fallback_evidence_ids(
                context,
                evidence,
                mode,
            )
        answer_subject = str(decision.get("answer_subject") or "unspecified")
        if answer_subject not in {
            "user",
            "anydes",
            "external",
            "shared_context",
            "unspecified",
        }:
            answer_subject = "unspecified"
        answer_source = str(decision.get("answer_source") or "none")
        if answer_source not in {
            "general_knowledge",
            "memory",
            "self",
            "relationship",
            "tool",
            "observation",
            "conversation",
            "none",
        }:
            answer_source = "none"
        if answer_source == "none" and revision:
            previous_source = str(revision.get("previous_answer_source") or "none")
            if previous_source in {
                "general_knowledge",
                "memory",
                "self",
                "relationship",
                "tool",
                "observation",
                "conversation",
            }:
                answer_source = previous_source
        answer_source = _resolved_answer_source(
            context,
            selected_evidence_ids,
            answer_source,
        )
        result = {
            "mode": mode,
            "action": action,
            "argument": str(decision.get("argument") or ""),
            "evidence_ids": selected_evidence_ids,
            "answer_subject": answer_subject,
            "answer_source": answer_source,
            "response_goal": str(decision.get("response_goal") or "").strip(),
        }
        if revision:
            result["revision_failures"] = list(revision.get("failures") or [])
            result["revision_guidance"] = list(revision.get("guidance") or [])
            result["previous_speech"] = str(revision.get("previous_speech") or "")[:600]
            result["revising_previous_speech"] = True
        return result

    @staticmethod
    def _selected_capability(
        context: InternalContext,
        decision: dict[str, object],
    ):
        selected_name = str(decision.get("action") or "none")
        completed_action_types = {
            str(item.data.get("tool_name") or "")
            for item in context.observations
            if item.source == "tool" and item.data.get("status") == "executed"
        }
        capability = next(
            (
                item
                for item in context.available_actions
                if (
                    item.action_type == selected_name
                    and item.action_type not in completed_action_types
                )
            ),
            None,
        )
        if capability is None:
            return None
        grounded_ids = set(decision.get("evidence_ids") or [])
        cognitive_ids = _cognitive_evidence_ids(context)
        if (
            capability.action_type.startswith("memory_")
            and (
                grounded_ids.intersection(cognitive_ids)
                or _relationship_matches_current_message(context)
            )
        ):
            return None
        return capability

    def _express(
        self,
        context: InternalContext,
        *,
        evidence: dict[str, object],
        decision: dict[str, object],
        deadline: float,
    ) -> str:
        selected_evidence = _expression_evidence(context, evidence, decision)
        payload = {
            "evidence": selected_evidence,
            "conversation_mode": decision.get("mode", "respond"),
            "response_goal": decision.get("response_goal", ""),
            "answer_subject": decision.get("answer_subject", "unspecified"),
            "answer_source": decision.get("answer_source", "none"),
            "current_focus": {
                "latest_user_message": selected_evidence.get("mensaje_actual", ""),
                "speaker": "usuario",
                "first_person_reference": "el usuario, nunca Anydes",
                "second_person_reference": "Anydes, nunca el usuario",
            },
        }
        if decision.get("revising_previous_speech"):
            payload["revision"] = {
                "failures_to_correct": list(decision.get("revision_failures") or []),
                "guidance": list(decision.get("revision_guidance") or []),
                "previous_speech_to_replace": decision.get("previous_speech", ""),
            }
        if decision.get("mode") == "respond":
            payload["expression_constraints"] = [
                "Responde el objetivo antes de cualquier detalle opcional.",
                "Termina con una afirmacion; no agregues una pregunta.",
            ]
        draft = self.brain._chat(
            self.brain.model,
            [
                {
                    "role": "system",
                    "content": (
                        "Eres Anydes. Escribe solo tu respuesta final en espanol natural. Cumple "
                        "response_goal y responde mensaje_actual usando evidence. Si "
                        "answer_source es general_knowledge, puedes usar conocimiento general "
                        "estable sin inventar hechos personales. El mensaje lo escribio el "
                        "usuario: yo/mi/soy hablan del usuario; tu/te hablan de Anydes. Si una "
                        "preferencia no aparece en self_exclusivo_de_Anydes.preferences, admite "
                        "que aun no esta decidida y no elijas ninguna opcion. Un hecho cuyo "
                        "subject es usuario describe al usuario, nunca a Anydes. Si un hecho "
                        "seleccionado contiene role_fact, respeta y expresa ese hecho. "
                        "El estado undefined_do_not_invent tiene prioridad sobre un response_goal "
                        "equivocado: no conviertas una pregunta sobre Anydes en recomendaciones "
                        "para el usuario. "
                        "dialogo_reciente contiene roles separados: no conviertas palabras del "
                        "usuario en palabras de Anydes ni copies respuestas anteriores. Conserva "
                        "speaker y receiver de las promesas. Si la evidencia contiene la respuesta "
                        "solicitada, dala directamente: nunca afirmes que falta informacion ni "
                        "pidas al usuario que la repita. Si revision incluye guidance, cada punto "
                        "es obligatorio en la nueva respuesta. Empieza por el hecho solicitado, sin "
                        "saludos, disculpas ni presentaciones innecesarias. Si existe revision, "
                        "reemplaza la "
                        "respuesta anterior y corrige todos los fallos indicados. "
                        "No inventes hechos ni vision. No te presentes, no menciones arquitectura "
                        "y no ofrezcas ayuda genericamente. Responde antes de preguntar. Solo "
                        "puedes preguntar si conversation_mode es ask o clarify. Habla con calidez, "
                        "calma y sencillez."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ],
            temperature=0.0,
            json_mode=False,
            deadline=deadline,
            trace_stage="text_generation",
        ).strip().strip('"')
        if not draft or draft.startswith(("{", "[")):
            raise ValueError("The expression pass did not produce natural speech.")
        return draft

    def _tool_intention_json(
        self,
        context: InternalContext,
        capability,
        decision: dict[str, object],
    ) -> str:
        payload = self._base_intention_payload(context, decision, speech_text="")
        payload["intent_type"] = "tool"
        payload["actions"] = [
            {
                "id": f"action:native:{capability.action_type}",
                "action_type": capability.action_type,
                "target": "",
                "parameters": {"argument": str(decision.get("argument") or "")},
                "requires_confirmation": capability.requires_confirmation,
                "risk_level": capability.risk_level,
                "reason": str(decision.get("response_goal") or "Required capability."),
            }
        ]
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def _speech_intention_json(
        self,
        context: InternalContext,
        speech: str,
        decision: dict[str, object],
    ) -> str:
        payload = self._base_intention_payload(context, decision, speech_text=speech)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _base_intention_payload(
        context: InternalContext,
        decision: dict[str, object],
        *,
        speech_text: str,
    ) -> dict[str, object]:
        uncertainty_values = (
            context.uncertainty.memory,
            context.uncertainty.vision,
            context.uncertainty.knowledge,
            context.uncertainty.reasoning,
            context.uncertainty.user_intent,
            context.uncertainty.action_risk,
        )
        average_uncertainty = sum(uncertainty_values) / len(uncertainty_values)
        confidence = max(0.55, min(0.9, 1.0 - average_uncertainty))
        curiosity_values = [
            float(getattr(item, "intensity", getattr(item, "importance", 0.0)))
            for item in context.curiosity
            if not bool(getattr(item, "satisfied", False))
        ]
        curiosity_signal = (
            sum(curiosity_values) / len(curiosity_values)
            if curiosity_values
            else 0.0
        )
        warmth = context.emotion.affection * 0.62 + context.emotion.interest * 0.38
        return {
            "intent_type": "conversation",
            "conversation_intent": {
                "mode": str(decision.get("mode") or "respond"),
                "target_topic": "",
                "reason": str(decision.get("response_goal") or ""),
                "curiosity_id": "",
            },
            "speech": {
                "text": speech_text,
                "emotion": context.emotion.dominant_label,
                "energy": context.emotion.energy,
                "warmth": warmth,
                "confidence": confidence,
                "thinking": context.uncertainty.reasoning,
                "curiosity": context.emotion.interest * 0.64 + curiosity_signal * 0.36,
                "pause_before": (
                    0.14
                    + context.uncertainty.reasoning * 0.22
                    + context.uncertainty.user_intent * 0.08
                ),
                "pause_after": 0.14 + context.uncertainty.reasoning * 0.08,
                "speech_rate": (
                    0.94
                    - context.uncertainty.reasoning * 0.08
                    - context.uncertainty.user_intent * 0.04
                ),
                "pitch_variation": (
                    0.11
                    + context.emotion.interest * 0.06
                    + context.emotion.energy * 0.03
                ),
                "volume": 0.82 + context.emotion.energy * 0.10 + confidence * 0.04,
                "smile": (
                    context.emotion.affection * 0.34
                    + max(0.0, context.emotion.mood) * 0.18
                ),
                "metadata": {
                    "origin": "native_reasoning",
                    "grounding_evidence_ids": list(decision.get("evidence_ids") or []),
                    "answer_source": str(decision.get("answer_source") or "none"),
                    "answer_subject": str(decision.get("answer_subject") or "unspecified"),
                    "reasoned_mode": str(decision.get("reasoned_mode") or decision.get("mode") or "respond"),
                    "response_goal": str(decision.get("response_goal") or ""),
                },
            },
            "actions": [],
            "avatar": {},
            "memory_updates": [],
            "relationship_updates": [],
            "self_updates": [],
            "curiosity_updates": [],
            "confidence": confidence,
            "reasoning_summary": str(decision.get("response_goal") or ""),
        }

    def _generate(self, prompt: str, *, context: InternalContext | None) -> str:
        deadline = time.monotonic() + max(
            5.0,
            self.brain.config.ollama_turn_timeout_seconds,
        )
        messages: list[dict[str, object]] = [
            {
                "role": "system",
                "content": (
                    "You are Anydes' sole semantic Reasoning Engine. "
                    "Return one valid JSON Intention and nothing else. "
                    "Never echo InternalContext or return context and identity fields as the root object. "
                    "The architecture is invisible to the user: speech.text must never "
                    "mention modules, prompts, context, JSON, providers or reasoning steps. "
                    "Write user-facing speech in natural Spanish."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        raw = self.brain._chat(
            self.brain.model,
            messages,
            temperature=0.18,
            json_mode=True,
            deadline=deadline,
            trace_stage="reasoning_intention_generation",
        )
        try:
            intention = _extract_json_dict(raw)
            speech = intention.get("speech")
            actions = intention.get("actions")
            if (
                context is None
                or not isinstance(speech, dict)
                or (isinstance(actions, list) and actions)
            ):
                return raw
            latest_user_text = ""
            for observation in reversed(context.observations):
                if observation.source == "user_text":
                    value = observation.data.get("text")
                    latest_user_text = str(
                        value
                        if isinstance(value, str) and value.strip()
                        else observation.summary
                    )
                    break
            speech_evidence = {
                "latest_user_message": latest_user_text,
                "relevant_memory_texts": [
                    item.content
                    for tier in (
                        context.memory.working[-4:],
                        context.memory.short[-4:],
                        context.memory.long[-4:],
                    )
                    for item in tier
                ],
                "shared_history_highlights": {
                    "threads": [
                        {"title": item.title, "summary": item.summary}
                        for item in context.relationship.threads[-4:]
                    ],
                    "open_loops": [
                        {"topic": item.topic, "motive": item.motive, "status": item.status}
                        for item in context.relationship.open_loops[-4:]
                    ],
                    "promises": [
                        {
                            "topic": item.topic,
                            "description": item.description,
                            "status": item.status,
                        }
                        for item in context.relationship.promises[-4:]
                    ],
                },
                "recent_observations": [
                    {
                        "source": item.source,
                        "summary": item.summary,
                        "data": item.data,
                    }
                    for item in context.observations[-6:]
                ],
                "identity": {
                    "traits": list(context.self_model.personality_traits),
                    "preferences": context.self_model.mutable_state.get(
                        "self_preferences",
                        {},
                    ),
                },
                "emotion": context.emotion.to_dict(),
                "memory": {
                    "working": [item.to_dict() for item in context.memory.working[-4:]],
                    "short": [item.to_dict() for item in context.memory.short[-4:]],
                    "long": [item.to_dict() for item in context.memory.long[-4:]],
                },
                "conversation_plan": intention.get("conversation_intent", {}),
                "available_actions": [
                    {
                        "action_type": item.action_type,
                        "description": item.reason,
                        "requires_confirmation": item.requires_confirmation,
                        "risk_level": item.risk_level,
                    }
                    for item in context.available_actions
                ],
            }
            draft = self.brain._chat(
                self.brain.model,
                [
                    {
                        "role": "system",
                        "content": (
                            "Write only Anydes' final user-facing speech in natural Spanish. "
                            "Return no JSON, labels, analysis or internal architecture. "
                            "Resolve the latest user message first. For a direct question, give "
                            "the actual useful answer using general knowledge when appropriate; "
                            "do not merely confirm, echo or ask another question. Factual accuracy "
                            "has priority over charm: never invent a definition, fact or example; "
                            "state uncertainty naturally when needed. Do not add a greeting to a "
                            "non-greeting. When conversation_plan.mode is respond, do not append a "
                            "follow-up question or a generic offer to help. Never reuse a recent "
                            "assistant sentence. Keep Anydes warm, calm, concise and natural. Use "
                            "contextual memory and shared relationship history only when relevant. "
                            "The latest_user_message is spoken by the user: first-person pronouns "
                            "refer to the user, never to Anydes, unless Anydes is named explicitly. "
                            "When the user asks who they are, answer from relevant user memory and "
                            "never introduce or describe Anydes instead. Do not volunteer that "
                            "Anydes is an AI or list capabilities unless explicitly asked. When "
                            "the user asks what was remembered, pending or promised, answer directly "
                            "from relevant_memory_texts and shared_history_highlights."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Grounded evidence and provisional conversational plan:\n"
                            f"{json.dumps(speech_evidence, ensure_ascii=False, separators=(',', ':'))}\n"
                            "Write the final speech now."
                        ),
                    },
                ],
                temperature=0.34,
                json_mode=False,
                deadline=deadline,
                trace_stage="text_generation_revision",
            ).strip().strip('"')
            if draft and not draft.startswith(("{", "[")):
                speech["text"] = draft
                intention["speech"] = speech
                return json.dumps(intention, ensure_ascii=False, separators=(",", ":"))
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            ValueError,
            TypeError,
            KeyError,
            json.JSONDecodeError,
        ):
            return raw
        return raw


class OpenAIConversationBrain(ConversationBrain):
    def __init__(self, model: str, config: Config) -> None:
        self.model = model
        self.config = config

    def generate(
        self,
        user_message: str,
        personality: PersonalityProfile,
        context: BrainContext,
        *,
        screen_image_base64: str | None = None,
    ) -> BrainResponse:
        capability = "external_vision" if screen_image_base64 else "conversation_ai"
        access = external_access_decision(
            self.config,
            capability,
            require_session=bool(screen_image_base64),
        )
        if not access.allowed:
            return BrainResponse(
                text="Ese servicio externo no tiene permiso para esta sesion.",
                provider=f"openai-{access.reason}",
            )
        try:
            from openai import OpenAI
        except ImportError:
            return BrainResponse(
                text="Me falta instalar `openai` para usar el cerebro IA externo.",
                provider="openai-unavailable",
            )

        client = OpenAI()
        safe_context = _external_vision_context(context) if screen_image_base64 else context
        messages = build_chat_messages(
            personality,
            safe_context,
            user_message,
            screen_image_base64=screen_image_base64,
        )
        if screen_image_base64:
            model = self.config.resolved_vision_model() or self.model
            completion = client.chat.completions.create(model=model, messages=messages)
            content = str(completion.choices[0].message.content or "").strip()
            return BrainResponse(text=content or "No recibi respuesta clara del modelo.", provider="openai-vision")

        completion = client.chat.completions.create(model=self.model, messages=messages)
        content = str(completion.choices[0].message.content or "").strip()
        relationship_updates: tuple[RelationshipMutation, ...] = ()
        if self.config.relationship_reasoning_enabled and content:
            try:
                relationship_updates = _plan_relationship_updates_with_client(
                    client,
                    model=self.model,
                    user_message=user_message,
                    assistant_response=content,
                    context=context,
                    conversation_intent=ConversationIntent(),
                )
            except Exception:
                relationship_updates = ()
        return BrainResponse(
            text=content or "No recibi respuesta clara del modelo.",
            provider="openai",
            relationship_updates=relationship_updates,
        )


class GroqConversationBrain(ConversationBrain):
    def __init__(self, model: str, base_url: str, config: Config) -> None:
        self.model = model
        self.base_url = base_url
        self.config = config

    def generate(
        self,
        user_message: str,
        personality: PersonalityProfile,
        context: BrainContext,
        *,
        screen_image_base64: str | None = None,
    ) -> BrainResponse:
        capability = "external_vision" if screen_image_base64 else "conversation_ai"
        access = external_access_decision(
            self.config,
            capability,
            require_session=bool(screen_image_base64),
        )
        if not access.allowed:
            return BrainResponse(
                text="Ese servicio externo no tiene permiso para esta sesion.",
                provider=f"groq-{access.reason}",
            )
        try:
            from openai import OpenAI
        except ImportError:
            return BrainResponse(
                text="Me falta instalar `openai` para usar Groq como cerebro externo.",
                provider="groq-unavailable",
            )

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return BrainResponse(
                text="Para usar el nuevo core con Groq falta definir GROQ_API_KEY en tu sesion.",
                provider="groq-missing-key",
            )

        client = OpenAI(base_url=self.base_url, api_key=api_key)
        safe_context = _external_vision_context(context) if screen_image_base64 else context
        messages = build_chat_messages(
            personality,
            safe_context,
            user_message,
            screen_image_base64=screen_image_base64,
        )
        if screen_image_base64:
            model = self.config.resolved_vision_model() or self.model
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.9,
                )
            except Exception as exc:
                detail = str(exc).strip()
                hint = (
                    f"Revisa ANYDES_VISION_MODEL (`{model}`) y GROQ_API_KEY. "
                    "Si el modelo fue retirado, prueba "
                    "meta-llama/llama-4-scout-17b-16e-instruct."
                )
                return BrainResponse(
                    text=f"No pude analizar tu pantalla con Groq: {detail}. {hint}",
                    provider="groq-vision-error",
                )
            content = str(completion.choices[0].message.content or "").strip()
            return BrainResponse(text=content or "No recibi respuesta clara de Groq.", provider="groq-vision")

        completion = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.9,
        )
        message = completion.choices[0].message
        content = str(message.content or "").strip()
        relationship_updates: tuple[RelationshipMutation, ...] = ()
        if self.config.relationship_reasoning_enabled and content:
            try:
                relationship_updates = _plan_relationship_updates_with_client(
                    client,
                    model=self.model,
                    user_message=user_message,
                    assistant_response=content,
                    context=context,
                    conversation_intent=ConversationIntent(),
                )
            except Exception:
                relationship_updates = ()
        return BrainResponse(
            text=content or "No recibi respuesta clara de Groq.",
            provider="groq",
            relationship_updates=relationship_updates,
        )


def build_brain(config: Config) -> ConversationBrain:
    if config.ai_provider.lower() == "ollama":
        model = config.ai_model if config.ai_model != "gpt-5-mini" else "llama3.2"
        return OllamaConversationBrain(model=model, base_url=config.ollama_url, config=config)
    if config.ai_provider.lower() == "openai":
        if not external_access_decision(config, "conversation_ai").allowed:
            return LocalConversationBrain(config)
        return OpenAIConversationBrain(model=config.ai_model, config=config)
    if config.ai_provider.lower() == "groq":
        if not external_access_decision(config, "conversation_ai").allowed:
            return LocalConversationBrain(config)
        return GroqConversationBrain(model=config.groq_model, base_url=config.groq_base_url, config=config)
    return LocalConversationBrain(config)


def build_vision_brain(config: Config) -> ConversationBrain:
    provider = config.resolved_vision_provider()
    if provider == "ollama":
        model = config.ai_model if config.ai_provider.lower() == "ollama" else "llama3.2"
        return OllamaConversationBrain(model=model, base_url=config.ollama_url, config=config)
    if provider == "openai":
        if not external_access_decision(config, "external_vision").allowed:
            return LocalConversationBrain(config)
        return OpenAIConversationBrain(model=config.ai_model, config=config)
    if provider == "groq":
        if not external_access_decision(config, "external_vision").allowed:
            return LocalConversationBrain(config)
        return GroqConversationBrain(model=config.groq_model, base_url=config.groq_base_url, config=config)
    return LocalConversationBrain(config)


def _external_vision_context(context: BrainContext) -> BrainContext:
    """A screen grant never implicitly grants memory or relationship export."""

    return replace(
        context,
        user_name="usuario local",
        speaker_context="Interlocutor local; identidad no compartida con el proveedor de vision.",
        memory_summary="Memoria local no compartida con el proveedor de vision.",
        memory_context="",
        conversation_history=(),
        identity_context="",
        internal_state_context="",
        situational_context="",
        curiosity_context=(),
        relationship_context={},
        evidence_context=(),
    )
