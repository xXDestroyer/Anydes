from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import re
from typing import Any

from anydes.cognition.action_manager import ActionManager
from anydes.funciones_extra import apps
from anydes.funciones_extra.registry import ToolRegistry, ToolResult
from anydes.nucleo.audit import AuditLog
from anydes.nucleo.permissions import PermissionManager
from anydes.realtime.capabilities import CapabilityState
from anydes.realtime.derived_insights import DerivedInsightStore
from anydes.realtime.journal import ConversationJournal, JournalQuery
from anydes.realtime.router import RealtimeAct, normalize
from anydes.realtime.session import RealtimeSessionState
from anydes.realtime.temporal import RealtimeTemporalService


_FAST_TOOLS = frozenset(
    {
        "current_time",
        "open_spotify",
        "open_app",
        "open_youtube",
        "chrome_search",
        "youtube_search",
        "screen_status",
    }
)


@dataclass(frozen=True)
class LocalRouteResult:
    handled: bool
    text: str = ""
    route: str = ""
    tool_result: ToolResult | None = None
    capability_state: CapabilityState | None = None


class RealtimeLocalRoutes:
    """Fast truthful paths whose answers depend on local state, not generation."""

    def __init__(
        self,
        tools: ToolRegistry,
        permissions: PermissionManager,
        audit: AuditLog,
        session: RealtimeSessionState,
        journal: ConversationJournal | None = None,
        temporal: RealtimeTemporalService | None = None,
        derived_insights: DerivedInsightStore | None = None,
    ) -> None:
        self.tools = tools
        self.permissions = permissions
        self.audit = audit
        self.session = session
        self.journal = journal
        self.temporal = temporal or RealtimeTemporalService()
        self.derived_insights = derived_insights
        self.action_manager = ActionManager(tools, permissions, audit)

    def resolve(
        self,
        user_text: str,
        act: RealtimeAct,
        intent: Any,
        capability: CapabilityState,
        *,
        allow_personal_memory: bool = True,
    ) -> LocalRouteResult:
        normalized = normalize(user_text)
        temporal = self.temporal.resolve(user_text)
        if temporal is not None:
            temporal_result, temporal_text = temporal
            self.audit.record(
                "realtime_temporal_query",
                **temporal_result.as_data(),
            )
            return LocalRouteResult(
                True,
                temporal_text,
                "local_temporal",
                capability_state=capability,
            )
        if act.primary == "history_query":
            text = self._resolve_history(
                act.history_target,
                user_text,
                include_personal=allow_personal_memory,
            )
            if text:
                return LocalRouteResult(True, text, "session_history", capability_state=capability)
        if act.preference_owner == "user" and act.is_question:
            value = self.session.confirmed_user_preference(act.preference_topic)
            if value:
                return LocalRouteResult(
                    True,
                    f"Es {value}, según lo que acabas de decirme.",
                    "session_user_preference",
                    capability_state=capability,
                )
        if (
            act.references_previous
            and self.session.snapshot().recently_confirmed_fact
            and re.fullmatch(
                r"(?:si|claro|exacto|ese|esa|eso|si ese|si esa|si ese mismo|"
                r"si esa misma|ese mismo|esa misma)",
                " ".join(
                    re.sub(r"[^a-z0-9 ]+", " ", normalized).split()
                ),
            )
        ):
            return LocalRouteResult(
                True,
                "Perfecto, queda así.",
                "session_reference_acknowledgement",
                capability_state=capability,
            )
        preference_ack = _user_preference_ack(user_text, act)
        if preference_ack:
            return LocalRouteResult(
                True,
                preference_ack,
                "user_preference_update",
                capability_state=capability,
            )
        if re.search(
            r"\b(?:cuanto tiempo|desde cuando)\s+.{0,35}"
            r"(?:tomando forma|desarrollandote|existiendo)\b",
            normalized,
        ):
            return LocalRouteResult(
                True,
                (
                    "Llevo cerca de tres meses tomando forma contigo. "
                    "Todavía no tenemos una fecha exacta de inicio definida."
                ),
                "self_identity_development_time",
                capability_state=capability,
            )
        if re.search(r"\b(?:cuantos anos|que edad|anos tienes|tu edad)\b", normalized):
            return LocalRouteResult(
                True,
                (
                    "Tengo 18 años de edad virtual. Y, si contamos desde que empezaste "
                    "a construirme, llevo cerca de tres meses tomando forma."
                ),
                "self_identity_age",
                capability_state=capability,
            )
        if re.search(
            r"\b(?:que eres|como describirias lo que eres|eres humana|"
            r"eres una ia|tu naturaleza)\b",
            normalized,
        ):
            return LocalRouteResult(
                True,
                (
                    "Soy Anydes, una presencia digital que sigue tomando forma contigo. "
                    "No soy humana, pero sí tengo una identidad propia dentro de este proyecto."
                ),
                "self_identity_nature",
                capability_state=capability,
            )
        if _asks_physical_history(normalized):
            return LocalRouteResult(
                True,
                (
                    "No de forma física. Puedo imaginarlo y conversar contigo sobre ello, "
                    "pero no voy a fingir que lo he comido, probado o cocinado."
                ),
                "self_physical_honesty",
                capability_state=capability,
            )
        if _asks_screen_perception(normalized):
            return self._vision_result(capability)
        capability_query = _application_capability_query(normalized)
        if capability_query:
            tool_name, argument = _application_tool(capability_query)
            if apps.app_availability(capability_query) is not False:
                self.session.register_action_proposal(
                    tool_name,
                    argument,
                    entity=capability_query,
                )
            return LocalRouteResult(
                True,
                _application_availability_text(capability_query),
                "capability_availability",
                capability_state=capability,
            )

        intent_name = str(getattr(intent, "name", "") or "")
        argument = str(getattr(intent, "argument", "") or "").strip()
        if intent_name == "screen_analyze" and (
            not normalized or _asks_screen_perception(normalized)
        ):
            return self._vision_result(capability)
        if not re.search(r"[a-z0-9]", normalized):
            return LocalRouteResult(
                True,
                "Estoy aquí.",
                "minimal_presence",
                capability_state=capability,
            )
        if intent_name != "tool":
            if _asks_local_time(normalized):
                result = _tool_result_from_action(
                    self.action_manager.execute_explicit_request(
                        "current_time",
                        "",
                    )
                )
                return LocalRouteResult(
                    True,
                    _tool_result_text("current_time", "", result, user_text=user_text),
                    "local_tool:current_time",
                    tool_result=result,
                    capability_state=capability.with_action_result(result),
                )
            return LocalRouteResult(False, capability_state=capability)

        tool_name, tool_argument = _split_tool_argument(argument)
        if tool_name not in _FAST_TOOLS:
            return LocalRouteResult(False, capability_state=capability)
        result = _tool_result_from_action(
            self.action_manager.execute_explicit_request(
                tool_name,
                tool_argument,
            )
        )
        state = capability.with_action_result(result)
        return LocalRouteResult(
            True,
            _tool_result_text(tool_name, tool_argument, result, user_text=user_text),
            f"local_tool:{tool_name}",
            tool_result=result,
            capability_state=state,
        )

    def _resolve_history(
        self,
        target: str,
        user_text: str,
        *,
        include_personal: bool,
    ) -> str | None:
        if self.journal is None:
            return self.session.resolve_history(target)
        if target == "first_user_message":
            event = self.journal.first_user_message_of_session(
                self.session.session_id,
                include_personal=include_personal,
            )
            return (
                f'Tu primer mensaje de esta sesión fue: "{event.exact_text}".'
                if event
                else "Este es tu primer mensaje de la sesión."
            )
        if target == "previous_session":
            events = self.journal.previous_session_events(
                self.session.session_id,
                limit=8,
                include_personal=include_personal,
            )
            if not events:
                return "No encuentro una conversación anterior en el historial autorizado."
            return _render_previous_session(events)
        if target == "first_user_question":
            events = self.journal.query(
                JournalQuery(
                    kind="questions",
                    session_id=self.session.session_id,
                    author="user",
                    limit=1_000,
                    include_personal=include_personal,
                )
            )
            return (
                f'Tu primera pregunta de esta sesión fue: "{events[0].exact_text}".'
                if events
                else "Todavía no habías hecho una pregunta en esta sesión."
            )
        if target == "first_user_message_day":
            event = self.journal.first_user_message_of_day(
                datetime.now().astimezone().date(),
                include_personal=include_personal,
            )
            return (
                f'Tu primer mensaje de hoy fue: "{event.exact_text}".'
                if event
                else "Todavía no encuentro un mensaje tuyo de hoy."
            )
        if target == "last_user_message":
            events = self.journal.query(
                JournalQuery(
                    kind="last",
                    session_id=self.session.session_id,
                    author="user",
                    include_personal=include_personal,
                )
            )
            return (
                f'Tu mensaje anterior fue: "{events[-1].exact_text}".'
                if events
                else "Este es tu primer mensaje de la sesión."
            )
        if target == "messages_on_date":
            requested = _requested_local_date(user_text)
            if requested is None:
                return "No pude reconocer con seguridad la fecha que mencionaste."
            events = self.journal.query(
                JournalQuery(
                    kind="recent",
                    local_date=requested.isoformat(),
                    author="user",
                    limit=6,
                    include_personal=include_personal,
                )
            )
            if not events:
                return f"No encuentro mensajes tuyos del {requested.isoformat()}."
            rendered = "; ".join(f'"{event.exact_text}"' for event in events)
            return f"Tus mensajes de esa fecha fueron: {rendered}."
        if target in {"corrections", "preferences", "commitments"}:
            events = self.journal.query(
                JournalQuery(
                    kind=target,
                    author="user",
                    limit=6,
                    include_personal=include_personal,
                )
            )
            if not events:
                return "No encuentro registros de ese tipo en el historial autorizado."
            rendered = "; ".join(f'"{event.exact_text}"' for event in events)
            return f"Encontré estos registros: {rendered}."
        if target == "user_topic":
            topic = _requested_history_topic(user_text)
            broad_profile = _asks_broad_user_profile(user_text)
            events = self.journal.query(
                JournalQuery(
                    kind="user_facts",
                    author="user",
                    topic="" if broad_profile else topic,
                    limit=6,
                    include_personal=include_personal,
                )
            )
            if not events:
                if topic:
                    return (
                        f"No encuentro algo que me hayas dicho sobre {topic} "
                        "en el historial autorizado."
                    )
                return "Todavía no encuentro datos tuyos relevantes en el historial autorizado."
            return _render_user_evidence(events, topic=topic)
        if target == "derived_insights":
            if self.derived_insights is None:
                return (
                    "Todavía no tengo inferencias provisionales guardadas sobre ti. "
                    "Prefiero no inventarlas."
                )
            insights = self.derived_insights.query(status="provisional", limit=5)
            if not insights:
                return (
                    "Todavía no tengo inferencias provisionales respaldadas sobre ti. "
                    "Prefiero quedarme solo con lo que sí me has contado."
                )
            rendered = "; ".join(
                f"{item.statement} (confianza {item.confidence:.0%})"
                for item in insights
            )
            return f"Por ahora tengo estas ideas provisionales: {rendered}."
        if target == "repeated_question":
            events = self.journal.query(
                JournalQuery(
                    kind="repeated_questions",
                    session_id=self.session.session_id,
                    author="user",
                    limit=8,
                    include_personal=include_personal,
                )
            )
            if not events:
                return "No encuentro una pregunta repetida en esta sesión."
            return f'La pregunta repetida más reciente fue: "{events[-1].exact_text}".'
        return self.session.resolve_history(target)

    @staticmethod
    def _vision_result(capability: CapabilityState) -> LocalRouteResult:
        if not capability.vision_enabled:
            text = (
                "Ahora mismo no tengo la visión de pantalla activada, "
                "así que no puedo ver lo que tienes abierto."
            )
        elif not capability.screen_capture_authorized:
            text = (
                "La visión está disponible, pero esta sesión no tiene permiso "
                "para capturar tu pantalla."
            )
        elif not capability.frame_available:
            text = "La visión está habilitada, pero no tengo una captura reciente de tu pantalla."
        elif not capability.frame_fresh:
            text = "La última captura ya es antigua, así que no sería honesto decirte qué veo ahora."
        elif not capability.visual_observation_fresh:
            text = (
                "Tengo una captura reciente, pero todavía no una descripción "
                "visual fiable de ella."
            )
        else:
            return LocalRouteResult(False, capability_state=capability)
        return LocalRouteResult(True, text, "screen_capability", capability_state=capability)


def _split_tool_argument(argument: str) -> tuple[str, str]:
    if not argument:
        return "", ""
    pieces = argument.split(maxsplit=1)
    return pieces[0], pieces[1] if len(pieces) > 1 else ""


def _tool_result_text(
    tool_name: str,
    tool_argument: str,
    result: ToolResult,
    *,
    user_text: str,
) -> str:
    if tool_name == "current_time" and result.status == "success":
        return _format_time_result(result.data, user_text=user_text)
    if tool_name == "screen_status" and result.status == "success":
        if bool(result.data.get("watching")):
            return "Sí, la observación de pantalla está activa en esta sesión."
        if bool(result.data.get("has_recent_capture")):
            return "No estoy observando continuamente, aunque existe una captura reciente."
        return "No, ahora mismo no estoy observando tu pantalla."
    label = _tool_label(tool_name, tool_argument)
    if result.status == "success" and bool(result.data.get("opened", True)):
        return f"Listo, abrí {label}."
    if result.status == "denied":
        return f"Puedo intentar abrir {label}, pero esta vez no tuve autorización."
    if result.status == "missing":
        return f"No tengo disponible la herramienta para abrir {label}."
    return f"Intenté abrir {label}, pero no pude."


def _tool_label(tool_name: str, argument: str) -> str:
    if tool_name == "open_spotify":
        return "Spotify"
    if tool_name == "open_youtube":
        return "YouTube"
    if tool_name == "open_app" and argument:
        return argument.strip()
    if tool_name == "chrome_search":
        return "la búsqueda en Chrome"
    if tool_name == "youtube_search":
        return "la búsqueda en YouTube"
    if tool_name == "screen_status":
        return "la visión de pantalla"
    return "esa aplicación"


def _format_time_result(data: dict[str, object], *, user_text: str) -> str:
    normalized = normalize(user_text)
    if re.search(r"\b(?:momento|parte|periodo)\s+del\s+dia\b", normalized):
        raw = str(data.get("current_time") or "")
        try:
            hour = datetime.fromisoformat(raw).hour
        except ValueError:
            hour = int(data.get("hour") or datetime.now().hour)
        if hour < 6:
            period = "madrugada"
        elif hour < 12:
            period = "mañana"
        elif hour < 19:
            period = "tarde"
        else:
            period = "noche"
        return f"Ahora estamos en la {period}."
    if any(marker in normalized for marker in ("fecha", "qué día", "que dia", "día es", "dia es")):
        raw_date = str(data.get("date") or "")
        try:
            current_date = datetime.fromisoformat(raw_date)
        except ValueError:
            return f"Hoy es {raw_date}." if raw_date else "No pude leer la fecha local."
        weekdays = (
            "lunes",
            "martes",
            "miércoles",
            "jueves",
            "viernes",
            "sábado",
            "domingo",
        )
        months = (
            "enero",
            "febrero",
            "marzo",
            "abril",
            "mayo",
            "junio",
            "julio",
            "agosto",
            "septiembre",
            "octubre",
            "noviembre",
            "diciembre",
        )
        return (
            f"Hoy es {weekdays[current_date.weekday()]} "
            f"{current_date.day} de {months[current_date.month - 1]} "
            f"de {current_date.year}."
        )
    raw = str(data.get("current_time") or "")
    try:
        moment = datetime.fromisoformat(raw)
    except ValueError:
        hour = int(data.get("hour") or 0)
        minute = int(data.get("minute") or 0)
        return f"Ahora mismo son las {_clock(hour, minute)}."
    return f"Ahora mismo son las {_clock(moment.hour, moment.minute)}"


def _clock(hour: int, minute: int) -> str:
    suffix = "a. m." if hour < 12 else "p. m."
    rendered_hour = hour % 12 or 12
    return f"{rendered_hour}:{minute:02d} {suffix}"


def _tool_result_from_action(action_result) -> ToolResult:
    output = action_result.output
    return ToolResult(
        tool_name=str(output.get("tool_name") or ""),
        status=str(output.get("tool_status") or "failed"),
        data=dict(output.get("data") or {}),
        error=action_result.error,
    )


def _asks_screen_perception(normalized: str) -> bool:
    return "pantalla" in normalized and bool(
        re.search(
            r"\b(?:ves|viendo|mirar|mira|observas|describe|"
            r"puedes\s+ver|puedes\s+mirar|tienes\s+acceso)\b",
            normalized,
        )
    )


def _asks_local_time(normalized: str) -> bool:
    return bool(
        re.search(
            r"\b(?:que hora|hora es|fecha actual|que dia|dia es|fecha de hoy|"
            r"momento del dia|parte del dia|periodo del dia)\b",
            normalized,
        )
        or re.search(
            r"\b(?:que fecha|fecha es)\b",
            normalized,
        )
    )


def _application_capability_query(normalized: str) -> str:
    match = re.search(
        r"\b(?:puedes|podrias|sabes)\s+abrir\s+(spotify|youtube|chrome|"
        r"google chrome|calculadora|bloc de notas|vscode|whatsapp)\b",
        normalized,
    )
    return match.group(1) if match else ""


def _application_availability_text(app_name: str) -> str:
    available = apps.app_availability(app_name)
    label = app_name.title() if app_name != "vscode" else "Visual Studio Code"
    if available is True:
        return f"Sí, puedo intentar abrir {label} cuando me lo pidas."
    if available is False:
        return f"No encuentro {label} disponible en este equipo."
    return (
        f"Tengo una ruta local para intentar abrir {label}, "
        "pero solo podré confirmarlo al ejecutarla."
    )


def _application_tool(app_name: str) -> tuple[str, str]:
    normalized = normalize(app_name)
    if normalized == "spotify":
        return "open_spotify", ""
    if normalized == "youtube":
        return "open_youtube", ""
    return "open_app", app_name


def _requested_local_date(text: str) -> date | None:
    normalized = normalize(text)
    today = datetime.now().astimezone().date()
    if re.search(r"\bhoy\b", normalized):
        return today
    if re.search(r"\bayer\b", normalized):
        return today - timedelta(days=1)
    match = re.search(r"\b(\d{1,4})[-/](\d{1,2})[-/](\d{1,4})\b", normalized)
    if match is None:
        return None
    first, middle, last = (int(item) for item in match.groups())
    try:
        if first > 31:
            return date(first, middle, last)
        return date(last, middle, first)
    except ValueError:
        return None


def _requested_history_topic(text: str) -> str:
    normalized = normalize(text).strip(" .,!?:;")
    patterns = (
        r"\b(?:que\s+(?:te\s+)?dije|que\s+sabes|recuerdas?)\s+"
        r"(?:sobre|acerca\s+de|de)\s+(.+)$",
        r"\bque\s+sabes\s+de\s+mis?\s+(.+)$",
        r"\b(?:mis?\s+)?(?:hobbies|aficiones|intereses|gustos)\s+(.+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match is None:
            continue
        topic = re.sub(
            r"\b(?:hoy|ayer|en\s+esta\s+sesion|en\s+el\s+historial)\b.*$",
            "",
            match.group(1),
        ).strip(" .,!?:;")
        return topic[:100]
    broad = re.search(r"\b(hobbies|aficiones|intereses|gustos)\b", normalized)
    return broad.group(1) if broad else ""


def _asks_broad_user_profile(text: str) -> bool:
    normalized = normalize(text)
    return bool(
        re.search(
            r"\bque\s+sabes\s+de\s+mi\b|"
            r"\b(?:mis\s+)?(?:hobbies|aficiones|intereses|gustos)\b",
            normalized,
        )
    )


def _render_previous_session(events) -> str:
    user_events = [event for event in events if event.author == "user"]
    assistant_events = [event for event in events if event.author == "anydes"]
    if not user_events:
        return "Encontré la sesión anterior, pero no un mensaje tuyo que pueda mostrar."
    first = user_events[0]
    last = user_events[-1]
    if first.event_id == last.event_id:
        detail = f'Tú dijiste: "{first.exact_text}".'
    else:
        detail = (
            f'Empezaste diciendo: "{first.exact_text}". '
            f'Y tu último mensaje fue: "{last.exact_text}".'
        )
    if assistant_events:
        return f"En nuestra conversación anterior, {detail}"
    return f"En tu sesión anterior, {detail}"


def _render_user_evidence(events, *, topic: str) -> str:
    records = "; ".join(f'"{event.exact_text}"' for event in events)
    if topic:
        return f"Sobre {topic}, esto es lo que me has dicho: {records}."
    return f"Esto es lo que recuerdo porque tú me lo contaste: {records}."


def _user_preference_ack(user_text: str, act: RealtimeAct) -> str:
    if act.primary not in {"user_preference", "correction"}:
        return ""
    match = re.search(
        r"\bmi\s+(.{1,45}?)\s+favorit([oa])\s+(?:ahora\s+)?es\s+([^,.!?;]{1,70})",
        user_text,
        flags=re.IGNORECASE,
    )
    if match is None:
        return ""
    topic = match.group(1).strip()
    ending = match.group(2).casefold()
    value = match.group(3).strip()
    adjective = "favorita" if ending == "a" else "favorito"
    prefix = "Entendido: ahora" if act.correction else "Vale, me quedo con que"
    return f"{prefix} tu {topic} {adjective} es {value}."


def _asks_physical_history(normalized: str) -> bool:
    return bool(
        re.search(
            r"\b(?:has|alguna vez has)\s+(?:comido|cocinado|probado|"
            r"olido|tocado|viajado|caminado)\b",
            normalized,
        )
    )
