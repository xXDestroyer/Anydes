from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Literal
from urllib.parse import urlparse

from anydes.cognition.action_manager import ActionManager
from anydes.funciones_extra import apps, calendar_tools
from anydes.funciones_extra.registry import ToolRegistry, ToolResult
from anydes.nucleo.audit import AuditLog
from anydes.nucleo.config import Config
from anydes.nucleo.consent import external_access_decision
from anydes.nucleo.permissions import PermissionManager


CapabilityAvailability = Literal["available", "degraded", "unavailable"]
ActionMode = Literal["inquire", "execute", "confirm"]


@dataclass(frozen=True)
class CapabilityDefinition:
    name: str
    tool_name: str
    kind: str
    requires_confirmation: bool = False
    asynchronous: bool = False
    aliases: tuple[str, ...] = ()
    fallback: str = ""


@dataclass(frozen=True)
class CapabilityResolution:
    capability: str
    availability: CapabilityAvailability
    tool_name: str = ""
    reason: str = ""
    requires_confirmation: bool = False
    asynchronous: bool = False
    fallback: str = ""
    local_executable: bool | None = None
    resolved_entity: str = ""
    alternatives: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticAction:
    capability: str
    mode: ActionMode = "execute"
    entity: str = ""
    parameters: dict[str, object] = field(default_factory=dict)
    origin_text: str = ""


@dataclass(frozen=True)
class ExecutionReceipt:
    capability: str
    status: str
    verified: bool
    data: dict[str, object] = field(default_factory=dict)
    error: str = ""
    tool_result: ToolResult | None = None

    def safe_dict(self) -> dict[str, object]:
        return {
            "capability": self.capability,
            "status": self.status,
            "verified": self.verified,
            "data": dict(self.data),
            "error": self.error,
        }

    def visible_text(self) -> str:
        if self.capability == "weather_current":
            return _weather_text(self)
        if self.capability == "calendar_create":
            return _calendar_text(self)
        if self.capability in {"spotify_search", "spotify_web_search"}:
            return _spotify_search_text(self)
        if self.capability == "open_browser_tab":
            if self.status != "success":
                return _failure_text(self, "una pestaña nueva")
            return (
                "Listo, abrí una pestaña nueva en el navegador."
                if self.verified
                else "Le pedí al navegador que abriera una pestaña nueva."
            )
        if self.capability == "open_web_service":
            label = str(self.data.get("service") or "ese servicio")
            if self.status != "success":
                return _failure_text(self, label)
            return (
                f"Listo, abrí {label} en el navegador."
                if self.verified
                else f"Le pedí al navegador que abriera {label}; no puedo confirmar que ya cargó."
            )
        if self.capability == "spotify_open":
            if self.status == "success":
                return (
                    "Spotify recibió la orden de abrirse."
                    if not self.verified
                    else "Listo, Spotify está abierto."
                )
            return _failure_text(self, "Spotify")
        if self.capability in {"web_search", "youtube_search"}:
            label = "la búsqueda"
            return (
                f"Listo, abrí {label} en el navegador."
                if self.status == "success" and self.verified
                else "Abrí la búsqueda en el navegador, pero no puedo confirmar que ya terminó de cargar."
                if self.status == "success"
                else _failure_text(self, label)
            )
        if self.capability in {"open_url", "youtube_open"}:
            label = "esa página" if self.capability == "open_url" else "YouTube"
            return (
                f"Listo, abrí {label}."
                if self.status == "success" and self.verified
                else f"Le pedí al navegador que abriera {label}; todavía no puedo confirmar la carga."
                if self.status == "success"
                else _failure_text(self, label)
            )
        if self.capability == "open_application":
            label = str(self.data.get("app") or "esa aplicación")
            if self.status == "success":
                return (
                    f"Listo, abrí {label}."
                    if self.verified
                    else f"Envié a Windows la orden de abrir {label}."
                )
            return _failure_text(self, label)
        return (
            "La acción terminó correctamente."
            if self.status == "success"
            else "No pude completar esa acción."
        )


class CapabilityRegistry:
    """Resolves semantic capabilities without allowing them to execute themselves."""

    def __init__(self, config: Config, tools: ToolRegistry) -> None:
        self.config = config
        self.tools = tools
        definitions = (
            CapabilityDefinition(
                "open_application",
                "open_app",
                "application",
                requires_confirmation=True,
                aliases=("app", "application"),
            ),
            CapabilityDefinition(
                "open_browser_tab",
                "open_url",
                "browser",
                requires_confirmation=True,
                aliases=("new_browser_tab",),
            ),
            CapabilityDefinition(
                "open_url",
                "open_url",
                "browser",
                requires_confirmation=True,
                aliases=("browser_url",),
            ),
            CapabilityDefinition(
                "open_web_service",
                "open_url",
                "browser",
                requires_confirmation=True,
                aliases=("web_service",),
            ),
            CapabilityDefinition(
                "web_search",
                "chrome_search",
                "browser",
                requires_confirmation=True,
                aliases=("search_web", "google_search"),
            ),
            CapabilityDefinition(
                "youtube_open",
                "open_youtube",
                "media",
                requires_confirmation=True,
            ),
            CapabilityDefinition(
                "youtube_search",
                "youtube_search",
                "media",
                requires_confirmation=True,
            ),
            CapabilityDefinition(
                "spotify_open",
                "open_spotify",
                "media",
                requires_confirmation=True,
                fallback="spotify_web_search",
            ),
            CapabilityDefinition(
                "spotify_search",
                "spotify_search",
                "media",
                requires_confirmation=True,
                fallback="spotify_web_search",
            ),
            CapabilityDefinition(
                "spotify_web_search",
                "spotify_web_search",
                "browser",
                requires_confirmation=True,
            ),
            CapabilityDefinition(
                "weather_current",
                "weather_current",
                "external_information",
                asynchronous=True,
                aliases=("weather",),
            ),
            CapabilityDefinition(
                "calendar_create",
                "calendar_create",
                "calendar",
                requires_confirmation=True,
                asynchronous=True,
                aliases=("create_calendar_event",),
            ),
        )
        self._definitions = {definition.name: definition for definition in definitions}
        self._aliases = {
            alias: definition.name
            for definition in definitions
            for alias in definition.aliases
        }

    def definition(self, name: str) -> CapabilityDefinition | None:
        canonical = self._aliases.get(name, name)
        return self._definitions.get(canonical)

    def resolve(self, action: SemanticAction) -> CapabilityResolution:
        definition = self.definition(action.capability)
        if definition is None:
            return CapabilityResolution(
                capability=action.capability,
                availability="unavailable",
                reason="unknown_capability",
            )
        known_tools = {tool.name for tool in self.tools.list_tools()}
        if definition.tool_name not in known_tools:
            return CapabilityResolution(
                capability=definition.name,
                availability="unavailable",
                reason="tool_not_registered",
                fallback=definition.fallback,
            )
        if definition.name == "weather_current":
            access = external_access_decision(self.config, "weather")
            if not access.allowed:
                return CapabilityResolution(
                    capability=definition.name,
                    availability="unavailable",
                    tool_name=definition.tool_name,
                    reason=access.reason,
                    asynchronous=True,
                )
        if definition.name == "calendar_create":
            if calendar_tools.outlook_calendar_available():
                return CapabilityResolution(
                    capability=definition.name,
                    availability="available",
                    tool_name=definition.tool_name,
                    reason="outlook_connected",
                    requires_confirmation=True,
                    asynchronous=True,
                    local_executable=True,
                )
            if calendar_tools.microsoft365_web_available():
                return CapabilityResolution(
                    capability=definition.name,
                    availability="degraded",
                    tool_name=definition.tool_name,
                    reason="microsoft365_web_requires_user_save",
                    requires_confirmation=True,
                    asynchronous=False,
                    local_executable=False,
                )
            return CapabilityResolution(
                capability=definition.name,
                availability="unavailable",
                tool_name=definition.tool_name,
                reason="calendar_integration_unavailable",
                requires_confirmation=True,
                asynchronous=False,
            )
        if definition.name in {"spotify_open", "spotify_search"}:
            installed = apps.app_availability("spotify")
            if installed is False:
                return CapabilityResolution(
                    capability=definition.name,
                    availability="unavailable",
                    tool_name=definition.tool_name,
                    reason="spotify_not_installed",
                    requires_confirmation=True,
                    fallback=definition.fallback,
                    local_executable=False,
                )
            availability: CapabilityAvailability = (
                "available" if installed is True else "degraded"
            )
            return CapabilityResolution(
                capability=definition.name,
                availability=availability,
                tool_name=definition.tool_name,
                reason="installed" if installed is True else "installation_unverified",
                requires_confirmation=True,
                fallback=definition.fallback,
                local_executable=installed,
            )
        if definition.name == "open_application":
            app_name = str(action.parameters.get("name") or action.entity)
            app_resolution = apps.resolve_application(app_name)
            if app_resolution.status == "ambiguous":
                return CapabilityResolution(
                    capability=definition.name,
                    availability="unavailable",
                    tool_name=definition.tool_name,
                    reason="application_ambiguous",
                    requires_confirmation=True,
                    alternatives=tuple(
                        item.display_name for item in app_resolution.alternatives
                    ),
                )
            if app_resolution.status != "resolved":
                web_url = _web_service_url(app_name)
                return CapabilityResolution(
                    capability=definition.name,
                    availability="unavailable",
                    tool_name=definition.tool_name,
                    reason="application_not_available",
                    requires_confirmation=True,
                    fallback="open_web_service" if web_url else "",
                    local_executable=False,
                )
            return CapabilityResolution(
                capability=definition.name,
                availability="available",
                tool_name=definition.tool_name,
                reason="installed",
                requires_confirmation=True,
                local_executable=True,
                resolved_entity=app_resolution.candidate.display_name,
            )
        if definition.name == "open_web_service":
            service = str(action.parameters.get("service") or action.entity)
            raw_url = str(
                action.parameters.get("url")
                or action.parameters.get("web_url")
                or _web_service_url(service)
            )
            if not _valid_http_target(raw_url):
                return CapabilityResolution(
                    capability=definition.name,
                    availability="unavailable",
                    tool_name=definition.tool_name,
                    reason="unknown_web_service",
                    requires_confirmation=True,
                )
            return CapabilityResolution(
                capability=definition.name,
                availability="available",
                tool_name=definition.tool_name,
                reason="web_service_known",
                requires_confirmation=True,
                resolved_entity=service,
            )
        if definition.name == "open_url":
            raw_url = str(action.parameters.get("url") or "")
            if action.mode == "inquire" and not raw_url:
                return CapabilityResolution(
                    capability=definition.name,
                    availability="available",
                    tool_name=definition.tool_name,
                    reason="target_required",
                    requires_confirmation=True,
                )
            if not _valid_http_target(raw_url):
                return CapabilityResolution(
                    capability=definition.name,
                    availability="unavailable",
                    tool_name=definition.tool_name,
                    reason="invalid_url",
                    requires_confirmation=True,
                )
        return CapabilityResolution(
            capability=definition.name,
            availability="available",
            tool_name=definition.tool_name,
            reason="available",
            requires_confirmation=definition.requires_confirmation,
            asynchronous=definition.asynchronous,
            fallback=definition.fallback,
        )


class ActionExecutor:
    """Executes a resolved semantic action and returns a verifiable receipt."""

    def __init__(
        self,
        registry: CapabilityRegistry,
        tools: ToolRegistry,
        permissions: PermissionManager,
        audit: AuditLog,
    ) -> None:
        self.registry = registry
        self.action_manager = ActionManager(tools, permissions, audit)

    def execute(self, action: SemanticAction) -> ExecutionReceipt:
        resolution = self.registry.resolve(action)
        if resolution.availability == "unavailable":
            return ExecutionReceipt(
                capability=action.capability,
                status="unavailable",
                verified=False,
                error=resolution.reason,
            )
        argument = self._argument(action, resolution)
        action_result = self.action_manager.execute_explicit_request(
            resolution.tool_name,
            argument,
        )
        output = dict(action_result.output or {})
        data = dict(output.get("data") or {})
        if action.capability == "open_web_service":
            data.setdefault(
                "service",
                str(action.parameters.get("service") or action.entity),
            )
        tool_status = str(output.get("tool_status") or "failed")
        tool_result = ToolResult(
            tool_name=resolution.tool_name,
            status=tool_status if tool_status in {"success", "denied", "failed", "missing"} else "failed",
            data=data,
            error=action_result.error,
        )
        if (
            action.capability == "calendar_create"
            and tool_status == "success"
            and data.get("prepared")
            and data.get("requires_user_save")
        ):
            status = "awaiting_user"
        else:
            status = (
                "success"
                if tool_status == "success" and _result_succeeded(action.capability, data)
                else "denied" if tool_status == "denied"
                else "unavailable" if tool_status == "missing"
                else "failed"
            )
        return ExecutionReceipt(
            capability=action.capability,
            status=status,
            verified=_result_verified(action.capability, data),
            data=data,
            error=str(action_result.error or data.get("error") or ""),
            tool_result=tool_result,
        )

    @staticmethod
    def _argument(
        action: SemanticAction,
        resolution: CapabilityResolution,
    ) -> str:
        _ = resolution
        if action.capability == "open_application":
            return str(action.parameters.get("name") or action.entity)
        if action.capability in {"open_url", "open_browser_tab"}:
            return str(action.parameters.get("url") or "")
        if action.capability == "open_web_service":
            service = str(action.parameters.get("service") or action.entity)
            return str(
                action.parameters.get("url")
                or action.parameters.get("web_url")
                or _web_service_url(service)
            )
        if action.capability in {
            "web_search",
            "youtube_search",
            "spotify_search",
            "spotify_web_search",
        }:
            return str(action.parameters.get("query") or action.entity)
        if action.capability == "calendar_create":
            return json.dumps(action.parameters, ensure_ascii=False, sort_keys=True)
        return ""


class RealtimeActionRouter:
    """Extracts explicit capability requests; it never invents an action."""

    def match(self, text: str, existing_intent=None) -> SemanticAction | None:
        normalized = _normalize(text)
        if _is_weather_request(normalized):
            return SemanticAction(
                capability="weather_current",
                entity="clima actual",
                origin_text=text,
            )
        if _is_calendar_request(normalized):
            inquiry = bool(
                re.search(r"\b(?:puedes|podrias|sabes)\b", normalized)
                and text.rstrip().endswith("?")
            )
            if inquiry:
                return SemanticAction(
                    capability="calendar_create",
                    mode="inquire",
                    entity="calendario",
                    origin_text=text,
                )
            draft = CalendarDraftParser().parse(text)
            return SemanticAction(
                capability="calendar_create",
                entity=draft.title,
                parameters=draft.as_parameters(),
                origin_text=text,
            )
        url = _extract_url(text)
        if url and re.search(r"\b(?:abre|abrir|pestana|navegador|ve a)\b", normalized):
            return SemanticAction(
                capability="open_url",
                entity=url,
                parameters={"url": url},
                origin_text=text,
            )
        web_service = _web_service_request(text, normalized)
        if web_service is not None:
            return web_service
        spotify = _media_query(text, normalized, service="spotify")
        if spotify is not None:
            return spotify
        youtube = _media_query(text, normalized, service="youtube")
        if youtube is not None:
            return youtube
        web_query = _web_query(text, normalized)
        if web_query:
            return SemanticAction(
                capability="web_search",
                entity=web_query,
                parameters={"query": web_query},
                origin_text=text,
            )
        if re.search(r"\bpesta[nñ]a\b", normalized) and re.search(
            r"\bnavegador\b", normalized
        ):
            return SemanticAction(
                capability="open_browser_tab",
                mode="inquire",
                entity="pestaña del navegador",
                parameters={"url": ""},
                origin_text=text,
            )
        application = _application_request(text, normalized)
        if application is not None:
            return application
        intent_name = str(getattr(existing_intent, "name", "") or "")
        argument = str(getattr(existing_intent, "argument", "") or "").strip()
        if intent_name != "tool" or not argument:
            return None
        tool_name, _, tool_argument = argument.partition(" ")
        mapping = {
            "open_spotify": "spotify_open",
            "open_youtube": "youtube_open",
            "youtube_search": "youtube_search",
            "chrome_search": "web_search",
            "open_url": "open_url",
            "open_browser_tab": "open_browser_tab",
            "open_app": "open_application",
        }
        capability = mapping.get(tool_name)
        if capability is None:
            return None
        inquiry = bool(
            re.search(r"\b(?:puedes|podrias|sabes)\s+abrir\b", normalized)
        )
        parameters: dict[str, object] = {}
        if capability == "open_application":
            parameters["name"] = tool_argument
        elif capability in {"open_url", "open_browser_tab"}:
            parameters["url"] = tool_argument
        elif capability in {"web_search", "youtube_search"}:
            parameters["query"] = tool_argument
        return SemanticAction(
            capability=capability,
            mode="inquire" if inquiry else "execute",
            entity=tool_argument or capability.replace("_", " "),
            parameters=parameters,
            origin_text=text,
        )

    def complete_pending(self, text: str, pending_action) -> SemanticAction | None:
        capability = str(getattr(pending_action, "capability", "") or "")
        parameters = dict(getattr(pending_action, "parameters", {}) or {})
        if capability == "open_application" and parameters.get("alternatives"):
            requested = _normalize(text)
            matches = [
                str(item)
                for item in parameters.get("alternatives", ())
                if requested
                and (
                    requested == _normalize(str(item))
                    or requested in _normalize(str(item))
                )
            ]
            if len(matches) != 1:
                return None
            return SemanticAction(
                capability="open_application",
                mode="execute",
                entity=matches[0],
                parameters={"name": matches[0]},
                origin_text=text,
            )
        if capability in {"open_url", "open_browser_tab"} and not parameters.get("url"):
            target = _extract_url(text) or _domain_target(text)
            if not target:
                return None
            return SemanticAction(
                capability="open_url",
                mode="execute",
                entity=target,
                parameters={"url": target},
                origin_text=text,
            )
        if capability == "calendar_create":
            updated = CalendarDraftParser().merge_update(text, parameters)
            if updated is not None:
                return SemanticAction(
                    capability="calendar_create",
                    mode="execute",
                    entity=str(updated.get("title") or "Actividad"),
                    parameters=updated,
                    origin_text=text,
                )
        if capability == "calendar_create" and parameters.get("missing_fields"):
            parsed = CalendarDraftParser().parse(text).as_parameters()
            merged = dict(parameters)
            for key in (
                "title",
                "activity",
                "subject",
                "start_iso",
                "end_iso",
                "timezone",
                "participants",
                "location",
                "notes",
            ):
                value = parsed.get(key)
                if value and not (
                    key == "title"
                    and value == "Actividad"
                    and merged.get("title")
                ):
                    merged[key] = value
            missing = [
                key
                for key, value_key in (
                    ("title", "title"),
                    ("date", "start_iso"),
                    ("time", "start_iso"),
                )
                if not merged.get(value_key)
            ]
            merged["missing_fields"] = list(dict.fromkeys(missing))
            merged["draft_version"] = int(parameters.get("draft_version") or 1) + 1
            merged["changed_fields"] = [
                key for key in merged if merged.get(key) != parameters.get(key)
            ]
            return SemanticAction(
                capability="calendar_create",
                mode="execute",
                entity=str(merged.get("title") or "Actividad"),
                parameters=merged,
                origin_text=text,
            )
        return None


@dataclass(frozen=True)
class CalendarDraft:
    title: str
    start: datetime | None
    end: datetime | None
    timezone: str
    activity: str = ""
    subject: str = ""
    participants: tuple[str, ...] = ()
    location: str = ""
    notes: str = ""
    missing_fields: tuple[str, ...] = ()

    def as_parameters(self) -> dict[str, object]:
        return {
            "title": self.title,
            "activity": self.activity,
            "subject": self.subject,
            "start_iso": self.start.isoformat() if self.start else "",
            "end_iso": self.end.isoformat() if self.end else "",
            "timezone": self.timezone,
            "participants": list(self.participants),
            "location": self.location,
            "notes": self.notes,
            "missing_fields": list(self.missing_fields),
        }

    def summary(self) -> str:
        if self.start is None:
            return self.title
        return (
            f"{self.title}, el {self.start.strftime('%d/%m/%Y')} "
            f"a las {self.start.strftime('%H:%M')}"
        )


class CalendarDraftParser:
    _weekdays = {
        "lunes": 0,
        "martes": 1,
        "miercoles": 2,
        "jueves": 3,
        "viernes": 4,
        "sabado": 5,
        "domingo": 6,
    }

    def parse(
        self,
        text: str,
        *,
        now: datetime | None = None,
    ) -> CalendarDraft:
        current = now or datetime.now().astimezone()
        normalized = _normalize(text)
        event_date = self._date(normalized, current)
        event_time = self._time(normalized)
        missing: list[str] = []
        if event_date is None:
            missing.append("date")
        if event_time is None:
            missing.append("time")
        start = (
            datetime.combine(event_date, event_time, tzinfo=current.tzinfo)
            if event_date is not None and event_time is not None
            else None
        )
        duration = self._duration(normalized)
        end = start + duration if start is not None else None
        participants = self._participants(text)
        activity, subject, title = self._title_parts(text, participants)
        if not title:
            title = "Actividad"
            missing.append("title")
        return CalendarDraft(
            title=title,
            start=start,
            end=end,
            timezone=getattr(current.tzinfo, "key", None) or str(current.tzinfo),
            activity=activity,
            subject=subject,
            participants=participants,
            missing_fields=tuple(missing),
        )

    def merge_update(
        self,
        text: str,
        current: dict[str, object],
        *,
        now: datetime | None = None,
    ) -> dict[str, object] | None:
        """Apply explicit edits to one existing draft without creating another."""

        normalized = _normalize(text)
        changed: dict[str, object] = {}
        title = self._edited_title(text)
        if title:
            changed["title"] = title
            changed["activity"] = title
            changed["subject"] = title

        current_start = _parse_iso_datetime(current.get("start_iso"))
        current_end = _parse_iso_datetime(current.get("end_iso"))
        reference = now or current_start or datetime.now().astimezone()
        edited_date = self._date(normalized, reference)
        edited_time = self._time(normalized)
        has_date_signal = bool(
            re.search(
                r"\b(?:hoy|manana|lunes|martes|miercoles|jueves|viernes|"
                r"sabado|domingo|\d{1,2}/\d{1,2}/\d{4}|"
                r"\d{4}-\d{1,2}-\d{1,2})\b",
                normalized,
            )
        )
        has_time_signal = edited_time is not None
        if has_date_signal or has_time_signal:
            selected_date = (
                edited_date
                if has_date_signal
                else current_start.date() if current_start else None
            )
            selected_time = (
                edited_time
                if has_time_signal
                else current_start.timetz().replace(tzinfo=None)
                if current_start
                else None
            )
            if selected_date is not None and selected_time is not None:
                timezone_info = (
                    current_start.tzinfo if current_start else reference.tzinfo
                )
                start = datetime.combine(
                    selected_date,
                    selected_time,
                    tzinfo=timezone_info,
                )
                duration = (
                    current_end - current_start
                    if current_start is not None
                    and current_end is not None
                    and current_end > current_start
                    else timedelta(hours=1)
                )
                explicit_duration = self._duration_if_present(normalized)
                if explicit_duration is not None:
                    duration = explicit_duration
                changed["start_iso"] = start.isoformat()
                changed["end_iso"] = (start + duration).isoformat()

        duration = self._duration_if_present(normalized)
        if (
            duration is not None
            and "start_iso" not in changed
            and current_start is not None
        ):
            changed["end_iso"] = (current_start + duration).isoformat()

        participants = self._participants(text)
        if participants and re.search(r"\b(?:con|participa|invita)\b", normalized):
            changed["participants"] = list(participants)
        location = self._edited_location(text)
        if location:
            changed["location"] = location
        notes = self._edited_notes(text)
        if notes:
            changed["notes"] = notes
        if not changed:
            return None

        merged = dict(current)
        actual_changes = [
            key for key, value in changed.items() if merged.get(key) != value
        ]
        if not actual_changes:
            return None
        merged.update(changed)
        merged["draft_version"] = int(current.get("draft_version") or 1) + 1
        merged["changed_fields"] = actual_changes
        merged["missing_fields"] = _calendar_missing_fields(merged)
        return merged

    def _date(self, normalized: str, current: datetime) -> date | None:
        iso = re.search(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", normalized)
        if iso:
            try:
                return date(*(int(item) for item in iso.groups()))
            except ValueError:
                return None
        local = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", normalized)
        if local:
            day, month, year = (int(item) for item in local.groups())
            try:
                return date(year, month, day)
            except ValueError:
                return None
        if re.search(r"\bmanana\b", normalized):
            return current.date() + timedelta(days=1)
        if re.search(r"\bhoy\b", normalized):
            return current.date()
        for name, weekday in self._weekdays.items():
            if re.search(rf"\b{name}\b", normalized):
                delta = (weekday - current.weekday()) % 7
                if delta == 0:
                    delta = 7
                return current.date() + timedelta(days=delta)
        return None

    @staticmethod
    def _time(normalized: str) -> time | None:
        match = re.search(
            r"\b(?:a|para)\s+las?\s+(\d{1,2})(?::(\d{2}))?\s*"
            r"(a\.?\s*m\.?|p\.?\s*m\.?|am|pm|de la manana|de la tarde|de la noche)?\b",
            normalized,
        )
        if match is None:
            return None
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        period = (match.group(3) or "").replace(" ", "")
        if period in {"pm", "p.m.", "delatarde", "delanoche"} and hour < 12:
            hour += 12
        if period in {"am", "a.m.", "delamanana"} and hour == 12:
            hour = 0
        try:
            return time(hour, minute)
        except ValueError:
            return None

    @staticmethod
    def _duration(normalized: str) -> timedelta:
        return CalendarDraftParser._duration_if_present(normalized) or timedelta(hours=1)

    @staticmethod
    def _duration_if_present(normalized: str) -> timedelta | None:
        match = re.search(
            r"\b(?:durante|por)\s+(\d+)\s*(horas?|minutos?)\b",
            normalized,
        )
        if match is None:
            return None
        amount = int(match.group(1))
        return (
            timedelta(hours=amount)
            if match.group(2).startswith("hora")
            else timedelta(minutes=amount)
        )

    @staticmethod
    def _title_parts(
        text: str,
        participants: tuple[str, ...],
    ) -> tuple[str, str, str]:
        clean = re.sub(
            r"(?i)^\s*(?:(?:s[ií]|vale|claro)[,\s]+)+",
            "",
            text,
        )
        clean = re.sub(
            r"(?i)^\s*(?:por favor\s+)?(?:quiero\s+que\s+)?"
            r"(?:crea|agenda|programa|anade|añade|"
            r"agrega|pon|guarda)\s+(?:un\s+)?(?:evento|actividad|cita|"
            r"recordatorio)?\s*(?:en\s+(?:mi\s+)?calendario)?\s*(?:para|de)?\s*",
            "",
            clean,
        )
        clean = re.sub(
            r"(?i)\b(?:hoy|mañana|manana|lunes|martes|miércoles|miercoles|"
            r"jueves|viernes|sábado|sabado|domingo)\b.*$",
            "",
            clean,
        )
        clean = re.sub(r"\s+", " ", clean).strip(" ,.;:")
        clean = re.sub(r"(?i)^(?:si\s+)+", "", clean).strip()
        purpose = re.search(r"(?i)\bpara\s+(.+)$", clean)
        core = purpose.group(1).strip() if purpose else clean
        core = re.sub(r"(?i)\bcontigo\b", "", core)
        core = re.sub(
            r"(?i)\bcon\s+[^,.;]+$",
            "",
            core,
        ).strip(" ,.;:")
        core = re.sub(
            r"(?i)^(?:desarrollar|revisar|preparar|planificar|organizar|"
            r"trabajar|estudiar|hablar\s+de)\s+",
            "",
            core,
        ).strip(" ,.;:")
        activity = core or clean
        subject = core
        title = _sentence_title(core or clean)
        if participants:
            companion = participants[0]
            if _normalize(companion) not in _normalize(title):
                title = f"{title} con {companion}".strip()
        return activity[:120], subject[:120], title[:120]

    @staticmethod
    def _edited_title(text: str) -> str:
        patterns = (
            r"(?i)\b(?:cambia|modifica)\s+(?:el\s+)?t[ií]tulo\s+(?:a|por)\s+(.+)$",
            r"(?i)\b(?:ponle|ll[aá]malo|que\s+se\s+llame)\s+(.+)$",
            r"(?i)\b(?:el\s+)?t[ií]tulo\s+(?:ser[aá]|es)\s+(.+)$",
        )
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return _sentence_title(match.group(1).strip(" ,.;:"))[:120]
        return ""

    @staticmethod
    def _edited_location(text: str) -> str:
        match = re.search(
            r"(?i)\b(?:lugar|ubicaci[oó]n)\s+(?:ser[aá]|es|a)\s+([^,.;]+)",
            text,
        )
        return match.group(1).strip() if match else ""

    @staticmethod
    def _edited_notes(text: str) -> str:
        match = re.search(
            r"(?i)\b(?:nota|notas|descripci[oó]n)\s*[:=]?\s+(.+)$",
            text,
        )
        return match.group(1).strip() if match else ""

    @staticmethod
    def _participants(text: str) -> tuple[str, ...]:
        if re.search(r"(?i)\bcontigo\b", text):
            return ("Anydes",)
        match = re.search(
            r"(?i)\bcon\s+([^,.;]+?)(?=\s+(?:hoy|mañana|manana|el\s+\d|"
            r"a\s+las|para\s+las|durante)\b|[,.;]|$)",
            text,
        )
        if match is None:
            return ()
        return (match.group(1).strip(),)


def _parse_iso_datetime(value) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value or ""))
    except ValueError:
        return None


def _calendar_missing_fields(parameters: dict[str, object]) -> list[str]:
    missing: list[str] = []
    title = str(parameters.get("title") or "").strip()
    if not title or title == "Actividad":
        missing.append("title")
    if not str(parameters.get("start_iso") or "").strip():
        missing.extend(("date", "time"))
    return list(dict.fromkeys(missing))


def _sentence_title(value: str) -> str:
    compact = " ".join(value.split()).strip(" ,.;:")
    if not compact:
        return ""
    return compact[0].upper() + compact[1:]


_WEB_SERVICES = {
    "spotify": "https://open.spotify.com",
    "youtube": "https://www.youtube.com",
    "word": "https://www.microsoft365.com/launch/word",
    "microsoft word": "https://www.microsoft365.com/launch/word",
    "excel": "https://www.microsoft365.com/launch/excel",
    "powerpoint": "https://www.microsoft365.com/launch/powerpoint",
    "outlook": "https://outlook.office.com",
    "calendar": "https://outlook.office.com/calendar",
    "calendario": "https://outlook.office.com/calendar",
}


def _web_service_url(name: str) -> str:
    return _WEB_SERVICES.get(_normalize(name), "")


def _web_service_request(
    text: str,
    normalized: str,
) -> SemanticAction | None:
    if not re.search(r"\b(?:navegador|web|pestana)\b", normalized):
        return None
    match = re.search(
        r"^\s*(?:por favor\s+)?(?:abre|abrir|inicia|ve a)\s+"
        r"(.+?)\s+(?:en|desde)\s+(?:el|mi)?\s*"
        r"(?:navegador|web|una\s+pestana)\s*[.!?]*$",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    service = match.group(1).strip(" ,.;:")
    if re.search(r"\b(?:pestana|tab|navegador)\b", _normalize(service)):
        return None
    url = _web_service_url(service)
    if not url:
        return SemanticAction(
            capability="web_search",
            entity=service,
            parameters={
                "query": service,
                "preferred_medium": "browser",
                "requested_as_service": True,
            },
            origin_text=text,
        )
    return SemanticAction(
        capability="open_web_service",
        entity=service,
        parameters={
            "service": service,
            "url": url,
            "preferred_medium": "browser",
        },
        origin_text=text,
    )


def _application_request(
    text: str,
    normalized: str,
) -> SemanticAction | None:
    if re.search(r"\b(?:navegador|web|pestana|url|pagina)\b", normalized):
        return None
    inquiry = bool(
        re.match(
            r"^\s*(?:puedes|podrias|sabes)\s+(?:abrir|iniciar|ejecutar)\b",
            normalized,
        )
    )
    pattern = (
        r"^\s*[¿?]*\s*(?:(?:puedes|podrias|sabes)\s+)?"
        r"(?:abre|abrir|inicia|iniciar|ejecuta|ejecutar|lanza|lanzar)\s+"
        r"(.+?)\s*[.!?]*$"
    )
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match is None:
        return None
    name = re.sub(
        r"(?i)\s+(?:por favor|porfa)$",
        "",
        match.group(1),
    ).strip(" ,.;:")
    if not name:
        return None
    return SemanticAction(
        capability="open_application",
        mode="inquire" if inquiry else "execute",
        entity=name,
        parameters={
            "name": name,
            "web_url": _web_service_url(name),
        },
        origin_text=text,
    )


def _media_query(text: str, normalized: str, *, service: str) -> SemanticAction | None:
    if service not in normalized:
        return None
    inquiry = bool(re.search(r"\b(?:puedes|podrias|sabes)\s+abrir\b", normalized))
    action_words = r"(?:abre|abrir|pon|reproduce|reproducir|busca|buscar)"
    if not re.search(rf"\b{action_words}\b", normalized):
        return None
    if inquiry:
        capability = "spotify_open" if service == "spotify" else "youtube_open"
        return SemanticAction(
            capability=capability,
            mode="inquire",
            entity=service,
            origin_text=text,
        )
    query = re.sub(
        rf"(?i)^\s*(?:por favor\s+)?{action_words}\s*",
        "",
        text,
    )
    query = re.sub(rf"(?i)\s+en\s+{service}\s*$", "", query).strip(" ,.;:")
    query = re.sub(rf"(?i)^(?:{service})\s*", "", query).strip(" ,.;:")
    if not query or _normalize(query) == service:
        capability = "spotify_open" if service == "spotify" else "youtube_open"
        return SemanticAction(
            capability=capability,
            mode="inquire" if inquiry else "execute",
            entity=service,
            origin_text=text,
        )
    capability = "spotify_search" if service == "spotify" else "youtube_search"
    return SemanticAction(
        capability=capability,
        mode="inquire" if inquiry else "execute",
        entity=query,
        parameters={"query": query, "playback_requested": "reproduc" in normalized or "pon " in normalized},
        origin_text=text,
    )


def _web_query(text: str, normalized: str) -> str:
    match = re.search(
        r"\b(?:busca|buscar|googlea|investiga)\s+(?:en\s+(?:google|internet|la web)\s+)?(.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        match = re.search(
            r"\babre\s+(?:una\s+)?pesta[nñ]a\s+(?:en\s+mi\s+navegador\s+)?"
            r"(?:y\s+)?busca\s+(.+)$",
            text,
            flags=re.IGNORECASE,
        )
    if match is None:
        return ""
    query = match.group(1).strip(" ,.;:")
    if _normalize(query) in {"google", "internet", "la web"}:
        return ""
    return query


def _is_weather_request(normalized: str) -> bool:
    return bool(
        re.search(
            r"\b(?:clima|temperatura|pronostico|esta lloviendo|hace frio|hace calor)\b",
            normalized,
        )
        and re.search(
            r"\b(?:que|cual|como|dime|consulta|revisa|mira|hace|esta)\b",
            normalized,
        )
    )


def _is_calendar_request(normalized: str) -> bool:
    return bool(
        re.search(r"\b(?:calendario|evento|cita|agenda)\b", normalized)
        and re.search(
            r"\b(?:crea|crear|agenda|agendar|programa|programar|anade|anadir|"
            r"agrega|agregar|pon|poner|guarda|guardar)\b",
            normalized,
        )
    )


def _extract_url(text: str) -> str:
    match = re.search(r"\bhttps?://[^\s<>()]+", text, flags=re.IGNORECASE)
    return match.group(0).rstrip(".,;:") if match else ""


def _domain_target(text: str) -> str:
    compact = text.strip().strip(".,;:")
    if re.fullmatch(
        r"(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s]*)?",
        compact,
        flags=re.IGNORECASE,
    ):
        return "https://" + compact
    return ""


def _valid_http_target(raw: str) -> bool:
    try:
        parsed = urlparse(raw)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _result_succeeded(capability: str, data: dict[str, object]) -> bool:
    if capability == "weather_current":
        return bool(data.get("available"))
    if capability == "calendar_create":
        return bool(data.get("created"))
    if capability in {
        "open_application",
        "open_browser_tab",
        "open_url",
        "open_web_service",
        "web_search",
        "youtube_open",
        "youtube_search",
        "spotify_open",
        "spotify_search",
        "spotify_web_search",
    }:
        return bool(data.get("opened") or data.get("launch_requested"))
    return True


def _result_verified(capability: str, data: dict[str, object]) -> bool:
    if capability == "weather_current":
        return bool(data.get("available") and data.get("observed_at"))
    if capability == "calendar_create":
        return bool(data.get("created") and data.get("entry_id"))
    if capability in {"spotify_search", "spotify_web_search"}:
        return bool(data.get("search_opened"))
    if capability == "spotify_open":
        return bool(data.get("opened_verified"))
    if capability in {
        "open_browser_tab",
        "open_url",
        "open_web_service",
        "web_search",
        "youtube_open",
        "youtube_search",
    } and data.get("launch_requested"):
        return bool(data.get("url_loaded_verified"))
    return bool(data.get("opened_verified", data.get("opened", False)))


def _weather_text(receipt: ExecutionReceipt) -> str:
    if receipt.status != "success":
        if receipt.error in {
            "external_network_disabled",
            "weather_not_allowed",
            "weather_location_not_configured",
        }:
            return "No tengo habilitada una ubicación meteorológica autorizada."
        return "No pude consultar el clima real ahora mismo."
    data = receipt.data
    labels = {
        "sunny": "está despejado",
        "cloudy": "está nublado",
        "rain": "está lloviendo",
        "storm": "hay tormenta",
        "snow": "está nevando",
    }
    condition = labels.get(str(data.get("weather")), "el clima está variable")
    temperature = data.get("temperature_c")
    location = str(data.get("location_label") or "tu ubicación configurada")
    if temperature is None:
        return f"En {location}, {condition}."
    return f"En {location}, {condition} y hay {temperature:g} °C."


def _calendar_text(receipt: ExecutionReceipt) -> str:
    if receipt.status == "success":
        title = str(receipt.data.get("title") or "el evento")
        return f"Listo, guardé {title} en el calendario conectado."
    if receipt.status == "awaiting_user":
        title = str(receipt.data.get("title") or "el evento")
        return (
            f"Preparé {title} en Microsoft 365. "
            "Revísalo y pulsa Guardar para confirmar que se agregue."
        )
    if receipt.error == "calendar_integration_unavailable":
        return "No encuentro un calendario local conectado en el que pueda crear el evento."
    if receipt.status == "denied":
        return "No creé el evento porque no recibí autorización."
    return "Intenté crear el evento, pero el calendario no confirmó que se guardara."


def _spotify_search_text(receipt: ExecutionReceipt) -> str:
    query = str(receipt.data.get("query") or "ese contenido")
    if receipt.status != "success":
        return _failure_text(receipt, f"la búsqueda de {query}")
    if receipt.data.get("playback_confirmed"):
        return f"Listo, {query} se está reproduciendo."
    service = (
        "Spotify Web"
        if receipt.capability == "spotify_web_search"
        else "Spotify"
    )
    return (
        f"Abrí la búsqueda de {query} en {service}. "
        "No puedo confirmar que ya se esté reproduciendo."
    )


def _failure_text(receipt: ExecutionReceipt, label: str) -> str:
    if receipt.status == "denied":
        return f"No abrí {label} porque no recibí autorización."
    if receipt.status == "unavailable":
        return f"No encuentro disponible {label} en este equipo."
    return f"Intenté abrir {label}, pero Windows no confirmó la acción."


def _normalize(text: str) -> str:
    folded = "".join(
        character
        for character in unicodedata.normalize("NFKD", text.casefold())
        if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", " ", folded).strip()
