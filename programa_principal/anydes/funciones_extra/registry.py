from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Literal

from anydes.funciones_extra import (
    apps,
    calendar_tools,
    local,
    memory_tools,
    network,
    weather,
)
from anydes.vision import tools as vision_tools
from anydes.nucleo.audit import AuditLog
from anydes.nucleo.config import Config
from anydes.nucleo.permissions import PermissionManager

ToolStatus = Literal["success", "denied", "failed", "missing"]
ToolPayload = dict[str, object]
ToolHandler = Callable[[str], ToolPayload | str]


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    status: ToolStatus
    data: ToolPayload
    error: str | None = None

    def to_legacy_text(self) -> str:
        if self.status == "success":
            return json.dumps(self.data, ensure_ascii=False, indent=2, sort_keys=True)
        payload: dict[str, object] = {"status": self.status, "tool_name": self.tool_name}
        if self.error:
            payload["error"] = self.error
        payload.update(self.data)
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    handler: ToolHandler


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    @classmethod
    def default(cls, config: Config | None = None) -> "ToolRegistry":
        memory = memory_tools.handlers_for_config(config) if config is not None else {}
        vision = vision_tools.handlers_for_config(config) if config is not None else {}
        return cls(
            [
                Tool("current_time", "Muestra la fecha y hora local.", local.current_time),
                Tool(
                    "system_info",
                    "Muestra sistema operativo, version de Python y datos tecnicos del dispositivo; no identifica personas.",
                    local.system_info,
                ),
                Tool("word_count", "Cuenta palabras, lineas y caracteres.", local.word_count),
                Tool("format_json", "Formatea JSON.", local.format_json),
                Tool("open_url", "Abre una URL en el navegador.", apps.open_url),
                Tool("chrome_search", "Busca texto en Google/Chrome.", apps.chrome_search),
                Tool("open_youtube", "Abre YouTube.", apps.open_youtube),
                Tool("youtube_search", "Busca videos en YouTube.", apps.youtube_search),
                Tool("open_spotify", "Abre Spotify.", apps.open_spotify),
                Tool(
                    "spotify_search",
                    "Abre una búsqueda de contenido en Spotify sin afirmar reproducción.",
                    apps.spotify_search,
                ),
                Tool(
                    "spotify_web_search",
                    "Abre una búsqueda en Spotify Web.",
                    apps.spotify_web_search,
                ),
                Tool("open_app", "Abre una aplicacion local conocida.", apps.open_app),
                Tool(
                    "weather_current",
                    "Consulta el clima actual de la ubicación configurada.",
                    (
                        weather.handler_for_config(config)
                        if config is not None
                        else lambda _: {
                            "available": False,
                            "reason": "weather_config_unavailable",
                        }
                    ),
                ),
                Tool(
                    "calendar_create",
                    "Crea un evento verificado o prepara su guardado en Microsoft 365.",
                    calendar_tools.create_calendar_event,
                ),
                Tool("screen_capture", "Captura una imagen de la pantalla.", vision.get("screen_capture", _missing_vision_tool)),
                Tool(
                    "screen_watch_start",
                    "Empieza a ver la pantalla en segundo plano.",
                    vision.get("screen_watch_start", _missing_vision_tool),
                ),
                Tool("screen_watch_stop", "Deja de ver la pantalla.", vision.get("screen_watch_stop", _missing_vision_tool)),
                Tool("screen_status", "Muestra si Anydes esta viendo la pantalla.", vision.get("screen_status", _missing_vision_tool)),
                Tool(
                    "external_vision_session",
                    "Autoriza vision externa solamente durante esta sesion.",
                    vision.get("external_vision_session", _missing_vision_tool),
                ),
                Tool("memory_profile", "Resume memoria local sin mostrar secretos.", memory.get("memory_profile", memory_tools.memory_profile)),
                Tool("memory_conversation_search", "Busca en el historial completo de conversaciones.", memory.get("memory_conversation_search", memory_tools.memory_conversation_search)),
                Tool("memory_recent", "Muestra conversaciones recientes guardadas.", memory.get("memory_recent", memory_tools.memory_recent)),
                Tool("memory_permanent_read", "Lee memoria permanente de Anydes.", memory.get("memory_permanent_read", memory_tools.memory_permanent_read)),
                Tool("memory_permanent_write", "Escribe memoria permanente de Anydes.", memory.get("memory_permanent_write", memory_tools.memory_permanent_write)),
                Tool("memory_temporary_read", "Lee memoria temporal de Anydes.", memory.get("memory_temporary_read", memory_tools.memory_temporary_read)),
                Tool("memory_temporary_write", "Escribe memoria temporal de Anydes.", memory.get("memory_temporary_write", memory_tools.memory_temporary_write)),
                Tool("memory_temporary_clear", "Borra memoria temporal de Anydes.", memory.get("memory_temporary_clear", memory_tools.memory_temporary_clear)),
                Tool("memory_protected_read", "Lee memoria protegida de Anydes.", memory.get("memory_protected_read", memory_tools.memory_protected_read)),
                Tool("memory_protected_write", "Escribe memoria protegida de Anydes.", memory.get("memory_protected_write", memory_tools.memory_protected_write)),
                Tool("memory_read", "Alias: lee memoria permanente.", memory.get("memory_read", memory_tools.memory_permanent_read)),
                Tool("memory_write", "Alias: escribe memoria permanente.", memory.get("memory_write", memory_tools.memory_permanent_write)),
                Tool("fetch_url", "Descarga una URL. Bloqueado por defecto.", network.fetch_url),
                Tool("open_browser", "Abre una pagina web. Bloqueado por defecto.", network.open_browser),
                Tool("run_command", "Ejecuta un comando del sistema. Bloqueado.", local.run_command),
            ]
        )

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def run_structured(
        self,
        name: str,
        argument: str,
        permissions: PermissionManager,
        audit: AuditLog,
        *,
        interactive: bool,
        explicit_request: bool = False,
    ) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            audit.record("tool_missing", tool=name)
            return ToolResult(
                tool_name=name,
                status="missing",
                data={"requested_tool": name},
                error="unknown_tool",
            )

        audit.record("tool_requested", tool=name, argument_length=len(argument))
        authorized = (
            permissions.authorize_explicit_request(name, tool.description)
            if explicit_request
            else permissions.authorize(name, tool.description, interactive=interactive)
        )
        if not authorized:
            audit.record("tool_denied", tool=name)
            return ToolResult(
                tool_name=name,
                status="denied",
                data={"requested_tool": name, "permission_required": True},
                error="permission_denied",
            )

        try:
            result = tool.handler(argument)
        except Exception as exc:
            audit.record("tool_failed", tool=name, error_type=type(exc).__name__)
            return ToolResult(
                tool_name=name,
                status="failed",
                data={"requested_tool": name},
                error="tool_execution_failed",
            )

        audit.record("tool_completed", tool=name)
        return ToolResult(tool_name=name, status="success", data=_normalize_tool_payload(result))

    def run(
        self,
        name: str,
        argument: str,
        permissions: PermissionManager,
        audit: AuditLog,
        *,
        interactive: bool,
    ) -> str:
        return self.run_structured(
            name,
            argument,
            permissions,
            audit,
            interactive=interactive,
            explicit_request=False,
        ).to_legacy_text()


def _normalize_tool_payload(value: ToolPayload | str) -> ToolPayload:
    if isinstance(value, dict):
        return _json_ready_dict(value)
    text = str(value)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"text": text}
    if isinstance(parsed, dict):
        return _json_ready_dict(parsed)
    return {"value": _json_ready(parsed)}


def _json_ready_dict(value: dict[str, Any]) -> ToolPayload:
    return {str(key): _json_ready(item) for key, item in value.items()}


def _json_ready(value: Any) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _missing_vision_tool(_: str = "") -> dict[str, object]:
    return {
        "available": False,
        "reason": "screen_vision_unavailable_in_this_mode",
    }
