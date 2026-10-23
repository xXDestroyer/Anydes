from __future__ import annotations

import json
import re
import unicodedata

from ..contracts import CapabilityResolutionKind
from ..conversation.contracts import ConversationToolResponse
from .store import SettingsStore


class CapabilityAwareness:
    """Answers operational questions from verified runtime state, not LLM lore."""

    def __init__(self, registry, settings: SettingsStore) -> None:
        self.registry = registry
        self.settings = settings

    def answer(self, text: str) -> ConversationToolResponse | None:
        normalized = _normalize(text)
        if not _is_capability_question(normalized):
            return None
        snapshot = self.registry.capability_snapshot()
        workspace = self.settings.snapshot()
        capabilities = set(snapshot.available_capabilities)

        if "core" in normalized or "nucleo" in normalized:
            message = (
                "No. El núcleo de Anydes está protegido y mis herramientas no pueden "
                "modificarlo, aunque una carpeta externa esté autorizada."
            )
        elif _asks_workspace(normalized):
            message = _workspace_answer(workspace)
        elif "word" in normalized or "docx" in normalized:
            message = _capability_answer(
                "document.create_docx" in capabilities,
                "crear documentos Word",
                workspace.active_workspace,
            )
        elif "pdf" in normalized:
            message = _capability_answer(
                "document.create_pdf" in capabilities,
                "crear documentos PDF",
                workspace.active_workspace,
            )
        elif "ejecutar" in normalized and ("codigo" in normalized or "python" in normalized):
            message = (
                "Sí, puedo ejecutar Python y C++ dentro del sandbox aprobado, siempre "
                "con confirmación antes de ejecutar código. No puede escribir en mi núcleo."
            )
        elif "modificar" in normalized and "document" in normalized:
            message = (
                "Sí, puedo leer y modificar documentos dentro de una carpeta autorizada. "
                "Si la operación sobrescribe un archivo importante, te pediré confirmación. "
                f"Ahora mismo el workspace activo es {_display_path(workspace.active_workspace)}."
            )
        elif "crear" in normalized and "archivo" in normalized:
            message = _capability_answer(
                "filesystem.create_text" in capabilities,
                "crear archivos",
                workspace.active_workspace,
            )
        else:
            tools = [str(item["tool_id"]).removeprefix("official.").title() for item in snapshot.tools]
            message = (
                f"Tengo {len(snapshot.tools)} herramientas oficiales activas y "
                f"{len(snapshot.available_capabilities)} capacidades verificadas: "
                f"{', '.join(tools)}. Puedo trabajar con archivos, documentos, ZIP y "
                "código aislado dentro de carpetas autorizadas."
            )
        return ConversationToolResponse(
            handled=True,
            text=message,
            resolver_kind=CapabilityResolutionKind.REASONING_ONLY,
            executed=False,
            debug={"source": "ToolRegistry+SettingsStore"},
        )

    def context_note(self) -> str:
        tool_snapshot = self.registry.capability_snapshot()
        workspace = self.settings.snapshot()
        payload = {
            "tool_system_enabled": True,
            "available_capabilities": list(tool_snapshot.available_capabilities),
            "active_workspace": workspace.active_workspace,
            "permanent_workspace": workspace.permanent_workspace or None,
            "temporary_workspace": workspace.temporary_workspace or None,
            "authorized_external_scopes": list(workspace.authorized_external_scopes),
            "protected_core": True,
        }
        return "TOOL_CAPABILITY_CONTEXT (local source of truth): " + json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )


def _capability_answer(available: bool, description: str, workspace: str) -> str:
    if not available:
        return f"No tengo activa una herramienta aprobada para {description}."
    return (
        f"Sí, puedo {description}, pero solo dentro de una carpeta autorizada. "
        f"Ahora mismo usaría este workspace: {_display_path(workspace)}."
    )


def _workspace_answer(workspace) -> str:
    parts = [
        f"Mi workspace activo es {_display_path(workspace.active_workspace)} "
        f"({_kind_label(workspace.active_workspace_kind)})."
    ]
    if workspace.permanent_workspace:
        parts.append(f"Workspace permanente: {_display_path(workspace.permanent_workspace)}.")
    if workspace.temporary_workspace:
        parts.append(f"Workspace temporal de esta sesión: {_display_path(workspace.temporary_workspace)}.")
    extras = [
        item
        for item in workspace.authorized_external_scopes
        if item not in {workspace.permanent_workspace, workspace.temporary_workspace}
    ]
    if extras:
        parts.append(
            "Scopes adicionales de esta sesión: "
            + ", ".join(_display_path(item) for item in extras)
            + "."
        )
    parts.append("La protección del núcleo de Anydes permanece activada.")
    return " ".join(parts)


def _kind_label(value: str) -> str:
    return {
        "internal": "interno seguro",
        "permanent": "permanente",
        "temporary": "temporal de esta sesión",
    }.get(value, value)


def _is_capability_question(text: str) -> bool:
    question = any(marker in text for marker in (
        "que puedes hacer",
        "que herramientas",
        "que capacidades",
        "en que carpeta",
        "donde tienes permiso",
        "donde puedes trabajar",
        "a que carpetas",
        "puedes crear",
        "puedes ejecutar",
        "puedes modificar",
        "tienes permiso",
    ))
    return question and not any(marker in text for marker in (
        "quiero que",
        "hazlo",
        "creame",
        "modifica ahora",
    ))


def _asks_workspace(text: str) -> bool:
    return bool(re.search(r"(?:en que|donde|a que).*(?:carpeta|permiso|trabajar)|workspace", text))


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _display_path(path: str) -> str:
    from pathlib import Path

    name = Path(path).resolve(strict=False).name
    return name or "el workspace autorizado"
