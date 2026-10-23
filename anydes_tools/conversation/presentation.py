from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Callable


_INLINE_WINDOWS_PATH = re.compile(r"`([A-Za-z]:\\[^`\r\n]+)`")
_TECHNICAL_BASENAME = re.compile(r"(?<![\w])(?:[\w-]+\.)+[A-Za-z0-9]{1,12}(?![\w])")
_BRACKET_TOOL_CALL = re.compile(
    r"\[(?:official\.)?(?:document|filesystem|developer|archive)\."
    r"[a-z0-9_.-]+(?:\s+[^\]\r\n]*)?\]",
    re.IGNORECASE,
)
_XML_TOOL_CALL = re.compile(
    r"<tool_call\b[^>]*>.*?</tool_call\s*>|<tool_call\b[^>]*/\s*>",
    re.IGNORECASE | re.DOTALL,
)
_FENCED_ACTION_JSON = re.compile(
    r"```(?:json)?\s*(\{.*?(?:\"tool_id\"|\"capability\"|\"action_type\"|\"action\").*?\})\s*```",
    re.IGNORECASE | re.DOTALL,
)
_OPEN_BRACKET_TAIL = re.compile(r"\[[^\]\r\n]*$")
_INLINE_ACTION_JSON = re.compile(
    r"\{[^{}]*(?:\"(?:tool_id|capability|action_type|action|arguments)\"\s*:)[^{}]*\}",
    re.IGNORECASE | re.DOTALL,
)
_PSEUDO_FUNCTION = re.compile(
    r"(?:official\.)?(?:document|filesystem|developer|archive)\.[a-z0-9_.-]+\s*\([^)]*\)",
    re.IGNORECASE | re.DOTALL,
)
_CONTROL_XML = re.compile(
    r"<(?:tool_call|action_request|function_call)\b[^>]*>.*?</(?:tool_call|action_request|function_call)\s*>"
    r"|<(?:tool_call|action_request|function_call)\b[^>]*/\s*>",
    re.IGNORECASE | re.DOTALL,
)


def normalize_chat_preserving_paths(
    text: str,
    normalizer: Callable[[str], str],
) -> str:
    """Protect explicit Windows paths from the math/LaTeX presentation pass."""

    paths: list[str] = []

    def protect(match: re.Match[str]) -> str:
        token = f"ANYDESPATHPLACEHOLDER{len(paths)}TOKEN"
        paths.append(match.group(0))
        return token

    protected = _INLINE_WINDOWS_PATH.sub(protect, str(text))
    rendered = normalizer(_TECHNICAL_BASENAME.sub(protect, protected))
    for index, path in enumerate(paths):
        rendered = rendered.replace(
            f"ANYDESPATHPLACEHOLDER{index}TOKEN",
            path,
        )
    return rendered


def sanitize_visible_model_text(text: str, *, partial: bool = False) -> tuple[str, bool]:
    """Remove model-simulated tool protocol from conversational presentation."""

    source = str(text or "")
    detected = False
    rendered, count = _XML_TOOL_CALL.subn("", source)
    detected = detected or count > 0
    rendered, count = _BRACKET_TOOL_CALL.subn("", rendered)
    detected = detected or count > 0
    rendered, count = _FENCED_ACTION_JSON.subn("", rendered)
    detected = detected or count > 0
    rendered, count = _INLINE_ACTION_JSON.subn("", rendered)
    detected = detected or count > 0
    rendered, count = _PSEUDO_FUNCTION.subn("", rendered)
    detected = detected or count > 0
    rendered, count = _CONTROL_XML.subn("", rendered)
    detected = detected or count > 0

    stripped = rendered.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            payload = json.loads(stripped)
        except (TypeError, ValueError):
            payload = None
        if isinstance(payload, dict) and {
            "tool_id", "capability", "action_type", "action", "arguments"
        }.intersection(payload):
            rendered = ""
            detected = True

    if partial:
        tail = _OPEN_BRACKET_TAIL.search(rendered)
        if tail is not None:
            candidate = tail.group(0).casefold()
            if any(name.startswith(candidate[1:].split(".", 1)[0]) for name in (
                "document", "filesystem", "developer", "archive",
            )):
                rendered = rendered[:tail.start()]
                detected = True
        if rendered.lstrip().startswith("{") and any(
            marker in rendered for marker in ('"tool_id"', '"capability"', '"arguments"', '"action"')
        ):
            rendered = ""
            detected = True

    clean = re.sub(r"\n{3,}", "\n\n", rendered).strip()
    if detected and not clean and not partial:
        clean = (
            "No pude convertir esa intención en un plan de herramienta válido, "
            "así que no ejecuté ninguna acción."
        )
    return clean, detected


def gate_untrusted_model_output(text: str, *, partial: bool = False) -> tuple[str, bool, str]:
    """Keep model prose separate from the local action protocol and evidence."""

    from .response_gate import model_output_claims_side_effect

    clean, protocol_detected = sanitize_visible_model_text(text, partial=partial)
    side_effect_claim = model_output_claims_side_effect(clean)
    if not protocol_detected and not side_effect_claim:
        return clean, False, "CONVERSATIONAL_TEXT"
    if partial:
        return "", True, "MODEL_ACTION_PROTOCOL_BLOCKED"
    return (
        "No pude ejecutar esa acción mediante una herramienta verificada, así que no realicé ningún cambio.",
        True,
        "MODEL_ACTION_PROTOCOL_BLOCKED",
    )


def final_response_gate(
    text: str,
    *,
    verified_action_result: bool = False,
) -> tuple[str, bool, str]:
    """Single UI/TTS gate for model and local-tool response paths."""

    from .response_gate import model_output_claims_side_effect

    clean, protocol_detected = sanitize_visible_model_text(text)
    unsupported_claim = model_output_claims_side_effect(clean) and not verified_action_result
    if not protocol_detected and not unsupported_claim:
        return clean, False, "VERIFIED_ACTION_RESULT" if verified_action_result else "CONVERSATIONAL_TEXT"
    return (
        "No pude ejecutar esa acción mediante una herramienta verificada, así que no realicé ningún cambio.",
        True,
        "FINAL_RESPONSE_GATE_BLOCKED",
    )


def present_workspace_paths(
    text: str,
    workspace_paths: tuple[str, ...],
    *,
    reveal_full_path: bool,
) -> str:
    """Keep canonical paths internal unless this turn explicitly requests one."""

    rendered = str(text or "")
    if reveal_full_path:
        return rendered
    unique = sorted(
        {str(value).rstrip("\\/") for value in workspace_paths if str(value).strip()},
        key=len,
        reverse=True,
    )
    for canonical in unique:
        display = Path(canonical).resolve(strict=False).name or "workspace autorizado"
        variants = {
            canonical,
            canonical + "\\",
            canonical.replace("\\", "\\\\"),
            canonical.replace("\\", "/"),
        }
        for variant in sorted(variants, key=len, reverse=True):
            rendered = rendered.replace(variant, display)
    return rendered


def install_math_path_preservation() -> None:
    from anydes.continuity_v2 import math_presentation

    if getattr(math_presentation, "_tool_path_preservation_installed", False):
        return
    original = math_presentation.normalize_chat

    def normalize_chat_with_paths(text: str) -> str:
        return normalize_chat_preserving_paths(text, original)

    math_presentation.normalize_chat = normalize_chat_with_paths
    math_presentation._tool_path_preservation_installed = True
