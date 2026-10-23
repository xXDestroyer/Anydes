from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
import re

from ..contracts import ExecutionStatus, PermissionEffect, ToolExecutionResult
from .contracts import PlannedToolAction


class ResponseType(StrEnum):
    INFORMATION = "INFORMATION"
    SIDE_EFFECT_SUCCESS = "SIDE_EFFECT_SUCCESS"
    SIDE_EFFECT_FAILURE = "SIDE_EFFECT_FAILURE"


@dataclass(frozen=True)
class ResponseGateDecision:
    allowed: bool
    response_type: ResponseType
    reason: str
    expected_side_effect: str = ""
    expected_path: str = ""
    linked_request_id: str = ""


_MODEL_SIDE_EFFECT_CLAIM = re.compile(
    r"\b(?:aqui\s+tienes|he\s+(?:creado|guardado|editado|eliminado)|"
    r"ya\s+(?:cree|guarde|edite|elimine)|(?:cree|guarde|edite|elimine)\s+(?:el|la)|"
    r"listo[, :]*(?:cree|guarde|edite|elimine|el\s+archivo|el\s+documento))\b",
    re.IGNORECASE,
)


def gate_action_response(
    action: PlannedToolAction,
    results: tuple[ToolExecutionResult, ...],
    verification: dict,
) -> ResponseGateDecision:
    if action.operation == "delete_collection":
        targets = {
            _canonical(str(item.get("absolute_canonical_path", "")))
            for item in action.arguments.get("delete_plan", {}).get("targets", ())
            if item.get("absolute_canonical_path")
        }
        evidenced = {
            _canonical(side_effect.path)
            for result in results
            if result.status == ExecutionStatus.SUCCESS
            for side_effect in result.side_effects
            if side_effect.effect == PermissionEffect.DELETE
        }
        deleted = {
            _canonical(str(item)) for item in verification.get("deleted", ()) if item
        }
        failed = {
            _canonical(str(item.get("path", "")))
            for item in verification.get("failed", ()) if item.get("path")
        }
        skipped = {
            _canonical(str(item.get("path", "")))
            for item in verification.get("skipped", ()) if item.get("path")
        }
        accounted = deleted | failed | skipped
        allowed = (
            bool(targets)
            and deleted == evidenced
            and targets == accounted
            and verification.get("status") in {"VERIFIED", "VERIFIED_PARTIAL"}
        )
        return ResponseGateDecision(
            allowed,
            ResponseType.SIDE_EFFECT_SUCCESS if allowed else ResponseType.SIDE_EFFECT_FAILURE,
            "linked_delete_collection_success" if allowed else "delete_collection_evidence_mismatch",
            PermissionEffect.DELETE.value,
            str(action.arguments.get("delete_plan", {}).get("workspace", "")),
            results[-1].request_id if allowed and results else "",
        )
    expected = _expected_file_effect(action)
    if expected is None:
        return ResponseGateDecision(True, ResponseType.INFORMATION, "no_file_side_effect_expected")
    effect, expected_path = expected
    canonical = _canonical(expected_path)
    for result in results:
        if result.status != ExecutionStatus.SUCCESS:
            continue
        for side_effect in result.side_effects:
            compatible = side_effect.effect == effect
            if effect == PermissionEffect.CREATE:
                compatible = side_effect.effect in {PermissionEffect.CREATE, PermissionEffect.OVERWRITE}
            if compatible and _canonical(side_effect.path) == canonical:
                if action.operation in {
                    "verified_create_document", "structured_edit_text",
                    "generated_create_document", "generated_document_edit",
                    "document_transform", "append_document", "document_conversion",
                    "structural_document_edit", "rename_existing",
                    "copy_existing", "general_generate_code", "general_modify_code",
                    "general_generated_document",
                }:
                    if verification.get("status") != "VERIFIED":
                        return ResponseGateDecision(
                            False,
                            ResponseType.SIDE_EFFECT_FAILURE,
                            "post_write_verification_missing",
                            effect.value,
                            canonical,
                            result.request_id,
                        )
                return ResponseGateDecision(
                    True,
                    ResponseType.SIDE_EFFECT_SUCCESS,
                    "linked_action_result_success",
                    effect.value,
                    canonical,
                    result.request_id,
                )
    return ResponseGateDecision(
        False,
        ResponseType.SIDE_EFFECT_FAILURE,
        "matching_action_result_side_effect_missing",
        effect.value,
        canonical,
    )


def model_output_claims_side_effect(text: str) -> bool:
    normalized = _strip_accents(str(text or "").casefold())
    return bool(_MODEL_SIDE_EFFECT_CLAIM.search(normalized))


def _expected_file_effect(action: PlannedToolAction) -> tuple[PermissionEffect, str] | None:
    args = action.arguments
    path = str(args.get("path") or args.get("destination") or action.target_path or "")
    if not path:
        return None
    if action.operation in {
        "verified_create_document", "generated_create_document",
        "document_conversion", "text_files_to_pdf", "general_generate_code",
        "general_generated_document",
    }:
        return PermissionEffect.CREATE, path
    if action.operation == "general_modify_code":
        return PermissionEffect.OVERWRITE, path
    if action.operation == "copy_existing":
        return PermissionEffect.CREATE, str(args.get("destination", ""))
    if action.operation in {
        "structured_edit_text", "generated_document_edit",
        "document_transform", "append_document", "structural_document_edit",
    }:
        return PermissionEffect.EDIT, path
    if action.operation == "rename_existing":
        return PermissionEffect.MOVE, str(args.get("path", action.target_path))
    if action.intent == "delete":
        return PermissionEffect.DELETE, path
    if action.intent in {"patch_code"}:
        return PermissionEffect.EDIT, path
    if action.intent in {"compress_zip"}:
        return PermissionEffect.CREATE, path
    return None


def _canonical(path: str) -> str:
    return str(Path(path).resolve(strict=False)).casefold()


def _strip_accents(value: str) -> str:
    import unicodedata

    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(char for char in decomposed if not unicodedata.combining(char))
