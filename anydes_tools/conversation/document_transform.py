from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
import hashlib
from pathlib import Path
import re
from typing import Any, Callable, Protocol, Sequence

from .contracts import ConversationToolState


class DocumentOperation(StrEnum):
    CREATE_NEW = "CREATE_NEW"
    READ_EXISTING = "READ_EXISTING"
    EDIT_EXISTING = "EDIT_EXISTING"
    TRANSFORM_EXISTING = "TRANSFORM_EXISTING"
    APPEND_EXISTING = "APPEND_EXISTING"
    OVERWRITE_EXISTING = "OVERWRITE_EXISTING"
    COPY_EXISTING = "COPY_EXISTING"
    MOVE_EXISTING = "MOVE_EXISTING"
    DELETE_EXISTING = "DELETE_EXISTING"
    CONVERT_EXISTING = "CONVERT_EXISTING"
    RENAME_EXISTING = "RENAME_EXISTING"
    REASONING_ONLY = "REASONING_ONLY"


class PredicateKind(StrEnum):
    DETERMINISTIC = "DETERMINISTIC"
    SEMANTIC = "SEMANTIC"


@dataclass(frozen=True)
class OperationIntent:
    operation: DocumentOperation
    source_text: str
    requires_existing_target: bool


@dataclass(frozen=True)
class SemanticPredicate:
    description: str
    kind: PredicateKind = PredicateKind.SEMANTIC


@dataclass(frozen=True)
class DocumentTransformationSpec:
    target_path: str
    content_type: str
    iteration_unit: str
    condition: dict[str, Any]
    operation: str
    payload: str
    preserve_formatting: bool = True
    expected_changes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DocumentTransformationSpec":
        return cls(
            target_path=str(value.get("target_path", "")),
            content_type=str(value.get("content_type", "text")),
            iteration_unit=str(value.get("iteration_unit", "LINE")),
            condition=dict(value.get("condition", {})),
            operation=str(value.get("operation", "APPEND_TO_MATCH")),
            payload=str(value.get("payload", "")),
            preserve_formatting=bool(value.get("preserve_formatting", True)),
            expected_changes=(
                int(value["expected_changes"])
                if value.get("expected_changes") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class DocumentPatch:
    line_id: int
    operation: str
    text: str


@dataclass(frozen=True)
class PatchPlan:
    target_path: str
    before_hash: str
    patches: tuple[DocumentPatch, ...]
    expected_changes: int
    selected_line_ids: tuple[int, ...]


@dataclass(frozen=True)
class PatchVerification:
    status: str
    before_hash: str
    after_hash: str
    expected_changes: int
    actual_changes: int
    selected_lines_match: bool
    unselected_lines_unchanged: bool


class SemanticLineEvaluator(Protocol):
    def __call__(self, predicate: SemanticPredicate, lines: Sequence[str]) -> Sequence[int]:
        ...


class TargetResolver:
    """Resolve existing documents only from explicit or recorded canonical state."""

    @staticmethod
    def resolve(
        explicit_path: str,
        state: ConversationToolState,
        explicit_filename: str = "",
    ) -> str:
        if explicit_path:
            path = Path(explicit_path)
            if not path.is_absolute():
                path = Path(state.active_workspace) / path
            return str(path.resolve(strict=False))
        if explicit_filename:
            workspace_candidate = (
                Path(state.active_workspace) / explicit_filename
            ).resolve(strict=False)
            if workspace_candidate.exists():
                return str(workspace_candidate)
            matching_recent = [
                Path(value).resolve(strict=False)
                for value in reversed(state.recent_files)
                if Path(value).name.casefold() == explicit_filename.casefold()
            ]
            if matching_recent:
                return str(matching_recent[0])
            return str(workspace_candidate)
        if state.active_document_ref is not None:
            return state.active_document_ref.absolute_canonical_path
        if state.active_document:
            return str(Path(state.active_document).resolve(strict=False))
        document_candidates = [
            value for value in reversed(state.recent_files)
            if Path(value).suffix.casefold() in {".txt", ".md", ".docx"}
        ]
        return document_candidates[0] if len(document_candidates) == 1 else ""


def classify_document_operation(text: str) -> DocumentOperation:
    normalized = _normalize(text)
    if re.search(
        r"\b(?:cambi\w*|modific\w*)\s+(?:el\s+)?nombre\b|\brenombr\w*\b",
        normalized,
    ):
        return DocumentOperation.RENAME_EXISTING
    if re.search(r"\b(?:conviert\w*|transform\w*)\b.*\b(?:pdf|docx|word|txt|md|markdown)\b", normalized):
        return DocumentOperation.CONVERT_EXISTING
    if re.search(
        r"\b(?:al\s+(?:final|inicio)\s+(?:pon|agreg|anad|escrib)\w*|"
        r"(?:pon|agreg|anad|escrib)\w*\s+(?:al\s+)?(?:final|inicio))\b",
        normalized,
    ):
        return DocumentOperation.EDIT_EXISTING
    if re.search(r"\b(?:edit\w*|modific\w*|corrig\w*|marc\w*)\b", normalized):
        return DocumentOperation.EDIT_EXISTING
    if (
        re.search(r"\b(?:cre\w*|hac\w*|hag\w*|haz\w*|gener\w*|prepar\w*)\b", normalized)
        and re.search(r"\b(?:archivo|documento|txt|docx|word|pdf|markdown)\b", normalized)
    ):
        return DocumentOperation.CREATE_NEW
    rules = (
        (DocumentOperation.DELETE_EXISTING, r"\b(?:borr\w*|elimin\w*)\b"),
        (DocumentOperation.MOVE_EXISTING, r"\b(?:muev\w*|mov\w*)\b"),
        (DocumentOperation.COPY_EXISTING, r"\b(?:copi\w*|duplic\w*)\b"),
        (DocumentOperation.OVERWRITE_EXISTING, r"\b(?:sobrescrib\w*|reemplaza\s+todo)\b"),
        (DocumentOperation.APPEND_EXISTING, r"\b(?:agreg\w*|anad\w*)\b.*\b(?:linea|final)\b"),
        (DocumentOperation.TRANSFORM_EXISTING, r"\btransform\w*\b"),
        (DocumentOperation.READ_EXISTING, r"\b(?:lee\w*|revis\w*|inspeccion\w*)\b"),
    )
    for operation, pattern in rules:
        if re.search(pattern, normalized):
            return operation
    return DocumentOperation.REASONING_ONLY


def resolve_operation_intent(text: str) -> OperationIntent:
    operation = classify_document_operation(text)
    return OperationIntent(
        operation=operation,
        source_text=text,
        requires_existing_target=operation in {
            DocumentOperation.READ_EXISTING,
            DocumentOperation.EDIT_EXISTING,
            DocumentOperation.TRANSFORM_EXISTING,
            DocumentOperation.APPEND_EXISTING,
            DocumentOperation.OVERWRITE_EXISTING,
            DocumentOperation.COPY_EXISTING,
            DocumentOperation.MOVE_EXISTING,
            DocumentOperation.DELETE_EXISTING,
            DocumentOperation.CONVERT_EXISTING,
            DocumentOperation.RENAME_EXISTING,
        },
    )


def parse_semantic_transformation(text: str, target_path: str) -> DocumentTransformationSpec | None:
    if not target_path:
        return None
    normalized = _normalize(text)
    condition = _semantic_condition(text, normalized)
    if not condition:
        return None
    payload = _transformation_payload(text, normalized)
    if not payload:
        return None
    return DocumentTransformationSpec(
        target_path=str(Path(target_path).resolve(strict=False)),
        content_type=Path(target_path).suffix.casefold().removeprefix(".") or "text",
        iteration_unit="LINE",
        condition=asdict(SemanticPredicate(condition)),
        operation="APPEND_TO_MATCH",
        payload=payload,
        preserve_formatting=True,
    )


def build_semantic_patch_plan(
    content: str,
    specification: DocumentTransformationSpec,
    evaluator: SemanticLineEvaluator,
) -> PatchPlan:
    lines = content.splitlines()
    predicate = SemanticPredicate(str(specification.condition.get("description", "")).strip())
    selected = tuple(dict.fromkeys(int(value) for value in evaluator(predicate, tuple(lines))))
    if not selected or any(value < 1 or value > len(lines) for value in selected):
        raise ValueError("semantic_selection_invalid")
    patches = tuple(
        DocumentPatch(value, specification.operation, specification.payload)
        for value in selected
    )
    return PatchPlan(
        target_path=specification.target_path,
        before_hash=_hash_text(content),
        patches=patches,
        expected_changes=len(patches),
        selected_line_ids=selected,
    )


def apply_patch_plan(content: str, plan: PatchPlan) -> str:
    trailing_newline = content.endswith(("\n", "\r"))
    lines = content.splitlines()
    for patch in plan.patches:
        index = patch.line_id - 1
        if patch.operation != "APPEND_TO_MATCH":
            raise ValueError("unsupported_document_patch_operation")
        spacer = "" if not lines[index] or patch.text[:1].isspace() else " "
        lines[index] = f"{lines[index]}{spacer}{patch.text}"
    rendered = "\n".join(lines)
    return rendered + ("\n" if trailing_newline else "")


def verify_patch_plan(before: str, after: str, expected: str, plan: PatchPlan) -> PatchVerification:
    before_lines = before.splitlines()
    after_lines = after.splitlines()
    expected_lines = expected.splitlines()
    selected = set(plan.selected_line_ids)
    selected_match = all(
        line_id <= len(after_lines)
        and after_lines[line_id - 1] == expected_lines[line_id - 1]
        for line_id in selected
    )
    unselected_match = (
        len(before_lines) == len(after_lines)
        and all(
            before_lines[index] == after_lines[index]
            for index in range(len(before_lines))
            if index + 1 not in selected
        )
    )
    actual_changes = sum(
        left != right for left, right in zip(before_lines, after_lines)
    ) + abs(len(before_lines) - len(after_lines))
    before_hash = _hash_text(before)
    after_hash = _hash_text(after)
    passed = (
        before_hash == plan.before_hash
        and before_hash != after_hash
        and after == expected
        and actual_changes == plan.expected_changes
        and selected_match
        and unselected_match
    )
    return PatchVerification(
        "VERIFIED" if passed else "VERIFY_FAILED",
        before_hash,
        after_hash,
        plan.expected_changes,
        actual_changes,
        selected_match,
        unselected_match,
    )


def _semantic_condition(text: str, normalized: str) -> str:
    patterns = (
        r"(?:cada\s+vez\s+que|cuando)\s+(.+?)(?=\s+(?:escrib\w*|agreg\w*|anad\w*|pon\w*)\b|$)",
        r"(?:marca|marcar)\s+(.+)$",
        r"(?:a|en)\s+(?:las|los)\s+que\s+(.+?)(?=\s+(?:agreg\w*|anad\w*|pon\w*)\b|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized, re.IGNORECASE | re.DOTALL)
        if match:
            value = match.group(1).strip(" ,.;")
            if value:
                return value
    return ""


def _transformation_payload(text: str, normalized: str) -> str:
    quoted = re.findall(r"[\"']([^\"']+)[\"']", text)
    if quoted:
        return quoted[-1].strip()
    marker = re.search(r"\b(?:marca|marcar)\b", normalized)
    if marker:
        return "*"
    tail = re.search(
        r"\b(?:escrib\w*|agreg\w*|anad\w*|pon\w*)\b(?:\s+al\s+lado)?\s+(.+)$",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    return tail.group(1).strip(" ,.;") if tail else ""


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize(value: str) -> str:
    import unicodedata

    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))
