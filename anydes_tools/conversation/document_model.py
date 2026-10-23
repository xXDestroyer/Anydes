from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
import re
from typing import Any


class DocumentEditOperation(StrEnum):
    APPEND_PARAGRAPH = "APPEND_PARAGRAPH"
    PREPEND_PARAGRAPH = "PREPEND_PARAGRAPH"
    REPLACE_TEXT = "REPLACE_TEXT"
    INSERT_AFTER = "INSERT_AFTER"
    INSERT_BEFORE = "INSERT_BEFORE"


@dataclass(frozen=True)
class DocumentRun:
    text: str
    bold: bool = False
    italic: bool = False


@dataclass(frozen=True)
class DocumentModel:
    paragraphs: tuple[str, ...]
    headings: tuple[str, ...] = ()
    lists: tuple[tuple[str, ...], ...] = ()
    tables: tuple[dict[str, Any], ...] = ()
    runs: tuple[tuple[DocumentRun, ...], ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_extracted_text(cls, text: str) -> "DocumentModel":
        return cls(tuple(str(text).splitlines()))

    def render_text(self) -> str:
        return "\n".join(self.paragraphs)


@dataclass(frozen=True)
class DocumentEditSpec:
    target_path: str
    operation: DocumentEditOperation
    payload: str
    anchor: str = ""
    preserve_structure: bool = True

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["operation"] = self.operation.value
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DocumentEditSpec":
        return cls(
            target_path=str(value.get("target_path", "")),
            operation=DocumentEditOperation(str(value.get("operation", "APPEND_PARAGRAPH"))),
            payload=str(value.get("payload", "")),
            anchor=str(value.get("anchor", "")),
            preserve_structure=bool(value.get("preserve_structure", True)),
        )


def parse_document_edit_spec(text: str, target_path: str) -> DocumentEditSpec | None:
    payload = _extract_positional_payload(text, "final")
    if payload:
        return DocumentEditSpec(target_path, DocumentEditOperation.APPEND_PARAGRAPH, payload)
    payload = _extract_positional_payload(text, "inicio")
    if payload:
        return DocumentEditSpec(target_path, DocumentEditOperation.PREPEND_PARAGRAPH, payload)
    replacement = re.search(
        r"\breemplaza\s+[\"'](.+?)[\"']\s+por\s+[\"'](.*?)[\"']",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if replacement:
        return DocumentEditSpec(
            target_path,
            DocumentEditOperation.REPLACE_TEXT,
            replacement.group(2),
            replacement.group(1),
        )
    inserted = re.search(
        r"\b(?:despues|después|antes)\s+de\s+[\"'](.+?)[\"']\s+"
        r"(?:pon|agrega|anade|añade|inserta)\w*\s+[\"']?(.+?)[\"']?\s*$",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if inserted:
        operation = (
            DocumentEditOperation.INSERT_AFTER
            if inserted.group(0).casefold().lstrip().startswith(("despues", "después"))
            else DocumentEditOperation.INSERT_BEFORE
        )
        return DocumentEditSpec(
            target_path,
            operation,
            inserted.group(2).strip(" .\"'"),
            inserted.group(1),
        )
    return None


def apply_document_edit(model: DocumentModel, specification: DocumentEditSpec) -> DocumentModel:
    paragraphs = list(model.paragraphs)
    if specification.operation == DocumentEditOperation.APPEND_PARAGRAPH:
        paragraphs.append(specification.payload)
    elif specification.operation == DocumentEditOperation.PREPEND_PARAGRAPH:
        paragraphs.insert(0, specification.payload)
    elif specification.operation == DocumentEditOperation.REPLACE_TEXT:
        rendered = model.render_text()
        if specification.anchor not in rendered:
            raise ValueError("document_edit_anchor_missing")
        paragraphs = rendered.replace(specification.anchor, specification.payload).splitlines()
    elif specification.operation in {
        DocumentEditOperation.INSERT_AFTER,
        DocumentEditOperation.INSERT_BEFORE,
    }:
        matching = [index for index, paragraph in enumerate(paragraphs) if specification.anchor in paragraph]
        if len(matching) != 1:
            raise ValueError("document_edit_anchor_not_unique")
        offset = 1 if specification.operation == DocumentEditOperation.INSERT_AFTER else 0
        paragraphs.insert(matching[0] + offset, specification.payload)
    else:
        raise ValueError("document_edit_operation_not_implemented")
    return DocumentModel(
        tuple(paragraphs), model.headings, model.lists, model.tables, model.runs, model.metadata
    )


def verify_document_edit(before: str, after: str, expected: str, specification: DocumentEditSpec) -> dict[str, Any]:
    content_match = after == expected
    prior_preserved = all(paragraph in after for paragraph in before.splitlines() if paragraph)
    payload_present = specification.payload in after
    placement = True
    if specification.operation == DocumentEditOperation.APPEND_PARAGRAPH:
        placement = after.rstrip().endswith(specification.payload.rstrip())
    elif specification.operation == DocumentEditOperation.PREPEND_PARAGRAPH:
        placement = after.lstrip().startswith(specification.payload.lstrip())
    elif specification.operation in {
        DocumentEditOperation.INSERT_AFTER,
        DocumentEditOperation.INSERT_BEFORE,
    }:
        lines = after.splitlines()
        anchor_lines = [index for index, line in enumerate(lines) if specification.anchor in line]
        if len(anchor_lines) != 1:
            placement = False
        else:
            expected_index = anchor_lines[0] + (
                1 if specification.operation == DocumentEditOperation.INSERT_AFTER else -1
            )
            placement = 0 <= expected_index < len(lines) and lines[expected_index] == specification.payload
    passed = content_match and prior_preserved and payload_present and placement and before != after
    return {
        "status": "VERIFIED" if passed else "VERIFY_FAILED",
        "content_match": content_match,
        "prior_content_preserved": prior_preserved,
        "payload_present": payload_present,
        "placement_correct": placement,
    }


def _extract_positional_payload(text: str, position: str) -> str:
    patterns = (
        rf"\b(?:y\s+)?al\s+{position}\s+(?:pon|agrega|anade|añade|escribe)\w*\s+(.+)$",
        rf"\b(?:pon|agrega|anade|añade|escribe)\w*\s+(?:al\s+)?{position}\s+(.+)$",
        rf"\b(?:pon|agrega|anade|añade|escribe)\w*\s+(.+?)\s+al\s+{position}\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip(" \t\r\n.,;:\"'")
    return ""
