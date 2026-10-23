from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
import re
from typing import Any


@dataclass(frozen=True)
class DocumentConversionSpec:
    source_path: str
    target_path: str
    source_format: str
    target_format: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DocumentConversionSpec":
        return cls(
            source_path=str(value.get("source_path", "")),
            target_path=str(value.get("target_path", "")),
            source_format=str(value.get("source_format", "")),
            target_format=str(value.get("target_format", "")),
        )


@dataclass(frozen=True)
class ConversionVerification:
    status: str
    source_nonempty: bool
    target_nonempty: bool
    content_similarity: float
    target_format_correct: bool


class SourceResolver:
    @staticmethod
    def resolve(
        *,
        explicit_path: str,
        explicit_filename: str,
        explicit_stem: str,
        active_document: str,
        recent_files: list[str],
        active_workspace: str,
    ) -> str:
        if explicit_path:
            path = Path(explicit_path)
            if not path.is_absolute():
                path = Path(active_workspace) / path
            return str(path.resolve(strict=False))
        if explicit_filename:
            local = (Path(active_workspace) / explicit_filename).resolve(strict=False)
            if local.exists():
                return str(local)
            matches = [Path(item).resolve(strict=False) for item in reversed(recent_files) if Path(item).name.casefold() == explicit_filename.casefold()]
            return str(matches[0] if matches else local)
        if explicit_stem:
            matches = [
                Path(item).resolve(strict=False)
                for item in reversed(recent_files)
                if Path(item).stem.casefold() == explicit_stem.casefold()
            ]
            local_matches = [
                item for item in Path(active_workspace).glob(f"{explicit_stem}.*")
                if item.is_file()
            ]
            if local_matches:
                return str(local_matches[0].resolve(strict=False))
            if matches:
                return str(matches[0])
            return ""
        if active_document:
            return str(Path(active_document).resolve(strict=False))
        documents = [
            item for item in reversed(recent_files)
            if Path(item).suffix.casefold() in {".txt", ".md", ".docx", ".pdf"}
        ]
        return str(Path(documents[0]).resolve(strict=False)) if documents else ""


def verify_conversion(
    source_text: str,
    target_text: str,
    target_path: str,
    target_format: str,
) -> ConversionVerification:
    source = _collapse(source_text)
    target = _collapse(target_text)
    similarity = SequenceMatcher(None, source, target).ratio() if source and target else 0.0
    correct = Path(target_path).suffix.casefold() == f".{target_format.casefold()}"
    passed = bool(source and target and correct and similarity >= 0.90)
    return ConversionVerification(
        "VERIFIED" if passed else "VERIFY_FAILED",
        bool(source),
        bool(target),
        round(similarity, 6),
        correct,
    )


def extract_source_stem(text: str) -> str:
    match = re.search(
        r"\b(?:archivo|documento)\s+([A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9_-]+)"
        r"\s+(?:a|en)\s+(?:un\s+)?(?:pdf|docx|txt|markdown)\b",
        text,
        re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


def _collapse(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().casefold()
