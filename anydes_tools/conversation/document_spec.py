from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any
import unicodedata

from .content_generation import ContentGenerationSpec


@dataclass(frozen=True)
class DocumentSpec:
    title: str
    paragraphs: tuple[str, ...]
    headings: tuple[str, ...] = ()
    lists: tuple[tuple[str, ...], ...] = ()
    tables: tuple[dict[str, Any], ...] = ()
    spacing: str = "normal"
    emphasis: tuple[dict[str, str], ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)
    include_title: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RenderedDocument:
    content: str
    title: str
    expected_extractable_text: str
    format_id: str


class DocumentRenderer:
    @staticmethod
    def render(specification: DocumentSpec, format_id: str) -> RenderedDocument:
        body = "\n\n".join(specification.paragraphs).strip()
        title = specification.title.strip() if specification.include_title else ""
        if format_id == "markdown":
            content = f"# {title}\n\n{body}" if title else body
            return RenderedDocument(content, "", content, format_id)
        if format_id == "txt":
            content = f"{title}\n\n{body}" if title else body
            return RenderedDocument(content, "", content, format_id)
        expected = f"{title}\n{body}" if title else body
        return RenderedDocument(body, title, expected, format_id)


def build_document_spec(
    content: str,
    generation: ContentGenerationSpec,
    *,
    requested_title: str,
    request_text: str,
) -> DocumentSpec:
    normalized = _normalize(request_text)
    include_title = not bool(re.search(r"\b(?:sin\s+titulo|solo\s+el\s+texto)\b", normalized))
    title = requested_title.strip() or generation.topic.strip()
    paragraphs = _paragraphize(str(content).strip())
    return DocumentSpec(
        title=title,
        paragraphs=paragraphs,
        spacing="normal",
        metadata={
            "language": generation.language,
            "content_kind": generation.content_kind,
            "topic": generation.topic,
        },
        include_title=include_title,
    )


def document_spec_from_extracted_text(content: str) -> DocumentSpec:
    return DocumentSpec(
        title="",
        paragraphs=_paragraphize(str(content).strip(), max_words=140),
        include_title=False,
    )


def _paragraphize(content: str, *, max_words: int = 120) -> tuple[str, ...]:
    explicit = tuple(part.strip() for part in re.split(r"\n\s*\n", content) if part.strip())
    if len(explicit) > 1:
        return explicit
    words = content.split()
    if not words:
        return ()
    return tuple(
        " ".join(words[index:index + max_words])
        for index in range(0, len(words), max_words)
    )


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))
