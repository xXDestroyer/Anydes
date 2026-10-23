from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import re
from typing import Any, Protocol
import unicodedata


@dataclass(frozen=True)
class ContentGenerationSpec:
    content_kind: str
    topic: str
    purpose: str
    target_length_words: int
    language: str
    tone: str
    structure: str
    constraints: tuple[str, ...]
    source_context: str = ""
    formatting_requirements: tuple[str, ...] = ()
    source_mode: str = "GENERATIVE"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ContentGenerationSpec":
        return cls(
            content_kind=str(value.get("content_kind", "expository_text")),
            topic=str(value.get("topic", "")).strip(),
            purpose=str(value.get("purpose", "document")),
            target_length_words=max(1, int(value.get("target_length_words", 300))),
            language=str(value.get("language", "es")),
            tone=str(value.get("tone", "natural")),
            structure=str(value.get("structure", "paragraphs")),
            constraints=tuple(str(item) for item in value.get("constraints", ())),
            source_context=str(value.get("source_context", "")),
            formatting_requirements=tuple(
                str(item) for item in value.get("formatting_requirements", ())
            ),
            source_mode=str(value.get("source_mode", "GENERATIVE")),
        )

    def with_source_context(self, content: str) -> "ContentGenerationSpec":
        return replace(self, source_context=str(content))


@dataclass(frozen=True)
class ContentValidationResult:
    status: str
    word_count: int
    minimum_words: int
    maximum_words: int
    topic_grounded: bool | None
    pseudo_tool_markup: bool
    placeholder_detected: bool
    reason: str = ""


class SemanticProseGenerator(Protocol):
    def __call__(self, specification: ContentGenerationSpec) -> str:
        ...


class ContentValidator:
    @staticmethod
    def validate(
        content: str,
        specification: ContentGenerationSpec,
        *,
        tolerance: float = 0.20,
    ) -> ContentValidationResult:
        return validate_generated_content(content, specification, tolerance=tolerance)


class PostActionVerifier:
    @staticmethod
    def generated_creation(
        actual: str,
        expected: str,
        specification: ContentGenerationSpec,
        *,
        generated_body: str | None = None,
    ) -> dict[str, Any]:
        # Body length belongs to generated content; presentation is validated separately.
        # Exact whole-document readback below proves this is the same body on disk.
        content_match = _collapse_document_text(actual) == _collapse_document_text(expected)
        validation = ContentValidator.validate(
            generated_body if generated_body is not None and content_match else actual, specification)
        return {
            **asdict(validation),
            "status": "VERIFIED" if content_match and validation.status == "VALID" else "VERIFY_FAILED",
            "content_match": content_match,
            "file_nonempty": bool(actual.strip()),
        }

    @staticmethod
    def generated_edit(
        before: str,
        actual: str,
        expected: str,
        generated: str,
    ) -> dict[str, Any]:
        before_hash = _hash_text(before)
        after_hash = _hash_text(actual)
        content_match = actual.rstrip() == expected.rstrip()
        generated_present = generated in actual
        verified = content_match and before_hash != after_hash and generated_present
        return {
            "status": "VERIFIED" if verified else "VERIFY_FAILED",
            "before_hash": before_hash,
            "after_hash": after_hash,
            "content_match": content_match,
            "generated_content_present": generated_present,
        }


_CONTENT_KINDS = (
    ("summary", r"\bresumen\b"),
    ("essay", r"\bensayo\b"),
    ("explanation", r"\bexplicacion\b"),
    ("analysis", r"\banalisis\b"),
    ("article", r"\barticulo\b"),
    ("letter", r"\bcarta\b"),
    ("conclusion", r"\bconclusion\b"),
)

_STOPWORDS = {
    "acerca", "como", "con", "de", "del", "el", "en", "es", "la", "las",
    "lo", "los", "para", "por", "que", "sobre", "su", "una", "unas", "uno",
    "unos", "y",
}

_PROTOCOL_MARKUP = re.compile(
    r"\[(?:official\.)?(?:document|filesystem|developer|archive)\.[^\]]+\]"
    r"|<tool_call\b|\"tool_id\"\s*:|\"action_type\"\s*:",
    re.IGNORECASE,
)

_PLACEHOLDER = re.compile(
    r"\b(?:lorem ipsum|todo:|tbd|placeholder|insertar (?:texto|contenido) aqui)\b"
    r"|\[(?:completar|insertar)[^\]]*\]",
    re.IGNORECASE,
)


def parse_content_generation_spec(
    text: str,
    *,
    purpose: str,
    default_words: int = 300,
    allow_open_content: bool = True,
) -> ContentGenerationSpec | None:
    normalized = _normalize(text)
    self_description = _self_description_request(normalized)
    content_kind = ""
    for candidate, pattern in _CONTENT_KINDS:
        if re.search(pattern, normalized):
            content_kind = candidate
            break
    length_match = re.search(
        r"\b(?:de\s+)?(?:unas?|unos?|aproximadamente|alrededor\s+de)?\s*(\d{2,5})\s+palabras\b",
        normalized,
    )
    open_content = re.search(
        r"\b(?:donde\s+(?:este|incluya|contenga|lleve)|"
        r"que\s+(?:incluya|contenga|lleve)|"
        r"en\s+(?:el|la)\s+que)\b\s+(.+?)"
        r"(?=\s+(?:y\s+ponle|con\s+nombre|llamad[oa])\b|$|[?.!](?:\s|$))",
        normalized,
    ) if allow_open_content else None
    if self_description:
        content_kind = "SELF_DESCRIPTION"
    if not content_kind and length_match is None and open_content is None:
        return None
    target_length = int(length_match.group(1)) if length_match else default_words
    if target_length < 30 or target_length > 5000:
        raise ValueError("semantic_prose_length_out_of_range")
    topic = "Anydes" if self_description else _extract_topic(text, content_kind)
    if not topic and open_content is not None:
        topic = open_content.group(1).strip(" \t\r\n.,;:!?\"")
    if not topic and content_kind != "conclusion":
        return None
    source_mode = "SELF_CONTEXT" if self_description else _source_mode(normalized, topic)
    constraints = [
        "no_tool_markup",
        "no_placeholders",
        "stay_on_topic",
        "respect_approximate_length",
    ]
    if source_mode != "GENERATIVE":
        constraints.extend(("do_not_claim_retrieval", "do_not_invent_sources"))
    return ContentGenerationSpec(
        content_kind=content_kind or "expository_text",
        topic=topic,
        purpose=purpose,
        target_length_words=target_length,
        language="es",
        tone="natural",
        structure="paragraphs",
        constraints=tuple(constraints),
        source_mode=source_mode,
    )


def _source_mode(normalized: str, topic: str) -> str:
    if re.search(
        r"\b(?:te|a\s+ti)\s+(?:describ\w*|present\w*)\b|"
        r"\b(?:sobre|acerca\s+de)\s+ti\s+misma\b",
        normalized,
    ):
        return "SELF_CONTEXT"
    if re.search(r"\b(?:actual|actualizada|oficial|hoy|ultima|reciente|vigente)\b", normalized):
        return "FACTUAL_REQUIRES_RETRIEVAL"
    factual_markers = r"\b(?:vision|mision|historia|requisitos|normativa|datos)\b"
    if re.search(factual_markers, _normalize(topic)):
        return "FACTUAL_FROM_AVAILABLE_CONTEXT"
    return "GENERATIVE"


def _self_description_request(normalized: str) -> bool:
    """Recognize a request for the assistant to describe itself as a content source."""

    reflexive_action = re.search(
        r"\b(?:(?:describ|present)\w*(?:te|se)|"
        r"(?:te|se)\s+(?:describ|present)\w*)\b"
        r"(?:\s+a\s+(?:ti|usted)\s+mism[ao])?",
        normalized,
    )
    explicit_subject = re.search(
        r"\b(?:descripcion|presentacion|perfil|semblanza)\s+de\s+"
        r"(?:ti|anydes)(?:\s+mism[ao])?\b",
        normalized,
    )
    first_person_instruction = re.search(
        r"\b(?:habla|escribe|redacta)\w*\s+(?:sobre|acerca\s+de)\s+"
        r"(?:ti|anydes)(?:\s+mism[ao])?\b",
        normalized,
    )
    return bool(reflexive_action or explicit_subject or first_person_instruction)


def validate_generated_content(
    content: str,
    specification: ContentGenerationSpec,
    *,
    tolerance: float = 0.20,
) -> ContentValidationResult:
    rendered = str(content or "").strip()
    count = len(re.findall(r"\b[^\W_]+(?:[-'][^\W_]+)*\b", rendered, re.UNICODE))
    minimum = max(1, int(specification.target_length_words * (1.0 - tolerance)))
    maximum = max(minimum, int(specification.target_length_words * (1.0 + tolerance)))
    protocol = bool(_PROTOCOL_MARKUP.search(rendered))
    placeholder = bool(_PLACEHOLDER.search(_normalize(rendered)))
    # Semantic alignment belongs to TaskAdherence, not lexical overlap in a precheck.
    grounded = None
    reason = ""
    if not rendered:
        reason = "generated_content_empty"
    elif protocol:
        reason = "generated_content_contains_tool_markup"
    elif placeholder:
        reason = "generated_content_contains_placeholder"
    elif count < minimum or count > maximum:
        reason = "generated_content_length_out_of_tolerance"
    return ContentValidationResult(
        status="VALID" if not reason else "INVALID",
        word_count=count,
        minimum_words=minimum,
        maximum_words=maximum,
        topic_grounded=grounded,
        pseudo_tool_markup=protocol,
        placeholder_detected=placeholder,
        reason=reason,
    )


def _extract_topic(text: str, content_kind: str) -> str:
    topic = re.search(
        r"\b(?:sobre|acerca\s+de)\s+(.+?)(?=$|[?.!](?:\s|$))",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if topic:
        return topic.group(1).strip(" \t\r\n.,;:!?\"")
    if content_kind == "conclusion":
        return "contenido del documento"
    return ""


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value).casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _hash_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _collapse_document_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()
