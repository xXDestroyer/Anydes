from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import string
import unicodedata
from typing import Any, Callable, Protocol, Sequence


@dataclass(frozen=True)
class TextCreationSpec:
    source_kind: str
    values: tuple[str, ...]
    separator: str = "\n"
    layout: str = "lines"
    suffix: str = ""
    suggested_stem: str = "nota"
    conditions: tuple[dict[str, Any], ...] = ()

    def render(self) -> str:
        if self.layout == "suffix_lines":
            return "\n".join(f"{value} {self.suffix}".rstrip() for value in self.values)
        if self.layout == "inline":
            return self.separator.join(self.values)
        return "\n".join(self.values)

    def payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "TextCreationSpec":
        return cls(
            source_kind=str(payload.get("source_kind", "literal")),
            values=tuple(str(item) for item in payload.get("values", ())),
            separator=str(payload.get("separator", "\n")),
            layout=str(payload.get("layout", "lines")),
            suffix=str(payload.get("suffix", "")),
            suggested_stem=str(payload.get("suggested_stem", "nota")),
            conditions=tuple(dict(item) for item in payload.get("conditions", ())),
        )


@dataclass(frozen=True)
class TextPredicate:
    scope: str
    operator: str
    value: str
    case_sensitive: bool = False


@dataclass(frozen=True)
class TextMutation:
    operation: str
    text: str = ""
    replacement: str = ""


@dataclass(frozen=True)
class TextTransformSpec:
    predicate: TextPredicate
    mutation: TextMutation

    def payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "TextTransformSpec":
        predicate = dict(payload.get("predicate", {}))
        mutation = dict(payload.get("mutation", {}))
        return cls(
            predicate=TextPredicate(
                scope=str(predicate.get("scope", "token")),
                operator=str(predicate.get("operator", "contains")),
                value=str(predicate.get("value", "")),
                case_sensitive=bool(predicate.get("case_sensitive", False)),
            ),
            mutation=TextMutation(
                operation=str(mutation.get("operation", "")),
                text=str(mutation.get("text", "")),
                replacement=str(mutation.get("replacement", "")),
            ),
        )


@dataclass(frozen=True)
class TransformOutcome:
    content: str
    expected_matches: int
    applied_changes: int
    changed: bool


@dataclass(frozen=True)
class ContentSourceDefinition:
    source_kind: str
    patterns: tuple[str, ...]
    values_factory: Callable[[], tuple[str, ...]]
    suggested_stem: str
    item_label: str = "elemento"


@dataclass(frozen=True)
class SemanticContentRequest:
    description: str
    semantic_type: str
    count: int
    uniqueness: bool = True


class SemanticContentGenerator(Protocol):
    def __call__(self, request: SemanticContentRequest) -> Sequence[str]:
        ...


_CONTENT_SOURCES = (
    ContentSourceDefinition(
        source_kind="ordered_symbols.latin_lowercase",
        patterns=(r"\bletras?\s+del\s+abecedario\b", r"\babecedario\b"),
        values_factory=lambda: tuple(string.ascii_lowercase),
        suggested_stem="abecedario",
    ),
)

_COUNT_WORDS = {
    "un": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4,
    "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9,
    "diez": 10,
}


class ContentBuilder:
    """Build bounded document content from a typed, declarative source grammar."""

    def __init__(
        self,
        semantic_generator: SemanticContentGenerator | None = None,
        semantic_evaluator=None,
    ) -> None:
        self.semantic_generator = semantic_generator
        self.semantic_evaluator = semantic_evaluator
        self._extractors = (
            self._numeric_range,
            self._registered_sequence,
            self._explicit_items,
            self._headings,
            self._semantic_sequence,
        )

    def build(self, text: str, literal_content: str = "") -> TextCreationSpec:
        if literal_content:
            return TextCreationSpec('literal', (literal_content,), '', 'inline', suggested_stem='nota')
        normalized = _normalize(text)
        for extractor in self._extractors:
            source = extractor(text, normalized)
            if source is not None:
                return self._apply_layout(source, text, normalized)
        return TextCreationSpec(
            "literal",
            (literal_content,),
            "",
            "inline",
            suggested_stem="nota",
        )

    def _numeric_range(self, text: str, normalized: str) -> TextCreationSpec | None:
        match = re.search(
            r"(?:"
            r"enumer\w*(?:\s+(?:desde|del))?"
            r"|numeros?(?:\s+(?:desde|del))?"
            r"|desde|del"
            r")\s+(-?\d+)\s+(?:hasta|al)\s+(-?\d+)",
            normalized,
        )
        if match is None:
            return None
        start, end = int(match.group(1)), int(match.group(2))
        if abs(end - start) > 5000:
            raise ValueError("structured_range_too_large")
        step = 1 if end >= start else -1
        values = tuple(str(value) for value in range(start, end + step, step))
        return TextCreationSpec("range", values, suggested_stem="numeros")

    def _explicit_items(self, text: str, normalized: str) -> TextCreationSpec | None:
        del normalized
        match = re.search(
            r"(?:\b(?:con\s+)?palabras?\s*[:=]?\s*|\bcon\s+)"
            r"([^\r\n]+?)\s+(?:separad\w*|unid\w*)\s+(?:por|con)\s+",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if match is None:
            return None
        candidate = match.group(1).strip()
        if "," not in candidate and not re.match(r"^(?:\S+\s+){1,20}\S+$", candidate):
            return None
        values = tuple(
            item.strip(" \t\r\n\"'")
            for item in re.split(r"\s*,\s*|\s+", candidate)
            if item.strip(" \t\r\n\"'")
        )
        return TextCreationSpec("items", values, suggested_stem="lista") if values else None

    def _registered_sequence(self, text: str, normalized: str) -> TextCreationSpec | None:
        for definition in _CONTENT_SOURCES:
            if any(re.search(pattern, normalized) for pattern in definition.patterns):
                values = definition.values_factory()
                return TextCreationSpec(
                    definition.source_kind,
                    values,
                    suggested_stem=definition.suggested_stem,
                )
        return None

    def _semantic_sequence(self, text: str, normalized: str) -> TextCreationSpec | None:
        match = re.search(
            r"\b(\d+)\s+([a-záéíóúüñ][a-záéíóúüñ _-]{1,60}?)"
            r"(?:\s+diferentes?|\s+distint\w*)?"
            r"(?=\s*[,.;]|\s+(?:uno|una|cada|separad\w*|y\s+si|y\s+cuando)|$)",
            text,
            re.IGNORECASE,
        )
        if match is None:
            return None
        count = int(match.group(1))
        if count < 1 or count > 200:
            raise ValueError("semantic_source_count_out_of_range")
        if self.semantic_generator is None:
            raise ValueError("semantic_content_generator_unavailable")
        semantic_type = re.sub(r"\s+", " ", match.group(2)).strip(" ,.;")
        request = SemanticContentRequest(
            description=text.strip(),
            semantic_type=semantic_type,
            count=count,
            uniqueness=True,
        )
        values = tuple(str(value).strip() for value in self.semantic_generator(request))
        if len(values) != count or any(not value for value in values):
            raise ValueError("semantic_content_count_mismatch")
        normalized_values = tuple(_normalize(value) for value in values)
        if len(set(normalized_values)) != count:
            raise ValueError("semantic_content_not_unique")
        stem = re.sub(r"[^a-z0-9_-]+", "_", _normalize(semantic_type)).strip("_")
        generated = TextCreationSpec(
            "semantic_list",
            values,
            suggested_stem=stem or "lista",
            conditions=({
                "kind": "SEMANTIC_GENERATION",
                "description": text.strip(),
                "count": count,
                "uniqueness": True,
            },),
        )
        return self._apply_conditional_annotations(generated, text)

    def _apply_conditional_annotations(
        self,
        source: TextCreationSpec,
        text: str,
    ) -> TextCreationSpec:
        deterministic = parse_transform_spec(text)
        if deterministic is not None:
            rendered: list[str] = []
            matches: list[int] = []
            for index, value in enumerate(source.values, 1):
                outcome = transform_text(value, deterministic)
                rendered.append(outcome.content)
                if outcome.expected_matches:
                    matches.append(index)
            return TextCreationSpec(
                source.source_kind,
                tuple(rendered),
                source.separator,
                source.layout,
                source.suffix,
                source.suggested_stem,
                (*source.conditions, {
                    "kind": "DETERMINISTIC_PREDICATE",
                    "predicate": asdict(deterministic.predicate),
                    "mutation": asdict(deterministic.mutation),
                    "expected_match_ids": matches,
                    "base_values": list(source.values),
                }),
            )
        if self.semantic_evaluator is None:
            return source
        from .document_transform import parse_semantic_transformation, SemanticPredicate

        semantic = parse_semantic_transformation(text, "semantic-list.txt")
        if semantic is None:
            return source
        predicate = SemanticPredicate(str(semantic.condition.get("description", "")))
        selected = tuple(dict.fromkeys(
            int(value) for value in self.semantic_evaluator(predicate, source.values)
        ))
        if any(index < 1 or index > len(source.values) for index in selected):
            raise ValueError("semantic_selection_invalid")
        rendered = list(source.values)
        for index in selected:
            value = rendered[index - 1]
            spacer = "" if not value or semantic.payload[:1].isspace() else " "
            rendered[index - 1] = f"{value}{spacer}{semantic.payload}"
        return TextCreationSpec(
            source.source_kind,
            tuple(rendered),
            source.separator,
            source.layout,
            source.suffix,
            source.suggested_stem,
            (*source.conditions, {
                "kind": "SEMANTIC_PREDICATE",
                "description": predicate.description,
                "expected_match_ids": list(selected),
                "base_values": list(source.values),
                "payload": semantic.payload,
            }),
        )

    def _headings(self, text: str, normalized: str) -> TextCreationSpec | None:
        del text
        match = re.search(r"\b(\d+|[a-z]+)\s+titulos?\b", normalized)
        if match is None:
            return None
        raw_count = match.group(1)
        count = int(raw_count) if raw_count.isdigit() else _COUNT_WORDS.get(raw_count, 0)
        if count < 1 or count > 50:
            return None
        values = tuple(f"# Título {index}" for index in range(1, count + 1))
        return TextCreationSpec("headings", values, suggested_stem="titulos")

    @staticmethod
    def _apply_layout(
        source: TextCreationSpec,
        text: str,
        normalized: str,
    ) -> TextCreationSpec:
        separator = _separator(text)
        if _one_item_per_line(normalized):
            return TextCreationSpec(
                source.source_kind,
                source.values,
                "\n",
                "lines",
                suggested_stem=source.suggested_stem,
                conditions=source.conditions,
            )
        if _requests_suffix_layout(normalized) and separator:
            return TextCreationSpec(
                source.source_kind,
                source.values,
                "\n",
                "suffix_lines",
                separator,
                source.suggested_stem,
                source.conditions,
            )
        if _requests_separator(normalized) or separator:
            return TextCreationSpec(
                source.source_kind,
                source.values,
                separator or " ",
                "inline",
                suggested_stem=source.suggested_stem,
                conditions=source.conditions,
            )
        return source


def parse_creation_spec(
    text: str,
    literal_content: str = "",
    *,
    builder: ContentBuilder | None = None,
) -> TextCreationSpec:
    return (builder or ContentBuilder()).build(text, literal_content)


def parse_transform_spec(text: str) -> TextTransformSpec | None:
    normalized = _normalize(text)
    replacement = _parse_replacement(text)
    if replacement is not None:
        old, new = replacement
        return TextTransformSpec(
            TextPredicate("document", "contains", old),
            TextMutation("replace", replacement=new),
        )

    scope, operator, condition = _deterministic_condition(text, normalized)
    if not condition:
        return None
    prefix = _quoted_after(text, r"(?:prefijo|anteponer|antepone)")
    if prefix is not None:
        return TextTransformSpec(
            TextPredicate(scope, operator, condition),
            TextMutation("prefix", text=prefix),
        )
    suffix = _quoted_after(text, r"(?:sufijo|anadir|añadir|agregar|agrega|escribir|escribe|tenga|tuviera)")
    if suffix is None:
        suffix = _payload_before_condition(text)
    if suffix is None:
        suffix = _tail_after_mutation(text)
    if suffix:
        return TextTransformSpec(
            TextPredicate(scope, operator, condition),
            TextMutation("append_text", text=suffix),
        )
    return None


def transform_text(content: str, spec: TextTransformSpec) -> TransformOutcome:
    if spec.mutation.operation == "replace" and spec.predicate.scope == "document":
        count = _count(content, spec.predicate.value, spec.predicate.case_sensitive)
        updated = _replace_all(
            content,
            spec.predicate.value,
            spec.mutation.replacement,
            spec.predicate.case_sensitive,
        )
        return TransformOutcome(updated, count, count, updated != content)

    if spec.predicate.scope == "line":
        return _transform_lines(content, spec)
    return _transform_tokens(content, spec)


def verify_creation(
    actual: str,
    expected: str,
    file_type: str,
    specification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if file_type in {"txt", "md", "docx"}:
        matched = _normalize_newlines(actual) == _normalize_newlines(expected)
    else:
        matched = _collapse_space(actual) == _collapse_space(expected)
    expected_items = int((specification or {}).get("item_count", 0) or 0)
    layout = str((specification or {}).get("layout", ""))
    separator = str((specification or {}).get("separator", ""))
    actual_items = _actual_item_count(actual, layout, separator) if expected_items else 0
    layout_match = not expected_items or actual_items == expected_items
    expected_match_ids: list[int] = []
    actual_match_ids: list[int] = []
    for condition in (specification or {}).get("conditions", ()):
        if condition.get("kind") != "DETERMINISTIC_PREDICATE":
            continue
        expected_match_ids = [int(value) for value in condition.get("expected_match_ids", ())]
        mutation = dict(condition.get("mutation", {}))
        marker = str(mutation.get("text") or mutation.get("replacement") or "").strip()
        if marker:
            actual_match_ids = [
                index for index, line in enumerate(actual.splitlines(), 1)
                if marker in line
            ]
    predicate_match = expected_match_ids == actual_match_ids
    return {
        "status": "VERIFIED" if matched and layout_match and predicate_match else "VERIFY_FAILED",
        "expected_characters": len(expected),
        "actual_characters": len(actual),
        "content_match": matched,
        "expected_items": expected_items,
        "actual_items": actual_items,
        "layout_match": layout_match,
        "expected_match_ids": expected_match_ids,
        "actual_match_ids": actual_match_ids,
        "predicate_match": predicate_match,
    }


def verify_transform(actual: str, expected: str, outcome: TransformOutcome) -> dict[str, Any]:
    matched = _normalize_newlines(actual) == _normalize_newlines(expected)
    return {
        "status": "VERIFIED" if matched and outcome.changed else "VERIFY_FAILED",
        "expected_matches": outcome.expected_matches,
        "applied_changes": outcome.applied_changes,
        "content_match": matched,
        "changed": outcome.changed,
    }


def _transform_tokens(content: str, spec: TextTransformSpec) -> TransformOutcome:
    pattern = re.compile(r"\d+" if spec.predicate.scope == "number" else r"[^\W_]+", re.UNICODE)
    matches = 0

    def apply(match: re.Match[str]) -> str:
        nonlocal matches
        token = match.group(0)
        if not _predicate_matches(token, spec.predicate):
            return token
        matches += 1
        return _mutate(token, spec.mutation)

    updated = pattern.sub(apply, content)
    return TransformOutcome(updated, matches, matches, updated != content)


def _transform_lines(content: str, spec: TextTransformSpec) -> TransformOutcome:
    chunks = content.splitlines(keepends=True)
    matches = 0
    output: list[str] = []
    for chunk in chunks:
        ending_match = re.search(r"(\r\n|\n|\r)$", chunk)
        ending = ending_match.group(1) if ending_match else ""
        line = chunk[:-len(ending)] if ending else chunk
        if _predicate_matches(line, spec.predicate):
            matches += 1
            line = _mutate(line, spec.mutation)
        output.append(line + ending)
    updated = "".join(output)
    return TransformOutcome(updated, matches, matches, updated != content)


def _mutate(value: str, mutation: TextMutation) -> str:
    if mutation.operation == "prefix":
        return mutation.text + value
    if mutation.operation == "append_text":
        spacer = "" if not value or value[-1].isspace() or mutation.text[:1].isspace() else " "
        return value + spacer + mutation.text
    if mutation.operation == "replace":
        return mutation.replacement
    raise ValueError(f"unsupported_text_mutation:{mutation.operation}")


def _predicate_matches(value: str, predicate: TextPredicate) -> bool:
    haystack = value if predicate.case_sensitive else value.casefold()
    needle = predicate.value if predicate.case_sensitive else predicate.value.casefold()
    if predicate.operator == "contains":
        return needle in haystack
    if predicate.operator == "equals":
        return haystack == needle
    if predicate.operator == "starts_with":
        return haystack.startswith(needle)
    if predicate.operator == "ends_with":
        return haystack.endswith(needle)
    raise ValueError(f"unsupported_text_predicate:{predicate.operator}")


def _separator(text: str) -> str:
    patterns = (
        r"(?:separad\w*|separador|usando|usar|con)\s+(?:por\s+)?[\"']([^\"'\r\n]+)[\"']",
        r"[\"']([^\"'\r\n]+)[\"']\s+como\s+(?:separador|espacio)",
        r"(?:separad\w*|separador|usando|usar)\s+(?:por\s+)?([,;|:\-])(?:\s|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def _requests_separator(normalized: str) -> bool:
    return (
        "separador" in normalized
        or bool(re.search(r"\bseparad\w*\s+por\b", normalized))
        or bool(re.search(r"\b(?:usando|usar)\s+[\"']?[,;|:\-]", normalized))
    )


def _requests_suffix_layout(normalized: str) -> bool:
    return "como espacio" in normalized or "al final" in normalized


def _one_item_per_line(normalized: str) -> bool:
    patterns = (
        r"\b(?:cada|por)\s+linea\b",
        r"\b(?:una|uno|un item|cada item)\s+por\s+linea\b",
        r"\bsalto\s+de\s+linea\b",
        r"\bseparad\w*\s+(?:con|por)\s+(?:un\s+)?enter\b",
        r"\ben\s+cada\s+linea\b",
    )
    return any(re.search(pattern, normalized) for pattern in patterns)


def _scope(normalized: str) -> str:
    match = re.search(r"\b(numero|token|palabra|linea)s?\b", normalized)
    if not match:
        return "token"
    return {"numero": "number", "linea": "line"}.get(match.group(1), "token")


def _condition_value(text: str) -> str:
    match = re.search(
        r"\b(?:conteng\w*|incluy\w*|veas?|encuentr\w*|aparezc\w*)\s+"
        r"(?:el\s+texto\s+|el\s+numero\s+|un\s+|una\s+|el\s+|la\s+)?"
        r"(?:[\"']([^\"']+)[\"']|([^,.;\s]+))",
        text,
        re.IGNORECASE,
    )
    return (match.group(1) or match.group(2)).strip() if match else ""


def _deterministic_condition(text: str, normalized: str) -> tuple[str, str, str]:
    starts = re.search(
        r"\b(?:palabras?|tokens?|lineas?)?\s*(?:que\s+)?"
        r"(?:empiez\w*|comienz\w*)\s+(?:por|con)\s+"
        r"(?:[\"']([^\"']+)[\"']|([^,.;\s]+))",
        text,
        re.IGNORECASE,
    )
    if starts:
        value = (starts.group(1) or starts.group(2)).strip()
        return _scope(normalized), "starts_with", value
    ends = re.search(
        r"\b(?:palabras?|tokens?|lineas?)?\s*(?:que\s+)?"
        r"(?:termin\w*|acab\w*)\s+(?:por|con|en)\s+"
        r"(?:[\"']([^\"']+)[\"']|([^,.;\s]+))",
        text,
        re.IGNORECASE,
    )
    if ends:
        value = (ends.group(1) or ends.group(2)).strip()
        return _scope(normalized), "ends_with", value
    return _scope(normalized), "contains", _condition_value(text)


def _payload_before_condition(text: str) -> str | None:
    match = re.search(
        r"\b(?:anad\w*|añad\w*|agreg\w*|pon\w*)\s+"
        r"(?:[\"']([^\"']+)[\"']|([^\s]+))\s+"
        r"(?:a|en)\s+(?:las|los)\s+que\b",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    return (match.group(1) or match.group(2)).strip()


def _quoted_after(text: str, marker: str) -> str | None:
    match = re.search(
        rf"\b{marker}\w*\b(?:\s+el\s+texto)?(?:\s+al\s+lado)?\s+[\"']([^\"']+)[\"']",
        text,
        re.IGNORECASE,
    )
    return match.group(1) if match else None


def _tail_after_mutation(text: str) -> str:
    match = re.search(
        r"\b(?:escrib\w*|agreg\w*|anad\w*|añad\w*|teng\w*|tuvier\w*)\b(?:\s+el\s+texto)?\s+(.+)$",
        text,
        re.IGNORECASE,
    )
    return match.group(1).strip(" \t\r\n.,;:\"'") if match else ""


def _parse_replacement(text: str) -> tuple[str, str] | None:
    quoted = re.search(
        r"\breemplaz\w*\s+[\"']([^\"']+)[\"']\s+por\s+[\"']([^\"']*)[\"']",
        text,
        re.IGNORECASE,
    )
    if quoted:
        return quoted.group(1), quoted.group(2)
    plain = re.search(r"\breemplaz\w*\s+(\S+)\s+por\s+(\S+)", text, re.IGNORECASE)
    return (plain.group(1), plain.group(2).rstrip(".,;")) if plain else None


def _count(content: str, needle: str, case_sensitive: bool) -> int:
    if case_sensitive:
        return content.count(needle)
    return content.casefold().count(needle.casefold())


def _replace_all(content: str, old: str, new: str, case_sensitive: bool) -> str:
    if case_sensitive:
        return content.replace(old, new)
    return re.sub(re.escape(old), lambda _match: new, content, flags=re.IGNORECASE)


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _normalize_newlines(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _collapse_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _actual_item_count(content: str, layout: str, separator: str) -> int:
    if layout in {"lines", "suffix_lines"}:
        return len(content.splitlines())
    if layout == "inline" and separator:
        return len(content.split(separator))
    return 1 if content or layout == "inline" else 0
