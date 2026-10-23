from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re
import unicodedata


class DocumentFormat(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MARKDOWN = "markdown"


class ConfirmationDecision(StrEnum):
    CONFIRM = "CONFIRM"
    REJECT = "REJECT"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True)
class ExplicitFilenameSpan:
    raw_span: str
    display_name: str
    start: int
    end: int


_FORMAT_PATTERNS = (
    (DocumentFormat.PDF, r"\b(?:pdf|p\s*d\s*f|pd\s+f|pe\s+de\s+efe)\b"),
    (DocumentFormat.DOCX, r"\b(?:docx|word)\b"),
    (DocumentFormat.TXT, r"\b(?:txt|archivo\s+de\s+texto)\b"),
    (DocumentFormat.MARKDOWN, r"\b(?:markdown|md)\b"),
)


def normalize_control_text(text: str) -> str:
    """Normalize only command entities; never use this as document content."""

    normalized = _fold(text)
    normalized = re.sub(r"\b(?:p\s*d\s*f|pd\s+f|pe\s+de\s+efe)\b", "pdf", normalized)
    normalized = re.sub(r"\baslo\b", "hazlo", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def resolve_document_format(text: str, *, target_only: bool = False) -> DocumentFormat | None:
    normalized = normalize_control_text(text)
    if target_only:
        target = re.search(
            r"\b(?:a|en)\s+(?:un\s+)?(?:archivo\s+)?(?:formato\s+)?"
            r"(pdf|docx|word|txt|markdown|md)\b",
            normalized,
        )
        if target:
            return _format_from_token(target.group(1))
    found: list[tuple[int, DocumentFormat]] = []
    for format_id, pattern in _FORMAT_PATTERNS:
        found.extend((match.start(), format_id) for match in re.finditer(pattern, normalized))
    if not found:
        return None
    found.sort(key=lambda item: item[0])
    return found[-1][1] if target_only else found[0][1]


def extract_explicit_filename_span(text: str) -> ExplicitFilenameSpan | None:
    marker = re.compile(
        r"\b(?:"
        r"llamad[oa]|que\s+se\s+llame|con\s+nombre|"
        r"ponle(?:\s+como\s+nombre)?(?:\s+al\s+(?:archivo|documento))?"
        r")\s+",
        re.IGNORECASE,
    )
    matches = list(marker.finditer(text))
    if not matches:
        return None
    start = matches[-1].end()
    tail = text[start:]
    quoted = re.match(r"[\"']([^\"']+)[\"']", tail)
    if quoted:
        raw = quoted.group(1).strip()
        end = start + quoted.end()
    else:
        folded_tail = _fold(tail)
        stop = re.search(
            r"\s+(?:"
            r"con\s+(?:un|una|el|la|este|esta)\s+(?:resumen|ensayo|texto|explicacion|analisis|lista|contenido)\b|"
            r"con\s+(?:unas?|unos?|aproximadamente)?\s*\d+\s+palabras\b|"
            r"que\s+(?:incluya|contenga|lleve|diga|enumere|explique|describa)\b|"
            r"donde\s+(?:incluya|contenga|lleve|diga|enumere|explique|describa|este)\b|"
            r"de\s+(?:unas?|unos?|aproximadamente)\s+\d+\s+palabras\b|"
            r"y\s+(?:describ|present|redact|explic|escrib|inclu|cont|enum|analiz|resume)\w*\b"
            r")|[,;!?]",
            folded_tail,
        )
        raw = tail[:stop.start() if stop else len(tail)].strip(" .")
        end = start + (stop.start() if stop else len(tail))
    raw = re.sub(r"^(?:al\s+(?:archivo|documento)\s+)", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\.(?:txt|md|docx|pdf)$", "", raw.strip(), flags=re.IGNORECASE)
    safe = sanitize_windows_stem(raw)
    if not safe:
        return None
    display = safe[0].upper() + safe[1:] if safe else safe
    return ExplicitFilenameSpan(text[start:end], display, start, end)


def sanitize_windows_stem(value: str) -> str:
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(value)).strip(" .")
    reserved = {
        "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4",
        "COM5", "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3",
        "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
    }
    return "" if not stem or stem.upper() in reserved else stem


def classify_confirmation(text: str, *, destructive: bool = False) -> ConfirmationDecision:
    normalized = normalize_control_text(text).strip(" .,!¡¿?")
    if re.fullmatch(r"(?:no|no gracias|mejor no|cancela|cancelar|detente|no lo hagas|dejalo)", normalized):
        return ConfirmationDecision.REJECT
    strong = {
        "si", "si confirmo", "confirmo", "adelante", "si adelante",
        "si estoy seguro", "si eliminalos", "si borralos",
    }
    benign_phonetic = {"see", "c", "si", "yes", "claro", "dale", "hazlo", "adelante", "confirmo"}
    if destructive:
        return ConfirmationDecision.CONFIRM if normalized in strong else ConfirmationDecision.AMBIGUOUS
    tokens = normalized.split()
    allowed = benign_phonetic | {"de", "acuerdo", "esta", "bien", "ok", "okay", "por", "favor", "porfa", "gracias"}
    if tokens and len(tokens) <= 7 and tokens[0] in benign_phonetic and all(token in allowed for token in tokens):
        return ConfirmationDecision.CONFIRM
    return ConfirmationDecision.AMBIGUOUS


def asks_for_full_path(text: str) -> bool:
    normalized = normalize_control_text(text)
    return bool(re.search(
        r"\b(?:cual\s+es\s+la\s+ruta|ruta\s+completa|ubicacion\s+completa|"
        r"donde\s+esta\s+exactamente|direccion\s+completa)\b",
        normalized,
    ))


def _format_from_token(token: str) -> DocumentFormat:
    aliases = {
        "word": DocumentFormat.DOCX,
        "md": DocumentFormat.MARKDOWN,
    }
    return aliases[token] if token in aliases else DocumentFormat(token)


def _fold(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value).casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))
