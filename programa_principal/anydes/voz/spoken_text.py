from __future__ import annotations

import re
import unicodedata


def prepare_for_speech(text: str, *, max_chars: int = 4000) -> str:
    clean = _remove_code_blocks(text)
    clean = _strip_markdown(clean)
    clean = _strip_nonverbal_symbols(clean)
    clean = _collapse_whitespace(clean)

    if len(clean) > max_chars:
        clean = _truncate_at_sentence(clean, max_chars)

    return clean


def prepare_for_continuous_speech(text: str, *, max_chars: int = 4000) -> str:
    """Keep one natural utterance while avoiding hard resets at every full stop."""

    clean = prepare_for_speech(text, max_chars=max_chars)
    return _merge_internal_declarative_stops(clean)


def _remove_code_blocks(text: str) -> str:
    return re.sub(r"```.*?```", "Te dejo ese bloque en pantalla.", text, flags=re.S)


def _strip_markdown(text: str) -> str:
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.M)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text


def _collapse_whitespace(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+([.,!?;:])", r"\1", collapsed)


def _strip_nonverbal_symbols(text: str) -> str:
    return "".join(
        character
        for character in text
        if character not in {"\ufe0f", "\u200d"}
        and unicodedata.category(character) not in {"So", "Sk"}
    )


def _truncate_at_sentence(text: str, max_chars: int) -> str:
    shortened = text[:max_chars].rstrip()
    sentence_end = max(shortened.rfind("."), shortened.rfind("?"), shortened.rfind("!"))
    if sentence_end > 180:
        shortened = shortened[: sentence_end + 1]
    return shortened


def _merge_internal_declarative_stops(text: str) -> str:
    matches = tuple(re.finditer(r"\.(?=\s+\S)", text))
    if not matches:
        return text
    result: list[str] = []
    cursor = 0
    for match in matches:
        token_match = re.search(r"([\w]+)$", text[: match.start()])
        token = token_match.group(1) if token_match else ""
        result.append(text[cursor : match.start()])
        if len(token) <= 3 or (token.isupper() and len(token) <= 5):
            result.append(".")
        else:
            result.append(";")
        cursor = match.end()
    result.append(text[cursor:])
    return "".join(result)
