from __future__ import annotations

import html
import json
import re
import unicodedata
from dataclasses import dataclass
from dataclasses import field
from typing import Mapping
from urllib.parse import urlsplit


_MONTHS = (
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "octubre",
    "noviembre",
    "diciembre",
)
_SMALL = {
    0: "cero",
    1: "uno",
    2: "dos",
    3: "tres",
    4: "cuatro",
    5: "cinco",
    6: "seis",
    7: "siete",
    8: "ocho",
    9: "nueve",
    10: "diez",
    11: "once",
    12: "doce",
    13: "trece",
    14: "catorce",
    15: "quince",
    16: "dieciseis",
    17: "diecisiete",
    18: "dieciocho",
    19: "diecinueve",
    20: "veinte",
    21: "veintiuno",
    22: "veintidos",
    23: "veintitres",
    24: "veinticuatro",
    25: "veinticinco",
    26: "veintiseis",
    27: "veintisiete",
    28: "veintiocho",
    29: "veintinueve",
}
_TENS = {30: "treinta", 40: "cuarenta", 50: "cincuenta", 60: "sesenta", 70: "setenta", 80: "ochenta", 90: "noventa"}
_HUNDREDS = {100: "cien", 200: "doscientos", 300: "trescientos", 400: "cuatrocientos", 500: "quinientos", 600: "seiscientos", 700: "setecientos", 800: "ochocientos", 900: "novecientos"}


@dataclass(frozen=True)
class SpeechTextNormalizer:
    """Produces speakable Spanish without changing the visible response."""

    pronunciation_lexicon: Mapping[str, str] = field(default_factory=dict)

    def normalize(self, visible_text: str) -> str:
        text = unicodedata.normalize("NFKC", html.unescape(str(visible_text or "")))
        text = "".join(char for char in text if char in "\n\t" or unicodedata.category(char) != "Cc")
        text = re.sub(r"```[\s\S]*?```", " ", text)
        text = re.sub(r"`([^`]+)`", r"\1", text)
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
        text = re.sub(r"(?<!\w)(?:\*\*|__)(.+?)(?:\*\*|__)(?!\w)", r"\1", text)
        text = re.sub(r"(?<!\w)(?:\*|_)([^\n*_]+?)(?:\*|_)(?!\w)", r"\1", text)
        text = re.sub(r"(?m)^\s{0,3}(?:#{1,6}|[-*+]\s+|>\s*)", "", text)
        text = self._remove_structured_payload(text)
        text = re.sub(r"\b(?:tool|provider|intention|evidence|trace|request|observation):[\w:.-]+\b", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"https?://[^\s]+", self._url, text)
        text = re.sub(r"\b(\d{4})-(\d{2})-(\d{2})\b", self._date, text)
        text = re.sub(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", self._time, text)
        text = re.sub(r"\b\d+(?:[.,]\d+)?\b", self._number, text)
        text = self._apply_pronunciation_lexicon(text)
        replacements = {
            r"\bSr\.\b": "senor",
            r"\bSra\.\b": "senora",
            r"\bDr\.\b": "doctor",
            r"\bDra\.\b": "doctora",
            r"\betc\.\b": "etcetera",
            r"\bQwen\b": "Kuen",
            r"\bDota\s+dos\b": "Dota dos",
        }
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        text = "".join(
            char
            for char in text
            if unicodedata.category(char) not in {"So", "Cs"}
        )
        text = re.sub(r"([!?.,;:])\1+", r"\1", text)
        text = re.sub(r"[\t ]+", " ", text)
        text = re.sub(r"\s*\n\s*", ". ", text)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        return text.strip(" \t\n")

    def _apply_pronunciation_lexicon(self, text: str) -> str:
        for visible, spoken in sorted(
            self.pronunciation_lexicon.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            pattern = rf"(?<!\w){re.escape(visible)}(?!\w)"
            text = re.sub(pattern, spoken, text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def _remove_structured_payload(text: str) -> str:
        stripped = text.strip()
        if not stripped or stripped[0] not in "[{":
            return text
        try:
            json.loads(stripped)
        except (json.JSONDecodeError, TypeError):
            return text
        return ""

    @staticmethod
    def _url(match: re.Match[str]) -> str:
        parsed = urlsplit(match.group(0).rstrip(".,;:!?"))
        host = (parsed.hostname or "sitio web").removeprefix("www.")
        return "sitio " + host.replace(".", " punto ")

    @staticmethod
    def _date(match: re.Match[str]) -> str:
        year, month, day = (int(item) for item in match.groups())
        if not 1 <= month <= 12 or not 1 <= day <= 31:
            return match.group(0)
        return f"{_number_words(day)} de {_MONTHS[month - 1]} de {_number_words(year)}"

    @staticmethod
    def _time(match: re.Match[str]) -> str:
        hour, minute = (int(item) for item in match.groups())
        if minute == 0:
            return f"{_number_words(hour)} horas"
        return f"{_number_words(hour)} horas con {_number_words(minute)} minutos"

    @staticmethod
    def _number(match: re.Match[str]) -> str:
        raw = match.group(0).replace(",", ".")
        if "." in raw:
            whole, decimal = raw.split(".", 1)
            return f"{_number_words(int(whole))} coma {' '.join(_SMALL[int(item)] for item in decimal)}"
        return _number_words(int(raw))


def _number_words(value: int) -> str:
    if value < 0:
        return "menos " + _number_words(abs(value))
    if value < 30:
        return _SMALL[value]
    if value < 100:
        base = value // 10 * 10
        return _TENS[base] if value == base else f"{_TENS[base]} y {_SMALL[value - base]}"
    if value < 1000:
        base = value // 100 * 100
        prefix = "ciento" if base == 100 and value != 100 else _HUNDREDS[base]
        return prefix if value == base else f"{prefix} {_number_words(value - base)}"
    if value < 2000:
        return "mil" if value == 1000 else f"mil {_number_words(value - 1000)}"
    if value < 1_000_000:
        thousands, rest = divmod(value, 1000)
        prefix = f"{_number_words(thousands)} mil"
        return prefix if rest == 0 else f"{prefix} {_number_words(rest)}"
    return " ".join(_SMALL[int(item)] for item in str(value))
