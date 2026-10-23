from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

from anydes.funciones_extra.apps import APP_ALIASES


# Misrecognitions comunes de Vosk/Windows STT en espanol.
_WORD_CORRECTIONS: dict[str, str] = {
    "espotifai": "spotify",
    "espotifi": "spotify",
    "espotify": "spotify",
    "spotifi": "spotify",
    "spotifay": "spotify",
    "spoti fi": "spotify",
    "espotifay": "spotify",
    "isportify": "spotify",
    "sportify": "spotify",
    "youtub": "youtube",
    "iu tub": "youtube",
    "iu tu": "youtube",
    "you tube": "youtube",
    "iu tube": "youtube",
    "gugle": "google",
    "gugel": "google",
    "gogle": "google",
    "guugle": "google",
    "crome": "chrome",
    "cron": "chrome",
    "anides": "anydes",
    "anidez": "anydes",
    "anidesh": "anydes",
    "anide": "anydes",
    "any des": "anydes",
    "anydez": "anydes",
    "anyde": "anydes",
    "anydess": "anydes",
    "whatsapp": "whatsapp",
    "guasap": "whatsapp",
    "wasap": "whatsapp",
    "wassap": "whatsapp",
    "watsap": "whatsapp",
    "calculadora": "calculadora",
    "calculadota": "calculadora",
    "notepad": "notepad",
    "note pad": "notepad",
    "vscode": "vscode",
    "visual studio code": "vscode",
    "explorador": "explorador",
    "edge": "edge",
}

# Frases que suelen fallar parcialmente en STT.
_PHRASE_CORRECTIONS: tuple[tuple[str, str], ...] = (
    ("abre espotifai", "abre spotify"),
    ("abrir espotifai", "abrir spotify"),
    ("abre espotifi", "abre spotify"),
    ("abre sportify", "abre spotify"),
    ("abre iu tub", "abre youtube"),
    ("abre you tube", "abre youtube"),
    ("busca en gugle", "busca en google"),
    ("busca en gogle", "busca en google"),
    ("mira mi pantalla", "mira mi pantalla"),
    ("mira la pantalla", "mira mi pantalla"),
    ("mirar pantalla", "mira mi pantalla"),
    ("que ves en pantalla", "que ves en mi pantalla"),
    ("comparte pantalla", "comparte pantalla"),
    ("compartir pantalla", "comparte pantalla"),
    ("deja de ver pantalla", "deja de ver mi pantalla"),
)

_KNOWN_APP_NAMES = tuple(sorted(APP_ALIASES.keys(), key=len, reverse=True))


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def normalize_transcript(text: str) -> str:
    """Corrige errores tipicos de STT y alinea nombres de apps/comandos."""
    if not text or not text.strip():
        return text

    cleaned = re.sub(r"\s+", " ", text.strip())
    lowered = cleaned.lower()

    for wrong, right in _PHRASE_CORRECTIONS:
        if wrong in lowered:
            lowered = lowered.replace(wrong, right)

    tokens = lowered.split()
    corrected_tokens: list[str] = []
    for token in tokens:
        bare = re.sub(r"[^\w]", "", token)
        replacement = _WORD_CORRECTIONS.get(bare)
        if replacement:
            corrected_tokens.append(replacement)
        else:
            corrected_tokens.append(token)

    result = " ".join(corrected_tokens)
    result = _fix_app_after_open(result)
    return result


def _fix_app_after_open(text: str) -> str:
    """Si el usuario dijo 'abre X' y X suena parecido a una app conocida, corrige X."""
    match = re.match(r"^(abre|abrir|pon|abre la app|abre aplicacion)\s+(.+)$", text)
    if not match:
        return text

    prefix, app_part = match.group(1), match.group(2).strip()
    if not app_part:
        return text

    normalized_app = _strip_accents(app_part)
    if normalized_app in APP_ALIASES:
        return f"{prefix} {normalized_app}"

    best_name = ""
    best_score = 0.0
    for known in _KNOWN_APP_NAMES:
        score = SequenceMatcher(None, normalized_app, _strip_accents(known)).ratio()
        if score > best_score:
            best_score = score
            best_name = known

    if best_score >= 0.72:
        return f"{prefix} {best_name}"
    return text
