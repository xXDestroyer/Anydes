"""Shared contracts for the frozen Parakeet generalization validation."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import unicodedata


HERE = Path(__file__).resolve().parent
PROGRAM = HERE.parents[2]
VOICE = PROGRAM / "anydes" / "voice_input"
BENCH = HERE
CORPUS = VOICE / "HUMAN_GENERALIZATION_CORPUS_30.csv"
OUT = PROGRAM / "datos" / "telemetria" / "stt_parakeet"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def load_corpus() -> list[dict]:
    with CORPUS.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 30 or len({row["id"] for row in rows}) != 30:
        raise RuntimeError("El corpus congelado debe contener exactamente 30 IDs unicos.")
    return rows


def normalize_words(text: str) -> list[str]:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.findall(r"\w+|\+\+", text)


def normalize_chars(text: str) -> str:
    return "".join(normalize_words(text))


def edit_alignment(reference: list[str], hypothesis: list[str]) -> dict:
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    cost = [[0] * cols for _ in range(rows)]
    op = [[""] * cols for _ in range(rows)]
    for i in range(1, rows):
        cost[i][0] = i
        op[i][0] = "delete"
    for j in range(1, cols):
        cost[0][j] = j
        op[0][j] = "insert"
    for i in range(1, rows):
        for j in range(1, cols):
            choices = [
                (cost[i - 1][j] + 1, "delete"),
                (cost[i][j - 1] + 1, "insert"),
                (cost[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]),
                    "equal" if reference[i - 1] == hypothesis[j - 1] else "substitute"),
            ]
            cost[i][j], op[i][j] = min(choices, key=lambda item: item[0])
    counts = {"delete": 0, "insert": 0, "substitute": 0}
    i, j = len(reference), len(hypothesis)
    while i or j:
        action = op[i][j]
        if action == "equal":
            i -= 1
            j -= 1
        elif action == "substitute":
            counts[action] += 1
            i -= 1
            j -= 1
        elif action == "delete":
            counts[action] += 1
            i -= 1
        else:
            counts["insert"] += 1
            j -= 1
    counts["errors"] = cost[-1][-1]
    return counts


def text_metrics(reference: str, hypothesis: str) -> dict:
    reference_words = normalize_words(reference)
    hypothesis_words = normalize_words(hypothesis)
    word_ops = edit_alignment(reference_words, hypothesis_words)
    reference_chars = normalize_chars(reference)
    hypothesis_chars = normalize_chars(hypothesis)
    char_ops = edit_alignment(list(reference_chars), list(hypothesis_chars))
    return {
        "exact": reference_words == hypothesis_words,
        "word_errors": word_ops["errors"],
        "reference_words": len(reference_words),
        "wer": word_ops["errors"] / max(1, len(reference_words)),
        "omissions": word_ops["delete"],
        "insertions": word_ops["insert"],
        "substitutions": word_ops["substitute"],
        "character_errors": char_ops["errors"],
        "reference_characters": len(reference_chars),
        "cer": char_ops["errors"] / max(1, len(reference_chars)),
    }


SPANISH_FUNCTION_WORDS = {
    "a", "al", "algo", "con", "de", "del", "el", "en", "es", "esta", "este",
    "la", "las", "lo", "los", "me", "mi", "no", "o", "para", "por", "que",
    "se", "si", "sin", "su", "te", "un", "una", "y", "ya",
}


def intent_preserved_preliminary(reference: str, hypothesis: str, metrics: dict) -> bool:
    """Conservative content-word check; final intent remains a blind human label."""
    reference_content = [word for word in normalize_words(reference)
        if len(word) >= 3 and word not in SPANISH_FUNCTION_WORDS]
    hypothesis_words = normalize_words(hypothesis)
    matched = 0
    for expected in reference_content:
        allowed = max(1, round(len(expected) * 0.2))
        if any(edit_alignment(list(expected), list(actual))["errors"] <= allowed
                for actual in hypothesis_words):
            matched += 1
    coverage = matched / max(1, len(reference_content))
    return bool(hypothesis.strip()) and coverage >= 0.7 and metrics["cer"] <= 0.45


ENGLISH_MARKERS = {
    "and", "are", "can", "could", "do", "does", "for", "from", "have", "hello",
    "how", "i", "is", "it", "listen", "need", "please", "that", "the", "this",
    "to", "what", "when", "with", "you", "your",
}


def language_drift(text: str) -> dict:
    words = normalize_words(text)
    markers = [word for word in words if word in ENGLISH_MARKERS]
    return {
        "language_drift": len(markers) >= 2,
        "english_markers": markers,
        "language_drift_method": "diagnostic_lexical_markers_no_postprocessing",
    }


def latest_session(complete_only: bool = False) -> Path | None:
    if not OUT.is_dir():
        return None
    candidates = sorted(OUT.glob("generalization-recording-*"), reverse=True)
    for directory in candidates:
        state = directory / "recording-state.json"
        if not state.is_file():
            continue
        payload = json.loads(state.read_text(encoding="utf-8"))
        if not complete_only or payload.get("complete"):
            return directory
    return None
