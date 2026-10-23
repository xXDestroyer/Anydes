from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from difflib import SequenceMatcher
from pathlib import Path
import re
import time
import unicodedata
from uuid import uuid4

from .intent_normalization import resolve_document_format
from .dataflow import ArtifactBindings, ArtifactRef, ArtifactState, DataflowError


@dataclass
class ArtifactRecord:
    artifact_id: str
    absolute_canonical_path: str
    filename: str
    format: str
    created_at: float
    modified_at: float
    task_id: str = ""
    semantic_topic: str = ""
    relationship: dict[str, str] = field(default_factory=dict)
    is_active: bool = False
    deleted_at: float | None = None
    runtime_ref: ArtifactRef | None = None

    def to_dict(self) -> dict:
        value = asdict(self)
        value.pop("runtime_ref")
        ref = self.runtime_ref
        value.update(display_name=self.filename, artifact_type=ref.artifact_type if ref else "",
            producer_step_id=ref.producer_step_id if ref else "", version=ref.version if ref else 0,
            hash=ref.hash if ref else "", state=ref.state.value if ref else "LEGACY",
            committed_at=self.modified_at, binding_source="RUNTIME_COMMIT" if ref else "LEGACY_ADAPTER")
        return value


class RecentArtifactIndex:
    def __init__(self, limit: int = 16) -> None:
        self.limit = limit
        self.artifacts: list[ArtifactRecord] = []
        self.active_artifact_id = ""
        self.focus_artifact_ids: tuple[str, ...] = ()
        self.focus_unavailable = False
        self.last_resolution = {"status": "NOT_FOUND", "artifact_id": "", "candidates": []}

    def _selection(self, candidates):
        status = "RESOLVED" if len(candidates) == 1 else "AMBIGUOUS" if candidates else "NOT_FOUND"
        self.last_resolution = {"status": status,
            "artifact_id": candidates[0].artifact_id if len(candidates) == 1 else "",
            "candidates": [item.artifact_id for item in candidates]}
        return candidates[0].absolute_canonical_path if len(candidates) == 1 else ""

    def register(self, ref: ArtifactRef) -> ArtifactRecord | None:
        if ref.state not in {ArtifactState.COMMITTED, ArtifactState.COMMITTED_UNVERIFIED} or ref.metadata.get("user_addressable", True) is not True:
            return None
        ledger = ArtifactBindings(ref.task_id, {"reference": ref})
        try:
            ledger.require("reference", verified=True, allow_unverified=True)
        except (DataflowError, OSError):
            return None
        existing = next((r for r in self.artifacts if r.artifact_id == ref.artifact_id), None)
        if existing and existing.runtime_ref and ref.version < existing.runtime_ref.version:
            return existing
        # An overwritten path supersedes its old identity, not another same-named file.
        for item in self.artifacts:
            if item.artifact_id != ref.artifact_id and item.absolute_canonical_path.casefold() == ref.path.casefold():
                self.remove(item.absolute_canonical_path)
        record = existing or self.find_by_path(ref.path)
        if record is None:
            record = self.remember(ref.path, task_id=ref.task_id)
        record.artifact_id = ref.artifact_id
        record.absolute_canonical_path = ref.path
        record.filename = ref.display_name
        record.format = ref.format.casefold()
        record.task_id = ref.task_id
        record.runtime_ref = ref
        record.deleted_at = None
        record.modified_at = time.time()
        for item in self.artifacts:
            item.is_active = item is record
        self.active_artifact_id = ref.artifact_id
        self.focus_artifact_ids = (ref.artifact_id,)
        self.focus_unavailable = False
        self.artifacts = self.artifacts[-self.limit:]
        return record

    def ref_by_path(self, path: str) -> ArtifactRef | None:
        item = self.find_by_path(path)
        return item.runtime_ref if item and self._selectable(item) else None

    def resolve_ref(self, text: str, **constraints) -> ArtifactRef | None:
        path = self.resolve(text, **constraints)
        item = self.find_by_path(path) if path else None
        # Keep the selected immutable identity even if the file changes immediately
        # afterwards: execution checks its hash, rather than rebinding new bytes.
        if item and item.runtime_ref:
            self.focus_artifact_ids = (item.artifact_id,)
            return item.runtime_ref
        return None

    @staticmethod
    def _selectable(item):
        if item.deleted_at is not None:
            return False
        if item.runtime_ref is None:
            return Path(item.absolute_canonical_path).is_file()
        ref = item.runtime_ref
        if ref.state not in {ArtifactState.COMMITTED, ArtifactState.COMMITTED_UNVERIFIED} or ref.metadata.get("user_addressable", True) is not True:
            return False
        try:
            ArtifactBindings(ref.task_id, {"reference": ref}).require("reference", verified=True, allow_unverified=True)
            return True
        except (DataflowError, OSError):
            return False

    def remember(
        self,
        path: str,
        *,
        task_id: str = "",
        semantic_topic: str = "",
        relationship: dict[str, str] | None = None,
    ) -> ArtifactRecord:
        canonical = str(Path(path).resolve(strict=False))
        now = time.time()
        existing = next((item for item in self.artifacts if item.absolute_canonical_path.casefold() == canonical.casefold()), None)
        for item in self.artifacts:
            item.is_active = False
        if existing is None:
            existing = ArtifactRecord(
                artifact_id=f"artifact:{uuid4().hex}",
                absolute_canonical_path=canonical,
                filename=Path(canonical).name,
                format=Path(canonical).suffix.casefold().removeprefix("."),
                created_at=now,
                modified_at=now,
            )
            self.artifacts.append(existing)
        existing.modified_at = now
        existing.deleted_at = None
        existing.task_id = task_id or existing.task_id
        existing.semantic_topic = semantic_topic or existing.semantic_topic
        existing.relationship = dict(relationship or existing.relationship)
        existing.is_active = True
        self.active_artifact_id = existing.artifact_id
        self.artifacts = self.artifacts[-self.limit:]
        return existing

    def remove(self, path: str) -> ArtifactRecord | None:
        canonical = str(Path(path).resolve(strict=False)).casefold()
        existing = next(
            (item for item in self.artifacts if item.absolute_canonical_path.casefold() == canonical),
            None,
        )
        if existing is not None:
            existing.deleted_at = time.time()
            existing.modified_at = existing.deleted_at
            existing.is_active = False
            if existing.runtime_ref:
                existing.runtime_ref = replace(existing.runtime_ref, state=ArtifactState.DELETED)
        if existing is not None and existing.artifact_id == self.active_artifact_id:
            self.active_artifact_id = ""
        self.focus_artifact_ids = tuple(a for a in self.focus_artifact_ids
            if existing is None or a != existing.artifact_id)
        return existing

    def find_by_path(self, path: str) -> ArtifactRecord | None:
        canonical = str(Path(path).resolve(strict=False)).casefold()
        return next(
            (item for item in reversed(self.artifacts) if item.absolute_canonical_path.casefold() == canonical),
            None,
        )

    def relate(self, path: str, **relationship: str) -> None:
        item = self.find_by_path(path)
        if item is not None:
            item.relationship.update({key: value for key, value in relationship.items() if value})

    def resolve(self, text: str, *, explicit_filename: str = "", require_unique: bool = False,
                excluded_names: tuple[str, ...] = (), artifact_type: str = '') -> str:
        self._selection([])
        existing = [
            item for item in self.artifacts
            if self._selectable(item)
        ]
        if artifact_type:
            existing = [item for item in existing if item.runtime_ref and item.runtime_ref.artifact_type == artifact_type]
        if explicit_filename:
            matches = [
                item for item in existing
                if item.filename.casefold() == explicit_filename.casefold()
            ]
            return self._selection(matches)
        excluded = {name.casefold() for name in excluded_names}
        named = set(re.findall(r"(?<![\w.])([\w-]+\.[\w]+)(?![\w.])", text.casefold())) - excluded
        if named and not any(item.filename.casefold() in named for item in existing):
            return ""
        exact = [item for item in existing if item.filename.casefold() not in excluded and re.search(
            r"(?<![\w.])" + re.escape(item.filename.casefold()) + r"(?![\w.])", text.casefold())]
        if exact:
            return self._selection(exact)
        if require_unique and self.focus_unavailable:
            return ''
        spoken_name = _extract_spoken_filename(text)
        if spoken_name:
            spoken_candidates = existing
            format_id = resolve_document_format(_source_reference_text(text))
            if format_id is not None:
                expected = "md" if format_id.value == "markdown" else format_id.value
                spoken_candidates = [item for item in existing if item.format == expected]
            matched = _unique_filename_match(spoken_name, spoken_candidates)
            if matched is not None:
                return self._selection([matched])
        normalized = _fold(text)
        recency_reference = bool(re.search(
            r"\b(?:ultimo|ultima)\s+(?:archivo|documento|word|txt|pdf)\b|"
            r"\b(?:archivo|documento)\s+que\s+acabas?\s+de\s+(?:crear|generar|convertir)\b",
            normalized,
        ))
        format_id = resolve_document_format(_source_reference_text(text))
        candidates = existing
        if format_id is not None:
            expected = "md" if format_id.value == "markdown" else format_id.value
            candidates = [item for item in candidates if item.format == expected]
        else:
            # Declarative conversational type constraints; never filename extensions.
            reference_types = {"CODE": {"programa", "codigo"}, "ARCHIVE": {"archivo comprimido"}}
            constrained = {kind for kind, aliases in reference_types.items()
                if any(re.search(r"\b" + re.escape(alias) + r"\b", normalized) for alias in aliases)}
            if constrained:
                candidates = [item for item in candidates if item.runtime_ref
                    and item.runtime_ref.artifact_type in constrained]
        if recency_reference and self.active_artifact_id:
            active = next((item for item in candidates if item.artifact_id == self.active_artifact_id), None)
            if active is not None:
                return self._selection([active])
        topic_tokens = _reference_topic_tokens(text) if not any(item.runtime_ref for item in candidates) else ()
        if topic_tokens:
            scored = [
                (sum(token in _fold(f"{item.filename} {item.semantic_topic}") for token in topic_tokens), item)
                for item in candidates
            ]
            scored = [item for score, item in scored if score > 0]
            if scored:
                candidates = scored
        focused = [item for item in candidates if item.artifact_id in self.focus_artifact_ids]
        if len(focused) == 1:
            return self._selection(focused)
        related_ids = {a for item in existing if item.artifact_id in self.focus_artifact_ids and item.runtime_ref
            for a in item.runtime_ref.metadata.get('consumed_artifacts', [])}
        related = [item for item in candidates if item.artifact_id in related_ids]
        if not focused and len(related) == 1:
            return self._selection(related)
        if len(candidates) != 1 and (require_unique or any(item.runtime_ref for item in candidates)):
            return self._selection(candidates)
        return self._selection(candidates[-1:])

    def resolve_from_workspace(self, text: str, workspace: str) -> str:
        """Deprecated: filesystem discovery is not conversational authority."""
        return ""

    def snapshot(self) -> list[dict]:
        return [item.to_dict() for item in self.artifacts]


# Compatibility name, same index instance/contract, not a parallel document index.
RecentArtifactState = RecentArtifactIndex


def _reference_topic_tokens(text: str) -> tuple[str, ...]:
    normalized = _fold(text)
    words = re.findall(r"[a-z0-9]+", normalized)
    ignored = {
        "archivo", "documento", "word", "docx", "txt", "pdf", "markdown",
        "ultimo", "anterior", "creado", "creaste", "transforma", "convierte",
        "edita", "abre", "el", "la", "de", "del", "un", "una", "a", "en",
    }
    return tuple(word for word in words if len(word) >= 4 and word not in ignored)


def _extract_spoken_filename(text: str) -> str:
    normalized = _fold(text)
    patterns = (
        r"\b(?:se\s+llama|llamad[oa]|con\s+nombre)\s+(.+?)(?=$|[,.!?])",
        r"\b(?:duplica|copia|abre|edita|elimina|borra|convierte)\w*\s+"
        r"(?:el\s+)?(?:archivo|documento)?\s*(?:de\s+word|word|txt|pdf|docx)?\s*"
        r"(?:que\s+se\s+llama\s+)?(.+?)(?=$|\s+(?:a|en)\s+(?:un\s+)?(?:pdf|docx|txt|markdown)\b|[,.!?])",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            candidate = re.sub(
                r"^(?:el|la|los|las)\s+|\b(?:archivo|documento)\b$",
                "",
                match.group(1).strip(),
            ).strip()
            if candidate and candidate not in {"este", "ese", "ultimo", "anterior"}:
                return candidate
    return ""


def _unique_filename_match(spoken_name: str, records: list[ArtifactRecord]) -> ArtifactRecord | None:
    query = _filename_key(spoken_name)
    if not query:
        return None
    exact = [item for item in records if _filename_key(item.filename) == query]
    if len(exact) == 1:
        return exact[0]
    stem_exact = [
        item for item in records
        if _filename_key(Path(item.filename).stem) == _filename_key(Path(spoken_name).stem)
    ]
    if len(stem_exact) == 1:
        return stem_exact[0]
    if any(item.runtime_ref for item in records):
        return None
    ranked = sorted(
        ((SequenceMatcher(None, query, _filename_key(Path(item.filename).stem)).ratio(), item) for item in records),
        key=lambda value: value[0],
        reverse=True,
    )
    if not ranked or ranked[0][0] < 0.86:
        return None
    if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 0.08:
        return None
    return ranked[0][1]


def _filename_key(value: str) -> str:
    stem = _fold(Path(str(value)).stem)
    words = re.findall(r"[a-z0-9]+", stem)
    return " ".join(_number_words_to_digits(words))


def _number_words_to_digits(words: list[str]) -> list[str]:
    units = {
        "cero": 0, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4,
        "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9,
        "diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14,
        "quince": 15, "dieciseis": 16, "diecisiete": 17, "dieciocho": 18,
        "diecinueve": 19, "veinte": 20, "veintiuno": 21, "veintidos": 22,
        "veintitres": 23, "veinticuatro": 24, "veinticinco": 25,
        "veintiseis": 26, "veintisiete": 27, "veintiocho": 28, "veintinueve": 29,
    }
    tens = {"treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60,
            "setenta": 70, "ochenta": 80, "noventa": 90}
    result: list[str] = []
    index = 0
    while index < len(words):
        word = words[index]
        if word in tens:
            value = tens[word]
            consumed = 1
            if index + 2 < len(words) and words[index + 1] == "y" and words[index + 2] in units and units[words[index + 2]] < 10:
                value += units[words[index + 2]]
                consumed = 3
            result.append(str(value))
            index += consumed
            continue
        if word in units:
            result.append(str(units[word]))
        else:
            result.append(word)
        index += 1
    return result


def _source_reference_text(text: str) -> str:
    normalized = _fold(text)
    if re.search(r"\b(?:convierte|convertir|transforma|transformar)\b", normalized):
        return re.split(
            r"\s+(?:a|en)\s+(?:un\s+)?(?:archivo\s+)?(?:formato\s+)?"
            r"(?:pdf|p\s*d\s*f|pd\s+f|pe\s+de\s+efe|docx|word|txt|markdown|md)\b",
            normalized,
            maxsplit=1,
        )[0]
    return text


def _fold(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value).casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))
