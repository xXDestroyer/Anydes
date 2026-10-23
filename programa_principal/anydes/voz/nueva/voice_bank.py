from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, NoReturn


SCORE_DIMENSIONS = (
    "identity",
    "youth",
    "diction",
    "naturalness",
    "sweetness",
    "attitude",
    "accent",
)


class VoiceBankError(ValueError):
    pass


ELEVENLABS_ML_BLOCK_MESSAGE = (
    "Project policy permanently prohibits using ElevenLabs output or derivatives "
    "for conditioning, training, feature extraction, or any other ML input."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def require_ml_conditioning_authorization(path: Path) -> dict[str, Any]:
    """Legacy gate retained for compatibility; ElevenLabs ML use is always blocked."""
    del path
    reject_elevenlabs_machine_use()


def reject_elevenlabs_machine_use() -> NoReturn:
    """Apply the permanent project-level ban to historical ElevenLabs material."""
    raise VoiceBankError(ELEVENLABS_ML_BLOCK_MESSAGE)


def require_voice_source_authorization(path: Path) -> dict[str, Any]:
    """Validate a non-ElevenLabs source before it can enter an ML voice workflow."""
    path = Path(path)
    if not path.exists():
        raise VoiceBankError("Voice-source authorization metadata is required.")
    authorization = json.loads(path.read_text(encoding="utf-8"))
    provider = str(authorization.get("provider", "")).strip().casefold()
    origin = str(authorization.get("origin", "")).strip().casefold()
    if "elevenlabs" in provider or "elevenlabs" in origin:
        reject_elevenlabs_machine_use()

    allowed_types = {
        "authorized_original_recording",
        "license_compatible_dataset",
        "license_compatible_synthetic_output",
    }
    source_type = authorization.get("source_type")
    if source_type not in allowed_types:
        raise VoiceBankError(f"Unsupported or unverified voice source type: {source_type!r}")
    if authorization.get("machine_use_allowed") is not True:
        raise VoiceBankError("The source does not explicitly allow machine use.")
    scopes = set(authorization.get("scopes", []))
    if "ml_inference_conditioning" not in scopes:
        raise VoiceBankError("Authorization is missing scope: ml_inference_conditioning")
    if source_type == "authorized_original_recording":
        if authorization.get("explicit_performer_consent") is not True:
            raise VoiceBankError("An original recording requires explicit performer consent.")
    elif not authorization.get("license_id"):
        raise VoiceBankError("Licensed datasets and synthetic outputs require a license ID.")
    if source_type == "license_compatible_synthetic_output":
        if authorization.get("output_reuse_explicit") is not True:
            raise VoiceBankError(
                "Synthetic output requires explicit permission for reuse as ML input."
            )
    if not authorization.get("evidence_sha256"):
        raise VoiceBankError("Authorization must include hashed evidence of its provenance.")
    return authorization


@dataclass(frozen=True)
class ManualVoiceScores:
    identity: float
    youth: float
    diction: float
    naturalness: float
    sweetness: float
    attitude: float
    accent: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> "ManualVoiceScores":
        missing = set(SCORE_DIMENSIONS) - set(values)
        extra = set(values) - set(SCORE_DIMENSIONS)
        if missing or extra:
            raise VoiceBankError(
                f"Invalid score dimensions; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        normalized = {key: float(values[key]) for key in SCORE_DIMENSIONS}
        if any(value < 0.0 or value > 10.0 for value in normalized.values()):
            raise VoiceBankError("Voice scores must be between 0 and 10.")
        return cls(**normalized)


class ApprovedVoiceBank:
    """Stores candidates without ever promoting them automatically."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.gold_dir = self.root / "gold_reference"
        self.pending_dir = self.root / "pending"
        self.approved_dir = self.root / "approved"
        self.baseline_dir = self.root / "baselines"

    def ensure_layout(self) -> None:
        for directory in (
            self.gold_dir,
            self.pending_dir,
            self.approved_dir,
            self.baseline_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def preserve_gold_reference(
        self,
        source: Path,
        *,
        expected_sha256: str,
    ) -> Path:
        self.ensure_layout()
        source = Path(source)
        actual = sha256_file(source)
        expected = expected_sha256.upper()
        if actual != expected:
            raise VoiceBankError("The gold reference does not match its authoritative hash.")
        target = self.gold_dir / source.name
        if target.exists() and sha256_file(target) != expected:
            raise VoiceBankError("The preserved gold reference was modified.")
        if not target.exists():
            shutil.copy2(source, target)
        self._write_json(
            self.gold_dir / "reference.json",
            {
                "schema_version": "1.0",
                "kind": "gold_reference",
                "immutable": True,
                "source_name": source.name,
                "preserved_file": target.name,
                "sha256": expected,
                "human_auditory_reference_only": True,
                "machine_use_allowed": False,
                "feature_extraction_allowed": False,
                "training_authorized": False,
                "official_voice_changed": False,
            },
        )
        return target

    def register_pending(
        self,
        *,
        candidate_id: str,
        audio: Path,
        text: str,
        engine: str,
        references: list[dict[str, Any]],
        parameters: Mapping[str, Any],
        seed: int,
        created_at: str | None = None,
    ) -> Path:
        self.ensure_layout()
        candidate_id = self._validated_id(candidate_id)
        self._validate_references(references)
        audio = Path(audio)
        target_dir = self.pending_dir / candidate_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target_audio = target_dir / audio.name
        if target_audio.exists() and sha256_file(target_audio) != sha256_file(audio):
            raise VoiceBankError(f"Candidate {candidate_id} already has different audio.")
        if not target_audio.exists():
            shutil.copy2(audio, target_audio)
        metadata = {
            "schema_version": "1.0",
            "candidate_id": candidate_id,
            "status": "pending_manual_approval",
            "audio": target_audio.name,
            "audio_sha256": sha256_file(target_audio),
            "text": text,
            "engine": engine,
            "references": references,
            "parameters": dict(parameters),
            "seed": int(seed),
            "created_at": created_at or datetime.now().astimezone().isoformat(),
            "manual_scores": None,
            "approved_by": None,
            "eligible_as_synthetic_reference": False,
        }
        self._write_json(target_dir / "metadata.json", metadata)
        return target_dir

    def approve(
        self,
        candidate_id: str,
        *,
        approved_by: str,
        scores: Mapping[str, float],
        notes: str = "",
    ) -> Path:
        if approved_by.strip().casefold() != "DemoUser":
            raise VoiceBankError("Only DemoUser's explicit approval can approve a voice.")
        candidate_id = self._validated_id(candidate_id)
        pending = self.pending_dir / candidate_id
        metadata_path = pending / "metadata.json"
        if not metadata_path.exists():
            raise VoiceBankError(f"Unknown pending candidate: {candidate_id}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        manual_scores = ManualVoiceScores.from_mapping(scores)
        target = self.approved_dir / candidate_id
        if target.exists():
            raise VoiceBankError(f"Candidate {candidate_id} is already approved.")
        shutil.copytree(pending, target)
        metadata.update(
            {
                "status": "approved_by_DemoUser",
                "manual_scores": asdict(manual_scores),
                "approved_by": "DemoUser",
                "approved_at": datetime.now().astimezone().isoformat(),
                "approval_notes": notes,
                "eligible_as_synthetic_reference": manual_scores.identity >= 9.0,
            }
        )
        self._write_json(target / "metadata.json", metadata)
        return target

    def verify_gold_reference(self) -> bool:
        metadata_path = self.gold_dir / "reference.json"
        if not metadata_path.exists():
            return False
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        preserved = self.gold_dir / metadata["preserved_file"]
        return preserved.exists() and sha256_file(preserved) == metadata["sha256"]

    @staticmethod
    def _validated_id(value: str) -> str:
        if not value or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in value):
            raise VoiceBankError("Candidate IDs may contain lowercase letters, numbers, '-' and '_'.")
        return value

    @staticmethod
    def _validate_references(references: list[dict[str, Any]]) -> None:
        for reference in references:
            provider = str(reference.get("provider", "")).casefold()
            origin = str(reference.get("origin", "")).casefold()
            if "elevenlabs" in provider or "elevenlabs" in origin:
                reject_elevenlabs_machine_use()
            if reference.get("machine_use_allowed") is not True:
                raise VoiceBankError(
                    "Every conditioning reference must carry explicit machine-use authorization."
                )

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
