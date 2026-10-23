from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VrmModelInfo:
    path: Path
    valid: bool
    spec_version: str = ""
    expressions: tuple[str, ...] = ()
    humanoid_bones: tuple[str, ...] = ()
    look_at_type: str = ""
    node_count: int = 0
    mesh_count: int = 0
    material_count: int = 0
    error: str = ""

    @property
    def supports_lip_sync(self) -> bool:
        return {"aa", "ee", "ih", "oh", "ou"}.issubset(self.expressions)

    @property
    def supports_blink(self) -> bool:
        return "blink" in self.expressions or {
            "blinkLeft",
            "blinkRight",
        }.issubset(self.expressions)

    @property
    def supports_gaze(self) -> bool:
        return "head" in self.humanoid_bones and bool(self.look_at_type)

    def public_payload(self) -> dict[str, Any]:
        # The renderer owns the private filesystem path. Cognitive context only
        # needs a stable asset identifier and capability metadata.
        return {
            "available": self.valid,
            "format": "vrm",
            "spec_version": self.spec_version,
            "preferred_vrm": self.path.name if self.valid else "",
            "expressions": list(self.expressions),
            "supports_lip_sync": self.supports_lip_sync,
            "supports_blink": self.supports_blink,
            "supports_gaze": self.supports_gaze,
            "node_count": self.node_count,
            "mesh_count": self.mesh_count,
            "material_count": self.material_count,
        }


def preferred_vrm_model(model_root: Path) -> Path | None:
    """Choose a reviewed Anydes model deterministically without modifying it."""

    candidates = (
        model_root / "final_review" / "exports" / "Anydes_v001_hands_in_pockets.vrm",
        model_root / "final_review" / "exports" / "Anydes_v001_review.vrm",
        model_root / "Anydes_v001.vrm",
        model_root / "Anydes_Base.vrm",
    )
    for candidate in candidates:
        if candidate.is_file() and inspect_vrm(candidate).valid:
            return candidate
    for candidate in sorted(model_root.rglob("*.vrm")) if model_root.exists() else ():
        if inspect_vrm(candidate).valid:
            return candidate
    return None


@lru_cache(maxsize=32)
def inspect_vrm(path: Path) -> VrmModelInfo:
    """Read only the JSON chunk of a GLB/VRM container."""

    resolved = path.resolve(strict=False)
    try:
        with resolved.open("rb") as handle:
            header = handle.read(20)
            if len(header) < 20:
                raise ValueError("truncated_header")
            magic, container_version, declared_length, json_length, chunk_type = struct.unpack(
                "<4sIII4s", header
            )
            if magic != b"glTF" or container_version != 2 or chunk_type != b"JSON":
                raise ValueError("unsupported_container")
            actual_length = resolved.stat().st_size
            if declared_length > actual_length or json_length > actual_length - 20:
                raise ValueError("invalid_chunk_length")
            document = json.loads(
                handle.read(json_length).decode("utf-8").rstrip(" \x00")
            )
        if not isinstance(document, dict):
            raise ValueError("invalid_json_root")
        vrm = _vrm_extension(document)
        if not vrm:
            raise ValueError("missing_vrm_extension")
        expressions = vrm.get("expressions") if isinstance(vrm.get("expressions"), dict) else {}
        preset = expressions.get("preset") if isinstance(expressions.get("preset"), dict) else {}
        custom = expressions.get("custom") if isinstance(expressions.get("custom"), dict) else {}
        humanoid = vrm.get("humanoid") if isinstance(vrm.get("humanoid"), dict) else {}
        bones = humanoid.get("humanBones") if isinstance(humanoid.get("humanBones"), dict) else {}
        look_at = vrm.get("lookAt") if isinstance(vrm.get("lookAt"), dict) else {}
        return VrmModelInfo(
            path=resolved,
            valid=True,
            spec_version=str(vrm.get("specVersion") or "1.0"),
            expressions=tuple(sorted({*preset, *custom})),
            humanoid_bones=tuple(sorted(bones)),
            look_at_type=str(look_at.get("type") or ""),
            node_count=len(document.get("nodes", [])),
            mesh_count=len(document.get("meshes", [])),
            material_count=len(document.get("materials", [])),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, struct.error) as exc:
        return VrmModelInfo(
            path=resolved,
            valid=False,
            error=str(exc),
        )


def _vrm_extension(document: dict[str, Any]) -> dict[str, Any]:
    extensions = document.get("extensions")
    if not isinstance(extensions, dict):
        return {}
    modern = extensions.get("VRMC_vrm")
    if isinstance(modern, dict):
        return modern
    legacy = extensions.get("VRM")
    return legacy if isinstance(legacy, dict) else {}
