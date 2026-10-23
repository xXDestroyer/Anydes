from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
import hashlib
from pathlib import Path
import re
from typing import Any, Sequence
from uuid import uuid4


class DeleteScope(StrEnum):
    SINGLE_TARGET = "DELETE_SINGLE_TARGET"
    COLLECTION = "DELETE_COLLECTION"
    BY_FILTER = "DELETE_BY_FILTER"


class DeleteArtifactKind(StrEnum):
    USER_ARTIFACT = "USER_ARTIFACT"
    SYSTEM_TEMP = "SYSTEM_TEMP"
    APPLICATION_LOCK = "APPLICATION_LOCK"
    TRANSACTION_TEMP = "TRANSACTION_TEMP"


@dataclass(frozen=True)
class DeleteSelectionRequest:
    scope: DeleteScope
    workspace: str
    pattern: str
    recursive: bool = False


@dataclass(frozen=True)
class DeleteTarget:
    absolute_canonical_path: str
    display_name: str
    size: int
    sha256: str = ""
    artifact_kind: DeleteArtifactKind = DeleteArtifactKind.USER_ARTIFACT


@dataclass(frozen=True)
class BatchActionResult:
    deleted: tuple[str, ...]
    failed: tuple[dict[str, str], ...]
    skipped: tuple[dict[str, str], ...]
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeletePlan:
    plan_id: str
    scope: DeleteScope
    workspace: str
    targets: tuple[DeleteTarget, ...]
    selection_rule: str
    requires_confirmation: bool = True
    confirmed: bool = False
    verification: str = "PENDING"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DeletePlan":
        return cls(
            plan_id=str(value.get("plan_id", "")),
            scope=DeleteScope(str(value.get("scope", DeleteScope.COLLECTION.value))),
            workspace=str(value.get("workspace", "")),
            targets=tuple(
                DeleteTarget(
                    absolute_canonical_path=str(item.get("absolute_canonical_path", "")),
                    display_name=str(item.get("display_name", "")),
                    size=int(item.get("size", 0)),
                    sha256=str(item.get("sha256", "")),
                    artifact_kind=DeleteArtifactKind(
                        str(item.get("artifact_kind", DeleteArtifactKind.USER_ARTIFACT.value))
                    ),
                )
                for item in value.get("targets", ())
            ),
            selection_rule=str(value.get("selection_rule", "")),
            requires_confirmation=bool(value.get("requires_confirmation", True)),
            confirmed=bool(value.get("confirmed", False)),
            verification=str(value.get("verification", "PENDING")),
        )


class CollectionResolver:
    """Freeze a delete collection from real filesystem search results."""

    @staticmethod
    def resolve(request: DeleteSelectionRequest, candidates: Sequence[str]) -> DeletePlan:
        workspace = Path(request.workspace).resolve(strict=True)
        targets: list[DeleteTarget] = []
        for raw in candidates:
            candidate = Path(raw).resolve(strict=True)
            try:
                relative = candidate.relative_to(workspace)
            except ValueError:
                continue
            if not candidate.is_file():
                continue
            if not request.recursive and len(relative.parts) != 1:
                continue
            if request.pattern != "*" and not candidate.match(request.pattern):
                continue
            size = candidate.stat().st_size
            targets.append(DeleteTarget(
                absolute_canonical_path=str(candidate),
                display_name=candidate.name,
                size=size,
                sha256=_sha256(candidate) if size <= 64 * 1024 * 1024 else "",
                artifact_kind=classify_delete_artifact(candidate),
            ))
        unique = {
            target.absolute_canonical_path.casefold(): target for target in targets
        }
        ordered = tuple(sorted(unique.values(), key=lambda item: item.absolute_canonical_path.casefold()))
        return DeletePlan(
            plan_id=f"delete:{uuid4().hex}",
            scope=request.scope,
            workspace=str(workspace),
            targets=ordered,
            selection_rule=request.pattern,
        )


def classify_delete_artifact(path: Path) -> DeleteArtifactKind:
    name = path.name.casefold()
    if name.startswith("~$"):
        return DeleteArtifactKind.APPLICATION_LOCK
    if name.startswith(".anydes-") or ".anydes-" in name:
        return DeleteArtifactKind.TRANSACTION_TEMP
    if name.startswith("~") or name.endswith((".tmp", ".temp", ".lock")):
        return DeleteArtifactKind.SYSTEM_TEMP
    return DeleteArtifactKind.USER_ARTIFACT


def target_matches_snapshot(target: DeleteTarget) -> bool:
    path = Path(target.absolute_canonical_path)
    if not path.is_file() or path.stat().st_size != target.size:
        return False
    return not target.sha256 or _sha256(path) == target.sha256


def parse_delete_selection(text: str, workspace: str) -> DeleteSelectionRequest | None:
    normalized = _normalize(text)
    if not re.search(r"\b(?:borr\w*|elimin\w*)\b", normalized):
        return None
    if not re.search(r"\b(?:todos?|todas?)\b", normalized):
        return None
    extension = re.search(r"\.([a-z0-9]{1,12})\b", normalized)
    if extension:
        return DeleteSelectionRequest(
            DeleteScope.BY_FILTER,
            str(Path(workspace).resolve(strict=False)),
            f"*.{extension.group(1).casefold()}",
        )
    if re.search(r"\b(?:archivos?|documentos?|elementos?)\b", normalized):
        return DeleteSelectionRequest(
            DeleteScope.COLLECTION,
            str(Path(workspace).resolve(strict=False)),
            "*",
        )
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize(value: str) -> str:
    import unicodedata

    decomposed = unicodedata.normalize("NFKD", str(value).casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))
