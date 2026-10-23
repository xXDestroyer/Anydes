"""Host-owned, task-local bindings. Missing bindings never trigger path search."""
from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
from pathlib import Path
from uuid import uuid4


class DataflowError(ValueError):
    pass


class ArtifactState(str, Enum):
    TEMPORARY = "TEMPORARY"
    VERIFIED = "VERIFIED"
    COMMITTED = "COMMITTED"
    COMMITTED_UNVERIFIED = "COMMITTED_UNVERIFIED"
    FAILED = "FAILED"
    DELETED = "DELETED"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_version(path):
    path = Path(path)
    return file_hash(path) if path.is_file() else "ABSENT"


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    artifact_type: str
    format: str
    producer_step_id: str
    task_id: str
    version: int
    hash: str
    state: ArtifactState
    temporary_path_ref: str = ""
    committed_path_ref: str = ""
    display_name: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def path(self):
        return self.committed_path_ref or self.temporary_path_ref


def validate_dataflow(plan):
    available = set(plan.typed_inputs.get("artifact_bindings", {}))
    types = {key: ref.artifact_type for key, ref in plan.typed_inputs.get("artifact_bindings", {}).items()}
    values = set(plan.typed_inputs.get("value_bindings", {}))
    seen = set()
    for step in plan.steps:
        if step.step_id in seen or any(d not in seen for d in step.depends_on):
            raise DataflowError("DATAFLOW_DEPENDENCY_ORDER:" + step.step_id)
        if len(set(step.consumes_artifacts)) != len(step.consumes_artifacts):
            raise DataflowError("DATAFLOW_DUPLICATE_INPUT:" + step.step_id)
        for binding in step.consumes_artifacts:
            if binding not in available:
                raise DataflowError("DATAFLOW_MISSING_PRODUCER:" + binding)
            if step.capability_id.startswith("developer.") and types.get(binding) not in {None, "CODE"}:
                raise DataflowError("DATAFLOW_TYPE_MISMATCH:" + binding)
        for binding in step.produces_artifacts:
            if not binding or binding in available:
                raise DataflowError("DATAFLOW_AMBIGUOUS_PRODUCER:" + binding)
            available.add(binding)
            if step.capability_id == "developer.write_code":
                types[binding] = "CODE"
            elif step.capability_id.startswith("document.create"):
                types[binding] = "DOCUMENT"
            elif step.capability_id == "archive.compress_zip":
                types[binding] = "ARCHIVE"
        for binding in step.consumes_values:
            if binding not in values:
                raise DataflowError("DATAFLOW_MISSING_VALUE_PRODUCER:" + binding)
        for binding in step.produces_values:
            if not binding or binding in values:
                raise DataflowError("DATAFLOW_AMBIGUOUS_VALUE_PRODUCER:" + binding)
            values.add(binding)
        seen.add(step.step_id)


class ArtifactBindings:
    def __init__(self, task_id, initial=None):
        self.task_id = task_id
        self.bindings = dict(initial or {})
        self.artifacts = {a.artifact_id: a for a in self.bindings.values()}
        self.values = {}
        self.path_snapshots = {}

    def snapshot(self, path):
        canonical = str(Path(path).resolve(strict=False))
        self.path_snapshots.setdefault(canonical, path_version(canonical))
        return self.path_snapshots[canonical]

    def require(self, binding, *, verified=False, artifact_type="", allow_unverified=False):
        ref = self.bindings.get(binding)
        if ref is None:
            raise DataflowError("DATAFLOW_MISSING_ARTIFACT:" + binding)
        if ref.state in {ArtifactState.FAILED, ArtifactState.DELETED}:
            raise DataflowError("DATAFLOW_FAILED_ARTIFACT:" + ref.artifact_id)
        if verified and ref.state not in {ArtifactState.VERIFIED, ArtifactState.COMMITTED} and not (
                allow_unverified and ref.state == ArtifactState.COMMITTED_UNVERIFIED
                and ref.metadata.get('user_accepted_unverified') is True):
            raise DataflowError("DATAFLOW_UNVERIFIED_ARTIFACT:" + ref.artifact_id)
        if artifact_type and ref.artifact_type != artifact_type:
            raise DataflowError("DATAFLOW_TYPE_MISMATCH:" + ref.artifact_id)
        try:
            if Path(ref.path).resolve(strict=True) != Path(ref.path):
                raise DataflowError("STALE_ARTIFACT:canonical_path_changed")
        except OSError as exc:
            raise DataflowError("STALE_ARTIFACT:unavailable") from exc
        if path_version(ref.path) != ref.hash:
            raise DataflowError("STALE_ARTIFACT:" + ref.artifact_id)
        return ref

    def produce(self, binding, path, step_id, *, state=ArtifactState.TEMPORARY,
                artifact_type="", metadata=None):
        if binding in self.bindings:
            raise DataflowError("DATAFLOW_AMBIGUOUS_PRODUCER:" + binding)
        canonical = str(Path(path).resolve(strict=True))
        suffix = Path(canonical).suffix.lower().lstrip(".")
        kind = artifact_type or ("CODE" if suffix in {"py", "cpp", "h", "c"}
                                else "ARCHIVE" if suffix == "zip" else "DOCUMENT")
        ref = ArtifactRef(f"artifact:{uuid4().hex}", kind, suffix, step_id, self.task_id,
            1, file_hash(canonical), state,
            canonical if state != ArtifactState.COMMITTED else "",
            canonical if state == ArtifactState.COMMITTED else "", Path(canonical).name,
            dict(metadata or {}))
        self.bindings[binding] = ref
        self.artifacts[ref.artifact_id] = ref
        return ref

    def transition(self, binding, state, *, path="", new_version=False):
        old = self.bindings[binding]
        canonical = str(Path(path).resolve(strict=True)) if path else old.path
        ref = replace(old, state=state, hash=file_hash(canonical),
            version=old.version + int(new_version),
            temporary_path_ref="" if state == ArtifactState.COMMITTED else canonical,
            committed_path_ref=canonical if state == ArtifactState.COMMITTED else "",
            display_name=Path(canonical).name)
        self.bindings[binding] = ref
        self.artifacts[ref.artifact_id] = ref
        return ref

    def resolve_arguments(self, step):
        allow_unverified = step.capability_id in {'archive.compress_zip','developer.read_code','developer.write_code',
            'developer.run_python','developer.run_cpp','developer.test','filesystem.read','filesystem.inspect'}
        refs = [self.require(b, verified=step.requires_verified_inputs, allow_unverified=allow_unverified)
                for b in step.consumes_artifacts]
        if step.capability_id.startswith("developer.") and any(r.artifact_type != "CODE" for r in refs):
            raise DataflowError("DATAFLOW_TYPE_MISMATCH:" + step.step_id)
        def resolve(value):
            if isinstance(value, dict):
                return {k: resolve(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [resolve(v) for v in value]
            if isinstance(value, str) and value.startswith("$artifacts."):
                binding = value[len("$artifacts."):]
                if binding not in step.consumes_artifacts:
                    raise DataflowError("DATAFLOW_UNDECLARED_INPUT:" + binding)
                return self.require(binding, verified=step.requires_verified_inputs).path
            if isinstance(value, str) and value.startswith("$values."):
                key = value[len("$values."):]
                if key not in step.consumes_values:
                    raise DataflowError("DATAFLOW_UNDECLARED_VALUE:" + key)
                if key not in self.values:
                    raise DataflowError("DATAFLOW_MISSING_VALUE:" + key)
                return self.values[key]
            return value
        args = resolve(step.args)
        for binding in step.consumes_values:
            if binding not in self.values:
                raise DataflowError("DATAFLOW_MISSING_VALUE:" + binding)
        if refs:
            # Input slots derive from the capability contract, never from topic text.
            cap = step.capability_id
            if cap == "archive.compress_zip":
                args["sources"] = [ref.path for ref in refs]
            elif cap in {"filesystem.copy", "filesystem.move", "filesystem.rename"}:
                if len(refs) != 1:
                    raise DataflowError("DATAFLOW_SINGLE_INPUT_REQUIRED")
                args["source"] = refs[0].path
            elif cap.startswith(("document.read", "developer.run", "developer.test",
                                  "developer.read", "document.edit", "filesystem.read")):
                if len(refs) != 1:
                    raise DataflowError("DATAFLOW_SINGLE_INPUT_REQUIRED")
                args["path"] = refs[0].path
        return args, refs
