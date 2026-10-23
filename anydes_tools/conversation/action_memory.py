from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import threading
import unicodedata
from typing import TYPE_CHECKING, Any, Sequence
from uuid import uuid4

from ..contracts import ExecutionStatus, ToolExecutionResult
from .artifact_state import RecentArtifactState
if TYPE_CHECKING:
    from .contracts import PlannedToolAction


@dataclass(frozen=True)
class ExperienceEvent:
    event_id: str
    timestamp: str
    turn_id: str
    task_id: str
    action_type: str
    tool_id: str
    capability: str
    artifact_ids: tuple[str, ...]
    source_artifact_ids: tuple[str, ...]
    target_artifact_ids: tuple[str, ...]
    user_request_summary: str
    action_summary: str
    result_status: str
    verification_status: str
    workspace_id: str
    topic: str
    entities: tuple[str, ...]
    error_code: str = ""
    provenance: str = "ACTION_ATTEMPT"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionMemoryItem:
    event_id: str
    timestamp: str
    action_summary: str
    result_status: str
    artifact_ids: tuple[str, ...]
    topic: str
    provenance: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RecentActionState:
    limit: int = 20
    events: list[ExperienceEvent] = field(default_factory=list)

    def remember(self, event: ExperienceEvent) -> None:
        self.events.append(event)
        del self.events[:-self.limit]

    def snapshot(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.events]


class ActionExperienceMemory:
    """Append-only action evidence plus a small selected conversational memory."""

    def __init__(self, tools_root: Path, recent_state: RecentActionState | None = None) -> None:
        self.tools_root = tools_root.resolve()
        self.log_path = self.tools_root / "logs" / "action-experience-v001_5.jsonl"
        self.memory_path = self.tools_root / "memory" / "action-memory-v001_5.jsonl"
        self.recent = recent_state or RecentActionState()
        self._lock = threading.RLock()

    def record(
        self,
        action: PlannedToolAction,
        results: Sequence[ToolExecutionResult],
        *,
        turn_id: str,
        verification: dict[str, Any],
        performed_success: bool,
        workspace: str,
        artifacts: RecentArtifactState,
    ) -> ExperienceEvent:
        source_paths, target_paths = _artifact_paths(action, results)
        source_ids = _artifact_ids(source_paths, artifacts)
        target_ids = _artifact_ids(target_paths, artifacts)
        all_ids = tuple(dict.fromkeys((*source_ids, *target_ids)))
        failure = next((item for item in results if item.status != ExecutionStatus.SUCCESS), None)
        result_status = "SUCCESS" if performed_success else "FAILED"
        topic = _task_topic(action)
        event = ExperienceEvent(
            event_id=f"experience:{uuid4().hex}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            turn_id=str(turn_id),
            task_id=(action.task_spec.task_id if action.task_spec else ""),
            action_type=action.intent or action.operation,
            tool_id=(
                action.structured_action_plan.tool_id
                if action.structured_action_plan is not None
                else (results[-1].tool_id if results else "")
            ),
            capability=(
                action.structured_action_plan.capability
                if action.structured_action_plan is not None
                else (action.capabilities[0] if action.capabilities else "")
            ),
            artifact_ids=all_ids,
            source_artifact_ids=source_ids,
            target_artifact_ids=target_ids,
            user_request_summary=_safe_summary(action.source_text),
            action_summary=_action_summary(action, source_paths, target_paths, result_status),
            result_status=result_status,
            verification_status=str(verification.get("status", "NOT_REQUIRED")),
            workspace_id=Path(workspace).name,
            topic=topic,
            entities=tuple(Path(value).name for value in (*source_paths, *target_paths)),
            error_code=str(failure.error if failure is not None else ""),
            provenance=("DIRECT_ACTION_EXPERIENCE" if performed_success else "ACTION_ATTEMPT"),
        )
        with self._lock:
            self.recent.remember(event)
            _append_jsonl(self.log_path, event.to_dict())
            if _should_promote(action, event):
                memory = ActionMemoryItem(
                    event.event_id,
                    event.timestamp,
                    event.action_summary,
                    event.result_status,
                    event.artifact_ids,
                    event.topic,
                    event.provenance,
                )
                _append_jsonl(self.memory_path, memory.to_dict())
        return event

    def retrieve(self, query: str, *, limit: int = 5) -> tuple[ExperienceEvent, ...]:
        terms = _query_terms(query)
        with self._lock:
            values = list(self.recent.events)
            values.extend(_load_events(self.log_path, tail=800))
        unique: dict[str, ExperienceEvent] = {item.event_id: item for item in values}
        scored: list[tuple[int, str, ExperienceEvent]] = []
        for event in unique.values():
            searchable = " ".join((
                event.action_summary,
                event.user_request_summary,
                event.topic,
                " ".join(event.entities),
            ))
            overlap = len(terms & _query_terms(searchable))
            if overlap or not terms:
                status_bonus = 2 if event.result_status == "SUCCESS" else 0
                scored.append((overlap * 4 + status_bonus, event.timestamp, event))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return tuple(item[2] for item in scored[:limit])

    @staticmethod
    def is_retrieval_query(text: str) -> bool:
        normalized = _fold(text)
        return bool(re.search(
            r"\b(?:recuerdas?\s+que\s+(?:creaste|editaste|renombraste|convertiste|"
            r"copiaste|eliminaste|hiciste)|recordar\s+(?:la\s+)?accion|"
            r"que\s+(?:archivo|documento|cambio|accion)\w*\s+"
            r"(?:creamos|hiciste|realizaste|editaste|cambiaste)|que\s+le\s+cambiaste|"
            r"archivos?\s+creamos)\b",
            normalized,
        ))


def evidence_payload(events: Sequence[ExperienceEvent]) -> list[dict[str, Any]]:
    return [
        {
            "event_id": item.event_id,
            "timestamp": item.timestamp,
            "action_summary": item.action_summary,
            "result_status": item.result_status,
            "verification_status": item.verification_status,
            "entities": list(item.entities),
            "provenance": item.provenance,
        }
        for item in events
    ]


def _artifact_paths(
    action: PlannedToolAction,
    results: Sequence[ToolExecutionResult],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    args = action.arguments
    source: list[str] = []
    target: list[str] = []
    if action.operation in {"copy_existing", "rename_existing"}:
        source.append(str(args.get("source") or args.get("path") or ""))
        target.append(str(args.get("destination") or ""))
    elif action.operation == "document_conversion":
        spec = dict(args.get("conversion_spec", {}))
        source.append(str(spec.get("source_path", "")))
        target.append(str(spec.get("target_path", "")))
    elif action.intent in {"delete", "delete_collection"}:
        if action.operation == "delete_collection":
            target.extend(
                str(item.get("absolute_canonical_path", ""))
                for item in args.get("delete_plan", {}).get("targets", ())
            )
        else:
            target.append(str(args.get("path") or action.target_path or ""))
    else:
        target.append(str(args.get("path") or args.get("destination") or action.target_path or ""))
    for result in results:
        output = result.output
        if action.operation in {"copy_existing", "rename_existing"}:
            if output.get("source"):
                source.append(str(output["source"]))
            if output.get("destination"):
                target.append(str(output["destination"]))
    return _canonical_nonempty(source), _canonical_nonempty(target)


def _canonical_nonempty(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        str(Path(value).resolve(strict=False)) for value in values if str(value).strip()
    ))


def _artifact_ids(paths: Sequence[str], artifacts: RecentArtifactState) -> tuple[str, ...]:
    values = []
    for path in paths:
        item = artifacts.find_by_path(path)
        if item is not None:
            values.append(item.artifact_id)
    return tuple(dict.fromkeys(values))


def _task_topic(action: PlannedToolAction) -> str:
    if action.task_spec is None:
        return ""
    generation = dict(action.task_spec.content_requirements.get("content_generation_spec", {}))
    return str(generation.get("topic", ""))[:300]


def _action_summary(
    action: PlannedToolAction,
    sources: Sequence[str],
    targets: Sequence[str],
    status: str,
) -> str:
    source_names = ", ".join(Path(value).name for value in sources)
    target_names = ", ".join(Path(value).name for value in targets)
    object_name = target_names or source_names or "la acción solicitada"
    verbs = {
        "create_document": "crear",
        "generate_document": "crear",
        "structural_document_edit": "editar",
        "generate_document_edit": "editar",
        "structured_edit_text": "editar",
        "document_transform": "editar",
        "rename": "cambiar el nombre de",
        "copy_artifact": "duplicar",
        "convert_document": "convertir",
        "delete": "eliminar",
        "delete_collection": "eliminar",
    }
    verb = verbs.get(action.intent, action.intent.replace("_", " ") or "realizar")
    prefix = "Se completó" if status == "SUCCESS" else "Falló el intento de"
    return f"{prefix} {verb} {object_name}".strip()


def _should_promote(action: PlannedToolAction, event: ExperienceEvent) -> bool:
    if event.result_status != "SUCCESS":
        return False
    if action.intent in {"read_document", "show_full_path", "inspect_error"}:
        return False
    return bool(event.artifact_ids or event.topic or action.intent in {
        "rename", "copy_artifact", "convert_document", "delete", "delete_collection",
    })


def _safe_summary(value: str) -> str:
    text = re.sub(r"(?i)(?:api[_-]?key|token|password|secret)\s*[:=]\s*\S+", "[REDACTED]", str(value))
    return re.sub(r"\s+", " ", text).strip()[:500]


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _load_events(path: Path, *, tail: int) -> list[ExperienceEvent]:
    if not path.is_file():
        return []
    rows = path.read_text(encoding="utf-8").splitlines()[-tail:]
    events: list[ExperienceEvent] = []
    for row in rows:
        try:
            value = json.loads(row)
            value["artifact_ids"] = tuple(value.get("artifact_ids", ()))
            value["source_artifact_ids"] = tuple(value.get("source_artifact_ids", ()))
            value["target_artifact_ids"] = tuple(value.get("target_artifact_ids", ()))
            value["entities"] = tuple(value.get("entities", ()))
            events.append(ExperienceEvent(**value))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    return events


def _query_terms(value: str) -> set[str]:
    ignored = {"que", "cual", "como", "cuando", "recuerdas", "hiciste", "creamos", "ayer", "archivo", "documento"}
    return {
        token for token in re.findall(r"[a-z0-9]+", _fold(value))
        if len(token) >= 3 and token not in ignored
    }


def _fold(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value).casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))
