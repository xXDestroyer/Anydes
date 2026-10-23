from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any

from ..contracts import CapabilityResolutionKind, PermissionEffect
from .artifact_state import RecentArtifactState
from .action_memory import RecentActionState
from .failure_state import LastToolFailure, ActiveFailureTopic


@dataclass(frozen=True)
class ActiveDocumentReference:
    absolute_canonical_path: str
    display_name: str
    workspace: str
    file_type: str


@dataclass(frozen=True)
class UserTaskSpec:
    task_id: str
    goal: str
    target_type: str
    output_format: str
    filename: str
    content_requirements: dict[str, Any] = field(default_factory=dict)
    layout_requirements: dict[str, Any] = field(default_factory=dict)
    constraints: tuple[str, ...] = ()
    destination: str = ""
    status: str = "PLANNED"


@dataclass(frozen=True)
class StructuredActionPlan:
    plan_id: str
    action_type: str
    tool_id: str
    capability: str
    arguments: dict[str, Any]
    workspace: str
    task_id: str
    status: str = "VALIDATED"


@dataclass(frozen=True)
class PlannedToolAction:
    intent: str
    capabilities: tuple[str, ...] = ()
    operation: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    target_path: str = ""
    source_text: str = ""
    explicit_request: bool = True
    task_spec: UserTaskSpec | None = None
    structured_action_plan: StructuredActionPlan | None = None
    resolved_artifacts: tuple[Any, ...] = ()
    resolved_artifact_bindings: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PendingToolAction:
    action: PlannedToolAction
    reason: str
    confirmed_effects: tuple[PermissionEffect, ...] = ()
    scopes_to_authorize: tuple[str, ...] = ()
    origin_turn_id: str = ""
    pending_request_id: str = ""
    task_id: str = ""
    plan_id: str = ""
    confirmation_expected: bool = True
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 900.0)
    state: str = "WAITING_CONFIRMATION"


@dataclass
class PendingTaskState:
    task_id: str = ""
    plan_revision: int = 1
    status: str = "IDLE"
    pending_control: str = "NONE"
    required_confirmation: PendingToolAction | None = None
    failure_ref: LastToolFailure | None = None
    plan: Any = None
    action: PlannedToolAction | None = None
    requirements: UserTaskSpec | None = None
    structured_compat: StructuredActionPlan | None = None
    failed_action: PlannedToolAction | None = None
    failed_plan: Any = None
    committed_effects: bool = False
    retry_count: int = 0
    attempt_number: int = 1
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class ConversationToolState:
    active_workspace: str
    authorized_scopes: tuple[str, ...] = ()
    active_document: str = ""
    active_document_ref: ActiveDocumentReference | None = None
    source_document: str = ""
    related_artifacts: list[dict[str, str]] = field(default_factory=list)
    active_code_file: str = ""
    recent_files: list[str] = field(default_factory=list)
    recent_tool_results: list[dict[str, Any]] = field(default_factory=list)
    pending_task: PendingTaskState = field(default_factory=PendingTaskState)
    failure_topic_active: bool = False
    failure_explanation_depth: int = 0
    active_failure_topic: ActiveFailureTopic | None = None
    latest_runtime_error: LastToolFailure | None = None
    previous_assistant_act: str = 'NONE'
    last_error: dict[str, Any] = field(default_factory=dict)
    last_verification: dict[str, Any] = field(default_factory=dict)
    failure_history: list[LastToolFailure] = field(default_factory=list)
    verification_evidence: dict[str, Any] = field(default_factory=dict)
    failure_plans: dict[str, Any] = field(default_factory=dict)
    attempt_history: list[Any] = field(default_factory=list)
    replan_count: int = 0
    recent_artifact_state: RecentArtifactState = field(default_factory=RecentArtifactState)
    recent_action_state: RecentActionState = field(default_factory=RecentActionState)

    # Deprecated accessors: all legacy consumers read/write the same state owner.
    @property
    def pending_tool_action(self):
        return self.pending_task.required_confirmation

    @pending_tool_action.setter
    def pending_tool_action(self, value):
        if value is not None and not value.task_id:
            from dataclasses import replace
            value = replace(value, task_id=self.pending_task.task_id,
                plan_id=value.plan_id or f"plan:{self.pending_task.task_id}:r{self.pending_task.plan_revision}")
        self.pending_task.required_confirmation = value
        self.pending_task.pending_control = "CONFIRM" if value else "NONE"
        if value:
            self.pending_task.status = value.state
            self.pending_task.updated_at = time.time()

    @property
    def last_tool_failure(self):
        return self.pending_task.failure_ref

    @last_tool_failure.setter
    def last_tool_failure(self, value):
        self.pending_task.failure_ref = value
        self.latest_runtime_error = value or self.latest_runtime_error
        self.failure_topic_active = value is not None
        self.failure_explanation_depth = 0
        if value is not None:
            self.active_failure_topic = ActiveFailureTopic.from_failure(value)
        elif self.active_failure_topic is not None:
            self.active_failure_topic.topic_status = 'RESOLVED'
        if value is not None and not any(f.failure_id == value.failure_id for f in self.failure_history):
            self.failure_history.append(value)
            del self.failure_history[:-32]

    def failure_topic_record(self):
        topic = self.active_failure_topic
        if topic and topic.topic_status == 'ACTIVE' and self.failure_topic_active:
            return next((f for f in self.failure_history if f.failure_id == topic.root_failure_id), None)
        return None

    def record_explanation_error(self, failure):
        self.latest_runtime_error = failure
        self.failure_history.append(failure)
        # Preserve the root evidence even under repeated secondary failures.
        root = self.failure_topic_record()
        self.failure_history[:] = self.failure_history[-32:]
        if root and root not in self.failure_history:
            self.failure_history.insert(0, root)

    @property
    def last_failed_action(self):
        return self.pending_task.failed_action

    @last_failed_action.setter
    def last_failed_action(self, value):
        self.pending_task.failed_action = value

    @property
    def retry_count(self):
        return self.pending_task.retry_count

    @retry_count.setter
    def retry_count(self, value):
        self.pending_task.retry_count = value

    @property
    def pending_user_task(self):
        return self.pending_task.requirements

    @pending_user_task.setter
    def pending_user_task(self, value):
        self.pending_task.requirements = value

    @property
    def pending_structured_plan(self):
        return self.pending_task.structured_compat

    @pending_structured_plan.setter
    def pending_structured_plan(self, value):
        self.pending_task.structured_compat = value

    @property
    def pending_task_action(self):
        return self.pending_task.action

    @pending_task_action.setter
    def pending_task_action(self, value):
        self.pending_task.action = value

    def remember_file(self, path: str) -> None:
        resolved = str(Path(path).resolve())
        self.recent_files = [item for item in self.recent_files if item != resolved]
        self.recent_files.append(resolved)
        del self.recent_files[:-12]
        suffix = Path(resolved).suffix.casefold()
        if suffix in {".txt", ".md", ".docx", ".pdf"}:
            topic = ""
            task_id = ""
            if self.pending_user_task is not None:
                task_id = self.pending_user_task.task_id
                generation = dict(self.pending_user_task.content_requirements.get("content_generation_spec", {}))
                topic = str(generation.get("topic", ""))
            # Compatibility path notification; the commit host supplies the canonical ref.
            self.recent_artifact_state.remember(resolved, task_id=task_id, semantic_topic=topic)
            self.active_document = resolved
            self.active_document_ref = ActiveDocumentReference(
                absolute_canonical_path=resolved,
                display_name=Path(resolved).name,
                workspace=str(Path(resolved).parent),
                file_type=suffix.removeprefix("."),
            )
        if suffix in {".py", ".cpp", ".cc", ".cxx"}:
            self.active_code_file = resolved

    def remember_result(self, result: dict[str, Any]) -> None:
        self.recent_tool_results.append(result)
        del self.recent_tool_results[:-12]

    def remember_conversion(self, source: str, target: str) -> None:
        source_path = str(Path(source).resolve(strict=False))
        target_path = str(Path(target).resolve(strict=False))
        self.source_document = source_path
        self.related_artifacts.append({
            "converted_from": source_path,
            "converted_to": target_path,
        })
        del self.related_artifacts[:-12]
        self.remember_file(source_path)
        self.remember_file(target_path)
        self.recent_artifact_state.remember(source_path, relationship={"converted_to": target_path})
        self.recent_artifact_state.remember(target_path, relationship={"converted_from": source_path})


@dataclass(frozen=True)
class ConversationToolResponse:
    handled: bool
    text: str
    resolver_kind: CapabilityResolutionKind | None = None
    selected_tools: tuple[str, ...] = ()
    executed: bool = False
    debug: dict[str, Any] = field(default_factory=dict)
