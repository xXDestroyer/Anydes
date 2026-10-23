from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol
from datetime import datetime, timezone


JsonObject = dict[str, Any]


class ToolStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    DISABLED = "DISABLED"
    BROKEN = "BROKEN"
    QUARANTINED = "QUARANTINED"
    MISSING_DEPENDENCY = "MISSING_DEPENDENCY"


class ExecutionStatus(StrEnum):
    SUCCESS = "SUCCESS"
    DENIED = "DENIED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    INVALID_INPUT = "INVALID_INPUT"
    INVALID_OUTPUT = "INVALID_OUTPUT"
    MISSING = "MISSING"
    VERIFY_FAILED = "VERIFY_FAILED"


class PermissionEffect(StrEnum):
    READ = "READ"
    CREATE = "CREATE"
    EDIT = "EDIT"
    OVERWRITE = "OVERWRITE"
    MOVE = "MOVE"
    DELETE = "DELETE"
    EXECUTE_CODE = "EXECUTE_CODE"
    INSTALL_DEPENDENCY = "INSTALL_DEPENDENCY"
    EXTERNAL_ACTION = "EXTERNAL_ACTION"


class CapabilityResolutionKind(StrEnum):
    EXACT_TOOL = "EXACT_TOOL"
    COMPOSABLE = "COMPOSABLE"
    REASONING_ONLY = "REASONING_ONLY"
    TEMPORARY_SCRIPT = "TEMPORARY_SCRIPT"
    MISSING_CAPABILITY = "MISSING_CAPABILITY"
    UNSUPPORTED = "UNSUPPORTED"
    FORBIDDEN = "FORBIDDEN"


@dataclass(frozen=True)
class ToolManifest:
    tool_id: str
    name: str
    version: str
    category: str
    description: str
    capabilities: tuple[str, ...]
    input_schema: JsonObject
    output_schema: JsonObject
    side_effects: tuple[str, ...]
    risk_level: str
    permission_policy: JsonObject
    dependencies: tuple[str, ...]
    timeout: float
    filesystem_scope: tuple[str, ...]
    status: ToolStatus
    capability_contracts: JsonObject = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: JsonObject) -> "ToolManifest":
        required = {
            "tool_id",
            "name",
            "version",
            "category",
            "description",
            "capabilities",
            "input_schema",
            "output_schema",
            "side_effects",
            "risk_level",
            "permission_policy",
            "dependencies",
            "timeout",
            "filesystem_scope",
            "status",
        }
        missing = sorted(required.difference(data))
        if missing:
            raise ValueError(f"manifest_missing_fields:{','.join(missing)}")
        tool_id = str(data["tool_id"]).strip()
        capabilities = tuple(str(item).strip() for item in data["capabilities"])
        if not tool_id or not capabilities or any(not item for item in capabilities):
            raise ValueError("manifest_invalid_identity_or_capabilities")
        if len(set(capabilities)) != len(capabilities):
            raise ValueError("manifest_duplicate_capability")
        timeout = float(data["timeout"])
        if timeout <= 0 or timeout > 300:
            raise ValueError("manifest_invalid_timeout")
        return cls(
            tool_id=tool_id,
            name=str(data["name"]),
            version=str(data["version"]),
            category=str(data["category"]),
            description=str(data["description"]),
            capabilities=capabilities,
            input_schema=dict(data["input_schema"]),
            output_schema=dict(data["output_schema"]),
            side_effects=tuple(str(item) for item in data["side_effects"]),
            risk_level=str(data["risk_level"]),
            permission_policy=dict(data["permission_policy"]),
            dependencies=tuple(str(item) for item in data["dependencies"]),
            timeout=timeout,
            filesystem_scope=tuple(str(item) for item in data["filesystem_scope"]),
            status=ToolStatus(str(data["status"])),
            capability_contracts=dict(data.get("capability_contracts", {})),
        )


@dataclass(frozen=True)
class CapabilityIntent:
    required_capabilities: tuple[str, ...] = ()
    operation: str = ""
    target_path: str = ""
    allow_temporary_script: bool = False


@dataclass(frozen=True)
class CapabilityResolution:
    kind: CapabilityResolutionKind
    tool_ids: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    reason: str = ""
    recipe_id: str = ""


@dataclass(frozen=True)
class ToolCapabilitySnapshot:
    source: str
    available_capabilities: tuple[str, ...]
    tools: tuple[JsonObject, ...]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass(frozen=True)
class ToolRequest:
    request_id: str
    turn_id: str
    tool_id: str
    capability: str
    arguments: JsonObject = field(default_factory=dict)
    authorized_scopes: tuple[str, ...] = ()
    confirmed_effects: tuple[PermissionEffect, ...] = ()
    reflection_approved: bool = False
    explicit_user_request: bool = False


@dataclass(frozen=True)
class FileEffect:
    effect: PermissionEffect
    path: str
    scope: str = "USER_TASK"
    user_visible: bool = True
    task_semantic: bool = True
    committed: bool = True


@dataclass(frozen=True)
class ToolExecutionResult:
    request_id: str
    turn_id: str
    tool_id: str
    capability: str
    status: ExecutionStatus
    output: JsonObject = field(default_factory=dict)
    error: str = ""
    duration_ms: float = 0.0
    side_effects: tuple[FileEffect, ...] = ()
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class CompositionStep:
    step_id: str
    capability: str
    arguments: JsonObject


@dataclass(frozen=True)
class CompositionPlan:
    plan_id: str
    steps: tuple[CompositionStep, ...]


@dataclass(frozen=True)
class CompositionResult:
    plan_id: str
    status: ExecutionStatus
    steps: tuple[ToolExecutionResult, ...]
    error: str = ""


@dataclass(frozen=True)
class CapabilityGap:
    required_capability: str
    reason: str


@dataclass(frozen=True)
class ToolSpecification:
    tool_id: str
    capabilities: tuple[str, ...]
    constraints: JsonObject = field(default_factory=dict)


@dataclass(frozen=True)
class GeneratedToolCandidate:
    specification: ToolSpecification
    source_path: str
    enabled: bool = False


@dataclass(frozen=True)
class ToolValidationResult:
    accepted: bool
    issues: tuple[str, ...] = ()


class OfficialTool(Protocol):
    manifest: ToolManifest

    def effect_for(self, capability: str, arguments: JsonObject) -> PermissionEffect:
        ...

    def execute(self, capability: str, arguments: JsonObject, context: Any) -> JsonObject:
        ...

    def health(self) -> tuple[ToolStatus, str]:
        ...
