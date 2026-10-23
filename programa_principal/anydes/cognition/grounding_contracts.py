from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from anydes.primitives import JsonValue, clamp01


InteractionKind = Literal[
    "user_turn",
    "proactive_opportunity",
    "tool_result",
    "system_event",
    "recovery",
]
EvidenceSource = Literal[
    "user",
    "self",
    "memory",
    "relationship",
    "observation",
    "tool",
    "emotion",
    "goal",
    "body",
    "environment",
    "system",
]
EvidenceFreshness = Literal["fresh", "stale", "timeless", "unknown"]
EvidencePrivacy = Literal["normal", "personal", "protected"]
CoverageStatus = Literal[
    "complete",
    "partial",
    "stale",
    "unavailable",
    "unauthorized",
    "not_consulted",
]
EpistemicStatus = Literal[
    "known",
    "unknown",
    "uncertain",
    "unavailable",
    "not_applicable",
]
PrimaryObligation = Literal[
    "answer",
    "acknowledge",
    "clarify",
    "act",
    "observe",
    "listen",
    "initiate",
    "recover",
]
ClaimRole = Literal["answer", "support", "context", "caveat"]
EngagementMode = Literal["speak", "listen", "silence"]
ActionPhase = Literal[
    "none",
    "request",
    "awaiting_result",
    "answer_from_result",
]


def grounding_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_ready(value: Any) -> JsonValue:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _metadata(value: dict[str, Any] | None) -> dict[str, JsonValue]:
    ready = _json_ready(value or {})
    return ready if isinstance(ready, dict) else {}


class GroundingContractMixin:
    contract_version: str = "1.0"

    def to_dict(self) -> dict[str, JsonValue]:
        ready = _json_ready(self)
        if not isinstance(ready, dict):
            raise TypeError("Grounding contract serialization produced a non-object.")
        return ready

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=True, sort_keys=True)


@dataclass(frozen=True)
class InteractionFrame(GroundingContractMixin):
    id: str
    kind: InteractionKind
    primary_observation_id: str = ""
    speaker_ref: str = "unknown"
    channel: str = "system"
    response_expected: bool = True
    silence_allowed: bool = False
    tool_cycle_id: str = ""
    constraints: tuple[str, ...] = ()
    occurred_at: str = field(default_factory=grounding_utc_now)
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in {
            "user_turn",
            "proactive_opportunity",
            "tool_result",
            "system_event",
            "recovery",
        }:
            raise ValueError(f"Unsupported interaction kind: {self.kind}")
        object.__setattr__(self, "constraints", tuple(self.constraints))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InteractionFrame":
        return cls(
            id=str(data["id"]),
            kind=data.get("kind", "user_turn"),
            primary_observation_id=str(data.get("primary_observation_id", "")),
            speaker_ref=str(data.get("speaker_ref", "unknown")),
            channel=str(data.get("channel", "system")),
            response_expected=bool(data.get("response_expected", True)),
            silence_allowed=bool(data.get("silence_allowed", False)),
            tool_cycle_id=str(data.get("tool_cycle_id", "")),
            constraints=tuple(str(item) for item in data.get("constraints", [])),
            occurred_at=str(data.get("occurred_at") or grounding_utc_now()),
            metadata=_metadata(data.get("metadata")),
        )


@dataclass(frozen=True)
class EvidenceItem(GroundingContractMixin):
    id: str
    source: EvidenceSource
    source_ref: str
    subject: str
    predicate: str
    value: JsonValue = None
    summary: str = ""
    confidence: float = 1.0
    relevance: float = 0.5
    freshness: EvidenceFreshness = "fresh"
    observed_at: str = ""
    expires_at: str = ""
    privacy_scope: EvidencePrivacy = "normal"
    authorized: bool = True
    modality: str = "state"
    supports: tuple[str, ...] = ()
    contradicts: tuple[str, ...] = ()
    attributes: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source not in {
            "user",
            "self",
            "memory",
            "relationship",
            "observation",
            "tool",
            "emotion",
            "goal",
            "body",
            "environment",
            "system",
        }:
            raise ValueError(f"Unsupported evidence source: {self.source}")
        if self.freshness not in {"fresh", "stale", "timeless", "unknown"}:
            raise ValueError(f"Unsupported evidence freshness: {self.freshness}")
        if self.privacy_scope not in {"normal", "personal", "protected"}:
            raise ValueError(f"Unsupported evidence privacy: {self.privacy_scope}")
        object.__setattr__(self, "value", _json_ready(self.value))
        object.__setattr__(self, "confidence", clamp01(self.confidence))
        object.__setattr__(self, "relevance", clamp01(self.relevance))
        object.__setattr__(self, "supports", tuple(self.supports))
        object.__setattr__(self, "contradicts", tuple(self.contradicts))
        object.__setattr__(self, "attributes", _metadata(self.attributes))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceItem":
        return cls(
            id=str(data["id"]),
            source=data.get("source", "observation"),
            source_ref=str(data.get("source_ref", "")),
            subject=str(data.get("subject", "")),
            predicate=str(data.get("predicate", "")),
            value=_json_ready(data.get("value")),
            summary=str(data.get("summary", "")),
            confidence=float(data.get("confidence", 1.0)),
            relevance=float(data.get("relevance", 0.5)),
            freshness=data.get("freshness", "fresh"),
            observed_at=str(data.get("observed_at", "")),
            expires_at=str(data.get("expires_at", "")),
            privacy_scope=data.get("privacy_scope", "normal"),
            authorized=bool(data.get("authorized", True)),
            modality=str(data.get("modality", "state")),
            supports=tuple(str(item) for item in data.get("supports", [])),
            contradicts=tuple(str(item) for item in data.get("contradicts", [])),
            attributes=_metadata(data.get("attributes")),
        )


@dataclass(frozen=True)
class KnowledgeGap(GroundingContractMixin):
    id: str
    subject: str
    predicate: str
    status: EpistemicStatus = "unknown"
    reason: str = ""
    consulted_sources: tuple[str, ...] = ()
    related_evidence_ids: tuple[str, ...] = ()
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.status not in {"unknown", "uncertain", "unavailable"}:
            raise ValueError("KnowledgeGap status must describe missing knowledge.")
        object.__setattr__(self, "consulted_sources", tuple(self.consulted_sources))
        object.__setattr__(
            self,
            "related_evidence_ids",
            tuple(self.related_evidence_ids),
        )
        object.__setattr__(self, "confidence", clamp01(self.confidence))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnowledgeGap":
        return cls(
            id=str(data["id"]),
            subject=str(data.get("subject", "")),
            predicate=str(data.get("predicate", "")),
            status=data.get("status", "unknown"),
            reason=str(data.get("reason", "")),
            consulted_sources=tuple(
                str(item) for item in data.get("consulted_sources", [])
            ),
            related_evidence_ids=tuple(
                str(item) for item in data.get("related_evidence_ids", [])
            ),
            confidence=float(data.get("confidence", 1.0)),
        )


@dataclass(frozen=True)
class CapabilityEvidence(GroundingContractMixin):
    id: str
    action_type: str
    description: str = ""
    availability: str = "available"
    requires_confirmation: bool = False
    risk_level: str = "low"
    parameter_names: tuple[str, ...] = ()
    source_ref: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameter_names", tuple(self.parameter_names))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CapabilityEvidence":
        return cls(
            id=str(data["id"]),
            action_type=str(data.get("action_type", "")),
            description=str(data.get("description", "")),
            availability=str(data.get("availability", "available")),
            requires_confirmation=bool(data.get("requires_confirmation", False)),
            risk_level=str(data.get("risk_level", "low")),
            parameter_names=tuple(
                str(item) for item in data.get("parameter_names", [])
            ),
            source_ref=str(data.get("source_ref", "")),
        )


@dataclass(frozen=True)
class EvidenceCoverage(GroundingContractMixin):
    source: str
    snapshot_revision: str = ""
    available_count: int = 0
    selected_count: int = 0
    status: CoverageStatus = "complete"
    omitted_reason: str = ""

    def __post_init__(self) -> None:
        available = max(0, int(self.available_count))
        selected = max(0, int(self.selected_count))
        if selected > available and self.status not in {"unavailable", "not_consulted"}:
            raise ValueError("EvidenceCoverage selected_count exceeds available_count.")
        object.__setattr__(self, "available_count", available)
        object.__setattr__(self, "selected_count", selected)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceCoverage":
        return cls(
            source=str(data.get("source", "")),
            snapshot_revision=str(data.get("snapshot_revision", "")),
            available_count=int(data.get("available_count", 0)),
            selected_count=int(data.get("selected_count", 0)),
            status=data.get("status", "complete"),
            omitted_reason=str(data.get("omitted_reason", "")),
        )


@dataclass(frozen=True)
class EvidenceSelectionBudget(GroundingContractMixin):
    max_items: int = 24
    max_characters: int = 12000
    selected_items: int = 0
    estimated_characters: int = 0
    truncated: bool = False

    def __post_init__(self) -> None:
        max_items = max(1, int(self.max_items))
        max_characters = max(256, int(self.max_characters))
        selected = max(0, int(self.selected_items))
        estimated = max(0, int(self.estimated_characters))
        if selected > max_items:
            raise ValueError("Evidence selection exceeds max_items.")
        object.__setattr__(self, "max_items", max_items)
        object.__setattr__(self, "max_characters", max_characters)
        object.__setattr__(self, "selected_items", selected)
        object.__setattr__(self, "estimated_characters", estimated)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceSelectionBudget":
        return cls(
            max_items=int(data.get("max_items", 24)),
            max_characters=int(data.get("max_characters", 12000)),
            selected_items=int(data.get("selected_items", 0)),
            estimated_characters=int(data.get("estimated_characters", 0)),
            truncated=bool(data.get("truncated", False)),
        )


@dataclass(frozen=True)
class EvidenceBundle(GroundingContractMixin):
    id: str
    context_revision: str
    interaction: InteractionFrame
    items: tuple[EvidenceItem, ...] = ()
    knowledge_gaps: tuple[KnowledgeGap, ...] = ()
    capabilities: tuple[CapabilityEvidence, ...] = ()
    coverage: tuple[EvidenceCoverage, ...] = ()
    selection_budget: EvidenceSelectionBudget = field(
        default_factory=EvidenceSelectionBudget
    )
    created_at: str = field(default_factory=grounding_utc_now)
    extensions: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        interaction = self.interaction
        if isinstance(interaction, dict):
            interaction = InteractionFrame.from_dict(interaction)
        budget = self.selection_budget
        if isinstance(budget, dict):
            budget = EvidenceSelectionBudget.from_dict(budget)
        object.__setattr__(self, "interaction", interaction)
        object.__setattr__(
            self,
            "items",
            tuple(
                EvidenceItem.from_dict(item) if isinstance(item, dict) else item
                for item in self.items
            ),
        )
        object.__setattr__(
            self,
            "knowledge_gaps",
            tuple(
                KnowledgeGap.from_dict(item) if isinstance(item, dict) else item
                for item in self.knowledge_gaps
            ),
        )
        object.__setattr__(
            self,
            "capabilities",
            tuple(
                CapabilityEvidence.from_dict(item)
                if isinstance(item, dict)
                else item
                for item in self.capabilities
            ),
        )
        object.__setattr__(
            self,
            "coverage",
            tuple(
                EvidenceCoverage.from_dict(item) if isinstance(item, dict) else item
                for item in self.coverage
            ),
        )
        object.__setattr__(self, "selection_budget", budget)
        object.__setattr__(self, "extensions", _metadata(self.extensions))
        identifiers = [
            item.id
            for collection in (
                self.items,
                self.knowledge_gaps,
                self.capabilities,
            )
            for item in collection
        ]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("EvidenceBundle identifiers must be unique.")
        if budget.selected_items != len(self.items):
            raise ValueError("EvidenceSelectionBudget selected_items is inconsistent.")

    @property
    def reference_ids(self) -> frozenset[str]:
        return frozenset(
            item.id
            for collection in (
                self.items,
                self.knowledge_gaps,
                self.capabilities,
            )
            for item in collection
        )

    @property
    def authorized_evidence_ids(self) -> frozenset[str]:
        return frozenset(item.id for item in self.items if item.authorized)

    def contains_reference(self, reference_id: str) -> bool:
        return reference_id in self.reference_ids

    def to_model_dict(self) -> dict[str, JsonValue]:
        """Return the compact, lossless-enough projection used for reasoning.

        Runtime bookkeeping such as coverage counters and serialization budgets
        remains available on the complete contract, but does not compete with
        the selected evidence for the model's attention.
        """

        projected = [
            {
                "id": item.id,
                "source": item.source,
                "subject": item.subject,
                "predicate": item.predicate,
                "value": item.value,
                "summary": item.summary,
                "confidence": item.confidence,
                "role": item.attributes.get("evidence_role", "fact"),
                "epistemic_use": item.attributes.get(
                    "epistemic_use",
                    "factual_support",
                ),
                "privacy_scope": item.privacy_scope,
            }
            for item in self.items
        ]
        requests = [item for item in projected if item["role"] == "request"]
        background_roles = {"background", "state", "event"}
        background = [
            item for item in projected if item["role"] in background_roles
        ]
        facts = [
            item
            for item in projected
            if item["role"] != "request"
            and item["role"] not in background_roles
        ]
        return {
            "id": self.id,
            "context_revision": self.context_revision,
            "interaction": {
                "id": self.interaction.id,
                "kind": self.interaction.kind,
                "primary_observation_id": self.interaction.primary_observation_id,
                "primary_evidence_id": (
                    f"evidence:observation:"
                    f"{self.interaction.primary_observation_id}"
                    if self.interaction.primary_observation_id
                    else ""
                ),
                "speaker_ref": self.interaction.speaker_ref,
                "channel": self.interaction.channel,
                "response_expected": self.interaction.response_expected,
                "silence_allowed": self.interaction.silence_allowed,
                "constraints": list(self.interaction.constraints),
                "metadata": self.interaction.metadata,
            },
            "requests": requests,
            "facts": facts,
            "knowledge_gaps": [item.to_dict() for item in self.knowledge_gaps],
            "background": background,
            "capabilities": [item.to_dict() for item in self.capabilities],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceBundle":
        return cls(
            id=str(data["id"]),
            context_revision=str(data.get("context_revision", "")),
            interaction=InteractionFrame.from_dict(data.get("interaction", {})),
            items=tuple(EvidenceItem.from_dict(item) for item in data.get("items", [])),
            knowledge_gaps=tuple(
                KnowledgeGap.from_dict(item)
                for item in data.get("knowledge_gaps", [])
            ),
            capabilities=tuple(
                CapabilityEvidence.from_dict(item)
                for item in data.get("capabilities", [])
            ),
            coverage=tuple(
                EvidenceCoverage.from_dict(item)
                for item in data.get("coverage", [])
            ),
            selection_budget=EvidenceSelectionBudget.from_dict(
                data.get("selection_budget", {})
            ),
            created_at=str(data.get("created_at") or grounding_utc_now()),
            extensions=_metadata(data.get("extensions")),
        )


@dataclass(frozen=True)
class AttentionCommitment(GroundingContractMixin):
    target_ids: tuple[str, ...] = ()
    topic: str = ""
    focus: str = ""
    priority: float = 0.5
    ignored_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_ids", tuple(self.target_ids))
        object.__setattr__(self, "ignored_ids", tuple(self.ignored_ids))
        object.__setattr__(self, "priority", clamp01(self.priority))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AttentionCommitment":
        return cls(
            target_ids=tuple(str(item) for item in data.get("target_ids", [])),
            topic=str(data.get("topic", "")),
            focus=str(data.get("focus", "")),
            priority=float(data.get("priority", 0.5)),
            ignored_ids=tuple(str(item) for item in data.get("ignored_ids", [])),
        )


@dataclass(frozen=True)
class KnowledgeCommitment(GroundingContractMixin):
    subject: str = "unspecified"
    source: str = "none"
    epistemic_status: EpistemicStatus = "not_applicable"
    evidence_ids: tuple[str, ...] = ()
    knowledge_gap_ids: tuple[str, ...] = ()
    boundaries: tuple[str, ...] = ()
    confidence: float = 0.5

    def __post_init__(self) -> None:
        if self.epistemic_status not in {
            "known",
            "unknown",
            "uncertain",
            "unavailable",
            "not_applicable",
        }:
            raise ValueError(
                f"Unsupported epistemic status: {self.epistemic_status}"
            )
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        object.__setattr__(
            self,
            "knowledge_gap_ids",
            tuple(self.knowledge_gap_ids),
        )
        object.__setattr__(self, "boundaries", tuple(self.boundaries))
        object.__setattr__(self, "confidence", clamp01(self.confidence))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnowledgeCommitment":
        return cls(
            subject=str(data.get("subject", "unspecified")),
            source=str(data.get("source", "none")),
            epistemic_status=data.get("epistemic_status", "not_applicable"),
            evidence_ids=tuple(str(item) for item in data.get("evidence_ids", [])),
            knowledge_gap_ids=tuple(
                str(item) for item in data.get("knowledge_gap_ids", [])
            ),
            boundaries=tuple(str(item) for item in data.get("boundaries", [])),
            confidence=float(data.get("confidence", 0.5)),
        )


@dataclass(frozen=True)
class GroundedClaim(GroundingContractMixin):
    id: str
    subject: str
    predicate: str
    content: str
    value: JsonValue = None
    role: ClaimRole = "answer"
    evidence_ids: tuple[str, ...] = ()
    knowledge_gap_ids: tuple[str, ...] = ()
    certainty: float = 0.5
    required: bool = True

    def __post_init__(self) -> None:
        if self.role not in {"answer", "support", "context", "caveat"}:
            raise ValueError(f"Unsupported grounded claim role: {self.role}")
        object.__setattr__(self, "value", _json_ready(self.value))
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        object.__setattr__(
            self,
            "knowledge_gap_ids",
            tuple(self.knowledge_gap_ids),
        )
        object.__setattr__(self, "certainty", clamp01(self.certainty))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GroundedClaim":
        return cls(
            id=str(data["id"]),
            subject=str(data.get("subject", "")),
            predicate=str(data.get("predicate", "")),
            content=str(data.get("content", "")),
            value=_json_ready(data.get("value")),
            role=data.get("role", "answer"),
            evidence_ids=tuple(str(item) for item in data.get("evidence_ids", [])),
            knowledge_gap_ids=tuple(
                str(item) for item in data.get("knowledge_gap_ids", [])
            ),
            certainty=float(data.get("certainty", 0.5)),
            required=bool(data.get("required", True)),
        )


@dataclass(frozen=True)
class EngagementCommitment(GroundingContractMixin):
    mode: EngagementMode = "speak"
    should_speak: bool = True
    initiative: bool = False
    reason_code: str = ""
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in {"speak", "listen", "silence"}:
            raise ValueError(f"Unsupported engagement mode: {self.mode}")
        if self.mode == "silence" and self.should_speak:
            raise ValueError("Silence engagement cannot require speech.")
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EngagementCommitment":
        return cls(
            mode=data.get("mode", "speak"),
            should_speak=bool(data.get("should_speak", True)),
            initiative=bool(data.get("initiative", False)),
            reason_code=str(data.get("reason_code", "")),
            evidence_ids=tuple(str(item) for item in data.get("evidence_ids", [])),
        )


@dataclass(frozen=True)
class ActionCommitment(GroundingContractMixin):
    phase: ActionPhase = "none"
    capability_id: str = ""
    action_type: str = ""
    objective: str = ""
    evidence_ids: tuple[str, ...] = ()
    expected_observation: str = ""
    requires_followup: bool = False

    def __post_init__(self) -> None:
        if self.phase not in {
            "none",
            "request",
            "awaiting_result",
            "answer_from_result",
        }:
            raise ValueError(f"Unsupported action phase: {self.phase}")
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionCommitment":
        return cls(
            phase=data.get("phase", "none"),
            capability_id=str(data.get("capability_id", "")),
            action_type=str(data.get("action_type", "")),
            objective=str(data.get("objective", "")),
            evidence_ids=tuple(str(item) for item in data.get("evidence_ids", [])),
            expected_observation=str(data.get("expected_observation", "")),
            requires_followup=bool(data.get("requires_followup", False)),
        )


@dataclass(frozen=True)
class ExpressionCommitment(GroundingContractMixin):
    stance: str = "calm"
    warmth: float = 0.7
    energy: float = 0.4
    formality: float = 0.25
    closeness: float = 0.6
    intensity: float = 0.35
    attention_target: str = "user"
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("warmth", "energy", "formality", "closeness", "intensity"):
            object.__setattr__(self, name, clamp01(getattr(self, name)))
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExpressionCommitment":
        return cls(
            stance=str(data.get("stance", "calm")),
            warmth=float(data.get("warmth", 0.7)),
            energy=float(data.get("energy", 0.4)),
            formality=float(data.get("formality", 0.25)),
            closeness=float(data.get("closeness", 0.6)),
            intensity=float(data.get("intensity", 0.35)),
            attention_target=str(data.get("attention_target", "user")),
            evidence_ids=tuple(str(item) for item in data.get("evidence_ids", [])),
        )


@dataclass(frozen=True)
class DeferredIntent(GroundingContractMixin):
    id: str
    kind: str
    target: str = ""
    reason: str = ""
    evidence_ids: tuple[str, ...] = ()
    priority: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        object.__setattr__(self, "priority", clamp01(self.priority))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeferredIntent":
        return cls(
            id=str(data["id"]),
            kind=str(data.get("kind", "")),
            target=str(data.get("target", "")),
            reason=str(data.get("reason", "")),
            evidence_ids=tuple(str(item) for item in data.get("evidence_ids", [])),
            priority=float(data.get("priority", 0.5)),
        )


@dataclass(frozen=True)
class OutputManifest(GroundingContractMixin):
    conversation_required: bool = True
    speech_required: bool = True
    body_required: bool = True
    action_ids: tuple[str, ...] = ()
    memory_proposal_ids: tuple[str, ...] = ()
    relationship_proposal_ids: tuple[str, ...] = ()
    self_proposal_ids: tuple[str, ...] = ()
    curiosity_proposal_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "action_ids",
            "memory_proposal_ids",
            "relationship_proposal_ids",
            "self_proposal_ids",
            "curiosity_proposal_ids",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OutputManifest":
        return cls(
            conversation_required=bool(data.get("conversation_required", True)),
            speech_required=bool(data.get("speech_required", True)),
            body_required=bool(data.get("body_required", True)),
            action_ids=tuple(str(item) for item in data.get("action_ids", [])),
            memory_proposal_ids=tuple(
                str(item) for item in data.get("memory_proposal_ids", [])
            ),
            relationship_proposal_ids=tuple(
                str(item) for item in data.get("relationship_proposal_ids", [])
            ),
            self_proposal_ids=tuple(
                str(item) for item in data.get("self_proposal_ids", [])
            ),
            curiosity_proposal_ids=tuple(
                str(item) for item in data.get("curiosity_proposal_ids", [])
            ),
        )


@dataclass(frozen=True)
class GroundedResponsePlan(GroundingContractMixin):
    id: str
    evidence_bundle_id: str
    primary_obligation: PrimaryObligation
    response_goal: str
    attention: AttentionCommitment = field(default_factory=AttentionCommitment)
    knowledge: KnowledgeCommitment = field(default_factory=KnowledgeCommitment)
    claims: tuple[GroundedClaim, ...] = ()
    engagement: EngagementCommitment = field(default_factory=EngagementCommitment)
    action: ActionCommitment = field(default_factory=ActionCommitment)
    expression: ExpressionCommitment = field(default_factory=ExpressionCommitment)
    deferred_intents: tuple[DeferredIntent, ...] = ()
    output_manifest: OutputManifest = field(default_factory=OutputManifest)
    confidence: float = 0.5
    created_at: str = field(default_factory=grounding_utc_now)
    extensions: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.primary_obligation not in {
            "answer",
            "acknowledge",
            "clarify",
            "act",
            "observe",
            "listen",
            "initiate",
            "recover",
        }:
            raise ValueError(
                f"Unsupported primary obligation: {self.primary_obligation}"
            )
        conversions = (
            ("attention", AttentionCommitment),
            ("knowledge", KnowledgeCommitment),
            ("engagement", EngagementCommitment),
            ("action", ActionCommitment),
            ("expression", ExpressionCommitment),
            ("output_manifest", OutputManifest),
        )
        for name, contract in conversions:
            value = getattr(self, name)
            if isinstance(value, dict):
                object.__setattr__(self, name, contract.from_dict(value))
        object.__setattr__(
            self,
            "claims",
            tuple(
                GroundedClaim.from_dict(item) if isinstance(item, dict) else item
                for item in self.claims
            ),
        )
        object.__setattr__(
            self,
            "deferred_intents",
            tuple(
                DeferredIntent.from_dict(item) if isinstance(item, dict) else item
                for item in self.deferred_intents
            ),
        )
        object.__setattr__(self, "confidence", clamp01(self.confidence))
        object.__setattr__(self, "extensions", _metadata(self.extensions))
        claim_ids = [item.id for item in self.claims]
        deferred_ids = [item.id for item in self.deferred_intents]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("GroundedResponsePlan claim identifiers must be unique.")
        if len(deferred_ids) != len(set(deferred_ids)):
            raise ValueError("Deferred intent identifiers must be unique.")

    @property
    def referenced_evidence_ids(self) -> frozenset[str]:
        references = set(self.attention.target_ids)
        references.update(self.knowledge.evidence_ids)
        references.update(self.knowledge.knowledge_gap_ids)
        references.update(self.engagement.evidence_ids)
        references.update(self.action.evidence_ids)
        references.update(self.expression.evidence_ids)
        for claim in self.claims:
            references.update(claim.evidence_ids)
            references.update(claim.knowledge_gap_ids)
        for item in self.deferred_intents:
            references.update(item.evidence_ids)
        return frozenset(references)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GroundedResponsePlan":
        return cls(
            id=str(data["id"]),
            evidence_bundle_id=str(data.get("evidence_bundle_id", "")),
            primary_obligation=data.get("primary_obligation", "answer"),
            response_goal=str(data.get("response_goal", "")),
            attention=AttentionCommitment.from_dict(data.get("attention", {})),
            knowledge=KnowledgeCommitment.from_dict(data.get("knowledge", {})),
            claims=tuple(
                GroundedClaim.from_dict(item) for item in data.get("claims", [])
            ),
            engagement=EngagementCommitment.from_dict(
                data.get("engagement", {})
            ),
            action=ActionCommitment.from_dict(data.get("action", {})),
            expression=ExpressionCommitment.from_dict(
                data.get("expression", {})
            ),
            deferred_intents=tuple(
                DeferredIntent.from_dict(item)
                for item in data.get("deferred_intents", [])
            ),
            output_manifest=OutputManifest.from_dict(
                data.get("output_manifest", {})
            ),
            confidence=float(data.get("confidence", 0.5)),
            created_at=str(data.get("created_at") or grounding_utc_now()),
            extensions=_metadata(data.get("extensions")),
        )


@dataclass(frozen=True)
class BodyIntention(GroundingContractMixin):
    plan_id: str
    expression: str = "neutral"
    gesture: str = "idle"
    gaze: str = "user"
    posture: str = "relaxed"
    attention: str = "user"
    intensity: float = 0.35
    movement_energy: float = 0.3
    microexpressions: tuple[str, ...] = ()
    lip_sync_hint: str = "speech"
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "intensity", clamp01(self.intensity))
        object.__setattr__(self, "movement_energy", clamp01(self.movement_energy))
        object.__setattr__(self, "microexpressions", tuple(self.microexpressions))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_avatar_payload(self) -> dict[str, JsonValue]:
        return {
            "expression": self.expression,
            "gesture": self.gesture,
            "gaze": self.gaze,
            "posture": self.posture,
            "attention": self.attention,
            "expression_intensity": self.intensity,
            "movement_energy": self.movement_energy,
            "microexpressions": list(self.microexpressions),
            "lip_sync_hint": self.lip_sync_hint,
            "plan_id": self.plan_id,
            **self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BodyIntention":
        return cls(
            plan_id=str(data.get("plan_id", "")),
            expression=str(data.get("expression", "neutral")),
            gesture=str(data.get("gesture", "idle")),
            gaze=str(data.get("gaze", "user")),
            posture=str(data.get("posture", "relaxed")),
            attention=str(data.get("attention", "user")),
            intensity=float(data.get("intensity", 0.35)),
            movement_energy=float(data.get("movement_energy", 0.3)),
            microexpressions=tuple(
                str(item) for item in data.get("microexpressions", [])
            ),
            lip_sync_hint=str(data.get("lip_sync_hint", "speech")),
            metadata=_metadata(data.get("metadata")),
        )
