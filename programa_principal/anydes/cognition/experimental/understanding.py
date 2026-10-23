from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from anydes.cognition.grounding_contracts import EvidenceBundle
from anydes.cognition.provenance_resolver import (
    ProvenanceResolution,
    resolve_grounded_wire_provenance,
)
from anydes.cognition.reasoning import extract_reasoning_model_object
from anydes.primitives import JsonValue, clamp01


ActKind = Literal[
    "assertion",
    "question",
    "request",
    "instruction",
    "correction",
    "preference",
    "hypothesis",
    "emotion",
    "clarification",
    "commitment",
    "cancellation",
    "acknowledgement",
]
SubjectRef = str
EpistemicMode = Literal[
    "user_stated",
    "user_asked",
    "user_requested",
    "user_hypothesized",
    "user_corrected",
    "user_expressed",
    "inferred_from_user",
]
ObligationKind = Literal[
    "answer_question",
    "acknowledge",
    "acknowledge_correction",
    "preserve_preference",
    "fulfill_request",
    "clarify_ambiguity",
    "emotional_support",
    "consider_tool",
    "listen",
    "allow_silence",
    "continue_topic",
    "record_commitment",
]


_ACT_KINDS = frozenset(ActKind.__args__)
_EPISTEMIC_MODES = frozenset(EpistemicMode.__args__)
_OBLIGATION_KINDS = frozenset(ObligationKind.__args__)
_FORBIDDEN_WIRE_KEYS = frozenset(
    {
        "action",
        "actions",
        "answer",
        "body",
        "body_intention",
        "final_response",
        "personality",
        "promise",
        "promises",
        "response",
        "self_update",
        "speech",
        "speech_intention",
        "style",
        "tool_decision",
    }
)


class UnderstandingModel(Protocol):
    def generate_for_bundle(self, bundle: EvidenceBundle, prompt: str) -> str:
        """Return one UnderstandingFrame wire object."""


class UnderstandingContractError(ValueError):
    """The model output cannot represent understanding without guessing."""


@dataclass(frozen=True)
class CommunicativeAct:
    key: str
    kind: str
    actor_ref: str
    subject_ref: str
    content_summary: str
    epistemic_mode: str
    evidence_ids: tuple[str, ...]
    modifies_refs: tuple[str, ...] = ()
    confidence: float = 0.5
    inferred: bool = False
    uncertainty: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        object.__setattr__(self, "modifies_refs", tuple(self.modifies_refs))
        object.__setattr__(self, "confidence", clamp01(self.confidence))
        object.__setattr__(self, "uncertainty", clamp01(self.uncertainty))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "key": self.key,
            "kind": self.kind,
            "actor_ref": self.actor_ref,
            "subject_ref": self.subject_ref,
            "content_summary": self.content_summary,
            "epistemic_mode": self.epistemic_mode,
            "evidence_ids": list(self.evidence_ids),
            "modifies_refs": list(self.modifies_refs),
            "confidence": self.confidence,
            "inferred": self.inferred,
            "uncertainty": self.uncertainty,
        }


@dataclass(frozen=True)
class EntityReference:
    key: str
    mention: str
    entity_type: str
    resolved_ref: str
    evidence_ids: tuple[str, ...]
    confidence: float = 0.5
    ambiguous: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        object.__setattr__(self, "confidence", clamp01(self.confidence))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "key": self.key,
            "mention": self.mention,
            "entity_type": self.entity_type,
            "resolved_ref": self.resolved_ref,
            "evidence_ids": list(self.evidence_ids),
            "confidence": self.confidence,
            "ambiguous": self.ambiguous,
        }


@dataclass(frozen=True)
class TurnRelation:
    key: str
    relation_type: str
    current_act_keys: tuple[str, ...]
    prior_evidence_ids: tuple[str, ...]
    confidence: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "current_act_keys", tuple(self.current_act_keys))
        object.__setattr__(
            self,
            "prior_evidence_ids",
            tuple(self.prior_evidence_ids),
        )
        object.__setattr__(self, "confidence", clamp01(self.confidence))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "key": self.key,
            "relation_type": self.relation_type,
            "current_act_keys": list(self.current_act_keys),
            "prior_evidence_ids": list(self.prior_evidence_ids),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class UnknownInformation:
    key: str
    subject_ref: str
    description: str
    evidence_ids: tuple[str, ...]
    knowledge_gap_ids: tuple[str, ...] = ()
    confidence: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        object.__setattr__(
            self,
            "knowledge_gap_ids",
            tuple(self.knowledge_gap_ids),
        )
        object.__setattr__(self, "confidence", clamp01(self.confidence))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "key": self.key,
            "subject_ref": self.subject_ref,
            "description": self.description,
            "evidence_ids": list(self.evidence_ids),
            "knowledge_gap_ids": list(self.knowledge_gap_ids),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class CandidateObligation:
    key: str
    kind: str
    target: str
    priority: float
    confidence: float
    evidence_ids: tuple[str, ...]
    depends_on: tuple[str, ...] = ()
    conflicts_with: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "priority", clamp01(self.priority))
        object.__setattr__(self, "confidence", clamp01(self.confidence))
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        object.__setattr__(self, "depends_on", tuple(self.depends_on))
        object.__setattr__(self, "conflicts_with", tuple(self.conflicts_with))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "key": self.key,
            "kind": self.kind,
            "target": self.target,
            "priority": self.priority,
            "confidence": self.confidence,
            "evidence_ids": list(self.evidence_ids),
            "depends_on": list(self.depends_on),
            "conflicts_with": list(self.conflicts_with),
        }


@dataclass(frozen=True)
class UnderstandingAmbiguity:
    key: str
    description: str
    affected_keys: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    confidence: float = 0.5
    requires_clarification: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "affected_keys", tuple(self.affected_keys))
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        object.__setattr__(self, "confidence", clamp01(self.confidence))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "key": self.key,
            "description": self.description,
            "affected_keys": list(self.affected_keys),
            "evidence_ids": list(self.evidence_ids),
            "confidence": self.confidence,
            "requires_clarification": self.requires_clarification,
        }


@dataclass(frozen=True)
class UnderstandingFrame:
    id: str
    evidence_bundle_id: str
    primary_evidence_id: str
    acts: tuple[CommunicativeAct, ...]
    entities: tuple[EntityReference, ...]
    relations: tuple[TurnRelation, ...]
    new_information_act_keys: tuple[str, ...]
    unknowns: tuple[UnknownInformation, ...]
    candidate_obligations: tuple[CandidateObligation, ...]
    ambiguities: tuple[UnderstandingAmbiguity, ...]
    provenance: tuple[ProvenanceResolution, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "acts",
            "entities",
            "relations",
            "new_information_act_keys",
            "unknowns",
            "candidate_obligations",
            "ambiguities",
            "provenance",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "id": self.id,
            "evidence_bundle_id": self.evidence_bundle_id,
            "primary_evidence_id": self.primary_evidence_id,
            "acts": [item.to_dict() for item in self.acts],
            "entities": [item.to_dict() for item in self.entities],
            "relations": [item.to_dict() for item in self.relations],
            "new_information_act_keys": list(self.new_information_act_keys),
            "unknowns": [item.to_dict() for item in self.unknowns],
            "candidate_obligations": [
                item.to_dict() for item in self.candidate_obligations
            ],
            "ambiguities": [item.to_dict() for item in self.ambiguities],
            "provenance": [item.to_dict() for item in self.provenance],
        }


@dataclass(frozen=True)
class UnderstandingValidation:
    valid: bool
    issues: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, JsonValue]:
        return {"valid": self.valid, "issues": list(self.issues)}


@dataclass
class UnderstandingEngine:
    model: UnderstandingModel

    def understand(self, bundle: EvidenceBundle) -> UnderstandingFrame:
        raw = self.model.generate_for_bundle(
            bundle,
            build_understanding_prompt(bundle),
        )
        payload = extract_reasoning_model_object(raw)
        return understanding_frame_from_model_dict(payload, bundle)


class UnderstandingFrameValidator:
    """Validate representation only; never select or repair an obligation."""

    def validate(
        self,
        frame: UnderstandingFrame,
        bundle: EvidenceBundle,
    ) -> UnderstandingValidation:
        issues: list[str] = []
        if frame.evidence_bundle_id != bundle.id:
            issues.append("evidence_bundle_mismatch")
        if frame.primary_evidence_id != _primary_evidence_id(bundle):
            issues.append("primary_evidence_mismatch")
        for record in frame.provenance:
            if not record.succeeded:
                issues.append(
                    f"provenance:{record.field_path}:{record.failure_reason}"
                )

        item_by_id = {item.id: item for item in bundle.items}
        gap_ids = {item.id for item in bundle.knowledge_gaps}
        known_refs = bundle.reference_ids
        unauthorized = {item.id for item in bundle.items if not item.authorized}
        act_keys = _unique_keys("act", frame.acts, issues)
        entity_keys = _unique_keys("entity", frame.entities, issues)
        relation_keys = _unique_keys("relation", frame.relations, issues)
        unknown_keys = _unique_keys("unknown", frame.unknowns, issues)
        obligation_keys = _unique_keys(
            "obligation",
            frame.candidate_obligations,
            issues,
        )
        ambiguity_keys = _unique_keys("ambiguity", frame.ambiguities, issues)
        local_keys = (
            act_keys
            | entity_keys
            | relation_keys
            | unknown_keys
            | obligation_keys
            | ambiguity_keys
        )
        primary_id = frame.primary_evidence_id

        for act in frame.acts:
            self._validate_refs(
                f"act:{act.key}",
                act.evidence_ids,
                known_refs,
                unauthorized,
                issues,
            )
            if primary_id and primary_id not in act.evidence_ids:
                issues.append(f"act_missing_primary_evidence:{act.key}")
            if act.kind not in _ACT_KINDS:
                issues.append(f"unsupported_act_kind:{act.key}:{act.kind}")
            if act.epistemic_mode not in _EPISTEMIC_MODES:
                issues.append(
                    f"unsupported_epistemic_mode:{act.key}:"
                    f"{act.epistemic_mode}"
                )
            if not _is_user_actor(act.actor_ref, bundle):
                issues.append(f"user_self_confusion:actor:{act.key}")
            if not _mode_matches_act(act.kind, act.epistemic_mode):
                issues.append(f"epistemic_mode_mismatch:{act.key}")
            if (
                act.kind == "preference"
                and _is_self_ref(act.subject_ref, bundle)
            ):
                issues.append(f"user_self_confusion:preference:{act.key}")
            if act.kind == "correction":
                if not act.modifies_refs:
                    issues.append(f"correction_target_missing:{act.key}")
                for reference in act.modifies_refs:
                    if reference not in local_keys and reference not in known_refs:
                        issues.append(
                            f"correction_target_unknown:{act.key}:{reference}"
                        )
            if act.inferred and (
                act.uncertainty <= 0.0 or act.confidence >= 1.0
            ):
                issues.append(f"inference_uncertainty_lost:{act.key}")
            if act.kind == "assertion" and _only_request_evidence(
                act.evidence_ids,
                item_by_id,
            ):
                issues.append(f"request_treated_as_fact:{act.key}")

        if (
            bundle.interaction.metadata.get("utterance_form") == "question"
            and not any(item.kind == "question" for item in frame.acts)
        ):
            issues.append("question_not_represented_as_question")

        for entity in frame.entities:
            self._validate_refs(
                f"entity:{entity.key}",
                entity.evidence_ids,
                known_refs,
                unauthorized,
                issues,
            )
            if not entity.evidence_ids:
                issues.append(f"entity_without_evidence:{entity.key}")
        for relation in frame.relations:
            self._validate_refs(
                f"relation:{relation.key}",
                relation.prior_evidence_ids,
                known_refs,
                unauthorized,
                issues,
            )
            if primary_id and primary_id in relation.prior_evidence_ids:
                issues.append(
                    f"relation_prior_is_current:{relation.key}"
                )
            for key in relation.current_act_keys:
                if key not in act_keys:
                    issues.append(
                        f"relation_act_unknown:{relation.key}:{key}"
                    )
        for key in frame.new_information_act_keys:
            if key not in act_keys:
                issues.append(f"new_information_act_unknown:{key}")
        for unknown in frame.unknowns:
            self._validate_refs(
                f"unknown:{unknown.key}",
                (*unknown.evidence_ids, *unknown.knowledge_gap_ids),
                known_refs,
                unauthorized,
                issues,
            )
            if not unknown.evidence_ids:
                issues.append(f"unknown_without_evidence:{unknown.key}")
            for reference in unknown.knowledge_gap_ids:
                if reference not in gap_ids:
                    issues.append(
                        f"knowledge_gap_reference_not_gap:"
                        f"{unknown.key}:{reference}"
                    )
        priorities = [item.priority for item in frame.candidate_obligations]
        if priorities != sorted(priorities, reverse=True):
            issues.append("candidate_priorities_not_ordered")
        for candidate in frame.candidate_obligations:
            self._validate_refs(
                f"obligation:{candidate.key}",
                candidate.evidence_ids,
                known_refs,
                unauthorized,
                issues,
            )
            if not candidate.evidence_ids:
                issues.append(
                    f"candidate_without_evidence:{candidate.key}"
                )
            elif primary_id and primary_id not in candidate.evidence_ids:
                issues.append(
                    f"candidate_missing_primary_evidence:{candidate.key}"
                )
            if candidate.kind not in _OBLIGATION_KINDS:
                issues.append(
                    f"unsupported_obligation_kind:{candidate.key}:"
                    f"{candidate.kind}"
                )
            for key in candidate.depends_on:
                if key not in local_keys:
                    issues.append(
                        f"candidate_dependency_unknown:{candidate.key}:{key}"
                    )
            for key in candidate.conflicts_with:
                if key not in obligation_keys:
                    issues.append(
                        f"candidate_conflict_unknown:{candidate.key}:{key}"
                    )
            if candidate.key in candidate.depends_on:
                issues.append(f"candidate_self_dependency:{candidate.key}")
            if candidate.key in candidate.conflicts_with:
                issues.append(f"candidate_self_conflict:{candidate.key}")
        for ambiguity in frame.ambiguities:
            self._validate_refs(
                f"ambiguity:{ambiguity.key}",
                ambiguity.evidence_ids,
                known_refs,
                unauthorized,
                issues,
            )
            for key in ambiguity.affected_keys:
                if key not in local_keys:
                    issues.append(
                        f"ambiguity_target_unknown:{ambiguity.key}:{key}"
                    )

        return UnderstandingValidation(
            valid=not issues,
            issues=tuple(dict.fromkeys(issues)),
        )

    @staticmethod
    def _validate_refs(
        owner: str,
        references: tuple[str, ...],
        known_refs: frozenset[str],
        unauthorized: set[str],
        issues: list[str],
    ) -> None:
        for reference in references:
            if reference not in known_refs:
                issues.append(f"evidence_reference_unknown:{owner}:{reference}")
            elif reference in unauthorized:
                issues.append(
                    f"evidence_reference_unauthorized:{owner}:{reference}"
                )


def build_understanding_prompt(bundle: EvidenceBundle) -> str:
    evidence_json = json.dumps(
        bundle.to_model_dict(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "\n".join(
        [
            "Analiza exclusivamente lo que comunica el usuario en este turno.",
            "No decidas que hara Anydes y no redactes ninguna respuesta.",
            "Devuelve un unico UnderstandingFrame JSON, sin razonamiento libre.",
            "Distingue actos: assertion|question|request|instruction|correction|preference|hypothesis|emotion|clarification|commitment|cancellation|acknowledgement.",
            "actor_ref identifica al usuario emisor; subject_ref es una referencia concreta a la persona, entidad o tema del acto.",
            "epistemic_mode: user_stated|user_asked|user_requested|user_hypothesized|user_corrected|user_expressed|inferred_from_user.",
            "Una pregunta o peticion prueba que el usuario la formulo, no que su contenido sea un hecho.",
            "Una emocion inferida debe conservar incertidumbre y confianza menor que uno.",
            "Una correccion debe enlazar mediante modifies_refs aquello que modifica.",
            "Representa referencias a turnos anteriores solo con evidencia autorizada.",
            "Produce entre una y cuatro obligaciones candidatas, ordenadas por prioridad; no elijas una decision final.",
            "Obligaciones: answer_question|acknowledge|acknowledge_correction|preserve_preference|fulfill_request|clarify_ambiguity|emotional_support|consider_tool|listen|allow_silence|continue_topic|record_commitment.",
            "No incluyas Speech, Body, acciones, herramientas decididas, personalidad, estilo, promesas, SELF updates ni respuesta final.",
            "Usa exclusivamente IDs presentes en el EvidenceBundle.",
            "EVIDENCE BUNDLE:",
            evidence_json,
            "Completa este esquema. Los valores vacios son marcadores, nunca contenido:",
            '{"acts":[{"key":"","kind":"","actor_ref":"user","subject_ref":"","content_summary":"","epistemic_mode":"","evidence_ids":[],"modifies_refs":[],"confidence":0.0,"inferred":false,"uncertainty":0.0}],'
            '"entities":[{"key":"","mention":"","entity_type":"","resolved_ref":"","evidence_ids":[],"confidence":0.0,"ambiguous":false}],'
            '"relations":[{"key":"","relation_type":"","current_act_keys":[],"prior_evidence_ids":[],"confidence":0.0}],'
            '"new_information_act_keys":[],"unknowns":[{"key":"","subject_ref":"","description":"","evidence_ids":[],"knowledge_gap_ids":[],"confidence":0.0}],'
            '"candidate_obligations":[{"key":"","kind":"","target":"","priority":0.0,"confidence":0.0,"evidence_ids":[],"depends_on":[],"conflicts_with":[]}],'
            '"ambiguities":[{"key":"","description":"","affected_keys":[],"evidence_ids":[],"confidence":0.0,"requires_clarification":false}]}',
            "Elimina de la salida los objetos opcionales que no existan; conserva las listas como listas.",
        ]
    )


def understanding_frame_from_model_dict(
    data: dict[str, object],
    bundle: EvidenceBundle,
) -> UnderstandingFrame:
    normalized = _strip_keys(data)
    nested = normalized.get("understanding_frame")
    if isinstance(nested, dict):
        normalized = _strip_keys(nested)
    forbidden = _find_forbidden_keys(normalized)
    if forbidden:
        raise UnderstandingContractError(
            "forbidden_wire_fields:" + ",".join(sorted(forbidden))
        )
    resolved = resolve_grounded_wire_provenance(normalized, bundle)
    payload = resolved.payload
    acts = _bounded_objects(payload.get("acts"), "acts", 6)
    entities = _bounded_objects(payload.get("entities"), "entities", 8)
    relations = _bounded_objects(payload.get("relations"), "relations", 6)
    unknowns = _bounded_objects(payload.get("unknowns"), "unknowns", 4)
    obligations = _bounded_objects(
        payload.get("candidate_obligations"),
        "candidate_obligations",
        4,
    )
    ambiguities = _bounded_objects(
        payload.get("ambiguities"),
        "ambiguities",
        4,
    )
    if not acts:
        raise UnderstandingContractError("communicative_acts_missing")
    if not obligations:
        raise UnderstandingContractError("candidate_obligations_missing")
    return UnderstandingFrame(
        id=f"understanding:{bundle.context_revision[:16]}",
        evidence_bundle_id=bundle.id,
        primary_evidence_id=_primary_evidence_id(bundle),
        acts=tuple(_act(item, index) for index, item in enumerate(acts, 1)),
        entities=tuple(
            _entity(item, index) for index, item in enumerate(entities, 1)
        ),
        relations=tuple(
            _relation(item, index) for index, item in enumerate(relations, 1)
        ),
        new_information_act_keys=_strings(
            payload.get("new_information_act_keys")
        ),
        unknowns=tuple(
            _unknown(item, index) for index, item in enumerate(unknowns, 1)
        ),
        candidate_obligations=tuple(
            _obligation(item, index)
            for index, item in enumerate(obligations, 1)
        ),
        ambiguities=tuple(
            _ambiguity(item, index)
            for index, item in enumerate(ambiguities, 1)
        ),
        provenance=resolved.records,
    )


def _act(data: dict[str, object], index: int) -> CommunicativeAct:
    return CommunicativeAct(
        key=str(data.get("key") or f"act:{index}"),
        kind=str(data.get("kind") or ""),
        actor_ref=str(data.get("actor_ref") or ""),
        subject_ref=str(data.get("subject_ref") or "unknown"),
        content_summary=str(data.get("content_summary") or ""),
        epistemic_mode=str(data.get("epistemic_mode") or ""),
        evidence_ids=_strings(data.get("evidence_ids")),
        modifies_refs=_strings(data.get("modifies_refs")),
        confidence=_float(data.get("confidence"), 0.5),
        inferred=_boolean(data.get("inferred"), False),
        uncertainty=_float(data.get("uncertainty"), 0.0),
    )


def _entity(data: dict[str, object], index: int) -> EntityReference:
    return EntityReference(
        key=str(data.get("key") or f"entity:{index}"),
        mention=str(data.get("mention") or ""),
        entity_type=str(data.get("entity_type") or "unknown"),
        resolved_ref=str(data.get("resolved_ref") or "unknown"),
        evidence_ids=_strings(data.get("evidence_ids")),
        confidence=_float(data.get("confidence"), 0.5),
        ambiguous=_boolean(data.get("ambiguous"), False),
    )


def _relation(data: dict[str, object], index: int) -> TurnRelation:
    return TurnRelation(
        key=str(data.get("key") or f"relation:{index}"),
        relation_type=str(data.get("relation_type") or "related"),
        current_act_keys=_strings(data.get("current_act_keys")),
        prior_evidence_ids=_strings(data.get("prior_evidence_ids")),
        confidence=_float(data.get("confidence"), 0.5),
    )


def _unknown(data: dict[str, object], index: int) -> UnknownInformation:
    return UnknownInformation(
        key=str(data.get("key") or f"unknown:{index}"),
        subject_ref=str(data.get("subject_ref") or "unknown"),
        description=str(data.get("description") or ""),
        evidence_ids=_strings(data.get("evidence_ids")),
        knowledge_gap_ids=_strings(data.get("knowledge_gap_ids")),
        confidence=_float(data.get("confidence"), 0.5),
    )


def _obligation(data: dict[str, object], index: int) -> CandidateObligation:
    return CandidateObligation(
        key=str(data.get("key") or f"obligation:{index}"),
        kind=str(data.get("kind") or ""),
        target=str(data.get("target") or ""),
        priority=_float(data.get("priority"), 0.0),
        confidence=_float(data.get("confidence"), 0.0),
        evidence_ids=_strings(data.get("evidence_ids")),
        depends_on=_strings(data.get("depends_on")),
        conflicts_with=_strings(data.get("conflicts_with")),
    )


def _ambiguity(data: dict[str, object], index: int) -> UnderstandingAmbiguity:
    return UnderstandingAmbiguity(
        key=str(data.get("key") or f"ambiguity:{index}"),
        description=str(data.get("description") or ""),
        affected_keys=_strings(data.get("affected_keys")),
        evidence_ids=_strings(data.get("evidence_ids")),
        confidence=_float(data.get("confidence"), 0.5),
        requires_clarification=_boolean(
            data.get("requires_clarification"),
            False,
        ),
    )


def _mode_matches_act(kind: str, mode: str) -> bool:
    expected = {
        "question": {"user_asked"},
        "request": {"user_requested"},
        "instruction": {"user_requested"},
        "hypothesis": {"user_hypothesized"},
        "correction": {"user_corrected"},
        "emotion": {"user_expressed", "inferred_from_user"},
        "assertion": {"user_stated", "user_expressed"},
        "preference": {"user_stated", "user_expressed"},
        "clarification": {"user_stated", "user_corrected"},
        "commitment": {"user_stated"},
        "cancellation": {"user_stated", "user_corrected", "user_requested"},
        "acknowledgement": {"user_stated", "user_expressed"},
    }
    return mode in expected.get(kind, set())


def _is_user_actor(value: str, bundle: EvidenceBundle) -> bool:
    normalized = value.strip().casefold()
    aliases = {
        "user",
        str(bundle.interaction.speaker_ref).strip().casefold(),
        str(bundle.interaction.metadata.get("first_person_ref") or "")
        .strip()
        .casefold(),
    }
    return normalized in aliases


def _is_self_ref(value: str, bundle: EvidenceBundle) -> bool:
    normalized = value.strip().casefold()
    aliases = {
        "self",
        str(bundle.interaction.metadata.get("second_person_ref") or "")
        .strip()
        .casefold(),
    }
    return bool(normalized) and normalized in aliases


def _only_request_evidence(
    evidence_ids: tuple[str, ...],
    item_by_id: dict[str, Any],
) -> bool:
    items = [item_by_id[item] for item in evidence_ids if item in item_by_id]
    return bool(items) and all(
        item.attributes.get("epistemic_use") == "interaction_request"
        for item in items
    )


def _unique_keys(
    owner: str,
    values: tuple[Any, ...],
    issues: list[str],
) -> set[str]:
    keys = [str(item.key) for item in values]
    if len(keys) != len(set(keys)):
        issues.append(f"duplicate_{owner}_key")
    if any(not key.strip() for key in keys):
        issues.append(f"empty_{owner}_key")
    return set(keys)


def _primary_evidence_id(bundle: EvidenceBundle) -> str:
    observation_id = bundle.interaction.primary_observation_id
    if not observation_id:
        return ""
    candidate = f"evidence:observation:{observation_id}"
    return candidate if bundle.contains_reference(candidate) else ""


def _find_forbidden_keys(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).strip().casefold()
            if normalized in _FORBIDDEN_WIRE_KEYS:
                found.add(normalized)
            found.update(_find_forbidden_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_find_forbidden_keys(item))
    return found


def _bounded_objects(
    value: object,
    name: str,
    maximum: int,
) -> tuple[dict[str, object], ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise UnderstandingContractError(f"{name}_must_be_list")
    if len(value) > maximum:
        raise UnderstandingContractError(f"{name}_limit_exceeded:{maximum}")
    if any(not isinstance(item, dict) for item in value):
        raise UnderstandingContractError(f"{name}_items_must_be_objects")
    return tuple(value)


def _strings(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if str(item).strip())
    return (str(value),) if str(value).strip() else ()


def _float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _boolean(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return default


def _strip_keys(value: dict[str, object]) -> dict[str, object]:
    return {str(key).strip(): item for key, item in value.items()}
