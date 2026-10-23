from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from anydes.cognition.grounding_contracts import EvidenceBundle


@dataclass(frozen=True)
class ProvenanceResolution:
    field_path: str
    original_value: str
    normalized_value: str
    rule: str
    evidence_ids: tuple[str, ...] = ()
    certainty: float = 1.0
    failure_reason: str = ""

    @property
    def succeeded(self) -> bool:
        return not self.failure_reason

    def to_dict(self) -> dict[str, object]:
        return {
            "field_path": self.field_path,
            "original_value": self.original_value,
            "normalized_value": self.normalized_value,
            "rule": self.rule,
            "evidence_ids": list(self.evidence_ids),
            "certainty": self.certainty,
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class ProvenanceResolutionResult:
    payload: dict[str, object]
    records: tuple[ProvenanceResolution, ...] = ()

    @property
    def failures(self) -> tuple[ProvenanceResolution, ...]:
        return tuple(item for item in self.records if not item.succeeded)


_REFERENCE_FIELDS = frozenset(
    {
        "attention_ids",
        "engagement_evidence_ids",
        "evidence_ids",
        "gap_ids",
        "knowledge_gap_ids",
    }
)
_SOURCE_FIELDS = ("answer_source", "source")
_SOURCE_ALIASES = {
    "current_user": "user",
    "request": "user",
    "user_message": "user",
    "user_observation": "user",
    "user_report": "user",
}
_OBSERVATION_SOURCE_FAMILY = frozenset({"user", "observation"})
_NON_EVIDENCE_SOURCES = frozenset(
    {
        "",
        "none",
        "conversation",
        "general_knowledge",
        "model_knowledge",
        "unknown",
    }
)


def resolve_grounded_wire_provenance(
    payload: dict[str, object],
    bundle: EvidenceBundle,
) -> ProvenanceResolutionResult:
    """Normalize wire-level provenance without changing semantic commitments."""

    normalized = deepcopy(payload)
    records: list[ProvenanceResolution] = []
    reference_aliases = _reference_aliases(bundle)
    _normalize_reference_fields(
        normalized,
        path="$",
        aliases=reference_aliases,
        records=records,
    )

    decision = (
        normalized.get("decision")
        if isinstance(normalized.get("decision"), dict)
        else normalized
    )
    if isinstance(decision, dict):
        _normalize_capability(decision, bundle, records)
        _normalize_epistemic_source(decision, bundle, records)

    return ProvenanceResolutionResult(
        payload=normalized,
        records=tuple(records),
    )


def _reference_aliases(bundle: EvidenceBundle) -> dict[str, set[str]]:
    aliases: dict[str, set[str]] = {}

    def register(alias: str, canonical: str) -> None:
        cleaned = alias.strip()
        if cleaned:
            aliases.setdefault(cleaned, set()).add(canonical)

    for item in bundle.items:
        if not item.authorized:
            continue
        register(item.id, item.id)
        register(item.source_ref, item.id)
    for gap in bundle.knowledge_gaps:
        register(gap.id, gap.id)
    for capability in bundle.capabilities:
        register(capability.id, capability.id)
        register(capability.source_ref, capability.id)
    primary_id = bundle.interaction.primary_observation_id
    primary_evidence_id = (
        f"evidence:observation:{primary_id}" if primary_id else ""
    )
    if primary_evidence_id in bundle.authorized_evidence_ids:
        register(bundle.interaction.id, primary_evidence_id)
    return aliases


def _normalize_reference_fields(
    value: object,
    *,
    path: str,
    aliases: dict[str, set[str]],
    records: list[ProvenanceResolution],
) -> None:
    if isinstance(value, list):
        for index, item in enumerate(value):
            _normalize_reference_fields(
                item,
                path=f"{path}[{index}]",
                aliases=aliases,
                records=records,
            )
        return
    if not isinstance(value, dict):
        return

    for key, item in tuple(value.items()):
        field_path = f"{path}.{key}"
        if key in _REFERENCE_FIELDS and isinstance(item, list):
            resolved_values: list[object] = []
            for index, raw in enumerate(item):
                if not isinstance(raw, str):
                    resolved_values.append(raw)
                    continue
                resolved_values.append(
                    _resolve_reference(
                        raw,
                        path=f"{field_path}[{index}]",
                        aliases=aliases,
                        records=records,
                    )
                )
            value[key] = resolved_values
            continue
        _normalize_reference_fields(
            item,
            path=field_path,
            aliases=aliases,
            records=records,
        )


def _resolve_reference(
    original: str,
    *,
    path: str,
    aliases: dict[str, set[str]],
    records: list[ProvenanceResolution],
) -> str:
    matches = aliases.get(original.strip(), set())
    if len(matches) == 1:
        canonical = next(iter(matches))
        if canonical != original:
            records.append(
                ProvenanceResolution(
                    field_path=path,
                    original_value=original,
                    normalized_value=canonical,
                    rule="authorized_entity_alias_to_evidence_id",
                    evidence_ids=(canonical,),
                    certainty=1.0,
                )
            )
        return canonical
    if len(matches) > 1:
        records.append(
            ProvenanceResolution(
                field_path=path,
                original_value=original,
                normalized_value=original,
                rule="authorized_entity_alias_to_evidence_id",
                evidence_ids=tuple(sorted(matches)),
                certainty=0.0,
                failure_reason="ambiguous_reference",
            )
        )
    else:
        records.append(
            ProvenanceResolution(
                field_path=path,
                original_value=original,
                normalized_value=original,
                rule="authorized_entity_alias_to_evidence_id",
                certainty=0.0,
                failure_reason="reference_not_found",
            )
        )
    return original


def _normalize_capability(
    decision: dict[str, object],
    bundle: EvidenceBundle,
    records: list[ProvenanceResolution],
) -> None:
    raw = decision.get("action")
    if not isinstance(raw, str) or raw.strip().casefold() in {
        "",
        "none",
        "ask",
        "listen",
        "respond",
        "silence",
        "speak",
    }:
        return

    original = raw.strip()
    matches = {
        capability.action_type
        for capability in bundle.capabilities
        if original
        in {
            capability.id,
            capability.action_type,
            capability.source_ref,
            f"capability:{capability.action_type}",
        }
    }
    supporting_ids = tuple(
        sorted(
            capability.id
            for capability in bundle.capabilities
            if capability.action_type in matches
        )
    )
    if len(matches) == 1:
        canonical = next(iter(matches))
        if canonical != original:
            decision["action"] = canonical
            records.append(
                ProvenanceResolution(
                    field_path="$.decision.action",
                    original_value=original,
                    normalized_value=canonical,
                    rule="capability_alias_to_action_type",
                    evidence_ids=supporting_ids,
                    certainty=1.0,
                )
            )
        return
    records.append(
        ProvenanceResolution(
            field_path="$.decision.action",
            original_value=original,
            normalized_value=original,
            rule="capability_alias_to_action_type",
            evidence_ids=supporting_ids,
            certainty=0.0,
            failure_reason=(
                "ambiguous_capability" if len(matches) > 1 else "capability_not_found"
            ),
        )
    )


def _normalize_epistemic_source(
    decision: dict[str, object],
    bundle: EvidenceBundle,
    records: list[ProvenanceResolution],
) -> None:
    source_key = next(
        (
            key
            for key in _SOURCE_FIELDS
            if isinstance(decision.get(key), str)
        ),
        None,
    )
    if source_key is None:
        return

    evidence_ids = [
        str(item)
        for item in decision.get("evidence_ids", [])
        if isinstance(item, str)
    ] if isinstance(decision.get("evidence_ids"), list) else []
    evidence_by_id = {
        item.id: item for item in bundle.items if item.authorized
    }
    supporting = tuple(
        item_id for item_id in evidence_ids if item_id in evidence_by_id
    )
    actual_sources = {
        evidence_by_id[item_id].source for item_id in supporting
    }
    if not actual_sources:
        return

    original = str(decision[source_key]).strip()
    declared = _SOURCE_ALIASES.get(original.casefold(), original.casefold())
    if len(actual_sources) > 1:
        if declared not in actual_sources:
            records.append(
                ProvenanceResolution(
                    field_path=f"$.decision.{source_key}",
                    original_value=original,
                    normalized_value=original,
                    rule="derive_source_from_authorized_evidence",
                    evidence_ids=supporting,
                    certainty=0.0,
                    failure_reason="ambiguous_epistemic_source",
                )
            )
        return

    actual = next(iter(actual_sources))
    if declared == actual:
        return
    if declared in _OBSERVATION_SOURCE_FAMILY and actual in _OBSERVATION_SOURCE_FAMILY:
        decision[source_key] = actual
        records.append(
            ProvenanceResolution(
                field_path=f"$.decision.{source_key}",
                original_value=original,
                normalized_value=actual,
                rule="user_observation_source_alias",
                evidence_ids=supporting,
                certainty=1.0,
            )
        )
        return
    if (
        {declared, actual}.intersection({"self"})
        and {declared, actual}.intersection({"user"})
    ):
        records.append(
            ProvenanceResolution(
                field_path=f"$.decision.{source_key}",
                original_value=original,
                normalized_value=original,
                rule="preserve_distinct_epistemic_subjects",
                evidence_ids=supporting,
                certainty=0.0,
                failure_reason=f"source_conflicts_with_evidence:{actual}",
            )
        )
        return
    if declared in _NON_EVIDENCE_SOURCES or declared != actual:
        decision[source_key] = actual
        records.append(
            ProvenanceResolution(
                field_path=f"$.decision.{source_key}",
                original_value=original,
                normalized_value=actual,
                rule="derive_source_from_authorized_evidence",
                evidence_ids=supporting,
                certainty=1.0,
            )
        )
        return
