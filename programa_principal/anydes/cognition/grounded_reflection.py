from __future__ import annotations

from anydes.cognition.contracts import InternalContext, Intention


def grounded_intention_issues(
    intention: Intention,
    context: InternalContext | None,
) -> tuple[str, ...]:
    """Validate a plan-first Intention without interpreting or rewriting it."""

    plan = intention.grounded_plan
    if plan is None:
        return ()
    issues: list[str] = []
    if context is None:
        return ("grounded_context_missing",)
    bundle = context.evidence_bundle
    if bundle is None:
        return ("evidence_bundle_missing",)

    if not plan.id.strip():
        issues.append("grounded_plan_missing_id")
    if plan.evidence_bundle_id != bundle.id:
        issues.append("grounded_plan_bundle_mismatch")
    if not plan.response_goal.strip():
        issues.append("grounded_plan_missing_response_goal")
    wire_issues = plan.extensions.get("wire_issues", [])
    if isinstance(wire_issues, list):
        issues.extend(f"grounded_wire:{item}" for item in wire_issues)

    known_references = bundle.reference_ids
    evidence_by_id = {item.id: item for item in bundle.items}
    unauthorized = {
        item.id for item in bundle.items if not item.authorized
    }
    for reference_id in sorted(plan.referenced_evidence_ids - known_references):
        issues.append(f"grounded_plan_unknown_evidence:{reference_id}")
    for reference_id in sorted(plan.referenced_evidence_ids.intersection(unauthorized)):
        issues.append(f"grounded_plan_unauthorized_evidence:{reference_id}")

    primary_id = bundle.interaction.primary_observation_id
    primary_evidence_id = (
        f"evidence:observation:{primary_id}" if primary_id else ""
    )
    if (
        bundle.interaction.kind in {"user_turn", "tool_result"}
        and primary_evidence_id
        and primary_evidence_id not in plan.attention.target_ids
    ):
        issues.append("grounded_plan_ignores_primary_observation")
    if (
        plan.engagement.mode == "silence"
        and not bundle.interaction.silence_allowed
    ):
        issues.append("grounded_plan_silence_not_allowed")

    claim_ids = {item.id for item in plan.claims}
    required_claim_ids = {item.id for item in plan.claims if item.required}
    for claim in plan.claims:
        if not claim.content.strip():
            issues.append(f"grounded_claim_empty:{claim.id}")
        references = set(claim.evidence_ids).union(claim.knowledge_gap_ids)
        for reference_id in sorted(references - known_references):
            issues.append(f"grounded_claim_unknown_evidence:{claim.id}:{reference_id}")
        if (
            claim.required
            and not references
            and plan.knowledge.source
            not in {"model_knowledge", "general_knowledge", "conversation"}
            and plan.knowledge.epistemic_status not in {"not_applicable", "unknown"}
        ):
            issues.append(f"grounded_claim_without_support:{claim.id}")
        request_evidence_ids = {
            reference_id
            for reference_id in claim.evidence_ids
            if (
                reference_id in evidence_by_id
                and evidence_by_id[reference_id].attributes.get("evidence_role")
                == "request"
            )
        }
        if (
            request_evidence_ids
            and plan.knowledge.epistemic_status == "known"
            and plan.knowledge.source
            not in {"conversation"}
        ):
            issues.append(f"grounded_claim_uses_request_as_fact:{claim.id}")

    if (
        plan.knowledge.epistemic_status == "unknown"
        and not plan.knowledge.knowledge_gap_ids
    ):
        issues.append("unknown_knowledge_without_gap")
    if plan.knowledge.epistemic_status == "unknown":
        for claim in plan.claims:
            if claim.predicate != "uncertainty" or claim.role != "caveat":
                issues.append(
                    f"unknown_knowledge_has_assertive_claim:{claim.id}"
                )
    unestablished_preference_gap = next(
        (
            item
            for item in bundle.knowledge_gaps
            if item.predicate == "unlisted_preferences"
        ),
        None,
    )
    established_preferences = next(
        (
            item
            for item in bundle.items
            if item.source == "self"
            and item.predicate == "established_preferences"
        ),
        None,
    )
    if (
        unestablished_preference_gap is not None
        and (
            established_preferences is None
            or not established_preferences.value
        )
        and _targets_self_knowledge(plan.knowledge.subject, plan.knowledge.source)
        and plan.knowledge.epistemic_status == "known"
        and unestablished_preference_gap.id
        not in plan.knowledge.knowledge_gap_ids
    ):
        issues.append("known_self_preference_conflicts_with_gap")
    if (
        plan.knowledge.epistemic_status == "known"
        and plan.knowledge.source
        not in {"model_knowledge", "general_knowledge", "conversation"}
        and not plan.knowledge.evidence_ids
    ):
        issues.append("known_knowledge_without_evidence")
    expected_source = {
        "user": "user",
        "self": "self",
        "memory": "memory",
        "relationship": "relationship",
        "tool": "tool",
        "observation": "observation",
        "environment": "environment",
    }.get(plan.knowledge.source)
    if expected_source is not None and plan.knowledge.epistemic_status == "known":
        matching_source = any(
            reference_id in evidence_by_id
            and evidence_by_id[reference_id].source == expected_source
            for reference_id in plan.knowledge.evidence_ids
        )
        if not matching_source:
            issues.append(
                f"knowledge_source_without_matching_evidence:{expected_source}"
            )

    conversation = intention.conversation_intent
    if conversation.plan_id != plan.id:
        issues.append("conversation_plan_link_mismatch")
    for claim_id in sorted(set(conversation.claim_ids) - claim_ids):
        issues.append(f"conversation_unknown_claim:{claim_id}")
    if (
        plan.output_manifest.conversation_required
        and required_claim_ids
        and not required_claim_ids.issubset(conversation.claim_ids)
    ):
        issues.append("conversation_missing_required_claim")

    speech = intention.speech_intention
    if speech.plan_id != plan.id:
        issues.append("speech_plan_link_mismatch")
    for claim_id in sorted(set(speech.realized_claim_ids) - claim_ids):
        issues.append(f"speech_unknown_claim:{claim_id}")
    if plan.engagement.should_speak:
        if not intention.speech.strip():
            issues.append("grounded_speech_missing")
        if (
            plan.output_manifest.speech_required
            and not required_claim_ids.issubset(speech.realized_claim_ids)
        ):
            issues.append("speech_missing_required_claim")
    elif intention.speech.strip():
        issues.append("speech_present_when_plan_requires_silence")
    if abs(speech.warmth - plan.expression.warmth) > 0.4:
        issues.append("speech_warmth_diverges_from_plan")
    if abs(speech.energy - plan.expression.energy) > 0.4:
        issues.append("speech_energy_diverges_from_plan")

    body = intention.body_intention
    if plan.output_manifest.body_required and body is None:
        issues.append("body_intention_missing")
    if body is not None:
        if body.plan_id != plan.id:
            issues.append("body_plan_link_mismatch")
        if abs(body.intensity - plan.expression.intensity) > 0.4:
            issues.append("body_intensity_diverges_from_plan")

    actual_action_ids = {item.id for item in intention.actions}
    issues.extend(
        _manifest_issues(
            "action",
            set(plan.output_manifest.action_ids),
            actual_action_ids,
        )
    )
    if plan.action.phase == "none" and intention.actions:
        issues.append("actions_present_without_action_commitment")
    if plan.action.phase == "request" and not intention.actions:
        issues.append("action_commitment_missing_request")
    capability_ids = {item.id for item in bundle.capabilities}
    if plan.action.capability_id and plan.action.capability_id not in capability_ids:
        issues.append(f"unknown_capability:{plan.action.capability_id}")
    for action in intention.actions:
        if action.plan_id != plan.id:
            issues.append(f"action_plan_link_mismatch:{action.id}")
        for evidence_id in sorted(set(action.evidence_ids) - known_references):
            issues.append(f"action_unknown_evidence:{action.id}:{evidence_id}")

    if (
        bundle.interaction.kind == "proactive_opportunity"
        and plan.engagement.should_speak
    ):
        initiative_refs = {
            reference_id
            for reference_id in plan.engagement.evidence_ids
            if reference_id in evidence_by_id
            and evidence_by_id[reference_id].attributes.get("evidence_role")
            not in {"background", "state", "event", "request"}
        }
        if not initiative_refs:
            issues.append("proactive_speech_without_grounded_trigger")

    proposal_groups = (
        (
            "memory",
            set(plan.output_manifest.memory_proposal_ids),
            {item.id for item in intention.memory_updates},
        ),
        (
            "relationship",
            set(plan.output_manifest.relationship_proposal_ids),
            {item.id for item in intention.relationship_updates},
        ),
        (
            "self",
            set(plan.output_manifest.self_proposal_ids),
            {item.id for item in intention.self_updates},
        ),
        (
            "curiosity",
            set(plan.output_manifest.curiosity_proposal_ids),
            {item.id for item in intention.curiosity_updates},
        ),
    )
    for label, expected, actual in proposal_groups:
        issues.extend(_manifest_issues(label, expected, actual))
    for memory in intention.memory_updates:
        if str(memory.metadata.get("plan_id", "")) != plan.id:
            issues.append(f"memory_plan_link_mismatch:{memory.id}")

    return tuple(dict.fromkeys(issues))


def _targets_self_knowledge(subject: str, source: str) -> bool:
    normalized = subject.strip().casefold()
    return source == "self" or normalized in {
        "anydes",
        "self",
        "anydes_self",
        "la propia anydes",
    }


def _manifest_issues(
    label: str,
    expected: set[str],
    actual: set[str],
) -> list[str]:
    issues: list[str] = []
    for identifier in sorted(expected - actual):
        issues.append(f"{label}_manifest_missing:{identifier}")
    for identifier in sorted(actual - expected):
        issues.append(f"{label}_manifest_unlisted:{identifier}")
    return issues
