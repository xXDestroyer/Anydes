"""Task relevance is separate from physical validity and universal factual truth."""
from dataclasses import asdict, dataclass, field, replace
import hashlib
import re
from uuid import uuid4
from .verification_types import VerificationStatus
from ..schema_validation import Draft202012Validator


ADHERENCE_SCHEMA = {"type": "object", "additionalProperties": False,
    "required": ["status", "goal_aligned", "subject_aligned", "reason_code"],
    "properties": {"status": {"enum": ["PASS", "FAIL", "INCONCLUSIVE"]},
        "goal_aligned": {"type": "boolean"}, "subject_aligned": {"type": "boolean"},
        "reason_code": {"enum": ["ALIGNED", "GOAL_MISMATCH", "SUBJECT_MISMATCH", "INSUFFICIENT_EVIDENCE"]},
        "safe_reason": {"type": "string", "maxLength": 400},
        "evidence_basis": {"type": "string", "maxLength": 400}}}


def bounded_context(text, limit=6000):
    return re.sub(r"(?im)^.*(?:api[_-]?key|password|token|secret|credential).*$",
        "[REDACTED]", str(text))[:limit]


@dataclass(frozen=True)
class TaskAdherenceResult:
    status: VerificationStatus
    goal_aligned: bool = False
    subject_aligned: bool = False
    reason_code: str = "INSUFFICIENT_EVIDENCE"
    evaluation_source: str = "UNAVAILABLE"
    task_id: str = ""
    content_hash: str = ""
    context_refs: tuple = ()
    diagnostic_evidence: dict = field(default_factory=dict)
    safe_reason: str = ""
    evidence_basis: str = ""
    adherence_id: str = field(default_factory=lambda: "adherence:" + uuid4().hex)

    def to_dict(self):
        return {**asdict(self), "status": self.status.value}

    @classmethod
    def from_payload(cls, payload, **identity):
        if not Draft202012Validator(ADHERENCE_SCHEMA).is_valid(payload):
            raise ValueError("adherence_schema_invalid")
        status = VerificationStatus(payload["status"])
        aligned = payload["goal_aligned"] and payload["subject_aligned"]
        if (status == VerificationStatus.PASS and (not aligned or payload["reason_code"] != "ALIGNED")) or (
            status == VerificationStatus.FAIL and payload["reason_code"] not in {"GOAL_MISMATCH", "SUBJECT_MISMATCH"}):
            raise ValueError("adherence_contradiction")
        from .failure_state import safe_detail
        return cls(status, payload["goal_aligned"], payload["subject_aligned"], payload["reason_code"],
            safe_reason=safe_detail(payload.get("safe_reason", ""), 400),
            evidence_basis=safe_detail(payload.get("evidence_basis", ""), 400), **identity)


def evaluate_adherence(evaluator, *, goal, specification, content, task_id="", context=""):
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    source = bounded_context(specification.source_context or context)
    refs = (hashlib.sha256(source.encode()).hexdigest(),) if source else ()
    identity = {"task_id": task_id, "content_hash": digest, "context_refs": refs}
    from .failure_state import safe_detail
    identity["diagnostic_evidence"] = {
        "requested_goal": safe_detail(bounded_context(goal, 1000), 1000),
        "requested_subject": safe_detail(bounded_context(specification.topic, 500), 500),
        "requested_output_type": safe_detail(specification.purpose, 100),
        "content_characters": len(content), "content_words": len(content.split()),
        "paragraph_count": len([p for p in content.split("\n\n") if p.strip()]),
        "paragraph_count_basis": "BLANK_LINE_SEPARATORS",
        "nonempty_line_count": len([line for line in content.splitlines() if line.strip()]),
        "content_snippet": safe_detail(bounded_context(content, 700), 700)}
    # Oversized content is inconclusive, never evaluated by only its misleading prefix.
    if evaluator is None or len(content) > 20000:
        return TaskAdherenceResult(VerificationStatus.INCONCLUSIVE, **identity)
    evidence = {"original_goal": bounded_context(goal, 2000), "requested_subject": specification.topic[:1000],
        "requested_output_type": specification.purpose[:100], "context_refs": refs,
        "relevant_context": source, "produced_content": bounded_context(content, 20000),
        "content_hash": digest, "evaluation_scope": "CONTENT_SEMANTICS_ONLY",
        "content_requirements": {"structure": specification.structure,
            "constraints": [bounded_context(c, 300) for c in specification.constraints[:12]]},
        "host_verified_responsibilities": ["ARTIFACT_MATERIALIZATION", "PHYSICAL_VALIDITY", "COMMIT_PUBLICATION"]}
    try:
        return TaskAdherenceResult.from_payload(evaluator(evidence), evaluation_source="SEMANTIC_EVALUATOR", **identity)
    except (ValueError, TypeError, KeyError, RuntimeError, AttributeError):
        return TaskAdherenceResult(VerificationStatus.INCONCLUSIVE, **identity)


def aggregate_adherence(verification, adherence):
    """Fail dominates; inconclusive never becomes success. No oracle is invented."""
    evidence = {**verification.evidence, "task_adherence": adherence.to_dict()}
    if verification.status == VerificationStatus.FAIL:
        return replace(verification, evidence=evidence)
    if adherence.status != VerificationStatus.PASS:
        status = VerificationStatus.FAIL if adherence.status == VerificationStatus.FAIL else VerificationStatus.INCONCLUSIVE
        return replace(verification, status=status, evidence=evidence, failed_check="task_adherence",
            error_code="TASK_ADHERENCE_" + status.value,
            safe_message="El contenido no corresponde al objetivo solicitado." if status == VerificationStatus.FAIL
                else "No pude comprobar que el contenido corresponda al objetivo solicitado.")
    return replace(verification, evidence=evidence)
