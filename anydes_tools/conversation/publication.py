"""Staging publication authority. Rendering never executes or mutates runtime facts."""
from dataclasses import asdict, dataclass, field
from enum import StrEnum
import json
import re
from uuid import uuid4
from .typed_content import ContentKind, ContentSegment


class OutcomeStatus(StrEnum):
    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    FAILED = "FAILED"
    INCONCLUSIVE = "INCONCLUSIVE"
    PENDING_CONFIRMATION = "PENDING_CONFIRMATION"
    CLARIFICATION = "CLARIFICATION"
    UNSUPPORTED = "UNSUPPORTED"
    CONVERSATIONAL = "CONVERSATIONAL"


@dataclass(frozen=True)
class RuntimeOutcome:
    outcome_id: str = field(default_factory=lambda: "outcome:" + uuid4().hex)
    interaction_kind: str = "CONVERSATION"
    task_id: str = ""
    status: OutcomeStatus = OutcomeStatus.CONVERSATIONAL
    user_goal: str = ""
    committed_artifacts: tuple[dict, ...] = ()
    completed_steps: tuple[str, ...] = ()
    failed_step: str = ""
    blocked_steps: tuple[str, ...] = ()
    failure_ref: str = ""
    verification_summary: tuple[dict, ...] = ()
    pending_action: dict = field(default_factory=dict)
    clarification: str = ""
    safe_evidence: dict = field(default_factory=dict)
    presentation_hints: dict = field(default_factory=dict)
    attempt_number: int = 1
    plan_revision: int = 1


@dataclass(frozen=True)
class PublicationDecision:
    outcome_id: str
    status: OutcomeStatus
    allowed_claims: tuple[str, ...]
    artifact_ids: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class FinalResponse:
    outcome_id: str
    visible_text: str
    spoken_text: str
    artifact_names: tuple[str, ...]
    decision: PublicationDecision
    sanitized: bool = False
    content_segments: tuple[ContentSegment, ...] = ()


_INTERNAL_KEYS = frozenset({"capability_id", "capability", "tool_id", "artifact_id",
    "failure_id", "operation_descriptor", "requested_final_outputs", "schema_version",
    "consumes_artifacts", "produces_artifacts", "original_plan_ref", "$schema"})
_INTERNAL_IDS = re.compile(r"\b(?:artifact|failure|verification|outcome|task|plan):[\w:.-]+"
    r"|\b(?:official|developer|document|filesystem|archive)\.[a-z_][\w.]*")
_INTERNAL_MARKERS = re.compile(r"\b(?:BrainPlanProposal|ExecutableTaskPlan|ArtifactRef|FailureRecord)\b"
    r"|</?(?:think|analysis|internal|tool_call|action_request|function_call)\b"
    r"|\"(?:" + "|".join(re.escape(k) for k in sorted(_INTERNAL_KEYS)) + r")\"\s*:")
_PRIVATE_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|\\\\)[^\r\n`<>\"|?*]+")
_SECRET = re.compile(r"\b(?:Bearer\s+\S+|(?:gsk_|sk-)[A-Za-z0-9_-]{12,})"
    r"|\b(?:api[_ -]?key|access[_ -]?token|password|secret)\s*[=:]\s*\S+", re.I)
_FIRST_PERSON_EFFECT = re.compile(r"\b(?:ya\s+)?(?:lo\s+|la\s+)?"
    r"(?:guarde|cree|edite|elimine|comprimi|converti|ejecute|renombre|movi)\b"
    r"|\bhe\s+(?:guardado|creado|editado|eliminado|comprimido|convertido|ejecutado)\b")


def unbacked_action_claim(text):
    from .response_gate import model_output_claims_side_effect, _strip_accents
    normalized = _strip_accents(str(text).casefold())
    # An introduction to requested prose is not evidence of a filesystem effect.
    without_intro = re.sub(r"\baqui\s+tienes\b", "", normalized)
    return bool(_FIRST_PERSON_EFFECT.search(normalized)) or model_output_claims_side_effect(without_intro)


def _contains_protocol(value):
    if isinstance(value, dict):
        return bool(_INTERNAL_KEYS.intersection(value)) or any(_contains_protocol(v) for v in value.values())
    if isinstance(value, list):
        return any(_contains_protocol(v) for v in value)
    return False


def sanitize_public_text(text, *, reveal_paths=False):
    from .presentation import sanitize_visible_model_text
    text = str(text or "")
    # Structured protocol is rejected as a whole, not partially rendered as prose.
    for candidate in (text.strip(), text.strip().removeprefix("```json").removesuffix("```").strip()):
        try:
            if _contains_protocol(json.loads(candidate)):
                return "", True
        except (ValueError, TypeError):
            pass
    clean, blocked = sanitize_visible_model_text(text)
    if blocked or _INTERNAL_IDS.search(clean) or _INTERNAL_MARKERS.search(clean) or "Traceback (most recent call last)" in clean:
        return "", True
    clean = _SECRET.sub("[dato protegido]", clean)
    if not reveal_paths:
        clean = _PRIVATE_PATH.sub("Carpeta de trabajo de Anydes", clean)
    return clean.strip(), clean.strip() != text.strip()


def friendly_name(value):
    # Display names are metadata, never filesystem authority.
    name = str(value or "el archivo").replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    clean, _ = sanitize_public_text(name)
    return clean[:160] or "el archivo"


def publication_decision(outcome):
    retained = tuple(a for a in outcome.committed_artifacts if a.get('state') == 'COMMITTED_UNVERIFIED'
        and a.get('task_id') == outcome.task_id and a.get('metadata',{}).get('user_accepted_unverified') is True)
    if retained and outcome.safe_evidence.get('quality_retention'):
        return PublicationDecision(outcome.outcome_id, OutcomeStatus.INCONCLUSIVE,
            ('ARTIFACT_RETAINED',), tuple(a['artifact_id'] for a in retained), 'USER_ACCEPTED_QUALITY_NOT_SECURITY')
    artifacts = tuple(a for a in outcome.committed_artifacts
        if a.get("state") == "COMMITTED" and a.get("task_id") == outcome.task_id)
    claims = []
    status = outcome.status
    verified = bool(outcome.verification_summary) and all(
        v.get("status") == "PASS" for v in outcome.verification_summary)
    if status == OutcomeStatus.SUCCESS and any(v.get("status") in {"FAIL", "INCONCLUSIVE"}
            for v in outcome.verification_summary):
        status = OutcomeStatus.FAILED if any(v.get("status") == "FAIL" for v in outcome.verification_summary) else OutcomeStatus.INCONCLUSIVE
    if status == OutcomeStatus.SUCCESS:
        if (artifacts and verified) or outcome.safe_evidence.get("verified_effects"):
            claims.append("EFFECT_COMPLETED")
            if artifacts:
                claims.append("ARTIFACT_COMMITTED")
            if verified:
                claims.append("VERIFICATION_PASSED")
        elif outcome.safe_evidence.get("information_only"):
            claims.append("INFORMATION")
        else:
            status = OutcomeStatus.INCONCLUSIVE
    elif status == OutcomeStatus.PARTIAL_SUCCESS:
        if artifacts or outcome.safe_evidence.get("verified_effects"):
            claims.append("PARTIAL_EFFECT_COMPLETED")
        else:
            status = OutcomeStatus.FAILED
    return PublicationDecision(outcome.outcome_id, status, tuple(claims),
        tuple(a["artifact_id"] for a in artifacts if a.get("artifact_id")),
        "EVIDENCE_BOUND" if claims else "NO_SUCCESS_CLAIM")


class FinalResponseComposer:
    @staticmethod
    def compose(outcome, *, conversational_text="", reveal_paths=False):
        decision = publication_decision(outcome)
        names = tuple(friendly_name(a.get("display_name")) for a in outcome.committed_artifacts
            if a.get("artifact_id") in decision.artifact_ids)
        status = decision.status
        detail = outcome.safe_evidence.get("failure_message", "")
        source = outcome.safe_evidence.get('read_source_text')
        if 'INFORMATION' in decision.allowed_claims and outcome.safe_evidence.get('read_kind') == 'CODE' and isinstance(source,str):
            clean, _ = sanitize_public_text(source, reveal_paths=reveal_paths)
            if clean != source.strip():
                safe = 'No puedo mostrar literalmente este codigo porque contiene informacion protegida o interna.'
                return FinalResponse(outcome.outcome_id,safe,safe,names,decision,True)
            heading = 'Contenido de '+friendly_name(outcome.safe_evidence['read_name'])+':\n'
            segments = (ContentSegment(ContentKind.TEXT,heading),
                ContentSegment(ContentKind.CODE,source,outcome.safe_evidence.get('read_language','')))
            text = heading+source
            return FinalResponse(outcome.outcome_id,text,text,names,decision,False,segments)
        if status == OutcomeStatus.CONVERSATIONAL or "INFORMATION" in decision.allowed_claims:
            text = outcome.safe_evidence.get('read_content', conversational_text)
            if not outcome.safe_evidence.get('content_is_read_data') and unbacked_action_claim(text):
                text = ("Tengo experiencias de acciones verificadas relacionadas con eso."
                    if outcome.safe_evidence.get("historical_action_evidence") else
                    "No tengo una acción verificada que respalde esa afirmación.")
        elif status == OutcomeStatus.SUCCESS:
            prefix = "Ahora sí, el nuevo intento terminó." if outcome.attempt_number > 1 else "Listo."
            text = prefix + (" Quedó guardado: " + ", ".join(names) + "." if names
                else " El cambio quedó verificado.")
            if "VERIFICATION_PASSED" in decision.allowed_claims:
                text += " Pasó las verificaciones requeridas."
            if outcome.safe_evidence.get('unverified_inputs'):
                text += ' El archivo de origen sigue sin confirmar como correcto.'
        elif status == OutcomeStatus.PARTIAL_SUCCESS:
            text = ("Quedó guardado: " + ", ".join(names) + ". " if names else "Una parte del cambio quedó hecha. ")
            batch = outcome.safe_evidence.get("batch", {})
            if batch:
                text = f"Eliminé {batch['deleted']} archivo(s). "
            text += "No pude completar el resto. " + (detail or "No tengo más detalle registrado.")
            if batch:
                text += f" {batch['failed']} no se eliminaron y {batch['skipped']} se omitieron."
            if outcome.blocked_steps:
                text += " Los pasos dependientes no se ejecutaron."
        elif status == OutcomeStatus.INCONCLUSIVE:
            if 'ARTIFACT_RETAINED' in decision.allowed_claims:
                text = 'Lo guarde para que puedas revisarlo: ' + ', '.join(names) + '. '
                text += ('No supero las pruebas; es un borrador, no un programa confirmado como correcto.'
                    if any(v.get('status') == 'FAIL' for v in outcome.verification_summary) else
                    'No pude confirmar que funcione correctamente; sigue sin verificar.')
                clean, sanitized = sanitize_public_text(text)
                return FinalResponse(outcome.outcome_id, clean, clean, names, decision, sanitized)
            text = ("El programa se ejecutó, pero no tengo evidencia suficiente para confirmar que el resultado sea correcto."
                if outcome.safe_evidence.get("code_executed") else
                "No tengo evidencia suficiente para confirmar que el resultado sea correcto.")
            text += " No publiqué un nuevo archivo." if not names else " No confirmé el objetivo completo."
        elif status == OutcomeStatus.FAILED:
            text = detail or "No pude terminarlo y no tengo una causa más específica registrada."
        elif status == OutcomeStatus.PENDING_CONFIRMATION:
            effects = outcome.pending_action.get("effects", ())
            labels = {"EXECUTE_CODE": "ejecutar el código", "DELETE": "eliminar los archivos",
                "OVERWRITE": "reemplazar el archivo", "MOVE": "mover el archivo"}
            purpose = ", ".join(labels[e] for e in effects if e in labels) or "continuar con la acción pendiente"
            text = f"Necesito tu aprobación para {purpose}. ¿Quieres que continúe?"
            targets = outcome.pending_action.get("target_names", ())
            if targets:
                text = f"Necesito tu aprobación para {purpose}: " + ", ".join(friendly_name(n) for n in targets) + ". ¿Quieres que continúe?"
            if "DELETE" in effects:
                text += " Esta eliminación no se puede deshacer."
        elif status == OutcomeStatus.CLARIFICATION:
            text = outcome.clarification or "¿Qué archivo o resultado quieres usar?"
        else:
            text = "No tengo todavía una herramienta autorizada para hacer eso."
        clean, sanitized = sanitize_public_text(text, reveal_paths=reveal_paths)
        if not clean:
            clean = "No puedo mostrar ese contenido interno."
        # The same approved payload feeds chat and the speech intention.
        return FinalResponse(outcome.outcome_id, clean, clean, names, decision, sanitized)


def outcome_from_controller(controller, response=None, *, user_goal="", prior_failure_id=None):
    """Compatibility adapter: read current-turn evidence, never re-run planning."""
    resolution = controller.last_resolution
    kind = resolution.kind.value if resolution else "CONVERSATION"
    control = resolution.task_control.value if resolution else "NONE"
    explaining = control in {'EXPLAIN_FAILURE', 'CONTINUE_EXPLANATION', 'REMEDIATION_QUERY'}
    state = controller.state
    pending = state.pending_task
    base = dict(interaction_kind=kind, user_goal=user_goal,
        task_id=pending.task_id if kind == "TOOL_ACTION" or explaining else "",
        attempt_number=pending.attempt_number, plan_revision=pending.plan_revision)
    if response and response.debug.get('response_source') == 'READ_CONTENT':
        return RuntimeOutcome(**base, status=OutcomeStatus.SUCCESS, safe_evidence={
            'information_only':True, 'content_is_read_data':True,
            'read_content':'Contenido de '+friendly_name(response.debug['display_name'])+':\n'+response.text,
            'read_source_text':response.text, 'read_kind':response.debug.get('artifact_type',''),
            'read_name':response.debug['display_name'], 'read_language':response.debug.get('format',''),
            'read_artifact':{k:response.debug[k] for k in ('artifact_id','version','hash')}})
    if response and response.debug.get('response_source') == 'READ_UNSUPPORTED':
        return RuntimeOutcome(**base, status=OutcomeStatus.UNSUPPORTED)
    if response and response.debug.get('response_source') == 'EXPLANATION_SECONDARY_FAILURE':
        base['task_id'] = response.debug['task_id']
        return RuntimeOutcome(**base, status=OutcomeStatus.FAILED,
            failure_ref=response.debug['failure_ref'], safe_evidence={
                'failure_message':response.text, 'reason_code':response.debug['reason_code']})
    if response and response.debug.get("response_source") in {"ARTIFACT_REFERENCE_CLARIFICATION", "FAILURE_REFERENCE_CLARIFICATION"}:
        return RuntimeOutcome(**base, status=OutcomeStatus.CLARIFICATION,
            clarification=response.text, safe_evidence={"reason_code":"ARTIFACT_REFERENCE_AMBIGUOUS_OR_MISSING"})
    if kind in {"CONVERSATION", "CONTENT_RESPONSE"} and not explaining:
        return RuntimeOutcome(**base, safe_evidence={
            "historical_action_evidence": bool(response and response.debug.get("evidence_ids"))})
    if kind == "CLARIFICATION":
        if getattr(resolution, "source", "") == "LOCAL_FAIL_CLOSED":
            failure = state.last_tool_failure
            if response and response.debug.get('response_source') == 'INTERNAL_PIPELINE_FAILURE' and failure:
                from .failure_state import FailureResponseRenderer
                base['task_id'] = failure.task_id
                return RuntimeOutcome(**base,status=OutcomeStatus.FAILED,failure_ref=failure.failure_id,
                    safe_evidence={'failure_message':FailureResponseRenderer.render(failure),
                        'reason_code':failure.error_code, 'failure_stage':failure.stage})
            return RuntimeOutcome(**base, status=OutcomeStatus.FAILED,
                safe_evidence={"failure_message": "No pude obtener una interpretación válida de esta petición."})
        return RuntimeOutcome(**base, status=OutcomeStatus.CLARIFICATION,
            clarification=response.text if response else "")
    if kind == "UNSUPPORTED":
        return RuntimeOutcome(**base, status=OutcomeStatus.UNSUPPORTED)
    if control == "CANCEL":
        return RuntimeOutcome(**base, status=OutcomeStatus.CONVERSATIONAL)
    failure = state.failure_topic_record() if explaining else state.last_tool_failure
    if explaining and failure:
        base['task_id'] = failure.task_id
    linked = failure is not None and failure.task_id == base["task_id"]
    if linked and control == "NONE" and failure.failure_id == prior_failure_id:
        linked = False
        if not controller.last_results:
            base["task_id"] = ""
    verification = state.verification_evidence.get(failure.verification_ref) if linked else None
    if explaining:
        from .failure_state import FailureResponseRenderer
        return RuntimeOutcome(**base, status=OutcomeStatus.FAILED,
            failure_ref=failure.failure_id if linked else "", safe_evidence={
                "failure_message": response.text if linked and response else FailureResponseRenderer.render(failure) if linked else ""})
    if state.pending_tool_action is not None:
        if state.pending_tool_action.reason == "missing_rename_name":
            return RuntimeOutcome(**base, status=OutcomeStatus.CLARIFICATION,
                clarification="¿Qué nombre quieres darle al archivo?")
        action = state.pending_tool_action.action
        targets = tuple(t.get("display_name", "el archivo") for t in action.arguments.get("delete_plan", {}).get("targets", ()))
        if not targets and action.target_path:
            targets = (friendly_name(action.target_path),)
        return RuntimeOutcome(**base, status=OutcomeStatus.PENDING_CONFIRMATION,
            pending_action={"effects": [e.value for e in state.pending_tool_action.confirmed_effects],
                "reason": state.pending_tool_action.reason, "target_names": targets})
    # A summary from a previous turn cannot authorize this turn's publication.
    summary = controller.last_dataflow_summary if controller.last_step_results else {}
    if summary.get("task_id") != base["task_id"]:
        summary = {}
    artifacts = tuple(summary.get("committed_artifacts", ()))
    retained = tuple(a for a in artifacts if a.get('state') == 'COMMITTED_UNVERIFIED')
    if retained:
        return RuntimeOutcome(**base, status=OutcomeStatus.INCONCLUSIVE, committed_artifacts=retained,
            verification_summary=tuple(a.get('metadata',{}).get('verification_result',{}) for a in retained),
            safe_evidence={'quality_retention':True})
    results = controller.last_results
    all_success = bool(results) and all(r.status.value == "SUCCESS" for r in results)
    effects = tuple({"effect": e.effect.value, "name": friendly_name(e.path)}
        for r in results if r.status.value == "SUCCESS" for e in r.side_effects
        if not friendly_name(e.path).startswith(".anydes-"))
    verified_effects = effects if (all_success or pending.status == "COMPLETED_PARTIAL") and pending.status in {"COMPLETED", "COMPLETED_PARTIAL"} else ()
    checks = tuple(v.verification_result for v in state.verification_evidence.values()
        if v.verification_result.get("artifact_id") in {a.get("artifact_id") for a in artifacts})
    if not checks and all_success and state.last_verification.get("status") == "VERIFIED":
        checks = ({"status": "PASS", "source": "LEGACY_VERIFIED_GATE"},)
    status = OutcomeStatus.FAILED
    if artifacts and (linked or summary.get("failed_step")):
        status = OutcomeStatus.PARTIAL_SUCCESS
    elif linked and verification and verification.verification_result.get("status") == "INCONCLUSIVE":
        status = OutcomeStatus.INCONCLUSIVE
    elif all_success and pending.status == "COMPLETED":
        status = OutcomeStatus.SUCCESS
    elif pending.status == "COMPLETED_PARTIAL":
        status = OutcomeStatus.PARTIAL_SUCCESS
    elif response and response.resolver_kind and response.resolver_kind.value in {"MISSING_CAPABILITY", "UNSUPPORTED"}:
        status = OutcomeStatus.UNSUPPORTED
    return RuntimeOutcome(**base, status=status, committed_artifacts=artifacts,
        completed_steps=tuple(summary.get("completed_steps", ())), failed_step=summary.get("failed_step", ""),
        blocked_steps=tuple(summary.get("blocked_steps", ())),
        failure_ref=failure.failure_id if linked else "", verification_summary=checks,
        safe_evidence={"failure_message": failure.safe_user_message if linked else "",
            "reason_code": failure.error_code if linked else "",
            "failure_stage": failure.stage if linked else "",
            "verified_effects": verified_effects,
            "batch": {"deleted": len(state.last_verification.get("deleted", ())),
                "failed": len(state.last_verification.get("failed", ())),
                "skipped": len(state.last_verification.get("skipped", ()))}
                if pending.status == "COMPLETED_PARTIAL" and state.last_verification.get("status") == "VERIFIED_PARTIAL" else {},
            "code_executed": any(r.capability.startswith("developer.run_") and r.status.value == "SUCCESS"
                and r.output.get("returncode") == 0 for r in results),
            'unverified_inputs': 'COMMITTED_UNVERIFIED' in summary.get('artifact_states',{}).values(),
            "information_only": all_success and not effects and not artifacts})
