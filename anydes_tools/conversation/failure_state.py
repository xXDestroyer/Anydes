from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import re
import time
import unicodedata
from uuid import uuid4


def safe_detail(value: str, limit: int = 500) -> str:
    value = str(value)
    value = re.sub(r"gsk_[A-Za-z0-9]+|Bearer\s+\S+|AKIA[A-Z0-9]{16}", "<redacted>", value, flags=re.I)
    value = re.sub(r"(?i)(api[_-]?key|token|password|secret|credential)\s*[:=]\s*[^\s,;]+",
        r"\1=<redacted>", value)
    value = re.sub(r"[A-Za-z]:[\\/][^\s\"']+", "<path>", value)
    return value[:limit]


def text_summary(value: str) -> dict:
    encoded = str(value).encode("utf-8")
    return {"characters": len(str(value)), "sha256": hashlib.sha256(encoded).hexdigest()}


def plan_reference(plan) -> str:
    if plan is None:
        return ""
    encoded = json.dumps(asdict(plan), sort_keys=True, ensure_ascii=True,
        default=str, separators=(",", ":")).encode("utf-8")
    return f"plan:{plan.task_id}:r{plan.revision}:sha256:{hashlib.sha256(encoded).hexdigest()}"


def artifact_type_for(operation: str, capability: str) -> str:
    if operation == "document_conversion":
        return "CONVERSION"
    if operation in {"general_generate_code", "general_modify_code", "run_inline_python"}:
        return "CODE"
    for prefix, kind in (("developer.", "CODE"), ("document.", "DOCUMENT"),
                         ("archive.", "ARCHIVE"), ("filesystem.", "FILESYSTEM")):
        if capability.startswith(prefix):
            return kind
    return "UNKNOWN"


@dataclass(frozen=True)
class VerificationEvidence:
    verification_id: str
    verifier_type: str
    failed_check: str
    cause_code: str
    input_summary: dict
    expected_summary: tuple[dict, ...]
    actual_summary: dict
    exit_code: int | None
    stdout_summary: dict
    stderr_summary: dict
    execution_request_id: str
    verification_result: dict = field(default_factory=dict)

    @classmethod
    def code_test(cls, test, run, passed: bool, verification=None):
        stdout = str(run.output.get("stdout", ""))
        stderr = str(run.output.get("stderr", ""))
        exit_code = run.output.get("returncode")
        return cls(
            verification_id=verification.verification_id if verification else f"verification:{uuid4().hex}", verifier_type="CODE",
            failed_check="" if passed else safe_detail(test.test_id, 120),
            cause_code="" if passed else verification.error_code if verification else (
                "TEST_ASSERTION_FAILED" if run.status.value == "SUCCESS" and exit_code == 0
                else "CODE_EXECUTION_FAILED"),
            input_summary=text_summary(test.stdin),
            expected_summary=tuple(text_summary(v) for v in test.expected_stdout_contains[:16]) or (text_summary(test.expected),),
            actual_summary=text_summary(stdout), exit_code=exit_code,
            stdout_summary=text_summary(stdout), stderr_summary=text_summary(stderr),
            execution_request_id=run.request_id,
            verification_result=verification.to_dict() if verification else {})


@dataclass(frozen=True)
class FailureRecord:
    task_id: str
    stage: str
    error_code: str
    safe_user_message: str
    retryable: bool
    timestamp: float
    plan_id: str = ""
    technical_message: str = ""
    failure_id: str = field(default_factory=lambda: f"failure:{uuid4().hex}")
    plan_revision: int = 1
    attempt_number: int = 1
    step_id: str = ""
    operation_type: str = ""
    artifact_type: str = "UNKNOWN"
    capability_id: str = ""
    original_plan_ref: str = ""
    verification_ref: str = ""
    validation_detail: dict = field(default_factory=dict)
    caused_during_explanation_of: str = ""

    @classmethod
    def create(
        cls,
        *,
        task_id: str,
        stage: str,
        error_code: str,
        safe_user_message: str,
        retryable: bool,
        plan_id: str = "",
        technical_message: str = "",
        plan_revision: int = 1,
        attempt_number: int = 1,
        step_id: str = "",
        operation_type: str = "",
        artifact_type: str = "UNKNOWN",
        capability_id: str = "",
        original_plan_ref: str = "",
        verification_ref: str = "",
        validation_detail: dict | None = None,
        caused_during_explanation_of: str = "",
    ) -> "FailureRecord":
        return cls(
            task_id=task_id,
            stage=stage,
            error_code=error_code,
            safe_user_message=safe_detail(safe_user_message),
            retryable=retryable,
            timestamp=time.time(),
            plan_id=plan_id,
            technical_message=safe_detail(technical_message),
            plan_revision=plan_revision, attempt_number=attempt_number,
            step_id=step_id, operation_type=operation_type, artifact_type=artifact_type,
            capability_id=capability_id, original_plan_ref=original_plan_ref,
            verification_ref=verification_ref,
            caused_during_explanation_of=caused_during_explanation_of,
            validation_detail={key: [safe_detail(v,240) for v in value[:8]] if isinstance(value,list)
                else value if type(value) in {bool,int} else safe_detail(value,240)
                for key,value in (validation_detail or {}).items() if key in {
                    'validator','validation_stage','reason_code','failed_constraint','repair_attempted','repair_result','errors'}},
        )


# One record type; legacy imports retain their API during the staging migration.
LastToolFailure = FailureRecord


@dataclass
class ActiveFailureTopic:
    failure_id: str
    task_id: str
    root_failure_id: str
    stage: str
    safe_summary: str
    detailed_evidence_ref: str
    explanation_depth: int = 0
    topic_status: str = "ACTIVE"
    superseded_by_user_action: str = ""

    @classmethod
    def from_failure(cls, failure):
        return cls(failure.failure_id, failure.task_id, failure.failure_id,
            failure.stage, failure.safe_user_message, failure.failure_id)


@dataclass(frozen=True)
class AttemptOutcome:
    task_id: str
    plan_revision: int
    attempt_number: int
    status: str
    original_plan_ref: str
    failure_id: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class RemediationAssessment:
    failure_id: str
    remediation_possible: str
    evidence_basis: tuple[str, ...]
    retry_possible: bool
    corrective_action_available: str
    safe_reason: str

    @classmethod
    def assess(cls, failure, *, retry_available=False):
        codes = {failure.error_code.split(':')[0],
            failure.validation_detail.get('failed_constraint', '')}
        for error in failure.validation_detail.get('errors', ()):
            codes.update(str(error).split(':')[:2])
        possible = 'UNKNOWN'
        reason = 'No puedo determinar con la evidencia registrada si se puede corregir.'
        if failure.error_code == 'ORACLE_UNVERIFIED':
            reason = ('Para confirmar el resultado hace falta una referencia independiente validada. '
                'No tengo evidencia de que el programa este mal; repetirlo no aporta por si solo esa referencia.')
        elif failure.stage == 'PLAN_VALIDATION' and codes & {
                'dependency_not_preceding','input_binding','final_output',
                'requested_execution_missing','requested_test_suite_missing','verification_required'}:
            possible = 'POSSIBLE'
            reason = ('El plan puede volver a prepararse con las dependencias, resultados y comprobaciones '
                'que faltaban, y tendria que superar otra vez la validacion local. No es una garantia de exito.')
        elif codes & {'UNKNOWN_CAPABILITY','UNSUPPORTED_OPERATION'}:
            reason = ('La propuesta no tenia una herramienta compatible y habilitada. '
                'No puedo asegurar una correccion con las capacidades que constan en ese fallo.')
        retry = failure.retryable and retry_available
        return cls(failure.failure_id, possible,
            (failure.stage, failure.error_code,
                'REPAIR_ATTEMPTED' if failure.validation_detail.get('repair_attempted') else 'NO_REPAIR_EVIDENCE'),
            retry, 'UNKNOWN', reason + (
                ' Puedo reintentar la misma tarea si me lo pides, sujeto a los permisos y comprobaciones habituales.'
                if retry else ' No hay un reintento disponible para esta tarea en su estado actual.'))


class FailureResponseRenderer:
    """Render only recorded failure facts; never ask the brain to invent a cause."""

    _STAGE_LABELS = {
        "PLANNING": "la planificación de la acción",
        "CONTENT_SPECIFICATION": "la preparación de la especificación del contenido",
        "SELF_CONTEXT": "la preparación del contexto de identidad permitido",
        "CONTENT_GENERATION": "la generación del contenido",
        "CONTENT_VALIDATION": "la validación del contenido generado",
        "DOCUMENT_SPEC": "la estructuración del documento",
        "RENDER": "la creación temporal del documento",
        "VERIFY": "la verificación del resultado",
        "EXECUTION": "la ejecución de la herramienta",
    }
    _VERIFY_LABELS = {
        "CODE": "la verificación del programa",
        "DOCUMENT": "la verificación del documento",
        "ARCHIVE": "la verificación del archivo ZIP",
        "CONVERSION": "la verificación de la conversión",
        "FILESYSTEM": "la verificación del cambio en el archivo",
    }

    @classmethod
    def render(cls, failure: LastToolFailure, *, depth: int = 0) -> str:
        if failure.stage == 'INTERACTION_RESOLUTION':
            reasons = {
                'operation_intent_conflict':'La intencion solicitada y la operacion propuesta no coincidieron.',
                'requested_effect_conflict':'Los cambios propuestos no eran coherentes con la operacion solicitada.',
                'current_operation_not_failure_question':'Se mezclo una nueva accion con una pregunta sobre un fallo anterior.',
                'speech_act_conflict':'Se mezclo una orden de actuar con una consulta de informacion.',
                'read_existing_must_be_non_mutating':'Una peticion de lectura incluia cambios que no estaban autorizados.',
                'create_new_cannot_require_existing_target':'Se intento tratar una creacion nueva como cambio de un archivo anterior.',
                'interaction_schema':'La interpretacion no tenia la estructura requerida para continuar con seguridad.',
                'interaction_array_invalid':'Una referencia no se represento de la forma requerida.',
                'classification_unavailable':'No tengo registrado el detalle exacto que impidio interpretar la solicitud.',
            }
            recorded = [failure.validation_detail.get('failed_constraint',''), *failure.validation_detail.get('errors',())]
            details = list(dict.fromkeys(reasons[part] for error in recorded for part in str(error).split(':') if part in reasons))
            text = 'No pude interpretar la solicitud de forma coherente y no hice cambios.'
            if depth:
                text += ' '+(' '.join(details[:2]) if details else 'No tengo registrado el detalle exacto de la contradiccion.')
            if depth > 1:
                text += ' Esto ocurrio antes de ejecutar herramientas: no demuestra que tu archivo este mal ni que la tarea sea imposible.'
                if failure.validation_detail.get('repair_attempted'):
                    text += ' La unica correccion automatica permitida tampoco produjo una interpretacion valida.'
            return text
        # Only registered error codes are translated. Arbitrary exception text
        # remains private and is never treated as a trustworthy explanation.
        execution_causes = {
            'ValueError:python_import_not_allowed':
                'El codigo generado usa una importacion que la politica actual no permite. No puedo quitar esa restriccion para ejecutarlo.',
        }
        if failure.stage == 'EXECUTION' and failure.technical_message in execution_causes:
            text = execution_causes[failure.technical_message]
            if depth > 1:
                text += ' El rechazo ocurrio al validar el codigo antes de ejecutarlo; no demuestra que el programa tenga un error de calculo.'
            return text
        if failure.error_code == 'ORACLE_UNVERIFIED' and depth:
            text = 'No tengo una referencia independiente y confiable para comprobar las respuestas del programa.'
            if depth > 1:
                text += ' Los ejemplos propuestos por el modelo no bastan para certificarlo; no demuestran que este mal ni que sea correcto.'
            return text
        if failure.stage == 'PLAN_VALIDATION':
            reasons = {
                'final_output': 'La secuencia propuesta no identificaba un archivo final producido por sus pasos.',
                'dependency_not_preceding': 'Un paso necesitaba un resultado que ningun paso anterior habia producido.',
                'verification_required': 'La secuencia no incluia la comprobacion solicitada.',
                'input_binding': 'No quedo identificado de forma unica el archivo de entrada.',
                'CONTEXT_ARTIFACT_BINDING_REQUIRED': 'La propuesta no conservaba el archivo seleccionado para esta modificacion.',
                'code_generation_tests_required':'Se pidio probar el programa, pero faltaban las pruebas.',
                'test_spec_timeout_invalid':'Una prueba tenia un tiempo maximo fuera de los limites permitidos.',
                'UNKNOWN_CAPABILITY':'La propuesta necesitaba una herramienta que no esta disponible.',
                'UNSUPPORTED_OPERATION':'No hay una herramienta habilitada que cubra esa operacion.',
                'input_type':'Un paso necesitaba un tipo de archivo distinto del que recibia.',
                'source_filename_invalid':'El nombre propuesto no era un nombre de archivo seguro.',
                'requested_execution_missing':'Pediste ejecutar el programa, pero la secuencia propuesta no incluia ese paso.',
                'requested_test_suite_missing':'Pediste probar el programa, pero la secuencia propuesta no incluia las pruebas necesarias.',
                'primary_goal_requires_content_generation':'La propuesta no reconocia que tu peticion necesitaba producir contenido nuevo o modificado.',
            }
            recorded = [failure.validation_detail.get('failed_constraint', ''),
                failure.error_code, *failure.validation_detail.get('errors', ())]
            details = []
            for value in recorded:
                for code in str(value).split(':'):
                    if code in reasons and reasons[code] not in details:
                        details.append(reasons[code])
            detail = ' '.join(details[:3]) or 'No tengo mas detalle explicable registrado sobre ese requisito.'
            text = 'No pude preparar una secuencia de acciones valida para esa peticion.'
            if depth:
                text += ' ' + detail
            if depth > 1:
                text += ' Por eso detuve la tarea antes de ejecutar cambios; no significa que tu peticion sea imposible.'
            if depth > 2:
                text += (' Se puede volver a intentar la misma tarea con los permisos habituales.' if failure.retryable else
                    ' No hay un reintento automatico habilitado para este fallo. Se puede preparar otra propuesta que cumpla esos requisitos, pero no puedo garantizar que quede resuelto.')
            return text
        stage = cls._STAGE_LABELS.get(failure.stage, "la ejecución de la acción")
        if failure.stage == "VERIFY":
            stage = cls._VERIFY_LABELS.get(failure.artifact_type, stage)
        detail = failure.safe_user_message.strip()
        if detail:
            return f"Falló durante {stage}: {detail}"
        return (
            f"Falló durante {stage}, pero no tengo una causa más específica "
            "registrada."
        )


def is_failure_explanation_request(text: str) -> bool:
    normalized = _normalize(text).strip(" .,!¡¿?")
    return bool(re.search(
        r"^(?:sabes\s+)?por\s+que(?:\s+(?:fallo|no\s+pudo|no\s+funciono))?$|"
        r"^(?:que|cual)\s+(?:paso|fue\s+el\s+error|fue\s+la\s+causa)$|"
        r"^que\s+paso\s+con\s+.+$|"
        r"^por\s+que\s+no\s+(?:lo\s+)?(?:hiciste|creaste|funciono)$",
        normalized,
    ))


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value).casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))
