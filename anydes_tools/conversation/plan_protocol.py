"""Provider-independent, bounded plan protocol. No execution authority."""
import hashlib
import json
import re

from ..schema_validation import Draft202012Validator


def string_array():
    return {"type": "array", "items": {"type": "string", "maxLength": 200}, "maxItems": 32}


OUTPUT = {"type": "object", "required": ["binding", "artifact_type", "format"],
    "properties": {"binding": {"type": "string", "minLength": 1, "maxLength": 200},
        "artifact_type": {"enum": ["CODE", "DOCUMENT", "ARCHIVE"]},
        "format": {"type": "string", "maxLength": 20}}, "additionalProperties": False}
STEP = {"type": "object", "required": ["step_id", "capability", "purpose", "arguments",
    "depends_on", "produces_artifacts", "consumes_artifacts"], "properties": {
    "step_id": {"type": "string", "minLength": 1, "maxLength": 100},
    "step_type": {"enum": ["CAPABILITY", "CONVERSION"]},
    "capability": {"type": "string", "minLength": 1, "maxLength": 100},
    "purpose": {"type": "string", "maxLength": 1000}, "arguments": {"type": "object"},
    **{k: string_array() for k in ("depends_on", "produces_artifacts", "consumes_artifacts",
        "produces_values", "consumes_values")}}, "additionalProperties": False}
PLAN_SCHEMA = {"type": "object", "required": ["goal", "task_type", "inputs", "steps",
    "required_capabilities", "desired_outputs", "requested_final_outputs", "artifact_requirements",
    "needs_tools", "needs_content_generation"], "properties": {
    "task_id": {"type": "string", "maxLength": 100},
    "goal": {"type": "string", "minLength": 1, "maxLength": 2000},
    "task_type": {"enum": ["GENERATE_CODE", "MODIFY_CODE", "GENERATE_DOCUMENT", "TOOL_COMPOSITION",
        "REASONING_ONLY", "GENERATE_CONVERSATIONAL_CONTENT", "MISSING_CAPABILITY", "UNSUPPORTED", "FORBIDDEN"]},
    "inputs": {"type": "object"}, "artifact_requirements": {"type": "object"},
    "desired_outputs": {"type": "array", "items": OUTPUT, "maxItems": 16},
    "requested_final_outputs": {"type": "array", "items": OUTPUT, "maxItems": 16},
    "steps": {"type": "array", "items": STEP, "maxItems": 8},
    **{k: string_array() for k in ("constraints", "required_capabilities", "ambiguities", "success_criteria")},
    **{k: {"type": "boolean"} for k in ("needs_content_generation", "needs_reasoning", "needs_tools")},
    "risk": {"enum": ["none", "low", "medium", "high"]}}, "additionalProperties": False}


class PlanProtocolError(ValueError):
    def __init__(self, detail):
        self.detail = detail[:300]
        super().__init__("PLAN_PROTOCOL_ERROR:" + self.detail)


def budget(request):
    return request.setdefault("plan_repair_budget", {"used": False, "reason": ""})


def consume_repair(request, reason):
    state = budget(request)
    if state["used"]:
        raise PlanProtocolError("REPAIR_BUDGET_EXHAUSTED")
    state.update(used=True, reason=reason)


def validate_payload(payload):
    from ..schema_validation import _bounded
    if not _bounded(payload):
        raise PlanProtocolError("SCHEMA:BOUNDS_EXCEEDED")
    errors = sorted(Draft202012Validator(PLAN_SCHEMA).iter_errors(payload), key=lambda e: str(list(e.path)))
    if errors:
        # Never include the offending value or validator's full message.
        raise PlanProtocolError("SCHEMA:" + ";".join(
            "/".join(str(p) for p in e.path) + ":" + e.validator for e in errors[:6]))
    return payload


def parse_plan(text, *, validator=validate_payload):
    candidate = str(text or "").strip()
    if len(candidate) > 64000:
        raise PlanProtocolError("RESPONSE_TOO_LARGE")
    if not candidate:
        raise PlanProtocolError("EMPTY_RESPONSE")
    # Exactly one JSON object. No brace search, fences, preface or trailing prose.
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise PlanProtocolError("DUPLICATE_KEY")
            result[key] = value
        return result
    try:
        value = json.loads(candidate, object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(PlanProtocolError("NONFINITE_NUMBER")))
    except json.JSONDecodeError as exc:
        raise PlanProtocolError(f"JSON_SYNTAX:line={exc.lineno}:column={exc.colno}:offset={exc.pos}") from exc
    except RecursionError as exc:
        raise PlanProtocolError("JSON_DEPTH_EXCEEDED") from exc
    return validator(value)


def protocol_metadata(text):
    text = str(text or "")
    return {"characters": len(text), "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "starts_object": text.lstrip().startswith("{"), "fenced": text.lstrip().startswith("```")}


def sanitized_repair_response(text):
    text = str(text or "")[:12000]
    text = re.sub(r'(?i)(api[_-]?key|password|secret|token)(\s*["\s:=]+)[^,\n}]+', r'\1\2[REDACTED]', text)
    text = re.sub(r'(?i)\b(?:sk-|gsk_)[a-z0-9_-]+', '[REDACTED]', text)
    text = re.sub(r'[A-Za-z]:[\\/][^"\n]*', '[PATH]', text)
    return text
