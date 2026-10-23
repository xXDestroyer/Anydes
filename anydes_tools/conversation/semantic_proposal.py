"""Semantic brain vocabulary: no paths, tool IDs or execution authority."""
from dataclasses import dataclass
import json
from pathlib import Path
from uuid import uuid4
from .plan_protocol import PlanProtocolError, parse_plan
from ..schema_validation import Draft202012Validator, _bounded

KINDS = ["GENERATE_CONTENT", "GENERATE_CODE", "CREATE_DOCUMENT", "CONVERT_DOCUMENT",
    "TEST_CODE", "CREATE_ARCHIVE", "CREATE_FILE", "EDIT_ARTIFACT", "MOVE_ARTIFACT",
    "RENAME_ARTIFACT", "DELETE_ARTIFACT"]
NAMES = {"type":"array", "items":{"type":"string","minLength":1,"maxLength":100}, "maxItems":16}
OUTPUT = {"type":"object", "required":["name","artifact_type","format"], "properties":{
    "name":{"type":"string","minLength":1,"maxLength":100},
    "artifact_type":{"enum":["CONTENT","CODE","DOCUMENT","ARCHIVE"]},
    "format":{"enum":["text","py","cpp","txt","md","docx","pdf","zip"]},
    "filename":{"type":"string","maxLength":180}}, "additionalProperties":False}
SEMANTIC_SCHEMA = {"type":"object", "required":["goal","operations","requested_final_outputs"],
    "properties":{"goal":{"type":"string","minLength":1,"maxLength":2000},
        "constraints":{"type":"array","items":{"type":"string","maxLength":500},"maxItems":16},
        "requested_final_outputs":NAMES,
        "operations":{"type":"array","minItems":1,"maxItems":8,"items":{
            "type":"object","required":["operation_id","operation_kind","inputs","outputs","specification"],
            "properties":{"operation_id":{"type":"string","minLength":1,"maxLength":100},
                "operation_kind":{"enum":KINDS}, "inputs":NAMES,
                "outputs":{"type":"array","items":OUTPUT,"maxItems":16},
                "specification":{"type":"object"}}, "additionalProperties":False}}}, "additionalProperties":False}
FORBIDDEN = {"capability","capability_id","tool_id","artifact_id","path","source_path",
    "destination","verifier_type","permission_requirements","risk_class", "capabilities",
    "required_capabilities", "permissions", "scopes", "risk_level", "producer_step_id",
    "absolute_canonical_path", "task_id"}

def validate_semantic(value):
    errors = list(Draft202012Validator(SEMANTIC_SCHEMA).iter_errors(value))
    if not _bounded(value) or errors:
        raise PlanProtocolError("SEMANTIC_SCHEMA:" + (errors[0].validator if errors else "bounds"))
    def inspect(obj):
        if isinstance(obj, dict):
            if FORBIDDEN.intersection(obj):
                raise PlanProtocolError("SEMANTIC_FORBIDDEN_OPERATIONAL_FIELD")
            for item in obj.values(): inspect(item)
        elif isinstance(obj, list):
            for item in obj: inspect(item)
    inspect(value)
    return value

def parse_semantic(text):
    # Reuse the strict JSON parser while supplying the semantic schema.
    return parse_plan(text, validator=validate_semantic)

@dataclass(frozen=True)
class SemanticProposal:
    payload: dict
    task_id: str

    @classmethod
    def from_dict(cls, value):
        return cls(validate_semantic(value), "task:" + uuid4().hex)

    def to_dict(self):
        return self.payload

def catalog():
    return json.loads((Path(__file__).parents[1] / "registry" / "semantic_operations.json").read_text())

def semantic_vocabulary(registry):
    rows = [{"operation_kind":"GENERATE_CONTENT","inputs":[],"output_type":"CONTENT","formats":["text"]}]
    for row in catalog()["contracts"]:
        if registry.find_by_capability(row["capability"]) and all(
                registry.find_by_capability(cap) for cap in row.get("requires", [])):
            item = {k:v for k,v in row.items() if k not in {"capability", "requires"}}
            if item not in rows: rows.append(item)
    return rows


def operation_constraint(resolution):
    """Host authorization constrains planning; it does not grant permissions."""
    operation = resolution['operation_intent']
    return {'requested_operation':operation,
        'requires_execution':resolution['requires_execution'],
        'required_effects':resolution['effect_intent'],
        'verification_requirement':resolution['verification_requirement'],
        'allowed_semantic_operations':['TEST_CODE'] if operation == 'EXECUTE_EXISTING' else None,
        'external_input':'selected_artifact' if operation == 'EXECUTE_EXISTING' else None}


def operation_mismatches(proposal, constraint):
    operations = proposal.payload['operations']
    runs = [op for op in operations if op['operation_kind'] == 'TEST_CODE']
    errors = []
    if constraint['requires_execution'] and not runs:
        errors.append('SEMANTIC_OPERATION_MISMATCH:execution_required')
    if constraint['requested_operation'] == 'EXECUTE_EXISTING':
        if any(op['operation_kind'] != 'TEST_CODE' for op in operations):
            errors.append('SEMANTIC_OPERATION_MISMATCH:existing_execution_only')
        if any(op['inputs'] != ['selected_artifact'] or op['outputs'] for op in runs):
            errors.append('SEMANTIC_OPERATION_MISMATCH:exact_existing_input_required')
        if proposal.payload['requested_final_outputs']:
            errors.append('SEMANTIC_OPERATION_MISMATCH:execution_has_no_new_artifact')
    if constraint['requested_operation'] == 'CREATE_NEW' and constraint['requires_execution']:
        created = {out['name'] for op in operations if op['operation_kind'] == 'GENERATE_CODE'
            for out in op['outputs'] if out['artifact_type'] == 'CODE'}
        if not created or any(not op['inputs'] or not set(op['inputs']) <= created for op in runs):
            errors.append('SEMANTIC_OPERATION_MISMATCH:execute_exact_generated_code')
    return tuple(errors)
