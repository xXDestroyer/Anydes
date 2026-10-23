"""Preserve declared transformations; lower them to existing oracle vocabulary locally."""
from dataclasses import dataclass, replace
from ..schema_validation import Draft202012Validator


OPERATION_DESCRIPTOR_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["semantic_kind", "input_type", "output_type", "parameters"],
    "properties": {"semantic_kind": {"enum": ["UNIT_CONVERSION", "CHARACTER_COUNT"]},
        "input_type": {"enum": ["NUMBER", "TEXT"]}, "output_type": {"const": "NUMBER"},
        "parameters": {"type": "object"}},
    "oneOf": [
        {"properties": {
            "semantic_kind": {"const": "UNIT_CONVERSION"}, "input_type": {"const": "NUMBER"},
            "parameters": {
                "type": "object", "additionalProperties": False,
                "required": ["source_unit", "target_unit"],
                "properties": {"source_unit": {"enum": ["C", "F", "K"]},
                    "target_unit": {"enum": ["C", "F", "K"]}}
            }
        }},
        {"properties": {
            "semantic_kind": {"const": "CHARACTER_COUNT"}, "input_type": {"const": "TEXT"},
            "parameters": {
                "type": "object", "additionalProperties": False,
                "required": ["alphabet", "casefold", "strip_accents"],
                "properties": {"alphabet": {"type": "string", "minLength": 1, "maxLength": 256},
                    "casefold": {"type": "boolean"}, "strip_accents": {"type": "boolean"}}
            }
        }}]
}

# Local declarative profiles, not model-selected oracle IDs or prompt matching.
VERIFICATION_PROFILES = (
    {"semantic_kind": "UNIT_CONVERSION", "input_type": "NUMBER", "output_type": "NUMBER",
        "oracle_semantic_kind": "TEMPERATURE_CONVERSION", "parameters": ("source_unit", "target_unit")},
    {"semantic_kind": "CHARACTER_COUNT", "input_type": "TEXT", "output_type": "NUMBER",
        "oracle_semantic_kind": "COUNT_CHARACTERS", "parameters": ("alphabet", "casefold", "strip_accents")},
)


@dataclass(frozen=True)
class OperationSemantics:
    semantic_kind: str
    input_type: str
    output_type: str
    parameters: dict

    @classmethod
    def from_dict(cls, value):
        if not Draft202012Validator(OPERATION_DESCRIPTOR_SCHEMA).is_valid(value):
            raise ValueError("operation_descriptor_invalid")
        return cls(value["semantic_kind"], value["input_type"], value["output_type"], dict(value["parameters"]))

    def to_dict(self):
        return {"semantic_kind": self.semantic_kind, "input_type": self.input_type,
            "output_type": self.output_type, "parameters": dict(self.parameters)}

    def bind(self, test):
        if test.invalid_input_behavior != "UNSPECIFIED":
            return test  # Error-handling is not a functional transformation oracle.
        profiles = [p for p in VERIFICATION_PROFILES if all(p[k] == getattr(self, k)
            for k in ("semantic_kind", "input_type", "output_type"))]
        if len(profiles) != 1:
            return replace(test, operation_semantics={}, oracle_requirement={})
        p = profiles[0]
        return replace(test, operation_semantics={"kind": p["oracle_semantic_kind"],
                **{key: self.parameters[key] for key in p["parameters"]}},
            oracle_requirement={"input_type": self.input_type, "output_type": self.output_type,
                "minimum_confidence": "DETERMINISTIC"})


def canonical_descriptor(value):
    return {} if value is None or value == {} else OperationSemantics.from_dict(value).to_dict()
