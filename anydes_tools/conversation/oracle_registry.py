"""Host-owned reference implementations selected by typed semantics, never prompt text."""
from dataclasses import dataclass, replace
import math
from fractions import Fraction
import unicodedata
from ..schema_validation import Draft202012Validator


ORACLE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["oracle_id", "oracle_type", "supported_test_kind", "supported_input_types",
        "supported_output_types", "operation_semantics", "confidence_level", "deterministic", "version", "enabled"],
    "properties": {
        "oracle_id": {"type": "string", "minLength": 1, "maxLength": 120},
        "oracle_type": {"enum": ["NUMERIC_FORMULA", "REFERENCE_FUNCTION", "PROPERTY_CHECK", "STRUCTURED_INVARIANT"]},
        "supported_test_kind": {"enum": ["FUNCTIONAL", "ERROR_HANDLING"]},
        "supported_input_types": {"type": "array", "minItems": 1, "maxItems": 8, "items": {"enum": ["NUMBER", "TEXT", "JSON"]}},
        "supported_output_types": {"type": "array", "minItems": 1, "maxItems": 8, "items": {"enum": ["NUMBER", "TEXT", "JSON"]}},
        "operation_semantics": {"type": "array", "minItems": 1, "maxItems": 16, "items": {"type": "string", "maxLength": 80}},
        "confidence_level": {"enum": ["DETERMINISTIC", "REFERENCE", "PROPERTY"]},
        "deterministic": {"type": "boolean"}, "version": {"type": "integer", "minimum": 1},
        "enabled": {"type": "boolean"},
    },
}


DEFAULT_ORACLES = (
    {"oracle_id": "units.affine", "oracle_type": "NUMERIC_FORMULA", "supported_test_kind": "FUNCTIONAL",
     "supported_input_types": ["NUMBER"], "supported_output_types": ["NUMBER"],
     "operation_semantics": ["TEMPERATURE_CONVERSION"], "confidence_level": "DETERMINISTIC",
     "deterministic": True, "version": 1, "enabled": True},
    {"oracle_id": "text.character_count", "oracle_type": "REFERENCE_FUNCTION", "supported_test_kind": "FUNCTIONAL",
     "supported_input_types": ["TEXT"], "supported_output_types": ["NUMBER"],
     "operation_semantics": ["COUNT_CHARACTERS", "COUNT_VOWELS"], "confidence_level": "DETERMINISTIC",
     "deterministic": True, "version": 1, "enabled": True},
)


def affine_units(value, semantics):
    # Unit -> (scale, offset) in a common canonical unit; no generated expression is evaluated.
    units = {"C": (Fraction(1), Fraction(0)), "F": (Fraction(5, 9), Fraction(-160, 9)),
        "K": (Fraction(1), Fraction("-273.15"))}
    source, target = units[semantics["source_unit"]], units[semantics["target_unit"]]
    number = float(value.strip())
    if not math.isfinite(number) or abs(number) > 1e12:
        raise ValueError("invalid_numeric_input")
    return float((Fraction(value.strip())*source[0]+source[1]-target[1])/target[0])


def character_count(value, semantics):
    if len(value) > 10000:
        raise ValueError("input_limit")
    text = value.rstrip("\r\n")
    if "casefold" not in semantics:
        raise ValueError("normalization_unspecified")
    if semantics["casefold"]:
        text = text.casefold()
    if semantics["kind"] == "COUNT_VOWELS":
        if "include_accented" not in semantics:
            raise ValueError("accent_policy_unspecified")
        alphabet = "aeiou" + ("áéíóúü" if semantics["include_accented"] else "")
    else:
        alphabet = semantics["alphabet"]
        if not alphabet or len(alphabet) > 256:
            raise ValueError("alphabet_invalid")
    if semantics.get("strip_accents", False):
        text = "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))
    return sum(c in alphabet for c in text)


@dataclass(frozen=True)
class OracleSelection:
    definition: dict | None = None
    reason: str = "ORACLE_UNAVAILABLE"


class OracleRegistry:
    IMPLEMENTATIONS = {"units.affine": affine_units, "text.character_count": character_count}

    def __init__(self, definitions=None):
        self.definitions = tuple(dict(d) for d in (DEFAULT_ORACLES if definitions is None else definitions))
        for definition in self.definitions:
            Draft202012Validator(ORACLE_SCHEMA).validate(definition)

    def evaluate(self, selection, test):
        return self.IMPLEMENTATIONS[selection.definition["oracle_id"]](test.stdin, test.operation_semantics)


from .pipeline_trace import traced_stage


class OracleSelector:
    RANK = {"PROPERTY": 1, "REFERENCE": 2, "DETERMINISTIC": 3}

    def __init__(self, registry=None):
        self.registry = registry or OracleRegistry()

    @traced_stage("ORACLE_SELECTION")
    def select(self, test):
        requirement, semantics = test.oracle_requirement, test.operation_semantics
        if not semantics or not requirement:
            return OracleSelection(reason="ORACLE_SEMANTICS_MISSING")
        candidates = [d for d in self.registry.definitions if d["enabled"]
            and d["oracle_id"] in self.registry.IMPLEMENTATIONS
            and d["supported_test_kind"] == "FUNCTIONAL"
            and semantics.get("kind") in d["operation_semantics"]
            and requirement.get("input_type") in d["supported_input_types"]
            and requirement.get("output_type") in d["supported_output_types"]
            and self.RANK[d["confidence_level"]] >= self.RANK[requirement.get("minimum_confidence", "DETERMINISTIC")]]
        if not candidates:
            return OracleSelection()
        rank = max(self.RANK[d["confidence_level"]] for d in candidates)
        candidates = [d for d in candidates if self.RANK[d["confidence_level"]] == rank]
        # Version supersession is meaningful only within the same oracle lineage.
        if len({d["oracle_id"] for d in candidates}) != 1:
            return OracleSelection(reason="ORACLE_AMBIGUOUS")
        version = max(d["version"] for d in candidates)
        candidates = [d for d in candidates if d["version"] == version]
        return OracleSelection(candidates[0], "SELECTED") if len(candidates) == 1 else OracleSelection(reason="ORACLE_AMBIGUOUS")

    def resolve(self, test):
        selection = self.select(test)
        if not selection.definition:
            return selection, None
        try:
            expected = self.registry.evaluate(selection, test)
        except (ValueError, KeyError, TypeError, OverflowError):
            return OracleSelection(selection.definition, "INPUT_OR_SEMANTICS_UNVERIFIED"), None
        # Numerical references compare numerical values, not spelling of floats.
        mode = "FLOAT_TOLERANCE" if test.comparison_mode == "FLOAT_TOLERANCE" else "NUMERIC"
        return selection, replace(test, expected=expected, reference={}, comparison_mode=mode,
            tolerance=test.tolerance if mode == "FLOAT_TOLERANCE" else 0.0,
            oracle_source="DETERMINISTIC_REFERENCE")


