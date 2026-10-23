"""Uniform verification evidence and typed tests; no execution authority."""
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
from ..schema_validation import Draft202012Validator, _bounded

TEST_SPEC_SCHEMA = {"type":"object", "required":["test_id"], "additionalProperties":False,
    "properties":{"test_id":{"type":"string","minLength":1,"maxLength":120},
        "stdin":{"type":"string","maxLength":10000}, "input":{"type":"string","maxLength":10000},
        "expected":{}, "expected_stdout_contains":{"type":["array","string"],
            "maxItems":16,"items":{"type":"string","maxLength":10000},"maxLength":10000},
        "oracle_source":{"type":"string","maxLength":100},
        "expected_hint":{},
        "oracle_requirement":{"type":"object","additionalProperties":False,"properties":{
            "input_type":{"enum":["NUMBER","TEXT","JSON"]},"output_type":{"enum":["NUMBER","TEXT","JSON"]},
            "minimum_confidence":{"enum":["DETERMINISTIC","REFERENCE","PROPERTY"]}}},
        "operation_semantics":{"type":"object","additionalProperties":False,"properties":{
            "kind":{"type":"string","maxLength":80},"source_unit":{"type":"string","maxLength":16},
            "target_unit":{"type":"string","maxLength":16},"alphabet":{"type":"string","maxLength":256},
            "casefold":{"type":"boolean"},"strip_accents":{"type":"boolean"},
            "include_accented":{"type":"boolean"}}},
        "invalid_input_behavior":{"enum":["UNSPECIFIED","NONZERO_EXIT","ERROR_MESSAGE","REPROMPT"]},
        "comparison_mode":{"enum":["EXACT","NUMERIC","FLOAT_TOLERANCE","STRUCTURED","PROPERTY","CONTAINS"]},
        "tolerance":{"type":"number","minimum":0,"maximum":100},
        "timeout":{"type":"number","exclusiveMinimum":0,"maximum":30},
        "expected_exit_code":{"type":"integer","minimum":-2147483648,"maximum":2147483647},
        "reference":{"type":"object","additionalProperties":False,"properties":{
            "kind":{"enum":["EXPRESSION","COUNT_CHARACTERS"]},
            "expression":{"type":"string","maxLength":256},"variables":{"type":"object","maxProperties":16,
                "additionalProperties":{"type":"integer","minimum":0,"maximum":32}},
            "alphabet":{"type":"string","maxLength":256}, "casefold":{"type":"boolean"}, "strip_accents":{"type":"boolean"}}}}}

_test_spec_diagnostic_context = ContextVar('test_spec_diagnostic_context', default={})


@contextmanager
def test_spec_diagnostics(record=None, **context):
    token = _test_spec_diagnostic_context.set({'record':record, **context})
    try:
        yield
    finally:
        _test_spec_diagnostic_context.reset(token)


class TestSpecSchemaError(ValueError):
    def __init__(self, candidate, errors):
        def schema_fields(schema):
            names=set()
            if isinstance(schema,dict):
                names.update(schema.get('properties',{}))
                for item in schema.values():
                    names.update(schema_fields(item))
            elif isinstance(schema,list):
                for item in schema:
                    names.update(schema_fields(item))
            return names
        known_fields=schema_fields(TEST_SPEC_SCHEMA)
        def safe_field(key):
            return key if isinstance(key,int) or key in known_fields else (
                'field_sha256:'+hashlib.sha256(str(key).encode()).hexdigest())
        def shape(value):
            if isinstance(value, dict):
                return {safe_field(key):shape(item)
                    for key,item in value.items()}
            if isinstance(value, list):
                return [shape(item) for item in value[:16]]
            if isinstance(value, str):
                return {'type':'string','length':len(value),
                    'sha256':hashlib.sha256(value.encode()).hexdigest()}
            return {'type':type(value).__name__}
        fields = []
        for error in errors[:8]:
            fields.append({'field_path':[safe_field(k) for k in error.absolute_path],
                'constraint':error.validator, 'expected':error.validator_value,
                'received_type':type(error.instance).__name__,
                'received_value_class':'unrecognized_enum' if error.validator=='enum' else 'schema_mismatch'})
        self.diagnostic = {'error_code':'TEST_SPEC_SCHEMA_INVALID','schema_version':'TEST_SPEC_V1',
            'schema_sha256':hashlib.sha256(json.dumps(TEST_SPEC_SCHEMA,sort_keys=True).encode()).hexdigest(),
            'candidate':shape(candidate),'errors':fields}
        context = _test_spec_diagnostic_context.get()
        self.diagnostic.update({k:v for k,v in context.items() if k!='record'})
        record = context.get('record')
        if record:
            record('test_spec_schema_rejected', **self.diagnostic)
        super().__init__('test_spec_schema_invalid')


class VerificationStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"
    SKIPPED = "SKIPPED"


class VerificationRequirement(str, Enum):
    NONE = "NONE"
    BASIC = "BASIC"
    EXECUTION = "EXECUTION"
    TEST_SUITE = "TEST_SUITE"
    STRUCTURAL = "STRUCTURAL"
    CONVERSION_PRESERVATION = "CONVERSION_PRESERVATION"


def code_verification_requirement(specification):
    explicit = getattr(specification, "verification_requirement", None)
    if explicit is not None:
        return VerificationRequirement(explicit)
    # Compatibility for pre-E.2 specs: preserve the existing execution safety policy.
    return VerificationRequirement.TEST_SUITE if specification.execution_required else VerificationRequirement.BASIC


@dataclass(frozen=True)
class VerificationResult:
    verifier_type: str
    artifact_type: str
    status: VerificationStatus
    checks: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)
    artifact_id: str = ""
    failed_check: str = ""
    error_code: str = ""
    safe_message: str = ""
    technical_detail: str = ""
    verification_id: str = field(default_factory=lambda: "verification:" + uuid4().hex)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self):
        return {**asdict(self), "status": self.status.value}

    def legacy(self):
        return {"status": "VERIFIED" if self.status == VerificationStatus.PASS else "VERIFY_FAILED",
            "verification_result": self.to_dict()}


@dataclass(frozen=True)
class TestSpec:
    test_id: str
    stdin: str = ""
    expected_stdout_contains: tuple[str, ...] = ()  # Deprecated input spelling, EXACT semantics.
    expected: Any = None
    oracle_source: str = "LOCAL_DECLARATION"
    comparison_mode: str = "EXACT"
    tolerance: float = 0.0
    timeout: float = 8.0
    reference: dict = field(default_factory=dict)
    expected_exit_code: int = 0
    oracle_requirement: dict = field(default_factory=dict)
    operation_semantics: dict = field(default_factory=dict)
    expected_hint: Any = None
    invalid_input_behavior: str = "UNSPECIFIED"

    def __post_init__(self):
        if self.expected is None and len(self.expected_stdout_contains) == 1:
            object.__setattr__(self, "expected", self.expected_stdout_contains[0])

    @classmethod
    def from_dict(cls, value):
        def json_shape(item, depth=0):
            if depth > 30:
                raise ValueError("test_spec_depth")
            if isinstance(item, (list, tuple)):
                if len(item) > 5000:
                    raise ValueError("test_spec_size")
                return [json_shape(v, depth+1) for v in item]
            if isinstance(item, dict):
                if len(item) > 200:
                    raise ValueError("test_spec_size")
                return {k: json_shape(v, depth+1) for k,v in item.items()}
            if isinstance(item, str) and len(item) > 20000:
                raise ValueError("test_spec_size")
            return item
        value = json_shape(value)  # Internal dataclass tuples are JSON arrays, not a looser wire schema.
        errors = list(Draft202012Validator(TEST_SPEC_SCHEMA).iter_errors(value))
        if not _bounded(value) or errors:
            raise TestSpecSchemaError(value, errors)
        old = value.get("expected_stdout_contains") or ()
        if isinstance(old, str):
            old = (old,)
        if len(old) > 16:
            raise ValueError("test_expectation_limit")
        expected = value.get("expected", old[0] if len(old) == 1 else list(old) if old else None)
        mode = str(value.get("comparison_mode", "EXACT")).upper()
        if mode not in {"EXACT", "NUMERIC", "FLOAT_TOLERANCE", "STRUCTURED", "PROPERTY", "CONTAINS"}:
            raise ValueError("test_comparison_mode")
        timeout, tolerance = float(value.get("timeout", 8)), float(value.get("tolerance", 0))
        if not 0 < timeout <= 30 or not 0 <= tolerance <= 100:
            raise ValueError("test_limits")
        return cls(str(value.get("test_id", ""))[:120], str(value.get("stdin", value.get("input", "")))[:10000],
            tuple(str(v) for v in old), expected, str(value.get("oracle_source", "LOCAL_DECLARATION")),
            mode, tolerance, timeout, dict(value.get("reference") or {}), int(value.get("expected_exit_code", 0)),
            dict(value.get("oracle_requirement") or {}), dict(value.get("operation_semantics") or {}),
            value.get("expected_hint"), str(value.get("invalid_input_behavior", "UNSPECIFIED")))


def build_runtime_tests(tests, *, specification=None, plan=None, semantic_cases=False,
                        operation_descriptor=None):
    """One operational binding point. Parsing alone never grants oracle authority."""
    from .operation_semantics import canonical_descriptor, OperationSemantics
    if semantic_cases and (not isinstance(tests,(list,tuple)) or len(tests)>8):
        raise ValueError('semantic_test_invalid:bounded_case_array_required')
    descriptors = []
    if operation_descriptor:
        descriptors.append(operation_descriptor)
    if specification is not None:
        descriptors.append(getattr(specification, "operation_descriptor", {}))
    if plan is not None:
        descriptors.append(plan.typed_inputs.get("operation_descriptor", {}))
        descriptors.extend(s.operation_descriptor for s in plan.steps if s.verifier_type == "CODE")
    declared = [canonical_descriptor(d) for d in descriptors if d]
    if declared and any(d != declared[0] for d in declared):
        raise ValueError("verification_descriptor_conflict")
    semantics = OperationSemantics.from_dict(declared[0]) if declared else None
    result = []
    for index, raw in enumerate(tests):
        if semantic_cases:
            if not isinstance(raw, dict):
                raise ValueError('semantic_test_invalid:case_object_required')
            semantic_fields = {'case_id','test_id','stdin','input','expected_hint','expected','expected_stdout_contains'}
            internal_fields = set(TEST_SPEC_SCHEMA['properties']) - semantic_fields
            internal_fields.update({'schema_version','oracle_id','verifier_id'})
            if set(raw) - semantic_fields - internal_fields:
                raise ValueError('semantic_test_invalid:unsupported_field:allowed=case_id,stdin,expected_hint')
            if 'stdin' not in raw and 'input' not in raw:
                raise ValueError('semantic_test_invalid:missing_input:required=stdin')
            if 'stdin' in raw and 'input' in raw and raw['stdin'] != raw['input']:
                raise ValueError('semantic_test_invalid:conflicting_input')
            stdin = raw.get('stdin',raw.get('input'))
            if not isinstance(stdin,str) or len(stdin)>10000:
                raise ValueError('semantic_test_invalid:stdin_requires_bounded_text')
            # The model supplies cases, not an internal schema or oracle authority.
            # Even legacy expected output is only a hint; local semantics choose the oracle.
            legacy_expected = raw.get('expected')
            old = raw.get('expected_stdout_contains')
            if legacy_expected is None and isinstance(old,(list,tuple)) and len(old)==1:
                legacy_expected = old[0]
            # Legacy expectation data still needs the unchanged trusted_oracle gate.
            # A hint is never promoted into that field or granted trust here.
            test = TestSpec.from_dict({'test_id':'case_'+str(index+1), 'stdin':stdin,
                'expected':legacy_expected,
                'expected_hint':raw.get('expected_hint',raw.get('expected',raw.get('expected_stdout_contains'))),
                'oracle_source':'MODEL_GENERATED'})
        else:
            test = raw if isinstance(raw, TestSpec) else TestSpec.from_dict(raw)
        if semantics is not None:
            test = semantics.bind(test)
            if test.operation_semantics and semantics.output_type == "NUMBER" and test.comparison_mode == "EXACT":
                test = replace(test, comparison_mode="NUMERIC")
        if semantic_cases:
            test = TestSpec.from_dict(asdict(test))
        result.append(test)
    context = _test_spec_diagnostic_context.get()
    record = context.get('record')
    if record:
        record('canonical_test_spec_built', builder='build_runtime_tests', schema_version='TEST_SPEC_V1',
            source_step=context.get('source_step',''), proposal_origin=context.get('proposal_origin',''),
            task_id=context.get('task_id',''), semantic_cases=semantic_cases,
            tests=[{'test_id':t.test_id,'input_sha256':hashlib.sha256(t.stdin.encode()).hexdigest(),
                'comparison_mode':t.comparison_mode,'has_semantics':bool(t.operation_semantics),
                'has_oracle_requirement':bool(t.oracle_requirement)} for t in result])
    return tuple(result)
