"""Pinned JSON Schema implementation used only by staging Tool Architecture."""
from pathlib import Path
import math
import sys

_VENDOR = Path(__file__).parent / "dependencies" / "jsonschema_b1"
if str(_VENDOR) not in sys.path:
    sys.path.append(str(_VENDOR))
from jsonschema import Draft202012Validator
from referencing import Registry


def _bounded(value, depth=0):
    if depth > 30:
        return False
    if isinstance(value, float) and not math.isfinite(value):
        return False
    if isinstance(value, dict):
        return len(value) <= 200 and all(isinstance(k, str) and _bounded(v, depth + 1) for k, v in value.items())
    if isinstance(value, list):
        return len(value) <= 5000 and all(_bounded(v, depth + 1) for v in value)
    return value is None or isinstance(value, (str, bool, int, float))


def check_schema(schema):
    Draft202012Validator.check_schema(schema)
    def refs(value):
        if isinstance(value, dict):
            if "$ref" in value and not str(value["$ref"]).startswith("#"):
                raise ValueError("DENIED_REMOTE_SCHEMA_REFERENCE")
            for item in value.values(): refs(item)
        elif isinstance(value, list):
            for item in value: refs(item)
    refs(schema)


def validate(value, schema):
    if not _bounded(value):
        return False
    try:
        check_schema(schema)
        return Draft202012Validator(schema, registry=Registry()).is_valid(value)
    except Exception:
        return False
