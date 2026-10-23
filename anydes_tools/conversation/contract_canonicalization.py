"""Explicit representation equivalences, not semantic repair or authority."""
import math


class ConfidenceValue:
    LABELS = {"LOW": 0.30, "MEDIUM": 0.65, "HIGH": 0.85}

    @classmethod
    def canonical(cls, value):
        if isinstance(value, str) and value.upper() in cls.LABELS:
            return cls.LABELS[value.upper()]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("confidence_representation_invalid")
        return float(value)


class ContractCanonicalizer:
    @staticmethod
    def interaction(payload):
        if not isinstance(payload, dict):
            raise ValueError("interaction_object_required")
        result = dict(payload)
        result["confidence"] = ConfidenceValue.canonical(payload.get("confidence", 0.0))
        return result
