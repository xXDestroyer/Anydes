"""Source-linked claims interface, not a general factuality judge."""
from .verification_types import VerificationResult, VerificationStatus
from .failure_state import text_summary


def verify_grounding(claims, sources, *, required=False):
    if not required:
        return VerificationResult("CONTENT_GROUNDING", "CONTENT", VerificationStatus.SKIPPED,
            evidence={"scope": "not_requested"})
    # Only literal attributed quotations are demonstrable with this minimal interface.
    checks, evidence = {}, []
    for index, claim in enumerate(claims[:32]):
        quote, source_id = str(claim.get("quote", "")), str(claim.get("source_id", ""))
        checks[str(index)] = bool(quote and source_id in sources and quote in sources[source_id]
            and str(claim.get("claim", "")) == quote)
        evidence.append({"source_id": source_id, "claim_hash": text_summary(claim.get("claim", ""))})
    # Valid quotes do not prove absence of additional unsupported prose claims.
    return VerificationResult("CONTENT_GROUNDING", "CONTENT", VerificationStatus.INCONCLUSIVE,
        checks=checks, evidence={"claims": evidence, "coverage": "LITERAL_QUOTES_ONLY"},
        error_code="FACTUAL_COVERAGE_UNPROVEN", safe_message="No puedo demostrar el respaldo de todas las afirmaciones.")
