"""UNIT tests for diagnostics only, not human-runtime acceptance."""
import http.client
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from anydes_tools.tests.runtime_observation import Observer, safe
from anydes_tools.tests.run_full_runtime_phase_a import BudgetStop, RequestBudget, CASES


class ObservationTests(unittest.TestCase):
    def test_secret_never_hashed_or_rendered(self):
        value = safe({"api_key": "private-value", "headers": {"Authorization": "private-value"}})
        self.assertEqual(value, {"api_key": "REDACTED", "headers": "REDACTED"})

    def test_body_is_fingerprint_only(self):
        value = safe({"content": "private document", "code": "print(42)"})
        self.assertEqual(value["content"]["characters"], 16)
        self.assertNotIn("private document", str(value))

    def test_error_code_preserved(self):
        self.assertEqual(safe({"error_code": "VERIFY_FAILED"})["error_code"], "VERIFY_FAILED")

    def test_bearer_redacted_in_unexpected_field(self):
        self.assertEqual(safe("Bearer private-value", "reason"), "REDACTED")

    def test_profile_restored(self):
        with tempfile.TemporaryDirectory() as folder:
            old = sys.getprofile()
            with Observer(Path(folder) / "trace.jsonl") as observer:
                observer.emit("unit")
            self.assertIs(sys.getprofile(), old)
            self.assertEqual(observer.errors, 0)

    def test_budget_counts_attempts_and_forwards_unchanged(self):
        class Sink:
            def emit(self, *args, **kwargs):
                pass
        with patch.object(http.client.HTTPSConnection, "request", return_value="real-result") as request:
            original = http.client.HTTPSConnection.request
            with RequestBudget(Sink(), 1) as budget:
                conn = http.client.HTTPSConnection("api.groq.com")
                self.assertEqual(conn.request("POST", "/x", body='{"model":"test"}'), "real-result")
                with self.assertRaises(BudgetStop):
                    conn.request("POST", "/x")
                self.assertEqual(budget.count, 1)
                self.assertEqual(request.call_count, 1)
            self.assertIs(http.client.HTTPSConnection.request, original)

    def test_six_base_cases_unchanged(self):
        self.assertEqual(len(CASES), 6)
        self.assertEqual(CASES[4], "vuelve a intentarlo")


if __name__ == "__main__":
    unittest.main()
