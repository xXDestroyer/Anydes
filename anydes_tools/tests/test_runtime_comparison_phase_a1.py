import json
from pathlib import Path
import unittest

from anydes_tools.tests.runtime_comparison import STAGES, checkpoints, format_evidence, fingerprint


class ComparisonTests(unittest.TestCase):
    def test_empty_json_evidence(self):
        self.assertTrue(format_evidence("")["empty"])
        self.assertEqual(format_evidence("")["format_shape"], "")

    def test_snippet_has_no_identifiers(self):
        evidence = format_evidence('{"password": "personal value"}')
        self.assertNotIn("personal", json.dumps(evidence))
        self.assertNotIn("password", evidence["format_shape"])

    def test_secret_is_redacted(self):
        self.assertEqual(format_evidence("Bearer secretValue")["format_shape"], "REDACTED")

    def test_parser_checkpoint(self):
        self.assertEqual(checkpoints("anydes_tools.conversation.semantic_runtime", "_parse_json_object", "NoneType", {}), ["PARSE_RESULT"])

    def test_prompt_hash_includes_tail(self):
        self.assertNotEqual(fingerprint("a" * 1000 + "x"), fingerprint("a" * 1000 + "y"))

    def test_real_fast_path_name(self):
        self.assertEqual(checkpoints("anydes_tools.conversation.planner", "plan_from_text", "NoneType", {}), ["FAST_PATH_RESULT"])

    def test_no_fake_execution_on_codegen(self):
        stages = checkpoints("anydes_tools.conversation.semantic_runtime", "__call__", "AssistantCodeGenerator", {})
        self.assertEqual(stages, ["CODE_GENERATION"])

    def test_exact_confirmation_and_partial_label(self):
        root = Path(__file__).parent / "fixtures"
        for name in ("phase_a1_fresh.json", "phase_a1_partial_human.json"):
            fixture = json.loads((root / name).read_text(encoding="utf-8"))
            self.assertEqual(fixture["inputs"][-1], "si dale")
            self.assertFalse(fixture["historical_state_reproduced"])
        self.assertEqual(len(STAGES), 17)


if __name__ == "__main__":
    unittest.main()
