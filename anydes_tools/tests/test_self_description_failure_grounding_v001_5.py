from __future__ import annotations

import json
from pathlib import Path
import shutil
import unittest
from uuid import uuid4

from anydes_tools.audit import ToolAuditLog
from anydes_tools.conversation.controller import ToolConversationController
from anydes_tools.conversation.presentation import present_workspace_paths
from anydes_tools.settings.store import SettingsStore
from anydes_tools.tests.test_conversation_v001_5 import _semantic_prose_fixture


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "anydes_tools"
EXACT_REQUEST = (
    "Anydes crea un Word llamado Informe Personal y descríbete a ti misma "
    "en unas 250 palabras."
)


class SelfDescriptionFailureGroundingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = TOOLS / "workspaces" / f"self_failure_{uuid4().hex[:10]}"
        self.workspace.mkdir(parents=True)
        settings = SettingsStore(ROOT, TOOLS, settings_path=self.workspace / "settings.json")
        settings.set_temporary_workspace(self.workspace)
        self.controller = ToolConversationController(
            ROOT,
            TOOLS,
            settings=settings,
            semantic_prose_generator=_semantic_prose_fixture,
        )
        self.trace_path = self.workspace / "pipeline-trace.jsonl"
        self.controller.audit = ToolAuditLog(self.trace_path)
        self.turn = 0

    def tearDown(self) -> None:
        self.controller.close()
        shutil.rmtree(self.workspace, ignore_errors=True)

    def ask(self, text: str):
        self.turn += 1
        return self.controller.handle(text, turn_id=f"self:{self.turn}")

    def test_01_exact_human_phrase_creates_named_nonempty_docx(self):
        response = self.ask(EXACT_REQUEST)
        target = self.workspace / "Informe Personal.docx"
        extracted = str(self.controller.last_results[-1].output.get("content", ""))
        self.assertTrue(response.executed)
        self.assertTrue(target.is_file())
        self.assertGreater(target.stat().st_size, 100)
        self.assertGreaterEqual(len(extracted.split()), 200)
        self.assertLessEqual(len(extracted.split()), 300)
        rows = [json.loads(line) for line in self.trace_path.read_text(encoding="utf-8").splitlines()]
        traces = [row["details"] for row in rows if row["event"] == "conversation_content_pipeline_trace"]
        final = traces[-1]
        self.assertEqual(final["task_type"], "generate_document")
        self.assertEqual(final["content_kind"], "SELF_DESCRIPTION")
        self.assertEqual(final["source_context_type"], "SELF_CONTEXT")
        self.assertTrue(final["brain_called"])
        self.assertEqual(final["content_validator_result"], "VALID")
        self.assertTrue(final["document_spec_created"])
        self.assertTrue(final["renderer_called"])
        self.assertIn("SUCCESS", final["action_result"])

    def test_02_failure_question_uses_recorded_validator_cause(self):
        self.controller.semantic_prose_generator = lambda _specification: ""
        failed = self.ask(EXACT_REQUEST)
        explanation = self.ask("¿sabes por qué?")
        record = self.controller.state.last_tool_failure
        self.assertFalse(failed.executed)
        self.assertEqual(record.stage, "CONTENT_VALIDATION")
        self.assertEqual(record.error_code, "CONTENT_EMPTY")
        self.assertIn("contenido generado quedó vacío", explanation.text)
        self.assertNotIn("reserv", explanation.text.casefold())
        self.assertEqual(explanation.debug["response_source"], "STRUCTURED_FAILURE_RESULT")
        rows = [json.loads(line) for line in self.trace_path.read_text(encoding="utf-8").splitlines()]
        recorded = [row["details"] for row in rows if row["event"] == "conversation_tool_failure_state"]
        self.assertEqual(recorded[-1]["error_code"], "CONTENT_EMPTY")

    def test_03_actionable_capability_question_executes(self):
        response = self.ask("¿Puedes crear un documento de Word donde te describas a ti misma?")
        self.assertTrue(response.executed)
        self.assertNotIn("sí, puedo crear", response.text.casefold())
        self.assertTrue(Path(self.controller.state.active_document).is_file())

    def test_04_capability_awareness_uses_friendly_workspace_name(self):
        response = self.ask("¿en qué carpeta tienes permiso?")
        self.assertIn(self.workspace.name, response.text)
        self.assertNotIn(str(self.workspace.resolve()), response.text)

    def test_05_final_path_presentation_hides_known_canonical_workspace(self):
        source = f"Puedo trabajar en {self.workspace.resolve()}\\Informe Personal.docx"
        hidden = present_workspace_paths(
            source,
            (str(self.workspace.resolve()),),
            reveal_full_path=False,
        )
        visible = present_workspace_paths(
            source,
            (str(self.workspace.resolve()),),
            reveal_full_path=True,
        )
        self.assertNotIn(str(self.workspace.resolve()), hidden)
        self.assertIn(self.workspace.name, hidden)
        self.assertIn(str(self.workspace.resolve()), visible)

    def test_06_hazlo_tu_entonces_retries_same_failed_task_once(self):
        calls = 0

        def flaky(specification):
            nonlocal calls
            calls += 1
            return "" if calls == 1 else _semantic_prose_fixture(specification)

        self.controller.semantic_prose_generator = flaky
        first = self.ask(EXACT_REQUEST)
        retried = self.ask("hazlo tú entonces")
        self.assertFalse(first.executed)
        self.assertTrue(retried.executed)
        self.assertEqual(calls, 2)
        self.assertTrue((self.workspace / "Informe Personal.docx").is_file())
        self.assertEqual(self.controller.state.retry_count, 0)


if __name__ == "__main__":
    unittest.main()
