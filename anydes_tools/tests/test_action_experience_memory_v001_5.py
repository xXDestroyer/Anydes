from __future__ import annotations

import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4

from anydes_tools.conversation.content_generation import (
    ContentGenerationSpec,
    parse_content_generation_spec,
)
from anydes_tools.conversation.controller import ToolConversationController
from anydes_tools.conversation.contracts import ConversationToolResponse
from anydes_tools.conversation.intent_normalization import extract_explicit_filename_span
from anydes_tools.conversation.presentation import final_response_gate
from anydes_tools.conversation.semantic_runtime import _safe_self_context
from anydes_tools.settings.store import SettingsStore
from anydes_tools.tests.test_conversation_v001_5 import (
    _semantic_content_fixture,
    _semantic_line_fixture,
    _semantic_prose_fixture,
)


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "anydes_tools"


class ActionExperienceMemoryV0015Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = TOOLS / "workspaces" / f"action_memory_{uuid4().hex[:10]}"
        self.workspace.mkdir(parents=True)
        settings = SettingsStore(ROOT, TOOLS, settings_path=self.workspace / "settings.json")
        settings.set_temporary_workspace(self.workspace)
        self.controller = ToolConversationController(
            ROOT,
            TOOLS,
            settings=settings,
            semantic_evaluator=_semantic_line_fixture,
            semantic_content_generator=_semantic_content_fixture,
            semantic_prose_generator=_semantic_prose_fixture,
        )
        self.turn = 0

    def tearDown(self) -> None:
        self.controller.close()
        shutil.rmtree(self.workspace, ignore_errors=True)

    def ask(self, text: str):
        self.turn += 1
        return self.controller.handle(text, turn_id=f"experience:{self.turn}")

    def test_01_semantic_docx_never_commits_empty_content(self):
        response = self.ask("haz un documento de Word en el que te describas a ti misma")
        path = Path(self.controller.state.active_document)
        extracted = str(self.controller.last_results[-1].output.get("content", ""))
        self.assertTrue(response.executed)
        self.assertEqual(path.suffix.casefold(), ".docx")
        self.assertTrue(extracted.strip())
        self.assertGreater(path.stat().st_size, 100)

    def test_02_self_description_is_a_generic_semantic_source_mode(self):
        specification = parse_content_generation_spec(
            "haz un Word en el que te describas a ti misma",
            purpose="create_document",
        )
        self.assertIsInstance(specification, ContentGenerationSpec)
        self.assertEqual(specification.content_kind, "SELF_DESCRIPTION")
        self.assertEqual(specification.source_mode, "SELF_CONTEXT")
        self.assertEqual(specification.topic, "Anydes")

    def test_03_empty_self_description_fails_before_final_artifact(self):
        self.controller.semantic_prose_generator = lambda _specification: ""
        response = self.ask("haz un documento de Word en el que te describas a ti misma")
        failed = self.controller.state.last_failed_action
        self.assertFalse(response.executed)
        self.assertIsNotNone(failed)
        self.assertFalse(Path(failed.target_path).exists())
        self.assertEqual(self.controller.state.recent_action_state.events[-1].result_status, "FAILED")
        self.assertEqual(self.controller.state.recent_action_state.events[-1].provenance, "ACTION_ATTEMPT")

    def test_04_standalone_positional_append_edits_existing_docx(self):
        self.ask("crea un Word llamado Base de prueba con un resumen de 80 palabras sobre energía")
        path = self.workspace / "Base de prueba.docx"
        before = path.read_bytes()
        response = self.ask("al final escribe Hecho por Anydes")
        after = str(self.controller.last_results[-1].output.get("content", ""))
        self.assertTrue(response.executed)
        self.assertNotEqual(before, path.read_bytes())
        self.assertTrue(after.rstrip().endswith("Hecho por Anydes"))
        self.assertEqual(self.controller.state.last_verification["status"], "VERIFIED")

    def test_05_rename_presentation_hides_move_primitive(self):
        self.ask("crea Original.txt que diga contenido")
        pending = self.ask("cambia el nombre al último archivo a Renombrado")
        self.assertFalse(pending.executed)
        self.assertIn("cambiar el nombre", pending.text.casefold())
        self.assertNotIn("mover", pending.text.casefold())

    def test_06_spoken_filename_resolves_and_duplicates_recent_artifact(self):
        self.ask("crea un Word llamado prueba 75 que diga contenido verificable")
        response = self.ask("duplica el archivo de Word que se llama prueba setenta y cinco")
        source = self.workspace / "Prueba 75.docx"
        copy = self.workspace / "Prueba 75 copia.docx"
        self.assertTrue(response.executed)
        self.assertTrue(source.is_file())
        self.assertTrue(copy.is_file())
        self.assertEqual(source.read_bytes(), copy.read_bytes())
        record = self.controller.state.recent_artifact_state.find_by_path(str(copy))
        self.assertEqual(record.relationship.get("copied_from"), str(source.resolve()))

    def test_07_workspace_fallback_resolves_unique_spoken_filename(self):
        path = self.workspace / "muestra 42.txt"
        path.write_text("contenido", encoding="utf-8")
        response = self.ask("duplica el archivo que se llama muestra cuarenta y dos")
        self.assertTrue(response.executed)
        self.assertTrue((self.workspace / "muestra 42 copia.txt").is_file())

    def test_08_explicit_filename_span_preserves_full_semantic_name(self):
        span = extract_explicit_filename_span(
            "crea un Word llamado Informe sobre termodinámica con un resumen de 80 palabras sobre energía"
        )
        self.assertIsNotNone(span)
        self.assertEqual(span.display_name, "Informe sobre termodinámica")
        response = self.ask(
            "crea un Word llamado Informe sobre termodinámica con un resumen de 80 palabras sobre energía"
        )
        self.assertTrue(response.executed)
        self.assertTrue((self.workspace / "Informe sobre termodinámica.docx").is_file())

    def test_09_universal_response_gate_blocks_protocol_variants(self):
        variants = (
            '{"action":"filesystem.copy","arguments":{"source":"a","destination":"b"}}',
            '[document.create_txt name="x.txt"]',
            '<action_request>{"tool_id":"official.document"}</action_request>',
            'filesystem.copy(source="a", destination="b")',
        )
        for value in variants:
            clean, blocked, _source = final_response_gate(value)
            self.assertTrue(blocked, value)
            self.assertNotIn("filesystem.", clean)
            self.assertNotIn("document.", clean)

    def test_10_successful_create_records_direct_action_experience(self):
        response = self.ask("crea Evidencia Unica.txt que diga prueba")
        event = self.controller.state.recent_action_state.events[-1]
        self.assertTrue(response.executed)
        self.assertEqual(event.result_status, "SUCCESS")
        self.assertEqual(event.provenance, "DIRECT_ACTION_EXPERIENCE")
        self.assertIn("Evidencia Unica.txt", event.entities)
        self.assertTrue(event.artifact_ids)
        self.assertTrue(self.controller.action_memory.log_path.is_file())
        self.assertTrue(self.controller.action_memory.memory_path.is_file())

    def test_11_rename_copy_and_convert_keep_artifact_relations_and_history(self):
        self.ask("crea Relacion.txt que diga texto")
        rename_pending = self.ask("cambia el nombre al último archivo a Relacion Nueva")
        self.assertIn("cambiar el nombre", rename_pending.text.casefold())
        self.ask("sí")
        renamed = self.workspace / "Relacion Nueva.txt"
        renamed_record = self.controller.state.recent_artifact_state.find_by_path(str(renamed))
        self.assertTrue(renamed_record.relationship.get("renamed_from"))
        self.ask("duplica el archivo que se llama Relacion Nueva.txt")
        copied = self.workspace / "Relacion Nueva copia.txt"
        copied_record = self.controller.state.recent_artifact_state.find_by_path(str(copied))
        self.assertEqual(copied_record.relationship.get("copied_from"), str(renamed.resolve()))
        self.ask("convierte ese mismo documento a PDF")
        pdf = self.workspace / "Relacion Nueva copia.pdf"
        pdf_record = self.controller.state.recent_artifact_state.find_by_path(str(pdf))
        self.assertEqual(pdf_record.relationship.get("converted_from"), str(copied.resolve()))
        action_types = [item.action_type for item in self.controller.state.recent_action_state.events]
        self.assertIn("rename", action_types)
        self.assertIn("copy_artifact", action_types)
        self.assertIn("convert_document", action_types)

    def test_12_delete_records_event_without_erasing_historical_artifact(self):
        self.ask("crea Historia.txt que diga temporal")
        path = self.workspace / "Historia.txt"
        artifact_id = self.controller.state.recent_artifact_state.find_by_path(str(path)).artifact_id
        self.ask("borra este archivo")
        self.ask("sí confirmo")
        record = self.controller.state.recent_artifact_state.find_by_path(str(path))
        event = self.controller.state.recent_action_state.events[-1]
        self.assertFalse(path.exists())
        self.assertEqual(record.artifact_id, artifact_id)
        self.assertIsNotNone(record.deleted_at)
        self.assertEqual(event.action_type, "delete")
        self.assertEqual(event.result_status, "SUCCESS")

    def test_13_action_memory_retrieves_only_real_evidence(self):
        self.ask("crea Recuerdo verificable.txt que diga evidencia")
        response = self.ask("¿recuerdas que creaste Recuerdo verificable.txt?")
        self.assertTrue(response.handled)
        self.assertIn("Recuerdo verificable.txt", response.text)
        self.assertTrue(response.debug.get("verified_action_evidence"))
        absent = self.ask("¿recuerdas que creaste Jamás existió.txt?")
        self.assertNotIn("Jamás existió.txt", absent.text)

    def test_14_safe_self_context_preserves_identity_and_redacts_secrets(self):
        class Value:
            pass

        assistant = Value()
        assistant.personality = Value()
        assistant.personality.prompt_text = "Soy Anydes y mantengo separadas identidad y experiencia."
        assistant.config = Value()
        assistant.config.profile_name = "Anydes"
        assistant._realtime_service = Value()
        assistant._realtime_service.session = Value()
        assistant._realtime_service.session.tool_capability_note = (
            "Puedo crear documentos en el workspace autorizado.\n"
            "GROQ_API_KEY=secret-value\n"
            "token: secret-token"
        )

        context = _safe_self_context(assistant)

        self.assertIn("IDENTITY_NAME: Anydes", context)
        self.assertIn("Puedo crear documentos", context)
        self.assertNotIn("secret-value", context)
        self.assertNotIn("secret-token", context)
        self.assertGreaterEqual(context.count("[REDACTED]"), 2)


class ExactLauncherActionExperienceE2E(unittest.TestCase):
    def test_exact_staging_runtime_executes_append_copy_and_filters_control_text(self):
        from probar_anydes_tools_conversacion_v001_5 import install_tool_conversation_runtime
        from anydes.nucleo.assistant import AnydesAssistant
        from anydes_tools.conversation.runtime_patch import ensure_controller_for

        workspace = TOOLS / "workspaces" / "E2E Action Experience Staging"
        shutil.rmtree(workspace, ignore_errors=True)
        workspace.mkdir(parents=True)
        install_tool_conversation_runtime()

        class Audit:
            def __init__(self):
                self.events = []

            def record(self, event, **details):
                self.events.append({"event": event, "details": details})

        assistant = object.__new__(AnydesAssistant)
        assistant.audit = Audit()
        assistant._realtime_service = None
        assistant.last_action_results = ()
        assistant.last_intention = None
        controller = ensure_controller_for(assistant, ROOT)
        controller.use_isolated_action_memory(workspace / ".action-memory-test")
        controller.semantic_evaluator = _semantic_line_fixture
        controller.semantic_content_generator = _semantic_content_fixture
        controller.content_builder.semantic_generator = _semantic_content_fixture
        controller.semantic_prose_generator = _semantic_prose_fixture
        controller.action_memory_responder = None
        controller.settings.set_temporary_workspace(workspace)
        visible: list[str] = []
        try:
            created = assistant.respond_realtime(
                "haz un documento Word en el que te describas a ti misma",
                on_text=visible.append,
            )
            source = Path(controller.state.active_document)
            appended = assistant.respond_realtime(
                "al final escribe Firmado por Anydes",
                on_text=visible.append,
            )
            copied = assistant.respond_realtime(
                f"duplica el archivo que se llama {source.stem}",
                on_text=visible.append,
            )
            remembered = assistant.respond_realtime(
                f"¿recuerdas que creaste {source.name}?",
                on_text=visible.append,
            )
            with patch.object(
                controller,
                "handle",
                return_value=ConversationToolResponse(
                    True,
                    '{"action":"filesystem.copy","arguments":{"source":"a","destination":"b"}}',
                    executed=False,
                ),
            ):
                blocked_control = assistant.respond_realtime(
                    "ruta simulada de control",
                    on_text=visible.append,
                )
            copy_path = source.with_name(f"{source.stem} copia{source.suffix}")
            payload = {
                "created_nonempty": source.is_file() and source.stat().st_size > 100,
                "append_executed": "actualicé" in appended,
                "copy_verified": copy_path.is_file() and copy_path.read_bytes() == source.read_bytes(),
                "memory_evidence": source.name in remembered,
                "universal_response_gate": "filesystem.copy" not in blocked_control,
                "pseudo_tool_markup_visible": any(
                    marker in value
                    for value in visible
                    for marker in ('"action"', "filesystem.", "document.", "<tool_call")
                ),
                "responses": {"created": created, "appended": appended, "copied": copied},
            }
            evidence = TOOLS / "logs" / "action-experience-e2e-v001_5.json"
            evidence.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            self.assertTrue(payload["created_nonempty"])
            self.assertTrue(payload["append_executed"])
            self.assertTrue(payload["copy_verified"])
            self.assertTrue(payload["memory_evidence"])
            self.assertTrue(payload["universal_response_gate"])
            self.assertFalse(payload["pseudo_tool_markup_visible"])
        finally:
            controller.close()
            shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
