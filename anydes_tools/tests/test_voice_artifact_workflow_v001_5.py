from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import unittest
from uuid import uuid4

from anydes_tools.conversation.content_generation import ContentGenerationSpec
from anydes_tools.conversation.controller import ToolConversationController
from anydes_tools.conversation.intent_normalization import (
    ConfirmationDecision,
    classify_confirmation,
    extract_explicit_filename_span,
    resolve_document_format,
)
from anydes_tools.conversation.document_model import (
    DocumentEditOperation,
    DocumentEditSpec,
    DocumentModel,
    apply_document_edit,
)
from anydes_tools.conversation.planner import plan_from_text
from anydes_tools.conversation.structured_text import ContentBuilder
from anydes_tools.settings.store import SettingsStore
from anydes_tools.tests.test_conversation_v001_5 import (
    _semantic_content_fixture,
    _semantic_line_fixture,
    _semantic_prose_fixture,
)


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "anydes_tools"
EVIDENCE = TOOLS / "logs" / "voice-artifact-workflow-e2e-v001_5.json"


class VoiceArtifactWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = TOOLS / "workspaces" / f"voice_artifacts_{uuid4().hex[:10]}"
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
        return self.controller.handle(text, turn_id=f"voice:{self.turn}")

    def test_01_filename_span_uses_semantic_slot(self):
        span = extract_explicit_filename_span(
            "crea un Word y ponle como nombre al documento Informe Energía"
        )
        self.assertIsNotNone(span)
        self.assertEqual(span.display_name, "Informe Energía")
        self.assertNotIn("al documento", span.display_name.casefold())

    def test_02_safe_create_is_immediate_and_path_is_friendly(self):
        result = self.ask("crea un Word llamado Informe Final con un resumen de 80 palabras sobre energía")
        path = self.workspace / "Informe Final.docx"
        self.assertTrue(result.executed)
        self.assertTrue(path.is_file())
        self.assertIsNone(self.controller.state.pending_tool_action)
        self.assertIn(path.name, result.text)
        self.assertIn(self.workspace.name, result.text)
        self.assertNotIn(str(self.workspace.resolve()), result.text)
        location = self.ask("¿cuál es la ruta completa?")
        self.assertIn(str(path.resolve()), location.text)

    def test_03_structural_docx_append_preserves_previous_content(self):
        self.ask("crea un Word llamado Informe Base con un resumen de 80 palabras sobre energía")
        path = self.workspace / "Informe Base.docx"
        before = str(self.controller.last_results[-1].output.get("content", ""))
        result = self.ask("edita el último Word y al final pon Firmado por Anydes")
        after = str(self.controller.last_results[-1].output.get("content", ""))
        self.assertTrue(result.executed)
        self.assertIn(before, after)
        self.assertTrue(after.rstrip().endswith("Firmado por Anydes"))
        self.assertTrue(path.is_file())
        self.assertEqual(self.controller.state.last_verification["status"], "VERIFIED")

    def test_04_artifact_reference_by_type_beats_active_txt(self):
        self.ask("crea un Word llamado Documento Tema con un resumen de 80 palabras sobre energía")
        word = self.workspace / "Documento Tema.docx"
        self.ask("crea Activo.txt que diga texto activo")
        self.assertEqual(Path(self.controller.state.active_document).suffix, ".txt")
        result = self.ask("convierte el archivo de Word a Pd F")
        self.assertTrue(result.executed)
        self.assertEqual(Path(self.controller.state.source_document), word)
        self.assertTrue((self.workspace / "Documento Tema.pdf").is_file())

    def test_05_rename_missing_name_never_becomes_create(self):
        self.ask("crea Original.txt que diga contenido")
        missing = self.ask("cambia el nombre al último archivo")
        self.assertFalse(missing.executed)
        self.assertIn("qué nombre", missing.text.casefold())
        self.assertEqual(len(list(self.workspace.glob("*.txt"))), 1)
        confirm_move = self.ask("Resultado final")
        self.assertIn("cambiar el nombre", confirm_move.text.casefold())
        self.assertNotIn("mover", confirm_move.text.casefold())
        completed = self.ask("sí")
        self.assertTrue(completed.executed)
        self.assertFalse((self.workspace / "Original.txt").exists())
        self.assertTrue((self.workspace / "Resultado final.txt").is_file())

    def test_06_semantic_pdf_is_nonempty_extractable_and_source_aware(self):
        action = plan_from_text(
            "crea un documento PDF donde esté la visión y misión de una organización ficticia",
            self.controller.state,
            content_builder=self.controller.content_builder,
        )
        specification = ContentGenerationSpec.from_dict(
            action.arguments["content_generation_spec"]
        )
        self.assertEqual(specification.source_mode, "FACTUAL_FROM_AVAILABLE_CONTEXT")
        result = self.controller._resolve_and_prepare(action, turn_id="source-aware")
        path = Path(action.target_path)
        extracted = str(self.controller.last_results[-1].output.get("content", ""))
        self.assertTrue(result.executed)
        self.assertGreater(path.stat().st_size, 100)
        self.assertTrue(extracted.strip())

    def test_07_empty_semantic_pdf_fails_closed_without_artifact(self):
        self.controller.semantic_prose_generator = lambda _specification: ""
        result = self.ask("crea un PDF donde esté una explicación sobre un sistema ficticio")
        failed = self.controller.state.last_failed_action
        self.assertFalse(result.executed)
        self.assertIsNotNone(failed)
        self.assertFalse(Path(failed.target_path).exists())
        self.assertNotIn("listo", result.text.casefold())

    def test_08_stt_normalization_is_bounded_to_control_entities(self):
        self.assertEqual(resolve_document_format("documento Pd F").value, "pdf")
        self.assertEqual(classify_confirmation("see"), ConfirmationDecision.CONFIRM)
        self.assertEqual(
            classify_confirmation("see", destructive=True),
            ConfirmationDecision.AMBIGUOUS,
        )

    def test_09_overwrite_and_delete_still_require_confirmation(self):
        path = self.workspace / "Existe.txt"
        path.write_text("antes", encoding="utf-8")
        overwrite = self.ask(f'crea un txt en "{path}" que diga después')
        self.assertFalse(overwrite.executed)
        self.assertIn("sobrescribir", overwrite.text.casefold())
        self.ask("sí")
        delete = self.ask("borra este archivo")
        self.assertFalse(delete.executed)
        ambiguous = self.ask("see")
        self.assertFalse(ambiguous.executed)
        self.assertTrue(path.exists())
        deleted = self.ask("sí confirmo")
        self.assertTrue(deleted.executed)
        self.assertFalse(path.exists())

    def test_10_document_model_supports_general_structural_operations(self):
        base = DocumentModel(("Título", "Párrafo uno", "Párrafo dos"))
        appended = apply_document_edit(
            base,
            DocumentEditSpec("x.docx", DocumentEditOperation.APPEND_PARAGRAPH, "Final"),
        )
        prepended = apply_document_edit(
            base,
            DocumentEditSpec("x.docx", DocumentEditOperation.PREPEND_PARAGRAPH, "Inicio"),
        )
        replaced = apply_document_edit(
            base,
            DocumentEditSpec("x.docx", DocumentEditOperation.REPLACE_TEXT, "Uno", "uno"),
        )
        inserted = apply_document_edit(
            base,
            DocumentEditSpec("x.docx", DocumentEditOperation.INSERT_AFTER, "Intermedio", "Párrafo uno"),
        )
        self.assertEqual(appended.paragraphs[-1], "Final")
        self.assertEqual(prepended.paragraphs[0], "Inicio")
        self.assertIn("Párrafo Uno", replaced.render_text())
        self.assertEqual(inserted.paragraphs[2], "Intermedio")

    def test_11_recent_artifacts_resolve_txt_and_docx_independently(self):
        self.ask("crea un Word llamado Tema Uno con un resumen de 80 palabras sobre energía")
        word = self.workspace / "Tema Uno.docx"
        self.ask("crea Datos.txt que diga datos")
        txt = self.workspace / "Datos.txt"
        self.assertEqual(
            Path(self.controller.state.recent_artifact_state.resolve("el archivo de Word")),
            word,
        )
        self.assertEqual(
            Path(self.controller.state.recent_artifact_state.resolve("el TXT anterior")),
            txt,
        )

    def test_12_benign_voice_confirmation_resumes_execute_code(self):
        pending = self.ask("ejecuta este código Python: print('voz')")
        self.assertFalse(pending.executed)
        completed = self.ask("see")
        self.assertTrue(completed.executed)
        self.assertIn("voz", completed.text)

    def test_13_current_official_fact_request_marks_retrieval_requirement(self):
        action = plan_from_text(
            "crea un PDF donde esté la misión oficial actual de una institución",
            self.controller.state,
            content_builder=self.controller.content_builder,
        )
        specification = ContentGenerationSpec.from_dict(
            action.arguments["content_generation_spec"]
        )
        self.assertEqual(specification.source_mode, "FACTUAL_REQUIRES_RETRIEVAL")


class ExactStagingVoiceArtifactE2E(unittest.TestCase):
    def test_exact_launcher_adapter_voice_like_sequence(self):
        from probar_anydes_tools_conversacion_v001_5 import install_tool_conversation_runtime
        from anydes.nucleo.assistant import AnydesAssistant
        from anydes_tools.conversation.runtime_patch import ensure_controller_for

        workspace = TOOLS / "workspaces" / "E2E Voice Artifact Staging"
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
        memory_root = workspace.parent / ".test-action-memory-voice-e2e"
        shutil.rmtree(memory_root, ignore_errors=True)
        controller.use_isolated_action_memory(memory_root)
        controller.semantic_evaluator = _semantic_line_fixture
        controller.content_builder = ContentBuilder(
            _semantic_content_fixture,
            _semantic_line_fixture,
        )
        controller.semantic_prose_generator = _semantic_prose_fixture
        controller.settings.set_temporary_workspace(workspace)
        visible: list[str] = []

        try:
            created = assistant.respond_realtime(
                "crea un documento Word con un resumen de 100 palabras sobre termodinámica "
                "y ponle como nombre al documento Informe Termodinámica",
                on_text=visible.append,
            )
            word = workspace / "Informe Termodinámica.docx"
            edited = assistant.respond_realtime(
                "edita el último Word y al final pon Firmado por Anydes",
                on_text=visible.append,
            )
            word_text = str(controller.last_results[-1].output.get("content", ""))
            assistant.respond_realtime(
                "crea Activo.txt que diga documento de control",
                on_text=visible.append,
            )
            converted = assistant.respond_realtime(
                "convierte el archivo de Word a Pd F",
                on_text=visible.append,
            )
            pdf = workspace / "Informe Termodinámica.pdf"
            factual = assistant.respond_realtime(
                "crea un documento Pd F donde esté la visión y misión de una organización ficticia",
                on_text=visible.append,
            )
            factual_path = Path(controller.state.active_document)
            factual_text = str(controller.last_results[-1].output.get("content", ""))
            rename_missing = assistant.respond_realtime(
                "cambia el nombre al último archivo",
                on_text=visible.append,
            )
            rename_confirmation = assistant.respond_realtime(
                "Documento institucional",
                on_text=visible.append,
            )
            rename_done = assistant.respond_realtime("sí", on_text=visible.append)
            renamed = workspace / "Documento institucional.pdf"
            delete_pending = assistant.respond_realtime(
                "borra todos los archivos del workspace",
                on_text=visible.append,
            )
            delete_ambiguous = assistant.respond_realtime("see", on_text=visible.append)
            remained_after_ambiguous = sorted(item.name for item in workspace.iterdir())
            delete_done = assistant.respond_realtime("sí confirmo", on_text=visible.append)
            remaining = sorted(item.name for item in workspace.iterdir())

            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "launcher": "Probar Anydes Tools Conversacion v001.5.bat",
                "created_directly": word.is_file() or "Informe Termodinámica.docx" in created,
                "docx_append_verified": "Firmado por Anydes" in word_text,
                "converted_correct_source": Path(controller.state.source_document).name == word.name,
                "converted_pdf_created": pdf.is_file() or "Informe Termodinámica.pdf" in converted,
                "semantic_pdf_nonempty": bool(factual_text.strip()),
                "semantic_pdf_path": str(factual_path),
                "rename_requested_argument": "qué nombre" in rename_missing.casefold(),
                "rename_uses_natural_wording": (
                    "cambiar el nombre" in rename_confirmation.casefold()
                    and "mover" not in rename_confirmation.casefold()
                ),
                "rename_verified": renamed.is_file() or "Documento institucional.pdf" in rename_done,
                "delete_required_confirmation": "quieres" in delete_pending.casefold(),
                "ambiguous_delete_blocked": bool(remained_after_ambiguous),
                "delete_verified": not remaining,
                "pseudo_tool_markup_visible": any(
                    marker in value
                    for value in visible
                    for marker in ("[document.", "<tool_call", '"tool_id"')
                ),
                "friendly_paths": all(str(workspace.resolve()) not in value for value in visible),
                "responses": {
                    "created": created,
                    "edited": edited,
                    "converted": converted,
                    "factual": factual,
                    "rename_done": rename_done,
                    "delete_done": delete_done,
                    "delete_ambiguous": delete_ambiguous,
                },
            }
            EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
            EVIDENCE.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self.assertTrue(payload["created_directly"])
            self.assertTrue(payload["docx_append_verified"])
            self.assertTrue(payload["converted_correct_source"])
            self.assertTrue(payload["converted_pdf_created"])
            self.assertTrue(payload["semantic_pdf_nonempty"])
            self.assertTrue(payload["rename_requested_argument"])
            self.assertTrue(payload["rename_uses_natural_wording"])
            self.assertTrue(payload["rename_verified"])
            self.assertTrue(payload["delete_required_confirmation"])
            self.assertTrue(payload["ambiguous_delete_blocked"])
            self.assertTrue(payload["delete_verified"])
            self.assertFalse(payload["pseudo_tool_markup_visible"])
            self.assertTrue(payload["friendly_paths"])
        finally:
            controller.close()
            shutil.rmtree(workspace, ignore_errors=True)
            shutil.rmtree(memory_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
