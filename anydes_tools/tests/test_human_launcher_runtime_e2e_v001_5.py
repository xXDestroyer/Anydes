from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import unittest


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "anydes_tools" / "workspaces" / "E2E Carpeta de trabajo de Anydes"
RESULT = ROOT / "anydes_tools" / "logs" / "human-launcher-e2e-v001_5.json"


class _AuditRecorder:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def record(self, event: str, **details) -> None:
        self.events.append({"event": event, "details": details})


def _content_generator(request):
    values = [
        "Rojo", "Azul", "Amarillo", "Verde", "Gris", "Plata", "Naranja",
        "Violeta", "Rosa", "Negro", "Blanco", "Cian", "Magenta", "Turquesa",
        "Dorado", "Coral", "Oliva", "Lima", "Índigo", "Salmón",
    ]
    return values[:request.count]


def _line_evaluator(predicate, lines):
    condition = predicate.description.casefold()
    if "primari" in condition:
        return [
            index for index, line in enumerate(lines, 1)
            if line.casefold() in {"rojo", "azul", "amarillo"}
        ]
    if "gris" in condition:
        return [
            index for index, line in enumerate(lines, 1)
            if line.casefold() in {"gris", "plata"}
        ]
    return []


def _prose_generator(specification):
    words = (
        f"{specification.content_kind} sobre {specification.topic} "
        "con contenido claro y verificable"
    ).split()
    while len(words) < specification.target_length_words:
        words.extend(("contexto", "explicación", "evidencia", "aplicación"))
    return " ".join(words[:specification.target_length_words])


class HumanLauncherRuntimeE2ETests(unittest.TestCase):
    def test_documents_and_collection_delete_use_exact_staging_adapter(self):
        from probar_anydes_tools_conversacion_v001_5 import (
            install_tool_conversation_runtime,
        )
        from anydes.nucleo.assistant import AnydesAssistant
        from anydes_tools.conversation.runtime_patch import ensure_controller_for

        shutil.rmtree(WORKSPACE, ignore_errors=True)
        WORKSPACE.mkdir(parents=True)
        install_tool_conversation_runtime()

        assistant = object.__new__(AnydesAssistant)
        assistant.audit = _AuditRecorder()
        assistant._realtime_service = None
        assistant.last_action_results = ()
        assistant.last_intention = None

        controller = ensure_controller_for(assistant, ROOT)
        memory_root = WORKSPACE.parent / ".test-action-memory-human-runtime"
        shutil.rmtree(memory_root, ignore_errors=True)
        controller.use_isolated_action_memory(memory_root)
        from anydes_tools.conversation.structured_text import ContentBuilder

        controller.semantic_evaluator = _line_evaluator
        controller.content_builder = ContentBuilder(_content_generator, _line_evaluator)
        controller.semantic_prose_generator = _prose_generator
        controller.settings.set_temporary_workspace(WORKSPACE)
        visible: list[str] = []
        request = (
            "necesito que hagas un archivo .docx que enumere 20 colores "
            "diferentes, uno por línea"
        )
        completed = assistant.respond_realtime(request, on_text=visible.append)
        path = Path(controller.state.active_document)
        before_edit = path.read_bytes()
        edit_response = assistant.respond_realtime(
            "edita el archivo que acabas de crear y marca los colores primarios",
            on_text=visible.append,
        )
        after_first_edit = str(controller.last_results[-1].output.get("content", ""))
        second_response = assistant.respond_realtime(
            "edita el archivo que acabas de crear y marca las tonalidades de gris",
            on_text=visible.append,
        )
        after_edit = str(controller.last_results[-1].output.get("content", ""))
        lines = after_edit.splitlines()
        same_active_document_after_edit = Path(controller.state.active_document) == path
        file_changed_after_edit = before_edit != path.read_bytes()
        summary_completed = assistant.respond_realtime(
            "crea resumen.docx con un resumen de unas 500 palabras sobre energía solar",
            on_text=visible.append,
        )
        summary_path = Path(controller.state.active_document)
        summary_text = str(controller.last_results[-1].output.get("content", ""))
        text_completed = assistant.respond_realtime(
            "crea otro.txt que diga archivo adicional",
            on_text=visible.append,
        )
        text_path = Path(controller.state.active_document)
        delete_pending = assistant.respond_realtime(
            "borra todos los archivos del workspace",
            on_text=visible.append,
        )
        frozen_delete_plan = dict(controller.state.pending_tool_action.action.arguments["delete_plan"])
        delete_completed = assistant.respond_realtime("sí confirmo", on_text=visible.append)
        remaining_files = [str(item) for item in WORKSPACE.iterdir() if item.is_file()]

        tool_events = [
            row for row in controller.state.recent_tool_results
            if row.get("capability") == "document.create_txt"
        ]
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "launcher": "Probar Anydes Tools Conversacion v001.5.bat",
            "adapter_active": bool(getattr(
                AnydesAssistant.respond_realtime,
                "_anydes_tool_conversation_v001_5",
                False,
            )),
            "planner_selected": controller.state.pending_user_task is not None,
            "task_id": controller.state.pending_user_task.task_id,
            "plan_id": controller.state.pending_structured_plan.plan_id,
            "tool_id": controller.state.pending_structured_plan.tool_id,
            "capability": controller.state.pending_structured_plan.capability,
            "executor_called": bool(tool_events),
            "action_result_status": tool_events[-1]["status"] if tool_events else "MISSING",
            "response_source": "VERIFIED_ACTION_RESULT",
            "physical_path": str(path.resolve()),
            "file_exists_after_delete": path.is_file(),
            "line_count": len(lines),
            "same_active_document_after_edit": same_active_document_after_edit,
            "file_changed_after_edit": file_changed_after_edit,
            "first_semantic_edit_applied": after_first_edit.count("*") == 3,
            "second_semantic_edit_applied": after_edit.count("*") == 5,
            "summary_path": str(summary_path),
            "summary_word_count": len(summary_text.split()),
            "summary_nonempty": bool(summary_text.strip()),
            "text_path": str(text_path),
            "delete_target_count": len(frozen_delete_plan["targets"]),
            "delete_verified": controller.state.last_verification.get("status") == "VERIFIED",
            "remaining_files": remaining_files,
            "unexpected_note_files": [
                str(item) for item in WORKSPACE.glob("nota*.txt")
            ],
            "pseudo_tool_markup_visible": any(
                marker in text
                for text in visible
                for marker in ("[document.", "<tool_call", '"tool_id"')
            ),
            "completed_response": completed,
            "edit_response": edit_response,
            "second_response": second_response,
            "summary_completed": summary_completed,
            "text_completed": text_completed,
            "delete_pending": delete_pending,
            "delete_completed": delete_completed,
        }
        RESULT.parent.mkdir(parents=True, exist_ok=True)
        RESULT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        self.assertTrue(payload["adapter_active"])
        self.assertTrue(payload["planner_selected"])
        self.assertTrue(payload["executor_called"])
        self.assertEqual(payload["action_result_status"], "SUCCESS")
        self.assertFalse(path.is_file())
        self.assertEqual(path.name, "colores.docx")
        self.assertEqual(len(lines), 20)
        self.assertTrue(payload["same_active_document_after_edit"])
        self.assertTrue(payload["file_changed_after_edit"])
        self.assertTrue(payload["first_semantic_edit_applied"])
        self.assertTrue(payload["second_semantic_edit_applied"])
        self.assertGreaterEqual(payload["summary_word_count"], 500)
        self.assertTrue(payload["summary_nonempty"])
        self.assertEqual(payload["delete_target_count"], 3)
        self.assertTrue(payload["delete_verified"])
        self.assertEqual(payload["remaining_files"], [])
        self.assertEqual(payload["unexpected_note_files"], [])
        self.assertFalse(payload["pseudo_tool_markup_visible"])
        self.assertIn(path.parent.name, completed)
        self.assertNotIn(str(path.parent.resolve()), completed)
        controller.close()
        shutil.rmtree(memory_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
