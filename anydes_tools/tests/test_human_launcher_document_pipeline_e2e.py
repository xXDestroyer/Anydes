from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "anydes_tools" / "workspaces" / "E2E Document Pipeline"
RESULT = ROOT / "anydes_tools" / "logs" / "human-launcher-document-pipeline-e2e.json"


class _AuditRecorder:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def record(self, event: str, **details) -> None:
        self.events.append({"event": event, "details": details})


def _content_generator(request):
    if "dos cifras" in request.description.casefold():
        return [
            "15", "20", "25", "31", "42", "50", "61", "72", "85", "99",
            "10", "11", "12", "13", "14", "16", "17", "18", "19", "21",
        ][:request.count]
    return [f"{request.semantic_type} {index}" for index in range(1, request.count + 1)]


def _line_evaluator(_predicate, _lines):
    return []


def _prose_generator(specification):
    words = (
        f"{specification.content_kind} sobre {specification.topic} "
        "con contenido claro y verificable"
    ).split()
    while len(words) < specification.target_length_words:
        words.extend(("contexto", "explicación", "evidencia", "aplicación"))
    return " ".join(words[:specification.target_length_words])


class HumanLauncherDocumentPipelineE2E(unittest.TestCase):
    def test_exact_staging_runtime_commits_only_verified_documents(self):
        from probar_anydes_tools_conversacion_v001_5 import install_tool_conversation_runtime
        from anydes.nucleo.assistant import AnydesAssistant
        from anydes_tools.conversation.runtime_patch import ensure_controller_for
        from anydes_tools.conversation.structured_text import ContentBuilder

        shutil.rmtree(WORKSPACE, ignore_errors=True)
        WORKSPACE.mkdir(parents=True)
        install_tool_conversation_runtime()

        assistant = object.__new__(AnydesAssistant)
        assistant.audit = _AuditRecorder()
        assistant._realtime_service = None
        assistant.last_action_results = ()
        assistant.last_intention = None
        controller = ensure_controller_for(assistant, ROOT)
        memory_root = WORKSPACE.parent / ".test-action-memory-document-pipeline"
        shutil.rmtree(memory_root, ignore_errors=True)
        controller.use_isolated_action_memory(memory_root)
        controller.semantic_evaluator = _line_evaluator
        controller.content_builder = ContentBuilder(_content_generator, _line_evaluator)
        controller.semantic_prose_generator = _prose_generator
        controller.settings.set_temporary_workspace(WORKSPACE)
        visible: list[str] = []

        try:
            created_message = assistant.respond_realtime(
                "crea un Word llamado Informe Energía con unas 400 palabras sobre energía",
                on_text=visible.append,
            )
            docx = WORKSPACE / "Informe Energía.docx"
            docx_text = str(controller.last_results[-1].output.get("content", ""))

            converted_message = assistant.respond_realtime(
                "convierte ese mismo documento a PDF",
                on_text=visible.append,
            )
            pdf = WORKSPACE / "Informe Energía.pdf"
            pdf_text = str(controller.last_results[-1].output.get("content", ""))
            conversion_verification = dict(controller.state.last_verification)

            numbers_message = assistant.respond_realtime(
                'crea Enumeracion.txt con 20 números de dos cifras, uno por línea, y cada vez '
                'que veas un 5 escribe al lado "- Aquí hay un cinco"',
                on_text=visible.append,
            )
            numbers = WORKSPACE / "Enumeracion.txt"
            number_lines = numbers.read_text(encoding="utf-8").splitlines()
            deterministic_complete = all(
                ("Aquí hay un cinco" in line) == ("5" in line.split()[0])
                for line in number_lines
            )

            original = controller._execute_capability

            def always_corrupt(capability, arguments, **kwargs):
                result = original(capability, arguments, **kwargs)
                path = Path(str(arguments.get("path", "")))
                if path.exists() and ".anydes-" in path.name and capability in {
                    "document.create_txt", "document.edit_document"
                }:
                    path.write_text(
                        path.read_text(encoding="utf-8").replace(
                            "85 - Aquí hay un cinco", "85", 1
                        ),
                        encoding="utf-8",
                    )
                return result

            with patch.object(controller, "_execute_capability", side_effect=always_corrupt):
                failed_message = assistant.respond_realtime(
                    'crea Invalido.txt con 20 números de dos cifras, uno por línea, y cada vez '
                    'que veas un 5 escribe al lado "- Aquí hay un cinco"',
                    on_text=visible.append,
                )

            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "launcher": "Probar Anydes Tools Conversacion v001.5.bat",
                "adapter_active": bool(getattr(
                    AnydesAssistant.respond_realtime,
                    "_anydes_tool_conversation_v001_5",
                    False,
                )),
                "docx_path": str(docx),
                "docx_exists": docx.is_file(),
                "docx_title_present": docx_text.startswith("Informe Energía\n"),
                "docx_word_count": len(docx_text.split()),
                "pdf_path": str(pdf),
                "pdf_exists": pdf.is_file(),
                "pdf_nonempty": pdf.is_file() and pdf.stat().st_size > 0,
                "pdf_text_nonempty": bool(pdf_text.strip()),
                "conversion_similarity": conversion_verification.get("content_similarity", 0.0),
                "atomic_conversion": conversion_verification.get("atomic_commit", False),
                "related_artifacts": list(controller.state.related_artifacts),
                "numbers_path": str(numbers),
                "number_count": len(number_lines),
                "deterministic_complete": deterministic_complete,
                "contains_85_annotation": "85 - Aquí hay un cinco" in number_lines,
                "invalid_final_exists": (WORKSPACE / "Invalido.txt").exists(),
                "failed_response": failed_message,
                "pseudo_tool_markup_visible": any(
                    marker in text
                    for text in visible
                    for marker in ("[document.", "<tool_call", '"tool_id"')
                ),
                "transaction_temp_files": [
                    str(item) for item in WORKSPACE.iterdir() if ".anydes-" in item.name
                ],
                "created_response": created_message,
                "converted_response": converted_message,
                "numbers_response": numbers_message,
            }
            RESULT.parent.mkdir(parents=True, exist_ok=True)
            RESULT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            self.assertTrue(payload["adapter_active"])
            self.assertTrue(payload["docx_exists"])
            self.assertTrue(payload["docx_title_present"])
            self.assertGreaterEqual(payload["docx_word_count"], 400)
            self.assertTrue(payload["pdf_exists"])
            self.assertTrue(payload["pdf_nonempty"])
            self.assertTrue(payload["pdf_text_nonempty"])
            self.assertGreaterEqual(payload["conversion_similarity"], 0.90)
            self.assertTrue(payload["atomic_conversion"])
            self.assertEqual(payload["number_count"], 20)
            self.assertTrue(payload["deterministic_complete"])
            self.assertTrue(payload["contains_85_annotation"])
            self.assertFalse(payload["invalid_final_exists"])
            self.assertEqual(payload["transaction_temp_files"], [])
            self.assertFalse(payload["pseudo_tool_markup_visible"])
            self.assertNotIn("Listo", failed_message)
        finally:
            controller.close()
            shutil.rmtree(WORKSPACE, ignore_errors=True)
            shutil.rmtree(memory_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
