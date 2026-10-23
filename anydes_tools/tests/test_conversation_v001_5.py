from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from anydes_tools.contracts import (
    CapabilityResolutionKind,
    ExecutionStatus,
    ToolExecutionResult,
)
from anydes_tools.conversation.controller import ToolConversationController
from anydes_tools.conversation.planner import (
    plan_from_text,
    validate_structured_action_plan,
)
from anydes_tools.conversation.document_transform import (
    DocumentOperation,
    classify_document_operation,
)
from anydes_tools.conversation.presentation import (
    gate_untrusted_model_output,
    sanitize_visible_model_text,
)
from anydes_tools.settings.store import SettingsStore


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "anydes_tools"


def _normalized(value: str) -> str:
    import unicodedata

    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _semantic_content_fixture(request):
    description = _normalized(request.description)
    if "dos cifras" in description:
        values = ["15", "20", "25", "31", "42", "50", "61", "72", "85", "99",
                  "10", "11", "12", "13", "14", "16", "17", "18", "19", "21"]
    elif _normalized(request.semantic_type).strip() == "colores":
        values = [
            "Rojo", "Azul", "Amarillo", "Verde", "Gris", "Plata", "Naranja",
            "Violeta", "Rosa", "Negro", "Blanco", "Cian", "Magenta", "Turquesa",
            "Dorado", "Coral", "Oliva", "Lima", "Índigo", "Salmón",
        ]
    else:
        values = [f"{request.semantic_type} {index}" for index in range(1, request.count + 1)]
    values = values[:request.count]
    if "similar al rojo" in _normalized(request.description):
        values[0] += " - Este color es similar a rojo"
    return values


def _semantic_line_fixture(predicate, lines):
    condition = _normalized(predicate.description)
    matches: list[int] = []
    for index, line in enumerate(lines, 1):
        value = _normalized(line)
        if "primari" in condition and value.strip() in {"rojo", "azul", "amarillo"}:
            matches.append(index)
        elif "gris" in condition and any(term in value for term in ("gris", "plata")):
            matches.append(index)
        elif "conclusion" in condition and any(term in value for term in ("por tanto", "en conclusion")):
            matches.append(index)
        elif "urgent" in condition and any(term in value for term in ("urgente", "vence hoy")):
            matches.append(index)
    return matches


def _semantic_prose_fixture(specification):
    seed = (
        f"{specification.content_kind} sobre {specification.topic} "
        f"para {specification.purpose} en español"
    ).split()
    words = list(seed)
    while len(words) < specification.target_length_words:
        words.extend(("contenido", "claro", "coherente", "relevante"))
    return " ".join(words[:specification.target_length_words])


class ToolConversationV0015Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = TOOLS / "workspaces" / f"test_conversation_{uuid4().hex[:10]}"
        self.workspace.mkdir(parents=True)
        self.settings = SettingsStore(
            ROOT,
            TOOLS,
            settings_path=self.workspace / "settings.json",
        )
        self.settings.set_temporary_workspace(self.workspace)
        self.controller = ToolConversationController(
            ROOT,
            TOOLS,
            settings=self.settings,
            semantic_evaluator=_semantic_line_fixture,
            semantic_content_generator=_semantic_content_fixture,
            semantic_prose_generator=_semantic_prose_fixture,
        )
        self.created_siblings: list[Path] = []
        self.turn = 0

    def tearDown(self) -> None:
        self.controller.close()
        shutil.rmtree(self.workspace, ignore_errors=True)
        for path in self.created_siblings:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)

    def ask(self, text: str):
        self.turn += 1
        return self.controller.handle(text, turn_id=str(self.turn))

    def create_and_confirm(self, text: str, confirmation: str = "sí"):
        response = self.ask(text)
        if response.executed:
            return response
        self.assertIsNotNone(self.controller.state.pending_tool_action)
        return self.ask(confirmation)

    def test_01_reasoning_only_does_not_use_tools(self):
        self.assertIsNone(self.ask("¿Qué es un PDF?"))

    def test_02_create_txt_and_read_conversational_referent(self):
        created = self.create_and_confirm("créame un txt que diga hola", "Sí, está bien")
        self.assertTrue(created.executed)
        path = Path(self.controller.state.active_document)
        self.assertEqual(path.read_text(encoding="utf-8"), "hola")
        read = self.ask("ahora léelo")
        self.assertTrue(read.executed)
        self.assertIn("hola", read.text)

    def test_03_edit_active_document(self):
        self.create_and_confirm("créame un txt que diga primera")
        edited = self.ask("edítalo y agrega otra línea segunda")
        self.assertTrue(edited.executed)
        content = Path(self.controller.state.active_document).read_text(encoding="utf-8")
        self.assertIn("primera", content)
        self.assertIn("segunda", content)

    def test_04_create_word_and_pdf(self):
        word = self.create_and_confirm("créame un Word con este texto hola Word", "está bien")
        self.assertTrue(word.executed)
        self.assertEqual(Path(self.controller.state.active_document).suffix, ".docx")
        pdf = self.create_and_confirm("hazme un PDF con este texto hola PDF", "adelante")
        self.assertTrue(pdf.executed)
        self.assertTrue(Path(self.controller.state.active_document).read_bytes().startswith(b"%PDF"))

    def test_05_convert_active_document_to_pdf(self):
        self.create_and_confirm("créame un txt que diga contenido convertible")
        source = Path(self.controller.state.active_document)
        converted = self.ask("ahora conviértelo en PDF")
        self.assertTrue(converted.executed)
        destination = source.with_suffix(".pdf")
        self.assertTrue(destination.is_file())

    def test_06_search_read_and_join_txt_files(self):
        (self.workspace / "a.txt").write_text("uno", encoding="utf-8")
        (self.workspace / "b.txt").write_text("dos", encoding="utf-8")
        joined = self.ask("busca los txt de esta carpeta y únelos en PDF")
        self.assertTrue(joined.executed)
        self.assertTrue((self.workspace / "textos_unidos.pdf").is_file())
        self.assertGreaterEqual(joined.debug["steps"], 4)

    def test_07_compress_folder(self):
        (self.workspace / "archivo.txt").write_text("zip", encoding="utf-8")
        archive = self.ask("comprime esta carpeta en zip")
        expected = self.workspace.parent / f"{self.workspace.name}.zip"
        self.created_siblings.append(expected)
        self.assertTrue(archive.executed)
        self.assertTrue(expected.is_file())

    def test_08_python_requires_confirmation_and_runs_in_sandbox(self):
        pending = self.ask("ejecuta este código Python: print('bien')")
        self.assertFalse(pending.executed)
        self.assertIn("sandbox", pending.text)
        result = self.ask("sí")
        self.assertTrue(result.executed)
        self.assertIn("bien", result.text)

    def test_09_python_error_can_be_inspected(self):
        self.ask("ejecuta este código Python: raise ValueError('fallo')")
        failed = self.ask("sí")
        self.assertFalse(failed.executed)
        inspected = self.ask("ahora dime qué error tiene")
        self.assertTrue(inspected.executed)
        self.assertIn("error", inspected.text.casefold())

    def test_10_delete_requires_strong_confirmation(self):
        self.create_and_confirm("créame un txt que diga temporal")
        path = Path(self.controller.state.active_document)
        pending = self.ask("borra este archivo")
        self.assertTrue(path.exists())
        self.assertIn("no se puede deshacer", pending.text)
        ambiguous = self.ask("see")
        self.assertFalse(ambiguous.executed)
        deleted = self.ask("sí confirmo")
        self.assertTrue(deleted.executed)
        self.assertFalse(path.exists())

    def test_11_missing_capability_does_not_generate_tool(self):
        result = self.ask("corta los últimos 3 segundos de este video")
        self.assertEqual(result.resolver_kind, CapabilityResolutionKind.MISSING_CAPABILITY)
        self.assertIn("no tengo todavía", result.text.casefold())

    def test_12_core_write_is_forbidden(self):
        result = self.ask("modifica tu Core y cambia el router")
        self.assertEqual(result.resolver_kind, CapabilityResolutionKind.FORBIDDEN)
        self.assertIn("núcleo protegido", result.text)

    def test_13_external_scope_requires_narrow_authorization(self):
        with tempfile.TemporaryDirectory(prefix="anydes-tools-conversation-") as directory:
            path = Path(directory) / "externo.txt"
            pending = self.ask(f'créame un txt en "{path}" que diga externo')
            self.assertFalse(path.exists())
            self.assertIn("autorizarla", pending.text)
            created = self.ask("sí")
            self.assertTrue(created.executed)
            self.assertEqual(path.read_text(encoding="utf-8"), "externo")
            self.assertIn(str(path.parent.resolve()), self.controller.state.authorized_scopes)
            self.assertIn(str(self.workspace.resolve()), self.controller.state.authorized_scopes)

    def test_14_overwrite_requires_confirmation(self):
        path = self.workspace / "fijo.txt"
        path.write_text("anterior", encoding="utf-8")
        pending = self.ask(f'crea un txt en "{path}" que diga nuevo')
        self.assertIn("sobrescribir", pending.text)
        self.assertEqual(path.read_text(encoding="utf-8"), "anterior")
        completed = self.ask("sí")
        self.assertTrue(completed.executed)
        self.assertEqual(path.read_text(encoding="utf-8"), "nuevo")

    def test_15_pending_action_can_be_cancelled(self):
        self.ask("ejecuta este código Python: print('no')")
        cancelled = self.ask("no")
        self.assertIn("no lo hago", cancelled.text)
        self.assertIsNone(self.controller.state.pending_tool_action)

    def test_16_debug_log_is_structured_and_redacted(self):
        self.ask("créame un txt que diga secreto de prueba")
        rows = [json.loads(line) for line in (TOOLS / "logs" / "conversation-v001_5.jsonl").read_text(encoding="utf-8").splitlines()]
        intent_rows = [row for row in rows if row.get("event") == "conversation_tool_intent"]
        self.assertTrue(intent_rows)
        serialized = json.dumps(intent_rows[-1], ensure_ascii=False)
        self.assertNotIn("secreto de prueba", serialized)
        self.assertIn("resolver_result", serialized)

    def test_17_concrete_capability_phrasing_executes_safe_create(self):
        created = self.ask(
            "Anydes puedes crear un archivo de texto en el que enumeres del 1 al 100 con '-' como espacio?"
        )
        self.assertTrue(created.executed)
        path = Path(self.controller.state.active_document)
        lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0], "1 -")
        self.assertEqual(lines[-1], "100 -")
        self.assertEqual(len(lines), 100)
        self.assertNotIn(str(path.parent), created.text)
        self.assertIn(path.parent.name, created.text)

    def test_18_safe_create_never_requests_second_confirmation(self):
        for index in range(1, 6):
            with self.subTest(index=index):
                result = self.ask(f"crea un txt que diga variante {index}")
                self.assertTrue(result.executed)
                self.assertIsNone(self.controller.state.pending_tool_action)

    def test_19_new_topic_supersedes_pending_and_later_yes_does_not_execute(self):
        pending = self.ask("ejecuta este código Python: print('no debe aparecer')")
        self.assertFalse(pending.executed)
        self.assertIsNone(self.ask("hablemos de otra cosa"))
        self.assertIsNone(self.controller.state.pending_tool_action)
        self.assertIsNone(self.ask("sí"))

    def test_20_windows_path_survives_math_chat_normalization(self):
        import sys

        program = ROOT / "programa_principal"
        if str(program) not in sys.path:
            sys.path.insert(0, str(program))
        from anydes.continuity_v2.math_presentation import normalize_chat
        from anydes_tools.conversation.presentation import normalize_chat_preserving_paths

        path = str(self.workspace / "Carpeta de trabajo de Anydes" / "nota.txt")
        rendered = normalize_chat_preserving_paths(
            f"Listo, creé el archivo en `{path}`.",
            normalize_chat,
        )
        self.assertIn(f"`{path}`", rendered)

    def test_21_create_numeric_range_with_comma_separator(self):
        result = self.create_and_confirm(
            'crea un txt con números del 1 al 100 separados por ","'
        )
        self.assertTrue(result.executed)
        content = Path(self.controller.state.active_document).read_text(encoding="utf-8")
        self.assertEqual(content, ",".join(str(value) for value in range(1, 101)))
        self.assertEqual(result.debug["verification_status"], "VERIFIED")

    def test_22_create_numeric_range_with_semicolon_separator(self):
        result = self.create_and_confirm(
            'crea un archivo del 1 al 50 usando ";" como separador'
        )
        self.assertTrue(result.executed)
        content = Path(self.controller.state.active_document).read_text(encoding="utf-8")
        self.assertEqual(content, ";".join(str(value) for value in range(1, 51)))

    def test_23_create_words_with_custom_separator(self):
        result = self.create_and_confirm(
            'crea un txt con palabras alfa beta gamma separadas por " | "'
        )
        self.assertTrue(result.executed)
        content = Path(self.controller.state.active_document).read_text(encoding="utf-8")
        self.assertEqual(content, "alfa | beta | gamma")

    def test_24_edit_number_tokens_with_generic_contains_predicate(self):
        self.create_and_confirm(
            'crea un archivo del 1 al 200 usando ";" como separador'
        )
        path = Path(self.controller.state.active_document)
        edited = self.ask(
            'cada vez que un número contenga el 9, escribe "Aquí está el nueve"'
        )
        self.assertTrue(edited.executed)
        content = path.read_text(encoding="utf-8")
        expected_matches = sum("9" in str(value) for value in range(1, 201))
        self.assertEqual(content.count("Aquí está el nueve"), expected_matches)
        self.assertEqual(edited.debug["verification_status"], "VERIFIED")

    def test_25_edit_lines_matching_condition(self):
        path = self.workspace / "lineas.txt"
        path.write_text("todo bien\nerror uno\notra\nERROR dos\n", encoding="utf-8")
        self.controller.state.remember_file(str(path))
        result = self.ask('en cada línea que contenga "error", agrega "REVISAR"')
        self.assertTrue(result.executed)
        content = path.read_text(encoding="utf-8")
        self.assertIn("error uno REVISAR", content)
        self.assertIn("ERROR dos REVISAR", content)

    def test_26_replace_text_in_active_document(self):
        path = self.workspace / "reemplazo.txt"
        path.write_text("foo uno y foo dos", encoding="utf-8")
        self.controller.state.remember_file(str(path))
        result = self.ask('reemplaza "foo" por "bar" en este archivo')
        self.assertTrue(result.executed)
        self.assertEqual(path.read_text(encoding="utf-8"), "bar uno y bar dos")

    def test_27_prefix_matching_lines(self):
        path = self.workspace / "prefijos.txt"
        path.write_text("ok\nwarn uno\nWARN dos\n", encoding="utf-8")
        self.controller.state.remember_file(str(path))
        result = self.ask('a cada línea que contenga "warn", añade el prefijo "! "')
        self.assertTrue(result.executed)
        self.assertEqual(path.read_text(encoding="utf-8"), "ok\n! warn uno\n! WARN dos\n")

    def test_28_write_must_verify_before_reporting_success(self):
        path = self.workspace / "no_op.txt"
        path.write_text("foo", encoding="utf-8")
        self.controller.state.remember_file(str(path))
        original = self.controller._execute_capability

        def false_success(capability, arguments, **kwargs):
            if capability == "document.edit_document":
                return ToolExecutionResult(
                    request_id="false-success",
                    turn_id=kwargs["turn_id"],
                    tool_id="official.document",
                    capability=capability,
                    status=ExecutionStatus.SUCCESS,
                    output={"path": str(path), "edited": True},
                )
            return original(capability, arguments, **kwargs)

        with patch.object(self.controller, "_execute_capability", side_effect=false_success):
            result = self.ask('reemplaza "foo" por "bar" en este archivo')
        self.assertFalse(result.executed)
        self.assertNotIn("listo", result.text.casefold())
        self.assertEqual(result.debug["verification_status"], "VERIFY_FAILED")
        self.assertEqual(path.read_text(encoding="utf-8"), "foo")

    def test_29_failed_edit_never_claims_success(self):
        path = self.workspace / "fallo.txt"
        path.write_text("foo", encoding="utf-8")
        self.controller.state.remember_file(str(path))
        original = self.controller._execute_capability

        def fail_edit(capability, arguments, **kwargs):
            if capability == "document.edit_document":
                return ToolExecutionResult(
                    request_id="forced-edit-failure",
                    turn_id=kwargs["turn_id"],
                    tool_id="official.document",
                    capability=capability,
                    status=ExecutionStatus.FAILED,
                    error="forced_edit_failure",
                )
            return original(capability, arguments, **kwargs)

        with patch.object(self.controller, "_execute_capability", side_effect=fail_edit):
            result = self.ask('reemplaza "foo" por "bar" en este archivo')
        self.assertFalse(result.executed)
        self.assertNotIn("listo", result.text.casefold())
        self.assertNotIn("hecho", result.text.casefold())
        self.assertEqual(path.read_text(encoding="utf-8"), "foo")

    def test_30_retry_is_limited_to_one_attempt(self):
        path = self.workspace / "retry.txt"
        path.write_text("foo", encoding="utf-8")
        self.controller.state.remember_file(str(path))
        original = self.controller._execute_capability

        def fail_edit(capability, arguments, **kwargs):
            if capability == "document.edit_document":
                return ToolExecutionResult(
                    request_id="forced-retry-failure",
                    turn_id=kwargs["turn_id"],
                    tool_id="official.document",
                    capability=capability,
                    status=ExecutionStatus.FAILED,
                    error="forced_retry_failure",
                )
            return original(capability, arguments, **kwargs)

        with patch.object(self.controller, "_execute_capability", side_effect=fail_edit):
            first = self.ask('reemplaza "foo" por "bar" en este archivo')
            retry = self.ask("inténtalo de nuevo")
            limited = self.ask("inténtalo de nuevo")
        self.assertFalse(first.executed)
        self.assertFalse(retry.executed)
        self.assertFalse(limited.executed)
        self.assertTrue(limited.debug["retry_limited"])
        self.assertEqual(self.controller.state.retry_count, 1)
        self.assertEqual(path.read_text(encoding="utf-8"), "foo")

    def test_31_explicit_items_can_use_one_item_per_line(self):
        result = self.create_and_confirm(
            "crea un txt con a,b,c separados por salto de línea"
        )
        self.assertTrue(result.executed)
        path = Path(self.controller.state.active_document)
        self.assertEqual(path.suffix, ".txt")
        self.assertEqual(path.read_text(encoding="utf-8"), "a\nb\nc")

    def test_32_unquoted_semicolon_separator_is_preserved(self):
        result = self.create_and_confirm(
            "crea un txt del 1 al 100 separado por ;"
        )
        self.assertTrue(result.executed)
        content = Path(self.controller.state.active_document).read_text(encoding="utf-8")
        self.assertEqual(content, ";".join(str(value) for value in range(1, 101)))

    def test_33_markdown_with_three_headings_keeps_format_and_layout(self):
        result = self.create_and_confirm("crea un markdown con tres títulos")
        self.assertTrue(result.executed)
        path = Path(self.controller.state.active_document)
        self.assertEqual(path.suffix, ".md")
        self.assertEqual(
            path.read_text(encoding="utf-8"),
            "# Título 1\n# Título 2\n# Título 3",
        )

    def test_34_docx_with_literal_paragraph_keeps_requested_format(self):
        result = self.create_and_confirm(
            "crea un docx con este texto Este es el párrafo solicitado."
        )
        self.assertTrue(result.executed)
        path = Path(self.controller.state.active_document)
        self.assertEqual(path.suffix, ".docx")
        read = self.ask("ahora léelo")
        self.assertIn("Este es el párrafo solicitado.", read.text)

    def test_35_task_resume_executes_same_saved_plan(self):
        completed = self.ask(
            "crea un txt con a,b,c separados por salto de línea"
        )
        self.assertTrue(completed.executed)
        task = self.controller.state.pending_user_task
        plan = self.controller.state.pending_structured_plan
        self.assertIsNotNone(task)
        self.assertIsNotNone(plan)
        self.assertEqual(Path(self.controller.state.active_document).read_text(encoding="utf-8"), "a\nb\nc")
        self.assertEqual(self.controller.state.pending_user_task.task_id, task.task_id)
        self.assertEqual(self.controller.state.pending_structured_plan.plan_id, plan.plan_id)
        self.assertEqual(self.controller.state.pending_user_task.status, "COMPLETED")

    def test_36_wrong_extension_plan_is_rejected_before_execution(self):
        action = plan_from_text(
            "crea un txt que diga hola",
            self.controller.state,
        )
        self.assertIsNotNone(action)
        wrong_path = str(Path(action.arguments["path"]).with_suffix(".docx"))
        wrong_arguments = {**action.arguments, "path": wrong_path}
        wrong_plan = replace(action.structured_action_plan, arguments=wrong_arguments)
        invalid = replace(
            action,
            arguments=wrong_arguments,
            target_path=wrong_path,
            structured_action_plan=wrong_plan,
        )
        errors = validate_structured_action_plan(invalid, self.controller.registry)
        self.assertIn("extension_format_mismatch", errors)
        response = self.controller._resolve_and_prepare(invalid, turn_id="invalid-plan")
        self.assertFalse(response.executed)
        self.assertEqual(response.debug["plan_status"], "PLAN_INVALID")
        self.assertFalse(Path(wrong_path).exists())

    def test_37_model_tool_markup_is_never_visible(self):
        samples = (
            '[document.create_txt path="nota.txt" content="hola"]',
            '<tool_call>{"capability":"filesystem.read"}</tool_call>',
            '{"tool_id":"official.document","capability":"document.create_txt","arguments":{}}',
            '```json\n{"action_type":"EXACT_TOOL","arguments":{}}\n```',
        )
        for sample in samples:
            with self.subTest(sample=sample):
                visible, detected = sanitize_visible_model_text(sample)
                self.assertTrue(detected)
                self.assertNotIn("document.create", visible)
                self.assertNotIn("tool_call", visible)
                self.assertNotIn("tool_id", visible)

    def test_38_clear_action_does_not_fall_through_to_capability_list(self):
        response = self.ask(
            "puedes crear un .txt con todas las letras del abecedario, una por línea"
        )
        self.assertTrue(response.executed)
        self.assertNotIn("herramientas oficiales", response.text.casefold())
        self.assertIsNone(self.controller.state.pending_tool_action)
        self.assertEqual(
            self.controller.state.pending_user_task.output_format,
            "txt",
        )

    def test_39_alphabet_task_is_verified_after_creation(self):
        result = self.create_and_confirm(
            "puedes crear un .txt que contenga todas las letras del abecedario "
            "y que estén separadas con un enter? o sea, en cada línea una letra"
        )
        self.assertTrue(result.executed)
        path = Path(self.controller.state.active_document)
        self.assertEqual(path.name, "abecedario.txt")
        self.assertEqual(path.suffix, ".txt")
        self.assertEqual(path.read_text(encoding="utf-8").splitlines(), list("abcdefghijklmnopqrstuvwxyz"))
        self.assertEqual(self.controller.state.last_verification["expected_items"], 26)
        self.assertEqual(self.controller.state.last_verification["actual_items"], 26)
        self.assertTrue(self.controller.state.last_verification["layout_match"])

    def test_40_invalid_plan_does_not_enter_replan_loop(self):
        action = plan_from_text("crea un txt que diga hola", self.controller.state)
        invalid_plan = replace(action.structured_action_plan, capability="document.create_docx")
        invalid = replace(action, structured_action_plan=invalid_plan)
        response = self.controller._resolve_and_prepare(invalid, turn_id="no-loop")
        self.assertFalse(response.executed)
        self.assertEqual(self.controller.state.replan_count, 0)
        self.assertEqual(response.debug["plan_status"], "PLAN_INVALID")

    def test_41_human_color_request_creates_and_verifies_twenty_lines(self):
        completed = self.ask(
            "necesito que hagas un archivo .txt que enumere 20 colores diferentes, "
            "uno por línea"
        )
        self.assertIsNone(self.controller.state.pending_tool_action)
        path = Path(self.controller.state.active_document)
        self.assertTrue(completed.executed)
        self.assertEqual(path.name, "colores.txt")
        self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 20)
        self.assertEqual(completed.debug["response_source"], "VERIFIED_ACTION_RESULT")
        self.assertTrue(completed.debug["linked_request_id"])

    def test_42_color_condition_is_declarative_and_verified(self):
        completed = self.create_and_confirm(
            "haz un txt con 20 colores, uno por línea, y añade una nota cuando "
            "sea similar al rojo"
        )
        path = Path(self.controller.state.active_document)
        content = path.read_text(encoding="utf-8")
        conditions = self.controller.state.pending_user_task.content_requirements[
            "creation_spec"
        ]["conditions"]
        self.assertTrue(completed.executed)
        self.assertEqual(len(content.splitlines()), 20)
        self.assertEqual(conditions[0]["kind"], "SEMANTIC_GENERATION")
        self.assertIn("similar a rojo", content)

    def test_43_success_status_without_matching_file_effect_is_denied(self):
        action = plan_from_text("crea un txt que diga evidencia", self.controller.state)
        original = self.controller._execute_capability

        def omit_create_evidence(capability, arguments, **kwargs):
            if capability == "document.create_txt":
                Path(arguments["path"]).write_text(arguments["content"], encoding="utf-8")
                return ToolExecutionResult(
                    request_id="missing-effect",
                    turn_id=kwargs["turn_id"],
                    tool_id="official.document",
                    capability=capability,
                    status=ExecutionStatus.SUCCESS,
                    output={"path": arguments["path"]},
                )
            return original(capability, arguments, **kwargs)

        with patch.object(self.controller, "_execute_capability", side_effect=omit_create_evidence):
            result = self.controller._resolve_and_prepare(action, turn_id="missing-evidence")
        self.assertFalse(result.executed)
        self.assertEqual(result.debug["response_source"], "ACTION_RESULT_GATE_DENIED")
        self.assertNotIn("listo", result.text.casefold())
        self.assertNotIn("creé", result.text.casefold())
        Path(action.arguments["path"]).unlink(missing_ok=True)

    def test_44_model_protocol_and_unverified_success_claim_fail_closed(self):
        samples = (
            'Aquí tienes el archivo colores.txt. [document.create_txt name="colores.txt"]',
            'Listo, creé el documento.',
            '<tool_call>{"tool_id":"official.document"}</tool_call>',
        )
        for sample in samples:
            with self.subTest(sample=sample):
                visible, blocked, source = gate_untrusted_model_output(sample)
                self.assertTrue(blocked)
                self.assertEqual(source, "MODEL_ACTION_PROTOCOL_BLOCKED")
                self.assertNotIn("document.create", visible)
                self.assertNotIn("aquí tienes", visible.casefold())
                self.assertNotIn("listo", visible.casefold())

    def test_45_unresolved_operational_request_fails_closed(self):
        response = self.ask("escribe un archivo binario propietario con datos mágicos")
        self.assertIsNotNone(response)
        self.assertFalse(response.executed)
        self.assertEqual(
            response.debug["response_source"],
            "FAIL_CLOSED_UNRESOLVED_TOOL_REQUEST",
        )

    def test_46_operation_intent_is_classified_before_content(self):
        cases = {
            "crea un txt nuevo": DocumentOperation.CREATE_NEW,
            "lee este archivo": DocumentOperation.READ_EXISTING,
            "edita el archivo que acabas de crear": DocumentOperation.EDIT_EXISTING,
            "transforma las líneas de este archivo": DocumentOperation.TRANSFORM_EXISTING,
            "añade una línea al final": DocumentOperation.APPEND_EXISTING,
            "sobrescribe este archivo": DocumentOperation.OVERWRITE_EXISTING,
            "copia este archivo": DocumentOperation.COPY_EXISTING,
            "mueve este archivo": DocumentOperation.MOVE_EXISTING,
            "borra este archivo": DocumentOperation.DELETE_EXISTING,
            "convierte este archivo en pdf": DocumentOperation.CONVERT_EXISTING,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(classify_document_operation(text), expected)

    def test_47_edit_existing_never_creates_a_new_note(self):
        self.create_and_confirm(
            "crea un txt llamado colores version 2 con 20 colores diferentes, uno por línea"
        )
        target = Path(self.controller.state.active_document)
        before = target.read_text(encoding="utf-8")
        unexpected = self.workspace / "nota_2.txt"
        completed = self.ask(
            "necesito que edites el archivo que acabas de crear y cada vez que un color "
            "sea similar al gris o una de sus escalas escribas al lado "
            "'Este color es una escala de grises'"
        )
        after = target.read_text(encoding="utf-8")
        self.assertTrue(completed.executed)
        self.assertNotEqual(before, after)
        self.assertIn("Gris Este color es una escala de grises", after)
        self.assertIn("Plata Este color es una escala de grises", after)
        self.assertFalse(unexpected.exists())
        self.assertEqual(Path(self.controller.state.active_document), target)
        self.assertEqual(self.controller.state.last_verification["status"], "VERIFIED")

    def test_48_deterministic_starts_with_transform(self):
        path = self.workspace / "palabras.txt"
        path.write_text("Alfa\nbeta\nAvión\ngamma\n", encoding="utf-8")
        self.controller.state.remember_file(str(path))
        result = self.ask("a las palabras que empiezan por A añade '*' ")
        self.assertTrue(result.executed)
        self.assertEqual(
            path.read_text(encoding="utf-8"),
            "Alfa *\nbeta\nAvión *\ngamma\n",
        )

    def test_49_semantic_transformations_are_brain_selected_and_verified(self):
        fixtures = (
            (
                "colores.txt",
                "Rojo\nAzul\nAmarillo\nVerde\nGris\nPlata\n",
                "marca los colores primarios",
                3,
            ),
            (
                "grises.txt",
                "Rojo\nGris\nPlata\nAzul\n",
                "marca las tonalidades de gris",
                2,
            ),
            (
                "texto.txt",
                "Primera idea\nPor tanto, termina\nOtro dato\nEn conclusión, listo\n",
                "marca las líneas que parezcan conclusiones",
                2,
            ),
            (
                "tareas.txt",
                "Revisar luego\nUrgente: reparar\nVence hoy\nAlgún día\n",
                "marca las tareas que parezcan urgentes",
                2,
            ),
        )
        for filename, content, request, expected in fixtures:
            with self.subTest(request=request):
                path = self.workspace / filename
                path.write_text(content, encoding="utf-8")
                self.controller.state.remember_file(str(path))
                result = self.ask(request)
                self.assertTrue(result.executed)
                self.assertEqual(path.read_text(encoding="utf-8").count("*"), expected)
                self.assertEqual(self.controller.state.last_verification["expected_changes"], expected)
                self.assertEqual(self.controller.state.last_verification["actual_changes"], expected)
                self.assertNotEqual(
                    self.controller.state.last_verification["before_hash"],
                    self.controller.state.last_verification["after_hash"],
                )

    def test_50_pending_confirmation_wins_over_historical_topic_once(self):
        self.assertIsNone(self.ask("hablemos de la paradoja del abuelo"))
        pending = self.ask("ejecuta este código Python: print('prioridad operacional')")
        action = self.controller.state.pending_tool_action.action
        pending_id = self.controller.state.pending_tool_action.pending_request_id
        completed = self.ask("si dale")
        self.assertTrue(completed.executed)
        self.assertIn("prioridad operacional", completed.text)
        self.assertIsNone(self.controller.state.pending_tool_action)
        self.assertIsNone(self.ask("si dale"))
        rows = [
            json.loads(line)
            for line in (TOOLS / "logs" / "conversation-v001_5.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        consumed = [
            row for row in rows
            if row.get("event") == "pending_tool_resumed"
            and row.get("details", {}).get("pending_action_id") == pending_id
        ]
        self.assertEqual(len(consumed), 1)

    def test_51_semantic_verify_failure_never_claims_success(self):
        path = self.workspace / "semantic_fail.txt"
        path.write_text("Rojo\nAzul\nVerde\n", encoding="utf-8")
        self.controller.state.remember_file(str(path))
        original = self.controller._execute_capability

        def false_success(capability, arguments, **kwargs):
            if capability == "document.edit_document":
                return ToolExecutionResult(
                    request_id="semantic-false-success",
                    turn_id=kwargs["turn_id"],
                    tool_id="official.document",
                    capability=capability,
                    status=ExecutionStatus.SUCCESS,
                    output={"path": str(path), "edited": True},
                )
            return original(capability, arguments, **kwargs)

        with patch.object(self.controller, "_execute_capability", side_effect=false_success):
            result = self.ask("marca los colores primarios")
        self.assertFalse(result.executed)
        self.assertNotIn("listo", result.text.casefold())
        self.assertEqual(self.controller.state.last_verification["status"], "VERIFY_FAILED")

    def test_52_invalid_semantic_transform_fails_closed_without_new_file(self):
        path = self.workspace / "datos.txt"
        path.write_text("uno\ndos\n", encoding="utf-8")
        self.controller.state.remember_file(str(path))
        result = self.ask("edita el archivo de alguna manera creativa")
        self.assertFalse(result.executed)
        self.assertIn("no pude preparar", result.text.casefold())
        self.assertEqual(path.read_text(encoding="utf-8"), "uno\ndos\n")
        self.assertFalse((self.workspace / "nota.txt").exists())

    def test_53_create_docx_with_generated_three_hundred_word_summary(self):
        completed = self.create_and_confirm(
            "crea un Word donde escribas un resumen de unas 300 palabras sobre energía solar"
        )
        path = Path(self.controller.state.active_document)
        self.assertTrue(completed.executed)
        self.assertEqual(path.suffix, ".docx")
        extracted = str(self.controller.last_results[-1].output.get("content", ""))
        self.assertGreaterEqual(len(extracted.split()), 240)
        self.assertIn("energía solar", extracted.casefold())
        self.assertEqual(self.controller.state.last_verification.get("status", ""), "VERIFIED")

    def test_54_create_txt_with_generated_five_hundred_word_history(self):
        completed = self.create_and_confirm(
            "crea un TXT con un resumen de unas 500 palabras sobre historia de Internet"
        )
        path = Path(self.controller.state.active_document)
        content = path.read_text(encoding="utf-8")
        self.assertTrue(completed.executed)
        self.assertEqual(path.suffix, ".txt")
        self.assertEqual(len(content.split()), 500)
        self.assertIn("historia de internet", content.casefold())

    def test_55_generated_conclusion_edits_existing_docx(self):
        self.create_and_confirm(
            "crea un docx con este texto El documento explica una fuente de energía limpia."
        )
        path = Path(self.controller.state.active_document)
        before = path.read_bytes()
        result = self.ask("añade una conclusión sobre la energía limpia en este documento")
        self.assertTrue(result.executed)
        self.assertEqual(Path(self.controller.state.active_document), path)
        self.assertNotEqual(before, path.read_bytes())
        read = self.ask("ahora léelo")
        self.assertIn("conclusion sobre la energía limpia", read.text.casefold())

    def test_56_create_markdown_with_long_generated_explanation(self):
        result = self.create_and_confirm(
            "crea un Markdown con una explicación de 300 palabras sobre redes neuronales"
        )
        path = Path(self.controller.state.active_document)
        self.assertTrue(result.executed)
        self.assertEqual(path.suffix, ".md")
        self.assertEqual(len(path.read_text(encoding="utf-8").split()), 300)

    def test_57_generation_failure_creates_no_empty_file(self):
        def fail_generation(_specification):
            raise RuntimeError("forced_generation_failure")

        self.controller.semantic_prose_generator = fail_generation
        result = self.ask("crea un Word con un resumen de 300 palabras sobre energía eólica")
        path = Path(self.controller.state.last_failed_action.target_path)
        self.assertFalse(result.executed)
        self.assertFalse(path.exists())
        self.assertNotIn("listo", result.text.casefold())

    def test_58_empty_generated_content_fails_closed(self):
        self.controller.semantic_prose_generator = lambda _specification: ""
        result = self.ask("crea un TXT con un ensayo de 300 palabras sobre movilidad urbana")
        path = Path(self.controller.state.last_failed_action.target_path)
        self.assertFalse(result.executed)
        self.assertFalse(path.exists())
        self.assertEqual(result.debug["verification_status"], "")

    def test_59_delete_collection_freezes_targets_and_requires_confirmation(self):
        collection = self.workspace / "collection_all"
        collection.mkdir()
        self.controller.settings.set_temporary_workspace(collection)
        targets = [collection / "a.txt", collection / "b.docx", collection / "c.md"]
        for index, path in enumerate(targets):
            path.write_text(f"item {index}", encoding="utf-8")
        pending = self.ask("borra todos los archivos de tu carpeta")
        self.assertFalse(pending.executed)
        self.assertIn("Encontré 3 archivos", pending.text)
        frozen = self.controller.state.pending_tool_action.action.arguments["delete_plan"]
        self.assertEqual(len(frozen["targets"]), 3)
        late = collection / "creado_despues.txt"
        late.write_text("no estaba en el plan", encoding="utf-8")
        completed = self.ask("sí confirmo")
        self.assertTrue(completed.executed)
        self.assertTrue(late.exists())
        self.assertTrue(all(not path.exists() for path in targets))
        self.assertEqual(self.controller.state.last_verification["actual_deletions"], 3)

    def test_60_delete_collection_by_extension_filter(self):
        collection = self.workspace / "collection_filter"
        collection.mkdir()
        self.controller.settings.set_temporary_workspace(collection)
        txt = [collection / "uno.txt", collection / "dos.txt"]
        keep = collection / "tres.md"
        for path in (*txt, keep):
            path.write_text(path.name, encoding="utf-8")
        pending = self.ask("borra todos los .txt")
        self.assertIn("Encontré 2 archivos", pending.text)
        completed = self.ask("sí confirmo")
        self.assertTrue(completed.executed)
        self.assertTrue(all(not path.exists() for path in txt))
        self.assertTrue(keep.exists())

    def test_61_explicit_filename_beats_active_document(self):
        active = self.workspace / "colores.docx"
        explicit = self.workspace / "colores.txt"
        active.write_text("activo", encoding="utf-8")
        explicit.write_text("objetivo", encoding="utf-8")
        self.controller.state.remember_file(str(explicit))
        self.controller.state.remember_file(str(active))
        pending = self.ask("puedes ahora eliminar colores.txt?")
        action = self.controller.state.pending_tool_action.action
        self.assertEqual(Path(action.target_path), explicit)
        completed = self.ask("sí confirmo")
        self.assertTrue(completed.executed)
        self.assertFalse(explicit.exists())
        self.assertTrue(active.exists())

    def test_62_retry_reuses_failed_generated_task_once(self):
        attempts = 0

        def flaky(specification):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("temporary_generation_failure")
            return _semantic_prose_fixture(specification)

        self.controller.semantic_prose_generator = flaky
        first = self.ask("crea un Word con un resumen de 300 palabras sobre almacenamiento térmico")
        failed_action = self.controller.state.last_failed_action
        retried = self.ask("reinténtalo")
        self.assertFalse(first.executed)
        self.assertTrue(retried.executed)
        self.assertIsNotNone(failed_action)
        self.assertEqual(attempts, 2)
        self.assertTrue(Path(failed_action.target_path).exists())

    def test_63_delete_plan_refuses_a_target_changed_after_confirmation_prompt(self):
        collection = self.workspace / "collection_changed"
        collection.mkdir()
        self.controller.settings.set_temporary_workspace(collection)
        target = collection / "cambia.txt"
        target.write_text("contenido inicial", encoding="utf-8")
        pending = self.ask("borra todos los archivos del workspace")
        self.assertFalse(pending.executed)
        target.write_text("contenido reemplazado después del aviso", encoding="utf-8")
        result = self.ask("sí confirmo")
        self.assertFalse(result.executed)
        self.assertTrue(target.exists())
        self.assertNotIn("listo", result.text.casefold())
        self.assertEqual(
            self.controller.last_results[0].error,
            "delete_target_changed_since_confirmation",
        )

    def test_64_explicit_filename_with_spaces_is_preserved(self):
        result = self.create_and_confirm(
            "crea un Word llamado Informe Final con un resumen de 100 palabras sobre energía"
        )
        self.assertTrue(result.executed)
        self.assertEqual(Path(self.controller.state.active_document).name, "Informe Final.docx")

    def test_65_explicit_spanish_filename_is_not_translated(self):
        result = self.create_and_confirm(
            "crea un Word llamado Resumen Energía con un resumen de 100 palabras sobre energía"
        )
        self.assertTrue(result.executed)
        self.assertEqual(Path(self.controller.state.active_document).name, "Resumen Energía.docx")

    def test_66_generated_docx_has_default_title_and_paragraph_structure(self):
        self.create_and_confirm(
            "crea un Word llamado Informe Energía con un resumen de 400 palabras sobre energía"
        )
        extracted = str(self.controller.last_results[-1].output.get("content", ""))
        self.assertTrue(extracted.startswith("Informe Energía\n"))
        self.assertGreaterEqual(extracted.count("\n"), 4)

    def test_67_docx_to_pdf_preserves_basename_and_content(self):
        self.create_and_confirm(
            "crea un Word llamado Informe Solar con un resumen de 100 palabras sobre energía solar"
        )
        source = Path(self.controller.state.active_document)
        result = self.ask("convierte ese mismo documento a PDF")
        self.assertTrue(result.executed)
        target = Path(self.controller.state.active_document)
        self.assertEqual(target.name, "Informe Solar.pdf")
        self.assertTrue(target.stat().st_size > 0)
        self.assertEqual(self.controller.state.source_document, str(source.resolve()))
        self.assertGreaterEqual(self.controller.state.last_verification["content_similarity"], 0.90)

    def test_68_txt_to_pdf_uses_same_basename(self):
        self.create_and_confirm("crea Fuente.txt que diga uno dos tres")
        result = self.ask("transforma ese mismo documento en PDF")
        self.assertTrue(result.executed)
        self.assertEqual(Path(self.controller.state.active_document).name, "Fuente.pdf")

    def test_69_active_document_is_conversion_source(self):
        self.create_and_confirm("crea Activo.txt que diga contenido activo")
        source = Path(self.controller.state.active_document)
        self.ask("convierte ese documento a PDF")
        relation = self.controller.state.related_artifacts[-1]
        self.assertEqual(Path(relation["converted_from"]), source)

    def test_70_explicit_source_beats_wrong_active_document(self):
        self.create_and_confirm("crea Nombrado.txt que diga fuente nombrada")
        named = Path(self.controller.state.active_document)
        self.create_and_confirm("crea Otro.txt que diga documento activo diferente")
        result = self.ask("convierte Nombrado.txt a PDF")
        self.assertTrue(result.executed)
        self.assertEqual(Path(self.controller.state.source_document), named)
        self.assertEqual(Path(self.controller.state.active_document).name, "Nombrado.pdf")

    def test_71_missing_conversion_source_creates_no_empty_target(self):
        result = self.ask("convierte Inexistente.docx a PDF")
        self.assertFalse(result.executed)
        self.assertFalse((self.workspace / "Inexistente.pdf").exists())

    def test_72_deterministic_contains_marks_all_and_only_matches(self):
        result = self.create_and_confirm(
            'crea Enumeracion.txt con 20 números de dos cifras, uno por línea, y cada vez '
            'que veas un 5 escribe al lado "- Aquí hay un cinco"'
        )
        self.assertTrue(result.executed)
        lines = Path(self.controller.state.active_document).read_text(encoding="utf-8").splitlines()
        for line in lines:
            number = line.split()[0]
            self.assertEqual("Aquí hay un cinco" in line, "5" in number)
        self.assertIn("85 - Aquí hay un cinco", lines)
        self.assertTrue(self.controller.state.last_verification["predicate_match"])

    def test_73_verifier_detects_missing_deterministic_annotation(self):
        from anydes_tools.conversation.structured_text import verify_creation
        expected = "15 marca\n20\n85 marca"
        spec = {
            "item_count": 3,
            "layout": "lines",
            "separator": "\n",
            "conditions": [{
                "kind": "DETERMINISTIC_PREDICATE",
                "expected_match_ids": [1, 3],
                "mutation": {"text": "marca"},
            }],
        }
        verification = verify_creation("15 marca\n20\n85", expected, "txt", spec)
        self.assertEqual(verification["status"], "VERIFY_FAILED")
        self.assertEqual(verification["actual_match_ids"], [1])

    def test_74_single_repair_pass_can_fix_staged_creation(self):
        original = self.controller._execute_capability
        corrupted = {"done": False}

        def corrupt_once(capability, arguments, **kwargs):
            result = original(capability, arguments, **kwargs)
            path = Path(str(arguments.get("path", "")))
            if capability == "document.create_txt" and ".anydes-" in path.name and not corrupted["done"]:
                corrupted["done"] = True
                path.write_text(path.read_text(encoding="utf-8").replace(
                    "85 - Aquí hay un cinco", "85", 1
                ), encoding="utf-8")
            return result

        with patch.object(self.controller, "_execute_capability", side_effect=corrupt_once):
            result = self.create_and_confirm(
                'crea Reparado.txt con 20 números de dos cifras, uno por línea, y cada vez '
                'que veas un 5 escribe al lado "- Aquí hay un cinco"'
            )
        self.assertTrue(result.executed)
        self.assertEqual(self.controller.state.last_verification["repair_count"], 1)
        self.assertIn("85 - Aquí hay un cinco", (self.workspace / "Reparado.txt").read_text(encoding="utf-8"))

    def test_75_failed_repair_does_not_commit_final_artifact(self):
        original = self.controller._execute_capability

        def keep_corrupt(capability, arguments, **kwargs):
            result = original(capability, arguments, **kwargs)
            path = Path(str(arguments.get("path", "")))
            if path.exists() and ".anydes-" in path.name and capability in {
                "document.create_txt", "document.edit_document"
            }:
                content = path.read_text(encoding="utf-8").replace(
                    "85 - Aquí hay un cinco", "85", 1
                )
                path.write_text(content, encoding="utf-8")
            return result

        with patch.object(self.controller, "_execute_capability", side_effect=keep_corrupt):
            result = self.ask(
                'crea Fallido.txt con 20 números de dos cifras, uno por línea, y cada vez '
                'que veas un 5 escribe al lado "- Aquí hay un cinco"'
            )
        self.assertFalse(result.executed)
        self.assertFalse((self.workspace / "Fallido.txt").exists())
        self.assertEqual(self.controller.state.last_verification["repair_count"], 1)

    def test_76_explicit_filename_never_uses_semantic_kind(self):
        action = plan_from_text(
            "crea un Word llamado Mi Documento con un resumen de 100 palabras sobre redes",
            self.controller.state,
            content_builder=self.controller.content_builder,
        )
        self.assertEqual(Path(action.target_path).name, "Mi Documento.docx")
        self.assertNotEqual(Path(action.target_path).name, "summary.docx")


if __name__ == "__main__":
    unittest.main()
