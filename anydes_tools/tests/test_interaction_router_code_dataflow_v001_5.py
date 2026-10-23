from __future__ import annotations

from pathlib import Path
import shutil
import unittest
from uuid import uuid4
import zipfile

from anydes_tools.conversation.code_generation import GeneratedCodeArtifact
from anydes_tools.conversation.controller import ToolConversationController
from anydes_tools.conversation.failure_state import LastToolFailure
from anydes_tools.conversation.general_reasoning import GeneralTaskPlan
from anydes_tools.conversation.interaction_router import (
    InteractionCategory,
    InteractionDecision,
    RequestedOutcome,
    InteractionResolution, InteractionKind, TaskControl,
)
from anydes_tools.settings.store import SettingsStore


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "anydes_tools"


def plan(**overrides) -> GeneralTaskPlan:
    value = {
        "task_id": f"route:{uuid4().hex}",
        "goal": "complete bounded action",
        "task_type": "REASONING_ONLY",
        "inputs": {},
        "desired_outputs": [],
        "constraints": [],
        "required_capabilities": [],
        "steps": [],
        "artifact_requirements": {},
        "needs_content_generation": False,
        "needs_reasoning": True,
        "needs_tools": False,
        "risk": "none",
        "ambiguities": [],
        "success_criteria": [],
    }
    value.update(overrides)
    return GeneralTaskPlan.from_dict(value)


class QueueClassifier:
    def __init__(self, *values: InteractionDecision) -> None:
        self.values = list(values)

    def __call__(self, request):
        return self.values.pop(0)


class QueueReasoner:
    def __init__(self, *values: GeneralTaskPlan) -> None:
        self.values = list(values)
        self.calls = 0
        self.repairs = 0

    def __call__(self, request):
        self.calls += 1
        return self.values.pop(0)

    def repair(self, request, invalid, errors):
        self.repairs += 1
        return self.values.pop(0)


def decision(category, outcome=RequestedOutcome.NONE, *, content=False):
    return InteractionDecision(
        category,
        1.0,
        primary_goal="preserve dominant user goal",
        requested_outcome=outcome,
        needs_content_generation=content,
    )


class InteractionRouterAndCodeDataflowTests(unittest.TestCase):
    def setUp(self):
        self.workspace = TOOLS / "workspaces" / f"router_code_{uuid4().hex[:10]}"
        self.workspace.mkdir(parents=True)
        self.settings = SettingsStore(
            ROOT,
            TOOLS,
            settings_path=self.workspace / "settings.json",
        )
        self.settings.set_temporary_workspace(self.workspace)
        self.controllers = []

    def tearDown(self):
        for controller in self.controllers:
            controller.close()
        shutil.rmtree(self.workspace, ignore_errors=True)

    def controller(self, **kwargs):
        controller = ToolConversationController(
            ROOT,
            TOOLS,
            settings=self.settings,
            **kwargs,
        )
        self.controllers.append(controller)
        return controller

    def test_conversation_and_question_never_enter_action_reasoner(self):
        reasoner = QueueReasoner(plan(), plan())
        controller = self.controller(
            interaction_classifier=QueueClassifier(
                decision(InteractionCategory.CONVERSATION),
                decision(InteractionCategory.QUESTION),
            ),
            general_task_reasoner=reasoner,
        )
        self.assertIsNone(controller.handle("saludo social no lexical", turn_id="social"))
        self.assertIsNone(controller.handle("pregunta explicativa abierta", turn_id="question"))
        self.assertEqual(reasoner.calls, 0)

    def test_goal_consistency_repairs_secondary_domain_misclassification(self):
        invalid = plan(
            task_type="GENERATE_CONVERSATIONAL_CONTENT",
            inputs={"content_spec": {"topic": "weather"}},
            needs_content_generation=True,
        )
        repaired = plan(
            task_type="GENERATE_CODE",
            inputs={"code_spec": {
                "language": "python",
                "goal": "convert a user supplied measurement",
                "functional_requirements": ["read a value", "convert it", "print the result"],
                "inputs": ["numeric stdin"],
                "outputs": ["converted value"],
                "constraints": [],
                "dependencies": [],
                "target_filename": "convertidor.py",
                "tests": [{"test_id": "known", "stdin": "0\n", "expected_stdout_contains": ["32"]}],
                "execution_required": True,
                "user_interactive": True,
                "input_prompts": ["request the value with units"],
            }},
            required_capabilities=["developer.write_code", "developer.run_python"],
            steps=[
                {"step_id": "write", "capability": "developer.write_code", "purpose": "write", "produces_artifacts": ["code"]},
                {"step_id": "test", "capability": "developer.run_python", "purpose": "test", "depends_on": ["write"], "consumes_artifacts": ["code"]},
            ],
            artifact_requirements={"format": "python", "filename": "convertidor.py"},
            needs_content_generation=True,
            needs_tools=True,
            risk="high",
        )
        reasoner = QueueReasoner(invalid, repaired)
        generator = lambda spec: GeneratedCodeArtifact.from_dict({
            "language": "python",
            "filename": spec.target_filename,
            "code": (
                "try:\n"
                "    value = float(input('Ingresa el valor en grados Celsius: '))\n"
                "    print((value * 9 / 5) + 32)\n"
                "except ValueError:\n"
                "    print('Entrada inválida')\n"
            ),
            "tests": [{"test_id": "known", "stdin": "0\n", "expected_stdout_contains": ["32"]}],
        })
        controller = self.controller(
            interaction_classifier=QueueClassifier(
                decision(InteractionCategory.ACTION_REQUEST, RequestedOutcome.CODE_ARTIFACT, content=True),
                InteractionResolution(InteractionKind.TOOL_ACTION, task_control=TaskControl.CONFIRM, confidence=1),
            ),
            general_task_reasoner=reasoner,
            code_generator=generator,
        )
        pending = controller.handle("solicitud nueva de código con dominio secundario", turn_id="goal")
        self.assertEqual(reasoner.repairs, 1)
        self.assertIn("sandbox", pending.text)
        completed = controller.handle("sí", turn_id="goal-confirm")
        self.assertTrue(completed.executed)
        source = (self.workspace / "convertidor.py").read_text(encoding="utf-8")
        self.assertIn("Ingresa el valor", source)

    def test_existing_code_is_modified_tested_and_atomically_replaced(self):
        target = self.workspace / "convertidor.py"
        original = "value = float(input('Valor: '))\nprint((value * 9 / 5) + 32)\n"
        target.write_text(original, encoding="utf-8")
        modification = plan(
            task_type="MODIFY_CODE",
            inputs={"code_modification_spec": {
                "target_artifact": target.name,
                "language": "python",
                "requested_change": "support conversion in both directions",
                "preserve_requirements": ["keep the original conversion"],
                "new_requirements": ["accept a direction", "convert both ways"],
                "new_tests": [
                    {"test_id": "forward", "stdin": "C\n0\n", "expected_stdout_contains": ["32"]},
                    {"test_id": "reverse", "stdin": "F\n32\n", "expected_stdout_contains": ["0"]},
                ],
                "execution_required": True,
            }},
            required_capabilities=[
                "developer.read_code",
                "developer.write_code",
                "developer.run_python",
            ],
            steps=[
                {"step_id": "read", "capability": "developer.read_code", "purpose": "read", "produces_artifacts": ["source"]},
                {"step_id": "write", "capability": "developer.write_code", "purpose": "modify", "depends_on": ["read"], "consumes_artifacts": ["source"], "produces_artifacts": ["modified"]},
                {"step_id": "test", "capability": "developer.run_python", "purpose": "test", "depends_on": ["write"], "consumes_artifacts": ["modified"]},
            ],
            needs_content_generation=True,
            needs_tools=True,
            risk="high",
        )
        modified = GeneratedCodeArtifact.from_dict({
            "language": "python",
            "filename": target.name,
            "code": (
                "direction = input('Elige C o F: ').strip().upper()\n"
                "value = float(input('Ingresa la temperatura: '))\n"
                "print((value * 9 / 5) + 32 if direction == 'C' else (value - 32) * 5 / 9)\n"
            ),
            "tests": [],
        })
        controller = self.controller(
            interaction_classifier=QueueClassifier(
                decision(InteractionCategory.ACTION_FOLLOWUP, RequestedOutcome.CODE_ARTIFACT, content=True),
                InteractionResolution(InteractionKind.TOOL_ACTION, task_control=TaskControl.CONFIRM, confidence=1),
            ),
            general_task_reasoner=QueueReasoner(modification),
            code_modifier=lambda spec: modified,
        )
        controller.state.remember_file(str(target))
        pending = controller.handle("cambio funcional del programa existente", turn_id="modify")
        self.assertIn("sandbox", pending.text)
        completed = controller.handle("sí", turn_id="modify-confirm")
        self.assertTrue(completed.executed)
        self.assertNotEqual(target.read_text(encoding="utf-8"), original)
        self.assertEqual(controller.state.last_verification["status"], "VERIFIED")

    def test_code_archive_contains_only_bound_artifact(self):
        unrelated = self.workspace / "unrelated.txt"
        unrelated.write_text("must not be archived", encoding="utf-8")
        code_plan = plan(
            task_type="GENERATE_CODE",
            inputs={"code_spec": {
                "language": "python",
                "goal": "count matching characters",
                "functional_requirements": ["read text", "print count"],
                "inputs": ["text"],
                "outputs": ["count"],
                "constraints": [],
                "dependencies": [],
                "target_filename": "contador.py",
                "tests": [{"test_id": "smoke", "stdin": "aei\n", "expected_stdout_contains": ["3"]}],
                "execution_required": True,
            }},
            required_capabilities=["developer.write_code", "developer.run_python", "archive.compress_zip"],
            steps=[
                {"step_id": "write", "capability": "developer.write_code", "purpose": "write", "produces_artifacts": ["code"]},
                {"step_id": "test", "capability": "developer.run_python", "purpose": "test", "depends_on": ["write"], "consumes_artifacts": ["code"]},
                {"step_id": "archive", "capability": "archive.compress_zip", "purpose": "archive", "depends_on": ["test"], "consumes_artifacts": ["code"]},
            ],
            needs_content_generation=True,
            needs_tools=True,
            risk="high",
        )
        generator = lambda spec: GeneratedCodeArtifact.from_dict({
            "language": "python",
            "filename": spec.target_filename,
            "code": "text = input('Ingresa una frase: ')\nprint(sum(c in 'aeiou' for c in text.lower()))\n",
            "tests": [{"test_id": "smoke", "stdin": "aei\n", "expected_stdout_contains": ["3"]}],
        })
        controller = self.controller(
            interaction_classifier=QueueClassifier(
                decision(InteractionCategory.ACTION_REQUEST, RequestedOutcome.CODE_ARTIFACT, content=True),
                InteractionResolution(InteractionKind.TOOL_ACTION, task_control=TaskControl.CONFIRM, confidence=1),
            ),
            general_task_reasoner=QueueReasoner(code_plan),
            code_generator=generator,
        )
        controller.handle("bounded composed code request", turn_id="zip")
        completed = controller.handle("sí", turn_id="zip-confirm")
        self.assertTrue(completed.executed)
        with zipfile.ZipFile(self.workspace / "contador.zip") as archive:
            members = [Path(name).name for name in archive.namelist() if not name.endswith("/")]
        self.assertEqual(members, ["contador.py"])

    def test_semantic_failure_query_uses_recorded_source_empty_reason(self):
        controller = self.controller(
            interaction_classifier=QueueClassifier(
                decision(InteractionCategory.FAILURE_QUERY)
            )
        )
        controller.state.last_tool_failure = LastToolFailure.create(
            task_id="conversion",
            plan_id="plan-conversion",
            stage="VERIFY",
            error_code="DOCUMENT_CONVERSION_SOURCE_EMPTY",
            technical_message="document_conversion_source_empty",
            safe_user_message=(
                "el documento de origen no contenía texto extraíble, por eso no inicié la conversión"
            ),
            retryable=True,
        )
        response = controller.handle("referencia semántica al fallo anterior", turn_id="why")
        self.assertIn("no contenía texto extraíble", response.text)

    def test_empty_semantic_document_never_commits(self):
        document_plan = plan(
            task_type="GENERATE_DOCUMENT",
            inputs={"content_spec": {
                "content_kind": "project_summary",
                "topic": "bounded project context",
                "target_length_words": 100,
                "source_mode": "USER_CONTEXT",
            }},
            required_capabilities=["document.create_docx"],
            steps=[{"step_id": "write", "capability": "document.create_docx", "purpose": "write"}],
            artifact_requirements={"format": "docx", "filename": "Project Context.docx"},
            needs_content_generation=True,
            needs_tools=True,
        )
        controller = self.controller(
            interaction_classifier=QueueClassifier(
                decision(InteractionCategory.ACTION_REQUEST, RequestedOutcome.DOCUMENT_ARTIFACT, content=True)
            ),
            general_task_reasoner=QueueReasoner(document_plan),
            semantic_prose_generator=lambda spec: "",
            safe_context_provider=lambda kind, query: "PROVENANCE: USER_FACT",
        )
        response = controller.handle("semantic document request", turn_id="empty-doc")
        self.assertFalse(response.executed)
        self.assertFalse((self.workspace / "Project Context.docx").exists())


if __name__ == "__main__":
    unittest.main()
