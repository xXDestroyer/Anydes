from __future__ import annotations

from pathlib import Path
import shutil
import unittest
from uuid import uuid4

from anydes_tools.conversation.code_generation import GeneratedCodeArtifact
from anydes_tools.conversation.controller import ToolConversationController
from anydes_tools.conversation.failure_state import LastToolFailure
from anydes_tools.conversation.general_reasoning import (
    GeneralTaskPlan,
    GeneralTaskPlanValidator,
)
from anydes_tools.conversation.interaction_router import (
    InteractionCategory,
    InteractionDecision,
    RequestedOutcome,
    InteractionResolution, InteractionKind, TaskControl,
)
from anydes_tools.conversation.planner import plan_from_text
from anydes_tools.settings.store import SettingsStore


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "anydes_tools"


def _plan(**overrides) -> GeneralTaskPlan:
    value = {
        "task_id": f"general:{uuid4().hex}",
        "goal": "resolver la solicitud",
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
        "success_criteria": ["respuesta coherente"],
    }
    value.update(overrides)
    return GeneralTaskPlan.from_dict(value)


class QueueReasoner:
    def __init__(self, *plans: GeneralTaskPlan) -> None:
        self.plans = list(plans)
        self.requests: list[dict] = []
        self.repairs = 0

    def __call__(self, request: dict) -> GeneralTaskPlan:
        self.requests.append(request)
        return self.plans.pop(0)

    def repair(self, request, plan, errors):
        self.repairs += 1
        return self.plans.pop(0)


class QueueInteractionClassifier:
    def __init__(self, *decisions: InteractionDecision) -> None:
        self.decisions = list(decisions)

    def __call__(self, request: dict) -> InteractionDecision:
        return self.decisions.pop(0)


def _action_decision(outcome=RequestedOutcome.TOOL_RESULT):
    return InteractionDecision(
        InteractionCategory.ACTION_REQUEST,
        1.0,
        primary_goal="complete requested action",
        requested_outcome=outcome,
        needs_content_generation=outcome in {
            RequestedOutcome.CODE_ARTIFACT,
            RequestedOutcome.DOCUMENT_ARTIFACT,
        },
    )

def _prose(specification):
    seed = f"perfil usuario DemoUser proyecto Anydes {specification.topic}".split()
    words = list(seed)
    while len(words) < specification.target_length_words:
        words.extend(("contexto", "permitido", "memoria", "conversación"))
    return " ".join(words[:specification.target_length_words])


def _context(context_type: str, query: str) -> str:
    return (
        "PROVENANCE: USER_FACT, RETRIEVED_MEMORY, DIRECT_CONVERSATION\n"
        "DemoUser desarrolla el proyecto Anydes y prefiere validaciones verificables."
    )


class GeneralTaskReasonerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = TOOLS / "workspaces" / f"general_reasoner_{uuid4().hex[:10]}"
        self.workspace.mkdir(parents=True)
        self.settings = SettingsStore(ROOT, TOOLS, settings_path=self.workspace / "settings.json")
        self.settings.set_temporary_workspace(self.workspace)
        self.controllers: list[ToolConversationController] = []

    def tearDown(self) -> None:
        for controller in self.controllers:
            controller.close()
        shutil.rmtree(self.workspace, ignore_errors=True)

    def controller(
        self,
        reasoner,
        *,
        code_generator=None,
        interaction_classifier=None,
    ) -> ToolConversationController:
        controller = ToolConversationController(
            ROOT,
            TOOLS,
            settings=self.settings,
            semantic_prose_generator=_prose,
            interaction_classifier=(
                interaction_classifier
                or QueueInteractionClassifier(_action_decision())
            ),
            general_task_reasoner=reasoner,
            code_generator=code_generator,
            safe_context_provider=_context,
        )
        self.controllers.append(controller)
        return controller

    def test_conversational_report_does_not_require_format_or_destination(self):
        reasoner = QueueReasoner(_plan(
            goal="resumir el perfil permitido del usuario",
            task_type="GENERATE_CONVERSATIONAL_CONTENT",
            inputs={
                "content_spec": {
                    "content_kind": "USER_PROFILE_SUMMARY",
                    "topic": "perfil del usuario",
                    "purpose": "conversation",
                    "target_length_words": 200,
                    "language": "es",
                    "tone": "natural",
                    "structure": "paragraphs",
                    "constraints": ["ground_in_user_context"],
                    "source_mode": "USER_CONTEXT",
                }
            },
            needs_content_generation=True,
        ))
        controller = self.controller(
            reasoner,
            interaction_classifier=QueueInteractionClassifier(
                InteractionDecision(InteractionCategory.QUESTION, 1.0)
            ),
        )
        prompt = "haz un informe de unas 200 palabras sobre todo lo que sepas de mí"
        self.assertIsNone(plan_from_text(prompt, controller.state, content_builder=controller.content_builder))
        response = controller.handle(prompt, turn_id="report")
        self.assertIsNone(response)
        self.assertEqual(reasoner.requests, [])

    def test_generated_interactive_python_is_written_tested_and_committed(self):
        plan = _plan(
            goal="crear un programa Python según requisitos funcionales",
            task_type="GENERATE_CODE",
            inputs={"code_spec": {
                "language": "python",
                "goal": "leer un entero y mostrar una clasificación calculada",
                "functional_requirements": ["leer un entero", "calcular el resultado", "mostrarlo"],
                "inputs": ["entero por stdin"],
                "outputs": ["resultado por stdout"],
                "constraints": ["stdlib only"],
                "dependencies": [],
                "target_filename": "verificador.py",
                "tests": [
                    {"test_id": "case-a", "stdin": "2\n", "expected_stdout_contains": ["PRIME"]},
                    {"test_id": "case-b", "stdin": "4\n", "expected_stdout_contains": ["NOT_PRIME"]},
                ],
                "execution_required": True,
            }},
            required_capabilities=["developer.write_code", "developer.run_python"],
            steps=[
                {"step_id": "write", "capability": "developer.write_code", "purpose": "guardar código"},
                {"step_id": "test", "capability": "developer.run_python", "purpose": "probar código", "depends_on": ["write"]},
            ],
            artifact_requirements={"format": "python", "filename": "verificador.py"},
            needs_content_generation=True,
            needs_tools=True,
            risk="high",
            success_criteria=["código válido", "pruebas pasan"],
        )

        def generate(specification):
            return GeneratedCodeArtifact.from_dict({
                "language": "python",
                "filename": specification.target_filename,
                "code": (
                    "n = int(input().strip())\n"
                    "es = n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))\n"
                    "print('PRIME' if es else 'NOT_PRIME')\n"
                ),
                "tests": [
                    {"test_id": "case-a", "stdin": "2\n", "expected_stdout_contains": ["PRIME"]},
                    {"test_id": "case-b", "stdin": "4\n", "expected_stdout_contains": ["NOT_PRIME"]},
                ],
            })

        controller = self.controller(
            QueueReasoner(plan),
            code_generator=generate,
            interaction_classifier=QueueInteractionClassifier(
                _action_decision(RequestedOutcome.CODE_ARTIFACT),
                InteractionResolution(InteractionKind.TOOL_ACTION, task_control=TaskControl.CONFIRM, confidence=1),
            ),
        )
        pending = controller.handle(
            "haz un programa en Python que te pida ingresar un número y verifique una propiedad",
            turn_id="code-plan",
        )
        self.assertFalse(pending.executed)
        self.assertIn("sandbox", pending.text)
        completed = controller.handle("sí", turn_id="code-confirm")
        self.assertTrue(completed.executed)
        target = self.workspace / "verificador.py"
        self.assertTrue(target.is_file())
        self.assertIn("input()", target.read_text(encoding="utf-8"))
        self.assertEqual(controller.state.last_verification["status"], "VERIFIED")
        self.assertEqual(controller.state.last_verification["test_count"], 2)

    def test_code_then_zip_is_a_bounded_composition(self):
        plan = _plan(
            goal="crear, probar y comprimir un programa",
            task_type="GENERATE_CODE",
            inputs={"code_spec": {
                "language": "python",
                "goal": "emitir un saludo",
                "functional_requirements": ["imprimir saludo"],
                "inputs": [],
                "outputs": ["saludo"],
                "constraints": [],
                "dependencies": [],
                "target_filename": "saludo.py",
                "tests": [{"test_id": "smoke", "expected_stdout_contains": ["hola"]}],
                "execution_required": True,
            }},
            required_capabilities=["developer.write_code", "developer.run_python", "archive.compress_zip"],
            steps=[
                {"step_id": "write", "capability": "developer.write_code", "purpose": "guardar", "produces_artifacts": ["code"]},
                {"step_id": "test", "capability": "developer.run_python", "purpose": "probar", "depends_on": ["write"], "consumes_artifacts": ["code"]},
                {"step_id": "zip", "capability": "archive.compress_zip", "purpose": "comprimir", "depends_on": ["test"], "consumes_artifacts": ["code"]},
            ],
            artifact_requirements={"format": "python", "filename": "saludo.py", "archive": True},
            needs_content_generation=True,
            needs_tools=True,
            risk="high",
        )
        generator = lambda spec: GeneratedCodeArtifact.from_dict({
            "language": "python", "filename": spec.target_filename,
            "code": "print('hola')\n", "tests": [
                {"test_id": "smoke", "expected_stdout_contains": ["hola"]}
            ],
        })
        controller = self.controller(
            QueueReasoner(plan),
            code_generator=generator,
            interaction_classifier=QueueInteractionClassifier(
                _action_decision(RequestedOutcome.CODE_ARTIFACT),
                InteractionResolution(InteractionKind.TOOL_ACTION, task_control=TaskControl.CONFIRM, confidence=1),
            ),
        )
        controller.handle("crea ese código, pruébalo y comprímelo", turn_id="zip-plan")
        response = controller.handle("sí", turn_id="zip-confirm")
        self.assertTrue(response.executed)
        self.assertTrue((self.workspace / "saludo.py").is_file())
        self.assertTrue((self.workspace / "saludo.zip").is_file())

    def test_invalid_brain_plan_gets_only_one_repair(self):
        invalid = _plan(
            task_type="GENERATE_CODE",
            needs_content_generation=True,
            needs_tools=True,
            required_capabilities=["core.modify"],
            steps=[{"step_id": "bad", "capability": "core.modify", "purpose": "write"}],
        )
        repaired = _plan(task_type="REASONING_ONLY")
        reasoner = QueueReasoner(invalid, repaired)
        controller = self.controller(
            reasoner,
            interaction_classifier=QueueInteractionClassifier(_action_decision()),
        )
        response = controller.handle("una petición nueva", turn_id="repair")
        self.assertFalse(response.executed)
        self.assertIn("aclarar", response.text)
        self.assertEqual(reasoner.repairs, 1)

    def test_incomplete_code_spec_is_repaired_before_execution(self):
        invalid = _plan(
            goal="crear una utilidad",
            task_type="GENERATE_CODE",
            inputs={"code_spec": {
                "language": "python",
                "description": "especificación con contrato ajeno",
                "target_filename": "utilidad.py",
                "tests": [],
                "execution_required": True,
            }},
            required_capabilities=["developer.write_code", "developer.run_python"],
            steps=[
                {"step_id": "write", "capability": "developer.write_code", "purpose": "guardar"},
                {"step_id": "run", "capability": "developer.run_python", "purpose": "probar", "depends_on": ["write"]},
            ],
            needs_content_generation=True,
            needs_tools=True,
        )
        repaired = _plan(
            goal="crear una utilidad",
            task_type="GENERATE_CODE",
            inputs={"code_spec": {
                "language": "python",
                "goal": "imprimir un resultado verificable",
                "functional_requirements": ["imprimir resultado"],
                "inputs": [],
                "outputs": ["resultado"],
                "constraints": [],
                "dependencies": [],
                "target_filename": "utilidad.py",
                "tests": [{"test_id": "smoke", "expected_stdout_contains": ["ok"]}],
                "execution_required": True,
            }},
            required_capabilities=["developer.write_code", "developer.run_python"],
            steps=[
                {"step_id": "write", "capability": "developer.write_code", "purpose": "guardar"},
                {"step_id": "run", "capability": "developer.run_python", "purpose": "probar", "depends_on": ["write"]},
            ],
            needs_content_generation=True,
            needs_tools=True,
            risk="high",
        )
        reasoner = QueueReasoner(invalid, repaired)
        generator = lambda spec: GeneratedCodeArtifact.from_dict({
            "language": "python",
            "filename": spec.target_filename,
            "code": "print('ok')\n",
            "tests": [{"test_id": "smoke", "expected_stdout_contains": ["ok"]}],
        })
        controller = self.controller(
            reasoner,
            code_generator=generator,
            interaction_classifier=QueueInteractionClassifier(
                _action_decision(RequestedOutcome.CODE_ARTIFACT)
            ),
        )
        response = controller.handle("prepara una utilidad nueva", turn_id="nested-repair")
        self.assertIn("sandbox", response.text)
        self.assertEqual(reasoner.repairs, 1)
        self.assertFalse((self.workspace / "utilidad.py").exists())

    def test_plan_validator_rejects_unbounded_and_core_steps(self):
        controller = self.controller(QueueReasoner(_plan()))
        steps = [
            {"step_id": f"s{i}", "capability": "core.modify", "purpose": "bad"}
            for i in range(9)
        ]
        plan = _plan(
            task_type="TOOL_COMPOSITION",
            required_capabilities=["core.modify"],
            steps=steps,
            needs_tools=True,
        )
        validation = GeneralTaskPlanValidator(
            controller.registry, controller.resolver, controller.boundary
        ).validate(plan, authorized_scopes=())
        self.assertFalse(validation.valid)
        self.assertIn("step_limit_exceeded", validation.errors)
        self.assertTrue(any("capability_forbidden" in item for item in validation.errors))

    def test_delete_collection_skips_word_lock_and_reports_partial(self):
        user_file = self.workspace / "informe.docx"
        lock_file = self.workspace / "~$informe.docx"
        user_file.write_bytes(b"doc")
        lock_file.write_bytes(b"lock")
        controller = self.controller(QueueReasoner(_plan()),
            interaction_classifier=QueueInteractionClassifier(
                _action_decision(),
                InteractionResolution(InteractionKind.TOOL_ACTION, task_control=TaskControl.CONFIRM, confidence=1),
            ))
        pending = controller.handle("borra todos los archivos del workspace", turn_id="delete-plan")
        self.assertIn("¿Quieres", pending.text)
        response = controller.handle("sí confirmo", turn_id="delete-confirm")
        self.assertTrue(response.executed)
        self.assertFalse(user_file.exists())
        self.assertTrue(lock_file.exists())
        self.assertEqual(controller.state.last_verification["status"], "VERIFIED_PARTIAL")
        self.assertIn("temporales", response.text)

    def test_last_failure_survives_casual_turn(self):
        controller = self.controller(
            QueueReasoner(_plan(task_type="REASONING_ONLY")),
            interaction_classifier=QueueInteractionClassifier(
                InteractionDecision(InteractionCategory.CONVERSATION, 1.0),
                InteractionDecision(InteractionCategory.FAILURE_QUERY, 1.0),
            ),
        )
        controller.state.last_tool_failure = LastToolFailure.create(
            task_id="failed-delete",
            stage="EXECUTION",
            error_code="FILE_IN_USE",
            safe_user_message="un archivo estaba en uso",
            retryable=True,
        )
        self.assertIsNone(controller.handle("todo bien, relax", turn_id="casual"))
        response = controller.handle("¿qué pasó con el borrado?", turn_id="why")
        self.assertIn("archivo estaba en uso", response.text)

    def test_holdout_paraphrases_use_general_reasoner_without_new_parser_rules(self):
        prompts = (
            "Prepárame una utilidad que convierta una temperatura ingresada a otra escala.",
            "Necesito una pequeña aplicación que cuente las vocales de lo que escriba.",
            "Redacta una explicación breve de lo que recuerdas de nuestro proyecto.",
        )
        plans = [
            _plan(
                goal=f"generar código para objetivo holdout {index}",
                task_type="GENERATE_CODE",
                inputs={"code_spec": {
                    "language": "python",
                    "goal": f"objetivo funcional {index}",
                    "functional_requirements": ["procesar entrada y producir salida"],
                    "inputs": ["texto"],
                    "outputs": ["resultado"],
                    "constraints": ["stdlib only"],
                    "dependencies": [],
                    "target_filename": f"holdout_{index}.py",
                    "tests": [],
                    "execution_required": False,
                }},
                required_capabilities=["developer.write_code"],
                steps=[{
                    "step_id": "write",
                    "capability": "developer.write_code",
                    "purpose": "guardar el código generado",
                }],
                artifact_requirements={"format": "python", "filename": f"holdout_{index}.py"},
                needs_content_generation=True,
                needs_tools=True,
                risk="low",
            )
            for index in (1, 2)
        ]
        plans.append(_plan(
            goal="explicar el proyecto usando contexto permitido",
            task_type="GENERATE_CONVERSATIONAL_CONTENT",
            inputs={"content_spec": {
                "content_kind": "PROJECT_SUMMARY",
                "topic": "proyecto Anydes",
                "purpose": "conversation",
                "target_length_words": 120,
                "source_mode": "USER_CONTEXT",
            }},
            needs_content_generation=True,
        ))
        reasoner = QueueReasoner(*plans)

        def generator(specification):
            return GeneratedCodeArtifact(
                language="python",
                filename=specification.target_filename,
                code="print('resultado')\n",
                tests=(),
            )

        controller = self.controller(
            reasoner,
            code_generator=generator,
            interaction_classifier=QueueInteractionClassifier(
                _action_decision(RequestedOutcome.CODE_ARTIFACT),
                _action_decision(RequestedOutcome.CODE_ARTIFACT),
                InteractionDecision(InteractionCategory.QUESTION, 1.0),
            ),
        )
        for index, prompt in enumerate(prompts, 1):
            self.assertIsNone(
                plan_from_text(prompt, controller.state, content_builder=controller.content_builder)
            )
            response = controller.handle(prompt, turn_id=f"holdout-{index}")
            if index < 3:
                self.assertTrue(response.handled)
            else:
                self.assertIsNone(response)
        self.assertEqual(len(reasoner.requests), 2)
        self.assertTrue((self.workspace / "holdout_1.py").is_file())
        self.assertTrue((self.workspace / "holdout_2.py").is_file())

    def test_general_document_can_read_source_then_generate_verified_copy(self):
        source = self.workspace / "base.txt"
        source.write_text(
            "La energía solar reduce emisiones y complementa otras fuentes renovables.",
            encoding="utf-8",
        )
        reasoner = QueueReasoner(_plan(
            goal="crear una copia con una conclusión sobre energía solar",
            task_type="GENERATE_DOCUMENT",
            inputs={"content_spec": {
                "content_kind": "conclusion",
                "topic": "energía solar",
                "purpose": "document",
                "target_length_words": 80,
                "source_mode": "TRANSFORM_EXISTING",
            }},
            required_capabilities=["document.read_document", "document.create_txt"],
            steps=[
                {"step_id": "read", "capability": "document.read_document", "purpose": "leer la fuente"},
                {"step_id": "write", "capability": "document.create_txt", "purpose": "crear la copia", "depends_on": ["read"]},
            ],
            artifact_requirements={
                "format": "txt",
                "filename": "base con conclusion.txt",
                "source_filename": "base.txt",
            },
            needs_content_generation=True,
            needs_tools=True,
            risk="low",
        ))
        controller = self.controller(
            reasoner,
            interaction_classifier=QueueInteractionClassifier(
                _action_decision(RequestedOutcome.DOCUMENT_ARTIFACT)
            ),
        )
        controller.state.remember_file(str(source))
        response = controller.handle(
            "Produce una versión derivada del material activo incorporando un cierre argumental.",
            turn_id="source-transform",
        )
        self.assertTrue(response.executed)
        target = self.workspace / "base con conclusion.txt"
        self.assertTrue(target.is_file())
        self.assertGreater(len(target.read_text(encoding="utf-8").split()), 60)


if __name__ == "__main__":
    unittest.main()
