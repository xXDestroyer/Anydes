from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[2]
for _entry in (str(ROOT), str(ROOT / "programa_principal")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)
WORKSPACE = ROOT / "anydes_tools" / "workspaces" / "general-reasoner-e2e"
RESULT = ROOT / "anydes_tools" / "logs" / "general-task-reasoner-e2e-v001_5.json"


class AuditRecorder:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def record(self, event: str, **details) -> None:
        self.events.append({"event": event, "details": details})


class QueueReasoner:
    def __init__(self, plans) -> None:
        self.plans = list(plans)

    def __call__(self, request):
        return self.plans.pop(0)

    def repair(self, request, plan, errors):
        raise RuntimeError("controlled_e2e_does_not_expect_repair")


def run() -> dict:
    from probar_anydes_tools_conversacion_v001_5 import install_tool_conversation_runtime
    from anydes.nucleo.assistant import AnydesAssistant
    from anydes_tools.conversation.code_generation import GeneratedCodeArtifact
    from anydes_tools.conversation.general_reasoning import GeneralTaskPlan
    from anydes_tools.conversation.runtime_patch import ensure_controller_for

    shutil.rmtree(WORKSPACE, ignore_errors=True)
    WORKSPACE.mkdir(parents=True)
    install_tool_conversation_runtime()

    assistant = object.__new__(AnydesAssistant)
    assistant.audit = AuditRecorder()
    assistant._realtime_service = None
    assistant.last_action_results = ()
    assistant.last_intention = None
    controller = ensure_controller_for(assistant, ROOT)
    controller.audit = assistant.audit
    controller.executor.audit = assistant.audit
    controller.settings.set_temporary_workspace(WORKSPACE)
    controller.use_isolated_action_memory(WORKSPACE / ".action-memory")

    report_plan = GeneralTaskPlan.from_dict({
        "goal": "resumir información permitida del usuario",
        "task_type": "GENERATE_CONVERSATIONAL_CONTENT",
        "inputs": {"content_spec": {
            "content_kind": "USER_PROFILE_SUMMARY",
            "topic": "perfil del usuario",
            "purpose": "conversation",
            "target_length_words": 200,
            "language": "es",
            "tone": "natural",
            "structure": "paragraphs",
            "source_mode": "USER_CONTEXT",
        }},
        "desired_outputs": [{"type": "conversation_text"}],
        "constraints": ["ground_in_provided_context"],
        "required_capabilities": [],
        "steps": [],
        "artifact_requirements": {},
        "needs_content_generation": True,
        "needs_reasoning": True,
        "needs_tools": False,
        "risk": "none",
        "ambiguities": [],
        "success_criteria": ["roughly 200 words", "no invented facts"],
    })
    code_plan = GeneralTaskPlan.from_dict({
        "goal": "crear y probar un programa Python interactivo",
        "task_type": "GENERATE_CODE",
        "inputs": {"code_spec": {
            "language": "python",
            "goal": "leer un entero y determinar una propiedad matemática",
            "functional_requirements": ["leer stdin", "calcular", "mostrar resultado"],
            "inputs": ["entero"],
            "outputs": ["PRIME o NOT_PRIME"],
            "constraints": ["stdlib only"],
            "dependencies": [],
            "target_filename": "verificador_general.py",
            "tests": [
                {"test_id": "two", "stdin": "2\n", "expected_stdout_contains": ["PRIME"]},
                {"test_id": "four", "stdin": "4\n", "expected_stdout_contains": ["NOT_PRIME"]},
            ],
            "execution_required": True,
        }},
        "desired_outputs": [{"type": "python_file"}],
        "constraints": ["sandbox", "verify_before_commit"],
        "required_capabilities": ["developer.write_code", "developer.run_python"],
        "steps": [
            {"step_id": "write", "capability": "developer.write_code", "purpose": "guardar código"},
            {"step_id": "test", "capability": "developer.run_python", "purpose": "probar", "depends_on": ["write"]},
        ],
        "artifact_requirements": {"format": "python", "filename": "verificador_general.py"},
        "needs_content_generation": True,
        "needs_reasoning": True,
        "needs_tools": True,
        "risk": "high",
        "ambiguities": [],
        "success_criteria": ["tests pass", "artifact committed"],
    })
    controller.general_task_reasoner = QueueReasoner((report_plan, code_plan))
    controller.safe_context_provider = lambda kind, query: (
        "PROVENANCE: USER_FACT, RETRIEVED_MEMORY\n"
        "DemoUser desarrolla Anydes y prefiere resultados comprobables."
    )

    def prose(specification):
        words = f"perfil usuario DemoUser proyecto Anydes {specification.topic}".split()
        while len(words) < specification.target_length_words:
            words.extend(("contexto", "permitido", "evidencia", "verificable"))
        return " ".join(words[:specification.target_length_words])

    controller.semantic_prose_generator = prose
    controller.code_generator = lambda spec: GeneratedCodeArtifact.from_dict({
        "language": "python",
        "filename": spec.target_filename,
        "code": (
            "n = int(input().strip())\n"
            "prime = n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))\n"
            "print('PRIME' if prime else 'NOT_PRIME')\n"
        ),
        "tests": [
            {"test_id": "two", "stdin": "2\n", "expected_stdout_contains": ["PRIME"]},
            {"test_id": "four", "stdin": "4\n", "expected_stdout_contains": ["NOT_PRIME"]},
        ],
    })

    report_text = assistant.respond_realtime(
        "haz un informe de unas 200 palabras sobre todo lo que sepas de mí"
    )
    pending_text = assistant.respond_realtime(
        "haz un programa en Python que te pida ingresar un número y verifique si es primo"
    )
    completed_text = assistant.respond_realtime("sí")
    artifact = WORKSPACE / "verificador_general.py"
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "launcher": "Probar Anydes Tools Conversacion v001.5.bat",
        "same_bootstrap": True,
        "controlled_provider": True,
        "report_word_count": len(report_text.split()),
        "report_has_no_artifact_requirement": True,
        "code_confirmation_requested": "sandbox" in pending_text.casefold(),
        "code_response": completed_text,
        "artifact": str(artifact),
        "artifact_exists": artifact.is_file(),
        "verification": dict(controller.state.last_verification),
        "general_plan_events": sum(
            item["event"] == "general_task_plan" for item in assistant.audit.events
        ),
        "pseudo_tool_markup_visible": any(
            marker in (report_text + pending_text + completed_text)
            for marker in ("<tool_call", "[developer.", '"tool_id"')
        ),
    }
    RESULT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    controller.close()
    if not (
        payload["report_word_count"] == 200
        and payload["code_confirmation_requested"]
        and payload["artifact_exists"]
        and payload["verification"].get("status") == "VERIFIED"
        and not payload["pseudo_tool_markup_visible"]
    ):
        raise AssertionError(payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
