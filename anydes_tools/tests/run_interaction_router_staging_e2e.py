from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from probar_anydes_tools_conversacion_v001_5 import install_tool_conversation_runtime
from anydes.nucleo.assistant import AnydesAssistant
from anydes_tools.conversation.code_generation import GeneratedCodeArtifact
from anydes_tools.conversation.interaction_router import (
    InteractionCategory,
    InteractionDecision,
    InteractionRouter,
    RequestedOutcome,
)
from anydes_tools.conversation.runtime_patch import ensure_controller_for
from anydes_tools.tests.test_interaction_router_code_dataflow_v001_5 import (
    QueueClassifier,
    QueueReasoner,
    decision,
    plan,
)


WORKSPACE = ROOT / "anydes_tools" / "workspaces" / "interaction-router-e2e"
RESULT = ROOT / "anydes_tools" / "logs" / "interaction-router-e2e-v001_5.json"


class Audit:
    def __init__(self) -> None:
        self.events = []

    def record(self, event: str, **details) -> None:
        self.events.append({"event": event, "details": details})


def code_plan(*, modify: bool = False, archive: bool = False):
    if modify:
        return plan(
            task_type="MODIFY_CODE",
            inputs={"code_modification_spec": {
                "target_artifact": "conversor.py",
                "language": "python",
                "requested_change": "support both directions",
                "preserve_requirements": ["keep original behavior"],
                "new_requirements": ["select direction"],
                "new_tests": [
                    {"test_id": "c", "stdin": "C\n0\n", "expected_stdout_contains": ["32"]},
                    {"test_id": "f", "stdin": "F\n32\n", "expected_stdout_contains": ["0"]},
                ],
                "execution_required": True,
            }},
            required_capabilities=["developer.read_code", "developer.write_code", "developer.run_python"],
            steps=[
                {"step_id": "read", "capability": "developer.read_code", "purpose": "read", "produces_artifacts": ["source"]},
                {"step_id": "write", "capability": "developer.write_code", "purpose": "modify", "depends_on": ["read"], "consumes_artifacts": ["source"], "produces_artifacts": ["modified"]},
                {"step_id": "test", "capability": "developer.run_python", "purpose": "test", "depends_on": ["write"], "consumes_artifacts": ["modified"]},
            ],
            needs_content_generation=True,
            needs_tools=True,
            risk="high",
        )
    capabilities = ["developer.write_code", "developer.run_python"]
    steps = [
        {"step_id": "write", "capability": "developer.write_code", "purpose": "write", "produces_artifacts": ["code"]},
        {"step_id": "test", "capability": "developer.run_python", "purpose": "test", "depends_on": ["write"], "consumes_artifacts": ["code"]},
    ]
    if archive:
        capabilities.append("archive.compress_zip")
        steps.append({
            "step_id": "archive",
            "capability": "archive.compress_zip",
            "purpose": "archive only generated code",
            "depends_on": ["test"],
            "consumes_artifacts": ["code"],
        })
    return plan(
        task_type="GENERATE_CODE",
        inputs={"code_spec": {
            "language": "python",
            "goal": "convert values with a clear interactive prompt",
            "functional_requirements": ["read a value", "convert", "print"],
            "inputs": ["stdin"],
            "outputs": ["converted value"],
            "constraints": [],
            "dependencies": [],
            "target_filename": "conversor.py" if not archive else "contador.py",
            "tests": [{
                "test_id": "smoke",
                "stdin": "0\n" if not archive else "aei\n",
                "expected_stdout_contains": ["32"] if not archive else ["3"],
            }],
            "execution_required": True,
            "user_interactive": True,
            "input_prompts": ["describe expected input"],
        }},
        required_capabilities=capabilities,
        steps=steps,
        needs_content_generation=True,
        needs_tools=True,
        risk="high",
    )


def main() -> int:
    shutil.rmtree(WORKSPACE, ignore_errors=True)
    WORKSPACE.mkdir(parents=True)
    install_tool_conversation_runtime()
    assistant = object.__new__(AnydesAssistant)
    assistant.audit = Audit()
    assistant._realtime_service = None
    assistant.last_action_results = ()
    assistant.last_intention = None
    controller = ensure_controller_for(assistant, ROOT)
    controller.settings.set_temporary_workspace(WORKSPACE)

    controller.interaction_router = InteractionRouter(QueueClassifier(
        decision(InteractionCategory.CONVERSATION),
        decision(InteractionCategory.QUESTION),
        decision(InteractionCategory.ACTION_REQUEST, RequestedOutcome.CODE_ARTIFACT, content=True),
        decision(InteractionCategory.ACTION_FOLLOWUP, RequestedOutcome.CODE_ARTIFACT, content=True),
        decision(InteractionCategory.ACTION_REQUEST, RequestedOutcome.CODE_ARTIFACT, content=True),
    ))
    controller.general_task_reasoner = QueueReasoner(
        code_plan(),
        code_plan(modify=True),
        code_plan(archive=True),
    )
    controller.code_generator = lambda spec: GeneratedCodeArtifact.from_dict({
        "language": "python",
        "filename": spec.target_filename,
        "code": (
            "text = input('Ingresa una frase: ')\nprint(sum(c in 'aeiou' for c in text.lower()))\n"
            if spec.target_filename == "contador.py"
            else "value = float(input('Ingresa el valor en Celsius: '))\nprint(value * 9 / 5 + 32)\n"
        ),
        "tests": [],
    })
    controller.code_modifier = lambda spec: GeneratedCodeArtifact.from_dict({
        "language": "python",
        "filename": "conversor.py",
        "code": (
            "direction = input('Elige C o F: ').strip().upper()\n"
            "value = float(input('Ingresa el valor: '))\n"
            "print(value * 9 / 5 + 32 if direction == 'C' else (value - 32) * 5 / 9)\n"
        ),
        "tests": [],
    })

    assert controller.handle("saludo cotidiano holdout", turn_id="social") is None
    assert controller.handle("pregunta explicativa holdout", turn_id="question") is None

    create_pending = controller.handle("acción de código nueva", turn_id="create")
    assert not create_pending.executed
    created = controller.handle("sí", turn_id="create-confirm")
    assert created.executed and (WORKSPACE / "conversor.py").is_file()

    modify_pending = controller.handle("cambio funcional del mismo programa", turn_id="modify")
    assert not modify_pending.executed
    modified = controller.handle("sí", turn_id="modify-confirm")
    assert modified.executed
    assert "Elige C o F" in (WORKSPACE / "conversor.py").read_text(encoding="utf-8")

    (WORKSPACE / "unrelated.txt").write_text("not an archive member", encoding="utf-8")
    zip_pending = controller.handle("código probado y archivado", turn_id="zip")
    assert not zip_pending.executed
    zipped = controller.handle("sí", turn_id="zip-confirm")
    assert zipped.executed
    with zipfile.ZipFile(WORKSPACE / "contador.zip") as archive:
        members = [Path(name).name for name in archive.namelist() if not name.endswith("/")]
    assert members == ["contador.py"]

    payload = {
        "test_type": "INTEGRATION",
        "test_doubles": ["QueueClassifier", "QueueReasoner", "code_generator", "code_modifier", "Assistant.__init__ bypassed"],
        "full_runtime_e2e": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "launcher": "Probar Anydes Tools Conversacion v001.5.bat",
        "adapter_active": bool(getattr(
            AnydesAssistant.respond_realtime,
            "_anydes_tool_conversation_v001_5",
            False,
        )),
        "conversation_bypassed_action_planner": True,
        "question_bypassed_action_planner": True,
        "code_created": True,
        "same_code_modified": True,
        "tests_verified": controller.state.last_verification.get("status") == "VERIFIED",
        "zip_members": members,
        "workspace_was_not_archived": "unrelated.txt" not in members,
        "production_launcher_modified": False,
    }
    RESULT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    controller.close()
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
