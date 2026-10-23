from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import sys


ROOT = Path(__file__).resolve().parents[2]
PROGRAM = ROOT / "programa_principal"
for _entry in (str(ROOT), str(PROGRAM)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)
WORKSPACE = ROOT / "anydes_tools" / "workspaces" / "general-reasoner-live-groq"
RESULT = ROOT / "anydes_tools" / "logs" / "general-task-reasoner-live-groq-v001_5.json"


class AuditRecorder:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def record(self, event: str, **details) -> None:
        self.events.append({"event": event, "details": details})


def run() -> dict:
    import anydes_hybrid_runtime as hybrid_runtime

    hybrid_runtime._load_non_secret_profile()
    hybrid_runtime._load_secret()
    primary = (
        sys.argv[1].strip()
        if len(sys.argv) > 1
        else os.getenv("ANYDES_PRIMARY_GROQ_MODEL", "openai/gpt-oss-120b").strip()
    )
    mode = sys.argv[2].strip().casefold() if len(sys.argv) > 2 else "all"
    if mode not in {"all", "report", "code"}:
        raise ValueError("mode_must_be_all_report_or_code")
    os.environ["PRIMARY_GROQ_MODEL"] = primary
    if not os.getenv("GROQ_API_KEY", "").strip():
        raise RuntimeError("GROQ_API_KEY_missing")

    from anydes.continuity_v2.groq_client import GroqChatClientV2
    from anydes.hybrid.router import HybridRouterConfig
    from anydes.nucleo.assistant import AnydesAssistant
    from anydes_tools.conversation.runtime_patch import ensure_controller_for
    from anydes_tools.conversation.semantic_runtime import (
        AssistantCodeGenerator,
        AssistantGeneralTaskReasoner,
        AssistantSemanticProseGenerator,
    )
    from probar_anydes_tools_conversacion_v001_5 import install_tool_conversation_runtime

    config = replace(
        HybridRouterConfig.from_environment(),
        connect_timeout_seconds=6.0,
        first_token_timeout_seconds=12.0,
        total_request_timeout_seconds=60.0,
        max_output_tokens=2400,
        auto_housekeeping=False,
    )
    transport = GroqChatClientV2(config)
    shutil.rmtree(WORKSPACE, ignore_errors=True)
    WORKSPACE.mkdir(parents=True)
    install_tool_conversation_runtime()

    assistant = object.__new__(AnydesAssistant)
    assistant.audit = AuditRecorder()
    assistant._realtime_service = SimpleNamespace(transport=transport, session=None)
    assistant.last_action_results = ()
    assistant.last_intention = None
    controller = ensure_controller_for(assistant, ROOT)
    controller.audit = assistant.audit
    controller.executor.audit = assistant.audit
    controller.settings.set_temporary_workspace(WORKSPACE)
    controller.use_isolated_action_memory(WORKSPACE / ".action-memory")
    controller.general_task_reasoner = AssistantGeneralTaskReasoner(assistant)
    controller.code_generator = AssistantCodeGenerator(assistant)
    controller.semantic_prose_generator = AssistantSemanticProseGenerator(assistant)
    controller.safe_context_provider = lambda kind, query: (
        "PROVENANCE: USER_FACT, RETRIEVED_MEMORY, DIRECT_CONVERSATION\n"
        "DemoUser desarrolla un proyecto de asistente llamado Anydes y prioriza resultados "
        "verificables, seguridad y conversación natural."
    )

    report_response = (
        controller.handle(
            "haz un informe de unas 200 palabras sobre todo lo que sepas de mí",
            turn_id="live-report",
        )
        if mode in {"all", "report"}
        else None
    )
    report = report_response.text if report_response is not None else ""
    code_response = (
        controller.handle(
            "haz un programa en Python que te pida ingresar un número y verifique si es primo",
            turn_id="live-code-plan",
        )
        if mode in {"all", "code"}
        else None
    )
    code_first = code_response.text if code_response is not None else ""
    final_code_response = code_response
    if code_response is not None and controller.state.pending_tool_action is not None:
        confirmed = controller.handle("sí", turn_id="live-code-confirm")
        if confirmed is None:
            raise RuntimeError("live_code_confirmation_not_handled")
        final_code_response = confirmed
        code_final = confirmed.text
    else:
        code_final = code_first
    code_paths = sorted(WORKSPACE.glob("*.py"))
    verification = dict(controller.state.last_verification)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": "Groq",
        "model": primary,
        "mode": mode,
        "context": "synthetic_controlled_only",
        "secret_logged": False,
        "report_word_count": len(report.split()),
        "report_handled": report_response is not None,
        "report_preview": report[:240],
        "code_response": code_final,
        "code_handled": code_response is not None,
        "code_debug": dict(final_code_response.debug) if final_code_response is not None else {},
        "code_artifacts": [item.name for item in code_paths],
        "code_verified": verification.get("status") == "VERIFIED",
        "verification": verification,
        "last_failure": (
            {
                "stage": controller.state.last_tool_failure.stage,
                "error_code": controller.state.last_tool_failure.error_code,
                "retryable": controller.state.last_tool_failure.retryable,
            }
            if controller.state.last_tool_failure is not None
            else None
        ),
        "audit_event_names": [item["event"] for item in assistant.audit.events],
        "general_plan_events": sum(item["event"] == "general_task_plan" for item in assistant.audit.events),
        "general_outcome_events": sum(item["event"] == "general_task_outcome" for item in assistant.audit.events),
        "plan_events": [
            item["details"] for item in assistant.audit.events
            if item["event"] in {
                "general_task_plan",
                "general_task_reasoner",
                "general_task_reasoner_failed",
                "general_task_repair_failed",
            }
        ],
        "pseudo_tool_markup_visible": any(
            marker in (report + code_first + code_final)
            for marker in ("<tool_call", "[developer.", '"tool_id"', '"action_type"')
        ),
    }
    RESULT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    controller.close()
    report_ok = mode == "code" or 150 <= payload["report_word_count"] <= 250
    code_ok = mode == "report" or (bool(code_paths) and payload["code_verified"])
    if not (report_ok and code_ok and not payload["pseudo_tool_markup_visible"]):
        raise AssertionError(payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
