from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[2]
PROGRAM = ROOT / "programa_principal"
TOOLS = ROOT / "anydes_tools"
for value in (str(ROOT), str(PROGRAM)):
    if value not in sys.path:
        sys.path.insert(0, value)


class ControlledSelfGenerator:
    """Deterministic provider substitute; controller and bootstrap remain real."""

    def __init__(self, assistant) -> None:
        self.assistant = assistant

    def prepare_specification(self, specification):
        from anydes_tools.conversation.semantic_runtime import _safe_self_context

        if specification.source_mode == "SELF_CONTEXT" and not specification.source_context:
            return specification.with_source_context(_safe_self_context(self.assistant))
        return specification

    def __call__(self, specification) -> str:
        words = (
            "Soy Anydes, una asistente orientada a conversar, razonar y ayudar con "
            "herramientas verificadas dentro de espacios autorizados. Mantengo separadas "
            "mi identidad, las preferencias del usuario, los hechos del mundo y las "
            "acciones que realmente he ejecutado. Procuro responder en español natural, "
            "con atención al contexto, prudencia y claridad."
        ).split()
        while len(words) < specification.target_length_words:
            words.extend((
                "Puedo", "colaborar", "sin", "inventar", "experiencias", "ni", "capacidades",
                "y", "protejo", "el", "núcleo", "del", "sistema", "mediante", "permisos",
                "locales",
            ))
        return " ".join(words[: specification.target_length_words])


class EmptyGenerator(ControlledSelfGenerator):
    def __call__(self, specification) -> str:
        return ""


class FailOnceGenerator(ControlledSelfGenerator):
    def __init__(self, assistant) -> None:
        super().__init__(assistant)
        self.calls = 0

    def __call__(self, specification) -> str:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("controlled_first_attempt_failure")
        return super().__call__(specification)


def _bootstrap():
    import anydes_hybrid_runtime as hybrid
    from lanzar_anydes import OFFICIAL_PROFILE, OFFICIAL_VISUAL_PROFILE, load_profile

    hybrid._verify_release()
    hybrid._load_non_secret_profile()
    hybrid._load_secret()
    load_profile(OFFICIAL_PROFILE)
    load_profile(OFFICIAL_VISUAL_PROFILE)
    hybrid._load_non_secret_profile()
    primary = os.getenv("ANYDES_PRIMARY_GROQ_MODEL", "openai/gpt-oss-120b").strip()
    os.environ["PRIMARY_GROQ_MODEL"] = primary
    os.environ["ANYDES_REALTIME_MODEL"] = f"groq:{primary}|fallback:qwen3:8b"
    os.environ["ANYDES_AI_MODEL"] = "qwen3:8b"
    os.environ["ANYDES_EXPECTED_BRAIN"] = os.environ["ANYDES_REALTIME_MODEL"]
    from anydes.voice_input.environment import setup

    setup()
    hybrid._install_hybrid_transport()
    hybrid._install_runtime_lease()
    import anydes.continuity_v2.runtime_patch as continuity

    continuity.install()
    from probar_anydes_tools_conversacion_v001_5 import install_tool_conversation_runtime

    install_tool_conversation_runtime()
    from anydes.interfaz.console import build_assistant

    return build_assistant()


def main() -> int:
    from anydes_tools.audit import ToolAuditLog
    from anydes_tools.conversation.runtime_patch import ensure_controller_for

    workspace = (TOOLS / "workspaces" / "e2e_self_description_human_runtime").resolve()
    workspace_root = (TOOLS / "workspaces").resolve()
    if workspace.parent != workspace_root:
        raise RuntimeError("unsafe_e2e_workspace")
    shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True)
    assistant = _bootstrap()
    controller = ensure_controller_for(assistant, ROOT)
    controller.settings.set_temporary_workspace(workspace)
    controller.use_isolated_action_memory(workspace / ".action-memory-test")
    controller.audit = ToolAuditLog(workspace / "pipeline-trace.jsonl")
    visible: list[str] = []
    evidence: dict[str, object] = {
        "bootstrap": "same staging profile, hybrid router, continuity patch, controller and runtime patch",
        "semantic_provider": "controlled substitute; external network denied to automated runner",
    }
    try:
        controller.semantic_prose_generator = ControlledSelfGenerator(assistant)
        exact = assistant.respond_realtime(
            "Anydes crea un Word llamado Informe Personal y descríbete a ti misma en unas 250 palabras.",
            on_text=visible.append,
        )
        exact_path = workspace / "Informe Personal.docx"
        exact_content = str(controller.last_results[-1].output.get("content", ""))
        evidence["exact_self_description"] = {
            "executed": exact_path.is_file(),
            "nonempty": exact_path.is_file() and exact_path.stat().st_size > 100,
            "word_count": len(exact_content.split()),
            "response": exact,
        }

        controller.semantic_prose_generator = EmptyGenerator(assistant)
        assistant.respond_realtime(
            "crea un Word llamado Fallo Controlado y descríbete en 120 palabras",
            on_text=visible.append,
        )
        explained = assistant.respond_realtime("¿sabes por qué?", on_text=visible.append)
        failure = controller.snapshot().get("last_tool_failure") or {}
        evidence["grounded_failure"] = {
            "stage": failure.get("stage"),
            "error_code": failure.get("error_code"),
            "response": explained,
            "invented_reserved_name": "reserv" in explained.casefold(),
        }

        controller.semantic_prose_generator = ControlledSelfGenerator(assistant)
        actionable = assistant.respond_realtime(
            "¿Puedes crear un documento de Word donde te describas a ti misma?",
            on_text=visible.append,
        )
        evidence["action_over_awareness"] = {
            "created": bool(controller.last_results),
            "capability_only": "sí, puedo crear" in actionable.casefold(),
        }
        awareness = assistant.respond_realtime(
            "¿en qué carpeta tienes permiso?",
            on_text=visible.append,
        )
        evidence["friendly_path"] = {
            "response": awareness,
            "canonical_exposed": str(workspace) in awareness,
        }

        retry = FailOnceGenerator(assistant)
        controller.semantic_prose_generator = retry
        assistant.respond_realtime(
            "crea un Word llamado Reintento Personal y descríbete en 100 palabras",
            on_text=visible.append,
        )
        retried = assistant.respond_realtime("hazlo tú entonces", on_text=visible.append)
        evidence["failed_task_retry"] = {
            "calls": retry.calls,
            "created": (workspace / "Reintento Personal.docx").is_file(),
            "response": retried,
        }
        evidence["canonical_path_visible_anywhere"] = any(str(workspace) in item for item in visible)
        evidence["passed"] = all((
            evidence["exact_self_description"]["executed"],
            evidence["exact_self_description"]["nonempty"],
            200 <= evidence["exact_self_description"]["word_count"] <= 300,
            evidence["grounded_failure"]["error_code"] == "CONTENT_EMPTY",
            not evidence["grounded_failure"]["invented_reserved_name"],
            evidence["action_over_awareness"]["created"],
            not evidence["action_over_awareness"]["capability_only"],
            not evidence["friendly_path"]["canonical_exposed"],
            evidence["failed_task_retry"]["calls"] == 2,
            evidence["failed_task_retry"]["created"],
            not evidence["canonical_path_visible_anywhere"],
        ))
    finally:
        assistant.close()
        from anydes.hybrid.router import close_all_routers

        close_all_routers()
    output = TOOLS / "logs" / "self-description-human-runtime-e2e-v001_5.json"
    output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": evidence.get("passed"), "evidence": str(output)}, ensure_ascii=True))
    shutil.rmtree(workspace, ignore_errors=True)
    return 0 if evidence.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
