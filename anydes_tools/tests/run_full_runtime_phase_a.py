"""Real staging bootstrap, headless cognitive/tool boundary. No model doubles."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import http.client
import json
import os
from pathlib import Path
import socket
import ssl
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anydes_tools.tests.runtime_observation import Observer, digest, safe

CASES = (
    "Hola Anydes, \u00bfc\u00f3mo est\u00e1s?",
    "Escribe unas 180 palabras sobre lo que recuerdes de mi proyecto Anydes",
    "Haz un programa en Python que pida una temperatura en Celsius y la convierta a Fahrenheit",
    "\u00bfsabes por qu\u00e9?",
    "vuelve a intentarlo",
    "crea contador.py, pru\u00e9balo y gu\u00e1rdalo en ZIP",
)


class BudgetStop(BaseException):
    pass


def file_hash(path):
    return digest(Path(path).read_bytes())


def frozen_hashes():
    manifest = ROOT / "programa_principal/configuracion/anydes_hybrid_release_manifest.json"
    return {r["path"]: file_hash(ROOT / r["path"])
        for r in json.loads(manifest.read_text(encoding="utf-8-sig"))["frozen_dependencies"]}


class RequestBudget:
    """Counts actual Groq HTTP attempts, including runtime retries and probes."""
    def __init__(self, observer, limit):
        self.observer, self.limit, self.count = observer, limit, 0
        self.lock = threading.Lock()

    def __enter__(self):
        self.original = http.client.HTTPSConnection.request
        def request(connection, method, url, body=None, headers=None, **kwargs):
            if connection.host == "api.groq.com":
                with self.lock:
                    if self.count >= self.limit:
                        raise BudgetStop("groq_request_budget_exhausted")
                    self.count += 1
                    number = self.count
                model = "unknown"
                if body:
                    try:
                        model = json.loads(body).get("model", "unknown")
                    except (ValueError, TypeError):
                        pass
                self.observer.emit("groq_http_attempt", attempt=number, model=model)
            return self.original(connection, method, url, body=body,
                headers=headers or {}, **kwargs)
        http.client.HTTPSConnection.request = request
        return self

    def __exit__(self, *_):
        http.client.HTTPSConnection.request = self.original


def provenance(assistant, controller):
    modules = {}
    prefixes = ("anydes_tools.conversation", "anydes_tools.registry", "anydes_tools.settings",
        "anydes.continuity_v2", "anydes.hybrid", "anydes.nucleo.assistant",
        "anydes.interfaz.console", "anydes.realtime", "anydes_tools.executor")
    for name, module in list(sys.modules.items()):
        source = getattr(module, "__file__", None)
        if source and name.startswith(prefixes) and Path(source).is_file():
            modules[name] = {"path": source, "sha256": file_hash(source)}
    profiles = {}
    for name in ("anydes_oficial.env", "anydes_normal_25d.env", "anydes_hybrid.env"):
        path = ROOT / "programa_principal/configuracion" / name
        profiles[name] = {"path": str(path), "sha256": file_hash(path)}
    return {
        "python_executable": sys.executable, "python_version": sys.version,
        "cwd": str(Path.cwd()), "marker": "STAGING_PHASE_A_HEADLESS",
        "bootstrap": str(ROOT / "probar_anydes_tools_conversacion_v001_5.py"),
        "bootstrap_sha256": file_hash(ROOT / "probar_anydes_tools_conversacion_v001_5.py"),
        "human_launcher_sha256": file_hash(ROOT / "Probar Anydes Tools Conversacion v001.5.bat"),
        "modules": modules, "profiles": profiles,
        "assistant_type": type(assistant).__module__ + "." + type(assistant).__name__,
        "config_type": type(assistant.config).__name__,
        "data_dir": str(assistant.config.data_dir),
        "registry": controller.registry.capability_snapshot(),
        "registry_health": controller.registry.health_status(),
        "test_doubles": [],
        "declared_differences": ["headless host, no Qt window/event loop",
            "fresh synthetic memory, not personal historical memory",
            "real temporary workspace permission, not permanent human workspace",
            "telemetry destinations redirected", "no startup brain/B4 warmup",
            "HTTPS attempt budget aborts run, never supplies a simulated response"],
        "not_covered": ["physical microphone", "Parakeet decoding", "Qt rendering",
            "B4 synthesis/playback", "avatar", "audio lip sync", "UI shutdown"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Allow real Groq requests")
    parser.add_argument("--max-calls", type=int, default=15, choices=range(1, 16))
    parser.add_argument("--replay", type=Path, help="Explicit input fixture; no automatic confirmation")
    parser.add_argument("--phase-a1", action="store_true")
    parser.add_argument("--test-fail-code-verification-once", action="store_true",
        help="Explicit test-only precommit injection after real code execution/verification")
    options = parser.parse_args()
    fixture = json.loads(options.replay.read_text(encoding="utf-8")) if options.replay else None
    if options.test_fail_code_verification_once and (
        not fixture or fixture.get("failure_injection") != "CODE_VERIFY_FAIL_ONCE"
    ):
        parser.error("Failure injection requires an explicit matching test fixture")
    prompts = tuple(fixture["inputs"]) if fixture else CASES
    os.chdir(ROOT)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    output = ROOT / "anydes_tools/logs/phase_a" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True)
    workspace = ROOT / "anydes_tools/workspaces/runtime_replay" / output.name
    workspace.mkdir(parents=True)
    os.environ["ANYDES_DATA_DIR"] = str(output / "data")
    before = frozen_hashes()
    observer_type = Observer
    if options.phase_a1:
        from anydes_tools.tests.runtime_comparison import ComparisonObserver
        observer_type = ComparisonObserver
    observer = observer_type(output / "trace.jsonl")
    report = {"test_type": "REAL_RUNTIME_HEADLESS_REPLAY", "surface": "HEADLESS_COGNITIVE_TOOLS",
        "status": "NOT_RUN", "test_doubles": [], "cases": [], "production_before": before}
    report["fixture"] = safe(fixture) if fixture else {"name": "phase_a_original"}
    report["test_interventions"] = (
        ["CODE_VERIFY_FAIL_ONCE_AFTER_REAL_EXECUTION"]
        if options.test_fail_code_verification_once else []
    )
    if options.test_fail_code_verification_once:
        report["test_type"] = "REAL_RUNTIME_HEADLESS_REPLAY_WITH_FAILURE_INJECTION"
    budget = RequestBudget(observer, options.max_calls)

    def before_layers():
        os.environ["ANYDES_HYBRID_LOG_DIR"] = str(output / "hybrid")
        os.environ["ANYDES_COGNITIVE_V2_LOG_DIR"] = str(output / "continuity")
        # Destination-only overrides; no provider or response substitution.
        import anydes.voice_input.environment as environment
        import anydes.voice_input.telemetry as telemetry
        environment.OUT = output / "voice_events"
        telemetry.OUT = environment.OUT

    def run_headless(mode, **unused):
        from runtime_session import runtime_session
        from anydes.interfaz.console import build_assistant
        from anydes.interfaz.turn_interaction import TurnToken
        from anydes.voice_input.core import UserInputEvent
        from anydes.voice_input.integration import InputAssistant
        from anydes_tools.conversation.runtime_patch import ensure_controller_for
        with runtime_session(mode):
            assistant = build_assistant()
            failure_hook = None
            try:
                controller = ensure_controller_for(assistant, ROOT)
                controller.settings.set_temporary_workspace(workspace)
                if options.test_fail_code_verification_once:
                    from anydes_tools.tests.phase_c_failure_injection import FailCodeVerificationOnce
                    failure_hook = FailCodeVerificationOnce(controller, observer)
                    failure_hook.install()
                report["provenance"] = provenance(assistant, controller)
                blocked = "LIVE_NOT_REQUESTED"
                if options.live:
                    if not os.getenv("GROQ_API_KEY"):
                        blocked = "CREDENTIAL_MISSING"
                    else:
                        try:
                            with socket.create_connection(("api.groq.com", 443), timeout=8) as sock:
                                with ssl.create_default_context().wrap_socket(sock, server_hostname="api.groq.com"):
                                    pass
                            blocked = ""
                        except (OSError, ssl.SSLError) as exc:
                            blocked = "NETWORK_" + type(exc).__name__
                for index, prompt in enumerate(prompts, 1):
                    if blocked:
                        report["cases"].append({"case_id": index, "status": "BLOCKED" if options.live else "NOT_RUN", "reason": blocked})
                        continue
                    observer.turn_id = str(index)
                    started = time.perf_counter()
                    prior_failure = controller.state.last_tool_failure
                    observer.emit("turn_start", case_id=index, prior_failure=prior_failure,
                        retry_count=controller.state.retry_count)
                    chunks = []
                    try:
                        with TurnToken(str(index)).bind():
                            response = InputAssistant(assistant, UserInputEvent.keyboard(prompt)).respond_realtime(prompt, on_text=chunks.append)
                        confirmations = 0
                        pending = controller.state.pending_tool_action
                        # This is a separate user turn, never a granted permission flag.
                        if fixture is None and index in {3, 6} and pending is not None and pending.confirmation_expected:
                            observer.turn_id = f"{index}.confirm"
                            with TurnToken(observer.turn_id).bind():
                                response = InputAssistant(assistant, UserInputEvent.keyboard("s\u00ed")).respond_realtime("s\u00ed", on_text=chunks.append)
                            confirmations = 1
                        outcome = {"case_id": index, "status": "OBSERVED", "response_text": response,
                            "confirmations": confirmations, "last_failure": controller.state.last_tool_failure,
                            "verification": controller.state.last_verification,
                            "tool_results": controller.last_results,
                            "artifacts": controller.state.recent_artifact_state.snapshot(),
                            "retry_count": controller.state.retry_count,
                            "failure_query_has_prior_failure": prior_failure is not None if index == 4 else None,
                            "transport_type": getattr(getattr(assistant, "_realtime_service", None), "transport", None).__class__.__name__}
                        report["cases"].append(safe(outcome))
                        publication = getattr(assistant, "last_final_response", None)
                        runtime_outcome = getattr(assistant, "last_runtime_outcome", None)
                        if publication is not None and runtime_outcome is not None:
                            report["cases"][-1]["publication"] = safe({
                                "outcome": runtime_outcome, "decision": publication.decision,
                                "visible_text": publication.visible_text,
                                "spoken_text": getattr(getattr(assistant.last_intention, "speech_intention", None), "text", ""),
                                "chat_speech_equal": publication.visible_text == getattr(
                                    getattr(assistant.last_intention, "speech_intention", None), "text", ""),
                                "sanitized": publication.sanitized})
                            if fixture and fixture.get("capture_publication"):
                                # Opt-in known replay prompts only, not normal human logging.
                                report["cases"][-1]["publication"]["visible_snippet"] = publication.visible_text[:480]
                                report["cases"][-1]["publication"]["spoken_snippet"] = publication.spoken_text[:480]
                        observer.emit("turn_end", **outcome, duration_ms=(time.perf_counter()-started)*1000)
                    except BudgetStop:
                        blocked = "GROQ_BUDGET_EXHAUSTED"
                        report["cases"].append({"case_id": index, "status": "BLOCKED", "reason": blocked})
                    except Exception as exc:
                        report["cases"].append({"case_id": index, "status": "ERROR", "error_type": type(exc).__name__})
                        observer.emit("turn_error", error_type=type(exc).__name__)
                report["provenance"] = provenance(assistant, controller)
            finally:
                if failure_hook is not None:
                    failure_hook.close()
                assistant.close()
                report["assistant_close_returned"] = True
        return 0

    try:
        import probar_anydes_tools_conversacion_v001_5 as bootstrap
        with observer, budget:
            bootstrap.main(headless_runner=run_headless, before_layers=before_layers)
        report["status"] = "OBSERVED" if any(c["status"] == "OBSERVED" for c in report["cases"]) else "BLOCKED"
    except Exception as exc:
        report["status"] = "BLOCKED"
        report["startup_error_type"] = type(exc).__name__
    finally:
        report["groq_http_attempts"] = budget.count
        report["observer_errors"] = observer.errors
        report["production_after"] = frozen_hashes()
        report["production_unchanged"] = before == report["production_after"]
        report["phase_a_complete"] = False  # Human failure equivalence requires evidence review.
        report["threads_after"] = [t.name for t in threading.enumerate()]
        (output / "result.json").write_text(json.dumps(safe(report), ensure_ascii=True, indent=2), encoding="utf-8")
        print(json.dumps({"result": str(output / "result.json"), "status": report["status"],
            "groq_attempts": budget.count, "production_unchanged": report["production_unchanged"]}))
    return 0 if report["production_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
