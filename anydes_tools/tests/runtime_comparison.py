"""A.1 observation for the unchanged human bootstrap and headless replay."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import threading
import time

from anydes_tools.tests.runtime_observation import Observer, digest, safe

STAGES = (
    "INPUT", "INTERACTION_RESOLUTION", "FAST_PATH_RESULT", "GENERAL_REASONER_CALL",
    "RAW_REASONER_RESPONSE", "PARSE_RESULT", "PLAN_VALIDATION", "CAPABILITY_RESOLUTION",
    "PERMISSION", "CODE_GENERATION", "CODE_VALIDATION", "WRITE_TEMP", "EXECUTION",
    "TESTS", "VERIFICATION", "COMMIT", "FINAL_RESPONSE",
)


def fingerprint(value):
    encoded = json.dumps(safe(value, "text" if isinstance(value, str) else ""), sort_keys=True, ensure_ascii=True).encode()
    return {"bytes": len(encoded), "sha256": digest(encoded)}


def format_evidence(value):
    # Retain syntax only, never prose, source code, identifiers or secret fragments.
    text = str(value or "")
    scrubbed = safe(text, "text")
    shape = "REDACTED" if scrubbed == "REDACTED" else re.sub(r'[^{}\[\]:,\"`\s]', 'x', text[:100])
    return {"length": len(text), "body": scrubbed, "format_shape": shape,
            "empty": not text.strip(), "starts_object": text.lstrip().startswith("{")}


def checkpoints(module, name, owner, locals_):
    stages = []
    if module == "anydes_tools.conversation.runtime_patch" and name == "respond_realtime":
        stages += ["INPUT", "FINAL_RESPONSE"]
    if module.endswith("interaction_router") and name in {"route", "resolve"}: stages += ["INTERACTION_RESOLUTION"]
    if module.endswith(".planner") and name == "plan_from_text": stages += ["FAST_PATH_RESULT"]
    if owner == "AssistantGeneralTaskReasoner" and name in {"__call__", "repair"}:
        stages += ["GENERAL_REASONER_CALL"]
    if module.endswith("semantic_runtime") and name == "complete": stages += ["RAW_REASONER_RESPONSE"]
    if module.endswith("semantic_runtime") and name == "_parse_json_object": stages += ["PARSE_RESULT"]
    if owner == "GeneralTaskPlanValidator" and name == "validate": stages += ["PLAN_VALIDATION"]
    if owner == "ExecutablePlanAdapter" and name in {"compile", "revalidate"}: stages += ["PLAN_VALIDATION"]
    if owner == "CapabilityResolver" and name == "resolve": stages += ["CAPABILITY_RESOLUTION"]
    if "permission" in module and not name.startswith("_"): stages += ["PERMISSION"]
    if owner == "AssistantCodeGenerator" and name == "__call__": stages += ["CODE_GENERATION"]
    if module.endswith("code_generation") and name == "validate": stages += ["CODE_VALIDATION"]
    if name == "verify_code_test": stages += ["TESTS", "VERIFICATION"]
    if module == "anydes_tools.executor" and name == "execute":
        request = locals_.get("request")
        cap = getattr(request, "capability", "")
        stages += ["EXECUTION"]
        if cap == "developer.write_code": stages += ["WRITE_TEMP"]
        if cap == "filesystem.atomic_replace": stages += ["COMMIT"]
    return stages


class ComparisonObserver(Observer):
    def __init__(self, destination, *, human=False):
        super().__init__(destination)
        self.human = human
        self.human_turn = 0
        self.active = {}
        self.seen = {}
        self.sequence = 0

    def profile(self, frame, event, result):
        super().profile(frame, event, result)
        if event not in {"call", "return"}: return
        try:
            module = frame.f_globals.get("__name__", "")
            name = frame.f_code.co_name
            owner = frame.f_locals.get("self")
            owner_name = type(owner).__name__
            local = frame.f_locals
            stages = checkpoints(module, name, owner_name, local)
            if stages:
                if event == "call":
                    if "INPUT" in stages and self.human:
                        self.human_turn += 1
                        self.turn_id = "human-" + str(self.human_turn)
                    self.sequence += 1
                    self.active[id(frame)] = (time.perf_counter(), self.sequence, str(self.turn_id))
                    self.seen.setdefault(str(self.turn_id), set()).update(stages)
                started, call_id, turn = self.active.get(id(frame), (time.perf_counter(), None, str(self.turn_id)))
                self.emit("checkpoint", stages=stages, entered=True, edge=event,
                    observed_turn_id=turn, call_id=call_id, module=module, function=name,
                    owner_type=owner_name, duration_ms=(time.perf_counter()-started)*1000,
                    status="ENTERED" if event == "call" else "RETURNED" if result is not None else "RETURNED_NONE_OR_EXCEPTION",
                    result=result if event == "return" else None,
                    bindings={k: local[k] for k in ("turn_id", "task_id", "plan_id", "request", "action", "decision", "specification") if k in local})
                if "INPUT" in stages:
                    service = getattr(owner, "_realtime_service", None)
                    self.emit("state_fingerprint", edge=event, input_text=local.get("text", ""),
                        assistant_type=owner_name, session=fingerprint(vars(service.session) if service is not None and hasattr(getattr(service, "session", None), "__dict__") else getattr(service, "session", None)),
                        session_type=type(getattr(service, "session", None)).__name__)
                if "PARSE_RESULT" in stages:
                    self.emit("parse_input", edge=event, evidence=format_evidence(local.get("text", "")),
                        parser_file=frame.f_code.co_filename, result_type=type(result).__name__)
                if event == "return": self.active.pop(id(frame), None)
            if module.startswith(("anydes.", "anydes_tools.")) and name in {"install", "install_tool_conversation_runtime", "build_assistant"}:
                self.emit("bootstrap_order", edge=event, module=module, function=name, file=frame.f_code.co_filename)
            if module.endswith(("groq_client", "hybrid.router")) and name == "stream":
                config = getattr(owner, "config", None)
                self.emit("provider_boundary", edge=event, module=module, owner_type=owner_name,
                    model=getattr(owner, "model", None), router_state=getattr(owner, "state", None),
                    parameters={k: getattr(config, k, None) for k in ("primary_model", "local_model", "connect_timeout_seconds", "first_token_timeout_seconds", "total_request_timeout_seconds", "max_output_tokens", "inline_retries")},
                    messages_fingerprint=fingerprint(local.get("messages", [])),
                    prompt_parts=[{"role": m.get("role"), "fingerprint": fingerprint(m.get("content", ""))} for m in local.get("messages", [])],
                    request_parameters={k: v for k, v in local.get("payload", {}).items() if k in {"model", "temperature", "top_p", "max_tokens", "stream", "reasoning_format", "reasoning_effort", "response_format", "seed"}},
                    requested_output_tokens=local.get("max_output_tokens"),
                    response=result if event == "return" else None,
                    raw_format=format_evidence(getattr(result, "text", "")) if event == "return" else None)
            if module == "anydes_tools.conversation.controller" and name == "handle" and event == "call":
                self.emit("controller_snapshot", state=getattr(owner, "state", None),
                    registry=owner.registry.capability_snapshot(), health=owner.registry.health_status())
            if module == "anydes_tools.conversation.controller" and name == "handle" and event == "return":
                self.emit("phase_d_dataflow", task_execution=getattr(owner, "last_dataflow_summary", {}))
                pending = owner.state.pending_task
                failure = pending.failure_ref
                plan = pending.failed_plan or pending.plan
                failed_steps = [s for s in owner.last_step_results if s.status != "SUCCESS"]
                failure_details = None
                rendered_matches = None
                if failure is not None:
                    from anydes_tools.conversation.failure_state import FailureResponseRenderer
                    failure_details = {
                        "task_id": failure.task_id, "plan_id": failure.plan_id,
                        "stage": failure.stage, "error_code": failure.error_code,
                        "technical_detail": failure.technical_message,
                        "safe_user_reason": failure.safe_user_message,
                        "retryable": failure.retryable, "timestamp": failure.timestamp,
                        "failure_id": failure.failure_id, "plan_revision": failure.plan_revision,
                        "attempt_number": failure.attempt_number, "step_id": failure.step_id,
                        "operation_type": failure.operation_type, "artifact_type": failure.artifact_type,
                        "capability_id": failure.capability_id, "original_plan_ref": failure.original_plan_ref,
                        "verification_ref": failure.verification_ref,
                    }
                    rendered_matches = getattr(result, "text", "") == FailureResponseRenderer.render(failure)
                self.emit("phase_c_failure_link", pending_status=pending.status,
                    task_id=pending.task_id, plan_revision=pending.plan_revision,
                    retry_count=pending.retry_count, committed_effects=pending.committed_effects,
                    attempt_number=pending.attempt_number,
                    failure_history=owner.state.failure_history,
                    attempt_history=owner.state.attempt_history,
                    verification_evidence=list(owner.state.verification_evidence.values()),
                    failure_record=failure_details,
                    original_plan_ref=fingerprint(plan) if plan is not None else None,
                    plan_budget=plan.execution_budget if plan is not None else None,
                    step_ids=[s.step_id for s in failed_steps],
                    tool_errors=[r.error for r in owner.last_results if r.error],
                    failure_response_matches_record=rendered_matches,
                    response_source=getattr(result, "debug", {}).get("response_source", ""))
            if module == "http.client" and getattr(owner, "host", "") == "api.groq.com":
                if name == "request" and event == "call":
                    payload = json.loads(local.get("body") or "{}")
                    self.emit("http_request_metadata", provider="Groq", endpoint=local.get("url"),
                        parameters={k: v for k, v in payload.items() if k in {"model", "temperature", "top_p", "max_tokens", "stream", "reasoning_format", "reasoning_effort", "response_format", "seed"}},
                        prompts=[{"role": m.get("role"), "fingerprint": fingerprint(m.get("content", ""))} for m in payload.get("messages", [])])
                if name == "getresponse" and event == "return":
                    self.emit("http_response_metadata", provider="Groq", status=getattr(result, "status", None))
        except Exception:
            self.errors += 1

    def __exit__(self, *args):
        super().__exit__(*args)
        for turn, seen in self.seen.items():
            for stage in STAGES:
                self.emit("checkpoint_coverage", observed_turn_id=turn, stage=stage,
                    entered=stage in seen, status="OBSERVED" if stage in seen else "NOT_OBSERVED")


def observe_human(main):
    from anydes_tools.tests.run_full_runtime_phase_a import ROOT, frozen_hashes, file_hash
    directory = ROOT / "anydes_tools/logs/phase_a1_human" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True)
    observer = ComparisonObserver(directory / "trace.jsonl", human=True)
    before = frozen_hashes()
    info = {"test_type": "INSTRUMENTED_HUMAN_RUNTIME", "test_doubles": [],
            "python": sys.executable, "cwd": str(Path.cwd()), "production_before": before,
            "config_overrides": [], "modules": {}}
    try:
        with observer:
            return main()
    finally:
        info["production_after"] = frozen_hashes()
        info["production_unchanged"] = before == info["production_after"]
        info["observer_errors"] = observer.errors
        info["profiles"] = {str(p): file_hash(p) for p in (ROOT / "programa_principal/configuracion").glob("anydes*.env") if p.name in {"anydes_oficial.env", "anydes_normal_25d.env", "anydes_hybrid.env"}}
        info["environment_allowlist"] = {k: os.environ.get(k) for k in ("ANYDES_DATA_DIR", "ANYDES_TOOL_CONVERSATION_V001_5", "PRIMARY_GROQ_MODEL", "PYTHONHASHSEED", "ANYDES_PHASE_A1_OBSERVE")}
        for name, module in list(sys.modules.items()):
            source = getattr(module, "__file__", None)
            if source and name.startswith(("anydes_tools", "anydes.", "probar_anydes")) and Path(source).is_file():
                info["modules"][name] = {"path": source, "sha256": file_hash(source)}
        (directory / "result.json").write_text(json.dumps(safe(info), indent=2), encoding="utf-8")
