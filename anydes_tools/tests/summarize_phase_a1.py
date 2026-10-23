"""Read-only evidence extraction; writes only the requested diagnostic output."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anydes_tools.tests.runtime_observation import safe, digest
from anydes_tools.tests.runtime_comparison import STAGES, checkpoints


def rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def summarize(directory):
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    events = list(rows(directory / "trace.jsonl"))
    turns = {}
    for event in events:
        turn = str(event.get("observed_turn_id", event.get("turn_id", "startup")))
        entry = turns.setdefault(turn, {"checkpoints": {s: {"entered": False, "evidence": []} for s in STAGES}, "audit": [], "http": []})
        if event["event"] == "checkpoint":
            for stage in event["stages"]:
                entry["checkpoints"][stage]["entered"] = True
                entry["checkpoints"][stage]["evidence"].append({k: event.get(k) for k in ("module", "function", "edge", "status", "duration_ms", "call_id")})
        if event["event"] == "stage_return":
            # Recover method coverage even if an earlier observer missed its label.
            stages = checkpoints(event.get("module", ""), event.get("function", ""), event.get("owner_type", ""), {})
            for stage in stages:
                entry["checkpoints"][stage]["entered"] = True
        if event["event"] == "runtime_audit":
            entry["audit"].append(event)
        if event["event"] in {"http_request_metadata", "http_response_metadata", "groq_http_attempt"}:
            entry["http"].append(event)
    return {"directory": str(directory), "result_sha256": digest((directory / "result.json").read_bytes()),
            "status": result["status"], "groq_attempts": result["groq_http_attempts"],
            "observer_errors": result["observer_errors"], "cases": result["cases"], "turns": turns}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    human_path = ROOT / "anydes_tools/logs/conversation-v001_5.jsonl"
    historical = []
    keep = {"conversation_interaction_route", "general_task_plan", "conversation_user_task_planned",
            "conversation_permission_pending", "pending_tool_resumed", "conversation_executor_dispatch",
            "tool_requested", "tool_completed", "conversation_tool_failure_state", "general_task_outcome",
            "conversation_action_result"}
    for event in rows(human_path):
        if event.get("timestamp", "").startswith("2026-09-27T23:14:") and event.get("event") in keep:
            historical.append(safe(event))
    payload = {"historical_human": {"source": str(human_path), "timestamp_window_utc": "2026-09-27T23:14", "events": historical,
                "provider_model": "NOT_RECORDED_IN_THIS_AUDIT", "complete_input_replay": False},
               "runs": [summarize(p) for p in args.runs], "no_runtime_execution": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
