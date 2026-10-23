from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable


_TRAILING_CONNECTOR = re.compile(
    r"(?:\b(?:y|o|pero|porque|adem[aá]s|entonces|por ejemplo|es decir|"
    r"primero|segundo|finalmente|que|de|del|con|sin|para)\s*|[:,;([{])$",
    re.I,
)


class CompletionTelemetry:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._lock = threading.Lock()

    def record(self, payload: dict) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {"timestamp": time.time(), **payload}
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def structurally_incomplete(text: str) -> bool:
    clean = str(text or "").strip()
    if not clean:
        return True
    if clean.count("```") % 2:
        return True
    pairs = (("(", ")"), ("[", "]"), ("{", "}"))
    if any(clean.count(left) > clean.count(right) for left, right in pairs):
        return True
    if _TRAILING_CONNECTOR.search(clean.casefold()):
        return True
    return False


def merge_continuation(previous: str, continuation: str) -> str:
    left = previous.rstrip()
    right = continuation.lstrip()
    if not left:
        return right
    if not right:
        return left
    maximum = min(240, len(left), len(right))
    overlap = 0
    for size in range(maximum, 11, -1):
        if left[-size:].casefold() == right[:size].casefold():
            overlap = size
            break
    separator = "" if left.endswith((" ", "\n", "-", "/")) else " "
    return left + separator + right[overlap:].lstrip()


def output_limit_for(messages: list[dict[str, str]], requested: int | None, default: int) -> int:
    prompt = "\n".join(str(item.get("content") or "") for item in messages[-2:]).casefold()
    long_request = bool(
        re.search(r"\b(?:500|750|1000|mil)\s+palabras\b|\b(?:detallad[oa]|en profundidad|paso a paso)\b|```", prompt)
    )
    floor = 1400 if long_request else 320
    return max(floor, int(requested or default))


def stream_with_continuations(
    call: Callable,
    messages: list[dict[str, str]],
    *,
    on_content=None,
    max_output_tokens: int | None,
    default_limit: int,
    telemetry: CompletionTelemetry | None = None,
    max_continuations: int = 2,
):
    limit = output_limit_for(messages, max_output_tokens, default_limit)
    outcome = call(messages, on_content=on_content, max_output_tokens=limit)
    combined = outcome.text
    total_wall = outcome.wall_ms
    total_output = outcome.output_tokens
    continuation_count = 0
    reason = outcome.done_reason or "provider_unspecified"
    token_limited = reason == "length" or outcome.output_tokens >= max(1, limit - 3)
    incomplete = token_limited or structurally_incomplete(combined)

    while incomplete and continuation_count < max_continuations and outcome.connection_state == "completed":
        continuation_count += 1
        continuation_messages = [
            *messages,
            {"role": "assistant", "content": combined},
            {
                "role": "user",
                "content": (
                    "Continúa exactamente desde donde quedó la respuesta anterior. "
                    "No repitas lo ya dicho. Cierra la explicación de forma completa y natural."
                ),
            },
        ]

        def publish(partial: str) -> None:
            if on_content is not None:
                on_content(merge_continuation(combined, partial))

        next_outcome = call(
            continuation_messages,
            on_content=publish,
            max_output_tokens=limit,
        )
        combined = merge_continuation(combined, next_outcome.text)
        total_wall += next_outcome.wall_ms
        total_output += next_outcome.output_tokens
        outcome = next_outcome
        reason = outcome.done_reason or "provider_unspecified"
        token_limited = reason == "length" or outcome.output_tokens >= max(1, limit - 3)
        incomplete = token_limited or structurally_incomplete(combined)

    complete = not incomplete and outcome.connection_state == "completed"
    final = replace(
        outcome,
        text=combined,
        wall_ms=total_wall,
        output_tokens=total_output,
        done_reason=(outcome.done_reason if complete else "incomplete_after_continuation"),
    )
    if telemetry is not None:
        telemetry.record(
            {
                "event": "response_completion",
                "finish_reason": reason,
                "tokens_used": total_output,
                "continuation_count": continuation_count,
                "response_complete": complete,
                "connection_state": outcome.connection_state,
                "structurally_incomplete": incomplete,
            }
        )
    return final

