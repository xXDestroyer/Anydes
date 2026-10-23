from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
import http.client
from contextlib import contextmanager
from anydes.interfaz.turn_interaction import current_turn
from dataclasses import dataclass
from typing import Callable


@contextmanager
def cancellable_response(request, timeout):
    token = current_turn()
    if token is None:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            yield response
        return
    url = urllib.parse.urlsplit(request.full_url)
    connection_type = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(url.hostname, url.port, timeout=timeout)
    def abort():
        sock = connection.sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            # On Windows shutdown alone need not wake a timed makefile read.
            # Close this request's socket handle, not BufferedReader.close(),
            # which waits for the reading thread's lock.
            try:
                socket.close(sock.detach())
            except OSError:
                pass
    try:
        with token.abort_with(abort):
            connection.connect()
            token.check()
            connection.request(request.get_method(), url.path + ("?" + url.query if url.query else ""), request.data, dict(request.header_items()))
            response = connection.getresponse()
            token.check()
            if response.status >= 400:
                raise urllib.error.HTTPError(request.full_url, response.status, response.reason, response.headers, response)
            with response:
                yield response
    except Exception:
        token.check()
        raise
    finally:
        connection.close()


@dataclass(frozen=True)
class StreamOutcome:
    text: str
    wall_ms: float
    time_to_first_token_ms: float
    time_to_first_content_ms: float
    input_tokens: int
    output_tokens: int
    tokens_per_second: float
    load_ms: float
    prompt_eval_ms: float
    generation_ms: float
    done_reason: str
    timed_out: bool = False
    connection_state: str = "completed"
    interruption_reason: str = ""
    error_type: str = ""


class OllamaRealtimeTransport:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        keep_alive: str,
        timeout_seconds: float,
        max_output_tokens: int,
        num_ctx: int = 2048,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.keep_alive = keep_alive
        self.timeout_seconds = max(0.7, float(timeout_seconds))
        self.max_output_tokens = max(8, int(max_output_tokens))
        self.num_ctx = max(512, int(num_ctx))

    def warmup(self, *, timeout_seconds: float = 60.0) -> float:
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(
                {
                    "model": self.model,
                    "keep_alive": self.keep_alive,
                    "stream": False,
                    "think": False,
                    "messages": [
                        {
                            "role": "system",
                            "content": "Responde de forma breve.",
                        },
                        {
                            "role": "user",
                            "content": "Di solamente: lista.",
                        },
                    ],
                    "options": {
                        "num_ctx": self.num_ctx,
                        "num_predict": 2,
                        "temperature": 0.0,
                    },
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        with urllib.request.urlopen(
            request,
            timeout=max(5.0, float(timeout_seconds)),
        ) as response:
            response.read()
        return (time.perf_counter() - started) * 1000.0

    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        on_content: Callable[[str], None] | None = None,
        max_output_tokens: int | None = None,
    ) -> StreamOutcome:
        output_limit = (
            self.max_output_tokens
            if max_output_tokens is None
            else max(8, int(max_output_tokens))
        )
        payload = {
            "model": self.model,
            "stream": True,
            "think": False,
            "keep_alive": self.keep_alive,
            "messages": messages,
            "options": {
                "temperature": 0.5,
                "top_p": 0.9,
                "repeat_penalty": 1.08,
                "num_ctx": self.num_ctx,
                "num_predict": output_limit,
            },
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        first_token_at: float | None = None
        first_content_at: float | None = None
        parts: list[str] = []
        final: dict[str, object] = {}
        timed_out = False
        connection_state = "completed"
        interruption_reason = ""
        error_type = ""
        try:
            with cancellable_response(request, self.timeout_seconds) as response:
                for raw_line in response:
                    if current_turn() is not None:
                        current_turn().check()
                    now = time.perf_counter()
                    if now - started > self.timeout_seconds:
                        timed_out = True
                        connection_state = "interrupted"
                        interruption_reason = "wall_timeout"
                        break
                    line = raw_line.strip()
                    if not line:
                        continue
                    event = json.loads(line.decode("utf-8"))
                    message = event.get("message") or {}
                    content = str(message.get("content") or "")
                    thinking = str(message.get("thinking") or "")
                    if first_token_at is None and (content or thinking):
                        first_token_at = now
                    if content:
                        if first_content_at is None:
                            first_content_at = now
                        parts.append(content)
                        if on_content is not None:
                            on_content("".join(parts))
                    if event.get("done"):
                        final = event
        except (TimeoutError, socket.timeout):
            timed_out = True
            connection_state = "interrupted"
            interruption_reason = "socket_timeout"
            error_type = "TimeoutError"
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                timed_out = True
                interruption_reason = "url_timeout"
                error_type = type(exc.reason).__name__
            else:
                interruption_reason = "url_error"
                error_type = type(exc).__name__
            connection_state = "interrupted"
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            connection_state = "interrupted"
            interruption_reason = "invalid_stream_event"
            error_type = type(exc).__name__

        if current_turn() is not None:
            current_turn().check()
        completed = time.perf_counter()
        wall_ms = (completed - started) * 1000.0
        generation_ns = int(final.get("eval_duration") or 0)
        rendered = "".join(parts).strip()
        output_tokens = int(final.get("eval_count") or 0)
        if output_tokens <= 0 and rendered:
            output_tokens = max(1, (len(rendered) + 3) // 4)
        generation_seconds = generation_ns / 1_000_000_000
        if generation_seconds <= 0 and first_content_at is not None:
            generation_seconds = max(0.001, completed - first_content_at)
        return StreamOutcome(
            text=rendered,
            wall_ms=wall_ms,
            time_to_first_token_ms=(
                (first_token_at - started) * 1000.0
                if first_token_at is not None
                else wall_ms
            ),
            time_to_first_content_ms=(
                (first_content_at - started) * 1000.0
                if first_content_at is not None
                else wall_ms
            ),
            input_tokens=int(final.get("prompt_eval_count") or 0),
            output_tokens=output_tokens,
            tokens_per_second=(
                output_tokens / generation_seconds if generation_seconds > 0 else 0.0
            ),
            load_ms=int(final.get("load_duration") or 0) / 1_000_000,
            prompt_eval_ms=int(final.get("prompt_eval_duration") or 0) / 1_000_000,
            generation_ms=generation_ns / 1_000_000,
            done_reason=str(final.get("done_reason") or ("timeout" if timed_out else "")),
            timed_out=timed_out,
            connection_state=connection_state,
            interruption_reason=interruption_reason,
            error_type=error_type,
        )
