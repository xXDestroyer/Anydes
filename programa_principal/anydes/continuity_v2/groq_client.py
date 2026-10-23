from __future__ import annotations

import http.client
import json
import os
import socket
import time


class GroqChatClientV2:
    """Drop-in Groq client that preserves the provider's real finish_reason."""

    host = "api.groq.com"

    def __init__(self, config) -> None:
        self.config = config
        self.model = config.primary_model
        self.api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY_missing")

    def stream(self, messages, *, on_content=None, max_output_tokens=None):
        from anydes.hybrid import router as hybrid

        limit = max(8, int(max_output_tokens or self.config.max_output_tokens))
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.5,
            "top_p": 0.9,
            "max_tokens": limit,
            "stream": True,
            "reasoning_format": "hidden",
            "reasoning_effort": "none" if self.model.startswith("qwen/") else "low",
        }
        connection = http.client.HTTPSConnection(
            self.host, timeout=self.config.connect_timeout_seconds
        )
        started = time.perf_counter()
        first_token = None
        first_content = None
        parts: list[str] = []
        usage: dict[str, object] = {}
        rate_limits: dict[str, str] = {}
        finish_reason = ""
        saw_done = False
        json_buffer = ""
        try:
            connection.request(
                "POST",
                "/openai/v1/chat/completions",
                body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "Anydes-Cognitive-Memory-V2/1.0",
                },
            )
            response = connection.getresponse()
            rate_limits = {
                key.lower(): value
                for key, value in response.getheaders()
                if key.lower().startswith("x-ratelimit-") or key.lower() == "retry-after"
            }
            if response.status >= 400:
                retry_after = hybrid._parse_retry_after(response.getheader("Retry-After"))
                response.read(2048)
                raise hybrid.ProviderFailure(
                    hybrid._http_error_type(response.status),
                    status=response.status,
                    retry_after=retry_after,
                    rate_limits=rate_limits,
                )
            if connection.sock is not None:
                connection.sock.settimeout(self.config.first_token_timeout_seconds)
            while True:
                if time.perf_counter() - started > self.config.total_request_timeout_seconds:
                    raise hybrid.ProviderFailure(
                        "total_request_timeout",
                        partial_text="".join(parts),
                        rate_limits=rate_limits,
                    )
                raw = response.fp.readline() if response.fp is not None else b""
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                if line.startswith("data:"):
                    data = line[5:].strip()
                elif json_buffer:
                    data = line
                else:
                    continue
                if data == "[DONE]":
                    saw_done = True
                    break
                json_buffer += data
                try:
                    event = json.loads(json_buffer)
                except json.JSONDecodeError as exc:
                    if exc.pos >= max(0, len(json_buffer) - 8) or "Unterminated" in exc.msg:
                        continue
                    raise
                json_buffer = ""
                now = time.perf_counter()
                if first_token is None:
                    first_token = now
                    if connection.sock is not None:
                        remaining = max(
                            1.0,
                            self.config.total_request_timeout_seconds - (now - started),
                        )
                        connection.sock.settimeout(remaining)
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                if choice.get("finish_reason"):
                    finish_reason = str(choice["finish_reason"])
                content = str((choice.get("delta") or {}).get("content") or "")
                if content:
                    if first_content is None:
                        first_content = now
                    parts.append(content)
                    if on_content is not None:
                        on_content("".join(parts))
        except hybrid.ProviderFailure:
            raise
        except (TimeoutError, socket.timeout) as exc:
            kind = "first_token_timeout" if first_token is None else "stream_timeout"
            raise hybrid.ProviderFailure(
                kind, partial_text="".join(parts), rate_limits=rate_limits
            ) from exc
        except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
            raise hybrid.ProviderFailure(
                type(exc).__name__, partial_text="".join(parts), rate_limits=rate_limits
            ) from exc
        finally:
            connection.close()

        completed = time.perf_counter()
        text = "".join(parts).strip()
        output_tokens = (
            int(usage.get("completion_tokens") or max(1, (len(text) + 3) // 4))
            if text
            else 0
        )
        input_tokens = int(usage.get("prompt_tokens") or 0)
        generation_s = max(0.001, completed - (first_content or first_token or started))
        wall_ms = (completed - started) * 1000.0
        return hybrid._stream_outcome(
            text=text,
            wall_ms=wall_ms,
            time_to_first_token_ms=((first_token or completed) - started) * 1000.0,
            time_to_first_content_ms=((first_content or completed) - started) * 1000.0,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tokens_per_second=output_tokens / generation_s if output_tokens else 0.0,
            prompt_eval_ms=0.0,
            generation_ms=generation_s * 1000.0,
            done_reason=finish_reason or ("stop" if saw_done else "stream_eof"),
        )

    def probe(self):
        return self.stream(
            [
                {"role": "system", "content": "Responde únicamente OK."},
                {"role": "user", "content": "Estado."},
            ],
            max_output_tokens=4,
        )
