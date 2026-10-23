from __future__ import annotations

import os
from pathlib import Path

from .completion import CompletionTelemetry, stream_with_continuations
from .core import WorkingConversationStateV2
from .groq_client import GroqChatClientV2
from .turn_style import apply_response_style


_INSTALLED = False


def configure_environment() -> None:
    os.environ["ANYDES_COGNITIVE_CONTINUITY_V2"] = "true"
    os.environ["ANYDES_REALTIME_CONTEXT_TOKEN_BUDGET"] = "1800"
    os.environ["ANYDES_REALTIME_MAX_OUTPUT_TOKENS"] = "320"
    os.environ["ANYDES_REALTIME_MAX_RESPONSE_WORDS"] = "1400"
    os.environ["ANYDES_REALTIME_NUM_CTX"] = "4096"
    os.environ["ANYDES_MATH_PRESENTATION_V1"] = "true"
    os.environ.setdefault("ANYDES_TTS_FIRST_CHUNK_TIMEOUT_SECONDS", "45")
    os.environ.setdefault("ANYDES_TTS_PROGRESS_TIMEOUT_SECONDS", "20")


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    configure_environment()

    import anydes.realtime.service as service_module
    import anydes.realtime.session as session_module
    import anydes.hybrid.router as hybrid_module

    hybrid_module.GroqChatClient = GroqChatClientV2

    base_session = session_module.RealtimeSessionState
    dialogue_resolution = session_module.DialogueResolution
    telemetry_root = Path(
        os.environ.get(
            "ANYDES_COGNITIVE_V2_LOG_DIR",
            str(Path(__file__).resolve().parents[2] / "datos" / "telemetria" / "continuity_v2"),
        )
    )

    class RealtimeSessionStateV2(base_session):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.continuity_v2 = WorkingConversationStateV2(
                telemetry_path=telemetry_root / "working_state_v2.jsonl"
            )

        def resolve_reference(self, text):
            base = super().resolve_reference(text)
            if base.kind in {
                "action_confirmation", "action_rejection", "offer_acceptance",
                "offer_rejection", "question_answer",
            }:
                return base
            resolved = self.continuity_v2.resolve(text)
            if resolved.kind:
                return dialogue_resolution(
                    kind="previous_reference" if resolved.kind not in {"pending_question"} else "question_answer",
                    canonical_text=resolved.canonical_text,
                    target=resolved.target,
                    confidence=resolved.confidence,
                )
            return base

        def context_note(self, act, user_text=""):
            original = super().context_note(act, user_text)
            v2 = self.continuity_v2.context_note()
            return f"{original}\n{v2}".strip()

        def commit(self, user_text, assistant_text, act):
            super().commit(user_text, assistant_text, act)
            self.continuity_v2.observe_turn(user_text, assistant_text)

    session_module.RealtimeSessionStateV2 = RealtimeSessionStateV2
    session_module.DialogueState = RealtimeSessionStateV2
    service_module.RealtimeSessionState = RealtimeSessionStateV2

    original_classify = service_module.RealtimeConversationService.classify

    def classify_v2(self, user_text, *, has_tool_intent=False):
        result = original_classify(self, user_text, has_tool_intent=has_tool_intent)
        if result.route == "complex" and not has_tool_intent:
            return service_module.TurnClassification(
                route="everyday",
                reasons=(*result.reasons, "cognitive_v2_full_response"),
            )
        return result

    service_module.RealtimeConversationService.classify = classify_v2

    original_stream = hybrid_module.HybridRealtimeTransport.stream
    completion_telemetry = CompletionTelemetry(telemetry_root / "response_completion.jsonl")

    def stream_v2(self, messages, *, on_content=None, max_output_tokens=None):
        default = int(getattr(self.router.config, "max_output_tokens", 320))
        constrained_messages, _style = apply_response_style(messages)
        return stream_with_continuations(
            lambda payload, **options: original_stream(self, payload, **options),
            constrained_messages,
            on_content=on_content,
            max_output_tokens=max_output_tokens,
            default_limit=default,
            telemetry=completion_telemetry,
        )

    hybrid_module.HybridRealtimeTransport.stream = stream_v2
    from .math_runtime import install as install_math_runtime
    install_math_runtime()
    _INSTALLED = True
