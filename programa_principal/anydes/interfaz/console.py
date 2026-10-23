from __future__ import annotations

from anydes.cerebro.llm import build_brain, build_vision_brain
from anydes.cognition.live_runtime import (
    CognitiveEventBus,
    InMemoryMetricsCollector,
    NullEventPublisher,
    NullMetricsCollector,
    RuntimeSettings,
    build_current_provider_adapters,
    build_live_runtime,
    connect_default_invalidation_routes,
)
from anydes.funciones_extra.registry import ToolRegistry
from anydes.nucleo.assistant import AnydesAssistant
from anydes.nucleo.audit import AuditLog
from anydes.nucleo.config import config
from anydes.nucleo.permissions import PermissionManager
from anydes.nucleo.personality import PersonalityProfile
from anydes.nucleo.security import SpeakerVerifier


def build_assistant() -> AnydesAssistant:
    verifier = SpeakerVerifier(config)
    runtime = None
    events = NullEventPublisher()
    metrics = InMemoryMetricsCollector() if config.developer_mode else NullMetricsCollector()
    if config.live_runtime_enabled:
        bundle = build_live_runtime(
            RuntimeSettings(
                mode="simulated" if config.live_runtime_simulated else "live",
                max_workers=max(1, config.live_runtime_workers),
                developer_metrics=config.developer_mode,
            )
        )
        runtime = bundle.runtime
        metrics = bundle.metrics
        events = CognitiveEventBus(metrics)
        for provider in build_current_provider_adapters(config):
            runtime.register(provider)
        connect_default_invalidation_routes(events, runtime)
        runtime.start()
    if config.developer_mode:
        metrics.enable_interaction_tracing(
            config.interaction_trace_path,
            config.slow_interaction_trace_path,
            slow_threshold_ms=config.slow_interaction_threshold_ms,
        )
    try:
        return AnydesAssistant(
            config=config,
            tools=ToolRegistry.default(config),
            permissions=PermissionManager(verifier=verifier),
            audit=AuditLog(config.audit_log_path),
            personality=PersonalityProfile.load(config.personality_prompt_path),
            brain=build_brain(config),
            vision_brain=build_vision_brain(config),
            metrics=metrics,
            events=events,
            live_runtime=runtime,
        )
    except Exception:
        if runtime is not None:
            runtime.stop()
        raise


def main() -> None:
    assistant = build_assistant()
    print("Escribe 'ayuda' para ver comandos o 'salir' para cerrar.")
    try:
        while True:
            try:
                message = input("\nTu: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nAnydes: Sesion cerrada.")
                return

            if message.lower() in {"salir", "exit", "quit"}:
                print("Anydes: Sesion cerrada.")
                return

            response = assistant.respond(message, interactive=True, input_mode="typed")
            print(f"Anydes: {response}")
    finally:
        assistant.close()
