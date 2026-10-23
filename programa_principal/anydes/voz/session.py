from __future__ import annotations

from anydes.interfaz.console import build_assistant
from anydes.nucleo.config import config
from anydes.voz.controller import VoiceController


class VoiceSession:
    def __init__(self) -> None:
        self.assistant = build_assistant()
        self.controller = VoiceController(
            assistant=self.assistant,
            config=config,
            activation=config.voice_activation,
        )

    def run(self) -> None:
        self.controller.start()
        if self.controller._is_continuous_mode():
            try:
                while self.controller.is_running():
                    if self.controller._thread is not None:
                        self.controller._thread.join(timeout=0.5)
            except KeyboardInterrupt:
                self.controller.stop()
        else:
            print("Modo mixto: presiona Enter para hablar, escribe texto para chatear, o escribe `salir`.")
            print("Voces: usa `registrar voz Nombre` para guardar una voz, o `voces` para listar perfiles.")
            print(f"STT={config.stt_provider} | TTS={config.tts_provider} | activacion={config.voice_activation}")
            while self.controller.is_running():
                try:
                    command = input("\nTu texto o Enter para hablar: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                normalized = command.lower()
                if normalized in {"salir", "exit", "quit"}:
                    break
                if normalized == "voces":
                    print(self.controller.speakers.list_profiles())
                    continue
                if normalized.startswith("registrar voz "):
                    name = command[len("registrar voz ") :].strip()
                    self.controller._enroll_voice(name)
                    continue
                if command:
                    self.controller._handle_user_text(command, input_mode="typed")
                else:
                    self.controller.request_listen()
        self.controller.stop()
        print("Anydes: Sesion de voz cerrada.")


def main() -> None:
    VoiceSession().run()
