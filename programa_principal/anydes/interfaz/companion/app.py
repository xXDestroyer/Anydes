from __future__ import annotations

import threading
from pathlib import Path
from tkinter import BOTH, Tk, ttk

from anydes.interfaz.companion.components import AvatarPanel, ChatPanel, InnerWorldPanel, InputPanel, ThoughtBar, ToolButton
from anydes.interfaz.companion.theme import CompanionTheme, apply_companion_theme
from anydes.interfaz.companion.view_model import (
    build_inner_world_cards,
    status_label,
    view_state_from_intention,
)
from anydes.interfaz.console import build_assistant
from anydes.nucleo.config import PROJECT_DIR, config
from anydes.voz.controller import VoiceCallbacks, VoiceController


class CompanionApp:
    """Ventana principal de la nueva experiencia Companion.

    El avatar domina la pantalla, la conversacion queda como apoyo y el Mundo
    Interior solo aparece cuando el usuario lo abre.
    """

    def __init__(self) -> None:
        self.root = Tk()
        self.root.title("Anydes")
        self.root.geometry("1180x820")
        self.root.minsize(960, 680)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        apply_companion_theme(self.root)

        self.assistant = build_assistant()
        self.voice_controller: VoiceController | None = None
        self._voice_lock = threading.Lock()
        self._inner_visible = False

        self._build_layout()
        self._refresh_inner_world()

    def run(self) -> None:
        self.root.mainloop()

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0)
        self.root.rowconfigure(0, weight=65)
        self.root.rowconfigure(1, weight=0)
        self.root.rowconfigure(2, weight=24)
        self.root.rowconfigure(3, weight=0)

        top = ttk.Frame(self.root, style="Companion.Root.TFrame", padding=(18, 16, 18, 6))
        top.grid(row=0, column=0, sticky="nsew")
        top.rowconfigure(0, weight=1)
        top.columnconfigure(0, weight=1)

        avatar_images = _avatar_image_candidates()
        self.avatar = AvatarPanel(top, image_paths=avatar_images)
        self.avatar.grid(row=0, column=0, sticky="nsew")

        floating = ttk.Frame(top, style="Companion.Root.TFrame")
        floating.place(relx=0.985, rely=0.04, anchor="ne")
        ToolButton(floating, text="Mundo Interior", command=self.toggle_inner_world).pack()

        self.thought_bar = ThoughtBar(self.root)
        self.thought_bar.grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 8))

        self.chat = ChatPanel(self.root)
        self.chat.grid(row=2, column=0, sticky="nsew", padx=18, pady=(0, 8))

        self.input_panel = InputPanel(self.root, on_send=self.send_text, on_voice=self.voice_button)
        self.input_panel.grid(row=3, column=0, sticky="ew", padx=18, pady=(0, 16))

        self.inner_world = InnerWorldPanel(self.root, on_close=self.toggle_inner_world)
        self.inner_world.grid(row=0, column=1, rowspan=4, sticky="nsew", padx=(0, 18), pady=16)
        self.inner_world.grid_remove()

    def send_text(self, text: str) -> None:
        self.chat.add_message("Tu", text, tag="user")
        self._set_status("thinking")
        threading.Thread(target=self._respond_worker, args=(text,), daemon=True).start()

    def voice_button(self) -> None:
        with self._voice_lock:
            if self.voice_controller is None or not self.voice_controller.is_running():
                self._start_voice_controller()
            self._set_status("listening")
            self.voice_controller.request_listen()

    def toggle_inner_world(self) -> None:
        self._inner_visible = not self._inner_visible
        self._refresh_inner_world()
        if self._inner_visible:
            self.inner_world.grid()
        else:
            self.inner_world.grid_remove()

    def close(self) -> None:
        if self.voice_controller is not None:
            self.voice_controller.stop()
        self.root.destroy()

    def _respond_worker(self, text: str) -> None:
        try:
            response = self.assistant.respond(text, interactive=True, input_mode="typed")
        except Exception as exc:
            self.assistant.audit.record(
                "interface_response_failure",
                error_type=type(exc).__name__,
                detail=str(exc)[:500],
            )
            response = "Perdon... algo interrumpio mi respuesta. Podrias repetirmelo?"
        self.root.after(0, lambda: self._show_response(response))

    def _show_response(self, response: str) -> None:
        self.chat.add_message("Anydes", response)
        state = view_state_from_intention(self.assistant.last_intention)
        self.thought_bar.set_status(state.status)
        self.avatar.set_pose(expression=state.expression, gesture=state.gesture, gaze=state.gaze)
        self._refresh_inner_world()

    def _start_voice_controller(self) -> None:
        callbacks = VoiceCallbacks(
            on_user=lambda text: self._ui(lambda: self.chat.add_message("Tu", text, tag="user")),
            on_assistant=lambda text: self._ui(lambda: self._show_response(text)),
            on_system=lambda text: self._ui(lambda: self.chat.add_message("Sistema", text, tag="system")),
            on_status=lambda text: self._ui(lambda: self.thought_bar.set_status(_friendly_status(text))),
        )
        self.voice_controller = VoiceController(
            assistant=self.assistant,
            config=config,
            activation="button",
            callbacks=callbacks,
            embedded=True,
            disable_internal_proactive=True,
        )
        self.voice_controller.start()

    def _ui(self, callback) -> None:
        self.root.after(0, callback)

    def _set_status(self, status: str) -> None:
        self.thought_bar.set_status(status_label(status))
        if status == "listening":
            self.avatar.set_pose(expression="listening", gesture="listening", gaze="user")
        elif status == "thinking":
            self.avatar.set_pose(expression="thinking", gesture="thinking", gaze="down")

    def _refresh_inner_world(self) -> None:
        cards = build_inner_world_cards(self.assistant.last_internal_context, self.assistant.last_intention)
        self.inner_world.set_cards(cards)


def _friendly_status(text: str) -> str:
    lower = text.lower()
    if "escuch" in lower:
        return status_label("listening")
    if "pens" in lower or "proces" in lower:
        return status_label("thinking")
    if "habl" in lower:
        return status_label("speaking")
    if "record" in lower:
        return status_label("remembering")
    return status_label("idle")


def _avatar_image_candidates() -> tuple[Path, ...]:
    return (
        PROJECT_DIR / "Anydes 3D Modelo" / "renders" / "anydes_avatar_v004_front.png",
        PROJECT_DIR / "Anydes 3D Modelo" / "final_review" / "renders" / "anydes_v001_review_front.png",
        PROJECT_DIR / "Anydes 3D Modelo" / "Anydes_reference_definitive.png",
        PROJECT_DIR / "programa_principal" / "avatar" / "assets" / "anydes_happy.png",
        PROJECT_DIR / "programa_principal" / "avatar" / "assets" / "anydes_neutral.png",
    )


def main() -> None:
    from anydes.interfaz.pyside_companion.app import main as companion_main

    companion_main()


if __name__ == "__main__":
    main()
