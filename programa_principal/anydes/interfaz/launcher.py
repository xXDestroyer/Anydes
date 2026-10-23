"""Deprecated Tk launcher retained as a compatibility module.

Its ``main`` entrypoint redirects to the supported PySide6 Companion interface.
"""

from __future__ import annotations

import socket
import sys
import threading
import webbrowser
from dataclasses import replace
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tkinter import BooleanVar, Canvas, DoubleVar, Scrollbar, StringVar, Tk, Toplevel, filedialog, messagebox, simpledialog, ttk
from tkinter.scrolledtext import ScrolledText

from anydes.cerebro.llm import build_brain, build_vision_brain
from anydes.cerebro.ollama_vision import pull_ollama_model, resolve_ollama_vision_model
from anydes.cognition import Intention
from anydes.funciones_extra.registry import ToolRegistry
from anydes.interfaz.console import build_assistant
from anydes.interfaz.theme import AnydesTheme, apply_theme, make_card, make_header
from anydes.interfaz.runtime_env import build_runtime_env, write_env_file
from anydes.nucleo.assistant import AnydesAssistant
from anydes.nucleo.audit import AuditLog
from anydes.nucleo.config import PROGRAM_DIR, PROJECT_DIR, Config, config
from anydes.nucleo.consent import (
    external_access_decision,
    is_external_provider,
    session_consents,
)
from anydes.nucleo.sensors import sensor_hub
from anydes.nucleo.orquestador import BehaviorOrchestrator
from anydes.personas.profile import ProfileManager
from anydes.nucleo.permissions import PermissionManager, SENSITIVE_TOOLS
from anydes.nucleo.personality import PersonalityProfile
from anydes.nucleo.security import SpeakerVerifier
from anydes.memoria.manager import MemoryManager
from anydes.proactivo.autonomo import ProactiveDaemon
from anydes.vision.autonomo import ControladorVisionAutonoma
from anydes.vision.tools import ensure_screen_watching, stop_all_screen_watching
from anydes.vision.state import get_screen_state
from anydes.voz.controller import VoiceCallbacks, VoiceController
from anydes.voz.manager_factory import build_voice_manager

# Fallback preference order (best -> worst)
FALLBACK_AI_PROVIDERS = ["local", "ollama", "groq", "openai"]
FALLBACK_TTS = ["piper", "edge", "windows", "openai", "none"]
FALLBACK_STT = ["vosk", "windows", "openai", "none"]


SENSITIVE_CONFIRM_TOOLS = SENSITIVE_TOOLS


class GuiPermissionManager(PermissionManager):
    def __init__(self, root: Tk, verifier=None, *, vision_enabled: bool = False) -> None:
        super().__init__(verifier=verifier)
        self.root = root
        self.vision_enabled = vision_enabled

    def authorize(self, tool_name: str, description: str, *, interactive: bool) -> bool:
        if not self.interlocutor_allows(tool_name):
            self._show_info(
                "Memoria privada",
                "La identidad actual no permite acceder a la memoria personal o protegida.",
            )
            return False
        policy = self.policy_for(tool_name)
        if policy.level == "safe":
            return True
        if policy.level == "blocked":
            self._show_info("Accion bloqueada", f"{tool_name}\n\n{policy.reason}")
            return False
        if not interactive:
            return False

        allowed = self._ask_yes_no(
            "Confirmar accion",
            f"Anydes quiere ejecutar:\n\n{tool_name}\n\n{description}\n\n{policy.reason}",
        )
        if not allowed:
            return False

        if self.verifier is not None and tool_name in SENSITIVE_CONFIRM_TOOLS:
            phrase = self._ask_text(
                "Verificacion",
                "Escribe tu frase de seguridad para esta accion sensible.",
            )
            if not phrase:
                return False
            return bool(self.verifier.verify_text(phrase).ok)
        return True

    def _ask_yes_no(self, title: str, message: str) -> bool:
        event = threading.Event()
        result = {"value": False}

        def ask() -> None:
            result["value"] = bool(messagebox.askyesno(title, message, parent=self.root))
            event.set()

        self.root.after(0, ask)
        event.wait()
        return result["value"]

    def _ask_text(self, title: str, message: str) -> str:
        event = threading.Event()
        result = {"value": ""}

        def ask() -> None:
            result["value"] = simpledialog.askstring(title, message, parent=self.root, show="*") or ""
            event.set()

        self.root.after(0, ask)
        event.wait()
        return result["value"]

    def _show_info(self, title: str, message: str) -> None:
        self.root.after(0, lambda: messagebox.showinfo(title, message, parent=self.root))


class AvatarServer:
    def __init__(self) -> None:
        self.httpd: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.port: int | None = None

    def start(self) -> str:
        if self.httpd is not None and self.port is not None:
            return f"http://127.0.0.1:{self.port}/avatar/"

        self.port = _free_port(8765)
        handler = partial(SimpleHTTPRequestHandler, directory=str(PROGRAM_DIR))
        self.httpd = ThreadingHTTPServer(("127.0.0.1", self.port), handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return f"http://127.0.0.1:{self.port}/avatar/"

    def stop(self) -> None:
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
        self.httpd = None
        self.thread = None
        self.port = None


class AnydesLauncher:
    def __init__(self) -> None:
        self.root = Tk()
        self.root.title("Anydes · Asistente personal")
        self.root.geometry("1240x760")
        self.root.minsize(1020, 680)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.avatar_server = AvatarServer()
        self.shared_speaking_event = threading.Event()
        self.voice_controller: VoiceController | None = None
        self.assistant: AnydesAssistant | None = None
        self.assistant_signature = ""
        self.vision_controller: ControladorVisionAutonoma | None = None
        self.proactive_daemon: ProactiveDaemon | None = None

        self.ai_provider = StringVar(value=config.ai_provider)
        self.ai_model = StringVar(value=config.ai_model if config.ai_model != "gpt-5-mini" else "llama3.2")
        self.stt_provider = StringVar(value=config.stt_provider)
        self.tts_provider = StringVar(value=config.tts_provider)
        self.voice_activation = StringVar(value=config.voice_activation or "continuous_wake_word")
        self.wake_word = StringVar(value=config.wake_word)
        self.voice_seconds = StringVar(value=str(config.voice_seconds))
        self.proactive_enabled = BooleanVar(value=config.proactive_enabled)
        self.proactive_interval = StringVar(value=str(config.proactive_interval_seconds))
        self.vision_enabled = BooleanVar(value=config.vision_enabled)
        self.vision_provider = StringVar(value=config.vision_provider or "auto")
        self.vision_auto_pull = BooleanVar(value=config.vision_auto_pull)
        self.vision_model = StringVar(value=config.vision_model or config.resolved_vision_model())
        self.vision_reactivity = DoubleVar(value=config.vision_reactivity_level)
        self.screen_interval = StringVar(value=str(config.screen_watch_interval_seconds))
        self.screen_live_mode = BooleanVar(value=config.screen_live_mode)
        self.screen_live_interval = StringVar(value=str(config.screen_live_interval_seconds))
        self.screen_fresh_on_analyze = BooleanVar(value=config.screen_fresh_on_analyze)
        # Defaults: speak with voice enabled and continuous listening with wake word
        self.speak_chat = BooleanVar(value=True)
        self.status = StringVar(value="Lista · elige un modo o escribe en el chat")
        self.orchestrator = BehaviorOrchestrator(
            config=config,
            user_name=getattr(config, "user_name", "DemoUser"),
            profile_name=getattr(config, "profile_name", "Anydes"),
        )
        self.profile_manager = ProfileManager(config)
        # attach profile manager so orchestrator can persist inferred preferences
        try:
            self.orchestrator.set_profile_manager(self.profile_manager)
        except Exception:
            pass
        from anydes.voz.recorder import AudioRecorder
        self._recorder = AudioRecorder(seconds=4)

        apply_theme(self.root)
        self._build_layout()
        if self._load_chat_history_from_disk():
            self.status.set("Historial cargado. Lista para conversar.")
        
        # Iniciar visión autónoma si está habilitada
        if config.vision_enabled:
            self.root.after(0, self._request_vision_session)

        # Iniciar proactividad autónoma si está habilitada
        if config.proactive_enabled:
            self._iniciar_proactividad_autonoma()

    def run(self) -> None:
        self.root.mainloop()

    def _build_layout(self) -> None:
        make_header(self.root).pack(fill="x")
        shell = ttk.Frame(self.root, style="Root.TFrame", padding=(18, 14, 18, 10))
        shell.pack(fill="both", expand=True)
        shell.columnconfigure(0, weight=0, minsize=220)
        shell.columnconfigure(1, weight=1)
        shell.columnconfigure(2, weight=0, minsize=300)
        shell.rowconfigure(0, weight=1)

        # Left quick panel with gear button to open full preferences
        self._build_quick_panel(shell)
        self._build_chat(shell)
        self._build_actions(shell)

        status_wrap = ttk.Frame(self.root, style="Root.TFrame")
        status_wrap.pack(fill="x", side="bottom", padx=18, pady=(0, 12))
        status_card, status_inner = make_card(status_wrap, padding=10)
        status_card.pack(fill="x")
        ttk.Label(status_inner, textvariable=self.status, style="Status.TLabel").pack(anchor="w")

    def _build_settings(self, parent: ttk.Frame) -> None:
        # Build a cleaner settings UI with Basic and Advanced sections.
        card_outer, panel = make_card(parent)
        card_outer.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        ttk.Label(panel, text="Preferencias", style="Title.TLabel").pack(anchor="w")
        ttk.Label(panel, text="Ajustes comunes y acceso a opciones avanzadas — INTERFAZ REDISEÑADA", style="Small.TLabel").pack(anchor="w", pady=(0, 10))

        # BASIC section
        basic_card, basic_inner = make_card(panel, padding=8)
        basic_card.pack(fill="x", pady=(0, 8))
        ttk.Label(basic_inner, text="Configuración Básica", style="Body.TLabel").pack(anchor="w")

        self._combo(basic_inner, "Cerebro", self.ai_provider, ["local", "ollama", "groq", "openai"])
        self._entry(basic_inner, "Modelo IA", self.ai_model)
        self._combo(basic_inner, "Escucha (STT)", self.stt_provider, ["vosk", "windows", "openai", "none"])
        self._combo(basic_inner, "Voz (TTS)", self.tts_provider, ["piper", "windows", "edge", "openai", "none"])
        self._combo(
            basic_inner,
            "Modo de voz",
            self.voice_activation,
            ["button", "continuous", "continuous_wake_word"],
        )
        self._entry(basic_inner, "Palabra clave (wake)", self.wake_word)
        ttk.Checkbutton(
            basic_inner,
            text="Anydes puede hablar sola (proactividad)",
            variable=self.proactive_enabled,
            style="Dark.TCheckbutton",
        ).pack(anchor="w", pady=(8, 4))

        # Vision basic toggles and reactivity
        ttk.Separator(basic_inner).pack(fill="x", pady=8)
        ttk.Label(basic_inner, text="Visión de pantalla (básico)", style="Body.TLabel").pack(anchor="w")
        ttk.Checkbutton(
            basic_inner,
            text="Activar visión de pantalla",
            variable=self.vision_enabled,
            style="Dark.TCheckbutton",
        ).pack(anchor="w", pady=(6, 4))

        ttk.Label(basic_inner, text="Reactividad de Anydes (0-100%)", style="Body.TLabel").pack(anchor="w", pady=(8, 2))
        reactivity_frame = ttk.Frame(basic_inner, style="Card.TFrame")
        reactivity_frame.pack(fill="x", pady=(0, 4))
        reactivity_frame.columnconfigure(1, weight=1)
        reactivity_scale = ttk.Scale(
            reactivity_frame,
            from_=0,
            to=100,
            orient="horizontal",
            variable=self.vision_reactivity,
            style="Dark.Horizontal.TScale",
        )
        reactivity_scale.grid(row=0, column=0, sticky="ew")
        reactivity_label = ttk.Label(reactivity_frame, text=f"{int(config.vision_reactivity_level * 100)}%", style="Body.TLabel", width=6)
        reactivity_label.grid(row=0, column=1, padx=(8, 0))
        reactivity_scale.config(command=lambda v: reactivity_label.config(text=f"{int(float(v))}%"))

        ttk.Button(basic_inner, text="Instalar modelo visión ahora", command=self.install_vision_model, style="Ghost.TButton").pack(fill="x", pady=(8, 0))

        # ADVANCED section (collapsible)
        advanced_visible = {"v": False}
        def _toggle_advanced():
            advanced_visible["v"] = not advanced_visible["v"]
            if advanced_visible["v"]:
                advanced_frame.pack(fill="both", expand=True, pady=(8, 0))
                advanced_toggle.config(text="Ocultar Avanzado ▲")
            else:
                advanced_frame.pack_forget()
                advanced_toggle.config(text="Mostrar Avanzado ▼")

        advanced_toggle = ttk.Button(panel, text="Mostrar Avanzado ▼", command=_toggle_advanced, style="TButton")
        advanced_toggle.pack(fill="x")

        advanced_frame = ttk.Frame(panel, style="Card.TFrame")
        # populate advanced options (hidden by default)
        adv_inner = ttk.Frame(advanced_frame, style="Card.TFrame")
        adv_inner.pack(fill="both", expand=True, padx=6, pady=6)

        # Keep advanced items for power users
        self._combo(adv_inner, "Proveedor visión (avanzado)", self.vision_provider, ["auto", "ollama", "groq", "openai"])
        ttk.Checkbutton(
            adv_inner,
            text="Descargar modelo visión si falta",
            variable=self.vision_auto_pull,
            style="Dark.TCheckbutton",
        ).pack(anchor="w", pady=(0, 4))
        ttk.Checkbutton(
            adv_inner,
            text="Modo en vivo (juegos, MOBA, etc.)",
            variable=self.screen_live_mode,
            style="Dark.TCheckbutton",
        ).pack(anchor="w", pady=(0, 4))
        self._entry(adv_inner, "Intervalo captura (seg)", self.screen_interval)
        self._entry(adv_inner, "Intervalo en vivo (seg)", self.screen_live_interval)
        self._entry(adv_inner, "Modelo visión (avanzado)", self.vision_model)
        ttk.Separator(adv_inner).pack(fill="x", pady=8)
        self._entry(adv_inner, "Cada cuantos segundos (proactividad)", self.proactive_interval)

        footer = ttk.Frame(panel, style="Card.TFrame")
        footer.pack(fill="x", pady=(12, 0))
        ttk.Button(footer, text="Guardar preferencias", command=self.save_preferences, style="Accent.TButton").pack(fill="x", pady=(0, 6))
        ttk.Button(footer, text="Restablecer asistente", command=self.reset_assistant, style="Ghost.TButton").pack(fill="x")

    def _build_quick_panel(self, parent: ttk.Frame) -> None:
        """Left-side compact panel with essential controls and a gear to open full settings."""
        quick_card, quick_inner = make_card(parent, padding=12)
        quick_card.grid(row=0, column=0, sticky="nsw", padx=(0, 10), pady=(0, 0))
        quick_inner.columnconfigure(0, weight=1)

        ttk.Label(quick_inner, text="Controles rápidos", style="Title.TLabel").grid(row=0, column=0, sticky="w")

        # Profile selector
        ttk.Label(quick_inner, text="Perfil activo", style="Body.TLabel").grid(row=1, column=0, sticky="w", pady=(8, 2))
        self._profile_var = StringVar(value="")
        profiles = self.profile_manager.list_profiles()
        self._profile_box = ttk.Combobox(quick_inner, textvariable=self._profile_var, values=profiles, state="readonly", style="Dark.TCombobox")
        self._profile_box.grid(row=2, column=0, sticky="ew")

        def _refresh_profiles():
            vals = self.profile_manager.list_profiles()
            self._profile_box.config(values=vals)

        ttk.Button(quick_inner, text="Crear perfil", command=lambda: self._create_profile(_refresh_profiles), style="TButton").grid(row=3, column=0, sticky="ew", pady=(6, 0))

        # Current mode indicators
        ttk.Label(quick_inner, textvariable=self.status, style="Body.TLabel").grid(row=1, column=0, sticky="w", pady=(6, 8))

        # Speak with voice toggle (quick)
        ttk.Checkbutton(
            quick_inner,
            text="Responder con voz",
            variable=self.speak_chat,
            style="Dark.TCheckbutton",
            command=lambda: self.status.set("Responder con voz " + ("ON" if self.speak_chat.get() else "OFF")),
        ).grid(row=2, column=0, sticky="w")

        # Start/stop continuous listen with wake word
        def _toggle_listen():
            activation = self.voice_activation.get() or "continuous_wake_word"
            if self.voice_controller is None or not self.voice_controller.is_running():
                self.start_voice(activation)
            else:
                self.stop_voice()

        ttk.Button(quick_inner, text="Iniciar/Detener escucha", command=_toggle_listen, style="Accent.TButton").grid(row=3, column=0, sticky="ew", pady=(8, 8))

        # Gear button to open full settings modal
        def _open_settings():
            self._open_settings_modal()

        gear = ttk.Button(quick_inner, text="⚙️ Preferencias", command=_open_settings, style="TButton")
        gear.grid(row=4, column=0, sticky="ew")

        # Quick access buttons
        ttk.Button(quick_inner, text="Mostrar avatar", command=lambda: webbrowser.open(self.avatar_server.start()), style="Ghost.TButton").grid(row=5, column=0, sticky="ew", pady=(8, 0))
        
        # Profile voice actions
        def _enroll_voice():
            name = self._profile_var.get()
            if not name:
                messagebox.showwarning("Perfil", "Selecciona o crea primero un perfil.", parent=self.root)
                return
            try:
                tmp = PROJECT_DIR / "datos" / "personas" / name / "samples"
                tmp.mkdir(parents=True, exist_ok=True)
                out = tmp / "enroll.wav"
                self.status.set("Grabando muestra de voz para registro...")
                self._recorder.record_wav(out)
                match = self.profile_manager.enroll_voice(name, out)
                messagebox.showinfo("Registro voz", f"Voz registrada: {match.label()}", parent=self.root)
                self.status.set("Registro de voz completado.")
            except Exception as exc:
                messagebox.showerror("Error registro", f"No pude registrar la voz: {exc}", parent=self.root)

        def _identify_voice():
            try:
                tmp = PROJECT_DIR / "datos" / "personas" / "tmp"
                tmp.mkdir(parents=True, exist_ok=True)
                out = tmp / "identify.wav"
                self.status.set("Grabando muestra para identificar...")
                self._recorder.record_wav(out)
                match = self.profile_manager.identify_voice(out)
                if match.matched and match.name:
                    # set active person
                    prefs = self.profile_manager.load_profile(match.name) or {}
                    self.orchestrator.set_active_person(match.name, prefs.get("preferences"))
                    self._profile_var.set(match.name)
                    self._profile_box.set(match.name)
                    messagebox.showinfo("Identificacion", f"Te reconocí como: {match.label()}", parent=self.root)
                    self.status.set(f"Voz identificada: {match.label()}")
                else:
                    messagebox.showinfo("Identificacion", f"Voz no reconocida (confianza {match.confidence:.0%})", parent=self.root)
                    self.status.set("Voz no reconocida.")
            except Exception as exc:
                messagebox.showerror("Error identificar", f"No pude identificar la voz: {exc}", parent=self.root)

            ttk.Button(quick_inner, text="Registrar voz (perfil)", command=_enroll_voice, style="Ghost.TButton").grid(row=6, column=0, sticky="ew", pady=(8, 2))
            ttk.Button(quick_inner, text="Identificar voz ahora", command=_identify_voice, style="Ghost.TButton").grid(row=7, column=0, sticky="ew")

            # Face enrol/identify using webcam (optional, requires opencv + face_recognition)
            def _capture_frame_from_camera():
                try:
                    import cv2
                    import numpy as _np
                except Exception as exc:
                    raise RuntimeError("OpenCV (opencv-python) and numpy are required for camera capture") from exc
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    # try without CAP_DSHOW on some setups
                    cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    raise RuntimeError("No pude abrir la camara (device 0)")
                ret, frame = cap.read()
                cap.release()
                if not ret or frame is None:
                    raise RuntimeError("No pude capturar imagen de la camara")
                # Convert BGR (OpenCV) to RGB for face_recognition
                rgb = frame[:, :, ::-1]
                return rgb

            def _enroll_face():
                name = self._profile_var.get()
                if not name:
                    messagebox.showwarning("Perfil", "Selecciona o crea primero un perfil.", parent=self.root)
                    return
                try:
                    self.status.set("Capturando rostro (cámara)...")
                    rgb = _capture_frame_from_camera()
                    # Save a copy of the image to the persona folder
                    person_dir = PROJECT_DIR / "datos" / "personas" / name
                    person_dir.mkdir(parents=True, exist_ok=True)
                    from PIL import Image
                    import time

                    img_path = person_dir / f"face_{int(time.time())}.jpg"
                    Image.fromarray(rgb).save(img_path)
                    # store embedding and image path
                    try:
                        self.profile_manager.add_face_embedding(name, rgb)
                        self.profile_manager.add_face_image(name, img_path)
                    except Exception as exc:
                        messagebox.showerror("Error rostro", f"No pude registrar el rostro: {exc}", parent=self.root)
                        self.status.set("Error registro rostro.")
                        return
                    messagebox.showinfo("Registro rostro", f"Rostro registrado para perfil: {name}", parent=self.root)
                    self.status.set("Registro de rostro completado.")
                except Exception as exc:
                    messagebox.showerror("Error registro rostro", f"No pude acceder a la camara o procesar la imagen: {exc}", parent=self.root)

            def _identify_face():
                try:
                    self.status.set("Capturando rostro para identificar...")
                    rgb = _capture_frame_from_camera()
                    match_name, dist = self.profile_manager.identify_face(rgb)
                    if match_name:
                        prefs = self.profile_manager.load_profile(match_name) or {}
                        self.orchestrator.set_active_person(match_name, prefs.get("preferences"))
                        self._profile_var.set(match_name)
                        self._profile_box.set(match_name)
                        messagebox.showinfo("Identificacion rostro", f"Te reconocí como: {match_name} (dist {dist:.3f})", parent=self.root)
                        self.status.set(f"Rostro identificado: {match_name}")
                    else:
                        messagebox.showinfo("Identificacion rostro", "Rostro no reconocido.", parent=self.root)
                        self.status.set("Rostro no reconocido.")
                except Exception as exc:
                    messagebox.showerror("Error identificar rostro", f"No pude identificar el rostro: {exc}", parent=self.root)

            ttk.Button(quick_inner, text="Registrar rostro (perfil)", command=_enroll_face, style="Ghost.TButton").grid(row=8, column=0, sticky="ew", pady=(8, 2))
            ttk.Button(quick_inner, text="Identificar rostro ahora", command=_identify_face, style="Ghost.TButton").grid(row=9, column=0, sticky="ew")

    def _open_settings_modal(self) -> None:
        if getattr(self, "_settings_window", None) is not None:
            try:
                self._settings_window.lift()
                return
            except Exception:
                self._settings_window = None

        toplevel = Toplevel(self.root)
        self._settings_window = toplevel
        toplevel.title("Preferencias avanzadas")
        toplevel.geometry("760x680")
        apply_theme(toplevel)
        toplevel.transient(self.root)
        toplevel.grab_set()
        container = ttk.Frame(toplevel, style="Root.TFrame", padding=12)
        container.pack(fill="both", expand=True)
        # Reuse the existing settings builder inside the modal container
        self._build_settings(container)

        def _on_close() -> None:
            try:
                toplevel.grab_release()
            except Exception:
                pass
            try:
                toplevel.destroy()
            finally:
                self._settings_window = None

        toplevel.protocol("WM_DELETE_WINDOW", _on_close)

    def _create_profile(self, refresh_cb=None) -> None:
        name = simpledialog.askstring("Crear perfil", "Nombre del perfil (ej: DemoUser)", parent=self.root)
        if not name:
            return
        try:
            self.profile_manager.create_profile(name)
            if refresh_cb:
                refresh_cb()
            messagebox.showinfo("Perfil", f"Perfil '{name}' creado.", parent=self.root)
        except Exception as exc:
            messagebox.showerror("Error", f"No pude crear el perfil: {exc}", parent=self.root)

    def _build_chat(self, parent: ttk.Frame) -> None:
        card_outer, panel = make_card(parent, padding=16)
        card_outer.grid(row=0, column=1, sticky="nsew")
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)

        header = ttk.Frame(panel, style="Card.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Chat con Anydes", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(
            header,
            text="Responder con voz",
            variable=self.speak_chat,
            style="Dark.TCheckbutton",
        ).grid(row=0, column=1, sticky="e")

        log_wrap = ttk.Frame(panel, style="Card.TFrame")
        log_wrap.grid(row=1, column=0, sticky="nsew")
        log_wrap.columnconfigure(0, weight=1)
        log_wrap.rowconfigure(0, weight=1)

        self.log = ScrolledText(
            log_wrap,
            wrap="word",
            height=20,
            bg=AnydesTheme.input_bg,
            fg=AnydesTheme.text,
            insertbackground=AnydesTheme.accent,
            relief="flat",
            font=AnydesTheme.font_chat,
            padx=14,
            pady=12,
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=AnydesTheme.input_border,
            highlightcolor=AnydesTheme.accent,
        )
        self.log.grid(row=0, column=0, sticky="nsew")
        self.log.configure(state="disabled")
        self.log.tag_configure("speaker_user", foreground=AnydesTheme.lavender, font=("Segoe UI", 10, "bold"))
        self.log.tag_configure("speaker_anydes", foreground=AnydesTheme.accent_soft, font=("Segoe UI", 10, "bold"))
        self.log.tag_configure("speaker_system", foreground=AnydesTheme.text_muted, font=("Segoe UI", 10, "italic"))
        self.log.tag_configure("body", foreground=AnydesTheme.text)

        composer = ttk.Frame(panel, style="Card.TFrame")
        composer.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        composer.columnconfigure(0, weight=1)
        self.message = StringVar()
        entry = ttk.Entry(composer, textvariable=self.message, style="Dark.TEntry")
        entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        entry.bind("<Return>", lambda _event: self.send_message())
        ttk.Button(composer, text="Enviar ✦", command=self.send_message, style="Accent.TButton").grid(row=0, column=1)

    def _build_actions(self, parent: ttk.Frame) -> None:
        card_outer, panel = make_card(parent)
        card_outer.grid(row=0, column=2, sticky="nsew", padx=(10, 0))

        ttk.Label(panel, text="Acciones", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            panel,
            text="Modos rapidos sin usar la terminal.",
            style="Small.TLabel",
            wraplength=250,
        ).pack(anchor="w", pady=(0, 12))

        ttk.Label(panel, text="Voz y cuerpo", style="Body.TLabel").pack(anchor="w", pady=(0, 6))
        actions = [
            ("✦ Abrir cuerpo virtual", self.open_avatar),
            ("🎙 Voz por boton", lambda: self.start_voice("button")),
            ("🎤 Escuchar ahora", self.listen_now),
            ("👂 Escucha continua", lambda: self.start_voice("continuous")),
            ("🔑 Con palabra clave", lambda: self.start_voice("continuous_wake_word")),
            ("⏹ Detener voz", self.stop_voice),
        ]
        for label, command in actions:
            ttk.Button(panel, text=label, command=command, style="Ghost.TButton").pack(fill="x", pady=3)

        ttk.Separator(panel).pack(fill="x", pady=12)
        ttk.Label(panel, text="Pantalla y apps", style="Body.TLabel").pack(anchor="w", pady=(0, 6))
        quick_actions = [
            ("👀 Mira mi pantalla", "mira mi pantalla"),
            ("📺 Compartir pantalla", "comparte pantalla"),
            ("🛑 Dejar de ver pantalla", "deja de ver mi pantalla"),
            ("▶ YouTube", "abre youtube"),
            ("🔎 Google", "busca en google "),
            ("🎵 Spotify", "abre spotify"),
            ("💬 Memoria reciente", "que hablamos recien?"),
        ]
        for label, prompt in quick_actions:
            ttk.Button(
                panel,
                text=label,
                command=lambda text=prompt: self.use_prompt(text),
                style="Quick.TButton",
            ).pack(fill="x", pady=3)

        ttk.Separator(panel).pack(fill="x", pady=12)
        ttk.Button(panel, text="Carpeta de diseno", command=self.open_design_folder, style="Ghost.TButton").pack(fill="x")

    def _combo(self, parent: ttk.Frame, label: str, variable: StringVar, values: list[str]) -> None:
        ttk.Label(parent, text=label, style="Body.TLabel").pack(anchor="w", pady=(8, 2))
        box = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly", style="Dark.TCombobox")
        box.pack(fill="x")
        box.bind("<<ComboboxSelected>>", lambda _event: self.reset_assistant())

    def _entry(self, parent: ttk.Frame, label: str, variable: StringVar) -> None:
        ttk.Label(parent, text=label, style="Body.TLabel").pack(anchor="w", pady=(8, 2))
        field = ttk.Entry(parent, textvariable=variable, style="Dark.TEntry")
        field.pack(fill="x")
        field.bind("<FocusOut>", lambda _event: self.reset_assistant())

    def send_message(self) -> None:
        text = self.message.get().strip()
        if not text:
            return
        self._notify_user_activity()
        self.message.set("")
        normalized = text.lower()
        if normalized in {"salir", "exit", "quit", "detener voz", "parar voz"}:
            self.stop_voice()
            self._append_log("Sistema", "Voz detenida.")
            return
        self._append_log("Tu", text)
        self.status.set("Pensando...")
        threading.Thread(target=self._respond_in_background, args=(text,), daemon=True).start()

    def use_prompt(self, text: str) -> None:
        self.message.set(text)
        if not text.endswith(" "):
            self.send_message()

    def _respond_in_background(self, text: str) -> None:
        try:
            self._notify_user_activity()
            if self.orchestrator is not None:
                self.orchestrator.on_user_message(text)

            # Notificar a la visión que Anydes va a hablar (evita interrupciones)
            if self.vision_controller is not None:
                self.vision_controller.notificar_hablando(True)
            
            assistant = self._assistant_for_current_settings()
            response = assistant.respond(text, interactive=True, input_mode="typed")
            self.root.after(0, lambda: self._append_log("Anydes", response))
            if self.orchestrator is not None:
                self.orchestrator.on_assistant_response(response)
            if self.speak_chat.get():
                self.shared_speaking_event.set()
                try:
                    self._speak_intention_with_tts_fallback(
                        assistant.last_intention or self._voice_intention_from_text(response)
                    )
                finally:
                    self.shared_speaking_event.clear()
            self.root.after(0, lambda: self.status.set("Lista."))
        except Exception as exc:
            if self.assistant is not None:
                self.assistant.audit.record(
                    "interface_response_failure",
                    error_type=type(exc).__name__,
                    detail=str(exc)[:500],
                )
            self.root.after(0, lambda: self.status.set("Necesito intentarlo otra vez."))
            self.root.after(
                0,
                lambda: self._append_log(
                    "Anydes",
                    "Perdon... algo interrumpio mi respuesta. Podrias repetirmelo?",
                ),
            )
        finally:
            # Notificar que terminó de hablar
            if self.vision_controller is not None:
                self.vision_controller.notificar_hablando(False)

    def _assistant_for_current_settings(self) -> AnydesAssistant:
        signature = repr(self._env_values())
        if self.assistant is not None and self.assistant_signature == signature:
            return self.assistant

        cfg = self._current_config()
        verifier = SpeakerVerifier(cfg)
        # Resolve brain with fallback providers if building fails
        brain = None
        vision_brain = None
        last_err: Exception | None = None
        for provider in ([cfg.ai_provider] + [p for p in FALLBACK_AI_PROVIDERS if p != cfg.ai_provider]):
            try:
                cfg_try = replace(cfg, ai_provider=provider)
                brain = build_brain(cfg_try)
                cfg = cfg_try
                break
            except Exception as e:
                last_err = e
                continue
        if brain is None:
            raise last_err or RuntimeError("No pude inicializar ningun cerebro AI")

        # Vision brain fallback
        last_err = None
        for provider in ([cfg.vision_provider] + [p for p in ["auto", "ollama", "groq", "openai"] if p != cfg.vision_provider]):
            try:
                cfg_try = replace(cfg, vision_provider=provider)
                vision_brain = build_vision_brain(cfg_try)
                cfg = cfg_try
                break
            except Exception as e:
                last_err = e
                continue
        if vision_brain is None:
            # not critical, proceed without vision_brain
            vision_brain = None

        self.assistant = AnydesAssistant(
            config=cfg,
            tools=ToolRegistry.default(cfg),
            permissions=GuiPermissionManager(
                self.root,
                verifier=verifier,
                vision_enabled=cfg.vision_enabled,
            ),
            audit=AuditLog(cfg.audit_log_path),
            personality=PersonalityProfile.load(cfg.personality_prompt_path),
            brain=brain,
            vision_brain=vision_brain,
        )
        self.assistant_signature = signature
        return self.assistant

    def _current_config(self) -> Config:
        try:
            voice_seconds = int(self.voice_seconds.get())
        except ValueError:
            voice_seconds = config.voice_seconds
        try:
            proactive_interval = int(self.proactive_interval.get())
        except ValueError:
            proactive_interval = config.proactive_interval_seconds
        try:
            screen_interval = int(self.screen_interval.get())
        except ValueError:
            screen_interval = config.screen_watch_interval_seconds
        try:
            screen_live_interval = float(self.screen_live_interval.get())
        except ValueError:
            screen_live_interval = config.screen_live_interval_seconds

        return replace(
            config,
            ai_provider=self.ai_provider.get(),
            ai_model=self.ai_model.get().strip() or config.ai_model,
            stt_provider=self.stt_provider.get(),
            tts_provider=self.tts_provider.get(),
            voice_activation=self.voice_activation.get(),
            wake_word=self.wake_word.get().strip() or config.wake_word,
            voice_seconds=voice_seconds,
            proactive_enabled=bool(self.proactive_enabled.get()),
            proactive_interval_seconds=proactive_interval,
            vision_enabled=bool(self.vision_enabled.get()),
            vision_provider=self.vision_provider.get().strip() or "auto",
            vision_model=self.vision_model.get().strip(),
            vision_auto_pull=bool(self.vision_auto_pull.get()),
            vision_reactivity_level=float(self.vision_reactivity.get()),
            screen_watch_interval_seconds=screen_interval,
            screen_live_mode=bool(self.screen_live_mode.get()),
            screen_live_interval_seconds=screen_live_interval,
            screen_fresh_on_analyze=bool(self.screen_fresh_on_analyze.get()),
        )

    def _env_values(self) -> dict[str, str]:
        return {
            "ANYDES_AI_PROVIDER": self.ai_provider.get(),
            "ANYDES_AI_MODEL": self.ai_model.get().strip() or "llama3.2",
            "ANYDES_STT_PROVIDER": self.stt_provider.get(),
            "ANYDES_TTS_PROVIDER": self.tts_provider.get(),
            "ANYDES_VOICE_ACTIVATION": self.voice_activation.get(),
            "ANYDES_WAKE_WORD": self.wake_word.get().strip() or "anydes",
            "ANYDES_VOICE_SECONDS": self.voice_seconds.get().strip() or "6",
            "ANYDES_PROACTIVE_ENABLED": "true" if self.proactive_enabled.get() else "false",
            "ANYDES_PROACTIVE_INTERVAL_SECONDS": self.proactive_interval.get().strip() or "1800",
            "ANYDES_VISION_ENABLED": "true" if self.vision_enabled.get() else "false",
            "ANYDES_VISION_PROVIDER": self.vision_provider.get().strip() or "auto",
            "ANYDES_VISION_MODEL": self.vision_model.get().strip(),
            "ANYDES_VISION_AUTO_PULL": "true" if self.vision_auto_pull.get() else "false",
            "ANYDES_VISION_REACTIVITY_LEVEL": str(self.vision_reactivity.get()),
            "ANYDES_SCREEN_INTERVAL_SECONDS": self.screen_interval.get().strip() or "5",
            "ANYDES_SCREEN_LIVE_MODE": "true" if self.screen_live_mode.get() else "false",
            "ANYDES_SCREEN_LIVE_INTERVAL_SECONDS": self.screen_live_interval.get().strip() or "1.5",
            "ANYDES_SCREEN_FRESH_ON_ANALYZE": "true" if self.screen_fresh_on_analyze.get() else "false",
        }

    def install_vision_model(self) -> None:
        cfg = self._current_config()
        provider = cfg.resolved_vision_provider()
        if provider == "groq":
            messagebox.showinfo(
                "Modelo de vision",
                (
                    "Vision con Groq usa la API en la nube (no hace falta instalar nada local). "
                    "Solo necesitas GROQ_API_KEY y un modelo como "
                    "meta-llama/llama-4-scout-17b-16e-instruct."
                ),
                parent=self.root,
            )
            return
        if provider == "openai":
            messagebox.showinfo(
                "Modelo de vision",
                (
                    "Vision con OpenAI usa la API en la nube. Define OPENAI_API_KEY, "
                    "ANYDES_NETWORK_ENABLED=true y ANYDES_ALLOW_EXTERNAL_VISION=true. "
                    "Tambien se pedira consentimiento en cada sesion."
                ),
                parent=self.root,
            )
            return
        if provider != "ollama":
            messagebox.showinfo(
                "Modelo de vision",
                "Para instalar un modelo local, elige Proveedor vision = ollama en preferencias.",
                parent=self.root,
            )
            return

        model_name = cfg.resolved_vision_model() or "moondream"
        self.status.set(f"Descargando modelo de vision `{model_name}`... puede tardar unos minutos.")
        threading.Thread(target=self._install_vision_model_background, args=(cfg, model_name), daemon=True).start()

    def _install_vision_model_background(self, cfg: Config, model_name: str) -> None:
        ok, detail = pull_ollama_model(cfg.ollama_url, model_name)
        if ok:
            resolved, _error = resolve_ollama_vision_model(
                cfg.ollama_url,
                model_name,
                auto_pull=False,
            )
            if resolved:
                self.root.after(0, lambda: self.vision_model.set(resolved))
                self.root.after(0, lambda: self.status.set(f"Modelo de vision listo: {resolved}"))
                self.root.after(0, self.reset_assistant)
                return
        self.root.after(
            0,
            lambda: self.status.set(f"No pude instalar `{model_name}`: {detail[:120]}"),
        )

    def _speak_with_tts_fallback(self, text: str) -> None:
        """Intentar hablar usando TTS con orden de fallback si falla el proveedor actual."""
        self._speak_intention_with_tts_fallback(self._voice_intention_from_text(text))

    def _speak_intention_with_tts_fallback(self, intention: Intention) -> None:
        """Intentar hablar usando TTS desde una Intention, con orden de fallback."""
        cfg = self._current_config()
        tried = []
        last_exc: Exception | None = None
        for provider in ([cfg.tts_provider] + [p for p in FALLBACK_TTS if p != cfg.tts_provider]):
            try:
                cfg_try = replace(cfg, tts_provider=provider)
                voice_manager = build_voice_manager(cfg_try)
                result = voice_manager.speak(intention.speech_intention)
                if result.status == "failed":
                    raise RuntimeError(result.error or "TTS failed")
                # update config choice for future
                return
            except Exception as exc:
                last_exc = exc
                tried.append(provider)
                continue
        # If we reach here, none worked; log error in UI
        self.root.after(0, lambda: self._append_log("Sistema", f"No pude reproducir TTS (proveedores probados: {tried}): {last_exc}"))

    def _voice_intention_from_text(self, text: str) -> Intention:
        return Intention(
            id="intention:gui_voice_output",
            intent_type="compatibility:gui_voice_output",
            speech=text,
            avatar={
                "expression": "speaking" if text.strip() else "neutral",
                "gesture": "talk" if text.strip() else "idle",
            },
            confidence=0.9,
            reasoning_summary="GUI compatibility adapter created an Intention for TTS output.",
        )

    def save_preferences(self) -> None:
        write_env_file(PROJECT_DIR / ".env", self._env_values())
        self.reset_assistant()
        if self.vision_controller is not None:
            try:
                self.vision_controller.establecer_reactividad(float(self.vision_reactivity.get()))
            except Exception:
                pass
        current_cfg = self._current_config()
        if current_cfg.proactive_enabled:
            self._iniciar_proactividad_autonoma()
        else:
            self._detener_proactividad_autonoma()
        self.status.set("Preferencias guardadas en .env.")

    def reset_assistant(self) -> None:
        self.assistant = None
        self.assistant_signature = ""
        # Reiniciar visión si cambió la configuración

    def start_voice(self, activation: str) -> None:
        self.stop_voice()
        self.voice_activation.set(activation)
        cfg = self._current_config()
        assistant = self._assistant_for_current_settings()
        if cfg.vision_enabled and session_consents.allows("screen_observation"):
            ensure_screen_watching(cfg, always=True)
        self.voice_controller = VoiceController(
            assistant=assistant,
            config=cfg,
            activation=activation,
            callbacks=VoiceCallbacks(
                on_user=lambda text: self.root.after(0, lambda t=text: self._append_log("Tu", t)),
                on_assistant=lambda text: self.root.after(0, lambda t=text: self._append_log("Anydes", t)),
                on_system=lambda text: self.root.after(0, lambda t=text: self._append_log("Sistema", t)),
                on_status=lambda text: self.root.after(0, lambda t=text: self.status.set(t)),
            ),
            embedded=True,
            shared_speaking_event=self.shared_speaking_event,
            disable_internal_proactive=True,
        )
        if self.orchestrator is not None:
            original_on_user = self.voice_controller.callbacks.on_user
            original_on_assistant = self.voice_controller.callbacks.on_assistant
            self.voice_controller.callbacks.on_user = lambda text: (
                self.orchestrator.on_user_message(text),
                self.root.after(0, lambda t=text: original_on_user(t)),
            )
            self.voice_controller.callbacks.on_assistant = lambda text: (
                self.orchestrator.on_assistant_response(text),
                self.root.after(0, lambda t=text: original_on_assistant(t)),
            )
        self.voice_controller.start()
        self.status.set(f"Modo de voz en la interfaz: {activation}.")
        if cfg.vision_enabled and get_screen_state().is_watching():
            self._append_log(
                "Sistema",
                "Vision de pantalla activa junto con la escucha continua.",
            )
        if activation == "button":
            self.listen_now()

    def listen_now(self) -> None:
        self._notify_user_activity()
        if self.voice_controller is None or not self.voice_controller.is_running():
            self.start_voice(self.voice_activation.get() or "button")
            return
        self.voice_controller.request_listen()
        self.status.set("Escuchando...")

    def stop_voice(self) -> None:
        if self.voice_controller is not None:
            self.voice_controller.stop()
            self.voice_controller = None
            self.status.set("Modo de voz detenido.")

    def _iniciar_vision_autonoma(self) -> None:
        """Inicia el controlador de visión autónoma en segundo plano."""
        if self.vision_controller is not None:
            return  # Ya está activo
        
        if not session_consents.allows("screen_observation"):
            self._request_vision_session()
            return

        try:
            cfg = self._current_config()
            self.vision_controller = ControladorVisionAutonoma(
                config=cfg,
                callback_mostrar=lambda t: self._append_log("Sistema", f"👁️ {t}"),
                callback_apend_log=self._append_log,
                speaking_event=self.shared_speaking_event,
                orchestrator=self.orchestrator,
            )
            self.vision_controller.iniciar()
            self._append_log("Sistema", "👁️ Visión autónoma activada: Anydes observa tu pantalla en tiempo real.")
        except Exception as exc:
            self._append_log("Sistema", f"⚠️ Error activando visión: {exc}")

    def _request_vision_session(self) -> None:
        cfg = self._current_config()
        if not cfg.vision_enabled or session_consents.allows("screen_observation"):
            return
        if not messagebox.askyesno(
            "Vision de pantalla",
            "Permitir que Anydes observe la pantalla durante esta sesion?",
            parent=self.root,
        ):
            self.status.set("Vision desactivada para esta sesion.")
            return
        provider = cfg.resolved_vision_provider()
        if is_external_provider(provider):
            if not external_access_decision(cfg, "external_vision").allowed:
                self.status.set("La vision externa esta bloqueada por privacidad.")
                return
            if not messagebox.askyesno(
                "Vision externa",
                f"Enviar capturas a {provider} durante esta sesion? La memoria no se incluira.",
                parent=self.root,
            ):
                self.status.set("No se enviaran capturas a servicios externos.")
                return
            session_consents.grant("external_vision")
        session_consents.grant("screen_observation")
        self._iniciar_vision_autonoma()

    def _detener_vision_autonoma(self) -> None:
        """Detiene el controlador de visión autónoma."""
        if self.vision_controller is not None:
            self.vision_controller.detener()
            self.vision_controller = None
            self._append_log("Sistema", "👁️ Visión autónoma desactivada.")

    def _iniciar_proactividad_autonoma(self) -> None:
        """Inicia el hilo de proactividad autónoma."""
        if self.proactive_daemon is not None and self.proactive_daemon.is_active():
            return

        cfg = self._current_config()
        self.proactive_daemon = ProactiveDaemon(
            config=cfg,
            callback_apend_log=lambda speaker, text: self.root.after(0, lambda s=speaker, t=text: self._append_log(s, t)),
            shared_speaking_event=self.shared_speaking_event,
            orchestrator=self.orchestrator,
        )
        self.proactive_daemon.start()

    def _detener_proactividad_autonoma(self) -> None:
        """Detiene el hilo de proactividad autónoma."""
        if self.proactive_daemon is not None:
            self.proactive_daemon.stop()
            self.proactive_daemon = None

    def _notify_user_activity(self) -> None:
        if self.proactive_daemon is not None:
            self.proactive_daemon.notify_user_activity()
        if self.orchestrator is not None:
            self.orchestrator.notify_user_activity()

    def open_avatar(self) -> None:
        url = self.avatar_server.start()
        webbrowser.open(url)
        self.status.set(f"Cuerpo virtual abierto: {url}")

    def open_design_folder(self) -> None:
        folder = filedialog.askdirectory(initialdir=str(PROJECT_DIR / "material_diseno"))
        if folder:
            self.status.set(f"Carpeta seleccionada: {folder}")
            messagebox.showinfo(
                "Diseno de Anydes",
                "Para cambiar las imagenes activas, reemplaza los PNG en programa_principal/avatar/assets.",
            )

    def _load_chat_history_from_disk(self) -> bool:
        memory = MemoryManager.from_config(self._current_config())
        turns = memory.conversation.recent_turns(limit=80)
        if not turns:
            return False
        for turn in turns:
            user = turn.user.strip()
            if user == "[iniciativa automatica]":
                self._append_log("Anydes", turn.assistant)
            else:
                self._append_log("Tu", user)
                self._append_log("Anydes", turn.assistant)
        return True

    def _append_log(self, speaker: str, text: str) -> None:
        tag = "speaker_anydes"
        if speaker.lower() in {"tu", "tú"}:
            tag = "speaker_user"
        elif speaker.lower() == "sistema":
            tag = "speaker_system"

        self.log.configure(state="normal")
        self.log.insert("end", f"{speaker}: ", tag)
        self.log.insert("end", f"{text}\n\n", "body")
        self.log.see("end")
        self.log.configure(state="disabled")

    def close(self) -> None:
        self.stop_voice()
        self._detener_vision_autonoma()
        sensor_hub.stop_all()
        stop_all_screen_watching()
        session_consents.revoke_all()
        self.avatar_server.stop()
        self.root.destroy()


def _free_port(start: int) -> int:
    for port in range(start, start + 60):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("No encontre un puerto libre para abrir el avatar.")


def main() -> None:
    from anydes.interfaz.pyside_companion.app import main as companion_main

    companion_main()
