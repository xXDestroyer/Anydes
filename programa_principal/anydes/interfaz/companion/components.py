from __future__ import annotations

import math
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, VERTICAL, Canvas, Entry, Frame, Label, PhotoImage, Text, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Callable

from anydes.interfaz.companion.theme import CompanionTheme
from anydes.interfaz.companion.view_model import InnerWorldCardData


class GlassFrame(Frame):
    """Superficie reutilizable tipo vidrio para contener componentes."""

    def __init__(self, parent, *, bg: str | None = None, border: str | None = None, padx: int = 1, pady: int = 1):
        super().__init__(parent, bg=border or CompanionTheme.glass_border, bd=0, highlightthickness=0)
        self.body = Frame(self, bg=bg or CompanionTheme.glass, bd=0, highlightthickness=0)
        self.body.pack(fill=BOTH, expand=True, padx=padx, pady=pady)


class ToolButton(ttk.Button):
    """Boton compacto para acciones claras como voz, enviar o Mundo Interior."""

    def __init__(self, parent, *, text: str, command: Callable[[], None], primary: bool = False):
        super().__init__(
            parent,
            text=text,
            command=command,
            style="Companion.Primary.TButton" if primary else "Companion.Ghost.TButton",
        )


class StatusChip(GlassFrame):
    """Barra pequena para estados visibles, sin razonamiento interno."""

    def __init__(self, parent, *, text: str = "✨ Tranquila"):
        super().__init__(parent, bg=CompanionTheme.glass_soft, border=CompanionTheme.glass_soft)
        self.value = Label(
            self.body,
            text=text,
            bg=CompanionTheme.glass_soft,
            fg=CompanionTheme.hair_soft,
            font=CompanionTheme.font_small,
            padx=12,
            pady=4,
        )
        self.value.pack(anchor="center")

    def set_text(self, text: str) -> None:
        self.value.configure(text=text)


class ThoughtBar(Frame):
    """Zona de estado minima entre avatar y conversacion."""

    def __init__(self, parent):
        super().__init__(parent, bg=CompanionTheme.bg)
        self.chip = StatusChip(self)
        self.chip.pack(anchor="center")

    def set_status(self, text: str) -> None:
        self.chip.set_text(text)


class AvatarPanel(GlassFrame):
    """Escenario principal del avatar.

    Esta version usa imagen 2D/preview con animaciones de respiracion, parpadeo
    y movimiento leve. El componente esta aislado para sustituirlo despues por
    un renderer VRM/3D sin cambiar el resto de la interfaz.
    """

    def __init__(self, parent, *, image_paths: tuple[Path, ...] = ()):
        super().__init__(parent, bg=CompanionTheme.bg_lifted, border=CompanionTheme.bg_lifted)
        self.canvas = Canvas(self.body, bg=CompanionTheme.bg_lifted, bd=0, highlightthickness=0)
        self.canvas.pack(fill=BOTH, expand=True)
        self._tick = 0
        self._image: PhotoImage | None = self._load_image(image_paths)
        self._image_item = None
        self._blink_item = None
        self._name_item = None
        self.expression = "neutral"
        self.gesture = "idle"
        self.canvas.bind("<Configure>", lambda _event: self._draw())
        self.after(80, self._animate)

    def set_pose(self, *, expression: str = "neutral", gesture: str = "idle", gaze: str = "user") -> None:
        self.expression = expression
        self.gesture = gesture
        _ = gaze
        self._draw()

    def _load_image(self, paths: tuple[Path, ...]) -> PhotoImage | None:
        for path in paths:
            try:
                if path.exists():
                    image = PhotoImage(file=str(path))
                    largest = max(image.width(), image.height())
                    if largest > 620:
                        factor = max(1, math.ceil(largest / 620))
                        image = image.subsample(factor, factor)
                    return image
            except Exception:
                continue
        return None

    def _draw(self) -> None:
        self.canvas.delete("stage")
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        self.canvas.create_rectangle(0, 0, width, height, fill=CompanionTheme.bg_lifted, outline="", tags="stage")
        self.canvas.create_line(0, height - 1, width, height - 1, fill=CompanionTheme.glass_border, tags="stage")
        self.canvas.create_text(
            28,
            28,
            text="Anydes",
            anchor="nw",
            fill=CompanionTheme.text,
            font=("Segoe UI", 18, "bold"),
            tags="stage",
        )
        self.canvas.create_text(
            30,
            58,
            text="un espacio tranquilo para estar contigo",
            anchor="nw",
            fill=CompanionTheme.text_dim,
            font=CompanionTheme.font_small,
            tags="stage",
        )
        if self._image is not None:
            self._image_item = self.canvas.create_image(width // 2, height // 2 + 20, image=self._image, tags="stage")
        else:
            self._draw_fallback_avatar(width, height)
        self._blink_item = self.canvas.create_rectangle(0, 0, 0, 0, fill=CompanionTheme.bg_lifted, outline="", tags="stage")

    def _draw_fallback_avatar(self, width: int, height: int) -> None:
        cx = width // 2
        cy = height // 2 + 15
        self.canvas.create_oval(cx - 90, cy - 145, cx + 90, cy + 35, fill="#f3c3d8", outline=CompanionTheme.hair, width=2, tags="stage")
        self.canvas.create_oval(cx - 58, cy - 105, cx + 58, cy + 10, fill="#ffd7e6", outline="", tags="stage")
        self.canvas.create_arc(cx - 90, cy - 150, cx + 90, cy + 20, start=15, extent=150, fill=CompanionTheme.hair, outline="", tags="stage")
        self.canvas.create_oval(cx - 33, cy - 55, cx - 20, cy - 42, fill=CompanionTheme.bg, outline="", tags="stage")
        self.canvas.create_oval(cx + 20, cy - 55, cx + 33, cy - 42, fill=CompanionTheme.bg, outline="", tags="stage")
        mouth = "smile" if self.expression in {"happy", "speaking"} else "soft"
        if mouth == "smile":
            self.canvas.create_arc(cx - 28, cy - 35, cx + 28, cy - 5, start=200, extent=140, style="arc", outline=CompanionTheme.bg, width=2, tags="stage")
        else:
            self.canvas.create_line(cx - 18, cy - 18, cx + 18, cy - 18, fill=CompanionTheme.bg, width=2, tags="stage")
        self.canvas.create_polygon(cx - 118, cy + 150, cx - 64, cy + 24, cx + 64, cy + 24, cx + 118, cy + 150, fill=CompanionTheme.hoodie, outline=CompanionTheme.hoodie_soft, tags="stage")

    def _animate(self) -> None:
        self._tick += 1
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        if self._image_item:
            breath = math.sin(self._tick / 18) * 5
            sway = math.sin(self._tick / 40) * 4
            self.canvas.coords(self._image_item, width // 2 + sway, height // 2 + 20 + breath)
        if self._blink_item:
            blink = self._tick % 70
            if 0 <= blink <= 3:
                self.canvas.coords(self._blink_item, width // 2 - 115, height // 2 - 40, width // 2 + 115, height // 2 - 20)
            else:
                self.canvas.coords(self._blink_item, 0, 0, 0, 0)
        self.after(80, self._animate)


class ChatPanel(GlassFrame):
    """Conversacion compacta y secundaria."""

    def __init__(self, parent):
        super().__init__(parent, bg=CompanionTheme.glass)
        self.text = ScrolledText(
            self.body,
            bg=CompanionTheme.glass,
            fg=CompanionTheme.text,
            insertbackground=CompanionTheme.hair,
            relief="flat",
            wrap="word",
            height=7,
            font=CompanionTheme.font_chat,
            padx=14,
            pady=10,
        )
        self.text.pack(fill=BOTH, expand=True)
        self.text.tag_configure("user", foreground=CompanionTheme.hair_soft, spacing1=6, spacing3=3)
        self.text.tag_configure("assistant", foreground=CompanionTheme.text, spacing1=6, spacing3=3)
        self.text.tag_configure("system", foreground=CompanionTheme.text_dim, spacing1=6, spacing3=3)
        self.text.configure(state="disabled")

    def add_message(self, speaker: str, text: str, *, tag: str = "assistant") -> None:
        self.text.configure(state="normal")
        self.text.insert(END, f"{speaker}: ", tag)
        self.text.insert(END, f"{text.strip()}\n", tag)
        self.text.configure(state="disabled")
        self.text.see(END)


class InputPanel(GlassFrame):
    """Barra inferior para escribir, hablar y enviar."""

    def __init__(self, parent, *, on_send: Callable[[str], None], on_voice: Callable[[], None]):
        super().__init__(parent, bg=CompanionTheme.glass)
        self.on_send = on_send
        self.entry = Entry(
            self.body,
            bg=CompanionTheme.input_bg,
            fg=CompanionTheme.text,
            insertbackground=CompanionTheme.hair,
            relief="flat",
            font=("Segoe UI", 12),
        )
        self.entry.pack(side=LEFT, fill=BOTH, expand=True, padx=(12, 8), pady=10, ipady=9)
        self.entry.bind("<Return>", lambda _event: self._send())
        ToolButton(self.body, text="🎤", command=on_voice).pack(side=LEFT, padx=(0, 8), pady=10)
        ToolButton(self.body, text="Enviar", command=self._send, primary=True).pack(side=RIGHT, padx=(0, 12), pady=10)

    def _send(self) -> None:
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, END)
        self.on_send(text)


class InfoCard(GlassFrame):
    """Tarjeta base para descripciones naturales del Mundo Interior."""

    def __init__(self, parent, data: InnerWorldCardData):
        super().__init__(parent, bg=CompanionTheme.glass_soft, border=CompanionTheme.glass_border)
        Label(
            self.body,
            text=data.title,
            bg=CompanionTheme.glass_soft,
            fg=CompanionTheme.text,
            font=CompanionTheme.font_section,
        ).pack(anchor="w", padx=12, pady=(10, 2))
        Label(
            self.body,
            text=data.value,
            bg=CompanionTheme.glass_soft,
            fg=CompanionTheme.hair_soft,
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w", padx=12)
        Label(
            self.body,
            text=data.description,
            bg=CompanionTheme.glass_soft,
            fg=CompanionTheme.text_muted,
            font=CompanionTheme.font_small,
            wraplength=230,
            justify="left",
        ).pack(anchor="w", padx=12, pady=(4, 12))


class MemoryCard(InfoCard):
    """Tarjeta especializada para recuerdos recientes."""


class GoalCard(InfoCard):
    """Tarjeta especializada para objetivo principal."""


class EmotionCard(InfoCard):
    """Tarjeta especializada para estado emocional."""


class Sidebar(GlassFrame):
    """Panel lateral reutilizable para superficies secundarias."""

    def __init__(self, parent):
        super().__init__(parent, bg=CompanionTheme.glass)


class InnerWorldPanel(Sidebar):
    """Panel lateral ocultable con el Mundo Interior de Anydes."""

    def __init__(self, parent, *, on_close: Callable[[], None]):
        super().__init__(parent)
        header = Frame(self.body, bg=CompanionTheme.glass)
        header.pack(fill="x", padx=14, pady=(12, 4))
        Label(header, text="Mundo Interior", bg=CompanionTheme.glass, fg=CompanionTheme.text, font=CompanionTheme.font_title).pack(side=LEFT)
        ToolButton(header, text="Cerrar", command=on_close).pack(side=RIGHT)
        self.content = Frame(self.body, bg=CompanionTheme.glass)
        self.content.pack(fill=BOTH, expand=True, padx=12, pady=(4, 12))

    def set_cards(self, cards: tuple[InnerWorldCardData, ...]) -> None:
        for child in self.content.winfo_children():
            child.destroy()
        for card in cards:
            cls = InfoCard
            if "Recuerdos" in card.title:
                cls = MemoryCard
            elif "Objetivo" in card.title:
                cls = GoalCard
            elif "emocional" in card.title:
                cls = EmotionCard
            item = cls(self.content, card)
            item.pack(fill="x", pady=(0, 10))
