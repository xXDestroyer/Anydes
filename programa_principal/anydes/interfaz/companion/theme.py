from __future__ import annotations

from tkinter import Tk, ttk


class CompanionTheme:
    """Paleta visual de la interfaz Companion.

    La paleta evita el aspecto de panel tecnico: fondo oscuro, acentos suaves
    del hoodie/cabello de Anydes y superficies tipo vidrio ligero.
    """

    bg = "#08070d"
    bg_lifted = "#11101a"
    glass = "#181423"
    glass_soft = "#21182d"
    glass_border = "#3a2a4d"
    hoodie = "#7c3aed"
    hoodie_soft = "#a78bfa"
    hair = "#f472b6"
    hair_soft = "#f9a8d4"
    text = "#fbf7ff"
    text_muted = "#b9acc8"
    text_dim = "#81748f"
    white_detail = "#fff7fb"
    input_bg = "#15111f"
    success = "#9ee6c6"
    warning = "#ffd6a5"

    font_title = ("Segoe UI", 18, "bold")
    font_section = ("Segoe UI", 12, "bold")
    font_body = ("Segoe UI", 10)
    font_chat = ("Segoe UI", 10)
    font_small = ("Segoe UI", 9)
    font_button = ("Segoe UI", 10, "bold")


def apply_companion_theme(root: Tk) -> ttk.Style:
    """Aplica estilos base para componentes Tk/ttk de Companion."""

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    root.configure(bg=CompanionTheme.bg)
    style.configure("Companion.Root.TFrame", background=CompanionTheme.bg)
    style.configure("Companion.Glass.TFrame", background=CompanionTheme.glass, borderwidth=0)
    style.configure("Companion.Soft.TFrame", background=CompanionTheme.glass_soft, borderwidth=0)
    style.configure(
        "Companion.Title.TLabel",
        background=CompanionTheme.glass,
        foreground=CompanionTheme.text,
        font=CompanionTheme.font_title,
    )
    style.configure(
        "Companion.Section.TLabel",
        background=CompanionTheme.glass,
        foreground=CompanionTheme.text,
        font=CompanionTheme.font_section,
    )
    style.configure(
        "Companion.Body.TLabel",
        background=CompanionTheme.glass,
        foreground=CompanionTheme.text,
        font=CompanionTheme.font_body,
    )
    style.configure(
        "Companion.Muted.TLabel",
        background=CompanionTheme.glass,
        foreground=CompanionTheme.text_muted,
        font=CompanionTheme.font_small,
    )
    style.configure(
        "Companion.Status.TLabel",
        background=CompanionTheme.glass_soft,
        foreground=CompanionTheme.hair_soft,
        font=CompanionTheme.font_small,
    )
    style.configure(
        "Companion.Primary.TButton",
        background=CompanionTheme.hair,
        foreground=CompanionTheme.white_detail,
        borderwidth=0,
        padding=(14, 9),
        font=CompanionTheme.font_button,
    )
    style.map(
        "Companion.Primary.TButton",
        background=[("active", CompanionTheme.hair_soft), ("pressed", CompanionTheme.hoodie)],
        foreground=[("active", CompanionTheme.bg)],
    )
    style.configure(
        "Companion.Ghost.TButton",
        background=CompanionTheme.glass_soft,
        foreground=CompanionTheme.text,
        borderwidth=0,
        padding=(12, 8),
        font=CompanionTheme.font_body,
    )
    style.map(
        "Companion.Ghost.TButton",
        background=[("active", CompanionTheme.glass_border), ("pressed", CompanionTheme.hoodie)],
    )
    style.configure(
        "Companion.Icon.TButton",
        background=CompanionTheme.glass_soft,
        foreground=CompanionTheme.white_detail,
        borderwidth=0,
        padding=(10, 8),
        font=CompanionTheme.font_button,
    )
    style.map("Companion.Icon.TButton", background=[("active", CompanionTheme.hoodie)])
    style.configure(
        "Companion.TEntry",
        fieldbackground=CompanionTheme.input_bg,
        foreground=CompanionTheme.text,
        bordercolor=CompanionTheme.glass_border,
        lightcolor=CompanionTheme.glass_border,
        darkcolor=CompanionTheme.glass_border,
        insertcolor=CompanionTheme.hair,
        padding=10,
    )
    return style
