from __future__ import annotations

from tkinter import BOTH, E, N, S, W, Frame, Label, ttk


class AnydesTheme:
    bg = "#120c18"
    bg_soft = "#1a1224"
    card = "#221830"
    card_border = "#3d2a52"
    accent = "#f472b6"
    accent_soft = "#fb7185"
    accent_dark = "#be185d"
    lavender = "#c4b5fd"
    text = "#f8f4ff"
    text_muted = "#a899bc"
    input_bg = "#2b1c3d"
    input_border = "#4c3568"
    user_bubble = "#7c3aed"
    bot_bubble = "#db2777"
    success = "#34d399"
    status_bg = "#1f1430"

    font_title = ("Segoe UI", 22, "bold")
    font_subtitle = ("Segoe UI", 10)
    font_section = ("Segoe UI", 13, "bold")
    font_body = ("Segoe UI", 10)
    font_small = ("Segoe UI", 9)
    font_button = ("Segoe UI", 10, "bold")
    font_chat = ("Segoe UI", 11)


def apply_theme(root) -> ttk.Style:
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    root.configure(bg=AnydesTheme.bg)

    style.configure("Root.TFrame", background=AnydesTheme.bg)
    style.configure("Card.TFrame", background=AnydesTheme.card, borderwidth=0)
    style.configure("Header.TFrame", background=AnydesTheme.bg_soft)
    style.configure("Title.TLabel", background=AnydesTheme.card, foreground=AnydesTheme.text, font=AnydesTheme.font_section)
    style.configure("Hero.TLabel", background=AnydesTheme.bg_soft, foreground=AnydesTheme.text, font=AnydesTheme.font_title)
    style.configure("HeroSub.TLabel", background=AnydesTheme.bg_soft, foreground=AnydesTheme.text_muted, font=AnydesTheme.font_subtitle)
    style.configure("Body.TLabel", background=AnydesTheme.card, foreground=AnydesTheme.text, font=AnydesTheme.font_body)
    style.configure("Small.TLabel", background=AnydesTheme.card, foreground=AnydesTheme.text_muted, font=AnydesTheme.font_small)
    style.configure("Status.TLabel", background=AnydesTheme.status_bg, foreground=AnydesTheme.lavender, font=AnydesTheme.font_small)

    style.configure(
        "Accent.TButton",
        font=AnydesTheme.font_button,
        background=AnydesTheme.accent,
        foreground=AnydesTheme.text,
        borderwidth=0,
        padding=(14, 10),
    )
    style.map(
        "Accent.TButton",
        background=[("active", AnydesTheme.accent_soft), ("pressed", AnydesTheme.accent_dark)],
        foreground=[("active", AnydesTheme.text)],
    )

    style.configure(
        "Ghost.TButton",
        font=AnydesTheme.font_body,
        background=AnydesTheme.input_bg,
        foreground=AnydesTheme.text,
        borderwidth=1,
        padding=(12, 8),
    )
    style.map(
        "Ghost.TButton",
        background=[("active", AnydesTheme.card_border)],
        foreground=[("active", AnydesTheme.text)],
    )

    style.configure(
        "Quick.TButton",
        font=AnydesTheme.font_body,
        background=AnydesTheme.bg_soft,
        foreground=AnydesTheme.lavender,
        borderwidth=0,
        padding=(10, 8),
    )
    style.map(
        "Quick.TButton",
        background=[("active", AnydesTheme.card_border)],
        foreground=[("active", AnydesTheme.text)],
    )

    style.configure(
        "Dark.TCombobox",
        fieldbackground=AnydesTheme.input_bg,
        background=AnydesTheme.input_bg,
        foreground=AnydesTheme.text,
        arrowcolor=AnydesTheme.accent,
        bordercolor=AnydesTheme.input_border,
        lightcolor=AnydesTheme.input_border,
        darkcolor=AnydesTheme.input_border,
    )
    style.configure(
        "Dark.TEntry",
        fieldbackground=AnydesTheme.input_bg,
        foreground=AnydesTheme.text,
        bordercolor=AnydesTheme.input_border,
        lightcolor=AnydesTheme.input_border,
        darkcolor=AnydesTheme.input_border,
        insertcolor=AnydesTheme.accent,
    )
    style.configure(
        "Dark.TCheckbutton",
        background=AnydesTheme.card,
        foreground=AnydesTheme.text,
        font=AnydesTheme.font_body,
    )
    style.map(
        "Dark.TCheckbutton",
        background=[("active", AnydesTheme.card)],
        foreground=[("active", AnydesTheme.text)],
    )

    return style


def make_card(parent, *, padding: int = 18) -> Frame:
    outer = Frame(parent, bg=AnydesTheme.card_border, bd=0, highlightthickness=0)
    inner = Frame(outer, bg=AnydesTheme.card, bd=0, highlightthickness=0, padx=padding, pady=padding)
    inner.pack(fill=BOTH, expand=True, padx=1, pady=1)
    return outer, inner


def make_header(parent) -> Frame:
    header = Frame(parent, bg=AnydesTheme.bg_soft, bd=0, highlightthickness=0)
    accent = Frame(header, bg=AnydesTheme.accent, height=3, bd=0, highlightthickness=0)
    accent.pack(fill="x")
    body = Frame(header, bg=AnydesTheme.bg_soft, bd=0, highlightthickness=0, padx=22, pady=16)
    body.pack(fill="x")
    Label(body, text="Anydes", bg=AnydesTheme.bg_soft, fg=AnydesTheme.text, font=AnydesTheme.font_title).pack(anchor=W)
    Label(
        body,
        text="Tu asistente personal · local, cercana y lista para acompañarte",
        bg=AnydesTheme.bg_soft,
        fg=AnydesTheme.text_muted,
        font=AnydesTheme.font_subtitle,
    ).pack(anchor=W, pady=(4, 0))
    return header
