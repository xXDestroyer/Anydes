from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    background: str = "#111116"
    panel: str = "#17171D"
    panel_secondary: str = "#202028"
    hover: str = "#2A2A34"
    text: str = "#FFFFFF"
    text_secondary: str = "#D1CBD8"
    purple: str = "#6D4ACF"
    pink: str = "#F48FBF"
    border: str = "#34313D"


@dataclass(frozen=True)
class Motion:
    breath_ms: int = 5200
    blink_ms: int = 4200
    panel_ms: int = 260
    hover_ms: int = 160


@dataclass(frozen=True)
class Layout:
    avatar_ratio: float = 0.70
    chat_ratio: float = 0.20
    bottom_height: int = 82
    inner_world_width: int = 360
    scene_padding: int = 28


PALETTE = Palette()
MOTION = Motion()
LAYOUT = Layout()


def companion_stylesheet() -> str:
    """QSS principal de la experiencia PySide6.

    Mantiene una sola escena visual: superficies suaves, sin divisiones duras y
    con el avatar como centro emocional.
    """

    p = PALETTE
    return f"""
    QWidget {{
        background: {p.background};
        color: {p.text};
        font-family: "Segoe UI", "Arial";
        font-size: 14px;
    }}

    #SceneRoot {{
        background: {p.background};
    }}

    #AvatarStage {{
        background: transparent;
        border: none;
    }}

    #StatusPill {{
        background: rgba(18, 18, 24, 178);
        color: {p.text_secondary};
        border: 1px solid rgba(52, 49, 61, 125);
        border-radius: 16px;
        padding: 7px 16px;
        font-size: 13px;
    }}

    #ChatPanel {{
        background: rgba(17, 17, 22, 116);
        border: 1px solid rgba(52, 49, 61, 82);
        border-radius: 22px;
    }}

    QTextEdit {{
        background: transparent;
        color: {p.text};
        border: none;
        padding: 14px 18px;
        selection-background-color: {p.purple};
    }}

    #InputDock {{
        background: rgba(17, 17, 22, 168);
        border: 1px solid rgba(52, 49, 61, 118);
        border-radius: 27px;
    }}

    QLineEdit {{
        background: rgba(24, 24, 30, 205);
        color: {p.text};
        border: 1px solid rgba(52, 49, 61, 125);
        border-radius: 21px;
        padding: 11px 16px;
        font-size: 15px;
    }}

    QLineEdit:focus {{
        border: 1px solid {p.pink};
        background: rgba(32, 32, 40, 220);
    }}

    QPushButton {{
        background: rgba(24, 24, 30, 190);
        color: {p.text};
        border: 1px solid rgba(52, 49, 61, 120);
        border-radius: 20px;
        padding: 10px 16px;
    }}

    QPushButton:hover {{
        background: {p.hover};
        border-color: {p.pink};
    }}

    QPushButton#VoiceButton {{
        min-width: 44px;
        max-width: 44px;
        min-height: 44px;
        max-height: 44px;
        border-radius: 22px;
        font-size: 18px;
    }}

    QPushButton#SendButton {{
        background: #4B3A7F;
        border: 1px solid rgba(244, 143, 191, 130);
        border-radius: 22px;
        font-weight: 600;
        padding-left: 22px;
        padding-right: 22px;
    }}

    QPushButton#SendButton:hover {{
        background: {p.pink};
        color: {p.background};
    }}

    QPushButton#InnerWorldButton {{
        background: rgba(17, 17, 22, 132);
        border-radius: 18px;
        color: {p.text_secondary};
        padding: 8px 14px;
        font-size: 13px;
    }}

    #SensorIndicator {{
        background: rgba(17, 17, 22, 184);
        color: {p.text};
        border: 1px solid rgba(244, 143, 191, 115);
        border-radius: 16px;
        padding: 7px 12px;
        font-size: 12px;
    }}

    QPushButton#StopSensorsButton, QPushButton#ShareScreenButton {{
        background: rgba(17, 17, 22, 132);
        color: {p.text_secondary};
        border: 1px solid rgba(200, 189, 217, 42);
        border-radius: 16px;
        padding: 7px 12px;
        font-size: 12px;
    }}

    QPushButton#StopSensorsButton:hover, QPushButton#ShareScreenButton:hover {{
        background: rgba(48, 35, 71, 196);
        color: {p.text};
    }}

    QDialog {{
        background: {p.panel};
        color: {p.text};
    }}

    QComboBox {{
        min-height: 34px;
        padding: 5px 10px;
        color: {p.text};
        background: {p.panel_secondary};
        border: 1px solid {p.border};
        border-radius: 8px;
    }}

    QCheckBox {{
        color: {p.text_secondary};
        spacing: 8px;
    }}

    #InnerWorldPanel {{
        background: rgba(17, 17, 22, 228);
        border-left: 1px solid rgba(52, 49, 61, 155);
    }}

    #InnerWorldTitle {{
        color: {p.text};
        font-size: 22px;
        font-weight: 700;
        background: transparent;
    }}

    #InnerWorldSubtitle {{
        color: {p.text_secondary};
        font-size: 12px;
        background: transparent;
    }}

    #WorldCard {{
        background: rgba(24, 24, 30, 188);
        border: 1px solid rgba(52, 49, 61, 120);
        border-radius: 18px;
    }}

    #WorldCardTitle {{
        background: transparent;
        color: {p.text_secondary};
        font-size: 13px;
    }}

    #WorldCardValue {{
        background: transparent;
        color: {p.text};
        font-size: 18px;
        font-weight: 700;
    }}

    #WorldCardDescription {{
        background: transparent;
        color: {p.text_secondary};
        font-size: 12px;
        line-height: 1.35;
    }}
    """
