from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QCursor, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QWidget


@dataclass(frozen=True)
class MascotVisualState:
    eye: str
    mouth: str
    blush: float
    motion: float
    gaze_x: float = 0.0
    gaze_y: float = 0.0


VISUAL_STATES: dict[str, MascotVisualState] = {
    "idle": MascotVisualState("open", "soft", 0.12, 0.55),
    "listening": MascotVisualState("open", "soft", 0.08, 0.35, -0.8, 0.0),
    "thinking": MascotVisualState("thinking", "small", 0.05, 0.28, 1.2, -0.8),
    "speaking": MascotVisualState("open", "talk", 0.12, 0.55),
    "happy": MascotVisualState("happy", "smile", 0.22, 1.0),
    "curious": MascotVisualState("wide", "small", 0.10, 0.65, 1.1, -0.4),
    "shy": MascotVisualState("shy", "soft", 0.42, 0.34, -0.5, 0.9),
    "surprised": MascotVisualState("wide", "round", 0.13, 1.05),
    "sleepy": MascotVisualState("sleepy", "soft", 0.08, 0.16, 0.0, 0.7),
    "calm": MascotVisualState("calm", "soft", 0.08, 0.20),
}


SPRITE_ASSET_DIR = Path(__file__).resolve().parent / "assets" / "mascot"
SPRITE_FOR_STATE = {
    "idle": "idle",
    "calm": "idle",
    "listening": "listening",
    "thinking": "thinking",
    "speaking": "speaking_closed",
    "happy": "happy",
    "curious": "curious",
    "shy": "shy",
    "surprised": "surprised",
    "sleepy": "sleepy",
}

SPEAKING_SPRITES = {
    "closed": "speaking_closed",
    "small": "speaking_small",
    "medium": "speaking_medium",
    "smile": "speaking_smile",
    "round": "speaking_round",
}
BLINK_SPRITES = {
    "idle": "idle_blink",
    "listening": "listening_blink",
    "thinking": "thinking_blink",
    "happy": "happy_blink",
    "curious": "curious_blink",
    "shy": "shy_blink",
    "surprised": "surprised_blink",
    "speaking_closed": "speaking_closed_blink",
    "speaking_small": "speaking_small_blink",
    "speaking_medium": "speaking_medium_blink",
    "speaking_smile": "speaking_smile_blink",
    "speaking_round": "speaking_round_blink",
}
VISEME_TO_SPEAKING_SPRITE = {
    "neutral": "closed",
    "ih": "small",
    "ee": "smile",
    "aa": "medium",
    "oh": "round",
    "ou": "round",
}


def mascot_visual_state(state: str) -> MascotVisualState:
    return VISUAL_STATES.get(str(state).strip().casefold(), VISUAL_STATES["idle"])


class MiniAnydesCanvas(QWidget):
    drag_started = Signal(QPoint)
    drag_moved = Signal(QPoint)
    drag_finished = Signal()
    chat_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        self.setMinimumSize(190, 250)
        self._state = "idle"
        self._previous_state = "idle"
        self._state_changed_at = time.perf_counter()
        self._started = time.perf_counter()
        self._dragging = False
        self._timeline: list[dict[str, object]] = []
        self._lip_started = 0.0
        self._lip_duration_ms = 0.0
        self._lip_offset_ms = 0.0
        self._speaking_shape = "closed"
        self._speaking_shape_since = time.perf_counter()
        self._force_blink = False
        self._blink_started_at = 0.0
        self._blink_interval_index = 0
        self._next_blink_at = self._started + 2.6
        self._sprites = {
            name: QPixmap(str(SPRITE_ASSET_DIR / f"{name}.png"))
            for name in (
                set(SPRITE_FOR_STATE.values())
                | set(SPEAKING_SPRITES.values())
                | set(BLINK_SPRITES.values())
            )
        }
        self._sprite_assets_ready = all(not sprite.isNull() for sprite in self._sprites.values())
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self.update)
        self._timer.start()

    @property
    def state(self) -> str:
        return self._state

    def set_state(self, state: str) -> None:
        normalized = str(state).strip().casefold()
        normalized = normalized if normalized in VISUAL_STATES else "idle"
        if normalized != self._state:
            self._previous_state = self._state
            self._state = normalized
            self._state_changed_at = time.perf_counter()
        self.update()

    def start_lip_sync(
        self,
        timeline: list[dict[str, object]],
        duration_ms: float,
        offset_ms: float = 0.0,
        start_delay_ms: float = 0.0,
    ) -> None:
        self._timeline = list(timeline)
        self._lip_started = time.perf_counter() + max(0.0, float(start_delay_ms)) / 1000.0
        self._lip_duration_ms = max(0.0, float(duration_ms))
        self._lip_offset_ms = max(0.0, float(offset_ms))
        self._speaking_shape = "closed"
        self._speaking_shape_since = time.perf_counter() - 0.08
        self.set_state("speaking")

    def stop_speaking(self) -> None:
        self._timeline = []
        self._lip_duration_ms = 0.0
        self._speaking_shape = "closed"
        if self._state == "speaking":
            self.set_state("idle")

    def set_force_blink(self, enabled: bool) -> None:
        """Hold the integrated blink frame for previews and visual tests."""
        self._force_blink = bool(enabled)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        elapsed = time.perf_counter() - self._started
        profile = mascot_visual_state(self._state)
        if self._sprite_assets_ready:
            self._paint_sprite_companion(painter, profile, elapsed)
            return
        scale = min(self.width() / 220.0, self.height() / 284.0)
        float_y = math.sin(elapsed * 1.55) * (0.7 + profile.motion * 0.75)
        body_breath = math.sin(elapsed * 1.9) * 0.7
        sway = math.sin(elapsed * 0.72) * 0.45 * profile.motion

        painter.translate(self.width() / 2.0 + sway, 3.0 + float_y)
        painter.scale(scale, scale)
        self._draw_shadow(painter, elapsed)
        self._draw_character(painter, profile, elapsed, body_breath)

    def _paint_sprite_companion(
        self,
        painter: QPainter,
        profile: MascotVisualState,
        elapsed: float,
    ) -> None:
        sprite_key = self._sprite_key_for_current_frame(elapsed)
        sprite = self._sprites[sprite_key]
        target_height = max(1.0, self.height() - 9.0)
        target_width = target_height * (sprite.width() / float(sprite.height()))
        if target_width > self.width() - 6.0:
            target_width = self.width() - 6.0
            target_height = target_width * (sprite.height() / float(sprite.width()))
        base_x = (self.width() - target_width) / 2.0
        base_y = self.height() - target_height - 2.0
        float_y = math.sin(elapsed * 1.58) * (0.65 + profile.motion * 0.9)
        if self._state == "speaking":
            float_y = 0.0
            breathing = 1.0
            sway = 0.0
        else:
            breathing = 1.0 + math.sin(elapsed * 1.92) * 0.0028
            sway = math.sin(elapsed * 0.74) * (0.22 + profile.motion * 0.34)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(23, 14, 29, 66))
        shadow_width = target_width * 0.42
        painter.drawEllipse(
            QRectF(
                self.width() / 2.0 - shadow_width / 2.0,
                self.height() - 7.0,
                shadow_width,
                6.0,
            )
        )

        now = time.perf_counter()
        transition = min(1.0, (now - self._state_changed_at) / 0.16)
        if transition < 1.0:
            previous_key = SPRITE_FOR_STATE[self._previous_state]
            painter.save()
            painter.setOpacity(1.0 - transition)
            self._draw_sprite_pose(
                painter,
                self._sprites[previous_key],
                base_x,
                base_y + float_y,
                target_width,
                target_height,
                sway,
                breathing,
            )
            painter.restore()

        painter.save()
        painter.setOpacity(transition)
        self._draw_sprite_pose(
            painter,
            sprite,
            base_x,
            base_y + float_y,
            target_width,
            target_height,
            sway,
            breathing,
        )
        painter.restore()

    def _sprite_key_for_current_frame(self, elapsed: float) -> str:
        if self._state == "speaking":
            mouth = self._current_mouth(elapsed, mascot_visual_state("speaking"))
            requested = VISEME_TO_SPEAKING_SPRITE.get(mouth, "closed")
            now = time.perf_counter()
            if requested != self._speaking_shape and (requested == "closed" or now - self._speaking_shape_since >= 0.04):
                self._speaking_shape = requested
                self._speaking_shape_since = now
            sprite_key = SPEAKING_SPRITES[self._speaking_shape]
        else:
            sprite_key = SPRITE_FOR_STATE[self._state]

        if self._blink_active() and sprite_key in BLINK_SPRITES:
            return BLINK_SPRITES[sprite_key]
        return sprite_key

    def _draw_sprite_pose(
        self,
        painter: QPainter,
        sprite: QPixmap,
        x: float,
        y: float,
        width: float,
        height: float,
        sway: float,
        breathing: float,
    ) -> None:
        center_x = x + width / 2.0
        bottom_y = y + height
        painter.translate(center_x, bottom_y)
        painter.rotate(sway)
        painter.scale(1.0, breathing)
        painter.translate(-center_x, -bottom_y)
        painter.drawPixmap(QRectF(x, y, width, height), sprite, QRectF(sprite.rect()))

    def _blink_active(self) -> bool:
        if self._force_blink:
            return True
        if self._state == "sleepy":
            return False
        now = time.perf_counter()
        if now >= self._next_blink_at:
            intervals = (3.8, 4.6, 3.3, 5.0)
            self._blink_started_at = now
            self._next_blink_at = now + intervals[self._blink_interval_index]
            self._blink_interval_index = (self._blink_interval_index + 1) % len(intervals)
        return 0.0 <= now - self._blink_started_at <= 0.17

    def _draw_shadow(self, painter: QPainter, elapsed: float) -> None:
        width = 63.0 + math.sin(elapsed * 1.55) * 1.3
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(22, 14, 28, 58))
        painter.drawEllipse(QRectF(-width / 2.0, 266.0, width, 9.0))

    def _draw_character(
        self,
        painter: QPainter,
        profile: MascotVisualState,
        elapsed: float,
        breath: float,
    ) -> None:
        outline = QPen(QColor("#3B293F"), 1.45)
        soft_outline = QPen(QColor(45, 31, 49, 175), 1.05)

        painter.setPen(outline)
        painter.setBrush(QColor("#D66F92"))
        hair_back = QPainterPath()
        hair_back.moveTo(-48, 48)
        hair_back.cubicTo(-58, 70, -52, 155, -41, 204)
        hair_back.cubicTo(-27, 226, 27, 226, 41, 204)
        hair_back.cubicTo(52, 155, 58, 70, 48, 48)
        hair_back.cubicTo(35, 16, -35, 16, -48, 48)
        painter.drawPath(hair_back)

        painter.setBrush(QColor("#343039"))
        painter.setPen(soft_outline)
        painter.drawRoundedRect(QRectF(-27, 231, 20, 35), 8, 8)
        painter.drawRoundedRect(QRectF(7, 231, 20, 35), 8, 8)

        painter.setBrush(QColor("#7C2632"))
        skirt = QPainterPath()
        skirt.moveTo(-40, 205)
        skirt.lineTo(40, 205)
        skirt.lineTo(47, 235)
        skirt.cubicTo(24, 241, -24, 241, -47, 235)
        skirt.closeSubpath()
        painter.drawPath(skirt)
        painter.setPen(QPen(QColor(177, 69, 81, 170), 1.0))
        for x in (-29, -15, 0, 15, 29):
            painter.drawLine(QPointF(x, 209), QPointF(x * 1.12, 236))

        painter.setPen(outline)
        painter.setBrush(QColor("#563067"))
        hood = QPainterPath()
        hood.moveTo(-38, 108)
        hood.cubicTo(-56, 114, -60, 146, -46, 158)
        hood.lineTo(-34, 132)
        hood.cubicTo(-19, 119, 19, 119, 34, 132)
        hood.lineTo(46, 158)
        hood.cubicTo(60, 146, 56, 114, 38, 108)
        hood.cubicTo(24, 100, -24, 100, -38, 108)
        painter.drawPath(hood)

        painter.setBrush(QColor("#663B78"))
        body = QPainterPath()
        body.moveTo(-38, 122)
        body.cubicTo(-47, 142, -47, 188, -41, 213 + breath * 0.25)
        body.cubicTo(-22, 221, 22, 221, 41, 213 + breath * 0.25)
        body.cubicTo(47, 188, 47, 142, 38, 122)
        body.cubicTo(23, 113, -23, 113, -38, 122)
        painter.drawPath(body)

        painter.setBrush(QColor("#684079"))
        left_arm = QPainterPath()
        left_arm.moveTo(-35, 127)
        left_arm.cubicTo(-58, 135, -63, 164, -49, 193)
        left_arm.cubicTo(-43, 205, -31, 201, -24, 187)
        left_arm.cubicTo(-31, 160, -28, 142, -35, 127)
        left_arm.closeSubpath()
        painter.drawPath(left_arm)
        right_arm = QPainterPath()
        right_arm.moveTo(35, 127)
        right_arm.cubicTo(58, 135, 63, 164, 49, 193)
        right_arm.cubicTo(43, 205, 31, 201, 24, 187)
        right_arm.cubicTo(31, 160, 28, 142, 35, 127)
        right_arm.closeSubpath()
        painter.drawPath(right_arm)

        painter.setPen(QPen(QColor(52, 28, 62, 170), 1.2))
        painter.setBrush(QColor("#724686"))
        pocket = QPainterPath()
        pocket.moveTo(-27, 177)
        pocket.cubicTo(-31, 184, -35, 198, -31, 207)
        pocket.cubicTo(-14, 213, 14, 213, 31, 207)
        pocket.cubicTo(35, 198, 31, 184, 27, 177)
        pocket.cubicTo(13, 181, -13, 181, -27, 177)
        painter.drawPath(pocket)
        painter.drawLine(QPointF(-27, 180), QPointF(-17, 198))
        painter.drawLine(QPointF(27, 180), QPointF(17, 198))

        painter.setPen(outline)
        painter.setBrush(QColor("#F3D8CF"))
        painter.drawEllipse(QRectF(-38, 36, 76, 83))

        painter.setBrush(QColor("#E7839E"))
        fringe = QPainterPath()
        fringe.moveTo(-42, 45)
        fringe.cubicTo(-32, 11, 31, 8, 43, 44)
        fringe.cubicTo(26, 34, 17, 55, 10, 65)
        fringe.lineTo(4, 39)
        fringe.cubicTo(-4, 57, -13, 68, -19, 72)
        fringe.lineTo(-17, 40)
        fringe.cubicTo(-25, 54, -33, 61, -39, 63)
        fringe.closeSubpath()
        painter.drawPath(fringe)
        painter.setPen(QPen(QColor(169, 76, 108, 130), 1.0))
        painter.drawLine(QPointF(-37, 68), QPointF(-38, 173))
        painter.drawLine(QPointF(37, 68), QPointF(38, 173))

        self._draw_face(painter, profile, elapsed)

        painter.setPen(QPen(QColor(50, 28, 60, 180), 1.0))
        painter.drawLine(QPointF(-10, 124), QPointF(-9, 151))
        painter.drawLine(QPointF(10, 124), QPointF(9, 151))
        painter.setBrush(QColor("#403047"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QRectF(-11, 149, 4, 5))
        painter.drawEllipse(QRectF(7, 149, 4, 5))

    def _draw_face(
        self,
        painter: QPainter,
        profile: MascotVisualState,
        elapsed: float,
    ) -> None:
        blink_phase = elapsed % 4.6
        blinking = 3.93 < blink_phase < 4.08
        eye_y = 82.0
        painter.setPen(QPen(QColor("#1A171C"), 3.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(QRectF(-31, 68, 28, 25), 9, 9)
        painter.drawRoundedRect(QRectF(3, 68, 28, 25), 9, 9)
        painter.drawLine(QPointF(-3, 77), QPointF(3, 77))

        if blinking or profile.eye == "happy":
            painter.setPen(QPen(QColor("#392B34"), 2.0))
            curve = QPainterPath()
            curve.moveTo(-25, eye_y)
            curve.quadTo(-17, eye_y + (3 if profile.eye == "happy" else 1), -9, eye_y)
            painter.drawPath(curve)
            curve = QPainterPath()
            curve.moveTo(9, eye_y)
            curve.quadTo(17, eye_y + (3 if profile.eye == "happy" else 1), 25, eye_y)
            painter.drawPath(curve)
        else:
            height = 11.0
            if profile.eye in {"sleepy", "calm", "shy", "thinking"}:
                height = 6.5 if profile.eye != "sleepy" else 4.5
            painter.setPen(QPen(QColor("#43303A"), 1.2))
            painter.setBrush(QColor("#F9F7F4"))
            painter.drawEllipse(QRectF(-25, eye_y - height / 2, 16, height))
            painter.drawEllipse(QRectF(9, eye_y - height / 2, 16, height))
            pupil_y = eye_y - 4 + profile.gaze_y
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#B84D76"))
            painter.drawEllipse(QRectF(-20 + profile.gaze_x, pupil_y, 7.5, 9.5))
            painter.drawEllipse(QRectF(12.5 + profile.gaze_x, pupil_y, 7.5, 9.5))
            painter.setBrush(QColor("#30232E"))
            painter.drawEllipse(QRectF(-17.8 + profile.gaze_x, pupil_y + 2, 3.4, 5.2))
            painter.drawEllipse(QRectF(14.7 + profile.gaze_x, pupil_y + 2, 3.4, 5.2))
            painter.setBrush(QColor(255, 255, 255, 225))
            painter.drawEllipse(QRectF(-19 + profile.gaze_x, pupil_y + 1, 2.1, 2.1))
            painter.drawEllipse(QRectF(13.5 + profile.gaze_x, pupil_y + 1, 2.1, 2.1))

        if profile.blush > 0:
            alpha = int(130 * profile.blush)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(233, 110, 139, alpha))
            painter.drawEllipse(QRectF(-31, 95, 13, 5))
            painter.drawEllipse(QRectF(18, 95, 13, 5))

        mouth = self._current_mouth(elapsed, profile)
        painter.setPen(QPen(QColor("#A65166"), 1.6))
        painter.setBrush(QColor("#C8697F"))
        if mouth == "round":
            painter.drawEllipse(QRectF(-3.5, 102, 7, 8))
        elif mouth in {"aa", "oh"}:
            height = 7 if mouth == "aa" else 5
            painter.drawEllipse(QRectF(-5, 101, 10, height))
        elif mouth in {"ih", "ee"}:
            painter.drawRoundedRect(QRectF(-7, 103, 14, 3.4), 2, 2)
        elif mouth == "ou":
            painter.drawEllipse(QRectF(-3, 103, 6, 4))
        elif mouth == "smile":
            painter.setBrush(Qt.BrushStyle.NoBrush)
            curve = QPainterPath()
            curve.moveTo(-7, 102)
            curve.quadTo(0, 110, 7, 102)
            painter.drawPath(curve)
        elif mouth == "small":
            painter.drawEllipse(QRectF(-2.3, 104, 4.6, 3.2))
        else:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            curve = QPainterPath()
            curve.moveTo(-6, 103)
            curve.quadTo(0, 107, 6, 103)
            painter.drawPath(curve)

    def _current_mouth(self, elapsed: float, profile: MascotVisualState) -> str:
        if self._lip_duration_ms > 0:
            position_ms = (
                (time.perf_counter() - self._lip_started) * 1000.0 + self._lip_offset_ms
            )
            if position_ms < 0.0:
                return "neutral"
            if position_ms <= self._lip_duration_ms:
                for cue in self._timeline:
                    start = float(cue.get("start_ms") or 0.0)
                    end = float(cue.get("end_ms") or 0.0)
                    if start <= position_ms < end:
                        return str(cue.get("viseme") or "soft")
                return "neutral"
            # The stretched final region ends with the PCM playback; there is
            # no synthetic post-roll after the source timeline.
            return "neutral"
        if profile.mouth == "talk" and self._lip_duration_ms > 0:
            return ("aa", "ih", "ou", "neutral")[int(elapsed * 7.0) % 4]
        return profile.mouth

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
            self.drag_started.emit(event.globalPosition().toPoint())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._dragging and event.buttons() & Qt.MouseButton.LeftButton:
            self.drag_moved.emit(event.globalPosition().toPoint())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
            self.drag_finished.emit()
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.LeftButton:
            self.chat_requested.emit()
        super().mouseDoubleClickEvent(event)
