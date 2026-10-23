from __future__ import annotations

import re
import threading
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Literal

from anydes.nucleo.config import Config
from anydes.nucleo.consent import (
    external_access_decision,
    is_external_provider,
    session_consents,
)
from anydes.vision.capture import CaptureSource, list_capture_sources
from anydes.vision.observation_store import get_vision_observation_store
from anydes.vision.state import get_screen_state
from anydes.vision.watcher import ScreenWatcher


VisionSessionStatus = Literal[
    "stopped",
    "awaiting_permission",
    "starting",
    "active",
    "degraded",
]


@dataclass(frozen=True)
class VisionSessionSnapshot:
    status: VisionSessionStatus
    authorized: bool
    frame_available: bool
    source_id: str = ""
    source_kind: str = ""
    source_label: str = ""
    provider: str = "local"
    external_analysis_authorized: bool = False
    started_at: str = ""
    last_error: str = ""


class VisionSessionController:
    """Owns one explicit, volatile screen-sharing permission session."""

    def __init__(
        self,
        config: Config,
        *,
        watcher: ScreenWatcher | None = None,
        sources_provider: Callable[[], tuple[CaptureSource, ...]] = list_capture_sources,
    ) -> None:
        self.config = config
        self._watcher = watcher or ScreenWatcher(config)
        self._sources_provider = sources_provider
        self._status: VisionSessionStatus = "stopped"
        self._started_at = ""
        self._last_error = ""
        self._lock = threading.RLock()

    def available_sources(self) -> tuple[CaptureSource, ...]:
        return self._sources_provider()

    def request_permission(self) -> tuple[CaptureSource, ...]:
        with self._lock:
            if self._status != "active":
                self._status = "awaiting_permission"
                self._last_error = ""
        try:
            return self.available_sources()
        except Exception as exc:
            with self._lock:
                self._status = "degraded"
                self._last_error = type(exc).__name__
            return ()

    def start(
        self,
        source_id: str,
        *,
        allow_external_analysis: bool = False,
    ) -> VisionSessionSnapshot:
        sources = {source.id: source for source in self.available_sources()}
        source = sources.get(source_id)
        if source is None:
            return self._fail("capture_source_unavailable")

        provider = self.config.resolved_vision_provider()
        external = is_external_provider(provider)
        if external and allow_external_analysis:
            access = external_access_decision(
                self.config,
                "external_vision",
                require_session=False,
            )
            if not access.allowed:
                return self._fail(access.reason)

        with self._lock:
            self._status = "starting"
            self._last_error = ""
        get_vision_observation_store().clear()
        get_screen_state().clear_capture()
        session_consents.grant("screen_observation")
        if external and allow_external_analysis:
            session_consents.grant("external_vision")
        else:
            session_consents.revoke("external_vision")

        if not self._watcher.start(source):
            session_consents.revoke("screen_observation")
            session_consents.revoke("external_vision")
            get_screen_state().clear_capture()
            return self._fail("screen_capture_failed")

        watching, _captured_at, frame_available = (
            get_screen_state().perception_snapshot()
        )
        if not watching or not frame_available:
            self.stop()
            return self._fail("first_frame_unavailable")

        with self._lock:
            self._status = "active"
            self._started_at = datetime.now(timezone.utc).isoformat()
        return self.snapshot()

    def stop(self) -> VisionSessionSnapshot:
        self._watcher.stop()
        get_screen_state().clear_capture()
        get_vision_observation_store().clear()
        session_consents.revoke("screen_observation")
        session_consents.revoke("external_vision")
        with self._lock:
            self._status = "stopped"
            self._started_at = ""
            self._last_error = ""
        return self.snapshot()

    def snapshot(self) -> VisionSessionSnapshot:
        watching, _captured_at, frame_available = (
            get_screen_state().perception_snapshot()
        )
        source_id, source_kind, source_label, _session_id, state_error = (
            get_screen_state().source_snapshot()
        )
        with self._lock:
            status = self._status
            started_at = self._started_at
            last_error = self._last_error or state_error
        return VisionSessionSnapshot(
            status=status,
            authorized=session_consents.allows("screen_observation"),
            frame_available=bool(frame_available and watching),
            source_id=source_id,
            source_kind=source_kind,
            source_label=source_label,
            provider=self.config.resolved_vision_provider(),
            external_analysis_authorized=session_consents.allows(
                "external_vision"
            ),
            started_at=started_at,
            last_error=last_error,
        )

    def _fail(self, reason: str) -> VisionSessionSnapshot:
        with self._lock:
            self._status = "degraded"
            self._last_error = reason
        return self.snapshot()


def vision_permission_requested(text: str) -> bool:
    normalized = _normalize(text)
    has_visual_target = bool(
        re.search(r"\b(?:pantalla|ventana|vision|ver\s+lo\s+que)\b", normalized)
    )
    has_permission_act = bool(
        re.search(
            r"\b(?:compart|permiso|autoriz|activa|activar|puedes\s+ver|"
            r"te\s+dejo\s+ver)\w*\b",
            normalized,
        )
    )
    return has_visual_target and has_permission_act


def _normalize(text: str) -> str:
    folded = "".join(
        character
        for character in unicodedata.normalize("NFKD", text.casefold())
        if not unicodedata.combining(character)
    )
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", folded).split())
