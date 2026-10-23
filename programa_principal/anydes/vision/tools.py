from __future__ import annotations

import base64

from anydes.nucleo.config import Config
from anydes.nucleo.consent import session_consents
from anydes.nucleo.consent import external_access_decision
from anydes.vision.capture import capture_screen_jpeg
from anydes.vision.state import get_screen_state
from anydes.vision.watcher import ScreenWatcher
from anydes.vision.observation_store import get_vision_observation_store

_watchers: dict[int, ScreenWatcher] = {}


def _watcher_for_config(config: Config) -> ScreenWatcher:
    key = id(config)
    watcher = _watchers.get(key)
    if watcher is None:
        watcher = ScreenWatcher(config)
        _watchers[key] = watcher
    return watcher


def handlers_for_config(config: Config) -> dict[str, object]:
    state = get_screen_state()

    def screen_capture(_: str = "") -> dict[str, object]:
        if not session_consents.allows("screen_observation"):
            return {
                "captured": False,
                "reason": "screen_observation_session_consent_required",
            }
        jpeg = capture_screen_jpeg(
            monitor=config.screen_monitor,
            max_width=config.screen_max_width,
            max_height=config.screen_max_height,
            quality=config.screen_jpeg_quality,
        )
        state.store_capture(jpeg)
        size_kb = max(1, len(jpeg) // 1024)
        return {
            "captured": True,
            "size_kb": size_kb,
            "monitor": config.screen_monitor,
            "max_width": config.screen_max_width,
            "max_height": config.screen_max_height,
        }

    def screen_watch_start(_: str = "") -> dict[str, object]:
        if not session_consents.allows("screen_observation"):
            return {
                "watching": False,
                "reason": "screen_observation_session_consent_required",
            }
        started = _watcher_for_config(config).start()
        if not started:
            return {"watching": False, "reason": "screen_capture_failed"}
        if config.screen_live_mode:
            interval = config.resolved_screen_watch_interval_seconds()
            return {
                "watching": True,
                "live_mode": True,
                "interval_seconds": interval,
            }
        return {
            "watching": True,
            "live_mode": False,
            "interval_seconds": config.resolved_screen_watch_interval_seconds(),
        }

    def screen_watch_stop(_: str = "") -> dict[str, object]:
        _watcher_for_config(config).stop()
        state.clear_capture()
        get_vision_observation_store().clear()
        session_consents.revoke("screen_observation")
        session_consents.revoke("external_vision")
        return {"watching": False, "latest_capture_cleared": True}

    def screen_status(_: str = "") -> dict[str, object]:
        if state.is_watching():
            return {"watching": True, "has_recent_capture": bool(state.latest_jpeg())}
        if state.latest_jpeg():
            return {"watching": False, "has_recent_capture": True}
        return {"watching": False, "has_recent_capture": False}

    def external_vision_session(_: str = "") -> dict[str, object]:
        decision = external_access_decision(config, "external_vision")
        if not decision.allowed:
            return {"granted": False, "reason": decision.reason}
        session_consents.grant("external_vision")
        return {"granted": True, "scope": "current_session"}

    return {
        "screen_capture": screen_capture,
        "screen_watch_start": screen_watch_start,
        "screen_watch_stop": screen_watch_stop,
        "screen_status": screen_status,
        "external_vision_session": external_vision_session,
    }


def ensure_screen_watching(config: Config, *, always: bool = False) -> bool:
    if not session_consents.allows("screen_observation"):
        return False
    if always or config.screen_live_mode:
        _watcher_for_config(config).start()
        return True
    return False


def capture_fresh_screen_base64(config: Config) -> str | None:
    if not session_consents.allows("screen_observation"):
        return None
    state = get_screen_state()
    try:
        jpeg = capture_screen_jpeg(
            monitor=config.screen_monitor,
            max_width=config.screen_max_width,
            max_height=config.screen_max_height,
            quality=config.screen_jpeg_quality,
        )
        state.store_capture(jpeg)
    except RuntimeError:
        return None
    if not jpeg:
        return None
    return base64.b64encode(jpeg).decode("ascii")


def latest_screen_base64(config: Config, *, fresh: bool = False) -> str | None:
    if not session_consents.allows("screen_observation"):
        get_screen_state().clear_capture()
        return None
    if fresh:
        return capture_fresh_screen_base64(config)

    state = get_screen_state()
    jpeg = state.latest_jpeg()
    if not jpeg:
        try:
            jpeg = capture_screen_jpeg(
                monitor=config.screen_monitor,
                max_width=config.screen_max_width,
                max_height=config.screen_max_height,
                quality=config.screen_jpeg_quality,
            )
            state.store_capture(jpeg)
        except RuntimeError:
            return None
    if not jpeg:
        return None
    return base64.b64encode(jpeg).decode("ascii")


def stop_all_screen_watching() -> None:
    for watcher in tuple(_watchers.values()):
        watcher.stop()
    get_screen_state().clear_capture()
    get_vision_observation_store().clear()
    session_consents.revoke("screen_observation")
    session_consents.revoke("external_vision")
