from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from anydes.funciones_extra import apps
from anydes.funciones_extra.registry import ToolRegistry, ToolResult
from anydes.nucleo.config import Config
from anydes.nucleo.consent import session_consents
from anydes.vision.observation_store import get_vision_observation_store
from anydes.vision.state import get_screen_state


@dataclass(frozen=True)
class CapabilityState:
    """Deterministic facts about what Realtime can actually perceive or do."""

    vision_enabled: bool
    frame_available: bool
    frame_age_seconds: float | None
    screen_capture_authorized: bool
    frame_fresh: bool
    capture_active: bool = False
    capture_source_id: str = ""
    capture_source_kind: str = ""
    capture_source_label: str = ""
    capture_error: str = ""
    frame_hash: str = ""
    visual_observation: str = ""
    visual_observation_fresh: bool = False
    tool_available: bool = False
    requested_tool: str = ""
    application_local_executable: bool | None = None
    action_result: ToolResult | None = None

    @property
    def can_claim_screen_perception(self) -> bool:
        return (
            self.vision_enabled
            and self.screen_capture_authorized
            and self.frame_available
            and self.frame_fresh
            and self.visual_observation_fresh
            and bool(self.visual_observation.strip())
        )

    @property
    def action_succeeded(self) -> bool:
        return (
            self.action_result is not None
            and self.action_result.status == "success"
            and bool(self.action_result.data.get("opened", True))
        )

    def with_action_result(self, result: ToolResult) -> "CapabilityState":
        return CapabilityState(
            vision_enabled=self.vision_enabled,
            frame_available=self.frame_available,
            frame_age_seconds=self.frame_age_seconds,
            screen_capture_authorized=self.screen_capture_authorized,
            frame_fresh=self.frame_fresh,
            capture_active=self.capture_active,
            capture_source_id=self.capture_source_id,
            capture_source_kind=self.capture_source_kind,
            capture_source_label=self.capture_source_label,
            capture_error=self.capture_error,
            frame_hash=self.frame_hash,
            visual_observation=self.visual_observation,
            visual_observation_fresh=self.visual_observation_fresh,
            tool_available=self.tool_available,
            requested_tool=self.requested_tool,
            application_local_executable=(
                self.application_local_executable
                if result.status != "success"
                else bool(result.data.get("opened", True))
            ),
            action_result=result,
        )


def capture_capability_state(
    config: Config,
    tools: ToolRegistry,
    *,
    requested_tool: str = "",
    frame_max_age_seconds: float = 12.0,
) -> CapabilityState:
    now = datetime.now(timezone.utc)
    screen = get_screen_state()
    watching, _captured_at, _frame_available = screen.perception_snapshot()
    frame = screen.latest_frame()
    captured_at = frame.captured_at if frame is not None else None
    frame_available = frame is not None
    state_source_id, state_source_kind, state_source_label, _session_id, capture_error = (
        screen.source_snapshot()
    )
    source_id = frame.source_id if frame is not None else state_source_id
    source_kind = frame.source_kind if frame is not None else state_source_kind
    source_label = frame.source_label if frame is not None else state_source_label
    frame_age = (
        max(0.0, (now - captured_at).total_seconds())
        if captured_at is not None
        else None
    )
    latest = (
        get_vision_observation_store().latest_for_frame(
            frame_hash=frame.frame_hash,
            source_id=frame.source_id,
            session_id=frame.session_id,
            max_age_seconds=frame_max_age_seconds,
        )
        if frame is not None
        else None
    )
    if (
        latest is not None
        and captured_at is not None
        and latest.observed_at < captured_at
    ):
        latest = None
    tool_names = {tool.name for tool in tools.list_tools()}
    app_name = ""
    if requested_tool == "open_spotify":
        app_name = "spotify"
    return CapabilityState(
        vision_enabled=bool(config.vision_enabled),
        frame_available=frame_available,
        frame_age_seconds=frame_age,
        screen_capture_authorized=session_consents.allows("screen_observation"),
        frame_fresh=frame_age is not None and frame_age <= frame_max_age_seconds,
        capture_active=watching,
        capture_source_id=source_id,
        capture_source_kind=source_kind,
        capture_source_label=source_label,
        capture_error=capture_error,
        frame_hash=frame.frame_hash if frame is not None else "",
        visual_observation=latest.summary if latest is not None else "",
        visual_observation_fresh=latest is not None,
        tool_available=bool(requested_tool and requested_tool in tool_names),
        requested_tool=requested_tool,
        application_local_executable=(
            apps.app_availability(app_name) if app_name else None
        ),
    )
