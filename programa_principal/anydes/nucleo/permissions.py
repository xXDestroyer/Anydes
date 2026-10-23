from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from anydes.nucleo.interlocutor import InterlocutorContext
from anydes.nucleo.consent import session_consents

PermissionLevel = Literal["safe", "confirm", "blocked"]


@dataclass(frozen=True)
class ToolPolicy:
    level: PermissionLevel
    reason: str


DEFAULT_POLICIES: dict[str, ToolPolicy] = {
    "current_time": ToolPolicy("safe", "Only reads the system clock."),
    "system_info": ToolPolicy("safe", "Only reads basic OS and Python metadata."),
    "word_count": ToolPolicy("safe", "Only processes text you provide."),
    "format_json": ToolPolicy("safe", "Only formats text you provide."),
    "open_url": ToolPolicy("confirm", "Opens a webpage in the local browser."),
    "chrome_search": ToolPolicy("confirm", "Opens a Google search in the local browser."),
    "open_youtube": ToolPolicy("confirm", "Opens YouTube in the local browser."),
    "youtube_search": ToolPolicy("confirm", "Opens a YouTube search in the local browser."),
    "open_spotify": ToolPolicy("confirm", "Opens Spotify locally."),
    "spotify_search": ToolPolicy("confirm", "Opens a Spotify search locally."),
    "spotify_web_search": ToolPolicy("confirm", "Opens Spotify Web in the browser."),
    "open_app": ToolPolicy("confirm", "Opens a known local application."),
    "weather_current": ToolPolicy(
        "safe",
        "Reads weather for the explicitly configured location.",
    ),
    "calendar_create": ToolPolicy(
        "confirm",
        "Creates a calendar event after conversational confirmation.",
    ),
    "memory_profile": ToolPolicy("safe", "Only reads non-sensitive local memory summaries."),
    "memory_conversation_search": ToolPolicy("safe", "Searches non-protected local conversation history."),
    "memory_recent": ToolPolicy("safe", "Shows recent non-protected local conversation turns."),
    "memory_permanent_read": ToolPolicy("safe", "Only reads Anydes permanent local memory."),
    "memory_permanent_write": ToolPolicy("confirm", "Writes a value into permanent local memory."),
    "memory_temporary_read": ToolPolicy("safe", "Only reads the current temporary session memory."),
    "memory_temporary_write": ToolPolicy("safe", "Writes non-sensitive temporary context locally."),
    "memory_temporary_clear": ToolPolicy("safe", "Clears temporary context only."),
    "memory_protected_read": ToolPolicy("confirm", "Reads locally protected sensitive memory."),
    "memory_protected_write": ToolPolicy("confirm", "Writes sensitive memory using local passphrase protection."),
    "memory_read": ToolPolicy("safe", "Compatibility alias for permanent memory read."),
    "memory_write": ToolPolicy("confirm", "Compatibility alias for permanent memory write."),
    "fetch_url": ToolPolicy("blocked", "Network access is disabled in the clean base."),
    "open_browser": ToolPolicy("blocked", "Automatic browser opening is disabled."),
    "run_command": ToolPolicy("blocked", "Shell command execution is not allowed."),
    "screen_capture": ToolPolicy("confirm", "Captures a screenshot of your display."),
    "screen_watch_start": ToolPolicy("confirm", "Starts continuous screen watching in the background."),
    "screen_watch_stop": ToolPolicy("safe", "Stops continuous screen watching."),
    "screen_status": ToolPolicy("safe", "Reports whether screen watching is active."),
    "external_vision_session": ToolPolicy(
        "confirm",
        "Sends the current screen capture to the configured external vision provider for this session.",
    ),
}


SCREEN_VISION_TOOLS = frozenset({"screen_capture", "screen_watch_start"})
OWNER_MEMORY_TOOLS = frozenset(
    name for name in DEFAULT_POLICIES if name.startswith("memory_")
)
SENSITIVE_TOOLS = frozenset({"memory_protected_read", "memory_protected_write"})


class PermissionManager:
    def __init__(
        self,
        policies: dict[str, ToolPolicy] | None = None,
        verifier=None,
        *,
        auto_approve: set[str] | None = None,
    ) -> None:
        self.policies = policies or DEFAULT_POLICIES
        self.verifier = verifier
        self.auto_approve = set(auto_approve or ())
        self.interlocutor = InterlocutorContext.verification_unavailable(
            channel="typed",
            reason="assistant_not_bound",
        )

    def set_interlocutor(self, interlocutor: InterlocutorContext) -> None:
        self.interlocutor = interlocutor

    def interlocutor_allows(self, tool_name: str) -> bool:
        return tool_name not in OWNER_MEMORY_TOOLS or self.interlocutor.allows_personal_memory

    def grant_session(self, tool_names: set[str]) -> None:
        self.auto_approve.update(tool_names)

    def policy_for(self, tool_name: str) -> ToolPolicy:
        return self.policies.get(
            tool_name,
            ToolPolicy("blocked", "Unknown tools are blocked by default."),
        )

    def authorize(self, tool_name: str, description: str, *, interactive: bool) -> bool:
        if not self.interlocutor_allows(tool_name):
            print("Acceso a memoria no disponible para el interlocutor actual.")
            return False

        if tool_name in SCREEN_VISION_TOOLS and session_consents.allows("screen_observation"):
            return True
        if tool_name == "external_vision_session" and session_consents.allows("external_vision"):
            return True

        if tool_name in self.auto_approve and tool_name not in SENSITIVE_TOOLS:
            return True

        policy = self.policy_for(tool_name)

        if policy.level == "safe":
            return True

        if policy.level == "blocked":
            print(f"Bloqueado: {tool_name}. {policy.reason}")
            return False

        if not interactive:
            print(f"Necesita confirmacion: {tool_name}. {policy.reason}")
            return False

        answer = input(
            f"Anydes quiere ejecutar '{tool_name}': {description}\n"
            f"Razon de permiso: {policy.reason}\n"
            "Permitir una vez? [s/N]: "
        ).strip().lower()
        if answer not in {"s", "si", "sí", "y", "yes"}:
            return False
        if tool_name in SENSITIVE_TOOLS:
            if self.verifier is None:
                print("No hay verificacion independiente disponible para esta accion sensible.")
                return False
            print("Esta accion necesita verificacion adicional.")
            return bool(self.verifier.verify_interactive())
        return True

    def authorize_explicit_request(self, tool_name: str, description: str) -> bool:
        """Authorize one direct owner request without granting a lasting permission.

        This is limited to non-sensitive confirmation-level tools. The caller
        must have already matched an explicit command to this exact tool.
        """
        _ = description
        if not self.interlocutor_allows(tool_name):
            return False
        policy = self.policy_for(tool_name)
        if policy.level == "safe":
            return True
        if policy.level != "confirm" or tool_name in SENSITIVE_TOOLS:
            return False
        return (
            self.interlocutor.owner_verified
            and self.interlocutor.state in {"typed_owner", "verified_voice"}
        )
