from __future__ import annotations

import json

from anydes.cognition.contracts import SelfModel
from anydes.identity import build_identity_core
from anydes.identity.self_state_store import SelfStateStore
from anydes.memoria.manager import MemoryManager
from anydes.nucleo.config import PROGRAM_DIR, PROJECT_DIR, Config
from anydes.nucleo.personality import PersonalityProfile


class SelfProvider:
    """Builds a SelfModel snapshot from current config, personality and avatar state."""

    def __init__(self, config: Config, personality: PersonalityProfile | None = None) -> None:
        self.config = config
        self.personality = personality or PersonalityProfile.load(config.personality_prompt_path)

    def snapshot(self) -> SelfModel:
        avatar_state = self._avatar_state()
        has_3d_model = (PROJECT_DIR / "Anydes 3D Modelo").exists()
        memory = MemoryManager.from_config(self.config)
        identity_core = build_identity_core(
            self.config,
            personality_profile=self.personality,
            memory=memory,
        )
        identity_data = identity_core.to_dict()
        self_store = SelfStateStore.from_config(self.config)
        stored_preferences, self_revision = self_store.snapshot()
        self_preferences = {
            key: preference.to_dict()
            for key, preference in stored_preferences.items()
        }

        return SelfModel(
            name=self.config.profile_name,
            kind=str(identity_data["identity"]["kind"]),
            identity_summary=str(identity_data["identity"]["purpose"]),
            personality_traits=tuple(str(item) for item in identity_data["personality"]["traits"]),
            boundaries=tuple(str(item) for item in identity_data["identity"]["boundaries"]),
            mutable_state={
                "user_name": self.config.user_name,
                "ai_provider": self.config.ai_provider,
                "voice_activation": self.config.voice_activation,
                "vision_enabled": self.config.vision_enabled,
                "external_ai_allowed": self.config.external_ai_allowed,
                "external_network_enabled": self.config.network_enabled,
                "external_vision_allowed": self.config.external_vision_allowed,
                "external_tts_allowed": self.config.external_tts_allowed,
                "external_stt_allowed": self.config.external_stt_allowed,
                "weather_allowed": self.config.weather_allowed,
                "identity_core": identity_data,
                "architectural_question": identity_core.architectural_question(),
                "self_preferences": self_preferences,
                "self_state_revision": self_revision,
            },
            body_state={
                "avatar_state": avatar_state.get("state", "unknown"),
                "avatar_message": avatar_state.get("message", ""),
                "avatar_updated_at": avatar_state.get("updated_at", ""),
                "has_2d_avatar": (PROGRAM_DIR / "avatar").exists(),
                "has_3d_model": has_3d_model,
                "model_3d_root": str(PROJECT_DIR / "Anydes 3D Modelo") if has_3d_model else "",
                "body_identity": identity_data["body_identity"],
            },
        )

    def _avatar_state(self) -> dict[str, str]:
        path = self.config.avatar_state_path
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {str(key): str(value) for key, value in data.items()}
