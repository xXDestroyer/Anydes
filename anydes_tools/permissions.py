from __future__ import annotations

from dataclasses import dataclass

from .contracts import PermissionEffect, ToolManifest, ToolRequest


@dataclass(frozen=True)
class PermissionDecision:
    allowed: bool
    reason: str


class PermissionBridge:
    """Local bridge enforcing reflection plus effect-level permission policy."""

    _AUTO_EFFECTS = frozenset({
        PermissionEffect.READ,
        PermissionEffect.CREATE,
        PermissionEffect.EDIT,
    })
    _CONFIRM_EFFECTS = frozenset({
        PermissionEffect.OVERWRITE,
        PermissionEffect.MOVE,
        PermissionEffect.DELETE,
        PermissionEffect.EXECUTE_CODE,
        PermissionEffect.INSTALL_DEPENDENCY,
    })

    def authorize(
        self,
        manifest: ToolManifest,
        request: ToolRequest,
        effect: PermissionEffect,
    ) -> PermissionDecision:
        if request.reflection_approved is not True:
            return PermissionDecision(False, "DENIED_REFLECTION_REQUIRED")
        if request.explicit_user_request is not True:
            return PermissionDecision(False, "DENIED_EXPLICIT_REQUEST_REQUIRED")
        configured = str(manifest.permission_policy.get(effect.value, "deny"))
        if configured == "deny" or effect in {PermissionEffect.EXTERNAL_ACTION, PermissionEffect.INSTALL_DEPENDENCY}:
            return PermissionDecision(False, "DENIED_POLICY")
        if effect in self._AUTO_EFFECTS and configured == "allow_in_scope":
            return PermissionDecision(True, "AUTHORIZED_IN_SCOPE")
        if effect in self._CONFIRM_EFFECTS or configured == "confirm":
            if effect not in request.confirmed_effects:
                return PermissionDecision(False, f"DENIED_CONFIRMATION_REQUIRED:{effect.value}")
            return PermissionDecision(True, "AUTHORIZED_CONFIRMED")
        return PermissionDecision(False, "DENIED_UNKNOWN_PERMISSION_STATE")
