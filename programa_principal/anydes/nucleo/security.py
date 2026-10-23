from __future__ import annotations

import hmac
from dataclasses import dataclass

from anydes.nucleo.config import Config


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    message: str


class SpeakerVerifier:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.voice_match = None

    def set_voice_match(self, voice_match) -> None:
        self.voice_match = voice_match

    def verify_text(self, spoken_or_typed_text: str) -> VerificationResult:
        mode = self.config.sensitive_action_verification_mode.lower()
        if mode in {"off", "none", "disabled"}:
            return VerificationResult(
                False,
                "La verificacion independiente para acciones sensibles esta desactivada.",
            )
        if mode in {"voice", "voice_only"}:
            return VerificationResult(
                False,
                "La coincidencia acustica no autoriza acciones sensibles.",
            )

        expected = self.config.speaker_verification_phrase.strip().lower()
        actual = spoken_or_typed_text.strip().lower()
        if expected and hmac.compare_digest(expected, actual):
            return VerificationResult(True, "Identidad verificada por frase local.")
        return VerificationResult(False, "No pude verificar la frase de seguridad.")

    def verify_interactive(self) -> bool:
        mode = self.config.sensitive_action_verification_mode.lower()
        if mode in {"off", "none", "disabled"}:
            print("La verificacion independiente para acciones sensibles esta desactivada.")
            return False
        if mode in {"voice", "voice_only"}:
            print("La coincidencia acustica no autoriza acciones sensibles.")
            return False
        phrase = input(
            "Verificacion sensible. Escribe tu frase de seguridad "
            f"(configurada como ANYDES_SPEAKER_PHRASE): "
        )
        result = self.verify_text(phrase)
        if not result.ok:
            print(result.message)
        return result.ok

    def _voice_is_trusted(self) -> bool:
        match = self.voice_match
        return bool(match is not None and match.matched and match.confidence >= 0.70)
