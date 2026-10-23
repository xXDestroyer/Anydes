from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from anydes.avatar.vrm import VrmModelInfo, inspect_vrm, preferred_vrm_model
from anydes.nucleo.config import Config


_PROFILE_PATH = Path(__file__).with_name("avatar_self.v1.json")


def _normalize(text: str) -> str:
    folded = "".join(
        character
        for character in unicodedata.normalize("NFKD", text.casefold())
        if not unicodedata.combining(character)
    )
    return " ".join(re.sub(r"[^a-z0-9]+", " ", folded).split())


@dataclass(frozen=True)
class AvatarVisualTrait:
    key: str
    label: str
    value: str
    category: str
    permanence: str
    aliases: tuple[str, ...] = ()
    conflicting_claims: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "AvatarVisualTrait":
        return cls(
            key=str(data.get("key") or ""),
            label=str(data.get("label") or ""),
            value=str(data.get("value") or ""),
            category=str(data.get("category") or ""),
            permanence=str(data.get("permanence") or "canonical"),
            aliases=tuple(str(item) for item in data.get("aliases", [])),
            conflicting_claims=tuple(
                str(item) for item in data.get("conflicting_claims", [])
            ),
        )

    def is_relevant(self, normalized_query: str) -> bool:
        return any(
            re.search(rf"\b{re.escape(_normalize(alias))}\b", normalized_query)
            for alias in self.aliases
            if _normalize(alias)
        )


@dataclass(frozen=True)
class AvatarSelfDescription:
    """Authoritative description of Anydes' represented body, not her identity."""

    avatar_id: str
    representation: str
    apparent_age: str
    style: str
    traits: tuple[AvatarVisualTrait, ...]
    main_colors: tuple[str, ...]
    variable_aspects: tuple[str, ...]
    sources: tuple[str, ...]
    model: VrmModelInfo | None = None

    @classmethod
    def from_config(
        cls,
        config: Config,
        *,
        profile_path: Path | None = None,
    ) -> "AvatarSelfDescription":
        path = profile_path or _PROFILE_PATH
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("avatar_self_profile_must_be_an_object")
        model_path = preferred_vrm_model(config.avatar_model_dir)
        model = inspect_vrm(model_path) if model_path is not None else None
        traits = tuple(
            AvatarVisualTrait.from_dict(item)
            for item in payload.get("traits", [])
            if isinstance(item, dict)
        )
        if not traits or any(not item.key or not item.value for item in traits):
            raise ValueError("avatar_self_profile_has_invalid_traits")
        return cls(
            avatar_id=str(payload.get("avatar_id") or "anydes"),
            representation=str(payload.get("representation") or "avatar digital"),
            apparent_age=str(payload.get("apparent_age") or ""),
            style=str(payload.get("style") or ""),
            traits=traits,
            main_colors=tuple(str(item) for item in payload.get("main_colors", [])),
            variable_aspects=tuple(
                str(item) for item in payload.get("variable_aspects", [])
            ),
            sources=tuple(str(item) for item in payload.get("sources", [])),
            model=model,
        )

    def is_relevant_query(self, text: str) -> bool:
        normalized = _normalize(text)
        return bool(
            re.search(
                r"\b(?:avatar|apariencia|aspecto|como te ves|como eres fisicamente|"
                r"cuerpo|ropa|atuendo|cabello|pelo|ojos|gafas|lentes|hoodie|"
                r"falda|medias|rostro|cara)\b",
                normalized,
            )
        )

    def relevant_traits(self, text: str) -> tuple[AvatarVisualTrait, ...]:
        normalized = _normalize(text)
        selected = tuple(
            trait for trait in self.traits if trait.is_relevant(normalized)
        )
        if selected:
            return selected
        if self.is_relevant_query(text):
            preferred = {"hair", "eyes", "glasses", "hoodie", "skirt", "stockings"}
            return tuple(trait for trait in self.traits if trait.key in preferred)
        return ()

    def prompt_evidence(self, text: str) -> str:
        traits = self.relevant_traits(text)
        if not traits:
            return ""
        model_note = (
            f"VRM activo y verificado: {self.model.path.name}."
            if self.model is not None and self.model.valid
            else "La referencia visual canonica existe, pero no hay un VRM activo verificado."
        )
        lines = [
            "SELF VISUAL AUTORITATIVO:",
            f"- representacion: {self.representation}",
            f"- edad representada: {self.apparent_age}",
            f"- estilo: {self.style}",
            f"- recurso: {model_note}",
        ]
        lines.extend(
            f"- {trait.label}: {trait.value} [{trait.permanence}]"
            for trait in traits
        )
        lines.append(
            "- Describe solo estos rasgos. Distingue el avatar representado de "
            "experiencias fisicas reales y no improvises detalles ausentes."
        )
        return "\n".join(lines)

    def validation_issues(self, response: str) -> tuple[str, ...]:
        normalized = _normalize(response)
        if not normalized:
            return ()
        for trait in self.traits:
            if not trait.is_relevant(normalized):
                continue
            if any(
                _normalize(claim) in normalized
                for claim in trait.conflicting_claims
                if _normalize(claim)
            ):
                return ("unsupported_avatar_self_claim",)
        return ()

    def safe_summary(self, query: str) -> str:
        traits = self.relevant_traits(query)
        values = [trait.value for trait in traits[:4]]
        if not values:
            return (
                "No tengo ese detalle definido en mi representación visual todavía."
            )
        if len(values) == 1:
            return f"En mi avatar actual tengo {values[0]}."
        return "En mi avatar actual tengo " + ", ".join(values[:-1]) + f" y {values[-1]}."
