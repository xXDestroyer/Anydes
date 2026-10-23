from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from anydes.avatar.vrm import preferred_vrm_model
from anydes.nucleo.config import PROJECT_DIR


@dataclass(frozen=True)
class AvatarRenderProfile:
    """Perfil visual para presentar a Anydes con lectura premium.

    Estos valores corrigen el problema del render gris/tecnico actual dentro de
    la experiencia: encuadre grande, fondo oscuro calido, sombra suave y escala
    de personaje dominante. El renderer VRM futuro debe respetar este perfil.
    """

    fov_degrees: float = 28.0
    camera_height_ratio: float = 0.55
    subject_scale: float = 1.42
    subject_vertical_bias: float = 0.18
    key_light_intensity: float = 1.35
    fill_light_intensity: float = 0.42
    rim_light_intensity: float = 1.05
    shadow_softness: float = 0.72
    material_roughness: float = 0.48
    material_specular: float = 0.32


PROFILE = AvatarRenderProfile()


def avatar_asset_candidates() -> tuple[Path, ...]:
    """Orden de assets usados por el escenario PySide6.

    Prioriza renders verificados de Blender y deja VRM/GLB como fuentes para el
    renderer real posterior. La UI no depende de Tkinter ni del visor HTML.
    """

    root = PROJECT_DIR / "Anydes 3D Modelo"
    return (
        root / "renders" / "anydes_hands_in_pockets_front.png",
        root / "renders" / "anydes_natural_idle_pose_front.png",
        root / "renders" / "anydes_avatar_v004_front.png",
        root / "final_review" / "renders" / "anydes_v001_review_front.png",
        root / "Anydes_reference_definitive.png",
        PROJECT_DIR / "programa_principal" / "avatar" / "assets" / "anydes_happy.png",
        PROJECT_DIR / "programa_principal" / "avatar" / "assets" / "anydes_neutral.png",
    )


def preferred_avatar_model(model_root: Path | None = None) -> Path:
    root = model_root or PROJECT_DIR / "Anydes 3D Modelo"
    return preferred_vrm_model(root) or (
        root / "final_review" / "exports" / "Anydes_v001_review.vrm"
    )
