from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional

from anydes.nucleo.config import PROJECT_DIR, Config, config
from anydes.voz.speaker_id import SpeakerRegistry, SpeakerMatch

face_recognition: Optional[ModuleType] = None
np: Optional[ModuleType] = None
mp: Optional[ModuleType] = None


def _optional_import(module_name: str) -> Optional[ModuleType]:
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def _ensure_numpy() -> ModuleType:
    global np
    if np is None:
        np = _optional_import("numpy")
    if np is None:
        raise ImportError("numpy is required for face processing")
    return np


face_recognition = _optional_import("face_recognition")
np = _optional_import("numpy")
mp = _optional_import("mediapipe")


@dataclass
class PersonProfile:
    name: str
    display_name: str
    created_at: str
    voice_profile: Optional[str]
    face_images: List[str]
    preferences: Dict[str, object]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "created_at": self.created_at,
            "voice_profile": self.voice_profile,
            "face_images": self.face_images,
            "preferences": self.preferences,
        }


class ProfileManager:
    """Gestiona perfiles de personas en datos/personas.

    Cada perfil es un JSON en `PROJECT_DIR / 'datos' / 'personas' / <name>.json`.
    También usa `SpeakerRegistry` para enroll/identify de voz.
    """

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or config
        self.base = PROJECT_DIR / "datos" / "personas"
        self.base.mkdir(parents=True, exist_ok=True)
        self._speaker_registry = SpeakerRegistry(self.cfg.speaker_profiles_path, threshold=self.cfg.speaker_match_threshold)

    def _path_for(self, name: str) -> Path:
        safe = name.strip().lower().replace(" ", "_")
        return self.base / f"{safe}.json"

    def list_profiles(self) -> List[str]:
        return [p.stem for p in sorted(self.base.glob("*.json"))]

    def load_profile(self, name: str) -> Optional[dict]:
        p = self._path_for(name)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def create_profile(self, name: str, display_name: Optional[str] = None) -> dict:
        from datetime import datetime

        display = display_name or name
        profile = {
            "name": name,
            "display_name": display,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "voice_profile": None,
            "face_images": [],
            "preferences": {},
        }
        p = self._path_for(name)
        p.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
        return profile

    def save_profile(self, profile: dict) -> None:
        p = self._path_for(profile.get("name", "unknown"))
        p.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

    def enroll_voice(self, profile_name: str, audio_path: Path) -> SpeakerMatch:
        # Enroll in speaker registry under the profile name
        match = self._speaker_registry.enroll(profile_name, audio_path)
        profile = self.load_profile(profile_name) or self.create_profile(profile_name)
        profile["voice_profile"] = profile_name
        self.save_profile(profile)
        return match

    def identify_voice(self, audio_path: Path) -> SpeakerMatch:
        return self._speaker_registry.identify(audio_path)

    def get_preferences(self, profile_name: str) -> dict:
        profile = self.load_profile(profile_name) or {}
        return profile.get("preferences", {})

    def add_face_image(self, profile_name: str, image_path: Path) -> None:
        profile = self.load_profile(profile_name) or self.create_profile(profile_name)
        rel = str(image_path)
        profile.setdefault("face_images", []).append(rel)
        self.save_profile(profile)

    def add_face_embedding(self, profile_name: str, image_array) -> dict:
        """Extract a face embedding from an RGB image array and store it in the profile.

        `image_array` should be a numpy array in RGB order (H,W,3).
        Returns the embedding dict stored.
        """
        # Prefer face_recognition if available (dlib-based embeddings)
        if face_recognition is not None and np is not None:
            encodings = face_recognition.face_encodings(image_array)
            if not encodings:
                raise ValueError("No face found in image")
            emb = encodings[0].tolist()
        else:
            # Fallback: use MediaPipe face mesh landmarks as a simple embedding
            if mp is None:
                raise ImportError("No supported face library installed. Install 'mediapipe' or 'face_recognition'")
            _np = _ensure_numpy()

            emb_vec = self._mediapipe_embedding_from_rgb(image_array)
            emb = _np.asarray(emb_vec).tolist()

        profile = self.load_profile(profile_name) or self.create_profile(profile_name)
        profile.setdefault("face_embeddings", []).append(emb)
        self.save_profile(profile)
        return {"profile": profile_name, "distance": None}

    def identify_face(self, image_array, threshold: float = 0.6) -> tuple[Optional[str], Optional[float]]:
        """Identify a face in `image_array`. Returns (profile_name, distance) or (None, None).

        Uses Euclidean distance via `face_recognition.face_distance` and returns the best match under `threshold`.
        """
        # Build probe embedding via preferred library
        import math

        if face_recognition is not None and np is not None:
            encodings = face_recognition.face_encodings(image_array)
            if not encodings:
                return None, None
            probe = np.asarray(encodings[0])
            use_method = "fr"
        else:
            if mp is None:
                raise ImportError("No supported face library installed. Install 'mediapipe' or 'face_recognition'")
            _np = _ensure_numpy()

            probe = _np.asarray(self._mediapipe_embedding_from_rgb(image_array))
            use_method = "mp"

        best_name: Optional[str] = None
        best_distance: Optional[float] = None
        for name in self.list_profiles():
            prof = self.load_profile(name) or {}
            embeds = prof.get("face_embeddings", [])
            if not embeds:
                continue
            _np = _ensure_numpy()

            known = _np.asarray(embeds)
            # ensure shapes match (fall back to pairwise distances)
            if known.ndim == 1:
                known = known.reshape(1, -1)
            if probe.shape != known.shape[1:]:
                # try flatten/reshape compatibility
                pass
            # compute euclidean distances
            dists = _np.linalg.norm(known - probe, axis=1)
            min_idx = int(_np.argmin(dists))
            dist = float(dists[min_idx])
            if best_distance is None or dist < best_distance:
                best_distance = dist
                best_name = name

        if best_distance is not None and best_distance <= threshold:
            return best_name, best_distance
        return None, best_distance

    def _mediapipe_embedding_from_rgb(self, image_array):
        """Return a flattened landmark vector from a RGB image using MediaPipe FaceMesh.

        The embedding is the concatenation of normalized (x,y,z) landmarks.
        """
        if mp is None:
            raise ImportError("mediapipe is not available")
        _np = _ensure_numpy()

        mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
        try:
            # Convert to RGB numpy array if needed
            img = _np.asarray(image_array)
            results = mp_face.process(img)
            if not results.multi_face_landmarks:
                raise ValueError("No face found in image")
            lm = results.multi_face_landmarks[0].landmark
            vec = []
            for p in lm:
                vec.extend([p.x, p.y, p.z])
            # normalize vector length
            arr = _np.asarray(vec, dtype=_np.float32)
            norm = _np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            return arr.tolist()
        finally:
            mp_face.close()
