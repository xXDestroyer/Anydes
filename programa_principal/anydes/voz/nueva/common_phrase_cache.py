from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

from anydes.voz.nueva.contracts import AudioBuffer


class CommonPhraseAudioCache:
    """Read-only cache for explicitly approved, non-private reusable phrases."""

    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = Path(manifest_path)
        self._entries: dict[str, dict[str, object]] = {}
        self.identity_id = ""
        self.voice_version = ""
        if not self.manifest_path.is_file():
            return
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.identity_id = str(payload.get("identity_id") or "")
        self.voice_version = str(payload.get("voice_version") or "")
        for entry in payload.get("entries", []):
            if not isinstance(entry, dict) or entry.get("scope") != "public_reusable":
                continue
            text = " ".join(str(entry.get("spoken_text") or "").split()).strip()
            if text:
                self._entries[text] = entry

    def lookup(self, *, identity_id: str, spoken_text: str) -> AudioBuffer | None:
        if identity_id != self.identity_id or self.voice_version != "Anydes Canonical Voice v1":
            return None
        text = " ".join(spoken_text.split()).strip()
        entry = self._entries.get(text)
        if entry is None:
            return None
        audio_path = (self.manifest_path.parent / str(entry.get("audio") or "")).resolve()
        if not audio_path.is_file():
            return None
        expected = str(entry.get("sha256") or "").upper()
        actual = hashlib.sha256(audio_path.read_bytes()).hexdigest().upper()
        if not expected or actual != expected:
            return None
        with wave.open(str(audio_path), "rb") as source:
            channels = source.getnchannels()
            sample_rate = source.getframerate()
            frame_count = source.getnframes()
            payload = source.readframes(frame_count)
            if source.getsampwidth() != 2:
                return None
        return AudioBuffer(payload, sample_rate, channels, frame_count)

