"""Guided one-take recorder for the frozen 30-sentence human corpus."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys

from common import CORPUS, OUT, atomic_json, latest_session, load_corpus, sha256

VOICE = Path(__file__).resolve().parents[1] / "voice_input_v001"
sys.path.insert(0, str(VOICE))
from environment import setup
setup()

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel, QMessageBox, QProgressBar
from core import VoiceInput
from ui import VoicePanel


class CaptureOnlyRecognizer:
    def __init__(self, name="capture_only"):
        self.name = name
        self.load_ms = 0.0

    def transcribe(self, audio):
        seconds = len(audio) / 16000
        return {
            "text": "capture_saved",
            "confidence": None,
            "stt_ms": 0.0,
            "rtf": 0.0,
            "audio_seconds": seconds,
            "backend": "capture_only",
            "model": "none",
            "config": "frozen_corpus_recording_v1",
            "language_requested": "es",
            "language_backend": "not_decoded",
            "language_probability": None,
            "context": "none",
            "condition_on_previous_text": False,
            "temperature": None,
            "beam_size": None,
            "initial_prompt": None,
            "hotwords": None,
            "whisper_segment_count": 0,
            "whisper_segments": [],
            "no_speech_probability_max": None,
            "no_speech_probability_mean": None,
            "hallucination": {"suspected": False, "reasons": []},
        }


def new_session() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    directory = OUT / f"generalization-recording-{stamp}"
    directory.mkdir(parents=True, exist_ok=False)
    atomic_json(directory / "recording-state.json", {
        "schema": "anydes.stt_generalization_recording.v1",
        "created": datetime.now().astimezone().isoformat(),
        "corpus_path": str(CORPUS.resolve()),
        "corpus_sha256": sha256(CORPUS),
        "protocol": {
            "takes_per_phrase": 1,
            "speech_style": "natural",
            "sample_rate": 16000,
            "channels": 1,
            "physical_pre_roll_ms": 400,
            "stt_input": "identity_from_saved_segment",
        },
        "complete": False,
        "recordings": [],
    })
    return directory


def choose_session() -> Path:
    candidate = latest_session(complete_only=False)
    if candidate:
        state = json.loads((candidate / "recording-state.json").read_text(encoding="utf-8"))
        if not state.get("complete") and state.get("corpus_sha256") == sha256(CORPUS):
            return candidate
    return new_session()


class GeneralizationRecorder(VoicePanel):
    def __init__(self, session: Path):
        super().__init__(baseline_lab=True)
        self.setWindowTitle("Anydes | Corpus humano STT - 30 frases")
        self.setMinimumSize(700, 820)
        self.session = session
        self.rows = load_corpus()
        self.state_path = session / "recording-state.json"
        self.recording_state = json.loads(self.state_path.read_text(encoding="utf-8"))
        if self.recording_state["corpus_sha256"] != sha256(CORPUS):
            raise RuntimeError("El corpus cambio despues de crear esta sesion.")
        self.engine = VoiceInput(self.bridge.received.emit,
            recognizer_factory=CaptureOnlyRecognizer)
        self.backend.clear()
        self.backend.addItem("Solo captura; sin decodificar durante la grabacion", "capture_only")
        self.decoder_info.setText("CAPTURE ONLY | 30 frases congeladas | 1 toma por frase | sin feedback STT")
        self.reference.setReadOnly(True)
        self.vad.hide()
        self.headphones.hide()
        self.transcript.hide()
        self.metrics.setText("La transcripcion se ejecutara despues sobre el mismo audio.")
        self.log_path = session / "voice-events.jsonl"
        self.resource_path.setText("Sesion congelada: " + str(session))
        self.pending_gid = None
        self.completed = {item["id"] for item in self.recording_state["recordings"]}
        self.current_index = next((index for index, row in enumerate(self.rows)
            if row["id"] not in self.completed), len(self.rows))

        self.corpus_progress = QProgressBar()
        self.corpus_progress.setRange(0, len(self.rows))
        self.corpus_progress.setValue(len(self.completed))
        self.phrase_title = QLabel()
        self.phrase_title.setStyleSheet("font-size:16px;font-weight:600;color:#f2d2df;")
        self.phrase_text = QLabel()
        self.phrase_text.setWordWrap(True)
        self.phrase_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.phrase_text.setStyleSheet("font-size:20px;color:#ffffff;padding:10px;")
        layout = self.layout()
        layout.insertWidget(2, self.corpus_progress)
        layout.insertWidget(3, self.phrase_title)
        layout.insertWidget(4, self.phrase_text)
        self.prepare_ptt.setText("Preparar microfono para las 30 frases")
        self.ptt.setText("Mantener mientras lees esta frase")
        self._show_current()

    def _show_current(self):
        if self.current_index >= len(self.rows):
            self.phrase_title.setText("Corpus completo: 30/30")
            self.phrase_text.setText("La sesion quedo congelada y lista para replay A/B.")
            self.reference.clear()
            self.ptt.setEnabled(False)
            self.prepare_ptt.setEnabled(False)
            return
        row = self.rows[self.current_index]
        self.phrase_title.setText(
            f"Frase {self.current_index + 1}/30 | {row['id']} | {row['category']}")
        self.phrase_text.setText(row["text"])
        self.reference.setText(row["text"])

    def configure_diagnostics(self):
        self.engine.configure_diagnostics(True, self.reference.text(), self.session)

    def begin_ptt(self):
        if self.current_index >= len(self.rows) or self.pending_gid:
            return
        self.configure_diagnostics()
        self.pending_gid = self.rows[self.current_index]["id"]
        super().begin_ptt()
        if not self.ptt_down:
            self.pending_gid = None

    def on_event(self, event):
        kind = event.get("kind")
        if kind in {"transcript", "hallucination_suspected"}:
            self.log(event)
            self.busy = False
            if self.engine.capture_thread and self.engine.capture_thread.is_alive():
                self.set_state("ptt_armed", "PTT preparado")
            return
        super().on_event(event)
        if kind == "diagnostic_saved":
            self._commit_recording(Path(event["diagnostic"]["folder"]), event["audio_id"])

    def _commit_recording(self, folder: Path, audio_id: str):
        if not self.pending_gid:
            self.error.setText("Se guardo audio sin una frase activa; sesion detenida.")
            self.stop()
            return
        row = self.rows[self.current_index]
        if row["id"] != self.pending_gid or row["id"] in self.completed:
            self.error.setText("Intento de toma duplicada bloqueado.")
            self.stop()
            return
        metadata_path = folder / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        voice_duration = metadata.get("endpointing", {}).get("metrics", {}).get(
            "voice_detected_duration_s", 0)
        if metadata["stt_segment"]["duration_s"] < 0.25 or voice_duration < 0.12:
            self.pending_gid = None
            self.error.setText("Toma sin voz suficiente; no fue contada. Repite la misma frase.")
            return
        audio_path = folder / "stt_segment_float32.npy"
        recording = {
            "index": self.current_index,
            "id": row["id"],
            "category": row["category"],
            "reference": row["text"],
            "audio_id": audio_id,
            "audio_path": str(audio_path.resolve()),
            "audio_sha256": sha256(audio_path),
            "metadata_path": str(metadata_path.resolve()),
            "sample_rate": metadata["capture"]["sample_rate"],
            "duration_s": metadata["stt_segment"]["duration_s"],
            "device_name": metadata["capture"]["device_name"],
        }
        self.recording_state["recordings"].append(recording)
        self.completed.add(row["id"])
        self.recording_state["complete"] = len(self.completed) == len(self.rows)
        atomic_json(self.state_path, self.recording_state)
        self.corpus_progress.setValue(len(self.completed))
        self.pending_gid = None
        self.current_index += 1
        self._show_current()
        if self.recording_state["complete"]:
            atomic_json(self.session / "immutable-audio-manifest.json", self.recording_state)
            self.stop()
            QMessageBox.information(self, "Corpus completo",
                "Las 30 tomas quedaron congeladas. Ya puedes ejecutar el replay Parakeet/Moonshine.")


def main() -> int:
    app = QApplication([])
    panel = GeneralizationRecorder(choose_session())
    panel.show()
    app.aboutToQuit.connect(panel.shutdown)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
