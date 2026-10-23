"""Free-speech Parakeet laboratory; no cognition or production integration."""
from __future__ import annotations

import time

from .common import OUT, language_drift
from anydes.voice_input.environment import setup
setup()

from PySide6.QtWidgets import QApplication, QLabel
from .backend import ParakeetTdtQ8, diagnostics
from anydes.voice_input.core import VoiceInput
from anydes.voice_input.ui import VoicePanel


class ParakeetLiveRecognizer:
    name = "parakeet"

    def __init__(self, _name="parakeet"):
        started = time.perf_counter()
        self.model = ParakeetTdtQ8()
        self.load_ms = (time.perf_counter() - started) * 1000

    def transcribe(self, audio):
        started = time.perf_counter()
        text, details = self.model.transcribe(audio, 16000)
        elapsed = (time.perf_counter() - started) * 1000
        seconds = len(audio) / 16000
        diagnostic = diagnostics(text, seconds)
        return {
            "text": text,
            "confidence": None,
            "stt_ms": elapsed,
            "rtf": elapsed / max(1, seconds * 1000),
            "audio_seconds": seconds,
            "backend": "NeMo-Speech.cpp CPU Q8",
            "model": "Parakeet TDT 0.6B v3 Q8",
            "config": "parakeet_tdt_q8_cpu_clean_es_request_v1",
            "language_requested": "es",
            "language_backend": "no_metadata",
            "language_probability": None,
            "context": "none; TDT word boosting unsupported",
            "condition_on_previous_text": False,
            "temperature": None,
            "beam_size": None,
            "initial_prompt": None,
            "hotwords": None,
            "whisper_segment_count": details.get("segments", 1),
            "whisper_segments": [],
            "no_speech_probability_max": None,
            "no_speech_probability_mean": None,
            "hallucination": {
                "suspected": diagnostic["hallucination_suspected"],
                "reasons": diagnostic["hallucination_reasons"],
                "pathological_repetition": diagnostic["pathological_repetition"],
            },
            **language_drift(text),
        }

    def close(self):
        self.model.close()


class ParakeetVoiceInput(VoiceInput):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_label = "Parakeet TDT 0.6B v3 Q8"
        self.config_label = "parakeet_tdt_q8_cpu_clean_es_request_v1"
        self.language_label = "es"
        self.context_label = "none; TDT word boosting unsupported"

    def close(self):
        clean = super().close()
        if self.recognizer is not None and hasattr(self.recognizer, "close"):
            self.recognizer.close()
            self.recognizer = None
        return clean


class ParakeetLivePanel(VoicePanel):
    def __init__(self, parent=None, on_input=None, on_barge=None):
        super().__init__(parent, on_input, on_barge, baseline_lab=False)
        self.setWindowTitle("Anydes | Parakeet STT - laboratorio libre")
        self.engine = ParakeetVoiceInput(self.bridge.received.emit,
            recognizer_factory=ParakeetLiveRecognizer)
        self.backend.clear()
        self.backend.addItem("Parakeet TDT 0.6B v3 Q8 / CPU", "parakeet")
        self.backend.setEnabled(False)
        self.decoder_info.setText("MODEL: Parakeet TDT 0.6B v3 Q8 | CPU | CONTEXT: none | "
            "language_code enviado: es | language conditioning: no soportado por TDT")
        notice = QLabel("Prueba libre. PTT es la ruta principal; VAD requiere auriculares. "
            "LANGUAGE_DRIFT y HALLUCINATION_SUSPECTED se muestran sin corregir el texto.")
        notice.setWordWrap(True)
        notice.setStyleSheet("color:#f2d2df;")
        self.layout().insertWidget(2, notice)
        self.diagnostic.setChecked(True)
        self.log_path = OUT / "free-lab-events.jsonl"
        self.resource_path.setText("Registros locales: " + str(OUT))

    def on_event(self, event):
        super().on_event(event)
        if event.get("kind") in {"transcript", "hallucination_suspected"}:
            if event.get("language_drift"):
                markers = ", ".join(event.get("english_markers", []))
                self.error.setText("LANGUAGE_DRIFT | marcadores: " + markers)
            if event.get("hallucination", {}).get("suspected"):
                self.error.setText("HALLUCINATION_SUSPECTED | resultado no enviado a ningun pipeline")
            self.confidence.setText("No disponible/calibrada para este head TDT")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    app = QApplication([])
    panel = ParakeetLivePanel()
    panel.show()
    app.aboutToQuit.connect(panel.shutdown)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
