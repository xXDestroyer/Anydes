from __future__ import annotations

from pathlib import Path
import threading


class AudioRecorder:
    def __init__(
        self,
        sample_rate: int = 16000,
        seconds: int = 5,
        *,
        min_seconds: float = 0.8,
        silence_seconds: float = 0.9,
        speech_threshold: float = 0.012,
    ) -> None:
        self.sample_rate = sample_rate
        self.seconds = seconds
        self.min_seconds = min_seconds
        self.silence_seconds = silence_seconds
        self.speech_threshold = speech_threshold
        self._cancel_event = threading.Event()

    def reset(self) -> None:
        self._cancel_event.clear()

    def cancel(self) -> None:
        self._cancel_event.set()

    def record_wav(self, output_path: Path) -> Path:
        try:
            import numpy as np
            import sounddevice as sd
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError(
                "Faltan dependencias de voz. Instala `sounddevice` y `soundfile`."
            ) from exc

        output_path.parent.mkdir(parents=True, exist_ok=True)
        chunk_size = int(self.sample_rate * 0.1)
        max_chunks = max(1, int(self.seconds / 0.1))
        min_chunks = max(1, int(self.min_seconds / 0.1))
        silence_limit = max(1, int(self.silence_seconds / 0.1))

        collected: list = []
        speech_started = False
        trailing_silence = 0

        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="float32") as stream:
            for _ in range(max_chunks):
                if self._cancel_event.is_set():
                    break
                chunk, _overflowed = stream.read(chunk_size)
                flat = chunk[:, 0]
                collected.append(flat.copy())
                energy = float(abs(flat).mean())

                if energy >= self.speech_threshold:
                    speech_started = True
                    trailing_silence = 0
                elif speech_started:
                    trailing_silence += 1
                    if len(collected) >= min_chunks and trailing_silence >= silence_limit:
                        break

        if self._cancel_event.is_set():
            raise RuntimeError("La escucha fue detenida por el usuario.")
        audio = np.concatenate(collected) if collected else np.zeros(chunk_size, dtype="float32")
        sf.write(output_path, audio, self.sample_rate)
        return output_path
