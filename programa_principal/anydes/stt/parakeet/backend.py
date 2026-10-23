"""Benchmark one local STT candidate in a fresh process over immutable human arrays."""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import unicodedata

PROGRAM = Path(__file__).resolve().parents[3]
VOICE = PROGRAM / "anydes" / "voice_input"
HERE = Path(__file__).resolve().parent
VOICE_VENDOR = PROGRAM / "runtime" / "python_voice_deps"
sys.path.append(str(VOICE_VENDOR))

import numpy as np
import psutil


def normalize_words(text: str) -> list[str]:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.findall(r"\w+|\+\+", text)


def normalize_chars(text: str) -> str:
    return "".join(normalize_words(text))


def distance(left, right) -> int:
    row = list(range(len(right) + 1))
    for index, first in enumerate(left, 1):
        next_row = [index]
        for column, second in enumerate(right, 1):
            next_row.append(min(next_row[-1] + 1, row[column] + 1,
                row[column - 1] + (first != second)))
        row = next_row
    return row[-1]


def errors(reference: str, hypothesis: str) -> dict:
    rw, hw = normalize_words(reference), normalize_words(hypothesis)
    rc, hc = normalize_chars(reference), normalize_chars(hypothesis)
    return {
        "word_errors": distance(rw, hw),
        "reference_words": len(rw),
        "wer": distance(rw, hw) / max(1, len(rw)),
        "character_errors": distance(rc, hc),
        "reference_characters": len(rc),
        "cer": distance(rc, hc) / max(1, len(rc)),
    }


def understandable(hypothesis: str) -> bool:
    """Conversational utility rubric for this frozen three-word regression phrase."""
    tokens = normalize_words(hypothesis)
    action = any(distance(token, "busca") <= 2 or distance(token, "buscar") <= 2 or "buzca" in token for token in tokens)
    target = any(distance(token, "manzana") <= 2 for token in tokens)
    return action and target


def diagnostics(text: str, audio_seconds: float) -> dict:
    tokens = normalize_words(text)
    reasons = []
    if len(tokens) > max(20, int(audio_seconds * 8)):
        reasons.append("disproportionate_word_count")
    repeated_phrase = None
    repeated_count = 0
    for width in range(1, min(6, len(tokens) // 4) + 1):
        grams = [" ".join(tokens[index:index + width]) for index in range(len(tokens) - width + 1)]
        for candidate in set(grams):
            count = grams.count(candidate)
            coverage = count * width / max(1, len(tokens))
            if count >= 4 and coverage >= 0.5 and count > repeated_count:
                repeated_phrase, repeated_count = candidate, count
    if repeated_phrase:
        reasons.append("pathological_repetition")
    english = {"i", "you", "the", "this", "that", "need", "with", "come", "and", "it", "is", "are"}
    english_markers = sum(token in english for token in tokens)
    return {
        "hallucination_suspected": bool(reasons),
        "hallucination_reasons": reasons,
        "pathological_repetition": repeated_phrase is not None,
        "repeated_phrase": repeated_phrase,
        "repeated_phrase_count": repeated_count,
        "word_count": len(tokens),
        "words_per_second": len(tokens) / max(audio_seconds, 0.001),
        "incorrect_language_suspected": english_markers >= 2,
        "english_marker_count": english_markers,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset(session: Path, manifest: Path) -> list[dict]:
    approved = json.loads(manifest.read_text(encoding="utf-8"))["audio_sha256_by_utterance"]
    clips = []
    for metadata_path in sorted(session.rglob("metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        reference = metadata.get("reference_text", "").strip()
        if not reference:
            continue
        path = metadata_path.parent / metadata["files"]["segment_float32"]
        actual = sha256(path)
        expected = approved.get(metadata["utterance_id"])
        if actual != expected:
            raise RuntimeError(f"Audio inmutable no coincide: {metadata['utterance_id']}: {actual} != {expected}")
        pcm = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
        if not pcm.flags.c_contiguous:
            pcm = np.ascontiguousarray(pcm)
        clips.append({
            "utterance_id": metadata["utterance_id"],
            "reference": reference,
            "path": str(path),
            "sha256": actual,
            "pcm": pcm,
            "sample_rate": int(metadata["capture"]["sample_rate"]),
            "duration_s": len(pcm) / int(metadata["capture"]["sample_rate"]),
        })
    if set(approved) != {clip["utterance_id"] for clip in clips}:
        raise RuntimeError("El conjunto de utterances no coincide con el manifiesto aprobado.")
    return clips


def gpu_process_memory_mib(pid: int) -> float | None:
    try:
        result = subprocess.run([
            "nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"
        ], capture_output=True, text=True, timeout=5, check=False)
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) == pid:
                return float(parts[1])
        return 0.0
    except Exception:
        return None


class ResourceMonitor:
    def __init__(self) -> None:
        self.process = psutil.Process()
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="stt-resource-monitor")

    def _run(self) -> None:
        self.process.cpu_percent(None)
        while not self.stop.wait(0.05):
            self.samples.append({
                "at": time.monotonic(),
                "rss_bytes": self.process.memory_info().rss,
                "cpu_percent": self.process.cpu_percent(None),
                "system_available_bytes": psutil.virtual_memory().available,
                "system_commit_percent": psutil.virtual_memory().percent,
            })

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()


class WhisperSmall:
    name = "whisper_small_int8"
    backend = "CPU / CTranslate2 INT8"

    def __init__(self):
        sys.path.append(str(VOICE_VENDOR))
        from faster_whisper import WhisperModel
        self.model_path = VOICE / "models" / "small"
        self.model = WhisperModel(str(self.model_path), device="cpu", compute_type="int8",
            cpu_threads=2, num_workers=1, local_files_only=True)

    def transcribe(self, pcm: np.ndarray, sample_rate: int) -> tuple[str, dict]:
        if sample_rate != 16000:
            raise ValueError("Whisper esperaba 16 kHz; no se permite resampling en este replay.")
        segments, info = self.model.transcribe(pcm, language="es", temperature=0,
            condition_on_previous_text=False, vad_filter=False, beam_size=3,
            initial_prompt=None, hotwords=None, word_timestamps=False)
        rows = list(segments)
        text = " ".join(row.text.strip() for row in rows if row.no_speech_prob < 0.6).strip()
        return text, {
            "language_backend": info.language,
            "no_speech_probability_max": max((float(row.no_speech_prob) for row in rows), default=None),
            "segments": len(rows),
        }

    def model_files(self) -> list[Path]:
        return [self.model_path / "model.bin"]


class MoonshineSpanishBase:
    name = "moonshine_spanish_base"
    backend = "CPU / ONNX Runtime quantized"

    def __init__(self):
        sys.path.insert(0, str(HERE / "moonshine_vendor"))
        from moonshine_voice import ModelArch, Transcriber
        self.model_path = HERE / "models" / "moonshine" / "download.moonshine.ai" / "model" / "base-es" / "quantized" / "base-es"
        self.transcriber = Transcriber(self.model_path, ModelArch.BASE)

    def transcribe(self, pcm: np.ndarray, sample_rate: int) -> tuple[str, dict]:
        transcript = self.transcriber.transcribe_without_streaming(pcm.tolist(), sample_rate=sample_rate)
        text = " ".join(line.text.strip() for line in transcript.lines if line.text.strip()).strip()
        return text, {"language_backend": "es", "segments": len(transcript.lines), "no_speech_probability_max": None}

    def model_files(self) -> list[Path]:
        return sorted(path for path in self.model_path.iterdir() if path.is_file())


class BackendConfig(ctypes.Structure):
    _fields_ = [("size", ctypes.c_size_t), ("gpu", ctypes.c_int32)]


class ModelConfig(ctypes.Structure):
    _fields_ = [("size", ctypes.c_size_t), ("path", ctypes.c_char_p), ("name", ctypes.c_char_p)]


class RecognizerConfig(ctypes.Structure):
    _fields_ = [("size", ctypes.c_size_t), ("backend", ctypes.POINTER(BackendConfig)),
        ("model", ctypes.POINTER(ModelConfig)), ("streaming", ctypes.c_void_p),
        ("decoder", ctypes.c_void_p), ("vad", ctypes.c_void_p),
        ("endpointing", ctypes.c_void_p), ("postproc", ctypes.c_void_p),
        ("diar", ctypes.c_void_p), ("batching", ctypes.c_void_p)]


class RecognitionOptions(ctypes.Structure):
    _fields_ = [("size", ctypes.c_size_t), ("request_id", ctypes.c_char_p),
        ("language_code", ctypes.c_char_p), ("interim_results", ctypes.c_bool),
        ("enable_word_time_offsets", ctypes.c_bool), ("enable_automatic_punctuation", ctypes.c_bool),
        ("verbatim_transcripts", ctypes.c_bool), ("profanity_filter", ctypes.c_bool),
        ("stop_history_eou_ms", ctypes.c_int32), ("speech_contexts", ctypes.c_void_p),
        ("speech_context_count", ctypes.c_size_t), ("max_alternatives", ctypes.c_int32),
        ("enable_speaker_diarization", ctypes.c_bool), ("max_speaker_count", ctypes.c_int32)]


class ParakeetTdtQ8:
    name = "parakeet_tdt_0.6b_v3_q8_cpu"
    backend = "CPU / NeMo-Speech.cpp GGUF Q8"

    def __init__(self):
        self.bin_path = PROGRAM / "runtime" / "stt" / "nemo-speech" / "bin"
        self.model_path = PROGRAM / "runtime" / "stt" / "models" / "parakeet" / "parakeet-tdt-0.6b-v3.q8_0.gguf"
        os.add_dll_directory(str(self.bin_path))
        self.lib = ctypes.CDLL(str(self.bin_path / "nemo_speech_asr_c.dll"))
        self.lib.nemo_speech_asr_create.argtypes = [ctypes.POINTER(RecognizerConfig), ctypes.POINTER(ctypes.c_void_p)]
        self.lib.nemo_speech_asr_create.restype = ctypes.c_int
        self.lib.nemo_speech_asr_destroy.argtypes = [ctypes.c_void_p]
        self.lib.nemo_speech_asr_recognition_options_default.restype = RecognitionOptions
        self.lib.nemo_speech_asr_recognize_f32.argtypes = [ctypes.c_void_p, ctypes.POINTER(RecognitionOptions),
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.nemo_speech_asr_recognize_f32.restype = ctypes.c_int
        self.lib.nemo_speech_asr_result_transcript.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.lib.nemo_speech_asr_result_transcript.restype = ctypes.c_char_p
        self.lib.nemo_speech_asr_result_alternative_count.argtypes = [ctypes.c_void_p]
        self.lib.nemo_speech_asr_result_alternative_count.restype = ctypes.c_size_t
        self.lib.nemo_speech_asr_result_destroy.argtypes = [ctypes.c_void_p]
        self.lib.nemo_speech_asr_last_error.restype = ctypes.c_char_p
        self.backend_config = BackendConfig(ctypes.sizeof(BackendConfig), -1)
        self.model_bytes = str(self.model_path).encode("utf-8")
        self.model_config = ModelConfig(ctypes.sizeof(ModelConfig), self.model_bytes, None)
        config = RecognizerConfig(ctypes.sizeof(RecognizerConfig), ctypes.pointer(self.backend_config),
            ctypes.pointer(self.model_config), None, None, None, None, None, None, None)
        self.handle = ctypes.c_void_p()
        status = self.lib.nemo_speech_asr_create(ctypes.byref(config), ctypes.byref(self.handle))
        if status != 0:
            message = self.lib.nemo_speech_asr_last_error()
            raise RuntimeError(f"Parakeet create fallo {status}: {message.decode('utf-8', 'replace') if message else ''}")

    def transcribe(self, pcm: np.ndarray, sample_rate: int) -> tuple[str, dict]:
        options = self.lib.nemo_speech_asr_recognition_options_default()
        options.language_code = b"es"
        options.request_id = None
        options.enable_automatic_punctuation = True
        result = ctypes.c_void_p()
        pointer = pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        status = self.lib.nemo_speech_asr_recognize_f32(self.handle, ctypes.byref(options),
            pointer, len(pcm), sample_rate, ctypes.byref(result))
        if status != 0:
            message = self.lib.nemo_speech_asr_last_error()
            raise RuntimeError(f"Parakeet recognize fallo {status}: {message.decode('utf-8', 'replace') if message else ''}")
        try:
            count = int(self.lib.nemo_speech_asr_result_alternative_count(result))
            raw = self.lib.nemo_speech_asr_result_transcript(result, 0) if count else None
            text = raw.decode("utf-8", "replace").strip() if raw else ""
            return text, {"language_backend": "es", "segments": 1, "alternatives": count,
                "no_speech_probability_max": None}
        finally:
            self.lib.nemo_speech_asr_result_destroy(result)

    def model_files(self) -> list[Path]:
        return [self.model_path]

    def close(self) -> None:
        if getattr(self, "handle", None):
            self.lib.nemo_speech_asr_destroy(self.handle)
            self.handle = None


CANDIDATES = {
    "whisper_small": WhisperSmall,
    "moonshine_es_base": MoonshineSpanishBase,
    "parakeet_tdt_q8_cpu": ParakeetTdtQ8,
}


def percentile(values: list[float], p: int) -> float:
    return float(np.percentile(values, p)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    clips = dataset(args.session, args.manifest)
    process = psutil.Process()
    rss_before = process.memory_info().rss
    available_before = psutil.virtual_memory().available
    gpu_before = gpu_process_memory_mib(process.pid)
    load_started = time.perf_counter()
    model = CANDIDATES[args.candidate]()
    load_ms = (time.perf_counter() - load_started) * 1000
    rss_after_load = process.memory_info().rss
    gpu_after_load = gpu_process_memory_mib(process.pid)
    model_files = [{"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in model.model_files()]
    rows = []
    began_all = time.perf_counter()
    cpu_all_before = process.cpu_times()
    with ResourceMonitor() as monitor:
        for index, clip in enumerate(clips):
            started = time.perf_counter()
            cpu_before = process.cpu_times()
            try:
                hypothesis, details = model.transcribe(clip["pcm"], clip["sample_rate"])
                error = None
            except Exception as exc:
                hypothesis, details, error = "", {}, repr(exc)
            elapsed_ms = (time.perf_counter() - started) * 1000
            cpu_after = process.cpu_times()
            diag = diagnostics(hypothesis, clip["duration_s"])
            rows.append({
                "index": index,
                "utterance_id": clip["utterance_id"],
                "reference": clip["reference"],
                "hypothesis": hypothesis,
                "exact": normalize_words(clip["reference"]) == normalize_words(hypothesis),
                "understandable": understandable(hypothesis),
                "audio_path": clip["path"],
                "audio_sha256": clip["sha256"],
                "audio_duration_s": clip["duration_s"],
                "sample_rate": clip["sample_rate"],
                "input_transform": "none_float32_mono_16khz",
                "latency_ms": elapsed_ms,
                "rtf": elapsed_ms / max(1, clip["duration_s"] * 1000),
                "cpu_seconds": cpu_after.user + cpu_after.system - cpu_before.user - cpu_before.system,
                "rss_after_bytes": process.memory_info().rss,
                "error": error,
                **details,
                **diag,
                **errors(clip["reference"], hypothesis),
            })
    cpu_all_after = process.cpu_times()
    wall_s = time.perf_counter() - began_all
    if hasattr(model, "close"):
        model.close()
    samples = monitor.samples
    peak_rss = max([rss_after_load] + [row["rss_after_bytes"] for row in rows] +
        [sample["rss_bytes"] for sample in samples])
    hypotheses = [" ".join(normalize_words(row["hypothesis"])) for row in rows]
    modes = {value: hypotheses.count(value) for value in set(hypotheses)}
    cpu_seconds_total = cpu_all_after.user + cpu_all_after.system - cpu_all_before.user - cpu_all_before.system
    result = {
        "schema": "anydes.local_stt_candidate.v1",
        "candidate": args.candidate,
        "model": model.name,
        "backend": model.backend,
        "session": str(args.session.resolve()),
        "manifest": str(args.manifest.resolve()),
        "immutable_audio_verified": True,
        "utterances": len(rows),
        "input_contract": {
            "format": "float32 mono",
            "sample_rate": 16000,
            "normalization": None,
            "resampling": None,
            "vad": None,
            "preprocessing": None,
            "language": "es",
            "context": None,
            "hotwords": None,
            "post_replacements": None,
        },
        "model_files": model_files,
        "summary": {
            "exact": sum(row["exact"] for row in rows),
            "understandable": sum(row["understandable"] for row in rows),
            "corpus_wer": sum(row["word_errors"] for row in rows) / max(1, sum(row["reference_words"] for row in rows)),
            "corpus_cer": sum(row["character_errors"] for row in rows) / max(1, sum(row["reference_characters"] for row in rows)),
            "hallucinations": sum(row["hallucination_suspected"] for row in rows),
            "pathological_repetitions": sum(row["pathological_repetition"] for row in rows),
            "incorrect_language_suspected": sum(row["incorrect_language_suspected"] for row in rows),
            "errors": sum(row["error"] is not None for row in rows),
            "unique_normalized_transcripts": len(modes),
            "modal_transcript_count": max(modes.values(), default=0),
            "latency_p50_ms": percentile([row["latency_ms"] for row in rows], 50),
            "latency_p95_ms": percentile([row["latency_ms"] for row in rows], 95),
            "rtf_p50": percentile([row["rtf"] for row in rows], 50),
            "rtf_p95": percentile([row["rtf"] for row in rows], 95),
            "first_transcription_ms": rows[0]["latency_ms"],
            "load_ms": load_ms,
            "wall_s": wall_s,
            "cpu_seconds_total": cpu_seconds_total,
            "cpu_percent_average": 100 * cpu_seconds_total / max(wall_s, 0.001),
            "cpu_percent_p95_sampled": percentile([sample["cpu_percent"] for sample in samples], 95),
            "rss_before_bytes": rss_before,
            "rss_after_load_bytes": rss_after_load,
            "rss_load_increment_bytes": max(0, rss_after_load - rss_before),
            "rss_peak_bytes": peak_rss,
            "rss_peak_increment_bytes": max(0, peak_rss - rss_before),
            "system_available_before_bytes": available_before,
            "system_available_min_bytes": min([available_before] + [sample["system_available_bytes"] for sample in samples]),
            "gpu_process_memory_before_mib": gpu_before,
            "gpu_process_memory_after_load_mib": gpu_after_load,
            "gpu_process_memory_peak_mib": gpu_process_memory_mib(process.pid),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "candidate": args.candidate,
        "summary": result["summary"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
