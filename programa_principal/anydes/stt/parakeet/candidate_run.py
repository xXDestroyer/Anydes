"""Run one STT candidate in a fresh process over the frozen 30-phrase manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from common import (BENCH, CORPUS, atomic_json, intent_preserved_preliminary,
    language_drift, load_corpus, sha256, text_metrics)

sys.path.insert(0, str(BENCH))
from benchmark_candidate import (MoonshineSpanishBase, ParakeetTdtQ8,
    ResourceMonitor, diagnostics, gpu_process_memory_mib, percentile)
import numpy as np
import psutil


CANDIDATES = {
    "parakeet": ParakeetTdtQ8,
    "moonshine": MoonshineSpanishBase,
}


def load_manifest(path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("complete") or len(payload.get("recordings", [])) != 30:
        raise RuntimeError("La sesion no contiene las 30 tomas congeladas.")
    if payload.get("corpus_sha256") != sha256(CORPUS):
        raise RuntimeError("El corpus CSV ya no coincide con el hash congelado.")
    corpus = {row["id"]: row for row in load_corpus()}
    clips = []
    for recording in payload["recordings"]:
        expected = corpus.get(recording["id"])
        if expected is None or expected["text"] != recording["reference"]:
            raise RuntimeError(f"Ground truth no coincide para {recording['id']}.")
        path = Path(recording["audio_path"])
        if sha256(path) != recording["audio_sha256"]:
            raise RuntimeError(f"Hash de audio alterado: {recording['id']}.")
        pcm = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
        clips.append({**recording, "pcm": pcm})
    return payload, clips


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--determinism-pass", action="store_true")
    args = parser.parse_args()
    manifest, clips = load_manifest(args.manifest)
    process = psutil.Process()
    rss_before = process.memory_info().rss
    available_before = psutil.virtual_memory().available
    gpu_before = gpu_process_memory_mib(process.pid)
    load_started = time.perf_counter()
    model = CANDIDATES[args.candidate]()
    load_ms = (time.perf_counter() - load_started) * 1000
    rss_after_load = process.memory_info().rss
    rows = []
    began_all = time.perf_counter()
    cpu_all_before = process.cpu_times()
    with ResourceMonitor() as monitor:
        for clip in clips:
            started = time.perf_counter()
            cpu_before = process.cpu_times()
            try:
                hypothesis, details = model.transcribe(clip["pcm"], clip["sample_rate"])
                error = None
            except Exception as exc:
                hypothesis, details, error = "", {}, repr(exc)
            elapsed_ms = (time.perf_counter() - started) * 1000
            cpu_after = process.cpu_times()
            metrics = text_metrics(clip["reference"], hypothesis)
            drift = language_drift(hypothesis)
            hallucination = diagnostics(hypothesis, clip["duration_s"])
            row = {
                "id": clip["id"],
                "category": clip["category"],
                "reference": clip["reference"],
                "hypothesis": hypothesis,
                "audio_path": clip["audio_path"],
                "audio_sha256": clip["audio_sha256"],
                "audio_duration_s": clip["duration_s"],
                "sample_rate": clip["sample_rate"],
                "input_transform": "none_float32_mono_16khz",
                "latency_ms": elapsed_ms,
                "rtf": elapsed_ms / max(1, clip["duration_s"] * 1000),
                "cpu_seconds": cpu_after.user + cpu_after.system - cpu_before.user - cpu_before.system,
                "rss_after_bytes": process.memory_info().rss,
                "error": error,
                **details,
                **metrics,
                **drift,
                **hallucination,
            }
            row["intent_preserved_preliminary"] = intent_preserved_preliminary(
                clip["reference"], hypothesis, metrics)
            row["intent_preserved_human"] = None
            rows.append(row)
    cpu_all_after = process.cpu_times()
    wall_s = time.perf_counter() - began_all

    repeat_identical = None
    if args.determinism_pass:
        repeat_identical = 0
        for clip, row in zip(clips, rows):
            repeated, _ = model.transcribe(clip["pcm"], clip["sample_rate"])
            row["repeat_hypothesis"] = repeated
            row["repeat_identical"] = repeated == row["hypothesis"]
            repeat_identical += int(row["repeat_identical"])

    if hasattr(model, "close"):
        model.close()
    samples = monitor.samples
    peak_rss = max([rss_after_load] + [row["rss_after_bytes"] for row in rows] +
        [sample["rss_bytes"] for sample in samples])
    cpu_seconds = (cpu_all_after.user + cpu_all_after.system -
        cpu_all_before.user - cpu_all_before.system)
    model_files = [{"path": str(path.resolve()), "bytes": path.stat().st_size,
        "sha256": sha256(path)} for path in model.model_files()]
    total_words = sum(row["reference_words"] for row in rows)
    total_chars = sum(row["reference_characters"] for row in rows)
    result = {
        "schema": "anydes.stt_generalization_candidate.v1",
        "candidate": args.candidate,
        "model": model.name,
        "backend": model.backend,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "corpus_sha256": manifest["corpus_sha256"],
        "immutable_audio_verified": True,
        "input_contract": {
            "format": "float32 mono 16kHz",
            "vad": "already frozen by recording session",
            "normalization": None,
            "resampling": None,
            "context": None,
            "hotwords": None,
            "post_replacements": None,
            "language_code_sent": "es",
            "language_note": ("Parakeet TDT is multilingual but this head has no request-time "
                "language conditioning or language metadata" if args.candidate == "parakeet"
                else "Spanish-specific Moonshine model"),
            "native_anydes_hint": ("unsupported_and_ignored_by_tdt_head"
                if args.candidate == "parakeet" else "not_tested_in_clean_run"),
        },
        "model_files": model_files,
        "summary": {
            "utterances": len(rows),
            "exact": sum(row["exact"] for row in rows),
            "intent_preserved_preliminary": sum(row["intent_preserved_preliminary"] for row in rows),
            "intent_preserved_human": None,
            "corpus_wer": sum(row["word_errors"] for row in rows) / max(1, total_words),
            "corpus_cer": sum(row["character_errors"] for row in rows) / max(1, total_chars),
            "omissions": sum(row["omissions"] for row in rows),
            "insertions": sum(row["insertions"] for row in rows),
            "substitutions": sum(row["substitutions"] for row in rows),
            "language_drift": sum(row["language_drift"] for row in rows),
            "hallucinations": sum(row["hallucination_suspected"] for row in rows),
            "pathological_repetitions": sum(row["pathological_repetition"] for row in rows),
            "errors": sum(row["error"] is not None for row in rows),
            "latency_p50_ms": percentile([row["latency_ms"] for row in rows], 50),
            "latency_p95_ms": percentile([row["latency_ms"] for row in rows], 95),
            "rtf_p50": percentile([row["rtf"] for row in rows], 50),
            "rtf_p95": percentile([row["rtf"] for row in rows], 95),
            "load_ms": load_ms,
            "first_transcription_ms": rows[0]["latency_ms"] if rows else None,
            "cpu_percent_p50_sampled": percentile([sample["cpu_percent"] for sample in samples], 50),
            "cpu_percent_p95_sampled": percentile([sample["cpu_percent"] for sample in samples], 95),
            "cpu_seconds_total": cpu_seconds,
            "wall_s": wall_s,
            "rss_load_increment_bytes": max(0, rss_after_load - rss_before),
            "rss_peak_increment_bytes": max(0, peak_rss - rss_before),
            "system_available_before_bytes": available_before,
            "system_available_min_bytes": min([available_before] +
                [sample["system_available_bytes"] for sample in samples]),
            "gpu_process_memory_before_mib": gpu_before,
            "gpu_process_memory_peak_mib": gpu_process_memory_mib(process.pid),
            "repeat_identical": repeat_identical,
        },
        "rows": rows,
    }
    atomic_json(args.output, result)
    print(json.dumps({"output": str(args.output), "summary": result["summary"]},
        indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
