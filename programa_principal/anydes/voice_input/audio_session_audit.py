"""Objective boundary and signal audit for an immutable manual speech session."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def rms(samples: np.ndarray) -> float:
    if not len(samples):
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))


def dbfs(value: float) -> float:
    return float(20.0 * np.log10(max(value, 1e-12)))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_utterance(metadata_path: Path) -> dict:
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    folder = metadata_path.parent
    sample_rate = int(meta["capture"]["sample_rate"])
    raw_path = folder / meta["files"]["raw_float32"]
    vad_path = folder / meta["files"]["vad_segment_float32"]
    stt_path = folder / meta["files"]["segment_float32"]
    raw = np.load(raw_path, allow_pickle=False).astype(np.float32, copy=False)
    vad = np.load(vad_path, allow_pickle=False).astype(np.float32, copy=False)
    stt = np.load(stt_path, allow_pickle=False).astype(np.float32, copy=False)
    offset = int(meta["vad"]["segment_start_offset_samples"])
    raw_slice = raw[offset:offset + len(stt)]
    exact_raw_slice = len(raw_slice) == len(stt) and np.array_equal(raw_slice, stt)
    exact_vad_stt = len(vad) == len(stt) and np.array_equal(vad, stt)
    endpoint = meta.get("endpointing", {}).get("metrics", {})
    pre_s = float(endpoint.get("silence_before_voice_s", 0.0))
    tail_s = float(endpoint.get("silence_after_voice_s", 0.0))
    speech_rms = float(endpoint.get("speech_rms", rms(stt)))
    tail_rms = float(endpoint.get("tail_rms", 0.0))
    peak = float(np.max(np.abs(stt))) if len(stt) else 0.0
    clipping_percent = float(np.mean(np.abs(stt) >= 0.999) * 100.0) if len(stt) else 0.0
    snr_db = float(20.0 * np.log10(max(speech_rms, 1e-12) / max(tail_rms, 1e-12)))
    edge_n = min(len(stt), max(1, round(sample_rate * 0.04)))
    start_edge_rms = rms(stt[:edge_n])
    end_edge_rms = rms(stt[-edge_n:])
    possible_start_cut = pre_s < 0.12 or start_edge_rms > max(0.002, speech_rms * 0.45)
    possible_end_cut = tail_s < 0.12 or end_edge_rms > max(0.002, speech_rms * 0.45)
    low_level = speech_rms < 0.006
    poor_snr = snr_db < 15.0
    clipped = clipping_percent > 0.1
    issues = []
    if not exact_raw_slice:
        issues.append("stt_not_exact_raw_slice")
    if not exact_vad_stt:
        issues.append("vad_stt_mismatch")
    if possible_start_cut:
        issues.append("possible_start_cut")
    if possible_end_cut:
        issues.append("possible_end_cut")
    if low_level:
        issues.append("low_level")
    if poor_snr:
        issues.append("poor_snr")
    if clipped:
        issues.append("clipping")
    return {
        "utterance_id": meta["utterance_id"],
        "reference": meta.get("reference_text"),
        "device": meta["capture"].get("device_name"),
        "sample_rate": sample_rate,
        "raw_duration_s": len(raw) / sample_rate,
        "stt_duration_s": len(stt) / sample_rate,
        "voice_duration_s": float(endpoint.get("voice_detected_duration_s", 0.0)),
        "pre_voice_s": pre_s,
        "tail_s": tail_s,
        "speech_rms": speech_rms,
        "speech_dbfs": dbfs(speech_rms),
        "tail_rms": tail_rms,
        "snr_db": snr_db,
        "peak": peak,
        "peak_dbfs": dbfs(peak),
        "clipping_percent": clipping_percent,
        "start_edge_rms": start_edge_rms,
        "end_edge_rms": end_edge_rms,
        "exact_raw_slice": exact_raw_slice,
        "exact_vad_stt": exact_vad_stt,
        "possible_start_cut": possible_start_cut,
        "possible_end_cut": possible_end_cut,
        "low_level": low_level,
        "poor_snr": poor_snr,
        "clipped": clipped,
        "issues": issues,
        "raw_sha256": sha256(raw_path),
        "stt_sha256": sha256(stt_path),
        "metadata": str(metadata_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    args = parser.parse_args()
    rows = [audit_utterance(path) for path in sorted(args.session.rglob("metadata.json"))]
    rows = [row for row in rows if row["reference"]]
    if not rows:
        raise SystemExit("No hay utterances RCA con referencia.")
    critical = [row for row in rows if any(issue in row["issues"] for issue in
        ("stt_not_exact_raw_slice", "vad_stt_mismatch", "possible_start_cut", "low_level", "poor_snr", "clipping"))]
    capture_pipeline_ok = len(critical) <= max(1, len(rows) // 10)
    result = {
        "schema": "anydes.speech_audio_session_audit.v1",
        "session": str(args.session.resolve()),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "capture_pipeline_status": "CAPTURE_PIPELINE_OK" if capture_pipeline_ok else "CAPTURE_PIPELINE_NEEDS_WORK",
        "utterances": len(rows),
        "exact_raw_slice_count": sum(row["exact_raw_slice"] for row in rows),
        "exact_vad_stt_count": sum(row["exact_vad_stt"] for row in rows),
        "possible_start_cuts": sum(row["possible_start_cut"] for row in rows),
        "possible_end_cuts": sum(row["possible_end_cut"] for row in rows),
        "low_level_count": sum(row["low_level"] for row in rows),
        "poor_snr_count": sum(row["poor_snr"] for row in rows),
        "clipped_count": sum(row["clipped"] for row in rows),
        "speech_rms_min": min(row["speech_rms"] for row in rows),
        "speech_rms_median": float(np.median([row["speech_rms"] for row in rows])),
        "speech_rms_max": max(row["speech_rms"] for row in rows),
        "snr_db_min": min(row["snr_db"] for row in rows),
        "snr_db_median": float(np.median([row["snr_db"] for row in rows])),
        "peak_max": max(row["peak"] for row in rows),
        "rows": rows,
        "rules": {
            "start_cut": "pre_voice < 120 ms or first 40 ms RMS > max(-54 dBFS, 45% speech RMS)",
            "end_cut": "tail < 120 ms or last 40 ms RMS > max(-54 dBFS, 45% speech RMS)",
            "low_level": "speech RMS < 0.006 (-44.4 dBFS)",
            "poor_snr": "estimated speech/tail SNR < 15 dB",
            "clipping": "> 0.1% samples at full scale",
        },
    }
    json_path = args.session / "audio-session-audit.json"
    csv_path = args.session / "audio-session-audit.csv"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key for key in rows[0] if key not in {"issues"}])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if key != "issues"})
    print(json.dumps({key: value for key, value in result.items() if key not in {"rows", "rules"}}, indent=2, ensure_ascii=False))
    print(json.dumps({"json": str(json_path), "csv": str(csv_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
