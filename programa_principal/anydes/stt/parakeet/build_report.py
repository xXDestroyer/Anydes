"""Build comparison artifacts for the frozen generalization replay."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from common import atomic_json


def mib(value):
    return float(value) / 1048576


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parakeet", type=Path, required=True)
    parser.add_argument("--moonshine", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    results = [json.loads(path.read_text(encoding="utf-8"))
        for path in (args.parakeet, args.moonshine)]
    first_hashes = {row["id"]: row["audio_sha256"] for row in results[0]["rows"]}
    if any({row["id"]: row["audio_sha256"] for row in result["rows"]} != first_hashes
            for result in results[1:]):
        raise RuntimeError("Los candidatos no recibieron exactamente los mismos audios.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison = []
    for result in results:
        summary = result["summary"]
        comparison.append({
            "candidate": result["candidate"],
            "exact": summary["exact"],
            "exact_percent": 100 * summary["exact"] / 30,
            "intent_preserved_preliminary": summary["intent_preserved_preliminary"],
            "intent_preserved_preliminary_percent": 100 * summary["intent_preserved_preliminary"] / 30,
            "intent_preserved_human": "PENDING_BLIND_REVIEW",
            "wer": summary["corpus_wer"],
            "cer": summary["corpus_cer"],
            "omissions": summary["omissions"],
            "insertions": summary["insertions"],
            "substitutions": summary["substitutions"],
            "language_drift": summary["language_drift"],
            "hallucinations": summary["hallucinations"],
            "pathological_repetitions": summary["pathological_repetitions"],
            "latency_p50_ms": summary["latency_p50_ms"],
            "latency_p95_ms": summary["latency_p95_ms"],
            "rtf_p50": summary["rtf_p50"],
            "rtf_p95": summary["rtf_p95"],
            "cpu_p50": summary["cpu_percent_p50_sampled"],
            "cpu_p95": summary["cpu_percent_p95_sampled"],
            "ram_load_increment_mib": mib(summary["rss_load_increment_bytes"]),
            "ram_peak_increment_mib": mib(summary["rss_peak_increment_bytes"]),
            "load_ms": summary["load_ms"],
            "first_transcription_ms": summary["first_transcription_ms"],
            "repeat_identical": summary["repeat_identical"],
            "vram_peak_mib": summary["gpu_process_memory_peak_mib"],
            "errors": summary["errors"],
        })
    summary_csv = args.output_dir / "GENERALIZATION_COMPARISON.csv"
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison[0].keys())
        writer.writeheader()
        writer.writerows(comparison)

    review_csv = args.output_dir / "INTENT_BLIND_REVIEW.csv"
    parakeet, moonshine = results
    with review_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["id", "category", "reference", "candidate_a", "candidate_b",
            "intent_a_yes_no", "intent_b_yes_no", "notes"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (a, b) in enumerate(zip(parakeet["rows"], moonshine["rows"])):
            # Alternating labels avoids making one column synonymous with one model.
            swap = index % 2 == 1
            writer.writerow({"id": a["id"], "category": a["category"],
                "reference": a["reference"],
                "candidate_a": b["hypothesis"] if swap else a["hypothesis"],
                "candidate_b": a["hypothesis"] if swap else b["hypothesis"],
                "intent_a_yes_no": "", "intent_b_yes_no": "", "notes": ""})
    key = {row["id"]: {"candidate_a": "moonshine" if index % 2 else "parakeet",
        "candidate_b": "parakeet" if index % 2 else "moonshine"}
        for index, row in enumerate(parakeet["rows"])}
    atomic_json(args.output_dir / "INTENT_BLIND_REVIEW_KEY.json", key)

    lines = [
        "# ANYDES STT GENERALIZATION - RESULTADO PROVISIONAL",
        "",
        "Los 30 audios y hashes fueron identicos para ambos candidatos. No se aplicaron hints, contexto, normalizacion ni reemplazos.",
        "",
        "La columna `intent_preserved_preliminary` es una comprobacion automatica conservadora. La decision final requiere completar `INTENT_BLIND_REVIEW.csv`.",
        "",
        "| Modelo | Exactas | Intencion preliminar | WER | CER | Drift | Halluc. | P50/P95 ms | RTF P50/P95 | RAM pico MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison:
        lines.append(f"| {row['candidate']} | {row['exact']}/30 | "
            f"{row['intent_preserved_preliminary']}/30 | {row['wer']:.3f} | {row['cer']:.3f} | "
            f"{row['language_drift']} | {row['hallucinations']} | "
            f"{row['latency_p50_ms']:.0f}/{row['latency_p95_ms']:.0f} | "
            f"{row['rtf_p50']:.3f}/{row['rtf_p95']:.3f} | {row['ram_peak_increment_mib']:.0f} |")
    lines.extend([
        "",
        "## Capacidad de hint e idioma de Parakeet TDT",
        "",
        "El runtime envia `language_code=es`, pero Parakeet TDT no ofrece condicionamiento por idioma ni metadata de idioma. El cambio a ingles se registra como `LANGUAGE_DRIFT` mediante diagnostico, no se corrige despues.",
        "",
        "El head TDT tampoco soporta word boosting/speech contexts: el runtime documenta que los ignora. Por eso no se fabrica una corrida `+ Anydes` que seria tecnicamente un no-op. El nombre se evaluara limpio.",
        "",
        "## Estado",
        "",
        "No promovido. Coexistencia con Qwen/B4 bloqueada hasta aprobar generalizacion y prueba libre.",
    ])
    report = args.output_dir / "GENERALIZATION_REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report), "summary_csv": str(summary_csv),
        "blind_review": str(review_csv)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
