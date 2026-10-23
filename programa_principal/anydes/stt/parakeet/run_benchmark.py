"""Orchestrate clean, isolated Parakeet/Moonshine runs over the latest corpus."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import subprocess
import sys

from common import latest_session


def run(command: list[str]) -> None:
    print("\n> " + " ".join(command))
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def main() -> None:
    session = latest_session(complete_only=True)
    if session is None:
        raise SystemExit("No existe una sesion completa de 30 frases. Ejecuta primero la opcion 1.")
    manifest = session / "immutable-audio-manifest.json"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = session / f"benchmark-{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    script = str(Path(__file__).with_name("candidate_run.py"))
    for candidate in ("parakeet", "moonshine"):
        run([sys.executable, script, "--candidate", candidate, "--manifest", str(manifest),
            "--output", str(output / f"{candidate}.json"), "--determinism-pass"])
    report = str(Path(__file__).with_name("build_report.py"))
    run([sys.executable, report, "--parakeet", str(output / "parakeet.json"),
        "--moonshine", str(output / "moonshine.json"), "--output-dir", str(output)])
    print("\nBenchmark terminado: " + str(output))


if __name__ == "__main__":
    main()
