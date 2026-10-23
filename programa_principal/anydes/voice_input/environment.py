"""Dependencies are isolated from the official environment and loaded on demand."""
from pathlib import Path
import os
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'datos'/'telemetria'/'voice_input'
MODELS=ROOT/'runtime'/'stt'/'models'


def setup():
    # Append: do not shadow production numpy/Qt/HTTP dependencies.
    vendor=str(ROOT/'runtime'/'python_voice_deps')
    if vendor not in sys.path:
        sys.path.append(vendor)
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
