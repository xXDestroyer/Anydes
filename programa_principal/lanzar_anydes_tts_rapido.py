from __future__ import annotations

import os

from lanzar_anydes import load_profile


def main() -> None:
    load_profile()
    os.environ["ANYDES_NEW_VOICE_ENGINE"] = "qwen3-tts-faster"
    os.environ["ANYDES_NEW_VOICE_DELIVERY_MODE"] = "native_stream"
    os.environ["ANYDES_NEW_VOICE_COMMON_PHRASE_CACHE_ENABLED"] = "false"
    os.environ["ANYDES_NEW_VOICE_WARMUP_ON_START"] = "true"
    os.environ["ANYDES_VISION_KEEP_ALIVE"] = "0s"
    os.environ["ANYDES_REALTIME_NUM_CTX"] = "768"
    os.environ["ANYDES_VISION_COORDINATE_GPU_RESIDENCY"] = "true"
    os.environ["ANYDES_FAST_TTS_MAX_SEQ_LEN"] = "1152"
    os.environ["ANYDES_VRAM_SAFETY_ENABLED"] = "true"
    from anydes.interfaz.pyside_companion.app import main as run_companion

    run_companion()


if __name__ == "__main__":
    main()
