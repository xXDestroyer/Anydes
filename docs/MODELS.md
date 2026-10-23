# Models and Optional Runtime Components

**No weights, native model runtime, model cache or voice reference is included.**
Download/configure components separately only after checking their upstream license,
hardware needs and redistribution terms. No right to redistribute weights is
asserted here. This project does not need Git LFS for excluded models.

| Role | Accepted configured system / source evidence | Setup boundary |
| --- | --- | --- |
| Primary reasoning | Groq API, openai/gpt-oss-120b in accepted harnesses | GROQ_API_KEY from private environment; explicit external-AI consent |
| Local fallback | qwen3:8b through local Ollama adapter | Local model service on loopback; obtain model separately |
| STT | Parakeet TDT 0.6B v3 Q8 CPU adapter | Separate GGUF/native nemo-speech DLL runtime |
| TTS / B4 | Accepted Qwen TTS native PCM/backend integration | Separate GPU/Torch environment, model and personally licensed voice configuration |
| Vision | Moondream/vision adapter family | Separate model/service and explicit vision consent |
| Visual presence | Procedural Three.js hologram; Qt embedding | No neural weights; npm Three.js 0.180.0 |

The Parakeet adapter's expected relative layout is documented in its backend.py:
`programa_principal/runtime/stt/nemo-speech/bin` and
`programa_principal/runtime/stt/models/parakeet`. Those directories and their DLLs/
GGUF files are intentionally absent. Configure an equivalent supported runtime;
do not rename unrelated binaries to satisfy a path check.

The Qwen/B4 backends have separate version/hardware requirements. The optional
voice requirements are only integration dependencies, not a claim that they install
the complete accepted voice engine or reproduce its identity. No reference voice
recording is distributed.

Use your own local data/profile directories. The original installation-specific
integrity manifests are not portable manifests for this anonymized repository.
See PORTABILITY.md before trying the preserved full runtime entry points.
