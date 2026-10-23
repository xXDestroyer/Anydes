"""Experimental voice-v2 architecture, isolated from the official voice path."""

from anydes.voz.nueva.contracts import (
    AudioBuffer,
    EngineError,
    PlaybackResult,
    SynthesisResult,
    VisemeCue,
    VoiceDirection,
    VoiceEvent,
    VoicePipelineTrace,
    VoiceSynthesisRequest,
)
from anydes.voz.nueva.director import VoiceDirector
from anydes.voz.nueva.manager import NewVoiceManager

__all__ = [
    "AudioBuffer",
    "EngineError",
    "NewVoiceManager",
    "PlaybackResult",
    "SynthesisResult",
    "VisemeCue",
    "VoiceDirection",
    "VoiceDirector",
    "VoiceEvent",
    "VoicePipelineTrace",
    "VoiceSynthesisRequest",
]
