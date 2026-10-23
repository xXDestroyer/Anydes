"""Context providers that adapt current Anydes state into cognitive contracts."""

from anydes.cognition.providers.curiosity import CuriosityProvider
from anydes.cognition.providers.emotion import EmotionProvider
from anydes.cognition.providers.environment import EnvironmentProvider
from anydes.cognition.providers.goals import GoalProvider
from anydes.cognition.providers.memory import MemoryProvider
from anydes.cognition.providers.relationship import RelationshipProvider
from anydes.cognition.providers.self_model import SelfProvider
from anydes.cognition.providers.uncertainty import UncertaintyProvider
from anydes.cognition.providers.vision import VisionProvider

__all__ = [
    "CuriosityProvider",
    "EmotionProvider",
    "EnvironmentProvider",
    "GoalProvider",
    "MemoryProvider",
    "RelationshipProvider",
    "SelfProvider",
    "UncertaintyProvider",
    "VisionProvider",
]
