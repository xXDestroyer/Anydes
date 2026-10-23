from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


SegmentationStrategy = Literal["fixed15", "short_first", "adaptive"]
_WORD = re.compile(r"\S+")
_MAJOR_END = re.compile(r"(?:[.!?]+|…+|[.!?]+[\"')\]]*)$")
_MINOR_END = re.compile(r"(?:[,;:]+|[,;:]+[\"')\]]*)$")


@dataclass(frozen=True)
class ProsodicSegment:
    index: int
    text: str
    word_count: int
    boundary: str


@dataclass(frozen=True)
class SegmentationProfile:
    first_target: int
    first_minimum: int
    first_maximum: int
    next_target: int
    next_minimum: int
    next_maximum: int


PROFILES: dict[SegmentationStrategy, SegmentationProfile] = {
    "fixed15": SegmentationProfile(15, 8, 18, 15, 8, 18),
    "short_first": SegmentationProfile(12, 8, 18, 25, 15, 35),
    "adaptive": SegmentationProfile(12, 8, 18, 24, 14, 34),
}


class ProsodicSegmenter:
    """Split speech at natural boundaries without changing its words."""

    def __init__(self, strategy: SegmentationStrategy = "adaptive") -> None:
        if strategy not in PROFILES:
            raise ValueError(f"unsupported_voice_segmentation_strategy:{strategy}")
        self.strategy = strategy
        self.profile = PROFILES[strategy]

    def split(self, text: str) -> tuple[ProsodicSegment, ...]:
        normalized = " ".join(text.split()).strip()
        words = _WORD.findall(normalized)
        if not words:
            return ()
        if len(words) <= self.profile.first_maximum:
            return (ProsodicSegment(0, normalized, len(words), self._boundary(words[-1])),)

        chunks: list[list[str]] = []
        cursor = 0
        while cursor < len(words):
            first = not chunks
            target = self.profile.first_target if first else self.profile.next_target
            minimum = self.profile.first_minimum if first else self.profile.next_minimum
            maximum = self.profile.first_maximum if first else self.profile.next_maximum
            remaining = len(words) - cursor
            if remaining <= maximum:
                chunks.append(words[cursor:])
                break

            local_max = min(maximum, remaining)
            candidates: list[tuple[float, int]] = []
            for count in range(minimum, local_max + 1):
                token = words[cursor + count - 1]
                boundary = self._boundary(token)
                if boundary == "none":
                    continue
                priority = 0 if boundary == "major" else 3
                score = abs(count - target) + priority
                candidates.append((score, count))
            count = min(candidates)[1] if candidates else min(target, local_max)

            trailing = remaining - count
            if 0 < trailing < 3:
                count -= 3 - trailing
            chunks.append(words[cursor : cursor + count])
            cursor += count

        if len(chunks) > 1 and len(chunks[-1]) < 3:
            chunks[-2].extend(chunks.pop())

        return tuple(
            ProsodicSegment(
                index=index,
                text=" ".join(chunk),
                word_count=len(chunk),
                boundary=self._boundary(chunk[-1]),
            )
            for index, chunk in enumerate(chunks)
        )

    @staticmethod
    def _boundary(token: str) -> str:
        if _MAJOR_END.search(token):
            return "major"
        if _MINOR_END.search(token):
            return "minor"
        return "none"

