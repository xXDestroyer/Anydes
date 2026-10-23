from __future__ import annotations

from typing import TypeAlias


JsonValue: TypeAlias = (
    str | int | float | bool | None | dict[str, "JsonValue"] | list["JsonValue"]
)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
