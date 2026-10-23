from __future__ import annotations

import re
from dataclasses import dataclass
import time

from anydes.voz.nueva.contracts import AudioBuffer, VisemeCue


_VISEME = {
    "a": "aa",
    "e": "ee",
    "i": "ih",
    "o": "oh",
    "u": "ou",
}

VISUAL_LIP_SYNC_FINAL_STRETCH = 1.15
VISUAL_LIP_SYNC_FINAL_REGION = 0.20


def visual_alignment_from_audio_clock(
    audio_clock: float,
    *,
    observed_clock: float | None = None,
) -> tuple[float, float]:
    """Return elapsed and pending milliseconds against the audible PCM clock."""

    observed = time.perf_counter() if observed_clock is None else float(observed_clock)
    delta_ms = (observed - float(audio_clock)) * 1000.0
    return max(0.0, delta_ms), max(0.0, -delta_ms)


def stretch_final_visual_lip_sync(
    timeline: list[dict[str, object]],
    duration_ms: float,
    *,
    stretch: float = VISUAL_LIP_SYNC_FINAL_STRETCH,
    final_region: float = VISUAL_LIP_SYNC_FINAL_REGION,
) -> tuple[list[dict[str, object]], float]:
    """Slow the final viseme region without adding post-roll mouth shapes.

    The transformed line remains locked to the PCM duration. The final region is
    stretched in place while the preceding region absorbs the small timing
    difference. Cue count, order, viseme and weight are preserved.
    """

    duration = max(0.0, float(duration_ms))
    if duration <= 0.0 or not timeline:
        return [dict(raw) for raw in timeline], duration
    normalized = [_normalized_visual_cue(raw) for raw in timeline]
    active = [cue for cue in normalized if cue[2] != "neutral"]
    if not active:
        return [dict(raw) for raw in timeline], duration

    tail_ratio = max(0.15, min(0.25, float(final_region)))
    tail_stretch = max(1.0, float(stretch))
    active_start = min(cue[0] for cue in active)
    active_end = max(cue[1] for cue in active)
    active_span = max(0.0, active_end - active_start)
    if active_span <= 0.0:
        return [dict(raw) for raw in timeline], duration

    source_boundary = active_end - active_span * tail_ratio
    source_tail_duration = active_end - source_boundary
    target_tail_duration = source_tail_duration * tail_stretch
    target_active_end = min(duration, active_end + target_tail_duration - source_tail_duration)
    target_boundary = target_active_end - target_tail_duration
    prefix_scale = target_boundary / source_boundary if source_boundary > 0.0 else 1.0
    trailing_source = max(0.0, duration - active_end)
    trailing_target = max(0.0, duration - target_active_end)

    def remap(timestamp_ms: float) -> float:
        timestamp = max(0.0, min(duration, timestamp_ms))
        if timestamp <= source_boundary:
            return timestamp * prefix_scale
        if timestamp <= active_end:
            return target_boundary + (timestamp - source_boundary) * tail_stretch
        if trailing_source <= 0.0:
            return duration
        return target_active_end + (timestamp - active_end) * trailing_target / trailing_source

    stretched: list[dict[str, object]] = []
    for raw in timeline:
        start, end, viseme, weight = _normalized_visual_cue(raw)
        mapped_start = remap(start)
        mapped_end = remap(end)
        if mapped_end <= mapped_start:
            continue
        cue = dict(raw)
        cue.update(
            {
                "start_ms": mapped_start,
                "end_ms": mapped_end,
                "viseme": viseme,
                "weight": weight,
            }
        )
        stretched.append(cue)
    return stretched, duration


def viseme_change_timestamps(
    timeline: list[dict[str, object]],
    *,
    last: int = 10,
) -> list[dict[str, object]]:
    """Return the latest actual shape transitions for visual validation."""

    changes: list[dict[str, object]] = []
    previous = ""
    previous_at: float | None = None
    for raw in sorted(timeline, key=lambda cue: _normalized_visual_cue(cue)[0]):
        start, _, viseme, _ = _normalized_visual_cue(raw)
        if viseme == previous:
            continue
        changes.append(
            {
                "timestamp_ms": round(start, 3),
                "viseme": viseme,
                "interval_ms": None if previous_at is None else round(start - previous_at, 3),
            }
        )
        previous = viseme
        previous_at = start
    return changes[-max(1, int(last)) :]


def _normalized_visual_cue(raw: dict[str, object]) -> tuple[float, float, str, float]:
    start = max(0.0, float(raw.get("start_ms", raw.get("startMs", 0.0)) or 0.0))
    end = max(start, float(raw.get("end_ms", raw.get("endMs", start)) or start))
    viseme = str(raw.get("viseme") or "neutral")
    weight = max(0.0, min(1.0, float(raw.get("weight") or 0.0)))
    return start, end, viseme, weight


@dataclass(frozen=True)
class LipSyncPlanner:
    """Fits Spanish grapheme visemes to the audio that will actually play."""

    def build(
        self,
        spoken_text: str,
        audio: AudioBuffer,
        engine_alignment: tuple[VisemeCue, ...] = (),
    ) -> tuple[VisemeCue, ...]:
        duration_ms = audio.duration_seconds * 1000.0
        if duration_ms <= 0:
            return ()
        if engine_alignment:
            return self._sanitize_alignment(engine_alignment, duration_ms)
        active_start, active_end = self._active_audio_bounds(audio)
        active_start_ms = active_start / audio.sample_rate * 1000.0
        active_end_ms = active_end / audio.sample_rate * 1000.0
        if active_end_ms <= active_start_ms:
            return (VisemeCue(0.0, duration_ms, "neutral", 0.0),)
        units = self._units(spoken_text)
        total_weight = sum(weight for _, weight in units) or 1.0
        cursor = active_start_ms
        scale = (active_end_ms - active_start_ms) / total_weight
        cues: list[VisemeCue] = []
        if active_start_ms > 1.0:
            cues.append(VisemeCue(0.0, active_start_ms, "neutral", 0.0))
        for viseme, weight in units:
            end = min(active_end_ms, cursor + weight * scale)
            cue = VisemeCue(cursor, end, viseme, 0.0 if viseme == "neutral" else 0.72)
            if cues and cues[-1].viseme == cue.viseme and abs(cues[-1].end_ms - cue.start_ms) < 1.0:
                previous = cues.pop()
                cues.append(VisemeCue(previous.start_ms, cue.end_ms, cue.viseme, cue.weight))
            else:
                cues.append(cue)
            cursor = end
        if active_end_ms < duration_ms:
            cues.append(VisemeCue(active_end_ms, duration_ms, "neutral", 0.0))
        return tuple(cues)

    @staticmethod
    def _units(text: str) -> list[tuple[str, float]]:
        normalized = re.sub(r"\s+", " ", text.casefold())
        units: list[tuple[str, float]] = []
        last_vowel = "aa"
        for char in normalized:
            base = {
                "á": "a",
                "é": "e",
                "í": "i",
                "ó": "o",
                "ú": "u",
                "ü": "u",
            }.get(char, char)
            if base in _VISEME:
                last_vowel = _VISEME[base]
                units.append((last_vowel, 1.0))
            elif char in ".!?":
                units.append(("neutral", 2.0))
            elif char in ",;:":
                units.append(("neutral", 1.25))
            elif char.isspace():
                units.append(("neutral", 0.38))
            elif char.isalpha():
                units.append((last_vowel, 0.42))
        return units or [("neutral", 1.0)]

    @staticmethod
    def _active_audio_bounds(audio: AudioBuffer) -> tuple[int, int]:
        try:
            import numpy as np
        except ImportError:
            return 0, audio.frame_count
        samples = np.frombuffer(audio.pcm_s16le, dtype="<i2")
        if audio.channels > 1:
            samples = samples.reshape(-1, audio.channels).mean(axis=1)
        if samples.size == 0:
            return 0, 0
        peak = float(np.max(np.abs(samples)))
        if peak <= 0:
            return 0, len(samples)
        active = np.flatnonzero(np.abs(samples) >= max(96.0, peak * 0.018))
        if not active.size:
            return 0, len(samples)
        padding = max(1, int(audio.sample_rate * 0.012))
        return max(0, int(active[0]) - padding), min(len(samples), int(active[-1]) + padding)

    @staticmethod
    def _sanitize_alignment(
        alignment: tuple[VisemeCue, ...],
        duration_ms: float,
    ) -> tuple[VisemeCue, ...]:
        allowed = {"aa", "ih", "ou", "ee", "oh", "neutral"}
        result = []
        for cue in sorted(alignment, key=lambda item: item.start_ms):
            if cue.viseme not in allowed:
                continue
            start = max(0.0, min(duration_ms, cue.start_ms))
            end = max(start, min(duration_ms, cue.end_ms))
            if end > start:
                result.append(VisemeCue(start, end, cue.viseme, cue.weight))
        return tuple(result)


def prepare_pcm_timeline(audio: AudioBuffer, timeline: tuple[VisemeCue, ...]):
    """Fit the last existing articulation to measured PCM, then apply energy.

    No added post-roll or repeated vowel cycle. If native streaming exhausts
    its guessed text slice, actual PCM energy drives existing open/small forms
    instead of falsely declaring the remainder silent. This is not forced
    phonetic alignment and does not claim to recover missing phonemes.
    """
    import numpy as np

    samples = np.frombuffer(audio.pcm_s16le, dtype="<i2").astype(np.float32)
    if audio.channels > 1:
        samples = samples.reshape(-1, audio.channels).mean(axis=1)
    hop = max(1, round(audio.sample_rate * 0.010))
    levels = np.array([
        float(np.sqrt(np.mean(samples[start:start + hop] ** 2)))
        for start in range(0, len(samples), hop)
    ])
    reference = max(1.0, float(np.percentile(levels, 90))) if levels.size else 1.0
    threshold = max(32.0, reference * 0.004)
    active = np.flatnonzero(levels >= threshold)
    duration = audio.duration_seconds * 1000
    if not active.size:
        return [VisemeCue(0, duration, "neutral", 0).to_dict()], 0.0, 0.0
    start_ms = int(active[0]) * hop / audio.sample_rate * 1000
    end_ms = min(audio.frame_count, (int(active[-1]) + 1) * hop) / audio.sample_rate * 1000
    source = list(LipSyncPlanner._sanitize_alignment(timeline, duration))
    voiced = [cue for cue in source if cue.viseme != "neutral" and cue.weight > 0]
    if voiced:
        first, last = voiced[0].start_ms, voiced[-1].end_ms
        boundary = max(first, last - (last - first) * 0.20)
        if end_ms > last and last > boundary:
            def remap(t):
                if t <= boundary:
                    return t
                if t <= last:
                    return boundary + (t - boundary) * (end_ms - boundary) / (last - boundary)
                return end_ms + (t - last) * max(0, duration - end_ms) / max(1, duration - last)
            source = [VisemeCue(remap(c.start_ms), remap(c.end_ms), c.viseme, c.weight) for c in source]
    result = []
    smoothed = 0.0
    fallback_shape = "ih"
    for index, level in enumerate(levels):
        begin = index * hop / audio.sample_rate * 1000
        end = min(audio.frame_count, (index + 1) * hop) / audio.sample_rate * 1000
        smoothed += (min(1.0, float(level) / reference) - smoothed) * 0.32
        cue = next((c for c in source if c.start_ms <= (begin + end) / 2 < c.end_ms), None)
        if level < threshold:
            name, weight = "neutral", 0.0
        elif cue is not None and cue.viseme != "neutral" and cue.weight > 0:
            name = cue.viseme
            weight = cue.weight * (0.55 + 0.45 * smoothed ** 0.5)
        elif voiced and begin < voiced[-1].end_ms:
            name, weight = "neutral", 0.0
        else:
            # Hysteresis follows measured syllabic energy, never elapsed time.
            if smoothed > 0.48:
                fallback_shape = "aa"
            elif smoothed < 0.30:
                fallback_shape = "ih"
            name, weight = fallback_shape, 0.28 + 0.50 * smoothed ** 0.5
        result.append(VisemeCue(begin, end, name, weight).to_dict())
    return result, start_ms, end_ms
