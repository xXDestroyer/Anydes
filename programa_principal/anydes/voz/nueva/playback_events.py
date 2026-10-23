"""Deliver future DAC segments only when they actually reach the output.

Shared by Mini and the full avatar through NewVoiceManager.poll_events.
"""
from collections import deque
import time


class PlaybackEventClock:
    def __init__(self):
        self._pending = []
        self._finished_ids = deque(maxlen=128)
        self._cancelled_before = float("-inf")

    def deliver(self, events, now=None):
        now = time.perf_counter() if now is None else now
        ready = []
        for event in events:
            payload = event.payload
            pid = payload.get("playback_id", 0)
            if event.kind == "playback_started" and payload.get("clock_source") == "portaudio_output_dac":
                if pid in self._finished_ids or event.created_monotonic <= self._cancelled_before:
                    continue
                segment = payload.get("segment_index", 0)
                self._pending = [e for e in self._pending if (e.payload.get("playback_id", 0), e.payload.get("segment_index", 0)) != (pid, segment)]
                self._pending.append(event)
            else:
                # Manager-wide cancellation has no playback_id. It must also
                # invalidate future DAC starts already waiting for the GUI.
                if event.kind == "playback_interrupted" and not pid:
                    self._cancelled_before = max(self._cancelled_before, event.created_monotonic)
                    self._pending = [e for e in self._pending if e.created_monotonic > self._cancelled_before]
                    ready = [e for e in ready if e.kind != "playback_started" or e.created_monotonic > self._cancelled_before]
                if event.kind in {"playback_stopped", "playback_interrupted", "voice_failed"}:
                    if event.kind == "playback_stopped":
                        ready.extend(e for e in self._pending if e.payload.get("playback_id", 0) == pid and float(e.payload["audio_clock"]) <= now)
                    self._pending = [e for e in self._pending if e.payload.get("playback_id", 0) != pid]
                    self._finished_ids.append(pid)
                ready.append(event)
        ready.extend(e for e in self._pending if float(e.payload["audio_clock"]) <= now)
        self._pending = [e for e in self._pending if float(e.payload["audio_clock"]) > now]
        return tuple(sorted(ready, key=lambda e: float(e.payload.get("audio_clock", e.created_monotonic)) if e.kind == "playback_started" else e.created_monotonic))
