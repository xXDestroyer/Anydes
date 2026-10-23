"""Bounded PCM handoff with timestamps from PortAudio's output callback.

The callback only copies already prepared PCM and records timing. Timeline
planning, event sinks and waits run outside the real-time callback.
"""
from __future__ import annotations

from collections import deque
import threading
import time


class DacOutputStream:
    def __init__(self, sd, *, samplerate, channels, blocksize, event_sink, cancel_event=None):
        self.rate = samplerate
        self.frame_bytes = channels * 2
        self._sd = sd
        self._cancel_event = cancel_event or threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._sink = event_sink
        self._packets = deque()
        self._events = deque()
        self._silence = bytes(blocksize * self.frame_bytes)
        self._segment = None
        self._current_segment = None
        self._segment_frames = 0
        self._packet_offset = 0
        self._segment_clock = 0.0
        self._eof = False
        self._aborted = False
        self._error = None
        self._finished = threading.Event()
        self._dispatch_stop = threading.Event()
        self.submitted_frames = 0
        self.played_frames = 0
        self.last_dac_end = 0.0
        self.first_dac_start = 0.0
        self.last_active_end = 0.0
        self.device_underflows = 0
        self.starvation_frames = 0
        self._started = False
        self._raw = sd.RawOutputStream(
            samplerate=samplerate, channels=channels, dtype="int16",
            blocksize=blocksize, callback=self._callback,
            finished_callback=self._finished.set,
        )
        self._dispatcher = threading.Thread(target=self._dispatch, name="anydes-dac-events", daemon=True)

    @property
    def latency(self):
        return self._raw.latency

    def start(self):
        self._dispatcher.start()

    def set_segment(self, payload):
        self._segment = dict(payload)
        self._segment["generation_id"] = getattr(self._cancel_event, "generation", None)

    def write(self, pcm):
        deadline = time.perf_counter() + 5.0
        while len(self._packets) >= 4 and not self._aborted:
            self._check_error()
            if time.perf_counter() > deadline:
                raise TimeoutError("dac_consumer_stalled")
            self._finished.wait(0.002)
        self._check_error()
        if self._aborted:
            return
        with self._lifecycle_lock:
            if self._aborted or self._cancel_event.is_set():
                return
            self._packets.append((memoryview(pcm), self._segment))
            self.submitted_frames += len(pcm) // self.frame_bytes
            if not self._started:
                self._started = True
                self._raw.start()

    def _check_error(self):
        if self._error is not None:
            raise RuntimeError("dac_callback_or_event_failed") from self._error

    def _callback(self, outdata, frames, timing, status):
        out = memoryview(outdata).cast("B")
        out[:] = self._silence[:len(out)]
        try:
            if self._aborted or self._cancel_event.is_set():
                raise self._sd.CallbackAbort
            if status.output_underflow:
                self.device_underflows += 1
            # Both PortAudio timestamps share one clock. Convert their delta to
            # perf_counter, never treating time of queue submission as audible.
            dac = time.perf_counter() + float(timing.outputBufferDacTime - timing.currentTime)
            cursor = 0
            while cursor < frames and self._packets:
                pcm, segment = self._packets[0]
                if segment is not self._current_segment:
                    self._current_segment = segment
                    self._segment_frames = 0
                    self._segment_clock = dac + cursor / self.rate
                    if not self.first_dac_start:
                        self.first_dac_start = self._segment_clock
                    self._events.append((segment, self._segment_clock, False))
                elif cursor == 0:
                    observed_start = dac - self._segment_frames / self.rate
                    if abs(observed_start - self._segment_clock) > 0.002:
                        self._segment_clock = observed_start
                        self._events.append((segment, observed_start, True))
                count = min(frames - cursor, (len(pcm) - self._packet_offset) // self.frame_bytes)
                size = count * self.frame_bytes
                begin = cursor * self.frame_bytes
                out[begin:begin + size] = pcm[self._packet_offset:self._packet_offset + size]
                self._packet_offset += size
                self._segment_frames += count
                self.played_frames += count
                cursor += count
                self.last_dac_end = dac + cursor / self.rate
                active_end = float(segment.get("active_end_ms", 0)) / 1000
                active_start = float(segment.get("active_start_ms", 0)) / 1000
                if active_end > active_start and self._segment_frames / self.rate >= active_start:
                    self.last_active_end = self._segment_clock + min(active_end, self._segment_frames / self.rate)
                if self._packet_offset == len(pcm):
                    self._packets.popleft()
                    self._packet_offset = 0
            if self._aborted or self._cancel_event.is_set():
                out[:] = self._silence[:len(out)]
                raise self._sd.CallbackAbort
            if not self._packets and self._eof:
                raise self._sd.CallbackStop
            if cursor < frames and self.played_frames:
                self.starvation_frames += frames - cursor
        except (self._sd.CallbackStop, self._sd.CallbackAbort):
            raise
        except BaseException as exc:
            self._error = exc
            raise self._sd.CallbackAbort

    def _dispatch(self):
        while not self._dispatch_stop.is_set() or self._events:
            while self._events:
                segment, clock, correction = self._events.popleft()
                if self._aborted or self._cancel_event.is_set():
                    continue
                payload = dict(segment)
                payload.update(
                    audio_clock=clock,
                    audible_end_clock=clock + float(segment["duration_ms"]) / 1000,
                    active_end_clock=clock + float(segment.get("active_end_ms", 0)) / 1000,
                    clock_source="portaudio_output_dac",
                    clock_correction=correction,
                    device_latency_ms=float(self.latency) * 1000,
                )
                try:
                    self._sink(payload)
                except Exception as exc:
                    self._error = exc
            self._dispatch_stop.wait(0.002)

    def stop(self):
        self._eof = True  # Separate from PCM: EOS cannot displace valid audio.
        if not self._started:
            self._finished.set()
        pending_seconds = max(0, self.submitted_frames - self.played_frames) / self.rate
        if not self._finished.wait(pending_seconds + float(self.latency) + 5.0):
            self.abort()
            raise TimeoutError("dac_drain_timeout")
        self._check_error()

    def abort(self):
        self._cancel_event.set()
        with self._lifecycle_lock:
            self._aborted = True
            if self._started:
                self._raw.abort()
            self._packets.clear()
            self._finished.set()

    def close(self):
        self._dispatch_stop.set()
        if self._dispatcher.is_alive():
            self._dispatcher.join(timeout=2.0)
        self._raw.close()

    def diagnostics(self):
        return {
            "clock_source": "portaudio_output_dac",
            "submitted_frames": self.submitted_frames,
            "dac_frames": self.played_frames,
            "audio_end_clock": self.last_dac_end,
            "first_playback_clock": self.first_dac_start,
            "active_audio_end_clock": self.last_active_end,
            "device_underflows": self.device_underflows,
            "starvation_frames": self.starvation_frames,
            "discarded_frames": max(0, self.submitted_frames - self.played_frames) if self._aborted else 0,
            "discard_reason": "user_interrupt" if self._aborted else None,
        }
