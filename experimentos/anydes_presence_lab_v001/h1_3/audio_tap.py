"""Read-only prepared PCM tap. Never executes inside the PortAudio callback."""
from collections import deque
from functools import wraps
import queue
import threading
import time
from itertools import count
import numpy as np


class AudioTap:
    def __init__(self):
        self.pending = queue.Queue(64)
        self.stopped = threading.Event()
        self.features = deque(maxlen=800)
        self.clocks = {}
        self.offsets = {}
        self.held = deque(maxlen=128)
        self.errors = self.dropped = self.packets = 0
        self.offer_cost_us = deque(maxlen=4096)
        self.output_diagnostics = deque(maxlen=64)
        self.callback_modified = False
        self.stream_ids = count(1)
        self.thread = threading.Thread(target=self._run, name='PresenceAudioFeatures')
        self.thread.start()

    def offer(self, item):
        try:
            self.pending.put_nowait(item)
        except queue.Full:
            self.dropped += 1
        except Exception:
            self.errors += 1

    def pcm(self, stream, pcm):
        # This runs AFTER the original write, never before playback submission.
        try:
            view = memoryview(pcm)
            if not view.readonly or view.nbytes > 2_000_000:
                self.dropped += 1
                return
            segment = stream._segment
            key = (stream._presence_token, segment.get('segment_index',0))
            count = view.nbytes // stream.frame_bytes
            offset = self.offsets.get(key, 0)
            self.offsets[key] = offset+count
            if len(self.offsets)>128: self.offsets.pop(next(iter(self.offsets)))
            self.offer(('pcm', key, view, offset, stream.rate, stream.frame_bytes//2,
                        getattr(stream._cancel_event, 'generation', None)))
        except Exception:
            self.errors += 1

    def _consume(self, item):
        if item[0] == 'clock':
            self.clocks[item[1]] = item[2]
            if len(self.clocks)>128: self.clocks.pop(next(iter(self.clocks)))
            return
        _, key, pcm, offset, rate, channels, generation = item
        samples = np.frombuffer(pcm, dtype='<i2').reshape(-1, channels)
        stride = max(1, round(rate*0.02))
        for start in range(0, len(samples), stride):
            block = samples[start:start+stride].astype(np.float32)/32768
            rms = float(np.sqrt(np.mean(block*block)))
            peak = float(np.max(np.abs(block)))
            self.held.append((key, (offset+start)/rate, rms, peak, generation))
        self.packets += 1

    def _run(self):
        while not self.stopped.is_set():
            try:
                self._consume(self.pending.get(timeout=0.01))
            except queue.Empty:
                pass
            except Exception:
                self.errors += 1
            # Features wait for the authoritative segment DAC clock.
            for _ in range(len(self.held)):
                key, offset, rms, peak, generation = self.held.popleft()
                clock = self.clocks.get(key)
                if clock is None:
                    self.held.append((key, offset, rms, peak, generation))
                else:
                    self.features.append((clock+offset, rms, peak, generation))

    def snapshot(self, generation, now=None):
        now = time.perf_counter() if now is None else now
        eligible = [x for x in tuple(self.features) if x[0] <= now and x[3] == generation]
        latest = max(eligible, default=None, key=lambda x:x[0])
        if latest is None or now-latest[0] > 0.12:
            return 0.0, 0.0, False
        return latest[1], latest[2], latest[1] > 0.006

    def install(self):
        from anydes.voz.nueva.dac_output import DacOutputStream
        original_init, original_write = DacOutputStream.__init__, DacOutputStream.write
        tap = self
        @wraps(original_init)
        def init(stream, *args, **kwargs):
            original_init(stream, *args, **kwargs)
            try: attach(stream)
            except Exception: tap.errors += 1
        def attach(stream):
            stream._presence_token = next(tap.stream_ids)
            sink = stream._sink
            def observe(payload):
                result = sink(payload)
                try:
                    tap.offer(('clock', (stream._presence_token,payload.get('segment_index',0)), payload['audio_clock']))
                except Exception:
                    tap.errors += 1
                return result
            stream._sink = observe
        @wraps(original_write)
        def write(stream, pcm):
            result = original_write(stream, pcm)
            started = time.perf_counter()
            try: tap.pcm(stream, pcm)
            except Exception: tap.errors += 1
            tap.offer_cost_us.append((time.perf_counter()-started)*1e6)
            return result
        DacOutputStream.__init__, DacOutputStream.write = init, write
        original_diagnostics = DacOutputStream.diagnostics
        @wraps(original_diagnostics)
        def diagnostics(stream):
            result = original_diagnostics(stream)
            try:
                tap.output_diagnostics.append({k:result[k] for k in ('submitted_frames','dac_frames','device_underflows','starvation_frames')})
            except Exception: tap.errors += 1
            return result
        DacOutputStream.diagnostics = diagnostics

    def close(self):
        self.stopped.set()
        self.thread.join(2)
        self.features.clear(); self.held.clear(); self.offsets.clear(); self.clocks.clear()
