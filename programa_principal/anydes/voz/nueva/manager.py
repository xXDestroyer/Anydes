from __future__ import annotations

import queue
import threading
import time
from dataclasses import asdict

from anydes.cognition.contracts import SpeechIntention, VoiceOutputResult
from anydes.voz.identity import VoiceIdentity
from anydes.voz.nueva.audio_pipeline import AudioPipeline
from anydes.voz.nueva.audio_pipeline import QueuedAudioBlock
from anydes.voz.nueva.common_phrase_cache import CommonPhraseAudioCache
from anydes.voz.nueva.contracts import (
    EngineError,
    SynthesisResult,
    VoiceEvent,
    VoicePipelineTrace,
    VoiceSynthesisRequest,
)
from anydes.voz.nueva.director import VoiceDirector
from anydes.voz.nueva.engines import TTSEngine
from anydes.voz.nueva.lip_sync import LipSyncPlanner
from anydes.voz.nueva.playback_events import PlaybackEventClock
from anydes.voz.nueva.normalizer import SpeechTextNormalizer
from anydes.voz.nueva.segmenter import ProsodicSegmenter


class NewVoiceManager:
    """Single authoritative path from SpeechIntention to audible playback."""

    def __init__(
        self,
        *,
        identity: VoiceIdentity,
        engine: TTSEngine,
        audio: AudioPipeline,
        emergency_engine: TTSEngine | None = None,
        delivery_mode: str = "monolithic",
        segmentation_strategy: str = "adaptive",
        phrase_cache: CommonPhraseAudioCache | None = None,
        idle_gpu_cache_release_seconds: float = 0.0,
    ) -> None:
        self.identity = identity
        self.engine = engine
        self.audio = audio
        self.emergency_engine = emergency_engine
        self.delivery_mode = delivery_mode
        self.segmenter = ProsodicSegmenter(segmentation_strategy)
        self.phrase_cache = phrase_cache
        self.director = VoiceDirector(identity)
        self.normalizer = SpeechTextNormalizer(identity.pronunciation_lexicon)
        self.lip_sync = LipSyncPlanner()
        self.provider_name = engine.engine_id
        self._events: queue.SimpleQueue[VoiceEvent] = queue.SimpleQueue()
        self._playback_event_clock = PlaybackEventClock()
        self._lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._generation = 0
        self._active_generation = 0
        self.stale_events_discarded = 0
        self._cancelled = threading.Event()
        self._warmup_started = False
        self._vram_safety = None
        self._idle_gpu_cache_release_seconds = max(0.0, float(idle_gpu_cache_release_seconds))
        self._idle_release_lock = threading.Lock()
        self._idle_release_timer: threading.Timer | None = None
        self._idle_release_generation = 0
        self._idle_release_attempts = 0
        self._idle_release_completed = 0
        self._idle_release_reuse_cycles = 0
        self._idle_release_waiting_for_reuse = False
        self._idle_release_cuda_graphs_preserved = True
        self._idle_release_last_result: dict[str, object] = {}
        self._closed = False
        self.last_trace = VoicePipelineTrace(engine_id=engine.engine_id, voice_id=engine.voice_id)

    @property
    def backend(self) -> "NewVoiceManager":
        """Compatibility surface used only by existing diagnostics."""

        return self

    @property
    def generation(self) -> int:
        with self._state_lock:
            return self._generation

    def speak(self, speech: SpeechIntention, *, generation: int | None = None) -> VoiceOutputResult:
        if not isinstance(speech, SpeechIntention):
            raise TypeError("NewVoiceManager.speak accepts only SpeechIntention.")
        generation = self.generation if generation is None else generation
        self.cancel_idle_gpu_cache_release()
        with self._idle_release_lock:
            if self._idle_release_waiting_for_reuse:
                self._idle_release_reuse_cycles += 1
                self._idle_release_waiting_for_reuse = False
        try:
            return self._speak_managed(speech, generation)
        finally:
            self.schedule_idle_gpu_cache_release()

    def _speak_managed(self, speech: SpeechIntention, generation: int) -> VoiceOutputResult:
        if generation != self.generation or self._closed:
            return self._result(speech, speech.plan_id or "speech:intention", "skipped", "superseded_turn")
        safety = self._vram_safety
        if safety is not None and not safety.prepare_for_task(
            "tts",
            primary=True,
            explicitly_requested=True,
        ):
            intention_id = str(
                speech.metadata.get("intention_id")
                or speech.plan_id
                or "speech:intention"
            )
            self.last_trace = VoicePipelineTrace(
                status="failed",
                phase="vram_safety",
                error_code="vram_critical_voice_suspended",
                input_characters=len(speech.text),
                engine_id=self.engine.engine_id,
                voice_id=self.engine.voice_id,
                identity_id=self.identity.identity_id,
            )
            self._emit("voice_failed", {"error_code": "vram_critical_voice_suspended", "voice_generation": generation})
            return self._result(
                speech,
                intention_id,
                "failed",
                "vram_critical_voice_suspended",
            )
        with self._lock:
            with self._state_lock:
                if generation != self._generation or self._closed:
                    return self._result(speech, speech.plan_id or "speech:intention", "skipped", "superseded_turn")
                # A cancelled request owns its Event forever. A later turn must
                # never clear it while an old producer is still unwinding.
                self._cancelled = threading.Event()
                self._cancelled.generation = generation
                self._active_generation = generation
            try:
                if safety is None:
                    return self._speak_locked(speech)
                with safety.active_task("tts"):
                    return self._speak_locked(speech)
            except Exception as exc:
                error_code = f"voice_pipeline_{type(exc).__name__}"
                intention_id = str(
                    speech.metadata.get("intention_id")
                    or speech.plan_id
                    or "speech:intention"
                )
                self.audio.cancel()
                self.last_trace = VoicePipelineTrace(
                    status="failed",
                    phase="voice_pipeline",
                    error_code=error_code,
                    input_characters=len(speech.text),
                    engine_id=self.engine.engine_id,
                    voice_id=self.engine.voice_id,
                    identity_id=self.identity.identity_id,
                )
                self._emit("voice_failed", {"error_code": error_code, "voice_generation": generation})
                return self._result(speech, intention_id, "failed", error_code)

    def attach_vram_safety(self, safety) -> None:
        self._vram_safety = safety

    def release_temporary_gpu_cache(self) -> dict[str, object]:
        release = getattr(self.engine, "release_temporary_gpu_cache", None)
        if not callable(release):
            return {"status": "unsupported"}
        result = dict(release() or {})
        self._emit("tts_temporary_cache_released", dict(result))
        return result

    def release_temporary_gpu_cache_if_idle(
        self,
        *,
        trigger: str,
    ) -> dict[str, object]:
        """Reclaim temporary CUDA memory only outside synthesis and playback."""

        safety = self._vram_safety
        if self._closed:
            result = {
                "status": "skipped",
                "reason": "voice_manager_closed",
                "voice_pipeline_idle": False,
            }
        elif safety is not None and safety.task_active("tts"):
            result = {
                "status": "busy",
                "reason": "tts_task_active",
                "voice_pipeline_idle": False,
            }
        elif not self._lock.acquire(blocking=False):
            result = {
                "status": "busy",
                "reason": "voice_pipeline_active",
                "voice_pipeline_idle": False,
            }
        else:
            try:
                result = {
                    **self.release_temporary_gpu_cache(),
                    "voice_pipeline_idle": True,
                    "tts_task_active": False,
                }
            finally:
                self._lock.release()
        payload = {**dict(result), "trigger": str(trigger or "predictive")}
        self._emit("tts_predictive_reclaim", payload)
        return payload

    def cancel_idle_gpu_cache_release(self) -> None:
        with self._idle_release_lock:
            self._idle_release_generation += 1
            timer = self._idle_release_timer
            self._idle_release_timer = None
        if timer is not None:
            timer.cancel()

    def schedule_idle_gpu_cache_release(self) -> None:
        delay = self._idle_gpu_cache_release_seconds
        if delay <= 0 or self._closed:
            return
        with self._idle_release_lock:
            self._idle_release_generation += 1
            generation = self._idle_release_generation
            previous = self._idle_release_timer
            timer = threading.Timer(delay, self._release_idle_gpu_cache, args=(generation,))
            timer.daemon = True
            self._idle_release_timer = timer
        if previous is not None:
            previous.cancel()
        timer.start()

    def idle_gpu_cache_release_snapshot(self) -> dict[str, object]:
        with self._idle_release_lock:
            return {
                "enabled": self._idle_gpu_cache_release_seconds > 0,
                "delay_seconds": self._idle_gpu_cache_release_seconds,
                "scheduled": self._idle_release_timer is not None,
                "attempts": self._idle_release_attempts,
                "completed": self._idle_release_completed,
                "reuse_cycles": self._idle_release_reuse_cycles,
                "cuda_graphs_preserved": self._idle_release_cuda_graphs_preserved,
                "last_result": dict(self._idle_release_last_result),
            }

    def _release_idle_gpu_cache(self, generation: int) -> None:
        with self._idle_release_lock:
            if self._closed or generation != self._idle_release_generation:
                return
            self._idle_release_timer = None
            self._idle_release_attempts += 1
        self._emit("tts_idle_release_started", {"generation": generation})
        if not self._lock.acquire(blocking=False):
            self.schedule_idle_gpu_cache_release()
            return
        try:
            try:
                result = self.release_temporary_gpu_cache()
            except Exception as exc:
                result = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}:{exc}",
                }
        finally:
            self._lock.release()
        with self._idle_release_lock:
            self._idle_release_last_result = dict(result)
            if result.get("status") == "completed":
                self._idle_release_completed += 1
                self._idle_release_waiting_for_reuse = True
                self._idle_release_cuda_graphs_preserved = (
                    self._idle_release_cuda_graphs_preserved
                    and bool(result.get("cuda_graphs_preserved", False))
                )

    def _speak_locked(self, speech: SpeechIntention) -> VoiceOutputResult:
        cancel_token = self._cancelled
        emit = self._turn_emitter()
        playback_sink = self._playback_sink()
        started = time.perf_counter()
        visible_text = speech.text.strip()
        intention_id = str(speech.metadata.get("intention_id") or speech.plan_id or "speech:intention")
        if not visible_text:
            return self._result(speech, intention_id, "skipped", "empty_intention_speech")

        direction = self.director.direct(speech)
        spoken_text = self.normalizer.normalize(direction.text)
        if not spoken_text:
            self.last_trace = VoicePipelineTrace(
                status="failed",
                phase="normalization",
                error_code="empty_normalized_speech",
                input_characters=len(visible_text),
                engine_id=self.engine.engine_id,
                voice_id=self.engine.voice_id,
                identity_id=self.identity.identity_id,
            )
            emit("voice_failed", {"error_code": "empty_normalized_speech"})
            return self._result(speech, intention_id, "failed", "empty_normalized_speech")

        request = VoiceSynthesisRequest(
            spoken_text=spoken_text,
            direction=direction,
            identity_id=self.identity.identity_id,
            language=self.identity.language,
            locale=self.identity.locale,
            reference_audio_path=self.identity.reference_audio_path,
        )
        if self.delivery_mode == "native_stream":
            stream = getattr(self.engine, "synthesize_stream", None)
            if callable(stream):
                return self._speak_native_stream_locked(
                    speech,
                    intention_id,
                    visible_text,
                    spoken_text,
                    direction,
                    request,
                    started,
                )
        if self.delivery_mode == "prosodic":
            segments = self.segmenter.split(spoken_text)
            if len(segments) > 1:
                return self._speak_segmented_locked(
                    speech,
                    intention_id,
                    visible_text,
                    spoken_text,
                    direction,
                    segments,
                    started,
                )
        emit("synthesis_started", {"engine_id": self.engine.engine_id})
        try:
            synthesis = self._synthesize(request)
        except Exception as exc:
            synthesis = SynthesisResult(
                status="failed",
                engine_id=self.engine.engine_id,
                voice_id=self.engine.voice_id,
                identity_id=self.identity.identity_id,
                error=EngineError(
                    f"synthesis_{type(exc).__name__}",
                    "synthesis",
                    self.engine.engine_id,
                    recoverable=True,
                ),
            )
        selected_engine = self.engine
        if (
            synthesis.status == "failed"
            and self.emergency_engine is not None
            and not cancel_token.is_set()
        ):
            # This path exists only after explicit opt-in. It is never silent.
            emit(
                "voice_failed",
                {
                    "error_code": synthesis.error.code if synthesis.error else "synthesis_failed",
                    "emergency_voice_requested": True,
                },
            )
            selected_engine = self.emergency_engine
            emit(
                "emergency_voice_started",
                {
                    "engine_id": selected_engine.engine_id,
                    "voice_id": selected_engine.voice_id,
                },
            )
            synthesis = selected_engine.synthesize(request)

        if synthesis.status != "ready" or synthesis.audio is None:
            error_code = synthesis.error.code if synthesis.error else "synthesis_failed"
            cancelled = synthesis.status == "cancelled" or cancel_token.is_set()
            self.last_trace = VoicePipelineTrace(
                status="cancelled" if cancelled else "failed",
                phase=synthesis.error.stage if synthesis.error else "synthesis",
                error_code=error_code,
                input_characters=len(visible_text),
                spoken_characters=len(spoken_text),
                synthesis_ms=synthesis.synthesis_ms,
                first_audio_ms=synthesis.first_audio_ms,
                cancelled=cancelled,
                engine_id=selected_engine.engine_id,
                voice_id=selected_engine.voice_id,
                identity_id=self.identity.identity_id,
                spoken_text=spoken_text,
            )
            if cancelled:
                error_code = "playback_cancelled"
                emit("playback_interrupted", {"reason": "user_interruption"})
            else:
                emit("voice_failed", {"error_code": error_code})
            status = "skipped" if cancelled else "failed"
            return self._result(speech, intention_id, status, error_code, selected_engine.engine_id)

        timeline = self.lip_sync.build(spoken_text, synthesis.audio, synthesis.alignment)
        emit(
            "audio_ready",
            {
                "duration_ms": synthesis.audio.duration_seconds * 1000.0,
                "engine_id": synthesis.engine_id,
                "voice_id": synthesis.voice_id,
            },
        )
        playback = self.audio.play(synthesis.audio, timeline, event_sink=playback_sink, cancel_event=cancel_token)
        total_ms = (time.perf_counter() - started) * 1000.0
        status = {
            "completed": "spoken",
            "interrupted": "skipped",
            "failed": "failed",
        }[playback.status]
        error_code = playback.error.code if playback.error else (
            "playback_cancelled" if playback.status == "interrupted" else ""
        )
        spoken_characters = round(len(spoken_text) * playback.spoken_fraction)
        self.last_trace = VoicePipelineTrace(
            status="failed" if status == "failed" else playback.status,
            phase="playback",
            error_code=error_code,
            input_characters=len(visible_text),
            spoken_characters=spoken_characters,
            synthesis_ms=synthesis.synthesis_ms,
            processing_ms=max(0.0, total_ms - synthesis.synthesis_ms - playback.playback_ms),
            playback_ms=playback.playback_ms,
            first_audio_ms=synthesis.first_audio_ms,
            sample_rate=self.audio.sample_rate,
            channels=self.audio.channels,
            frame_count=synthesis.audio.frame_count,
            cancelled=playback.status == "interrupted",
            engine_id=synthesis.engine_id,
            voice_id=synthesis.voice_id,
            identity_id=self.identity.identity_id,
            spoken_text=spoken_text[:spoken_characters],
        )
        return self._result(
            speech,
            intention_id,
            status,
            error_code,
            synthesis.engine_id,
            render_plan=asdict(direction),
        )

    def _speak_native_stream_locked(
        self,
        speech,
        intention_id,
        visible_text,
        spoken_text,
        direction,
        request,
        started,
    ) -> VoiceOutputResult:
        cancel_token = self._cancelled
        emit = self._turn_emitter()
        playback_sink = self._playback_sink()
        results: queue.Queue[object] = queue.Queue(maxsize=3)
        producer_finished = threading.Event()
        synthesis_elapsed_ms = 0.0
        synthesis_error: list[EngineError] = []
        generated_blocks: list[QueuedAudioBlock] = []
        discarded_blocks: list[QueuedAudioBlock] = []
        words = spoken_text.split()

        def discard(block: QueuedAudioBlock, reason: str) -> None:
            discarded_blocks.append(block)
            emit(
                "segment_discarded",
                {
                    "segment_index": block.index,
                    "text": block.text,
                    "frame_count": block.audio.frame_count,
                    "chunk_bytes": len(block.audio.pcm_s16le),
                    "discard_reason": reason,
                },
            )

        def put(block: QueuedAudioBlock) -> bool:
            while not cancel_token.is_set():
                try:
                    results.put(block, timeout=0.1)
                    return True
                except queue.Full:
                    continue
            discard(block, "user_interrupt")
            return False

        def produce() -> None:
            nonlocal synthesis_elapsed_ms
            cursor = 0
            stream = None
            try:
                stream = self.engine.synthesize_stream(request)
                for index, synthesis in enumerate(stream):
                    if cancel_token.is_set():
                        self.engine.cancel()
                        break
                    synthesis_elapsed_ms = max(synthesis_elapsed_ms, synthesis.synthesis_ms)
                    if synthesis.status != "ready" or synthesis.audio is None:
                        if synthesis.error is not None:
                            synthesis_error.append(synthesis.error)
                        break
                    word_count = max(1, round(synthesis.audio.duration_seconds * 2.7))
                    chunk_words = words[cursor : cursor + word_count]
                    cursor += len(chunk_words)
                    chunk_text = " ".join(chunk_words)
                    timeline = self.lip_sync.build(
                        chunk_text,
                        synthesis.audio,
                        synthesis.alignment,
                    )
                    block = QueuedAudioBlock(
                        index,
                        synthesis.audio,
                        timeline,
                        chunk_text,
                        "native",
                    )
                    generated_blocks.append(block)
                    emit(
                        "segment_audio_ready",
                        {
                            "segment_index": index,
                            "text": chunk_text,
                            "duration_ms": synthesis.audio.duration_seconds * 1000.0,
                            "synthesis_ms": synthesis.synthesis_ms,
                            "frame_count": synthesis.audio.frame_count,
                            "chunk_bytes": len(synthesis.audio.pcm_s16le),
                            "queue_depth_before_put": results.qsize(),
                            "engine_metadata": dict(synthesis.metadata),
                            "native_stream": True,
                        },
                    )
                    if not put(block):
                        break
                    emit(
                        "segment_enqueued",
                        {
                            "segment_index": index,
                            "queue_depth_after_put": results.qsize(),
                        },
                    )
            except Exception as exc:
                synthesis_error.append(
                    EngineError(
                        f"native_stream_{type(exc).__name__}",
                        "synthesis",
                        self.engine.engine_id,
                        recoverable=True,
                    )
                )
            finally:
                close_stream = getattr(stream, "close", None)
                if callable(close_stream):
                    close_stream()
                producer_finished.set()
                emit(
                    "stream_producer_finished",
                    {
                        "generated_segment_count": len(generated_blocks),
                        "generated_frame_count": sum(
                            block.audio.frame_count for block in generated_blocks
                        ),
                        "queue_depth": results.qsize(),
                        "cancelled": cancel_token.is_set(),
                    },
                )

        def blocks():
            while True:
                try:
                    item = results.get(timeout=0.1)
                except queue.Empty:
                    if cancel_token.is_set():
                        return
                    if producer_finished.is_set():
                        return
                    continue
                emit(
                    "segment_dequeued",
                    {
                        "segment_index": getattr(item, "index", -1),
                        "queue_depth_after_get": results.qsize(),
                    },
                )
                yield item

        emit("synthesis_started", {"engine_id": self.engine.engine_id, "native_stream": True})
        producer = threading.Thread(target=produce, name="anydes-native-voice-stream", daemon=True)
        producer.start()
        playback = self.audio.play_sequence(blocks(), event_sink=playback_sink, cancel_event=cancel_token)
        producer.join(timeout=2.0)
        if cancel_token.is_set():
            while True:
                try:
                    pending = results.get_nowait()
                except queue.Empty:
                    break
                if isinstance(pending, QueuedAudioBlock):
                    discard(pending, "user_interrupt")
        total_ms = (time.perf_counter() - started) * 1000.0
        cancelled = playback.status == "interrupted" or cancel_token.is_set()
        generated_frames = sum(block.audio.frame_count for block in generated_blocks)
        failure = synthesis_error[0] if synthesis_error else playback.error
        if not cancelled and failure is None and not generated_blocks:
            failure = EngineError(
                "native_stream_empty",
                "streaming",
                self.engine.engine_id,
                recoverable=True,
            )
        if failure is not None and not cancelled:
            emit("voice_failed", {"error_code": failure.code})
        status = "skipped" if cancelled else (
            "failed" if failure is not None or playback.status == "failed" else "spoken"
        )
        error_code = "playback_cancelled" if cancelled else (
            failure.code if failure is not None else ""
        )
        first_clock = float(playback.metadata.get("first_playback_clock", 0.0))
        first_audio_ms = max(0.0, (first_clock - started) * 1000.0) if first_clock else 0.0
        played_indices = list(playback.metadata.get("played_segment_indices") or [])
        expected_indices = [block.index for block in generated_blocks]
        stream_integrity_ok = bool(
            cancelled
            or (
                producer_finished.is_set()
                and not discarded_blocks
                and playback.total_frames == generated_frames
                and int(playback.metadata.get("segment_count", 0)) == len(generated_blocks)
                and (not played_indices or played_indices == expected_indices)
            )
        )
        if not cancelled and not stream_integrity_ok:
            failure = EngineError(
                "native_stream_integrity_mismatch",
                "streaming",
                self.engine.engine_id,
                recoverable=True,
            )
            status = "failed"
            error_code = failure.code
            emit(
                "voice_failed",
                {
                    "error_code": failure.code,
                    "generated_frames": generated_frames,
                    "played_frames": playback.total_frames,
                    "generated_segments": len(generated_blocks),
                    "played_segments": int(playback.metadata.get("segment_count", 0)),
                },
            )
        self.last_trace = VoicePipelineTrace(
            status="failed" if status == "failed" else playback.status,
            phase="native_stream_playback",
            error_code=error_code,
            input_characters=len(visible_text),
            spoken_characters=round(len(spoken_text) * playback.spoken_fraction),
            synthesis_ms=synthesis_elapsed_ms,
            processing_ms=max(0.0, total_ms - synthesis_elapsed_ms - playback.playback_ms),
            playback_ms=playback.playback_ms,
            first_audio_ms=first_audio_ms,
            sample_rate=self.audio.sample_rate,
            channels=self.audio.channels,
            frame_count=playback.total_frames,
            cancelled=cancelled,
            output_mode="native_pcm_stream",
            engine_id=self.engine.engine_id,
            voice_id=self.engine.voice_id,
            identity_id=self.identity.identity_id,
            spoken_text=spoken_text[: round(len(spoken_text) * playback.spoken_fraction)],
            segment_count=int(playback.metadata.get("segment_count", 0)),
            max_gap_ms=float(playback.metadata.get("max_gap_ms", 0.0)),
            queue_underruns=int(playback.metadata.get("queue_underruns", 0)),
            generated_segment_count=len(generated_blocks),
            generated_frame_count=generated_frames,
            discarded_segment_count=len(discarded_blocks),
            discarded_frame_count=sum(
                block.audio.frame_count for block in discarded_blocks
            ),
            stream_integrity_ok=stream_integrity_ok,
        )
        return self._result(
            speech,
            intention_id,
            status,
            error_code,
            self.engine.engine_id,
            render_plan={**asdict(direction), "delivery_mode": "native_stream"},
        )

    def _speak_segmented_locked(
        self,
        speech,
        intention_id,
        visible_text,
        spoken_text,
        direction,
        segments,
        started,
    ) -> VoiceOutputResult:
        cancel_token = self._cancelled
        emit = self._turn_emitter()
        playback_sink = self._playback_sink()
        results: queue.Queue[QueuedAudioBlock] = queue.Queue(maxsize=2)
        producer_finished = threading.Event()
        synthesis_total_ms = 0.0
        synthesis_error: list[EngineError] = []
        generated_blocks: list[QueuedAudioBlock] = []
        discarded_blocks: list[QueuedAudioBlock] = []

        def discard(block: QueuedAudioBlock, reason: str) -> None:
            discarded_blocks.append(block)
            emit(
                "segment_discarded",
                {
                    "segment_index": block.index,
                    "text": block.text,
                    "frame_count": block.audio.frame_count,
                    "chunk_bytes": len(block.audio.pcm_s16le),
                    "discard_reason": reason,
                },
            )

        def put(block: QueuedAudioBlock) -> bool:
            while not cancel_token.is_set():
                try:
                    results.put(block, timeout=0.1)
                    return True
                except queue.Full:
                    continue
            discard(block, "user_interrupt")
            return False

        def produce() -> None:
            nonlocal synthesis_total_ms
            try:
                for segment in segments:
                    if cancel_token.is_set():
                        break
                    emit(
                        "segment_synthesis_started",
                        {"segment_index": segment.index, "word_count": segment.word_count},
                    )
                    request = VoiceSynthesisRequest(
                        spoken_text=segment.text,
                        direction=direction,
                        identity_id=self.identity.identity_id,
                        language=self.identity.language,
                        locale=self.identity.locale,
                        reference_audio_path=self.identity.reference_audio_path,
                    )
                    synthesis = self._synthesize(request)
                    synthesis_total_ms += synthesis.synthesis_ms
                    if synthesis.status != "ready" or synthesis.audio is None:
                        if synthesis.error is not None:
                            synthesis_error.append(synthesis.error)
                        break
                    timeline = self.lip_sync.build(
                        segment.text,
                        synthesis.audio,
                        synthesis.alignment,
                    )
                    block = QueuedAudioBlock(
                        segment.index,
                        synthesis.audio,
                        timeline,
                        segment.text,
                        segment.boundary,
                    )
                    generated_blocks.append(block)
                    emit(
                        "segment_audio_ready",
                        {
                            "segment_index": segment.index,
                            "duration_ms": synthesis.audio.duration_seconds * 1000.0,
                            "synthesis_ms": synthesis.synthesis_ms,
                            "frame_count": synthesis.audio.frame_count,
                            "chunk_bytes": len(synthesis.audio.pcm_s16le),
                            "queue_depth_before_put": results.qsize(),
                        },
                    )
                    if not put(block):
                        break
                    emit(
                        "segment_enqueued",
                        {
                            "segment_index": segment.index,
                            "queue_depth_after_put": results.qsize(),
                        },
                    )
            except Exception as exc:
                synthesis_error.append(
                    EngineError(
                        f"segmented_synthesis_{type(exc).__name__}",
                        "synthesis",
                        self.engine.engine_id,
                        recoverable=True,
                    )
                )
            finally:
                producer_finished.set()
                emit(
                    "stream_producer_finished",
                    {
                        "generated_segment_count": len(generated_blocks),
                        "generated_frame_count": sum(
                            block.audio.frame_count for block in generated_blocks
                        ),
                        "queue_depth": results.qsize(),
                        "cancelled": cancel_token.is_set(),
                    },
                )

        def blocks():
            while True:
                try:
                    item = results.get(timeout=0.1)
                except queue.Empty:
                    if cancel_token.is_set():
                        return
                    if producer_finished.is_set():
                        return
                    continue
                emit(
                    "segment_dequeued",
                    {
                        "segment_index": item.index,
                        "queue_depth_after_get": results.qsize(),
                    },
                )
                yield item

        producer = threading.Thread(target=produce, name="anydes-voice-producer", daemon=True)
        producer.start()
        playback = self.audio.play_sequence(blocks(), event_sink=playback_sink, cancel_event=cancel_token)
        producer.join(timeout=2.0)
        if cancel_token.is_set():
            while True:
                try:
                    pending = results.get_nowait()
                except queue.Empty:
                    break
                discard(pending, "user_interrupt")
        total_ms = (time.perf_counter() - started) * 1000.0
        cancelled = playback.status == "interrupted" or cancel_token.is_set()
        generated_frames = sum(block.audio.frame_count for block in generated_blocks)
        failure = synthesis_error[0] if synthesis_error else playback.error
        if not cancelled and failure is None and not generated_blocks:
            failure = EngineError(
                "segmented_stream_empty",
                "streaming",
                self.engine.engine_id,
                recoverable=True,
            )
        if failure is not None and not cancelled:
            emit("voice_failed", {"error_code": failure.code})
        status = "skipped" if cancelled else (
            "failed" if failure is not None or playback.status == "failed" else "spoken"
        )
        error_code = (
            "playback_cancelled" if cancelled else (failure.code if failure is not None else "")
        )
        segment_count = int(playback.metadata.get("segment_count", 0))
        max_gap_ms = float(playback.metadata.get("max_gap_ms", 0.0))
        queue_underruns = int(playback.metadata.get("queue_underruns", 0))
        first_clock = float(playback.metadata.get("first_playback_clock", 0.0))
        first_audio_ms = max(0.0, (first_clock - started) * 1000.0) if first_clock else 0.0
        played_indices = list(playback.metadata.get("played_segment_indices") or [])
        expected_indices = [block.index for block in generated_blocks]
        stream_integrity_ok = bool(
            cancelled
            or (
                producer_finished.is_set()
                and not discarded_blocks
                and playback.total_frames == generated_frames
                and segment_count == len(generated_blocks)
                and (not played_indices or played_indices == expected_indices)
            )
        )
        if not cancelled and not stream_integrity_ok:
            failure = EngineError(
                "segmented_stream_integrity_mismatch",
                "streaming",
                self.engine.engine_id,
                recoverable=True,
            )
            status = "failed"
            error_code = failure.code
            emit(
                "voice_failed",
                {
                    "error_code": failure.code,
                    "generated_frames": generated_frames,
                    "played_frames": playback.total_frames,
                    "generated_segments": len(generated_blocks),
                    "played_segments": segment_count,
                },
            )
        self.last_trace = VoicePipelineTrace(
            status="failed" if status == "failed" else playback.status,
            phase="segmented_playback",
            error_code=error_code,
            input_characters=len(visible_text),
            spoken_characters=round(len(spoken_text) * playback.spoken_fraction),
            synthesis_ms=synthesis_total_ms,
            processing_ms=max(0.0, total_ms - synthesis_total_ms - playback.playback_ms),
            playback_ms=playback.playback_ms,
            first_audio_ms=first_audio_ms,
            sample_rate=self.audio.sample_rate,
            channels=self.audio.channels,
            frame_count=playback.total_frames,
            cancelled=cancelled,
            output_mode="prosodic_queue",
            engine_id=self.engine.engine_id,
            voice_id=self.engine.voice_id,
            identity_id=self.identity.identity_id,
            spoken_text=spoken_text[: round(len(spoken_text) * playback.spoken_fraction)],
            segment_count=segment_count,
            max_gap_ms=max_gap_ms,
            queue_underruns=queue_underruns,
            generated_segment_count=len(generated_blocks),
            generated_frame_count=generated_frames,
            discarded_segment_count=len(discarded_blocks),
            discarded_frame_count=sum(
                block.audio.frame_count for block in discarded_blocks
            ),
            stream_integrity_ok=stream_integrity_ok,
        )
        return self._result(
            speech,
            intention_id,
            status,
            error_code,
            self.engine.engine_id,
            render_plan={**asdict(direction), "delivery_mode": "prosodic"},
        )

    def _synthesize(self, request: VoiceSynthesisRequest) -> SynthesisResult:
        if self.phrase_cache is not None:
            cached = self.phrase_cache.lookup(
                identity_id=request.identity_id,
                spoken_text=request.spoken_text,
            )
            if cached is not None:
                return SynthesisResult(
                    status="ready",
                    engine_id=self.engine.engine_id,
                    voice_id=self.engine.voice_id,
                    identity_id=request.identity_id,
                    audio=cached,
                    synthesis_ms=0.0,
                    first_audio_ms=0.0,
                    metadata={
                        "common_phrase_cache_hit": True,
                        "voice_version": "Anydes Canonical Voice v1",
                    },
                )
        return self.engine.synthesize(request)

    def cancel(self) -> None:
        with self._state_lock:
            self._generation += 1
            self._cancelled.set()
            self.audio.cancel()
            for engine in (self.engine, self.emergency_engine):
                if engine is None:
                    continue
                try:
                    engine.cancel()
                except Exception:
                    pass
            self._emit("playback_interrupted", {"reason": "user_interruption", "voice_generation": self._generation})

    def close(self) -> None:
        self._closed = True
        self.cancel_idle_gpu_cache_release()
        self.cancel()
        for engine in (self.engine, self.emergency_engine):
            if engine is not None:
                try:
                    engine.close()
                except Exception:
                    pass

    def warmup(self) -> bool:
        warmup = getattr(self.engine, "warmup", None)
        if not callable(warmup):
            return True
        try:
            return bool(warmup())
        except Exception:
            return False

    def warmup_async(self) -> None:
        """Prepare an opt-in engine without blocking the application UI."""
        if self._warmup_started:
            return
        self._warmup_started = True

        def prepare() -> None:
            if not self.warmup():
                self._emit(
                    "voice_warmup_failed",
                    {"engine_id": self.engine.engine_id},
                )

        self._warmup_thread = threading.Thread(
            target=prepare,
            name="anydes-voice-warmup",
            daemon=True,
        )
        self._warmup_thread.start()

    def poll_events(self, limit: int = 32) -> tuple[VoiceEvent, ...]:
        visual_events = {"playback_started", "playback_stopped", "playback_interrupted", "voice_failed", "emergency_voice_started"}
        events: list[VoiceEvent] = []
        for _ in range(max(0, limit)):
            try:
                event = self._events.get_nowait()
                event_generation = event.payload.get(
                    "generation_id",
                    event.payload.get("voice_generation"),
                )
                if event_generation is not None and event_generation != self.generation:
                    self.stale_events_discarded += 1
                    continue
                events.append(event)
            except queue.Empty:
                break
        delivered = self._playback_event_clock.deliver(events)
        current: list[VoiceEvent] = []
        for event in delivered:
            event_generation = event.payload.get(
                "generation_id",
                event.payload.get("voice_generation"),
            )
            if event_generation is not None and event_generation != self.generation:
                self.stale_events_discarded += 1
                continue
            current.append(event)
        return tuple(current)

    def _turn_emitter(self):
        generation = self._active_generation
        return lambda kind, payload: self._emit(kind, {**payload, "voice_generation": generation, "generation_id": generation})

    def _playback_sink(self):
        generation = self._active_generation
        def receive(event):
            if event.kind == "playback_started" and generation != self._generation:
                self.stale_events_discarded += 1
                return
            self._events.put(VoiceEvent(event.kind, event.created_monotonic, {**event.payload, "voice_generation": generation, "generation_id": generation}))
        return receive

    def _emit(self, kind: str, payload: dict[str, object]) -> None:
        self._events.put(VoiceEvent(kind=kind, created_monotonic=time.perf_counter(), payload=payload))

    def _result(
        self,
        speech: SpeechIntention,
        intention_id: str,
        status: str,
        error: str = "",
        provider: str = "",
        *,
        render_plan: dict[str, object] | None = None,
    ) -> VoiceOutputResult:
        return VoiceOutputResult(
            intention_id=intention_id,
            status=status,
            text=speech.text,
            provider=provider or self.engine.engine_id,
            error=error or None,
            speech=speech,
            render_plan=render_plan or {},
        )
