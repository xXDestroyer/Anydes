"""Opt-in adapters using the SAME workers, turns, memory and TTS as keyboard.

The integration uses process-local subclasses so shutdown restores a clean runtime.
Voice is deliberately not upgraded to the keyboard owner's trust level.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import os

from .core import InputText,UserInputEvent
from .ui import VoicePanel
import time
from .telemetry import mark


def playback_tracking_key(payload):
    generation=payload.get('generation_id',payload.get('voice_generation'))
    playback_id=payload.get('playback_id')
    return ('generation',generation) if generation is not None else ('playback',playback_id)


class InputAssistant:
    def __init__(self,assistant,event):
        self.assistant=assistant;self.event=event

    def __getattr__(self,name):
        return getattr(self.assistant,name)

    def _respond(self,method,text,**kwargs):
        with self.assistant._interaction_lock:
            if self.event.source=='voice':
                # Never inherit the last keyboard user's verified identity.
                self.assistant.set_current_speaker(None)
                kwargs['input_mode']='voice'
            else:kwargs['input_mode']='typed'
            self.assistant.audit.record('user_input_received',source=self.event.source,
                audio_id=self.event.audio_id,text_characters=len(text),monotonic=time.monotonic())
            if callable(kwargs.get('on_text')):
                publish=kwargs['on_text'];first=True
                def on_text(partial):
                    nonlocal first
                    if first:
                        self.assistant.audit.record('voice_input_first_text',audio_id=self.event.audio_id,
                            monotonic=time.monotonic())
                        first=False
                    return publish(partial)
                kwargs['on_text']=on_text
            mark('reasoning_started',source=self.event.source,audio_id=self.event.audio_id)
            try:
                return getattr(self.assistant,method)(str(text),**kwargs)
            finally:
                mark('reasoning_finished',source=self.event.source,audio_id=self.event.audio_id)

    def respond(self,text,**kwargs):return self._respond('respond',text,**kwargs)
    def respond_realtime(self,text,**kwargs):return self._respond('respond_realtime',text,**kwargs)


def install(module,performance=False,panel_factory=None):
    panel_factory = VoicePanel if panel_factory is None else panel_factory
    Worker=module.AssistantWorker
    class InputWorker(Worker):
        def __init__(self,assistant,text):
            event=getattr(text,'event',UserInputEvent.keyboard(str(text)))
            super().__init__(InputAssistant(assistant,event),text)
    module.AssistantWorker=InputWorker
    Base=module.PerformanceCompanionWindow if performance else module.AnydesPySideWindow

    class VoiceWindow(Base):
        def __init__(self,*args,**kwargs):
            mark('mode_window_constructing')
            super().__init__(*args,**kwargs)
            self.input_panel=panel_factory(self,self.accept_voice,self.barge_in)
            self._producer_waits_by_playback={}
            self._playback_clock_corrections={}
            try:
                self._install_render_probe(performance)
            except (AttributeError, TypeError):
                # Lightweight contract tests use a non-Qt window double.
                pass
            original_poll=getattr(self.voice_manager,'poll_events',None)
            if callable(original_poll):
                def poll_observed(*args,**kwargs):
                    events=original_poll(*args,**kwargs)
                    for event in events:
                        payload=getattr(event,'payload',{})
                        source_at=getattr(event,'created_monotonic',None)
                        timing={k:payload[k] for k in ('playback_id','voice_generation','audio_clock',
                            'generation_id','clock_correction','device_latency_ms',
                            'audible_end_clock','clock_source','frame_count','played_frames','discarded_frames',
                            'submitted_frames','dac_frames','device_underflows','starvation_frames',
                            'audio_end_clock','active_audio_end_clock','first_playback_clock',
                            'discard_reason','segment_count','max_gap_ms',
                            'gap_ms','before_segment','chunk_bytes','duration_ms') if k in payload}
                        generation=payload.get('generation_id',payload.get('voice_generation'))
                        playback_id=payload.get('playback_id')
                        # Producer waits are emitted before a playback_id may
                        # exist, but generation_id is stable end-to-end.
                        playback_key=playback_tracking_key(payload)
                        is_clock_correction=(event.kind=='playback_started'
                            and bool(payload.get('clock_correction')))
                        if is_clock_correction:
                            self._playback_clock_corrections[playback_key]=(
                                self._playback_clock_corrections.get(playback_key,0)+1)
                        elif event.kind=='queue_underrun':
                            waits=self._producer_waits_by_playback.setdefault(playback_key,[])
                            waits.append(float(payload.get('gap_ms') or 0.0))
                            mark('output_producer_wait',source_at=source_at,
                                source_clock='VoiceEvent.created_monotonic / perf_counter',
                                observed_at=time.monotonic(),turn_revision=self._turns.revision,payload=timing,
                                queue_wait_start_estimate=source_at-float(payload['gap_ms'])/1000
                                if source_at is not None and 'gap_ms' in payload else None,
                                classification='pending_dac_integrity')
                        elif event.kind in {'synthesis_started','audio_ready','playback_started',
                                'playback_stopped','playback_interrupted','voice_failed'}:
                            mark('output_'+event.kind,source_at=source_at,
                                source_clock='VoiceEvent.created_monotonic / perf_counter',
                                observed_at=time.monotonic(),turn_revision=self._turns.revision,payload=timing)
                        if event.kind in {'playback_stopped','playback_interrupted'}:
                            waits=self._producer_waits_by_playback.pop(playback_key,[])
                            device_underflows=int(payload.get('device_underflows') or 0)
                            starvation_frames=int(payload.get('starvation_frames') or 0)
                            audible=bool(device_underflows or starvation_frames)
                            integrity=dict(
                                playback_id=playback_id,generation_id=generation,
                                producer_wait_count=len(waits),
                                producer_wait_max_ms=max(waits,default=0.0),
                                device_underflows=device_underflows,
                                starvation_frames=starvation_frames,
                                audible_underrun=audible,
                                classification='audible_underflow' if audible else
                                    ('buffer_absorbed_wait' if waits else 'continuous'),
                                clock_corrections=self._playback_clock_corrections.pop(playback_key,0),
                            )
                            mark('output_playback_integrity',source_at=source_at,payload=integrity)
                            self.input_panel.log(dict(kind='output_playback_integrity',
                                at=time.monotonic(),source_at=source_at,payload=integrity))
                            if audible:
                                mark('output_audible_underrun',source_at=source_at,payload=integrity)
                                self.input_panel.log(dict(kind='output_queue_underrun',
                                    at=time.monotonic(),source_at=source_at,payload=integrity))
                        event_generation=generation
                        current_generation=getattr(self.voice_manager,'generation',None)
                        if (event_generation is not None and current_generation is not None
                                and int(event_generation)!=int(current_generation)):
                            mark('output_stale_event_observed',event_kind=event.kind,
                                event_generation=event_generation,current_generation=current_generation,
                                playback_id=payload.get('playback_id'))
                        if not is_clock_correction:
                            panel_kind='output_producer_wait' if event.kind=='queue_underrun' else 'output_'+event.kind
                            self.input_panel.log(dict(kind=panel_kind,at=time.monotonic(),
                                source_at=source_at,turn_revision=self._turns.revision,payload=timing))
                    return events
                self.voice_manager.poll_events=poll_observed
            if performance:
                button=self.chat_window.controls[1]
                button.clicked.disconnect()
                button.clicked.connect(self.input_panel.toggle_capture)
            self.input_panel.show()
            mark('mode_window_ready')

        def _install_render_probe(self,performance):
            from PySide6.QtCore import QTimer, Qt
            self._render_probe_timer=QTimer(self)
            if performance:
                self._mini_probe_frames=0;self._mini_probe_started=time.perf_counter()
                def tick():
                    self._mini_probe_frames+=1
                    elapsed=time.perf_counter()-self._mini_probe_started
                    if elapsed>=2.0:
                        fps=self._mini_probe_frames/elapsed
                        mark('avatar_metrics',renderer='mini_2d',fps=fps,
                            frame_ms=1000.0/max(fps,.001),visual_glitches=0,
                            metric_source='QTimer delivery cadence')
                        self._mini_probe_frames=0;self._mini_probe_started=time.perf_counter()
                avatar_timer=getattr(self.avatar,'_timer',None)
                if avatar_timer is not None:
                    # Windows' coarse timer can quantize 33 ms to ~46.9 ms
                    # (about 21 FPS).  PreciseTimer preserves the intended
                    # 30 Hz cadence without changing Mini's artwork or motion.
                    was_active=avatar_timer.isActive()
                    if was_active:avatar_timer.stop()
                    avatar_timer.setTimerType(Qt.TimerType.PreciseTimer)
                    if was_active:avatar_timer.start()
                    avatar_timer.timeout.connect(tick)
                    mark('avatar_timer_configured',renderer='mini_2d',
                        interval_ms=avatar_timer.interval(),timer_type='PreciseTimer')
                return
            self._render_probe_signature=None
            self._render_probe_ticks=0

            def compact(value,limit=8):
                if isinstance(value,dict):
                    return {str(key):compact(item,limit) for key,item in list(value.items())[:limit]}
                if isinstance(value,(list,tuple)):
                    return [compact(item,limit) for item in list(value)[-limit:]]
                if isinstance(value,(str,int,float,bool)) or value is None:return value
                return repr(value)[:240]

            def poll():
                self._render_probe_ticks+=1
                view=getattr(getattr(self,'avatar',None),'_vrm_view',None)
                if view is None:
                    signature=('missing',os.environ.get('ANYDES_AVATAR_PRESENTATION',''))
                    if signature!=self._render_probe_signature:
                        self._render_probe_signature=signature
                        mark('avatar_probe',renderer='fallback_pixmap',view_present=False,
                            presentation=os.environ.get('ANYDES_AVATAR_PRESENTATION',''),
                            reason='avatar_stage_has_no_webengine_view')
                    return
                if not getattr(view,'render_ready',False):
                    diagnostic=compact(getattr(view,'diagnostic_state',{}),12)
                    messages=compact(getattr(view,'diagnostic_messages',()),8)
                    signature=(
                        type(view).__name__,
                        bool(getattr(view,'page_loaded',False)),
                        str(getattr(view,'render_error','')),
                        repr(diagnostic)[:800],
                        repr(messages)[:800],
                    )
                    if signature!=self._render_probe_signature or self._render_probe_ticks%5==0:
                        self._render_probe_signature=signature
                        mark('avatar_probe',renderer='2.5d_hq',view_present=True,
                            view_class=type(view).__name__,render_ready=False,
                            page_loaded=getattr(view,'page_loaded',None),
                            render_error=getattr(view,'render_error',''),
                            diagnostic=diagnostic,console_messages=messages)
                    return
                ready_signature=(type(view).__name__,'ready')
                if ready_signature!=self._render_probe_signature:
                    self._render_probe_signature=ready_signature
                    mark('avatar_probe',renderer='2.5d_hq',view_present=True,
                        view_class=type(view).__name__,render_ready=True,
                        page_loaded=getattr(view,'page_loaded',None),
                        diagnostic=compact(getattr(view,'diagnostic_state',{}),12),
                        console_messages=compact(getattr(view,'diagnostic_messages',()),8))
                def receive(value):
                    try:
                        status=json.loads(value) if isinstance(value,str) else value
                        if not isinstance(status,dict):return
                        mark('avatar_metrics',renderer='2.5d_hq',
                            fps=status.get('fps'),frame_ms=status.get('frame_ms'),
                            draw_calls=status.get('draw_calls'),textures=status.get('textures'),
                            geometries=status.get('geometries'),speaking=status.get('speaking'),
                            state=status.get('state'),visual_stability=status.get('visual_stability'),
                            pixel_ratio=status.get('pixel_ratio'),
                            drawing_buffer=status.get('drawing_buffer'),
                            gaze_frame_count=status.get('gaze_frame_count'),
                            gaze_source_frame_count=status.get('gaze_source_frame_count'),
                            gaze_cache_limit=status.get('gaze_cache_limit'),
                            runtime_safety=status.get('runtime_safety'),
                            metric_source='AnyDes25DAvatar.status')
                    except Exception as exc:
                        mark('avatar_metrics_error',error=f'{type(exc).__name__}: {exc}')
                view.page().runJavaScript(
                    'JSON.stringify(window.Anydes25DAvatar?.status?.() ?? null);',receive)
            self._render_probe_timer.timeout.connect(poll)
            self._render_probe_timer.start(2000)

        def voice_requested(self):
            self.input_panel.toggle_capture()

        def _record_voice_trace(self,event_kind):
            trace=getattr(self.voice_manager,'last_trace',None)
            if trace is None:return
            trace_payload=asdict(trace) if is_dataclass(trace) else dict(
                getattr(trace,'__dict__',{}) or {})
            mark('output_voice_trace',event_kind=event_kind,
                turn_revision=self._turns.revision,trace=trace_payload)

        def _handle_voice_completed(self,*args,**kwargs):
            result=super()._handle_voice_completed(*args,**kwargs)
            self._record_voice_trace('normal_voice_worker_completed')
            return result

        def _voice_completed(self,*args,**kwargs):
            result=super()._voice_completed(*args,**kwargs)
            self._record_voice_trace('performance_voice_worker_completed')
            return result

        def accept_voice(self,event):
            if not self._closing:self.send_text(InputText(event))

        def send_text(self,text):
            if not isinstance(text,InputText):
                if hasattr(self,'input_panel'):self.input_panel.keyboard_superseded()
                text=InputText(UserInputEvent.keyboard(str(text)))
            old_turn=self._turns.revision
            old_generation=getattr(self.voice_manager,'generation',None)
            super().send_text(text)
            mark('turn_replaced',source=text.event.source,
                old_turn_id=old_turn,new_turn_id=self._turns.revision,
                old_generation_id=old_generation,
                new_generation_id=getattr(self.voice_manager,'generation',None),
                pending_turns=int(self._turns.pending is not None))

        def barge_in(self):
            if self._closing:return
            old_turn=self._turns.revision
            old_generation=getattr(self.voice_manager,'generation',None)
            self._turns.close()
            if self._worker is not None:self._worker.requestInterruption()
            if performance:
                self.voice_manager.cancel();self.avatar.stop_speaking()
                self._active_playback_id=0;self._voice_worker=None
                self._streaming=False
            else:
                self._cancel_active_voice();self._latest_voice_worker=None
                self._streaming_answer=False
            self.assistant.audit.record('voice_barge_in',turn_revision=self._turns.revision,
                voice_generation=getattr(self.voice_manager,'generation',None))
            mark('barge_in',old_turn_id=old_turn,new_turn_id=self._turns.revision,
                old_generation_id=old_generation,
                new_generation_id=getattr(self.voice_manager,'generation',None),
                playback_id=getattr(self,'_active_playback_id',None))

        def stop_all_sensors(self):
            if hasattr(self,'input_panel'):self.input_panel.stop()
            parent=getattr(super(),'stop_all_sensors',None)
            if parent:parent()

        def closeEvent(self,event):
            if not self.input_panel.shutdown():
                self.input_panel.error.setText('STT terminando. Espera y vuelve a cerrar.')
                event.ignore();return
            super().closeEvent(event)

    if performance:module.PerformanceCompanionWindow=VoiceWindow
    else:module.AnydesPySideWindow=VoiceWindow
