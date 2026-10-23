"""The only projection boundary from existing runtime events to presence."""
from functools import wraps
import threading

from state_adapter import Domain, RuntimeEvent, opaque


WORK_STAGES = {'EXECUTION_DISPATCH': Domain.EXECUTION, 'B2_EXECUTION': Domain.EXECUTION,
               'TESTSPEC_BUILD': Domain.VERIFY, 'VERIFICATION_DISPATCH': Domain.VERIFY,
               'ORACLE_SELECTION': Domain.VERIFY, 'COMMIT': Domain.COMMIT}
MODEL_STAGES = {'INTERACTION_RESOLUTION', 'SEMANTIC_PLANNING'}


class RuntimeObserver:
    def __init__(self, service):
        self.service = service
        self.generation = None
        self.playbacks = set()
        self.syntheses = set()
        self.failures = 0
        self.input_revision = None

    def emit(self, domain, phase, identity, task_id=None):
        self.service.submit(RuntimeEvent(domain, phase, opaque(identity), opaque(task_id) if task_id else None))

    def safe(self, method, *args):
        try:
            method(*args)
        except Exception:
            # Only observation is isolated. The original core call is outside this boundary.
            self.failures += 1
            self.service.submit('DISCONNECT')

    def mark(self, kind, fields):
        if kind == 'mode_window_ready': self.service.submit('READY')
        elif kind in {'reasoning_started', 'reasoning_finished'}:
            self.emit(Domain.INFERENCE, 'START' if kind == 'reasoning_started' else 'END',
                      ('reasoning', threading.get_ident()))

    def audit(self, event, fields):
        if event != 'pipeline_stage_trace': return
        stage = fields.get('stage')
        domain = WORK_STAGES.get(stage) or (Domain.INFERENCE if stage in MODEL_STAGES else None)
        if domain is None: return
        if fields.get('entered') is True: phase = 'START'
        elif fields.get('exited') is True: phase = 'END'
        else: return  # NOT_REACHED, TRANSITION and validation diagnostics are not work.
        identity = (fields.get('turn_id'), fields.get('task_id'), fields.get('plan_revision'), stage)
        self.emit(domain, phase, identity, fields.get('task_id'))

    def voice_input(self, event, revision):
        if event.get('revision') != revision: return
        if self.input_revision != revision:
            if self.input_revision is not None:
                self.emit(Domain.CAPTURE,'END',('input',self.input_revision))
                self.emit(Domain.STT,'END',('input',self.input_revision))
            self.input_revision = revision
        kind = event.get('kind')
        identity = ('input', revision)
        if kind in {'mic_ready', 'listening', 'voice_detected'}:
            self.emit(Domain.CAPTURE, 'START', identity)
        elif kind in {'transcribing', 'stt_loading', 'transcription_started'}:
            self.emit(Domain.STT, 'START', identity)
        elif kind in {'capture_closed', 'ptt_armed'}:
            # Armed pre-roll is not an active utterance; STT may still be running.
            self.emit(Domain.CAPTURE, 'END', identity)
        elif kind in {'transcript', 'hallucination_suspected', 'no_speech', 'error', 'mic_off'}:
            self.emit(Domain.CAPTURE, 'END', identity)
            self.emit(Domain.STT, 'END', identity)

    def playback(self, events, current_generation):
        if self.generation != current_generation:
            for key in self.playbacks: self.emit(Domain.PLAYBACK, 'END', key)
            for key in self.syntheses: self.emit(Domain.SYNTHESIS, 'END', key)
            self.playbacks.clear(); self.syntheses.clear()
            self.generation = current_generation
        for event in events:
            payload = event.payload
            generation = payload.get('generation_id', payload.get('voice_generation'))
            if generation is not None and generation != current_generation: continue
            key = ('voice', generation if generation is not None else payload.get('playback_id'))
            kind = event.kind
            if kind == 'synthesis_started':
                self.syntheses.add(key); self.emit(Domain.SYNTHESIS, 'START', key)
            elif kind == 'playback_started' and not payload.get('clock_correction'):
                self.playbacks.add(key); self.emit(Domain.PLAYBACK, 'START', key)
            elif kind in {'playback_stopped', 'playback_interrupted', 'voice_failed'}:
                self.playbacks.discard(key); self.syntheses.discard(key)
                self.emit(Domain.PLAYBACK, 'END', key)
                self.emit(Domain.SYNTHESIS, 'END', key)

    def install(self):
        import anydes.voice_input.telemetry as telemetry
        import anydes.voice_input.integration as integration
        from anydes.voice_input.ui import VoicePanel
        from anydes_tools.audit import ToolAuditLog
        original_mark = telemetry.mark
        @wraps(original_mark)
        def mark(kind, **fields):
            result = original_mark(kind, **fields)
            self.safe(self.mark, kind, fields)
            return result
        telemetry.mark = mark
        integration.mark = mark
        original_audit = ToolAuditLog.record
        @wraps(original_audit)
        def audit(log, event, **fields):
            result = original_audit(log, event, **fields)
            self.safe(self.audit, event, fields)
            return result
        ToolAuditLog.record = audit
        original_event = VoicePanel.on_event
        @wraps(original_event)
        def on_event(panel, event):
            if not panel.shutting_down:
                self.safe(self.voice_input, event, panel.engine.revision)
            return original_event(panel, event)
        VoicePanel.on_event = on_event

    def observe_window(self, window):
        manager = window.voice_manager
        original_poll = manager.poll_events
        @wraps(original_poll)
        def poll(*args, **kwargs):
            events = original_poll(*args, **kwargs)
            self.safe(self.playback, events, getattr(manager, 'generation', None))
            return events
        manager.poll_events = poll
