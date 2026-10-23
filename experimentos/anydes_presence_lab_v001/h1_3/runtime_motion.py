"""One staging observer boundary for audio and typed work modifiers."""
from contextlib import contextmanager
from functools import wraps
import threading
from event_service import PresenceService
from runtime_observer import RuntimeObserver
from state_adapter import Domain
from audio_tap import AudioTap

MODES = {'NONE','PROGRAMMING','CREATING_FILE','EDITING','EXECUTING','VERIFYING','CONVERTING'}
CAP_MODES = {'developer.write_code':'PROGRAMMING','developer.run_python':'EXECUTING',
             'developer.run_cpp':'EXECUTING','developer.test':'VERIFYING',
             'document.edit_document':'EDITING','filesystem.write_text':'CREATING_FILE',
             'document.create_txt':'CREATING_FILE','document.create_markdown':'CREATING_FILE',
             'document.create_pdf':'CONVERTING','document.create_docx':'CREATING_FILE'}


class VoiceMotionService(PresenceService):
    def __init__(self, evidence, port=8781):
        self.tap = AudioTap()
        self.generation = None
        self.modes = {}
        self.motion_sequence = 0
        self.motion_frames = self.voiced_frames = 0
        super().__init__(evidence, port, motion_provider=self.motion)

    def motion(self, state):
        self.motion_sequence += 1
        rms, peak, active = self.tap.snapshot(self.generation)
        if not state['connected'] or state['state'] != 'SPEAKING': rms,peak,active=0.0,0.0,False
        mode = 'NONE'
        if state['state'] == 'WORKING':
            verifying = state['source']=='tools.verification' or any(e.domain==Domain.VERIFY for e in tuple(self.adapter.active.values()))
            mode = 'VERIFYING' if verifying else next(reversed(tuple(self.modes.values())), 'NONE')
        self.motion_frames += 1
        self.voiced_frames += bool(active)
        return dict(schema='anydes.presence.motion.v1',session=state['session'],sequence=self.motion_sequence,
                    timestamp=state['timestamp'],working_mode=mode,rms=round(rms,5),peak=round(peak,5),voice_active=active)

    def close(self):
        super().close()
        self.tap.close()
        costs=sorted(self.tap.offer_cost_us)
        self._log('audio_observer_shutdown',dict(alive=self.tap.thread.is_alive(),errors=self.tap.errors,
            dropped=self.tap.dropped,packets=self.tap.packets,frames=self.motion_frames,voiced=self.voiced_frames,
            dac_callback_modified=False, offer_cost_median_us=costs[len(costs)//2] if costs else 0,
            offer_cost_max_us=max(costs,default=0), output_diagnostics=list(self.tap.output_diagnostics)))


class MotionObserver(RuntimeObserver):
    def playback(self, events, current_generation):
        self.service.generation = current_generation
        super().playback(events,current_generation)

    @contextmanager
    def work(self, mode):
        key = ('work',threading.get_ident(),object())
        self.service.modes[key] = mode
        self.safe(self.emit,Domain.EXECUTION,'START',key)
        try: yield
        finally:
            self.service.modes.pop(key,None)
            self.safe(self.emit,Domain.EXECUTION,'END',key)

    def install(self):
        super().install()
        self.service.tap.install()
        from anydes_tools.conversation.controller import ToolConversationController
        observer = self
        def wrap(name, select):
            original = getattr(ToolConversationController,name)
            @wraps(original)
            def call(controller,*args,**kwargs):
                mode = select(args)
                if mode == 'NONE': return original(controller,*args,**kwargs)
                with observer.work(mode): return original(controller,*args,**kwargs)
            setattr(ToolConversationController,name,call)
        wrap('_execute_capability',lambda args:CAP_MODES.get(args[0],'NONE'))
        wrap('_execute_general_code',lambda args:'PROGRAMMING')
        wrap('_execute_general_code_modification',lambda args:'EDITING')
        wrap('_execute_generated_create',lambda args:'CREATING_FILE')
        wrap('_execute_document_conversion',lambda args:'CONVERTING')
