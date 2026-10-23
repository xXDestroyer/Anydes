import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from state_adapter import PresenceStateAdapter, RuntimeEvent, Domain, opaque
from runtime_observer import RuntimeObserver
from event_service import PresenceService


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.adapter = PresenceStateAdapter(clock=lambda:self.now, session=opaque('test'))
        self.adapter.ready()
    def event(self, domain, phase='START', key='one'):
        return RuntimeEvent(domain,phase,opaque(key),opaque('task'))
    def test_idle_without_activity(self):
        self.assertEqual(self.adapter.snapshot()['state'],'IDLE')
    def test_mic_and_stt(self):
        self.assertEqual(self.adapter.apply(self.event(Domain.CAPTURE))['state'],'LISTENING')
        self.adapter.apply(self.event(Domain.STT))
        self.adapter.apply(self.event(Domain.CAPTURE,'END'))
        self.assertEqual(self.adapter.snapshot()['state'],'LISTENING')
    def test_inference_is_thinking(self):
        self.assertEqual(self.adapter.apply(self.event(Domain.INFERENCE))['state'],'THINKING')
    def test_playback_start_and_end(self):
        self.assertEqual(self.adapter.apply(self.event(Domain.PLAYBACK))['state'],'SPEAKING')
        self.adapter.apply(self.event(Domain.PLAYBACK,'END')); self.now+=0.121
        self.assertEqual(self.adapter.advance()['state'],'IDLE')
    def test_tool_operation_and_verification_are_working(self):
        for domain in (Domain.EXECUTION,Domain.VERIFY,Domain.COMMIT):
            adapter=PresenceStateAdapter(); adapter.ready()
            self.assertEqual(adapter.apply(self.event(domain))['state'],'WORKING')
    def test_nested_operation_does_not_go_idle_when_child_finishes(self):
        self.adapter.apply(self.event(Domain.EXECUTION,key='parent'))
        self.adapter.apply(self.event(Domain.VERIFY,key='child'))
        self.adapter.apply(self.event(Domain.VERIFY,'END',key='child'))
        self.now+=1; self.adapter.advance()
        self.assertEqual(self.adapter.snapshot()['state'],'WORKING')
    def test_rapid_model_to_playback_has_no_idle_flash(self):
        self.adapter.apply(self.event(Domain.INFERENCE))
        self.assertIsNone(self.adapter.apply(self.event(Domain.INFERENCE,'END')))
        self.now+=0.03; self.assertIsNone(self.adapter.advance())
        self.assertEqual(self.adapter.apply(self.event(Domain.PLAYBACK))['state'],'SPEAKING')
    def test_real_gap_becomes_idle_within_120ms(self):
        self.adapter.apply(self.event(Domain.INFERENCE))
        self.adapter.apply(self.event(Domain.INFERENCE,'END'))
        self.now+=0.121; self.assertEqual(self.adapter.advance()['state'],'IDLE')
    def test_playback_has_precedence_and_restores_actual_work(self):
        self.adapter.apply(self.event(Domain.EXECUTION))
        self.adapter.apply(self.event(Domain.PLAYBACK))
        self.assertEqual(self.adapter.snapshot()['state'],'SPEAKING')
        self.assertEqual(self.adapter.apply(self.event(Domain.PLAYBACK,'END'))['state'],'WORKING')
    def test_disconnect_clears_all_activity(self):
        self.adapter.apply(self.event(Domain.PLAYBACK))
        self.assertFalse(self.adapter.disconnect()['connected'])
        self.assertEqual(self.adapter.snapshot()['state'],'IDLE')
        self.assertIsNone(self.adapter.apply(self.event(Domain.EXECUTION)))
    def test_payload_is_small_and_has_no_free_text(self):
        self.adapter.apply(self.event(Domain.EXECUTION))
        frame=self.adapter.snapshot()
        self.assertEqual(set(frame),{'schema','session','sequence','state','source','task_id','timestamp','connected'})
        self.assertLess(len(json.dumps(frame)),400)
        with self.assertRaises(ValueError): RuntimeEvent(Domain.INFERENCE,'START','my secret prompt')


class ObserverTests(unittest.TestCase):
    def setUp(self):
        self.events=[]
        self.observer=RuntimeObserver(SimpleNamespace(submit=self.events.append))
    def test_not_reached_is_not_work(self):
        self.observer.audit('pipeline_stage_trace',dict(stage='B2_EXECUTION',status='NOT_REACHED',entered=False,exited=False))
        self.assertEqual(self.events,[])
    def test_stage_projection_discards_sensitive_fields(self):
        self.observer.audit('pipeline_stage_trace',dict(stage='VERIFICATION_DISPATCH',entered=True,
            task_id='secret filename',turn_id='turn',prompt='private prompt',memory='private memory',api_key='secret'))
        self.assertEqual(self.events[0].domain,Domain.VERIFY)
        self.assertNotIn('secret',repr(self.events)); self.assertNotIn('private',repr(self.events))
    def test_stale_input_is_ignored_and_no_text_is_examined(self):
        self.observer.voice_input(dict(kind='transcript',revision=1,input={'text':'private'}),2)
        self.assertEqual(self.events,[])
        self.observer.voice_input(dict(kind='transcribing',revision=2,input={'text':'irrelevant'}),2)
        self.assertEqual(self.events[0].domain,Domain.STT)
    def test_capture_closed_does_not_cancel_inflight_stt(self):
        self.observer.voice_input(dict(kind='capture_closed',revision=1),1)
        self.assertEqual([e.domain for e in self.events],[Domain.CAPTURE])
    def test_new_revision_clears_obsolete_capture(self):
        self.observer.voice_input(dict(kind='listening',revision=1),1)
        self.observer.voice_input(dict(kind='mic_off',revision=2),2)
        self.assertTrue(any(e.phase=='END' and e.activity_id==opaque(('input',1)) for e in self.events))
    def test_stale_playback_end_cannot_cancel_new_generation(self):
        event=lambda kind,g:SimpleNamespace(kind=kind,payload={'generation_id':g})
        self.observer.playback([event('playback_started',1)],1)
        self.observer.playback([event('playback_started',2),event('playback_stopped',1)],2)
        self.assertEqual(len(self.observer.playbacks),1)
        self.assertIn(('voice',2),self.observer.playbacks)
        self.assertEqual(self.events[-1].phase,'START')
    def test_clock_corrections_do_not_start_extra_playback(self):
        self.observer.playback([SimpleNamespace(kind='playback_started',payload={'generation_id':1,'clock_correction':True})],1)
        self.assertEqual(self.events,[])
    def test_observer_failure_does_not_raise_into_core(self):
        self.observer.safe(lambda:1/0)
        self.assertEqual(self.events,['DISCONNECT'])


class TransportTests(unittest.TestCase):
    def test_read_only_sse_origin_privacy_and_shutdown(self):
        with tempfile.TemporaryDirectory() as temp:
            service=PresenceService(temp,port=0).start()
            try:
                service.submit('READY'); time.sleep(0.06)
                url=f'http://127.0.0.1:{service.port}'
                with urlopen(Request(url+'/events',headers={'Origin':'http://127.0.0.1:8766'}),timeout=2) as response:
                    self.assertEqual(response.headers['Content-Type'],'text/event-stream')
                    while True:
                        line=response.readline().decode()
                        if line.startswith('data: '): break
                    frame=json.loads(line[6:]);self.assertTrue(frame['connected']);self.assertEqual(frame['state'],'IDLE')
                with self.assertRaises(HTTPError) as denied:
                    urlopen(Request(url+'/health',headers={'Origin':'https://untrusted.example'}),timeout=2)
                self.assertEqual(denied.exception.code,403)
                with self.assertRaises(HTTPError) as no_command:
                    urlopen(Request(url+'/events',data=b'execute',method='POST'),timeout=2)
                self.assertIn(no_command.exception.code,{405,501})
            finally:
                service.close()
            self.assertFalse(service.worker.is_alive());self.assertFalse(service.http.is_alive())


if __name__=='__main__': unittest.main()
