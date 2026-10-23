import sys
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import time
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent/'h1_2'),str(HERE.parents[2]/'programa_principal')]
from audio_tap import AudioTap
from runtime_motion import VoiceMotionService, MotionObserver


class AudioTests(unittest.TestCase):
    def setUp(self):
        self.tap=AudioTap()
        self.addCleanup(self.tap.close)

    def test_silence_soft_loud_extract_actual_samples(self):
        for strength in [0,0.02,0.3]:
            samples=(np.sin(np.arange(960)*0.13)*strength*32767).astype('<i2').tobytes()
            self.tap._consume(('clock',(1,strength),10.0))
            self.tap._consume(('pcm',(1,strength),memoryview(samples),0,48000,1,7))
        deadline=time.monotonic()+1
        while len(self.tap.features)<3 and time.monotonic()<deadline: time.sleep(.005)
        rms=[f[1] for f in self.tap.features]
        self.assertEqual(len(rms),3)
        self.assertEqual(rms[0],0)
        self.assertLess(rms[1],rms[2])
        self.assertGreater(rms[2],.1)

    def test_clock_prevents_early_audio_and_stale_generation(self):
        self.tap.features.append((20,.1,.3,7))
        self.assertEqual(self.tap.snapshot(7,19.9),(0,0,False))
        self.assertEqual(self.tap.snapshot(8,20.01),(0,0,False))
        self.assertTrue(self.tap.snapshot(7,20.01)[2])
        self.assertEqual(self.tap.snapshot(7,20.2),(0,0,False))

    def test_writable_buffer_rejected_and_queue_bounded(self):
        stream=SimpleNamespace(_segment={},_presence_token=1,frame_bytes=2,rate=48000,
                               _cancel_event=SimpleNamespace(generation=1))
        self.tap.pcm(stream,bytearray(100))
        self.assertEqual(self.tap.dropped,1)
        self.assertEqual(self.tap.pending.maxsize,64)
        self.assertEqual(self.tap.features.maxlen,800)

    def test_failed_tap_does_not_affect_original_write_or_callback(self):
        from anydes.voz.nueva.dac_output import DacOutputStream
        init,write,callback,diagnostics=DacOutputStream.__init__,DacOutputStream.write,DacOutputStream._callback,DacOutputStream.diagnostics
        calls=[]
        def original(stream,pcm): calls.append(pcm); return 'AUDIO_UNCHANGED'
        DacOutputStream.write=original
        try:
            self.tap.install()
            with patch.object(self.tap,'pcm',side_effect=RuntimeError('observer broken')):
                self.assertEqual(DacOutputStream.write(object(),b'12'),'AUDIO_UNCHANGED')
            self.assertEqual(calls,[b'12'])
            self.assertIs(DacOutputStream._callback,callback)
            self.assertEqual(self.tap.errors,1)
        finally: DacOutputStream.__init__,DacOutputStream.write,DacOutputStream.diagnostics=init,write,diagnostics

    def test_close_joins_worker(self):
        self.tap.close()
        self.assertFalse(self.tap.thread.is_alive())

    def test_real_dac_callback_bytes_and_dispatch_clock_feed_tap(self):
        from anydes.voz.nueva.dac_output import DacOutputStream
        init,write,diagnostics=DacOutputStream.__init__,DacOutputStream.write,DacOutputStream.diagnostics
        class Raw:
            latency=0.0
            def __init__(self,**kwargs): pass
            def start(self): pass
            def close(self): pass
        class FakeSD:
            RawOutputStream=Raw
            class CallbackStop(Exception): pass
            class CallbackAbort(Exception): pass
        stream=None
        try:
            self.tap.install()
            token=SimpleNamespace(generation=4,is_set=lambda:False)
            stream=DacOutputStream(FakeSD,samplerate=48000,channels=1,blocksize=960,event_sink=lambda payload:None,cancel_event=token)
            stream.set_segment({'duration_ms':20,'sample_rate':48000,'frame_count':960,'playback_id':1,'segment_index':0})
            stream.start()
            pcm=(np.sin(np.arange(960)*.13)*6000).astype('<i2').tobytes()
            stream.write(pcm)
            output=bytearray(len(pcm))
            stream._callback(output,960,SimpleNamespace(outputBufferDacTime=1.0,currentTime=1.0),SimpleNamespace(output_underflow=False))
            self.assertEqual(bytes(output),pcm)
            deadline=time.monotonic()+1
            while not self.tap.features and time.monotonic()<deadline: time.sleep(.002)
            self.assertTrue(self.tap.features)
            self.assertGreater(self.tap.snapshot(4)[0],0)
            self.assertEqual(stream.played_frames,960)
        finally:
            if stream: stream.close()
            DacOutputStream.__init__,DacOutputStream.write,DacOutputStream.diagnostics=init,write,diagnostics


class MotionProjectionTests(unittest.TestCase):
    def test_signal_is_gated_by_real_playback_state_and_small_payload(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            service=VoiceMotionService(tmp,port=0)
            try:
                service.generation=3
                service.tap.features.append((time.perf_counter(),.1,.3,3))
                state=service.adapter.ready()
                self.assertEqual(service.motion(state)['rms'],0)
                state.update(state='SPEAKING')
                frame=service.motion(state)
                self.assertGreater(frame['rms'],0)
                self.assertEqual(set(frame),{'schema','session','sequence','timestamp','working_mode','rms','peak','voice_active'})
                state.update(state='WORKING',source='tools.verification')
                self.assertEqual(service.motion(state)['working_mode'],'VERIFYING')
                service.modes[1]='PROGRAMMING'; state['source']='tools.execution'
                self.assertEqual(service.motion(state)['working_mode'],'PROGRAMMING')
            finally: service.close()

    def test_work_scope_releases_even_when_core_raises(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            service=VoiceMotionService(tmp,port=0)
            observer=MotionObserver(service)
            try:
                with self.assertRaisesRegex(RuntimeError,'core'):
                    with observer.work('PROGRAMMING'):
                        self.assertTrue(service.modes)
                        raise RuntimeError('core')
                self.assertEqual(service.modes,{})
            finally: service.close()

if __name__=='__main__': unittest.main()
