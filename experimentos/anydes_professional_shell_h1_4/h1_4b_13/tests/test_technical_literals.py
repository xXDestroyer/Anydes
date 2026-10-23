"""Approved chat is immutable; only proven math spans are prepared for speech."""
from dataclasses import replace
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from anydes_tools.conversation.technical_literals import present_technical,literal_spans
from anydes_tools.tests import test_metadata_publication as metadata


class Literals(unittest.TestCase):
    def test_technical_structures_never_transform(self):
        samples=[r'Lo guarde en C:\Users\DemoUser Name\letter_counter.py.',
            '/home/demo/letter_counter.py','letter_counter.py',
            'https://example.com/path?a=1&x_1=2','artifact:abc123def task:0123456789',
            'sha256:'+'a'*64,'b'*64,'python script.py','`x_1` y `value_name`',
            r'\\server\share\sample.py','./src/sample.py','foo_bar']
        for source in samples:
            with self.subTest(source=source):
                p=present_technical(source,transform=lambda _:self.fail('technical span reached math transform'))
                self.assertEqual(p.chat_text,source)
                self.assertEqual(p.speech_text,source)
                self.assertEqual(p.math_regions,0)

    def test_math_still_adapts_speech_only(self):
        for source in ['x_1 + x_2','E=mc^2','sqrt(x)']:
            p=present_technical(source)
            self.assertEqual(p.chat_text,source)
            self.assertGreater(p.math_regions,0)
            self.assertNotEqual(p.speech_text,source)

    def test_mixed_math_and_path(self):
        path=r'C:\x\y.py'
        source='Archivo '+path+' y formula E=mc^2'
        p=present_technical(source)
        self.assertEqual(p.chat_text,source)
        self.assertIn(path,p.speech_text)
        self.assertGreater(p.math_regions,0)

    def test_adapter_marker_never_introduced(self):
        for marker in ['TOKEN','TECH_LITERAL_0','PATH_TOKEN','ANYDESPATHPLACEHOLDER0TOKEN']:
            source='E=mc^2'
            p=present_technical(source,transform=lambda _,m=marker:(m,1))
            self.assertEqual(p.chat_text,source)
            self.assertEqual(p.speech_text,source)
            self.assertTrue(p.fallback_used)

    def test_exception_and_empty_speech_fall_back_to_original(self):
        def fail(_):raise RuntimeError('controlled')
        for transform in [fail,lambda _:('',1)]:
            source=r'Archivo C:\x\y.py y E=mc^2'
            p=present_technical(source,transform=transform)
            self.assertEqual(p.chat_text,source)
            self.assertEqual(p.speech_text,source)
            self.assertTrue(p.fallback_used)

    def test_chat_does_not_call_legacy_normalizer(self):
        with patch('anydes.continuity_v2.math_presentation.normalize_chat',side_effect=RuntimeError('legacy')):
            source=r'Esta en C:\Users\DemoUser\letter_counter.py'
            self.assertEqual(present_technical(source).chat_text,source)

    def test_speech_preparation_boundary_failure_keeps_both_surfaces_safe(self):
        source=r'Archivo C:\Test\letter_counter.py y E=mc^2'
        with patch('anydes_tools.conversation.technical_literals.literal_spans',side_effect=OSError('controlled')):
            p=present_technical(source)
        self.assertEqual(p.chat_text,source)
        self.assertEqual(p.speech_text,source)
        self.assertTrue(p.fallback_used)

    def test_runtime_boundary_installed_without_voice_engine_change(self):
        from anydes.continuity_v2 import math_runtime
        source=r'C:\x\letter_counter.py'
        self.assertEqual(math_runtime.present_math(source).chat_text,source)


class ApprovedMetadata(unittest.TestCase):
    setUp=metadata.GroundedMetadata.setUp
    tearDown=metadata.GroundedMetadata.tearDown
    controller=metadata.GroundedMetadata.controller
    source=metadata.GroundedMetadata.source
    ready=metadata.GroundedMetadata.ready
    publish=metadata.GroundedMetadata.publish

    def test_location_through_naturalization_publication_adapter_and_ui_hook(self):
        from anydes.continuity_v2 import math_runtime
        c,ref=self.ready()
        text='Lo guarde en '+ref.path
        final,_,_=self.publish(c,text)
        calls=[]
        class Window:
            def _handle_partial(self,text):calls.append(('partial',text))
            def _handle_answer(self,text):
                calls.append(('chat',text))
                calls.append(('speech',self.assistant.last_intention.speech_intention.text))
        module=NS(AnydesPySideWindow=Window)
        math_runtime._patch_ui_module(module,performance=False)
        window=Window()
        window.assistant=NS(last_intention=NS(speech_intention=None),audit=None)
        # The actual speech intent is a dataclass, as required by the real hook.
        from dataclasses import dataclass
        @dataclass(frozen=True)
        class Speech:
            text:str
            metadata:dict
        @dataclass(frozen=True)
        class Intention:
            speech_intention:Speech
        window.assistant.last_intention=Intention(Speech(final.spoken_text,{}))
        window._handle_answer(final.visible_text)
        self.assertEqual(dict(calls)['chat'],text)
        self.assertEqual(dict(calls)['speech'],text)
        self.assertIn(ref.path,dict(calls)['chat'])
        self.assertFalse(c._executor_called or c._reasoner_called)
