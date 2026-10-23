"""Qt component tests; no assistant, voice manager, tools or model is built."""
import sys
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'programa_principal'))
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from PySide6.QtCore import QEventLoop, QTimer
from shell import PresenceChoice, PresenceSurface
from policy import RendererKind
from anydes.interfaz.pyside_companion.app import AvatarStage

APP=QApplication.instance() or QApplication([])
APP.setQuitOnLastWindowClosed(False)


def wait(milliseconds):
    # QEventLoop.exec releases the GIL for the Python loopback asset threads.
    loop=QEventLoop()
    QTimer.singleShot(milliseconds,loop.quit)
    loop.exec()


class RendererTests(unittest.TestCase):
    def test_selector_previews_and_choice(self):
        picker=PresenceChoice()
        picker.show(); wait(200)
        picker.grab().save(str(HERE/'evidence/selector.png'))
        picker.select(RendererKind.HOLOGRAM)
        self.assertEqual(picker.kind,RendererKind.HOLOGRAM)
        picker.close(); picker.deleteLater(); wait(50)

    def test_hologram_embedded_and_failure_isolated(self):
        records=[]
        surface=PresenceSurface(RendererKind.HOLOGRAM,AvatarStage,lambda event,**kw:records.append(event))
        surface.resize(1000,480);surface.show()
        for _ in range(100):
            wait(300)
            if surface.ready or surface.failed:break
        self.assertTrue(surface.ready,records)
        self.assertFalse(surface.failed)
        self.assertIsNone(surface.legacy)
        surface.grab().save(str(HERE/'evidence/qt-hologram.png'))
        surface.fail('TEST_PROCESS_FAILURE')
        self.assertTrue(surface.failed)
        self.assertIs(surface.stack.currentWidget(),surface.neutral)
        surface.begin_turn(1);surface.stop_speaking()
        surface.shutdown(); self.assertFalse(surface.server.thread.is_alive())
        count=len(records)
        surface.fail('INTENTIONAL_PROCESS_SHUTDOWN')
        self.assertEqual(len(records),count)
        surface.close();surface.deleteLater();wait(100)

    def test_avatar_existing_renderer_compatibility(self):
        import os
        from lanzar_anydes import load_profile, OFFICIAL_VISUAL_PROFILE
        load_profile(OFFICIAL_VISUAL_PROFILE)
        os.environ['ANYDES_AVATAR_PRESENTATION']='2d_hq'
        records=[]
        surface=PresenceSurface(RendererKind.AVATAR_2_5D,AvatarStage,lambda event,**kw:records.append(event))
        self.addCleanup(surface.shutdown)
        surface.resize(1000,500);surface.show()
        for _ in range(200):
            wait(300)
            if surface.ready or surface.failed:break
        view=surface.legacy._vrm_view
        self.assertTrue(surface.ready,dict(records=records,diagnostics=view.diagnostic_state,messages=view.diagnostic_messages))
        self.assertIsNone(surface.web)
        self.assertIsNotNone(surface.legacy._vrm_view)
        surface.set_presence(dict(state='THINKING',connected=True),'NONE')
        self.assertEqual(surface.legacy._state.gesture,'thinking')
        surface.grab().save(str(HERE/'evidence/qt-avatar.png'))
        surface.shutdown();surface.close();surface.deleteLater();wait(100)


if __name__=='__main__':
    unittest.main()
