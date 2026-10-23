"""Read-only startup diagnostics do not create, move or hide application windows."""
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace as NS
import json
import unittest
from unittest.mock import patch
from PySide6.QtCore import QRect
import startup_observer as observer


class Diagnostics(unittest.TestCase):
    def snapshot(self,rect,visible=True,minimized=False):
        screen=NS(availableGeometry=lambda:QRect(0,0,1920,1080))
        widget=NS(frameGeometry=lambda:rect,isVisible=lambda:visible,isHidden=lambda:not visible,
            isMinimized=lambda:minimized,windowHandle=lambda:NS(isVisible=lambda:visible,isExposed=lambda:visible and not minimized),
            screen=lambda:screen)
        app=NS(screens=lambda:[screen],topLevelWidgets=lambda:[widget])
        with patch('PySide6.QtWidgets.QApplication',app):
            result=observer.snapshot(widget)
        return result

    def test_valid_geometry_only_observed_not_changed(self):
        rect=QRect(120,90,1120,900)
        result=self.snapshot(rect)
        self.assertTrue(result['intersects_active_display'])
        self.assertEqual(result['geometry'],[120,90,1120,900])
        self.assertEqual(observer.rectangle(rect),[120,90,1120,900])

    def test_offscreen_state_detectable_without_speculative_recovery(self):
        result=self.snapshot(QRect(5000,5000,1120,900))
        self.assertFalse(result['intersects_active_display'])

    def test_hidden_and_minimized_are_distinct_diagnostics(self):
        hidden=self.snapshot(QRect(0,0,100,100),False)
        minimized=self.snapshot(QRect(0,0,100,100),True,True)
        self.assertTrue(hidden['hidden'])
        self.assertFalse(hidden['minimized'])
        self.assertFalse(minimized['hidden'])
        self.assertTrue(minimized['minimized'])

    def test_wrapper_preserves_return_value_and_does_not_log_arguments(self):
        host=NS(run=lambda value:value+1)
        output=StringIO()
        with redirect_stdout(output):
            observer.wrap(host,'run','boundary')
            self.assertEqual(host.run(123456),123457)
        events=[json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([e['startup_event'] for e in events],['boundary_entered','boundary_returned'])
        self.assertNotIn('123456',output.getvalue())

    def test_exception_preserved_without_message_content(self):
        private='NOT_FOR_DIAGNOSTICS'
        def fail():
            raise ValueError(private)
        host=NS(run=fail);output=StringIO()
        with redirect_stdout(output):
            observer.wrap(host,'run','boundary')
            with self.assertRaises(ValueError):
                host.run()
        self.assertNotIn(private,output.getvalue())
        self.assertIn('ValueError',output.getvalue())
        self.assertIn('traceback',output.getvalue())


if __name__=='__main__':
    unittest.main(verbosity=2)
