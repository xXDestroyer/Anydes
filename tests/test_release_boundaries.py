"""Portable tests against real exported classifier serialization code, not a mock core."""
from pathlib import Path
import importlib.util
import json
import unittest

ROOT = Path(__file__).resolve().parents[1]
WIRE = ROOT / 'experimentos/anydes_professional_shell_h1_4/h1_4b_13/overlay/conversation/classifier_wire.py'
spec = importlib.util.spec_from_file_location('exported_classifier_wire', WIRE)
wire = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wire)


class ClassifierSerialization(unittest.TestCase):
    def test_pure_object(self):
        self.assertEqual(wire.extract('{"intent":"CONTINUE_TASK"}'), {'intent': 'CONTINUE_TASK'})

    def test_fenced_object(self):
        self.assertEqual(wire.extract('```json\n{"value":1}\n```'), {'value': 1})

    def test_surrounding_prose(self):
        self.assertEqual(wire.extract('Result:\n{"value":1}\nDone.'), {'value': 1})

    def test_multiple_objects_rejected(self):
        with self.assertRaises(wire.ClassifierWireError):
            wire.extract('{"a":1} {"b":2}')

    def test_duplicate_keys_rejected(self):
        with self.assertRaises(wire.ClassifierWireError):
            wire.extract('{"a":1,"a":2}')

    def test_nonfinite_rejected(self):
        with self.assertRaises(wire.ClassifierWireError):
            wire.extract('{"a":NaN}')

    def test_non_json_rejected(self):
        with self.assertRaises(json.JSONDecodeError):
            wire.extract('not JSON')

    def test_array_not_promoted(self):
        with self.assertRaises(ValueError):
            wire.extract('[{"value":1}]')

    def test_source_escaping_preserved(self):
        source = 'path = r"C:\\demo\\letter_counter.py"\nprint(f"letters: {3}")\n# Unicode: \u03bb\n'
        self.assertEqual(wire.extract(json.dumps({'source': source}))['source'], source)

    def test_diagnostics_do_not_retain_raw_content(self):
        raw = '{"private_example":"do_not_retain"}'
        data = wire.metadata(raw)
        self.assertNotIn('do_not_retain', json.dumps(data))
        self.assertEqual(data['candidate_json_object_count'], 1)
        self.assertEqual(len(data['response_sha256']), 64)


if __name__ == '__main__':
    unittest.main()
