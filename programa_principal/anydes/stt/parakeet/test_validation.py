"""Contract tests; no microphone or STT models are opened."""
from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from common import (CORPUS, edit_alignment, intent_preserved_preliminary,
    language_drift, load_corpus, sha256, text_metrics)


class ValidationContractTests(unittest.TestCase):
    def test_frozen_corpus_hash_and_size(self):
        self.assertEqual(sha256(CORPUS),
            "d83b6f5c4e7b4845dd8fa39f52d580805db0c702bbb5d2c0dfc9d1166125abc1")
        rows = load_corpus()
        self.assertEqual(len(rows), 30)
        self.assertEqual(rows[8]["id"], "G09")
        self.assertIn("Anydes", rows[8]["text"])

    def test_edit_metrics_separate_omissions_and_insertions(self):
        metrics = text_metrics("abre el archivo", "abre archivo nuevo")
        self.assertEqual(metrics["omissions"], 1)
        self.assertEqual(metrics["insertions"], 1)

    def test_intent_preliminary_is_generic_not_phrase_specific(self):
        reference = "Busca informacion sobre tomates"
        hypothesis = "Busca informacion de tomates"
        metrics = text_metrics(reference, hypothesis)
        self.assertTrue(intent_preserved_preliminary(reference, hypothesis, metrics))

    def test_language_drift_is_diagnostic_only(self):
        result = language_drift("I need this file please")
        self.assertTrue(result["language_drift"])
        self.assertEqual(language_drift("Necesito este archivo")["language_drift"], False)


if __name__ == "__main__":
    unittest.main()
