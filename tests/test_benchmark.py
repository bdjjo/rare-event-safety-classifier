import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rare_event.data import generate, assert_disjoint, fingerprint, load_jsonl
from rare_event.model import SafetyClassifier, clean_text
from rare_event.metrics import budget_metrics, rank_indices, evaluate, wilson


class DataTests(unittest.TestCase):
    def test_paired_shift_keeps_labels_and_ids(self):
        a = generate(1000, .01, 3, "test")
        b = generate(1000, .01, 3, "test", 1, 1)
        np.testing.assert_array_equal(a.label, b.label)
        np.testing.assert_array_equal(a.id, b.id)
        self.assertEqual(a.label.sum(), 10)
        self.assertTrue((a.loc[a.label == 1, "text"] != b.loc[b.label == 1, "text"]).all())
        self.assertEqual(fingerprint(a), fingerprint(generate(1000, .01, 3, "test")))

    def test_leakage_guard_and_metadata_removal(self):
        a, b = generate(1000, .01, 3, "train"), generate(1000, .01, 4, "calibration")
        assert_disjoint([a, b])
        with self.assertRaises(ValueError):
            assert_disjoint([a, a])
        self.assertEqual(clean_text("Safe [ticket=train-5-12]"), "safe")

    def test_loader_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            path.write_text('{"id":"a","text":"ok","label":0}\n{"id":"b","text":"risk","label":1}\n')
            self.assertEqual(len(load_jsonl(path)), 2)
            path.write_text('{"id":"a","text":"ok","label":3}\n')
            with self.assertRaises(ValueError):
                load_jsonl(path)
            path.write_text('{"id":"a","text":123,"label":0}\n{"id":"b","text":"risk","label":1}\n')
            with self.assertRaises(ValueError):
                load_jsonl(path)


class MetricTests(unittest.TestCase):
    def test_exact_capacity_and_hand_computed_metrics(self):
        m = budget_metrics([1, 0, 1, 0], [.9, .8, .7, .1], list("abcd"), .5)
        self.assertEqual((m["k"], m["tp"], m["fp"], m["fn"]), (2, 1, 1, 1))
        self.assertEqual((m["precision"], m["recall"], m["lift"]), (.5, .5, 1.))

    def test_ties_are_invariant_to_row_order(self):
        ids, scores = np.array(list("abcde")), np.ones(5)
        a = ids[rank_indices(scores, ids)]
        perm = np.array([4, 2, 0, 3, 1])
        b = ids[perm][rank_indices(scores[perm], ids[perm])]
        np.testing.assert_array_equal(a, b)

    def test_empty_alert_queue_is_defined(self):
        m = evaluate([0, 1], [.1, .2], ["a", "b"], [.5], .9)
        self.assertIsNone(m["frozen_threshold"]["precision"])
        self.assertEqual(m["frozen_threshold"]["recall"], 0)
        self.assertAlmostEqual(m["brier"], .325)

    def test_invalid_scores_and_wilson_boundary(self):
        with self.assertRaises(ValueError):
            rank_indices([np.nan], ["a"])
        lo, hi = wilson(0, 200)
        self.assertAlmostEqual(lo, 0)
        self.assertGreater(hi, 0)


class ModelTests(unittest.TestCase):
    def test_separate_calibration_and_monotonic_ranking(self):
        a = generate(2000, .01, 2, "train")
        b = generate(2000, .01, 3, "calibration")
        # A calibration-only token must never enter the fitted TF-IDF vocabulary.
        b["text"] = b.text + " calibrationonlysentinel"
        model = SafetyClassifier().fit(a.text, a.label, b.text, b.label)
        self.assertNotIn("calibrationonlysentinel", model.vectorizer.vocabulary_)
        scores = np.linspace(-5, 5, 100)
        p = model.predict_from_scores(scores)
        self.assertTrue(np.all(np.diff(p) >= 0))
        self.assertTrue(np.all((0 <= p) & (p <= 1)))
        self.assertGreater(model.calibrator.coef_[0, 0], 0)


if __name__ == "__main__":
    unittest.main()
