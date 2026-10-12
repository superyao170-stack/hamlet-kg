import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from hamletkg.detection.disk_metrics import DiskScores
from hamletkg.detection.metrics import binary_metrics, choose_threshold, daily_budget_metrics


class DiskMetricTests(unittest.TestCase):
    def test_matches_reference_metrics_with_ties_utc_days_and_zero_budget(self):
        labels, scores = [0, 1, 1, 0, 0, 1], [.8, .8, .6, .4, .9, .1]
        timestamps = ["2026-01-01T00:00:00Z"] * 4 + ["2026-01-02T00:00:00Z"] * 2
        ids = ["b", "a", "d", "c", "e", "f"]
        samples = [SimpleNamespace(event_id=i, timestamp=t, events=()) for i,t in zip(ids,timestamps)]
        with tempfile.TemporaryDirectory() as directory, DiskScores(Path(directory)/"scores.db", create=True) as store:
            store.add_batch("event", "validation", samples, scores, labels)
            threshold = store.choose_threshold("event")
            self.assertEqual(choose_threshold(labels, scores), threshold)
            self.assertEqual(binary_metrics(labels, scores, threshold), store.binary_metrics("event", "validation", threshold))
            for fraction in (.001, .5, 1):
                expected = daily_budget_metrics(labels, scores, timestamps, ids, fraction)
                actual = store.daily_budget_metrics("event", "validation", fraction)
                for item in actual["daily"]:
                    self.assertEqual(0, item.pop("selected_event_ids_omitted"))
                self.assertEqual(expected, actual)
            capped = store.daily_budget_metrics("event", "validation", 1, selected_ids_limit=1)
            self.assertEqual(6, capped["review_alerts"])
            self.assertEqual(3, capped["true_positive"])
            self.assertEqual(4, sum(day["selected_event_ids_omitted"] for day in capped["daily"]))

    def test_unknown_labels_and_invalid_scores_never_become_negative(self):
        sample = SimpleNamespace(event_id="x", timestamp="2026-01-01T00:00:00Z", events=())
        with tempfile.TemporaryDirectory() as directory, DiskScores(Path(directory)/"scores.db", create=True) as store:
            for score, label in ((math.nan, 0), (2, 0), (.5, True), (.5, 0.0)):
                with self.assertRaises(ValueError):
                    store.add_batch("event", "test", [sample], [score], [label])
            store.add_batch("event", "test", [sample], [.5], [None])
            self.assertEqual(1, store.counts("event", "test")["unlabeled"])
            with self.assertRaises(ValueError):
                store.binary_metrics("event", "test", .5)

    def test_no_positive_class_stays_undefined_and_threshold_tie_is_highest(self):
        samples = [SimpleNamespace(event_id=str(i), timestamp="2026-01-01T00:00:00Z", events=()) for i in range(3)]
        with tempfile.TemporaryDirectory() as directory, DiskScores(Path(directory)/"scores.db", create=True) as store:
            store.add_batch("base", "test", samples, [.8, .5, .2], [0, 0, 0])
            result = store.binary_metrics("base", "test", .5)
            for metric in ("average_precision", "recall", "f1"):
                self.assertIsNone(result[metric])
            with self.assertRaises(ValueError):
                store.choose_threshold("base", "test")
            store.commit()
            store.export("base", "test", .5, directory)
            self.assertTrue((Path(directory)/"predictions_base.csv").exists())
