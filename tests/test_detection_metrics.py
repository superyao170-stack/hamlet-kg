from __future__ import annotations

import json
import math
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from hamletkg.detection.metrics import (
    binary_metrics,
    choose_threshold,
    chronological_split,
    daily_budget_metrics,
)


class DetectionMetricsTests(unittest.TestCase):
    def test_average_precision_treats_equal_scores_as_a_group(self) -> None:
        first = binary_metrics([1, 0, 1], [0.9, 0.9, 0.4])
        second = binary_metrics([0, 1, 1], [0.9, 0.9, 0.4])
        self.assertAlmostEqual(7 / 12, first["average_precision"])
        self.assertEqual(first, second)
        self.assertEqual(1, first["true_positive"])
        self.assertEqual(1, first["false_positive"])
        self.assertEqual(1, first["false_negative"])
        self.assertAlmostEqual(1000 / 3, first["false_positives_per_1000"])
        self.assertAlmostEqual(2 / 3, first["prevalence"])

    def test_no_positive_and_all_positive_samples(self) -> None:
        negative = binary_metrics([0, 0], [0.8, 0.2])
        self.assertIsNone(negative["average_precision"])
        self.assertIsNone(negative["recall"])
        self.assertIsNone(negative["f1"])
        self.assertEqual(0.0, negative["precision"])
        positive = binary_metrics([1, 1, 1], [0.9, 0.9, 0.2])
        self.assertEqual(1.0, positive["average_precision"])
        self.assertAlmostEqual(2 / 3, positive["recall"])
        self.assertAlmostEqual(0.8, positive["f1"])
        json.dumps(negative, allow_nan=False)

    def test_all_tied_average_precision_equals_prevalence(self) -> None:
        result = binary_metrics([1, 0, 0, 1], [0.5] * 4)
        self.assertEqual(0.5, result["average_precision"])

    def test_threshold_above_one_selects_no_alerts(self) -> None:
        result = binary_metrics([0, 1], [0.9, 1.0], math.nextafter(1.0, math.inf))
        self.assertEqual(0, result["predicted_positive"])
        self.assertEqual(0.0, result["precision"])
        self.assertEqual(0.0, result["recall"])
        self.assertEqual(0.0, result["f1"])

    def test_choose_threshold_maximizes_f1_and_prefers_higher_tie(self) -> None:
        # At 0.9: F1=2/(2+1)=2/3. At 0.2: F1=4/(2+4)=2/3.
        labels, scores = [1, 0, 0, 1], [0.9, 0.8, 0.7, 0.2]
        self.assertEqual(0.9, choose_threshold(labels, scores))
        self.assertEqual(0.2, choose_threshold([0, 1, 1], [0.9, 0.8, 0.2]))
        self.assertEqual(0.0, choose_threshold([0, 1], [0.0, 0.0]))

    def test_threshold_choice_rejects_single_class_validation(self) -> None:
        for labels in ([0, 0], [1, 1]):
            with self.subTest(labels=labels), self.assertRaisesRegex(ValueError, "validation.*both"):
                choose_threshold(labels, [0.1, 0.9])

    def test_invalid_labels_scores_and_thresholds(self) -> None:
        invalid_pairs = [([], []), ([0], []), ([2], [0.5]), (["1"], [0.5]), ([1], [None]), ([1], [math.nan]), ([1], [math.inf]), ([1], [-0.1]), ([1], [1.1])]
        for labels, scores in invalid_pairs:
            with self.subTest(labels=labels, scores=scores), self.assertRaises(ValueError):
                binary_metrics(labels, scores)
        for threshold in (math.nan, math.inf, -1, None):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                binary_metrics([0, 1], [0.2, 0.8], threshold)

    def test_daily_budget_uses_floor_and_reports_zero_budget_days(self) -> None:
        result = daily_budget_metrics(
            [1, 0, 1, 0, 0], [0.8, 0.8, 0.9, 0.2, 0.1],
            ["2025-01-01T12:00:00Z"] * 2 + ["2025-01-02T12:00:00Z"] * 3,
            ["b", "a", "c", "d", "e"], budget_fraction=0.4,
        )
        self.assertEqual([0, 1], [day["budget"] for day in result["daily"]])
        self.assertEqual([], result["daily"][0]["selected_event_ids"])
        self.assertEqual(1, result["review_alerts"])
        self.assertEqual(1.0, result["precision"])
        self.assertEqual(0.5, result["recall"])
        self.assertEqual(5, result["total"])
        json.dumps(result, allow_nan=False)

    def test_daily_budget_ties_use_event_id_and_timestamps_use_utc(self) -> None:
        result = daily_budget_metrics(
            [1, 0], [0.8, 0.8],
            ["2025-01-02T01:00:00+08:00", "2025-01-01T20:00:00Z"],
            ["z", "a"], budget_fraction=0.5,
        )
        self.assertEqual("2025-01-01", result["daily"][0]["date"])
        self.assertEqual(["a"], result["daily"][0]["selected_event_ids"])
        self.assertEqual(1, result["false_positive"])
        self.assertEqual(0.0, result["recall"])
        self.assertEqual(500.0, result["false_positives_per_1000"])

    def test_daily_budget_rejects_missing_candidates_and_bad_inputs(self) -> None:
        with self.assertRaises(ValueError):
            daily_budget_metrics([1, 0], [0.8, None], ["2025-01-01"] * 2, ["a", "b"])
        with self.assertRaises(ValueError):
            daily_budget_metrics([1], [0.8], [], ["a"])
        with self.assertRaisesRegex(ValueError, "unique"):
            daily_budget_metrics([1, 0], [0.8, 0.2], ["2025-01-01"] * 2, ["a", "a"])
        for fraction in (-0.1, 1.1, math.nan):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                daily_budget_metrics([1], [0.8], ["2025-01-01"], ["a"], fraction)

    def test_split_sorts_original_indices_and_keeps_complete_groups(self) -> None:
        origin = datetime(2025, 1, 1, tzinfo=timezone.utc)
        samples = [SimpleNamespace(timestamp=origin + timedelta(days=day), event_id=name)
                   for day, name in [(3, "z"), (0, "b"), (1, "c"), (0, "a"), (2, "d"), (3, "y")]]
        split = chronological_split(samples, 0.5, 0.2)
        self.assertEqual([3, 1, 2], split["train"])
        self.assertEqual([4], split["validation"])
        self.assertEqual([5, 0], split["test"])
        self.assertLess(max(samples[i].timestamp for i in split["train"]), min(samples[i].timestamp for i in split["validation"]))
        self.assertLess(max(samples[i].timestamp for i in split["validation"]), min(samples[i].timestamp for i in split["test"]))

    def test_split_normalizes_equivalent_timezone_offsets(self) -> None:
        samples = [SimpleNamespace(timestamp=stamp, event_id=str(index)) for index, stamp in enumerate(
            ["2025-01-01T00:00:00Z", "2025-01-01T08:00:00+08:00", "2025-01-02", "2025-01-03"])]
        split = chronological_split(samples)
        self.assertEqual([0, 1], split["train"])
        self.assertEqual([2], split["validation"])
        self.assertEqual([3], split["test"])

    def test_split_rejects_small_or_group_insufficient_data_and_bad_fractions(self) -> None:
        samples = [SimpleNamespace(timestamp="2025-01-01", event_id=str(index)) for index in range(4)]
        with self.assertRaisesRegex(ValueError, "distinct timestamps"):
            chronological_split(samples)
        with self.assertRaisesRegex(ValueError, "three samples"):
            chronological_split(samples[:2])
        for train, validation in [(0, 0.2), (0.8, 0.2), (0.6, -0.1), (math.nan, 0.2)]:
            with self.subTest(train=train, validation=validation), self.assertRaises(ValueError):
                chronological_split(samples, train, validation)


if __name__ == "__main__":
    unittest.main()
