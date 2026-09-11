from __future__ import annotations

import unittest
from datetime import datetime, timezone

from hamletkg.evaluation import evaluate_risk_event_detection
from hamletkg.models import RiskEvent


def event(event_id: str, anchor: str, atomic_ids: tuple[str, ...]) -> RiskEvent:
    timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return RiskEvent(
        risk_event_id=event_id,
        risk_type="TEST-RISK",
        pattern_id="P",
        anchor_entity_id=anchor,
        atomic_event_ids=atomic_ids,
        start_time=timestamp,
        end_time=timestamp,
    )


class RiskEventEvaluationTests(unittest.TestCase):
    def test_atomic_iou_matching_reports_instance_metrics(self) -> None:
        reference = [
            event("r1", "A", ("a", "b", "c")),
            event("r2", "B", ("d", "e")),
        ]
        predicted = [
            event("p1", "A", ("a", "b", "c")),
            event("p2", "A", ("z",)),
        ]
        metrics = evaluate_risk_event_detection(predicted, reference)
        self.assertEqual(1, metrics.true_positive)
        self.assertEqual(1, metrics.false_positive)
        self.assertEqual(1, metrics.false_negative)
        self.assertAlmostEqual(0.5, metrics.precision)
        self.assertAlmostEqual(0.5, metrics.recall)
        self.assertAlmostEqual(0.5, metrics.f1)

    def test_invalid_iou_threshold_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_risk_event_detection([], [], minimum_atomic_iou=1.1)


if __name__ == "__main__":
    unittest.main()
