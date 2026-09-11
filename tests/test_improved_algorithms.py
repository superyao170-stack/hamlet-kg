from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from hamletkg.graph import build_atomic_relations, build_risk_relations
from hamletkg.materializer import IndexedWindowPatternMatcher, PatternMatcher
from hamletkg.models import AtomicEvent, CaseGraph, Entity, RiskPattern
from hamletkg.patterns import calibrate_pattern_reliability
from hamletkg.similarity import FusedGraphCaseMatcher, RiskCaseIndex


ROOT = Path(__file__).resolve().parents[1]


def two_episode_case() -> tuple[CaseGraph, RiskPattern]:
    identifiers = ["A", "B", "C", "D", "E", "X"]
    entities = {
        value: Entity(entity_id=value, entity_type="ACCOUNT")
        for value in identifiers
    }
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = [
        ("e1", "A", "X", 50, 0),
        ("e2", "B", "X", 50, 10),
        ("e3", "X", "C", 90, 20),
        ("e4", "D", "X", 60, 10_000),
        ("e5", "E", "X", 40, 10_010),
        ("e6", "X", "C", 85, 10_020),
    ]
    events = {
        event_id: AtomicEvent(
            event_id=event_id,
            action_type="TRANSFER",
            roles={"sender": sender, "receiver": receiver},
            timestamp=start + timedelta(seconds=offset),
            attributes={"amount": amount},
        )
        for event_id, sender, receiver, amount, offset in rows
    }
    pattern = RiskPattern.from_dict(
        {
            "pattern_id": "two-episode-fan-in-out",
            "risk_type": "FAN_IN_THEN_OUT",
            "clauses": [
                {
                    "clause_id": "incoming",
                    "action_type": "TRANSFER",
                    "anchor_role": "receiver",
                    "min_count": 2,
                    "distinct_role": "sender",
                },
                {
                    "clause_id": "outgoing",
                    "action_type": "TRANSFER",
                    "anchor_role": "sender",
                    "min_count": 1,
                },
            ],
            "constraints": [
                {
                    "constraint_type": "ordered_within",
                    "parameters": {
                        "earlier": "incoming",
                        "later": "outgoing",
                        "max_seconds": 3600,
                    },
                },
                {
                    "constraint_type": "sum_ratio",
                    "parameters": {
                        "numerator": "outgoing",
                        "denominator": "incoming",
                        "attribute": "amount",
                        "minimum": 0.8,
                        "maximum": 1.1,
                    },
                },
            ],
        }
    )
    return CaseGraph(case_id="TWO-EPISODES", entities=entities, atomic_events=events), pattern


def prepared_cases() -> list[CaseGraph]:
    pattern_values = json.loads((ROOT / "examples/patterns.json").read_text(encoding="utf-8"))
    patterns = [RiskPattern.from_dict(item) for item in pattern_values["patterns"]]
    matcher = PatternMatcher()
    cases = []
    for path in sorted((ROOT / "examples/raw_cases").glob("*.json")):
        case = CaseGraph.from_dict(json.loads(path.read_text(encoding="utf-8")))
        build_atomic_relations(case)
        for pattern in patterns:
            events, _ = matcher.match(case, pattern)
            case.risk_events.update({item.risk_event_id: item for item in events})
        build_risk_relations(case)
        cases.append(case)
    return cases


class IndexedWindowMatcherTests(unittest.TestCase):
    def test_separated_episodes_become_separate_risk_events(self) -> None:
        case, pattern = two_episode_case()
        legacy, _ = PatternMatcher().match(case, pattern)
        improved, _, diagnostics = IndexedWindowPatternMatcher().match_with_diagnostics(
            case, pattern
        )
        self.assertEqual(1, len(legacy))
        self.assertEqual(2, len(improved))
        self.assertEqual([3, 3], sorted(len(item.atomic_event_ids) for item in improved))
        self.assertEqual(1, diagnostics.candidate_anchors)
        self.assertEqual(len(case.entities) - 1, diagnostics.pruned_anchors)
        self.assertEqual(2, diagnostics.accepted_windows)

    def test_windowed_results_are_deterministic(self) -> None:
        case, pattern = two_episode_case()
        first, _ = IndexedWindowPatternMatcher().match(case, pattern)
        second, _ = IndexedWindowPatternMatcher().match(case, pattern)
        self.assertEqual(
            [item.risk_event_id for item in first],
            [item.risk_event_id for item in second],
        )

    def test_event_confidence_is_null_until_pattern_is_calibrated(self) -> None:
        case, pattern = two_episode_case()
        uncalibrated, _ = IndexedWindowPatternMatcher().match(case, pattern)
        self.assertTrue(all(item.confidence is None for item in uncalibrated))
        calibrated_pattern = calibrate_pattern_reliability(
            pattern, true_positive=8, false_positive=2
        )
        calibrated, _ = IndexedWindowPatternMatcher().match(case, calibrated_pattern)
        self.assertTrue(all(item.confidence == 0.75 for item in calibrated))


class FusedRetrievalTests(unittest.TestCase):
    def test_fused_matcher_preserves_expected_ranking(self) -> None:
        cases = prepared_cases()
        by_id = {item.case_id: item for item in cases}
        results = FusedGraphCaseMatcher().rank(by_id["CASE-A"], cases)
        self.assertEqual("CASE-B", results[0].candidate_case_id)
        self.assertEqual("fused_graph_sinkhorn", results[0].algorithm)

    def test_risk_case_index_prunes_before_reranking(self) -> None:
        cases = prepared_cases()
        by_id = {item.case_id: item for item in cases}
        results, diagnostics = RiskCaseIndex(cases).search(
            by_id["CASE-A"], top_k=1, candidate_pool_size=1
        )
        self.assertEqual("CASE-B", results[0].candidate_case_id)
        self.assertEqual(1, diagnostics.reranked_candidates)
        self.assertGreaterEqual(diagnostics.pruned_candidates, 1)

    def test_fused_self_similarity_is_one(self) -> None:
        case = prepared_cases()[0]
        result = FusedGraphCaseMatcher().compare(case, case)
        self.assertAlmostEqual(1.0, result.similarity, places=6)

    def test_fused_matcher_rewards_matching_topology(self) -> None:
        query = prepared_cases()[0]
        matching_value = query.to_dict()
        matching_value["case_id"] = "MATCHING-TOPOLOGY"
        matching = CaseGraph.from_dict(matching_value)
        disconnected_value = query.to_dict()
        disconnected_value["case_id"] = "DISCONNECTED-TOPOLOGY"
        disconnected_value["risk_relations"] = []
        disconnected = CaseGraph.from_dict(disconnected_value)
        matcher = FusedGraphCaseMatcher(semantic_weight=0.35)
        self.assertGreater(
            matcher.compare(query, matching).similarity,
            matcher.compare(query, disconnected).similarity,
        )


if __name__ == "__main__":
    unittest.main()
