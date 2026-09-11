from __future__ import annotations

import json
import unittest
from pathlib import Path

from hamletkg.evaluation import evaluate_retrieval
from hamletkg.graph import build_atomic_relations, build_risk_relations
from hamletkg.materializer import PatternMatcher
from hamletkg.models import CaseGraph, RiskPattern
from hamletkg.similarity import HierarchicalCaseMatcher


ROOT = Path(__file__).resolve().parents[1]


def prepared_cases() -> list[CaseGraph]:
    pattern_values = json.loads((ROOT / "examples/patterns.json").read_text(encoding="utf-8"))
    patterns = [RiskPattern.from_dict(item) for item in pattern_values["patterns"]]
    matcher = PatternMatcher()
    cases = []
    for path in sorted((ROOT / "examples/raw_cases").glob("*.json")):
        case = CaseGraph.from_dict(json.loads(path.read_text(encoding="utf-8")))
        build_atomic_relations(case)
        for pattern in patterns:
            risk_events, _ = matcher.match(case, pattern)
            case.risk_events.update({item.risk_event_id: item for item in risk_events})
        build_risk_relations(case)
        cases.append(case)
    return cases


class HierarchicalSimilarityTests(unittest.TestCase):
    def test_similar_pattern_case_ranks_first(self) -> None:
        cases = prepared_cases()
        by_id = {item.case_id: item for item in cases}
        results = HierarchicalCaseMatcher().rank(by_id["CASE-A"], cases)
        self.assertEqual("CASE-B", results[0].candidate_case_id)
        self.assertGreater(results[0].similarity, results[-1].similarity)

    def test_self_similarity_is_one(self) -> None:
        case = prepared_cases()[0]
        result = HierarchicalCaseMatcher().compare(case, case)
        self.assertAlmostEqual(1.0, result.similarity, places=6)

    def test_retrieval_metrics_are_reproducible(self) -> None:
        cases = prepared_cases()
        metrics = evaluate_retrieval(
            cases,
            {"CASE-A": ["CASE-B"], "CASE-B": ["CASE-A"]},
            top_k=1,
        )
        self.assertEqual(2, metrics.evaluated_queries)
        self.assertAlmostEqual(1.0, metrics.recall_at_k)
        self.assertAlmostEqual(1.0, metrics.mrr)


if __name__ == "__main__":
    unittest.main()

