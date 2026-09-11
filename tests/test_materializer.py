from __future__ import annotations

import json
import unittest
from pathlib import Path

from hamletkg.graph import build_atomic_relations
from hamletkg.materializer import PatternMatcher
from hamletkg.models import AtomicEvent, CaseGraph, Entity, RiskPattern


ROOT = Path(__file__).resolve().parents[1]


def load_patterns() -> list[RiskPattern]:
    values = json.loads((ROOT / "examples/patterns.json").read_text(encoding="utf-8"))
    return [RiskPattern.from_dict(item) for item in values["patterns"]]


def load_case(name: str) -> CaseGraph:
    value = json.loads((ROOT / f"examples/raw_cases/{name}.json").read_text(encoding="utf-8"))
    return CaseGraph.from_dict(value)


class PatternMaterializationTests(unittest.TestCase):
    def materialize(self, name: str) -> CaseGraph:
        case = load_case(name)
        build_atomic_relations(case)
        matcher = PatternMatcher()
        for pattern in load_patterns():
            events, _ = matcher.match(case, pattern)
            case.risk_events.update({item.risk_event_id: item for item in events})
        return case

    def test_suspicious_case_materializes_three_risk_events(self) -> None:
        case = self.materialize("case_a")
        self.assertEqual(
            {"FAST_PASS_THROUGH", "FAN_IN_THEN_OUT", "RAPID_CASH_OUT"},
            {item.risk_type for item in case.risk_events.values()},
        )

    def test_normal_payroll_does_not_materialize_a_risk_event(self) -> None:
        case = self.materialize("case_c")
        self.assertEqual({}, case.risk_events)

    def test_atomic_graph_contains_temporal_and_flow_relations(self) -> None:
        case = load_case("case_a")
        relations = build_atomic_relations(case)
        types = {item.relation_type for item in relations}
        self.assertIn("NEXT", types)
        self.assertIn("FUNDS_FLOW_TO", types)

    def test_attribute_range_selects_sub_threshold_events_before_count(self) -> None:
        case = CaseGraph(
            case_id="STRUCTURING",
            entities={"A": Entity("A", "ACCOUNT")},
            atomic_events={
                item.event_id: item
                for item in [
                    AtomicEvent("D1", "CASH_DEPOSIT", {"account": "A"}, "2026-01-01T00:00:00Z", {"amount": 4000}),
                    AtomicEvent("D2", "CASH_DEPOSIT", {"account": "A"}, "2026-01-01T01:00:00Z", {"amount": 3500}),
                    AtomicEvent("D3", "CASH_DEPOSIT", {"account": "A"}, "2026-01-01T02:00:00Z", {"amount": 3000}),
                    AtomicEvent("D4", "CASH_DEPOSIT", {"account": "A"}, "2026-01-01T03:00:00Z", {"amount": 15000}),
                ]
            },
        )
        pattern = RiskPattern.from_dict(
            {
                "pattern_id": "P-STRUCT",
                "risk_type": "STRUCTURING",
                "clauses": [
                    {
                        "clause_id": "cash",
                        "action_type": "CASH_DEPOSIT",
                        "anchor_role": "account",
                        "min_count": 3,
                    }
                ],
                "constraints": [
                    {
                        "constraint_type": "attribute_range",
                        "parameters": {"clause": "cash", "attribute": "amount", "maximum": 9999.99},
                    },
                    {
                        "constraint_type": "total_amount",
                        "parameters": {"clause": "cash", "attribute": "amount", "minimum": 10000},
                    },
                ],
            }
        )
        events, _ = PatternMatcher().match(case, pattern)
        self.assertEqual(1, len(events))
        self.assertEqual({"D1", "D2", "D3"}, set(events[0].atomic_event_ids))

    def test_role_overlap_detects_u_turn_counterparty(self) -> None:
        case = CaseGraph(
            case_id="U-TURN",
            entities={
                entity_id: Entity(entity_id, "ACCOUNT")
                for entity_id in ("A", "B", "C")
            },
            atomic_events={
                "T1": AtomicEvent("T1", "TRANSFER", {"sender": "B", "receiver": "A"}, "2026-01-01T00:00:00Z", {"amount": 100}),
                "T2": AtomicEvent("T2", "TRANSFER", {"sender": "A", "receiver": "B"}, "2026-01-01T01:00:00Z", {"amount": 95}),
                "T3": AtomicEvent("T3", "TRANSFER", {"sender": "A", "receiver": "C"}, "2026-01-01T02:00:00Z", {"amount": 5}),
            },
        )
        pattern = RiskPattern.from_dict(
            {
                "pattern_id": "P-U",
                "risk_type": "U_TURN",
                "clauses": [
                    {"clause_id": "incoming", "action_type": "TRANSFER", "anchor_role": "receiver"},
                    {"clause_id": "outgoing", "action_type": "TRANSFER", "anchor_role": "sender"},
                ],
                "constraints": [
                    {"constraint_type": "ordered_within", "parameters": {"earlier": "incoming", "later": "outgoing", "max_seconds": 86400}},
                    {"constraint_type": "role_overlap", "parameters": {"left_clause": "incoming", "left_role": "sender", "right_clause": "outgoing", "right_role": "receiver"}},
                ],
            }
        )
        events, _ = PatternMatcher().match(case, pattern)
        self.assertEqual(1, len(events))


if __name__ == "__main__":
    unittest.main()
