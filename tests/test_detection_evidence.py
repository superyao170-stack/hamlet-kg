from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from hamletkg.detection.evidence import explain_transaction
from hamletkg.detection.stream import StreamConfig, build_samples
from hamletkg.models import AtomicEvent, CaseGraph, Entity


class CandidateCountDetector:
    def predict(self, samples):
        return [min(.99, .1 + .1 * sum(len(event.members) - 1 for event in sample.events))
                for sample in samples]


def evidence_case() -> CaseGraph:
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [("I1", "S1", "H", 0, 30), ("NOISE", "X", "Y", 30, 200),
            ("I2", "S2", "H", 60, 30), ("I3", "S3", "H", 120, 30),
            ("TARGET", "H", "Z", 180, 90), ("FUTURE", "Q", "H", 240, 100)]
    events = {identifier: AtomicEvent(identifier, "TRANSFER", {"sender": sender, "receiver": receiver},
              origin + timedelta(seconds=seconds), {"amount": amount, "currency": "USD"}, identifier)
              for identifier, sender, receiver, seconds, amount in rows}
    accounts = {account for event in events.values() for account in event.roles.values()}
    return CaseGraph("EVIDENCE", {account: Entity(account, "ACCOUNT") for account in accounts}, events)


class EvidenceAuditTests(unittest.TestCase):
    def test_deletion_rebuilds_members_and_features_without_future_or_target_removal(self):
        case = evidence_case()
        config = StreamConfig()
        sample = next(item for item in build_samples(case, config) if item.event_id == "TARGET")
        result = explain_transaction(case, sample, CandidateCountDetector(), config, seed=42)
        self.assertTrue(result["positive_support_found"])
        audit = result["audited_candidates"][0]
        self.assertEqual(["I1", "I2", "I3"], audit["historical_evidence_ids"])
        self.assertAlmostEqual(.3, audit["deletion_score_drop"])
        self.assertEqual(3, len(audit["random_control_ids"]))
        self.assertNotIn("FUTURE", audit["random_control_ids"])
        self.assertNotIn("TARGET", audit["random_control_ids"])
        self.assertAlmostEqual(result["score"], audit["evidence_only_score"])
        graph_ids = {node["id"] for node in audit["graph"]["nodes"]}
        self.assertIn("TARGET", graph_ids)
        self.assertNotIn("FUTURE", graph_ids)
        member_edges = [edge for edge in audit["graph"]["edges"] if edge["relation"] == "MEMBER_OF"]
        self.assertEqual({"incoming", "outgoing"}, {edge["member_role"] for edge in member_edges})
        self.assertIn("not matched", result["random_control_limitation"])

    def test_no_event_is_reported_without_manufacturing_an_explanation(self):
        case = evidence_case()
        sample = build_samples(case)[0]
        result = explain_transaction(case, sample, CandidateCountDetector(), StreamConfig())
        self.assertEqual("no_event_candidate", result["status"])
        self.assertEqual([], result["audited_candidates"])
        self.assertFalse(result["positive_support_found"])

    def test_negative_support_is_reported_honestly(self):
        class InverseDetector:
            def predict(self, samples):
                return [1 - value for value in CandidateCountDetector().predict(samples)]
        case = evidence_case()
        sample = next(item for item in build_samples(case) if item.event_id == "TARGET")
        result = explain_transaction(case, sample, InverseDetector(), StreamConfig())
        self.assertFalse(result["positive_support_found"])
        self.assertLess(result["audited_candidates"][0]["deletion_score_drop"], 0)

    def test_invalid_candidate_limit_is_rejected(self):
        case = evidence_case()
        with self.assertRaises(ValueError):
            explain_transaction(case, build_samples(case)[0], CandidateCountDetector(),
                                StreamConfig(), max_candidates=0)


if __name__ == "__main__":
    unittest.main()
