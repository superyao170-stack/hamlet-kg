from __future__ import annotations

import copy
import json
import unittest
from unittest.mock import patch

from hamletkg.detection.narrative import generate_evidence_report
from hamletkg.llm import LLMError


def evidence() -> dict:
    graph = {
        "nodes": [
            {"id": "INJECTED-SOURCE", "type": "ACCOUNT"},
            {"id": "INJECTED-HUB", "type": "ACCOUNT"},
            {"id": "NORMAL-DEST", "type": "ACCOUNT"},
            {"id": "TX-HISTORY", "type": "TRANSACTION",
             "timestamp": "2026-10-08T01:00:00Z", "source_record_id": "row-1",
             "attributes": {"amount": 100, "currency": "USD", "Is Laundering": 1}},
            {"id": "TX-TARGET", "type": "TRANSACTION",
             "timestamp": "2026-10-08T01:01:00Z", "source_record_id": "row-2",
             "attributes": {"amount": 95, "paid_currency": "USD", "label": 1}},
            {"id": "EVENT:INJECTED-CANDIDATE", "type": "BEHAVIOR_EVENT",
             "motif_type": "pass_through", "anchor_id": "INJECTED-HUB",
             "span_seconds": 60, "amount_ratio": .95},
        ],
        "edges": [
            {"source": "INJECTED-SOURCE", "relation": "SENDER_OF", "target": "TX-HISTORY"},
            {"source": "INJECTED-HUB", "relation": "RECEIVER_OF", "target": "TX-HISTORY"},
            {"source": "INJECTED-HUB", "relation": "SENDER_OF", "target": "TX-TARGET"},
            {"source": "NORMAL-DEST", "relation": "RECEIVER_OF", "target": "TX-TARGET"},
            {"source": "TX-HISTORY", "relation": "MEMBER_OF", "target": "EVENT:INJECTED-CANDIDATE",
             "member_role": "incoming", "elapsed_seconds": 60},
            {"source": "TX-TARGET", "relation": "MEMBER_OF", "target": "EVENT:INJECTED-CANDIDATE",
             "member_role": "outgoing", "elapsed_seconds": 0},
        ],
    }
    return {
        "metadata": {"Is Laundering": 1, "dataset": "secret-label-metadata"},
        "transactions": [{
            "event_id": "TX-TARGET", "observation_time": "2026-10-08T01:01:00Z",
            "score": .8, "candidate_count": 1, "status": "audited", "label": 1,
            "audited_candidates": [{
                "instance_id": "INJECTED-CANDIDATE", "motif_type": "pass_through",
                "historical_evidence_ids": ["TX-HISTORY"],
                "deletion_score_drop": .15, "without_evidence_score": .65,
                "random_deletion_score_drop": .02, "evidence_only_score": .76,
                "graph": graph,
            }],
        }],
    }


class FakeClient:
    def __init__(self, transform=None) -> None:
        self.transform = transform
        self.calls: list[dict] = []
        self.last_response_metadata = {"model": "test-model"}

    def complete_json(self, *, system: str, user: str) -> dict:
        parsed = json.loads(user)
        self.calls.append({"system": system, "user": user, "parsed": parsed})
        result = {
            "event_id": parsed["target_event_id"],
            "selected_fact_ids": [fact["fact_id"] for fact in parsed["facts"]],
            "assessment": "requires_review", "review_actions": ["review_source_records"],
        }
        return self.transform(result) if self.transform else result


class EvidenceNarrativeTests(unittest.TestCase):
    def test_report_preserves_bounded_transport_metadata_for_reproduction(self) -> None:
        client = FakeClient()
        client.last_response_metadata = {
            "requested_model": "configured-alias", "response_model": "actual-model",
            "finish_reason": "stop", "attempts": 1,
            "usage": {"prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125,
                      "provider_private_field": "excluded"},
            "headers": {"Authorization": "not-for-output"}, "provider_body": "excluded",
        }
        result = generate_evidence_report(evidence(), client)
        self.assertEqual("actual-model", result["model"])
        metadata = result["reports"][0]["llm_response_metadata"]
        self.assertEqual({"requested_model": "configured-alias", "response_model": "actual-model",
                          "finish_reason": "stop", "attempts": 1,
                          "usage": {"prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125}}, metadata)
        self.assertNotIn("not-for-output", json.dumps(result))

    def test_invalid_or_missing_metadata_does_not_invent_token_counts(self) -> None:
        for source in (None, {"attempts": True, "response_model": [],
                              "usage": {"prompt_tokens": True, "completion_tokens": -1,
                                        "total_tokens": "125"}}):
            with self.subTest(metadata=source):
                client = FakeClient()
                client.last_response_metadata = source
                result = generate_evidence_report(evidence(), client)
                self.assertTrue(result["validated"])
                metadata = result["reports"][0]["llm_response_metadata"]
                self.assertNotIn("attempts", metadata)
                self.assertEqual({}, metadata.get("usage", {}))

    def test_program_renders_only_verified_facts_and_aliases(self) -> None:
        payload, client = evidence(), FakeClient()
        original = copy.deepcopy(payload)
        result = generate_evidence_report(payload, client)
        self.assertTrue(result["validated"])
        self.assertEqual("test-model", result["model"])
        self.assertEqual(1, result["coverage"]["llm_call_count"])
        report = result["reports"][0]
        allowed = {"TX-HISTORY", "TX-TARGET"}
        for observation in report["observations"]:
            self.assertTrue(observation["transaction_ids"])
            self.assertLessEqual(set(observation["transaction_ids"]), allowed)
        transaction_facts = [fact["values"] for fact in report["fact_whitelist"]
                             if fact["kind"] == "observed_transaction"]
        self.assertEqual({100, 95}, {fact["amount"] for fact in transaction_facts})
        self.assertEqual({"USD"}, {fact["currency"] for fact in transaction_facts})
        self.assertEqual({"row-1", "row-2"}, {fact["source_record_id"] for fact in transaction_facts})
        event_fact = next(fact for fact in report["fact_whitelist"] if fact["kind"] == "candidate_event_membership")
        self.assertEqual(60, event_fact["values"]["span_seconds"])
        self.assertEqual(.95, event_fact["values"]["amount_ratio"])
        request = client.calls[0]["user"]
        for forbidden in ("INJECTED", "NORMAL-DEST", "Is Laundering", "secret-label-metadata", '"label"'):
            self.assertNotIn(forbidden, request)
        self.assertNotIn("INJECTED", result["markdown"])
        self.assertIn("INJECTED-HUB", result["account_aliases"].values())
        self.assertEqual(original, payload)

    def test_unknown_target_and_fact_references_are_rejected(self) -> None:
        for update in ({"event_id": "FOREIGN-TARGET"}, {"selected_fact_ids": ["T999"]},
                       {"selected_fact_ids": ["TX-INVENTED"]}, {"selected_fact_ids": []},
                       {"selected_fact_ids": ["S1", "S1"]}):
            with self.subTest(update=update):
                client = FakeClient(lambda response: {**response, **update})
                with self.assertRaises(LLMError):
                    generate_evidence_report(evidence(), client)

    def test_added_amount_time_free_prose_and_unknown_fields_are_rejected(self) -> None:
        for update in ({"amount": 99999}, {"timestamp": "2099-01-01"},
                       {"summary": "这是一笔已确认洗钱，金额99999美元。"},
                       {"observations": [{"statement": "新增金额99999", "transaction_ids": ["TX-TARGET"]}]},
                       {"analysis": "extra"}):
            with self.subTest(update=update):
                with self.assertRaises(LLMError):
                    generate_evidence_report(evidence(), FakeClient(lambda response: {**response, **update}))

    def test_definite_laundering_assessment_and_unknown_actions_are_rejected(self) -> None:
        for update in ({"assessment": "money_laundering_confirmed"},
                       {"assessment": "已确认洗钱"}, {"review_actions": ["freeze_account"]}):
            with self.subTest(update=update):
                with self.assertRaises(LLMError):
                    generate_evidence_report(evidence(), FakeClient(lambda response: {**response, **update}))

    def test_model_cannot_remove_nonpositive_support_or_weak_control_limits(self) -> None:
        payload = evidence()
        payload["transactions"][0]["audited_candidates"][0]["deletion_score_drop"] = -.1
        client = FakeClient(lambda response: {**response, "selected_fact_ids": ["S1"]})
        report = generate_evidence_report(payload, client)["reports"][0]
        limits = " ".join(report["limitations"])
        self.assertIn("未经概率校准", limits)
        self.assertIn("未降低模型评分", limits)
        self.assertIn("仅抽样一次", limits)
        self.assertIn("不证明现实世界", limits)
        self.assertEqual(["S1"], report["selected_fact_ids"])

    def test_no_event_status_is_preserved(self) -> None:
        payload = {"transactions": [{"event_id": "TX-NO-EVENT", "score": .4,
                                     "observation_time": "2026-10-08T00:00:00Z",
                                     "candidate_count": 0, "status": "no_event_candidate",
                                     "audited_candidates": []}]}
        report = generate_evidence_report(payload, FakeClient())["reports"][0]
        self.assertEqual("no_event_candidate", report["audit_status"]["status"])
        self.assertIn("没有候选知识事件", " ".join(report["limitations"]))
        self.assertEqual({"uncalibrated_model_score", "observation_time"},
                         {fact["kind"] for fact in report["fact_whitelist"]})

    def test_failed_and_missing_audits_are_not_reported_as_success(self) -> None:
        payload = evidence()
        payload["transactions"][0].update(status="failed", audited_candidates=[{"error": "private-error-details"}])
        report = generate_evidence_report(payload, FakeClient())["reports"][0]
        self.assertEqual("failed", report["audit_status"]["status"])
        self.assertEqual(1, report["audit_status"]["incomplete_audit_count"])
        self.assertIn("审计失败", " ".join(report["limitations"]))
        self.assertNotIn("private-error-details", json.dumps(report))

    def test_client_failure_raises_instead_of_returning_partial_success(self) -> None:
        class FailedClient:
            def complete_json(self, **kwargs):
                raise RuntimeError("private provider failure detail")
        with self.assertRaises(LLMError) as context:
            generate_evidence_report(evidence(), FailedClient())
        self.assertNotIn("private provider failure detail", str(context.exception))

    def test_supplied_client_does_not_load_credentials(self) -> None:
        with patch("hamletkg.llm.LLMConfig.from_environment", side_effect=AssertionError("must not load keys")) as loader:
            generate_evidence_report(evidence(), FakeClient())
            loader.assert_not_called()

    def test_limit_and_empty_input_coverage_are_explicit(self) -> None:
        payload = evidence()
        payload["transactions"] *= 4
        client = FakeClient()
        result = generate_evidence_report(payload, client, max_transactions=1)
        self.assertEqual(1, len(client.calls))
        self.assertEqual(3, result["coverage"]["omitted_transactions"])
        empty = generate_evidence_report({"transactions": []}, FakeClient())
        self.assertEqual("not_called", empty["model"])
        self.assertEqual(0, empty["coverage"]["llm_call_count"])
        for maximum in (0, -1, True, 1.5):
            with self.subTest(maximum=maximum), self.assertRaises(ValueError):
                generate_evidence_report(payload, FakeClient(), max_transactions=maximum)

    def test_graph_membership_cannot_forge_foreign_transaction_references(self) -> None:
        payload = evidence()
        payload["transactions"][0]["audited_candidates"][0]["graph"]["edges"][-1]["source"] = "FOREIGN-TX"
        client = FakeClient()
        with self.assertRaises(LLMError):
            generate_evidence_report(payload, client)
        self.assertEqual([], client.calls)


if __name__ == "__main__":
    unittest.main()
