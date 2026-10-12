"""Mock-only tests for bounded policy compilation and atomic extraction."""

from __future__ import annotations

import copy
import unittest

from hamletkg.llm import LLMAugmentor, LLMError


class SequenceClient:
    def __init__(self, values: list[object]) -> None:
        self.values = list(values)
        self.calls: list[dict[str, str]] = []
        self.last_response_metadata = {
            "requested_model": "mock-model", "response_model": "mock-model", "attempts": 1,
        }

    def complete_json(self, *, system: str, user: str) -> object:
        self.calls.append({"system": system, "user": user})
        value = self.values.pop(0)
        if isinstance(value, Exception):
            raise value
        return copy.deepcopy(value)


def policy_payload(*, operator: str = "fan_in", rule_id: str = "R1",
                   policy_id: str = "P1", version: str = "1.0",
                   window_seconds: int = 3600) -> dict:
    rule = {"rule_id": rule_id, "operator": operator, "window_seconds": window_seconds}
    if operator in ("fan_in", "fan_out"):
        rule["min_counterparties"] = 3
    else:
        rule.update({"ratio_min": .8, "ratio_max": 1.2})
        if operator == "pass_through":
            rule["min_incoming"] = 2
    return {"policy_id": policy_id, "version": version, "rules": [rule]}


def extraction_payload() -> dict:
    return {
        "entities": [
            {"entity_id": "A", "entity_type": "ACCOUNT", "attributes": {}},
            {"entity_id": "B", "entity_type": "ACCOUNT", "attributes": {}},
        ],
        "atomic_events": [{
            "event_id": "TX1", "action_type": "TRANSFER",
            "roles": {"sender": "A", "receiver": "B"},
            "timestamp": "2026-01-01T10:00:00Z",
            "attributes": {"amount": 100.0, "currency": "USD"},
            "source_record_id": "source-line-1",
        }],
    }


class DetectionPolicyCompilationTests(unittest.TestCase):
    def test_valid_policy_has_separate_uncalibrated_provenance(self) -> None:
        original = policy_payload()
        client = SequenceClient([original])
        policy, quality = LLMAugmentor(client).compile_detection_policy("三人一小时内汇入")
        self.assertEqual(original, policy.to_dict())
        self.assertEqual({"policy_id", "version", "rules"}, set(policy.to_dict()))
        self.assertEqual("llm_detection_policy_compilation", quality["source"])
        self.assertEqual(1, quality["valid_samples"])
        self.assertEqual(1.0, quality["valid_sample_rate"])
        self.assertEqual(0, quality["repair_attempts"])
        self.assertTrue(quality["requires_domain_review"])
        self.assertEqual("not_verified_by_schema_or_agreement", quality["semantic_correctness"])
        self.assertEqual([client.last_response_metadata], quality["response_metadata"])
        self.assertNotIn("quality", policy.to_dict())

    def test_each_supported_operator_can_compile(self) -> None:
        for operator in ("fan_in", "fan_out", "pass_through", "relay"):
            with self.subTest(operator=operator):
                value = policy_payload(operator=operator)
                policy, _ = LLMAugmentor(SequenceClient([value])).compile_detection_policy("rule")
                self.assertEqual(value, policy.to_dict())

    def test_extra_top_level_fields_do_not_silently_enter_policy(self) -> None:
        for name in ("description", "label", "quality", "metadata", "constraints", "policy"):
            with self.subTest(field=name):
                value = policy_payload()
                value[name] = {"untrusted": True}
                client = SequenceClient([value])
                with self.assertRaisesRegex(LLMError, "schema validation"):
                    LLMAugmentor(client).compile_detection_policy("rule", max_repair_attempts=0)
                self.assertEqual(1, len(client.calls))

    def test_missing_fields_and_wrong_shapes_are_rejected(self) -> None:
        malformed: list[object] = [[], None, "not-an-object", {}]
        for name in ("policy_id", "version", "rules"):
            value = policy_payload()
            del value[name]
            malformed.append(value)
        for rules in ([], None, "not-a-list", [{"operator": "fan_in"}]):
            value = policy_payload()
            value["rules"] = rules
            malformed.append(value)
        for index, value in enumerate(malformed):
            with self.subTest(index=index), self.assertRaises(LLMError):
                LLMAugmentor(SequenceClient([value])).compile_detection_policy(
                    "rule", max_repair_attempts=0)

    def test_operator_specific_unknown_fields_are_rejected(self) -> None:
        for operator, field in (("fan_in", "ratio_min"), ("fan_out", "min_incoming"),
                                ("relay", "path_length"), ("relay", "min_incoming"),
                                ("pass_through", "amount_threshold")):
            with self.subTest(operator=operator, field=field):
                value = policy_payload(operator=operator)
                value["rules"][0][field] = 1
                with self.assertRaises(LLMError):
                    LLMAugmentor(SequenceClient([value])).compile_detection_policy(
                        "rule", max_repair_attempts=0)

    def test_invalid_candidate_is_repaired_once_using_original_rule(self) -> None:
        invalid = policy_payload()
        invalid["rules"][0]["min_counterparties"] = "three"
        client = SequenceClient([invalid, policy_payload()])
        policy, quality = LLMAugmentor(client).compile_detection_policy("三人一小时内汇入")
        self.assertEqual("fan_in", policy.rules[0].operator)
        self.assertEqual(2, len(client.calls))
        self.assertEqual(1, quality["repair_attempts"])
        self.assertIn("三人一小时内汇入", client.calls[1]["user"])
        self.assertIn("failed validation", client.calls[1]["user"])
        self.assertEqual(client.calls[0]["system"], client.calls[1]["system"])

    def test_failed_repair_has_no_second_repair_or_default_policy(self) -> None:
        invalid = policy_payload()
        invalid["rules"][0]["operator"] = "cycle"
        client = SequenceClient([invalid, invalid, policy_payload()])
        with self.assertRaisesRegex(LLMError, "schema validation"):
            LLMAugmentor(client).compile_detection_policy("cycle", max_repair_attempts=1)
        self.assertEqual(2, len(client.calls))
        self.assertEqual(1, len(client.values))

    def test_semantic_votes_ignore_ids_version_and_rule_order(self) -> None:
        first = policy_payload(policy_id="P-A", rule_id="R-A")
        first["rules"].append(policy_payload(operator="relay", rule_id="R-B")["rules"][0])
        equivalent = copy.deepcopy(first)
        equivalent["policy_id"] = "P-RENAMED"
        equivalent["version"] = "different-display-version"
        equivalent["rules"].reverse()
        equivalent["rules"][0]["rule_id"] = "R-X"
        equivalent["rules"][1]["rule_id"] = "R-Y"
        minority = policy_payload(operator="fan_out")
        client = SequenceClient([first, minority, equivalent])
        policy, quality = LLMAugmentor(client).compile_detection_policy("rule", samples=3)
        self.assertEqual({"fan_in", "relay"}, {item.operator for item in policy.rules})
        self.assertEqual(3, quality["valid_samples"])
        self.assertAlmostEqual(2 / 3, quality["semantic_agreement"])
        self.assertEqual(3, len(client.calls))
        self.assertEqual("not_verified_by_schema_or_agreement", quality["semantic_correctness"])

    def test_tied_semantic_vote_is_independent_of_sample_order(self) -> None:
        inbound, outbound = policy_payload(), policy_payload(operator="fan_out")
        first, _ = LLMAugmentor(SequenceClient([inbound, outbound])).compile_detection_policy(
            "rule", samples=2)
        second, _ = LLMAugmentor(SequenceClient([outbound, inbound])).compile_detection_policy(
            "rule", samples=2)
        self.assertEqual(first.rules[0].operator, second.rules[0].operator)

    def test_invalid_samples_are_counted_in_valid_sample_rate(self) -> None:
        invalid = policy_payload()
        invalid["rules"] = []
        client = SequenceClient([invalid, policy_payload()])
        _, quality = LLMAugmentor(client).compile_detection_policy(
            "rule", samples=2, max_repair_attempts=0)
        self.assertEqual(1, quality["valid_samples"])
        self.assertEqual(.5, quality["valid_sample_rate"])
        self.assertEqual(1., quality["semantic_agreement"])

    def test_repair_count_includes_failed_candidates(self) -> None:
        invalid = policy_payload()
        invalid["rules"] = []
        client = SequenceClient([invalid, invalid, policy_payload()])
        _, quality = LLMAugmentor(client).compile_detection_policy(
            "rule", samples=2, max_repair_attempts=1)
        self.assertEqual(3, len(client.calls))
        self.assertEqual(1, quality["valid_samples"])
        self.assertEqual(1, quality["repair_attempts"])

    def test_unsupported_sentinel_aborts_without_repair_or_fallback(self) -> None:
        client = SequenceClient([{"unsupported": True}, policy_payload()])
        with self.assertRaisesRegex(LLMError, "outside"):
            LLMAugmentor(client).compile_detection_policy("需要任意长度环路", samples=2)
        self.assertEqual(1, len(client.calls))
        self.assertEqual(1, len(client.values))

    def test_provider_402_aborts_without_repair_or_default_fallback(self) -> None:
        client = SequenceClient([LLMError("LLM HTTP 402: insufficient balance"), policy_payload()])
        with self.assertRaisesRegex(LLMError, "402"):
            LLMAugmentor(client).compile_detection_policy("rule", samples=2)
        self.assertEqual(1, len(client.calls))
        self.assertEqual(1, len(client.values))

    def test_provider_failure_does_not_return_an_earlier_valid_candidate(self) -> None:
        client = SequenceClient([policy_payload(), LLMError("LLM HTTP 402: insufficient balance")])
        with self.assertRaisesRegex(LLMError, "402"):
            LLMAugmentor(client).compile_detection_policy("rule", samples=2)
        self.assertEqual(2, len(client.calls))

    def test_parameter_limits_are_strict_integers_and_validate_before_client_call(self) -> None:
        for samples in (0, 6, -1, 1.5, True, None):
            with self.subTest(parameter="samples", value=samples):
                client = SequenceClient([policy_payload()])
                with self.assertRaises(ValueError):
                    LLMAugmentor(client).compile_detection_policy("rule", samples=samples)
                self.assertEqual([], client.calls)
        for repairs in (-1, 2, .5, True, None):
            with self.subTest(parameter="repairs", value=repairs):
                client = SequenceClient([policy_payload()])
                with self.assertRaises(ValueError):
                    LLMAugmentor(client).compile_detection_policy("rule", max_repair_attempts=repairs)
                self.assertEqual([], client.calls)

    def test_maximum_sample_count_and_zero_repairs_are_allowed(self) -> None:
        client = SequenceClient([policy_payload() for _ in range(5)])
        _, quality = LLMAugmentor(client).compile_detection_policy(
            "rule", samples=5, max_repair_attempts=0)
        self.assertEqual(5, len(client.calls))
        self.assertEqual(5, quality["valid_samples"])
        self.assertEqual(0, quality["repair_attempts"])


class AtomicExtractionConstraintsTests(unittest.TestCase):
    def extract(self, value: object):
        return LLMAugmentor(SequenceClient([value])).extract_case(case_id="USER-CASE", text="explicit text")

    def test_valid_extraction_keeps_source_fields_without_derived_outputs(self) -> None:
        payload = extraction_payload()
        payload["case_id"] = "MODEL-CANNOT-OVERRIDE"
        case = self.extract(payload)
        self.assertEqual("USER-CASE", case.case_id)
        self.assertEqual({"A", "B"}, set(case.entities))
        self.assertEqual({"TX1"}, set(case.atomic_events))
        self.assertEqual("source-line-1", case.atomic_events["TX1"].source_record_id)
        self.assertEqual({"amount": 100., "currency": "USD"}, case.atomic_events["TX1"].attributes)
        self.assertEqual({}, case.risk_events)
        self.assertEqual([], case.atomic_relations)
        self.assertEqual([], case.risk_relations)
        self.assertEqual({}, case.metadata)

    def test_empty_derived_collections_are_allowed(self) -> None:
        payload = extraction_payload()
        payload.update({"risk_events": [], "atomic_relations": [], "risk_relations": [], "metadata": {}})
        self.assertEqual({}, self.extract(payload).risk_events)

    def test_derived_risks_relations_and_metadata_are_rejected(self) -> None:
        injected = {
            "risk_events": [{"risk_event_id": "model-inferred-risk"}],
            "atomic_relations": [{"relation_type": "FUNDS_FLOW_TO"}],
            "risk_relations": [{"relation_type": "RELATED_TO"}],
            "metadata": {"evaluation_labels": {"TX1": 1}},
        }
        for field, value in injected.items():
            with self.subTest(field=field):
                payload = extraction_payload()
                payload[field] = value
                with self.assertRaisesRegex(LLMError, "may not inject"):
                    self.extract(payload)

    def test_nonempty_provenance_metadata_is_also_rejected(self) -> None:
        payload = extraction_payload()
        payload["metadata"] = {"source": "model-claimed-source"}
        with self.assertRaises(LLMError):
            self.extract(payload)

    def test_known_label_attributes_are_rejected_on_entities_and_events(self) -> None:
        for collection in ("entities", "atomic_events"):
            for label in ("is_laundering", "Is Laundering", "isFraud", "fraudStep", "label", "evaluation_labels"):
                with self.subTest(collection=collection, label=label):
                    payload = extraction_payload()
                    payload[collection][0]["attributes"][label] = 1
                    with self.assertRaisesRegex(LLMError, "label attributes"):
                        self.extract(payload)

    def test_label_spelling_variations_cannot_bypass_isolation(self) -> None:
        for label in ("Label", "LABEL", "IS_LAUNDERING", "isfraud", "Evaluation Labels"):
            with self.subTest(label=label):
                payload = extraction_payload()
                payload["atomic_events"][0]["attributes"][label] = 1
                with self.assertRaises(LLMError):
                    self.extract(payload)

    def test_nested_labels_cannot_enter_extracted_attributes(self) -> None:
        for collection in ("entities", "atomic_events"):
            with self.subTest(collection=collection):
                payload = extraction_payload()
                payload[collection][0]["attributes"]["details"] = {"evaluation_labels": {"TX1": 1}}
                with self.assertRaises(LLMError):
                    self.extract(payload)

    def test_unknown_top_level_fields_are_rejected(self) -> None:
        for field in ("label", "evaluation_labels", "summary", "criminal_conclusion"):
            with self.subTest(field=field):
                payload = extraction_payload()
                payload[field] = "unsupported"
                with self.assertRaisesRegex(LLMError, "unsupported"):
                    self.extract(payload)

    def test_unknown_entity_and_event_fields_are_rejected(self) -> None:
        for collection in ("entities", "atomic_events"):
            for field in ("label", "risk_score", "metadata", "generated_conclusion"):
                with self.subTest(collection=collection, field=field):
                    payload = extraction_payload()
                    payload[collection][0][field] = "unsupported"
                    with self.assertRaisesRegex(LLMError, "unsupported"):
                        self.extract(payload)

    def test_collections_and_identifiers_are_validated(self) -> None:
        for collection in ("entities", "atomic_events"):
            for value in (None, {}, "not-a-list"):
                with self.subTest(collection=collection, shape=type(value).__name__):
                    payload = extraction_payload()
                    payload[collection] = value
                    with self.assertRaises(LLMError):
                        self.extract(payload)
            field = "entity_id" if collection == "entities" else "event_id"
            for value in (None, 123):
                with self.subTest(collection=collection, identifier=value):
                    payload = extraction_payload()
                    payload[collection][0][field] = value
                    with self.assertRaises(LLMError):
                        self.extract(payload)

    def test_duplicate_entity_or_event_identifiers_are_rejected(self) -> None:
        for collection in ("entities", "atomic_events"):
            with self.subTest(collection=collection):
                payload = extraction_payload()
                payload[collection].append(copy.deepcopy(payload[collection][0]))
                with self.assertRaisesRegex(LLMError, "duplicate"):
                    self.extract(payload)

    def test_empty_or_whitespace_identifiers_are_rejected(self) -> None:
        for collection in ("entities", "atomic_events"):
            field = "entity_id" if collection == "entities" else "event_id"
            for value in ("", "   "):
                with self.subTest(collection=collection, identifier=repr(value)):
                    payload = extraction_payload()
                    payload[collection][0][field] = value
                    if collection == "entities":
                        payload["atomic_events"][0]["roles"]["sender"] = value
                    with self.assertRaises(LLMError):
                        self.extract(payload)

    def test_nonempty_nondict_attributes_are_rejected(self) -> None:
        for collection in ("entities", "atomic_events"):
            payload = extraction_payload()
            payload[collection][0]["attributes"] = ["not-an-attribute-object"]
            with self.subTest(collection=collection), self.assertRaises(LLMError):
                self.extract(payload)

    def test_falsy_nondict_attributes_are_not_silently_coerced_to_empty(self) -> None:
        for collection in ("entities", "atomic_events"):
            for invalid in ([], 0, False, ""):
                with self.subTest(collection=collection, kind=type(invalid).__name__):
                    payload = extraction_payload()
                    payload[collection][0]["attributes"] = invalid
                    with self.assertRaises(LLMError):
                        self.extract(payload)

    def test_derived_empty_fields_still_require_correct_container_types(self) -> None:
        for field in ("risk_events", "atomic_relations", "risk_relations"):
            for invalid in ({}, 0, False, ""):
                with self.subTest(field=field, kind=type(invalid).__name__):
                    payload = extraction_payload()
                    payload[field] = invalid
                    with self.assertRaises(LLMError):
                        self.extract(payload)
        for invalid in ([], 0, False, ""):
            with self.subTest(field="metadata", kind=type(invalid).__name__):
                payload = extraction_payload()
                payload["metadata"] = invalid
                with self.assertRaises(LLMError):
                    self.extract(payload)

    def test_incomplete_atomic_event_raises_controlled_error(self) -> None:
        for missing in ("timestamp", "action_type", "roles"):
            with self.subTest(missing=missing):
                payload = extraction_payload()
                del payload["atomic_events"][0][missing]
                with self.assertRaises(LLMError):
                    self.extract(payload)

    def test_invalid_timestamp_and_dangling_roles_raise_controlled_error(self) -> None:
        invalid_time = extraction_payload()
        invalid_time["atomic_events"][0]["timestamp"] = "not-a-date"
        invalid_roles = extraction_payload()
        invalid_roles["atomic_events"][0]["roles"]["sender"] = "UNKNOWN"
        for index, value in enumerate((invalid_time, invalid_roles)):
            with self.subTest(index=index), self.assertRaises(LLMError):
                self.extract(value)

    def test_provider_402_is_not_replaced_with_empty_case(self) -> None:
        client = SequenceClient([LLMError("LLM HTTP 402: insufficient balance")])
        with self.assertRaisesRegex(LLMError, "402"):
            LLMAugmentor(client).extract_case(case_id="USER-CASE", text="explicit text")
        self.assertEqual(1, len(client.calls))


if __name__ == "__main__":
    unittest.main()
