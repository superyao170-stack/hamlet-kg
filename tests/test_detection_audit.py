from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from hamletkg.detection.audit import audit_detection_case
from hamletkg.models import AtomicEvent, CaseGraph, Entity


START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def transfer(event_id, seconds=0, *, timestamp=None, attributes=None, roles=None, source_record_id=None):
    values = {
        "amount": 100.0, "received_amount": 100.0,
        "paid_currency": "USD", "received_currency": "USD", "channel": "ACH",
    }
    if attributes is not None:
        values = attributes
    return AtomicEvent(
        event_id, "TRANSFER", roles or {"sender": "A", "receiver": "B"},
        timestamp or START + timedelta(seconds=seconds), values, source_record_id,
    )


def graph(events, metadata=None):
    entities = {name: Entity(name, "ACCOUNT") for name in ("A", "B", "C", "UNUSED")}
    return CaseGraph("AUDIT", entities, {event.event_id: event for event in events}, metadata=metadata or {})


class DetectionAuditTests(unittest.TestCase):
    def test_counts_distinguish_atomic_events_candidates_and_accounts(self):
        transfer_event = transfer("T1")
        other_event = AtomicEvent("LOGIN", "LOGIN", {"actor": "C"}, START - timedelta(days=1))
        result = audit_detection_case(graph([other_event, transfer_event]))
        self.assertEqual({
            "atomic_events": 2, "transfer_candidates": 1, "account_entities": 4,
            "participating_accounts": 2, "unknown_participant_ids": 0,
        }, result["counts"])
        self.assertEqual("2026-01-01T00:00:00Z", result["time"]["start"])
        self.assertEqual("2025-12-31T00:00:00Z", result["time"]["all_atomic_start"])

    def test_timezone_normalization_groups_the_same_instant(self):
        result = audit_detection_case(graph([
            transfer("UTC", timestamp="2026-01-01T00:00:00Z"),
            transfer("OFFSET", timestamp="2026-01-01T08:00:00+08:00"),
            transfer("NAIVE", timestamp="2026-01-01T00:00:00"),
        ]))
        self.assertEqual("UTC", result["time"]["timezone"])
        self.assertEqual(1, result["time"]["distinct_timestamp_groups"])
        self.assertEqual(1, result["time"]["same_timestamp_groups"])
        self.assertEqual(3, result["time"]["transactions_in_same_timestamp_groups"])
        self.assertEqual(1.0, result["time"]["same_timestamp_transaction_ratio"])
        self.assertEqual(3, result["duplicates"]["transactions_in_groups"])

    def test_business_duplicates_ignore_ids_source_rows_and_labels_without_removing(self):
        first = transfer("FIRST", source_record_id="1")
        attrs = {
            "received_currency": " usd ", "paid_currency": "usd", "channel": " ach ",
            "received_amount": "100.00", "amount": 100,
            "event_id": "external-ID", "source_line_number": 200, "is_laundering": 1,
        }
        second = transfer("SECOND", attributes=attrs, source_record_id="200")
        case = graph([first, second], {"evaluation_labels": {"FIRST": 0, "SECOND": 1}})
        result = audit_detection_case(case)
        self.assertEqual(1, result["duplicates"]["business_field_groups"])
        self.assertEqual(1, result["duplicates"]["extra_transactions"])
        self.assertTrue(result["duplicates"]["retained"])
        self.assertEqual(["FIRST", "SECOND"], list(case.atomic_events))
        self.assertEqual(2, result["counts"]["transfer_candidates"])

    def test_different_time_roles_or_business_attributes_are_not_exact_duplicates(self):
        base = transfer("BASE", attributes={**transfer("X").attributes, "details": {"branch": "north"}})
        events = [
            base, replace(base, event_id="TIME", timestamp=START + timedelta(seconds=1)),
            replace(base, event_id="ROLE", roles={"sender": "A", "receiver": "C"}),
            replace(base, event_id="DETAIL", attributes={**base.attributes, "details": {"branch": "south"}}),
            replace(base, event_id="EXTRA_ROLE", roles={**base.roles, "id": "C"}),
        ]
        self.assertEqual(0, audit_detection_case(graph(events))["duplicates"]["business_field_groups"])

    def test_input_order_and_same_timestamp_statistics_are_separate(self):
        result = audit_detection_case(graph([
            transfer("LATE", 10), transfer("EARLY", 0), transfer("TIE1", 5), transfer("TIE2", 5),
        ]))
        time = result["time"]
        self.assertFalse(time["input_order_nondecreasing"])
        self.assertEqual(1, time["adjacent_time_inversions"])
        self.assertEqual(1, time["same_timestamp_groups"])
        self.assertEqual(2, time["transactions_in_same_timestamp_groups"])
        self.assertEqual(0.5, time["same_timestamp_transaction_ratio"])

    def test_missing_labels_stay_unknown_instead_of_negative(self):
        result = audit_detection_case(graph([transfer("P"), transfer("U", 1)], {
            "evaluation_labels": {"P": 1}, "is_laundering": {"U": 0},
        }))
        labels = result["labels"]
        self.assertEqual(1, labels["labeled"])
        self.assertEqual(1, labels["unlabeled"])
        self.assertEqual(1, labels["positive"])
        self.assertEqual(0, labels["negative"])
        self.assertEqual(1.0, labels["positive_rate_among_labeled"])
        self.assertEqual(0.5, labels["coverage"])
        self.assertFalse(labels["unlabeled_is_negative"])
        self.assertFalse(labels["used_as_features"])

    def test_invalid_and_entity_labels_do_not_become_transaction_labels(self):
        events = [transfer("GOOD"), transfer("BOOL"), transfer("STRING"), transfer("FLOAT"), transfer("BAD")]
        result = audit_detection_case(graph(events, {"evaluation_labels": {
            "GOOD": 0, "BOOL": True, "STRING": "1", "FLOAT": 1.0, "BAD": 2,
            "A": 1, "UNKNOWN": 0,
        }}))
        labels = result["labels"]
        self.assertEqual(1, labels["labeled"])
        self.assertEqual(4, labels["invalid"])
        self.assertEqual(4, labels["unlabeled"])
        self.assertEqual(1, labels["negative"])
        self.assertEqual(2, labels["unmatched_entries"])
        self.assertEqual(1, labels["entity_level_entries"])

    def test_non_mapping_labels_are_reported_without_failing(self):
        result = audit_detection_case(graph([transfer("T1")], {"evaluation_labels": [1]}))
        self.assertFalse(result["labels"]["mapping_valid"])
        self.assertEqual(1, result["labels"]["unlabeled"])
        self.assertIn("invalid_evaluation_label_mapping", {risk["code"] for risk in result["risks"]})

    def test_currency_and_payment_format_distributions_are_normalized(self):
        attrs = {**transfer("X").attributes, "paid_currency": " eur ", "received_currency": "jpy", "channel": " wire "}
        result = audit_detection_case(graph([transfer("USD"), transfer("FX", 1, attributes=attrs)]))
        self.assertEqual({"EUR": 1, "USD": 1}, result["currencies"]["paid"])
        self.assertEqual({"JPY": 1, "USD": 1}, result["currencies"]["received"])
        self.assertEqual({"EUR->JPY": 1, "USD->USD": 1}, result["currencies"]["pairs"])
        self.assertEqual({"ACH": 1, "WIRE": 1}, result["payment_formats"])
        self.assertTrue(result["currencies"]["amounts_are_not_aggregated_across_currencies"])

    def test_correspondence_compares_only_known_same_currency_finite_amounts(self):
        equal = transfer("EQ")
        fee = transfer("FEE", 1, attributes={**equal.attributes, "received_amount": 99})
        fx = transfer("FX", 2, attributes={**equal.attributes, "received_currency": "EUR", "received_amount": 90})
        missing = transfer("MISSING", 3, attributes={"amount": 100, "currency": "USD"})
        pairs = audit_detection_case(graph([equal, fee, fx, missing]))["amounts"]["paid_received_correspondence"]
        self.assertEqual(3, pairs["same_currency"])
        self.assertEqual(1, pairs["cross_currency"])
        self.assertEqual(2, pairs["same_currency_finite_pairs"])
        self.assertEqual(1, pairs["same_currency_equal_amounts"])
        self.assertEqual(1, pairs["same_currency_different_amounts"])
        self.assertEqual(2, pairs["unavailable_amount_comparisons"])

    def test_missing_fields_are_counted_and_no_received_amount_is_inferred(self):
        event = transfer("MISSING", attributes={"amount": 100}, roles={"sender": "A"})
        result = audit_detection_case(graph([event]))
        for field in ("receiver", "received_amount", "paid_currency", "received_currency", "channel"):
            self.assertEqual(1, result["missing_values"][field])
        self.assertEqual(1, result["amounts"]["received_amount"]["missing"])
        self.assertEqual(1, result["amounts"]["paid_received_correspondence"]["unknown_currency"])

    def test_zero_negative_nonfinite_and_invalid_amounts_have_distinct_counts(self):
        values = [0, -10, float("nan"), float("inf"), "-Infinity", "broken", True, {}, None, "  ", "1.50"]
        events = [transfer(f"T{index}", index, attributes={**transfer("X").attributes, "amount": value}) for index, value in enumerate(values)]
        result = audit_detection_case(graph(events))
        amount = result["amounts"]["amount"]
        self.assertEqual((1, 1, 3, 3, 2, 1), tuple(amount[field] for field in ("zero", "negative", "non_finite", "invalid", "missing", "positive")))
        self.assertEqual(8, result["invalid_values"]["amount"])
        json.dumps(result, allow_nan=False)

    def test_invalid_role_currency_channel_and_timestamp_are_reported(self):
        event = transfer("BAD", roles={"sender": 42, "receiver": "UNKNOWN"}, attributes={
            "amount": 10, "received_amount": 10, "paid_currency": 9, "received_currency": "USD", "channel": [],
        })
        # Normal loaders reject invalid timestamps; audit direct/corrupted graphs
        # without invoking validate() or concealing other field-quality risks.
        object.__setattr__(event, "timestamp", "unparseable")
        result = audit_detection_case(graph([event]))
        for field in ("sender", "paid_currency", "channel", "timestamp"):
            self.assertEqual(1, result["invalid_values"][field])
        self.assertEqual(1, result["counts"]["unknown_participant_ids"])
        self.assertFalse(result["time"]["input_order_complete"])
        self.assertIsNone(result["time"]["start"])

    def test_legacy_currency_fallback_is_explicit_and_does_not_hide_missing_fields(self):
        result = audit_detection_case(graph([transfer("LEGACY", attributes={"amount": 10, "currency": " usd "})]))
        self.assertEqual({"USD": 1}, result["currencies"]["paid"])
        self.assertEqual({"paid": 1, "received": 1}, result["currencies"]["legacy_currency_fallbacks"])
        self.assertEqual(1, result["missing_values"]["paid_currency"])
        self.assertEqual(1, result["missing_values"]["received_amount"])

    def test_source_hash_and_metadata_allowlist_do_not_assert_authenticity(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "public.csv"
            content = b"one,two\n1,2\n"
            source.write_bytes(content)
            result = audit_detection_case(graph([transfer("T1")], {
                "dataset": "claimed real transactions", "dataset_version": "v1",
                "labels_are_features": False, "api_key": "secret-key",
                "credentials": {"password": "secret-password"}, "source_url": "https://user:secret@example.com/data",
                "injected_chains": ["T1"],
            }), source_path=source)
        self.assertEqual(hashlib.sha256(content).hexdigest(), result["source"]["sha256"])
        self.assertEqual(len(content), result["source"]["size_bytes"])
        self.assertEqual("public.csv", result["source"]["filename"])
        self.assertEqual("computed", result["source"]["digest_status"])
        self.assertEqual({"dataset", "dataset_version", "labels_are_features"}, set(result["source"]["dataset_metadata"]))
        self.assertNotIn("secret", json.dumps(result))
        self.assertEqual("unknown", result["source"]["synthetic_or_real"])
        self.assertEqual("unknown", result["source"]["injected_chain_membership"])

    def test_metadata_source_path_is_never_followed_and_missing_explicit_file_is_reported(self):
        case = graph([transfer("T1")], {"source_file": "/never/read/metadata/path.csv"})
        self.assertEqual("not_provided", audit_detection_case(case)["source"]["digest_status"])
        with tempfile.TemporaryDirectory() as directory:
            result = audit_detection_case(case, source_path=Path(directory) / "missing.csv")
        self.assertEqual("unreadable", result["source"]["digest_status"])
        self.assertIsNone(result["source"]["sha256"])
        self.assertIn("source_file_unreadable", {risk["code"] for risk in result["risks"]})

    def test_audit_and_returned_metadata_do_not_mutate_input(self):
        case = graph([transfer("T1"), transfer("T2")], {"dataset": "example", "evaluation_labels": {"T1": 1}})
        snapshot = copy.deepcopy(case)
        result = audit_detection_case(case)
        result["source"]["dataset_metadata"]["dataset"] = "changed"
        result["duplicates"]["examples"][0]["event_ids"].clear()
        self.assertEqual(snapshot, case)

    def test_label_changes_do_not_affect_other_audit_measurements(self):
        case = graph([transfer("T1"), transfer("T2")])
        unlabeled = audit_detection_case(case)
        case.metadata["evaluation_labels"] = {"T1": 0, "T2": 1}
        labeled = audit_detection_case(case)
        for field in ("counts", "time", "currencies", "payment_formats", "amounts", "duplicates"):
            self.assertEqual(unlabeled[field], labeled[field])

    def test_decimal_normalization_keeps_precision_for_duplicate_comparison(self):
        values = [Decimal("123456789012345678901234567890.00"), "123456789012345678901234567890", Decimal("123456789012345678901234567891")]
        events = [transfer(f"T{index}", attributes={**transfer("X").attributes, "amount": value}) for index, value in enumerate(values)]
        result = audit_detection_case(graph(events))
        self.assertEqual(1, result["duplicates"]["business_field_groups"])
        self.assertEqual(1, result["duplicates"]["extra_transactions"])

    def test_unsupported_business_objects_do_not_create_false_duplicates(self):
        events = [transfer(f"T{index}", attributes={**transfer("X").attributes, "custom": object()}) for index in range(2)]
        result = audit_detection_case(graph(events))
        self.assertEqual(0, result["duplicates"]["business_field_groups"])
        self.assertEqual(2, result["duplicates"]["uncheckable_transactions"])
        json.dumps(result, allow_nan=False)

    def test_malformed_business_structures_are_reported_without_false_duplicates(self):
        first, second = transfer("FIRST"), transfer("SECOND")
        object.__setattr__(first, "attributes", None)
        object.__setattr__(second, "attributes", [])
        object.__setattr__(second, "roles", None)
        result = audit_detection_case(graph([first, second]))
        self.assertEqual(0, result["duplicates"]["business_field_groups"])
        self.assertEqual(2, result["duplicates"]["uncheckable_transactions"])
        self.assertEqual(2, result["missing_values"]["amount"])
        risks = {risk["code"]: risk["count"] for risk in result["risks"]}
        self.assertEqual(2, risks["malformed_attributes"])
        self.assertEqual(1, risks["malformed_roles"])

    def test_empty_case_has_no_invented_ranges_or_label_rates(self):
        result = audit_detection_case(graph([]))
        self.assertEqual(0, result["counts"]["transfer_candidates"])
        self.assertIsNone(result["time"]["start"])
        self.assertIsNone(result["time"]["same_timestamp_transaction_ratio"])
        self.assertIsNone(result["labels"]["positive_rate_among_labeled"])
        self.assertIsNone(result["labels"]["coverage"])


if __name__ == "__main__":
    unittest.main()
