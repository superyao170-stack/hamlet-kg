from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from hamletkg.adapters.ibm_aml import case_from_transaction_csv, iter_transaction_csv
from hamletkg.detection.dataset import PreparedDataset, prepare_detection_dataset


HEADER = [
    "Timestamp", "From Bank", "Account", "To Bank", "Account", "Amount Paid",
    "Payment Currency", "Amount Received", "Receiving Currency", "Payment Format", "Is Laundering",
]


def row(timestamp="2022/09/01 00:00", *, sender="A", receiver="B", label="0", amount="100"):
    return [timestamp, "1", sender, "2", receiver, amount, "USD", "95", "EUR", "Wire", label]


class PreparedDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.source = self.directory / "source.csv"
        self.output = self.directory / "prepared"

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, rows, header=HEADER):
        with self.source.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)

    def prepare(self, **kwargs):
        return prepare_detection_dataset(self.source, self.output, case_id="TEST", **kwargs)

    def manifest(self):
        return json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))

    def test_disk_rows_equal_original_adapter_and_labels_are_separate(self):
        self.write([row(label="1"), row("2022/09/01 00:01", sender="B", label=""), row("2022/09/01 00:02")])
        expected = case_from_transaction_csv(self.source, case_id="TEST", build_relations=False)
        with patch("hamletkg.adapters.ibm_aml.CaseGraph", side_effect=AssertionError("must stream")):
            manifest = self.prepare(batch_size=1)
        dataset = PreparedDataset(self.output)
        dataset.validate_integrity()
        actual = {event.event_id: event.to_dict() for event in dataset.iter_events()}
        self.assertEqual({key: value.to_dict() for key, value in expected.atomic_events.items()}, actual)
        actual_labels = {key: label for key, label in dataset.iter_labels() if label is not None}
        self.assertEqual(expected.metadata["evaluation_labels"], actual_labels)
        self.assertEqual({"read_rows": 3, "kept_transactions": 3, "positive": 1, "negative": 1, "unlabeled": 1}, manifest["counts"])
        self.assertFalse(manifest["labels_are_features"])
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), manifest["source"]["sha256"])
        for event in dataset.iter_events():
            self.assertEqual({"amount", "received_amount", "paid_currency", "received_currency", "channel"}, set(event.attributes))
        with sqlite3.connect(dataset.database_path) as connection:
            columns = [entry[1] for entry in connection.execute("PRAGMA table_info(transactions)")]
            self.assertNotIn("label", columns)
            self.assertEqual(3, connection.execute("SELECT COUNT(*) FROM labels").fetchone()[0])

    def test_original_row_identity_formula_survives_disk_sort_and_duplicates(self):
        self.write([row("2022/09/01 00:02"), row(), row(), row("2022/09/01 00:01")])
        self.prepare(batch_size=2)
        events = list(PreparedDataset(self.output).iter_events())
        self.assertEqual(4, len(set(event.event_id for event in events)))
        self.assertEqual({"2", "3"}, {event.source_record_id for event in events[:2]})
        self.assertEqual(["4", "1"], [event.source_record_id for event in events[2:]])
        for event in events:
            raw = ["2022/09/01 00:02", "2022/09/01 00:00", "2022/09/01 00:00", "2022/09/01 00:01"][int(event.source_record_id) - 1]
            identity = "|".join(["TEST", event.source_record_id, "ACCOUNT:1:A", "ACCOUNT:2:B", raw])
            self.assertEqual("TX-" + hashlib.sha256(identity.encode()).hexdigest()[:16], event.event_id)

    def test_microseconds_offsets_and_ties_use_exact_utc_sort(self):
        self.write([
            row("2022-09-01T00:00:00.000002Z"),
            row("2022-09-01T08:00:00.000001+08:00"),
            row("2022-09-01T00:00:00.000001Z", sender="C"),
            row("1960-09-01T00:00:00.999999Z"),
        ])
        self.prepare()
        dataset = PreparedDataset(self.output)
        events = list(dataset.iter_events())
        self.assertEqual([999999, 1, 1, 2], [event.timestamp.microsecond for event in events])
        self.assertEqual(sorted(event.event_id for event in events[1:3]), [event.event_id for event in events[1:3]])
        self.assertTrue(all(event.timestamp.tzinfo == timezone.utc for event in events))
        self.assertEqual([1, 2, 1], [count for _, count in dataset.timestamp_groups()])

    def test_prefix_limit_precedes_time_filter_and_keeps_original_ids(self):
        self.write([row("2022/09/01 00:03"), row("2022/09/01 00:01", label="1"), row(), row("2022/09/01 00:01", label="1")])
        manifest = self.prepare(limit=3, after="2022-09-01T00:01:00Z", before="2022-09-01T00:03:00Z")
        event = next(PreparedDataset(self.output).iter_events())
        self.assertEqual("2", event.source_record_id)
        self.assertEqual(3, manifest["counts"]["read_rows"])
        self.assertEqual(1, manifest["counts"]["kept_transactions"])
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), manifest["source"]["sha256"])
        original = list(iter_transaction_csv(self.source, case_id="TEST"))[1][0]
        self.assertEqual(original, event)

    def test_read_only_time_view_applies_to_all_readers_without_changing_manifest(self):
        self.write([row(), row("2022/09/01 00:01", label="1"), row("2022/09/01 00:01", label=""), row("2022/09/01 00:02")])
        original = self.prepare()
        full = PreparedDataset(self.output)
        events = list(full.iter_events())
        view = PreparedDataset(self.output, after="2022-09-01T08:01:00+08:00", before="2022-09-01T00:02:00Z")
        view.validate_integrity()
        self.assertEqual(original, view.manifest)
        self.assertEqual({"after": "2022-09-01T00:01:00Z", "before": "2022-09-01T00:02:00Z", "transactions": 2,
                          "positive": 1, "negative": 0, "unlabeled": 1}, view.selection)
        self.assertEqual({"2", "3"}, {event.source_record_id for event in view.iter_events()})
        selected_ids = {event.event_id for event in view.iter_events()}
        self.assertEqual(selected_ids, {event_id for event_id, _ in view.iter_labels()})
        self.assertEqual(selected_ids, set(view.fetch_transactions(event.event_id for event in events)))
        self.assertEqual([(datetime(2022, 9, 1, 0, 1, tzinfo=timezone.utc), 2)], view.timestamp_groups())
        self.assertEqual(original, self.manifest())

    def test_view_boundaries_are_independent_of_labels_and_include_whole_ties(self):
        self.write([row(), row("2022/09/01 00:01", label="1"), row("2022/09/01 00:01", label="0"), row("2022/09/01 00:02")])
        self.prepare()
        view = PreparedDataset(self.output, after="2022-09-01T00:01:00Z")
        self.assertEqual(3, view.selection["transactions"])
        self.assertEqual(1, view.selection["positive"])
        self.assertEqual(2, view.selection["negative"])
        earlier = PreparedDataset(self.output, before="2022-09-01T00:01:00Z")
        self.assertEqual(["1"], [event.source_record_id for event in earlier.iter_events()])

    def test_empty_declared_interval_and_header_only_input_remain_valid(self):
        for rows in ([], [row()]):
            with self.subTest(rows=rows):
                self.write(rows)
                output = self.directory / ("empty" if not rows else "excluded")
                manifest = prepare_detection_dataset(self.source, output, after="2023-01-01T00:00:00Z")
                dataset = PreparedDataset(output)
                dataset.validate_integrity()
                self.assertEqual(0, dataset.selection["transactions"])
                self.assertEqual([], list(dataset.iter_events()))
                self.assertEqual([], list(dataset.iter_labels()))
                self.assertEqual([], dataset.timestamp_groups())
                self.assertIsNone(manifest["time"]["start"])

    def test_missing_label_column_keeps_every_label_unknown(self):
        self.write([row()[:-1], row("2022/09/01 00:01")[:-1]], header=HEADER[:-1])
        self.prepare()
        dataset = PreparedDataset(self.output)
        self.assertEqual(2, dataset.selection["unlabeled"])
        self.assertTrue(all(label is None for _, label in dataset.iter_labels()))

    def test_bank_attributes_with_colons_preserve_original_adapter_behavior(self):
        values = row()
        values[1], values[3] = "BANK:ONE", "BANK:TWO"
        self.write([values])
        case = case_from_transaction_csv(self.source, case_id="TEST", build_relations=False)
        self.assertEqual({"bank_id": "BANK:ONE"}, case.entities["ACCOUNT:BANK:ONE:A"].attributes)
        self.assertEqual({"bank_id": "BANK:TWO"}, case.entities["ACCOUNT:BANK:TWO:B"].attributes)

    def test_validation_failure_marks_failed_and_never_publishes_database(self):
        self.write([row(), row("2022/09/01 00:01", amount="NaN")])
        with self.assertRaisesRegex(ValueError, "finite"):
            self.prepare(batch_size=1)
        self.assertEqual("failed", self.manifest()["status"])
        self.assertEqual("ValueError", self.manifest()["error_type"])
        self.assertFalse((self.output / "transactions.sqlite").exists())
        self.assertFalse((self.output / "transactions.sqlite.partial").exists())
        with self.assertRaisesRegex(ValueError, "complete"):
            PreparedDataset(self.output)

    def test_out_of_interval_invalid_row_is_still_rejected(self):
        self.write([row(), row("2025/01/01 00:00", label="2")])
        with self.assertRaisesRegex(ValueError, "label"):
            self.prepare(before="2023-01-01T00:00:00Z")

    def test_invalid_row_after_prefix_is_not_part_of_validation(self):
        self.write([row(), row(amount="invalid")])
        manifest = self.prepare(limit=1)
        self.assertEqual(1, manifest["counts"]["read_rows"])
        self.assertEqual(1, manifest["counts"]["kept_transactions"])

    def test_keyboard_interrupt_marks_interrupted_and_cleans_committed_partial_batches(self):
        self.write([row()])
        event = next(iter_transaction_csv(self.source, case_id="TEST"))

        def interrupted(*args, **kwargs):
            yield event
            raise KeyboardInterrupt()

        with patch("hamletkg.detection.dataset.iter_transaction_csv", side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                self.prepare(batch_size=1)
        self.assertEqual("interrupted", self.manifest()["status"])
        self.assertEqual(["manifest.json"], [path.name for path in self.output.iterdir()])

    def test_changed_source_is_rejected_after_streaming(self):
        self.write([row()])
        original = iter_transaction_csv

        def changed(*args, **kwargs):
            yield from original(*args, **kwargs)
            self.source.write_bytes(self.source.read_bytes() + b"\n")

        with patch("hamletkg.detection.dataset.iter_transaction_csv", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "changed"):
                self.prepare()
        self.assertEqual("failed", self.manifest()["status"])

    def test_changed_source_is_detected_even_if_size_and_mtime_are_restored(self):
        self.write([row()])
        original = iter_transaction_csv
        original_stat = self.source.stat()

        def changed(*args, **kwargs):
            yield from original(*args, **kwargs)
            self.source.write_bytes(self.source.read_bytes().replace(b",100,", b",200,"))
            os.utime(self.source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

        with patch("hamletkg.detection.dataset.iter_transaction_csv", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "changed"):
                self.prepare()
        self.assertEqual("failed", self.manifest()["status"])

    def test_database_content_tamper_is_detected_by_hash(self):
        self.write([row()])
        self.prepare()
        dataset = PreparedDataset(self.output)
        with sqlite3.connect(dataset.database_path) as connection:
            connection.execute("UPDATE transactions SET amount=200")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            dataset.validate_integrity()

    def test_manifest_provenance_and_counts_tamper_are_rejected(self):
        self.write([row(label="1")])
        original = self.prepare()
        for field in ("source", "counts"):
            with self.subTest(field=field):
                manifest = json.loads(json.dumps(original))
                if field == "source":
                    manifest["source"]["sha256"] = "0" * 64
                else:
                    manifest["counts"]["positive"] = 0
                (self.output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "provenance|counts"):
                    PreparedDataset(self.output).validate_integrity()

    def test_manifest_time_bounds_tamper_is_detected(self):
        self.write([row()])
        manifest = self.prepare()
        manifest["time"]["start"] = "2000-01-01T00:00:00Z"
        (self.output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "time bounds"):
            PreparedDataset(self.output).validate_integrity()

    def test_manifest_invalid_types_are_rejected_as_controlled_errors(self):
        self.write([row()])
        original = self.prepare()
        for field, value in (("schema_version", True), ("source", []), ("counts", []), ("labels_are_features", True)):
            with self.subTest(field=field):
                manifest = {**original, field: value}
                (self.output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaises(ValueError):
                    PreparedDataset(self.output)

    def test_hash_verification_is_explicit_and_not_repeated_per_iterator(self):
        self.write([row()])
        self.prepare()
        with patch("hamletkg.detection.dataset._sha256", side_effect=AssertionError("do not rehash per reader")):
            dataset = PreparedDataset(self.output)
            self.assertEqual(1, len(list(dataset.iter_events())))
            self.assertEqual(1, len(list(dataset.iter_labels())))
            self.assertEqual(1, len(dataset.timestamp_groups()))
            self.assertEqual(1, len(dataset.fetch_transactions(event.event_id for event in dataset.iter_events())))

    def test_reads_succeed_without_source_and_connection_rejects_writes(self):
        self.write([row()])
        self.prepare()
        self.source.unlink()
        dataset = PreparedDataset(self.output)
        dataset.validate_integrity()
        self.assertEqual(1, len(list(dataset.iter_events())))
        with dataset._connect() as connection:
            with self.assertRaises(sqlite3.OperationalError):
                connection.execute("DELETE FROM transactions")

    def test_lookup_batches_more_than_sqlite_parameter_limit_without_label_reads(self):
        self.write([row(f"2022-09-01T00:00:{index % 60:02d}Z", sender=str(index)) for index in range(1100)])
        self.prepare(batch_size=123)
        dataset = PreparedDataset(self.output)
        events = list(dataset.iter_events())
        fetched = dataset.fetch_transactions(event.event_id for event in events)
        self.assertEqual({event.event_id: event for event in events}, fetched)
        self.assertEqual({}, dataset.fetch_transactions(["UNKNOWN"]))
        for invalid in ("TX-123", [None], [""], [123]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                dataset.fetch_transactions(invalid)

    def test_nonempty_output_and_invalid_options_are_rejected_before_mutating(self):
        self.write([row()])
        for kwargs in ({"limit": 0}, {"limit": True}, {"batch_size": 0}, {"batch_size": True},
                       {"after": "invalid"}, {"after": 0},
                       {"after": "2022-09-02T00:00:00Z", "before": "2022-09-01T00:00:00Z"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.prepare(**kwargs)
            self.assertFalse(self.output.exists())
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("preserve", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "new or empty"):
            self.prepare()
        self.assertEqual("preserve", marker.read_text(encoding="utf-8"))

    def test_missing_corrupt_or_path_traversal_manifest_is_rejected(self):
        self.write([row()])
        with self.assertRaisesRegex(ValueError, "manifest"):
            PreparedDataset(self.output)
        self.prepare()
        original = self.manifest()
        for value in ([], {**original, "status": "failed"},
                      {**original, "database": {**original["database"], "filename": "../elsewhere.sqlite"}}):
            with self.subTest(value=value):
                (self.output / "manifest.json").write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "complete"):
                    PreparedDataset(self.output)

    def test_view_options_are_validated_without_mutating_prepared_manifest(self):
        self.write([row()])
        original = self.prepare()
        for options in ({"after": []}, {"before": "bad"},
                        {"after": "2022-09-01T00:00:00Z", "before": "2022-09-01T00:00:00Z"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                PreparedDataset(self.output, **options)
        self.assertEqual(original, self.manifest())


if __name__ == "__main__":
    unittest.main()
