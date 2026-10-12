from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import sqlite3
import struct
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from hamletkg.detection.cache import CachedSamples, build_sample_cache
from hamletkg.detection.dataset import PreparedDataset, prepare_detection_dataset
from hamletkg.detection.metrics import chronological_split
from hamletkg.detection.policy import DetectionPolicy, DetectionRule
from hamletkg.detection import stream
from hamletkg.detection.stream import BASE_FEATURE_NAMES, StreamConfig, build_samples, iter_samples
from hamletkg.models import AtomicEvent, CaseGraph


START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def transfer(event_id, sender, receiver, seconds, amount=100, currency="USD"):
    return AtomicEvent(event_id, "TRANSFER", {"sender": sender, "receiver": receiver},
                       START + timedelta(seconds=seconds),
                       {"amount": amount, "received_amount": amount,
                        "paid_currency": currency, "received_currency": currency})


def graph(events):
    return CaseGraph("C", {}, {event.event_id: event for event in events})


class StubDataset:
    """Events and labels are separate readers, mirroring PreparedDataset."""

    def __init__(self, directory, events, labels=None):
        self.database_path = Path(directory) / "source.sqlite3"
        self.database_path.write_bytes(b"source fixture")
        self.events = sorted(events, key=lambda event: (event.timestamp, event.event_id))
        self.labels = labels or {event.event_id: index % 2 for index, event in enumerate(self.events)}
        self.manifest = {"database": {"sha256": hashlib.sha256(b"source fixture").hexdigest()},
                         "source": {"sha256": "a" * 64}}
        self.selection = {"after": None, "before": None, "transactions": len(self.events)}
        self.validated = False
        self.events_read = 0
        self.labels_read = 0

    def validate_integrity(self):
        self.validated = True

    def iter_events(self):
        self.events_read += 1
        yield from self.events

    def iter_labels(self):
        self.labels_read += 1
        # Intentionally reverse: cache labels must join IDs, not assume order.
        yield from reversed(list(self.labels.items()))


class CausalIteratorTests(unittest.TestCase):
    def test_iterator_matches_sorted_case_and_keeps_peers_hidden_across_consumption(self):
        events = [transfer("T1", "A", "B", 0), transfer("T2", "B", "C", 0),
                  transfer("T3", "B", "D", 10), transfer("T4", "D", "E", 20)]
        iterator = iter_samples(iter(events))
        first = next(iterator)
        peer = next(iterator)
        self.assertEqual(0, peer.features[BASE_FEATURE_NAMES.index("sender_in_log_count")])
        rest = list(iterator)
        self.assertEqual([first, peer, *rest], build_samples(graph(reversed(events))))
        self.assertEqual(math.log1p(1), rest[0].features[BASE_FEATURE_NAMES.index("sender_in_log_count")])

    def test_iterator_future_extension_and_exclusions_do_not_leak(self):
        events = [transfer("T1", "A", "B", 0), transfer("T2", "B", "C", 10),
                  transfer("T3", "C", "D", 20)]
        self.assertEqual(list(iter_samples(events[:2])), list(iter_samples(events))[:2])
        self.assertEqual(list(iter_samples(events[1:])), list(iter_samples(events, excluded_ids=["T1"])))

    def test_iterator_rejects_decreasing_order_and_duplicate_group_ids(self):
        for events in ([transfer("T1", "A", "B", 10), transfer("T2", "B", "C", 0)],
                       [transfer("T2", "A", "B", 0), transfer("T1", "B", "C", 0)],
                       [transfer("T1", "A", "B", 0), transfer("T1", "B", "C", 0)]):
            with self.subTest(events=events), self.assertRaisesRegex(ValueError, "strictly sorted"):
                list(iter_samples(events))

    def test_global_expiry_discards_inactive_accounts_and_keeps_inclusive_boundary(self):
        history = stream._History(StreamConfig(window_seconds=10, max_history=2))
        for index in range(50):
            history.add(stream._transfer(transfer(f"T{index}", f"A{index}", f"B{index}", 0)))
        self.assertEqual(100, len(history._expiry_heap))
        history.expire(START + timedelta(seconds=10))
        self.assertEqual(50, len(history.incoming))
        history.expire(START + timedelta(seconds=10, microseconds=1))
        self.assertEqual({}, history.incoming)
        self.assertEqual({}, history.outgoing)
        self.assertEqual([], history._expiry_heap)
        self.assertEqual(set(), history._scheduled)

    def test_scheduler_does_not_accumulate_a_node_per_repeated_transfer(self):
        history = stream._History(StreamConfig(window_seconds=5, max_history=2))
        for index in range(1000):
            timestamp = START + timedelta(seconds=index / 10)
            history.expire(timestamp)
            history.add(stream._transfer(transfer(f"T{index}", "A", "B", index / 10)))
            self.assertEqual(2, len(history._expiry_heap))
            self.assertLessEqual(len(history.incoming[("B", "USD")]), 2)

    def test_global_expiry_matches_lazy_legacy_history_results(self):
        rng = random.Random(19)
        events = [transfer(f"T{i:04}", f"A{rng.randrange(8)}", f"A{rng.randrange(8)}",
                           i * 4, rng.randrange(1, 200), rng.choice(("USD", "EUR"))) for i in range(300)]
        config = StreamConfig(window_seconds=20, max_history=4)
        actual = list(iter_samples(events, config))

        class LazyHistory(stream._History):
            def expire(self, timestamp):
                pass

        with patch.object(stream, "_History", LazyHistory):
            legacy = list(iter_samples(events, config))
        self.assertEqual(legacy, actual)


class SampleCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.events = [transfer("T1", "A", "B", 0), transfer("T2", "B", "C", 10),
                       transfer("T3", "C", "D", 20), transfer("T4", "E", "B", 20),
                       transfer("T5", "B", "F", 30), transfer("T6", "F", "G", 40)]
        self.source = StubDataset(self.directory, self.events)
        self.output = self.directory / "cache"

    def tearDown(self):
        self.temporary.cleanup()

    def build(self, **kwargs):
        build_sample_cache(self.source, self.output, **kwargs)
        return CachedSamples(self.output)

    def all_samples(self, cache, split="all", batch_size=1):
        return [sample for batch in cache.iter_batches(split, batch_size) for sample in batch]

    def test_cache_matches_samples_and_frozen_historical_member_vectors(self):
        cache = self.build()
        expected = build_samples(graph(self.events))
        self.assertEqual(expected, self.all_samples(cache, batch_size=1))
        self.assertEqual(expected, self.all_samples(cache, batch_size=1000))
        self.assertEqual(6, len(cache))
        self.assertTrue(self.source.validated)
        self.assertEqual(1, self.source.events_read)
        self.assertEqual(1, self.source.labels_read)
        by_id = {sample.event_id: sample for sample in expected}
        relay = next(instance for instance in self.all_samples(cache)[2].events if instance.motif_type == "relay")
        self.assertEqual(by_id["T1"].features, relay.members[0].features)
        self.assertNotEqual(by_id["T1"].features, by_id["T3"].features)
        cache.validate_integrity(check_members=True)
        self.assertEqual("a" * 64, cache.manifest["dataset"]["source_sha256"])

    def test_cache_stores_features_once_and_labels_in_a_separate_table(self):
        cache = self.build()
        with sqlite3.connect(cache.database_path) as connection:
            rows = connection.execute("SELECT features,events_json FROM samples ORDER BY seq").fetchall()
            self.assertTrue(all(len(features) == 24 * 8 for features, _ in rows))
            self.assertEqual(6, connection.execute("SELECT COUNT(*) FROM evaluation_labels").fetchone()[0])
            for _, raw in rows:
                for event in json.loads(raw):
                    for member in event["members"]:
                        self.assertEqual({"event_id", "role", "elapsed_seconds"}, set(member))
                self.assertNotIn("label", raw)

    def test_split_matches_chronological_split_and_labels_and_ids_align(self):
        cache = self.build()
        expected = build_samples(graph(self.events))
        split = chronological_split(expected)
        for name, indices in split.items():
            self.assertEqual([expected[index] for index in indices], self.all_samples(cache, name, 1))
            ids = [expected[index].event_id for index in indices]
            self.assertEqual(ids, list(cache.iter_ids(name)))
            self.assertEqual([self.source.labels[event_id] for event_id in ids], list(cache.iter_labels(name)))
        self.assertEqual({**{name: len(indices) for name, indices in split.items()}, "all": 6}, cache.split_counts())
        peer_split = {sample.event_id: name for name in split for sample in self.all_samples(cache, name)}
        self.assertEqual(peer_split["T3"], peer_split["T4"])

    def test_varied_group_sizes_and_tie_boundaries_match_existing_split(self):
        rng = random.Random(51)
        for attempt in range(8):
            events = [transfer(f"T{group:02}-{position:02}", "A", "B", group * 10)
                      for group in range(3 + attempt) for position in range(rng.randrange(1, 12))]
            source = StubDataset(self.directory, events)
            output = self.directory / f"split-{attempt}"
            build_sample_cache(source, output, train_fraction=.37, validation_fraction=.29)
            cache = CachedSamples(output)
            samples = build_samples(graph(events))
            for name, indices in chronological_split(samples, .37, .29).items():
                self.assertEqual([samples[index].event_id for index in indices], list(cache.iter_ids(name)))

    def test_changing_labels_cannot_change_features_or_time_boundaries(self):
        first = self.build()
        self.source.labels = {event.event_id: None if index == 2 else (index + 1) % 2 for index, event in enumerate(self.source.events)}
        second_output = self.directory / "second"
        build_sample_cache(self.source, second_output)
        second = CachedSamples(second_output)
        self.assertEqual(self.all_samples(first), self.all_samples(second))
        self.assertEqual(first.manifest["split"], second.manifest["split"])
        self.assertNotEqual(list(first.iter_labels("all")), list(second.iter_labels("all")))
        self.assertIn(None, list(second.iter_labels("all")))

    def test_explicit_policy_cache_matches_policy_causal_samples(self):
        policy = DetectionPolicy("P", "1", (DetectionRule("R", "relay", 100),))
        cache = self.build(policy=policy)
        self.assertEqual(build_samples(graph(self.events), policy=policy), self.all_samples(cache))
        self.assertEqual(policy.to_dict(), cache.manifest["policy"])

    def test_invalid_split_arguments_and_nonempty_outputs_are_rejected(self):
        for kwargs in ({"train_fraction": 0}, {"validation_fraction": float("nan")},
                       {"train_fraction": .8, "validation_fraction": .3}, {"train_fraction": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_sample_cache(self.source, self.output, **kwargs)
            self.assertFalse(self.output.exists())
        self.build()
        with self.assertRaisesRegex(ValueError, "new or empty"):
            build_sample_cache(self.source, self.output)

    def test_failed_and_interrupted_builds_cannot_be_read_as_complete(self):
        for index, error in enumerate((ValueError("fixture"), KeyboardInterrupt())):
            output = self.directory / f"failure-{index}"
            with patch("hamletkg.detection.cache.iter_samples", side_effect=error), self.assertRaises(type(error)):
                build_sample_cache(self.source, output)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual("interrupted" if index else "failed", manifest["status"])
            with self.assertRaisesRegex(ValueError, "complete"):
                CachedSamples(output)

    def test_too_few_timestamps_and_missing_or_invalid_labels_fail_build(self):
        self.source.events = self.events[:2]
        with self.assertRaisesRegex(ValueError, "three distinct"):
            build_sample_cache(self.source, self.output)
        self.source.events = self.events
        for index, labels in enumerate(({"T1": 0}, {event.event_id: True for event in self.events},
                                       {**self.source.labels, "UNKNOWN": 0},
                                       {**{event.event_id: 0 for event in self.events[:-1]}, None: 0})):
            self.source.labels = labels
            with self.assertRaises((ValueError, sqlite3.IntegrityError)):
                build_sample_cache(self.source, self.directory / f"bad-labels-{index}")

    def test_manifest_edits_and_database_byte_edits_are_rejected(self):
        cache = self.build()
        manifest_path = self.output / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["split"]["train"]["count"] += 1
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "frozen manifest"):
            CachedSamples(self.output)
        manifest_path.write_text(json.dumps(cache.manifest))
        with cache.database_path.open("ab") as destination:
            destination.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "checksum"):
            cache.validate_integrity()

    def test_missing_member_and_same_timestamp_member_are_rejected_when_read(self):
        cache = self.build()
        with sqlite3.connect(cache.database_path) as connection:
            raw = connection.execute("SELECT events_json FROM samples WHERE event_id='T3'").fetchone()[0]
            events = json.loads(raw)
            events[0]["members"][0]["event_id"] = "MISSING"
            connection.execute("UPDATE samples SET events_json=? WHERE event_id='T3'", (json.dumps(events),))
        with self.assertRaisesRegex(ValueError, "missing member"):
            self.all_samples(cache)
        events = json.loads(raw)
        events[0]["members"][0]["event_id"] = "T4"
        events[0]["members"][0]["elapsed_seconds"] = 0
        with sqlite3.connect(cache.database_path) as connection:
            connection.execute("UPDATE samples SET events_json=? WHERE event_id='T3'", (json.dumps(events),))
        with self.assertRaisesRegex(ValueError, "causality"):
            self.all_samples(cache)

    def test_invalid_member_roles_or_elapsed_times_and_nonfinite_features_are_rejected(self):
        cache = self.build()
        with sqlite3.connect(cache.database_path) as connection:
            original = connection.execute("SELECT events_json FROM samples WHERE event_id='T3'").fetchone()[0]
        for field, value in (("role", "unknown"), ("elapsed_seconds", -1), ("elapsed_seconds", True)):
            events = json.loads(original)
            events[0]["members"][0][field] = value
            with sqlite3.connect(cache.database_path) as connection:
                connection.execute("UPDATE samples SET events_json=? WHERE event_id='T3'", (json.dumps(events),))
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.all_samples(cache)
        with sqlite3.connect(cache.database_path) as connection:
            connection.execute("UPDATE samples SET events_json=? WHERE event_id='T3'", (original,))
            connection.execute("UPDATE samples SET features=? WHERE event_id='T1'", (struct.pack("<24d", *([float("nan")] * 24)),))
        with self.assertRaisesRegex(ValueError, "finite"):
            self.all_samples(cache)

    def test_bad_reader_split_and_batch_size_are_rejected(self):
        cache = self.build()
        for size in (0, -1, True, 1.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                list(cache.iter_batches(batch_size=size))
        for method in (cache.iter_batches, cache.iter_labels, cache.iter_ids):
            with self.assertRaisesRegex(ValueError, "split must"):
                list(method("unknown"))

    def test_real_prepared_dataset_view_preserves_source_identity(self):
        source = self.directory / "transactions.csv"
        with source.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Timestamp", "From Bank", "Account", "To Bank", "Account.1", "Amount Received", "Receiving Currency", "Amount Paid", "Payment Currency", "Payment Format", "Is Laundering"])
            for index in range(8):
                writer.writerow([f"2026/01/01 00:0{index}", "1", f"A{index}", "2", f"A{index + 1}", "100", "USD", "100", "USD", "Wire", index % 2])
        prepared = self.directory / "prepared"
        prepare_detection_dataset(source, prepared)
        dataset = PreparedDataset(prepared, after="2026-01-01T00:02:00Z", before="2026-01-01T00:07:00Z")
        build_sample_cache(dataset, self.output)
        cache = CachedSamples(self.output)
        self.assertEqual(5, len(cache))
        self.assertEqual(dataset.selection, cache.manifest["dataset"]["selection"])
        self.assertEqual(dataset.manifest["source"]["sha256"], cache.manifest["dataset"]["source_sha256"])
        self.assertEqual(list(iter_samples(dataset.iter_events())), self.all_samples(cache))


if __name__ == "__main__":
    unittest.main()
