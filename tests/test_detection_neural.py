from __future__ import annotations

import copy
import math
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import torch
except ImportError:
    TORCH_AVAILABLE = False
else:
    from hamletkg.detection.neural import NeuralDetector, VARIANTS
    from hamletkg.detection.stream import EventInstance, EventMember, TransactionSample
    TORCH_AVAILABLE = True


def sample(index: int, *, with_event: bool = True) -> "TransactionSample":
    value = 1.0 if index % 2 else -1.0
    features = (value, float(index % 3))
    members = (
        EventMember(f"history-{index}", "incoming", (value * 2, 1.0), 30.0),
        EventMember(f"transaction-{index}", "outgoing", features, 0.0),
    )
    event = EventInstance(f"group-{index}", "pass_through", f"account-{index}",
                          members, 30.0, 0.95)
    return TransactionSample(f"transaction-{index}",
                             datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=index),
                             features, (event,) if with_event else ())


@unittest.skipUnless(TORCH_AVAILABLE, "optional PyTorch dependency is not installed")
class NeuralDetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.original_threads)

    def train(self, variant: str = "event", *, epochs: int = 3) -> "NeuralDetector":
        detector = NeuralDetector(("amount", "count"), variant=variant, hidden_dim=12, seed=7)
        detector.fit([sample(index) for index in range(12)],
                     [index % 2 for index in range(12)],
                     epochs=epochs, learning_rate=0.01, batch_size=4)
        return detector

    def test_fit_predict_uses_minibatches_and_learns_without_motifs(self) -> None:
        samples = [sample(index, with_event=False) for index in range(12)]
        detector = NeuralDetector(("amount", "count"), variant="event", hidden_dim=12, seed=7)
        history = detector.fit(samples, [index % 2 for index in range(12)],
                               epochs=15, learning_rate=0.02, batch_size=4)
        self.assertEqual(15, len(history))
        self.assertLess(history[-1]["loss"], history[0]["loss"])
        scores = detector.predict(samples)
        self.assertTrue(all(math.isfinite(score) and 0 <= score <= 1 for score in scores))
        self.assertGreater(sum(scores[1::2]), sum(scores[::2]))
        self.assertEqual([], detector.predict([]))

    def test_scaler_includes_training_history_and_stays_frozen(self) -> None:
        first = sample(0)
        high_member = replace(first.events[0].members[0], features=(100.0, 1.0))
        high_event = replace(first.events[0], members=(high_member, first.events[0].members[1]))
        detector = NeuralDetector(("amount", "count"), hidden_dim=8)
        detector.fit([replace(first, events=(high_event,)), sample(1)], [0, 1], epochs=1)
        self.assertGreater(detector.scaler["mean"][0], 10)
        snapshot = copy.deepcopy(detector.scaler)
        detector.predict([replace(sample(2), features=(10000.0, 50000.0))])
        self.assertEqual(snapshot, detector.scaler)

    def test_constant_training_columns_have_unit_scale(self) -> None:
        detector = NeuralDetector(("amount", "count"), hidden_dim=8)
        detector.fit([sample(0, with_event=False)], [0], epochs=1)
        self.assertEqual([1.0, 1.0], detector.scaler["scale"])

    def test_save_load_preserves_scores_and_primitive_configuration(self) -> None:
        detector = self.train()
        samples = [sample(1), sample(3, with_event=False)]
        before = detector.predict(samples)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "detector.pt"
            detector.save(destination)
            payload = torch.load(destination, weights_only=True, map_location="cpu")
            self.assertEqual(detector.config, payload["config"])
            self.assertEqual(detector.scaler, payload["scaler"])
            loaded = NeuralDetector.load(destination)
            self.assertEqual(detector.config, loaded.config)
            self.assertEqual(detector.scaler, loaded.scaler)
            self.assertEqual(before, loaded.predict(samples))

    def test_identifiers_and_absolute_timestamps_are_not_model_features(self) -> None:
        detector = self.train()
        original = sample(3)
        event = replace(original.events[0], instance_id="new-instance", anchor_id="new-anchor",
                        members=tuple(replace(member, event_id=f"changed-{index}")
                                      for index, member in enumerate(original.events[0].members)))
        renamed = replace(original, event_id="new-transaction",
                          timestamp=original.timestamp + timedelta(days=900), events=(event,))
        self.assertEqual(detector.predict([original]), detector.predict([renamed]))

    def test_no_role_and_no_time_ignore_disabled_event_inputs(self) -> None:
        original = sample(3)
        changed_roles = replace(original.events[0], members=tuple(
            replace(member, role="relay_end") for member in original.events[0].members))
        changed_times = replace(original.events[0], span_seconds=50000.0, members=tuple(
            replace(member, elapsed_seconds=50000.0) for member in original.events[0].members))
        for variant, event in (("no_role", changed_roles), ("no_time", changed_times)):
            with self.subTest(variant=variant):
                detector = self.train(variant)
                self.assertEqual(detector.predict([original]),
                                 detector.predict([replace(original, events=(event,))]))

    def test_untyped_grouping_ignores_all_event_semantics(self) -> None:
        original = sample(3)
        event = replace(original.events[0], motif_type="fan_out", span_seconds=50000,
                        amount_ratio=5000, members=tuple(
                            replace(member, role="relay_end", elapsed_seconds=50000)
                            for member in original.events[0].members))
        detector = self.train("untyped_event")
        self.assertEqual(detector.predict([original]),
                         detector.predict([replace(original, events=(event,))]))

    def test_flat_uses_summaries_without_member_feature_groups(self) -> None:
        original = sample(3)
        event = replace(original.events[0], members=tuple(
            replace(member, features=(1000.0, 2000.0), role="relay_end", elapsed_seconds=900)
            for member in original.events[0].members))
        detector = self.train("flat")
        self.assertEqual(detector.predict([original]),
                         detector.predict([replace(original, events=(event,))]))

    def test_seed_is_reproducible_and_all_variants_score_empty_events(self) -> None:
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                first, second = self.train(variant, epochs=1), self.train(variant, epochs=1)
                samples = [sample(1), sample(2, with_event=False)]
                self.assertEqual(first.predict(samples), second.predict(samples))
                self.assertTrue(all(math.isfinite(score) for score in first.predict(samples)))

    def test_padding_of_larger_event_histories_does_not_change_other_scores(self) -> None:
        detector = self.train()
        original = sample(1)
        other = sample(3)
        larger = replace(other.events[0], members=other.events[0].members * 3)
        other = replace(other, events=(larger, other.events[0]))
        isolated = detector.predict([original])[0]
        batched = detector.predict([original, other])[0]
        self.assertAlmostEqual(isolated, batched, places=6)

    def test_labels_are_separate_and_validated(self) -> None:
        detector = NeuralDetector(("amount", "count"))
        with self.assertRaises(ValueError):
            detector.fit([sample(1)], [], epochs=1)
        with self.assertRaises(ValueError):
            detector.fit([sample(1)], [2], epochs=1)
        with self.assertRaises(RuntimeError):
            detector.predict([sample(1)])
        self.assertIsNone(detector.scaler)

    def test_balanced_class_weight_uses_only_training_labels(self) -> None:
        samples = [sample(index) for index in range(5)]
        labels = [0, 0, 0, 0, 1]
        balanced = NeuralDetector(("amount", "count"), hidden_dim=12, seed=7)
        balanced.fit(samples, labels, epochs=2, learning_rate=0.01,
                     batch_size=5, class_weight="balanced")
        self.assertEqual("balanced", balanced.training_config["class_weight"])
        self.assertEqual(4.0, balanced.training_config["positive_weight"])
        before = copy.deepcopy(balanced.training_config)
        balanced.predict([sample(19), sample(20)])
        self.assertEqual(before, balanced.training_config)
        unweighted = NeuralDetector(("amount", "count"), hidden_dim=12, seed=7)
        unweighted.fit(samples, labels, epochs=2, learning_rate=0.01, batch_size=5)
        self.assertEqual("none", unweighted.training_config["class_weight"])
        self.assertEqual(1.0, unweighted.training_config["positive_weight"])
        self.assertNotEqual(unweighted.predict(samples), balanced.predict(samples))

    def test_balanced_class_weight_rejects_missing_classes_and_unknown_modes(self) -> None:
        for labels, mode in (([0, 0], "balanced"), ([1, 1], "balanced"),
                             ([0, 1], "unknown")):
            with self.subTest(labels=labels, mode=mode):
                detector = NeuralDetector(("amount", "count"), hidden_dim=8)
                with self.assertRaises(ValueError):
                    detector.fit([sample(0), sample(1)], labels, epochs=1, class_weight=mode)
                self.assertIsNone(detector.scaler)


if __name__ == "__main__":
    unittest.main()
