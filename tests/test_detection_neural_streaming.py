from __future__ import annotations

import copy
import math
import statistics
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ImportError:
    TORCH_AVAILABLE = False
else:
    from hamletkg.detection.neural import NeuralDetector
    from hamletkg.detection.stream import EventInstance, EventMember, TransactionSample
    TORCH_AVAILABLE = True


def make_sample(index: int) -> "TransactionSample":
    features = (float(index - 3), float(index % 3))
    members = (
        EventMember(f"old-{index}", "incoming", (float(3 * index + 1), 2.0), 60.0),
        EventMember(f"tx-{index}", "outgoing", features, 0.0),
    )
    event = EventInstance(f"event-{index}", "pass_through", f"account-{index}", members, 60.0, 0.9)
    return TransactionSample(f"tx-{index}", datetime(2026, 1, 1, tzinfo=timezone.utc)
                             + timedelta(seconds=index), features, (event,))


def make_factory(samples: list["TransactionSample"], labels: list[int], batch_size: int = 3):
    def factory():
        for offset in range(0, len(samples), batch_size):
            yield samples[offset:offset + batch_size], labels[offset:offset + batch_size]
    return factory


@unittest.skipUnless(TORCH_AVAILABLE, "optional PyTorch dependency is not installed")
class NeuralStreamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.original_threads)

    def detector(self, variant="event") -> "NeuralDetector":
        return NeuralDetector(("amount", "count"), variant=variant, hidden_dim=8, seed=19)

    def train(self, variant="event") -> "NeuralDetector":
        detector = self.detector(variant)
        samples = [make_sample(index) for index in range(9)]
        detector.fit_batches(make_factory(samples, [index % 2 for index in range(9)]),
                             epochs=2, batch_size=3, class_weight="balanced")
        return detector

    def test_stream_scaler_matches_independent_statistics_and_old_path(self) -> None:
        samples = [make_sample(index) for index in range(7)]
        labels = [index % 2 for index in range(7)]
        vectors = [vector for sample in samples for vector in
                   (sample.features, *(member.features for event in sample.events for member in event.members))]
        for variant in ("base", "flat", "event"):
            with self.subTest(variant=variant):
                streamed, previous = self.detector(variant), self.detector(variant)
                streamed.fit_batches(make_factory(samples, labels, 2), epochs=1, batch_size=2)
                previous.fit(samples, labels, epochs=1, batch_size=2)
                self.assertEqual(previous.scaler, streamed.scaler)
                for dimension in range(2):
                    values = [vector[dimension] for vector in vectors]
                    self.assertAlmostEqual(statistics.mean(values), streamed.scaler["mean"][dimension])
                    self.assertAlmostEqual(statistics.pstdev(values), streamed.scaler["scale"][dimension])
                self.assertEqual(len(vectors), streamed.training_config["scaler_vector_count"])

    def test_future_predictions_do_not_change_train_scaler_or_weights(self) -> None:
        detector = self.train()
        before = copy.deepcopy((detector.scaler, detector.training_config))
        future = replace(make_sample(10000), features=(500000.0, 900000.0))
        list(detector.predict_batches(lambda: iter([[future]])))
        self.assertEqual(before, (detector.scaler, detector.training_config))
        self.assertEqual({"negative": 5, "positive": 4}, detector.training_config["class_counts"])
        self.assertEqual(1.25, detector.training_config["positive_weight"])

    def test_factory_restarts_for_each_epoch(self) -> None:
        calls = []
        samples = [make_sample(index) for index in range(6)]
        def factory():
            calls.append(len(calls))
            yield samples[:4], [0, 1, 0, 1]
            yield samples[4:], [0, 1]
        detector = self.detector()
        history = detector.fit_batches(factory, epochs=2, batch_size=2)
        self.assertEqual(3, len(calls))
        self.assertEqual([1, 2], [row["epoch"] for row in history])
        self.assertTrue(all(math.isfinite(row["loss"]) for row in history))
        self.assertEqual("completed", detector.training_config["training_status"])
        self.assertEqual("source_order", detector.training_config["sequence_strategy"])
        self.assertEqual(3, detector.training_config["source_factory_passes"])

    def test_one_shot_factory_is_rejected_instead_of_succeeding_empty_epochs(self) -> None:
        source = iter([([make_sample(0), make_sample(1)], [0, 1])])
        detector = self.detector()
        with self.assertRaisesRegex(ValueError, "restart"):
            detector.fit_batches(lambda: source, epochs=2)
        self.assertEqual("failed", detector.training_config["training_status"])
        self.assertEqual(2, detector.training_config["source_factory_passes"])
        self.assertEqual(0, detector.training_config["completed_epochs"])

    def test_epoch_loss_is_weighted_by_actual_minibatch_sample_counts(self) -> None:
        samples = [make_sample(index) for index in range(5)]
        def criterion(logits, targets):
            # Preserve a gradient connection while exposing unequal batch
            # sizes in the reported mean. An unweighted batch mean gives 2.5.
            return logits.sum() * 0 + len(targets)
        with patch("hamletkg.detection.neural.nn.BCEWithLogitsLoss", return_value=criterion):
            history = self.detector().fit_batches(
                make_factory(samples, [0, 1, 0, 1, 0], 5), epochs=1, batch_size=3,
            )
        self.assertEqual(2.6, history[0]["loss"])

    def test_changed_counts_labels_features_and_source_identity_are_rejected(self) -> None:
        original = [make_sample(0), make_sample(1)]
        changes = (
            ([original[0]], [0]),
            (original, [1, 0]),
            ([replace(original[0], features=(123.0, 5.0)), original[1]], [0, 1]),
            ([replace(original[0], event_id="different-row"), original[1]], [0, 1]),
            ([replace(original[0], timestamp=original[0].timestamp + timedelta(days=1)), original[1]], [0, 1]),
            (list(reversed(original)), [1, 0]),
        )
        for changed_samples, changed_labels in changes:
            with self.subTest(changed_samples=changed_samples, changed_labels=changed_labels):
                calls = []
                def factory():
                    calls.append(1)
                    yield (original, [0, 1]) if len(calls) == 1 else (changed_samples, changed_labels)
                with self.assertRaisesRegex(ValueError, "unchanged"):
                    self.detector().fit_batches(factory, epochs=1)

    def test_batch_boundaries_can_change_without_changing_source_identity(self) -> None:
        samples = [make_sample(index) for index in range(6)]
        labels = [index % 2 for index in range(6)]
        calls = []
        def factory():
            calls.append(1)
            source_batch_size = 2 if len(calls) == 1 else 4
            yield from make_factory(samples, labels, source_batch_size)()
        detector = self.detector()
        detector.fit_batches(factory, epochs=2, batch_size=2)
        self.assertEqual("completed", detector.training_config["training_status"])

    def test_training_does_not_collect_source_or_allocate_global_permutation(self) -> None:
        calls, tensorized, source_reads = [], [], []
        detector = self.detector("base")
        original_tensorize = detector._tensorize
        class NoLengthHintIterator:
            def __init__(self):
                self.index = 0
            def __iter__(self):
                return self
            def __length_hint__(self):
                raise AssertionError("must not collect the full source")
            def __next__(self):
                if self.index == 6:
                    raise StopIteration
                if len(calls) > 1 and self.index:
                    if not tensorized:
                        raise AssertionError("training must start before consuming the next source batch")
                current = self.index
                self.index += 1
                source_reads.append((len(calls), self.index))
                return [make_sample(current)], [current % 2]
        def factory():
            calls.append(1)
            return NoLengthHintIterator()
        def tensorize(samples):
            tensorized.append(len(samples))
            return original_tensorize(samples)
        with patch.object(detector, "_tensorize", side_effect=tensorize), \
                patch("hamletkg.detection.neural.torch.randperm", side_effect=AssertionError("no full permutation")):
            detector.fit_batches(factory, epochs=2, batch_size=2)
        self.assertEqual(12, len(tensorized))
        self.assertEqual(18, len(source_reads))

    def test_oversized_source_batches_are_tensorized_as_bounded_minibatches(self) -> None:
        detector = self.detector()
        samples = [make_sample(index) for index in range(13)]
        sizes = []
        tensorize = detector._tensorize
        def inspect(batch):
            sizes.append(len(batch))
            return tensorize(batch)
        with patch.object(detector, "_tensorize", side_effect=inspect):
            detector.fit_batches(make_factory(samples, [index % 2 for index in range(13)], 13),
                                 epochs=1, batch_size=3)
        self.assertEqual([3, 3, 3, 3, 1], sizes)

    def test_bounded_shuffle_is_reproducible_and_records_its_different_order(self) -> None:
        samples = [make_sample(index) for index in range(18)]
        labels = [index % 2 for index in range(18)]
        first, second = self.detector(), self.detector()
        seen = []
        original_tensorize = first._tensorize
        def capture(batch):
            seen.extend(sample.event_id for sample in batch)
            return original_tensorize(batch)
        with patch.object(first, "_tensorize", side_effect=capture):
            first.fit_batches(make_factory(samples, labels, 2), epochs=2, batch_size=2,
                              shuffle_buffer_batches=3)
        second.fit_batches(make_factory(samples, labels, 2), epochs=2, batch_size=2,
                           shuffle_buffer_batches=3)
        self.assertEqual(first.predict(samples), second.predict(samples))
        self.assertEqual("bounded_batch_shuffle", first.training_config["sequence_strategy"])
        self.assertEqual(3, first.training_config["shuffle_buffer_batches"])
        self.assertNotEqual([sample.event_id for sample in samples], seen[:18])
        self.assertCountEqual([sample.event_id for sample in samples], seen[:18])

    def test_bounded_shuffle_never_reads_the_whole_epoch_before_training(self) -> None:
        calls, source_reads, processed = [], [0], [0]
        detector = self.detector("base")
        original_tensorize = detector._tensorize
        def factory():
            calls.append(1)
            for index in range(15):
                if len(calls) > 1:
                    source_reads[0] += 1
                yield [make_sample(index)], [index % 2]
        def capture(samples):
            self.assertLessEqual(source_reads[0] - processed[0], 3)
            processed[0] += len(samples)
            return original_tensorize(samples)
        with patch.object(detector, "_tensorize", side_effect=capture):
            detector.fit_batches(factory, epochs=1, batch_size=1, shuffle_buffer_batches=3)
        self.assertEqual(15, processed[0])

    def test_prediction_batches_match_regular_prediction_and_keep_scaler_frozen(self) -> None:
        detector = self.train()
        samples = [make_sample(index) for index in range(15, 24)]
        before = copy.deepcopy(detector.scaler)
        source = lambda: iter([samples[:2], samples[2:7], samples[7:]])
        scores = [score for batch in detector.predict_batches(source) for score in batch]
        for old, new in zip(detector.predict(samples), scores):
            self.assertAlmostEqual(old, new, places=6)
        self.assertEqual(before, detector.scaler)

    def test_prediction_yields_lazily_and_does_not_leak_no_grad_context(self) -> None:
        detector = self.train()
        consumed = []
        def factory():
            for index in range(4):
                consumed.append(index)
                yield [make_sample(index)]
        iterator = detector.predict_batches(factory)
        self.assertEqual([], consumed)
        self.assertTrue(torch.is_grad_enabled())
        self.assertEqual(1, len(next(iterator)))
        self.assertEqual([0], consumed)
        self.assertTrue(torch.is_grad_enabled())
        iterator.close()
        self.assertTrue(torch.is_grad_enabled())

    def test_save_load_retains_stream_training_configuration_and_scores(self) -> None:
        detector = self.train()
        factory = lambda: iter([[make_sample(8)], [make_sample(20), make_sample(21)]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            detector.save(path)
            restored = NeuralDetector.load(path)
            self.assertEqual(detector.training_config, restored.training_config)
            self.assertEqual(list(detector.predict_batches(factory)), list(restored.predict_batches(factory)))
            self.assertEqual(detector.scaler, restored.scaler)

    def test_invalid_labels_are_rejected_before_scaler_is_set(self) -> None:
        for label in (2, -1, True, False, 0.0, "1", None):
            with self.subTest(label=label):
                detector = self.detector()
                with self.assertRaises(ValueError):
                    detector.fit_batches(lambda: iter([([make_sample(0)], [label])]), epochs=1)
                self.assertIsNone(detector.scaler)

    def test_old_fit_also_rejects_boolean_and_float_labels(self) -> None:
        for label in (True, False, 0.0, 1.0):
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    self.detector().fit([make_sample(0)], [label], epochs=1)

    def test_invalid_batches_and_empty_stream_are_rejected(self) -> None:
        batches = ([], [([], [])], [([make_sample(0)], [])], [([make_sample(0)], [0, 1])],
                   [([make_sample(0)],)], [(iter([make_sample(0)]), [0])],
                   [([make_sample(0)], iter([0]))], [None])
        for source in batches:
            with self.subTest(source=source):
                detector = self.detector()
                with self.assertRaises(ValueError):
                    detector.fit_batches(lambda: iter(source), epochs=1)
                self.assertIsNone(detector.scaler)

    def test_nonfinite_features_and_invalid_event_descriptors_are_rejected(self) -> None:
        original = make_sample(1)
        event = original.events[0]
        invalid = (
            replace(original, features=(float("inf"), 1.0)),
            replace(original, features=(float("nan"), 1.0)),
            replace(original, features=(1.0,)),
            replace(original, events=(replace(event, members=()),)),
            replace(original, events=(replace(event, span_seconds=float("inf")),)),
            replace(original, events=(replace(event, amount_ratio=-0.1),)),
            replace(original, events=(replace(event, members=(replace(event.members[0],
                             features=(float("nan"), 2.0)),)),)),
            replace(original, events=(replace(event, members=(replace(event.members[0],
                             elapsed_seconds=-1.0),)),)),
        )
        for value in invalid:
            with self.subTest(value=value):
                detector = self.detector()
                with self.assertRaises(ValueError):
                    detector.fit_batches(lambda: iter([([original, value], [0, 1])]), epochs=1)
                self.assertIsNone(detector.scaler)

    def test_first_full_validation_pass_rejects_late_bad_batch_before_training(self) -> None:
        detector = self.detector()
        before = {name: value.clone() for name, value in detector.model.state_dict().items()}
        invalid = replace(make_sample(5), features=(float("inf"), 0.0))
        def factory():
            yield [make_sample(0), make_sample(1)], [0, 1]
            yield [invalid], [0]
        with patch("hamletkg.detection.neural.torch.optim.Adam", side_effect=AssertionError("must validate entire source first")):
            with self.assertRaises(ValueError):
                detector.fit_batches(factory, epochs=1)
        self.assertIsNone(detector.scaler)
        for name, value in before.items():
            self.assertTrue(torch.equal(value, detector.model.state_dict()[name]))

    def test_scaler_rejects_overflow_even_when_input_features_are_finite(self) -> None:
        samples = [replace(make_sample(0), features=(1e308, 0.0), events=()),
                   replace(make_sample(1), features=(-1e308, 0.0), events=())]
        detector = self.detector()
        with self.assertRaisesRegex(ValueError, "statistics are not finite"):
            detector.fit_batches(lambda: iter([(samples, [0, 1])]), epochs=1)
        self.assertIsNone(detector.scaler)

    def test_invalid_options_do_not_open_factory(self) -> None:
        for options in ({"epochs": 0}, {"epochs": True}, {"epochs": 1.5}, {"batch_size": False},
                        {"batch_size": 0}, {"learning_rate": 0}, {"learning_rate": float("nan")},
                        {"learning_rate": True}, {"shuffle_buffer_batches": -1},
                        {"shuffle_buffer_batches": True}, {"class_weight": "unknown"}):
            with self.subTest(options=options), patch.object(self, "train", side_effect=AssertionError("no factory")):
                with self.assertRaises(ValueError):
                    self.detector().fit_batches(lambda: self.train(), **options)

    def test_balanced_weight_requires_both_training_classes(self) -> None:
        for labels in ([0, 0], [1, 1]):
            detector = self.detector()
            with self.assertRaisesRegex(ValueError, "both classes"):
                detector.fit_batches(make_factory([make_sample(0), make_sample(1)], labels),
                                     epochs=1, class_weight="balanced")
            self.assertIsNone(detector.scaler)

    def test_prediction_rejects_empty_nonfinite_batches_and_unfitted_model(self) -> None:
        with self.assertRaises(RuntimeError):
            next(self.detector().predict_batches(lambda: iter([[make_sample(0)]])))
        detector = self.train()
        for batch in ([], iter([make_sample(0)]), [replace(make_sample(0), features=(float("inf"), 0.0))]):
            with self.subTest(batch=batch):
                with self.assertRaises(ValueError):
                    list(detector.predict_batches(lambda: iter([batch])))
        self.assertEqual([], list(detector.predict_batches(lambda: iter([]))))


if __name__ == "__main__":
    unittest.main()
