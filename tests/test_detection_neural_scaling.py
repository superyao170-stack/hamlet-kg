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
    from hamletkg.detection.neural import NeuralDetector
    from hamletkg.detection.stream import EventInstance, EventMember, TransactionSample
    TORCH_AVAILABLE = True


FEATURES = ("log_amount", "hour_sin", "hour_cos", "backup_hour_cos")


def periodic_sample(minute: int) -> "TransactionSample":
    phase = minute / (24 * 60) * 2 * math.pi
    previous = max(0, minute - 1) / (24 * 60) * 2 * math.pi
    values = (math.log1p(minute + 1), math.sin(phase), math.cos(phase), float(minute * 2))
    historic = (math.log1p(minute + 2), math.sin(previous), math.cos(previous), float(minute * 3))
    event = EventInstance(
        f"event-{minute}", "pass_through", "account",
        (EventMember(f"old-{minute}", "incoming", historic, 60.0),
         EventMember(f"tx-{minute}", "outgoing", values, 0.0)), 60.0, 0.95,
    )
    return TransactionSample(f"tx-{minute}", datetime(2026, 1, 1, tzinfo=timezone.utc)
                             + timedelta(minutes=minute), values, (event,))


def factory(samples, labels):
    def batches():
        for offset in range(0, len(samples), 3):
            yield samples[offset:offset + 3], labels[offset:offset + 3]
    return batches


@unittest.skipUnless(TORCH_AVAILABLE, "optional PyTorch dependency is not installed")
class NeuralScalingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.original_threads)

    def setUp(self) -> None:
        self.samples = [periodic_sample(minute) for minute in range(0, 31, 5)]
        self.labels = [index % 2 for index in range(len(self.samples))]

    def train(self, scaling="fixed_periodic", streaming=True, variant="event"):
        detector = NeuralDetector(FEATURES, variant=variant, hidden_dim=8, seed=21, scaling=scaling)
        if streaming:
            detector.fit_batches(factory(self.samples, self.labels), epochs=1, batch_size=3)
        else:
            detector.fit(self.samples, self.labels, epochs=1, batch_size=3)
        return detector

    def test_fixed_periodic_preserves_later_phase_coordinates_in_current_and_members(self) -> None:
        fixed, standard = self.train(), self.train("standard")
        future = periodic_sample(240)
        fixed_batch, standard_batch = fixed._tensorize([future]), standard._tensorize([future])
        # A half-hour training range has tiny cosine variance. The later
        # unseen phase remains within [-1,1] under its declared intrinsic scale.
        self.assertGreater(abs(float(standard_batch["base"][0, 2])), 100)
        for dimension in (1, 2):
            self.assertEqual(0.0, fixed.scaler["mean"][dimension])
            self.assertEqual(1.0, fixed.scaler["scale"][dimension])
            self.assertAlmostEqual(future.features[dimension], float(fixed_batch["base"][0, dimension]), places=6)
            for index, member in enumerate(future.events[0].members):
                self.assertAlmostEqual(member.features[dimension],
                                       float(fixed_batch["members"][0, 0, index, dimension]), places=6)
        for dimension in (0, 3):
            self.assertEqual(standard.scaler["mean"][dimension], fixed.scaler["mean"][dimension])
            self.assertEqual(standard.scaler["scale"][dimension], fixed.scaler["scale"][dimension])
            self.assertEqual(float(standard_batch["base"][0, dimension]), float(fixed_batch["base"][0, dimension]))

    def test_both_fit_paths_apply_the_identical_policy_for_every_main_variant(self) -> None:
        for variant in ("base", "flat", "event"):
            for scaling in ("standard", "fixed_periodic"):
                with self.subTest(variant=variant, scaling=scaling):
                    streamed = self.train(scaling, variant=variant)
                    memory = self.train(scaling, streaming=False, variant=variant)
                    self.assertEqual(memory.scaler, streamed.scaler)
                    for detector in (streamed, memory):
                        self.assertEqual(scaling, detector.config["scaling"])
                        self.assertEqual(scaling, detector.training_config["scaling"])
                        fixed = ["hour_sin", "hour_cos"] if scaling == "fixed_periodic" else []
                        self.assertEqual(fixed, detector.training_config["fixed_scale_feature_names"])
                        if fixed:
                            self.assertEqual("training_transactions_and_historical_members_except_fixed_periodic",
                                             detector.training_config["scaler_scope"])

    def test_periodic_scaler_depends_only_on_feature_definition_not_training_labels(self) -> None:
        first, second = self.train(), NeuralDetector(FEATURES, hidden_dim=8, seed=21, scaling="fixed_periodic")
        second.fit_batches(factory(self.samples, [1 - label for label in self.labels]), epochs=1, batch_size=3)
        self.assertEqual(first.scaler, second.scaler)
        self.assertEqual(first.training_config["fixed_scale_feature_names"], second.training_config["fixed_scale_feature_names"])

    def test_future_phase_inference_keeps_scaler_and_strategy_frozen(self) -> None:
        detector = self.train()
        before = copy.deepcopy((detector.scaler, detector.training_config, detector.config))
        scores = list(detector.predict_batches(lambda: iter([[periodic_sample(240)], [periodic_sample(720)]])))
        self.assertTrue(all(math.isfinite(score) and 0 <= score <= 1 for batch in scores for score in batch))
        self.assertEqual(before, (detector.scaler, detector.training_config, detector.config))

    def test_default_standard_matches_explicit_standard_training(self) -> None:
        explicit = self.train("standard")
        implicit = NeuralDetector(FEATURES, hidden_dim=8, seed=21)
        implicit.fit_batches(factory(self.samples, self.labels), epochs=1, batch_size=3)
        self.assertEqual("standard", implicit.scaling)
        self.assertEqual(explicit.scaler, implicit.scaler)
        self.assertEqual(explicit.predict(self.samples), implicit.predict(self.samples))

    def test_fixed_feature_names_are_exact_and_independent_of_column_order(self) -> None:
        names = ("backup_hour_cos", "hour_cos", "log_amount", "hour_sin")
        detector = NeuralDetector(names, hidden_dim=8, scaling="fixed_periodic")
        # Reorder values along with their names. Prefix/suffix matches must
        # retain the statistics of their observed training values.
        samples = [replace(sample, features=(sample.features[3], sample.features[2], sample.features[0], sample.features[1]), events=())
                   for sample in self.samples]
        detector.fit(samples, self.labels, epochs=1, batch_size=3)
        self.assertEqual(["hour_cos", "hour_sin"], detector.training_config["fixed_scale_feature_names"])
        for dimension in (1, 3):
            self.assertEqual(0.0, detector.scaler["mean"][dimension])
            self.assertEqual(1.0, detector.scaler["scale"][dimension])
        self.assertGreater(detector.scaler["mean"][0], 1.0)
        self.assertGreater(detector.scaler["scale"][0], 1.0)

    def test_no_named_periodic_features_retains_standard_statistics(self) -> None:
        standard = NeuralDetector(("amount", "count"), hidden_dim=8)
        fixed = NeuralDetector(("amount", "count"), hidden_dim=8, scaling="fixed_periodic")
        samples = [replace(sample, features=(sample.features[0], sample.features[3]), events=())
                   for sample in self.samples]
        for detector in (standard, fixed):
            detector.fit(samples, self.labels, epochs=1)
        self.assertEqual(standard.scaler, fixed.scaler)
        self.assertEqual([], fixed.training_config["fixed_scale_feature_names"])
        self.assertEqual("training_transactions_and_historical_members", fixed.training_config["scaler_scope"])

    def test_fixed_mode_checkpoint_restores_policy_scaler_and_predictions(self) -> None:
        detector = self.train()
        samples = [periodic_sample(240), periodic_sample(720)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixed.pt"
            detector.save(path)
            payload = torch.load(path, weights_only=True, map_location="cpu")
            self.assertEqual("fixed_periodic", payload["config"]["scaling"])
            restored = NeuralDetector.load(path)
            self.assertEqual(detector.config, restored.config)
            self.assertEqual(detector.training_config, restored.training_config)
            self.assertEqual(detector.scaler, restored.scaler)
            self.assertEqual(detector.predict(samples), restored.predict(samples))

    def test_old_checkpoint_without_scaling_field_loads_as_standard(self) -> None:
        detector = self.train("standard")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            detector.save(path)
            payload = torch.load(path, weights_only=True, map_location="cpu")
            payload["config"].pop("scaling")
            payload["training_config"].pop("scaling")
            payload["training_config"].pop("fixed_scale_feature_names")
            torch.save(payload, path)
            restored = NeuralDetector.load(path)
            self.assertEqual("standard", restored.scaling)
            self.assertEqual(detector.scaler, restored.scaler)
            self.assertEqual(detector.predict(self.samples), restored.predict(self.samples))

    def test_checkpoint_rejects_policy_and_statistics_disagreement(self) -> None:
        detector = self.train()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixed.pt"
            for field in ("scaler", "training_config"):
                with self.subTest(field=field):
                    detector.save(path)
                    payload = torch.load(path, weights_only=True, map_location="cpu")
                    if field == "scaler":
                        payload[field]["scale"][2] = 0.001
                    else:
                        payload[field]["scaling"] = "standard"
                    torch.save(payload, path)
                    with self.assertRaisesRegex(ValueError, "scal"):
                        NeuralDetector.load(path)

    def test_unknown_scaling_modes_are_rejected(self) -> None:
        for scaling in ("unknown", "fixed", None, True):
            with self.subTest(scaling=scaling):
                with self.assertRaisesRegex(ValueError, "Unknown scaling"):
                    NeuralDetector(FEATURES, scaling=scaling)


if __name__ == "__main__":
    unittest.main()
