from __future__ import annotations

import builtins
import copy
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hamletkg.detection.stream import EventInstance, EventMember, MOTIF_TYPES, TransactionSample
from hamletkg.detection.tabular import FLAT_FEATURE_NAMES, LightGBMDetector, flat_event_summary

try:
    import numpy
    import lightgbm
except (ImportError, OSError) as error:
    LIGHTGBM_AVAILABLE = False
    LIGHTGBM_SKIP_REASON = f"optional LightGBM dependency or native runtime is unavailable ({type(error).__name__})"
else:
    LIGHTGBM_AVAILABLE = True
    LIGHTGBM_SKIP_REASON = ""


def sample(index: int, *, with_event: bool = True) -> TransactionSample:
    value = 1.0 if index % 2 else -1.0
    features = (value, float(index % 3))
    members = (
        EventMember(f"history-{index}", "incoming", (value * 1000, 1000.0), 30.0),
        EventMember(f"transaction-{index}", "outgoing", features, 0.0),
    )
    event = EventInstance(f"group-{index}", "pass_through", f"account-{index}", members, 30.0, 0.95)
    return TransactionSample(
        f"transaction-{index}", datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=index),
        features, (event,) if with_event else (),
    )


class TabularInputTests(unittest.TestCase):
    def test_module_import_does_not_load_optional_packages_or_torch(self) -> None:
        source = """
import builtins
import sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('numpy', 'lightgbm', 'torch'):
        raise AssertionError('optional dependency imported eagerly: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from hamletkg.detection.tabular import LightGBMDetector
detector = LightGBMDetector(('amount', 'count'))
assert detector.scaler is None
assert all(name not in sys.modules for name in ('numpy', 'lightgbm', 'torch'))
"""
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        result = subprocess.run([sys.executable, "-c", source], env=environment,
                                capture_output=True, text=True, check=False)
        self.assertEqual(0, result.returncode, result.stderr)

    def test_missing_numpy_or_lightgbm_or_native_runtime_has_clear_error(self) -> None:
        for package, error in (("numpy", ImportError("missing numpy")),
                               ("lightgbm", ImportError("missing lightgbm")),
                               ("lightgbm", OSError("missing libomp"))):
            with self.subTest(package=package, error=type(error).__name__):
                original = builtins.__import__

                def blocked(name, *args, **kwargs):
                    if name == package:
                        raise error
                    return original(name, *args, **kwargs)

                detector = LightGBMDetector(("amount", "count"))
                with patch("builtins.__import__", side_effect=blocked):
                    with self.assertRaisesRegex(RuntimeError, "numpy and lightgbm.*libomp"):
                        detector.fit([sample(0), sample(1)], [0, 1])
                self.assertIsNone(detector.scaler)
                self.assertIsNone(detector.model)

    def test_constructor_validates_parameters_without_optional_dependencies(self) -> None:
        invalid = (
            {"feature_names": ()}, {"feature_names": None}, {"feature_names": 24},
            {"feature_names": ("amount", "amount")},
            {"feature_names": ("",)}, {"feature_names": "amount"},
            {"feature_names": ("amount total",)}, {"feature_names": ("amount:total",)},
            {"variant": "event"}, {"seed": -1}, {"seed": True}, {"seed": 2**31},
            {"n_estimators": 0}, {"n_estimators": 1.5}, {"n_estimators": True},
            {"learning_rate": 0}, {"learning_rate": float("nan")},
            {"learning_rate": float("inf")}, {"learning_rate": True},
            {"num_leaves": 1}, {"num_leaves": 131073}, {"num_leaves": 2.5},
            {"class_weight": "unknown"},
            {"feature_names": (FLAT_FEATURE_NAMES[0],), "variant": "lgbm_flat"},
        )
        for changed in invalid:
            with self.subTest(changed=changed):
                parameters = {"feature_names": ("amount", "count"), **changed}
                with self.assertRaises(ValueError):
                    LightGBMDetector(**parameters)

    def test_base_ignores_events_entirely(self) -> None:
        detector = LightGBMDetector(("amount", "count"))
        original = sample(1)
        malformed_event = SimpleNamespace(span_seconds=float("nan"), members=())
        self.assertEqual(detector.feature_vector(original),
                         detector.feature_vector(replace(original, events=(malformed_event,))))

    def test_flat_summary_matches_neural_formula_and_vocabulary_order(self) -> None:
        first = sample(1).events[0]
        second = replace(first, members=first.members * 2, span_seconds=90.0, amount_ratio=1.05)
        fan_in = replace(first, motif_type="fan_in", members=first.members[:1], span_seconds=10.0, amount_ratio=2.0)
        unknown = replace(first, motif_type="unknown", span_seconds=float("nan"))
        summary = flat_event_summary((first, second, fan_in, unknown))
        self.assertEqual(16, len(summary))
        expected = {
            "fan_in": (math.log1p(1), math.log1p(1), math.log1p(10) / 10, math.log1p(2)),
            "pass_through": (math.log1p(2), math.log1p(6), math.log1p(60) / 10, math.log1p(1)),
        }
        for index, motif_type in enumerate(MOTIF_TYPES):
            self.assertEqual(expected.get(motif_type, (0.0,) * 4), summary[index * 4:index * 4 + 4])
        self.assertEqual((0.0,) * 16, flat_event_summary(()))
        detector = LightGBMDetector(("amount", "count"), variant="lgbm_flat")
        transaction = replace(sample(1), events=(first, second, fan_in, unknown))
        self.assertEqual(transaction.features + summary, detector.feature_vector(transaction))
        self.assertEqual(("amount", "count") + FLAT_FEATURE_NAMES, detector.model_feature_names)

    def test_flat_ignores_member_features_roles_times_and_identifiers(self) -> None:
        original = sample(1)
        event = replace(original.events[0], instance_id="different", anchor_id="different",
                        members=tuple(replace(member, features=(float("nan"),), role="different",
                                              elapsed_seconds=float("nan"), event_id="different")
                                      for member in original.events[0].members))
        changed = replace(original, event_id="changed", timestamp=original.timestamp + timedelta(days=900), events=(event,))
        detector = LightGBMDetector(("amount", "count"), variant="lgbm_flat")
        self.assertEqual(detector.feature_vector(original), detector.feature_vector(changed))

    def test_input_validation_rejects_only_encoded_bad_values(self) -> None:
        for variant in ("lgbm_base", "lgbm_flat"):
            detector = LightGBMDetector(("amount", "count"), variant=variant)
            for features in ((1.0,), (float("nan"), 1), (float("inf"), 1), ("one", 1)):
                with self.subTest(variant=variant, features=features), self.assertRaises(ValueError):
                    detector.feature_vector(replace(sample(1), features=features))
        detector = LightGBMDetector(("amount", "count"), variant="lgbm_flat")
        event = sample(1).events[0]
        for changed in (replace(event, members=()), replace(event, span_seconds=-1),
                        replace(event, amount_ratio=float("nan"))):
            with self.subTest(event=changed), self.assertRaises(ValueError):
                detector.feature_vector(replace(sample(1), events=(changed,)))

    def test_labels_are_separate_binary_inputs_and_balanced_needs_both_classes(self) -> None:
        for labels in ([], [0], [0, 2], [0, "1"], [0, 1.0], [False, True], [0, 0], [1, 1]):
            with self.subTest(labels=labels):
                detector = LightGBMDetector(("amount", "count"))
                with self.assertRaises(ValueError):
                    detector.fit([sample(0), sample(1)], labels)
                self.assertIsNone(detector.scaler)
                self.assertEqual({}, detector.training_config)
        detector = LightGBMDetector(("amount", "count"))
        with self.assertRaisesRegex(RuntimeError, "fit"):
            detector.predict([])
        with self.assertRaisesRegex(RuntimeError, "unfitted"):
            detector.save("unused.txt")


@unittest.skipUnless(LIGHTGBM_AVAILABLE, LIGHTGBM_SKIP_REASON)
class LightGBMDetectorTests(unittest.TestCase):
    def train(self, variant: str = "lgbm_base", *, class_weight: str = "balanced") -> LightGBMDetector:
        detector = LightGBMDetector(("amount", "count"), variant=variant, seed=7,
                                   n_estimators=12, learning_rate=0.1, class_weight=class_weight)
        detector.fit([sample(index) for index in range(80)], [index % 2 for index in range(80)])
        return detector

    def test_fit_learns_from_training_inputs_and_returns_finite_probabilities(self) -> None:
        samples = [sample(index, with_event=False) for index in range(80)]
        detector = LightGBMDetector(("amount", "count"), n_estimators=12, learning_rate=0.1, seed=7)
        history = detector.fit(samples, numpy.asarray([index % 2 for index in range(80)], dtype=numpy.int64))
        self.assertEqual(12, len(history))
        self.assertLess(history[-1]["loss"], history[0]["loss"])
        scores = detector.predict(samples)
        self.assertTrue(all(math.isfinite(score) and 0 <= score <= 1 for score in scores))
        self.assertGreater(sum(scores[1::2]), sum(scores[::2]))
        self.assertEqual([], detector.predict([]))

    def test_scaler_uses_only_training_transactions_and_stays_frozen(self) -> None:
        for variant in ("lgbm_base", "lgbm_flat"):
            with self.subTest(variant=variant):
                detector = self.train(variant)
                self.assertAlmostEqual(0.0, detector.scaler["mean"][0], places=12)
                self.assertAlmostEqual(1.0, detector.scaler["scale"][0], places=12)
                self.assertLess(detector.scaler["mean"][1], 2)
                before = copy.deepcopy(detector.scaler)
                detector.predict([replace(sample(90), features=(10000.0, 50000.0))])
                self.assertEqual(before, detector.scaler)
        constant = LightGBMDetector(("amount", "count"), n_estimators=1, class_weight="none")
        constant.fit([replace(sample(0), features=(3.0, 9.0))], [0])
        self.assertEqual({"mean": [3.0, 9.0], "scale": [1.0, 1.0]}, constant.scaler)

    def test_flat_can_learn_event_signal_when_base_features_are_constant(self) -> None:
        samples = [replace(sample(index, with_event=bool(index % 2)), features=(0.0, 0.0))
                   for index in range(80)]
        labels = [index % 2 for index in range(80)]
        base = LightGBMDetector(("amount", "count"), n_estimators=12)
        flat = LightGBMDetector(("amount", "count"), variant="lgbm_flat", n_estimators=12)
        base.fit(samples, labels)
        flat.fit(samples, labels)
        base_scores, flat_scores = base.predict(samples), flat.predict(samples)
        self.assertEqual(base_scores[0], base_scores[1])
        self.assertGreater(flat_scores[1], flat_scores[0])

    def test_labels_on_samples_are_never_read(self) -> None:
        class LabelPoison:
            def __init__(self, transaction):
                self.features, self.events = transaction.features, transaction.events

            @property
            def label(self):
                raise AssertionError("A detector must not read labels from samples")

        inputs = [LabelPoison(sample(index)) for index in range(80)]
        detector = LightGBMDetector(("amount", "count"), variant="lgbm_flat", n_estimators=2)
        detector.fit(inputs, [index % 2 for index in range(80)])
        self.assertEqual(2, len(detector.predict(inputs[:2])))

    def test_balanced_weights_use_only_training_label_counts(self) -> None:
        inputs = [replace(sample(index), features=(0.0, 0.0)) for index in range(80)]
        labels = [0] * 72 + [1] * 8
        balanced = LightGBMDetector(("amount", "count"), n_estimators=2)
        unweighted = LightGBMDetector(("amount", "count"), n_estimators=2, class_weight="none")
        balanced.fit(inputs, labels)
        unweighted.fit(inputs, labels)
        self.assertEqual({"0": 72, "1": 8}, balanced.training_config["class_counts"])
        self.assertEqual({"0": 80 / 144, "1": 5.0}, balanced.training_config["class_weights"])
        self.assertEqual(9.0, balanced.training_config["positive_weight"])
        before = copy.deepcopy(balanced.training_config)
        balanced.predict([sample(1000), sample(1001)])
        self.assertEqual(before, balanced.training_config)
        self.assertAlmostEqual(0.5, balanced.predict(inputs[:1])[0], places=6)
        self.assertAlmostEqual(0.1, unweighted.predict(inputs[:1])[0], places=6)
        self.assertEqual({"0": 1.0, "1": 1.0}, unweighted.training_config["class_weights"])

    def test_native_text_json_round_trip_and_seed_are_reproducible(self) -> None:
        for variant in ("lgbm_base", "lgbm_flat"):
            with self.subTest(variant=variant):
                detector = self.train(variant)
                inputs = [sample(1), sample(2, with_event=False)]
                scores = detector.predict(inputs)
                self.assertEqual(scores, self.train(variant).predict(inputs))
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "detector.txt"
                    detector.save(path)
                    self.assertTrue(path.read_text(encoding="utf-8").startswith("tree\n"))
                    payload = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
                    self.assertEqual(detector.config, payload["config"])
                    self.assertEqual(list(detector.model_feature_names), payload["model_feature_names"])
                    loaded = LightGBMDetector.load(path)
                    self.assertEqual(detector.config, loaded.config)
                    self.assertEqual(detector.scaler, loaded.scaler)
                    self.assertEqual(detector.training_config, loaded.training_config)
                    self.assertEqual(scores, loaded.predict(inputs))

    def test_loading_rejects_wrong_scaler_feature_order_and_model_hash(self) -> None:
        detector = self.train("lgbm_flat")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "detector.txt"
            detector.save(path)
            sidecar = path.with_suffix(".json")
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            invalid_scaler = copy.deepcopy(payload)
            invalid_scaler["scaler"]["scale"][0] = 0
            invalid_features = copy.deepcopy(payload)
            invalid_features["model_feature_names"].reverse()
            for changed, message in ((invalid_scaler, "scales"), (invalid_features, "feature order")):
                with self.subTest(message=message):
                    sidecar.write_text(json.dumps(changed), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        LightGBMDetector.load(path)
            sidecar.write_text(json.dumps(payload), encoding="utf-8")
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                LightGBMDetector.load(path)

    def test_failed_sidecar_save_preserves_an_existing_checkpoint(self) -> None:
        old = self.train("lgbm_base")
        new = self.train("lgbm_flat")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "detector.txt"
            old.save(path)
            before = {name: (Path(directory) / name).read_bytes() for name in ("detector.txt", "detector.json")}
            original_replace = os.replace

            def fail_sidecar(source, destination):
                if Path(destination).suffix == ".json":
                    raise OSError("simulated sidecar write failure")
                return original_replace(source, destination)

            with patch("hamletkg.detection.tabular.os.replace", side_effect=fail_sidecar):
                with self.assertRaisesRegex(OSError, "simulated"):
                    new.save(path)
            self.assertEqual(before, {name: (Path(directory) / name).read_bytes() for name in before})
            loaded = LightGBMDetector.load(path)
            self.assertEqual(old.predict([sample(1)]), loaded.predict([sample(1)]))
            self.assertEqual({"detector.txt", "detector.json"}, {file.name for file in Path(directory).iterdir()})
            fresh = Path(directory) / "fresh.txt"
            with patch("hamletkg.detection.tabular.os.replace", side_effect=fail_sidecar):
                with self.assertRaisesRegex(OSError, "simulated"):
                    new.save(fresh)
            self.assertFalse(fresh.exists())
            self.assertFalse(fresh.with_suffix(".json").exists())


if __name__ == "__main__":
    unittest.main()
