from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hamletkg.detection.cache import CachedSamples, build_sample_cache
from hamletkg.detection.dataset import PreparedDataset, prepare_detection_dataset
from hamletkg.detection.demo import write_demo_csv
from hamletkg.detection.disk_runner import benchmark_cached, labeled_batches, score_cached
from hamletkg.detection.policy import DetectionPolicy, DetectionRule
from hamletkg.detection.stream import BASE_FEATURE_NAMES, StreamConfig
from hamletkg.detection.tabular import LightGBMDetector, flat_event_summary


HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_LIGHTGBM = importlib.util.find_spec("lightgbm") is not None
HAS_NUMPY = importlib.util.find_spec("numpy") is not None


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def predictions(path):
    with Path(path).open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


@contextlib.contextmanager
def replace_bytes(path, replacement):
    original = path.read_bytes()
    try:
        path.write_bytes(replacement)
        yield
    finally:
        path.write_bytes(original)


class CachedCLITests(unittest.TestCase):
    def test_all_cached_commands_parse_without_loading_optional_runtimes(self):
        program = """
import sys
from hamletkg.cli import build_parser
from hamletkg.detection.disk_runner import benchmark_cached
parser = build_parser()
prepare = parser.parse_args(['prepare-data','--csv','input.csv','--output','prepared'])
assert prepare.limit is None and prepare.batch_size == 2000
cache = parser.parse_args(['cache-data','--dataset','prepared','--output','cache','--before','2026-01-02T00:00:00Z'])
assert cache.before == '2026-01-02T00:00:00Z' and cache.max_history == 32
benchmark = parser.parse_args(['detect-cached-benchmark','--cache','cache','--output','out'])
assert benchmark.seeds == [13,42,87] and benchmark.shuffle_buffer_batches == 0
score = parser.parse_args(['detect-cache-score','--cache','cache','--model','model.txt','--output','scored'])
assert score.split == 'test'
assert all(name not in sys.modules for name in ('torch','numpy','lightgbm'))
"""
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        result = subprocess.run([sys.executable, "-c", program], env=environment, capture_output=True, text=True, timeout=30)
        self.assertEqual(0, result.returncode, result.stderr)

    def test_prepare_and_cache_cli_preserve_time_view_and_policy_provenance_without_llm(self):
        from hamletkg.cli import main
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, prepared, output, policy_path = root / "input.csv", root / "prepared", root / "cache", root / "policy.json"
            write_demo_csv(source, days=6)
            policy = DetectionPolicy("cli-test", "1", (DetectionRule("passing", "pass_through", 86400),))
            policy_path.write_text(json.dumps({"policy": policy.to_dict(), "compilation": {"source": "mocked_provenance_fixture"}}))
            with patch("hamletkg.cli._llm_client", side_effect=AssertionError("prepared/cache CLI must not call LLM")), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(0, main(["prepare-data", "--csv", str(source), "--output", str(prepared)]))
                self.assertEqual(0, main(["cache-data", "--dataset", str(prepared), "--output", str(output),
                                         "--policy", str(policy_path), "--before", "2026-01-04T00:00:00Z"]))
            cache = CachedSamples(output)
            self.assertEqual(135, len(cache))
            self.assertEqual("2026-01-04T00:00:00Z", cache.manifest["dataset"]["selection"]["before"])
            self.assertEqual(policy.to_dict(), cache.manifest["policy"])
            self.assertEqual(hashlib.sha256(policy_path.read_bytes()).hexdigest(), cache.manifest["policy_provenance"]["file_sha256"])
            self.assertEqual({"source": "mocked_provenance_fixture"}, cache.manifest["policy_provenance"]["compilation"])


@unittest.skipUnless(HAS_TORCH and HAS_LIGHTGBM and HAS_NUMPY, "optional mixed detection runtimes unavailable")
class CachedBenchmarkIntegrationTests(unittest.TestCase):
    """Small real training runs exercise complete cache/model/evaluation wiring."""

    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        from hamletkg.detection.neural import NeuralDetector

        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.source = cls.root / "fixture.csv"
        write_demo_csv(cls.source, seed=42, days=6)
        cls.prepared = cls.root / "prepared"
        prepare_detection_dataset(cls.source, cls.prepared, case_id="DISK-TEST")
        cls.dataset = PreparedDataset(cls.prepared)
        cls.cache_path = cls.root / "cache"
        cls.policy = DetectionPolicy("test-policy", "1", (
            DetectionRule("fanin", "fan_in", 86400),
            DetectionRule("passing", "pass_through", 86400),
        ))
        cls.provenance = {"source": "unit_test_fixture", "compilation": {"samples": 2, "valid_rate": 1.0}}
        build_sample_cache(cls.dataset, cls.cache_path, config=StreamConfig(max_history=8),
                           policy=cls.policy, policy_provenance=cls.provenance)
        cls.cache = CachedSamples(cls.cache_path)
        cls.training = [sample for batch in cls.cache.iter_batches("train") for sample in batch]
        cls.training_labels = list(cls.cache.iter_labels("train"))
        cls.fit_inputs, cls.matrix_inputs = {}, {}
        original_fit, original_matrix = NeuralDetector.fit_batches, LightGBMDetector.fit_matrix

        def capture_fit(detector, factory, **options):
            passes = []
            cls.fit_inputs[detector.seed] = passes

            def recorded_factory():
                observed = []
                passes.append(observed)
                for batch, labels in factory():
                    observed.extend((sample.event_id, label) for sample, label in zip(batch, labels))
                    yield batch, labels

            return original_fit(detector, recorded_factory, **options)

        def capture_matrix(detector, features, labels, **options):
            cls.matrix_inputs[(detector.seed, detector.variant)] = (
                np.asarray(features).copy(), np.asarray(labels).copy(),
            )
            return original_matrix(detector, features, labels, **options)

        cls.benchmark_path = cls.root / "benchmark"
        with patch.object(NeuralDetector, "fit_batches", autospec=True, side_effect=capture_fit), \
                patch.object(LightGBMDetector, "fit_matrix", autospec=True, side_effect=capture_matrix), \
                patch("hamletkg.cli._llm_client", side_effect=AssertionError("cached training must not call an LLM")):
            cls.result = benchmark_cached(cls.cache_path, cls.benchmark_path, dataset_path=cls.prepared,
                                          seeds=(13, 42), variants=("event", "lgbm_base", "lgbm_flat"),
                                          epochs=1, hidden_dim=8, batch_size=32, tree_estimators=3,
                                          tree_leaves=4, budget_fractions=(.05, .2), data_role="smoke")

    @classmethod
    def tearDownClass(cls):
        import torch
        torch.set_num_threads(cls.original_threads)
        cls.temporary.cleanup()

    def output(self, name):
        return self.root / f"{self._testMethodName}-{name}"

    def test_mixed_two_seed_run_uses_identical_training_split_and_separate_labels(self):
        import numpy as np
        self.assertEqual("completed", self.result["status"])
        self.assertEqual(270, self.result["transactions"])
        manifest = read_json(self.benchmark_path / "benchmark_manifest.json")
        self.assertEqual([13, 42], manifest["completed_seeds"])
        self.assertEqual(self.cache.manifest["dataset"], manifest["source"])
        self.assertEqual(self.provenance, self.cache.manifest["policy_provenance"])
        expected = list(zip([sample.event_id for sample in self.training], self.training_labels))
        test_ids = set(self.cache.iter_ids("test"))
        for seed, passes in self.fit_inputs.items():
            self.assertGreaterEqual(len(passes), 2)
            for observed in passes:
                self.assertEqual(expected, observed)
                self.assertFalse(test_ids.intersection(event_id for event_id, _ in observed))
        base = np.asarray([sample.features for sample in self.training])
        flat = np.asarray([(*sample.features, *flat_event_summary(sample.events)) for sample in self.training])
        for (seed, variant), (matrix, labels) in self.matrix_inputs.items():
            np.testing.assert_array_equal(base if variant == "lgbm_base" else flat, matrix)
            np.testing.assert_array_equal(self.training_labels, labels)
        stored_matrix = np.load(self.benchmark_path / "tree_training_features.npy", mmap_mode="r", allow_pickle=False)
        stored_labels = np.load(self.benchmark_path / "tree_training_labels.npy", mmap_mode="r", allow_pickle=False)
        np.testing.assert_array_equal(flat, stored_matrix)
        np.testing.assert_array_equal(self.training_labels, stored_labels)

    def test_scalers_use_only_training_vectors_and_seed_aggregates_remain_a_fixed_split(self):
        import numpy as np
        from hamletkg.detection.neural import NeuralDetector

        current = np.asarray([sample.features for sample in self.training])
        event_vectors = np.asarray([vector for sample in self.training
                                    for vector in (sample.features, *(member.features for event in sample.events for member in event.members))])
        event_mean = event_vectors.mean(axis=0)
        event_scale = event_vectors.std(axis=0)
        event_scale[event_scale < 1e-6] = 1
        for seed in (13, 42):
            directory = self.benchmark_path / f"seed_{seed}"
            event = NeuralDetector.load(directory / "model_event.pt")
            np.testing.assert_allclose(event.scaler["mean"], event_mean, atol=1e-12)
            np.testing.assert_allclose(event.scaler["scale"], event_scale, atol=1e-12)
            self.assertEqual(len(event_vectors), event.training_config["scaler_vector_count"])
            self.assertEqual(sum(label == 0 for label in self.training_labels) / sum(self.training_labels), event.training_config["positive_weight"])
            for variant in ("lgbm_base", "lgbm_flat"):
                tree = LightGBMDetector.load(directory / f"model_{variant}.txt")
                np.testing.assert_allclose(tree.scaler["mean"], current.mean(axis=0), atol=1e-12)
                self.assertEqual(len(self.training), tree.training_config["training_sample_count"])
                self.assertEqual({"0": self.training_labels.count(0), "1": self.training_labels.count(1)}, tree.training_config["class_counts"])
        summary = read_json(self.benchmark_path / "summary.json")
        self.assertEqual(2, summary["seed_count"])
        for variant in ("event", "lgbm_base", "lgbm_flat"):
            expected = [read_json(self.benchmark_path / f"seed_{seed}" / "metrics.json")[variant]["test"]["average_precision"] for seed in (13, 42)]
            stats = summary["variants"][variant]["test"]["average_precision"]
            self.assertAlmostEqual(sum(expected) / 2, stats["mean"])
            self.assertAlmostEqual(abs(expected[0] - expected[1]) / math.sqrt(2), stats["std"])
            self.assertEqual(2, stats["n"])

    def test_all_models_restore_to_identical_scores_sources_decisions_and_metrics(self):
        for seed in (13, 42):
            directory = self.benchmark_path / f"seed_{seed}"
            metrics = read_json(directory / "metrics.json")
            for variant in ("event", "lgbm_base", "lgbm_flat"):
                suffix = ".pt" if variant == "event" else ".txt"
                output = self.output(f"restored-{seed}-{variant}")
                result = score_cached(self.cache_path, directory / f"model_{variant}{suffix}", output,
                                      dataset_path=self.prepared, batch_size=7)
                expected = predictions(directory / f"predictions_{variant}.csv")
                actual = predictions(output / f"predictions_{variant}.csv")
                self.assertEqual(expected, actual)
                self.assertTrue(all(row["source_record_id"] for row in actual))
                self.assertEqual(metrics[variant]["test"], result["metrics"])
                self.assertFalse(result["threshold_reselected_during_scoring"])
                self.assertEqual("training_cache_validation", result["threshold_selection"])
                self.assertEqual("not_audited", result["evidence_status"])
                self.assertEqual("completed", read_json(output / "summary.json")["status"])

    def test_test_label_changes_cannot_change_model_scores_scalers_or_validation_threshold(self):
        test_records = self.dataset.fetch_transactions(self.cache.iter_ids("test"))
        test_rows = {int(event.source_record_id) for event in test_records.values()}
        with self.source.open(newline="") as handle:
            rows = list(csv.reader(handle))
        for index in test_rows:
            rows[index][-1] = str(1 - int(rows[index][-1]))
        source, prepared, cache_path = self.output("changed.csv"), self.output("prepared"), self.output("cache")
        with source.open("w", newline="") as handle:
            csv.writer(handle).writerows(rows)
        prepare_detection_dataset(source, prepared, case_id="DISK-TEST")
        build_sample_cache(PreparedDataset(prepared), cache_path, config=StreamConfig(max_history=8),
                           policy=self.policy, policy_provenance=self.provenance)
        output = self.output("changed-benchmark")
        benchmark_cached(cache_path, output, seeds=(13,), variants=("event", "lgbm_base", "lgbm_flat"),
                         epochs=1, hidden_dim=8, batch_size=32, tree_estimators=3, tree_leaves=4,
                         budget_fractions=(.05, .2), data_role="smoke")
        first, second = self.benchmark_path / "seed_13", output / "seed_13"
        from hamletkg.detection.neural import NeuralDetector
        for variant in ("event", "lgbm_base", "lgbm_flat"):
            sidecar = ".json" if variant == "event" else ".settings.json"
            self.assertEqual(read_json(first / f"model_{variant}{sidecar}")["threshold"], read_json(second / f"model_{variant}{sidecar}")["threshold"])
            left, right = predictions(first / f"predictions_{variant}.csv"), predictions(second / f"predictions_{variant}.csv")
            self.assertEqual([row["event_id"] for row in left], [row["event_id"] for row in right])
            self.assertEqual([row["score"] for row in left], [row["score"] for row in right])
            self.assertEqual([row["threshold_alert"] for row in left], [row["threshold_alert"] for row in right])
            self.assertTrue(all(one["label"] != two["label"] for one, two in zip(left, right)))
            if variant == "event":
                one, two = NeuralDetector.load(first / "model_event.pt"), NeuralDetector.load(second / "model_event.pt")
                self.assertEqual(one.scaler, two.scaler)
            else:
                one, two = LightGBMDetector.load(first / f"model_{variant}.txt"), LightGBMDetector.load(second / f"model_{variant}.txt")
                self.assertEqual(one.scaler, two.scaler)
            self.assertEqual(one.training_config["positive_weight"], two.training_config["positive_weight"])

    def test_native_model_and_tree_configuration_edits_are_rejected_before_output(self):
        directory = self.benchmark_path / "seed_13"
        model = directory / "model_lgbm_base.txt"
        output = self.output("modified-model")
        with replace_bytes(model, model.read_bytes() + b"\nmodified"):
            with self.assertRaisesRegex(ValueError, "settings"):
                score_cached(self.cache_path, model, output)
            self.assertFalse(output.exists())
        sidecar = model.with_suffix(".json")
        payload = read_json(sidecar)
        payload["scaler"]["mean"][0] += 1
        with replace_bytes(sidecar, json.dumps(payload).encode()):
            with self.assertRaisesRegex(ValueError, "configuration was modified"):
                score_cached(self.cache_path, model, output)
            self.assertFalse(output.exists())

    def test_input_settings_feature_order_policy_and_valid_threshold_edits_are_rejected(self):
        for variant in ("event", "lgbm_base"):
            directory = self.benchmark_path / "seed_13"
            model = directory / f"model_{variant}{'.pt' if variant == 'event' else '.txt'}"
            settings = model.with_suffix(".json" if variant == "event" else ".settings.json")
            for field in ("stream_config", "feature_names", "detection_policy", "threshold"):
                payload = read_json(settings)
                if field == "stream_config":
                    payload[field]["max_history"] += 1
                elif field == "feature_names":
                    payload[field] = list(reversed(payload[field]))
                elif field == "detection_policy":
                    payload[field]["version"] = "changed"
                else:
                    payload[field] = .123456789
                output = self.output(f"changed-{variant}-{field}")
                with replace_bytes(settings, json.dumps(payload).encode()), self.assertRaises(ValueError):
                    score_cached(self.cache_path, model, output)
                self.assertFalse(output.exists())

    def test_benchmark_failures_and_interruptions_record_status_without_completion(self):
        for index, error in enumerate((RuntimeError("training fixture failure"), KeyboardInterrupt())):
            output = self.output(f"failure-{index}")
            with patch("hamletkg.detection.neural.NeuralDetector.fit_batches", side_effect=error), self.assertRaises(type(error)):
                benchmark_cached(self.cache_path, output, seeds=(13,), variants=("event",), epochs=1, hidden_dim=8)
            manifest = read_json(output / "benchmark_manifest.json")
            self.assertEqual("interrupted" if index else "failed", manifest["status"])
            self.assertEqual([], manifest["completed_seeds"])
            self.assertFalse((output / "summary.json").exists())

    def test_prediction_failures_and_interruptions_record_score_summary_status(self):
        model = self.benchmark_path / "seed_13" / "model_event.pt"
        for index, error in enumerate((ValueError("prediction fixture failure"), KeyboardInterrupt())):
            output = self.output(f"failure-{index}")
            with patch("hamletkg.detection.neural.NeuralDetector.predict", side_effect=error), self.assertRaises(type(error)):
                score_cached(self.cache_path, model, output)
            summary = read_json(output / "summary.json")
            self.assertEqual("interrupted" if index else "failed", summary["status"])
            self.assertNotEqual("completed", summary["status"])

    def test_fixed_periodic_scaling_handles_narrow_training_phase_and_restores_with_frozen_mode(self):
        from hamletkg.detection.neural import NeuralDetector
        source, prepared, cache_path = self.output("periodic.csv"), self.output("periodic-prepared"), self.output("periodic-cache")
        with source.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Timestamp", "From Bank", "From Account", "To Bank", "To Account",
                             "Amount Paid", "Payment Currency", "Amount Received", "Receiving Currency",
                             "Payment Format", "Is Laundering"])
            for index in range(20):
                hour, minute = (0, index) if index < 12 else (6, index - 12) if index < 16 else (12, index - 16)
                writer.writerow([f"2026/01/01 {hour:02}:{minute:02}:00", 1, f"A{index % 5}",
                                 1, f"A{(index + 1) % 5}", 100 + index % 3, "USD", 100 + index % 3,
                                 "USD", "ACH", index % 2])
        prepare_detection_dataset(source, prepared, case_id="PERIODIC-TEST")
        build_sample_cache(PreparedDataset(prepared), cache_path, config=StreamConfig(max_history=4))
        cache = CachedSamples(cache_path)
        self.assertEqual({"train": 12, "validation": 4, "test": 4, "all": 20}, cache.split_counts())
        outputs, detectors = {}, {}
        for mode in ("standard", "fixed_periodic"):
            output = self.output(f"periodic-{mode}")
            benchmark_cached(cache_path, output, seeds=(13,), variants=("event",), epochs=1,
                             hidden_dim=8, batch_size=4, budget_fractions=(.25,), data_role="smoke", neural_scaling=mode)
            outputs[mode] = output
            detectors[mode] = NeuralDetector.load(output / "seed_13" / "model_event.pt")
        standard, fixed = detectors["standard"], detectors["fixed_periodic"]
        self.assertEqual("standard", standard.scaling)
        self.assertEqual("fixed_periodic", fixed.scaling)
        self.assertEqual("fixed_periodic", fixed.config["scaling"])
        self.assertEqual("fixed_periodic", fixed.training_config["scaling"])
        test = [sample for batch in cache.iter_batches("test") for sample in batch]
        cosine = BASE_FEATURE_NAMES.index("hour_cos")
        shifted = abs((test[0].features[cosine] - standard.scaler["mean"][cosine]) / standard.scaler["scale"][cosine])
        self.assertGreater(shifted, 100)
        for index, name in enumerate(BASE_FEATURE_NAMES):
            if name in ("hour_sin", "hour_cos"):
                self.assertEqual(0, fixed.scaler["mean"][index])
                self.assertEqual(1, fixed.scaler["scale"][index])
                self.assertTrue(all(abs((sample.features[index] - fixed.scaler["mean"][index]) / fixed.scaler["scale"][index]) <= 1 for sample in test))
            else:
                self.assertEqual(standard.scaler["mean"][index], fixed.scaler["mean"][index])
                self.assertEqual(standard.scaler["scale"][index], fixed.scaler["scale"][index])
        directory = outputs["fixed_periodic"] / "seed_13"
        model = directory / "model_event.pt"
        settings_path = model.with_suffix(".json")
        settings = read_json(settings_path)
        self.assertEqual("fixed_periodic", settings["neural_scaling"])
        manifest = read_json(outputs["fixed_periodic"] / "benchmark_manifest.json")
        self.assertEqual("fixed_periodic", manifest["neural_config"]["scaling"])
        restored = self.output("periodic-restored")
        result = score_cached(cache_path, model, restored, batch_size=1)
        expected, actual = predictions(directory / "predictions_event.csv"), predictions(restored / "predictions_event.csv")
        self.assertEqual(expected, actual)
        self.assertTrue(all(math.isfinite(float(row["score"])) for row in actual))
        self.assertEqual(settings["threshold"], result["threshold"])
        self.assertEqual("fixed_periodic", NeuralDetector.load(model).scaling)
        settings["neural_scaling"] = "standard"
        rejected = self.output("periodic-tampered")
        with replace_bytes(settings_path, json.dumps(settings).encode()), self.assertRaises(ValueError):
            score_cached(cache_path, model, rejected)
        self.assertFalse(rejected.exists())

    def test_invalid_parameters_and_existing_outputs_fail_before_creating_artifacts(self):
        for index, options in enumerate(({"seeds": (13, 13)}, {"seeds": (True,)}, {"variants": ("unknown",)},
                                         {"batch_size": True}, {"epochs": 0}, {"learning_rate": float("nan")},
                                         {"tree_estimators": 0}, {"budget_fractions": (True,)}, {"shuffle_buffer_batches": -1},
                                         {"neural_scaling": "unknown"}, {"neural_scaling": True})):
            output = self.output(f"invalid-{index}")
            with self.assertRaises(ValueError):
                benchmark_cached(self.cache_path, output, **options)
            self.assertFalse(output.exists())
        output = self.output("existing")
        output.mkdir()
        (output / "marker").write_text("keep")
        with self.assertRaises(ValueError):
            benchmark_cached(self.cache_path, output)
        with self.assertRaises(ValueError):
            score_cached(self.cache_path, self.benchmark_path / "seed_13" / "model_event.pt", output)
        self.assertEqual("keep", (output / "marker").read_text())

    def test_source_dataset_mismatch_is_rejected_and_provenance_must_be_a_dictionary(self):
        source, prepared = self.output("other.csv"), self.output("other-prepared")
        write_demo_csv(source, seed=100, days=6)
        prepare_detection_dataset(source, prepared, case_id="DISK-TEST")
        output = self.output("mismatch")
        with self.assertRaisesRegex(ValueError, "do not belong together"):
            benchmark_cached(self.cache_path, output, dataset_path=prepared)
        self.assertFalse(output.exists())
        with self.assertRaisesRegex(ValueError, "policy_provenance"):
            build_sample_cache(self.dataset, self.output("invalid-provenance"), policy_provenance=["unexpected"])

    def test_labeled_batches_reject_alignment_errors(self):
        cache = self.cache
        with patch.object(cache, "iter_labels", return_value=iter([0])), self.assertRaisesRegex(ValueError, "aligned"):
            list(labeled_batches(cache, "train", 32))
        with patch.object(cache, "iter_labels", return_value=iter([*self.training_labels, 0])), self.assertRaisesRegex(ValueError, "without corresponding"):
            list(labeled_batches(cache, "train", 32))


@unittest.skipUnless(HAS_LIGHTGBM and HAS_NUMPY, "optional tabular runtimes unavailable")
class LightGBMMatrixTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        source, prepared, cache_path = self.root / "fixture.csv", self.root / "prepared", self.root / "cache"
        write_demo_csv(source, days=6)
        prepare_detection_dataset(source, prepared)
        build_sample_cache(PreparedDataset(prepared), cache_path)
        cache = CachedSamples(cache_path)
        self.training = [sample for batch in cache.iter_batches("train") for sample in batch]
        self.labels = list(cache.iter_labels("train"))

    def test_readonly_memmap_fit_matches_sample_fit_and_native_restore(self):
        import numpy as np
        for variant in ("lgbm_base", "lgbm_flat"):
            reference = LightGBMDetector(BASE_FEATURE_NAMES, variant=variant, seed=13, n_estimators=3, num_leaves=4)
            reference.fit(self.training, self.labels)
            rows = np.asarray([reference.feature_vector(sample) for sample in self.training], dtype=np.float64)
            matrix_path, label_path = self.root / f"{variant}.npy", self.root / f"labels_{variant}.npy"
            writer = np.lib.format.open_memmap(matrix_path, mode="w+", dtype=np.float64, shape=rows.shape)
            writer[:] = rows
            writer.flush()
            del writer
            np.save(label_path, np.asarray(self.labels, dtype=np.int8), allow_pickle=False)
            matrix, labels = np.load(matrix_path, mmap_mode="r", allow_pickle=False), np.load(label_path, mmap_mode="r", allow_pickle=False)
            self.assertFalse(matrix.flags.writeable)
            detector = LightGBMDetector(BASE_FEATURE_NAMES, variant=variant, seed=13, n_estimators=3, num_leaves=4)
            history = detector.fit_matrix(matrix, labels, working_directory=self.root, chunk_rows=7)
            self.assertTrue(history)
            np.testing.assert_allclose(reference.scaler["mean"], detector.scaler["mean"], atol=1e-12)
            np.testing.assert_allclose(reference.scaler["scale"], detector.scaler["scale"], atol=1e-12)
            np.testing.assert_allclose(reference.predict(self.training), detector.predict(self.training), atol=1e-15)
            self.assertEqual(reference.training_config["class_weights"], detector.training_config["class_weights"])
            self.assertEqual("disk_backed_feature_matrix", detector.training_config["training_mode"])
            model = self.root / f"model_{variant}.txt"
            detector.save(model)
            restored = LightGBMDetector.load(model)
            self.assertEqual(detector.predict(self.training), restored.predict(self.training))
            np.testing.assert_array_equal(rows, matrix)
            self.assertFalse(any(path.name.startswith("hamlet-tree-") for path in self.root.iterdir()))

    def test_matrix_validation_rejects_bool_nan_bad_shapes_and_unknown_labels(self):
        import numpy as np
        valid = np.asarray([sample.features for sample in self.training], dtype=np.float64)
        labels = np.asarray(self.labels, dtype=np.int32)
        bad = [
            (valid[:0], labels[:0]), (valid[:, :23], labels), (valid[0], labels),
            (valid, labels.reshape(-1, 1)), (valid, labels.astype(np.float64)),
            (valid, labels.astype(np.bool_)), (valid.astype(np.bool_), labels),
            (valid.astype(object), labels), (valid, labels[:-1]),
            (valid, np.full(labels.shape, 2, dtype=np.int32)),
            (valid, np.zeros_like(labels)),
        ]
        for value in (float("nan"), float("inf")):
            changed = valid.copy()
            changed[-1, -1] = value
            bad.append((changed, labels))
        for index, (features, target) in enumerate(bad):
            detector = LightGBMDetector(BASE_FEATURE_NAMES, n_estimators=3)
            with self.subTest(index=index), self.assertRaises(ValueError):
                detector.fit_matrix(features, target, working_directory=self.root, chunk_rows=7)
            self.assertIsNone(detector.model)
            self.assertIsNone(detector.scaler)
        for chunk_rows in (0, -1, True, 1.5):
            with self.subTest(chunk_rows=chunk_rows), self.assertRaises(ValueError):
                LightGBMDetector(BASE_FEATURE_NAMES).fit_matrix(valid, labels, chunk_rows=chunk_rows)
        self.assertFalse(any(path.name.startswith("hamlet-tree-") for path in self.root.iterdir()))


if __name__ == "__main__":
    unittest.main()
