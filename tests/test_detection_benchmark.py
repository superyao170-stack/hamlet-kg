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

from hamletkg.cli import build_parser, main
from hamletkg.detection.benchmark import benchmark_detection
from hamletkg.detection.demo import write_demo_csv
from hamletkg.detection.metrics import chronological_split
from hamletkg.detection.runner import load_detection_case, score_detection
from hamletkg.detection.stream import BASE_FEATURE_NAMES, build_samples


HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_LIGHTGBM = importlib.util.find_spec("lightgbm") is not None


def read_json(filename):
    return json.loads(filename.read_text(encoding="utf-8"))


def read_predictions(filename):
    with filename.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class BenchmarkCLIInputTests(unittest.TestCase):
    def test_audit_cli_limits_loaded_rows_but_hashes_full_source_and_keeps_missing_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "transactions.csv", root / "audit.json"
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow([
                    "Timestamp", "From Bank", "From Account", "To Bank", "To Account",
                    "Amount Paid", "Payment Currency", "Amount Received", "Receiving Currency",
                    "Payment Format", "Is Laundering",
                ])
                for day, label in ((1, 1), (2, ""), (3, 0)):
                    writer.writerow([f"2026/01/{day:02d} 10:00:00", 1, f"sender-{day}",
                                     1, f"receiver-{day}", 100, "USD", 100, "USD", "ACH", label])
            full_source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
            stdout = io.StringIO()
            with patch("hamletkg.cli._llm_client", side_effect=AssertionError("audit must not use LLM")), \
                    patch("hamletkg.detection.stream.build_samples", side_effect=AssertionError("audit must not build features")), \
                    contextlib.redirect_stdout(stdout):
                code = main(["audit-data", "--csv", str(source), "--output", str(output), "--limit", "2"])
            self.assertEqual(0, code)
            audit = read_json(output)
            self.assertEqual(2, audit["input_row_limit"])
            self.assertEqual(2, audit["counts"]["transfer_candidates"])
            self.assertEqual(full_source_digest, audit["source"]["sha256"])
            self.assertEqual((1, 1, 1, 0), tuple(audit["labels"][key] for key in ("labeled", "unlabeled", "positive", "negative")))
            self.assertFalse(audit["labels"]["unlabeled_is_negative"])
            self.assertEqual("2026-01-02T10:00:00Z", audit["time"]["end"])
            self.assertEqual(2, json.loads(stdout.getvalue())["counts"]["transfer_candidates"])

    def test_benchmark_parser_exposes_mixed_backends_and_explicit_data_role(self):
        args = build_parser().parse_args([
            "detect-benchmark", "--csv", "sample.csv", "--output", "benchmark",
            "--seeds", "5", "8", "--variants", "event", "lgbm_flat",
            "--tree-estimators", "7", "--tree-learning-rate", "0.1",
            "--tree-leaves", "4", "--data-role", "smoke", "--limit", "300",
        ])
        self.assertEqual([5, 8], args.seeds)
        self.assertEqual(["event", "lgbm_flat"], args.variants)
        self.assertEqual((7, 0.1, 4, "smoke", 300),
                         (args.tree_estimators, args.tree_learning_rate, args.tree_leaves,
                          args.data_role, args.limit))
        self.assertIsNone(args.case)

    def test_parser_and_benchmark_module_keep_optional_runtimes_lazy(self):
        program = (
            "import sys; from hamletkg.cli import build_parser; "
            "from hamletkg.detection.benchmark import benchmark_detection; "
            "build_parser().parse_args(['detect-benchmark','--csv','x','--output','y']); "
            "assert 'torch' not in sys.modules; assert 'lightgbm' not in sys.modules; "
            "assert 'numpy' not in sys.modules"
        )
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        result = subprocess.run([sys.executable, "-c", program], env=environment,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(0, result.returncode, result.stderr)


@unittest.skipUnless(HAS_TORCH, "optional PyTorch not installed")
class DetectionBenchmarkIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.original_thread_count = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        import torch
        torch.set_num_threads(cls.original_thread_count)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "input.csv"
        write_demo_csv(self.source, days=6)
        self.case = load_detection_case(csv_path=self.source, case_id="BENCHMARK-TEST")

    def benchmark(self, name="benchmark", **kwargs):
        options = {
            "seeds": (13, 42), "variants": ("base", "event"), "epochs": 1,
            "hidden_dim": 8, "batch_size": 32, "budget_fractions": (0.05, 0.2),
            "tree_estimators": 3, "tree_leaves": 4, "source_path": self.source,
            "data_role": "smoke",
        }
        options.update(kwargs)
        return benchmark_detection(self.case, self.root / name, **options)

    def test_fixed_samples_train_and_threshold_boundaries_and_multiseed_summary(self):
        from hamletkg.detection.neural import NeuralDetector
        from hamletkg.detection.metrics import choose_threshold

        built, fitted, predicted = [], [], []
        original_fit, original_predict = NeuralDetector.fit, NeuralDetector.predict

        def capture_build(*args, **kwargs):
            samples = build_samples(*args, **kwargs)
            built.append(samples)
            return samples

        def capture_fit(detector, samples, labels, **kwargs):
            fitted.append((detector.seed, detector.variant, list(samples), list(labels)))
            return original_fit(detector, samples, labels, **kwargs)

        def capture_predict(detector, samples):
            predicted.append((detector.seed, detector.variant, list(samples)))
            return original_predict(detector, samples)

        with patch("hamletkg.detection.benchmark.build_samples", side_effect=capture_build) as builder, \
                patch.object(NeuralDetector, "fit", autospec=True, side_effect=capture_fit), \
                patch.object(NeuralDetector, "predict", autospec=True, side_effect=capture_predict), \
                patch("hamletkg.detection.benchmark.choose_threshold", wraps=choose_threshold) as chooser:
            result = self.benchmark()
        builder.assert_called_once()
        splits = chronological_split(built[0])
        expected_ids = {split: [built[0][index].event_id for index in indices]
                        for split, indices in splits.items()}
        train_labels = [self.case.metadata["evaluation_labels"][event] for event in expected_ids["train"]]
        validation_labels = [self.case.metadata["evaluation_labels"][event] for event in expected_ids["validation"]]
        self.assertEqual(4, len(fitted))
        for seed, variant, samples, labels in fitted:
            with self.subTest(seed=seed, variant=variant):
                self.assertEqual(expected_ids["train"], [sample.event_id for sample in samples])
                self.assertEqual(train_labels, labels)
                self.assertLess(max(sample.timestamp for sample in samples),
                                min(built[0][index].timestamp for index in splits["validation"]))
                self.assertTrue(all(sample is built[0][index] for sample, index in zip(samples, splits["train"])))
        self.assertEqual(8, len(predicted))
        for offset, (_, _, samples) in enumerate(predicted):
            split = "validation" if offset % 2 == 0 else "test"
            self.assertEqual(expected_ids[split], [sample.event_id for sample in samples])
        self.assertEqual(4, chooser.call_count)
        for call in chooser.call_args_list:
            self.assertEqual(validation_labels, call.args[0])
            self.assertEqual(len(validation_labels), len(call.args[1]))

        directory = self.root / "benchmark"
        manifest, summary = read_json(directory / "benchmark_manifest.json"), read_json(directory / "summary.json")
        self.assertEqual("completed", result["status"])
        self.assertEqual(("completed", [13, 42]), (manifest["status"], manifest["completed_seeds"]))
        self.assertEqual(expected_ids, read_json(directory / "splits.json"))
        self.assertEqual(270, result["transactions"])
        self.assertEqual("smoke", summary["data_role"])
        self.assertEqual(64, len(summary["source_sha256"]))
        for variant in ("base", "event"):
            values = [read_json(directory / f"seed_{seed}" / "metrics.json")[variant]["test"]["average_precision"]
                      for seed in (13, 42)]
            stats = summary["variants"][variant]["test"]["average_precision"]
            self.assertAlmostEqual(sum(values) / 2, stats["mean"])
            self.assertAlmostEqual(abs(values[0] - values[1]) / math.sqrt(2), stats["std"])
            self.assertEqual((2, 0), (stats["n"], stats["n_missing"]))
            for seed in (13, 42):
                sidecar = read_json(directory / f"seed_{seed}" / f"model_{variant}.json")
                stats_threshold = next(item["value"] for item in summary["variants"][variant]["thresholds"]["validation"]
                                       if item["seed"] == seed)
                self.assertEqual(sidecar["threshold"], stats_threshold)
        self.assertEqual([13, 42], summary["paired_deltas"]["base"]["common_seeds"])
        self.assertEqual(2, summary["paired_deltas"]["base"]["test_average_precision"]["n"])
        self.assertTrue((directory / "data_audit.json").is_file())
        self.assertTrue((directory / "summary.csv").is_file())
        self.assertIn("not independent test datasets", (directory / "summary.md").read_text())

    def test_changing_only_test_labels_cannot_change_model_or_selected_threshold(self):
        self.benchmark(name="original", seeds=(13,), variants=("base",))
        original = read_json(self.root / "original" / "seed_13" / "model_base.json")
        samples = build_samples(self.case)
        splits = chronological_split(samples)
        labels = self.case.metadata["evaluation_labels"]
        for index in splits["test"]:
            labels[samples[index].event_id] = 1 - labels[samples[index].event_id]
        self.benchmark(name="changed", seeds=(13,), variants=("base",))
        changed = read_json(self.root / "changed" / "seed_13" / "model_base.json")
        self.assertEqual(original["threshold"], changed["threshold"])
        first = read_predictions(self.root / "original" / "seed_13" / "predictions_base.csv")
        second = read_predictions(self.root / "changed" / "seed_13" / "predictions_base.csv")
        self.assertEqual([row["score"] for row in first], [row["score"] for row in second])
        self.assertEqual([row["threshold_alert"] for row in first], [row["threshold_alert"] for row in second])
        self.assertTrue(all(left["label"] != right["label"] for left, right in zip(first, second)))

    @unittest.skipUnless(HAS_LIGHTGBM, "optional LightGBM not installed")
    def test_mixed_backends_restore_label_free_and_tree_scaler_uses_only_training_rows(self):
        variants = ("event", "lgbm_base", "lgbm_flat")
        self.benchmark(seeds=(13, 42), variants=variants)
        directory = self.root / "benchmark"
        manifest = read_json(directory / "benchmark_manifest.json")
        split_ids = read_json(directory / "splits.json")
        samples = build_samples(self.case)
        training = [sample for sample in samples if sample.event_id in set(split_ids["train"])]
        expected_mean = [sum(sample.features[index] for sample in training) / len(training)
                         for index in range(len(BASE_FEATURE_NAMES))]
        summary = read_json(directory / "summary.json")
        for variant in variants:
            self.assertEqual(2, summary["variants"][variant]["test"]["average_precision"]["n"])
        for baseline in ("lgbm_base", "lgbm_flat"):
            self.assertEqual(2, summary["paired_deltas"][baseline]["test_average_precision"]["n"])
            detector_sidecar = read_json(directory / "seed_13" / f"model_{baseline}.json")
            self.assertEqual(len(training), detector_sidecar["training_config"]["training_sample_count"])
            for actual, expected in zip(detector_sidecar["scaler"]["mean"], expected_mean):
                self.assertAlmostEqual(expected, actual, places=10)
        self.case.metadata.clear()
        for variant in variants:
            with self.subTest(variant=variant):
                model = directory / "seed_13" / f"model_{variant}{'.pt' if variant == 'event' else '.txt'}"
                restored = self.root / f"restored_{variant}"
                score_detection(self.case, model, restored, after=manifest["split_summary"]["test"]["first_time"],
                                explain_top_k=0)
                original = read_predictions(directory / "seed_13" / f"predictions_{variant}.csv")
                restored_rows = read_predictions(restored / "predictions.csv")
                self.assertEqual([row["event_id"] for row in original], [row["event_id"] for row in restored_rows])
                for left, right in zip(original, restored_rows):
                    self.assertAlmostEqual(float(left["score"]), float(right["score"]), places=6)
                    self.assertEqual(left["threshold_alert"], right["threshold_alert"])
                    self.assertEqual("", right["label"])

    @unittest.skipUnless(HAS_LIGHTGBM, "optional LightGBM not installed")
    def test_restoring_tree_rejects_model_sidecar_stream_and_variant_tampering_before_output(self):
        self.benchmark(seeds=(13,), variants=("lgbm_base",))
        model = self.root / "benchmark" / "seed_13" / "model_lgbm_base.txt"
        sidecar, settings = model.with_suffix(".json"), model.with_suffix(".settings.json")
        originals = {item: item.read_bytes() for item in (model, sidecar, settings)}

        def modify_settings(field, value):
            payload = read_json(settings)
            payload[field] = value
            settings.write_text(json.dumps(payload), encoding="utf-8")

        def modify_stream():
            payload = read_json(settings)
            payload["stream_config"]["max_history"] += 1
            settings.write_text(json.dumps(payload), encoding="utf-8")

        def modify_scaler():
            payload = read_json(sidecar)
            payload["scaler"]["mean"][0] += 1
            sidecar.write_text(json.dumps(payload), encoding="utf-8")

        changes = (
            ("model", lambda: model.write_bytes(originals[model] + b"\n# altered\n"), "same saved model"),
            ("sidecar", modify_scaler, "frozen saved configuration"),
            ("stream", modify_stream, "frozen training settings"),
            ("variant", lambda: modify_settings("variant", "lgbm_flat"), "variants disagree"),
            ("feature_order", lambda: modify_settings("feature_names", list(reversed(BASE_FEATURE_NAMES))), "feature order"),
        )
        for name, change, error in changes:
            with self.subTest(change=name):
                for filename, contents in originals.items():
                    filename.write_bytes(contents)
                change()
                destination = self.root / f"invalid_{name}"
                with self.assertRaisesRegex(ValueError, error):
                    score_detection(self.case, model, destination, explain_top_k=0)
                self.assertFalse(destination.exists())

    def test_failure_or_interrupt_after_completed_seed_records_status_and_partial_results(self):
        from hamletkg.detection.neural import NeuralDetector
        original_fit = NeuralDetector.fit

        for exception_class, status in ((RuntimeError, "failed"), (KeyboardInterrupt, "interrupted")):
            with self.subTest(exception=exception_class.__name__):
                def fail_second_seed(detector, samples, labels, **kwargs):
                    if detector.seed == 42:
                        raise exception_class("injected fit failure for experiment status test")
                    return original_fit(detector, samples, labels, **kwargs)

                with patch.object(NeuralDetector, "fit", autospec=True, side_effect=fail_second_seed), \
                        self.assertRaisesRegex(exception_class, "injected fit failure"):
                    self.benchmark(name=status, variants=("base",))
                directory = self.root / status
                manifest = read_json(directory / "benchmark_manifest.json")
                self.assertEqual((status, exception_class.__name__, [13]),
                                 (manifest["status"], manifest["failure_type"], manifest["completed_seeds"]))
                self.assertTrue((directory / "seed_13" / "metrics.json").is_file())
                for filename in ("summary.json", "summary.csv", "summary.md"):
                    self.assertFalse((directory / filename).exists())

    def test_neural_checkpoint_and_frozen_stream_tampering_are_rejected_before_output(self):
        self.benchmark(seeds=(13,), variants=("base",))
        model = self.root / "benchmark" / "seed_13" / "model_base.pt"
        settings = model.with_suffix(".json")
        original_model, original_settings = model.read_bytes(), settings.read_bytes()
        model.write_bytes(original_model + b"changed")
        with self.assertRaisesRegex(ValueError, "same saved model"):
            score_detection(self.case, model, self.root / "bad_neural_model", explain_top_k=0)
        self.assertFalse((self.root / "bad_neural_model").exists())
        model.write_bytes(original_model)
        payload = json.loads(original_settings)
        payload["stream_config"]["window_seconds"] += 1
        settings.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "frozen training settings"):
            score_detection(self.case, model, self.root / "bad_neural_stream", explain_top_k=0)
        self.assertFalse((self.root / "bad_neural_stream").exists())

    def test_single_class_validation_rejected_before_any_training_or_manifest(self):
        samples = build_samples(self.case)
        for index in chronological_split(samples)["validation"]:
            self.case.metadata["evaluation_labels"][samples[index].event_id] = 0
        with patch("hamletkg.detection.neural.NeuralDetector.fit", side_effect=AssertionError("invalid split must not train")), \
                self.assertRaisesRegex(ValueError, "validation needs both"):
            self.benchmark()
        self.assertFalse((self.root / "benchmark").exists())

    def test_invalid_preflight_inputs_and_missing_labels_do_not_publish_manifest(self):
        for options in ({"seeds": (13, 13)}, {"variants": ("base", "base")},
                        {"budget_fractions": (0.1, 0.1)}, {"data_role": "publication_proof"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.benchmark(**options)
            self.assertFalse((self.root / "benchmark").exists())
        self.case.metadata["evaluation_labels"].pop(next(iter(self.case.atomic_events)))
        with self.assertRaisesRegex(ValueError, "transaction-level label"):
            self.benchmark()
        self.assertFalse((self.root / "benchmark").exists())
        self.case.metadata["evaluation_labels"][next(iter(self.case.atomic_events))] = True
        with self.assertRaisesRegex(ValueError, "transaction-level label"):
            self.benchmark()
        self.assertFalse((self.root / "benchmark").exists())

    def test_single_class_training_rejected_and_existing_output_preserved(self):
        self.case.metadata["evaluation_labels"] = {event: 0 for event in self.case.atomic_events}
        with self.assertRaisesRegex(ValueError, "train needs both"):
            self.benchmark()
        self.assertFalse((self.root / "benchmark").exists())
        existing = self.root / "benchmark"
        existing.mkdir()
        sentinel = existing / "existing_result.json"
        sentinel.write_text("preserve this existing result", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "new or empty directory"):
            self.benchmark()
        self.assertEqual("preserve this existing result", sentinel.read_text())

    @unittest.skipUnless(HAS_LIGHTGBM, "optional LightGBM not installed")
    def test_cli_executes_tree_only_benchmark_without_llm_client(self):
        output = self.root / "cli-benchmark"
        stdout = io.StringIO()
        with patch("hamletkg.cli._llm_client", side_effect=AssertionError("benchmark must not use LLM")), \
                contextlib.redirect_stdout(stdout):
            code = main([
                "detect-benchmark", "--csv", str(self.source), "--output", str(output),
                "--seeds", "13", "--variants", "lgbm_base", "--tree-estimators", "3",
                "--tree-leaves", "4", "--data-role", "smoke", "--budget-fractions", "0.2",
            ])
        self.assertEqual(0, code)
        result = json.loads(stdout.getvalue())
        self.assertEqual(("completed", ["lgbm_base"], "smoke"),
                         (result["status"], result["variants"], result["data_role"]))
        self.assertTrue((output / "seed_13" / "model_lgbm_base.settings.json").is_file())


if __name__ == "__main__":
    unittest.main()
