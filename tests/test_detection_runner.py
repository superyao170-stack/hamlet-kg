from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from hamletkg.cli import build_parser
from hamletkg.detection.demo import write_demo_csv
from hamletkg.detection.runner import load_detection_case, score_detection, train_detection


class DetectionCLIInputTests(unittest.TestCase):
    def test_training_inputs_are_exclusive_and_optional_neural_import_is_lazy(self):
        parser = build_parser()
        args = parser.parse_args(["detect-train", "--csv", "x.csv", "--output", "out"])
        self.assertEqual(["event", "base", "flat"], args.variants)
        self.assertEqual("balanced", args.class_weight)
        self.assertIsNone(args.case)
        with self.assertRaises(ValueError):
            load_detection_case(csv_path="x", case_path="y")

    def test_demo_generation_is_deterministic_and_labeled_without_existing_data(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "a.csv", Path(directory) / "b.csv"
            info = write_demo_csv(first, days=6)
            self.assertEqual(info, write_demo_csv(second, days=6))
            self.assertEqual(first.read_bytes(), second.read_bytes())
            case = load_detection_case(csv_path=first, case_id="TEST")
            self.assertEqual(270, len(case.atomic_events))
            self.assertEqual([], case.atomic_relations)
            self.assertEqual(30, sum(case.metadata["evaluation_labels"].values()))


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch not installed")
class DetectionRunnerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "input.csv"
        write_demo_csv(self.source, days=6)
        self.case = load_detection_case(csv_path=self.source, case_id="TEST")

    def train(self, variants=("event",)):
        return train_detection(self.case, self.root / "train", epochs=1, hidden_dim=8,
                               variants=variants, explain_top_k=0, source_path=self.source)

    def test_train_restore_score_and_artifact_semantics(self):
        result = self.train(("base", "flat", "event"))
        self.assertEqual(270, result["transactions"])
        output = self.root / "train"
        manifest = json.loads((output / "manifest.json").read_text())
        metrics = json.loads((output / "metrics.json").read_text())
        self.assertEqual(64, len(manifest["source_sha256"]))
        for variant in ("base", "flat", "event"):
            self.assertEqual("balanced", metrics[variant]["training_config"]["class_weight"])
            summary = manifest["split_summary"]["train"]
            expected_weight = (summary["transactions"] - summary["positives"]) / summary["positives"]
            self.assertAlmostEqual(expected_weight, metrics[variant]["training_config"]["positive_weight"])
            self.assertTrue((output / f"model_{variant}.pt").exists())
        with (output / "predictions_event.csv").open(newline="") as handle:
            training_predictions = list(csv.DictReader(handle))
        after = manifest["split_summary"]["test"]["first_time"]
        score_detection(self.case, output / "model_event.pt", self.root / "score",
                        after=after, explain_top_k=0)
        with (self.root / "score" / "predictions.csv").open(newline="") as handle:
            restored_predictions = list(csv.DictReader(handle))
        self.assertEqual([row["event_id"] for row in training_predictions],
                         [row["event_id"] for row in restored_predictions])
        for trained, restored in zip(training_predictions, restored_predictions):
            self.assertAlmostEqual(float(trained["score"]), float(restored["score"]), places=6)
            self.assertEqual("", restored["label"])
        self.case.metadata.clear()
        # Scoring is label-free; the very same model accepts missing metadata.
        score_detection(self.case, output / "model_event.pt", self.root / "unlabeled",
                        after=after, explain_top_k=0)

    def test_missing_labels_and_single_class_training_do_not_write_success_artifacts(self):
        self.case.metadata["evaluation_labels"].pop(next(iter(self.case.atomic_events)))
        with self.assertRaisesRegex(ValueError, "transaction-level label"):
            self.train()
        self.assertFalse((self.root / "train").exists())
        self.case.metadata["evaluation_labels"] = {key: 0 for key in self.case.atomic_events}
        with self.assertRaisesRegex(ValueError, "train needs both"):
            self.train()
        self.assertFalse((self.root / "train").exists())

    def test_wrong_checkpoint_pair_and_invalid_explanation_settings_fail_before_output(self):
        self.train()
        checkpoint = self.root / "train" / "model_event.pt"
        for kwargs in ({"explain_top_k": -1}, {"explain_top_k": 0, "explanation_max_candidates": 0}):
            with self.assertRaises(ValueError):
                score_detection(self.case, checkpoint, self.root / "bad-score", **kwargs)
            self.assertFalse((self.root / "bad-score").exists())
        sidecar = checkpoint.with_suffix(".json")
        settings = json.loads(sidecar.read_text())
        settings["model_sha256"] = "0" * 64
        sidecar.write_text(json.dumps(settings))
        with self.assertRaisesRegex(ValueError, "same saved model"):
            score_detection(self.case, checkpoint, self.root / "bad-pair", explain_top_k=0)
        self.assertFalse((self.root / "bad-pair").exists())

    def test_loaded_feature_order_must_match_sidecar(self):
        from hamletkg.detection.neural import NeuralDetector
        from hamletkg.detection.runner import _source_digest
        self.train()
        checkpoint = self.root / "train" / "model_event.pt"
        detector = NeuralDetector.load(checkpoint)
        detector.feature_names = tuple(reversed(detector.feature_names))
        detector.save(checkpoint)
        sidecar = checkpoint.with_suffix(".json")
        settings = json.loads(sidecar.read_text())
        settings["model_sha256"] = _source_digest(checkpoint)
        sidecar.write_text(json.dumps(settings))
        with self.assertRaisesRegex(ValueError, "feature orders"):
            score_detection(self.case, checkpoint, self.root / "bad-order", explain_top_k=0)
        self.assertFalse((self.root / "bad-order").exists())


if __name__ == "__main__":
    unittest.main()
