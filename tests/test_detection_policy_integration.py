from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from hamletkg.detection.evidence import explain_transaction
from hamletkg.detection.policy import DetectionPolicy, DetectionRule
from hamletkg.detection.runner import score_detection, train_detection
from hamletkg.detection.stream import BASE_FEATURE_NAMES, StreamConfig, build_samples
from hamletkg.models import AtomicEvent, CaseGraph, Entity


def digest(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def training_case():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    events, labels = {}, {}
    for day in range(6):
        for step in range(4):
            event_id = f"D{day}-T{step}"
            sender = f"S{day}-{step}" if step < 3 else f"H{day}"
            receiver = f"H{day}" if step < 3 else f"Z{day}"
            events[event_id] = AtomicEvent(
                event_id, "TRANSFER", {"sender": sender, "receiver": receiver},
                start + timedelta(days=day, seconds=step * 10),
                {"amount": 30 if step < 3 else 90, "currency": "USD"}, event_id,
            )
            labels[event_id] = day % 2
    accounts = {account for event in events.values() for account in event.roles.values()}
    return CaseGraph("POLICY-TOY", {account: Entity(account, "ACCOUNT") for account in accounts},
                     events, metadata={"evaluation_labels": labels})


def pass_policy():
    return DetectionPolicy("ONLY-PASS", "1", (
        DetectionRule("PASS-35S", "pass_through", 35, min_incoming=2,
                      ratio_min=.9, ratio_max=1.1),
    ))


class RecordingDetector:
    def __init__(self):
        self.seen = []

    def predict(self, samples):
        self.seen.extend(samples)
        return [min(.99, .1 + .1 * sum(len(event.members) - 1 for event in sample.events))
                for sample in samples]


class PolicyEvidenceIntegrationTests(unittest.TestCase):
    def test_all_deletion_replays_receive_the_same_policy_and_exclude_disabled_operators(self):
        case = training_case()
        config = StreamConfig(window_seconds=100, max_history=8)
        policy = pass_policy()
        sample = next(item for item in build_samples(case, config, policy=policy)
                      if item.event_id == "D2-T3")
        detector = RecordingDetector()
        with patch("hamletkg.detection.evidence.build_samples", wraps=build_samples) as replay:
            result = explain_transaction(case, sample, detector, config, policy=policy, seed=7)
        self.assertEqual(3, replay.call_count)
        for call in replay.call_args_list:
            self.assertIs(policy, call.kwargs["policy"])
            self.assertNotIn(sample.event_id, call.kwargs["excluded_ids"])
        for seen in detector.seen:
            self.assertEqual("D2-T3", seen.event_id)
            self.assertTrue(all(event.motif_type == "pass_through" for event in seen.events))
            self.assertTrue(all(event.rule_id == "PASS-35S" for event in seen.events))
        audit = result["audited_candidates"][0]
        self.assertEqual(["D2-T0", "D2-T1", "D2-T2"], audit["historical_evidence_ids"])
        self.assertAlmostEqual(.1, audit["without_evidence_score"])
        self.assertAlmostEqual(result["score"], audit["evidence_only_score"])
        event_nodes = [node for node in audit["graph"]["nodes"] if node["type"] == "BEHAVIOR_EVENT"]
        self.assertEqual(["PASS-35S"], [node["rule_id"] for node in event_nodes])
        self.assertTrue(all(node["candidate_only"] for node in event_nodes))

    def test_policy_disabled_candidate_does_not_create_a_default_explanation(self):
        case = training_case()
        config = StreamConfig(window_seconds=100, max_history=8)
        policy = pass_policy()
        event_id = "D2-T2"
        default = next(item for item in build_samples(case, config) if item.event_id == event_id)
        self.assertEqual({"fan_in"}, {event.motif_type for event in default.events})
        sample = next(item for item in build_samples(case, config, policy=policy) if item.event_id == event_id)
        with patch("hamletkg.detection.evidence.build_samples", wraps=build_samples) as replay:
            result = explain_transaction(case, sample, RecordingDetector(), config, policy=policy)
        replay.assert_not_called()
        self.assertEqual("no_event_candidate", result["status"])
        self.assertEqual([], result["audited_candidates"])
        self.assertFalse(result["positive_support_found"])


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch not installed")
class PolicyCheckpointIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        import torch
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.case = training_case()
        self.config = StreamConfig(window_seconds=100, max_history=8)
        self.policy = pass_policy()

    def train(self, policy):
        train_detection(self.case, self.root / "train", config=self.config,
                        policy=policy, variants=("event",), epochs=1, batch_size=8,
                        hidden_dim=8, seed=7, explain_top_k=0, budget_fractions=(.5,),
                        fixture_info={"kind": "unit_test_fixture_not_research_evidence"})
        return self.root / "train" / "model_event.pt"

    def rows(self, path):
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))

    def test_training_binds_policy_and_stream_and_scoring_restores_same_events_and_scores(self):
        from hamletkg.detection.neural import NeuralDetector
        checkpoint = self.train(self.policy)
        manifest = json.loads((checkpoint.parent / "manifest.json").read_text())
        sidecar = json.loads(checkpoint.with_suffix(".json").read_text())
        detector = NeuralDetector.load(checkpoint)
        self.assertEqual(self.policy.to_dict(), manifest["detection_policy"])
        self.assertEqual(self.policy.to_dict(), sidecar["detection_policy"])
        self.assertEqual(digest(self.policy.to_dict()), manifest["policy_sha256"])
        self.assertEqual(digest(self.policy.to_dict()), detector.training_config["policy_sha256"])
        self.assertEqual(digest(asdict(self.config)), detector.training_config["stream_config_sha256"])
        self.assertEqual(list(BASE_FEATURE_NAMES), sidecar["feature_names"])
        original = self.rows(checkpoint.parent / "predictions_event.csv")
        after = manifest["split_summary"]["test"]["first_time"]
        # No labels may be consulted during restored inference.
        self.case.metadata.clear()
        score_detection(self.case, checkpoint, self.root / "score", after=after,
                        explain_top_k=len(original))
        restored = self.rows(self.root / "score" / "predictions.csv")
        self.assertEqual([row["event_id"] for row in original], [row["event_id"] for row in restored])
        for before, actual in zip(original, restored):
            self.assertAlmostEqual(float(before["score"]), float(actual["score"]), places=6)
            self.assertEqual(before["candidate_events"], actual["candidate_events"])
            self.assertEqual("", actual["label"])
        evidence = json.loads((self.root / "score" / "evidence.json").read_text())
        for transaction in evidence["transactions"]:
            for audit in transaction["audited_candidates"]:
                self.assertEqual("pass_through", audit["motif_type"])
                event_nodes = [node for node in audit["graph"]["nodes"] if node["type"] == "BEHAVIOR_EVENT"]
                self.assertEqual(["PASS-35S"], [node["rule_id"] for node in event_nodes])

    def test_valid_but_changed_or_removed_policy_is_rejected_before_scoring_outputs(self):
        checkpoint = self.train(self.policy)
        sidecar = checkpoint.with_suffix(".json")
        original = json.loads(sidecar.read_text())
        variants = []
        changed_window = copy.deepcopy(original)
        changed_window["detection_policy"]["rules"][0]["window_seconds"] = 34
        variants.append(changed_window)
        changed_ratio = copy.deepcopy(original)
        changed_ratio["detection_policy"]["rules"][0]["ratio_max"] = 1.2
        variants.append(changed_ratio)
        changed_operator = copy.deepcopy(original)
        changed_operator["detection_policy"] = DetectionPolicy("ONLY-FAN", "1", (
            DetectionRule("FAN", "fan_in", 35),
        )).to_dict()
        variants.append(changed_operator)
        removed = copy.deepcopy(original)
        removed.pop("detection_policy")
        variants.append(removed)
        for index, settings in enumerate(variants):
            sidecar.write_text(json.dumps(settings), encoding="utf-8")
            output = self.root / f"tampered-policy-{index}"
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "frozen training policy"):
                score_detection(self.case, checkpoint, output, explain_top_k=0)
            self.assertFalse(output.exists())

    def test_changed_stream_settings_are_rejected_even_if_policy_remains_valid(self):
        checkpoint = self.train(self.policy)
        sidecar = checkpoint.with_suffix(".json")
        original = json.loads(sidecar.read_text())
        for field, value in (("window_seconds", 90), ("max_history", 9), ("min_fan", 4)):
            settings = copy.deepcopy(original)
            settings["stream_config"][field] = value
            sidecar.write_text(json.dumps(settings), encoding="utf-8")
            output = self.root / f"tampered-stream-{field}"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "frozen training settings"):
                score_detection(self.case, checkpoint, output, explain_top_k=0)
            self.assertFalse(output.exists())

    def test_legacy_no_policy_checkpoint_without_policy_bindings_still_scores(self):
        from hamletkg.detection.neural import NeuralDetector
        checkpoint = self.train(None)
        detector = NeuralDetector.load(checkpoint)
        detector.training_config.pop("policy_sha256")
        detector.training_config.pop("stream_config_sha256")
        detector.save(checkpoint)
        sidecar = checkpoint.with_suffix(".json")
        settings = json.loads(sidecar.read_text())
        settings.pop("detection_policy")
        settings["model_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        sidecar.write_text(json.dumps(settings), encoding="utf-8")
        result = score_detection(self.case, checkpoint, self.root / "legacy-score", explain_top_k=0)
        self.assertEqual(24, result["transactions"])
        expected = detector.predict(build_samples(self.case, self.config))
        rows = self.rows(self.root / "legacy-score" / "predictions.csv")
        for row, score in zip(rows, expected):
            self.assertAlmostEqual(score, float(row["score"]), places=6)
        # Adding a new policy to a legacy model is not a supported policy migration.
        settings["detection_policy"] = self.policy.to_dict()
        sidecar.write_text(json.dumps(settings), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "frozen training policy"):
            score_detection(self.case, checkpoint, self.root / "legacy-changed", explain_top_k=0)
        self.assertFalse((self.root / "legacy-changed").exists())


if __name__ == "__main__":
    unittest.main()
