"""Reproducible chronological training and offline causal stream scoring."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Sequence

from ..adapters import case_from_transaction_csv
from .. import __version__
from ..io import read_json, write_json
from ..models import CaseGraph, iso_time, parse_time
from .evidence import explain_transaction
from .metrics import binary_metrics, choose_threshold, chronological_split, daily_budget_metrics
from .stream import BASE_FEATURE_NAMES, StreamConfig, TransactionSample, build_samples
from .policy import DetectionPolicy, validate_policy_for_stream


VARIANTS = ("base", "flat", "event", "no_role", "no_time", "untyped_event")


def load_detection_policy(path: str | Path) -> tuple[DetectionPolicy, dict]:
    document = read_json(path)
    if isinstance(document, dict) and "policy" in document:
        if set(document) - {"policy", "compilation"}:
            raise ValueError("detection policy document contains unsupported wrapper fields")
        policy = DetectionPolicy.from_dict(document["policy"])
        provenance = document.get("compilation", {})
        if not isinstance(provenance, dict):
            raise ValueError("policy compilation provenance must be a JSON object")
    else:
        policy = DetectionPolicy.from_dict(document)
        provenance = {"source": "provided_detection_policy", "semantic_correctness": "not_verified"}
    return policy, {"file": str(Path(path).resolve()), "file_sha256": _source_digest(path),
                    "compilation": provenance}


def _configuration_digest(value: dict | None) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_detection_case(
    *, csv_path: str | Path | None = None, case_path: str | Path | None = None,
    case_id: str = "IBM-AML", limit: int | None = None,
) -> CaseGraph:
    if (csv_path is None) == (case_path is None):
        raise ValueError("provide exactly one CSV or CaseGraph JSON input")
    if csv_path is not None:
        return case_from_transaction_csv(csv_path, case_id=case_id, limit=limit,
                                         build_relations=False)
    if limit is not None:
        raise ValueError("--limit applies to CSV rows, not CaseGraph JSON")
    case = CaseGraph.from_dict(read_json(case_path))
    case.validate()
    return case


def _labels(case: CaseGraph, samples: Sequence[TransactionSample]) -> list[int]:
    mapping = case.metadata.get("evaluation_labels", {})
    labels: list[int] = []
    for item in samples:
        label = mapping.get(item.event_id)
        if isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1):
            raise ValueError(f"transaction {item.event_id} lacks a binary transaction-level label")
        labels.append(label)
    return labels


def _versions() -> dict:
    packages = {}
    packages["hamletkg"] = __version__
    for name in ("torch", "numpy", "lightgbm", "scipy", "narwhals"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not_installed"
    # Editable installations can change without a new version number. Bind
    # the record to the installed Python sources as well as package versions.
    package_root = Path(__file__).resolve().parent.parent
    source_files = sorted(package_root.rglob("*.py"))
    fingerprint = hashlib.sha256()
    for source in source_files:
        fingerprint.update(source.relative_to(package_root).as_posix().encode("utf-8") + b"\0")
        fingerprint.update(hashlib.sha256(source.read_bytes()).digest())
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages, "code_fingerprint": {
                "sha256": fingerprint.hexdigest(), "python_source_files": len(source_files),
                "scope": "installed hamletkg Python sources; excludes tests, data, dependencies and git identity",
            }}


def _source_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_predictions(path: Path, case: CaseGraph, samples: Sequence[TransactionSample],
                       scores: Sequence[float], threshold: float, labels: Sequence[int] | None) -> None:
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["event_id", "timestamp", "source_record_id", "score",
                         "threshold_alert", "label", "candidate_events"])
        for index, (sample, score) in enumerate(zip(samples, scores)):
            writer.writerow([sample.event_id, iso_time(sample.timestamp),
                             case.atomic_events[sample.event_id].source_record_id,
                             score, int(score >= threshold),
                             labels[index] if labels is not None else "",
                             len(sample.events)])
    temporary.replace(path)


def _write_alerts(path: Path, case: CaseGraph, samples: Sequence[TransactionSample],
                  scores: Sequence[float], threshold: float) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for sample, score in zip(samples, scores):
            if score < threshold:
                continue
            transaction = case.atomic_events[sample.event_id]
            value = {
                "event_id": sample.event_id, "timestamp": iso_time(sample.timestamp),
                "score": score, "decision": "suspicious_transaction_alert",
                "source_record_id": transaction.source_record_id,
                "candidate_event_ids": [event.instance_id for event in sample.events],
                "roles": transaction.roles,
            }
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _explanations(case: CaseGraph, samples: Sequence[TransactionSample], scores: Sequence[float],
                  detector: object, config: StreamConfig, *, top_k: int, seed: int,
                  max_candidates: int, policy: DetectionPolicy | None = None) -> list[dict]:
    if top_k < 0:
        raise ValueError("explain_top_k cannot be negative")
    ranked = sorted(zip(samples, scores), key=lambda pair: (-pair[1], pair[0].event_id))
    return [explain_transaction(case, sample, detector, config, seed=seed + index,
                                max_candidates=max_candidates, policy=policy)
            for index, (sample, _) in enumerate(ranked[:top_k])]


def _peak_rss_mb() -> float | None:
    try:
        import resource
        high_water = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return high_water / (1024 ** 2 if sys.platform == "darwin" else 1024)
    except (ImportError, AttributeError):
        return None


def train_detection(
    case: CaseGraph, output: str | Path, *, config: StreamConfig = StreamConfig(),
    variants: Sequence[str] = ("event", "base", "flat"), seed: int = 42,
    epochs: int = 20, learning_rate: float = .001, batch_size: int = 64,
    class_weight: str = "balanced",
    hidden_dim: int = 32, train_fraction: float = .6, validation_fraction: float = .2,
    budget_fractions: Sequence[float] = (.001, .005, .01), explain_top_k: int = 3,
    explanation_max_candidates: int = 3, source_path: str | Path | None = None,
    load_seconds: float = 0., fixture_info: dict | None = None,
    policy: DetectionPolicy | None = None, policy_provenance: dict | None = None,
) -> dict:
    from .neural import NeuralDetector

    if not variants or len(set(variants)) != len(variants) or set(variants) - set(VARIANTS):
        raise ValueError(f"variants must be a nonempty distinct subset of {VARIANTS}")
    if class_weight not in ("none", "balanced"):
        raise ValueError("class_weight must be none or balanced")
    if epochs < 1 or batch_size < 1 or hidden_dim < 4 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("epochs/batch_size must be positive, hidden_dim >= 4 and learning_rate finite positive")
    if not budget_fractions or any(not math.isfinite(value) or not 0 < value <= 1
                                   for value in budget_fractions):
        raise ValueError("budget fractions must be finite values in (0, 1]")
    if explain_top_k < 0 or explanation_max_candidates < 1:
        raise ValueError("explain_top_k must be nonnegative and max candidates positive")
    if policy is not None:
        validate_policy_for_stream(policy, window_seconds=config.window_seconds, max_history=config.max_history)
    policy_dict = policy.to_dict() if policy is not None else None
    begin = time.perf_counter()
    samples = build_samples(case, config, policy=policy)
    feature_seconds = time.perf_counter() - begin
    labels = _labels(case, samples)
    splits = chronological_split(samples, train_fraction, validation_fraction)
    for name in ("train", "validation"):
        if {labels[index] for index in splits[name]} != {0, 1}:
            raise ValueError(f"{name} needs both label classes; use more chronological data")

    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1, "created_at": iso_time(datetime.now(timezone.utc)),
        "task": "causal_transaction_laundering_detection",
        "method_status": "research_prototype_not_validated_for_publication",
        "observation_policy": "strictly_earlier_history_plus_current_transaction",
        "same_timestamp_policy": "simultaneous_group_no_peer_context",
        "label_source": "case.metadata.evaluation_labels_only",
        "input_case_id": case.case_id, "source": str(Path(source_path).resolve()) if source_path else None,
        "source_sha256": _source_digest(source_path) if source_path else None,
        "dataset_metadata": {key: case.metadata.get(key) for key in ("dataset",)},
        "fixture_info": fixture_info, "runtime": _versions(),
        "stream_config": asdict(config), "feature_names": list(BASE_FEATURE_NAMES),
        "detection_policy": policy_dict, "policy_provenance": policy_provenance,
        "policy_sha256": _configuration_digest(policy_dict),
        "seed": seed, "variants": list(variants), "epochs": epochs,
        "learning_rate": learning_rate, "batch_size": batch_size, "hidden_dim": hidden_dim,
        "class_weight": class_weight,
        "split_fractions": {"train": train_fraction, "validation": validation_fraction},
        "budget_fractions": list(budget_fractions),
        "explanation_top_k": explain_top_k,
        "explanation_max_candidates": explanation_max_candidates,
        "split_summary": {},
        "warnings": [
            "Chronology does not guarantee injected laundering chains are split-disjoint.",
            "Daily top-B evaluation ranks a complete day retrospectively; it is not an online allocator.",
            "Scores are uncalibrated classifier outputs, not calibrated laundering probabilities.",
            "Internal ablations do not replace GFP/LightGBM and Multi-GNN external baselines.",
        ],
    }
    split_rows = {}
    for name, indices in splits.items():
        split_labels = [labels[index] for index in indices]
        manifest["split_summary"][name] = {
            "transactions": len(indices), "positives": sum(split_labels),
            "prevalence": sum(split_labels) / len(indices),
            "first_time": iso_time(samples[indices[0]].timestamp),
            "last_time": iso_time(samples[indices[-1]].timestamp),
        }
        split_rows[name] = [samples[index].event_id for index in indices]
    if fixture_info:
        manifest["warnings"].append("Toy fixture results verify execution only and are not research evidence.")
    write_json(root / "splits.json", split_rows)
    write_json(root / "manifest.json", manifest)

    training = [samples[index] for index in splits["train"]]
    train_labels = [labels[index] for index in splits["train"]]
    test_samples = [samples[index] for index in splits["test"]]
    test_labels = [labels[index] for index in splits["test"]]
    validation_labels = [labels[index] for index in splits["validation"]]
    results = {}
    for variant in variants:
        detector = NeuralDetector(BASE_FEATURE_NAMES, variant=variant,
                                  hidden_dim=hidden_dim, seed=seed)
        begin = time.perf_counter()
        losses = detector.fit(training, train_labels, epochs=epochs,
                              learning_rate=learning_rate, batch_size=batch_size,
                              class_weight=class_weight)
        detector.training_config["policy_sha256"] = _configuration_digest(policy_dict)
        detector.training_config["stream_config_sha256"] = _configuration_digest(asdict(config))
        fit_seconds = time.perf_counter() - begin
        begin = time.perf_counter()
        all_scores = detector.predict(samples)
        inference_seconds = time.perf_counter() - begin
        validation_scores = [all_scores[index] for index in splits["validation"]]
        test_scores = [all_scores[index] for index in splits["test"]]
        threshold = choose_threshold(validation_labels, validation_scores)
        checkpoint = root / f"model_{variant}.pt"
        detector.save(checkpoint)
        write_json(checkpoint.with_suffix(".json"), {
            "schema_version": 1, "variant": variant, "threshold": threshold,
            "model_sha256": _source_digest(checkpoint),
            "stream_config": asdict(config), "feature_names": list(BASE_FEATURE_NAMES),
            "detection_policy": policy_dict,
            "case_id": case.case_id, "seed": seed, "training_manifest": "manifest.json",
            "observation_policy": manifest["observation_policy"],
        })
        _write_predictions(root / f"predictions_{variant}.csv", case, test_samples,
                           test_scores, threshold, test_labels)
        _write_alerts(root / f"alerts_{variant}.jsonl", case, test_samples, test_scores, threshold)
        latencies = []
        for sample in test_samples[:min(64, len(test_samples))]:
            begin = time.perf_counter()
            detector.predict([sample])
            latencies.append((time.perf_counter() - begin) * 1000)
        p95 = sorted(latencies)[math.ceil(.95 * len(latencies)) - 1]
        replay_seconds = load_seconds + feature_seconds + inference_seconds
        results[variant] = {
            "validation": binary_metrics(validation_labels, validation_scores, threshold),
            "test": binary_metrics(test_labels, test_scores, threshold),
            "test_daily_budgets": [daily_budget_metrics(
                test_labels, test_scores, [item.timestamp for item in test_samples],
                [item.event_id for item in test_samples], fraction)
                for fraction in budget_fractions],
            "training_history": losses,
            "training_config": detector.training_config,
            "timing": {
                "load_seconds": load_seconds, "causal_feature_seconds": feature_seconds,
                "training_seconds": fit_seconds, "all_transaction_inference_seconds": inference_seconds,
                "offline_replay_transactions_per_second": len(samples) / max(replay_seconds, 1e-12),
                "model_single_transaction_p95_ms": p95, "latency_samples": len(latencies),
                "scope": "replay throughput includes input load, features and inference; excludes training, output writes and evidence audits; p95 is model-only CPU",
            },
        }
        # Expensive source-deletion audits are run for the event variant, or the
        # first requested variant if a run intentionally omits it.
        explained_variant = "event" if "event" in variants else variants[0]
        if variant == explained_variant:
            begin = time.perf_counter()
            audits = _explanations(case, test_samples, test_scores, detector, config,
                                   top_k=explain_top_k, seed=seed,
                                   max_candidates=explanation_max_candidates, policy=policy)
            write_json(root / "evidence.json", {
                "schema_version": 1, "variant": variant, "transactions": audits,
                "model_sha256": _source_digest(checkpoint),
                "policy_sha256": _configuration_digest(policy_dict),
                "stream_config_sha256": _configuration_digest(asdict(config)),
                "observation_policy": manifest["observation_policy"],
            })
            results[variant]["timing"]["offline_evidence_audit_seconds"] = time.perf_counter() - begin
        write_json(root / "metrics.json", results)

    manifest["process_peak_rss_mb"] = _peak_rss_mb()
    manifest["memory_scope"] = "whole-process high-water mark, including all trained variants and PyTorch"
    write_json(root / "manifest.json", manifest)
    return {"output": str(root.resolve()), "transactions": len(samples),
            "split_summary": manifest["split_summary"],
            "test_metrics": {name: result["test"] for name, result in results.items()},
            "fixture_info": fixture_info}


def score_detection(
    case: CaseGraph, model_path: str | Path, output: str | Path, *,
    after: str | None = None, threshold: float | None = None,
    explain_top_k: int = 3, explanation_max_candidates: int = 3,
) -> dict:
    if explain_top_k < 0 or explanation_max_candidates < 1:
        raise ValueError("explain_top_k must be nonnegative and max candidates positive")
    checkpoint = Path(model_path)
    tabular = checkpoint.suffix == ".txt"
    settings = read_json(checkpoint.with_suffix(".settings.json" if tabular else ".json"))
    if settings.get("schema_version") != 1 or settings.get("feature_names") != list(BASE_FEATURE_NAMES):
        raise ValueError("checkpoint settings have an unsupported schema or feature order")
    if settings.get("model_sha256") != _source_digest(checkpoint):
        raise ValueError("checkpoint and JSON settings do not belong to the same saved model")
    config = StreamConfig(**settings["stream_config"])
    policy_dict = settings.get("detection_policy")
    policy = DetectionPolicy.from_dict(policy_dict) if policy_dict is not None else None
    if policy is not None:
        validate_policy_for_stream(policy, window_seconds=config.window_seconds, max_history=config.max_history)
    selected_threshold = float(settings["threshold"] if threshold is None else threshold)
    if not math.isfinite(selected_threshold) or selected_threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    if tabular:
        from .tabular import LightGBMDetector
        if settings.get("detector_sidecar_sha256") != _source_digest(checkpoint.with_suffix(".json")):
            raise ValueError("LightGBM detector settings differ from the frozen saved configuration")
        detector = LightGBMDetector.load(checkpoint)
    else:
        from .neural import NeuralDetector
        detector = NeuralDetector.load(checkpoint)
    if detector.variant != settings["variant"]:
        raise ValueError("checkpoint and settings variants disagree")
    if list(detector.feature_names) != settings["feature_names"]:
        raise ValueError("checkpoint and settings feature orders disagree")
    trained_policy_digest = detector.training_config.get("policy_sha256")
    trained_stream_digest = detector.training_config.get("stream_config_sha256")
    if (trained_policy_digest is not None or policy is not None) and trained_policy_digest != _configuration_digest(policy_dict):
        raise ValueError("detection policy differs from the frozen training policy; retrain the model")
    if trained_stream_digest is not None and trained_stream_digest != _configuration_digest(asdict(config)):
        raise ValueError("stream settings differ from the frozen training settings; retrain the model")
    samples = build_samples(case, config, policy=policy)
    if after is not None:
        cutoff = parse_time(after)
        samples = [item for item in samples if item.timestamp >= cutoff]
    if not samples:
        raise ValueError("no transactions remain for scoring")
    scores = detector.predict(samples)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _write_predictions(root / "predictions.csv", case, samples, scores, selected_threshold, None)
    _write_alerts(root / "alerts.jsonl", case, samples, scores, selected_threshold)
    audits = _explanations(case, samples, scores, detector, config, top_k=explain_top_k,
                           seed=settings["seed"], max_candidates=explanation_max_candidates, policy=policy)
    write_json(root / "evidence.json", {
        "schema_version": 1, "variant": settings["variant"], "transactions": audits,
        "model_sha256": _source_digest(checkpoint),
        "policy_sha256": _configuration_digest(policy_dict),
        "stream_config_sha256": _configuration_digest(asdict(config)),
        "observation_policy": "strictly_earlier_history_plus_current_transaction",
    })
    summary = {"output": str(root.resolve()), "transactions": len(samples),
               "threshold_alerts": sum(score >= selected_threshold for score in scores),
               "threshold": selected_threshold, "model": str(checkpoint.resolve()),
               "after": after, "history_policy": "all strictly earlier input rows are context; input starts with empty state"}
    write_json(root / "summary.json", summary)
    return summary
