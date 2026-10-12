"""Fixed-split, multiple-seed neural and tabular detection experiments.

All variants receive the same causally built samples and chronological split.
The benchmark never contacts an LLM or selects settings from test labels.
"""
from __future__ import annotations

import csv
import math
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from ..io import write_json
from ..models import CaseGraph, iso_time
from .audit import audit_detection_case
from .benchmark_summary import benchmark_summary_rows, render_benchmark_markdown, summarize_benchmark_runs
from .metrics import binary_metrics, choose_threshold, chronological_split, daily_budget_metrics
from .policy import DetectionPolicy
from .runner import (_configuration_digest, _labels, _peak_rss_mb, _source_digest,
                     _versions, _write_alerts, _write_predictions, VARIANTS as NEURAL_VARIANTS)
from .stream import BASE_FEATURE_NAMES, StreamConfig, build_samples


TABULAR_VARIANTS = ("lgbm_base", "lgbm_flat")
VARIANTS = (*NEURAL_VARIANTS, *TABULAR_VARIANTS)


def benchmark_detection(
    case: CaseGraph, output: str | Path, *, seeds: Sequence[int] = (13, 42, 87),
    variants: Sequence[str] = ("base", "flat", "event", "lgbm_base", "lgbm_flat"),
    config: StreamConfig = StreamConfig(), policy: DetectionPolicy | None = None,
    policy_provenance: dict | None = None, source_path: str | Path | None = None,
    input_row_limit: int | None = None, data_role: str = "pilot",
    epochs: int = 20, learning_rate: float = .001, batch_size: int = 64,
    hidden_dim: int = 32, class_weight: str = "balanced",
    train_fraction: float = .6, validation_fraction: float = .2,
    budget_fractions: Sequence[float] = (.001, .005, .01),
    tree_estimators: int = 100, tree_learning_rate: float = .05, tree_leaves: int = 15,
) -> dict:
    seeds, variants, budget_fractions = tuple(seeds), tuple(variants), tuple(budget_fractions)
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2147483647 for seed in seeds)):
        raise ValueError("seeds must be distinct integers in [0, 2147483647]")
    if not variants or len(set(variants)) != len(variants) or set(variants) - set(VARIANTS):
        raise ValueError(f"variants must be a distinct nonempty subset of {VARIANTS}")
    if data_role not in ("smoke", "pilot", "public_dataset_run"):
        raise ValueError("data_role must be smoke, pilot or public_dataset_run")
    if class_weight not in ("none", "balanced"):
        raise ValueError("class_weight must be none or balanced")
    if (isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1
            or isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1
            or isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim < 4
            or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("epochs/batch_size must be positive, hidden_dim >=4 and learning_rate finite positive")
    if (not budget_fractions or len(set(budget_fractions)) != len(budget_fractions)
            or any(isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= 1 for value in budget_fractions)):
        raise ValueError("budget fractions must be distinct finite values in (0, 1]")
    if input_row_limit is not None and (isinstance(input_row_limit, bool) or not isinstance(input_row_limit, int) or input_row_limit < 1):
        raise ValueError("input_row_limit must be a positive integer")
    root = Path(output)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError("benchmark output must be a new or empty directory")

    # Validate optional runtimes before publishing an in-progress experiment.
    if set(variants).intersection(NEURAL_VARIANTS):
        from .neural import NeuralDetector
    if set(variants).intersection(TABULAR_VARIANTS):
        from .tabular import LightGBMDetector, _dependencies
        _dependencies()
        for variant in set(variants).intersection(TABULAR_VARIANTS):
            LightGBMDetector(BASE_FEATURE_NAMES, variant=variant, seed=seeds[0],
                             n_estimators=tree_estimators, learning_rate=tree_learning_rate,
                             num_leaves=tree_leaves, class_weight=class_weight)

    audit = audit_detection_case(case, source_path=source_path)
    begin = time.perf_counter()
    samples = build_samples(case, config, policy=policy)
    feature_seconds = time.perf_counter() - begin
    labels = _labels(case, samples)
    splits = chronological_split(samples, train_fraction, validation_fraction)
    for split in ("train", "validation"):
        if {labels[index] for index in splits[split]} != {0, 1}:
            raise ValueError(f"{split} needs both label classes; use more chronological data")
    subsets = {name: [samples[index] for index in indices] for name, indices in splits.items()}
    split_labels = {name: [labels[index] for index in indices] for name, indices in splits.items()}
    policy_dict = policy.to_dict() if policy is not None else None
    manifest = {
        "schema_version": 1, "status": "running", "created_at": iso_time(datetime.now(timezone.utc)),
        "task": "fixed_split_multiseed_causal_transaction_detection", "data_role": data_role,
        "publication_validation": "not_established_by_this_benchmark",
        "source": str(Path(source_path).resolve()) if source_path else None,
        "source_sha256": _source_digest(source_path) if source_path else None,
        "input_row_limit": input_row_limit, "input_case_id": case.case_id,
        "observation_policy": "strictly_earlier_history_plus_current_transaction",
        "same_timestamp_policy": "simultaneous_group_no_peer_context",
        "stream_config": asdict(config), "detection_policy": policy_dict,
        "policy_provenance": policy_provenance, "policy_sha256": _configuration_digest(policy_dict),
        "feature_names": list(BASE_FEATURE_NAMES), "seeds": list(seeds), "variants": list(variants),
        "neural_config": {"epochs": epochs, "learning_rate": learning_rate, "batch_size": batch_size,
                          "hidden_dim": hidden_dim, "class_weight": class_weight},
        "tabular_config": {"n_estimators": tree_estimators, "learning_rate": tree_learning_rate,
                           "num_leaves": tree_leaves, "class_weight": class_weight,
                           "num_threads": 1, "reproduction_scope": "causal_base_and_flat_controls_not_GFP"},
        "split_fractions": {"train": train_fraction, "validation": validation_fraction},
        "budget_fractions": list(budget_fractions), "split_summary": {}, "runtime": _versions(),
        "shared_causal_feature_seconds": feature_seconds, "completed_seeds": [],
        "limitations": [
            "Seed repetitions share one chronological split and measure optimization variability, not independent test datasets.",
            "All causal samples remain in memory; per-account bounded history does not imply constant total memory.",
            "Tabular controls are not reproductions of GFP or Multi-GNN; external strong baselines remain required.",
            "Daily budget ranking is retrospective within each UTC day, not an online budget allocator.",
            "The benchmark does not conduct expensive source-deletion audits; use detect-score for selected models.",
            "public_dataset_run declares intended usage and does not independently establish data authenticity or publication readiness.",
        ],
    }
    for name, subset in subsets.items():
        manifest["split_summary"][name] = {
            "transactions": len(subset), "positives": sum(split_labels[name]),
            "prevalence": sum(split_labels[name]) / len(subset),
            "first_time": iso_time(subset[0].timestamp), "last_time": iso_time(subset[-1].timestamp),
        }
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "data_audit.json", audit)
    write_json(root / "splits.json", {name: [item.event_id for item in subset] for name, subset in subsets.items()})
    write_json(root / "benchmark_manifest.json", manifest)
    runs = []
    try:
        for seed in seeds:
            directory = root / f"seed_{seed}"
            directory.mkdir()
            metrics = {}
            for variant in variants:
                neural = variant in NEURAL_VARIANTS
                if neural:
                    detector = NeuralDetector(BASE_FEATURE_NAMES, variant=variant, hidden_dim=hidden_dim, seed=seed)
                else:
                    detector = LightGBMDetector(BASE_FEATURE_NAMES, variant=variant, seed=seed,
                                               n_estimators=tree_estimators, learning_rate=tree_learning_rate,
                                               num_leaves=tree_leaves, class_weight=class_weight)
                begin = time.perf_counter()
                losses = (detector.fit(subsets["train"], split_labels["train"], epochs=epochs,
                                       learning_rate=learning_rate, batch_size=batch_size, class_weight=class_weight)
                          if neural else detector.fit(subsets["train"], split_labels["train"]))
                fit_seconds = time.perf_counter() - begin
                detector.training_config["policy_sha256"] = _configuration_digest(policy_dict)
                detector.training_config["stream_config_sha256"] = _configuration_digest(asdict(config))
                begin = time.perf_counter()
                validation_scores = detector.predict(subsets["validation"])
                test_scores = detector.predict(subsets["test"])
                inference_seconds = time.perf_counter() - begin
                threshold = choose_threshold(split_labels["validation"], validation_scores)
                model_path = directory / f"model_{variant}{'.pt' if neural else '.txt'}"
                detector.save(model_path)
                settings = {
                    "schema_version": 1, "variant": variant, "threshold": threshold,
                    "model_sha256": _source_digest(model_path), "stream_config": asdict(config),
                    "detection_policy": policy_dict, "feature_names": list(BASE_FEATURE_NAMES),
                    "seed": seed, "case_id": case.case_id, "observation_policy": manifest["observation_policy"],
                    "training_manifest": "../benchmark_manifest.json",
                }
                if neural:
                    settings_path = model_path.with_suffix(".json")
                else:
                    settings["detector_sidecar_sha256"] = _source_digest(model_path.with_suffix(".json"))
                    settings_path = model_path.with_suffix(".settings.json")
                write_json(settings_path, settings)
                _write_predictions(directory / f"predictions_{variant}.csv", case, subsets["test"],
                                   test_scores, threshold, split_labels["test"])
                _write_alerts(directory / f"alerts_{variant}.jsonl", case, subsets["test"], test_scores, threshold)
                metrics[variant] = {
                    "validation": binary_metrics(split_labels["validation"], validation_scores, threshold),
                    "test": binary_metrics(split_labels["test"], test_scores, threshold),
                    "test_daily_budgets": [daily_budget_metrics(
                        split_labels["test"], test_scores, [item.timestamp for item in subsets["test"]],
                        [item.event_id for item in subsets["test"]], fraction) for fraction in budget_fractions],
                    "training_history": losses, "training_config": detector.training_config,
                    "timing": {"training_seconds": fit_seconds,
                               "validation_test_inference_seconds": inference_seconds,
                               "scope": "features shared across runs; model inference excludes feature construction and output writes"},
                }
                write_json(directory / "metrics.json", metrics)
            runs.append({"seed": seed, "metrics": metrics, "run_directory": str(directory.resolve())})
            manifest["completed_seeds"].append(seed)
            write_json(root / "benchmark_manifest.json", manifest)
        summary = summarize_benchmark_runs(runs)
        summary["data_role"] = data_role
        summary["source_sha256"] = manifest["source_sha256"]
        write_json(root / "summary.json", summary)
        rows = benchmark_summary_rows(summary)
        with (root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
        (root / "summary.md").write_text(
            f"数据用途：{data_role}。本表不构成数据真实性或投稿条件验证。\n\n" + render_benchmark_markdown(summary),
            encoding="utf-8")
        manifest["status"] = "completed"
        manifest["process_peak_rss_mb"] = _peak_rss_mb()
        write_json(root / "benchmark_manifest.json", manifest)
    except (Exception, KeyboardInterrupt) as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["failure_type"] = type(exc).__name__
        write_json(root / "benchmark_manifest.json", manifest)
        raise
    return {"output": str(root.resolve()), "transactions": len(samples), "seeds": list(seeds),
            "variants": list(variants), "status": manifest["status"], "data_role": data_role,
            "test_average_precision": {name: item["test"]["average_precision"]
                                       for name, item in summary["variants"].items()}}
