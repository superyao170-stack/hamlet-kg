"""Train against immutable causal caches; evaluate scores on disk."""
from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Sequence

from ..io import read_json, write_json
from .benchmark import VARIANTS, NEURAL_VARIANTS
from .benchmark_summary import summarize_benchmark_runs, benchmark_summary_rows, render_benchmark_markdown
from .cache import CachedSamples
from .dataset import PreparedDataset
from .disk_metrics import DiskScores
from .runner import _versions, _source_digest, _peak_rss_mb, _configuration_digest
from .stream import BASE_FEATURE_NAMES


def labeled_batches(cache: CachedSamples, split: str, batch_size: int):
    labels = iter(cache.iter_labels(split))
    missing = object()
    for batch in cache.iter_batches(split, batch_size=batch_size):
        targets = [next(labels, missing) for _ in batch]
        if any(value is missing for value in targets):
            raise ValueError("cache labels and samples are not aligned")
        yield batch, targets
    if next(labels, missing) is not missing:
        raise ValueError("cache has labels without corresponding samples")


def _label_counts(cache):
    result = {}
    for split in ("train", "validation", "test"):
        count = positive = unknown = 0
        for label in cache.iter_labels(split):
            count += 1
            unknown += label is None
            positive += label == 1
        if count != cache.split_counts()[split]:
            raise ValueError("cache label counts and split disagree")
        result[split] = {"transactions": count, "positive": positive,
                         "negative": count - positive - unknown, "unlabeled": unknown}
    return result


def _dataset(cache, directory):
    if directory is None:
        return None
    selection = cache.manifest["dataset"]["selection"] or {}
    dataset = PreparedDataset(directory, after=selection.get("after"), before=selection.get("before"))
    if dataset.manifest["database"]["sha256"] != cache.manifest["dataset"]["database_sha256"]:
        raise ValueError("prepared dataset and sample cache do not belong together")
    if _configuration_digest(dataset.manifest) != cache.manifest["dataset"]["manifest_sha256"]:
        raise ValueError("prepared source manifest differs from the frozen cache source")
    if dataset.selection != selection:
        raise ValueError("prepared dataset selection differs from the frozen cache selection")
    dataset.validate_integrity()
    return dataset


def _score_split(detector, cache, split, store, variant, batch_size, dataset=None):
    for batch, labels in labeled_batches(cache, split, batch_size):
        scores = detector.predict(batch)
        sources = None
        if dataset is not None:
            records = dataset.fetch_transactions([sample.event_id for sample in batch])
            if len(records) != len(batch):
                raise ValueError("prepared source is missing a cached target")
            sources = [records[sample.event_id].source_record_id for sample in batch]
        store.add_batch(variant, split, batch, scores, labels, sources)
    store.commit()


def _tree_matrix(cache, root, batch_size):
    import numpy as np
    from .tabular import flat_event_summary
    count = cache.split_counts()["train"]
    path, label_path = root / "tree_training_features.npy", root / "tree_training_labels.npy"
    matrix = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=(count, 40))
    labels = np.lib.format.open_memmap(label_path, mode="w+", dtype=np.int8, shape=(count,))
    offset = 0
    for batch, targets in labeled_batches(cache, "train", batch_size):
        matrix[offset:offset + len(batch)] = [(*sample.features, *flat_event_summary(sample.events)) for sample in batch]
        labels[offset:offset + len(batch)] = targets
        offset += len(batch)
    if offset != count:
        raise ValueError("training matrix size differs from the cache")
    matrix.flush()
    labels.flush()
    del matrix, labels
    return path, label_path


def benchmark_cached(cache_path: str | Path, output: str | Path, *, dataset_path: str | Path | None = None,
                     seeds: Sequence[int] = (13, 42, 87),
                     variants: Sequence[str] = ("base", "flat", "event", "lgbm_base", "lgbm_flat"),
                     epochs: int = 20, learning_rate: float = .001, batch_size: int = 64,
                     hidden_dim: int = 32, class_weight: str = "balanced", shuffle_buffer_batches: int = 0,
                     neural_scaling: str = "standard",
                     tree_estimators: int = 100, tree_learning_rate: float = .05, tree_leaves: int = 15,
                     budget_fractions: Sequence[float] = (.001, .005, .01), data_role: str = "pilot") -> dict:
    seeds, variants, budget_fractions = tuple(seeds), tuple(variants), tuple(budget_fractions)
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or not 0 <= seed <= 2147483647 for seed in seeds):
        raise ValueError("seeds must be distinct nonnegative 32-bit integers")
    if not variants or len(set(variants)) != len(variants) or set(variants) - set(VARIANTS):
        raise ValueError("unsupported or duplicate model variants")
    if (type(epochs) is not int or epochs < 1 or type(batch_size) is not int or batch_size < 1
            or type(hidden_dim) is not int or hidden_dim < 4 or type(shuffle_buffer_batches) is not int or shuffle_buffer_batches < 0
            or isinstance(learning_rate, bool) or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("invalid neural training or batch parameters")
    if class_weight not in ("none", "balanced") or data_role not in ("smoke", "pilot", "public_dataset_run"):
        raise ValueError("unsupported class_weight or data_role")
    if neural_scaling not in ("standard", "fixed_periodic"):
        raise ValueError("unsupported neural_scaling")
    if not budget_fractions or len(set(budget_fractions)) != len(budget_fractions) or any(
            isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= 1 for value in budget_fractions):
        raise ValueError("budget fractions must be distinct values in (0,1]")
    root = Path(output)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError("benchmark output must be new or empty")
    cache = CachedSamples(cache_path)
    cache.validate_integrity()
    counts = _label_counts(cache)
    if any(item["unlabeled"] for item in counts.values()):
        raise ValueError("supervised benchmark requires labels for every candidate")
    for split in ("train", "validation"):
        if not counts[split]["positive"] or not counts[split]["negative"]:
            raise ValueError(f"{split} needs both label classes")
    dataset = _dataset(cache, dataset_path)
    neural_variants = set(variants).intersection(NEURAL_VARIANTS)
    if neural_variants:
        from .neural import NeuralDetector
    if set(variants) - set(NEURAL_VARIANTS):
        from .tabular import LightGBMDetector, _dependencies
        _dependencies()
        LightGBMDetector(BASE_FEATURE_NAMES, seed=seeds[0], n_estimators=tree_estimators,
                         learning_rate=tree_learning_rate, num_leaves=tree_leaves, class_weight=class_weight)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "status": "running", "task": "cached_causal_transaction_detection",
                "cache": str(Path(cache_path).resolve()), "cache_sha256": cache.manifest["database_sha256"],
                "source": cache.manifest["dataset"], "stream_config": cache.manifest["stream_config"],
                "detection_policy": cache.manifest["policy"], "split": cache.manifest["split"],
                "policy_provenance": cache.manifest.get("policy_provenance"),
                "label_counts": counts, "seeds": list(seeds), "variants": list(variants),
                "data_role": data_role, "publication_validation": "not_established_by_this_run",
                "runtime": _versions(), "completed_seeds": [],
                "neural_config": {"epochs": epochs, "learning_rate": learning_rate, "batch_size": batch_size,
                                  "hidden_dim": hidden_dim, "class_weight": class_weight,
                                  "shuffle_buffer_batches": shuffle_buffer_batches,
                                  "scaling": neural_scaling},
                "tree_config": {"n_estimators": tree_estimators, "learning_rate": tree_learning_rate,
                                "num_leaves": tree_leaves, "class_weight": class_weight},
                "budget_fractions": list(budget_fractions),
                "limitations": ["Seed repetitions share one frozen chronological split; no significance guarantee.",
                                "Neural training uses sequential or bounded batch shuffle, not global sample permutation.",
                                "LightGBM still allocates its native dataset in RAM despite disk-backed input matrices.",
                                "Daily budgets are retrospective UTC-day rankings; selected ID lists are capped, counts remain exact.",
                                "This command does not perform source-deletion audits or produce LLM reports."]}
    write_json(root / "benchmark_manifest.json", manifest)
    runs = []
    try:
        matrices = _tree_matrix(cache, root, batch_size) if set(variants) - set(NEURAL_VARIANTS) else None
        for seed in seeds:
            directory = root / f"seed_{seed}"
            directory.mkdir()
            metrics = {}
            with DiskScores(directory / "scores.sqlite", create=True) as store:
                for variant in variants:
                    neural = variant in NEURAL_VARIANTS
                    begin = time.perf_counter()
                    if neural:
                        detector = NeuralDetector(BASE_FEATURE_NAMES, variant=variant, seed=seed, hidden_dim=hidden_dim,
                                                  scaling=neural_scaling)
                        history = detector.fit_batches(lambda: labeled_batches(cache, "train", batch_size), epochs=epochs,
                                                       learning_rate=learning_rate, batch_size=batch_size,
                                                       class_weight=class_weight, shuffle_buffer_batches=shuffle_buffer_batches)
                    else:
                        import numpy as np
                        detector = LightGBMDetector(BASE_FEATURE_NAMES, variant=variant, seed=seed, n_estimators=tree_estimators,
                                                   learning_rate=tree_learning_rate, num_leaves=tree_leaves, class_weight=class_weight)
                        matrix, labels = (np.load(path, mmap_mode="r", allow_pickle=False) for path in matrices)
                        history = detector.fit_matrix(matrix[:, :24] if variant == "lgbm_base" else matrix, labels,
                                                      working_directory=root)
                        del matrix, labels
                    fit_seconds = time.perf_counter() - begin
                    detector.training_config.update({"policy_sha256": cache.manifest["policy_sha256"],
                                                     "stream_config_sha256": cache.manifest["stream_config_sha256"],
                                                     "training_cache_sha256": cache.manifest["database_sha256"]})
                    begin = time.perf_counter()
                    for split in ("validation", "test"):
                        _score_split(detector, cache, split, store, variant, batch_size, dataset)
                    inference_seconds = time.perf_counter() - begin
                    threshold = store.choose_threshold(variant, "validation")
                    detector.training_config["validation_threshold"] = threshold
                    model = directory / f"model_{variant}{'.pt' if neural else '.txt'}"
                    detector.save(model)
                    settings = {"schema_version": 1, "variant": variant, "threshold": threshold,
                                "model_sha256": _source_digest(model), "stream_config": cache.manifest["stream_config"],
                                "detection_policy": cache.manifest["policy"], "feature_names": list(BASE_FEATURE_NAMES),
                                "seed": seed, "training_cache_sha256": cache.manifest["database_sha256"],
                                "observation_policy": "strictly_earlier_history_plus_current_transaction"}
                    if not neural:
                        settings["detector_sidecar_sha256"] = _source_digest(model.with_suffix(".json"))
                    else:
                        settings["neural_scaling"] = neural_scaling
                    write_json(model.with_suffix(".json" if neural else ".settings.json"), settings)
                    metrics[variant] = {"validation": store.binary_metrics(variant, "validation", threshold),
                                        "test": store.binary_metrics(variant, "test", threshold),
                                        "test_daily_budgets": [store.daily_budget_metrics(variant, "test", fraction) for fraction in budget_fractions],
                                        "training_history": history, "training_config": detector.training_config,
                                        "timing": {"training_seconds": fit_seconds, "cached_validation_test_score_seconds": inference_seconds,
                                                   "scope": "score timing includes cache decode, model, source lookup and score database writes; excludes feature construction and metrics"}}
                    store.export(variant, "test", threshold, directory)
                    write_json(directory / "metrics.json", metrics)
            runs.append({"seed": seed, "metrics": metrics, "run_directory": str(directory.resolve())})
            manifest["completed_seeds"].append(seed)
            write_json(root / "benchmark_manifest.json", manifest)
        summary = summarize_benchmark_runs(runs)
        summary.update({"data_role": data_role, "cache_sha256": cache.manifest["database_sha256"], "source": manifest["source"]})
        write_json(root / "summary.json", summary)
        rows = benchmark_summary_rows(summary)
        with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (root / "summary.md").write_text(f"数据用途：{data_role}；固定缓存与时间划分。\n\n" + render_benchmark_markdown(summary), encoding="utf-8")
        manifest["status"] = "completed"
        manifest["process_peak_rss_mb"] = _peak_rss_mb()
        write_json(root / "benchmark_manifest.json", manifest)
    except (Exception, KeyboardInterrupt) as error:
        manifest.update({"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "failure_type": type(error).__name__})
        write_json(root / "benchmark_manifest.json", manifest)
        raise
    return {"output": str(root.resolve()), "transactions": len(cache), "status": manifest["status"],
            "label_counts": counts, "test_average_precision": {variant: values["test"]["average_precision"] for variant, values in summary["variants"].items()}}


def score_cached(cache_path, model_path, output, *, split="test", dataset_path=None, batch_size=64):
    if split not in ("train", "validation", "test", "all") or type(batch_size) is not int or batch_size < 1:
        raise ValueError("invalid split or batch_size")
    root, model = Path(output), Path(model_path)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError("score output must be new or empty")
    cache = CachedSamples(cache_path)
    cache.validate_integrity()
    dataset = _dataset(cache, dataset_path)
    neural = model.suffix == ".pt"
    if model.suffix not in (".pt", ".txt"):
        raise ValueError("model must be .pt or native LightGBM .txt")
    settings = read_json(model.with_suffix(".json" if neural else ".settings.json"))
    if not isinstance(settings, dict) or settings.get("schema_version") != 1 or settings.get("model_sha256") != _source_digest(model):
        raise ValueError("model and settings do not belong together")
    if settings.get("feature_names") != cache.manifest["feature_names"]:
        raise ValueError("model feature order differs from sample cache")
    if settings.get("stream_config") != cache.manifest["stream_config"] or settings.get("detection_policy") != cache.manifest["policy"]:
        raise ValueError("sample cache input definition differs from frozen model settings")
    if neural:
        from .neural import NeuralDetector
        detector = NeuralDetector.load(model)
        if settings.get("neural_scaling", "standard") != detector.scaling:
            raise ValueError("neural scaling differs from the frozen model configuration")
    else:
        from .tabular import LightGBMDetector
        if settings.get("detector_sidecar_sha256") != _source_digest(model.with_suffix(".json")):
            raise ValueError("tree detector configuration was modified")
        detector = LightGBMDetector.load(model)
    if (detector.variant != settings.get("variant") or list(detector.feature_names) != settings["feature_names"]
            or detector.training_config.get("policy_sha256") != cache.manifest["policy_sha256"]
            or detector.training_config.get("stream_config_sha256") != cache.manifest["stream_config_sha256"]):
        raise ValueError("detector and frozen sample definition disagree")
    threshold = settings["threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold < 0:
        raise ValueError("saved threshold is invalid")
    if detector.training_config.get("validation_threshold") != threshold:
        raise ValueError("threshold differs from the frozen training validation threshold; legacy models use detect-score")
    if settings.get("training_cache_sha256") != detector.training_config.get("training_cache_sha256"):
        raise ValueError("saved training cache provenance differs from detector configuration")
    root.mkdir(parents=True, exist_ok=True)
    summary = {"status": "running", "threshold": threshold, "model_sha256": _source_digest(model),
               "cache_sha256": cache.manifest["database_sha256"], "training_cache_sha256": settings.get("training_cache_sha256"),
               "split": split, "threshold_reselected_during_scoring": False,
               "threshold_selection": "training_cache_validation", "evidence_status": "not_audited"}
    write_json(root / "summary.json", summary)
    try:
        with DiskScores(root / "scores.sqlite", create=True) as store:
            _score_split(detector, cache, split, store, detector.variant, batch_size, dataset)
            store.export(detector.variant, split, threshold, root)
            counts = store.counts(detector.variant, split)
            summary.update({"transactions": counts["total"], "evaluation_label_counts": counts,
                            "metrics": store.binary_metrics(detector.variant, split, threshold) if not counts["unlabeled"] else None,
                            "status": "completed"})
        write_json(root / "summary.json", summary)
    except (Exception, KeyboardInterrupt) as error:
        summary.update({"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "failure_type": type(error).__name__})
        write_json(root / "summary.json", summary)
        raise
    return summary
