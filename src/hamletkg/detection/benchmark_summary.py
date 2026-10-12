"""Descriptive summaries of repeated seeds on one fixed detection split.

This module has no optional dependencies. The original metrics and independently
selected thresholds remain available in ``runs``; aggregate thresholds must not
be used to score a model. Paired deltas use only seeds present in both variants.
"""
from __future__ import annotations

import copy
import math
import statistics
from collections.abc import Mapping, Sequence


CLASSIFICATION_METRICS = (
    "average_precision", "precision", "recall", "f1",
    "false_positives_per_1000",
)
BUDGET_METRICS = ("recall", "precision", "review_alerts", "true_positive")
BASELINES = ("base", "flat", "lgbm_base", "lgbm_flat")
INTERPRETATION_NOTES = (
    "All seeds use the same fixed chronological split. Seed variation measures "
    "optimization randomness, not independent test datasets.",
    "Standard deviations are sample standard deviations (ddof=1); fewer than "
    "two valid values have no standard deviation. Missing values remain null.",
    "Each seed keeps its own validation-selected threshold. No averaged "
    "threshold is substituted for those thresholds.",
    "Paired deltas are event minus baseline on common seeds with both values "
    "available; these descriptive summaries do not establish significance.",
)


def _number(value: object, location: str) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a finite number or None")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{location} must be a finite number or None")
    return value


def _stats(values: list[dict]) -> dict:
    valid = [item["value"] for item in values if item["value"] is not None]
    return {
        "mean": statistics.mean(valid) if valid else None,
        "std": statistics.stdev(valid) if len(valid) > 1 else None,
        "n": len(valid),
        "n_missing": len(values) - len(valid),
        "seeds": [item["seed"] for item in values if item["value"] is not None],
        "values": values,
    }


def _split_metrics(metrics: Mapping, split: str, location: str) -> Mapping:
    values = metrics.get(split)
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise ValueError(f"{location}.{split} must be a mapping or None")
    return values


def summarize_benchmark_runs(runs: Sequence[dict]) -> dict:
    """Summarize JSON-compatible runner metrics without imputing missing values.

    Each run has a unique integer ``seed`` and a ``metrics`` mapping from variant
    names to runner ``metrics.json`` entries. Missing variants, metric keys or
    explicit None values contribute missing observations, never zeroes. Budget
    lists, when supplied, must contain the same distinct fraction set across all
    runs and variants; list order does not matter. An omitted/None budget list
    contributes missing observations for the known fractions.
    """
    if not isinstance(runs, Sequence) or isinstance(runs, (str, bytes)) or not runs:
        raise ValueError("runs must be a nonempty sequence")
    checked = []
    seen_seeds: set[int] = set()
    variants: set[str] = set()
    budgets: dict[tuple[int, str], dict[float, Mapping]] = {}
    fractions: set[float] | None = None
    for run in runs:
        if not isinstance(run, Mapping):
            raise ValueError("each run must be a mapping")
        seed = run.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("each run seed must be an integer")
        if seed in seen_seeds:
            raise ValueError(f"duplicate seed: {seed}")
        seen_seeds.add(seed)
        metrics = run.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError(f"seed {seed} metrics must be a variant mapping")
        if run.get("run_directory") is not None and not isinstance(run["run_directory"], str):
            raise ValueError(f"seed {seed} run_directory must be a string")
        seen_variants: set[str] = set()
        for variant, data in metrics.items():
            if not isinstance(variant, str) or not variant or variant.strip() != variant:
                raise ValueError(f"seed {seed} variant names must be nonempty strings without outer whitespace")
            if variant in seen_variants:
                raise ValueError(f"seed {seed} duplicate variant: {variant}")
            seen_variants.add(variant)
            variants.add(variant)
            if not isinstance(data, Mapping):
                raise ValueError(f"seed {seed} variant {variant} metrics must be a mapping")
            location = f"seed {seed} variant {variant}"
            for split in ("validation", "test"):
                values = _split_metrics(data, split, location)
                for name in (*CLASSIFICATION_METRICS, "threshold"):
                    _number(values.get(name), f"{location}.{split}.{name}")
            entries = data.get("test_daily_budgets")
            by_fraction: dict[float, Mapping] = {}
            if entries is not None:
                if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
                    raise ValueError(f"{location}.test_daily_budgets must be a sequence or None")
                for entry in entries:
                    if not isinstance(entry, Mapping):
                        raise ValueError(f"{location} budget entries must be mappings")
                    fraction = _number(entry.get("budget_fraction"), f"{location}.budget_fraction")
                    if fraction is None or not 0 <= fraction <= 1:
                        raise ValueError(f"{location} budget_fraction must be between 0 and 1")
                    fraction = float(fraction)
                    if fraction in by_fraction:
                        raise ValueError(f"{location} duplicate budget_fraction: {fraction}")
                    for name in BUDGET_METRICS:
                        _number(entry.get(name), f"{location}.budget[{fraction}].{name}")
                    by_fraction[fraction] = entry
                current = set(by_fraction)
                if fractions is None:
                    fractions = current
                elif fractions != current:
                    raise ValueError("test_daily_budgets budget fractions must be consistent across runs and variants")
            budgets[seed, variant] = by_fraction
        checked.append(copy.deepcopy(dict(run)))
    if not variants:
        raise ValueError("runs must contain at least one variant")
    checked.sort(key=lambda item: item["seed"])
    seeds = [run["seed"] for run in checked]
    ordered_fractions = sorted(fractions or ())

    def classification_values(variant: str, split: str, metric: str) -> list[dict]:
        return [
            {"seed": run["seed"], "value": _split_metrics(
                run["metrics"].get(variant, {}), split, "metrics").get(metric)}
            for run in checked
        ]

    def budget_values(variant: str, fraction: float, metric: str) -> list[dict]:
        return [{"seed": seed, "value": budgets.get((seed, variant), {}).get(fraction, {}).get(metric)}
                for seed in seeds]

    summaries = {}
    for variant in sorted(variants):
        summaries[variant] = {
            split: {name: _stats(classification_values(variant, split, name))
                    for name in CLASSIFICATION_METRICS}
            for split in ("validation", "test")
        }
        summaries[variant]["thresholds"] = {
            split: classification_values(variant, split, "threshold")
            for split in ("validation", "test")
        }
        summaries[variant]["test_daily_budgets"] = [
            {"budget_fraction": fraction,
             **{name: _stats(budget_values(variant, fraction, name)) for name in BUDGET_METRICS}}
            for fraction in ordered_fractions
        ]

    paired = {}
    if "event" in variants:
        for baseline in BASELINES:
            if baseline not in variants:
                continue
            common = [run["seed"] for run in checked
                      if "event" in run["metrics"] and baseline in run["metrics"]]

            def delta(event_values: list[dict], baseline_values: list[dict]) -> dict:
                right = {item["seed"]: item["value"] for item in baseline_values}
                return _stats([
                    {"seed": item["seed"], "value": (
                        item["value"] - right[item["seed"]]
                        if item["value"] is not None and right[item["seed"]] is not None else None)}
                    for item in event_values if item["seed"] in common
                ])

            paired[baseline] = {
                "common_seeds": common,
                **{f"{split}_average_precision": delta(
                    classification_values("event", split, "average_precision"),
                    classification_values(baseline, split, "average_precision"))
                   for split in ("validation", "test")},
                "test_daily_budget_recall": [
                    {"budget_fraction": fraction, "stats": delta(
                        budget_values("event", fraction, "recall"),
                        budget_values(baseline, fraction, "recall"))}
                    for fraction in ordered_fractions
                ],
            }
    return {
        "schema_version": 1, "seeds": seeds, "seed_count": len(seeds),
        "interpretation_notes": list(INTERPRETATION_NOTES),
        "runs": checked, "variants": summaries, "paired_deltas": paired,
    }


def benchmark_summary_rows(summary: dict) -> list[dict]:
    """Return stable long-form rows for ``csv.DictWriter`` (None stays empty)."""
    rows = []

    def add(kind: str, variant: str, split: str, metric: str, stats: dict,
            fraction: float | None = None, seed: int | None = None,
            value: float | int | None = None) -> None:
        rows.append({
            "kind": kind, "variant": variant, "split": split, "metric": metric,
            "budget_fraction": fraction, "seed": seed, "value": value,
            "mean": stats.get("mean"), "std": stats.get("std"),
            "n": stats.get("n"), "n_missing": stats.get("n_missing"),
        })

    for variant, data in summary["variants"].items():
        for split in ("validation", "test"):
            for metric in CLASSIFICATION_METRICS:
                add("aggregate", variant, split, metric, data[split][metric])
            for item in data["thresholds"][split]:
                add("threshold", variant, split, "threshold", {},
                    seed=item["seed"], value=item["value"])
        for budget in data["test_daily_budgets"]:
            for metric in BUDGET_METRICS:
                add("aggregate", variant, "test_daily_budget", metric, budget[metric],
                    fraction=budget["budget_fraction"])
    for baseline, data in summary["paired_deltas"].items():
        for split in ("validation", "test"):
            add("paired_delta", f"event - {baseline}", split, "average_precision",
                data[f"{split}_average_precision"])
        for budget in data["test_daily_budget_recall"]:
            add("paired_delta", f"event - {baseline}", "test_daily_budget", "recall",
                budget["stats"], fraction=budget["budget_fraction"])
    return rows


def render_benchmark_markdown(summary: dict) -> str:
    """Render the descriptive summary and each seed's own thresholds."""
    def display(value: object) -> str:
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["# Detection benchmark", "",
             "Seeds: " + ", ".join(str(seed) for seed in summary["seeds"]), ""]
    lines.extend(f"- {note}" for note in summary["interpretation_notes"])
    lines.extend(["", "| Kind | Variant | Split | Metric | Budget fraction | Mean | Sample std | Valid n | Missing n |",
                  "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |"])
    rows = benchmark_summary_rows(summary)
    for row in rows:
        if row["kind"] != "threshold":
            lines.append("| " + " | ".join(display(row[name]) for name in (
                "kind", "variant", "split", "metric", "budget_fraction", "mean", "std", "n", "n_missing")) + " |")
    lines.extend(["", "## Per-seed thresholds", "",
                  "| Variant | Split | Seed | Threshold |", "| --- | --- | ---: | ---: |"])
    for row in rows:
        if row["kind"] == "threshold":
            lines.append("| " + " | ".join(display(row[name]) for name in (
                "variant", "split", "seed", "value")) + " |")
    return "\n".join(lines) + "\n"
