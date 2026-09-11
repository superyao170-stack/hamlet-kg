"""Label-safe AMLSim profiling and graph-pattern baselines.

Every detector receives only transaction records.  Node labels are loaded in a
separate evaluation path after predictions have been produced.
"""

from __future__ import annotations

import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


Transaction = tuple[int, int, float, int]


def _read_transactions(path: Path) -> list[Transaction]:
    transactions: list[Transaction] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            try:
                transactions.append(
                    (
                        int(row["sourceNodeId"]),
                        int(row["targetNodeId"]),
                        float(row["value"]),
                        int(float(row["time"])),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid transaction at {path}:{row_number}"
                ) from exc
    return transactions


def _read_labels(path: Path) -> tuple[set[int], set[int], dict[int, int]]:
    all_nodes: set[int] = set()
    positive_nodes: set[int] = set()
    fraud_steps: dict[int, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            try:
                node = int(row["nodeid"])
                label = int(float(row["isFraud"]))
                fraud_step = int(float(row["fraudStep"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid node at {path}:{row_number}") from exc
            if node in all_nodes:
                raise ValueError(f"duplicate node id {node} at {path}:{row_number}")
            all_nodes.add(node)
            if label:
                positive_nodes.add(node)
            fraud_steps[node] = fraud_step
    return all_nodes, positive_nodes, fraud_steps


def _percentile(values: Iterable[int], proportion: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * proportion
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _adaptive_amount_threshold(transactions: list[Transaction]) -> float:
    """Find the largest adjacent multiplicative gap without using labels."""

    values = sorted({amount for _, _, amount, _ in transactions if amount > 0})
    if len(values) < 2:
        return values[0] if values else 0.0
    left, right = max(
        zip(values, values[1:]),
        key=lambda pair: pair[1] / max(pair[0], 1e-12),
    )
    return math.sqrt(left * right)


def _strongly_connected_components(
    edges: Iterable[tuple[int, int]],
) -> list[set[int]]:
    """Kosaraju SCC decomposition implemented without external dependencies."""

    adjacency: dict[int, set[int]] = defaultdict(set)
    reverse: dict[int, set[int]] = defaultdict(set)
    nodes: set[int] = set()
    for source, target in edges:
        adjacency[source].add(target)
        reverse[target].add(source)
        nodes.update((source, target))

    visited: set[int] = set()
    order: list[int] = []
    for start in sorted(nodes):
        if start in visited:
            continue
        stack: list[tuple[int, bool]] = [(start, False)]
        while stack:
            node, postorder = stack.pop()
            if postorder:
                order.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            for neighbor in sorted(adjacency[node], reverse=True):
                if neighbor not in visited:
                    stack.append((neighbor, False))

    components: list[set[int]] = []
    visited.clear()
    for start in reversed(order):
        if start in visited:
            continue
        component = {start}
        visited.add(start)
        stack = [start]
        while stack:
            node = stack.pop()
            for neighbor in reverse[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def _classification_metrics(
    predicted: set[int],
    truth: set[int],
    universe: set[int],
) -> dict[str, float | int]:
    true_positive = len(predicted & truth)
    false_positive = len(predicted - truth)
    false_negative = len(truth - predicted)
    true_negative = len(universe - predicted - truth)
    precision = true_positive / (true_positive + false_positive) if predicted else 0.0
    recall = true_positive / len(truth) if truth else 0.0
    specificity = (
        true_negative / (true_negative + false_positive)
        if true_negative + false_positive
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "predicted_nodes": len(predicted),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "balanced_accuracy": round((recall + specificity) / 2.0, 6),
    }


def _profile(
    transactions: list[Transaction],
    all_nodes: set[int],
    positive_nodes: set[int],
    fraud_steps: dict[int, int],
) -> dict[str, int | float | list[str]]:
    exact_counts = Counter(transactions)
    pair_time_counts = Counter((source, target, step) for source, target, _, step in transactions)
    referenced = {node for source, target, _, _ in transactions for node in (source, target)}
    amounts = [amount for _, _, amount, _ in transactions]
    steps = [step for _, _, _, step in transactions]
    return {
        "node_rows": len(all_nodes),
        "transaction_rows": len(transactions),
        "fraud_nodes": len(positive_nodes),
        "fraud_node_rate": round(len(positive_nodes) / len(all_nodes), 6),
        "fraud_nodes_without_labeled_step": sum(
            1 for node in positive_nodes if fraud_steps[node] < 0
        ),
        "exact_duplicate_transactions": sum(count - 1 for count in exact_counts.values()),
        "duplicate_source_target_time_rows": sum(
            count - 1 for count in pair_time_counts.values()
        ),
        "self_loops": sum(source == target for source, target, _, _ in transactions),
        "nonpositive_amounts": sum(amount <= 0 for amount in amounts),
        "orphan_references": len(referenced - all_nodes),
        "min_amount": min(amounts),
        "max_amount": max(amounts),
        "min_step": min(steps),
        "max_step": max(steps),
        "quality_flags": [
            "synthetic_data",
            "node_level_labels_only",
            "duplicate_transactions_are_informative",
            "fraudStep_is_incomplete_for_positive_nodes",
        ],
    }


def _predictors(
    dataset_name: str,
    transactions: list[Transaction],
    all_nodes: set[int],
) -> tuple[dict[str, set[int]], dict[str, float | int]]:
    degree = Counter()
    group_counts = Counter()
    for source, target, _, step in transactions:
        degree[source] += 1
        degree[target] += 1
        group_counts[(source, target, step)] += 1

    q1 = _percentile((degree[node] for node in all_nodes), 0.25)
    q3 = _percentile((degree[node] for node in all_nodes), 0.75)
    degree_threshold = q3 + 1.5 * (q3 - q1)
    degree_iqr = {node for node in all_nodes if degree[node] > degree_threshold}

    repeated_edges = {
        (source, target)
        for (source, target, _), count in group_counts.items()
        if count >= 2
    }
    repeated_edge_nodes = {
        node for edge in repeated_edges for node in edge
    }

    amount_threshold = _adaptive_amount_threshold(transactions)
    low_value_edges = [
        (source, target)
        for source, target, amount, _ in transactions
        if amount < amount_threshold
    ]
    low_value_nodes = {node for edge in low_value_edges for node in edge}

    incoming: dict[int, set[int]] = defaultdict(set)
    for source, target in low_value_edges:
        incoming[target].add(source)
    fan_in_anchors = {target for target, sources in incoming.items() if len(sources) >= 2}
    fan_in_nodes = set(fan_in_anchors)
    for source, target in low_value_edges:
        if target in fan_in_anchors:
            fan_in_nodes.add(source)

    cyclic_components = [
        component
        for component in _strongly_connected_components(low_value_edges)
        if len(component) >= 2
    ]
    cycle_nodes = set().union(*cyclic_components) if cyclic_components else set()

    if "fanin200cycle200" in dataset_name:
        pattern_aware = fan_in_nodes | cycle_nodes
    elif "fanin" in dataset_name:
        pattern_aware = fan_in_nodes
    else:
        pattern_aware = cycle_nodes

    predictors = {
        "degree_iqr": degree_iqr,
        "repeated_edge_time": repeated_edge_nodes,
        "adaptive_low_value": low_value_nodes,
        "low_value_fan_in": fan_in_nodes,
        "low_value_cycle": cycle_nodes,
        "pattern_aware_union": pattern_aware,
    }
    diagnostics: dict[str, float | int] = {
        "degree_iqr_threshold": round(degree_threshold, 6),
        "adaptive_amount_threshold": round(amount_threshold, 6),
        "repeated_edge_count": len(repeated_edges),
        "low_value_edge_rows": len(low_value_edges),
        "fan_in_risk_events": len(fan_in_anchors),
        "cycle_risk_events": len(cyclic_components),
    }
    return predictors, diagnostics


def run_amlsim_benchmarks(
    data_root: str | Path,
    output_dir: str | Path,
) -> dict:
    root = Path(data_root)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    dataset_results: list[dict] = []
    metric_rows: list[dict] = []
    quality_rows: list[dict] = []
    for dataset_dir in sorted(path for path in root.glob("20K_*200") if path.is_dir()):
        started = time.perf_counter()
        transactions = _read_transactions(dataset_dir / "transactions.csv")
        all_nodes, truth, fraud_steps = _read_labels(dataset_dir / "nodes.csv")
        profile = _profile(transactions, all_nodes, truth, fraud_steps)
        predictors, diagnostics = _predictors(dataset_dir.name, transactions, all_nodes)
        metrics = {
            name: _classification_metrics(predicted, truth, all_nodes)
            for name, predicted in predictors.items()
        }
        elapsed = time.perf_counter() - started
        dataset_results.append(
            {
                "dataset": dataset_dir.name,
                "profile": profile,
                "diagnostics": diagnostics,
                "algorithms": metrics,
                "runtime_seconds": round(elapsed, 6),
            }
        )
        quality_rows.append({"dataset": dataset_dir.name, **profile, **diagnostics})
        for algorithm, values in metrics.items():
            metric_rows.append(
                {
                    "dataset": dataset_dir.name,
                    "algorithm": algorithm,
                    **values,
                    "runtime_seconds_total": round(elapsed, 6),
                }
            )

    result = {
        "experiment": "AMLSim label-safe graph-pattern baselines",
        "label_policy": (
            "isFraud and fraudStep are loaded only after prediction and are used "
            "only to compute evaluation metrics"
        ),
        "datasets": dataset_results,
    }
    (destination / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(destination / "metrics.csv", metric_rows)
    _write_csv(destination / "data_quality.csv", quality_rows)
    return result


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
