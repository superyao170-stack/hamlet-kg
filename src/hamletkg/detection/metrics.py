"""Leakage-safe chronological splits and AML transaction detection metrics.

Average precision is the stepwise area under a precision-recall curve, with
equal scores evaluated as one group. It is not trapezoidal PR-AUC. Daily review
budgets cover every candidate, including days whose rounded budget is zero.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime
from numbers import Real
from typing import Any, Protocol, Sequence

from ..models import parse_time


class TimestampedSample(Protocol):
    timestamp: datetime | str
    event_id: str


def _finite_number(value: Any, name: str) -> float:
    if not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _timestamp(value: datetime | str) -> datetime:
    if not isinstance(value, (datetime, str)):
        raise ValueError("timestamps must be datetime objects or ISO 8601 strings")
    try:
        return parse_time(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"invalid timestamp: {value!r}") from error


def _event_id(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("event_ids must be nonempty strings")
    return value


def _validate_pairs(
    labels: Sequence[int], scores: Sequence[float]
) -> tuple[list[int], list[float]]:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    if not labels:
        raise ValueError("labels and scores must be nonempty")
    checked_labels: list[int] = []
    checked_scores: list[float] = []
    for index, (label, score) in enumerate(zip(labels, scores)):
        if not isinstance(label, Real) or label not in (0, 1):
            raise ValueError(f"labels[{index}] must be binary (0 or 1)")
        checked_labels.append(int(label))
        checked_score = _finite_number(score, f"scores[{index}]")
        if not 0.0 <= checked_score <= 1.0:
            raise ValueError(f"scores[{index}] must be between 0 and 1; None is not a score")
        checked_scores.append(checked_score)
    return checked_labels, checked_scores


def chronological_split(
    samples: Sequence[TimestampedSample],
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> dict[str, list[int]]:
    """Return original indices, sorted by UTC time and event ID.

    Complete timestamp groups are assigned to each split. Fractions are target
    sample counts, so large groups may cause actual proportions to differ.
    Boundaries are chosen closest to the requested cumulative sample counts,
    while reserving at least one timestamp group for every remaining split.
    """
    train_fraction = _finite_number(train_fraction, "train_fraction")
    validation_fraction = _finite_number(validation_fraction, "validation_fraction")
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("train_fraction and validation_fraction must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be less than 1")
    if len(samples) < 3:
        raise ValueError("chronological splitting requires at least three samples")

    records: list[tuple[datetime, str, int]] = []
    for index, sample in enumerate(samples):
        try:
            timestamp, event_id = sample.timestamp, sample.event_id
        except AttributeError as error:
            raise ValueError("samples must have timestamp and event_id attributes") from error
        records.append((_timestamp(timestamp), _event_id(event_id), index))
    records.sort()

    boundaries: list[int] = []
    for position in range(1, len(records)):
        if records[position][0] != records[position - 1][0]:
            boundaries.append(position)
    if len(boundaries) < 2:
        raise ValueError("chronological splitting requires at least three distinct timestamps")

    train_target = len(records) * train_fraction
    train_end = min(boundaries[:-1], key=lambda boundary: (abs(boundary - train_target), boundary))
    validation_target = len(records) * (train_fraction + validation_fraction)
    validation_end = min(
        (boundary for boundary in boundaries if boundary > train_end),
        key=lambda boundary: (abs(boundary - validation_target), boundary),
    )
    sorted_indices = [record[2] for record in records]
    return {
        "train": sorted_indices[:train_end],
        "validation": sorted_indices[train_end:validation_end],
        "test": sorted_indices[validation_end:],
    }


def _score_groups(labels: list[int], scores: list[float]):
    """Yield (score, count, positive count) in descending score order."""
    ranked = sorted(zip(scores, labels), reverse=True)
    position = 0
    while position < len(ranked):
        score = ranked[position][0]
        end = position + 1
        while end < len(ranked) and ranked[end][0] == score:
            end += 1
        yield score, end - position, sum(label for _, label in ranked[position:end])
        position = end


def _average_precision(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    if not positives:
        return None
    seen = true_positive = 0
    average_precision = 0.0
    for _, count, group_positive in _score_groups(labels, scores):
        seen += count
        true_positive += group_positive
        average_precision += group_positive / positives * true_positive / seen
    return average_precision


def _confusion_metrics(total: int, positives: int, alerts: int, true_positive: int) -> dict:
    false_positive = alerts - true_positive
    false_negative = positives - true_positive
    return {
        "total": total,
        "positives": positives,
        "prevalence": positives / total,
        "predicted_positive": alerts,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": total - positives - false_positive,
        "false_negative": false_negative,
        "precision": true_positive / alerts if alerts else 0.0,
        "recall": true_positive / positives if positives else None,
        "f1": 2 * true_positive / (positives + alerts) if positives else None,
        "false_positives_per_1000": false_positive / total * 1000,
    }


def binary_metrics(
    labels: Sequence[int], scores: Sequence[float], threshold: float = 0.5
) -> dict:
    """Evaluate all candidates using score >= threshold.

    A finite threshold above one selects no alerts. Precision is zero when no
    alerts are selected. AP, recall and F1 are None when no positives exist,
    because such data cannot assess positive-class detection performance.
    """
    checked_labels, checked_scores = _validate_pairs(labels, scores)
    threshold = _finite_number(threshold, "threshold")
    if threshold < 0:
        raise ValueError("threshold must be nonnegative")
    selected = [label for label, score in zip(checked_labels, checked_scores) if score >= threshold]
    result = _confusion_metrics(len(checked_labels), sum(checked_labels), len(selected), sum(selected))
    result.update({"average_precision": _average_precision(checked_labels, checked_scores), "threshold": threshold})
    return result


def choose_threshold(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Choose maximum positive-class F1 from validation data only.

    Equal-F1 choices keep the highest threshold, limiting alert volume. This
    routine cannot infer data provenance: callers must supply validation data,
    never test labels. Both classes are required for threshold selection.
    """
    checked_labels, checked_scores = _validate_pairs(labels, scores)
    positives = sum(checked_labels)
    if positives == 0 or positives == len(checked_labels):
        raise ValueError("validation data must contain both positive and negative labels to choose a threshold")
    best_threshold = math.nextafter(1.0, math.inf)
    best_positive, best_denominator = 0, 1
    alerts = true_positive = 0
    for threshold, count, group_positive in _score_groups(checked_labels, checked_scores):
        alerts += count
        true_positive += group_positive
        denominator = positives + alerts
        # Compare exact rational F1 values; sorting already puts high thresholds first.
        if true_positive * best_denominator > best_positive * denominator:
            best_threshold = threshold
            best_positive, best_denominator = true_positive, denominator
    return best_threshold


def daily_budget_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    timestamps: Sequence[datetime | str],
    event_ids: Sequence[str],
    budget_fraction: float = 0.01,
) -> dict:
    """Evaluate floor(fraction * daily candidates) reviews per UTC day.

    Every candidate must have a valid score. A caller must not discard
    candidates that receive no score; such None values are rejected. Ranking
    ties use ascending event ID. Zero-budget days remain in the recall and
    false-positive denominators and are explicitly reported.
    """
    checked_labels, checked_scores = _validate_pairs(labels, scores)
    if len(timestamps) != len(labels) or len(event_ids) != len(labels):
        raise ValueError("labels, scores, timestamps and event_ids must have the same length")
    budget_fraction = _finite_number(budget_fraction, "budget_fraction")
    if not 0 <= budget_fraction <= 1:
        raise ValueError("budget_fraction must be between 0 and 1")
    checked_ids = [_event_id(value) for value in event_ids]
    if len(set(checked_ids)) != len(checked_ids):
        raise ValueError("event_ids must be unique across candidates")

    by_date: dict[str, list[int]] = defaultdict(list)
    for index, timestamp in enumerate(timestamps):
        by_date[_timestamp(timestamp).date().isoformat()].append(index)
    daily: list[dict] = []
    total_alerts = total_true_positive = 0
    for date, indices in sorted(by_date.items()):
        budget = math.floor(len(indices) * budget_fraction)
        ranked = sorted(indices, key=lambda index: (-checked_scores[index], checked_ids[index]))
        selected = ranked[:budget]
        true_positive = sum(checked_labels[index] for index in selected)
        metrics = _confusion_metrics(len(indices), sum(checked_labels[index] for index in indices), len(selected), true_positive)
        metrics.update({"date": date, "budget": budget, "review_alerts": len(selected), "selected_event_ids": [checked_ids[index] for index in selected]})
        daily.append(metrics)
        total_alerts += len(selected)
        total_true_positive += true_positive
    result = _confusion_metrics(len(checked_labels), sum(checked_labels), total_alerts, total_true_positive)
    result.update({"budget_fraction": budget_fraction, "review_alerts": total_alerts, "daily": daily})
    return result
