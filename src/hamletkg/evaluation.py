"""Reproducible retrieval metrics for case-level experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import Mapping, Sequence

from .models import CaseGraph, RiskEvent
from .similarity import HierarchicalCaseMatcher


@dataclass(frozen=True)
class RetrievalMetrics:
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    evaluated_queries: int

    def to_dict(self) -> dict:
        return {
            "recall_at_k": round(self.recall_at_k, 6),
            "mrr": round(self.mrr, 6),
            "ndcg_at_k": round(self.ndcg_at_k, 6),
            "evaluated_queries": self.evaluated_queries,
        }


@dataclass(frozen=True)
class RiskEventDetectionMetrics:
    precision: float
    recall: float
    f1: float
    true_positive: int
    false_positive: int
    false_negative: int
    predicted_events: int
    reference_events: int
    matches: tuple[dict[str, float | str], ...]

    def to_dict(self) -> dict:
        return {
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "predicted_events": self.predicted_events,
            "reference_events": self.reference_events,
            "matches": list(self.matches),
        }


def evaluate_risk_event_detection(
    predicted: Sequence[RiskEvent],
    reference: Sequence[RiskEvent],
    *,
    minimum_atomic_iou: float = 0.5,
    require_same_type: bool = True,
    require_same_anchor: bool = True,
) -> RiskEventDetectionMetrics:
    """Evaluate event instances using maximum bipartite threshold matching."""

    if not 0 <= minimum_atomic_iou <= 1:
        raise ValueError("minimum_atomic_iou must be between 0 and 1")
    adjacency: list[list[tuple[int, float]]] = []
    for predicted_event in predicted:
        candidates = []
        predicted_ids = set(predicted_event.atomic_event_ids)
        for index, reference_event in enumerate(reference):
            if require_same_type and predicted_event.risk_type != reference_event.risk_type:
                continue
            if (
                require_same_anchor
                and predicted_event.anchor_entity_id != reference_event.anchor_entity_id
            ):
                continue
            reference_ids = set(reference_event.atomic_event_ids)
            union = predicted_ids | reference_ids
            iou = len(predicted_ids & reference_ids) / len(union) if union else 0.0
            if iou >= minimum_atomic_iou:
                candidates.append((index, iou))
        candidates.sort(key=lambda item: (-item[1], reference[item[0]].risk_event_id))
        adjacency.append(candidates)

    matched_reference: dict[int, int] = {}

    def augment(predicted_index: int, visited: set[int]) -> bool:
        for reference_index, _ in adjacency[predicted_index]:
            if reference_index in visited:
                continue
            visited.add(reference_index)
            previous = matched_reference.get(reference_index)
            if previous is None or augment(previous, visited):
                matched_reference[reference_index] = predicted_index
                return True
        return False

    for predicted_index in range(len(predicted)):
        augment(predicted_index, set())

    pairs = []
    for reference_index, predicted_index in sorted(matched_reference.items()):
        iou = next(
            score
            for candidate_index, score in adjacency[predicted_index]
            if candidate_index == reference_index
        )
        pairs.append(
            {
                "predicted_risk_event_id": predicted[predicted_index].risk_event_id,
                "reference_risk_event_id": reference[reference_index].risk_event_id,
                "atomic_iou": round(iou, 6),
            }
        )
    true_positive = len(pairs)
    false_positive = len(predicted) - true_positive
    false_negative = len(reference) - true_positive
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(reference) if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return RiskEventDetectionMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        predicted_events=len(predicted),
        reference_events=len(reference),
        matches=tuple(pairs),
    )


def evaluate_retrieval(
    cases: Sequence[CaseGraph],
    relevance: Mapping[str, Sequence[str]],
    *,
    top_k: int = 5,
    matcher: HierarchicalCaseMatcher | None = None,
) -> RetrievalMetrics:
    engine = matcher or HierarchicalCaseMatcher()
    by_id = {case.case_id: case for case in cases}
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    for query_id, relevant_values in relevance.items():
        if query_id not in by_id:
            continue
        relevant = set(relevant_values) & set(by_id)
        relevant.discard(query_id)
        if not relevant:
            continue
        ranked = engine.rank(by_id[query_id], cases, top_k=max(top_k, len(cases)))
        ranked_ids = [item.candidate_case_id for item in ranked]
        top_ids = ranked_ids[:top_k]
        recalls.append(len(relevant & set(top_ids)) / len(relevant))
        first_rank = next(
            (index for index, case_id in enumerate(ranked_ids, start=1) if case_id in relevant),
            None,
        )
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
        dcg = sum(
            (1.0 / math.log2(index + 1)) if case_id in relevant else 0.0
            for index, case_id in enumerate(top_ids, start=1)
        )
        ideal_hits = min(len(relevant), top_k)
        ideal = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
        ndcgs.append(dcg / ideal if ideal else 0.0)
    if not recalls:
        return RetrievalMetrics(0.0, 0.0, 0.0, 0)
    return RetrievalMetrics(
        recall_at_k=mean(recalls),
        mrr=mean(reciprocal_ranks),
        ndcg_at_k=mean(ndcgs),
        evaluated_queries=len(recalls),
    )
