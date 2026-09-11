"""Hierarchical case retrieval over materialized risk-event graphs."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .models import CaseGraph, RiskEvent


def _cosine(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 1.0
    numerator = sum(float(left.get(key, 0)) * float(right.get(key, 0)) for key in keys)
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left.values()))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(0.0, min(1.0, numerator / (left_norm * right_norm)))


def _text_units(value: str) -> set[str]:
    normalized = re.sub(r"\s+", "", value.lower())
    if len(normalized) < 2:
        return {normalized} if normalized else set()
    return {normalized[index : index + 2] for index in range(len(normalized) - 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right) if left | right else 0.0


def _ratio_similarity(left: float, right: float) -> float:
    if left == 0 and right == 0:
        return 1.0
    if left <= 0 or right <= 0:
        return 0.0
    return min(left, right) / max(left, right)


@dataclass(frozen=True)
class SimilarityWeights:
    risk_type: float = 0.35
    atomic_composition: float = 0.30
    role_profile: float = 0.20
    temporal_numeric: float = 0.15
    case_event_alignment: float = 0.55
    case_structure: float = 0.20
    case_sequence: float = 0.15
    case_cardinality: float = 0.10

    def validate(self) -> None:
        risk_total = (
            self.risk_type
            + self.atomic_composition
            + self.role_profile
            + self.temporal_numeric
        )
        case_total = (
            self.case_event_alignment
            + self.case_structure
            + self.case_sequence
            + self.case_cardinality
        )
        if abs(risk_total - 1.0) > 1e-9 or abs(case_total - 1.0) > 1e-9:
            raise ValueError("risk-event weights and case weights must each sum to 1")


@dataclass
class RiskEventSimilarity:
    score: float
    risk_type: float
    atomic_composition: float
    role_profile: float
    temporal_numeric: float


@dataclass
class CaseSimilarityResult:
    query_case_id: str
    candidate_case_id: str
    similarity: float
    event_alignment: float
    structure: float
    sequence: float
    cardinality: float
    aligned_risk_events: list[dict[str, float | str]] = field(default_factory=list)
    algorithm: str = "hierarchical_sinkhorn"
    diagnostics: dict[str, float | int | str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "query_case_id": self.query_case_id,
            "candidate_case_id": self.candidate_case_id,
            "algorithm": self.algorithm,
            "similarity": round(self.similarity, 6),
            "components": {
                "event_alignment": round(self.event_alignment, 6),
                "structure": round(self.structure, 6),
                "sequence": round(self.sequence, 6),
                "cardinality": round(self.cardinality, 6),
            },
            "aligned_risk_events": self.aligned_risk_events,
            "diagnostics": self.diagnostics,
        }


class HierarchicalCaseMatcher:
    """Compare case graphs at the risk-event level.

    Atomic events influence the representation of each risk event but are not
    globally matched across cases.  Entropic optimal transport softly aligns
    unequal risk-event sets before case-level structure is compared.
    """

    def __init__(
        self,
        weights: SimilarityWeights | None = None,
        *,
        regularization: float = 0.12,
        sinkhorn_iterations: int = 120,
    ) -> None:
        self.weights = weights or SimilarityWeights()
        self.weights.validate()
        self.regularization = regularization
        self.sinkhorn_iterations = sinkhorn_iterations

    def risk_event_similarity(
        self,
        left_case: CaseGraph,
        left: RiskEvent,
        right_case: CaseGraph,
        right: RiskEvent,
    ) -> RiskEventSimilarity:
        type_score = (
            1.0
            if left.risk_type == right.risk_type
            else _jaccard(_text_units(left.risk_type), _text_units(right.risk_type))
        )
        left_actions = Counter(
            left_case.atomic_events[event_id].action_type
            for event_id in left.atomic_event_ids
        )
        right_actions = Counter(
            right_case.atomic_events[event_id].action_type
            for event_id in right.atomic_event_ids
        )
        action_score = _cosine(left_actions, right_actions)

        left_relations = self._induced_atomic_relation_profile(left_case, left)
        right_relations = self._induced_atomic_relation_profile(right_case, right)
        relation_score = _cosine(left_relations, right_relations)
        atomic_score = 0.75 * action_score + 0.25 * relation_score

        role_score = _cosine(
            self._role_profile(left_case, left),
            self._role_profile(right_case, right),
        )
        left_duration = max(1.0, (left.end_time - left.start_time).total_seconds())
        right_duration = max(1.0, (right.end_time - right.start_time).total_seconds())
        duration_score = _ratio_similarity(left_duration, right_duration)
        amount_score = _ratio_similarity(
            self._risk_amount(left_case, left),
            self._risk_amount(right_case, right),
        )
        temporal_numeric = 0.6 * duration_score + 0.4 * amount_score

        score = (
            self.weights.risk_type * type_score
            + self.weights.atomic_composition * atomic_score
            + self.weights.role_profile * role_score
            + self.weights.temporal_numeric * temporal_numeric
        )
        return RiskEventSimilarity(
            score=max(0.0, min(1.0, score)),
            risk_type=type_score,
            atomic_composition=atomic_score,
            role_profile=role_score,
            temporal_numeric=temporal_numeric,
        )

    def compare(self, query: CaseGraph, candidate: CaseGraph) -> CaseSimilarityResult:
        query_events = sorted(
            query.risk_events.values(),
            key=lambda item: (
                item.start_time,
                item.end_time,
                item.risk_type,
                item.risk_event_id,
            ),
        )
        candidate_events = sorted(
            candidate.risk_events.values(),
            key=lambda item: (
                item.start_time,
                item.end_time,
                item.risk_type,
                item.risk_event_id,
            ),
        )
        if not query_events or not candidate_events:
            return CaseSimilarityResult(
                query_case_id=query.case_id,
                candidate_case_id=candidate.case_id,
                similarity=0.0,
                event_alignment=0.0,
                structure=0.0,
                sequence=0.0,
                cardinality=0.0,
            )

        # Reflexivity is a required property of the proposed case similarity.
        # Entropic transport otherwise assigns a small mass to off-diagonal
        # pairs even when a case is compared with its identical representation.
        if query.to_dict() == candidate.to_dict():
            return CaseSimilarityResult(
                query_case_id=query.case_id,
                candidate_case_id=candidate.case_id,
                similarity=1.0,
                event_alignment=1.0,
                structure=1.0,
                sequence=1.0,
                cardinality=1.0,
                aligned_risk_events=[
                    {
                        "query_risk_event_id": item.risk_event_id,
                        "candidate_risk_event_id": item.risk_event_id,
                        "query_risk_type": item.risk_type,
                        "candidate_risk_type": item.risk_type,
                        "pair_similarity": 1.0,
                        "transport_mass": round(1.0 / len(query_events), 6),
                    }
                    for item in query_events
                ],
            )

        matrix: list[list[float]] = []
        for left in query_events:
            row_details = [
                self.risk_event_similarity(query, left, candidate, right)
                for right in candidate_events
            ]
            matrix.append([item.score for item in row_details])

        transport = self._sinkhorn(matrix)
        event_alignment = sum(
            transport[i][j] * matrix[i][j]
            for i in range(len(query_events))
            for j in range(len(candidate_events))
        )
        structure = _cosine(
            Counter(item.relation_type for item in query.risk_relations),
            Counter(item.relation_type for item in candidate.risk_relations),
        )
        sequence = self._sequence_similarity(query_events, candidate_events)
        cardinality = min(len(query_events), len(candidate_events)) / max(
            len(query_events), len(candidate_events)
        )
        total = (
            self.weights.case_event_alignment * event_alignment
            + self.weights.case_structure * structure
            + self.weights.case_sequence * sequence
            + self.weights.case_cardinality * cardinality
        )
        alignments = []
        for i, left in enumerate(query_events):
            j = max(range(len(candidate_events)), key=lambda index: transport[i][index])
            right = candidate_events[j]
            alignments.append(
                {
                    "query_risk_event_id": left.risk_event_id,
                    "candidate_risk_event_id": right.risk_event_id,
                    "query_risk_type": left.risk_type,
                    "candidate_risk_type": right.risk_type,
                    "pair_similarity": round(matrix[i][j], 6),
                    "transport_mass": round(transport[i][j], 6),
                }
            )
        return CaseSimilarityResult(
            query_case_id=query.case_id,
            candidate_case_id=candidate.case_id,
            similarity=max(0.0, min(1.0, total)),
            event_alignment=event_alignment,
            structure=structure,
            sequence=sequence,
            cardinality=cardinality,
            aligned_risk_events=alignments,
        )

    def rank(
        self,
        query: CaseGraph,
        candidates: Sequence[CaseGraph],
        *,
        top_k: int = 5,
    ) -> list[CaseSimilarityResult]:
        results = [
            self.compare(query, candidate)
            for candidate in candidates
            if candidate.case_id != query.case_id
        ]
        results.sort(key=lambda item: (-item.similarity, item.candidate_case_id))
        return results[: max(0, top_k)]

    def _sinkhorn(self, similarities: list[list[float]]) -> list[list[float]]:
        rows, columns = len(similarities), len(similarities[0])
        kernel = [
            [math.exp(-(1.0 - score) / self.regularization) for score in row]
            for row in similarities
        ]
        row_mass = [1.0 / rows] * rows
        column_mass = [1.0 / columns] * columns
        u = [1.0] * rows
        v = [1.0] * columns
        epsilon = 1e-12
        for _ in range(self.sinkhorn_iterations):
            u = [
                row_mass[i]
                / max(epsilon, sum(kernel[i][j] * v[j] for j in range(columns)))
                for i in range(rows)
            ]
            v = [
                column_mass[j]
                / max(epsilon, sum(kernel[i][j] * u[i] for i in range(rows)))
                for j in range(columns)
            ]
        return [
            [u[i] * kernel[i][j] * v[j] for j in range(columns)]
            for i in range(rows)
        ]

    @staticmethod
    def _induced_atomic_relation_profile(
        case: CaseGraph, risk_event: RiskEvent
    ) -> Counter:
        event_ids = set(risk_event.atomic_event_ids)
        return Counter(
            relation.relation_type
            for relation in case.atomic_relations
            if relation.source_id in event_ids and relation.target_id in event_ids
        )

    @staticmethod
    def _role_profile(case: CaseGraph, risk_event: RiskEvent) -> Counter:
        result: Counter = Counter()
        for event_id in risk_event.atomic_event_ids:
            event = case.atomic_events[event_id]
            for role, entity_id in event.roles.items():
                entity_type = case.entities[entity_id].entity_type
                result[f"{event.action_type}:{role}:{entity_type}"] += 1
        return result

    @staticmethod
    def _risk_amount(case: CaseGraph, risk_event: RiskEvent) -> float:
        amounts = []
        for event_id in risk_event.atomic_event_ids:
            try:
                amounts.append(float(case.atomic_events[event_id].attributes.get("amount", 0)))
            except (TypeError, ValueError):
                continue
        return sum(amounts)

    @staticmethod
    def _sequence_similarity(
        left: Sequence[RiskEvent], right: Sequence[RiskEvent]
    ) -> float:
        left_types = [item.risk_type for item in left]
        right_types = [item.risk_type for item in right]
        table = [[0] * (len(right_types) + 1) for _ in range(len(left_types) + 1)]
        for i, left_type in enumerate(left_types, start=1):
            for j, right_type in enumerate(right_types, start=1):
                if left_type == right_type:
                    table[i][j] = table[i - 1][j - 1] + 1
                else:
                    table[i][j] = max(table[i - 1][j], table[i][j - 1])
        return table[-1][-1] / max(len(left_types), len(right_types))


class FusedGraphCaseMatcher(HierarchicalCaseMatcher):
    """Entropic FGW-style alignment over risk-event semantics and topology.

    This implementation keeps the project dependency-free.  It alternates a
    Sinkhorn transport update with a Gromov-style relational distortion cost.
    It is intended as the proposed reproducible matcher; POT's FGW remains an
    external reference implementation for experiments.
    """

    def __init__(
        self,
        weights: SimilarityWeights | None = None,
        *,
        semantic_weight: float = 0.65,
        regularization: float = 0.08,
        sinkhorn_iterations: int = 100,
        fused_iterations: int = 20,
        tolerance: float = 1e-7,
    ) -> None:
        super().__init__(
            weights=weights,
            regularization=regularization,
            sinkhorn_iterations=sinkhorn_iterations,
        )
        if not 0 <= semantic_weight <= 1:
            raise ValueError("semantic_weight must be between 0 and 1")
        if fused_iterations < 1:
            raise ValueError("fused_iterations must be positive")
        self.semantic_weight = float(semantic_weight)
        self.fused_iterations = int(fused_iterations)
        self.tolerance = float(tolerance)

    def compare(self, query: CaseGraph, candidate: CaseGraph) -> CaseSimilarityResult:
        query_events = self._ordered_events(query)
        candidate_events = self._ordered_events(candidate)
        if not query_events or not candidate_events:
            return CaseSimilarityResult(
                query_case_id=query.case_id,
                candidate_case_id=candidate.case_id,
                similarity=0.0,
                event_alignment=0.0,
                structure=0.0,
                sequence=0.0,
                cardinality=0.0,
                algorithm="fused_graph_sinkhorn",
            )
        if query.to_dict() == candidate.to_dict():
            result = super().compare(query, candidate)
            result.algorithm = "fused_graph_sinkhorn"
            result.diagnostics = {
                "semantic_weight": self.semantic_weight,
                "fused_iterations": 0,
                "converged": 1,
            }
            return result

        similarities = [
            [
                self.risk_event_similarity(query, left, candidate, right).score
                for right in candidate_events
            ]
            for left in query_events
        ]
        semantic_cost = [[1.0 - value for value in row] for row in similarities]
        query_distances = self._relational_distances(query, query_events)
        candidate_distances = self._relational_distances(candidate, candidate_events)
        transport = self._sinkhorn(similarities)
        iterations = 0
        converged = False
        normalized_structural_cost = [
            [0.0 for _ in candidate_events] for _ in query_events
        ]
        for iteration in range(self.fused_iterations):
            structural_cost = self._gromov_cost(
                query_distances, candidate_distances, transport
            )
            normalized_structural_cost = self._normalize_matrix(structural_cost)
            fused_cost = [
                [
                    self.semantic_weight * semantic_cost[i][j]
                    + (1.0 - self.semantic_weight) * normalized_structural_cost[i][j]
                    for j in range(len(candidate_events))
                ]
                for i in range(len(query_events))
            ]
            updated = self._sinkhorn_cost(fused_cost)
            delta = sum(
                abs(updated[i][j] - transport[i][j])
                for i in range(len(query_events))
                for j in range(len(candidate_events))
            )
            transport = updated
            iterations = iteration + 1
            if delta <= self.tolerance:
                converged = True
                break

        semantic_alignment = sum(
            transport[i][j] * similarities[i][j]
            for i in range(len(query_events))
            for j in range(len(candidate_events))
        )
        structural_distortion = sum(
            transport[i][j] * normalized_structural_cost[i][j]
            for i in range(len(query_events))
            for j in range(len(candidate_events))
        )
        structural_alignment = max(0.0, min(1.0, 1.0 - structural_distortion))
        sequence = self._sequence_similarity(query_events, candidate_events)
        cardinality = min(len(query_events), len(candidate_events)) / max(
            len(query_events), len(candidate_events)
        )
        total = (
            self.weights.case_event_alignment * semantic_alignment
            + self.weights.case_structure * structural_alignment
            + self.weights.case_sequence * sequence
            + self.weights.case_cardinality * cardinality
        )
        alignments = self._alignment_rows(
            query_events, candidate_events, similarities, transport
        )
        return CaseSimilarityResult(
            query_case_id=query.case_id,
            candidate_case_id=candidate.case_id,
            similarity=max(0.0, min(1.0, total)),
            event_alignment=semantic_alignment,
            structure=structural_alignment,
            sequence=sequence,
            cardinality=cardinality,
            aligned_risk_events=alignments,
            algorithm="fused_graph_sinkhorn",
            diagnostics={
                "semantic_weight": self.semantic_weight,
                "regularization": self.regularization,
                "fused_iterations": iterations,
                "converged": int(converged),
                "structural_distortion": round(structural_distortion, 8),
            },
        )

    def _sinkhorn_cost(self, costs: list[list[float]]) -> list[list[float]]:
        similarities = [
            [max(0.0, min(1.0, 1.0 - value)) for value in row]
            for row in self._normalize_matrix(costs)
        ]
        return self._sinkhorn(similarities)

    @staticmethod
    def _ordered_events(case: CaseGraph) -> list[RiskEvent]:
        return sorted(
            case.risk_events.values(),
            key=lambda item: (
                item.start_time,
                item.end_time,
                item.risk_type,
                item.risk_event_id,
            ),
        )

    @staticmethod
    def _normalize_matrix(values: list[list[float]]) -> list[list[float]]:
        flattened = [value for row in values for value in row]
        if not flattened:
            return values
        minimum, maximum = min(flattened), max(flattened)
        if maximum - minimum <= 1e-12:
            return [[0.0 for _ in row] for row in values]
        scale = maximum - minimum
        return [[(value - minimum) / scale for value in row] for row in values]

    @staticmethod
    def _relational_distances(
        case: CaseGraph,
        events: Sequence[RiskEvent],
    ) -> list[list[float]]:
        size = len(events)
        if size == 1:
            return [[0.0]]
        positions = {item.risk_event_id: index for index, item in enumerate(events)}
        unreachable = float(size + 1)
        distances = [
            [0.0 if i == j else unreachable for j in range(size)]
            for i in range(size)
        ]
        relation_weights = {"SAME_ANCHOR": 0.75, "SHARES_ENTITY": 1.0, "PRECEDES": 1.25}
        for relation in case.risk_relations:
            if relation.source_id not in positions or relation.target_id not in positions:
                continue
            left, right = positions[relation.source_id], positions[relation.target_id]
            weight = relation_weights.get(relation.relation_type, 1.5)
            distances[left][right] = min(distances[left][right], weight)
            distances[right][left] = min(distances[right][left], weight)
        for pivot in range(size):
            for left in range(size):
                for right in range(size):
                    through = distances[left][pivot] + distances[pivot][right]
                    if through < distances[left][right]:
                        distances[left][right] = through
        maximum = max(value for row in distances for value in row)
        if maximum <= 0:
            return distances
        return [[value / maximum for value in row] for row in distances]

    @staticmethod
    def _gromov_cost(
        left_distances: list[list[float]],
        right_distances: list[list[float]],
        transport: list[list[float]],
    ) -> list[list[float]]:
        rows, columns = len(transport), len(transport[0])
        result = [[0.0 for _ in range(columns)] for _ in range(rows)]
        for i in range(rows):
            for j in range(columns):
                total = 0.0
                for k in range(rows):
                    for l in range(columns):
                        difference = left_distances[i][k] - right_distances[j][l]
                        total += difference * difference * transport[k][l]
                result[i][j] = total
        return result

    @staticmethod
    def _alignment_rows(
        query_events: Sequence[RiskEvent],
        candidate_events: Sequence[RiskEvent],
        similarities: list[list[float]],
        transport: list[list[float]],
    ) -> list[dict[str, float | str]]:
        result = []
        for i, left in enumerate(query_events):
            j = max(range(len(candidate_events)), key=lambda index: transport[i][index])
            right = candidate_events[j]
            result.append(
                {
                    "query_risk_event_id": left.risk_event_id,
                    "candidate_risk_event_id": right.risk_event_id,
                    "query_risk_type": left.risk_type,
                    "candidate_risk_type": right.risk_type,
                    "pair_similarity": round(similarities[i][j], 6),
                    "transport_mass": round(transport[i][j], 6),
                }
            )
        return result


@dataclass
class RetrievalDiagnostics:
    total_candidates: int
    indexed_candidates: int
    reranked_candidates: int
    pruned_candidates: int

    def to_dict(self) -> dict[str, int]:
        return {
            "total_candidates": self.total_candidates,
            "indexed_candidates": self.indexed_candidates,
            "reranked_candidates": self.reranked_candidates,
            "pruned_candidates": self.pruned_candidates,
        }


class RiskCaseIndex:
    """Inverted risk-type index followed by exact fused-graph re-ranking."""

    def __init__(self, cases: Sequence[CaseGraph]) -> None:
        self.cases = {case.case_id: case for case in cases}
        self.by_risk_type: dict[str, set[str]] = {}
        for case in cases:
            for risk_type in {item.risk_type for item in case.risk_events.values()}:
                self.by_risk_type.setdefault(risk_type, set()).add(case.case_id)

    def search(
        self,
        query: CaseGraph,
        *,
        matcher: HierarchicalCaseMatcher | None = None,
        top_k: int = 5,
        candidate_pool_size: int = 50,
    ) -> tuple[list[CaseSimilarityResult], RetrievalDiagnostics]:
        matcher = matcher or FusedGraphCaseMatcher()
        available = {
            case_id for case_id in self.cases if case_id != query.case_id
        }
        indexed: set[str] = set()
        for risk_type in {item.risk_type for item in query.risk_events.values()}:
            indexed.update(self.by_risk_type.get(risk_type, set()))
        indexed.intersection_update(available)
        if len(indexed) < min(top_k, len(available)):
            indexed = available
        coarse = sorted(
            indexed,
            key=lambda case_id: (
                -self._coarse_similarity(query, self.cases[case_id]),
                case_id,
            ),
        )
        pool_size = max(top_k, candidate_pool_size)
        selected = coarse[:pool_size]
        results = [matcher.compare(query, self.cases[case_id]) for case_id in selected]
        results.sort(key=lambda item: (-item.similarity, item.candidate_case_id))
        diagnostics = RetrievalDiagnostics(
            total_candidates=len(available),
            indexed_candidates=len(indexed),
            reranked_candidates=len(selected),
            pruned_candidates=len(available) - len(selected),
        )
        return results[: max(0, top_k)], diagnostics

    @staticmethod
    def _coarse_similarity(left: CaseGraph, right: CaseGraph) -> float:
        left_types = Counter(item.risk_type for item in left.risk_events.values())
        right_types = Counter(item.risk_type for item in right.risk_events.values())
        left_relations = Counter(item.relation_type for item in left.risk_relations)
        right_relations = Counter(item.relation_type for item in right.risk_relations)
        return 0.75 * _cosine(left_types, right_types) + 0.25 * _cosine(
            left_relations, right_relations
        )
