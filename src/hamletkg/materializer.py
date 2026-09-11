"""Deterministic materialization of risk events from atomic-event patterns."""

from __future__ import annotations

import hashlib
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable

from .graph import build_risk_relations
from .models import AtomicEvent, CaseGraph, RiskEvent, RiskPattern
from .patterns import validate_pattern


@dataclass
class MatchTrace:
    pattern_id: str
    anchor_entity_id: str
    accepted: bool
    reason: str
    clause_event_ids: dict[str, list[str]]
    metrics: dict[str, float]
    stage: str = "full_history"
    window_start: str | None = None
    window_end: str | None = None


@dataclass
class MatchingDiagnostics:
    """Operational counters used in pruning and scalability experiments."""

    total_entities: int = 0
    candidate_anchors: int = 0
    pruned_anchors: int = 0
    windows_evaluated: int = 0
    accepted_windows: int = 0
    duplicate_windows: int = 0
    overlap_suppressed: int = 0
    event_lookups: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "total_entities": self.total_entities,
            "candidate_anchors": self.candidate_anchors,
            "pruned_anchors": self.pruned_anchors,
            "windows_evaluated": self.windows_evaluated,
            "accepted_windows": self.accepted_windows,
            "duplicate_windows": self.duplicate_windows,
            "overlap_suppressed": self.overlap_suppressed,
            "event_lookups": self.event_lookups,
        }


class AtomicEventIndex:
    """Reusable inverted index over action type, participant role and entity."""

    def __init__(self, case: CaseGraph) -> None:
        self.by_signature: dict[tuple[str, str, str], list[AtomicEvent]] = defaultdict(list)
        self.entities_by_clause: dict[tuple[str, str], set[str]] = defaultdict(set)
        for event in case.atomic_events.values():
            for role, entity_id in event.roles.items():
                self.by_signature[(event.action_type, role, entity_id)].append(event)
                self.entities_by_clause[(event.action_type, role)].add(entity_id)
        for events in self.by_signature.values():
            events.sort(key=lambda item: (item.timestamp, item.event_id))

    def events(self, action_type: str, role: str, entity_id: str) -> list[AtomicEvent]:
        return self.by_signature.get((action_type, role, entity_id), [])

    def candidate_entities(self, action_type: str, role: str) -> set[str]:
        return self.entities_by_clause.get((action_type, role), set())


class PatternMatcher:
    """Match bounded aggregation patterns around an anchor entity.

    The DSL intentionally describes a tractable pattern family rather than
    arbitrary subgraph isomorphism.  Every clause selects atomic actions by
    type and by the role played by one shared anchor entity.
    """

    def match(
        self,
        case: CaseGraph,
        pattern: RiskPattern,
    ) -> tuple[list[RiskEvent], list[MatchTrace]]:
        validate_pattern(pattern)
        anchors = sorted(case.entities)
        event_index: dict[tuple[str, str, str], list[AtomicEvent]] = defaultdict(list)
        for event in case.atomic_events.values():
            for role, entity_id in event.roles.items():
                event_index[(event.action_type, role, entity_id)].append(event)
        for events in event_index.values():
            events.sort(key=lambda item: (item.timestamp, item.event_id))
        matched: list[RiskEvent] = []
        traces: list[MatchTrace] = []
        for anchor in anchors:
            groups = self._candidate_groups(pattern, anchor, event_index)
            accepted, reason, metrics = self._apply_constraints(pattern, groups)
            selected_ids = {
                clause_id: [event.event_id for event in events]
                for clause_id, events in groups.items()
            }
            traces.append(
                MatchTrace(
                    pattern_id=pattern.pattern_id,
                    anchor_entity_id=anchor,
                    accepted=accepted,
                    reason=reason,
                    clause_event_ids=selected_ids,
                    metrics=metrics,
                )
            )
            if not accepted:
                continue
            events = sorted(
                {event.event_id: event for items in groups.values() for event in items}.values(),
                key=lambda item: (item.timestamp, item.event_id),
            )
            identity = "|".join(
                [case.case_id, pattern.pattern_id, anchor, *(item.event_id for item in events)]
            )
            suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
            matched.append(
                RiskEvent(
                    risk_event_id=f"RISK-{suffix}",
                    risk_type=pattern.risk_type,
                    pattern_id=pattern.pattern_id,
                    anchor_entity_id=anchor,
                    atomic_event_ids=tuple(item.event_id for item in events),
                    start_time=events[0].timestamp,
                    end_time=events[-1].timestamp,
                    metrics=metrics,
                    confidence=self._pattern_confidence(pattern),
                )
            )
        return matched, traces

    def _candidate_groups(
        self,
        pattern: RiskPattern,
        anchor: str,
        event_index: dict[tuple[str, str, str], list[AtomicEvent]],
    ) -> dict[str, list[AtomicEvent]]:
        groups: dict[str, list[AtomicEvent]] = {}
        for clause in pattern.clauses:
            events = list(event_index.get((clause.action_type, clause.anchor_role, anchor), []))
            if clause.max_count is not None:
                events = events[: clause.max_count]
            groups[clause.clause_id] = events
        return groups

    def _apply_constraints(
        self,
        pattern: RiskPattern,
        groups: dict[str, list[AtomicEvent]],
    ) -> tuple[bool, str, dict[str, float]]:
        metrics: dict[str, float] = {}

        # Attribute ranges are selection predicates, not whole-window vetoes.
        # For example, a structuring rule selects sub-threshold cash deposits
        # before checking whether enough such deposits occurred in the window.
        for constraint in pattern.constraints:
            if constraint.constraint_type != "attribute_range":
                continue
            params = constraint.parameters
            clause_id = str(params["clause"])
            attribute = str(params["attribute"])
            minimum = params.get("minimum")
            maximum = params.get("maximum")
            selected: list[AtomicEvent] = []
            for event in groups[clause_id]:
                try:
                    value = float(event.attributes[attribute])
                except (KeyError, TypeError, ValueError):
                    continue
                if minimum is not None and value < float(minimum):
                    continue
                if maximum is not None and value > float(maximum):
                    continue
                selected.append(event)
            groups[clause_id] = selected
            metrics[f"{clause_id}_{attribute}_range_selected"] = float(len(selected))

        for constraint in pattern.constraints:
            if constraint.constraint_type != "ordered_within":
                continue
            params = constraint.parameters
            earlier_id = str(params["earlier"])
            later_id = str(params["later"])
            minimum = float(params.get("min_seconds", 0))
            maximum = float(params["max_seconds"])
            valid_earlier: dict[str, AtomicEvent] = {}
            valid_later: dict[str, AtomicEvent] = {}
            deltas: list[float] = []
            for earlier in groups[earlier_id]:
                for later in groups[later_id]:
                    delta = (later.timestamp - earlier.timestamp).total_seconds()
                    if minimum <= delta <= maximum:
                        valid_earlier[earlier.event_id] = earlier
                        valid_later[later.event_id] = later
                        deltas.append(delta)
            groups[earlier_id] = sorted(
                valid_earlier.values(), key=lambda item: item.timestamp
            )
            groups[later_id] = sorted(
                valid_later.values(), key=lambda item: item.timestamp
            )
            if deltas:
                metrics[f"{earlier_id}_{later_id}_min_delta_seconds"] = min(deltas)
                metrics[f"{earlier_id}_{later_id}_max_delta_seconds"] = max(deltas)

        for clause in pattern.clauses:
            events = groups[clause.clause_id]
            if len(events) < clause.min_count:
                return False, f"clause {clause.clause_id} has insufficient events", metrics
            if clause.distinct_role:
                distinct = {
                    event.roles.get(clause.distinct_role)
                    for event in events
                    if event.roles.get(clause.distinct_role)
                }
                metrics[f"{clause.clause_id}_distinct_{clause.distinct_role}"] = float(
                    len(distinct)
                )
                if len(distinct) < clause.min_count:
                    return (
                        False,
                        f"clause {clause.clause_id} has insufficient distinct role values",
                        metrics,
                    )
            metrics[f"{clause.clause_id}_count"] = float(len(events))
            amount = self._sum_attribute(events, "amount")
            if amount:
                metrics[f"{clause.clause_id}_amount"] = amount

        all_events = [event for events in groups.values() for event in events]
        if not all_events:
            return False, "pattern has no matching events", metrics
        duration = (
            max(item.timestamp for item in all_events)
            - min(item.timestamp for item in all_events)
        ).total_seconds()
        metrics["duration_seconds"] = duration

        for constraint in pattern.constraints:
            kind = constraint.constraint_type
            params = constraint.parameters
            if kind == "ordered_within":
                continue
            if kind == "attribute_range":
                continue
            if kind == "sum_ratio":
                numerator_id = str(params["numerator"])
                denominator_id = str(params["denominator"])
                attribute = str(params.get("attribute", "amount"))
                numerator = self._sum_attribute(groups[numerator_id], attribute)
                denominator = self._sum_attribute(groups[denominator_id], attribute)
                ratio = numerator / denominator if denominator > 0 else 0.0
                metrics[f"{numerator_id}_{denominator_id}_{attribute}_ratio"] = ratio
                if ratio < float(params.get("minimum", 0)):
                    return False, f"sum_ratio {ratio:.6f} is below minimum", metrics
                maximum = params.get("maximum")
                if maximum is not None and ratio > float(maximum):
                    return False, f"sum_ratio {ratio:.6f} exceeds maximum", metrics
            elif kind == "total_amount":
                clause_id = str(params["clause"])
                attribute = str(params.get("attribute", "amount"))
                total = self._sum_attribute(groups[clause_id], attribute)
                metrics[f"{clause_id}_{attribute}_total"] = total
                if total < float(params.get("minimum", 0)):
                    return False, f"total_amount {total:.6f} is below minimum", metrics
                maximum = params.get("maximum")
                if maximum is not None and total > float(maximum):
                    return False, f"total_amount {total:.6f} exceeds maximum", metrics
            elif kind == "duration_max":
                if duration > float(params["max_seconds"]):
                    return False, f"duration {duration:.0f}s exceeds maximum", metrics
            elif kind == "role_overlap":
                left_id = str(params["left_clause"])
                right_id = str(params["right_clause"])
                left_role = str(params["left_role"])
                right_role = str(params["right_role"])
                left_values = {
                    event.roles[left_role]
                    for event in groups[left_id]
                    if event.roles.get(left_role)
                }
                right_values = {
                    event.roles[right_role]
                    for event in groups[right_id]
                    if event.roles.get(right_role)
                }
                overlap = left_values & right_values
                metrics[
                    f"{left_id}_{left_role}_{right_id}_{right_role}_overlap"
                ] = float(len(overlap))
                minimum = int(params.get("minimum", 1))
                if len(overlap) < minimum:
                    return False, "role_overlap has insufficient shared values", metrics

        return True, "all clauses and constraints satisfied", metrics

    @staticmethod
    def _sum_attribute(events: Iterable[AtomicEvent], attribute: str) -> float:
        total = 0.0
        for event in events:
            value = event.attributes.get(attribute)
            try:
                total += float(value)
            except (TypeError, ValueError):
                continue
        return total

    @staticmethod
    def _pattern_confidence(pattern: RiskPattern) -> float | None:
        value = pattern.quality.get("calibrated_confidence")
        if value is None:
            return None
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return None


class IndexedWindowPatternMatcher(PatternMatcher):
    """CP-REM prototype: indexed, window-bounded risk-event materialization.

    The matcher first prunes entities that cannot satisfy the clause counts,
    then evaluates only windows ending at relevant event timestamps.  Accepted
    windows are de-duplicated and strongly overlapping windows are suppressed.
    This lets one account produce multiple temporally separated risk events.
    """

    def __init__(
        self,
        *,
        default_window_seconds: float = 86400.0,
        overlap_threshold: float = 0.8,
    ) -> None:
        if default_window_seconds <= 0:
            raise ValueError("default_window_seconds must be positive")
        if not 0 <= overlap_threshold <= 1:
            raise ValueError("overlap_threshold must be between 0 and 1")
        self.default_window_seconds = float(default_window_seconds)
        self.overlap_threshold = float(overlap_threshold)
        self.last_diagnostics = MatchingDiagnostics()

    def match(
        self,
        case: CaseGraph,
        pattern: RiskPattern,
    ) -> tuple[list[RiskEvent], list[MatchTrace]]:
        events, traces, diagnostics = self.match_with_diagnostics(case, pattern)
        self.last_diagnostics = diagnostics
        return events, traces

    def match_with_diagnostics(
        self,
        case: CaseGraph,
        pattern: RiskPattern,
    ) -> tuple[list[RiskEvent], list[MatchTrace], MatchingDiagnostics]:
        validate_pattern(pattern)
        index = AtomicEventIndex(case)
        diagnostics = MatchingDiagnostics(total_entities=len(case.entities))
        anchors = self._candidate_anchors(pattern, index, diagnostics)
        diagnostics.candidate_anchors = len(anchors)
        diagnostics.pruned_anchors = diagnostics.total_entities - len(anchors)
        window_seconds = self._window_seconds(pattern)

        traces: list[MatchTrace] = []
        raw_candidates: list[dict] = []
        seen_event_sets: set[tuple[str, ...]] = set()
        for anchor in sorted(anchors):
            complete_groups = self._indexed_groups(pattern, anchor, index, diagnostics)
            group_times = {
                clause_id: [item.timestamp for item in items]
                for clause_id, items in complete_groups.items()
            }
            timeline = sorted(
                {event.event_id: event for group in complete_groups.values() for event in group}.values(),
                key=lambda item: (item.timestamp, item.event_id),
            )
            for window_end in (item.timestamp for item in timeline):
                window_start = window_end - timedelta(seconds=window_seconds)
                groups: dict[str, list[AtomicEvent]] = {}
                for clause in pattern.clauses:
                    times = group_times[clause.clause_id]
                    left = bisect_left(times, window_start)
                    right = bisect_right(times, window_end)
                    selected = complete_groups[clause.clause_id][left:right]
                    if clause.max_count is not None:
                        selected = selected[-clause.max_count :]
                    groups[clause.clause_id] = selected
                if not self._minimum_counts_possible(pattern, groups):
                    continue
                diagnostics.windows_evaluated += 1
                accepted, reason, metrics = self._apply_constraints(pattern, groups)
                selected_ids = {
                    clause_id: [event.event_id for event in items]
                    for clause_id, items in groups.items()
                }
                trace = MatchTrace(
                    pattern_id=pattern.pattern_id,
                    anchor_entity_id=anchor,
                    accepted=accepted,
                    reason=reason,
                    clause_event_ids=selected_ids,
                    metrics=metrics,
                    stage="indexed_window",
                    window_start=window_start.isoformat(),
                    window_end=window_end.isoformat(),
                )
                traces.append(trace)
                if not accepted:
                    continue
                selected = sorted(
                    {event.event_id: event for items in groups.values() for event in items}.values(),
                    key=lambda item: (item.timestamp, item.event_id),
                )
                identity = tuple(item.event_id for item in selected)
                if identity in seen_event_sets:
                    trace.accepted = False
                    trace.reason = "duplicate accepted window"
                    diagnostics.duplicate_windows += 1
                    continue
                seen_event_sets.add(identity)
                raw_candidates.append(
                    {
                        "anchor": anchor,
                        "events": selected,
                        "metrics": metrics,
                        "trace": trace,
                    }
                )

        accepted_candidates: list[dict] = []
        for candidate in sorted(
            raw_candidates,
            key=lambda item: (
                item["events"][-1].timestamp,
                len(item["events"]),
                item["events"][0].timestamp,
            ),
        ):
            event_ids = {item.event_id for item in candidate["events"]}
            overlaps = [
                self._overlap_coefficient(
                    event_ids, {item.event_id for item in kept["events"]}
                )
                for kept in accepted_candidates
                if kept["anchor"] == candidate["anchor"]
            ]
            if overlaps and max(overlaps) >= self.overlap_threshold:
                candidate["trace"].accepted = False
                candidate["trace"].reason = "overlapping accepted window suppressed"
                diagnostics.overlap_suppressed += 1
                continue
            accepted_candidates.append(candidate)

        risk_events = [
            self._materialize_candidate(case, pattern, item)
            for item in accepted_candidates
        ]
        diagnostics.accepted_windows = len(risk_events)
        self.last_diagnostics = diagnostics
        return risk_events, traces, diagnostics

    def _candidate_anchors(
        self,
        pattern: RiskPattern,
        index: AtomicEventIndex,
        diagnostics: MatchingDiagnostics,
    ) -> set[str]:
        clause_candidates: list[set[str]] = []
        for clause in pattern.clauses:
            valid: set[str] = set()
            for entity_id in index.candidate_entities(clause.action_type, clause.anchor_role):
                events = index.events(clause.action_type, clause.anchor_role, entity_id)
                diagnostics.event_lookups += 1
                if len(events) < clause.min_count:
                    continue
                if clause.distinct_role:
                    distinct = {
                        event.roles.get(clause.distinct_role)
                        for event in events
                        if event.roles.get(clause.distinct_role)
                    }
                    if len(distinct) < clause.min_count:
                        continue
                valid.add(entity_id)
            clause_candidates.append(valid)
        if not clause_candidates:
            return set()
        result = set(clause_candidates[0])
        for values in clause_candidates[1:]:
            result.intersection_update(values)
        return result

    @staticmethod
    def _indexed_groups(
        pattern: RiskPattern,
        anchor: str,
        index: AtomicEventIndex,
        diagnostics: MatchingDiagnostics,
    ) -> dict[str, list[AtomicEvent]]:
        groups = {}
        for clause in pattern.clauses:
            diagnostics.event_lookups += 1
            groups[clause.clause_id] = list(
                index.events(clause.action_type, clause.anchor_role, anchor)
            )
        return groups

    @staticmethod
    def _minimum_counts_possible(
        pattern: RiskPattern,
        groups: dict[str, list[AtomicEvent]],
    ) -> bool:
        for clause in pattern.clauses:
            events = groups[clause.clause_id]
            if len(events) < clause.min_count:
                return False
            if clause.distinct_role:
                distinct = {
                    event.roles.get(clause.distinct_role)
                    for event in events
                    if event.roles.get(clause.distinct_role)
                }
                if len(distinct) < clause.min_count:
                    return False
        return True

    def _window_seconds(self, pattern: RiskPattern) -> float:
        ordered = [
            float(item.parameters["max_seconds"])
            for item in pattern.constraints
            if item.constraint_type == "ordered_within"
        ]
        duration_limits = [
            float(item.parameters["max_seconds"])
            for item in pattern.constraints
            if item.constraint_type == "duration_max"
        ]
        value = sum(ordered) if ordered else self.default_window_seconds
        if duration_limits:
            value = min(value, min(duration_limits))
        return max(1.0, value)

    @staticmethod
    def _overlap_coefficient(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / min(len(left), len(right))

    @staticmethod
    def _materialize_candidate(
        case: CaseGraph,
        pattern: RiskPattern,
        candidate: dict,
    ) -> RiskEvent:
        events = candidate["events"]
        anchor = candidate["anchor"]
        identity = "|".join(
            [case.case_id, pattern.pattern_id, anchor, *(item.event_id for item in events)]
        )
        suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
        return RiskEvent(
            risk_event_id=f"RISK-{suffix}",
            risk_type=pattern.risk_type,
            pattern_id=pattern.pattern_id,
            anchor_entity_id=anchor,
            atomic_event_ids=tuple(item.event_id for item in events),
            start_time=events[0].timestamp,
            end_time=events[-1].timestamp,
            metrics=candidate["metrics"],
            confidence=PatternMatcher._pattern_confidence(pattern),
        )


def materialize_risk_events(
    case: CaseGraph,
    patterns: Iterable[RiskPattern],
    *,
    matcher: PatternMatcher | None = None,
) -> list[MatchTrace]:
    matcher = matcher or IndexedWindowPatternMatcher()
    traces: list[MatchTrace] = []
    materialized: dict[str, RiskEvent] = {}
    for pattern in patterns:
        events, current_traces = matcher.match(case, pattern)
        traces.extend(current_traces)
        materialized.update({item.risk_event_id: item for item in events})
    case.risk_events = materialized
    build_risk_relations(case)
    case.validate()
    return traces
