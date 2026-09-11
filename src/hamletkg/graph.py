"""Construction of computable atomic-event and risk-event relations."""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta

from .models import CaseGraph, GraphRelation


def build_atomic_relations(
    case: CaseGraph,
    *,
    max_gap: timedelta = timedelta(days=30),
) -> list[GraphRelation]:
    """Create sparse temporal and flow relations between atomic actions.

    Only consecutive events for the same entity are connected.  This avoids
    producing a quadratic temporal closure while preserving a traversable
    event stream.  Full temporal reachability can be recovered transitively.
    """

    by_entity: dict[str, list] = defaultdict(list)
    for event in case.atomic_events.values():
        for entity_id in set(event.roles.values()):
            by_entity[entity_id].append(event)

    relations: dict[tuple[str, str, str], GraphRelation] = {}
    for entity_id, events in by_entity.items():
        ordered = sorted(events, key=lambda item: (item.timestamp, item.event_id))
        for left, right in zip(ordered, ordered[1:]):
            delta = right.timestamp - left.timestamp
            if delta > max_gap:
                continue
            key = (left.event_id, "NEXT", right.event_id)
            relations[key] = GraphRelation(
                source_id=left.event_id,
                relation_type="NEXT",
                target_id=right.event_id,
                attributes={
                    "shared_entity_id": entity_id,
                    "delta_seconds": delta.total_seconds(),
                },
            )

    events = sorted(case.atomic_events.values(), key=lambda item: item.timestamp)
    outgoing_by_entity: dict[str, list] = defaultdict(list)
    for event in events:
        for role, entity_id in event.roles.items():
            if role.lower() in {"sender", "payer", "source"}:
                outgoing_by_entity[entity_id].append(event)
    for left in events:
        receivers = {
            value
            for role, value in left.roles.items()
            if role.lower() in {"receiver", "beneficiary", "destination"}
        }
        if not receivers:
            continue
        for through_entity in receivers:
            for right in outgoing_by_entity.get(through_entity, []):
                if right.timestamp < left.timestamp or right.event_id == left.event_id:
                    continue
                if right.timestamp - left.timestamp > max_gap:
                    break
                key = (left.event_id, "FUNDS_FLOW_TO", right.event_id)
                existing = relations.get(key)
                through_ids = set(
                    (existing.attributes.get("through_entity_ids") if existing else []) or []
                )
                through_ids.add(through_entity)
                relations[key] = GraphRelation(
                    source_id=left.event_id,
                    relation_type="FUNDS_FLOW_TO",
                    target_id=right.event_id,
                    attributes={"through_entity_ids": sorted(through_ids)},
                )

    case.atomic_relations = sorted(
        relations.values(),
        key=lambda item: (item.source_id, item.relation_type, item.target_id),
    )
    return case.atomic_relations


def build_risk_relations(case: CaseGraph) -> list[GraphRelation]:
    """Build the case-level graph whose nodes are materialized risk events."""

    risk_events = sorted(
        case.risk_events.values(),
        key=lambda item: (item.start_time, item.end_time, item.risk_event_id),
    )
    atomic_entities = {
        event_id: set(case.atomic_events[event_id].roles.values())
        for event_id in case.atomic_events
    }
    relations: dict[tuple[str, str, str], GraphRelation] = {}

    for index, left in enumerate(risk_events):
        left_entities = set().union(
            *(atomic_entities[event_id] for event_id in left.atomic_event_ids)
        )
        for right in risk_events[index + 1 :]:
            right_entities = set().union(
                *(atomic_entities[event_id] for event_id in right.atomic_event_ids)
            )
            if left.end_time <= right.start_time:
                key = (left.risk_event_id, "PRECEDES", right.risk_event_id)
                relations[key] = GraphRelation(
                    source_id=left.risk_event_id,
                    relation_type="PRECEDES",
                    target_id=right.risk_event_id,
                    attributes={
                        "delta_seconds": (
                            right.start_time - left.end_time
                        ).total_seconds()
                    },
                )
            shared = sorted(left_entities & right_entities)
            if shared:
                key = (left.risk_event_id, "SHARES_ENTITY", right.risk_event_id)
                relations[key] = GraphRelation(
                    source_id=left.risk_event_id,
                    relation_type="SHARES_ENTITY",
                    target_id=right.risk_event_id,
                    attributes={"entity_ids": shared},
                )
            if left.anchor_entity_id == right.anchor_entity_id:
                key = (left.risk_event_id, "SAME_ANCHOR", right.risk_event_id)
                relations[key] = GraphRelation(
                    source_id=left.risk_event_id,
                    relation_type="SAME_ANCHOR",
                    target_id=right.risk_event_id,
                    attributes={},
                )

    case.risk_relations = sorted(
        relations.values(),
        key=lambda item: (item.source_id, item.relation_type, item.target_id),
    )
    return case.risk_relations
