"""Typed domain models for the three event granularities.

The project deliberately keeps source provenance as metadata instead of a
standalone evidence layer.  Risk events are derived objects and therefore
always retain the identifiers of the atomic events that produced them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable


def parse_time(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        normalized = value.strip().replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def iso_time(value: datetime) -> str:
    return parse_time(value).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class Entity:
    entity_id: str
    entity_type: str
    attributes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Entity":
        return cls(
            entity_id=str(value["entity_id"]),
            entity_type=str(value["entity_type"]).upper(),
            attributes=dict(value.get("attributes") or {}),
        )


@dataclass(frozen=True)
class AtomicEvent:
    event_id: str
    action_type: str
    roles: dict[str, str]
    timestamp: datetime
    attributes: dict[str, Any] = field(default_factory=dict)
    source_record_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_type", self.action_type.upper())
        object.__setattr__(self, "timestamp", parse_time(self.timestamp))
        if not self.roles:
            raise ValueError("atomic event must contain at least one participant role")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AtomicEvent":
        return cls(
            event_id=str(value["event_id"]),
            action_type=str(value["action_type"]),
            roles={str(k): str(v) for k, v in value["roles"].items()},
            timestamp=parse_time(value["timestamp"]),
            attributes=dict(value.get("attributes") or {}),
            source_record_id=(
                str(value["source_record_id"])
                if value.get("source_record_id") is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["timestamp"] = iso_time(self.timestamp)
        return result


@dataclass(frozen=True)
class GraphRelation:
    source_id: str
    relation_type: str
    target_id: str
    attributes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GraphRelation":
        return cls(
            source_id=str(value["source_id"]),
            relation_type=str(value["relation_type"]).upper(),
            target_id=str(value["target_id"]),
            attributes=dict(value.get("attributes") or {}),
        )


@dataclass(frozen=True)
class ActionClause:
    clause_id: str
    action_type: str
    anchor_role: str
    min_count: int = 1
    max_count: int | None = None
    distinct_role: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ActionClause":
        return cls(
            clause_id=str(value["clause_id"]),
            action_type=str(value["action_type"]).upper(),
            anchor_role=str(value["anchor_role"]),
            min_count=int(value.get("min_count", 1)),
            max_count=(int(value["max_count"]) if value.get("max_count") else None),
            distinct_role=(str(value["distinct_role"]) if value.get("distinct_role") else None),
        )


@dataclass(frozen=True)
class PatternConstraint:
    constraint_type: str
    parameters: dict[str, Any]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PatternConstraint":
        return cls(
            constraint_type=str(value["constraint_type"]).lower(),
            parameters=dict(value.get("parameters") or {}),
        )


@dataclass(frozen=True)
class RiskPattern:
    pattern_id: str
    risk_type: str
    description: str
    clauses: tuple[ActionClause, ...]
    constraints: tuple[PatternConstraint, ...] = ()
    version: str = "1.0"
    quality: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.clauses:
            raise ValueError("risk pattern must contain at least one action clause")
        clause_ids = [clause.clause_id for clause in self.clauses]
        if len(clause_ids) != len(set(clause_ids)):
            raise ValueError("risk pattern clause_id values must be unique")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RiskPattern":
        return cls(
            pattern_id=str(value["pattern_id"]),
            risk_type=str(value["risk_type"]),
            description=str(value.get("description") or ""),
            clauses=tuple(ActionClause.from_dict(item) for item in value["clauses"]),
            constraints=tuple(
                PatternConstraint.from_dict(item)
                for item in value.get("constraints") or []
            ),
            version=str(value.get("version") or "1.0"),
            quality=dict(value.get("quality") or {}),
            provenance=dict(value.get("provenance") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RiskEvent:
    risk_event_id: str
    risk_type: str
    pattern_id: str
    anchor_entity_id: str
    atomic_event_ids: tuple[str, ...]
    start_time: datetime
    end_time: datetime
    metrics: dict[str, float] = field(default_factory=dict)
    confidence: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "start_time", parse_time(self.start_time))
        object.__setattr__(self, "end_time", parse_time(self.end_time))
        if self.end_time < self.start_time:
            raise ValueError("risk event end_time cannot precede start_time")
        if not self.atomic_event_ids:
            raise ValueError("risk event must be materialized from atomic events")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RiskEvent":
        return cls(
            risk_event_id=str(value["risk_event_id"]),
            risk_type=str(value["risk_type"]),
            pattern_id=str(value["pattern_id"]),
            anchor_entity_id=str(value["anchor_entity_id"]),
            atomic_event_ids=tuple(str(item) for item in value["atomic_event_ids"]),
            start_time=parse_time(value["start_time"]),
            end_time=parse_time(value["end_time"]),
            metrics={str(k): float(v) for k, v in (value.get("metrics") or {}).items()},
            confidence=(
                float(value["confidence"])
                if value.get("confidence") is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["start_time"] = iso_time(self.start_time)
        result["end_time"] = iso_time(self.end_time)
        return result


@dataclass
class CaseGraph:
    case_id: str
    entities: dict[str, Entity]
    atomic_events: dict[str, AtomicEvent]
    atomic_relations: list[GraphRelation] = field(default_factory=list)
    risk_events: dict[str, RiskEvent] = field(default_factory=dict)
    risk_relations: list[GraphRelation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        event_ids = set(self.atomic_events)
        entity_ids = set(self.entities)
        for event in self.atomic_events.values():
            unknown = set(event.roles.values()) - entity_ids
            if unknown:
                raise ValueError(f"event {event.event_id} references unknown entities: {sorted(unknown)}")
        for risk_event in self.risk_events.values():
            unknown = set(risk_event.atomic_event_ids) - event_ids
            if unknown:
                raise ValueError(
                    f"risk event {risk_event.risk_event_id} references unknown atomic events: {sorted(unknown)}"
                )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CaseGraph":
        entities = {
            item["entity_id"]: Entity.from_dict(item)
            for item in value.get("entities") or []
        }
        atomic_events = {
            item["event_id"]: AtomicEvent.from_dict(item)
            for item in value.get("atomic_events") or []
        }
        risk_events = {
            item["risk_event_id"]: RiskEvent.from_dict(item)
            for item in value.get("risk_events") or []
        }
        result = cls(
            case_id=str(value["case_id"]),
            entities=entities,
            atomic_events=atomic_events,
            atomic_relations=[
                GraphRelation.from_dict(item)
                for item in value.get("atomic_relations") or []
            ],
            risk_events=risk_events,
            risk_relations=[
                GraphRelation.from_dict(item)
                for item in value.get("risk_relations") or []
            ],
            metadata=dict(value.get("metadata") or {}),
        )
        result.validate()
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "entities": [asdict(item) for item in self.entities.values()],
            "atomic_events": [item.to_dict() for item in self.atomic_events.values()],
            "atomic_relations": [asdict(item) for item in self.atomic_relations],
            "risk_events": [item.to_dict() for item in self.risk_events.values()],
            "risk_relations": [asdict(item) for item in self.risk_relations],
            "metadata": self.metadata,
        }

    def atomic_events_for(self, ids: Iterable[str]) -> list[AtomicEvent]:
        return [self.atomic_events[item] for item in ids if item in self.atomic_events]
