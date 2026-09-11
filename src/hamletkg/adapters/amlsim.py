"""Adapters for the public IBM AMLSim sample schema.

The node labels are retained only under ``CaseGraph.metadata`` for evaluation.
They are never copied into entity attributes or atomic-event features.
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..graph import build_atomic_relations
from ..models import AtomicEvent, CaseGraph, Entity, parse_time


def case_from_amlsim_csv(
    transactions_path: str | Path,
    nodes_path: str | Path,
    *,
    case_id: str,
    limit: int | None = None,
    base_time: str | datetime = "2017-01-01T00:00:00Z",
    step_seconds: int = 3600,
    build_relations: bool = True,
) -> CaseGraph:
    """Convert AMLSim CSV files into a typed atomic-event graph.

    ``isFraud`` and ``fraudStep`` are outcome labels.  Keeping them in metadata
    makes evaluation possible while preventing label leakage into the graph.
    """

    transaction_source = Path(transactions_path)
    node_source = Path(nodes_path)
    start = parse_time(base_time)

    entities: dict[str, Entity] = {}
    evaluation_labels: dict[str, int] = {}
    evaluation_steps: dict[str, int] = {}
    with node_source.open("r", encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=2):
            raw_id = str(row.get("nodeid", "")).strip()
            if not raw_id:
                raise ValueError(f"row {index} in nodes.csv is missing nodeid")
            entity_id = f"ACCOUNT:AMLSIM:{raw_id}"
            try:
                initial_balance = float(row.get("init_balance", 0) or 0)
            except ValueError as exc:
                raise ValueError(
                    f"row {index} has invalid init_balance: {row.get('init_balance')}"
                ) from exc
            entities[entity_id] = Entity(
                entity_id=entity_id,
                entity_type="ACCOUNT",
                attributes={"initial_balance": initial_balance},
            )
            try:
                evaluation_labels[entity_id] = int(float(row.get("isFraud", 0) or 0))
                evaluation_steps[entity_id] = int(float(row.get("fraudStep", -1) or -1))
            except ValueError as exc:
                raise ValueError(f"row {index} contains an invalid evaluation label") from exc

    events: dict[str, AtomicEvent] = {}
    with transaction_source.open("r", encoding="utf-8-sig", newline="") as handle:
        for record_index, row in enumerate(csv.DictReader(handle), start=1):
            if limit is not None and record_index > limit:
                break
            sender = f"ACCOUNT:AMLSIM:{str(row.get('sourceNodeId', '')).strip()}"
            receiver = f"ACCOUNT:AMLSIM:{str(row.get('targetNodeId', '')).strip()}"
            if sender not in entities or receiver not in entities:
                raise ValueError(
                    f"transaction row {record_index + 1} references an unknown account"
                )
            try:
                amount = float(row.get("value", 0) or 0)
                step = int(float(row.get("time", 0) or 0))
            except ValueError as exc:
                raise ValueError(
                    f"transaction row {record_index + 1} has invalid value or time"
                ) from exc
            event_id = f"AMLSIM-TX-{record_index:09d}"
            events[event_id] = AtomicEvent(
                event_id=event_id,
                action_type="TRANSFER",
                roles={"sender": sender, "receiver": receiver},
                timestamp=start + timedelta(seconds=step * step_seconds),
                attributes={"amount": amount, "simulation_step": step},
                source_record_id=str(record_index),
            )

    case = CaseGraph(
        case_id=case_id,
        entities=entities,
        atomic_events=events,
        metadata={
            "dataset": "IBM AMLSim",
            "transactions_file": str(transaction_source.resolve()),
            "nodes_file": str(node_source.resolve()),
            "base_time": start.astimezone(timezone.utc).isoformat(),
            "step_seconds": step_seconds,
            "evaluation_labels": evaluation_labels,
            "evaluation_fraud_steps": evaluation_steps,
            "labels_are_features": False,
        },
    )
    if build_relations:
        build_atomic_relations(case)
    case.validate()
    return case
