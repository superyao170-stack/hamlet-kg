"""Adapter for the public IBM synthetic AML transaction CSV schema."""

from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..graph import build_atomic_relations
from ..models import AtomicEvent, CaseGraph, Entity


def _first(row: dict[str, str], *names: str, default: str = "") -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def _timestamp(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        parsed = None
        for format_string in (
            "%Y/%m/%d %H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
        ):
            try:
                parsed = datetime.strptime(normalized, format_string)
                break
            except ValueError:
                continue
        if parsed is None:
            raise ValueError(f"unsupported transaction timestamp: {value}")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _amount(value: str) -> float:
    try:
        return float(value.replace(",", ""))
    except (AttributeError, ValueError):
        return 0.0


def case_from_transaction_csv(
    path: str | Path,
    *,
    case_id: str,
    limit: int | None = None,
) -> CaseGraph:
    """Convert transaction rows without exposing the laundering label as input.

    The original label, when present, is retained only under case metadata for
    evaluation.  It is never included in an atomic event's attributes.
    """

    source = Path(path)
    entities: dict[str, Entity] = {}
    events: dict[str, AtomicEvent] = {}
    labels: dict[str, int] = {}
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            if limit is not None and index > limit:
                break
            sender_bank = _first(row, "From Bank", "from_bank", "sender_bank")
            sender_account = _first(
                row, "From Account", "Account", "from_account", "sender_account"
            )
            receiver_bank = _first(row, "To Bank", "to_bank", "receiver_bank")
            receiver_account = _first(
                row,
                "To Account",
                "Account.1",
                "to_account",
                "receiver_account",
            )
            if not sender_account or not receiver_account:
                raise ValueError(f"row {index} is missing sender or receiver account")
            sender_id = f"ACCOUNT:{sender_bank}:{sender_account}"
            receiver_id = f"ACCOUNT:{receiver_bank}:{receiver_account}"
            entities.setdefault(
                sender_id,
                Entity(sender_id, "ACCOUNT", {"bank_id": sender_bank}),
            )
            entities.setdefault(
                receiver_id,
                Entity(receiver_id, "ACCOUNT", {"bank_id": receiver_bank}),
            )
            row_identity = "|".join(
                [case_id, str(index), sender_id, receiver_id, _first(row, "Timestamp", "timestamp")]
            )
            event_id = "TX-" + hashlib.sha256(row_identity.encode("utf-8")).hexdigest()[:16]
            amount = _amount(
                _first(row, "Amount Paid", "amount_paid", "amount", default="0")
            )
            events[event_id] = AtomicEvent(
                event_id=event_id,
                action_type="TRANSFER",
                roles={"sender": sender_id, "receiver": receiver_id},
                timestamp=_timestamp(_first(row, "Timestamp", "timestamp")),
                attributes={
                    "amount": amount,
                    "paid_currency": _first(row, "Payment Currency", "payment_currency"),
                    "received_amount": _amount(
                        _first(row, "Amount Received", "amount_received", default="0")
                    ),
                    "received_currency": _first(
                        row, "Receiving Currency", "receiving_currency"
                    ),
                    "channel": _first(row, "Payment Format", "payment_format", "channel"),
                },
                source_record_id=str(index),
            )
            raw_label = _first(row, "Is Laundering", "is_laundering", default="0")
            try:
                labels[event_id] = int(float(raw_label))
            except ValueError:
                labels[event_id] = 0

    case = CaseGraph(
        case_id=case_id,
        entities=entities,
        atomic_events=events,
        metadata={
            "dataset": "IBM synthetic AML transactions",
            "source_file": str(source.resolve()),
            "evaluation_labels": labels,
        },
    )
    build_atomic_relations(case)
    case.validate()
    return case

