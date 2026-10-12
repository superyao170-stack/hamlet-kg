"""Adapter for the public IBM synthetic AML transaction CSV schema."""

from __future__ import annotations

import csv
import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

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
        amount = float(value.replace(",", ""))
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"invalid transaction amount: {value!r}") from exc
    if not math.isfinite(amount) or amount <= 0:
        raise ValueError(f"transaction amount must be finite and positive: {value!r}")
    return amount


def _normalized_header(header: list[str]) -> list[str]:
    """Preserve both Account columns in the original IBM CSV schema."""

    normalized: list[str] = []
    account_count = 0
    for raw_name in header:
        name = raw_name.strip()
        if name == "Account":
            account_count += 1
            name = "Account" if account_count == 1 else f"Account.{account_count - 1}"
        if not name or name in normalized:
            raise ValueError(f"empty or ambiguous duplicate CSV column: {raw_name!r}")
        normalized.append(name)
    return normalized


def _required(row: dict[str, str], row_number: int, field: str, *aliases: str) -> str:
    value = _first(row, field, *aliases)
    if not value:
        raise ValueError(f"row {row_number} is missing {field}")
    return value


def _label(value: str, row_number: int) -> int:
    try:
        numeric = float(value)
    except ValueError as exc:
        raise ValueError(f"row {row_number} has invalid laundering label: {value!r}") from exc
    if not math.isfinite(numeric) or numeric not in (0.0, 1.0):
        raise ValueError(f"row {row_number} laundering label must be 0 or 1: {value!r}")
    return int(numeric)


def _validate_header(columns: list[str]) -> None:
    required_aliases = (
        ("Timestamp", "timestamp"),
        ("From Bank", "from_bank", "sender_bank"),
        ("From Account", "Account", "from_account", "sender_account"),
        ("To Bank", "to_bank", "receiver_bank"),
        ("To Account", "Account.1", "to_account", "receiver_account"),
        ("Amount Paid", "amount_paid", "amount"),
        ("Amount Received", "amount_received", "received_amount"),
        ("Payment Currency", "payment_currency", "paid_currency"),
        ("Receiving Currency", "receiving_currency", "received_currency"),
        ("Payment Format", "payment_format", "channel"),
    )
    for aliases in required_aliases:
        if not set(aliases).intersection(columns):
            raise ValueError(f"transaction CSV is missing required column: {aliases[0]}")


def _iter_transaction_rows(
    path: str | Path,
    *,
    case_id: str,
    limit: int | None = None,
) -> Iterator[tuple[AtomicEvent, int | None, str, str]]:
    """Validate and stream original rows while keeping labels separate.

    The identity uses the original, one-based CSV row index, including when a
    downstream caller selects a time interval. Duplicate business rows remain
    distinct source records. No entity dictionary or whole graph is built.
    """

    if not case_id:
        raise ValueError("case_id must not be empty")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be a positive integer or None")
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise ValueError("transaction CSV is empty")
        columns = _normalized_header(header)
        _validate_header(columns)
        for index, values in enumerate(reader, start=1):
            if limit is not None and index > limit:
                break
            row_number = index + 1
            if len(values) != len(columns):
                raise ValueError(f"row {row_number} has {len(values)} columns; expected {len(columns)}")
            row = dict(zip(columns, values))
            sender_bank = _required(row, row_number, "From Bank", "from_bank", "sender_bank")
            sender_account = _required(
                row, row_number, "From Account", "Account", "from_account", "sender_account"
            )
            receiver_bank = _required(row, row_number, "To Bank", "to_bank", "receiver_bank")
            receiver_account = _required(
                row, row_number,
                "To Account",
                "Account.1",
                "to_account",
                "receiver_account",
            )
            raw_timestamp = _required(row, row_number, "Timestamp", "timestamp")
            try:
                timestamp = _timestamp(raw_timestamp)
                amount = _amount(_required(row, row_number, "Amount Paid", "amount_paid", "amount"))
                received_amount = _amount(
                    _required(row, row_number, "Amount Received", "amount_received", "received_amount")
                )
            except ValueError as exc:
                raise ValueError(f"row {row_number}: {exc}") from exc
            paid_currency = _required(row, row_number, "Payment Currency", "payment_currency", "paid_currency")
            received_currency = _required(
                row, row_number, "Receiving Currency", "receiving_currency", "received_currency"
            )
            channel = _required(row, row_number, "Payment Format", "payment_format", "channel")
            sender_id = f"ACCOUNT:{sender_bank}:{sender_account}"
            receiver_id = f"ACCOUNT:{receiver_bank}:{receiver_account}"
            row_identity = "|".join(
                [case_id, str(index), sender_id, receiver_id, raw_timestamp]
            )
            event_id = "TX-" + hashlib.sha256(row_identity.encode("utf-8")).hexdigest()[:16]
            event = AtomicEvent(
                event_id=event_id,
                action_type="TRANSFER",
                roles={"sender": sender_id, "receiver": receiver_id},
                timestamp=timestamp,
                attributes={
                    "amount": amount,
                    "paid_currency": paid_currency,
                    "received_amount": received_amount,
                    "received_currency": received_currency,
                    "channel": channel,
                },
                source_record_id=str(index),
            )
            raw_label = _first(row, "Is Laundering", "is_laundering")
            yield event, _label(raw_label, row_number) if raw_label else None, sender_bank, receiver_bank


def iter_transaction_csv(
    path: str | Path,
    *,
    case_id: str,
    limit: int | None = None,
) -> Iterator[tuple[AtomicEvent, int | None]]:
    """Yield validated atomic transfers and separate optional evaluation labels."""

    for event, label, _, _ in _iter_transaction_rows(path, case_id=case_id, limit=limit):
        yield event, label


def case_from_transaction_csv(
    path: str | Path,
    *,
    case_id: str,
    limit: int | None = None,
    build_relations: bool = True,
) -> CaseGraph:
    """Convert validated rows; evaluation labels remain outside attributes."""

    source = Path(path)
    entities: dict[str, Entity] = {}
    events: dict[str, AtomicEvent] = {}
    labels: dict[str, int] = {}
    for event, label, sender_bank, receiver_bank in _iter_transaction_rows(
        source, case_id=case_id, limit=limit
    ):
        for role, bank_id in (("sender", sender_bank), ("receiver", receiver_bank)):
            entity_id = event.roles[role]
            entities.setdefault(entity_id, Entity(entity_id, "ACCOUNT", {"bank_id": bank_id}))
        events[event.event_id] = event
        if label is not None:
            labels[event.event_id] = label

    case = CaseGraph(
        case_id=case_id,
        entities=entities,
        atomic_events=events,
        metadata={
            "dataset": "IBM synthetic AML transactions",
            "source_file": str(source.resolve()),
            "evaluation_labels": labels,
            "labels_are_features": False,
        },
    )
    if build_relations:
        build_atomic_relations(case)
    case.validate()
    return case
