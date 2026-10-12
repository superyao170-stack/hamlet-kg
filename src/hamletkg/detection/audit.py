"""Read-only quality audit for transaction-level detection experiments.

This module uses no model, features, network service or outcome inference.
Evaluation labels are read only to measure their coverage and validity. An
audit describes the loaded CaseGraph: source rows collapsed by a loader cannot
be recovered here, and a file digest establishes identity, not authenticity.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from ..models import CaseGraph, iso_time, parse_time


_DATASET_METADATA_FIELDS = (
    "dataset", "dataset_version", "dataset_release", "dataset_license",
    "license", "split_name", "labels_are_features",
)
_IGNORED_ATTRIBUTE_FIELDS = frozenset({
    "id", "eventid", "transactionid", "recordid", "sourcerecordid",
    "sourcerow", "sourcerownumber", "sourceline", "sourcelinenumber",
    "rowid", "rownumber", "rowindex", "linenumber",
    "sourcefile", "sourcepath", "label", "labels", "evaluationlabel",
    "evaluationlabels", "islaundering", "isfraud", "fraudstep", "groundtruth",
})
_AMOUNT_FIELDS = ("amount", "received_amount")
_MAX_EXAMPLES = 20


def _field_name(value: object) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


def _missing(value: object) -> bool:
    return value is None or isinstance(value, str) and not value.strip()


def _category(value: object) -> tuple[str | None, str]:
    if _missing(value):
        return None, "missing"
    if not isinstance(value, str):
        return None, "invalid"
    return value.strip().upper(), "valid"


def _number(value: object) -> tuple[Decimal | None, str]:
    if _missing(value):
        return None, "missing"
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        return None, "invalid"
    try:
        numeric = Decimal(str(value).strip())
    except (InvalidOperation, ValueError, TypeError):
        return None, "invalid"
    if not numeric.is_finite():
        return None, "non_finite"
    if numeric.is_zero():
        return numeric, "zero"
    if numeric < 0:
        return numeric, "negative"
    return numeric, "positive"


def _decimal_text(value: Decimal) -> str:
    # Decimal.normalize() can round under the ambient decimal context. The
    # tuple representation preserves arbitrary precision without such rounding.
    if value.is_zero():
        return "0"
    sign, digits, exponent = value.as_tuple()
    digits = list(digits)
    while digits[-1] == 0:
        digits.pop()
        exponent += 1
    return ("-" if sign else "") + "".join(map(str, digits)) + "e" + str(exponent)


def _normalized_value(value: Any, *, field: str = "", ignore_fields: bool = False) -> Any:
    """Create a JSON-safe, type-tagged value for a business-field key."""
    field = field.strip().casefold()
    if field in _AMOUNT_FIELDS:
        numeric, status = _number(value)
        if numeric is not None:
            return ["number", _decimal_text(numeric)]
        if status == "non_finite":
            return ["non_finite", str(value).strip().casefold()]
    if isinstance(value, Mapping):
        return ["object", sorted(
            (str(key).strip().casefold(), _normalized_value(item, field=str(key), ignore_fields=ignore_fields))
            for key, item in value.items()
            if not ignore_fields or _field_name(key) not in _IGNORED_ATTRIBUTE_FIELDS
        )]
    if isinstance(value, (list, tuple)):
        return ["array", [_normalized_value(item, ignore_fields=ignore_fields) for item in value]]
    if isinstance(value, str):
        normalized = value.strip()
        if field in {"currency", "paid_currency", "received_currency", "channel", "payment_format"}:
            normalized = normalized.upper()
        return ["string", normalized]
    if value is None or isinstance(value, bool):
        return [type(value).__name__, value]
    if isinstance(value, (int, float, Decimal)):
        numeric, status = _number(value)
        return ["number", _decimal_text(numeric)] if numeric is not None else [status, str(value)]
    if isinstance(value, datetime):
        return ["datetime", iso_time(value)]
    # Unsupported objects are reported as a type, never serialized with repr
    # (which might disclose arbitrary data or have nondeterministic addresses).
    return ["unsupported", type(value).__name__]


def _comparable(value: Any, *, ignore_fields: bool = False) -> bool:
    if isinstance(value, Mapping):
        return all(_comparable(item, ignore_fields=ignore_fields) for key, item in value.items()
                   if not ignore_fields or _field_name(key) not in _IGNORED_ATTRIBUTE_FIELDS)
    if isinstance(value, (list, tuple)):
        return all(_comparable(item, ignore_fields=ignore_fields) for item in value)
    return value is None or isinstance(value, (bool, int, float, str, Decimal, datetime))


def _time(value: object) -> tuple[datetime | None, str]:
    if _missing(value):
        return None, "missing"
    if not isinstance(value, (str, datetime)):
        return None, "invalid"
    try:
        return parse_time(value), "valid"
    except (ValueError, TypeError, OverflowError, AttributeError):
        return None, "invalid"


def _source(path: Path | str | None, metadata: Mapping) -> dict:
    allowed_metadata = {}
    for field in _DATASET_METADATA_FIELDS:
        value = metadata.get(field)
        if isinstance(value, (str, int, float, bool)):
            if not isinstance(value, float) or _number(value)[1] != "non_finite":
                allowed_metadata[field] = value
    result = {
        "dataset_metadata": allowed_metadata, "filename": None,
        "sha256": None, "size_bytes": None, "digest_status": "not_provided",
        "authenticity": "unknown", "synthetic_or_real": "unknown",
        "injected_chain_membership": "unknown",
    }
    if path is not None:
        source = Path(path)
        result["filename"] = source.name
        digest = hashlib.sha256()
        size = 0
        try:
            with source.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    size += len(chunk)
                    digest.update(chunk)
        except OSError as exc:
            result["digest_status"] = "unreadable"
            result["digest_error"] = type(exc).__name__
        else:
            result.update(sha256=digest.hexdigest(), size_bytes=size, digest_status="computed")
    return result


def audit_detection_case(case: CaseGraph, *, source_path: Path | str | None = None) -> dict:
    """Audit all loaded events, preserving duplicates, labels and input order.

    All TRANSFER events are candidates, including incomplete or invalid ones.
    Missing and invalid labels remain unknown and are never converted to 0.
    Amount comparisons require two explicitly observed finite amounts and the
    same known currency; currency conversion rates and fees are not inferred.
    ``source_path`` is the only file read; metadata paths are never followed.
    """
    if not isinstance(case, CaseGraph):
        raise TypeError("case must be a CaseGraph")
    events = list(case.atomic_events.values())
    transfers = [event for event in events if event.action_type == "TRANSFER"]
    total = len(transfers)
    metadata = case.metadata if isinstance(case.metadata, Mapping) else {}
    missing = Counter({field: 0 for field in (
        "sender", "receiver", "timestamp", "amount", "received_amount",
        "paid_currency", "received_currency", "channel",
    )})
    invalid = Counter({field: 0 for field in missing})
    amount_statuses = {field: Counter({status: 0 for status in (
        "missing", "invalid", "non_finite", "zero", "negative", "positive",
    )}) for field in _AMOUNT_FIELDS}
    amount_bounds: dict[str, list[Decimal]] = {field: [] for field in _AMOUNT_FIELDS}
    currencies = {field: Counter() for field in ("paid", "received")}
    currency_pairs = Counter()
    formats = Counter()
    timestamp_counts = Counter()
    input_times = []
    participants = set()
    unknown_participants = set()
    duplicate_keys: dict[str, list[str]] = {}
    invalid_examples = []
    comparisons = Counter({field: 0 for field in (
        "same_currency", "cross_currency", "unknown_currency",
        "same_currency_finite_pairs", "same_currency_equal_amounts",
        "same_currency_different_amounts", "unavailable_amount_comparisons",
    )})
    malformed_attributes = 0
    malformed_roles = 0
    uncheckable_duplicates = 0
    fallback_currency = Counter({"paid": 0, "received": 0})

    for event in transfers:
        roles = event.roles if isinstance(event.roles, Mapping) else {}
        attributes = event.attributes if isinstance(event.attributes, Mapping) else {}
        malformed_roles += not isinstance(event.roles, Mapping)
        malformed_attributes += not isinstance(event.attributes, Mapping)
        bad_fields = []
        for field in ("sender", "receiver"):
            value = roles.get(field)
            if _missing(value):
                missing[field] += 1
                bad_fields.append(field)
            elif not isinstance(value, str):
                invalid[field] += 1
                bad_fields.append(field)
            else:
                participants.add(value)
                if value not in case.entities:
                    unknown_participants.add(value)
        timestamp, time_status = _time(event.timestamp)
        if time_status == "valid":
            input_times.append(timestamp)
            timestamp_counts[iso_time(timestamp)] += 1
        else:
            (missing if time_status == "missing" else invalid)["timestamp"] += 1
            bad_fields.append("timestamp")
        numbers = {}
        for field in _AMOUNT_FIELDS:
            number, status = _number(attributes.get(field))
            amount_statuses[field][status] += 1
            numbers[field] = number
            if number is not None:
                amount_bounds[field].append(number)
            if status == "missing":
                missing[field] += 1
            elif status != "positive":
                invalid[field] += 1
            if status != "positive":
                bad_fields.append(field)
        effective_currencies = {}
        for side, field in (("paid", "paid_currency"), ("received", "received_currency")):
            raw_value = attributes.get(field)
            _, raw_status = _category(raw_value)
            if raw_status != "valid":
                (missing if raw_status == "missing" else invalid)[field] += 1
                bad_fields.append(field)
            value = raw_value
            if _missing(value):
                value = attributes.get("currency")
                if _category(value)[1] == "valid":
                    fallback_currency[side] += 1
            currency, _ = _category(value)
            effective_currencies[side] = currency
            if currency is not None:
                currencies[side][currency] += 1
        channel, channel_status = _category(attributes.get("channel"))
        if channel_status == "valid":
            formats[channel] += 1
        else:
            (missing if channel_status == "missing" else invalid)["channel"] += 1
            bad_fields.append("channel")
        paid, received = (effective_currencies[side] for side in ("paid", "received"))
        if paid is None or received is None:
            comparisons["unknown_currency"] += 1
        else:
            currency_pairs[f"{paid}->{received}"] += 1
            comparisons["same_currency" if paid == received else "cross_currency"] += 1
        if paid is not None and paid == received and all(value is not None for value in numbers.values()):
            comparisons["same_currency_finite_pairs"] += 1
            equal = numbers["amount"] == numbers["received_amount"]
            comparisons["same_currency_equal_amounts" if equal else "same_currency_different_amounts"] += 1
        else:
            comparisons["unavailable_amount_comparisons"] += 1
        key = json.dumps([
            iso_time(timestamp) if timestamp is not None else _normalized_value(event.timestamp),
            _normalized_value(roles), _normalized_value(attributes, ignore_fields=True),
        ], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if (timestamp is not None and isinstance(event.roles, Mapping)
                and isinstance(event.attributes, Mapping) and _comparable(roles)
                and _comparable(attributes, ignore_fields=True)):
            digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
            duplicate_keys.setdefault(digest, []).append(event.event_id)
        else:
            uncheckable_duplicates += 1
        if bad_fields and len(invalid_examples) < _MAX_EXAMPLES:
            invalid_examples.append({"event_id": event.event_id, "fields": sorted(set(bad_fields))})

    duplicate_groups = [ids for ids in duplicate_keys.values() if len(ids) > 1]
    duplicate_transactions = sum(map(len, duplicate_groups))
    ties = [count for count in timestamp_counts.values() if count > 1]
    tied_transactions = sum(ties)
    all_times = [timestamp for event in events if (timestamp := _time(event.timestamp)[0]) is not None]
    adjacent_inversions = sum(left > right for left, right in zip(input_times, input_times[1:]))
    label_mapping = metadata.get("evaluation_labels", {})
    valid_label_mapping = isinstance(label_mapping, Mapping)
    label_mapping = label_mapping if valid_label_mapping else {}
    labeled = positives = invalid_labels = missing_labels = 0
    for event in transfers:
        label = label_mapping.get(event.event_id)
        if _missing(label):
            missing_labels += 1
        elif type(label) is int and label in (0, 1):
            labeled += 1
            positives += label
        else:
            invalid_labels += 1
    transfer_ids = {event.event_id for event in transfers}
    unmatched_labels = sum(key not in transfer_ids for key in label_mapping)
    entity_labels = sum(key in case.entities and key not in transfer_ids for key in label_mapping)
    source = _source(source_path, metadata)
    risks = []
    for field in missing:
        if missing[field]:
            risks.append({"code": "missing_transfer_field", "field": field, "count": missing[field]})
        if invalid[field]:
            risks.append({"code": "invalid_transfer_field", "field": field, "count": invalid[field]})
    for code, count in (
        ("business_field_duplicates", duplicate_transactions),
        ("simultaneous_transactions_have_no_observed_internal_order", tied_transactions),
        ("transaction_labels_unknown", total - labeled),
        ("invalid_evaluation_labels", invalid_labels),
        ("labels_outside_transaction_grain", unmatched_labels),
        ("unknown_transfer_participants", len(unknown_participants)),
        ("malformed_attributes", malformed_attributes),
        ("malformed_roles", malformed_roles),
        ("uncheckable_business_duplicate_comparison", uncheckable_duplicates),
        ("input_order_time_inversions", adjacent_inversions),
    ):
        if count:
            risks.append({"code": code, "count": count})
    if not valid_label_mapping:
        risks.append({"code": "invalid_evaluation_label_mapping", "count": 1})
    if source["digest_status"] == "unreadable":
        risks.append({"code": "source_file_unreadable", "count": 1})

    return {
        "schema_version": 1, "case_id": case.case_id,
        "counts": {
            "atomic_events": len(events), "transfer_candidates": total,
            "account_entities": sum(str(entity.entity_type).upper() == "ACCOUNT" for entity in case.entities.values()),
            "participating_accounts": len(participants),
            "unknown_participant_ids": len(unknown_participants),
        },
        "time": {
            "timezone": "UTC", "scope": "TRANSFER candidates",
            "start": iso_time(min(input_times)) if input_times else None,
            "end": iso_time(max(input_times)) if input_times else None,
            "all_atomic_start": iso_time(min(all_times)) if all_times else None,
            "all_atomic_end": iso_time(max(all_times)) if all_times else None,
            "all_atomic_invalid_or_missing_timestamps": len(events) - len(all_times),
            "valid_timestamps": len(input_times),
            "input_order_nondecreasing": all(left <= right for left, right in zip(input_times, input_times[1:])),
            "input_order_complete": len(input_times) == total,
            "adjacent_time_inversions": adjacent_inversions,
            "distinct_timestamp_groups": len(timestamp_counts),
            "same_timestamp_groups": len(ties),
            "transactions_in_same_timestamp_groups": tied_transactions,
            "same_timestamp_transaction_ratio": tied_transactions / total if total else None,
            "max_timestamp_group_size": max(timestamp_counts.values(), default=0),
        },
        "currencies": {
            "paid": dict(sorted(currencies["paid"].items())),
            "received": dict(sorted(currencies["received"].items())),
            "pairs": dict(sorted(currency_pairs.items())),
            "legacy_currency_fallbacks": dict(fallback_currency),
            "amounts_are_not_aggregated_across_currencies": True,
        },
        "payment_formats": dict(sorted(formats.items())),
        "missing_values": dict(missing), "invalid_values": dict(invalid),
        "invalid_field_examples": invalid_examples,
        "amounts": {
            **{field: {
                **dict(amount_statuses[field]),
                "finite_min": str(min(amount_bounds[field])) if amount_bounds[field] else None,
                "finite_max": str(max(amount_bounds[field])) if amount_bounds[field] else None,
            } for field in _AMOUNT_FIELDS},
            "paid_received_correspondence": dict(comparisons),
            "comparison_policy": "Only explicit finite amount pairs with the same known currency are compared; fees and FX rates are not inferred.",
        },
        "duplicates": {
            "business_field_groups": len(duplicate_groups),
            "transactions_in_groups": duplicate_transactions,
            "extra_transactions": sum(len(ids) - 1 for ids in duplicate_groups),
            "uncheckable_transactions": uncheckable_duplicates,
            "transaction_ratio": duplicate_transactions / total if total else None,
            "examples": [{"event_ids": ids[:_MAX_EXAMPLES], "group_size": len(ids)} for ids in duplicate_groups[:_MAX_EXAMPLES]],
            "retained": True,
            "comparison_policy": "UTC timestamp, normalized participant roles and business attributes; event/record IDs, source row fields and label fields are ignored.",
        },
        "labels": {
            "source": "case.metadata.evaluation_labels_only",
            "scope": "transaction-level TRANSFER labels",
            "mapping_valid": valid_label_mapping,
            "labeled": labeled, "unlabeled": total - labeled,
            "missing": missing_labels, "invalid": invalid_labels,
            "positive": positives, "negative": labeled - positives,
            "coverage": labeled / total if total else None,
            "positive_rate_among_labeled": positives / labeled if labeled else None,
            "unmatched_entries": unmatched_labels, "entity_level_entries": entity_labels,
            "unlabeled_is_negative": False, "used_as_features": False,
        },
        "source": source, "risks": risks,
        "limitations": [
            "The audit covers loaded CaseGraph events; duplicate source IDs already collapsed by ingestion cannot be recovered.",
            "Dataset metadata and SHA256 do not verify whether data is synthetic or real, or whether it represents operational bank activity.",
            "Transaction labels do not establish membership in an injected laundering chain; chain membership remains unknown.",
            "Same-time transfers have no observed internal order; missing timestamps cannot establish chronological completeness.",
            "Missing labels remain unknown; no labels, feature values, FX rates or received amounts are manufactured.",
        ],
    }
