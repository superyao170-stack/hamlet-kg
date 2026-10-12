"""Causal, bounded transaction histories and role-preserving event instances.

The current transfer and strictly earlier transfers are the only inputs to a
sample. Transfers sharing a timestamp are scored before any of them enters
history. Evaluation metadata, account identifiers and simulation labels are
never encoded in numerical features. Instances are hypotheses, not labels.
"""

from __future__ import annotations

import hashlib
import heapq
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from itertools import groupby
from numbers import Real
from typing import Iterable, Iterator

from ..models import AtomicEvent, CaseGraph
from .policy import DetectionPolicy, validate_policy_for_stream


BASE_FEATURE_NAMES = (
    "log_amount", "log_received_amount", "same_currency", "self_transfer",
    "hour_sin", "hour_cos",
    "sender_in_log_count", "sender_out_log_count",
    "receiver_in_log_count", "receiver_out_log_count",
    "sender_in_log_amount_total", "sender_out_log_amount_total",
    "receiver_in_log_amount_total", "receiver_out_log_amount_total",
    "sender_in_log_unique_counterparties", "sender_out_log_unique_counterparties",
    "receiver_in_log_unique_counterparties", "receiver_out_log_unique_counterparties",
    "sender_in_log_seconds_since_last", "sender_out_log_seconds_since_last",
    "receiver_in_log_seconds_since_last", "receiver_out_log_seconds_since_last",
    "sender_out_amount_log_deviation", "receiver_in_amount_log_deviation",
)
MOTIF_TYPES = ("fan_in", "fan_out", "pass_through", "relay")
MEMBER_ROLES = ("incoming", "outgoing", "relay_start", "relay_middle", "relay_end")


@dataclass(frozen=True)
class StreamConfig:
    window_seconds: float = 86400
    max_history: int = 32
    min_fan: int = 3
    ratio_min: float = 0.8
    ratio_max: float = 1.2

    def __post_init__(self) -> None:
        if not isinstance(self.window_seconds, Real) or isinstance(self.window_seconds, bool) or not math.isfinite(self.window_seconds) or self.window_seconds <= 0:
            raise ValueError("window_seconds must be finite and positive")
        for name, minimum in (("max_history", 1), ("min_fan", 2)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if (
            not isinstance(self.ratio_min, Real) or not isinstance(self.ratio_max, Real)
            or isinstance(self.ratio_min, bool) or isinstance(self.ratio_max, bool)
            or not math.isfinite(self.ratio_min) or not math.isfinite(self.ratio_max)
            or self.ratio_min <= 0 or self.ratio_min > self.ratio_max
        ):
            raise ValueError("amount ratio bounds must be finite with 0 < ratio_min <= ratio_max")


@dataclass(frozen=True)
class EventMember:
    event_id: str
    role: str
    features: tuple[float, ...]
    elapsed_seconds: float


@dataclass(frozen=True)
class EventInstance:
    instance_id: str
    motif_type: str
    anchor_id: str
    members: tuple[EventMember, ...]
    span_seconds: float
    amount_ratio: float
    rule_id: str | None = None


@dataclass(frozen=True)
class TransactionSample:
    event_id: str
    timestamp: datetime
    features: tuple[float, ...]
    events: tuple[EventInstance, ...]


@dataclass(frozen=True)
class _Transfer:
    event: AtomicEvent
    sender: str
    receiver: str
    amount: float
    received_amount: float
    paid_currency: str
    received_currency: str
    features: tuple[float, ...] = ()


def _positive_amount(value: object, event_id: str, field: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"transfer {event_id} has invalid {field}") from exc
    if isinstance(value, bool) or not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"transfer {event_id} {field} must be finite and positive")
    return numeric


def _transfer(event: AtomicEvent) -> _Transfer:
    sender = event.roles.get("sender", "")
    receiver = event.roles.get("receiver", "")
    if not sender or not receiver:
        raise ValueError(f"transfer {event.event_id} requires sender and receiver")
    attributes = event.attributes
    paid_currency = str(attributes.get("paid_currency") or attributes.get("currency") or "").strip().upper()
    received_currency = str(attributes.get("received_currency") or attributes.get("currency") or "").strip().upper()
    if not paid_currency or not received_currency:
        raise ValueError(f"transfer {event.event_id} requires paid and received currencies")
    amount = _positive_amount(attributes.get("amount"), event.event_id, "amount")
    received = attributes.get("received_amount")
    if received is None and paid_currency == received_currency:
        received = amount
    return _Transfer(
        event, sender, receiver, amount,
        _positive_amount(received, event.event_id, "received_amount"),
        paid_currency, received_currency,
    )


class _History:
    def __init__(self, config: StreamConfig) -> None:
        self.config = config
        self.incoming: dict[tuple[str, str], deque[_Transfer]] = defaultdict(deque)
        self.outgoing: dict[tuple[str, str], deque[_Transfer]] = defaultdict(deque)
        # One scheduled entry per account/currency/direction bucket. A bucket
        # whose oldest transfer changes keeps its earlier wakeup; at that
        # wakeup it is trimmed and, if still active, scheduled once again.
        self._expiry_heap: list[tuple[datetime, int, bool, tuple[str, str]]] = []
        self._scheduled: set[tuple[bool, tuple[str, str]]] = set()
        self._schedule_sequence = 0

    def _schedule(self, key: tuple[str, str], *, incoming: bool) -> None:
        identity = (incoming, key)
        if identity in self._scheduled:
            return
        mapping = self.incoming if incoming else self.outgoing
        self._schedule_sequence += 1
        heapq.heappush(self._expiry_heap, (mapping[key][0].event.timestamp,
                                         self._schedule_sequence, incoming, key))
        self._scheduled.add(identity)

    def expire(self, timestamp: datetime) -> None:
        """Remove inactive buckets even when their accounts are never queried."""
        while self._expiry_heap and (timestamp - self._expiry_heap[0][0]).total_seconds() > self.config.window_seconds:
            _, _, incoming, key = heapq.heappop(self._expiry_heap)
            self._scheduled.remove((incoming, key))
            mapping = self.incoming if incoming else self.outgoing
            history = mapping.get(key)
            if history is None:
                continue
            while history and (timestamp - history[0].event.timestamp).total_seconds() > self.config.window_seconds:
                history.popleft()
            if history:
                self._schedule(key, incoming=incoming)
            else:
                del mapping[key]

    def get(self, account: str, currency: str, timestamp: datetime, *, incoming: bool) -> list[_Transfer]:
        mapping = self.incoming if incoming else self.outgoing
        key = (account, currency)
        history = mapping.get(key)
        if history is None:
            return []
        while history and (timestamp - history[0].event.timestamp).total_seconds() > self.config.window_seconds:
            history.popleft()
        if not history:
            del mapping[key]
            return []
        return list(history)

    def add(self, transfer: _Transfer) -> None:
        for incoming, mapping, key in (
            (True, self.incoming, (transfer.receiver, transfer.received_currency)),
            (False, self.outgoing, (transfer.sender, transfer.paid_currency)),
        ):
            history = mapping[key]
            while history and (transfer.event.timestamp - history[0].event.timestamp).total_seconds() > self.config.window_seconds:
                history.popleft()
            history.append(transfer)
            while len(history) > self.config.max_history:
                history.popleft()
            self._schedule(key, incoming=incoming)


def _log_total(values: list[float]) -> float:
    if not values:
        return 0.0
    maximum = max(values)
    if maximum < 1e150:
        return math.log1p(math.fsum(values))
    # Avoid an infinite aggregate even when every individual amount is finite.
    return math.log(maximum) + math.log(math.fsum(value / maximum for value in values))


def _mean(values: list[float]) -> float:
    maximum = max(values)
    return maximum * (math.fsum(value / maximum for value in values) / len(values))


def _features(transfer: _Transfer, groups: tuple[list[_Transfer], ...], config: StreamConfig) -> tuple[float, ...]:
    timestamp = transfer.event.timestamp
    hour = timestamp.hour + timestamp.minute / 60 + timestamp.second / 3600
    angle = 2 * math.pi * hour / 24
    incoming_flags = (True, False, True, False)
    amounts = tuple(
        [item.received_amount if incoming else item.amount for item in group]
        for group, incoming in zip(groups, incoming_flags)
    )
    values = [
        math.log1p(transfer.amount), math.log1p(transfer.received_amount),
        float(transfer.paid_currency == transfer.received_currency),
        float(transfer.sender == transfer.receiver), math.sin(angle), math.cos(angle),
    ]
    values.extend(math.log1p(len(group)) for group in groups)
    values.extend(_log_total(group_amounts) for group_amounts in amounts)
    values.extend(
        math.log1p(len({item.sender if incoming else item.receiver for item in group}))
        for group, incoming in zip(groups, incoming_flags)
    )
    values.extend(
        math.log1p((timestamp - group[-1].event.timestamp).total_seconds() if group else config.window_seconds)
        for group in groups
    )
    for current, historical in ((transfer.amount, amounts[1]), (transfer.received_amount, amounts[2])):
        values.append(math.log1p(current) - math.log1p(_mean(historical)) if historical else 0.0)
    return tuple(values)


def _instance(motif: str, anchor: str, transfers: list[_Transfer], roles: list[str], ratio: float,
              *, rule_id: str | None = None) -> EventInstance:
    if not math.isfinite(ratio) or ratio < 0:
        raise ValueError(
            f"{motif} event for transfer {transfers[-1].event.event_id} "
            "amount_ratio must be finite and nonnegative"
        )
    end = transfers[-1].event.timestamp
    members = tuple(
        EventMember(item.event.event_id, role, item.features, (end - item.event.timestamp).total_seconds())
        for item, role in zip(transfers, roles)
    )
    identity = "|".join([motif, anchor, *(f"{item.event_id}:{item.role}" for item in members)])
    if rule_id is not None:
        identity += "|rule:" + rule_id
    return EventInstance(
        "EVENT-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16],
        motif, anchor, members, (end - transfers[0].event.timestamp).total_seconds(), float(ratio), rule_id,
    )


def _event_instances(current: _Transfer, groups: tuple[list[_Transfer], ...], history: _History) -> tuple[EventInstance, ...]:
    config = history.config
    incoming_sender, outgoing_sender, incoming_receiver, _ = groups
    instances: list[EventInstance] = []
    if current.sender == current.receiver:
        return ()
    fan_in = [item for item in incoming_receiver if item.sender != item.receiver] + [current]
    if len({item.sender for item in fan_in}) >= config.min_fan:
        instances.append(_instance("fan_in", current.receiver, fan_in, ["incoming"] * len(fan_in), 1.0))
    fan_out = [item for item in outgoing_sender if item.sender != item.receiver] + [current]
    if len({item.receiver for item in fan_out}) >= config.min_fan:
        instances.append(_instance("fan_out", current.sender, fan_out, ["outgoing"] * len(fan_out), 1.0))
    incoming = [item for item in incoming_sender if item.sender != item.receiver]
    if incoming:
        maximum = max(item.received_amount for item in incoming)
        ratio = (current.amount / maximum) / math.fsum(item.received_amount / maximum for item in incoming)
        if config.ratio_min <= ratio <= config.ratio_max:
            members = incoming + [current]
            instances.append(_instance("pass_through", current.sender, members, ["incoming"] * len(incoming) + ["outgoing"], ratio))
    relay_count = 0
    for middle in reversed(incoming):
        if not config.ratio_min <= current.amount / middle.received_amount <= config.ratio_max:
            continue
        starts = history.get(middle.sender, middle.paid_currency, current.event.timestamp, incoming=True)
        for start in reversed(starts):
            if start.event.timestamp >= middle.event.timestamp or start.sender == start.receiver:
                continue
            if len({start.sender, middle.sender, current.sender, current.receiver}) < 4:
                continue
            if not config.ratio_min <= middle.amount / start.received_amount <= config.ratio_max:
                continue
            ratio = (middle.amount / start.received_amount) * (current.amount / middle.received_amount)
            instances.append(_instance(
                "relay", current.sender, [start, middle, current],
                ["relay_start", "relay_middle", "relay_end"], ratio,
            ))
            relay_count += 1
            if relay_count >= config.max_history:
                return tuple(instances)
    return tuple(instances)


def _policy_event_instances(current: _Transfer, groups: tuple[list[_Transfer], ...],
                            history: _History, policy: DetectionPolicy) -> tuple[EventInstance, ...]:
    if current.sender == current.receiver:
        return ()
    instances: list[EventInstance] = []
    for rule in policy.rules:
        def in_window(items: list[_Transfer]) -> list[_Transfer]:
            return [item for item in items if item.sender != item.receiver
                    and (current.event.timestamp - item.event.timestamp).total_seconds() <= rule.window_seconds]

        incoming_sender, outgoing_sender, incoming_receiver, _ = tuple(in_window(items) for items in groups)
        if rule.operator in ("fan_in", "fan_out"):
            incoming = rule.operator == "fan_in"
            members = (incoming_receiver if incoming else outgoing_sender) + [current]
            counterparties = {item.sender if incoming else item.receiver for item in members}
            if len(counterparties) >= rule.min_counterparties:
                role = "incoming" if incoming else "outgoing"
                anchor = current.receiver if incoming else current.sender
                instances.append(_instance(rule.operator, anchor, members,
                                           [role] * len(members), 1.0, rule_id=rule.rule_id))
        elif rule.operator == "pass_through":
            if len(incoming_sender) < rule.min_incoming:
                continue
            maximum = max(item.received_amount for item in incoming_sender)
            ratio = (current.amount / maximum) / math.fsum(item.received_amount / maximum for item in incoming_sender)
            if rule.ratio_min <= ratio <= rule.ratio_max:
                instances.append(_instance(
                    rule.operator, current.sender, incoming_sender + [current],
                    ["incoming"] * len(incoming_sender) + ["outgoing"], ratio, rule_id=rule.rule_id,
                ))
        else:  # Validated relay: exactly three ordered transfers and four accounts.
            relay_count = 0
            for middle in reversed(incoming_sender):
                if not rule.ratio_min <= current.amount / middle.received_amount <= rule.ratio_max:
                    continue
                starts = in_window(history.get(middle.sender, middle.paid_currency,
                                               current.event.timestamp, incoming=True))
                for start in reversed(starts):
                    if start.event.timestamp >= middle.event.timestamp:
                        continue
                    if len({start.sender, middle.sender, current.sender, current.receiver}) < 4:
                        continue
                    if not rule.ratio_min <= middle.amount / start.received_amount <= rule.ratio_max:
                        continue
                    ratio = (middle.amount / start.received_amount) * (current.amount / middle.received_amount)
                    instances.append(_instance(
                        rule.operator, current.sender, [start, middle, current],
                        ["relay_start", "relay_middle", "relay_end"], ratio, rule_id=rule.rule_id,
                    ))
                    relay_count += 1
                    if relay_count >= history.config.max_history:
                        break
                if relay_count >= history.config.max_history:
                    break
    return tuple(instances)


def build_samples(case: CaseGraph, config: StreamConfig = StreamConfig(), *, excluded_ids: Iterable[str] = (),
                  policy: DetectionPolicy | None = None) -> list[TransactionSample]:
    """Build chronological samples; excluded evidence is removed before replay.

    History is bounded separately for each account/currency/direction, and
    inactive buckets expire globally. Output identifiers exist solely for joining
    labels and tracing evidence. No operation reads ``case.metadata``.
    """

    ordered = sorted(
        (item for item in case.atomic_events.values() if item.action_type == "TRANSFER"),
        key=lambda item: (item.timestamp, item.event_id),
    )
    return list(iter_samples(ordered, config, excluded_ids=excluded_ids, policy=policy))


def iter_samples(events: Iterable[AtomicEvent], config: StreamConfig = StreamConfig(), *,
                 excluded_ids: Iterable[str] = (), policy: DetectionPolicy | None = None) -> Iterator[TransactionSample]:
    """Yield samples from events sorted strictly by ``(timestamp, event_id)``.

    No label or case metadata is accessed. Same-timestamp transfers remain
    staged until their complete group has been scored, including when callers
    consume this iterator across batch boundaries. Memory depends on retained
    active history and the largest timestamp group, rather than dataset length.
    """
    if not isinstance(config, StreamConfig):
        raise TypeError("config must be a StreamConfig")
    if policy is not None:
        validate_policy_for_stream(policy, window_seconds=config.window_seconds,
                                   max_history=config.max_history)
    excluded = set(excluded_ids)

    def ordered_transfers() -> Iterator[AtomicEvent]:
        previous: tuple[datetime, str] | None = None
        for event in events:
            if not isinstance(event, AtomicEvent):
                raise TypeError("events must contain AtomicEvent objects")
            if not isinstance(event.event_id, str) or not event.event_id.strip():
                raise ValueError("event_id must be a nonempty string")
            current = (event.timestamp, event.event_id)
            if previous is not None and current <= previous:
                raise ValueError("events must be strictly sorted by timestamp and event_id")
            previous = current
            if event.action_type == "TRANSFER" and event.event_id not in excluded:
                yield event

    history = _History(config)
    for timestamp, simultaneous in groupby(ordered_transfers(), key=lambda item: item.timestamp):
        history.expire(timestamp)
        staged: list[_Transfer] = []
        for event in simultaneous:
            transfer = _transfer(event)
            groups = (
                history.get(transfer.sender, transfer.paid_currency, timestamp, incoming=True),
                history.get(transfer.sender, transfer.paid_currency, timestamp, incoming=False),
                history.get(transfer.receiver, transfer.received_currency, timestamp, incoming=True),
                history.get(transfer.receiver, transfer.received_currency, timestamp, incoming=False),
            )
            features = _features(transfer, groups, config)
            transfer = _Transfer(
                transfer.event, transfer.sender, transfer.receiver, transfer.amount,
                transfer.received_amount, transfer.paid_currency, transfer.received_currency, features,
            )
            instances = (_event_instances(transfer, groups, history) if policy is None
                         else _policy_event_instances(transfer, groups, history, policy))
            staged.append(transfer)
            yield TransactionSample(event.event_id, timestamp, features, instances)
        for transfer in staged:
            history.add(transfer)
