"""Validated policy for four fixed causal detection operators.

This schema configures executable operators; it is intentionally distinct from
the project's general risk-pattern DSL. No operator is silently approximated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any


SUPPORTED_OPERATORS = ("fan_in", "fan_out", "pass_through", "relay")
_COMMON_FIELDS = {"rule_id", "operator", "window_seconds"}
_OPERATOR_FIELDS = {
    "fan_in": {"min_counterparties"},
    "fan_out": {"min_counterparties"},
    "pass_through": {"min_incoming", "ratio_min", "ratio_max"},
    "relay": {"ratio_min", "ratio_max"},
}


def _string(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a nonempty string without surrounding whitespace")


def _positive(value: Any, name: str) -> None:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} must be finite and positive")
    try:
        valid = math.isfinite(value) and value > 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be finite and positive")


def _count(value: Any, name: str, minimum: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _schema(value: Any, required: set[str], allowed: set[str], name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    missing = required - value.keys()
    extra = value.keys() - allowed
    if missing:
        raise ValueError(f"{name} is missing required fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"{name} contains unsupported fields: {sorted(map(str, extra))}")


@dataclass(frozen=True)
class DetectionRule:
    rule_id: str
    operator: str
    window_seconds: float
    min_counterparties: int = 3
    min_incoming: int = 1
    ratio_min: float = 0.8
    ratio_max: float = 1.2

    def __post_init__(self) -> None:
        _string(self.rule_id, "rule_id")
        if not isinstance(self.operator, str) or self.operator not in SUPPORTED_OPERATORS:
            raise ValueError(f"operator must be one of {SUPPORTED_OPERATORS}")
        _positive(self.window_seconds, "window_seconds")
        _count(self.min_counterparties, "min_counterparties", 2)
        _count(self.min_incoming, "min_incoming", 1)
        _positive(self.ratio_min, "ratio_min")
        _positive(self.ratio_max, "ratio_max")
        if self.ratio_min > self.ratio_max:
            raise ValueError("ratio_min cannot exceed ratio_max")
        if self.operator in ("fan_in", "fan_out"):
            if self.min_incoming != 1 or self.ratio_min != .8 or self.ratio_max != 1.2:
                raise ValueError(f"{self.operator} does not support min_incoming or amount-ratio parameters")
        elif self.min_counterparties != 3:
            raise ValueError(f"{self.operator} does not support min_counterparties")
        if self.operator == "relay" and self.min_incoming != 1:
            raise ValueError("relay has a fixed three-transaction path and does not support min_incoming")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DetectionRule":
        _schema(value, _COMMON_FIELDS, _COMMON_FIELDS | set().union(*_OPERATOR_FIELDS.values()), "detection rule")
        operator = value["operator"]
        if not isinstance(operator, str) or operator not in SUPPORTED_OPERATORS:
            raise ValueError(f"operator must be one of {SUPPORTED_OPERATORS}")
        _schema(value, _COMMON_FIELDS, _COMMON_FIELDS | _OPERATOR_FIELDS[operator], f"{operator} rule")
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        result = {"rule_id": self.rule_id, "operator": self.operator,
                  "window_seconds": self.window_seconds}
        for name in sorted(_OPERATOR_FIELDS[self.operator]):
            result[name] = getattr(self, name)
        return result


@dataclass(frozen=True)
class DetectionPolicy:
    policy_id: str
    version: str
    rules: tuple[DetectionRule, ...]

    def __post_init__(self) -> None:
        _string(self.policy_id, "policy_id")
        _string(self.version, "version")
        if not isinstance(self.rules, (tuple, list)) or not self.rules:
            raise ValueError("rules must be a nonempty sequence of DetectionRule objects")
        rules = tuple(self.rules)
        if any(not isinstance(rule, DetectionRule) for rule in rules):
            raise ValueError("rules must contain only DetectionRule objects")
        if len({rule.rule_id for rule in rules}) != len(rules):
            raise ValueError("rule_ids must be unique within a detection policy")
        object.__setattr__(self, "rules", rules)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DetectionPolicy":
        fields = {"policy_id", "version", "rules"}
        _schema(value, fields, fields, "detection policy")
        if not isinstance(value["rules"], list) or not value["rules"]:
            raise ValueError("rules must be a nonempty JSON array")
        return cls(value["policy_id"], value["version"],
                   tuple(DetectionRule.from_dict(rule) for rule in value["rules"]))

    def to_dict(self) -> dict[str, Any]:
        return {"policy_id": self.policy_id, "version": self.version,
                "rules": [rule.to_dict() for rule in self.rules]}


def validate_policy_for_stream(policy: DetectionPolicy, *, window_seconds: float,
                               max_history: int) -> None:
    """Reject requests exceeding the history actually retained by the engine."""
    if not isinstance(policy, DetectionPolicy):
        raise ValueError("policy must be a DetectionPolicy")
    _positive(window_seconds, "stream window_seconds")
    _count(max_history, "stream max_history", 1)
    for rule in policy.rules:
        if rule.window_seconds > window_seconds:
            raise ValueError(f"rule {rule.rule_id} window exceeds stream window_seconds")
        if rule.operator in ("fan_in", "fan_out") and rule.min_counterparties > max_history + 1:
            raise ValueError(f"rule {rule.rule_id} min_counterparties exceeds max_history + current transaction")
        if rule.operator == "pass_through" and rule.min_incoming > max_history:
            raise ValueError(f"rule {rule.rule_id} min_incoming exceeds max_history historical transactions")
