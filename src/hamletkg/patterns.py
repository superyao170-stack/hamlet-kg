"""Validation for the executable risk-pattern JSON DSL."""

from __future__ import annotations

import math
from dataclasses import replace

from .models import RiskPattern


SUPPORTED_CONSTRAINTS = {
    "ordered_within",
    "sum_ratio",
    "total_amount",
    "duration_max",
    "attribute_range",
    "role_overlap",
}

ACTION_ROLES = {
    "TRANSFER": {"sender", "receiver"},
    "CASH_WITHDRAWAL": {"account"},
    "CASH_DEPOSIT": {"account"},
}


class PatternValidationError(ValueError):
    pass


def validate_pattern_payload(value: dict) -> None:
    """Reject schema drift before a generated dictionary reaches the model."""

    top_required = {"pattern_id", "risk_type", "clauses", "constraints"}
    top_allowed = top_required | {"description", "version"}
    missing = sorted(top_required - set(value))
    extra = sorted(set(value) - top_allowed)
    if missing:
        raise PatternValidationError(f"pattern payload is missing keys: {missing}")
    if extra:
        raise PatternValidationError(f"pattern payload has unsupported keys: {extra}")
    if not isinstance(value.get("clauses"), list) or not value["clauses"]:
        raise PatternValidationError("pattern payload clauses must be a non-empty list")
    if not isinstance(value.get("constraints"), list):
        raise PatternValidationError("pattern payload constraints must be a list")
    clause_required = {"clause_id", "action_type", "anchor_role", "min_count"}
    clause_allowed = clause_required | {"max_count", "distinct_role"}
    for index, clause in enumerate(value["clauses"]):
        if not isinstance(clause, dict):
            raise PatternValidationError(f"clause {index} must be an object")
        missing = sorted(clause_required - set(clause))
        extra = sorted(set(clause) - clause_allowed)
        if missing or extra:
            raise PatternValidationError(
                f"clause {index} schema mismatch; missing={missing}, extra={extra}"
            )
    constraint_required = {"constraint_type", "parameters"}
    for index, constraint in enumerate(value["constraints"]):
        if not isinstance(constraint, dict):
            raise PatternValidationError(f"constraint {index} must be an object")
        missing = sorted(constraint_required - set(constraint))
        extra = sorted(set(constraint) - constraint_required)
        if missing or extra:
            raise PatternValidationError(
                f"constraint {index} schema mismatch; missing={missing}, extra={extra}"
            )


def validate_pattern(pattern: RiskPattern) -> None:
    clause_ids = {clause.clause_id for clause in pattern.clauses}
    for clause in pattern.clauses:
        if clause.min_count < 1:
            raise PatternValidationError(
                f"{clause.clause_id}: min_count must be at least 1"
            )
        if clause.max_count is not None and clause.max_count < clause.min_count:
            raise PatternValidationError(
                f"{clause.clause_id}: max_count cannot be smaller than min_count"
            )
        supported_roles = ACTION_ROLES.get(clause.action_type)
        if supported_roles is not None and clause.anchor_role not in supported_roles:
            raise PatternValidationError(
                f"{clause.clause_id}: role {clause.anchor_role} is invalid for "
                f"{clause.action_type}; expected one of {sorted(supported_roles)}"
            )
        if (
            supported_roles is not None
            and clause.distinct_role is not None
            and clause.distinct_role not in supported_roles
        ):
            raise PatternValidationError(
                f"{clause.clause_id}: distinct_role {clause.distinct_role} is invalid "
                f"for {clause.action_type}"
            )

    ordering_edges: list[tuple[str, str]] = []
    for constraint in pattern.constraints:
        kind = constraint.constraint_type
        params = constraint.parameters
        if kind not in SUPPORTED_CONSTRAINTS:
            raise PatternValidationError(f"unsupported constraint: {kind}")
        referenced: list[str] = []
        if kind == "ordered_within":
            _require_parameters(kind, params, {"earlier", "later", "max_seconds"})
            earlier, later = str(params["earlier"]), str(params["later"])
            referenced = [earlier, later]
            minimum = _finite_number(kind, params, "min_seconds", default=0.0)
            maximum = _finite_number(kind, params, "max_seconds")
            if minimum < 0:
                raise PatternValidationError("ordered_within.min_seconds must be non-negative")
            if maximum <= 0:
                raise PatternValidationError("ordered_within.max_seconds must be positive")
            if minimum > maximum:
                raise PatternValidationError(
                    "ordered_within.min_seconds cannot exceed max_seconds"
                )
            if earlier == later:
                raise PatternValidationError(
                    "ordered_within must reference two different clauses"
                )
            ordering_edges.append((earlier, later))
        elif kind == "sum_ratio":
            _require_parameters(kind, params, {"numerator", "denominator"})
            referenced = [str(params["numerator"]), str(params["denominator"])]
            minimum = _finite_number(kind, params, "minimum", default=0.0)
            maximum = _finite_number(kind, params, "maximum", default=None)
            if minimum < 0:
                raise PatternValidationError("sum_ratio.minimum must be non-negative")
            if maximum is not None and maximum < minimum:
                raise PatternValidationError(
                    "sum_ratio.maximum cannot be smaller than minimum"
                )
        elif kind == "total_amount":
            _require_parameters(kind, params, {"clause"})
            referenced = [str(params["clause"])]
            minimum = _finite_number(kind, params, "minimum", default=0.0)
            maximum = _finite_number(kind, params, "maximum", default=None)
            if maximum is not None and maximum < minimum:
                raise PatternValidationError(
                    "total_amount.maximum cannot be smaller than minimum"
                )
        elif kind == "duration_max":
            _require_parameters(kind, params, {"max_seconds"})
            if _finite_number(kind, params, "max_seconds") <= 0:
                raise PatternValidationError("duration_max.max_seconds must be positive")
        elif kind == "attribute_range":
            _require_parameters(kind, params, {"clause", "attribute"})
            referenced = [str(params["clause"])]
            minimum = _finite_number(kind, params, "minimum", default=None)
            maximum = _finite_number(kind, params, "maximum", default=None)
            if minimum is None and maximum is None:
                raise PatternValidationError(
                    "attribute_range requires minimum or maximum"
                )
            if minimum is not None and maximum is not None and maximum < minimum:
                raise PatternValidationError(
                    "attribute_range.maximum cannot be smaller than minimum"
                )
        elif kind == "role_overlap":
            _require_parameters(
                kind,
                params,
                {"left_clause", "left_role", "right_clause", "right_role"},
            )
            left_clause = str(params["left_clause"])
            right_clause = str(params["right_clause"])
            referenced = [left_clause, right_clause]
            minimum = _finite_number(kind, params, "minimum", default=1.0)
            if minimum < 1 or not minimum.is_integer():
                raise PatternValidationError(
                    "role_overlap.minimum must be a positive integer"
                )
            clauses_by_id = {clause.clause_id: clause for clause in pattern.clauses}
            for clause_id, role_key in (
                (left_clause, "left_role"),
                (right_clause, "right_role"),
            ):
                clause = clauses_by_id.get(clause_id)
                if clause is None:
                    continue
                supported_roles = ACTION_ROLES.get(clause.action_type)
                role = str(params[role_key])
                if supported_roles is not None and role not in supported_roles:
                    raise PatternValidationError(
                        f"role_overlap.{role_key} {role} is invalid for "
                        f"{clause.action_type}"
                    )
        unknown = set(referenced) - clause_ids
        if unknown:
            raise PatternValidationError(
                f"constraint {kind} references unknown clauses: {sorted(unknown)}"
            )
    _validate_acyclic_ordering(clause_ids, ordering_edges)


def _require_parameters(kind: str, params: dict, required: set[str]) -> None:
    missing = sorted(key for key in required if key not in params)
    if missing:
        raise PatternValidationError(
            f"constraint {kind} is missing parameters: {missing}"
        )


def _finite_number(
    kind: str,
    params: dict,
    key: str,
    *,
    default: float | None | object = ...,
) -> float | None:
    if key not in params:
        if default is ...:
            raise PatternValidationError(f"constraint {kind} is missing parameter {key}")
        return default  # type: ignore[return-value]
    try:
        value = float(params[key])
    except (TypeError, ValueError) as exc:
        raise PatternValidationError(
            f"constraint {kind}.{key} must be numeric"
        ) from exc
    if not math.isfinite(value):
        raise PatternValidationError(f"constraint {kind}.{key} must be finite")
    return value


def _validate_acyclic_ordering(
    clause_ids: set[str], edges: list[tuple[str, str]]
) -> None:
    adjacency = {clause_id: [] for clause_id in clause_ids}
    for source, target in edges:
        if source in adjacency and target in adjacency:
            adjacency[source].append(target)
    state = {clause_id: 0 for clause_id in clause_ids}

    def visit(clause_id: str) -> None:
        if state[clause_id] == 1:
            raise PatternValidationError(
                "ordered_within constraints contain a directed cycle"
            )
        if state[clause_id] == 2:
            return
        state[clause_id] = 1
        for neighbor in adjacency[clause_id]:
            visit(neighbor)
        state[clause_id] = 2

    for clause_id in sorted(clause_ids):
        visit(clause_id)


def validate_patterns(patterns: list[RiskPattern]) -> None:
    identifiers = [item.pattern_id for item in patterns]
    if len(identifiers) != len(set(identifiers)):
        raise PatternValidationError("pattern_id values must be unique")
    for pattern in patterns:
        validate_pattern(pattern)


def calibrate_pattern_reliability(
    pattern: RiskPattern,
    *,
    true_positive: int,
    false_positive: int,
    false_negative: int = 0,
) -> RiskPattern:
    """Attach validation-set reliability without treating LLM confidence as truth.

    ``calibrated_confidence`` is Laplace-smoothed precision.  Raw precision and
    recall are retained for reporting, and the original pattern remains
    unchanged because ``RiskPattern`` is immutable.
    """

    if min(true_positive, false_positive, false_negative) < 0:
        raise ValueError("calibration counts must be non-negative")
    predicted = true_positive + false_positive
    positives = true_positive + false_negative
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / positives if positives else 0.0
    calibrated = (true_positive + 1.0) / (predicted + 2.0)
    quality = dict(pattern.quality)
    quality.update(
        {
            "confidence_status": "calibrated",
            "calibration_examples": predicted,
            "empirical_precision": precision,
            "empirical_recall": recall,
            "calibrated_confidence": calibrated,
        }
    )
    return replace(pattern, quality=quality)
