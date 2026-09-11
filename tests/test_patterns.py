from __future__ import annotations

import unittest

from hamletkg.models import RiskPattern
from hamletkg.patterns import (
    PatternValidationError,
    calibrate_pattern_reliability,
    validate_pattern,
    validate_patterns,
)


class PatternValidationTests(unittest.TestCase):
    def test_unknown_clause_reference_is_rejected(self) -> None:
        pattern = RiskPattern.from_dict(
            {
                "pattern_id": "P",
                "risk_type": "R",
                "clauses": [
                    {
                        "clause_id": "a",
                        "action_type": "TRANSFER",
                        "anchor_role": "sender"
                    }
                ],
                "constraints": [
                    {
                        "constraint_type": "ordered_within",
                        "parameters": {
                            "earlier": "a",
                            "later": "missing",
                            "max_seconds": 10
                        }
                    }
                ]
            }
        )
        with self.assertRaises(PatternValidationError):
            validate_pattern(pattern)

    def test_invalid_action_role_is_rejected(self) -> None:
        pattern = RiskPattern.from_dict(
            {
                "pattern_id": "P-ROLE",
                "risk_type": "R",
                "clauses": [
                    {
                        "clause_id": "a",
                        "action_type": "TRANSFER",
                        "anchor_role": "account",
                    }
                ],
            }
        )
        with self.assertRaises(PatternValidationError):
            validate_pattern(pattern)

    def test_contradictory_numeric_range_is_rejected(self) -> None:
        pattern = RiskPattern.from_dict(
            {
                "pattern_id": "P-RANGE",
                "risk_type": "R",
                "clauses": [
                    {
                        "clause_id": "a",
                        "action_type": "TRANSFER",
                        "anchor_role": "sender",
                    }
                ],
                "constraints": [
                    {
                        "constraint_type": "total_amount",
                        "parameters": {"clause": "a", "minimum": 20, "maximum": 10},
                    }
                ],
            }
        )
        with self.assertRaises(PatternValidationError):
            validate_pattern(pattern)

    def test_cyclic_ordering_is_rejected(self) -> None:
        pattern = RiskPattern.from_dict(
            {
                "pattern_id": "P-CYCLE",
                "risk_type": "R",
                "clauses": [
                    {"clause_id": "a", "action_type": "TRANSFER", "anchor_role": "sender"},
                    {"clause_id": "b", "action_type": "TRANSFER", "anchor_role": "receiver"},
                ],
                "constraints": [
                    {"constraint_type": "ordered_within", "parameters": {"earlier": "a", "later": "b", "max_seconds": 10}},
                    {"constraint_type": "ordered_within", "parameters": {"earlier": "b", "later": "a", "max_seconds": 10}},
                ],
            }
        )
        with self.assertRaises(PatternValidationError):
            validate_pattern(pattern)

    def test_pattern_reliability_uses_smoothed_validation_precision(self) -> None:
        pattern = RiskPattern.from_dict(
            {
                "pattern_id": "P-CAL",
                "risk_type": "R",
                "clauses": [
                    {
                        "clause_id": "a",
                        "action_type": "TRANSFER",
                        "anchor_role": "sender",
                    }
                ],
            }
        )
        calibrated = calibrate_pattern_reliability(
            pattern, true_positive=8, false_positive=2, false_negative=2
        )
        self.assertAlmostEqual(0.8, calibrated.quality["empirical_precision"])
        self.assertAlmostEqual(0.8, calibrated.quality["empirical_recall"])
        self.assertAlmostEqual(0.75, calibrated.quality["calibrated_confidence"])

    def test_attribute_range_requires_a_bound(self) -> None:
        pattern = RiskPattern.from_dict(
            {
                "pattern_id": "P-RANGE-BOUND",
                "risk_type": "R",
                "clauses": [
                    {
                        "clause_id": "cash",
                        "action_type": "CASH_DEPOSIT",
                        "anchor_role": "account",
                    }
                ],
                "constraints": [
                    {
                        "constraint_type": "attribute_range",
                        "parameters": {"clause": "cash", "attribute": "amount"},
                    }
                ],
            }
        )
        with self.assertRaises(PatternValidationError):
            validate_pattern(pattern)

    def test_role_overlap_rejects_invalid_role(self) -> None:
        pattern = RiskPattern.from_dict(
            {
                "pattern_id": "P-OVERLAP-ROLE",
                "risk_type": "R",
                "clauses": [
                    {
                        "clause_id": "incoming",
                        "action_type": "TRANSFER",
                        "anchor_role": "receiver",
                    },
                    {
                        "clause_id": "outgoing",
                        "action_type": "TRANSFER",
                        "anchor_role": "sender",
                    },
                ],
                "constraints": [
                    {
                        "constraint_type": "role_overlap",
                        "parameters": {
                            "left_clause": "incoming",
                            "left_role": "account",
                            "right_clause": "outgoing",
                            "right_role": "receiver",
                        },
                    }
                ],
            }
        )
        with self.assertRaises(PatternValidationError):
            validate_pattern(pattern)

    def test_literature_pattern_catalog_is_valid(self) -> None:
        import json
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        payload = json.loads(
            (root / "patterns/aml_risk_patterns_v1.json").read_text(encoding="utf-8")
        )
        patterns = [RiskPattern.from_dict(item) for item in payload["patterns"]]
        validate_patterns(patterns)
        self.assertEqual(7, len(patterns))
        self.assertTrue(all(item.provenance for item in patterns))


if __name__ == "__main__":
    unittest.main()
