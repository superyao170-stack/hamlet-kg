from __future__ import annotations

import unittest

from hamletkg.llm import LLMAugmentor, LLMError


class SequenceClient:
    def __init__(self, values):
        self.values = list(values)
        self.calls = []

    def complete_json(self, *, system: str, user: str):
        self.calls.append({"system": system, "user": user})
        return self.values.pop(0)


def pattern_payload(anchor_role: str = "sender", clause_id: str = "a") -> dict:
    return {
        "pattern_id": "P-LLM",
        "risk_type": "TEST",
        "description": "test",
        "version": "1.0",
        "clauses": [
            {
                "clause_id": clause_id,
                "action_type": "TRANSFER",
                "anchor_role": anchor_role,
                "min_count": 1,
            }
        ],
        "constraints": [],
    }


class LLMCompilationTests(unittest.TestCase):
    def test_invalid_candidate_is_repaired_once(self) -> None:
        client = SequenceClient([pattern_payload("account"), pattern_payload("sender")])
        pattern = LLMAugmentor(client=client).compile_pattern("test rule")
        self.assertEqual("sender", pattern.clauses[0].anchor_role)
        self.assertEqual(2, len(client.calls))
        self.assertIn("failed deterministic validation", client.calls[1]["user"])

    def test_invalid_candidate_fails_when_repairs_are_disabled(self) -> None:
        client = SequenceClient([pattern_payload("account")])
        with self.assertRaises(LLMError):
            LLMAugmentor(client=client).compile_pattern(
                "test rule", max_repair_attempts=0
            )

    def test_multiple_samples_report_semantic_agreement(self) -> None:
        client = SequenceClient(
            [pattern_payload("sender"), pattern_payload("receiver"), pattern_payload("sender")]
        )
        pattern = LLMAugmentor(client=client).compile_pattern(
            "test rule", samples=3, max_repair_attempts=0
        )
        self.assertEqual("sender", pattern.clauses[0].anchor_role)
        self.assertEqual(3, pattern.quality["valid_samples"])
        self.assertAlmostEqual(2 / 3, pattern.quality["semantic_agreement"])
        self.assertEqual("uncalibrated", pattern.quality["confidence_status"])

    def test_agreement_ignores_clause_identifier_renaming(self) -> None:
        client = SequenceClient(
            [pattern_payload("sender", "incoming"), pattern_payload("sender", "c1")]
        )
        pattern = LLMAugmentor(client=client).compile_pattern(
            "test rule", samples=2, max_repair_attempts=0
        )
        self.assertEqual(1.0, pattern.quality["semantic_agreement"])
        self.assertEqual(1, pattern.quality["semantic_variant_count"])


if __name__ == "__main__":
    unittest.main()
