from __future__ import annotations

import copy
import unittest
from dataclasses import FrozenInstanceError

from hamletkg.detection.policy import DetectionPolicy, DetectionRule, validate_policy_for_stream


def payload():
    return {
        "policy_id": "P1", "version": "1",
        "rules": [
            {"rule_id": "R1", "operator": "fan_in", "window_seconds": 3600, "min_counterparties": 3},
            {"rule_id": "R2", "operator": "pass_through", "window_seconds": 300,
             "min_incoming": 2, "ratio_min": .9, "ratio_max": 1.1},
        ],
    }


class DetectionPolicyTests(unittest.TestCase):
    def test_roundtrip_only_serializes_operator_relevant_fields(self):
        policy = DetectionPolicy.from_dict(payload())
        self.assertEqual(policy, DetectionPolicy.from_dict(policy.to_dict()))
        self.assertEqual(payload(), policy.to_dict())
        fan = DetectionRule("F", "fan_out", 10)
        relay = DetectionRule("L", "relay", 10)
        self.assertEqual({"rule_id", "operator", "window_seconds", "min_counterparties"}, set(fan.to_dict()))
        self.assertEqual({"rule_id", "operator", "window_seconds", "ratio_min", "ratio_max"}, set(relay.to_dict()))

    def test_input_and_serialized_containers_do_not_mutate_policy(self):
        value = payload()
        policy = DetectionPolicy.from_dict(value)
        original = policy.to_dict()
        value["rules"][0]["min_counterparties"] = 99
        value["rules"].clear()
        serialized = policy.to_dict()
        serialized["rules"][0]["window_seconds"] = 0
        serialized["rules"].clear()
        self.assertEqual(original, policy.to_dict())
        mutable = [DetectionRule("R", "fan_in", 10)]
        constructed = DetectionPolicy("P", "1", mutable)
        mutable.clear()
        self.assertEqual(1, len(constructed.rules))
        with self.assertRaises(FrozenInstanceError):
            constructed.rules[0].window_seconds = 30

    def test_unknown_legacy_or_irrelevant_fields_are_rejected(self):
        for field in ("description", "quality", "constraints", "pattern_id", "provenance", "extra"):
            value = payload()
            value[field] = "not-supported"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "unsupported"):
                DetectionPolicy.from_dict(value)
        irrelevant = {
            "fan_in": ("min_incoming", "ratio_min", "ratio_max"),
            "fan_out": ("min_incoming", "ratio_min", "ratio_max"),
            "pass_through": ("min_counterparties",),
            "relay": ("min_counterparties", "min_incoming"),
        }
        for operator, fields in irrelevant.items():
            for field in fields:
                rule = {"rule_id": "R", "operator": operator, "window_seconds": 10, field: 1}
                with self.subTest(operator=operator, field=field), self.assertRaisesRegex(ValueError, "unsupported"):
                    DetectionRule.from_dict(rule)

    def test_missing_fields_empty_rules_and_duplicate_rule_ids_fail(self):
        for field in ("policy_id", "version", "rules"):
            value = payload()
            del value[field]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "missing"):
                DetectionPolicy.from_dict(value)
        for rules in ([], "rules", {"rule_id": "R"}, [payload()["rules"][0]] * 2):
            value = payload()
            value["rules"] = rules
            with self.subTest(rules=rules), self.assertRaises(ValueError):
                DetectionPolicy.from_dict(value)
        for field in ("rule_id", "operator", "window_seconds"):
            rule = copy.deepcopy(payload()["rules"][0])
            del rule[field]
            with self.subTest(rule_field=field), self.assertRaisesRegex(ValueError, "missing"):
                DetectionRule.from_dict(rule)

    def test_identifiers_operator_and_numbers_are_strictly_validated(self):
        for changes in (
            {"rule_id": ""}, {"rule_id": 1}, {"rule_id": " R "},
            {"operator": "cycle"}, {"operator": []},
            {"window_seconds": 0}, {"window_seconds": float("nan")},
            {"window_seconds": float("inf")}, {"window_seconds": "30"},
            {"window_seconds": 10 ** 400},
            {"window_seconds": True}, {"min_counterparties": 1},
            {"min_counterparties": 3.0}, {"min_counterparties": True},
        ):
            rule = payload()["rules"][0] | changes
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                DetectionRule.from_dict(rule)
        for changes in (
            {"min_incoming": 0}, {"min_incoming": 1.5}, {"min_incoming": True},
            {"ratio_min": 0}, {"ratio_max": float("inf")},
            {"ratio_min": "0.8"}, {"ratio_min": True},
            {"ratio_min": 2, "ratio_max": 1},
        ):
            rule = payload()["rules"][1] | changes
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                DetectionRule.from_dict(rule)
        for key in ("policy_id", "version"):
            for invalid in (None, 1, "", " value "):
                value = payload() | {key: invalid}
                with self.subTest(key=key, invalid=invalid), self.assertRaises(ValueError):
                    DetectionPolicy.from_dict(value)

    def test_direct_construction_cannot_silently_drop_irrelevant_nondefaults(self):
        for kwargs in (
            {"operator": "fan_in", "ratio_min": .5},
            {"operator": "fan_out", "min_incoming": 2},
            {"operator": "pass_through", "min_counterparties": 4},
            {"operator": "relay", "min_incoming": 2},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DetectionRule("R", window_seconds=10, **kwargs)

    def test_retained_history_capacity_is_validated_explicitly(self):
        compatible = DetectionPolicy("P", "1", (
            DetectionRule("F", "fan_in", 10, min_counterparties=3),
            DetectionRule("P", "pass_through", 10, min_incoming=2),
        ))
        validate_policy_for_stream(compatible, window_seconds=10, max_history=2)
        for rule in (
            DetectionRule("W", "relay", 11),
            DetectionRule("F", "fan_in", 10, min_counterparties=4),
            DetectionRule("O", "fan_out", 10, min_counterparties=4),
            DetectionRule("P", "pass_through", 10, min_incoming=3),
        ):
            with self.subTest(rule=rule), self.assertRaises(ValueError):
                validate_policy_for_stream(DetectionPolicy("P", "1", (rule,)), window_seconds=10, max_history=2)


if __name__ == "__main__":
    unittest.main()
