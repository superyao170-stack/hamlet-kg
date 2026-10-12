from __future__ import annotations

import math
import hashlib
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from hamletkg.detection.stream import (
    BASE_FEATURE_NAMES, MEMBER_ROLES, MOTIF_TYPES, StreamConfig, build_samples,
)
from hamletkg.models import AtomicEvent, CaseGraph, Entity
from hamletkg.detection.policy import DetectionPolicy, DetectionRule


START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def transfer(event_id, sender, receiver, seconds, amount=100.0, currency="USD", *, received_currency=None, received_amount=None):
    return AtomicEvent(
        event_id, "TRANSFER", {"sender": sender, "receiver": receiver},
        START + timedelta(seconds=seconds),
        {
            "amount": amount, "paid_currency": currency,
            "received_amount": amount if received_amount is None else received_amount,
            "received_currency": received_currency or currency,
        },
    )


def case(events, metadata=None):
    ids = {account for event in events for account in event.roles.values()}
    return CaseGraph("C", {account: Entity(account, "ACCOUNT") for account in ids}, {event.event_id: event for event in events}, metadata=metadata or {})


def feature(sample, name):
    return sample.features[BASE_FEATURE_NAMES.index(name)]


class CausalStreamTests(unittest.TestCase):
    def test_features_ignore_labels_and_numeric_account_id_values(self):
        events = [transfer("T1", "A", "B", 0), transfer("T2", "B", "C", 10)]
        unlabeled = build_samples(case(events))
        labeled = build_samples(case(events, {"evaluation_labels": {"T1": 1, "T2": 0}, "fraudStep": 99}))
        self.assertEqual(unlabeled, labeled)
        renamed = [replace(item, roles={role: f"ACCOUNT:9999:{value}" for role, value in item.roles.items()}) for item in events]
        renamed_samples = build_samples(case(renamed))
        self.assertEqual([item.features for item in unlabeled], [item.features for item in renamed_samples])
        for item in unlabeled:
            self.assertEqual(len(BASE_FEATURE_NAMES), len(item.features))
            self.assertTrue(all(math.isfinite(value) for value in item.features))

    def test_future_transactions_do_not_change_earlier_samples(self):
        events = [transfer("T1", "A", "B", 0), transfer("T2", "B", "C", 10)]
        prefix = build_samples(case(events))
        extension = build_samples(case(events + [transfer("T3", "C", "D", 20)]))
        self.assertEqual(prefix, extension[:2])

    def test_same_timestamp_transactions_do_not_see_each_other(self):
        first = transfer("T1", "A", "B", 0)
        peer = transfer("T2", "B", "C", 0)
        alone = build_samples(case([first]))[0]
        simultaneous = build_samples(case([peer, first]))
        self.assertEqual(alone, simultaneous[0])
        self.assertEqual(0.0, feature(simultaneous[1], "sender_in_log_count"))
        self.assertEqual((), simultaneous[1].events)
        followup = build_samples(case([peer, first, transfer("T3", "B", "D", 10)]))[-1]
        self.assertEqual(math.log1p(1), feature(followup, "sender_in_log_count"))
        self.assertEqual(math.log1p(1), feature(followup, "sender_out_log_count"))

    def test_currency_buckets_do_not_mix_amounts_or_motifs(self):
        samples = build_samples(case([
            transfer("T1", "A", "H", 0, currency="USD"),
            transfer("T2", "B", "H", 1, currency="EUR"),
            transfer("T3", "C", "H", 2, currency="EUR"),
            transfer("T4", "H", "D", 3, currency="USD"),
        ]))
        self.assertFalse(any(item.motif_type == "fan_in" for item in samples[2].events))
        outgoing = samples[-1]
        self.assertEqual(math.log1p(1), feature(outgoing, "sender_in_log_count"))
        event = next(item for item in outgoing.events if item.motif_type == "pass_through")
        self.assertEqual({"T1", "T4"}, {item.event_id for item in event.members})

    def test_foreign_exchange_uses_received_currency_and_amount_at_intermediary(self):
        samples = build_samples(case([
            transfer("T1", "A", "H", 0, 100, "USD", received_currency="EUR", received_amount=90),
            transfer("T2", "H", "B", 1, 90, "EUR"),
        ]))
        instance = next(item for item in samples[-1].events if item.motif_type == "pass_through")
        self.assertEqual(1.0, instance.amount_ratio)
        self.assertEqual(0.0, feature(samples[0], "same_currency"))

    def test_fan_in_and_out_require_distinct_counterparties(self):
        incoming = [transfer(f"I{i}", f"S{i}", "H", i) for i in range(3)]
        outgoing = [transfer(f"O{i}", "H", f"D{i}", i + 4) for i in range(3)]
        samples = build_samples(case(incoming + outgoing))
        fan_in = next(item for item in samples[2].events if item.motif_type == "fan_in")
        fan_out = next(item for item in samples[-1].events if item.motif_type == "fan_out")
        self.assertEqual({"incoming"}, {item.role for item in fan_in.members})
        self.assertEqual({"outgoing"}, {item.role for item in fan_out.members})
        repeated = build_samples(case([transfer(f"R{i}", "A", "H", i) for i in range(3)]))
        self.assertFalse(any(item.motif_type == "fan_in" for item in repeated[-1].events))

    def test_relay_preserves_roles_causality_and_arrival_features(self):
        samples = build_samples(case([
            transfer("T1", "A", "B", 0), transfer("T2", "B", "C", 10),
            transfer("T3", "C", "D", 20),
        ]))
        relay = next(item for item in samples[-1].events if item.motif_type == "relay")
        self.assertEqual(["relay_start", "relay_middle", "relay_end"], [item.role for item in relay.members])
        self.assertEqual([20.0, 10.0, 0.0], [item.elapsed_seconds for item in relay.members])
        self.assertEqual(20.0, relay.span_seconds)
        by_id = {item.event_id: item for item in samples}
        for instance in samples[-1].events:
            self.assertIn(instance.motif_type, MOTIF_TYPES)
            self.assertIn("T3", {item.event_id for item in instance.members})
            for member in instance.members:
                self.assertIn(member.role, MEMBER_ROLES)
                self.assertEqual(by_id[member.event_id].features, member.features)

    def test_relay_does_not_invent_order_inside_a_timestamp(self):
        samples = build_samples(case([
            transfer("T1", "A", "B", 0), transfer("T2", "B", "C", 0),
            transfer("T3", "C", "D", 20),
        ]))
        self.assertFalse(any(item.motif_type == "relay" for item in samples[-1].events))

    def test_default_relay_rejects_overflowing_product_of_finite_hop_ratios(self):
        events = [
            transfer("T1", "A", "B", 0, 1e-200),
            transfer("T2", "B", "C", 1, 1.0),
            transfer("T3", "C", "D", 2, 1e200),
        ]
        config = StreamConfig(window_seconds=10, ratio_min=1e199, ratio_max=1e201)
        self.assertEqual(2, len(build_samples(case(events[:2]), config)))
        with self.assertRaisesRegex(ValueError, "relay event for transfer T3 amount_ratio must be finite"):
            build_samples(case(events), config)

    def test_history_count_and_time_window_are_bounded(self):
        transfers = [transfer(f"T{i}", f"S{i}", "H", i) for i in range(5)]
        samples = build_samples(case(transfers), StreamConfig(max_history=2))
        last = samples[-1]
        self.assertEqual(math.log1p(2), feature(last, "receiver_in_log_count"))
        fan = next(item for item in last.events if item.motif_type == "fan_in")
        self.assertEqual({"T2", "T3", "T4"}, {item.event_id for item in fan.members})
        expired = build_samples(case(transfers + [transfer("LATE", "H", "D", 100)]), StreamConfig(window_seconds=10))[-1]
        self.assertEqual(0.0, feature(expired, "sender_in_log_count"))
        self.assertEqual((), expired.events)

    def test_excluded_evidence_is_removed_before_rebuilding(self):
        graph = case([transfer("T1", "A", "B", 0), transfer("T2", "B", "C", 10)])
        removed = build_samples(graph, excluded_ids={"T1"})
        direct = build_samples(case([graph.atomic_events["T2"]]))
        self.assertEqual(direct, removed)
        self.assertEqual((), removed[0].events)

    def test_bad_stream_inputs_and_configuration_raise(self):
        for kwargs in ({"window_seconds": 0}, {"window_seconds": "bad"}, {"window_seconds": float("nan")}, {"max_history": 0}, {"max_history": 1.5}, {"min_fan": 1}, {"ratio_min": 2, "ratio_max": 1}, {"ratio_min": "bad"}, {"ratio_max": float("inf")}):
            with self.subTest(config=kwargs), self.assertRaises(ValueError):
                StreamConfig(**kwargs)
        original = transfer("T", "A", "B", 0)
        for attributes in ({"amount": 0, "currency": "USD"}, {"amount": float("nan"), "currency": "USD"}, {"amount": 100}, {"amount": 100, "paid_currency": "USD", "received_currency": "EUR"}):
            with self.subTest(attributes=attributes), self.assertRaises(ValueError):
                build_samples(case([replace(original, attributes=attributes)]))

    def test_finite_individual_amounts_do_not_overflow_aggregate_features(self):
        samples = build_samples(case([
            transfer("T1", "A", "H", 0, 1e308),
            transfer("T2", "B", "H", 1, 1e308),
            transfer("T3", "C", "H", 2, 1e308),
        ]))
        self.assertTrue(all(math.isfinite(value) for sample in samples for value in sample.features))


class PolicyStreamTests(unittest.TestCase):
    def policy(self, *rules):
        return DetectionPolicy("TEST-POLICY", "1", rules)

    def test_policy_enables_only_its_operators_without_changing_base_features(self):
        graph = case([
            transfer("I1", "A", "H", 0, 100), transfer("I2", "B", "H", 1, 100),
            transfer("I3", "C", "H", 2, 100), transfer("O1", "H", "D", 3, 285),
        ])
        default = build_samples(graph)
        policy = self.policy(DetectionRule("ONLY-PASS", "pass_through", 100))
        controlled = build_samples(graph, policy=policy)
        self.assertEqual([sample.features for sample in default], [sample.features for sample in controlled])
        self.assertTrue(any(event.motif_type == "fan_in" for sample in default for event in sample.events))
        self.assertEqual({"pass_through"}, {event.motif_type for sample in controlled for event in sample.events})
        self.assertEqual({"ONLY-PASS"}, {event.rule_id for sample in controlled for event in sample.events})

    def test_fan_windows_and_counterparty_thresholds_change_members_and_hits(self):
        graph = case([
            transfer("I1", "A", "H", 0), transfer("I2", "B", "H", 10),
            transfer("I3", "C", "H", 20),
        ])
        policy = self.policy(
            DetectionRule("LONG", "fan_in", 30, min_counterparties=3),
            DetectionRule("SHORT", "fan_in", 10, min_counterparties=3),
            DetectionRule("TWO", "fan_in", 10, min_counterparties=2),
            DetectionRule("FOUR", "fan_in", 30, min_counterparties=4),
        )
        last = build_samples(graph, policy=policy)[-1]
        self.assertEqual({"LONG", "TWO"}, {event.rule_id for event in last.events})
        by_rule = {event.rule_id: event for event in last.events}
        self.assertEqual({"I1", "I2", "I3"}, {member.event_id for member in by_rule["LONG"].members})
        self.assertEqual({"I2", "I3"}, {member.event_id for member in by_rule["TWO"].members})
        self.assertEqual(10, by_rule["TWO"].span_seconds)
        outgoing = case([transfer(f"O{i}", "H", f"D{i}", i * 10) for i in range(3)])
        self.assertEqual((), build_samples(outgoing, policy=self.policy(DetectionRule("SHORT", "fan_out", 10)))[-1].events)
        self.assertEqual("fan_out", build_samples(outgoing, policy=self.policy(DetectionRule("LONG", "fan_out", 30)))[-1].events[0].motif_type)

    def test_pass_through_windows_incoming_counts_and_ratios_are_executable(self):
        graph = case([
            transfer("I1", "A", "H", 0, 100), transfer("I2", "B", "H", 10, 100),
            transfer("O", "H", "D", 20, 190),
        ])
        policy = self.policy(
            DetectionRule("MATCH", "pass_through", 30, min_incoming=2, ratio_min=.9, ratio_max=1),
            DetectionRule("TOO-FEW", "pass_through", 30, min_incoming=3),
            DetectionRule("TOO-HIGH", "pass_through", 30, ratio_min=1, ratio_max=1.1),
            DetectionRule("TOO-SHORT", "pass_through", 10, min_incoming=2),
            DetectionRule("ONE-IN", "pass_through", 10, ratio_min=1.8, ratio_max=2),
        )
        last = build_samples(graph, policy=policy)[-1]
        by_rule = {event.rule_id: event for event in last.events}
        self.assertEqual({"MATCH", "ONE-IN"}, set(by_rule))
        self.assertEqual(.95, by_rule["MATCH"].amount_ratio)
        self.assertEqual(1.9, by_rule["ONE-IN"].amount_ratio)
        self.assertEqual(["incoming", "incoming", "outgoing"], [member.role for member in by_rule["MATCH"].members])

    def test_relay_window_applies_to_entire_path_and_ratio_to_each_hop(self):
        graph = case([
            transfer("T1", "A", "B", 0, 100), transfer("T2", "B", "C", 10, 90),
            transfer("T3", "C", "D", 20, 81),
        ])
        policy = self.policy(
            DetectionRule("GOOD", "relay", 20, ratio_min=.89, ratio_max=.91),
            DetectionRule("SHORT", "relay", 10),
            DetectionRule("STRICT", "relay", 30, ratio_min=.95, ratio_max=1),
        )
        last = build_samples(graph, policy=policy)[-1]
        self.assertEqual(["GOOD"], [event.rule_id for event in last.events])
        relay = last.events[0]
        self.assertAlmostEqual(.81, relay.amount_ratio)
        self.assertEqual(["T1", "T2", "T3"], [member.event_id for member in relay.members])
        self.assertEqual(["relay_start", "relay_middle", "relay_end"], [member.role for member in relay.members])

    def test_policy_relay_rejects_overflowing_product_of_finite_hop_ratios(self):
        events = [
            transfer("T1", "A", "B", 0, 1e-200),
            transfer("T2", "B", "C", 1, 1.0),
            transfer("T3", "C", "D", 2, 1e200),
        ]
        policy = self.policy(DetectionRule("EXTREME", "relay", 10,
                                           ratio_min=1e199, ratio_max=1e201))
        self.assertEqual(2, len(build_samples(case(events[:2]), policy=policy)))
        with self.assertRaisesRegex(ValueError, "relay event for transfer T3 amount_ratio must be finite"):
            build_samples(case(events), policy=policy)

    def test_multiple_identical_operator_rules_have_distinct_instance_ids(self):
        graph = case([transfer("T1", "A", "H", 0), transfer("T2", "H", "B", 1)])
        policy = self.policy(DetectionRule("R1", "pass_through", 10), DetectionRule("R2", "pass_through", 10))
        events = build_samples(graph, policy=policy)[-1].events
        self.assertEqual({"R1", "R2"}, {event.rule_id for event in events})
        self.assertEqual(2, len({event.instance_id for event in events}))
        self.assertEqual(events[0].members, events[1].members)

    def test_none_policy_preserves_old_identity_exactly(self):
        graph = case([transfer("T1", "A", "H", 0), transfer("T2", "H", "B", 1)])
        event = build_samples(graph)[-1].events[0]
        identity = "pass_through|H|T1:incoming|T2:outgoing"
        expected = "EVENT-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        self.assertEqual(expected, event.instance_id)
        self.assertIsNone(event.rule_id)
        self.assertEqual(build_samples(graph), build_samples(graph, policy=None))

    def test_policy_future_and_same_timestamp_peers_do_not_affect_past_samples(self):
        policy = self.policy(DetectionRule("P", "pass_through", 100), DetectionRule("L", "relay", 100))
        events = [transfer("T1", "A", "B", 0), transfer("T2", "B", "C", 10)]
        prefix = build_samples(case(events), policy=policy)
        extended = build_samples(case(events + [transfer("T3", "C", "D", 20)]), policy=policy)
        self.assertEqual(prefix, extended[:2])
        simultaneous = build_samples(case([transfer("T1", "A", "B", 0), transfer("T2", "B", "C", 0)]), policy=policy)
        self.assertEqual((), simultaneous[-1].events)

    def test_policy_keeps_currency_and_label_isolation_and_deletion_replay(self):
        policy = self.policy(DetectionRule("P", "pass_through", 100, ratio_min=.9, ratio_max=1.1))
        graph = case([
            transfer("I1", "A", "H", 0, 100, "EUR"),
            transfer("I2", "B", "H", 1, 100, "USD"),
            transfer("O", "H", "D", 2, 100, "USD"),
        ])
        sample = build_samples(graph, policy=policy)[-1]
        self.assertEqual({"I2", "O"}, {member.event_id for member in sample.events[0].members})
        graph.metadata["evaluation_labels"] = {"I1": 1, "I2": 1, "O": 0}
        self.assertEqual(sample, build_samples(graph, policy=policy)[-1])
        self.assertEqual((), build_samples(graph, policy=policy, excluded_ids={"I2"})[-1].events)

    def test_unretainable_policies_are_rejected_before_building(self):
        graph = case([transfer("T", "A", "B", 0)])
        config = StreamConfig(window_seconds=10, max_history=2)
        for rule in (DetectionRule("W", "fan_in", 11),
                     DetectionRule("F", "fan_in", 10, min_counterparties=4),
                     DetectionRule("P", "pass_through", 10, min_incoming=3)):
            with self.subTest(rule=rule), self.assertRaises(ValueError):
                build_samples(graph, config, policy=self.policy(rule))


if __name__ == "__main__":
    unittest.main()
