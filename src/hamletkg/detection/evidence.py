"""Offline evidence audits with deletion and full causal feature rebuilding.

Deleting an event embedding alone would leave its information in historical
features. We therefore remove source transactions and replay the causal prefix.
These perturbations are model audits, not causal claims about the real world.
"""

from __future__ import annotations

import random
from typing import Any

from ..models import CaseGraph, iso_time
from .stream import StreamConfig, TransactionSample, build_samples
from .policy import DetectionPolicy


def evidence_graph(case: CaseGraph, sample: TransactionSample, event: Any) -> dict:
    """Export the observed account/transaction/event relations for one candidate."""
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    event_node = "EVENT:" + event.instance_id
    nodes[event_node] = {
        "id": event_node, "type": "BEHAVIOR_EVENT", "motif_type": event.motif_type,
        "anchor_id": event.anchor_id, "span_seconds": event.span_seconds,
        "amount_ratio": event.amount_ratio, "candidate_only": True,
        "rule_id": event.rule_id,
    }
    for member in event.members:
        transaction = case.atomic_events[member.event_id]
        nodes[member.event_id] = {
            "id": member.event_id, "type": "TRANSACTION",
            "timestamp": iso_time(transaction.timestamp),
            "source_record_id": transaction.source_record_id,
            "attributes": transaction.attributes,
            "is_target": member.event_id == sample.event_id,
        }
        for role, account_id in transaction.roles.items():
            nodes[account_id] = {"id": account_id, "type": "ACCOUNT"}
            if role == "sender":
                edges.append({"source": account_id, "relation": "SENDER_OF",
                              "target": member.event_id})
            elif role == "receiver":
                edges.append({"source": account_id, "relation": "RECEIVER_OF",
                              "target": member.event_id})
        edges.append({"source": member.event_id, "relation": "MEMBER_OF",
                      "target": event_node, "member_role": member.role,
                      "elapsed_seconds": member.elapsed_seconds})
    return {"nodes": list(nodes.values()), "edges": edges}


def explain_transaction(
    case: CaseGraph,
    sample: TransactionSample,
    detector: Any,
    config: StreamConfig,
    *,
    seed: int = 42,
    max_candidates: int = 3,
    policy: DetectionPolicy | None = None,
) -> dict:
    """Audit a bounded set of candidate events at a fixed observation time.

The target always remains observed. The random control removes the same number
of earlier transactions. The sufficiency audit retains only candidate members
and the target. Both audits recompute all derived features and event instances.
"""
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    observed = {
        key: event for key, event in case.atomic_events.items()
        if event.timestamp < sample.timestamp or key == sample.event_id
    }
    prefix = CaseGraph(case_id=case.case_id, entities=case.entities,
                       atomic_events=observed)
    history = sorted(key for key in observed if key != sample.event_id)
    score = detector.predict([sample])[0]
    result: dict = {
        "event_id": sample.event_id, "observation_time": iso_time(sample.timestamp),
        "score": score, "method": "source_transaction_deletion_with_causal_replay",
        "candidate_count": len(sample.events), "audited_candidates": [],
        "random_control_policy": "one_uniform_draw_from_all_observed_history_per_candidate",
        "random_control_limitation": "not matched to model receptive context, role or time; may remove irrelevant records",
        "status": "no_event_candidate" if not sample.events else "audited",
    }

    def replay(excluded: set[str]) -> float:
        rebuilt = build_samples(prefix, config, excluded_ids=excluded, policy=policy)
        target = next(item for item in rebuilt if item.event_id == sample.event_id)
        return detector.predict([target])[0]

    # This bounded, deterministic candidate selection is reported explicitly.
    # It is not an exhaustive search for a minimal sufficient explanation.
    candidates = sorted(sample.events,
                        key=lambda item: (-len(item.members), item.instance_id))[:max_candidates]
    for index, event in enumerate(candidates):
        support = {member.event_id for member in event.members} - {sample.event_id}
        if not support:
            continue
        without_score = replay(support)
        random_ids = set(random.Random(seed + index).sample(history, len(support)))
        random_score = replay(random_ids)
        retained_score = replay(set(history) - support)
        audit = {
            "instance_id": event.instance_id, "motif_type": event.motif_type,
            "historical_evidence_ids": sorted(support),
            "without_evidence_score": without_score,
            "deletion_score_drop": score - without_score,
            "random_control_ids": sorted(random_ids),
            "random_deletion_score": random_score,
            "random_deletion_score_drop": score - random_score,
            "evidence_only_score": retained_score,
            "evidence_only_absolute_change": abs(score - retained_score),
            "graph": evidence_graph(case, sample, event),
        }
        result["audited_candidates"].append(audit)
    audits = result["audited_candidates"]
    if audits:
        strongest = max(audits, key=lambda item: item["deletion_score_drop"])
        result["strongest_candidate_id"] = strongest["instance_id"]
        result["positive_support_found"] = strongest["deletion_score_drop"] > 0
    else:
        result["positive_support_found"] = False
    return result
