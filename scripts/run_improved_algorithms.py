"""Run deterministic comparisons for the proposed algorithm modules."""

from __future__ import annotations

import json
import time
from pathlib import Path

from hamletkg.graph import build_atomic_relations, build_risk_relations
from hamletkg.materializer import IndexedWindowPatternMatcher, PatternMatcher
from hamletkg.models import CaseGraph, RiskPattern
from hamletkg.similarity import FusedGraphCaseMatcher, HierarchicalCaseMatcher, RiskCaseIndex


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "improved-algorithms" / "results.json"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_example_cases() -> list[CaseGraph]:
    values = read_json(ROOT / "examples" / "patterns.json")["patterns"]
    patterns = [RiskPattern.from_dict(item) for item in values]
    cases = []
    for path in sorted((ROOT / "examples" / "raw_cases").glob("*.json")):
        case = CaseGraph.from_dict(read_json(path))
        build_atomic_relations(case)
        matcher = IndexedWindowPatternMatcher()
        for pattern in patterns:
            events, _ = matcher.match(case, pattern)
            case.risk_events.update({item.risk_event_id: item for item in events})
        build_risk_relations(case)
        cases.append(case)
    return cases


def materialization_comparison() -> dict:
    case = CaseGraph.from_dict(
        read_json(ROOT / "outputs" / "amlsim-baselines" / "fanin-5k-atomic-graph.json")
    )
    pattern = RiskPattern.from_dict(
        read_json(ROOT / "outputs" / "deepseek" / "generated-pattern.json")
    )
    legacy = PatternMatcher()
    started = time.perf_counter()
    legacy_events, _ = legacy.match(case, pattern)
    legacy_seconds = time.perf_counter() - started

    improved = IndexedWindowPatternMatcher()
    started = time.perf_counter()
    improved_events, _, diagnostics = improved.match_with_diagnostics(case, pattern)
    improved_seconds = time.perf_counter() - started

    labels = case.metadata.get("evaluation_labels", {})
    def anchor_metrics(events):
        anchors = {item.anchor_entity_id for item in events}
        true_positive = sum(int(labels.get(anchor, 0)) for anchor in anchors)
        return {
            "risk_events": len(events),
            "unique_anchors": len(anchors),
            "fraud_anchors": true_positive,
            "anchor_precision": true_positive / len(anchors) if anchors else 0.0,
        }

    return {
        "dataset": "20K_fanin200 first 5,000 transactions",
        "pattern": pattern.pattern_id,
        "legacy_full_history": {
            **anchor_metrics(legacy_events),
            "runtime_seconds": round(legacy_seconds, 9),
        },
        "indexed_window": {
            **anchor_metrics(improved_events),
            "runtime_seconds": round(improved_seconds, 9),
            "diagnostics": diagnostics.to_dict(),
            "anchor_pruning_rate": round(
                diagnostics.pruned_anchors / diagnostics.total_entities, 9
            ),
        },
        "interpretation": (
            "The generated rule remains poorly calibrated to this AMLSim slice; "
            "the proposed matcher improves temporal instance semantics and exposes "
            "pruning behavior, but does not manufacture additional positives."
        ),
    }


def retrieval_comparison() -> dict:
    cases = prepare_example_cases()
    by_id = {item.case_id: item for item in cases}
    query = by_id["CASE-A"]
    candidate = by_id["CASE-B"]

    legacy = HierarchicalCaseMatcher()
    started = time.perf_counter()
    legacy_result = legacy.compare(query, candidate)
    legacy_seconds = time.perf_counter() - started

    fused = FusedGraphCaseMatcher()
    started = time.perf_counter()
    fused_result = fused.compare(query, candidate)
    fused_seconds = time.perf_counter() - started

    indexed_results, index_diagnostics = RiskCaseIndex(cases).search(
        query,
        matcher=fused,
        top_k=1,
        candidate_pool_size=1,
    )
    return {
        "query": query.case_id,
        "candidate": candidate.case_id,
        "hierarchical_sinkhorn": {
            **legacy_result.to_dict(),
            "runtime_seconds": round(legacy_seconds, 9),
        },
        "fused_graph_sinkhorn": {
            **fused_result.to_dict(),
            "runtime_seconds": round(fused_seconds, 9),
        },
        "indexed_top1": {
            "candidate_case_id": indexed_results[0].candidate_case_id,
            "similarity": round(indexed_results[0].similarity, 6),
            "diagnostics": index_diagnostics.to_dict(),
        },
        "scope_note": "The three example cases verify behavior, not statistical superiority.",
    }


def main() -> None:
    result = {
        "experiment": "HAMLET-KG proposed algorithm smoke comparison",
        "materialization": materialization_comparison(),
        "retrieval": retrieval_comparison(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
