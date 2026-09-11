#!/usr/bin/env python3
"""Run standard reference algorithms used by the proposed method.

Requires ``requirements-experiments.txt``.  Results are deterministic and are
written as JSON so they can be cited by the experiment notebook.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import ot
from scipy.optimize import linear_sum_assignment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hamletkg.io import read_json  # noqa: E402
from hamletkg.models import CaseGraph  # noqa: E402
from hamletkg.similarity import HierarchicalCaseMatcher  # noqa: E402


def _load_case(path: Path) -> CaseGraph:
    return CaseGraph.from_dict(read_json(path))


def _structural_cost(case: CaseGraph, event_ids: list[str]) -> np.ndarray:
    graph = nx.Graph()
    graph.add_nodes_from(event_ids)
    valid = set(event_ids)
    for relation in case.risk_relations:
        if relation.source_id in valid and relation.target_id in valid:
            graph.add_edge(relation.source_id, relation.target_id)
    distances = nx.floyd_warshall_numpy(graph, nodelist=event_ids).astype(float)
    finite = distances[np.isfinite(distances)]
    fill = float(finite.max() + 1.0) if finite.size else 1.0
    distances[~np.isfinite(distances)] = fill
    maximum = float(distances.max())
    return distances / maximum if maximum else distances


def run_transport_references(query_path: Path, candidate_path: Path) -> dict:
    query = _load_case(query_path)
    candidate = _load_case(candidate_path)
    matcher = HierarchicalCaseMatcher()
    left = sorted(query.risk_events.values(), key=lambda item: item.risk_event_id)
    right = sorted(candidate.risk_events.values(), key=lambda item: item.risk_event_id)
    similarities = np.array(
        [
            [matcher.risk_event_similarity(query, a, candidate, b).score for b in right]
            for a in left
        ],
        dtype=float,
    )
    costs = 1.0 - similarities
    p = np.full(len(left), 1.0 / len(left))
    q = np.full(len(right), 1.0 / len(right))

    started = time.perf_counter()
    row_indices, column_indices = linear_sum_assignment(costs)
    hard_score = float(similarities[row_indices, column_indices].mean())
    hard_score *= min(len(left), len(right)) / max(len(left), len(right))
    hungarian_seconds = time.perf_counter() - started

    started = time.perf_counter()
    sinkhorn_plan = ot.sinkhorn(p, q, costs, reg=0.12, numItermax=1000)
    sinkhorn_score = float(np.sum(sinkhorn_plan * similarities))
    sinkhorn_seconds = time.perf_counter() - started

    started = time.perf_counter()
    fgw_cost = float(
        ot.gromov.fused_gromov_wasserstein2(
            costs,
            _structural_cost(query, [item.risk_event_id for item in left]),
            _structural_cost(candidate, [item.risk_event_id for item in right]),
            p=p,
            q=q,
            alpha=0.5,
            max_iter=1000,
        )
    )
    fgw_seconds = time.perf_counter() - started

    return {
        "query": query.case_id,
        "candidate": candidate.case_id,
        "risk_events": {"query": len(left), "candidate": len(right)},
        "hungarian": {
            "similarity": round(hard_score, 6),
            "runtime_seconds": round(hungarian_seconds, 9),
        },
        "sinkhorn": {
            "similarity": round(sinkhorn_score, 6),
            "runtime_seconds": round(sinkhorn_seconds, 9),
        },
        "fused_gromov_wasserstein": {
            "objective": round(fgw_cost, 6),
            "similarity_proxy": round(max(0.0, 1.0 - fgw_cost), 6),
            "runtime_seconds": round(fgw_seconds, 9),
        },
        "hamlet_hierarchical": matcher.compare(query, candidate).to_dict(),
    }


def run_vf2_reference(transactions_path: Path, *, limit: int) -> dict:
    graph = nx.DiGraph()
    rows = 0
    with transactions_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in itertools.islice(csv.DictReader(handle), limit):
            graph.add_edge(int(row["sourceNodeId"]), int(row["targetNodeId"]))
            rows += 1

    pattern = nx.DiGraph()
    pattern.add_edges_from([("source-1", "anchor"), ("source-2", "anchor")])
    matcher = nx.algorithms.isomorphism.DiGraphMatcher(graph, pattern)
    cap = 10_000
    started = time.perf_counter()
    vf2_matches = sum(
        1 for _ in itertools.islice(matcher.subgraph_monomorphisms_iter(), cap)
    )
    vf2_seconds = time.perf_counter() - started

    started = time.perf_counter()
    incoming: dict[int, set[int]] = {}
    for source, target in graph.edges:
        incoming.setdefault(target, set()).add(source)
    indexed_matches = sum(math.comb(len(sources), 2) for sources in incoming.values() if len(sources) >= 2)
    indexed_seconds = time.perf_counter() - started
    return {
        "input_transaction_rows": rows,
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "pattern": "two-source fan-in",
        "vf2_monomorphisms": vf2_matches,
        "vf2_capped": vf2_matches == cap,
        "vf2_runtime_seconds": round(vf2_seconds, 9),
        "indexed_unordered_matches": indexed_matches,
        "indexed_runtime_seconds": round(indexed_seconds, 9),
        "note": "VF2 counts source permutations; indexed count uses unordered source pairs.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="outputs/demo/case_a.json")
    parser.add_argument("--candidate", default="outputs/demo/case_b.json")
    parser.add_argument(
        "--transactions",
        default="data/raw/amlsim/20K_fanin200/transactions.csv",
    )
    parser.add_argument("--vf2-limit", type=int, default=5_000)
    parser.add_argument("--output", default="outputs/reference-algorithms/results.json")
    args = parser.parse_args()

    result = {
        "transport_references": run_transport_references(
            Path(args.query), Path(args.candidate)
        ),
        "subgraph_matching_reference": run_vf2_reference(
            Path(args.transactions), limit=args.vf2_limit
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
