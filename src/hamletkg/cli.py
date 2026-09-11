"""Command-line interface for reproducible research runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .adapters import case_from_amlsim_csv, case_from_transaction_csv
from .evaluation import evaluate_retrieval, evaluate_risk_event_detection
from .experiments import run_amlsim_benchmarks
from .graph import build_atomic_relations
from .io import read_json, write_json
from .llm import LLMAugmentor
from .materializer import IndexedWindowPatternMatcher, PatternMatcher
from .models import CaseGraph, RiskPattern
from .patterns import validate_patterns
from .similarity import FusedGraphCaseMatcher, HierarchicalCaseMatcher, RiskCaseIndex


def _patterns(path: str | Path) -> list[RiskPattern]:
    raw = read_json(path)
    if isinstance(raw, dict) and "patterns" in raw:
        values = raw["patterns"]
    elif isinstance(raw, dict) and "pattern_id" in raw:
        values = [raw]
    else:
        values = raw
    if not isinstance(values, list):
        raise ValueError(
            "pattern file must contain one pattern, a list, or a patterns list"
        )
    patterns = [RiskPattern.from_dict(item) for item in values]
    validate_patterns(patterns)
    return patterns


def _case_graphs(directory: str | Path) -> list[CaseGraph]:
    """Load case graphs while ignoring summaries and other JSON artifacts."""

    cases = []
    for path in sorted(Path(directory).glob("*.json")):
        value = read_json(path)
        if not isinstance(value, dict) or "case_id" not in value:
            continue
        cases.append(CaseGraph.from_dict(value))
    return cases


def _materialize(
    case: CaseGraph,
    patterns: Sequence[RiskPattern],
    *,
    matcher_name: str = "windowed",
) -> tuple[list[dict], dict]:
    build_atomic_relations(case)
    matcher = (
        PatternMatcher()
        if matcher_name == "legacy"
        else IndexedWindowPatternMatcher()
    )
    traces = []
    diagnostics: dict[str, dict] = {}
    for pattern in patterns:
        risk_events, current_traces = matcher.match(case, pattern)
        case.risk_events.update({item.risk_event_id: item for item in risk_events})
        traces.extend(current_traces)
        if isinstance(matcher, IndexedWindowPatternMatcher):
            diagnostics[pattern.pattern_id] = matcher.last_diagnostics.to_dict()
    from .graph import build_risk_relations

    build_risk_relations(case)
    return [
        {
            "pattern_id": item.pattern_id,
            "anchor_entity_id": item.anchor_entity_id,
            "accepted": item.accepted,
            "reason": item.reason,
            "clause_event_ids": item.clause_event_ids,
            "metrics": item.metrics,
            "stage": item.stage,
            "window_start": item.window_start,
            "window_end": item.window_end,
        }
        for item in traces
    ], diagnostics


def _case_matcher(name: str) -> HierarchicalCaseMatcher:
    if name == "hierarchical":
        return HierarchicalCaseMatcher()
    return FusedGraphCaseMatcher()


def command_materialize(args: argparse.Namespace) -> int:
    case = CaseGraph.from_dict(read_json(args.case))
    traces, diagnostics = _materialize(
        case, _patterns(args.patterns), matcher_name=args.matcher
    )
    result = case.to_dict()
    result["materialization_trace"] = traces
    result["materialization_diagnostics"] = diagnostics
    write_json(args.output, result)
    print(json.dumps({"output": str(Path(args.output).resolve()), "risk_events": len(case.risk_events)}, ensure_ascii=False))
    return 0


def command_similarity(args: argparse.Namespace) -> int:
    query = CaseGraph.from_dict(read_json(args.query))
    candidate = CaseGraph.from_dict(read_json(args.candidate))
    result = _case_matcher(args.algorithm).compare(query, candidate)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


def command_rank(args: argparse.Namespace) -> int:
    query = CaseGraph.from_dict(read_json(args.query))
    candidates = _case_graphs(args.candidates)
    matcher = _case_matcher(args.algorithm)
    if args.candidate_pool_size is not None:
        results, diagnostics = RiskCaseIndex(candidates).search(
            query,
            matcher=matcher,
            top_k=args.top_k,
            candidate_pool_size=args.candidate_pool_size,
        )
        payload = {
            "algorithm": args.algorithm,
            "retrieval_diagnostics": diagnostics.to_dict(),
            "results": [item.to_dict() for item in results],
        }
    else:
        results = matcher.rank(query, candidates, top_k=args.top_k)
        payload = [item.to_dict() for item in results]
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    cases = _case_graphs(args.cases)
    relevance = read_json(args.relevance)
    metrics = evaluate_retrieval(cases, relevance, top_k=args.top_k)
    print(json.dumps(metrics.to_dict(), ensure_ascii=False, indent=2))
    return 0


def command_evaluate_events(args: argparse.Namespace) -> int:
    predicted_case = CaseGraph.from_dict(read_json(args.predicted))
    reference_case = CaseGraph.from_dict(read_json(args.reference))
    predicted = list(predicted_case.risk_events.values())
    reference = list(reference_case.risk_events.values())
    if args.pattern_id:
        predicted = [item for item in predicted if item.pattern_id == args.pattern_id]
        reference = [item for item in reference if item.pattern_id == args.pattern_id]
    metrics = evaluate_risk_event_detection(
        predicted,
        reference,
        minimum_atomic_iou=args.minimum_atomic_iou,
        require_same_type=not args.ignore_type,
        require_same_anchor=not args.ignore_anchor,
    )
    print(json.dumps(metrics.to_dict(), ensure_ascii=False, indent=2))
    return 0


def command_import_ibm(args: argparse.Namespace) -> int:
    case = case_from_transaction_csv(args.csv, case_id=args.case_id, limit=args.limit)
    write_json(args.output, case.to_dict())
    print(json.dumps({"output": str(Path(args.output).resolve()), "atomic_events": len(case.atomic_events)}, ensure_ascii=False))
    return 0


def command_import_amlsim(args: argparse.Namespace) -> int:
    case = case_from_amlsim_csv(
        args.transactions,
        args.nodes,
        case_id=args.case_id,
        limit=args.limit,
        base_time=args.base_time,
        step_seconds=args.step_seconds,
        build_relations=not args.skip_relations,
    )
    write_json(args.output, case.to_dict())
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "entities": len(case.entities),
                "atomic_events": len(case.atomic_events),
                "atomic_relations": len(case.atomic_relations),
            },
            ensure_ascii=False,
        )
    )
    return 0


def command_benchmark_amlsim(args: argparse.Namespace) -> int:
    result = run_amlsim_benchmarks(args.data_root, args.output)
    summary = []
    for dataset in result["datasets"]:
        best_name, best_metrics = max(
            dataset["algorithms"].items(),
            key=lambda item: (item[1]["f1"], item[0]),
        )
        summary.append(
            {
                "dataset": dataset["dataset"],
                "transactions": dataset["profile"]["transaction_rows"],
                "best_baseline": best_name,
                "precision": best_metrics["precision"],
                "recall": best_metrics["recall"],
                "f1": best_metrics["f1"],
            }
        )
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_extract_case(args: argparse.Namespace) -> int:
    text = Path(args.text).read_text(encoding="utf-8")
    case = LLMAugmentor().extract_case(case_id=args.case_id, text=text)
    build_atomic_relations(case)
    write_json(args.output, case.to_dict())
    return 0


def command_compile_pattern(args: argparse.Namespace) -> int:
    rule = Path(args.rule).read_text(encoding="utf-8")
    pattern = LLMAugmentor().compile_pattern(rule, samples=args.samples)
    write_json(args.output, pattern.to_dict())
    return 0


def command_demo(args: argparse.Namespace) -> int:
    root = Path(args.examples)
    patterns = _patterns(root / "patterns.json")
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    cases = []
    for source in sorted((root / "raw_cases").glob("*.json")):
        case = CaseGraph.from_dict(read_json(source))
        _materialize(case, patterns, matcher_name=args.matcher)
        write_json(output_root / source.name, case.to_dict())
        cases.append(case)
    query = next(item for item in cases if item.case_id == "CASE-A")
    ranked = _case_matcher(args.algorithm).rank(query, cases, top_k=5)
    relevance = read_json(root / "relevance.json")
    metrics = evaluate_retrieval(cases, relevance, top_k=2)
    summary = {
        "materialized_cases": len(cases),
        "query": query.case_id,
        "ranking": [item.to_dict() for item in ranked],
        "metrics_at_2": metrics.to_dict(),
        "output": str(output_root.resolve()),
    }
    write_json(output_root / "demo_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hamletkg",
        description="LLM-augmented atomic-to-risk event KG research prototype",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    materialize = subparsers.add_parser("materialize", help="materialize risk events")
    materialize.add_argument("--case", required=True)
    materialize.add_argument("--patterns", required=True)
    materialize.add_argument("--output", required=True)
    materialize.add_argument(
        "--matcher", choices=["windowed", "legacy"], default="windowed"
    )
    materialize.set_defaults(handler=command_materialize)

    similarity = subparsers.add_parser("similarity", help="compare two materialized cases")
    similarity.add_argument("--query", required=True)
    similarity.add_argument("--candidate", required=True)
    similarity.add_argument(
        "--algorithm", choices=["fused", "hierarchical"], default="fused"
    )
    similarity.set_defaults(handler=command_similarity)

    rank = subparsers.add_parser("rank", help="rank candidate cases")
    rank.add_argument("--query", required=True)
    rank.add_argument("--candidates", required=True)
    rank.add_argument("--top-k", type=int, default=5)
    rank.add_argument(
        "--algorithm", choices=["fused", "hierarchical"], default="fused"
    )
    rank.add_argument(
        "--candidate-pool-size",
        type=int,
        help="enable risk-type indexing and rerank at most this many cases",
    )
    rank.set_defaults(handler=command_rank)

    evaluate = subparsers.add_parser("evaluate", help="evaluate retrieval metrics")
    evaluate.add_argument("--cases", required=True)
    evaluate.add_argument("--relevance", required=True)
    evaluate.add_argument("--top-k", type=int, default=5)
    evaluate.set_defaults(handler=command_evaluate)

    evaluate_events = subparsers.add_parser(
        "evaluate-events", help="evaluate materialized risk-event instances"
    )
    evaluate_events.add_argument("--predicted", required=True)
    evaluate_events.add_argument("--reference", required=True)
    evaluate_events.add_argument("--pattern-id")
    evaluate_events.add_argument("--minimum-atomic-iou", type=float, default=0.5)
    evaluate_events.add_argument("--ignore-type", action="store_true")
    evaluate_events.add_argument("--ignore-anchor", action="store_true")
    evaluate_events.set_defaults(handler=command_evaluate_events)

    importer = subparsers.add_parser("import-ibm", help="import IBM AML transaction CSV")
    importer.add_argument("--csv", required=True)
    importer.add_argument("--case-id", required=True)
    importer.add_argument("--limit", type=int)
    importer.add_argument("--output", required=True)
    importer.set_defaults(handler=command_import_ibm)

    amlsim_importer = subparsers.add_parser(
        "import-amlsim", help="import an IBM AMLSim sample into an atomic-event graph"
    )
    amlsim_importer.add_argument("--transactions", required=True)
    amlsim_importer.add_argument("--nodes", required=True)
    amlsim_importer.add_argument("--case-id", required=True)
    amlsim_importer.add_argument("--limit", type=int)
    amlsim_importer.add_argument("--base-time", default="2017-01-01T00:00:00Z")
    amlsim_importer.add_argument("--step-seconds", type=int, default=3600)
    amlsim_importer.add_argument("--skip-relations", action="store_true")
    amlsim_importer.add_argument("--output", required=True)
    amlsim_importer.set_defaults(handler=command_import_amlsim)

    amlsim_benchmark = subparsers.add_parser(
        "benchmark-amlsim", help="run label-safe AMLSim graph-pattern baselines"
    )
    amlsim_benchmark.add_argument("--data-root", default="data/raw/amlsim")
    amlsim_benchmark.add_argument("--output", default="outputs/amlsim-baselines")
    amlsim_benchmark.set_defaults(handler=command_benchmark_amlsim)

    extract = subparsers.add_parser("extract-case", help="extract atomic events with an LLM")
    extract.add_argument("--case-id", required=True)
    extract.add_argument("--text", required=True)
    extract.add_argument("--output", required=True)
    extract.set_defaults(handler=command_extract_case)

    compile_pattern = subparsers.add_parser(
        "compile-pattern", help="compile an AML rule into the pattern DSL with an LLM"
    )
    compile_pattern.add_argument("--rule", required=True)
    compile_pattern.add_argument("--output", required=True)
    compile_pattern.add_argument(
        "--samples",
        type=int,
        default=1,
        help="compile independently this many times and report semantic agreement",
    )
    compile_pattern.set_defaults(handler=command_compile_pattern)

    demo = subparsers.add_parser("demo", help="run the offline reproducible demo")
    demo.add_argument("--examples", default="examples")
    demo.add_argument("--output", default="outputs/demo")
    demo.add_argument(
        "--matcher", choices=["windowed", "legacy"], default="windowed"
    )
    demo.add_argument(
        "--algorithm", choices=["fused", "hierarchical"], default="fused"
    )
    demo.set_defaults(handler=command_demo)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
