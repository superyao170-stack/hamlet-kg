"""Command-line interface for reproducible research runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Sequence

from .adapters import case_from_amlsim_csv, case_from_transaction_csv
from .evaluation import evaluate_retrieval, evaluate_risk_event_detection
from .experiments import run_amlsim_benchmarks
from .graph import build_atomic_relations
from .io import read_json, write_json
from .llm import LLMAugmentor, LLMConfig, LLMError, OpenAICompatibleClient
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
    case = case_from_transaction_csv(args.csv, case_id=args.case_id, limit=args.limit,
                                     build_relations=not args.skip_relations)
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
    case = LLMAugmentor(_llm_client(args)).extract_case(case_id=args.case_id, text=text)
    build_atomic_relations(case)
    write_json(args.output, case.to_dict())
    return 0


def command_compile_pattern(args: argparse.Namespace) -> int:
    rule = Path(args.rule).read_text(encoding="utf-8")
    pattern = LLMAugmentor(_llm_client(args)).compile_pattern(rule, samples=args.samples)
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


def _stream_config(args: argparse.Namespace):
    from .detection.stream import StreamConfig

    return StreamConfig(window_seconds=args.window_hours * 3600,
                        max_history=args.max_history, min_fan=args.min_fan,
                        ratio_min=args.ratio_min, ratio_max=args.ratio_max)


def _train_detection_case(args: argparse.Namespace, case: CaseGraph, *, source_path,
                          load_seconds: float, fixture_info=None) -> int:
    from .detection.runner import load_detection_policy, train_detection

    policy, provenance = load_detection_policy(args.policy) if args.policy else (None, None)

    result = train_detection(
        case, args.output, config=_stream_config(args), variants=args.variants,
        seed=args.seed, epochs=args.epochs, learning_rate=args.learning_rate,
        batch_size=args.batch_size, hidden_dim=args.hidden_dim, class_weight=args.class_weight,
        train_fraction=args.train_fraction, validation_fraction=args.validation_fraction,
        budget_fractions=args.budget_fractions, explain_top_k=args.explain_top_k,
        explanation_max_candidates=args.explanation_max_candidates,
        source_path=source_path, load_seconds=load_seconds, fixture_info=fixture_info,
        policy=policy, policy_provenance=provenance,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def command_detect_train(args: argparse.Namespace) -> int:
    from .detection.runner import load_detection_case

    begin = time.perf_counter()
    case = load_detection_case(csv_path=args.csv, case_path=args.case,
                               case_id=args.case_id, limit=args.limit)
    return _train_detection_case(args, case, source_path=args.csv or args.case,
                                 load_seconds=time.perf_counter() - begin)


def command_detect_demo(args: argparse.Namespace) -> int:
    from .detection.demo import write_demo_csv
    from .detection.runner import load_detection_case

    source = Path(args.output) / "smoke_transactions.csv"
    fixture_info = write_demo_csv(source, seed=args.seed, days=args.days)
    begin = time.perf_counter()
    case = load_detection_case(csv_path=source, case_id="SMOKE-ONLY")
    return _train_detection_case(args, case, source_path=source,
                                 load_seconds=time.perf_counter() - begin,
                                 fixture_info=fixture_info)


def command_detect_score(args: argparse.Namespace) -> int:
    from .detection.runner import load_detection_case, score_detection

    case = load_detection_case(csv_path=args.csv, case_path=args.case,
                               case_id=args.case_id, limit=args.limit)
    result = score_detection(case, args.model, args.output, after=args.after,
                             threshold=args.threshold, explain_top_k=args.explain_top_k,
                             explanation_max_candidates=args.explanation_max_candidates)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def command_audit_data(args: argparse.Namespace) -> int:
    from .detection.audit import audit_detection_case
    from .detection.runner import load_detection_case

    case = load_detection_case(csv_path=args.csv, case_path=args.case, case_id=args.case_id, limit=args.limit)
    result = audit_detection_case(case, source_path=args.csv or args.case)
    result["input_row_limit"] = args.limit
    write_json(args.output, result)
    print(json.dumps({"output": str(Path(args.output).resolve()), "counts": result["counts"],
                      "labels": result["labels"], "risks": result["risks"]}, ensure_ascii=False, indent=2))
    return 0


def command_detect_benchmark(args: argparse.Namespace) -> int:
    from .detection.benchmark import benchmark_detection
    from .detection.runner import load_detection_case, load_detection_policy

    case = load_detection_case(csv_path=args.csv, case_path=args.case, case_id=args.case_id, limit=args.limit)
    policy, provenance = load_detection_policy(args.policy) if args.policy else (None, None)
    result = benchmark_detection(
        case, args.output, seeds=args.seeds, variants=args.variants, config=_stream_config(args),
        policy=policy, policy_provenance=provenance, source_path=args.csv or args.case,
        input_row_limit=args.limit, data_role=args.data_role,
        epochs=args.epochs, learning_rate=args.learning_rate, batch_size=args.batch_size,
        hidden_dim=args.hidden_dim, class_weight=args.class_weight,
        train_fraction=args.train_fraction, validation_fraction=args.validation_fraction,
        budget_fractions=args.budget_fractions,
        tree_estimators=args.tree_estimators, tree_learning_rate=args.tree_learning_rate,
        tree_leaves=args.tree_leaves,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def command_prepare_data(args: argparse.Namespace) -> int:
    from .detection.dataset import prepare_detection_dataset
    result = prepare_detection_dataset(args.csv, args.output, case_id=args.case_id, limit=args.limit,
                                       batch_size=args.batch_size, after=args.after, before=args.before)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def command_cache_data(args: argparse.Namespace) -> int:
    from .detection.dataset import PreparedDataset
    from .detection.cache import build_sample_cache
    from .detection.runner import load_detection_policy
    dataset = PreparedDataset(args.dataset, after=args.after, before=args.before)
    policy, provenance = load_detection_policy(args.policy) if args.policy else (None, None)
    result = build_sample_cache(dataset, args.output, config=_stream_config(args), policy=policy,
                                policy_provenance=provenance, train_fraction=args.train_fraction,
                                validation_fraction=args.validation_fraction)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def command_detect_cached(args: argparse.Namespace) -> int:
    from .detection.disk_runner import benchmark_cached
    result = benchmark_cached(args.cache, args.output, dataset_path=args.dataset, seeds=args.seeds,
                              variants=args.variants, epochs=args.epochs, learning_rate=args.learning_rate,
                              batch_size=args.batch_size, hidden_dim=args.hidden_dim, class_weight=args.class_weight,
                              shuffle_buffer_batches=args.shuffle_buffer_batches,
                              neural_scaling=args.neural_scaling,
                              tree_estimators=args.tree_estimators, tree_learning_rate=args.tree_learning_rate,
                              tree_leaves=args.tree_leaves, budget_fractions=args.budget_fractions,
                              data_role=args.data_role)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def command_detect_cache_score(args: argparse.Namespace) -> int:
    from .detection.disk_runner import score_cached
    result = score_cached(args.cache, args.model, args.output, split=args.split,
                          dataset_path=args.dataset, batch_size=args.batch_size)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _llm_client(args: argparse.Namespace) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(LLMConfig.from_environment(getattr(args, "env_file", None)))


def command_llm_check(args: argparse.Namespace) -> int:
    client = _llm_client(args)
    try:
        result = {"ok": True, **client.check(generate=args.generate)}
        code = 0
    except LLMError as error:
        result = {"ok": False, "endpoint": client.config.base_url,
                  "requested_model": client.config.model, "error": str(error),
                  "generation_requested": args.generate}
        code = 2
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return code


def command_compile_detection_rules(args: argparse.Namespace) -> int:
    rule_text = Path(args.rule).read_text(encoding="utf-8")
    policy, quality = LLMAugmentor(_llm_client(args)).compile_detection_policy(
        rule_text, samples=args.samples)
    # Keep provenance outside the strict executable policy schema.
    quality["rule_text_sha256"] = hashlib.sha256(rule_text.encode("utf-8")).hexdigest()
    write_json(args.output, {"policy": policy.to_dict(), "compilation": quality})
    print(json.dumps({"output": str(Path(args.output).resolve()),
                      "rules": len(policy.rules), "operators": [rule.operator for rule in policy.rules],
                      "quality": quality}, ensure_ascii=False, indent=2))
    return 0


def command_describe_alerts(args: argparse.Namespace) -> int:
    from .detection.narrative import generate_evidence_report

    source = Path(args.evidence)
    source_bytes = source.read_bytes()
    result = generate_evidence_report(json.loads(source_bytes), _llm_client(args),
                                      max_transactions=args.max_transactions)
    result["evidence_source"] = {"file": str(source.resolve()),
                                 "sha256": hashlib.sha256(source_bytes).hexdigest()}
    output = Path(args.output)
    markdown_path, json_path = output / "report.md", output / "report.json"
    markdown_stage, json_stage = output / ".report.md.pending", output / ".report.json.pending"
    try:
        output.mkdir(parents=True, exist_ok=True)
        if markdown_path.is_dir() or json_path.is_dir():
            raise OSError("report destination is a directory")
        markdown_stage.write_text(result["markdown"], encoding="utf-8")
        json_stage.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        markdown_stage.replace(markdown_path)
        # Publish the validated success marker only after Markdown is written.
        json_stage.replace(json_path)
    except OSError:
        raise LLMError("unable to write the evidence report files") from None
    finally:
        for stage in (markdown_stage, json_stage):
            try:
                stage.unlink(missing_ok=True)
            except OSError:
                # Cleanup must not replace the original controlled write error.
                pass
    print(json.dumps({"output": str(output.resolve()), "validated": result["validated"],
                      "coverage": result["coverage"], "model": result["model"]},
                     ensure_ascii=False, indent=2))
    return 0


def _detection_input(parser: argparse.ArgumentParser) -> None:
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--csv", help="IBM AML transaction CSV, with original or normalized headers")
    inputs.add_argument("--case", help="CaseGraph JSON with transaction labels in metadata for training")
    parser.add_argument("--case-id", default="IBM-AML")
    parser.add_argument("--limit", type=int, help="CSV debug row limit; not a publication sampling protocol")


def _detection_training_options(parser: argparse.ArgumentParser, *, benchmark: bool = False) -> None:
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy", help="validated detection policy JSON, frozen with the trained model")
    choices = ["base", "flat", "event", "no_role", "no_time", "untyped_event"]
    if benchmark:
        choices.extend(["lgbm_base", "lgbm_flat"])
    parser.add_argument("--variants", nargs="+", default=(["base", "flat", "event", "lgbm_base", "lgbm_flat"] if benchmark else ["event", "base", "flat"]),
                        choices=choices)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="balanced",
                        help="balanced neural BCE or tree sample weights are computed from training labels only")
    parser.add_argument("--hidden-dim", type=int, default=32)
    if benchmark:
        parser.add_argument("--seeds", nargs="+", type=int, default=[13, 42, 87])
        parser.add_argument("--tree-estimators", type=int, default=100)
        parser.add_argument("--tree-learning-rate", type=float, default=.05)
        parser.add_argument("--tree-leaves", type=int, default=15)
        parser.add_argument("--data-role", choices=["smoke", "pilot", "public_dataset_run"], default="pilot")
    else:
        parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=.6)
    parser.add_argument("--validation-fraction", type=float, default=.2)
    parser.add_argument("--window-hours", type=float, default=24)
    parser.add_argument("--max-history", type=int, default=32)
    parser.add_argument("--min-fan", type=int, default=3)
    parser.add_argument("--ratio-min", type=float, default=.8)
    parser.add_argument("--ratio-max", type=float, default=1.2)
    parser.add_argument("--budget-fractions", nargs="+", type=float, default=[.001, .005, .01])
    if not benchmark:
        parser.add_argument("--explain-top-k", type=int, default=3,
                            help="audit top-scoring test transactions; 0 disables expensive offline audits")
        parser.add_argument("--explanation-max-candidates", type=int, default=3)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hamletkg",
        description="Knowledge-guided temporal event graph AML detection research",
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
    importer.add_argument("--skip-relations", action="store_true",
                          help="skip legacy offline atomic relation construction")
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
    extract.add_argument("--env-file", help="literal local LLM configuration; process environment takes precedence")
    extract.set_defaults(handler=command_extract_case)

    compile_pattern = subparsers.add_parser(
        "compile-pattern", help="compile an AML rule into the pattern DSL with an LLM"
    )
    compile_pattern.add_argument("--rule", required=True)
    compile_pattern.add_argument("--output", required=True)
    compile_pattern.add_argument("--env-file")
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

    detection_demo = subparsers.add_parser("detect-demo", help="train on a generated smoke fixture, not research data")
    _detection_training_options(detection_demo)
    detection_demo.add_argument("--days", type=int, default=24)
    detection_demo.set_defaults(handler=command_detect_demo)

    detection_train = subparsers.add_parser("detect-train", help="chronological transaction detection training and ablations")
    _detection_input(detection_train)
    _detection_training_options(detection_train)
    detection_train.set_defaults(handler=command_detect_train)

    detection_score = subparsers.add_parser("detect-score", help="score a causally replayed transaction stream using a frozen model")
    _detection_input(detection_score)
    detection_score.add_argument("--model", required=True, help=".pt + .json neural checkpoint, or .txt + .json + .settings.json LightGBM checkpoint")
    detection_score.add_argument("--output", required=True)
    detection_score.add_argument("--after", help="ISO timestamp; earlier input is context, output starts here")
    detection_score.add_argument("--threshold", type=float, help="override the validation threshold")
    detection_score.add_argument("--explain-top-k", type=int, default=3)
    detection_score.add_argument("--explanation-max-candidates", type=int, default=3)
    detection_score.set_defaults(handler=command_detect_score)

    audit = subparsers.add_parser("audit-data", help="audit transaction quality and provenance without dropping rows")
    _detection_input(audit)
    audit.add_argument("--output", required=True)
    audit.set_defaults(handler=command_audit_data)

    benchmark = subparsers.add_parser("detect-benchmark", help="compare neural and LightGBM controls on one chronological split across seeds")
    _detection_input(benchmark)
    _detection_training_options(benchmark, benchmark=True)
    benchmark.set_defaults(handler=command_detect_benchmark)

    prepare = subparsers.add_parser("prepare-data", help="validate CSV and build a disk time index without a CaseGraph")
    prepare.add_argument("--csv", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--case-id", default="IBM-AML")
    prepare.add_argument("--limit", type=int, help="first N original source rows; diagnostic use")
    prepare.add_argument("--batch-size", type=int, default=2000)
    prepare.add_argument("--after", help="include timestamps >= this UTC time")
    prepare.add_argument("--before", help="include timestamps < this UTC time")
    prepare.set_defaults(handler=command_prepare_data)

    cache = subparsers.add_parser("cache-data", help="build causal features once; labels remain in a separate table")
    cache.add_argument("--dataset", required=True)
    cache.add_argument("--output", required=True)
    cache.add_argument("--policy")
    cache.add_argument("--after", help="start selected view with empty history; no earlier warmup")
    cache.add_argument("--before", help="exclusive end of the selected UTC view")
    cache.add_argument("--train-fraction", type=float, default=.6)
    cache.add_argument("--validation-fraction", type=float, default=.2)
    cache.add_argument("--window-hours", type=float, default=24)
    cache.add_argument("--max-history", type=int, default=32)
    cache.add_argument("--min-fan", type=int, default=3)
    cache.add_argument("--ratio-min", type=float, default=.8)
    cache.add_argument("--ratio-max", type=float, default=1.2)
    cache.set_defaults(handler=command_cache_data)

    cached = subparsers.add_parser("detect-cached-benchmark", help="batch training and disk evaluation on one frozen causal cache")
    cached.add_argument("--cache", required=True)
    cached.add_argument("--dataset", help="matching prepared source for original row IDs in predictions")
    cached.add_argument("--output", required=True)
    cached.add_argument("--seeds", nargs="+", type=int, default=[13, 42, 87])
    cached.add_argument("--variants", nargs="+", default=["base", "flat", "event", "lgbm_base", "lgbm_flat"],
                        choices=["base", "flat", "event", "no_role", "no_time", "untyped_event", "lgbm_base", "lgbm_flat"])
    cached.add_argument("--epochs", type=int, default=20)
    cached.add_argument("--learning-rate", type=float, default=.001)
    cached.add_argument("--batch-size", type=int, default=64)
    cached.add_argument("--hidden-dim", type=int, default=32)
    cached.add_argument("--class-weight", choices=["none", "balanced"], default="balanced")
    cached.add_argument("--shuffle-buffer-batches", type=int, default=0, help="bounded minibatch shuffle; zero means chronological training order")
    cached.add_argument("--neural-scaling", choices=["standard", "fixed_periodic"], default="standard",
                        help="fixed_periodic leaves hour_sin/hour_cos in [-1,1]; other features use training statistics")
    cached.add_argument("--tree-estimators", type=int, default=100)
    cached.add_argument("--tree-learning-rate", type=float, default=.05)
    cached.add_argument("--tree-leaves", type=int, default=15)
    cached.add_argument("--budget-fractions", nargs="+", type=float, default=[.001, .005, .01])
    cached.add_argument("--data-role", choices=["smoke", "pilot", "public_dataset_run"], default="pilot")
    cached.set_defaults(handler=command_detect_cached)

    cached_score = subparsers.add_parser("detect-cache-score", help="score a compatible causal cache with frozen weights and validation threshold")
    cached_score.add_argument("--cache", required=True)
    cached_score.add_argument("--dataset")
    cached_score.add_argument("--model", required=True)
    cached_score.add_argument("--output", required=True)
    cached_score.add_argument("--split", choices=["train", "validation", "test", "all"], default="test")
    cached_score.add_argument("--batch-size", type=int, default=64)
    cached_score.set_defaults(handler=command_detect_cache_score)

    llm_check = subparsers.add_parser("llm-check", help="check model listing; optionally verify a tiny JSON completion")
    llm_check.add_argument("--env-file")
    llm_check.add_argument("--generate", action="store_true", help="make a small potentially billable JSON generation request")
    llm_check.add_argument("--output", help="save sanitized connectivity status JSON")
    llm_check.set_defaults(handler=command_llm_check)

    detection_rules = subparsers.add_parser("compile-detection-rules", help="compile natural language to a validated four-operator detection policy")
    detection_rules.add_argument("--rule", required=True)
    detection_rules.add_argument("--output", required=True)
    detection_rules.add_argument("--samples", type=int, default=1)
    detection_rules.add_argument("--env-file")
    detection_rules.set_defaults(handler=command_compile_detection_rules)

    narrative = subparsers.add_parser("describe-alerts", help="organize an evidence-limited report with optional LLM fact selection")
    narrative.add_argument("--evidence", required=True)
    narrative.add_argument("--output", required=True, help="directory for validated report.json and report.md")
    narrative.add_argument("--max-transactions", type=int, default=3)
    narrative.add_argument("--env-file")
    narrative.set_defaults(handler=command_describe_alerts)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (ValueError, FileNotFoundError, ImportError, LLMError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
