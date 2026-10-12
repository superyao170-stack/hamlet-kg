from __future__ import annotations

import copy
import csv
import io
import json
import math
import unittest

from hamletkg.detection.benchmark_summary import (
    benchmark_summary_rows,
    render_benchmark_markdown,
    summarize_benchmark_runs,
)


def variant_metrics(ap=0.5, recall=0.5, threshold=0.4, fractions=(0.01, 0.02)):
    split = {
        "average_precision": ap, "precision": 0.75, "recall": recall,
        "f1": 0.6, "false_positives_per_1000": 2.0, "threshold": threshold,
    }
    return {
        "validation": copy.deepcopy(split), "test": copy.deepcopy(split),
        "test_daily_budgets": [{
            "budget_fraction": fraction, "recall": recall, "precision": 0.8,
            "review_alerts": 10, "true_positive": 8,
            "daily": [{"date": "2026-01-01", "selected_event_ids": ["a"]}],
        } for fraction in fractions],
        "training_history": [0.7, 0.6],
    }


class DetectionBenchmarkSummaryTests(unittest.TestCase):
    def test_aggregates_sample_standard_deviation_and_keeps_each_threshold(self):
        runs = [
            {"seed": 9, "metrics": {"event": variant_metrics(ap=0.8, threshold=0.9)},
             "run_directory": "/tmp/seed-9"},
            {"seed": 2, "metrics": {"event": variant_metrics(ap=0.2, threshold=0.1)}},
        ]
        original = copy.deepcopy(runs)
        summary = summarize_benchmark_runs(runs)
        self.assertEqual([2, 9], summary["seeds"])
        self.assertEqual(2, summary["seed_count"])
        stats = summary["variants"]["event"]["test"]["average_precision"]
        self.assertAlmostEqual(0.5, stats["mean"])
        self.assertAlmostEqual(math.sqrt(0.18), stats["std"])
        self.assertEqual((2, 0), (stats["n"], stats["n_missing"]))
        self.assertEqual([2, 9], stats["seeds"])
        thresholds = summary["variants"]["event"]["thresholds"]["validation"]
        self.assertEqual([{"seed": 2, "value": 0.1}, {"seed": 9, "value": 0.9}], thresholds)
        self.assertEqual(original, runs)
        self.assertEqual("/tmp/seed-9", summary["runs"][1]["run_directory"])
        self.assertEqual(["a"], summary["runs"][0]["metrics"]["event"]["test_daily_budgets"][0]["daily"][0]["selected_event_ids"])
        summary["runs"][0]["metrics"]["event"]["test"]["threshold"] = 999
        self.assertEqual(original, runs)

    def test_missing_values_stay_missing_and_zero_remains_valid(self):
        first = variant_metrics(ap=None, recall=None)
        first["test"]["precision"] = None
        first["test"]["f1"] = None
        del first["test"]["false_positives_per_1000"]
        first["test_daily_budgets"][0]["review_alerts"] = None
        second = variant_metrics(ap=0.0, recall=0.0)
        summary = summarize_benchmark_runs([
            {"seed": 1, "metrics": {"event": first}},
            {"seed": 2, "metrics": {"event": second}},
            {"seed": 3, "metrics": {"base": variant_metrics()}},
        ])
        data = summary["variants"]["event"]
        stats = data["test"]["average_precision"]
        self.assertEqual(0.0, stats["mean"])
        self.assertIsNone(stats["std"])
        self.assertEqual((1, 2), (stats["n"], stats["n_missing"]))
        self.assertEqual([2], stats["seeds"])
        self.assertEqual([None, 0.0, None], [item["value"] for item in stats["values"]])
        selected = data["test_daily_budgets"][0]["review_alerts"]
        self.assertEqual((10, 1, 2), (selected["mean"], selected["n"], selected["n_missing"]))
        self.assertIsNone(summary["runs"][0]["metrics"]["event"]["test"]["average_precision"])

    def test_single_seed_and_all_missing_metrics_have_no_std(self):
        data = variant_metrics()
        data["validation"] = None
        summary = summarize_benchmark_runs([{"seed": 0, "metrics": {"event": data}}])
        stats = summary["variants"]["event"]["validation"]["recall"]
        self.assertEqual({"mean": None, "std": None, "n": 0, "n_missing": 1,
                          "seeds": [], "values": [{"seed": 0, "value": None}]}, stats)
        self.assertIsNone(summary["variants"]["event"]["test"]["recall"]["std"])

    def test_every_requested_metric_and_budget_count_is_aggregated(self):
        first, second = variant_metrics(), variant_metrics()
        names = ("average_precision", "precision", "recall", "f1", "false_positives_per_1000")
        for split in ("validation", "test"):
            for name in names:
                first[split][name] = 0.2
                second[split][name] = 0.4
        for budget in first["test_daily_budgets"]:
            budget.update(recall=0.2, precision=0.4, review_alerts=20, true_positive=8)
        for budget in second["test_daily_budgets"]:
            budget.update(recall=0.4, precision=0.6, review_alerts=40, true_positive=24)
        data = summarize_benchmark_runs([
            {"seed": 1, "metrics": {"event": first}},
            {"seed": 2, "metrics": {"event": second}},
        ])["variants"]["event"]
        for split in ("validation", "test"):
            self.assertEqual(set(names), set(data[split]))
            for name in names:
                with self.subTest(split=split, metric=name):
                    self.assertAlmostEqual(0.3, data[split][name]["mean"])
                    self.assertAlmostEqual(math.sqrt(0.02), data[split][name]["std"])
        for budget in data["test_daily_budgets"]:
            for name, expected in (("recall", 0.3), ("precision", 0.5),
                                   ("review_alerts", 30), ("true_positive", 16)):
                with self.subTest(fraction=budget["budget_fraction"], metric=name):
                    self.assertAlmostEqual(expected, budget[name]["mean"])
                    self.assertEqual((2, 0), (budget[name]["n"], budget[name]["n_missing"]))

    def test_paired_deltas_use_common_valid_seeds_instead_of_unpaired_means(self):
        summary = summarize_benchmark_runs([
            {"seed": 1, "metrics": {"event": variant_metrics(ap=0.8, recall=0.9),
                                    "base": variant_metrics(ap=0.3, recall=0.6)}},
            {"seed": 2, "metrics": {"event": variant_metrics(ap=0.6, recall=0.5),
                                    "base": variant_metrics(ap=0.5, recall=0.4)}},
            {"seed": 3, "metrics": {"event": variant_metrics(ap=None, recall=None),
                                    "base": variant_metrics(ap=0.2, recall=0.4)}},
            {"seed": 4, "metrics": {"event": variant_metrics(ap=1.0)}},
            {"seed": 5, "metrics": {"base": variant_metrics(ap=0.0)}},
        ])
        pair = summary["paired_deltas"]["base"]
        self.assertEqual([1, 2, 3], pair["common_seeds"])
        for split in ("validation", "test"):
            ap = pair[f"{split}_average_precision"]
            self.assertAlmostEqual(0.3, ap["mean"])
            self.assertAlmostEqual(math.sqrt(0.08), ap["std"])
            self.assertEqual((2, 1), (ap["n"], ap["n_missing"]))
            self.assertEqual([1, 2], ap["seeds"])
            self.assertIsNone(ap["values"][2]["value"])
        recall = pair["test_daily_budget_recall"][0]["stats"]
        self.assertAlmostEqual(0.2, recall["mean"])
        self.assertAlmostEqual(math.sqrt(0.02), recall["std"])
        self.assertEqual((2, 1), (recall["n"], recall["n_missing"]))

    def test_all_supported_baselines_and_empty_common_seed_set(self):
        metrics = {name: variant_metrics() for name in ("event", "base", "flat", "lgbm_base", "lgbm_flat")}
        summary = summarize_benchmark_runs([{"seed": 1, "metrics": metrics}])
        self.assertEqual(["base", "flat", "lgbm_base", "lgbm_flat"], list(summary["paired_deltas"]))
        self.assertEqual(0, summary["paired_deltas"]["lgbm_flat"]["test_average_precision"]["mean"])
        self.assertEqual({}, summarize_benchmark_runs([{"seed": 1, "metrics": {"base": variant_metrics()}}])["paired_deltas"])
        disjoint = summarize_benchmark_runs([
            {"seed": 1, "metrics": {"event": variant_metrics()}},
            {"seed": 2, "metrics": {"base": variant_metrics()}},
        ])
        pair = disjoint["paired_deltas"]["base"]
        self.assertEqual([], pair["common_seeds"])
        self.assertEqual((None, None, 0, 0), tuple(pair["test_average_precision"][name] for name in ("mean", "std", "n", "n_missing")))

    def test_budget_fraction_order_is_irrelevant_and_missing_lists_are_preserved(self):
        first = variant_metrics(fractions=(0.02, 0.01))
        second = variant_metrics()
        second["test_daily_budgets"] = None
        summary = summarize_benchmark_runs([
            {"seed": 1, "metrics": {"event": first, "base": variant_metrics()}},
            {"seed": 2, "metrics": {"event": second}},
        ])
        budgets = summary["variants"]["event"]["test_daily_budgets"]
        self.assertEqual([0.01, 0.02], [entry["budget_fraction"] for entry in budgets])
        self.assertEqual((1, 1), (budgets[0]["recall"]["n"], budgets[0]["recall"]["n_missing"]))
        self.assertIsNone(summary["runs"][1]["metrics"]["event"]["test_daily_budgets"])

    def test_rejects_duplicate_seeds_and_inconsistent_or_duplicate_fractions(self):
        with self.assertRaisesRegex(ValueError, "duplicate seed"):
            summarize_benchmark_runs([{"seed": 1, "metrics": {"event": variant_metrics()}}] * 2)
        for fractions in ((0.01,), (), (0.01, 0.03)):
            with self.subTest(fractions=fractions), self.assertRaisesRegex(ValueError, "consistent"):
                summarize_benchmark_runs([
                    {"seed": 1, "metrics": {"event": variant_metrics()}},
                    {"seed": 2, "metrics": {"event": variant_metrics(fractions=fractions)}},
                ])
        with self.assertRaisesRegex(ValueError, "duplicate budget_fraction"):
            summarize_benchmark_runs([{"seed": 1, "metrics": {"event": variant_metrics(fractions=(0.01, 0.01))}}])

    def test_rejects_malformed_input_and_nonfinite_or_boolean_numbers(self):
        bad_inputs = ([], "bad", [{"seed": True, "metrics": {}}],
                      [{"seed": 1.0, "metrics": {}}], [{"seed": 1, "metrics": []}],
                      [{"seed": 1, "metrics": {}}], [{"seed": 1, "metrics": {" event": {}}}],
                      [{"seed": 1, "metrics": {"event": []}}],
                      [{"seed": 1, "metrics": {"event": {}}, "run_directory": 12}])
        for runs in bad_inputs:
            with self.subTest(runs=runs), self.assertRaises(ValueError):
                summarize_benchmark_runs(runs)
        for invalid in (math.nan, math.inf, -math.inf, True, "0.4", 10 ** 400):
            with self.subTest(value=invalid), self.assertRaisesRegex(ValueError, "finite number"):
                summarize_benchmark_runs([{"seed": 1, "metrics": {"event": variant_metrics(ap=invalid)}}])
        for invalid in (None, -0.1, 1.1):
            with self.subTest(fraction=invalid), self.assertRaisesRegex(ValueError, "budget_fraction"):
                summarize_benchmark_runs([{"seed": 1, "metrics": {"event": variant_metrics(fractions=(invalid,))}}])

    def test_renderers_have_stable_csv_schema_and_json_nulls(self):
        summary = summarize_benchmark_runs([{"seed": 7, "metrics": {
            "event": variant_metrics(ap=None, threshold=0.15), "base": variant_metrics(),
        }}])
        encoded = json.dumps(summary, allow_nan=False)
        self.assertEqual(summary, json.loads(encoded))
        rows = benchmark_summary_rows(summary)
        self.assertEqual(rows, benchmark_summary_rows(summary))
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        parsed = list(csv.DictReader(io.StringIO(stream.getvalue())))
        event_ap = next(row for row in parsed if row["kind"] == "aggregate" and
                        row["variant"] == "event" and row["split"] == "test" and row["metric"] == "average_precision")
        self.assertEqual("", event_ap["mean"])
        self.assertEqual("1", event_ap["n_missing"])
        threshold = next(row for row in rows if row["kind"] == "threshold" and row["variant"] == "event")
        self.assertEqual((7, 0.15, None), (threshold["seed"], threshold["value"], threshold["mean"]))
        markdown = render_benchmark_markdown(summary)
        self.assertIn("optimization randomness", markdown)
        self.assertIn("not independent test datasets", markdown)
        self.assertIn("do not establish significance", markdown)
        self.assertIn("| event | validation | 7 | 0.15 |", markdown)
        self.assertIn("event - base", markdown)
        self.assertNotIn("p-value", markdown)
        self.assertEqual(markdown, render_benchmark_markdown(summary))


if __name__ == "__main__":
    unittest.main()
