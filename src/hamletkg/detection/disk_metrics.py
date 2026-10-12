"""Exact grouped-score evaluation using SQLite instead of all-score lists."""
from __future__ import annotations

import csv
import json
import math
import sqlite3
from numbers import Integral, Real
from pathlib import Path
from datetime import timezone
from typing import Sequence

from ..models import iso_time, parse_time
from .metrics import _confusion_metrics


class DiskScores:
    def __init__(self, path: str | Path, *, create: bool = False):
        self.path = Path(path)
        if create:
            if self.path.exists():
                raise ValueError("score database already exists")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.path)
            self.connection.executescript("""
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=NORMAL;
                PRAGMA temp_store=FILE;
                CREATE TABLE predictions(
                    variant TEXT NOT NULL, split TEXT NOT NULL,
                    event_id TEXT NOT NULL, time_us INTEGER NOT NULL,
                    timestamp TEXT NOT NULL, day TEXT NOT NULL,
                    score REAL NOT NULL CHECK(score >= 0 AND score <= 1),
                    label INTEGER CHECK(label IN (0,1)), source_record_id TEXT,
                    candidate_events INTEGER NOT NULL,
                    PRIMARY KEY(variant,split,event_id));
                CREATE INDEX score_order ON predictions(variant,split,score DESC);
                CREATE INDEX daily_order ON predictions(variant,split,day,score DESC,event_id);
            """)
        else:
            self.connection = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)
            self.connection.execute("PRAGMA temp_store=FILE")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.connection.close()

    def commit(self):
        self.connection.commit()

    def add_batch(self, variant: str, split: str, samples: Sequence, scores: Sequence,
                  labels: Sequence[int | None], source_record_ids: Sequence[str | None] | None = None):
        if any(not isinstance(value, str) or not value.strip() for value in (variant, split)):
            raise ValueError("variant and split must be nonempty strings")
        if not samples or len(samples) != len(scores) or len(samples) != len(labels):
            raise ValueError("score batches must be nonempty and aligned")
        sources = [None] * len(samples) if source_record_ids is None else source_record_ids
        if len(sources) != len(samples):
            raise ValueError("source IDs must be aligned with samples")
        rows = []
        for sample, score, label, source in zip(samples, scores, labels, sources):
            if not isinstance(sample.event_id, str) or not sample.event_id.strip():
                raise ValueError("event IDs must be nonempty strings")
            if isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("scores must be finite values in [0,1]")
            if label is not None and (isinstance(label, bool) or not isinstance(label, Integral) or label not in (0, 1)):
                raise ValueError("labels must be binary integers or unknown")
            timestamp = parse_time(sample.timestamp).astimezone(timezone.utc)
            epoch = timestamp - parse_time("1970-01-01T00:00:00Z")
            microseconds = (epoch.days * 86400 + epoch.seconds) * 1000000 + epoch.microseconds
            rows.append((variant, split, sample.event_id, microseconds, iso_time(timestamp),
                         timestamp.date().isoformat(), float(score), int(label) if label is not None else None, source, len(sample.events)))
        self.connection.executemany("INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?)", rows)

    def counts(self, variant: str, split: str) -> dict:
        count, labeled, positive = self.connection.execute(
            "SELECT count(*),count(label),coalesce(sum(label),0) FROM predictions WHERE variant=? AND split=?",
            (variant, split)).fetchone()
        return {"total": count, "labeled": labeled, "positive": positive, "unlabeled": count - labeled}

    def _complete_counts(self, variant: str, split: str):
        counts = self.counts(variant, split)
        if not counts["total"] or counts["unlabeled"]:
            raise ValueError("evaluation requires nonempty, completely labeled candidates")
        return counts["total"], counts["positive"]

    def _groups(self, variant: str, split: str):
        return self.connection.execute(
            "SELECT score,count(*),sum(label) FROM predictions WHERE variant=? AND split=? GROUP BY score ORDER BY score DESC",
            (variant, split))

    def choose_threshold(self, variant: str, split: str = "validation") -> float:
        total, positive = self._complete_counts(variant, split)
        if positive == 0 or positive == total:
            raise ValueError("validation needs both label classes")
        best = math.nextafter(1.0, math.inf)
        best_positive, best_denominator, seen, tp = 0, 1, 0, 0
        for score, count, group_positive in self._groups(variant, split):
            seen += count
            tp += group_positive
            denominator = positive + seen
            if tp * best_denominator > best_positive * denominator:
                best, best_positive, best_denominator = score, tp, denominator
        return best

    def binary_metrics(self, variant: str, split: str, threshold: float) -> dict:
        if isinstance(threshold, bool) or not isinstance(threshold, Real) or not math.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold must be finite and nonnegative")
        total, positive = self._complete_counts(variant, split)
        alerts, tp = self.connection.execute(
            "SELECT count(*),coalesce(sum(label),0) FROM predictions WHERE variant=? AND split=? AND score>=?",
            (variant, split, threshold)).fetchone()
        ap = None
        if positive:
            ap, seen, accumulated = 0.0, 0, 0
            for _, count, group_positive in self._groups(variant, split):
                seen += count
                accumulated += group_positive
                ap += group_positive / positive * accumulated / seen
        return {**_confusion_metrics(total, positive, alerts, tp), "average_precision": ap, "threshold": threshold}

    def daily_budget_metrics(self, variant: str, split: str, fraction: float, *, selected_ids_limit: int = 1000) -> dict:
        if isinstance(fraction, bool) or not isinstance(fraction, Real) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("budget fraction must be finite in [0,1]")
        if type(selected_ids_limit) is not int or selected_ids_limit < 0:
            raise ValueError("selected_ids_limit must be a nonnegative integer")
        total, positive = self._complete_counts(variant, split)
        daily, total_alerts, total_tp = [], 0, 0
        dates = self.connection.execute(
            "SELECT day,count(*),sum(label) FROM predictions WHERE variant=? AND split=? GROUP BY day ORDER BY day",
            (variant, split))
        for day, count, day_positive in dates:
            budget = math.floor(count * fraction)
            tp = self.connection.execute(
                "SELECT coalesce(sum(label),0) FROM (SELECT label FROM predictions WHERE variant=? AND split=? AND day=? ORDER BY score DESC,event_id LIMIT ?)",
                (variant, split, day, budget)).fetchone()[0]
            selected = [row[0] for row in self.connection.execute(
                "SELECT event_id FROM predictions WHERE variant=? AND split=? AND day=? ORDER BY score DESC,event_id LIMIT ?",
                (variant, split, day, min(budget, selected_ids_limit)))]
            item = {**_confusion_metrics(count, day_positive, budget, tp), "date": day,
                    "budget": budget, "review_alerts": budget, "selected_event_ids": selected,
                    "selected_event_ids_omitted": budget - len(selected)}
            daily.append(item)
            total_alerts += budget
            total_tp += tp
        return {**_confusion_metrics(total, positive, total_alerts, total_tp), "budget_fraction": fraction,
                "review_alerts": total_alerts, "daily": daily}

    def export(self, variant: str, split: str, threshold: float, directory: str | Path):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f"predictions_{variant}.csv").open("w", newline="", encoding="utf-8") as handle, \
                (directory / f"alerts_{variant}.jsonl").open("w", encoding="utf-8") as alerts:
            writer = csv.writer(handle)
            writer.writerow(["event_id", "timestamp", "source_record_id", "score", "threshold_alert", "label", "candidate_events"])
            for event_id, timestamp, source, score, label, count in self.connection.execute(
                "SELECT event_id,timestamp,source_record_id,score,label,candidate_events FROM predictions WHERE variant=? AND split=? ORDER BY time_us,event_id",
                (variant, split)):
                writer.writerow([event_id, timestamp, source, score, int(score >= threshold), label, count])
                if score >= threshold:
                    alerts.write(json.dumps({"event_id": event_id, "timestamp": timestamp, "source_record_id": source,
                                             "score": score, "decision": "suspicious_transaction_alert",
                                             "candidate_events": count, "evidence_status": "not_audited"}) + "\n")
