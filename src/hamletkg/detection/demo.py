"""Tiny deterministic synthetic fixture for exercising the detection pipeline.

This is NOT IBM AMLworld and is NOT an independent research benchmark.
Both laundering and benign examples include superficially similar motifs.
"""

from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path


def write_demo_csv(path: str | Path, *, seed: int = 42, days: int = 24) -> dict:
    if days < 6:
        raise ValueError("the smoke fixture needs at least six days")
    rng = random.Random(seed)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows: list[tuple] = []

    def add(timestamp: datetime, sender: str, receiver: str, amount: float, label: int) -> None:
        rows.append((timestamp, 1, sender, 1, receiver, round(amount, 2),
                     "USD", round(amount, 2), "USD", "ACH", label))

    for day in range(days):
        origin = start + timedelta(days=day)
        for _ in range(36):
            sender, receiver = rng.sample(range(70), 2)
            add(origin + timedelta(minutes=rng.randrange(1440)),
                f"A{sender}", f"A{receiver}", rng.uniform(100, 1400), 0)
        # The benign group is deliberately structurally similar to the positive
        # group: a rule hit is never itself a laundering label.
        for prefix, label in (("BENIGN", 0), ("INJECTED", 1)):
            begin = origin + timedelta(minutes=rng.randrange(60, 1200))
            amounts = [rng.uniform(120, 550) for _ in range(3)]
            hub, destination = f"{prefix}-H{day}", f"{prefix}-D{day}"
            for offset, amount in enumerate(amounts):
                add(begin + timedelta(minutes=offset * 3),
                    f"{prefix}-S{day}-{offset}", hub, amount, label)
            add(begin + timedelta(minutes=12), hub, destination,
                sum(amounts) * .95, label)
            if label:
                add(begin + timedelta(minutes=18), destination,
                    f"{prefix}-END{day}", sum(amounts) * .93, label)

    rows.sort(key=lambda row: (row[0], row[2], row[4]))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        # Preserve the actual IBM duplicate-header shape to exercise the adapter.
        writer.writerow(["Timestamp", "From Bank", "Account", "To Bank", "Account",
                         "Amount Paid", "Payment Currency", "Amount Received",
                         "Receiving Currency", "Payment Format", "Is Laundering"])
        for row in rows:
            writer.writerow([row[0].strftime("%Y/%m/%d %H:%M:%S"), *row[1:]])
    return {"kind": "synthetic_smoke_fixture_not_research_evidence", "seed": seed,
            "days": days, "transactions": len(rows),
            "positives": sum(row[-1] for row in rows)}
