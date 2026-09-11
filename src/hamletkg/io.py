"""JSON persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import CaseGraph, RiskPattern


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(destination)


def load_case(path: str | Path) -> CaseGraph:
    return CaseGraph.from_dict(read_json(path))


def save_case(path: str | Path, case: CaseGraph) -> None:
    write_json(path, case.to_dict())


def load_patterns(path: str | Path) -> list[RiskPattern]:
    value = read_json(path)
    items = value.get("patterns", value) if isinstance(value, dict) else value
    if not isinstance(items, list):
        raise ValueError("pattern file must contain a JSON list or a patterns list")
    return [RiskPattern.from_dict(item) for item in items]
