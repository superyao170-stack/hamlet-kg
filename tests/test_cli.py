from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hamletkg.cli import _case_graphs


class CaseDirectoryLoadingTests(unittest.TestCase):
    def test_non_case_json_artifacts_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "summary.json").write_text(
                json.dumps({"materialized_cases": 3}), encoding="utf-8"
            )
            (root / "case.json").write_text(
                json.dumps(
                    {
                        "case_id": "CASE-1",
                        "entities": [],
                        "atomic_events": [],
                        "atomic_relations": [],
                        "risk_events": [],
                        "risk_relations": [],
                    }
                ),
                encoding="utf-8",
            )
            cases = _case_graphs(root)
            self.assertEqual(["CASE-1"], [item.case_id for item in cases])


if __name__ == "__main__":
    unittest.main()
