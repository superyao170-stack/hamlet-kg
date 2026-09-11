import csv
import tempfile
import unittest
from pathlib import Path

from hamletkg.adapters.amlsim import case_from_amlsim_csv
from hamletkg.experiments.amlsim_baselines import run_amlsim_benchmarks


class AMLSimAdapterTests(unittest.TestCase):
    def _dataset(self, root: Path) -> Path:
        dataset = root / "20K_fanin200"
        dataset.mkdir()
        with (dataset / "nodes.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["nodeid", "isFraud", "init_balance", "fraudStep"])
            writer.writerow([1, 1, 100, 5])
            writer.writerow([2, 1, 120, -1])
            writer.writerow([3, 0, 150, -1])
        with (dataset / "transactions.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["sourceNodeId", "targetNodeId", "value", "time"])
            writer.writerow([1, 2, 4, 5])
            writer.writerow([1, 2, 4, 5])
            writer.writerow([3, 2, 200, 6])
        return dataset

    def test_labels_are_metadata_only(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(Path(directory))
            case = case_from_amlsim_csv(
                dataset / "transactions.csv",
                dataset / "nodes.csv",
                case_id="AMLSIM-TEST",
            )
            self.assertEqual(len(case.atomic_events), 3)
            self.assertEqual(case.metadata["evaluation_labels"]["ACCOUNT:AMLSIM:1"], 1)
            self.assertNotIn("isFraud", case.entities["ACCOUNT:AMLSIM:1"].attributes)
            self.assertNotIn("fraudStep", case.entities["ACCOUNT:AMLSIM:1"].attributes)
            for event in case.atomic_events.values():
                self.assertNotIn("isFraud", event.attributes)
                self.assertNotIn("fraudStep", event.attributes)

    def test_baseline_outputs_are_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._dataset(root)
            first = run_amlsim_benchmarks(root, root / "out-1")
            second = run_amlsim_benchmarks(root, root / "out-2")
            first["datasets"][0].pop("runtime_seconds")
            second["datasets"][0].pop("runtime_seconds")
            self.assertEqual(first, second)
            self.assertTrue((root / "out-1" / "metrics.csv").exists())


if __name__ == "__main__":
    unittest.main()
