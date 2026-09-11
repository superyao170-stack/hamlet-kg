from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hamletkg.adapters import case_from_transaction_csv


class IBMAdapterTests(unittest.TestCase):
    def test_label_is_kept_out_of_event_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transactions.csv"
            path.write_text(
                "Timestamp,From Bank,Account,To Bank,Account.1,Amount Paid,Payment Currency,Amount Received,Receiving Currency,Payment Format,Is Laundering\n"
                "2026-01-01 10:00:00,1,A,2,B,100,USD,100,USD,Wire,1\n",
                encoding="utf-8",
            )
            case = case_from_transaction_csv(path, case_id="TEST")
            event = next(iter(case.atomic_events.values()))
            self.assertNotIn("is_laundering", event.attributes)
            self.assertEqual(1, case.metadata["evaluation_labels"][event.event_id])


if __name__ == "__main__":
    unittest.main()
