from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hamletkg.adapters import case_from_transaction_csv


class IBMAdapterTests(unittest.TestCase):
    HEADER = "Timestamp,From Bank,Account,To Bank,Account,Amount Paid,Payment Currency,Amount Received,Receiving Currency,Payment Format,Is Laundering\n"

    def _load(self, rows: str, *, header: str | None = None, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transactions.csv"
            path.write_text((self.HEADER if header is None else header) + rows, encoding="utf-8")
            return case_from_transaction_csv(path, case_id="TEST", **kwargs)

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

    def test_original_duplicate_account_header_preserves_both_accounts(self) -> None:
        case = self._load("2022/09/01 00:00,1,SENDER,2,RECEIVER,100,USD,90,EUR,Wire,1\n")
        event = next(iter(case.atomic_events.values()))
        self.assertEqual("ACCOUNT:1:SENDER", event.roles["sender"])
        self.assertEqual("ACCOUNT:2:RECEIVER", event.roles["receiver"])
        self.assertEqual(90.0, event.attributes["received_amount"])
        self.assertEqual("EUR", event.attributes["received_currency"])
        self.assertFalse(case.metadata["labels_are_features"])

    def test_missing_labels_are_unlabeled_rather_than_negative(self) -> None:
        for header, row in (
            (self.HEADER, "2022/09/01 00:00,1,A,2,B,100,USD,100,USD,Wire,\n"),
            (self.HEADER.rsplit(",Is Laundering", 1)[0] + "\n", "2022/09/01 00:00,1,A,2,B,100,USD,100,USD,Wire\n"),
        ):
            with self.subTest(header=header):
                case = self._load(row, header=header)
                self.assertEqual({}, case.metadata["evaluation_labels"])

    def test_mixed_labeled_and_unlabeled_rows_preserve_only_observed_labels(self) -> None:
        case = self._load(
            "2022/09/01 00:00,1,A,2,B,100,USD,100,USD,Wire,0\n"
            "2022/09/01 00:01,1,C,2,D,100,USD,100,USD,Wire,\n"
        )
        self.assertEqual(1, len(case.metadata["evaluation_labels"]))
        self.assertEqual([0], list(case.metadata["evaluation_labels"].values()))

    def test_bad_amount_label_timestamp_and_required_fields_raise(self) -> None:
        valid = ["2022/09/01 00:00", "1", "A", "2", "B", "100", "USD", "100", "USD", "Wire", "0"]
        invalid_values = {
            0: ("not-a-date", ""),
            1: ("",), 2: ("",), 3: ("",), 4: ("",),
            5: ("bad", "NaN", "inf", "-1", "0", ""),
            6: ("",), 7: ("bad", "NaN", "-1", "0", ""),
            8: ("",), 9: ("",), 10: ("bad", "NaN", "inf", "2", "0.5", "-1"),
        }
        for index, values in invalid_values.items():
            for value in values:
                with self.subTest(column=index, value=value):
                    row = list(valid)
                    row[index] = value
                    with self.assertRaises(ValueError):
                        self._load(",".join(row) + "\n")

    def test_bad_row_shape_and_ambiguous_headers_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "columns"):
            self._load("2022/09/01 00:00,1,A\n")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self._load("", header="Timestamp,Timestamp\n")
        with self.assertRaisesRegex(ValueError, "required column"):
            self._load("", header="Timestamp,From Bank,Account,To Bank,Account\n")

    def test_relation_construction_can_be_disabled(self) -> None:
        with patch("hamletkg.adapters.ibm_aml.build_atomic_relations") as builder:
            case = self._load("2022/09/01 00:00,1,A,2,B,100,USD,100,USD,Wire,1\n", build_relations=False)
            builder.assert_not_called()
            self.assertEqual([], case.atomic_relations)

    def test_default_still_builds_relations(self) -> None:
        with patch("hamletkg.adapters.ibm_aml.build_atomic_relations") as builder:
            self._load("2022/09/01 00:00,1,A,2,B,100,USD,100,USD,Wire,1\n")
            builder.assert_called_once()


if __name__ == "__main__":
    unittest.main()
