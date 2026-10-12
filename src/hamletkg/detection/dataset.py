"""Disk-backed IBM transaction preparation with labels kept in a separate table.

SQLite sorts the validated source rows without materializing a CaseGraph.
Prepared views select a declared continuous UTC interval; selection never
depends on an evaluation label. Each reader opens a fresh read-only connection.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator

from ..adapters.ibm_aml import _normalized_header, _validate_header, iter_transaction_csv
from ..models import AtomicEvent, iso_time, parse_time


SCHEMA_VERSION = 1
DATABASE_FILENAME = "transactions.sqlite"
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_FETCH_SIZE = 2000
_EVENT_COLUMNS = (
    "event_id", "timestamp_us", "sender", "receiver", "amount", "received_amount",
    "paid_currency", "received_currency", "channel", "source_record_id",
)
_EVENT_SELECT = ", ".join(_EVENT_COLUMNS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp_us(timestamp: datetime) -> int:
    delta = parse_time(timestamp) - _EPOCH
    return (delta.days * 86400 + delta.seconds) * 1000000 + delta.microseconds


def _from_timestamp_us(value: int) -> datetime:
    return _EPOCH + timedelta(microseconds=value)


def _bound(value: str | datetime | None, name: str) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, (str, datetime)):
        raise ValueError(f"{name} must be an ISO timestamp, datetime, or None")
    try:
        return parse_time(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a valid timestamp") from exc


def _bounds(after: str | datetime | None, before: str | datetime | None) -> tuple[datetime | None, datetime | None]:
    lower, upper = _bound(after, "after"), _bound(before, "before")
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError("after must precede before")
    return lower, upper


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _write_manifest(directory: Path, manifest: dict) -> None:
    temporary = directory / "manifest.json.partial"
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(directory / "manifest.json")


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _nonnegative_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _columns(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), None)
    if header is None:
        raise ValueError("transaction CSV is empty")
    result = _normalized_header(header)
    _validate_header(result)
    return result


def _schema(connection: sqlite3.Connection) -> None:
    connection.executescript("""
        CREATE TABLE transactions (
            event_id TEXT PRIMARY KEY NOT NULL,
            timestamp_us INTEGER NOT NULL,
            sender TEXT NOT NULL,
            receiver TEXT NOT NULL,
            amount REAL NOT NULL CHECK (amount > 0),
            received_amount REAL NOT NULL CHECK (received_amount > 0),
            paid_currency TEXT NOT NULL,
            received_currency TEXT NOT NULL,
            channel TEXT NOT NULL,
            source_record_id TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE labels (
            event_id TEXT PRIMARY KEY NOT NULL REFERENCES transactions(event_id),
            label INTEGER CHECK (label IS NULL OR label IN (0, 1))
        ) WITHOUT ROWID;
        CREATE TABLE dataset_info (
            key TEXT PRIMARY KEY NOT NULL,
            value TEXT NOT NULL
        ) WITHOUT ROWID;
    """)


def prepare_detection_dataset(
    csv_path: str | Path,
    output: str | Path,
    *,
    case_id: str = "IBM-AML",
    limit: int | None = None,
    batch_size: int = 2000,
    after: str | datetime | None = None,
    before: str | datetime | None = None,
) -> dict:
    """Prepare validated rows in a new/empty directory, publishing only on success.

    ``limit`` selects the first N original rows before applying [after, before).
    The SHA identifies the entire source file even for a limited preparation.
    Invalid rows within that prefix fail validation, including excluded times.
    """

    if not isinstance(case_id, str) or not case_id:
        raise ValueError("case_id must not be empty")
    if limit is not None:
        _positive_integer(limit, "limit")
    _positive_integer(batch_size, "batch_size")
    lower, upper = _bounds(after, before)
    source = Path(csv_path).resolve()
    original_identity = _file_identity(source)
    directory = Path(output).resolve()
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ValueError("prepared dataset output must be a new or empty directory")
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / (DATABASE_FILENAME + ".partial")
    database = directory / DATABASE_FILENAME
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "hamletkg.prepared_detection_dataset",
        "status": "preparing",
        "case_id": case_id,
        "source": {"path": str(source), "filename": source.name, "size_bytes": original_identity[2]},
        "selection": {
            "limit": limit,
            "after": iso_time(lower) if lower is not None else None,
            "before": iso_time(upper) if upper is not None else None,
            "interval": "[after, before)",
            "limit_applies_to_original_file_rows": True,
            "label_filtering": False,
        },
        "counts": {"read_rows": 0, "kept_transactions": 0, "positive": 0, "negative": 0, "unlabeled": 0},
        "labels_are_features": False,
        "identity_fields_are_features": False,
        "limitations": [
            "Preparation and time views do not isolate simulated laundering chains.",
            "Public synthetic labels do not establish a real-world criminal finding.",
            "Missing evaluation labels remain unknown; they are not converted to negatives.",
        ],
    }
    _write_manifest(directory, manifest)
    connection: sqlite3.Connection | None = None
    try:
        manifest["source"]["sha256"] = _sha256(source)
        if _file_identity(source) != original_identity:
            raise ValueError("transaction CSV changed while computing its SHA256")
        columns = _columns(source)
        manifest["validation"] = {"normalized_columns": columns, "required_columns_validated": True}
        connection = sqlite3.connect(temporary)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA cache_size = -8192")
        connection.execute("PRAGMA temp_store = FILE")
        _schema(connection)
        connection.executemany("INSERT INTO dataset_info(key,value) VALUES (?,?)", (
            ("schema_version", str(SCHEMA_VERSION)),
            ("case_id", case_id),
            ("source_sha256", manifest["source"]["sha256"]),
        ))
        connection.commit()
        rows: list[tuple] = []
        labels: list[tuple] = []
        insert = f"INSERT INTO transactions ({_EVENT_SELECT}) VALUES ({','.join('?' for _ in _EVENT_COLUMNS)})"

        def flush() -> None:
            if not rows:
                return
            with connection:
                connection.executemany(insert, rows)
                connection.executemany("INSERT INTO labels(event_id,label) VALUES (?,?)", labels)
            rows.clear()
            labels.clear()

        for event, label in iter_transaction_csv(source, case_id=case_id, limit=limit):
            manifest["counts"]["read_rows"] += 1
            if (lower is not None and event.timestamp < lower) or (upper is not None and event.timestamp >= upper):
                continue
            attributes = event.attributes
            rows.append((
                event.event_id, _timestamp_us(event.timestamp), event.roles["sender"], event.roles["receiver"],
                attributes["amount"], attributes["received_amount"], attributes["paid_currency"],
                attributes["received_currency"], attributes["channel"], event.source_record_id,
            ))
            labels.append((event.event_id, label))
            manifest["counts"]["kept_transactions"] += 1
            manifest["counts"]["unlabeled" if label is None else "positive" if label else "negative"] += 1
            if len(rows) >= batch_size:
                flush()
        flush()
        if _file_identity(source) != original_identity:
            raise ValueError("transaction CSV changed while preparing the dataset")
        if _sha256(source) != manifest["source"]["sha256"] or _file_identity(source) != original_identity:
            raise ValueError("transaction CSV changed while preparing the dataset")
        with connection:
            connection.execute("CREATE INDEX transactions_time ON transactions(timestamp_us,event_id)")
        first, last, groups = connection.execute(
            "SELECT MIN(timestamp_us),MAX(timestamp_us),COUNT(DISTINCT timestamp_us) FROM transactions"
        ).fetchone()
        manifest["time"] = {
            "start": iso_time(_from_timestamp_us(first)) if first is not None else None,
            "end": iso_time(_from_timestamp_us(last)) if last is not None else None,
            "timestamp_groups": groups,
            "sort_order": ["timestamp_us", "event_id"],
            "timezone": "UTC",
        }
        connection.close()
        connection = None
        manifest["database"] = {
            "filename": DATABASE_FILENAME,
            "sha256": _sha256(temporary),
            "size_bytes": temporary.stat().st_size,
        }
        temporary.replace(database)
        manifest["status"] = "complete"
        _write_manifest(directory, manifest)
        return manifest
    except BaseException as exc:
        if connection is not None:
            connection.close()
            connection = None
        for path in (temporary, database, Path(str(temporary) + "-journal")):
            path.unlink(missing_ok=True)
        manifest["status"] = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        manifest["error_type"] = type(exc).__name__
        try:
            _write_manifest(directory, manifest)
        except OSError:
            pass
        raise


class PreparedDataset:
    """A read-only dataset with an optional frozen, label-independent UTC view."""

    def __init__(self, directory: str | Path, *, after: str | datetime | None = None,
                 before: str | datetime | None = None) -> None:
        self.directory = Path(directory).resolve()
        lower, upper = _bounds(after, before)
        try:
            manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError("prepared dataset manifest is missing or invalid") from exc
        if (
            not isinstance(manifest, dict)
            or type(manifest.get("schema_version")) is not int
            or manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("artifact_type") != "hamletkg.prepared_detection_dataset"
            or manifest.get("status") != "complete"
            or not isinstance(manifest.get("database"), dict)
            or manifest["database"].get("filename") != DATABASE_FILENAME
        ):
            raise ValueError("prepared dataset manifest is not a supported complete dataset")
        source, counts, database = manifest.get("source"), manifest.get("counts"), manifest["database"]
        if (
            not isinstance(manifest.get("case_id"), str) or not manifest["case_id"]
            or not isinstance(source, dict) or not _digest(source.get("sha256"))
            or not isinstance(counts, dict)
            or not all(_nonnegative_integer(counts.get(key)) for key in
                       ("read_rows", "kept_transactions", "positive", "negative", "unlabeled"))
            or counts["read_rows"] < counts["kept_transactions"]
            or not _digest(database.get("sha256"))
            or not _nonnegative_integer(database.get("size_bytes"))
            or manifest.get("labels_are_features") is not False
        ):
            raise ValueError("prepared dataset manifest provenance or counts are invalid")
        self.manifest = manifest
        self.database_path = self.directory / DATABASE_FILENAME
        if not self.database_path.is_file():
            raise ValueError("prepared dataset database is missing")
        predicates: list[str] = []
        parameters: list[int] = []
        if lower is not None:
            predicates.append("timestamp_us >= ?")
            parameters.append(_timestamp_us(lower))
        if upper is not None:
            predicates.append("timestamp_us < ?")
            parameters.append(_timestamp_us(upper))
        self._where = " WHERE " + " AND ".join(predicates) if predicates else ""
        self._parameters = tuple(parameters)
        try:
            with self._connect() as connection:
                transactions, positive, negative, unlabeled = connection.execute(
                    "SELECT COUNT(*),COALESCE(SUM(label=1),0),COALESCE(SUM(label=0),0),"
                    "COALESCE(SUM(label IS NULL),0) FROM transactions JOIN labels USING(event_id)"
                    + self._where, self._parameters,
                ).fetchone()
        except sqlite3.Error as exc:
            raise ValueError("prepared dataset database is invalid") from exc
        self.selection = {
            "after": iso_time(lower) if lower is not None else None,
            "before": iso_time(upper) if upper is not None else None,
            "transactions": transactions, "positive": positive,
            "negative": negative, "unlabeled": unlabeled,
        }

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path.as_uri() + "?mode=ro", uri=True)
        try:
            connection.execute("PRAGMA query_only = ON")
            connection.execute("PRAGMA cache_size = -8192")
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _event(row: tuple) -> AtomicEvent:
        event_id, timestamp, sender, receiver, amount, received, paid_currency, received_currency, channel, source_id = row
        return AtomicEvent(
            event_id=event_id, action_type="TRANSFER", roles={"sender": sender, "receiver": receiver},
            timestamp=_from_timestamp_us(timestamp),
            attributes={"amount": amount, "received_amount": received, "paid_currency": paid_currency,
                        "received_currency": received_currency, "channel": channel},
            source_record_id=source_id,
        )

    def iter_events(self) -> Iterator[AtomicEvent]:
        """Read only transaction columns, with complete timestamp groups contiguous."""

        with self._connect() as connection:
            cursor = connection.execute(
                f"SELECT {_EVENT_SELECT} FROM transactions" + self._where
                + " ORDER BY timestamp_us,event_id", self._parameters,
            )
            while rows := cursor.fetchmany(_FETCH_SIZE):
                for row in rows:
                    yield self._event(row)

    def iter_labels(self) -> Iterator[tuple[str, int | None]]:
        """Read evaluation labels separately; row order is not a feature order."""

        with self._connect() as connection:
            cursor = connection.execute(
                "SELECT labels.event_id,labels.label FROM labels JOIN transactions USING(event_id)"
                + self._where, self._parameters,
            )
            while rows := cursor.fetchmany(_FETCH_SIZE):
                yield from rows

    def fetch_transactions(self, event_ids: Iterable[str]) -> dict[str, AtomicEvent]:
        """Look up only selected source transfers, in bounded SQL parameter groups."""

        if isinstance(event_ids, (str, bytes)):
            raise ValueError("event_ids must be an iterable of strings")
        result: dict[str, AtomicEvent] = {}
        chunk: list[str] = []
        with self._connect() as connection:
            def fetch() -> None:
                placeholders = ",".join("?" for _ in chunk)
                conjunction = " AND " if self._where else " WHERE "
                cursor = connection.execute(
                    f"SELECT {_EVENT_SELECT} FROM transactions" + self._where
                    + conjunction + f"event_id IN ({placeholders})",
                    (*self._parameters, *chunk),
                )
                while rows := cursor.fetchmany(_FETCH_SIZE):
                    for row in rows:
                        event = self._event(row)
                        result[event.event_id] = event
                chunk.clear()

            for event_id in event_ids:
                if not isinstance(event_id, str) or not event_id:
                    raise ValueError("event_ids must contain nonempty strings")
                chunk.append(event_id)
                if len(chunk) >= 500:
                    fetch()
            if chunk:
                fetch()
        return result

    def timestamp_groups(self) -> list[tuple[datetime, int]]:
        result: list[tuple[datetime, int]] = []
        with self._connect() as connection:
            cursor = connection.execute(
                "SELECT timestamp_us,COUNT(*) FROM transactions" + self._where
                + " GROUP BY timestamp_us ORDER BY timestamp_us", self._parameters,
            )
            while rows := cursor.fetchmany(_FETCH_SIZE):
                result.extend((_from_timestamp_us(timestamp), count) for timestamp, count in rows)
        return result

    def validate_integrity(self) -> None:
        """Explicitly verify the published file and full underlying table counts.

        This hashes the database once per caller request, never once per reader.
        The original source SHA remains provenance; reading does not require the
        source CSV to still be available beside the prepared database.
        """

        expected = self.manifest["database"]
        if (
            self.database_path.stat().st_size != expected.get("size_bytes")
            or _sha256(self.database_path) != expected.get("sha256")
        ):
            raise ValueError("prepared dataset database SHA256 or size does not match its manifest")
        try:
            with self._connect() as connection:
                actual_columns = tuple(row[1] for row in connection.execute("PRAGMA table_info(transactions)"))
                if actual_columns != _EVENT_COLUMNS:
                    raise ValueError("prepared dataset transaction schema is invalid")
                label_columns = tuple(row[1] for row in connection.execute("PRAGMA table_info(labels)"))
                if label_columns != ("event_id", "label"):
                    raise ValueError("prepared dataset label schema is invalid")
                info = dict(connection.execute("SELECT key,value FROM dataset_info"))
                if info != {"schema_version": str(SCHEMA_VERSION), "case_id": self.manifest.get("case_id"),
                            "source_sha256": self.manifest.get("source", {}).get("sha256")}:
                    raise ValueError("prepared dataset database provenance does not match its manifest")
                total = connection.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
                labeled_rows, positive, negative, unlabeled = connection.execute(
                    "SELECT COUNT(*),COALESCE(SUM(label=1),0),COALESCE(SUM(label=0),0),"
                    "COALESCE(SUM(label IS NULL),0) FROM labels"
                ).fetchone()
                counts = self.manifest.get("counts", {})
                if (
                    total != counts.get("kept_transactions") or labeled_rows != total
                    or positive != counts.get("positive") or negative != counts.get("negative")
                    or unlabeled != counts.get("unlabeled")
                ):
                    raise ValueError("prepared dataset counts do not match its manifest")
                if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
                    raise ValueError("prepared dataset labels reference missing transactions")
                first, last, groups = connection.execute(
                    "SELECT MIN(timestamp_us),MAX(timestamp_us),COUNT(DISTINCT timestamp_us) FROM transactions"
                ).fetchone()
                actual_time = {
                    "start": iso_time(_from_timestamp_us(first)) if first is not None else None,
                    "end": iso_time(_from_timestamp_us(last)) if last is not None else None,
                    "timestamp_groups": groups, "sort_order": ["timestamp_us", "event_id"], "timezone": "UTC",
                }
                if self.manifest.get("time") != actual_time:
                    raise ValueError("prepared dataset time bounds do not match its manifest")
        except sqlite3.Error as exc:
            raise ValueError("prepared dataset database integrity check failed") from exc
