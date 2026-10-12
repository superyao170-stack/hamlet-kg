"""Disk-backed causal samples with separately stored evaluation labels.

Each transaction's 24 arrival features are stored once as binary float64.
Event members reference that row rather than duplicating historical vectors.
SQLite preserves a complete timestamp group's split and deterministic order.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import struct
from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from ..io import write_json
from .policy import DetectionPolicy, validate_policy_for_stream
from .stream import (
    BASE_FEATURE_NAMES, MEMBER_ROLES, MOTIF_TYPES, EventInstance, EventMember,
    StreamConfig, TransactionSample, iter_samples,
)

if TYPE_CHECKING:
    from .dataset import PreparedDataset


_SCHEMA_VERSION = 1
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_FEATURE_STRUCT = struct.Struct("<" + "d" * len(BASE_FEATURE_NAMES))
_SPLITS = ("train", "validation", "test")


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _digest_json(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp_us(timestamp: datetime) -> int:
    delta = timestamp - _EPOCH
    return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds


def _datetime(timestamp_us: int) -> datetime:
    return _EPOCH + timedelta(microseconds=timestamp_us)


def _iso(timestamp_us: int) -> str:
    return _datetime(timestamp_us).isoformat().replace("+00:00", "Z")


def _code_fingerprint() -> dict[str, str]:
    directory = Path(__file__).parent
    return {name: _hash_file(directory / name) for name in ("cache.py", "stream.py", "policy.py")}


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _fractions(train: float, validation: float) -> None:
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value <= 0 for value in (train, validation)):
        raise ValueError("train_fraction and validation_fraction must be finite and positive")
    if train + validation >= 1:
        raise ValueError("train_fraction + validation_fraction must be less than 1")


def _events_json(sample: TransactionSample) -> str:
    return _canonical([
        {"instance_id": instance.instance_id, "motif_type": instance.motif_type,
         "anchor_id": instance.anchor_id, "span_seconds": instance.span_seconds,
         "amount_ratio": instance.amount_ratio, "rule_id": instance.rule_id,
         "members": [{"event_id": member.event_id, "role": member.role,
                      "elapsed_seconds": member.elapsed_seconds} for member in instance.members]}
        for instance in sample.events
    ])


def _split_plan(connection: sqlite3.Connection, total: int, train_fraction: float,
                validation_fraction: float) -> dict[str, dict]:
    """Find the same complete-group boundaries as chronological_split.

    Only aggregate counts are scanned; neither labels nor all sample objects
    are loaded. Ties choose the earlier boundary, and each split keeps a group.
    """
    distinct = connection.execute("SELECT COUNT(DISTINCT timestamp_us) FROM samples").fetchone()[0]
    if total < 3 or distinct < 3:
        raise ValueError("chronological splitting requires at least three distinct timestamps")
    group_query = "SELECT timestamp_us,COUNT(*) FROM samples GROUP BY timestamp_us ORDER BY timestamp_us"
    train_target = total * train_fraction
    cumulative = 0
    best_train: tuple[float, int] | None = None
    for group_number, (_, count) in enumerate(connection.execute(group_query), start=1):
        cumulative += count
        if group_number <= distinct - 2:
            candidate = (abs(cumulative - train_target), cumulative)
            if best_train is None or candidate < best_train:
                best_train = candidate
    assert best_train is not None
    train_end = best_train[1]
    validation_target = total * (train_fraction + validation_fraction)
    cumulative = 0
    best_validation: tuple[float, int] | None = None
    for group_number, (_, count) in enumerate(connection.execute(group_query), start=1):
        cumulative += count
        if cumulative > train_end and group_number < distinct:
            candidate = (abs(cumulative - validation_target), cumulative)
            if best_validation is None or candidate < best_validation:
                best_validation = candidate
    assert best_validation is not None
    validation_end = best_validation[1]
    result = {}
    for name, start, end in (("train", 0, train_end), ("validation", train_end, validation_end),
                             ("test", validation_end, total)):
        first = connection.execute("SELECT timestamp_us FROM samples WHERE seq=?", (start,)).fetchone()[0]
        last = connection.execute("SELECT timestamp_us FROM samples WHERE seq=?", (end - 1,)).fetchone()[0]
        result[name] = {"count": end - start, "start_seq": start, "end_seq": end,
                        "start_time": _iso(first), "end_time": _iso(last)}
    return result


def build_sample_cache(dataset: PreparedDataset, output: str | Path, *,
                       config: StreamConfig = StreamConfig(), policy: DetectionPolicy | None = None,
                       policy_provenance: dict | None = None,
                       train_fraction: float = .6, validation_fraction: float = .2) -> dict:
    """Build once from prepared events; labels are read in a separate pass.

    The output must be new or empty. Interrupted and failed builds retain a
    manifest whose status cannot be accepted by CachedSamples.
    """
    if not isinstance(config, StreamConfig):
        raise TypeError("config must be a StreamConfig")
    if policy_provenance is not None and not isinstance(policy_provenance, dict):
        raise ValueError("policy_provenance must be a dictionary or None")
    if policy is not None:
        validate_policy_for_stream(policy, window_seconds=config.window_seconds, max_history=config.max_history)
    _fractions(train_fraction, validation_fraction)
    destination = Path(output)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("sample cache output must be a new or empty directory")
    dataset.validate_integrity()
    source_manifest = dataset.manifest
    source_sha = source_manifest.get("database", {}).get("sha256") or source_manifest.get("database_sha256") or _hash_file(Path(dataset.database_path))
    selection = getattr(dataset, "selection", None)
    manifest = {
        "schema_version": _SCHEMA_VERSION, "status": "running", "database_filename": "samples.sqlite3",
        "dataset": {"database_sha256": source_sha, "manifest_sha256": _digest_json(source_manifest),
                    "source_sha256": source_manifest.get("source", {}).get("sha256"),
                    "selection": selection},
        "feature_names": list(BASE_FEATURE_NAMES), "motif_types": list(MOTIF_TYPES),
        "member_roles": list(MEMBER_ROLES), "stream_config": asdict(config),
        "stream_config_sha256": _digest_json(asdict(config)),
        "policy": policy.to_dict() if policy is not None else None,
        "policy_provenance": json.loads(_canonical(policy_provenance)) if policy_provenance is not None else None,
        "policy_sha256": _digest_json(policy.to_dict() if policy is not None else None),
        "code_fingerprint": _code_fingerprint(), "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "limitations": ["Time splits use complete timestamp groups and never labels.",
                        "A selected time view starts with empty history; rows before its after boundary are not replayed.",
                        "Active history and the largest simultaneous timestamp group remain in memory.",
                        "Simulation labels are evaluation targets, not evidence of real-world money laundering."],
    }
    # Validate JSON-serializable provenance before creating any artifacts.
    _canonical(manifest)
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "manifest.json", manifest)
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(destination / manifest["database_filename"])
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA cache_size=-8192")
        connection.executescript(f"""
            CREATE TABLE cache_metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE samples (
                seq INTEGER PRIMARY KEY,event_id TEXT NOT NULL UNIQUE,timestamp_us INTEGER NOT NULL,
                features BLOB NOT NULL CHECK(length(features)={_FEATURE_STRUCT.size}),events_json TEXT NOT NULL
            );
            CREATE TABLE evaluation_labels (
                event_id TEXT PRIMARY KEY NOT NULL REFERENCES samples(event_id),
                label INTEGER CHECK(label IS NULL OR label IN (0,1))
            );
            CREATE INDEX samples_timestamp ON samples(timestamp_us,seq);
        """)
        id_digest = hashlib.sha256()
        first_id = last_id = None
        count = 0
        for sample in iter_samples(dataset.iter_events(), config, policy=policy):
            if len(sample.features) != len(BASE_FEATURE_NAMES) or any(not math.isfinite(x) for x in sample.features):
                raise ValueError("sample features must be 24 finite values")
            connection.execute("INSERT INTO samples VALUES(?,?,?,?,?)", (
                count, sample.event_id, _timestamp_us(sample.timestamp),
                _FEATURE_STRUCT.pack(*sample.features), _events_json(sample),
            ))
            encoded = sample.event_id.encode("utf-8")
            id_digest.update(len(encoded).to_bytes(8, "big"))
            id_digest.update(encoded)
            if count == 0:
                first_id = sample.event_id
            last_id = sample.event_id
            count += 1
            if count % 2000 == 0:
                connection.commit()
        connection.commit()
        splits = _split_plan(connection, count, train_fraction, validation_fraction)
        label_count = 0
        for event_id, label in dataset.iter_labels():
            if not isinstance(event_id, str) or not event_id:
                raise ValueError("evaluation label event_id must be a nonempty string")
            if label is not None and (isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1)):
                raise ValueError("evaluation labels must be binary integers or None")
            connection.execute("INSERT INTO evaluation_labels VALUES(?,?)", (event_id, label))
            label_count += 1
            if label_count % 2000 == 0:
                connection.commit()
        if label_count != count:
            raise ValueError("evaluation labels must contain exactly one row for every sample")
        manifest.update({"sample_count": count, "split": splits,
                         "split_counts": {**{name: splits[name]["count"] for name in _SPLITS}, "all": count},
                         "source_ids": {"count": count, "first_event_id": first_id,
                                        "last_event_id": last_id, "ordered_event_ids_sha256": id_digest.hexdigest()}})
        # Metadata is duplicated inside the database so manifest edits are
        # detected independently of the database checksum.
        frozen = {key: value for key, value in manifest.items() if key not in ("status", "database_sha256")}
        connection.execute("INSERT INTO cache_metadata VALUES('manifest',?)", (_canonical(frozen),))
        connection.commit()
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise ValueError("sample cache has invalid label references")
        connection.close()
        manifest["database_sha256"] = _hash_file(destination / manifest["database_filename"])
        manifest["status"] = "complete"
        write_json(destination / "manifest.json", manifest)
        return manifest
    except BaseException as error:
        if connection is not None:
            connection.close()
        manifest["status"] = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
        manifest["failure_type"] = type(error).__name__
        try:
            write_json(destination / "manifest.json", manifest)
        except OSError:
            pass
        raise


class CachedSamples:
    """Read immutable batches with a bounded cache of referenced member rows."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        try:
            self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError("sample cache manifest cannot be read") from error
        manifest = self.manifest
        if not isinstance(manifest, dict) or manifest.get("schema_version") != _SCHEMA_VERSION or manifest.get("status") != "complete":
            raise ValueError("sample cache must have a complete supported manifest")
        if manifest.get("database_filename") != "samples.sqlite3":
            raise ValueError("sample cache has an invalid database filename")
        self.database_path = self.directory / "samples.sqlite3"
        if not self.database_path.is_file():
            raise ValueError("sample cache database is missing")
        if manifest.get("feature_names") != list(BASE_FEATURE_NAMES) or manifest.get("motif_types") != list(MOTIF_TYPES) or manifest.get("member_roles") != list(MEMBER_ROLES):
            raise ValueError("sample cache feature or event vocabulary mismatch")
        if _digest_json(manifest.get("stream_config")) != manifest.get("stream_config_sha256") or _digest_json(manifest.get("policy")) != manifest.get("policy_sha256"):
            raise ValueError("sample cache configuration digest mismatch")
        try:
            config = StreamConfig(**manifest["stream_config"])
            if manifest["policy"] is not None:
                policy = DetectionPolicy.from_dict(manifest["policy"])
                validate_policy_for_stream(policy, window_seconds=config.window_seconds, max_history=config.max_history)
        except (TypeError, KeyError, ValueError) as error:
            raise ValueError("sample cache configuration is invalid") from error
        self._validate_schema_and_counts()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path.resolve().as_uri() + "?mode=ro", uri=True)

    def __len__(self) -> int:
        return self.manifest["sample_count"]

    def split_counts(self) -> dict[str, int]:
        return dict(self.manifest["split_counts"])

    def _bounds(self, split: str) -> tuple[int, int]:
        if split == "all":
            return 0, len(self)
        if split not in _SPLITS:
            raise ValueError("split must be train, validation, test or all")
        settings = self.manifest["split"][split]
        return settings["start_seq"], settings["end_seq"]

    def _validate_schema_and_counts(self) -> None:
        try:
            with closing(self._connect()) as connection:
                columns = {table: [row[1] for row in connection.execute(f"PRAGMA table_info({table})")]
                           for table in ("samples", "evaluation_labels", "cache_metadata")}
                if columns != {"samples": ["seq", "event_id", "timestamp_us", "features", "events_json"],
                               "evaluation_labels": ["event_id", "label"], "cache_metadata": ["key", "value"]}:
                    raise ValueError("sample cache schema mismatch")
                metadata = connection.execute("SELECT value FROM cache_metadata WHERE key='manifest'").fetchone()
                frozen = {key: value for key, value in self.manifest.items() if key not in ("status", "database_sha256")}
                if metadata is None or metadata[0] != _canonical(frozen):
                    raise ValueError("sample cache frozen manifest mismatch")
                total, lowest, highest = connection.execute("SELECT COUNT(*),MIN(seq),MAX(seq) FROM samples").fetchone()
                if isinstance(self.manifest.get("sample_count"), bool) or total != self.manifest.get("sample_count") or lowest != 0 or highest != total - 1:
                    raise ValueError("sample cache sample counts mismatch")
                if connection.execute("SELECT COUNT(*) FROM evaluation_labels").fetchone()[0] != total:
                    raise ValueError("sample cache evaluation label counts mismatch")
                if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
                    raise ValueError("sample cache has invalid label references")
                expected = _split_plan(connection, total, self.manifest["train_fraction"], self.manifest["validation_fraction"])
                if expected != self.manifest.get("split") or self.split_counts() != {**{key: value["count"] for key, value in expected.items()}, "all": total}:
                    raise ValueError("sample cache split boundaries mismatch")
        except sqlite3.DatabaseError as error:
            raise ValueError("sample cache database is invalid") from error

    def validate_integrity(self, *, check_members: bool = False) -> None:
        """Explicitly hash the entire database; iteration does not rehash it."""
        if _hash_file(self.database_path) != self.manifest.get("database_sha256"):
            raise ValueError("sample cache database checksum mismatch")
        self._validate_schema_and_counts()
        if check_members:
            # The optional full reference scan is more expensive than hashing;
            # every normal batch still checks its own references and causality.
            for _ in self.iter_batches("all", batch_size=256):
                pass

    def iter_labels(self, split: str = "train") -> Iterator[int | None]:
        start, end = self._bounds(split)
        connection = self._connect()
        try:
            cursor = connection.execute("SELECT l.label FROM samples s JOIN evaluation_labels l USING(event_id) WHERE s.seq>=? AND s.seq<? ORDER BY s.seq", (start, end))
            while rows := cursor.fetchmany(2000):
                for (label,) in rows:
                    if label is not None and (not isinstance(label, int) or label not in (0, 1)):
                        raise ValueError("sample cache has invalid evaluation labels")
                    yield label
        finally:
            connection.close()

    def iter_ids(self, split: str = "train") -> Iterator[str]:
        start, end = self._bounds(split)
        connection = self._connect()
        try:
            cursor = connection.execute("SELECT event_id FROM samples WHERE seq>=? AND seq<? ORDER BY seq", (start, end))
            while rows := cursor.fetchmany(2000):
                yield from (row[0] for row in rows)
        finally:
            connection.close()

    def iter_batches(self, split: str = "train", batch_size: int = 64) -> Iterator[list[TransactionSample]]:
        _positive_integer(batch_size, "batch_size")
        start, end = self._bounds(split)
        connection = self._connect()
        try:
            cursor = connection.execute("SELECT event_id,timestamp_us,features,events_json FROM samples WHERE seq>=? AND seq<? ORDER BY seq", (start, end))
            while rows := cursor.fetchmany(batch_size):
                yield self._decode_batch(connection, rows)
        finally:
            connection.close()

    @staticmethod
    def _unpack(features: bytes) -> tuple[float, ...]:
        if not isinstance(features, bytes) or len(features) != _FEATURE_STRUCT.size:
            raise ValueError("sample cache features have an invalid binary size")
        values = _FEATURE_STRUCT.unpack(features)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("sample cache features must be finite")
        return values

    def _decode_batch(self, connection: sqlite3.Connection, rows: list[tuple]) -> list[TransactionSample]:
        references = {row[0]: (self._unpack(row[2]), row[1]) for row in rows}
        event_lists = []
        member_ids = set()
        for _, _, _, raw in rows:
            try:
                events = json.loads(raw)
            except (TypeError, ValueError) as error:
                raise ValueError("sample cache event JSON is invalid") from error
            if not isinstance(events, list):
                raise ValueError("sample cache events must be a list")
            for event in events:
                if not isinstance(event, dict) or set(event) != {"instance_id", "motif_type", "anchor_id", "members", "span_seconds", "amount_ratio", "rule_id"}:
                    raise ValueError("sample cache event fields are invalid")
                if event["motif_type"] not in MOTIF_TYPES or not isinstance(event["members"], list) or not event["members"]:
                    raise ValueError("sample cache event vocabulary or members are invalid")
                for name in ("instance_id", "anchor_id"):
                    if not isinstance(event[name], str) or not event[name]:
                        raise ValueError("sample cache event identifiers are invalid")
                if event["rule_id"] is not None and (not isinstance(event["rule_id"], str) or not event["rule_id"]):
                    raise ValueError("sample cache rule identifier is invalid")
                for name in ("span_seconds", "amount_ratio"):
                    value = event[name]
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                        raise ValueError("sample cache event statistics are invalid")
                seen = set()
                for member in event["members"]:
                    if not isinstance(member, dict) or set(member) != {"event_id", "role", "elapsed_seconds"}:
                        raise ValueError("sample cache member fields are invalid")
                    event_id, role, elapsed = member["event_id"], member["role"], member["elapsed_seconds"]
                    if not isinstance(event_id, str) or not event_id or event_id in seen or role not in MEMBER_ROLES:
                        raise ValueError("sample cache member identifier or role is invalid")
                    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
                        raise ValueError("sample cache member elapsed time is invalid")
                    seen.add(event_id)
                    member_ids.add(event_id)
                roles = [member["role"] for member in event["members"]]
                motif = event["motif_type"]
                if ((motif == "fan_in" and any(role != "incoming" for role in roles))
                    or (motif == "fan_out" and any(role != "outgoing" for role in roles))
                    or (motif == "pass_through" and (len(roles) < 2 or roles[-1] != "outgoing"
                                                     or any(role != "incoming" for role in roles[:-1])))
                    or (motif == "relay" and roles != ["relay_start", "relay_middle", "relay_end"])):
                    raise ValueError("sample cache member roles do not match their event motif")
            event_lists.append(events)
        missing = list(member_ids - references.keys())
        # A bounded SQL parameter list supports old SQLite builds as well as
        # large event-rich batches. References are discarded after this batch.
        for offset in range(0, len(missing), 400):
            ids = missing[offset:offset + 400]
            placeholders = ",".join("?" for _ in ids)
            for event_id, features, timestamp in connection.execute(f"SELECT event_id,features,timestamp_us FROM samples WHERE event_id IN ({placeholders})", ids):
                references[event_id] = (self._unpack(features), timestamp)
        if member_ids - references.keys():
            raise ValueError("sample cache contains a missing member reference")
        samples = []
        for (event_id, timestamp_us, _, _), events in zip(rows, event_lists):
            instances = []
            for event in events:
                members = []
                for member in event["members"]:
                    features, member_timestamp = references[member["event_id"]]
                    expected_elapsed = (timestamp_us - member_timestamp) / 1_000_000
                    if expected_elapsed != member["elapsed_seconds"] or (member["event_id"] != event_id and member_timestamp >= timestamp_us):
                        raise ValueError("sample cache member violates arrival-time causality")
                    members.append(EventMember(member["event_id"], member["role"], features, member["elapsed_seconds"]))
                if members[-1].event_id != event_id or event["span_seconds"] != max(member.elapsed_seconds for member in members):
                    raise ValueError("sample cache event has an invalid current member or span")
                instances.append(EventInstance(event["instance_id"], event["motif_type"], event["anchor_id"], tuple(members),
                                               event["span_seconds"], event["amount_ratio"], event["rule_id"]))
            samples.append(TransactionSample(event_id, _datetime(timestamp_us), references[event_id][0], tuple(instances)))
        return samples
