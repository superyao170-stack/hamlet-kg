"""CPU PyTorch research detector with a shared transaction/event encoder.

Labels are passed to ``fit`` separately. Identifiers and absolute timestamps
are never tensorized. The event variants differ only in their enabled event
inputs; ``no_time`` removes event time, while retaining the base features.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

try:
    import torch
    from torch import nn
except ImportError as exc:  # The standard-library CLI stays usable without torch.
    raise ImportError(
        "Neural detection requires PyTorch. Install hamletkg[detection] "
        "or install torch separately."
    ) from exc

from .stream import MEMBER_ROLES, MOTIF_TYPES, TransactionSample


VARIANTS = ("base", "flat", "event", "no_role", "no_time", "untyped_event")
SCALING_OPTIONS = ("standard", "fixed_periodic")
FIXED_PERIODIC_FEATURE_NAMES = ("hour_sin", "hour_cos")


def _log_size(value: float) -> float:
    return math.log1p(max(0.0, float(value)))


def _feature_vectors(sample: TransactionSample) -> Iterator[Sequence[float]]:
    yield sample.features
    for event in sample.events:
        for member in event.members:
            yield member.features


class _ScalerAccumulator:
    """Welford statistics in the same vector order as the in-memory path."""

    def __init__(self, dimension: int) -> None:
        self.mean = [0.0] * dimension
        self.squared = [0.0] * dimension
        self.count = 0

    def update(self, vector: Sequence[float]) -> None:
        self.count += 1
        for index, value in enumerate(vector):
            value = float(value)
            delta = value - self.mean[index]
            self.mean[index] += delta / self.count
            self.squared[index] += delta * (value - self.mean[index])

    def finish(self) -> dict[str, list[float]]:
        if self.count == 0:
            raise ValueError("Training batches must contain at least one sample")
        if not all(math.isfinite(value) for value in self.mean + self.squared):
            raise ValueError("Training scaler statistics are not finite; inspect feature values")
        deviations = [math.sqrt(max(value / self.count, 0.0)) for value in self.squared]
        return {
            "mean": self.mean,
            "scale": [value if value >= 1e-6 else 1.0 for value in deviations],
        }


class _TrainingStreamIdentity:
    """Constant-size consistency check; identifiers are never model inputs."""

    def __init__(self) -> None:
        self.digest = hashlib.sha256()
        self.encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False)
        self.count = 0
        self.positives = 0

    def update(self, sample: TransactionSample, label: int) -> None:
        # Source identity includes IDs and timestamps solely to detect source
        # replacement between factory passes. Tensorization still omits them.
        payload = {
            "event_id": sample.event_id, "timestamp": sample.timestamp.isoformat(),
            "features": [float(value) for value in sample.features], "label": label,
            "events": [
                {
                    "instance_id": event.instance_id, "motif_type": event.motif_type,
                    "anchor_id": event.anchor_id, "rule_id": event.rule_id,
                    "span_seconds": float(event.span_seconds),
                    "amount_ratio": float(event.amount_ratio),
                    "members": [
                        {
                            "event_id": member.event_id, "role": member.role,
                            "features": [float(value) for value in member.features],
                            "elapsed_seconds": float(member.elapsed_seconds),
                        }
                        for member in event.members
                    ],
                }
                for event in sample.events
            ],
        }
        # Serialize only the current sample, whose history is bounded by the
        # stream configuration. The optimized encoder avoids per-token Python
        # hashing overhead across millions of rows.
        self.digest.update(self.encoder.encode(payload).encode("utf-8"))
        self.digest.update(b"\n")
        self.count += 1
        self.positives += label

    @property
    def negatives(self) -> int:
        return self.count - self.positives

    def signature(self) -> tuple[int, int, str]:
        return self.count, self.positives, self.digest.hexdigest()


class _DetectorNetwork(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, variant: str) -> None:
        super().__init__()
        self.variant = variant
        self.transaction_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        embedding_dim = max(4, hidden_dim // 4)
        if variant == "flat":
            self.flat_encoder = nn.Sequential(
                nn.Linear(len(MOTIF_TYPES) * 4, hidden_dim), nn.ReLU(),
            )
        elif variant != "base":
            # All grouped variants retain identical dimensions and capacity.
            # Disabled ablation inputs are zeroed, rather than relabeled.
            self.role_embedding = nn.Embedding(len(MEMBER_ROLES) + 1, embedding_dim)
            self.type_embedding = nn.Embedding(len(MOTIF_TYPES) + 1, embedding_dim)
            self.member_encoder = nn.Sequential(
                nn.Linear(hidden_dim + embedding_dim + 1, hidden_dim), nn.ReLU(),
            )
            self.event_encoder = nn.Sequential(
                nn.Linear(hidden_dim + embedding_dim + 3, hidden_dim), nn.ReLU(),
            )
            self.event_gate = nn.Linear(hidden_dim, 1)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim if variant == "base" else 2 * hidden_dim, hidden_dim),
            nn.ReLU(), nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        base = self.transaction_encoder(batch["base"])
        if self.variant == "base":
            return self.scorer(base).squeeze(-1)
        if self.variant == "flat":
            summary = self.flat_encoder(batch["flat"])
        else:
            members = self.transaction_encoder(batch["members"])
            roles = self.role_embedding(batch["roles"])
            elapsed = batch["elapsed"]
            if self.variant in ("no_role", "untyped_event"):
                roles = torch.zeros_like(roles)
            if self.variant in ("no_time", "untyped_event"):
                elapsed = torch.zeros_like(elapsed)
            members = self.member_encoder(torch.cat((members, roles, elapsed), dim=-1))
            member_mask = batch["member_mask"].unsqueeze(-1)
            grouped = (members * member_mask).sum(dim=2) / member_mask.sum(dim=2).clamp_min(1)
            types = self.type_embedding(batch["types"])
            descriptors = batch["descriptors"]
            if self.variant == "no_time":
                descriptors = descriptors.clone()
                descriptors[..., 0] = 0
            if self.variant == "untyped_event":
                types = torch.zeros_like(types)
                descriptors = torch.zeros_like(descriptors)
            events = self.event_encoder(torch.cat((grouped, types, descriptors), dim=-1))
            mask = batch["event_mask"].unsqueeze(-1)
            gates = torch.sigmoid(self.event_gate(events)) * mask
            # A learned gate can suppress the sole event, too. Normalize by
            # event count, not gate sum, and retain zero for empty histories.
            summary = (events * gates).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.scorer(torch.cat((base, summary), dim=-1)).squeeze(-1)


class NeuralDetector:
    """Knowledge-event aggregation prototype, trained with minibatch BCE.

    ``flat`` uses per-type event count, member count, mean span and mean
    amount ratio. ``untyped_event`` uses only the grouped member features.
    Scaling statistics are fit on training transactions and their historical
    members, then frozen for inference. Balanced weighting uses training-label
    counts only. Scores are uncalibrated; threshold selection is external.
    ``fixed_periodic`` retains named ``hour_sin``/``hour_cos`` coordinates at
    their intrinsic scale; the default ``standard`` preserves prior behavior.
    """

    def __init__(
        self, feature_names: Sequence[str], variant: str = "event",
        hidden_dim: int = 32, seed: int = 42, scaling: str = "standard",
    ) -> None:
        self.feature_names = tuple(feature_names)
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be nonempty and unique")
        if any(not isinstance(name, str) or not name for name in self.feature_names):
            raise ValueError("feature_names must contain nonempty strings")
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}; expected one of {VARIANTS}")
        if scaling not in SCALING_OPTIONS:
            raise ValueError(f"Unknown scaling {scaling!r}; expected one of {SCALING_OPTIONS}")
        if hidden_dim < 4:
            raise ValueError("hidden_dim must be at least 4")
        self.variant = variant
        self.hidden_dim = int(hidden_dim)
        self.seed = int(seed)
        self.scaling = scaling
        self.scaler: dict[str, list[float]] | None = None
        self.training_config: dict[str, Any] = {}
        self.batch_size = 64
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.model = _DetectorNetwork(len(self.feature_names), self.hidden_dim, variant).cpu()

    @property
    def config(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names), "variant": self.variant,
            "hidden_dim": self.hidden_dim, "seed": self.seed, "scaling": self.scaling,
        }

    def _apply_scaling_policy(self, scaler: dict[str, list[float]]) -> dict[str, list[float]]:
        if self.scaling == "fixed_periodic":
            for index, name in enumerate(self.feature_names):
                if name in FIXED_PERIODIC_FEATURE_NAMES:
                    # These two features are defined on [-1,1]. A short early
                    # training interval can have tiny variance, making later
                    # phases explode under Z-score scaling. Fixed coordinates
                    # depend only on their exact names, never on future rows.
                    scaler["mean"][index] = 0.0
                    scaler["scale"][index] = 1.0
        return scaler

    def _scaling_metadata(self) -> dict[str, Any]:
        fixed = [name for name in self.feature_names
                 if self.scaling == "fixed_periodic" and name in FIXED_PERIODIC_FEATURE_NAMES]
        return {
            "scaling": self.scaling,
            "scaler_scope": (
                "training_transactions_and_historical_members_except_fixed_periodic"
                if fixed else "training_transactions_and_historical_members"
            ),
            "fixed_scale_feature_names": fixed,
        }

    def _check_features(self, values: Sequence[float]) -> None:
        if len(values) != len(self.feature_names):
            raise ValueError("Transaction feature dimension does not match feature_names")
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Transaction features must be finite")

    def _validate_samples(self, samples: Sequence[Any]) -> None:
        for sample in samples:
            self._check_features(sample.features)
            for event in sample.events:
                if not event.members:
                    raise ValueError("Event instances must have at least one member")
                if not all(math.isfinite(float(value)) and float(value) >= 0
                           for value in (event.span_seconds, event.amount_ratio)):
                    raise ValueError("Event span and amount ratio must be finite and nonnegative")
                for member in event.members:
                    self._check_features(member.features)
                    if not math.isfinite(float(member.elapsed_seconds)) or member.elapsed_seconds < 0:
                        raise ValueError("Member elapsed time must be finite and nonnegative")

    def _fit_scaler(self, samples: Sequence[Any]) -> None:
        # Welford updates avoid materializing all duplicated history vectors.
        accumulator = _ScalerAccumulator(len(self.feature_names))
        for sample in samples:
            for vector in _feature_vectors(sample):
                accumulator.update(vector)
        # Constant training columns use unit scale, rather than amplifying a
        # previously unseen value by an arbitrary million-fold factor.
        self.scaler = self._apply_scaling_policy(accumulator.finish())

    def _training_batches(
        self,
        batch_factory: Callable[[], Iterator[tuple[list[TransactionSample], list[int]]]],
        identity: _TrainingStreamIdentity,
    ) -> Iterator[tuple[Sequence[TransactionSample], Sequence[int]]]:
        """Validate one source batch at a time, retaining no earlier samples."""
        for item in batch_factory():
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("Each training batch must be a (samples, labels) pair")
            samples, labels = item
            if (
                not isinstance(samples, Sequence) or isinstance(samples, (str, bytes))
                or not isinstance(labels, Sequence) or isinstance(labels, (str, bytes))
                or not samples or len(samples) != len(labels)
            ):
                raise ValueError("Training batches must be nonempty with equally sized samples and labels")
            if any(type(label) is not int or label not in (0, 1) for label in labels):
                raise ValueError("Training labels must be binary integers (0 or 1), not booleans")
            self._validate_samples(samples)
            for sample, label in zip(samples, labels):
                identity.update(sample, label)
            yield samples, labels

    def _minibatches(
        self,
        batches: Iterator[tuple[Sequence[TransactionSample], Sequence[int]]],
    ) -> Iterator[tuple[Sequence[TransactionSample], Sequence[int]]]:
        for samples, labels in batches:
            for offset in range(0, len(samples), self.batch_size):
                yield samples[offset:offset + self.batch_size], labels[offset:offset + self.batch_size]

    @staticmethod
    def _shuffle_batches(
        batches: Iterator[tuple[Sequence[TransactionSample], Sequence[int]]],
        buffer_size: int, generator: random.Random,
    ) -> Iterator[tuple[Sequence[TransactionSample], Sequence[int]]]:
        # Buffer whole minibatches after slicing. This is a bounded streaming
        # order, not the global sample permutation used by ``fit``.
        buffer: list[tuple[Sequence[TransactionSample], Sequence[int]]] = []
        for batch in batches:
            buffer.append(batch)
            if len(buffer) >= buffer_size:
                yield buffer.pop(generator.randrange(len(buffer)))
        while buffer:
            yield buffer.pop(generator.randrange(len(buffer)))

    def _tensorize(self, samples: Sequence[Any]) -> dict[str, torch.Tensor]:
        if self.scaler is None:
            raise RuntimeError("Call fit before inference")
        batch_count, feature_dim = len(samples), len(self.feature_names)
        mean = torch.tensor(self.scaler["mean"], dtype=torch.float32)
        scale = torch.tensor(self.scaler["scale"], dtype=torch.float32)

        def standardize(values: Sequence[float]) -> torch.Tensor:
            return (torch.tensor(values, dtype=torch.float32) - mean) / scale

        result = {"base": torch.stack([standardize(sample.features) for sample in samples])}
        if self.variant == "base":
            return result
        if self.variant == "flat":
            flat = torch.zeros(batch_count, len(MOTIF_TYPES) * 4)
            for row, sample in enumerate(samples):
                for type_index, motif_type in enumerate(MOTIF_TYPES):
                    events = [event for event in sample.events if event.motif_type == motif_type]
                    if events:
                        count = len(events)
                        flat[row, type_index * 4:type_index * 4 + 4] = torch.tensor([
                            _log_size(count), _log_size(sum(len(event.members) for event in events)),
                            _log_size(sum(event.span_seconds for event in events) / count) / 10,
                            _log_size(sum(event.amount_ratio for event in events) / count),
                        ])
            result["flat"] = flat
            return result
        event_count = max(1, max(len(sample.events) for sample in samples))
        member_count = max(1, max((len(event.members) for sample in samples for event in sample.events), default=0))
        result.update({
            "members": torch.zeros(batch_count, event_count, member_count, feature_dim),
            "roles": torch.zeros(batch_count, event_count, member_count, dtype=torch.long),
            "elapsed": torch.zeros(batch_count, event_count, member_count, 1),
            "member_mask": torch.zeros(batch_count, event_count, member_count),
            "types": torch.zeros(batch_count, event_count, dtype=torch.long),
            "descriptors": torch.zeros(batch_count, event_count, 3),
            "event_mask": torch.zeros(batch_count, event_count),
        })
        role_indices = {name: index + 1 for index, name in enumerate(MEMBER_ROLES)}
        type_indices = {name: index + 1 for index, name in enumerate(MOTIF_TYPES)}
        for row, sample in enumerate(samples):
            for event_index, event in enumerate(sample.events):
                result["event_mask"][row, event_index] = 1
                result["types"][row, event_index] = type_indices.get(event.motif_type, 0)
                result["descriptors"][row, event_index] = torch.tensor([
                    _log_size(event.span_seconds) / 10,
                    _log_size(event.amount_ratio), _log_size(len(event.members)),
                ])
                for member_index, member in enumerate(event.members):
                    coordinates = (row, event_index, member_index)
                    result["members"][coordinates] = standardize(member.features)
                    result["roles"][coordinates] = role_indices.get(member.role, 0)
                    result["elapsed"][coordinates + (0,)] = _log_size(member.elapsed_seconds) / 10
                    result["member_mask"][coordinates] = 1
        return result

    def fit(
        self, samples: Sequence[Any], labels: Sequence[int], *, epochs: int = 20,
        learning_rate: float = 0.001, batch_size: int = 64,
        class_weight: str = "none",
    ) -> list[dict[str, float | int]]:
        samples, labels = list(samples), list(labels)
        if not samples or len(samples) != len(labels):
            raise ValueError("Training samples and labels must be nonempty and have equal length")
        if any(type(label) is not int or label not in (0, 1) for label in labels):
            raise ValueError("Training labels must be binary integers (0 or 1), not booleans")
        if class_weight not in ("none", "balanced"):
            raise ValueError("class_weight must be 'none' or 'balanced'")
        positives = sum(labels)
        negatives = len(labels) - positives
        if class_weight == "balanced" and (positives == 0 or negatives == 0):
            raise ValueError("Balanced class weighting requires both classes in training labels")
        positive_weight = negatives / positives if class_weight == "balanced" else 1.0
        if epochs < 1 or batch_size < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("epochs, batch_size and learning_rate must be positive")
        self._validate_samples(samples)
        self._fit_scaler(samples)
        self.batch_size = int(batch_size)
        self.training_config = {
            "epochs": int(epochs), "learning_rate": float(learning_rate),
            "batch_size": self.batch_size, "training_sample_count": len(samples),
            "class_weight": class_weight, "positive_weight": float(positive_weight),
            **self._scaling_metadata(),
        }
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(positive_weight, dtype=torch.float32),
        )
        targets = torch.tensor(labels, dtype=torch.float32)
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        history = []
        self.model.train()
        for epoch in range(epochs):
            permutation = torch.randperm(len(samples), generator=generator).tolist()
            total_loss = 0.0
            for offset in range(0, len(samples), self.batch_size):
                indices = permutation[offset:offset + self.batch_size]
                batch = self._tensorize([samples[index] for index in indices])
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.model(batch), targets[indices])
                if not torch.isfinite(loss):
                    raise ValueError("Training loss is not finite; inspect feature values")
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * len(indices)
            history.append({"epoch": epoch + 1, "loss": total_loss / len(samples)})
        self.model.eval()
        return history

    def fit_batches(
        self,
        batch_factory: Callable[[], Iterator[tuple[list[TransactionSample], list[int]]]],
        *, epochs: int = 20, learning_rate: float = 0.001, batch_size: int = 64,
        class_weight: str = "none", shuffle_buffer_batches: int = 0,
    ) -> list[dict[str, float | int]]:
        """Train with a restartable, ordered factory containing training rows.

        The first full pass validates inputs and fits scaling and class counts.
        Each epoch restarts the factory and checks its complete ordered digest.
        Validation/test samples must never enter this factory. By default the
        source order is retained; optional minibatch shuffling retains at most
        ``shuffle_buffer_batches`` minibatches. Neither order is equivalent to
        ``fit``'s full-dataset random permutation. Only one minibatch is
        tensorized at a time, with no full-dataset labels or permutation tensor.
        """
        if not callable(batch_factory):
            raise ValueError("batch_factory must be callable and restartable")
        if (
            type(epochs) is not int or epochs < 1
            or type(batch_size) is not int or batch_size < 1
            or isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (int, float))
            or not math.isfinite(learning_rate) or learning_rate <= 0
        ):
            raise ValueError("epochs, batch_size and learning_rate must be positive")
        if type(shuffle_buffer_batches) is not int or shuffle_buffer_batches < 0:
            raise ValueError("shuffle_buffer_batches must be a nonnegative integer")
        if class_weight not in ("none", "balanced"):
            raise ValueError("class_weight must be 'none' or 'balanced'")

        expected = _TrainingStreamIdentity()
        accumulator = _ScalerAccumulator(len(self.feature_names))
        for samples, _ in self._training_batches(batch_factory, expected):
            for sample in samples:
                for vector in _feature_vectors(sample):
                    accumulator.update(vector)
        if expected.count == 0:
            raise ValueError("Training batches must contain at least one sample")
        if class_weight == "balanced" and (expected.positives == 0 or expected.negatives == 0):
            raise ValueError("Balanced class weighting requires both classes in training labels")
        scaler = self._apply_scaling_policy(accumulator.finish())
        positive_weight = expected.negatives / expected.positives if class_weight == "balanced" else 1.0
        self.scaler = scaler
        self.batch_size = batch_size
        self.training_config = {
            "epochs": epochs, "learning_rate": float(learning_rate),
            "batch_size": batch_size, "training_sample_count": expected.count,
            "class_weight": class_weight, "positive_weight": float(positive_weight),
            "class_counts": {"negative": expected.negatives, "positive": expected.positives},
            "training_mode": "streaming_batches", "training_status": "in_progress",
            "seed": self.seed, "shuffle_buffer_batches": shuffle_buffer_batches,
            "sequence_strategy": "bounded_batch_shuffle" if shuffle_buffer_batches else "source_order",
            **self._scaling_metadata(),
            "scaler_vector_count": accumulator.count,
            "source_fingerprint_sha256": expected.digest.hexdigest(),
            "fingerprint_scope": "ordered_training_ids_timestamps_features_events_labels",
            "source_factory_passes": 1, "completed_epochs": 0,
        }
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, dtype=torch.float32))
        generator = random.Random(self.seed)
        history: list[dict[str, float | int]] = []
        self.model.train()
        try:
            for epoch in range(epochs):
                self.training_config["source_factory_passes"] += 1
                observed = _TrainingStreamIdentity()
                batches = self._minibatches(self._training_batches(batch_factory, observed))
                if shuffle_buffer_batches:
                    batches = self._shuffle_batches(batches, shuffle_buffer_batches, generator)
                total_loss = 0.0
                processed = 0
                for samples, labels in batches:
                    batch = self._tensorize(samples)
                    targets = torch.tensor(labels, dtype=torch.float32)
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(self.model(batch), targets)
                    if not torch.isfinite(loss):
                        raise ValueError("Training loss is not finite; inspect feature values")
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.detach()) * len(samples)
                    processed += len(samples)
                if observed.signature() != expected.signature() or processed != expected.count:
                    raise ValueError("Training batch factory must restart with unchanged samples, labels and source order")
                history.append({"epoch": epoch + 1, "loss": total_loss / processed})
                self.training_config["completed_epochs"] = epoch + 1
        except Exception:
            self.training_config["training_status"] = "failed"
            raise
        finally:
            self.model.eval()
        self.training_config["training_status"] = "completed"
        return history

    def predict(self, samples: Sequence[Any]) -> list[float]:
        if self.scaler is None:
            raise RuntimeError("Call fit before inference")
        samples = list(samples)
        self._validate_samples(samples)
        scores: list[float] = []
        self.model.eval()
        with torch.no_grad():
            for offset in range(0, len(samples), self.batch_size):
                batch = self._tensorize(samples[offset:offset + self.batch_size])
                values = torch.sigmoid(self.model(batch))
                if not torch.isfinite(values).all():
                    raise ValueError("Predicted scores are not finite; inspect feature values")
                scores.extend(values.tolist())
        return scores

    def predict_batches(
        self, batch_factory: Callable[[], Iterator[list[TransactionSample]]],
    ) -> Iterator[list[float]]:
        """Yield one score list per input batch without collecting the stream."""
        if self.scaler is None:
            raise RuntimeError("Call fit before inference")
        if not callable(batch_factory):
            raise ValueError("batch_factory must be callable")
        for samples in batch_factory():
            if (
                not isinstance(samples, Sequence) or isinstance(samples, (str, bytes))
                or not samples
            ):
                raise ValueError("Prediction batches must be nonempty sample sequences")
            # ``predict`` limits tensors to self.batch_size. Its no_grad context
            # exits before yield, so the caller's gradient mode stays unchanged.
            yield self.predict(samples)

    def save(self, path: str | Path) -> None:
        if self.scaler is None:
            raise RuntimeError("Cannot save an unfitted detector")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1, "config": self.config, "scaler": self.scaler,
            "training_config": self.training_config,
            "vocabulary": {"motif_types": list(MOTIF_TYPES), "member_roles": list(MEMBER_ROLES)},
            "state_dict": self.model.state_dict(),
        }
        descriptor, temporary_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        os.close(descriptor)
        try:
            torch.save(payload, temporary_path)
            os.replace(temporary_path, path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    @classmethod
    def load(cls, path: str | Path) -> "NeuralDetector":
        # Only primitive configuration and tensors are saved. Loading never
        # opts into arbitrary Python pickle object execution.
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported neural detector checkpoint schema")
        vocabulary = {"motif_types": list(MOTIF_TYPES), "member_roles": list(MEMBER_ROLES)}
        if payload.get("vocabulary") != vocabulary:
            raise ValueError("Checkpoint event vocabulary differs from this installation")
        detector = cls(**payload["config"])
        scaler = payload["scaler"]
        dimension = len(detector.feature_names)
        if len(scaler["mean"]) != dimension or len(scaler["scale"]) != dimension:
            raise ValueError("Checkpoint scaler has an inconsistent feature dimension")
        if not all(math.isfinite(value) for value in scaler["mean"]):
            raise ValueError("Checkpoint means must be finite")
        if not all(math.isfinite(value) and value > 0 for value in scaler["scale"]):
            raise ValueError("Checkpoint scales must be finite and positive")
        if detector.scaling == "fixed_periodic":
            for index, name in enumerate(detector.feature_names):
                if name in FIXED_PERIODIC_FEATURE_NAMES and (
                    scaler["mean"][index] != 0.0 or scaler["scale"][index] != 1.0
                ):
                    raise ValueError("Checkpoint fixed_periodic scaler differs from the feature definition")
        detector.scaler = {"mean": list(scaler["mean"]), "scale": list(scaler["scale"])}
        detector.training_config = payload.get("training_config", {})
        if detector.training_config.get("scaling", detector.scaling) != detector.scaling:
            raise ValueError("Checkpoint training scaling differs from detector configuration")
        detector.batch_size = int(detector.training_config.get("batch_size", 64))
        if detector.batch_size < 1:
            raise ValueError("Checkpoint batch_size must be positive")
        detector.model.load_state_dict(payload["state_dict"], strict=True)
        detector.model.eval()
        return detector
