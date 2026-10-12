"""Optional native LightGBM controls for the causal transaction features.

The flat control adds exactly the per-type event summaries used by
``neural.flat``. Member features, identifiers, roles and absolute timestamps
are not encoded.
NumPy and LightGBM are loaded only when training or loading a fitted model.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Sequence

from .stream import MOTIF_TYPES


VARIANTS = ("lgbm_base", "lgbm_flat")
FLAT_FEATURE_NAMES = tuple(
    f"{motif_type}_{name}"
    for motif_type in MOTIF_TYPES
    for name in (
        "log_event_count", "log_member_count", "log_mean_span_seconds_scaled",
        "log_mean_amount_ratio",
    )
)


def _dependencies() -> tuple[Any, Any]:
    try:
        import numpy
        import lightgbm
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "LightGBM detection requires the optional numpy and lightgbm packages. "
            "Install numpy and lightgbm; if the native library cannot load on "
            "macOS, install the OpenMP runtime (libomp)."
        ) from exc
    return numpy, lightgbm


def _finite(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result) or (nonnegative and result < 0):
        qualifier = "finite and nonnegative" if nonnegative else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def flat_event_summary(events: Sequence[Any]) -> tuple[float, ...]:
    """Return the same 16 numeric summaries as the neural flat control.

    The vocabulary order is ``MOTIF_TYPES``. For each type the values are
    log1p(event count), log1p(member count), log1p(mean span)/10 and
    log1p(mean amount ratio). Unrecognized types have no contribution.
    """
    result: list[float] = []
    for motif_type in MOTIF_TYPES:
        matches = [event for event in events if event.motif_type == motif_type]
        if not matches:
            result.extend((0.0, 0.0, 0.0, 0.0))
            continue
        member_count = 0
        spans, ratios = [], []
        for event in matches:
            if not event.members:
                raise ValueError("Event instances must have at least one member")
            member_count += len(event.members)
            spans.append(_finite(event.span_seconds, "Event span", nonnegative=True))
            ratios.append(_finite(event.amount_ratio, "Event amount ratio", nonnegative=True))
        count = len(matches)
        # Deliberately preserve neural.flat's log-of-mean formula and order.
        mean_span = sum(spans) / count
        mean_ratio = sum(ratios) / count
        if not math.isfinite(mean_span) or not math.isfinite(mean_ratio):
            raise ValueError("Aggregated event span and amount ratio must be finite")
        result.extend((math.log1p(count), math.log1p(member_count),
                       math.log1p(mean_span) / 10, math.log1p(mean_ratio)))
    return tuple(result)


class LightGBMDetector:
    """A binary CPU LightGBM detector with training-only preprocessing.

    ``fit`` accepts only training samples and separately supplied labels.
    Base-feature means and population standard deviations use those current
    transactions only, and are frozen for prediction. The flat event summary
    is appended without further scaling. Balanced sample weights are
    n_samples / (2 * training_class_count), with both training classes needed.
    Scores are uncalibrated; validation threshold selection is external.

    ``save('model.txt')`` writes a native text model and ``model.json``. The
    JSON sidecar records configuration, scaler and ordered input names, and
    binds them to the model with SHA256. Loading never uses pickle.
    """

    def __init__(
        self, feature_names: Sequence[str], variant: str = "lgbm_base",
        seed: int = 42, n_estimators: int = 100, learning_rate: float = 0.05,
        num_leaves: int = 15, class_weight: str = "balanced",
    ) -> None:
        if isinstance(feature_names, (str, bytes)):
            raise ValueError("feature_names must be a sequence of nonempty unique strings")
        try:
            self.feature_names = tuple(feature_names)
        except TypeError as exc:
            raise ValueError("feature_names must be a sequence of nonempty unique strings") from exc
        if not self.feature_names or any(
            not isinstance(name, str) or not name or any(character.isspace() for character in name)
            or any(character in name for character in '[]{}:,"')
            for name in self.feature_names
        ):
            raise ValueError("feature_names must contain nonempty LightGBM-compatible strings")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be unique")
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}; expected one of {VARIANTS}")
        for name, value, minimum, maximum in (
            ("seed", seed, 0, 2147483647),
            ("n_estimators", n_estimators, 1, None),
            ("num_leaves", num_leaves, 2, 131072),
        ):
            if (isinstance(value, bool) or not isinstance(value, Integral)
                    or value < minimum or (maximum is not None and value > maximum)):
                bound = f" between {minimum} and {maximum}" if maximum is not None else f" >= {minimum}"
                raise ValueError(f"{name} must be an integer{bound}")
        rate = _finite(learning_rate, "learning_rate")
        if rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if class_weight not in ("none", "balanced"):
            raise ValueError("class_weight must be 'none' or 'balanced'")
        self.variant = variant
        self.seed = int(seed)
        self.n_estimators = int(n_estimators)
        self.learning_rate = rate
        self.num_leaves = int(num_leaves)
        self.class_weight = class_weight
        self.model_feature_names = self.feature_names + (FLAT_FEATURE_NAMES if variant == "lgbm_flat" else ())
        if len(set(self.model_feature_names)) != len(self.model_feature_names):
            raise ValueError("Base feature names must not collide with flat event feature names")
        self.scaler: dict[str, list[float]] | None = None
        self.training_config: dict[str, Any] = {}
        self.model: Any = None

    @property
    def config(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names), "variant": self.variant,
            "seed": self.seed, "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate, "num_leaves": self.num_leaves,
            "class_weight": self.class_weight,
        }

    def feature_vector(self, sample: Any) -> tuple[float, ...]:
        """Encode only permitted numerical inputs, before standardization."""
        values = tuple(sample.features)
        if len(values) != len(self.feature_names):
            raise ValueError("Transaction feature dimension does not match feature_names")
        base = tuple(_finite(value, "Transaction features") for value in values)
        return base + (flat_event_summary(sample.events) if self.variant == "lgbm_flat" else ())

    def _fit_scaler(self, rows: Sequence[Sequence[float]]) -> dict[str, list[float]]:
        dimension = len(self.feature_names)
        mean, squared = [0.0] * dimension, [0.0] * dimension
        for count, row in enumerate(rows, start=1):
            for index in range(dimension):
                value = row[index]
                delta = value - mean[index]
                mean[index] += delta / count
                squared[index] += delta * (value - mean[index])
        deviations = [math.sqrt(max(value / len(rows), 0.0)) for value in squared]
        scale = [value if value >= 1e-6 else 1.0 for value in deviations]
        if not all(math.isfinite(value) for value in mean + scale):
            raise ValueError("Training scaler is not finite; inspect feature magnitudes")
        return {"mean": mean, "scale": scale}

    def _matrix(self, rows: Sequence[Sequence[float]], scaler: dict[str, list[float]], numpy: Any) -> Any:
        dimension = len(self.feature_names)
        values = [
            [(row[index] - scaler["mean"][index]) / scaler["scale"][index]
             for index in range(dimension)] + list(row[dimension:])
            for row in rows
        ]
        if any(not math.isfinite(value) for row in values for value in row):
            raise ValueError("Standardized transaction features must be finite")
        return numpy.asarray(values, dtype=numpy.float64)

    def fit(self, samples: Sequence[Any], labels: Sequence[int]) -> list[dict[str, float | int]]:
        samples, labels = list(samples), list(labels)
        if not samples or len(samples) != len(labels):
            raise ValueError("Training samples and labels must be nonempty and have equal length")
        if any(isinstance(label, bool) or not isinstance(label, Integral) or label not in (0, 1)
               for label in labels):
            raise ValueError("Training labels must be binary integers (0 or 1), not booleans")
        checked_labels = [int(label) for label in labels]
        positives = sum(checked_labels)
        negatives = len(checked_labels) - positives
        if self.class_weight == "balanced" and (positives == 0 or negatives == 0):
            raise ValueError("Balanced class weighting requires both classes in training labels")
        class_weights = {
            "0": len(labels) / (2 * negatives) if self.class_weight == "balanced" else 1.0,
            "1": len(labels) / (2 * positives) if self.class_weight == "balanced" else 1.0,
        }
        rows = [self.feature_vector(sample) for sample in samples]
        scaler = self._fit_scaler(rows)
        numpy, lightgbm = _dependencies()
        data = lightgbm.Dataset(
            self._matrix(rows, scaler, numpy), label=numpy.asarray(checked_labels, dtype=numpy.int32),
            weight=numpy.asarray([class_weights[str(label)] for label in checked_labels], dtype=numpy.float64),
            feature_name=list(self.model_feature_names),
        )
        parameters = {
            "objective": "binary", "metric": "binary_logloss", "verbosity": -1,
            "learning_rate": self.learning_rate, "num_leaves": self.num_leaves,
            "seed": self.seed, "num_threads": 1, "deterministic": True,
            "force_col_wise": True,
        }
        evaluations: dict[str, Any] = {}
        model = lightgbm.train(
            parameters, data, num_boost_round=self.n_estimators,
            valid_sets=[data], valid_names=["training"],
            callbacks=[lightgbm.record_evaluation(evaluations)],
        )
        self.model, self.scaler = model, scaler
        self.training_config = {
            "training_sample_count": len(samples), "class_weight": self.class_weight,
            "class_counts": {"0": negatives, "1": positives}, "class_weights": class_weights,
            "positive_weight": negatives / positives if self.class_weight == "balanced" else 1.0,
            "n_estimators": self.n_estimators, "parameters": parameters,
        }
        return [{"epoch": index + 1, "loss": float(loss)} for index, loss in enumerate(
            evaluations["training"]["binary_logloss"]
        )]

    def predict(self, samples: Sequence[Any]) -> list[float]:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Call fit before inference")
        rows = [self.feature_vector(sample) for sample in samples]
        if not rows:
            return []
        numpy, _ = _dependencies()
        scores = self.model.predict(self._matrix(rows, self.scaler, numpy), num_threads=1).tolist()
        if any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
            raise ValueError("Predicted scores must be finite binary probabilities")
        return [float(score) for score in scores]

    def fit_matrix(self, features: Any, labels: Any, *, working_directory: str | Path | None = None,
                   chunk_rows: int = 4096) -> list[dict[str, float | int]]:
        """Fit from a numerical matrix, including a read-only NumPy memmap.

        Only training rows may be supplied. Temporary standardized values are
        disk-backed, but LightGBM's native binned dataset still consumes RAM.
        Input columns must match ``model_feature_names`` exactly.
        """
        numpy, lightgbm = _dependencies()
        values, targets = numpy.asarray(features), numpy.asarray(labels)
        if (values.ndim != 2 or not len(values) or values.shape[1] != len(self.model_feature_names)
                or values.dtype.kind not in "fiu" or targets.shape != (len(values),)
                or targets.dtype.kind not in "iu"):
            raise ValueError("Training matrix and integer labels have inconsistent shapes or types")
        if type(chunk_rows) is not int or chunk_rows < 1:
            raise ValueError("chunk_rows must be a positive integer")
        if not numpy.isin(targets, (0, 1)).all():
            raise ValueError("Training labels must be binary integers")
        positives = int(numpy.count_nonzero(targets))
        negatives = len(targets) - positives
        if self.class_weight == "balanced" and (not positives or not negatives):
            raise ValueError("Balanced class weighting requires both training classes")
        dimension = len(self.feature_names)
        mean, squared, count = numpy.zeros(dimension), numpy.zeros(dimension), 0
        for offset in range(0, len(values), chunk_rows):
            chunk = numpy.asarray(values[offset:offset + chunk_rows], dtype=numpy.float64)
            if not numpy.isfinite(chunk).all():
                raise ValueError("Training matrix contains nonfinite features")
            base = chunk[:, :dimension]
            group_count = len(base)
            group_mean = base.mean(axis=0)
            group_squared = ((base - group_mean) ** 2).sum(axis=0)
            delta = group_mean - mean
            updated_count = count + group_count
            squared += group_squared + delta ** 2 * count * group_count / updated_count
            mean += delta * group_count / updated_count
            count = updated_count
        deviations = numpy.sqrt(numpy.maximum(squared / count, 0))
        scale = numpy.where(deviations >= 1e-6, deviations, 1.0)
        if not numpy.isfinite(mean).all() or not numpy.isfinite(scale).all():
            raise ValueError("Training scaler is not finite")
        weights = {"0": len(targets) / (2 * negatives) if self.class_weight == "balanced" else 1.0,
                   "1": len(targets) / (2 * positives) if self.class_weight == "balanced" else 1.0}
        parameters = {"objective": "binary", "metric": "binary_logloss", "verbosity": -1,
                      "learning_rate": self.learning_rate, "num_leaves": self.num_leaves,
                      "seed": self.seed, "num_threads": 1, "deterministic": True, "force_col_wise": True}
        evaluations: dict[str, Any] = {}
        with tempfile.TemporaryDirectory(prefix="hamlet-tree-", dir=working_directory) as temporary:
            matrix = numpy.lib.format.open_memmap(Path(temporary) / "standardized.npy", mode="w+",
                                                  dtype=numpy.float64, shape=values.shape)
            for offset in range(0, len(values), chunk_rows):
                chunk = numpy.asarray(values[offset:offset + chunk_rows], dtype=numpy.float64)
                matrix[offset:offset + len(chunk), :dimension] = (chunk[:, :dimension] - mean) / scale
                matrix[offset:offset + len(chunk), dimension:] = chunk[:, dimension:]
                if not numpy.isfinite(matrix[offset:offset + len(chunk)]).all():
                    raise ValueError("Standardized matrix contains nonfinite features")
            matrix.flush()
            data = lightgbm.Dataset(matrix, label=targets,
                                   weight=numpy.where(targets == 1, weights["1"], weights["0"]),
                                   feature_name=list(self.model_feature_names))
            model = lightgbm.train(parameters, data, num_boost_round=self.n_estimators,
                                   valid_sets=[data], valid_names=["training"],
                                   callbacks=[lightgbm.record_evaluation(evaluations)])
            model.free_dataset()
            del data, matrix
        self.model, self.scaler = model, {"mean": mean.tolist(), "scale": scale.tolist()}
        self.training_config = {"training_sample_count": count, "training_mode": "disk_backed_feature_matrix",
                                "class_weight": self.class_weight, "class_counts": {"0": negatives, "1": positives},
                                "class_weights": weights, "positive_weight": negatives / positives if self.class_weight == "balanced" else 1.0,
                                "n_estimators": self.n_estimators, "parameters": parameters,
                                "preprocessing_chunk_rows": chunk_rows,
                                "memory_scope": "source and standardized arrays may be memmaps; native LightGBM bins and weights consume RAM"}
        return [{"epoch": index + 1, "loss": float(loss)} for index, loss in enumerate(evaluations["training"]["binary_logloss"])]

    @staticmethod
    def _sidecar(path: Path) -> Path:
        if path.suffix.lower() == ".json":
            raise ValueError("Use a native model path such as model.txt, not the JSON sidecar")
        return path.with_suffix(".json")

    def save(self, path: str | Path) -> None:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Cannot save an unfitted detector")
        path = Path(path)
        sidecar = self._sidecar(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_paths: list[Path] = []
        backup_path: Path | None = None
        try:
            for destination in (path, sidecar):
                descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=path.parent)
                os.close(descriptor)
                temporary_paths.append(Path(name))
            self.model.save_model(str(temporary_paths[0]))
            payload = {
                "schema_version": 1, "format": "lightgbm_native_text",
                "config": self.config, "scaler": self.scaler,
                "model_feature_names": list(self.model_feature_names),
                "motif_types": list(MOTIF_TYPES), "training_config": self.training_config,
                "model_sha256": hashlib.sha256(temporary_paths[0].read_bytes()).hexdigest(),
            }
            temporary_paths[1].write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            if path.exists():
                descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.backup.", dir=path.parent)
                os.close(descriptor)
                backup_path = Path(name)
                temporary_paths.append(backup_path)
                shutil.copyfile(path, backup_path)
            os.replace(temporary_paths[0], path)
            try:
                os.replace(temporary_paths[1], sidecar)
            except OSError:
                # The old JSON still refers to the previous model. Preserve
                # that usable pair if publishing the new sidecar fails.
                if backup_path is not None:
                    os.replace(backup_path, path)
                else:
                    path.unlink(missing_ok=True)
                raise
        finally:
            for temporary_path in temporary_paths:
                temporary_path.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> "LightGBMDetector":
        path = Path(path)
        payload = json.loads(cls._sidecar(path).read_text(encoding="utf-8"))
        if (not isinstance(payload, dict) or payload.get("schema_version") != 1
                or payload.get("format") != "lightgbm_native_text"):
            raise ValueError("Unsupported LightGBM detector checkpoint schema")
        if payload.get("motif_types") != list(MOTIF_TYPES):
            raise ValueError("Checkpoint event vocabulary differs from this installation")
        try:
            detector = cls(**payload["config"])
        except (KeyError, TypeError) as exc:
            raise ValueError("Checkpoint detector configuration is invalid") from exc
        if payload.get("model_feature_names") != list(detector.model_feature_names):
            raise ValueError("Checkpoint feature order differs from detector configuration")
        scaler = payload.get("scaler")
        dimension = len(detector.feature_names)
        if (not isinstance(scaler, dict) or not isinstance(scaler.get("mean"), list)
                or not isinstance(scaler.get("scale"), list)
                or len(scaler["mean"]) != dimension or len(scaler["scale"]) != dimension):
            raise ValueError("Checkpoint scaler has an inconsistent feature dimension")
        means = [_finite(value, "Checkpoint means") for value in scaler["mean"]]
        scales = [_finite(value, "Checkpoint scales") for value in scaler["scale"]]
        if any(value <= 0 for value in scales):
            raise ValueError("Checkpoint scales must be finite and positive")
        if hashlib.sha256(path.read_bytes()).hexdigest() != payload.get("model_sha256"):
            raise ValueError("Checkpoint native model SHA256 does not match the JSON sidecar")
        training_config = payload.get("training_config", {})
        if not isinstance(training_config, dict):
            raise ValueError("Checkpoint training configuration must be a JSON object")
        _, lightgbm = _dependencies()
        model = lightgbm.Booster(model_file=str(path))
        if model.num_feature() != len(detector.model_feature_names) or model.feature_name() != list(detector.model_feature_names):
            raise ValueError("Checkpoint native model has inconsistent feature names or dimension")
        objective = model.dump_model().get("objective", "")
        if not isinstance(objective, str) or objective.split()[:1] != ["binary"]:
            raise ValueError("Checkpoint native model must use the binary objective")
        detector.model, detector.scaler = model, {"mean": means, "scale": scales}
        detector.training_config = training_config
        return detector
