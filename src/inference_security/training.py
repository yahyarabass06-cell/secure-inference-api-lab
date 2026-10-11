"""Train on one period and select a threshold on benign calibration only."""

import hashlib
import json
import math
import warnings
from datetime import datetime
from importlib.metadata import version

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from inference_security.measurements import extract_features, parse_measurements
from inference_security.model import LogisticModel, reject_constant, unique_object


def load_period(path):
    with path.open("rb") as source:
        raw = source.read(2_000_001)
    if not raw or len(raw) > 2_000_000:
        raise ValueError("Invalid dataset size.")

    rows = []
    ids = set()
    previous = None
    for line in raw.decode("utf-8").splitlines():
        data = json.loads(
            line,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
        expected = {
            "record_id",
            "group_id",
            "timestamp",
            "scenario",
            "label",
            "measurements",
        }
        if not isinstance(data, dict) or set(data) != expected:
            raise ValueError("Unexpected dataset fields.")

        for name in ("record_id", "group_id", "scenario"):
            value = data[name]
            if (
                type(value) is not str
                or not value
                or value != value.strip()
                or len(value) > 128
                or any(ord(character) < 32 for character in value)
            ):
                raise ValueError("Invalid dataset identifier.")

        if type(data["label"]) is not int or data["label"] not in (0, 1):
            raise ValueError("Invalid binary label.")
        if type(data["timestamp"]) is not str:
            raise ValueError("Invalid timestamp.")
        timestamp = datetime.fromisoformat(data["timestamp"])
        if timestamp.utcoffset() is None:
            raise ValueError("Timezone required.")
        if previous is not None and timestamp <= previous:
            raise ValueError("Dataset must be strictly chronological.")
        previous = timestamp

        if data["record_id"] in ids:
            raise ValueError("Duplicate record identifier.")
        ids.add(data["record_id"])
        rows.append(
            {
                **data,
                "timestamp": timestamp,
                "measurements": parse_measurements(data["measurements"]),
            }
        )
        if len(rows) > 10000:
            raise ValueError("Too many dataset records.")

    if not rows:
        raise ValueError("Empty dataset.")
    return rows, hashlib.sha256(raw).hexdigest()


def check_separation(earlier, later):
    if earlier[-1]["timestamp"] >= later[0]["timestamp"]:
        raise ValueError("Dataset periods overlap.")
    for field in ("record_id", "group_id"):
        if {row[field] for row in earlier} & {row[field] for row in later}:
            raise ValueError(f"Dataset {field} overlap.")


def train_model(train_path, calibration_path):
    training, train_hash = load_period(train_path)
    calibration, calibration_hash = load_period(calibration_path)
    check_separation(training, calibration)

    labels = [row["label"] for row in training]
    if set(labels) != {0, 1}:
        raise ValueError("Training requires both classes.")
    benign = [row for row in calibration if row["label"] == 0]
    if len(benign) < 20:
        raise ValueError("At least twenty benign calibration records required.")

    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000),
    )
    features = [extract_features(row["measurements"]) for row in training]
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(features, labels)

    scaler = pipeline.named_steps["standardscaler"]
    classifier = pipeline.named_steps["logisticregression"]
    if classifier.classes_.tolist() != [0, 1]:
        raise ValueError("Unexpected class ordering.")

    preliminary = LogisticModel(
        mean=tuple(float(value) for value in scaler.mean_),
        scale=tuple(float(value) for value in scaler.scale_),
        coefficients=tuple(float(value) for value in classifier.coef_[0]),
        intercept=float(classifier.intercept_[0]),
        threshold=0.5,
    )

    scores = sorted(preliminary.probability(row["measurements"]) for row in benign)
    target_fpr = 0.05
    rank = math.ceil((1 - target_fpr) * len(scores)) - 1
    threshold = scores[rank]

    model = LogisticModel(
        mean=preliminary.mean,
        scale=preliminary.scale,
        coefficients=preliminary.coefficients,
        intercept=preliminary.intercept,
        threshold=threshold,
    )

    combined = training + calibration
    reference = pipeline.predict_proba(
        [extract_features(row["measurements"]) for row in combined]
    )[:, 1]
    maximum_error = max(
        abs(model.probability(row["measurements"]) - float(expected))
        for row, expected in zip(combined, reference, strict=True)
    )
    if maximum_error > 1e-12:
        raise ValueError("Exported inference differs from fitted pipeline.")

    report = {
        "schema_version": 1,
        "training_records": len(training),
        "calibration_records": len(calibration),
        "train_sha256": train_hash,
        "calibration_sha256": calibration_hash,
        "scikit_learn_version": version("scikit-learn"),
        "settings": {
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 1000,
            "threshold_rule": "strictly greater than benign 95th percentile",
        },
        "threshold": threshold,
        "target_calibration_false_positive_rate": target_fpr,
        "observed_calibration_false_positive_rate": (
            sum(score > threshold for score in scores) / len(scores)
        ),
        "maximum_export_probability_error": maximum_error,
        "test_period_used": False,
    }
    return model, report
