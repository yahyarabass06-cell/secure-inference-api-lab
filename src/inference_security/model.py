"""Strict JSON model parameters and inference without deserialization."""

import json
import math
from dataclasses import dataclass

from inference_security.measurements import FEATURE_NAMES, extract_features


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field.")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"Invalid JSON constant: {value}")


def valid_number(value):
    return (
        type(value) in (int, float) and math.isfinite(value) and abs(value) <= 1_000_000
    )


@dataclass(frozen=True)
class LogisticModel:
    mean: tuple
    scale: tuple
    coefficients: tuple
    intercept: float
    threshold: float

    def __post_init__(self):
        for name in ("mean", "scale", "coefficients"):
            values = getattr(self, name)
            if (
                type(values) is not tuple
                or len(values) != len(FEATURE_NAMES)
                or not all(valid_number(value) for value in values)
            ):
                raise ValueError(f"Invalid model vector: {name}")

        if not all(value >= 0.000001 for value in self.scale):
            raise ValueError("Model scales must be positive and bounded.")
        if not valid_number(self.intercept):
            raise ValueError("Invalid intercept.")
        if not valid_number(self.threshold) or not 0 < self.threshold < 1:
            raise ValueError("Threshold must be strictly between zero and one.")

    def probability(self, measurements):
        features = extract_features(measurements)
        score = (
            math.fsum(
                coefficient * ((feature - mean) / scale)
                for feature, mean, scale, coefficient in zip(
                    features, self.mean, self.scale, self.coefficients, strict=True
                )
            )
            + self.intercept
        )

        if score >= 0:
            return 1 / (1 + math.exp(-score))
        exponential = math.exp(score)
        return exponential / (1 + exponential)

    def predict(self, measurements):
        probability = self.probability(measurements)
        return {
            "probability": probability,
            "flagged": probability > self.threshold,
            "threshold": self.threshold,
        }


def model_fields(model):
    if not isinstance(model, LogisticModel):
        raise ValueError("Validated model required.")
    return {
        "schema_version": 1,
        "model_type": "standardized_logistic_regression",
        "feature_names": list(FEATURE_NAMES),
        "mean": list(model.mean),
        "scale": list(model.scale),
        "coefficients": list(model.coefficients),
        "intercept": model.intercept,
        "threshold": model.threshold,
    }


def parse_model(raw):
    if type(raw) is not bytes or not 0 < len(raw) <= 16384:
        raise ValueError("Model must contain at most 16384 bytes.")

    try:
        data = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeError, RecursionError) as error:
        raise ValueError("Invalid model encoding or nesting.") from error

    expected = {
        "schema_version",
        "model_type",
        "feature_names",
        "mean",
        "scale",
        "coefficients",
        "intercept",
        "threshold",
    }
    if not isinstance(data, dict) or set(data) != expected:
        raise ValueError("Unexpected model fields.")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("Unsupported model schema.")
    if data["model_type"] != "standardized_logistic_regression":
        raise ValueError("Unsupported model type.")
    if data["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("Feature contract mismatch.")

    for name in ("mean", "scale", "coefficients"):
        if type(data[name]) is not list:
            raise ValueError("Model vectors must be JSON arrays.")

    return LogisticModel(
        mean=tuple(data["mean"]),
        scale=tuple(data["scale"]),
        coefficients=tuple(data["coefficients"]),
        intercept=data["intercept"],
        threshold=data["threshold"],
    )
