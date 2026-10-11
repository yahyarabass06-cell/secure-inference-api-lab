import json
from dataclasses import replace

import pytest

from inference_security.measurements import Measurements
from inference_security.model import LogisticModel, model_fields, parse_model


@pytest.fixture
def model():
    return LogisticModel(
        mean=(0, 0, 0),
        scale=(1, 1, 1),
        coefficients=(0, 0, 0),
        intercept=0,
        threshold=0.5,
    )


def test_model_round_trip(model):
    raw = json.dumps(model_fields(model), allow_nan=False).encode()
    assert parse_model(raw) == model


def test_threshold_tie_is_not_flagged(model):
    result = model.predict(Measurements(10, 100, 50000))
    assert result == {
        "probability": 0.5,
        "flagged": False,
        "threshold": 0.5,
    }


@pytest.mark.parametrize("intercept, expected", [(1000, 1), (-1000, 0)])
def test_extreme_scores_do_not_overflow(model, intercept, expected):
    changed = replace(model, intercept=intercept)
    assert changed.probability(Measurements(10, 100, 50000)) == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mean", [0, 0, 0]),
        ("mean", (0, 0)),
        ("mean", (True, 0, 0)),
        ("scale", (0, 1, 1)),
        ("scale", (-1, 1, 1)),
        ("scale", (0.0000001, 1, 1)),
        ("coefficients", (float("nan"), 0, 0)),
        ("coefficients", (1_000_001, 0, 0)),
        ("intercept", float("inf")),
        ("threshold", True),
        ("threshold", 0),
        ("threshold", 1),
    ],
)
def test_invalid_parameters_are_rejected(model, field, value):
    with pytest.raises(ValueError):
        replace(model, **{field: value})


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"\xff",
        b"[]",
        b'{"schema_version": 1, "schema_version": 1}',
        b'{"threshold": NaN}',
        b"x" * 16385,
    ],
)
def test_invalid_model_bytes_are_rejected(raw):
    with pytest.raises(ValueError):
        parse_model(raw)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("model_type", "pickle"),
        ("feature_names", ["label", "record_id", "scenario"]),
        ("mean", None),
        ("coefficients", [0, 0]),
    ],
)
def test_invalid_model_contract_is_rejected(model, field, value):
    data = model_fields(model)
    data[field] = value
    with pytest.raises(ValueError):
        parse_model(json.dumps(data).encode())


def test_extra_model_field_is_rejected(model):
    data = model_fields(model)
    data["execute"] = "unused"
    with pytest.raises(ValueError):
        parse_model(json.dumps(data).encode())
