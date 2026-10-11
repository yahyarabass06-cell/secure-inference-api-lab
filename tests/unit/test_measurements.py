import math
from dataclasses import FrozenInstanceError, replace

import pytest

from inference_security.measurements import (
    FEATURE_NAMES,
    Measurements,
    extract_features,
    parse_measurements,
)


@pytest.fixture
def measurements():
    return Measurements(
        duration_seconds=10,
        packet_count=100,
        byte_count=50000,
    )


def test_feature_contract(measurements):
    assert FEATURE_NAMES == (
        "log_duration_seconds",
        "log_packet_rate",
        "log_bytes_per_packet",
    )
    assert extract_features(measurements) == pytest.approx(
        (math.log1p(10), math.log1p(10), math.log1p(500))
    )


def test_zero_bytes_are_supported(measurements):
    features = extract_features(replace(measurements, byte_count=0))
    assert features[2] == 0
    assert all(math.isfinite(value) for value in features)


def test_measurements_are_immutable(measurements):
    with pytest.raises(FrozenInstanceError):
        measurements.packet_count = 200


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("duration_seconds", True),
        ("duration_seconds", "10"),
        ("duration_seconds", 0),
        ("duration_seconds", 86401),
        ("duration_seconds", float("nan")),
        ("duration_seconds", float("inf")),
        ("packet_count", True),
        ("packet_count", 0),
        ("packet_count", 1.5),
        ("packet_count", 1_000_000_001),
        ("byte_count", True),
        ("byte_count", -1),
        ("byte_count", 1.5),
        ("byte_count", 1_000_000_000_001),
    ],
)
def test_invalid_measurements_are_rejected(measurements, field, value):
    with pytest.raises(ValueError):
        replace(measurements, **{field: value})


@pytest.mark.parametrize(
    "data",
    [
        None,
        [],
        {},
        {
            "duration_seconds": 10,
            "packet_count": 100,
            "byte_count": 50000,
            "label": 1,
        },
        {
            "duration_seconds": 10,
            "packet_count": 100,
        },
    ],
)
def test_inexact_request_fields_are_rejected(data):
    with pytest.raises(ValueError):
        parse_measurements(data)


def test_valid_mapping_is_parsed(measurements):
    assert (
        parse_measurements(
            {
                "duration_seconds": 10,
                "packet_count": 100,
                "byte_count": 50000,
            }
        )
        == measurements
    )


def test_unvalidated_feature_input_is_rejected():
    with pytest.raises(ValueError):
        extract_features({"packet_count": 100})
