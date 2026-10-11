import json
from pathlib import Path

import pytest

from inference_security.model import model_fields, parse_model
from inference_security.training import (
    check_separation,
    load_period,
    train_model,
)

DATA = Path(__file__).resolve().parents[2] / "data/synthetic/v1"


def test_training_export_matches_pipeline_and_uses_no_test_period():
    model, report = train_model(DATA / "train.jsonl", DATA / "calibration.jsonl")
    exported = parse_model(json.dumps(model_fields(model)).encode())
    assert exported == model
    assert report["test_period_used"] is False
    assert report["maximum_export_probability_error"] <= 1e-12
    assert report["observed_calibration_false_positive_rate"] <= 0.05
    assert report["training_records"] == 800
    assert report["calibration_records"] == 400


def test_same_period_cannot_be_training_and_calibration():
    with pytest.raises(ValueError, match="overlap"):
        train_model(DATA / "train.jsonl", DATA / "train.jsonl")


def test_groups_cannot_cross_periods():
    training, _ = load_period(DATA / "train.jsonl")
    calibration, _ = load_period(DATA / "calibration.jsonl")
    calibration[0]["group_id"] = training[0]["group_id"]
    with pytest.raises(ValueError, match="group_id"):
        check_separation(training, calibration)
