import hashlib
import json
from pathlib import Path

import pytest

from inference_security import server
from inference_security.measurements import Measurements

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "data/model/v1/model.json"


def fingerprint():
    return hashlib.sha256(MODEL.read_bytes()).hexdigest()


def test_verified_model_is_independent_of_later_file_changes(tmp_path):
    path = tmp_path / "model.json"
    path.write_bytes(MODEL.read_bytes())
    model = server.load_verified_model(path, fingerprint())
    measurements = Measurements(10, 100, 50000)
    before = model.predict(measurements)

    path.write_bytes(b"changed after loading")
    assert model.predict(measurements) == before


@pytest.mark.parametrize("expected", ["", "invalid", "0" * 64])
def test_invalid_fingerprint_rejects_model(expected):
    with pytest.raises(ValueError):
        server.load_verified_model(MODEL, expected)


def test_matching_hash_does_not_bypass_model_validation(tmp_path):
    path = tmp_path / "model.json"
    raw = json.dumps({"unsupported": True}).encode()
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        server.load_verified_model(path, hashlib.sha256(raw).hexdigest())


def test_missing_secret_prevents_server_start(monkeypatch, capsys):
    monkeypatch.delenv("INFERENCE_API_TOKEN", raising=False)

    def unexpected_run(*args, **kwargs):
        raise AssertionError("Server must not start.")

    monkeypatch.setattr(server.uvicorn, "run", unexpected_run)
    assert server.main(["--model", str(MODEL)]) == 2
    assert "Startup failed" in capsys.readouterr().err


def test_server_configuration_is_loopback_and_bounded(monkeypatch):
    monkeypatch.setenv("INFERENCE_API_TOKEN", "a" * 64)
    monkeypatch.setenv("INFERENCE_MODEL_SHA256", fingerprint())
    captured = {}

    def record_run(app, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(server.uvicorn, "run", record_run)
    assert server.main(["--model", str(MODEL)]) == 0
    assert captured["host"] == "127.0.0.1"
    assert captured["workers"] == 1
    assert captured["limit_concurrency"] == 16
    assert captured["proxy_headers"] is False
    assert captured["access_log"] is False
