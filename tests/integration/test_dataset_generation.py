import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from inference_security.measurements import parse_measurements

ROOT = Path(__file__).resolve().parents[2]


def generate(destination):
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/generate_dataset.py"),
            "--output",
            str(destination),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_generation_is_reproducible_and_partitions_are_separate(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    assert generate(first).returncode == 0
    assert generate(second).returncode == 0

    metadata = json.loads((first / "metadata.json").read_text())
    all_ids = set()
    all_groups = set()
    previous_end = None

    for name, count in (("train", 800), ("calibration", 400), ("test", 400)):
        raw = (first / f"{name}.jsonl").read_bytes()
        assert raw == (second / f"{name}.jsonl").read_bytes()
        details = metadata["partitions"][name]
        assert hashlib.sha256(raw).hexdigest() == details["sha256"]

        rows = [json.loads(line) for line in raw.splitlines()]
        assert len(rows) == count
        assert sum(row["label"] for row in rows) == count // 4

        ids = {row["record_id"] for row in rows}
        groups = {row["group_id"] for row in rows}
        assert len(ids) == count
        assert not ids & all_ids
        assert not groups & all_groups
        all_ids.update(ids)
        all_groups.update(groups)

        times = [datetime.fromisoformat(row["timestamp"]) for row in rows]
        assert all(time.utcoffset().total_seconds() == 0 for time in times)
        assert all(left < right for left, right in zip(times, times[1:]))
        if previous_end is not None:
            assert previous_end < times[0]
        previous_end = times[-1]

        for row in rows:
            parse_measurements(row["measurements"])
            assert row["label"] == int(row["scenario"] == "suspicious_high_rate")

    assert (first / "metadata.json").read_bytes() == (
        second / "metadata.json"
    ).read_bytes()


def test_existing_dataset_is_preserved(tmp_path):
    destination = tmp_path / "dataset"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("original")
    assert generate(destination).returncode != 0
    assert sentinel.read_text() == "original"
    assert list(destination.iterdir()) == [sentinel]
