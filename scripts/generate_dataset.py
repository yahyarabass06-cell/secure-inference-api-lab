"""Generate independently labeled, chronological synthetic flow periods."""

import argparse
import hashlib
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

from inference_security.measurements import Measurements

SEED = 20261011
SCENARIOS = (
    "ordinary",
    "legitimate_bulk",
    "unusual_benign",
    "suspicious_high_rate",
)


def generate_partition(name, count, day, rng):
    rows = []
    start = datetime(2026, 2, day, tzinfo=timezone.utc)

    for index in range(count):
        scenario = SCENARIOS[index % len(SCENARIOS)]
        duration = rng.uniform(1, 120)

        if scenario == "ordinary":
            rate = rng.uniform(5, 100)
            bytes_per_packet = rng.uniform(100, 900)
        elif scenario == "legitimate_bulk":
            rate = rng.uniform(80, 500)
            bytes_per_packet = rng.uniform(900, 1450)
        elif scenario == "unusual_benign":
            rate = rng.uniform(200, 1000)
            bytes_per_packet = rng.uniform(60, 1400)
        else:
            rate = rng.uniform(450, 1800)
            bytes_per_packet = rng.uniform(40, 500)

        packets = max(1, round(duration * rate))
        measurements = Measurements(
            duration_seconds=round(duration, 6),
            packet_count=packets,
            byte_count=round(packets * bytes_per_packet),
        )
        rows.append(
            {
                "record_id": f"{name}-{index:04d}",
                "group_id": f"{name}-session-{index // 4:04d}",
                "timestamp": (start + timedelta(seconds=index)).isoformat(),
                "scenario": scenario,
                "label": int(scenario == "suspicious_high_rate"),
                "measurements": {
                    "duration_seconds": measurements.duration_seconds,
                    "packet_count": measurements.packet_count,
                    "byte_count": measurements.byte_count,
                },
            }
        )
    return rows


def generate(destination):
    if destination.exists():
        raise ValueError("Destination already exists; nothing overwritten.")

    rng = random.Random(SEED)
    prepared = {}
    metadata = {
        "schema_version": 1,
        "data_kind": "synthetic_simulation",
        "seed": SEED,
        "positive_label": "suspicious_high_rate simulation",
        "partitions": {},
    }

    for name, count, day in (
        ("train", 800, 1),
        ("calibration", 400, 2),
        ("test", 400, 3),
    ):
        rows = generate_partition(name, count, day, rng)
        raw = "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows
        ).encode("utf-8")
        prepared[name] = raw
        metadata["partitions"][name] = {
            "records": count,
            "positive_records": sum(row["label"] for row in rows),
            "first_timestamp": rows[0]["timestamp"],
            "last_timestamp": rows[-1]["timestamp"],
            "sha256": hashlib.sha256(raw).hexdigest(),
            "scenario_counts": {
                scenario: sum(row["scenario"] == scenario for row in rows)
                for scenario in SCENARIOS
            },
        }

    destination.mkdir(parents=True)
    for name, raw in prepared.items():
        with (destination / f"{name}.jsonl").open("xb") as output:
            output.write(raw)

    with (destination / "metadata.json").open("x", encoding="utf-8") as output:
        json.dump(metadata, output, indent=2, allow_nan=False)
        output.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic/v1"))
    arguments = parser.parse_args()
    generate(arguments.output)
    print(f"Synthetic dataset written to {arguments.output}")


if __name__ == "__main__":
    main()
