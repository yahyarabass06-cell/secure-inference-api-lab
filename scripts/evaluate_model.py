"""Evaluate an already frozen JSON model without changing its parameters."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from inference_security.evaluation import decision_metrics
from inference_security.model import parse_model
from inference_security.training import check_separation, load_period


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/synthetic/v1"))
    parser.add_argument("--model", type=Path, default=Path("data/model/v1/model.json"))
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    if arguments.output.exists() or arguments.output.is_symlink():
        raise ValueError("Output already exists; nothing overwritten.")
    if not arguments.output.parent.is_dir():
        raise ValueError("Output directory must already exist.")

    with arguments.model.open("rb") as source:
        raw = source.read(16385)
    model = parse_model(raw)

    training, train_hash = load_period(arguments.data / "train.jsonl")
    calibration, calibration_hash = load_period(arguments.data / "calibration.jsonl")
    test, test_hash = load_period(arguments.data / "test.jsonl")
    check_separation(training, calibration)
    check_separation(calibration, test)
    check_separation(training, test)

    started = time.perf_counter()
    decisions = [model.predict(row["measurements"]) for row in test]
    inference_seconds = time.perf_counter() - started

    labels = [row["label"] for row in test]
    predictions = [decision["flagged"] for decision in decisions]
    scenarios = sorted({row["scenario"] for row in test})

    report = {
        "schema_version": 1,
        "analysis_status": "complete",
        "model_sha256": hashlib.sha256(raw).hexdigest(),
        "input_sha256": {
            "train": train_hash,
            "calibration": calibration_hash,
            "test": test_hash,
        },
        "threshold": model.threshold,
        "model_retrained": False,
        "metrics": decision_metrics(labels, predictions),
        "by_scenario": {},
        "timing": {
            "inference_seconds": inference_seconds,
            "scope": "400 sequential predictions; excludes loading and parsing",
        },
        "limitations": [
            "Synthetic scenario labels do not establish real-world accuracy.",
            "Model probability estimates are not independently calibrated.",
            "Repeated executions do not create new independent test evidence.",
        ],
    }

    for scenario in scenarios:
        indexes = [
            index for index, row in enumerate(test) if row["scenario"] == scenario
        ]
        report["by_scenario"][scenario] = decision_metrics(
            [labels[index] for index in indexes],
            [predictions[index] for index in indexes],
        )

    with arguments.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")

    print("Rapport :", arguments.output)
    print(json.dumps(report["metrics"], indent=2))
    for scenario, metrics in report["by_scenario"].items():
        print(scenario, ":", metrics["confusion_matrix"])


if __name__ == "__main__":
    main()
