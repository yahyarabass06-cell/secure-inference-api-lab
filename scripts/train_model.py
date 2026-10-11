"""Export a frozen model without examining the test period."""

import argparse
import hashlib
import json
from pathlib import Path

from inference_security.model import model_fields
from inference_security.training import train_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/synthetic/v1"))
    parser.add_argument("--output", type=Path, default=Path("data/model/v1"))
    arguments = parser.parse_args()

    if arguments.output.exists():
        raise ValueError("Output directory exists; nothing overwritten.")

    model, report = train_model(
        arguments.data / "train.jsonl",
        arguments.data / "calibration.jsonl",
    )
    raw = (json.dumps(model_fields(model), indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    report["model_sha256"] = hashlib.sha256(raw).hexdigest()

    arguments.output.mkdir(parents=True)
    with (arguments.output / "model.json").open("xb") as output:
        output.write(raw)
    with (arguments.output / "training.json").open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
