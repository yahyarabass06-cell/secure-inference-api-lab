"""Start a loopback-only server with a verified in-memory model."""

import argparse
import hashlib
import hmac
import os
import re
import sys
from pathlib import Path

import uvicorn

from inference_security.api import ApiSettings, create_app
from inference_security.model import parse_model


def load_verified_model(path, expected_sha256):
    if type(expected_sha256) is not str or not re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256
    ):
        raise ValueError("Expected model fingerprint is missing or invalid.")

    with path.open("rb") as source:
        raw = source.read(16385)

    actual = hashlib.sha256(raw).hexdigest()
    if not hmac.compare_digest(actual, expected_sha256):
        raise ValueError("Model fingerprint mismatch.")
    return parse_model(raw)


def prepare_app(model_path, environment):
    settings = ApiSettings(token=environment.get("INFERENCE_API_TOKEN", ""))
    model = load_verified_model(
        model_path, environment.get("INFERENCE_MODEL_SHA256", "")
    )
    return create_app(model=model, settings=settings)


def port_number(value):
    try:
        port = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Port must be an integer.") from error
    if not 1024 <= port <= 65535:
        raise argparse.ArgumentTypeError("Port must be between 1024 and 65535.")
    return port


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--port", type=port_number, default=8000)
    arguments = parser.parse_args(argv)

    try:
        app = prepare_app(arguments.model, os.environ)
    except (OSError, ValueError):
        print(
            "Startup failed: check model and trusted environment configuration.",
            file=sys.stderr,
        )
        return 2

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=arguments.port,
        workers=1,
        loop="asyncio",
        http="h11",
        ws="none",
        limit_concurrency=16,
        backlog=32,
        timeout_keep_alive=2,
        timeout_graceful_shutdown=5,
        proxy_headers=False,
        server_header=False,
        access_log=False,
        log_level="warning",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
