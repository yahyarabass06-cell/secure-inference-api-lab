# Secure Inference API Lab

A local inference API with authenticated requests, bounded inputs,
a verified JSON model and reproducible synthetic evaluation.

This educational project uses a trained logistic regression model.
It does not establish real-world network attack detection accuracy.

## Features

- Train on a separate chronological training period.
- Select the decision threshold using benign calibration records only.
- Export validated model parameters as JSON.
- Verify the expected SHA-256 before loading model parameters.
- Retain the validated model in memory during serving.
- Authenticate prediction requests before reading their bodies.
- Reject invalid, oversized and slow request bodies.
- Apply a shared per-process authenticated request budget.
- Bind the server to the loopback interface.

## Installation

Use Python 3.12 or newer on Linux. Run from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --only-binary=:all: -r requirements-dev.lock
python -m pip install --no-build-isolation --no-deps .
python -m pip check
```

The dependency snapshot pins versions but does not contain artifact hashes.
Development tools include Scikit-learn; serving uses exported JSON parameters.

## Start the Local Server

Generate a temporary secret without displaying it:

```bash
export INFERENCE_API_TOKEN="$(python -c 'import secrets; print(secrets.token_hex(32))')"
export INFERENCE_MODEL_SHA256="b3b6b4e95afde0e8415cfb5c0caa0e3f918eaa5b52b7f6828ec6bdc28d2abfc4"

secure-inference-api --model data/model/v1/model.json
```

The expected fingerprint is a trusted configuration input.
Review it when approving a new model; do not automatically derive it
from an untrusted replacement file.

The server runs on http://127.0.0.1:8000.
Use Ctrl+C to stop it.

In a second terminal, activate the environment. To make an authenticated
request without placing the secret in command-line arguments, enter the
same token through a hidden prompt and use Python:

```bash
python - <<'PY'
import getpass
import json

import httpx

token = getpass.getpass("API token: ")
with httpx.Client(trust_env=False, timeout=5) as client:
    response = client.post(
        "http://127.0.0.1:8000/predict",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "duration_seconds": 10,
            "packet_count": 100,
            "byte_count": 50000,
        },
    )
    print("HTTP status:", response.status_code)
    print(json.dumps(response.json(), indent=2))
PY
```

For a convenient lab demonstration, send requests from a script that
starts the server with its own generated token rather than displaying
or copying the token.

## API Behavior

| Endpoint or condition | Behavior |
| --- | --- |
| GET /health | Readiness response; no authentication |
| POST /predict with valid authentication and measurements | Probability and threshold decision |
| Missing or invalid authentication | 401 |
| Wrong content type | 415 |
| Invalid measurements | 422 |
| Body exceeding 1024 bytes | 413 |
| Body reception exceeding two seconds | 408 |
| Authenticated request budget exhausted | 429 |
| Uvicorn concurrency limit reached | 503 |
| Internal inference failure | Generic 500 response |

Exactly three measurement fields are accepted.
The maximum authenticated request budget is 60 requests per rolling minute.
Authenticated malformed requests also consume this budget.

The server uses one worker, a concurrency limit of 16 and a backlog of 32.
The three-feature inference calculation runs synchronously on one event loop.
These controls are not a complete denial-of-service defense.

## Model and Data

Features are logarithms of duration, packet rate and bytes per packet.
Labels, scenario names and record identifiers are not predictive inputs.

The bundled synthetic periods contain:

- 800 training records.
- 400 calibration records.
- 400 test records.

Each period contains ordinary traffic, legitimate bulk transfers,
unusual benign traffic and simulated suspicious high-rate activity.

StandardScaler and LogisticRegression are fitted using training data only.
The threshold is selected from benign calibration scores.
A score must be strictly greater than the threshold to be flagged.

The returned probability is a model estimate.
Probability calibration has not been independently evaluated.

## Reproduce the Experiment

Use new output directories; existing destinations are refused.

```bash
python scripts/generate_dataset.py --output /tmp/inference-data-new
python scripts/train_model.py \
  --data /tmp/inference-data-new \
  --output /tmp/inference-model-new

mkdir -p reports
python scripts/evaluate_model.py \
  --data /tmp/inference-data-new \
  --model /tmp/inference-model-new/model.json \
  --output reports/new-assessment.json
```

The training command does not read the test period.
Do not tune model settings against the observed test results.
A repeated evaluation is not new independent evidence.

## Validation

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
```

The latest observed local suite passed 93 tests.
Real loopback HTTP checks confirmed authenticated prediction and responses
401, 413, 415, 422 and 429. The temporary server was then stopped.

A clean Git clone and non-editable installation in a fresh Python 3.14 environment passed all 93 tests. The installed API verified authentication and prediction, and independent evaluation reproduced the observed confusion matrix.

Remote CI remains pending verification.

## Limitations

- Data and labels are synthetic.
- The model detects patterns in this simulation, not proven attacks.
- The shared token is not a complete identity or authorization system.
- Authentication secrets and fingerprints rely on trusted configuration.
- Fingerprints establish byte equality, not publisher identity or model safety.
- Request budgets reset on restart and are not shared between processes.
- Inference is synchronous and intended only for this small fixed model.
- The server has no TLS and is intended for local loopback demonstrations.
- Local privileged processes can observe environment secrets.
- Header reception and all OS resource usage are not bounded by application code.
- Application tests do not establish production-scale resilience.

## Documentation

See [Evaluation](docs/evaluation.md) and [Security Considerations](SECURITY.md).

## License

Distributed under the [MIT License](LICENSE).
