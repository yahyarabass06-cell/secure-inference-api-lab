# Secure Inference API Lab

## Goal

Build a local inference API around a small trained model.
Evaluate prediction behavior separately from API security controls.

Use synthetic data. No cloud deployment is required.

## Model Boundary

- Define the prediction task and feature contract explicitly.
- Separate training, calibration and evaluation data.
- Record data fingerprints, model settings and observed results.
- Export parameters in a strictly validated JSON format.
- Never load pickle or joblib files supplied by a caller.
- Load validated model parameters before serving requests.
- Keep the loaded model immutable during request processing.

Prediction results do not establish that a security incident occurred.

## API Boundary

- Bind demonstrations to the loopback interface.
- Obtain authentication secrets from trusted environment configuration.
- Never commit real credentials or include them in responses.
- Authenticate before running model inference.
- Validate request structure, numeric ranges and finite values.
- Bound request bodies and concurrent inference.
- Return explicit errors without exposing internal exceptions.
- Do not expose shell execution, arbitrary file paths or model uploads.

Authentication establishes possession of a configured secret.
It is not a complete user identity or authorization system.

## Evaluation

Test authorized and unauthorized requests, malformed inputs,
oversized requests and resource-limit behavior.

Verify that rejected requests do not invoke inference.
Measure model loading and request latency with their scope documented.

Use application tests and real local HTTP checks.
Record limitations of single-process resource controls.

## Deliverables

- Reproducible synthetic dataset and training procedure.
- Validated model parameter artifact.
- Local inference API.
- Unit and integration tests.
- Documented security boundaries and evaluation.
- Reproducible installation and GitHub Actions checks.

## Out of Scope

- Public production hosting.
- Automatic incident response.
- Arbitrary model execution.
- Production identity management.
- Distributed rate limiting.
- Claims of complete denial-of-service protection.
