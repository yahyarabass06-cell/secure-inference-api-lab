# Security Considerations

## Intended Use

Run this educational laboratory locally with synthetic inputs.
The provided entry point binds only to 127.0.0.1.
Do not expose it publicly without a separate deployment security design.

## Authentication

Configure a fresh 32-byte random secret encoded as 64 hexadecimal characters.
The application checks possession of a shared secret, not individual identity.

Do not commit secrets, print them in reports or pass them as command-line
arguments. Environment variables remain observable to sufficiently privileged
local processes. Access logging is disabled by the provided server entry point.

Authentication occurs before request body consumption and inference.

## Model Trust

Supply the expected model fingerprint through trusted configuration.
Changing both the model and the expected fingerprint defeats the integrity
comparison. Review both as security-sensitive inputs.

The application reads bounded bytes, compares their fingerprint and parses
those same bytes. Serving uses the resulting immutable in-memory parameters.

No pickle or joblib artifact is loaded.
Valid parameters and a matching fingerprint do not prove model safety,
publisher identity, accuracy or absence of backdoors.

## Resource Controls

Bodies are limited to 1024 bytes, including streamed bodies without a
Content-Length header. Body reception has a two-second application deadline.

Authenticated requests share a rolling 60-per-minute budget in one process.
Invalid authenticated requests consume this budget. Unauthorized requests
do not consume it. A holder of the shared token can exhaust this budget.

The provided server uses one worker and bounded concurrency.
The application does not provide distributed rate limiting or comprehensive
protection against slow headers, connection floods or OS resource exhaustion.

Inference is a small synchronous calculation.
Replacing it with expensive computation requires a different scheduling design.

## Responses and Data

Prediction responses contain no credentials or raw internal exceptions.
Structural validation does not establish that caller-supplied measurements
are truthful. A flagged prediction does not prove compromise.

## Reporting

Report reproducible issues privately to the repository maintainer.
Include synthetic reproduction steps and affected versions.
Do not include real tokens, sensitive network data or private configurations.
