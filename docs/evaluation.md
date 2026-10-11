# Evaluation

## Protocol

Synthetic scenario definitions were established before model evaluation.
Training, calibration and test periods are chronological and have separate
record identifiers and session groups.

StandardScaler and LogisticRegression were fitted on 800 training records.
The decision threshold was selected using 300 benign calibration records.

The frozen model was then evaluated on 400 test records.
Neither fitting nor threshold selection used the test period.

## Model Evidence

- Scikit-learn version: 1.9.1.
- Logistic regression: C=1.0, solver=lbfgs, max_iter=1000.
- Threshold: 0.4296634987032077.
- Benign calibration false-positive rate: 0.05.
- Maximum exported probability difference: 3.3306690738754696e-16.

The export comparison used training and calibration records.
It verifies numerical implementation agreement, not model correctness.

## Observed Test Results

| Measurement | Observed value |
| --- | ---: |
| True negatives | 289 |
| False positives | 11 |
| False negatives | 13 |
| True positives | 87 |
| Precision | 0.8877551020408163 |
| Recall | 0.87 |
| F1 | 0.8787878787878788 |
| False-positive rate | 0.03666666666666667 |

All 11 false positives occurred in the unusual benign scenario.
Ordinary and legitimate bulk scenarios each had 100 true negatives.
The suspicious scenario had 87 true positives and 13 false negatives.

These results describe synthetic simulation labels.
They do not estimate real-world attack detection accuracy.
Probability calibration has not been independently measured.

## Application Evidence

The latest observed local suite passed 93 tests.

Tests cover measurement and model validation, feature contracts,
chronological separation, JSON export agreement, authentication,
request budgets, streamed body bounds and body deadlines.

Rejected requests are tested with a counting model to verify that inference
was not invoked. Model loading tests verify fingerprint rejection and
independence from later changes to the model file.

Real loopback HTTP checks confirmed:
- Valid authenticated inference: 200.
- Unauthorized request: 401.
- Oversized body: 413.
- Unsupported content type: 415.
- Invalid measurement: 422.
- Exhausted authenticated request budget: 429.
- Disabled documentation endpoint: 404.

Uvicorn concurrency configuration is tested.
Overload behavior has not been measured through a load experiment.

## Reproducibility and Limits

Data and model fingerprints are recorded in the generated metadata.
The dependency snapshot pins versions without artifact hashes.

Independent installation and remote CI still require verification.
No production throughput, latency percentile or memory ceiling is claimed.
