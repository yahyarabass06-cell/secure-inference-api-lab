"""Validated measurements and a fixed predictive feature contract."""

import math
from dataclasses import dataclass

FEATURE_NAMES = (
    "log_duration_seconds",
    "log_packet_rate",
    "log_bytes_per_packet",
)


@dataclass(frozen=True)
class Measurements:
    duration_seconds: float
    packet_count: int
    byte_count: int

    def __post_init__(self):
        duration = self.duration_seconds
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or not 0.001 <= duration <= 86400
        ):
            raise ValueError("duration_seconds must be finite and within bounds.")

        for name, minimum, maximum in (
            ("packet_count", 1, 1_000_000_000),
            ("byte_count", 0, 1_000_000_000_000),
        ):
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer within bounds.")

        object.__setattr__(self, "duration_seconds", float(duration))


def parse_measurements(data):
    expected = {"duration_seconds", "packet_count", "byte_count"}
    if not isinstance(data, dict) or set(data) != expected:
        raise ValueError("Exactly the three measurement fields are required.")
    return Measurements(**data)


def extract_features(measurements):
    if not isinstance(measurements, Measurements):
        raise ValueError("Validated measurements are required.")

    return (
        math.log1p(measurements.duration_seconds),
        math.log1p(measurements.packet_count / measurements.duration_seconds),
        math.log1p(measurements.byte_count / measurements.packet_count),
    )
