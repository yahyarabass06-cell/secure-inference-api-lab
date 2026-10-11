"""Local inference API with explicit, single-process resource limits."""

import asyncio
import hmac
import json
import re
import threading
import time
from collections import deque
from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from inference_security.measurements import parse_measurements
from inference_security.model import reject_constant, unique_object


@dataclass(frozen=True)
class ApiSettings:
    token: str
    max_body_bytes: int = 1024
    requests_per_minute: int = 60
    body_timeout_seconds: float = 2.0

    def __post_init__(self):
        if type(self.token) is not str or not re.fullmatch(r"[0-9a-f]{64}", self.token):
            raise ValueError("Token must be 64 lowercase hexadecimal characters.")
        for name, maximum in (
            ("max_body_bytes", 4096),
            ("requests_per_minute", 1000),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"Invalid setting: {name}")
        if (
            type(self.body_timeout_seconds) not in (int, float)
            or not 0 < self.body_timeout_seconds <= 10
        ):
            raise ValueError("Invalid body timeout.")


class RequestBudget:
    def __init__(self, limit):
        self.limit = limit
        self.times = deque()
        self.lock = threading.Lock()

    def admit(self):
        now = time.monotonic()
        with self.lock:
            while self.times and self.times[0] <= now - 60:
                self.times.popleft()
            if len(self.times) >= self.limit:
                return False
            self.times.append(now)
            return True


def error_response(status, reason):
    return JSONResponse(
        {"error": reason},
        status_code=status,
        headers={"Cache-Control": "no-store"},
    )


def create_app(*, model, settings):
    if not isinstance(settings, ApiSettings):
        raise ValueError("Validated API settings required.")
    if not callable(getattr(model, "predict", None)):
        raise ValueError("Inference model required.")

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    budget = RequestBudget(settings.requests_per_minute)

    @app.get("/health")
    async def health():
        return {"status": "ready"}

    @app.post("/predict")
    async def predict(request: Request):
        authorization = request.headers.getlist("authorization")
        expected = f"Bearer {settings.token}"
        if (
            len(authorization) != 1
            or len(authorization[0]) != len(expected)
            or not hmac.compare_digest(
                authorization[0].encode("utf-8"), expected.encode("ascii")
            )
        ):
            return error_response(401, "unauthorized")

        if not budget.admit():
            return error_response(429, "request_budget_exceeded")

        content_types = request.headers.getlist("content-type")
        if (
            len(content_types) != 1
            or content_types[0].split(";", 1)[0].strip().lower() != "application/json"
        ):
            return error_response(415, "application_json_required")

        lengths = request.headers.getlist("content-length")
        if lengths:
            if (
                len(lengths) != 1
                or not lengths[0].isascii()
                or not lengths[0].isdigit()
            ):
                return error_response(400, "invalid_content_length")
            if len(lengths[0]) > 10 or int(lengths[0]) > settings.max_body_bytes:
                return error_response(413, "body_too_large")

        async def read_body():
            body = bytearray()
            async for chunk in request.stream():
                if len(body) + len(chunk) > settings.max_body_bytes:
                    raise ValueError("body_too_large")
                body.extend(chunk)
            return bytes(body)

        try:
            raw = await asyncio.wait_for(
                read_body(), timeout=settings.body_timeout_seconds
            )
        except TimeoutError:
            return error_response(408, "body_timeout")
        except ValueError:
            return error_response(413, "body_too_large")
        except Exception:
            return error_response(400, "body_read_failed")

        try:
            data = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=unique_object,
                parse_constant=reject_constant,
            )
            measurements = parse_measurements(data)
        except (ValueError, UnicodeError, RecursionError):
            return error_response(422, "invalid_measurements")

        try:
            # This fixed three-feature calculation runs synchronously.
            # One event loop therefore executes one inference at a time.
            result = model.predict(measurements)
            return JSONResponse(result, headers={"Cache-Control": "no-store"})
        except Exception:
            return error_response(500, "inference_failed")

    return app
