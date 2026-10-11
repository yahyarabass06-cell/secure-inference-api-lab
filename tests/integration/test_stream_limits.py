import asyncio
import json

import httpx

from inference_security.api import ApiSettings, create_app

TOKEN = "b" * 64


class CountingModel:
    def __init__(self):
        self.calls = 0

    def predict(self, measurements):
        self.calls += 1
        return {"probability": 0.5, "flagged": False, "threshold": 0.5}


def request_with_stream(chunks, *, timeout=1, headers=None):
    model = CountingModel()
    app = create_app(
        model=model,
        settings=ApiSettings(
            token=TOKEN,
            body_timeout_seconds=timeout,
        ),
    )

    async def execute():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://lab",
        ) as client:
            return await client.post(
                "/predict",
                content=chunks(),
                headers=headers
                or {
                    "Authorization": f"Bearer {TOKEN}",
                    "Content-Type": "application/json",
                },
            )

    return asyncio.run(execute()), model


def test_chunked_oversized_body_is_rejected_without_content_length():
    async def chunks():
        yield b"x" * 600
        yield b"x" * 600

    response, model = request_with_stream(chunks)
    assert "content-length" not in response.request.headers
    assert response.status_code == 413
    assert model.calls == 0


def test_valid_chunked_request_is_supported():
    raw = json.dumps(
        {
            "duration_seconds": 10,
            "packet_count": 100,
            "byte_count": 50000,
        }
    ).encode()

    async def chunks():
        yield raw[:20]
        yield raw[20:]

    response, model = request_with_stream(chunks)
    assert response.status_code == 200
    assert model.calls == 1


def test_slow_body_times_out_before_inference():
    async def chunks():
        yield b"{"
        await asyncio.sleep(0.2)
        yield b"}"

    response, model = request_with_stream(chunks, timeout=0.02)
    assert response.status_code == 408
    assert model.calls == 0


def test_unauthorized_body_is_not_consumed():
    consumed = []

    async def chunks():
        consumed.append(True)
        yield b"x" * 2000

    response, model = request_with_stream(
        chunks,
        headers={
            "Authorization": "Bearer wrong",
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 401
    assert consumed == []
    assert model.calls == 0


def test_duplicate_authorization_headers_are_rejected():
    async def chunks():
        yield b"{}"

    response, model = request_with_stream(
        chunks,
        headers=[
            ("Authorization", f"Bearer {TOKEN}"),
            ("Authorization", f"Bearer {TOKEN}"),
            ("Content-Type", "application/json"),
        ],
    )
    assert response.status_code == 401
    assert model.calls == 0
