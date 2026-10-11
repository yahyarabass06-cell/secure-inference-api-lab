import asyncio

import httpx
import pytest

from inference_security.api import ApiSettings, create_app
from inference_security.model import LogisticModel

TOKEN = "a" * 64
VALID = {
    "duration_seconds": 10,
    "packet_count": 100,
    "byte_count": 50000,
}


class CountingModel:
    def __init__(self):
        self.calls = 0
        self.model = LogisticModel(
            mean=(0, 0, 0),
            scale=(1, 1, 1),
            coefficients=(0, 0, 0),
            intercept=0,
            threshold=0.5,
        )

    def predict(self, measurements):
        self.calls += 1
        return self.model.predict(measurements)


def run_request(app, **kwargs):
    async def execute():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://lab",
        ) as client:
            return await client.post("/predict", **kwargs)

    return asyncio.run(execute())


def setup_app(**settings):
    model = CountingModel()
    app = create_app(model=model, settings=ApiSettings(token=TOKEN, **settings))
    return app, model


def test_authorized_prediction():
    app, model = setup_app()
    response = run_request(
        app, json=VALID, headers={"Authorization": f"Bearer {TOKEN}"}
    )
    assert response.status_code == 200
    assert response.json()["probability"] == 0.5
    assert response.headers["cache-control"] == "no-store"
    assert model.calls == 1


@pytest.mark.parametrize("authorization", [None, "Bearer wrong", "Basic unused"])
def test_unauthorized_requests_never_invoke_model(authorization):
    app, model = setup_app()
    headers = {} if authorization is None else {"Authorization": authorization}
    response = run_request(app, content=b"x" * 5000, headers=headers)
    assert response.status_code == 401
    assert model.calls == 0


@pytest.mark.parametrize(
    "content",
    [
        b"{",
        b'{"packet_count": 1, "packet_count": 2}',
        b'{"duration_seconds": NaN}',
        b"[]",
        b"\xff",
        b'{"duration_seconds": 10, "packet_count": true, "byte_count": 10}',
        b'{"duration_seconds": 10, "packet_count": 1, "byte_count": 10, "label": 1}',
    ],
)
def test_invalid_inputs_never_invoke_model(content):
    app, model = setup_app()
    response = run_request(
        app,
        content=content,
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 422
    assert model.calls == 0


def test_oversized_body_never_invokes_model():
    app, model = setup_app()
    response = run_request(
        app,
        content=b"x" * 1025,
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 413
    assert model.calls == 0


def test_authenticated_request_budget():
    app, model = setup_app(requests_per_minute=1)
    arguments = {"json": VALID, "headers": {"Authorization": f"Bearer {TOKEN}"}}
    assert run_request(app, **arguments).status_code == 200
    assert run_request(app, **arguments).status_code == 429
    assert model.calls == 1


def test_model_failure_does_not_expose_exception():
    class FailingModel:
        def predict(self, measurements):
            raise RuntimeError("private internal detail")

    app = create_app(model=FailingModel(), settings=ApiSettings(token=TOKEN))
    response = run_request(
        app, json=VALID, headers={"Authorization": f"Bearer {TOKEN}"}
    )
    assert response.status_code == 500
    assert response.json() == {"error": "inference_failed"}
    assert "private internal detail" not in response.text
