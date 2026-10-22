import json

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from gateway.app import main as gw
from gateway.app.backends import TritonBackend, VLLMBackend


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("MODEL_REPOSITORY", "model_repository")
    monkeypatch.delenv("TRITON_URL", raising=False)
    monkeypatch.delenv("VLLM_URL", raising=False)
    with TestClient(gw.app) as c:
        yield c


def _rows(model, n=4):
    return np.load(f"model_repository/{model}/testset.npz")["X"][:n].tolist()


def test_predict_local_backend(client):
    r = client.post("/v1/models/fraud_mlp/predict", json={"instances": _rows("fraud_mlp")})
    assert r.status_code == 200
    body = r.json()
    assert len(body["labels"]) == 4 and len(body["probabilities"][0]) == 2
    assert r.headers["x-model-version"] == "2"


def test_version_pinning_and_agreement(client):
    rows = _rows("digits_mlp", 50)
    v1 = client.post("/v1/models/digits_mlp/predict?version=1", json={"instances": rows}).json()
    v2 = client.post("/v1/models/digits_mlp/predict?version=2", json={"instances": rows}).json()
    assert v1["version"] == "1" and v2["version"] == "2"
    assert np.mean(np.array(v1["labels"]) == np.array(v2["labels"])) > 0.98


def test_traffic_split(client):
    gw.ctx["settings"] = gw.Settings(traffic={"risk_mlp": {"1": 0.5, "2": 0.5}})
    seen = {
        client.post("/v1/models/risk_mlp/predict", json={"instances": _rows("risk_mlp", 1)}).json()["version"]
        for _ in range(60)
    }
    assert seen == {"1", "2"}


def test_errors(client):
    assert client.post("/v1/models/nope/predict", json={"instances": [[1.0]]}).status_code == 404
    assert (
        client.post("/v1/models/fraud_mlp/predict?version=7", json={"instances": _rows("fraud_mlp")}).status_code == 404
    )
    assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 503


def test_load_shedding(client):
    import asyncio

    gw.ctx["slots"] = asyncio.Semaphore(0)  # simulate saturation
    r = client.post("/v1/models/fraud_mlp/predict", json={"instances": _rows("fraud_mlp")})
    assert r.status_code == 429 and r.headers["retry-after"] == "1"
    assert client.get("/healthz").status_code == 200  # health is never shed


def test_readyz_and_metrics(client):
    assert client.get("/readyz").json()["ready"] is True
    client.post("/v1/models/fraud_mlp/predict", json={"instances": _rows("fraud_mlp")})
    assert 'gateway_requests_total{model="fraud_mlp"' in client.get("/metrics").text


@pytest.mark.anyio
async def test_triton_backend_speaks_kserve_v2():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/v2/models/fraud_mlp/versions/2/infer"
        payload = json.loads(req.content)
        assert payload["inputs"][0]["datatype"] == "FP32" and payload["inputs"][0]["shape"] == [2, 3]
        return httpx.Response(
            200,
            json={
                "outputs": [
                    {"name": "label", "shape": [2], "datatype": "INT64", "data": [0, 1]},
                    {"name": "probabilities", "shape": [2, 2], "datatype": "FP32", "data": [0.9, 0.1, 0.2, 0.8]},
                ]
            },
        )

    be = TritonBackend("http://triton:8000", "model_repository", 5)
    be._client = httpx.AsyncClient(base_url="http://triton:8000", transport=httpx.MockTransport(handler))
    out = await be.predict("fraud_mlp", "2", np.zeros((2, 3), dtype=np.float32))
    assert out["labels"] == [0, 1]
    assert out["probabilities"][1] == pytest.approx([0.2, 0.8])


@pytest.mark.anyio
async def test_vllm_backend_streams():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b'data: {"choices":[{"text":"hi"}]}\n\ndata: [DONE]\n\n')

    be = VLLMBackend("http://vllm:8000", 5)
    be._client = httpx.AsyncClient(base_url="http://vllm:8000", transport=httpx.MockTransport(handler))
    chunks = [c async for c in be.stream("/v1/completions", {"prompt": "x", "stream": True})]
    assert b"[DONE]" in b"".join(chunks)
