"""Model backends behind the gateway.

TritonBackend   KServe v2 HTTP protocol against Triton Inference Server
LocalBackend    the same ONNX files run in-process with ONNX Runtime (dev/CI)
VLLMBackend     proxy to vLLM's OpenAI-compatible server, streaming included
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Protocol

import httpx
import numpy as np


class BackendError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


class ModelBackend(Protocol):
    async def predict(self, model: str, version: str, rows: np.ndarray) -> dict[str, Any]: ...

    async def ready(self) -> bool: ...

    def models(self) -> dict[str, list[str]]: ...


def _scan_repository(repo: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if not repo.exists():
        return out
    for model_dir in sorted(p for p in repo.iterdir() if p.is_dir()):
        versions = sorted(v.name for v in model_dir.iterdir() if (v / "model.onnx").exists())
        if versions:
            out[model_dir.name] = versions
    return out


class TritonBackend:
    def __init__(self, url: str, repo: str, timeout: float):
        self._client = httpx.AsyncClient(
            base_url=url.rstrip("/"),
            timeout=timeout,
            limits=httpx.Limits(max_connections=256, max_keepalive_connections=128),
        )
        self._models = _scan_repository(Path(repo))

    async def predict(self, model: str, version: str, rows: np.ndarray) -> dict[str, Any]:
        payload = {
            "inputs": [{"name": "input", "shape": list(rows.shape), "datatype": "FP32", "data": rows.ravel().tolist()}],
            "outputs": [{"name": "label"}, {"name": "probabilities"}],
        }
        r = await self._client.post(f"/v2/models/{model}/versions/{version}/infer", json=payload)
        if r.status_code != 200:
            raise BackendError(502 if r.status_code >= 500 else r.status_code, r.text)
        outs = {o["name"]: o for o in r.json()["outputs"]}
        probs = np.asarray(outs["probabilities"]["data"], dtype=np.float32).reshape(outs["probabilities"]["shape"])
        return {"labels": [int(x) for x in outs["label"]["data"]], "probabilities": probs.round(6).tolist()}

    async def ready(self) -> bool:
        try:
            return (await self._client.get("/v2/health/ready")).status_code == 200
        except httpx.HTTPError:
            return False

    def models(self) -> dict[str, list[str]]:
        return self._models

    async def aclose(self) -> None:
        await self._client.aclose()


class LocalBackend:
    """Runs ONNX models in a thread pool so the event loop never blocks on inference."""

    def __init__(self, repo: str, threads: int = 1):
        import onnxruntime as ort

        self._sessions: dict[tuple[str, str], Any] = {}
        self._models = _scan_repository(Path(repo))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        for model, versions in self._models.items():
            for v in versions:
                self._sessions[(model, v)] = ort.InferenceSession(
                    str(Path(repo) / model / v / "model.onnx"), opts, providers=["CPUExecutionProvider"]
                )

    async def predict(self, model: str, version: str, rows: np.ndarray) -> dict[str, Any]:
        sess = self._sessions.get((model, version))
        if sess is None:
            raise BackendError(404, f"model {model} version {version} not found")
        label, probs = await asyncio.to_thread(sess.run, None, {"input": rows})
        return {"labels": label.astype(int).tolist(), "probabilities": probs.round(6).tolist()}

    async def ready(self) -> bool:
        return bool(self._sessions)

    def models(self) -> dict[str, list[str]]:
        return self._models

    async def aclose(self) -> None:
        return None


class VLLMBackend:
    def __init__(self, url: str, timeout: float):
        self._client = httpx.AsyncClient(
            base_url=url.rstrip("/"),
            timeout=httpx.Timeout(timeout, read=None),  # streams can run long
            limits=httpx.Limits(max_connections=512, max_keepalive_connections=256),
        )

    async def forward(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        r = await self._client.post(path, json=body)
        if r.status_code != 200:
            raise BackendError(r.status_code, r.text)
        return r.json()

    async def stream(self, path: str, body: dict[str, Any]) -> AsyncIterator[bytes]:
        async with self._client.stream("POST", path, json=body) as r:
            if r.status_code != 200:
                raise BackendError(r.status_code, (await r.aread()).decode())
            async for chunk in r.aiter_bytes():
                yield chunk

    async def ready(self) -> bool:
        try:
            return (await self._client.get("/health")).status_code == 200
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()
