"""Inference gateway: one front door for LLM and non-LLM models.

  POST /v1/models/{model}/predict    tabular models on Triton (or in-process ONNX Runtime)
  POST /v1/completions               proxied to vLLM (streaming supported)
  POST /v1/chat/completions          proxied to vLLM (streaming supported)
  GET  /v1/models                    models and versions available
  GET  /healthz                      liveness
  GET  /readyz                       readiness: 200 only when every configured backend is ready
  GET  /metrics                      Prometheus

Version routing: a request can pin a version with `?version=1`. Otherwise the
per-model TRAFFIC_SPLIT decides (e.g. 10% FP32, 90% INT8), falling back to
DEFAULT_MODEL_VERSION. The version that served each response is returned in the
`x-model-version` header and recorded in metrics, so a canary can be compared
against the incumbent on the same dashboard.
"""

from __future__ import annotations

import asyncio
import random
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from .backends import BackendError, LocalBackend, TritonBackend, VLLMBackend
from .config import Settings

REQS = Counter("gateway_requests_total", "Requests", ["route", "model", "version", "status"])
LAT = Histogram(
    "gateway_request_seconds",
    "Gateway latency",
    ["route", "model", "version"],
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)
ROWS = Counter("gateway_rows_total", "Rows scored by tabular models", ["model", "version"])
IN_FLIGHT = Gauge("gateway_in_flight", "Requests currently being processed")
SHED = Counter("gateway_shed_total", "Requests rejected by load shedding")

ctx: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = Settings()
    ctx["settings"] = s
    ctx["models"] = (
        TritonBackend(s.triton_url, s.model_repository, s.upstream_timeout_s)
        if s.triton_url
        else LocalBackend(s.model_repository)
    )
    ctx["llm"] = VLLMBackend(s.vllm_url, s.upstream_timeout_s) if s.vllm_url else None
    ctx["slots"] = asyncio.Semaphore(s.max_in_flight)
    yield
    await ctx["models"].aclose()
    if ctx["llm"]:
        await ctx["llm"].aclose()


class LoadSheddingMiddleware:
    """Reject /v1/* requests with 429 once MAX_IN_FLIGHT are being processed.

    Written as raw ASGI rather than @app.middleware("http"): Starlette's
    BaseHTTPMiddleware re-wraps every response as a stream, which costs real
    throughput on a hot path and releases the slot before a streamed LLM
    response has finished. Here the slot is held until the last byte is sent.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope["path"].startswith("/v1/"):
            return await self.app(scope, receive, send)
        slots: asyncio.Semaphore = ctx["slots"]
        if slots.locked():
            SHED.inc()
            resp = JSONResponse({"detail": "overloaded, retry later"}, status_code=429, headers={"retry-after": "1"})
            return await resp(scope, receive, send)
        async with slots:
            IN_FLIGHT.inc()
            try:
                await self.app(scope, receive, send)
            finally:
                IN_FLIGHT.dec()


app = FastAPI(title="ML Inference Gateway", version="1.0.0", lifespan=lifespan)
app.add_middleware(LoadSheddingMiddleware)


class PredictRequest(BaseModel):
    instances: list[list[float]] = Field(..., min_length=1)


def pick_version(model: str, pinned: str | None, available: list[str]) -> str:
    if pinned:
        return pinned
    split: dict[str, float] = ctx["settings"].traffic.get(model, {})
    if split:
        versions, weights = zip(*split.items(), strict=True)
        return random.choices(versions, weights=weights, k=1)[0]
    default = ctx["settings"].default_version
    return default if default in available else available[-1]


@app.post("/v1/models/{model}/predict")
async def predict(model: str, body: PredictRequest, response: Response, version: str | None = None):
    available = ctx["models"].models().get(model)
    if not available:
        raise HTTPException(404, f"unknown model {model}")
    if len(body.instances) > ctx["settings"].max_batch_rows:
        raise HTTPException(413, f"max {ctx['settings'].max_batch_rows} rows per request")
    v = pick_version(model, version, available)
    rows = np.asarray(body.instances, dtype=np.float32)
    if rows.ndim != 2:
        raise HTTPException(422, "instances must be a 2-D array")

    start = time.perf_counter()
    try:
        out = await ctx["models"].predict(model, v, rows)
    except BackendError as e:
        REQS.labels("predict", model, v, str(e.status)).inc()
        raise HTTPException(e.status, e.detail) from e
    LAT.labels("predict", model, v).observe(time.perf_counter() - start)
    REQS.labels("predict", model, v, "200").inc()
    ROWS.labels(model, v).inc(len(rows))
    response.headers["x-model-version"] = v
    return {"model": model, "version": v, **out}


async def _llm(route: str, path: str, request: Request):
    llm: VLLMBackend | None = ctx["llm"]
    if llm is None:
        raise HTTPException(503, "no LLM backend configured (set VLLM_URL)")
    body = await request.json()
    model = str(body.get("model", "default"))
    start = time.perf_counter()
    if body.get("stream"):
        # TTFT-dominated; record total stream duration when the generator finishes.
        async def gen():
            status = "200"
            try:
                async for chunk in llm.stream(path, body):
                    yield chunk
            except BackendError as e:
                status = str(e.status)
                yield f'data: {{"error": {e.detail!r}}}\n\n'.encode()
            finally:
                LAT.labels(route, model, "stream").observe(time.perf_counter() - start)
                REQS.labels(route, model, "stream", status).inc()

        return StreamingResponse(gen(), media_type="text/event-stream")
    try:
        out = await llm.forward(path, body)
    except BackendError as e:
        REQS.labels(route, model, "-", str(e.status)).inc()
        raise HTTPException(e.status, e.detail) from e
    LAT.labels(route, model, "-").observe(time.perf_counter() - start)
    REQS.labels(route, model, "-", "200").inc()
    return out


@app.post("/v1/completions")
async def completions(request: Request):
    return await _llm("completions", "/v1/completions", request)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _llm("chat", "/v1/chat/completions", request)


@app.get("/v1/models")
def list_models():
    return {"tabular": ctx["models"].models(), "llm_backend": bool(ctx["llm"])}


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    checks = {"models": await ctx["models"].ready()}
    if ctx["llm"]:
        checks["llm"] = await ctx["llm"].ready()
    ok = all(checks.values())
    return JSONResponse({"ready": ok, **checks}, status_code=200 if ok else 503)


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
