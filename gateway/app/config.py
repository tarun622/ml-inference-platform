from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


def _json_env(name: str, default: dict) -> dict:
    raw = os.getenv(name)
    return json.loads(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    # Where non-LLM models are served. Empty -> run them in-process with ONNX Runtime (local dev / CI).
    triton_url: str = field(default_factory=lambda: os.getenv("TRITON_URL", ""))
    model_repository: str = field(default_factory=lambda: os.getenv("MODEL_REPOSITORY", "model_repository"))
    # vLLM OpenAI-compatible server. Empty -> LLM routes return 503.
    vllm_url: str = field(default_factory=lambda: os.getenv("VLLM_URL", ""))
    # Per-model traffic split across versions, e.g. {"fraud_mlp": {"1": 0.1, "2": 0.9}}
    traffic: dict = field(default_factory=lambda: _json_env("TRAFFIC_SPLIT", {}))
    default_version: str = field(default_factory=lambda: os.getenv("DEFAULT_MODEL_VERSION", "2"))
    # Load shedding: requests beyond this many in flight get 429 instead of queueing forever.
    max_in_flight: int = int(os.getenv("MAX_IN_FLIGHT", "512"))
    upstream_timeout_s: float = float(os.getenv("UPSTREAM_TIMEOUT_S", "30"))
    max_batch_rows: int = int(os.getenv("MAX_BATCH_ROWS", "1024"))
