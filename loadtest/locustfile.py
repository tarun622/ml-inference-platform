"""Locust scenario for the gateway.

Mixed traffic: mostly single-row tabular predictions (the latency-sensitive
path), some small batches, and optional LLM completions when an LLM backend is
up (run with --tags llm, or leave untagged to include it).

  # 30-minute soak at 4,000 concurrent users, the target in the README
  locust -f loadtest/locustfile.py --host http://GATEWAY --headless \
         --users 4000 --spawn-rate 200 --run-time 30m --csv loadtest/soak

Use several workers (`--processes -1`) or distributed mode: one Locust
process tops out around a few thousand RPS by itself.
"""

import random
from pathlib import Path

import numpy as np
from locust import FastHttpUser, constant_throughput, tag, task

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = {
    m: np.load(ROOT / "model_repository" / m / "testset.npz")["X"][:2000].tolist()
    for m in ("fraud_mlp", "digits_mlp", "risk_mlp")
}


class TabularClient(FastHttpUser):
    wait_time = constant_throughput(2)  # each user aims for 2 req/s

    @tag("tabular")
    @task(20)
    def single_row(self):
        model = random.choice(list(SAMPLES))
        self.client.post(
            f"/v1/models/{model}/predict",
            json={"instances": [random.choice(SAMPLES[model])]},
            name=f"predict {model} x1",
        )

    @tag("tabular")
    @task(3)
    def small_batch(self):
        model = random.choice(list(SAMPLES))
        self.client.post(
            f"/v1/models/{model}/predict",
            json={"instances": random.sample(SAMPLES[model], 32)},
            name=f"predict {model} x32",
        )

    @tag("llm")
    @task(1)
    def completion(self):
        self.client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5-7b",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "Summarise why dynamic batching helps GPU throughput."}],
            },
            name="llm chat",
        )
