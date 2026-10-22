"""Compare FP32 (version 1) and INT8 (version 2) of every model.

For each model it reports:
  - on-disk size
  - F1 on the held-out test split (macro for multi-class)
  - prediction agreement between FP32 and INT8
  - single-request latency p50/p99 (batch 1)
  - batched throughput (rows/s at batch 256)

Run on the hardware you deploy to. CPU numbers do not transfer to GPUs: on a
GPU, INT8 gains come from TensorRT INT8 kernels and memory bandwidth, which
this script cannot measure.

  python quantization/benchmark.py --out quantization/results.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from sklearn.metrics import f1_score


def session(path: Path, threads: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def bench(path: Path, X: np.ndarray, y: np.ndarray, threads: int, n_single: int = 2000) -> dict:
    sess = session(path, threads)

    label, probs = sess.run(None, {"input": X})
    avg = "binary" if probs.shape[1] == 2 else "macro"
    f1 = f1_score(y, label, average=avg)

    rng = np.random.default_rng(0)
    rows = X[rng.integers(0, len(X), n_single)]
    for i in range(100):  # warm-up
        sess.run(None, {"input": rows[i : i + 1]})
    lat = []
    for i in range(n_single):
        t = time.perf_counter()
        sess.run(None, {"input": rows[i : i + 1]})
        lat.append((time.perf_counter() - t) * 1e6)
    lat.sort()

    batch = np.ascontiguousarray(X[:256] if len(X) >= 256 else np.resize(X, (256, X.shape[1])))
    n_iter, t0 = 200, time.perf_counter()
    for _ in range(n_iter):
        sess.run(None, {"input": batch})
    tput = n_iter * len(batch) / (time.perf_counter() - t0)

    return {
        "size_mb": round(path.stat().st_size / 1e6, 3),
        "f1": round(float(f1), 4),
        "p50_us": round(lat[len(lat) // 2], 1),
        "p99_us": round(lat[int(len(lat) * 0.99)], 1),
        "rows_per_s_batch256": round(tput),
        "_labels": label,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="model_repository")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default="quantization/results.json")
    args = ap.parse_args()

    repo = Path(args.repo)
    results: dict = {
        "machine": {
            "cpu": platform.processor() or platform.machine(),
            "cores": os.cpu_count(),
            "threads": args.threads,
            "onnxruntime": ort.__version__,
        },
        "models": {},
    }
    for model_dir in sorted(p for p in repo.iterdir() if (p / "2" / "model.onnx").exists()):
        data = np.load(model_dir / "testset.npz")
        X, y = data["X"].astype(np.float32), data["y"]
        fp32 = bench(model_dir / "1" / "model.onnx", X, y, args.threads)
        int8 = bench(model_dir / "2" / "model.onnx", X, y, args.threads)
        agreement = float((fp32.pop("_labels") == int8.pop("_labels")).mean())
        results["models"][model_dir.name] = {
            "fp32": fp32,
            "int8": int8,
            "size_reduction_x": round(fp32["size_mb"] / int8["size_mb"], 2),
            "f1_delta_points": round(100 * (int8["f1"] - fp32["f1"]), 2),
            "prediction_agreement": round(agreement, 4),
            "batched_speedup_x": round(int8["rows_per_s_batch256"] / fp32["rows_per_s_batch256"], 2),
        }
        r = results["models"][model_dir.name]
        print(
            f"{model_dir.name:12s} size {fp32['size_mb']:.2f}->{int8['size_mb']:.2f}MB ({r['size_reduction_x']}x) | "
            f"F1 {fp32['f1']:.4f}->{int8['f1']:.4f} ({r['f1_delta_points']:+.2f} pts) | agree {agreement:.4f} | "
            f"p50 {fp32['p50_us']:.0f}->{int8['p50_us']:.0f}us | batch {r['batched_speedup_x']}x"
        )
    Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
