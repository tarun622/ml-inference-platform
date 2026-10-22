"""INT8 quantization of ONNX models with ONNX Runtime.

Two modes:

  dynamic  Weights are quantized to INT8 offline; activations are quantized on
           the fly per batch. No calibration data needed. Best fit for MLPs and
           transformers dominated by MatMul/Gemm.

  static   Weights and activations are quantized offline using ranges collected
           from a calibration set (QDQ format, which TensorRT and the Triton
           ONNX Runtime backend can both consume). Usually faster at inference
           but needs representative data.

The FP32 model stays as version 1 and the INT8 model is written as version 2
of the same Triton model, so both can be served side by side and the gateway
can pin or shift traffic between versions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process


class NpzCalibrationReader(CalibrationDataReader):
    def __init__(self, X: np.ndarray, input_name: str = "input", batch: int = 64, max_batches: int = 50):
        self._batches = iter({input_name: X[i : i + batch]} for i in range(0, min(len(X), batch * max_batches), batch))

    def get_next(self):
        return next(self._batches, None)


def quantize_model(model_dir: Path, mode: str = "dynamic") -> Path:
    fp32 = model_dir / "1" / "model.onnx"
    out_dir = model_dir / "2"
    out_dir.mkdir(exist_ok=True)
    int8 = out_dir / "model.onnx"
    prep = model_dir / "model.prep.onnx"

    # Shape inference + graph optimisation first; quantization works better on a cleaned graph.
    quant_pre_process(str(fp32), str(prep), skip_symbolic_shape=True)

    if mode == "dynamic":
        quantize_dynamic(str(prep), str(int8), weight_type=QuantType.QInt8, per_channel=True)
    else:
        X = np.load(model_dir / "calib.npz")["X"]  # train-split sample, not the test set
        quantize_static(
            str(prep),
            str(int8),
            NpzCalibrationReader(X),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
        )
    prep.unlink(missing_ok=True)
    return int8


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="model_repository")
    ap.add_argument("--mode", choices=["dynamic", "static"], default="dynamic")
    ap.add_argument("--models", nargs="*")
    args = ap.parse_args()

    repo = Path(args.repo)
    names = args.models or sorted(p.name for p in repo.iterdir() if (p / "1" / "model.onnx").exists())
    for name in names:
        out = quantize_model(repo / name, args.mode)
        fp32_size = (repo / name / "1" / "model.onnx").stat().st_size
        print(f"{name}: {fp32_size / 1e6:.2f} MB -> {out.stat().st_size / 1e6:.2f} MB ({args.mode})")


if __name__ == "__main__":
    main()
