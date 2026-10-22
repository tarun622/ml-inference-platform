"""Train the non-LLM models served by Triton and export them to ONNX (FP32).

Three MLP classifiers on tabular data, chosen so everything trains in a couple of
minutes on a laptop CPU and runs without downloading datasets:

  fraud_mlp    imbalanced binary fraud-style data (synthetic, 40 features, ~3% positives)
  digits_mlp   10-class handwritten digits (sklearn's bundled 8x8 digits)
  risk_mlp     binary tumour malignancy (sklearn's bundled breast cancer data)

Each model is written to model_repository/<name>/1/model.onnx with a held-out
test split (for the benchmark) and a calibration sample from the training split
(for static quantization) saved next to it.

Swap in your own models: anything exported to ONNX with a float input named
`input` drops into the same Triton layout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
from sklearn.datasets import load_breast_cancer, load_digits, make_classification
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def datasets(seed: int) -> dict[str, tuple[np.ndarray, np.ndarray, tuple[int, ...]]]:
    fx, fy = make_classification(
        n_samples=120_000,
        n_features=40,
        n_informative=18,
        n_redundant=8,
        weights=[0.97, 0.03],
        class_sep=0.9,
        flip_y=0.005,
        random_state=seed,
    )
    dx, dy = load_digits(return_X_y=True)
    bx, by = load_breast_cancer(return_X_y=True)
    return {
        "fraud_mlp": (fx, fy, (512, 512, 256)),
        "digits_mlp": (dx, dy, (512, 256)),
        "risk_mlp": (bx, by, (256, 128)),
    }


def main(out_dir: str = "model_repository", seed: int = 42) -> None:
    summary = {}
    for name, (X, y, hidden) in datasets(seed).items():
        X = X.astype(np.float32)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
        model = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden, early_stopping=True, max_iter=200, random_state=seed),
        )
        model.fit(X_tr, y_tr)
        avg = "binary" if len(np.unique(y)) == 2 else "macro"
        f1 = f1_score(y_te, model.predict(X_te), average=avg)

        onx = to_onnx(
            model,
            initial_types=[("input", FloatTensorType([None, X.shape[1]]))],
            options={id(model.steps[-1][1]): {"zipmap": False}},
            target_opset=17,
        )
        version_dir = Path(out_dir) / name / "1"
        version_dir.mkdir(parents=True, exist_ok=True)
        (version_dir / "model.onnx").write_bytes(onx.SerializeToString())
        np.savez_compressed(Path(out_dir) / name / "testset.npz", X=X_te, y=y_te)
        # calibration data for static quantization comes from the TRAIN split, never the test split
        rng = np.random.default_rng(seed)
        calib = X_tr[rng.choice(len(X_tr), size=min(3200, len(X_tr)), replace=False)]
        np.savez_compressed(Path(out_dir) / name / "calib.npz", X=calib)
        summary[name] = {
            "features": X.shape[1],
            "classes": int(len(np.unique(y))),
            "hidden": hidden,
            "f1_sklearn": round(f1, 4),
        }
        print(f"{name}: F1={f1:.4f} -> {version_dir / 'model.onnx'}")
    (Path(out_dir) / "models.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="model_repository")
    args = ap.parse_args()
    main(args.out)
