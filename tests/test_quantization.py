import numpy as np
import onnxruntime as ort
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier

from quantization.quantize import quantize_model


def test_dynamic_int8_shrinks_model_and_keeps_predictions(tmp_path):
    X, y = make_classification(n_samples=2000, n_features=20, random_state=0)
    X = X.astype(np.float32)
    clf = MLPClassifier((256, 128), max_iter=60, random_state=0).fit(X, y)
    onx = to_onnx(
        clf,
        initial_types=[("input", FloatTensorType([None, 20]))],
        options={id(clf): {"zipmap": False}},
        target_opset=17,
    )
    model_dir = tmp_path / "m"
    (model_dir / "1").mkdir(parents=True)
    (model_dir / "1" / "model.onnx").write_bytes(onx.SerializeToString())

    int8 = quantize_model(model_dir, "dynamic")

    fp32_size = (model_dir / "1" / "model.onnx").stat().st_size
    assert int8.stat().st_size < fp32_size / 3

    a = ort.InferenceSession(str(model_dir / "1" / "model.onnx")).run(None, {"input": X})[0]
    b = ort.InferenceSession(str(int8)).run(None, {"input": X})[0]
    assert (a == b).mean() > 0.98
