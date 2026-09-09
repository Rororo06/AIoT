"""Convert every trained variant to float32 and full-integer int8 TFLite, then
evaluate both with the TFLite interpreter and record the trade-off metrics.

The int8 conversion is the "AI on Edge" optimisation technique applied in this
project. Settings used (post-training full-integer quantization):

    optimizations          = [tf.lite.Optimize.DEFAULT]
    representative_dataset = 300 class-stratified TRAIN windows
    supported_ops          = [TFLITE_BUILTINS_INT8]
    inference_input_type   = tf.int8
    inference_output_type  = tf.int8

Weights are per-channel int8, activations per-tensor int8. The input tensor is
int8 as well, so the ESP32 quantises the normalised window itself and no float
kernels are linked into the firmware.

Run:  python -m training.quantize
"""

from __future__ import annotations

import json
import time

import numpy as np
import tensorflow as tf
import keras

from training import config as C
from training.data import Dataset


def convert_float32(model: keras.Model) -> bytes:
    return tf.lite.TFLiteConverter.from_keras_model(model).convert()


def convert_int8(model: keras.Model, rep_windows: np.ndarray) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset():
        for w in rep_windows:
            yield [w[np.newaxis, ...].astype(np.float32)]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def tflite_predict(tflite_model: bytes, X: np.ndarray) -> tuple[np.ndarray, dict]:
    """Run the interpreter sample by sample and return predictions + timings."""
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    in_scale, in_zero = inp["quantization"]
    quantized_input = inp["dtype"] == np.int8

    preds = np.empty(X.shape[0], dtype=np.int64)
    latencies = np.empty(X.shape[0], dtype=np.float64)
    for i, window in enumerate(X):
        x = window[np.newaxis, ...]
        if quantized_input:
            x = np.clip(np.round(x / in_scale + in_zero), -128, 127).astype(np.int8)
        else:
            x = x.astype(np.float32)
        interpreter.set_tensor(inp["index"], x)
        t0 = time.perf_counter()
        interpreter.invoke()
        latencies[i] = (time.perf_counter() - t0) * 1000.0
        preds[i] = int(np.argmax(interpreter.get_tensor(out["index"])[0]))

    info = {
        "host_latency_ms_mean": float(latencies[5:].mean()),
        "host_latency_ms_p95": float(np.percentile(latencies[5:], 95)),
        "input_dtype": np.dtype(inp["dtype"]).name,
        "output_dtype": np.dtype(out["dtype"]).name,
        "input_scale": float(in_scale),
        "input_zero_point": int(in_zero),
    }
    return preds, info


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import (
        accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
    )

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(C.N_CLASSES), zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": {
            C.CLASS_NAMES[k]: {
                "precision": float(prec[k]),
                "recall": float(rec[k]),
                "f1": float(f1[k]),
                "support": int(support[k]),
            }
            for k in range(C.N_CLASSES)
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=range(C.N_CLASSES)
        ).tolist(),
    }


def main() -> None:
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ds = Dataset()
    X_test, y_test = ds.split("test")
    rep = ds.representative_windows(C.REPRESENTATIVE_SAMPLES)
    print(f"representative dataset: {rep.shape[0]} windows from TRAIN")

    records = []
    predictions = {}
    for variant in C.MODEL_VARIANTS:
        ckpt = C.MODELS_DIR / f"har_{variant}.keras"
        if not ckpt.exists():
            print(f"skip {variant}: {ckpt} missing")
            continue
        model = keras.models.load_model(ckpt)

        for precision, blob in (
            ("float32", convert_float32(model)),
            ("int8", convert_int8(model, rep)),
        ):
            path = C.MODELS_DIR / f"har_{variant}_{precision}.tflite"
            path.write_bytes(blob)
            preds, info = tflite_predict(blob, X_test)
            m = metrics(y_test, preds)
            record = {
                "variant": variant,
                "precision": precision,
                "params": int(model.count_params()),
                "tflite_bytes": len(blob),
                **{k: v for k, v in info.items() if v is not None},
                **{k: v for k, v in m.items() if k != "per_class"},
                "per_class": m["per_class"],
                "artifact": str(path.relative_to(C.ROOT)).replace("\\", "/"),
            }
            records.append(record)
            predictions[f"{variant}_{precision}"] = preds
            print(
                f"{variant:6s} {precision:8s} size={len(blob):6d}B "
                f"acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f} "
                f"fall_recall={m['per_class']['FALLING']['recall']:.4f}"
            )

    (C.RESULTS_DIR / "tflite_metrics.json").write_text(json.dumps(records, indent=2))
    np.savez_compressed(
        C.RESULTS_DIR / "test_predictions.npz", y_true=y_test, **predictions
    )
    print(f"\nwrote {C.RESULTS_DIR / 'tflite_metrics.json'}")


if __name__ == "__main__":
    main()
