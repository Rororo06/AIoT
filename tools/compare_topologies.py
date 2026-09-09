"""Evidence for the Conv1D -> Conv2D design iteration described in the report.

Both topologies compute the same function; only the tensor layout differs.
This script converts each of them to int8 TFLite and prints the resulting
operator list, operator count and file size.

Run:  python tools/compare_topologies.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training import config as C            # noqa: E402
from training.data import Dataset           # noqa: E402
from training.models import build_variant   # noqa: E402


def to_int8(model, rep: np.ndarray) -> bytes:
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([w[np.newaxis, ...]] for w in rep)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def main() -> None:
    ds = Dataset()
    rep_flat = ds.representative_windows(64)                 # (N, 128, 1, 6)
    rep_1d = rep_flat[:, :, 0, :]                            # (N, 128, 6)

    rows = []
    for label, conv1d, rep in (("Conv2D (deployed)", False, rep_flat),
                               ("Conv1D (first iteration)", True, rep_1d)):
        model = build_variant("small", conv1d=conv1d)
        blob = to_int8(model, rep)
        it = tf.lite.Interpreter(model_content=blob)
        it.allocate_tensors()
        ops = [d["op_name"] for d in it._get_ops_details() if d["op_name"] != "DELEGATE"]
        rows.append((label, len(blob), len(ops), sorted(set(ops))))

    print()
    print(f"{'topology':26s} {'int8 bytes':>10s} {'#ops':>5s}  unique kernels")
    for label, size, n_ops, uniq in rows:
        print(f"{label:26s} {size:10d} {n_ops:5d}  {', '.join(uniq)}")
    print()
    print("The deployed model needs fewer TFLite Micro kernels registered in the")
    print("MicroMutableOpResolver, which is what shrinks the firmware.")


if __name__ == "__main__":
    main()
