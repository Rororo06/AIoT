"""Print the operator set and tensor quantization details of a .tflite model.

Used to justify the TFLite Micro op resolver registrations in the firmware and
to document the quantization parameters in the report.

Run:  python tools/inspect_tflite.py models/har_small_int8.tflite
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]


def inspect(path: Path) -> None:
    blob = path.read_bytes()
    it = tf.lite.Interpreter(model_content=blob)
    it.allocate_tensors()

    print(f"\n=== {path.name} ({len(blob)} bytes) ===")
    for kind, details in (("input", it.get_input_details()), ("output", it.get_output_details())):
        for d in details:
            scale, zero = d["quantization"]
            print(f"{kind:6s} {d['name']:24s} shape={tuple(d['shape'])} "
                  f"dtype={np.dtype(d['dtype']).name} scale={scale:.10g} zero_point={zero}")

    ops = [d["op_name"] for d in it._get_ops_details()]
    print(f"operators ({len(ops)}): {ops}")
    print(f"unique    : {sorted(set(ops))}")

    total = 0
    for d in it.get_tensor_details():
        nbytes = int(np.prod(d["shape"])) * np.dtype(d["dtype"]).itemsize if d["shape"].size else 0
        total += nbytes
    print(f"sum of tensor bytes (upper bound on arena): {total}")


def main(argv: list[str]) -> None:
    paths = [Path(a) for a in argv] or sorted((ROOT / "models").glob("*.tflite"))
    for p in paths:
        inspect(p if p.is_absolute() else ROOT / p)


if __name__ == "__main__":
    main(sys.argv[1:])
