"""Model definitions.

A small convolutional network over the (128, 6) window. Design constraints come
from the target device rather than from accuracy:

* Only ops that have int8 TFLite Micro reference kernels.
* No BatchNorm. It is foldable, but leaving it out keeps the float and int8
  graphs trivially comparable and shrinks the op set on device.
* GlobalAveragePooling instead of Flatten + Dense: this is what keeps the
  parameter count, and therefore the flash footprint, small.

Iteration note (kept in the repo on purpose, see report):
`build_cnn1d` was the first implementation. Keras `Conv1D` / `MaxPooling1D` /
`GlobalAveragePooling1D` are lowered by the TFLite converter into
EXPAND_DIMS -> CONV_2D -> RESHAPE triplets, giving 16 operators and forcing the
firmware to link EXPAND_DIMS and RESHAPE kernels. Rewriting the same maths with
an explicit (time, 1, channel) layout and `Conv2D` halves the operator count to
8 and removes two kernels from the firmware. `build_cnn` is the deployed
version; `build_cnn1d` is kept so the comparison is reproducible.
"""

from __future__ import annotations

import keras

from training import config as C

INPUT_SHAPE = (C.WINDOW, 1, C.N_CHANNELS)


def build_cnn(filters: tuple[int, int], dense: int, name: str) -> keras.Model:
    """Deployed topology: time on axis 0, a dummy axis 1, channels on axis 2."""
    inputs = keras.Input(shape=INPUT_SHAPE, name="window")
    x = keras.layers.Conv2D(filters[0], (5, 1), padding="same", activation="relu", name="conv1")(inputs)
    x = keras.layers.MaxPooling2D((2, 1), name="pool1")(x)
    x = keras.layers.Conv2D(filters[1], (3, 1), padding="same", activation="relu", name="conv2")(x)
    x = keras.layers.MaxPooling2D((2, 1), name="pool2")(x)
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.Dense(dense, activation="relu", name="fc1")(x)
    outputs = keras.layers.Dense(C.N_CLASSES, activation="softmax", name="probs")(x)
    return keras.Model(inputs, outputs, name=name)


def build_cnn1d(filters: tuple[int, int], dense: int, name: str) -> keras.Model:
    """First iteration, kept for the operator-count comparison in the report."""
    inputs = keras.Input(shape=(C.WINDOW, C.N_CHANNELS), name="window")
    x = keras.layers.Conv1D(filters[0], 5, padding="same", activation="relu", name="conv1")(inputs)
    x = keras.layers.MaxPooling1D(2, name="pool1")(x)
    x = keras.layers.Conv1D(filters[1], 3, padding="same", activation="relu", name="conv2")(x)
    x = keras.layers.MaxPooling1D(2, name="pool2")(x)
    x = keras.layers.GlobalAveragePooling1D(name="gap")(x)
    x = keras.layers.Dense(dense, activation="relu", name="fc1")(x)
    outputs = keras.layers.Dense(C.N_CLASSES, activation="softmax", name="probs")(x)
    return keras.Model(inputs, outputs, name=name)


def build_variant(variant: str, conv1d: bool = False) -> keras.Model:
    if variant not in C.MODEL_VARIANTS:
        raise KeyError(f"unknown variant {variant!r}; choose from {list(C.MODEL_VARIANTS)}")
    spec = C.MODEL_VARIANTS[variant]
    builder = build_cnn1d if conv1d else build_cnn
    suffix = "_conv1d" if conv1d else ""
    return builder(tuple(spec["filters"]), spec["dense"], name=f"har_{variant}{suffix}")
