"""Train the float32 baselines (one per width variant).

Run:  python -m training.train            # all variants
      python -m training.train medium     # a single variant
"""

from __future__ import annotations

import json
import random
import sys

import numpy as np
import tensorflow as tf
import keras

from training import config as C
from training.data import Dataset
from training.models import build_variant


def set_seeds(seed: int = C.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)


def train_variant(variant: str, ds: Dataset) -> dict:
    set_seeds()
    X_train, y_train = ds.split("train")
    X_val, y_val = ds.split("val")

    model = build_variant(variant)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=C.LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    ckpt = C.MODELS_DIR / f"har_{variant}.keras"
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=C.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=C.REDUCE_LR_PATIENCE, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=C.EPOCHS,
        batch_size=C.BATCH_SIZE,
        class_weight=ds.class_weights(),
        callbacks=callbacks,
        verbose=2,
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    record = {
        "variant": variant,
        "params": int(model.count_params()),
        "epochs_run": len(history.history["loss"]),
        "val_loss": float(val_loss),
        "val_accuracy": float(val_acc),
        "checkpoint": str(ckpt.relative_to(C.ROOT)).replace("\\", "/"),
    }
    print(f"[{variant}] params={record['params']} val_acc={val_acc:.4f}")
    return record


def main(argv: list[str]) -> None:
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    variants = argv or list(C.MODEL_VARIANTS)
    ds = Dataset()
    records = [train_variant(v, ds) for v in variants]
    out = C.RESULTS_DIR / "training_summary.json"
    out.write_text(json.dumps(records, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
