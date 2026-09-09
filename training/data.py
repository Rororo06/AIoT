"""Loading helpers shared by training, quantization and evaluation."""

from __future__ import annotations

import numpy as np

from training import config as C


class Dataset:
    def __init__(self, path=None):
        path = path or (C.PROCESSED_DIR / "har_dataset.npz")
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found - run `python -m training.prepare_data` first"
            )
        with np.load(path, allow_pickle=False) as z:
            self.mean = z["mean"]
            self.std = z["std"]
            self.raw = {k: z[k] for k in z.files if k != "mean" and k != "std"}

    def normalize(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype(np.float32)

    @staticmethod
    def as_model_input(X: np.ndarray) -> np.ndarray:
        """(N, 128, 6) -> (N, 128, 1, 6); a view, the memory layout is identical
        to what the firmware keeps in its ring buffer."""
        return X[:, :, np.newaxis, :] if X.ndim == 3 else X

    def split(self, name: str, normalized: bool = True, model_layout: bool = True):
        X = self.raw[f"X_{name}"]
        y = self.raw[f"y_{name}"]
        if normalized:
            X = self.normalize(X)
        if model_layout:
            X = self.as_model_input(X)
        return X, y

    def subjects(self, name: str) -> np.ndarray:
        return self.raw[f"subj_{name}"]

    def trials(self, name: str) -> np.ndarray:
        return self.raw[f"trial_{name}"]

    def class_weights(self) -> dict[int, float]:
        y = self.raw["y_train"]
        counts = np.bincount(y, minlength=C.N_CLASSES).astype(np.float64)
        counts[counts == 0] = 1.0
        w = counts.sum() / (C.N_CLASSES * counts)
        return {i: float(w[i]) for i in range(C.N_CLASSES)}

    def representative_windows(self, n: int, seed: int = C.SEED) -> np.ndarray:
        """Class-stratified sample of TRAIN windows for the quantization calibration."""
        X, y = self.split("train")
        rng = np.random.default_rng(seed)
        per_class = max(1, n // C.N_CLASSES)
        picked = []
        for k in range(C.N_CLASSES):
            idx = np.flatnonzero(y == k)
            take = min(per_class, idx.size)
            picked.append(rng.choice(idx, size=take, replace=False))
        idx = np.concatenate(picked)
        rng.shuffle(idx)
        return X[idx]
