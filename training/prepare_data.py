"""Build the windowed HAR dataset from the raw SisFall recordings.

Pipeline
--------
raw .txt (200 Hz, ADC counts)
  -> physical units (g, deg/s)
  -> anti-aliased decimation to 50 Hz
  -> per-class window extraction rules
  -> subject-wise train / val / test split
  -> per-channel standardisation with TRAIN statistics only

Run:  python -m training.prepare_data
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.signal import decimate

from training import config as C


# --------------------------------------------------------------------------- #
# Raw file handling
# --------------------------------------------------------------------------- #
def parse_trial(path: Path) -> np.ndarray:
    """Parse one SisFall trial file into an (n_samples, 9) int array.

    Records are ';'-terminated and comma separated. A few files in the dataset
    contain trailing whitespace or truncated last records, which are skipped.
    """
    text = path.read_text(errors="ignore")
    rows = []
    for record in text.split(";"):
        record = record.strip()
        if not record:
            continue
        parts = record.split(",")
        if len(parts) != 9:
            continue
        try:
            rows.append([int(p) for p in parts])
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"no parsable records in {path}")
    return np.asarray(rows, dtype=np.int32)


def to_physical(raw: np.ndarray) -> np.ndarray:
    """Convert ADC counts to (g, g, g, dps, dps, dps) using the dataset Readme."""
    accel = raw[:, list(C.COL_ACCEL)].astype(np.float32) * C.ADXL345_G_PER_LSB
    gyro = raw[:, list(C.COL_GYRO)].astype(np.float32) * C.ITG3200_DPS_PER_LSB
    signal = np.concatenate([accel, gyro], axis=1)
    # Clip to the dynamic range the MPU6050 is configured for on the device, so
    # that training data cannot contain values the deployed sensor could not
    # report. With +-16 g / +-2000 dps this is a no-op for SisFall, which is
    # exactly why those ranges were chosen.
    signal[:, :3] = np.clip(signal[:, :3], -C.MPU6050_ACCEL_RANGE_G, C.MPU6050_ACCEL_RANGE_G)
    signal[:, 3:] = np.clip(signal[:, 3:], -C.MPU6050_GYRO_RANGE_DPS, C.MPU6050_GYRO_RANGE_DPS)
    return signal


def downsample(signal: np.ndarray) -> np.ndarray:
    """200 Hz -> 50 Hz with a zero-phase FIR anti-aliasing filter."""
    out = decimate(signal, C.DECIMATION, ftype="fir", zero_phase=True, axis=0)
    return np.ascontiguousarray(out, dtype=np.float32)


def load_trial(path: Path) -> np.ndarray:
    return downsample(to_physical(parse_trial(path)))


# --------------------------------------------------------------------------- #
# Window extraction rules (one per class)
# --------------------------------------------------------------------------- #
def sliding_windows(signal: np.ndarray, stride: int) -> list[np.ndarray]:
    n = signal.shape[0]
    return [
        signal[i:i + C.WINDOW]
        for i in range(0, n - C.WINDOW + 1, stride)
    ]


def walking_windows(signal: np.ndarray) -> list[np.ndarray]:
    """Steady-state walking: drop WALK_TRIM_S at both ends of the trial."""
    trim = int(C.WALK_TRIM_S * C.FS_HZ)
    core = signal[trim:signal.shape[0] - trim] if signal.shape[0] > 2 * trim + C.WINDOW else signal
    return sliding_windows(core, C.STRIDE)


def _longest_stationary_segment(signal: np.ndarray) -> tuple[int, int]:
    """Longest run where the rolling std of |a| stays below the still threshold."""
    mag = np.linalg.norm(signal[:, :3], axis=1)
    w = C.SIT_STATIONARY_WIN
    if mag.size < w:
        return 0, 0
    # rolling std via cumulative sums
    c1 = np.concatenate([[0.0], np.cumsum(mag, dtype=np.float64)])
    c2 = np.concatenate([[0.0], np.cumsum(mag.astype(np.float64) ** 2)])
    mean = (c1[w:] - c1[:-w]) / w
    var = np.maximum((c2[w:] - c2[:-w]) / w - mean ** 2, 0.0)
    still = np.sqrt(var) < C.SIT_STATIONARY_STD_G      # length n - w + 1

    best_len = best_start = 0
    run_start = None
    for i, flag in enumerate(still):
        if flag and run_start is None:
            run_start = i
        elif not flag and run_start is not None:
            if i - run_start > best_len:
                best_len, best_start = i - run_start, run_start
            run_start = None
    if run_start is not None and still.size - run_start > best_len:
        best_len, best_start = still.size - run_start, run_start
    # a "still" flag at index i covers samples [i, i + w)
    return best_start, best_start + best_len + w - 1


def sitting_windows(signal: np.ndarray) -> list[np.ndarray]:
    """Seated hold phase of a sit-down / stand-up trial."""
    start, end = _longest_stationary_segment(signal)
    segment = signal[start:end]
    if segment.shape[0] < C.WINDOW:
        return []
    return sliding_windows(segment, C.SIT_STRIDE)


def falling_windows(signal: np.ndarray) -> list[np.ndarray]:
    """One window centred on the impact peak of |a|."""
    mag = np.linalg.norm(signal[:, :3], axis=1)
    peak = int(np.argmax(mag))
    start = peak - C.WINDOW // 2
    start = max(0, min(start, signal.shape[0] - C.WINDOW))
    if signal.shape[0] < C.WINDOW:
        return []
    return [signal[start:start + C.WINDOW]]


EXTRACTORS = {
    C.LABEL_WALKING: (C.WALKING_CODES, walking_windows),
    C.LABEL_SITTING: (C.SITTING_CODES, sitting_windows),
    C.LABEL_FALLING: (C.FALL_CODES, falling_windows),
}


# --------------------------------------------------------------------------- #
# Dataset assembly
# --------------------------------------------------------------------------- #
def build() -> dict[str, np.ndarray]:
    if not C.RAW_DIR.exists():
        sys.exit(
            f"raw dataset not found at {C.RAW_DIR}\n"
            "run: python tools/download_dataset.py"
        )

    windows: list[np.ndarray] = []
    labels: list[int] = []
    subjects: list[str] = []
    trials: list[str] = []

    all_subjects = sorted(p.name for p in C.RAW_DIR.iterdir() if p.is_dir())
    for subject in all_subjects:
        for path in sorted((C.RAW_DIR / subject).glob("*.txt")):
            code = path.name.split("_")[0]
            label = next((lb for lb, (codes, _) in EXTRACTORS.items() if code in codes), None)
            if label is None:
                continue
            try:
                signal = load_trial(path)
            except ValueError as exc:
                print(f"  skip {path.name}: {exc}")
                continue
            for w in EXTRACTORS[label][1](signal):
                if w.shape == (C.WINDOW, C.N_CHANNELS):
                    windows.append(w)
                    labels.append(label)
                    subjects.append(subject)
                    trials.append(path.stem)
        print(f"  {subject}: {len(windows)} windows so far")

    X = np.stack(windows).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    subj = np.asarray(subjects)
    trial = np.asarray(trials)

    splits = {
        "train": np.isin(subj, C.TRAIN_SUBJECTS),
        "val": np.isin(subj, C.VAL_SUBJECTS),
        "test": np.isin(subj, C.TEST_SUBJECTS),
    }

    rng = np.random.default_rng(C.SEED)
    # Class balancing: WALKING dominates by construction (100 s trials). Cap it
    # in train/val so the loss is not swamped; the test split keeps its natural
    # distribution and is reported with macro-F1 and per-class recall.
    for name in ("train", "val"):
        mask = splits[name]
        idx = np.flatnonzero(mask)
        n_fall = int((y[idx] == C.LABEL_FALLING).sum())
        cap = max(n_fall * 2, 1)
        walk_idx = idx[y[idx] == C.LABEL_WALKING]
        if walk_idx.size > cap:
            drop = rng.choice(walk_idx, size=walk_idx.size - cap, replace=False)
            mask = mask.copy()
            mask[drop] = False
            splits[name] = mask

    out: dict[str, np.ndarray] = {}
    for name, mask in splits.items():
        out[f"X_{name}"] = X[mask]
        out[f"y_{name}"] = y[mask]
        out[f"subj_{name}"] = subj[mask]
        out[f"trial_{name}"] = trial[mask]

    # Standardisation statistics from the TRAIN split only.
    mean = out["X_train"].reshape(-1, C.N_CHANNELS).mean(axis=0)
    std = out["X_train"].reshape(-1, C.N_CHANNELS).std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    out["mean"] = mean.astype(np.float32)
    out["std"] = std.astype(np.float32)
    return out


def main() -> None:
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"reading {C.RAW_DIR}")
    data = build()

    npz_path = C.PROCESSED_DIR / "har_dataset.npz"
    np.savez_compressed(npz_path, **data)

    summary = {
        "window": C.WINDOW,
        "stride": C.STRIDE,
        "fs_hz": C.FS_HZ,
        "channels": list(C.CHANNELS),
        "classes": list(C.CLASS_NAMES),
        "norm_mean": data["mean"].tolist(),
        "norm_std": data["std"].tolist(),
        "splits": {},
    }
    for name in ("train", "val", "test"):
        y = data[f"y_{name}"]
        counts = Counter(y.tolist())
        summary["splits"][name] = {
            "windows": int(y.size),
            "subjects": sorted(set(data[f"subj_{name}"].tolist())),
            "per_class": {C.CLASS_NAMES[k]: int(counts.get(k, 0)) for k in range(C.N_CLASSES)},
        }
        print(f"{name:5s} n={y.size:6d}  " + "  ".join(
            f"{C.CLASS_NAMES[k]}={counts.get(k, 0)}" for k in range(C.N_CLASSES)))

    (C.PROCESSED_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {npz_path}")
    print(f"norm mean = {np.round(data['mean'], 4).tolist()}")
    print(f"norm std  = {np.round(data['std'], 4).tolist()}")


if __name__ == "__main__":
    main()
