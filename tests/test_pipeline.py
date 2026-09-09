"""Unit tests for the parts of the pipeline that are easy to get silently wrong.

Run:  python -m pytest -q
"""

from __future__ import annotations

import numpy as np
import pytest

from training import config as C
from training import prepare_data as P
from training.export_firmware import (
    ACCEL_LSB_PER_G, GYRO_LSB_PER_DPS, cfloat, to_raw_counts,
)


# --------------------------------------------------------------- unit scaling
def test_sensitivity_constants_match_datasheets():
    # SisFall Readme: ADXL345 13 bit +-16 g, ITG3200 16 bit +-2000 deg/s
    assert C.ADXL345_G_PER_LSB == pytest.approx(2 * 16 / 2 ** 13)
    assert C.ITG3200_DPS_PER_LSB == pytest.approx(2 * 2000 / 2 ** 16)
    # MPU6050 register map: AFS_SEL=3 -> 2048 LSB/g, FS_SEL=3 -> 16.4 LSB/(deg/s)
    assert ACCEL_LSB_PER_G == pytest.approx(2048.0)
    assert GYRO_LSB_PER_DPS == pytest.approx(16.384, rel=1e-3)


def test_to_physical_converts_counts_to_g_and_dps():
    raw = np.zeros((4, 9), dtype=np.int32)
    raw[:, 2] = 256          # ADXL345 z: 256 counts * 0.00390625 g = 1 g
    raw[:, 3] = 16           # ITG3200 x: 16 counts * 0.061035 = 0.977 deg/s
    out = P.to_physical(raw)
    assert out.shape == (4, 6)
    assert out[0, 2] == pytest.approx(1.0)
    assert out[0, 3] == pytest.approx(16 * C.ITG3200_DPS_PER_LSB)


def test_to_physical_clips_to_the_deployed_sensor_range():
    raw = np.zeros((1, 9), dtype=np.int32)
    raw[0, 0] = 10 ** 6      # far beyond +-16 g
    raw[0, 3] = 10 ** 6
    out = P.to_physical(raw)
    assert out[0, 0] == pytest.approx(C.MPU6050_ACCEL_RANGE_G)
    assert out[0, 3] == pytest.approx(C.MPU6050_GYRO_RANGE_DPS)


def test_raw_count_roundtrip_is_within_one_lsb():
    rng = np.random.default_rng(0)
    window = np.empty((C.WINDOW, C.N_CHANNELS), dtype=np.float32)
    window[:, :3] = rng.uniform(-4, 4, size=(C.WINDOW, 3))
    window[:, 3:] = rng.uniform(-500, 500, size=(C.WINDOW, 3))

    counts = to_raw_counts(window)
    back = np.empty_like(window)
    back[:, :3] = counts[:, :3] / ACCEL_LSB_PER_G
    back[:, 3:] = counts[:, 3:] / GYRO_LSB_PER_DPS

    assert np.abs(back[:, :3] - window[:, :3]).max() < 1.0 / ACCEL_LSB_PER_G
    assert np.abs(back[:, 3:] - window[:, 3:]).max() < 1.0 / GYRO_LSB_PER_DPS


# ------------------------------------------------------------- decimation
def test_downsample_ratio_and_antialiasing():
    n = 2000
    t = np.arange(n) / C.FS_RAW_HZ
    # 5 Hz signal survives, 80 Hz (above the 25 Hz Nyquist of 50 Hz) must not
    signal = np.zeros((n, C.N_CHANNELS), dtype=np.float32)
    signal[:, 0] = np.sin(2 * np.pi * 5 * t)
    signal[:, 1] = np.sin(2 * np.pi * 80 * t)

    out = P.downsample(signal)
    assert out.shape[0] == pytest.approx(n / C.DECIMATION, abs=1)

    core = out[20:-20]
    assert core[:, 0].std() > 0.5      # 5 Hz preserved
    assert core[:, 1].std() < 0.05     # 80 Hz rejected, so it cannot alias


# ------------------------------------------------------ window extraction
def _synthetic(n: int, accel_noise: float, gyro_noise: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    sig = np.zeros((n, C.N_CHANNELS), dtype=np.float32)
    sig[:, 2] = 1.0
    sig += rng.normal(0, 1, size=sig.shape).astype(np.float32) * np.array(
        [accel_noise] * 3 + [gyro_noise] * 3, dtype=np.float32
    )
    return sig


def test_sliding_windows_have_the_configured_shape_and_stride():
    sig = _synthetic(400, 0.3, 20.0)
    ws = P.sliding_windows(sig, C.STRIDE)
    assert all(w.shape == (C.WINDOW, C.N_CHANNELS) for w in ws)
    assert len(ws) == (400 - C.WINDOW) // C.STRIDE + 1


def test_sitting_extractor_only_returns_the_still_segment():
    moving = _synthetic(200, 0.4, 40.0, seed=1)
    still = _synthetic(300, 0.005, 0.1, seed=2)
    trial = np.concatenate([moving, still, moving])

    ws = P.sitting_windows(trial)
    assert ws, "expected at least one sitting window"
    for w in ws:
        # every returned window must itself be quasi-stationary
        assert np.linalg.norm(w[:, :3], axis=1).std() < C.SIT_STATIONARY_STD_G


def test_sitting_extractor_rejects_a_trial_that_is_never_still():
    assert P.sitting_windows(_synthetic(600, 0.5, 50.0, seed=3)) == []


def test_falling_window_is_centred_on_the_impact_peak():
    sig = _synthetic(400, 0.02, 1.0, seed=4)
    sig[250, 0] = 12.0                       # impact
    ws = P.falling_windows(sig)
    assert len(ws) == 1
    peak_in_window = int(np.argmax(np.linalg.norm(ws[0][:, :3], axis=1)))
    assert peak_in_window == pytest.approx(C.WINDOW // 2, abs=2)


def test_falling_window_handles_a_peak_near_the_end_of_the_trial():
    sig = _synthetic(140, 0.02, 1.0, seed=5)
    sig[138, 1] = 9.0
    ws = P.falling_windows(sig)
    assert len(ws) == 1 and ws[0].shape == (C.WINDOW, C.N_CHANNELS)


def test_walking_extractor_trims_transients():
    sig = _synthetic(600, 0.3, 30.0, seed=6)
    trim = int(C.WALK_TRIM_S * C.FS_HZ)
    ws = P.walking_windows(sig)
    expected = (600 - 2 * trim - C.WINDOW) // C.STRIDE + 1
    assert len(ws) == expected


# ------------------------------------------------------------ split hygiene
def test_splits_are_disjoint_and_cover_every_subject():
    train, val, test = set(C.TRAIN_SUBJECTS), set(C.VAL_SUBJECTS), set(C.TEST_SUBJECTS)
    assert not (train & val) and not (train & test) and not (val & test)
    assert train | val | test == set(C.SA_SUBJECTS) | set(C.SE_SUBJECTS)
    # falls only exist for SA subjects (and SE06), so the test split needs SA
    assert any(s.startswith("SA") for s in test)


def test_window_length_matches_the_documented_duration():
    assert C.WINDOW / C.FS_HZ == pytest.approx(2.56)
    assert C.STRIDE / C.FS_HZ == pytest.approx(1.28)


# ------------------------------------------------------- firmware codegen
def test_c_float_literals_are_valid_cpp():
    # `2048f` does not compile; `2048.0f` does. This bit us once already.
    assert cfloat(2048) == "2048.0f"
    assert cfloat(0.5) == "0.5f"
    assert cfloat(16.384) == "16.384f"
    for text in (cfloat(v) for v in (0, 1, -1, 1e-8, 12345.678)):
        assert text.endswith("f")
        assert "." in text or "e" in text
