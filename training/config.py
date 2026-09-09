"""Central configuration for the AIoT human activity recognition project.

Every number that influences the sensing pipeline, the dataset construction or
the model is declared here so that the report and the firmware can quote the
exact same values.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "extracted" / "SisFall_dataset"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIRMWARE_DIR = ROOT / "firmware" / "har_esp32"
WOKWI_SCENARIO_DIR = ROOT / "wokwi" / "scenarios"

# --------------------------------------------------------------------------- #
# Sensing pipeline
# --------------------------------------------------------------------------- #
# SisFall was recorded at 200 Hz. The deployed ESP32 samples the MPU6050 at
# 50 Hz, which the literature reports as sufficient for fall detection, so the
# training data is decimated by 4 with an anti-aliasing filter.
FS_RAW_HZ = 200
FS_HZ = 50
DECIMATION = FS_RAW_HZ // FS_HZ

# 128 samples @ 50 Hz = 2.56 s. Long enough to contain >2 gait cycles and the
# complete free-fall + impact + rest sequence of a fall.
WINDOW = 128
# 50 % overlap -> a new inference every 1.28 s.
STRIDE = WINDOW // 2

CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz")
N_CHANNELS = len(CHANNELS)

# SisFall column layout (9 columns, see dataset Readme.txt):
#   0..2 ADXL345  accelerometer  13 bit, +-16 g
#   3..5 ITG3200  gyroscope      16 bit, +-2000 deg/s
#   6..8 MMA8451Q accelerometer  14 bit, +-8 g   (unused)
# We use ADXL345 + ITG3200 because their ranges match the MPU6050
# configuration used on the device (see below); the MMA8451Q would clip at 8 g.
COL_ACCEL = (0, 1, 2)
COL_GYRO = (3, 4, 5)
ADXL345_G_PER_LSB = (2 * 16) / (2 ** 13)          # 0.00390625 g/LSB
ITG3200_DPS_PER_LSB = (2 * 2000) / (2 ** 16)      # 0.061035 deg/s/LSB

# MPU6050 configuration used by the firmware. Chosen to cover the dynamic range
# of the training data without clipping.
MPU6050_ACCEL_RANGE_G = 16
MPU6050_GYRO_RANGE_DPS = 2000

# --------------------------------------------------------------------------- #
# Classes
# --------------------------------------------------------------------------- #
CLASS_NAMES = ("SITTING", "WALKING", "FALLING")
N_CLASSES = len(CLASS_NAMES)
LABEL_SITTING, LABEL_WALKING, LABEL_FALLING = 0, 1, 2

# SisFall activity codes mapped onto the three target classes.
WALKING_CODES = ("D01", "D02", "D05", "D06")   # level walking + stairs
SITTING_CODES = ("D07", "D08", "D09", "D10")   # sit down, hold, stand up
FALL_CODES = tuple(f"F{i:02d}" for i in range(1, 16))

# The sit/stand trials contain transitions. Only the longest quasi-stationary
# segment of each trial is labelled SITTING; the detector below is a label
# definition rule, not a classifier feature.
SIT_STATIONARY_WIN = 25          # 0.5 s rolling window @50 Hz
SIT_STATIONARY_STD_G = 0.06      # max std of |a| inside a "still" segment
SIT_STRIDE = 32                  # denser stride, SITTING is the rarest ADL

# Walking trials are trimmed to remove start/stop transients.
WALK_TRIM_S = 2.0

# --------------------------------------------------------------------------- #
# Subject-wise split (no subject appears in more than one split).
# Falls were only recorded for the SA group and for SE06, so SE06 stays in
# train and the test split contains SA subjects to keep all three classes.
# --------------------------------------------------------------------------- #
SA_SUBJECTS = tuple(f"SA{i:02d}" for i in range(1, 24))
SE_SUBJECTS = tuple(f"SE{i:02d}" for i in range(1, 16))

TEST_SUBJECTS = ("SA03", "SA08", "SA13", "SA18", "SA23", "SE04", "SE12")
VAL_SUBJECTS = ("SA05", "SA10", "SA15", "SA20", "SE08")
TRAIN_SUBJECTS = tuple(
    s for s in SA_SUBJECTS + SE_SUBJECTS
    if s not in TEST_SUBJECTS and s not in VAL_SUBJECTS
)

# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
SEED = 1337
BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5

# Model candidates: identical topology, different widths. Used to trace the
# accuracy / size / latency Pareto frontier.
MODEL_VARIANTS = {
    "small": dict(filters=(8, 16), dense=8),
    "medium": dict(filters=(16, 32), dense=16),
    "large": dict(filters=(32, 64), dense=32),
}

# --------------------------------------------------------------------------- #
# Quantization
# --------------------------------------------------------------------------- #
REPRESENTATIVE_SAMPLES = 300     # windows drawn from the TRAIN split only

# --------------------------------------------------------------------------- #
# Deployment
# --------------------------------------------------------------------------- #
# Number of held-out test windows embedded in the firmware as a self test.
SELFTEST_WINDOWS_PER_CLASS = 3
# Windows exported as Wokwi automation scenarios (one per class).
SCENARIO_CLASSES = CLASS_NAMES
