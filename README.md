# On-device Human Activity Recognition on ESP32 (AIoT course project)

An ESP32 reads a 6-axis MPU6050 and decides whether the wearer is sitting,
walking or falling. Nothing is sent anywhere. The optimisation technique from the
AI on Edge chapter that I applied is post-training full-integer (int8)
quantization, and every accuracy, latency and memory number quoted below can be
regenerated from this repository.

```
MPU6050 (I2C, ±16 g, ±2000 °/s, DLPF 21 Hz)
   → 50 Hz sampling → 128-sample ring buffer (2.56 s)
   → per-channel standardisation → int8 quantisation
   → TFLite Micro CNN on the ESP32 (inference every 1.28 s)
   → class + confidence on serial, one LED per class
```

The demo does not cheat. Wokwi's automation scenarios push real held-out sensor
windows into the *simulated MPU6050*, and the firmware then reads them back over
I2C the same way it would read a physical part. No class name is hard coded
anywhere in the sketch.

**Report:** [`docs/report.pdf`](docs/report.pdf) ·
**Demo video:** [`docs/demo.mp4`](docs/demo.mp4)

---

## 1. Results

Accuracy comes from a subject-wise test split: 7 subjects the model never saw,
2 899 windows. Latency and RAM are a different story. Those were measured on the
ESP32 itself, because the desktop interpreter turned out to be a poor proxy. The
firmware embeds all six candidates, times each one and prints the results, which
land in `results/wokwi_modelzoo.log` and get parsed into
`results/ondevice_metrics.json`.

| variant | precision | params | .tflite | macro-F1 | FALLING recall | ESP32 latency | ESP32 arena |
|---|---|---:|---:|---:|---:|---:|---:|
| small | float32 | 811 | 6 632 B | 0.9327 | 0.984 | 413.3 ms | 8 504 B |
| **small** | **int8** | **811** | **5 520 B** | **0.9312** | **0.981** | **94.4 ms** | **3 484 B** |
| medium | float32 | 2 643 | 13 984 B | 0.9292 | 0.981 | 1 139.4 ms | 13 624 B |
| medium | int8 | 2 643 | 8 240 B | 0.9308 | 0.984 | 221.5 ms | 5 020 B |
| large | float32 | 9 379 | 40 904 B | 0.9318 | 0.987 | 3 534.4 ms ✗ | 25 912 B |
| large | int8 | 9 379 | 16 680 B | 0.9327 | 0.987 | 618.2 ms | 8 668 B |

✗ marks a model slower than the 1 280 ms decision interval. It cannot keep up.

What quantization does to each variant:

| variant | .tflite | ESP32 arena | ESP32 latency | macro-F1 |
|---|---|---|---|---|
| small | 6 632 → 5 520 B (16.8 %) | 8 504 → 3 484 B (2.44×) | 413 → 94 ms (4.38×) | 0.9327 → 0.9312 |
| medium | 13 984 → 8 240 B (41.1 %) | 13 624 → 5 020 B (2.71×) | 1 139 → 222 ms (5.14×) | 0.9292 → 0.9308 |
| large | 40 904 → 16 680 B (59.2 %) | 25 912 → 8 668 B (2.99×) | 3 534 → 618 ms (5.72×) | 0.9318 → 0.9327 |

Accuracy moves by somewhere between −0.0015 and +0.0016 macro-F1, which is noise.
Inference gets 4.4 to 5.7 times faster. The arena shrinks by 2.4 to 3.0 times.

Look at the `.tflite` column on its own and you would conclude that quantization
is worth about 17 % on the model I actually deployed. That conclusion is wrong,
and it is wrong in a way you cannot see without a device. This mismatch is the
main Pareto argument of the project.

Firmware footprint on `esp32:esp32:esp32`
([`results/firmware_sizes.json`](results/firmware_sizes.json)):

| build | flash | static RAM |
|---|---:|---:|
| live model = int8 | 382 756 B | 40 460 B |
| live model = float32 | 383 880 B | 45 580 B |

Evidence that it works, all of it sitting in `results/wokwi_*.log`:

* The built-in self test feeds 9 held-out windows to the device as raw MPU6050
  register values. All 9 come back correct, and that holds for each of the six
  candidates.
* The three scenario replays reached the class they were supposed to: SITTING at
  p = 0.844, WALKING at p = 0.996, FALLING at p = 0.996.
* Not one sample deadline was missed once sampling moved to its own core. See §4.

---

## 2. Repository layout

```
training/            dataset construction, training, quantization, evaluation, codegen
  config.py            every constant the report quotes
  prepare_data.py      SisFall → 50 Hz windows → subject-wise splits
  models.py            the CNN (and the abandoned Conv1D version)
  train.py             float32 baselines, one per width variant
  quantize.py          float32 + int8 TFLite conversion and evaluation
  evaluate.py          metrics.csv, Pareto plots, confusion matrices
  export_firmware.py   model byte arrays + norm constants + on-device self test
  export_wokwi.py      held-out windows → Wokwi automation scenarios
  export_model_zoo.py  all six candidates in one header, for the device benchmark
tools/
  download_dataset.py  fetch + verify the SisFall archive
  inspect_tflite.py    operator set and quantization parameters of a model
  compare_topologies.py evidence for the Conv1D → Conv2D design change
  measure_firmware.py  builds 4 configurations, records flash/RAM
  run_simulation.py    drives Wokwi, collects the on-device measurements
  parse_logs.py        re-parse existing logs without re-simulating
firmware/har_esp32/  Arduino sketch, generated model_data.h, libraries.txt
diagram.json         Wokwi circuit (ESP32 + MPU6050 + 3 LEDs)
wokwi.toml           points Wokwi CLI at the compiled firmware
wokwi/scenarios/     sitting.yaml / walking.yaml / falling.yaml
models/              .keras checkpoints and .tflite artifacts
results/             metrics, figures, firmware sizes, simulation logs
tests/               pytest unit tests for the sensing/label pipeline
docs/                report.pdf and demo.mp4
```

---

## 3. Reproducing everything

### 3.1 Prerequisites

* Python 3.11. I used 3.11.9 on Windows 11.
* `arduino-cli` 1.5 or newer with the `esp32` core, but only if you want to build
  the firmware.
* A Wokwi CLI token, only if you want to run the simulation headlessly. Free
  accounts come with 50 simulation minutes a month, which is plenty.

### 3.2 Python environment

```bash
python -m venv .venv
.venv/Scripts/activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Windows users need the Microsoft Visual C++ redistributable as well, otherwise
TensorFlow fails to import with a missing DLL error:
`winget install Microsoft.VCRedist.2015+.x64`.

### 3.3 Data to model to firmware headers

```bash
python tools/download_dataset.py          # ~222 MB, sha256 verified
python -m training.prepare_data           # writes data/processed/har_dataset.npz
python -m training.train                  # trains small / medium / large
python -m training.quantize               # float32 + int8 .tflite + test metrics
python -m training.evaluate               # metrics.csv + figures
python -m training.export_firmware small  # firmware/har_esp32/model_data.h
python -m training.export_model_zoo       # firmware/har_esp32/model_zoo.h
python -m training.export_wokwi           # wokwi/scenarios/*.yaml
python -m pytest -q                       # 14 tests
```

Seeds are fixed at `config.SEED = 1337` and the split is pinned to subject IDs
rather than drawn at random, so a clean run lands on the numbers above.

### 3.4 Build the firmware

```bash
arduino-cli core install esp32:esp32 \
  --additional-urls https://espressif.github.io/arduino-esp32/package_esp32_index.json
arduino-cli lib install "Adafruit MPU6050" "Chirale_TensorFLowLite"

python tools/measure_firmware.py    # builds all 3 configurations + size report
```

### 3.5 Run it

**Option A: wokwi.com, no local toolchain needed.** Start a new ESP32 project.
Paste in `firmware/har_esp32/har_esp32.ino` and `model_data.h`, add the two
libraries listed in `firmware/har_esp32/libraries.txt`, then swap the project's
`diagram.json` for the one here. Press play. The serial monitor walks through the
self test and the benchmark before it starts printing live predictions.

**Option B: headless, through the Wokwi CLI.**

```bash
export WOKWI_CLI_TOKEN=wok_...            # PowerShell: $env:WOKWI_CLI_TOKEN='wok_...'
python tools/run_simulation.py            # boot + model zoo + 3 scenarios + single-core
python tools/parse_logs.py                # re-parse existing logs without re-simulating
```

Budget about 20 minutes of wall time for `run_simulation.py`. The simulator runs
slower than real time, so most of that is waiting. It costs roughly 6 minutes of
Wokwi CI quota. Single runs, if you only want one thing:

```bash
wokwi-cli . --timeout 60000 --serial-log-file results/wokwi_boot.log
wokwi-cli . --elf firmware/build/deploy_int8/har_esp32.ino.elf \
            --scenario wokwi/scenarios/falling.yaml --timeout 60000
```

---

## 4. Design decisions worth knowing

### Sampling and inference must not share a core

A single int8 inference costs about 94 ms. A sample is due every 20 ms. My first
version ran the inference inline in `loop()`, which meant the sensor stream just
stopped for 4.7 sample periods every time the model ran. The firmware counts its
own missed deadlines, so the damage was easy to quantify once I looked. Same
walking window in both runs, `results/wokwi_singlecore.log` against
`results/wokwi_boot.log`:

| scheduling | worst sample lateness | samples skipped |
|---|---:|---:|
| inline in `loop()` (abandoned) | 89 094 µs | 24 in 7 windows (~4 per window) |
| sampler task on core 0 (shipped) | **0 µs** | **0** |

I left the broken version buildable behind `-DUSE_SAMPLING_TASK=0`, so anyone can
reproduce the comparison instead of taking my word for it.

### Everything else

| decision | reason |
|---|---|
| 50 Hz sampling | sufficient for fall detection in the literature; 4× decimation of SisFall's 200 Hz |
| 128-sample window (2.56 s) | >2 gait cycles, and the whole free-fall → impact → rest sequence |
| stride 64 (50 % overlap) | a decision every 1.28 s; inference runs on 1 of every 64 sample ticks |
| MPU6050 at ±16 g / ±2000 °/s | matches the SisFall sensor ranges, so no training value is unrepresentable on the device |
| DLPF 21 Hz | sits below the 25 Hz Nyquist limit of 50 Hz sampling, so anti-aliasing happens in hardware |
| ADXL345 + ITG3200 columns | the third SisFall accelerometer (MMA8451Q) clips at 8 g, which falls exceed |
| explicit `Conv2D`, not `Conv1D` | `Conv1D` is lowered into EXPAND_DIMS→CONV_2D→RESHAPE triplets: 16 ops instead of 8, and 2 extra TFLM kernels to link (`python tools/compare_topologies.py`) |
| `MicroMutableOpResolver<5>` | only CONV_2D, MAX_POOL_2D, MEAN, FULLY_CONNECTED, SOFTMAX are registered, instead of `AllOpsResolver` |
| no BatchNorm | keeps the float and int8 graphs directly comparable and the op set minimal |
| int8 input tensor | the ESP32 quantises the window itself, so no float kernels are linked |
| subject-wise split | random window splits leak subjects across splits and inflate accuracy |
| natural test distribution | train/val cap the dominant WALKING class; the test split is left untouched and reported with macro-F1 and per-class recall |
| label rules, not hand labels | SITTING = longest quasi-stationary segment of a sit/stand trial; FALLING = window centred on the impact peak |

---

## 5. Data

Everything here is built on SisFall (Sucerquia, López and Vargas-Bonilla,
*Sensors* 2017, [doi:10.3390/s17010198](https://doi.org/10.3390/s17010198)).
It holds 200 Hz waist-worn IMU recordings from 38 subjects, covering 19 daily
activities and 15 kinds of fall.

The original institutional download link no longer resolves. `tools/download_dataset.py`
therefore pulls the CC BY 4.0 Hugging Face mirror and checks its SHA-256 before
unpacking. The raw recordings are not committed here.

Activity codes I used: `D01, D02, D05, D06` become WALKING, `D07` through `D10`
become SITTING, and `F01` through `F15` become FALLING.

---

## 6. Known limitations

* Wokwi replays recordings. It cannot tell me what happens with a real wearer, a
  sensor taped somewhere else, or a device mounted at a different angle.
* The training recordings come from an ADXL345 and ITG3200 pair at the waist,
  while the target is an MPU6050. That domain gap is invisible in simulation.
* Simulator latency is not a hardware guarantee. I label it as
  simulator-measured everywhere it appears.
* Energy numbers in the report are estimates worked out from datasheet currents
  and the measured duty cycle. I had no power meter.
* The falls were performed by volunteers, and the elderly participants barely
  performed any. Real FALLING recall for that group is probably lower than 0.981.
