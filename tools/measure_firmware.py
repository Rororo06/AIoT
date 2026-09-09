"""Build the firmware in several configurations and record flash / RAM usage.

This is the source of the "before and after optimization" memory numbers in the
report. Three builds are produced:

  deploy_int8     live classifier = int8   model, no benchmark code
  deploy_float32  live classifier = float32 model, no benchmark code
  benchmark       both models embedded, self test + on-device timing (the build
                  that is actually simulated in Wokwi)

Requires arduino-cli with the esp32 core; see README.

Run:  python tools/measure_firmware.py
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKETCH = ROOT / "firmware" / "har_esp32"
BUILD_ROOT = ROOT / "firmware" / "build"
FQBN = "esp32:esp32:esp32"
RESULTS = ROOT / "results" / "firmware_sizes.json"

CONFIGS = {
    # the two builds that isolate the cost of the optimisation
    "deploy_int8": "-DRUN_BENCHMARK=0 -DRUN_SELFTEST=1 -DPRIMARY_FLOAT32=0",
    "deploy_float32": "-DRUN_BENCHMARK=0 -DRUN_SELFTEST=1 -DPRIMARY_FLOAT32=1",
    # the build simulated in Wokwi: self test + on-device int8/float32 timing
    "benchmark": "-DRUN_BENCHMARK=1 -DRUN_SELFTEST=1 -DPRIMARY_FLOAT32=0",
    # the abandoned single-core scheduling, kept measurable on purpose
    "singlecore": "-DRUN_BENCHMARK=0 -DRUN_SELFTEST=0 -DUSE_SAMPLING_TASK=0",
    # all six candidates in one image, for the device-measured Pareto frontier
    "modelzoo": "-DRUN_MODEL_ZOO=1 -DRUN_BENCHMARK=0 -DRUN_SELFTEST=0",
}

FLASH_RE = re.compile(r"Sketch uses (\d+) bytes")
RAM_RE = re.compile(r"Global variables use (\d+) bytes")


def arduino_cli() -> str:
    exe = shutil.which("arduino-cli")
    if exe:
        return exe
    fallback = Path(r"C:\Program Files\Arduino CLI\arduino-cli.exe")
    if fallback.exists():
        return str(fallback)
    sys.exit("arduino-cli not found on PATH")


def build(name: str, flags: str, cli: str) -> dict:
    out_dir = BUILD_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        cli, "compile",
        "--fqbn", FQBN,
        "--output-dir", str(out_dir),
        "--build-property", f"compiler.cpp.extra_flags={flags}",
        str(SKETCH),
    ]
    print(f"\n$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log = proc.stdout + proc.stderr
    if proc.returncode != 0:
        print(log)
        sys.exit(f"build {name} failed")

    flash = FLASH_RE.search(log)
    ram = RAM_RE.search(log)
    binary = out_dir / f"{SKETCH.name}.ino.bin"
    record = {
        "config": name,
        "flags": flags,
        "flash_bytes": int(flash.group(1)) if flash else None,
        "ram_bytes": int(ram.group(1)) if ram else None,
        "bin_bytes": binary.stat().st_size if binary.exists() else None,
    }
    print(f"  flash={record['flash_bytes']} ram={record['ram_bytes']} "
          f"bin={record['bin_bytes']}")
    return record


def main() -> None:
    cli = arduino_cli()
    records = [build(name, flags, cli) for name, flags in CONFIGS.items()]
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(records, indent=2))

    by_name = {r["config"]: r for r in records}
    i8, f32 = by_name["deploy_int8"], by_name["deploy_float32"]
    print("\n--- cost of deploying float32 instead of int8 ---")
    print(f"flash {i8['flash_bytes']} -> {f32['flash_bytes']} B "
          f"({f32['flash_bytes'] - i8['flash_bytes']:+d} B)")
    print(f"RAM   {i8['ram_bytes']} -> {f32['ram_bytes']} B "
          f"({f32['ram_bytes'] - i8['ram_bytes']:+d} B)")
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
