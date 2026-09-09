"""Run the Wokwi simulation headlessly and collect the on-device measurements.

Four runs:
  boot        benchmark build - self test, int8 vs float32 latency, arena usage
  scenario x3 sitting / walking / falling windows replayed through the simulated
              MPU6050; each run fails if the firmware never reports that class
  singlecore  the abandoned scheduling, to quantify how many samples it loses

Writes results/wokwi_*.log and results/ondevice_metrics.json.

Requires WOKWI_CLI_TOKEN (free accounts include 50 simulation minutes/month) and
firmware built by tools/measure_firmware.py.

Run:  python tools/run_simulation.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BUILD = ROOT / "firmware" / "build"
SCENARIOS = ("sitting", "walking", "falling")
BOOT_TIMEOUT_MS = 60_000
SCENARIO_TIMEOUT_MS = 60_000

BENCH_RE = re.compile(
    r"BENCH (\S+)\s+model=\s*(\d+) B arena=\s*(\d+) B mean=\s*([\d.]+) us "
    r"p95=\s*(\d+) us min=\s*(\d+) us max=\s*(\d+) us duty=([\d.]+)%"
)
PRED_RE = re.compile(
    r"PRED (\w+)\s+p=([\d.]+) .*t=(\d+) us w=(\d+) late=(\d+) worst=(\d+) us skipped=(\d+)"
)
SELFTEST_RE = re.compile(r"SELFTEST (\d+)/(\d+) correct")
ARENA_RE = re.compile(r"\[(\w+)\] arena_used=(\d+) B of (\d+) B")
ZOO_RE = re.compile(
    r"ZOO (\w+) (\w+) (\d+) (\d+) ([\d.]+) (\d+) (\d+)/(\d+)"
)


def wokwi_cli() -> str:
    exe = shutil.which("wokwi-cli")
    if exe:
        return exe
    local = Path(os.environ.get("LOCALAPPDATA", "")) / "wokwi-cli" / "wokwi-cli.exe"
    if local.exists():
        return str(local)
    sys.exit("wokwi-cli not found; see https://docs.wokwi.com/wokwi-ci/cli-installation")


def run(cli: str, args: list[str], log: Path) -> tuple[int, str]:
    cmd = [cli, str(ROOT), "--serial-log-file", str(log), *args]
    print(f"\n$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout[-2000:])
        sys.stdout.write(proc.stderr[-2000:])
    return proc.returncode, log.read_text(errors="ignore") if log.exists() else ""


def last_pred(text: str) -> dict | None:
    matches = PRED_RE.findall(text)
    if not matches:
        return None
    cls, p, t, w, late, worst, skipped = matches[-1]
    return {
        "class": cls,
        "confidence": float(p),
        "latency_us": int(t),
        "windows": int(w) + 1,
        "deadline_misses": int(late),
        "worst_late_us": int(worst),
        "samples_skipped": int(skipped),
    }


def main() -> None:
    if not os.environ.get("WOKWI_CLI_TOKEN"):
        sys.exit("set WOKWI_CLI_TOKEN first (https://wokwi.com/dashboard/ci)")
    bench_bin = BUILD / "benchmark" / "har_esp32.ino.bin"
    if not bench_bin.exists():
        sys.exit(f"{bench_bin} missing - run `python tools/measure_firmware.py`")
    RESULTS.mkdir(parents=True, exist_ok=True)
    cli = wokwi_cli()
    summary: dict = {}

    # ---------------------------------------------------------------- boot ---
    print("=== boot run: self test + on-device int8/float32 benchmark ===")
    _, text = run(cli, ["--timeout", str(BOOT_TIMEOUT_MS)], RESULTS / "wokwi_boot.log")

    summary["arena_used_bytes"] = {
        tag: {"used": int(used), "allocated": int(alloc)}
        for tag, used, alloc in ARENA_RE.findall(text)
    }
    summary["benchmark"] = {
        tag: {
            "model_bytes": int(model),
            "arena_bytes": int(arena),
            "mean_us": float(mean),
            "p95_us": int(p95),
            "min_us": int(mn),
            "max_us": int(mx),
            "duty_percent": float(duty),
        }
        for tag, model, arena, mean, p95, mn, mx, duty in BENCH_RE.findall(text)
    }
    st = SELFTEST_RE.search(text)
    summary["selftest"] = {"passed": int(st.group(1)), "total": int(st.group(2))} if st else None
    summary["live_dualcore"] = last_pred(text)

    for tag, b in summary["benchmark"].items():
        print(f"  {tag:8s} mean={b['mean_us']/1000:7.2f} ms  arena={b['arena_bytes']} B")
    if summary["selftest"]:
        print(f"  selftest {summary['selftest']['passed']}/{summary['selftest']['total']}")

    b = summary["benchmark"]
    if "int8" in b and "float32" in b:
        summary["speedup_int8_vs_float32"] = round(
            b["float32"]["mean_us"] / b["int8"]["mean_us"], 3)
        summary["arena_ratio_float32_vs_int8"] = round(
            b["float32"]["arena_bytes"] / b["int8"]["arena_bytes"], 3)
        print(f"  int8 is {summary['speedup_int8_vs_float32']}x faster and uses "
              f"{summary['arena_ratio_float32_vs_int8']}x less arena")

    # ------------------------------------------------------------ model zoo ---
    zoo_elf = BUILD / "modelzoo" / "har_esp32.ino.elf"
    if zoo_elf.exists():
        print("\n=== model zoo: every candidate measured on the ESP32 ===")
        _, text = run(
            cli,
            ["--elf", str(zoo_elf), "--timeout", "240000",
             "--expect-text", "ZOO DONE"],
            RESULTS / "wokwi_modelzoo.log",
        )
        zoo = {}
        for variant, precision, nbytes, arena, mean, p95, ok, total in ZOO_RE.findall(text):
            zoo[f"{variant}_{precision}"] = {
                "variant": variant,
                "precision": precision,
                "tflite_bytes": int(nbytes),
                "arena_bytes": int(arena),
                "device_latency_ms": round(float(mean) / 1000, 3),
                "device_p95_ms": round(int(p95) / 1000, 3),
                "selftest_correct": int(ok),
                "selftest_total": int(total),
                # a decision is due every HAR_STRIDE samples = 1.28 s
                "duty_percent": round(float(mean) / 1000 / 1280 * 100, 2),
                "realtime_feasible": float(mean) / 1000 < 1280,
            }
            print(f"  {variant:6s} {precision:8s} {float(mean)/1000:9.1f} ms  "
                  f"arena={arena:>6s} B  selftest {ok}/{total}"
                  f"{'' if float(mean)/1000 < 1280 else '   <-- too slow for 1.28 s'}")
        summary["model_zoo"] = zoo

    # ----------------------------------------------------------- scenarios ---
    deploy_elf = BUILD / "deploy_int8" / "har_esp32.ino.elf"
    summary["scenarios"] = {}
    failures = []
    for name in SCENARIOS:
        print(f"\n=== scenario: {name} ===")
        code, text = run(
            cli,
            ["--elf", str(deploy_elf),
             "--scenario", f"wokwi/scenarios/{name}.yaml",
             "--timeout", str(SCENARIO_TIMEOUT_MS)],
            RESULTS / f"wokwi_{name}.log",
        )
        record = last_pred(text) or {}
        record["passed"] = code == 0
        summary["scenarios"][name] = record
        print(f"  expected {name.upper():8s} -> {'PASS' if code == 0 else 'FAIL'}"
              f"  last={record.get('class')} p={record.get('confidence')}")
        if code != 0:
            failures.append(name)

    # ------------------------------------------- scheduling comparison ------
    single_elf = BUILD / "singlecore" / "har_esp32.ino.elf"
    if single_elf.exists():
        print("\n=== abandoned single-core scheduling (same walking window) ===")
        _, text = run(
            cli,
            ["--elf", str(single_elf),
             "--scenario", "wokwi/scenarios/walking.yaml",
             "--timeout", str(SCENARIO_TIMEOUT_MS)],
            RESULTS / "wokwi_singlecore.log",
        )
        summary["live_singlecore"] = last_pred(text)
        single, dual = summary["live_singlecore"], summary["live_dualcore"]
        if single and dual:
            print(f"  single core: {single['samples_skipped']} samples skipped, "
                  f"worst lateness {single['worst_late_us']} us")
            print(f"  dual core  : {dual['samples_skipped']} samples skipped, "
                  f"worst lateness {dual['worst_late_us']} us")

    out = RESULTS / "ondevice_metrics.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")
    if failures:
        sys.exit(f"failed scenarios: {', '.join(failures)}")
    print("all scenarios reached their expected class")


if __name__ == "__main__":
    main()
