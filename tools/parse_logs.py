"""Re-parse existing results/wokwi_*.log files into results/ondevice_metrics.json.

Useful when the simulation has already been run (Wokwi CI minutes are limited)
and only the parsing needs to be repeated. `tools/run_simulation.py` performs the
same parsing while it runs.

Run:  python tools/parse_logs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from run_simulation import ZOO_RE  # noqa: E402

RESULTS = ROOT / "results"
OUT = RESULTS / "ondevice_metrics.json"
DECISION_INTERVAL_MS = 1280.0


def main() -> None:
    summary = json.loads(OUT.read_text()) if OUT.exists() else {}

    log = RESULTS / "wokwi_modelzoo.log"
    if not log.exists():
        sys.exit(f"{log} not found - run tools/run_simulation.py")

    text = log.read_text(errors="ignore")
    zoo = {}
    for variant, precision, nbytes, arena, mean, p95, ok, total in ZOO_RE.findall(text):
        latency_ms = float(mean) / 1000
        zoo[f"{variant}_{precision}"] = {
            "variant": variant,
            "precision": precision,
            "tflite_bytes": int(nbytes),
            "arena_bytes": int(arena),
            "device_latency_ms": round(latency_ms, 3),
            "device_p95_ms": round(int(p95) / 1000, 3),
            "selftest_correct": int(ok),
            "selftest_total": int(total),
            "duty_percent": round(latency_ms / DECISION_INTERVAL_MS * 100, 2),
            "realtime_feasible": latency_ms < DECISION_INTERVAL_MS,
        }
    if not zoo:
        sys.exit("no ZOO lines found in the log")

    summary["model_zoo"] = zoo
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"wrote {OUT} with {len(zoo)} model-zoo entries")
    for key, v in zoo.items():
        flag = "" if v["realtime_feasible"] else "  <-- exceeds the decision interval"
        print(f"  {key:16s} {v['device_latency_ms']:9.1f} ms  arena={v['arena_bytes']:6d} B"
              f"  selftest {v['selftest_correct']}/{v['selftest_total']}{flag}")


if __name__ == "__main__":
    main()
