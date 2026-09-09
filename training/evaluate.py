"""Turn results/tflite_metrics.json into the tables and figures used in the report.

Outputs
-------
results/metrics.csv                   one row per (variant, precision)
results/pareto_accuracy_size.png      macro-F1 vs .tflite size
results/pareto_accuracy_latency.png   macro-F1 vs host latency
results/confusion_matrix_<tag>.png    confusion matrix of the deployed model
results/per_class_table.md            per-class precision/recall/F1

Run:  python -m training.evaluate
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from training import config as C

METRICS_JSON = C.RESULTS_DIR / "tflite_metrics.json"
DEVICE_JSON = C.RESULTS_DIR / "ondevice_metrics.json"
ORDER = list(C.MODEL_VARIANTS)
# A decision is due every STRIDE samples; anything slower cannot keep up.
REALTIME_BUDGET_MS = C.STRIDE / C.FS_HZ * 1000


def load_records() -> list[dict]:
    if not METRICS_JSON.exists():
        raise FileNotFoundError(f"{METRICS_JSON} missing - run `python -m training.quantize`")
    return json.loads(METRICS_JSON.read_text())


def load_device() -> dict:
    """Per-candidate measurements taken on the ESP32 itself, if available."""
    if not DEVICE_JSON.exists():
        return {}
    return json.loads(DEVICE_JSON.read_text()).get("model_zoo", {})


def to_frame(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        row = {
            "variant": r["variant"],
            "precision": r["precision"],
            "params": r["params"],
            "tflite_bytes": r["tflite_bytes"],
            "tflite_kib": round(r["tflite_bytes"] / 1024, 2),
            "accuracy": round(r["accuracy"], 4),
            "macro_f1": round(r["macro_f1"], 4),
            "host_latency_ms": round(r["host_latency_ms_mean"], 3),
        }
        for cls in C.CLASS_NAMES:
            row[f"recall_{cls.lower()}"] = round(r["per_class"][cls]["recall"], 4)
        rows.append(row)
    df = pd.DataFrame(rows)

    device = load_device()
    if device:
        keys = df["variant"] + "_" + df["precision"]
        df["device_latency_ms"] = [device.get(k, {}).get("device_latency_ms") for k in keys]
        df["device_arena_bytes"] = [device.get(k, {}).get("arena_bytes") for k in keys]
        df["duty_percent"] = [device.get(k, {}).get("duty_percent") for k in keys]
        df["realtime_feasible"] = [device.get(k, {}).get("realtime_feasible") for k in keys]

    df["variant"] = pd.Categorical(df["variant"], ORDER, ordered=True)
    return df.sort_values(["variant", "precision"]).reset_index(drop=True)


def pareto_front(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Indices that are not dominated (minimise x, maximise y)."""
    keep = []
    for i in range(x.size):
        dominated = np.any((x <= x[i]) & (y >= y[i]) & ((x < x[i]) | (y > y[i])))
        if not dominated:
            keep.append(i)
    return np.asarray(sorted(keep, key=lambda i: x[i]))


def scatter_pareto(df: pd.DataFrame, xcol: str, xlabel: str, path, title: str,
                   budget: float | None = None, logx: bool = False) -> None:
    df = df.dropna(subset=[xcol])
    if df.empty:
        print(f"skip {path.name}: no data for {xcol}")
        return
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    markers = {"float32": "o", "int8": "s"}
    colors = {"small": "#1f77b4", "medium": "#ff7f0e", "large": "#2ca02c"}
    for _, row in df.iterrows():
        ax.scatter(
            row[xcol], row["macro_f1"],
            marker=markers[row["precision"]], s=110,
            facecolor=colors[row["variant"]] if row["precision"] == "int8" else "none",
            edgecolor=colors[row["variant"]], linewidths=1.8, zorder=3,
        )
        # int8 labels above, float32 labels below, so the two members of a pair
        # never collide on the log axis
        offset = (8, 5) if row["precision"] == "int8" else (8, -12)
        ax.annotate(
            f"{row['variant']}/{row['precision']}",
            (row[xcol], row["macro_f1"]),
            textcoords="offset points", xytext=offset, fontsize=8,
        )

    x = df[xcol].to_numpy(float)
    y = df["macro_f1"].to_numpy(float)
    front = pareto_front(x, y)
    ax.plot(x[front], y[front], "--", color="0.35", lw=1.2, zorder=2,
            label="Pareto frontier")

    # connect each float32 model to its int8 counterpart
    for variant in ORDER:
        sub = df[df["variant"] == variant]
        if len(sub) == 2:
            ax.plot(sub[xcol], sub["macro_f1"], ":", color=colors[variant], lw=1.0, zorder=1)

    if budget is not None:
        ax.axvline(budget, color="crimson", lw=1.4, ls="-.",
                   label=f"real-time budget {budget:.0f} ms")
        ax.axvspan(budget, max(ax.get_xlim()[1], budget * 1.05), color="crimson", alpha=0.06)

    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("macro-F1 (subject-wise test split)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


def confusion_figure(cm: np.ndarray, path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]*100:.1f}%",
                    ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(C.N_CLASSES), C.CLASS_NAMES, fontsize=8)
    ax.set_yticks(range(C.N_CLASSES), C.CLASS_NAMES, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records()
    df = to_frame(records)

    csv_path = C.RESULTS_DIR / "metrics.csv"
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False))
    print(f"\nwrote {csv_path}")

    scatter_pareto(
        df, "tflite_kib", "model size (KiB, .tflite)",
        C.RESULTS_DIR / "pareto_accuracy_size.png",
        "Accuracy vs size trade-off",
    )
    scatter_pareto(
        df, "host_latency_ms", "host interpreter latency (ms/window)",
        C.RESULTS_DIR / "pareto_accuracy_latency.png",
        "Accuracy vs latency trade-off (host reference)",
    )
    if "device_latency_ms" in df:
        scatter_pareto(
            df, "device_latency_ms", "ESP32 inference latency (ms/window, log scale)",
            C.RESULTS_DIR / "pareto_device_latency.png",
            "Accuracy vs latency measured on the ESP32",
            budget=REALTIME_BUDGET_MS, logx=True,
        )
        scatter_pareto(
            df, "device_arena_bytes", "TFLite Micro tensor arena on the ESP32 (bytes)",
            C.RESULTS_DIR / "pareto_device_arena.png",
            "Accuracy vs RAM measured on the ESP32",
        )

    for r in records:
        tag = f"{r['variant']}_{r['precision']}"
        confusion_figure(
            np.asarray(r["confusion_matrix"]),
            C.RESULTS_DIR / f"confusion_matrix_{tag}.png",
            f"{r['variant']} / {r['precision']}",
        )

    # per-class markdown table
    lines = ["| model | class | precision | recall | F1 | support |",
             "|---|---|---:|---:|---:|---:|"]
    for r in records:
        for cls in C.CLASS_NAMES:
            p = r["per_class"][cls]
            lines.append(
                f"| {r['variant']}/{r['precision']} | {cls} | "
                f"{p['precision']:.3f} | {p['recall']:.3f} | {p['f1']:.3f} | {p['support']} |"
            )
    out = C.RESULTS_DIR / "per_class_table.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")

    # size / F1 delta caused by quantization
    print("\nquantization effect:")
    lines = ["| variant | .tflite | arena (ESP32) | latency (ESP32) | macro-F1 |",
             "|---|---|---|---|---|"]
    for variant in ORDER:
        f32 = df[(df.variant == variant) & (df.precision == "float32")]
        i8 = df[(df.variant == variant) & (df.precision == "int8")]
        if f32.empty or i8.empty:
            continue
        f32, i8 = f32.iloc[0], i8.iloc[0]
        print(
            f"  {variant:6s} size {f32.tflite_bytes:6d} -> {i8.tflite_bytes:6d} B "
            f"({100*(1-i8.tflite_bytes/f32.tflite_bytes):5.1f}% smaller)   "
            f"macro-F1 {f32.macro_f1:.4f} -> {i8.macro_f1:.4f} "
            f"({i8.macro_f1-f32.macro_f1:+.4f})"
        )
        cell_arena = cell_lat = "n/a"
        if df.get("device_latency_ms") is not None and not pd.isna(f32.get("device_latency_ms")):
            print(f"         ESP32 latency {f32.device_latency_ms:8.1f} -> "
                  f"{i8.device_latency_ms:8.1f} ms "
                  f"({f32.device_latency_ms/i8.device_latency_ms:.2f}x faster)   "
                  f"arena {int(f32.device_arena_bytes):6d} -> {int(i8.device_arena_bytes):6d} B "
                  f"({f32.device_arena_bytes/i8.device_arena_bytes:.2f}x smaller)")
            cell_arena = (f"{int(f32.device_arena_bytes)} → {int(i8.device_arena_bytes)} B "
                          f"({f32.device_arena_bytes/i8.device_arena_bytes:.2f}×)")
            cell_lat = (f"{f32.device_latency_ms:.0f} → {i8.device_latency_ms:.0f} ms "
                        f"({f32.device_latency_ms/i8.device_latency_ms:.2f}×)")
        lines.append(
            f"| {variant} | {f32.tflite_bytes} → {i8.tflite_bytes} B "
            f"({100*(1-i8.tflite_bytes/f32.tflite_bytes):.1f}%) | {cell_arena} | "
            f"{cell_lat} | {f32.macro_f1:.4f} → {i8.macro_f1:.4f} |"
        )
    table = C.RESULTS_DIR / "quantization_effect.md"
    table.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {table}")

    if "realtime_feasible" in df:
        infeasible = df[df.realtime_feasible == False]  # noqa: E712
        if not infeasible.empty:
            print(f"\nnot real-time capable (inference > {REALTIME_BUDGET_MS:.0f} ms):")
            for _, row in infeasible.iterrows():
                print(f"  {row.variant}/{row.precision}: {row.device_latency_ms:.0f} ms "
                      f"= {row.duty_percent:.0f}% of the decision interval")


if __name__ == "__main__":
    main()
