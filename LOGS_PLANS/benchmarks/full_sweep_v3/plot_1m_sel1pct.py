"""Two slide-ready plots for the 1M / 1% selectivity workload.

Inputs:
  sift1m-full-n1000000-b100.csv  (selectivity = 1/100 = 1%, all 8 cells)

Outputs:
  recall_1M_sel1pct.png
  latency_1M_sel1pct.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV = HERE / "sift1m-full-n1000000-b100.csv"

CELL_LABELS = {
    "000": ("baseline", "#8c8c8c"),
    "100": ("adaptive", "#1f77b4"),
    "010": ("bloom", "#ff7f0e"),
    "001": ("synopsis", "#2ca02c"),
    "110": ("adapt+bloom", "#9467bd"),
    "101": ("adapt+syn", "#e377c2"),
    "011": ("bloom+syn", "#d62728"),
    "111": ("all three", "#000000"),
}
CELL_ORDER = list(CELL_LABELS.keys())


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV, dtype={"cell": str})
    df["cell"] = df["cell"].str.zfill(3)
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("cell")
        .agg(
            recall=("recall_at_k", "mean"),
            latency_mean=("latency_ms", "mean"),
            latency_p99=("latency_ms", lambda x: np.quantile(x, 0.99)),
        )
        .reindex(CELL_ORDER)
    )


def plot_recall(agg: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = [CELL_LABELS[c][1] for c in agg.index]
    bars = ax.bar(range(len(agg)), agg["recall"], color=colors,
                  edgecolor="#222", linewidth=0.6)
    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels([CELL_LABELS[c][0] for c in agg.index],
                       rotation=20, ha="right", fontsize=11)
    ax.set_ylabel("recall@10 (mean)", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title("Recall by strategy — SIFT1M, N=1M, 1% filter selectivity",
                 fontsize=13, pad=12)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(1.0, color="#bbb", linewidth=0.6, linestyle="--")

    for bar, val in zip(bars, agg["recall"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + 0.015,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"wrote {out.name}")


def plot_latency(agg: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(agg))
    w = 0.4
    colors = [CELL_LABELS[c][1] for c in agg.index]

    b1 = ax.bar(x - w / 2, agg["latency_mean"], w,
                color=colors, edgecolor="#222", linewidth=0.6, label="mean")
    b2 = ax.bar(x + w / 2, agg["latency_p99"], w,
                color=colors, edgecolor="#222", linewidth=0.6,
                alpha=0.45, label="p99", hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels([CELL_LABELS[c][0] for c in agg.index],
                       rotation=20, ha="right", fontsize=11)
    ax.set_ylabel("latency (ms)", fontsize=12)
    ax.set_title("Latency by strategy — SIFT1M, N=1M, 1% filter selectivity",
                 fontsize=13, pad=12)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=10, loc="upper right")

    ymax = max(agg["latency_p99"].max(), agg["latency_mean"].max())
    pad = 0.02 * ymax
    for bar, val in zip(b1, agg["latency_mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + pad,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(b2, agg["latency_p99"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + pad,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9, alpha=0.85)

    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"wrote {out.name}")


def main():
    df = load()
    sel = df["selectivity"].unique()
    n = df["n_records"].unique()
    assert list(sel) == [0.01], f"unexpected selectivity values: {sel}"
    assert list(n) == [1_000_000], f"unexpected n_records: {n}"
    agg = aggregate(df)
    plot_recall(agg, HERE / "recall_1M_sel1pct.png")
    plot_latency(agg, HERE / "latency_1M_sel1pct.png")
    print(agg.round(3))


if __name__ == "__main__":
    main()
