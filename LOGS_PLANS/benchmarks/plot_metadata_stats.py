#!/usr/bin/env python3
"""Compare per-centroid metadata-stats variants vs fixed/adaptive.

For each filter shape and base_nprobe, draw recall@k vs selectivity for
all four strategies (fixed, adaptive, metadata_stats, metadata_stats_hybrid).
Also produce a latency-vs-recall scatter so the report can talk about the
recall/latency frontier honestly.
"""

import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


STRAT_ORDER = ["fixed", "adaptive", "metadata_stats", "metadata_stats_hybrid"]
STRAT_COLORS = {
    "fixed": "tab:gray",
    "adaptive": "tab:blue",
    "metadata_stats": "tab:orange",
    "metadata_stats_hybrid": "tab:green",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    df = pd.concat([pd.read_csv(p) for p in args.csv], ignore_index=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    g = df.groupby(["filter_shape", "strategy", "base_nprobe", "selectivity"]).agg(
        recall=("recall_at_k", "mean"),
        recall_p10=("recall_at_k", lambda s: s.quantile(0.10)),
        latency_mean=("latency_ms", "mean"),
        latency_p95=("latency_ms", lambda s: s.quantile(0.95)),
        nprobe_used=("nprobe_used", "mean"),
        returned=("returned_count", "mean"),
    ).reset_index().round(3)

    shapes = ["bernoulli", "range", "adversarial"]
    bases = sorted(df["base_nprobe"].unique())

    # Recall grid: rows = base_nprobe, cols = filter_shape.
    fig, axes = plt.subplots(len(bases), len(shapes),
                             figsize=(4.5 * len(shapes), 3.5 * len(bases)),
                             sharex=True, sharey=True)
    if len(bases) == 1:
        axes = [axes]
    for r, b in enumerate(bases):
        for c, shape in enumerate(shapes):
            ax = axes[r][c] if len(bases) > 1 else axes[c]
            sub = g[(g["filter_shape"] == shape) & (g["base_nprobe"] == b)]
            for strat in STRAT_ORDER:
                ssub = sub[sub["strategy"] == strat].sort_values("selectivity")
                if ssub.empty:
                    continue
                ax.plot(ssub["selectivity"], ssub["recall"], marker="o",
                        color=STRAT_COLORS[strat], label=strat)
            ax.set_xscale("log")
            ax.set_title(f"{shape}, base={int(b)}")
            ax.grid(True, which="both", alpha=0.3)
            if c == 0:
                ax.set_ylabel("recall@k (mean)")
            if r == len(bases) - 1:
                ax.set_xlabel("selectivity (log)")
            ax.set_ylim(0, 1.05)
            if r == 0 and c == len(shapes) - 1:
                ax.legend(loc="lower right", fontsize=7)
    fig.suptitle("Per-centroid metadata-stats vs fixed/adaptive")
    fig.tight_layout()
    p = out_dir / "metadata_stats_recall_grid.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"wrote {p}")

    # Recall/latency frontier scatter (one panel per base_nprobe).
    fig, axes = plt.subplots(1, len(bases), figsize=(5 * len(bases), 4.5),
                             sharey=True)
    if len(bases) == 1:
        axes = [axes]
    for c, b in enumerate(bases):
        ax = axes[c]
        sub = g[g["base_nprobe"] == b]
        for strat in STRAT_ORDER:
            ssub = sub[sub["strategy"] == strat]
            ax.scatter(ssub["latency_mean"], ssub["recall"], marker="o",
                       color=STRAT_COLORS[strat], label=strat, s=40, alpha=0.8)
            for _, row in ssub.iterrows():
                ax.annotate(f"sel={row['selectivity']:g}",
                            (row["latency_mean"], row["recall"]),
                            fontsize=6, alpha=0.6,
                            xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("latency_ms (mean)")
        if c == 0:
            ax.set_ylabel("recall@k (mean)")
        ax.set_title(f"base_nprobe = {int(b)}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=7)
    fig.suptitle("Recall vs latency — same axes, all strategies")
    fig.tight_layout()
    p = out_dir / "metadata_stats_recall_vs_latency.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"wrote {p}")

    md = out_dir / "metadata_stats_summary.md"
    md.write_text(g.to_markdown(index=False))
    print(f"wrote {md}")


if __name__ == "__main__":
    main()
