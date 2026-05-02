#!/usr/bin/env python3
"""Filter-shape comparison: recall by shape × strategy, faceted by base_nprobe."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    df = pd.concat([pd.read_csv(p) for p in args.csv], ignore_index=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    g = df.groupby(["filter_shape", "strategy", "base_nprobe", "selectivity"]).agg(
        recall_mean=("recall_at_k", "mean"),
        recall_p10=("recall_at_k", lambda s: s.quantile(0.10)),
        latency_p95=("latency_ms", lambda s: s.quantile(0.95)),
        nprobe_used=("nprobe_used", "mean"),
        returned_mean=("returned_count", "mean"),
    ).reset_index().round(3)

    shapes = ["bernoulli", "categorical", "range", "adversarial"]
    bases = sorted(df["base_nprobe"].unique())

    fig, axes = plt.subplots(len(bases), len(shapes),
                             figsize=(4 * len(shapes), 3.5 * len(bases)),
                             sharex=True, sharey=True)
    if len(bases) == 1:
        axes = [axes]
    for r, b in enumerate(bases):
        for c, shape in enumerate(shapes):
            ax = axes[r][c] if len(bases) > 1 else axes[c]
            sub = g[(g["filter_shape"] == shape) & (g["base_nprobe"] == b)]
            for strat, ssub in sub.groupby("strategy"):
                ssub = ssub.sort_values("selectivity")
                ax.plot(ssub["selectivity"], ssub["recall_mean"], marker="o", label=strat)
            ax.set_xscale("log")
            ax.set_title(f"{shape}, base={int(b)}")
            ax.grid(True, which="both", alpha=0.3)
            if c == 0:
                ax.set_ylabel("recall@k (mean)")
            if r == len(bases) - 1:
                ax.set_xlabel("selectivity (log)")
            ax.legend(loc="best", fontsize=7)
            ax.set_ylim(0, 1.05)
    fig.suptitle("Filter-shape variants: recall@k by strategy")
    fig.tight_layout()
    out = out_dir / "filter_shape_recall.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")

    md = out_dir / "filter_shape_summary.md"
    md.write_text(g.to_markdown(index=False))
    print(f"wrote {md}")


if __name__ == "__main__":
    main()
