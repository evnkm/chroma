#!/usr/bin/env python3
"""Multi-panel scale plot.

Reads several CSVs, groups by n_records (cols) × strategy (rows), draws
recall@k vs selectivity per (strategy, base_nprobe). Output to a single PNG.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label", default="scale")
    args = ap.parse_args()

    df = pd.concat([pd.read_csv(p) for p in args.csv], ignore_index=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ns = sorted(df["n_records"].unique())
    strats = sorted(df["strategy"].unique())

    for metric, ylabel, fname in [
        ("recall_at_k", "recall@k (mean)", "scale_recall_grid.png"),
        ("latency_ms", "latency_ms (p95)", "scale_p95_latency_grid.png"),
        ("returned_count", "returned_count (mean)", "scale_returned_grid.png"),
    ]:
        fig, axes = plt.subplots(
            len(strats),
            len(ns),
            figsize=(4.5 * len(ns), 3.5 * len(strats)),
            sharex=True,
            sharey=(metric == "recall_at_k"),
        )
        if len(strats) == 1:
            axes = [axes]
        if len(ns) == 1:
            axes = [[ax] for ax in axes]

        for r, strat in enumerate(strats):
            for c, n in enumerate(ns):
                ax = axes[r][c]
                sub = df[(df["n_records"] == n) & (df["strategy"] == strat)]
                if metric == "latency_ms":
                    g = sub.groupby(["base_nprobe", "selectivity"])[metric].quantile(0.95).reset_index()
                else:
                    g = sub.groupby(["base_nprobe", "selectivity"])[metric].mean().reset_index()
                for nb, sub_b in g.groupby("base_nprobe"):
                    sub_b = sub_b.sort_values("selectivity")
                    ax.plot(
                        sub_b["selectivity"],
                        sub_b[metric],
                        marker="o",
                        label=f"np={int(nb)}",
                    )
                ax.set_xscale("log")
                ax.set_title(f"{strat}, n={n:,}")
                ax.grid(True, which="both", alpha=0.3)
                if r == len(strats) - 1:
                    ax.set_xlabel("selectivity (log)")
                if c == 0:
                    ax.set_ylabel(ylabel)
                ax.legend(loc="best", fontsize=7)
        fig.suptitle(f"{args.label}: {ylabel} vs selectivity, faceted by n × strategy", fontsize=11)
        fig.tight_layout()
        out = out_dir / fname
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"wrote {out}")

    # Summary table.
    g = df.groupby(["n_records", "strategy", "base_nprobe", "selectivity"]).agg(
        recall_mean=("recall_at_k", "mean"),
        recall_p10=("recall_at_k", lambda s: s.quantile(0.10)),
        latency_p50=("latency_ms", "median"),
        latency_p95=("latency_ms", lambda s: s.quantile(0.95)),
        nprobe_used=("nprobe_used", "mean"),
        returned_mean=("returned_count", "mean"),
        n=("recall_at_k", "size"),
    ).reset_index().round(3)
    md_path = out_dir / "scale_summary.md"
    md_path.write_text(f"# {args.label}\n\n{g.to_markdown(index=False)}\n")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
