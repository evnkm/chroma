#!/usr/bin/env python3
"""HNSW prefilter vs postfilter baseline plots for the §1 motivation section.

Sources are CSVs imported from the `hnsw-baseline-archive` branch:
  - baseline_sweep.csv: HNSW prefilter latency phases vs selectivity
  - compare_full.csv:   prefilter vs postfilter:N recall/latency
  - msmarco_smoke.csv:  same on real MSMARCO MiniLM embeddings
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_baseline(csv_path, out_dir):
    df = pd.read_csv(csv_path)
    # Aggregate trials.
    g = df.groupby(["size", "selectivity"]).agg(
        prefilter_ms=("prefilter_ms_mean", "mean"),
        ann_ms=("ann_ms_mean", "mean"),
        hydrate_ms=("hydrate_ms_mean", "mean"),
        total_ms=("total_ms_mean", "mean"),
        total_p95=("total_ms_p95", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for n, sub in g.groupby("size"):
        sub = sub.sort_values("selectivity")
        axes[0].plot(sub["selectivity"], sub["prefilter_ms"], marker="o",
                     label=f"prefilter (n={int(n):,})")
        axes[0].plot(sub["selectivity"], sub["ann_ms"], marker="s",
                     linestyle="--", label=f"ann (n={int(n):,})")
        axes[1].plot(sub["selectivity"], sub["total_ms"], marker="o",
                     label=f"total (n={int(n):,})")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("selectivity (log)")
    axes[0].set_ylabel("phase latency (ms)")
    axes[0].set_title("HNSW prefilter vs ANN cost (synthetic, dim=128, k=10)")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("selectivity (log)")
    axes[1].set_ylabel("total query latency (ms)")
    axes[1].set_title("HNSW total prefilter latency vs selectivity")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    p = out_dir / "hnsw_baseline_phases.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"wrote {p}")


def plot_compare(csv_path, out_dir, label, fname):
    df = pd.read_csv(csv_path)
    size_col = "size" if "size" in df.columns else "n_passages"
    sel_col = "selectivity_target"
    g = df.groupby([size_col, sel_col, "strategy"]).agg(
        recall=("recall_mean", "mean"),
        latency=("total_ms_mean", "mean"),
        latency_p95=("total_ms_p95", "mean"),
        returned=("n_returned_mean", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    biggest_n = g[size_col].max()
    sub = g[g[size_col] == biggest_n]
    for strat, ssub in sub.groupby("strategy"):
        ssub = ssub.sort_values(sel_col)
        axes[0].plot(ssub[sel_col], ssub["recall"], marker="o", label=strat)
        axes[1].plot(ssub[sel_col], ssub["latency"], marker="o", label=strat)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("selectivity (log)")
    axes[0].set_ylabel("recall@k")
    axes[0].set_title(f"{label}: recall@k by HNSW strategy (n={int(biggest_n):,})")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("selectivity (log)")
    axes[1].set_ylabel("latency (mean ms)")
    axes[1].set_title(f"{label}: latency by HNSW strategy")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    p = out_dir / fname
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"wrote {p}")
    md = out_dir / f"{Path(fname).stem}.md"
    md.write_text(g.round(3).to_markdown(index=False))
    print(f"wrote {md}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="LOGS_PLANS/benchmarks/v3/d9-hnsw-baseline")
    ap.add_argument("--out-dir", default="LOGS_PLANS/benchmarks/v3/d9-hnsw-baseline")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_baseline(in_dir / "baseline_sweep.csv", out_dir)
    plot_compare(in_dir / "compare_full.csv", out_dir,
                 label="HNSW (synthetic)",
                 fname="hnsw_compare_synthetic.png")
    plot_compare(in_dir / "msmarco_smoke.csv", out_dir,
                 label="HNSW (MSMARCO smoke)",
                 fname="hnsw_compare_msmarco.png")


if __name__ == "__main__":
    main()
