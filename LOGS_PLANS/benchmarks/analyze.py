#!/usr/bin/env python3
"""Analyse SPANN filtered-recall CSVs and produce plots + a markdown summary.

Usage:
    python analyze.py <csv_path> [<csv_path2> ...] [--out-dir DIR] [--label LABEL]

If multiple CSVs are passed, they are concatenated. The `strategy` column is
preserved per row, so passing both a fixed-strategy CSV and an adaptive-strategy
CSV produces overlay plots automatically.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load(csv_paths):
    frames = []
    for p in csv_paths:
        df = pd.read_csv(p)
        df["__source"] = Path(p).name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def aggregate(df):
    group_cols = ["dataset", "n_records", "k", "strategy", "base_nprobe", "selectivity"]
    agg = (
        df.groupby(group_cols)
        .agg(
            recall_mean=("recall_at_k", "mean"),
            recall_p10=("recall_at_k", lambda s: s.quantile(0.10)),
            recall_p50=("recall_at_k", "median"),
            latency_mean=("latency_ms", "mean"),
            latency_p50=("latency_ms", "median"),
            latency_p95=("latency_ms", lambda s: s.quantile(0.95)),
            returned_mean=("returned_count", "mean"),
            nprobe_used_mean=("nprobe_used", "mean"),
            cands_before_mean=("candidates_before_filter", "mean"),
            cands_after_mean=("candidates_after_filter", "mean"),
            n=("recall_at_k", "size"),
        )
        .reset_index()
    )
    return agg


def plot_curves(agg, out_dir, label):
    """For each (strategy, base_nprobe) draw a curve over selectivity."""
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("recall_mean", "recall@k (mean)", "recall_vs_selectivity.png"),
        ("latency_mean", "latency_ms (mean)", "latency_vs_selectivity.png"),
        ("returned_mean", "returned_count (mean)", "returned_vs_selectivity.png"),
    ]

    for metric, ylabel, fname in metrics:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for (strat, np_), sub in agg.groupby(["strategy", "base_nprobe"]):
            sub = sub.sort_values("selectivity")
            ax.plot(
                sub["selectivity"],
                sub[metric],
                marker="o",
                label=f"{strat} np={np_}",
            )
        ax.set_xscale("log")
        ax.set_xlabel("filter selectivity (log)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{label}: {ylabel} vs selectivity")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        out_path = out_dir / fname
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"wrote {out_path}")


def write_summary_md(agg, out_dir, label):
    out = out_dir / "summary.md"
    cols = [
        "strategy",
        "base_nprobe",
        "selectivity",
        "nprobe_used_mean",
        "recall_mean",
        "recall_p10",
        "latency_mean",
        "latency_p95",
        "returned_mean",
        "cands_before_mean",
        "cands_after_mean",
        "n",
    ]
    formatted = agg[cols].copy()
    formatted = formatted.sort_values(["strategy", "base_nprobe", "selectivity"])
    for c in (
        "recall_mean",
        "recall_p10",
        "latency_mean",
        "latency_p95",
        "returned_mean",
        "cands_before_mean",
        "cands_after_mean",
        "nprobe_used_mean",
    ):
        formatted[c] = formatted[c].round(3)
    md = formatted.to_markdown(index=False)
    out.write_text(f"# {label} — aggregated summary\n\n{md}\n")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--label", default="spann-filtered")
    args = ap.parse_args()

    df = load(args.csv)
    agg = aggregate(df)

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.csv[0]).parent / args.label
    plot_curves(agg, out_dir, args.label)
    write_summary_md(agg, out_dir, args.label)
    print(f"\nrows in: {len(df)}, groups out: {len(agg)}")


if __name__ == "__main__":
    main()
