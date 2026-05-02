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


PHASE_COLS = ["t_centers_ms", "t_fetch_pl_ms", "t_bf_pl_ms", "t_merge_ms"]


def aggregate(df):
    group_cols = ["dataset", "n_records", "k", "strategy", "base_nprobe", "selectivity"]
    agg_kwargs = dict(
        recall_mean=("recall_at_k", "mean"),
        recall_p10=("recall_at_k", lambda s: s.quantile(0.10)),
        recall_p25=("recall_at_k", lambda s: s.quantile(0.25)),
        recall_p50=("recall_at_k", "median"),
        latency_mean=("latency_ms", "mean"),
        latency_p50=("latency_ms", "median"),
        latency_p90=("latency_ms", lambda s: s.quantile(0.90)),
        latency_p95=("latency_ms", lambda s: s.quantile(0.95)),
        latency_p99=("latency_ms", lambda s: s.quantile(0.99)),
        returned_mean=("returned_count", "mean"),
        nprobe_used_mean=("nprobe_used", "mean"),
        cands_before_mean=("candidates_before_filter", "mean"),
        cands_after_mean=("candidates_after_filter", "mean"),
        n=("recall_at_k", "size"),
    )
    # Per-phase mean timings if present in the CSV (D1 instrumentation).
    for col in PHASE_COLS:
        if col in df.columns:
            agg_kwargs[f"{col}_mean"] = (col, "mean")
    agg = df.groupby(group_cols).agg(**agg_kwargs).reset_index()
    return agg


def plot_curves(agg, out_dir, label):
    """For each (strategy, base_nprobe) draw a curve over selectivity."""
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("recall_mean", "recall@k (mean)", "recall_vs_selectivity.png"),
        ("recall_p10", "recall@k (p10)", "recall_p10_vs_selectivity.png"),
        ("latency_mean", "latency_ms (mean)", "latency_vs_selectivity.png"),
        ("latency_p95", "latency_ms (p95)", "latency_p95_vs_selectivity.png"),
        ("returned_mean", "returned_count (mean)", "returned_vs_selectivity.png"),
    ]

    for metric, ylabel, fname in metrics:
        if metric not in agg.columns:
            continue
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

    # Per-phase stacked-bar plot if instrumentation columns are present.
    phase_cols = [f"{c}_mean" for c in PHASE_COLS if f"{c}_mean" in agg.columns]
    if phase_cols:
        for strat in agg["strategy"].unique():
            sub = agg[agg["strategy"] == strat].copy()
            sub = sub.sort_values(["base_nprobe", "selectivity"])
            sub["xkey"] = sub.apply(
                lambda r: f"np={int(r['base_nprobe'])}\nsel={r['selectivity']:g}",
                axis=1,
            )
            fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(sub)), 4.5))
            bottoms = [0.0] * len(sub)
            for col in phase_cols:
                vals = sub[col].fillna(0).tolist()
                ax.bar(sub["xkey"], vals, bottom=bottoms, label=col.replace("_mean", ""))
                bottoms = [b + v for b, v in zip(bottoms, vals)]
            ax.set_ylabel("latency_ms (mean)")
            ax.set_title(f"{label}: per-phase share — strategy={strat}")
            ax.legend(loc="best", fontsize=8)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=6)
            fig.tight_layout()
            out_path = out_dir / f"phase_share_{strat}.png"
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
