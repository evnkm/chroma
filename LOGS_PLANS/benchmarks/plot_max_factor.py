#!/usr/bin/env python3
"""Plot max_factor sensitivity at fixed base_nprobe."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    df = pd.concat([pd.read_csv(p) for p in args.csv], ignore_index=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    g = df.groupby(["selectivity", "max_factor"]).agg(
        recall_mean=("recall_at_k", "mean"),
        recall_p10=("recall_at_k", lambda s: s.quantile(0.10)),
        latency_p50=("latency_ms", "median"),
        latency_p95=("latency_ms", lambda s: s.quantile(0.95)),
        nprobe_used=("nprobe_used", "mean"),
        returned_mean=("returned_count", "mean"),
    ).reset_index()

    # Recall vs max_factor.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for sel, sub in g.groupby("selectivity"):
        sub = sub.sort_values("max_factor")
        axes[0].plot(sub["max_factor"], sub["recall_mean"], marker="o", label=f"sel={sel:g}")
        axes[1].plot(sub["max_factor"], sub["latency_p95"], marker="o", label=f"sel={sel:g}")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("max_factor (log2)")
    axes[0].set_ylabel("recall@k (mean)")
    axes[0].set_title("Recall vs max_factor (base_nprobe=8)")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(loc="best")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("max_factor (log2)")
    axes[1].set_ylabel("latency_ms (p95)")
    axes[1].set_title("p95 latency vs max_factor")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(loc="best")
    fig.suptitle("max_factor sensitivity at sel ∈ {0.001, 0.01, 0.05}, base=8")
    fig.tight_layout()
    out = out_dir / "max_factor_sensitivity.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")

    md = out_dir / "max_factor_summary.md"
    md.write_text(g.round(3).to_markdown(index=False))
    print(f"wrote {md}")


if __name__ == "__main__":
    main()
