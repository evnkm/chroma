#!/usr/bin/env python3
"""Compare iterative top-up to single-shot adaptive."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("topup_csv")
    ap.add_argument("adaptive_csv")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    topup = pd.read_csv(args.topup_csv)
    adapt = pd.read_csv(args.adaptive_csv)
    df = pd.concat([topup, adapt], ignore_index=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    g = df.groupby(["strategy", "base_nprobe", "selectivity"]).agg(
        recall_mean=("recall_at_k", "mean"),
        recall_p10=("recall_at_k", lambda s: s.quantile(0.10)),
        latency_mean=("latency_ms", "mean"),
        latency_p50=("latency_ms", "median"),
        latency_p95=("latency_ms", lambda s: s.quantile(0.95)),
        latency_p99=("latency_ms", lambda s: s.quantile(0.99)),
        nprobe_used=("nprobe_used", "mean"),
        returned_mean=("returned_count", "mean"),
        n=("recall_at_k", "size"),
    ).reset_index().round(3)

    # Plot: recall, p95 latency, mean nprobe — adaptive vs topup.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for (strat, b), sub in g.groupby(["strategy", "base_nprobe"]):
        sub = sub.sort_values("selectivity")
        marker = "o" if strat == "adaptive" else "s"
        axes[0].plot(sub["selectivity"], sub["recall_mean"], marker=marker,
                     label=f"{strat} np={int(b)}")
        axes[1].plot(sub["selectivity"], sub["latency_p95"], marker=marker,
                     label=f"{strat} np={int(b)}")
        axes[2].plot(sub["selectivity"], sub["nprobe_used"], marker=marker,
                     label=f"{strat} np={int(b)}")
    for ax in axes:
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=7)
        ax.set_xlabel("selectivity (log)")
    axes[0].set_ylabel("recall@k (mean)")
    axes[1].set_ylabel("latency_ms (p95)")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("nprobe_used (mean, log)")
    fig.suptitle("Iterative top-up vs single-shot adaptive (n=10K, k=10, 50 queries)")
    fig.tight_layout()
    out = out_dir / "topup_vs_adaptive.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")

    md = out_dir / "topup_summary.md"
    md.write_text(g.to_markdown(index=False))
    print(f"wrote {md}")


if __name__ == "__main__":
    main()
