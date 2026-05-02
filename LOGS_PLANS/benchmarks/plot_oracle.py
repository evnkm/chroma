#!/usr/bin/env python3
"""Oracle baseline: per-(query, sel), find min nprobe achieving target recall.

Reads a wide-grid `fixed`-strategy CSV (one row per nprobe candidate) and
finds, for each (query, selectivity), the smallest nprobe whose recall@k
meets a target. Compares this to adaptive nprobe on the same sweep.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("oracle_csv", help="wide-grid fixed-strategy CSV")
    ap.add_argument("--target", type=float, default=0.95)
    ap.add_argument("--max-factor", type=float, default=8.0)
    ap.add_argument("--epsilon", type=float, default=0.001)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.oracle_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # For each (query_id, selectivity) find min nprobe_used with recall>=target.
    rows = []
    for (qid, sel), sub in df.groupby(["query_id", "selectivity"]):
        sub = sub.sort_values("nprobe_used")
        ok = sub[sub["recall_at_k"] >= args.target]
        if not ok.empty:
            oracle_nprobe = int(ok.iloc[0]["nprobe_used"])
            oracle_lat = float(ok.iloc[0]["latency_ms"])
        else:
            # Couldn't reach target even with the largest nprobe.
            oracle_nprobe = int(sub.iloc[-1]["nprobe_used"])
            oracle_lat = float(sub.iloc[-1]["latency_ms"])
        rows.append(
            dict(
                query_id=qid,
                selectivity=sel,
                oracle_nprobe=oracle_nprobe,
                oracle_latency_ms=oracle_lat,
            )
        )
    oracle = pd.DataFrame(rows)

    # Adaptive nprobe (deterministic from base + sel + clamp).
    def adaptive_nprobe(base, sel):
        raw = base / max(sel, args.epsilon)
        return int(round(min(max(raw, base), base * args.max_factor)))

    bases = [8, 32, 64]
    fixed_curves = {b: oracle.copy() for b in bases}
    for b in bases:
        fixed_curves[b]["adaptive_nprobe"] = fixed_curves[b]["selectivity"].apply(
            lambda s, b=b: adaptive_nprobe(b, s)
        )

    summary = oracle.groupby("selectivity").agg(
        oracle_nprobe_p50=("oracle_nprobe", "median"),
        oracle_nprobe_p90=("oracle_nprobe", lambda s: s.quantile(0.90)),
        oracle_latency_p50=("oracle_latency_ms", "median"),
        oracle_latency_p90=("oracle_latency_ms", lambda s: s.quantile(0.90)),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    # Left: nprobe vs selectivity, oracle median + adaptive curves for each base.
    sub = summary.sort_values("selectivity")
    axes[0].plot(sub["selectivity"], sub["oracle_nprobe_p50"], marker="o",
                 label="oracle (p50)", linewidth=2)
    axes[0].plot(sub["selectivity"], sub["oracle_nprobe_p90"], marker="x",
                 label="oracle (p90)", alpha=0.7)
    for b in bases:
        c = fixed_curves[b].drop_duplicates("selectivity").sort_values("selectivity")
        axes[0].plot(c["selectivity"], c["adaptive_nprobe"], marker="s",
                     linestyle="--", label=f"adaptive (base={b})")
        axes[0].plot(c["selectivity"], [b] * len(c), marker=".",
                     linestyle=":", label=f"fixed np={b}", alpha=0.5)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("selectivity (log)")
    axes[0].set_ylabel("nprobe_used (log)")
    axes[0].set_title(f"Oracle vs adaptive nprobe (target recall ≥ {args.target})")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(loc="best", fontsize=7)

    # Right: oracle latency distribution per sel.
    for sel, sub in oracle.groupby("selectivity"):
        axes[1].hist(sub["oracle_latency_ms"], bins=15, alpha=0.4, label=f"sel={sel:g}")
    axes[1].set_xlabel("oracle latency_ms (per query)")
    axes[1].set_ylabel("count")
    axes[1].set_title("Per-query oracle latency distribution")
    axes[1].legend(loc="best", fontsize=7)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out = out_dir / "oracle_vs_adaptive.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")

    md = out_dir / "oracle_summary.md"
    md.write_text(summary.round(2).to_markdown(index=False))
    print(f"wrote {md}")


if __name__ == "__main__":
    main()
