"""Plot HeadSynopsis ablation results.

Reads CSVs from LOGS_PLANS/benchmarks/synopsis_study/ and produces:

  size_sweep.png       — recall, drop_ratio, latency vs n_records (sel=1%)
  selectivity_sweep.png — drop_ratio + iso-I/O Δrecall vs selectivity (n=10K)
  recall_summary.png   — bar plot of iso-probe vs iso-I/O recall, both
                         strategies, all selectivity buckets
  scatter_recall_io.png — recall vs heads_fetched scatter, per query

Run from repo root:

  python LOGS_PLANS/benchmarks/synopsis_study/plot.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# ---------- Load all CSVs --------------------------------------------------

frames = []
for path in sorted(HERE.glob("sift1m-synopsis-n*-b*.csv")):
    df = pd.read_csv(path, dtype={"strategy": str, "scenario": str})
    df["strategy"] = df["strategy"].str.zfill(3)
    frames.append(df)
all_df = pd.concat(frames, ignore_index=True)

# ---------- Size sweep at sel=1% (buckets=100) ----------------------------

size_df = all_df[all_df["selectivity"].between(0.0099, 0.0101)].copy()
size_pivot = (
    size_df.groupby(["n_records", "strategy", "scenario"])
    .agg(
        recall=("recall_at_k", "mean"),
        drop_ratio=("drop_ratio", "mean"),
        heads_fetched=("heads_fetched", "mean"),
        latency=("latency_ms", "mean"),
    )
    .reset_index()
)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
sizes = sorted(size_pivot["n_records"].unique())

# Recall vs size
ax = axes[0]
for strat, color in [("000", "#888"), ("001", "#1f77b4")]:
    sub = size_pivot[
        (size_pivot["strategy"] == strat) & (size_pivot["scenario"] == "iso_io")
    ].sort_values("n_records")
    ax.plot(
        sub["n_records"],
        sub["recall"],
        marker="o",
        color=color,
        label=f"iso-I/O {strat}",
        linewidth=2,
    )
    sub2 = size_pivot[
        (size_pivot["strategy"] == strat) & (size_pivot["scenario"] == "iso_probe")
    ].sort_values("n_records")
    ax.plot(
        sub2["n_records"],
        sub2["recall"],
        marker="s",
        color=color,
        linestyle="--",
        label=f"iso-probe {strat}",
    )
ax.set_xscale("log")
ax.set_xlabel("n_records (SIFT1M subset)")
ax.set_ylabel("recall@10 (mean)")
ax.set_title("Recall vs collection size, sel=1%")
ax.set_xticks(sizes)
ax.set_xticklabels([f"{s//1000}K" for s in sizes])
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.3)
ax.legend(loc="lower left", fontsize=8)

# drop_ratio vs size (synopsis only)
ax = axes[1]
sub = size_pivot[
    (size_pivot["strategy"] == "001") & (size_pivot["scenario"] == "iso_probe")
].sort_values("n_records")
ax.plot(sub["n_records"], sub["drop_ratio"], marker="o", color="#2ca02c", linewidth=2)
ax.set_xscale("log")
ax.set_xlabel("n_records")
ax.set_ylabel("drop_ratio (heads dropped / heads probed)")
ax.set_title("Synopsis drop_ratio vs size, sel=1%")
ax.set_xticks(sizes)
ax.set_xticklabels([f"{s//1000}K" for s in sizes])
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.3)

# Latency
ax = axes[2]
for strat, color, label in [
    ("000", "#888", "000 (no synopsis)"),
    ("001", "#1f77b4", "001 iso-I/O"),
]:
    scen = "iso_io"
    sub = size_pivot[
        (size_pivot["strategy"] == strat) & (size_pivot["scenario"] == scen)
    ].sort_values("n_records")
    ax.plot(sub["n_records"], sub["latency"], marker="o", color=color, label=label, linewidth=2)
ax.set_xscale("log")
ax.set_xlabel("n_records")
ax.set_ylabel("latency_ms (mean per query)")
ax.set_title("Latency vs size, sel=1% (iso-I/O cell)")
ax.set_xticks(sizes)
ax.set_xticklabels([f"{s//1000}K" for s in sizes])
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", fontsize=8)

plt.tight_layout()
plt.savefig(HERE / "size_sweep.png", dpi=130, bbox_inches="tight")
plt.close()
print("wrote size_sweep.png")

# ---------- Selectivity sweep at n=10K ------------------------------------

sel_df = all_df[all_df["n_records"] == 10000].copy()

# We computed selectivity = 1/buckets in the bench, so map back.
sel_df["buckets"] = (1.0 / sel_df["selectivity"]).round().astype(int)
sel_pivot = (
    sel_df.groupby(["selectivity", "strategy", "scenario"])
    .agg(
        recall=("recall_at_k", "mean"),
        drop_ratio=("drop_ratio", "mean"),
        heads_fetched=("heads_fetched", "mean"),
    )
    .reset_index()
)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

# drop_ratio vs selectivity
ax = axes[0]
sub = sel_pivot[
    (sel_pivot["strategy"] == "001") & (sel_pivot["scenario"] == "iso_probe")
].sort_values("selectivity")
ax.plot(
    sub["selectivity"],
    sub["drop_ratio"],
    marker="o",
    color="#2ca02c",
    linewidth=2,
)
ax.set_xscale("log")
ax.set_xlabel("filter selectivity (fraction of docs matching)")
ax.set_ylabel("drop_ratio")
ax.set_title("Synopsis drop_ratio vs selectivity (n=10K)")
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)

# recall: 000 vs 001 (iso-I/O)
ax = axes[1]
for strat, color, label in [
    ("000", "#888", "000 baseline"),
    ("001", "#1f77b4", "001 synopsis"),
]:
    sub = sel_pivot[
        (sel_pivot["strategy"] == strat) & (sel_pivot["scenario"] == "iso_io")
    ].sort_values("selectivity")
    ax.plot(
        sub["selectivity"],
        sub["recall"],
        marker="o",
        color=color,
        label=label,
        linewidth=2,
    )
ax.set_xscale("log")
ax.set_xlabel("selectivity")
ax.set_ylabel("recall@10 (mean)")
ax.set_title("Recall vs selectivity (iso-I/O, n=10K)")
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

# Δrecall (iso-I/O 001 - 000)
ax = axes[2]
recall_000 = (
    sel_pivot[(sel_pivot["strategy"] == "000") & (sel_pivot["scenario"] == "iso_io")]
    .set_index("selectivity")["recall"]
)
recall_001 = (
    sel_pivot[(sel_pivot["strategy"] == "001") & (sel_pivot["scenario"] == "iso_io")]
    .set_index("selectivity")["recall"]
)
delta = (recall_001 - recall_000).sort_index()
ax.bar(
    range(len(delta)),
    delta.values,
    color=["#d62728" if v < 0 else "#2ca02c" for v in delta.values],
)
ax.set_xticks(range(len(delta)))
ax.set_xticklabels([f"{v*100:g}%" for v in delta.index], rotation=45)
ax.set_xlabel("filter selectivity")
ax.set_ylabel("Δrecall (synopsis − baseline)")
ax.set_title("Iso-I/O Δrecall vs selectivity (n=10K)")
ax.axhline(0, color="black", linewidth=0.5)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(HERE / "selectivity_sweep.png", dpi=130, bbox_inches="tight")
plt.close()
print("wrote selectivity_sweep.png")

# ---------- Recall summary bar plot ---------------------------------------

fig, ax = plt.subplots(figsize=(11, 5))
all_sel = sorted(sel_pivot["selectivity"].unique())
x = np.arange(len(all_sel))
w = 0.2

def get(strat, scen):
    out = []
    for s in all_sel:
        m = sel_pivot[
            (sel_pivot["strategy"] == strat)
            & (sel_pivot["scenario"] == scen)
            & (sel_pivot["selectivity"] == s)
        ]
        out.append(m["recall"].iloc[0] if len(m) else float("nan"))
    return out

ax.bar(x - 1.5 * w, get("000", "iso_probe"), w, label="000 iso-probe", color="#999")
ax.bar(x - 0.5 * w, get("001", "iso_probe"), w, label="001 iso-probe", color="#1f77b4")
ax.bar(x + 0.5 * w, get("000", "iso_io"), w, label="000 iso-I/O", color="#666")
ax.bar(x + 1.5 * w, get("001", "iso_io"), w, label="001 iso-I/O", color="#2ca02c")
ax.set_xticks(x)
ax.set_xticklabels([f"{v*100:g}%" for v in all_sel])
ax.set_xlabel("filter selectivity (lower = more selective)")
ax.set_ylabel("recall@10 (mean)")
ax.set_title("Recall by strategy and scenario (SIFT1M n=10K)")
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3, axis="y")
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig(HERE / "recall_summary.png", dpi=130, bbox_inches="tight")
plt.close()
print("wrote recall_summary.png")

# ---------- Scatter: recall vs heads_fetched, all queries -----------------

fig, ax = plt.subplots(figsize=(8, 6))
focus = all_df[
    (all_df["n_records"] == 10000)
    & (all_df["selectivity"].between(0.0099, 0.0101))
]
for (strat, scen), color, marker, label in [
    (("000", "iso_probe"), "#888", "o", "000 iso-probe"),
    (("001", "iso_probe"), "#1f77b4", "o", "001 iso-probe"),
    (("000", "iso_io"), "#444", "x", "000 iso-I/O"),
    (("001", "iso_io"), "#2ca02c", "^", "001 iso-I/O"),
]:
    sub = focus[(focus["strategy"] == strat) & (focus["scenario"] == scen)]
    ax.scatter(
        sub["heads_fetched"],
        sub["recall_at_k"],
        c=color,
        marker=marker,
        s=24,
        alpha=0.6,
        label=label,
    )
ax.set_xlabel("heads_fetched (per-query I/O)")
ax.set_ylabel("recall@10 (per query)")
ax.set_title("Recall vs heads_fetched (SIFT1M n=10K, sel=1%)")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(HERE / "scatter_recall_io.png", dpi=130, bbox_inches="tight")
plt.close()
print("wrote scatter_recall_io.png")

# ---------- Print summary table ------------------------------------------

print("\n=== Size sweep (sel=1%) ===")
size_summary = (
    size_pivot.pivot_table(
        index="n_records",
        columns=["scenario", "strategy"],
        values=["recall", "drop_ratio", "heads_fetched", "latency"],
        aggfunc="first",
    )
)
print(size_summary.to_string())

print("\n=== Selectivity sweep (n=10K) ===")
sel_summary = (
    sel_pivot.pivot_table(
        index="selectivity",
        columns=["scenario", "strategy"],
        values=["recall", "drop_ratio"],
        aggfunc="first",
    )
)
print(sel_summary.to_string())

# Bad-drops sanity
bad = all_df["bad_drops"].sum()
print(f"\nTotal bad_drops across all CSVs: {bad}  (must be 0 for correctness)")
