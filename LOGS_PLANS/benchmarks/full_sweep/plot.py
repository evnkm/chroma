"""Plot the 8-cell hypercube sweep across {adaptive, bloom, synopsis}.

Reads CSVs from this directory. Produces:

  hypercube_n10k.png       — 4-panel summary at N=10K, sel=1%
  hypercube_n100k.png      — 4-panel summary at N=100K, sel=1%
  recall_vs_io.png         — recall vs heads_fetched scatter, all cells × scales
  selectivity_scan.png     — recall + drop_ratio + heads_fetched vs selectivity
                             at N=10K (lines per cell)
  storage_cost.png         — bloom + synopsis blob bytes vs N
  audit_table.png          — bad_drops per cell (must be all-zero)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# Cell-code → (label, color). Cells are ordered for natural reading.
CELL_LABELS = {
    "000": ("baseline", "#8c8c8c"),
    "100": ("adaptive", "#1f77b4"),
    "010": ("bloom", "#ff7f0e"),
    "001": ("synopsis", "#2ca02c"),
    "110": ("adapt+bloom", "#9467bd"),
    "101": ("adapt+syn", "#e377c2"),
    "011": ("bloom+syn", "#d62728"),
    "111": ("all three", "#000000"),
}
CELL_ORDER = list(CELL_LABELS.keys())


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"cell": str})
    df["cell"] = df["cell"].str.zfill(3)
    return df


def aggregate_per_cell(df: pd.DataFrame) -> pd.DataFrame:
    """Mean per-cell across queries."""
    agg = (
        df.groupby("cell")
        .agg(
            recall=("recall_at_k", "mean"),
            heads_rng=("heads_rng", "mean"),
            heads_fetched=("heads_fetched", "mean"),
            drop_ratio=("drop_ratio", "mean"),
            latency_mean=("latency_ms", "mean"),
            latency_p99=("latency_ms", lambda x: np.quantile(x, 0.99)),
            candidates_before=("candidates_before_filter", "mean"),
            candidates_after=("candidates_after_filter", "mean"),
            bad_drops=("bad_drops", "sum"),
            nprobe_used=("nprobe_used", "max"),
        )
        .reindex(CELL_ORDER)
    )
    return agg


def four_panel(df: pd.DataFrame, title: str, summary: dict, out_path: Path):
    agg = aggregate_per_cell(df)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # Recall.
    ax = axes[0, 0]
    bars = ax.bar(
        range(len(agg)),
        agg["recall"],
        color=[CELL_LABELS[c][1] for c in agg.index],
    )
    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels([CELL_LABELS[c][0] for c in agg.index], rotation=30, ha="right")
    ax.set_ylabel("recall@10 (mean)")
    ax.set_title("Recall by cell")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, agg["recall"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # heads_fetched.
    ax = axes[0, 1]
    ax.bar(
        range(len(agg)),
        agg["heads_fetched"],
        color=[CELL_LABELS[c][1] for c in agg.index],
    )
    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels([CELL_LABELS[c][0] for c in agg.index], rotation=30, ha="right")
    ax.set_ylabel("heads_fetched (mean)")
    ax.set_title("I/O per query")
    ax.grid(True, alpha=0.3, axis="y")
    for i, val in enumerate(agg["heads_fetched"]):
        ax.text(i, val * 1.02, f"{val:.0f}", ha="center", va="bottom", fontsize=8)

    # Latency mean + p99.
    ax = axes[1, 0]
    x = np.arange(len(agg))
    ax.bar(
        x - 0.2,
        agg["latency_mean"],
        0.4,
        label="mean",
        color=[CELL_LABELS[c][1] for c in agg.index],
    )
    ax.bar(
        x + 0.2,
        agg["latency_p99"],
        0.4,
        label="p99",
        color=[CELL_LABELS[c][1] for c in agg.index],
        alpha=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([CELL_LABELS[c][0] for c in agg.index], rotation=30, ha="right")
    ax.set_ylabel("latency (ms)")
    ax.set_title("Latency: mean and p99")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    # drop_ratio (only meaningful for cells that gate).
    ax = axes[1, 1]
    bars = ax.bar(
        range(len(agg)),
        agg["drop_ratio"],
        color=[CELL_LABELS[c][1] for c in agg.index],
    )
    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels([CELL_LABELS[c][0] for c in agg.index], rotation=30, ha="right")
    ax.set_ylabel("drop_ratio")
    ax.set_title("Heads dropped by gate")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, agg["drop_ratio"]):
        if val > 0.01:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.02,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path.name}")


def recall_vs_io_scatter(out_path: Path):
    """Combine all available primary CSVs (10K and 100K, sel=1%) into one
    scatter showing the recall/I-O Pareto frontier per cell."""
    files = [
        ("N=10K", HERE / "sift1m-full-n10000-q50-b100.csv"),
        ("N=100K", HERE / "sift1m-full-n100000-q50-b100.csv"),
    ]
    fig, ax = plt.subplots(figsize=(9, 6))
    markers = {"N=10K": "o", "N=100K": "s"}
    for label, path in files:
        if not path.exists():
            continue
        df = load_csv(path)
        agg = aggregate_per_cell(df)
        for cell, row in agg.iterrows():
            color = CELL_LABELS[cell][1]
            ax.scatter(
                row["heads_fetched"],
                row["recall"],
                s=120,
                marker=markers[label],
                color=color,
                edgecolors="black",
                linewidth=0.7,
            )
            ax.annotate(
                f"{cell}",
                (row["heads_fetched"], row["recall"]),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=8,
            )
    # Build a custom legend that combines cell-color and N-marker.
    cell_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=col, markersize=10, label=lbl)
        for code, (lbl, col) in CELL_LABELS.items()
    ]
    n_handles = [
        plt.Line2D(
            [0], [0], marker=m, color="w", markerfacecolor="#aaa", markersize=10, label=lbl,
            markeredgecolor="black",
        )
        for lbl, m in markers.items()
    ]
    ax.legend(handles=cell_handles + n_handles, fontsize=8, loc="lower right", ncol=2)
    ax.set_xlabel("heads_fetched (mean per query)")
    ax.set_ylabel("recall@10 (mean)")
    ax.set_title("Recall vs I/O — 8 cells × 2 collection sizes (sel=1%, probe seed=32)")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path.name}")


def selectivity_scan(out_path: Path):
    """Recall, drop_ratio, heads_fetched as a function of selectivity for each cell."""
    files = sorted(HERE.glob("sift1m-full-n10000-b*.csv"))
    if not files:
        return
    rows = []
    for f in files:
        m = re.search(r"-b(\d+)\.csv$", f.name)
        if not m:
            continue
        buckets = int(m.group(1))
        sel = 1.0 / buckets
        df = load_csv(f)
        agg = aggregate_per_cell(df)
        for cell, r in agg.iterrows():
            rows.append(
                {
                    "selectivity": sel,
                    "buckets": buckets,
                    "cell": cell,
                    "recall": r["recall"],
                    "drop_ratio": r["drop_ratio"],
                    "heads_fetched": r["heads_fetched"],
                    "latency": r["latency_mean"],
                }
            )
    sel_df = pd.DataFrame(rows).sort_values("selectivity")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    for cell in CELL_ORDER:
        sub = sel_df[sel_df["cell"] == cell].sort_values("selectivity")
        if sub.empty:
            continue
        ax.plot(
            sub["selectivity"],
            sub["recall"],
            marker="o",
            color=CELL_LABELS[cell][1],
            label=CELL_LABELS[cell][0],
        )
    ax.set_xscale("log")
    ax.set_xlabel("filter selectivity")
    ax.set_ylabel("recall@10")
    ax.set_title("Recall vs selectivity (N=10K, probe seed=32)")
    ax.set_ylim(0.4, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower right", ncol=2)

    ax = axes[1]
    for cell in CELL_ORDER:
        sub = sel_df[sel_df["cell"] == cell].sort_values("selectivity")
        if sub.empty:
            continue
        ax.plot(
            sub["selectivity"],
            sub["drop_ratio"],
            marker="o",
            color=CELL_LABELS[cell][1],
        )
    ax.set_xscale("log")
    ax.set_xlabel("filter selectivity")
    ax.set_ylabel("drop_ratio")
    ax.set_title("Drop ratio (gate effectiveness)")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    for cell in CELL_ORDER:
        sub = sel_df[sel_df["cell"] == cell].sort_values("selectivity")
        if sub.empty:
            continue
        ax.plot(
            sub["selectivity"],
            sub["heads_fetched"],
            marker="o",
            color=CELL_LABELS[cell][1],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("filter selectivity")
    ax.set_ylabel("heads_fetched (mean)")
    ax.set_title("I/O cost")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path.name}")


def storage_cost(out_path: Path):
    """Plot blob storage bytes per writer flavor across collection sizes."""
    summaries = []
    for f in HERE.glob("*.summary.json"):
        with open(f) as fh:
            data = json.load(fh)
        if data.get("buckets") != 100:  # only sel=1% files
            continue
        for flavor, meta in data["index_metadata"].items():
            summaries.append(
                {
                    "n_records": data["n_records"],
                    "flavor": flavor,
                    "blob_bytes": meta["blob_storage_bytes"],
                    "build_ms": meta["build_ms"],
                }
            )
    if not summaries:
        return
    sdf = pd.DataFrame(summaries).sort_values(["n_records", "flavor"])
    flavors = ["none", "bloom", "synopsis", "both"]
    n_values = sorted(sdf["n_records"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    x = np.arange(len(n_values))
    width = 0.2
    flavor_colors = {"none": "#bbb", "bloom": "#ff7f0e", "synopsis": "#2ca02c", "both": "#d62728"}
    for i, flavor in enumerate(flavors):
        sub = sdf[sdf["flavor"] == flavor].sort_values("n_records")
        bytes_per_n = [
            sub[sub["n_records"] == n]["blob_bytes"].sum() if (sub["n_records"] == n).any() else 0
            for n in n_values
        ]
        # Convert to MB.
        ax.bar(
            x + (i - 1.5) * width,
            [v / 1e6 for v in bytes_per_n],
            width,
            color=flavor_colors[flavor],
            label=flavor,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n//1000}K" if n >= 1000 else str(n) for n in n_values])
    ax.set_xlabel("collection size (n_records)")
    ax.set_ylabel("blob storage (MB)")
    ax.set_title("Gate blob storage cost (sel=1%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    for i, flavor in enumerate(flavors):
        sub = sdf[sdf["flavor"] == flavor].sort_values("n_records")
        ax.plot(
            sub["n_records"],
            sub["build_ms"] / 1000.0,
            marker="o",
            color=flavor_colors[flavor],
            label=flavor,
        )
    ax.set_xscale("log")
    ax.set_xlabel("collection size")
    ax.set_ylabel("build time (s)")
    ax.set_title("Index build time vs N")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path.name}")


def audit_check(out_path: Path):
    """Verify bad_drops=0 across all CSVs; produce a small summary heatmap."""
    rows = []
    for f in HERE.glob("sift1m-full-n*-*.csv"):
        m = re.search(r"-n(\d+)-(?:q\d+-)?b(\d+)\.csv$", f.name)
        if not m:
            continue
        n_records = int(m.group(1))
        buckets = int(m.group(2))
        df = load_csv(f)
        agg = aggregate_per_cell(df)
        for cell, r in agg.iterrows():
            rows.append(
                {
                    "label": f"N={n_records//1000}K, b={buckets}",
                    "cell": cell,
                    "bad_drops": int(r["bad_drops"]),
                }
            )
    if not rows:
        return
    audit_df = pd.DataFrame(rows)
    pivot = audit_df.pivot_table(
        index="label", columns="cell", values="bad_drops", aggfunc="sum", fill_value=0
    )
    pivot = pivot.reindex(columns=CELL_ORDER)
    fig, ax = plt.subplots(figsize=(11, max(3, 0.5 * len(pivot))))
    cmap = plt.cm.Reds
    img = ax.imshow(pivot.values, cmap=cmap, aspect="auto", vmin=0, vmax=max(1, pivot.values.max()))
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(
        [f"{c}\n{CELL_LABELS[c][0]}" for c in pivot.columns], fontsize=8
    )
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(len(pivot)):
        for j in range(len(pivot.columns)):
            ax.text(
                j,
                i,
                str(int(pivot.values[i, j])),
                ha="center",
                va="center",
                color="black" if pivot.values[i, j] == 0 else "white",
                fontsize=9,
            )
    fig.colorbar(img, ax=ax, label="bad_drops (must be 0)")
    ax.set_title("Gate-audit invariant — bad_drops per (workload, cell)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    total = audit_df["bad_drops"].sum()
    print(f"wrote {out_path.name} (total bad_drops across all measurements: {total})")


def main():
    primary_n10k = HERE / "sift1m-full-n10000-q50-b100.csv"
    primary_n100k = HERE / "sift1m-full-n100000-q50-b100.csv"

    if primary_n10k.exists():
        df = load_csv(primary_n10k)
        with open(str(primary_n10k).replace(".csv", ".summary.json")) as fh:
            summary = json.load(fh)
        four_panel(
            df,
            "Hypercube sweep — SIFT1M N=10K, sel=1%, probe seed=32",
            summary,
            HERE / "hypercube_n10k.png",
        )
    if primary_n100k.exists():
        df = load_csv(primary_n100k)
        with open(str(primary_n100k).replace(".csv", ".summary.json")) as fh:
            summary = json.load(fh)
        four_panel(
            df,
            "Hypercube sweep — SIFT1M N=100K, sel=1%, probe seed=32",
            summary,
            HERE / "hypercube_n100k.png",
        )
    recall_vs_io_scatter(HERE / "recall_vs_io.png")
    selectivity_scan(HERE / "selectivity_scan.png")
    storage_cost(HERE / "storage_cost.png")
    audit_check(HERE / "audit_table.png")


if __name__ == "__main__":
    main()
