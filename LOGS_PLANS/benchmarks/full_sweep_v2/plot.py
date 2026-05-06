"""Plot the v2 full-sweep matrix: N ∈ {10K, 100K, 1M} × sel ∈ {1%, 0.1%}.

Re-tuned adaptive parameters (MAX_FACTOR=16, size_based 32/64/128 at
100K/1M/>1M, floor over params.search_nprobe).

Reads CSVs from this directory. Produces:

  hypercube_<N>_<sel>.png         — 4-panel summary, 6 of these.
  recall_vs_io_v2.png             — recall vs heads_fetched, all 6 workloads.
  scaling_with_N.png              — recall, drop_ratio, latency vs N (rows: sel).
  selectivity_compare.png         — sel=1% vs sel=0.1% side-by-side per cell.
  storage_cost_v2.png             — blob bytes + build time across 3 scales.
  audit_table_v2.png              — bad_drops heatmap; must be all-zero.
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
    return (
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


def parse_filename(path: Path):
    m = re.match(r"sift1m-full-n(\d+)-b(\d+)\.csv$", path.name)
    if not m:
        return None
    n_records = int(m.group(1))
    buckets = int(m.group(2))
    return n_records, buckets


def four_panel(df: pd.DataFrame, title: str, out_path: Path):
    agg = aggregate_per_cell(df)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

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

    ax = axes[1, 0]
    x = np.arange(len(agg))
    w = 0.4
    ax.bar(
        x - w / 2,
        agg["latency_mean"],
        w,
        label="mean",
        color=[CELL_LABELS[c][1] for c in agg.index],
    )
    ax.bar(
        x + w / 2,
        agg["latency_p99"],
        w,
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
    ax.set_ylim(0, 1.05)
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


def all_workloads():
    """Yield (n_records, buckets, sel, df, agg) for every CSV in the dir."""
    out = []
    for path in sorted(HERE.glob("sift1m-full-n*-b*.csv")):
        meta = parse_filename(path)
        if not meta:
            continue
        n, b = meta
        sel = 1.0 / b
        df = load_csv(path)
        agg = aggregate_per_cell(df)
        out.append((n, b, sel, df, agg))
    return out


def recall_vs_io_v2(out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 7))
    markers = {0.01: ("o", 12), 0.001: ("D", 11)}
    sizes_marker = {10000: 1.0, 100000: 1.6, 1000000: 2.4}
    for n, b, sel, _, agg in all_workloads():
        m, base_size = markers.get(sel, ("s", 11))
        sz = base_size * sizes_marker.get(n, 1.0)
        for cell, row in agg.iterrows():
            color = CELL_LABELS[cell][1]
            ax.scatter(
                row["heads_fetched"],
                row["recall"],
                s=sz ** 2,
                marker=m,
                color=color,
                edgecolors="black",
                linewidth=0.6,
                alpha=0.85,
            )
            ax.annotate(
                cell,
                (row["heads_fetched"], row["recall"]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=7,
            )
    cell_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=col, markersize=10, label=lbl)
        for code, (lbl, col) in CELL_LABELS.items()
    ]
    sel_handles = [
        plt.Line2D([0], [0], marker=m, color="w", markerfacecolor="#aaa",
                   markersize=10, markeredgecolor="black",
                   label=f"sel={s*100:.1f}%")
        for s, (m, _) in markers.items()
    ]
    ax.legend(handles=cell_handles + sel_handles, fontsize=7, loc="lower right", ncol=2)
    ax.set_xlabel("heads_fetched (mean per query, log scale)")
    ax.set_ylabel("recall@10 (mean)")
    ax.set_title("Recall vs I/O across the full N × selectivity matrix\n"
                 "(marker size encodes N; circle=sel 1%, diamond=sel 0.1%)")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path.name}")


def scaling_with_N(out_path: Path):
    """Per selectivity, plot recall, drop_ratio, latency vs N; one row per sel."""
    rows = []
    for n, b, sel, _, agg in all_workloads():
        for cell, r in agg.iterrows():
            rows.append(
                {
                    "n_records": n,
                    "selectivity": sel,
                    "cell": cell,
                    "recall": r["recall"],
                    "drop_ratio": r["drop_ratio"],
                    "heads_fetched": r["heads_fetched"],
                    "latency": r["latency_mean"],
                }
            )
    df = pd.DataFrame(rows)
    sels = sorted(df["selectivity"].unique(), reverse=True)
    fig, axes = plt.subplots(len(sels), 3, figsize=(16, 4.5 * len(sels)))
    if len(sels) == 1:
        axes = axes.reshape(1, -1)
    for r, sel in enumerate(sels):
        sub = df[df["selectivity"] == sel]
        # recall
        ax = axes[r, 0]
        for cell in CELL_ORDER:
            d = sub[sub["cell"] == cell].sort_values("n_records")
            if d.empty:
                continue
            ax.plot(d["n_records"], d["recall"], marker="o",
                    color=CELL_LABELS[cell][1],
                    label=CELL_LABELS[cell][0])
        ax.set_xscale("log")
        ax.set_xlabel("N (records)")
        ax.set_ylabel("recall@10")
        ax.set_title(f"Recall vs N — sel={sel*100:g}%")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        if r == 0:
            ax.legend(fontsize=7, loc="lower left", ncol=2)
        # drop_ratio
        ax = axes[r, 1]
        for cell in CELL_ORDER:
            d = sub[sub["cell"] == cell].sort_values("n_records")
            if d.empty:
                continue
            ax.plot(d["n_records"], d["drop_ratio"], marker="o",
                    color=CELL_LABELS[cell][1])
        ax.set_xscale("log")
        ax.set_xlabel("N (records)")
        ax.set_ylabel("drop_ratio")
        ax.set_title(f"Gate effectiveness — sel={sel*100:g}%")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        # heads_fetched
        ax = axes[r, 2]
        for cell in CELL_ORDER:
            d = sub[sub["cell"] == cell].sort_values("n_records")
            if d.empty:
                continue
            ax.plot(d["n_records"], d["heads_fetched"], marker="o",
                    color=CELL_LABELS[cell][1])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("N (records)")
        ax.set_ylabel("heads_fetched")
        ax.set_title(f"I/O cost — sel={sel*100:g}%")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path.name}")


def selectivity_compare(out_path: Path):
    """Per cell, compare metrics at sel=1% and sel=0.1%."""
    rows = []
    for n, b, sel, _, agg in all_workloads():
        for cell, r in agg.iterrows():
            rows.append(
                {
                    "n_records": n,
                    "selectivity": sel,
                    "cell": cell,
                    "recall": r["recall"],
                    "heads_fetched": r["heads_fetched"],
                    "drop_ratio": r["drop_ratio"],
                }
            )
    df = pd.DataFrame(rows)
    sels = sorted(df["selectivity"].unique(), reverse=True)
    if len(sels) < 2:
        return
    sizes = sorted(df["n_records"].unique())
    fig, axes = plt.subplots(len(sizes), 2, figsize=(14, 4.5 * len(sizes)))
    if len(sizes) == 1:
        axes = axes.reshape(1, -1)
    for i, n in enumerate(sizes):
        # Recall side-by-side
        ax = axes[i, 0]
        x = np.arange(len(CELL_ORDER))
        w = 0.35
        for j, sel in enumerate(sels):
            sub = df[(df["n_records"] == n) & (df["selectivity"] == sel)]
            sub = sub.set_index("cell").reindex(CELL_ORDER)
            offset = (j - (len(sels) - 1) / 2) * w
            ax.bar(
                x + offset,
                sub["recall"],
                w,
                label=f"sel={sel*100:g}%",
                color=[CELL_LABELS[c][1] for c in CELL_ORDER],
                alpha=0.6 + 0.4 * j,
                edgecolor="black",
                linewidth=0.5,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([CELL_LABELS[c][0] for c in CELL_ORDER], rotation=30, ha="right")
        ax.set_ylabel("recall@10")
        ax.set_title(f"N={n//1000}K — recall by cell × selectivity")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=8)
        # heads_fetched side-by-side
        ax = axes[i, 1]
        for j, sel in enumerate(sels):
            sub = df[(df["n_records"] == n) & (df["selectivity"] == sel)]
            sub = sub.set_index("cell").reindex(CELL_ORDER)
            offset = (j - (len(sels) - 1) / 2) * w
            ax.bar(
                x + offset,
                sub["heads_fetched"],
                w,
                label=f"sel={sel*100:g}%",
                color=[CELL_LABELS[c][1] for c in CELL_ORDER],
                alpha=0.6 + 0.4 * j,
                edgecolor="black",
                linewidth=0.5,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([CELL_LABELS[c][0] for c in CELL_ORDER], rotation=30, ha="right")
        ax.set_ylabel("heads_fetched")
        ax.set_title(f"N={n//1000}K — I/O by cell × selectivity")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path.name}")


def storage_cost_v2(out_path: Path):
    summaries = []
    for f in HERE.glob("*.summary.json"):
        with open(f) as fh:
            data = json.load(fh)
        for flavor, meta in data["index_metadata"].items():
            summaries.append(
                {
                    "n_records": data["n_records"],
                    "buckets": data["buckets"],
                    "selectivity": data.get("selectivity", 1.0 / data["buckets"]),
                    "flavor": flavor,
                    "blob_bytes": meta["blob_storage_bytes"],
                    "build_ms": meta["build_ms"],
                }
            )
    if not summaries:
        return
    sdf = pd.DataFrame(summaries)
    # For storage: take the max blob size across selectivities (they're
    # approximately equal; differ only because of bucket count effect on
    # synopsis storage).
    flavor_colors = {"none": "#bbb", "bloom": "#ff7f0e", "synopsis": "#2ca02c", "both": "#d62728"}
    # Aggregate to (n_records, flavor) → max blob, mean build_ms.
    grouped = (
        sdf.groupby(["n_records", "flavor"])
        .agg(blob_bytes=("blob_bytes", "max"), build_ms=("build_ms", "mean"))
        .reset_index()
    )
    n_values = sorted(grouped["n_records"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    x = np.arange(len(n_values))
    width = 0.2
    flavors = ["none", "bloom", "synopsis", "both"]
    for i, flavor in enumerate(flavors):
        sub = grouped[grouped["flavor"] == flavor].set_index("n_records").reindex(n_values)
        bytes_per_n = sub["blob_bytes"].fillna(0).tolist()
        ax.bar(
            x + (i - 1.5) * width,
            [v / 1e6 for v in bytes_per_n],
            width,
            color=flavor_colors[flavor],
            label=flavor,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n//1000}K" if n < 1_000_000 else f"{n//1_000_000}M" for n in n_values])
    ax.set_xlabel("collection size (n_records)")
    ax.set_ylabel("blob storage (MB)")
    ax.set_title("Gate blob storage by writer flavor")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_yscale("log")

    ax = axes[1]
    for i, flavor in enumerate(flavors):
        sub = grouped[grouped["flavor"] == flavor].sort_values("n_records")
        ax.plot(
            sub["n_records"],
            sub["build_ms"] / 1000.0,
            marker="o",
            color=flavor_colors[flavor],
            label=flavor,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
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
    rows = []
    for n, b, sel, _, agg in all_workloads():
        for cell, r in agg.iterrows():
            rows.append(
                {
                    "label": f"N={n//1000 if n<1_000_000 else f'{n//1_000_000}M'}, sel={sel*100:.1f}%",
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
    fig, ax = plt.subplots(figsize=(11, max(3, 0.55 * len(pivot))))
    cmap = plt.cm.Reds
    img = ax.imshow(pivot.values, cmap=cmap, aspect="auto", vmin=0,
                    vmax=max(1, pivot.values.max()))
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c}\n{CELL_LABELS[c][0]}" for c in pivot.columns], fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(len(pivot)):
        for j in range(len(pivot.columns)):
            ax.text(
                j, i, str(int(pivot.values[i, j])),
                ha="center", va="center",
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
    workloads = all_workloads()
    if not workloads:
        print("No CSVs found.")
        return
    for n, b, sel, df, _ in workloads:
        n_label = f"{n//1000}K" if n < 1_000_000 else f"{n//1_000_000}M"
        title = f"Hypercube — SIFT1M N={n_label}, sel={sel*100:g}%, probe seed=32, MAX_FACTOR=16"
        out = HERE / f"hypercube_n{n}_sel{int(round(1/sel))}.png"
        four_panel(df, title, out)
    recall_vs_io_v2(HERE / "recall_vs_io_v2.png")
    scaling_with_N(HERE / "scaling_with_N.png")
    selectivity_compare(HERE / "selectivity_compare.png")
    storage_cost_v2(HERE / "storage_cost_v2.png")
    audit_check(HERE / "audit_table_v2.png")


if __name__ == "__main__":
    main()
