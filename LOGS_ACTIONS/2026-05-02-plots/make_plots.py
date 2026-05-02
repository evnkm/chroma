#!/usr/bin/env python3
"""
Generate plots summarising the bloom-filter recall study.
Reads CSVs from LOGS_PLANS/benchmarks/recall_study/, writes PNGs alongside
this script.

Schema (long-form, one row per (query, condition)):
  dataset, n_records, dim, query_id, k, selectivity,
  scenario, probe_nbr_off, probe_nbr_used,
  heads_rng, heads_fetched, drop_ratio,
  recall_at_k, bad_drops, latency_ms

scenario ∈ {bloom_off, bloom_on_iso_probe, bloom_on_iso_io}
"""

from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = Path(__file__).resolve().parent.parent.parent
CSV_DIR = ROOT / "LOGS_PLANS" / "benchmarks" / "recall_study"
OUT_DIR = Path(__file__).resolve().parent

mpl.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 130,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "bloom_off":          "#888888",
    "bloom_on_iso_probe": "#1f77b4",
    "bloom_on_iso_io":    "#d62728",
}
LABELS = {
    "bloom_off":          "bloom off (baseline)",
    "bloom_on_iso_probe": "bloom on, iso-probe",
    "bloom_on_iso_io":    "bloom on, iso-I/O",
}
MARKERS = {
    "bloom_off":          "o",
    "bloom_on_iso_probe": "s",
    "bloom_on_iso_io":    "D",
}


def load(name: str) -> pd.DataFrame:
    p = CSV_DIR / name
    if not p.exists():
        sys.stderr.write(f"missing {p}\n")
        return pd.DataFrame()
    return pd.read_csv(p)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Average over query_id for each (scenario, probe_nbr_off)."""
    if df.empty:
        return df
    return (
        df.groupby(["scenario", "probe_nbr_off"], as_index=False)
          .agg(
              recall=("recall_at_k", "mean"),
              recall_std=("recall_at_k", "std"),
              heads_rng=("heads_rng", "mean"),
              heads_fetched=("heads_fetched", "mean"),
              drop_ratio=("drop_ratio", "mean"),
              latency_ms=("latency_ms", "mean"),
              bad_drops=("bad_drops", "sum"),
              n=("recall_at_k", "size"),
          )
          .sort_values(["scenario", "probe_nbr_off"])
    )


def fig_recall_vs_probe(df: pd.DataFrame, title: str, out: str):
    s = summarise(df)
    if s.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    # Plot bloom_off underneath as a thick grey line; iso_probe sits on top
    # as a dashed blue line so the viewer sees they overlap exactly.
    style = {
        "bloom_off":          dict(lw=4.0, ls="-",  alpha=0.6, ms=10),
        "bloom_on_iso_probe": dict(lw=1.7, ls="--", alpha=1.0, ms=6),
        "bloom_on_iso_io":    dict(lw=2.3, ls="-",  alpha=1.0, ms=7),
    }
    for sc in ["bloom_off", "bloom_on_iso_probe", "bloom_on_iso_io"]:
        rows = s[s.scenario == sc]
        if rows.empty:
            continue
        ax.plot(
            rows.probe_nbr_off, rows.recall,
            color=COLORS[sc], marker=MARKERS[sc],
            label=LABELS[sc],
            **style[sc],
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(s.probe_nbr_off.unique())
    ax.set_xticklabels([str(int(x)) for x in s.probe_nbr_off.unique()])
    ax.set_xlabel("probe_nbr (centroids visited by HNSW)")
    ax.set_ylabel("recall@10")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(title)
    ax.grid(True, axis="both", alpha=0.25)
    ax.legend(loc="lower right")
    # Annotate the off / iso-probe overlap.
    ax.text(
        0.02, 0.98,
        "Note: 'bloom off' and 'iso-probe' overlap exactly\n"
        "(gate is one-sided → cannot reduce recall).",
        transform=ax.transAxes, va="top", ha="left",
        fontsize=9, alpha=0.7,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray", alpha=0.85),
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / out)
    plt.close(fig)
    print(f"wrote {out}")


def fig_recall_vs_fetches(df: pd.DataFrame, title: str, out: str):
    """Plot recall vs heads_fetched (the actual I/O the user pays for)."""
    s = summarise(df)
    if s.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for sc in ["bloom_off", "bloom_on_iso_probe", "bloom_on_iso_io"]:
        rows = s[s.scenario == sc].sort_values("heads_fetched")
        if rows.empty:
            continue
        ax.plot(
            rows.heads_fetched, rows.recall,
            color=COLORS[sc], marker=MARKERS[sc], lw=2,
            label=LABELS[sc],
        )
    ax.set_xscale("log")
    ax.set_xlabel("heads_fetched (mean PL reads per query)")
    ax.set_ylabel("recall@10")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(title)
    ax.grid(True, axis="both", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / out)
    plt.close(fig)
    print(f"wrote {out}")


def fig_selectivity_grid(out: str):
    """Recall vs probe_nbr, one line per selectivity, two panels (off/on)."""
    cells = [
        (10,    "selectivity 10%"),
        (100,   "selectivity 1%"),
        (1000,  "selectivity 0.1%"),
        (10000, "selectivity 0.01%"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    sel_colors = plt.cm.viridis([0.15, 0.40, 0.65, 0.90])
    for buckets, label in cells:
        df = load(f"sweep_selectivity_n50k_b{buckets}.csv")
        s = summarise(df)
        if s.empty:
            continue
        c = sel_colors[[c for c, _ in cells].index(buckets)]
        for ax, sc in zip(axes, ["bloom_off", "bloom_on_iso_io"]):
            rows = s[s.scenario == sc].sort_values("probe_nbr_off")
            ax.plot(
                rows.probe_nbr_off, rows.recall,
                color=c, marker="o", lw=2, label=label,
            )
    for ax, sc in zip(axes, ["bloom off", "bloom on (iso-I/O)"]):
        ax.set_xscale("log", base=2)
        xs = sorted(load("sweep_selectivity_n50k_b100.csv").probe_nbr_off.unique())
        ax.set_xticks(xs)
        ax.set_xticklabels([str(int(x)) for x in xs])
        ax.set_xlabel("probe_nbr")
        ax.set_title(sc)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("recall@10")
    axes[1].legend(loc="lower right", title="filter selectivity")
    fig.suptitle("Recall vs probe budget across filter selectivities (SIFT1M, n=50K)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / out)
    plt.close(fig)
    print(f"wrote {out}")


def fig_drop_ratio(out: str):
    """drop_ratio across selectivities at fixed probe."""
    rows = []
    for buckets in [10, 100, 1000, 10000]:
        df = load(f"sweep_selectivity_n50k_b{buckets}.csv")
        if df.empty:
            continue
        s = summarise(df)
        on = s[(s.scenario == "bloom_on_iso_probe") & (s.probe_nbr_off == 32)]
        if on.empty:
            continue
        rows.append({
            "selectivity": 1.0 / buckets,
            "buckets": buckets,
            "drop_ratio": float(on.drop_ratio.iloc[0]),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(df.selectivity, df.drop_ratio, "o-", color="#d62728", lw=2)
    for _, r in df.iterrows():
        ax.annotate(f"{r.drop_ratio:.3f}", (r.selectivity, r.drop_ratio),
                    textcoords="offset points", xytext=(8, -3))
    ax.set_xscale("log")
    ax.set_xlabel("filter selectivity (matches / collection)")
    ax.set_ylabel("drop_ratio (gate-dropped heads / probed heads)")
    ax.set_title("Gate effectiveness vs filter selectivity (probe_nbr=32)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25)
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(OUT_DIR / out)
    plt.close(fig)
    print(f"wrote {out}")


def fig_size_sweep(out: str):
    """Recall vs probe_nbr at three index sizes (1% selectivity)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    sizes = [(10000, "n=10 000"), (50000, "n=50 000"), (100000, "n=100 000")]
    style = {
        "bloom_off":          dict(lw=4.0, ls="-",  alpha=0.6, ms=10),
        "bloom_on_iso_probe": dict(lw=1.7, ls="--", alpha=1.0, ms=6),
        "bloom_on_iso_io":    dict(lw=2.3, ls="-",  alpha=1.0, ms=7),
    }
    for ax, (n, label) in zip(axes, sizes):
        df = load(f"sweep_size_n{n}_b100.csv")
        s = summarise(df)
        if s.empty:
            continue
        for sc in ["bloom_off", "bloom_on_iso_probe", "bloom_on_iso_io"]:
            rows = s[s.scenario == sc].sort_values("probe_nbr_off")
            if rows.empty:
                continue
            ax.plot(
                rows.probe_nbr_off, rows.recall,
                color=COLORS[sc], marker=MARKERS[sc],
                label=LABELS[sc],
                **style[sc],
            )
        ax.set_xscale("log", base=2)
        xs = sorted(df.probe_nbr_off.unique())
        ax.set_xticks(xs)
        ax.set_xticklabels([str(int(x)) for x in xs])
        ax.set_xlabel("probe_nbr")
        ax.set_title(label)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("recall@10")
    axes[2].legend(loc="lower right")
    fig.suptitle("Recall scaling across index sizes (1% selectivity)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / out)
    plt.close(fig)
    print(f"wrote {out}")


def fig_maintenance(out: str):
    """drop_ratio + recall@iso-IO across maintenance modes."""
    modes = [
        ("mvp",  "Phase 1 MVP"),
        ("opt1", "Option 1 only"),
        ("opt2", "Option 2 only"),
        ("both", "Both"),
    ]
    rows = []
    for tag, label in modes:
        df = load(f"sweep_maintenance_{tag}_n50k_b100.csv")
        s = summarise(df)
        if s.empty:
            continue
        for sc in s.scenario.unique():
            for pn in s.probe_nbr_off.unique():
                r = s[(s.scenario == sc) & (s.probe_nbr_off == pn)]
                if r.empty:
                    continue
                rows.append({
                    "mode": label, "scenario": sc, "probe_nbr": pn,
                    "drop_ratio": float(r.drop_ratio.iloc[0]),
                    "recall": float(r.recall.iloc[0]),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: drop_ratio per mode at probe=32 (iso-probe condition shows the gate's actual drops)
    on = df[(df.scenario == "bloom_on_iso_probe") & (df.probe_nbr == 32)]
    on = on.set_index("mode").reindex([m for _, m in modes])
    axes[0].bar(on.index, on.drop_ratio, color=["#888888", "#888888", "#1f77b4", "#d62728"])
    for x, v in zip(range(len(on)), on.drop_ratio):
        axes[0].text(x, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    axes[0].set_ylabel("drop_ratio (probe_nbr=32, 1% selectivity)")
    axes[0].set_title("Gate effectiveness across maintenance modes")
    axes[0].set_ylim(0, 1.05)
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].grid(True, axis="y", alpha=0.25)

    # Right: bloom_off vs bloom_on_iso_io recall per mode at probe=32
    off = df[(df.scenario == "bloom_off") & (df.probe_nbr == 32)].set_index("mode")
    iso = df[(df.scenario == "bloom_on_iso_io") & (df.probe_nbr == 32)].set_index("mode")
    common = [m for _, m in modes if m in off.index and m in iso.index]
    x = range(len(common))
    width = 0.35
    axes[1].bar([i - width/2 for i in x], [off.loc[m, "recall"] for m in common],
                width=width, color="#888888", label="bloom off")
    axes[1].bar([i + width/2 for i in x], [iso.loc[m, "recall"] for m in common],
                width=width, color="#d62728", label="bloom on (iso-I/O)")
    for i, m in enumerate(common):
        axes[1].text(i - width/2, off.loc[m, "recall"] + 0.02,
                     f"{off.loc[m, 'recall']:.3f}", ha="center", fontsize=9)
        axes[1].text(i + width/2, iso.loc[m, "recall"] + 0.02,
                     f"{iso.loc[m, 'recall']:.3f}", ha="center", fontsize=9)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(common, rotation=15)
    axes[1].set_ylabel("recall@10 (probe_nbr=32, 1% selectivity)")
    axes[1].set_title("Recall lift at iso-I/O")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(loc="upper left")
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.suptitle("Maintenance-mode ablation (n=50K, buckets=100)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / out)
    plt.close(fig)
    print(f"wrote {out}")


def fig_iso_recall_io(out: str):
    """Iso-recall I/O comparison: at recall ≥ 0.95, how many PL fetches?"""
    df = load("sweep_highprobe_n50k_b100.csv")
    if df.empty:
        return
    s = summarise(df)
    targets = [0.50, 0.75, 0.90, 0.95, 0.99]
    rows = []
    for sc in ["bloom_off", "bloom_on_iso_io"]:
        sub = s[s.scenario == sc].sort_values("probe_nbr_off")
        for t in targets:
            hit = sub[sub.recall >= t]
            if hit.empty:
                continue
            r = hit.iloc[0]
            rows.append({
                "scenario": sc, "target": t,
                "heads_fetched": float(r.heads_fetched),
                "probe_nbr_off": int(r.probe_nbr_off),
            })
    rdf = pd.DataFrame(rows)
    if rdf.empty:
        return
    pivot = rdf.pivot_table(index="target", columns="scenario", values="heads_fetched")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    width = 0.35
    xs = range(len(pivot))
    if "bloom_off" in pivot.columns:
        ax.bar([x - width/2 for x in xs], pivot["bloom_off"].fillna(0),
               width=width, color="#888888", label="bloom off")
        for i, v in enumerate(pivot["bloom_off"]):
            if pd.notna(v):
                ax.text(i - width/2, v * 1.05, f"{v:.0f}", ha="center", fontsize=9)
    if "bloom_on_iso_io" in pivot.columns:
        ax.bar([x + width/2 for x in xs], pivot["bloom_on_iso_io"].fillna(0),
               width=width, color="#d62728", label="bloom on (iso-I/O)")
        for i, v in enumerate(pivot["bloom_on_iso_io"]):
            if pd.notna(v):
                ax.text(i + width/2, v * 1.05, f"{v:.0f}", ha="center", fontsize=9)
    ax.set_yscale("log")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"≥ {t:.2f}" for t in pivot.index])
    ax.set_xlabel("target recall@10")
    ax.set_ylabel("heads_fetched required (mean PL reads / query, log)")
    ax.set_title("I/O cost to hit each recall target (1% selectivity, n=50K)")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / out)
    plt.close(fig)
    print(f"wrote {out}")


def fig_baseline_climbs(out: str):
    """Baseline does reach high recall — just at much higher probe budgets."""
    df = load("sweep_highprobe_n50k_b100.csv")
    if df.empty:
        return
    s = summarise(df)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for sc in ["bloom_off", "bloom_on_iso_io"]:
        rows = s[s.scenario == sc].sort_values("probe_nbr_off")
        if rows.empty:
            continue
        ax.plot(
            rows.probe_nbr_off, rows.recall,
            color=COLORS[sc], marker=MARKERS[sc], lw=2,
            label=LABELS[sc],
        )

    # Annotate the recall=0.95 crossover.
    for sc, color in [("bloom_off", "#888888"), ("bloom_on_iso_io", "#d62728")]:
        sub = s[s.scenario == sc].sort_values("probe_nbr_off")
        hit = sub[sub.recall >= 0.95]
        if not hit.empty:
            x = float(hit.probe_nbr_off.iloc[0])
            y = float(hit.recall.iloc[0])
            ax.annotate(
                f"{int(x)} probes → r={y:.3f}",
                xy=(x, y), xytext=(x * 1.1, y - 0.15),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                color=color, fontsize=10,
            )

    ax.axhline(0.95, color="black", lw=0.7, ls=":", alpha=0.6)
    ax.text(s.probe_nbr_off.min() * 0.9, 0.96, "recall = 0.95", fontsize=9, alpha=0.7)
    ax.set_xscale("log", base=2)
    xs = sorted(s.probe_nbr_off.unique())
    ax.set_xticks(xs)
    ax.set_xticklabels([str(int(x)) for x in xs])
    ax.set_xlabel("probe_nbr (centroids visited by HNSW)")
    ax.set_ylabel("recall@10")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(
        "Baseline reaches near-perfect recall — but needs 16× more probing\n"
        "(SIFT1M, n=50K, 1% selectivity)"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / out)
    plt.close(fig)
    print(f"wrote {out}")


FOURWAY_COLORS = {
    "none":     "#888888",
    "adaptive": "#2ca02c",
    "bloom":    "#1f77b4",
    "both":     "#d62728",
}
FOURWAY_LABELS = {
    "none":     "neither (probe-blind)",
    "adaptive": "adaptive nprobe only",
    "bloom":    "bloom gate only",
    "both":     "adaptive + bloom",
}
FOURWAY_MARKERS = {
    "none":     "o",
    "adaptive": "^",
    "bloom":    "s",
    "both":     "D",
}


def summarise_4way(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return (
        df.groupby(["scenario", "probe_base"], as_index=False)
          .agg(
              recall=("recall_at_k", "mean"),
              probe_used=("probe_nbr_used", "mean"),
              heads_rng=("heads_rng", "mean"),
              heads_fetched=("heads_fetched", "mean"),
              drop_ratio=("drop_ratio", "mean"),
              latency_ms=("latency_ms", "mean"),
              bad_drops=("bad_drops", "sum"),
          )
          .sort_values(["scenario", "probe_base"])
    )


def fig_4way_recall_vs_probe(name: str, title: str, out: str):
    df = load(name)
    if df.empty:
        return
    s = summarise_4way(df)
    fig, ax = plt.subplots(figsize=(8, 5))
    for sc in ["none", "adaptive", "bloom", "both"]:
        rows = s[s.scenario == sc].sort_values("probe_base")
        if rows.empty:
            continue
        ax.plot(
            rows.probe_base, rows.recall,
            color=FOURWAY_COLORS[sc], marker=FOURWAY_MARKERS[sc], lw=2.2, ms=8,
            label=FOURWAY_LABELS[sc],
        )
    ax.set_xscale("log", base=2)
    xs = sorted(s.probe_base.unique())
    ax.set_xticks(xs)
    ax.set_xticklabels([str(int(x)) for x in xs])
    ax.set_xlabel("probe_base (HNSW probe budget before adaptive boost)")
    ax.set_ylabel("recall@10")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / out)
    plt.close(fig)
    print(f"wrote {out}")


def fig_4way_recall_vs_io(name: str, title: str, out: str):
    """The Pareto plot — recall vs actual PL fetches, four lines."""
    df = load(name)
    if df.empty:
        return
    s = summarise_4way(df)
    fig, ax = plt.subplots(figsize=(8, 5))
    for sc in ["none", "adaptive", "bloom", "both"]:
        rows = s[s.scenario == sc].sort_values("heads_fetched")
        if rows.empty:
            continue
        ax.plot(
            rows.heads_fetched, rows.recall,
            color=FOURWAY_COLORS[sc], marker=FOURWAY_MARKERS[sc], lw=2.2, ms=8,
            label=FOURWAY_LABELS[sc],
        )
    ax.set_xscale("log")
    ax.set_xlabel("heads_fetched (mean PL reads per query, log scale)")
    ax.set_ylabel("recall@10")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / out)
    plt.close(fig)
    print(f"wrote {out}")


def fig_4way_bars(name: str, probe_base: int, title: str, out: str):
    """Side-by-side bar at one operating point: recall and PL fetches."""
    df = load(name)
    if df.empty:
        return
    s = summarise_4way(df)
    s = s[s.probe_base == probe_base]
    if s.empty:
        return
    s = s.set_index("scenario").reindex(["none", "adaptive", "bloom", "both"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    colors = [FOURWAY_COLORS[sc] for sc in s.index]
    labels = [FOURWAY_LABELS[sc] for sc in s.index]

    axes[0].bar(labels, s.recall, color=colors)
    for x, v in zip(range(len(s)), s.recall):
        axes[0].text(x, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    axes[0].set_ylabel(f"recall@10 (probe_base={probe_base})")
    axes[0].set_ylim(0, 1.1)
    axes[0].set_title("Recall")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(labels, s.heads_fetched, color=colors)
    for x, v in zip(range(len(s)), s.heads_fetched):
        axes[1].text(x, v * 1.05, f"{v:.1f}", ha="center", fontsize=10)
    axes[1].set_ylabel(f"PL fetches per query (probe_base={probe_base})")
    axes[1].set_yscale("log")
    axes[1].set_title("I/O cost (PL fetches, log scale)")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / out)
    plt.close(fig)
    print(f"wrote {out}")


def fig_4way_iso_recall(name: str, title: str, out: str):
    """At each recall target, how many PL fetches does each strategy need?"""
    df = load(name)
    if df.empty:
        return
    s = summarise_4way(df)
    targets = [0.50, 0.75, 0.90, 0.95, 0.99]
    rows = []
    for sc in ["none", "adaptive", "bloom", "both"]:
        sub = s[s.scenario == sc].sort_values("probe_base")
        for t in targets:
            hit = sub[sub.recall >= t]
            if hit.empty:
                continue
            r = hit.iloc[0]
            rows.append({"scenario": sc, "target": t, "fetches": float(r.heads_fetched)})
    if not rows:
        return
    rdf = pd.DataFrame(rows)
    pivot = rdf.pivot_table(index="target", columns="scenario", values="fetches")

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.18
    xs = list(range(len(pivot)))
    offsets = [-1.5, -0.5, 0.5, 1.5]
    for sc, off in zip(["none", "adaptive", "bloom", "both"], offsets):
        if sc not in pivot.columns:
            continue
        vals = pivot[sc].values
        ax.bar(
            [x + off * width for x in xs], vals,
            width=width, color=FOURWAY_COLORS[sc],
            label=FOURWAY_LABELS[sc],
        )
        for i, v in enumerate(vals):
            if pd.notna(v):
                ax.text(i + off * width, v * 1.05, f"{v:.0f}", ha="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"≥ {t:.2f}" for t in pivot.index])
    ax.set_xlabel("target recall@10")
    ax.set_ylabel("PL fetches per query needed (log scale)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / out)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Per-selectivity recall vs probe_nbr
    fig_recall_vs_probe(
        load("sweep_selectivity_n50k_b100.csv"),
        "Recall vs probe budget — 1% selectivity (SIFT1M, n=50K)",
        "01_recall_vs_probe_sel1pct.png",
    )
    fig_recall_vs_probe(
        load("sweep_selectivity_n50k_b1000.csv"),
        "Recall vs probe budget — 0.1% selectivity (SIFT1M, n=50K)",
        "02_recall_vs_probe_sel01pct.png",
    )

    # Recall vs actual fetched PLs (the cost the user pays for)
    fig_recall_vs_fetches(
        load("sweep_highprobe_n50k_b100.csv"),
        "Recall vs PL fetches — 1% selectivity (SIFT1M, n=50K)",
        "03_recall_vs_fetches_sel1pct.png",
    )

    # 2x4 grid across selectivities (off vs on)
    fig_selectivity_grid("04_selectivity_grid.png")

    # drop_ratio vs selectivity
    fig_drop_ratio("05_drop_ratio_vs_selectivity.png")

    # Index size sweep
    fig_size_sweep("06_size_sweep.png")

    # Maintenance modes
    fig_maintenance("07_maintenance_modes.png")

    # Iso-recall I/O bar chart
    fig_iso_recall_io("08_iso_recall_io_cost.png")

    # The "Chroma can hit high recall too" plot
    fig_baseline_climbs("09_baseline_climbs.png")

    # 4-way feature stack-up plots: none / adaptive / bloom / both
    fig_4way_recall_vs_probe(
        "sweep_4way_n50k_b100.csv",
        "4-way feature stack-up — recall vs probe_base (1% selectivity, n=50K)",
        "10_4way_recall_vs_probe_sel1pct.png",
    )
    fig_4way_recall_vs_probe(
        "sweep_4way_n50k_b1000.csv",
        "4-way feature stack-up — recall vs probe_base (0.1% selectivity, n=50K)",
        "11_4way_recall_vs_probe_sel01pct.png",
    )
    fig_4way_recall_vs_io(
        "sweep_4way_n50k_b100.csv",
        "Recall vs I/O Pareto — 4-way comparison (1% selectivity, n=50K)",
        "12_4way_pareto_sel1pct.png",
    )
    fig_4way_recall_vs_io(
        "sweep_4way_n50k_b1000.csv",
        "Recall vs I/O Pareto — 4-way comparison (0.1% selectivity, n=50K)",
        "13_4way_pareto_sel01pct.png",
    )
    fig_4way_bars(
        "sweep_4way_n50k_b100.csv", probe_base=32,
        title="Single operating point: probe_base=32, 1% selectivity, n=50K",
        out="14_4way_bars_probe32_sel1pct.png",
    )
    fig_4way_bars(
        "sweep_4way_n50k_b1000.csv", probe_base=32,
        title="Single operating point: probe_base=32, 0.1% selectivity, n=50K",
        out="15_4way_bars_probe32_sel01pct.png",
    )
    fig_4way_iso_recall(
        "sweep_4way_n50k_b100.csv",
        "I/O cost to hit each recall target — 4-way (1% selectivity, n=50K)",
        "16_4way_iso_recall_io_sel1pct.png",
    )

    print("done.")


if __name__ == "__main__":
    main()
