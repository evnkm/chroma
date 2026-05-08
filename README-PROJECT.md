# Filtered ANN at scale in Chroma's SPANN

A 6.5830 project on top of a Chroma fork. We add three orthogonal optimizations
to Chroma's SPANN inverted index — **adaptive `nprobe`**, **per-head bloom
filters**, and **per-head metadata synopses** — and study the full 8-cell
on/off hypercube on SIFT1M and a side-by-side demo on MS MARCO.

This file covers how to run the demo and the benchmarks.

---

## What's in the repo

| Path | What it is |
|---|---|
| `rust/index/src/spann/head_bloom.rs` | per-head bloom filter (gate-1) |
| `rust/index/src/spann/head_synopsis.rs` | per-head metadata synopsis (gate-2) |
| `rust/index/src/spann/types.rs::SpannIndexReader::determine_search_nprobe` | filter-aware adaptive `nprobe` |
| `rust/worker/benches/spann_full_sweep.rs` | the 8-cell SIFT1M sweep |
| `rust/worker/benches/spann_*.rs` | per-feature ablations |
| `rust/worker/src/bin/demo/spann_demo*.rs` | demo binaries (build + race REPL) |
| `demo/` | Python TUI + MS MARCO loaders for the live demo |
| `LOGS_PLANS/benchmarks/` | CSV outputs, plots, and `plot.py` scripts |

---

## Prerequisites

```bash
# macOS only — Command Line Tools include paths must be exported before
# anything pulls in hnswlib's C++ headers.
export SDKROOT="/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"
export CPLUS_INCLUDE_PATH="$SDKROOT/usr/include/c++/v1:$SDKROOT/usr/include"
export C_INCLUDE_PATH="$SDKROOT/usr/include"

. "$HOME/.cargo/env"
export CARGO_TARGET_DIR=/tmp/chroma-target   # share build artifacts; ~20 GB

# Python venv (uv-managed, Python 3.11). pip is NOT installed — use `uv pip ...`.
source .venv/bin/activate
```

---

## Running the demo

The demo is a side-by-side TUI race between two SPANN indexes built from the
**same** MS MARCO 100K corpus and the **same** metadata predicate, but with
different feature toggles:

| Cell | Adaptive nprobe | Bloom-gate | Synopsis-gate |
|------|-----------------|------------|----------------|
| **Baseline (000)**  | off | off | off |
| **Optimized (111)** | on  | on  | on  |

For each free-text query + predicate, the demo embeds the query, races both
cells in parallel, and the TUI streams a live side-by-side stage timeline.

### One-time setup

```bash
# 1. Build the demo binaries (~5 min cold).
cargo build --release -p worker --bin spann_demo --bin spann_demo_build

# 2. Cache MS MARCO 100K + embed with MiniLM (~3 min).
python demo/build_msmarco_demo_cache.py --n 100000 --queries 30 --buckets 1000
python demo/convert_cache_for_rust.py

# 3. Build the two SPANN indexes (~10 min).
/tmp/chroma-target/release/spann_demo_build
```

Output lands in `demo/data/`: the corpus, two index dirs (`baseline/`,
`optimized/`), and `queries.json`.

### Run the TUI

```bash
python demo/tui.py
```

Type a free-text query and pick a predicate; both indexes race. Useful
predicates already wired up:

- `source_domain = "wikihow.com"` — 0.74% selectivity (headline realistic)
- `source_domain = "mayoclinic.org"` — 0.62%
- `source_domain = "investopedia.com"` — 0.28%
- `topic_bucket = 0` — 0.10% synthetic (matches the v2 paper headline)
- `query_type = "PERSON"` / `"LOCATION"` / `"ENTITY"` — 2.86% / 4.36% / 9.18%

A non-interactive smoke test of the protocol: `python demo/test_demo_protocol.py`.
A pre-recorded GIF/MP4 of a run lives under `demo/recordings/`.

---

## Running the benchmarks

All benches are Rust criterion-style targets under
`rust/worker/benches/`. The headline one is the 8-cell hypercube sweep
(`spann_full_sweep`); the rest are per-feature ablations driven the same way.

### The 8-cell hypercube — `spann_full_sweep`

Builds **one** unified SPANN index with both bloom and synopsis enabled and
evaluates all 8 on/off combinations of (adaptive_nprobe, bloom, synopsis) at
read time, so cross-cell comparisons are apples-to-apples on the same
posting lists.

```bash
BLOOM_DOC_TOKENS_CACHE=1 BLOOM_COMMIT_REBUILD=1 \
SYNOPSIS_TOP_K=64 SYNOPSIS_MAX_CARD=1024 \
BENCH_N_RECORDS=100000 BENCH_N_QUERIES=50 BENCH_BUCKETS=100 BENCH_PROBE_NBR=32 \
cargo bench -p worker --bench spann_full_sweep
```

Knobs:

| Env var | Default | Meaning |
|---|---|---|
| `BENCH_N_RECORDS` | `10000` | corpus size (we sweep `{10K, 100K, 1M}`) |
| `BENCH_N_QUERIES` | `50` | queries per cell |
| `BENCH_K` | `10` | top-k |
| `BENCH_BUCKETS` | `100` | metadata buckets — selectivity = `1 / BENCH_BUCKETS` |
| `BENCH_PROBE_NBR` | `32` | probe budget seed (before adaptive boost) |
| `BENCH_OUTPUT` | `LOGS_PLANS/benchmarks/full_sweep/...csv` | CSV destination |
| `FULL_AUDIT_NO_PANIC` | unset | unset = panic on a `bad_drop` (gate that drops a head with matching docs); we never set this |

The bench writes one row per `(cell, query)` to a CSV plus a `.summary.json`
per workload. The companion `plot.py` produces hypercube, recall-vs-I/O,
selectivity, and storage-cost figures from those CSVs:

```bash
python LOGS_PLANS/benchmarks/full_sweep_v3/plot.py
```

The CSV schema is shared across all SPANN benchmarks:

```
dataset, n_records, dim, query_id, k, selectivity, strategy,
base_nprobe, nprobe_used, returned_count, recall_at_k,
latency_ms, centers, candidates_before_filter, candidates_after_filter
```

### Per-feature ablations

Same invocation pattern; pick a different bench target:

```bash
cargo bench -p worker --bench spann_filtered_recall      # adaptive vs. fixed nprobe baseline
cargo bench -p worker --bench spann_bloom_ablation       # bloom on/off
cargo bench -p worker --bench spann_synopsis_ablation    # synopsis on/off
cargo bench -p worker --bench spann_bloom_sweep          # bloom params sweep
cargo bench -p worker --bench spann_bloom_adaptive_sweep # bloom × adaptive sweep
```

Each writes CSVs under `LOGS_PLANS/benchmarks/` (subdir varies — see the
`BENCH_OUTPUT` default in each `.rs` file).

---

## Datasets

### SIFT1M (primary benchmark dataset)

1M 128-dim float vectors, the classical ANN benchmark. We use random subsets
of `{10K, 100K, 1M}`. Loaded via the existing
[`Sift1MData`](rust/benchmark/src/datasets/sift.rs) loader; the loader fetches
on first use and caches under `rust/benchmark/dataset_files/`.

Metadata is **synthetic**: each point gets `id % BENCH_BUCKETS` as a single
integer label, so bucket-0 hits a fraction `1 / BENCH_BUCKETS` of the
corpus. We sweep `BENCH_BUCKETS ∈ {100, 1000}` (selectivity `1%` and `0.1%`).

### MS MARCO 100K (demo dataset)

100K real passages embedded with MiniLM (384-dim). Built once via
`demo/build_msmarco_demo_cache.py`, which downloads from HuggingFace,
embeds, and writes a `.npz` cache. `convert_cache_for_rust.py` flattens it
into a `data.bin + meta.json` layout that
[`msmarco_demo`](rust/benchmark/src/datasets/msmarco_demo.rs) loads.

Metadata is **real** — derived from the `url` field (`source_domain`),
length bucket, and a coarse query-type tag. Used only for the live demo
because the bucket label is naturally non-uniform and produces realistic
low-selectivity predicates.

### Why two datasets

SIFT1M gives a controllable selectivity knob and is what the benchmark
suite measures; MS MARCO with `source_domain` predicates gives a
realistic demo where the recall collapse at low selectivity is visibly
compelling.
