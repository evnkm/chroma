# CLAUDE.md — Adaptive `nprobe` for SPANN in Chroma

## Project Overview

**Goal:** In Chroma's SPANN inverted-index vector search, make `nprobe` (the
number of centroids probed per query) **adapt to the metadata filter** so
filtered-query recall stays high without paying the full cost on unfiltered
queries. This is the "sibling" of Hammad's per-centroid-metadata-statistics
idea — simpler to land, composes with it later.

**Workstream owner:** Evan Kim (this CLAUDE.md / project directory).
**Team:** Kartik Pingle, Zach Marinov take other SPANN workstreams (TBD).
**Advisor:** Prof. Tianyu. **Midterm due:** 2026-04-15.

The master plan lives at `PLANS/spann-plan.md`. The original proposal is
`PLANS/DB-project-proposal.md` (top block = Hammad call notes; bottom block =
the written proposal submitted earlier).

---

## What changed (important)

Earlier in this project we started instrumenting the Python `LocalExecutor`
for an HNSW pre/post-filter planner. That work lives on the
`hnsw-baseline-archive` branch and serves only as **baseline motivation**
(shows the general filtered-ANN tradeoff). The actual project is SPANN-side
and lives in Rust.

Do NOT revive the Python planner path unless explicitly asked. Expect all new
implementation to land in Rust.

---

## Where SPANN lives in the codebase

| Purpose | Path | Notes |
|---|---|---|
| SPANN KNN orchestration | `rust/worker/src/execution/orchestration/spann_knn.rs` | main target |
| Quantized variant | `rust/worker/src/execution/orchestration/quantized_spann_knn.rs` | second target |
| Centroid selection operator | `rust/worker/src/execution/operators/spann_centers_search.rs` | `nprobe` lives here (likely) |
| Posting-list fetch | `rust/worker/src/execution/operators/spann_fetch_pl.rs` | |
| Brute-force within posting list | `rust/worker/src/execution/operators/spann_bf_pl.rs` | |
| KNN filter orchestration | `rust/worker/src/execution/orchestration/knn_filter.rs` | metadata bitmask wiring |
| SPANN index core | `rust/index/src/spann/` | centroid store, posting lists |
| Metadata index core | `rust/index/src/metadata/` | selectivity counts live here |

All SPANN work requires the **distributed stack up** (frontend + worker +
sysdb + log). The Python `SegmentAPI` path does not exercise SPANN.

---

## The Three Strategies (reframed for SPANN)

1. **Fixed `nprobe` (today's behavior):** same `nprobe` regardless of filter.
   - Fails: low-selectivity filter → all candidates in probed centroids may be
     masked out; top-k falls below k; recall tanks.

2. **Adaptive `nprobe` (this workstream):** raise `nprobe` when the filter is
   restrictive so enough unfiltered candidates remain after masking.
   - Rule-of-thumb: `nprobe' = clip(N / max(sel, eps), N, N * MAX_FACTOR)`.
   - Needs a cheap selectivity estimate from the metadata index before
     centroid search runs.

3. **Per-centroid metadata statistics (Hammad, sibling workstream):** use
   stored per-centroid metadata histograms to probe the *right* centroids,
   not just more of them. Better pruning; more implementation work.
   Composes with adaptive `nprobe`.

---

## Development Environment

```bash
# Activate the uv-managed venv (there is no plain `venv/`)
source .venv/bin/activate

# Python bindings for running the Rust path locally. Required for SPANN work.
# macOS Command Line Tools quirk: export SDK include paths first, else
# hnswlib's C++ headers fail to resolve.
export SDKROOT="/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"
export CPLUS_INCLUDE_PATH="$SDKROOT/usr/include/c++/v1:$SDKROOT/usr/include"
export C_INCLUDE_PATH="$SDKROOT/usr/include"
. "$HOME/.cargo/env"
maturin dev     # builds chromadb_rust_bindings; ~5 min cold, 2 min warm
```

Python version: **3.11** (see `.python-version`). pip is NOT installed in the
uv venv — use `uv pip ...` for package operations.

### Running the full distributed stack

For end-to-end SPANN queries from Python, the worker + sysdb + log services
need to be up. Look at `Tiltfile` (preferred dev loop) or
`docker-compose.yml`. **Not yet exercised in this project — verify against
the repo's DEVELOP.md before trying.**

### Tests
- Rust: `cargo test -p chroma-worker` for orchestration, `-p chroma-index`
  for index math. SPANN-specific tests are typically in `rust/worker/tests/`
  and `rust/index/tests/`.
- Python integration: see `chromadb/test/` — most of these hit the Rust path
  through `RustBindingsAPI` by default.

---

## Key Instrumentation Points (SPANN)

To build the cost model, we want per-phase timing + candidate counts on:

1. **`spann_centers_search`** — how many centroids, how long.
2. **`spann_fetch_pl`** — bytes read per centroid, total candidates.
3. **Metadata bitmask apply** — where in the flow (orchestrator or operator),
   how many survive.
4. **`spann_bf_pl`** — brute-force work on surviving candidates.
5. **Final top-k** — how many items actually returned vs. requested.

Instrumentation should be gated behind a tracing flag or env var so normal
queries pay zero cost. The Python-side analogue we built for HNSW
(`chromadb/execution/executor/profiling.py` on the `hnsw-baseline-archive`
branch) is a useful template.

---

## Benchmark Setup

Datasets:
- **Synthetic (primary for midterm):** random float32 embeddings; synthetic
  metadata with tunable selectivity (`bucket` label, 10 buckets, bucket 0
  hits `P = target_selectivity`).
- **MS MARCO passages (secondary, post-midterm):** real MiniLM embeddings.
  Loader cached as `.npz`; see `hnsw-baseline-archive` branch
  `PLANS/benchmarks/data/msmarco.py` for the existing version — it's reusable
  once we've moved benchmarks to the Rust-worker path.

Metrics: filtered recall@k vs. a ground-truth exhaustive run; end-to-end
latency; per-phase latency; `nprobe` actually used; planning overhead.

Sweep axes: selectivity, collection size, embedding dim, `nprobe_base`.

---

## Timeline

| Milestone | Date | Tasks |
|---|---|---|
| Mid-term report | 2026-04-15 | SPANN-path map, measure recall collapse on stock SPANN, sketch adaptive-nprobe mechanism + cost model. Include HNSW baseline numbers as preamble. |
| Project presentation | ~2026-05 | Adaptive-nprobe implemented; synthetic sweeps show recall/latency win. |
| Final hand-in | ~2026-05 | MS MARCO eval, cost-model refinement, writeup, PR to Chroma. |

---

## Gotchas (learned the hard way)

- Default `chroma_api_impl` is `RustBindingsAPI`, not `SegmentAPI`. Clients
  created via `chromadb.Client()` go through the Rust worker path — which is
  what we want for SPANN, but note that any experiment needs the full stack.
- `cargo clean` frees ~20 GB. Safe when not actively iterating on Rust.
- Collection names must be 3–512 chars from `[a-zA-Z0-9._-]`, starting/ending
  with alnum. Short dummy names like `'t'` fail validation.
- BEIR/msmarco streaming can stall past a few thousand rows while fetching
  new shards. For large sweeps, pre-materialize the subset once and cache.
