# Final Project Getting Started

This document turns `PLANS/DB-project-proposal.md` into an executable project
plan for the current Chroma codebase. It assumes the project has pivoted from a
generic HNSW prefilter/postfilter planner to a narrower SPANN filtered-query
recall project.

## Scope in one paragraph

The final project should show that Chroma's SPANN filtered search can lose
recall because center selection does not know how restrictive the metadata
filter is. Chroma already builds a metadata filter bitmask before vector search,
but SPANN still chooses centers using only vector distance, then applies the
filter while brute-forcing posting lists. The minimum viable project is:

1. Measure filtered recall collapse in stock SPANN.
2. Implement or prototype an adaptive SPANN probing strategy.
3. Evaluate recall/latency tradeoffs across filter selectivity.
4. Write up when the approach helps and what it costs.

Do not try to build a full general-purpose prefilter/postfilter query planner
unless the SPANN path is already working and benchmarked. That planner was the
original proposal framing, but it is too broad for the remaining timeline.

## Are we doing prefiltering or postfiltering?

Not in the broad database-planner sense.

Chroma's Rust SPANN query path currently does this:

```text
metadata filter -> bitmask
query vector -> SPANN center search, ignoring the filter
selected centers -> posting lists
posting-list brute force -> skip ids excluded by the bitmask
merge -> final top-k
```

So the filter is evaluated before vector search, but the SPANN candidate
generation step is not filter-aware. The practical project scope is to make
SPANN candidate generation filter-aware, starting with adaptive `nprobe`.

Useful terminology for the report:

- `Current Chroma SPANN`: precomputes a filter bitmask, then applies it inside
  posting-list brute force.
- `Adaptive nprobe`: uses filter selectivity to probe more centers when the
  filter is restrictive.
- `Centroid-aware filtering`: uses per-center metadata summaries to probe
  centers likely to contain matching records. This is the stronger/stretch
  version.

## What counts as done

Minimum defensible final deliverable:

- A benchmark that varies metadata selectivity and compares fixed SPANN against
  adaptive SPANN.
- Plots or tables for recall@k, latency, and number of centers/candidates
  scanned.
- A Rust patch or isolated Rust benchmark prototype showing adaptive `nprobe`.
- A report explaining the failure mode, implementation, and evaluation.

Strong final deliverable:

- Adaptive `nprobe` integrated into the Rust SPANN path.
- Synthetic benchmark plus one real-ish dataset, or a synthetic workload built
  from real embeddings.
- Oracle comparison: for each query, sweep `nprobe` and show the best possible
  recall/latency curve.
- Optional centroid-aware metadata experiment showing whether "probe smarter"
  beats "probe more".

Out of scope unless everything above is done:

- Learned cost model.
- Full prefilter vs postfilter planner across HNSW, SPANN, sparse, and hybrid
  search.
- SPFresh/incremental index maintenance.
- Production-grade per-centroid metadata persistence.

## Parallel workstreams

Three people can work independently if the interfaces are kept small.

### Workstream A: Baselines and benchmark harness

Owner goal: produce the experimental evidence.

Tasks:

- Create a synthetic dataset with controllable metadata selectivity.
- Run stock SPANN with fixed/adaptive built-in behavior and sweep:
  `selectivity`, `k`, `nprobe`, dataset size, and embedding dimension.
- Compute exact filtered ground truth by brute force over records satisfying
  the filter.
- Output CSV files suitable for plots.

Primary code references:

- `rust/worker/benches/spann.rs`
- `rust/worker/src/execution/operators/spann_bf_pl.rs`
- `rust/index/src/spann/utils.rs`

Metrics to collect:

- `recall@k`
- returned result count, especially `returned < k`
- end-to-end or benchmark-loop latency
- number of centers probed
- posting-list candidates before filtering
- candidates surviving the filter

### Workstream B: Adaptive nprobe implementation

Owner goal: make the smallest Rust change that improves filtered recall.

Tasks:

- Compute filter selectivity from `KnnFilterOutput.filter_output`.
- Thread selectivity into SPANN center search.
- Change `SpannIndexReader::determine_search_nprobe` to scale `nprobe` up for
  selective filters.
- Add unit tests for the nprobe calculation.

Primary code references:

- `rust/worker/src/execution/orchestration/spann_knn.rs`
- `rust/worker/src/execution/operators/spann_centers_search.rs`
- `rust/segment/src/distributed_spann.rs`
- `rust/index/src/spann/types.rs`
- `rust/worker/src/execution/orchestration/knn_filter.rs`
- `rust/worker/src/execution/operators/filter.rs`

Likely implementation shape:

```rust
// Sketch only.
let selectivity = compact_offset_ids_len / total_records_post_compaction;
let boosted = base_nprobe as f64 * (target_survival_rate / selectivity).clamp(1.0, max_factor);
```

Keep clamps. Do not allow tiny selectivity to explode into an unbounded scan.

### Workstream C: Smarter probing / oracle experiments

Owner goal: answer whether the project should stop at adaptive `nprobe` or
include a stronger approach.

Tasks:

- Build an offline oracle: for each query/filter, sweep `nprobe` and find the
  smallest `nprobe` that reaches target recall.
- Prototype per-center metadata summaries in the benchmark only:
  center id -> count of matching metadata values.
- Compare:
  fixed `nprobe`
  adaptive `nprobe`
  oracle `nprobe`
  centroid-aware center ranking

Primary code references:

- `rust/index/src/spann/types.rs`
- `rust/index/src/spann/utils.rs`
- `rust/index/src/metadata/types.rs`
- `rust/worker/benches/spann.rs`

This workstream should not block the adaptive nprobe implementation. It can
stay as an experimental result in the final report if production integration is
too risky.

## Metrics that matter

Primary:

- `filtered recall@k`: overlap between returned ids and exact top-k among only
  records matching the filter.
- `latency`: report total query time and, if instrumented, phase times.
- `returned_count`: if a query requests `k=10` and returns 2 results, that is a
  clear failure case.

Secondary:

- `nprobe_used`: centers requested from center search.
- `centers_returned`: centers actually searched.
- `posting_list_candidates`: total postings loaded before bitmask filtering.
- `surviving_candidates`: postings that pass the metadata filter.
- `planning_overhead`: time to compute selectivity or choose nprobe.
- `bytes_read`: useful if easy to get from posting-list fetch/blockfile code.

Sweep axes:

- filter selectivity: example `0.001`, `0.01`, `0.05`, `0.10`, `0.50`, `1.0`
- `k`: example `10`, `50`
- dataset size: start with `10K`, then scale only if builds/runs are stable
- embedding dimension: start with `128`
- base `nprobe`: example `8`, `16`, `32`, `64`

Report both average and tail behavior. Low-selectivity filters often have
high variance, so include p50/p95 latency if possible.

## Code map

Entry points:

- `rust/worker/src/server.rs`: dispatches queries to HNSW, SPANN, or
  Quantized SPANN based on segment type.
- `rust/worker/src/execution/orchestration/knn_filter.rs`: runs metadata
  filtering first and produces `KnnFilterOutput`.
- `rust/worker/src/execution/operators/filter.rs`: defines `FilterOutput` with
  `log_offset_ids` and `compact_offset_ids` bitmaps.

Non-quantized SPANN path:

- `rust/worker/src/execution/orchestration/spann_knn.rs`: SPANN KNN
  orchestration.
- `rust/worker/src/execution/operators/spann_centers_search.rs`: center
  search operator.
- `rust/segment/src/distributed_spann.rs`: segment reader wrapper; forwards
  `rng_query` to the index reader.
- `rust/index/src/spann/types.rs`: `SpannIndexReader`,
  `determine_search_nprobe`, posting-list fetch.
- `rust/index/src/spann/utils.rs`: low-level center search helper.
- `rust/worker/src/execution/operators/spann_fetch_pl.rs`: loads a posting
  list for a center.
- `rust/worker/src/execution/operators/spann_bf_pl.rs`: brute-force over a
  posting list and applies the filter bitmask.

Quantized SPANN path:

- `rust/worker/src/execution/orchestration/quantized_spann_knn.rs`
- `rust/worker/src/execution/operators/quantized_spann_center_search.rs`
- `rust/worker/src/execution/operators/quantized_spann_bruteforce.rs`

Treat Quantized SPANN as follow-up unless the benchmark specifically uses
`SegmentType::QuantizedSpann`.

Config:

- `rust/types/src/spann_configuration.rs`: public/internal SPANN config,
  including default `search_nprobe`.
- `rust/index/src/config.rs`: `SpannProviderConfig.adaptive_search_nprobe`.
- `rust/worker/chroma_config.yaml`: local worker config currently sets
  `adaptive_search_nprobe: false`.

Existing plans:

- `PLANS/spann-plan.md`
- `PLANS/spann-query-path.md`
- `CLAUDE.md`

## Approaches to try

### Approach 1: Fixed nprobe baseline

Run the existing behavior across selectivity. This is required even if no
implementation lands, because it establishes the failure mode.

Expected result:

- High selectivity: fixed nprobe is fine.
- Low selectivity: many posting-list entries are skipped by the bitmask,
  returned results can be fewer than `k`, and recall drops.

### Approach 2: Global-selectivity adaptive nprobe

Use only total filter selectivity:

```text
nprobe_adaptive = clamp(base_nprobe / max(selectivity, epsilon),
                        base_nprobe,
                        base_nprobe * max_factor)
```

This is the best first implementation because it does not need new index
storage. It should improve recall for selective filters while preserving the
baseline path for no-filter or high-selectivity queries.

Suggested clamps:

- `epsilon`: `0.001` or `0.005`
- `max_factor`: `4x` or `8x`
- absolute max: avoid probing more centers than exist

### Approach 3: Iterative top-up

Run baseline `nprobe`. If fewer than `k` candidates survive, ask for more
centers and continue until either `k` results are available or a budget is hit.

Pros:

- Easy to explain.
- Directly addresses `returned < k`.

Cons:

- More orchestration changes.
- Multiple center-search/fetch rounds can add latency and complexity.

Use this if adaptive nprobe is too blunt or if it performs poorly.

### Approach 4: Per-centroid metadata statistics

Track metadata counts per center/posting list, then prefer centers likely to
contain matching records.

Pros:

- Better than simply probing more centers.
- Closest to the original Hammad/Chroma idea.

Cons:

- Requires new stats maintenance and persistence if implemented properly.
- Riskier than adaptive nprobe.

Recommended use for this semester: benchmark/offline prototype or stretch
feature, not the primary deliverable.

## Minimal build commands

Avoid full workspace builds. Do not run `cargo build --workspace` or
`cargo test --workspace` unless you have enough disk and time.

On this macOS machine, set the SDK include paths before checking packages that
pull in `hnswlib`; otherwise `hnswlib` can fail with `fatal error: 'queue' file
not found`.

```bash
export SDKROOT="/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"
export CPLUS_INCLUDE_PATH="$SDKROOT/usr/include/c++/v1:$SDKROOT/usr/include"
export C_INCLUDE_PATH="$SDKROOT/usr/include"
export CARGO_TARGET_DIR="/tmp/chroma-target"
```

Good first checks:

```bash
cargo check -p chroma-index
cargo check -p worker --lib
```

When editing SPANN config types:

```bash
cargo check -p chroma-types
cargo check -p chroma-index
cargo check -p worker --lib
```

When editing only SPANN operators/orchestrators:

```bash
cargo check -p worker --lib
cargo test -p worker spann_bf_pl
```

When editing SPANN index internals:

```bash
cargo check -p chroma-index
cargo test -p chroma-index spann
```

When running the existing SPANN benchmark:

```bash
cargo bench -p worker --bench spann
```

This benchmark can be heavy because it uses benchmark dataset helpers. Prefer
small synthetic tests first.

Only build Python bindings if you need Python integration:

```bash
maturin dev
```

Only build services if you are running the distributed stack:

```bash
cargo build -p worker --bin query_service
cargo build -p worker --bin compaction_service
```

Likely relevant packages:

- `worker`: query orchestration, SPANN operators, services, SPANN benchmark.
- `chroma-index`: SPANN index reader/writer and center-search internals.
- `chroma-types`: SPANN config types and shared query/filter types.
- `chroma-benchmark`: benchmark dataset helpers.
- `chromadb_rust_bindings`: Python binding layer, only needed for Python E2E.

Disk-saving tips:

- Prefer `cargo check` over `cargo build`.
- Use package-specific commands with `-p`.
- If `target/` gets too large, `cargo clean` can reclaim a lot of space but
  will force a rebuild.
- If you want scratch build artifacts outside the repo, use:

```bash
CARGO_TARGET_DIR=/tmp/chroma-target cargo check -p worker --lib
```

## Suggested immediate next steps

1. Workstream A creates the baseline CSV and exact filtered-recall evaluator.
2. Workstream B threads `filter_selectivity: Option<f64>` from
   `SpannKnnOrchestrator` to `SpannIndexReader::determine_search_nprobe`.
3. Workstream C builds the oracle sweep and optional centroid-aware prototype.
4. Everyone agrees on one shared CSV schema so plots can combine results.

Shared CSV columns:

```text
dataset, n_records, dim, query_id, k, selectivity, strategy,
base_nprobe, nprobe_used, returned_count, recall_at_k,
latency_ms, centers, candidates_before_filter, candidates_after_filter
```

The final narrative should be simple: fixed SPANN ignores filter selectivity
during center choice; restrictive filters remove most candidates after center
selection; adaptive probing recovers recall with bounded extra work.
