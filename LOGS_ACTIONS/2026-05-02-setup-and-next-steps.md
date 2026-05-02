# 2026-05-02 Setup and Next Steps

## Project Direction Captured

Created `LOGS_PLANS/getting-started.md` to turn the original proposal into a
concrete implementation plan for the current codebase.

Main scope decision:

- Treat this as a SPANN filtered-recall project, not a broad HNSW/SPANN
  prefilter-vs-postfilter planner.
- Focus on the failure mode where Chroma computes a metadata bitmask before
  vector search, but SPANN center selection still ignores filter selectivity.
- Use adaptive `nprobe` as the primary implementation path.
- Keep per-centroid metadata statistics as a stretch/offline experiment.

Documented workstreams:

- Workstream A: baseline metrics and benchmark harness.
- Workstream B: adaptive `nprobe` implementation.
- Workstream C: oracle and smarter probing experiments.

Documented key metrics:

- `filtered recall@k`
- `latency_ms`
- `returned_count`
- `nprobe_used`
- `centers_returned`
- `posting_list_candidates`
- `surviving_candidates`
- `planning_overhead`

## Codebase Setup Completed

Verified the narrow Rust setup needed for SPANN work without building the full
workspace.

Commands that passed:

```bash
cargo check -p chroma-types
cargo check -p chroma-index
cargo check -p worker --lib
cargo test -p worker spann_bf_pl
```

Focused test result:

```text
test execution::operators::spann_bf_pl::test::test_spann_bf_pl_operator ... ok
1 passed
```

The `chroma-index` check initially failed on macOS because `hnswlib` could not
find the C++ standard library header `queue`. Added the required SDK exports to
`LOGS_PLANS/getting-started.md`:

```bash
export SDKROOT="/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"
export CPLUS_INCLUDE_PATH="$SDKROOT/usr/include/c++/v1:$SDKROOT/usr/include"
export C_INCLUDE_PATH="$SDKROOT/usr/include"
export CARGO_TARGET_DIR="/tmp/chroma-target"
```

Build artifacts were kept out of the repo by using:

```bash
export CARGO_TARGET_DIR="/tmp/chroma-target"
```

Disk state after setup:

- `/tmp/chroma-target` was about `11G`.
- The machine had about `18G` free after setup.
- Avoid full workspace builds unless absolutely necessary.

## Relevant Code Paths Identified

SPANN query dispatch:

- `rust/worker/src/server.rs`

Metadata filtering and bitmask production:

- `rust/worker/src/execution/orchestration/knn_filter.rs`
- `rust/worker/src/execution/operators/filter.rs`

Non-quantized SPANN path:

- `rust/worker/src/execution/orchestration/spann_knn.rs`
- `rust/worker/src/execution/operators/spann_centers_search.rs`
- `rust/worker/src/execution/operators/spann_fetch_pl.rs`
- `rust/worker/src/execution/operators/spann_bf_pl.rs`
- `rust/segment/src/distributed_spann.rs`
- `rust/index/src/spann/types.rs`
- `rust/index/src/spann/utils.rs`

Where `nprobe` is selected:

- `rust/index/src/spann/types.rs`
- `SpannIndexReader::determine_search_nprobe`

Where the filter is applied today:

- `rust/worker/src/execution/operators/spann_bf_pl.rs`
- The posting-list brute-force loop skips entries excluded by
  `SignedRoaringBitmap`.

Existing benchmark references:

- `rust/worker/benches/spann.rs`
- `rust/benchmark/src/datasets/sift.rs`
- `rust/benchmark/src/datasets/gist.rs`
- `rust/benchmark/src/datasets/scidocs.rs`

## Dataset Recommendation

Use **SIFT1M subsets with synthetic metadata filters** as the primary dataset
for analysis.

Why this is the best balance:

- SIFT1M is a standard ANN benchmark with real vector structure, so it gives a
  better signal than random Gaussian vectors.
- The vectors are 128-dimensional, which is small enough for fast local
  iteration.
- The repo already has a `Sift1MData` helper in
  `rust/benchmark/src/datasets/sift.rs` that downloads base vectors, query
  vectors, and unfiltered ground truth.
- We can subset it aggressively for iteration: start with `10K` or `50K`
  records and `100` queries, then scale to `100K` or more if runtime is fine.
- Synthetic metadata lets us control selectivity exactly, which is the variable
  this project needs to isolate.

Recommended evaluation setup:

```text
dataset = sift1m-subset
n_records = 10K, 50K, 100K
dim = 128
queries = first 100-500 SIFT query vectors
k = 10, optionally 50
metadata = synthetic bucket labels with controlled selectivity
selectivity = 0.001, 0.01, 0.05, 0.10, 0.50, 1.0
ground_truth = exact brute force over only records matching the filter
```

Do not use `GistDataset` as the primary dataset yet. In this checkout,
`rust/benchmark/src/datasets/gist.rs` points at a developer-local path:

```text
/Users/sanketkedia/Downloads/siftsmall/siftsmall_base.fvecs
```

That makes it poor for reproducible team work unless someone first fixes the
loader.

Use purely synthetic random vectors only for smoke tests. They are fast, but
they are not a strong final evaluation signal because random vectors may not
stress SPANN the same way real ANN benchmark vectors do.

Use SciDocs/MS MARCO only as a final stretch validation. They require more
download/preprocessing/embedding decisions and can slow iteration.

## Next Steps

1. Get baseline metric numbers.

   Build a benchmark path that runs stock SPANN on SIFT1M subsets with
   synthetic metadata selectivity. Record results in CSV and add the summary
   tables/plots to Notion.

2. Use a shared CSV schema.

```text
dataset, n_records, dim, query_id, k, selectivity, strategy,
base_nprobe, nprobe_used, returned_count, recall_at_k,
latency_ms, centers, candidates_before_filter, candidates_after_filter
```

3. Compute exact filtered ground truth.

   For each query/filter pair, brute force over records satisfying the metadata
   filter and take the exact top-k. Use this to compute `filtered recall@k`.

4. Proceed with adaptive `nprobe` implementation.

   Thread `filter_selectivity: Option<f64>` through:

   - `rust/worker/src/execution/orchestration/spann_knn.rs`
   - `rust/worker/src/execution/operators/spann_centers_search.rs`
   - `rust/segment/src/distributed_spann.rs`
   - `rust/index/src/spann/types.rs`

5. Log the metrics needed for clean fixed-vs-adaptive comparison.

   At minimum, capture:

   - `strategy`
   - `base_nprobe`
   - `nprobe_used`
   - `returned_count`
   - `recall_at_k`
   - `latency_ms`
   - `candidates_before_filter`
   - `candidates_after_filter`

6. Keep the implementation bounded.

   Start with global-selectivity adaptive `nprobe`:

```text
nprobe_adaptive = clamp(base_nprobe / max(selectivity, epsilon),
                        base_nprobe,
                        base_nprobe * max_factor)
```

   Use clamps such as `epsilon = 0.001` and `max_factor = 4x` or `8x`.
   Tune only after baseline curves exist.

7. Re-run focused checks after implementation.

```bash
export SDKROOT="/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"
export CPLUS_INCLUDE_PATH="$SDKROOT/usr/include/c++/v1:$SDKROOT/usr/include"
export C_INCLUDE_PATH="$SDKROOT/usr/include"
export CARGO_TARGET_DIR="/tmp/chroma-target"

cargo check -p chroma-types
cargo check -p chroma-index
cargo check -p worker --lib
cargo test -p worker spann_bf_pl
```

## Current Status

Ready to start implementation and experiments.

No adaptive `nprobe` code has been implemented yet. The next concrete task is
to produce baseline fixed-SPANN metrics on SIFT1M subsets, then make the
adaptive `nprobe` patch and compare against that baseline.
