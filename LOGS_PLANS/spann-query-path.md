# SPANN Filtered Query Path

Mapped 2026-04-14 for midterm. All paths relative to repo root.

## Pipeline (in order)

```
Python client.query(where=F, n_results=k)
  │
  └─► KnnFilterOrchestrator
        rust/worker/src/execution/orchestration/knn_filter.rs
        → Evaluates metadata filter F against the blockfile index
        → Produces FilterOutput {
             log_offset_ids:     SignedRoaringBitmap,  // recent uncommitted records
             compact_offset_ids: SignedRoaringBitmap,  // compacted segment records ← our selectivity signal
          }

  └─► SpannKnnOrchestrator
        rust/worker/src/execution/orchestration/spann_knn.rs
        Receives: KnnFilterOutput (contains FilterOutput above)

        ┌── [parallel] ──────────────────────────────────────────────────────┐
        │  KnnLog (recent log records, distance scored)                       │
        │  SpannCentersSearch                                                  │
        │    rust/worker/src/execution/operators/spann_centers_search.rs      │
        │    Input: normalized_query, k, collection_num_records_post_comp     │
        │    NOTE: filter NOT passed here                                     │
        │    → calls reader.rng_query() → calls determine_search_nprobe()    │
        │    → returns center_ids: Vec<usize>  (nprobe centroid IDs)         │
        └────────────────────────────────────────────────────────────────────┘

        for each centroid_id in center_ids:
          SpannFetchPl
            rust/worker/src/execution/operators/spann_fetch_pl.rs
            → fetches raw posting list (doc_offset_id + embedding) for one centroid

          SpannBfPl                                      ← FILTER APPLIED HERE
            rust/worker/src/execution/operators/spann_bf_pl.rs
            Input: posting_list, k, filter (compact_offset_ids), query
            → for each posting:
                 if filter excludes posting.doc_offset_id → skip
                 else → score with distance_function, add to max-heap
            → returns top-k from this centroid

  └─► KnnMerge
        Merges results from all SpannBfPl + KnnLog → final top-k
```

## nprobe: where and how

**Default:** 64 (`default_search_nprobe` in `rust/types/src/spann_configuration.rs:7-8`)

**Config:** `InternalSpannConfiguration.search_nprobe: u32` + `adaptive_search_nprobe: bool`

**Runtime calculation:**
```rust
// rust/types/src/spann_configuration.rs:2819-2838
fn determine_search_nprobe(
    &self,
    collection_num_records_post_compaction: usize,
    k: usize,
) -> u32 {
    let min_nprobe = ((k * 20) as f64 / self.params.split_threshold as f64).ceil() as u32;
    let optimal_nprobe = if self.adaptive_search_nprobe {
        match collection_num_records_post_compaction {
            ..=500_000  => 24,
            ..=1_000_000 => 32,
            _           => 64,
        }
    } else {
        self.params.search_nprobe
    };
    optimal_nprobe.max(min_nprobe)
}
```

**Called from:** `spann_centers_search.rs` via `reader.rng_query()`.

## Where recall dies on filtered queries

`SpannBfPl` skips any posting entry that the bitmask excludes. If the filter
selects only 5% of the collection and the probed centroids happen to have
low coverage of those 5%, the brute-force loop skips 95% of candidates and
the heap may fill with fewer than k items → return < k results, recall = 0.

The filter outcome is **not observable by `SpannCentersSearch`** today; it
fires with the same `nprobe` whether the filter passes 1% or 99%.

## The fix: thread selectivity into determine_search_nprobe

**Selectivity is already available** in `KnnFilterOutput.filter_output.compact_offset_ids`.
The bitmask `.len()` divided by `collection_num_records_post_compaction` gives selectivity.

Proposed change to `determine_search_nprobe`:
```rust
fn determine_search_nprobe(
    &self,
    collection_num_records_post_compaction: usize,
    k: usize,
    filter_selectivity: Option<f64>,   // NEW: None = no filter
) -> u32 {
    let base = /* existing size-based logic */;
    let selectivity_factor = match filter_selectivity {
        None => 1.0,
        Some(sel) if sel >= 0.5 => 1.0,        // permissive filter: no boost needed
        Some(sel) => (0.2 / sel.max(0.001)).min(MAX_NPROBE_FACTOR),
    };
    ((base as f64 * selectivity_factor).ceil() as u32).clamp(base, MAX_NPROBE)
}
```

**Files to modify (estimated):**
1. `rust/types/src/spann_configuration.rs` — add `filter_selectivity` param to `determine_search_nprobe`
2. `rust/worker/src/execution/operators/spann_centers_search.rs` — add `filter_selectivity: Option<f64>` to `SpannCentersSearchInput`
3. `rust/worker/src/execution/orchestration/spann_knn.rs` — compute selectivity from `FilterOutput` and pass through

**Estimated LOC:** ~30 across 3 files.

## Key types

```rust
// rust/worker/src/execution/operators/filter.rs
struct FilterOutput {
    log_offset_ids:     SignedRoaringBitmap,
    compact_offset_ids: SignedRoaringBitmap,  // ← .len() / total = selectivity
}

// rust/worker/src/execution/operators/spann_centers_search.rs
struct SpannCentersSearchInput<'a> {
    reader: Option<SpannSegmentReaderShard<'a>>,
    normalized_query: Vec<f32>,
    collection_num_records_post_compaction: usize,
    k: usize,
    // ADD: filter_selectivity: Option<f64>
}

struct SpannCentersSearchOutput {
    center_ids: Vec<usize>,
}
```

## Tests to run / write

- Existing: `rust/worker/benches/spann.rs` (lines 160–270) — exercises the full pipeline
- New: unit test on `determine_search_nprobe` with `filter_selectivity = Some(0.01)`
  verifying nprobe scales up.
- New: integration test: insert 10K vectors with `bucket` metadata, query with
  `bucket=0` at selectivity 0.05. Assert returned ids >= k with adaptive nprobe
  and < k with fixed nprobe.
