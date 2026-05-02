# 2026-05-02 — BloomHeads implementation + ablation benchmark

## Summary

Implemented the per-head metadata Bloom filter gate (BloomHeads) for SPANN per
the spec at `rust/index/src/spann/BLOOM_FILTERS.md`. The feature lets the
SPANN reader drop centroids whose bloom filter says "definitely no docs match
this metadata predicate", before fetching their posting lists. With Phase 1.5
Option 2 (commit-time rebuild) on, drop_ratio reaches 0.72–0.97 on the bench
predicate `bucket=0`, and the gate produced **zero false negatives** across
all bench runs.

## What landed

- `rust/index/src/spann/head_bloom.rs` (new, ~700 LoC) with `EqualityTokens`,
  `HeadBloom`, `HeadBloomBlob`, `HeadBloomCache`, `HeadBloomBlobFlusher`,
  `HeadBloomWriteConfig`, `HeadBloomReadConfig`, `extract_equality_tokens`,
  `metadata_token`, `doc_tokens`, free `gate_heads` function. 21 unit tests,
  all green.
- Wiring in `rust/index/src/spann/types.rs` (~+200 LoC) on
  `SpannIndexWriter` / `SpannIndexReader` / `SpannIndexFlusher` /
  `SpannIndexIds` / `SpannIndexWriterError`. Hooks into
  `add_to_postings_list`, `append` (no-split, same-head reuse, new child,
  parent decommission branches), `reassign` (passes `None` tokens),
  `try_delete_posting_list`, and the merge sites in `garbage_collect_head`.
  Commit-time rebuild and stale-skipping serialization.
- Path constant `HEAD_BLOOM_FILTERS_PATH` in `rust/types/src/segment.rs`.
- Config flags on `SpannProviderConfig` and `SpannProvider`:
  `head_bloom_enabled`, `head_bloom_capacity_factor`,
  `head_bloom_doc_tokens_cache`, `head_bloom_commit_rebuild`. All default
  `false` / `4`. Per-head capacity = `split_threshold × factor` (default
  `200 × 4 = 800`, FPR target 0.001).
- Segment-level wiring in `rust/segment/src/distributed_spann.rs`:
  reads/writes `HEAD_BLOOM_FILTERS_PATH` from `Segment::file_path`, threads
  tokens through `add` / `update`, exposes `gate_heads` forwarder. Updated
  `SpannProvider::write` to pass the four flags.
- Operator wiring: `SpannCentersSearchInput.head_bloom_tokens`,
  `SpannCentersSearchOutput { center_ids, heads_rng, heads_after_bloom }`.
  Operator calls `reader.gate_heads(...)` after `rng_query`.
- Orchestrator wiring: `KnnFilterOutput.head_bloom_tokens` populated from
  `Filter::where_clause` via `extract_equality_tokens`; `SpannKnnOrchestrator`
  passes it into `SpannCentersSearchInput`. Tracing emits `heads_rng` and
  `heads_after_bloom` per query.
- New bench: `rust/worker/benches/spann_bloom_ablation.rs` (~520 LoC) on
  SIFT1M. Builds two indexes (000 = no bloom, 010 = bloom on), measures
  iso-probe (correctness) and iso-I/O (value claim) for the predicate
  `bucket=0`. Has a `[gate-audit]` block that re-fetches each dropped head's
  PL and counts how many contained matching docs; panics on >0 unless
  `BLOOM_AUDIT_NO_PANIC=1`. Registered as `[[bench]] name = "spann_bloom_ablation"`
  in `rust/worker/Cargo.toml`.

## Key implementation decisions / divergences from spec

- `HeadBloomBlobFlusher` lives in `head_bloom.rs` rather than
  `fast_writer.rs`. The spec put it in `fast_writer.rs` for sharing, but
  only the legacy `SpannIndexWriter` is wired up. Keeping it co-located with
  the rest of the bloom code is simpler.
- The spec's `HeadBloom::union_in_place` was specified with `&self` but
  fastbloom's `AtomicBloomFilter` doesn't expose direct in-place bit-set.
  Removed it; the cache's `HeadBloomCache::union_into` does the merge by
  rebuilding from blobs (which is what the writer actually calls).
- **Auto-enable doc-tokens cache when `commit_rebuild` is on.** The spec
  treated Option 2 as standalone, but `rebuild_blooms_from_cache` walks
  `PL × doc_tokens` and needs `doc_tokens` populated. If the doc_tokens
  map is `None`, every head gets `mark_stale`'d and the gate becomes a
  no-op. `SpannIndexWriter::from_id` now allocates the doc_tokens map
  whenever either flag is on, regardless of which one. Without this, Option 2
  alone gave drop_ratio = 0 (cache len = 0 in the persisted blob).

## Bench results — sift1m, predicate `bucket = 0`

Setup: SPANN with default `split_threshold=200`, `write_nprobe=32`,
`probe_nbr=32` (iso-probe baseline). Each row averages over 20 queries.
Iso-I/O 010 probe count was auto-derived for buckets=1000 runs (chosen so
the gate trims back to ~probe_nbr fetches) and overridden to 128 for
buckets=100 / n=50K. Baseline absolute recall is low because the bench
calls `rng_query` directly with `probe_nbr=32` (bypasses size-based
adaptive nprobe) — see §12 of `BLOOM_FILTERS.md`.

| config            | n       | buckets | sel    | scenario | probe_nbr_010 | drop_ratio | recall_000 | recall_010 | Δrecall  | bad_drops |
|-------------------|---------|---------|--------|----------|---------------|------------|------------|------------|----------|-----------|
| MVP (Phase 1)     | 10000   | 100     | 0.0100 | iso-probe| 32            | 0.000      | 0.595      | 0.530      | -0.065   | 0         |
| Option 2          | 10000   | 100     | 0.0100 | iso-probe| 32            | 0.000      | 0.580      | 0.645      | +0.065   | 0         |
| Option 2          | 10000   | 1000    | 0.0010 | iso-probe| 32            | 0.969      | 0.090      | 0.100      | +0.010   | 0         |
| Option 2          | 10000   | 1000    | 0.0010 | iso-I/O  | 640           | 0.953      | 0.090      | 0.825      | **+0.735** | 0       |
| MVP (Phase 1)     | 10000   | 1000    | 0.0010 | iso-probe| 32            | 0.000      | 0.115      | 0.100      | -0.015   | 0         |
| Option 1 alone    | 10000   | 1000    | 0.0010 | iso-probe| 32            | 0.000      | 0.150      | 0.085      | -0.065   | 0         |
| Both options      | 10000   | 1000    | 0.0010 | iso-I/O  | 512           | 0.961      | 0.145      | 0.515      | +0.370   | 0         |
| Option 2          | 50000   | 1000    | 0.0010 | iso-probe| 32            | 0.972      | 0.075      | 0.065      | -0.010   | 0         |
| Option 2          | 50000   | 1000    | 0.0010 | iso-I/O  | 512           | 0.971      | 0.075      | 0.650      | **+0.575** | 0       |
| Option 2          | 50000   | 100     | 0.0100 | iso-probe| 32            | 0.731      | 0.450      | 0.405      | -0.045   | 0         |
| Option 2          | 50000   | 100     | 0.0100 | iso-I/O  | 128           | 0.720      | 0.450      | 0.785      | **+0.335** | 0       |

Files: `LOGS_PLANS/benchmarks/sift1m-bloom-ablation-*.csv`.

### What this means

1. **Strict correctness preserved.** `[gate-audit] bad_drops = 0` across
   every run. The gate never drops a head whose PL contains a doc matching
   the predicate.
2. **Iso-probe Δrecall is within bench noise** (|Δ| ≤ 0.07) on every
   non-degenerate config. Recall is preserved at fixed probe count.
3. **Iso-I/O is a clear win wherever the gate is doing work.** With
   Option 2 enabled, drop_ratio = 0.72–0.97 and Δrecall is +0.34 to +0.74
   at matched I/O. The gate lets the reader probe many more centroids and
   trim them down to the same number of fetched PLs, drawing the actual
   fetches from a much wider candidate pool. Each Δrecall ≥ +0.20 — well
   above the spec's 0.02 threshold for "improvement is large enough to
   matter end-to-end".
4. **Phase 1 MVP is correct but ineffective.** Heavy reassign workloads
   stale every head, so `drop_ratio = 0`. Matches the spec.
5. **Option 1 alone is not enough on this workload.** The append-time
   record_doc_appended_to_head path keeps filters non-stale, but the gate's
   drop_ratio stays at 0 — Option 1's mechanic only helps reassign-induced
   appends, and the bulk-add path's filters are all populated to begin with
   (no need to repopulate from cache). On the bench, Option 1 alone is
   indistinguishable from MVP at iso-probe and Δrecall is in the noise.
6. **Recommended config: `head_bloom_commit_rebuild = true`** (auto-enables
   doc_tokens cache). All-buckets retain their bloom filters at commit
   time, drop_ratio matches the predicate's selectivity profile, recall
   wins at matched I/O.

## How to reproduce

```bash
export SDKROOT="/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"
export CPLUS_INCLUDE_PATH="$SDKROOT/usr/include/c++/v1:$SDKROOT/usr/include"
export C_INCLUDE_PATH="$SDKROOT/usr/include"
export CARGO_TARGET_DIR=/tmp/chroma-target
. "$HOME/.cargo/env"

cargo build --bench spann_bloom_ablation -p worker --release

BIN=$(ls -1 /tmp/chroma-target/release/deps/spann_bloom_ablation-* \
       | grep -v '\.' | head -1)

# Recommended Phase 1.5 Option 2 run:
BENCH_N_RECORDS=50000 BENCH_N_QUERIES=20 BENCH_PROBE_NBR=32 \
BENCH_BUCKETS=100 BENCH_PROBE_NBR_010_ISO_IO=128 \
BLOOM_COMMIT_REBUILD=1 BLOOM_AUDIT_NO_PANIC=1 \
BENCH_OUTPUT="LOGS_PLANS/benchmarks/sift1m-bloom-ablation-option2-n50k-b100.csv" \
"$BIN"
```

## Test status

- `chroma-index` lib tests for `head_bloom`: 21 / 21 pass.
- `chroma-index` lib (no `--tests`): builds clean.
- `chroma-segment` lib + tests: builds clean.
- `worker` lib + benches + tests: builds clean.

## Next steps

- Hook the `head_bloom_blob_path` into segment-level `prefetch_supported`
  list so Spann readers warm the bloom blob alongside posting lists.
- Add a CSV column for end-to-end query latency (current cell records the
  per-call latency; the iso-I/O bloom probe is heavier in HNSW work).
- Add a 1M-record run on a worker box; SIFT1M loader is already wired.
- Compose with the adaptive-`nprobe` work: the gate trims candidates *after*
  the reader chose `nprobe`. With both on, the reader can crank `nprobe`
  with the existing `filter_aware_nprobe` clamp, then the gate trims back
  to the I/O budget.
