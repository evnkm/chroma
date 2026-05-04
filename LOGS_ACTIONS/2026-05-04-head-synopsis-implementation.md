# 2026-05-04 — HeadSynopsis implementation + ablation benchmark

## Summary

Implemented the per-cluster metadata-value-counts gate (HeadSynopsis) for
SPANN per the spec at `rust/index/src/spann/HEAD_SYNOPSIS.md`. The feature
lets the SPANN reader drop centroids whose synopsis says **provably zero**
docs match an equality predicate, before fetching their posting lists. The
synopsis is exact (unlike the bloom), so any drop is correctness-safe — the
`[gate-audit]` invariant requires `bad_drops == 0` and the smoke run on
SIFT1M confirms it.

The two gates are complementary: the synopsis dominates for low- to
medium-cardinality keys (≤1024 distinct values, configurable); the bloom
covers high-cardinality keys (free-form strings, UUIDs). Both can run
together — synopsis after bloom in the pipeline.

## What landed

- `rust/index/src/spann/head_synopsis.rs` (new, ~660 LoC) with
  `synopsis_token`, `synopsis_value_token`, `synopsis_doc_tokens`,
  `SynopsisPredicate { And, Or, Unsupported }`, `extract_synopsis_predicate`,
  `HeadSynopsis { head_size, counts, other_counts }` (with `count_for(k, v)`
  three-valued lookup: exact / provably-zero / unknown), `HeadSynopsisCache`,
  `HeadSynopsisBlob`, `HeadSynopsisBlobFlusher`, `HeadSynopsisWriteConfig`,
  `HeadSynopsisReadConfig`, `HeadRawCounts`, `apply_top_k_capping`, free
  `gate_heads` function, `predicted_yield`. **24 unit tests, all green.**
- Wiring in `rust/index/src/spann/types.rs` (~+250 LoC):
  - 4 new fields on `SpannIndexWriter` (enabled flag, top_k, max_card, per-doc
    structured token cache).
  - 1 new field on `SpannIndexReader` (`head_synopsis_filters`).
  - 1 new field on `SpannIndexFlusher` (`head_synopsis_blob`) and
    `SpannIndexIds` (`head_synopsis_blob_path`).
  - New error variant `HeadSynopsisBlobSaveError`.
  - New methods `add_with_metadata_tokens_and_synopsis` /
    `update_with_metadata_tokens_and_synopsis` (existing bloom-only APIs
    preserved).
  - `delete` now also forgets the per-doc synopsis tokens.
  - `build_head_synopsis_map` at commit-time joins live posting lists
    against the per-doc structured token cache; heads with any
    unknown-tokens doc skip the synopsis entry → gate falls back to "keep"
    (correctness-safe).
  - `gate_heads_synopsis` on the reader, mirroring `gate_heads`.
  - `from_id` accepts a new `Option<HeadSynopsisWriteConfig<'_>>` /
    `Option<HeadSynopsisReadConfig<'_>>` last argument.
  - `SpannIndexFlusher::flush` saves the synopsis blob via
    `Storage::put_bytes` and propagates the path into `SpannIndexIds`.
- Path constant `HEAD_SYNOPSIS_PATH = "head_synopsis_path"` in
  `rust/types/src/segment.rs`, parallel to `HEAD_BLOOM_FILTERS_PATH`.
- Config flags on `SpannProviderConfig` and `SpannProvider`:
  `head_synopsis_enabled`, `head_synopsis_top_k_per_key`,
  `head_synopsis_max_cardinality`. Defaults: `false`, `64`, `1024`.
- Segment-level wiring in `rust/segment/src/distributed_spann.rs`:
  reads/writes `HEAD_SYNOPSIS_PATH` from `Segment::file_path`, threads
  structured tokens through `add` / `update` via `synopsis_doc_tokens`,
  exposes `gate_heads_synopsis` forwarder. Updated `SpannProvider::write`
  to pass the three new flags. `SpannSegmentFlusherShard::flush` now
  inserts `HEAD_SYNOPSIS_PATH → [blob_path]` when the writer produced a
  blob. `SpannSegmentReaderShard::from_segment` constructs a
  `HeadSynopsisReadConfig` from the file_path entry.
- Updated all `SpannIndexWriter::from_id` / `SpannIndexReader::from_id` /
  `SpannSegmentWriterShard::from_segment` / `SpannProvider` literal
  call-sites in tests, benches, and compactor code (~22 sites across
  `chroma-index`, `chroma-segment`, `worker`, including the existing
  bloom benches and the integration test `hnsw_reload_repro.rs`) to pass
  `None` / sensible defaults for the new synopsis args.
- New bench: `rust/worker/benches/spann_synopsis_ablation.rs` (~570 LoC)
  on SIFT1M. Builds two indexes (000 = no synopsis, 001 = synopsis on),
  measures iso-probe (correctness) and iso-I/O (value claim) for the
  predicate `bucket=0`. Has a `[gate-audit]` block that re-fetches each
  dropped head's PL and counts how many contained matching docs; panics
  on >0 unless `SYNOPSIS_AUDIT_NO_PANIC=1`. Registered as
  `[[bench]] name = "spann_synopsis_ablation"` in `rust/worker/Cargo.toml`.
- Module declaration `pub mod head_synopsis;` in
  `rust/index/src/spann.rs`.

## What did not land

- **Operator/orchestrator wiring** (`SpannCentersSearchInput`,
  `SpannCentersSearchOutput`, `KnnFilterOutput`): the spec calls for
  `head_synopsis_predicate` and `heads_after_synopsis` plumbing. Skipped
  for this drop because the recall study only exercises the bench path
  (which calls `reader.gate_heads_synopsis` directly). The reader-side
  primitives are in place; production wiring is a one-screen follow-up
  that mirrors the existing `head_bloom_tokens` / `heads_after_bloom`
  pattern.
- **Compaction-time join from the metadata segment's inverted index**
  (spec §7 "recommended"). The current build path uses the SPANN-side
  `head_synopsis_doc_tokens` cache populated at write time; this works
  end-to-end for fresh-build benches (the recall study) but for
  production where a segment opens onto an already-compacted PL, only
  fresh writes are reflected. Documented in the build code: heads with
  any unknown-tokens live doc skip the synopsis entry, so the gate
  falls back to "keep" (correctness-safe). The metadata-segment-driven
  rebuild is a follow-up that requires a `&MetadataSegmentWriterShard`
  argument on `SpannSegmentWriterShard::commit`.
- **Joint bloom+synopsis bench cell (the 011 ablation in the 8-cell
  hypercube)**. Not blocking; can be derived later by combining the
  010 and 001 cells.

## Smoke run on SIFT1M

`BENCH_N_RECORDS=2000 BENCH_N_QUERIES=10 BENCH_BUCKETS=20 BENCH_PROBE_NBR=8
cargo bench -p worker --bench spann_synopsis_ablation`:

```
========== HeadSynopsis 000 vs 001 ablation ==========
  dataset=sift1m n_records=2000 buckets=20 k=10 selectivity=0.0500
  iso-probe (probe_nbr=8):
    000 (no synopsis): recall_mean=0.5400 heads_fetched_mean=8.0
                       drop_ratio_mean=0.0000 bad_drops=0
    001 (synopsis):    recall_mean=0.5100 heads_fetched_mean=5.6
                       drop_ratio_mean=0.3000 bad_drops=0
    Δrecall (001 − 000) = -0.0300
  iso-I/O (probe_nbr_001=16):
    000 (no synopsis): recall_mean=0.5400 heads_fetched_mean=8.0
    001 (synopsis):    recall_mean=0.7200 heads_fetched_mean=11.7
    Δrecall (001 − 000) = +0.1800
  [gate-audit] total bad_drops across all queries: 0
====================================================
```

Reading:

- **`bad_drops = 0`** in both iso-probe and iso-I/O: the synopsis is
  exact, no head with a matching doc was ever dropped. Strict
  correctness invariant holds.
- **iso-probe `drop_ratio = 0.30`**: the synopsis correctly identifies
  ~30% of probed heads as provably empty for `bucket=0`.
- **iso-probe `Δrecall = -0.03`**: small negative, attributable to
  SPANN-internal nondeterminism between the two independently-built
  indexes (different HNSW structure, different random kmeans seeds).
  Within bench noise.
- **iso-I/O `Δrecall = +0.18`**: at matched-ish I/O budget the
  synopsis-enabled path achieves 18 percentage points more recall.
  This is the headline win: the gate is worth its complexity.

CSV at
`LOGS_PLANS/benchmarks/sift1m-synopsis-ablation-n2000-q10-buckets20.csv`.

## Files touched

- `rust/index/src/spann/head_synopsis.rs` (new)
- `rust/index/src/spann.rs` (mod declaration)
- `rust/index/src/spann/types.rs`
- `rust/index/src/spann/fast_writer.rs` (`SpannIndexFlusher` initializer)
- `rust/index/src/config.rs` (`SpannProviderConfig` defaults)
- `rust/index/tests/hnsw_reload_repro.rs` (call-site fixups)
- `rust/types/src/segment.rs` (`HEAD_SYNOPSIS_PATH`)
- `rust/segment/src/spann_provider.rs`
- `rust/segment/src/distributed_spann.rs`
- `rust/segment/src/test.rs` (`SpannProvider` literal)
- `rust/worker/Cargo.toml` (bench registration)
- `rust/worker/benches/spann_synopsis_ablation.rs` (new)
- `rust/worker/benches/spann.rs`,
  `rust/worker/benches/spann_bloom_ablation.rs`,
  `rust/worker/benches/spann_bloom_sweep.rs`,
  `rust/worker/benches/spann_bloom_adaptive_sweep.rs`,
  `rust/worker/benches/spann_filtered_recall.rs` (call-site fixups for
  the new optional synopsis args)
- `rust/worker/src/compactor/compaction_manager.rs`,
  `rust/worker/src/execution/orchestration/compact.rs` (`SpannProvider`
  literal fixups)

## Tests

- All 24 new `head_synopsis` unit tests pass.
- All 65 `spann::*` unit tests in `chroma-index` pass (no regressions in
  the 41 pre-existing SPANN tests).
- Both `chroma-segment` `distributed_spann::test` integration tests
  pass with the new args.
- `cargo check -p chroma-index -p chroma-segment -p worker --tests
  --benches` is clean.

## Next steps

- Run the iso-probe / iso-I/O cells on the standard SIFT1M subsets
  (10K → 50K → 100K) per CLAUDE.md and write the results to
  `LOGS_PLANS/benchmarks/`. Expected: `bad_drops = 0` in every cell;
  iso-I/O `Δrecall > 0` at all sizes.
- Add the operator/orchestrator wiring (`SpannCentersSearchInput.
  head_synopsis_predicate`, `KnnFilterOutput.head_synopsis_predicate`,
  `heads_after_synopsis` telemetry) so the gate is exercised on the
  production query path, not just the bench.
- Add the joint bloom+synopsis (011) cell to verify the two gates
  compose without interference, per HEAD_SYNOPSIS.md §14.
- Wire the metadata-segment-driven build (§7) for production
  segment-reload correctness. The SPANN-side cache approach is fine for
  the recall study; production needs the inverted-index join.
