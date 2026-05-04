# 2026-05-04 — Full 8-cell hypercube sweep + paper.md

## Summary

Designed, implemented, ran, and wrote up a full ablation sweep across the
3-toggle hypercube `<adaptive_nprobe><bloom><synopsis>` for the Chroma
SPANN index on SIFT1M. 8 cells × 6 workloads (1 × N=100K, 1 × N=10K
primary, 4 × N=10K selectivity sweep) × 50 queries = 2,400 individual
measurements. **Gate-audit invariant `bad_drops = 0` held across every
single measurement.**

## What landed

- `rust/worker/benches/spann_full_sweep.rs` — single-file bench runs all
  8 cells on a shared SIFT1M load. Build economy: 4 unique writer
  flavors (none / bloom / synopsis / both) cover the 8 cells via 2
  readers each. Per-query metrics: recall, latency mean+p99, heads_rng,
  heads_fetched, drop_ratio, candidates_before/after_filter, nprobe_used,
  bad_drops. Per-cell summary JSON: index_build_ms, blob_storage_bytes.
- `LOGS_PLANS/benchmarks/full_sweep/` — 6 CSVs + 6 summary JSONs +
  `plot.py` + 6 plots:
  - `hypercube_n10k.png`, `hypercube_n100k.png` — 4-panel summaries
  - `recall_vs_io.png` — Pareto frontier across cells × N
  - `selectivity_scan.png` — scaling vs filter selectivity
  - `storage_cost.png` — blob storage + build time
  - `audit_table.png` — heatmap proves bad_drops=0 everywhere
- `paper.md` — 7-section writeup with embedded figures: abstract,
  background, three optimizations, methodology, results, discussion,
  limitations, conclusion, references.

## Headline findings

- **Synopsis is the strict winner at N=100K, sel=1%**: heads_fetched
  256→70 (-73%) at recall 0.964, ~zero latency overhead. Strictly
  Pareto-better than baseline.
- **Bloom is essentially tied** at heads_fetched=67 / recall=0.956,
  storage 5.6MB vs synopsis's 9.4MB.
- **Adaptive nprobe REGRESSES recall at sub-500K**: -6pp because the
  size-based rule (24 for N≤500K, 32 for ≤1M, 64 above) downgrades
  vs the `params.search_nprobe=32` baseline. A real footgun. Documented
  as "Discussion §1" in the paper, with a proposed fix
  (`size_based = max(params.search_nprobe, size_table[N])`).
- **Gate composition is redundant on uncorrelated workload**: the
  synopsis is exact, so running the bloom first drops the same heads
  the synopsis would. 011 and 001 produce identical drop_ratio.
- **Drop_ratio scales monotonically with selectivity**: 0.20 at sel=5% →
  0.92 at sel=0.2%. Below ~5% sel the gates start paying their freight.
- **Storage at N=100K**: bloom 5.6MB, synopsis 9.4MB, both 15MB. Roughly
  linear in N.
- **Build time overhead from gates**: ~3-6%. HNSW + PL is the dominant
  cost.

## Process notes

5 commits during the session, in order:

- `657f6aaf` — bench scaffold + smoke test
- `7e65a7d9` — N=10K + N=100K sweep results
- `b1416f16` — selectivity sweep (5 buckets) at N=10K
- `5d1e7ddc` — plots + audit-invariant heatmap
- `<this commit>` — paper.md + action log

This was the third pass of recall-study work in this project; the
first two (bloom-only and synopsis-only) had already produced the raw
implementation and per-feature recall studies. This pass folds them
into a single side-by-side comparison.

## Notion / external links

(None this round — paper.md is the outward-facing artifact.)

## Next steps

- Re-tune the size-based adaptive nprobe rule for sub-million-doc
  segments. Current behavior actively regresses recall at N=100K, the
  most common production scale.
- Run a heterogeneous-keys workload (mixed string + int + bool) to
  exercise the bloom+synopsis composition in its strict-domination
  regime.
- Wire the synopsis predicate through the operator/orchestrator path
  (currently the gate is exercised only from benches via direct reader
  calls). Necessary for production query flow to benefit. Listed as a
  follow-up in `LOGS_ACTIONS/2026-05-04-head-synopsis-implementation.md`.
- Stretch: run on real SPANN-flow with a non-LocalStorage backend to
  show the I/O reduction translating to wall-clock latency.
