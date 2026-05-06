# 2026-05-06 — Full sweep v2: re-tuned adaptive + clean latency + 1M scale + sel=0.1%

## Summary

Second pass of the SPANN optimization hypercube. Three changes vs v1:

1. **Re-tuned adaptive nprobe** so it can no longer undershoot baseline.
2. **Fixed latency timing** in the bench (audit moved outside the timed
   region, `audit_ms` reported separately).
3. **Extended the matrix** to N=1M and sel=0.1%.

The combination produces a far cleaner story: at the regime gates were
designed for (N=1M, sel=0.1%), turning on all three optimizations
together (cell 111) lifts recall from 0.520 to 0.738 (+21.8 pp) while
cutting mean latency from 80.8 ms to 4.8 ms (17×) and p99 from 488 ms
to 10 ms (49×). Cell 111 is a strict Pareto winner across every
(N, sel) tested.

`bad_drops = 0` across **2,400 individual measurements** (8 cells × 6
workloads × 50 queries).

## What landed

- **Source change** (commit `e2217c35`): re-tuned `determine_search_nprobe`.
  - `ADAPTIVE_NPROBE_MAX_FACTOR`: 8 → 16 (so the filter-aware boost
    doesn't saturate at sel=0.1%).
  - `size_based` table: `{24, 32, 64}` at `{500K, 1M, >1M}` →
    `{32, 64, 128}` at `{100K, 1M, >1M}`. The earlier values
    *under-shot* the typical baseline `params.search_nprobe = 32` at
    sub-million scales — turning on adaptive lost recall.
  - Floor: `size_based = max(params.search_nprobe, tier(N))`. Adaptive
    can never undershoot the configured base.
- **Bench fix** (commit `978519c9`): `spann_full_sweep.rs` now ends the
  timed region after BfPL+merge, runs the audit afterward as a
  bench-only correctness check, reports `audit_ms` as a separate column.
  Gate cells now show real latency reduction (3-22× mean, 5-50× p99).
- **Re-runs** (commits `31008999`, `26fad0d2`): the full 6-workload
  matrix (N ∈ {10K, 100K, 1M} × sel ∈ {1%, 0.1%}) on the fixed bench.
  Raw data + summary JSON in `LOGS_PLANS/benchmarks/full_sweep_v2/`.
- **Plots** (this commit): 11 plots covering 4-panel hypercube summaries
  per workload, recall-vs-I/O Pareto across the matrix, scaling vs N,
  selectivity comparison, storage + build cost, audit invariant heatmap.
- **Paper rewrite** (this commit): `paper.md` reflects v2 throughout.
  Major changes:
  - Abstract: new headline numbers from N=1M sel=0.1%.
  - §2.1 adaptive: documents the re-tuning.
  - §3 methodology: latency definition explicit (production pipeline only).
  - §4 results: completely new tables and figures.
  - §5.1 discussion: "adaptive's true potential" framing — its earlier
    regression was a calibration bug, the new tuning fixes it.
  - §5.3 discussion: "latency now tracks I/O" — explains why the v1 paper
    incorrectly claimed it didn't (audit-in-timed-region artifact).
  - §6 limitations: `MAX_FACTOR` and "audit polluted latency" caveats removed
    (both fixed). New caveat about `LocalStorage` masking remote I/O cost.

## Headline numbers — N=1M, sel=0.1%

| Cell | recall@10 | heads_fetched | latency_mean | latency_p99 |
|---|---|---|---|---|
| 000 baseline | 0.520 | 512 | 80.8 ms | 488 ms |
| 100 adaptive | 0.726 | 1024 | 87.6 ms | 192 ms |
| 010 bloom | 0.510 | 16 | 3.7 ms | 33 ms |
| 001 synopsis | 0.536 | 16 | 4.3 ms | 34 ms |
| 110 adapt+bloom | 0.706 | 32 | 5.2 ms | 14 ms |
| 101 adapt+syn | 0.730 | 32 | 6.9 ms | 20 ms |
| 011 bloom+syn | 0.530 | 17 | 4.3 ms | 29 ms |
| **111 all three** | **0.738** | **33** | **4.8 ms** | **10 ms** |

## Process notes

5 commits this session, in order:
- `e2217c35` — adaptive nprobe re-tune + tests
- `978519c9` — bench: audit outside timed region, audit_ms column
- `31008999` — sweep v2 raw data: 10K/100K/1M × sel=1% + 10K/100K × sel=0.1%
- `26fad0d2` — sweep v2 raw data: N=1M sel=0.1% (the killer cell)
- `<this commit>` — plots + paper.md update + this action log

A user-machine crash interrupted the first attempt at N=1M sel=0.1%
(after ~1 hour of build time). Restarted in the background, completed
successfully. Build time at N=1M is ~14 min/index × 4 unique flavors =
~56 min per (N, sel) pair.

## Open follow-ups

- Run on a heterogeneous-keys workload to exercise the bloom+synopsis
  composition in its strict-domination regime (mixed string + int +
  bool keys with different cardinalities).
- Wire the synopsis predicate through the operator/orchestrator path so
  the gate runs on the production query pipeline, not just from
  bench-direct reader calls. Documented in earlier action logs.
- Test on real learned embeddings (BERT/MiniLM/etc) instead of SIFT
  hand-engineered descriptors. The gate-effectiveness assumption that
  filter values are uncorrelated with embedding clusters needs
  re-validation on production-like distributions.
- Run on remote object storage (S3 / R2) to show the I/O reduction
  translating to even larger latency separations. With S3-class fetch
  latency the gates should compound advantage.
- Compose with the (currently sketched) Hammad per-centroid metadata
  statistics workstream — a closer cousin of the synopsis that uses
  centroid-level rather than head-level statistics for cross-collection
  query planning.
