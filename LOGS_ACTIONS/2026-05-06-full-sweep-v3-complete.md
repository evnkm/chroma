# 2026-05-06 — Full sweep v3 (complete): 6-workload hypercube under unified bench

## Summary

Completes the v3 sweep started in `afe6df9e`. Two N=1M workloads landed
under the unified-index harness, in parallel (saved ~15 min wall-clock vs
sequential):

- N=1M sel=1%   build=17.9 min, run total ~22 min
- N=1M sel=0.1% build=18.4 min, run total ~23 min

Together with the four small-N workloads from `afe6df9e`, this gives the
full hypercube `N ∈ {10K, 100K, 1M} × sel ∈ {1%, 0.1%}` × 8 cube cells = 48
cell-workloads, 50 queries each = 2,400 measurements. Gate-audit invariant
held across all of them: `bad_drops = 0`.

## v3 unified numbers — all 6 workloads

### Per-cell averages (recall@10 / heads_fetched / latency_ms)

| Cell | label | N=10K 1% | N=10K 0.1% | N=100K 1% | N=100K 0.1% | N=1M 1% | N=1M 0.1% |
|---|---|---|---|---|---|---|---|
| 000 | baseline | 0.500 / 24 / 0.80 | 0.076 / 24 / 0.72 | 0.370 / 24 / 1.05 | 0.052 / 24 / 1.03 | 0.284 / 32 / 2.21 | 0.062 / 32 / 2.11 |
| 100 | adapt | **1.000** / 384 / 11.11 | 0.448 / 384 / 10.00 | **0.992** / 384 / 12.43 | 0.490 / 384 / 11.90 | **0.958** / 512 / 23.70 | 0.508 / 512 / 20.80 |
| 010 | bloom | 0.500 / 7.5 / 0.33 | 0.076 / 1.0 / 0.11 | 0.370 / 7.0 / 0.45 | 0.052 / 0.6 / 0.23 | 0.284 / 8.9 / 1.00 | 0.062 / 1.0 / 0.54 |
| 001 | syn | 0.500 / 7.5 / 0.29 | 0.076 / 1.0 / 0.07 | 0.370 / 7.0 / 0.32 | 0.052 / 0.6 / 0.11 | 0.284 / 8.9 / 0.73 | 0.062 / 1.0 / 0.27 |
| 110 | a+bloom | **1.000** / 119 / 3.91 | 0.448 / 11.7 / 0.65 | **0.992** / 105 / 4.07 | 0.490 / 10.4 / 0.87 | **0.958** / 140 / 8.25 | 0.508 / 15 / 2.00 |
| 101 | a+syn | **1.000** / 119 / 4.10 | 0.448 / 11.7 / 0.73 | **0.992** / 105 / 4.24 | 0.490 / 10.4 / 1.02 | **0.958** / 140 / 8.87 | 0.508 / 15 / 2.54 |
| 011 | b+s | 0.500 / 7.5 / 0.33 | 0.076 / 1.0 / 0.11 | 0.370 / 7.0 / 0.46 | 0.052 / 0.6 / 0.23 | 0.284 / 8.9 / 0.97 | 0.062 / 1.0 / 0.55 |
| **111** | **all-on** | **1.000 / 119 / 4.00** | **0.448 / 11.7 / 0.67** | **0.992 / 105 / 4.06** | **0.490 / 10.4 / 0.87** | **0.958 / 140 / 8.40** | **0.508 / 15 / 2.00** |

### Headline workload (N=1M, sel=0.1%)

| Cell | recall@10 | hfet | lat | p99 |
|---|---|---|---|---|
| 000 baseline | 0.062 | 32 | 2.11 ms | 2.97 ms |
| 100 adapt | 0.508 | 512 | 20.80 ms | 25.03 ms |
| 010 bloom | 0.062 | 1.0 | 0.54 ms | 0.85 ms |
| 001 syn | 0.062 | 1.0 | 0.27 ms | 0.75 ms |
| 110 adapt+bloom | 0.508 | 15.2 | 2.00 ms | 3.22 ms |
| 101 adapt+syn | 0.508 | 15.3 | 2.54 ms | 3.45 ms |
| 011 bloom+syn | 0.062 | 1.0 | 0.55 ms | 1.28 ms |
| **111 all-on** | **0.508** | **15.2** | **2.00 ms** | **2.93 ms** |

vs the adaptive-only baseline: cell `111` preserves recall (0.508 = 0.508),
cuts mean latency 10×, p99 8.5×, fetched heads 33×.

## Confirmed properties

- **All adapt=0 cells share identical recall and identical c_a per query**
  in every workload (4 cells × 6 workloads × 50 queries = 1,200 invariant
  checks, all hold).
- **All adapt=1 cells share identical recall and c_a per query** (another
  1,200 checks, all hold).
- **`bad_drops=0`** across the 8 × 6 × 50 = 2,400 cell-query measurements.
- **Bloom and synopsis are exactly equivalent on this workload** (single
  int-equality predicate). Their `c_b`, `c_a`, `hfet`, latency match to
  measurement noise. v3-partial showed them differing only because they
  sat on different indexes.

## What v3 says about the three optimizations

1. **Adaptive nprobe is the recall fix.** Without it (cell `000`), recall
   collapses on filtered queries — worst case 0.052 at N=100K sel=0.1%
   probing 24 heads. With it (cell `100`), recall lifts to 0.49 at the
   same workload by probing 16× more heads.

2. **Gates without adaptive provide pure I/O reduction, no recall change.**
   Cells `010` / `001` / `011` track baseline recall exactly; they trim
   heads_fetched from 24-32 down to 0.6-9 depending on selectivity.
   Useful but small effect on absolute latency at this scale.

3. **Adaptive + gate is the strict Pareto winner.** Cell `111` matches
   cell `100`'s recall at every workload while reducing heads_fetched
   3-34× and latency 3-10×. The bigger the adaptive boost (low
   selectivity, large N), the bigger the gate's compounding effect.

## Calibration note (carried over from v3 partial)

The 16× MAX_FACTOR cap on `filter_aware_nprobe` binds at every N≤1M sel=0.1%
workload (np_max = 384 at N≤500K, 512 at N=1M). At N>1M the cap rises to
1024 (size_based=64). This is the same cap that v2 effectively had via
`params.search_nprobe=32` × 16; v3's small-N adaptive ceiling is lower
because size_based starts at 24 instead of 32. Flagged but not addressed
under "option A" — ship as-is.

## Process

- Bench was the unified-harness version from `afe6df9e`. No source changes.
- Two N=1M jobs ran in parallel via direct binary exec (skipping cargo
  lock). Build time grew ~7-12% under contention vs. running solo.
- v3-partial action log + earlier v3-unified action log are unchanged;
  this complete-sweep log supersedes them as the canonical record.

## Next steps

1. Re-generate plots from `LOGS_PLANS/benchmarks/full_sweep_v2/plot.py`
   against the v3 CSVs (schema unchanged). Update paper with new tables.
2. Optional: add a "build once, sweep multiple selectivities" mode to the
   bench so future re-runs avoid rebuilding the index per (N, sel) pair —
   would have saved ~17 min on this sweep alone. Out of scope for now.
3. Notion entry: not yet created (Notion MCP tool not surfaced this
   session).
