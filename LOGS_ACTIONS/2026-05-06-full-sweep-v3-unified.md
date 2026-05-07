# 2026-05-06 — Full sweep v3 (unified bench): N=10K + N=100K with shared index

## Why this re-run was needed

Earlier today we committed v3 partial data (`4d33747c`) with cell `000` showing
the corrected baseline collapse. That data had a confounder we missed at
first: the bench built **four separate writer flavors** (`none` / `bloom` /
`synopsis` / `both`), and the SPANN writer uses `rand::thread_rng()` during
record assignment ([rust/index/src/spann/types.rs:1440, 2447, 3904, …](rust/index/src/spann/types.rs)).
Even two `none`-flavor builds of the same dataset produce different HNSW
structures.

That made every cross-gate comparison (000 vs 010 vs 001 vs 011 at the same
nprobe) measure **index-build variance, not gate quality**. Empirical
confirmation from the v3-partial CSVs:

| Workload | non-adaptive cells with differing c_a | adaptive cells with differing c_a |
|---|---|---|
| N=10K sel=1% | 50/50 | 50/50 |
| N=10K sel=0.1% | 40/50 | 50/50 |
| N=100K sel=1% | 50/50 | 50/50 |
| N=100K sel=0.1% | 35/50 | 50/50 |

If gates were correctness-preserving over a shared index, `c_a` (the count of
allowed records seen across kept heads) would be **identical** across all
four cells per query — because gates only drop heads with no allowed records.
The data showed the opposite: it differed in nearly every query, in every
workload. The four indexes really did expose different head sets at the same
`nprobe`.

A second confirming signal in the v3-partial data: at N=100K sel=1%, cell
`110` (adapt+bloom) reported recall 0.994 while cell `100` (adapt only) was
0.982. Were they reading from the same index, `110`'s kept heads ⊆ `100`'s,
so `110`'s recall must be ≤ `100`'s. The reverse ordering is only possible
under index-build divergence.

## What landed

- **Bench harness fix** (this commit). `rust/worker/benches/spann_full_sweep.rs`
  now builds **one** writer with both bloom + synopsis enabled and opens
  **two** readers (one per `adaptive_search_nprobe` setting) on it. All 8
  cells share the same underlying HNSW + posting lists; cell-level
  differences are applied at READ time only by toggling whether
  `gate_heads` and/or `gate_heads_synopsis` are called per query.
- **Re-run of all 4 small workloads** under the unified harness. CSVs at
  `LOGS_PLANS/benchmarks/full_sweep_v3/` overwrite the v3-partial files.
- **Validation**: under the unified harness, all 4 adapt=0 cells
  (`000`/`010`/`001`/`011`) report identical `recall_at_k` and identical
  `candidates_after_filter` per query. Same for the 4 adapt=1 cells. Gates
  are now provably correctness-preserving in the data, exactly as the audit
  invariant predicts.

## v3 unified numbers

### N=10K, sel=1%

| Cell | recall@10 | heads_fetched | latency |
|---|---|---|---|
| 000 baseline | 0.500 | 24 | 0.80 ms |
| 100 adapt | **1.000** | 384 | 11.11 ms |
| 010 bloom (no adapt) | 0.500 | 7.5 | 0.33 ms |
| 001 syn (no adapt) | 0.500 | 7.5 | 0.29 ms |
| 110 adapt+bloom | **1.000** | 119 | 3.91 ms |
| 101 adapt+syn | **1.000** | 119 | 4.10 ms |
| 011 b+s (no adapt) | 0.500 | 7.5 | 0.33 ms |
| **111 all-on** | **1.000** | **119** | **4.00 ms** |

### N=10K, sel=0.1%

| Cell | recall@10 | heads_fetched | latency |
|---|---|---|---|
| 000 baseline | 0.076 | 24 | 0.72 ms |
| 100 adapt | **0.448** | 384 | 10.00 ms |
| 010 bloom (no adapt) | 0.076 | 1.0 | 0.11 ms |
| 001 syn (no adapt) | 0.076 | 1.0 | 0.07 ms |
| 110 adapt+bloom | **0.448** | 11.7 | 0.65 ms |
| 101 adapt+syn | **0.448** | 11.7 | 0.73 ms |
| 011 b+s (no adapt) | 0.076 | 1.0 | 0.11 ms |
| **111 all-on** | **0.448** | **11.7** | **0.67 ms** |

### N=100K, sel=1%

| Cell | recall@10 | heads_fetched | latency |
|---|---|---|---|
| 000 baseline | 0.370 | 24 | 1.05 ms |
| 100 adapt | **0.992** | 384 | 12.43 ms |
| 010 bloom (no adapt) | 0.370 | 7.0 | 0.45 ms |
| 001 syn (no adapt) | 0.370 | 7.0 | 0.32 ms |
| 110 adapt+bloom | **0.992** | 105 | 4.07 ms |
| 101 adapt+syn | **0.992** | 105 | 4.24 ms |
| 011 b+s (no adapt) | 0.370 | 7.0 | 0.46 ms |
| **111 all-on** | **0.992** | **105** | **4.06 ms** |

### N=100K, sel=0.1%

| Cell | recall@10 | heads_fetched | latency |
|---|---|---|---|
| 000 baseline | 0.052 | 24 | 1.03 ms |
| 100 adapt | **0.490** | 384 | 11.90 ms |
| 010 bloom (no adapt) | 0.052 | 0.6 | 0.23 ms |
| 001 syn (no adapt) | 0.052 | 0.6 | 0.11 ms |
| 110 adapt+bloom | **0.490** | 10.4 | 0.87 ms |
| 101 adapt+syn | **0.490** | 10.4 | 1.02 ms |
| 011 b+s (no adapt) | 0.052 | 0.6 | 0.23 ms |
| **111 all-on** | **0.490** | **10.4** | **0.87 ms** |

## What this changes about the story

Three statements that were *correct* in v3-partial remain correct:

- **Baseline collapse is real.** N=100K sel=0.1% baseline rec=0.052.
- **Adaptive nprobe is the recall fix.** It lifts to 0.490 at the same workload.
- **Gate-audit invariant holds.** `bad_drops = 0` across all 4 workloads,
  8 cells, 50 queries.

Three things become **clearer / cleaner**:

- **Bloom and synopsis are exactly equivalent on this workload.** Same
  recall, same `heads_fetched`, same `c_a`, same `c_b`. Both gates are
  exact for the simple `bucket=0` predicate; v3-partial showed them
  drifting only because they sat on different indexes.
- **Gates without adaptive provide pure I/O reduction, no recall impact.**
  Cells `010`/`001`/`011` have *identical* recall to cell `000`. The
  story is "gates trim work without touching recall," not "gates fix
  recall" (which v3-partial misleadingly suggested for the synopsis-only
  cell).
- **All-on cell 111 is the strict Pareto winner.** Same recall as adaptive
  alone, ~3.7× fewer heads fetched, ~3× lower latency at every workload.

## Build-time benefit

Building one index instead of four also makes the bench ~4× faster on the
build-bound workloads. At N=100K, build went from ~58 sec/flavor × 4 =
~3.9 min total in v3-partial down to ~62 sec total in v3-unified. At N=1M
the partial bench's ~16 min/flavor × 4 = ~64 min was the dominant cost; the
unified bench should finish each (N=1M, sel) workload in ~17–20 min total.

## Process

- v3-partial CSVs in this directory are overwritten by this commit. The git
  history at `4d33747c` preserves the noisy data should we want to re-do
  the empirical noise diagnosis later.
- Bench harness change is purely additive on the source side: the writer/
  reader APIs were not touched; only the bench main() body changed.
- One unused method warning (`Cell::writer_flavor`) survives — left in place
  in case we want a per-flavor mode again behind a flag.

## Next steps

1. Run N=1M sel=1% and N=1M sel=0.1% under the unified harness. ~40 min
   total wall-clock budget.
2. Re-run plots from `LOGS_PLANS/benchmarks/full_sweep_v2/plot.py` against
   v3-unified data; update `paper.md` with new tables.
3. Create Notion entry referencing this commit.
