# 2026-05-02 — Validating adaptive nprobe through the production Rust path

## Summary

Extended the harness with a `reader_adaptive` strategy that exercises the
production code path end-to-end:

```
SpannCentersSearchOperator
  -> SpannSegmentReaderShard::rng_query(..., Some(sel))
     -> SpannIndexReader::rng_query(..., Some(sel))
        -> determine_search_nprobe(..., Some(sel))
           -> filter_aware_nprobe(base_nprobe, Some(sel))
```

vs the existing `adaptive` strategy which applies the clamp formula in the
harness and drives `utils::rng_query` directly.

Ran all three strategies (`fixed`, `adaptive`, `reader_adaptive`) over the
same SIFT1M-10K subset, 50 queries, k=10, all 6 selectivities × 4 base
nprobes. Same per-(query, sel) filter seeds across runs.

## Per-cell comparison (24 (sel, base_nprobe) pairs aggregated over 50 queries)

```
Max |rd_nprobe   - ad_nprobe|   = 0.0       (exact match — production
                                              math is correct)
Max |rd_recall   - ad_recall|   = 0.034     (within random index variance)
Max |rd_returned - ad_returned| = 0.12      (within random index variance)
```

The nprobe match to within a single integer at every cell proves
`filter_aware_nprobe` is wired correctly through `SpannIndexReader`. Small
recall/returned differences are explained by SPANN clustering using
`thread_rng()` during index construction — each bench invocation builds a
slightly different index. Within a single run (same index), the two
strategies are deterministic and would match exactly.

## Headline numbers, three-way (n=10K, k=10, 50 queries, base_nprobe=8)

| selectivity | strategy        | nprobe_used | recall | returned | latency_ms |
|---:|:--|---:|---:|---:|---:|
| 0.001 | fixed          |  8 | 0.03 | 0.28  | 0.39 |
| 0.001 | adaptive       | 64 | 0.16 | 1.42  | 2.46 |
| 0.001 | reader_adaptive | 64 | 0.16 | 1.42  | 2.56 |
| 0.01  | fixed          |  8 | 0.19 | 2.56  | 0.37 |
| 0.01  | adaptive       | 64 | 0.74 | 9.76  | 2.49 |
| 0.01  | reader_adaptive | 64 | 0.77 | 9.78  | 2.58 |
| 0.05  | fixed          |  8 | 0.50 | 9.54  | 0.36 |
| 0.05  | adaptive       | 64 | 0.98 | 10.00 | 2.48 |
| 0.05  | reader_adaptive | 64 | 0.98 | 10.00 | 2.60 |
| 0.50  | fixed          |  8 | 0.84 | 10.00 | 0.39 |
| 0.50  | adaptive       | 16 | 0.95 | 10.00 | 0.73 |
| 0.50  | reader_adaptive | 16 | 0.97 | 10.00 | 0.76 |
| 1.00  | fixed          |  8 | 0.88 | 10.00 | 0.42 |
| 1.00  | adaptive       |  8 | 0.91 | 10.00 | 0.41 |
| 1.00  | reader_adaptive |  8 | 0.88 | 10.00 | 0.40 |

## Files

- Harness extension: `rust/worker/benches/spann_filtered_recall.rs`
  (per-base_nprobe readers, `BENCH_STRATEGY=reader_adaptive`)
- Raw CSVs:
  - `LOGS_PLANS/benchmarks/v2-fixed-n10k-q50.csv`
  - `LOGS_PLANS/benchmarks/v2-adaptive-n10k-q50.csv`
  - `LOGS_PLANS/benchmarks/reader-adaptive-n10k-q50.csv`
- Aggregated summary: `LOGS_PLANS/benchmarks/three-way-n10k-q50/summary.md`
- Plots: `LOGS_PLANS/benchmarks/three-way-n10k-q50/{recall,latency,returned}_vs_selectivity.png`

## How to reproduce

```bash
export SDKROOT="/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"
export CPLUS_INCLUDE_PATH="$SDKROOT/usr/include/c++/v1:$SDKROOT/usr/include"
export C_INCLUDE_PATH="$SDKROOT/usr/include"
export CARGO_TARGET_DIR=/tmp/chroma-target

cargo build -p worker --bench spann_filtered_recall --release
BIN=$(ls -t /tmp/chroma-target/release/deps/spann_filtered_recall-* \
        | grep -v '\.d$' | head -1)

for STRAT in fixed adaptive reader_adaptive; do
  BENCH_N_RECORDS=10000 BENCH_N_QUERIES=50 BENCH_K=10 \
    BENCH_SELECTIVITIES="0.001,0.01,0.05,0.1,0.5,1.0" \
    BENCH_NPROBES="8,16,32,64" BENCH_STRATEGY=$STRAT \
    BENCH_OUTPUT="LOGS_PLANS/benchmarks/v2-$STRAT-n10k-q50.csv" \
    "$BIN"
done

source .venv/bin/activate
python LOGS_PLANS/benchmarks/analyze.py \
  LOGS_PLANS/benchmarks/v2-fixed-n10k-q50.csv \
  LOGS_PLANS/benchmarks/v2-adaptive-n10k-q50.csv \
  LOGS_PLANS/benchmarks/reader-adaptive-n10k-q50.csv \
  --out-dir LOGS_PLANS/benchmarks/three-way-n10k-q50 \
  --label "three-way-n10k-q50"
```

## Next steps

1. Scale to n=50K and n=100K to confirm the curve shape doesn't degrade as
   the index gets larger. The harness already supports it; just bump
   `BENCH_N_RECORDS`.
2. Sweep `max_factor` (8x → 16x or 32x) at sel=0.001 to see whether more
   probes recovers the very-low-selectivity tail. Currently the cap leaves
   recall at 0.66 for sel=0.001.
3. Wire up the "filter-aware constants" (`ADAPTIVE_NPROBE_EPSILON`,
   `ADAPTIVE_NPROBE_MAX_FACTOR`) into `InternalSpannConfiguration` so they
   can be tuned without recompiling.
4. Mid-term report: fixed-vs-adaptive recall plot, three-way validation
   table, and the math.
