# 2026-05-02 — SPANN Baseline + Adaptive-nprobe Sweep (n=10K, 50 queries)

## Summary

Built a self-contained Rust harness (`rust/worker/benches/spann_filtered_recall.rs`)
that constructs a SPANN index over a SIFT1M subset, generates synthetic
metadata filters at controlled selectivity, runs queries against the index,
and computes filtered recall@k against an exact brute-force ground truth
restricted to filter-matching records.

Ran two sweeps over the same fixed (query, filter) set so the runs are
directly comparable:

- `strategy=fixed`    — `nprobe_used = base_nprobe`
- `strategy=adaptive` — `nprobe_used = clamp(base_nprobe / max(sel, eps), base_nprobe, base_nprobe * max_factor)`
  with `eps=0.001`, `max_factor=8`.

Both strategies are evaluated entirely in the harness (driving `rng_query`
directly). The actual Rust SPANN path change is the next deliverable.

## Headline numbers (n=10K, dim=128, k=10, 50 queries)

| selectivity | base_nprobe | strategy | nprobe_used | recall@k | returned | latency_ms |
|---:|---:|:--|---:|---:|---:|---:|
| 0.001 | 8  | fixed    | 8  | 0.04 | 0.34 | 0.41 |
| 0.001 | 8  | adaptive | 64 | 0.17 | 1.44 | 2.50 |
| 0.01  | 8  | fixed    | 8  | 0.19 | 2.42 | 0.37 |
| 0.01  | 8  | adaptive | 64 | 0.79 | 9.66 | 2.51 |
| 0.05  | 8  | fixed    | 8  | 0.48 | 9.64 | 0.36 |
| 0.05  | 8  | adaptive | 64 | 0.99 | 10.00 | 2.52 |
| 0.10  | 8  | fixed    | 8  | 0.58 | 10.00 | 0.37 |
| 0.10  | 8  | adaptive | 64 | 0.99 | 10.00 | 2.55 |
| 0.50  | 8  | fixed    | 8  | 0.85 | 10.00 | 0.39 |
| 0.50  | 8  | adaptive | 16 | 0.95 | 10.00 | 0.74 |
| 1.00  | 8  | fixed    | 8  | 0.87 | 10.00 | 0.39 |
| 1.00  | 8  | adaptive | 8  | 0.92 | 10.00 | 0.41 |

## Interpretation

- **Recall collapse at low selectivity is real.** Fixed `nprobe=8` drops to
  recall 0.19 / mean returned 2.4 at sel=0.01. Even fixed `nprobe=64` only
  recovers to 0.75 / 9.6 at sel=0.01. Most posting-list candidates get
  removed by the filter, so probing more centers is the only lever.
- **Adaptive recovers most of the recall.** At base=8, adaptive boosts to 64
  (the cap) for sel ≤ 0.01 and gets 0.79 / 0.99 / 0.99 at sel = 0.01 / 0.05 /
  0.10. At sel ≥ 0.5 the cap brings nprobe back down (no boost needed for
  unfiltered queries).
- **Latency cost of adaptive scales with the boost ratio.** At base=8, the
  adaptive path pays ~6× latency at sel=0.01 (2.5ms vs 0.4ms) — but moves
  recall from 0.19 → 0.79. At sel=1.0 it's identical to fixed.
- **Very low selectivity (sel=0.001) is still hard.** Even with adaptive at
  64 probes, recall is only 0.17 — because <10 records actually match. The
  bottleneck shifts: there are simply not enough hits in the probed centers.
  This motivates Hammad's per-centroid metadata stats as a follow-up.

## Files

- Harness: `rust/worker/benches/spann_filtered_recall.rs`
- Bench registration: `rust/worker/Cargo.toml`
- Analysis script: `LOGS_PLANS/benchmarks/analyze.py`
- Raw CSV (fixed):    `LOGS_PLANS/benchmarks/baseline-fixed-n10k-q50.csv`
- Raw CSV (adaptive): `LOGS_PLANS/benchmarks/baseline-adaptive-n10k-q50.csv`
- Aggregated summary: `LOGS_PLANS/benchmarks/baseline-n10k-q50/summary.md`
- Plots: `LOGS_PLANS/benchmarks/baseline-n10k-q50/{recall,latency,returned}_vs_selectivity.png`

## How to reproduce

```bash
export SDKROOT="/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"
export CPLUS_INCLUDE_PATH="$SDKROOT/usr/include/c++/v1:$SDKROOT/usr/include"
export C_INCLUDE_PATH="$SDKROOT/usr/include"
export CARGO_TARGET_DIR=/tmp/chroma-target

# Build once.
cargo bench -p worker --bench spann_filtered_recall --no-run

# Run fixed sweep.
BENCH_N_RECORDS=10000 BENCH_N_QUERIES=50 BENCH_K=10 \
  BENCH_SELECTIVITIES="0.001,0.01,0.05,0.1,0.5,1.0" \
  BENCH_NPROBES="8,16,32,64" BENCH_STRATEGY=fixed \
  BENCH_OUTPUT="LOGS_PLANS/benchmarks/baseline-fixed-n10k-q50.csv" \
  /tmp/chroma-target/release/deps/spann_filtered_recall-*

# Run adaptive sweep with the same (query, filter) seeds.
BENCH_N_RECORDS=10000 BENCH_N_QUERIES=50 BENCH_K=10 \
  BENCH_SELECTIVITIES="0.001,0.01,0.05,0.1,0.5,1.0" \
  BENCH_NPROBES="8,16,32,64" BENCH_STRATEGY=adaptive \
  BENCH_MAX_FACTOR=8.0 BENCH_EPSILON=0.001 \
  BENCH_OUTPUT="LOGS_PLANS/benchmarks/baseline-adaptive-n10k-q50.csv" \
  /tmp/chroma-target/release/deps/spann_filtered_recall-*

# Aggregate + plot.
source .venv/bin/activate
python LOGS_PLANS/benchmarks/analyze.py \
  LOGS_PLANS/benchmarks/baseline-fixed-n10k-q50.csv \
  LOGS_PLANS/benchmarks/baseline-adaptive-n10k-q50.csv \
  --out-dir LOGS_PLANS/benchmarks/baseline-n10k-q50 \
  --label "baseline-n10k-q50"
```

## Next steps

1. Implement adaptive `nprobe` inside the actual Rust SPANN path
   (`SpannIndexReader::determine_search_nprobe` + plumbing of filter
   selectivity from `SpannKnnOrchestrator` through `SpannCentersSearchInput`).
2. Re-run with a 50K and 100K subset to confirm the curve shape doesn't
   degrade as the index gets larger.
3. Sweep `max_factor` (8x → 16x) at sel=0.001 to see whether more probes
   recovers the very-low-selectivity tail.
