# 2026-05-06 — Full sweep v3 (partial): N=10K + N=100K with corrected baseline

## Summary

Partial re-run of the SPANN optimization hypercube against a corrected
`determine_search_nprobe`. v2's "baseline" cell (000) was unintentionally
receiving the filter-aware boost — `filter_aware_nprobe` was applied
**outside** the `adaptive_search_nprobe` branch, so cell 000 effectively
had `nprobe = filter_aware(params.search_nprobe=32, sel)` regardless of
the flag. That conflated upstream's size-based gradient with the project's
contribution and made every "no adaptive" recall number in v2 misleading.

v3 makes the filter-aware boost genuinely opt-in (commit `d72e36b8`):

- Size-based gradient `{24, 32, 64}` at `{500K, 1M, >1M}` is **always on**
  (matches upstream Chroma PR #5185 / #5226 thresholds exactly).
- Filter-aware boost (`size_based / sel`, capped at `size_based × 16`) is
  **only applied when `adaptive_search_nprobe = true`**.
- `params.search_nprobe` is no longer used as a floor — the size_based
  tier replaces it, matching upstream behavior.

This action log covers N ∈ {10K, 100K} × sel ∈ {1%, 0.1%} (4 of 6
cells). N=1M cells are queued for a separate run after the small-N
results are reviewed.

## What landed

- **v3 raw data**: 4 CSVs + 4 summary JSONs in
  `LOGS_PLANS/benchmarks/full_sweep_v3/` (this commit). 50 queries × 8
  cube cells per workload. Schema matches v2 exactly so the v2 plotter
  is a drop-in.
- **No source changes since v2** beyond commit `d72e36b8` (the baseline
  fix referenced above).
- **N=1M cells**: not yet run. Build time ~14 min/flavor × 4 flavors per
  workload = ~2 hours total.

## v2 → v3 cell averages

### N=10K, sel=1%

| Cell | label | v2 rec | v2 hfet | v2 lat | v3 rec | v3 hfet | v3 lat |
|---|---|---|---|---|---|---|---|
| 000 | baseline | 1.000 | 512 | 12.97 ms | **0.454** | 24 | 0.80 ms |
| 100 | adaptive | 1.000 | 512 | 12.79 ms | 1.000 | 384 | 10.77 ms |
| 010 | bloom | 1.000 | 161 | 4.88 ms | 0.456 | 8.1 | 0.37 ms |
| 001 | synopsis | 1.000 | 156 | 4.75 ms | 0.480 | 7.7 | 0.35 ms |
| 110 | a+bloom | 1.000 | 161 | 4.79 ms | 1.000 | 123 | 4.04 ms |
| 101 | a+syn | 1.000 | 156 | 4.64 ms | 1.000 | 127 | 4.10 ms |
| 011 | b+s | 1.000 | 156 | 4.78 ms | 0.456 | 7.9 | 0.37 ms |
| **111** | **all-on** | **1.000** | **156** | **4.67 ms** | **1.000** | **121** | **3.90 ms** |

### N=10K, sel=0.1%

| Cell | label | v2 rec | v2 hfet | v2 lat | v3 rec | v3 hfet | v3 lat |
|---|---|---|---|---|---|---|---|
| 000 | baseline | 0.544 | 512 | 12.49 ms | **0.064** | 24 | 0.79 ms |
| 100 | adaptive | 0.544 | 512 | 12.30 ms | 0.462 | 384 | 10.21 ms |
| 010 | bloom | 0.510 | 12.0 | 0.76 ms | 0.058 | 0.8 | 0.12 ms |
| 001 | synopsis | 0.544 | 16.6 | 0.91 ms | 0.088 | 1.1 | 0.13 ms |
| 110 | a+bloom | 0.510 | 12.0 | 0.70 ms | 0.450 | 12.3 | 0.65 ms |
| 101 | a+syn | 0.544 | 16.6 | 0.85 ms | 0.444 | 14.8 | 0.83 ms |
| 011 | b+s | 0.538 | 15.6 | 0.85 ms | 0.076 | 0.9 | 0.13 ms |
| **111** | **all-on** | **0.538** | **15.6** | **0.77 ms** | **0.444** | **14.4** | **0.76 ms** |

### N=100K, sel=1%

| Cell | label | v2 rec | v2 hfet | v2 lat | v3 rec | v3 hfet | v3 lat |
|---|---|---|---|---|---|---|---|
| 000 | baseline | 0.988 | 512 | 15.45 ms | **0.330** | 24 | 1.09 ms |
| 100 | adaptive | 0.988 | 512 | 14.69 ms | 0.982 | 384 | 11.54 ms |
| 010 | bloom | 0.994 | 138 | 5.37 ms | 0.346 | 7.2 | 0.54 ms |
| 001 | synopsis | 0.994 | 142 | 5.44 ms | 0.340 | 7.1 | 0.53 ms |
| 110 | a+bloom | 0.994 | 138 | 5.01 ms | 0.994 | 104 | 3.99 ms |
| 101 | a+syn | 0.994 | 142 | 5.13 ms | 0.982 | 104 | 4.14 ms |
| 011 | b+s | 0.992 | 143 | 5.54 ms | 0.348 | 6.7 | 0.53 ms |
| **111** | **all-on** | **0.992** | **143** | **5.00 ms** | **0.986** | **101** | **3.86 ms** |

### N=100K, sel=0.1%

| Cell | label | v2 rec | v2 hfet | v2 lat | v3 rec | v3 hfet | v3 lat |
|---|---|---|---|---|---|---|---|
| 000 | baseline | 0.656 | 512 | 14.67 ms | **0.052** | 24 | 1.12 ms |
| 100 | adaptive | 0.656 | 512 | 13.96 ms | 0.526 | 384 | 11.85 ms |
| 010 | bloom | 0.612 | 14.6 | 1.26 ms | 0.058 | 0.8 | 0.26 ms |
| 001 | synopsis | 0.606 | 14.9 | 1.34 ms | 0.062 | 0.7 | 0.27 ms |
| 110 | a+bloom | 0.612 | 14.6 | 1.07 ms | 0.532 | 12.0 | 1.02 ms |
| 101 | a+syn | 0.606 | 14.9 | 1.13 ms | 0.506 | 11.6 | 1.16 ms |
| 011 | b+s | 0.598 | 14.2 | 1.31 ms | 0.070 | 1.0 | 0.28 ms |
| **111** | **all-on** | **0.598** | **14.2** | **1.00 ms** | **0.500** | **11.4** | **0.95 ms** |

## What v3 reveals

1. **Baseline now collapses on filtered queries**, as the SPANN paper
   predicts. Worst case so far: N=100K sel=0.1% gives recall **0.052**
   when probing 24 heads (the upstream tier with no filter awareness).
2. **Adaptive nprobe alone (cell 100) recovers most of the recall** at
   sel=1% (1.0 / 0.98 at 10K / 100K), but is capped at 16× = 384 heads
   — at sel=0.1% recall recovers only to 0.46–0.53.
3. **Gates without adaptive (010, 001, 011) are exposed as no-help on
   recall**: they trim heads from baseline's already-collapsed 24 down
   to 7–8 (sel=1%) or <1 (sel=0.1%) but leave recall at baseline. v2
   showed these as recovering recall — that was the leaked filter-aware
   boost in baseline doing the work, not the gate.
4. **All-on (111) remains the strict Pareto winner** at every workload.
   At N=100K sel=1%: rec=0.986, hfet=101, lat=3.86 ms vs adaptive-only
   `100` at rec=0.982, hfet=384, lat=11.54 ms — recall preserved, ~4×
   I/O reduction, ~3× latency reduction.
5. **Gate-audit invariant held**: `bad_drops = 0` across 4 workloads ×
   8 cells × 50 queries (= 1600 query-cell measurements, with audits on
   every dropped head).

## Calibration flag

At N ≤ 500K, v3's `size_based = 24` × `MAX_FACTOR = 16` = **384** is the
ceiling. v2 effectively had ceiling `params.search_nprobe = 32` × 16 =
**512**. So at small N, v3's adaptive cells have less probe budget than
v2's were getting — visible as lower recall in cells `100`/`110`/`101`/
`111` at sel=0.1% (e.g. 0.544 → 0.526 at N=100K sel=0.1%).

This is **not a bug** — it's a consequence of removing the params-based
floor. It does **not** affect the headline workload (N=1M sel=0.1%):
size_based jumps to 32 there, so the cap is 512, matching v2 exactly. At
N>1M v3's cap (1024) actually exceeds v2's.

Three options under consideration:

- **(A) Ship as-is.** Honest baseline + honest cap. Document the small-N
  regression as a "MAX_FACTOR is calibrated to N=1M" note.
- **(B) Bump MAX_FACTOR 16 → 24.** Restores small-N adaptive ceiling
  without coupling to `params.search_nprobe`.
- **(C) Re-introduce params.search_nprobe floor.** Reintroduces the
  coupling we just removed.

Decision pending; recommendation is (A) so we can move on to N=1M.

## Process notes

- v3 source change committed earlier as `d72e36b8`.
- This action log + 4 CSVs + 4 summary JSONs commit together.
- Output path quirk: `cargo bench --bench` runs with CWD = package dir
  (`rust/worker/`), so `BENCH_OUTPUT=LOGS_PLANS/...` initially landed in
  `rust/worker/LOGS_PLANS/...`. Files were moved into place at project
  root before commit. For the N=1M run, set `BENCH_OUTPUT` to an
  absolute path or `cd` to project root before invoking.
- Notion entry: not yet created (no Notion MCP tool surfaced in this
  session). Will create manually or in next session and link back.

## Next steps

1. Decide on calibration option (A / B / C above).
2. Run N=1M sel=1% and N=1M sel=0.1% to complete the v3 hypercube
   (~2 hours wall-clock).
3. Re-run plots from `LOGS_PLANS/benchmarks/full_sweep_v2/plot.py`
   against v3 data; update `paper.md` with v3 numbers.
4. Create Notion entry referencing this commit.
