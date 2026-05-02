# SPANN Adaptive-Probe Plan (Option B)

**Written:** 2026-04-13 (pivoted late evening; midterm due 2026-04-15).

## One-line pitch

Filtered-query recall in SPANN collapses because the fixed `nprobe` selects
centroids by pure vector distance; under a restrictive metadata filter, many
probed centroids contribute zero surviving candidates. We'll make `nprobe`
**adaptive** to the filter: when the filter is restrictive, probe more
centroids so top-k is defensible; when it is permissive, keep `nprobe` at
baseline.

This is simpler than Hammad's full per-centroid metadata-statistics idea
(~200 LOC there) and composes with it later: per-centroid stats would let the
adaptive probe target the *right* centroids, not merely *more* of them.

## Target files (Rust)

| Purpose | Path |
|---|---|
| SPANN KNN orchestration | `rust/worker/src/execution/orchestration/spann_knn.rs` |
| Quantized SPANN variant | `rust/worker/src/execution/orchestration/quantized_spann_knn.rs` |
| Centroid selection operator | `rust/worker/src/execution/operators/spann_centers_search.rs` |
| Posting-list fetch | `rust/worker/src/execution/operators/spann_fetch_pl.rs` |
| Brute-force within posting list | `rust/worker/src/execution/operators/spann_bf_pl.rs` |
| KNN filter | `rust/worker/src/execution/orchestration/knn_filter.rs` |
| SPANN index (if needed) | `rust/index/src/spann/` |

(Paths discovered but not yet read. First real task is mapping the data flow.)

## Core idea

Today (roughly):
```
query vector q, filter F, requested k, fixed nprobe = N
  -> centers_search(q, N)       -> set of N centroids
  -> fetch_pl(centroids)         -> candidate ids
  -> apply bitmask(F)            -> survivors
  -> brute_force(q, survivors)   -> top-k
```

Problem: if `|survivors| << k`, recall tanks.

Proposed:
```
query vector q, filter F, requested k
  sel = estimate_selectivity(F)     # cheap, from metadata index counts
  nprobe_adaptive = f(sel, N, k)    # e.g. N * clip(1/sel, 1, MAX)
  -> centers_search(q, nprobe_adaptive)
  -> (rest unchanged)
```

Selectivity estimation comes from a call the metadata index already supports
(COUNT of matching ids). Centroid probing is the existing op; we just change
its count argument.

## What counts as "done" for midterm (Tuesday)

Honesty first: 2 days is not enough for a working Rust patch + benchmarks. The
midterm expectations from Tianyu are "a range of results from things tried so
far — no need for a cohesive story yet." Goal for midterm:

1. HNSW baseline results from today as preamble ("this is the general
   problem shape; SPANN inherits a version of it").
2. SPANN query-path map: a diagram of the orchestration + operators and
   which operator the metadata bitmask is applied in.
3. One early experiment: measure filtered-recall degradation as a function of
   filter selectivity on a stock SPANN query (just reading numbers from the
   existing code, no new logic yet).
4. Proposed mechanism + cost model sketch.

## Plan of attack (in order)

**P0 (now) — env + space.**
- `cargo clean` (reclaim 19 GB; we'll rebuild when we start on Rust).
- Delete the Python `PLANS/planner/` experimental code; keep
  `chromadb/execution/executor/profiling.py` and the instrumentation block in
  `chromadb/execution/executor/local.py` for later reuse / reference.
- Keep `PLANS/benchmarks/baseline_profile.py` and the CSV results as the
  HNSW baseline artifacts for midterm.

**P1 — map SPANN data flow.** Read-only pass with Explore agent:
- Orchestrator: how `spann_knn.rs` fans the query through operators.
- Where is `nprobe` configured? (likely in collection config or a knob).
- Where does the metadata filter enter? Before centroid search, after, or
  inside posting-list scan?
- What does `spann_centers_search` actually return?
- What stats does the metadata index expose (counts by key, histograms)?

Output: a short diagram + pointer list under
`PLANS/spann-query-path.md`.

**P2 — build what's needed.** Only after P1. Likely:
- Full-stack dev loop: Tilt or docker-compose.
- `maturin dev` so the Python client can hit the Rust worker.

**P3 — instrument.** Same philosophy as today's Python profiling:
- Per-phase timers (centers_search, fetch_pl, apply_filter, brute_force).
- Record: nprobe, n_candidates_before_filter, n_candidates_after_filter,
  returned_k, true_selectivity, recall_proxy.
- Gate behind a tracing flag / env var so it's zero-cost off.

**P4 — first experiment.** Run synthetic + MS MARCO through SPANN with a
  sweep over selectivity. Confirm recall collapse at low selectivity. This is
  the "before" chart of the mid-term story.

**P5 — adaptive nprobe.** Implement the heuristic
  `nprobe' = clip(N / max(sel, eps), N, N * MAX_FACTOR)`. A/B against fixed
  nprobe at equal-latency and equal-nprobe budgets.

**P6 — cost-model framing.** Express the probe-count choice as a cost
  optimization: minimize (latency(nprobe)) subject to (expected_recall >=
  target). `expected_recall` ≈ P(at least k survivors) given `sel` and
  centroid candidate distribution.

## Deliberately out of scope

- Per-centroid metadata statistics (Hammad's idea) — kept as a follow-up that
  composes with this plan. We don't need them to show the probe-count win.
- SPFresh updates (incremental index maintenance).
- Learned cost model.

## Open questions (ask team / advisor)

- Ownership: Evan on adaptive nprobe; Kartik and Zach take other SPANN
  ideas (centroid stats? hybrid? something else)? Proposal says parallel.
- Is there an existing "selectivity estimator" in the metadata index, or do
  we need to add one?
- What does Chroma set `nprobe` to by default, and is it per-collection?
