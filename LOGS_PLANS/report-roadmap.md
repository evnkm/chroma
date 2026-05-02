# SPANN Adaptive-nprobe — Roadmap to a Comprehensive Final Report

Author: Evan Kim. Drafted 2026-05-02. Lives next to `getting-started.md` and
`spann-plan.md`. The mid-term has passed; this roadmap tracks the work
between today and the final hand-in.

## 0. Goal

Have enough substance and metrics to write a 6–10 page technical report with
the structure below. Every section needs at least one figure or table.

| § | Section                                  | What we need from the harness                          |
| - | ---------------------------------------- | ------------------------------------------------------ |
| 1 | Motivation: filtered-ANN recall problem  | HNSW pre/post-filter baseline (archive branch reuse)   |
| 2 | Failure-mode characterization            | Recall-collapse curves on multiple `n` and multiple `k`|
| 3 | Mechanism (math + wiring diagram)        | Already done. Will polish.                             |
| 4 | Empirical evaluation                     | Scale, k, filter-shape, real-dataset sweeps            |
| 5 | Cost analysis (per-phase, cost model)    | Per-phase latency instrumentation + regression         |
| 6 | Comparison to alternatives               | Oracle, iterative top-up, optional per-centroid stats  |
| 7 | Failure modes (where adaptive doesn't help) | Adversarial filter sweep                            |
| 8 | Limits & future work                     | (prose, no extra runs)                                 |

## 1. What we already have (commits on `spann-adaptive-nprobe` branch)

- `7558e7329` — bench harness + fixed-vs-adaptive on SIFT1M-10K, 50 queries.
  Mean recall, mean latency, mean returned by (sel, base_nprobe).
- `26f3d5fdd` — Rust adaptive-nprobe code change (filter selectivity threaded
  through `SpannCentersSearchInput → SpannSegmentReaderShard::rng_query →
  SpannIndexReader::rng_query → determine_search_nprobe`). Five unit tests
  on the math.
- `ec3d43b47` — `reader_adaptive` strategy validating the production code
  path against the harness math. nprobe matches exactly across 24/24 cells.

## 2. Gap analysis

Every gap below is a thing the report needs **and** the harness can produce.

| Section | Gap | Why it matters |
|---|---|---|
| §2 Failure mode | Only n=10K, only k=10 | Reviewers will ask "does this scale" and "does k matter" |
| §4 Empirical eval | Only Bernoulli filter | All real workloads have correlated filters |
| §4 | No real-dataset sweep | SIFT is geometric; semantic embeddings are different |
| §5 Cost analysis | `latency_ms` only — no per-phase | Can't explain *where* the boost cost goes |
| §5 | No tail metrics | Filtered queries are high-variance; means hide it |
| §6 Alternatives | No oracle | Can't quantify "headroom" adaptive leaves on the table |
| §6 | No iterative top-up baseline | Direct competitor — paper needs to dispatch it |
| §6 | No per-centroid prototype | Hammad/Tianyu will ask if probe-smarter beats probe-more |
| §7 Failure modes | No adversarial filter | Without it the paper looks one-sided |
| §4 | `max_factor=8` is hardcoded | Need a tuning curve to justify the default |

## 3. Deliverables (ordered for execution)

Each deliverable = (harness work) + (run) + (commit) + (Notion entry with
plots). Estimates are wall-clock for a single Apple-silicon dev box, not
including the existing 5 min release-mode rebuild.

### D1 — Per-phase + tail-metrics instrumentation

**Why first.** Every other deliverable that follows uses these columns.
Adding them later means re-running everything.

**Harness changes:**
- Add per-phase timing in `spann_filtered_recall.rs`: `t_centers_ms`,
  `t_fetch_pl_ms`, `t_bf_pl_ms`, `t_merge_ms`. Total still recorded as
  `latency_ms` for backward compat.
- Extend CSV schema (additive, won't break existing readers).
- Update `analyze.py` to emit p10/p50/p95/p99 latency and p10/p25 recall.

**Run:** Re-run the n=10K fixed + adaptive + reader_adaptive sweeps so the
new columns exist for the regression to come.

**Estimate:** 1.5 hr work + ~10 min runs.

**Output for the report:** Stacked bar chart of phase shares as nprobe
grows (Section 5 figure). Tail-latency table (Section 5 sidebar).

### D2 — Scale + k sweep

**Harness changes:** none (just `BENCH_N_RECORDS` and `BENCH_K`).

**Runs:**
- n ∈ {10K, 50K, 100K} × strategy ∈ {fixed, adaptive} × k ∈ {10, 50}.
  6 runs of 50 queries each. n=100K sweeps cost more (index build ~60s,
  inner loop with bigger posting lists slower) — budget ~90 min total.

**Output:** Figure: recall@k vs selectivity, panels by (n, k). Should show
adaptive's win narrowing for larger n (more centers → cap matters more)
and growing for larger k (min_nprobe biting). Concrete table for the
report's headline numbers.

**Estimate:** 0 hr work + ~90 min runs.

### D3 — `max_factor` sensitivity at the tail

**Harness changes:** small — accept `BENCH_MAX_FACTOR` as a sweep axis
inside one run (already an env var; just plumb into the CSV strategy
column as `adaptive-mf{factor}`).

**Run:** n=10K, base=8, sels={0.001, 0.01, 0.05}, max_factor ∈
{2, 4, 8, 16, 32, 64}. 50 queries.

**Output:** Curve of recall and latency vs `max_factor` per selectivity.
Should show diminishing returns past 16× and a knee point. Justifies the
default in the writeup.

**Estimate:** 0.5 hr work + ~10 min run.

### D4 — Filter-shape variants

**Harness changes:**
- Add `BENCH_FILTER_SHAPE` ∈ {bernoulli, categorical, range, adversarial}.
- *categorical*: assign each record a bucket label `b ∈ [0..9]`; filter =
  `b == 0` for sel=0.1, etc. (less random than Bernoulli; matters when the
  index is built on the data).
- *range*: filter = `x_norm ∈ [a, b]` where `x_norm` is the L2-norm of the
  embedding. Correlated with vector geometry — should be where adaptive
  works hardest.
- *adversarial*: for each query, mask is "exclude the true top-50". Worst
  case for adaptive (probing more doesn't help — the right candidates are
  banned).

**Run:** n=10K, base ∈ {8, 32}, sel ∈ {0.01, 0.1, 0.5}, all 4 shapes,
50 queries.

**Output:** Bar chart: recall@k by filter shape × strategy. Adaptive
should help on bernoulli/categorical/range, fail on adversarial (this is
the §7 figure). Motivates the per-centroid follow-up.

**Estimate:** 2 hr work + ~15 min runs.

### D5 — Oracle baseline (per-query optimal nprobe)

**Harness changes:** new `BENCH_STRATEGY=oracle` mode. For each query and
selectivity, sweep nprobe ∈ {8, 16, ..., max_centers} and emit one row
per nprobe. Post-process picks the smallest nprobe with recall ≥ 0.95.

**Output:** Plot: oracle nprobe vs adaptive nprobe vs fixed nprobe,
faceted by selectivity. Quantifies the headroom adaptive leaves on the
table. Section 6 anchor figure.

**Estimate:** 1 hr work + ~20 min runs (but lots of data).

### D6 — Iterative top-up (Approach 3 in `getting-started.md`)

**Harness changes:** new `BENCH_STRATEGY=topup`. Run base nprobe; if
`returned < k`, request more centers in chunks until either `k` results
exist or `nprobe ≥ base * max_factor`.

**Output:** Direct competitor table: adaptive vs top-up at the same
worst-case latency budget. Hypothesis: top-up has higher tail latency
because it runs centers search twice; adaptive predicts up front.

**Estimate:** 2 hr work + ~10 min runs.

### D7 — End-to-end orchestrator integration test

**Harness changes:** none — this is a Rust integration test, not a bench.

**Code:** New test in `rust/worker/tests/spann_filtered_recall_e2e.rs`
that builds a SPANN segment, runs `SpannKnnOrchestrator` with three filter
shapes (no filter, sparse Include, sparse Exclude(non-empty)), and asserts
that:
- with no filter, `nprobe_used == params.search_nprobe`
- with a 1% Include filter, `nprobe_used == filter_aware_nprobe(base, 0.01)`
- with a 90% Exclude filter, ditto for `filter_aware_nprobe(base, 0.1)`

The orchestrator path is currently only covered by the unit tests on
`filter_aware_nprobe`. This test closes the wiring gap.

**Estimate:** 2 hr.

### D8 — Tuning memo + config plumbing

Move `ADAPTIVE_NPROBE_EPSILON` and `ADAPTIVE_NPROBE_MAX_FACTOR` from
hardcoded consts into `InternalSpannConfiguration` so they can be tuned
per-collection. Defaults from D3.

**Estimate:** 1 hr.

### D9 — HNSW pre/post-filter baseline (re-import)

**Source:** `hnsw-baseline-archive` branch, the data at
`PLANS/benchmarks/baseline_profile.py` and the table already in the
2026-04-13 Notion entry.

**Action:** copy the 2026-04-13 Notion table as Section 1 motivation in
the writeup. No new runs.

### D10 — MSMARCO real-dataset sweep (stretch)

**Harness changes:** add a `BENCH_DATASET=msmarco` loader that reads
pre-embedded vectors from a cached `.npz`. The embed pipeline (MiniLM)
should not run inside the bench — pre-materialize once.

**Run:** n=10K, full sweep (mirror D1 but on real embeddings).

**Output:** Section 4 final-evaluation figure. Confirms the failure mode
isn't a SIFT artifact.

**Estimate:** 4 hr (mostly the embed pipeline).

### D11 — Per-centroid metadata stats prototype (stretch)

Offline in the harness only — no Rust code change. For each centroid
posting list, count how many records satisfy each filter-bucket. At query
time, *score* centroids by their predicted hit count and probe the top-k
in score order rather than distance order.

**Output:** Adaptive vs per-centroid on the adversarial filter from D4.
This is where probe-smarter should beat probe-more. Frames the future-work
section and supports Hammad's sibling workstream.

**Estimate:** 6 hr.

## 4. Out-of-scope for this iteration

- **Quantized SPANN.** Same wiring pattern would work but requires
  parallel changes in `quantized_spann_center_search.rs` and
  `quantized_spann_knn.rs`. Document as future work.
- **Production GA-quality config rollout.** D8 ships defaults for the
  report; Chroma maintainers will tune for their own workloads.
- **Inserts / writes.** This project changes only the read path.
- **PR to upstream chroma-core/chroma.** Out of project scope; will
  prepare a clean branch if asked.

## 5. Execution order and grouping

Group into commit-and-log batches:

1. **D1** alone — instrumentation foundation. Ship it before everything else.
2. **D2 + D3** — scale + tuning curve. One commit, one Notion entry with
   four plots.
3. **D4** alone — filter-shape variants are visually distinct enough to
   own one entry.
4. **D5** alone — oracle is the "headroom" entry.
5. **D6 + D7** — alternative-baselines and orchestrator-integration test.
6. **D8** — config plumbing. Code-only entry.
7. **D9** — HNSW reuse. Notion entry, no commit.
8. **D10**, **D11** — only if time permits.

Each batch follows the CLAUDE.md `Deliverable workflow`: commit → capture
hash → Notion entry under "6.5830 Final Project" with plot URLs pinned to
the commit SHA on `evnkm/chroma`.

## 6. Definition of done

The report has a figure or table for every section §1–§7, a plausibly
defensible default for `max_factor` and `eps`, and at least one
real-dataset corroboration. D1–D8 + D9 + at least one of {D10, D11}.
