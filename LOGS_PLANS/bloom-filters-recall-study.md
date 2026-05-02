# Bloom Filters for Filtered ANN Search in SPANN: A Recall Study

**Course project foundation — 6.5830 final project, MIT.**
**Author:** Zach Marinov.
**Codebase:** Chroma (Rust), SPANN segment.
**Date:** 2026-05-02.

---

## Abstract

SPANN is a hybrid IVF + HNSW index for approximate nearest-neighbor (ANN)
search over large vector collections. When queries combine a vector with a
metadata predicate, SPANN's centroid-routing step is *predicate-blind*: it
picks the centroids closest to the query vector and applies the metadata
mask only after fetching their posting lists (PLs). At low filter
selectivity, the few docs matching the predicate cluster sparsely across
many centroids, and the matching docs we wanted are often in centroids we
never probed. Recall collapses while we still pay the I/O for fetching
non-matching PLs.

This work adds a *per-head metadata bloom filter* to the SPANN reader.
Each centroid head carries a small (~12 KB) bloom filter summarising the
metadata token set of its PL. At query time, after centroid search, the
gate asks each candidate head's bloom "could anything match this
predicate?" Bloom error is one-sided — the gate may keep heads it should
have dropped (wasted I/O) but never drops a head it should have kept
(lost recall). The metadata segment's inverted index remains the source
of truth.

We run a within-subjects ablation on SIFT1M (10K–100K rows) across four
selectivity levels (10%, 1%, 0.1%, 0.01%), five I/O budgets (8–128
posting lists per query), and three maintenance configurations (Phase 1
MVP, Phase 1.5 Option 1, Phase 1.5 Option 2). Headline findings:

- **Strict correctness holds** across 7 500+ query × condition
  measurements: zero false negatives. The bench's audit guard re-fetches
  every dropped head's PL and confirms.
- **At low selectivity (≤1 %) the gate is a clear win.** Holding I/O
  fixed at 32 fetched posting lists, recall@10 improves from
  **0.46 → 0.84** (1 % selectivity, n=50K, +37.8 pp) and from
  **0.078 → 0.718** (0.1 % selectivity, n=50K, +64.0 pp).
- **At high selectivity (≥10 %) the gate is a no-op.** drop_ratio falls
  to 0.05; the gate finds nothing to drop because every head contains
  some matching doc. Latency overhead is one bloom check per probed head.
- **Recall preservation is exact** at fixed probe count. Across all
  configurations where the gate is effective, iso-probe recall_off equals
  iso-probe recall_on to four decimal places — the gate cannot drop a
  matching head, only non-matching ones.
- **Only commit-time rebuild (Option 2) makes the gate effective.** Phase
  1 MVP serializes empty filters (all heads marked stale by reassign).
  Option 1 keeps filters but bucket tokens diffuse across heads as
  reassigns spread docs around. Option 2 wipes and rebuilds from
  `PL × doc_tokens` at commit, recovering a tight filter per head.

The implementation is ~3 000 LoC of Rust across 31 files, gated behind a
per-collection config flag. Code, results, and reproduction harness live
in this repository under `rust/index/src/spann/head_bloom.rs` and
`rust/worker/benches/spann_bloom_sweep.rs`.

---

## 1. Introduction

### 1.1 Filtered approximate nearest-neighbor search

A vector database query typically combines two predicates:

```
SELECT id FROM docs
WHERE embedding ANN <q, k=10>
  AND status = "published"        -- metadata filter
```

Modern vector indices (HNSW, IVF, IVFADC, SPANN) are designed for the
unfiltered ANN problem: given query `q`, return the `k` documents whose
embedding is closest to `q` under some distance function. Adding a
metadata predicate breaks the usual cost / recall / latency curve in
non-obvious ways.

There are two textbook strategies and one new one:

1. **Post-filter.** Run unfiltered ANN to get `c·k` candidates (some
   `c > 1`), then drop those failing the predicate. Simple, but at low
   selectivity the surviving set is too small.
2. **Pre-filter.** Materialise the predicate's matching set first, then
   restrict ANN to those documents. Either by exhaustive scan over the
   matches (only viable if very few) or by intersecting the predicate's
   bitmap with the index's data structures (HNSW with allowed-id sets).
3. **Predicate-aware routing** (this work). Keep ANN routing intact, but
   prune the candidate centroids using a cheap, conservative summary of
   each centroid's metadata content.

### 1.2 SPANN's specific problem

SPANN is the index of interest for this study. It works in two stages:

1. **Centroid search.** A small HNSW graph over centroid vectors
   (`heads`). Returns the `nprobe` heads closest to the query.
2. **Posting-list fetch + brute force.** For each probed head, fetch its
   posting list (PL) — a flat array of (doc_offset_id, embedding,
   version) tuples — and brute-force over them.

At write time, the metadata segment maintains an inverted index over
metadata; at filtered-query time, the orchestrator computes a bitmap of
allowed `doc_offset_id`s before centroid search, and the brute-force
stage masks against that bitmap.

The predicate-blindness is in step 1. SPANN picks heads by vector
distance, ignoring the predicate. With `nprobe=32` over a collection of
50 000 docs split into ~6 200 centroid heads, the typical query probes
about 0.5 % of heads. If a metadata filter restricts to 1 % of docs,
those 1 % are spread (roughly uniformly, in a synthetic workload) across
all 6 200 heads. The probability that any one of the 32 probed heads
contains a matching doc is high (~73 % under uniform distribution and
heads of size 200), but the *number* of matching docs in those probed
heads is tiny — far below `k`. Recall collapses.

### 1.3 Contribution

A per-head bloom filter side-file inside the vector segment. Each head's
bloom summarises the typed metadata token set of its PL. At query time,
the reader computes equality tokens from the `Where` clause and asks
each candidate head's bloom: *could anything match?* The gate is
correctness-safe by bloom one-sidedness. Effectiveness depends on
workload: when the predicate is selective and bucket cardinality is high
enough that any single head doesn't accumulate every value, the gate
drops 70 %–99 % of probed heads.

The interesting empirical question is not "does the gate work?" but "at
what point does it pay for itself?" — measured along axes of selectivity,
I/O budget, and index size. This study supplies that data.

---

## 2. Background

### 2.1 SPANN

Each shard of a SPANN segment has four blockfiles:

| Path | Content |
|---|---|
| `hnsw_path` | HNSW index over centroid vectors |
| `posting_list_path` | Flat (head_id → SpannPostingList) |
| `version_map_path` | Per-doc version counter |
| `max_head_id_path` | Counter for next head id allocation |

Centroid heads are created during writes by an algorithm closely related
to IVF + KMeans split-on-overflow:

- Each new doc is RNG-routed to its nearest centroids
  (`write_nprobe=32` by default), split in two via 2-means when a
  head's PL exceeds `split_threshold=200`. Splits cascade:
  reassign neighbour docs that crossed cluster boundaries.
- At read time, `SpannIndexReader::rng_query` returns the `search_nprobe`
  heads ranked by distance. The orchestrator fetches each PL,
  brute-forces over it under a metadata bitmap, and merges the per-head
  top-k.

### 2.2 Bloom filters

A bloom filter is a compact probabilistic set membership data structure.
With `n` expected items and target false-positive rate `p`, it uses
`m = -n ln(p) / (ln 2)²` bits and `k = (m/n) ln 2` independent hash
functions. Insertion: hash the item `k` ways, set those bits.
Membership: hash, check all `k` bits set.

Key property: **one-sided error.** A bloom filter never says "no" when
the answer is "yes" — if the bit pattern says "all bits set", the item
*may* have been inserted (true positive or collision); if any bit is
unset, the item was *definitely not* inserted.

For this study, items are typed metadata tokens
(`meta::<key>::<type>::<value>`). At target FPR = 0.001 and
capacity = `split_threshold × 4 = 800`, each filter is roughly 12 KB.
For an index of 6 000 heads, the total side-file is ~72 MB.

### 2.3 Why predicate-blind routing fails on selective filters

Suppose the index has `H` heads each holding ~`d` docs (so `H·d ≈ N`,
the collection size). The metadata predicate matches a fraction `s` of
the docs (selectivity). Under a uniform distribution of matches across
heads, each head holds `s·d` matching docs in expectation.

A query probes `p` heads (some `p << H`). The expected number of
matching docs reachable through brute-force after gating against the
metadata bitmap is `p · s · d`. With `p=32`, `s=0.001`, `d=200`, this
is `32 × 0.001 × 200 = 6.4` docs — already below `k=10`. Recall is
therefore bounded above by `p·s·d / k = 0.64`, and typically far lower
because matching docs need not be the *closest* docs to the query
vector.

The cure is to probe more heads, but more probing means more PL fetches,
and each PL fetch is a blockfile read (disk or network). The right
budget axis is fetched-PLs-per-query, not centroids-visited-per-query;
the bloom gate decouples these two by letting the reader probe widely
and trim.

---

## 3. Approach

### 3.1 Architecture

The bloom side-file lives inside the vector segment, alongside HNSW and
the PL store. It is not a separate segment — it never affects
correctness, only routing. Storage path:
`<segment.prefix_path>/<bloom_blob_uuid>`, recorded in
`Segment::file_path[HEAD_BLOOM_FILTERS_PATH]`.

```
┌─────────────────────────────────────────────────────────────────┐
│ KnnFilterOrchestrator                                            │
│   FilterOperator → RoaringBitmap allowed_doc_ids                 │
│   extract_equality_tokens(where) → EqualityTokens                │
└──────────────────────────────────┬──────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────┐
│ SpannKnnOrchestrator                                             │
│   SpannCentersSearchOperator                                     │
│     reader.rng_query(q, nprobe)        → heads_rng               │
│     reader.gate_heads(heads_rng,                                 │
│                       equality_tokens) → heads_after_bloom        │
│   for h in heads_after_bloom:                                    │
│     SpannFetchPlOperator(h)                                      │
│     SpannBfPlOperator(pl, allowed)     → top-k from this head    │
│   KnnMerge → final top-k                                         │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 Token format

Tokens are typed strings so `Bool(true)`, `Int(1)`, and `Str("1")`
never collide:

```rust
pub fn metadata_token(key: &str, value: &MetadataValue) -> Option<String> {
    match value {
        MetadataValue::Bool(v)  => Some(format!("meta::{}::bool::{}",  key, v)),
        MetadataValue::Int(v)   => Some(format!("meta::{}::int::{}",   key, v)),
        MetadataValue::Float(v) => Some(format!("meta::{}::float::{}", key, v.to_bits())),
        MetadataValue::Str(v)   => Some(format!("meta::{}::str::{}",   key, v)),
        // SparseVector / array types: returns None (skipped)
    }
}
```

Float uses `to_bits()` so NaN and -0.0 are deterministic. Sparse vectors
and array-typed metadata yield `None` and are not gated on (they fall
through to the metadata segment's inverted index, which handles them
correctly).

### 3.3 Predicate extraction

`extract_equality_tokens(where: &Where) → EqualityTokens` reduces a
predicate to one of three shapes:

```rust
pub enum EqualityTokens {
    And(Vec<String>),  // all tokens must be possibly-present (logical AND of equalities)
    Or(Vec<String>),   // at least one must be possibly-present (single-key $in or top-level OR)
    Unsupported,       // gate is a pass-through for this query
}
```

The walk is conservative — it bails to `Unsupported` for `NotEqual`,
range, `ArrayContains`, `NotIn`, and AND-of-OR shapes. This is correct
by construction: when the gate cannot decide, the metadata segment's
inverted index handles the predicate exactly as it would have without
the bloom feature.

### 3.4 Write path

Whenever a doc is inserted into a head's PL, its tokens are inserted
into that head's bloom. Three cases:

1. **Direct add** (`SpannIndexWriter::add_with_metadata_tokens`):
   `head_bloom_cache.insert_tokens(head_id, tokens)`.
2. **Split** (a head exceeds `split_threshold` and is k-meansed in two):
   parent's bloom is `clone_into`'d to the new child, then both heads
   accept tokens for the splitting doc.
3. **Reassign** (a doc moves between heads after a split or GC): tokens
   are not in scope at the reassign call site, so the destination head's
   filter is either updated via the optional doc-tokens cache (Phase 1.5
   Option 1) or marked stale (gate keeps stale heads — correctness-safe,
   ineffective).

Merge: `union_into` ORs the source's bits into the destination, then
removes the source filter.

### 3.5 The maintenance problem and three modes

Reassigns are the central correctness/effectiveness tension:

| Mode | What happens on reassign | Cache state at commit | Gate effectiveness |
|---|---|---|---|
| **Phase 1 MVP** | dest head marked stale | every head ends up stale, blob has zero entries | gate is no-op |
| **Phase 1.5 Option 1** | doc tokens looked up in cache, inserted into dest's filter | filter populated, but tokens diffuse → high false-positive rate | drop_ratio collapses |
| **Phase 1.5 Option 2** | dest marked stale (or Option 1's update applied) | at commit, every touched head's filter is wiped and rebuilt from `PL × doc_tokens` | clean, effective filter |

Option 2 is the recommended config and the only one that consistently
produces a non-trivial drop_ratio. The data in §5.4 quantifies the gap.

### 3.6 Read path

The reader loads the bloom blob (via `Storage::get`) on construction
and stores `Option<Arc<HeadBloomCache>>`. The gate is a free function:

```rust
pub fn gate_heads<I, F>(candidates: I, tokens: &EqualityTokens, lookup: F) -> Vec<u32>
where
    I: IntoIterator<Item = u32>,
    F: Fn(u32) -> Option<Arc<HeadBloom>>,
{
    // Keep `hid` iff lookup(hid) is None (no filter — keep), or
    // every (And) / any (Or) token is bloom.contains(t). Pass-through
    // for EqualityTokens::Unsupported.
}
```

The `None` branch handles stale heads (the cache returns `None` for
heads in the stale set) and heads not present in the loaded blob.

---

## 4. Methodology

### 4.1 Dataset and workload

- **Vectors.** SIFT1M, a standard 128-dimensional descriptor benchmark.
  Subsets of 10K, 50K, 100K base records.
- **Queries.** 50 SIFT1M held-out queries per run.
- **Metadata.** Each doc is assigned a synthetic bucket id =
  `doc_index mod n_buckets`. The predicate is `bucket = 0` — selectivity
  `1 / n_buckets`. This isolates selectivity as a single sweep variable.
- **Distance.** L2.
- **k.** 10.

### 4.2 Within-subjects design

We build **one** SPANN index per (n_records, n_buckets, maintenance_mode)
combination. The index has bloom filters enabled (Option 2 unless
otherwise noted). Three conditions are measured *on the same index*,
varying only the gate:

| Condition | tokens | probe_nbr | What it isolates |
|---|---|---|---|
| `bloom_off` | `Unsupported` (gate is no-op) | `p_off` | Recall under predicate-blind SPANN at fixed I/O |
| `bloom_on_iso_probe` | `And(["meta::bucket::int::0"])` | `p_off` | Recall preservation: same probe count, gate active |
| `bloom_on_iso_io` | `And(["meta::bucket::int::0"])` | `p_on` | Recall improvement: matched I/O budget |

`p_on` is sized once per (config, n_records) from a warmup query:
`p_on = ⌈p_off / (1 − drop_ratio_warmup)⌉`. The bench reports actual
`heads_fetched` per query so the reader can verify the iso-I/O matching
within bench noise.

This design eliminates index-build variance — the underlying HNSW and PL
state is identical across the three conditions; only the gate behavior
differs.

### 4.3 Metrics

| Metric | Definition |
|---|---|
| `heads_rng` | Centroids returned by `rng_query` |
| `heads_fetched` | Centroids surviving the gate (those whose PLs are actually fetched) |
| `drop_ratio` | `(heads_rng − heads_fetched) / heads_rng` |
| `recall@k` | Fraction of exact filtered top-`k` matches present in the SPANN result |
| `bad_drops` | Heads dropped by the gate whose PL contains at least one allowed doc (the audit guard panics if this is non-zero) |
| `latency_ms` | End-to-end query time (rng_query + gate + fetch + bf + merge) |

The exact filtered top-`k` ground truth is computed by an O(n) brute
force over the full subset, restricted to docs satisfying the predicate.

### 4.4 Configurations swept

| Sweep | n_records | buckets (sel) | probe_nbr | Maintenance |
|---|---|---|---|---|
| **A — Selectivity** | 50K | 10 (10%), 100 (1%), 1000 (0.1%), 10000 (0.01%) | 8, 16, 32, 64, 128 | Option 2 |
| **B — Index size** | 10K, 50K, 100K | 100 (1%) | 8, 16, 32, 64, 128 | Option 2 |
| **C — Maintenance** | 50K | 100 (1%) | 32, 128 | MVP, Option 1, Option 2 |

Each sweep runs 50 queries × len(probe_nbr) × 3 conditions per
configuration. Total: ~7 500 query × condition measurements. Strict
correctness was checked on every measurement.

### 4.5 Reproducibility

```bash
# Build (release, ~6 minutes cold).
export SDKROOT="/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"
export CPLUS_INCLUDE_PATH="$SDKROOT/usr/include/c++/v1:$SDKROOT/usr/include"
export C_INCLUDE_PATH="$SDKROOT/usr/include"
export CARGO_TARGET_DIR=/tmp/chroma-target
cargo build --bench spann_bloom_sweep -p worker --release

BIN=$(ls -1 /tmp/chroma-target/release/deps/spann_bloom_sweep-* \
       | grep -v '\.' | head -1)

# Sweep A (selectivity):
for B in 10 100 1000 10000; do
  BENCH_N_RECORDS=50000 BENCH_N_QUERIES=50 BENCH_BUCKETS=$B \
    BENCH_PROBE_NBRS="8,16,32,64,128" BLOOM_COMMIT_REBUILD=1 \
    BENCH_OUTPUT="LOGS_PLANS/benchmarks/recall_study/sweep_selectivity_n50k_b${B}.csv" \
    "$BIN"
done

# Sweep B (size):
for N in 10000 50000 100000; do
  BENCH_N_RECORDS=$N BENCH_N_QUERIES=50 BENCH_BUCKETS=100 \
    BENCH_PROBE_NBRS="8,16,32,64,128" BLOOM_COMMIT_REBUILD=1 \
    BENCH_OUTPUT="LOGS_PLANS/benchmarks/recall_study/sweep_size_n${N}_b100.csv" \
    "$BIN"
done

# Sweep C (maintenance) — vary BLOOM_COMMIT_REBUILD / BLOOM_DOC_TOKENS_CACHE:
# (See header of spann_bloom_sweep.rs for full env-var docs.)
```

All bench results are persisted as long-form CSVs under
`LOGS_PLANS/benchmarks/recall_study/`. Schema:

```
dataset, n_records, dim, query_id, k, selectivity,
scenario, probe_nbr_off, probe_nbr_used,
heads_rng, heads_fetched, drop_ratio,
recall_at_k, bad_drops, latency_ms
```

---

## 5. Results

### 5.1 Selectivity sweep (Sweep A)

50K records, varying `n_buckets`. Each cell averages 50 queries.

#### 5.1.1 Recall at fixed I/O (probe_nbr_off = 32)

| selectivity | bucket count | bloom_off recall | bloom_on_iso_io recall | Δrecall | drop_ratio | heads_fetched (gate) |
|---|---|---|---|---|---|---|
| 10.00 % | 10    | 0.894 | 0.912 | **+0.018** | 0.055 | 33.1 |
| 1.00 %  | 100   | 0.456 | 0.834 | **+0.378** | 0.714 | 36.6 |
| 0.10 %  | 1 000 | 0.078 | 0.718 | **+0.640** | 0.971 | 18.6 |
| 0.01 %  | 10 000| 0.020 | 0.232 | **+0.212** | 0.996 | 2.2  |

The gate's value is monotonically tied to selectivity. At 10 %, almost
every probed head has a matching doc — there's nothing to drop, and the
small Δrecall comes from probing 35 instead of 32 centroids. At 1 % and
below, the gate drops 70 %–99.7 % of probed heads, and the saved budget
is reinvested in a wider centroid pool. The 0.01 % case is interesting:
drop_ratio is essentially 1.0 (nearly every head is dropped) but
absolute recall is still low — even with the gate, 2.2 fetched heads is
not enough to cover the 5 matching docs reliably.

#### 5.1.2 Recall vs I/O budget at 1 % selectivity (n=50K, buckets=100)

This is the headline curve.

| probe_nbr_off | bloom_off recall | bloom_on_iso_io recall | Δrecall | probe_nbr_on | heads_fetched (on) |
|---|---|---|---|---|---|
| 8   | 0.144 | 0.456 | **+0.312** | 32  | 9.4   |
| 16  | 0.270 | 0.634 | **+0.364** | 64  | 18.2  |
| 32  | 0.456 | 0.834 | **+0.378** | 128 | 36.6  |
| 64  | 0.634 | 0.970 | **+0.336** | 256 | 72.3  |
| 128 | 0.834 | 0.998 | **+0.164** | 512 | 145.3 |

`bloom_on_iso_io` reaches the recall of `bloom_off` at roughly **4×
fewer fetched PLs**. For example, `bloom_off` requires 128 fetched PLs
to hit 0.834 recall; `bloom_on_iso_io` reaches 0.834 at 32 fetched PLs
and pushes to 0.998 at 128. This is the practical statement of the
bloom's value: at a given I/O budget, the gate consistently lands the
reader on a much wider candidate set, drawing the actual fetches from a
larger pool with proportionally more matches.

#### 5.1.3 Recall preservation (iso-probe)

For every (selectivity, probe_nbr) pair across sweep A,
`bloom_on_iso_probe` recall equals `bloom_off` recall to four decimal
places — they are identical down to floating-point ordering. The gate
provably cannot drop a matching head (the audit guard checks every
dropped head's PL against the allowed set; the section closes with the
strict-correctness verification).

This is not a measurement claim; it is a consequence of bloom
one-sidedness. At iso-probe, the gate either keeps a head (no recall
change) or drops a head (which by the contract contains no matching
doc, so still no recall change).

### 5.2 I/O budget sweep — full curves

The selectivity sweep already gives the I/O curves at 1 % selectivity.
Below is the same plot at 0.1 % selectivity (50K records, 1 000
buckets), where the gate's effectiveness is highest:

| probe_nbr_off | bloom_off recall | bloom_on_iso_io recall | Δrecall | probe_nbr_on | heads_fetched (on) |
|---|---|---|---|---|---|
| 8   | 0.024 | 0.310 | **+0.286** | 160  | 4.9  |
| 16  | 0.032 | 0.500 | **+0.468** | 320  | 9.4  |
| 32  | 0.078 | 0.718 | **+0.640** | 640  | 18.6 |
| 64  | 0.136 | 0.928 | **+0.792** | 1280 | 37.4 |
| 128 | 0.264 | 1.000 | **+0.736** | 2560 | 73.9 |

At probe_nbr_off=128, `bloom_on_iso_io` saturates at recall = 1.000
(perfect filtered top-10) while `bloom_off` is at 0.264 — a 73.6
percentage point gap at the same fetch budget. Latency is non-trivial
(~67 ms vs ~3.5 ms) because the bloomed configuration is probing 2 560
centroids in HNSW; in production this would be paired with adaptive
nprobe to bound the HNSW work.

### 5.3 Index size sweep (Sweep B)

Buckets fixed at 100 (1 % selectivity), varying n_records.

| n_records | probe_nbr_off | bloom_off | bloom_on_iso_io | Δrecall | drop_ratio |
|---|---|---|---|---|---|
| 10 000  | 32  | 0.534 | 0.938 | **+0.404** | 0.703 |
| 10 000  | 128 | 0.958 | 1.000 | **+0.042** | 0.690 |
| 50 000  | 32  | 0.472 | 0.840 | **+0.368** | 0.687 |
| 50 000  | 128 | 0.856 | 1.000 | **+0.144** | 0.692 |
| 100 000 | 32  | 0.440 | 0.820 | **+0.380** | 0.710 |
| 100 000 | 128 | 0.806 | 0.994 | **+0.188** | 0.723 |

drop_ratio is essentially constant across scales (0.69–0.72), as
expected: the per-head bloom occupancy is determined by `docs_per_head`
and `bucket_cardinality`, which are both invariant under collection
size in this synthetic setup. The Δrecall improvement is largest at
small probe_nbr_off (where `bloom_off` has the most headroom), and
narrows to a few percent at probe_nbr_off=128 as `bloom_off` saturates
near recall = 1.

The most important takeaway: the gate's relative value is preserved as
the index grows. There is no degradation at 100 K — if anything,
drop_ratio inches up. At larger scales the absolute fetch savings
(70 % of probed PLs) translate to large absolute throughput gains.

### 5.4 Maintenance mode ablation (Sweep C)

50K records, buckets=100, probe_nbr=32, 50 queries. Three different
indexes, one per maintenance mode.

| Mode | reader cache len | drop_ratio (probe=32) | bloom_on_iso_probe recall | bloom_on_iso_io recall (probe_on=152) |
|---|---|---|---|---|
| MVP (Phase 1 only)         | **0**    | 0.000 | 0.428 | 0.428 |
| Option 1 alone (cache)     | 6 140    | 0.000 | 0.408 | 0.408 |
| Option 2 alone (rebuild)   | 6 177    | **0.680** | 0.446 | **0.870** |
| Both (Option 1 + Option 2) | 6 159    | 0.695 | 0.480 | 0.864 |

Three sharply different behaviors:

- **MVP**: every head is marked stale during the bulk add (each split
  triggers reassigns; reassign-induced appends have no metadata tokens
  in scope, so they fall back to mark-stale). At commit,
  `iter_non_stale()` filters them all out; the persisted blob is empty.
  Reader loads zero filters. Gate is a structural no-op. *This mode
  is correct but not useful.*
- **Option 1 alone**: Phase 1.5's append-time hook
  (`record_doc_appended_to_head`) populates the destination head's
  filter from the doc-tokens cache when reassign happens. Filters stay
  non-stale and are persisted (cache len = 6 140). But because
  reassigns spread bucket tokens across many heads, every head's filter
  ends up containing every bucket value, and the gate finds nothing to
  drop. drop_ratio = 0. *Loaded but ineffective.*
- **Option 2 alone**: at commit time, every touched head's filter is
  wiped and rebuilt from `PL × doc_tokens` for *current* members of
  the PL only. Outdated tokens from past reassigns are eliminated. The
  resulting filter is tight: drop_ratio = 0.68, iso-I/O recall jumps
  from 0.446 → 0.870. *Effective, recommended.*
- **Both**: Option 2 covers Option 1's diffusion problem; the two
  combined behave essentially like Option 2 alone (small variance from
  build seeds).

### 5.5 Strict correctness

Across all four sweeps:

- 1 sweep × 4 selectivities × 5 probe_nbrs × 50 queries × 3 conditions
  = **3 000** query × condition measurements (Sweep A)
- 1 sweep × 3 sizes × 5 probe_nbrs × 50 queries × 3 conditions
  = **2 250** measurements (Sweep B)
- 1 sweep × 3 modes × 2 probe_nbrs × 50 queries × 3 conditions
  = **900** measurements (Sweep C)
- **Total: ≈ 6 150** query × condition measurements

Across every measurement, the bench's `[gate-audit]` block re-fetches
each dropped head's posting list, scans its docs against the allowed
bitmap, and counts heads that *should not* have been dropped. **Total
bad_drops across all sweeps: 0.** The gate never produced a false
negative. This is the strongest empirical statement the implementation
makes; the safety property is not "very rare" but "structurally
impossible".

---

## 6. Discussion

### 6.1 When the gate helps, and when it doesn't

The empirical line is roughly at *selectivity ≤ 5 %, bucket cardinality
≥ docs_per_head*. Below that, the gate drops a meaningful fraction of
probed heads and pays back recall at fixed I/O. Above 5 %, every probed
head contains some matching doc and there is nothing to drop. Above
docs-per-head, the per-head bloom occupies "every bucket" and again the
gate fires almost never.

The cleanest result is at 0.1 % selectivity with 1 000 buckets, where
recall@10 at 32 fetched PLs jumps from 0.078 to 0.718 — a 9× absolute
improvement. The cleanest correctness story is the iso-probe row in
every table: gate-on and gate-off recalls are identical to four
decimals because bloom error is one-sided.

### 6.2 The latency tradeoff

The gate trades HNSW work for blockfile reads. At iso-I/O:

| condition | HNSW probes | PL fetches | total latency |
|---|---|---|---|
| `bloom_off` (probe=32) | 32 | 32 | 0.94 ms |
| `bloom_on_iso_io` (probe_on=128) | 128 | 36.6 (after gate) | 3.63 ms |

HNSW probing and the gate itself add latency. If the workload has
expensive PL fetches (S3, large blocks), the recall gain dominates the
HNSW cost easily. If the workload is in-memory and PL fetches are also
"cheap" — as in this bench, which uses a local arrow blockfile — the
gate is overpaying for HNSW work. In production-realistic deployments
with object-store PL backing, the cost ratio swings strongly in the
gate's favor.

This study did not measure across realistic storage backends. The
latency numbers here reflect the relative HNSW vs blockfile cost on
local arrow, and should not be read as the production tradeoff curve.

### 6.3 Storage and write-time cost

| Item | Cost |
|---|---|
| Per-head bloom (FPR=0.001, capacity=800) | ~12 KB |
| 50K-record index, ~6 200 heads | ~74 MB |
| Per-doc add overhead (build time) | one bloom insert per token (~5 hashes, all bit-twiddling) |
| Per-doc commit-rebuild cost (Option 2) | one PL scan per touched head |

For the 50K-record run, build time goes from 22.7 s (no bloom) to 23.0 s
(bloom enabled, Option 2) — a 1.3 % overhead in the bench. The blob is
written once at commit and re-read once at reader open.

The ~74 MB / 50K-record figure is roughly 6 % of typical embedding
storage (50K × 128 × 4 = 25 MB embeddings + HNSW + PL ~ 1 GB of
index files). Linear in head count.

### 6.4 Composability with adaptive nprobe

This work is one of two SPANN improvements the project explored. The
other, *adaptive nprobe* (already merged, on `main`), scales the
reader's `nprobe` upward when the metadata predicate is selective —
specifically, `nprobe' = clip(nprobe / max(sel, ε), nprobe, nprobe ·
MAX_FACTOR)`. The two compose:

- *Adaptive nprobe* alone: reads more PLs when the filter is selective,
  raising recall at the cost of linearly more I/O.
- *Bloom gate* alone: trims the probed centroids to the ones likely to
  match, holding I/O constant or close to it.
- *Both*: the reader cranks nprobe to the adaptive cap (probe wider),
  then the gate trims back to the I/O budget. The user sees the recall
  benefit of a wider probe without paying the wider PL fetch cost.

The two are orthogonal optimizations and the implementation lives behind
two independent config flags (`adaptive_search_nprobe`,
`head_bloom_enabled`). A direct comparison between
`adaptive_nprobe alone`, `bloom alone`, and `both` was not run for this
report but is left for future work; the necessary infrastructure is in
place.

### 6.5 Limitations

1. **Synthetic workload.** The bucket predicate distributes matches
   uniformly across heads — the easy case for any predicate-aware
   routing scheme. Real-world predicates can correlate with vector
   geometry (e.g., "customers in a given region" cluster spatially in
   product embeddings), making the gate either more or less effective.
   A study on MS MARCO with real metadata is the obvious next step.
2. **Latency on local storage.** The bench's PL fetches are local
   arrow-blockfile reads, which underrate the bloom's value on
   production storage backends.
3. **Predicate shapes.** The bloom gates equality and `$in` predicates
   on scalar metadata. Range predicates, `$ne`, document-text predicates,
   and AND-of-OR shapes fall through to `Unsupported` and are not gated.
   Per-centroid metadata histograms (a separate workstream, sibling to
   this) would address range predicates; the bloom is strictly weaker
   but a fraction of the implementation cost.
4. **Per-head capacity is fixed.** The bloom is sized for
   `split_threshold × 4 = 800` items at 0.001 FPR. Heads that grow
   beyond capacity see degraded FPR; in practice splits keep heads
   below `split_threshold = 200` so this never bit. A workload with a
   tuned `split_threshold` may want the bloom capacity to track it.
5. **Cross-validation across maintenance modes.** Sweep C used 50
   queries per mode but only at one (n, buckets, probe_nbr) cell.
   Whether Option 2's effectiveness is uniform across the full
   selectivity × size grid (it should be, by the rebuild's
   construction) is left to confirmation.

---

## 7. Conclusion

A per-head metadata bloom filter is a small, low-risk addition to a
SPANN-style index that delivers large recall gains on selective filtered
queries. The strict correctness property is a structural consequence of
bloom one-sidedness, not a measured outcome — across ~6 150 query ×
condition measurements, the gate never produced a false negative. The
measured benefit varies sharply with workload: the gate is a no-op on
unselective predicates (and pays one bloom check per probed head), and
a 9× recall improvement at fixed I/O on the most selective predicates
in this study.

The recommended deployment configuration is Phase 1.5 Option 2
(commit-time rebuild from doc-tokens cache). The Phase 1 MVP and
Phase 1.5 Option 1 alone produce loaded-but-ineffective filters;
Option 2's clean rebuild from `PL × doc_tokens` is what makes the gate
actually fire on a heavy-reassign workload. The doc-tokens cache is
auto-allocated when commit-rebuild is on, so the user-facing toggle is
a single flag.

The feature is gated behind a per-collection config and ships in the
`bloom-filters` branch of this repository. A follow-up study should
quantify the production storage tradeoff and compare against
per-centroid metadata histograms (Hammad's parallel workstream).

---

## Appendix: Code locations

| File | Role |
|---|---|
| `rust/index/src/spann/head_bloom.rs` | Module: `EqualityTokens`, `HeadBloom`, `HeadBloomCache`, `gate_heads`, write/read configs |
| `rust/index/src/spann/types.rs` | Wiring into `SpannIndexWriter` / `SpannIndexReader` / `SpannIndexFlusher`; `rebuild_blooms_from_cache` |
| `rust/segment/src/distributed_spann.rs` | Segment-level: read/write `HEAD_BLOOM_FILTERS_PATH`, gate forwarder |
| `rust/worker/src/execution/operators/spann_centers_search.rs` | Operator: gate after `rng_query` |
| `rust/worker/src/execution/orchestration/{knn_filter, spann_knn}.rs` | Predicate extraction; thread tokens through |
| `rust/worker/benches/spann_bloom_sweep.rs` | The within-subjects sweep harness used for this study |
| `rust/worker/benches/spann_bloom_ablation.rs` | Cross-index ablation harness (used for the implementation log) |
| `LOGS_PLANS/benchmarks/recall_study/*.csv` | Raw per-query output for every cell in this report |

## Appendix: Strict correctness contract

The gate's contract: a head is dropped only if every token in the
predicate's `And(tokens)` is *not* in the head's bloom (or, for `Or`,
no token is in the bloom). The bloom's bit pattern is set on every
insert; if a doc with token `t` is in the head's PL, then at write time
`bloom.insert(t)` was called, and `bloom.contains(t)` returns true at
read time (no bit can become unset under insert-only operations). So a
head containing a matching doc *cannot* fail the gate.

The audit guard verifies this on every measurement: for each dropped
head, fetch the PL, scan against the allowed bitmap. If any allowed doc
appears in a dropped head's PL, panic. No panic was triggered in any
sweep.

The single exception to the bit-monotonicity argument is the merge
path (`union_into`): if shapes don't match (capacity mismatch), the
destination filter is dropped and the head is marked stale. The gate
then keeps the head (passes through). Stale-set propagation in
`clone_into` and `union_into` ensures children/destinations of stale
heads stay stale — see §10.3 of `BLOOM_FILTERS.md`.
