# Head Synopsis — Implementation Guide

A per-cluster table of metadata value counts, used as an **exact gate**
on SPANN cluster routing for equality predicates. Companion to
`BLOOM_FILTERS.md`; the two are complementary, not redundant.

This is meant to be a self-contained design + replication guide. An
engineer who has never seen this codebase should be able to read this
document end-to-end and produce a working implementation.

---

## Table of contents

1. [Goal & framing](#1-goal--framing)
2. [Why exact counts beat a bloom for the equality gate](#2-why-exact-counts-beat-a-bloom-for-the-equality-gate)
3. [Architectural placement](#3-architectural-placement)
4. [Data structures](#4-data-structures)
5. [Storage layout](#5-storage-layout)
6. [Configuration](#6-configuration)
7. [Build path — compaction-time join (recommended)](#7-build-path--compaction-time-join-recommended)
8. [Build path — incremental during writes (alternative, *not* recommended)](#8-build-path--incremental-during-writes-alternative-not-recommended)
9. [Query path — the gate](#9-query-path--the-gate)
10. [Cardinality control](#10-cardinality-control)
11. [Edge cases & limitations](#11-edge-cases--limitations)
12. [Comparison with BloomHeads](#12-comparison-with-bloomheads)
13. [Tests](#13-tests)
14. [Benchmarking & recall validation](#14-benchmarking--recall-validation)
15. [Replication checklist](#15-replication-checklist)
16. [Operational guarantees](#16-operational-guarantees)

---

## 1. Goal & framing

For each SPANN cluster (head) `h`, maintain a table

```
synopsis[h] = {
    head_size: u64,
    counts: { key: { value: count } }
}
```

where `count` is the **exact** number of docs currently in cluster `h`
that have metadata `key = value`. At query time, given a predicate
like `bucket = 0`, look up `synopsis[h]["bucket"][0]`. If it's `0`,
cluster `h` has zero matching docs — **skip it**. If it's positive,
the cluster has *exactly that many* matching candidates and is worth
fetching.

Two distinct uses, both available from the same data:

- **Hard gate** (the primary use): drop clusters with `count == 0`.
  No false positives; cluster is *provably* empty for the predicate.
- **Yield estimation** (secondary, optional): rank kept clusters by
  predicted yield `head_size × selectivity` for fetch ordering or
  early-termination policies.

This document focuses on the hard gate. Yield-based ranking is a
straightforward extension once the data is in place.

---

## 2. Why exact counts beat a bloom for the equality gate

A bloom filter answers "*may* this cluster contain key=value?" with
some false-positive rate. The synopsis answers "*does* this cluster
contain key=value, and how many?" exactly.

For the equality gate:

| | Bloom filter | Synopsis count |
|---|---|---|
| Definitely-no answer | yes | yes |
| False positives | yes (target FPR ≈ 0.001) | no |
| False negatives | only if filter is incomplete (Phase 1.5 problem) | never |
| Cluster ranking | no (binary) | yes (count gives selectivity) |
| Storage per cluster | constant (~12 KB at FPR 0.001) | proportional to distinct (key,value) pairs |
| Scales to high-cardinality keys | yes | no — needs top-K cap |

Synopsis dominates for low- and medium-cardinality keys: tags,
categories, booleans, enums, popular discrete IDs. Bloom dominates
for high-cardinality keys: free-form strings, UUIDs, free text. A
production system can run both, picking per key based on
cardinality.

For the workload Phase 1's bloom was tuned for (`bucket = id % 100`,
100 distinct values), the synopsis is strictly better — it adds
~100 entries per cluster, gives exact answers, and enables ranking
"for free."

---

## 3. Architectural placement

Same as the bloom blob: a side-file inside the SPANN vector segment.

Three segments per shard:
- **Record** — canonical doc store
- **Metadata** — inverted index on metadata, source of truth for
  filtering
- **Vector** — SPANN (HNSW + posting lists). The synopsis lives
  here.

The synopsis does not change correctness. The metadata segment's
inverted index is still the source of truth for whether a doc
matches the predicate. The synopsis only changes which clusters get
fetched. Dropping a cluster with `count == 0` is safe because the
metadata bitmap (intersected against that cluster's doc IDs) would
have been empty anyway — the synopsis just lets us skip the fetch.

Query flow with synopsis enabled:

```
KnnFilterOrchestrator
    │
    ├─▶ FilterOperator (reads metadata segment)
    │       → RoaringBitmap of allowed doc_offset_ids   ← ground truth
    │
    └─▶ extract_equality_tokens(where_clause)
            → SynopsisPredicate

SpannKnnOrchestrator
    │
    └─▶ SpannCentersSearchOperator
            ├─ HNSW rng_query → candidate head_ids
            ├─ (optional) BloomHeads gate
            ├─ Synopsis gate ── drop where count == 0  ← hard gate
            └─ (optional) Synopsis rank ── reorder kept heads by yield
```

If both bloom and synopsis are enabled, run synopsis after bloom.
The synopsis is exact, so it strictly subsumes whatever the bloom
kept; running bloom first cuts the synopsis's input set without
costing recall.

---

## 4. Data structures

### Token extraction (shared with bloom)

Synopsis predicates use the same canonical typed token format as
bloom, except split into `(key, value_token)` pairs instead of flat
strings:

```rust
pub type SynopsisToken = (String, String);

pub fn synopsis_token(key: &str, value: &MetadataValue) -> Option<SynopsisToken> {
    let v = match value {
        MetadataValue::Bool(b)  => format!("bool::{}",  b),
        MetadataValue::Int(i)   => format!("int::{}",   i),
        MetadataValue::Float(f) => format!("float::{}", f.to_bits()),
        MetadataValue::Str(s)   => format!("str::{}",   s),
        MetadataValue::SparseVector(_) | MetadataValue::*Array(_) => return None,
    };
    Some((key.to_string(), v))
}
```

Float uses bit-pattern so NaN/-0.0 are deterministic. Type tags
prevent `Int(1)` / `Bool(true)` / `Str("1")` from colliding.

### `HeadSynopsis` — one cluster's table

```rust
#[derive(Clone, Debug, Default)]
pub struct HeadSynopsis {
    pub head_size: u64,
    /// key → value_token → count
    pub counts: HashMap<String, HashMap<String, u64>>,
}
```

`head_size` is the number of live (non-deleted, current-version)
docs in the cluster. `counts[k][v]` is the number of those docs that
have `metadata[k] == v`.

### `SynopsisPredicate` — extracted from a `Where` clause

```rust
pub enum SynopsisPredicate {
    /// Every token must have count > 0 in the cluster.
    And(Vec<SynopsisToken>),
    /// At least one token must have count > 0.
    Or(Vec<SynopsisToken>),
    /// Predicate uses operators we can't express via exact equality
    /// counts (range, $ne, $contains on arrays, document text, …).
    /// Gate degrades to "always keep" — no information.
    Unsupported,
}
```

Extraction parser (`extract_synopsis_predicate(where: &Where) -> SynopsisPredicate`)
walks the `Where` tree:

- `Metadata(Primitive(Equal, v))` with a tokenizable value →
  `And([token])`
- `Metadata(Set(In, [...]))` → `Or([token, ...])`
- `Composite(And, children)`: each child must collapse to `And(_)`.
  Concatenate tokens. Otherwise `Unsupported`.
- `Composite(Or, children)`: each child must be `And` of length 1 or
  `Or`. Concatenate. Otherwise `Unsupported`.
- Anything else → `Unsupported`.

Identical shape to bloom's `EqualityTokens`; the two predicates can
share extraction logic (one walk, two outputs).

### `HeadSynopsisCache` — segment-level table

```rust
pub struct HeadSynopsisCache {
    /// head_id → synopsis. Loaded from blob on segment open.
    synopses: Arc<DashMap<u32, Arc<HeadSynopsis>>>,
}
```

Read-only after load (under the recommended build-at-compaction
strategy). The writer rebuilds the entire cache during compaction
and atomically replaces it on flush.

### `HeadSynopsisBlob` — serialized form

```rust
#[derive(Serialize, Deserialize)]
pub struct HeadSynopsisBlob {
    pub synopses: HashMap<u32, HeadSynopsis>,
}
```

One bincode-serialized blob per segment shard, parallel to the bloom
blob.

### Public API (free functions)

```rust
/// Apply the gate. Returns `candidate_head_ids` filtered to those
/// whose synopsis does not provably-rule-out the predicate. Heads
/// with no synopsis entry are kept (conservative — synopsis hasn't
/// been built or this head is from after the last compaction).
pub fn gate_heads<I, F>(
    candidate_head_ids: I,
    predicate: &SynopsisPredicate,
    lookup: F,
) -> Vec<u32>
where
    I: IntoIterator<Item = u32>,
    F: Fn(u32) -> Option<Arc<HeadSynopsis>>;

/// Compute predicted yield for a cluster — head_size × selectivity.
/// Used for ranking; safe to use for early-termination policies.
pub fn predicted_yield(synopsis: &HeadSynopsis, predicate: &SynopsisPredicate) -> f64;
```

Gate semantics:
- `Unsupported`: keep all (no information).
- `And(empty)`: keep all (vacuous truth).
- `And(toks)`: drop head if **any** token has `count == 0`.
- `Or(empty)`: keep all (vacuous; never constructed in practice).
- `Or(toks)`: drop head if **all** tokens have `count == 0`.

`predicted_yield` formula:
- `And(toks)`: independence approximation across keys —
  `head_size × Π (count[k][v] / head_size)`. Same-key conjunctions
  with conflicting values short-circuit to `0`.
- `Or(toks)`: cap-sum — `min(head_size, Σ count[k][v])`. Exact when
  tokens share a key (a doc has at most one value per key).
- `Unsupported`: returns `head_size` (neutral — preserves input
  order under sort-by-yield).

---

## 5. Storage layout

Path constant in `chroma_types::segment`:

```rust
pub const HEAD_SYNOPSIS_PATH: &str = "head_synopsis_path";
```

Blob format: one bincode object per shard at
`<segment.prefix_path>/<fresh_uuid>`. Immutable; never overwritten —
each compaction writes a new blob and updates the segment's
`file_path`.

Persistence lifecycle (writer side):

```rust
pub struct HeadSynopsisBlobFlusher {
    pub bytes: Vec<u8>,
    pub path: String,
    pub blob_id: Uuid,
    pub storage: Option<Arc<Storage>>,
}

impl HeadSynopsisBlobFlusher {
    pub async fn save(&self) -> Result<(), …> {
        let Some(storage) = &self.storage else { return Ok(()); };
        storage.put_bytes(&self.path, self.bytes.clone(), PutOptions::default()).await?;
        Ok(())
    }
}
```

`SpannIndexFlusher` adds:
- `head_synopsis_blob: Option<HeadSynopsisBlobFlusher>`

`SpannIndexIds` adds:
- `head_synopsis_blob_path: Option<String>`

`SpannIndexFlusher::flush` saves the blob after the existing PL /
versions / HNSW flushes.

Persistence lifecycle (reader side): mirror the bloom blob —
`SpannIndexReader::from_id` accepts an `Option<HeadSynopsisReadConfig<'_>>`,
loads the blob via `Storage::get`, deserializes into a
`HeadSynopsisCache`, stores `Option<Arc<HeadSynopsisCache>>` on the
reader. Missing/unreadable blob → `None` → gate is a no-op
(correctness-safe).

Segment-level wiring in `SpannSegmentWriterShard::from_segment` and
`SpannSegmentReaderShard::from_segment`: read existing blob path
from `segment.file_path[HEAD_SYNOPSIS_PATH]`; pass into the
respective config; on flush, record the new blob path back into
`file_path`.

---

## 6. Configuration

```rust
pub struct SpannProviderConfig {
    // existing...
    #[serde(default = "default_head_synopsis_enabled")]
    pub head_synopsis_enabled: bool,

    /// Top-K most frequent values to track per key. Values beyond
    /// the top K go into an "other" bucket. Default 64.
    #[serde(default = "default_synopsis_top_k_per_key")]
    pub head_synopsis_top_k_per_key: u32,

    /// Skip indexing keys whose distinct-value count exceeds this
    /// threshold (in the inverted index). Default 1024 — beyond
    /// this, even with top-K capping, the synopsis adds little
    /// signal. Such keys should rely on the bloom filter instead.
    #[serde(default = "default_synopsis_max_cardinality")]
    pub head_synopsis_max_cardinality: u32,
}
```

Defaults: `false`, `64`, `1024`. The flag is master; the two
cardinality knobs only matter when `enabled = true`.

Threading: `SpannProviderConfig` → `SpannProvider` (three new
fields, populated in `try_from_config`) → `SpannSegmentWriterShard::from_segment`
→ `SpannIndexWriter::from_id` (via a new `head_synopsis_config:
Option<HeadSynopsisWriteConfig<'_>>` parameter).

---

## 7. Build path — compaction-time join (recommended)

This is the recommended architecture. Detailed reasoning in §8 for
why incremental builds are *not* recommended.

### When it runs

At the end of `SpannSegmentWriterShard::commit`, after all log
records have been applied to SPANN's PL and the metadata segment's
inverted index, but **before** the index flusher writes its blobs.

### Inputs

1. SPANN's posting-list state: `head_id → Vec<(doc_id, version)>`.
   Already in memory in the writer; or read back from the
   `posting_list_writer` if needed.
2. Versions map: `doc_id → version`. Used to skip stale entries.
3. Metadata segment's inverted indices: per-type maps of
   `key → value → RoaringBitmap<doc_id>`. Already in memory in the
   `MetadataSegmentWriterShard`.
4. Cardinality config: `head_synopsis_top_k_per_key`,
   `head_synopsis_max_cardinality`.

### Algorithm

```rust
async fn build_synopsis_from_inverted_index(
    spann_writer: &SpannIndexWriter,
    metadata_writer: &MetadataSegmentWriterShard<'_>,
    config: &HeadSynopsisWriteConfig,
) -> HashMap<u32, HeadSynopsis> {
    // Step 1 — invert SPANN's PLs to a doc_id → head_ids map.
    //          One doc can be in multiple heads (write_nprobe replication).
    let mut doc_to_heads: HashMap<u32, SmallVec<[u32; 4]>> = HashMap::new();
    let mut head_size: HashMap<u32, u64> = HashMap::new();
    let versions = spann_writer.versions_map.read().await;
    for (head_id, posting_list) in spann_writer.iter_live_posting_lists().await {
        for (doc_id, version) in posting_list {
            let cur = versions.versions_map.get(&doc_id).copied().unwrap_or(0);
            if cur == 0 || version < cur { continue; } // skip stale / deleted
            doc_to_heads.entry(doc_id).or_default().push(head_id);
            *head_size.entry(head_id).or_default() += 1;
        }
    }

    // Step 2 — for each (key, value) in the inverted index, add to the
    //          counts of every head containing each member doc_id.
    let mut counts: HashMap<u32, HashMap<String, HashMap<String, u64>>> = HashMap::new();
    for (key, distinct_values_count) in metadata_writer.iter_keys_with_cardinality() {
        if distinct_values_count > config.max_cardinality {
            continue; // too high-card; rely on bloom for this key
        }
        // Get the top-K values by global popularity. Easy heuristic:
        // bitmap.cardinality() per (key,value).
        let top_values = metadata_writer
            .top_k_values_for_key(&key, config.top_k_per_key);

        for (value_token, doc_bitmap) in top_values {
            for doc_id in doc_bitmap {
                if let Some(heads) = doc_to_heads.get(&doc_id) {
                    for &head_id in heads {
                        *counts.entry(head_id)
                               .or_default()
                               .entry(key.clone())
                               .or_default()
                               .entry(value_token.clone())
                               .or_default() += 1;
                    }
                }
            }
        }
    }

    // Step 3 — package into HeadSynopsis structs.
    head_size.into_iter().map(|(head_id, size)| {
        let counts = counts.remove(&head_id).unwrap_or_default();
        (head_id, HeadSynopsis { head_size: size, counts })
    }).collect()
}
```

Two helper methods on `MetadataSegmentWriterShard` are needed:

1. `iter_keys_with_cardinality() -> impl Iterator<Item = (String, u32)>` —
   walk the four type-keyed indices (`String`, `U32`, `F32`, `Bool`)
   and yield `(key, distinct_values_count)`.
2. `top_k_values_for_key(key: &str, k: u32) -> Vec<(String, RoaringBitmap)>` —
   for a given key, return the top-K most frequent values (by
   bitmap cardinality) along with their bitmaps. Float and U32
   values are bit-encoded into the value-token format above.

### Cost

Time: `O(total_docs × avg_replicas + Σ_keys top_k × Σ_values bitmap_size)`.
On a 9 000-doc segment with `write_nprobe=32` and 100 buckets, this
is `~290 K + 100 × 90 = ~300 K` increment operations. Sub-second.

Memory: peak during build is `O(num_docs × avg_replicas)` for the
`doc_to_heads` map, plus `O(num_heads × num_keys × top_k)` for the
output counts. For a 1 M-doc segment with ~10 keys × top-K=64 ×
~1 000 heads, that's ~640 K count entries — a few hundred MB at
worst.

If memory is a concern, build streamingly: emit `(head_id, key,
value, doc_id)` tuples sorted by `head_id`, then aggregate per
head_id into a final synopsis on the fly. But for now, keep it
simple and in-memory.

### Why this is clean

- No write-path mutation. The synopsis is built once, after
  everything else is settled. No reassign-during-build problems.
- The metadata segment's inverted index is already the canonical
  source — we don't redundantly re-extract metadata from records.
- Splits/merges/reassigns that happened during compaction are
  invisible at this stage: they're already reflected in the final
  posting lists.
- Failure mode: if synopsis build fails, log and skip persisting
  the blob. The reader sees no blob, gate is a no-op, queries still
  return correct (just slower) results.

### Cross-segment dependency

The SPANN writer must be able to call into the metadata writer at
commit time. There are two ways to wire this:

1. **Shared compactor handle**: `SpannSegmentWriterShard::commit`
   accepts a `&MetadataSegmentWriterShard` argument. The compactor,
   which owns both, hands the metadata writer in. This is the
   simplest plumbing.
2. **Two-phase commit**: SPANN writer signals "ready" without
   building synopsis; an outer step reads from both committed
   segments to build the synopsis post-hoc, then attaches it to the
   SPANN segment's `file_path`. More complex but decouples the
   writers.

Option 1 is **implemented**. See `commit_with_metadata_snapshot` on
`SpannSegmentWriterShard`, `snapshot_inverted_index_for_synopsis` on
`MetadataSegmentWriterShard`, and `set_synopsis_inverted_index` on
`SpannIndexWriter`. The compactor calls
`spann_writer.commit_with_metadata_snapshot(&metadata_shard)` which:

1. Snapshots the metadata segment's typed inverted indexes
   (`String`/`U32`/`Bool` — float keys are intentionally skipped per
   §11) into an `InvertedIndexSnapshot { by_key: key → value_token →
   bitmap<doc_id> }`.
2. Hands that snapshot to the SPANN writer via
   `set_synopsis_inverted_index`.
3. Calls the regular `commit()`, which detects the snapshot is set
   and dispatches to `build_synopsis_via_inverted_index` instead of
   the doc-tokens cache fallback in §8.

The snapshot path is correct under segment reload: every live doc
appears in the metadata inverted index regardless of which
compaction wrote it. The §8 fallback only sees docs the SPANN writer
touched this commit cycle, so heads with any "untouched" live doc
are skipped (gate falls back to keep — safe but lossy). Empirical
parity validated in `LOGS_PLANS/synopsis-recall-study.md`: both
build paths produce equivalent drop_ratio (~0.69) and iso-I/O
Δrecall (+0.35–0.43) with `bad_drops = 0`.

---

## 8. Build path — incremental during writes (alternative, *not* recommended)

For completeness; not the recommended path.

### What it would look like

Maintain `synopsis: DashMap<u32, RwLock<HeadSynopsis>>` in the SPANN
writer. On every `add_with_metadata_tokens(doc_id, …, tokens)`:
- For each destination `head_id`: `synopsis[head_id].head_size += 1`
  and `counts[k][v] += 1` for each token.

On `update`: subtract old tokens, add new. Requires the writer to
remember each doc's previous tokens — a `doc_tokens: DashMap<doc_id,
Vec<(k,v)>>` cache.

On `delete`: subtract tokens at every head where the doc lived.
Requires a `doc_heads: DashMap<doc_id, HashSet<head_id>>` cache.

On reassign-induced append: doc moves from `prev_head` to `new_head`
without tokens at the call site. Use the `doc_tokens` cache to
look them up; decrement at `prev_head`, increment at `new_head`.

On split: parent splits into two children. Each doc in the parent's
PL is reassigned to one child. Update doc_heads, update synopses
for both children + parent.

On merge: target absorbs source. Sum source's counts into target;
drop source's synopsis.

### Why this is *not* recommended

This is structurally identical to the Phase 1.5 `doc_tokens` cache
that the bloom filter tried (and which still has a known
false-negative leak — see `BLOOM_FILTERS.md` §10.5). The reassign
path through the SPANN writer has subtle code paths where a doc
ends up in a head's PL without going through the per-head synopsis
update (the same paths that bit the bloom). Debugging that without
a commit-time consistency check is hard.

If you must build incrementally, do it with the same structure as
the Phase 1.5 `head_bloom_commit_rebuild` flag: keep a per-write
cache that updates eagerly and a rebuild step at commit that
verifies / replaces it. But at that point you've added complexity
for no gain over just rebuilding from the inverted index in the
first place.

The compaction-time join wins on simplicity, correctness, and
fits Chroma's existing eventually-consistent staleness model.

---

## 9. Query path — the gate

### Operator integration

`SpannCentersSearchInput` gains:

```rust
pub(crate) head_synopsis_predicate: SynopsisPredicate,
```

`SpannCentersSearchOutput` gains:

```rust
pub(crate) heads_after_synopsis: usize,  // post-gate telemetry
```

Inside `SpannCentersSearchOperator::run`, after the existing bloom
gate (if enabled) and before fetching posting lists:

```rust
let center_ids = reader.gate_heads_synopsis(
    &center_ids,
    &input.head_synopsis_predicate,
);
let heads_after_synopsis = center_ids.len();
```

`SpannSegmentReaderShard::gate_heads_synopsis` is a thin forwarder to
`SpannIndexReader::gate_heads_synopsis`, which calls the
free-function `head_synopsis::gate_heads(...)` with a closure that
looks up `Arc<HeadSynopsis>` from the reader's loaded cache.

### Orchestrator wiring

`KnnFilterOutput` gains:

```rust
pub head_synopsis_predicate: SynopsisPredicate,
```

Populated once in `KnnFilterOrchestrator::on_filter_complete`:

```rust
let head_synopsis_predicate = match self.filter.where_clause.as_ref() {
    Some(w) => extract_synopsis_predicate(w),
    None    => SynopsisPredicate::Unsupported,
};
```

`SpannKnnOrchestrator::initial_tasks` passes
`self.knn_filter_output.head_synopsis_predicate.clone()` into
`SpannCentersSearchInput`.

### Telemetry

In `SpannKnnOrchestrator`'s centers-search-output handler:

```rust
tracing::debug!(
    heads_rng = output.heads_rng,
    heads_after_bloom = output.heads_after_bloom,
    heads_after_synopsis = output.heads_after_synopsis,
    "spann centers search completed",
);
```

For estimation-error metrics (yield prediction vs observed valid
counts), the orchestrator can stash predicted yields per head_id
alongside the kept set and compare to the actual filter-bitmap
intersections after the BfPL stage. Optional; not required for the
gate to function.

### Composition with the bloom gate

If both gates are enabled, run **bloom first, synopsis second**:

```
candidates → bloom_gate → synopsis_gate → fetch PLs → BfPL
```

The bloom can over-include (false positives); the synopsis is
exact, so it strictly tightens whatever the bloom kept. Running the
bloom first costs zero recall and may save a few synopsis lookups.

If only one is enabled, run that one. If neither, the operator
passes through `center_ids` unchanged.

---

## 10. Cardinality control

Two knobs, applied **per-key with auto-promotion**: the regime is
chosen based on the key's global distinct-value count, not on a
single global cap.

### Regimes

For each key independently:

- **Low/medium cardinality** (`distinct_values ≤ max_cardinality`):
  track all values exactly. The gate is exact for every queried
  value of this key — no `other_counts` pollution. This is the
  common case for typed metadata: booleans, enums, low/medium-card
  categoricals (tags, languages, status codes, popular IDs).
- **High cardinality** (`distinct_values > max_cardinality`):
  compress to top-K + other. Storage-bounded; gate is exact for
  queried values that happen to be in top-K, falls back to "keep"
  otherwise. UUIDs and free-form text land here; they are better
  served by the bloom filter, but the synopsis still gates cleanly
  for top-K hits.

The earlier policy ("always cap at top_k_per_key, even if the key
has only slightly more distinct values than top_k") caused
near-universal "other-bucket pollution": with default `top_k=64`
and a workload of, say, 100 distinct values, ~99% of heads ended
up with `other_counts[key] > 0`, forcing the gate to return
"unknown" for every query and silently degrading to a no-op.
Auto-promotion eliminates that trap by giving low/medium-card
keys the exact-tracking regime regardless of `top_k_per_key`.

### `head_synopsis_max_cardinality`

The threshold that selects the regime. Default 1024.

- Keys with `distinct_values ≤ max_cardinality` are tracked exactly.
- Keys with `distinct_values > max_cardinality` are compressed via
  top-K + other.

### `head_synopsis_top_k_per_key`

Default 64. Only consulted in the high-cardinality regime; ignored
when the key fits within `max_cardinality`. For genuinely
high-cardinality keys, this controls the size/precision tradeoff:
top-K largest exact entries per head, rest collapsed into one
`other_counts` integer per head.

### Gate logic for `key = v`

```
if counts[key].contains(v):
    return drop iff counts[key][v] == 0
elif other_counts[key] == 0 (or absent):
    return drop  (provably zero — exact regime, or v isn't a known value)
else:
    return keep  (unknown — high-card mode and v not in top-K)
```

### Data structure

```rust
pub struct HeadSynopsis {
    pub head_size: u64,
    pub counts: HashMap<String, HashMap<String, u64>>,
    /// Per-key count of docs with a value not in `counts[key]`. Always
    /// 0 (or absent) under the exact-tracking regime; only positive
    /// under the high-card top-K + other regime.
    pub other_counts: HashMap<String, u64>,
}
```

### Memory budget

Worst-case per head, per key:
- Exact regime: `O(distinct_values)` entries — bounded by
  `max_cardinality` (default ≤1024).
- Compressed regime: `O(top_k_per_key)` entries + 1 integer.

Across a 1000-head, 10-keyed segment with default config and typical
metadata cardinalities (≤100 values per key), storage runs ~5–20 MB
per segment shard. Well within SPANN segment budgets.

For a fixed-budget alternative, use Misra-Gries / Space-Saving
streaming top-K instead of exact top-K (which requires sorting full
distributions). Skipped here for MVP simplicity.

---

## 11. Edge cases & limitations

### What the gate can express

- `key = value` (exact equality)
- `key IN [v1, v2, ...]` (any-of)
- `AND` of equality / IN
- `OR` of equality / IN at the top level (no nested AND-of-OR)

### What it can't express (synopsis falls back to "keep")

- Range (`>`, `<`, `>=`, `<=`)
- `key != value` (negation; could be derived as `head_size -
  counts[key][value] > 0`, but introduces a footgun if the gate is
  wrong; skip for MVP)
- `NOT IN`
- `$contains` on arrays / document text
- Nested boolean structures the parser can't flatten

For all of these, `extract_synopsis_predicate` returns
`Unsupported` and the gate is a no-op. Correct, just unused.

### Float values

Float keys are tokenized via `to_bits()`. Equality comparison on
floats is exact bit-pattern equality, so `0.1 + 0.2 = 0.3` does
*not* match because the bit patterns differ. Document this; or, if
the application semantics require float-fuzzy matching, exclude
float keys via `head_synopsis_max_cardinality` or a per-key opt-out.

### Sparse vectors and arrays

Skipped at token-extraction time. The synopsis has no entry for
these keys; gate falls back to "keep."

### Logs not yet compacted

The synopsis covers the last-compacted segment state. In-flight log
records are filtered by the orchestrator's `FilterOperator` against
the metadata segment + log records, separately from the synopsis
gate. There is no consistency issue — the gate is layered on top
of compacted segments only.

### Empty cluster after rebuild

If a cluster's `head_size` drops to zero between compactions (all
its docs deleted or reassigned), the rebuild produces no entry for
that cluster. Gate treats missing entries as "keep" — this cluster
will be probed and the BfPL stage will return zero matches. Not
a correctness problem, just a missed optimization until next
compaction.

### Schema evolution

Keys can be added between compactions. New keys' synopsis entries
appear after the next rebuild. Until then, predicates on those
keys fall back to "keep." Same staleness model as the metadata
segment's inverted index itself.

---

## 12. Comparison with BloomHeads

| Property | BloomHeads | HeadSynopsis |
|---|---|---|
| Cluster gate | yes (probabilistic) | yes (exact) |
| False positives | yes (~0.1% target FPR) | none |
| False negatives | no, but Phase 1.5 has a real-world leak | no |
| Per-cluster size | ~12 KB (fixed) | proportional to (keys × top-K) |
| Scales to high-cardinality keys | yes | no — needs top-K cap or skip |
| Update pattern | incremental during writes | rebuilt at compaction |
| Reassign correctness | requires per-doc cache + rebuild | trivial (rebuild after) |
| Cluster ranking | no | yes (counts give selectivity) |
| Storage cost (1 M docs, 10 keys, top-K 64, 1 K heads) | ~12 MB | ~20 MB |

**Recommended deployment**:

- Synopsis on for low- and medium-cardinality keys (≤ ~1024 distinct
  values).
- Bloom on for high-cardinality keys.
- Both running together, synopsis after bloom in the gate pipeline.

Since the synopsis is exact, it *supersedes* the bloom for keys it
covers. The bloom's false positives are filtered out by the
synopsis's exact zero-counts. The bloom only contributes for keys
the synopsis skipped (high-card).

---

## 13. Tests

### Unit tests (in `head_synopsis.rs::tests`)

Predicate parsing:
- `synopsis_token_format_is_typed`
- `arrays_and_sparse_yield_no_token`
- `extract_single_equality`
- `extract_in_yields_or`
- `extract_and_of_equalities`
- `nested_or_inside_and_is_unsupported`
- `ranges_and_ne_are_unsupported`

Data structure:
- `head_synopsis_count_increment_decrement`
- `head_synopsis_other_bucket_for_non_top_k`

Gate semantics:
- `gate_drops_zero_count_heads_for_and`
- `gate_keeps_heads_with_at_least_one_match_for_or`
- `gate_falls_back_to_keep_on_unsupported`
- `gate_falls_back_to_keep_when_value_in_other_bucket`
- `gate_keeps_heads_missing_synopsis_entry`

Yield estimation:
- `yield_monotonic_with_count`
- `yield_for_in_predicate_sums_within_key`

Round-trip:
- `bincode_serialize_deserialize_preserves_counts`

### Integration tests (in `types.rs::tests`)

- `test_head_synopsis_compaction_build_from_inverted_index`:
  build a tiny segment with known metadata; run the
  compaction-time build; verify synopsis matches the expected
  per-cluster counts.
- `test_head_synopsis_disabled_does_not_track`: with
  `head_synopsis_config = None`, after writes the synopsis blob is
  not produced; reader sees no blob, gate is no-op.
- `test_head_synopsis_top_k_capping`: a key with 100 distinct
  values, top-K = 10. Verify only the top 10 are tracked
  individually; the rest go to `other_counts`. Verify the gate
  correctly distinguishes top-K hits, top-K misses (definite
  zero), and non-top-K queries (fall back to keep).

### Property tests (optional)

Random workload generator: add N docs with random metadata, run a
random sequence of queries, assert that for every query the gate's
keep-set is a superset of the BfPL's actual-match-set
(no false negatives).

---

## 14. Benchmarking & recall validation

Same methodology as `BLOOM_FILTERS.md` §12. Two cells required:

### Iso-probe cell (correctness)

```
========== HeadSynopsis 000 vs 001 ablation (iso-probe) ==========
  selectivity=1%  buckets=100  k=10  probe_nbr=32
  000 (no synopsis)  recall_mean=…  heads_fetched=32  drop_ratio=0.0000
  001 (synopsis)     recall_mean=…  heads_fetched=…   drop_ratio=…       Δrecall=…
  [gate-audit] dropped N heads; M would have contained matching docs
====================================================================
```

Required: `M = 0` (synopsis is exact, so this should always be 0;
any positive value is a serious bug); `|Δrecall| < 0.02`;
`drop_ratio > 0`.

### Iso-I/O cell (value claim)

```
========== HeadSynopsis 000 vs 001 ablation (iso-I/O, B=32) ==========
  selectivity=1%  buckets=100  k=10  io_budget=32
  000 (no synopsis)  probe_nbr=32   heads_fetched=32  recall_mean=R0
  001 (synopsis)     probe_nbr=…    heads_fetched=≈32 recall_mean=R1   Δrecall=R1-R0
========================================================================
```

Required: `R1 > R0`. Recommended: `R1 > R0 + 0.02`.

If `R1 ≤ R0`, the gate has correct plumbing but no product win on
this workload. Don't enable.

### Joint bloom + synopsis cell (the 011 ablation)

```
========== Bloom+Synopsis 000 vs 011 ablation ==========
  …
  000                  recall=…   heads_fetched=…
  010 (bloom only)     recall=…   heads_fetched=…   drop_ratio=…
  001 (synopsis only)  recall=…   heads_fetched=…   drop_ratio=…
  011 (both)           recall=…   heads_fetched=…   drop_ratio=…
========================================================
```

Useful for the 8-cell ablation hypercube. The 011 cell should match
or beat 001 alone; if 011 is materially better, the bloom is doing
useful work on top of synopsis (likely covering high-cardinality
keys synopsis skipped).

### Strict correctness invariant

The bench's `[gate-audit]` block (same shape as the bloom audit)
counts dropped heads whose PL contained a matching doc and panics
on any positive count. Synopsis-induced false negatives must always
be 0 — counts are exact, so a positive value indicates either a
build bug (counts wrong) or a query bug (token mismatch between
build and gate).

---

## 15. Replication checklist

In dependency order, from a clean tree:

1. **Add path constant** `HEAD_SYNOPSIS_PATH` in `chroma_types::segment`.
2. **Add deps** to `chroma-index/Cargo.toml`: `bincode` (already
   present if bloom is in), `serde`.
3. **Write `head_synopsis.rs`** module with:
   - `synopsis_token`, `extract_synopsis_predicate`
   - `SynopsisPredicate` enum
   - `HeadSynopsis` struct + `HeadSynopsisBlob` + serde derives
   - `HeadSynopsisCache` with `get`, `iter`, `is_empty`
   - `HeadSynopsisWriteConfig` / `HeadSynopsisReadConfig`
   - `gate_heads` and `predicted_yield` free functions
4. **Add module declaration** to `spann.rs`.
5. **Write 12+ unit tests** covering token format, predicate
   parsing, gate semantics for AND/OR/Unsupported, top-K
   `other_counts` fallback, yield estimation, bincode round-trip.
6. **Add `head_synopsis_blob` field** to `SpannIndexFlusher`. Add
   `head_synopsis_blob_path` field to `SpannIndexIds`. Add
   `HeadSynopsisBlobSaveError` variant. Update
   `SpannIndexFlusher::flush` to call `blob.save().await?`.
7. **Add `HeadSynopsisBlobFlusher`** struct (parallel to
   `HeadBloomBlobFlusher` if bloom is in; otherwise as the canonical
   synopsis flusher). `save` method calls `Storage::put_bytes`.
8. **Plumb `head_synopsis_config` through `SpannIndexWriter::from_id`** —
   accept `Option<HeadSynopsisWriteConfig<'_>>`. Load existing
   blob if path provided.
9. **Add fields to `SpannIndexWriter`**: `head_synopsis_enabled`,
   `head_synopsis_top_k_per_key`, `head_synopsis_max_cardinality`,
   `head_synopsis_cache: Option<Arc<HeadSynopsisCache>>` (loaded
   from prior compaction's blob; used only as a starting point or
   for stats — the rebuild produces a fresh cache).
10. **Implement `build_synopsis_from_inverted_index`** as a free
    function or method on `SpannSegmentWriterShard` (the latter
    has access to both writers via the compactor). The body
    follows §7's algorithm exactly.
11. **Add helper methods to `MetadataSegmentWriterShard`**:
    `iter_keys_with_cardinality()` and
    `top_k_values_for_key(key, k)`. These walk the four type-keyed
    indices (`String`/`U32`/`F32`/`Bool`) and return values
    bit-encoded into the synopsis token format.
12. **Wire `SpannSegmentWriterShard::commit`** to call the build
    function before the index commit, when
    `head_synopsis_enabled = true`. Attach the resulting blob
    flusher to the index flusher.
13. **Plumb `head_synopsis_config` through `SpannIndexReader::from_id`** —
    accept `Option<HeadSynopsisReadConfig<'_>>`. Add
    `load_head_synopses` that bincode-deserializes the blob into a
    `HeadSynopsisCache`.
14. **Add `gate_heads_synopsis` method** on `SpannIndexReader`,
    forwarding to `head_synopsis::gate_heads`.
15. **Add config fields** to `SpannProviderConfig`:
    `head_synopsis_enabled`, `head_synopsis_top_k_per_key`,
    `head_synopsis_max_cardinality`. All default off / sensible
    values.
16. **Add fields to `SpannProvider`** + populate in
    `try_from_config`.
17. **Update `SpannSegmentWriterShard::from_segment`** signature to
    accept the three flags. Read existing blob path from
    `segment.file_path[HEAD_SYNOPSIS_PATH]`. Build
    `HeadSynopsisWriteConfig`. Pass to `SpannIndexWriter::from_id`.
18. **Update `SpannSegmentReaderShard::from_segment`**: same path
    lookup. Build `HeadSynopsisReadConfig`. Pass to
    `SpannIndexReader::from_id`. Add `gate_heads_synopsis`
    forwarder.
19. **Update segment writer's `flush`** to insert
    `HEAD_SYNOPSIS_PATH → [blob_path]` when
    `index_ids.head_synopsis_blob_path` is `Some`.
20. **Add `head_synopsis_predicate: SynopsisPredicate` to
    `KnnFilterOutput`**, populated in the orchestrator's
    filter-completion handler via `extract_synopsis_predicate`.
21. **Add `head_synopsis_predicate` to
    `SpannCentersSearchInput`** and `heads_after_synopsis: usize`
    to `SpannCentersSearchOutput`. In the operator, call
    `reader.gate_heads_synopsis(&center_ids, &input.head_synopsis_predicate)`
    after the bloom gate.
22. **`SpannKnnOrchestrator`** passes
    `self.knn_filter_output.head_synopsis_predicate.clone()` into
    the centers-search input. Add tracing for `heads_after_synopsis`.
23. **Update test code**: `SpannProvider` literals, `from_id`
    callers (add the new optional config or `None`).
24. **Add the iso-probe bench cell `bench_spann_synopsis_ablation`**
    (correctness check) with the env-var toggles
    (`SYNOPSIS_TOP_K`, `SYNOPSIS_MAX_CARD`), progress prints,
    `user_filter_matches` early-bail, and the `[gate-audit]`
    panic guard. Holds `probe_nbr` constant; reports `Δrecall`
    (must be ≈ 0) and `drop_ratio` (must be > 0).
25. **Add the iso-I/O bench cell** (value claim). Holds
    `heads_fetched` constant; reports `Δrecall` (must be > 0).
    Suggested cell name: `bench_spann_synopsis_iso_io`. Without
    this cell the feature has no demonstrated product win and
    should not ship.
26. **Add the joint bloom+synopsis bench cell** (`011` in the
    8-cell hypercube) to verify the two gates compose correctly
    and don't regress each other.

---

## 16. Operational guarantees

- **Disabled** (`head_synopsis_enabled = false`): no overhead. No
  blob written or loaded. Gate is a no-op.
- **Enabled** with sensible top-K and max-cardinality defaults:
  - Build cost: O(total docs × write_nprobe) at compaction.
    Sub-second for a 9 K-doc segment; bounded for production
    segments (1 M docs ⇒ ~30 M increment ops, a few seconds).
  - Storage cost: bounded by `top_k × keys × heads`. Default
    settings cap at ~20 MB per segment shard.
  - Query cost: one hashmap lookup per (head, token) pair. Cheap.
- **Correctness contract**: gate is exact. Any false-negative drop
  is a *bug*, not a tunable. The bench's `[gate-audit]` panic
  guard enforces this.
- **Compositional with bloom**: synopsis after bloom in the gate
  pipeline; both correct, both can drop heads independently;
  intersection is the kept set.

The gate is **strictly an optimization**. Disabling all synopsis
flags reverts to existing SPANN behavior with no semantic change.
The metadata segment's inverted index handles all filter
correctness.
