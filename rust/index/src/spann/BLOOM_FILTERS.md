# BloomHeads — Implementation Guide

A replication-grade reference for the per-head metadata Bloom filter gate
in SPANN. Phase 1 + Phase 1.5; Phase 2 (HMS) is intentionally out of scope.

This document is meant to be sufficient to rebuild the feature from a
clean checkout: every file touched, every data structure, every wiring
point, every bug found and how it was fixed, and the benchmarks needed
to validate that recall is preserved.

---

## Table of contents

1. [Architectural placement](#1-architectural-placement)
2. [Files added / modified](#2-files-added--modified)
3. [Data structures](#3-data-structures)
4. [Storage layout](#4-storage-layout)
5. [Configuration](#5-configuration)
6. [Write path](#6-write-path)
7. [Phase 1.5 commit-time rebuild](#7-phase-15-commit-time-rebuild)
8. [Read path / query gate](#8-read-path--query-gate)
9. [Test coverage](#9-test-coverage)
10. [Bugs found during implementation](#10-bugs-found-during-implementation)
11. [Replication checklist](#11-replication-checklist)
12. [Benchmarking & recall validation](#12-benchmarking--recall-validation)
13. [Operational guarantees](#13-operational-guarantees)

---

## 1. Architectural placement

A Chroma collection shard has three independent segments — Record
(canonical doc store), Metadata (inverted index, source of truth for
filtering), and Vector (SPANN or HNSW). Bloom filters are
**side-files inside the vector segment**, not a separate segment.
They optimize SPANN's cluster-routing decision when the query has a
metadata predicate.

Query flow when a filter is present:

1. `KnnFilterOrchestrator` → `FilterOperator` reads the metadata
   segment's inverted index → produces a `RoaringBitmap` of allowed
   `doc_offset_id`s. **This bitmap is the source of truth for which
   docs match the predicate.**
2. `KnnFilterOrchestrator` also extracts a typed `EqualityTokens` from
   the original `Where` clause.
3. `SpannKnnOrchestrator` runs `SpannCentersSearchOperator` → HNSW
   `rng_query` returns candidate `head_ids` → **bloom gate** drops
   heads whose filters say "definitely no token T" → fetch posting
   lists for surviving heads → `BfPL` does final filtering with the
   bitmap.

**Correctness contract.** The bloom gate must never produce a false
negative: it must never drop a head whose PL contains a doc matching
the predicate. Over-positives (keeping a head that doesn't match) are
fine — wasted I/O, never wrong answers.

The reason it's correctness-critical despite being "just routing":
once a head is dropped, no doc in its PL ever reaches BfPL, so the
metadata bitmap from step 1 doesn't help.

---

## 2. Files added / modified

**Added:**
- `rust/index/src/spann/head_bloom.rs` — module
- `rust/index/src/spann/HEAD_BLOOM_STALENESS.md` — Phase 1.5 design doc
- `rust/index/src/spann/BLOOM_FILTERS.md` — this file

**Modified (in dependency order):**
- `rust/types/src/segment.rs` — `HEAD_BLOOM_FILTERS_PATH` constant
- `rust/index/Cargo.toml` — `fastbloom`, `bincode` deps
- `rust/index/src/config.rs` — `SpannProviderConfig` flags
- `rust/index/src/spann.rs` — module declaration
- `rust/index/src/spann/types.rs` — `SpannIndexWriter` / `SpannIndexReader`
  / `SpannIndexFlusher` / `SpannIndexIds` / `SpannIndexWriterError`
  plumbing; commit serialization; reader load; `gate_heads`
- `rust/index/src/spann/fast_writer.rs` — `HeadBloomBlobFlusher` lives
  here (used by both writers)
- `rust/segment/src/spann_provider.rs` — `SpannProvider` fields +
  `try_from_config`
- `rust/segment/src/distributed_spann.rs` —
  `SpannSegmentWriterShard::from_segment` extracts metadata tokens,
  `SpannSegmentReaderShard::from_segment` loads blob path,
  `SpannSegmentReaderShard::gate_heads` forwarder, flush path records
  `HEAD_BLOOM_FILTERS_PATH`
- `rust/segment/src/test.rs` — `SpannProvider` literal in test
- `rust/worker/src/execution/operators/spann_centers_search.rs` —
  input/output types, gate call
- `rust/worker/src/execution/orchestration/knn_filter.rs` —
  `KnnFilterOutput.head_bloom_tokens`, `extract_equality_tokens` call
- `rust/worker/src/execution/orchestration/spann_knn.rs` — pass tokens
  to operator
- `rust/worker/src/execution/orchestration/compact.rs` +
  `rust/worker/src/compactor/compaction_manager.rs` — `SpannProvider`
  literals
- `rust/worker/benches/spann.rs` — bench cell `bench_spann_bloom_ablation`
  + `user_filter_matches` early-bail + audit guard

---

## 3. Data structures

### Module constants

```rust
// In head_bloom.rs:
pub const HEAD_BLOOM_SEED: u128 = 0xCAFE_F00D_BEEF_BABE_DEAD_C0DE_FACE_BEAD;
pub const TARGET_FPR: f64 = 0.001;
```

The fixed seed is **critical**: it makes hashes stable across writer
instances and across serialization round-trips.

### Token format

Tokens are flat strings with type tags so that
`Int(1)`/`Bool(true)`/`Str("1")` never collide:

```rust
pub fn metadata_token(key: &str, value: &MetadataValue) -> Option<String> {
    match value {
        MetadataValue::Bool(v)  => Some(format!("meta::{}::bool::{}",  key, v)),
        MetadataValue::Int(v)   => Some(format!("meta::{}::int::{}",   key, v)),
        MetadataValue::Float(v) => Some(format!("meta::{}::float::{}", key, v.to_bits())),
        MetadataValue::Str(v)   => Some(format!("meta::{}::str::{}",   key, v)),
        // SparseVector / arrays → None (skipped)
    }
}
```

Float uses `to_bits()` so NaN/-0.0 are deterministic.

`pub fn doc_tokens(metadata: &HashMap<String, MetadataValue>) -> Vec<String>`
walks a doc's metadata and returns all valid tokens.

### `EqualityTokens` — predicate type

```rust
pub enum EqualityTokens {
    And(Vec<String>),     // every token must be possibly-present
    Or(Vec<String>),      // at least one must be possibly-present
    Unsupported,          // gate is a no-op for this query
}
```

`is_gateable()` returns `false` for `Unsupported` and for empty
`And`/`Or`.

### `extract_equality_tokens(where: &Where) -> EqualityTokens`

Recursive walk:

- **Leaf** `Metadata(MetadataExpression)`:
  - `Primitive(Equal, v)` with a tokenizable value → `And(vec![tok])`
  - `Set(In, [...])` → `Or(vec![toks])` (each list element a token)
  - Anything else (`NotEqual`, range, `ArrayContains`, `NotIn`) →
    `Unsupported`
- **Document(...)** → `Unsupported`
- **Composite(And, children)**: every child must collapse to `And(_)`.
  Concatenate. Otherwise `Unsupported`.
- **Composite(Or, children)**: each child must be `And` of length 1 or
  `Or`. Concatenate to `Or`. Otherwise `Unsupported` — i.e. AND-of-OR
  is not supported (would need product expansion).

Deliberately bails to `Unsupported` rather than guessing on shapes
that would need range or set-difference reasoning.

### `HeadBloom` — one head's filter

```rust
pub struct HeadBloom {
    inner: AtomicBloomFilter,                       // fastbloom 0.17, atomic
    capacity: u64,                                   // sized-for value
    live_count: std::sync::atomic::AtomicU64,        // diagnostic
}
```

Constructed via:

```rust
let inner = AtomicBloomFilter::with_false_pos(TARGET_FPR)
    .seed(&HEAD_BLOOM_SEED)
    .expected_items(capacity as usize);
```

API:
- `insert(token: &str)`
- `insert_many<I: IntoIterator<Item = S>, S: AsRef<str>>(&self, tokens: I)`
- `contains(token: &str) -> bool`
- `union_in_place(&self, other: &HeadBloom) -> bool` — ORs the bit
  vectors. Returns `false` and leaves self untouched if bit-vector
  lengths differ.
- `clone()` — deep copy of bits via `AtomicBloomFilter::clone()`.

### `HeadBloomBlob` — serialized form

```rust
#[derive(Serialize, Deserialize)]
pub struct HeadBloomBlob {
    pub bits: Vec<u64>,
    pub num_hashes: u32,
    pub capacity: u64,
    pub live_count: u64,
}
```

`HeadBloom::to_blob()` and `HeadBloom::from_blob(blob)` round-trip via
`AtomicBloomFilter::iter()` (yields `u64` chunks) and
`AtomicBloomFilter::from_vec(bits).seed(...).hashes(...)`.

### `HeadBloomCache` — writer-side state

```rust
pub struct HeadBloomCache {
    filters: Arc<DashMap<u32, Arc<HeadBloom>>>,             // head_id → filter
    stale: Arc<dashmap::DashSet<u32>>,                       // monotonic
    doc_tokens: Option<Arc<DashMap<u32, Arc<Vec<String>>>>>, // Phase 1.5
    capacity: u64,
}
```

Two constructors:
- `HeadBloomCache::new(capacity)` — Phase 1 MVP (no `doc_tokens` map)
- `HeadBloomCache::new_with_doc_tokens(capacity)` — Phase 1.5

Phase 1 methods:
- `insert_tokens(head_id, &[String])` — gets-or-creates filter, calls
  `insert_many`. Cheap.
- `insert_loaded(head_id, HeadBloom)` — used by reader on load.
- `clone_into(src_head_id, dst_head_id)` — split: child inherits
  parent's filter. **Critically, does not hold the source `Ref<>`
  while inserting** (deadlock fix; see §10). Propagates `stale` flag
  src → dst.
- `union_into(src, dst)` — merge: dst absorbs src's bits. Falls back
  to dropping dst's filter (and marking stale) on capacity mismatch.
  Propagates stale.
- `remove(head_id)` — true removal (genuine head deletion).
- `mark_stale(head_id)` — adds to monotonic stale set.
- `is_stale(head_id) -> bool`
- `get(head_id) -> Option<Arc<HeadBloom>>` — returns `None` for stale
  heads (gate then keeps them).
- `iter()` / `iter_non_stale()` — for commit serialization (skips
  stale, since reader has no stale set).
- `is_empty()`, `len()`, `capacity_per_head()`

Phase 1.5 additions:
- `seed_doc_tokens(doc_id, &[String])` — populates doc_tokens map.
  No-op when map absent.
- `forget_doc(doc_id)` — delete-path counterpart.
- `record_doc_appended_to_head(head_id, doc_id) -> bool` — Option 1's
  reassign hook. Looks up doc's tokens, inserts into dest filter,
  returns true on cache hit.
- `get_doc_tokens(doc_id) -> Option<Arc<Vec<String>>>` — Option 2 uses
  this during rebuild.
- `replace_filter(head_id, HeadBloom)` — Option 2's commit-rebuild
  hook. Atomic replace + stale clear.
- `touched_head_ids() -> HashSet<u32>` — union of filter keys and
  stale keys.

### Config bundles

```rust
pub struct HeadBloomWriteConfig<'a> {
    pub capacity_per_head: u64,
    pub existing_blob_path: Option<&'a str>,
    pub doc_tokens_cache_enabled: bool,    // Phase 1.5 Option 1
    pub commit_rebuild_enabled: bool,      // Phase 1.5 Option 2
}

pub struct HeadBloomReadConfig<'a> {
    pub blob_path: Option<&'a str>,
}
```

Both are `Option<...>` at the `from_id` boundary. `None` disables the
feature end-to-end.

### `gate_heads` (free function)

```rust
pub fn gate_heads<I, F>(
    candidate_head_ids: I,
    tokens: &EqualityTokens,
    lookup: F,
) -> Vec<u32>
where
    I: IntoIterator<Item = u32>,
    F: Fn(u32) -> Option<Arc<HeadBloom>>,
```

Filters candidates by:
- `Unsupported` or empty: pass-through.
- `And(toks)`: keep `hid` iff lookup is `None` (no filter — keep) OR
  every token is `bloom.contains(t)`.
- `Or(toks)`: keep `hid` iff lookup is `None` OR any token is
  `bloom.contains(t)`.

The closure-returns-`Arc` signature is intentional: lets
`HeadBloomCache::get` return an owned Arc and avoid lifetime tangles
with DashMap's `Ref<>`.

---

## 4. Storage layout

### Path constant

In `rust/types/src/segment.rs`:

```rust
pub const HEAD_BLOOM_FILTERS_PATH: &str = "head_bloom_filters_path";
```

This is a key in `Segment::file_path: HashMap<String, Vec<String>>`
(or `SegmentShard::file_path: HashMap<String, String>` per shard).

### Blob format

A single binary object per segment shard:

```rust
let map: HashMap<u32, HeadBloomBlob> = /* from cache */;
let bytes = bincode::serialize(&map)?;
```

Storage path is `<segment.prefix_path>/<fresh_uuid>` (immutable
artifacts; never overwritten).

### Persistence lifecycle (writer side)

`SpannIndexFlusher` (in types.rs) holds a
`head_bloom_blob: Option<HeadBloomBlobFlusher>` (struct lives in
`fast_writer.rs` for sharing reasons):

```rust
pub struct HeadBloomBlobFlusher {
    pub bytes: Vec<u8>,
    pub path: String,
    pub blob_id: Uuid,
    pub storage: Option<Arc<Storage>>,
}
```

`HeadBloomBlobFlusher::save()` calls
`Storage::put_bytes(path, bytes, PutOptions::default())`. No-ops if
`storage` is None (in-memory blockfile provider).

In `SpannIndexWriter::commit`:

```rust
let head_bloom_blob = if self.head_bloom_enabled && !self.head_bloom_cache.is_empty() {
    let mut map = HashMap::new();
    for (head_id, filter) in self.head_bloom_cache.iter_non_stale() {
        map.insert(head_id, filter.to_blob());
    }
    match bincode::serialize(&map) {
        Ok(bytes) => {
            let blob_id = Uuid::new_v4();
            let path = if self.prefix_path.is_empty() {
                blob_id.to_string()
            } else {
                format!("{}/{}", self.prefix_path, blob_id)
            };
            Some(HeadBloomBlobFlusher { bytes, path, blob_id, storage: ... })
        }
        Err(_) => None,
    }
} else { None };
```

`SpannIndexFlusher::flush()`:

```rust
let head_bloom_blob_path = self.head_bloom_blob.as_ref().map(|b| b.path.clone());
// ... existing PL/versions/HNSW flush ...
if let Some(blob) = self.head_bloom_blob {
    blob.save().await?;
    res.head_bloom_blob_path = head_bloom_blob_path;
}
```

`SpannIndexIds` adds:

```rust
pub head_bloom_blob_path: Option<String>,
```

Error type adds:

```rust
#[error("Error saving head bloom filter blob: {0}")]
HeadBloomBlobSaveError(String),
```

### Persistence lifecycle (reader side)

`SpannIndexReader::from_id` accepts
`head_bloom_config: Option<HeadBloomReadConfig<'_>>`. If the config
has a `blob_path` and the BlockfileProvider has storage, it loads:

```rust
async fn load_head_bloom_filters(storage: &Storage, path: &str) -> Result<HeadBloomCache, String> {
    let bytes = storage.get(path, GetOptions::new(StorageRequestPriority::P0)).await?;
    let map: HashMap<u32, HeadBloomBlob> = bincode::deserialize(bytes.as_slice())?;
    let cache = HeadBloomCache::new(0); // capacity unused on reader
    for (head_id, blob) in map {
        cache.insert_loaded(head_id, HeadBloom::from_blob(blob));
    }
    Ok(cache)
}
```

Stored on `SpannIndexReader::head_bloom_filters: Option<Arc<HeadBloomCache>>`.
`None` on missing/unreadable blob — gate degrades to no-op
(correctness-safe).

### Segment-level wiring

`SpannSegmentWriterShard::from_segment`:

```rust
let head_bloom_blob_path: Option<String> = segment
    .file_path
    .get(chroma_types::HEAD_BLOOM_FILTERS_PATH)
    .cloned();
let head_bloom_config = if head_bloom_enabled {
    let capacity = (params.split_threshold as u64)
        .saturating_mul(head_bloom_capacity_factor.max(1) as u64)
        .max(1);
    Some(HeadBloomWriteConfig {
        capacity_per_head: capacity,
        existing_blob_path: head_bloom_blob_path.as_deref(),
        doc_tokens_cache_enabled: head_bloom_doc_tokens_cache,
        commit_rebuild_enabled: head_bloom_commit_rebuild,
    })
} else { None };
```

After flush, `SpannSegmentFlusherShard::flush`:

```rust
if let Some(blob_path) = index_ids.head_bloom_blob_path.as_ref() {
    index_id_map.insert(
        chroma_types::HEAD_BLOOM_FILTERS_PATH.to_string(),
        vec![blob_path.clone()],
    );
}
```

So the segment's `file_path` now references the persisted blob, which
the next reader instance picks up.

---

## 5. Configuration

`rust/index/src/config.rs`:

```rust
pub struct SpannProviderConfig {
    // existing...
    #[serde(default = "default_head_bloom_enabled")]
    pub head_bloom_enabled: bool,                       // master switch

    #[serde(default = "default_head_bloom_capacity_factor")]
    pub head_bloom_capacity_factor: u32,                // capacity = split_threshold × factor

    #[serde(default = "default_head_bloom_doc_tokens_cache")]
    pub head_bloom_doc_tokens_cache: bool,              // Phase 1.5 Option 1

    #[serde(default = "default_head_bloom_commit_rebuild")]
    pub head_bloom_commit_rebuild: bool,                // Phase 1.5 Option 2
}
```

All four default `false` / `4`. The two Phase 1.5 flags only take
effect when `head_bloom_enabled = true`.

`SpannProvider` (in `rust/segment/src/spann_provider.rs`) carries the
same four fields, populated by `try_from_config`. `SpannProvider::write`
passes them all to `SpannSegmentWriterShard::from_segment`.

Capacity calculation in `from_segment` (after `params: InternalSpannConfiguration`
is loaded from schema):

```rust
let capacity = (params.split_threshold as u64)
    .saturating_mul(head_bloom_capacity_factor.max(1) as u64)
    .max(1);
```

Default: `split_threshold * 4`. With `split_threshold=200` and
FPR=0.001 → ~12 KB per filter.

---

## 6. Write path

The legacy `SpannIndexWriter` (in `rust/index/src/spann/types.rs`) is
what's actually used in production. The "fast" `FastSpannIndexWriter`
has parallel scaffolding but isn't wired to any orchestrator yet.

Writer fields added:

```rust
pub head_bloom_enabled: bool,
pub head_bloom_cache: HeadBloomCache,
pub head_bloom_doc_tokens_cache_enabled: bool,
pub head_bloom_commit_rebuild_enabled: bool,
```

`from_id` reads `head_bloom_config: Option<HeadBloomWriteConfig<'_>>`
→ unpacks the four fields → allocates either `HeadBloomCache::new` or
`HeadBloomCache::new_with_doc_tokens` → if `existing_blob_path` is
set, loads the blob into the cache.

### Public mutation entry points

```rust
pub async fn add(&self, id: u32, embedding: &[f32])
    → calls add_with_tokens(id, e, &[])
pub async fn add_with_tokens(&self, id, embedding, &[String])
    → calls add_with_metadata_tokens(...)
pub async fn add_with_metadata_tokens(&self, id, embedding, bloom_tokens, synopsis_tokens) {
    let version = self.add_versions_map(id).await;
    // ... normalize embedding ...
    if self.head_bloom_enabled {
        self.head_bloom_cache.seed_doc_tokens(id, bloom_tokens); // Phase 1.5
    }
    self.add_to_postings_list(id, version, &normalized, Some(bloom_tokens)).await
}
```

`update_with_metadata_tokens` is symmetric (replaces seed tokens;
bloom can't unset bits, so old positives stay until rebuild).

`delete(id)`:

```rust
let mut guard = self.versions_map.write().await;
guard.versions_map.insert(id, 0);
drop(guard);
if self.head_bloom_enabled {
    self.head_bloom_cache.forget_doc(id);  // Phase 1.5
}
```

### Internal: `add_to_postings_list`

Called by `add`/`update`. Two branches:

**Branch A — `ids.is_empty()` (no centroids exist yet)**: create a new
head with `next_head_id.fetch_add(1)`. Write the PL with one doc.
Then bloom hook:

```rust
if self.head_bloom_enabled {
    match metadata_tokens {
        Some(toks) => self.head_bloom_cache.insert_tokens(next_id, toks),
        None       => self.handle_reassign_bloom(next_id, id),
    }
}
```

**Branch B — RNG returned head_ids**: loop `head_ids` and call
`append(head_id, id, version, embedding, head_emb, metadata_tokens)`.
Each `append` may trigger split.

### Internal: `append`

Acquires the partitioned mutex on `head_id`. Fetches PL, appends the
doc, runs version cleanup. Two sub-branches:

**B1 — no split (`up_to_date_index <= split_threshold`)**: write PL
back. Bloom:

```rust
if self.head_bloom_enabled {
    match metadata_tokens {
        Some(toks) => self.head_bloom_cache.insert_tokens(head_id, toks),
        None       => self.handle_reassign_bloom(head_id, id),
    }
}
```

**B2 — split**: KMeans → 2 clusters. For each `k`:
- If `same_head` (centroid matches existing head_emb within ε): reuse
  `head_id` for cluster `k`. Write PL.
- Else: allocate `next_id`, write PL, add to HNSW.

Bloom hooks for B2 (per-cluster, inside the loop):

- New child (`!same_head`):
  ```rust
  if self.head_bloom_enabled {
      match metadata_tokens {
          Some(toks) => {
              self.head_bloom_cache.clone_into(head_id, next_id as u32);
              self.head_bloom_cache.insert_tokens(next_id as u32, toks);
          }
          None => self.handle_reassign_bloom(next_id as u32, id),
      }
  }
  ```
- After the loop, if `!same_head`: parent head is decommissioned.
  `posting_list_writer.delete`, HNSW delete, and:
  ```rust
  if self.head_bloom_enabled {
      self.head_bloom_cache.remove(head_id);
  }
  ```
- After the loop, if `same_head`: parent reused. Bloom:
  ```rust
  if self.head_bloom_enabled {
      match metadata_tokens {
          Some(toks) => self.head_bloom_cache.insert_tokens(head_id, toks),
          None       => self.handle_reassign_bloom(head_id, id),
      }
  }
  ```

After the loop, `collect_and_reassign(...)` runs. That eventually
calls `reassign(...)` → `append(..., metadata_tokens=None)`
recursively.

### Internal: `reassign`

```rust
async fn reassign(&self, doc_offset_id, doc_version, doc_embedding, prev_head_id, reason) {
    // version checks, bail if outdated
    // RNG query for new heads
    // bail if prev_head_id is in nearest_head_ids (no-op reassign)
    // increment doc_version
    for nearest_head_id in nearest_head_ids {
        // skip if outdated meanwhile
        self.append(nearest_head_id, doc_offset_id, next_version,
                    doc_embedding, nearest_head_emb,
                    None /* metadata_tokens — not in scope */).await?;
    }
}
```

The `None` propagates into `append`'s bloom hook, which routes through
`handle_reassign_bloom`.

### `handle_reassign_bloom` — the core Phase 1.5 helper

```rust
fn handle_reassign_bloom(&self, head_id: u32, doc_id: u32) {
    if !self.head_bloom_enabled { return; }
    let used_cache = if self.head_bloom_doc_tokens_cache_enabled {
        self.head_bloom_cache.record_doc_appended_to_head(head_id, doc_id)
    } else { false };
    if !used_cache {
        self.head_bloom_cache.mark_stale(head_id);
    }
}
```

So:
- Phase 1 MVP (`doc_tokens_cache=false`): every reassign-induced
  append marks the destination stale.
- Phase 1.5 Option 1 (`doc_tokens_cache=true`): cache lookup. On hit,
  the doc's tokens are inserted into the destination's filter — gate
  stays effective. On miss, fall back to mark_stale.

### `try_delete_posting_list` — head deletion

Called when a head's PL becomes all-outdated (after a reassign sweep).
Genuine head removal:

```rust
self.posting_list_writer.delete::<...>("", head_id).await?;
HNSW delete...
if self.head_bloom_enabled {
    self.head_bloom_cache.remove(head_id);
}
```

### `merge_posting_lists` — GC merge

Two `head_bloom_cache.remove(...)` calls (one per merge direction):
the source head is decommissioned; its filter is dropped. The target
absorbs the source's bloom via `union_into`:

```rust
if self.head_bloom_enabled {
    self.head_bloom_cache.union_into(head_id as u32, nearest_head_id as u32);
    self.head_bloom_cache.remove(head_id as u32);
}
```

`union_into` propagates stale and falls back to dropping the target's
filter on capacity mismatch.

---

## 7. Phase 1.5 commit-time rebuild

In `SpannIndexWriter::commit`, before serializing the cache:

```rust
async fn rebuild_blooms_from_cache(&self) -> Result<(), SpannIndexWriterError> {
    let head_ids = self.head_bloom_cache.touched_head_ids();
    for head_id in head_ids {
        if self.is_head_deleted(head_id as usize).await? { continue; }
        let pl = match self.posting_list_writer
            .get_owned::<u32, &SpannPostingList<'_>>("", head_id).await {
            Ok(Some(pl)) => pl,
            Ok(None) => continue,
            Err(e) => return Err(...),
        };
        let (doc_offset_ids, doc_versions, _) = pl;
        let fresh = HeadBloom::new(self.head_bloom_cache.capacity_per_head().max(1));
        let mut any_miss = false;
        let version_map_guard = self.versions_map.read().await;
        for (doc_id, doc_version) in doc_offset_ids.iter().zip(doc_versions.iter()) {
            let current = match version_map_guard.versions_map.get(doc_id) {
                Some(v) => *v,
                None => continue,
            };
            if current == 0 || *doc_version < current { continue; }
            let Some(tokens) = self.head_bloom_cache.get_doc_tokens(*doc_id) else {
                any_miss = true;
                break;
            };
            fresh.insert_many(tokens.iter().map(|s| s.as_str()));
        }
        drop(version_map_guard);
        if any_miss {
            self.head_bloom_cache.mark_stale(head_id);
        } else {
            self.head_bloom_cache.replace_filter(head_id, fresh);
        }
    }
}
```

Called only when `head_bloom_enabled && head_bloom_commit_rebuild_enabled`.
After this pass, `iter_non_stale` serializes the rebuilt filters.

The iteration-by-`touched_head_ids` includes both currently-have-a-filter
heads and stale heads, so reassign-induced staleness is the rebuild's
primary target.

---

## 8. Read path / query gate

### `SpannIndexReader` field

```rust
pub head_bloom_filters: Option<Arc<HeadBloomCache>>,
```

`from_id` loads via `load_head_bloom_filters` (see §4) when the read
config has a path.

### `SpannIndexReader::gate_heads`

```rust
pub fn gate_heads(&self, candidate_head_ids: &[usize], tokens: &EqualityTokens) -> Vec<usize> {
    let Some(cache) = &self.head_bloom_filters else { return candidate_head_ids.to_vec(); };
    if !tokens.is_gateable() { return candidate_head_ids.to_vec(); }
    let kept = head_bloom::gate_heads(
        candidate_head_ids.iter().map(|h| *h as u32),
        tokens,
        |hid| cache.get(hid),  // Option<Arc<HeadBloom>>
    );
    kept.into_iter().map(|h| h as usize).collect()
}
```

`SpannSegmentReaderShard::gate_heads` is a thin forwarder.

### Operator wiring — `spann_centers_search.rs`

```rust
pub(crate) struct SpannCentersSearchInput<'a> {
    // existing: reader, normalized_query, k, filter_selectivity, ...
    pub(crate) head_bloom_tokens: EqualityTokens,
}

pub(crate) struct SpannCentersSearchOutput {
    pub(crate) center_ids: Vec<usize>,
    pub(crate) heads_rng: usize,         // before any filtering
    pub(crate) heads_after_bloom: usize, // after Phase 1 gate
}
```

Inside `Operator::run`:

```rust
let res = reader.rng_query(...).await?;
let center_ids = res.0;
let heads_rng = center_ids.len();
let center_ids = reader.gate_heads(&center_ids, &input.head_bloom_tokens);
let heads_after_bloom = center_ids.len();
```

### Orchestrator wiring

`KnnFilterOutput` (in `rust/worker/src/execution/orchestration/knn_filter.rs`)
gains:

```rust
pub head_bloom_tokens: EqualityTokens,
```

Populated once in `KnnFilterOrchestrator::on_filter_complete`:

```rust
let head_bloom_tokens = match self.filter.where_clause.as_ref() {
    Some(w) => extract_equality_tokens(w),
    None    => EqualityTokens::Unsupported,
};
```

`SpannKnnOrchestrator::initial_tasks` passes
`self.knn_filter_output.head_bloom_tokens.clone()` into
`SpannCentersSearchInput`.

Telemetry on `SpannCentersSearchOutput`:

```rust
tracing::debug!(
    heads_rng = output.heads_rng,
    heads_after_bloom = output.heads_after_bloom,
    "spann centers search completed",
);
```

---

## 9. Test coverage

### Unit tests in `head_bloom.rs::tests`

Phase 1 (12 tests):
- `token_format_is_typed` — types in tokens prevent collisions
- `arrays_and_sparse_yield_no_token` — unsupported types skipped
- `extract_single_equality` / `extract_and_of_equalities` /
  `extract_in_yields_or` — predicate parsing
- `ne_and_ranges_are_unsupported` /
  `nested_or_inside_and_is_unsupported` — fallback to `Unsupported`
- `bloom_no_false_negative` — direct insert, contains-true;
  absent-token → not contained
- `bloom_roundtrip` — bincode serialize → deserialize → membership
  preserved
- `cache_clone_preserves_membership` — `clone_into` copies bits
- `cache_union_merges` — `union_into` ORs bits
- `gate_drops_definite_misses` — gate with `And` correctly drops/keeps

Phase 1.5 (6 additional tests):
- `doc_tokens_cache_record_appended_inserts_token` — Option 1's
  lookup-then-insert path
- `doc_tokens_cache_miss_returns_false` — caller knows to fall back
- `doc_tokens_cache_forget_drops_entries` — delete-path
- `mvp_cache_ignores_doc_tokens_calls` — MVP cache has
  `doc_tokens=None`, all calls no-op
- `replace_filter_clears_stale` — Option 2 hook
- `touched_head_ids_union` — enumeration target

### Integration tests in `types.rs::tests`

- `test_head_bloom_writer_inserts_tokens` — full writer roundtrip
  with `head_bloom_config = Some(...)`. Adds 100 docs with manual
  tokens, asserts the cache is non-empty and one filter contains the
  inserted token.
- `test_head_bloom_disabled_does_not_track` — with
  `head_bloom_config = None`, after 100 inserts the cache is empty.
  Confirms no overhead when disabled.

### Bench cell — `bench_spann_bloom_ablation`

In `rust/worker/benches/spann.rs`:
- Two index builds:
  `add_to_index_and_get_reader_with_bloom(records, bloom_enabled=false, ...)`
  and `(records, bloom_enabled=true, ...)`.
- Bench id: `"spann_bloom_ablation_010"` (000 vs 010 in the 8-cell
  ablation framework).
- Predicate: `bucket = id % 100 == 0`, 100 buckets, 1% selectivity.
  Tokens: `EqualityTokens::And(vec!["meta::bucket::int::0"])`.
- For each query, runs `rng_query` directly with `probe_nbr=32`,
  then `reader.gate_heads(...)`, fetches PLs for survivors, runs
  `SpannBfPlOperator` against the bitmap, computes recall against
  brute-force ground truth.

Bench-level safeguards:
- `user_filter_matches(bench_name)` early-bail at the top of every
  `bench_*` function (sees `cargo bench -- spann_bloom_ablation_010`
  and short-circuits other benches' setup).
- `[setup] building SPANN index ...` progress prints during the index
  builds.
- `[gate-audit]` line that counts dropped heads whose PL contains a
  matching doc; **panics on any false negative** (gated by
  `BLOOM_AUDIT_NO_PANIC=1` env var for diagnostics).

Env-var toggles for Phase 1.5:
- `BLOOM_DOC_TOKENS_CACHE=1` → enable Option 1
- `BLOOM_COMMIT_REBUILD=1` → enable Option 2
- Both default off (MVP).

---

## 10. Bugs found during implementation

### 10.1 DashMap shard-deadlock in `clone_into`

**Symptom**: `cargo bench` hangs forever during the second index
build. The first split creates a new head whose ID happens to share a
DashMap shard with the parent.

**Root cause**:

```rust
let Some(src) = self.filters.get(&src_head_id) else { return; };
// `src` is Ref<>, holds shard read lock for its lifetime
let copy: HeadBloom = (**src).clone();
self.filters.insert(dst_head_id, ...);  // needs shard write lock
// `src` still in scope here — read lock + write lock on same shard = deadlock
```

**Fix**: scope the `Ref<>`:

```rust
let copy = {
    let Some(src) = self.filters.get(&src_head_id) else { return; };
    (**src).clone()
    // Ref dropped here, lock released
};
self.filters.insert(dst_head_id, Arc::new(copy));
```

### 10.2 `remove(head_id)` on reassign destroys other docs' tokens

**Symptom**: 15% false-negative rate in the bench audit.

**Root cause**: The original Phase 1 code did
`head_bloom_cache.remove(head_id)` on reassign-induced appends, on the
theory that the head's filter was no longer trustworthy. But `remove`
deletes the entire filter, losing tokens of *other* docs that were
correctly recorded earlier. Subsequent direct inserts re-create the
filter from scratch with only the new tokens.

**Fix**: replace `remove` with `mark_stale`. The filter is preserved;
the head is marked as having uncertain tokens; the gate treats stale
heads as no-filter (always keep).

### 10.3 Stale propagation through split/merge

**Symptom**: Even with `mark_stale`, recall regressed by ~6%.

**Root cause**: When a stale parent splits, `clone_into` copied the
parent's (incomplete) filter to a new child without propagating the
stale flag. The child appeared trustworthy but had the parent's
missing-token problem.

**Fix**: in `clone_into`, capture
`src_was_stale = self.stale.contains(&src_head_id)` before doing
anything. After the insert, if the parent was stale,
`self.stale.insert(dst_head_id)`. Same in `union_into`.

### 10.4 Stale set isn't serialized — reader trusts incomplete filters

**Symptom**: Recall still regressed even after the propagation fix.

**Root cause**: The blob serialization iterated `head_bloom_cache.iter()`
and serialized every head's filter. Stale heads' (incomplete) filters
were persisted. The reader has no stale-set concept on load and
trusted them.

**Fix**: add `iter_non_stale()` to `HeadBloomCache`. Use it in commit.
Stale heads' filters are simply omitted from the blob; the reader
sees `None` for those `head_id`s and the gate keeps them.

### 10.5 Phase 1.5 Option 1 bug (still open)

**Symptom**: With `head_bloom_doc_tokens_cache=true` and
`head_bloom_commit_rebuild=false`, the bench audit shows ~1%
false-negative rate. The bench panics with the gate-audit guard.

**Root cause**: Unknown. Some path through the SPANN
write/split/reassign cascade lets a doc reach a head's PL without
`record_doc_appended_to_head` being called for that
`(head_id, doc_id)` pair. The seed-then-cache-then-record sequence is
correct in unit tests; the leak only shows under realistic write
loads.

**Workaround**: `head_bloom_commit_rebuild=true` (Option 2) covers
this — at commit, every head's filter is rebuilt from
`PL × doc_tokens`, so Option 1's leak is healed before the blob is
persisted.

The diagnostic plan if pursuing this: add a writer-internal
consistency assertion at commit that walks every head's PL and
verifies each live doc's tokens are in the bloom; bisect the workload
smaller until reproducible; then trace every PL append on the failing
head.

### 10.6 Bench setup costs charged to filtered-out benches

**Symptom**: `cargo bench -- spann_bloom_ablation_010` appears to
hang.

**Root cause**: Criterion's filter check happens *inside*
`c.bench_function`, but the per-bench setup (dataset load, two SPANN
index builds) lives *outside* it in regular Rust code. So even when
the user filters to one bench, the other benches still run their full
setup before `c.bench_function` short-circuits.

**Fix**: a `user_filter_matches(bench_name) -> bool` helper at the top
of each `bench_*` function, re-implementing criterion's CLI filter
parsing (`--exact`, regex matching, value-flag detection). Returns
early with `return;` before any expensive setup.

---

## 11. Replication checklist

In dependency order, this is what to do from a clean tree:

1. **Add the path constant** (`HEAD_BLOOM_FILTERS_PATH`) in
   `chroma_types::segment`.
2. **Add deps** to `chroma-index/Cargo.toml`: `fastbloom`, `bincode`.
3. **Write `head_bloom.rs`** (the module) with:
   - `metadata_token`, `doc_tokens`, `extract_equality_tokens`
   - `EqualityTokens` enum
   - `HeadBloom` struct + `HeadBloomBlob` + `to_blob`/`from_blob`
   - `HeadBloomCache` with both constructors, `mark_stale`,
     `is_stale`, `insert_tokens`, `insert_loaded`, `clone_into`,
     `union_into`, `remove`, `get`, `iter`, `iter_non_stale`,
     `is_empty`, `len`, `capacity_per_head`. **Critical: scope the
     `Ref<>` in `clone_into`. Propagate stale in both `clone_into` and
     `union_into`.**
   - `HeadBloomWriteConfig` / `HeadBloomReadConfig`
   - `gate_heads` free function
4. **Add module declaration** to `spann.rs`.
5. **Write 12 unit tests** covering token extraction, predicate
   parsing, bloom round-trip, cache operations, gate semantics.
6. **Add `head_bloom_blob` field** to `SpannIndexFlusher`. Add
   `head_bloom_blob_path` field to `SpannIndexIds`. Add
   `HeadBloomBlobSaveError` variant. Update `SpannIndexFlusher::flush`
   to call `blob.save().await?`.
7. **Add `HeadBloomBlobFlusher` struct** in `fast_writer.rs` with a
   `save(&self)` method that calls `Storage::put_bytes`.
8. **Plumb `head_bloom_config` through `SpannIndexWriter::from_id`** —
   accept `Option<HeadBloomWriteConfig<'_>>`. Allocate the cache.
   Load existing blob if path provided.
9. **Add 4 fields** to `SpannIndexWriter`: `head_bloom_enabled`,
   `head_bloom_cache`, plus the two Phase 1.5 flags.
10. **Wire `add_with_metadata_tokens` / `update_with_metadata_tokens`**
    to seed `doc_tokens` and call
    `add_to_postings_list(..., Some(bloom_tokens))`.
11. **Wire `delete`** to call `forget_doc`.
12. **Wire `add_to_postings_list`'s new-head branch** with
    `insert_tokens` / `handle_reassign_bloom`.
13. **Wire `append`'s no-split branch** with `insert_tokens` /
    `handle_reassign_bloom`.
14. **Wire `append`'s split branch**:
    - For each new child: `clone_into(parent, child)` then
      `insert_tokens(child, toks)` for direct writes;
      `handle_reassign_bloom(child, doc_id)` for reassigns.
    - For same_head (parent reuse): `insert_tokens(parent, toks)` /
      `handle_reassign_bloom(parent, doc_id)`.
    - When `!same_head` (parent decommissioned):
      `head_bloom_cache.remove(parent)`.
15. **Add `handle_reassign_bloom` private helper** that consults the
    Option 1 flag → cache lookup with mark_stale fallback.
16. **Wire `try_delete_posting_list`** with
    `head_bloom_cache.remove(head_id)`.
17. **Wire `merge_posting_lists`** (both directions): `union_into`
    then `remove` the source.
18. **Wire commit**: serialize via `iter_non_stale()`, build a
    `HeadBloomBlobFlusher`, attach to `SpannIndexFlusher`. If
    `head_bloom_commit_rebuild_enabled`, call
    `rebuild_blooms_from_cache` first.
19. **Implement `rebuild_blooms_from_cache`**: walk
    `touched_head_ids`, fetch each PL, build a fresh `HeadBloom` from
    cached doc-tokens for live entries, `replace_filter`. Mark stale
    on cache miss.
20. **Plumb `head_bloom_config` through `SpannIndexReader::from_id`**
    — accept `Option<HeadBloomReadConfig<'_>>`. Add
    `load_head_bloom_filters` that bincode-deserializes and populates
    a fresh `HeadBloomCache` via `insert_loaded`.
21. **Add `gate_heads` method** on `SpannIndexReader`.
22. **Add four config fields** to `SpannProviderConfig` (master +
    capacity factor + 2 Phase 1.5 flags).
23. **Add four fields to `SpannProvider`** + populate in
    `try_from_config`.
24. **Update `SpannSegmentWriterShard::from_segment`** signature to
    accept the four flags. Read existing blob path from
    `segment.file_path[HEAD_BLOOM_FILTERS_PATH]`. Build
    `HeadBloomWriteConfig` (capacity = split_threshold × factor). Pass
    to `SpannIndexWriter::from_id`.
25. **Update `SpannSegmentReaderShard::from_segment`**: same path
    lookup. Build `HeadBloomReadConfig`. Pass to
    `SpannIndexReader::from_id`. Add a `gate_heads` forwarder.
26. **Update segment writer's `flush`** to insert
    `HEAD_BLOOM_FILTERS_PATH → [blob_path]` when
    `index_ids.head_bloom_blob_path` is `Some`.
27. **Wire the segment writer's `add` / `update`** to extract bloom
    tokens via `head_bloom::doc_tokens(&record.merged_metadata())` and
    pass to `add_with_metadata_tokens`.
28. **Add `head_bloom_tokens: EqualityTokens` to `KnnFilterOutput`**,
    populated in the orchestrator's filter-completion handler via
    `extract_equality_tokens`.
29. **Add `head_bloom_tokens` to `SpannCentersSearchInput`** and
    `heads_rng` / `heads_after_bloom` to `SpannCentersSearchOutput`.
    In the operator, call
    `reader.gate_heads(&center_ids, &input.head_bloom_tokens)` after
    `rng_query`.
30. **`SpannKnnOrchestrator`** passes
    `self.knn_filter_output.head_bloom_tokens.clone()` into the
    centers-search input. Add tracing for `heads_rng` /
    `heads_after_bloom`.
31. **Update test code**: `SpannProvider` literals, `from_id` callers
    (add the two new bool args or `None` to optional config).
32. **Add the iso-probe bench cell `bench_spann_bloom_ablation`**
    (correctness check) with the env-var toggles, progress prints,
    `user_filter_matches` early-bail, and the `[gate-audit]` panic
    guard. Holds `probe_nbr` constant across 000 and 010, lets
    `heads_fetched` vary; reports `Δrecall` (must be ≈ 0) and
    `drop_ratio` (must be > 0 with Phase 1.5 Option 2 enabled).
33. **Add the iso-I/O bench cell** (value claim) — the headline
    measurement for the gate. Holds `heads_fetched` constant across
    000 and 010, lets `probe_nbr` vary; reports `Δrecall` (must be
    > 0). See §12 for the methodology and the two implementation
    strategies (fixed multiplier vs adaptive per-query). Suggested
    cell name: `bench_spann_bloom_iso_io`. Without this cell the
    feature has no demonstrated product win and should not ship.

---

## 12. Benchmarking & recall validation

> **The whole point of the gate is to improve recall at a fixed I/O
> budget, by letting you probe many more clusters and trim them to the
> ones likely to contain matches. Drop ratio measures effectiveness;
> recall preservation at fixed probe count is the *correctness check*;
> recall *improvement* at fixed I/O is the *value claim*. The bench
> must measure both.**

### What "improvement" means here

A bloom gate is a routing optimization layered on top of SPANN's
existing HNSW + posting-list architecture. With a correct
implementation, three claims can be made about it, and they need to
be measured separately:

1. **Recall preserved at the same probe count** (`probe_nbr` fixed,
   bloom on/off, same `heads_fetched_after_gate` is incidental):
   `recall_010 ≈ recall_000` within statistical noise. This is the
   **strict correctness check** — any regression is a bug. Phase 1's
   `[gate-audit]` panic guards against this. **Required.**

2. **Lower I/O at the same recall** (`probe_nbr` fixed, bloom on):
   `heads_after_bloom < heads_rng`, which means fewer posting lists
   are fetched per query for similar-or-equal recall. This is the
   **effectiveness signal** — `drop_ratio > 0`. **Required.**

3. **Higher recall at the same I/O budget** (held fixed: number of
   posting lists fetched. Free: `probe_nbr`): with the gate trimming
   useless heads, you can crank `probe_nbr` much higher and still
   fetch the same number of useful PLs. Recall improves because the
   PLs you do fetch are drawn from a much larger candidate pool.
   **This is the actual product win — required to claim the gate is
   worth its complexity.**

Claims (1) and (2) say "we didn't break anything and we did less
work." Claim (3) says "we did a *better* job with the same work
budget." The first two are the safety net; the third is the reason
to build the feature.

### What "iso-I/O" means

"Iso-" is Greek for "equal." An **iso-I/O** comparison holds the I/O
cost (concretely: the number of fetched posting lists per query)
constant across two configurations, then compares recall.

The contrast is **iso-probe**, which holds `probe_nbr` constant.
Iso-probe is what the current bench measures.

Why iso-I/O is the right axis for evaluating the bloom gate:

- **What's free**: probing more centroids in HNSW. The HNSW graph is
  in memory; visiting more nodes costs nanoseconds.
- **What's expensive**: fetching posting lists. Each PL is a blockfile
  read — disk or network round-trip.

So a user's real-world budget is "how many PL reads can I afford per
query," not "how many HNSW nodes can I visit." Iso-I/O comparisons
measure: at the same number of PL reads, did the gate let us pick
*better* PLs?

The mechanic:

```
Without bloom:  probe_nbr clusters  →  probe_nbr PLs fetched.
With bloom:     probe_nbr clusters  →  probe_nbr × (1 − drop_ratio) PLs fetched.
```

So at a fixed I/O budget `B`:

```
Baseline (000):  probe_nbr = B,                       fetch B PLs.
Bloomed  (010):  probe_nbr = B / (1 − drop_ratio),    fetch ≈B PLs (post-gate).
```

In the bloomed case we *consider* `B / (1 − drop_ratio)` clusters —
far more candidates — but the gate trims them down to the same `B`
fetches. The `B` PLs we actually read are drawn from a much wider
pool, so they're more likely to contain matching docs. **That's where
the recall win comes from.**

Concrete example with `drop_ratio = 0.7` and budget `B = 32`:
- 000: probe 32 clusters, fetch all 32 → recall over those 32.
- 010: probe ~107 clusters, gate drops ~75, fetch the surviving ~32 →
  recall over a hand-picked 32 from a pool of 107.

If `recall_010 > recall_000` in this comparison, the gate is doing
its job. If `recall_010 ≤ recall_000`, the gate isn't pulling its
weight on this workload (likely false-positive rate too high or
predicate selectivity wrong for the bloom design) and shouldn't ship.

### The validation matrix

The `bench_spann_bloom_ablation_010` cell must produce **both**
iso-probe and iso-I/O numbers. Current implementation only does
iso-probe; iso-I/O is **a required addition before claiming any
recall benefit**.

#### Iso-probe (correctness check, currently implemented)

```
========== BloomHeads 000 vs 010 ablation (iso-probe) ==========
  selectivity=1%  buckets=100  k=10  probe_nbr=32
  000 (no bloom)  recall_mean=…  heads_fetched=32  drop_ratio=0.0000
  010 (bloom)     recall_mean=…  heads_fetched=…   drop_ratio=…       Δrecall=…
  [gate-audit] dropped N heads; M would have contained matching docs
=================================================================
```

#### Iso-I/O (value claim, REQUIRED ADDITION)

```
========== BloomHeads 000 vs 010 ablation (iso-I/O, B=32) ==========
  selectivity=1%  buckets=100  k=10  io_budget=32
  000 (no bloom)  probe_nbr=32   heads_fetched=32  recall_mean=R0
  010 (bloom)     probe_nbr=…    heads_fetched=≈32 recall_mean=R1   Δrecall=R1-R0
====================================================================
```

The 010 row picks `probe_nbr_010` adaptively or with a fixed
multiplier so that `heads_fetched_010 ≈ heads_fetched_000`. The
report shows the recall delta at matched I/O.

Acceptable headline result: `R1 > R0 + 0.02` (recall improved by at
least 2 percentage points at the same I/O budget). Anything less is
either workload-mismatched or gate-broken.

#### Rules for accepting a run

| Check | Where | Required for ship | What it means |
|---|---|---|---|
| `[gate-audit] M=0` | iso-probe cell | **Yes** (bench panics otherwise) | No false negatives. Strict correctness. |
| `\|Δrecall\| < 0.02` | iso-probe cell | **Yes** | Recall preserved within bench noise at fixed probe count. |
| `drop_ratio > 0` | iso-probe cell | **Yes** | Gate is doing work. If 0, the gate is a no-op (e.g. all heads stale on heavy-reassign workloads). |
| `R1 > R0` (iso-I/O) | iso-I/O cell | **Yes** | The gate is worth its complexity — recall improved at fixed I/O. |
| `R1 > R0 + 0.02` (iso-I/O) | iso-I/O cell | Recommended | Improvement is large enough to matter end-to-end. |

Any false-negative count > 0 means the bloom is producing wrong
answers — that's a Phase 1 contract violation, not a regression to
"tune away." Either fix the bug or disable the responsible flag
combination.

A non-positive `R1 − R0` at fixed I/O means the gate has correct
plumbing but the workload doesn't benefit. Don't enable the feature
in that configuration; investigate whether selectivity, drop_ratio,
or capacity sizing is the cause.

### Caveat: absolute recall is artificially low in the current cell

The bench calls `rng_query` directly with `probe_nbr=32` and bypasses
`SpannSegmentReaderShard::rng_query`, which would have applied
**adaptive nprobe** (scaling probe count by `1/filter_selectivity`).
At 1% selectivity, the adaptive path would expand to ~3 200 probes
(clamped to `params.search_nprobe`); the bench shorts that out and
sticks with 32, which only reaches a fraction of the matching docs.
That's why the absolute recall numbers hover around 0.55 instead of
the 0.95+ a tuned ANN should hit.

This is **fine for ablation** — both 000 and 010 use the same fixed
`probe_nbr`, so any recall delta is attributable to the gate, not to
probe coverage. But it is **not fine for "is this index any good?"** —
which is a different question.

To get realistic absolute numbers, change the bench in one of these
ways:

- **Use `SpannSegmentReaderShard::rng_query` instead of the bare
  `rng_query`** — picks up adaptive nprobe.
- **Bump `probe_nbr`** — try 64, 128. Recall climbs toward 0.9.
- **Lower selectivity** — fewer buckets means more matches per query;
  the gate's `drop_ratio` will be lower but recall higher.

### How to actually run the benchmarks

**Prereq: build the bench binary in release mode once.**

```bash
cargo build --bench spann -p worker --release
```

The binary path is something like
`target/release/deps/spann-<hash>` — find it with `ls -lt target/release/deps/spann-* | grep -v '\\.'`.

**MVP run (Phase 1 only, both Phase 1.5 flags off):**

```bash
./target/release/deps/spann-<hash> \
    --bench spann_bloom_ablation_010 \
    --warm-up-time 1 --measurement-time 1
```

Expected on the current bench: `[gate-audit] M=0`, `Δrecall < 0.03`,
`drop_ratio = 0.0` (every head stale on this heavy-reassign
workload).

**Phase 1.5 Option 2 (the recommended config):**

```bash
BLOOM_COMMIT_REBUILD=1 \
  ./target/release/deps/spann-<hash> \
    --bench spann_bloom_ablation_010 \
    --warm-up-time 1 --measurement-time 1
```

Expected: `[gate-audit] M=0`, `Δrecall ≈ 0`, `drop_ratio ≈ 0.7`. The
gate is dropping 70% of probed heads with zero false negatives.

**Both Phase 1.5 options on:**

```bash
BLOOM_DOC_TOKENS_CACHE=1 BLOOM_COMMIT_REBUILD=1 \
  ./target/release/deps/spann-<hash> \
    --bench spann_bloom_ablation_010 \
    --warm-up-time 1 --measurement-time 1
```

Expected: `[gate-audit] M=0`, `Δrecall ≈ 0`, `drop_ratio ≈ 0.7`.
Same as Option 2 alone (Option 2 covers Option 1's leak at commit
time).

**Phase 1.5 Option 1 only (NOT RECOMMENDED — has known bug):**

```bash
BLOOM_DOC_TOKENS_CACHE=1 \
  ./target/release/deps/spann-<hash> \
    --bench spann_bloom_ablation_010 \
    --warm-up-time 1 --measurement-time 1
```

Expected: bench panics with `[gate-audit] M=143` or similar — Option
1 alone leaks false negatives. See §10.5.

To soft-fail and inspect the regression:

```bash
BLOOM_DOC_TOKENS_CACHE=1 BLOOM_AUDIT_NO_PANIC=1 BLOOM_AUDIT_VERBOSE=1 \
  ./target/release/deps/spann-<hash> \
    --bench spann_bloom_ablation_010 \
    --warm-up-time 1 --measurement-time 1
```

This prints the first five `(head_id, doc_id)` pairs that triggered
the false negative.

### Required iso-I/O cell (not yet implemented)

The current `spann_bloom_ablation_010` cell only measures iso-probe.
**Before claiming the gate adds value, an iso-I/O cell must be added.**

Two implementation strategies:

1. **Fixed multiplier**: pick `probe_nbr_010 = probe_nbr_000 / (1 −
   expected_drop_ratio)` with a small safety margin. Run both, report
   `heads_fetched_after_gate` for each, and recall. The 010 row's
   `heads_fetched` should be ≈ the 000 row's.

2. **Adaptive search**: for each query, increase `probe_nbr_010`
   until `heads_after_bloom == probe_nbr_000`. This keeps I/O exact
   per-query at the cost of an extra HNSW probe loop. Slightly more
   complex but cleaner numbers.

Either strategy is acceptable. Strategy 1 is recommended for a first
implementation — it's a few lines of bench code and produces
interpretable averages.

Other cells worth adding eventually:

- **`spann_bloom_recall_curve`**: sweep `probe_nbr` for both 000 and
  010, plot recall vs `heads_fetched_after_gate`. The two curves
  should diverge — at the same x-coordinate (fetched PLs), the 010
  curve sits above the 000 curve.
- **8-cell ablation hypercube** (`{adaptive_nprobe, bloom, hms} ×
  {on, off}`) — the original phase plan target.
  `spann_bloom_ablation_010` is one corner.

### Strict correctness invariants the bench enforces

The bench's `[gate-audit]` block computes:

```rust
for h in &head_ids {
    if kept_set.contains(h) { continue; }
    total_drops += 1;
    let pl = spann_reader.fetch_posting_list(*h as u32).await...;
    if pl.iter().any(|p| allowed.contains(p.doc_offset_id)) {
        bad_drops += 1;
    }
}
if bad_drops > 0 && std::env::var("BLOOM_AUDIT_NO_PANIC").is_err() {
    panic!("Phase 1 contract violation: …");
}
```

`fetch_posting_list` already filters out stale entries via
`is_version_outdated`, so `bad_drops` counts only **live** matching
docs that the gate erroneously hid from the BfPL stage. **Any
positive `bad_drops` is a real correctness regression** — not a
"tunable" — and the bench refuses to continue.

If a future change pushes `bad_drops` above zero, that's the loudest
possible signal that something in the bloom write/clone/union/stale
machinery is broken. Don't paper over it.

---

## 13. Operational guarantees

- **Phase 1 MVP** (`head_bloom_enabled=true`, both Phase 1.5 flags
  `false`): zero false negatives, but on heavy-reassign workloads
  (`write_nprobe=32`, default `nreplica_count`), every head ends up
  stale and `drop_ratio = 0`. Correct but ineffective.
- **Phase 1.5 Option 1 alone**: ~1% false-negative leak. **Don't
  ship standalone.**
- **Phase 1.5 Option 2 alone**: zero false negatives, `drop_ratio`
  ~0.7 on the bench predicate, recall preserved. **Recommended.**
- **Both on**: zero false negatives, `drop_ratio` ~0.7 on the bench
  predicate. Belt-and-suspenders — Option 2 papers over Option 1's
  leak.

The gate is **strictly an optimization**. Disabling all bloom flags
reverts to the existing SPANN behavior with no semantic change. The
metadata segment's inverted index handles all filter correctness.
