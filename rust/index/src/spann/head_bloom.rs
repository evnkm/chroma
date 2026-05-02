use chroma_storage::{PutOptions, Storage, StorageError};
use chroma_types::{
    BooleanOperator, MetadataComparison, MetadataSetValue, MetadataValue, PrimitiveOperator,
    SetOperator, Where,
};
use dashmap::{DashMap, DashSet};
use fastbloom::AtomicBloomFilter;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use uuid::Uuid;

pub const HEAD_BLOOM_SEED: u128 = 0xCAFE_F00D_BEEF_BABE_DEAD_C0DE_FACE_BEAD;
pub const TARGET_FPR: f64 = 0.001;

/// Build a typed token string for a (key, value) pair so that values of
/// different scalar types do not collide. Returns None for unsupported
/// types (sparse vectors, arrays).
pub fn metadata_token(key: &str, value: &MetadataValue) -> Option<String> {
    match value {
        MetadataValue::Bool(v) => Some(format!("meta::{}::bool::{}", key, v)),
        MetadataValue::Int(v) => Some(format!("meta::{}::int::{}", key, v)),
        MetadataValue::Float(v) => Some(format!("meta::{}::float::{}", key, v.to_bits())),
        MetadataValue::Str(v) => Some(format!("meta::{}::str::{}", key, v)),
        MetadataValue::SparseVector(_)
        | MetadataValue::BoolArray(_)
        | MetadataValue::IntArray(_)
        | MetadataValue::FloatArray(_)
        | MetadataValue::StringArray(_) => None,
    }
}

/// Walk a doc's metadata and emit one token per supported scalar field.
pub fn doc_tokens(metadata: &HashMap<String, MetadataValue>) -> Vec<String> {
    let mut out = Vec::with_capacity(metadata.len());
    for (k, v) in metadata.iter() {
        if let Some(tok) = metadata_token(k, v) {
            out.push(tok);
        }
    }
    out
}

/// Compact token build for set values (used when extracting `$in` predicates).
fn set_value_tokens(key: &str, value: &MetadataSetValue) -> Vec<String> {
    match value {
        MetadataSetValue::Bool(vs) => vs
            .iter()
            .map(|v| format!("meta::{}::bool::{}", key, v))
            .collect(),
        MetadataSetValue::Int(vs) => vs
            .iter()
            .map(|v| format!("meta::{}::int::{}", key, v))
            .collect(),
        MetadataSetValue::Float(vs) => vs
            .iter()
            .map(|v| format!("meta::{}::float::{}", key, v.to_bits()))
            .collect(),
        MetadataSetValue::Str(vs) => vs
            .iter()
            .map(|v| format!("meta::{}::str::{}", key, v))
            .collect(),
    }
}

/// Equality predicate flattened to flat tokens. Anything we can't gate on
/// becomes `Unsupported` and the gate degrades to a pass-through.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EqualityTokens {
    /// All listed tokens must be possibly-present (AND of equalities).
    And(Vec<String>),
    /// At least one must be possibly-present (single-key `$in` or top-level OR
    /// of single-token AND children).
    Or(Vec<String>),
    /// Predicate cannot be reduced to membership tests; gate is a no-op.
    Unsupported,
}

impl EqualityTokens {
    pub fn is_gateable(&self) -> bool {
        match self {
            EqualityTokens::And(v) => !v.is_empty(),
            EqualityTokens::Or(v) => !v.is_empty(),
            EqualityTokens::Unsupported => false,
        }
    }
}

/// Convert a Where clause to a flat token predicate. Conservative: anything
/// that needs range, NotEqual, NotIn, set-difference, or AND-of-OR semantics
/// returns `Unsupported`, so the gate becomes a no-op rather than wrong.
pub fn extract_equality_tokens(where_clause: &Where) -> EqualityTokens {
    match where_clause {
        Where::Document(_) => EqualityTokens::Unsupported,
        Where::Metadata(expr) => match &expr.comparison {
            MetadataComparison::Primitive(PrimitiveOperator::Equal, value) => {
                match metadata_token(&expr.key, value) {
                    Some(tok) => EqualityTokens::And(vec![tok]),
                    None => EqualityTokens::Unsupported,
                }
            }
            MetadataComparison::Set(SetOperator::In, set) => {
                let toks = set_value_tokens(&expr.key, set);
                if toks.is_empty() {
                    EqualityTokens::Unsupported
                } else {
                    EqualityTokens::Or(toks)
                }
            }
            _ => EqualityTokens::Unsupported,
        },
        Where::Composite(comp) => match comp.operator {
            BooleanOperator::And => {
                let mut all = Vec::new();
                for child in &comp.children {
                    match extract_equality_tokens(child) {
                        EqualityTokens::And(toks) => all.extend(toks),
                        _ => return EqualityTokens::Unsupported,
                    }
                }
                if all.is_empty() {
                    EqualityTokens::Unsupported
                } else {
                    EqualityTokens::And(all)
                }
            }
            BooleanOperator::Or => {
                let mut all = Vec::new();
                for child in &comp.children {
                    match extract_equality_tokens(child) {
                        EqualityTokens::And(mut toks) if toks.len() == 1 => {
                            all.push(toks.pop().unwrap())
                        }
                        EqualityTokens::Or(toks) => all.extend(toks),
                        _ => return EqualityTokens::Unsupported,
                    }
                }
                if all.is_empty() {
                    EqualityTokens::Unsupported
                } else {
                    EqualityTokens::Or(all)
                }
            }
        },
    }
}

/// One bloom filter for one centroid head.
pub struct HeadBloom {
    inner: AtomicBloomFilter,
    capacity: u64,
    live_count: AtomicU64,
}

impl HeadBloom {
    pub fn new(capacity: u64) -> Self {
        let cap = capacity.max(1);
        let inner = AtomicBloomFilter::with_false_pos(TARGET_FPR)
            .seed(&HEAD_BLOOM_SEED)
            .expected_items(cap as usize);
        Self {
            inner,
            capacity: cap,
            live_count: AtomicU64::new(0),
        }
    }

    pub fn insert(&self, token: &str) {
        self.inner.insert(token);
        self.live_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn insert_many<I, S>(&self, tokens: I)
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        for tok in tokens {
            self.insert(tok.as_ref());
        }
    }

    pub fn contains(&self, token: &str) -> bool {
        self.inner.contains(token)
    }

    pub fn live_count(&self) -> u64 {
        self.live_count.load(Ordering::Relaxed)
    }

    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    pub fn to_blob(&self) -> HeadBloomBlob {
        HeadBloomBlob {
            bits: self.inner.iter().collect(),
            num_hashes: self.inner.num_hashes(),
            capacity: self.capacity,
            live_count: self.live_count(),
        }
    }

    pub fn from_blob(blob: HeadBloomBlob) -> Self {
        let inner = AtomicBloomFilter::from_vec(blob.bits)
            .seed(&HEAD_BLOOM_SEED)
            .hashes(blob.num_hashes);
        Self {
            inner,
            capacity: blob.capacity,
            live_count: AtomicU64::new(blob.live_count),
        }
    }
}

impl Clone for HeadBloom {
    fn clone(&self) -> Self {
        // Round-trip via blob to deep-copy the bit vector.
        let blob = self.to_blob();
        HeadBloom::from_blob(blob)
    }
}

#[derive(Serialize, Deserialize)]
pub struct HeadBloomBlob {
    pub bits: Vec<u64>,
    pub num_hashes: u32,
    pub capacity: u64,
    pub live_count: u64,
}

/// Writer-side state. `filters` maps head_id → bloom filter. `stale` is a
/// monotonic set of head_ids whose filter has gaps (e.g. due to reassign
/// without a known token); the gate treats stale heads as "no filter", i.e.
/// keeps them. `doc_tokens` is the Phase 1.5 cache: doc_id → tokens.
#[derive(Clone)]
pub struct HeadBloomCache {
    filters: Arc<DashMap<u32, Arc<HeadBloom>>>,
    stale: Arc<DashSet<u32>>,
    doc_tokens: Option<Arc<DashMap<u32, Arc<Vec<String>>>>>,
    capacity: u64,
}

impl HeadBloomCache {
    pub fn new(capacity: u64) -> Self {
        Self {
            filters: Arc::new(DashMap::new()),
            stale: Arc::new(DashSet::new()),
            doc_tokens: None,
            capacity,
        }
    }

    pub fn new_with_doc_tokens(capacity: u64) -> Self {
        Self {
            filters: Arc::new(DashMap::new()),
            stale: Arc::new(DashSet::new()),
            doc_tokens: Some(Arc::new(DashMap::new())),
            capacity,
        }
    }

    pub fn capacity_per_head(&self) -> u64 {
        self.capacity
    }

    pub fn is_empty(&self) -> bool {
        self.filters.is_empty() && self.stale.is_empty()
    }

    pub fn len(&self) -> usize {
        self.filters.len()
    }

    fn get_or_create(&self, head_id: u32) -> Arc<HeadBloom> {
        if let Some(existing) = self.filters.get(&head_id) {
            return existing.clone();
        }
        let new_filter = Arc::new(HeadBloom::new(self.capacity));
        self.filters
            .entry(head_id)
            .or_insert(new_filter)
            .clone()
    }

    pub fn insert_tokens(&self, head_id: u32, tokens: &[String]) {
        if tokens.is_empty() {
            // Still get-or-create so the head has a filter (empty inserts are
            // a valid case for docs with no metadata; the gate keeps them).
            let _ = self.get_or_create(head_id);
            return;
        }
        let filter = self.get_or_create(head_id);
        filter.insert_many(tokens.iter().map(|s| s.as_str()));
    }

    pub fn insert_loaded(&self, head_id: u32, filter: HeadBloom) {
        self.filters.insert(head_id, Arc::new(filter));
    }

    /// Split: copy parent's filter to a new child. Critically, scope the
    /// DashMap `Ref<>` lookup so we don't hold a shard read lock while
    /// inserting (would deadlock if shards collide). Propagate stale.
    pub fn clone_into(&self, src_head_id: u32, dst_head_id: u32) {
        let src_was_stale = self.stale.contains(&src_head_id);
        let copy_opt = {
            self.filters.get(&src_head_id).map(|r| (**r).clone())
        };
        if let Some(copy) = copy_opt {
            self.filters.insert(dst_head_id, Arc::new(copy));
        }
        if src_was_stale {
            self.stale.insert(dst_head_id);
        }
    }

    /// Merge: dst absorbs src's bits. Falls back to dropping dst's filter
    /// (and marking stale) on capacity mismatch. Propagates stale.
    pub fn union_into(&self, src_head_id: u32, dst_head_id: u32) {
        let src_was_stale = self.stale.contains(&src_head_id);
        let src_clone_opt = {
            self.filters.get(&src_head_id).map(|r| (**r).clone())
        };
        let Some(src_clone) = src_clone_opt else {
            if src_was_stale {
                self.stale.insert(dst_head_id);
            }
            return;
        };
        let dst_exists = self.filters.contains_key(&dst_head_id);
        if !dst_exists {
            self.filters.insert(dst_head_id, Arc::new(src_clone));
            if src_was_stale {
                self.stale.insert(dst_head_id);
            }
            return;
        }
        // Capacity check: if shapes differ, drop dst's filter and mark stale.
        let shapes_match = {
            let dst = self.filters.get(&dst_head_id);
            let src_bits_len = src_clone.to_blob().bits.len();
            match dst {
                Some(d) => d.to_blob().bits.len() == src_bits_len,
                None => false,
            }
        };
        if !shapes_match {
            self.filters.remove(&dst_head_id);
            self.stale.insert(dst_head_id);
            return;
        }
        // Merge by reconstructing a fresh union filter from blobs.
        let merged = {
            let dst = self.filters.get(&dst_head_id).unwrap();
            let mut a = src_clone.to_blob();
            let b = dst.to_blob();
            for (i, w) in b.bits.iter().enumerate() {
                a.bits[i] |= *w;
            }
            a.live_count = a.live_count.saturating_add(b.live_count);
            HeadBloom::from_blob(a)
        };
        self.filters.insert(dst_head_id, Arc::new(merged));
        if src_was_stale {
            self.stale.insert(dst_head_id);
        }
    }

    pub fn remove(&self, head_id: u32) {
        self.filters.remove(&head_id);
        self.stale.remove(&head_id);
    }

    pub fn mark_stale(&self, head_id: u32) {
        self.stale.insert(head_id);
    }

    pub fn is_stale(&self, head_id: u32) -> bool {
        self.stale.contains(&head_id)
    }

    /// Returns None for stale heads — gate then keeps them.
    pub fn get(&self, head_id: u32) -> Option<Arc<HeadBloom>> {
        if self.stale.contains(&head_id) {
            return None;
        }
        self.filters.get(&head_id).map(|r| r.clone())
    }

    pub fn iter(&self) -> Vec<(u32, Arc<HeadBloom>)> {
        self.filters
            .iter()
            .map(|kv| (*kv.key(), kv.value().clone()))
            .collect()
    }

    /// Skips stale heads — used by commit.
    pub fn iter_non_stale(&self) -> Vec<(u32, Arc<HeadBloom>)> {
        self.filters
            .iter()
            .filter_map(|kv| {
                if self.stale.contains(kv.key()) {
                    None
                } else {
                    Some((*kv.key(), kv.value().clone()))
                }
            })
            .collect()
    }

    // ---- Phase 1.5 additions ---------------------------------------------

    pub fn seed_doc_tokens(&self, doc_id: u32, tokens: &[String]) {
        if let Some(map) = &self.doc_tokens {
            map.insert(doc_id, Arc::new(tokens.to_vec()));
        }
    }

    pub fn forget_doc(&self, doc_id: u32) {
        if let Some(map) = &self.doc_tokens {
            map.remove(&doc_id);
        }
    }

    pub fn get_doc_tokens(&self, doc_id: u32) -> Option<Arc<Vec<String>>> {
        self.doc_tokens.as_ref().and_then(|m| m.get(&doc_id).map(|r| r.clone()))
    }

    /// Reassign hook: caller signals a doc was appended to a head. Returns
    /// true on cache hit (we recorded the doc's tokens into the dest filter),
    /// false otherwise (caller should fall back to mark_stale).
    pub fn record_doc_appended_to_head(&self, head_id: u32, doc_id: u32) -> bool {
        let Some(map) = &self.doc_tokens else {
            return false;
        };
        let toks = match map.get(&doc_id) {
            Some(r) => r.clone(),
            None => return false,
        };
        let filter = self.get_or_create(head_id);
        filter.insert_many(toks.iter().map(|s| s.as_str()));
        true
    }

    pub fn replace_filter(&self, head_id: u32, fresh: HeadBloom) {
        self.filters.insert(head_id, Arc::new(fresh));
        self.stale.remove(&head_id);
    }

    pub fn touched_head_ids(&self) -> HashSet<u32> {
        let mut out: HashSet<u32> = self.filters.iter().map(|kv| *kv.key()).collect();
        for h in self.stale.iter() {
            out.insert(*h);
        }
        out
    }
}

impl std::fmt::Debug for HeadBloomCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HeadBloomCache")
            .field("filters", &self.filters.len())
            .field("stale", &self.stale.len())
            .field("doc_tokens", &self.doc_tokens.as_ref().map(|m| m.len()))
            .field("capacity", &self.capacity)
            .finish()
    }
}

/// Configuration for plumbing through the writer/reader boundary.
pub struct HeadBloomWriteConfig<'a> {
    pub capacity_per_head: u64,
    pub existing_blob_path: Option<&'a str>,
    pub doc_tokens_cache_enabled: bool,
    pub commit_rebuild_enabled: bool,
}

pub struct HeadBloomReadConfig<'a> {
    pub blob_path: Option<&'a str>,
}

/// Serialized head bloom blob ready to be written to storage.
pub struct HeadBloomBlobFlusher {
    pub bytes: Vec<u8>,
    pub path: String,
    pub blob_id: Uuid,
    pub storage: Option<Arc<Storage>>,
}

impl HeadBloomBlobFlusher {
    pub async fn save(&self) -> Result<(), StorageError> {
        if let Some(storage) = &self.storage {
            storage
                .put_bytes(&self.path, self.bytes.clone(), PutOptions::default())
                .await?;
        }
        Ok(())
    }
}

/// Free function gate. Generic over candidate iterator and lookup closure.
/// Lookup returns `None` to mean "no filter" — the gate keeps such heads
/// (correctness: stale or missing → never drop).
pub fn gate_heads<I, F>(candidate_head_ids: I, tokens: &EqualityTokens, lookup: F) -> Vec<u32>
where
    I: IntoIterator<Item = u32>,
    F: Fn(u32) -> Option<Arc<HeadBloom>>,
{
    let pass_through = !tokens.is_gateable();
    let mut out = Vec::new();
    for hid in candidate_head_ids {
        if pass_through {
            out.push(hid);
            continue;
        }
        let bloom = lookup(hid);
        let keep = match (&bloom, tokens) {
            (None, _) => true,
            (Some(_), EqualityTokens::Unsupported) => true,
            (Some(b), EqualityTokens::And(toks)) => {
                toks.iter().all(|t| b.contains(t))
            }
            (Some(b), EqualityTokens::Or(toks)) => {
                toks.iter().any(|t| b.contains(t))
            }
        };
        if keep {
            out.push(hid);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use chroma_types::{
        BooleanOperator, CompositeExpression, ContainsOperator, MetadataComparison, MetadataExpression,
        MetadataSetValue, MetadataValue, PrimitiveOperator, SetOperator, Where,
    };

    fn meta_eq(key: &str, value: MetadataValue) -> Where {
        Where::Metadata(MetadataExpression {
            key: key.to_string(),
            comparison: MetadataComparison::Primitive(PrimitiveOperator::Equal, value),
        })
    }

    fn meta_in(key: &str, values: MetadataSetValue) -> Where {
        Where::Metadata(MetadataExpression {
            key: key.to_string(),
            comparison: MetadataComparison::Set(SetOperator::In, values),
        })
    }

    #[test]
    fn token_format_is_typed() {
        let bool_t = metadata_token("k", &MetadataValue::Bool(true)).unwrap();
        let int_t = metadata_token("k", &MetadataValue::Int(1)).unwrap();
        let str_t = metadata_token("k", &MetadataValue::Str("1".into())).unwrap();
        assert_ne!(bool_t, int_t);
        assert_ne!(int_t, str_t);
        assert_ne!(bool_t, str_t);
    }

    #[test]
    fn arrays_and_sparse_yield_no_token() {
        assert!(metadata_token("k", &MetadataValue::IntArray(vec![1])).is_none());
        assert!(metadata_token("k", &MetadataValue::StringArray(vec!["a".into()])).is_none());
    }

    #[test]
    fn extract_single_equality() {
        let w = meta_eq("color", MetadataValue::Str("red".into()));
        let toks = extract_equality_tokens(&w);
        match toks {
            EqualityTokens::And(v) => assert_eq!(v.len(), 1),
            _ => panic!("expected And"),
        }
    }

    #[test]
    fn extract_and_of_equalities() {
        let w = Where::Composite(CompositeExpression {
            operator: BooleanOperator::And,
            children: vec![
                meta_eq("color", MetadataValue::Str("red".into())),
                meta_eq("size", MetadataValue::Int(5)),
            ],
        });
        match extract_equality_tokens(&w) {
            EqualityTokens::And(v) => assert_eq!(v.len(), 2),
            _ => panic!("expected And"),
        }
    }

    #[test]
    fn extract_in_yields_or() {
        let w = meta_in("color", MetadataSetValue::Str(vec!["red".into(), "blue".into()]));
        match extract_equality_tokens(&w) {
            EqualityTokens::Or(v) => assert_eq!(v.len(), 2),
            _ => panic!("expected Or"),
        }
    }

    #[test]
    fn ne_and_ranges_are_unsupported() {
        let ne = Where::Metadata(MetadataExpression {
            key: "k".into(),
            comparison: MetadataComparison::Primitive(
                PrimitiveOperator::NotEqual,
                MetadataValue::Int(1),
            ),
        });
        assert!(matches!(extract_equality_tokens(&ne), EqualityTokens::Unsupported));
        let gt = Where::Metadata(MetadataExpression {
            key: "k".into(),
            comparison: MetadataComparison::Primitive(
                PrimitiveOperator::GreaterThan,
                MetadataValue::Int(1),
            ),
        });
        assert!(matches!(extract_equality_tokens(&gt), EqualityTokens::Unsupported));
        let arr_contains = Where::Metadata(MetadataExpression {
            key: "k".into(),
            comparison: MetadataComparison::ArrayContains(
                ContainsOperator::Contains,
                MetadataValue::Int(1),
            ),
        });
        assert!(matches!(extract_equality_tokens(&arr_contains), EqualityTokens::Unsupported));
    }

    #[test]
    fn nested_or_inside_and_is_unsupported() {
        let inner_or = Where::Composite(CompositeExpression {
            operator: BooleanOperator::Or,
            children: vec![
                meta_eq("a", MetadataValue::Int(1)),
                meta_eq("b", MetadataValue::Int(2)),
            ],
        });
        let outer_and = Where::Composite(CompositeExpression {
            operator: BooleanOperator::And,
            children: vec![
                inner_or,
                meta_eq("c", MetadataValue::Int(3)),
            ],
        });
        assert!(matches!(
            extract_equality_tokens(&outer_and),
            EqualityTokens::Unsupported
        ));
    }

    #[test]
    fn bloom_no_false_negative() {
        let b = HeadBloom::new(64);
        b.insert("hello");
        b.insert("world");
        assert!(b.contains("hello"));
        assert!(b.contains("world"));
        // Token never inserted; with capacity=64 and FPR=0.001 the FP odds
        // are tiny but not zero. Use a token unlikely to collide and assert
        // structure: insertion count tracked.
        assert_eq!(b.live_count(), 2);
    }

    #[test]
    fn bloom_roundtrip() {
        let b = HeadBloom::new(64);
        b.insert("hello");
        b.insert("world");
        let blob = b.to_blob();
        let bytes = bincode::serialize(&blob).unwrap();
        let blob2: HeadBloomBlob = bincode::deserialize(&bytes).unwrap();
        let restored = HeadBloom::from_blob(blob2);
        assert!(restored.contains("hello"));
        assert!(restored.contains("world"));
    }

    #[test]
    fn cache_clone_preserves_membership() {
        let cache = HeadBloomCache::new(64);
        cache.insert_tokens(1, &vec!["alpha".to_string()]);
        cache.clone_into(1, 2);
        let f2 = cache.get(2).unwrap();
        assert!(f2.contains("alpha"));
    }

    #[test]
    fn cache_union_merges() {
        let cache = HeadBloomCache::new(64);
        cache.insert_tokens(1, &vec!["alpha".to_string()]);
        cache.insert_tokens(2, &vec!["beta".to_string()]);
        cache.union_into(1, 2);
        let f2 = cache.get(2).unwrap();
        assert!(f2.contains("alpha"));
        assert!(f2.contains("beta"));
    }

    #[test]
    fn gate_drops_definite_misses() {
        let cache = HeadBloomCache::new(64);
        cache.insert_tokens(1, &vec!["meta::k::int::1".to_string()]);
        cache.insert_tokens(2, &vec!["meta::k::int::2".to_string()]);
        let toks = EqualityTokens::And(vec!["meta::k::int::1".to_string()]);
        let kept = gate_heads(vec![1u32, 2u32], &toks, |h| cache.get(h));
        assert!(kept.contains(&1));
        // Filter for head 2 contains a different token — gate should drop.
        // (Not strictly guaranteed: false positive possible but very unlikely
        // for a single token at FPR=0.001.)
        assert!(!kept.contains(&2));
    }

    #[test]
    fn doc_tokens_cache_record_appended_inserts_token() {
        let cache = HeadBloomCache::new_with_doc_tokens(64);
        cache.seed_doc_tokens(7, &vec!["meta::k::int::5".to_string()]);
        let used = cache.record_doc_appended_to_head(11, 7);
        assert!(used);
        let f = cache.get(11).unwrap();
        assert!(f.contains("meta::k::int::5"));
    }

    #[test]
    fn doc_tokens_cache_miss_returns_false() {
        let cache = HeadBloomCache::new_with_doc_tokens(64);
        let used = cache.record_doc_appended_to_head(11, 999);
        assert!(!used);
    }

    #[test]
    fn doc_tokens_cache_forget_drops_entries() {
        let cache = HeadBloomCache::new_with_doc_tokens(64);
        cache.seed_doc_tokens(7, &vec!["t".to_string()]);
        cache.forget_doc(7);
        assert!(cache.get_doc_tokens(7).is_none());
    }

    #[test]
    fn mvp_cache_ignores_doc_tokens_calls() {
        let cache = HeadBloomCache::new(64);
        cache.seed_doc_tokens(7, &vec!["t".to_string()]);
        assert!(cache.get_doc_tokens(7).is_none());
        let used = cache.record_doc_appended_to_head(1, 7);
        assert!(!used);
    }

    #[test]
    fn replace_filter_clears_stale() {
        let cache = HeadBloomCache::new(64);
        cache.mark_stale(5);
        assert!(cache.is_stale(5));
        cache.replace_filter(5, HeadBloom::new(64));
        assert!(!cache.is_stale(5));
    }

    #[test]
    fn touched_head_ids_union() {
        let cache = HeadBloomCache::new(64);
        cache.insert_tokens(1, &vec!["t".to_string()]);
        cache.mark_stale(2);
        let ids = cache.touched_head_ids();
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
    }

    #[test]
    fn clone_into_propagates_stale() {
        let cache = HeadBloomCache::new(64);
        cache.insert_tokens(1, &vec!["t".to_string()]);
        cache.mark_stale(1);
        cache.clone_into(1, 2);
        assert!(cache.is_stale(2));
    }

    #[test]
    fn union_into_propagates_stale() {
        let cache = HeadBloomCache::new(64);
        cache.insert_tokens(1, &vec!["t1".to_string()]);
        cache.insert_tokens(2, &vec!["t2".to_string()]);
        cache.mark_stale(1);
        cache.union_into(1, 2);
        assert!(cache.is_stale(2));
    }

    #[test]
    fn iter_non_stale_skips() {
        let cache = HeadBloomCache::new(64);
        cache.insert_tokens(1, &vec!["t".to_string()]);
        cache.insert_tokens(2, &vec!["t".to_string()]);
        cache.mark_stale(2);
        let kept: Vec<u32> = cache.iter_non_stale().into_iter().map(|(k, _)| k).collect();
        assert!(kept.contains(&1));
        assert!(!kept.contains(&2));
    }
}
