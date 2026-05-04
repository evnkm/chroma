use chroma_storage::{PutOptions, Storage, StorageError};
use chroma_types::{
    BooleanOperator, MetadataComparison, MetadataSetValue, MetadataValue, PrimitiveOperator,
    SetOperator, Where,
};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

/// `(key, value_token)`. The value side carries a type tag so values of
/// different scalar types do not collide (e.g. `Int(1)` vs `Bool(true)`).
pub type SynopsisToken = (String, String);

/// Build the typed value-token half of a synopsis token. Returns None for
/// unsupported types (sparse vectors, arrays).
pub fn synopsis_value_token(value: &MetadataValue) -> Option<String> {
    match value {
        MetadataValue::Bool(v) => Some(format!("bool::{}", v)),
        MetadataValue::Int(v) => Some(format!("int::{}", v)),
        MetadataValue::Float(v) => Some(format!("float::{}", v.to_bits())),
        MetadataValue::Str(v) => Some(format!("str::{}", v)),
        MetadataValue::SparseVector(_)
        | MetadataValue::BoolArray(_)
        | MetadataValue::IntArray(_)
        | MetadataValue::FloatArray(_)
        | MetadataValue::StringArray(_) => None,
    }
}

/// Convenience: produce a `(key, value_token)` pair for a single field.
pub fn synopsis_token(key: &str, value: &MetadataValue) -> Option<SynopsisToken> {
    synopsis_value_token(value).map(|v| (key.to_string(), v))
}

/// Walk a doc's metadata and emit one `(key, value_token)` pair per supported
/// scalar field.
pub fn synopsis_doc_tokens(metadata: &HashMap<String, MetadataValue>) -> Vec<SynopsisToken> {
    let mut out = Vec::with_capacity(metadata.len());
    for (k, v) in metadata.iter() {
        if let Some(tok) = synopsis_value_token(v) {
            out.push((k.clone(), tok));
        }
    }
    out
}

fn set_value_synopsis_tokens(key: &str, value: &MetadataSetValue) -> Vec<SynopsisToken> {
    match value {
        MetadataSetValue::Bool(vs) => vs
            .iter()
            .map(|v| (key.to_string(), format!("bool::{}", v)))
            .collect(),
        MetadataSetValue::Int(vs) => vs
            .iter()
            .map(|v| (key.to_string(), format!("int::{}", v)))
            .collect(),
        MetadataSetValue::Float(vs) => vs
            .iter()
            .map(|v| (key.to_string(), format!("float::{}", v.to_bits())))
            .collect(),
        MetadataSetValue::Str(vs) => vs
            .iter()
            .map(|v| (key.to_string(), format!("str::{}", v)))
            .collect(),
    }
}

/// Predicate flattened to typed tokens. Anything we can't express via exact
/// equality counts becomes `Unsupported` and the gate degrades to a
/// pass-through (no information).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SynopsisPredicate {
    /// Every token must have a positive count in the cluster.
    And(Vec<SynopsisToken>),
    /// At least one token must have a positive count.
    Or(Vec<SynopsisToken>),
    /// Predicate cannot be reduced to exact-equality counts; gate is no-op.
    Unsupported,
}

impl SynopsisPredicate {
    pub fn is_gateable(&self) -> bool {
        match self {
            SynopsisPredicate::And(v) => !v.is_empty(),
            SynopsisPredicate::Or(v) => !v.is_empty(),
            SynopsisPredicate::Unsupported => false,
        }
    }
}

/// Convert a Where clause to a synopsis predicate. Conservative: anything
/// that needs range, NotEqual, NotIn, set-difference, or AND-of-OR
/// semantics returns `Unsupported`, so the gate becomes a no-op rather
/// than wrong.
pub fn extract_synopsis_predicate(where_clause: &Where) -> SynopsisPredicate {
    match where_clause {
        Where::Document(_) => SynopsisPredicate::Unsupported,
        Where::Metadata(expr) => match &expr.comparison {
            MetadataComparison::Primitive(PrimitiveOperator::Equal, value) => {
                match synopsis_token(&expr.key, value) {
                    Some(tok) => SynopsisPredicate::And(vec![tok]),
                    None => SynopsisPredicate::Unsupported,
                }
            }
            MetadataComparison::Set(SetOperator::In, set) => {
                let toks = set_value_synopsis_tokens(&expr.key, set);
                if toks.is_empty() {
                    SynopsisPredicate::Unsupported
                } else {
                    SynopsisPredicate::Or(toks)
                }
            }
            _ => SynopsisPredicate::Unsupported,
        },
        Where::Composite(comp) => match comp.operator {
            BooleanOperator::And => {
                let mut all = Vec::new();
                for child in &comp.children {
                    match extract_synopsis_predicate(child) {
                        SynopsisPredicate::And(toks) => all.extend(toks),
                        _ => return SynopsisPredicate::Unsupported,
                    }
                }
                if all.is_empty() {
                    SynopsisPredicate::Unsupported
                } else {
                    SynopsisPredicate::And(all)
                }
            }
            BooleanOperator::Or => {
                let mut all = Vec::new();
                for child in &comp.children {
                    match extract_synopsis_predicate(child) {
                        SynopsisPredicate::And(mut toks) if toks.len() == 1 => {
                            all.push(toks.pop().unwrap())
                        }
                        SynopsisPredicate::Or(toks) => all.extend(toks),
                        _ => return SynopsisPredicate::Unsupported,
                    }
                }
                if all.is_empty() {
                    SynopsisPredicate::Unsupported
                } else {
                    SynopsisPredicate::Or(all)
                }
            }
        },
    }
}

/// Per-cluster table of metadata value counts.
///
/// `head_size` is the number of live (non-deleted, current-version) docs
/// in the cluster. `counts[k][v]` is the number of those docs whose
/// `metadata[k] == v`. Keys that exceed the configured max-cardinality are
/// omitted entirely (their absence means the gate falls back to "keep").
/// For tracked keys, `other_counts[k]` is the number of docs whose value
/// for `k` is *not* among the per-key top-K — the gate treats those as
/// "may match" (no information, fall back to "keep").
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct HeadSynopsis {
    pub head_size: u64,
    /// key → value_token → count (top-K only).
    pub counts: HashMap<String, HashMap<String, u64>>,
    /// key → number of docs with a non-top-K value for this key.
    #[serde(default)]
    pub other_counts: HashMap<String, u64>,
}

impl HeadSynopsis {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns Some(0) if the cluster *provably* has zero docs with
    /// `key = value`, Some(n>0) if the count is known exactly, and None
    /// when the gate has no information (key not tracked, or value lives
    /// in the "other" bucket).
    pub fn count_for(&self, key: &str, value_token: &str) -> Option<u64> {
        let by_key = match self.counts.get(key) {
            Some(c) => c,
            None => return None, // key not tracked — no information
        };
        if let Some(c) = by_key.get(value_token) {
            return Some(*c);
        }
        // Not in top-K. If `other_counts[key]` is positive, the cluster
        // *may* contain a doc with this value — we cannot tell which
        // non-top-K value it is. Conservatively return None ("keep").
        // If the other-bucket is zero, then provably no doc with any
        // non-top-K value exists in this cluster, so this exact value
        // also has count 0.
        match self.other_counts.get(key).copied().unwrap_or(0) {
            0 => Some(0),
            _ => None,
        }
    }
}

/// Predicted yield of a cluster: head_size × selectivity. Used for ranking
/// kept clusters; never affects the keep/drop decision.
///
/// `And`: independence approximation across keys. Same-key conjunctions
///        with conflicting values short-circuit to 0.
/// `Or`:  cap-sum (exact when tokens share a key — a doc has at most one
///        value per key).
/// `Unsupported`: returns `head_size` (neutral — preserves input order
///        under sort-by-yield).
pub fn predicted_yield(synopsis: &HeadSynopsis, predicate: &SynopsisPredicate) -> f64 {
    if synopsis.head_size == 0 {
        return 0.0;
    }
    let n = synopsis.head_size as f64;
    match predicate {
        SynopsisPredicate::Unsupported => n,
        SynopsisPredicate::And(toks) => {
            // Group by key. Same-key AND of distinct values is provably 0.
            let mut by_key: HashMap<&String, &String> = HashMap::new();
            for (k, v) in toks {
                if let Some(prev_v) = by_key.get(k) {
                    if prev_v != &v {
                        return 0.0;
                    }
                } else {
                    by_key.insert(k, v);
                }
            }
            let mut acc = n;
            for (k, v) in by_key {
                let count = match synopsis.count_for(k, v) {
                    Some(c) => c as f64,
                    None => n, // unknown — neutral factor of 1.0
                };
                if count == 0.0 {
                    return 0.0;
                }
                acc *= count / n;
            }
            acc
        }
        SynopsisPredicate::Or(toks) => {
            let mut sum: u64 = 0;
            let mut had_unknown = false;
            for (k, v) in toks {
                match synopsis.count_for(k, v) {
                    Some(c) => sum = sum.saturating_add(c),
                    None => {
                        had_unknown = true;
                    }
                }
            }
            if had_unknown {
                // At least one token contributes unknown probability — fall
                // back to neutral. (Could be tighter but consistent with
                // the "no information" convention used elsewhere.)
                return n;
            }
            std::cmp::min(synopsis.head_size, sum) as f64
        }
    }
}

/// Segment-level synopsis table. Loaded from the persisted blob on segment
/// open; read-only thereafter under the recommended commit-time-rebuild
/// strategy.
#[derive(Clone, Default)]
pub struct HeadSynopsisCache {
    synopses: Arc<DashMap<u32, Arc<HeadSynopsis>>>,
}

impl HeadSynopsisCache {
    pub fn new() -> Self {
        Self {
            synopses: Arc::new(DashMap::new()),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.synopses.is_empty()
    }

    pub fn len(&self) -> usize {
        self.synopses.len()
    }

    pub fn insert(&self, head_id: u32, synopsis: HeadSynopsis) {
        self.synopses.insert(head_id, Arc::new(synopsis));
    }

    pub fn insert_arc(&self, head_id: u32, synopsis: Arc<HeadSynopsis>) {
        self.synopses.insert(head_id, synopsis);
    }

    pub fn get(&self, head_id: u32) -> Option<Arc<HeadSynopsis>> {
        self.synopses.get(&head_id).map(|r| r.clone())
    }

    pub fn iter(&self) -> Vec<(u32, Arc<HeadSynopsis>)> {
        self.synopses
            .iter()
            .map(|kv| (*kv.key(), kv.value().clone()))
            .collect()
    }
}

impl std::fmt::Debug for HeadSynopsisCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HeadSynopsisCache")
            .field("synopses", &self.synopses.len())
            .finish()
    }
}

/// Serialized form of the entire segment shard's synopsis table.
#[derive(Serialize, Deserialize, Default)]
pub struct HeadSynopsisBlob {
    pub synopses: HashMap<u32, HeadSynopsis>,
}

/// Configuration plumbed from the segment writer to the index writer.
pub struct HeadSynopsisWriteConfig<'a> {
    pub top_k_per_key: u32,
    pub max_cardinality: u32,
    pub existing_blob_path: Option<&'a str>,
}

pub struct HeadSynopsisReadConfig<'a> {
    pub blob_path: Option<&'a str>,
}

/// Serialized synopsis blob ready to be written to storage.
pub struct HeadSynopsisBlobFlusher {
    pub bytes: Vec<u8>,
    pub path: String,
    pub blob_id: Uuid,
    pub storage: Option<Arc<Storage>>,
}

impl HeadSynopsisBlobFlusher {
    pub async fn save(&self) -> Result<(), StorageError> {
        if let Some(storage) = &self.storage {
            storage
                .put_bytes(&self.path, self.bytes.clone(), PutOptions::default())
                .await?;
        }
        Ok(())
    }
}

/// Free function gate. Drops heads whose synopsis *provably* has zero
/// matches for the predicate. A `lookup` returning `None` keeps the head
/// (no information). An `Unsupported` predicate keeps everything.
pub fn gate_heads<I, F>(
    candidate_head_ids: I,
    predicate: &SynopsisPredicate,
    lookup: F,
) -> Vec<u32>
where
    I: IntoIterator<Item = u32>,
    F: Fn(u32) -> Option<Arc<HeadSynopsis>>,
{
    let pass_through = !predicate.is_gateable();
    let mut out = Vec::new();
    for hid in candidate_head_ids {
        if pass_through {
            out.push(hid);
            continue;
        }
        let synopsis = lookup(hid);
        let keep = match (&synopsis, predicate) {
            (None, _) => true,
            (Some(_), SynopsisPredicate::Unsupported) => true,
            (Some(s), SynopsisPredicate::And(toks)) => {
                // Drop iff any token is provably zero.
                let mut all_positive = true;
                for (k, v) in toks {
                    match s.count_for(k, v) {
                        Some(0) => {
                            all_positive = false;
                            break;
                        }
                        // Some(n>0) or None ("unknown") — both keep.
                        _ => {}
                    }
                }
                all_positive
            }
            (Some(s), SynopsisPredicate::Or(toks)) => {
                // Keep iff any token is unknown (no information) or has
                // a positive count. Drop iff every token is provably zero.
                let mut any_possible = false;
                for (k, v) in toks {
                    match s.count_for(k, v) {
                        Some(0) => {}
                        _ => {
                            any_possible = true;
                            break;
                        }
                    }
                }
                any_possible
            }
        };
        if keep {
            out.push(hid);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Build helpers — used by the writer at commit time to construct the full
// segment-shard synopsis table from per-doc structured tokens × posting lists.
// ---------------------------------------------------------------------------

/// Tally of one head's raw counts before top-K capping.
#[derive(Default)]
pub struct HeadRawCounts {
    pub head_size: u64,
    pub counts: HashMap<String, HashMap<String, u64>>,
}

/// Compress raw per-head counts into per-head synopses with **per-key
/// auto-promotion**: low-/medium-cardinality keys are tracked exactly,
/// high-cardinality keys fall back to top-K + other.
///
/// Two regimes per key, based on *global* distinct-value count:
///
/// 1. `distinct_values ≤ max_cardinality`: **track all values exactly**.
///    No `other_counts` pollution; the gate is exact for every queried
///    value of this key. This is the common case for typed metadata —
///    booleans, enums, low/medium-cardinality categoricals (tags,
///    languages, status codes, popular IDs).
///
/// 2. `distinct_values > max_cardinality`: **top-K + other-bucket**.
///    Storage-bounded; precision degrades for non-top-K queries (gate
///    falls back to "keep" via the `other_counts > 0` path). This is
///    the high-cardinality case (UUIDs, free-form text); the bloom
///    filter is the better fit here, but the synopsis still gates
///    cleanly when the queried value happens to be in top-K.
///
/// Why auto-promotion? The earlier policy (always cap at `top_k_per_key`,
/// even if the key has only slightly more distinct values than `top_k`)
/// caused near-universal "other-bucket pollution": with default
/// `top_k=64` and a workload of, say, 100 distinct values, ~99% of
/// heads ended up with `other_counts[key] > 0`, forcing the gate to
/// return "unknown" for every query and silently degrading to a no-op.
/// See `LOGS_PLANS/synopsis-recall-study.md` "Skepticism / caveats" for
/// the empirical write-up of that trap.
///
/// Under auto-promotion, `top_k_per_key` only kicks in for keys that
/// genuinely exceed `max_cardinality` — exactly the keys where
/// compression is needed.
pub fn build_head_synopses(
    raw: HashMap<u32, HeadRawCounts>,
    top_k_per_key: u32,
    max_cardinality: u32,
) -> HashMap<u32, HeadSynopsis> {
    // Step 1 — global popularity per (key, value).
    let mut global: HashMap<String, HashMap<String, u64>> = HashMap::new();
    for raw_head in raw.values() {
        for (k, by_v) in &raw_head.counts {
            let g_by_v = global.entry(k.clone()).or_default();
            for (v, c) in by_v {
                *g_by_v.entry(v.clone()).or_insert(0) += *c;
            }
        }
    }

    // Step 2 — pick allowed values per key (auto-promotion logic).
    let mut top_set: HashMap<String, HashMap<String, ()>> = HashMap::new();
    for (k, by_v) in &global {
        let allowed: HashMap<String, ()> = if (by_v.len() as u32) <= max_cardinality {
            // Regime 1: low/medium card — track all exactly.
            by_v.keys().cloned().map(|v| (v, ())).collect()
        } else {
            // Regime 2: high card — top-K + other.
            let mut entries: Vec<(&String, &u64)> = by_v.iter().collect();
            entries.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
            entries
                .into_iter()
                .take(top_k_per_key as usize)
                .map(|(v, _)| (v.clone(), ()))
                .collect()
        };
        top_set.insert(k.clone(), allowed);
    }

    // Step 3 — emit per-head synopses.
    let mut out: HashMap<u32, HeadSynopsis> = HashMap::new();
    for (head_id, raw_head) in raw {
        let mut counts: HashMap<String, HashMap<String, u64>> = HashMap::new();
        let mut other_counts: HashMap<String, u64> = HashMap::new();
        for (k, by_v) in raw_head.counts {
            let Some(allowed) = top_set.get(&k) else {
                continue;
            };
            let mut kept: HashMap<String, u64> = HashMap::new();
            let mut other: u64 = 0;
            for (v, c) in by_v {
                if allowed.contains_key(&v) {
                    kept.insert(v, c);
                } else {
                    other = other.saturating_add(c);
                }
            }
            if !kept.is_empty() {
                counts.insert(k.clone(), kept);
            }
            if other > 0 {
                other_counts.insert(k, other);
            }
        }
        out.insert(
            head_id,
            HeadSynopsis {
                head_size: raw_head.head_size,
                counts,
                other_counts,
            },
        );
    }
    out
}

/// Deprecated alias for backward compatibility — prefer `build_head_synopses`.
#[doc(hidden)]
pub fn apply_top_k_capping(
    raw: HashMap<u32, HeadRawCounts>,
    top_k_per_key: u32,
    max_cardinality: u32,
) -> HashMap<u32, HeadSynopsis> {
    build_head_synopses(raw, top_k_per_key, max_cardinality)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chroma_types::{
        BooleanOperator, CompositeExpression, ContainsOperator, MetadataComparison,
        MetadataExpression, MetadataSetValue, MetadataValue, PrimitiveOperator, SetOperator, Where,
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
    fn synopsis_token_format_is_typed() {
        let bool_t = synopsis_token("k", &MetadataValue::Bool(true)).unwrap();
        let int_t = synopsis_token("k", &MetadataValue::Int(1)).unwrap();
        let str_t = synopsis_token("k", &MetadataValue::Str("1".into())).unwrap();
        assert_eq!(bool_t.0, "k");
        assert_ne!(bool_t.1, int_t.1);
        assert_ne!(int_t.1, str_t.1);
        assert_ne!(bool_t.1, str_t.1);
    }

    #[test]
    fn arrays_and_sparse_yield_no_token() {
        assert!(synopsis_value_token(&MetadataValue::IntArray(vec![1])).is_none());
        assert!(synopsis_value_token(&MetadataValue::StringArray(vec!["a".into()])).is_none());
    }

    #[test]
    fn extract_single_equality() {
        let w = meta_eq("color", MetadataValue::Str("red".into()));
        match extract_synopsis_predicate(&w) {
            SynopsisPredicate::And(v) => {
                assert_eq!(v.len(), 1);
                assert_eq!(v[0].0, "color");
            }
            _ => panic!("expected And"),
        }
    }

    #[test]
    fn extract_in_yields_or() {
        let w = meta_in(
            "color",
            MetadataSetValue::Str(vec!["red".into(), "blue".into()]),
        );
        match extract_synopsis_predicate(&w) {
            SynopsisPredicate::Or(v) => {
                assert_eq!(v.len(), 2);
                assert!(v.iter().all(|t| t.0 == "color"));
            }
            _ => panic!("expected Or"),
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
        match extract_synopsis_predicate(&w) {
            SynopsisPredicate::And(v) => {
                assert_eq!(v.len(), 2);
                let keys: Vec<&str> = v.iter().map(|t| t.0.as_str()).collect();
                assert!(keys.contains(&"color"));
                assert!(keys.contains(&"size"));
            }
            _ => panic!("expected And"),
        }
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
            children: vec![inner_or, meta_eq("c", MetadataValue::Int(3))],
        });
        assert!(matches!(
            extract_synopsis_predicate(&outer_and),
            SynopsisPredicate::Unsupported
        ));
    }

    #[test]
    fn ranges_and_ne_are_unsupported() {
        let ne = Where::Metadata(MetadataExpression {
            key: "k".into(),
            comparison: MetadataComparison::Primitive(
                PrimitiveOperator::NotEqual,
                MetadataValue::Int(1),
            ),
        });
        assert!(matches!(
            extract_synopsis_predicate(&ne),
            SynopsisPredicate::Unsupported
        ));
        let gt = Where::Metadata(MetadataExpression {
            key: "k".into(),
            comparison: MetadataComparison::Primitive(
                PrimitiveOperator::GreaterThan,
                MetadataValue::Int(1),
            ),
        });
        assert!(matches!(
            extract_synopsis_predicate(&gt),
            SynopsisPredicate::Unsupported
        ));
        let arr_contains = Where::Metadata(MetadataExpression {
            key: "k".into(),
            comparison: MetadataComparison::ArrayContains(
                ContainsOperator::Contains,
                MetadataValue::Int(1),
            ),
        });
        assert!(matches!(
            extract_synopsis_predicate(&arr_contains),
            SynopsisPredicate::Unsupported
        ));
    }

    fn build_synopsis(
        head_size: u64,
        entries: &[(&str, &str, u64)],
        other: &[(&str, u64)],
    ) -> HeadSynopsis {
        let mut counts: HashMap<String, HashMap<String, u64>> = HashMap::new();
        for (k, v, c) in entries {
            counts
                .entry((*k).to_string())
                .or_default()
                .insert((*v).to_string(), *c);
        }
        let mut other_counts: HashMap<String, u64> = HashMap::new();
        for (k, c) in other {
            other_counts.insert((*k).to_string(), *c);
        }
        HeadSynopsis {
            head_size,
            counts,
            other_counts,
        }
    }

    #[test]
    fn count_for_returns_exact_when_in_top_k() {
        let s = build_synopsis(100, &[("bucket", "int::0", 7)], &[]);
        assert_eq!(s.count_for("bucket", "int::0"), Some(7));
    }

    #[test]
    fn count_for_returns_zero_when_other_bucket_empty() {
        // Key tracked, value not in top-K, no other-bucket → provably zero.
        let s = build_synopsis(100, &[("bucket", "int::0", 7)], &[]);
        assert_eq!(s.count_for("bucket", "int::99"), Some(0));
    }

    #[test]
    fn count_for_returns_none_when_value_in_other_bucket() {
        // Key tracked, value not in top-K, other-bucket > 0 → unknown.
        let s = build_synopsis(100, &[("bucket", "int::0", 7)], &[("bucket", 5)]);
        assert_eq!(s.count_for("bucket", "int::42"), None);
    }

    #[test]
    fn count_for_returns_none_when_key_not_tracked() {
        let s = build_synopsis(100, &[("bucket", "int::0", 7)], &[]);
        assert_eq!(s.count_for("color", "str::red"), None);
    }

    #[test]
    fn gate_drops_zero_count_heads_for_and() {
        let cache = HeadSynopsisCache::new();
        cache.insert(1, build_synopsis(50, &[("bucket", "int::0", 5)], &[]));
        cache.insert(2, build_synopsis(50, &[("bucket", "int::1", 5)], &[]));
        let p = SynopsisPredicate::And(vec![("bucket".to_string(), "int::0".to_string())]);
        let kept = gate_heads(vec![1u32, 2u32], &p, |h| cache.get(h));
        assert!(kept.contains(&1));
        assert!(!kept.contains(&2));
    }

    #[test]
    fn gate_keeps_heads_with_at_least_one_match_for_or() {
        let cache = HeadSynopsisCache::new();
        cache.insert(1, build_synopsis(50, &[("bucket", "int::0", 5)], &[]));
        cache.insert(2, build_synopsis(50, &[("bucket", "int::5", 5)], &[]));
        cache.insert(3, build_synopsis(50, &[("bucket", "int::9", 5)], &[]));
        let p = SynopsisPredicate::Or(vec![
            ("bucket".to_string(), "int::0".to_string()),
            ("bucket".to_string(), "int::5".to_string()),
        ]);
        let kept = gate_heads(vec![1u32, 2u32, 3u32], &p, |h| cache.get(h));
        assert!(kept.contains(&1));
        assert!(kept.contains(&2));
        assert!(!kept.contains(&3));
    }

    #[test]
    fn gate_falls_back_to_keep_on_unsupported() {
        let cache = HeadSynopsisCache::new();
        cache.insert(1, build_synopsis(50, &[("bucket", "int::0", 5)], &[]));
        let kept = gate_heads(vec![1u32, 2u32, 3u32], &SynopsisPredicate::Unsupported, |h| {
            cache.get(h)
        });
        // All three kept: pred is non-gateable.
        assert_eq!(kept, vec![1, 2, 3]);
    }

    #[test]
    fn gate_falls_back_to_keep_when_value_in_other_bucket() {
        // Predicate value not in top-K but other-bucket is positive ⇒ keep.
        let cache = HeadSynopsisCache::new();
        cache.insert(
            1,
            build_synopsis(50, &[("bucket", "int::0", 5)], &[("bucket", 3)]),
        );
        let p = SynopsisPredicate::And(vec![("bucket".to_string(), "int::42".to_string())]);
        let kept = gate_heads(vec![1u32], &p, |h| cache.get(h));
        assert_eq!(kept, vec![1]);
    }

    #[test]
    fn gate_keeps_heads_missing_synopsis_entry() {
        // No entry in cache for head 7 ⇒ lookup returns None ⇒ keep.
        let cache = HeadSynopsisCache::new();
        let p = SynopsisPredicate::And(vec![("bucket".to_string(), "int::0".to_string())]);
        let kept = gate_heads(vec![7u32, 8u32], &p, |h| cache.get(h));
        assert_eq!(kept, vec![7, 8]);
    }

    #[test]
    fn yield_monotonic_with_count() {
        let s_lo = build_synopsis(100, &[("bucket", "int::0", 1)], &[]);
        let s_hi = build_synopsis(100, &[("bucket", "int::0", 10)], &[]);
        let p = SynopsisPredicate::And(vec![("bucket".to_string(), "int::0".to_string())]);
        let y_lo = predicted_yield(&s_lo, &p);
        let y_hi = predicted_yield(&s_hi, &p);
        assert!(y_hi > y_lo);
    }

    #[test]
    fn yield_for_in_predicate_sums_within_key() {
        // Same-key OR: yield = count[a] + count[b], capped at head_size.
        let s = build_synopsis(
            100,
            &[("bucket", "int::0", 4), ("bucket", "int::1", 6)],
            &[],
        );
        let p = SynopsisPredicate::Or(vec![
            ("bucket".to_string(), "int::0".to_string()),
            ("bucket".to_string(), "int::1".to_string()),
        ]);
        assert_eq!(predicted_yield(&s, &p), 10.0);
    }

    #[test]
    fn yield_unsupported_returns_head_size() {
        let s = build_synopsis(42, &[("bucket", "int::0", 5)], &[]);
        assert_eq!(predicted_yield(&s, &SynopsisPredicate::Unsupported), 42.0);
    }

    #[test]
    fn yield_and_independence_across_keys() {
        // 1/4 docs have color=red, 1/2 docs have size=5; product ⇒ 1/8 of head.
        let s = build_synopsis(
            80,
            &[
                ("color", "str::red", 20),
                ("size", "int::5", 40),
            ],
            &[],
        );
        let p = SynopsisPredicate::And(vec![
            ("color".to_string(), "str::red".to_string()),
            ("size".to_string(), "int::5".to_string()),
        ]);
        // 80 * (20/80) * (40/80) = 80 * 0.25 * 0.5 = 10.
        assert!((predicted_yield(&s, &p) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn yield_and_same_key_conflict_short_circuits_to_zero() {
        let s = build_synopsis(
            80,
            &[("bucket", "int::0", 20), ("bucket", "int::1", 30)],
            &[],
        );
        let p = SynopsisPredicate::And(vec![
            ("bucket".to_string(), "int::0".to_string()),
            ("bucket".to_string(), "int::1".to_string()),
        ]);
        assert_eq!(predicted_yield(&s, &p), 0.0);
    }

    #[test]
    fn bincode_serialize_deserialize_preserves_counts() {
        let mut blob = HeadSynopsisBlob::default();
        blob.synopses.insert(
            1,
            build_synopsis(50, &[("bucket", "int::0", 5)], &[("bucket", 3)]),
        );
        let bytes = bincode::serialize(&blob).expect("serialize");
        let back: HeadSynopsisBlob = bincode::deserialize(&bytes).expect("deserialize");
        let s = back.synopses.get(&1).expect("head present");
        assert_eq!(s.head_size, 50);
        assert_eq!(s.counts["bucket"]["int::0"], 5);
        assert_eq!(s.other_counts["bucket"], 3);
    }

    #[test]
    fn auto_promotion_tracks_all_when_distinct_le_max_cardinality() {
        // Two heads with 5 distinct global values. `top_k=2`, `max_card=1024`:
        // since 5 ≤ max_card, auto-promotion keeps everything exactly even
        // though 5 > top_k. No `other_counts` pollution.
        let mut raw: HashMap<u32, HeadRawCounts> = HashMap::new();
        let mut h1 = HeadRawCounts { head_size: 10, counts: HashMap::new() };
        let mut h2 = HeadRawCounts { head_size: 10, counts: HashMap::new() };
        h1.counts
            .entry("bucket".to_string())
            .or_default()
            .extend(vec![
                ("int::0".to_string(), 4u64),
                ("int::1".to_string(), 3u64),
                ("int::2".to_string(), 2u64),
                ("int::4".to_string(), 1u64),
            ]);
        h2.counts
            .entry("bucket".to_string())
            .or_default()
            .extend(vec![
                ("int::0".to_string(), 3u64),
                ("int::1".to_string(), 2u64),
                ("int::3".to_string(), 2u64),
            ]);
        raw.insert(1, h1);
        raw.insert(2, h2);
        // top_k=2 would normally cap, but max_card=1024 >> 5 distinct ⇒ exact.
        let out = build_head_synopses(raw, 2, 1024);
        let s1 = out.get(&1).unwrap();
        assert_eq!(s1.counts["bucket"].get("int::0").copied(), Some(4));
        assert_eq!(s1.counts["bucket"].get("int::1").copied(), Some(3));
        assert_eq!(s1.counts["bucket"].get("int::2").copied(), Some(2));
        assert_eq!(s1.counts["bucket"].get("int::4").copied(), Some(1));
        // No "other" pollution under auto-promotion.
        assert!(!s1.other_counts.contains_key("bucket"));
        let s2 = out.get(&2).unwrap();
        assert_eq!(s2.counts["bucket"].get("int::0").copied(), Some(3));
        assert_eq!(s2.counts["bucket"].get("int::3").copied(), Some(2));
        assert!(!s2.other_counts.contains_key("bucket"));
    }

    #[test]
    fn high_cardinality_keys_compress_to_top_k_plus_other() {
        // Forces high-cardinality regime by setting max_cardinality < distinct.
        // 5 distinct values, max_card=2, top_k=2 ⇒ compress to top-2 + other.
        let mut raw: HashMap<u32, HeadRawCounts> = HashMap::new();
        let mut h1 = HeadRawCounts { head_size: 10, counts: HashMap::new() };
        let mut h2 = HeadRawCounts { head_size: 10, counts: HashMap::new() };
        // Globally: int::0 → 7 total (popular), int::1 → 5, int::2 → 2,
        //           int::3 → 2, int::4 → 1.
        h1.counts
            .entry("bucket".to_string())
            .or_default()
            .extend(vec![
                ("int::0".to_string(), 4u64),
                ("int::1".to_string(), 3u64),
                ("int::2".to_string(), 2u64),
                ("int::4".to_string(), 1u64),
            ]);
        h2.counts
            .entry("bucket".to_string())
            .or_default()
            .extend(vec![
                ("int::0".to_string(), 3u64),
                ("int::1".to_string(), 2u64),
                ("int::3".to_string(), 2u64),
            ]);
        raw.insert(1, h1);
        raw.insert(2, h2);
        let out = build_head_synopses(raw, 2, 2);
        let s1 = out.get(&1).unwrap();
        // Top-K kept: int::0, int::1 (most popular globally).
        assert_eq!(s1.counts["bucket"].get("int::0").copied(), Some(4));
        assert_eq!(s1.counts["bucket"].get("int::1").copied(), Some(3));
        assert!(!s1.counts["bucket"].contains_key("int::2"));
        // Rest rolled into other_counts: int::2 (2) + int::4 (1) = 3.
        assert_eq!(s1.other_counts.get("bucket").copied(), Some(3));
        let s2 = out.get(&2).unwrap();
        assert_eq!(s2.counts["bucket"].get("int::0").copied(), Some(3));
        assert_eq!(s2.counts["bucket"].get("int::1").copied(), Some(2));
        assert_eq!(s2.other_counts.get("bucket").copied(), Some(2));
    }

    #[test]
    fn auto_promotion_at_exact_threshold_still_tracks_all() {
        // distinct == max_cardinality (boundary): track all exactly.
        let mut raw: HashMap<u32, HeadRawCounts> = HashMap::new();
        let mut h1 = HeadRawCounts { head_size: 3, counts: HashMap::new() };
        h1.counts.entry("k".to_string()).or_default().extend(vec![
            ("v::a".to_string(), 1u64),
            ("v::b".to_string(), 1u64),
            ("v::c".to_string(), 1u64),
        ]);
        raw.insert(1, h1);
        // 3 distinct, max_card=3 ⇒ exact.
        let out = build_head_synopses(raw, 1, 3);
        let s = out.get(&1).unwrap();
        assert_eq!(s.counts["k"].len(), 3);
        assert!(s.other_counts.get("k").copied().unwrap_or(0) == 0);
    }

    #[test]
    fn high_card_keys_no_longer_skipped_entirely() {
        // 3 distinct, max_card=2 ⇒ compressed mode, NOT skipped.
        // Verifies the regime change: high-card keys retain a partial
        // synopsis (top-K + other) rather than being dropped entirely.
        let mut raw: HashMap<u32, HeadRawCounts> = HashMap::new();
        let mut h1 = HeadRawCounts { head_size: 5, counts: HashMap::new() };
        h1.counts.entry("huge".to_string()).or_default().extend(vec![
            ("str::a".to_string(), 1u64),
            ("str::b".to_string(), 1u64),
            ("str::c".to_string(), 3u64),
        ]);
        h1.counts
            .entry("small".to_string())
            .or_default()
            .insert("str::x".to_string(), 5u64);
        raw.insert(1, h1);
        let out = build_head_synopses(raw, 2, 2);
        let s1 = out.get(&1).unwrap();
        // "huge" present in top-K mode.
        assert!(s1.counts.contains_key("huge"));
        assert_eq!(s1.counts["huge"].len(), 2); // top-2
        // The 3rd value was compressed into other.
        assert!(s1.other_counts.get("huge").copied().unwrap_or(0) > 0);
        // "small" still tracked exactly.
        assert_eq!(s1.counts["small"].get("str::x").copied(), Some(5));
    }
}
