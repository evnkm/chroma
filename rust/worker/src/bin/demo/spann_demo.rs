// Phase 3 of the SPANN race demo. Loads two persistent SPANN indexes
// (baseline = cell 000, optimized = cell 111), spawns a Python embedding
// sidecar, and accepts JSON-line requests on stdin. Each request races the
// two indexes and emits stage events on stdout.
//
// Stdin (one JSON object per line):
//   {"query_id": 0|1|...|"text": "free text", "predicate": {"key":"...", "value":"..."}, "k": 10}
// At least one of `query_id` (selecting a cached query) or `query_text` must
// be present. `k` defaults to 5.
//
// Stdout (one JSON object per line):
//   {"event":"ready","baseline_n":..., "optimized_n":...,"queries":[...]}
//   {"event":"embedded","t_ms":..,"dim":..}
//   {"event":"ground_truth","n_match":...,"selectivity":...,"t_ms":...}
//   {"cell":"baseline"|"optimized","stage":"probing"|"gating"|"fetching"|"scoring"|"done", ... fields per stage }
//   {"event":"error","message":"..."}
//
// Build: cargo build --release -p worker --bin spann_demo
// Run:   ./target/release/spann_demo \
//          --baseline demo/data/baseline \
//          --optimized demo/data/optimized \
//          --queries demo/data/queries.json \
//          --cache demo/data/msmarco_100k_rust \
//          --embed-sidecar demo/embed_sidecar.py

use std::{
    collections::HashSet,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::Arc,
    time::Instant,
};

use chroma_benchmark::datasets::msmarco_demo::MsMarcoDemoData;
use chroma_blockstore::{
    arrow::{config::BlockManagerConfig, provider::ArrowBlockfileProvider},
    provider::BlockfileProvider,
};
use chroma_cache::{new_cache_for_test, new_non_persistent_cache_for_test};
use chroma_distance::DistanceFunction;
use chroma_index::{
    hnsw_provider::HnswIndexProvider,
    spann::{
        head_bloom::{EqualityTokens, HeadBloomReadConfig},
        head_synopsis::{HeadSynopsisReadConfig, SynopsisPredicate},
        types::SpannIndexReader,
        utils::rng_query,
    },
    IndexUuid,
};
use chroma_storage::{local::LocalStorage, Storage};
use chroma_system::Operator;
use chroma_types::{
    operator::Merge, CollectionUuid, InternalSpannConfiguration, SignedRoaringBitmap,
};
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;
use worker::execution::operators::{
    knn_merge::KnnMergeInput,
    spann_bf_pl::{SpannBfPlInput, SpannBfPlOperator},
};

// ---------- CLI / config ----------

#[derive(Debug, Clone)]
struct Config {
    baseline_dir: PathBuf,
    optimized_dir: PathBuf,
    queries_path: PathBuf,
    cache_dir: PathBuf,
    embed_sidecar: PathBuf,
    python: String,
    use_embed_sidecar: bool,
}

impl Config {
    fn from_args() -> Self {
        let mut args = std::env::args().skip(1);
        let mut cfg = Config {
            baseline_dir: "demo/data/baseline".into(),
            optimized_dir: "demo/data/optimized".into(),
            queries_path: "demo/data/queries.json".into(),
            cache_dir: "demo/data/msmarco_100k_rust".into(),
            embed_sidecar: "demo/embed_sidecar.py".into(),
            python: ".venv/bin/python".to_string(),
            use_embed_sidecar: true,
        };
        while let Some(a) = args.next() {
            match a.as_str() {
                "--baseline" => cfg.baseline_dir = args.next().unwrap().into(),
                "--optimized" => cfg.optimized_dir = args.next().unwrap().into(),
                "--queries" => cfg.queries_path = args.next().unwrap().into(),
                "--cache" => cfg.cache_dir = args.next().unwrap().into(),
                "--embed-sidecar" => cfg.embed_sidecar = args.next().unwrap().into(),
                "--python" => cfg.python = args.next().unwrap(),
                "--no-embed-sidecar" => cfg.use_embed_sidecar = false,
                "-h" | "--help" => {
                    eprintln!("usage: spann_demo [--baseline DIR] [--optimized DIR] [--queries PATH] [--cache DIR] [--embed-sidecar PATH] [--python PATH] [--no-embed-sidecar]");
                    std::process::exit(0);
                }
                _ => {
                    eprintln!("[spann_demo] unrecognised arg {a}");
                    std::process::exit(2);
                }
            }
        }
        cfg
    }
}

// ---------- Manifest ----------

#[derive(Debug, Deserialize)]
struct Manifest {
    flavor: String,
    dim: usize,
    n_records: usize,
    collection_id: String,
    prefix_path: String,
    pl_id: String,
    versions_map_id: String,
    #[allow(dead_code)]
    max_head_id_id: String,
    hnsw_id: String,
    head_bloom_blob_path: Option<String>,
    head_synopsis_blob_path: Option<String>,
    #[allow(dead_code)]
    build_ms: f64,
    #[allow(dead_code)]
    blob_bytes: u64,
    params: ManifestParams,
    adaptive_nprobe: bool,
    bloom_enabled: bool,
    synopsis_enabled: bool,
}

#[derive(Debug, Deserialize)]
struct ManifestParams {
    search_nprobe: u32,
    split_threshold: u32,
    ef_search: usize,
    search_rng_epsilon: f32,
    search_rng_factor: f32,
}

fn load_manifest(dir: &Path) -> anyhow::Result<Manifest> {
    let data = std::fs::read(dir.join("manifest.json"))?;
    let m: Manifest = serde_json::from_slice(&data)?;
    Ok(m)
}

#[derive(Debug, Deserialize)]
struct CachedQuery {
    id: usize,
    text: String,
    query_type: String,
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct QueriesFile {
    #[allow(dead_code)]
    dim: usize,
    #[allow(dead_code)]
    n_buckets: i32,
    queries: Vec<CachedQuery>,
}

fn load_queries(path: &Path) -> anyhow::Result<QueriesFile> {
    let data = std::fs::read(path)?;
    let q: QueriesFile = serde_json::from_slice(&data)?;
    Ok(q)
}

// ---------- Live index handle ----------

#[allow(dead_code)]
struct LiveIndex {
    flavor: String,
    n_records: usize,
    blockfile_provider: BlockfileProvider,
    hnsw_provider: HnswIndexProvider,
    pl_id: Uuid,
    versions_map_id: Uuid,
    hnsw_id: IndexUuid,
    collection_id: CollectionUuid,
    prefix_path: String,
    head_bloom_blob_path: Option<String>,
    head_synopsis_blob_path: Option<String>,
    params: InternalSpannConfiguration,
    adaptive: bool,
    bloom_enabled: bool,
    synopsis_enabled: bool,
    dim: usize,
}

impl LiveIndex {
    fn from_dir(dir: &Path) -> anyhow::Result<Self> {
        let m = load_manifest(dir)?;
        let storage = Storage::Local(LocalStorage::new(
            dir.join("storage").to_str().unwrap(),
        ));
        let block_cache = new_cache_for_test();
        let sparse_index_cache = new_cache_for_test();
        let arrow_blockfile_provider = ArrowBlockfileProvider::new(
            storage.clone(),
            8_388_608,
            block_cache,
            sparse_index_cache,
            BlockManagerConfig::default_num_concurrent_block_flushes(),
            BlockManagerConfig::default_max_concurrent_block_loads(),
        );
        let blockfile_provider =
            BlockfileProvider::ArrowBlockfileProvider(arrow_blockfile_provider);
        let hnsw_cache = new_non_persistent_cache_for_test();
        let hnsw_provider = HnswIndexProvider::new(storage.clone(), hnsw_cache, 16);

        let pl_id = Uuid::parse_str(&m.pl_id)?;
        let versions_map_id = Uuid::parse_str(&m.versions_map_id)?;
        let hnsw_id = IndexUuid(Uuid::parse_str(&m.hnsw_id)?);
        let collection_id = CollectionUuid(Uuid::parse_str(&m.collection_id)?);

        let mut params = InternalSpannConfiguration::default();
        params.search_nprobe = m.params.search_nprobe;
        params.split_threshold = m.params.split_threshold;
        params.ef_search = m.params.ef_search;
        params.search_rng_epsilon = m.params.search_rng_epsilon;
        params.search_rng_factor = m.params.search_rng_factor;

        Ok(Self {
            flavor: m.flavor.clone(),
            n_records: m.n_records,
            blockfile_provider,
            hnsw_provider,
            pl_id,
            versions_map_id,
            hnsw_id,
            collection_id,
            prefix_path: m.prefix_path.clone(),
            head_bloom_blob_path: m.head_bloom_blob_path.clone(),
            head_synopsis_blob_path: m.head_synopsis_blob_path.clone(),
            params,
            adaptive: m.adaptive_nprobe,
            bloom_enabled: m.bloom_enabled,
            synopsis_enabled: m.synopsis_enabled,
            dim: m.dim,
        })
    }

    async fn open_reader(&self) -> anyhow::Result<SpannIndexReader<'_>> {
        let head_bloom_read = self
            .head_bloom_blob_path
            .as_deref()
            .map(|p| HeadBloomReadConfig { blob_path: Some(p) });
        let head_synopsis_read = self
            .head_synopsis_blob_path
            .as_deref()
            .map(|p| HeadSynopsisReadConfig { blob_path: Some(p) });
        let reader = Box::pin(SpannIndexReader::from_id(
            Some(&self.hnsw_id),
            &self.hnsw_provider,
            &self.collection_id,
            self.params.clone().space.into(),
            self.dim,
            self.params.ef_search,
            Some(&self.pl_id),
            Some(&self.versions_map_id),
            &self.blockfile_provider,
            &self.prefix_path,
            self.adaptive,
            self.params.clone(),
            head_bloom_read,
            head_synopsis_read,
        ))
        .await
        .map_err(|e| anyhow::anyhow!("SpannIndexReader::from_id: {e}"))?;
        Ok(reader)
    }
}

// ---------- Predicate parsing & helpers ----------

#[derive(Debug, Clone)]
struct ParsedPredicate {
    /// Bloom token form, e.g. "meta::source_domain::str::wikihow.com".
    bloom_token: String,
    /// Synopsis predicate form: ("source_domain", "str::wikihow.com").
    synopsis_key: String,
    synopsis_value: String,
    /// Display form (for events).
    display: String,
    /// Match function over the in-memory metadata.
    matcher: PredicateMatcher,
}

#[derive(Debug, Clone)]
enum PredicateMatcher {
    SourceDomain(String),
    LengthBucket(String),
    QueryType(String),
    TopicBucket(i32),
}

impl PredicateMatcher {
    fn matches(&self, i: usize, data: &MsMarcoDemoData) -> bool {
        match self {
            PredicateMatcher::SourceDomain(s) => &data.meta.source_domains[i] == s,
            PredicateMatcher::LengthBucket(s) => &data.meta.length_buckets[i] == s,
            PredicateMatcher::QueryType(s) => &data.meta.query_types[i] == s,
            PredicateMatcher::TopicBucket(b) => data.topic_buckets[i] == *b,
        }
    }
}

fn parse_predicate(key: &str, value: &str) -> anyhow::Result<ParsedPredicate> {
    let display = format!("{key} = {value:?}");
    Ok(match key {
        "source_domain" => ParsedPredicate {
            bloom_token: format!("meta::source_domain::str::{value}"),
            synopsis_key: "source_domain".to_string(),
            synopsis_value: format!("str::{value}"),
            display,
            matcher: PredicateMatcher::SourceDomain(value.to_string()),
        },
        "length_bucket" => ParsedPredicate {
            bloom_token: format!("meta::length_bucket::str::{value}"),
            synopsis_key: "length_bucket".to_string(),
            synopsis_value: format!("str::{value}"),
            display,
            matcher: PredicateMatcher::LengthBucket(value.to_string()),
        },
        "query_type" => ParsedPredicate {
            bloom_token: format!("meta::query_type::str::{value}"),
            synopsis_key: "query_type".to_string(),
            synopsis_value: format!("str::{value}"),
            display,
            matcher: PredicateMatcher::QueryType(value.to_string()),
        },
        "topic_bucket" => {
            let v: i32 = value.parse()?;
            ParsedPredicate {
                bloom_token: format!("meta::topic_bucket::int::{v}"),
                synopsis_key: "topic_bucket".to_string(),
                synopsis_value: format!("int::{v}"),
                display,
                matcher: PredicateMatcher::TopicBucket(v),
            }
        }
        _ => anyhow::bail!("unsupported predicate key: {key}"),
    })
}

// ---------- Race ----------

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "stage")]
enum StageEvent {
    #[serde(rename = "probing")]
    Probing { cell: String, t_ms: f64 },
    #[serde(rename = "probed")]
    Probed { cell: String, heads_rng: usize, t_ms: f64, nprobe_used: u32 },
    #[serde(rename = "gating")]
    Gating { cell: String, gate: String, t_ms: f64 },
    #[serde(rename = "gated")]
    Gated { cell: String, gate: String, kept: usize, dropped: usize, t_ms: f64 },
    #[serde(rename = "fetching")]
    Fetching { cell: String, heads: usize, t_ms: f64 },
    #[serde(rename = "fetched")]
    Fetched { cell: String, candidates: usize, t_ms: f64 },
    #[serde(rename = "scoring")]
    Scoring { cell: String, candidates: usize, t_ms: f64 },
    #[serde(rename = "done")]
    Done {
        cell: String,
        lat_ms: f64,
        recall_at_k: f64,
        returned: usize,
        nprobe_used: u32,
        heads_rng: usize,
        heads_fetched: usize,
        candidates_before: usize,
        candidates_after: usize,
        top_k: Vec<TopKItem>,
    },
}

#[derive(Debug, Clone, Serialize)]
struct TopKItem {
    id: u32,
    score: f64,
    text: String,
    source_domain: String,
    topic_bucket: i32,
    length_bucket: String,
    query_type: String,
}

fn write_stdout_line(line: String) {
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(line.as_bytes()).ok();
    stdout.write_all(b"\n").ok();
    stdout.flush().ok();
}

fn emit_event(v: &impl Serialize) {
    if let Ok(s) = serde_json::to_string(v) {
        write_stdout_line(s);
    }
}

// ---------- Main loop ----------

fn main() -> anyhow::Result<()> {
    let cfg = Config::from_args();

    eprintln!("[spann_demo] loading cache ...");
    let data = MsMarcoDemoData::load(&cfg.cache_dir)?;
    eprintln!(
        "[spann_demo] cache: n={} dim={} q={} buckets={}",
        data.n(),
        data.dim,
        data.q(),
        data.n_buckets
    );

    eprintln!("[spann_demo] opening baseline manifest ...");
    let baseline = Arc::new(LiveIndex::from_dir(&cfg.baseline_dir)?);
    eprintln!("[spann_demo] opening optimized manifest ...");
    let optimized = Arc::new(LiveIndex::from_dir(&cfg.optimized_dir)?);

    let queries = load_queries(&cfg.queries_path)?;
    eprintln!("[spann_demo] loaded {} cached queries", queries.queries.len());

    // Spawn embedding sidecar.
    let mut sidecar = if cfg.use_embed_sidecar {
        eprintln!("[spann_demo] spawning embed sidecar: {} {}", cfg.python, cfg.embed_sidecar.display());
        Some(EmbedSidecarBuilt::spawn(&cfg.python, &cfg.embed_sidecar)?)
    } else {
        None
    };

    // Build tokio runtime for serving requests.
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    // Open both readers up front so subsequent queries are fast. Then
    // lifetime-erase to 'static so we can share via Arc into spawned tasks.
    // SAFETY: each reader borrows from the corresponding `LiveIndex`'s
    // BlockfileProvider + HnswIndexProvider. Both `LiveIndex`es are held in
    // Arcs that outlive every spawned task that uses the reader (we await
    // each task before dropping the Arc).
    let br_arc: Arc<SpannIndexReader<'static>>;
    let or_arc: Arc<SpannIndexReader<'static>>;
    {
        let (br, or) = rt.block_on(async {
            let br = baseline.open_reader().await?;
            let or = optimized.open_reader().await?;
            anyhow::Ok((br, or))
        })?;
        br_arc = Arc::new(unsafe {
            std::mem::transmute::<SpannIndexReader<'_>, SpannIndexReader<'static>>(br)
        });
        or_arc = Arc::new(unsafe {
            std::mem::transmute::<SpannIndexReader<'_>, SpannIndexReader<'static>>(or)
        });
    }

    // Emit ready event.
    emit_event(&serde_json::json!({
        "event": "ready",
        "baseline_n": baseline.n_records,
        "optimized_n": optimized.n_records,
        "dim": data.dim,
        "queries": queries.queries.iter().map(|q| serde_json::json!({
            "id": q.id, "text": q.text, "query_type": q.query_type,
        })).collect::<Vec<_>>(),
    }));

    // Per-pred caches for matching set + ground truth (per query).
    let cache_arc = Arc::new(data);
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[spann_demo] stdin error: {e}");
                break;
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        let req: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                emit_event(&serde_json::json!({"event": "error", "message": format!("invalid request json: {e}")}));
                continue;
            }
        };

        if req.get("event").and_then(|v| v.as_str()) == Some("quit") {
            break;
        }

        let k = req.get("k").and_then(|v| v.as_u64()).unwrap_or(5) as usize;

        // Resolve query text + embedding.
        let (q_text, q_emb): (String, Vec<f32>) = match (req.get("query_id"), req.get("query_text")) {
            (Some(qid), _) => {
                let qid = qid.as_u64().unwrap_or(0) as usize;
                if qid >= queries.queries.len() {
                    emit_event(&serde_json::json!({"event":"error","message":format!("query_id {qid} out of range")}));
                    continue;
                }
                let q = &queries.queries[qid];
                (q.text.clone(), q.embedding.clone())
            }
            (None, Some(qt)) => {
                let qt = qt.as_str().unwrap_or("").to_string();
                if qt.trim().is_empty() {
                    emit_event(&serde_json::json!({"event":"error","message":"query_text empty"}));
                    continue;
                }
                let t0 = Instant::now();
                let emb = match sidecar.as_mut() {
                    Some(sc) => match sc.embed(&qt) {
                        Ok(e) => e,
                        Err(err) => {
                            emit_event(&serde_json::json!({"event":"error","message":format!("embedding failed: {err}")}));
                            continue;
                        }
                    },
                    None => {
                        emit_event(&serde_json::json!({"event":"error","message":"embed sidecar disabled — pass query_id instead"}));
                        continue;
                    }
                };
                emit_event(&serde_json::json!({"event":"embedded","t_ms":t0.elapsed().as_secs_f64()*1000.0,"dim":emb.len()}));
                (qt, emb)
            }
            (None, None) => {
                emit_event(&serde_json::json!({"event":"error","message":"need query_id or query_text"}));
                continue;
            }
        };

        // Parse predicate.
        let pred_obj = req.get("predicate").cloned().unwrap_or(Value::Null);
        let pred_key = pred_obj.get("key").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let pred_value = pred_obj.get("value").and_then(|v| v.as_str()).map(|s| s.to_string()).unwrap_or_else(|| {
            pred_obj.get("value").map(|v| v.to_string()).unwrap_or_default()
        });
        let predicate = match parse_predicate(&pred_key, &pred_value) {
            Ok(p) => p,
            Err(e) => {
                emit_event(&serde_json::json!({"event":"error","message":format!("predicate parse: {e}")}));
                continue;
            }
        };

        // Compute matching set & ground truth.
        let gt_t = Instant::now();
        let (allowed, allowed_set) = compute_allowed(&cache_arc, &predicate);
        let selectivity = if cache_arc.n() > 0 {
            allowed_set.len() as f64 / cache_arc.n() as f64
        } else {
            0.0
        };
        let signed = SignedRoaringBitmap::Include(allowed.clone());
        let distance: DistanceFunction = baseline.params.clone().space.into();
        let gt = ground_truth(&cache_arc, &q_emb, &allowed_set, &distance, k);
        let gt_set: HashSet<u32> = gt.iter().map(|(id, _)| *id).collect();
        emit_event(&serde_json::json!({
            "event":"ground_truth",
            "n_match": allowed_set.len(),
            "selectivity": selectivity,
            "t_ms": gt_t.elapsed().as_secs_f64()*1000.0,
        }));
        emit_event(&serde_json::json!({
            "event":"query_received",
            "k": k,
            "query_text": q_text,
            "predicate": predicate.display.clone(),
        }));

        // Race the two cells in parallel by spawning each on its own task.
        let q_emb_arc: Arc<Vec<f32>> = Arc::new(q_emb);
        let allowed_set_arc = Arc::new(allowed_set);
        let signed_arc = Arc::new(signed);
        let pred_arc = Arc::new(predicate);
        let gt_set_arc = Arc::new(gt_set);
        let dist_arc = Arc::new(distance);
        let n_records_b = baseline.n_records;
        let n_records_o = optimized.n_records;

        let t0 = Instant::now();
        let br_for_b = br_arc.clone();
        let or_for_o = or_arc.clone();
        let cache_for_b = cache_arc.clone();
        let cache_for_o = cache_arc.clone();
        let q_b = q_emb_arc.clone();
        let q_o = q_emb_arc.clone();
        let signed_b = signed_arc.clone();
        let signed_o = signed_arc.clone();
        let allowed_b = allowed_set_arc.clone();
        let allowed_o = allowed_set_arc.clone();
        let pred_b = pred_arc.clone();
        let pred_o = pred_arc.clone();
        let gt_b = gt_set_arc.clone();
        let gt_o = gt_set_arc.clone();
        let dist_b = dist_arc.clone();
        let dist_o = dist_arc.clone();

        rt.block_on(async move {
            let baseline_handle = tokio::spawn(async move {
                run_cell(
                    "baseline",
                    &br_for_b,
                    n_records_b,
                    &q_b,
                    &dist_b,
                    selectivity,
                    &signed_b,
                    &allowed_b,
                    &gt_b,
                    &pred_b,
                    /*do_bloom=*/ false,
                    /*do_synopsis=*/ false,
                    k,
                    t0,
                    cache_for_b,
                )
                .await
            });
            let optimized_handle = tokio::spawn(async move {
                run_cell(
                    "optimized",
                    &or_for_o,
                    n_records_o,
                    &q_o,
                    &dist_o,
                    selectivity,
                    &signed_o,
                    &allowed_o,
                    &gt_o,
                    &pred_o,
                    /*do_bloom=*/ true,
                    /*do_synopsis=*/ true,
                    k,
                    t0,
                    cache_for_o,
                )
                .await
            });
            let _ = baseline_handle.await;
            let _ = optimized_handle.await;
        });
    }

    // Optional: send {"event": "quit"} to embed sidecar so it exits.
    drop(sidecar);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn run_cell(
    cell: &str,
    reader: &SpannIndexReader<'_>,
    n_records: usize,
    query: &Arc<Vec<f32>>,
    distance: &DistanceFunction,
    selectivity: f64,
    signed: &SignedRoaringBitmap,
    allowed_set: &Arc<HashSet<u32>>,
    gt_set: &HashSet<u32>,
    predicate: &ParsedPredicate,
    do_bloom: bool,
    do_synopsis: bool,
    k: usize,
    t0: Instant,
    _cache: Arc<MsMarcoDemoData>,
) {
    let cell_s = cell.to_string();

    // Probe.
    emit_event(&StageEvent::Probing {
        cell: cell_s.clone(),
        t_ms: ms_since(t0),
    });
    let nprobe_used = reader.determine_search_nprobe(n_records, k, Some(selectivity));
    let (head_ids, _, _) = match rng_query(
        query.as_slice(),
        reader.hnsw_index.clone(),
        nprobe_used as usize,
        None,
        reader.params.search_rng_epsilon,
        reader.params.search_rng_factor,
        distance.clone(),
        false,
    )
    .await
    {
        Ok(t) => t,
        Err(e) => {
            emit_event(&serde_json::json!({"event":"error","cell":cell_s,"message":format!("rng_query: {e}")}));
            return;
        }
    };
    let heads_rng = head_ids.len();
    emit_event(&StageEvent::Probed {
        cell: cell_s.clone(),
        heads_rng,
        nprobe_used,
        t_ms: ms_since(t0),
    });

    // Gates.
    let mut kept: Vec<usize> = head_ids.clone();
    if do_bloom {
        emit_event(&StageEvent::Gating {
            cell: cell_s.clone(),
            gate: "bloom".to_string(),
            t_ms: ms_since(t0),
        });
        let bloom_pred = EqualityTokens::And(vec![predicate.bloom_token.clone()]);
        let pre = kept.len();
        kept = reader.gate_heads(&kept, &bloom_pred);
        emit_event(&StageEvent::Gated {
            cell: cell_s.clone(),
            gate: "bloom".to_string(),
            kept: kept.len(),
            dropped: pre.saturating_sub(kept.len()),
            t_ms: ms_since(t0),
        });
    }
    if do_synopsis {
        emit_event(&StageEvent::Gating {
            cell: cell_s.clone(),
            gate: "synopsis".to_string(),
            t_ms: ms_since(t0),
        });
        let synopsis_pred = SynopsisPredicate::And(vec![(
            predicate.synopsis_key.clone(),
            predicate.synopsis_value.clone(),
        )]);
        let pre = kept.len();
        kept = reader.gate_heads_synopsis(&kept, &synopsis_pred);
        emit_event(&StageEvent::Gated {
            cell: cell_s.clone(),
            gate: "synopsis".to_string(),
            kept: kept.len(),
            dropped: pre.saturating_sub(kept.len()),
            t_ms: ms_since(t0),
        });
    }
    let heads_fetched = kept.len();

    // Fetch posting lists.
    emit_event(&StageEvent::Fetching {
        cell: cell_s.clone(),
        heads: kept.len(),
        t_ms: ms_since(t0),
    });
    let mut candidates_before: usize = 0;
    let mut candidates_after: usize = 0;
    let mut merge_list = Vec::with_capacity(kept.len());
    for h in &kept {
        let pl = match reader.fetch_posting_list(*h as u32).await {
            Ok(p) => p,
            Err(e) => {
                emit_event(&serde_json::json!({"event":"error","cell":cell_s,"message":format!("fetch_pl: {e}")}));
                return;
            }
        };
        candidates_before += pl.len();
        candidates_after += pl
            .iter()
            .filter(|p| allowed_set.contains(&p.doc_offset_id))
            .count();
        let bf = SpannBfPlOperator::new();
        let out = match bf
            .run(&SpannBfPlInput {
                posting_list: pl,
                k,
                query: query.as_ref().clone(),
                distance_function: distance.clone(),
                filter: signed.clone(),
            })
            .await
        {
            Ok(o) => o,
            Err(e) => {
                emit_event(&serde_json::json!({"event":"error","cell":cell_s,"message":format!("bf_pl: {e}")}));
                return;
            }
        };
        merge_list.push(out.records);
    }
    emit_event(&StageEvent::Fetched {
        cell: cell_s.clone(),
        candidates: candidates_after,
        t_ms: ms_since(t0),
    });

    // Score / merge.
    emit_event(&StageEvent::Scoring {
        cell: cell_s.clone(),
        candidates: candidates_after,
        t_ms: ms_since(t0),
    });
    let merged = match (Merge { k: k as u32 })
        .run(&KnnMergeInput {
            batch_measures: merge_list,
        })
        .await
    {
        Ok(m) => m,
        Err(e) => {
            emit_event(&serde_json::json!({"event":"error","cell":cell_s,"message":format!("merge: {e}")}));
            return;
        }
    };
    let returned: HashSet<u32> = merged.measures.iter().map(|r| r.offset_id).collect();
    let hits = gt_set.iter().filter(|id| returned.contains(id)).count();
    let recall = if gt_set.is_empty() {
        0.0
    } else {
        hits as f64 / gt_set.len() as f64
    };

    let top_k_items: Vec<TopKItem> = merged
        .measures
        .iter()
        .take(k)
        .map(|r| {
            let i = (r.offset_id as usize).saturating_sub(1);
            let cache = _cache.clone();
            TopKItem {
                id: r.offset_id,
                score: r.measure as f64,
                text: cache.meta.texts.get(i).cloned().unwrap_or_default(),
                source_domain: cache
                    .meta
                    .source_domains
                    .get(i)
                    .cloned()
                    .unwrap_or_default(),
                topic_bucket: *cache.topic_buckets.get(i).unwrap_or(&-1),
                length_bucket: cache
                    .meta
                    .length_buckets
                    .get(i)
                    .cloned()
                    .unwrap_or_default(),
                query_type: cache.meta.query_types.get(i).cloned().unwrap_or_default(),
            }
        })
        .collect();

    emit_event(&StageEvent::Done {
        cell: cell_s,
        lat_ms: ms_since(t0),
        recall_at_k: recall,
        returned: merged.measures.len(),
        nprobe_used,
        heads_rng,
        heads_fetched,
        candidates_before,
        candidates_after,
        top_k: top_k_items,
    });
}

fn compute_allowed(
    data: &MsMarcoDemoData,
    predicate: &ParsedPredicate,
) -> (RoaringBitmap, HashSet<u32>) {
    let mut bm = RoaringBitmap::new();
    let mut set = HashSet::new();
    let n = data.n();
    for i in 0..n {
        if predicate.matcher.matches(i, data) {
            let id = (i as u32) + 1;
            bm.insert(id);
            set.insert(id);
        }
    }
    if bm.is_empty() {
        // Avoid degenerate divide-by-zero downstream.
        bm.insert(1);
        set.insert(1);
    }
    (bm, set)
}

fn ground_truth(
    data: &MsMarcoDemoData,
    query: &[f32],
    allowed: &HashSet<u32>,
    distance: &DistanceFunction,
    k: usize,
) -> Vec<(u32, f32)> {
    let mut gt: Vec<(u32, f32)> = (0..data.n())
        .filter_map(|i| {
            let id = (i as u32) + 1;
            if allowed.contains(&id) {
                Some((id, distance.distance(&data.embeddings[i], query)))
            } else {
                None
            }
        })
        .collect();
    gt.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    gt.into_iter().take(k).collect()
}

fn ms_since(t0: Instant) -> f64 {
    t0.elapsed().as_secs_f64() * 1000.0
}

// ---------- Embed sidecar ----------

struct EmbedSidecarBuilt {
    child: Child,
    next_id: u64,
}

impl EmbedSidecarBuilt {
    fn spawn(python: &str, script: &Path) -> anyhow::Result<Self> {
        let mut child = Command::new(python)
            .arg(script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;
        // Wait for the {"ready": true} line.
        {
            let stdout = child.stdout.as_mut().unwrap();
            let mut br = BufReader::new(stdout);
            let mut line = String::new();
            loop {
                line.clear();
                br.read_line(&mut line)?;
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let v: serde_json::Value = match serde_json::from_str(trimmed) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if v.get("ready").and_then(|x| x.as_bool()).unwrap_or(false) {
                    eprintln!(
                        "[spann_demo] embed sidecar ready (dim={})",
                        v.get("dim").unwrap_or(&serde_json::Value::Null)
                    );
                    break;
                }
                if v.get("error").is_some() {
                    anyhow::bail!("embed sidecar failed to start: {v}");
                }
            }
        }
        Ok(Self { child, next_id: 1 })
    }

    fn embed(&mut self, text: &str) -> anyhow::Result<Vec<f32>> {
        let id = self.next_id;
        self.next_id += 1;
        let req = serde_json::json!({"id": id, "text": text}).to_string();

        // Write the request.
        {
            let stdin = self.child.stdin.as_mut().unwrap();
            stdin.write_all(req.as_bytes())?;
            stdin.write_all(b"\n")?;
            stdin.flush()?;
        }
        // Read one line of response.
        let stdout = self.child.stdout.as_mut().unwrap();
        let mut br = BufReader::new(stdout);
        let mut line = String::new();
        br.read_line(&mut line)?;
        let resp: serde_json::Value = serde_json::from_str(line.trim())?;
        if let Some(err) = resp.get("error") {
            anyhow::bail!("embed sidecar error: {err}");
        }
        let emb = resp
            .get("embedding")
            .and_then(|x| x.as_array())
            .ok_or_else(|| anyhow::anyhow!("missing embedding in sidecar response"))?;
        let v: Vec<f32> = emb
            .iter()
            .filter_map(|x| x.as_f64().map(|x| x as f32))
            .collect();
        Ok(v)
    }
}

impl Drop for EmbedSidecarBuilt {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
