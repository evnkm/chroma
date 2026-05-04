// Full 8-cell sweep across the 3-toggle hypercube:
//   adaptive_nprobe ∈ {off, on}
//   bloom_filter   ∈ {off, on}
//   metadata_synopsis ∈ {off, on}
//
// Cell labels follow the existing convention: "<a><b><s>" (e.g. 011 = bloom
// + synopsis, no adaptive). The all-off cell (000) is the baseline.
//
// Builds 4 unique SPANN indexes (none / bloom / synopsis / both) and reuses
// each via 2 readers (adaptive on/off). Per-query metrics: recall@k,
// latency, heads_rng (post probe), heads_fetched (post gate), drop_ratio,
// candidates_before/after_filter, bad_drops. Per-cell: index_build_ms,
// blob_storage_bytes.
//
// Uses the production probe-budget rule:
//   probe_nbr = reader.determine_search_nprobe(N, k, Some(selectivity))
// which combines size-based adaptive nprobe + filter-aware boost. Gates
// run in spec order: bloom → synopsis when both enabled (synopsis is
// exact, so it strictly tightens whatever bloom kept).
//
// `[gate-audit]` invariant: every dropped head's PL is rescanned for
// matching docs; bad_drops > 0 is a panic (override with FULL_AUDIT_NO_PANIC).
//
// Env knobs:
//   BENCH_N_RECORDS (default 10000)
//   BENCH_N_QUERIES (default 50)
//   BENCH_K (default 10)
//   BENCH_BUCKETS (default 100, => 1% selectivity)
//   BENCH_PROBE_NBR (default 32) — params.search_nprobe seed
//   BLOOM_DOC_TOKENS_CACHE (default unset = off)
//   BLOOM_COMMIT_REBUILD   (default unset = off)
//   SYNOPSIS_TOP_K   (default 64)
//   SYNOPSIS_MAX_CARD (default 1024)
//   FULL_AUDIT_NO_PANIC (default unset = panic on bad_drop)

use std::{
    collections::HashSet,
    env,
    fs::{create_dir_all, File},
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};

use chroma_benchmark::{benchmark::tokio_multi_thread, datasets::sift::Sift1MData};
use chroma_blockstore::{
    arrow::{config::BlockManagerConfig, provider::ArrowBlockfileProvider},
    provider::BlockfileProvider,
};
use chroma_cache::{new_cache_for_test, new_non_persistent_cache_for_test};
use chroma_config::{registry::Registry, Configurable};
use chroma_distance::DistanceFunction;
use chroma_index::{
    config::{HnswGarbageCollectionConfig, PlGarbageCollectionConfig},
    hnsw_provider::HnswIndexProvider,
    spann::{
        head_bloom::{EqualityTokens, HeadBloomReadConfig, HeadBloomWriteConfig},
        head_synopsis::{
            HeadSynopsisReadConfig, HeadSynopsisWriteConfig, InvertedIndexSnapshot,
            SynopsisPredicate, SynopsisToken,
        },
        types::{GarbageCollectionContext, SpannIndexReader, SpannIndexWriter, SpannMetrics},
        utils::rng_query,
    },
};
use chroma_storage::{local::LocalStorage, Storage};
use chroma_system::Operator;
use chroma_types::{
    operator::Merge, CollectionUuid, InternalSpannConfiguration, SignedRoaringBitmap,
};
use roaring::RoaringBitmap;
use worker::execution::operators::{
    knn_merge::KnnMergeInput,
    spann_bf_pl::{SpannBfPlInput, SpannBfPlOperator},
};

// --- Cell taxonomy --------------------------------------------------------

/// Cell labels follow `<adaptive><bloom><synopsis>`.
const CELL_CODES: [&str; 8] = ["000", "100", "010", "001", "110", "101", "011", "111"];

#[derive(Clone, Copy, Debug)]
struct Cell {
    code: &'static str,
    adaptive: bool,
    bloom: bool,
    synopsis: bool,
}

impl Cell {
    const fn from_code(code: &'static str) -> Self {
        let bytes = code.as_bytes();
        Cell {
            code,
            adaptive: bytes[0] == b'1',
            bloom: bytes[1] == b'1',
            synopsis: bytes[2] == b'1',
        }
    }

    /// Index-flavor key shared across cells with the same writer config.
    fn writer_flavor(&self) -> &'static str {
        match (self.bloom, self.synopsis) {
            (false, false) => "none",
            (true, false) => "bloom",
            (false, true) => "synopsis",
            (true, true) => "both",
        }
    }
}

const CELLS: [Cell; 8] = [
    Cell::from_code("000"),
    Cell::from_code("100"),
    Cell::from_code("010"),
    Cell::from_code("001"),
    Cell::from_code("110"),
    Cell::from_code("101"),
    Cell::from_code("011"),
    Cell::from_code("111"),
];

// --- Env helpers ---------------------------------------------------------

fn env_parse<T: std::str::FromStr>(key: &str, default: T) -> T {
    env::var(key)
        .ok()
        .and_then(|s| s.parse::<T>().ok())
        .unwrap_or(default)
}

fn env_string(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_string())
}

fn env_flag(key: &str) -> bool {
    matches!(
        env::var(key).ok().as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
    )
}

fn main() {
    let runtime = tokio_multi_thread();
    runtime.block_on(async {
        run().await.expect("benchmark failed");
    });
}

// --- Built-index handle --------------------------------------------------

#[allow(dead_code)]
struct BuiltIndex {
    flavor: &'static str,
    blockfile_provider: BlockfileProvider,
    hnsw_provider: HnswIndexProvider,
    paths: chroma_index::spann::types::SpannIndexIds,
    /// Combined storage cost of the gate blobs (bloom + synopsis) on disk.
    blob_storage_bytes: u64,
    build_ms: f64,
    /// Saved tmp dir keeps the on-disk blockfiles alive for readers.
    _tmp: tempfile::TempDir,
    storage: Storage,
    /// Collection id that the writer used; needed to open the reader.
    collection_id: CollectionUuid,
    params: InternalSpannConfiguration,
    dim: usize,
}

#[allow(clippy::too_many_arguments)]
async fn build_index(
    flavor: &'static str,
    records: &[(u32, Vec<f32>)],
    record_buckets: &[u32],
    dim: usize,
    params: InternalSpannConfiguration,
    bloom_capacity_factor: u32,
    bloom_doc_tokens_cache: bool,
    bloom_commit_rebuild: bool,
    synopsis_top_k: u32,
    synopsis_max_card: u32,
) -> BuiltIndex {
    let bloom_enabled = matches!(flavor, "bloom" | "both");
    let synopsis_enabled = matches!(flavor, "synopsis" | "both");

    let tmp = tempfile::tempdir().expect("tmp");
    let storage = Storage::Local(LocalStorage::new(tmp.path().to_str().unwrap()));

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
    let blockfile_provider = BlockfileProvider::ArrowBlockfileProvider(arrow_blockfile_provider);
    let hnsw_cache = new_non_persistent_cache_for_test();
    let hnsw_provider = HnswIndexProvider::new(storage.clone(), hnsw_cache, 16);
    let collection_id = CollectionUuid::new();
    let gc_context = GarbageCollectionContext::try_from_config(
        &(
            PlGarbageCollectionConfig::default(),
            HnswGarbageCollectionConfig::default(),
        ),
        &Registry::default(),
    )
    .await
    .expect("gc context");
    let prefix_path = "";

    let head_bloom_config = if bloom_enabled {
        let capacity = (params.split_threshold as u64)
            .saturating_mul(bloom_capacity_factor.max(1) as u64)
            .max(1);
        Some(HeadBloomWriteConfig {
            capacity_per_head: capacity,
            existing_blob_path: None,
            doc_tokens_cache_enabled: bloom_doc_tokens_cache,
            commit_rebuild_enabled: bloom_commit_rebuild,
        })
    } else {
        None
    };
    let head_synopsis_config = if synopsis_enabled {
        Some(HeadSynopsisWriteConfig {
            top_k_per_key: synopsis_top_k,
            max_cardinality: synopsis_max_card,
            existing_blob_path: None,
        })
    } else {
        None
    };

    let writer = SpannIndexWriter::from_id(
        &hnsw_provider,
        None,
        None,
        None,
        None,
        &collection_id,
        prefix_path,
        dim,
        &blockfile_provider,
        params.clone(),
        gc_context,
        5 * 1024 * 1024,
        SpannMetrics::default(),
        None,
        head_bloom_config,
        head_synopsis_config,
    )
    .await
    .expect("spann writer");

    eprintln!(
        "[setup] flavor={} building index over {} records...",
        flavor,
        records.len()
    );
    let t = Instant::now();
    for (i, (id, record)) in records.iter().enumerate() {
        let bucket = record_buckets[i];
        let bloom_tokens = if bloom_enabled {
            vec![format!("meta::bucket::int::{}", bucket)]
        } else {
            Vec::new()
        };
        let synopsis_tokens: Vec<SynopsisToken> = if synopsis_enabled {
            vec![("bucket".to_string(), format!("int::{}", bucket))]
        } else {
            Vec::new()
        };
        writer
            .add_with_metadata_tokens_and_synopsis(
                *id,
                record.as_slice(),
                &bloom_tokens,
                &synopsis_tokens,
            )
            .await
            .expect("add");
    }

    // Production-mode synopsis build: feed an InvertedIndexSnapshot directly
    // (mirrors what SpannSegmentWriterShard::commit_with_metadata_snapshot
    // does in production).
    if synopsis_enabled {
        let mut snap = InvertedIndexSnapshot::new();
        let mut by_bucket: std::collections::HashMap<u32, RoaringBitmap> =
            std::collections::HashMap::new();
        for (i, (id, _)) in records.iter().enumerate() {
            by_bucket
                .entry(record_buckets[i])
                .or_default()
                .insert(*id);
        }
        for (bucket, bm) in by_bucket {
            snap.insert_int("bucket", bucket, bm);
        }
        writer.set_synopsis_inverted_index(snap).await;
    }

    let flusher = Box::pin(writer.commit()).await.expect("commit");
    let paths = Box::pin(flusher.flush()).await.expect("flush");
    let build_ms = t.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "[setup] flavor={} built in {:.0} ms",
        flavor, build_ms
    );

    // Storage-cost accounting: re-fetch the blob bytes from storage.
    let blob_storage_bytes: u64 = {
        let mut total: u64 = 0;
        for path in [&paths.head_bloom_blob_path, &paths.head_synopsis_blob_path] {
            if let Some(p) = path.as_ref() {
                if let Ok(bytes) = storage
                    .get(
                        p,
                        chroma_storage::GetOptions::new(
                            chroma_storage::admissioncontrolleds3::StorageRequestPriority::P0,
                        ),
                    )
                    .await
                {
                    total = total.saturating_add(bytes.len() as u64);
                }
            }
        }
        total
    };

    BuiltIndex {
        flavor,
        blockfile_provider,
        hnsw_provider,
        paths,
        blob_storage_bytes,
        build_ms,
        _tmp: tmp,
        storage,
        collection_id,
        params,
        dim,
    }
}

async fn make_reader(idx: &BuiltIndex, adaptive: bool) -> SpannIndexReader<'static> {
    let head_bloom_read =
        idx.paths
            .head_bloom_blob_path
            .as_ref()
            .map(|p| HeadBloomReadConfig {
                blob_path: Some(p.as_str()),
            });
    let head_synopsis_read = idx
        .paths
        .head_synopsis_blob_path
        .as_ref()
        .map(|p| HeadSynopsisReadConfig {
            blob_path: Some(p.as_str()),
        });
    let reader = Box::pin(SpannIndexReader::from_id(
        Some(&idx.paths.hnsw_id),
        &idx.hnsw_provider,
        &idx.collection_id,
        idx.params.clone().space.into(),
        idx.dim,
        idx.params.ef_search,
        Some(&idx.paths.pl_id),
        Some(&idx.paths.versions_map_id),
        &idx.blockfile_provider,
        "",
        adaptive,
        idx.params.clone(),
        head_bloom_read,
        head_synopsis_read,
    ))
    .await
    .expect("spann reader");
    // SAFETY: Reader borrows from blockfile_provider/hnsw_provider, which
    // we keep alive in `BuiltIndex` for the duration of the bench. We
    // erase the borrow lifetime so the reader can live alongside the
    // 'static index handles in the per-cell map.
    unsafe { std::mem::transmute::<SpannIndexReader<'_>, SpannIndexReader<'static>>(reader) }
}

// --- Query path ----------------------------------------------------------

async fn ground_truth(
    records: &[(u32, Vec<f32>)],
    query: &[f32],
    distance_function: &DistanceFunction,
    allowed: &HashSet<u32>,
    k: usize,
) -> Vec<u32> {
    let mut gt: Vec<(u32, f32)> = records
        .iter()
        .filter(|(id, _)| allowed.contains(id))
        .map(|(id, e)| (*id, distance_function.distance(e, query)))
        .collect();
    gt.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    gt.into_iter().take(k).map(|(id, _)| id).collect()
}

#[derive(Clone, Default)]
struct CellRow {
    recall: f64,
    heads_rng: usize,
    heads_fetched: usize,
    drop_ratio: f64,
    candidates_before_filter: usize,
    candidates_after_filter: usize,
    latency_ms: f64,
    bad_drops: usize,
    nprobe_used: u32,
}

#[allow(clippy::too_many_arguments)]
async fn run_query_for_cell(
    cell: &Cell,
    reader: &SpannIndexReader<'_>,
    n_records: usize,
    query: &[f32],
    rng_epsilon: f32,
    rng_factor: f32,
    distance_function: &DistanceFunction,
    selectivity: f64,
    bloom_pred: &EqualityTokens,
    synopsis_pred: &SynopsisPredicate,
    allowed_signed: &SignedRoaringBitmap,
    allowed_set: &HashSet<u32>,
    gt: &[u32],
    k: usize,
) -> CellRow {
    let t0 = Instant::now();

    // Probe budget: production logic. determine_search_nprobe combines the
    // size-based adaptive rule (gated by reader.adaptive_search_nprobe) and
    // the always-on filter-aware boost.
    let probe_nbr =
        reader.determine_search_nprobe(n_records, k, Some(selectivity)) as usize;

    let (head_ids, _, _) = rng_query(
        query,
        reader.hnsw_index.clone(),
        probe_nbr,
        None,
        rng_epsilon,
        rng_factor,
        distance_function.clone(),
        false,
    )
    .await
    .expect("rng query");
    let heads_rng = head_ids.len();

    // Apply gates in spec order: bloom first, then synopsis (exact gate
    // strictly tightens whatever the bloom kept).
    let mut kept = head_ids.clone();
    if cell.bloom {
        kept = reader.gate_heads(&kept, bloom_pred);
    }
    if cell.synopsis {
        kept = reader.gate_heads_synopsis(&kept, synopsis_pred);
    }
    let heads_fetched = kept.len();
    let kept_set: HashSet<usize> = kept.iter().copied().collect();

    // Audit: any head dropped that contained a matching doc?
    let mut bad_drops = 0usize;
    for h in &head_ids {
        if kept_set.contains(h) {
            continue;
        }
        let pl = reader
            .fetch_posting_list(*h as u32)
            .await
            .expect("fetch pl");
        if pl.iter().any(|p| allowed_set.contains(&p.doc_offset_id)) {
            bad_drops += 1;
        }
    }

    // Run BfPL on kept heads + merge.
    let mut merge_list = Vec::new();
    let mut candidates_before: usize = 0;
    let mut candidates_after: usize = 0;
    for head_id in kept {
        let pl = reader
            .fetch_posting_list(head_id as u32)
            .await
            .expect("fetch pl");
        candidates_before += pl.len();
        candidates_after += pl
            .iter()
            .filter(|p| allowed_set.contains(&p.doc_offset_id))
            .count();
        let bf = SpannBfPlOperator::new();
        let out = bf
            .run(&SpannBfPlInput {
                posting_list: pl,
                k,
                query: query.to_vec(),
                distance_function: distance_function.clone(),
                filter: allowed_signed.clone(),
            })
            .await
            .expect("bf op");
        merge_list.push(out.records);
    }
    let merged = Merge { k: k as u32 }
        .run(&KnnMergeInput {
            batch_measures: merge_list,
        })
        .await
        .expect("merge");

    let returned: HashSet<u32> = merged.measures.iter().map(|r| r.offset_id).collect();
    let hits = gt.iter().filter(|id| returned.contains(id)).count();
    let recall = if gt.is_empty() {
        0.0
    } else {
        hits as f64 / gt.len() as f64
    };
    let drop_ratio = if heads_rng > 0 {
        (heads_rng - heads_fetched) as f64 / heads_rng as f64
    } else {
        0.0
    };
    let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;

    CellRow {
        recall,
        heads_rng,
        heads_fetched,
        drop_ratio,
        candidates_before_filter: candidates_before,
        candidates_after_filter: candidates_after,
        latency_ms,
        bad_drops,
        nprobe_used: probe_nbr as u32,
    }
}

// --- Aggregator ----------------------------------------------------------

#[derive(Default, Clone)]
struct CellAgg {
    queries: usize,
    recall_sum: f64,
    heads_rng_sum: usize,
    heads_fetched_sum: usize,
    drop_ratio_sum: f64,
    candidates_before_sum: usize,
    candidates_after_sum: usize,
    latency_sum: f64,
    latency_p99: f64,
    bad_drops: usize,
    nprobe_max: u32,
    latencies: Vec<f64>,
}

impl CellAgg {
    fn update(&mut self, row: &CellRow) {
        self.queries += 1;
        self.recall_sum += row.recall;
        self.heads_rng_sum += row.heads_rng;
        self.heads_fetched_sum += row.heads_fetched;
        self.drop_ratio_sum += row.drop_ratio;
        self.candidates_before_sum += row.candidates_before_filter;
        self.candidates_after_sum += row.candidates_after_filter;
        self.latency_sum += row.latency_ms;
        self.bad_drops += row.bad_drops;
        if row.nprobe_used > self.nprobe_max {
            self.nprobe_max = row.nprobe_used;
        }
        self.latencies.push(row.latency_ms);
    }
    fn finalize(&mut self) {
        if self.latencies.is_empty() {
            return;
        }
        let mut sorted = self.latencies.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = (((sorted.len() as f64) * 0.99).ceil() as usize).saturating_sub(1);
        self.latency_p99 = sorted[idx.min(sorted.len() - 1)];
    }

    fn fmt(&self) -> String {
        if self.queries == 0 {
            return "n/a".into();
        }
        format!(
            "rec={:.4} hrng={:.1} hfet={:.1} drop={:.4} c_b={:.0} c_a={:.0} lat={:.2}ms p99={:.2}ms bad={} np_max={}",
            self.recall_sum / self.queries as f64,
            self.heads_rng_sum as f64 / self.queries as f64,
            self.heads_fetched_sum as f64 / self.queries as f64,
            self.drop_ratio_sum / self.queries as f64,
            self.candidates_before_sum as f64 / self.queries as f64,
            self.candidates_after_sum as f64 / self.queries as f64,
            self.latency_sum / self.queries as f64,
            self.latency_p99,
            self.bad_drops,
            self.nprobe_max,
        )
    }
}

// --- Driver --------------------------------------------------------------

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let n_records: usize = env_parse("BENCH_N_RECORDS", 10_000usize);
    let n_queries: usize = env_parse("BENCH_N_QUERIES", 50usize);
    let k: usize = env_parse("BENCH_K", 10usize);
    let n_buckets: u32 = env_parse("BENCH_BUCKETS", 100u32);
    let probe_nbr: u32 = env_parse("BENCH_PROBE_NBR", 32u32);
    let bloom_capacity_factor: u32 = env_parse("BLOOM_CAPACITY_FACTOR", 4u32);
    let bloom_doc_tokens_cache = env_flag("BLOOM_DOC_TOKENS_CACHE");
    let bloom_commit_rebuild = env_flag("BLOOM_COMMIT_REBUILD");
    let synopsis_top_k: u32 = env_parse("SYNOPSIS_TOP_K", 64u32);
    let synopsis_max_card: u32 = env_parse("SYNOPSIS_MAX_CARD", 1024u32);
    let panic_on_bad_drop = !env_flag("FULL_AUDIT_NO_PANIC");

    let dataset_label = "sift1m";
    let default_out = format!(
        "LOGS_PLANS/benchmarks/full_sweep/{}-fullsweep-n{}-q{}-buckets{}.csv",
        dataset_label, n_records, n_queries, n_buckets
    );
    let output = env_string("BENCH_OUTPUT", &default_out);
    if let Some(parent) = Path::new(&output).parent() {
        create_dir_all(parent).ok();
    }

    eprintln!(
        "[bench] dataset={} n_records={} n_queries={} k={} buckets={} probe_nbr={} top_k={} max_card={}",
        dataset_label, n_records, n_queries, k, n_buckets, probe_nbr, synopsis_top_k, synopsis_max_card
    );
    eprintln!("[bench] output={}", output);

    eprintln!("[bench] loading SIFT1M");
    let mut sift = Sift1MData::init().await?;
    let base = sift.data_range(0..n_records).await?;
    let dim = base.first().map(|v| v.len()).unwrap_or(128);
    let queries_with_gt = sift.query().await?;
    let queries: Vec<Vec<f32>> = queries_with_gt
        .iter()
        .take(n_queries)
        .map(|(q, _)| q.clone())
        .collect();
    eprintln!(
        "[bench] loaded {} base records, {} queries (dim={})",
        base.len(),
        queries.len(),
        dim
    );

    let records: Vec<(u32, Vec<f32>)> = base
        .into_iter()
        .enumerate()
        .map(|(i, v)| (i as u32 + 1, v))
        .collect();
    let record_buckets: Vec<u32> = (0..records.len()).map(|i| (i as u32) % n_buckets).collect();
    let mut allowed = RoaringBitmap::new();
    for (i, (id, _)) in records.iter().enumerate() {
        if record_buckets[i] == 0 {
            allowed.insert(*id);
        }
    }
    let allowed_set: HashSet<u32> = allowed.iter().collect();
    let allowed_signed = SignedRoaringBitmap::Include(allowed);
    let selectivity = 1.0 / n_buckets as f64;
    eprintln!(
        "[bench] allowed (predicate bucket=0): {} records, sel={:.4}",
        allowed_set.len(),
        selectivity
    );

    // Build the 4 unique writer flavors. params.search_nprobe seeds the
    // non-adaptive cells; adaptive cells overlay the size-based rule.
    let mut params = InternalSpannConfiguration::default();
    params.search_nprobe = probe_nbr;
    let distance_function: DistanceFunction = params.clone().space.into();
    let rng_epsilon = params.search_rng_epsilon;
    let rng_factor = params.search_rng_factor;

    let mut indexes: std::collections::HashMap<&'static str, BuiltIndex> =
        std::collections::HashMap::new();
    for flavor in ["none", "bloom", "synopsis", "both"] {
        let idx = build_index(
            flavor,
            &records,
            &record_buckets,
            dim,
            params.clone(),
            bloom_capacity_factor,
            bloom_doc_tokens_cache,
            bloom_commit_rebuild,
            synopsis_top_k,
            synopsis_max_card,
        )
        .await;
        indexes.insert(flavor, idx);
    }

    // Build readers — 8 of them, one per cell.
    eprintln!("[setup] opening readers...");
    let mut readers: std::collections::HashMap<&'static str, SpannIndexReader<'static>> =
        std::collections::HashMap::new();
    for cell in &CELLS {
        let idx = indexes.get(cell.writer_flavor()).expect("flavor present");
        let reader = make_reader(idx, cell.adaptive).await;
        readers.insert(cell.code, reader);
    }

    let bloom_pred = EqualityTokens::And(vec!["meta::bucket::int::0".to_string()]);
    let synopsis_pred =
        SynopsisPredicate::And(vec![("bucket".to_string(), "int::0".to_string())]);

    let mut out = BufWriter::new(File::create(&output)?);
    writeln!(
        out,
        "dataset,n_records,dim,query_id,k,selectivity,cell,adaptive,bloom,synopsis,nprobe_used,heads_rng,heads_fetched,drop_ratio,recall_at_k,candidates_before_filter,candidates_after_filter,latency_ms,bad_drops"
    )?;

    let mut aggs: std::collections::HashMap<&'static str, CellAgg> =
        CELLS.iter().map(|c| (c.code, CellAgg::default())).collect();
    let mut total_bad_drops = 0usize;

    for (q_idx, query) in queries.iter().enumerate() {
        let gt = ground_truth(&records, query, &distance_function, &allowed_set, k).await;
        for cell in &CELLS {
            let reader = readers.get(cell.code).unwrap();
            let row = run_query_for_cell(
                cell,
                reader,
                n_records,
                query,
                rng_epsilon,
                rng_factor,
                &distance_function,
                selectivity,
                &bloom_pred,
                &synopsis_pred,
                &allowed_signed,
                &allowed_set,
                &gt,
                k,
            )
            .await;
            total_bad_drops += row.bad_drops;
            aggs.get_mut(cell.code).unwrap().update(&row);

            writeln!(
                out,
                "{},{},{},{},{},{:.6},{},{},{},{},{},{},{},{:.4},{:.4},{},{},{:.3},{}",
                dataset_label,
                n_records,
                dim,
                q_idx,
                k,
                selectivity,
                cell.code,
                cell.adaptive as u32,
                cell.bloom as u32,
                cell.synopsis as u32,
                row.nprobe_used,
                row.heads_rng,
                row.heads_fetched,
                row.drop_ratio,
                row.recall,
                row.candidates_before_filter,
                row.candidates_after_filter,
                row.latency_ms,
                row.bad_drops
            )?;
        }
    }
    out.flush()?;
    for code in CELL_CODES {
        if let Some(a) = aggs.get_mut(code) {
            a.finalize();
        }
    }

    // Cell-level summary metadata to a sibling JSON for the plotter.
    let summary_path = format!(
        "{}.summary.json",
        output.trim_end_matches(".csv"),
    );
    {
        let mut s = String::new();
        s.push_str("{\n  \"dataset\": \"sift1m\",\n");
        s.push_str(&format!("  \"n_records\": {},\n", n_records));
        s.push_str(&format!("  \"n_queries\": {},\n", n_queries));
        s.push_str(&format!("  \"buckets\": {},\n", n_buckets));
        s.push_str(&format!("  \"selectivity\": {},\n", selectivity));
        s.push_str(&format!("  \"probe_nbr_seed\": {},\n", probe_nbr));
        s.push_str("  \"index_metadata\": {\n");
        for (i, flavor) in ["none", "bloom", "synopsis", "both"].iter().enumerate() {
            let idx = indexes.get(*flavor).unwrap();
            s.push_str(&format!(
                "    \"{}\": {{\"build_ms\": {:.0}, \"blob_storage_bytes\": {}}}{}\n",
                flavor,
                idx.build_ms,
                idx.blob_storage_bytes,
                if i + 1 < 4 { "," } else { "" }
            ));
        }
        s.push_str("  }\n}\n");
        std::fs::write(&summary_path, s)?;
    }

    eprintln!();
    eprintln!("========== Full 8-cell sweep ==========");
    eprintln!(
        "  dataset=sift1m n_records={} buckets={} k={} sel={:.4} probe_nbr_seed={}",
        n_records, n_buckets, k, selectivity, probe_nbr
    );
    for code in CELL_CODES {
        let a = aggs.get(code).unwrap();
        let cell = CELLS.iter().find(|c| c.code == code).unwrap();
        let idx = indexes.get(cell.writer_flavor()).unwrap();
        eprintln!(
            "  cell={} (adapt={} bloom={} synop={}) {}  | build={:.0}ms blob={}B",
            code,
            cell.adaptive as u32,
            cell.bloom as u32,
            cell.synopsis as u32,
            a.fmt(),
            idx.build_ms,
            idx.blob_storage_bytes,
        );
    }
    eprintln!(
        "  [gate-audit] total bad_drops across all cells/queries: {}",
        total_bad_drops
    );
    eprintln!("=======================================");
    eprintln!("[bench] wrote {}", output);
    eprintln!("[bench] wrote {}", summary_path);

    if total_bad_drops > 0 && panic_on_bad_drop {
        panic!(
            "Gate-audit invariant violated: {} heads dropped contained matching docs",
            total_bad_drops
        );
    }
    let _ = (indexes, readers); // keep alive for lifetimes
    Ok(())
}
