// HeadSynopsis ablation: iso-probe (correctness check) + iso-I/O (value claim).
//
// Builds two SPANN indexes on a SIFT1M subset — one with synopsis disabled
// (000) and one with synopsis enabled (001). For each query (and the same
// metadata predicate `bucket = 0` at ~1% selectivity), measures:
//
//   iso-probe: same probe_nbr, both indexes
//     - recall@k must be preserved within bench noise (synopsis is exact
//       so Δrecall must be ≈ 0, modulo SPANN-internal nondeterminism)
//     - drop_ratio (heads dropped by gate / heads from rng_query) must be > 0
//     - [gate-audit] must report 0 heads dropped that contained matching
//       docs (the synopsis is exact, so any positive count is a serious bug)
//
//   iso-I/O: 000 fetches `B` PLs; 001 probes more centroids and the gate
//     trims down to ~B fetches. Recall@k is compared at matched I/O.
//     The headline measurement: iso-I/O Δrecall ≥ 0 means the gate is
//     worth its complexity.
//
// Output: a CSV under LOGS_PLANS/benchmarks/ with one row per
//   (q_idx, scenario [iso_probe|iso_io], strategy [000|001]).
//
// Env knobs:
//   BENCH_N_RECORDS (default 10000)
//   BENCH_N_QUERIES (default 50)
//   BENCH_K (default 10)
//   BENCH_BUCKETS (default 100, => 1% selectivity)
//   BENCH_PROBE_NBR (default 32)
//   BENCH_PROBE_NBR_001_ISO_IO (default 0 = auto from drop_ratio with margin)
//   SYNOPSIS_TOP_K (default 64)
//   SYNOPSIS_MAX_CARD (default 1024)
//   SYNOPSIS_AUDIT_NO_PANIC (default unset = panic on false negative)

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

#[allow(dead_code)]
struct BuiltIndex<'a> {
    reader: SpannIndexReader<'a>,
    blockfile_provider: BlockfileProvider,
    hnsw_provider: HnswIndexProvider,
    paths: chroma_index::spann::types::SpannIndexIds,
    synopsis_path: Option<String>,
}

/// Selects which synopsis-build path the bench exercises. The metadata
/// path is the one production code uses (HEAD_SYNOPSIS.md §7); the cache
/// path is the SPANN-side fallback (§8).
#[derive(Clone, Copy, Debug, PartialEq)]
enum BuildPath {
    Cache,
    Metadata,
}

impl BuildPath {
    fn from_env() -> Self {
        match std::env::var("BENCH_SYNOPSIS_BUILD")
            .unwrap_or_else(|_| "cache".into())
            .to_ascii_lowercase()
            .as_str()
        {
            "metadata" | "inverted" | "snapshot" => BuildPath::Metadata,
            _ => BuildPath::Cache,
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn build_index(
    storage: Storage,
    records: &[(u32, Vec<f32>)],
    record_buckets: &[u32],
    n_buckets: u32,
    dim: usize,
    params: InternalSpannConfiguration,
    synopsis_enabled: bool,
    top_k_per_key: u32,
    max_cardinality: u32,
    build_path: BuildPath,
) -> BuiltIndex<'static> {
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
    let hnsw_provider = HnswIndexProvider::new(storage, hnsw_cache, 16);
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

    let head_synopsis_config = if synopsis_enabled {
        Some(HeadSynopsisWriteConfig {
            top_k_per_key,
            max_cardinality,
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
        None,
        head_synopsis_config,
    )
    .await
    .expect("spann writer");

    eprintln!(
        "[setup] building SPANN index (synopsis={}, build_path={:?}, n={}, buckets={})",
        synopsis_enabled,
        build_path,
        records.len(),
        n_buckets
    );
    let t = Instant::now();
    for (i, (id, record)) in records.iter().enumerate() {
        // For Cache build path: feed structured tokens via add_with_…_synopsis.
        // For Metadata build path: skip per-doc tokens; we'll provide a
        // pre-built InvertedIndexSnapshot instead.
        let synopsis_tokens: Vec<SynopsisToken> =
            if synopsis_enabled && build_path == BuildPath::Cache {
                vec![("bucket".to_string(), format!("int::{}", record_buckets[i]))]
            } else {
                Vec::new()
            };
        writer
            .add_with_metadata_tokens_and_synopsis(
                *id,
                record.as_slice(),
                &[],
                &synopsis_tokens,
            )
            .await
            .expect("add record");
    }
    eprintln!("[setup] adds done in {:.1}s", t.elapsed().as_secs_f64());

    // Metadata build path: hand the writer a synthetic InvertedIndexSnapshot
    // built from the bench's known buckets. This is what the production
    // commit_with_metadata_snapshot path does, just without going through
    // a real MetadataSegmentWriterShard. Functionally equivalent.
    if synopsis_enabled && build_path == BuildPath::Metadata {
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
    eprintln!(
        "[setup] commit+flush done in {:.1}s",
        t.elapsed().as_secs_f64()
    );

    let synopsis_path = paths.head_synopsis_blob_path.clone();

    let head_synopsis_read = synopsis_path
        .as_ref()
        .map(|p| HeadSynopsisReadConfig {
            blob_path: Some(p.as_str()),
        });

    let reader = Box::pin(SpannIndexReader::from_id(
        Some(&paths.hnsw_id),
        &hnsw_provider,
        &collection_id,
        params.clone().space.into(),
        dim,
        params.ef_search,
        Some(&paths.pl_id),
        Some(&paths.versions_map_id),
        &blockfile_provider,
        prefix_path,
        false, // disable size-based adaptive_search_nprobe; bench drives nprobe directly
        params,
        None,
        head_synopsis_read,
    ))
    .await
    .expect("spann reader");

    BuiltIndex {
        // SAFETY: The reader borrows from blockfile_provider and hnsw_provider.
        // We move all three into BuiltIndex below; the lifetime erasure here
        // is safe because BuiltIndex owns all backing storage for the
        // reader's lifetime within this single-threaded bench.
        reader: unsafe {
            std::mem::transmute::<SpannIndexReader<'_>, SpannIndexReader<'static>>(reader)
        },
        blockfile_provider,
        hnsw_provider,
        paths,
        synopsis_path,
    }
}

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

#[derive(Default, Clone)]
struct ScenarioStats {
    queries: usize,
    recall_sum: f64,
    heads_rng_sum: usize,
    heads_fetched_sum: usize,
    drop_ratio_sum: f64,
    bad_drops: usize,
    latency_ms_sum: f64,
}

#[allow(clippy::too_many_arguments)]
async fn run_query(
    reader: &SpannIndexReader<'_>,
    query: &[f32],
    probe_nbr: usize,
    rng_epsilon: f32,
    rng_factor: f32,
    distance_function: &DistanceFunction,
    predicate: &SynopsisPredicate,
    allowed_signed: &SignedRoaringBitmap,
    allowed_set: &HashSet<u32>,
    gt: &[u32],
    k: usize,
) -> (f64, usize, usize, f64, usize) {
    let t0 = Instant::now();
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
    let kept = reader.gate_heads_synopsis(&head_ids, predicate);
    let heads_fetched = kept.len();
    let kept_set: HashSet<usize> = kept.iter().copied().collect();

    // Audit guard: count dropped heads whose PL contains a doc in `allowed`.
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
    for head_id in kept {
        let pl = reader
            .fetch_posting_list(head_id as u32)
            .await
            .expect("fetch pl");
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
    let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;
    (recall, heads_rng, heads_fetched, latency_ms, bad_drops)
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let n_records: usize = env_parse("BENCH_N_RECORDS", 10_000usize);
    let n_queries: usize = env_parse("BENCH_N_QUERIES", 50usize);
    let k: usize = env_parse("BENCH_K", 10usize);
    let n_buckets: u32 = env_parse("BENCH_BUCKETS", 100u32);
    let probe_nbr: usize = env_parse("BENCH_PROBE_NBR", 32usize);
    let probe_nbr_001_iso_io_override: usize = env_parse("BENCH_PROBE_NBR_001_ISO_IO", 0usize);
    let top_k_per_key: u32 = env_parse("SYNOPSIS_TOP_K", 64u32);
    let max_cardinality: u32 = env_parse("SYNOPSIS_MAX_CARD", 1024u32);
    let panic_on_bad_drop = !env_flag("SYNOPSIS_AUDIT_NO_PANIC");

    let dataset_label = "sift1m";
    let default_out = format!(
        "LOGS_PLANS/benchmarks/{}-synopsis-ablation-n{}-q{}-buckets{}.csv",
        dataset_label, n_records, n_queries, n_buckets
    );
    let output = env_string("BENCH_OUTPUT", &default_out);
    if let Some(parent) = Path::new(&output).parent() {
        create_dir_all(parent).ok();
    }

    eprintln!(
        "[bench] dataset={} n_records={} n_queries={} k={} buckets={} probe_nbr={}",
        dataset_label, n_records, n_queries, k, n_buckets, probe_nbr
    );
    eprintln!(
        "[bench] top_k_per_key={} max_cardinality={} probe_nbr_001_iso_io_override={}",
        top_k_per_key, max_cardinality, probe_nbr_001_iso_io_override
    );
    eprintln!("[bench] output={}", output);

    eprintln!("[bench] loading SIFT1M (cached on first run)");
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

    // Deterministic per-record bucket. bucket=0 docs are the ones the
    // predicate `bucket=0` matches, ~1/n_buckets selectivity.
    let record_buckets: Vec<u32> = (0..records.len()).map(|i| (i as u32) % n_buckets).collect();

    let mut allowed = RoaringBitmap::new();
    for (i, (id, _)) in records.iter().enumerate() {
        if record_buckets[i] == 0 {
            allowed.insert(*id);
        }
    }
    let allowed_set: HashSet<u32> = allowed.iter().collect();
    let allowed_signed = SignedRoaringBitmap::Include(allowed);
    eprintln!(
        "[bench] allowed (predicate bucket=0): {} records",
        allowed_set.len()
    );

    let tmp_000 = tempfile::tempdir()?;
    let tmp_001 = tempfile::tempdir()?;
    let storage_000 = Storage::Local(LocalStorage::new(tmp_000.path().to_str().unwrap()));
    let storage_001 = Storage::Local(LocalStorage::new(tmp_001.path().to_str().unwrap()));

    let params = InternalSpannConfiguration::default();
    let distance_function: DistanceFunction = params.clone().space.into();
    let rng_epsilon = params.search_rng_epsilon;
    let rng_factor = params.search_rng_factor;

    let build_path = BuildPath::from_env();

    let idx_000 = build_index(
        storage_000,
        &records,
        &record_buckets,
        n_buckets,
        dim,
        params.clone(),
        false,
        top_k_per_key,
        max_cardinality,
        build_path,
    )
    .await;

    let idx_001 = build_index(
        storage_001,
        &records,
        &record_buckets,
        n_buckets,
        dim,
        params.clone(),
        true,
        top_k_per_key,
        max_cardinality,
        build_path,
    )
    .await;
    eprintln!(
        "[setup] 001 synopsis blob path: {:?} (None means cache empty or storage missing)",
        idx_001.synopsis_path
    );
    eprintln!(
        "[setup] 001 reader has loaded synopsis: {}",
        idx_001.reader.head_synopsis_filters.is_some()
    );
    if let Some(cache) = idx_001.reader.head_synopsis_filters.as_ref() {
        eprintln!(
            "[setup] 001 reader synopsis cache: len={} (heads with persisted synopsis)",
            cache.len()
        );
    }

    let pred_off = SynopsisPredicate::Unsupported;
    let pred_on = SynopsisPredicate::And(vec![("bucket".to_string(), "int::0".to_string())]);

    let mut out = BufWriter::new(File::create(&output)?);
    writeln!(
        out,
        "dataset,n_records,dim,query_id,k,selectivity,strategy,scenario,probe_nbr,heads_rng,heads_fetched,drop_ratio,recall_at_k,bad_drops,latency_ms"
    )?;

    let selectivity = 1.0 / n_buckets as f64;

    let mut iso_probe_000 = ScenarioStats::default();
    let mut iso_probe_001 = ScenarioStats::default();
    let mut iso_io_000 = ScenarioStats::default();
    let mut iso_io_001 = ScenarioStats::default();
    let mut total_bad_drops = 0usize;

    // Probe the first query with synopsis-on to estimate drop_ratio so we
    // can pick a probe count for iso-I/O 001.
    let probed_drop_ratio = if probe_nbr_001_iso_io_override > 0 {
        None
    } else {
        let q = &queries[0];
        let gt = ground_truth(&records, q, &distance_function, &allowed_set, k).await;
        let (_recall, heads_rng, heads_fetched, _lat, _) = run_query(
            &idx_001.reader,
            q,
            probe_nbr,
            rng_epsilon,
            rng_factor,
            &distance_function,
            &pred_on,
            &allowed_signed,
            &allowed_set,
            &gt,
            k,
        )
        .await;
        let dr = if heads_rng > 0 {
            (heads_rng - heads_fetched) as f64 / heads_rng as f64
        } else {
            0.0
        };
        eprintln!(
            "[bench] auto: probed q0, heads_rng={} heads_fetched={} drop_ratio={:.3}",
            heads_rng, heads_fetched, dr
        );
        Some(dr)
    };
    let probe_nbr_001_iso_io = if probe_nbr_001_iso_io_override > 0 {
        probe_nbr_001_iso_io_override
    } else {
        let dr = probed_drop_ratio.unwrap_or(0.0).max(0.0).min(0.95);
        let scale = 1.0 / (1.0 - dr).max(0.05);
        ((probe_nbr as f64) * scale).ceil() as usize
    };
    eprintln!(
        "[bench] iso-I/O probe_nbr_001 = {} (probe_nbr_000 = {})",
        probe_nbr_001_iso_io, probe_nbr
    );

    for (q_idx, query) in queries.iter().enumerate() {
        let gt = ground_truth(&records, query, &distance_function, &allowed_set, k).await;

        // Iso-probe: same probe_nbr for 000 and 001.
        let (recall_000, heads_rng_000, heads_fetched_000, lat_000, bad_000) = run_query(
            &idx_000.reader,
            query,
            probe_nbr,
            rng_epsilon,
            rng_factor,
            &distance_function,
            &pred_off,
            &allowed_signed,
            &allowed_set,
            &gt,
            k,
        )
        .await;
        let dr_000 = if heads_rng_000 > 0 {
            (heads_rng_000 - heads_fetched_000) as f64 / heads_rng_000 as f64
        } else {
            0.0
        };
        iso_probe_000.queries += 1;
        iso_probe_000.recall_sum += recall_000;
        iso_probe_000.heads_rng_sum += heads_rng_000;
        iso_probe_000.heads_fetched_sum += heads_fetched_000;
        iso_probe_000.drop_ratio_sum += dr_000;
        iso_probe_000.latency_ms_sum += lat_000;
        let _ = bad_000;

        let (recall_001_isoprobe, heads_rng_001_ip, heads_fetched_001_ip, lat_001_ip, bad_001_ip) =
            run_query(
                &idx_001.reader,
                query,
                probe_nbr,
                rng_epsilon,
                rng_factor,
                &distance_function,
                &pred_on,
                &allowed_signed,
                &allowed_set,
                &gt,
                k,
            )
            .await;
        let dr_001_ip = if heads_rng_001_ip > 0 {
            (heads_rng_001_ip - heads_fetched_001_ip) as f64 / heads_rng_001_ip as f64
        } else {
            0.0
        };
        iso_probe_001.queries += 1;
        iso_probe_001.recall_sum += recall_001_isoprobe;
        iso_probe_001.heads_rng_sum += heads_rng_001_ip;
        iso_probe_001.heads_fetched_sum += heads_fetched_001_ip;
        iso_probe_001.drop_ratio_sum += dr_001_ip;
        iso_probe_001.latency_ms_sum += lat_001_ip;
        iso_probe_001.bad_drops += bad_001_ip;
        total_bad_drops += bad_001_ip;

        // Iso-I/O 000: probe_nbr (same as iso_probe). Reuse.
        iso_io_000.queries += 1;
        iso_io_000.recall_sum += recall_000;
        iso_io_000.heads_rng_sum += heads_rng_000;
        iso_io_000.heads_fetched_sum += heads_fetched_000;
        iso_io_000.drop_ratio_sum += dr_000;
        iso_io_000.latency_ms_sum += lat_000;

        // Iso-I/O 001: probe more centroids; gate trims to ~probe_nbr.
        let (recall_001_isoio, heads_rng_001_ii, heads_fetched_001_ii, lat_001_ii, bad_001_ii) =
            run_query(
                &idx_001.reader,
                query,
                probe_nbr_001_iso_io,
                rng_epsilon,
                rng_factor,
                &distance_function,
                &pred_on,
                &allowed_signed,
                &allowed_set,
                &gt,
                k,
            )
            .await;
        let dr_001_ii = if heads_rng_001_ii > 0 {
            (heads_rng_001_ii - heads_fetched_001_ii) as f64 / heads_rng_001_ii as f64
        } else {
            0.0
        };
        iso_io_001.queries += 1;
        iso_io_001.recall_sum += recall_001_isoio;
        iso_io_001.heads_rng_sum += heads_rng_001_ii;
        iso_io_001.heads_fetched_sum += heads_fetched_001_ii;
        iso_io_001.drop_ratio_sum += dr_001_ii;
        iso_io_001.latency_ms_sum += lat_001_ii;
        iso_io_001.bad_drops += bad_001_ii;
        total_bad_drops += bad_001_ii;

        writeln!(
            out,
            "{},{},{},{},{},{:.6},{},{},{},{},{},{:.4},{:.4},{},{:.3}",
            dataset_label,
            n_records,
            dim,
            q_idx,
            k,
            selectivity,
            "000",
            "iso_probe",
            probe_nbr,
            heads_rng_000,
            heads_fetched_000,
            dr_000,
            recall_000,
            0,
            lat_000
        )?;
        writeln!(
            out,
            "{},{},{},{},{},{:.6},{},{},{},{},{},{:.4},{:.4},{},{:.3}",
            dataset_label,
            n_records,
            dim,
            q_idx,
            k,
            selectivity,
            "001",
            "iso_probe",
            probe_nbr,
            heads_rng_001_ip,
            heads_fetched_001_ip,
            dr_001_ip,
            recall_001_isoprobe,
            bad_001_ip,
            lat_001_ip
        )?;
        writeln!(
            out,
            "{},{},{},{},{},{:.6},{},{},{},{},{},{:.4},{:.4},{},{:.3}",
            dataset_label,
            n_records,
            dim,
            q_idx,
            k,
            selectivity,
            "000",
            "iso_io",
            probe_nbr,
            heads_rng_000,
            heads_fetched_000,
            dr_000,
            recall_000,
            0,
            lat_000
        )?;
        writeln!(
            out,
            "{},{},{},{},{},{:.6},{},{},{},{},{},{:.4},{:.4},{},{:.3}",
            dataset_label,
            n_records,
            dim,
            q_idx,
            k,
            selectivity,
            "001",
            "iso_io",
            probe_nbr_001_iso_io,
            heads_rng_001_ii,
            heads_fetched_001_ii,
            dr_001_ii,
            recall_001_isoio,
            bad_001_ii,
            lat_001_ii
        )?;
    }

    out.flush()?;

    fn fmt(s: &ScenarioStats) -> String {
        if s.queries == 0 {
            return "n/a".to_string();
        }
        format!(
            "queries={} recall_mean={:.4} heads_rng_mean={:.1} heads_fetched_mean={:.1} drop_ratio_mean={:.4} latency_ms_mean={:.2} bad_drops={}",
            s.queries,
            s.recall_sum / s.queries as f64,
            s.heads_rng_sum as f64 / s.queries as f64,
            s.heads_fetched_sum as f64 / s.queries as f64,
            s.drop_ratio_sum / s.queries as f64,
            s.latency_ms_sum / s.queries as f64,
            s.bad_drops
        )
    }

    eprintln!();
    eprintln!("========== HeadSynopsis 000 vs 001 ablation ==========");
    eprintln!(
        "  dataset=sift1m n_records={} buckets={} k={} selectivity={:.4}",
        n_records, n_buckets, k, selectivity
    );
    eprintln!("  iso-probe (probe_nbr={}):", probe_nbr);
    eprintln!("    000 (no synopsis): {}", fmt(&iso_probe_000));
    eprintln!("    001 (synopsis):    {}", fmt(&iso_probe_001));
    if iso_probe_000.queries > 0 && iso_probe_001.queries > 0 {
        let r0 = iso_probe_000.recall_sum / iso_probe_000.queries as f64;
        let r1 = iso_probe_001.recall_sum / iso_probe_001.queries as f64;
        eprintln!("    Δrecall (001 − 000) = {:+.4}", r1 - r0);
    }
    eprintln!("  iso-I/O (probe_nbr_001={}):", probe_nbr_001_iso_io);
    eprintln!("    000 (no synopsis): {}", fmt(&iso_io_000));
    eprintln!("    001 (synopsis):    {}", fmt(&iso_io_001));
    if iso_io_000.queries > 0 && iso_io_001.queries > 0 {
        let r0 = iso_io_000.recall_sum / iso_io_000.queries as f64;
        let r1 = iso_io_001.recall_sum / iso_io_001.queries as f64;
        eprintln!("    Δrecall (001 − 000) = {:+.4}", r1 - r0);
    }
    eprintln!(
        "  [gate-audit] total bad_drops across all queries: {}",
        total_bad_drops
    );
    eprintln!("====================================================");

    if total_bad_drops > 0 && panic_on_bad_drop {
        panic!(
            "Synopsis correctness contract violation: gate dropped {} heads containing matching docs (recall lost)",
            total_bad_drops
        );
    }
    eprintln!("[bench] wrote {}", output);
    let _ = (idx_000, idx_001); // keep alive for reader lifetimes

    Ok(())
}
