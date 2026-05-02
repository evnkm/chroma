// BloomHeads ablation: iso-probe (correctness check) + iso-I/O (value claim).
//
// Builds two SPANN indexes on a SIFT1M subset — one with bloom disabled (000)
// and one with bloom enabled + Phase 1.5 Option 2 (010). For each query (and
// the same metadata predicate `bucket = 0` at ~1% selectivity), measures:
//
//   iso-probe: same probe_nbr, both indexes
//     - recall@k must be preserved within bench noise
//     - drop_ratio (heads dropped by gate / heads from rng_query) must be > 0
//     - [gate-audit] must report 0 heads dropped that contained matching docs
//
//   iso-I/O: 000 fetches `B` PLs; 010 probes more centroids and the gate
//     trims down to ~B fetches. Recall@k is compared at matched I/O.
//     The headline measurement: iso-I/O Δrecall ≥ 0 means the gate is worth
//     its complexity.
//
// Output: a CSV under LOGS_PLANS/benchmarks/ with one row per
//   (q_idx, scenario [iso_probe|iso_io], strategy [000|010]).
//
// Env knobs:
//   BENCH_N_RECORDS (default 10000)
//   BENCH_N_QUERIES (default 50)
//   BENCH_K (default 10)
//   BENCH_BUCKETS (default 100, => 1% selectivity)
//   BENCH_PROBE_NBR (default 32) — iso-probe baseline; iso-I/O 000 also uses
//     this and 010 is sized so heads_after_bloom ≈ this number.
//   BENCH_PROBE_NBR_010_ISO_IO (default 0 = auto from drop_ratio with margin)
//   BLOOM_AUDIT_NO_PANIC (default unset = panic on false negative)
//   BLOOM_DOC_TOKENS_CACHE (default unset = off)
//   BLOOM_COMMIT_REBUILD (default unset = off)

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
        types::{GarbageCollectionContext, SpannIndexReader, SpannIndexWriter, SpannMetrics},
        utils::rng_query,
    },
};
use chroma_storage::{local::LocalStorage, Storage};
use chroma_system::Operator;
use chroma_types::{operator::Merge, CollectionUuid, InternalSpannConfiguration, SignedRoaringBitmap};
use roaring::RoaringBitmap;
use worker::execution::operators::{
    knn_merge::KnnMergeInput,
    spann_bf_pl::{SpannBfPlInput, SpannBfPlOperator},
};

fn env_parse<T: std::str::FromStr>(key: &str, default: T) -> T {
    env::var(key).ok().and_then(|s| s.parse::<T>().ok()).unwrap_or(default)
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
    bloom_path: Option<String>,
}

#[allow(clippy::too_many_arguments)]
async fn build_index(
    storage: Storage,
    records: &[(u32, Vec<f32>)],
    record_buckets: &[u32],
    n_buckets: u32,
    dim: usize,
    params: InternalSpannConfiguration,
    bloom_enabled: bool,
    doc_tokens_cache: bool,
    commit_rebuild: bool,
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

    let head_bloom_config = if bloom_enabled {
        let capacity = (params.split_threshold as u64).saturating_mul(4).max(1);
        Some(HeadBloomWriteConfig {
            capacity_per_head: capacity,
            existing_blob_path: None,
            doc_tokens_cache_enabled: doc_tokens_cache,
            commit_rebuild_enabled: commit_rebuild,
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
    )
    .await
    .expect("spann writer");

    eprintln!(
        "[setup] building SPANN index (bloom={}, n={}, buckets={})",
        bloom_enabled, records.len(), n_buckets
    );
    let t = Instant::now();
    for (i, (id, record)) in records.iter().enumerate() {
        let tokens = if bloom_enabled {
            vec![format!("meta::bucket::int::{}", record_buckets[i])]
        } else {
            Vec::new()
        };
        writer
            .add_with_metadata_tokens(*id, record.as_slice(), &tokens)
            .await
            .expect("add record");
    }
    eprintln!("[setup] adds done in {:.1}s", t.elapsed().as_secs_f64());

    let flusher = Box::pin(writer.commit()).await.expect("commit");
    let paths = Box::pin(flusher.flush()).await.expect("flush");
    eprintln!("[setup] commit+flush done in {:.1}s", t.elapsed().as_secs_f64());

    let bloom_path = paths.head_bloom_blob_path.clone();

    let head_bloom_read = bloom_path.as_ref().map(|p| HeadBloomReadConfig {
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
        head_bloom_read,
    ))
    .await
    .expect("spann reader");

    BuiltIndex {
        // SAFETY: The reader borrows from blockfile_provider and hnsw_provider.
        // We move all three into BuiltIndex below; the lifetime erasure here is
        // safe because BuiltIndex owns all backing storage for the reader's
        // lifetime within this single-threaded bench.
        reader: unsafe {
            std::mem::transmute::<SpannIndexReader<'_>, SpannIndexReader<'static>>(reader)
        },
        blockfile_provider,
        hnsw_provider,
        paths,
        bloom_path,
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
    gt.sort_by(|a, b| {
        a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
    });
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
    bloom_tokens: &EqualityTokens,
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
    let kept = reader.gate_heads(&head_ids, bloom_tokens);
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
        .run(&KnnMergeInput { batch_measures: merge_list })
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
    let probe_nbr_010_iso_io_override: usize = env_parse("BENCH_PROBE_NBR_010_ISO_IO", 0usize);
    let doc_tokens_cache = env_flag("BLOOM_DOC_TOKENS_CACHE");
    let commit_rebuild = env_flag("BLOOM_COMMIT_REBUILD");
    let panic_on_bad_drop = !env_flag("BLOOM_AUDIT_NO_PANIC");

    let dataset_label = "sift1m";
    let default_out = format!(
        "LOGS_PLANS/benchmarks/{}-bloom-ablation-n{}-q{}-buckets{}.csv",
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
        "[bench] doc_tokens_cache={} commit_rebuild={} probe_nbr_010_iso_io_override={}",
        doc_tokens_cache, commit_rebuild, probe_nbr_010_iso_io_override
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
    let record_buckets: Vec<u32> = (0..records.len())
        .map(|i| (i as u32) % n_buckets)
        .collect();

    // Allowed bitmap (the predicate `bucket=0`).
    let mut allowed = RoaringBitmap::new();
    for (i, (id, _)) in records.iter().enumerate() {
        if record_buckets[i] == 0 {
            allowed.insert(*id);
        }
    }
    let allowed_set: HashSet<u32> = allowed.iter().collect();
    let allowed_signed = SignedRoaringBitmap::Include(allowed);
    eprintln!("[bench] allowed (predicate bucket=0): {} records", allowed_set.len());

    // Two storage temp dirs: one per index.
    let tmp_000 = tempfile::tempdir()?;
    let tmp_010 = tempfile::tempdir()?;
    let storage_000 = Storage::Local(LocalStorage::new(tmp_000.path().to_str().unwrap()));
    let storage_010 = Storage::Local(LocalStorage::new(tmp_010.path().to_str().unwrap()));

    let params = InternalSpannConfiguration::default();
    let distance_function: DistanceFunction = params.clone().space.into();
    let rng_epsilon = params.search_rng_epsilon;
    let rng_factor = params.search_rng_factor;

    let idx_000 = build_index(
        storage_000,
        &records,
        &record_buckets,
        n_buckets,
        dim,
        params.clone(),
        false,
        false,
        false,
    )
    .await;

    let idx_010 = build_index(
        storage_010,
        &records,
        &record_buckets,
        n_buckets,
        dim,
        params.clone(),
        true,
        doc_tokens_cache,
        commit_rebuild,
    )
    .await;
    eprintln!(
        "[setup] 010 bloom blob path: {:?} (None means cache empty or storage missing)",
        idx_010.bloom_path
    );
    eprintln!(
        "[setup] 010 reader has loaded filters: {}",
        idx_010.reader.head_bloom_filters.is_some()
    );
    if let Some(cache) = idx_010.reader.head_bloom_filters.as_ref() {
        eprintln!(
            "[setup] 010 reader filter cache: len={} (non-stale heads with persisted bloom)",
            cache.len()
        );
    }

    let tokens_000 = EqualityTokens::Unsupported;
    let tokens_010 = EqualityTokens::And(vec!["meta::bucket::int::0".to_string()]);

    let mut out = BufWriter::new(File::create(&output)?);
    writeln!(
        out,
        "dataset,n_records,dim,query_id,k,selectivity,strategy,scenario,probe_nbr,heads_rng,heads_fetched,drop_ratio,recall_at_k,bad_drops,latency_ms"
    )?;

    let selectivity = 1.0 / n_buckets as f64;

    let mut iso_probe_000 = ScenarioStats::default();
    let mut iso_probe_010 = ScenarioStats::default();
    let mut iso_io_000 = ScenarioStats::default();
    let mut iso_io_010 = ScenarioStats::default();
    let mut total_bad_drops = 0usize;

    // Probe the first query with bloom-on to estimate drop_ratio so we can
    // pick a probe count for iso-I/O 010.
    let probed_drop_ratio = if probe_nbr_010_iso_io_override > 0 {
        None
    } else {
        let q = &queries[0];
        let gt = ground_truth(&records, q, &distance_function, &allowed_set, k).await;
        let (_recall, heads_rng, heads_fetched, _lat, _) = run_query(
            &idx_010.reader,
            q,
            probe_nbr,
            rng_epsilon,
            rng_factor,
            &distance_function,
            &tokens_010,
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
    let probe_nbr_010_iso_io = if probe_nbr_010_iso_io_override > 0 {
        probe_nbr_010_iso_io_override
    } else {
        // Fixed-multiplier strategy: probe_nbr / (1 - drop_ratio) with margin.
        let dr = probed_drop_ratio.unwrap_or(0.0).max(0.0).min(0.95);
        let scale = 1.0 / (1.0 - dr).max(0.05);
        ((probe_nbr as f64) * scale).ceil() as usize
    };
    eprintln!(
        "[bench] iso-I/O probe_nbr_010 = {} (probe_nbr_000 = {})",
        probe_nbr_010_iso_io, probe_nbr
    );

    for (q_idx, query) in queries.iter().enumerate() {
        let gt = ground_truth(&records, query, &distance_function, &allowed_set, k).await;

        // Iso-probe scenarios: same probe_nbr for 000 and 010.
        let (recall_000, heads_rng_000, heads_fetched_000, lat_000, bad_000) = run_query(
            &idx_000.reader,
            query,
            probe_nbr,
            rng_epsilon,
            rng_factor,
            &distance_function,
            &tokens_000,
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

        let (recall_010_isoprobe, heads_rng_010_ip, heads_fetched_010_ip, lat_010_ip, bad_010_ip) =
            run_query(
                &idx_010.reader,
                query,
                probe_nbr,
                rng_epsilon,
                rng_factor,
                &distance_function,
                &tokens_010,
                &allowed_signed,
                &allowed_set,
                &gt,
                k,
            )
            .await;
        let dr_010_ip = if heads_rng_010_ip > 0 {
            (heads_rng_010_ip - heads_fetched_010_ip) as f64 / heads_rng_010_ip as f64
        } else {
            0.0
        };
        iso_probe_010.queries += 1;
        iso_probe_010.recall_sum += recall_010_isoprobe;
        iso_probe_010.heads_rng_sum += heads_rng_010_ip;
        iso_probe_010.heads_fetched_sum += heads_fetched_010_ip;
        iso_probe_010.drop_ratio_sum += dr_010_ip;
        iso_probe_010.latency_ms_sum += lat_010_ip;
        iso_probe_010.bad_drops += bad_010_ip;
        total_bad_drops += bad_010_ip;
        let _ = bad_000;

        // Iso-I/O 000: probe_nbr (same as iso_probe). Reuse those numbers
        // to avoid double work.
        iso_io_000.queries += 1;
        iso_io_000.recall_sum += recall_000;
        iso_io_000.heads_rng_sum += heads_rng_000;
        iso_io_000.heads_fetched_sum += heads_fetched_000;
        iso_io_000.drop_ratio_sum += dr_000;
        iso_io_000.latency_ms_sum += lat_000;

        // Iso-I/O 010: probe more centroids; gate trims down to ~probe_nbr.
        let (recall_010_isoio, heads_rng_010_ii, heads_fetched_010_ii, lat_010_ii, bad_010_ii) =
            run_query(
                &idx_010.reader,
                query,
                probe_nbr_010_iso_io,
                rng_epsilon,
                rng_factor,
                &distance_function,
                &tokens_010,
                &allowed_signed,
                &allowed_set,
                &gt,
                k,
            )
            .await;
        let dr_010_ii = if heads_rng_010_ii > 0 {
            (heads_rng_010_ii - heads_fetched_010_ii) as f64 / heads_rng_010_ii as f64
        } else {
            0.0
        };
        iso_io_010.queries += 1;
        iso_io_010.recall_sum += recall_010_isoio;
        iso_io_010.heads_rng_sum += heads_rng_010_ii;
        iso_io_010.heads_fetched_sum += heads_fetched_010_ii;
        iso_io_010.drop_ratio_sum += dr_010_ii;
        iso_io_010.latency_ms_sum += lat_010_ii;
        iso_io_010.bad_drops += bad_010_ii;
        total_bad_drops += bad_010_ii;

        // CSV rows (one per (scenario, strategy)).
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
            "010",
            "iso_probe",
            probe_nbr,
            heads_rng_010_ip,
            heads_fetched_010_ip,
            dr_010_ip,
            recall_010_isoprobe,
            bad_010_ip,
            lat_010_ip
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
            "010",
            "iso_io",
            probe_nbr_010_iso_io,
            heads_rng_010_ii,
            heads_fetched_010_ii,
            dr_010_ii,
            recall_010_isoio,
            bad_010_ii,
            lat_010_ii
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
    eprintln!("========== BloomHeads 000 vs 010 ablation ==========");
    eprintln!("  dataset=sift1m n_records={} buckets={} k={} selectivity={:.4}", n_records, n_buckets, k, selectivity);
    eprintln!("  iso-probe (probe_nbr={}):", probe_nbr);
    eprintln!("    000 (no bloom): {}", fmt(&iso_probe_000));
    eprintln!("    010 (bloom):    {}", fmt(&iso_probe_010));
    if iso_probe_000.queries > 0 && iso_probe_010.queries > 0 {
        let r0 = iso_probe_000.recall_sum / iso_probe_000.queries as f64;
        let r1 = iso_probe_010.recall_sum / iso_probe_010.queries as f64;
        eprintln!("    Δrecall (010 − 000) = {:+.4}", r1 - r0);
    }
    eprintln!("  iso-I/O (probe_nbr_010={}):", probe_nbr_010_iso_io);
    eprintln!("    000 (no bloom): {}", fmt(&iso_io_000));
    eprintln!("    010 (bloom):    {}", fmt(&iso_io_010));
    if iso_io_000.queries > 0 && iso_io_010.queries > 0 {
        let r0 = iso_io_000.recall_sum / iso_io_000.queries as f64;
        let r1 = iso_io_010.recall_sum / iso_io_010.queries as f64;
        eprintln!("    Δrecall (010 − 000) = {:+.4}", r1 - r0);
    }
    eprintln!("  [gate-audit] total bad_drops across all queries: {}", total_bad_drops);
    eprintln!("=====================================================");

    if total_bad_drops > 0 && panic_on_bad_drop {
        panic!(
            "Phase 1 contract violation: gate dropped {} heads containing matching docs (recall lost)",
            total_bad_drops
        );
    }
    eprintln!("[bench] wrote {}", output);
    let _ = (idx_000, idx_010); // keep alive for reader lifetimes

    Ok(())
}
