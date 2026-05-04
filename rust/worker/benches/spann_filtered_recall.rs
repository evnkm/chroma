// Filtered-recall sweep for SPANN on SIFT1M subsets.
//
// Builds a SPANN index once over a SIFT1M subset, then for each query sweeps
// over (selectivity, base_nprobe) and records filtered recall@k vs an exact
// brute-force ground truth restricted to filter-matching records. The harness
// drives `rng_query` directly so it controls the actual nprobe — this avoids
// going through the full distributed stack and bypasses the existing coarse
// `adaptive_search_nprobe` bucketing.
//
// Strategies:
//   BENCH_STRATEGY=fixed           => nprobe_used = base_nprobe
//   BENCH_STRATEGY=adaptive        => harness applies the clamp formula and
//                                     drives `utils::rng_query` directly with
//                                     the result. Used to demonstrate the
//                                     proposed strategy in isolation.
//   BENCH_STRATEGY=reader_adaptive => exercises the production code path
//                                     `SpannIndexReader::rng_query(..., Some(sel))`
//                                     so the reader internally calls
//                                     `filter_aware_nprobe`. Validates the
//                                     committed Rust change end-to-end.
//
// All three boost formulas are identical: clamp(base / max(sel, eps), base,
// base * max_factor). Numbers should match within HNSW determinism.
//
// Output CSV schema (shared across the project):
//   dataset, n_records, dim, query_id, k, selectivity, strategy,
//   base_nprobe, nprobe_used, returned_count, recall_at_k,
//   latency_ms, centers, candidates_before_filter, candidates_after_filter

use std::{
    collections::{HashMap, HashSet},
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
        types::{GarbageCollectionContext, SpannIndexReader, SpannIndexWriter, SpannMetrics},
        utils::rng_query,
    },
};
use chroma_storage::{local::LocalStorage, Storage};
use chroma_system::Operator;
use chroma_types::{
    operator::Merge, CollectionUuid, InternalSpannConfiguration, SignedRoaringBitmap,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
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

fn env_vec_f64(key: &str, default: Vec<f64>) -> Vec<f64> {
    match env::var(key) {
        Ok(s) => {
            let parsed: Vec<f64> = s
                .split(',')
                .filter_map(|p| p.trim().parse::<f64>().ok())
                .collect();
            if parsed.is_empty() {
                default
            } else {
                parsed
            }
        }
        Err(_) => default,
    }
}

fn env_vec_usize(key: &str, default: Vec<usize>) -> Vec<usize> {
    match env::var(key) {
        Ok(s) => {
            let parsed: Vec<usize> = s
                .split(',')
                .filter_map(|p| p.trim().parse::<usize>().ok())
                .collect();
            if parsed.is_empty() {
                default
            } else {
                parsed
            }
        }
        Err(_) => default,
    }
}

fn main() {
    let runtime = tokio_multi_thread();
    runtime.block_on(async {
        run().await.expect("benchmark failed");
    });
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let n_records: usize = env_parse("BENCH_N_RECORDS", 10_000usize);
    let n_queries: usize = env_parse("BENCH_N_QUERIES", 50usize);
    let k: usize = env_parse("BENCH_K", 10usize);
    let dataset = env_string("BENCH_DATASET", "sift1m");
    let strategy = env_string("BENCH_STRATEGY", "fixed");
    let max_factor: f64 = env_parse("BENCH_MAX_FACTOR", 8.0f64);
    let epsilon: f64 = env_parse("BENCH_EPSILON", 0.001f64);
    let selectivities = env_vec_f64(
        "BENCH_SELECTIVITIES",
        vec![0.001, 0.01, 0.05, 0.10, 0.50, 1.0],
    );
    let base_nprobes = env_vec_usize("BENCH_NPROBES", vec![8, 16, 32, 64]);

    let default_out = format!(
        "LOGS_PLANS/benchmarks/{}-{}-n{}-q{}.csv",
        dataset, strategy, n_records, n_queries
    );
    let output = env_string("BENCH_OUTPUT", &default_out);
    if let Some(parent) = Path::new(&output).parent() {
        create_dir_all(parent).ok();
    }

    eprintln!(
        "[bench] dataset={} strategy={} n_records={} n_queries={} k={}",
        dataset, strategy, n_records, n_queries, k
    );
    eprintln!(
        "[bench] selectivities={:?} nprobes={:?} max_factor={} eps={}",
        selectivities, base_nprobes, max_factor, epsilon
    );
    eprintln!("[bench] output={}", output);

    eprintln!("[bench] loading SIFT1M (downloads on first run, then cached)");
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

    let tmp_dir = tempfile::tempdir()?;
    let storage = Storage::Local(LocalStorage::new(tmp_dir.path().to_str().unwrap()));
    let block_cache = new_cache_for_test();
    let sparse_index_cache = new_cache_for_test();
    let max_block_size_bytes = 8_388_608;
    let arrow_blockfile_provider = ArrowBlockfileProvider::new(
        storage.clone(),
        max_block_size_bytes,
        block_cache,
        sparse_index_cache,
        BlockManagerConfig::default_num_concurrent_block_flushes(),
        BlockManagerConfig::default_max_concurrent_block_loads(),
    );
    let blockfile_provider = BlockfileProvider::ArrowBlockfileProvider(arrow_blockfile_provider);
    let hnsw_cache = new_non_persistent_cache_for_test();
    let hnsw_provider = HnswIndexProvider::new(storage.clone(), hnsw_cache, 16);
    let collection_id = CollectionUuid::new();
    let params = InternalSpannConfiguration::default();
    let ef_search = params.ef_search;
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
    let pl_block_size = 5 * 1024 * 1024;

    eprintln!("[bench] building SPANN index over {} records...", records.len());
    let build_t = Instant::now();
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
        pl_block_size,
        SpannMetrics::default(),
        None,
        None,
        None,
    )
    .await
    .expect("spann writer");
    for (id, record) in &records {
        writer
            .add(*id, record.as_slice())
            .await
            .expect("add record");
    }
    let flusher = Box::pin(writer.commit()).await.expect("commit");
    let paths = Box::pin(flusher.flush()).await.expect("flush");
    eprintln!("[bench] index built in {:.1}s", build_t.elapsed().as_secs_f64());

    // Build one reader per base_nprobe with `params.search_nprobe` set so that
    // the production code path (reader.rng_query) uses each base_nprobe as the
    // pre-boost starting point. All readers share the same blockfiles; only
    // the `params` differ.
    let mut readers: HashMap<usize, SpannIndexReader> = HashMap::new();
    for &base_np in &base_nprobes {
        let mut params_for_reader = params.clone();
        params_for_reader.search_nprobe = base_np as u32;
        let reader = Box::pin(SpannIndexReader::from_id(
            Some(&paths.hnsw_id),
            &hnsw_provider,
            &collection_id,
            params_for_reader.clone().space.into(),
            dim,
            ef_search,
            Some(&paths.pl_id),
            Some(&paths.versions_map_id),
            &blockfile_provider,
            prefix_path,
            false, // disable size-based adaptive_search_nprobe so base_nprobe is honored
            params_for_reader,
            None,
            None,
        ))
        .await
        .expect("spann reader");
        readers.insert(base_np, reader);
    }

    let distance_function: DistanceFunction = params.clone().space.into();

    let mut out = BufWriter::new(File::create(&output)?);
    writeln!(
        out,
        "dataset,n_records,dim,query_id,k,selectivity,strategy,base_nprobe,nprobe_used,returned_count,recall_at_k,latency_ms,centers,candidates_before_filter,candidates_after_filter"
    )?;

    for (q_idx, query) in queries.iter().enumerate() {
        for &sel in &selectivities {
            // Deterministic per-(query, selectivity) filter.
            let filter_seed =
                ((q_idx as u64).wrapping_shl(32)) ^ ((sel * 1_000_000.0) as u64);
            let mut rng_filter = StdRng::seed_from_u64(filter_seed);
            let mut allowed = RoaringBitmap::new();
            for r in &records {
                if rng_filter.gen::<f64>() < sel {
                    allowed.insert(r.0);
                }
            }
            if allowed.is_empty() && sel > 0.0 {
                allowed.insert(records[0].0);
            }
            let allowed_set: HashSet<u32> = allowed.iter().collect();
            let signed = SignedRoaringBitmap::Include(allowed.clone());

            // Exact filtered ground truth.
            let mut gt: Vec<(u32, f32)> = records
                .iter()
                .filter(|(id, _)| allowed_set.contains(id))
                .map(|(id, emb)| (*id, distance_function.distance(emb, query)))
                .collect();
            gt.sort_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            let gt_top: HashSet<u32> = gt.iter().take(k).map(|(id, _)| *id).collect();
            let gt_n = gt_top.len();

            for &base_np in &base_nprobes {
                let reader = &readers[&base_np];
                let nprobe_used: usize = match strategy.as_str() {
                    "adaptive" | "reader_adaptive" => {
                        let raw = base_np as f64 / sel.max(epsilon);
                        raw.clamp(base_np as f64, base_np as f64 * max_factor)
                            .round() as usize
                    }
                    _ => base_np,
                };

                let t0 = Instant::now();
                let (head_ids, _, _) = if strategy == "reader_adaptive" {
                    // Exercise the production code path: the reader computes
                    // nprobe internally via `filter_aware_nprobe(base, Some(sel))`.
                    reader
                        .rng_query(query, n_records, k, Some(sel))
                        .await
                        .expect("reader.rng_query")
                } else {
                    rng_query(
                        query,
                        reader.hnsw_index.clone(),
                        nprobe_used,
                        None,
                        params.search_rng_epsilon,
                        params.search_rng_factor,
                        distance_function.clone(),
                        false,
                    )
                    .await
                    .expect("rng_query")
                };

                let mut batch = Vec::with_capacity(head_ids.len());
                let mut cands_before: usize = 0;
                let mut cands_after: usize = 0;
                for h in &head_ids {
                    let pl = reader
                        .fetch_posting_list(*h as u32)
                        .await
                        .expect("fetch_pl");
                    cands_before += pl.len();
                    cands_after += pl
                        .iter()
                        .filter(|p| allowed_set.contains(&p.doc_offset_id))
                        .count();
                    let bf_input = SpannBfPlInput {
                        posting_list: pl,
                        k,
                        filter: signed.clone(),
                        distance_function: distance_function.clone(),
                        query: query.clone(),
                    };
                    let bf_out = SpannBfPlOperator::new()
                        .run(&bf_input)
                        .await
                        .expect("bf_pl");
                    batch.push(bf_out.records);
                }
                let merged = Merge { k: k as u32 }
                    .run(&KnnMergeInput {
                        batch_measures: batch,
                    })
                    .await
                    .expect("merge");
                let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
                let returned_count = merged.measures.len();
                let result_ids: HashSet<u32> =
                    merged.measures.iter().map(|r| r.offset_id).collect();
                let intersection = gt_top.iter().filter(|id| result_ids.contains(id)).count();
                // Recall@k normalised against the smaller of (k, available filtered records).
                let denom = gt_n.min(k);
                let recall = if denom == 0 {
                    1.0
                } else {
                    intersection as f64 / denom as f64
                };

                writeln!(
                    out,
                    "{},{},{},{},{},{:.6},{},{},{},{},{:.6},{:.3},{},{},{}",
                    dataset,
                    n_records,
                    dim,
                    q_idx,
                    k,
                    sel,
                    strategy,
                    base_np,
                    nprobe_used,
                    returned_count,
                    recall,
                    elapsed_ms,
                    head_ids.len(),
                    cands_before,
                    cands_after
                )?;
            }
        }
        if q_idx % 10 == 0 {
            eprintln!("[bench] q_idx={}/{}", q_idx, queries.len());
        }
    }
    out.flush()?;
    eprintln!("[bench] wrote {}", output);
    Ok(())
}
