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
//   BENCH_STRATEGY=topup           => start at base_nprobe; if returned < k,
//                                     re-issue rng_query with nprobe += base
//                                     until either k results or
//                                     nprobe >= base * max_factor. Sums latency
//                                     across iterations.
//
// adaptive and reader_adaptive use the same clamp formula:
//   clamp(base / max(sel, eps), base, base * max_factor).
//
// Filter shapes (BENCH_FILTER_SHAPE):
//   bernoulli   (default) — each record passes the filter independently with
//                           probability `sel`. Uniform random.
//   categorical            — bucket = hash(id) mod ceil(1/sel); bucket 0 passes.
//                           Bucket boundaries don't follow the vector-space
//                           geometry. Closer to "category=foo" filters in real
//                           workloads.
//   range                  — sort records by L2 norm of embedding, take a
//                           contiguous slice of size sel*N. Correlated with
//                           vector geometry; should be where adaptive earns the
//                           most.
//   adversarial            — exclude the true unfiltered top-50 nearest
//                           neighbors of the query, then sample sel*N from the
//                           rest. Probing more centers can't recover the banned
//                           records — this is the failure-mode benchmark.
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

fn shape_seed_offset(shape: &str) -> u64 {
    match shape {
        "bernoulli" => 0,
        "categorical" => 0xCAFE_BABE,
        "range" => 0xBEEF_F00D,
        "adversarial" => 0xDEAD_BEEF,
        _ => 0,
    }
}

fn generate_filter(
    shape: &str,
    q_idx: usize,
    sel: f64,
    records: &[(u32, Vec<f32>)],
    query: &[f32],
    distance_function: &DistanceFunction,
) -> RoaringBitmap {
    let filter_seed = ((q_idx as u64).wrapping_shl(32))
        ^ ((sel * 1_000_000.0) as u64)
        ^ shape_seed_offset(shape);
    let mut allowed = RoaringBitmap::new();
    match shape {
        "bernoulli" => {
            let mut rng = StdRng::seed_from_u64(filter_seed);
            for r in records {
                if rng.gen::<f64>() < sel {
                    allowed.insert(r.0);
                }
            }
        }
        "categorical" => {
            let n_buckets = ((1.0 / sel).max(1.0)).round() as u64;
            for r in records {
                // Mix the offset id with a fixed multiplier so contiguous ids
                // don't all land in the same bucket.
                let bucket = (r.0 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) % n_buckets;
                if bucket == 0 {
                    allowed.insert(r.0);
                }
            }
        }
        "range" => {
            let mut by_norm: Vec<(u32, f32)> = records
                .iter()
                .map(|(id, emb)| (*id, emb.iter().map(|x| x * x).sum::<f32>().sqrt()))
                .collect();
            by_norm.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let n = records.len();
            let take = ((sel * n as f64).round() as usize).max(1).min(n);
            let start = (filter_seed as usize) % (n - take + 1).max(1);
            for i in start..(start + take).min(n) {
                allowed.insert(by_norm[i].0);
            }
        }
        "adversarial" => {
            let mut all: Vec<(u32, f32)> = records
                .iter()
                .map(|(id, emb)| (*id, distance_function.distance(emb, query)))
                .collect();
            all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let banned: HashSet<u32> = all.iter().take(50).map(|(id, _)| *id).collect();
            let mut rng = StdRng::seed_from_u64(filter_seed);
            for r in records {
                if !banned.contains(&r.0) && rng.gen::<f64>() < sel {
                    allowed.insert(r.0);
                }
            }
        }
        _ => panic!("Unknown BENCH_FILTER_SHAPE: {}", shape),
    }
    if allowed.is_empty() && sel > 0.0 {
        allowed.insert(records[0].0);
    }
    allowed
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let n_records: usize = env_parse("BENCH_N_RECORDS", 10_000usize);
    let n_queries: usize = env_parse("BENCH_N_QUERIES", 50usize);
    let k: usize = env_parse("BENCH_K", 10usize);
    let dataset = env_string("BENCH_DATASET", "sift1m");
    let strategy = env_string("BENCH_STRATEGY", "fixed");
    let filter_shape = env_string("BENCH_FILTER_SHAPE", "bernoulli");
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
        "[bench] dataset={} strategy={} n_records={} n_queries={} k={} filter_shape={}",
        dataset, strategy, n_records, n_queries, k, filter_shape
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
        ))
        .await
        .expect("spann reader");
        readers.insert(base_np, reader);
    }

    let distance_function: DistanceFunction = params.clone().space.into();

    let mut out = BufWriter::new(File::create(&output)?);
    writeln!(
        out,
        "dataset,n_records,dim,query_id,k,selectivity,strategy,base_nprobe,nprobe_used,returned_count,recall_at_k,latency_ms,t_centers_ms,t_fetch_pl_ms,t_bf_pl_ms,t_merge_ms,centers,candidates_before_filter,candidates_after_filter,filter_shape,max_factor,epsilon"
    )?;

    for (q_idx, query) in queries.iter().enumerate() {
        for &sel in &selectivities {
            // Deterministic per-(query, sel, shape) filter.
            let allowed = generate_filter(
                &filter_shape,
                q_idx,
                sel,
                &records,
                query,
                &distance_function,
            );
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
                let initial_nprobe: usize = match strategy.as_str() {
                    "adaptive" | "reader_adaptive" => {
                        let raw = base_np as f64 / sel.max(epsilon);
                        raw.clamp(base_np as f64, base_np as f64 * max_factor)
                            .round() as usize
                    }
                    _ => base_np,
                };
                let cap_nprobe = (base_np as f64 * max_factor).round() as usize;

                // For all strategies except `topup`, the loop runs exactly once.
                // For `topup`, retry with a larger nprobe (incrementing by
                // `base_np`) until either `returned >= k` or we hit the cap.
                let mut nprobe_attempt = initial_nprobe;
                let mut t_centers_ms: f64 = 0.0;
                let mut t_fetch_pl_ms: f64 = 0.0;
                let mut t_bf_pl_ms: f64 = 0.0;
                let mut t_merge_ms: f64 = 0.0;
                let mut last_head_count: usize = 0;
                let mut cands_before: usize = 0;
                let mut cands_after: usize = 0;
                let mut last_returned: usize = 0;
                let mut last_result_ids: HashSet<u32> = HashSet::new();
                let mut nprobe_used: usize = nprobe_attempt;

                let t_total_start = Instant::now();
                loop {
                    let t_centers_start = Instant::now();
                    let (head_ids, _, _) = if strategy == "reader_adaptive" {
                        reader
                            .rng_query(query, n_records, k, Some(sel))
                            .await
                            .expect("reader.rng_query")
                    } else {
                        rng_query(
                            query,
                            reader.hnsw_index.clone(),
                            nprobe_attempt,
                            None,
                            params.search_rng_epsilon,
                            params.search_rng_factor,
                            distance_function.clone(),
                            false,
                        )
                        .await
                        .expect("rng_query")
                    };
                    t_centers_ms += t_centers_start.elapsed().as_secs_f64() * 1000.0;

                    let mut batch = Vec::with_capacity(head_ids.len());
                    let mut cb_iter = 0usize;
                    let mut ca_iter = 0usize;
                    for h in &head_ids {
                        let t_fetch_start = Instant::now();
                        let pl = reader
                            .fetch_posting_list(*h as u32)
                            .await
                            .expect("fetch_pl");
                        t_fetch_pl_ms += t_fetch_start.elapsed().as_secs_f64() * 1000.0;
                        cb_iter += pl.len();
                        ca_iter += pl
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
                        let t_bf_start = Instant::now();
                        let bf_out = SpannBfPlOperator::new()
                            .run(&bf_input)
                            .await
                            .expect("bf_pl");
                        t_bf_pl_ms += t_bf_start.elapsed().as_secs_f64() * 1000.0;
                        batch.push(bf_out.records);
                    }
                    let t_merge_start = Instant::now();
                    let merged = Merge { k: k as u32 }
                        .run(&KnnMergeInput {
                            batch_measures: batch,
                        })
                        .await
                        .expect("merge");
                    t_merge_ms += t_merge_start.elapsed().as_secs_f64() * 1000.0;

                    last_head_count = head_ids.len();
                    cands_before = cb_iter;
                    cands_after = ca_iter;
                    last_returned = merged.measures.len();
                    last_result_ids = merged.measures.iter().map(|r| r.offset_id).collect();
                    nprobe_used = nprobe_attempt;

                    if strategy != "topup" {
                        break;
                    }
                    if last_returned >= k || nprobe_attempt >= cap_nprobe {
                        break;
                    }
                    nprobe_attempt = (nprobe_attempt + base_np).min(cap_nprobe);
                }
                let elapsed_ms = t_total_start.elapsed().as_secs_f64() * 1000.0;
                let returned_count = last_returned;
                let result_ids = last_result_ids;
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
                    "{},{},{},{},{},{:.6},{},{},{},{},{:.6},{:.3},{:.3},{:.3},{:.3},{:.3},{},{},{},{},{:.4},{:.6}",
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
                    t_centers_ms,
                    t_fetch_pl_ms,
                    t_bf_pl_ms,
                    t_merge_ms,
                    last_head_count,
                    cands_before,
                    cands_after,
                    filter_shape,
                    max_factor,
                    epsilon
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
