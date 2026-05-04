// Within-subjects bloom-gate ablation across a probe_nbr sweep.
//
// Builds one SPANN index with bloom filters enabled (Phase 1.5 Option 2).
// Then for each (query, probe_off in sweep) measures three conditions on the
// SAME index, eliminating cross-index variance:
//
//   bloom_off          : EqualityTokens::Unsupported → gate is a pass-through
//   bloom_on_iso_probe : predicate tokens, same probe_nbr → recall preservation
//   bloom_on_iso_io    : predicate tokens, probe_on chosen so heads_fetched ≈ probe_off
//
// Iso-I/O sizing: probe_on = ceil(probe_off / (1 - drop_ratio)) using a warmup
// estimate of drop_ratio from a single query at the largest probe_off.
//
// CSV schema:
//   dataset,n_records,dim,query_id,k,selectivity,scenario,probe_nbr_off,
//   probe_nbr_used,heads_rng,heads_fetched,drop_ratio,recall_at_k,
//   bad_drops,latency_ms
//
// Env knobs:
//   BENCH_N_RECORDS    (default 50000)
//   BENCH_N_QUERIES    (default 50)
//   BENCH_K            (default 10)
//   BENCH_BUCKETS      (default 100, => 1% selectivity for predicate bucket=0)
//   BENCH_PROBE_NBRS   (default "8,16,32,64,128")  comma-separated
//   BLOOM_COMMIT_REBUILD     (default unset = off; set to 1 to enable Option 2)
//   BLOOM_DOC_TOKENS_CACHE   (default unset = off)
//   BLOOM_AUDIT_NO_PANIC     (default unset = panic on false negative)
//   BENCH_OUTPUT       (default LOGS_PLANS/benchmarks/sift1m-bloom-sweep-*.csv)

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

fn env_csv_usize(key: &str, default: Vec<usize>) -> Vec<usize> {
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

    let mut bad_drops = 0usize;
    for h in &head_ids {
        if kept_set.contains(h) {
            continue;
        }
        let pl = reader.fetch_posting_list(*h as u32).await.expect("fetch pl");
        if pl.iter().any(|p| allowed_set.contains(&p.doc_offset_id)) {
            bad_drops += 1;
        }
    }

    let mut merge_list = Vec::new();
    for head_id in kept {
        let pl = reader.fetch_posting_list(head_id as u32).await.expect("fetch pl");
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
    let n_records: usize = env_parse("BENCH_N_RECORDS", 50_000usize);
    let n_queries: usize = env_parse("BENCH_N_QUERIES", 50usize);
    let k: usize = env_parse("BENCH_K", 10usize);
    let n_buckets: u32 = env_parse("BENCH_BUCKETS", 100u32);
    let probe_nbrs_off: Vec<usize> = env_csv_usize("BENCH_PROBE_NBRS", vec![8, 16, 32, 64, 128]);
    let doc_tokens_cache = env_flag("BLOOM_DOC_TOKENS_CACHE");
    let commit_rebuild = env_flag("BLOOM_COMMIT_REBUILD");
    let panic_on_bad_drop = !env_flag("BLOOM_AUDIT_NO_PANIC");

    let dataset_label = "sift1m";
    let default_out = format!(
        "LOGS_PLANS/benchmarks/{}-bloom-sweep-n{}-q{}-buckets{}.csv",
        dataset_label, n_records, n_queries, n_buckets
    );
    let output = env_string("BENCH_OUTPUT", &default_out);
    if let Some(parent) = Path::new(&output).parent() {
        create_dir_all(parent).ok();
    }

    eprintln!(
        "[bench] dataset={} n_records={} n_queries={} k={} buckets={}",
        dataset_label, n_records, n_queries, k, n_buckets
    );
    eprintln!(
        "[bench] probe_nbrs={:?} commit_rebuild={} doc_tokens_cache={}",
        probe_nbrs_off, commit_rebuild, doc_tokens_cache
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
    eprintln!("[bench] loaded {} base records, {} queries (dim={})", base.len(), queries.len(), dim);

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
    eprintln!("[bench] allowed (predicate bucket=0): {} records, selectivity={:.6}", allowed_set.len(), selectivity);

    let tmp = tempfile::tempdir()?;
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
    let hnsw_provider = HnswIndexProvider::new(storage, hnsw_cache, 16);
    let collection_id = CollectionUuid::new();
    let params = InternalSpannConfiguration::default();
    let distance_function: DistanceFunction = params.clone().space.into();
    let rng_epsilon = params.search_rng_epsilon;
    let rng_factor = params.search_rng_factor;
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

    // Build idx_010 (bloom enabled, Option 2 by default).
    let head_bloom_config = Some(HeadBloomWriteConfig {
        capacity_per_head: (params.split_threshold as u64).saturating_mul(4).max(1),
        existing_blob_path: None,
        doc_tokens_cache_enabled: doc_tokens_cache,
        commit_rebuild_enabled: commit_rebuild,
    });
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
        None,
    )
    .await
    .expect("spann writer");

    eprintln!("[setup] adding {} records (with metadata tokens)…", records.len());
    let t = Instant::now();
    for (i, (id, record)) in records.iter().enumerate() {
        let tokens = vec![format!("meta::bucket::int::{}", record_buckets[i])];
        writer
            .add_with_metadata_tokens(*id, record.as_slice(), &tokens)
            .await
            .expect("add");
    }
    let flusher = Box::pin(writer.commit()).await.expect("commit");
    let paths = Box::pin(flusher.flush()).await.expect("flush");
    eprintln!("[setup] index built in {:.1}s", t.elapsed().as_secs_f64());

    let head_bloom_read = paths.head_bloom_blob_path.as_ref().map(|p| HeadBloomReadConfig {
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
        false,
        params,
        head_bloom_read,
        None,
    ))
    .await
    .expect("spann reader");
    let cache_len = reader.head_bloom_filters.as_ref().map(|c| c.len()).unwrap_or(0);
    eprintln!(
        "[setup] reader bloom cache len = {} (filters loaded from blob; 0 means rebuild was off or failed)",
        cache_len
    );

    let tokens_off = EqualityTokens::Unsupported;
    let tokens_on = EqualityTokens::And(vec!["meta::bucket::int::0".to_string()]);

    // Estimate drop_ratio at the largest probe_nbr to size iso-I/O probe counts.
    let max_probe_off = *probe_nbrs_off.iter().max().unwrap_or(&32);
    let warmup_q = &queries[0];
    let gt_warmup = ground_truth(&records, warmup_q, &distance_function, &allowed_set, k).await;
    let (_, heads_rng_w, heads_fetched_w, _, _) = run_query(
        &reader,
        warmup_q,
        max_probe_off,
        rng_epsilon,
        rng_factor,
        &distance_function,
        &tokens_on,
        &allowed_signed,
        &allowed_set,
        &gt_warmup,
        k,
    )
    .await;
    let drop_ratio_estimate = if heads_rng_w > 0 {
        (heads_rng_w - heads_fetched_w) as f64 / heads_rng_w as f64
    } else {
        0.0
    };
    eprintln!(
        "[setup] warmup: probe={} heads_rng={} heads_fetched={} drop_ratio={:.3}",
        max_probe_off, heads_rng_w, heads_fetched_w, drop_ratio_estimate
    );

    let probe_nbrs_on_iso_io: Vec<usize> = probe_nbrs_off
        .iter()
        .map(|&n| {
            let scale = 1.0 / (1.0 - drop_ratio_estimate.min(0.95)).max(0.05);
            ((n as f64) * scale).ceil() as usize
        })
        .collect();
    eprintln!("[setup] probe_nbrs_on_iso_io = {:?}", probe_nbrs_on_iso_io);

    let mut out = BufWriter::new(File::create(&output)?);
    writeln!(
        out,
        "dataset,n_records,dim,query_id,k,selectivity,scenario,probe_nbr_off,probe_nbr_used,heads_rng,heads_fetched,drop_ratio,recall_at_k,bad_drops,latency_ms"
    )?;

    // Aggregates per (scenario, probe_off) for the summary.
    use std::collections::BTreeMap;
    let mut agg: BTreeMap<(String, usize), (f64, f64, f64, f64, f64, usize, usize)> = BTreeMap::new();
    // (recall_sum, heads_rng_sum, heads_fetched_sum, drop_ratio_sum, latency_sum, bad_drops, n)

    let mut total_bad_drops: usize = 0;
    let queries_total = queries.len();
    for (q_idx, query) in queries.iter().enumerate() {
        let gt = ground_truth(&records, query, &distance_function, &allowed_set, k).await;

        for (i, &probe_off) in probe_nbrs_off.iter().enumerate() {
            let probe_on_io = probe_nbrs_on_iso_io[i];

            // Condition A: bloom_off (gate is no-op).
            let (recall_a, hr_a, hf_a, lat_a, bad_a) = run_query(
                &reader, query, probe_off, rng_epsilon, rng_factor,
                &distance_function, &tokens_off,
                &allowed_signed, &allowed_set, &gt, k,
            ).await;
            let dr_a = if hr_a > 0 { (hr_a - hf_a) as f64 / hr_a as f64 } else { 0.0 };
            // Condition B: bloom_on iso-probe.
            let (recall_b, hr_b, hf_b, lat_b, bad_b) = run_query(
                &reader, query, probe_off, rng_epsilon, rng_factor,
                &distance_function, &tokens_on,
                &allowed_signed, &allowed_set, &gt, k,
            ).await;
            let dr_b = if hr_b > 0 { (hr_b - hf_b) as f64 / hr_b as f64 } else { 0.0 };
            // Condition C: bloom_on iso-IO.
            let (recall_c, hr_c, hf_c, lat_c, bad_c) = run_query(
                &reader, query, probe_on_io, rng_epsilon, rng_factor,
                &distance_function, &tokens_on,
                &allowed_signed, &allowed_set, &gt, k,
            ).await;
            let dr_c = if hr_c > 0 { (hr_c - hf_c) as f64 / hr_c as f64 } else { 0.0 };

            total_bad_drops += bad_a + bad_b + bad_c;

            for (scenario, recall, hr, hf, dr, lat, bad, p_used) in [
                ("bloom_off",          recall_a, hr_a, hf_a, dr_a, lat_a, bad_a, probe_off),
                ("bloom_on_iso_probe", recall_b, hr_b, hf_b, dr_b, lat_b, bad_b, probe_off),
                ("bloom_on_iso_io",    recall_c, hr_c, hf_c, dr_c, lat_c, bad_c, probe_on_io),
            ] {
                writeln!(
                    out,
                    "{},{},{},{},{},{:.6},{},{},{},{},{},{:.4},{:.4},{},{:.3}",
                    dataset_label, n_records, dim, q_idx, k, selectivity,
                    scenario, probe_off, p_used, hr, hf, dr, recall, bad, lat
                )?;
                let entry = agg.entry((scenario.to_string(), probe_off)).or_default();
                entry.0 += recall;
                entry.1 += hr as f64;
                entry.2 += hf as f64;
                entry.3 += dr;
                entry.4 += lat;
                entry.5 += bad;
                entry.6 += 1;
            }
        }
        if (q_idx + 1) % 10 == 0 || q_idx + 1 == queries_total {
            eprintln!("[bench] {}/{} queries done", q_idx + 1, queries_total);
        }
    }

    out.flush()?;

    eprintln!();
    eprintln!("================ summary (n={} buckets={} sel={:.4}) ================", n_records, n_buckets, selectivity);
    eprintln!("                                            heads     drop_                   ");
    eprintln!("                                            fetched   ratio    recall   lat_ms  bad");
    for &probe_off in &probe_nbrs_off {
        for scenario in ["bloom_off", "bloom_on_iso_probe", "bloom_on_iso_io"] {
            if let Some(e) = agg.get(&(scenario.to_string(), probe_off)) {
                let n = e.6 as f64;
                eprintln!(
                    "  probe_off={:>3} {:>20}  hr={:>6.1}  hf={:>6.1}  dr={:.3}   r={:.4}  lat={:>6.2}  bd={}",
                    probe_off, scenario,
                    e.1 / n, e.2 / n, e.3 / n, e.0 / n, e.4 / n, e.5
                );
            }
        }
    }
    eprintln!("=====================================================================");
    eprintln!("  total bad_drops across all queries+conditions: {}", total_bad_drops);

    if total_bad_drops > 0 && panic_on_bad_drop {
        panic!("Phase 1 contract violation: {} false negatives", total_bad_drops);
    }
    eprintln!("[bench] wrote {}", output);
    Ok(())
}
