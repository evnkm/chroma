// Phase 2 of the SPANN race demo: build two persistent SPANN indexes from
// the cached MS MARCO 100K corpus (demo/data/msmarco_100k_rust/):
//
//   demo/data/baseline/   — cell 000 (no gates).
//   demo/data/optimized/  — cell 111 (bloom + synopsis, adaptive nprobe).
//
// Each index dir holds:
//   storage/      LocalStorage root (block files, hnsw, blobs).
//   manifest.json {pl_id, versions_map_id, max_head_id_id, hnsw_id,
//                  prefix_path, head_bloom_blob_path,
//                  head_synopsis_blob_path, dim, n_records,
//                  collection_id, params}.
//
// And we additionally write demo/data/queries.json with the cached query
// embeddings + texts so the demo binary doesn't need to load the .bin.
//
// Run:
//   cargo run --release -p worker --bin spann_demo_build
// Override paths or N with env vars:
//   DEMO_CACHE_DIR=demo/data/msmarco_100k_rust
//   DEMO_OUT_BASELINE=demo/data/baseline
//   DEMO_OUT_OPTIMIZED=demo/data/optimized
//   DEMO_QUERIES_OUT=demo/data/queries.json
//   DEMO_LIMIT=100000   (cap N for quick test runs)

use std::{
    collections::HashMap,
    env,
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use chroma_benchmark::{
    benchmark::tokio_multi_thread,
    datasets::msmarco_demo::MsMarcoDemoData,
};
use chroma_blockstore::{
    arrow::{config::BlockManagerConfig, provider::ArrowBlockfileProvider},
    provider::BlockfileProvider,
};
use chroma_cache::{new_cache_for_test, new_non_persistent_cache_for_test};
use chroma_config::{registry::Registry, Configurable};
use chroma_index::{
    config::{HnswGarbageCollectionConfig, PlGarbageCollectionConfig},
    hnsw_provider::HnswIndexProvider,
    spann::{
        head_bloom::HeadBloomWriteConfig,
        head_synopsis::{HeadSynopsisWriteConfig, InvertedIndexSnapshot, SynopsisToken},
        types::{GarbageCollectionContext, SpannIndexIds, SpannIndexWriter, SpannMetrics},
    },
};
use chroma_storage::{local::LocalStorage, Storage};
use chroma_types::{CollectionUuid, InternalSpannConfiguration};
use roaring::RoaringBitmap;

fn env_or<T: Into<String>>(key: &str, default: T) -> String {
    env::var(key).unwrap_or_else(|_| default.into())
}

fn env_parse<T: std::str::FromStr>(key: &str, default: T) -> T {
    env::var(key)
        .ok()
        .and_then(|s| s.parse::<T>().ok())
        .unwrap_or(default)
}

fn main() {
    let runtime = tokio_multi_thread();
    runtime.block_on(async {
        if let Err(e) = run().await {
            eprintln!("[demo-build] error: {e:#}");
            std::process::exit(1);
        }
    });
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cache_dir = env_or("DEMO_CACHE_DIR", "demo/data/msmarco_100k_rust");
    let out_baseline = PathBuf::from(env_or("DEMO_OUT_BASELINE", "demo/data/baseline"));
    let out_optimized = PathBuf::from(env_or("DEMO_OUT_OPTIMIZED", "demo/data/optimized"));
    let queries_out = PathBuf::from(env_or("DEMO_QUERIES_OUT", "demo/data/queries.json"));
    let limit = env_parse::<usize>("DEMO_LIMIT", usize::MAX);
    // Bloom: ~4 metadata tokens per doc × split_threshold (50) docs/centroid =
    // ~200 tokens/centroid. With capacity_factor=4 the bloom is saturated and
    // gates drop nothing. Use 64× → capacity 3200/centroid → headroom for
    // RNG replication (a doc can land in 1-3 centroids' blooms) and FPR ≪ 0.1%.
    let bloom_capacity_factor: u32 = env_parse("BLOOM_CAPACITY_FACTOR", 64u32);
    // Synopsis: source_domain has ~19k distinct values in MS MARCO 100K. To
    // gate on rare values without losing them under top-K truncation, push
    // max_cardinality above the distinct-value count so source_domain stays
    // in the "track all exactly" regime (no "other_counts" pollution that
    // forces the gate to fall back to "keep").
    let synopsis_top_k: u32 = env_parse("SYNOPSIS_TOP_K", 256u32);
    let synopsis_max_card: u32 = env_parse("SYNOPSIS_MAX_CARD", 50000u32);

    eprintln!("[demo-build] cache={cache_dir}");
    eprintln!("[demo-build] out_baseline={}", out_baseline.display());
    eprintln!("[demo-build] out_optimized={}", out_optimized.display());

    eprintln!("[demo-build] loading MS MARCO cache ...");
    let data = MsMarcoDemoData::load(&cache_dir)?;
    let dim = data.dim;
    let n_total = data.n();
    let n = n_total.min(limit);
    eprintln!(
        "[demo-build] loaded n={} (using {}) dim={} q={} buckets={}",
        n_total, n, dim, data.q(), data.n_buckets
    );

    let mut params = InternalSpannConfiguration::default();
    // Honor a non-trivial nprobe budget — the demo wants to show the gate's
    // effect on a meaningfully wide candidate set, not a 1-head probe.
    params.search_nprobe = 32;

    // Build cell 000 (baseline) and cell 111 (optimized).
    eprintln!("[demo-build] building baseline (cell 000) ...");
    let baseline = build_one(
        &out_baseline,
        BuildFlavor::Baseline,
        &data,
        n,
        params.clone(),
        bloom_capacity_factor,
        synopsis_top_k,
        synopsis_max_card,
    )
    .await?;

    eprintln!("[demo-build] building optimized (cell 111) ...");
    let optimized = build_one(
        &out_optimized,
        BuildFlavor::Optimized,
        &data,
        n,
        params.clone(),
        bloom_capacity_factor,
        synopsis_top_k,
        synopsis_max_card,
    )
    .await?;

    // Write queries.json (for the demo binary to load without re-reading the .bin).
    eprintln!("[demo-build] writing {} ...", queries_out.display());
    fs::create_dir_all(queries_out.parent().unwrap_or(Path::new(".")))?;
    let mut q_writer = BufWriter::new(File::create(&queries_out)?);
    let mut entries: Vec<serde_json::Value> = Vec::with_capacity(data.q());
    for (i, q) in data.meta.queries.iter().enumerate() {
        let emb: &Vec<f32> = &data.query_embeddings[i];
        entries.push(serde_json::json!({
            "id": i,
            "text": q,
            "query_type": data.meta.query_types_q[i],
            "embedding": emb,
        }));
    }
    let payload = serde_json::json!({
        "dim": dim,
        "n_buckets": data.n_buckets,
        "queries": entries,
    });
    q_writer.write_all(serde_json::to_string(&payload)?.as_bytes())?;
    q_writer.flush()?;

    eprintln!("[demo-build] done.");
    eprintln!(
        "[demo-build]   baseline:  build={:.0} ms blobs={} bytes",
        baseline.build_ms, baseline.blob_bytes
    );
    eprintln!(
        "[demo-build]   optimized: build={:.0} ms blobs={} bytes",
        optimized.build_ms, optimized.blob_bytes
    );
    eprintln!("[demo-build] manifests at {}/manifest.json and {}/manifest.json",
        out_baseline.display(), out_optimized.display());
    Ok(())
}

#[derive(Clone, Copy)]
enum BuildFlavor {
    /// Cell 000 — no bloom, no synopsis. Adaptive=off (baseline gets fixed nprobe).
    Baseline,
    /// Cell 111 — bloom + synopsis, adaptive=on.
    Optimized,
}

struct BuildResult {
    build_ms: f64,
    blob_bytes: u64,
}

async fn build_one(
    out_dir: &Path,
    flavor: BuildFlavor,
    data: &MsMarcoDemoData,
    n: usize,
    params: InternalSpannConfiguration,
    bloom_capacity_factor: u32,
    synopsis_top_k: u32,
    synopsis_max_card: u32,
) -> Result<BuildResult, Box<dyn std::error::Error>> {
    let bloom_enabled = matches!(flavor, BuildFlavor::Optimized);
    let synopsis_enabled = matches!(flavor, BuildFlavor::Optimized);

    // Reset the storage dir so we don't half-merge with prior runs.
    if out_dir.exists() {
        fs::remove_dir_all(out_dir)?;
    }
    fs::create_dir_all(out_dir)?;
    let storage_root = out_dir.join("storage");
    fs::create_dir_all(&storage_root)?;
    let storage = Storage::Local(LocalStorage::new(storage_root.to_str().unwrap()));

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

    let head_bloom_config = if bloom_enabled {
        let capacity = (params.split_threshold as u64)
            .saturating_mul(bloom_capacity_factor.max(1) as u64)
            .max(1);
        Some(HeadBloomWriteConfig {
            capacity_per_head: capacity,
            existing_blob_path: None,
            // Doc-tokens cache + commit-rebuild are required for bloom to
            // survive centroid splits during indexing. Without them,
            // every reassign during a split calls `handle_reassign_bloom`
            // which marks the head stale (since metadata_tokens is None
            // on the reassign path), and `iter_non_stale()` ends up
            // returning zero filters — the persisted bloom blob is empty
            // and the gate degrades to keep-everything. With doc_tokens
            // enabled the tokens are remembered per doc; with
            // commit_rebuild enabled, every touched head's bloom is
            // rebuilt from PL × doc-tokens at commit time.
            doc_tokens_cache_enabled: true,
            commit_rebuild_enabled: true,
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
        "",
        data.dim,
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

    let t0 = Instant::now();
    eprintln!("[demo-build:{:?}] adding {} records ...", flavor_name(flavor), n);
    let log_every = (n / 20).max(1);
    for i in 0..n {
        // Use sequential ids 1..=n. id i+1 corresponds to data.* index i.
        let id = (i as u32) + 1;
        let emb = &data.embeddings[i];
        let topic_bucket = data.topic_buckets[i];
        let length_bucket = &data.meta.length_buckets[i];
        let source_domain = &data.meta.source_domains[i];
        let query_type = &data.meta.query_types[i];

        let bloom_tokens: Vec<String> = if bloom_enabled {
            vec![
                format!("meta::topic_bucket::int::{}", topic_bucket),
                format!("meta::length_bucket::str::{}", length_bucket),
                format!("meta::source_domain::str::{}", source_domain),
                format!("meta::query_type::str::{}", query_type),
            ]
        } else {
            Vec::new()
        };
        let synopsis_tokens: Vec<SynopsisToken> = if synopsis_enabled {
            vec![
                ("topic_bucket".to_string(), format!("int::{}", topic_bucket)),
                ("length_bucket".to_string(), format!("str::{}", length_bucket)),
                ("source_domain".to_string(), format!("str::{}", source_domain)),
                ("query_type".to_string(), format!("str::{}", query_type)),
            ]
        } else {
            Vec::new()
        };
        writer
            .add_with_metadata_tokens_and_synopsis(
                id,
                emb.as_slice(),
                &bloom_tokens,
                &synopsis_tokens,
            )
            .await
            .expect("add");

        if (i + 1) % log_every == 0 {
            eprintln!(
                "[demo-build:{:?}]   {}/{} records ({:.0} rec/s)",
                flavor_name(flavor),
                i + 1,
                n,
                (i + 1) as f64 / t0.elapsed().as_secs_f64()
            );
        }
    }

    if synopsis_enabled {
        eprintln!("[demo-build:{:?}] inserting synopsis inverted index ...", flavor_name(flavor));
        let mut snap = InvertedIndexSnapshot::new();
        let mut by_topic: HashMap<i32, RoaringBitmap> = HashMap::new();
        let mut by_length: HashMap<String, RoaringBitmap> = HashMap::new();
        let mut by_source: HashMap<String, RoaringBitmap> = HashMap::new();
        let mut by_qtype: HashMap<String, RoaringBitmap> = HashMap::new();
        for i in 0..n {
            let id = (i as u32) + 1;
            by_topic
                .entry(data.topic_buckets[i])
                .or_default()
                .insert(id);
            by_length
                .entry(data.meta.length_buckets[i].clone())
                .or_default()
                .insert(id);
            by_source
                .entry(data.meta.source_domains[i].clone())
                .or_default()
                .insert(id);
            by_qtype
                .entry(data.meta.query_types[i].clone())
                .or_default()
                .insert(id);
        }
        for (k, bm) in by_topic {
            snap.insert_int("topic_bucket", k as u32, bm);
        }
        for (k, bm) in by_length {
            snap.insert_str("length_bucket", &k, bm);
        }
        for (k, bm) in by_source {
            snap.insert_str("source_domain", &k, bm);
        }
        for (k, bm) in by_qtype {
            snap.insert_str("query_type", &k, bm);
        }
        writer.set_synopsis_inverted_index(snap).await;
    }

    eprintln!("[demo-build:{:?}] commit + flush ...", flavor_name(flavor));
    let flusher = Box::pin(writer.commit()).await.expect("commit");
    let paths: SpannIndexIds = Box::pin(flusher.flush()).await.expect("flush");
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Capture blob sizes (for the manifest summary).
    let mut blob_bytes = 0u64;
    for p in [&paths.head_bloom_blob_path, &paths.head_synopsis_blob_path] {
        if let Some(path) = p.as_ref() {
            if let Ok(bytes) = storage
                .get(
                    path,
                    chroma_storage::GetOptions::new(
                        chroma_storage::admissioncontrolleds3::StorageRequestPriority::P0,
                    ),
                )
                .await
            {
                blob_bytes = blob_bytes.saturating_add(bytes.len() as u64);
            }
        }
    }

    // Persist manifest so the demo binary can reopen without rebuilding.
    let manifest_path = out_dir.join("manifest.json");
    let manifest = serde_json::json!({
        "flavor": flavor_name(flavor),
        "dim": data.dim,
        "n_records": n,
        "collection_id": collection_id.to_string(),
        "prefix_path": paths.prefix_path,
        "pl_id": paths.pl_id.to_string(),
        "versions_map_id": paths.versions_map_id.to_string(),
        "max_head_id_id": paths.max_head_id_id.to_string(),
        "hnsw_id": paths.hnsw_id.0.to_string(),
        "head_bloom_blob_path": paths.head_bloom_blob_path,
        "head_synopsis_blob_path": paths.head_synopsis_blob_path,
        "build_ms": build_ms,
        "blob_bytes": blob_bytes,
        "params": {
            "search_nprobe": params.search_nprobe,
            "split_threshold": params.split_threshold,
            "ef_search": params.ef_search,
            "search_rng_epsilon": params.search_rng_epsilon,
            "search_rng_factor": params.search_rng_factor,
        },
        "adaptive_nprobe": matches!(flavor, BuildFlavor::Optimized),
        "bloom_enabled": bloom_enabled,
        "synopsis_enabled": synopsis_enabled,
    });
    fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?)?;
    eprintln!(
        "[demo-build:{:?}] wrote {} (build={:.0} ms, blobs={}B)",
        flavor_name(flavor),
        manifest_path.display(),
        build_ms,
        blob_bytes
    );

    Ok(BuildResult {
        build_ms,
        blob_bytes,
    })
}

fn flavor_name(f: BuildFlavor) -> &'static str {
    match f {
        BuildFlavor::Baseline => "baseline",
        BuildFlavor::Optimized => "optimized",
    }
}
