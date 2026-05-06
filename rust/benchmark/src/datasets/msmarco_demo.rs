//! MS MARCO demo cache loader.
//!
//! Reads the Python-built cache at `demo/data/msmarco_100k_rust/`:
//!   data.bin   tightly packed: u64 n, u32 dim, u32 q, ids, embeddings,
//!              topic_buckets, query_embeddings.
//!   meta.json  texts + categorical metadata + queries.
//!
//! Used by the SPANN race demo binary.

use std::{
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
};

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct DemoMeta {
    pub n: usize,
    pub q: usize,
    pub dim: usize,
    pub n_buckets: i32,
    pub texts: Vec<String>,
    pub length_buckets: Vec<String>,
    pub source_domains: Vec<String>,
    pub query_types: Vec<String>,
    pub queries: Vec<String>,
    pub query_types_q: Vec<String>,
}

/// Tightly-packed binary plus parsed JSON metadata.
pub struct MsMarcoDemoData {
    pub ids: Vec<i64>,
    pub embeddings: Vec<Vec<f32>>,        // n × dim
    pub topic_buckets: Vec<i32>,           // n
    pub query_embeddings: Vec<Vec<f32>>,   // q × dim
    pub meta: DemoMeta,
    pub dim: usize,
    pub n_buckets: i32,
}

impl MsMarcoDemoData {
    /// Load from `demo/data/msmarco_100k_rust/`.
    ///
    /// The path is taken relative to the workspace root if it isn't absolute.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = if dir.as_ref().is_absolute() {
            dir.as_ref().to_path_buf()
        } else {
            // Walk up from CARGO_MANIFEST_DIR until we find a directory containing
            // `Cargo.toml` with `[workspace]` (the repo root) — handles bench
            // invocations from arbitrary CWDs.
            let mut here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            for _ in 0..5 {
                here.pop();
                if here.join("Cargo.toml").exists() {
                    break;
                }
            }
            here.join(dir)
        };
        if !dir.is_dir() {
            return Err(anyhow!(
                "msmarco demo cache dir not found at {}",
                dir.display()
            ));
        }
        let bin_path = dir.join("data.bin");
        let meta_path = dir.join("meta.json");
        let bin = read_bytes(&bin_path)
            .with_context(|| format!("opening {}", bin_path.display()))?;
        let meta: DemoMeta = {
            let f = File::open(&meta_path)
                .with_context(|| format!("opening {}", meta_path.display()))?;
            serde_json::from_reader(BufReader::new(f))
                .with_context(|| format!("parsing {}", meta_path.display()))?
        };
        Self::from_bytes(bin, meta)
    }

    fn from_bytes(bin: Vec<u8>, meta: DemoMeta) -> Result<Self> {
        let mut cur = 0usize;

        let n = read_u64_le(&bin, &mut cur)? as usize;
        let dim = read_u32_le(&bin, &mut cur)? as usize;
        let q = read_u32_le(&bin, &mut cur)? as usize;

        if n != meta.n || dim != meta.dim || q != meta.q {
            return Err(anyhow!(
                "data.bin/meta.json mismatch: n={}/{} dim={}/{} q={}/{}",
                n,
                meta.n,
                dim,
                meta.dim,
                q,
                meta.q
            ));
        }

        let ids = read_i64_vec(&bin, &mut cur, n)?;
        let emb_flat = read_f32_vec(&bin, &mut cur, n * dim)?;
        let topic_buckets = read_i32_vec(&bin, &mut cur, n)?;
        let qemb_flat = read_f32_vec(&bin, &mut cur, q * dim)?;

        if cur != bin.len() {
            return Err(anyhow!(
                "data.bin: {} bytes left over after parse",
                bin.len() - cur
            ));
        }

        let embeddings: Vec<Vec<f32>> = emb_flat
            .chunks_exact(dim)
            .map(|c| c.to_vec())
            .collect();
        let query_embeddings: Vec<Vec<f32>> = qemb_flat
            .chunks_exact(dim)
            .map(|c| c.to_vec())
            .collect();
        let n_buckets = meta.n_buckets;

        Ok(Self {
            ids,
            embeddings,
            topic_buckets,
            query_embeddings,
            meta,
            dim,
            n_buckets,
        })
    }

    pub fn n(&self) -> usize {
        self.meta.n
    }

    pub fn q(&self) -> usize {
        self.meta.q
    }
}

fn read_bytes(path: &Path) -> Result<Vec<u8>> {
    let mut f = File::open(path)?;
    let mut v = Vec::new();
    f.read_to_end(&mut v)?;
    Ok(v)
}

fn read_u64_le(bin: &[u8], cur: &mut usize) -> Result<u64> {
    if *cur + 8 > bin.len() {
        return Err(anyhow!("eof reading u64 at {}", cur));
    }
    let v = u64::from_le_bytes(bin[*cur..*cur + 8].try_into().unwrap());
    *cur += 8;
    Ok(v)
}

fn read_u32_le(bin: &[u8], cur: &mut usize) -> Result<u32> {
    if *cur + 4 > bin.len() {
        return Err(anyhow!("eof reading u32 at {}", cur));
    }
    let v = u32::from_le_bytes(bin[*cur..*cur + 4].try_into().unwrap());
    *cur += 4;
    Ok(v)
}

fn read_i64_vec(bin: &[u8], cur: &mut usize, n: usize) -> Result<Vec<i64>> {
    let bytes = n * 8;
    if *cur + bytes > bin.len() {
        return Err(anyhow!("eof reading i64[{}] at {}", n, cur));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let off = *cur + i * 8;
        out.push(i64::from_le_bytes(bin[off..off + 8].try_into().unwrap()));
    }
    *cur += bytes;
    Ok(out)
}

fn read_i32_vec(bin: &[u8], cur: &mut usize, n: usize) -> Result<Vec<i32>> {
    let bytes = n * 4;
    if *cur + bytes > bin.len() {
        return Err(anyhow!("eof reading i32[{}] at {}", n, cur));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let off = *cur + i * 4;
        out.push(i32::from_le_bytes(bin[off..off + 4].try_into().unwrap()));
    }
    *cur += bytes;
    Ok(out)
}

fn read_f32_vec(bin: &[u8], cur: &mut usize, n: usize) -> Result<Vec<f32>> {
    let bytes = n * 4;
    if *cur + bytes > bin.len() {
        return Err(anyhow!("eof reading f32[{}] at {}", n, cur));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let off = *cur + i * 4;
        out.push(f32::from_le_bytes(bin[off..off + 4].try_into().unwrap()));
    }
    *cur += bytes;
    Ok(out)
}
