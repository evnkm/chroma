"""Convert the .npz cache into a Rust-friendly layout.

Reads `demo/data/msmarco_100k.npz` (produced by build_msmarco_demo_cache.py)
and writes:

    demo/data/msmarco_100k_rust/
        data.bin          # tightly packed binary: header + arrays
        meta.json         # texts + categorical metadata + queries

`data.bin` layout (little-endian):
    u64  n                  # number of passages
    u32  dim                # embedding dim (384)
    u32  q                  # number of demo queries
    i64  ids[n]
    f32  embeddings[n*dim]
    i32  topic_buckets[n]
    f32  query_embeddings[q*dim]

`meta.json` payload:
    {
        "n": int, "q": int, "dim": int, "n_buckets": int,
        "texts": [str, ...],            # n
        "length_buckets": [str, ...],   # n
        "source_domains": [str, ...],   # n
        "query_types": [str, ...],      # n
        "queries": [str, ...],          # q
        "query_types_q": [str, ...],    # q
    }
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="demo/data/msmarco_100k.npz")
    ap.add_argument("--out-dir", default="demo/data/msmarco_100k_rust")
    args = ap.parse_args()

    if not os.path.exists(args.inp):
        print(f"error: {args.inp} not found", file=sys.stderr)
        return 1

    print(f"[convert] reading {args.inp} ...", flush=True)
    z = np.load(args.inp, allow_pickle=True)

    ids = z["ids"].astype(np.int64)
    emb = z["embeddings"].astype(np.float32)
    texts = list(z["texts"])
    topic_buckets = z["topic_buckets"].astype(np.int32)
    length_buckets = list(z["length_buckets"])
    source_domains = list(z["source_domains"])
    query_types = list(z["query_types"])
    queries = list(z["queries"])
    query_embeddings = z["query_embeddings"].astype(np.float32)
    query_types_q = list(z["query_types_q"])
    dim = int(z["dim"])
    n_buckets = int(z["n_buckets"])

    n = ids.shape[0]
    q = len(queries)
    assert emb.shape == (n, dim), f"emb shape {emb.shape} != ({n}, {dim})"
    assert query_embeddings.shape == (q, dim), f"qemb {query_embeddings.shape} != ({q}, {dim})"
    assert topic_buckets.shape == (n,), f"topic_buckets {topic_buckets.shape} != ({n},)"
    assert len(texts) == n
    assert len(length_buckets) == n
    assert len(source_domains) == n
    assert len(query_types) == n
    assert len(query_types_q) == q

    os.makedirs(args.out_dir, exist_ok=True)

    bin_path = os.path.join(args.out_dir, "data.bin")
    print(f"[convert] writing {bin_path} ...", flush=True)
    with open(bin_path, "wb") as f:
        f.write(struct.pack("<Q", n))
        f.write(struct.pack("<I", dim))
        f.write(struct.pack("<I", q))
        f.write(ids.tobytes())
        f.write(emb.tobytes())
        f.write(topic_buckets.tobytes())
        f.write(query_embeddings.tobytes())

    meta_path = os.path.join(args.out_dir, "meta.json")
    print(f"[convert] writing {meta_path} ...", flush=True)
    payload = {
        "n": n,
        "q": q,
        "dim": dim,
        "n_buckets": n_buckets,
        "texts": texts,
        "length_buckets": length_buckets,
        "source_domains": source_domains,
        "query_types": query_types,
        "queries": queries,
        "query_types_q": query_types_q,
    }
    with open(meta_path, "w") as f:
        json.dump(payload, f)

    bin_sz = os.path.getsize(bin_path)
    meta_sz = os.path.getsize(meta_path)
    print(f"[convert] done. data.bin={bin_sz/1e6:.1f}MB meta.json={meta_sz/1e6:.1f}MB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
