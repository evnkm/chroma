"""Build the demo MS MARCO 100K cache for the SPANN race demo.

Pulls passages from MS MARCO v2.1 (HuggingFace `microsoft/ms_marco`), embeds
them with `sentence-transformers/all-MiniLM-L6-v2` (384-d), and writes a
single .npz with everything the Rust loader needs.

Output schema (`demo/data/msmarco_100k.npz`):
    ids:               int64[N]
    embeddings:        float32[N, 384]
    texts:             object[N]   passage strings
    topic_buckets:     int32[N]    synthetic, id % B (B from --buckets)
    length_buckets:    object[N]   "short"/"medium"/"long" by token count
    source_domains:    object[N]   normalized URL hostname (or "unknown")
    query_types:       object[N]   MS MARCO query_type carried from the query
    queries:           object[Q]   query strings
    query_embeddings:  float32[Q, 384]
    query_types_q:     object[Q]   query_type for each demo query
    dim:               int64       384
    n_buckets:         int32       e.g. 1000

Run:
    python demo/build_msmarco_demo_cache.py \
        --n 100000 --queries 20 --buckets 1000 \
        --out demo/data/msmarco_100k.npz
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from urllib.parse import urlparse

import numpy as np
from tqdm import tqdm


def _norm_domain(url: str) -> str:
    if not url:
        return "unknown"
    try:
        p = urlparse(url if url.startswith(("http://", "https://")) else "http://" + url)
        host = (p.hostname or "").lower()
        if not host:
            return "unknown"
        # strip "www." but keep finer-grained subdomains (en.wikipedia.org).
        if host.startswith("www."):
            host = host[4:]
        return host or "unknown"
    except Exception:
        return "unknown"


def _length_bucket(text: str) -> str:
    n = len(text.split())
    if n < 50:
        return "short"
    if n < 150:
        return "medium"
    return "long"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000, help="number of passages to cache")
    ap.add_argument("--queries", type=int, default=20, help="number of demo queries")
    ap.add_argument("--buckets", type=int, default=1000, help="modulus B for synthetic topic_bucket = id % B")
    ap.add_argument("--out", type=str, default="demo/data/msmarco_100k.npz")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--dataset", type=str, default="microsoft/ms_marco")
    ap.add_argument("--config", type=str, default="v2.1")
    ap.add_argument("--split", type=str, default="train")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    print(f"[cache] target N={args.n:,} Q={args.queries} buckets={args.buckets}", flush=True)

    # Lazy imports — heavy.
    print("[cache] importing datasets + sentence-transformers ...", flush=True)
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    print(f"[cache] loading {args.dataset} {args.config} split={args.split} (streaming) ...", flush=True)
    ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True)

    # Each row in ms_marco is a query bundle:
    #   {"query": str, "query_id": int, "query_type": str,
    #    "passages": {"is_selected": [...], "passage_text": [...], "url": [...]},
    #    "answers": [...], "wellFormedAnswers": [...]}
    # We flatten passages but keep each passage's source query_type & url.
    ids: list[int] = []
    texts: list[str] = []
    urls: list[str] = []
    query_types: list[str] = []

    demo_queries: list[str] = []
    demo_query_types: list[str] = []

    seen_ids: set[int] = set()
    pid_counter = 0  # synthetic id, since v2.1 ms_marco rows don't carry per-passage ids
    n_target = args.n
    n_query_target = args.queries

    t0 = time.time()
    pbar = tqdm(total=n_target, desc="passages")
    for row in ds:
        if len(ids) >= n_target and len(demo_queries) >= n_query_target:
            break

        passages = row.get("passages") or {}
        ptexts = passages.get("passage_text") or []
        purls = passages.get("url") or []
        qtype = (row.get("query_type") or "UNKNOWN").upper()

        # Capture demo queries: take the first n_query_target rows that have a
        # non-trivial query string.
        if len(demo_queries) < n_query_target:
            q = (row.get("query") or "").strip()
            if q and 3 <= len(q.split()) <= 32:
                demo_queries.append(q)
                demo_query_types.append(qtype)

        # Then accumulate passages.
        if len(ids) < n_target:
            for i, t in enumerate(ptexts):
                if not t or not t.strip():
                    continue
                pid = pid_counter
                pid_counter += 1
                if pid in seen_ids:
                    continue
                seen_ids.add(pid)
                ids.append(pid)
                texts.append(t)
                urls.append(purls[i] if i < len(purls) else "")
                query_types.append(qtype)
                pbar.update(1)
                if len(ids) >= n_target:
                    break
    pbar.close()

    n = len(ids)
    q = len(demo_queries)
    print(f"[cache] gathered {n:,} passages, {q} demo queries in {time.time()-t0:.1f}s", flush=True)

    # Derive metadata fields.
    print("[cache] deriving metadata fields ...", flush=True)
    ids_arr = np.array(ids, dtype=np.int64)
    topic_buckets = (ids_arr % args.buckets).astype(np.int32)
    length_buckets = np.array([_length_bucket(t) for t in texts], dtype=object)
    source_domains = np.array([_norm_domain(u) for u in urls], dtype=object)
    query_types_arr = np.array(query_types, dtype=object)

    # Print quick selectivity report so we can pick a realistic predicate.
    from collections import Counter
    dom_top = Counter(source_domains.tolist()).most_common(8)
    print(f"[cache] top source_domains:")
    for d, c in dom_top:
        print(f"  {d:30s} {c:>7d}  ({100.0*c/n:.2f}%)")
    qt_top = Counter(query_types_arr.tolist()).most_common(8)
    print(f"[cache] query_type distribution (passages):")
    for q_, c in qt_top:
        print(f"  {q_:20s} {c:>7d}  ({100.0*c/n:.2f}%)")
    bkt_zero = int((topic_buckets == 0).sum())
    print(f"[cache] synthetic topic_bucket=0 hits {bkt_zero} passages "
          f"(target sel={1.0/args.buckets:.5f}, observed={bkt_zero/n:.5f})")

    # Embed corpus.
    print("[cache] loading MiniLM embedder ...", flush=True)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    dim = model.get_sentence_embedding_dimension()
    print(f"[cache] embedding {n:,} passages (batch={args.batch_size}, dim={dim}) ...", flush=True)
    emb = model.encode(
        texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=False,
    ).astype(np.float32)

    # Embed queries.
    print(f"[cache] embedding {q} demo queries ...", flush=True)
    q_emb = model.encode(
        demo_queries,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    ).astype(np.float32)

    # Save.
    print(f"[cache] writing {args.out} ...", flush=True)
    np.savez(
        args.out,
        ids=ids_arr,
        embeddings=emb,
        texts=np.array(texts, dtype=object),
        topic_buckets=topic_buckets,
        length_buckets=length_buckets,
        source_domains=source_domains,
        query_types=query_types_arr,
        queries=np.array(demo_queries, dtype=object),
        query_embeddings=q_emb,
        query_types_q=np.array(demo_query_types, dtype=object),
        dim=np.int64(dim),
        n_buckets=np.int32(args.buckets),
    )
    sz = os.path.getsize(args.out)
    print(f"[cache] done. {sz/1e6:.1f} MB on disk; total time {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
