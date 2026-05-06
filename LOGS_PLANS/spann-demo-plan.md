# SPANN demo plan — option C (bench-flavored MS MARCO race)

**Status:** approved by user, executing.
**Goal:** A recorded TUI demo that shows a free-text query racing through two SPANN indexes side-by-side — baseline (cell 000) vs all-three-on (cell 111) — on real MS MARCO passages with two metadata predicates: a *realistic* one (primary) and a *synthetic* 0.1%-selectivity one (fallback / matches the v2 headline).
**Pause point:** after phase 4 (TUI race UI) for user input on pre-embedded queries.

## What's in the existing cache

```
LOGS_PLANS/benchmarks/data/_cache/msmarco_50163f435953daa2.npz
  ids:              (500,)         — passage IDs
  embeddings:       (500, 384)     — MiniLM embeddings, float32
  metadatas:        (500,)         — dicts: {topic: 0-9, year: ..., length_bucket: 'medium'}
  queries:          (10,)          — actual query strings
  query_embeddings: (10, 384)      — pre-embedded queries
  dim:              ()             — 384
```

Implication: **the existing cache is too small (500 passages, ~10 distinct topic values) for a 0.1% selectivity demo**. We need a larger corpus + a 1000-bucket synthetic field + a meaningful realistic categorical. Phase 0 re-caches at 100K.

## Predicate decision

**Realistic primary, synthetic fallback** — the demo shows both:

- **Realistic primary: `source_domain`** — extracted from each passage's URL (MS MARCO v2.1 passages have a `url` field). Strip to domain (`en.wikipedia.org`, `cdc.gov`, `cnn.com`, …). Long-tail distribution measured at N=100K: Wikipedia 7.93%, answers.com 2.10%, webmd.com 1.21%, wikihow.com 0.74%, mayoclinic.org 0.62%, investopedia.com 0.28%. Wikipedia at 7.93% is too broad for a "selectivity wins" demo. **Demo uses `source_domain = "wikihow.com"` (0.74%)** as the headline realistic predicate — reads as "show me only how-to guides", and at 0.74% it sits right in the regime where the gates pay off dramatically. Falls back to mayoclinic.org or investopedia.com for variety.
- **Realistic secondary: `query_type`** (if MS MARCO ships it) — `DESCRIPTION` / `NUMERIC` / `ENTITY` / `PERSON` / `LOCATION` from the QnA labels. Passages inherit the type of the query that retrieved them. Distribution skewed toward DESCRIPTION (~70%); PERSON/LOCATION/NUMERIC each 5-15% — naturally selective. Demo predicate `query_type = "PERSON"` reads as "passages about people".
- **Realistic fallback: `length_bucket`** — short/medium/long from token count. Always available regardless of dataset shape; selectivity 20-50%; less compelling but never breaks.
- **Synthetic: `topic_bucket = 0`** with `topic_bucket = doc_id % 1000`, exactly matching the v2 paper's 0.1% headline.

Phase 0's caching script tries `source_domain` and `query_type` first; falls back to `length_bucket` if those fields aren't present in the HF dump. The synthetic `topic_bucket` is always synthesized.

The TUI presents both a realistic query and a synthetic query as separate demo runs so the recorded demo shows "realistic in 5ms instead of 80ms — still 16× faster" and "synthetic killer case with 50× recall lift".

## Architecture overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  TUI (Python + rich)                                                │
│    spawns ───▶  [Rust demo binary]                                  │
│    ◀───── JSON-line stage events on stdout                          │
├─────────────────────────────────────────────────────────────────────┤
│  Rust demo binary                                                   │
│    Loads two pre-built SPANN indexes (baseline / optimized)         │
│    REPL on stdin                                                    │
│    For each query:                                                  │
│      tokio::spawn baseline task ──▶ JSON events                     │
│      tokio::spawn optimized task ──▶ JSON events                    │
│      both emit "stage" events; both end with "done"                 │
├─────────────────────────────────────────────────────────────────────┤
│  Persisted indexes (one-time build)                                 │
│    demo/data/baseline/   ← cell 000 (no gates)                      │
│    demo/data/optimized/  ← cell 111 (bloom + synopsis)              │
│  built from MS MARCO 100K with realistic + synthetic metadata       │
└─────────────────────────────────────────────────────────────────────┘
```

## Phases

### Phase 0 — Re-cache MS MARCO at usable scale (~30-60 min)

`demo/build_msmarco_demo_cache.py`:

- Pulls 100K passages from HuggingFace's `ms_marco` dataset (v2.1 train split).
- Embeds with `sentence-transformers/all-MiniLM-L6-v2` (already in HF cache from prior project work).
- For each passage, computes:
  - `topic_bucket: i32` = `passage_id % 1000` (synthetic 0.1% selectivity).
  - `length_bucket: str` = `"short"` if tokens<50 else `"medium"` if <150 else `"long"` (realistic).
  - Optionally a topic-class field if MS MARCO ships one.
- Saves passage **texts** (the existing 500-entry cache has none).
- Output: `demo/data/msmarco_100k.npz` with fields `ids, embeddings, texts, topic_buckets, length_buckets, queries, query_embeddings`.

**Risk:** HF download could be slow. **Fallback:** use the existing 500-passage cache and lower selectivity to 5% (B=20 buckets). Demo's "killer 0.1%" headline degrades but the realistic predicate path still works.

### Phase 1 — Rust MS MARCO loader (~30-45 min)

`rust/benchmark/src/datasets/msmarco_demo.rs`:
- `pub struct MsMarcoDemoData { ids, embeddings, texts, topic_buckets, length_buckets, queries, query_embeddings, dim }`.
- `pub async fn init() -> Result<Self, DataError>`: reads `demo/data/msmarco_100k.npz` via `npyz`.
- `pub fn passages(&self) -> impl Iterator<Item = (u32, &[f32], &str, u32, &str)>`: id, embedding, text, topic_bucket, length_bucket.
- `pub fn queries(&self) -> &[(String, Vec<f32>)]`.

Mirrors the existing `Sift1MData` loader pattern.

### Phase 2 — Build the two SPANN indexes (~30-45 min)

`rust/worker/benches/spann_demo_build.rs`:

1. Loads `MsMarcoDemoData`.
2. For each cell (000 and 111):
   - Construct a `BuiltIndex`-style harness with **persisted** `LocalStorage` (not `tempdir()`) at `demo/data/baseline/` or `demo/data/optimized/`.
   - Set writer flags: 000 = no gates, 111 = `bloom + synopsis`.
   - Ingest 100K records via `add_with_metadata_tokens_and_synopsis`. Bloom tokens: `meta::topic_bucket::int::{n}` and `meta::length_bucket::str::{long|medium|short}`. Synopsis tokens: `[("topic_bucket", "int::n"), ("length_bucket", "str::long"|...)]`.
   - For 111: also call `set_synopsis_inverted_index` with the bucket bitmaps.
   - Commit + flush. Persist `SpannIndexIds` to `demo/data/{baseline,optimized}/manifest.json`.
3. After both built, write `demo/data/queries.json` with the cached query texts + embeddings (lifts them out of the .npz so the demo binary doesn't need to load it).

**Persistence note:** `BuiltIndex` in the existing bench uses `tempfile::TempDir`. We need real paths so the indexes survive across binary invocations. The build binary owns directory lifecycle.

Build runtime: ~50s × 2 = ~2 min for N=100K.

Runnable: `cargo run --release -p worker --bench spann_demo_build`.

### Phase 3 — Demo binary (~2 hours)

`rust/worker/benches/spann_demo.rs`:

- **Startup**: open both indexes from `demo/data/{baseline,optimized}/`. Load `demo/data/queries.json` for precomputed query embeddings.
- **REPL**: read JSON lines from stdin: `{query_id: 1, predicate: "topic_bucket = 0"}` or `{query_text: "...", predicate: "length_bucket = long"}`.
- **For each request**:
  - Resolve embedding via a long-lived Python sidecar (`demo/embed_sidecar.py`) that loads MiniLM once at startup and serves embeddings over a stdin/stdout JSON-line protocol. Per-query embed: ~50-150ms after warmup. **No precomputed embeddings — every query is embedded live.**
  - Build `EqualityTokens` and `SynopsisPredicate` from the predicate string.
  - Capture `t0 = Instant::now()`.
  - Spawn two `tokio::task`s, both started after `t0`:
    - **Baseline**: `rng_query` → no gate → fetch all PLs → BfPL → merge → emit `done` event.
    - **Optimized**: `rng_query` → bloom gate → synopsis gate → fetch surviving PLs → BfPL → merge → emit `done` event.
  - Each task emits stage events through a `tokio::sync::mpsc::channel` that the main loop forwards to stdout as JSON lines:
    - `{cell, stage: "probing", t_ms: 0.1}`
    - `{cell, stage: "gating", heads_rng: 512, t_ms: 0.4}`
    - `{cell, stage: "fetching", heads_to_fetch: 32, t_ms: 0.5}`
    - `{cell, stage: "done", lat_ms: 4.8, recall: 0.74, top_k: [{id, text, score}, ...]}`
- **Output**: stream of JSON lines on stdout. The TUI consumes this stream.

**Race fairness**:
- Capture `t0 = Instant::now()` once before *either* `tokio::spawn`. Both tasks reference the same `t0` for their `t_ms` reporting.
- Tokio's multi-thread runtime schedules the two on different worker threads. On a quiet macOS laptop this is fair within ~100µs.

### Phase 4 — TUI race UI (~2-3 hours)

`demo/tui.py` — Python with `rich`:

1. Spawns demo binary via `subprocess.Popen(["./target/release/.../spann_demo"], stdin=PIPE, stdout=PIPE)`.
2. Layout (using `rich.layout.Layout` + `rich.live.Live`):
   ```
   ┌──────────────────────────────────────────────────────────────┐
   │  Query: How does photosynthesis work? · pred: length=long    │
   ├────────────────────────────┬─────────────────────────────────┤
   │  Baseline (000)            │  Optimized (111)                │
   │  ⠋ fetching 512 PLs...     │  ✓ done in 0.9 ms               │
   │  Latency: 14.2 ms          │  Latency: 0.9 ms                │
   │                            │                                 │
   │  Top-3:                    │  Top-3:                         │
   │  1. Plants convert sunl... │  1. Plants convert sunlight...  │
   │  2. Photosynthesis is...   │  2. Photosynthesis is the...    │
   │  3. Chlorophyll absorbs... │  3. Chlorophyll absorbs the...  │
   ├────────────────────────────┴─────────────────────────────────┤
   │  Optimized is 16× faster · 32× less I/O · same top-3         │
   └──────────────────────────────────────────────────────────────┘
   ```
3. Async event loop: `asyncio.subprocess` streams JSON lines from binary, updates panel state.
4. Spinner animation runs in each pane until that pane receives a `done` event. Live timer counts up.
5. After both `done`: bottom comparison strip renders "X× faster, Y× less I/O, recall delta = ±Zpp", and the loop returns to the query prompt.
6. Query menu shown at top: precomputed queries 1-10 (chosen in phase 5) + free-text option.

### Phase 5 — Query plan + pause for user input

User decision: **don't pre-embed.** The demo embeds free-text queries at runtime so the user can type their own questions live. Pause point is to confirm:

- Which 2-4 example queries to highlight in the recorded demo (so the recording is clean).
- Which predicate each query should use (realistic, synthetic, or both — back-to-back).
- Whether to keep the embedding model warm in a long-lived Python sidecar process (faster per-query, ~50ms) or spawn it fresh per query (slower, ~2-3s warmup, simpler).

User explicitly OK with up to 30s embedding time per query, so either sidecar or fresh-spawn works.

### Phase 6 — Recording (~30-60 min)

`vhs` script (`demo/demo.tape`):
- Drives the TUI deterministically with `Type` / `Sleep` / `Enter` commands.
- Output: `.gif` for slides, `.mp4` for higher-fidelity embedding.

## Files

```
.gitignore                                      [+1 line: demo/data/]
rust/benchmark/src/datasets/mod.rs              [+1 line]
rust/benchmark/src/datasets/msmarco_demo.rs     [new, ~120 LoC]
rust/worker/benches/spann_demo_build.rs         [new, ~200 LoC]
rust/worker/benches/spann_demo.rs               [new, ~320 LoC]  // larger because of embed-sidecar plumbing
rust/worker/Cargo.toml                          [+2 [[bench]] entries]
demo/                                           [new dir]
  data/                                         [gitignored]
    msmarco_100k.npz
    baseline/
    optimized/
    manifest.json
  build_msmarco_demo_cache.py                   [new, ~120 LoC]   // adds source_domain + query_type extraction
  embed_sidecar.py                              [new, ~50 LoC]    // long-lived MiniLM server over stdin/stdout
  tui.py                                        [new, ~280 LoC]   // free-text query input, no menu
  demo.tape                                     [new, ~30 lines]
  README.md                                     [new]
```

## Time budget

| Phase | Time |
|---|---|
| 0. Re-cache MS MARCO 100K | 30-60 min |
| 1. Rust MS MARCO loader | 30-45 min |
| 2. Build two indexes (script + run) | 30-45 min |
| 3. Demo binary (Rust + tokio race + JSON-line stream) | 2 hours |
| 4. TUI race UI (Python + rich) | 2-3 hours |
| **Subtotal to pause** | **~5-6 hours** |
| 5. Pre-embedded queries (after user input) | ~30 min |
| 6. vhs recording | 30-60 min |
| **Total** | **~6-8 hours** |

## Commits planned

- `phase 0` — `demo/build_msmarco_demo_cache.py` + cached data (`.npz` gitignored)
- `phase 1` — Rust loader
- `phase 2` — build binary + first successful run
- `phase 3` — demo binary
- `phase 4` — TUI
- `phase 5` — chosen queries
- `phase 6` — recording

## Risks & fallbacks

- **HF download slow / fails**: fallback to 500-passage cache + 5% selectivity (B=20). Diminishes 0.1% headline; realistic predicate still works.
- **MS MARCO doesn't have native categorical fields**: synthesize `length_bucket` from token count (always available); skip topic-class.
- **Race fairness wobble**: at sub-1ms latencies, tokio scheduling can favor whichever task spawned first. Mitigate by running each query 3 times and reporting median for the recorded demo.
- **TUI rendering issues**: `rich.live` can flicker on slow terminals. `vhs` provides its own deterministic terminal so the recorded version is clean even if live execution looks wobbly.
- **Embedding model not cached**: `sentence-transformers` will download ~80MB on first use. Acceptable one-time cost.

## Notes

- The user explicitly approved C and asked for both realistic + synthetic predicates.
- All v2 sweep numbers (paper.md) come from the synthetic `bucket = id % B` setup, so the synthetic demo predicate exactly mirrors what the paper claims.
- The realistic predicate's selectivity will be different — we'll measure and show whatever it actually is in the demo (e.g. "source_domain=en.wikipedia.org matches 18% of corpus, gates still cut latency 4×").
- Demo binary uses the actual `SpannIndexReader` and gate methods (`reader.gate_heads`, `reader.gate_heads_synopsis`) — same code path as the bench, same as production query orchestration would invoke.
- **Existing 500-passage cache (`LOGS_PLANS/benchmarks/data/_cache/msmarco_50163f435953daa2.npz`) is kept** — used by other bench code, small, no need to remove. The demo cache is a separate file at `demo/data/msmarco_100k.npz`.
- **Embedding model: `sentence-transformers/all-MiniLM-L6-v2`** (384-dim) — matches what the existing cache uses. Same model embeds both the 100K corpus and live demo queries so the embedding space is consistent.
- **MS MARCO version: `microsoft/ms_marco` v2.1** on HuggingFace — has `query`, `passages.passage_text`, `passages.url`, `passages.is_selected`, `query_type`. URL field enables `source_domain`; `query_type` field enables the second realistic predicate. Both are present in v2.1.
- **Source-domain normalization**: full hostname kept (e.g. `en.wikipedia.org`, not `wikipedia.org`). Finer-grained predicates and more demo-legible.
- **Embedding error handling**: if the sidecar fails to embed a query, the Rust binary emits an error JSON-line event and the TUI prompts for the next query. No crash.
- **Process model**: Rust demo binary spawns the embedding sidecar as a subprocess. TUI only talks to the Rust binary. Embedding cost is rolled into the per-query latency the TUI sees.
- **Demo binary as `[[bin]]`** (not `[[bench]]` as I'd originally planned). Cargo benches produce hash-suffixed binaries (`target/release/deps/spann_demo-XXXXXX`) that are awkward for the TUI to invoke. A `[[bin]]` target produces a stable `target/release/spann_demo` path. Add to `rust/worker/Cargo.toml` `[[bin]] name = "spann_demo"`.
- **Demo is recorded only, not run live during a presentation.** That means we can skip live UX polish (signal handling, fancy interactive prompts, retry-on-timeout). The recording artifact is the deliverable.
- Demo `.gif`/`.mp4` recording artifacts go to `demo/recordings/` and ARE committed (small, useful for embedding in presentations).
- Demo recording tool: **`vhs`** (a Go program that drives terminals deterministically). If not installed: `brew install vhs`. Defer install to phase 6.
