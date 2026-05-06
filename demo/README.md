# SPANN race demo

Side-by-side TUI race between two SPANN indexes on the same MS MARCO 100K
corpus and the same metadata predicate, but with different opt-in features:

| Cell | Adaptive nprobe | Bloom-gate | Synopsis-gate |
|------|-----------------|------------|---------------|
| **Baseline (000)**  | off | off | off |
| **Optimized (111)** | on  | on  | on  |

For each free-text query + predicate (e.g. "show me how-to articles
about photosynthesis", `source_domain = "wikihow.com"`), the demo embeds
the query, races both cells in parallel, streams stage events on stdout,
and the TUI renders a live side-by-side panel.

## Pipeline

```
Phase 0  build_msmarco_demo_cache.py     pulls 100K MS MARCO passages, embeds with MiniLM (384d).
                                           output:  demo/data/msmarco_100k.npz
Phase 0' convert_cache_for_rust.py        flattens to a Rust-friendly (data.bin + meta.json) layout.
                                           output:  demo/data/msmarco_100k_rust/
Phase 1  rust/benchmark/.../msmarco_demo  Rust loader for that layout.
Phase 2  spann_demo_build (Rust bin)      builds the two persistent SPANN indexes from the cache.
                                           output:  demo/data/baseline/, demo/data/optimized/, demo/data/queries.json
Phase 3  spann_demo (Rust bin)            opens the two indexes, spawns the embed sidecar, REPLs JSON-line requests.
Phase 4  tui.py (Python rich)             prompts free text + predicate, drives the binary, renders the live race.
```

## Run it

```bash
# 1. Build Rust binaries (one-time):
export SDKROOT="/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"
export CPLUS_INCLUDE_PATH="$SDKROOT/usr/include/c++/v1:$SDKROOT/usr/include"
export C_INCLUDE_PATH="$SDKROOT/usr/include"
. "$HOME/.cargo/env"
export CARGO_TARGET_DIR=/tmp/chroma-target
cargo build --release -p worker --bin spann_demo --bin spann_demo_build

# 2. Cache MS MARCO 100K + convert to Rust layout (~3 min total):
source .venv/bin/activate
python demo/build_msmarco_demo_cache.py --n 100000 --queries 30 --buckets 1000
python demo/convert_cache_for_rust.py

# 3. Build the two SPANN indexes (~10 min):
/tmp/chroma-target/release/spann_demo_build

# 4. Run the TUI:
python demo/tui.py
```

## Predicates available

Realistic (from the MS MARCO `url` field, normalized to host name):

- `source_domain = "wikihow.com"` — 0.74% selectivity. **Headline realistic predicate.**
- `source_domain = "mayoclinic.org"` — 0.62%
- `source_domain = "investopedia.com"` — 0.28%

Synthetic (matches the v2 paper headline):

- `topic_bucket = 0` — 0.10% (each passage gets `id % 1000`).

Other realistic predicates available on the same indexes:

- `query_type = "PERSON"` (2.86%) / `"LOCATION"` (4.36%) / `"ENTITY"` (9.18%)
- `length_bucket = "long"` (0.17%) / `"medium"` (~50%) / `"short"` (~50%)

## Recording the demo

```bash
brew install vhs   # if not already installed
vhs demo/demo.tape
```

Outputs `demo/recordings/spann-race.gif` and `.mp4`.

## Files

- `build_msmarco_demo_cache.py` — Python: download + embed + cache → .npz.
- `convert_cache_for_rust.py` — Python: .npz → data.bin + meta.json.
- `embed_sidecar.py` — Python: stdin/stdout MiniLM embed server (long-lived).
- `tui.py` — Python: rich-based race TUI.
- `test_demo_protocol.py` — Python: non-interactive end-to-end test of the binary.
- `data/` — gitignored. Contains the 100K corpus, two indexes, and queries.json.
- `recordings/` — committed. `.gif` / `.mp4` artifacts.

Rust:

- `rust/benchmark/src/datasets/msmarco_demo.rs` — loader for the `_rust/` layout.
- `rust/worker/src/bin/demo/spann_demo_build.rs` — builds the two indexes.
- `rust/worker/src/bin/demo/spann_demo.rs` — race REPL.
