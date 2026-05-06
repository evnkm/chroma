"""Long-lived MiniLM embedding sidecar for the SPANN demo binary.

Spawned by the Rust demo binary as a subprocess. Reads JSON-line requests
from stdin, writes JSON-line responses to stdout. Loads the model once at
startup so per-query latency is ~50–150ms.

Protocol:
    request:  {"id": int, "text": str}
    response: {"id": int, "embedding": [float, ...]}   # on success
              {"id": int, "error": str}                # on failure
    Special: {"ready": true}                            # emitted once after load.

Model: sentence-transformers/all-MiniLM-L6-v2 (384-d).
"""

from __future__ import annotations

import json
import sys
import traceback


def main() -> int:
    print(json.dumps({"loading": True}), flush=True)
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        print(json.dumps({"error": f"failed to import sentence_transformers: {e}"}), flush=True)
        return 1

    try:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        print(json.dumps({"error": f"failed to load MiniLM: {e}"}), flush=True)
        return 1

    print(json.dumps({"ready": True, "dim": int(model.get_sentence_embedding_dimension())}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            print(json.dumps({"error": f"invalid json: {e}"}), flush=True)
            continue

        rid = req.get("id", -1)
        text = req.get("text", "")
        if not isinstance(text, str) or not text.strip():
            print(json.dumps({"id": rid, "error": "empty text"}), flush=True)
            continue

        try:
            emb = model.encode(
                [text],
                batch_size=1,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=False,
            )[0]
            print(
                json.dumps({"id": rid, "embedding": [float(x) for x in emb.tolist()]}),
                flush=True,
            )
        except Exception as e:
            tb = traceback.format_exc(limit=2)
            print(
                json.dumps({"id": rid, "error": f"encode failed: {e}\n{tb}"}),
                flush=True,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
