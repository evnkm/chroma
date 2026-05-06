"""End-to-end test of the SPANN demo binary's JSON-line protocol.

Spawns the binary, sends a few requests (cached query_id + free-text via
sidecar), parses every event, and prints a concise summary. Useful for
regression-checking gate effectiveness, parallelism, and recall before the
TUI demo recording.

Usage:
    python demo/test_demo_protocol.py [--bin path] [--no-embed]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent


async def read_event(stdout: asyncio.StreamReader) -> Optional[dict]:
    while True:
        line = await stdout.readline()
        if not line:
            return None
        line = line.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except Exception:
            return {"event": "_parse_error", "raw": line}


async def consume_until_ready(stdout: asyncio.StreamReader) -> dict:
    while True:
        ev = await read_event(stdout)
        if ev is None:
            raise RuntimeError("demo binary exited before ready")
        if ev.get("event") == "ready":
            return ev


async def run_one_request(proc, req: dict, label: str) -> dict:
    print(f"\n=== {label}: {json.dumps(req, default=str)} ===")
    proc.stdin.write((json.dumps(req) + "\n").encode())
    await proc.stdin.drain()
    states = {"baseline": {"events": []}, "optimized": {"events": []}}
    misc: list[dict] = []
    start = time.perf_counter()
    while True:
        try:
            ev = await asyncio.wait_for(read_event(proc.stdout), timeout=120.0)
        except asyncio.TimeoutError:
            print("  timeout")
            return {"states": states, "misc": misc}
        if ev is None:
            return {"states": states, "misc": misc}
        cell = ev.get("cell")
        stage = ev.get("stage")
        name = ev.get("event")
        if cell:
            states[cell]["events"].append(ev)
            print(f"  [{cell:<9}] stage={stage:<8} t_ms={ev.get('t_ms', 0):>7.2f}  {{{', '.join(f'{k}={v}' for k, v in ev.items() if k not in ('event','cell','stage','t_ms','top_k'))}}}")
            if stage == "done" and all(any(e.get("stage") == "done" for e in s["events"]) for s in states.values()):
                # Both done.
                break
        else:
            misc.append(ev)
            interesting = {k: v for k, v in ev.items() if k != "queries"}
            print(f"  [event   ] {interesting}")
    return {"states": states, "misc": misc, "wall_ms": (time.perf_counter() - start) * 1000.0}


def summarize(label: str, result: dict) -> None:
    print(f"\n--- {label} summary ---")
    for cell in ("baseline", "optimized"):
        events = result["states"][cell]["events"]
        done = next((e for e in events if e.get("stage") == "done"), None)
        if done is None:
            print(f"  {cell}: no 'done' event")
            continue
        probing_t = next((e["t_ms"] for e in events if e.get("stage") == "probing"), None)
        probed_t = next((e["t_ms"] for e in events if e.get("stage") == "probed"), None)
        first_gate_t = next((e["t_ms"] for e in events if e.get("stage") == "gating"), None)
        first_fetch_t = next((e["t_ms"] for e in events if e.get("stage") == "fetching"), None)
        print(
            f"  {cell:<9} lat={done['lat_ms']:.2f}ms  recall={done['recall_at_k']:.3f}  "
            f"nprobe={done['nprobe_used']}  heads_rng={done['heads_rng']}  "
            f"heads_fetched={done['heads_fetched']}  cands(before={done['candidates_before']}, after={done['candidates_after']})"
        )
        if probing_t is not None:
            print(
                f"           probing@{probing_t:.2f}ms"
                f"{f' probed@{probed_t:.2f}' if probed_t is not None else ''}"
                f"{f' gating@{first_gate_t:.2f}' if first_gate_t is not None else ''}"
                f"{f' fetching@{first_fetch_t:.2f}' if first_fetch_t is not None else ''}"
            )
    # Parallel race check.
    b_events = result["states"]["baseline"]["events"]
    o_events = result["states"]["optimized"]["events"]
    b_first = b_events[0]["t_ms"] if b_events else None
    o_first = o_events[0]["t_ms"] if o_events else None
    if b_first is not None and o_first is not None:
        delta = abs(o_first - b_first)
        if delta < 5.0:
            print(f"  parallel race: PASS (both started within {delta:.2f}ms of each other)")
        else:
            print(f"  parallel race: ⚠️  WARN — started {delta:.2f}ms apart (likely sequential)")
    # Speedup.
    b_done = next((e for e in b_events if e.get("stage") == "done"), None)
    o_done = next((e for e in o_events if e.get("stage") == "done"), None)
    if b_done and o_done:
        speedup = b_done["lat_ms"] / max(o_done["lat_ms"], 1e-6)
        print(f"  speedup (baseline/optimized): {speedup:.2f}×")


async def main_async(args) -> int:
    bin_path = Path(args.bin)
    if not bin_path.exists():
        print(f"binary not found: {bin_path}", file=sys.stderr)
        return 1

    cmd = [
        str(bin_path),
        "--baseline", str(REPO_ROOT / "demo" / "data" / "baseline"),
        "--optimized", str(REPO_ROOT / "demo" / "data" / "optimized"),
        "--queries", str(REPO_ROOT / "demo" / "data" / "queries.json"),
        "--cache", str(REPO_ROOT / "demo" / "data" / "msmarco_100k_rust"),
    ]
    if args.no_embed:
        cmd.append("--no-embed-sidecar")
    else:
        cmd += [
            "--embed-sidecar", str(REPO_ROOT / "demo" / "embed_sidecar.py"),
            "--python", str(REPO_ROOT / ".venv" / "bin" / "python"),
        ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=sys.stderr,
        cwd=str(REPO_ROOT),
    )
    print("[test] waiting for ready...")
    ready = await consume_until_ready(proc.stdout)
    print(f"[test] ready: baseline_n={ready['baseline_n']}, optimized_n={ready['optimized_n']}")

    cases = []
    # Cached queries.
    cases.append(("cached q=3 + wikihow", {"query_id": 3, "predicate": {"key": "source_domain", "value": "wikihow.com"}, "k": 5}))
    cases.append(("cached q=4 + topic_bucket=0", {"query_id": 4, "predicate": {"key": "topic_bucket", "value": "0"}, "k": 5}))
    if not args.no_embed:
        cases.append(("free-text + wikihow", {"query_text": "how to keep a houseplant alive", "predicate": {"key": "source_domain", "value": "wikihow.com"}, "k": 5}))
    for label, req in cases:
        result = await run_one_request(proc, req, label)
        summarize(label, result)

    proc.stdin.close()
    try:
        await asyncio.wait_for(proc.wait(), timeout=3.0)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="/tmp/chroma-target/release/spann_demo")
    ap.add_argument("--no-embed", action="store_true")
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
