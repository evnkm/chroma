"""SPANN race demo TUI.

Spawns the Rust demo binary and renders a side-by-side race between cell 000
(baseline) and cell 111 (optimized: bloom + synopsis + adaptive nprobe).

Layout:
    ┌──────────────────────────────────────────────────────────────┐
    │  Query: <text>     Predicate: <key = value>                  │
    ├────────────────────────────┬─────────────────────────────────┤
    │  Baseline (000)            │  Optimized (111)                │
    │  ⠋ <stage>...              │  ⠋ <stage>...                   │
    │  Latency: <live>           │  Latency: <live>                │
    │                            │                                 │
    │  Top-3:                    │  Top-3:                         │
    │  1. <text> ...             │  1. <text> ...                  │
    │  2. <text> ...             │  2. <text> ...                  │
    │  3. <text> ...             │  3. <text> ...                  │
    ├────────────────────────────┴─────────────────────────────────┤
    │  Optimized is N× faster · X heads probed vs Y · same top-3   │
    └──────────────────────────────────────────────────────────────┘

Usage:
    python demo/tui.py [--bin path/to/spann_demo]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BIN = REPO_ROOT / "target" / "release" / "spann_demo"
DEFAULT_BIN_ALT = Path("/tmp/chroma-target/release/spann_demo")
PRESET_PREDICATES = [
    ("realistic-strong", "source_domain", "wikihow.com", "How-to articles only"),
    ("realistic-medium", "source_domain", "mayoclinic.org", "Mayo Clinic only"),
    ("realistic-niche", "source_domain", "investopedia.com", "Investopedia only"),
    ("synthetic-killer", "topic_bucket", "0", "Synthetic 0.1% bucket"),
    ("query-type", "query_type", "PERSON", "Passages about people"),
    ("none", "topic_bucket", "-1", "No predicate (matches nothing — sanity)"),
]


@dataclass
class CellState:
    cell: str
    stage: str = "waiting"
    nprobe_used: int = 0
    heads_rng: int = 0
    heads_fetched: int = 0
    candidates: int = 0
    candidates_after: int = 0
    lat_ms: float = 0.0
    recall: Optional[float] = None
    top_k: list[dict] = field(default_factory=list)
    done: bool = False
    bloom_kept: Optional[int] = None
    bloom_dropped: Optional[int] = None
    synopsis_kept: Optional[int] = None
    synopsis_dropped: Optional[int] = None


@dataclass
class RaceState:
    query_text: str = ""
    predicate_display: str = ""
    selectivity: float = 0.0
    n_match: int = 0
    gt_t_ms: float = 0.0
    embed_t_ms: float = 0.0
    baseline: CellState = field(default_factory=lambda: CellState("baseline"))
    optimized: CellState = field(default_factory=lambda: CellState("optimized"))
    started_at: float = 0.0


def fmt_lat(ms: float) -> str:
    if ms < 1.0:
        return f"{ms*1000:.0f} µs"
    return f"{ms:.2f} ms"


def render_cell_panel(label: str, state: CellState, race: RaceState, color: str) -> Panel:
    spinner = Spinner("dots", text=f"{state.stage}...")
    if state.done:
        head = Text(f"✓ done in {fmt_lat(state.lat_ms)}", style=f"bold {color}")
    else:
        head_text = Text()
        head_text.append("⠋ ", style=color)
        head_text.append(f"{state.stage}", style="bold")
        live_lat = (time.perf_counter() - race.started_at) * 1000.0 if race.started_at else 0.0
        head_text.append(f"   ({fmt_lat(live_lat)})", style="dim")
        head = head_text

    body = Table.grid(padding=(0, 1))
    body.add_column(justify="right", style="dim")
    body.add_column()
    body.add_row("nprobe:", f"{state.nprobe_used}" if state.nprobe_used else "—")
    body.add_row("heads probed:", f"{state.heads_rng}" if state.heads_rng else "—")
    if state.bloom_kept is not None:
        body.add_row("after bloom:", f"{state.bloom_kept} (dropped {state.bloom_dropped})")
    if state.synopsis_kept is not None:
        body.add_row("after synopsis:", f"{state.synopsis_kept} (dropped {state.synopsis_dropped})")
    body.add_row("heads fetched:", f"{state.heads_fetched}" if state.heads_fetched else "—")
    body.add_row("candidates:", f"{state.candidates}" if state.candidates else "—")
    if state.recall is not None:
        recall_str = f"{state.recall:.2f}"
        body.add_row("recall@k:", recall_str)
    body.add_row("latency:", fmt_lat(state.lat_ms) if state.done else "—")

    if state.top_k:
        topk_lines: list[Text] = []
        for i, item in enumerate(state.top_k[:3], 1):
            txt = item.get("text", "")
            if len(txt) > 90:
                txt = txt[:87] + "..."
            t = Text()
            t.append(f"{i}. ", style="dim")
            t.append(f"({item.get('source_domain', '?')}) ", style="cyan")
            t.append(txt, style="white")
            topk_lines.append(t)
        topk_block = Group(*topk_lines)
    else:
        topk_block = Text("(no results yet)", style="dim")

    return Panel(
        Group(head, Text(""), body, Text(""), Text("Top results:", style="bold dim"), topk_block),
        title=f"[{color}]{label}[/{color}]",
        border_style=color,
    )


def render_summary(race: RaceState) -> Panel:
    if not (race.baseline.done and race.optimized.done):
        return Panel(
            Text("waiting for both cells...", style="dim"),
            title="Comparison",
            border_style="dim",
        )
    b = race.baseline
    o = race.optimized
    speedup = b.lat_ms / max(o.lat_ms, 1e-6)
    head_ratio = b.heads_fetched / max(o.heads_fetched, 1)
    recall_delta = (o.recall or 0.0) - (b.recall or 0.0)
    same_top = (
        bool(b.top_k)
        and bool(o.top_k)
        and [r.get("id") for r in b.top_k[:3]] == [r.get("id") for r in o.top_k[:3]]
    )

    line1 = Text()
    line1.append("⚡ ", style="yellow")
    line1.append(f"{speedup:.1f}× faster", style="bold green")
    line1.append("  ·  ", style="dim")
    line1.append(f"{head_ratio:.1f}× fewer heads fetched", style="bold green")
    line1.append("  ·  ", style="dim")
    if recall_delta > 0.01:
        line1.append(f"+{recall_delta*100:.0f}pp recall lift", style="bold green")
    elif recall_delta < -0.01:
        line1.append(f"{recall_delta*100:+.0f}pp recall", style="bold red")
    else:
        line1.append("same recall", style="bold")

    line2 = Text(
        f"  baseline: {fmt_lat(b.lat_ms)} (recall={b.recall:.2f})    "
        f"optimized: {fmt_lat(o.lat_ms)} (recall={o.recall:.2f})",
        style="dim",
    )
    line3 = Text(
        f"  predicate: {race.predicate_display}    matches {race.n_match} ({race.selectivity*100:.3f}%)    "
        f"GT compute: {race.gt_t_ms:.0f}ms",
        style="dim",
    )
    if same_top:
        line4 = Text("  Top-3 IDs match — same answer, different speed.", style="dim italic green")
    else:
        line4 = Text("  Top-3 IDs differ between cells.", style="dim italic yellow")

    return Panel(
        Group(line1, Text(""), line2, line3, line4),
        title="Comparison",
        border_style="green",
    )


def render_layout(race: RaceState) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="cells", ratio=2),
        Layout(name="summary", size=8),
    )
    header = Table.grid(padding=(0, 1))
    header.add_column()
    header.add_row(Text(f"Query: {race.query_text}", style="bold cyan"))
    pred_line = (
        f"Predicate: {race.predicate_display}"
        f"   |   matches {race.n_match}/{race.n_match if race.selectivity == 0 else int(race.n_match/race.selectivity)} "
        f"({race.selectivity*100:.3f}%)"
        if race.predicate_display
        else "Predicate: <waiting>"
    )
    header.add_row(Text(pred_line, style="dim"))
    layout["header"].update(Panel(header, border_style="cyan"))

    cells = Layout()
    cells.split_row(
        Layout(name="baseline"),
        Layout(name="optimized"),
    )
    cells["baseline"].update(render_cell_panel("Baseline (000)", race.baseline, race, "red"))
    cells["optimized"].update(render_cell_panel("Optimized (111)", race.optimized, race, "green"))
    layout["cells"].update(cells)

    layout["summary"].update(render_summary(race))
    return layout


async def read_event(stdout: asyncio.StreamReader) -> Optional[dict]:
    line = await stdout.readline()
    if not line:
        return None
    line = line.decode("utf-8", errors="replace").strip()
    if not line:
        return await read_event(stdout)
    try:
        return json.loads(line)
    except Exception:
        return {"event": "_parse_error", "raw": line}


def update_state(race: RaceState, ev: dict) -> None:
    cell = ev.get("cell")
    stage = ev.get("stage")
    name = ev.get("event")

    if cell:
        cs = race.baseline if cell == "baseline" else race.optimized
        if stage == "probing":
            cs.stage = "probing"
        elif stage == "probed":
            cs.heads_rng = ev.get("heads_rng", 0)
            cs.nprobe_used = ev.get("nprobe_used", 0)
            cs.stage = "probed"
        elif stage == "gating":
            cs.stage = f"gating ({ev.get('gate', '')})"
        elif stage == "gated":
            gate = ev.get("gate", "")
            if gate == "bloom":
                cs.bloom_kept = ev.get("kept")
                cs.bloom_dropped = ev.get("dropped")
            elif gate == "synopsis":
                cs.synopsis_kept = ev.get("kept")
                cs.synopsis_dropped = ev.get("dropped")
            cs.stage = f"after {gate}"
        elif stage == "fetching":
            cs.stage = "fetching PLs"
            cs.heads_fetched = ev.get("heads", 0)
        elif stage == "fetched":
            cs.stage = "fetched"
            cs.candidates = ev.get("candidates", 0)
        elif stage == "scoring":
            cs.stage = "scoring"
        elif stage == "done":
            cs.done = True
            cs.lat_ms = ev.get("lat_ms", 0.0)
            cs.recall = ev.get("recall_at_k")
            cs.heads_rng = ev.get("heads_rng", cs.heads_rng)
            cs.heads_fetched = ev.get("heads_fetched", cs.heads_fetched)
            cs.nprobe_used = ev.get("nprobe_used", cs.nprobe_used)
            cs.candidates = ev.get("candidates_after", cs.candidates)
            cs.candidates_after = ev.get("candidates_after", 0)
            cs.top_k = ev.get("top_k", [])
            cs.stage = "done"
    else:
        if name == "ground_truth":
            race.selectivity = ev.get("selectivity", 0.0)
            race.n_match = ev.get("n_match", 0)
            race.gt_t_ms = ev.get("t_ms", 0.0)
        elif name == "embedded":
            race.embed_t_ms = ev.get("t_ms", 0.0)
        elif name == "query_received":
            race.query_text = ev.get("query_text", race.query_text)
            race.predicate_display = ev.get("predicate", race.predicate_display)
        elif name == "error":
            console = Console()
            console.print(f"[red]error:[/red] {ev.get('message', '?')}")


async def race_one(
    proc: asyncio.subprocess.Process,
    console: Console,
    query: str,
    predicate_key: str,
    predicate_value: str,
    k: int,
) -> RaceState:
    race = RaceState(
        query_text=query,
        predicate_display=f"{predicate_key} = {predicate_value!r}",
        started_at=time.perf_counter(),
    )

    # Send request.
    req = json.dumps(
        {
            "query_text": query,
            "predicate": {"key": predicate_key, "value": predicate_value},
            "k": k,
        }
    )
    proc.stdin.write((req + "\n").encode())
    await proc.stdin.drain()

    with Live(render_layout(race), console=console, refresh_per_second=20, screen=False) as live:
        while True:
            try:
                ev = await asyncio.wait_for(read_event(proc.stdout), timeout=60.0)
            except asyncio.TimeoutError:
                console.print("[red]timeout waiting for event[/red]")
                break
            if ev is None:
                break
            update_state(race, ev)
            live.update(render_layout(race))
            if race.baseline.done and race.optimized.done:
                # Wait a small beat so the user sees the final frame.
                await asyncio.sleep(0.4)
                live.update(render_layout(race))
                break
    return race


async def consume_until_ready(proc: asyncio.subprocess.Process, console: Console) -> dict:
    while True:
        ev = await read_event(proc.stdout)
        if ev is None:
            raise RuntimeError("demo binary exited before ready")
        if ev.get("event") == "ready":
            return ev


def pick_predicate() -> tuple[str, str]:
    print("\n  Predicate presets:")
    for i, (slug, k, v, desc) in enumerate(PRESET_PREDICATES, 1):
        print(f"    {i}. [{slug}] {k} = {v!r}    {desc}")
    print("    c. custom")
    choice = Prompt.ask("  pick", default="1")
    if choice.strip().lower() == "c":
        k = Prompt.ask("    key (source_domain|length_bucket|query_type|topic_bucket)", default="source_domain")
        v = Prompt.ask("    value", default="wikihow.com")
        return k, v
    try:
        idx = int(choice) - 1
    except ValueError:
        idx = 0
    idx = max(0, min(idx, len(PRESET_PREDICATES) - 1))
    _, k, v, _ = PRESET_PREDICATES[idx]
    return k, v


async def main_async(args) -> int:
    console = Console()

    bin_path = Path(args.bin)
    if not bin_path.exists():
        # Try alternate location.
        if DEFAULT_BIN_ALT.exists() and bin_path == DEFAULT_BIN:
            bin_path = DEFAULT_BIN_ALT
        else:
            console.print(f"[red]demo binary not found at {bin_path}[/red]")
            return 1

    cmd = [
        str(bin_path),
        "--baseline",
        str(REPO_ROOT / "demo" / "data" / "baseline"),
        "--optimized",
        str(REPO_ROOT / "demo" / "data" / "optimized"),
        "--queries",
        str(REPO_ROOT / "demo" / "data" / "queries.json"),
        "--cache",
        str(REPO_ROOT / "demo" / "data" / "msmarco_100k_rust"),
        "--embed-sidecar",
        str(REPO_ROOT / "demo" / "embed_sidecar.py"),
        "--python",
        str(REPO_ROOT / ".venv" / "bin" / "python"),
    ]
    console.print(f"[dim]launching: {shlex.join(cmd)}[/dim]")

    env = os.environ.copy()
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=sys.stderr,
        env=env,
        cwd=str(REPO_ROOT),
    )

    console.print("[dim]waiting for demo to load indexes + embed sidecar...[/dim]")
    ready = await consume_until_ready(proc, console)
    console.print(f"[green]ready[/green]: baseline N={ready['baseline_n']}, optimized N={ready['optimized_n']}")
    console.print()

    try:
        while True:
            console.rule("[bold cyan]new query[/bold cyan]")
            query = Prompt.ask("  query (free text, or 'q' to quit)", default="how does photosynthesis work")
            if query.strip().lower() in ("q", "quit", "exit"):
                break
            pred_key, pred_value = pick_predicate()
            await race_one(proc, console, query, pred_key, pred_value, k=args.k)
    finally:
        try:
            proc.stdin.write(b'{"event":"quit"}\n')
            await proc.stdin.drain()
        except Exception:
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            proc.kill()
        await proc.wait()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=str(DEFAULT_BIN))
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
