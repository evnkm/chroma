# Slidev deck — Improving Recall in Predicate-based ANN Search in ChromaDB

5-minute video presentation. Light theme. Apple keynote / academic poster aesthetic.
All visualizations rendered inline in Slidev — no external animation pipeline.

## Run locally

```bash
cd slidev
npm install
npm run dev
```

Opens at `http://localhost:3030`. Edit `slides.md` and changes hot-reload.

## How animations work

The deck uses Slidev's `v-click` directives for step-by-step builds within a
single slide. **Spacebar (or right arrow) advances the next click**, not the
next slide. So when recording the video, you're effectively scrubbing through
a timeline.

The Bloom gate slide (slide 9) is the most heavily staged: 7 clicks total,
each revealing the next conceptual beat. See the inline comments at the top
of that slide for the click-by-click voiceover mapping.

Other slides are mostly single-state with smooth fade transitions between them.

## Recording the video

1. Record voiceover separately (Audacity / QuickTime).
2. In Slidev, hit `p` for presenter mode (or just present from `localhost:3030`).
3. Screen-record the deck while clicking through at the right tempo.
4. Mix VO underneath in DaVinci / Premiere / Final Cut.
5. Optional: soft underbed at -28 LUFS.

## Slide map

| # | Time | Slide | Visualization |
|---|---|---|---|
| 1 | 0:00 | Title | Typographic |
| 2 | 0:12 | The query | Highlighted clause |
| 3 | 0:12–0:35 | Vector search refresher | `NNScatter.vue` |
| 4 | 0:35–0:55 | Add the filter | `FilterScatter.vue` |
| 5 | 0:55–1:20 | How SPANN works | 4-step typographic |
| 6 | 1:20–1:50 | Recall collapses | Big number |
| 7 | 1:50–2:00 | The question | Beat slide |
| 8 | 2:00–2:25 | Adaptive nprobe | `RecallBars` |
| **9** | **2:25–3:00** | **Bloom gate** | **7-click staged build** |
| 9b | continued | Bloom result | `RecallBars` |
| 10 | 3:00–3:30 | Synopses | `ParetoPlot` |
| 11 | 3:30–4:10 | Demo | (drop in screen recording) |
| 12 | 4:10–4:35 | Why it matters | Big numbers |
| 13 | 4:35–4:50 | Takeaway | Beat slide |
| 14 | 4:50–5:00 | Credits | Typographic |

## Demo recording

Once Zach has the screen recording, drop it in `public/clips/demo.mp4` and
replace the fake terminal panes on slide 11 with:

```html
<video autoplay muted loop playsinline class="w-full max-w-6xl rounded-sm">
  <source src="/clips/demo.mp4" type="video/mp4">
</video>
```

## Design tokens

| Token | Value | Use |
|---|---|---|
| `--ink` | `#0E1626` | primary text |
| `--ink-soft` | `#2B3A55` | secondary text |
| `--paper` | `#FBFAF7` | background |
| `--rule` | `#D9D6CE` | dividers |
| `--query` | `#2E5BFF` | query / "active" semantic |
| `--match` | `#1E9E6A` | matches / "yes" / wins |
| `--miss` | `#D14A4A` | non-matches / "no" / problem |
| `--grid` | `#E8E5DC` | scatter grid |

## Placeholders to fill in

Search `slides.md` for:
- `[REC_BASELINE]` — baseline recall at 0.1% selectivity (current: 0.08)
- `[REC_ADAPT]` — adaptive nprobe recall (current: 0.46)
- `[REC_BLOOM]` — Bloom-only recall (current: 0.72)
- `[REC_SYN]` — synopsis recall
- `[PROBE_MULT]` — probe multiplier cap (current: 8)
- `[DROP_PCT_PLACEHOLDER]` — clusters fetched in demo
