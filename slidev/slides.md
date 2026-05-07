---
# Slidev deck for: Improving Recall in Predicate-based ANN Search in ChromaDB
# 6.5830 final-project video presentation, 5 minutes
# Theme: light / Apple keynote-academic poster hybrid
theme: default
title: Improving Recall in Predicate-based ANN Search in ChromaDB
info: |
  Zach Marinov, Kartik Pingle, Evan Kim — 6.5830, Spring 2026
class: text-center
highlighter: shiki
drawings:
  persist: false
transition: fade-out
mdc: true
fonts:
  sans: "Inter Tight"
  serif: "Newsreader"
  mono: "JetBrains Mono"
  fallbacks: false
colorSchema: light
aspectRatio: 16/9
canvasWidth: 1280
---

<style>
:root {
  --ink: #0E1626;
  --ink-soft: #2B3A55;
  --paper: #FBFAF7;
  --rule: #D9D6CE;
  --query: #2E5BFF;
  --match: #1E9E6A;
  --miss: #D14A4A;
  --grid: #E8E5DC;
}

.slidev-layout {
  background: var(--paper);
  color: var(--ink);
  font-family: var(--slidev-fonts-sans);
  font-feature-settings: "ss01", "cv11";
  letter-spacing: -0.01em;
}

.serif { font-family: var(--slidev-fonts-serif); font-weight: 400; }
.mono  { font-family: var(--slidev-fonts-mono); }

h1, h2, h3 { font-family: var(--slidev-fonts-serif); font-weight: 500; letter-spacing: -0.02em; color: var(--ink); }
h1 { font-size: 4.4rem; line-height: 1.05; }
h2 { font-size: 2.8rem; line-height: 1.1; }
h3 { font-size: 1.4rem; color: var(--ink-soft); font-weight: 400; }

.eyebrow {
  font-family: var(--slidev-fonts-sans);
  font-size: 0.78rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--ink-soft);
  font-weight: 500;
}

.rule {
  height: 1px;
  background: var(--rule);
  width: 4rem;
  margin: 1rem 0;
}

.dark .slidev-layout { background: #0B0E14; color: #F5F3EE; }

.bignum {
  font-family: var(--slidev-fonts-serif);
  font-feature-settings: "tnum";
  font-size: 8rem;
  font-weight: 500;
  line-height: 1;
  letter-spacing: -0.04em;
}

.caption { font-size: 0.95rem; color: var(--ink-soft); font-style: italic; }

.fade-mask {
  -webkit-mask-image: linear-gradient(to bottom, black 70%, transparent 100%);
}

/* footer rule for content slides */
.footer-rule {
  position: absolute;
  bottom: 1.6rem; left: 3rem; right: 3rem;
  display: flex; justify-content: space-between;
  font-size: 0.72rem; color: var(--ink-soft);
  letter-spacing: 0.08em; text-transform: uppercase;
  font-family: var(--slidev-fonts-sans);
}
/* slide 9 — bloom gate */
.bloom-cell {
  width: 12px; height: 36px;
  border: 1px solid #D9D6CE; background: #FBFAF7;
  border-radius: 1px; transition: all 0.4s ease;
}
.bloom-on   { background: #0E1626; border-color: #0E1626; }
.bloom-hit  { background: #2E5BFF !important; border-color: #2E5BFF !important;
              box-shadow: 0 0 0 2px rgba(46, 91, 255, 0.18); }
.bloom-miss { border-color: #D14A4A !important; border-width: 2px !important;
              box-shadow: 0 0 0 2px rgba(209, 74, 74, 0.18); }
.cluster-mini { width: 56px; height: 56px; transition: opacity 0.6s ease; }
.cluster-mini.cluster-skip { opacity: 0.35; }
.cluster-mini.cluster-keep { opacity: 1; }
/* slide 10 — synopses */
.synopsis-zero-row { background: rgba(209, 74, 74, 0.06); }
</style>
<!-- =====================================================================
SLIDE 1 — TITLE / OPEN
0:00 – 0:12
===================================================================== -->
<div class="absolute inset-0 flex flex-col justify-center px-24">
  <div class="eyebrow mb-6">6.5830 · Spring 2026 · Final Project</div>
  <h1 class="serif">
    Improving recall<br/>
    in predicate-based<br/>
    <span style="color: var(--query)">vector search.</span>
  </h1>
  <div class="self-center rule mt-10"></div>
  <div class="text-base mt-2" style="color: var(--ink-soft)">
    Zach Marinov &nbsp;·&nbsp; Kartik Pingle &nbsp;·&nbsp; Evan Kim
  </div>
</div>
<!-- speaker note ------------------------------------------------------ -->
<!--
Say you're searching a vector database. You want products similar to the
one you're looking at, but only ones under fifty bucks.
Two things going on in that query. A similarity search and a filter.
Today's vector databases handle the similarity part really well.
The filter part is where things break.
-->
---
layout: default
transition: fade
---
<!-- =====================================================================
SLIDE 2 — VECTOR SEARCH
0:12 – 0:35
Replace placeholder with Manim NN_intro clip when ready.
===================================================================== -->
<div class="absolute inset-0 grid grid-cols-12 gap-8 px-20 pt-24 pb-20">
  <div class="col-span-5 flex flex-col justify-center">
    <div class="eyebrow mb-6">vector search</div>
    <h2 class="serif mb-8">Embedding space<br/>Proximity means similarity</h2>
    <div class="text-lg leading-relaxed" style="color: var(--ink-soft)">
      Embed the query into the same space, grab the nearest neighbors.
    </div>
  </div>
  <div class="col-span-7 flex items-center justify-center min-h-0 min-w-0 overflow-visible">
    <NNScatter />
  </div>
</div>
<div class="footer-rule"><span>1 · vector search, briefly</span></div>
---
layout: default
transition: fade
---
<!-- =====================================================================
SLIDE 3 — CHROMADB
0:12 – 0:35
===================================================================== -->
<div class="absolute inset-0 grid grid-cols-12 gap-8 px-20 pt-24 pb-20">
  <div class="col-span-6 flex flex-col justify-center">
    <div class="eyebrow mb-6">the system we're building on</div>
    <h2 class="serif mb-8">An open-source<br/>vector database.</h2>
    <div class="text-lg leading-relaxed mb-5" style="color: var(--ink-soft)">
      Chroma stores embeddings alongside their metadata and serves
      filtered nearest-neighbor queries — the workload behind RAG, semantic
      search, and agent memory.
    </div>
    <div class="grid grid-cols-3 gap-6 mt-4">
      <div>
        <div class="serif text-3xl" style="color: var(--ink)">19k+</div>
        <div class="text-xs mt-1" style="color: var(--ink-soft)">GitHub stars</div>
      </div>
      <div>
        <div class="serif text-3xl" style="color: var(--ink)">Rust</div>
        <div class="text-xs mt-1" style="color: var(--ink-soft)">distributed core</div>
      </div>
      <div>
        <div class="serif text-3xl" style="color: var(--ink)">SPANN</div>
        <div class="text-xs mt-1" style="color: var(--ink-soft)">disk-based ANN index</div>
      </div>
    </div>
  </div>
  <div class="col-span-6 flex flex-col items-center justify-center min-h-0 min-w-0 overflow-visible">
    <img src="/chroma-logo.png" alt="Chroma logo"
         class="w-full max-w-2xl" />
  </div>
</div>
<div class="footer-rule"><span>2 · ChromaDB</span></div>
---
layout: default
transition: fade
---
<!-- =====================================================================
SLIDE 4 — Queries and filtered queries
Click 0: heading + plain similarity query (top-left)
Click 1: filtered query reveals below it
Click 2: FilterScatter graphic reveals on the right
===================================================================== -->
<div class="absolute inset-0 px-20 pt-20 pb-16 grid grid-cols-12 gap-10">
  <!-- LEFT: stacked queries -->
  <div class="col-span-6 flex flex-col">
    <div class="eyebrow mb-3">two kinds of query</div>
    <h2 class="serif mb-8 text-4xl">
      Similarity alone, or similarity
      <span style="color: var(--miss)">plus a predicate</span>.
    </h2>
    <div class="flex flex-col gap-6">
      <!-- TOP: plain similarity query (always visible) -->
      <div>
        <div class="eyebrow mb-2" style="color: var(--query)">similarity only</div>
        <div class="serif italic text-base mb-3" style="color: var(--ink-soft)">
          "How do I take care of my houseplant?"
        </div>
        <div class="rounded-sm px-5 py-3 mono text-sm"
             style="background: #F4F6FB; border: 1px solid var(--rule); color: var(--ink); line-height: 1.6">
          <span style="color: #7B8AAE">collection</span>.query(<br/>
          &nbsp;&nbsp;query_embeddings=[<span style="color: var(--query)">q</span>],<br/>
          &nbsp;&nbsp;n_results=<span style="color: var(--query)">10</span>,<br/>
          )
        </div>
      </div>
      <!-- BOTTOM: filtered query — reveals on click 1 -->
      <div v-click="1">
        <div class="eyebrow mb-2" style="color: var(--miss)">+ metadata filter</div>
        <div class="serif italic text-base mb-3" style="color: var(--ink-soft)">
          "How do I take care of my houseplant?,
          <span style="color: var(--ink); background: #FFF7CC; padding: 0 0.25rem; border-radius: 2px">but only results from WikiHow</span>."
        </div>
        <div class="rounded-sm px-5 py-3 mono text-sm"
             style="background: #F4F6FB; border: 1px solid var(--rule); color: var(--ink); line-height: 1.6">
          <span style="color: #7B8AAE">collection</span>.query(<br/>
          &nbsp;&nbsp;query_embeddings=[<span style="color: var(--query)">q</span>],<br/>
          &nbsp;&nbsp;n_results=<span style="color: var(--query)">10</span>,<br/>
          &nbsp;&nbsp;<span style="background: #FFF7CC; padding: 0 0.2rem; border-radius: 2px">where={<span style="color: var(--miss)">"domain"</span>: {<span style="color: var(--miss)">"="</span>: "wikihow.com"}},</span><br/>
          )
        </div>
      </div>
    </div>
  </div>
  <!-- RIGHT: FilterScatter — reveals on click 2 -->
  <div v-click="2" class="col-span-6 flex items-center justify-center min-h-0 min-w-0 overflow-visible">
    <FilterScatter />
  </div>
</div>
<div class="footer-rule"><span>3 · types of query</span></div>
---
layout: default
transition: fade
---
<!-- =====================================================================
SLIDE 5 — HOW SPANN WORKS (click-staged, 4 clicks)
0:55 – 1:20
Click 0: title + cluster step appears
Click 1: route step
Click 2: fetch step
Click 3: rank step
Click 4: arrows connect them, header line completes
===================================================================== -->
<div class="absolute inset-0 px-20 pt-20 pb-16 flex flex-col">
  <div class="eyebrow mb-4">how ChromaDB's spann searches</div>
  <h2 class="serif mb-10">
    <span>Cluster. </span><span v-click="1">Pick the closest. </span><span v-click="2">Fetch. </span><span v-click="3">Filter. </span><span v-click="4">Rank.</span>
  </h2>
  <div class="flex-1 grid grid-cols-5 gap-6 items-stretch relative">
    <div class="border-l-2 pl-6" style="border-color: var(--rule)">
      <div class="bignum" style="color: var(--query)">1</div>
      <div class="serif text-2xl mt-2">cluster</div>
      <div class="text-sm mt-3" style="color: var(--ink-soft)">k-means partitions the vector space. Each cluster has a centroid representing it.</div>
      <div class="mono text-xs mt-3" style="color: var(--ink-soft)">~1000 centroids for 100K docs</div>
    </div>
    <div v-click="1" class="border-l-2 pl-6" style="border-color: var(--rule)">
      <div class="bignum" style="color: var(--query)">2</div>
      <div class="serif text-2xl mt-2">route</div>
      <div class="text-sm mt-3" style="color: var(--ink-soft)">An HNSW graph over the centroids finds the nprobe closest ones to the query in milliseconds.</div>
      <div class="mono text-xs mt-3" style="color: var(--ink-soft)">nprobe ≈ 24–32 default</div>
    </div>
    <div v-click="2" class="border-l-2 pl-6" style="border-color: var(--rule)">
      <div class="bignum" style="color: var(--query)">3</div>
      <div class="serif text-2xl mt-2">fetch</div>
      <div class="text-sm mt-3" style="color: var(--ink-soft)">Pull each chosen cluster's posting list — all the document vectors and metadata it contains.</div>
      <div class="mono text-xs mt-3" style="color: var(--ink-soft)">one S3 round-trip per cluster</div>
    </div>
    <div v-click="3" class="border-l-2 pl-6" style="border-color: var(--rule)">
      <div class="bignum" style="color: var(--query)">4</div>
      <div class="serif text-2xl mt-2">filter</div>
      <div class="text-sm mt-3" style="color: var(--ink-soft)"> Apply filter using global inverted index to each value in the posting list</div>
      <div class="mono text-xs mt-3" style="color: var(--ink-soft)">filter has some selectivity</div>
    </div>
    <div v-click="4" class="border-l-2 pl-6" style="border-color: var(--rule)">
      <div class="bignum" style="color: var(--query)">5</div>
      <div class="serif text-2xl mt-2">rank</div>
      <div class="text-sm mt-3" style="color: var(--ink-soft)">Brute-force compute distance over the fetched candidates. Return top-k.</div>
      <div class="mono text-xs mt-3" style="color: var(--ink-soft)">k = 10</div>
    </div>
  </div>
</div>
<div class="footer-rule"><span>4 · spann pipeline</span></div>
---
layout: default
transition: fade
---
<!-- =====================================================================
SLIDE 6 — RECALL COLLAPSES (click-staged, 3 clicks)
1:20 – 1:50
Click 0: title + the visual setup (chosen 3 clusters, no matches yet)
Click 1: filter activates, matches reveal scattered across many clusters
Click 2: bignum recall slams in
===================================================================== -->
<div class="absolute inset-0 flex flex-col justify-center px-20 py-12">
  <div class="eyebrow mb-4" style="color: var(--miss)">but the routing is filter-blind</div>
  <h2 class="serif mb-6 max-w-4xl text-4xl">
    The chosen clusters might contain
    <span style="color: var(--miss)">zero matches</span>.
  </h2>
  <div class="grid grid-cols-12 gap-12 items-center mt-2">
    <!-- LEFT: visual showing the failure mechanism -->
    <div class="col-span-7 flex items-center justify-center">
      <svg viewBox="0 0 480 300" class="w-full max-w-2xl">
        <!-- 16 clusters in a 4x4 grid -->
        <g fill="#FBFAF7" stroke="#D9D6CE" stroke-width="1">
          <circle cx="80"  cy="60"  r="22"/><circle cx="180" cy="60"  r="22"/>
          <circle cx="280" cy="60"  r="22"/><circle cx="380" cy="60"  r="22"/>
          <circle cx="80"  cy="130" r="22"/><circle cx="180" cy="130" r="22"/>
          <circle cx="280" cy="130" r="22"/><circle cx="380" cy="130" r="22"/>
          <circle cx="80"  cy="200" r="22"/><circle cx="180" cy="200" r="22"/>
          <circle cx="280" cy="200" r="22"/><circle cx="380" cy="200" r="22"/>
        </g>
        <!-- Highlight 3 chosen clusters (the ones SPANN picks based on similarity) -->
        <g fill="none" stroke="#2E5BFF" stroke-width="2.5">
          <circle cx="180" cy="130" r="26"/>
          <circle cx="280" cy="130" r="26"/>
          <circle cx="180" cy="60"  r="26"/>
        </g>
        <text x="240" y="252" text-anchor="middle"
              style="font-family: var(--slidev-fonts-mono); font-size: 11px; fill: var(--query)">
          ↑ 3 clusters SPANN picked
        </text>
        <!-- Match dots — appear on click 1, none in chosen clusters -->
        <g v-click="1" fill="#1E9E6A">
          <circle cx="78" cy="62" r="3.5"/><circle cx="378" cy="58" r="3.5"/>
          <circle cx="80" cy="200" r="3.5"/><circle cx="380" cy="202" r="3.5"/>
          <circle cx="282" cy="198" r="3.5"/><circle cx="78" cy="132" r="3.5"/>
        </g>
        <text v-click="1" x="240" y="282" text-anchor="middle"
              style="font-family: var(--slidev-fonts-serif); font-style: italic; font-size: 13px; fill: var(--match)">
          matches scattered everywhere except inside the chosen ones
        </text>
      </svg>
    </div>
    <!-- RIGHT: bignum recall, appears on click 2 -->
    <div class="col-span-5 flex flex-col">
      <div v-click="2">
        <div class="bignum" style="color: var(--miss); font-size: 7rem">0.284</div>
        <div class="serif text-xl mt-2">baseline recall</div>
        <div class="caption mt-1">SIFT1M · N=1M · 1% filter selectivity (k = 10)</div>
        <div class="text-sm mt-6 leading-snug" style="color: var(--ink-soft)">
          Recall@k measures the fraction of top-k nearest neighbors the system actually returns. With selective filters, the baseline collapses.
        </div>
      </div>
    </div>
  </div>
</div>
<div class="footer-rule"><span>5 · recall collapse</span></div>
---
layout: center
transition: slide-up
---
<!-- =====================================================================
SLIDE 7 — three approaches
Beat slide. Single sentence. Pause.
===================================================================== -->
<div class="px-24 max-w-5xl">
  <div class="eyebrow mb-10 text-center">three approaches</div>
  <h2 class="serif text-center" style="font-size: 3.4rem; line-height: 1.15">
    1. Filter-adaptive nprobes <br/>
    2. Metadata synposes <br/>
    3. Bloom filters <br/>
  </h2>
</div>
---
layout: default
transition: fade
---
<!-- =====================================================================
SLIDE 8 — FIX 1: ADAPTIVE NPROBE (click-staged, 6 clicks)
2:00 – 2:25
Click 0: title + idea
Click 1: matches reveal — 6 green match clusters scattered across space
Click 2: default ring of 24 centroids appears — 0 of 6 inside default ring
Click 3: ring expands to adaptive (192) — matches now captured
Click 4: scaling formula appears (right side)
Click 5: recall bar updates
===================================================================== -->
<div class="absolute inset-0 px-20 pt-16 pb-16 flex flex-col">
  <div class="mb-6">
    <div class="eyebrow mb-3">approach 1 of 3</div>
    <h2 class="serif text-5xl">Probe more clusters.</h2>
  </div>
  <div class="flex-1 grid grid-cols-12 gap-12 items-center">
    <!-- LEFT: animated ring geometry (always visible, but ring expands and matches reveal on clicks) -->
    <div class="col-span-7 flex items-center justify-center">
      <svg viewBox="0 0 480 360" class="w-full max-w-2xl">
        <!-- Adaptive (wide) ring — appears on click 3 -->
        <circle v-click="3" cx="240" cy="180" r="150" fill="none"
                stroke="#2E5BFF" stroke-width="1.2" stroke-dasharray="5,3" opacity="0.7"/>
        <text v-click="3" x="240" y="22" text-anchor="middle"
              style="font-family: var(--slidev-fonts-mono); font-size: 13px; fill: var(--query)">
          adaptive · nprobe = 384
        </text>
        <!-- Default (tight) ring — appears on click 2 -->
        <circle v-click="2" cx="240" cy="180" r="60" fill="none"
                stroke="#0E1626" stroke-width="1.5" stroke-dasharray="3,2"/>
        <text v-click="2" x="240" y="262" text-anchor="middle"
              style="font-family: var(--slidev-fonts-mono); font-size: 12px; fill: var(--ink)">
          default · nprobe = 24
        </text>
        <!-- Centroids inside default ring (no matches) -->
        <g fill="#7B8AAE">
          <circle cx="220" cy="160" r="3"/><circle cx="260" cy="170" r="3"/>
          <circle cx="225" cy="200" r="3"/><circle cx="265" cy="205" r="3"/>
          <circle cx="245" cy="145" r="3"/><circle cx="280" cy="185" r="3"/>
          <circle cx="210" cy="185" r="3"/><circle cx="250" cy="215" r="3"/>
        </g>
        <!-- Centroids in adaptive ring only -->
        <g fill="#7B8AAE">
          <circle cx="180" cy="150" r="3"/><circle cx="300" cy="145" r="3"/>
          <circle cx="320" cy="190" r="3"/><circle cx="290" cy="230" r="3"/>
          <circle cx="200" cy="240" r="3"/><circle cx="160" cy="190" r="3"/>
          <circle cx="170" cy="230" r="3"/><circle cx="310" cy="240" r="3"/>
          <circle cx="195" cy="120" r="3"/><circle cx="280" cy="125" r="3"/>
          <circle cx="335" cy="160" r="3"/><circle cx="155" cy="160" r="3"/>
          <circle cx="340" cy="215" r="3"/><circle cx="270" cy="255" r="3"/>
          <circle cx="220" cy="270" r="3"/><circle cx="155" cy="210" r="3"/>
        </g>
        <!-- Matching centroids — reveal on click 1, all outside default ring -->
        <g v-click="1" stroke="#1E9E6A" stroke-width="2.5" fill="none">
          <circle cx="180" cy="150" r="8"/><circle cx="320" cy="190" r="8"/>
          <circle cx="200" cy="240" r="8"/><circle cx="280" cy="125" r="8"/>
          <circle cx="335" cy="160" r="8"/><circle cx="270" cy="255" r="8"/>
        </g>
        <!-- Query at center -->
        <circle cx="240" cy="180" r="7" fill="#2E5BFF" stroke="#FBFAF7" stroke-width="2"/>
        <!-- Legend -->
        <g v-click="1" transform="translate(40, 320)">
          <circle cx="6" cy="0" r="4" stroke="#1E9E6A" stroke-width="2" fill="none"/>
          <text x="18" y="4" style="font-family: var(--slidev-fonts-serif); font-style: italic; font-size: 12px; fill: var(--ink-soft)">
            cluster contains a match
          </text>
        </g>
      </svg>
    </div>
    <!-- RIGHT: formula and recall bars stack -->
    <div class="col-span-5 flex flex-col gap-8">
      <!-- Caption that swaps based on click state -->
      <div class="text-base leading-snug" style="color: var(--ink-soft); min-height: 4rem">
        <div v-click="[1, 2]" class="serif italic">
          With a 1% filter, six matches scattered across the centroid space.
        </div>
        <div v-click="[2, 3]" class="serif italic" style="color: var(--miss)">
          The default ring captures none of them.
        </div>
        <div v-click="[3, 4]" class="serif italic" style="color: var(--match)">
          Widen the ring. All six matches are now inside.
        </div>
        <!-- <div v-click="4" class="text-sm" style="color: var(--ink-soft)">
          Scale the probe count inversely with selectivity, capped at <span class="mono" style="color: var(--ink)">16×</span> the base.
        </div> -->
      </div>
      <!-- Formula card -->
      <div v-click="4" class="rounded-sm px-6 py-5"
           style="background: #F4F6FB; border: 1px solid var(--rule)">
        <div class="mono text-xs uppercase tracking-wider mb-3" style="color: var(--ink-soft); letter-spacing: 0.15em">
          adaptive nprobe rule
        </div>
        <div class="mono text-base" style="color: var(--ink)">
          Scale the probe count inversely with selectivity, <br> capped at 16x the base
        </div>
        <!-- <div class="mono text-xs mt-3" style="color: var(--ink-soft)">
          capped at 16x the base
        </div>
        <div class="mono text-xs mt-1" style="color: var(--ink-soft)">
          capped at base × MAX_FACTOR (= 16)
        </div> -->
      </div>
      <!-- Recall result -->
      <div v-click="5">
        <RecallBars
          :rows="[
            { label: 'baseline',          value: 0.284, color: '#D14A4A' },
            { label: 'adaptive nprobe',   value: 0.958, color: '#2E5BFF', highlight: true },
          ]"
        />
      </div>
    </div>
  </div>
</div>
<div class="footer-rule"><span>7 · approach 1: adaptive nprobe</span></div>
---
layout: default
transition: fade
---
<!-- =====================================================================
SLIDE 9 — FIX 3: SYNOPSES (click-staged, 5 clicks)
3:00 – 3:30
Click 0: title + idea
Click 1: value-count table appears for one cluster
Click 2: predicate "category = books" overlays, count of 0 highlights
Click 3: skip badge appears
Click 4: stage swaps to Pareto plot
===================================================================== -->
<div class="absolute inset-0 px-20 pt-16 pb-16 flex flex-col">
  <div class="mb-6">
    <div class="eyebrow mb-3">approach 2 of 3</div>
    <h2 class="serif text-5xl">Probe the right clusters<br/>based on exact counts.</h2>
  </div>
  <div class="flex-1 relative">
    <!-- Stage A: synopsis table mechanism, visible until click 4 -->
    <div v-click="[0, 4]" class="absolute inset-0 grid grid-cols-12 gap-12 items-center">
      <!-- LEFT: explanation -->
      <div class="col-span-5 flex flex-col gap-6">
        <div class="text-lg leading-relaxed" style="color: var(--ink-soft)">
          For each cluster, store the exact number of documents matching every metadata value.
        </div>
        <div v-click="2" class="text-base leading-snug" style="color: var(--ink-soft)">
          A predicate hits the table directly. If the count is provably zero, skip.
        </div>
        <div v-click="3" class="text-sm leading-snug" style="color: var(--ink-soft)">
          No false positives. No false negatives. ~94 MB total for 1M docs.
        </div>
      </div>
      <!-- RIGHT: synopsis table for a cluster -->
      <div class="col-span-7 flex flex-col items-center gap-6">
        <!-- Cluster glyph -->
        <div class="flex items-center gap-6">
          <svg viewBox="0 0 100 100" class="w-24 h-24">
            <circle cx="50" cy="50" r="38" fill="#FBFAF7" stroke="#D9D6CE" stroke-width="1.5"/>
            <g fill="#0E1626" opacity="0.7">
              <circle cx="38" cy="42" r="2"/><circle cx="58" cy="36" r="2"/>
              <circle cx="65" cy="52" r="2"/><circle cx="48" cy="58" r="2"/>
              <circle cx="34" cy="56" r="2"/><circle cx="60" cy="64" r="2"/>
              <circle cx="44" cy="48" r="2"/><circle cx="50" cy="68" r="2"/>
            </g>
          </svg>
          <div class="mono text-xs" style="color: var(--ink-soft)">cluster 47<br/>26 docs</div>
        </div>
        <!-- Synopsis table -->
        <div v-click="1" class="rounded-sm overflow-hidden"
             style="border: 1px solid var(--rule); background: var(--paper); width: 22rem">
          <div class="mono text-xs uppercase px-4 py-2"
               style="background: #F4F6FB; color: var(--ink-soft); letter-spacing: 0.15em; border-bottom: 1px solid var(--rule)">
            synopsis · domain
          </div>
          <table class="w-full mono text-sm">
            <tbody>
              <tr style="border-bottom: 1px solid var(--rule)">
                <td class="px-4 py-2" style="color: var(--ink)">reddit.com</td>
                <td class="px-4 py-2 text-right" style="color: var(--ink); font-weight: 500">15</td>
              </tr>
              <tr style="border-bottom: 1px solid var(--rule)">
                <td class="px-4 py-2" style="color: var(--ink)">wikipedia.org</td>
                <td class="px-4 py-2 text-right" style="color: var(--ink); font-weight: 500">8</td>
              </tr>
              <tr style="border-bottom: 1px solid var(--rule)"
                  :class="{ 'synopsis-zero-row': true }">
                <td class="px-4 py-2"
                    :style="{ color: 'var(--ink)', transition: 'color 0.4s' }">wikihow.com</td>
                <td class="px-4 py-2 text-right"
                    :style="{ fontWeight: 500, transition: 'color 0.4s' }">
                  <span :style="{ color: 'var(--ink)' }">0</span>
                </td>
              </tr>
              <tr>
                <td class="px-4 py-2" style="color: var(--ink)">webmd.com</td>
                <td class="px-4 py-2 text-right" style="color: var(--ink); font-weight: 500">3</td>
              </tr>
            </tbody>
          </table>
        </div>
        <!-- Predicate + verdict -->
        <div v-click="2" class="flex items-center gap-4 mt-2">
          <div class="px-4 py-2 rounded-sm border-2 mono text-sm"
               :style="{ borderColor: 'var(--query)', color: 'var(--query)' }">
            where domain = "wikihow.com"
          </div>
          <div class="serif italic" style="color: var(--ink-soft)">→ count is 0 →</div>
          <div v-click="3" class="px-4 py-2 rounded-sm border-2"
               :style="{ borderColor: 'var(--miss)', color: 'var(--miss)' }">
            <span class="serif font-medium">skip</span>
          </div>
        </div>
      </div>
    </div>
    <!-- Stage B: side-by-side recall + latency bars, appears on click 4 -->
    <div v-click="4" class="absolute inset-0 flex flex-col px-4">
      <div class="serif italic text-2xl mb-6" style="color: var(--ink-soft)">results</div>
      <div class="flex-1 grid grid-cols-2 gap-16 items-center">
      <div class="flex flex-col items-center">
        <div class="eyebrow mb-6 self-start">recall@10</div>
        <RecallBars
          :rows="[
            { label: 'baseline',                   value: 0.284, color: '#D14A4A' },
            { label: 'adaptive nprobe',            value: 0.958, color: '#7B8AAE' },
            { label: 'adaptive + synopses',        value: 0.958, color: '#2E5BFF', highlight: true },
          ]"
          metric="recall"
          :decimals="3"
          caption="↑ recall@10 · higher is better"
        />
      </div>
      <div class="flex flex-col items-center">
        <div class="eyebrow mb-6 self-start">mean latency</div>
        <RecallBars
          :rows="[
            { label: 'baseline',                   value: 2.21,  color: '#7B8AAE' },
            { label: 'adaptive nprobe',            value: 23.80, color: '#D14A4A' },
            { label: 'adaptive + synopses',        value: 8.87,  color: '#2E5BFF', highlight: true },
          ]"
          metric="latency"
          unit="ms"
          :max="25"
          :decimals="2"
          caption="↓ mean query latency · lower is better"
        />
      </div>
      </div>
    </div>
  </div>
</div>
<div class="footer-rule"><span>8 · approach 2: synopses</span></div>
---
layout: default
transition: fade
---
<!-- =====================================================================
SLIDE 10 — FIX 2: BLOOM FILTERS (click-staged build, 7 clicks)
2:25 – 3:00
Each click reveals the next conceptual beat. Spacebar advances.
See slide_bloom_gate.md for the click-by-click VO mapping.
===================================================================== -->
<div class="absolute inset-0 px-16 pt-16 pb-16 flex flex-col">
  <div class="mb-8">
    <div class="eyebrow mb-3">approach 3 of 3</div>
    <h2 class="serif text-5xl">Probe the clusters<br/>probabilistically</h2>
  </div>
  <div class="flex-1 relative">
    <!-- ===== STAGE A: cluster + bit array intro ===== -->
    <div v-click="[0, 5]" class="absolute inset-0 flex items-center justify-center gap-12">
      <div class="flex flex-col items-center">
        <svg viewBox="0 0 180 180" class="w-56 h-56">
          <circle cx="90" cy="90" r="68" fill="#FBFAF7" stroke="#D9D6CE" stroke-width="2"/>
          <g fill="#0E1626" opacity="0.7">
            <circle cx="65" cy="70" r="3"/><circle cx="100" cy="58" r="3"/>
            <circle cx="115" cy="85" r="3"/><circle cx="85" cy="105" r="3"/>
            <circle cx="55" cy="95" r="3"/><circle cx="105" cy="115" r="3"/>
            <circle cx="78" cy="78" r="3"/><circle cx="70" cy="120" r="3"/>
            <circle cx="120" cy="105" r="3"/><circle cx="92" cy="125" r="3"/>
          </g>
        </svg>
        <div class="mono text-xs mt-2" style="color: var(--ink-soft)">cluster · 220 items</div>
      </div>
      <div v-click="1" class="flex flex-col items-center">
        <svg width="80" height="20">
          <line x1="0" y1="10" x2="68" y2="10" stroke="#2B3A55" stroke-width="1.5"/>
          <polyline points="60,4 68,10 60,16" fill="none" stroke="#2B3A55" stroke-width="1.5"/>
        </svg>
        <div class="text-xs mt-1" style="color: var(--ink-soft); font-style: italic; font-family: var(--slidev-fonts-serif)">summarized as</div>
      </div>
      <div v-click="2" class="flex flex-col items-center">
        <div class="serif italic text-sm mb-3" style="color: var(--ink-soft)">
          bloom filter · <span class="mono" style="color: var(--ink)">2 KB</span>
        </div>
        <div class="flex gap-[3px]">
          <div v-for="(bit, i) in bloomBits" :key="i"
               class="bloom-cell" :class="{ 'bloom-on': bit }"></div>
        </div>
        <div class="mt-6 h-12 text-sm" style="color: var(--ink-soft)">
          <div v-click="[3, 4]" class="mono">cheap to store · cheap to check</div>
          <div v-click="4" class="serif italic">
            answers <span style="color: var(--miss); font-style: normal; font-weight: 500">"definitely not in here"</span>
            with no false negatives
          </div>
        </div>
      </div>
    </div>
    <!-- ===== STAGE B: query, hash, skip ===== -->
    <div v-click="[5, 7]" class="absolute inset-0 flex flex-col items-center justify-center">
      <div class="mb-12 flex items-center gap-3">
        <div class="mono text-base" style="color: var(--ink-soft)">query:</div>
        <div class="px-4 py-2 rounded-sm border-2 mono text-base"
             :style="{ borderColor: 'var(--query)', color: 'var(--query)', background: 'var(--paper)' }">
          where domain = "wikihow.com"
        </div>
      </div>
      <div class="relative">
        <svg class="absolute" width="380" height="60" style="top: -55px; left: -10px">
          <path d="M 190 0 Q 100 30 65 55"  stroke="#2B3A55" stroke-width="1" fill="none" opacity="0.6"/>
          <path d="M 190 0 Q 190 30 185 55" stroke="#2B3A55" stroke-width="1" fill="none" opacity="0.6"/>
          <path d="M 190 0 Q 280 30 275 55" stroke="#2B3A55" stroke-width="1" fill="none" opacity="0.6"/>
        </svg>
        <div class="flex gap-[3px]">
          <div v-for="(bit, i) in bloomBits" :key="i"
               class="bloom-cell"
               :class="{
                 'bloom-on': bit,
                 'bloom-hit': (i === 3 || i === 17),
                 'bloom-miss': i === 11
               }"></div>
        </div>
        <div class="mt-2 mono text-xs flex justify-between" style="color: var(--ink-soft); width: 380px">
          <span>↑ bit 3 = <span style="color: var(--query); font-weight: 600">1</span></span>
          <span>↑ bit 11 = <span style="color: var(--miss); font-weight: 600">0</span></span>
          <span>↑ bit 17 = <span style="color: var(--query); font-weight: 600">1</span></span>
        </div>
      </div>
      <div v-click="6" class="mt-12 flex items-center gap-6">
        <div class="text-base" style="color: var(--ink-soft); font-style: italic; font-family: var(--slidev-fonts-serif)">
          one bit is zero →
        </div>
        <div class="px-6 py-3 rounded-sm border-2"
             :style="{ borderColor: 'var(--miss)', color: 'var(--miss)' }">
          <span class="serif text-2xl font-medium">skip</span>
          <span class="ml-3 mono text-xs" style="color: var(--ink-soft)">don't fetch this cluster</span>
        </div>
      </div>
    </div>
    <!-- ===== STAGE C: zoom out, many clusters ===== -->
    <div v-click="7" class="absolute inset-0 flex items-center justify-center gap-16">
      <div class="grid grid-cols-8 gap-3">
        <div v-for="i in 32" :key="i"
             class="cluster-mini"
             :class="{ 'cluster-keep': keepIndices.has(i - 1), 'cluster-skip': !keepIndices.has(i - 1) }">
          <svg viewBox="0 0 60 60" class="w-full h-full">
            <circle cx="30" cy="30" r="22" fill="none"
                    :stroke="keepIndices.has(i - 1) ? '#1E9E6A' : '#D9D6CE'"
                    :stroke-width="keepIndices.has(i - 1) ? 2 : 1"
                    :opacity="keepIndices.has(i - 1) ? 1 : 0.25"/>
            <g :opacity="keepIndices.has(i - 1) ? 0.75 : 0.15">
              <circle cx="22" cy="25" r="1.5" fill="#0E1626"/>
              <circle cx="35" cy="22" r="1.5" fill="#0E1626"/>
              <circle cx="38" cy="35" r="1.5" fill="#0E1626"/>
              <circle cx="25" cy="38" r="1.5" fill="#0E1626"/>
              <circle cx="32" cy="32" r="1.5" fill="#0E1626"/>
            </g>
          </svg>
        </div>
      </div>
      <div class="flex flex-col gap-6 max-w-xs">
        <div>
          <div class="bignum text-7xl" style="color: var(--match); line-height: 1">~72%</div>
          <div class="serif text-xl mt-2">skipped before fetch</div>
        </div>
        <div class="rule"></div>
        <div class="text-base leading-snug" style="color: var(--ink-soft)">
          Same probe budget as baseline.<br/>
          We just stop fetching clusters that<br/>
          can't possibly contain a match.
        </div>
        <div class="mono text-xs" style="color: var(--ink-soft)">
          32 candidates → ~9 fetched
        </div>
      </div>
    </div>
  </div>
  <div class="footer-rule"><span>8 · approach 3: bloom filters</span></div>
</div>
<script setup>
const bloomBits = [
  false, true,  false, true,                                 // 0–3   (bit 3 ON)
  false, true,  false, true,  true,  false, false, false,    // 4–11  (bit 11 OFF)
  false, false, true,  false, false, true,                   // 12–17 (bit 17 ON)
  false, true,  false, false, true,  false                   // 18–23
];
const keepIndices = new Set([3, 7, 12, 18, 22, 25, 28, 30]);
</script>
---
layout: default
transition: fade
---
<!-- =====================================================================
SLIDE 9b — FINAL RESULTS COMPARISON
All 8 strategy combinations on SIFT1M, N=1M, 1% selectivity.
===================================================================== -->
<div class="absolute inset-0 flex flex-col px-16 pt-16 pb-16">
  <div class="mb-6">
    <div class="eyebrow mb-3">final results</div>
    <h2 class="serif text-4xl">
      Combine the gates:
      <span style="color: var(--query)">Adaptive</span>
      +
      <span style="color: var(--query)">Bloom filters</span>
      ideal
    </h2>
  </div>
  <div class="flex-1 grid grid-cols-2 gap-8 items-center min-h-0">
    <img src="/plots/recall_1M_sel1pct.png"
         class="w-full h-full object-contain rounded-sm"
         style="border: 1px solid var(--rule)"
         alt="Recall by strategy — SIFT1M, N=1M, 1% filter selectivity"/>
    <img src="/plots/latency_1M_sel1pct.png"
         class="w-full h-full object-contain rounded-sm"
         style="border: 1px solid var(--rule)"
         alt="Latency by strategy — SIFT1M, N=1M, 1% filter selectivity"/>
  </div>
  <div class="caption mt-4 text-center">
    SIFT1M · N = 1M · 1% filter selectivity · k = 10 · all eight gate combinations
  </div>
</div>
<div class="footer-rule"><span>9 · final results</span></div>
---
layout: default
transition: fade
---
<!-- =====================================================================
SLIDE 11 — DEMO
3:30 – 4:10
Title card — switch to live demo after this slide.
===================================================================== -->
<div class="absolute inset-0 flex flex-col justify-center items-center px-24">
  <div class="eyebrow mb-10">interlude</div>
  <h1 class="serif text-center" style="font-size: 6rem; line-height: 1">
    <span style="color: var(--query)">demo.</span>
  </h1>
  <div class="rule mt-12"></div>
  <div class="caption mt-6 text-center">
    MS MARCO dataset
  </div>
</div>
<div class="footer-rule"><span>10 · demo</span></div>
---
layout: default
transition: fade
---
<!-- =====================================================================
SLIDE 12 — WHY IT MATTERS
4:10 – 4:35
Connect the SIFT1M numbers to the real workload Chroma serves: RAG.
The demo query (MS MARCO, domain="wikihow") makes the case concrete.
===================================================================== -->
<div class="absolute inset-0 px-20 pt-16 pb-16 flex flex-col">
  <div class="mb-8">
    <div class="eyebrow mb-3">why this matters</div>
    <h2 class="serif text-4xl max-w-4xl">
      Chroma's biggest workload is
      <span style="color: var(--query)">RAG</span> —
      and almost every RAG query is
      <span style="color: var(--miss)">predicate-filtered</span>.
    </h2>
  </div>
  <div class="grid grid-cols-12 gap-10 flex-1 items-stretch">
    <!-- LEFT: the RAG predicate reality -->
    <div class="col-span-7 flex flex-col gap-5 justify-center">
      <div class="text-base leading-relaxed" style="color: var(--ink-soft)">
        Retrieval rarely runs over the whole corpus. The agent already knows
        the user, the tenant, the document set, the language, the recency
        window — and bakes that into a
        <span class="mono" style="color: var(--ink)">where</span> clause.
      </div>
      <div class="rounded-sm px-5 py-4 mono text-sm"
           style="background: #F4F6FB; border: 1px solid var(--rule); color: var(--ink); line-height: 1.6">
        <span style="color: #7B8AAE"># our demo: a wikiHow-style RAG retrieval</span><br/>
        <span style="color: #7B8AAE">collection</span>.query(<br/>
        &nbsp;&nbsp;query_texts=[<span style="color: var(--query)">"how do I unclog a drain?"</span>],<br/>
        &nbsp;&nbsp;n_results=<span style="color: var(--query)">10</span>,<br/>
        &nbsp;&nbsp;<span style="background: #FFF7CC; padding: 0 0.2rem; border-radius: 2px">where={<span style="color: var(--miss)">"domain"</span>: <span style="color: var(--miss)">"wikihow.com"</span>},</span><br/>
        )
      </div>
    </div>
    <!-- RIGHT: the two things that move -->
    <div class="col-span-5 flex flex-col gap-6 justify-center">
      <div>
        <div class="eyebrow mb-2" style="color: var(--match)">why recall matters</div>
        <div class="serif text-2xl mb-1">The model only sees what we retrieve.</div>
        <div class="text-sm leading-snug" style="color: var(--ink-soft)">
          Lost neighbors → wrong context → hallucinations. 
        </div>
      </div>
      <div class="rule"></div>
      <div>
        <div class="eyebrow mb-2" style="color: var(--query)">why latency matters</div>
        <div class="serif text-2xl mb-1">Retrieval blocks the first token.</div>
        <div class="text-sm leading-snug" style="color: var(--ink-soft)">
          Every ms here is a ms the user waits before the LLM streams
        </div>
      </div>
    </div>
  </div>
</div>
<div class="footer-rule"><span>11 · why it matters</span></div>
---
layout: default
transition: fade
---
<!-- =====================================================================
SLIDE 12b — SUMMARY
Recap of the three gates and the headline numbers.
===================================================================== -->
<div class="absolute inset-0 px-20 pt-16 pb-16 flex flex-col">
  <div class="mb-8">
    <div class="eyebrow mb-3">summary</div>
    <h2 class="serif text-4xl">
      Three gates on the routing stage of SPANN.
    </h2>
  </div>
  <div class="grid grid-cols-3 gap-8 flex-1">
    <div class="border-l-2 pl-6 flex flex-col" style="border-color: var(--rule)">
      <div class="bignum" style="color: var(--query)">1</div>
      <div class="serif text-2xl mt-2">Adaptive nprobe</div>
      <!-- <div class="text-sm mt-3" style="color: var(--ink-soft)">
        Probe more clusters when the filter is selective. Scales inversely with
        selectivity, capped at 8× the base.
      </div> -->
    </div>
    <div class="border-l-2 pl-6 flex flex-col" style="border-color: var(--rule)">
      <div class="bignum" style="color: var(--query)">2</div>
      <div class="serif text-2xl mt-2">Bloom-filter gate</div>
      <!-- <div class="text-sm mt-3" style="color: var(--ink-soft)">
        Per-cluster bloom over metadata values. Skip the fetch when the
        predicate provably can't match.
      </div> -->
    </div>
    <div class="border-l-2 pl-6 flex flex-col" style="border-color: var(--rule)">
      <div class="bignum" style="color: var(--query)">3</div>
      <div class="serif text-2xl mt-2">Metadata synopses</div>
      <!-- <div class="text-sm mt-3" style="color: var(--ink-soft)">
        Exact per-value counts per cluster. Zero counts → guaranteed skip,
        no false positives.
      </div> -->
    </div>
  </div>
  <div class="mt-8 rounded-sm px-8 py-5 flex items-center justify-between"
       style="background: #F4F6FB; border: 1px solid var(--rule)">
    <div>
      <div class="eyebrow mb-1">winner</div>
      <div class="serif text-2xl">
        <span style="color: var(--query)">Adaptive + bloom</span>
        is the best of the lot.
      </div>
    </div>
    <div class="mono text-sm text-right" style="color: var(--ink-soft)">
      <span style="color: var(--ink); font-weight: 500">0.958</span> recall@10
      &nbsp;·&nbsp;
      <span style="color: var(--ink); font-weight: 500">8.2 ms</span> mean latency<br/>
      <span style="font-size: 0.72rem">SIFT1M · N = 1M · 1% filter selectivity</span>
    </div>
  </div>
</div>
<div class="footer-rule"><span>12 · summary</span></div>
---
layout: center
transition: slide-up
---
<!-- =====================================================================
SLIDE 13 — TAKEAWAY
4:35 – 4:50
===================================================================== -->
<div class="px-24 max-w-5xl text-center">
  <div class="eyebrow mb-10">takeaway</div>
  <h2 class="serif" style="font-size: 3.6rem; line-height: 1.15">
    Filter at the routing stage,<br/>
    not just at ranking.
  </h2>
  <div class="rule mx-auto mt-12"></div>
  <div class="text-base mt-6" style="color: var(--ink-soft)">
    A few KB of summary per cluster recovers most of the recall<br/>
    that selective filters cost you.
  </div>
</div>
---
layout: default
transition: fade
---
<!-- =====================================================================
SLIDE 14 — CREDITS
4:50 – 5:00
===================================================================== -->
<div class="absolute inset-0 flex flex-col justify-end px-24 pb-32">
  <div class="serif text-2xl leading-relaxed">
    <span style="color: var(--ink)">Zach Marinov</span>
    <span style="color: var(--ink-soft)"> · </span>
    <span style="color: var(--ink)">Kartik Pingle</span>
    <span style="color: var(--ink-soft)"> · </span>
    <span style="color: var(--ink)">Evan Kim</span>
  </div>
  <div class="rule mt-6"></div>
  <div class="text-base mt-4" style="color: var(--ink-soft)">
    6.5830 Database Systems · Spring 2026 ·
    Thanks to Hammad Bashir (Chroma), Tianyu Li, and Mike Cafarella.
  </div>
</div>
