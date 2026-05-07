<template>
  <svg viewBox="0 0 520 420" class="w-full max-w-2xl">
    <!-- axes -->
    <g stroke="#2B3A55" stroke-width="1" fill="none">
      <line x1="70" y1="60" x2="70" y2="350"/>
      <line x1="70" y1="350" x2="470" y2="350"/>
    </g>

    <!-- axis labels -->
    <text x="70" y="40" font-family="Newsreader, serif" font-style="italic"
          font-size="13" fill="#2B3A55">recall@10</text>
    <text x="470" y="380" text-anchor="end" font-family="Newsreader, serif"
          font-style="italic" font-size="13" fill="#2B3A55">heads fetched →</text>

    <!-- y-axis ticks -->
    <g font-family="ui-monospace, monospace" font-size="10" fill="#2B3A55">
      <text x="62" y="68" text-anchor="end">1.0</text>
      <text x="62" y="148" text-anchor="end">0.8</text>
      <text x="62" y="228" text-anchor="end">0.6</text>
      <text x="62" y="308" text-anchor="end">0.4</text>
      <line x1="68" y1="65" x2="72" y2="65" stroke="#2B3A55"/>
      <line x1="68" y1="145" x2="72" y2="145" stroke="#2B3A55"/>
      <line x1="68" y1="225" x2="72" y2="225" stroke="#2B3A55"/>
      <line x1="68" y1="305" x2="72" y2="305" stroke="#2B3A55"/>
    </g>

    <!-- x-axis ticks -->
    <g font-family="ui-monospace, monospace" font-size="10" fill="#2B3A55">
      <text x="120" y="365" text-anchor="middle">50</text>
      <text x="220" y="365" text-anchor="middle">100</text>
      <text x="320" y="365" text-anchor="middle">175</text>
      <text x="420" y="365" text-anchor="middle">256</text>
    </g>

    <!-- Pareto frontier hint, very faint -->
    <path d="M 130 75 Q 220 80 320 200 T 420 78"
          stroke="#D9D6CE" stroke-width="1" stroke-dasharray="3,3" fill="none"/>

    <!-- Baseline: 0.954 recall, 256 heads -->
    <g>
      <circle cx="420" cy="78" r="6" fill="none" stroke="#7B8AAE" stroke-width="2"/>
      <text x="430" y="82" font-family="Newsreader, serif" font-size="13" fill="#2B3A55">
        baseline
      </text>
    </g>

    <!-- Bloom: 0.956 recall, 67 heads -->
    <g>
      <circle cx="138" cy="76" r="6" fill="#7B8AAE"/>
      <text x="148" y="80" font-family="Newsreader, serif" font-size="13" fill="#2B3A55">
        bloom
      </text>
    </g>

    <!-- Synopsis: 0.964 recall, 70 heads — the win -->
    <g>
      <circle cx="142" cy="71" r="9" fill="#1E9E6A"/>
      <circle cx="142" cy="71" r="16" fill="none" stroke="#1E9E6A" stroke-width="1" opacity="0.4"/>
      <text x="158" y="68" font-family="Newsreader, serif" font-size="14"
            fill="#0E1626" font-weight="500">synopsis</text>
      <text x="158" y="84" font-family="ui-monospace, monospace" font-size="10" fill="#2B3A55">
        recall 0.96 · 70 heads
      </text>
    </g>

    <!-- Adaptive: 0.890, 192 heads — degrades -->
    <g>
      <circle cx="328" cy="170" r="5" fill="#7B8AAE" opacity="0.8"/>
      <text x="338" y="174" font-family="Newsreader, serif" font-size="12"
            fill="#2B3A55" opacity="0.85">adaptive</text>
    </g>

    <!-- "better" arrow -->
    <g transform="translate(195, 180)">
      <path d="M 0 0 L -45 -65" stroke="#1E9E6A" stroke-width="1" fill="none"
            stroke-dasharray="2,3"/>
      <text x="0" y="14" font-family="Newsreader, serif" font-style="italic"
            font-size="12" fill="#1E9E6A">better</text>
    </g>

    <text x="270" y="405" text-anchor="middle" font-family="Newsreader, serif"
          font-style="italic" font-size="12" fill="#2B3A55">
      N = 100K · 1% selectivity
    </text>
  </svg>
</template>
