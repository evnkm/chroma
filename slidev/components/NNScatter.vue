<template>
  <svg viewBox="0 0 520 420" preserveAspectRatio="xMidYMid meet" class="scatter-svg">
    <!-- background grid -->
    <defs>
      <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
        <path d="M 40 0 L 0 0 0 40" fill="none" stroke="var(--grid, #E8E5DC)" stroke-width="0.6"/>
      </pattern>
      <radialGradient id="qpulse">
        <stop offset="0%" stop-color="#2E5BFF" stop-opacity="0.35"/>
        <stop offset="100%" stop-color="#2E5BFF" stop-opacity="0"/>
      </radialGradient>
    </defs>
    <rect width="520" height="420" fill="url(#grid)"/>

    <!-- ambient dots -->
    <g fill="#0E1626" opacity="0.55">
      <circle v-for="(p, i) in points" :key="i" :cx="p.x" :cy="p.y" :r="p.r"/>
    </g>

    <!-- query glow -->
    <circle cx="260" cy="210" r="58" fill="url(#qpulse)"/>

    <!-- nearest neighbors -->
    <g>
      <circle v-for="(n, i) in neighbors" :key="'n'+i"
              :cx="n.x" :cy="n.y" r="6"
              fill="#1E9E6A" stroke="#FBFAF7" stroke-width="2"/>
    </g>

    <!-- query point -->
    <circle cx="260" cy="210" r="7" fill="#2E5BFF" stroke="#FBFAF7" stroke-width="2.5"/>

    <!-- bounded caption; raw SVG text can overflow its parent in Slidev scaling -->
    <foreignObject x="40" y="378" width="440" height="34">
      <div xmlns="http://www.w3.org/1999/xhtml" class="scatter-caption">
        embedding space (2D for the picture · 768D in practice)
      </div>
    </foreignObject>
  </svg>
</template>

<script setup>
// Deterministic-ish poisson-like scatter
const rand = (seed) => {
  let s = seed;
  return () => { s = (s * 9301 + 49297) % 233280; return s / 233280; };
};
const r = rand(42);
const points = Array.from({ length: 110 }, () => ({
  x: 30 + r() * 460,
  y: 30 + r() * 340,
  r: 1.6 + r() * 1.6,
}));

// Hand-place a few neighbors near the query at (260, 210)
const neighbors = [
  { x: 244, y: 198 }, { x: 278, y: 196 }, { x: 252, y: 232 },
  { x: 286, y: 224 }, { x: 240, y: 224 },
];
</script>

<style scoped>
.scatter-svg {
  display: block;
  width: min(100%, 540px);
  height: auto;
  overflow: visible;
}

.scatter-caption {
  width: 100%;
  text-align: center;
  font-family: var(--slidev-fonts-serif);
  font-style: italic;
  font-size: 14px;
  line-height: 1.2;
  color: #2B3A55;
}
</style>
