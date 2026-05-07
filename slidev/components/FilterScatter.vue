<template>
  <svg
    viewBox="0 0 520 420"
    preserveAspectRatio="xMidYMid meet"
    class="scatter-svg"
  >
    <defs>
      <pattern id="grid2" width="40" height="40" patternUnits="userSpaceOnUse">
        <path
          d="M 40 0 L 0 0 0 40"
          fill="none"
          stroke="var(--grid, #E8E5DC)"
          stroke-width="0.6"
        />
      </pattern>
    </defs>
    <rect width="520" height="420" fill="url(#grid2)" />

    <!-- non-matching dots, very dim -->
    <g fill="#0E1626" opacity="0.12">
      <circle
        v-for="(p, i) in nonMatches"
        :key="'nm' + i"
        :cx="p.x"
        :cy="p.y"
        :r="p.r"
      />
    </g>

    <!-- matching dots, scattered, full opacity -->
    <g>
      <circle
        v-for="(m, i) in matches"
        :key="'m' + i"
        :cx="m.x"
        :cy="m.y"
        r="4.5"
        fill="#1E9E6A"
        stroke="#FBFAF7"
        stroke-width="1.5"
      />
    </g>

    <!-- query at center -->
    <circle
      cx="260"
      cy="210"
      r="7"
      fill="#2E5BFF"
      stroke="#FBFAF7"
      stroke-width="2.5"
    />

    <!-- WHERE clause label -->
    <g transform="translate(374, 32)">
      <rect
        x="0"
        y="0"
        width="124"
        height="42"
        rx="4"
        fill="#FBFAF7"
        stroke="#D9D6CE"
        stroke-width="1"
      />
      <foreignObject x="10" y="7" width="104" height="30">
        <div xmlns="http://www.w3.org/1999/xhtml" class="where-label">
          <div>WHERE category</div>
          <div class="where-value">= "shoes"</div>
        </div>
      </foreignObject>
    </g>

    <foreignObject x="40" y="378" width="440" height="34">
      <div xmlns="http://www.w3.org/1999/xhtml" class="scatter-caption">
        ~0.1% of items match · matches scattered uniformly
      </div>
    </foreignObject>
  </svg>
</template>

<script setup>
const rand = (seed) => {
  let s = seed;
  return () => {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };
};
const r = rand(7);
const nonMatches = Array.from({ length: 180 }, () => ({
  x: 30 + r() * 460,
  y: 30 + r() * 340,
  r: 1.6 + r() * 1.6,
}));
const matches = Array.from({ length: 8 }, () => ({
  x: 30 + r() * 460,
  y: 30 + r() * 340,
}));
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
  color: #2b3a55;
}

.where-label {
  width: 100%;
  font-family: var(--slidev-fonts-mono);
  font-size: 12px;
  line-height: 1.3;
  color: #2b3a55;
  white-space: nowrap;
}

.where-value {
  color: #d14a4a;
  font-weight: 700;
}
</style>
