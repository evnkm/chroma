<template>
  <div class="w-full max-w-2xl">
    <div v-for="(row, i) in rows" :key="i" class="mb-7 last:mb-0">
      <div class="flex justify-between items-baseline mb-2">
        <span class="text-base" :style="{
          color: row.highlight ? 'var(--ink)' : 'var(--ink-soft)',
          fontWeight: row.highlight ? 500 : 400
        }">
          {{ row.label }}
        </span>
        <span class="text-sm font-mono" style="color: var(--ink-soft)">
          {{ metric }} <span :style="{ color: row.color, fontWeight: 600 }">
            {{ row.placeholder || row.value.toFixed(decimals) }}{{ unit ? ' ' + unit : '' }}
          </span>
        </span>
      </div>
      <div class="h-3 rounded-sm" style="background: #EFEDE6">
        <div
          class="h-full rounded-sm transition-all"
          :style="{
            width: ((row.value / max) * 100) + '%',
            background: row.color,
            opacity: row.highlight === false ? 0.55 : 1,
            transition: 'width 0.6s ease-out'
          }"
        ></div>
      </div>
    </div>

    <div class="mt-6 text-xs font-mono" style="color: var(--ink-soft)">
      {{ caption }}
    </div>
  </div>
</template>

<script setup>
defineProps({
  rows:     { type: Array,  required: true },
  metric:   { type: String, default: 'recall' },
  unit:     { type: String, default: '' },
  max:      { type: Number, default: 1.0 },
  decimals: { type: Number, default: 2 },
  caption:  { type: String, default: '↑ recall@10 · 1% selectivity · k = 10' }
});
</script>
