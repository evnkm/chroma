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
          recall <span :style="{ color: row.color, fontWeight: 600 }">
            {{ row.placeholder || row.value.toFixed(2) }}
          </span>
        </span>
      </div>
      <div class="h-3 rounded-sm" style="background: #EFEDE6">
        <div
          class="h-full rounded-sm transition-all"
          :style="{
            width: (row.value * 100) + '%',
            background: row.color,
            opacity: row.highlight === false ? 0.55 : 1,
            transition: 'width 0.6s ease-out'
          }"
        ></div>
      </div>
    </div>

    <div class="mt-6 text-xs font-mono" style="color: var(--ink-soft)">
      ↑ recall@10 · 0.1% selectivity · k = 10
    </div>
  </div>
</template>

<script setup>
defineProps({
  rows: { type: Array, required: true }
});
</script>
