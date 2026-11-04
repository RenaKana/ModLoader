/** Project Strata's metadata-only allocation plan against the current hardware. */
export interface StrataResidentPlan {
  mode: 'resident' | 'arena' | 'mmap'
  gpu_device: string
  total_expert_bytes: number
  max_blob_bytes: number
  layer_blob_bytes: number[]
  rank_runs: [number, number][]
  gpu_rank_count: number
  unranked_counts: number[]
  cache_slots: number | null
  per_layer: boolean
  experts_per_layer: number
  headroom_bytes: number
  budget_bytes: number | null
  soft: boolean
  host_bytes: number
  exchange_bytes: number
  prefill: {
    auto: boolean
    borrow: boolean
    lend_percent: number
    ring_slots: number
    candidates: [number, number, number][]
    fallback: [number, number, number]
  } | null
}

interface Capacity {
  ram_available: number | null
  commit_available: number | null
  gpu_available: number | null
  gpu_fixed: number | null
  gpu_reserve: number | null
}

const align = (value: number) => Math.ceil(value / 256) * 256
const TRIM_MARGIN = 256 * 1024 ** 2

export function projectStrataMemory(plan: StrataResidentPlan, capacity: Capacity): { bytes: number; quality: 'estimate' | 'partial'; limited: boolean; shortfall?: number } {
  const { host_bytes: host, layer_blob_bytes: blobs } = plan
  const unknown = { bytes: host + (plan.mode === 'arena' ? plan.total_expert_bytes : 0), quality: 'partial' as const, limited: false }
  if (capacity.gpu_available === null || capacity.gpu_fixed === null || capacity.gpu_reserve === null) return unknown

  const availableGpu = Math.max(0, capacity.gpu_available - capacity.gpu_fixed - capacity.gpu_reserve)
  const slots = Math.floor(availableGpu / plan.max_blob_bytes)
  const cacheBudget = plan.cache_slots === null ? slots * plan.max_blob_bytes : Math.min(availableGpu, plan.cache_slots * plan.max_blob_bytes)
  const quota = plan.per_layer ? Math.min(plan.experts_per_layer, Math.floor(cacheBudget / blobs.reduce((sum, size) => sum + align(size), 0))) : Infinity
  const selected = blobs.map(() => 0)
  const gpuRuns: [number, number][] = [], remainingRuns: [number, number][] = []
  let gpuBytes = 0, gpuPayload = 0, gpuSlots = 0, ranked = 0, cacheFull = false
  for (const [layer, count] of plan.rank_runs) {
    const rankedCount = Math.max(0, Math.min(count, plan.gpu_rank_count - ranked))
    const size = align(blobs[layer])
    const take = cacheFull ? 0 : Math.max(0, Math.min(rankedCount, quota - selected[layer], Math.floor((cacheBudget - gpuBytes) / size)))
    if (take) {
      gpuRuns.push([layer, take])
      selected[layer] += take; gpuSlots += take
      gpuBytes += take * size; gpuPayload += take * blobs[layer]
    }
    if (take < count) remainingRuns.push([layer, count - take])
    if (!plan.per_layer && take < rankedCount) cacheFull = true
    ranked += count
  }
  if (plan.per_layer) gpuRuns.splice(0, gpuRuns.length, ...selected.flatMap((count, layer): [number, number][] => count ? [[layer, count]] : []))

  // A borrowed prefill arena consumes complete tail slots. Keep the engine's
  // 128-slot guard, and its stricter automatic lending percentage.
  let borrowed = 0, hostAfter = plan.prefill?.fallback[2] ?? 0
  if (plan.prefill?.borrow) for (const [, bytes, after] of plan.prefill.candidates) {
    let needed = bytes, count = 0
    for (let i = gpuRuns.length - 1; i >= 0 && needed > 0; i--) {
      const [layer, length] = gpuRuns[i]
      const take = Math.min(length, Math.ceil(needed / align(blobs[layer])))
      needed -= take * align(blobs[layer]); count += take
    }
    if (needed <= 0 && count + 128 <= gpuSlots && (!plan.prefill.auto || count * 100 <= plan.prefill.lend_percent * gpuSlots)) {
      borrowed = count; hostAfter = after
    }
  }
  if (plan.mode !== 'resident') return { bytes: host + hostAfter + (plan.mode === 'arena' ? plan.total_expert_bytes : 0), quality: 'estimate' as const, limited: false }
  if (capacity.ram_available === null) return unknown
  const availableRam = Math.min(capacity.ram_available, capacity.commit_available ?? capacity.ram_available)
  const room = Math.max(0, availableRam - host - plan.headroom_bytes)
  const complement = Math.max(0, plan.total_expert_bytes - gpuPayload)
  let resident = complement
  const budgetMode = plan.budget_bytes !== null || (plan.soft && complement > room)
  if (budgetMode) {
    const requested = plan.budget_bytes ?? complement
    let budget = requested > room ? Math.max(0, room - TRIM_MARGIN) : requested
    resident = 0
    // Native RAM selection skips an oversized expert and tries later ranks.
    for (const [layer, count] of remainingRuns) {
      const take = Math.min(count, Math.floor(budget / blobs[layer]))
      resident += take * blobs[layer]; budget -= take * blobs[layer]
    }
  } else if (complement <= room) {
    let remaining = room - complement
    for (let i = gpuRuns.length - 1; i >= 0 && borrowed > 0; i--) {
      const [layer, count] = gpuRuns[i]
      const wanted = Math.min(count, borrowed)
      const take = Math.min(wanted, Math.floor(remaining / blobs[layer]))
      resident += take * blobs[layer]; remaining -= take * blobs[layer]; borrowed -= take
      if (take < wanted) break
    }
  }
  return { bytes: host + resident + (resident > 0 ? plan.exchange_bytes : 0) + hostAfter,
    quality: 'estimate', limited: resident < complement,
    shortfall: !plan.soft && plan.budget_bytes === null ? Math.max(0, complement - room) : 0 }
}
