import { useEffect, useState } from 'react'
import type { Hardware } from './types'

export const HISTORY_MS = 60_000
export type HistoryPoint = { time: number; values: Record<string, number | null>; segment?: string }
export type ChartBounds = { minimum: number; maximum: number; updatedAt: number }
export const finite = (v: unknown): number | null => typeof v === 'number' && Number.isFinite(v) ? v : null
export const numberText = (v: unknown, digits = 2, fixed = false) => finite(v) === null ? '不可用' : (v as number).toLocaleString('zh-CN', { minimumFractionDigits: fixed ? digits : 0, maximumFractionDigits: digits })
export function rateText(v: unknown) {
  const n = finite(v)
  if (n === null) return '不可用'
  if (n > 0 && n < .001) return '<0.001'
  const digits = n < 10 ? 3 : n < 100 ? 2 : n < 1000 ? 1 : 0
  return n.toLocaleString('zh-CN', { minimumFractionDigits: digits, maximumFractionDigits: digits })
}
export function gpuList(hardware: Hardware) {
  return hardware.gpus?.length ? hardware.gpus : hardware.gpu && typeof hardware.gpu === 'object' ? [hardware.gpu as Record<string, unknown>] : []
}
export const gpuKey = (gpu: Record<string, unknown> | undefined) => String(gpu?.id ?? gpu?.index ?? gpu?.name ?? 'none')
export function gpuLabel(gpu: Record<string, unknown>) {
  const identity = gpu.index != null ? `GPU ${gpu.index}` : gpu.pci_bus_id ? String(gpu.pci_bus_id) : String(gpu.id ?? '').slice(-8)
  return [identity, gpu.name].filter(Boolean).join(' · ') || 'GPU'
}
export const pcieLink = (gpu: Record<string, unknown> | undefined, maximum = false) => {
  const gen = finite(gpu?.[maximum ? 'pcie_gen_max' : 'pcie_gen']), width = finite(gpu?.[maximum ? 'pcie_width_max' : 'pcie_width'])
  return gen !== null && width !== null ? `PCIe ${gen}.0 ×${width}` : '不可用'
}

export function appendHistory(history: HistoryPoint[], sample: HistoryPoint) {
  if (history.at(-1)?.time === sample.time) return history
  const recent = history.filter(point => point.time < sample.time && point.time >= sample.time - HISTORY_MS)
  return [...recent, sample]
}

export function useHistory(key: string, sample: HistoryPoint | null) {
  const [state, setState] = useState<{ key: string; points: HistoryPoint[] }>({ key, points: [] })
  useEffect(() => {
    setState(previous => {
      const points = previous.key === key ? previous.points : []
      return sample ? { key, points: appendHistory(points, sample) } : previous.key === key ? previous : { key, points }
    })
  }, [key, sample?.time])
  return state.key === key ? state.points : []
}

export function seriesSegments(points: HistoryPoint[], id: string) {
  const segments: Array<Array<{ time: number; value: number }>> = []
  let current: Array<{ time: number; value: number }> = []
  let segment: string | undefined
  for (const point of points) {
    const value = finite(point.values[id])
    if (value === null || (current.length && (point.time - current.at(-1)!.time > 2500 || point.segment !== segment))) {
      if (current.length) segments.push(current)
      current = []
    }
    if (value !== null) current.push({ time: point.time, value })
    segment = point.segment
  }
  if (current.length) segments.push(current)
  return segments
}

// Use a meaningful absolute/relative span, rounded limits and a ten-second
// shrink delay so small fluctuations do not fill the plot or move its axis.
export function adaptiveBounds(values: Array<number | null>, minimumSpan: number, time: number, previous?: ChartBounds): ChartBounds {
  const valid = values.filter((value): value is number => finite(value) !== null && value! >= 0)
  if (!valid.length) return { minimum: 0, maximum: minimumSpan, updatedAt: time }
  const low = Math.min(...valid), high = Math.max(...valid)
  const span = Math.max(high - low, minimumSpan, (low + high) / 2 * .2)
  const margin = (span - high + low) / 2 + span * .15
  const magnitude = 10 ** Math.floor(Math.log10(span / 8))
  const step = [1, 2, 5, 10].find(value => value * magnitude >= span / 8)! * magnitude
  const minimum = Math.max(0, Math.floor((low - margin) / step) * step)
  const candidate = { minimum, maximum: Math.ceil(Math.max(high + margin, minimum + span) / step) * step, updatedAt: time }
  if (!previous) return candidate
  const oldSpan = previous.maximum - previous.minimum
  const inside = low >= previous.minimum + (previous.minimum > 0 ? oldSpan * .1 : 0)
    && high <= previous.maximum - oldSpan * .1
  if (inside && (candidate.maximum - candidate.minimum >= oldSpan * .75 || time - previous.updatedAt < 10_000)) return previous
  return candidate
}
