import { area, curveMonotoneX, line } from 'd3-shape'
import { finite } from './resource-metrics'
import type { RequestHistoryUsageBucket } from './types'

export const usageSeries = [
  { key: 'total', label: '总 Token' }, { key: 'hit', label: '缓存命中率' },
  { key: 'input', label: '输入（含缓存）' }, { key: 'output', label: '输出' }, { key: 'cached', label: '缓存命中量' },
] as const
export type UsageSeries = typeof usageSeries[number]['key']
export function usageValue(bucket: RequestHistoryUsageBucket, key: UsageSeries) {
  return key === 'hit' ? bucket.cache_ratio == null ? null : bucket.cache_ratio * 100 : finite(bucket[key].value)
}
export function usagePlotValue(bucket: RequestHistoryUsageBucket, key: UsageSeries) {
  return key === 'hit' && bucket.request_count === 0 ? 0 : usageValue(bucket, key)
}
export function usagePartial(bucket: RequestHistoryUsageBucket) {
  return bucket.request_count > 0 && [bucket.total, bucket.input, bucket.output, bucket.cached, bucket.cache_pair].some(item => item.reported < bucket.request_count)
}
export function usageSlots(count: number, left: number, right: number) {
  return Array.from({ length: count }, (_, index) => count === 1 ? (left + right) / 2 : left + index * (right - left) / (count - 1))
}
export function nearestUsageSlot(pixel: number, count: number, left: number, right: number) {
  return Math.max(0, Math.min(count - 1, Math.round((pixel - left) / (right - left) * (count - 1))))
}
export function usageHitRange(slots: number[], index: number, left: number, right: number) {
  return [index ? (slots[index - 1] + slots[index]) / 2 : left, index < slots.length - 1 ? (slots[index] + slots[index + 1]) / 2 : right]
}
export function usagePaths(points: Array<{ x: number; value: number | null }>, y: (value: number) => number, bottom: number) {
  const known = points.filter((point): point is { x: number; value: number } => finite(point.value) !== null)
  if (known.length < 2) return { line: '', area: '' }
  return { line: line<typeof known[number]>().x(point => point.x).y(point => y(point.value)).curve(curveMonotoneX)(known) ?? '',
    area: area<typeof known[number]>().x(point => point.x).y0(bottom).y1(point => y(point.value)).curve(curveMonotoneX)(known) ?? '' }
}
export function nextUsageBoundary(next: number, bucketSeconds: number, now: number) {
  return next > now ? next : next + (Math.floor((now - next) / bucketSeconds) + 1) * bucketSeconds
}
