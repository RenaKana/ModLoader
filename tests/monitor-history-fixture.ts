import { expect, type Locator } from '@playwright/test'
import type { ExternalRequestRecord, RequestHistoryGroup, RequestHistoryPage, RequestHistoryStats, RequestHistoryUsage } from '../src/types'

export function historyUsage(summary = historyStats([])): RequestHistoryUsage {
  const now = Date.now() / 1000
  if (!summary.request_count) summary = { ...summary, total: { value: 0, reported: 0 }, input: { value: 0, reported: 0 }, output: { value: 0, reported: 0 }, cached: { value: 0, reported: 0 } }
  return { as_of: now, revision: 'fixture', summary, coverage: { legacy_message_count: 0, legacy_metrics_message_count: 0 }, issues: [],
    trend: { start_at: null, end_at: null, bucket_seconds: 60, next_refresh_at: (Math.floor(now / 60) + 1) * 60, buckets: [] } }
}

// Mock the public grouped-history response without using production aggregation.
export function historyStats(records: readonly ExternalRequestRecord[]): RequestHistoryStats {
  const numeric = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value)
  const totals = (values: (number | null | undefined)[]) => {
    const known = values.filter(numeric)
    return { value: known.length ? known.reduce((sum, value) => sum + value, 0) : null, reported: known.length }
  }
  const input = records.map(record => record.usage.prompt_tokens), output = records.map(record => record.usage.completion_tokens)
  const cached = records.map(record => record.usage.prompt_tokens_details?.cached_tokens)
  const pairs = records.filter((_, index) => numeric(input[index]) && numeric(cached[index]) && cached[index]! <= input[index]!)
  const pairInput = pairs.reduce((sum, record) => sum + record.usage.prompt_tokens!, 0)
  const pairCached = pairs.reduce((sum, record) => sum + record.usage.prompt_tokens_details!.cached_tokens!, 0)
  const rate = (tokenKey: 'predicted_n' | 'prompt_n', timeKey: 'predicted_ms' | 'prompt_ms') => {
    const known = records.filter(record => numeric(record.timings[tokenKey]) && numeric(record.timings[timeKey]) && record.timings[timeKey]! > 0)
    const tokens = known.reduce((sum, record) => sum + record.timings[tokenKey]!, 0), seconds = known.reduce((sum, record) => sum + record.timings[timeKey]! / 1000, 0)
    return { value: seconds ? tokens / seconds : null, tokens, seconds, reported: known.length }
  }
  const first = records.map(record => record.first_output_s).filter(numeric)
  const status_counts = { running: 0, completed: 0, failed: 0, cancelled: 0, incomplete: 0 }
  records.forEach(record => status_counts[record.status]++)
  return { request_count: records.length, input: totals(input), output: totals(output), cached: totals(cached),
    total: totals(records.map((_, index) => numeric(input[index]) && numeric(output[index]) ? input[index]! + output[index]! : undefined)),
    cache_pair: { input: pairs.length ? pairInput : null, cached: pairs.length ? pairCached : null, reported: pairs.length }, cache_ratio: pairInput ? pairCached / pairInput : null,
    decode: rate('predicted_n', 'predicted_ms'), prefill: rate('prompt_n', 'prompt_ms'),
    ttft: { value: first.length ? first.reduce((sum, value) => sum + value, 0) / first.length : null, reported: first.length, engine_reported: 0, gateway_reported: first.length },
    status_counts, record_incomplete_count: records.filter(record => record.record_incomplete).length }
}

export const fixtureGroupKey = (record: ExternalRequestRecord) => record.aggregation === 'turn' && record.turn_id
  ? JSON.stringify(['external', record.run_id, record.instance_id, record.turn_id])
  : JSON.stringify([record.run_id, record.instance_id, record.request_key])

export function historyPage(records: readonly ExternalRequestRecord[], query: URLSearchParams, retained: readonly ExternalRequestRecord[] = records): RequestHistoryPage {
  const offset = Number(query.get('offset') || 0), limit = Number(query.get('limit') || records.length || 50)
  const sorted = [...records].sort((a, b) => query.has('turn_id') ? a.started_at! - b.started_at! : b.started_at! - a.started_at!)
  const common = { run_id: query.get('run_id'), total: records.length, summary: historyStats(records), offset, limit, issues: [] }
  if (query.get('group_by') !== 'turn') return { ...common, items: sorted.slice(offset, offset + limit) }
  const perRequest = query.get('external_group_by') === 'request'
  const map = new Map<string, ExternalRequestRecord[]>()
  const keyOf = (record: ExternalRequestRecord) => perRequest ? JSON.stringify([record.run_id, record.instance_id, record.request_key]) : fixtureGroupKey(record)
  sorted.forEach(record => { const key = keyOf(record); map.set(key, [...(map.get(key) ?? []), record]) })
  const all = [...map.values()].sort((a, b) => Math.min(...b.map(record => record.started_at ?? 0)) - Math.min(...a.map(record => record.started_at ?? 0)))
  const groups = all.map<RequestHistoryGroup>(matched => {
    const members = retained.filter(record => keyOf(record) === keyOf(matched[0]))
    const record = matched[0], started = Math.min(...members.map(member => member.started_at ?? 0))
    const running = members.some(member => member.status === 'running'), ended = running ? null : Math.max(...members.map(member => member.ended_at ?? (member.started_at ?? 0) + member.duration_s))
    return { ...historyStats(members), run_id: record.run_id, instance_id: record.instance_id, turn_id: record.turn_id,
      matched_request_count: matched.length,
      request_key: record.request_key, aggregation: record.aggregation, started_at: started, ended_at: ended,
      elapsed_s: (ended ?? Date.now() / 1000) - started, status: perRequest ? record.status : running ? 'running' : record.record_incomplete ? 'incomplete' : record.turn_state ?? record.status }
  })
  return { ...common, ...(perRequest ? { external_group_by: 'request' as const } : {}), items: all.slice(offset, offset + limit).map(members => members[0]), groups: groups.slice(offset, offset + limit), internal_groups: [], group_total: all.length }
}

export function historyParent(table: Locator, record: ExternalRequestRecord) {
  return table.locator(`tr[data-group-key=${JSON.stringify(fixtureGroupKey(record))}]`)
}

export async function cycleWindow(monitor: Locator, seconds: 10 | 30 | 60) {
  const button = monitor.locator('button.monitor-window-cycle')
  await expect(button).toBeVisible()
  for (let step = 0; step < 3 && await button.textContent() !== `${seconds}s`; step++) await button.click()
  await expect(button).toHaveText(`${seconds}s`)
}

export async function cyclePhase(monitor: Locator, phase: 'Decode' | 'Prefill') {
  const button = monitor.locator('button.monitor-phase-toggle')
  await expect(button).toHaveCount(1)
  if (await button.textContent() !== phase) await button.click()
  await expect(button).toHaveText(phase)
}
