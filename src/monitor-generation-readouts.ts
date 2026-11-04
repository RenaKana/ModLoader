import { engineRateTotal, mergeRequestReport, requestFirstOutput, requestRate, requestTokens, turnTotals } from './chat-turn-metrics'
import type { GenerationAccounting, RequestHistoryStats } from './types'
import type { MonitorRequestView } from './monitor-generation'

export function accountingMatches(request: MonitorRequestView, accounting: GenerationAccounting) {
  return request.status === 'running' && (accounting.request.request_key === (request.requestKey ?? request.id)
    || accounting.turn.source === 'internal' && accounting.request.client_call_id === request.id)
}

/** Full native snapshots replace a request's contribution; they are never deltas. */
export function projectGenerationRequest(request: MonitorRequestView, accounting: GenerationAccounting | null): MonitorRequestView {
  if (!accounting || !accountingMatches(request, accounting)) return request
  const native = mergeRequestReport(request, accounting.request)
  return { ...request, ...native, decodeMean: requestRate(native, 'decode').value ?? null, prefillMean: requestRate(native, 'prefill').value ?? null }
}

function requestStats(requests: readonly MonitorRequestView[]): RequestHistoryStats {
  const totals = turnTotals(requests, 0, 0)
  const pairs = requests.map(requestTokens).filter(value => value.input !== undefined && value.cached !== undefined && value.cached <= value.input)
  const input = pairs.reduce((sum, value) => sum + value.input!, 0), cached = pairs.reduce((sum, value) => sum + value.cached!, 0)
  const both = requests.map(requestTokens).filter(value => value.input !== undefined && value.output !== undefined)
  const ttft = requests.map(requestFirstOutput).filter(value => value.value !== undefined)
  return { request_count: requests.length, input: totals.input, output: totals.output, cached: totals.cached,
    total: { value: both.length ? both.reduce((sum, value) => sum + value.input! + value.output!, 0) : null, reported: both.length },
    cache_pair: { input: pairs.length ? input : null, cached: pairs.length ? cached : null, reported: pairs.length }, cache_ratio: input > 0 ? cached / input : null,
    decode: totals.decode, prefill: totals.prefill, engine: engineRateTotal(requests),
    ttft: { value: ttft.length ? ttft.reduce((sum, value) => sum + value.value!, 0) / ttft.length : null, reported: ttft.length,
      engine_reported: ttft.filter(value => value.basis === '引擎').length, gateway_reported: ttft.filter(value => value.basis === '网关').length,
      client_reported: ttft.filter(value => value.basis === '客户端').length },
    status_counts: Object.fromEntries(['running', 'completed', 'failed', 'cancelled', 'incomplete'].map(status => [status, requests.filter(request => request.status === status).length])) as RequestHistoryStats['status_counts'],
    record_incomplete_count: requests.filter(request => request.external?.record_incomplete).length }
}

export function projectGenerationTotals(requests: readonly MonitorRequestView[], accounting: GenerationAccounting | null): RequestHistoryStats {
  return requestStats(requests.map(request => projectGenerationRequest(request, accounting)))
}
