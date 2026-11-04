import type { ExternalRequestRecord, RequestHistoryStats, RequestHistoryGroup } from './types'
import { externalRequestIdentity, type MonitorRequestView } from './monitor-generation'

export const externalTurnIdentity = (record: Pick<ExternalRequestRecord, 'run_id' | 'instance_id' | 'turn_id'>) =>
  JSON.stringify(['external', record.run_id, record.instance_id, record.turn_id])

export function requestErrorSummary(request: MonitorRequestView) {
  if (request.status !== 'failed' || !request.external?.upstream) return ''
  const upstream = request.external.upstream
  const detail = (upstream.reason?.trim() || String(upstream.code ?? '').trim()).replace(/\s+/g, ' ')
  const status = typeof upstream.status === 'number' && Number.isFinite(upstream.status) && upstream.status >= 400 ? String(upstream.status) : ''
  return [detail, status && !new RegExp(`\\b${status}\\b`).test(detail) ? `HTTP ${status}` : ''].filter(Boolean).join(' · ')
}

/** External rows always represent one HTTP request, including old turn-grouped archives. */
export function historyRequestGroups(records: readonly ExternalRequestRecord[], summaries: readonly RequestHistoryGroup[] = []) {
  return records.map(record => ({ key: externalRequestIdentity(record), turn: false, records: [record],
    summary: summaries.find(summary => summary.request_count === 1 && externalRequestIdentity(summary) === externalRequestIdentity(record)) }))
}

export type MonitorRequestTableEntry = {
  key: string; turn: boolean; requests: MonitorRequestView[]; representative?: MonitorRequestView
  alias: string; backend: string; startedAt?: number; status?: string; count?: number
  summary?: RequestHistoryStats; elapsed?: number; matchedRequestCount?: number
  expanded?: boolean; loading?: boolean; error?: string
}

