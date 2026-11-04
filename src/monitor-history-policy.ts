import type { HistoryInstance } from './monitor-history-dates'
import { localDate } from './monitor-history-dates'
import type { MonitorGenerationView, MonitorRequestView } from './monitor-generation'

export function currentHistoryInRange(startedAt: number, from: string, to: string, today: string) {
  const date = startedAt ? localDate(new Date(startedAt)) : today
  return from <= date && date <= to
}
export function historyDirectoryIsLive(instance: HistoryInstance | null, currentRun: string | undefined, from: string, to: string, today: string) {
  if (instance?.source !== 'internal' && instance?.run_id) return instance.run_id === currentRun
  return from <= today && today <= to
}
export function canFollowHistoryRequest(request: MonitorRequestView, source: MonitorGenerationView, live: MonitorGenerationView, currentRun: string | undefined) {
  if (request.status !== 'running') return false
  if (request.external) return request.external.run_id === currentRun && live.origin === 'external' && !live.historical && live.working
    && live.requests.some(current => current.id === request.id && current.status === 'running')
  return live.origin !== 'external' && !live.historical && live.working && source.sessionKey === live.sessionKey
    && live.requests.some(current => current.id === request.id && current.status === 'running')
}

import type { RequestHistoryGroup, RequestHistoryPage } from './types'

export function historyParentTotals(directory: RequestHistoryGroup | undefined, detail: RequestHistoryPage | undefined, live?: MonitorGenerationView) {
  return { count: live ? live.requests.length : detail?.total ?? directory?.request_count, summary: live ? undefined : detail ? detail.summary : directory }
}

import type { Telemetry } from './types'
import { matchingRequestLive } from './monitor-generation'

export type PendingHistoryFollow = { instanceId: string; request: MonitorRequestView; messageId?: string }
export function confirmHistoryFollow(pending: PendingHistoryFollow, observed: {
  instanceId: string; ready: boolean; payload: Telemetry | null; error: string; live: MonitorGenerationView; currentRun?: string; settled?: boolean
}): 'wait' | 'follow' | 'pin' {
  if (observed.instanceId !== pending.instanceId || !observed.ready || observed.error) return 'pin'
  if (!observed.payload || observed.payload.instance_id !== pending.instanceId) return observed.settled ? 'pin' : 'wait'
  if (pending.request.external) return canFollowHistoryRequest(pending.request, observed.live, observed.live, observed.currentRun) ? 'follow' : 'pin'
  const sample = observed.payload.live_generation
  const matchingActivity = observed.live.origin === 'internal' && !observed.live.historical && observed.live.working
    && (!pending.messageId || observed.live.sessionKey.endsWith(`:${pending.messageId}`))
    && observed.live.requests.some(request => request.id === pending.request.id && request.status === 'running')
  const matchingSample = sample?.scope === 'request' && sample.phase !== 'idle' && sample.backend_running !== 0
    && !!matchingRequestLive(sample, pending.request) && (!pending.messageId || sample.client_request_id === pending.messageId)
  return matchingActivity || matchingSample ? 'follow' : 'pin'
}
