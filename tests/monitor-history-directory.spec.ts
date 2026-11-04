import { test, expect } from '@playwright/test'
import { historyInstanceIdentity, historyInstances, historyInstanceMatches } from '../src/monitor-history-dates'
import { historyRequestGroups } from '../src/monitor-request-groups'
import type { RequestHistoryDates, RequestHistoryGroup, ExternalRequestRecord } from '../src/types'

test('internal history instances have a source namespace and work without a run or instance id', () => {
  const internal = { source: 'internal' as const, id: null, conversation_id: 'conversation', alias: 'Model', backend: '' }
  const external = { run_id: 'run', id: 'model', alias: 'Model', backend: '' }
  expect(historyInstanceIdentity(external)).toBe(JSON.stringify(['run', 'model']))
  expect(historyInstanceIdentity(internal)).not.toBe(historyInstanceIdentity({ ...internal, conversation_id: 'other' }))
  const catalog: RequestHistoryDates = { current_run_id: 'run', today: '2026-10-31', issues: [], dates: [{ date: '2026-10-31', request_count: 2, instances: [internal, external] }] }
  expect(historyInstances(catalog, '2026-10-31', '2026-10-31').map(item => item.label)).toEqual([expect.stringContaining('内置'), expect.stringContaining('外部')])
})

test('external request directory never adopts the old turn total from its representative', () => {
  const record = { run_id: 'r', instance_id: 'i', request_key: 'q', aggregation: 'turn', turn_id: 't' } as ExternalRequestRecord
  const summary = { ...record, request_count: 9, started_at: 1, elapsed_s: 20 } as RequestHistoryGroup
  const groups = historyRequestGroups([record], [summary])
  expect(groups[0].summary).toBeUndefined()
  expect(groups[0].records).toHaveLength(1)
})

test('live history overlay respects run and source filters for reused instance ids', () => {
  const selected = { run_id: 'old', id: 'model', alias: 'Model', backend: 'llama_cpp' }
  expect(historyInstanceMatches(selected, { ...selected, run_id: 'current' })).toBe(false)
  expect(historyInstanceMatches(selected, { ...selected, source: 'internal' })).toBe(false)
  expect(historyInstanceMatches(selected, selected)).toBe(true)
  expect(historyInstanceMatches(null, undefined)).toBe(true)
})

import { currentHistoryInRange, historyDirectoryIsLive, canFollowHistoryRequest } from '../src/monitor-history-policy'
import { externalGenerationView, type MonitorGenerationView, type MonitorRequestView } from '../src/monitor-generation'

test('untimed current backend requests belong to today rather than 1970', () => {
  expect(currentHistoryInRange(0, '2026-10-31', '2026-10-31', '2026-10-31')).toBe(true)
  expect(currentHistoryInRange(0, '2026-10-30', '2026-10-30', '2026-10-31')).toBe(false)
})

test('archived external run directory is static while current and internal today stay live', () => {
  const old = { id: 'same-id', run_id: 'old', alias: '', backend: '' }
  const check = (instance: typeof old | null) => historyDirectoryIsLive(instance, 'current', '2026-10-31', '2026-10-31', '2026-10-31')
  expect(check(old)).toBe(false)
  expect(check({ ...old, run_id: 'current' })).toBe(true)
  expect(check(null)).toBe(true)
  expect(historyDirectoryIsLive({ source: 'internal', id: 'same-id', alias: '', backend: '' }, 'current', '2026-10-31', '2026-10-31', '2026-10-31')).toBe(true)
})

test('stored unfinished internal requests stay pinned unless they are the actual nonhistorical live request', () => {
  const empty = externalGenerationView({ run_id: '', aggregation: 'request', turn_id: null, state: 'completed', requests: [] })
  const request = { id: 'current-call', status: 'running' } as MonitorRequestView
  const live: MonitorGenerationView = { ...empty, origin: 'internal', historical: false, working: true, sessionKey: 'instance:current-message', requests: [request] }
  const stored = { ...live, historical: true, sessionKey: 'instance:old-message' }
  expect(canFollowHistoryRequest(request, stored, live, 'current')).toBe(false)
  expect(canFollowHistoryRequest({ ...request, id: 'old-call' }, live, live, 'current')).toBe(false)
  expect(canFollowHistoryRequest(request, live, { ...live, historical: true }, 'current')).toBe(false)
  expect(canFollowHistoryRequest(request, live, { ...live, working: false }, 'current')).toBe(false)
  expect(canFollowHistoryRequest(request, live, live, 'current')).toBe(true)
})

import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { MonitorRequestTable } from '../src/MonitorRequestTable'
import { historyParentTotals } from '../src/monitor-history-policy'
import type { RequestHistoryPage } from '../src/types'

test('cross-midnight legacy turns display each retained request independently without changing the date summary', () => {
  const record = (key: string, started: number, input: number, output: number): ExternalRequestRecord => ({
    run_id: 'old', instance_id: 'model', request_key: key, alias: '跨午夜轮', backend: 'llama_cpp', aggregation: 'turn', turn_id: 'midnight', turn_state: 'completed',
    revision: 1, labels: [], status: 'completed', started_at: started, ended_at: started + 1, duration_s: 1,
    usage: { prompt_tokens: input, completion_tokens: output }, timings: { predicted_n: output, predicted_ms: 1000 },
  })
  const before = record('before-midnight', 100, 101, 11), after = record('after-midnight', 102, 202, 22)
  const directory = { request_count: 1, input: { value: 202, reported: 1 }, decode: { value: 22 } } as RequestHistoryGroup
  const detail: RequestHistoryPage = { run_id: 'old', items: [before, after], total: 2, offset: 0, limit: 2, issues: [] }
  const views = detail.items.map(record => externalGenerationView({ run_id: 'old', aggregation: 'turn', turn_id: 'midnight', state: 'completed', requests: [record] }))
  const metrics = historyParentTotals(directory, detail)
  expect(metrics.count).toBe(2)
  expect(metrics.summary).toBeUndefined()
  const html = renderToStaticMarkup(createElement(MonitorRequestTable, {
    entries: views.map((view, index) => ({ key: `midnight-${index}`, turn: false, alias: '跨午夜轮', backend: 'llama_cpp', requests: view.requests, count: 1 })),
    onSelectGroup() {}, onSelectRequest() {}, onToggleGroup() {},
  }))
  expect(html).toContain('data-request-key="before-midnight"')
  expect(html).toContain('data-request-key="after-midnight"')
  expect(html).toContain('aria-label="输入（含缓存） 101 Token"')
  expect(html).toContain('aria-label="输入（含缓存） 202 Token"')
  expect(html).toContain('aria-label="Decode 11.00 token/s"')
  expect(html).toContain('aria-label="Decode 22.00 token/s"')
  expect(directory.request_count).toBe(1)
  expect(directory.input.value).toBe(202)
  const fullSummary = { ...directory, request_count: 2, input: { value: 303, reported: 2 } }
  expect(historyParentTotals(directory, { ...detail, summary: fullSummary }).summary).toBe(fullSummary)
  expect(views.every(view => !view.turn)).toBe(true)
})

test('current-run stale running external records cannot resume follow for a different completed or active request', () => {
  const record = (key: string, instance = 'model'): ExternalRequestRecord => ({
    run_id: 'current', instance_id: instance, request_key: key, alias: 'Model', backend: 'llama_cpp', aggregation: 'request', turn_id: null, turn_state: null,
    revision: 1, labels: [], status: 'running', started_at: 100, duration_s: 1, usage: {}, timings: {},
  })
  const view = (item: ExternalRequestRecord) => externalGenerationView({ run_id: 'current', aggregation: 'request', turn_id: null, state: item.status, requests: [item] })
  const stale = view(record('upgrade')), finished = view({ ...record('new-current'), status: 'completed' }), active = view(record('new-current'))
  expect(canFollowHistoryRequest(stale.requests[0], stale, finished, 'current')).toBe(false)
  expect(canFollowHistoryRequest(stale.requests[0], stale, active, 'current')).toBe(false)
  const otherInstance = view(record('new-current', 'other-instance'))
  expect(canFollowHistoryRequest(otherInstance.requests[0], otherInstance, active, 'current')).toBe(false)
  expect(canFollowHistoryRequest(active.requests[0], active, { ...active, historical: true }, 'current')).toBe(false)
  expect(canFollowHistoryRequest(active.requests[0], active, active, 'current')).toBe(true)
})

import { confirmHistoryFollow } from '../src/monitor-history-policy'
import type { Telemetry } from '../src/types'

test('cross-instance handoff waits for one existing telemetry result and only follows its matching active request', () => {
  const record = { run_id: 'current', instance_id: 'target', request_key: 'picked', status: 'running', aggregation: 'request', turn_id: null, turn_state: null, labels: [], usage: {}, timings: {}, duration_s: 1 } as ExternalRequestRecord
  const view = externalGenerationView({ run_id: 'current', aggregation: 'request', turn_id: null, state: 'running', requests: [record] })
  const pending = { instanceId: 'target', request: view.requests[0] }
  const payload = { instance_id: 'target', state: 'ready', live_generation: { scope: 'request', request_key: 'picked', phase: 'output', backend_running: 1 } } as Telemetry
  const observed = { instanceId: 'target', ready: true, payload, error: '', live: view, currentRun: 'current' }
  expect(confirmHistoryFollow(pending, { ...observed, payload: null })).toBe('wait')
  expect(confirmHistoryFollow(pending, { ...observed, payload: null, settled: true })).toBe('pin')
  expect(confirmHistoryFollow(pending, observed)).toBe('follow')
  expect(confirmHistoryFollow(pending, { ...observed, error: '读取失败' })).toBe('pin')
  expect(confirmHistoryFollow(pending, { ...observed, instanceId: 'user-picked-other' })).toBe('pin')
  expect(confirmHistoryFollow(pending, { ...observed, live: { ...view, working: false } })).toBe('pin')
  expect(confirmHistoryFollow(pending, { ...observed, live: { ...view, requests: [{ ...view.requests[0], id: 'other-request' }] } })).toBe('pin')
  expect(confirmHistoryFollow(pending, { ...observed, payload: { ...payload, instance_id: 'previous' } })).toBe('wait')
})

test('cross-instance internal handoff accepts matching real request telemetry, not the stored running flag', () => {
  const empty = externalGenerationView({ run_id: '', aggregation: 'request', turn_id: null, state: 'completed', requests: [] })
  const request = { id: 'internal-call', requestKey: 'backend-key', status: 'running' } as MonitorRequestView
  const pending = { instanceId: 'target', request, messageId: 'message' }
  const payload = { instance_id: 'target', state: 'ready', live_generation: { scope: 'request', client_call_id: 'internal-call', client_request_id: 'message', request_key: 'backend-key', phase: 'output', backend_running: 1 } } as Telemetry
  const observed = { instanceId: 'target', ready: true, payload, error: '', live: empty }
  expect(confirmHistoryFollow(pending, observed)).toBe('follow')
  expect(confirmHistoryFollow(pending, { ...observed, payload: { ...payload, live_generation: { ...payload.live_generation!, client_call_id: 'different-call' } } })).toBe('pin')
  expect(confirmHistoryFollow(pending, { ...observed, payload: { ...payload, live_generation: { ...payload.live_generation!, client_request_id: 'different-message' } } })).toBe('pin')
  expect(confirmHistoryFollow(pending, { ...observed, payload: { ...payload, live_generation: { ...payload.live_generation!, phase: 'idle', backend_running: 0 } } })).toBe('pin')
})

import { monitorGenerationView } from '../src/monitor-generation'
import type { ChatGenerationActivity, RuntimeInstance } from '../src/types'

test('inspection presentation never supplies the live-follow decision for newer same-instance requests', () => {
  const instance = { id: 'model', state: 'ready', active_requests: 1 } as RuntimeInstance
  const old: ChatGenerationActivity = { instanceId: 'model', messageId: 'old-message', phase: 'completed', requests: [
    { id: 'old-call', index: 1, kind: 'generation', labels: [], status: 'completed', startedAt: 1000 },
  ] }
  const current: ChatGenerationActivity = { instanceId: 'model', messageId: 'new-message', phase: 'generating', requests: [
    { id: 'new-call', requestKey: 'new-backend-key', index: 1, kind: 'generation', labels: [], status: 'running', startedAt: 2000 },
  ] }
  const payload = { instance_id: 'model', backend: 'llama_cpp', state: 'ready', updated_at: 3, groups: [], unavailable: [],
    live_generation: { scope: 'request', request_key: 'new-backend-key', client_call_id: 'new-call', client_request_id: 'new-message', phase: 'output', backend_running: 1 } } as Telemetry
  const inspection = monitorGenerationView(payload, undefined, instance, false, [current], old)
  const internalLive = monitorGenerationView(payload, undefined, instance, false, [current])
  expect(inspection.historical).toBe(true)
  expect(canFollowHistoryRequest(internalLive.requests[0], internalLive, inspection, 'current')).toBe(false)
  expect(canFollowHistoryRequest(internalLive.requests[0], internalLive, internalLive, 'current')).toBe(true)
  const record = { run_id: 'current', instance_id: 'model', request_key: 'external-now', status: 'running', aggregation: 'request', turn_id: null, turn_state: null, labels: [], usage: {}, timings: {}, duration_s: 1 } as ExternalRequestRecord
  const externalLive = externalGenerationView({ run_id: 'current', aggregation: 'request', turn_id: null, state: 'running', requests: [record] })
  expect(canFollowHistoryRequest(externalLive.requests[0], externalLive, inspection, 'current')).toBe(false)
  expect(canFollowHistoryRequest(externalLive.requests[0], externalLive, externalLive, 'current')).toBe(true)

})


import { requireGroupedHistoryPage } from '../src/MonitorRequestHistory'

const emptyGroupedDirectory: RequestHistoryPage = {
  run_id: null, items: [], total: 0, groups: [], internal_groups: [], group_total: 0, offset: 0, limit: 50, issues: [],
  summary: { request_count: 0, input: { value: null, reported: 0 }, output: { value: null, reported: 0 }, total: { value: null, reported: 0 },
    cached: { value: null, reported: 0 }, cache_pair: { input: null, cached: null, reported: 0 }, cache_ratio: null,
    decode: { value: null, tokens: 0, seconds: 0, reported: 0 }, prefill: { value: null, tokens: 0, seconds: 0, reported: 0 },
    ttft: { value: null, reported: 0, engine_reported: 0, gateway_reported: 0 },
    status_counts: { running: 0, completed: 0, failed: 0, cancelled: 0, incomplete: 0 }, record_incomplete_count: 0 },
}

test('old request history directory schema reports the required development restart instead of accepting missing totals', () => {
  const old: RequestHistoryPage = { run_id: 'old', items: [], total: 0, offset: 0, limit: 50, issues: [] }
  expect(() => requireGroupedHistoryPage(old)).toThrow('运行监控接口尚未更新，请退出并重新打开开发版')
  for (const field of ['summary', 'groups', 'internal_groups', 'group_total'] as const) {
    const incomplete = { ...emptyGroupedDirectory }
    delete incomplete[field]
    expect(() => requireGroupedHistoryPage(incomplete), field).toThrow('运行监控接口尚未更新')
  }
})

test('valid empty grouped directory is accepted without optional coverage or fabricated zero token values', () => {
  expect(requireGroupedHistoryPage(emptyGroupedDirectory)).toBe(emptyGroupedDirectory)
  expect(emptyGroupedDirectory.group_total).toBe(0)
  expect(emptyGroupedDirectory.summary?.input.value).toBeNull()
})
