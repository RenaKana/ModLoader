import { test, expect } from '@playwright/test'
import type { ChatGenerationActivity, ExternalHistorySnapshot, ExternalRequestRecord, LiveGeneration, RequestHistoryDates, RuntimeInstance, Telemetry } from '../src/types'
import { externalGenerationView, externalRequestIdentity, generationSegments, monitorGenerationView, rememberGeneration } from '../src/monitor-generation'
import { historyDateRange, historyInstanceIdentity, historyInstances, localDate, previousDate } from '../src/monitor-history-dates'

const instance = { id: 'model', alias: 'model', backend: 'llama_cpp', state: 'ready', active_requests: 1 } as RuntimeInstance
const record = (request_key: string, extra: Partial<ExternalRequestRecord> = {}): ExternalRequestRecord => ({ request_key, run_id: 'run', instance_id: instance.id,
  alias: instance.alias, backend: instance.backend, revision: 1, aggregation: 'request', turn_id: null, turn_state: null, labels: [],
  status: 'running', started_at: 100, duration_s: 1, usage: {}, timings: {}, ...extra })
const snapshot = (requests: ExternalRequestRecord[], extra: Partial<ExternalHistorySnapshot> = {}): ExternalHistorySnapshot => ({ run_id: 'run', aggregation: 'request', turn_id: null, state: 'running', requests, ...extra })
const payload = (time: number, history: ExternalHistorySnapshot, live: Partial<LiveGeneration> = {}): Telemetry => ({ instance_id: instance.id, backend: instance.backend,
  state: 'ready', updated_at: time, groups: [], unavailable: [], external_history: history,
  live_generation: { request_key: null, scope: 'request', source: 'fixture', phase: 'idle', backend_running: 0, backend_queued: 0, tokens_per_second: null, window_s: null, available_for_chat: false, reason: null, ...live } })

test('A/B/A恢复对应请求采样并分段，新独立调用不淘汰旧组，每个请求仍只保留自身60秒', () => {
  const a = record('A'), b = record('B', { started_at: 101 })
  let memory = rememberGeneration(undefined, payload(100, snapshot([a]), { request_key: 'A', phase: 'output', tokens_per_second: 150 }))
  memory = rememberGeneration(memory, payload(101, snapshot([a]), { request_key: 'A', phase: 'output', tokens_per_second: 160 }))
  const side = payload(102, snapshot([b]), { request_key: 'B', phase: 'output', tokens_per_second: 900 })
  side.groups = [{ id: 'api_request', label: 'API请求', metrics: [], request_key: 'A', request_status: 'running' }]
  memory = rememberGeneration(memory, side)
  memory = rememberGeneration(memory, payload(103, snapshot([a]), { request_key: 'A', phase: 'output', tokens_per_second: 170 }))
  const group = snapshot([{ ...a, status: 'completed' }], { aggregation: 'turn', turn_id: 'T', state: 'waiting' })
  const view = externalGenerationView(group, memory, null, true)
  expect(view.requests[0].phases.decode.points.map(point => point.values.current)).toEqual([150, 160, 170])
  expect(generationSegments(view.requests[0].phases.decode.points).map(segment => segment.map(point => point.value))).toEqual([[150, 160], [170]])
  const frozen = view.requests[0].phases.decode.points
  memory = rememberGeneration(memory, payload(1000, snapshot([record('C')]), { request_key: 'C', phase: 'output', tokens_per_second: 20 }))
  expect(externalGenerationView(group, memory, null, true).requests[0].phases.decode.points).toBe(frozen)
  expect(memory.sessions?.map(session => session.key)).toEqual(['A', 'B', 'C'])
  memory = rememberGeneration(memory, payload(1070, snapshot([record('C')]), { request_key: 'C', phase: 'output', tokens_per_second: 30 }))
  expect(memory.sessions?.find(session => session.key === 'C')?.rates.decode.points.map(point => point.wallTime)).toEqual([1070000])
  expect(externalGenerationView(group, memory, null, true).requests[0].phases.decode.points).toBe(frozen)
})

test('工具组主T与side交替不会串计数；当前live的明确来源优先，外部终态不回退旧内置结果', () => {
  const main = record('main', { aggregation: 'turn', turn_id: 'T' })
  const history = snapshot([main], { aggregation: 'turn', turn_id: 'T' })
  const internal: ChatGenerationActivity = { instanceId: instance.id, messageId: 'internal-turn', phase: 'completed', requests: [
    { id: 'internal-call', requestKey: 'internal-key', index: 1, kind: 'generation', status: 'completed', startedAt: 200000, labels: [] },
  ] }
  const side = payload(300, history, { request_key: 'side', phase: 'output', tokens_per_second: 999, generated_tokens: 999 })
  let memory = rememberGeneration(undefined, side)
  let view = monitorGenerationView(side, memory, instance, false, [internal])
  expect(view.origin).toBe('external'); expect(view.current).toBeNull()
  expect(view.requests[0].usage).toEqual({}); expect(view.requests[0].phases.decode.points).toEqual([])
  const resumed = payload(301, history, { request_key: 'main', client_call_id: 'external-client-id', phase: 'output', tokens_per_second: 170, generated_tokens: 17 })
  memory = rememberGeneration(memory, resumed)
  view = monitorGenerationView(resumed, memory, instance, false, [internal])
  expect(view.origin).toBe('external'); expect(view.current).toBe(170)
  expect(view.requests[0].live?.generated_tokens).toBe(17)
  const internalRunning = { ...internal, phase: 'generating' as const, requests: [{ ...internal.requests![0], status: 'running' as const }] }
  const staleExternal = snapshot([record('old-external', { started_at: 250, status: 'completed' })], { state: 'completed' })
  const internalLive = payload(302, staleExternal, { request_key: 'internal-key', client_call_id: 'internal-call', client_request_id: 'internal-turn', phase: 'output', tokens_per_second: 80 })
  memory = rememberGeneration(memory, internalLive)
  expect(monitorGenerationView(internalLive, memory, instance, false, [internalRunning]).origin).toBe('internal')
  const noLive = payload(303, staleExternal)
  expect(monitorGenerationView(noLive, memory, instance, false, [internalRunning]).origin).toBe('internal')
  const completed = payload(401, snapshot([record('external-end', { started_at: 400, status: 'completed' })], { state: 'completed' }))
  memory = rememberGeneration(memory, completed)
  expect(monitorGenerationView(completed, memory, instance, false, [internal]).origin).toBe('external')
})

test('新gateway运行清除旧曲线；跨run同名request_key保持不同记录身份', () => {
  const a = record('same')
  const first = payload(100, snapshot([a]), { request_key: a.request_key, phase: 'output', tokens_per_second: 150 })
  let memory = rememberGeneration(undefined, first)
  const next = { ...a, run_id: 'next', alias: 'next alias' }
  memory = rememberGeneration(memory, payload(101, snapshot([next], { run_id: 'next' }), { request_key: next.request_key, phase: 'output', tokens_per_second: 20 }))
  expect(memory.gatewayRunId).toBe('next')
  expect(memory.sessions).toHaveLength(1)
  expect(memory.session?.rates.decode.points.map(point => point.values.current)).toEqual([20])
  const requests = [a, next].flatMap(record => externalGenerationView(snapshot([record])).requests)
  expect(requests.map(request => request.id)).toEqual([externalRequestIdentity(a), externalRequestIdentity(next)])
  expect(requests.map(request => request.external?.alias)).toEqual(['model', 'next alias'])
})

test('交叠请求以实际完成事件选最后结果，外部A先开始最后结束时不退回内置I', () => {
  const external = record('overlap-A', { status: 'completed', started_at: 100, ended_at: 300, duration_s: 200 })
  const internal: ChatGenerationActivity = { instanceId: instance.id, messageId: 'overlap-I', phase: 'completed', requests: [
    { id: 'internal-I', kind: 'generation', index: 1, status: 'completed', startedAt: 250000, endedAt: 260000, labels: [] },
  ] }
  const sample = payload(301, snapshot([external], { state: 'completed' }))
  const memory = rememberGeneration(undefined, sample)
  expect(monitorGenerationView(sample, memory, instance, false, [internal])).toMatchObject({ origin: 'external', key: 'overlap-A' })
  const laterPoll = { ...sample, updated_at: 10000 }
  expect(monitorGenerationView(laterPoll, memory, instance, false, [internal]).origin).toBe('external')
  const endedLast = { ...internal, requests: [{ ...internal.requests![0], endedAt: 400000 }] }
  expect(monitorGenerationView(sample, memory, instance, false, [endedLast]).origin).toBe('internal')
  const unknownEnd = payload(10000, snapshot([{ ...external, ended_at: undefined }], { state: 'completed' }))
  expect(monitorGenerationView(unknownEnd, rememberGeneration(memory, unknownEnd), instance, false, [internal]).origin).toBe('internal')
})

test('本地日期覆盖跨年、闰日、指定日和闭区间；日期范围不加入人为跨度上限', () => {
  expect(localDate(new Date(2026, 0, 1, 0, 1))).toBe('2026-01-01')
  expect(previousDate('2026-01-01')).toBe('2025-12-31')
  expect(previousDate('2024-03-01')).toBe('2024-02-29')
  expect(historyDateRange('today', '2026-10-30', '', '')).toEqual({ from: '2026-10-30', to: '2026-10-30', error: '' })
  expect(historyDateRange('yesterday', '2026-10-30', '', '')).toEqual({ from: '2026-10-29', to: '2026-10-29', error: '' })
  expect(historyDateRange('date', '2026-10-30', '2025-01-01', '')).toEqual({ from: '2025-01-01', to: '2025-01-01', error: '' })
  expect(historyDateRange('range', '2026-10-30', '2020-01-01', '2026-10-30').error).toBe('')
  expect(historyDateRange('range', '2026-10-30', '2026-10-30', '2026-10-29').error).toBeTruthy()
  expect(historyDateRange('date', '2026-10-30', '', '').error).toBeTruthy()
})

test('实例目录只来自日期范围，跨run同名实例独立且可区分，同一run跨天不重复', () => {
  const current = { run_id: 'run', id: 'same-id', alias: 'same alias', backend: 'llama_cpp', started_at: 200 }
  const old = { ...current, run_id: 'old', started_at: 100 }
  const catalog: RequestHistoryDates = { current_run_id: 'run', today: '2026-10-30', issues: [], dates: [
    { date: '2026-10-30', request_count: 2, instances: [current, old] },
    { date: '2026-10-29', request_count: 1, instances: [current] },
    { date: '2026-10-28', request_count: 1, instances: [{ ...old, id: 'removed' }] },
  ] }
  const options = historyInstances(catalog, '2026-10-29', '2026-10-30')
  expect(options.map(option => option.value)).toEqual([historyInstanceIdentity(current), historyInstanceIdentity(old)])
  expect(new Set(options.map(option => option.label)).size).toBe(2)
  expect(historyInstances(catalog, '2026-10-29', '2026-10-29').map(option => option.run_id)).toEqual(['run'])
  expect(historyInstances(catalog, '2026-10-27', '2026-10-27')).toEqual([])
})
