import { expect, test } from '@playwright/test'
import { createGenerationEventStore } from '../src/monitor-generation-events'
import type { GenerationAccounting, GenerationEvent, Telemetry } from '../src/types'
import type { MonitorRequestView } from '../src/monitor-generation'

const identity = { instanceId: 'instance', runId: 'run', requestKey: 'call' }
const payload = (key = 'call', updated = 10, run = 'run') => ({ instance_id: 'instance', updated_at: updated,
  live_generation: { request_key: key }, external_history: { run_id: run } } as Telemetry)
const event = (speed = 100, observed = 11): GenerationEvent => ({ instance_id: 'instance', run_id: 'run', observed_at: observed,
  accounting: null, finalized: [],
  live_generation: { request_key: 'call', phase: 'output', scope: 'request', tokens_per_second: speed, window_s: 3,
    source: 'SSE', backend_running: 1, backend_queued: 0, available_for_chat: true, reason: null } })
function fixture() {
  const frames: FrameRequestCallback[] = []
  const store = createGenerationEventStore(callback => frames.push(callback))
  const frame = () => frames.splice(0).forEach(callback => callback(0))
  store.reset(identity.instanceId); store.confirm(payload()); frame()
  return { store, frame, rate: () => store.rate(store.getSnapshot(), identity, 'decode') }
}

const finalAccounting = { request: { request_key: 'call', status: 'completed', duration_s: 3, started_at: 8, streaming: true,
  usage: { completion_tokens: 150 }, timings: { predicted_n: 150, predicted_ms: 3000 }, counters: { input: null, output: 150, cached: null },
  measurements: { decode: { tokens: 150, seconds: 3, value: 50, basis: 'backend_decode', reported: 1, coverage: 'complete' },
    prefill: { tokens: null, seconds: null, value: null, basis: null, reported: 0, coverage: 'unknown' } } },
  turn: { source: 'external', key: 'request:call', turn_id: null, client_request_id: null, request_keys: ['call'], prior: null } } as GenerationAccounting
const activeRequest = { id: 'call', requestKey: 'call', status: 'running', external: { run_id: 'run', instance_id: 'instance' } } as MonitorRequestView

test('已知自己的终态在断连后仍清current，不复活旧poll；终态计量等待慢照接管', () => {
  const f = fixture()
  f.store.receive({ ...event(), live_generation: null, finalized: [finalAccounting] }); f.frame()
  expect(f.store.accounting(activeRequest)).toBe(finalAccounting)
  f.store.clear(); f.frame()
  expect(f.rate()).toBeNull()
  expect(f.store.accounting(activeRequest)).toBe(finalAccounting)
})

test('慢照同key真实终态接管后释放快终态缓存，不在run内永久累积', () => {
  const f = fixture()
  f.store.receive({ ...event(), live_generation: null, finalized: [finalAccounting] }); f.frame()
  const slow = payload()
  slow.external_history!.requests = [{ ...finalAccounting.request, run_id: 'run', instance_id: 'instance' } as any]
  f.store.confirm(slow); f.frame()
  expect(f.store.accounting(activeRequest)).toBeNull()
})

test('同一时刻有序终态不能被误当旧快帧丢弃', () => {
  const f = fixture()
  f.store.receive(event(200, 11)); f.frame(); expect(f.rate()).toBe(200)
  f.store.receive({ ...event(200, 11), live_generation: null }); f.frame()
  expect(f.rate()).toBeNull()
})

test('真实事件只按绘制帧合并；首poll确认身份前不用，确认后立即可读最新值', () => {
  const f = fixture(); let notified = 0
  f.store.subscribe(() => notified++)
  f.store.receive(event(100)); f.store.receive(event(200, 12))
  expect(f.rate()).toBeUndefined(); expect(notified).toBe(0)
  f.frame(); expect(f.rate()).toBe(200); expect(notified).toBe(1)
  f.store.reset('instance'); f.frame()
  f.store.receive(event(300, 13)); f.frame(); expect(f.rate()).toBeUndefined()
  f.store.confirm(payload()); expect(f.rate()).toBe(300)
})

test('旧事件、wrong run/instance/request不覆盖已配对数值，旧快值也不覆盖较新的poll', () => {
  const f = fixture(); f.store.receive(event(100)); f.frame()
  for (const invalid of [event(999, 9), { ...event(999, 12), run_id: 'wrong' }, { ...event(999, 12), instance_id: 'wrong' },
    { ...event(999, 12), live_generation: { ...event().live_generation!, request_key: 'wrong' } }]) {
    f.store.receive(invalid); f.frame(); expect(f.rate()).toBe(100)
  }
  f.store.confirm(payload('call', 12)); expect(f.rate()).toBeUndefined()
})

test('waiting/终态null/并发清掉快读数，断连退回正常采样，新run不沿用旧事件', () => {
  const f = fixture()
  f.store.receive({ ...event(), live_generation: { ...event().live_generation!, phase: 'waiting', tokens_per_second: null } }); f.frame(); expect(f.rate()).toBeNull()
  f.store.receive(event(200, 12)); f.frame(); expect(f.rate()).toBe(200)
  f.store.receive({ ...event(200, 13), live_generation: null }); f.frame(); expect(f.rate()).toBeNull()
  f.store.receive({ ...event(300, 14), live_generation: { ...event(300, 14).live_generation!, backend_running: 2 } }); f.frame(); expect(f.rate()).toBeNull()
  f.store.clear(); f.frame(); expect(f.rate()).toBeUndefined()
  f.store.receive(event(400, 15)); f.frame(); f.store.confirm(payload('call', 16, 'next-run')); f.frame(); expect(f.rate()).toBeUndefined()
})

test('Prefill与Decode独立配对，零值保留，旧请求/已结束的身份不借当前值', () => {
  const f = fixture()
  f.store.receive({ ...event(), live_generation: { ...event().live_generation!, phase: 'prefill', prefill_tokens_per_second: 0 } }); f.frame()
  expect(f.store.rate(f.store.getSnapshot(), identity, 'prefill')).toBe(0)
  expect(f.rate()).toBeUndefined()
  expect(f.store.rate(f.store.getSnapshot(), undefined, 'prefill')).toBeUndefined()
  f.store.confirm(payload('next-call')); expect(f.rate()).toBeUndefined()
})

test('Decode事件不覆盖Prefill末次采样或均值，阶段切换仍沿用该阶段原数据', () => {
  const f = fixture(); f.store.receive(event(200)); f.frame()
  expect(f.rate()).toBe(200)
  expect(f.store.rate(f.store.getSnapshot(), identity, 'prefill')).toBeUndefined()
})

test('活跃同阶段无真实速率时沿用poll，不能把缺测当清除或把真实0丢掉', () => {
  const f = fixture()
  f.store.receive({ ...event(), live_generation: { ...event().live_generation!, tokens_per_second: null, available_for_chat: false } }); f.frame()
  expect(f.rate()).toBeUndefined()
  f.store.receive({ ...event(0, 12) }); f.frame(); expect(f.rate()).toBe(0)
  f.store.receive({ ...event(0, 13), live_generation: { ...event().live_generation!, phase: 'prefill', prefill_tokens_per_second: null, available_for_chat: false } }); f.frame()
  expect(f.store.rate(f.store.getSnapshot(), identity, 'prefill')).toBeUndefined()
})
