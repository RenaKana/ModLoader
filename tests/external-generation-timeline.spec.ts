import { expect, test } from '@playwright/test'
import { createExternalTimelineStore, projectExternalTimeline } from '../src/external-generation-timeline'
import type { ExternalRequestRecord, GenerationAccounting, GenerationEvent, LiveGeneration, Telemetry } from '../src/types'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { MonitorGeneration } from '../src/MonitorGeneration'
import { externalGenerationView } from '../src/monitor-generation'

const live = (key: string | null, phase = 'output', speed: number | null = 100): LiveGeneration => ({ request_key: key, phase,
  scope: 'request', source: 'isolated fixture', backend_running: key ? 1 : 0, backend_queued: 0, tokens_per_second: phase === 'prefill' ? null : speed,
  prefill_tokens_per_second: phase === 'prefill' ? speed : null, window_s: .2, available_for_chat: true, reason: null })
const record = (key: string, start: number, end?: number): ExternalRequestRecord => ({ request_key: key, run_id: 'run', instance_id: 'instance',
  alias: 'fixture', backend: 'strata', revision: 1, aggregation: 'request', turn_id: null, turn_state: null, labels: [],
  status: end === undefined ? 'running' : 'completed', started_at: start, ...(end === undefined ? {} : { ended_at: end }), duration_s: end === undefined ? 0 : end - start,
  usage: {}, timings: {} })
const telemetry = (at: number, requests: ExternalRequestRecord[], current = live(null, 'idle', null), run = 'run'): Telemetry => ({
  instance_id: 'instance', backend: 'strata', state: 'ready', updated_at: at, groups: [], unavailable: [], live_generation: current,
  external_history: { run_id: run, aggregation: 'request', turn_id: null, state: requests.some(request => request.status === 'running') ? 'running' : 'completed', requests } })
const accounting = (request: ExternalRequestRecord): GenerationAccounting => ({ request: request as GenerationAccounting['request'],
  turn: { source: 'external', key: request.request_key, turn_id: null, client_request_id: null, request_keys: [request.request_key], prior: null } })
const event = (at: number, request?: ExternalRequestRecord, current: LiveGeneration | null = null, finalized: ExternalRequestRecord[] = []): GenerationEvent => ({
  instance_id: 'instance', run_id: 'run', observed_at: at, live_generation: current, accounting: request ? accounting(request) : null, finalized: finalized.map(accounting) })
const fixture = () => { const store = createExternalTimelineStore(callback => { callback(0); return 0 }); store.select('instance'); return store }

test('连续多个外部请求保留真实墙钟间隔，两阶段和不同请求之间没有斜线', () => {
  const store = fixture(), a = record('a', 100), b = record('b', 104)
  store.confirm(telemetry(100, [a], live('a', 'prefill', 12000)))
  store.receive(event(100.2, a, live('a', 'prefill', 18000)))
  store.receive(event(100.4, a, live('a', 'output', 100)))
  store.receive(event(100.6, a, live('a', 'output', 120)))
  store.receive(event(101, undefined, null, [record('a', 100, 101)]))
  store.confirm(telemetry(102, [record('a', 100, 101)]))
  store.receive(event(104, b, live('b', 'output', 80)))
  store.receive(event(104.2, b, live('b', 'output', 90)))
  const plot = projectExternalTimeline(store.getSnapshot(), 105000, 10, 600)
  expect(plot.series.prefill.map(segment => segment.points.map(point => point.time))).toEqual([[100000, 100200]])
  expect(plot.series.decode.map(segment => segment.points.map(point => point.time))).toEqual([[100400, 100600], [104000, 104200]])
  expect(plot.axes.prefill.maximum).toBeGreaterThan(plot.axes.decode.maximum * 50)
  expect(plot.states.some(state => state.state === 'idle' && state.start === 102000 && state.end === 104000)).toBe(true)
})

test('poll之间完成的短调用从finalized保留：空心阶段均值位于真实结束事件而不是推造区间', () => {
  const store = fixture(), short = { ...record('short', 100.01, 100.08), usage: { prompt_tokens: 20000, completion_tokens: 4 },
    timings: { prompt_n: 100, prompt_ms: 20, predicted_n: 4, predicted_ms: 40, cache_n: 19900 } }
  store.confirm(telemetry(100, []))
  store.receive(event(100.1, undefined, null, [short, record('no-rate', 100.02, 100.09)]))
  store.confirm(telemetry(101, [record('later', 100.9, 100.95)]))
  const plot = projectExternalTimeline(store.getSnapshot(), 101000, 10, 300)
  expect(plot.averages.map(marker => [marker.requestKey, marker.phase, marker.time, marker.value, marker.seconds])).toEqual([
    ['short', 'decode', 100080, 100, .04], ['short', 'prefill', 100080, 5000, .02],
  ])
  expect(plot.requests.map(marker => marker.requestKey)).toEqual(expect.arrayContaining(['short', 'no-rate', 'later']))
  expect(plot.series.decode).toEqual([])
})

test('零/无效耗时、TTFT、含cache输入及partial计量不冒充完整阶段均值', () => {
  const store = fixture(), invalid = { ...record('invalid', 100, 101), first_output_s: .2,
    usage: { prompt_tokens: 100000, completion_tokens: 50, ttft_ms: 200 }, timings: { prompt_ms: 0, predicted_n: 50, predicted_ms: 0, predicted_per_second: 999999 } }
  const cacheOnly = { ...record('cache', 101, 102), usage: { prompt_tokens: 100000 }, timings: { prompt_ms: 20 } }
  const partial = { ...record('partial', 102, 103), measurements: { prefill: { tokens: 5, seconds: .01, value: 500, basis: 'backend_prefill', reported: 1, coverage: 'partial' },
    decode: { tokens: null, seconds: null, value: null, basis: null, reported: 0, coverage: 'unknown' } } } as ExternalRequestRecord
  store.confirm(telemetry(103, [invalid, cacheOnly, partial]))
  expect(projectExternalTimeline(store.getSnapshot(), 103000, 10, 300).averages).toEqual([])
})

test('空闲才归零；排队、等待、并发、阶段缺测及断流保留未知而不借上一次速度', () => {
  const store = fixture(), a = record('a', 100)
  store.confirm(telemetry(100, [a], live('a', 'output', 120)))
  expect(projectExternalTimeline(store.getSnapshot(), 100000, 10).readouts).toEqual({ decode: 120, prefill: null })
  store.confirm(telemetry(101, [a], { ...live('a', 'queued', 120), backend_running: 0, backend_queued: 1 }))
  expect(projectExternalTimeline(store.getSnapshot(), 101000, 10).readouts).toEqual({ decode: null, prefill: null })
  store.confirm(telemetry(102, [a], live('a', 'prefill', 9000)))
  expect(projectExternalTimeline(store.getSnapshot(), 102000, 10).readouts).toEqual({ decode: null, prefill: 9000 })
  store.confirm(telemetry(103, [a], live('a', 'prefill', null)))
  expect(projectExternalTimeline(store.getSnapshot(), 103000, 10).readouts).toEqual({ decode: null, prefill: null })
  store.receive(event(103.2, a, live('a', 'output', 100)))
  store.unavailable(103300, 'events')
  expect(projectExternalTimeline(store.getSnapshot(), 103300, 10).readouts).toEqual({ decode: null, prefill: null })
  store.receive(event(104, a, { ...live('a', 'output', 500), backend_running: 2 }))
  expect(projectExternalTimeline(store.getSnapshot(), 104000, 10).readouts.decode).toBeNull()
  store.receive(event(105))
  expect(projectExternalTimeline(store.getSnapshot(), 105000, 10).state).toBe('unknown')
  store.confirm(telemetry(106, []))
  expect(projectExternalTimeline(store.getSnapshot(), 106000, 10).readouts).toEqual({ decode: 0, prefill: 0 })
})

test('终态迟到采样不复活或增添速度点，结束/取消/失败都留终点', () => {
  const store = fixture(), a = record('a', 100)
  store.confirm(telemetry(100, [a], live('a', 'output', 100)))
  store.receive(event(101, undefined, null, [record('a', 100, 100.9), { ...record('cancel', 100, 100.8), status: 'cancelled' }, { ...record('fail', 100, 100.7), status: 'failed' }]))
  store.receive(event(102, a, live('a', 'output', 999)))
  const plot = projectExternalTimeline(store.getSnapshot(), 102000, 10)
  expect(plot.series.decode.flatMap(segment => segment.points.map(point => point.value))).toEqual([100])
  expect(plot.readouts.decode).toBeNull()
  expect(plot.requests.map(marker => marker.status)).toEqual(expect.arrayContaining(['completed', 'cancelled', 'failed']))
})

test('仅用屏幕像素判断短阶段的实心点，原有多点实线和实际时间跨度不变', () => {
  const store = fixture(), a = record('a', 100)
  store.confirm(telemetry(100, [a], live('a', 'output', 100)))
  store.receive(event(100.3, a, live('a', 'output', 120)))
  const narrow = projectExternalTimeline(store.getSnapshot(), 101000, 30, 240).series.decode[0]
  const wide = projectExternalTimeline(store.getSnapshot(), 101000, 10, 600).series.decode[0]
  expect(narrow.marker).toBe(true)
  expect(wide.marker).toBe(false)
  expect(narrow.points).toEqual(wide.points)
  expect(narrow.points[1].time - narrow.points[0].time).toBe(300)
})

test('实例及gateway run隔离；同实例切回保留，旧run事件不混入', () => {
  const store = fixture(), a = record('a', 100)
  store.confirm(telemetry(100, [a], live('a', 'output', 100)))
  store.select('other')
  store.receive(event(101, a, live('a', 'output', 999)))
  expect(store.getSnapshot().requests).toEqual([])
  store.select('instance')
  expect(store.getSnapshot().requests).toHaveLength(1)
  store.confirm(telemetry(102, [], live(null, 'idle'), 'next-run'))
  store.receive(event(103, a, live('a', 'output', 999)))
  expect(store.getSnapshot().requests).toEqual([])
  expect(store.getSnapshot().runId).toBe('next-run')
})

test('未能归属的异key快帧不覆盖已确认请求；迟到final也立即撤下该请求的live读数', () => {
  const store = fixture(), a = record('a', 100)
  store.confirm(telemetry(100, [a], live('a', 'output', 100)))
  store.receive(event(101, undefined, live('wrong', 'output', 999)))
  expect(projectExternalTimeline(store.getSnapshot(), 101000, 10).readouts.decode).toBe(100)
  store.receive(event(102, a, live('a', 'output', 120)))
  store.receive(event(101.5, undefined, null, [record('a', 100, 101.4)]))
  expect(projectExternalTimeline(store.getSnapshot(), 102000, 10).readouts.decode).toBeNull()
})

test('未知阶段的多点不因poll迟到倒退时间，窗口继续滚动且不补假0', () => {
  const store = fixture(), a = record('a', 100)
  store.confirm(telemetry(100, [a], live('a', 'output', 100)))
  store.receive(event(101, a, live('a', 'output', null)))
  store.confirm(telemetry(100.5, [a], live('a', 'output', 999)))
  const plot = projectExternalTimeline(store.getSnapshot(), 105000, 10)
  expect(plot.start).toBe(95000)
  expect(plot.end).toBe(105000)
  expect(plot.readouts.decode).toBeNull()
  expect(plot.series.decode.flatMap(segment => segment.points.map(point => point.value))).toEqual([100])
})

test('只有显式follow外部视图渲染双轴；同一历史请求仍是原来的单请求图', () => {
  const store = fixture(), a = record('a', Date.now() / 1000 - 1)
  const payload = telemetry(Date.now() / 1000, [a], live('a', 'output', 100))
  store.confirm(payload)
  const generation = externalGenerationView(payload.external_history!)
  const render = (extra: object) => renderToStaticMarkup(createElement(MonitorGeneration, { generation, scope: 'request', onSelectRequest: () => {}, ...extra }))
  expect(render({ externalTimeline: store })).toContain('data-mode="external-realtime"')
  expect(render({ externalTimeline: store })).toContain('Prefill 右轴')
  expect(render({})).toContain('data-mode="realtime"')
  expect(render({})).not.toContain('data-mode="external-realtime"')
})

test('常规实线的采样仍供nearest命中，不逐点生成DOM和Tab停靠点', () => {
  const store = fixture(), at = Date.now() / 1000 - 1, a = record('a', at)
  const payload = telemetry(at, [a], live('a', 'output', 100))
  store.confirm(payload)
  for (let index = 1; index <= 9; index++) store.receive(event(at + index / 10, a, live('a', 'output', 100 + index)))
  const generation = externalGenerationView(payload.external_history!)
  const html = renderToStaticMarkup(createElement(MonitorGeneration, { generation, scope: 'request', onSelectRequest: () => {}, externalTimeline: store }))
  expect(store.getSnapshot().requests[0].samples.decode).toHaveLength(10)
  expect(html.match(/class="monitor-timeline-hit"/g)).toHaveLength(1)
})

test('状态带有按真实区间计时的简短可聚焦提示，不增加常驻说明', () => {
  const store = fixture(), at = Date.now() / 1000 - 1, a = record('a', at)
  const payload = telemetry(at, [a], { ...live('a', 'queued', null), backend_running: 0, backend_queued: 1 })
  store.confirm(payload)
  store.confirm(telemetry(at + .5, [a], live('a', 'prefill', null)))
  const generation = externalGenerationView(payload.external_history!)
  const html = renderToStaticMarkup(createElement(MonitorGeneration, { generation, scope: 'request', onSelectRequest: () => {}, externalTimeline: store }))
  expect(html).toContain('aria-label="排队中 · 0.5 s"')
  expect(html).not.toContain('monitor-state-description')
})
