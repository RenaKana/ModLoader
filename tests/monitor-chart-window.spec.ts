import { expect, test } from '@playwright/test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { MonitorGeneration } from '../src/MonitorGeneration'
import { externalGenerationView, rememberChatStream, rememberGeneration, type GenerationMemory, type GenerationPoint } from '../src/monitor-generation'
import type { ExternalHistorySnapshot, ExternalRequestRecord, Telemetry } from '../src/types'
import { retainChartWindow, windowSegments } from '../src/monitor-chart-window'

const record: ExternalRequestRecord = { run_id: 'window-run', instance_id: 'window-instance', alias: 'window-fixture', backend: 'llama_cpp',
  request_key: 'window-request', revision: 1, aggregation: 'request', turn_id: null, turn_state: null, labels: [], status: 'running', started_at: 0,
  duration_s: 100.25, usage: {}, timings: {} }
const snapshot: ExternalHistorySnapshot = { run_id: record.run_id, aggregation: 'request', turn_id: null, state: 'running', requests: [record] }
const point = (time: number, current: number | null = 150, segment = 'request:decode'): GenerationPoint => ({ time, wallTime: time, values: { current }, segment })
const times = [...Array.from({ length: 101 }, (_, index) => index * 1000), 100250]

for (const seconds of [10, 30, 60]) test(`${seconds}秒窗口使用真实连续段裁边，边界值不写回采样`, () => {
  const raw = times.map(time => point(time, time / 1000))
  const saved = structuredClone(raw)
  const retained = retainChartWindow(raw, 40250)
  const segments = windowSegments([...(retained.boundaryPoint ? [retained.boundaryPoint] : []), ...retained.points], 100250 - seconds * 1000, 100250)
  expect(segments).toHaveLength(1)
  expect(segments[0][0]).toEqual({ time: 100250 - seconds * 1000, value: 100.25 - seconds })
  expect(segments[0].at(-1)).toEqual({ time: 100250, value: 100.25 })
  expect(retained.points.every(item => item.time >= 40250)).toBe(true)
  expect(retained.boundaryPoint).toBe(raw[40])
  expect(raw).toEqual(saved)
})

test('窗口裁边不跨null、采样间断或请求分段，不将短请求拉满横轴', () => {
  for (const points of [
    [point(9000), point(9900, null), point(10500)],
    [point(7000), point(10500)],
    [point(9500, 150, 'previous'), point(10500, 150, 'resumed')],
    [point(10500)],
  ]) expect(windowSegments(points, 10000, 20000)).toEqual([[{ time: 10500, value: 150 }]])
  expect(windowSegments([point(9000), point(10500, null)], 10000, 20000)).toEqual([])
})

test('恰好边界不重复；跨越两边界的同一线段只裁几何交点', () => {
  expect(windowSegments([point(1000, 100), point(2000, 200)], 1000, 2000)).toEqual([[{ time: 1000, value: 100 }, { time: 2000, value: 200 }]])
  expect(windowSegments([point(1000, 100), point(2000, 200)], 1250, 1750)).toEqual([[{ time: 1250, value: 125 }, { time: 1750, value: 175 }]])
  expect(retainChartWindow([point(1000), point(2000)], 3000)).toMatchObject({ points: [], boundaryPoint: { time: 2000 } })
})

test('连续采样跨过实时窗口时曲线抵达横轴左右边界，而不是窗口内第一点或内缩留白', () => {
  const view = externalGenerationView(snapshot)
  view.requests[0].phases.decode.points = times.map(time => point(time))
  view.requests[0].phases.decode.current = 150
  const html = renderToStaticMarkup(createElement(MonitorGeneration, { generation: view, scope: 'request', onSelectRequest: () => {} }))
  const trace = html.match(/class="trend-line" d="([^"]+)"/)?.[1]
  expect(trace).toBeDefined()
  expect(trace).toMatch(/^M0\.00,/)
  expect(trace).toMatch(/L600\.00,[\d.]+$/)
})

test('遥测与SSE保留60秒真实点，另存一个真实边界锚点而不扩充可见样本窗口', () => {
  let telemetryMemory: GenerationMemory | undefined, streamMemory: GenerationMemory | undefined
  for (const time of times) {
    const payload: Telemetry = { instance_id: record.instance_id, backend: record.backend, state: 'ready', updated_at: time / 1000,
      groups: [], unavailable: [], external_history: snapshot, live_generation: { request_key: record.request_key, phase: 'output', scope: 'request',
        source: 'fixture', backend_running: 1, backend_queued: 0, tokens_per_second: 150, window_s: 1, available_for_chat: true, reason: null } }
    telemetryMemory = rememberGeneration(telemetryMemory, payload)
    streamMemory = rememberChatStream(streamMemory, { instanceId: record.instance_id, messageId: 'message', phase: 'generating',
      requests: [{ id: 'call', index: 1, kind: 'generation', status: 'running', startedAt: 0, labels: [] }],
      stream: { requestId: 'call', sampledAt: time, tokensPerSecond: 150, windowSeconds: 1 } })
  }
  const phase = externalGenerationView(snapshot, telemetryMemory).requests[0].phases.decode
  expect(phase.points.every(item => item.time >= 40250)).toBe(true)
  expect(phase).toHaveProperty('boundaryPoint.time', 40000)
  expect(streamMemory!.streamDecode![0].points.every(item => item.time >= 40250)).toBe(true)
  expect(streamMemory!.streamDecode![0]).toHaveProperty('boundaryPoint.time', 40000)
})
