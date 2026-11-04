import { expect, test } from '@playwright/test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { MonitorGeneration } from '../src/MonitorGeneration'
import { MonitorRequestTable } from '../src/MonitorRequestTable'
import { externalGenerationView, monitorGenerationView, rememberGeneration, type MonitorGenerationView } from '../src/monitor-generation'
import { requestTokens, turnTotals, type ModelRequestRecord } from '../src/chat-turn-metrics'
import type { ExternalHistorySnapshot, ExternalRequestRecord, Telemetry } from '../src/types'

const record = (extra: Partial<ExternalRequestRecord> = {}): ExternalRequestRecord => ({ request_key: 'first', run_id: 'run', instance_id: 'instance', alias: 'fixture', backend: 'llama_cpp',
  revision: 1, aggregation: 'request', turn_id: null, turn_state: null, labels: [], status: 'completed', started_at: 100, ended_at: 104, duration_s: 4,
  usage: { prompt_tokens: 48267, prompt_tokens_details: { cached_tokens: 45738 }, completion_tokens: 13 }, timings: {}, ...extra })
const snapshot = (requests: ExternalRequestRecord[], extra: Partial<ExternalHistorySnapshot> = {}): ExternalHistorySnapshot => ({ run_id: 'run', aggregation: 'request', turn_id: null, state: 'completed', requests, ...extra })
const render = (generation: MonitorGenerationView, scope: 'turn' | 'request' = 'request') => renderToStaticMarkup(createElement(MonitorGeneration, { generation, scope, onSelectRequest: () => {} }))
const renderTable = (generation: MonitorGenerationView) => renderToStaticMarkup(createElement(MonitorRequestTable, {
  entries: [{ key: generation.sessionKey, turn: generation.aggregation === 'turn', requests: generation.requests, count: generation.requests.length, expanded: true,
    alias: 'fixture', backend: generation.requests[0]?.external?.backend ?? 'llama_cpp' }],
  onSelectGroup: () => {}, onToggleGroup: () => {}, onSelectRequest: () => {},
}))

test('单请求轮摘要保留无权重的真实后端均值，不伪造计数或时长', () => {
  const view = externalGenerationView(snapshot([record({ usage: {}, timings: { predicted_per_second: 141 } })]))
  expect(view.requests[0].decodeMean).toBe(141)
  expect(render(view, 'turn')).toContain('<strong>141.00</strong>')
  expect(view.requests[0].timings).toEqual({ predicted_per_second: 141 })
})

test('首次外部request有真实采样时立即显示自身图表，不借第二次请求或生成轮次', () => {
  const first = record({ status: 'running', ended_at: undefined })
  const history = snapshot([first], { state: 'running' })
  const telemetry: Telemetry = { instance_id: 'instance', backend: 'llama_cpp', state: 'ready', updated_at: 101, external_history: history, groups: [], unavailable: [],
    live_generation: { request_key: 'first', phase: 'output', scope: 'request', source: 'fixture', backend_running: 1, backend_queued: 0,
      tokens_per_second: 150, available_for_chat: true, window_s: 1, reason: null } }
  const view = externalGenerationView(history, rememberGeneration(undefined, telemetry), telemetry)
  const html = render(view)
  expect(html).toContain('class="monitor-generation-chart"')
  expect(html).toContain('data-mode="realtime"')
  expect(html).toContain('class="trend-dot"')
  expect(html).not.toContain('本轮')
  expect(html).not.toContain('aria-label="图表范围"')
  expect(view.aggregation).toBe('request')
  expect(view.turn).toBeUndefined()
})

test('外部输入显示含缓存总量，未缓存计算与三个用户样例原始计量仍保留', () => {
  for (const [input, cached, expected] of [[48267, 45738, '2,529'], [52630, 51322, '1,308'], [55350, 55279, '71']] as const) {
    const item = record({ usage: { prompt_tokens: input, prompt_tokens_details: { cached_tokens: cached }, completion_tokens: 13 } })
    const view = externalGenerationView(snapshot([item]))
    const html = renderTable(view)
    expect(html).toContain(`aria-label="输入（含缓存） ${input.toLocaleString('en-US')} Token`)
    expect(requestTokens(view.requests[0]).uncached?.toLocaleString('en-US')).toBe(expected)
    expect(render(view)).toContain(`<dt>请求长度</dt><dd><span class="monitor-number"><strong>${(input + 13).toLocaleString('en-US')}</strong>`)
    expect(view.requests[0].usage).toEqual(item.usage)
    expect(view.used).toBe(input + 13)
  }
})

test('外部输入保留未知、真实零与原生处理量，非法缓存不补零', () => {
  const cases = [
    { usage: { prompt_tokens: 100 }, expected: '未知' },
    { usage: { prompt_tokens_details: { cached_tokens: 10 } }, expected: '未知' },
    { usage: { prompt_tokens: 100, prompt_tokens_details: { cached_tokens: 0 } }, expected: '100 Token' },
    { usage: { prompt_tokens: 100, prompt_tokens_details: { cached_tokens: 100 } }, expected: '0 Token' },
    { usage: { prompt_tokens: 100, prompt_tokens_details: { cached_tokens: 101 } }, timings: { prompt_n: 20 }, expected: '未知' },
    { usage: { prompt_tokens: 100 }, timings: { prompt_n: 60 }, expected: '60 Token' },
    { usage: {}, prefill: { prefill_processed_tokens: 42 }, expected: '42 Token' },
  ]
  for (const { expected, ...extra } of cases) {
    const view = externalGenerationView(snapshot([record(extra)]))
    const uncached = requestTokens(view.requests[0]).uncached
    expect(uncached === undefined ? '未知' : `${uncached} Token`).toBe(expected)
    const input = requestTokens(view.requests[0]).input
    expect(renderTable(view)).toContain(`aria-label="输入（含缓存） ${input === undefined ? '未知' : `${input} Token`}`)
  }
})

test('旧外部轮次拆成请求后保持各自输入与缓存配对，不相互抵扣或累计', () => {
  const requests = [record({ request_key: 'input-only', usage: { prompt_tokens: 100, completion_tokens: 1 } }),
    record({ request_key: 'cache-only', usage: { prompt_tokens_details: { cached_tokens: 25 }, completion_tokens: 1 } }),
    record({ request_key: 'paired', usage: { prompt_tokens: 40, prompt_tokens_details: { cached_tokens: 10 }, completion_tokens: 1 } })]
    .map(request => ({ ...request, aggregation: 'turn' as const, turn_id: 'T', turn_state: 'completed' as const }))
  const views = requests.map(request => externalGenerationView(snapshot([request], { aggregation: 'turn', turn_id: 'T' })))
  expect(views.map(view => requestTokens(view.requests[0]).input)).toEqual([100, undefined, 40])
  expect(views.map(view => requestTokens(view.requests[0]).cached)).toEqual([undefined, 25, 10])
  expect(requestTokens(views[2].requests[0]).uncached).toBe(30)
  for (const [index, input] of [100, undefined, 40].entries()) {
    expect(renderTable(views[index])).toContain(`aria-label="输入（含缓存） ${input === undefined ? '未知' : `${input} Token`}`)
    expect(renderTable(views[index])).toContain('aria-label="输出 1 Token"')
    expect(views[index].turn).toBeUndefined()
  }
})

test('外部Spark无阶段采样只显示等待图框和原生AVG，不伪造Decode点或曲线', () => {
  const item = record({ backend: 'sparkinfer', usage: { prompt_tokens: 100, completion_tokens: 20, decode_tps: 50, generation_ms: 400, ttft_ms: 80 } })
  const view = externalGenerationView(snapshot([item]))
  const html = render(view)
  expect(html).toContain('class="monitor-generation-chart"')
  expect(html).toContain('这次请求没有实时采样点')
  expect(html).not.toContain('最后采样')
  expect(html).toContain('引擎平均 · 含首 Token 等待')
  expect(renderTable(view)).toContain('aria-label="引擎平均 50.00 token/s')
  expect(html).not.toContain('class="trend-dot"')
  expect(html).not.toContain('class="trend-line"')
  expect(html).not.toContain('本轮')
  expect(view.requests[0].decodeMean).toBeNull()
  expect(view.requests[0].phases.decode.points).toEqual([])
})

test('历史总列表继续只显示表格，普通请求详情显示自身图框', () => {
  const view = externalGenerationView(snapshot([record()]), undefined, null, true)
  expect(renderTable(view)).not.toContain('class="monitor-generation-chart"')
  expect(render(view)).toContain('class="monitor-generation-chart"')
})

test('内置聊天监控保留总输入、原有轮次统计与图表口径', () => {
  const request: ModelRequestRecord = { id: 'internal', index: 1, kind: 'generation', status: 'completed', startedAt: 100000, endedAt: 104000, elapsed: 4, labels: [],
    usage: { prompt_tokens: 100, prompt_tokens_details: { cached_tokens: 90 }, completion_tokens: 20 }, timings: { predicted_n: 20, predicted_ms: 1000 } }
  const turn = turnTotals([request], 100000, 4, 'completed', 104000)
  const view = monitorGenerationView(null, undefined, undefined, false, [], { instanceId: '', messageId: 'internal-message', phase: 'completed', requests: [request], turn })
  const html = render(view, 'turn')
  expect(renderTable(view)).toContain('aria-label="输入（含缓存） 100 Token"')
  expect(html).toContain('<dt>本轮输出</dt><dd><span class="monitor-number"><strong>20</strong>')
  expect(html).toContain('本轮平均')
  expect(view.turn?.input).toEqual({ value: 100, reported: 1 })
})
