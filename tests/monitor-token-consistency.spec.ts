import { expect, test } from '@playwright/test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { MonitorGeneration } from '../src/MonitorGeneration'
import { MessageMetrics } from '../src/ChatMessageMeta'
import { externalModelRequest, monitorGenerationView, rememberChatStream, rememberGeneration } from '../src/monitor-generation'
import { engineRateTotal, mergeRequestMetrics, mergeRequestReport, requestRate, requestTokens, sampleStreamDecode, turnTotals, type ModelRequestRecord } from '../src/chat-turn-metrics'
import type { ChatGenerationActivity, RuntimeInstance, Telemetry } from '../src/types'

const instance = { id: 'token-instance', alias: 'token fixture', backend: 'llama_cpp', state: 'ready', active_requests: 1 } as RuntimeInstance
const record: ModelRequestRecord = { id: 'new-call', requestKey: 'new-server-call', index: 1, kind: 'generation', status: 'running', startedAt: 200000, labels: [], usage: { prompt_tokens: 1000, completion_tokens: 30 }, timings: { predicted_n: 30, predicted_ms: 100 } }
const first = sampleStreamDecode(undefined, record.id, { timings: { predicted_n: 10, predicted_ms: 0 } }, 200000)!
const stream = sampleStreamDecode(first, record.id, { timings: { predicted_n: 30, predicted_ms: 100 } }, 200100)!.sample
const activity: ChatGenerationActivity = { instanceId: instance.id, messageId: 'new-turn', phase: 'generating', requests: [record],
  turn: turnTotals([record], record.startedAt, .1, 'running'), stream }
const oldTelemetry: Telemetry = { instance_id: instance.id, backend: instance.backend, state: 'ready', updated_at: 199,
  groups: [], unavailable: [], live_generation: { request_key: 'previous-server-call', client_request_id: 'previous-turn', client_call_id: 'previous-call',
    phase: 'output', scope: 'request', source: 'older polling result', tokens_per_second: 4, window_s: 3,
    backend_running: 1, backend_queued: 0, available_for_chat: true, reason: null } }

test('Prefill处理进度不是完整输入；已知真实total优先于processed加cache', () => {
  const progress = { ...record, usage: undefined, timings: { prompt_n: 50, cache_n: 100 },
    prefill: { prefill_total_tokens: 1000, prefill_processed_tokens: 50, prefill_cached_tokens: 100 } }
  expect(requestTokens(progress).input).toBe(1000)
})

test('fresh counters按本帧替换，不被sticky usage或timings盖住；null和0均为权威', () => {
  const first = mergeRequestMetrics({ ...record, usage: undefined, timings: undefined }, { usage: { prompt_tokens: 1000, completion_tokens: 100 }, timings: { predicted_n: 100, predicted_ms: 2500 } })
  const native = mergeRequestMetrics(first, { usage: {}, timings: { predicted_n: 150 } })
  expect(requestTokens(native).output).toBe(150)
  const usage = mergeRequestMetrics(native, { usage: { completion_tokens: 160 }, timings: {} })
  expect(requestTokens(usage).output).toBe(160)
  const both = mergeRequestMetrics(usage, { usage: { completion_tokens: 170 }, timings: { predicted_n: 180 } })
  expect(requestTokens(both).output).toBe(180)
  expect(requestTokens({ ...both, counters: { input: null, output: null, cached: null } }).output).toBeUndefined()
  expect(requestTokens({ ...both, counters: { input: 0, output: 0, cached: 0 } })).toMatchObject({ input: 0, output: 0, cached: 0 })
})

test('外部慢照和归档继续使用规范counter与partial配对，不在快缓存释放后回退sticky字段', () => {
  const paired = mergeRequestMetrics(record, { usage: { completion_tokens: 100 }, timings: { predicted_n: 100, predicted_ms: 2500 } })
  const partial = mergeRequestMetrics(paired, { usage: {}, timings: { predicted_n: 150 } })
  const mapped = externalModelRequest({ ...partial, run_id: 'run', instance_id: instance.id, request_key: 'call', status: 'completed',
    started_at: 200, duration_s: 3, alias: 'model', backend: 'ninfer', labels: [], revision: 2, aggregation: 'turn', turn_id: 'turn', turn_state: 'completed' }, 1)
  expect(requestTokens(mapped).output).toBe(150)
  expect(turnTotals([mapped], 0, 3).decode).toMatchObject({ value: 40, tokens: 100, seconds: 2.5, partial_reported: 1 })
})

test('仅原生per_token_ms仍是请求均值而非unknown，孤立新counter不能复活旧均值', () => {
  const base = { ...record, usage: undefined, timings: undefined }
  const mean = mergeRequestMetrics(base, { usage: {}, timings: { predicted_per_token_ms: 20, prompt_per_token_ms: .2 } })
  expect(requestRate(mean, 'decode').value).toBe(50)
  expect(requestRate(mean, 'prefill').value).toBe(5000)
  expect(turnTotals([mean], 0, 1).decode.reported).toBe(0)
  const counter = mergeRequestMetrics(mean, { usage: {}, timings: { predicted_n: 100, prompt_n: 500 } })
  expect(requestRate(counter, 'decode').value).toBeUndefined()
  expect(requestRate(counter, 'prefill').value).toBeUndefined()
})

test('终态引擎AVG的加权计数也读取规范counter，不把旧usage100配真实240的时钟', () => {
  const completed: ModelRequestRecord = { ...record, status: 'completed', timings: {}, usage: { completion_tokens: 100, generation_ms: 8000, decode_tps: 30 },
    counters: { input: 1000, output: 240, cached: 100 } }
  expect(engineRateTotal([completed])).toMatchObject({ value: 30, tokens: 240, seconds: 8, reported: 1 })
})

test('n大于1的内置chunk保留usage聚合，不把单分支native速度/计数当整请求', () => {
  const chunk = { usage: { prompt_tokens: 1000, completion_tokens: 100 }, timings: { predicted_n: 50, predicted_ms: 100, predicted_per_second: 500, prompt_n: 500, prompt_ms: 50, cache_n: 100 } }
  const multiple = mergeRequestMetrics({ ...record, usage: undefined, timings: undefined }, chunk, true)
  expect(requestTokens(multiple).output).toBe(100)
  expect(requestRate(multiple, 'decode').value).toBeUndefined()
  expect(turnTotals([multiple], 0, 1).decode.reported).toBe(0)
  expect(chunk.timings.predicted_n).toBe(50)
  const unknown = { tokens: null, seconds: null, value: null, basis: null, reported: 0 as const, coverage: 'unknown' as const }
  const terminal = mergeRequestReport({ ...record, timings: chunk.timings }, { request_key: record.requestKey!, status: 'completed', duration_s: 1,
    usage: chunk.usage, timings: {}, counters: { input: 1000, output: 100, cached: 100 }, measurements: { decode: unknown, prefill: unknown } })
  expect(requestRate(terminal, 'decode').value).toBeUndefined()
})

test('internal同次progress只使用真实total/cache/work，Decode后的孤立进度不回写Prefill', () => {
  const base = { ...record, usage: undefined, timings: undefined }
  const progress = mergeRequestMetrics(base, { usage: {}, timings: { prompt_ms: 10 }, prompt_progress: { total: 1000, processed: 100, cache: 50 } })
  expect(requestTokens(progress)).toMatchObject({ input: 1000, cached: 50 })
  expect(requestRate(progress, 'prefill')).toMatchObject({ tokens: 50, seconds: .01 })
  const processed = mergeRequestMetrics(progress, { usage: {}, timings: { prompt_n: 200, cache_n: 60 }, prompt_progress: { processed: 260, cache: 20 } })
  expect(requestTokens(processed)).toMatchObject({ input: 1000, cached: 60 })
  const decode = mergeRequestMetrics(processed, { usage: { completion_tokens: 10 }, timings: { predicted_n: 10, predicted_ms: 100 } })
  const late = mergeRequestMetrics(decode, { usage: {}, timings: {}, prompt_progress: { total: 10, processed: 10, cache: 0 } })
  expect(requestTokens(late)).toMatchObject({ input: 1000, output: 10, cached: 60 })
  expect(requestRate(late, 'prefill').tokens).toBe(requestRate(decode, 'prefill').tokens)
})

test('内置对话本轮原生配对前缀partial即使1/1有报告也保留部分标识', () => {
  const paired = mergeRequestMetrics(record, { usage: {}, timings: { predicted_n: 100, predicted_ms: 2500 } })
  const partial = mergeRequestMetrics(paired, { usage: {}, timings: { predicted_n: 150 } })
  const turn = turnTotals([partial], record.startedAt, 3)
  const html = renderToStaticMarkup(createElement(MessageMetrics, { message: { id: 'm', requests: [partial], turn } as any, generating: true, liveError: '' }))
  expect(html).toContain('部分')
})

test('同次原生配对前缀不被sticky旧毫秒和新counter/mean重组', () => {
  const base = { ...record, usage: undefined, timings: undefined, live: undefined }
  const report = (timings: Record<string, number>) => ({ request_key: record.requestKey!, status: 'running' as const, duration_s: 3, usage: {}, timings })
  const paired = mergeRequestReport(base, report({ predicted_n: 100, predicted_ms: 2500, prompt_n: 500, prompt_ms: 100 }))
  const counterOnly = mergeRequestReport(paired, report({ predicted_n: 150, prompt_n: 600 }))
  expect(requestTokens(counterOnly).output).toBe(150)
  expect(requestRate(counterOnly, 'decode')).toMatchObject({ tokens: 100, seconds: 2.5, value: 40 })
  expect(requestRate(counterOnly, 'prefill')).toMatchObject({ tokens: 500, seconds: .1, value: 5000 })
  const meanOnly = mergeRequestReport(counterOnly, report({ predicted_per_second: 50, prompt_per_second: 6000 }))
  expect(requestRate(meanOnly, 'decode').value).toBe(50)
  expect(turnTotals([meanOnly], 0, 3).decode).toMatchObject({ value: 40, tokens: 100, seconds: 2.5, partial_reported: 1 })
  const clockOnly = mergeRequestReport(meanOnly, report({ predicted_ms: 3000 }))
  expect(turnTotals([clockOnly], 0, 3).decode.value).toBe(40)
  expect(requestRate(clockOnly, 'decode').value).toBe(40)
  const nextPair = mergeRequestReport(clockOnly, report({ predicted_n: 150, predicted_ms: 3000 }))
  expect(turnTotals([nextPair], 0, 3).decode).toMatchObject({ value: 50, tokens: 150, seconds: 3 })
  const finalPair = mergeRequestReport(nextPair, { ...report({ predicted_n: 150, predicted_ms: 5000 }), status: 'completed' })
  expect(requestRate(finalPair, 'decode').value).toBe(30)
  const nextCounter = mergeRequestReport(finalPair, report({ predicted_n: 180 }))
  expect(requestRate(nextCounter, 'decode').value).toBe(30)
})

test('同帧native rate可以配对，但孤立rate不能反推旧counter；unknown与真实零分开', () => {
  const base = { ...record, usage: undefined, timings: undefined, live: undefined }
  const pair = mergeRequestReport(base, { request_key: record.requestKey!, status: 'running', duration_s: 1, usage: { completion_tokens: 100 }, timings: { predicted_per_second: 40 } })
  expect(requestRate(pair, 'decode')).toMatchObject({ tokens: 100, seconds: 2.5, value: 40 })
  const meanOnly = mergeRequestReport(pair, { request_key: record.requestKey!, status: 'running', duration_s: 2, usage: {}, timings: { predicted_per_second: 50 } })
  expect(requestRate(meanOnly, 'decode')).toMatchObject({ tokens: 100, seconds: 2.5, value: 50 })
  expect(turnTotals([meanOnly], 0, 2).decode.value).toBe(40)
  const unknown = mergeRequestReport(base, { request_key: record.requestKey!, status: 'running', duration_s: 1, usage: {}, timings: { predicted_n: 0 } })
  expect(turnTotals([unknown], 0, 1).decode).toMatchObject({ value: null, reported: 0 })
  const zero = mergeRequestReport(base, { request_key: record.requestKey!, status: 'running', duration_s: 1, usage: {}, timings: { predicted_n: 0, predicted_ms: 1000 } })
  expect(turnTotals([zero], 0, 1).decode).toMatchObject({ value: 0, reported: 1 })
})

test('新SSE请求已到达时，前一请求的轮询身份不能让默认监控丢掉逐token实时值', () => {
  const memory = rememberChatStream(rememberGeneration(undefined, oldTelemetry), activity)
  const view = monitorGenerationView(oldTelemetry, memory, instance, false, [activity])
  expect(view.origin).toBe('internal')
  expect(view.requests.map(request => request.id)).toEqual([record.id])
  expect(view.current).toBe(200)
  const html = renderToStaticMarkup(createElement(MonitorGeneration, { generation: view, scope: 'request', selectedRequestId: record.id, onSelectRequest: () => {} }))
  expect(html).toContain('<strong>200.00</strong>')
  expect(html).toContain('实时速度')
})

test('旧轮已完成但轮询身份未切换时，当前请求每个SSE计数事件仍更新可见值', () => {
  const finished: ChatGenerationActivity = { ...activity, messageId: 'previous-turn', phase: 'completed', stream: undefined,
    requests: [{ ...record, id: 'previous-call', requestKey: 'previous-server-call', status: 'completed', startedAt: 1000, endedAt: 9000 }] }
  let memory = rememberGeneration(undefined, oldTelemetry), state = first
  for (const [tokens, milliseconds, expected] of [[20, 100, 100], [40, 200, 150], [70, 300, 200]]) {
    state = sampleStreamDecode(state, record.id, { timings: { predicted_n: tokens, predicted_ms: milliseconds } }, 200000 + milliseconds)!
    const current = { ...record, usage: { prompt_tokens: 1000, completion_tokens: tokens }, timings: { predicted_n: tokens, predicted_ms: milliseconds } }
    const event = { ...activity, requests: [current], turn: turnTotals([current], record.startedAt, milliseconds / 1000), stream: state.sample }
    memory = rememberChatStream(memory, event)
    const view = monitorGenerationView(oldTelemetry, memory, instance, false, [finished, event])
    expect(view.requests[0].id).toBe(record.id)
    const html = renderToStaticMarkup(createElement(MonitorGeneration, { generation: view, scope: 'request', selectedRequestId: record.id, onSelectRequest: () => {} }))
    expect(html).toContain(`<strong>${expected.toFixed(2)}</strong>`)
    expect(html).toContain('实时速度')
    expect(view.requests[0].decodeMean).toBe(tokens / (milliseconds / 1000))
    expect(view.turn?.output.value).toBe(tokens)
    expect(view.turn?.decode.value).toBe(tokens / (milliseconds / 1000))
  }
})

test('并行内置活动仍只匹配live调用，不把另一轮的SSE速度借给所选请求', () => {
  const other = { ...activity, messageId: 'another-turn', requests: [{ ...record, id: 'another-call' }], stream: { ...stream, requestId: 'another-call', tokensPerSecond: 999 } }
  const matching = { ...oldTelemetry, live_generation: { ...oldTelemetry.live_generation!, client_request_id: activity.messageId, client_call_id: record.id, request_key: record.requestKey! } }
  const memory = rememberChatStream(rememberChatStream(rememberGeneration(undefined, matching), activity), other)
  const view = monitorGenerationView(matching, memory, instance, false, [activity, other])
  expect(view.requests.map(request => request.id)).toEqual([record.id])
  expect(view.requests[0].phases.decode.current).toBe(200)
})

test('当前SSE活跃请求在轮摘要显示均值；单请求显示实时值，历史轮不借实时值', () => {
  const memory = rememberChatStream(undefined, activity)
  const view = monitorGenerationView(null, memory, instance, false, [activity])
  const render = (scope: 'turn' | 'request') => renderToStaticMarkup(createElement(MonitorGeneration, { generation: view, scope, selectedRequestId: record.id, onSelectRequest: () => {} }))
  expect(render('request')).toContain('<strong>200.00</strong>')
  expect(render('request')).toContain('实时速度')
  expect(render('turn')).toContain('<strong>300.00</strong>')
  expect(render('turn')).toContain('本轮平均')
  const history = { ...activity, messageId: 'old-history', phase: 'completed' as const, requests: [{ ...record, id: 'old-history-call', status: 'completed' as const }] }
  const historical = monitorGenerationView(null, memory, instance, false, [activity], history)
  expect(historical.requests[0].phases.decode.current).toBeNull()
})

test('新鲜poll的另一身份与多个未匹配内置活动不能被旧SSE抢占', () => {
  const fresh = { ...oldTelemetry, updated_at: 201 }
  const memory = rememberChatStream(rememberGeneration(undefined, fresh), activity)
  expect(monitorGenerationView(fresh, memory, instance, false, [activity]).requests.some(request => request.id === record.id)).toBe(false)
  const other = { ...activity, messageId: 'another-turn', requests: [{ ...record, id: 'another-call' }] }
  const stale = rememberChatStream(rememberChatStream(rememberGeneration(undefined, oldTelemetry), activity), other)
  const ambiguous = monitorGenerationView(oldTelemetry, stale, instance, false, [activity, other])
  expect(ambiguous.requests.some(request => [record.id, 'another-call'].includes(request.id))).toBe(false)
})
