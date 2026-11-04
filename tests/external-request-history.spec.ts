import { expect, test } from '@playwright/test'
import type { ChatGenerationActivity, ExternalHistorySnapshot, ExternalRequestRecord, LiveGeneration, RuntimeInstance, Telemetry } from '../src/types'
import { externalGenerationView, externalRequestIdentity, monitorGenerationView, rememberGeneration, updatedHistorySelection } from '../src/monitor-generation'
import { requestEngineRate, requestFirstOutput, requestTokens, turnTotals } from '../src/chat-turn-metrics'

const instance = { id: 'external-instance', alias: 'fixture model', backend: 'llama_cpp', state: 'ready', active_requests: 0 } as RuntimeInstance
function record(key: string, extra: Partial<ExternalRequestRecord> = {}): ExternalRequestRecord {
  return { request_key: key, run_id: 'current', instance_id: instance.id, alias: instance.alias, backend: instance.backend,
    revision: 1, aggregation: 'request', turn_id: null, turn_state: null, labels: [], status: 'completed',
    started_at: 200, ended_at: 204, duration_s: 4, usage: { prompt_tokens: 100, completion_tokens: 20 },
    timings: { predicted_n: 20, predicted_ms: 2000 }, ...extra }
}
function snapshot(requests: ExternalRequestRecord[], extra: Partial<ExternalHistorySnapshot> = {}): ExternalHistorySnapshot {
  return { run_id: 'current', aggregation: 'request', turn_id: null, state: 'completed', requests, ...extra }
}
function telemetry(history: ExternalHistorySnapshot, live?: Partial<LiveGeneration>, time = 220): Telemetry {
  return { instance_id: instance.id, backend: instance.backend, state: 'ready', updated_at: time, groups: [], unavailable: [], external_history: history,
    live_generation: { request_key: null, scope: 'request', source: 'fixture', backend_running: 0, backend_queued: 0,
      tokens_per_second: null, window_s: null, reason: null, available_for_chat: false, phase: 'idle', ...live } }
}

test('外部普通调用即使重复客户端标识且有实时速度也保持独立请求，不合成轮次', () => {
  const first = record('first', { client_request_id: 'same-client' })
  const before = telemetry(snapshot([first]))
  let memory = rememberGeneration(undefined, before)
  const second = record('second', { started_at: 210, ended_at: undefined, status: 'running', client_request_id: 'same-client', duration_s: .1, usage: {}, timings: {} })
  const current = telemetry(snapshot([second], { state: 'running' }), { request_key: second.request_key, client_request_id: second.client_request_id,
    phase: 'output', backend_running: 1, tokens_per_second: 200, generated_tokens: 20, average_tokens_per_second: 180, average_basis: 'backend_decode' })
  memory = rememberGeneration(memory, current)
  const view = monitorGenerationView(current, memory, instance)
  expect(view).toMatchObject({ aggregation: 'request', origin: 'external', active: true, status: '正在输出' })
  expect(view.requests.map(request => request.id)).toEqual([externalRequestIdentity(second)])
  expect(view.turn).toBeUndefined()
  expect(view.requests[0].phases.decode.points.map(point => point.values.current)).toEqual([200])
  expect(view.tools).toEqual([])
  expect(monitorGenerationView(current, memory, instance, true)).toMatchObject({ status: '采样失败', active: false, working: false, current: null })
  expect(monitorGenerationView(null, memory, { ...instance, state: 'stopped' })).toMatchObject({ status: '已停止', active: false, working: false })
})

test('外部工具标签保留完整原名且去重，旧记录没有名字时不显示泛化标签', () => {
  const tool_names = ['mcp__web__search', 'functions.read_page', 'mcp__web__search']
  const named = record('named', { labels: ['tool_calls'], tool_names })
  const view = externalGenerationView(snapshot([named]))
  expect(view.requests[0].labels).toEqual(['mcp__web__search', 'functions.read_page'])
  expect(view.requests[0].external?.tool_names).toEqual(tool_names)
  const legacy = externalGenerationView(snapshot([record('legacy', { labels: ['tool_calls'] })])).requests[0]
  expect(legacy.labels).toEqual([])
  expect(legacy.usage).toEqual({ prompt_tokens: 100, completion_tokens: 20 })
  expect(legacy.decodeMean).toBe(10)
})

test('没有明确压缩证据的摘要请求保持普通请求和完整计量，不从名称或 Token 推断压缩', () => {
  const item = record('summary-request', {
    alias: 'conversation summary', labels: ['tool_calls'], tool_names: ['compress_context'], first_output_s: 0.75,
    usage: { prompt_tokens: 8000, completion_tokens: 400, prompt_tokens_details: { cached_tokens: 2000 } },
    timings: { prompt_n: 6000, prompt_ms: 1500, predicted_n: 400, predicted_ms: 2000 },
  })
  const view = externalGenerationView(snapshot([item]))
  expect(view.aggregation).toBe('request')
  expect(view.turn).toBeUndefined()
  expect(view.requests[0]).toMatchObject({ kind: 'generation', elapsed: 4, firstOutput: 0.75, decodeMean: 200, prefillMean: 4000, labels: ['compress_context'] })
  expect(view.requests[0].usage).toEqual(item.usage)
  expect(view.requests[0].timings).toEqual(item.timings)
  expect(requestTokens(view.requests[0])).toMatchObject({ input: 8000, uncached: 6000, cached: 2000, output: 400 })
  const grouped = externalGenerationView(snapshot([item], { aggregation: 'turn', turn_id: 'summary-turn' }))
  expect(grouped.turn).toBeUndefined()
  expect(grouped.requests[0]).toMatchObject({ usage: item.usage, decodeMean: 200, prefillMean: 4000 })
})

test('旧关联轮次中的短请求逐次可查，不把邻接请求的计量借给缺测请求', () => {
  const requests = [record('one', { aggregation: 'turn', turn_id: 'turn', timings: { predicted_n: 20, predicted_ms: 2000 } }),
    record('two', { aggregation: 'turn', turn_id: 'turn', started_at: 210, ended_at: 214, usage: { prompt_tokens: 200, completion_tokens: 60 }, timings: { predicted_n: 60, predicted_ms: 3000 } }),
    record('short', { aggregation: 'turn', turn_id: 'turn', started_at: 215, ended_at: 216, duration_s: 1, usage: {}, timings: {} })]
  const views = requests.map(request => externalGenerationView(snapshot([request], { aggregation: 'turn', turn_id: 'turn' })))
  expect(views.map(view => view.requests.length)).toEqual([1, 1, 1])
  expect(views.map(view => view.requests[0].decodeMean)).toEqual([10, 20, null])
  expect(views[2].requests[0]).toMatchObject({ decodeMean: null, prefillMean: null, firstOutput: undefined })
  expect(views.every(view => !view.turn && !view.requests[0].phases.decode.points.length && !view.tools.length && view.toolCount === undefined)).toBe(true)
})

test('外部工具调用完成即终态，旧waiting元数据不让时钟继续且保留最后曲线', () => {
  const item = record('one', { aggregation: 'turn', turn_id: 'turn', turn_state: 'waiting', labels: ['tool_calls'], tool_names: ['mcp__web__search'] })
  const active = telemetry(snapshot([{ ...item, status: 'running', ended_at: undefined }], { aggregation: 'turn', turn_id: 'turn', state: 'running' }),
    { request_key: item.request_key, phase: 'output', backend_running: 1, tokens_per_second: 155 }, 203)
  const sampled = rememberGeneration(undefined, active)
  const waiting = snapshot([item], { aggregation: 'turn', turn_id: 'turn', state: 'waiting' })
  const first = telemetry(waiting, undefined, 204), later = telemetry(waiting, undefined, 900)
  const memory = rememberGeneration(sampled, first)
  const a = monitorGenerationView(first, memory, instance), b = monitorGenerationView(later, rememberGeneration(memory, later), instance)
  expect(a).toMatchObject({ status: '已完成', aggregation: 'request', waiting: false, active: false, working: false, ended: true })
  expect(a.turn).toBeUndefined()
  expect(b.request?.duration).toBe(a.request?.duration)
  expect(b.requests[0].phases.decode.points).toBe(a.requests[0].phases.decode.points)
  expect(b.requests[0].labels).toEqual(['mcp__web__search'])
  expect(b.requests[0].external?.labels).toEqual(['tool_calls'])
  expect(b.tools).toEqual([])
})

test('被固定的普通请求可按精确结果升级轮次，不被过期分页或其他记录换走', () => {
  const selected = record('selected', { status: 'running', ended_at: undefined })
  const other = record('newest', { revision: 3 })
  const upgraded = { ...selected, revision: 2, aggregation: 'turn' as const, turn_id: 'detected-turn', turn_state: 'waiting' as const, status: 'completed' as const }
  expect(updatedHistorySelection(selected, [other])).toBe(selected)
  const current = updatedHistorySelection(selected, [upgraded])
  expect(current).toMatchObject({ request_key: selected.request_key, revision: 2, aggregation: 'turn', turn_id: 'detected-turn', turn_state: 'waiting', status: 'completed' })
  expect(updatedHistorySelection(current, [selected])).toBe(current)
  expect(updatedHistorySelection(current, [{ ...upgraded, run_id: 'another-run', revision: 3 }])).toBe(current)
  expect(updatedHistorySelection(null, [upgraded])).toBeNull()
})

test('内置聊天与外部调用以实际完成时间选源，终态不回退到旧内置活动', () => {
  const internalRequest = { id: 'internal', index: 1, kind: 'generation' as const, status: 'completed' as const, startedAt: 100000, endedAt: 105000, labels: [] }
  const activity: ChatGenerationActivity = { instanceId: instance.id, messageId: 'internal-message', phase: 'completed', requests: [internalRequest],
    turn: turnTotals([internalRequest], 100000, 5, 'completed', 105000) }
  const payload = telemetry(snapshot([record('external')]))
  const memory = rememberGeneration(undefined, payload)
  expect(monitorGenerationView(payload, memory, instance, false, [activity])).toMatchObject({ origin: 'external', status: '已完成' })
  expect(monitorGenerationView(null, memory, instance, false, [activity]).requests[0].id).toBe(externalRequestIdentity(record('external')))
  const newestRequest = { ...internalRequest, id: 'new-internal', startedAt: 300000, endedAt: 305000 }
  const newest = { ...activity, messageId: 'new-internal', requests: [newestRequest], turn: turnTotals([newestRequest], 300000, 5, 'completed', 305000) }
  expect(monitorGenerationView(payload, memory, instance, false, [activity, newest])).toMatchObject({ origin: 'internal', status: '已完成' })
  expect(monitorGenerationView(payload, memory, instance, false, [activity], activity).requests[0].id).toBe('internal')
})

test('请求终态不会遗留等待，旧运行与已移除实例的缺尾记录独立可读', () => {
  const complete = externalGenerationView(snapshot([record('done')]))
  expect(complete).toMatchObject({ status: '已完成', active: false, working: false, ended: true, waiting: false })
  expect(complete.turn).toBeUndefined()
  const archived = record('partial', { run_id: 'prior', instance_id: 'removed', status: 'incomplete', record_incomplete: true,
    ended_at: undefined, usage: {}, timings: {}, duration_s: 0 })
  const view = externalGenerationView(snapshot([archived], { run_id: 'prior', state: 'incomplete' }), undefined, null, true)
  expect(view).toMatchObject({ historical: true, status: '记录未完整', capacity: null, used: null })
  expect(view.requests[0]).toMatchObject({ status: 'incomplete', external: { instance_id: 'removed', record_incomplete: true } })
  expect(view.requests[0].status).not.toBe('failed')
})

test('外部Spark AVG与引擎TTFT保留原口径，不生成Decode或采样点', () => {
  const spark = record('spark', { backend: 'sparkinfer', first_output_s: 1.2, usage: { prompt_tokens: 1000, completion_tokens: 200,
    ttft_ms: 500, generation_ms: 4000, decode_tps: 50 }, timings: {} })
  const view = externalGenerationView(snapshot([spark]))
  expect(requestEngineRate(view.requests[0]).value).toBe(50)
  expect(requestFirstOutput(view.requests[0])).toEqual({ value: .5, basis: '引擎' })
  expect(view.requests[0]).toMatchObject({ decodeMean: null, prefillMean: null, phases: { decode: { current: null, points: [] } } })
  expect(view.turn).toBeUndefined()
})
