import { expect, test } from '@playwright/test'
import { engineRateTotal, footerRequest, mergeRequestReport, requestEngineRate, requestFirstOutput, requestRate, requestTokens, sampleStreamDecode, turnTotals, type ModelRequestRecord, type StreamDecodeState } from '../src/chat-turn-metrics'
import { chatRequestLive, matchingRequestLive, mergeRequestLive, monitorGenerationView, rememberChatStream, rememberGeneration } from '../src/monitor-generation'
import type { ChatGenerationActivity, LiveGeneration, RuntimeInstance, Telemetry } from '../src/types'
import { miniSnapshot } from '../src/mini-monitor'
import type { AppState } from '../src/types'
import type { MonitorState } from '../src/useMonitorState'

const request = (id: string, extra: Partial<ModelRequestRecord> = {}): ModelRequestRecord => ({ id, index: 1, kind: 'generation', status: 'running', startedAt: 1000, labels: [], ...extra })
const live = (extra: Partial<LiveGeneration> = {}): LiveGeneration => ({ client_request_id: 'turn', client_call_id: 'call', request_key: 'server-call',
  tokens_per_second: 40, window_s: 2, scope: 'request', source: 'existing backend sampler', backend_running: 1, backend_queued: 0,
  available_for_chat: true, reason: null, phase: 'output', generated_tokens: 20, ...extra })
const pair = (predicted_n: number, predicted_ms: number) => ({ timings: { predicted_n, predicted_ms, predicted_per_second: 999 } })

test('Strata终帧前的真实计数进入账本，缓存缺报不当零，原生终帧优先', () => {
  const observation = live({ prompt_tokens: 1000, generated_tokens: 200, average_tokens_per_second: 100, average_basis: 'backend_decode',
    average_prefill_tokens_per_second: 800, prefill_average_basis: 'backend_prefill', request_elapsed_s: 3, first_output_s: .8 })
  let record = mergeRequestLive(request('call'), observation)
  expect(record).toMatchObject({ requestKey: 'server-call', elapsed: 3, firstOutput: .8, firstOutputBasis: 'gateway' })
  expect(record.usage).toBeUndefined(); expect(record.timings).toBeUndefined()
  expect(requestTokens(record)).toMatchObject({ input: 1000, output: 200, cached: undefined, uncached: undefined })
  expect(requestRate(record, 'decode')).toEqual({ value: 100, tokens: 200, seconds: 2 })
  expect(requestRate(record, 'prefill')).toEqual({ value: 800, tokens: undefined, seconds: undefined })
  record = mergeRequestLive(record, { ...observation, prefill_cached_tokens: 600, prefill_processed_tokens: 400 })
  expect(requestTokens(record)).toMatchObject({ input: 1000, uncached: 400, cached: 600, output: 200 })
  expect(turnTotals([record], 0, 3)).toMatchObject({ input: { value: 1000 }, output: { value: 200 }, decode: { value: 100 }, prefill: { value: 800 } })
  const final = mergeRequestReport(record, { request_key: 'server-call', status: 'completed', duration_s: 4,
    usage: { prompt_tokens: 1000, completion_tokens: 300, prompt_tokens_details: { cached_tokens: 620 } },
    timings: { predicted_n: 300, predicted_ms: 2500, prompt_n: 380, prompt_ms: 400, cache_n: 620 } })
  expect(requestTokens(final)).toMatchObject({ input: 1000, uncached: 380, cached: 620, output: 300 })
  expect(requestRate(final, 'decode')).toEqual({ value: 120, tokens: 300, seconds: 2.5 })
  expect(requestRate(final, 'prefill')).toEqual({ value: 950, tokens: 380, seconds: .4 })
  expect(mergeRequestLive(final, observation)).toBe(final)
  const rateOnly = { ...record, timings: { predicted_per_second: 120 } }
  expect(requestRate(rateOnly, 'decode')).toEqual({ value: 120, tokens: undefined, seconds: undefined })
})

test('实时账本拒绝其他call和模型级统计，采样均值不冒充后端Decode', () => {
  const record = request('call')
  expect(mergeRequestLive(record, live({ client_call_id: 'other' }))).toBe(record)
  expect(mergeRequestLive(record, live({ client_call_id: null, request_key: 'unknown' }))).toBe(record)
  expect(mergeRequestLive(record, live({ scope: 'model' }))).toBe(record)
  const sampled = mergeRequestLive(record, live({ average_tokens_per_second: 100, average_basis: 'sampled_decode',
    prefill_processed_tokens: 100, average_prefill_tokens_per_second: 200, prefill_average_basis: 'sampled_prefill' }))
  expect(turnTotals([sampled], 0, 3)).toMatchObject({ decode: { value: null }, prefill: { value: null } })
  const zero = mergeRequestLive(record, live({ prompt_tokens: 100, generated_tokens: 0, prefill_cached_tokens: 0 }))
  expect(requestTokens(zero)).toMatchObject({ input: 100, uncached: 100, cached: 0, output: 0 })
})

test('SSE累计解码计数逐事件计算增量，首样本及累计平均不冒充实时速度', () => {
  let state = sampleStreamDecode(undefined, 'call', pair(1, 0), 1000)!
  expect(state.sample.tokensPerSecond).toBeNull()
  state = sampleStreamDecode(state, 'call', pair(11, 100), 1010)!
  expect(state.sample).toEqual({ requestId: 'call', sampledAt: 1010, tokensPerSecond: 100, windowSeconds: .1 })
  state = sampleStreamDecode(state, 'call', pair(31, 200), 1020)!
  expect(state.sample.tokensPerSecond).toBe(150)
  expect(state.sample.sampledAt).toBe(1020) // No 500 ms display throttle.
  expect(sampleStreamDecode(state, 'call', pair(31, 200), 1030)).toBe(state)
  expect(sampleStreamDecode(undefined, 'call', { usage: { completion_tokens: 200 }, timings: { predicted_per_second: 999 } }, 1000)?.sample).toMatchObject({ tokensPerSecond: null, clock: 'arrival' })
})

test('连续usage按接收时间逐事件刷新，真实计时优先且两种时钟不混合', () => {
  let state = sampleStreamDecode(undefined, 'usage-call', { usage: { completion_tokens: 10 } }, 1000)!
  expect(state.sample).toMatchObject({ tokensPerSecond: null, clock: 'arrival' })
  state = sampleStreamDecode(state, 'usage-call', { usage: { completion_tokens: 30 } }, 1100)!
  expect(state.sample).toMatchObject({ tokensPerSecond: 200, windowSeconds: .1, sampledAt: 1100 })
  expect(sampleStreamDecode(state, 'usage-call', {}, 1200)).toBe(state)
  const matched = chatRequestLive(null, request('usage-call', { usage: { completion_tokens: 30 } }), state.sample)
  expect(matched).toMatchObject({ tokens_per_second: 200, source: 'SSE 累计 Token / 接收时间增量' })
  state = sampleStreamDecode(state, 'usage-call', pair(40, 500), 1200)!
  expect(state.sample.tokensPerSecond).toBeNull()
  state = sampleStreamDecode(state, 'usage-call', pair(50, 600), 1400)!
  expect(state.sample).toMatchObject({ tokensPerSecond: 100, windowSeconds: .1 })
  expect(state.sample.clock).toBeUndefined()
})

test('Spark终态原生时长与平均速度保留独立口径，不伪装为Decode或实时曲线', () => {
  const original = request('spark', { firstOutput: .8, firstOutputBasis: 'gateway' })
  const record = mergeRequestReport(original, { request_key: 'spark', status: 'completed', duration_s: 3, usage: {
    prompt_tokens: 1000, completion_tokens: 100, prompt_tokens_details: { cached_tokens: 600 }, ttft_ms: 500, generation_ms: 2000, decode_tps: 50,
  }, timings: {} })
  expect(requestFirstOutput(record)).toEqual({ value: .5, basis: '引擎' })
  expect(record.firstOutput).toBe(.8)
  expect(requestEngineRate(record)).toEqual({ value: 50, tokens: 100, seconds: 2 })
  expect(requestRate(record, 'decode').value).toBeUndefined()
  expect(turnTotals([record], 0, 3, 'completed').decode.value).toBeNull()
  expect(engineRateTotal([record])).toEqual({ value: 50, tokens: 100, seconds: 2, reported: 1 })
  expect(engineRateTotal([{ ...record, status: 'cancelled' }]).value).toBeNull()
  expect(requestTokens(record)).toMatchObject({ input: 1000, uncached: 400, cached: 600, output: 100 })
  expect(sampleStreamDecode(undefined, record.id, { usage: record.usage }, 5000)?.sample.tokensPerSecond).toBeNull()
  expect(requestFirstOutput({ usage: { ttft_ms: 0 } })).toEqual({ value: 0, basis: '引擎' })
})

test('没有在途统计时明细标为终态可见，轮询计数与速度仍走原路径', () => {
  const instance = { id: 'instance', state: 'ready', active_requests: 1 } as RuntimeInstance
  const record = request('call')
  const activity: ChatGenerationActivity = { instanceId: instance.id, messageId: 'turn', phase: 'generating', requests: [record] }
  expect(monitorGenerationView(null, undefined, instance, false, [activity]).requests[0].liveMetrics).toBe(false)
  activity.requests = [mergeRequestLive(record, live({ generated_tokens: 10 }))]
  expect(monitorGenerationView(null, undefined, instance, false, [activity]).requests[0].liveMetrics).toBe(true)
})

test('外部Spark请求通过遥测终态保留原生时长与AVG，不依赖聊天SSE', () => {
  const telemetry: Telemetry = { instance_id: 'instance', backend: 'sparkinfer', state: 'ready', updated_at: 10, unavailable: [], groups: [{
    id: 'api_request', title: '请求', request_key: 'external', request_status: 'completed', metrics: [
      { id: 'prompt_tokens', label: '输入', value: 1000, kind: 'gauge' }, { id: 'completion_tokens', label: '输出', value: 200, kind: 'gauge' },
      { id: 'duration_s', label: '耗时', value: 6, kind: 'gauge' }, { id: 'first_output_s', label: '网关首输出', value: 1.2, kind: 'gauge' },
      { id: 'backend_ttft_ms', label: '原生首Token', value: 500, kind: 'gauge' }, { id: 'backend_generation_ms', label: '原生生成', value: 4000, kind: 'gauge' },
      { id: 'backend_average_tps', label: '引擎平均', value: 50, kind: 'gauge', rate_basis: 'end_to_end' },
      { id: 'output_speed', label: '引擎平均', value: 50, kind: 'gauge', rate_basis: 'end_to_end' },
    ],
  }] }
  const view = monitorGenerationView(telemetry, rememberGeneration(undefined, telemetry), { id: 'instance', state: 'ready', active_requests: 0 } as RuntimeInstance)
  expect(view.requests).toHaveLength(1)
  expect(requestEngineRate(view.requests[0])).toEqual({ value: 50, tokens: 200, seconds: 4 })
  expect(requestFirstOutput(view.requests[0])).toEqual({ value: .5, basis: '引擎' })
  expect(view.requests[0].decodeMean).toBeNull()
})

test('小窗跟随相同SSE计数和速度，Spark终态以引擎平均显示', () => {
  const instance = { id: 'instance', alias: 'fixture', state: 'ready', active_requests: 1 } as RuntimeInstance
  const record = request('call', { usage: { prompt_tokens: 1000, completion_tokens: 30 } })
  const activity: ChatGenerationActivity = { instanceId: instance.id, messageId: 'turn', phase: 'generating', requests: [record],
    stream: { requestId: record.id, sampledAt: 1000, tokensPerSecond: 150, windowSeconds: .2, clock: 'arrival' } }
  const memory = rememberChatStream(undefined, activity)
  const monitor = { instance, instanceId: instance.id, instances: [instance], ready: true, payload: null, sampleError: '', memory, gpus: [],
    generation: monitorGenerationView(null, memory, instance, false, [activity]) } as unknown as MonitorState
  const state = { settings: {}, hardware: {}, instances: [instance] } as AppState
  const first = miniSnapshot(state, monitor, true)
  expect(first.metrics.current.value).toBe('150.00')
  expect(first.metrics.generated.value).toBe('30')
  expect(first.metrics.context.value).toBe('1,030 / —')
  record.status = 'completed'; record.usage = { prompt_tokens: 1000, completion_tokens: 100, ttft_ms: 500, generation_ms: 2000, decode_tps: 50 }
  activity.phase = 'completed'
  monitor.generation = monitorGenerationView(null, memory, instance, false, [activity])
  const completed = miniSnapshot(state, monitor, true)
  expect(completed.metrics.average).toMatchObject({ value: '50.00', basis: '引擎平均' })
  expect(completed.metrics.first_output).toMatchObject({ value: '0.5', basis: '引擎' })
  expect(completed.metrics.generated).toMatchObject({ value: '100', basis: '上次' })
})

test('解码窗口只包含最近三秒计数增量，零增量保持真实零，稀疏报告不冒充实时', () => {
  let state: StreamDecodeState | undefined
  for (const [tokens, ms] of [[10, 0], [110, 1000], [210, 2000], [810, 3000], [1010, 4000]]) state = sampleStreamDecode(state, 'call', pair(tokens, ms), ms + 10000)
  expect(state!.sample).toMatchObject({ tokensPerSecond: 300, windowSeconds: 3 })
  expect(state!.counters[0]).toEqual({ tokens: 110, milliseconds: 1000 })
  state = sampleStreamDecode(state, 'call', pair(1010, 7000), 17000)
  expect(state!.sample).toMatchObject({ tokensPerSecond: 0, windowSeconds: 3 })
  state = sampleStreamDecode(state, 'call', pair(5000, 10001), 20001)
  expect(state!.sample.tokensPerSecond).toBeNull()
})

test('新请求、倒退计数、来源变化重建基线；缺失计时不会拼上旧计时', () => {
  const first = sampleStreamDecode(undefined, 'one', pair(1, 0), 1)!
  const speed = sampleStreamDecode(first, 'one', pair(11, 100), 2)!
  expect(sampleStreamDecode(speed, 'two', pair(30, 300), 3)!.sample.tokensPerSecond).toBeNull()
  expect(sampleStreamDecode(speed, 'one', pair(5, 120), 4)!.sample.tokensPerSecond).toBeNull()
  expect(sampleStreamDecode(speed, 'one', pair(20, 50), 4)!.sample.tokensPerSecond).toBeNull()
  expect(sampleStreamDecode(speed, 'one', { timings: { predicted_n: 20 } }, 4)).toBe(speed)
  expect(sampleStreamDecode(speed, 'one', { timings: { predicted_ms: 200 } }, 4)).toBe(speed)
  expect(sampleStreamDecode(speed, 'one', { usage: { completion_tokens: 20 }, timings: { predicted_ms: 200 } }, 4)!.sample.tokensPerSecond).toBeNull()
  const unchangedClock = sampleStreamDecode(speed, 'one', pair(15, 100), 5)!
  expect(unchangedClock.sample).toEqual(speed.sample)
  expect(sampleStreamDecode(unchangedClock, 'one', pair(25, 200), 6)!.sample.tokensPerSecond).toBe(120)
})

test('Footer运行中取当前调用，结束取末次；未知不会借前次统计或改写整轮总量', () => {
  const first = request('one', { status: 'completed', usage: { prompt_tokens: 100, completion_tokens: 10, prompt_tokens_details: { cached_tokens: 20 } } })
  const next = request('two', { index: 2 })
  const requests = [first, next]
  expect(footerRequest(requests, true)).toBe(next)
  expect(requestTokens(footerRequest(requests, true)!)).toMatchObject({ uncached: undefined, cached: undefined, output: undefined })
  next.status = 'completed'; next.usage = { prompt_tokens: 200, completion_tokens: 30, prompt_tokens_details: { cached_tokens: 50 } }
  expect(requestTokens(footerRequest(requests, false)!)).toMatchObject({ uncached: 150, cached: 50, output: 30 })
  expect(turnTotals(requests, 0, 10, 'completed')).toMatchObject({ input: { value: 300 }, cached: { value: 70 }, output: { value: 40 } })
  expect(footerRequest([], true)).toBeUndefined()
})

test('无SSE解码计时仍保留原遥测速度，严格匹配call而非整轮id', () => {
  const record = request('call', { usage: { completion_tokens: 25 } }), backend = live()
  expect(chatRequestLive(backend, record)).toMatchObject({ tokens_per_second: 40, source: 'existing backend sampler', generated_tokens: 25 })
  const sample = { requestId: record.id, sampledAt: 1010, tokensPerSecond: 100, windowSeconds: .1 }
  expect(chatRequestLive(null, record, sample)).toMatchObject({ tokens_per_second: 100, generated_tokens: 25, available_for_chat: true })
  const other = request('other')
  expect(matchingRequestLive(backend, other)).toBeNull()
  expect(chatRequestLive(backend, other, sample)).toBeNull()
  expect(matchingRequestLive(live({ client_call_id: null }), { ...record, requestKey: 'server-call' })).not.toBeNull()
  expect(matchingRequestLive(live({ client_call_id: 'different' }), { ...record, requestKey: 'server-call' })).toBeNull()
})

test('监控接收SSE事件即更新独立请求曲线，轮询及下一请求不覆盖或混合它', () => {
  const instance = { id: 'instance', state: 'ready', active_requests: 1 } as RuntimeInstance
  const record = request('call', { timings: { predicted_n: 11, predicted_ms: 100 }, usage: { prompt_tokens: 100, completion_tokens: 11 } })
  const activity: ChatGenerationActivity = { instanceId: instance.id, messageId: 'turn', phase: 'generating', requests: [record],
    turn: turnTotals([record], 1000, 1), stream: { requestId: 'call', sampledAt: 1000, tokensPerSecond: null, windowSeconds: null } }
  let memory = rememberChatStream(undefined, activity)
  activity.stream = { requestId: 'call', sampledAt: 1010, tokensPerSecond: 100, windowSeconds: .1 }
  memory = rememberChatStream(memory, activity)
  let view = monitorGenerationView(null, memory, instance, false, [activity])
  expect(view).toMatchObject({ current: 100, active: true, phase: 'decode', status: '正在生成' })
  expect(view.requests[0].phases.decode).toMatchObject({ current: 100, mean: 110, ended: false })
  expect(view.requests[0].phases.decode.points.map(point => point.values.current)).toEqual([null, 100])
  expect(view.turn?.decode.value).toBe(110) // Cumulative weighted DEC is unchanged.
  const telemetry: Telemetry = { instance_id: instance.id, backend: 'ninfer', state: 'ready', updated_at: 2, groups: [], unavailable: [], live_generation: live() }
  memory = rememberGeneration(memory, telemetry)
  expect(monitorGenerationView(telemetry, memory, instance, false, [activity]).requests[0].phases.decode.current).toBe(100)
  record.status = 'completed'
  const next = request('next', { index: 2 })
  activity.requests = [record, next]; activity.stream = undefined
  memory = rememberChatStream(memory, activity)
  view = monitorGenerationView(telemetry, memory, instance, false, [activity])
  expect(view.requests[0].phases.decode).toMatchObject({ current: 100, ended: true })
  expect(view.requests[1].phases.decode).toMatchObject({ current: null, mean: null, points: [] })
})
