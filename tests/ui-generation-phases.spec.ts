import { test, expect } from '@playwright/test'
import type { ChatGenerationActivity, LiveGeneration, RuntimeInstance, Telemetry, TelemetryGroup } from '../src/types'
import { generationSegments, generationView, monitorGenerationView, rememberGeneration, type GenerationMemory } from '../src/monitor-generation'

const instance = { id: 'one', state: 'ready', active_requests: 1 } as RuntimeInstance
function payload(time: number, live: Partial<LiveGeneration> = {}, groups: TelemetryGroup[] = []): Telemetry {
  return { instance_id: 'one', backend: 'ninfer', state: 'ready', updated_at: time, unavailable: [], groups,
    live_generation: { request_key: 'request-a', scope: 'request', source: 'native fixture', backend_running: 1, backend_queued: 0,
      tokens_per_second: null, window_s: null, average_tokens_per_second: null, phase: 'waiting',
      available_for_chat: false, reason: null, context_capacity: 200000, ...live } }
}
function terminal(key = 'request-a', status = 'completed', basis: 'decode' | 'end_to_end' = 'decode'): TelemetryGroup {
  return { id: 'api_request', title: '最近请求', request_key: key, request_status: status, metrics: [
    { id: 'prompt_tokens', label: '输入', value: 120000, kind: 'gauge' },
    { id: 'completion_tokens', label: '输出', value: 1000, kind: 'gauge' },
    { id: 'duration_s', label: '用时', value: 30, kind: 'gauge' },
    { id: 'first_output_s', label: '首输出', value: 20, kind: 'gauge' },
    { id: 'output_speed', label: '均值', value: 125, kind: 'gauge', rate_basis: basis },
  ] }
}
const idle = (time: number, recent?: TelemetryGroup) => payload(time, { request_key: null, backend_running: 0, phase: 'idle' }, recent ? [recent] : [])
function read(memory: GenerationMemory | undefined, sample: Telemetry, active = 1) {
  const next = rememberGeneration(memory, sample)
  return { memory: next, view: monitorGenerationView(sample, next, { ...instance, active_requests: active }) }
}

test('Prefill 进入 Decode 立即换序列和量级；思考到正文保留 Decode 采样', () => {
  let state = read(undefined, payload(100, { phase: 'prefill', prefill_tokens_per_second: 6400, average_prefill_tokens_per_second: 6100,
    prefill_window_s: 1, prompt_tokens: 120000, generated_tokens: 0, request_elapsed_s: 17 }))
  expect(state.view).toMatchObject({ phase: 'prefill', current: 6400, mean: 6100, status: '上下文填充', active: true,
    historyStartedAt: 100000, request: { prompt: 120000, generated: 0, duration: 17 } })
  state = read(state.memory, payload(101, { phase: 'thinking', generated_tokens: 1 }))
  expect(state.view).toMatchObject({ phase: 'decode', current: null, mean: null, status: '正在思考', historyStartedAt: 101000 })
  expect(state.view.points.map(point => point.values.current)).toEqual([null])
  state = read(state.memory, payload(102, { phase: 'thinking', tokens_per_second: 120, average_tokens_per_second: 118, generated_tokens: 121 }))
  const decodeKey = state.view.historyKey
  state = read(state.memory, payload(103, { phase: 'output', tokens_per_second: 155, average_tokens_per_second: 127.17, generated_tokens: 276 }))
  expect(state.view.status).toBe('正在输出')
  expect(state.view.historyKey).toBe(decodeKey)
  expect(state.view.historyStartedAt).toBe(101000)
  expect(state.view.points.map(point => point.values.current)).toEqual([null, 120, 155])
  expect(Math.max(...state.view.points.map(point => point.values.current ?? 0))).toBe(155)
  expect(state.memory.session?.rates.prefill.points.map(point => point.values.current)).toEqual([6400])
})

test('完成与中止冻结最后有效瞬时速度及时间轴，最终均值不替代最后速度', () => {
  for (const status of ['completed', 'cancelled', 'failed']) {
    let state = read(undefined, payload(100, { phase: 'output', tokens_per_second: 155, average_tokens_per_second: 127.17, generated_tokens: 999 }))
    const history = state.view.points
    state = read(state.memory, idle(101, terminal('request-a', status)), 0)
    expect(state.view).toMatchObject({ current: 155, mean: 125, active: false, ended: true, currentNote: '', request: { generated: 1000, duration: 30 } })
    expect(state.view.points).toBe(history)
    state = read(state.memory, idle(900, terminal('request-a', status)), 0)
    expect(state.view.points).toBe(history)
    expect(state.view.current).toBe(155)
    // The separately consumed mini-monitor API keeps its established idle-zero contract.
    expect(generationView(idle(900), state.memory, true, 0).current).toBe(0)
  }
})

test('Prefill 中止仍保留 Prefill；没有采样不能从终态均值伪造当前速度', () => {
  let state = read(undefined, payload(100, { phase: 'prefill', prefill_tokens_per_second: 6400, average_prefill_tokens_per_second: 6200 }))
  state = read(state.memory, idle(101, { ...terminal('request-a', 'cancelled'), metrics: [{ id: 'prefill_speed', label: 'Prefill', value: 6100, kind: 'gauge', rate_basis: 'prefill' }] }), 0)
  expect(state.view).toMatchObject({ phase: 'prefill', current: 6400, mean: 6100, status: '已中止', ended: true })
  const cold = read(undefined, idle(101, terminal()), 0)
  expect(cold.view).toMatchObject({ current: null, mean: 125, ended: true })
  expect(cold.view.points).toHaveLength(0)
  const e2e = read(undefined, idle(102, terminal('request-a', 'completed', 'end_to_end')), 0)
  expect(e2e.view.mean).toBeNull()
})

test('新请求在零输出阶段清空上次读数；旧终态记录不能覆盖当前请求', () => {
  const previous = terminal()
  let state = read(undefined, idle(100, previous), 0)
  state = read(state.memory, payload(101, { request_key: 'request-b', phase: 'waiting', request_elapsed_s: .1 }, [previous]))
  expect(state.view).toMatchObject({ key: 'request-b', current: null, mean: null, phase: null, ended: false,
    historyStartedAt: null, request: { key: 'request-b', generated: null, prompt: null, duration: .1 } })
  expect(state.view.points).toHaveLength(0)
  state = read(state.memory, payload(102, { request_key: 'request-b', phase: 'prefill', prefill_tokens_per_second: 2300 }, [previous]))
  const currentPoints = state.view.points
  state = read(state.memory, payload(103, { request_key: null, backend_running: 1 }, [previous]))
  expect(state.memory.session?.key).toBe('request-b')
  expect(state.view).toMatchObject({ current: null, mean: null, request: null })
  state = read(state.memory, idle(104, previous), 0)
  expect(state.view).toMatchObject({ key: 'request-b', current: 2300, ended: true, status: '已结束' })
  expect(state.view.points).toBe(currentPoints)
})

test('不把无输出当作 Prefill，未知采样不填零；缺测保持断线', () => {
  let state = read(undefined, payload(100, { phase: 'reading', generated_tokens: 0 }))
  expect(state.view).toMatchObject({ phase: null, current: null, status: '等待响应' })
  state = read(state.memory, payload(101, { phase: 'prefill', average_prefill_tokens_per_second: 6400 }))
  expect(state.view).toMatchObject({ current: null, mean: 6400 })
  state = read(state.memory, payload(102, { phase: 'output', tokens_per_second: 100 }))
  state = read(state.memory, payload(103, { phase: 'output' }))
  state = read(state.memory, payload(104, { phase: 'output', tokens_per_second: 0 }))
  expect(generationSegments(state.view.points).map(part => part.map(point => point.value))).toEqual([[100], [0]])
  state = read(state.memory, payload(105, { phase: 'prefill', prefill_tokens_per_second: 9000 }))
  expect(state.view.phase).toBe('decode')
  expect(state.view.current).toBeNull()
  expect(state.view.points.map(point => point.values.current)).not.toContain(9000)
})

test('停止实例仍可查看该实例的末次曲线；采样失败和多请求不伪装实时读数', () => {
  const sample = payload(100, { phase: 'output', tokens_per_second: 155, average_tokens_per_second: 127.17 })
  const state = read(undefined, sample)
  expect(monitorGenerationView(null, state.memory, { ...instance, state: 'stopped', active_requests: 0 })).toMatchObject({ current: 155, ended: true, status: '已停止' })
  expect(monitorGenerationView(null, state.memory, { ...instance, state: 'loading', active_requests: 0 })).toMatchObject({ current: null, mean: null, phase: null, points: [], request: null, status: '等待实例就绪' })
  expect(monitorGenerationView(sample, state.memory, instance, true)).toMatchObject({ current: null, mean: null, status: '采样失败' })
  const multiple = payload(101, { request_key: null, backend_running: 2, phase: 'concurrent' })
  expect(monitorGenerationView(multiple, state.memory, { ...instance, active_requests: 2 })).toMatchObject({ current: null, mean: null, status: '多请求运行', request: null, points: [] })
  expect(monitorGenerationView(null, undefined, { ...instance, id: 'other' }).points).toHaveLength(0)
})

test('滚动时只保留当前阶段最近60秒；已结束后不随下一次轮询清空', () => {
  let memory: GenerationMemory | undefined
  for (let second = 0; second <= 70; second++) memory = read(memory, payload(100 + second, { phase: 'output', tokens_per_second: 120 + second / 10 })).memory
  const ended = read(memory, idle(171, terminal()), 0)
  expect(ended.view.points[0].time).toBe(110000)
  expect(ended.view.points.at(-1)?.time).toBe(170000)
  expect(ended.view.historyStartedAt).toBe(100000)
  expect(read(ended.memory, idle(600, terminal()), 0).view.points).toBe(ended.view.points)
})

test('轮询抖动裁掉首样本后仍保留阶段起点；结束后不移动窗口', () => {
  let state = read(undefined, payload(100, { phase: 'output', tokens_per_second: 120 }))
  for (const time of [101, 102, 103, 159, 160, 160.02, 161, 161.02, 162]) {
    state = read(state.memory, payload(time, { phase: 'output', tokens_per_second: 125 }))
    expect(state.view.historyStartedAt).toBe(100000)
    expect(state.view.points.every(point => point.time >= time * 1000 - 60000)).toBe(true)
  }
  expect(state.view.points[0].time).toBe(102000)
  const ended = read(state.memory, idle(200, terminal()), 0)
  expect(ended.view.historyStartedAt).toBe(100000)
  expect(ended.view.points).toBe(state.view.points)
})

test('同一轮保留各次调用，实时序列不跨工具等待连接', () => {
  const client_request_id = 'assistant-turn-a'
  let state = read(undefined, payload(100, { client_request_id, phase: 'output', tokens_per_second: 150 }))
  state = read(state.memory, payload(101, { client_request_id, phase: 'output', tokens_per_second: 160 }))
  const firstPoints = state.view.points
  const done = { ...terminal(), client_request_id }
  state = read(state.memory, idle(102, done), 0)
  state = read(state.memory, payload(103, { request_key: 'request-b', client_request_id, phase: 'waiting' }, [done]))
  expect(state.view).toMatchObject({ key: 'request-b', phase: null, current: null, mean: null, status: '等待响应' })
  expect(state.view.points).toEqual([])
  expect(state.view.requests[0].phases.decode.points).toBe(firstPoints)
  state = read(state.memory, payload(104, { request_key: 'request-b', client_request_id, phase: 'output', tokens_per_second: 170 }))
  state = read(state.memory, payload(105, { request_key: 'request-b', client_request_id, phase: 'output', tokens_per_second: 180 }))
  expect(generationSegments(state.view.points).map(part => part.map(point => point.value))).toEqual([[170, 180]])
  expect(state.view.points.map(point => point.time)).toEqual([104000, 105000])
  expect(state.view.requests).toHaveLength(2)
  state = read(state.memory, idle(106, { ...terminal('request-b'), client_request_id }), 0)
  expect(state.view.current).toBe(180)
  expect(state.view.mean).toBe(125)
  expect(state.view.requests[0].phases.decode.points).toBe(firstPoints)
  state = read(state.memory, payload(107, { request_key: 'request-c', client_request_id: 'assistant-turn-b' }))
  expect(state.view.requests).toHaveLength(1)
  expect(state.view.points).toEqual([])
})

test('错过轮询的短请求有最终均值，既不丢失旧调用也不伪造实时样本', () => {
  const client_request_id = 'assistant-turn-a'
  let state = read(undefined, payload(100, { client_request_id, phase: 'output', tokens_per_second: 150 }))
  const firstPoints = state.view.points
  state = read(state.memory, idle(105, { ...terminal('short-call'), client_request_id }), 0)
  expect(state.view.points).toEqual([])
  expect(state.view).toMatchObject({ current: null, mean: 125, status: '已完成', request: { key: 'short-call' } })
  expect(state.view.requests).toHaveLength(2)
  expect(state.view.requests[0].phases.decode.points).toBe(firstPoints)
  expect(state.view.requests[1]).toMatchObject({ decodeMean: 125, phases: { decode: { points: [] } } })
  state = read(state.memory, payload(200, { request_key: 'request-c', client_request_id, phase: 'output', generated_tokens: 1 }))
  expect(state.view.points.map(point => point.values.current)).toEqual([null])
  state = read(state.memory, payload(201, { request_key: 'request-c', client_request_id, phase: 'output', tokens_per_second: 190 }))
  expect(state.view.points.map(point => point.values.current)).toEqual([null, 190])
  expect(state.view.points.map(point => point.time)).toEqual([200000, 201000])
  expect(state.view.requests).toHaveLength(3)
})

test('后续请求的 Prefill 和 Decode 独立，之前请求仍可选择', () => {
  const client_request_id = 'assistant-turn-a'
  let state = read(undefined, payload(100, { client_request_id, phase: 'prefill', prefill_tokens_per_second: 6000 }))
  state = read(state.memory, payload(101, { client_request_id, phase: 'output', tokens_per_second: 150 }))
  const decodePoints = state.view.points
  state = read(state.memory, payload(102, { request_key: 'request-b', client_request_id, phase: 'waiting' }))
  expect(state.view.points).toEqual([])
  state = read(state.memory, payload(103, { request_key: 'request-b', client_request_id, phase: 'prefill', prefill_tokens_per_second: 7000 }))
  expect(state.view).toMatchObject({ phase: 'prefill', current: 7000 })
  expect(generationSegments(state.view.points).map(part => part.map(point => point.value))).toEqual([[7000]])
  state = read(state.memory, payload(104, { request_key: 'request-b', client_request_id, phase: 'thinking', tokens_per_second: 170 }))
  expect(state.view.points.map(point => point.values.current)).toEqual([170])
  expect(state.view.requests[0].phases.decode.points).toBe(decodePoints)
})

test('阶段视图保留各自速度和均值，右侧请求统计不随查看阶段变化', () => {
  const client_request_id = 'phase-switch-turn'
  let state = read(undefined, payload(100, { client_request_id, phase: 'prefill', prefill_tokens_per_second: 6400,
    average_prefill_tokens_per_second: 6100, prefill_window_s: 3, prompt_tokens: 120000 }))
  const sessionKey = state.view.sessionKey
  state = read(state.memory, payload(101, { client_request_id, phase: 'output', tokens_per_second: 150,
    average_tokens_per_second: 145, window_s: 3, generated_tokens: 200, request_elapsed_s: 2 }))
  expect(state.view.sessionKey).toBe(sessionKey)
  expect(state.view.phases.prefill).toMatchObject({ current: 6400, mean: 6100, ended: true, currentNote: '' })
  expect(state.view.phases.decode).toMatchObject({ current: 150, mean: 145, ended: false, currentNote: '近 3 秒滚动' })
  expect(state.view.request).toMatchObject({ prompt: 120000, generated: 200, duration: 2 })
  state = read(state.memory, payload(105, { client_request_id, request_key: 'request-b', phase: 'waiting' }))
  expect(state.view.sessionKey).toBe(sessionKey)
  state = read(state.memory, payload(106, { client_request_id: 'new-turn', request_key: 'request-c', phase: 'waiting' }))
  expect(state.view.sessionKey).not.toBe(sessionKey)
  expect(state.view.phases.prefill.points).toEqual([])
  expect(state.view.phases.decode.points).toEqual([])
})

test('长工具等待保留旧请求采样，新请求使用真实时间轴及缺测断线', () => {
  const client_request_id = 'long-web-turn'
  let state = read(undefined, payload(100, { client_request_id, phase: 'output', tokens_per_second: 150 }))
  for (let time = 101; time <= 130; time++) state = read(state.memory, payload(time, { client_request_id, phase: 'output', tokens_per_second: 160 }))
  const done = { ...terminal(), client_request_id }
  state = read(state.memory, idle(131, done), 0)
  const frozen = state.view.points
  state = read(state.memory, idle(800, done), 0)
  expect(state.view.points).toBe(frozen)
  state = read(state.memory, payload(900, { client_request_id, request_key: 'request-b', phase: 'output', tokens_per_second: 170 }))
  expect(state.view.points).toHaveLength(1)
  expect(state.view.points[0]).toMatchObject({ time: 900000, wallTime: 900000, values: { current: 170 } })
  expect(state.view.requests[0].phases.decode.points).toBe(frozen)
  const repeated = read(state.memory, payload(900, { client_request_id, request_key: 'request-b', phase: 'output', tokens_per_second: 170 }))
  expect(repeated.view.points).toBe(state.view.points)
  state = read(state.memory, payload(905, { client_request_id, request_key: 'request-b', phase: 'output', tokens_per_second: 180 }))
  expect(state.view.points.at(-1)).toMatchObject({ time: 905000, wallTime: 905000 })
  expect(generationSegments(state.view.points)).toHaveLength(2)
  expect(state.view.historyStartedAt).toBe(900000)
})

test('缺测不填零，大幅升降保留数值但不拼接不同请求', () => {
  const client_request_id = 'warmup-turn'
  let state = read(undefined, payload(100, { client_request_id, phase: 'output', tokens_per_second: 10 }))
  state = read(state.memory, payload(101, { client_request_id, phase: 'output', tokens_per_second: 15 }))
  for (const time of [300, 301, 305, 310]) state = read(state.memory, payload(time, { client_request_id, request_key: 'high', phase: 'output', generated_tokens: 1 }))
  expect(state.view.points.map(point => point.values.current)).toEqual([null, null, null, null])
  state = read(state.memory, payload(311, { client_request_id, request_key: 'high', phase: 'output', tokens_per_second: 1500 }))
  state = read(state.memory, payload(600, { client_request_id, request_key: 'low', phase: 'output', tokens_per_second: 5 }))
  expect(state.view.points.map(point => point.time)).toEqual([600000])
  expect(state.view.requests.map(request => request.phases.decode.points.map(point => point.values.current))).toEqual([[10, 15], [null, null, null, null, 1500], [5]])
})

test('工具状态来自整轮聊天生命周期；工具前后不提前完成，读数冻结且不新增采样', () => {
  const client_request_id = 'tool-turn'
  const sample = payload(100, { client_request_id, phase: 'output', tokens_per_second: 155, average_tokens_per_second: 127 })
  const state = read(undefined, sample)
  const completed = idle(101, { ...terminal(), client_request_id })
  const memory = rememberGeneration(state.memory, completed)
  const activity = (phase: ChatGenerationActivity['phase']): ChatGenerationActivity[] => [{ instanceId: 'one', messageId: client_request_id, phase }]
  for (const phase of ['searching', 'reading'] as const) {
    // Phase events beat a lagging live telemetry sample, as well as its terminal record.
    expect(monitorGenerationView(sample, state.memory, instance, false, activity(phase))).toMatchObject({ status: '调用工具', active: false, working: true, current: 155, ended: true })
    const view = monitorGenerationView(completed, memory, { ...instance, active_requests: 0 }, false, activity(phase))
    expect(view).toMatchObject({ status: '调用工具', working: true, current: 155, mean: 125 })
    expect(view.points).toBe(state.view.points)
  }
  // Instance polling can still report one request after telemetry has gone idle.
  const waiting = monitorGenerationView(completed, memory, instance, false, activity('generating'))
  expect(waiting).toMatchObject({ status: '等待响应', working: true })
  expect(waiting.points).toBe(state.view.points)
  for (const [phase, status] of [['completed', '已完成'], ['cancelled', '已中止'], ['failed', '请求失败']] as const) {
    expect(monitorGenerationView(completed, memory, instance, false, activity(phase))).toMatchObject({ status, working: false, ended: true, current: 155 })
  }
  const resumed = read(memory, payload(200, { client_request_id, request_key: 'request-b', phase: 'thinking', tokens_per_second: 170 }))
  expect(monitorGenerationView(payload(200, { client_request_id, request_key: 'request-b', phase: 'thinking', tokens_per_second: 170 }), resumed.memory, instance, false, activity('generating'))).toMatchObject({ status: '正在思考', working: true, active: true })
})

test('工具活动按实例和消息隔离，外部请求及新回复不继承旧工具状态或曲线', () => {
  const client_request_id = 'old-turn'
  const sample = payload(100, { client_request_id, phase: 'output', tokens_per_second: 150 })
  const state = read(undefined, sample)
  const unrelated: ChatGenerationActivity = { instanceId: 'other', messageId: 'tool-turn', phase: 'searching' }
  expect(monitorGenerationView(sample, state.memory, instance, false, [unrelated])).toMatchObject({ status: '正在输出', current: 150 })
  const own = { ...unrelated, instanceId: 'one' }
  expect(monitorGenerationView(sample, state.memory, instance, false, [own])).toMatchObject({ status: '正在输出', current: 150 })
  expect(monitorGenerationView(idle(101), state.memory, { ...instance, active_requests: 0 }, false, [own])).toMatchObject({ status: '调用工具', points: [], working: true })
  expect(monitorGenerationView(idle(101), state.memory, { ...instance, active_requests: 0 }, false, [own, { ...own, messageId: 'second-tool' }])).toMatchObject({ status: '多请求运行', working: true })
  expect(monitorGenerationView(null, state.memory, { ...instance, state: 'stopped' }, false, [own])).toMatchObject({ status: '已停止', working: false })
})
