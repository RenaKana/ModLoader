import { expect, test } from '@playwright/test'
import { mergeRequestReport, modelRequestLabels, requestTokens, requestRate, turnTotals, toolMetricLabel, uncachedInputTotal, type ModelRequestRecord } from '../src/chat-turn-metrics'
import { monitorGenerationView } from '../src/monitor-generation'
import { createActivityRecorder } from '../src/chat-activity-recorder'

const request = (index: number, extra: Partial<ModelRequestRecord> = {}): ModelRequestRecord => ({ id: `call-${index}`, index, kind: 'generation', status: 'completed', startedAt: 1000 * index, labels: [], ...extra })

test('整轮累加实际各请求输入，缓存和思考不重加，Decode按token与对应时间加权', () => {
  const requests = [request(1, { usage: { prompt_tokens: 100, completion_tokens: 10, prompt_tokens_details: { cached_tokens: 50 }, completion_tokens_details: { reasoning_tokens: 4 } }, timings: { predicted_n: 10, predicted_ms: 100, prompt_n: 50, prompt_ms: 10, cache_n: 50 } }),
    request(2, { usage: { prompt_tokens: 140, completion_tokens: 20, prompt_tokens_details: { cached_tokens: 50 }, completion_tokens_details: { reasoning_tokens: 8 } }, timings: { predicted_n: 20, predicted_ms: 400, prompt_n: 90, prompt_ms: 30, cache_n: 50 } })]
  const totals = turnTotals(requests, 1000, 10, 'completed', 11000)
  expect(totals.input).toEqual({ value: 240, reported: 2 })
  expect(uncachedInputTotal(requests)).toEqual({ value: 140, reported: 2 })
  expect(totals.output).toEqual({ value: 30, reported: 2 })
  expect(totals.cached).toEqual({ value: 100, reported: 2 })
  expect(totals.reasoning.value).toBe(12)
  expect(totals.decode).toEqual({ value: 60, tokens: 30, seconds: .5, reported: 2 })
  expect(totals.prefill.value).toBe(3500)
  expect(totals.e2e).toBe(3)
})

test('SSE最终累计usage只覆盖同次请求，缺报留空并明确部分', () => {
  const first = request(1, { usage: { prompt_tokens: 100, completion_tokens: 3 } })
  const final = mergeRequestReport(first, { request_key: 'server-1', client_call_id: first.id, status: 'completed', duration_s: 1, first_output_s: .2, usage: { prompt_tokens: 100, completion_tokens: 10 }, timings: { predicted_n: 10, predicted_ms: 100 } })
  const totals = turnTotals([final, request(2)], 0, 5, 'failed')
  expect(totals.input).toEqual({ value: 100, reported: 1 })
  expect(totals.output).toEqual({ value: 10, reported: 1 })
  expect(totals.cached).toEqual({ value: null, reported: 0 })
  expect(uncachedInputTotal([final, request(2)])).toEqual({ value: null, reported: 0 })
  expect(totals.decode).toMatchObject({ value: 100, reported: 1 })
  expect(totals.e2e).toBeNull()
  expect(final).toMatchObject({ firstOutput: .2, firstOutputBasis: 'gateway', requestKey: 'server-1' })
})

test('未缓存输入逐请求配对，不把不同覆盖范围的总输入与缓存直接相减', () => {
  const requests = [request(1, { usage: { prompt_tokens: 100 } }),
    request(2, { usage: { prompt_tokens_details: { cached_tokens: 25 } } }),
    request(3, { usage: { prompt_tokens: 40, prompt_tokens_details: { cached_tokens: 10 } } })]
  const totals = turnTotals(requests, 0, 5, 'completed')
  expect(totals.input).toEqual({ value: 140, reported: 2 })
  expect(totals.cached).toEqual({ value: 35, reported: 2 })
  expect(uncachedInputTotal(requests)).toEqual({ value: 30, reported: 1 })
  expect(requestTokens(requests[0]).uncached).toBeUndefined()
  expect(requestTokens(requests[1]).uncached).toBeUndefined()
})

test('已报告处理量可独立提供未缓存输入，不改写原有总输入口径', () => {
  const requests = [request(1, { usage: { prompt_tokens: 100 }, timings: { prompt_n: 60 } }),
    request(2, { prefill: { prefill_processed_tokens: 20 } }),
    request(3, { timings: { prompt_n: 10, cache_n: 5 } })]
  expect(uncachedInputTotal(requests)).toEqual({ value: 90, reported: 3 })
  expect(requestTokens(requests[0])).toMatchObject({ input: 100, uncached: 60, cached: undefined })
  expect(requestTokens(requests[1])).toMatchObject({ input: undefined, uncached: 20, cached: undefined })
  expect(requestTokens(requests[2])).toMatchObject({ input: 15, uncached: 10, cached: 5 })
  expect(turnTotals(requests, 0, 5).input).toEqual({ value: 115, reported: 2 })
})

test('未缓存输入保留真实零与大数，缓存超过总输入时留未知而非夹成零', () => {
  const invalid = request(1, { usage: { prompt_tokens: 10, prompt_tokens_details: { cached_tokens: 20 } }, timings: { prompt_n: 3 } })
  expect(requestTokens(invalid).uncached).toBeUndefined()
  expect(uncachedInputTotal([invalid])).toEqual({ value: null, reported: 0 })
  const requests = [invalid, request(2, { usage: { prompt_tokens: 0, prompt_tokens_details: { cached_tokens: 0 } } }),
    request(3, { usage: { prompt_tokens: 1_234_567_890, prompt_tokens_details: { cached_tokens: 90 } } })]
  expect(uncachedInputTotal(requests)).toEqual({ value: 1_234_567_800, reported: 2 })
  expect(requestTokens(requests[1]).uncached).toBe(0)
  expect(uncachedInputTotal([])).toEqual({ value: null, reported: 0 })
})

test('只有后端解码平均速度与计数可推对应时间，E2E不会冒充Decode', () => {
  const noTiming = request(1, { elapsed: 5, usage: { prompt_tokens: 100, completion_tokens: 20 } })
  expect(requestRate(noTiming, 'decode').value).toBeUndefined()
  expect(turnTotals([noTiming], 0, 5, 'completed').decode.value).toBeNull()
  const timed = request(2, { usage: { completion_tokens: 20 }, timings: { predicted_per_second: 100 } })
  expect(requestRate(timed, 'decode')).toEqual({ value: 100, tokens: 20, seconds: .2 })
  expect(requestTokens(request(3, { timings: { prompt_n: 10 } })).input).toBeUndefined()
  expect(requestTokens(request(4, { timings: { prompt_n: 10, cache_n: 0, predicted_n: 0 } }))).toMatchObject({ input: 10, cached: 0, output: 0 })
  const zeroDuration = request(5, { timings: { predicted_n: 5, predicted_ms: 0 } })
  expect(requestRate(zeroDuration, 'decode').seconds).toBeUndefined()
  expect(turnTotals([timed, zeroDuration], 0, 5).decode).toMatchObject({ value: 100, reported: 1 })
  expect(requestRate(request(6, { prefill: { prefill_processed_tokens: 50, average_prefill_tokens_per_second: 100, prefill_average_basis: 'sampled_prefill' } }), 'prefill').value).toBeUndefined()
})

test('无轮询的规划、压缩、短生成均入轮次，未知均值不填零，历史可独立查看', () => {
  const requests = [request(1, { kind: 'compaction', timings: { predicted_n: 5, predicted_ms: 100 } }), request(2, { kind: 'planning', usage: { prompt_tokens: 50, completion_tokens: 3 } }), request(3, { status: 'cancelled', usage: { prompt_tokens: 80, completion_tokens: 4 }, timings: { predicted_n: 4, predicted_ms: 100 } })]
  const turn = turnTotals(requests, 1000, 15, 'cancelled', 16000)
  const view = monitorGenerationView(null, undefined, undefined, false, [], { instanceId: '', messageId: 'history', phase: 'cancelled', requests, turn })
  expect(view.requests.map(item => item.decodeMean)).toEqual([50, null, 40])
  expect(view.requests.every(item => item.phases.decode.points.length === 0)).toBe(true)
  expect(view.turn?.requests).toBe(3)
  expect(view.status).toBe('已中止')
})

test('工具结果输入不重放前次动作标签，引用类型仍由真实结果细化', () => {
  const calls = [{ id: 'attach', function: { name: 'attach_file', arguments: '{}' } }, { id: 'read', function: { name: 'read_file', arguments: '{}' } }]
  const input = [{ role: 'user', content: '请查看文件' }]
  expect(modelRequestLabels(input, calls)).toEqual(['引用本地资源', '读取文件'])
  const messages = [{ role: 'user', content: '请查看文件' }, { role: 'assistant', tool_calls: [{ id: 'attach', function: { name: 'attach_file', arguments: '{}' } }] }, { role: 'tool', tool_call_id: 'attach', content: JSON.stringify({ resource: { kind: 'image' } }) }]
  expect(modelRequestLabels(messages)).toEqual(['工具结果'])
  const nextInput = [...input, { role: 'assistant', tool_calls: calls }, { role: 'tool', tool_call_id: 'attach', content: JSON.stringify({ resource: { kind: 'image' } }) }, { role: 'tool', tool_call_id: 'read', content: '文件内容' }]
  expect(modelRequestLabels(nextInput)).toEqual(['工具结果'])
  expect(modelRequestLabels(nextInput, [{ function: { name: 'calculate', arguments: '{}' } }])).toEqual(['计算', '工具结果'])
  expect(toolMetricLabel('attach_file', { resource: { kind: 'image' } })).toBe('引用图片')
  expect(toolMetricLabel('attach_file', { resource: { kind: 'file' } })).toBe('引用文件')
  expect(modelRequestLabels([{ role: 'user', content: [{ type: 'image_url', image_url: { url: 'data:image/png;base64,fixture' } }] }], [{ function: { name: 'read_file', arguments: '{}' } }])).toEqual(['读取文件', '图片输入'])
})

test('工具按所属请求保留真实状态和耗时，不计成模型Decode', () => {
  const recorder = createActivityRecorder(), call = { id: 'calc', function: { name: 'calculate', arguments: '{"expression":"6*7"}' } }
  recorder.startPass('call-7'); recorder.capture('', '', [call]); recorder.startTool(call); recorder.finishTool(call, { result: 42 })
  expect(recorder.snapshot()[0]).toMatchObject({ kind: 'tool', requestId: 'call-7', status: 'completed', startedAt: expect.any(Number), endedAt: expect.any(Number), elapsed: expect.any(Number) })
  const requests = Array.from({ length: 257 }, (_, index) => request(index + 1, { usage: { prompt_tokens: 10, completion_tokens: 1 } }))
  expect(turnTotals(requests, 0, 500, 'completed').input.value).toBe(2570)
})
