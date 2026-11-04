import { test, expect } from '@playwright/test'
import { basicToolDefinitions, calculate, executeBasicTool, fileToolDefinitions } from '../src/chat-tools'
import { runWebChat, type WebCompletion, type WebModelRequest } from '../src/web-chat'
import type { ToolCall } from '../src/chat-history'

const call = (id: string, name: string, args: unknown): ToolCall => ({ id, type: 'function', function: { name, arguments: JSON.stringify(args) } })
test('基础工具执行确定性运算，不执行代码；时间支持显式时区', () => {
  expect(calculate('(12.5 + 7.5) * 3 / 2')).toBe(30)
  expect(calculate('-2**2 + 2**3**2')).toBe(508)
  expect(calculate('1e3 % 7')).toBe(6)
  for (const input of ['globalThis.process.exit()', '1/0', '2;alert(1)', '1 2', '', '(1+2']) expect(() => calculate(input)).toThrow()
  expect(executeBasicTool(call('time', 'current_time', { time_zone: 'Asia/Shanghai' }))).toMatchObject({ time_zone: 'Asia/Shanghai', utc: expect.stringMatching(/Z$/) })
  expect(() => executeBasicTool(call('time', 'current_time', { time_zone: 'not-a-zone' }))).toThrow()
})

function setup(completions: WebCompletion[]) {
  const requests: WebModelRequest[] = [], results: unknown[] = [], phases: string[] = []
  const controller = new AbortController()
  return { requests, results, phases, controller, options: {
    mode: 'agent' as const, webEnabled: false, messages: [{ role: 'user', content: '计算并写入' }], maxCalls: 3, signal: controller.signal,
    builtInTools: { definitions: basicToolDefinitions, execute: async (tool: ToolCall) => executeBasicTool(tool) },
    generate: async (request: WebModelRequest) => { requests.push(request); return completions.shift()! },
    search: async () => { throw new Error('must not use network') }, searchImages: async () => { throw new Error('must not use network') }, readPage: async () => { throw new Error('must not use network') },
    onPhase: (phase: string) => phases.push(phase), onEvidence: () => { throw new Error('must not publish web evidence') }, onToolResult: (_: ToolCall, result: unknown) => results.push(result),
  } }
}
test('关闭联网后基础工具仍完成调用、结果回传和最终回复；schema不含提交授权', async () => {
  const h = setup([{ message: { role: 'assistant', content: null, tool_calls: [call('a', 'calculate', { expression: '6*7' })] } }, { message: { role: 'assistant', content: '42' } }])
  await runWebChat(h.options)
  expect(h.requests).toHaveLength(2)
  expect(h.requests[0].tools).toEqual(basicToolDefinitions)
  expect(h.requests[1].messages.at(-1)).toEqual({ role: 'tool', tool_call_id: 'a', content: JSON.stringify({ expression: '6*7', result: 42 }) })
  expect(h.results).toEqual([{ expression: '6*7', result: 42 }])
  expect(h.phases).toEqual(['generating', 'tools', 'generating'])
  for (const tool of fileToolDefinitions) expect(Object.keys(tool.function.parameters.properties)).not.toEqual(expect.arrayContaining(['preparation_id']))
  expect(JSON.stringify(fileToolDefinitions)).not.toMatch(/preparation_id|approved/)
})
test('工具参数错误可回传，未注册工具不执行，调用预算仍有界', async () => {
  const h = setup([{ message: { role: 'assistant', content: null, tool_calls: [call('bad', 'calculate', { expression: '1/0' })] } }, { message: { role: 'assistant', content: '除数不能为零。' } }])
  await runWebChat(h.options)
  expect(h.results[0]).toMatchObject({ error: { tool: 'calculate' } })
  const unknown = setup([{ message: { role: 'assistant', content: null, tool_calls: [call('x', 'write_file', { path: 'x.txt', content: 'secret' })] } }])
  await expect(runWebChat(unknown.options)).rejects.toThrow('未执行工具')
  const budget = setup(Array.from({ length: 2 }, (_, i) => ({ message: { role: 'assistant' as const, content: null, tool_calls: [call(String(i), 'calculate', { expression: '1+1' })] } })))
  budget.options.maxCalls = 1
  await expect(runWebChat(budget.options)).rejects.toThrow('预算')
  expect(budget.results).toHaveLength(1)
})
test('停止待确认的本地工具不会继续调用模型', async () => {
  const h = setup([{ message: { role: 'assistant', content: null, tool_calls: [call('write', 'write_file', { path: 'test.txt', content: 'x' })] } }])
  let started = false
  h.options.builtInTools = { definitions: fileToolDefinitions, execute: async () => { started = true; return new Promise(() => {}) } }
  const running = runWebChat(h.options)
  await expect.poll(() => started).toBe(true)
  h.controller.abort()
  await expect(running).rejects.toMatchObject({ name: 'AbortError' })
  expect(h.requests).toHaveLength(1)
})
