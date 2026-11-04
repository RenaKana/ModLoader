import { expect, test } from '@playwright/test'
import type { Message, ToolCall } from '../src/chat-history'
import { describeToolActivity, formatToolDetail, messageActivity, normalizeChatActivity, toolResultStatus, type ChatActivityEvent } from '../src/chat-activity'

const call = (name: string, args: unknown = {}, id = 'same-call'): ToolCall => ({ id, type: 'function', function: { name, arguments: JSON.stringify(args) } })
const tool = (name: string, args: unknown, result?: unknown, status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' = 'completed'): Extract<ChatActivityEvent, { kind: 'tool' }> => ({ id: `event-${name}`, kind: 'tool', call: call(name, args), status, ...(result === undefined ? {} : { result: JSON.stringify(result) }) })
const message: Message = { id: 'message', role: 'assistant', text: '最终答案', images: [], createdAt: 1, status: 'completed' }

test('activity 归一保留事件顺序、重复 call id 与完整参数和结果，不限制长度', () => {
  const content = '完整内容\n'.repeat(25_000)
  const events: ChatActivityEvent[] = [
    { id: 'reason-1', kind: 'reasoning', text: '先检索' },
    { ...tool('write_file', { path: 'a.txt', content }, { written: true, content }), id: 'first' },
    { id: 'text-1', kind: 'text', text: '继续读取' },
    { ...tool('read_file', { path: 'a.txt' }, { content }), id: 'second' },
    { id: 'reason-2', kind: 'reasoning', text: '完成核对' },
  ]
  expect(normalizeChatActivity(events)).toEqual(events)
  expect(normalizeChatActivity([null, { id: 'bad', kind: 'tool', status: 'done', call: call('calculate') }, ...events]).map(event => event.id)).toEqual(events.map(event => event.id))
  expect(normalizeChatActivity(null)).toEqual([])
})

test('新时间线优先；旧记录只按思考后工具兼容展示，不重复最终答案或已完成请求', () => {
  const activity = [tool('calculate', { expression: '6*7' }, { result: 42 })]
  expect(messageActivity({ ...message, activity, reasoning: '不应重复', toolResults: [{ call: call('read_file'), content: '{}' }] })).toBe(activity)
  const result = call('calculate', { expression: '6*7' })
  const legacy = messageActivity({ ...message, text: '工具请求说明', reasoningHistory: '先前思考', reasoning: '最终思考', toolResults: [{ call: result, content: '{"result":42}' }], tools: [result, call('custom_tool', { query: '待处理内容' }, 'unknown')] })
  expect(legacy.map(event => event.kind)).toEqual(['reasoning', 'tool', 'text', 'tool'])
  expect(legacy[0]).toMatchObject({ text: '先前思考\n\n最终思考' })
  expect(legacy[2]).toMatchObject({ text: '工具请求说明' })
  expect(legacy[3]).toMatchObject({ status: 'pending', call: { function: { name: 'custom_tool' } } })
  expect(JSON.stringify(legacy)).not.toContain(message.text)
  expect(messageActivity(message)).toEqual([])
  for (const status of ['stopped', 'interrupted', 'failed'] as const) {
    expect(messageActivity({ ...message, text: '', status, tools: [call('custom_tool')] })[0]).toMatchObject({ status: status === 'failed' ? 'failed' : 'cancelled' })
  }
})

test('compact 摘要使用查询、路径、表达式及实际结果项数和行范围', () => {
  expect(describeToolActivity(tool('search_web', { query: 'Qwen' }, { sources: [1, 2] }))).toMatchObject({ action: '搜索网页', target: 'Qwen', outcome: '2 条结果' })
  expect(describeToolActivity(tool('search_images', { query: '鲸鱼' }, { images: [] }))).toMatchObject({ action: '关键词搜图', outcome: '0 张图片' })
  expect(describeToolActivity(tool('reverse_image_search', { image_id: 'image-1' }, { image_name: '插画.png', matches: [{ similarity: 92 }] }))).toMatchObject({ action: '以图搜源', target: '插画.png', outcome: '1 条匹配' })
  expect(describeToolActivity(tool('list_directory', {}, { entries: [1, 2], total_entries: 205 }))).toMatchObject({ target: '.', outcome: '2 项 · 共 205 项' })
  expect(describeToolActivity(tool('read_page', { url: 'https://example.com/docs' }, { start_line: 81, end_line: 160, total_lines: 305 }))).toMatchObject({ target: 'https://example.com/docs', outcome: '第 81–160 行' })
  expect(describeToolActivity(tool('read_file', { path: 'empty.txt' }, { start_line: 1, end_line: 0 }))).toMatchObject({ target: 'empty.txt', outcome: '空内容' })
  expect(describeToolActivity(tool('calculate', { expression: '6*7' }, { result: 42 }))).toMatchObject({ action: '计算', target: '6*7', outcome: '= 42' })
  expect(describeToolActivity(tool('current_time', { time_zone: 'Asia/Shanghai' }, { utc: '2026-10-27T00:00:00Z', local: '2026-10-27 08:00:00' }))).toMatchObject({ target: 'Asia/Shanghai', outcome: '2026-10-27 08:00:00' })
})

test('被拒写入不是绿色成功；工具错误直接返回错误短句，未知请求保留 pending', () => {
  const rejected = tool('write_file', { path: 'note.txt', content: '完整内容' }, { written: false, error: '用户拒绝了这次写入。不要重复请求同一变更。' })
  expect(describeToolActivity(rejected)).toMatchObject({ status: 'cancelled', outcome: '已拒绝', error: '' })
  expect(toolResultStatus(rejected.result!)).toBe('cancelled')
  expect(describeToolActivity(tool('read_file', { path: 'missing.txt' }, { error: { message: '文件不存在。' } }))).toMatchObject({ status: 'failed', error: '文件不存在。' })
  expect(describeToolActivity(tool('custom_tool', { token: '不显示此参数' }, undefined, 'pending'))).toMatchObject({ action: 'custom_tool', target: '', outcome: '', status: 'pending' })
  expect(describeToolActivity(tool('write_file', { path: 'note.txt' }, { written: true }))).toMatchObject({ status: 'completed', outcome: '已写入' })
})

test('详情展示保留非法 JSON 和普通文本，JSON 仅格式化不改数据', () => {
  expect(formatToolDetail('{"invalid":')).toBe('{"invalid":')
  expect(formatToolDetail('普通工具结果')).toBe('普通工具结果')
  expect(JSON.parse(formatToolDetail('{"result":[1,"中文",false]}'))).toEqual({ result: [1, '中文', false] })
})

test('网页阅读摘要与顺位切换的简洁状态', () => {
  const routing = { provider: 'openai', label: 'OpenAI', attempts: [{ provider: 'deepseek', label: 'DeepSeek', error: '请求超时' }, { provider: 'openai', label: 'OpenAI' }] }
  const result = describeToolActivity(tool('read_page', { url: 'https://example.com' }, { kind: 'summary', routing }))
  expect(result.outcome).toBe('阅读摘要 · DeepSeek → OpenAI')
  expect(result.status).toBe('completed')
  expect(result.error).toBe('')
})
