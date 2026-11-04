import { test, expect } from '@playwright/test'
import { createActivityRecorder } from '../src/chat-activity-recorder'
import { conversationMarkdown } from '../src/chat-export'
import { emptyConversation, type ToolCall } from '../src/chat-history'

const call = (id: string, name = 'read_file', args = '{}'): ToolCall => ({ id, type: 'function', function: { name, arguments: args } })

test('流式思考、工具前正文、多个模型轮次按发生顺序保存且旧快照不变', () => {
  const recorder = createActivityRecorder()
  const first = call('read', 'read_file', '{"path":')
  recorder.startPass()
  recorder.capture('先', '', [])
  const early = recorder.snapshot()
  recorder.capture('先读文件。', '读取 a.txt。', [first])
  first.function.arguments += '"a.txt"}'
  recorder.capture('先读文件。', '读取 a.txt。', [first])
  recorder.startTool(first); recorder.finishTool(first, { content: '42' })
  recorder.startPass()
  recorder.capture('结果是42，再核对。', '', [])
  const repeatedId = call('read', 'read_file', '{"path":"b.txt"}')
  recorder.capture('结果是42，再核对。', '', [repeatedId])
  recorder.startTool(repeatedId); recorder.finishTool(repeatedId, { error: { message: '文件不存在' } })
  recorder.startPass(); recorder.capture('可以回复了。', '最终答案42。', [])
  const events = recorder.snapshot()
  expect(events.map(e => e.kind)).toEqual(['reasoning', 'text', 'tool', 'reasoning', 'tool', 'reasoning'])
  expect(early).toEqual([{ id: 'activity-1', kind: 'reasoning', text: '先' }])
  expect(events[2]).toMatchObject({ status: 'completed', call: { function: { arguments: '{"path":"a.txt"}' } } })
  expect(events[4]).toMatchObject({ status: 'failed' })
  expect(new Set(events.map(e => e.id)).size).toBe(events.length)
  const markdown = conversationMarkdown({ ...emptyConversation(), messages: [{ id: 'answer', role: 'assistant', text: '最终答案42。', images: [], createdAt: 0, activity: events, reasoning: '重复旧思考', toolResults: [{ call: first, content: '重复旧结果' }] }] })
  expect(markdown.indexOf('先读文件')).toBeLessThan(markdown.indexOf('读取 a.txt'))
  expect(markdown.indexOf('读取 a.txt')).toBeLessThan(markdown.indexOf('### 工具调用'))
  expect(markdown).not.toContain('重复旧')
  expect(markdown.match(/最终答案42。/g)).toHaveLength(1)
})

test('拒绝写入与停止待执行调用保留各自终态', () => {
  const recorder = createActivityRecorder(), a = call('a'), b = call('b')
  recorder.capture('', '', [a, b]); recorder.startTool(a)
  recorder.finishTool(a, { written: false, error: '用户拒绝' }); recorder.interrupt(true)
  expect(recorder.snapshot().map(event => event.kind === 'tool' && event.status)).toEqual(['cancelled', 'cancelled'])
  recorder.startPass(); const c = call('c'); recorder.startTool(c); recorder.interrupt(false)
  expect(recorder.snapshot().at(-1)).toMatchObject({ status: 'failed' })
})
