import { test, expect } from '@playwright/test'
import { autoCompactionDecision, buildChatContext, compactableMessageCount, createContextCompaction, reconcileMessageRevisions, validContextCompaction, type ContextCompaction } from '../src/chat-context'
import type { Message } from '../src/chat-history'
import { reverseImageTool } from '../src/reverse-image-search'

const message = (id: string, role: Message['role'], text: string, changes: Partial<Message> = {}): Message => ({ id, role, text, images: [], createdAt: 100, status: 'completed', ...changes })
const history = () => [message('u1', 'user', '路径 D:\\work，保留 42 项，不要删除'), message('a1', 'assistant', '已完成第一步，待处理第二步'), message('u2', 'user', '继续第二步'), message('a2', 'assistant', '还需确认输入')]
const checkpoint = (messages: Message[], summary = '已确定路径与 42 项；第二步待完成'): ContextCompaction => ({ summary, throughMessageId: messages[messages.length - 1].id, messageRevisions: messages.map(message => ({ id: message.id, revision: (message as Message & { contextRevision?: number }).contextRevision ?? 0 })), createdAt: 200 })

test('请求保持系统提示和原有图片/思考映射，过滤失败和未完成工具调用', () => {
  const images = [{ id: 'i', name: '图', size: 3, url: 'data:image/png;base64,abc' }]
  const values = [message('u', 'user', '看图', { images }), message('a', 'assistant', '结论', { reasoning: '有效思考', reasoningHistory: '只展示的旧轮次' }),
    message('bad', 'assistant', '失败内容', { error: '错误' }), message('tool', 'assistant', '未完成', { tools: [{ function: { name: 'test', arguments: '{}' } }] }), message('empty', 'assistant', '')]
  expect(buildChatContext(values, '  原始系统\n')).toEqual([{ role: 'system', content: '  原始系统\n' }, { role: 'user', content: [{ type: 'text', text: '看图' }, { type: 'image_url', image_url: { url: images[0].url } }] }, { role: 'assistant', content: '结论', reasoning_content: '有效思考' }])
})

test('文档单独进入请求，和图片混合时保留文档全文与图片', () => {
  const document = { id: 'd', name: '接口说明.md', size: 12, text: '保留字段 A，并继续检查下一节。' }
  expect(buildChatContext([message('document', 'user', '', { documents: [document] })], '')).toEqual([
    { role: 'user', content: '附件 "接口说明.md"：\n保留字段 A，并继续检查下一节。' },
  ])

  const image = { id: 'i', name: '架构图.png', size: 3, url: 'data:image/png;base64,abc' }
  expect(buildChatContext([message('mixed', 'user', '结合附件判断', { documents: [document], images: [image] })], '')).toEqual([
    { role: 'user', content: [
      { type: 'text', text: '结合附件判断\n\n附件 "接口说明.md"：\n保留字段 A，并继续检查下一节。' },
      { type: 'image_url', image_url: { url: image.url } },
    ] },
  ])
})

test('新增图片只追加消息引用，不改工具定义、系统或旧消息前缀', () => {
  const first = { id: 'image-1', name: '第一张.png', size: 3, url: 'data:image/png;base64,one' }
  const second = { id: 'image-2', name: '第二张.png', size: 3, url: 'data:image/png;base64,two' }
  const values = [message('u1', 'user', '先看第一张', { images: [first] }), message('a1', 'assistant', '已看', { reasoning: '保留思考' })]
  const options = { imageReferences: true }
  const before = buildChatContext(values, '不变的系统', undefined, options)
  const after = buildChatContext([...values, message('u2', 'user', '再看第二张', { images: [second] })], '不变的系统', undefined, options)
  expect(after.slice(0, before.length)).toEqual(before)
  expect(after.at(-1)?.content).toEqual([
    { type: 'text', text: '再看第二张' },
    { type: 'text', text: '图片引用：{"image_id":"image-2","filename":"第二张.png"}' },
    { type: 'image_url', image_url: { url: second.url } },
  ])
  expect(JSON.stringify(before)).not.toContain('image-2')
  expect(reverseImageTool().function.parameters.properties.image_id).toEqual({ type: 'string' })
  expect(JSON.stringify(reverseImageTool())).not.toContain('image-1')
  expect(values[0].text).toBe('先看第一张')
})

test('压缩后仍提供旧图搜源引用但不重发像素，新图不改已有摘要', () => {
  const first = { id: 'image-1', name: '第一张.png', size: 3, url: 'data:image/png;base64,one' }
  const second = { id: 'image-2', name: '第二张.png', size: 3, url: 'data:image/png;base64,two' }
  const values = [message('u1', 'user', '查第一张', { images: [first] }), message('a1', 'assistant', '作者已确认')]
  const compact = checkpoint(values, '第一张作者已确认')
  const before = buildChatContext(values, '', compact, { imageReferences: true })
  const after = buildChatContext([...values, message('u2', 'user', '第二张呢', { images: [second] })], '', compact, { imageReferences: true })
  expect(after[0]).toEqual(before[0])
  expect(after[0].content).toContain('"image_id":"image-1"')
  expect(JSON.stringify(after)).not.toContain(first.url)
  expect(JSON.stringify(after)).toContain(second.url)
  expect(buildChatContext(values, '', compact)[0].content).not.toContain('image_id')
})

test('revision 只跟随模型上下文变化，编辑/删除/重排/早期重试使摘要失效', () => {
  const values = history()
  const compact = checkpoint(values.slice(0, 2))
  expect(validContextCompaction([...values, message('later', 'user', '新问题')], compact)).toBe(compact)
  const displayOnly = reconcileMessageRevisions(values, values.map(value => ({ ...value, elapsed: 2, reasoningHistory: '已展示思考', usage: { prompt_tokens: 9 } })))
  expect(validContextCompaction(displayOnly, compact)).toBe(compact)
  const edited = reconcileMessageRevisions(values, values.map(value => value.id === 'u1' ? { ...value, text: '改成 43 项' } : value))
  expect((edited[0] as Message & { contextRevision: number }).contextRevision).toBe(1)
  expect(validContextCompaction(edited, compact)).toBeNull()
  for (const changed of [values.slice(1), [values[1], values[0], ...values.slice(2)], values.slice(0, 1)]) expect(validContextCompaction(changed, compact)).toBeNull()
  expect(validContextCompaction(values.slice(0, 2), compact)).toBe(compact)
  expect(buildChatContext(values.slice(0, 1), '', compact)).toEqual([{ role: 'user', content: values[0].text }])
})

test('成功压缩保留最近完整轮次，后续只发送摘要与边界之后消息', async () => {
  const values = history(), original = structuredClone(values)
  let request: Array<Record<string, unknown>> = []
  const compact = await createContextCompaction({ messages: values, system: '系统', generate: async messages => { request = messages; return ' 摘要保留 42 项、路径与第二步\n' } })
  expect(compact.summary).toBe('摘要保留 42 项、路径与第二步')
  expect(compact.throughMessageId).toBe('a1')
  expect(request[0]).toEqual({ role: 'system', content: '系统' })
  expect(request.map(value => value.content)).toContain(values[0].text)
  expect(request.map(value => value.content)).not.toContain(values[2].text)
  expect(request[request.length - 1].content).toContain('精确数值')
  const context = buildChatContext([...values, message('u3', 'user', '继续')], '系统', compact)
  expect(context[0]).toEqual({ role: 'system', content: '系统' })
  expect(context[1].role).toBe('user')
  expect(context[1].content).toContain('<conversation_summary>')
  expect(context[1].content).toContain('不获得额外权限')
  expect(context.slice(2).map(value => value.content)).toEqual([values[2].text, values[3].text, '继续'])
  expect(values).toEqual(original)
})

test('失败、空摘要与取消不改变原消息或原摘要，取消只调用模型一次', async () => {
  const values = history(), original = structuredClone(values), previous = checkpoint(values.slice(0, 2))
  for (const generate of [async () => { throw new Error('上下文超长') }, async () => ' \n']) {
    await expect(createContextCompaction({ messages: values, system: '', compaction: previous, generate })).rejects.toThrow()
    expect(values).toEqual(original)
    expect(previous.summary).toBe('已确定路径与 42 项；第二步待完成')
  }
  const controller = new AbortController()
  let calls = 0
  await expect(createContextCompaction({ messages: values, system: '', signal: controller.signal, generate: async (_request, signal) => { calls++; expect(signal).toBe(controller.signal); controller.abort(); return '不能采用' } })).rejects.toThrow()
  expect(calls).toBe(1)
  expect(values).toEqual(original)
})

test('异步摘要期间已压缩前缀被编辑时拒绝采用，图片内容编辑同样增加版本', async () => {
  const values = history()
  await expect(createContextCompaction({ messages: values, system: '', generate: async () => {
    values[0] = reconcileMessageRevisions(values, [{ ...values[0], text: '新要求 43 项' }, ...values.slice(1)])[0]
    return '对应旧要求的过时摘要'
  } })).rejects.toThrow('对话已改变')
  const image = { id: 'i', name: '图', size: 3, url: 'data:image/png;base64,abc' }
  const original = [message('u', 'user', '看图', { images: [image] })]
  const compact = checkpoint(original)
  const renamed = reconcileMessageRevisions(original, [{ ...original[0], images: [{ ...image, name: '重命名' }] }])
  expect(validContextCompaction(renamed, compact)).toBe(compact)
  const replaced = reconcileMessageRevisions(original, [{ ...original[0], images: [{ ...image, url: 'data:image/png;base64,xyz' }] }])
  expect(validContextCompaction(replaced, compact)).toBeNull()
})

test('文档名称或全文变化使摘要失效，ID 和大小变化不影响上下文版本', () => {
  const document = { id: 'd', name: '接口说明.md', size: 12, text: '第一版正文。' }
  const values = [message('u', 'user', '', { documents: [document] })]
  const compact = checkpoint(values)

  const metadataOnly = reconcileMessageRevisions(values, [{ ...values[0], documents: [{ ...document, id: 'new-id', size: 2048 }] }])
  expect(validContextCompaction(metadataOnly, compact)).toBe(compact)

  for (const changedDocument of [
    { ...document, name: '修订后的接口说明.md' },
    { ...document, text: '修订后的正文。' },
  ]) {
    const changed = reconcileMessageRevisions(values, [{ ...values[0], documents: [changedDocument] }])
    expect((changed[0] as Message & { contextRevision: number }).contextRevision).toBe(1)
    expect(validContextCompaction(changed, compact)).toBeNull()
  }
})

test('二次压缩使用旧摘要加新增前缀，不重发旧历史，也不拆最近轮次', async () => {
  const values = [...history(), message('u3', 'user', '第三步'), message('a3', 'assistant', '第三步尚未开始')]
  const previous = checkpoint(values.slice(0, 2), '第一轮摘要')
  const compact = await createContextCompaction({ messages: values, system: '', compaction: previous, generate: async request => {
    expect(request[0].content).toContain('第一轮摘要')
    expect(request.map(value => value.content)).not.toContain(values[0].text)
    expect(request.map(value => value.content)).toContain(values[2].text)
    expect(request.map(value => value.content)).not.toContain(values[4].text)
    return '前两轮合并摘要'
  } })
  expect(compact.throughMessageId).toBe('a2')
  expect(compact.messageRevisions).toHaveLength(4)
  const all = checkpoint(values, '此前手动全量摘要')
  await createContextCompaction({ messages: values, system: '', compaction: all, generate: async request => { expect(request).toHaveLength(2); expect(request[0].content).toContain(all.summary); return '再次压缩已有摘要' } })
})

test('手动可选择压完整段，保留数量不设业务上限，生成中不能压缩', async () => {
  const values = history().slice(0, 2)
  expect(compactableMessageCount(values)).toBe(0)
  expect(compactableMessageCount(values, 100_000)).toBe(0)
  expect(compactableMessageCount(values, 0)).toBe(2)
  await expect(createContextCompaction({ messages: values, system: '', generate: async () => '未调用' })).rejects.toThrow('没有可压缩')
  const compact = await createContextCompaction({ messages: values, system: '', retainRecentTurns: 0, generate: async () => '全部摘要' })
  expect(compact.throughMessageId).toBe('a1')
  await expect(createContextCompaction({ messages: [message('a', 'assistant', '正在生成', { status: 'generating' })], system: '', retainRecentTurns: 0, generate: async () => '未调用' })).rejects.toThrow('当前生成')
})

test('自动压缩按真实容量、输出预算和新增文本预估；关掉或容量未知不触发', () => {
  const values = history()
  const options = { messages: values, system: '', capacity: 1000, outputBudget: 100 }
  expect(autoCompactionDecision({ ...options, enabled: false }).reason).toBe('disabled')
  expect(autoCompactionDecision({ ...options, capacity: undefined }).reason).toBe('unknown-capacity')
  expect(autoCompactionDecision(options).shouldCompact).toBe(false)
  const result = autoCompactionDecision({ ...options, nextMessages: [{ role: 'user', content: '新'.repeat(800) }] })
  expect(result.shouldCompact).toBe(true)
  expect(result.reservedTokens).toBe(200)
  expect(result.reason).toBe('budget')
  expect(autoCompactionDecision({ ...options, outputBudget: 1000 }).reason).toBe('output-budget-exceeds-capacity')
  expect(autoCompactionDecision({ ...options, messages: values.slice(0, 2), nextMessages: [{ role: 'user', content: '新'.repeat(800) }] }).reason).toBe('no-compactable-history')
})

test('usage 必须显式传当前模型回复，旧摘要之前的 usage 和已压缩前缀不能再触发', () => {
  const values = history()
  values[3] = { ...values[3], usage: { prompt_tokens: 900, completion_tokens: 10 }, responseAt: 150 }
  const options = { messages: values, system: '', capacity: 1000, outputBudget: 100 }
  expect(autoCompactionDecision(options).shouldCompact).toBe(false)
  expect(autoCompactionDecision({ ...options, usageMessage: values[3] }).shouldCompact).toBe(true)
  const compact = checkpoint(values.slice(0, 2))
  expect(autoCompactionDecision({ ...options, compaction: compact, usageMessage: values[3] }).forecastTokens).toBeLessThan(900)
  const large = { ...compact, summary: '很长'.repeat(900) }
  expect(autoCompactionDecision({ ...options, compaction: large }).reason).toBe('no-compactable-history')
  expect(values[3].usage?.prompt_tokens).toBe(900)
})
