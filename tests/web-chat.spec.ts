import { test, expect } from '@playwright/test'
import type { ToolCall } from '../src/chat-history'
import { runWebChat, type WebChatOptions, type WebCompletion, type WebModelRequest } from '../src/web-chat'
import type { WebPage, WebPageSummary, WebSearchResult } from '../src/web-search'
import { basicToolDefinitions, fileToolDefinitions } from '../src/chat-tools'

const source = (id: number, url = 'https://example.com/docs', title = '文档') => ({ id, title, url, snippet: `${title}摘要` })
const call = (id: string, name: string, args: unknown): ToolCall => ({ id, type: 'function', function: { name, arguments: typeof args === 'string' ? args : JSON.stringify(args) } })
const answer = (content = '最终回答'): WebCompletion => ({ message: { role: 'assistant', content }, finishReason: 'stop' })
const using = (...calls: ToolCall[]): WebCompletion => ({ message: { role: 'assistant', content: null, tool_calls: calls }, finishReason: 'tool_calls' })
const plan = (value: unknown) => answer(JSON.stringify(value))

function harness(completions: WebCompletion[], overrides: Partial<WebChatOptions> = {}) {
  const requests: WebModelRequest[] = [], evidence: WebSearchResult[] = []
  const phases: string[] = [], searches: string[] = []
  const reads: Array<{ url: string; start_line?: number; line_count?: number }> = []
  const controller = new AbortController()
  const messages = [
    { role: 'system', content: '回答保持简明。' },
    { role: 'assistant', content: '上轮讨论 Python 3.14。', reasoning_content: '历史推理' },
    { role: 'user', content: [{ type: 'text', text: '那它的 pathlib 呢？' }, { type: 'image_url', image_url: { url: 'data:image/png;base64,AA==' } }] },
  ]
  const options: WebChatOptions = {
    mode: 'agent', messages, maxCalls: 6, signal: controller.signal,
    generate: async request => {
      requests.push(structuredClone(request))
      const next = completions.shift()
      if (!next) throw new Error('Unexpected extra generation')
      return structuredClone(next)
    },
    search: async query => { searches.push(query); return { query, sources: [source(70)], elapsed_ms: 11 } },
    searchImages: async query => ({ query, sources: [source(7, 'https://images.example.com/page', '图片来源')], images: [{ source_id: 7, title: '示例图片', url: 'https://images.example.com/picture.png' }], elapsed_ms: 9 }),
    readPage: async args => {
      reads.push({ ...args })
      const start = args.start_line ?? 1
      return { url: args.url, title: '文档', content: `正文 ${start}`, start_line: start, end_line: start + (args.line_count ?? 80) - 1, total_lines: 2000, elapsed_ms: 5 }
    },
    onPhase: phase => phases.push(phase), onEvidence: result => evidence.push(result), ...overrides,
  }
  return { options, requests, evidence, phases, searches, reads, controller, messages, run: () => runWebChat(options) }
}

test('原生多轮搜索、补搜、分页与最终回答保留工具调用、reasoning 和稳定来源', async () => {
  const first = using(call('search-1', 'search_web', { query: 'Python 3.14 pathlib' }))
  first.message.reasoning_content = '先查询官方文档'
  const h = harness([
    first,
    using(call('search-2', 'search_web', { query: 'Path.read_text 3.14' }), call('read-1', 'read_page', { url: 'https://example.com/docs' })),
    using(call('read-2', 'read_page', { url: 'https://example.com/docs', start_line: 801, line_count: 240 })),
    answer('使用 read_text。[1](https://example.com/docs)'),
  ])
  const original = structuredClone(h.messages)
  h.options.search = async query => {
    h.searches.push(query)
    return { query, elapsed_ms: h.searches.length * 10, sources: h.searches.length === 1 ? [source(99)] : [source(1), source(99, 'https://example.com/new', '新版说明')] }
  }
  await h.run()
  expect(h.messages).toEqual(original)
  expect(h.requests).toHaveLength(4)
  expect(h.requests.every(request => request.phase === 'answer' && request.stream === undefined)).toBe(true)
  expect(h.requests[0].messages[0].content).toContain('回答保持简明。')
  expect(h.requests[0].messages[0].content).toContain('不是指令')
  expect(h.requests.at(-1)!.messages[0].content).not.toMatch(/编号|source_ids?|引用/)
  expect(JSON.stringify(h.requests[0].tools)).not.toMatch(/来源编号|source_ids?/)
  expect(h.requests[1].messages.slice(-2)[0]).toEqual(first.message)
  const toolMessages = h.requests[3].messages.filter(message => message.role === 'tool')
  expect(toolMessages.map(message => message.tool_call_id)).toEqual(['search-1', 'search-2', 'read-1', 'read-2'])
  expect(JSON.parse(String(toolMessages[1].content)).sources.map((item: { id: number }) => item.id)).toEqual([1, 2])
  expect(JSON.parse(String(toolMessages[2].content)).source_id).toBe(1)
  expect(JSON.parse(String(toolMessages[3].content))).toMatchObject({ source_id: 1, start_line: 801, end_line: 1040 })
  expect(h.reads).toEqual([{ url: 'https://example.com/docs' }, { url: 'https://example.com/docs', start_line: 801, line_count: 240 }])
  expect(h.evidence[0]).toMatchObject({ queries: ['Python 3.14 pathlib'], elapsed_ms: 10, pages: [], sources: [{ id: 1 }] })
  const latest = h.evidence.at(-1)!
  expect(latest).toMatchObject({ mode: 'agent', query: 'Python 3.14 pathlib', queries: ['Python 3.14 pathlib', 'Path.read_text 3.14'], elapsed_ms: 30 })
  expect(latest.sources.map(item => item.id)).toEqual([1, 2])
  expect(latest.pages).toHaveLength(2)
  expect(h.evidence[0].pages).toEqual([])
  expect(h.evidence[1].pages).toEqual([])
  expect(h.phases).toEqual(['generating', 'searching', 'generating', 'searching', 'tools', 'reading', 'generating', 'reading', 'generating'])
})

test('原生可以直接回答，不强制发起搜索；工具 schema 不添加任意上限', async () => {
  const h = harness([answer()])
  await h.run()
  expect(h.searches).toEqual([])
  expect(h.reads).toEqual([])
  expect(h.evidence).toEqual([])
  expect(h.requests[0].tools).toHaveLength(3)
  const tools = h.requests[0].tools as Array<{ function: { parameters: { properties: Record<string, Record<string, unknown>> } } }>
  expect(tools[1].function.parameters.properties.line_count).toEqual({ type: 'integer', minimum: 1 })
})

test('直接读取网页加入稳定来源，并识别读取后的重定向URL', async () => {
  const h = harness([
    using(call('a', 'read_page', { url: 'https://example.com/short' })),
    using(call('b', 'read_page', { url: 'https://example.com/canonical', start_line: 81 })), answer(),
  ])
  h.options.readPage = async args => ({ url: 'https://example.com/canonical', title: '实际页面', content: '正文', start_line: args.start_line ?? 1, end_line: 160, total_lines: 200, elapsed_ms: 8 })
  await h.run()
  expect(h.evidence.at(-1)).toMatchObject({ elapsed_ms: 0, queries: [], sources: [{ id: 1, title: '实际页面', url: 'https://example.com/canonical', snippet: '' }], pages: [{ source_id: 1 }, { source_id: 1 }] })
})

test('网络失败回传匹配的 tool error，剩余预算内允许模型选择下一步', async () => {
  const h = harness([
    using(call('fail-search', 'search_web', { query: '首次' })),
    using(call('success', 'search_web', { query: '重查' })),
    using(call('fail-read', 'read_page', { url: 'https://example.com/docs' })), answer('使用摘要作答'),
  ])
  h.options.search = async query => {
    h.searches.push(query)
    if (query === '首次') throw new Error('SearXNG 503')
    return { query, elapsed_ms: 11, sources: [source(1)] }
  }
  h.options.readPage = async () => { throw new Error('HTTP 403') }
  await h.run()
  expect(h.requests[1].messages.at(-1)).toMatchObject({ role: 'tool', tool_call_id: 'fail-search' })
  expect(JSON.parse(String(h.requests[1].messages.at(-1)!.content))).toEqual({ error: { tool: 'search_web', message: 'SearXNG 503' } })
  expect(JSON.parse(String(h.requests[3].messages.at(-1)!.content))).toEqual({ error: { tool: 'read_page', message: 'HTTP 403' } })
  expect(h.evidence).toHaveLength(1)
  expect(h.evidence[0].pages).toEqual([])
})

test('预算计算整个发送的搜索和读取操作，停止时保留已有证据', async () => {
  const h = harness([
    using(call('search', 'search_web', { query: 'a' })),
    using(call('read', 'read_page', { url: 'https://example.com/docs' })),
    using(call('extra', 'search_web', { query: 'b' })),
  ], { maxCalls: 2 })
  await expect(h.run()).rejects.toThrow('本轮预算（2 次）')
  expect(h.searches).toEqual(['a'])
  expect(h.reads).toHaveLength(1)
  expect(h.evidence.at(-1)!.pages).toHaveLength(1)
  expect(h.requests).toHaveLength(3)
})

test('网络失败也消耗操作预算，不无限重试', async () => {
  const h = harness([using(call('a', 'search_web', { query: 'a' })), using(call('b', 'search_web', { query: 'b' }))], { maxCalls: 1 })
  let attempts = 0
  h.options.search = async () => { attempts += 1; throw new Error('离线') }
  await expect(h.run()).rejects.toThrow('本轮预算（1 次）')
  expect(attempts).toBe(1)
})

for (const name of ['search_web', 'read_page', 'search_images']) {
  test(`同名自定义工具 ${name} 在调用模型前明确拒绝`, async () => {
    const h = harness([answer()], { tools: [{ type: 'function', function: { name, parameters: { type: 'object' } } }] })
    await expect(h.run()).rejects.toThrow('与内置联网工具冲突')
    expect(h.requests).toEqual([])
  })
}

for (const [label, toolCalls, error] of [
  ['未知工具', [call('a', 'run_shell', { command: 'echo' })], '只自动执行'],
  ['缺少id', [{ type: 'function', function: { name: 'search_web', arguments: '{"query":"a"}' } }], 'call id'],
  ['重复id', [call('a', 'search_web', { query: 'a' }), call('a', 'search_web', { query: 'b' })], '重复'],
  ['非法JSON', [call('a', 'search_web', '{oops')], '不是有效 JSON'],
  ['参数非对象', [call('a', 'search_web', '[]')], '必须是 JSON 对象'],
  ['空查询', [call('a', 'search_web', { query: '  ' })], 'query 必须是非空'],
  ['URL非HTTP', [call('a', 'read_page', { url: 'file:///D:/secret' })], 'HTTP(S)'],
  ['非法页码', [call('a', 'read_page', { url: 'https://example.com', start_line: 0 })], 'start_line 必须是正整数'],
  ['非整数行数', [call('a', 'read_page', { url: 'https://example.com', line_count: 1.5 })], 'line_count 必须是正整数'],
] as Array<[string, ToolCall[], string]>) {
  test(`原生调用${label}有界停止，不继续模型轮次`, async () => {
    const h = harness([using(...toolCalls)])
    await expect(h.run()).rejects.toThrow(error)
    expect(h.requests).toHaveLength(1)
    expect(h.searches).toEqual([])
    expect(h.reads).toEqual([])
  })
}

test('tool_calls 非数组或缺少实际调用时明确报错', async () => {
  const bad = using(); (bad.message as unknown as Record<string, unknown>).tool_calls = { length: 0 }
  await expect(harness([bad]).run()).rejects.toThrow('tool_calls 必须是数组')
  await expect(harness([using()]).run()).rejects.toThrow('没有返回完整的 tool_calls')
})

for (const stage of ['before', 'generate', 'search', 'read'] as const) {
  test(`Abort 在 ${stage} 阶段立即结束，不生成工具错误或继续请求`, async () => {
    const h = harness([using(call('a', stage === 'read' ? 'read_page' : 'search_web', stage === 'read' ? { url: 'https://example.com/docs' } : { query: 'a' })), answer()])
    const reason = new DOMException('用户停止', 'AbortError')
    if (stage === 'before') h.controller.abort(reason)
    if (stage === 'generate') h.options.generate = async request => { h.requests.push(request); h.controller.abort(reason); return answer() }
    if (stage === 'search') h.options.search = async () => { h.controller.abort(reason); throw reason }
    if (stage === 'read') h.options.readPage = async () => { h.controller.abort(reason); throw reason }
    await expect(h.run()).rejects.toBe(reason)
    expect(h.requests).toHaveLength(stage === 'before' ? 0 : 1)
    expect(h.evidence).toEqual([])
  })
}

test('回调独立抛出的 AbortError 也不包装成 tool error', async () => {
  const reason = new DOMException('请求已取消', 'AbortError')
  const h = harness([using(call('a', 'search_web', { query: 'a' })), answer()], { search: async () => { throw reason } })
  await expect(h.run()).rejects.toBe(reason)
  expect(h.requests).toHaveLength(1)
})

for (const stage of ['generate', 'search', 'read'] as const) {
  test(`Abort 中断尚未完成的 ${stage} 回调，迟到结果不写入证据或继续调用`, async () => {
    const h = harness([using(call('a', stage === 'read' ? 'read_page' : 'search_web', stage === 'read' ? { url: 'https://example.com/docs' } : { query: 'a' })), answer()])
    let started!: () => void, complete!: (value: unknown) => void
    const pending = new Promise<unknown>(resolve => { complete = resolve })
    const ready = new Promise<void>(resolve => { started = resolve })
    if (stage === 'generate') h.options.generate = async request => { h.requests.push(request); started(); return await pending as WebCompletion }
    if (stage === 'search') h.options.search = async () => { started(); return await pending as WebSearchResult }
    if (stage === 'read') h.options.readPage = async () => { started(); return await pending as WebPage }
    const running = h.run()
    await ready
    const reason = new DOMException('立即停止', 'AbortError')
    h.controller.abort(reason)
    await expect(running).rejects.toBe(reason)
    if (stage === 'generate') complete(answer())
    if (stage === 'search') complete({ query: 'a', sources: [source(1)], elapsed_ms: 10 })
    if (stage === 'read') complete({ url: 'https://example.com/docs', title: '文档', content: '迟到正文', start_line: 1, end_line: 1, total_lines: 1, elapsed_ms: 10 })
    await Promise.resolve()
    expect(h.evidence).toEqual([])
    expect(h.requests).toHaveLength(1)
  })
}

test('兼容模式用对话生成去重查询与阅读计划，最终只传原用户工具并保留输入', async () => {
  const tools = [{ type: 'function', function: { name: 'user_lookup', parameters: { type: 'object' } } }]
  const h = harness([
    answer('```json\n{"queries":["Python 3.14 pathlib"," Python 3.14 pathlib ","Path.read_text 3.14"]}\n```'),
    plan({ read_source_ids: [2, 2] }), answer(),
  ], { mode: 'retrieve', tools })
  h.options.search = async query => {
    h.searches.push(query)
    return { query, elapsed_ms: 15, sources: [source(90), source(1, 'https://example.com/new', '新版说明')] }
  }
  const original = structuredClone(h.messages), originalTools = structuredClone(tools)
  await h.run()
  expect(h.messages).toEqual(original)
  expect(tools).toEqual(originalTools)
  expect(h.searches).toEqual(['Python 3.14 pathlib', 'Path.read_text 3.14'])
  expect(h.reads).toEqual([{ url: 'https://example.com/new' }])
  expect(h.requests.map(request => request.phase)).toEqual(['planning', 'planning', 'answer'])
  expect(h.requests.slice(0, 2).every(request => request.stream === false && !('tools' in request))).toBe(true)
  expect(h.requests[0].messages[1].content).toBe('上轮讨论 Python 3.14。')
  expect(h.requests[0].messages[1].reasoning_content).toBe('历史推理')
  expect(h.requests[0].messages[0].content).toContain('解析指代')
  expect(h.requests[2].tools).toEqual(tools)
  expect(h.requests[2].messages[0].content).toContain('pages 才是已读取的正文片段')
  expect(h.requests[2].messages[0].content).not.toMatch(/编号|source_ids?|引用/)
  const lastContent = h.requests[2].messages.at(-1)!.content as Array<{ type: string; text?: string }>
  expect(lastContent.slice(1)).toEqual(h.messages.at(-1)!.content)
  expect(lastContent[0].text).toContain('"source_id":2')
  expect(h.evidence.at(-1)).toMatchObject({ mode: 'retrieve', elapsed_ms: 30, sources: [{ id: 1 }, { id: 2 }], pages: [{ source_id: 2 }] })
})

test('兼容模式允许摘要足够，不伪造已读网页', async () => {
  const h = harness([plan({ queries: ['a'] }), plan({ read_source_ids: [] }), answer()], { mode: 'retrieve' })
  await h.run()
  expect(h.reads).toEqual([])
  expect(h.evidence.at(-1)!.pages).toEqual([])
  expect(h.requests.at(-1)!.tools).toBeUndefined()
})

for (const mode of ['agent', 'retrieve'] as const) {
  test(`${mode} 保留 OpenAI 综合摘要并重映射引用，不伪造网页原文`, async () => {
    const h = harness(mode === 'agent' ? [
      using(call('first', 'search_web', { query: '旧来源' }), call('second', 'search_web', { query: '综合检索' })), answer(),
    ] : [plan({ queries: ['旧来源', '综合检索'] }), plan({ read_source_ids: [] }), answer()], { mode })
    const summary = { text: '综合结论：[新文档](https://example.com/new)，[文档](https://example.com/docs)。', source_ids: [88, 10] }
    h.options.search = async query => query === '旧来源'
      ? { query, elapsed_ms: 1, sources: [source(99)] }
      : { query, elapsed_ms: 2, sources: [{ ...source(88, 'https://example.com/new'), snippet: '' }, { ...source(10), snippet: '' }], summaries: [summary] }
    const original = structuredClone(summary)
    await h.run()
    expect(summary).toEqual(original)
    const evidence = h.evidence.at(-1)!
    expect(evidence.summaries).toEqual([{ ...summary, source_ids: [2, 1] }])
    expect(evidence.pages).toEqual([])
    expect(evidence.sources[1].snippet).toBe('')
    expect(h.reads).toEqual([])
    const request = h.requests.at(-1)!
    if (mode === 'agent') {
      const result = JSON.parse(String(request.messages.at(-1)!.content))
      expect(result.summaries).toEqual(evidence.summaries)
      expect(JSON.stringify(request.tools)).toContain('综合摘要')
    } else {
      expect(request.messages[0].content).toContain('服务生成的摘要，不是网页原文')
      const content = request.messages.at(-1)!.content as Array<{ text?: string }>
      expect(content[0].text).toContain(JSON.stringify(evidence.summaries))
    }
  })
}

test('兼容模式无搜索命中时跳过无用阅读计划，用空来源生成最终回答', async () => {
  const h = harness([plan({ queries: ['a'] }), answer()], { mode: 'retrieve', search: async query => ({ query, sources: [], elapsed_ms: 12 }) })
  await h.run()
  expect(h.requests.map(request => request.phase)).toEqual(['planning', 'answer'])
  expect(h.evidence.at(-1)!.sources).toEqual([])
})

for (const response of ['', 'not json', '参考：{"queries":["a"]}', '```json\n{"queries":["a"]}\n``` 后续说明', '[]', '{}', '{"queries":[]}', '{"queries":[" "]}', '{"queries":[1]}']) {
  test(`兼容查询计划拒绝坏响应 ${JSON.stringify(response)}`, async () => {
    const h = harness([answer(response)], { mode: 'retrieve' })
    await expect(h.run()).rejects.toThrow('检索查询计划')
    expect(h.searches).toEqual([])
    expect(h.requests).toHaveLength(1)
  })
}

for (const selection of [{ read_source_ids: [99] }, { read_source_ids: [0] }, { read_source_ids: ['1'] }, { read_source_ids: null }, { queries: ['a'] }]) {
  test(`兼容阅读计划拒绝非法来源 ${JSON.stringify(selection)}`, async () => {
    const h = harness([plan({ queries: ['a'] }), plan(selection)], { mode: 'retrieve' })
    await expect(h.run()).rejects.toThrow('read_source_ids 必须是本轮来源编号数组')
    expect(h.reads).toEqual([])
    expect(h.requests).toHaveLength(2)
    expect(h.evidence).toHaveLength(1)
  })
}

for (const stage of ['search', 'read'] as const) {
  test(`兼容模式 ${stage} 失败明确结束，不换模式或假装读取成功`, async () => {
    const h = harness([plan({ queries: ['a'] }), plan({ read_source_ids: [1] }), answer()], { mode: 'retrieve' })
    if (stage === 'search') h.options.search = async () => { throw new Error('HTTP 503') }
    else h.options.readPage = async () => { throw new Error('HTTP 403') }
    await expect(h.run()).rejects.toThrow(stage === 'search' ? '联网搜索失败：HTTP 503' : '读取网页失败：HTTP 403')
    expect(h.requests).toHaveLength(stage === 'search' ? 1 : 2)
    expect(h.evidence.at(-1)?.pages ?? []).toEqual([])
  })
}

test('兼容多查询超预算保留已得到来源，不发出额外搜索', async () => {
  const h = harness([plan({ queries: ['a', 'b'] })], { mode: 'retrieve', maxCalls: 1 })
  await expect(h.run()).rejects.toThrow('本轮预算（1 次）')
  expect(h.searches).toEqual(['a'])
  expect(h.evidence).toHaveLength(1)
})

test('发给使用方的证据快照与后续上下文互相隔离', async () => {
  const h = harness([using(call('a', 'search_web', { query: 'a' })), using(call('b', 'read_page', { url: 'https://example.com/docs' })), answer()])
  h.options.onEvidence = result => { h.evidence.push(result); result.sources[0].title = '使用方修改' }
  await h.run()
  const toolResult = JSON.parse(String(h.requests[1].messages.at(-1)!.content))
  expect(toolResult.sources[0].title).toBe('文档')
  expect(h.evidence[0].pages).toEqual([])
  expect(h.evidence[1].pages).toHaveLength(1)
})

test('任意正整数预算可用，无效预算在模型请求前拒绝', async () => {
  await harness([answer()], { maxCalls: 10000 }).run()
  for (const maxCalls of [0, -1, 1.5, Number.NaN, Number.POSITIVE_INFINITY]) {
    const h = harness([], { maxCalls })
    await expect(h.run()).rejects.toThrow('工具调用预算必须是正整数')
    expect(h.requests).toEqual([])
  }
})

test('搜图与网页共用稳定来源，重复搜图保留多个图片和正确来源编号', async () => {
  const h = harness([
    using(call('web', 'search_web', { query: '介绍' })),
    using(call('images', 'search_images', { query: '示意图' })),
    using(call('repeat', 'search_images', { query: '示意图 原图' })),
    answer('![示例图片](https://images.example.com/picture.png)[2](https://images.example.com/page)'),
  ])
  await h.run()
  const imageTool = JSON.parse(String(h.requests[2].messages.at(-1)!.content))
  expect(imageTool.images[0]).toMatchObject({ source_id: 2, url: 'https://images.example.com/picture.png' })
  expect(imageTool.sources[0].id).toBe(2)
  expect(h.evidence.at(-1)!.images).toHaveLength(1)
  expect(h.evidence.at(-1)!.sources.map(source => source.id)).toEqual([1, 2])
  expect(h.evidence.at(-1)!.elapsed_ms).toBe(29)
  expect(h.requests[0].messages[0].content).toContain('![图片说明](图片URL)')
})

test('兼容模式允许只搜图片并把直链交给最终回答', async () => {
  const h = harness([plan({ queries: [], image_queries: ['标志', ' 标志 '] }), plan({ read_source_ids: [] }), answer()], { mode: 'retrieve' })
  await h.run()
  expect(h.searches).toEqual([])
  expect(h.reads).toEqual([])
  expect(h.evidence.at(-1)!.queries).toEqual(['标志'])
  expect(h.evidence.at(-1)!.images).toMatchObject([{ source_id: 1, url: 'https://images.example.com/picture.png' }])
  expect(JSON.stringify(h.requests.at(-1)!.messages)).toContain('https://images.example.com/picture.png')
})

test('图片搜索同样消耗预算；取消期间的结果不写入证据', async () => {
  const budget = harness([using(call('first', 'search_images', { query: '图片' })), using(call('extra', 'search_web', { query: '网页' }))], { maxCalls: 1 })
  await expect(budget.run()).rejects.toThrow('本轮预算（1 次）')
  expect(budget.evidence.at(-1)!.images).toHaveLength(1)
  expect(budget.searches).toEqual([])
  const cancelled = harness([using(call('image', 'search_images', { query: '图片' }))])
  cancelled.options.searchImages = async query => {
    cancelled.controller.abort(new DOMException('用户停止', 'AbortError'))
    return { query, sources: [], images: [], elapsed_ms: 1 }
  }
  await expect(cancelled.run()).rejects.toThrow('用户停止')
  expect(cancelled.evidence).toEqual([])
  expect(cancelled.requests).toHaveLength(1)
})

for (const mode of ['agent', 'retrieve'] as const) test(`${mode}工具开始与结果按执行顺序发出，并保留失败状态`, async () => {
  const events: string[] = []
  const callbacks: Partial<WebChatOptions> = {
    onToolStart: tool => events.push(`start:${tool.function.name}`),
    onToolResult: (tool, result) => events.push(`end:${tool.function.name}:${'error' in (result as object) ? 'failed' : 'ok'}`),
  }
  const h = harness(mode === 'agent' ? [using(call('s', 'search_web', { query: 'test' })), using(call('r', 'read_page', { url: 'https://example.com/docs' })), answer()]
    : [plan({ queries: ['test'] }), plan({ read_source_ids: [1] }), answer()], { mode, ...callbacks })
  await h.run()
  expect(events).toEqual(['start:search_web', 'end:search_web:ok', 'start:read_page', 'end:read_page:ok'])
  events.length = 0
  const failed = harness(mode === 'agent' ? [using(call('s', 'search_web', { query: 'test' })), answer()]
    : [plan({ queries: ['test'] })], { mode, ...callbacks, search: async () => { throw new Error('断线') } })
  if (mode === 'retrieve') await expect(failed.run()).rejects.toThrow('断线')
  else await failed.run()
  expect(events).toEqual(['start:search_web', 'end:search_web:failed'])
})

for (const mode of ['agent', 'retrieve'] as const) {
  test(`${mode} 阅读摘要保留引用和目标URL，不伪造原文行号`, async () => {
    const url = 'https://example.com/docs'
    const summary: WebPageSummary = { kind: 'summary', url, sources: [source(101), source(9, 'https://example.com/extra')], summaries: [{ text: '阅读要点 [文档](https://example.com/docs)', source_ids: [101, 9], url }], elapsed_ms: 4 }
    const h = harness(mode === 'agent'
      ? [using(call('search', 'search_web', { query: 'query' })), using(call('read', 'read_page', { url })), answer()]
      : [plan({ queries: ['query'] }), plan({ read_source_ids: [1] }), answer()], { mode, readPage: async () => summary })
    await h.run()
    const result = h.evidence.at(-1)!
    expect(result.pages).toEqual([])
    expect(result.summaries).toEqual([{ ...summary.summaries[0], source_ids: [1, 2] }])
    expect(result.sources.map(item => item.id)).toEqual([1, 2])
    const request = h.requests.at(-1)!
    if (mode === 'agent') {
      const read = JSON.parse(String(request.messages.at(-1)!.content))
      expect(read).toMatchObject({ kind: 'summary', url, summaries: result.summaries })
      expect(read).not.toHaveProperty('content')
      expect(read).not.toHaveProperty('total_lines')
      expect(request.messages[0].content).toContain('kind:summary')
    } else {
      expect(request.messages[0].content).toContain('阅读摘要')
      expect(JSON.stringify(request.messages.at(-1)!.content)).toContain('阅读要点')
    }
  })
}

test('顺位选择后的原文续读固定渠道，切换详情进入工具结果而不污染原文证据', async () => {
  const url = 'https://example.com/docs'
  const routing = { provider: 'tavily', label: 'Tavily', attempts: [{ provider: 'deepseek', label: 'DeepSeek', error: '请求超时' }, { provider: 'tavily', label: 'Tavily' }] }
  const reads: any[] = []
  const h = harness([using(call('s', 'search_web', { query: 'q' })), using(call('r', 'read_page', { url })), using(call('r2', 'read_page', { url, start_line: 81 })), answer()], {
    search: async query => ({ query, sources: [source(1)], elapsed_ms: 35, routing }),
    readPage: async args => { reads.push(args); return { url, title: '文档', content: '原文', start_line: args.start_line || 1, end_line: args.start_line || 1, total_lines: 200, elapsed_ms: 1, ...(reads.length === 1 ? { routing } : {}) } },
  })
  await h.run()
  expect(reads).toEqual([{ url }, { url, start_line: 81, provider: 'tavily' }])
  expect(h.evidence.at(-1)!.pages!.every(page => !('routing' in page))).toBe(true)
  const tools = h.requests.at(-1)!.messages.filter(message => message.role === 'tool').map(message => JSON.parse(String(message.content)))
  expect(tools[0].routing).toEqual(routing)
  expect(tools[1].routing).toEqual(routing)
})

test('阅读摘要不能按行续读，不再请求付费渠道', async () => {
  const url = 'https://example.com/docs'; let reads = 0
  const h = harness([using(call('r', 'read_page', { url })), using(call('r2', 'read_page', { url, start_line: 81 })), answer()], {
    readPage: async () => { reads++; return { kind: 'summary', url, sources: [source(1)], summaries: [{ text: '摘要', source_ids: [1], url }], elapsed_ms: 1 } },
  })
  await h.run()
  expect(reads).toBe(1)
  expect(String(h.requests.at(-1)!.messages.at(-1)!.content)).toContain('阅读摘要不支持按行分页')
})

test('会话重开后的原文续读复用已保存的成功渠道，重新从首行读可改用新选择', async () => {
  const url = 'https://example.com/docs', reads: any[] = []
  const previous = { url, title: '文档', content: '正文', start_line: 1, end_line: 80, total_lines: 200, elapsed_ms: 1, routing: { provider: 'tavily', label: 'Tavily', attempts: [{ provider: 'tavily', label: 'Tavily' }] } }
  const h = harness([using(call('continue', 'read_page', { url, start_line: 81 })), using(call('fresh', 'read_page', { url })), using(call('new-continue', 'read_page', { url, start_line: 81 })), answer()], {
    readHistory: [{ activity: [{ id: 'previous', kind: 'tool', call: call('previous', 'read_page', { url }), status: 'completed', result: JSON.stringify(previous) }] }],
    readPage: async args => { reads.push(args); const { routing: _routing, ...page } = previous; return { ...page, start_line: args.start_line ?? 1, end_line: args.start_line ?? 1 } },
  })
  await h.run()
  expect(reads).toEqual([{ url, start_line: 81, provider: 'tavily' }, { url }, { url, start_line: 81 }])
})

function deferred<T>() {
  let resolve!: (value: T) => void, reject!: (error: unknown) => void
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no })
  return { promise, resolve, reject }
}
const found = (query: string): WebSearchResult => ({ query, elapsed_ms: 10, sources: [source(70, `https://example.com/${query}`)] })
const pageFor = (url: string, start_line = 1): WebPage => ({ url, title: url, content: '正文', start_line, end_line: start_line, total_lines: 200, elapsed_ms: 1 })

test('默认三路真实重叠、空位续发，乱序完成仍按 call id 配对且整批完成才继续模型', async () => {
  const started: string[] = [], finished: string[] = []
  const waiting = new Map<string, ReturnType<typeof deferred<WebSearchResult | WebPage>>>()
  let active = 0, peak = 0
  const work = async (key: string) => {
    started.push(key); active++; peak = Math.max(peak, active)
    const pending = deferred<WebSearchResult | WebPage>(); waiting.set(key, pending)
    try { return await pending.promise } finally { active-- }
  }
  const h = harness([using(
    call('web-a', 'search_web', { query: 'a' }), call('image-b', 'search_images', { query: 'b' }),
    call('read-c', 'read_page', { url: 'https://example.com/c' }), call('web-d', 'search_web', { query: 'd' }), call('web-e', 'search_web', { query: 'e' }),
  ), answer()], { maxCalls: 5,
    search: async query => await work(query) as WebSearchResult,
    searchImages: async query => await work(query) as WebSearchResult,
    readPage: async () => await work('c') as WebPage,
    onToolResult: tool => finished.push(tool.id!),
  })
  const running = h.run()
  await expect.poll(() => started).toEqual(['a', 'b', 'c'])
  expect(peak).toBe(3)
  const image = { source_id: 90, title: '图', url: 'https://example.com/image.png' }
  waiting.get('b')!.resolve({ query: 'b', sources: [source(90, 'https://example.com/shared')], images: [image], elapsed_ms: 4 })
  await expect.poll(() => started).toEqual(['a', 'b', 'c', 'd'])
  waiting.get('d')!.resolve({ ...found('d'), sources: [source(42, 'https://example.com/shared')], summaries: [{ text: '结论', source_ids: [42] }] })
  await expect.poll(() => started).toEqual(['a', 'b', 'c', 'd', 'e'])
  waiting.get('e')!.resolve(found('e'))
  await expect.poll(() => finished).toEqual(['image-b', 'web-d', 'web-e'])
  expect(h.requests).toHaveLength(1)
  waiting.get('c')!.resolve(pageFor('https://example.com/c'))
  waiting.get('a')!.resolve(found('a'))
  await running
  expect(peak).toBe(3)
  const tools = h.requests[1].messages.filter(message => message.role === 'tool')
  expect(tools.map(tool => tool.tool_call_id)).toEqual(['web-a', 'image-b', 'read-c', 'web-d', 'web-e'])
  const values = tools.map(tool => JSON.parse(String(tool.content)))
  expect(values.map(value => value.query ?? value.url)).toEqual(['a', 'b', 'https://example.com/c', 'd', 'e'])
  const evidence = h.evidence.at(-1)!
  expect(new Set(evidence.sources.map(source => source.url)).size).toBe(evidence.sources.length)
  const shared = evidence.sources.find(source => source.url === 'https://example.com/shared')!.id
  expect(values[1].images[0].source_id).toBe(shared)
  expect(values[3].summaries[0].source_ids).toEqual([shared])
  expect(evidence.elapsed_ms).toBe(34)
  expect(h.phases).toContain('tools')
})

for (const concurrency of [1, 5]) test(`用户选择 ${concurrency} 路控制实际并发，不套用默认三路上限`, async () => {
  const waiting = Array.from({ length: 6 }, () => deferred<WebSearchResult>())
  let started = 0, active = 0, peak = 0
  const h = harness([using(...waiting.map((_, index) => call(`q-${index}`, 'search_web', { query: String(index) }))), answer()], {
    toolConcurrency: concurrency, search: async query => {
      started++; active++; peak = Math.max(peak, active)
      try { return await waiting[Number(query)].promise } finally { active-- }
    },
  })
  const running = h.run()
  await expect.poll(() => started).toBe(concurrency)
  waiting.forEach((pending, index) => pending.resolve(found(String(index))))
  await running
  expect(started).toBe(6)
  expect(peak).toBe(concurrency)
})

test('并发按开始原子领取预算，耗尽后不发队列并保留已在途结果', async () => {
  const waiting = [deferred<WebSearchResult>(), deferred<WebSearchResult>()]
  const started: string[] = [], completed: string[] = []
  const h = harness([using(...Array.from({ length: 5 }, (_, index) => call(String(index), 'search_web', { query: String(index) })))], {
    maxCalls: 2, search: async query => { started.push(query); return waiting[Number(query)].promise },
    onToolResult: tool => completed.push(tool.id!),
  })
  const running = h.run()
  const rejected = expect(running).rejects.toThrow('本轮预算（2 次）')
  await expect.poll(() => started).toEqual(['0', '1'])
  waiting[1].resolve(found('1')); waiting[0].resolve(found('0'))
  await rejected
  expect(started).toEqual(['0', '1'])
  expect(completed.sort()).toEqual(['0', '1'])
  expect(h.evidence.at(-1)!.sources).toHaveLength(2)
  expect(h.requests).toHaveLength(1)
})

test('取消整批在途并阻止排队工具，迟到结果不更新证据、工具或阶段', async () => {
  const waiting = [deferred<WebSearchResult>(), deferred<WebSearchResult>()]
  const started: string[] = [], completed: string[] = []
  const h = harness([using(...Array.from({ length: 5 }, (_, index) => call(String(index), 'search_web', { query: String(index) }))), answer()], {
    toolConcurrency: 2, search: async query => { started.push(query); return waiting[Number(query)].promise },
    onToolResult: tool => completed.push(tool.id!),
  })
  const running = h.run()
  await expect.poll(() => started).toEqual(['0', '1'])
  const reason = new DOMException('停止整批', 'AbortError')
  h.controller.abort(reason)
  await expect(running).rejects.toBe(reason)
  const phases = [...h.phases]
  waiting[0].resolve(found('0')); waiting[1].resolve(found('1'))
  await Promise.resolve(); await Promise.resolve()
  expect(started).toEqual(['0', '1'])
  expect(completed).toEqual([])
  expect(h.evidence).toEqual([])
  expect(h.phases).toEqual(phases)
  expect(h.requests).toHaveLength(1)
})

test('并发单项失败保持 agent 独立 tool error，其他工具和队列继续', async () => {
  const waiting = ['a', 'b', 'c'].map(() => deferred<WebSearchResult>())
  const started: string[] = []
  const h = harness([using(...['a', 'b', 'c'].map(query => call(query, 'search_web', { query }))), answer()], {
    toolConcurrency: 2, search: async query => { started.push(query); return waiting['abc'.indexOf(query)].promise },
  })
  const running = h.run()
  await expect.poll(() => started).toEqual(['a', 'b'])
  waiting[0].reject(new Error('HTTP 503'))
  await expect.poll(() => started).toEqual(['a', 'b', 'c'])
  waiting[2].resolve(found('c')); waiting[1].resolve(found('b'))
  await running
  const tools = h.requests[1].messages.filter(message => message.role === 'tool')
  expect(tools.map(tool => tool.tool_call_id)).toEqual(['a', 'b', 'c'])
  expect(JSON.parse(String(tools[0].content))).toEqual({ error: { tool: 'search_web', message: 'HTTP 503' } })
  expect(h.evidence.at(-1)!.sources).toHaveLength(2)
})

test('兼容模式单项失败停止排队，等待已在途完成后结束而非继续生成', async () => {
  const waiting = [deferred<WebSearchResult>(), deferred<WebSearchResult>()]
  const started: string[] = [], completed: string[] = []
  const h = harness([plan({ queries: ['a', 'b', 'c', 'd'] }), answer()], {
    mode: 'retrieve', toolConcurrency: 2,
    search: async query => { started.push(query); return waiting['ab'.indexOf(query)].promise },
    onToolResult: tool => completed.push(tool.id!),
  })
  const running = h.run()
  const rejected = expect(running).rejects.toThrow('联网搜索失败：HTTP 503')
  await expect.poll(() => started).toEqual(['a', 'b'])
  waiting[0].reject(new Error('HTTP 503'))
  await expect.poll(() => completed).toEqual(['retrieve-1'])
  expect(started).toEqual(['a', 'b'])
  expect(h.requests).toHaveLength(1)
  waiting[1].resolve(found('b'))
  await rejected
  expect(started).toEqual(['a', 'b'])
  expect(completed).toEqual(['retrieve-1', 'retrieve-2'])
  expect(h.evidence.at(-1)!.queries).toEqual(['b'])
  expect(h.requests).toHaveLength(1)
})

test('写入及未知有副作用工具是双向屏障，读写依赖和两次审批不重叠', async () => {
  const operations = [call('read-before', 'read_file', { path: 'x.txt' }), call('list', 'list_directory', {}),
    call('write-one', 'write_file', { path: 'x.txt', content: '一' }), call('read-after', 'read_file', { path: 'x.txt' }),
    call('write-two', 'write_file', { path: 'x.txt', content: '二' }), call('attach', 'attach_file', { path: 'x.txt' }), call('unknown', 'future_tool', {})]
  const started: string[] = [], active = new Set<string>()
  const waiting = new Map(operations.map(tool => [tool.id!, deferred<void>()]))
  const observations: Array<[string, string]> = []
  let content = '原始'
  const h = harness([using(...operations), answer()], { maxCalls: 10, webEnabled: false,
    builtInTools: { definitions: [...basicToolDefinitions, ...fileToolDefinitions, { type: 'function', function: { name: 'future_tool' } }], execute: async tool => {
      started.push(tool.id!); active.add(tool.id!)
      if (!['read_file', 'list_directory'].includes(tool.function.name)) expect(active.size).toBe(1)
      await waiting.get(tool.id!)!.promise
      if (tool.function.name === 'write_file') content = String(JSON.parse(tool.function.arguments).content)
      if (tool.function.name === 'read_file') observations.push([tool.id!, content])
      active.delete(tool.id!); return { content }
    } },
  })
  const running = h.run()
  await expect.poll(() => started).toEqual(['read-before', 'list'])
  waiting.get('read-before')!.resolve(); waiting.get('list')!.resolve()
  for (const [index, id] of operations.slice(2).map(tool => tool.id!).entries()) {
    await expect.poll(() => started).toEqual(operations.slice(0, index + 3).map(tool => tool.id!))
    expect(active).toEqual(new Set([id]))
    waiting.get(id)!.resolve()
  }
  await running
  expect(observations).toEqual([['read-before', '原始'], ['read-after', '一']])
  expect(content).toBe('二')
  expect(h.requests[1].messages.filter(message => message.role === 'tool').map(message => message.tool_call_id)).toEqual(operations.map(tool => tool.id))
})

test('同页首读和 hash 分页保序，不同 URL 可并发，续读使用首读的 fallback 渠道', async () => {
  const first = deferred<WebPage>(), other = deferred<WebPage>()
  const started: any[] = []
  const url = 'https://example.com/docs'
  const routing = { provider: 'tavily', label: 'Tavily', attempts: [{ provider: 'tavily', label: 'Tavily' }] }
  const h = harness([using(call('first', 'read_page', { url: `${url}#top` }), call('other', 'read_page', { url: 'https://example.com/other' }), call('continue', 'read_page', { url: `${url}#next`, start_line: 81 })), answer()], {
    readPage: async args => {
      started.push(args)
      if (args.url.endsWith('#top')) return first.promise
      if (args.url.endsWith('/other')) return other.promise
      return pageFor(url, 81)
    },
  })
  const running = h.run()
  await expect.poll(() => started.length).toBe(2)
  first.resolve({ ...pageFor(url), routing }); other.resolve(pageFor('https://example.com/other'))
  await running
  expect(started[2]).toEqual({ url: `${url}#next`, start_line: 81, provider: 'tavily' })
  expect(h.evidence.at(-1)!.pages).toHaveLength(3)
})

test('同一批首读返回摘要时不发续读，已知重定向 URL 也保序', async () => {
  const url = 'https://example.com/docs', short = 'https://example.com/short'
  const first = deferred<WebPageSummary>(), started: string[] = []
  const previous = { ...pageFor(url), routing: { provider: 'old', label: 'Old', attempts: [{ provider: 'old', label: 'Old' }] } }
  const h = harness([using(call('first', 'read_page', { url: short }), call('continue', 'read_page', { url, start_line: 81 })), answer()], {
    readHistory: [{ activity: [{ id: 'previous', kind: 'tool', call: call('previous', 'read_page', { url: short }), status: 'completed', result: JSON.stringify(previous) }] }],
    readPage: async args => { started.push(args.url); return first.promise },
  })
  const running = h.run()
  await expect.poll(() => started).toEqual([short])
  first.resolve({ kind: 'summary', url, sources: [source(1)], summaries: [{ text: '摘要', source_ids: [1], url }], elapsed_ms: 1 })
  await running
  expect(started).toEqual([short])
  expect(String(h.requests[1].messages.at(-1)!.content)).toContain('阅读摘要不支持按行分页')
})

test('兼容搜索和搜图共用并发上限，来源选择与不同网页阅读均等待各自完整批次', async () => {
  const searchWaiting = new Map(['a', 'b', 'image'].map(query => [query, deferred<WebSearchResult>()]))
  const readWaiting = new Map<string, ReturnType<typeof deferred<WebPage>>>()
  const started: string[] = [], readStarted: string[] = []
  const h = harness([plan({ queries: ['a', 'b'], image_queries: ['image'] }), plan({ read_source_ids: [1, 2, 3] }), answer()], {
    mode: 'retrieve', toolConcurrency: 2,
    search: async query => { started.push(query); return searchWaiting.get(query)!.promise },
    searchImages: async query => { started.push(query); return searchWaiting.get(query)!.promise },
    readPage: async args => { readStarted.push(args.url); const pending = deferred<WebPage>(); readWaiting.set(args.url, pending); return pending.promise },
  })
  const running = h.run()
  await expect.poll(() => started).toEqual(['a', 'b'])
  searchWaiting.get('b')!.resolve(found('b'))
  await expect.poll(() => started).toEqual(['a', 'b', 'image'])
  searchWaiting.get('image')!.resolve(found('image'))
  expect(h.requests).toHaveLength(1)
  searchWaiting.get('a')!.resolve(found('a'))
  await expect.poll(() => readStarted.length).toBe(2)
  expect(h.requests).toHaveLength(2)
  readWaiting.get(readStarted[0])!.resolve(pageFor(readStarted[0]))
  await expect.poll(() => readStarted.length).toBe(3)
  expect(h.requests).toHaveLength(2)
  readStarted.slice(1).forEach(url => readWaiting.get(url)!.resolve(pageFor(url)))
  await running
  expect(h.requests.map(request => request.phase)).toEqual(['planning', 'planning', 'answer'])
  expect(h.evidence.at(-1)!.pages).toHaveLength(3)
})

test('并行数接受任意正整数，无效值在请求模型前拒绝', async () => {
  await harness([answer()], { toolConcurrency: 10000 }).run()
  for (const toolConcurrency of [0, -1, 1.5, Number.NaN, Number.POSITIVE_INFINITY]) {
    const h = harness([], { toolConcurrency })
    await expect(h.run()).rejects.toThrow('工具并行数必须是正整数')
    expect(h.requests).toEqual([])
  }
})
