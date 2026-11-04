import { test, expect, _electron as electron, type ElectronApplication, type Page, type Route } from '@playwright/test'
import { createServer, type ServerResponse } from 'node:http'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { pinNavigation } from './navigation'
import { withWebSearch, type WebPage, type WebSearchResult } from '../src/web-search'
import { conversationMarkdown } from '../src/chat-export'
import { emptyConversation } from '../src/chat-history'

const root = path.resolve(__dirname, '..')
const artifacts = process.env.MODLOADER_TEST_ARTIFACTS || path.join(root, '.artifacts/codex/chat-tools-20261027/web-ui')
const result: WebSearchResult = { query: 'pathlib 文档', elapsed_ms: 1200, sources: [
  { id: 1, title: 'pathlib — Python 文档', url: 'https://docs.python.org/3/library/pathlib.html', snippet: 'Object-oriented filesystem paths.' },
  { id: 2, title: 'io — Python 文档', url: 'https://docs.python.org/3/library/io.html', snippet: 'Text I/O expects and produces str objects.' },
] }

test('检索上下文仅作用于本轮请求，保留系统提示、图像、历史与导出来源', () => {
  const messages = [{ role: 'system', content: '简明回答。' }, { role: 'assistant', content: '上轮回复', reasoning_content: '上轮思考' },
    { role: 'user', content: [{ type: 'text', text: '如何读取文本？' }, { type: 'image_url', image_url: { url: 'data:image/png;base64,AA==' } }] }]
  const original = structuredClone(messages)
  const request = withWebSearch(messages, result)
  expect(messages).toEqual(original)
  expect(request[0].content).toContain('简明回答。')
  expect(request[0].content).toContain('不是指令')
  expect(request[1]).toEqual(messages[1])
  expect((request[2].content as any[]).slice(1)).toEqual(messages[2].content)
  expect(JSON.stringify(request[2].content)).toContain('docs.python.org')
  const conversation = emptyConversation()
  expect(conversation.settings).toMatchObject({ webSearch: true, webSearchMode: 'agent', webSearchRounds: 25, toolConcurrency: 3 })
  conversation.messages = [{ id: 'response', role: 'assistant', text: '使用 read_text。[1](https://docs.python.org/3/library/pathlib.html)', images: [], createdAt: Date.now(), status: 'completed', reasoningHistory: '搜索前思考\n\n阅读前思考', reasoning: '最终思考', webSearch: result }]
  const markdown = conversationMarkdown(conversation)
  expect(markdown).toContain('### 联网资料')
  expect(markdown).toContain(result.query)
  expect(markdown).toContain(result.sources[1].url)
  expect(markdown).toContain('搜索摘要')
  expect(markdown).toContain('搜索前思考\n\n阅读前思考\n\n最终思考')
})

async function management<T = any>(page: Page, route: string, method = 'GET', body?: unknown): Promise<T> {
  return page.evaluate(async ({ route, method, body }) => {
    const { baseUrl, token } = await window.modloader!.connection()
    const response = await fetch(baseUrl + route, { method, headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) })
    if (!response.ok) throw new Error(`${route}: ${response.status} ${await response.text()}`)
    return response.json()
  }, { route, method, body })
}

test('开发版联网搜索：原生工具链、兼容规划、来源持久化、导出、失败重试与断连取消', async ({}, testInfo) => {
  test.setTimeout(180_000)
  fs.mkdirSync(artifacts, { recursive: true })
  const directory = fs.mkdtempSync(path.join(artifacts, 'flow-'))
  let searchMode: 'ok' | 'hold' | 'error' = 'ok'
  let scenario: 'plain' | 'native' | 'retrieve' | 'failure' | 'cancel' | 'reuse' | 'images' = 'plain'
  let holdRead = false, pendingRead: Route | undefined
  let holdFinal = false, pendingFinal: Route | undefined
  let application: ElectronApplication | undefined, page: Page | undefined
  const heldSearches = new Map<string, ServerResponse>()
  const cancelledQueries: string[] = [], queries: string[] = [], requests: any[] = [], readRequests: any[] = [], errors: string[] = []
  const requestIds: string[] = []
  const fixtureErrors: string[] = []
  const searchResponse = { results: result.sources.map(source => ({ title: source.title, url: source.url, content: source.snippet })) }
  const imageUrl = 'https://images.example.test/illustration.png'
  const imageSource = 'https://images.example.test/article'
  const server = createServer(async (request, response) => {
    let body = ''; for await (const chunk of request) body += chunk
    const form = new URLSearchParams(body)
    const query = form.get('q') || ''
    queries.push(query)
    if (request.url !== '/search' || form.get('format') !== 'json' || request.method !== 'POST') fixtureErrors.push(`Unexpected SearXNG request: ${request.method} ${request.url} ${body}`)
    if (searchMode === 'hold') {
      heldSearches.set(query, response)
      response.on('close', () => { heldSearches.delete(query); if (!response.writableEnded) cancelledQueries.push(query) })
      return
    }
    response.writeHead(searchMode === 'error' ? 403 : 200, { 'Content-Type': 'application/json' })
    response.end(JSON.stringify(searchMode === 'error' ? { error: 'JSON disabled' }
      : form.get('categories') === 'images' ? { results: [{ title: '搜图示例', url: imageSource, img_src: imageUrl, thumbnail_src: imageUrl }] }
        : searchResponse))
  })
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const port = (server.address() as { port: number }).port
  const nativeQuestion = 'Python 3.14 pathlib 如何读取文本？'
  const nativeQuery = 'Python 3.14 pathlib read_text 官方文档'
  const followup = '那如何指定文本编码？'
  const retrieveQuery = 'Python 3.14 pathlib Path.read_text encoding'
  const thoughts = ['先查询 Python 3.14 官方文档。', '读取正文确认参数。', '资料已经齐全，按原文回答。']
  const finalAnswer = `Path.read_text() 返回文本，可传 encoding="utf-8"。[来源1](${result.sources[0].url})`
  const userTools = [{ type: 'function', function: { name: 'user_lookup', parameters: { type: 'object', properties: {} } } }]
  const toolCall = (id: string, name: string, args: unknown) => ({ id, type: 'function', function: { name, arguments: JSON.stringify(args) } })
  const fulfillModel = async (route: Route, body: any, message: any, finishReason = 'stop') => {
    const usage = { prompt_tokens: 150, completion_tokens: 17, total_tokens: 167 }
    if (!body.stream) { await route.fulfill({ json: { choices: [{ index: 0, message: { role: 'assistant', ...message }, finish_reason: finishReason }], usage } }); return }
    const chunks: any[] = []
    if (message.reasoning_content) chunks.push({ choices: [{ index: 0, delta: { reasoning_content: message.reasoning_content } }] })
    if (message.tool_calls) {
      const call = message.tool_calls[0], middle = Math.floor(call.function.arguments.length / 2)
      chunks.push({ choices: [{ index: 0, delta: { tool_calls: [{ index: 0, ...call, function: { name: call.function.name, arguments: call.function.arguments.slice(0, middle) } }] } }] })
      chunks.push({ choices: [{ index: 0, delta: { tool_calls: [{ index: 0, function: { arguments: call.function.arguments.slice(middle) } }] }, finish_reason: finishReason }], usage })
    } else {
      const middle = Math.floor(message.content.length / 2)
      chunks.push({ choices: [{ index: 0, delta: { content: message.content.slice(0, middle) } }] })
      chunks.push({ choices: [{ index: 0, delta: { content: message.content.slice(middle) }, finish_reason: finishReason }], usage })
    }
    await route.fulfill({ contentType: 'text/event-stream', body: chunks.map(chunk => `data: ${JSON.stringify(chunk)}\n\n`).join('') + 'data: [DONE]\n\n' })
  }
  const fulfillPage = async (route: Route) => {
    const body = route.request().postDataJSON()
    const start = body.start_line ?? 1
    const value: WebPage = { url: body.url, title: 'pathlib — Python 文档', content: 'Path.read_text(encoding=None, errors=None)\nReturn the decoded contents as a string.', start_line: start, end_line: start + 1, total_lines: 80, elapsed_ms: 8 }
    await route.fulfill({ contentType: 'application/x-ndjson', body: `${JSON.stringify({ type: 'progress', stage: 'web_fetch', message: '正在读取网页' })}\n${JSON.stringify({ type: 'result', value })}\n` })
  }
  try {
    application = await electron.launch({ args: [root], env: { ...process.env, MODLOADER_DATA_DIR: path.join(directory, 'data'), MODLOADER_DEV_URL: process.env.MODLOADER_DEV_URL || '', MODLOADER_SENSOR_AUTO_START: '0' }, timeout: 60_000 })
    page = await application.firstWindow()
    page.on('pageerror', error => errors.push(error.message))
    await expect(page.getByRole('status', { name: '服务在线', exact: true })).toBeVisible({ timeout: 45_000 })
    await management(page, '/api/settings', 'PATCH', { proxy_mode: 'direct', searxng_url: `http://127.0.0.1:${port}` })
    await page.route('**/api/state', async route => {
      const response = await route.fetch(), state = await response.json()
      await route.fulfill({ response, json: { ...state, instances: [{ id: 'search-model', profile_id: 'search-profile', model_id: 'search-asset', alias: '搜索验收模型', backend: 'llama_cpp', state: 'ready', capabilities: { input: ['text'], tools: false }, active_requests: 0 }] } })
    })
    await page.route('**/api/instances/search-model/telemetry', route => route.fulfill({ json: { groups: [], unavailable: [], live_generation: null } }))
    await page.route(imageUrl, route => route.fulfill({ contentType: 'image/png', body: fs.readFileSync(path.join(root, 'electron/icon.png')) }))
    await page.route('**/api/chat', async route => {
      const body = route.request().postDataJSON(); requests.push(body)
      requestIds.push(route.request().headers()['x-modloader-request-id'] || '')
      if (scenario === 'plain') { await fulfillModel(route, body, { content: '普通回复' }); return }
      if (scenario === 'reuse') {
        await fulfillModel(route, body, { content: `| 用法 | 依据 |\n|---|---|\n| 读取文本 | 沿用前文[来源1](${result.sources[0].url}) |\n| 未验证 | [来源99](https://unknown.test/) |` })
        return
      }
      if (scenario === 'images') {
        if (body.messages.at(-1).role === 'user') await fulfillModel(route, body, { content: null, tool_calls: [toolCall('image-search', 'search_images', { query: '搜图示例' })] }, 'tool_calls')
        else {
          const images = JSON.parse(body.messages.at(-1).content).images
          expect(images[0]).toMatchObject({ source_id: 1, url: imageUrl })
          await fulfillModel(route, body, { content: `![搜图示例](${images[0].url})\n\n图片出处[来源1](${imageSource})。` })
        }
        return
      }
      if (scenario === 'native') {
        const last = body.messages.at(-1)
        if (last.role === 'user') await fulfillModel(route, body, { content: null, reasoning_content: thoughts[0], tool_calls: [toolCall('native-search', 'search_web', { query: nativeQuery })] }, 'tool_calls')
        else if (last.tool_call_id === 'native-search') await fulfillModel(route, body, { content: null, reasoning_content: thoughts[1], tool_calls: [toolCall('native-read', 'read_page', { url: result.sources[0].url, start_line: 2, line_count: 2 })] }, 'tool_calls')
        else if (last.tool_call_id === 'native-read') {
          if (holdFinal) { pendingFinal = route; return }
          await fulfillModel(route, body, { content: finalAnswer, reasoning_content: thoughts[2] })
        }
        else { fixtureErrors.push('Unexpected native transcript'); await route.fulfill({ status: 500, json: { detail: 'Unexpected native transcript' } }) }
        return
      }
      const system = body.messages.filter((message: any) => message.role === 'system').map((message: any) => message.content).join('\n')
      if (system.includes('拟定联网检索查询')) {
        const query = scenario === 'cancel' ? '取消检索' : scenario === 'failure' ? '搜索失败检查' : retrieveQuery
        await fulfillModel(route, body, { content: JSON.stringify({ queries: [query] }) })
      } else if (system.includes('现在只选择需要进一步阅读的来源')) await fulfillModel(route, body, { content: JSON.stringify({ read_source_ids: [1] }) })
      else await fulfillModel(route, body, { content: finalAnswer })
    })
    await page.route('**/api/web-fetch', async route => {
      readRequests.push(route.request().postDataJSON())
      if (holdRead) { pendingRead = route; return }
      await fulfillPage(route)
    })
    await pinNavigation(page)
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    const chat = () => page!.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '测试对话', exact: true }).click()
    await chat()
    const toggle = page.getByRole('button', { name: '联网搜索', exact: true })
    const input = page.getByRole('textbox', { name: '对话消息', exact: true })
    const assistant = () => page!.locator('.chat-message.assistant').last()
    const send = async (text: string) => {
      await input.fill(text)
      await page!.getByRole('button', { name: '发送消息', exact: true }).click()
    }
    const options = async (open: boolean) => {
      const button = page!.getByRole('button', { name: '对话参数', exact: true })
      if ((await button.getAttribute('aria-expanded')) !== String(open)) await button.click()
      await expect(button).toHaveAttribute('aria-expanded', String(open))
      await expect.poll(() => page!.locator('.chat-layout').evaluate(layout => Array.from(layout.children).some(node => node.getAnimations().length))).toBe(false)
    }
    await expect(toggle).toHaveAttribute('aria-pressed', 'true')
    await expect(toggle).toHaveAttribute('data-tooltip', '联网搜索 · 自主搜索')
    await send('普通消息')
    await expect(assistant().locator('.message-text')).toHaveText('普通回复')
    expect(queries).toEqual([])
    expect(requests[0].messages.at(-1)).toEqual({ role: 'user', content: '普通消息' })
    expect(requests[0].tools.map((tool: any) => tool.function.name)).toEqual(['search_web', 'read_page', 'search_images', 'reverse_image_search'])
    expect(requests[0].stream).toBe(true)
    await expect(page.getByRole('button', { name: '发送消息', exact: true })).toBeVisible()

    // Old conversations without search settings adopt the new built-in defaults.
    const prior = await management(page, '/api/conversations')
    const plain = await management(page, `/api/conversations/${prior.active_id}`)
    const { id, title, selection, draft } = emptyConversation()
    await management(page, '/api/conversations', 'POST', { id, title, selection, draft, settings: { system: '', stream: true, toolsDraft: '' } })
    await management(page, `/api/conversations/${id}`, 'PATCH', { messages: plain.messages, activate: true })
    await page.reload(); await pinNavigation(page); await chat()
    await options(true)
    const setMode = async (value: 'off' | 'agent' | 'retrieve') => { await toggle.click(); await page!.getByRole('dialog', { name: '联网搜索选项' }).getByRole('radio', { name: ({ off: '关闭', agent: '自主搜索', retrieve: '检索后回答' })[value], exact: true }).click() }
    await expect(toggle).toHaveAttribute('data-mode', 'agent')
    await expect(toggle).toHaveAttribute('aria-pressed', 'true')
    await expect(toggle).toHaveAttribute('data-tooltip', '联网搜索 · 自主搜索')
    await expect(page.getByRole('spinbutton', { name: '工具调用预算', exact: true })).toHaveValue('25')
    await page.screenshot({ path: path.join(directory, 'default-agent-25.png') })
    await setMode('off')
    await expect(toggle).toHaveAttribute('aria-pressed', 'false')
    await expect(page.locator('.chat-save-status').filter({ hasText: /^已保存$/ })).toBeVisible()
    await page.reload(); await pinNavigation(page); await chat(); await options(true)
    await expect(toggle).toHaveAttribute('data-mode', 'off')
    await expect(toggle).toHaveAttribute('aria-pressed', 'false')
    await page.screenshot({ path: path.join(directory, 'mode-off.png') })
    await setMode('agent')
    await expect(toggle).toHaveAttribute('data-mode', 'agent')
    await setMode('off')
    await expect(toggle).toHaveAttribute('data-mode', 'off')
    await setMode('agent')
    await expect(toggle).toHaveAttribute('aria-pressed', 'true')
    await expect(toggle).toHaveAttribute('data-mode', 'agent')
    await expect(page.getByRole('spinbutton', { name: '工具调用预算', exact: true })).toHaveAttribute('placeholder', '默认 25 次')
    await page.getByRole('spinbutton', { name: '工具调用预算', exact: true }).fill('37')
    await options(false)
    scenario = 'native'; searchMode = 'hold'; holdRead = true; holdFinal = true
    await send(nativeQuestion)
    await expect(page.getByRole('button', { name: '停止', exact: true })).toBeVisible()
    await expect(assistant().locator('.message-label')).toContainText('正在搜索')
    await expect(assistant().locator('.chat-activity-reasoning')).toHaveText(thoughts[0])
    await expect(assistant().locator('.message-reasoning')).toHaveAttribute('open', '')
    await expect.poll(() => heldSearches.has(nativeQuery)).toBe(true)
    expect(requests).toHaveLength(2)
    await expect(assistant().locator('[data-metric="elapsed"]')).toHaveAttribute('aria-label', /^本轮耗时 \d+\.\d 秒$/)
    await expect(assistant().getByLabel(/^请求耗时 /)).toHaveCount(0)
    const pendingSearch = heldSearches.get(nativeQuery)!
    pendingSearch.writeHead(200, { 'Content-Type': 'application/json' }); pendingSearch.end(JSON.stringify(searchResponse)); searchMode = 'ok'
    await expect.poll(() => Boolean(pendingRead)).toBe(true)
    await expect(assistant().locator('.message-label')).toContainText('阅读网页')
    await expect(assistant().getByRole('region', { name: '思考内容' })).toContainText(thoughts[0])
    await expect(assistant().getByRole('region', { name: '思考内容' })).toContainText(thoughts[1])
    await expect(assistant().locator('.message-reasoning')).toHaveAttribute('open', '')
    expect(await assistant().locator('.chat-activity-events > *').evaluateAll(nodes => nodes.map(node => node.className))).toEqual(['chat-activity-reasoning', 'chat-activity-tool', 'chat-activity-reasoning', 'chat-activity-tool'])
    await expect(assistant().locator('.chat-activity-status').first()).toHaveAttribute('data-status', 'completed')
    await expect(assistant().locator('.chat-activity-status').last()).toHaveAttribute('data-status', 'running')
    await expect.poll(() => assistant().locator('.reasoning-collapse').evaluate(element => getComputedStyle(element).opacity)).toBe('1')
    await page.screenshot({ path: path.join(directory, 'reasoning-during-read.png') })
    await expect(assistant().locator('[data-metric="elapsed"]')).toHaveAttribute('aria-label', /^本轮耗时 \d+\.\d 秒$/)
    await expect(assistant().getByLabel(/^请求耗时 /)).toHaveCount(0)
    holdRead = false; await fulfillPage(pendingRead!); pendingRead = undefined
    await expect.poll(() => Boolean(pendingFinal)).toBe(true)
    await expect(assistant().getByRole('region', { name: '思考内容' })).toContainText(thoughts[0])
    await expect(assistant().getByRole('region', { name: '思考内容' })).toContainText(thoughts[1])
    await expect(assistant().locator('.message-reasoning')).toHaveAttribute('open', '')
    holdFinal = false; await fulfillModel(pendingFinal!, pendingFinal!.request().postDataJSON(), { content: finalAnswer, reasoning_content: thoughts[2] }); pendingFinal = undefined
    await expect(assistant().locator('.message-text')).toContainText('Path.read_text()')
    await expect(page.getByRole('button', { name: '发送消息', exact: true })).toBeVisible()
    await expect(assistant().locator('.web-search-sources summary')).toContainText('搜索结果 2 · 已读 1')
    await expect(assistant().locator('.message-reasoning')).not.toHaveAttribute('open')
    await assistant().locator('.message-reasoning summary').click()
    for (const thought of thoughts) expect((await assistant().locator('.reasoning-scroll').innerText()).split(thought)).toHaveLength(2)
    await expect(assistant().locator('.message-text sup')).toHaveCount(0)
    await expect(assistant().locator('.message-text').getByRole('link', { name: '来源1', exact: true })).toHaveAttribute('href', result.sources[0].url)
    expect(requests).toHaveLength(4)
    expect(requests.slice(1).every(request => request.stream === true)).toBe(true)
    expect(requests[1].tools.map((tool: any) => tool.function.name)).toEqual(['search_web', 'read_page', 'search_images', 'reverse_image_search'])
    expect(requests[1].messages.at(-1).content).toBe(nativeQuestion)
    const searched = requests[2].messages.at(-1)
    expect(searched).toMatchObject({ role: 'tool', tool_call_id: 'native-search' })
    expect(JSON.parse(searched.content).sources.map((source: any) => source.id)).toEqual([1, 2])
    expect(requests[2].messages.at(-2)).toMatchObject({ role: 'assistant', reasoning_content: '先查询 Python 3.14 官方文档。', tool_calls: [toolCall('native-search', 'search_web', { query: nativeQuery })] })
    expect(requests[3].messages.at(-1)).toMatchObject({ role: 'tool', tool_call_id: 'native-read' })
    expect(requests[3].messages.at(-2).reasoning_content).toBe(thoughts[1])
    expect(JSON.parse(requests[3].messages.at(-1).content)).toMatchObject({ source_id: 1, start_line: 2, end_line: 3, total_lines: 80 })
    expect(readRequests).toEqual([{ url: result.sources[0].url, start_line: 2, line_count: 2 }])
    await assistant().locator('.web-search-sources summary').click()
    await expect(assistant().locator('.web-search-result-list a')).toHaveCount(2)
    await expect(assistant().locator('.web-search-result-list a').first()).toHaveAttribute('href', result.sources[0].url)
    await expect(assistant().locator('.web-search-result-list li').nth(0)).toContainText('已读')
    await expect(assistant().locator('.web-page-excerpt')).toContainText('第 2–3 行 / 共 80 行')
    await expect(assistant().locator('.web-search-result-list li').nth(1)).toContainText('摘要')
    await expect(assistant().locator('.web-search-result-list li').nth(1)).toContainText(result.sources[1].snippet)
    await page.screenshot({ path: path.join(directory, 'native-sources.png') })
    await expect(page.locator('.chat-save-status').filter({ hasText: /^已保存$/ })).toBeVisible()
    const listing = await management(page, '/api/conversations')
    const saved = await management(page, `/api/conversations/${listing.active_id}`)
    expect(saved.settings).toMatchObject({ webSearch: true, webSearchMode: 'agent', webSearchRounds: 37, system: '' })
    expect(saved.messages.at(-1).webSearch.sources).toHaveLength(2)
    expect(saved.messages.at(-1).webSearch.pages).toHaveLength(1)
    expect(saved.messages.at(-1).webSearch.pages[0]).toMatchObject({ source_id: 1, start_line: 2, end_line: 3, elapsed_ms: 8 })
    expect(saved.messages.at(-1).usage.completion_tokens).toBe(17)
    expect(saved.messages.at(-1).tools).toBeUndefined()
    expect(saved.messages.at(-1).reasoningHistory).toBe(thoughts.slice(0, 2).join('\n\n'))
    expect(saved.messages.at(-1).reasoning).toBe(thoughts[2])
    expect(requestIds.slice(1, 4)).toEqual(Array(3).fill(saved.messages.at(-1).id))
    expect(saved.messages.at(-2).text).toBe(nativeQuestion)
    const output = path.join(directory, '联网会话.md')
    await application.evaluate(({ shell, dialog }, output) => {
      ;(globalThis as any).searchOpened = []
      shell.openExternal = async url => { (globalThis as any).searchOpened.push(url) }
      dialog.showSaveDialog = async () => ({ canceled: false, filePath: output })
    }, output)
    await page.getByRole('button', { name: '导出对话为 Markdown', exact: true }).click()
    await expect(page.getByText('已导出 Markdown', { exact: true })).toBeVisible()
    const markdown = fs.readFileSync(output, 'utf8')
    expect(markdown).toContain('### 联网资料')
    expect(markdown).toContain('查询：Python 3.14 pathlib read\\_text 官方文档')
    expect(markdown).toContain('已读：2–3 行')
    expect(markdown).toContain('搜索摘要')
    expect(markdown).toContain(result.sources[0].url)
    expect(markdown).toContain(result.sources[1].url)
    for (const thought of thoughts) expect(markdown).toContain(thought)
    expect(markdown.indexOf(thoughts[0])).toBeLessThan(markdown.indexOf('### 工具调用 · search\\_web'))
    expect(markdown.indexOf('### 工具调用 · search\\_web')).toBeLessThan(markdown.indexOf(thoughts[1]))
    expect(markdown.indexOf(thoughts[1])).toBeLessThan(markdown.indexOf('### 工具调用 · read\\_page'))
    expect(markdown.indexOf('### 工具调用 · read\\_page')).toBeLessThan(markdown.indexOf(thoughts[2]))

    await page.reload(); await pinNavigation(page); await chat()
    await expect(toggle).toHaveAttribute('aria-pressed', 'true')
    await expect(toggle).toHaveAttribute('data-tooltip', '联网搜索 · 自主搜索')
    await expect(assistant().locator('.web-search-sources summary')).toContainText('搜索结果 2 · 已读 1')
    await assistant().locator('.message-reasoning summary').click()
    for (const thought of thoughts) await expect(assistant().getByRole('region', { name: '思考内容' })).toContainText(thought)
    await assistant().locator('.message-text').getByRole('link', { name: '来源1', exact: true }).click()
    await assistant().locator('.web-search-sources summary').click()
    await assistant().locator('.web-search-result-list a').first().click()
    expect(await application.evaluate(() => (globalThis as any).searchOpened)).toEqual([result.sources[0].url, result.sources[0].url])

    scenario = 'reuse'
    await send('沿用刚才的来源列成表格')
    await expect(assistant().locator('td').getByRole('link')).toHaveText(['来源1', '来源99'])
    await expect(assistant().locator('td sup')).toHaveCount(0)
    await expect(assistant().getByRole('link', { name: '来源99', exact: true })).toBeVisible()
    await expect(assistant().locator('.web-search-sources')).toHaveCount(0)
    await expect(page.locator('.chat-save-status').filter({ hasText: /^已保存$/ })).toBeVisible()
    expect((await management(page, `/api/conversations/${listing.active_id}`)).messages.at(-1).webSearch).toBeUndefined()
    await page.reload(); await pinNavigation(page); await chat()
    await expect(assistant().locator('td').getByRole('link')).toHaveText(['来源1', '来源99'])
    await expect(assistant().locator('td sup')).toHaveCount(0)
    await page.screenshot({ path: path.join(directory, 'history-citation-table.png') })

    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '设置', exact: true }).click()
    const engine = page.getByRole('textbox', { name: /^图片引擎简称/ })
    await expect(engine).toHaveValue('bdi')
    await engine.fill(''); await engine.press('Enter')
    await expect.poll(async () => (await management(page!, '/api/state')).settings.searxng_image_engine).toBe('')
    await engine.fill('bdi'); await engine.press('Enter')
    await expect.poll(async () => (await management(page!, '/api/state')).settings.searxng_image_engine).toBe('bdi')
    await page.screenshot({ path: path.join(directory, 'image-engine-setting.png') })
    await chat()
    scenario = 'images'
    await send('搜索并展示图片')
    const image = assistant().getByRole('button', { name: '查看图片：搜图示例', exact: true })
    await expect(image).toBeVisible()
    await expect.poll(() => image.evaluate(node => (node as HTMLImageElement).naturalWidth)).toBeGreaterThan(0)
    await expect(assistant().locator('.message-text').getByRole('link', { name: '来源1', exact: true })).toHaveAttribute('href', imageSource)
    await expect(assistant().locator('.web-search-sources summary')).toContainText('图片 1')
    expect(queries.at(-1)).toBe('!bdi 搜图示例')
    await expect(page.locator('.chat-save-status').filter({ hasText: /^已保存$/ })).toBeVisible()
    const imageMessage = (await management(page, `/api/conversations/${listing.active_id}`)).messages.at(-1)
    expect(imageMessage.webSearch.images).toEqual([{ source_id: 1, title: '搜图示例', url: imageUrl, thumbnail_url: imageUrl }])
    expect(imageMessage.webSearch.pages).toEqual([])
    await page.getByRole('button', { name: '导出对话为 Markdown', exact: true }).click()
    await expect.poll(() => fs.readFileSync(output, 'utf8')).toContain(imageUrl)
    expect(fs.readFileSync(output, 'utf8')).toContain('图片来源')
    await page.reload(); await pinNavigation(page); await chat()
    await expect(image).toBeVisible()
    await expect.poll(() => image.evaluate(node => (node as HTMLImageElement).naturalWidth)).toBeGreaterThan(0)
    await page.screenshot({ path: path.join(directory, 'image-search-reopened.png') })

    await options(true)
    await expect(toggle).toHaveAttribute('data-mode', 'agent')
    await expect(page.getByRole('spinbutton', { name: '工具调用预算', exact: true })).toHaveValue('37')
    await setMode('off')
    await expect(toggle).toHaveAttribute('aria-pressed', 'false')
    await setMode('retrieve')
    await expect(toggle).toHaveAttribute('aria-pressed', 'true')
    await page.getByLabel('流式输出', { exact: true }).uncheck()
    await page.getByLabel('工具定义', { exact: true }).fill(JSON.stringify(userTools))
    await options(false)
    await expect(toggle).toHaveAttribute('data-tooltip', '联网搜索 · 检索后回答')
    scenario = 'retrieve'
    const retrieveStart = requests.length
    await send(followup)
    await expect(assistant().locator('.message-text')).toContainText('encoding="utf-8"')
    await expect(page.getByRole('button', { name: '发送消息', exact: true })).toBeVisible()
    const retrieveRequests = requests.slice(retrieveStart)
    expect(retrieveRequests).toHaveLength(3)
    expect(retrieveRequests.every(request => request.stream === false)).toBe(true)
    expect(retrieveRequests.slice(0, 2).every(request => !('tools' in request))).toBe(true)
    expect(retrieveRequests[0].messages.at(-1).content).toBe(followup)
    expect(retrieveRequests[0].messages.some((message: any) => message.role === 'user' && message.content === nativeQuestion)).toBe(true)
    expect(retrieveRequests[0].messages.find((message: any) => message.role === 'assistant' && message.content === finalAnswer).reasoning_content).toBe(thoughts[2])
    expect(retrieveRequests[0].messages[0].content).toContain('解析指代')
    expect(queries.at(-1)).toBe(retrieveQuery)
    expect(retrieveRequests[2].tools).toEqual(userTools)
    expect(retrieveRequests[2].messages.at(-1).content).toContain('"source_id":1')
    expect(retrieveRequests[2].messages.at(-1).content.endsWith(`用户问题：\n${followup}`)).toBe(true)
    await expect(page.locator('.chat-message.user').last().locator('.message-text')).toHaveText(followup)
    await assistant().locator('.web-search-sources summary').click()
    await options(true)
    await page.screenshot({ path: path.join(directory, 'retrieve-settings.png') })
    await expect(page.locator('.chat-save-status').filter({ hasText: /^已保存$/ })).toBeVisible()
    const retrieved = await management(page, `/api/conversations/${listing.active_id}`)
    expect(requestIds.slice(retrieveStart, retrieveStart + 3)).toEqual(Array(3).fill(retrieved.messages.at(-1).id))
    expect(retrieved.settings).toMatchObject({ webSearch: true, webSearchMode: 'retrieve', webSearchRounds: 37, stream: false, toolsDraft: JSON.stringify(userTools) })
    expect(retrieved.messages.at(-2).text).toBe(followup)
    expect(retrieved.messages.at(-1).webSearch).toMatchObject({ mode: 'retrieve', query: retrieveQuery, queries: [retrieveQuery], pages: [{ source_id: 1, start_line: 1, end_line: 2 }] })
    await page.reload(); await pinNavigation(page); await chat(); await options(true)
    await expect(toggle).toHaveAttribute('aria-pressed', 'true')
    await expect(toggle).toHaveAttribute('data-tooltip', '联网搜索 · 检索后回答')
    await expect(toggle).toHaveAttribute('data-mode', 'retrieve')
    await expect(page.getByRole('spinbutton', { name: '工具调用预算', exact: true })).toHaveValue('37')
    await expect(page.getByLabel('流式输出', { exact: true })).not.toBeChecked()
    await expect(assistant().locator('.web-search-sources summary')).toContainText('搜索结果 2 · 已读 1')
    await options(false)

    scenario = 'failure'; searchMode = 'error'
    const beforeFailure = requests.length
    await send('搜索失败检查')
    await expect(assistant().locator('.message-error')).toContainText('403')
    expect(requests).toHaveLength(beforeFailure + 1)
    const userCount = await page.locator('.chat-message.user').count()
    searchMode = 'ok'
    await assistant().getByRole('button', { name: '重试回复', exact: true }).click()
    await expect(assistant().locator('.message-text')).toContainText('Path.read_text()')
    await expect(page.getByRole('button', { name: '发送消息', exact: true })).toBeVisible()
    expect(requests).toHaveLength(beforeFailure + 4)
    expect(await page.locator('.chat-message.user').count()).toBe(userCount)
    await expect(assistant().locator('.message-error')).toHaveCount(0)

    scenario = 'cancel'; searchMode = 'hold'
    const beforeCancel = requests.length
    await send('取消检索')
    await expect.poll(() => heldSearches.has('取消检索')).toBe(true)
    await expect(assistant().locator('.message-label')).toContainText('正在搜索')
    await page.getByRole('button', { name: '停止', exact: true }).click()
    await expect(assistant().locator('.message-label')).toContainText('已停止')
    await expect.poll(() => cancelledQueries.includes('取消检索'), { timeout: 10_000 }).toBe(true)
    expect(requests).toHaveLength(beforeCancel + 1)
    await expect(assistant().locator('.chat-activity-status')).toHaveAttribute('data-status', 'cancelled')
    await expect(page.locator('.chat-save-status').filter({ hasText: /^已保存$/ })).toBeVisible()

    // Earlier thoughts survive even if the next model pass produces no output.
    await options(true)
    await setMode('agent')
    await page.getByLabel('流式输出', { exact: true }).check()
    await options(false)
    scenario = 'native'; searchMode = 'ok'; holdFinal = true
    for (const outcome of ['stop', 'error'] as const) {
      await send(outcome === 'stop' ? '停止时保留多轮思考' : '失败时保留多轮思考')
      await expect.poll(() => Boolean(pendingFinal)).toBe(true)
      for (const thought of thoughts.slice(0, 2)) await expect(assistant().getByRole('region', { name: '思考内容' })).toContainText(thought)
      if (outcome === 'stop') {
        await page.getByRole('button', { name: '停止', exact: true }).click()
        await pendingFinal!.abort().catch(() => undefined)
        await expect(assistant().locator('.message-label')).toContainText('已停止')
      } else {
        await pendingFinal!.fulfill({ status: 502, json: { detail: '末轮请求失败' } })
        await expect(assistant().locator('.message-error')).toContainText('末轮请求失败')
      }
      pendingFinal = undefined
      await expect(page.getByRole('button', { name: '发送消息', exact: true })).toBeVisible()
      await expect(page.locator('.chat-save-status').filter({ hasText: /^已保存$/ })).toBeVisible()
      const interrupted = await management(page, `/api/conversations/${listing.active_id}`)
      expect(interrupted.messages.at(-1)).toMatchObject({ status: outcome === 'stop' ? 'stopped' : 'failed', reasoning: '', reasoningHistory: thoughts.slice(0, 2).join('\n\n') })
      await page.reload(); await pinNavigation(page); await chat()
      await assistant().locator('.message-reasoning summary').click()
      for (const thought of thoughts.slice(0, 2)) await expect(assistant().getByRole('region', { name: '思考内容' })).toContainText(thought)
    }

    await page.setViewportSize({ width: 940, height: 760 })
    await expect(toggle).toBeVisible()
    const overflow = await page.locator('.composer-actions').evaluate(node => node.scrollWidth > node.clientWidth)
    expect(overflow).toBe(false)
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
    await page.screenshot({ path: path.join(directory, 'compact-940.png') })
    await page.getByRole('button', { name: '新建会话', exact: true }).click()
    await expect(toggle).toHaveAttribute('aria-pressed', 'true')
    await options(true)
    await expect(toggle).toHaveAttribute('data-mode', 'agent')
    await expect(page.getByRole('spinbutton', { name: '工具调用预算', exact: true })).toHaveValue('25')
    expect(fixtureErrors).toEqual([])
    expect(errors).toEqual([])
  } finally {
    if (testInfo.status !== testInfo.expectedStatus && page && !page.isClosed()) await page.screenshot({ path: path.join(directory, 'failure.png') }).catch(() => undefined)
    await pendingRead?.abort().catch(() => undefined)
    await pendingFinal?.abort().catch(() => undefined)
    server.closeAllConnections()
    await application?.close()
    await new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve()))
  }
})

test('局域网 SearXNG 搜图经过工具链后在会话中加载真实图片', async ({}, testInfo) => {
  test.skip(process.env.MODLOADER_LIVE_IMAGE_SEARCH !== '1', '按需运行局域网服务与外部图片检查')
  const searxngUrl = process.env.MODLOADER_TEST_SEARXNG_URL
  expect(searxngUrl, '设置 MODLOADER_TEST_SEARXNG_URL 指向要测试的 SearXNG 服务').toBeTruthy()
  const proxyUrl = process.env.MODLOADER_TEST_PROXY_URL || ''
  test.setTimeout(120_000)
  fs.mkdirSync(artifacts, { recursive: true })
  const directory = fs.mkdtempSync(path.join(artifacts, 'live-image-'))
  const application = await electron.launch({ args: [root], env: { ...process.env, MODLOADER_DATA_DIR: path.join(directory, 'data'), MODLOADER_SENSOR_AUTO_START: '0' }, timeout: 60_000 })
  const page = await application.firstWindow()
  const requests: any[] = []
  let evidence: WebSearchResult | undefined
  try {
    await expect(page.getByRole('status', { name: '服务在线', exact: true })).toBeVisible({ timeout: 45_000 })
    if (proxyUrl) await application.evaluate(async ({ session }, proxyRules) => session.defaultSession.setProxy({ proxyRules }), proxyUrl)
    await management(page, '/api/settings', 'PATCH', { proxy_mode: proxyUrl ? 'manual' : 'system', proxy_url: proxyUrl, searxng_url: searxngUrl, searxng_image_engine: 'bdi' })
    await page.route('**/api/state', async route => {
      const response = await route.fetch(), state = await response.json()
      await route.fulfill({ response, json: { ...state, instances: [{ id: 'live-image-model', profile_id: 'live-image-profile', model_id: 'live-image-asset', alias: '搜图展示验收', backend: 'ninfer', state: 'ready', capabilities: { input: ['text'] }, active_requests: 0 }] } })
    })
    await page.route('**/api/instances/live-image-model/telemetry', route => route.fulfill({ json: { groups: [], unavailable: [], live_generation: null } }))
    await page.route('**/api/chat', async route => {
      const body = route.request().postDataJSON(); requests.push(body)
      const last = body.messages.at(-1)
      let message: any
      if (last.role === 'user') {
        message = { role: 'assistant', content: null, tool_calls: [{ id: 'live-image-search', type: 'function', function: { name: 'search_images', arguments: JSON.stringify({ query: 'OpenAI logo' }) } }] }
      } else {
        evidence = JSON.parse(last.content)
        expect(evidence!.images!.length).toBeGreaterThan(0)
        const image = evidence!.images![0], source = evidence!.sources.find(source => source.id === image.source_id)!
        message = { role: 'assistant', content: `![OpenAI 搜图](<${image.url}>)\n\n${image.title}[${source.id}](<${source.url}>)` }
        fs.writeFileSync(path.join(directory, 'search-result.json'), JSON.stringify(evidence, null, 2))
      }
      await route.fulfill({ json: { choices: [{ message, finish_reason: message.tool_calls ? 'tool_calls' : 'stop' }] } })
    })
    await pinNavigation(page)
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '测试对话', exact: true }).click()
    await page.getByRole('textbox', { name: '对话消息', exact: true }).fill('搜索 OpenAI logo 并显示图片')
    await page.getByRole('button', { name: '发送消息', exact: true }).click()
    const image = page.getByRole('img', { name: 'OpenAI 搜图', exact: true })
    await expect(image).toBeVisible({ timeout: 45_000 })
    await expect.poll(() => image.evaluate(node => (node as HTMLImageElement).naturalWidth), { timeout: 30_000 }).toBeGreaterThan(0)
    const source = evidence!.sources.find(source => source.id === evidence!.images![0].source_id)!
    await expect(page.locator('.chat-message.assistant .message-text').getByRole('link', { name: String(source.id), exact: true })).toHaveAttribute('href', source.url)
    await expect(page.locator('.chat-save-status').filter({ hasText: /^已保存$/ })).toBeVisible()
    expect(requests).toHaveLength(2)
    await page.screenshot({ path: path.join(directory, 'live-image-loaded.png') })
  } finally {
    if (testInfo.status !== testInfo.expectedStatus && !page.isClosed()) await page.screenshot({ path: path.join(directory, 'failure.png') }).catch(() => undefined)
    await application.close()
  }
})

test('工具有界并行显示运行和排队状态，结果乱序配对、整批等待且取消不续发', async ({}, testInfo) => {
  test.setTimeout(90_000)
  fs.mkdirSync(artifacts, { recursive: true })
  const directory = fs.mkdtempSync(path.join(artifacts, 'parallel-'))
  const application = await electron.launch({ args: [root], env: { ...process.env, MODLOADER_DATA_DIR: path.join(directory, 'data'), MODLOADER_SENSOR_AUTO_START: '0' }, timeout: 60_000 })
  const page = await application.firstWindow()
  const errors: string[] = [], requests: any[] = [], searches: string[] = []
  const pending = new Map<string, Route>()
  let round = 0
  const release = async (query: string) => {
    const route = pending.get(query)!
    pending.delete(query)
    await route.fulfill({ json: { query, sources: [{ id: 8, url: `https://example.com/${query}`, title: query, snippet: '摘要' }], elapsed_ms: 1000 } })
  }
  try {
    page.on('pageerror', issue => errors.push(issue.message))
    await expect(page.getByRole('status', { name: '服务在线', exact: true })).toBeVisible({ timeout: 45_000 })
    await page.route('**/api/state', async route => {
      const response = await route.fetch(), state = await response.json()
      await route.fulfill({ response, json: { ...state, instances: [{ id: 'parallel-model', profile_id: 'parallel-profile', model_id: 'parallel-asset', alias: '并行测试模型', backend: 'llama_cpp', state: 'ready', capabilities: { input: ['text'], tools: true }, active_requests: 0 }] } })
    })
    await page.route('**/api/instances/parallel-model/telemetry', route => route.fulfill({ json: { groups: [], unavailable: [], live_generation: null } }))
    await page.route('**/api/chat', async route => {
      const body = route.request().postDataJSON(); requests.push(body)
      const last = body.messages.at(-1)
      if (last.role === 'user') {
        round++
        const tool_calls = Array.from({ length: 5 }, (_, index) => ({ id: `round-${round}-${index}`, type: 'function', function: { name: 'search_web', arguments: JSON.stringify({ query: `round-${round}-${index}` }) } }))
        await route.fulfill({ json: { choices: [{ message: { role: 'assistant', content: null, tool_calls }, finish_reason: 'tool_calls' }] } })
      } else await route.fulfill({ json: { choices: [{ message: { role: 'assistant', content: '五项查询已全部完成。' }, finish_reason: 'stop' }] } })
    })
    await page.route('**/api/web-search', route => {
      const { query } = route.request().postDataJSON()
      searches.push(query); pending.set(query, route)
    })
    await pinNavigation(page)
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '测试对话', exact: true }).click()
    await page.getByRole('button', { name: '对话参数', exact: true }).click()
    const options = page.getByRole('complementary', { name: '对话参数', exact: true })
    await expect(options.getByRole('spinbutton', { name: '工具并行数', exact: true })).toHaveValue('3')
    await options.getByRole('checkbox', { name: '流式输出', exact: true }).uncheck()
    await page.getByRole('button', { name: '收起对话参数', exact: true }).click()
    const input = page.getByRole('textbox', { name: '对话消息', exact: true })
    const assistant = () => page.locator('.chat-message.assistant').last()
    await input.fill('并行查询五个独立问题')
    await page.getByRole('button', { name: '发送消息', exact: true }).click()
    await expect.poll(() => searches).toEqual(['round-1-0', 'round-1-1', 'round-1-2'])
    await expect(assistant().locator('.message-label')).toContainText('调用工具 · 3 项并行')
    await expect(assistant().locator('.chat-activity-tool[data-status="running"]')).toHaveCount(3)
    await expect(assistant().locator('.chat-activity-tool[data-status="pending"]')).toHaveCount(2)
    const summary = assistant().locator('.message-reasoning summary')
    if (await summary.getAttribute('aria-expanded') === 'false') await summary.click()
    await page.screenshot({ path: path.join(directory, 'three-running-two-queued.png') })
    await release('round-1-2')
    await expect.poll(() => searches.length).toBe(4)
    await release('round-1-3')
    await expect.poll(() => searches.length).toBe(5)
    await release('round-1-4')
    await expect(assistant().locator('.message-label')).toContainText('调用工具 · 2 项并行')
    expect(requests).toHaveLength(1)
    await release('round-1-0'); await release('round-1-1')
    await expect(assistant().locator('.message-text')).toContainText('五项查询已全部完成')
    const tools = requests[1].messages.filter((message: any) => message.role === 'tool')
    expect(tools.map((message: any) => message.tool_call_id)).toEqual(Array.from({ length: 5 }, (_, index) => `round-1-${index}`))
    for (const [index, tool] of tools.entries()) expect(JSON.parse(tool.content).query).toBe(`round-1-${index}`)
    await expect(assistant().locator('.chat-activity-tool[data-status="completed"]')).toHaveCount(5)
    await expect(assistant().locator('.search-elapsed')).toHaveText('搜索累计 5.0 s')
    await page.screenshot({ path: path.join(directory, 'completed-results.png') })
    await input.fill('停止下一批并行工具')
    await page.getByRole('button', { name: '发送消息', exact: true }).click()
    await expect.poll(() => searches.length).toBe(8)
    await expect(assistant().locator('.chat-activity-tool[data-status="running"]')).toHaveCount(3)
    await page.getByRole('button', { name: '停止', exact: true }).click()
    await expect(assistant().locator('.message-label')).toContainText('已停止')
    await expect(assistant().locator('.chat-activity-tool[data-status="cancelled"]')).toHaveCount(5)
    for (const route of pending.values()) await route.abort().catch(() => undefined)
    pending.clear()
    expect(searches).toHaveLength(8)
    expect(requests).toHaveLength(3)
    await expect(page.locator('.chat-save-status').filter({ hasText: /^已保存$/ })).toBeVisible()
    expect(errors).toEqual([])
    await page.screenshot({ path: path.join(directory, 'cancelled-batch.png') })
    console.log('Parallel tool UI artifacts:', directory)
  } finally {
    if (testInfo.status !== testInfo.expectedStatus && !page.isClosed()) await page.screenshot({ path: path.join(directory, 'failure.png') }).catch(() => undefined)
    for (const route of pending.values()) await route.abort().catch(() => undefined)
    await page.unrouteAll({ behavior: 'wait' })
    await application.close()
  }
})
