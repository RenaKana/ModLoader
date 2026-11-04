import { test, expect, chromium, type Route } from '@playwright/test'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { pinNavigation } from './navigation'
import type { WebPage, WebSearchResult } from '../src/web-search'
import { emptyConversation, type Conversation } from '../src/chat-history'
import { historyPage, historyUsage } from './monitor-history-fixture'
import type { BackendRequestReport } from '../src/chat-turn-metrics'

test('主监控：真实对话工具等待保持调用工具，恢复生成后完成，停止工具后中止', async ({}, testInfo) => {
  test.setTimeout(90_000)
  const root = path.resolve(__dirname, '..')
  const artifacts = process.env.MODLOADER_TEST_ARTIFACTS || path.join(root, '.artifacts/codex/monitor-implementation-20261031/legacy-ui')
  fs.mkdirSync(artifacts, { recursive: true })
  const directory = fs.mkdtempSync(path.join(artifacts, 'tools-ui-'))
  const { preview } = await import('vite')
  const server = await preview({ root, configFile: false,
    build: { outDir: process.env.MODLOADER_TEST_DIST || path.join(root, 'dist') }, preview: { host: '127.0.0.1', port: 0 } })
  const address = server.httpServer.address()
  if (!address || typeof address === 'string') throw new Error('Fixture preview did not open')
  const browser = await chromium.launch({ channel: 'msedge', headless: true })
  const page = await browser.newPage({ viewport: { width: 1440, height: 1040 } })
  const errors: string[] = [], external: string[] = [], mutations: string[] = [], fixtureErrors: string[] = []
  const modelRequests: Array<{ clientRequestId: string; body: any }> = []
  const toolRequests: Array<{ name: string; body: any }> = []
  const held = new Map<'search_web' | 'read_page' | 'search_images' | 'answer', Route>()
  const reports = new Map<string, BackendRequestReport>()
  const instance = { id: 'monitor-tools-instance', profile_id: 'monitor-tools-profile', model_id: 'monitor-tools-model',
    alias: '工具监控 fixture', backend: 'llama_cpp', state: 'ready', active_requests: 0, capabilities: { input: ['text'], tools: true } }
  const conversation = emptyConversation({ profile_id: instance.profile_id, model_id: instance.model_id, instance_id: instance.id, alias: instance.alias })
  const conversations = new Map<string, Conversation>([[conversation.id, conversation]])
  let pass: { key: string; clientRequestId: string; clientCallId: string; completed: boolean } | undefined
  let samples = 0
  const sourceUrl = 'https://monitor-tools.example.test/document'
  const searchResult: WebSearchResult = { query: '工具连续性网页', elapsed_ms: 12,
    sources: [{ id: 1, title: '工具连续性资料', url: sourceUrl, snippet: '网页搜索 fixture。' }] }
  const webPage: WebPage = { url: sourceUrl, title: '工具连续性资料', content: '读取网页 fixture。',
    start_line: 1, end_line: 1, total_lines: 1, elapsed_ms: 8 }
  const imageResult: WebSearchResult = { query: '工具连续性图片', elapsed_ms: 10,
    sources: [{ id: 1, title: '图片来源', url: 'https://monitor-tools.example.test/image-source', snippet: '' }],
    images: [{ source_id: 1, title: '图片 fixture', url: 'https://monitor-tools.example.test/image.png' }] }
  const answer = '网页、正文与图片工具均已返回。'
  const toolCall = (id: string, name: string, args: unknown) => ({ id, type: 'function', function: { name, arguments: JSON.stringify(args) } })
  const completeModel = async (route: Route, message: Record<string, unknown>, finishReason = 'stop') => {
    // An individual model call has already ended while runWebChat awaits its tool.
    pass!.completed = true; instance.active_requests = 0
    const body = route.request().postDataJSON()
    const usage = { prompt_tokens: 120, completion_tokens: 24, total_tokens: 144 }
    const callId = route.request().headers()['x-modloader-call-id']
    reports.set(callId, { request_key: pass!.key, client_request_id: pass!.clientRequestId, client_call_id: callId,
      status: 'completed', duration_s: .2, first_output_s: .1, usage, timings: { predicted_n: 24, predicted_ms: 24 / 132 * 1000 } })
    if (body.stream) {
      await route.fulfill({ contentType: 'text/event-stream', body:
        `data: ${JSON.stringify({ choices: [{ index: 0, delta: message, finish_reason: finishReason }], usage })}\n\ndata: [DONE]\n\n` })
    } else await route.fulfill({ json: { choices: [{ index: 0, message: { role: 'assistant', ...message }, finish_reason: finishReason }], usage } })
  }
  const releaseTool = async (name: 'search_web' | 'read_page' | 'search_images', value: WebSearchResult | WebPage) => {
    const route = held.get(name)!
    held.delete(name)
    await route.fulfill({ contentType: 'application/x-ndjson', body:
      `${JSON.stringify({ type: 'progress', stage: name === 'read_page' ? 'web_fetch' : 'web_search', message: '工具 fixture 返回' })}\n${JSON.stringify({ type: 'result', value })}\n` })
  }
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/*', async route => {
    const request = route.request(), url = new URL(request.url())
    const historyWrite = ['POST', 'PATCH'].includes(request.method()) && /^\/api\/conversations(?:\/[^/]+)?$/.test(url.pathname)
    if (['http:', 'https:'].includes(url.protocol) && !['127.0.0.1', 'localhost'].includes(url.hostname)) {
      external.push(url.origin); await route.abort()
    } else if (url.pathname.startsWith('/api/') && !['GET', 'HEAD', 'OPTIONS'].includes(request.method()) && !historyWrite) {
      mutations.push(`${request.method()} ${url.pathname}`); await route.abort()
    } else await route.continue()
  })
  await page.route('**/api/**', async route => {
    const request = route.request(), url = new URL(request.url()), method = request.method()
    if (url.pathname.endsWith('/generation-events') && method === 'GET') return route.fulfill({ contentType: 'text/event-stream', body: ': fixture idle\n\n' })
    if (url.pathname === '/api/state' && method === 'GET') return route.fulfill({ json: { models: [], profiles: [], runtimes: [],
      instances: [instance], downloads: [], installations: [], migrations: [], settings: {},
      backends: [{ id: 'llama_cpp', label: 'llama.cpp', parameters: [] }], hardware: { cpu_info: { name: 'isolated fixture' }, gpus: [] } } })
    if (url.pathname === '/api/chat-presets' && method === 'GET') return route.fulfill({ json: { items: [], default_id: null } })
    if (url.pathname === '/api/conversations' && method === 'GET') return route.fulfill({ json: { items: [...conversations.values()], active_id: conversation.id } })
    if (url.pathname === '/api/conversations' && method === 'POST') {
      const value = request.postDataJSON()
      const document = { ...emptyConversation(), ...value }
      conversations.set(document.id, document)
      return route.fulfill({ json: document })
    }
    const id = /^\/api\/conversations\/([^/]+)$/.exec(url.pathname)?.[1], known = id && conversations.get(id)
    if (known && method === 'GET') return route.fulfill({ json: known })
    if (known && method === 'PATCH') {
      const { messages = [], activate: _activate, ...patch } = request.postDataJSON()
      const next: Conversation = { ...known, ...patch, updated_at: new Date().toISOString(),
        messages: [...new Map([...known.messages, ...messages].map(message => [message.id, message])).values()] }
      next.message_count = next.messages.length
      conversations.set(next.id, next)
      return route.fulfill({ json: next })
    }
    if (url.pathname === '/api/request-history/dates' && method === 'GET') return route.fulfill({ json: { current_run_id: 'tools-run', today: new Date().toLocaleDateString('en-CA'), dates: [], issues: [] } })
    if (url.pathname === '/api/request-history' && method === 'GET') return route.fulfill({ json: historyPage([], url.searchParams) })
    if (url.pathname === '/api/request-history/usage' && method === 'GET') return route.fulfill({ json: historyUsage() })
    const callId = new RegExp(`^/api/instances/${instance.id}/request-metrics/([^/]+)$`).exec(url.pathname)?.[1]
    if (callId && method === 'GET' && reports.has(callId)) return route.fulfill({ json: reports.get(callId) })
    fixtureErrors.push(`Unexpected fixture request: ${method} ${url.pathname}`)
    return route.fulfill({ status: 404, json: { detail: fixtureErrors.at(-1) } })
  })
  await page.route(`**/api/instances/${instance.id}/telemetry`, route => {
    samples++
    const running = !!pass && !pass.completed
    return route.fulfill({ json: { instance_id: instance.id, backend: instance.backend, state: instance.state, updated_at: Date.now() / 1000,
      live_generation: { request_key: running ? pass!.key : null, client_request_id: running ? pass!.clientRequestId : null, client_call_id: running ? pass!.clientCallId : null,
        phase: running ? 'thinking' : 'idle', backend_running: running ? 1 : 0, backend_queued: 0,
        tokens_per_second: running ? 140 : null, average_tokens_per_second: running ? 132 : null,
        average_basis: running ? 'backend_decode' : null, window_s: running ? 1 : null,
        context_tokens: running ? 144 : null, context_capacity: 32768, prompt_tokens: running ? 120 : null,
        generated_tokens: running ? 24 : 0, first_output_s: running ? .1 : null, request_elapsed_s: running ? .2 : null },
      groups: pass?.completed ? [{ id: 'api_request', title: '最近请求', request_key: pass.key,
        client_request_id: pass.clientRequestId, client_call_id: pass.clientCallId, request_status: 'completed', metrics: [
          { id: 'output_speed', label: '解码速度', value: 132, unit: 'token/s', kind: 'gauge', rate_basis: 'decode' },
          { id: 'prompt_tokens', label: '输入', value: 120, kind: 'gauge' },
          { id: 'completion_tokens', label: '输出', value: 24, kind: 'gauge' },
          { id: 'first_output_s', label: '首输出等待', value: .1, unit: 's', kind: 'gauge' },
          { id: 'duration_s', label: '请求耗时', value: .2, unit: 's', kind: 'gauge' },
        ] }] : [], unavailable: [] } })
  })
  await page.route('**/api/chat', async route => {
    const body = route.request().postDataJSON()
    const clientRequestId = route.request().headers()['x-modloader-request-id']
    const clientCallId = route.request().headers()['x-modloader-call-id']
    if (!clientRequestId || !clientCallId) {
      fixtureErrors.push('模型请求缺少 X-Modloader-Request-Id 或 X-Modloader-Call-Id')
      await route.fulfill({ status: 500, json: { detail: fixtureErrors.at(-1) } }); return
    }
    modelRequests.push({ clientRequestId, body })
    pass = { key: `monitor-tools-pass-${modelRequests.length}`, clientRequestId, clientCallId, completed: false }
    instance.active_requests = 1
    const last = body.messages.at(-1)
    if (last.role === 'user') await completeModel(route, { tool_calls: [{ index: 0, ...toolCall('fixture-search', 'search_web', { query: '工具连续性网页' }) }] }, 'tool_calls')
    else if (last.tool_call_id === 'fixture-search') await completeModel(route, { tool_calls: [{ index: 0, ...toolCall('fixture-read', 'read_page', { url: sourceUrl, start_line: 1, line_count: 1 }) }] }, 'tool_calls')
    else if (last.tool_call_id === 'fixture-read') await completeModel(route, { tool_calls: [{ index: 0, ...toolCall('fixture-images', 'search_images', { query: '工具连续性图片' }) }] }, 'tool_calls')
    else if (last.tool_call_id === 'fixture-images') held.set('answer', route)
    else {
      fixtureErrors.push(`未预期的模型工具结果：${last.tool_call_id || last.role}`)
      await route.fulfill({ status: 500, json: { detail: fixtureErrors.at(-1) } })
    }
  })
  await page.route('**/api/web-search', route => {
    const body = route.request().postDataJSON(), name = body.category === 'images' ? 'search_images' : 'search_web'
    toolRequests.push({ name, body }); held.set(name, route)
  })
  await page.route('**/api/web-fetch', route => {
    toolRequests.push({ name: 'read_page', body: route.request().postDataJSON() }); held.set('read_page', route)
  })
  const navigation = page.getByRole('navigation', { name: '主导航' })
  const chat = () => navigation.getByRole('button', { name: '测试对话', exact: true }).click()
  const monitor = page.locator('.monitor-page'), table = monitor.locator('.monitor-request-table')
  const parent = table.locator('tr[data-group-key]').first(), state = parent.locator('.monitor-ledger-dot')
  const refresh = monitor.getByRole('button', { name: '刷新到最新', exact: true })
  const assistant = () => page.locator('.chat-message.assistant').last()
  const send = async (text: string) => {
    await page.getByRole('textbox', { name: '对话消息', exact: true }).fill(text)
    await page.getByRole('button', { name: '发送消息', exact: true }).click()
  }
  const expectTool = async (name: 'search_web' | 'read_page' | 'search_images') => {
    await expect.poll(() => held.has(name)).toBe(true)
    expect(pass?.completed).toBe(true)
    expect(instance.active_requests).toBe(0)
    await navigation.getByRole('button', { name: '运行监控', exact: true }).click()
    await expect(page.locator('.chat-section')).toHaveAttribute('hidden', '')
    await expect(page.locator('.chat-section .chat-layout')).toHaveCount(1)
    await refresh.click()
    await expect(state).toHaveAttribute('data-status', 'waiting')
    await expect(state).toHaveAttribute('aria-label', '等待响应')
    await expect(state).toHaveCSS('background-color', 'rgb(71, 104, 242)')
    if (await parent.getAttribute('data-expanded') !== 'true') await parent.locator('td').last().click()
    const action = { search_web: '搜索网页', read_page: '读取网页', search_images: '关键词搜图' }[name]
    await expect(monitor.locator('.monitor-tool-readout').filter({ hasText: action }).locator('.monitor-tool-status')).toHaveAttribute('data-status', 'running')
    // Repeated completed telemetry must not terminate the still-running chat reply.
    const before = samples
    await expect.poll(() => samples).toBeGreaterThanOrEqual(before + 2)
    await expect(state).toHaveAttribute('data-status', 'waiting')
    await expect(monitor.locator('.monitor-tool-readout').filter({ hasText: action }).locator('.monitor-tool-status')).toHaveAttribute('data-status', 'running')
    await page.screenshot({ path: path.join(directory, `monitor-${toolRequests.length}-${name}.png`) })
  }
  try {
    page.setDefaultTimeout(8000)
    await page.goto(`http://127.0.0.1:${address.port}`)
    await pinNavigation(page)
    await expect(page.getByRole('status', { name: '服务在线', exact: true })).toBeVisible({ timeout: 60_000 })
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await chat()
    await expect(page.getByRole('button', { name: '联网搜索', exact: true })).toHaveAttribute('aria-pressed', 'true')
    await send('搜索网页、读取正文，再搜索图片。')
    await expect(assistant().locator('.message-label')).toContainText('正在搜索')
    await expectTool('search_web')
    await releaseTool('search_web', searchResult)
    await expectTool('read_page')
    await releaseTool('read_page', webPage)
    await expectTool('search_images')
    await releaseTool('search_images', imageResult)

    await expect.poll(() => held.has('answer')).toBe(true)
    await refresh.click()
    await expect(state).toHaveAttribute('data-status', 'running')
    await expect(state).toHaveAttribute('aria-label', '进行中')
    await expect(monitor.locator('.monitor-compact-identity')).toContainText('进行中')
    await expect(monitor.locator('.monitor-speed-current strong')).toHaveText('140.00')
    await page.screenshot({ path: path.join(directory, 'monitor-resumed-generation.png') })
    const final = held.get('answer')!; held.delete('answer')
    await completeModel(final, { content: answer })
    await expect(state).toHaveAttribute('data-status', 'completed')
    await expect(state).toHaveAttribute('aria-label', '已完成')
    await expect(state).toHaveCSS('background-color', 'rgb(19, 174, 131)')
    await refresh.click()
    await expect(state).toHaveAttribute('data-status', 'completed')
    await page.screenshot({ path: path.join(directory, 'monitor-completed.png') })
    await chat()
    await expect(assistant().locator('.message-text')).toHaveText(answer)
    await expect(page.getByRole('button', { name: '发送消息', exact: true })).toBeVisible()
    expect(modelRequests).toHaveLength(4)
    expect(new Set(modelRequests.map(request => request.clientRequestId)).size).toBe(1)
    expect(toolRequests).toEqual([
      { name: 'search_web', body: { query: searchResult.query } },
      { name: 'read_page', body: { url: sourceUrl, start_line: 1, line_count: 1 } },
      { name: 'search_images', body: { query: imageResult.query, category: 'images' } },
    ])

    await send('在工具等待期间停止。')
    await expectTool('search_web')
    const stoppedPass = pass!
    expect(modelRequests[4].clientRequestId).not.toBe(modelRequests[0].clientRequestId)
    await chat()
    await page.getByRole('button', { name: '停止', exact: true }).click()
    await expect(assistant().locator('.message-label')).toContainText('已停止')
    await expect(page.getByRole('button', { name: '发送消息', exact: true })).toBeVisible()
    await navigation.getByRole('button', { name: '运行监控', exact: true }).click()
    await refresh.click()
    await expect(state).toHaveAttribute('data-status', 'cancelled')
    await expect(state).toHaveAttribute('aria-label', '已中止')
    // Leave the last model record completed: only Chat knows this reply was stopped.
    expect(pass).toBe(stoppedPass); expect(pass?.completed).toBe(true)
    const afterStop = samples
    await expect.poll(() => samples).toBeGreaterThanOrEqual(afterStop + 2)
    await expect(state).toHaveAttribute('data-status', 'cancelled')
    await expect(state).not.toHaveAttribute('data-status', 'waiting')
    expect(modelRequests).toHaveLength(5)
    await page.screenshot({ path: path.join(directory, 'monitor-stopped-tool.png') })
    expect(errors).toEqual([]); expect(fixtureErrors).toEqual([]); expect(mutations).toEqual([]); expect(external).toEqual([])
    console.log(JSON.stringify({ directory, samples, modelCalls: modelRequests.length, toolCalls: toolRequests.length, errors, fixtureErrors, mutations, external }))
  } finally {
    if (testInfo.status !== testInfo.expectedStatus && !page.isClosed()) await page.screenshot({ path: path.join(directory, 'failure.png') }).catch(() => undefined)
    for (const route of held.values()) await route.abort().catch(() => undefined)
    await page.unrouteAll({ behavior: 'wait' })
    await browser.close()
    await new Promise<void>((resolve, reject) => server.httpServer.close(error => error ? reject(error) : resolve()))
  }
})

