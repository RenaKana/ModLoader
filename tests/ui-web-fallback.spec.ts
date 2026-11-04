import { test, expect, _electron as electron, type ElectronApplication, type Page } from '@playwright/test'
import { createServer } from 'node:http'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { pinNavigation } from './navigation'
import { conversationMarkdown } from '../src/chat-export'

const root = path.resolve(__dirname, '..')
const artifacts = process.env.MODLOADER_TEST_ARTIFACTS || path.join(root, '.artifacts/codex/20261029-routing-dev/ui')
const sourceUrl = 'https://docs.python.org/3/library/pathlib.html'
async function management(page: Page, route: string, method = 'GET', body?: unknown) {
  return page.evaluate(async ({ route, method, body }) => {
    const { baseUrl, token } = await window.modloader!.connection()
    const response = await fetch(baseUrl + route, { method, headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) })
    if (!response.ok) throw new Error(`${route}: ${response.status} ${await response.text()}`)
    return response.json()
  }, { route, method, body })
}

test('顺位设置、真实后端切换、阅读摘要两种聊天路径与历史/导出', async ({}, testInfo) => {
  test.setTimeout(150_000)
  fs.mkdirSync(artifacts, { recursive: true })
  const directory = fs.mkdtempSync(path.join(artifacts, 'fallback-'))
  const requests: Array<{ path: string; body: any }> = []
  const errors: string[] = []
  let application: ElectronApplication | undefined
  let mode: 'agent' | 'retrieve' = 'agent'
  let holdOpenAI = false, releaseOpenAI: (() => void) | undefined
  let holdReadSave = false, releaseReadSave: (() => void) | undefined
  const server = createServer(async (request, response) => {
    let raw = ''; for await (const chunk of request) raw += chunk
    const body = JSON.parse(raw); requests.push({ path: request.url!, body })
    if (!request.url?.endsWith('/responses')) {
      response.writeHead(503, { 'Content-Type': 'application/json' }); response.end('{}'); return
    }
    if (holdOpenAI) await new Promise<void>(resolve => { releaseOpenAI = resolve })
    const reading = body.model === 'fixture-read-model'
    const text = reading ? `pathlib 提供面向对象的文件系统路径。\n\n- Path.read_text() 返回文本。\n- Path.read_bytes() 返回字节。\n\n[Python 文档](${sourceUrl})` : 'Python pathlib 检索摘要'
    response.writeHead(200, { 'Content-Type': 'application/json' })
    response.end(JSON.stringify({ status: 'completed', output: [
      { type: 'web_search_call', status: 'completed', action: reading ? { type: 'open_page', url: sourceUrl } : { type: 'search', sources: [{ type: 'url', url: sourceUrl, title: 'Python pathlib 文档' }] } },
      { type: 'message', role: 'assistant', status: 'completed', content: [{ type: 'output_text', text, annotations: [{ type: 'url_citation', url: sourceUrl, title: 'Python pathlib 文档', start_index: text.indexOf('[Python'), end_index: text.length }] }] },
    ] }))
  })
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const base = `http://127.0.0.1:${(server.address() as { port: number }).port}`
  try {
    application = await electron.launch({ args: [root], env: { ...process.env, MODLOADER_DATA_DIR: path.join(directory, 'data'), MODLOADER_DEV_URL: process.env.MODLOADER_DEV_URL || '', MODLOADER_SENSOR_AUTO_START: '0' }, timeout: 60_000 })
    const page = await application.firstWindow()
    page.on('pageerror', error => errors.push(error.message))
    await expect(page.getByRole('status', { name: '服务在线', exact: true })).toBeVisible({ timeout: 45_000 })
    await management(page, '/api/settings', 'PATCH', { proxy_mode: 'direct', deepseek_url: base, openai_search_url: base + '/v1', secrets: { search_deepseek: 'fixture-key-deepseek', search_openai: 'fixture-key-openai' } })
    await pinNavigation(page)
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '设置', exact: true }).click()
    const card = page.getByRole('region', { name: '联网服务', exact: true })
    const settings = async () => (await management(page, '/api/state')).settings
    await expect(card.locator('details.web-routing-advanced')).toHaveCount(0)
    for (const [label, primary] of [['网页搜索', 'deepseek'], ['关键词搜图', 'searxng'], ['网页读取', 'deepseek']]) {
      await card.getByRole('combobox', { name: label, exact: true }).selectOption(primary)
      await card.getByRole('button', { name: `添加${label}备用渠道`, exact: true }).click()
      await card.getByRole('combobox', { name: `${label}备用渠道 1`, exact: true }).selectOption('openai')
      await card.getByRole('button', { name: `添加${label}备用渠道`, exact: true }).click()
      await card.getByRole('combobox', { name: `${label}备用渠道 2`, exact: true }).selectOption('tavily')
      await card.getByRole('button', { name: `上移${label}备用渠道 Tavily`, exact: true }).click()
      await expect(card.getByRole('combobox', { name: `${label}备用渠道 1`, exact: true })).toHaveValue('tavily')
      await expect(card.getByRole('combobox', { name: `${label}备用渠道 2`, exact: true })).toHaveValue('openai')
      await card.getByRole('button', { name: `删除${label}备用渠道 Tavily`, exact: true }).click()
      await expect(card.getByRole('combobox', { name: `${label}备用渠道 1`, exact: true })).toHaveValue('openai')
      await expect(card.getByRole('combobox', { name: `${label}备用渠道 2`, exact: true })).toHaveCount(0)
    }
    await card.getByRole('combobox', { name: '网页搜索', exact: true }).selectOption('openai')
    await expect(card.getByRole('combobox', { name: '网页搜索备用渠道 1', exact: true })).toHaveCount(0)
    await card.getByRole('combobox', { name: '网页搜索', exact: true }).selectOption('deepseek')
    await card.getByRole('button', { name: '添加网页搜索备用渠道', exact: true }).click()
    await card.getByRole('combobox', { name: '网页搜索备用渠道 1', exact: true }).selectOption('openai')
    await card.getByRole('tab', { name: 'OpenAI', exact: true }).click()
    const panel = card.getByRole('tabpanel', { name: 'OpenAI', exact: true })
    const readModel = panel.getByLabel('OpenAI 阅读模型', { exact: true })
    await expect(readModel).toHaveValue('gpt-5.4-mini')
    await page.route('**/api/settings', async route => {
      if (holdReadSave && route.request().method() === 'PATCH' && 'openai_read_model' in route.request().postDataJSON()) await new Promise<void>(resolve => { releaseReadSave = resolve })
      await route.continue()
    })
    holdReadSave = true
    await readModel.fill('fixture-read-model')
    const readTest = panel.getByRole('button', { name: '测试 OpenAI 网页读取', exact: true })
    await readTest.click()
    await expect.poll(() => Boolean(releaseReadSave)).toBe(true)
    expect(requests).toHaveLength(0)
    holdReadSave = false; releaseReadSave!(); releaseReadSave = undefined
    await expect(panel.getByRole('status')).toContainText('测试成功 · 阅读摘要')
    expect(requests).toHaveLength(1)
    expect(requests[0]).toMatchObject({ path: '/v1/responses', body: { model: 'fixture-read-model', store: false, reasoning: { effort: 'low' } } })
    expect(await settings()).toMatchObject({ web_search_provider: 'searxng', web_search_fallbacks: [], web_image_fallbacks: [], web_fetch_fallbacks: [] })
    await card.scrollIntoViewIfNeeded()
    await page.screenshot({ path: path.join(directory, 'routes.png') })
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '模型库', exact: true }).click()
    await expect.poll(settings).toMatchObject({ web_search_provider: 'deepseek', web_search_fallbacks: ['openai'], web_image_provider: 'searxng', web_image_fallbacks: ['openai'], web_fetch_provider: 'deepseek', web_fetch_fallbacks: ['openai'] })
    await page.reload(); await pinNavigation(page)
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '设置', exact: true }).click()
    await expect(card.getByRole('combobox', { name: '网页读取备用渠道 1', exact: true })).toHaveValue('openai')
    await expect(card.getByRole('combobox', { name: '网页读取备用渠道 1', exact: true }).locator('option:checked')).toHaveText('OpenAI 阅读摘要')
    await expect(card.getByRole('combobox', { name: '网页搜索', exact: true })).toHaveValue('deepseek')
    const width = await application.evaluate(({ BrowserWindow }) => { const w = BrowserWindow.getAllWindows()[0]; w.setSize(940, 800); w.webContents.setZoomFactor(1.2); return w.getContentBounds().width })
    await expect.poll(async () => Math.abs(await page.evaluate(() => window.innerWidth) - width / 1.2)).toBeLessThanOrEqual(1)
    await card.getByRole('combobox', { name: '网页搜索', exact: true }).scrollIntoViewIfNeeded()
    await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))))
    fs.writeFileSync(path.join(directory, 'routes-120.png'), Buffer.from(await application.evaluate(async ({ BrowserWindow }) => (await BrowserWindow.getAllWindows()[0].webContents.capturePage()).toPNG().toString('base64')), 'base64'))
    await application.evaluate(({ BrowserWindow }) => { const w = BrowserWindow.getAllWindows()[0]; w.setSize(1280, 900); w.webContents.setZoomFactor(1) })
    await page.route('**/api/state', async route => { const response = await route.fetch(), state = await response.json(); await route.fulfill({ response, json: { ...state, instances: [{ id: 'fallback-fixture', profile_id: 'fixture-profile', model_id: 'fixture-model', alias: '联网测试模型', backend: 'llama_cpp', state: 'ready', capabilities: { input: ['text'], tools: true }, active_requests: 0 }] } }) })
    await page.route('**/api/instances/fallback-fixture/telemetry', route => route.fulfill({ json: { groups: [], unavailable: [], live_generation: null } }))
    await page.route('**/api/chat', async route => {
      const body = route.request().postDataJSON(), last = body.messages.at(-1), system = body.messages.filter((item: any) => item.role === 'system').map((item: any) => item.content).join('\n')
      let message: any, reason = 'stop'
      if (mode === 'agent' && last.role === 'user') { message = { content: null, tool_calls: [{ id: 'search', type: 'function', function: { name: 'search_web', arguments: '{"query":"Python pathlib"}' } }] }; reason = 'tool_calls' }
      else if (mode === 'agent' && last.role === 'tool' && JSON.parse(last.content).kind !== 'summary') { message = { content: null, tool_calls: [{ id: 'read', type: 'function', function: { name: 'read_page', arguments: JSON.stringify({ url: sourceUrl }) } }] }; reason = 'tool_calls' }
      else if (mode === 'retrieve' && system.includes('拟定联网检索查询')) message = { content: '{"queries":["Python pathlib"]}' }
      else if (mode === 'retrieve' && system.includes('现在只选择需要进一步阅读的来源')) message = { content: '{"read_source_ids":[1]}' }
      else message = { content: `根据阅读摘要，Path.read_text() 返回文本。[1](${sourceUrl})` }
      if (!body.stream) { await route.fulfill({ json: { choices: [{ message: { role: 'assistant', ...message }, finish_reason: reason }] } }); return }
      const delta = message.tool_calls ? { tool_calls: message.tool_calls.map((call: any, index: number) => ({ index, ...call })) } : message
      await route.fulfill({ contentType: 'text/event-stream', body: `data: ${JSON.stringify({ choices: [{ index: 0, delta, finish_reason: reason }] })}\n\ndata: [DONE]\n\n` })
    })
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    const chat = () => page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '测试对话', exact: true }).click()
    await chat()
    for (mode of ['agent', 'retrieve'] as const) {
      await page.getByRole('button', { name: '联网搜索', exact: true }).click()
      await page.getByRole('dialog', { name: '联网搜索选项' }).getByRole('radio', { name: mode === 'agent' ? '自主搜索' : '检索后回答', exact: true }).click()
      const count = await page.locator('.chat-message.assistant').count(), before = requests.length
      await page.getByRole('textbox', { name: '对话消息', exact: true }).fill(`${mode}: 查找并阅读 pathlib 文档`)
      holdOpenAI = true
      await page.getByRole('button', { name: '发送消息', exact: true }).click()
      await expect(page.locator('.chat-message.assistant')).toHaveCount(count + 1)
      const assistant = page.locator('.chat-message.assistant').last()
      await expect(assistant.locator('.message-label')).toContainText('切换')
      await expect.poll(() => Boolean(releaseOpenAI)).toBe(true)
      holdOpenAI = false; releaseOpenAI!(); releaseOpenAI = undefined
      await expect(assistant.locator('.message-text')).toContainText('根据阅读摘要')
      await expect(page.getByRole('button', { name: '发送消息', exact: true })).toBeVisible()
      await expect(assistant.locator('.web-search-sources summary')).toContainText('阅读摘要 1')
      await assistant.locator('.web-search-sources summary').click()
      await expect(assistant.locator('.web-page-summary')).toContainText('Path.read_bytes()')
      await expect(assistant.locator('.web-page-summary a')).toHaveCount(1)
      await expect(assistant.locator('.web-page-summary a')).toHaveAttribute('href', sourceUrl)
      await expect(assistant.locator('.web-page-excerpt')).toHaveCount(0)
      expect(requests.slice(before).map(item => item.path)).toEqual(['/v1/tools/search', '/v1/responses', '/v1/tools/fetch', '/v1/responses'])
      await expect(page.locator('.chat-save-status').filter({ hasText: /^已保存$/ })).toBeVisible()
      const listing = await management(page, '/api/conversations'), saved = await management(page, `/api/conversations/${listing.active_id}`)
      expect(saved.messages.at(-1).webSearch).toMatchObject({ pages: [], summaries: [{ source_ids: [1] }, { url: sourceUrl, source_ids: [1] }] })
      const tools = saved.messages.at(-1).activity.filter((event: any) => event.kind === 'tool').map((event: any) => JSON.parse(event.result))
      expect(tools.at(-1)).toMatchObject({ kind: 'summary', routing: { provider: 'openai', attempts: [{ provider: 'deepseek', error: expect.stringContaining('503') }, { provider: 'openai' }] } })
      expect(conversationMarkdown(saved)).toContain('阅读摘要')
      expect(conversationMarkdown(saved)).toContain('Path.read_bytes()')
      await page.screenshot({ path: path.join(directory, `reading-${mode}.png`) })
    }
    await page.reload(); await pinNavigation(page); await chat()
    await expect(page.locator('.chat-message.assistant').last().locator('.web-search-sources summary')).toContainText('阅读摘要 1')
    expect(errors).toEqual([])
    for (const name of ['routes.png', 'routes-120.png', 'reading-agent.png', 'reading-retrieve.png']) await testInfo.attach(name, { path: path.join(directory, name), contentType: 'image/png' })
  } finally {
    releaseOpenAI?.(); releaseReadSave?.()
    await application?.close()
    server.closeAllConnections(); await new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve()))
  }
})
