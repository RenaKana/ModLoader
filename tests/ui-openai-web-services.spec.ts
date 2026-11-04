import { test, expect, _electron as electron, type ElectronApplication, type Page } from '@playwright/test'
import { createServer } from 'node:http'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { pinNavigation } from './navigation'

const root = path.resolve(__dirname, '..')
const artifacts = process.env.MODLOADER_TEST_ARTIFACTS || path.join(root, '.artifacts/codex/20261029-routing-dev/ui')
const dummyKey = 'openai-ui-fixture-not-a-real-key'
const sourceUrl = 'https://docs.python.org/3/library/pathlib.html'
const searchText = `Path.read_text() 可读取文本。[Python 文档](${sourceUrl})`
const imagePageUrl = 'https://images.example.test/python'
const imageUrls = ['https://images.example.test/one.png', 'https://images.example.test/two.png']

async function management(page: Page, route: string, method = 'GET', body?: unknown) {
  return page.evaluate(async ({ route, method, body }) => {
    const { baseUrl, token } = await window.modloader!.connection()
    const response = await fetch(baseUrl + route, { method, headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) })
    if (!response.ok) throw new Error(`${route}: ${response.status} ${await response.text()}`)
    return response.json()
  }, { route, method, body })
}

test('OpenAI：配置保存顺序、Responses 网页和关键词搜图、两种聊天模式与来源持久化', async ({}, testInfo) => {
  test.setTimeout(150_000)
  fs.mkdirSync(artifacts, { recursive: true })
  const directory = fs.mkdtempSync(path.join(artifacts, 'openai-'))
  const data = path.join(directory, 'data')
  const upstream: Array<{ path: string; body: any; authorized: boolean }> = []
  const rendererErrors: string[] = [], modelRequests: any[] = []
  let unauthorized = false, holdModelSave = false, releaseSave: (() => void) | undefined
  let mode: 'agent' | 'retrieve' = 'agent'
  let delayedModelField = 'openai_search_model'
  let imageSearch = false
  let application: ElectronApplication | undefined
  const server = createServer(async (request, response) => {
    let raw = ''; for await (const chunk of request) raw += chunk
    const body = JSON.parse(raw)
    upstream.push({ path: request.url || '', body, authorized: request.headers.authorization === `Bearer ${dummyKey}` })
    const images = body.tools?.[0]?.search_content_types?.includes('image')
    response.writeHead(unauthorized ? 401 : 200, { 'Content-Type': 'application/json' })
    response.end(JSON.stringify(unauthorized ? { error: { message: `private-upstream-body ${dummyKey}` } } : {
      id: 'resp_fixture', status: 'completed', output: images ? [
        { id: 'ws_images_fixture', type: 'web_search_call', status: 'completed', results: imageUrls.map((url, index) => ({
          type: 'image_result', image_url: url, source_website_url: imagePageUrl, caption: `Python 图片 ${index + 1}`,
          ...(index === 0 ? { thumbnail_url: url } : {}),
        })) },
      ] : [
        { id: 'ws_fixture', type: 'web_search_call', status: 'completed', action: { type: 'search', queries: ['Python pathlib'], sources: [
          { type: 'url', url: sourceUrl, title: 'Python pathlib 文档' },
          { type: 'url', url: 'https://docs.python.org/3/library/io.html', title: 'Python io 文档' },
        ] } },
        { type: 'message', role: 'assistant', status: 'completed', content: [{ type: 'output_text', text: searchText, annotations: [
          { type: 'url_citation', url: sourceUrl, title: 'Python pathlib 文档', start_index: searchText.indexOf('[Python'), end_index: searchText.length },
        ] }] },
      ],
    }))
  })
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const baseUrl = `http://127.0.0.1:${(server.address() as { port: number }).port}/v1`
  try {
    application = await electron.launch({ args: [root], env: { ...process.env, MODLOADER_DATA_DIR: data, MODLOADER_DEV_URL: process.env.MODLOADER_DEV_URL || '', MODLOADER_SENSOR_AUTO_START: '0' }, timeout: 60_000 })
    const page = await application.firstWindow()
    page.on('pageerror', error => rendererErrors.push(error.message))
    await expect(page.getByRole('status', { name: '服务在线', exact: true })).toBeVisible({ timeout: 45_000 })
    await management(page, '/api/settings', 'PATCH', { proxy_mode: 'direct' })
    await page.route('**/api/settings', async route => {
      const body = route.request().postDataJSON()
      if (holdModelSave && route.request().method() === 'PATCH' && Object.hasOwn(body || {}, delayedModelField)) {
        await new Promise<void>(resolve => { releaseSave = resolve })
      }
      await route.continue()
    })
    await pinNavigation(page)
    const settings = async () => (await management(page, '/api/state')).settings
    const openSettings = () => page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '设置', exact: true }).click()
    await openSettings()
    const card = page.getByRole('region', { name: '联网服务', exact: true })
    await expect(card.locator('details.web-routing-advanced')).toHaveCount(0)
    const search = card.getByRole('combobox', { name: '网页搜索', exact: true })
    await search.selectOption('openai')
    await expect(search).toHaveValue('openai')
    await expect(card.getByRole('combobox', { name: '关键词搜图', exact: true }).locator('option[value="openai"]')).toHaveCount(1)
    await expect(card.getByRole('combobox', { name: '网页读取', exact: true }).locator('option[value="openai"]')).toHaveCount(1)
    const panel = card.getByRole('tabpanel', { name: 'OpenAI', exact: true })
    const address = panel.getByLabel('OpenAI 地址', { exact: true })
    const model = panel.getByLabel('OpenAI 搜索模型', { exact: true })
    const key = panel.getByLabel('OpenAI API Key', { exact: true })
    const imageModel = panel.getByLabel('OpenAI 搜图模型', { exact: true })
    const button = panel.getByRole('button', { name: '测试 OpenAI 网页搜索', exact: true })
    await expect(address).toHaveValue('https://api.openai.com/v1')
    await expect(model).toHaveValue('gpt-4.1-mini')
    await expect(imageModel).toHaveValue('gpt-5.4-mini')
    await address.fill(baseUrl); await address.press('Enter')
    await expect.poll(async () => (await settings()).openai_search_url).toBe(baseUrl)
    await key.fill(dummyKey); await key.press('Enter')
    await expect.poll(async () => (await settings()).web_credentials.search_openai).toBe(true)
    await expect(key).toHaveValue('')
    expect(fs.readFileSync(path.join(data, 'credentials.dpapi.json'), 'utf8')).not.toContain(dummyKey)

    holdModelSave = true
    await model.fill('fixture-custom-search-model')
    await button.click()
    await expect.poll(() => Boolean(releaseSave)).toBe(true)
    await expect(button).toBeDisabled()
    await expect(model).toBeDisabled()
    expect(upstream).toHaveLength(0)
    holdModelSave = false; releaseSave!(); releaseSave = undefined
    await expect(panel.getByRole('status')).toContainText('测试成功')
    expect(upstream).toHaveLength(1)
    expect(upstream[0]).toMatchObject({ path: '/v1/responses', authorized: true, body: { model: 'fixture-custom-search-model', store: false, tools: [{ type: 'web_search' }], tool_choice: 'required', include: ['web_search_call.action.sources'] } })
    expect(await settings()).toMatchObject({ web_image_provider: 'searxng', web_fetch_provider: 'local' })
    await panel.scrollIntoViewIfNeeded()
    await page.screenshot({ path: path.join(directory, 'openai-settings.png') })

    await model.fill(''); await model.press('Enter')
    await expect(page.getByRole('alert').filter({ hasText: /模型/ })).toBeVisible()
    await button.click()
    await expect(panel.getByRole('alert')).toContainText('请先修正并保存此渠道的配置')
    expect(upstream).toHaveLength(1)
    await model.fill('fixture-custom-search-model'); await model.press('Enter')
    await expect.poll(async () => (await settings()).openai_search_model).toBe('fixture-custom-search-model')
    unauthorized = true
    await button.click()
    await expect(panel.getByRole('alert')).toContainText('HTTP 401')
    expect(await panel.innerText()).not.toContain('private-upstream-body')
    expect(await panel.innerText()).not.toContain(dummyKey)
    unauthorized = false
    expect((await settings()).web_search_provider).toBe('searxng')
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '模型库', exact: true }).click()
    await expect.poll(async () => (await settings()).web_search_provider).toBe('openai')
    await page.reload(); await pinNavigation(page); await openSettings()
    await expect(search).toHaveValue('openai')
    await card.getByRole('tab', { name: 'OpenAI', exact: true }).click()
    await expect(address).toHaveValue(baseUrl)
    await expect(model).toHaveValue('fixture-custom-search-model')
    await expect(key).toHaveValue('')
    await expect(panel.getByRole('button', { name: '清除', exact: true })).toBeVisible()
    const contentWidth = await application.evaluate(({ BrowserWindow }) => {
      const window = BrowserWindow.getAllWindows()[0]
      window.setSize(940, 800); window.webContents.setZoomFactor(1.2)
      return window.getContentBounds().width
    })
    await expect.poll(async () => Math.abs(await page.evaluate(() => window.innerWidth) - contentWidth / 1.2)).toBeLessThanOrEqual(1)
    await model.scrollIntoViewIfNeeded()
    await expect(model).toBeInViewport({ ratio: 1 })
    await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))))
    const png = await application.evaluate(async ({ BrowserWindow }) => (await BrowserWindow.getAllWindows()[0].webContents.capturePage()).toPNG().toString('base64'))
    fs.writeFileSync(path.join(directory, 'openai-settings-120.png'), Buffer.from(png, 'base64'))
    await application.evaluate(({ BrowserWindow }) => { const window = BrowserWindow.getAllWindows()[0]; window.setSize(1280, 900); window.webContents.setZoomFactor(1) })

    await page.route('**/api/state', async route => {
      const response = await route.fetch(), state = await response.json()
      await route.fulfill({ response, json: { ...state, instances: [{ id: 'openai-local-fixture', profile_id: 'fixture-profile', model_id: 'fixture-asset', alias: '联网测试模型', backend: 'llama_cpp', state: 'ready', capabilities: { input: ['text'], tools: true }, active_requests: 0 }] } })
    })
    await page.route('**/api/instances/openai-local-fixture/telemetry', route => route.fulfill({ json: { groups: [], unavailable: [], live_generation: null } }))
    for (const url of imageUrls) await page.route(url, route => route.fulfill({ contentType: 'image/png', body: fs.readFileSync(path.join(root, 'electron/icon.png')) }))
    await page.route('**/api/chat', async route => {
      const body = route.request().postDataJSON(); modelRequests.push(body)
      const system = body.messages.filter((message: any) => message.role === 'system').map((message: any) => message.content).join('\n')
      const last = body.messages.at(-1)
      let message: any, reason = 'stop'
      if (mode === 'agent' && last.role === 'user') {
        message = { content: null, tool_calls: [{ id: 'openai-search', type: 'function', function: { name: imageSearch ? 'search_images' : 'search_web', arguments: JSON.stringify({ query: imageSearch ? 'Python 图片' : 'Python pathlib documentation' }) } }] }; reason = 'tool_calls'
      } else if (mode === 'retrieve' && system.includes('拟定联网检索查询')) message = { content: JSON.stringify(imageSearch ? { queries: [], image_queries: ['Python 图片'] } : { queries: ['Python pathlib documentation'] }) }
      else if (mode === 'retrieve' && system.includes('现在只选择需要进一步阅读的来源')) message = { content: JSON.stringify({ read_source_ids: [] }) }
      else {
        const evidence = mode === 'agent' ? JSON.parse(last.content) : null
        if (imageSearch) {
          if (evidence) expect(evidence.images).toMatchObject([{ source_id: 1, url: imageUrls[0] }, { source_id: 1, url: imageUrls[1] }])
          else for (const url of imageUrls) expect(JSON.stringify(last.content)).toContain(url)
          message = { content: `找到两张 Python 图片。\n\n![图片一](${imageUrls[0]})\n\n![图片二](${imageUrls[1]})\n\n[1](${imagePageUrl})` }
        } else {
          if (evidence) expect(evidence.summaries[0]).toMatchObject({ source_ids: [1] })
          else expect(JSON.stringify(last.content)).toContain('summaries')
          message = { content: `使用 Path.read_text() 读取文本。[1](${sourceUrl})` }
        }
      }
      if (!body.stream) { await route.fulfill({ json: { choices: [{ index: 0, message: { role: 'assistant', ...message }, finish_reason: reason }] } }); return }
      const delta = message.tool_calls ? { tool_calls: message.tool_calls.map((call: any, index: number) => ({ index, ...call })) } : message
      await route.fulfill({ contentType: 'text/event-stream', body: `data: ${JSON.stringify({ choices: [{ index: 0, delta, finish_reason: reason }] })}\n\ndata: [DONE]\n\n` })
    })
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    const chat = () => page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '测试对话', exact: true }).click()
    await chat()
    for (mode of ['agent', 'retrieve'] as const) {
      if (mode === 'retrieve') {
        await page.getByRole('button', { name: '联网搜索', exact: true }).click()
        await page.getByRole('dialog', { name: '联网搜索选项' }).getByRole('radio', { name: '检索后回答', exact: true }).click()
      }
      const before = upstream.length
      const replies = await page.locator('.chat-message.assistant').count()
      await page.getByRole('textbox', { name: '对话消息', exact: true }).fill(`${mode}：如何读取文本？`)
      await expect(page.getByRole('button', { name: '发送消息', exact: true })).toBeEnabled()
      await page.getByRole('button', { name: '发送消息', exact: true }).click()
      await expect(page.locator('.chat-message.assistant')).toHaveCount(replies + 1)
      const assistant = page.locator('.chat-message.assistant').last()
      await expect(assistant.locator('.message-text')).toContainText('Path.read_text()')
      await expect(page.getByRole('button', { name: '发送消息', exact: true })).toBeVisible()
      await expect(assistant.locator('.message-text').getByRole('link', { name: '1', exact: true })).toHaveAttribute('href', sourceUrl)
      await expect(assistant.locator('.message-text sup')).toHaveCount(0)
      await assistant.locator('.web-search-sources summary').click()
      await expect(assistant.locator('.web-search-result-list li')).toHaveCount(2)
      await expect(assistant.locator('.web-search-domain').first()).toContainText('来源')
      await expect(assistant.locator('.web-page-excerpt')).toHaveCount(0)
      expect(upstream).toHaveLength(before + 1)
      await expect(page.locator('.chat-save-status').filter({ hasText: /^已保存$/ })).toBeVisible()
      const listing = await management(page, '/api/conversations')
      const saved = await management(page, `/api/conversations/${listing.active_id}`)
      expect(saved.messages.at(-1).webSearch).toMatchObject({ mode, sources: [{ id: 1, snippet: '' }, { id: 2, snippet: '' }], summaries: [{ source_ids: [1] }], pages: [] })
      expect(saved.messages.at(-1).webSearch.summaries[0].text).toContain('Path.read_text()')
      await page.screenshot({ path: path.join(directory, `openai-chat-${mode}.png`) })
    }
    await openSettings()
    await card.getByRole('combobox', { name: '关键词搜图', exact: true }).selectOption('openai')
    await expect(card.getByRole('combobox', { name: '关键词搜图', exact: true })).toHaveValue('openai')
    const imageTestCount = upstream.length
    delayedModelField = 'openai_image_model'; holdModelSave = true
    await imageModel.fill('fixture-custom-image-model')
    await panel.getByRole('button', { name: '测试 OpenAI 关键词搜图', exact: true }).click()
    await expect.poll(() => Boolean(releaseSave)).toBe(true)
    expect(upstream).toHaveLength(imageTestCount)
    holdModelSave = false; releaseSave!(); releaseSave = undefined
    await expect(panel.getByRole('status')).toContainText('测试成功 · 关键词搜图 · 2 条结果')
    expect(upstream.at(-1)!.body.model).toBe('fixture-custom-image-model')
    expect(upstream.at(-1)!.body.instructions).toContain('图片')
    expect((await settings()).openai_search_model).toBe('fixture-custom-search-model')
    expect(upstream).toHaveLength(imageTestCount + 1)
    expect(upstream.at(-1)!.body.tools[0]).toMatchObject({ type: 'web_search', search_content_types: ['image', 'text'], image_settings: { caption: true } })
    expect(upstream.at(-1)!.body.include).toContain('web_search_call.results')
    expect(await settings()).toMatchObject({ web_search_provider: 'openai', web_image_provider: 'searxng', web_fetch_provider: 'local' })
    await panel.scrollIntoViewIfNeeded()
    await page.screenshot({ path: path.join(directory, 'openai-image-settings.png') })
    await chat()
    await expect.poll(settings).toMatchObject({ web_search_provider: 'openai', web_image_provider: 'openai', web_fetch_provider: 'local' })
    imageSearch = true
    for (mode of ['agent', 'retrieve'] as const) {
      await page.getByRole('button', { name: '联网搜索', exact: true }).click()
      await page.getByRole('dialog', { name: '联网搜索选项' }).getByRole('radio', { name: mode === 'agent' ? '自主搜索' : '检索后回答', exact: true }).click()
      const before = upstream.length
      const replies = await page.locator('.chat-message.assistant').count()
      await page.getByRole('textbox', { name: '对话消息', exact: true }).fill(`${mode}：搜索两张 Python 图片`)
      await expect(page.getByRole('button', { name: '发送消息', exact: true })).toBeEnabled()
      await page.getByRole('button', { name: '发送消息', exact: true }).click()
      await expect(page.locator('.chat-message.assistant')).toHaveCount(replies + 1)
      const assistant = page.locator('.chat-message.assistant').last()
      await expect(assistant.locator('.message-text')).toContainText('找到两张 Python 图片')
      await expect(assistant.locator('.message-text img')).toHaveCount(2)
      await expect.poll(() => assistant.locator('.message-text img').evaluateAll(images => images.every(image => (image as HTMLImageElement).complete && (image as HTMLImageElement).naturalWidth > 0))).toBe(true)
      await expect(assistant.locator('.message-text').getByRole('link', { name: '1', exact: true })).toHaveAttribute('href', imagePageUrl)
      await expect(assistant.locator('.web-search-sources summary')).toContainText('图片 2')
      expect(upstream).toHaveLength(before + 1)
      await expect(page.locator('.chat-save-status').filter({ hasText: /^已保存$/ })).toBeVisible()
      const listing = await management(page, '/api/conversations')
      const saved = await management(page, `/api/conversations/${listing.active_id}`)
      expect(saved.messages.at(-1).webSearch).toMatchObject({ mode, sources: [{ id: 1, url: imagePageUrl }], images: [{ source_id: 1, url: imageUrls[0] }, { source_id: 1, url: imageUrls[1] }], pages: [] })
      await page.screenshot({ path: path.join(directory, `openai-images-${mode}.png`) })
    }
    await page.reload(); await pinNavigation(page); await chat()
    await expect(page.locator('.chat-message.assistant').last().locator('.message-text')).toContainText('找到两张 Python 图片')
    await expect(page.locator('.chat-message.assistant').last().locator('.message-text img')).toHaveCount(2)
    expect(rendererErrors).toEqual([])
    expect(modelRequests).toHaveLength(10)
    for (const name of ['openai-settings.png', 'openai-settings-120.png', 'openai-chat-agent.png', 'openai-chat-retrieve.png', 'openai-image-settings.png', 'openai-images-agent.png', 'openai-images-retrieve.png']) await testInfo.attach(name, { path: path.join(directory, name), contentType: 'image/png' })
  } finally {
    releaseSave?.()
    await application?.close()
    server.closeAllConnections()
    await new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve()))
  }
})
