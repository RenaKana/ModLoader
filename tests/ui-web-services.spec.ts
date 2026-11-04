import { test, expect, _electron as electron, type ElectronApplication, type Page, type Request } from '@playwright/test'
import { createServer } from 'node:http'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { pinNavigation } from './navigation'

const root = path.resolve(__dirname, '..')
const artifacts = process.env.MODLOADER_TEST_ARTIFACTS || path.join(root, '.artifacts', 'codex', '20261029-routing-dev', 'ui')
const dummyKey = 'modloader-ui-dummy-deepseek-key-not-a-real-key'
const dummySauceKey = 'modloader-ui-dummy-saucenao-key-not-a-real-key'
const upstreamMarker = 'private-upstream-error-body'
const pageUrl = 'https://web-services.example.test/guide'
const fetchedContent = '# 本地测试网页\nDeepSeek 网页读取经过真实核心路由。\n第三行正文。'
const reverseImage = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII='

async function management<T = any>(page: Page, route: string, method = 'GET', body?: unknown): Promise<T> {
  return page.evaluate(async ({ route, method, body }) => {
    const { baseUrl, token } = await window.modloader!.connection()
    const response = await fetch(baseUrl + route, {
      method, headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    })
    if (!response.ok) throw new Error(`${route}: ${response.status} ${await response.text()}`)
    return response.json()
  }, { route, method, body })
}

function isSettingsPatch(request: Request, secret: string) {
  return new URL(request.url()).pathname === '/api/settings' && request.method() === 'PATCH'
    && request.postDataJSON()?.secrets?.search_deepseek === secret
}

test('联网服务：用途渠道持久化、加密凭据与 DeepSeek 真实搜索/读取测试', async ({}, testInfo) => {
  test.setTimeout(150_000)
  fs.mkdirSync(artifacts, { recursive: true })
  const directory = fs.mkdtempSync(path.join(artifacts, 'settings-'))
  const data = path.join(directory, 'data')
  const upstreamRequests: { method: string; path: string; authorization: string; body: any }[] = []
  const serviceResponses: string[] = []
  const reverseRequests: any[] = []
  const fixtureErrors: string[] = [], rendererErrors: string[] = []
  let unauthorized = false
  let application: ElectronApplication | undefined, page: Page | undefined
  const server = createServer(async (request, response) => {
    let raw = ''; for await (const chunk of request) raw += chunk
    let body: any
    try { body = JSON.parse(raw) }
    catch { fixtureErrors.push(`Invalid upstream JSON: ${request.method} ${request.url}`) }
    const entry = { method: request.method || '', path: request.url || '', authorization: request.headers.authorization || '', body }
    upstreamRequests.push(entry)
    response.setHeader('Content-Type', 'application/json')
    if (request.method !== 'POST' || !['/v1/tools/search', '/v1/tools/fetch'].includes(entry.path)) {
      fixtureErrors.push(`Unexpected upstream request: ${entry.method} ${entry.path}`)
      response.writeHead(404); response.end(JSON.stringify({ error: 'Unexpected test endpoint' })); return
    }
    if (unauthorized) {
      response.writeHead(401)
      response.end(JSON.stringify({ error: { message: `${upstreamMarker}: rejected Bearer ${dummyKey}` } }))
      return
    }
    response.writeHead(200)
    response.end(JSON.stringify(entry.path === '/v1/tools/search' ? { results: [
      { title: '本地搜索结果一', url: pageUrl, summary: '第一条搜索摘要。', snippet: '第一条搜索片段。' },
      { title: '本地搜索结果二', url: 'https://web-services.example.test/reference', summary: '第二条搜索摘要。', snippet: '第二条搜索片段。' },
    ] } : { title: '本地测试网页', content: fetchedContent }))
  })
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const upstreamUrl = `http://127.0.0.1:${(server.address() as { port: number }).port}`
  try {
    application = await electron.launch({
      args: [root],
      env: { ...process.env, MODLOADER_DATA_DIR: data, MODLOADER_DEV_URL: process.env.MODLOADER_DEV_URL || '', MODLOADER_SENSOR_AUTO_START: '0' },
      timeout: 60_000,
    })
    page = await application.firstWindow()
    page.on('pageerror', error => rendererErrors.push(error.message))
    // requestProgress cancels its reader after the terminal event. Capture the
    // genuine core stream before forwarding it so CDP body eviction is irrelevant.
    await page.route('**/api/web-services/test', async route => {
      const response = await route.fetch()
      const body = await response.text()
      serviceResponses.push(body)
      await route.fulfill({ response, body })
    })
    await page.route('**/api/web-reverse-image', async route => {
      reverseRequests.push(route.request().postDataJSON())
      await route.fulfill({ json: { provider: 'saucenao', elapsed_ms: 15, matches: [{ title: '本地反向搜图结果' }] } })
    })
    await expect(page.getByRole('status', { name: '服务在线', exact: true })).toBeVisible({ timeout: 45_000 })
    await management(page, '/api/settings', 'PATCH', { proxy_mode: 'direct' })
    const settings = async () => (await management(page!, '/api/state')).settings
    const openSettings = async () => {
      await pinNavigation(page!)
      await page!.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '设置', exact: true }).click()
    }
    await openSettings()
    const card = page.getByRole('region', { name: '联网服务', exact: true })
    await expect(card.locator('details.web-routing-advanced')).toHaveCount(0)
    const search = card.getByRole('combobox', { name: '网页搜索', exact: true })
    const images = card.getByRole('combobox', { name: '关键词搜图', exact: true })
    const fetch = card.getByRole('combobox', { name: '网页读取', exact: true })
    const tabs = card.getByRole('tablist', { name: '渠道配置', exact: true })
    await expect(card).toBeVisible()
    await expect(search).toBeVisible()
    await expect(images).toBeVisible()
    await expect(fetch).toBeVisible()
    await expect(search).toHaveValue('searxng')
    await expect(images).toHaveValue('searxng')
    await expect(fetch).toHaveValue('local')
    expect(await settings()).toMatchObject({ web_search_provider: 'searxng', web_image_provider: 'searxng', web_fetch_provider: 'local' })
    await expect(tabs.getByRole('tab')).toHaveText(['SearXNG', 'DeepSeek', 'Tavily', 'Brave Search', 'Serper', 'OpenAI', 'SauceNAO'])
    await expect(search.locator('option[value="deepseek"]')).toHaveCount(1)
    await expect(fetch.locator('option[value="deepseek"]')).toHaveCount(1)
    await expect(images.locator('option[value="deepseek"]')).toHaveCount(0)

    // Channels remain selectable before credentials are configured.
    await search.selectOption('tavily')
    await expect(search).toHaveValue('tavily')
    await images.selectOption('brave')
    await expect(images).toHaveValue('brave')
    await fetch.selectOption('deepseek')
    await expect(fetch).toHaveValue('deepseek')
    await tabs.getByRole('tab', { name: 'DeepSeek', exact: true }).click()
    const panel = card.getByRole('tabpanel', { name: 'DeepSeek', exact: true })
    const address = panel.getByLabel('DeepSeek 地址', { exact: true })
    await address.fill(upstreamUrl); await address.press('Enter')
    await expect.poll(async () => (await settings()).deepseek_url).toBe(upstreamUrl)
    const key = panel.getByLabel('DeepSeek API Key', { exact: true })
    await expect(key).toHaveAttribute('type', 'password')
    await key.fill(dummyKey)
    await panel.getByRole('button', { name: '显示 DeepSeek API Key', exact: true }).click()
    await expect(key).toHaveAttribute('type', 'text')
    await expect(key).toHaveValue(dummyKey)
    await panel.getByRole('button', { name: '隐藏 DeepSeek API Key', exact: true }).click()
    await expect(key).toHaveAttribute('type', 'password')
    const savedResponse = page.waitForResponse(response => isSettingsPatch(response.request(), dummyKey))
    await key.press('Enter')
    const saved = await savedResponse
    expect(saved.ok()).toBe(true)
    const savedText = await saved.text()
    expect(savedText).not.toContain(dummyKey)
    expect(JSON.parse(savedText).web_credentials.search_deepseek).toBe(true)
    await expect.poll(async () => (await settings()).web_credentials.search_deepseek).toBe(true)
    await expect(key).toHaveValue('')
    const vaultPath = path.join(data, 'credentials.dpapi.json')
    const vaultText = fs.readFileSync(vaultPath, 'utf8')
    const encrypted = JSON.parse(vaultText).search_deepseek
    expect(typeof encrypted).toBe('string')
    expect(encrypted.length).toBeGreaterThan(0)
    expect(vaultText).not.toContain(dummyKey)
    expect(Buffer.from(encrypted, 'base64').includes(Buffer.from(dummyKey))).toBe(false)

    expect(await settings()).toMatchObject({ web_search_provider: 'searxng', web_image_provider: 'searxng', web_fetch_provider: 'local' })
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '模型库', exact: true }).click()
    await expect.poll(settings).toMatchObject({ web_search_provider: 'tavily', web_image_provider: 'brave', web_fetch_provider: 'deepseek' })
    await page.reload(); await openSettings()
    await expect(search).toHaveValue('tavily')
    await expect(images).toHaveValue('brave')
    await expect(fetch).toHaveValue('deepseek')
    await tabs.getByRole('tab', { name: 'DeepSeek', exact: true }).click()
    await expect(address).toHaveValue(upstreamUrl)
    await expect(key).toHaveValue('')
    await expect(key).toHaveAttribute('type', 'password')
    await expect(panel.getByRole('button', { name: '清除', exact: true })).toBeVisible()
    expect(JSON.stringify(await settings())).not.toContain(dummyKey)

    // Testing a provider must not silently select it for any purpose.
    await fetch.selectOption('local')
    const selected = { web_search_provider: 'tavily', web_image_provider: 'brave', web_fetch_provider: 'local' }
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '模型库', exact: true }).click()
    await expect.poll(settings).toMatchObject(selected)
    await openSettings()
    for (const capability of ['search', 'fetch'] as const) {
      const requestPromise = page.waitForRequest(request => new URL(request.url()).pathname === '/api/web-services/test' && request.method() === 'POST')
      const responsePromise = page.waitForResponse(response => new URL(response.url()).pathname === '/api/web-services/test' && response.request().method() === 'POST')
      await panel.getByRole('button', { name: `测试 DeepSeek ${capability === 'search' ? '网页搜索' : '网页读取'}`, exact: true }).click()
      const request = await requestPromise, response = await responsePromise
      const body = request.postDataJSON()
      expect(body).toMatchObject({ provider: 'deepseek', capability })
      expect(request.headers().accept).toBe('application/x-ndjson')
      expect(typeof body[capability === 'search' ? 'query' : 'url']).toBe('string')
      expect(body[capability === 'search' ? 'query' : 'url'].length).toBeGreaterThan(0)
      expect(response.ok()).toBe(true)
      expect(response.headers()['content-type']).toContain('application/x-ndjson')
      const responseText = serviceResponses.at(-1)!
      expect(responseText).not.toContain(dummyKey)
      const events = responseText.trim().split('\n').map(line => JSON.parse(line))
      expect(events.some(event => event.type === 'error')).toBe(false)
      const value = events.find(event => event.type === 'result')?.value
      if (capability === 'search') expect(value?.sources).toHaveLength(2)
      else expect(value).toMatchObject({ content: fetchedContent, total_lines: 3 })
      const status = panel.getByRole('status').filter({ hasText: '测试成功' })
      await expect(status).toBeVisible()
      await expect(status).toContainText(capability === 'search' ? /2\s*(条|个)/ : /3\s*行/)
      const upstream = upstreamRequests.at(-1)!
      expect(upstream).toMatchObject({ method: 'POST', path: `/v1/tools/${capability}`, authorization: `Bearer ${dummyKey}` })
      expect(upstream.body[capability === 'search' ? 'query' : 'url']).toBe(body[capability === 'search' ? 'query' : 'url'])
      expect(await settings()).toMatchObject(selected)
    }
    expect(upstreamRequests).toHaveLength(2)
    await expect(search).toHaveValue('tavily')
    await expect(images).toHaveValue('brave')
    await expect(fetch).toHaveValue('local')

    // Reverse-image testing uses an intercepted local fixture; it never calls SauceNAO.
    await tabs.getByRole('tab', { name: 'SauceNAO', exact: true }).click()
    const reversePanel = card.getByRole('tabpanel', { name: 'SauceNAO', exact: true })
    const reverseAddress = reversePanel.getByLabel('SauceNAO 地址', { exact: true })
    await expect(reverseAddress).toHaveValue('https://saucenao.com')
    const fixtureUrl = 'https://saucenao.fixture.test'
    await reverseAddress.fill(fixtureUrl)
    await reverseAddress.press('Enter')
    await expect.poll(async () => (await settings()).saucenao_url).toBe(fixtureUrl)
    const reverseKey = reversePanel.getByLabel('SauceNAO API Key', { exact: true })
    await reverseKey.fill(dummySauceKey)
    await reversePanel.getByRole('button', { name: '测试 SauceNAO 以图搜源', exact: true }).click()
    await expect(reversePanel.getByRole('status')).toContainText('测试成功 · 以图搜源 · 1 条匹配')
    expect(reverseRequests).toEqual([{ image: reverseImage, filename: 'modloader-test.png' }])
    expect(await settings()).toMatchObject({ saucenao_url: fixtureUrl, web_credentials: { search_saucenao: true }, ...selected })
    await tabs.getByRole('tab', { name: 'DeepSeek', exact: true }).click()

    unauthorized = true
    const failedResponse = page.waitForResponse(response => new URL(response.url()).pathname === '/api/web-services/test' && response.request().method() === 'POST')
    await panel.getByRole('button', { name: '测试 DeepSeek 网页搜索', exact: true }).click()
    const failed = await failedResponse
    expect(failed.ok()).toBe(true)
    expect(failed.headers()['content-type']).toContain('application/x-ndjson')
    const failedText = serviceResponses.at(-1)!
    const failure = failedText.trim().split('\n').map(line => JSON.parse(line)).find(event => event.type === 'error')
    expect(failure).toMatchObject({ type: 'error', status: 502 })
    expect(failure.detail).toMatch(/HTTP 401/)
    expect(failure.detail).toMatch(/API Key|权限/)
    expect(failedText).not.toContain(dummyKey)
    expect(failedText).not.toContain(upstreamMarker)
    await expect(panel.getByRole('alert')).toContainText(failure.detail)
    await expect(panel).not.toContainText(dummyKey)
    await expect(panel).not.toContainText(upstreamMarker)
    expect(upstreamRequests).toHaveLength(3)
    expect(upstreamRequests.at(-1)!.authorization).toBe(`Bearer ${dummyKey}`)
    expect(await settings()).toMatchObject(selected)

    await panel.getByRole('button', { name: '清除', exact: true }).click()
    const confirm = page.getByRole('dialog', { name: '清除 DeepSeek API Key？', exact: true })
    await expect(confirm).toBeVisible()
    const clearedResponse = page.waitForResponse(response => isSettingsPatch(response.request(), ''))
    await confirm.getByRole('button', { name: '清除凭据', exact: true }).click()
    const cleared = await clearedResponse
    expect(cleared.ok()).toBe(true)
    expect((await cleared.json()).web_credentials.search_deepseek).toBe(false)
    await expect(confirm).toBeHidden()
    await expect.poll(async () => (await settings()).web_credentials.search_deepseek).toBe(false)
    expect(JSON.parse(fs.readFileSync(vaultPath, 'utf8'))).not.toHaveProperty('search_deepseek')
    await page.reload(); await openSettings()
    await tabs.getByRole('tab', { name: 'DeepSeek', exact: true }).click()
    await expect(key).toHaveValue('')
    await expect(panel.getByRole('button', { name: '清除', exact: true })).toHaveCount(0)
    expect((await settings()).web_credentials.search_deepseek).toBe(false)
    expect(await settings()).toMatchObject(selected)

    for (const zoomFactor of [1, 1.2]) {
      const contentWidth = await application.evaluate(({ BrowserWindow }, zoomFactor) => {
        const window = BrowserWindow.getAllWindows()[0]
        if (window.isMaximized()) window.unmaximize()
        window.setSize(940, 800)
        window.webContents.setZoomFactor(zoomFactor)
        return window.getContentBounds().width
      }, zoomFactor)
      await expect.poll(async () => Math.abs(await page!.evaluate(() => window.innerWidth) - contentWidth / zoomFactor)).toBeLessThanOrEqual(1)
      await card.scrollIntoViewIfNeeded()
      await expect.poll(() => page!.locator('.app-header').evaluate(node => node.getAnimations({ subtree: true }).some(animation => animation.playState === 'running'))).toBe(false)
      await expect.poll(() => page!.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
      expect(await card.evaluate(node => node.scrollWidth <= node.clientWidth)).toBe(true)
      await expect(tabs).toHaveCSS('overflow-y', 'hidden')
      expect(await tabs.evaluate(element => {
        const node = element as HTMLElement, style = getComputedStyle(node)
        return node.offsetWidth - node.clientWidth - parseFloat(style.borderLeftWidth) - parseFloat(style.borderRightWidth)
      })).toBe(0)
      for (const control of [search, images, fetch, address, key]) {
        await expect(control).toBeVisible()
        const bounds = await control.boundingBox()
        expect(bounds).not.toBeNull()
        expect(bounds!.x).toBeGreaterThanOrEqual(0)
        expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(await page.evaluate(() => window.innerWidth))
      }
      const screenshot = path.join(directory, `web-services-narrow-940-${Math.round(zoomFactor * 100)}.png`)
      const png = await application.evaluate(async ({ BrowserWindow }) => (await BrowserWindow.getAllWindows()[0].webContents.capturePage()).toPNG().toString('base64'))
      fs.writeFileSync(screenshot, Buffer.from(png, 'base64'))
      await testInfo.attach(`web-services-${Math.round(zoomFactor * 100)}%`, { path: screenshot, contentType: 'image/png' })
    }
    expect(fixtureErrors).toEqual([])
    expect(rendererErrors).toEqual([])
  } finally {
    if (testInfo.status !== testInfo.expectedStatus && page && !page.isClosed()) await page.screenshot({ path: path.join(directory, 'failure.png') }).catch(() => undefined)
    server.closeAllConnections()
    await application?.close()
    await new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve()))
  }
})

test('联网服务测试等待地址与凭据保存，保存失败不使用旧配置', async ({}, testInfo) => {
  test.setTimeout(90_000)
  fs.mkdirSync(artifacts, { recursive: true })
  const directory = fs.mkdtempSync(path.join(artifacts, 'save-order-'))
  const replacementKey = 'modloader-ui-dummy-replacement-key-not-real'
  const invalidKey = 'invalid fake key with spaces'
  const timeline: { phase: string; event: string; field?: string; value?: string; credential?: string; status?: number }[] = []
  const observations: Record<string, unknown> = {}
  let phase = 'setup', holdField: 'address' | 'credential' | undefined, releasePatch: (() => void) | undefined
  let application: ElectronApplication | undefined, page: Page | undefined
  const server = createServer(async (request, response) => {
    for await (const _chunk of request) { /* Consume the search JSON body. */ }
    const credential = request.headers.authorization === `Bearer ${dummyKey}` ? 'original-key'
      : request.headers.authorization === `Bearer ${replacementKey}` ? 'replacement-key' : 'unexpected-key'
    timeline.push({ phase, event: 'upstream-request', value: request.url, credential })
    if (request.method !== 'POST' || !['/saved/v1/tools/search', '/pending/v1/tools/search'].includes(request.url || '') || credential === 'unexpected-key') {
      response.writeHead(400); response.end('Unexpected local test request'); return
    }
    response.writeHead(200, { 'Content-Type': 'application/json' })
    response.end(JSON.stringify({ results: [{ title: '本地地址顺序测试', url: pageUrl, summary: '由本地测试上游返回。' }] }))
  })
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const base = `http://127.0.0.1:${(server.address() as { port: number }).port}`
  const savedUrl = `${base}/saved`, pendingUrl = `${base}/pending`, invalidUrl = 'not-a-valid-http-url'
  try {
    application = await electron.launch({
      args: [root],
      env: { ...process.env, MODLOADER_DATA_DIR: path.join(directory, 'data'), MODLOADER_DEV_URL: process.env.MODLOADER_DEV_URL || '', MODLOADER_SENSOR_AUTO_START: '0' },
      timeout: 60_000,
    })
    page = await application.firstWindow()
    await expect(page.getByRole('status', { name: '服务在线', exact: true })).toBeVisible({ timeout: 45_000 })
    await management(page, '/api/settings', 'PATCH', { proxy_mode: 'direct', deepseek_url: savedUrl, secrets: { search_deepseek: dummyKey } })
    await page.route('**/api/settings', async route => {
      const body = route.request().postDataJSON()
      const field = Object.hasOwn(body || {}, 'deepseek_url') ? 'address'
        : Object.hasOwn(body?.secrets || {}, 'search_deepseek') ? 'credential' : undefined
      if (route.request().method() !== 'PATCH' || !field) { await route.continue(); return }
      const value = field === 'address' ? body.deepseek_url : body.secrets.search_deepseek === replacementKey ? 'replacement-key' : 'invalid-key'
      timeline.push({ phase, event: 'patch-dispatched', field, value })
      if (holdField === field) {
        await new Promise<void>(resolve => { releasePatch = resolve })
        timeline.push({ phase, event: 'patch-released', field, value })
      }
      const response = await route.fetch(), text = await response.text()
      timeline.push({ phase, event: 'patch-response', field, value, status: response.status() })
      await route.fulfill({ response, body: text })
    })
    await page.route('**/api/web-services/test', async route => {
      timeline.push({ phase, event: 'test-dispatched' })
      const response = await route.fetch(), body = await response.text()
      timeline.push({ phase, event: 'test-response', value: body, status: response.status() })
      await route.fulfill({ response, body })
    })
    await pinNavigation(page)
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '设置', exact: true }).click()
    const card = page.getByRole('region', { name: '联网服务', exact: true })
    await card.getByRole('tablist', { name: '渠道配置', exact: true }).getByRole('tab', { name: 'DeepSeek', exact: true }).click()
    const panel = card.getByRole('tabpanel', { name: 'DeepSeek', exact: true })
    const address = panel.getByLabel('DeepSeek 地址', { exact: true })
    const key = panel.getByLabel('DeepSeek API Key', { exact: true })
    const button = panel.getByRole('button', { name: '测试 DeepSeek 网页搜索', exact: true })
    const settingResponse = (field: string, value: string) => page!.waitForResponse(response => {
      if (response.request().method() !== 'PATCH' || new URL(response.url()).pathname !== '/api/settings') return false
      const body = response.request().postDataJSON()
      return field === 'address' ? body?.deepseek_url === value : body?.secrets?.search_deepseek === value
    })
    const serviceResponse = () => page!.waitForResponse(response => new URL(response.url()).pathname === '/api/web-services/test')
    const events = () => timeline.filter(event => event.phase === phase)
    const assertNoServiceCall = () => expect(events().filter(event => event.event === 'test-dispatched' || event.event === 'upstream-request')).toEqual([])
    const releaseSave = () => { const release = releasePatch!; releasePatch = undefined; holdField = undefined; release() }
    await expect(address).toHaveValue(savedUrl)

    phase = 'address-invalid'
    const invalidPatch = settingResponse('address', invalidUrl)
    await address.fill(invalidUrl); await address.press('Enter')
    expect((await invalidPatch).status()).toBe(400)
    await expect(address).toHaveValue(invalidUrl)
    await expect(page.getByRole('alert').filter({ hasText: /地址|HTTP|URL/ })).toBeVisible()
    await button.click()
    await expect(panel.getByRole('alert')).toContainText('请先修正并保存此渠道的配置')
    assertNoServiceCall()
    observations.invalid = {
      displayedUrl: await address.inputValue(),
      savedUrl: (await management(page, '/api/state')).settings.deepseek_url,
      message: await panel.getByRole('alert').textContent(), upstreamCalls: 0,
    }

    phase = 'address-delayed'; holdField = 'address'
    const delayedPatch = settingResponse('address', pendingUrl), delayedTest = serviceResponse()
    await address.fill(pendingUrl)
    await button.click()
    await expect.poll(() => Boolean(releasePatch)).toBe(true)
    await expect(button).toBeDisabled()
    await expect(address).toBeDisabled()
    assertNoServiceCall()
    expect((await management(page, '/api/state')).settings.deepseek_url).toBe(savedUrl)
    releaseSave()
    expect((await delayedPatch).ok()).toBe(true)
    await delayedTest
    await expect(panel.getByRole('status')).toContainText('测试成功')
    await expect.poll(async () => (await management(page!, '/api/state')).settings.deepseek_url).toBe(pendingUrl)
    expect(events().filter(event => event.event === 'upstream-request')).toEqual([{ phase, event: 'upstream-request', value: '/pending/v1/tools/search', credential: 'original-key' }])
    expect(events().findIndex(event => event.event === 'patch-response' && event.status === 200)).toBeLessThan(events().findIndex(event => event.event === 'test-dispatched'))
    observations.addressDelayed = { savedUrl: pendingUrl, upstreamPath: '/pending/v1/tools/search', saveCompletedBeforeTest: true }

    phase = 'credential-delayed'; holdField = 'credential'
    const credentialPatch = settingResponse('credential', replacementKey), credentialTest = serviceResponse()
    await key.fill(replacementKey)
    await button.click()
    await expect.poll(() => Boolean(releasePatch)).toBe(true)
    await expect(button).toBeDisabled()
    await expect(key).toBeDisabled()
    await expect(panel.getByRole('button', { name: '清除', exact: true })).toBeDisabled()
    assertNoServiceCall()
    releaseSave()
    expect((await credentialPatch).ok()).toBe(true)
    await credentialTest
    await expect(panel.getByRole('status')).toContainText('测试成功')
    await expect(key).toHaveValue('')
    expect(events().filter(event => event.event === 'upstream-request')).toEqual([{ phase, event: 'upstream-request', value: '/pending/v1/tools/search', credential: 'replacement-key' }])
    expect(events().findIndex(event => event.event === 'patch-response' && event.status === 200)).toBeLessThan(events().findIndex(event => event.event === 'test-dispatched'))
    observations.credentialDelayed = { saveCompletedBeforeTest: true, upstreamCredential: 'replacement-key', upstreamCalls: 1 }

    phase = 'credential-invalid'
    const rejectedKey = settingResponse('credential', invalidKey)
    await key.fill(invalidKey)
    await button.click()
    expect((await rejectedKey).status()).toBe(400)
    await expect(panel.getByRole('alert')).toContainText('API Key 保存失败')
    await expect(key).toHaveValue(invalidKey)
    assertNoServiceCall()
    observations.credentialInvalid = { message: await panel.getByRole('alert').textContent(), upstreamCalls: 0 }

    const contentWidth = await application.evaluate(({ BrowserWindow }) => {
      const window = BrowserWindow.getAllWindows()[0]
      window.setSize(940, 800); window.webContents.setZoomFactor(1.2)
      return window.getContentBounds().width
    })
    await expect.poll(async () => Math.abs(await page!.evaluate(() => window.innerWidth) - contentWidth / 1.2)).toBeLessThanOrEqual(1)
    await panel.getByRole('alert').scrollIntoViewIfNeeded()
    await expect(panel.getByRole('alert')).toBeInViewport({ ratio: 1 })
    await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))))
    const screenshot = path.join(directory, 'save-order-credential-failure-120.png')
    const png = await application.evaluate(async ({ BrowserWindow }) => (await BrowserWindow.getAllWindows()[0].webContents.capturePage()).toPNG().toString('base64'))
    fs.writeFileSync(screenshot, Buffer.from(png, 'base64'))
    await testInfo.attach('save-order-credential-failure-120%', { path: screenshot, contentType: 'image/png' })
    const evidence = JSON.stringify({ observations, timeline }, null, 2)
    fs.writeFileSync(path.join(directory, 'save-order-regression.json'), evidence)
    await testInfo.attach('save-order-regression', { body: evidence, contentType: 'application/json' })
    console.log(evidence)
  } finally {
    releasePatch?.()
    await application?.close()
    server.closeAllConnections()
    await new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve()))
  }
})
