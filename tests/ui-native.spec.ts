import { pinNavigation } from './navigation'
import { test, expect, _electron as electron } from '@playwright/test'
import * as path from 'node:path'
import * as fs from 'node:fs'

test('真实桌面：配置草稿、设置保存、模型源搜索与加载失败反馈', async () => {
  test.setTimeout(240_000)
  const root = path.resolve(__dirname, '..')
  const dataDirectory = path.join(root, 'work', 'ui-tests')
  const screenshotDirectory = path.join(root, 'work', 'screenshots')
  fs.mkdirSync(screenshotDirectory, { recursive: true })
  const application = await electron.launch({
    args: [root],
    env: { ...process.env, MODLOADER_DATA_DIR: dataDirectory },
    timeout: 60_000,
  })
  const page = await application.firstWindow()
  const pageErrors: string[] = []
  page.on('pageerror', error => pageErrors.push(error.message))
  const requestPath = (response: { url(): string }) => new URL(response.url()).pathname
  async function api(method: string, route: string, body?: unknown) {
    return page.evaluate(async ({ method, route, body }) => {
      const { baseUrl, token } = await window.modloader!.connection()
      const response = await fetch(baseUrl + route, {
        method,
        headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
        body: body === undefined ? undefined : JSON.stringify(body),
      })
      return { status: response.status, data: await response.json() }
    }, { method, route, body })
  }
  const navigate = async (label: string) => page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: label, exact: true }).click()

  try {
    await pinNavigation(page)
    await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
    await page.evaluate(() => document.fonts.ready)
    const config = process.env.MODLOADER_UI_CONFIG || 'D:\\Tool\\llama.cpp\\start-qwen38-highest.bat'
    const imported = await api('POST', '/api/config/import', { path: config })
    expect(imported.status).toBe(200)
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await expect(page.getByRole('heading', { name: imported.data.model.name, exact: true, level: 2 })).toBeVisible()

    await test.step('配置输入保持焦点与草稿，刷新后保存参数', async () => {
      await page.getByRole('button', { name: '新建配置', exact: true }).click()
      const dialog = page.getByRole('dialog', { name: '新建加载配置' })
      await dialog.getByLabel('配置名称', { exact: true }).fill('桌面界面验证')
      const alias = `ui-verification-${Date.now()}`
      await dialog.getByLabel('API 模型名称', { exact: true }).fill(alias)
      const context = dialog.getByLabel('上下文长度', { exact: true })
      await context.fill('4096')
      for (let update = 0; update < 2; update += 1) {
        await page.waitForResponse(response => requestPath(response) === '/api/state')
        await expect(context).toHaveValue('4096')
        await expect(context).toBeFocused()
      }
      await dialog.getByRole('tab', { name: '启动参数', exact: true }).click()
      const command = dialog.getByRole('textbox', { name: '启动参数', exact: true })
      await expect(command).not.toHaveValue('')
      await command.fill('--ctx-size abc\n--cache-type-k f1')
      await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
      await expect(dialog.locator('.startup-parameter-error')).toHaveCount(2)
      await expect(dialog).toBeVisible()
      await command.fill('--ctx-size 4096\n--metrics')
      const savedResponse = page.waitForResponse(response => requestPath(response) === '/api/profiles' && response.request().method() === 'POST')
      await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
      const profile = await (await savedResponse).json()
      await expect(dialog).toBeHidden()
      const card = page.locator('.profile-card').filter({ has: page.getByRole('heading', { name: '桌面界面验证', exact: true }) })
      await expect(card).toBeVisible()
      const state = await api('GET', '/api/state')
      const saved = state.data.profiles.find((item: { id: string }) => item.id === profile.id)
      expect(saved.parameters.context_length).toBe(4096)
      expect(saved.extra_args).toEqual(['--metrics'])
      expect(saved.parameter_editor_version).toBe(1)
      await page.screenshot({ path: path.join(screenshotDirectory, 'native-models.png') })

      const after = await api('GET', '/api/state')
      expect(after.data.instances.filter((item: { profile_id: string }) => item.profile_id === profile.id)).toHaveLength(0)
      await api('DELETE', `/api/profiles/${profile.id}`)
      await page.getByRole('button', { name: '关闭通知', exact: true }).click()
    })

    await test.step('设置自动保存，凭据显示切换不提交', async () => {
      await navigate('设置')
      const toggle = page.getByRole('switch', { name: '关闭到系统托盘' })
      const before = await toggle.getAttribute('aria-checked')
      const saveResponse = page.waitForResponse(response => requestPath(response) === '/api/settings' && response.request().method() === 'PATCH')
      await toggle.click()
      expect((await saveResponse).status()).toBe(200)
      await expect(page.locator('.save-indicator')).toHaveText('已保存')
      const current = await api('GET', '/api/state')
      expect(current.data.settings.close_to_tray).toBe(before !== 'true')

      const patches: unknown[] = []
      const onRequest = (request: { url(): string; method(): string; postDataJSON(): unknown }) => {
        if (new URL(request.url()).pathname === '/api/settings' && request.method() === 'PATCH') patches.push(request.postDataJSON())
      }
      page.on('request', onRequest)
      const credential = page.getByLabel('Hugging Face Token', { exact: true })
      await credential.fill('ui-verification-placeholder-token')
      await page.getByRole('button', { name: '显示 Hugging Face Token', exact: true }).click()
      await expect(credential).toHaveAttribute('type', 'text')
      await page.waitForResponse(response => requestPath(response) === '/api/state')
      expect(patches).toHaveLength(0)
      await expect(credential).toHaveValue('ui-verification-placeholder-token')
      await page.getByRole('button', { name: '隐藏 Hugging Face Token', exact: true }).click()
      expect(patches).toHaveLength(0)
      const credentialResponse = page.waitForResponse(response => requestPath(response) === '/api/settings' && response.request().method() === 'PATCH')
      await credential.press('Enter')
      expect((await credentialResponse).status()).toBe(200)
      await expect(credential).toHaveValue('')
      expect(patches).toHaveLength(1)
      page.off('request', onRequest)
      await api('PATCH', '/api/settings', { secrets: { hf: '' }, close_to_tray: before === 'true' })

      const port = page.getByLabel(/^API 端口/)
      await port.fill('invalid-port')
      await port.press('Enter')
      await expect(page.getByRole('alert')).toContainText('端口须为 1–65535 的整数')
      await port.fill('12340')
      await port.press('Enter')
      await expect(page.locator('.save-indicator')).toHaveText('已保存')
      await page.evaluate(() => window.scrollTo(0, 0))
      await page.screenshot({ path: path.join(screenshotDirectory, 'native-settings.png') })
    })

    await test.step('通过真实模型源搜索，结果或失败都有结束状态', async () => {
      await navigate('发现模型')
      for (const source of ['hf', 'modelscope']) {
        if (source === 'modelscope') await page.getByRole('button', { name: /ModelScope$/ }).click()
        await page.getByRole('textbox', { name: '搜索在线模型' }).fill('Qwen')
        const responsePromise = page.waitForResponse(response => requestPath(response) === `/api/sources/${source}/search`, { timeout: 100_000 })
        await page.getByRole('button', { name: '搜索', exact: true }).click()
        const response = await responsePromise
        const payload = await response.json()
        if (response.ok()) {
          expect(payload.items.length).toBeGreaterThan(0)
          await expect(page.locator('.repository-card').first()).toBeVisible()
        } else {
          await expect(page.getByRole('alert')).toBeVisible()
        }
        await expect(page.getByRole('button', { name: '搜索', exact: true })).toBeEnabled()
        console.log(JSON.stringify({ source, status: response.status(), results: payload.items?.length || 0 }))
      }
      await page.screenshot({ path: path.join(screenshotDirectory, 'native-discover.png') })
    })

    await navigate('运行环境')
    await page.screenshot({ path: path.join(screenshotDirectory, 'native-runtimes.png') })
    await navigate('测试对话')
    await page.screenshot({ path: path.join(screenshotDirectory, 'native-chat.png') })
    expect(pageErrors).toEqual([])
  } finally {
    await application.close()
  }
})

test('原生显示与监控界面契约：字号、窗口缩放与仅在当前页采样', async () => {
  test.setTimeout(120_000)
  const root = path.resolve(__dirname, '..')
  const directory = path.join(root, 'work', 'screenshots')
  fs.mkdirSync(directory, { recursive: true })
  const application = await electron.launch({ args: [root], env: { ...process.env, MODLOADER_DATA_DIR: path.join(root, 'work', 'ui-display-tests') }, timeout: 60_000 })
  const page = await application.firstWindow()
  const errors: string[] = []
  page.on('pageerror', issue => errors.push(issue.message))
  const navigation = page.getByRole('navigation', { name: '主导航' })
  async function nativeScreenshot(name: string) {
    await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))))
    const png = await application.evaluate(async ({ BrowserWindow }) => (await BrowserWindow.getAllWindows()[0].webContents.capturePage()).toPNG().toString('base64'))
    fs.writeFileSync(path.join(directory, name), Buffer.from(png, 'base64'))
  }
  let telemetryReads = 0
  let fixtureBackend = 'llama_cpp'
  const fixtureInstance = { id: 'ui-contract-fixture', model_id: 'ui-model-fixture', profile_id: 'ui-profile-fixture', profile_revision: 1, alias: '监控契约 fixture', backend: fixtureBackend, runtime_id: 'ui-runtime-fixture', state: 'ready', port: 0, started_at: '', active_requests: 0, capabilities: {} }
  try {
    await pinNavigation(page)
    await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
    // These routes verify rendering and sampling lifetime; they do not claim a real engine is running.
    await page.route('**/api/state', async route => {
      const response = await route.fetch()
      const state = await response.json()
      await route.fulfill({ response, json: { ...state, instances: [{ ...fixtureInstance, backend: fixtureBackend }] } })
    })
    await page.route('**/api/instances/ui-contract-fixture/telemetry', async route => {
      telemetryReads += 1
      const groupName = { llama_cpp: '请求与推理吞吐', strata: '专家与上下文', transformers: '生成与队列', vllm: '调度与缓存' }[fixtureBackend] || fixtureBackend
      await route.fulfill({ json: { instance_id: fixtureInstance.id, backend: fixtureBackend, state: 'ready', updated_at: new Date().toISOString(), groups: [
        { id: 'engine', title: groupName, metrics: [
          { id: 'rate', label: '输出吞吐', value: 12.5, unit: 'Token/s', kind: 'gauge', description: '界面契约 fixture 的当前速率' },
          { id: 'tokens', label: '输出 Token', value: 12345, unit: 'Token', kind: 'counter' },
          { id: 'queue', label: '队列请求', value: 0, kind: 'gauge' },
          { id: 'cache', label: '缓存占用', value: null, unit: '%', kind: 'gauge' },
        ] },
      ], unavailable: ['fixture 未提供缓存采样'] } })
    })
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await navigation.getByRole('button', { name: '运行监控', exact: true }).click()
    await expect(page.getByRole('heading', { name: '请求与推理吞吐', exact: true })).toBeVisible()
    await expect(page.locator('.embedded-metric').filter({ hasText: '输出 Token' })).toContainText('累计')
    await expect(page.locator('.embedded-metric').filter({ hasText: '队列请求' })).toContainText('0')
    await expect(page.locator('.embedded-metric').filter({ hasText: '缓存占用' })).toContainText('不可用')
    for (const backend of ['strata', 'transformers', 'vllm']) {
      fixtureBackend = backend
      await page.getByRole('button', { name: '刷新监控', exact: true }).click()
      const title = { strata: '专家与上下文', transformers: '生成与队列', vllm: '调度与缓存' }[backend]
      await expect(page.getByRole('heading', { name: title, exact: true })).toBeVisible()
    }
    await page.screenshot({ path: path.join(directory, 'monitor-contract-fixture.png') })

    await navigation.getByRole('button', { name: '设置', exact: true }).click()
    const afterLeaving = telemetryReads
    for (let update = 0; update < 2; update += 1) await page.waitForResponse(response => new URL(response.url()).pathname === '/api/state')
    expect(telemetryReads).toBe(afterLeaving)
    const scale = page.getByRole('combobox', { name: '文字大小', exact: true })
    await scale.selectOption('1')
    await expect.poll(() => page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue('--font-scale').trim())).toBe('1')
    const originalSize = await page.getByLabel(/^API 端口/).evaluate(element => parseFloat(getComputedStyle(element).fontSize))

    for (const display of [{ width: 1440, height: 960, zoom: 1, scale: '1' }, { width: 1050, height: 780, zoom: 1, scale: '1' }, { width: 1440, height: 960, zoom: 1.25, scale: '1.3' }, { width: 1050, height: 780, zoom: 1.25, scale: '1.3' }]) {
      await scale.selectOption(display.scale)
      await expect.poll(() => page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue('--font-scale').trim())).toBe(display.scale)
      await application.evaluate(({ BrowserWindow }, display) => { const window = BrowserWindow.getAllWindows()[0]; window.setSize(display.width, display.height); window.webContents.setZoomFactor(display.zoom) }, display)
      const geometry = await application.evaluate(({ BrowserWindow }) => { const window = BrowserWindow.getAllWindows()[0]; return { size: window.getContentSize(), zoom: window.webContents.getZoomFactor() } })
      await expect.poll(async () => Math.abs((await page.evaluate(() => window.innerWidth)) - geometry.size[0] / geometry.zoom)).toBeLessThan(2)
      await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
      const croppedControls = await page.locator('.navigation button, .button, .badge').evaluateAll(elements => elements.filter(element => {
        const html = element as HTMLElement
        return html.getClientRects().length && html.scrollHeight > html.clientHeight + 1
      }).length)
      expect(croppedControls).toBe(0)
      await page.evaluate(() => window.scrollTo(0, 0))
      await nativeScreenshot(`display-${display.width}-zoom${display.zoom}-text${display.scale}.png`)
    }
    const enlargedSize = await page.getByLabel(/^API 端口/).evaluate(element => parseFloat(getComputedStyle(element).fontSize))
    expect(enlargedSize / originalSize).toBeCloseTo(1.3, 1)
    await scale.selectOption('1')
    await expect.poll(() => page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue('--font-scale').trim())).toBe('1')
    await application.evaluate(({ BrowserWindow }) => { const window = BrowserWindow.getAllWindows()[0]; window.setSize(1440, 960); window.webContents.setZoomFactor(1) })
    await page.evaluate(() => window.scrollTo(0, 0))
    await page.screenshot({ path: path.join(directory, 'native-settings-final.png') })
    await navigation.getByRole('button', { name: '运行监控', exact: true }).click()
    await expect(page.getByRole('heading', { name: '调度与缓存', exact: true })).toBeVisible()
    await navigation.getByRole('button', { name: '测试对话', exact: true }).click()
    await page.screenshot({ path: path.join(directory, 'native-chat-final.png') })
    expect(errors).toEqual([])
  } finally {
    await application.close()
  }
})

test('大文字与界面缩放：原生导入弹窗完整显示', async () => {
  test.setTimeout(60_000)
  const root = path.resolve(__dirname, '..')
  const application = await electron.launch({ args: [root], env: { ...process.env, MODLOADER_DATA_DIR: path.join(root, 'work', 'ui-font-dialog') }, timeout: 60_000 })
  const page = await application.firstWindow()
  try {
    await pinNavigation(page)
    await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
    const navigation = page.getByRole('navigation', { name: '主导航' })
    await navigation.getByRole('button', { name: '设置', exact: true }).click()
    await page.getByRole('combobox', { name: '文字大小', exact: true }).selectOption('1.3')
    await expect.poll(() => page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue('--font-scale').trim())).toBe('1.3')
    const contentWidth = await application.evaluate(({ BrowserWindow }) => { const window = BrowserWindow.getAllWindows()[0]; window.setSize(1050, 780); window.webContents.setZoomFactor(1.25); return window.getContentSize()[0] })
    await expect.poll(async () => Math.abs((await page.evaluate(() => window.innerWidth)) - contentWidth / 1.25)).toBeLessThan(2)
    await navigation.getByRole('button', { name: '模型库', exact: true }).click()
    await page.getByRole('button', { name: '导入模型', exact: true }).click()
    const dialog = page.getByRole('dialog', { name: '导入模型文件', exact: true })
    await dialog.getByLabel(/^文件路径/).fill('D:\\Models\\model with spaces.gguf')
    await expect(dialog.getByLabel(/^文件路径/)).toHaveValue('D:\\Models\\model with spaces.gguf')
    expect(await dialog.evaluate(element => { const box = element.getBoundingClientRect(); return box.left >= 0 && box.right <= window.innerWidth && element.scrollWidth <= element.clientWidth })).toBe(true)
    await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))))
    const png = await application.evaluate(async ({ BrowserWindow }) => (await BrowserWindow.getAllWindows()[0].webContents.capturePage()).toPNG().toString('base64'))
    fs.mkdirSync(path.join(root, 'work', 'screenshots'), { recursive: true })
    fs.writeFileSync(path.join(root, 'work', 'screenshots', 'display-dialog-zoom1.25-text1.3.png'), Buffer.from(png, 'base64'))
    await dialog.getByRole('button', { name: '取消', exact: true }).click()
    await navigation.getByRole('button', { name: '设置', exact: true }).click()
    await page.getByRole('combobox', { name: '文字大小', exact: true }).selectOption('1')
    await expect.poll(() => page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue('--font-scale').trim())).toBe('1')
    await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].webContents.setZoomFactor(1))
  } finally { await application.close() }
})
