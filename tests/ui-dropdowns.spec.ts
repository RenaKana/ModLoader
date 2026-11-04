import { pinNavigation } from './navigation'
import { chatCapsule, chatModel, chatModelList, openChatControls, selectChatModel } from './chat-controls'
import { test, expect, _electron as electron, type Locator } from '@playwright/test'
import * as path from 'node:path'
import * as fs from 'node:fs'

test('真实 Electron 下拉菜单：统一外观、鼠标键盘及放大后的长选项', async () => {
  test.setTimeout(120_000)
  const root = path.resolve(__dirname, '..')
  const directory = fs.mkdtempSync(path.join(root, 'work', 'ui-dropdowns-'))
  const screenshots = path.join(directory, 'screenshots')
  fs.mkdirSync(screenshots, { recursive: true })
  const application = await electron.launch({
    ...(process.env.MODLOADER_TEST_EXECUTABLE || process.env.MODLOADER_TEST_PACKAGED ? { executablePath: process.env.MODLOADER_TEST_EXECUTABLE || path.join(root, 'release', 'win-unpacked', 'ModLoader.exe'), args: [] } : { args: [root] }),
    env: { ...process.env, MODLOADER_DATA_DIR: path.join(directory, 'data') }, timeout: 60_000,
  })
  const page = await application.firstWindow()
  const errors: string[] = []
  const externalRequests: string[] = []
  const lifecycleRequests: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  page.on('request', request => {
    const route = new URL(request.url()).pathname
    if (request.method() !== 'GET' && route !== '/api/profiles/preflight' && /\/api\/(?:sources|downloads|runtimes\/(?:install|import)|profiles|instances)/.test(route)) lifecycleRequests.push(route)
  })
  // Only the application and loopback management core are used by this check.
  await page.route('**/*', async route => {
    const url = new URL(route.request().url())
    if (['http:', 'https:'].includes(url.protocol) && !['127.0.0.1', 'localhost'].includes(url.hostname)) {
      externalRequests.push(url.origin)
      await route.abort()
    } else await route.continue()
  })
  const longName = '这是未选中的长模型名称，用来确认中文与 Outfit Latin model names 在等宽菜单中单行省略，完整名称仍可查看，不遮住右侧的选中标记'
  let scrollingFixture = false
  let instanceMode: 'multiple' | 'single' | 'none' | 'loading' | 'stopped' = 'multiple'
  const fixtureInstances = () => (instanceMode === 'none' ? [] : [
    { id: 'dropdown-instance-a', alias: '下拉界面 fixture A', backend: 'llama_cpp' },
    { id: 'dropdown-instance-b', alias: longName, backend: 'strata' },
    ...(scrollingFixture ? Array.from({ length: 24 }, (_, index) => ({ id: `dropdown-scroll-${index}`, alias: `列表滚动 fixture ${index + 1}`, backend: 'llama_cpp' })) : []),
  ].slice(0, instanceMode === 'multiple' ? undefined : 1)).map(instance => ({ ...instance, model_id: 'dropdown-model', profile_id: `dropdown-profile-${instance.id}`, profile_revision: 1, runtime_id: `dropdown-${instance.backend}-short`, state: instanceMode === 'loading' ? 'loading' : instanceMode === 'stopped' ? 'stopped' : 'ready', port: 0, started_at: '', active_requests: 0, capabilities: {} }))
  const navigation = page.getByRole('navigation', { name: '主导航' })
  const navigate = async (label: string) => navigation.getByRole('button', { name: label, exact: true }).click()
  async function nativeScreenshot(name: string) {
    await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))))
    const png = await application.evaluate(async ({ BrowserWindow }) => (await BrowserWindow.getAllWindows()[0].webContents.capturePage()).toPNG().toString('base64'))
    fs.writeFileSync(path.join(screenshots, name), Buffer.from(png, 'base64'))
  }
  async function openMenu(select: Locator, screenshot: string, compact = false) {
    await select.click()
    await expect.poll(() => select.evaluate(element => element.matches(':open'))).toBe(true)
    await expect.poll(() => select.evaluate(element => getComputedStyle(element, '::picker(select)').opacity)).toBe('1')
    await nativeScreenshot(screenshot)
    const display = await select.evaluate(element => {
      const select = element as HTMLSelectElement
      const control = getComputedStyle(select)
      const picker = getComputedStyle(select, '::picker(select)')
      const group = select.querySelector<HTMLElement>('.select-options')!
      const groupBox = group.getBoundingClientRect()
      return {
        appearance: control.appearance,
        pickerAppearance: picker.appearance,
        pickerFont: picker.fontFamily,
        pickerSize: picker.fontSize,
        controlSize: control.fontSize,
        controlWidth: select.getBoundingClientRect().width,
        controlHeight: select.getBoundingClientRect().height,
        pickerWidth: groupBox.width + parseFloat(picker.borderLeftWidth) + parseFloat(picker.borderRightWidth),
        group: { top: groupBox.top, bottom: groupBox.bottom, left: groupBox.left, right: groupBox.right, scrolling: group.scrollHeight > group.clientHeight + 1 },
        options: [...select.options].filter(option => !option.hidden).map(option => {
          const box = option.getBoundingClientRect()
          return { top: box.top, bottom: box.bottom, left: box.left, right: box.right, height: box.height, font: getComputedStyle(option).fontFamily, clippedText: option.scrollWidth > option.clientWidth + 1 }
        }),
        viewport: { width: innerWidth, height: innerHeight },
      }
    })
    expect(display.appearance).toBe('base-select')
    expect(display.pickerAppearance).toBe('base-select')
    expect(display.pickerFont).toContain('Outfit')
    expect(display.pickerSize).toBe(display.controlSize)
    if (compact) expect(display.pickerWidth).toBeGreaterThanOrEqual(display.controlWidth)
    else expect(Math.abs(display.pickerWidth - display.controlWidth)).toBeLessThan(1)
    expect(display.group.left).toBeGreaterThanOrEqual(0)
    expect(display.group.right).toBeLessThanOrEqual(display.viewport.width + 1)
    expect(display.group.top).toBeGreaterThanOrEqual(0)
    expect(display.group.bottom).toBeLessThanOrEqual(display.viewport.height + 1)
    for (const option of display.options) {
      expect(option.font).toContain('Outfit')
      expect(option.clippedText).toBe(false)
      if (compact) expect(option.height).toBeGreaterThanOrEqual(display.controlHeight)
      else expect(Math.abs(option.height - display.controlHeight), JSON.stringify({ screenshot, controlHeight: display.controlHeight, optionHeight: option.height, controlFont: display.controlSize, pickerFont: display.pickerSize })).toBeLessThan(1)
      if (!display.group.scrolling) {
        expect(option.left).toBeGreaterThanOrEqual(0)
        expect(option.right).toBeLessThanOrEqual(display.viewport.width + 1)
        expect(option.top).toBeGreaterThanOrEqual(0)
        expect(option.bottom).toBeLessThanOrEqual(display.viewport.height + 1)
      }
    }
  }
  async function dismissMenu(select: Locator) {
    await page.keyboard.press('Escape')
    const closed = await select.evaluate(element => new Promise(resolve => requestAnimationFrame(() => {
      const picker = getComputedStyle(element, '::picker(select)')
      resolve({ open: element.matches(':open'), display: picker.display, overlay: picker.getPropertyValue('overlay') })
    })))
    expect(closed).toEqual({ open: false, display: 'none', overlay: 'none' })
  }

  try {
    await pinNavigation(page)
    await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
    const capabilities = await page.evaluate(() => ({ baseSelect: CSS.supports('appearance', 'base-select'), picker: CSS.supports('selector(::picker(select))') }))
    expect(capabilities).toEqual({ baseSelect: true, picker: true })
    console.log(JSON.stringify({ electron: await application.evaluate(() => process.versions.electron), chromium: await application.evaluate(() => process.versions.chrome) }))
    await application.evaluate(({ BrowserWindow }) => { const window = BrowserWindow.getAllWindows()[0]; window.setSize(1440, 960); window.webContents.setZoomFactor(1) })
    // Fixture lists exercise the actual renderer and picker; no model or engine is created.
    await page.route('**/api/state', async route => {
      const response = await route.fetch()
      const state = await response.json()
      const runtimes = state.backends.flatMap((backend: { id: string; label: string }) => [
        { id: `dropdown-${backend.id}-short`, name: `${backend.label} 界面 fixture`, version: 'UI', backend: backend.id, path: path.join(directory, 'fixture-only'), managed: false, metadata: {} },
        { id: `dropdown-${backend.id}-long`, name: longName, version: 'UI', backend: backend.id, path: path.join(directory, 'fixture-only-long'), managed: false, metadata: {} },
      ])
      await route.fulfill({ response, json: { ...state,
        models: [{ id: 'dropdown-model', name: '下拉菜单界面 fixture', path: path.join(directory, 'fixture.gguf'), format: 'gguf', size_bytes: 0, files: [], modalities: ['text'], metadata: {}, source: {}, managed: false, warnings: [] }],
        profiles: [], runtimes, instances: fixtureInstances(), downloads: [], installations: [],
      } })
    })
    await page.route('**/api/runtimes/versions?*', route => route.fulfill({ json: { versions: [{ version: 'b-fixture', label: '界面测试版本', status: 'available', recommended: true, variants: [{ id: 'cuda', label: 'CUDA · x64' }, { id: 'cpu', label: 'CPU · x64' }] }], next_cursor: null } }))
    await page.route('**/api/profiles/preflight', route => route.fulfill({ json: { status: 'ready', message: '', candidates: [], errors: [], vision_enabled: false, mtp: { enabled: false, status: 'unsupported', backend_supported: false, message: '此 fixture 不提供 MTP', candidates: [] } } }))
    await page.route('**/api/instances/dropdown-*/telemetry', route => route.fulfill({ json: { instance_id: route.request().url().split('/').at(-2), backend: 'llama_cpp', state: 'ready', updated_at: '', groups: [], unavailable: ['界面 fixture，无真实引擎采样'] } }))
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await expect(page.getByRole('heading', { name: '下拉菜单界面 fixture', exact: true })).toBeVisible()
    await page.evaluate(() => document.fonts.ready)

    await test.step('筛选菜单可以鼠标选择，并保持 Arrow / Enter / Escape 行为', async () => {
      const filter = page.getByRole('combobox', { name: '筛选模型', exact: true })
      await openMenu(filter, 'filter-open.png')
      await filter.getByRole('option', { name: 'GGUF', exact: true }).hover()
      await nativeScreenshot('filter-hover.png')
      await filter.getByRole('option', { name: 'GGUF', exact: true }).click()
      await expect(filter).toHaveValue('gguf')
      await filter.selectOption('all')
      await filter.focus()
      await page.keyboard.press('Space')
      await expect.poll(() => filter.evaluate(element => element.matches(':open'))).toBe(true)
      await page.keyboard.press('ArrowDown')
      await page.keyboard.press('Enter')
      await expect(filter).toHaveValue('running')
      await filter.click()
      await page.keyboard.press('ArrowDown')
      await dismissMenu(filter)
      await expect(filter).toHaveValue('running')
      await filter.selectOption('all')
    })

    for (const display of [{ width: 1440, height: 960, zoom: 1, scale: '1' }, { width: 1050, height: 780, zoom: 1.25, scale: '1.3' }]) {
      const suffix = `zoom${display.zoom}-text${display.scale}.png`
      await navigate('设置')
      const font = page.getByRole('combobox', { name: '文字大小', exact: true })
      await font.selectOption(display.scale)
      await expect.poll(() => page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue('--font-scale').trim())).toBe(display.scale)
      await application.evaluate(({ BrowserWindow }, display) => { const window = BrowserWindow.getAllWindows()[0]; window.setSize(display.width, display.height); window.webContents.setZoomFactor(display.zoom) }, display)
      const contentWidth = await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].getContentSize()[0])
      await expect.poll(async () => Math.abs((await page.evaluate(() => innerWidth)) - contentWidth / display.zoom)).toBeLessThan(2)
      await openMenu(font, `font-${suffix}`)
      await dismissMenu(font)

      await navigate('模型库')
      await page.locator('.profile-panel > .panel-heading').getByRole('button', { name: '新建配置', exact: true }).click()
      const dialog = page.getByRole('dialog', { name: '新建加载配置', exact: true })
      const backend = dialog.getByRole('combobox', { name: '运行后端', exact: true })
      await openMenu(backend, `backend-${suffix}`)
      await backend.getByRole('option', { name: 'Strata', exact: true }).click()
      await expect(backend).toHaveValue('strata')
      const runtime = dialog.getByRole('combobox', { name: '运行环境', exact: true })
      await openMenu(runtime, `runtime-${suffix}`)
      const longRuntime = runtime.locator('option').last()
      expect(await longRuntime.locator('.select-option-label').evaluate(element => element.scrollWidth > element.clientWidth)).toBe(true)
      await expect(longRuntime).toHaveAttribute('title', new RegExp(longName))
      await dismissMenu(runtime)
      await expect(dialog).toBeVisible()
      const cache = dialog.getByRole('combobox', { name: 'KV 缓存', exact: true })
      await openMenu(cache, `parameter-${suffix}`)
      await cache.locator('option[value="fp16"]').click()
      await expect(cache).toHaveValue('fp16')
      await backend.selectOption('transformers')
      const boolean = dialog.getByRole('combobox', { name: 'torch.compile', exact: true })
      await openMenu(boolean, `boolean-${suffix}`)
      await boolean.getByRole('option', { name: '开启', exact: true }).click()
      await expect(boolean).toHaveValue('true')
      await page.keyboard.press('Escape')
      await expect(dialog).toBeHidden()

      await navigate('运行环境')
      await page.getByRole('button', { name: '导入运行环境', exact: true }).click()
      const importDialog = page.getByRole('dialog', { name: '导入已有运行环境', exact: true })
      const importBackend = importDialog.getByRole('combobox', { name: '运行后端', exact: true })
      await openMenu(importBackend, `runtime-import-${suffix}`)
      await dismissMenu(importBackend)
      await expect(importDialog).toBeVisible()
      await nativeScreenshot(`runtime-import-closed-${suffix}`)
      await importDialog.getByRole('button', { name: '取消', exact: true }).click()
      await page.locator('.backend-card').filter({ has: page.getByRole('heading', { name: 'llama.cpp', exact: true }) }).getByRole('button', { name: '安装新环境', exact: true }).click()
      const installDialog = page.getByRole('dialog', { name: '安装 llama.cpp', exact: true })
      await installDialog.getByRole('combobox', { name: '官方版本', exact: true }).selectOption('b-fixture')
      const flavor = installDialog.getByRole('combobox', { name: '运行类型', exact: true })
      await openMenu(flavor, `runtime-flavor-${suffix}`)
      await dismissMenu(flavor)
      await installDialog.getByRole('button', { name: '取消', exact: true }).click()

      await navigate('测试对话')
      const chat = chatModel(page)
      await selectChatModel(page, 'dropdown-instance-a', true)
      await chat.click()
      await expect(page.locator('.chat-model-view')).toHaveCSS('opacity', '1')
      await nativeScreenshot(`chat-long-${suffix}`)
      const longOption = chatModelList(page).locator('[data-model-id="dropdown-instance-b"]')
      expect(await longOption.locator('span').evaluate(element => element.scrollWidth <= element.clientWidth + 1)).toBe(true)
      await expect(longOption).toContainText(longName)
      await longOption.click()
      await expect(chat).toHaveJSProperty('value', 'dropdown-instance-b')
      expect(await chat.evaluate(element => element.scrollWidth <= element.clientWidth + 1)).toBe(true)
      await nativeScreenshot(`chat-selected-long-${suffix}`)
      await selectChatModel(page, 'dropdown-instance-a')

      await navigate('运行监控')
      const monitor = page.getByRole('combobox', { name: '选择监控实例', exact: true })
      await monitor.selectOption('dropdown-instance-a')
      await openMenu(monitor, `monitor-${suffix}`)
      await dismissMenu(monitor)
      await monitor.selectOption('dropdown-instance-b')
      expect(await monitor.evaluate(element => element.scrollWidth <= element.clientWidth + 1)).toBe(true)
      await nativeScreenshot(`monitor-selected-long-${suffix}`)
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
    }

    await test.step('选项数量增多时列表在当前窗口内滚动', async () => {
      scrollingFixture = true
      await page.getByRole('button', { name: '刷新状态', exact: true }).click()
      await navigate('测试对话')
      await openChatControls(page)
      const chat = chatModel(page)
      await chat.click()
      await expect(chatModelList(page).getByRole('option')).toHaveCount(26)
      await expect(page.locator('.chat-model-view')).toHaveCSS('opacity', '1')
      await nativeScreenshot('chat-scroll-zoom1.25-text1.3.png')
      await page.keyboard.press('End')
      const last = chatModelList(page).getByRole('option').last()
      await expect.poll(() => last.evaluate(element => { const box = element.getBoundingClientRect(); return box.top >= 0 && box.bottom <= innerHeight + 1 })).toBe(true)
      await nativeScreenshot('chat-scroll-last-zoom1.25-text1.3.png')
      await page.keyboard.press('Enter')
      await expect(chat).toHaveJSProperty('value', 'dropdown-scroll-23')
      await expect(page.locator('.chat-save-status')).toHaveCount(0)
      await page.emulateMedia({ reducedMotion: 'reduce' })
      await chat.click()
      await expect(page.locator('.chat-model-view')).toHaveCSS('transform', 'none')
      await page.keyboard.press('Escape')
      await expect(chatModelList(page)).toHaveCount(0)
    })
    await test.step('单模型保留模型胶囊，加载与停止状态保留，空态提供模型库入口', async () => {
      await page.keyboard.press('Escape')
      scrollingFixture = false
      instanceMode = 'single'
      await page.getByRole('button', { name: '刷新状态', exact: true }).click()
      await expect(chatModelList(page)).toHaveCount(0)
      await expect(chatCapsule(page)).toContainText('fixture A')
      await navigate('运行监控')
      await expect(page.getByRole('combobox', { name: '选择监控实例', exact: true })).toHaveCount(0)
      await expect(page.locator('.monitor-selector-main .model-instance-name')).toContainText('fixture A')
      await nativeScreenshot('monitor-single.png')
      instanceMode = 'loading'
      await page.getByRole('button', { name: '刷新状态', exact: true }).click()
      await expect(page.locator('.monitor-selector-main .badge')).toHaveText('加载中')
      await navigate('测试对话')
      await expect(page.locator('.chat-model-state .badge')).toHaveText('加载中')
      await expect(chatModelList(page)).toHaveCount(0)
      await navigate('运行监控')
      instanceMode = 'stopped'
      await page.getByRole('button', { name: '刷新状态', exact: true }).click()
      await expect(page.locator('.monitor-selector-main .badge')).toHaveText('已停止')
      await expect(page.locator('.monitor-selector-end').getByRole('button', { name: '日志', exact: true })).toBeVisible()
      await expect(page.locator('.monitor-selector-end').getByRole('button', { name: '停止', exact: true })).toHaveCount(0)
      instanceMode = 'none'
      await page.getByRole('button', { name: '刷新状态', exact: true }).click()
      await expect(page.locator('.monitor-selector-main')).toContainText('暂无运行模型')
      await expect(page.locator('.monitor-selector-main').getByRole('button', { name: '前往模型库' })).toBeVisible()
      await nativeScreenshot('monitor-empty.png')
      await navigate('测试对话')
      await expect(chatCapsule(page)).toBeDisabled()
      await expect(page.locator('.chat-empty').getByRole('button', { name: '前往模型库' })).toBeVisible()
      await nativeScreenshot('chat-empty.png')
    })
    expect(externalRequests).toEqual([])
    expect(lifecycleRequests).toEqual([])
    expect(errors).toEqual([])
  } finally {
    await application.close()
  }
})
