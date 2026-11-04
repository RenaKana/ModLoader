import { pinNavigation } from './navigation'
import { test, expect, _electron as electron, type Locator } from '@playwright/test'
import * as path from 'node:path'
import * as fs from 'node:fs'
import type { LoadProfile } from '../src/types'

test('加载配置空白默认提示、保存与 Strata 导入参数保留', async () => {
  test.setTimeout(90_000)
  const root = path.resolve(__dirname, '..')
  const directory = fs.mkdtempSync(path.join(root, 'work', 'ui-load-defaults-'))
  const screenshots = path.join(directory, 'screenshots')
  fs.mkdirSync(screenshots)
  const application = await electron.launch({
    ...(process.env.MODLOADER_TEST_EXECUTABLE ? { executablePath: process.env.MODLOADER_TEST_EXECUTABLE, args: [] } : { args: [root] }),
    env: { ...process.env, MODLOADER_DATA_DIR: path.join(directory, 'data') }, timeout: 60_000,
  })
  const page = await application.firstWindow()
  const errors: string[] = []
  const forbiddenRequests: string[] = []
  const saved: LoadProfile[] = []
  let patchRequests = 0
  let controlledPatch: { fail: boolean; started: Promise<void>; signalStarted: () => void; wait: Promise<void>; release: () => void } | null = null
  function holdNextPatch(fail: boolean) {
    let signalStarted!: () => void
    let release!: () => void
    const started = new Promise<void>(resolve => { signalStarted = resolve })
    const wait = new Promise<void>(resolve => { release = resolve })
    const gate = { fail, started, signalStarted, wait, release }
    controlledPatch = gate
    return gate
  }
  let backendFields: any[] = []
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/*', async route => {
    const url = new URL(route.request().url())
    if (['http:', 'https:'].includes(url.protocol) && !['127.0.0.1', 'localhost'].includes(url.hostname)) {
      forbiddenRequests.push(url.origin)
      await route.abort()
    } else if (route.request().method() !== 'GET' && /\/api\/(?:instances|runtimes|sources|downloads|config|models)/.test(url.pathname)) {
      forbiddenRequests.push(url.pathname)
      await route.abort()
    } else await route.continue()
  })
  const strataConfig = { args: ['--max-context', '262144', '--kv', 'int8', '--expert-cache', 'auto', '--vram-reserve-mib', '2048'] }
  const strataAliasConfig = { args: ['--max-context', '8192', '--kv', 'q4'] }
  let profiles: LoadProfile[] = [
    { id: 'llama-empty', model_id: 'fixture-model', name: 'llama 空白配置', backend: 'llama_cpp', runtime_id: 'runtime-llama_cpp', alias: 'llama-empty', parameters: {}, generation: {}, extra_args: [], revision: 1 },
    { id: 'strata-fixed', model_id: 'fixture-model', name: 'Strata 导入配置', backend: 'strata', runtime_id: 'runtime-strata', alias: 'strata-fixed',
      parameters: { context_length: 262144, kv_cache: 'int8', expert_cache: 'auto', vram_reserve_mib: 2048, strata_config: strataConfig }, generation: {}, extra_args: [], revision: 1 },
    { id: 'strata-q4-alias', model_id: 'fixture-model', name: 'Strata q4 别名导入配置', backend: 'strata', runtime_id: 'runtime-strata', alias: 'strata-q4-alias',
      parameters: { context_length: 8192, kv_cache: 'q4', strata_config: strataAliasConfig }, generation: {}, extra_args: [], revision: 1 },
  ]
  await page.route('**/api/state', async route => {
    const response = await route.fetch()
    const state = await response.json()
    backendFields = state.backends
    await route.fulfill({ response, json: { ...state,
      models: [{ id: 'fixture-model', name: '加载配置 UI fixture', path: path.join(directory, 'fixture.gguf'), format: 'gguf', size_bytes: 0, files: [], modalities: ['text'], metadata: {}, source: {}, managed: false, warnings: [] }],
      profiles, runtimes: [...state.backends.map((backend: { id: string; label: string }) => ({ id: `runtime-${backend.id}`, backend: backend.id, name: `${backend.label} UI fixture`, version: 'fixture', path: directory, managed: false, metadata: {} })), { id: 'runtime-llama_cpp-cpu', backend: 'llama_cpp', name: 'llama.cpp CPU', version: 'fixture', path: directory, managed: false, metadata: { variant: 'cpu' } }],
      instances: [], downloads: [], installations: [],
    } })
  })
  await page.route('**/api/resources/estimate', async route => route.fulfill({ json: { resources: [], notes: [], source: 'ui-load-defaults-fixture' } }))
  await page.route('**/api/profiles/**', async route => {
    if (new URL(route.request().url()).pathname.endsWith('/preflight')) return route.fulfill({ json: { parameters: backendFields.find(b => b.id === route.request().postDataJSON().backend)?.parameters.map((f: any) => f.key === 'gpu_layers' ? { ...f, maximum: route.request().postDataJSON().runtime_id.endsWith('-cpu') ? 0 : 66 } : f), status: 'ready', message: '', candidates: [], errors: [], vision_enabled: false, mtp: { enabled: false, status: 'unsupported', backend_supported: false, message: '默认值保留用例未提供 MTP 文件', candidates: [] } } })
    if (route.request().method() !== 'PATCH') {
      forbiddenRequests.push(new URL(route.request().url()).pathname)
      return route.abort()
    }
    patchRequests += 1
    const gate = controlledPatch
    if (gate) {
      gate.signalStarted()
      await gate.wait
      if (controlledPatch === gate) controlledPatch = null
      if (gate.fail) return route.fulfill({ status: 500, json: { message: 'fixture PATCH failure' } })
    }
    const body = route.request().postDataJSON() as LoadProfile
    const identifier = new URL(route.request().url()).pathname.split('/')[3]
    const profile = { ...body, id: identifier || 'new-empty', revision: 1 }
    saved.push(profile)
    profiles = [...profiles.filter(item => item.id !== profile.id), profile]
    await route.fulfill({ json: profile })
  })
  const card = (name: string) => page.locator('.profile-card').filter({ has: page.getByRole('heading', { name, exact: true }) })
  async function optionValues(select: Locator) {
    return select.locator('option').evaluateAll(options => options.map(option => (option as HTMLOptionElement).value))
  }
  async function nativeScreenshot(name: string) {
    await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))))
    const png = await application.evaluate(async ({ BrowserWindow }) => (await BrowserWindow.getAllWindows()[0].webContents.capturePage()).toPNG().toString('base64'))
    fs.writeFileSync(path.join(screenshots, name), Buffer.from(png, 'base64'))
  }
  try {
    await pinNavigation(page)
    await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await expect(page.getByRole('heading', { name: '加载配置 UI fixture', exact: true })).toBeVisible()
    const llamaEmptyCard = card('llama 空白配置')
    await expect(llamaEmptyCard.getByRole('button', { name: '日志', exact: true })).toHaveCount(0)
    await page.evaluate(() => document.fonts.ready)

    await card('llama 空白配置').getByRole('button', { name: '编辑配置', exact: true }).click()
    let dialog = page.getByRole('dialog', { name: '编辑加载配置', exact: true })
    const context = dialog.getByLabel('上下文长度', { exact: true })
    await expect(context).toHaveValue('')
    await expect(context).toHaveAttribute('placeholder', '默认 32768')
    await expect(dialog.getByLabel('GPU 层数', { exact: true })).toHaveValue('')
    await expect(dialog.getByLabel('GPU 层数', { exact: true })).toHaveAttribute('placeholder', '默认 66')
    await expect(dialog.getByLabel('GPU 卸载方式', { exact: true })).toHaveCount(0)
    await expect(dialog.getByLabel('并发槽位', { exact: true })).toHaveAttribute('placeholder', '默认 1')
    await expect(dialog.locator('.footer-note')).toBeEmpty()
    const kCache = dialog.getByLabel('K 缓存类型', { exact: true })
    const vCache = dialog.getByLabel('V 缓存类型', { exact: true })
    const llamaCacheValues = ['f32', 'f16', 'bf16', 'q8_0', 'q4_0', 'q4_1', 'iq4_nl', 'q5_0', 'q5_1']
    await expect(kCache).toBeVisible()
    await expect(vCache).toBeVisible()
    await expect(kCache).toHaveValue('')
    await expect(vCache).toHaveValue('')
    await expect(kCache.locator('option').first()).toHaveText('默认 q8_0')
    await expect(vCache.locator('option').first()).toHaveText('默认 q8_0')
    expect((await optionValues(kCache)).filter(Boolean).sort()).toEqual([...llamaCacheValues].sort())
    expect((await optionValues(vCache)).filter(Boolean).sort()).toEqual([...llamaCacheValues].sort())
    await nativeScreenshot('llama-empty-defaults.png')
    await context.fill('65536')
    await context.fill('')
    await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(dialog).toBeHidden()
    expect(saved.at(-1)!.parameters).toEqual({})

    await card('llama 空白配置').getByRole('button', { name: '编辑配置', exact: true }).click()
    dialog = page.getByRole('dialog', { name: '编辑加载配置', exact: true })
    const kCacheRetained = dialog.getByLabel('K 缓存类型', { exact: true })
    const vCacheRetained = dialog.getByLabel('V 缓存类型', { exact: true })
    await kCacheRetained.selectOption('q8_0')
    await vCacheRetained.selectOption('q4_0')
    await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(dialog).toBeHidden()
    expect(saved.at(-1)!.parameters).toEqual({ cache_type_k: 'q8_0', cache_type_v: 'q4_0' })

    await card('llama 空白配置').getByRole('button', { name: '编辑配置', exact: true }).click()
    dialog = page.getByRole('dialog', { name: '编辑加载配置', exact: true })
    await expect(dialog.getByLabel('K 缓存类型', { exact: true })).toHaveValue('q8_0')
    await expect(dialog.getByLabel('V 缓存类型', { exact: true })).toHaveValue('q4_0')
    await dialog.getByLabel('上下文长度', { exact: true }).fill('0')
    await dialog.getByLabel('GPU 层数', { exact: true }).fill('0')
    await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(dialog).toBeHidden()
    expect(saved.at(-1)!.parameters).toEqual({ context_length: 0, gpu_layers: 0, cache_type_k: 'q8_0', cache_type_v: 'q4_0' })

    // Numeric endpoints, direct typing, validation, and untouched legacy representations.
    for (const value of [23, 66]) {
      await card('llama 空白配置').getByRole('button', { name: '编辑配置', exact: true }).click()
      dialog = page.getByRole('dialog')
      const gpu = dialog.getByLabel('GPU 层数', { exact: true })
      await expect(gpu).toBeEnabled()
      await gpu.fill(String(value))
      await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
      await expect(dialog).toBeHidden()
      expect(saved.at(-1)!.parameters.gpu_layers).toBe(value)
    }
    await card('llama 空白配置').getByRole('button', { name: '编辑配置', exact: true }).click()
    dialog = page.getByRole('dialog')
    const gpu = dialog.getByLabel('GPU 层数', { exact: true })
    await expect(gpu).toHaveValue('66')
    await gpu.press('ArrowUp')
    await expect(gpu).toHaveValue('66')
    await gpu.fill('0')
    await gpu.press('ArrowDown')
    await expect(gpu).toHaveValue('0')
    for (const value of ['67', '-1', '1.5']) {
      const count = saved.length
      await gpu.fill(value)
      await expect(gpu).toHaveValue(value)
      await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
      await expect(gpu).toHaveAttribute('aria-invalid', 'true')
      expect(saved.length).toBe(count)
    }
    await gpu.fill('')
    await gpu.pressSequentially('66')
    await expect(gpu).toHaveValue('66')
    await dialog.getByLabel('运行环境', { exact: true }).selectOption('runtime-llama_cpp-cpu')
    await expect(gpu).toHaveAttribute('max', '0')
    await expect(gpu).toHaveAttribute('aria-invalid', 'true')
    await dialog.getByLabel('运行环境', { exact: true }).selectOption('runtime-llama_cpp')
    await expect(gpu).toHaveAttribute('max', '66')
    await gpu.fill('')
    await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(dialog).toBeHidden()
    expect(saved.at(-1)!.parameters).not.toHaveProperty('gpu_layers')
    for (const legacy of [-1, 'all', 'auto', 999, null]) {
      profiles = profiles.map(p => p.id === 'llama-empty' ? { ...p, parameters: { gpu_layers: legacy } } : p)
      await page.getByRole('button', { name: '刷新状态', exact: true }).click()
      await card('llama 空白配置').getByRole('button', { name: '编辑配置', exact: true }).click()
      dialog = page.getByRole('dialog')
      const gpu = dialog.getByLabel('GPU 层数', { exact: true })
      await expect(gpu).toBeEnabled()
      await expect(gpu).toHaveValue(legacy === 'auto' || legacy === null ? '' : '66')
      await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
      await expect(dialog).toBeHidden()
      expect(saved.at(-1)!.parameters).toEqual({ gpu_layers: legacy })
    }

    await card('Strata 导入配置').getByRole('button', { name: '编辑配置', exact: true }).click()
    dialog = page.getByRole('dialog', { name: '编辑加载配置', exact: true })
    await expect(dialog.getByLabel('上下文长度', { exact: true })).toHaveValue('262144')
    const strataCache = dialog.getByLabel('KV 缓存', { exact: true })
    await expect(strataCache).toHaveValue('int8')
    const strataCacheValues = ['int8', 'fp16', 'q4_0', 'k8v4']
    expect((await optionValues(strataCache)).filter(Boolean).sort()).toEqual([...strataCacheValues].sort())
    await expect(strataCache.locator('option[value="k8v4"]')).toHaveText(/K.*INT8.*V.*Q4_0/i)
    await expect(dialog.locator('.footer-note')).toBeEmpty()
    await dialog.getByRole('button', { name: '文件与启动选项', exact: true }).click()
    await expect(dialog).toBeVisible()
    await expect(dialog.getByLabel('专家缓存', { exact: true })).toHaveValue('auto')
    await expect(dialog.getByLabel('显存预留 MiB', { exact: true })).toHaveValue('2048')
    const notificationClose = page.getByRole('button', { name: '关闭通知', exact: true })
    if (await notificationClose.isVisible()) await notificationClose.click()
    await nativeScreenshot('strata-imported-values.png')
    await dialog.getByRole('tab', { name: '生成默认值', exact: true }).click()
    await expect(dialog.locator('.footer-note')).toBeEmpty()
    await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(dialog).toBeHidden()
    expect(saved.at(-1)!.parameters).toEqual({ context_length: 262144, kv_cache: 'int8', expert_cache: 'auto', vram_reserve_mib: 2048, strata_config: strataConfig })

    await card('Strata 导入配置').getByRole('button', { name: '编辑配置', exact: true }).click()
    dialog = page.getByRole('dialog', { name: '编辑加载配置', exact: true })
    const selectedStrataCache = dialog.getByLabel('KV 缓存', { exact: true })
    await expect(selectedStrataCache).toHaveValue('int8')
    await selectedStrataCache.selectOption('k8v4')
    await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(dialog).toBeHidden()
    expect(saved.at(-1)!.parameters.kv_cache).toBe('k8v4')

    await card('Strata 导入配置').getByRole('button', { name: '编辑配置', exact: true }).click()
    dialog = page.getByRole('dialog', { name: '编辑加载配置', exact: true })
    await expect(dialog.getByLabel('KV 缓存', { exact: true })).toHaveValue('k8v4')
    await dialog.getByRole('button', { name: '取消', exact: true }).click()

    await card('Strata q4 别名导入配置').getByRole('button', { name: '编辑配置', exact: true }).click()
    dialog = page.getByRole('dialog', { name: '编辑加载配置', exact: true })
    const strataLegacyCache = dialog.getByLabel('KV 缓存', { exact: true })
    await expect(strataLegacyCache).toHaveValue('q4')
    const savedAlias = strataLegacyCache.locator('option[value="q4"]')
    await expect(savedAlias).toHaveCount(1)
    await expect(savedAlias).toHaveText(/q4.*已保存值/i)
    await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(dialog).toBeHidden()
    expect(saved.at(-1)!.parameters.kv_cache).toBe('q4')

    await card('Strata q4 别名导入配置').getByRole('button', { name: '编辑配置', exact: true }).click()
    dialog = page.getByRole('dialog', { name: '编辑加载配置', exact: true })
    await expect(dialog.getByLabel('KV 缓存', { exact: true })).toHaveValue('q4')
    await expect(dialog.getByLabel('KV 缓存', { exact: true }).locator('option[value="q4"]')).toHaveText(/q4.*已保存值/i)
    await dialog.getByRole('button', { name: '取消', exact: true }).click()

    // Saving locks the native form controls and keeps the modal mounted until its PATCH completes.
    const editorButton = card('Strata 导入配置').getByRole('button', { name: '编辑配置', exact: true })
    await editorButton.click()
    dialog = page.getByRole('dialog', { name: '编辑加载配置', exact: true })
    const profileName = dialog.getByLabel('配置名称', { exact: true })
    const strataContext = dialog.getByLabel('上下文长度', { exact: true })
    const strataCacheForSave = dialog.getByLabel('KV 缓存', { exact: true })
    const generationTab = dialog.getByRole('tab', { name: '生成默认值', exact: true })
    await expect(profileName).toBeFocused()
    await strataContext.fill('131072')
    await strataCacheForSave.selectOption('fp16')
    const successfulGate = holdNextPatch(false)
    const requestsBeforeDelayedSave = patchRequests
    await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
    await successfulGate.started
    const saveButton = dialog.getByRole('button', { name: '保存配置', exact: true })
    const cancelButton = dialog.getByRole('button', { name: '取消', exact: true })
    const closeButton = dialog.getByRole('button', { name: '关闭', exact: true })
    await expect(dialog.locator('fieldset.profile-editor-fields')).toHaveJSProperty('disabled', true)
    await expect(profileName).toBeDisabled()
    await expect(strataContext).toBeDisabled()
    await expect(strataCacheForSave).toBeDisabled()
    await expect(generationTab).toBeDisabled()
    await expect(saveButton).toBeDisabled()
    await expect(cancelButton).toBeDisabled()
    await expect(closeButton).toBeDisabled()
    await page.keyboard.press('Tab')
    await expect.poll(() => dialog.evaluate(element => element.contains(document.activeElement))).toBe(true)
    await page.keyboard.press('Shift+Tab')
    await expect.poll(() => dialog.evaluate(element => element.contains(document.activeElement))).toBe(true)
    await page.keyboard.press('Escape')
    await expect(dialog).toBeVisible()
    await page.locator('.modal-overlay').click({ position: { x: 8, y: 8 } })
    await expect(dialog).toBeVisible()
    await closeButton.evaluate(button => (button as HTMLButtonElement).click())
    await expect(dialog).toBeVisible()
    await saveButton.evaluate(button => (button as HTMLButtonElement).click())
    expect(patchRequests).toBe(requestsBeforeDelayedSave + 1)
    successfulGate.release()
    await expect(dialog).toBeHidden()
    expect(saved.at(-1)!.parameters.context_length).toBe(131072)
    expect(saved.at(-1)!.parameters.kv_cache).toBe('fp16')

    await editorButton.click()
    dialog = page.getByRole('dialog', { name: '编辑加载配置', exact: true })
    await expect(dialog.getByLabel('上下文长度', { exact: true })).toHaveValue('131072')
    await expect(dialog.getByLabel('KV 缓存', { exact: true })).toHaveValue('fp16')
    await dialog.getByRole('button', { name: '取消', exact: true }).click()

    // A failed save keeps the draft available; it can be edited, dismissed, then retried.
    await editorButton.click()
    dialog = page.getByRole('dialog', { name: '编辑加载配置', exact: true })
    const failedContext = dialog.getByLabel('上下文长度', { exact: true })
    const failedCache = dialog.getByLabel('KV 缓存', { exact: true })
    await failedContext.fill('65536')
    await failedCache.selectOption('q4_0')
    const failedSaveCount = saved.length
    const failureGate = holdNextPatch(true)
    await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
    await failureGate.started
    failureGate.release()
    await expect(dialog).toBeVisible()
    await expect(failedContext).toHaveValue('65536')
    await expect(failedCache).toHaveValue('q4_0')
    await expect(failedContext).toBeEnabled()
    await expect(failedCache).toBeEnabled()
    await expect(dialog.getByRole('tab', { name: '生成默认值', exact: true })).toBeEnabled()
    await expect(dialog.getByRole('button', { name: '保存配置', exact: true })).toBeEnabled()
    await expect(dialog.getByRole('button', { name: '取消', exact: true })).toBeEnabled()
    await expect(dialog.getByRole('button', { name: '关闭', exact: true })).toBeEnabled()
    expect(saved).toHaveLength(failedSaveCount)
    await failedContext.fill('32768')
    await expect(failedContext).toHaveValue('32768')
    const patchesBeforeFailureDismiss = patchRequests
    await dialog.getByRole('button', { name: '取消', exact: true }).click()
    await expect(dialog).toBeHidden()
    expect(patchRequests).toBe(patchesBeforeFailureDismiss)

    await editorButton.click()
    dialog = page.getByRole('dialog', { name: '编辑加载配置', exact: true })
    await expect(dialog.getByLabel('上下文长度', { exact: true })).toHaveValue('131072')
    await expect(dialog.getByLabel('KV 缓存', { exact: true })).toHaveValue('fp16')
    await dialog.getByLabel('上下文长度', { exact: true }).fill('65536')
    await dialog.getByLabel('KV 缓存', { exact: true }).selectOption('q4_0')
    await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(dialog).toBeHidden()
    expect(saved.at(-1)!.parameters.context_length).toBe(65536)
    expect(saved.at(-1)!.parameters.kv_cache).toBe('q4_0')
    await editorButton.click()
    dialog = page.getByRole('dialog', { name: '编辑加载配置', exact: true })
    await expect(dialog.getByLabel('上下文长度', { exact: true })).toHaveValue('65536')
    await expect(dialog.getByLabel('KV 缓存', { exact: true })).toHaveValue('q4_0')
    const finalPatchCount = patchRequests
    await dialog.getByLabel('上下文长度', { exact: true }).fill('8192')
    await dialog.getByRole('button', { name: '取消', exact: true }).click()
    await expect(dialog).toBeHidden()
    expect(patchRequests).toBe(finalPatchCount)
    await expect(editorButton).toBeFocused()

    profiles = profiles.map(profile => profile.id === 'llama-empty' ? { ...profile, parameters: { startup_timeout: 240 } } : profile)
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    const llamaEditButton = llamaEmptyCard.getByRole('button', { name: '编辑配置', exact: true })
    await llamaEditButton.click()
    dialog = page.getByRole('dialog', { name: '编辑加载配置', exact: true })
    const startupTimeout = dialog.getByLabel('启动等待时间（秒）', { exact: true })
    const advancedToggle = dialog.getByRole('button', { name: '文件与启动选项', exact: true })
    await expect(advancedToggle).toHaveAttribute('aria-expanded', 'true')
    await expect(startupTimeout).toHaveValue('240')
    await advancedToggle.click()
    await expect(advancedToggle).toHaveAttribute('aria-expanded', 'false')
    await expect(startupTimeout).toBeHidden()
    await advancedToggle.click()
    await expect(startupTimeout).toHaveValue('240')
    const patchesBeforeTimeoutCancel = patchRequests
    await dialog.getByRole('button', { name: '取消', exact: true }).click()
    await expect(dialog).toBeHidden()
    expect(patchRequests).toBe(patchesBeforeTimeoutCancel)

    profiles = profiles.map(profile => profile.id === 'llama-empty' ? { ...profile, parameters: { startup_timeout: 0 } } : profile)
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await llamaEditButton.click()
    dialog = page.getByRole('dialog', { name: '编辑加载配置', exact: true })
    const zeroTimeout = dialog.getByLabel('启动等待时间（秒）', { exact: true })
    await expect(dialog.getByRole('button', { name: '文件与启动选项', exact: true })).toHaveAttribute('aria-expanded', 'true')
    await expect(zeroTimeout).toHaveValue('0')
    await dialog.getByRole('button', { name: '取消', exact: true }).click()
    await expect(dialog).toBeHidden()

    profiles = profiles.map(profile => profile.id === 'llama-empty' ? { ...profile, parameters: {} } : profile)
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await llamaEditButton.click()
    dialog = page.getByRole('dialog', { name: '编辑加载配置', exact: true })
    const emptyTimeoutToggle = dialog.getByRole('button', { name: '文件与启动选项', exact: true })
    await expect(emptyTimeoutToggle).toHaveAttribute('aria-expanded', 'false')
    await emptyTimeoutToggle.click()
    const editableTimeout = dialog.getByLabel('启动等待时间（秒）', { exact: true })
    await expect(editableTimeout).toBeVisible()
    await editableTimeout.fill('240')
    await expect(editableTimeout).toHaveValue('240')
    const patchesBeforeEmptyTimeoutCancel = patchRequests
    await dialog.getByRole('button', { name: '取消', exact: true }).click()
    await expect(dialog).toBeHidden()
    expect(patchRequests).toBe(patchesBeforeEmptyTimeoutCancel)

    expect(forbiddenRequests).toEqual([])
    expect(errors).toEqual([])
    console.log(JSON.stringify({ screenshots, profileSaves: saved.length, errors }))
  } finally {
    controlledPatch?.release()
    await application.close()
  }
})
