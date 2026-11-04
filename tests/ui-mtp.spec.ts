import { test, expect, _electron as electron } from '@playwright/test'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { pinNavigation } from './navigation'
import type { LoadProfile, ParameterField } from '../src/types'

test('复制完整配置、MTP 回显与显式关闭、跨后端能力和缺文件入口', async () => {
  test.setTimeout(120_000)
  const root = path.resolve(__dirname, '..')
  const directory = fs.mkdtempSync(path.join(root, 'work', 'ui-mtp-'))
  const application = await electron.launch({
    ...(process.env.MODLOADER_TEST_EXECUTABLE ? { executablePath: process.env.MODLOADER_TEST_EXECUTABLE, args: [] } : { args: [root] }),
    env: { ...process.env, MODLOADER_DATA_DIR: path.join(directory, 'data') }, timeout: 60_000,
  })
  const page = await application.firstWindow()
  const errors: string[] = [], writes: Array<{ method: string; body: LoadProfile }> = []
  page.on('pageerror', error => errors.push(error.message))
  const mtpPath = 'D:\\Fixture\\MTP rt'
  const original: LoadProfile = { id: 'original', model_id: 'gguf', name: '原始完整配置', backend: 'strata', runtime_id: 'strata', alias: 'original',
    parameters: { context_length: 262144, kv_cache: 'k8v4', strata_config: { args: ['--mtp', mtpPath, '--resident-experts', '--spec', '4'] }, custom_nested: { keep: [1, 2] } },
    generation: { temperature: .6, top_p: .95, custom_generation: { keep: true } }, extra_args: ['--spec-min-p', '0.5'], revision: 3 }
  let profiles = [structuredClone(original)]
  let gpuMaximum: number | null = 49
  let firstPreflight = true, originalChecks = 0
  let releaseFirstCheck: () => void = () => {}
  const firstCheckGate = new Promise<void>(resolve => { releaseFirstCheck = resolve })
  let backendFields: Record<string, ParameterField[]> = {}
  const models = [{ id: 'gguf', name: '01 原始模型', format: 'gguf' }, { id: 'hf', name: '02 HF 模型', format: 'hf' }, { id: 'missing', name: '03 未配套模型', format: 'gguf' }].map(model => ({ ...model,
    path: path.join(directory, model.id + '.gguf'), size_bytes: 1024, files: [], architecture: 'qwen4exp', modalities: ['text'], metadata: {}, source: {}, managed: false, warnings: [] }))
  await page.route('**/api/state', async route => {
    const response = await route.fetch(), state = await response.json()
    backendFields = Object.fromEntries(state.backends.map((backend: { id: string; parameters: ParameterField[] }) => [backend.id, backend.parameters]))
    await route.fulfill({ response, json: { ...state, models, profiles, instances: [], downloads: [], installations: [],
      runtimes: state.backends.map((backend: { id: string; label: string }) => ({ id: backend.id, backend: backend.id, name: backend.label, version: 'UI fixture', path: directory, managed: false, metadata: {} })) } })
  })
  await page.route('**/api/resources/estimate', route => route.fulfill({ json: { resources: [], notes: [], source: 'ui-fixture' } }))
  await page.route('**/api/profiles/preflight', async route => {
    const body = route.request().postDataJSON(), p = body.parameters
    if (body.id === original.id) {
      originalChecks++
      if (firstPreflight) { firstPreflight = false; await firstCheckGate }
    }
    const args = p.strata_config?.args || []
    const inheritedPath = args.includes('--mtp') ? args[args.indexOf('--mtp') + 1] : ''
    const resolvedPath = p.mtp_path ?? inheritedPath
    const candidates = body.backend === 'strata' && body.model_id === 'gguf' && !resolvedPath ? [{ label: original.name, path: mtpPath }] : []
    const supported = body.backend !== 'transformers'
    const status = !supported ? 'unsupported' : body.backend !== 'strata' ? 'builtin' : resolvedPath || candidates.length ? 'external' : 'missing'
    await route.fulfill({ json: { status: 'ready', message: '使用已准备的本地文件', source: { label: original.name, kind: 'imported' }, candidates: [], errors: [], vision_enabled: false,
      parameters: (backendFields[body.backend] || []).map(field => field.key === 'gpu_layers' ? { ...field, maximum: gpuMaximum } : field),
      mtp: { status, message: !supported ? '当前服务入口未支持原生 MTP' : status === 'missing' ? '需要配套 MTP 运行目录；启用不会自动下载或转换。' : status === 'builtin' ? '权重内含 MTP 张量，无需另外下载草稿模型。' : '同一模型已有配套文件。',
        tokens: p.mtp_tokens ?? (body.backend === 'strata' ? 3 : body.backend === 'vllm' ? 1 : null),
        token_field: { key: 'mtp_tokens', label: '最大预测 Token', type: 'number', minimum: 1, maximum: body.backend === 'strata' ? 7 : null, integer: true, step: 1, description: '每轮最多提前预测的 Token 数。', options: [], advanced: false, default: null },
        backend_supported: supported, enabled: p.mtp_enabled ?? !!inheritedPath, path: resolvedPath, candidates,
        path_kind: body.backend === 'strata' || body.backend === 'vllm' ? 'directory' : 'file', needs_path: body.backend === 'strata', resident_experts: p.resident_experts ?? args.includes('--resident-experts') } } })
  })
  await page.route(/\/api\/profiles(?:\/[^/]+)?$/, async route => {
    if (new URL(route.request().url()).pathname.endsWith('/preflight')) return route.fallback()
    const method = route.request().method()
    if (!['PATCH', 'POST'].includes(method)) return route.continue()
    const body = route.request().postDataJSON()
    writes.push({ method, body: structuredClone(body) })
    const id = method === 'POST' ? `new-${writes.length}` : new URL(route.request().url()).pathname.split('/').at(-1)!
    const saved = { ...body, id, revision: 1 }
    profiles = [...profiles.filter(item => item.id !== id), saved]
    await route.fulfill({ json: saved })
  })
  const card = (name: string) => page.locator('.profile-card').filter({ has: page.getByRole('heading', { name, exact: true }) })
  const dialog = () => page.getByRole('dialog')
  try {
    await pinNavigation(page)
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await expect(card(original.name)).toBeVisible()
    await expect.poll(() => originalChecks).toBe(1)
    await card(original.name).getByRole('button', { name: '复制配置', exact: true }).click()
    await expect(dialog()).toHaveAttribute('aria-label', '复制加载配置')
    await expect(dialog().locator('.profile-acceleration [role="status"]')).toHaveText('检查中…')
    await page.waitForTimeout(300) // A second request would have passed the edit debounce by now.
    expect(originalChecks).toBe(1)
    releaseFirstCheck()
    await expect(dialog().getByLabel('配置名称', { exact: true })).toHaveValue('原始完整配置 · 副本')
    await expect(dialog().getByLabel('API 模型名称', { exact: true })).toHaveValue('original-copy')
    await expect(dialog().getByLabel('MTP 推测加速', { exact: true })).toBeChecked()
    await expect(dialog().getByLabel('MTP 运行目录', { exact: true })).toHaveValue(mtpPath)
    await expect(dialog().getByLabel('最大预测 Token', { exact: true })).toHaveAttribute('placeholder', '默认 3')
    await dialog().getByRole('button', { name: '文件与启动选项', exact: true }).click()
    await expect(dialog().getByLabel('专家常驻 RAM', { exact: true })).toBeChecked()
    await page.screenshot({ path: path.join(directory, 'copy-inherited.png') })
    await dialog().getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(dialog()).toBeHidden()
    expect(writes[0].method).toBe('POST')
    for (const key of ['model_id', 'runtime_id', 'backend', 'parameters', 'generation', 'extra_args'] as const) expect(writes[0].body[key]).toEqual(original[key])
    expect(profiles.find(item => item.id === 'original')).toEqual(original)

    await card(original.name).getByRole('button', { name: '复制配置', exact: true }).click()
    await expect(dialog().getByLabel('API 模型名称', { exact: true })).toHaveValue('original-copy-2')
    await expect(dialog().getByLabel('配置名称', { exact: true })).toHaveValue('原始完整配置 · 副本 2')
    await dialog().getByRole('button', { name: '取消', exact: true }).click()
    await card('原始完整配置 · 副本').getByRole('button', { name: '编辑配置', exact: true }).click()
    await expect(dialog().getByLabel('MTP 推测加速', { exact: true })).toBeChecked()
    const depth = dialog().getByLabel('最大预测 Token', { exact: true })
    await depth.fill('8')
    await dialog().getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(depth).toHaveAttribute('aria-invalid', 'true')
    expect(writes.length).toBe(1)
    await depth.fill('7')
    await depth.press('ArrowUp')
    await expect(depth).toHaveValue('7')
    await depth.hover()
    await page.mouse.wheel(0, -120)
    await expect(depth).toHaveValue('7')
    await depth.fill('2.5')
    await expect(depth).toHaveAttribute('aria-invalid', 'true')
    await depth.fill('5')
    await dialog().getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(dialog()).toBeHidden()
    expect(writes.at(-1)!.body.parameters.mtp_tokens).toBe(5)
    await card('原始完整配置 · 副本').getByRole('button', { name: '编辑配置', exact: true }).click()
    await expect(dialog().getByLabel('最大预测 Token', { exact: true })).toHaveValue('5')
    await dialog().getByLabel('最大预测 Token', { exact: true }).fill('')
    await expect(dialog().getByLabel('最大预测 Token', { exact: true })).toHaveAttribute('placeholder', '默认 3')
    await dialog().getByLabel('最大预测 Token', { exact: true }).fill('2.5')
    await dialog().getByLabel('MTP 推测加速', { exact: true }).uncheck()
    await dialog().getByRole('button', { name: '文件与启动选项', exact: true }).click()
    await dialog().getByLabel('专家常驻 RAM', { exact: true }).uncheck()
    await dialog().getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(dialog()).toBeHidden()
    expect(writes.at(-1)!.body.parameters).toMatchObject({ mtp_enabled: false, mtp_tokens: 2.5, resident_experts: false, strata_config: original.parameters.strata_config })
    expect(profiles.find(item => item.id === 'original')).toEqual(original)

    // An unknown layer count is not a hardware limit. Keep editing and saving
    // available, while still rejecting invalid numeric domains.
    gpuMaximum = null
    await page.getByRole('button', { name: '新建配置', exact: true }).first().click()
    const unknownGpu = dialog().getByLabel('GPU 层数', { exact: true })
    await expect(unknownGpu).toBeEnabled()
    await expect(unknownGpu).not.toHaveAttribute('max')
    await unknownGpu.fill('-2')
    await expect(unknownGpu).toHaveAttribute('aria-invalid', 'true')
    await unknownGpu.fill('1000')
    await dialog().getByLabel('CPU 线程', { exact: true }).fill('1000')
    await dialog().getByLabel('上下文长度', { exact: true }).fill('1048576')
    await dialog().getByRole('tab', { name: '生成默认值', exact: true }).click()
    await dialog().getByLabel('温度', { exact: true }).fill('4')
    await dialog().getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(dialog()).toBeHidden()
    expect(writes.at(-1)!.body.parameters).toMatchObject({ gpu_layers: 1000, threads: 1000, context_length: 1048576 })
    expect(writes.at(-1)!.body.generation.temperature).toBe(4)
    gpuMaximum = 49

    await page.getByRole('button', { name: '新建配置', exact: true }).first().click()
    const gpuLayers = dialog().getByLabel('GPU 层数', { exact: true })
    await expect(gpuLayers).toHaveAttribute('max', '49')
    await gpuLayers.fill('1000')
    const savesBefore = writes.length
    await dialog().getByRole('button', { name: '保存配置', exact: true }).click()
    expect(writes.length).toBe(savesBefore)
    await expect(gpuLayers).toHaveAttribute('aria-invalid', 'true')
    await page.screenshot({ path: path.join(directory, 'gpu-out-of-range.png') })
    await gpuLayers.fill('49')
    await gpuLayers.press('ArrowUp')
    await expect(gpuLayers).toHaveValue('49')
    await expect(gpuLayers).toHaveValue('49')
    await expect(dialog().getByLabel('GPU 卸载方式', { exact: true })).toHaveCount(0)
    await dialog().getByLabel('上下文长度', { exact: true }).fill('-2')
    await dialog().getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(dialog().getByLabel('上下文长度', { exact: true })).toHaveAttribute('aria-invalid', 'true')
    await dialog().getByLabel('上下文长度', { exact: true }).fill('0')
    const threads = dialog().getByLabel('CPU 线程', { exact: true })
    await threads.fill('')
    await threads.pressSequentially('-1')
    await expect(threads).toHaveValue('-1')
    await dialog().getByRole('tab', { name: '生成默认值', exact: true }).click()
    await dialog().getByLabel('Top P', { exact: true }).fill('1.1')
    await dialog().getByRole('tab', { name: '加载参数', exact: true }).click()
    await dialog().getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(dialog().getByRole('tab', { name: '生成默认值', exact: true })).toHaveAttribute('aria-selected', 'true')
    expect(writes.length).toBe(savesBefore)
    await dialog().getByLabel('Top P', { exact: true }).fill('1')
    await dialog().getByRole('tab', { name: '加载参数', exact: true }).click()
    await dialog().getByLabel('运行后端', { exact: true }).selectOption('strata')
    await expect(dialog().getByLabel('MTP 推测加速', { exact: true })).toBeEnabled()
    await dialog().getByLabel('MTP 推测加速', { exact: true }).check()
    await expect(dialog().getByLabel('MTP 运行目录', { exact: true })).toHaveValue(mtpPath)
    await dialog().getByRole('button', { name: '取消', exact: true }).click()

    await page.getByRole('button', { name: /02 HF 模型/ }).click()
    await page.getByRole('button', { name: '新建配置', exact: true }).first().click()
    await expect(dialog().getByLabel('MTP 推测加速', { exact: true })).toBeDisabled()
    await dialog().getByLabel('运行后端', { exact: true }).selectOption('vllm')
    await expect(dialog().getByLabel('MTP 推测加速', { exact: true })).toBeEnabled()
    await dialog().getByLabel('MTP 推测加速', { exact: true }).check()
    await expect(dialog().getByLabel('MTP 配套权重', { exact: true })).toHaveValue('')
    await dialog().getByLabel('最大预测 Token', { exact: true }).fill('12')
    await dialog().getByLabel('GPU 显存比例', { exact: true }).fill('1.1')
    await dialog().getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(dialog().getByLabel('GPU 显存比例', { exact: true })).toHaveAttribute('aria-invalid', 'true')
    await dialog().getByLabel('GPU 显存比例', { exact: true }).fill('0.85')

    await dialog().getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(dialog()).toBeHidden()
    expect(writes.at(-1)!.body.parameters.mtp_enabled).toBe(true)
    expect(writes.at(-1)!.body.parameters.mtp_tokens).toBe(12)

    await page.getByRole('button', { name: /03 未配套模型/ }).click()
    await page.getByRole('button', { name: '新建配置', exact: true }).first().click()
    await dialog().getByLabel('运行后端', { exact: true }).selectOption('strata')
    await expect(dialog().locator('.profile-acceleration')).toContainText('缺少配套文件')
    await expect(dialog().getByRole('button', { name: '查找配套文件', exact: true })).toBeVisible()
    await dialog().getByLabel('MTP 推测加速', { exact: true }).check()
    await dialog().getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(dialog()).toContainText('启用 MTP 前请选择配套的本地文件')
    await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(940, 760))
    await expect(dialog().locator('.profile-acceleration')).toContainText('缺少配套文件')
    await dialog().locator('.profile-acceleration').scrollIntoViewIfNeeded()
    const toastClose = page.getByRole('button', { name: '关闭通知', exact: true })
    if (await toastClose.isVisible()) await toastClose.click()
    await page.screenshot({ path: path.join(directory, 'missing-940.png') })
    await dialog().getByRole('button', { name: '取消', exact: true }).click()
    await page.getByRole('button', { name: /01 原始模型/ }).click()
    await card(original.name).getByRole('button', { name: '复制配置', exact: true }).scrollIntoViewIfNeeded()
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth)).toBe(true)
    await page.screenshot({ path: path.join(directory, 'profile-actions-940.png') })
    expect(errors).toEqual([])
    console.log(JSON.stringify({ screenshots: directory, saves: writes.length, native: true, backend: 'UI fixture' }))
  } finally { releaseFirstCheck(); await application.close() }
})
