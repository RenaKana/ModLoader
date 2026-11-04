import { test, expect, _electron as electron } from '@playwright/test'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { pinNavigation } from './navigation'

test('Strata 四模式保存不改旧 MTP 数据，旧引擎仍用原控件', async () => {
  test.setTimeout(90_000)
  const root = path.resolve(__dirname, '..')
  const artifacts = path.join(root, '.artifacts/codex/engine-compat-20261103/ui')
  fs.mkdirSync(artifacts, { recursive: true })
  const directory = fs.mkdtempSync(path.join(artifacts, 'drafting-'))
  const application = await electron.launch({ args: [root], env: { ...process.env,
    MODLOADER_DATA_DIR: path.join(directory, 'data'), MODLOADER_SENSOR_AUTO_START: '0',
    ...(process.env.MODLOADER_TEST_UI_URL ? { MODLOADER_DEV_URL: process.env.MODLOADER_TEST_UI_URL } : {}) }, timeout: 60_000 })
  const page = await application.firstWindow()
  page.setDefaultTimeout(10_000)
  const oldParameters = { mtp_enabled: true, mtp_path: 'D:\\Fixture\\mtp\\rt',
    strata_config: { args: ['--mtp', 'D:\\Fixture\\mtp\\rt', '--spec', '4'] } }
  let profiles = [{ id: 'new-profile', model_id: 'model', name: '新版 Strata', backend: 'strata', runtime_id: 'new',
    alias: 'custom-alias', parameters: structuredClone(oldParameters) as Record<string, unknown>, generation: {}, extra_args: ['--raw-token-continuation'], revision: 1 },
    { id: 'old-profile', model_id: 'model', name: '旧版 Strata', backend: 'strata', runtime_id: 'old',
      alias: 'old-alias', parameters: structuredClone(oldParameters) as Record<string, unknown>, generation: {}, extra_args: [], revision: 1 },
    { id: 'import-profile', model_id: 'model', name: '导入 Auto 配置', backend: 'strata', runtime_id: 'new',
      alias: 'imported', parameters: { ...structuredClone(oldParameters), mtp_tokens: 99,
        strata_config: { ...oldParameters.strata_config, drafting: 'auto' } } as Record<string, unknown>, generation: {}, extra_args: [], revision: 1 }]
  const writes: any[] = [], errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/api/state', async route => {
    const response = await route.fetch(), state = await response.json()
    await route.fulfill({ response, json: { ...state, instances: [], downloads: [], installations: [], profiles,
      models: [{ id: 'model', name: 'Qwen3.8 Fixture', path: path.join(directory, 'model.gguf'), size_bytes: 1024,
        format: 'gguf', architecture: 'qwen4exp', modalities: ['text'], files: [], metadata: {}, source: {}, warnings: [], managed: false }],
      runtimes: [{ id: 'new', version: '0.1.56' }, { id: 'old', version: '0.1.48' }].map(r => ({ ...r,
        name: 'Strata ' + r.version, backend: 'strata', path: directory, managed: false, metadata: {} })) } })
  })
  await page.route('**/api/resources/estimate', route => route.fulfill({ json: { resources: [], notes: [], source: 'fixture' } }))
  await page.route('**/api/profiles/preflight', async route => {
    const body = route.request().postDataJSON(), p = body.parameters
    const mode = p.strata_drafting || p.strata_config?.drafting || 'legacy', supported = body.runtime_id === 'new'
    const effective = mode === 'legacy' ? 'mtp' : mode === 'auto' ? 'dflash2' : mode
    await route.fulfill({ json: { status: 'ready', source: null, candidates: [], errors: [], message: '', parameters: [],
      vision_enabled: false, drafting: { supported, mode, effective_mode: effective,
        status: effective === 'dflash2' && !p.dflash2_path ? 'needs_preparation' : 'ready',
        path: effective === 'dflash2' ? p.dflash2_path || '' : p.mtp_path, download_bytes: 1206141244 },
      mtp: { status: 'external', enabled: true, backend_supported: true, path: oldParameters.mtp_path,
        path_kind: 'directory', needs_path: true, candidates: [], resident_experts: false, message: '', tokens: 3,
        token_field: { key: 'mtp_tokens', label: '最大预测 Token', type: 'number', minimum: 1, maximum: 7, integer: true, options: [], special_values: [] } } } })
  })
  await page.route(/\/api\/profiles\/(?:new-profile|old-profile|import-profile)$/, async route => {
    const body = route.request().postDataJSON(), id = new URL(route.request().url()).pathname.split('/').at(-1)!
    writes.push(structuredClone(body)); profiles = profiles.map(p => p.id === id ? { ...p, ...body } : p)
    await route.fulfill({ json: profiles.find(p => p.id === id) })
  })
  const card = (name: string) => page.locator('.profile-card').filter({ has: page.getByRole('heading', { name, exact: true }) })
  try {
    await pinNavigation(page)
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await card('新版 Strata').getByRole('button', { name: '编辑配置', exact: true }).click()
    const dialog = page.getByRole('dialog'), mode = dialog.getByLabel('草稿模式', { exact: true })
    await expect(mode).toHaveValue('legacy')
    await mode.selectOption('dflash2')
    await expect(dialog.getByLabel('DFlash2 配套权重', { exact: true })).toBeVisible()
    await dialog.getByLabel('DFlash2 配套权重', { exact: true }).fill('D:\\Draft\\custom.pack')
    await expect(dialog.locator('.engine-acceleration-mode > [role="status"]')).toHaveText('')
    await page.screenshot({ path: path.join(directory, 'dflash2.png'), animations: 'disabled' })
    await mode.selectOption('off')
    await expect(dialog.getByLabel('DFlash2 配套权重', { exact: true })).toHaveCount(0)
    await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(dialog).toBeHidden()
    expect(writes[0].parameters).toMatchObject({ ...oldParameters, strata_drafting: 'off', dflash2_path: 'D:\\Draft\\custom.pack' })
    expect(writes[0].extra_args).toEqual(['--raw-token-continuation'])
    await card('新版 Strata').getByRole('button', { name: '编辑配置', exact: true }).click()
    await expect(mode).toHaveValue('off')
    await mode.selectOption('auto')
    await expect(dialog.getByLabel('DFlash2 配套权重', { exact: true })).toHaveValue('D:\\Draft\\custom.pack')
    await dialog.getByRole('button', { name: '取消', exact: true }).click()
    await card('旧版 Strata').getByRole('button', { name: '编辑配置', exact: true }).click()
    await expect(dialog.getByLabel('MTP 推测加速', { exact: true })).toBeChecked()
    await expect(mode).toHaveCount(0)
    await page.screenshot({ path: path.join(directory, 'legacy.png'), animations: 'disabled' })
    await dialog.getByRole('button', { name: '取消', exact: true }).click()
    expect(profiles.find(p => p.id === 'old-profile')!.parameters).toEqual(oldParameters)
    await card('导入 Auto 配置').getByRole('button', { name: '编辑配置', exact: true }).click()
    await expect(mode).toHaveValue('auto')
    await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(dialog).toBeHidden()
    expect(writes[1].parameters.mtp_tokens).toBe(99)
    expect(writes[1].parameters.strata_config.drafting).toBe('auto')
    expect(errors).toEqual([])
  } finally { await application.evaluate(({ app }) => app.exit()); await application.close() }
})

test('NInfer 新旧容量切换须显式移除，不转换单位或改动其他参数', async () => {
  test.setTimeout(90_000)
  const root = path.resolve(__dirname, '..'), artifacts = path.join(root, '.artifacts/codex/engine-compat-20261103/ui')
  fs.mkdirSync(artifacts, { recursive: true })
  const directory = fs.mkdtempSync(path.join(artifacts, 'capacity-'))
  const application = await electron.launch({ args: [root], env: { ...process.env,
    MODLOADER_DATA_DIR: path.join(directory, 'data'), MODLOADER_SENSOR_AUTO_START: '0',
    ...(process.env.MODLOADER_TEST_UI_URL ? { MODLOADER_DEV_URL: process.env.MODLOADER_TEST_UI_URL } : {}) }, timeout: 60_000 })
  const page = await application.firstWindow()
  let profile = { id: 'capacity', model_id: 'model', name: '容量迁移', backend: 'ninfer', runtime_id: 'new', alias: 'ninfer',
    parameters: { kv_capacity: 200000, context_length: 100000 } as Record<string, unknown>, generation: {}, extra_args: [], revision: 1 }
  const original = structuredClone(profile), writes: any[] = []
  const field = (key: string, label: string, type: string) => ({ key, label, type, options: [], special_values: [], advanced: false })
  await page.route('**/api/state', async route => {
    const response = await route.fetch(), state = await response.json()
    await route.fulfill({ response, json: { ...state, profiles: [profile], instances: [], downloads: [], installations: [],
      models: [{ id: 'model', name: 'NInfer Fixture', path: path.join(directory, 'model.ninfer'), size_bytes: 1024,
        format: 'ninfer', architecture: 'qwen3_5', modalities: ['text'], files: [], metadata: {}, source: {}, warnings: [], managed: false }],
      runtimes: [{ id: 'new', name: 'NInfer 新版', backend: 'ninfer', version: 'a754fcf3bf52', path: '/opt/new/ninfer-serve',
        platform: 'wsl2', managed: false, metadata: { distribution: 'ModLoader', supported_flags: ['--context-capacity'] } }] } })
  })
  await page.route('**/api/resources/estimate', route => route.fulfill({ json: { resources: [], notes: [], source: 'fixture' } }))
  await page.route('**/api/profiles/preflight', async route => {
    const p = route.request().postDataJSON().parameters
    const migration = p.kv_capacity == null ? undefined : { parameter: 'kv_capacity', message: '旧版 kv_capacity 使用 Token；新版 context_capacity 使用共享字节容量。请移除旧值后重新设置。' }
    await route.fulfill({ json: { status: migration ? 'error' : 'ready', source: null, candidates: [], errors: [], message: migration?.message || '',
      kind: 'ninfer_artifact', artifact: 'fixture.ninfer', parameter_migration: migration,
      parameters: [field('context_length', '上下文长度', 'number'), field('context_capacity', '上下文共享容量（字节）', 'text')],
      acceleration: { mode: 'none', enabled: false, status: 'disabled', kind: 'none', label: '', requires_draft: false, fields: [], message: '' } } })
  })
  await page.route('**/api/profiles/capacity', async route => {
    const body = route.request().postDataJSON(); writes.push(body); profile = { ...profile, ...body }
    await route.fulfill({ json: profile })
  })
  try {
    await pinNavigation(page)
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await page.getByRole('button', { name: '编辑配置', exact: true }).click()
    const dialog = page.getByRole('dialog')
    await expect(dialog.getByRole('button', { name: '移除旧容量参数', exact: true })).toBeVisible()
    await expect(dialog.getByText('旧版 kv_capacity 使用 Token；新版 context_capacity 使用共享字节容量。请移除旧值后重新设置。', { exact: true })).toHaveCount(1)
    await page.screenshot({ path: path.join(directory, 'capacity-migration.png'), animations: 'disabled' })
    await dialog.getByRole('button', { name: '移除旧容量参数', exact: true }).click()
    await expect(dialog.getByRole('button', { name: '移除旧容量参数', exact: true })).toHaveCount(0)
    expect(profile).toEqual(original)
    await dialog.getByLabel('上下文共享容量（字节）', { exact: true }).fill('2GiB')
    await expect(dialog.getByText('正在检查模型准备文件', { exact: true })).toHaveCount(0)
    await page.screenshot({ path: path.join(directory, 'capacity.png'), animations: 'disabled' })
    await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(dialog).toBeHidden()
    expect(writes[0].parameters).toEqual({ context_length: 100000, context_capacity: '2GiB' })
    expect(writes[0].extra_args).toEqual([])
  } finally { await application.evaluate(({ app }) => app.exit()); await application.close() }
})
