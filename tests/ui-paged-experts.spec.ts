import { test, expect, _electron as electron } from '@playwright/test'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { spawnSync } from 'node:child_process'
import { pinNavigation } from './navigation'

test('专家分页按环境展示，关闭时保留配置且不校验休眠 K', async () => {
  test.setTimeout(90_000)
  const root = path.resolve(__dirname, '..')
  const artifacts = path.join(root, '.artifacts/codex/engine-feature-completion-20261103/root/ui')
  fs.mkdirSync(artifacts, { recursive: true })
  const directory = fs.mkdtempSync(path.join(artifacts, 'paged-'))
  const data = path.join(directory, 'data')
  fs.mkdirSync(data)
  const seed = spawnSync(path.join(root, '.venv/Scripts/python.exe'), ['-c',
    "import sqlite3,sys; c=sqlite3.connect(sys.argv[1]); c.execute('CREATE TABLE records (kind TEXT NOT NULL,id TEXT NOT NULL,body TEXT NOT NULL,PRIMARY KEY(kind,id))'); c.execute('INSERT INTO records VALUES (?,?,?)',('settings','main','{\"id\":\"main\",\"api_port\":0}')); c.commit(); c.close()",
    path.join(data, 'modloader.sqlite3')], { encoding: 'utf8' })
  expect(seed.status, seed.stderr).toBe(0)
  const packaged = process.env.MODLOADER_TEST_EXECUTABLE
  const application = await electron.launch({ ...(packaged ? { executablePath: packaged, args: [] } : { args: [root] }),
    env: { ...process.env, MODLOADER_DATA_DIR: data, MODLOADER_SENSOR_AUTO_START: '0',
      MODLOADER_STRATA_DEV_RUNTIME: '', MODLOADER_STRATA_DEV_EXE: '',
      MODLOADER_DEV_URL: process.env.MODLOADER_TEST_UI_URL || '' }, timeout: 60_000 })
  const page = await application.firstWindow().catch(async error => {
    await application.evaluate(({ app }) => app.exit())
    await application.close()
    throw error
  })
  page.setDefaultTimeout(10_000)
  const parameters = { enable_paged_experts: false, paged_experts_num_resident: -7,
    moe_runner_backend: 'marlin', acceleration_mode: 'mtp', context_length: 8192 }
  let profiles = [{ id: 'new', model_id: 'model', name: '分页配置', backend: 'sglang', runtime_id: 'new', alias: 'paged',
    parameters: { ...parameters }, generation: {}, extra_args: ['--log-level', 'warning'], revision: 1 },
  { id: 'old', model_id: 'model', name: '旧版配置', backend: 'sglang', runtime_id: 'old', alias: 'old',
    parameters: { acceleration_mode: 'mtp', context_length: 8192 }, generation: {}, extra_args: [], revision: 1 },
  { id: 'advanced', model_id: 'model', name: '高级参数分页', backend: 'sglang', runtime_id: 'new', alias: 'advanced',
    parameters: { acceleration_mode: 'mtp', context_length: 8192, enable_paged_experts: false,
      paged_experts_num_resident: 8, moe_runner_backend: 'triton' }, generation: {}, revision: 1,
    extra_args: ['--enable-paged-experts', '--paged-experts-num-resident', '16', '--moe-runner-backend', 'marlin', '--log-level', 'error'] }]
  const writes: any[] = [], errors: string[] = []
  const field = (key: string, label: string, type: string, extra = {}) => ({ key, label, type, options: [], special_values: [], ...extra })
  const pagingFields = [field('enable_paged_experts', 'Paged Experts', 'boolean', { default: false }),
    field('paged_experts_num_resident', '每层驻留专家 K', 'number', { minimum: 4, integer: true, step: 1 }),
    field('moe_runner_backend', 'MoE 后端', 'text'), field('disable_shared_experts_fusion', '禁用共享专家融合', 'boolean')]
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/api/state', async route => {
    const response = await route.fetch(), state = await response.json()
    await route.fulfill({ response, json: { ...state, profiles, instances: [], downloads: [], installations: [],
      models: [{ id: 'model', name: 'MoE Fixture', path: path.join(directory, 'model'), size_bytes: 1024, format: 'hf',
        architecture: 'qwen3_moe', modalities: ['text'], files: [], metadata: {}, source: {}, warnings: [], managed: false }],
      runtimes: [{ id: 'new', version: '0.5.23' }, { id: 'old', version: '0.5.21' }].map(item => ({ ...item,
        name: 'SGLang ' + item.version, backend: 'sglang', path: '/opt/sglang/bin/python', platform: 'wsl2', managed: false,
        metadata: { distribution: 'ModLoader' } })) } })
  })
  await page.route('**/api/resources/estimate', route => route.fulfill({ json: { resources: [], notes: [], source: 'fixture' } }))
  await page.route('**/api/profiles/preflight', async route => {
    const body = route.request().postDataJSON()
    const legacy = body.id === 'advanced' && body.extra_args.includes('--enable-paged-experts')
    const p = legacy ? { ...body.parameters, enable_paged_experts: true, paged_experts_num_resident: 16, moe_runner_backend: 'marlin' } : body.parameters
    const enabled = p.enable_paged_experts === true
    const message = enabled && p.acceleration_mode === 'mtp' ? 'Paged Experts 不支持推测解码' : ''
    await route.fulfill({ json: { status: message ? 'error' : 'ready', source: null, candidates: [], errors: message ? [message] : [], message,
      parameters: [field('context_length', '上下文长度', 'number'), ...(body.runtime_id === 'new' ? pagingFields : [])],
      generation_parameters: [], paged_experts: { enabled, num_resident: p.paged_experts_num_resident,
        runner: p.moe_runner_backend, recommended_runner: 'marlin', top_k: 4, num_experts: 128,
        normalized_parameters: p, normalized_extra_args: legacy ? ['--log-level', 'error'] : body.extra_args,
        status: message ? 'unsupported' : enabled ? 'available' : 'disabled', message },
      acceleration: { mode: p.acceleration_mode, enabled: p.acceleration_mode !== 'none', status: message ? 'unsupported' : 'ready',
        kind: 'mtp', label: 'MTP', requires_draft: false, fields: [], message: '' } } })
  })
  await page.route(/\/api\/profiles\/(?:new|old|advanced)$/, async route => {
    const body = route.request().postDataJSON(), id = new URL(route.request().url()).pathname.split('/').at(-1)!
    writes.push(body); profiles = profiles.map(item => item.id === id ? { ...item, ...body } : item)
    await route.fulfill({ json: profiles.find(item => item.id === id) })
  })
  const card = (name: string) => page.locator('.profile-card').filter({ has: page.getByRole('heading', { name, exact: true }) })
  try {
    await pinNavigation(page)
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await card('分页配置').getByRole('button', { name: '编辑配置', exact: true }).click()
    const dialog = page.getByRole('dialog'), paging = dialog.getByLabel('Paged Experts', { exact: true })
    await expect(paging).toHaveValue('false')
    await expect(dialog.getByLabel('每层驻留专家 K', { exact: true })).toHaveCount(0)
    await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(dialog).toBeHidden()
    expect(writes[0].parameters).toEqual(parameters)
    expect(writes[0].extra_args).toEqual(['--log-level', 'warning'])
    await card('分页配置').getByRole('button', { name: '编辑配置', exact: true }).click()
    await paging.selectOption('true')
    await expect(dialog.getByText('Paged Experts 不支持推测解码', { exact: true })).toBeVisible()
    await expect(dialog.getByLabel('推测加速', { exact: true })).toHaveValue('mtp')
    await dialog.getByLabel('每层驻留专家 K', { exact: true }).fill('16')
    await dialog.getByLabel('推测加速', { exact: true }).selectOption('none')
    await expect(dialog.getByText('Paged Experts 不支持推测解码', { exact: true })).toHaveCount(0)
    await expect(dialog.getByLabel('MoE 后端', { exact: true })).toHaveValue('marlin')
    await expect(dialog.getByText('正在检查模型准备文件', { exact: true })).toHaveCount(0)
    await expect(dialog.getByLabel('模型准备', { exact: true })).toHaveCount(0)
    await page.screenshot({ path: path.join(directory, 'paged-experts.png'), animations: 'disabled' })
    await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(dialog).toBeHidden()
    expect(writes[1].parameters).toMatchObject({ enable_paged_experts: true, paged_experts_num_resident: 16,
      moe_runner_backend: 'marlin', acceleration_mode: 'none', context_length: 8192 })
    expect(writes[1].extra_args).toEqual(['--log-level', 'warning'])
    await card('旧版配置').getByRole('button', { name: '编辑配置', exact: true }).click()
    await expect(paging).toHaveCount(0)
    await expect(dialog.getByLabel('推测加速', { exact: true })).toHaveValue('mtp')
    await dialog.getByRole('button', { name: '取消', exact: true }).click()
    await card('高级参数分页').getByRole('button', { name: '编辑配置', exact: true }).click()
    await expect(paging).toHaveValue('true')
    await expect(dialog.getByLabel('每层驻留专家 K', { exact: true })).toHaveValue('16')
    await dialog.getByLabel('上下文长度', { exact: true }).fill('16384')
    await paging.selectOption('false')
    await expect(dialog.getByLabel('每层驻留专家 K', { exact: true })).toHaveCount(0)
    await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(dialog).toBeHidden()
    expect(writes[2].parameters).toMatchObject({ enable_paged_experts: false, paged_experts_num_resident: 16,
      moe_runner_backend: 'marlin', acceleration_mode: 'mtp', context_length: 16384 })
    expect(writes[2].extra_args).toEqual(['--log-level', 'error'])
    expect(errors).toEqual([])
    if (packaged) expect(await application.evaluate(({ app }) => app.isPackaged)).toBe(true)
  } finally { await application.evaluate(({ app }) => app.exit()); await application.close() }
})
