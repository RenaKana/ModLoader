import { test, expect, _electron as electron } from '@playwright/test'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { execFileSync } from 'node:child_process'
import { pinNavigation } from './navigation'
import type { LoadProfile, ParameterField } from '../src/types'

test('真实 Strata 单卡选卡：原配置副本经过预估接口匹配本机容量', async () => {
  const alias = process.env.MODLOADER_TEST_STRATA_PROFILE
  test.skip(!alias, '需要显式指定本机 Strata 配置；只读复制到隔离目录，不启动模型')
  test.setTimeout(90_000)
  const root = path.resolve(__dirname, '..'), directory = fs.mkdtempSync(path.join(root, 'work', 'strata-gpu-regression-'))
  const data = path.join(directory, 'data')
  execFileSync(path.join(root, '.venv', 'Scripts', 'python.exe'), ['-c', `
import json,sqlite3,sys
from pathlib import Path
source=sqlite3.connect(Path(sys.argv[1]).resolve().as_uri()+'?mode=ro',uri=True)
profile=json.loads(source.execute("SELECT body FROM records WHERE kind='profiles' AND json_extract(body,'$.alias')=?",(sys.argv[3],)).fetchone()[0])
assert profile['backend']=='strata'
records=[('profiles',profile)]
for kind,key in [('models','model_id'),('runtimes','runtime_id')]:
 records.append((kind,json.loads(source.execute('SELECT body FROM records WHERE kind=? AND id=?',(kind,profile[key])).fetchone()[0])))
source.close()
Path(sys.argv[2]).mkdir(parents=True)
target=sqlite3.connect(str(Path(sys.argv[2])/'modloader.sqlite3'))
target.execute('CREATE TABLE records (kind TEXT NOT NULL,id TEXT NOT NULL,body TEXT NOT NULL,PRIMARY KEY(kind,id))')
records.append(('settings',dict(id='main',api_port=0,close_to_tray=False)))
target.executemany('INSERT INTO records VALUES(?,?,?)',[(kind,item['id'],json.dumps(item)) for kind,item in records])
target.commit();target.close()
`, path.join(root, 'data', 'modloader.sqlite3'), data, alias!], { cwd: root })
  const application = await electron.launch({ args: [root], env: { ...process.env, MODLOADER_DATA_DIR: data, MODLOADER_SENSOR_AUTO_START: '0' }, timeout: 60_000 })
  const page = await application.firstWindow(), forecasts: unknown[] = []
  const errors: string[] = [], forbidden: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  page.on('response', async response => {
    if (new URL(response.url()).pathname === '/api/resources/estimate') {
      const value = await response.json().catch(() => null)
      forecasts.push({ resources: value?.resources?.filter((item: { kind: string }) => item.kind === 'vram').map((item: Record<string, unknown>) => ({ gpu_id: item.gpu_id, identity_status: item.identity_status, fixed_bytes: item.fixed_bytes })), diagnostic: value?.diagnostic })
    }
  })
  await page.route('**/api/**', async route => {
    const pathname = new URL(route.request().url()).pathname
    if (/\/(start|load|install|download|prepare)$/.test(pathname)) { forbidden.push(pathname); return route.abort() }
    return route.continue()
  })
  try {
    await pinNavigation(page)
    await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 30_000 })
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    const card = page.locator('.profile-card')
    await card.getByRole('button', { name: '编辑配置', exact: true }).click()
    const dialog = page.getByRole('dialog', { name: '编辑加载配置', exact: true })
    const group = dialog.locator('.profile-gpu')
    if (await group.getAttribute('open') === null) await group.locator('summary').click()
    await expect(group.locator('.gpu-device-choices button')).toHaveCount(1)
    await group.locator('input').fill('0')
    await expect.poll(() => forecasts.length, { timeout: 30_000 }).toBeGreaterThan(0)
    const vram = dialog.locator('.resource-preview-row[data-resource="vram"]')
    await expect(dialog.locator('.resource-preview')).toHaveAttribute('aria-busy', 'false', { timeout: 30_000 })
    await expect(vram).toHaveAttribute('data-gpu-id', /^GPU-/)
    await vram.scrollIntoViewIfNeeded()
    await expect(vram).not.toContainText('容量未读取')
    await expect(vram.getByRole('meter')).toBeVisible()
    await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(dialog).not.toBeVisible()
    const savedVram = card.locator('.resource-preview-row[data-resource="vram"]')
    await expect(card.locator('.resource-preview')).toHaveAttribute('aria-busy', 'false', { timeout: 30_000 })
    await expect(savedVram).toHaveAttribute('data-gpu-id', /^GPU-/)
    await expect(savedVram).not.toContainText('容量未读取')
    await expect(savedVram).not.toContainText('待估算')
    expect(forbidden).toEqual([]); expect(errors).toEqual([])
  } finally {
    const hardware = await page.evaluate(async () => {
      const connection = await window.modloader!.connection()
      const state = await (await fetch(connection.baseUrl + '/api/state', { headers: { Authorization: 'Bearer ' + connection.token } })).json()
      return { gpus: state.hardware.gpus?.map((gpu: Record<string, unknown>) => ({ id: gpu.id, memory_total: gpu.memory_total, memory_used: gpu.memory_used })), gpu_error: state.hardware.gpu_error }
    }).catch(() => null)
    console.log(JSON.stringify({ directory, forecasts, hardware }, null, 2))
    await page.screenshot({ path: path.join(directory, 'single-gpu.png') }).catch(() => {})
    await application.close()
  }
})

test('多卡原生配置保存回显，逐卡容量按 UUID 匹配，设备重排不串卡', async () => {
  test.setTimeout(120_000)
  const root = path.resolve(__dirname, '..'), directory = fs.mkdtempSync(path.join(root, 'work', 'ui-multi-gpu-'))
  const gib = 1024 ** 3
  const a = 'GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa', b = 'GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb'
  let hardware = [
    { id: a, gpu_id: a, index: 0, device: 'CUDA1', name: 'NVIDIA RTX 测试卡', memory_total: 16 * gib, memory_used: 2 * gib, utilization: 12 },
    { id: b, gpu_id: b, index: 1, device: 'CUDA0', name: 'NVIDIA RTX 测试卡', memory_total: 24 * gib, memory_used: 4 * gib, utilization: 76 },
  ]
  let profiles: LoadProfile[] = ['llama_cpp', 'vllm'].map((backend, i) => ({ id: backend, model_id: String(i), backend, runtime_id: backend,
    name: i ? 'HF 多卡' : 'GGUF 多卡', alias: backend, parameters: { context_length: 2048 }, generation: {}, extra_args: [], revision: 1 }))
  profiles.push({ ...profiles[0], id: 'strata', backend: 'strata', runtime_id: 'strata', alias: 'strata', name: 'Strata 已有选卡', parameters: { strata_config: { gpu: [1, 0] } } })
  let fields: Record<string, ParameterField[]> = {}, unresolved = false, samples = 0
  const writes: LoadProfile[] = [], errors: string[] = [], forbidden: string[] = []
  const application = await electron.launch({ args: [root], env: { ...process.env, MODLOADER_DATA_DIR: path.join(directory, 'data'), MODLOADER_SENSOR_AUTO_START: '0' }, timeout: 60_000 })
  const page = await application.firstWindow()
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/api/**', async route => {
    const request = route.request(), pathname = new URL(request.url()).pathname
    if (pathname === '/api/state') {
      samples++
      const response = await route.fetch(), state = await response.json()
      fields = Object.fromEntries(state.backends.map((item: { id: string; parameters: ParameterField[] }) => [item.id, item.parameters]))
      return route.fulfill({ response, json: { ...state, models: ['gguf', 'hf'].map((format, i) => ({ id: String(i), name: i ? '02 HF 模型' : '01 GGUF 模型', format,
        path: path.join(directory, format), size_bytes: 20 * gib, files: [], modalities: ['text'], metadata: {}, source: {}, managed: false, warnings: [] })), profiles,
        runtimes: ['llama_cpp', 'strata', 'vllm', 'sglang'].map(backend => ({ id: backend, backend, name: backend, path: directory, platform: ['llama_cpp', 'strata'].includes(backend) ? 'windows' : 'wsl2', version: 'fixture', metadata: { distribution: 'Fixture' }, managed: false })),
        hardware: { gpus: hardware, memory_total: 64 * gib, memory_used: 8 * gib, sampled_at: Date.now() / 1000 }, instances: [], downloads: [], installations: [], settings: { ...state.settings, font_scale: 1 } } })
    }
    if (pathname === '/api/profiles/preflight') {
      const body = request.postDataJSON()
      return route.fulfill({ json: { status: 'ready', parameters: fields[body.backend], message: '', errors: [], candidates: [], vision_enabled: false,
        acceleration: { mode: 'none', enabled: false, fields: fields[body.backend]?.filter(field => field.key.startsWith('speculative_')) ?? [], status: 'disabled' }, preparation_fields: [], mtp: { enabled: false, backend_supported: false, status: 'unsupported', candidates: [] } } })
    }
    if (pathname === '/api/resources/estimate') {
      const body = request.postDataJSON(), parallel = body.backend !== 'llama_cpp'
      return route.fulfill({ json: { source: 'multi-gpu-fixture', notes: [], resources: [
        ...(parallel ? [{ kind: 'vram', device: '多 GPU 合计', label: 'GPU 显存', aggregate: true, quality: 'partial', bytes: 20 * gib, notes: [],
          components: [{ label: '权重张量参考', bytes: unresolved ? null : 20 * gib }, { label: '单序列 KV 需求', bytes: unresolved ? null : 4 * gib, reference: true }] }] : []),
        ...[b, a].map((id, i) => ({ kind: 'vram', device: parallel ? 'CUDA（可见设备映射）' : `CUDA${i}`, logical_device: `CUDA${i}`, gpu_id: id, identity_status: unresolved ? 'unknown' : 'resolved', label: 'GPU 显存',
          bytes: parallel ? null : (i ? 8 : 10) * gib, quality: parallel ? 'dynamic' : 'estimate', notes: [],
          ...(parallel ? { budget_fraction: .5, budget_basis: body.backend === 'sglang' ? 'available' : 'total' } : {}) })),
      ] } })
    }
    if (request.method() === 'PATCH' && pathname.startsWith('/api/profiles/')) {
      const body = request.postDataJSON(), id = pathname.split('/').at(-1)!
      const saved = { ...body, id, revision: 2 }
      writes.push(saved); profiles = profiles.map(item => item.id === id ? saved : item)
      return route.fulfill({ json: saved })
    }
    if (request.method() !== 'GET') { forbidden.push(pathname); return route.abort() }
    return route.continue()
  })
  const dialog = page.getByRole('dialog', { name: '编辑加载配置', exact: true })
  const card = (name: string) => page.locator('.profile-card').filter({ has: page.getByRole('heading', { name, exact: true }) })
  const gpuRow = (scope: typeof dialog, index: number) => scope.locator(`.resource-preview-row[data-device="CUDA${index}"]`)
  const metric = (row: typeof dialog, label: string) => row.locator('dl > div').filter({ has: page.getByText(label, { exact: true }) }).locator('dd')
  const capture = async (name: string) => {
    await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))))
    const png = await application.evaluate(async ({ BrowserWindow }) => (await BrowserWindow.getAllWindows()[0].webContents.capturePage()).toPNG().toString('base64'))
    fs.writeFileSync(path.join(directory, name), Buffer.from(png, 'base64'))
  }
  try {
    await pinNavigation(page)
    await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(1280, 1000))
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await card('GGUF 多卡').getByRole('button', { name: '编辑配置', exact: true }).click()
    await dialog.locator('.profile-gpu > summary').click()
    await expect(dialog.locator('.gpu-device-choices button')).toHaveCount(2)
    await dialog.locator('.gpu-device-choices button').first().click()
    const deviceField = fields.llama_cpp.find(field => field.key === 'devices')!
    await expect(dialog.getByLabel(deviceField.label, { exact: true })).toHaveValue('CUDA1')
    await dialog.locator('.gpu-device-choices button').nth(1).click()
    await expect(dialog.getByLabel(deviceField.label, { exact: true })).toHaveValue('CUDA1,CUDA0')
    await dialog.getByLabel(fields.llama_cpp.find(field => field.key === 'split_mode')!.label, { exact: true }).selectOption('layer')
    await dialog.getByLabel(fields.llama_cpp.find(field => field.key === 'tensor_split')!.label, { exact: true }).fill('2,1')
    await dialog.getByLabel(fields.llama_cpp.find(field => field.key === 'main_gpu')!.label, { exact: true }).fill('1')
    await expect(metric(gpuRow(dialog, 0), '加载后剩余')).toHaveText('10 GB')
    await expect(metric(gpuRow(dialog, 1), '加载后剩余')).toHaveText('6 GB')
    await dialog.locator('.profile-gpu').evaluate(element => element.scrollIntoView({ block: 'start' })); await capture('llama-multi-gpu.png')
    await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(dialog).not.toBeVisible()
    expect(writes[0].parameters).toMatchObject({ devices: 'CUDA1,CUDA0', split_mode: 'layer', tensor_split: '2,1', main_gpu: 1 })
    await card('GGUF 多卡').getByRole('button', { name: '编辑配置', exact: true }).click()
    await expect(dialog.getByLabel(deviceField.label, { exact: true })).toHaveValue('CUDA1,CUDA0')
    await dialog.getByRole('button', { name: '取消', exact: true }).click()
    await card('Strata 已有选卡').getByRole('button', { name: '编辑配置', exact: true }).click()
    await dialog.locator('.profile-gpu > summary').click()
    const strataGpu = dialog.getByLabel(fields.strata.find(field => field.key === 'gpu')!.label, { exact: true })
    await expect(strataGpu).toHaveValue('')
    await expect(strataGpu).toHaveAttribute('placeholder', '继承 1,0')
    await strataGpu.fill('0'); await strataGpu.fill('')
    await expect(strataGpu).toHaveAttribute('placeholder', '继承 1,0')
    await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(dialog).not.toBeVisible()
    expect(writes[1].parameters).toEqual({ strata_config: { gpu: [1, 0] } })

    await page.locator('.model-item[data-model-id="1"]').click()
    await card('HF 多卡').getByRole('button', { name: '编辑配置', exact: true }).click()
    await dialog.locator('.profile-gpu > summary').click()
    await dialog.locator('.gpu-device-choices button').nth(1).click()
    await dialog.locator('.gpu-device-choices button').first().click()
    for (const key of ['tensor_parallel_size', 'pipeline_parallel_size']) await dialog.getByLabel(fields.vllm.find(field => field.key === key)!.label, { exact: true }).fill('2')
    await expect(dialog.getByLabel(fields.vllm.find(field => field.key === 'cuda_visible_devices')!.label, { exact: true })).toHaveValue(`${b},${a}`)
    await expect(metric(gpuRow(dialog, 0), '分配预算')).toHaveText('12 GB')
    await expect(gpuRow(dialog, 0).locator('.resource-preview-device')).toHaveText('GPU 1 · NVIDIA RTX 测试卡')
    await expect(gpuRow(dialog, 1).locator('.resource-preview-device')).toHaveText('GPU 0 · NVIDIA RTX 测试卡')
    await expect(metric(gpuRow(dialog, 1), '预算后剩余')).toHaveText('6 GB')
    await expect(dialog.locator('.resource-preview-row.aggregate')).toContainText('权重张量参考')
    await expect(dialog.locator('.resource-preview-row.aggregate [role="meter"]')).toHaveCount(0)
    await expect(dialog.locator('.resource-preview-row').first()).toHaveAttribute('data-gpu-id', b)
    hardware = [...hardware].reverse()
    const beforeReorder = samples
    await expect.poll(() => samples).toBeGreaterThan(beforeReorder)
    await expect(metric(gpuRow(dialog, 0), '分配预算')).toHaveText('12 GB')
    await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(dialog).not.toBeVisible()
    expect(writes[2].parameters).toMatchObject({ cuda_visible_devices: `${b},${a}`, tensor_parallel_size: 2, pipeline_parallel_size: 2 })
    await card('HF 多卡').getByRole('button', { name: '编辑配置', exact: true }).click()
    await dialog.getByLabel('运行后端', { exact: true }).selectOption('sglang')
    await expect(dialog.getByLabel(fields.sglang.find(field => field.key === 'pipeline_parallel_size')!.label, { exact: true })).toBeVisible()
    await dialog.getByLabel(fields.sglang.find(field => field.key === 'cuda_visible_devices')!.label, { exact: true }).fill(`${b},${a}`)
    await dialog.getByLabel(fields.sglang.find(field => field.key === 'tensor_parallel_size')!.label, { exact: true }).fill('2')
    await expect(metric(gpuRow(dialog, 0), '分配预算')).toHaveText('10 GB')
    await expect(metric(gpuRow(dialog, 1), '分配预算')).toHaveText('7 GB')
    await dialog.locator('.profile-gpu').evaluate(element => element.scrollIntoView({ block: 'start' })); await capture('sglang-multi-gpu.png')
    await dialog.locator('.resource-preview').scrollIntoViewIfNeeded(); await capture('per-gpu-budgets.png')
    unresolved = true
    await dialog.getByLabel(fields.sglang.find(field => field.key === 'tensor_parallel_size')!.label, { exact: true }).fill('3')
    await expect(gpuRow(dialog, 0)).toContainText('容量未读取')
    await expect(gpuRow(dialog, 0).getByRole('meter')).toHaveCount(0)
    await expect(dialog.locator('.resource-preview-row.aggregate')).toContainText('分项待估算')
    await dialog.getByRole('button', { name: '取消', exact: true }).click()
    await page.getByRole('navigation').getByRole('button', { name: '运行监控', exact: true }).click()
    await page.getByLabel('监控 GPU', { exact: true }).selectOption(b)
    await expect(page.locator('[data-resource="gpu"] strong')).toHaveText('76')
    hardware = [...hardware].reverse()
    await page.getByRole('button', { name: '刷新监控', exact: true }).click()
    await expect(page.getByLabel('监控 GPU', { exact: true })).toHaveValue(b)
    await expect(page.locator('[data-resource="gpu"] strong')).toHaveText('76')
    expect(errors).toEqual([]); expect(forbidden).toEqual([])
    console.log(`Multi-GPU screenshots: ${directory}`)
  } finally { await application.close() }
})
