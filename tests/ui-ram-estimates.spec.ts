import { test, expect, chromium } from '@playwright/test'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { pinNavigation } from './navigation'

test('加载预估保留状态和重试、移除诊断详情，并单列 Spark 快照池目标', async () => {
  test.setTimeout(90_000)
  const root = path.resolve(__dirname, '..')
  const task = process.env.MODLOADER_TEST_ARTIFACT_DIR || path.join(root, '.artifacts', 'codex', 'sparkinfer-load-estimate-20261028')
  fs.mkdirSync(task, { recursive: true })
  const directory = fs.mkdtempSync(path.join(task, 'ui-'))
  const { preview: servePreview } = await import('vite')
  const server = await servePreview({ root, configFile: false,
    build: { outDir: process.env.MODLOADER_TEST_DIST || path.join(root, 'dist') },
    preview: { host: '127.0.0.1', port: 0, strictPort: false } })
  const address = server.httpServer.address()
  if (!address || typeof address === 'string') throw new Error('Vite fixture did not open a local port')
  const browser = await chromium.launch({ channel: 'msedge', headless: true })
  const page = await browser.newPage({ viewport: { width: 1440, height: 1040 } })
  const gib = 1024 ** 3, errors: string[] = [], forbidden: string[] = []
  let mode: 'pending' | 'unknown' | 'error' = 'pending'
  const pending: (() => void)[] = []
  let requests = 0
  const fields = [{ key: 'context_length', label: '上下文长度', type: 'number', default: 4096, options: [] }]
  const unknownHost = [
    { label: '进程与运行时', bytes: null }, { label: '加载峰值与文件缓存', bytes: null },
  ]
  const backends = ['sglang', 'sparkinfer', 'strata']
  const state = {
    models: backends.map((backend, index) => ({ id: backend,
      name: ['SGLang · Qwen3.5-4B', 'SparkInfer · Qwen3.8-27B NVFP4', 'Strata · Qwen3.8-Flash-Next'][index],
      path: `/fixture/${backend}`, format: 'safetensors', files: [], size_bytes: (index ? 17 : 8) * gib,
      location: { platform: 'wsl2', distribution: 'Fixture' }, modalities: ['text'], metadata: {}, source: {}, managed: false, warnings: [] })),
    profiles: backends.map(backend => ({ id: backend, model_id: backend, backend, runtime_id: backend,
      name: '默认配置', alias: backend, parameters: {}, generation: {}, extra_args: [], revision: 1 })),
    runtimes: backends.map((backend, index) => ({ id: backend, backend, name: backend,
      path: '/fixture/never-launch', platform: 'wsl2', version: index ? '0.6.34' : '0.5.21', metadata: { distribution: 'Fixture' }, managed: false })),
    backends: backends.map(id => ({ id, label: id, parameters: fields })),
    hardware: { memory_total: 64 * gib, memory_used: 8 * gib,
      gpus: [{ id: 'GPU-fixture', gpu_id: 'GPU-fixture', device: 'CUDA0', name: 'NVIDIA Fixture', memory_total: 32 * gib, memory_used: 4 * gib }] },
    instances: [], downloads: [], installations: [], migrations: [], settings: {},
  }
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/api/**', async route => {
    const request = route.request(), pathname = new URL(request.url()).pathname
    if (pathname === '/api/state') return route.fulfill({ json: state })
    if (pathname === '/api/chat-presets') return route.fulfill({ json: { items: [], default_id: null } })
    if (pathname === '/api/conversations') return route.fulfill({ json: { items: [], active_id: null } })
    if (pathname === '/api/platforms/wsl') return route.fulfill({ json: { available: true, distributions: [{ name: 'Fixture', version: 2 }] } })
    if (pathname === '/api/profiles/preflight') return route.fulfill({ json: { status: 'ready', parameters: fields,
      errors: [], candidates: [], preparation_fields: [], acceleration: { enabled: false, mode: 'none', status: 'disabled', fields: [] },
      mtp: { enabled: false, backend_supported: false, status: 'unsupported', candidates: [] } } })
    if (pathname === '/api/resources/estimate') {
      requests++
      if (request.postDataJSON().backend === 'strata') return route.fulfill({ json: {
        source: 'strata-config', resources: [], notes: ['Strata 配置或配套文件不可用，无法估算'],
        diagnostic: 'Strata config fixture diagnostic',
      } })
      const spark = request.postDataJSON().backend === 'sparkinfer', phase = mode
      if (!spark && phase === 'pending') await new Promise<void>(resolve => pending.push(resolve))
      if (!spark && phase === 'error') return route.fulfill({ status: 400, json: { detail: 'RAM 预估接口失败 fixture' } })
      return route.fulfill({ json: { source: spark ? 'sparkinfer-metadata' : 'sglang-metadata', notes: [],
        ...(spark ? { diagnostic: 'Partial estimate fixture diagnostic' } : {}), resources: [
        { kind: 'vram', label: 'GPU 显存', device: 'CUDA0', gpu_id: 'GPU-fixture', identity_status: 'resolved',
          bytes: spark ? 15_253_267_456 : 8.5 * gib, quality: 'partial', notes: [] },
        { kind: 'ram', label: '系统内存', bytes: null, quality: spark ? 'dynamic' : 'partial', logical: false, notes: [],
          ...(spark ? { budget_bytes: 8_366_063_616, budget_component: '快照池预热' } : {}),
          components: spark ? [
            { label: '快照预热池（页锁定）', bytes: 4_183_031_808, reference: true }, { label: '快照预热池（可分页）', bytes: 4_183_031_808, reference: true },
            ...unknownHost, { label: '按需前缀缓存', bytes: null },
          ] : unknownHost },
      ] } }).catch(() => {})
    }
    if (request.method() !== 'GET') { forbidden.push(pathname); return route.abort() }
    return route.fulfill({ json: [] })
  })
  try {
    await page.goto(`http://127.0.0.1:${address.port}`)
    await pinNavigation(page)
    await expect(page.getByText('服务在线', { exact: true })).toBeVisible()
    await page.locator('.model-item[data-model-id="sglang"]').click()
    const card = page.locator('.profile-card'), preview = card.locator('.resource-preview')
    const ram = preview.locator('[data-resource="ram"]')
    await expect.poll(() => requests).toBeGreaterThan(0)
    await expect(preview).toHaveAttribute('aria-busy', 'true')
    await expect(ram).toContainText('待估算')
    await preview.screenshot({ path: path.join(directory, 'sglang-pending.png') })
    mode = 'unknown'; pending.splice(0).forEach(resolve => resolve())
    await expect(preview).toHaveAttribute('aria-busy', 'false')
    await expect(ram).toContainText('无法估算')
    await expect(ram).not.toContainText('待估算')
    await expect(ram.locator('.resource-preview-caveat')).toHaveCount(0)
    await expect(ram).not.toContainText('0 B')
    await preview.screenshot({ path: path.join(directory, 'sglang-complete-unknown.png') })

    mode = 'error'
    await card.getByRole('button', { name: '编辑配置', exact: true }).click()
    const dialog = page.getByRole('dialog', { name: '编辑加载配置', exact: true })
    const editPreview = dialog.locator('.resource-preview'), editRam = editPreview.locator('[data-resource="ram"]')
    await expect(editPreview).toContainText('资源估算失败')
    await expect(editRam).toContainText('估算失败')
    await expect(editRam).not.toContainText('待估算')
    await expect(editPreview.locator('details')).toHaveCount(0)
    await expect(editPreview).not.toContainText('诊断详情')
    await editPreview.screenshot({ path: path.join(directory, 'sglang-error.png') })
    mode = 'pending'
    await editPreview.getByRole('button', { name: '重试', exact: true }).click()
    await expect(editPreview).toHaveAttribute('aria-busy', 'true')
    await expect(editRam).toContainText('待估算')
    await expect.poll(() => pending.length).toBeGreaterThan(0)
    mode = 'unknown'; pending.splice(0).forEach(resolve => resolve())
    await expect(editPreview).toHaveAttribute('aria-busy', 'false')
    await expect(editRam).toContainText('无法估算')
    await expect(editPreview).not.toContainText('资源估算失败')
    await dialog.screenshot({ path: path.join(directory, 'sglang-editor-complete-unknown.png') })
    await dialog.getByRole('button', { name: '取消', exact: true }).click()

    await page.locator('.model-item[data-model-id="sparkinfer"]').click()
    await expect(preview).toHaveAttribute('aria-busy', 'false')
    await expect(ram).toContainText('快照池预热目标')
    await expect(ram).toContainText('7.8 GB')
    await expect(ram).toContainText('预计占用无法估算')
    await expect(ram).toContainText('加载后剩余无法估算')
    await expect(ram.locator('.resource-preview-caveat')).toHaveCount(0)
    await expect(ram).not.toContainText('至少占用')
    await expect(ram).not.toContainText('最多剩余')
    await expect(ram.getByRole('meter')).toHaveAttribute('aria-valuenow', '12.5')
    await expect(preview).toContainText('部分估算数据不可用')
    await expect(preview.locator('details')).toHaveCount(0)
    await expect(preview).not.toContainText('Partial estimate fixture diagnostic')
    await preview.screenshot({ path: path.join(directory, 'spark-snapshot-target.png') })

    await page.locator('.model-item[data-model-id="strata"]').click()
    await expect(preview).toHaveAttribute('aria-busy', 'false')
    await expect(preview.getByRole('alert')).toHaveText('Strata 配置或配套文件不可用，无法估算')
    await expect(preview.locator('.resource-preview-numbers dd')).toHaveText(['无法估算', '无法估算', '无法估算', '无法估算'])
    await expect(preview.locator('details')).toHaveCount(0)
    await expect(preview).not.toContainText('诊断详情')
    await expect(preview).not.toContainText('Strata config fixture diagnostic')
    await expect(card.getByRole('button', { name: '加载模型', exact: true })).toBeEnabled()
    await preview.screenshot({ path: path.join(directory, 'strata-unavailable.png') })
    expect(errors).toEqual([]); expect(forbidden).toEqual([])
    console.log(JSON.stringify({ screenshots: directory }))
  } catch (error) {
    console.log(JSON.stringify({ errors, body: await page.locator('body').innerText() }))
    await page.screenshot({ path: path.join(directory, 'failure.png') })
    throw error
  } finally {
    pending.splice(0).forEach(resolve => resolve())
    await browser.close()
    await new Promise<void>((resolve, reject) => server.httpServer.close(error => error ? reject(error) : resolve()))
  }
})
