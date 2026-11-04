import { pinNavigation } from './navigation'
import { test, expect, _electron as electron } from '@playwright/test'
import * as path from 'node:path'
import * as fs from 'node:fs'
import type { LoadProfile, ModelAsset } from '../src/types'
import { projectStrataMemory, type StrataResidentPlan } from '../src/strata-memory-preview'

const gib = 1024 ** 3
const ramPlan: StrataResidentPlan = {
  mode: 'resident', gpu_device: 'CUDA0', total_expert_bytes: 20 * gib, max_blob_bytes: gib,
  layer_blob_bytes: [gib], rank_runs: [[0, 20]], gpu_rank_count: 20, unranked_counts: [0],
  cache_slots: null, per_layer: false, experts_per_layer: 20,
  headroom_bytes: 4 * gib, budget_bytes: null, soft: true,
  host_bytes: gib, exchange_bytes: 0, prefill: null,
}

test('Strata RAM follows native cache, hotness, prefill loan and hard-budget rules', () => {
  const capacity = { ram_available: 50 * gib, commit_available: null, gpu_available: 9 * gib, gpu_fixed: 3 * gib, gpu_reserve: gib }
  expect(projectStrataMemory(ramPlan, capacity).bytes).toBe(16 * gib)
  const capped = projectStrataMemory(ramPlan, { ...capacity, commit_available: 10 * gib })
  expect(capped.bytes).toBe(5 * gib)
  expect(capped.limited).toBe(true)
  const hard = projectStrataMemory({ ...ramPlan, soft: false }, { ...capacity, commit_available: 10 * gib })
  expect(hard.bytes).toBe(16 * gib)
  expect(hard.shortfall).toBe(10 * gib)
  const uneven = { ...ramPlan, layer_blob_bytes: [3 * gib, gib], rank_runs: [[0, 1], [1, 4]] as [number, number][], gpu_rank_count: 5,
    total_expert_bytes: 7 * gib, max_blob_bytes: 3 * gib, host_bytes: 0, budget_bytes: 2 * gib }
  expect(projectStrataMemory(uneven, { ...capacity, gpu_available: 0 }).bytes).toBe(2 * gib)
  const mib = 1024 ** 2
  const loan: StrataResidentPlan = { ...ramPlan, total_expert_bytes: 512 * mib, max_blob_bytes: mib,
    layer_blob_bytes: [mib], rank_runs: [[0, 512]], gpu_rank_count: 512, experts_per_layer: 512,
    headroom_bytes: 0, host_bytes: 2 * mib, exchange_bytes: 16 * mib,
    prefill: { auto: true, borrow: true, lend_percent: 85, ring_slots: 96, candidates: [[1024, 64 * mib, 4 * mib]], fallback: [1024, 80 * mib, 3 * mib] } }
  const loanCapacity = { ...capacity, gpu_available: 256 * mib, gpu_fixed: 0, gpu_reserve: 0 }
  expect(projectStrataMemory(loan, loanCapacity).bytes).toBe(342 * mib)
  expect(projectStrataMemory({ ...loan, mode: 'arena' }, loanCapacity).bytes).toBe(518 * mib)
  expect(projectStrataMemory({ ...loan, mode: 'mmap' }, loanCapacity).bytes).toBe(6 * mib)
  const layered = { ...loan, per_layer: true, total_expert_bytes: 768 * mib, max_blob_bytes: 2 * mib,
    layer_blob_bytes: [mib, 2 * mib], rank_runs: [[1, 256], [0, 256]] as [number, number][], experts_per_layer: 256,
    prefill: { ...loan.prefill!, candidates: [[1024, 200 * mib, 4 * mib]] as [number, number, number][] } }
  expect(projectStrataMemory(layered, { ...loanCapacity, gpu_available: 384 * mib }).bytes).toBe(606 * mib)
  expect(projectStrataMemory(ramPlan, { ...capacity, gpu_available: null }).quality).toBe('partial')
})

test('真实 Electron 加载资源预估：随配置与硬件刷新，容量状态和未知提示准确', async () => {
  test.setTimeout(120_000)
  const root = path.resolve(__dirname, '..')
  const directory = fs.mkdtempSync(path.join(root, 'work', 'ui-resources-'))
  const screenshots = path.join(directory, 'screenshots')
  fs.mkdirSync(screenshots)
  const gib = 1024 ** 3
  const model: ModelAsset = {
    id: 'resource-preview-model', name: '资源预估 UI fixture', path: path.join(directory, 'fixture-only.gguf'), format: 'gguf',
    size_bytes: 8 * gib, architecture: 'fixture', quantization: 'Q4_K_M', files: [], modalities: ['text'], metadata: {}, source: {}, managed: false, warnings: [],
  }
  const profile: LoadProfile = {
    id: 'resource-preview-profile', model_id: model.id, name: '资源预估 fixture 配置', backend: 'llama_cpp', runtime_id: 'resource-preview-runtime',
    alias: 'resource-preview-fixture', parameters: { context_length: 2048 }, generation: {}, extra_args: [], revision: 1,
  }
  let fontScale = 1
  let ramTotal = 16 * gib
  let ramUsed = 4 * gib
  let gpuTotal = 12 * gib, gpuUsed = 3 * gib
  let commitAvailable: number | null = null
  let stateSamples = 0
  const estimateRequests: Array<{ context_length?: number }> = []
  const estimateOccurrences = new Map<number, number>()
  const externalRequests: string[] = []
  const unexpectedMutations: string[] = []
  const errors: string[] = []
  let delayedEditorEstimateStarted = false
  let releaseDelayedEstimate: (() => void) | undefined
  const delayedEstimate = new Promise<void>(resolve => { releaseDelayedEstimate = resolve })

  function forecast(contextLength: number | undefined) {
    if (contextLength === 20480) return {
      source: 'llama-fit-real-log-fixture', notes: [], resources: [
        { kind: 'vram', device: 'CUDA0', label: 'GPU 显存', bytes: 16.9 * gib, quality: 'estimate', notes: [] },
        { kind: 'ram', label: '系统内存', bytes: 52 * 1024 ** 2, model_reference_bytes: 994 * 1024 ** 2,
          quality: 'partial', logical: false, notes: [], components: [
            { label: '上下文缓冲', bytes: 0 }, { label: '计算缓冲', bytes: 52 * 1024 ** 2 },
            { label: '模型缓冲参考', bytes: 994 * 1024 ** 2, reference: true },
          ] },
      ],
    }
    if ([6144, 6145, 6146, 6147, 6148, 6149].includes(contextLength ?? 0)) return {
      source: 'zero-allocation-fixture', notes: [], resources: [
        { kind: 'vram', device: 'CUDA0', label: 'GPU 显存', bytes: contextLength === 6145 ? 2 * gib : 0, quality: 'estimate', notes: [] },
        { kind: 'ram', label: '系统内存', bytes: contextLength === 6144 ? 4 * gib : contextLength === 6145 ? 994 * 1024 ** 2 : contextLength === 6146 ? null : contextLength === 6148 ? 1 : 0,
          quality: contextLength === 6145 || contextLength === 6149 ? 'partial' : contextLength === 6146 ? 'unknown' : 'estimate',
          logical: contextLength === 6145, notes: [],
          components: contextLength === 6149 ? [{ label: '未估算缓冲', bytes: null }] : [] },
      ],
    }
    if ([12288, 24576, 49152].includes(contextLength ?? 0)) return {
      source: 'strata-metadata-fixture', notes: [], resources: [
        { kind: 'vram', device: 'CUDA0', label: 'GPU 显存', bytes: null, quality: 'dynamic',
          fixed_bytes: (contextLength === 49152 ? 11 : 3) * gib,
          ...(contextLength === 24576 ? { budget_bytes: 5 * gib } : { reserve_bytes: gib }), notes: [] },
        { kind: 'ram', label: '系统内存', bytes: null, quality: 'dynamic', resident_plan: ramPlan, notes: [] },
      ],
    }
    if (contextLength === 256000) return {
      source: 'ninfer-user-screenshot-fixture', notes: ['按 NInfer v3 张量布局估算。'], resources: [
        { kind: 'vram', device: 'CUDA0', label: 'GPU 显存', bytes: 24.4 * gib, quality: 'partial', logical: false, notes: [], components: [{ label: '工作区 / CUDA Graph 等', bytes: null }] },
        { kind: 'ram', label: '系统内存', bytes: 12.2 * 1024 ** 2, quality: 'partial', logical: false, notes: [], components: [{ label: '进程 / 上下文缓存 / 加载临时峰值', bytes: null }] },
      ],
    }
    if (contextLength === 16384 || contextLength === 32768 || contextLength === 65536) return {
      source: 'wsl-metadata-fixture', notes: ['仅读取模型头；工作区与峰值未估算。'], resources: [
        { kind: 'vram' as const, device: 'CUDA0', label: 'GPU 显存', bytes: contextLength === 65536 ? null : (contextLength === 16384 ? 7 : 5) * gib,
          quality: 'partial' as const, budget_fraction: contextLength === 16384 ? .85 : .5,
          budget_basis: contextLength === 16384 ? 'total' : 'available', notes: [], components: [
            { label: 'GPU 权重', bytes: contextLength === 65536 ? null : 5 * gib },
            { label: contextLength === 16384 ? 'KV 共享池' : '单序列 KV 需求', bytes: (contextLength === 16384 ? 2 : 4) * gib, reference: contextLength !== 16384 },
            { label: '工作区 / 实际 KV 扩容', bytes: null },
          ] },
        { kind: 'ram' as const, label: '系统内存', bytes: 2 * gib, quality: 'partial' as const, logical: false,
          notes: ['Host 张量是实际分配，不是文件映射。'], components: [
            { label: 'Host 常驻权重', bytes: 2 * gib }, { label: '进程 / 缓存', bytes: null },
          ] },
      ],
    }
    if (contextLength === 8192) return {
      source: 'fixture', notes: [], resources: [
        { kind: 'ram' as const, label: '普通档 RAM', bytes: 20 * gib, quality: 'estimate' as const, notes: [] },
        { kind: 'ram' as const, label: '偏高档 RAM', bytes: 30 * gib, quality: 'estimate' as const, notes: [] },
        { kind: 'ram' as const, label: '高占用 RAM', bytes: 40 * gib, quality: 'estimate' as const, notes: [] },
        { kind: 'ram' as const, label: '超容量档 RAM', bytes: 60 * gib, quality: 'estimate' as const, notes: [] },
        { kind: 'ram' as const, label: '部分估算 RAM', bytes: 10 * gib, quality: 'partial' as const, notes: [] },
        { kind: 'ram' as const, label: '未知 RAM', bytes: null, quality: 'unknown' as const, notes: [] },
        { kind: 'ram' as const, label: '动态 RAM', bytes: null, quality: 'dynamic' as const, budget_fraction: 0.25, reserve_bytes: gib, notes: [] },
      ],
    }
    return {
      source: 'fixture', notes: [], resources: [
        { kind: 'ram' as const, label: '系统内存', bytes: 4 * gib, quality: 'estimate' as const, notes: [] },
        { kind: 'vram' as const, device: 'CUDA0', label: 'GPU 显存', bytes: 2 * gib, quality: 'estimate' as const, notes: [] },
      ],
    }
  }

  const application = await electron.launch({
    ...(process.env.MODLOADER_TEST_EXECUTABLE ? { executablePath: process.env.MODLOADER_TEST_EXECUTABLE, args: [] } : { args: [root] }),
    env: { ...process.env, MODLOADER_DATA_DIR: path.join(directory, 'data') }, timeout: 60_000,
  })
  const page = await application.firstWindow()
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/*', async route => {
    const request = route.request()
    const url = new URL(request.url())
    if (['http:', 'https:'].includes(url.protocol) && !['127.0.0.1', 'localhost'].includes(url.hostname)) {
      externalRequests.push(url.origin)
      await route.abort()
    } else if (url.pathname.startsWith('/api/') && !['GET', 'HEAD', 'OPTIONS'].includes(request.method()) && !(request.method() === 'POST' && url.pathname === '/api/resources/estimate')) {
      unexpectedMutations.push(`${request.method()} ${url.pathname}`)
      await route.abort()
    } else await route.continue()
  })
  await page.route('**/api/state', async route => {
    const response = await route.fetch()
    const state = await response.json()
    stateSamples++
    const backends = [{ id: profile.backend, label: '界面样例引擎', parameters: [
      { key: 'context_length', label: '上下文长度', type: 'number', default: 4096, options: [], description: '', advanced: false },
    ] }]
    await route.fulfill({ response, json: {
      ...state,
      models: [model], profiles: [profile],
      runtimes: [{ id: 'resource-preview-runtime', backend: profile.backend, name: '界面样例运行环境', version: 'fixture', path: directory, managed: false, metadata: {} }],
      backends, instances: [], downloads: [], installations: [],
      hardware: { memory_total: ramTotal, memory_available: ramTotal - ramUsed, memory_used: ramUsed, memory_commit_available: commitAvailable, gpus: [{ device: 'CUDA0', memory_total: gpuTotal, memory_used: gpuUsed }] },
      settings: { ...state.settings, font_scale: fontScale },
    } })
  })
  await page.route('**/api/profiles/preflight', route => route.fulfill({ json: { status: 'ready', candidates: [], errors: [], message: '', mtp: { enabled: false, status: 'unsupported', backend_supported: false, message: '此 fixture 不提供 MTP', candidates: [] } } }))
  await page.route('**/api/resources/estimate', async route => {
    const body = route.request().postDataJSON() as { parameters?: { context_length?: number } }
    const contextLength = body.parameters?.context_length
    estimateRequests.push({ context_length: contextLength })
    const occurrence = (estimateOccurrences.get(contextLength ?? 0) || 0) + 1
    estimateOccurrences.set(contextLength ?? 0, occurrence)
    if (contextLength === 2048 && occurrence === 2) {
      delayedEditorEstimateStarted = true
      await delayedEstimate
      try { await route.fulfill({ json: forecast(contextLength) }) } catch { /* The renderer may abort this obsolete draft request. */ }
      return
    }
    await route.fulfill({ json: forecast(contextLength) })
  })

  const profileCard = page.locator('.profile-card').filter({ hasText: profile.name })
  const cardPreview = profileCard.locator('.resource-preview')
  const dialog = page.getByRole('dialog', { name: '编辑加载配置', exact: true })
  const editorPreview = dialog.locator('.resource-preview')
  const estimateRow = (preview: typeof cardPreview, label: string) => preview.locator('.resource-preview-row').filter({ hasText: label })
  const metric = (row: typeof cardPreview, label: string) => row.locator('dl > div').filter({ has: page.getByText(label, { exact: true }) }).locator('dd')
  async function nativeScreenshot(name: string) {
    await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))))
    const png = await application.evaluate(async ({ BrowserWindow }) => (await BrowserWindow.getAllWindows()[0].webContents.capturePage()).toPNG().toString('base64'))
    fs.writeFileSync(path.join(screenshots, name), Buffer.from(png, 'base64'))
  }

  try {
    await pinNavigation(page)
    await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
    await application.evaluate(({ BrowserWindow }) => { const window = BrowserWindow.getAllWindows()[0]; window.setSize(1440, 960); window.webContents.setZoomFactor(1) })
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await expect(page.getByRole('heading', { name: model.name, exact: true })).toBeVisible()
    await page.evaluate(() => document.fonts.ready)

    await test.step('卡片只显示配置占用与余量，硬件刷新不重复请求估算', async () => {
      const ram = estimateRow(cardPreview, '系统内存')
      const vram = estimateRow(cardPreview, 'GPU 显存')
      await expect(metric(ram, '预计占用')).toHaveText('4 GB')
      await expect(metric(ram, '加载后剩余')).toHaveText('8 GB')
      await expect(metric(vram, '预计占用')).toHaveText('2 GB')
      await expect(metric(vram, '加载后剩余')).toHaveText('7 GB')
      await expect(ram.getByRole('meter')).toHaveAttribute('aria-valuenow', '50')
      const ramBar = ram.locator('.resource-forecast-bar')
      await expect.poll(async () => ramBar.locator('.forecast-used').evaluate(element => element.getBoundingClientRect().width / element.parentElement!.getBoundingClientRect().width)).toBeCloseTo(.25, 2)
      await expect.poll(async () => ramBar.locator('.forecast-addition').evaluate(element => element.getBoundingClientRect().width / element.parentElement!.getBoundingClientRect().width)).toBeCloseTo(.25, 2)
      await expect(cardPreview).not.toContainText('当前已用')
      await expect(cardPreview).not.toContainText('总量')
      await expect(cardPreview.locator('details')).toHaveCount(0)
      const requestCount = estimateRequests.length
      const beforeSamples = stateSamples
      ramUsed = 5 * gib
      await expect.poll(() => stateSamples, { timeout: 4_000 }).toBeGreaterThan(beforeSamples)
      await expect(metric(ram, '加载后剩余')).toHaveText('7 GB')
      await expect(metric(ram, '预计占用')).toHaveText('4 GB')
      await expect(ram.getByRole('meter')).toHaveAttribute('aria-valuenow', '56.25')
      await expect.poll(() => stateSamples, { timeout: 4_000 }).toBeGreaterThan(beforeSamples + 1)
      expect(estimateRequests).toHaveLength(requestCount)
      await nativeScreenshot('resource-preview-compact.png')
    })

    await test.step('编辑上下文长度 450 ms 去抖，旧慢响应不能覆盖新结果', async () => {
      ramTotal = 100 * gib
      ramUsed = 50 * gib
      const beforeSamples = stateSamples
      await expect.poll(() => stateSamples, { timeout: 4_000 }).toBeGreaterThan(beforeSamples)
      await profileCard.getByRole('button', { name: '编辑配置', exact: true }).click()
      await expect(dialog).toBeVisible()
      await expect.poll(() => delayedEditorEstimateStarted, { timeout: 4_000 }).toBe(true)
      const callsBeforeEdit = estimateRequests.length
      const contextLength = dialog.getByLabel('上下文长度', { exact: true })
      await expect(contextLength).toBeVisible()
      await contextLength.fill('8192')
      await page.waitForTimeout(300)
      expect(estimateRequests).toHaveLength(callsBeforeEdit)
      await expect.poll(() => estimateRequests.length, { timeout: 5_000 }).toBe(callsBeforeEdit + 1)
      await expect(editorPreview.locator('.resource-preview-row')).toHaveCount(6)

      const normal = estimateRow(editorPreview, '普通档 RAM')
      const warning = estimateRow(editorPreview, '偏高档 RAM')
      const danger = estimateRow(editorPreview, '高占用 RAM')
      const exceeded = estimateRow(editorPreview, '超容量档 RAM')
      await expect(metric(normal, '加载后剩余')).toHaveText('30 GB')
      await expect(metric(warning, '加载后剩余')).toHaveText('20 GB')
      await expect(metric(danger, '加载后剩余')).toHaveText('10 GB')
      await expect(editorPreview).not.toContainText('危险')
      await expect(metric(exceeded, '预计超出')).toHaveText('10 GB')
      await expect(exceeded).toHaveClass(/exceeded/)
      await expect(metric(exceeded, '预计超出')).toHaveCSS('color', 'rgb(197, 70, 88)')
      await expect(exceeded.getByRole('meter')).toHaveAttribute('aria-valuenow', '100')
      await expect(exceeded.locator('.forecast-addition')).toHaveCSS('background-color', 'rgb(197, 70, 88)')
      const partial = estimateRow(editorPreview, '部分估算 RAM')
      await expect(partial).toHaveCount(0)
      const unknown = estimateRow(editorPreview, '未知 RAM')
      const dynamic = estimateRow(editorPreview, '动态 RAM')
      await expect(metric(unknown, '预计占用')).toHaveText('待估算')
      await expect(metric(unknown, '加载后剩余')).toHaveText('待估算')
      await expect(metric(dynamic, '分配预算')).toHaveText('25 GB')
      await expect(metric(dynamic, '预算后剩余')).toHaveText('25 GB')
      await expect(dynamic).not.toContainText('预计占用')
      await expect(partial.locator('.forecast-addition')).toHaveCount(0)
      await expect(unknown.locator('.forecast-addition')).toHaveCount(0)
      await expect(dynamic.locator('.forecast-addition.budget')).toHaveCount(1)
      await expect(dynamic.getByRole('meter')).toHaveAttribute('aria-valuenow', '50')
      await expect(dynamic.getByRole('meter')).toHaveAttribute('aria-valuetext', /分配预算 25 GB.*预算后剩余 25 GB/)
      expect(estimateRequests.at(-1)?.context_length).toBe(8192)

      releaseDelayedEstimate?.()
      await expect(editorPreview.locator('.resource-preview-row')).toHaveCount(6)
      await expect(metric(exceeded, '预计超出')).toHaveText('10 GB')
    })

    await test.step('Strata 固定占用与自动预算分段，目标预留不能掩盖固定超额', async () => {
      const contextLength = dialog.getByLabel('上下文长度', { exact: true })
      const gpu = estimateRow(editorPreview, 'GPU 显存')
      await contextLength.fill('12288')
      await expect(metric(gpu, '分配预算')).toHaveText('8 GB')
      await expect(metric(gpu, '预算后剩余')).toHaveText('1 GB')
      await expect.poll(async () => gpu.locator('.forecast-addition:not(.budget)').evaluate(element => element.getBoundingClientRect().width / element.parentElement!.getBoundingClientRect().width)).toBeCloseTo(.25, 2)
      await expect.poll(async () => gpu.locator('.forecast-addition.budget').evaluate(element => element.getBoundingClientRect().width / element.parentElement!.getBoundingClientRect().width)).toBeCloseTo(5 / 12, 2)
      await expect(gpu.getByRole('meter')).toHaveAttribute('aria-valuenow', '50')
      await expect(gpu.getByRole('meter')).toHaveAttribute('aria-valuetext', /已估固定占用 3 GB/)
      const ram = estimateRow(editorPreview, '系统内存')
      await expect(metric(ram, '预计占用')).toHaveText('16 GB')
      await expect(metric(ram, '加载后剩余')).toHaveText('34 GB')
      const requests = estimateRequests.length
      gpuUsed = 4 * gib
      await expect(metric(ram, '预计占用')).toHaveText('17 GB')
      await expect(metric(ram, '加载后剩余')).toHaveText('33 GB')
      commitAvailable = 10 * gib
      await expect(metric(ram, '预计占用')).toHaveText('5 GB')
      await expect(editorPreview).toContainText('RAM 驻留未覆盖全部专家')
      expect(estimateRequests.length).toBe(requests)
      gpuUsed = 3 * gib; commitAvailable = null
      await expect(metric(ram, '预计占用')).toHaveText('16 GB')
      await editorPreview.screenshot({ path: path.join(screenshots, 'resource-preview-strata-auto.png') })
      await contextLength.fill('24576')
      await expect(metric(gpu, '分配预算')).toHaveText('5 GB')
      await expect(metric(gpu, '预算后剩余')).toHaveText('4 GB')
      await contextLength.fill('49152')
      await expect(metric(gpu, '至少占用')).toHaveText('11 GB')
      await expect(metric(gpu, '至少超出')).toHaveText('2 GB')
      await expect(gpu.locator('.budget')).toHaveCount(0)
      await expect(gpu).toHaveClass(/exceeded/)
    })

    await test.step('部分估算明确上下界，不把动态预算或参考 KV 当作占用', async () => {
      const contextLength = dialog.getByLabel('上下文长度', { exact: true })
      await contextLength.fill('16384')
      const gpu = estimateRow(editorPreview, 'GPU 显存')
      const ram = estimateRow(editorPreview, '系统内存')
      await expect(metric(gpu, '至少占用')).toHaveText('7 GB')
      await expect(metric(gpu, '最多剩余')).toHaveText('2 GB')
      await expect(gpu.locator('.forecast-addition.budget')).toHaveCount(0)
      await expect.poll(async () => gpu.locator('.forecast-addition').evaluate(element => element.getBoundingClientRect().width / element.parentElement!.getBoundingClientRect().width)).toBeCloseTo(7 / 12, 2)
      await expect(metric(ram, '至少占用')).toHaveText('2 GB')
      await expect(metric(ram, '最多剩余')).toHaveText('48 GB')
      await expect(editorPreview).not.toContainText('仅读取模型头')
      await expect(gpu).not.toContainText('10.2 GB')
      await contextLength.fill('32768')
      await expect(metric(gpu, '至少占用')).toHaveText('5 GB')
      await expect(metric(gpu, '最多剩余')).toHaveText('4 GB')
      await expect(gpu).not.toContainText('单序列 KV 需求')
      await expect(gpu).not.toContainText('4.5 GB')
      await nativeScreenshot('resource-preview-wsl-partial.png')
      await contextLength.fill('65536')
      await expect(metric(gpu, '预计占用')).toHaveText('待估算')
      await expect(metric(gpu, '加载后剩余')).toHaveText('待估算')
      await expect(gpu.locator('.forecast-addition')).toHaveCount(0)
    })

    await test.step('零分配隐藏对应资源栏，未知和非零小值保留', async () => {
      const contextLength = dialog.getByLabel('上下文长度', { exact: true })
      gpuTotal = 31.8 * gib; gpuUsed = 4.5 * gib
      await contextLength.fill('20480')
      const host = estimateRow(editorPreview, '系统内存')
      await expect(metric(host, '至少占用')).toHaveText('52 MB')
      await expect(host).not.toContainText('待估算')
      await expect(host).not.toContainText('994 MB')
      await expect(metric(estimateRow(editorPreview, 'GPU 显存'), '加载后剩余')).toHaveText('10.4 GB')
      await editorPreview.screenshot({ path: path.join(screenshots, 'resource-preview-llama-host.png') })
      gpuTotal = 12 * gib; gpuUsed = 3 * gib
      await contextLength.fill('6144')
      await expect(metric(estimateRow(editorPreview, '系统内存'), '预计占用')).toHaveText('4 GB')
      await expect(editorPreview.locator('[data-resource="vram"]')).toHaveCount(0)
      await contextLength.fill('6145')
      await expect(metric(estimateRow(editorPreview, 'GPU 显存'), '预计占用')).toHaveText('2 GB')
      await expect(editorPreview.locator('[data-resource="ram"]')).toHaveCount(0)
      const singleRow = editorPreview.locator('.resource-preview-row')
      await expect(singleRow).toHaveCount(1)
      const widths = await singleRow.evaluate(element => [element.getBoundingClientRect().width, element.parentElement!.getBoundingClientRect().width])
      expect(Math.abs(widths[0] - widths[1])).toBeLessThan(1)
      await editorPreview.screenshot({ path: path.join(screenshots, 'resource-preview-zero-hidden.png') })
      await contextLength.fill('6146')
      await expect(metric(estimateRow(editorPreview, '系统内存'), '预计占用')).toHaveText('待估算')
      await expect(editorPreview.locator('[data-resource="vram"]')).toHaveCount(0)
      await contextLength.fill('6147')
      await expect(editorPreview).toHaveCount(0)
      await contextLength.fill('6148')
      await expect(metric(estimateRow(editorPreview, '系统内存'), '预计占用')).toHaveText('1 B')
      await contextLength.fill('6149')
      await expect(editorPreview.locator('[data-resource="ram"]')).toHaveAttribute('data-quality', 'partial')
      await expect(editorPreview.locator('[data-resource="ram"]')).toHaveCount(1)
      await expect(editorPreview.locator('[data-resource="vram"]')).toHaveCount(0)
    })

    await test.step('窄窗口下字体 1.3、缩放 1.2 可读且无横向溢出', async () => {
      await dialog.getByRole('button', { name: '取消', exact: true }).click()
      await expect(dialog).toBeHidden()
      fontScale = 1.3
      profile.parameters.context_length = 32768
      await application.evaluate(({ BrowserWindow }) => { const window = BrowserWindow.getAllWindows()[0]; window.setSize(940, 760); window.webContents.setZoomFactor(1.2) })
      await page.getByRole('button', { name: '刷新状态', exact: true }).click()
      await expect.poll(() => page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue('--font-scale').trim())).toBe('1.3')
      await expect(cardPreview.locator('.resource-preview-row')).toHaveCount(2)
      await expect(metric(estimateRow(cardPreview, 'GPU 显存'), '至少占用')).toHaveText('5 GB')
      await cardPreview.scrollIntoViewIfNeeded()
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBe(true)
      await nativeScreenshot('resource-preview-narrow-font1.3-zoom1.2.png')
    })

    await test.step('用户截图中的 NInfer 配置只保留占用与剩余上下界', async () => {
      fontScale = 1.3
      ramTotal = 61.4 * gib; ramUsed = 26.2 * gib
      gpuTotal = 31.8 * gib; gpuUsed = 4.5 * gib
      model.name = 'NInfer · Qwen3.8-27B NVFP4'
      model.format = 'ninfer'; model.size_bytes = 19.4 * gib
      model.quantization = 'bf16 / fp32 / fp8_e4m3fn_row_bf16 / nvfp4 / q8_g32_fp16'
      model.metadata.ninfer_tensor_formats = model.quantization.split(' / ')
      profile.backend = 'ninfer'; profile.parameters.context_length = 256000
      await application.evaluate(({ BrowserWindow }) => { const window = BrowserWindow.getAllWindows()[0]; window.setSize(1440, 960); window.webContents.setZoomFactor(1) })
      await page.getByRole('button', { name: '刷新状态', exact: true }).click()
      const gpu = estimateRow(cardPreview, 'GPU 显存')
      await expect(metric(gpu, '至少占用')).toHaveText('24.4 GB')
      await expect(metric(gpu, '最多剩余')).toHaveText('2.9 GB')
      await expect(cardPreview).not.toContainText('当前已用')
      await expect(cardPreview).not.toContainText('估算依据')
      await profileCard.scrollIntoViewIfNeeded()
      await nativeScreenshot('resource-preview-ninfer.png')
      await cardPreview.screenshot({ path: path.join(screenshots, 'resource-preview-ninfer-detail.png') })
    })

    expect(externalRequests).toEqual([])
    expect(unexpectedMutations).toEqual([])
    expect(errors).toEqual([])
    console.log(JSON.stringify({ screenshots, estimateRequests, stateSamples, electron: await application.evaluate(() => process.versions.electron) }))
  } finally {
    releaseDelayedEstimate?.()
    await application.close()
  }
})
