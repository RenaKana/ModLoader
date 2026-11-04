import { test, expect, _electron as electron, type Locator } from '@playwright/test'
import * as path from 'node:path'
import * as fs from 'node:fs'
import type { ModelAsset } from '../src/types'

test('真实 Electron 模型列表：排序、组合搜索、保留选择及大量模型独立滚动', async () => {
  test.setTimeout(120_000)
  const root = path.resolve(__dirname, '..')
  const artifacts = process.env.MODLOADER_TEST_ARTIFACT_DIR || path.join(root, 'work')
  fs.mkdirSync(artifacts, { recursive: true })
  const directory = fs.mkdtempSync(path.join(artifacts, 'ui-model-list-'))
  const screenshots = path.join(directory, 'screenshots')
  fs.mkdirSync(screenshots)
  const gib = 1024 ** 3
  const asset = (id: string, name: string, format: string, size: number, vendor: string, architecture: string, quantization: string): ModelAsset => ({
    id, name, format, size_bytes: size, architecture, quantization,
    path: path.join(directory, 'fixture-only', vendor, `${id}.${format.toLowerCase() === 'gguf' ? 'gguf' : 'safetensors'}`),
    files: [], modalities: ['text'], metadata: {}, source: {}, managed: false, warnings: [],
  })
  // Deliberately register the numerical names and equal-size models out of order.
  const sortingModels = [
    asset('sort-qwen10', '排序样例 Qwen10', 'gguf', 10 * gib, 'vendor-b', 'qwen2', 'Q8_0'),
    asset('sort-llama3', '排序样例 Llama3', 'hf', 5 * gib, 'vendor-d', 'llama', 'BF16'),
    asset('sort-beta4', '排序样例 Beta4', 'GGUF', 2 * gib, 'vendor-e', 'llama', 'Q5_K_M'),
    asset('sort-qwen2', '排序样例 Qwen2', 'gguf', 5 * gib, 'vendor-a', 'qwen2', 'Q4_K_M'),
    asset('sort-alpha1', '排序样例 Alpha1', 'safetensors', 0, 'vendor-c', 'mistral', 'FP16'),
  ]
  const longName = '长名称界面样例：这是用于确认中文与 Latin model names 混排后只占两行的模型名称，完整名称仍可查看，并且不会挤掉格式、文件大小、量化信息或路径'
  const longModel = asset('list-long-last', longName, 'gguf', 12 * gib, '厂商/系列/版本/量化/很长的模型目录/只作为界面样例而不创建实际模型文件', 'ui_fixture', 'Q4_K_M')
  const legacyWarnings = [
    '发现视觉投影文件；在加载配置中选择与模型匹配的文件后启用图片输入',
    '已检查 NInfer v3 容器、文件长度及续卷身份；完整模型组件与推理兼容性仍需官方检查工具和引擎确认。',
  ]
  longModel.warnings = [...legacyWarnings, '缺少权重分片：model-00002-of-00002.gguf']
  const ninferModels = [
    asset('ninfer-int', 'NInfer · Qwen3.6-27B', 'ninfer', 16.3 * gib, 'NInfer', 'qwen3_5', 'bf16 / fp32 / int32 / q4_g64_fp16 / q5_g64_fp16 / q6_g64_fp16 / q8_g32_fp16'),
    asset('ninfer-nvfp4', 'NInfer · Qwen3.8-27B NVFP4', 'ninfer', 19.4 * gib, 'NInfer', 'qwen3_5', 'bf16 / fp32 / fp8_e4m3fn_row_bf16 / nvfp4 / q8_g32_fp16'),
  ]
  ninferModels[1].metadata.ninfer_tensor_formats = ninferModels[1].quantization!.split(' / ')
  const models = [
    ...sortingModels,
    ...ninferModels,
    ...Array.from({ length: 120 }, (_, index) => asset(`list-model-${index + 1}`, `界面样例 Model ${index + 1}`, index % 3 ? 'gguf' : 'hf', (index + 1) * gib, '批量界面样例', 'ui_fixture', index % 2 ? 'Q8_0' : 'Q4_K_M')),
    longModel,
  ]
  const instances = [
    { id: 'list-ready', model_id: 'sort-qwen2', state: 'ready' },
    { id: 'list-loading', model_id: 'sort-llama3', state: 'loading' },
    { id: 'list-failed', model_id: 'sort-qwen10', state: 'failed' },
    { id: 'list-stopped', model_id: 'sort-alpha1', state: 'stopped' },
  ].map(instance => ({ ...instance, alias: instance.id, backend: 'llama_cpp', profile_id: `${instance.id}-profile`, profile_revision: 1, runtime_id: 'fixture-only-runtime', port: 0, started_at: '', active_requests: 0, capabilities: {} }))
  let fontScale = 1
  const application = await electron.launch({
    ...(process.env.MODLOADER_TEST_EXECUTABLE || process.env.MODLOADER_TEST_PACKAGED ? { executablePath: process.env.MODLOADER_TEST_EXECUTABLE || path.join(root, 'release', 'win-unpacked', 'ModLoader.exe'), args: [] } : { args: [root] }),
    env: { ...process.env, MODLOADER_DATA_DIR: path.join(directory, 'data') }, timeout: 60_000,
  })
  const page = await application.firstWindow()
  const errors: string[] = []
  const externalRequests: string[] = []
  const mutationRequests: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/*', async route => {
    const request = route.request()
    const url = new URL(request.url())
    if (['http:', 'https:'].includes(url.protocol) && !['127.0.0.1', 'localhost'].includes(url.hostname)) {
      externalRequests.push(url.origin)
      await route.abort()
    } else if (url.pathname.startsWith('/api/') && !url.pathname.startsWith('/api/conversations') && !['GET', 'HEAD', 'OPTIONS'].includes(request.method())) {
      mutationRequests.push(`${request.method()} ${url.pathname}`)
      await route.abort()
    } else await route.continue()
  })
  const list = page.getByRole('region', { name: '模型列表', exact: true })
  const items = list.locator('.model-item')
  const search = page.getByRole('textbox', { name: '搜索本地模型', exact: true })
  const filter = page.getByRole('combobox', { name: '筛选模型', exact: true })
  const sort = page.getByRole('combobox', { name: '模型排序', exact: true })
  const overview = page.locator('.model-overview')
  const locate = page.getByRole('button', { name: '定位当前模型', exact: true })
  const header = page.locator('.app-header')
  const selectionNote = page.locator('.library-selection-note')
  const item = (id: string) => list.locator(`.model-item[data-model-id="${id}"]`)
  const order = () => items.evaluateAll(elements => elements.map(element => element.getAttribute('data-model-id')))
  const settled = () => page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))))
  async function refreshState() {
    await header.hover()
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
  }
  async function nativeScreenshot(name: string) {
    await settled()
    const png = await application.evaluate(async ({ BrowserWindow }) => (await BrowserWindow.getAllWindows()[0].webContents.capturePage()).toPNG().toString('base64'))
    fs.writeFileSync(path.join(screenshots, name), Buffer.from(png, 'base64'))
  }
  async function withinList(target: Locator) {
    const region = await list.boundingBox()
    const box = await target.boundingBox()
    return !!region && !!box && box.y >= region.y - 1 && box.y + box.height <= region.y + region.height + 1
  }
  async function expectSelected(model: ModelAsset, visible = true) {
    await expect(overview.getByRole('heading', { name: model.name, exact: true })).toBeVisible()
    await expect(list.locator('.model-item[aria-pressed="true"]')).toHaveCount(visible ? 1 : 0)
    if (visible) {
      await expect(item(model.id)).toHaveAttribute('aria-pressed', 'true')
      await expect(item(model.id)).toHaveClass(/selected/)
      await expect(selectionNote).toBeHidden()
    } else {
      await expect(selectionNote).toBeVisible()
      await expect(selectionNote).toHaveAttribute('role', 'status')
    }
  }

  try {
    await header.hover({ timeout: 60_000 })
    await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
    const emptyLibrary = page.locator('.selection-empty')
    await expect(emptyLibrary.getByRole('heading', { name: '从一个本地模型开始', exact: true })).toBeVisible()
    await expect(emptyLibrary.locator('.import-choices button')).toHaveCount(3)
    await expect(emptyLibrary.locator('.empty-state > p, .selection-hint')).toHaveCount(0)
    await nativeScreenshot('models-empty.png')
    console.log(JSON.stringify({ electron: await application.evaluate(() => process.versions.electron), chromium: await application.evaluate(() => process.versions.chrome), fixtureModels: models.length, screenshots }))
    // Only /api/state is replaced; the real Electron renderer uses isolated UI assets.
    await page.route('**/api/state', async route => {
      const response = await route.fetch()
      const state = await response.json()
      await route.fulfill({ response, json: { ...state, models, instances, profiles: [], runtimes: [], downloads: [], installations: [], settings: { ...state.settings, font_scale: fontScale } } })
    })
    await application.evaluate(({ BrowserWindow }) => { const window = BrowserWindow.getAllWindows()[0]; window.setSize(1440, 960); window.webContents.setZoomFactor(1) })
    await refreshState()
    await expect(items).toHaveCount(models.length)
    await page.evaluate(() => document.fonts.ready)

    await test.step('NInfer 卡片使用简短精度摘要，原始格式仍可检索和查看', async () => {
      await search.fill('NInfer')
      await expect(items).toHaveCount(2)
      await expect(item('ninfer-int')).toContainText('混合整数量化')
      await expect(item('ninfer-nvfp4')).toContainText('NVFP4 混合精度')
      await expect(list).not.toContainText('q4_g64_fp16')
      await expect(list).not.toContainText('fp8_e4m3fn_row_bf16')
      await item('ninfer-nvfp4').click()
      await expect(overview).toContainText('NVFP4 混合精度')
      await expect(overview).not.toContainText('fp8_e4m3fn_row_bf16')
      await page.getByRole('button', { name: /模型文件/ }).click()
      await expect(page.locator('.files-panel')).toContainText(ninferModels[1].quantization!)
      await expect(page.locator('.file-list > .key-value')).toHaveCSS('align-items', 'stretch')
      await search.fill('q4_g64_fp16')
      await expect.poll(order).toEqual(['ninfer-int'])
      await search.fill('NInfer')
      await nativeScreenshot('models-ninfer-compact.png')
      await search.fill('')
      await item(sortingModels[0].id).click()
    })

    await test.step('文字 1.3、缩放 1.2 下到底后继续滚轮不会撑大模型库或产生空白', async () => {
      fontScale = 1.3
      await refreshState()
      await expect.poll(() => page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue('--font-scale').trim())).toBe('1.3')
      await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].webContents.setZoomFactor(1.2))
      await sort.selectOption('original')
      await list.evaluate(element => { element.scrollTop = element.scrollHeight })
      await item(longModel.id).click()
      await expectSelected(longModel)
      for (const warning of legacyWarnings) await expect(overview).not.toContainText(warning)
      await expect(overview.locator('.warning-note')).toHaveText('缺少权重分片：model-00002-of-00002.gguf')
      await expect(page.locator('.profile-panel .empty-state > p')).toHaveCount(0)
      await page.mouse.move(20, await page.evaluate(() => innerHeight - 28))
      await expect(header).not.toHaveClass(/is-open/)
      await expect(page.locator('.app-shell')).toHaveClass(/header-floating/)
      await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight))
      await settled()
      const geometry = () => page.evaluate(() => ({ scrollY, pageHeight: document.documentElement.scrollHeight, panelHeight: document.querySelector('.model-library')!.clientHeight }))
      const before = await geometry()
      await nativeScreenshot('models-bottom-before-wheel.png')
      const viewport = await page.evaluate(() => ({ width: innerWidth, height: innerHeight }))
      await page.mouse.move(viewport.width - 28, viewport.height - 28)
      const samples = []
      for (let index = 0; index < 12; index++) {
        await page.mouse.wheel(0, 480)
        await settled()
        samples.push(await geometry())
      }
      await nativeScreenshot('models-bottom-after-wheel.png')
      console.log(JSON.stringify({ bottomWheel: { before, samples } }))
      expect(Math.max(...samples.map(sample => sample.pageHeight))).toBeLessThanOrEqual(before.pageHeight + 1)
      expect(Math.max(...samples.map(sample => sample.panelHeight))).toBeLessThanOrEqual(before.panelHeight + 1)
      expect(Math.max(...samples.map(sample => sample.scrollY))).toBeLessThanOrEqual(before.scrollY + 1)
      await page.evaluate(() => window.scrollTo(0, 0))
      await header.hover()
      await page.getByRole('button', { name: '钉住顶部栏', exact: true }).click()
      await expect(page.locator('.app-shell')).toHaveClass(/header-pinned/)
      const listBottom = () => list.evaluate(element => element.getBoundingClientRect().bottom - innerHeight)
      await expect.poll(listBottom).toBeLessThanOrEqual(1)
      await page.getByRole('button', { name: '取消钉住顶部栏', exact: true }).click()
      await expect(page.locator('.app-shell')).toHaveClass(/header-floating/)
      await expect.poll(listBottom).toBeLessThanOrEqual(1)
      fontScale = 1
      await refreshState()
      await expect.poll(() => page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue('--font-scale').trim())).toBe('1')
      await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].webContents.setZoomFactor(1))
      await sort.selectOption('name-asc')
      await list.evaluate(element => { element.scrollTop = 0 })
      await item(sortingModels[0].id).click()
    })

    await test.step('自然名称、大小、格式、运行状态及登记顺序均有明确顺序', async () => {
      await search.fill('排序样例')
      await expect(items).toHaveCount(5)
      await expect(sort).toHaveValue('name-asc')
      const orders: Array<[string, string[]]> = [
        ['name-asc', ['sort-alpha1', 'sort-beta4', 'sort-llama3', 'sort-qwen2', 'sort-qwen10']],
        ['name-desc', ['sort-qwen10', 'sort-qwen2', 'sort-llama3', 'sort-beta4', 'sort-alpha1']],
        ['size-desc', ['sort-qwen10', 'sort-llama3', 'sort-qwen2', 'sort-beta4', 'sort-alpha1']],
        ['size-asc', ['sort-alpha1', 'sort-beta4', 'sort-llama3', 'sort-qwen2', 'sort-qwen10']],
        ['format', ['sort-beta4', 'sort-qwen2', 'sort-qwen10', 'sort-llama3', 'sort-alpha1']],
        ['running', ['sort-llama3', 'sort-qwen2', 'sort-alpha1', 'sort-beta4', 'sort-qwen10']],
        ['original', ['sort-qwen10', 'sort-llama3', 'sort-beta4', 'sort-qwen2', 'sort-alpha1']],
      ]
      for (const [value, expected] of orders) {
        await sort.selectOption(value)
        await expect.poll(order).toEqual(expected)
        await expectSelected(sortingModels[0])
      }
      await item('sort-qwen2').click()
      await expectSelected(sortingModels[3])
    })

    await test.step('组合搜索与筛选保持当前详情，空结果可恢复整个列表', async () => {
      await search.fill('   QWEN   q4_k_m VENDOR-A   ')
      await expect.poll(order).toEqual(['sort-qwen2'])
      await expectSelected(sortingModels[3])
      await search.fill('mistral safetensors VENDOR-C')
      await expect.poll(order).toEqual(['sort-alpha1'])
      await expectSelected(sortingModels[3], false)
      await expect(locate).toBeHidden()
      await search.fill('gguf Q8_0 vendor-b')
      await expect.poll(order).toEqual(['sort-qwen10'])
      await search.fill('排序样例')
      await filter.selectOption('gguf')
      await expect.poll(order).toEqual(['sort-qwen10', 'sort-beta4', 'sort-qwen2'])
      await filter.selectOption('running')
      await expect.poll(order).toEqual(['sort-llama3', 'sort-qwen2'])
      await expectSelected(sortingModels[3])
      await search.fill('没有任何模型匹配此关键词')
      await expect(items).toHaveCount(0)
      await expect(list.getByRole('heading', { name: '没有匹配的模型', exact: true })).toBeVisible()
      await expectSelected(sortingModels[3], false)
      await expect(locate).toBeHidden()
      await list.getByRole('button', { name: '清除筛选', exact: true }).click()
      await expect(search).toHaveValue('')
      await expect(filter).toHaveValue('all')
      await expect(items).toHaveCount(models.length)
      await expectSelected(sortingModels[3])
    })

    for (const display of [{ width: 1440, height: 960, zoom: 1, scale: 1 }, { width: 1050, height: 780, zoom: 1.25, scale: 1.3 }]) {
      await test.step(`${display.width}×${display.height}，缩放 ${display.zoom}、文字 ${display.scale} 下独立滚动并定位末项`, async () => {
        fontScale = display.scale
        await refreshState()
        await expect.poll(() => page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue('--font-scale').trim())).toBe(String(display.scale))
        await application.evaluate(({ BrowserWindow }, display) => { const window = BrowserWindow.getAllWindows()[0]; window.setSize(display.width, display.height); window.webContents.setZoomFactor(display.zoom) }, display)
        const contentWidth = await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].getContentSize()[0])
        await expect.poll(async () => Math.abs((await page.evaluate(() => innerWidth)) - contentWidth / display.zoom)).toBeLessThan(2)
        await sort.selectOption('original')
        await list.evaluate(element => { element.scrollTop = 0 })
        await item('sort-qwen10').click()
        await expectSelected(sortingModels[0])
        await settled()
        const before = await page.evaluate(() => {
          const list = document.querySelector('.model-list')!
          const toolbar = document.querySelector('.library-toolbar')!.getBoundingClientRect()
          const overview = document.querySelector('.model-overview')!.getBoundingClientRect()
          const region = list.getBoundingClientRect()
          return { scrollY, toolbarY: toolbar.top, overviewY: overview.top, listTop: region.top, listBottom: region.bottom, listHeight: list.clientHeight, contentHeight: list.scrollHeight, viewportHeight: innerHeight }
        })
        await nativeScreenshot(`models-${display.width}x${display.height}.png`)
        console.log(JSON.stringify({ display, geometry: before }))
        expect(before.listHeight).toBeGreaterThan(0)
        expect(before.contentHeight).toBeGreaterThan(before.listHeight * 3)
        expect(before.listTop).toBeGreaterThanOrEqual(0)
        expect(before.listBottom).toBeLessThanOrEqual(before.viewportHeight + 1)
        await list.evaluate(element => { element.scrollTop = element.scrollHeight })
        await expect.poll(() => withinList(item(longModel.id))).toBe(true)
        const after = await page.evaluate(() => ({ scrollY, toolbarY: document.querySelector('.library-toolbar')!.getBoundingClientRect().top, overviewY: document.querySelector('.model-overview')!.getBoundingClientRect().top }))
        expect(after.scrollY).toBe(before.scrollY)
        expect(Math.abs(after.toolbarY - before.toolbarY)).toBeLessThan(1)
        expect(Math.abs(after.overviewY - before.overviewY)).toBeLessThan(1)
        await item(longModel.id).click()
        await expectSelected(longModel)
        const name = item(longModel.id).locator('.model-item-name')
        const filePath = page.locator('.model-overview .path-strip > span')
        await expect(name).toHaveAttribute('title', longName)
        await expect(filePath).toHaveAttribute('title', longModel.path)
        const text = await item(longModel.id).evaluate(element => {
          const name = element.querySelector<HTMLElement>('.model-item-name')!
          const filePath = document.querySelector<HTMLElement>('.model-overview .path-strip > span')!
          const prefix = name.firstElementChild as HTMLElement
          return { nameHeight: name.clientHeight, nameClipped: prefix.scrollWidth > prefix.clientWidth, nameLineHeight: parseFloat(getComputedStyle(name).lineHeight), pathHeight: filePath.clientHeight, pathLineHeight: parseFloat(getComputedStyle(filePath).lineHeight), pathClipped: filePath.scrollWidth > filePath.clientWidth }
        })
        expect(text.nameHeight).toBeLessThanOrEqual(text.nameLineHeight * 2 + 1)
        expect(text.nameClipped).toBe(true)
        expect(text.pathHeight).toBeLessThanOrEqual(text.pathLineHeight + 1)
        expect(text.pathClipped).toBe(true)
        expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true)
        expect(await list.evaluate(element => element.scrollWidth <= element.clientWidth + 1)).toBe(true)
        await nativeScreenshot(`models-long-${display.width}x${display.height}.png`)
        await list.evaluate(element => { element.scrollTop = 0 })
        await expect.poll(() => withinList(item(longModel.id))).toBe(false)
        await settled()
        const beforeLocate = await page.evaluate(() => ({ scrollY, toolbarY: document.querySelector('.library-toolbar')!.getBoundingClientRect().top, overviewY: document.querySelector('.model-overview')!.getBoundingClientRect().top }))
        await locate.focus()
        await page.keyboard.press('Enter')
        await expect.poll(() => withinList(item(longModel.id))).toBe(true)
        await expectSelected(longModel)
        const afterLocate = await page.evaluate(() => ({ scrollY, toolbarY: document.querySelector('.library-toolbar')!.getBoundingClientRect().top, overviewY: document.querySelector('.model-overview')!.getBoundingClientRect().top }))
        expect(afterLocate.scrollY).toBe(beforeLocate.scrollY)
        expect(Math.abs(afterLocate.toolbarY - beforeLocate.toolbarY)).toBeLessThan(1)
        expect(Math.abs(afterLocate.overviewY - beforeLocate.overviewY)).toBeLessThan(1)
      })
    }

    await test.step('排序偏好在刷新界面后保留', async () => {
      await sort.selectOption('size-desc')
      await expect.poll(() => page.evaluate(() => localStorage.getItem('modloader.models.sort'))).toBe('size-desc')
      await page.reload()
      await expect(items).toHaveCount(models.length, { timeout: 60_000 })
      await expect(sort).toHaveValue('size-desc')
      await search.fill('排序样例')
      await expect.poll(order).toEqual(['sort-qwen10', 'sort-llama3', 'sort-qwen2', 'sort-beta4', 'sort-alpha1'])
    })
    expect(externalRequests).toEqual([])
    expect(mutationRequests).toEqual([])
    expect(errors).toEqual([])
  } finally {
    await page.unrouteAll({ behavior: 'wait' })
    await application.close()
  }
})
