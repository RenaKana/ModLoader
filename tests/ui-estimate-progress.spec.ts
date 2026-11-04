import { test, expect, _electron as electron } from '@playwright/test'
import * as path from 'node:path'
import * as fs from 'node:fs'
import { pinNavigation } from './navigation'

test('资源估算和预检实时阶段、取消旧参数、失败重试', async () => {
  test.setTimeout(90_000)
  const root = path.resolve(__dirname, '..')
  const directory = fs.mkdtempSync(path.join(root, 'work', 'ui-estimate-progress-'))
  const application = await electron.launch({ args: [root], env: { ...process.env,
    MODLOADER_DATA_DIR: path.join(directory, 'data'), MODLOADER_SENSOR_AUTO_START: '0' }, timeout: 60_000 })
  const page = await application.firstWindow()
  const errors: string[] = [], unexpected: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  const model = { id: 'm', name: '估算进度 fixture', path: 'C:\\fixture\\model.gguf', format: 'gguf', files: [], modalities: ['text'], metadata: {}, source: {}, managed: false, warnings: [] }
  const profile = { id: 'p', model_id: 'm', name: '估算配置', alias: 'fixture', runtime_id: 'r', backend: 'llama_cpp', parameters: { context_length: 4096 }, generation: {}, extra_args: [], revision: 1 }
  let present = false
  const state = { models: [model], profiles: [profile], runtimes: [{ id: 'r', backend: 'llama_cpp', name: 'fixture', path: 'C:\\fixture\\llama-server.exe', metadata: {} }],
    instances: [], downloads: [], installations: [], migrations: [], settings: { close_to_tray: false },
    hardware: { memory_total: 16 * 1024 ** 3, memory_available: 12 * 1024 ** 3 },
    backends: [{ id: 'llama_cpp', label: 'llama.cpp', parameters: [{ key: 'context_length', label: '上下文长度', type: 'number', default: 4096, options: [] }] }] }
  await page.route('**/api/**', async route => {
    const url = new URL(route.request().url())
    if (url.pathname === '/api/state') return route.fulfill({ json: { ...state, models: present ? state.models : [], profiles: present ? state.profiles : [] } })
    if (url.pathname === '/api/platforms/wsl') return route.fulfill({ json: { available: false, distributions: [] } })
    if (url.pathname.startsWith('/api/conversations') || url.pathname === '/api/chat-presets') return route.continue()
    unexpected.push(url.pathname)
    return route.abort()
  })
  try {
    await pinNavigation(page)
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
    await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(1360, 960))
    await page.evaluate(() => {
      const original = window.fetch.bind(window)
      const requests: any[] = []
      ;(window as any).estimateFixture = requests
      window.fetch = async (input, init) => {
        const url = new URL(typeof input === 'string' ? input : input instanceof Request ? input.url : input.href, location.href)
        if (!['/api/resources/estimate', '/api/profiles/preflight'].includes(url.pathname)) return original(input, init)
        const request: any = { path: url.pathname, body: JSON.parse(String(init?.body)), aborted: false }
        const stream = new ReadableStream({ start(controller) { request.controller = controller } })
        init?.signal?.addEventListener('abort', () => { request.aborted = true }, { once: true })
        requests.push(request)
        return new Response(stream, { headers: { 'Content-Type': 'application/x-ndjson' } })
      }
    })
    const captured = async (pathname: string, count: number) => {
      await expect.poll(() => page.evaluate(pathname => (window as any).estimateFixture.filter((item: any) => item.path === pathname).length, pathname)).toBe(count)
      return page.evaluate(pathname => (window as any).estimateFixture.map((item: any, index: number) => ({ item, index })).filter(({ item }: any) => item.path === pathname).at(-1).index, pathname)
    }
    const send = (index: number, frame: unknown) => page.evaluate(({ index, frame }) => {
      const controller = (window as any).estimateFixture[index].controller
      const bytes = new TextEncoder().encode(JSON.stringify(frame) + '\n')
      // Split a multibyte frame as a real network body can do.
      controller.enqueue(bytes.slice(0, 17)); controller.enqueue(bytes.slice(17))
      if ((frame as any).type !== 'progress') controller.close()
    }, { index, frame })
    const result = (gb: number) => ({ resources: [{ kind: 'ram', label: '系统内存', bytes: gb * 1024 ** 3, quality: 'estimate', notes: [] }], source: 'fixture', notes: [] })
    const ready = { status: 'ready', candidates: [], errors: [], mtp: { status: 'unsupported', backend_supported: false, enabled: false, candidates: [] } }
    present = true
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    const card = page.locator('.profile-card').filter({ hasText: profile.name })
    const preview = card.locator('.resource-preview')
    const first = await captured('/api/resources/estimate', 1)
    await send(first, { type: 'progress', stage: 'model_metadata', message: '读取分片信息', completed: 0, total: 4 })
    await expect(preview.getByRole('progressbar')).toHaveAttribute('value', '0')
    await expect(preview).toContainText('0 / 4')
    await send(first, { type: 'progress', stage: 'model_metadata', message: '读取分片信息', completed: 1, total: 4 })
    await expect(preview.getByRole('progressbar')).toHaveAttribute('value', '1')
    await expect(preview.locator('.operation-status-time')).toContainText('秒')
    await preview.screenshot({ path: path.join(directory, 'metadata-progress.png') })
    await send(first, { type: 'result', value: result(2) })
    await send(await captured('/api/profiles/preflight', 1), { type: 'result', value: ready })
    await expect(preview.locator('.operation-status')).toHaveCount(0)
    await expect(preview).toContainText('2 GB')

    await card.getByRole('button', { name: '编辑配置', exact: true }).click()
    const dialog = page.getByRole('dialog', { name: '编辑加载配置', exact: true })
    const editorPreview = dialog.locator('.resource-preview')
    const obsolete = await captured('/api/resources/estimate', 2)
    const check = await captured('/api/profiles/preflight', 2)
    await send(check, { type: 'progress', stage: 'mtp', message: '检查 MTP 支持与配套文件' })
    await expect(dialog.locator('.profile-acceleration .operation-status')).toContainText('检查 MTP')
    await send(check, { type: 'result', value: ready })
    const context = dialog.getByLabel('上下文长度', { exact: true })
    await context.fill('8192'); await context.fill('16384')
    const current = await captured('/api/resources/estimate', 3)
    await expect.poll(() => page.evaluate(i => (window as any).estimateFixture[i].aborted, obsolete)).toBe(true)
    await send(obsolete, { type: 'result', value: result(99) })
    await send(current, { type: 'progress', stage: 'estimate_queue', message: '等待前一个原生估算完成' })
    await expect(editorPreview).toContainText('等待前一个原生估算完成')
    await expect(editorPreview.getByRole('progressbar')).toHaveCount(0)
    await expect(editorPreview).not.toContainText('99 GB')
    await send(current, { type: 'error', status: 400, detail: '模型分片不可读取，请检查路径' })
    await expect(editorPreview).toContainText('资源估算失败')
    await expect(context).toHaveValue('16384')
    await editorPreview.getByRole('button', { name: '重试', exact: true }).click()
    const retry = await captured('/api/resources/estimate', 4)
    await send(retry, { type: 'progress', stage: 'native_estimate', message: '计算模型、上下文与缓冲占用' })
    await expect(editorPreview).toContainText('计算模型、上下文与缓冲占用')
    await dialog.screenshot({ path: path.join(directory, 'estimate-retry.png') })
    await send(retry, { type: 'result', value: result(4) })
    await expect(editorPreview).toContainText('4 GB')
    await expect(editorPreview.locator('.operation-status')).toHaveCount(0)
    await dialog.getByRole('button', { name: '取消', exact: true }).click()
    await expect(dialog).toHaveCount(0)
    expect(errors).toEqual([]); expect(unexpected).toEqual([])
    console.log(JSON.stringify({ screenshots: directory }))
  } finally { await application.close() }
})
