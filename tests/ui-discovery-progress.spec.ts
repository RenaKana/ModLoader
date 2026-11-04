import { test, expect, _electron as electron } from '@playwright/test'
import * as path from 'node:path'
import * as fs from 'node:fs'
import { pinNavigation } from './navigation'

const detectPath = '/api/runtimes/detect'
const resolvePath = '/api/sources/resolve'
const readmePath = '/api/sources/readme'
const planPath = '/api/downloads/plan'
const files = [{ path: 'model.gguf', size: 1024, role: 'model', group_id: 'model' }]
const groups = [{ id: 'model', name: 'model.gguf', format: 'gguf', kind: 'model', files: ['model.gguf'], size: 1024 }]
const repository = { source: 'hf', repo_id: 'fixture/changed', revision: 'commit', name: 'model', metadata: {}, files, groups }

async function fixture() {
  const root = path.resolve(__dirname, '..')
  const directory = fs.mkdtempSync(path.join(root, 'work', 'ui-discovery-progress-'))
  const application = await electron.launch({
    ...(process.env.MODLOADER_TEST_EXECUTABLE ? { executablePath: process.env.MODLOADER_TEST_EXECUTABLE, args: [] } : { args: [root] }),
    env: { ...process.env, MODLOADER_DATA_DIR: path.join(directory, 'data') }, timeout: 60_000,
  })
  const page = await application.firstWindow()
  const errors: string[] = [], unexpected: string[] = []
  const state: any = {
    models: [], profiles: [], downloads: [], installations: [], migrations: [], instances: [], hardware: {}, settings: { close_to_tray: false },
    backends: [{ id: 'llama_cpp', label: 'llama.cpp', parameters: [], supported_platforms: ['windows'], installable: false }],
    runtimes: [{ id: 'registered', backend: 'llama_cpp', name: 'Fixture', path: 'C:\\Fixture\\llama-server.exe', version: 'b1', metadata: {} }],
  }
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/*', async route => {
    const request = route.request(), url = new URL(request.url())
    if (['http:', 'https:'].includes(url.protocol) && !['127.0.0.1', 'localhost'].includes(url.hostname)) { unexpected.push(url.origin); await route.abort(); return }
    if (!url.pathname.startsWith('/api/')) { await route.continue(); return }
    if (url.pathname === '/api/state') { await route.fulfill({ json: state }); return }
    if (url.pathname === '/api/chat-presets') { await route.fulfill({ json: { items: [], default_id: null } }); return }
    if (url.pathname === '/api/platforms/wsl') { await route.fulfill({ json: { available: false, distributions: [] } }); return }
    if (url.pathname === '/api/conversations' || url.pathname.startsWith('/api/conversations/')) { await route.continue(); return }
    if (url.pathname === '/api/installations/history') { await route.fulfill({ json: { items: [], total: 0, page: 1, page_size: 10 } }); return }
    unexpected.push(`${request.method()} ${url.pathname}`)
    await route.abort()
  })
  await pinNavigation(page)
  await page.getByRole('button', { name: '刷新状态', exact: true }).click()
  await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
  await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(1440, 960))
  // Controlled JSON responses and browser ReadableStreams keep requests pending.
  // They deliberately accept late results after abort to exercise the consuming
  // component's stale-response guard, not just native fetch abort.
  await page.evaluate(() => {
    const original = window.fetch.bind(window)
    const requests: any[] = []
    ;(window as any).progressFixture = requests
    window.fetch = async (input, init) => {
      const url = new URL(typeof input === 'string' ? input : input instanceof Request ? input.url : input.href, location.href)
      const pathname = url.pathname
      const search = /^\/api\/sources\/[^/]+\/search$/.test(pathname)
      if (!search && !['/api/runtimes/detect', '/api/runtimes/import', '/api/sources/resolve', '/api/sources/readme', '/api/downloads/plan'].includes(pathname) && !/^\/api\/runtimes\/[^/]+\/refresh$/.test(pathname)) return original(input, init)
      if (init?.signal?.aborted) throw new DOMException('Aborted', 'AbortError')
      const request: any = { path: pathname, query: Object.fromEntries(url.searchParams), body: init?.body ? JSON.parse(String(init.body)) : null, aborted: false }
      init?.signal?.addEventListener('abort', () => { request.aborted = true }, { once: true })
      requests.push(request)
      if (search) return new Promise<Response>(resolve => { request.respond = (value: unknown) => resolve(new Response(JSON.stringify(value), { headers: { 'Content-Type': 'application/json' } })) })
      const stream = new ReadableStream({ start(controller) { request.controller = controller } })
      return new Response(stream, { headers: { 'Content-Type': 'application/x-ndjson' } })
    }
  })
  async function captured(pathname: string, count: number) {
    await expect.poll(() => page.evaluate(path => (window as any).progressFixture.filter((item: any) => item.path === path).length, pathname)).toBe(count)
    return page.evaluate(path => {
      const requests = (window as any).progressFixture
      return requests.map((item: any, index: number) => ({ item, index })).filter(({ item }: any) => item.path === path).at(-1).index
    }, pathname)
  }
  async function send(index: number, frame: unknown) {
    await page.evaluate(({ index, frame }) => {
      const request = (window as any).progressFixture[index]
      request.controller.enqueue(new TextEncoder().encode(JSON.stringify(frame) + '\n'))
      if ((frame as any).type !== 'progress') request.controller.close()
    }, { index, frame })
    await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => resolve())))
  }
  async function aborted(index: number) { await expect.poll(() => page.evaluate(i => (window as any).progressFixture[i].aborted, index)).toBe(true) }
  async function sendJson(index: number, value: unknown) {
    await page.evaluate(({ index, value }) => (window as any).progressFixture[index].respond(value), { index, value })
    await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => resolve())))
  }
  async function screenshot(name: string) { await page.screenshot({ path: path.join(directory, name), animations: 'disabled' }) }
  return { application, page, state, errors, unexpected, directory, captured, send, sendJson, aborted, screenshot }
}

test('模型查询和加载更多只标记当前请求，换输入后旧结果不生效', async () => {
  test.setTimeout(60_000)
  const f = await fixture()
  const { page } = f
  const searchPath = '/api/sources/hf/search'
  try {
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '发现模型', exact: true }).click()
    const input = page.getByRole('textbox', { name: '搜索在线模型', exact: true })
    await input.fill('fixture')
    await page.getByRole('button', { name: '搜索', exact: true }).click()
    let search = await f.captured(searchPath, 1)
    await expect(page.getByRole('button', { name: '搜索中…', exact: true })).toBeDisabled()
    await expect(page.locator('.catalog-empty')).toContainText('正在搜索模型…')
    await expect(page.getByText('查找需要的模型', { exact: true })).toHaveCount(0)
    await f.screenshot('repository-search-progress.png')
    await f.sendJson(search, { items: [{ id: 'fixture/First', name: 'First' }], next_cursor: 'next' })
    await expect(page.locator('.repository-card')).toHaveCount(1)
    await page.getByRole('button', { name: '加载更多', exact: true }).click()
    search = await f.captured(searchPath, 2)
    expect(await page.evaluate(index => (window as any).progressFixture[index].query, search)).toEqual({ q: 'fixture', cursor: 'next' })
    await expect(page.getByRole('button', { name: '读取更多…', exact: true })).toBeDisabled()
    await expect(page.getByRole('button', { name: '搜索', exact: true })).toBeEnabled()
    await expect(page.locator('.search-form .spin')).toHaveCount(0)
    await expect(page.locator('.load-more .spin')).toHaveCount(1)
    await f.screenshot('repository-more-progress.png')
    await f.sendJson(search, { items: [{ id: 'fixture/First', name: 'First' }, { id: 'fixture/Second', name: 'Second' }], next_cursor: 'older' })
    await expect(page.locator('.repository-card')).toHaveCount(2)
    await page.getByRole('button', { name: '加载更多', exact: true }).click()
    search = await f.captured(searchPath, 3)
    await input.fill('replacement')
    await f.aborted(search)
    await f.sendJson(search, { items: [{ id: 'fixture/Stale', name: 'Stale' }], next_cursor: null })
    await expect(page.locator('.repository-card')).toHaveCount(2)
    await expect(input).toHaveValue('replacement')
    await page.getByRole('button', { name: '搜索', exact: true }).click()
    search = await f.captured(searchPath, 4)
    expect(await page.evaluate(index => (window as any).progressFixture[index].query, search)).toEqual({ q: 'replacement' })
    await expect(page.getByRole('button', { name: '搜索中…', exact: true })).toBeDisabled()
    await expect(page.getByRole('button', { name: '加载更多', exact: true })).toBeDisabled()
    await expect(page.locator('.load-more .spin')).toHaveCount(0)
    await f.sendJson(search, { items: [{ id: 'fixture/Replacement', name: 'Replacement' }], next_cursor: null })
    await expect(page.locator('.repository-card')).toHaveCount(1)
    await expect(page.locator('.repository-card')).toContainText('fixture/Replacement')
    expect(f.errors).toEqual([]); expect(f.unexpected).toEqual([])
    console.log(JSON.stringify({ screenshots: f.directory }))
  } finally { await f.application.close() }
})

test('环境搜索和验证显示真实状态，取消后旧结果不覆盖界面', async () => {
  test.setTimeout(90_000)
  const f = await fixture()
  const { page } = f
  try {
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '运行环境', exact: true }).click()
    await page.getByRole('button', { name: '自动搜索环境', exact: true }).click()
    const modal = page.getByRole('dialog', { name: '自动搜索运行环境', exact: true })
    let scan = await f.captured(detectPath, 1)
    await f.send(scan, { type: 'progress', stage: 'runtime_discovery', message: '已检查 12 个目录 · 已发现 2 个环境 · PATH · C:\\Fixture\\tools' })
    await expect(modal.locator('.operation-status')).toContainText('已检查 12 个目录 · 已发现 2 个环境')
    await expect(modal.locator('.operation-status')).toContainText('PATH')
    await expect(modal.getByRole('progressbar')).toHaveCount(0)
    await f.screenshot('runtime-search-progress.png')
    await modal.getByRole('button', { name: '取消搜索', exact: true }).click()
    await f.aborted(scan)
    await f.send(scan, { type: 'result', value: { runtimes: [], searched: [], errors: [], notes: [] } })
    await expect(modal).toContainText('搜索已取消')
    const candidate = { id: 'candidate', backend: 'llama_cpp', name: 'llama.cpp', path: 'C:\\Candidate\\llama-server.exe', version: 'unknown', metadata: {} }
    await modal.getByRole('button', { name: '重新搜索', exact: true }).click()
    scan = await f.captured(detectPath, 2)
    await f.send(scan, { type: 'result', value: { runtimes: [candidate], searched: [], errors: [], notes: [] } })
    await modal.getByRole('button', { name: '导入', exact: true }).click()
    const importing = await f.captured('/api/runtimes/import', 1)
    await f.send(importing, { type: 'progress', stage: 'runtime_import', message: '检查运行入口与版本' })
    await expect(modal.getByRole('button', { name: '验证并导入…', exact: true })).toBeDisabled()
    await expect(modal.locator('.runtime-search-candidate .operation-status')).toContainText('检查运行入口与版本')
    await f.screenshot('runtime-import-progress.png')
    f.state.runtimes.push(candidate)
    await f.send(importing, { type: 'result', value: candidate })
    await expect(modal.getByRole('button', { name: '已导入', exact: true })).toBeDisabled()
    await modal.getByRole('button', { name: '关闭', exact: true }).last().click()
    const registered = page.locator('[data-runtime-id="registered"]')
    await registered.getByRole('button', { name: '刷新版本', exact: true }).click()
    const refreshing = await f.captured('/api/runtimes/registered/refresh', 1)
    await f.send(refreshing, { type: 'progress', stage: 'runtime_verify', message: '检查 llama.cpp --version' })
    await expect(registered.locator('.operation-status')).toContainText('检查 llama.cpp --version')
    await expect(registered.getByRole('button', { name: '正在检查环境', exact: true })).toHaveAttribute('aria-busy', 'true')
    await expect(page.locator('[data-runtime-id="candidate"] .operation-status')).toHaveCount(0)
    await f.screenshot('runtime-refresh-progress.png')
    f.state.runtimes[0].version = 'b2'
    await f.send(refreshing, { type: 'result', value: f.state.runtimes[0] })
    await expect(registered.locator('.operation-status')).toHaveCount(0)
    await expect(registered.getByRole('button', { name: '刷新版本', exact: true })).toBeEnabled()
    expect(f.errors).toEqual([]); expect(f.unexpected).toEqual([])
    console.log(JSON.stringify({ screenshots: f.directory }))
  } finally { await f.application.close() }
})

test('仓库、下载检查和 README 分阶段反馈，取消及换输入保留选择', async () => {
  test.setTimeout(90_000)
  const f = await fixture()
  const { page } = f
  try {
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '发现模型', exact: true }).click()
    await page.getByRole('button', { name: '打开模型仓库', exact: true }).first().click()
    const opening = page.getByRole('dialog', { name: '打开 Hugging Face 仓库', exact: true })
    await opening.getByRole('textbox', { name: '仓库 ID', exact: true }).fill('fixture/original')
    await opening.getByRole('textbox', { name: '版本或分支', exact: true }).fill('branch')
    await opening.getByRole('button', { name: '读取文件清单', exact: true }).click()
    let resolving = await f.captured(resolvePath, 1)
    await f.send(resolving, { type: 'progress', stage: 'repository', message: '读取仓库清单 · fixture/original' })
    await expect(opening.locator('.operation-status')).toContainText('读取仓库清单')
    await f.send(resolving, { type: 'progress', stage: 'weight_indexes', message: '检查权重索引 · weights.index.json', completed: 1, total: 3 })
    await expect(opening.locator('.operation-status')).toContainText('1 / 3')
    await f.screenshot('repository-resolve-progress.png')
    await opening.getByRole('button', { name: '取消读取', exact: true }).click()
    await f.aborted(resolving)
    await expect(opening.getByRole('textbox', { name: '仓库 ID', exact: true })).toHaveValue('fixture/original')
    await expect(opening.getByRole('textbox', { name: '版本或分支', exact: true })).toHaveValue('branch')
    await f.send(resolving, { type: 'result', value: repository })
    await expect(opening).toBeVisible()
    await opening.getByRole('button', { name: '读取文件清单', exact: true }).click()
    resolving = await f.captured(resolvePath, 2)
    await opening.getByRole('textbox', { name: '仓库 ID', exact: true }).fill('fixture/changed')
    await f.aborted(resolving)
    await f.send(resolving, { type: 'result', value: repository })
    await expect(opening).toBeVisible()
    await opening.getByRole('button', { name: '读取文件清单', exact: true }).click()
    resolving = await f.captured(resolvePath, 3)
    await f.send(resolving, { type: 'result', value: repository })
    const selection = page.getByRole('dialog', { name: '选择下载文件', exact: true })
    await selection.getByRole('checkbox', { name: 'model.gguf', exact: true }).check()
    await selection.getByRole('button', { name: '检查下载', exact: true }).click()
    let planning = await f.captured(planPath, 1)
    await f.send(planning, { type: 'progress', stage: 'model_config', message: '检查模型配置 · config.json', completed: 0, total: 2 })
    await expect(selection.locator('.operation-status')).toContainText('检查模型配置')
    await expect(selection.locator('.operation-status')).toContainText('0 / 2')
    await f.screenshot('repository-plan-progress.png')
    await selection.getByRole('textbox', { name: '仓库版本', exact: true }).fill('new-branch')
    await f.aborted(planning)
    await f.send(planning, { type: 'result', value: { files, revision: 'old-branch' } })
    await expect(selection.getByRole('checkbox', { name: 'model.gguf', exact: true })).toBeChecked()
    await expect(selection.getByRole('button', { name: '开始下载', exact: true })).toHaveCount(0)
    await selection.getByRole('button', { name: '检查下载', exact: true }).click()
    planning = await f.captured(planPath, 2)
    await f.send(planning, { type: 'error', status: 400, detail: '配置 fixture 读取失败' })
    await expect(page.getByText('配置 fixture 读取失败', { exact: true })).toBeVisible()
    await expect(selection.getByRole('textbox', { name: '仓库版本', exact: true })).toHaveValue('new-branch')
    await expect(selection.getByRole('checkbox', { name: 'model.gguf', exact: true })).toBeChecked()
    await selection.getByRole('button', { name: '检查下载', exact: true }).click()
    planning = await f.captured(planPath, 3)
    await f.send(planning, { type: 'result', value: { files, revision: 'new-commit' } })
    await expect(selection.getByRole('button', { name: '开始下载', exact: true })).toBeEnabled()
    await selection.getByRole('button', { name: '模型介绍', exact: true }).click()
    const intro = page.getByRole('dialog', { name: '模型介绍', exact: true })
    let reading = await f.captured(readmePath, 1)
    await f.send(reading, { type: 'progress', stage: 'readme', message: '读取 README · fixture/changed' })
    await expect(intro.locator('.operation-status')).toContainText('读取 README')
    await f.screenshot('repository-readme-progress.png')
    await intro.getByRole('button', { name: '下载文件', exact: true }).click()
    await f.aborted(reading)
    await f.send(reading, { type: 'result', value: { path: 'README.md', content: '# 过时文档', revision: 'commit', url: '' } })
    await expect(selection.getByRole('checkbox', { name: 'model.gguf', exact: true })).toBeChecked()
    await selection.getByRole('button', { name: '模型介绍', exact: true }).click()
    reading = await f.captured(readmePath, 2)
    await f.send(reading, { type: 'result', value: { path: 'README.md', content: '# 当前文档', revision: 'commit', url: '' } })
    await expect(intro.frameLocator('.readme-document').getByRole('heading', { name: '当前文档', exact: true })).toBeVisible()
    await intro.getByRole('button', { name: '关闭', exact: true }).last().click()
    await page.getByRole('button', { name: '直接链接下载', exact: true }).click()
    const direct = page.getByRole('dialog', { name: '直接链接下载', exact: true })
    await direct.getByRole('textbox', { name: '下载链接', exact: true }).fill('https://example.invalid/model.gguf')
    await direct.getByRole('button', { name: '检查下载', exact: true }).click()
    planning = await f.captured(planPath, 4)
    await f.send(planning, { type: 'progress', stage: 'file_info', message: '检查链接与文件大小' })
    await expect(direct.locator('.operation-status')).toContainText('检查链接与文件大小')
    await direct.getByRole('button', { name: '取消检查', exact: true }).click()
    await f.aborted(planning)
    await f.send(planning, { type: 'result', value: { files, total_bytes: 1024 } })
    await expect(direct.getByRole('textbox', { name: '下载链接', exact: true })).toHaveValue('https://example.invalid/model.gguf')
    await expect(direct.getByRole('button', { name: '开始下载', exact: true })).toHaveCount(0)
    await direct.getByRole('button', { name: '取消', exact: true }).click()
    expect(f.errors).toEqual([]); expect(f.unexpected).toEqual([])
    console.log(JSON.stringify({ screenshots: f.directory }))
  } finally { await f.application.close() }
})
