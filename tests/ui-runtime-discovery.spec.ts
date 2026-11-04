import { pinNavigation } from './navigation'
import { test, expect, _electron as electron } from '@playwright/test'
import * as path from 'node:path'
import * as fs from 'node:fs'

test('本机环境搜索、文件夹递归搜索与显式登记', async () => {
  test.setTimeout(90_000)
  const root = path.resolve(__dirname, '..')
  const task = fs.mkdtempSync(path.join(root, 'work', 'ui-runtime-discovery-'))
  const folder = path.join(task, 'tools')
  const strata = path.join(folder, 'portable', 'vendor', 'releases', 'current', 'Strata')
  for (const file of ['serve/server.py', 'engine/strata.exe', '.venv/Scripts/python.exe']) {
    const target = path.join(strata, file); fs.mkdirSync(path.dirname(target), { recursive: true }); fs.writeFileSync(target, 'fixture, never executed')
  }
  fs.writeFileSync(path.join(strata, 'engine', 'BUILD.json'), JSON.stringify({ version: 'ui-static-fixture' }))
  const executable = process.env.MODLOADER_TEST_EXECUTABLE
  const app = await electron.launch({ ...(executable ? { executablePath: executable, args: [] } : { args: [root] }), env: { ...process.env, MODLOADER_DATA_DIR: path.join(task, 'data') }, timeout: 60_000 })
  const page = await app.firstWindow()
  const errors: string[] = [], external: string[] = [], lifecycle: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  page.on('request', request => { if (/\/api\/(downloads|runtimes\/install|instances)/.test(new URL(request.url()).pathname) && request.method() !== 'GET') lifecycle.push(request.url()) })
  await page.route('**/*', async route => {
    const url = new URL(route.request().url())
    if (['http:', 'https:'].includes(url.protocol) && !['127.0.0.1', 'localhost'].includes(url.hostname)) { external.push(url.origin); await route.abort() }
    else await route.continue()
  })
  async function apiState() {
    return page.evaluate(async () => { const { baseUrl, token } = await window.modloader!.connection(); return (await fetch(baseUrl + '/api/state', { headers: { Authorization: 'Bearer ' + token } })).json() })
  }
  async function screenshot(name: string) {
    const png = await app.evaluate(async ({ BrowserWindow }) => (await BrowserWindow.getAllWindows()[0].webContents.capturePage()).toPNG().toString('base64'))
    fs.writeFileSync(path.join(task, name), Buffer.from(png, 'base64'))
  }
  try {
    await pinNavigation(page)
    await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
    await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(1440, 960))
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '运行环境', exact: true }).click()
    const scan = page.waitForResponse(response => new URL(response.url()).pathname === '/api/runtimes/detect')
    await page.getByRole('button', { name: '自动搜索环境', exact: true }).click()
    const scanResponse = await scan
    const scanText = await scanResponse.text()
    const realResults = scanResponse.headers()['content-type']?.includes('application/x-ndjson')
      ? JSON.parse(scanText.trim().split('\n').at(-1)!).value : JSON.parse(scanText)
    const modal = page.getByRole('dialog', { name: '自动搜索运行环境', exact: true })
    await expect(modal.getByText(`找到 ${realResults.runtimes.length} 个环境`, { exact: true })).toBeVisible()
    expect((await apiState()).runtimes).toHaveLength(0)
    await screenshot('default-discovery.png')
    // Only the native folder dialog result is supplied; scanning and registration use the real core.
    await app.evaluate(({ dialog }, selected) => { dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [selected] }) }, folder)
    await modal.getByRole('button', { name: '选择文件夹搜索', exact: true }).click()
    await expect(modal.getByText('找到 1 个环境', { exact: true })).toBeVisible()
    await expect(modal.locator('.runtime-search-candidate .mono')).toHaveText(strata)
    await modal.getByRole('button', { name: '导入', exact: true }).click()
    await expect(modal.getByRole('button', { name: '已导入', exact: true })).toBeDisabled()
    await modal.getByRole('button', { name: '重新搜索', exact: true }).click()
    await expect(modal.getByRole('button', { name: '已导入', exact: true })).toBeDisabled()
    const state = await apiState()
    expect(state.runtimes).toHaveLength(1)
    expect(state.runtimes[0].path).toBe(strata)
    expect(state.runtimes[0].version).toBe('ui-static-fixture')
    expect(state.instances).toHaveLength(0)
    expect(state.downloads).toHaveLength(0)
    await app.evaluate(({ BrowserWindow }) => { const win = BrowserWindow.getAllWindows()[0]; win.setSize(1050, 780); win.webContents.setZoomFactor(1.25) })
    await page.evaluate(() => { document.documentElement.style.setProperty('--font-scale', '1.3'); document.querySelector('.app-shell')?.classList.add('font-expanded') })
    await modal.getByText(/^已搜索的位置（/).click()
    await expect.poll(() => modal.evaluate(element => element.scrollWidth <= element.clientWidth + 1)).toBe(true)
    await screenshot('folder-discovery-zoom.png')
    let releaseScan!: () => void
    const pausedScan = new Promise<void>(resolve => { releaseScan = resolve })
    await page.route('**/api/runtimes/detect?*', async route => { await pausedScan; await route.abort().catch(() => {}) })
    await modal.getByRole('button', { name: '重新搜索', exact: true }).click()
    await expect(modal.getByRole('status', { name: '正在查找运行环境', exact: true })).toBeVisible()
    await modal.getByRole('button', { name: '取消搜索', exact: true }).click()
    releaseScan()
    await expect(modal.getByText('搜索已取消', { exact: true })).toBeVisible()
    await modal.getByRole('button', { name: '关闭', exact: true }).last().click()
    expect(errors).toEqual([]); expect(external).toEqual([]); expect(lifecycle).toEqual([])
    console.log(JSON.stringify({ screenshots: task, defaultCandidates: realResults.runtimes.map((runtime: { backend: string; version: string }) => ({ backend: runtime.backend, version: runtime.version })) }))
  } finally { await app.close() }
})
