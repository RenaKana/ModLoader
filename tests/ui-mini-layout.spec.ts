import { expect, test, _electron as electron, type ElectronApplication, type Page } from '@playwright/test'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { build, preview, type PreviewServer } from 'vite'
import { miniMetrics, MINI_PREFERENCE_KEY, type MiniMonitorSnapshot } from '../src/mini-monitor'

const root = path.resolve(__dirname, '..')
const artifacts = path.join(root, '.artifacts/codex/number-layout-fix-20261101/mini')
const visible = ['duration', 'current', 'average', 'context', 'prompt', 'generated']
let server: PreviewServer, baseURL: string

test.beforeAll(async () => {
  test.setTimeout(90_000)
  fs.mkdirSync(artifacts, { recursive: true })
  const outDir = process.env.MODLOADER_TEST_DIST || path.join(artifacts, 'fixture-dist')
  if (!process.env.MODLOADER_TEST_DIST) await build({ root, cacheDir: path.join(artifacts, 'vite-cache'), build: { outDir } })
  server = await preview({ root, configFile: false, build: { outDir }, preview: { host: '127.0.0.1', port: 0 } })
  const address = server.httpServer.address()
  if (!address || typeof address === 'string') throw Error('No fixture port')
  baseURL = `http://127.0.0.1:${address.port}/index.html`
})

test.afterAll(async () => {
  if (server) await new Promise<void>(resolve => server.httpServer.close(() => resolve()))
})

function snapshot(long = false): MiniMonitorSnapshot {
  const metrics = Object.fromEntries(miniMetrics.map(metric => [metric.id, { value: '12.5', unit: '%' }])) as MiniMonitorSnapshot['metrics']
  Object.assign(metrics, {
    duration: { value: long ? '1,234.56' : '99.99', unit: 's', basis: '上次' },
    current: { value: long ? '12,345.67' : '99.99', unit: 'token/s' },
    average: { value: long ? '11,234.56' : '99.99', unit: 'token/s', basis: '引擎平均' },
    context: { value: long ? '1,023,999 / 1,048,576' : '99,999 / 262,144', unit: 'token', basis: '上次采样' },
    prompt: { value: long ? '999,999' : '98,999', unit: 'token', basis: '上次' },
    generated: { value: long ? '24,000' : '1,000', unit: 'token', basis: '上次' },
    memory: { value: '38.7 / 127.7', unit: 'GB' }, vram: { value: '19.6 / 24', unit: 'GB' },
    first_output: { value: '0.42', unit: 's', basis: '引擎' },
    running: { value: '2', unit: '个' }, queued: { value: '0', unit: '个' },
    cpu_temperature: { value: '58', unit: '°C' }, gpu_temperature: { value: '67', unit: '°C' },
    cpu_power: { value: '89.6', unit: 'W' }, gpu_power: { value: '286.4', unit: 'W' },
    disk: { value: '1,024.8', unit: 'MB/s' }, pcie_rx: { value: '12.48', unit: 'GB/s' }, pcie_tx: { value: '1.32', unit: 'GB/s' },
  })
  return { instanceId: 'fixture', instances: [{ id: 'fixture', label: 'Qwen3.8-Flash', state: 'ready' }],
    gpuId: 'gpu', gpus: [{ id: 'gpu', label: 'GPU' }], stateLabel: '空闲', tone: 'idle', fontScale: 1.1, metrics }
}

async function fixture() {
  const directory = fs.mkdtempSync(path.join(artifacts, 'native-'))
  const data = path.join(directory, 'data'), launcher = path.join(directory, 'main.cjs')
  fs.writeFileSync(launcher, `const { app, BrowserWindow, ipcMain } = require('electron');
    const { createMiniMonitor } = require(${JSON.stringify(path.join(root, 'electron/mini-monitor.cjs'))});
    app.setPath('userData', ${JSON.stringify(data)});
    app.whenReady().then(async () => {
      const main = new BrowserWindow({ show: false }); await main.loadURL('about:blank');
      const trusted = event => event.sender === main.webContents && event.senderFrame === main.webContents.mainFrame;
      const controller = createMiniMonitor({ mainWindow: main, dataRoot: ${JSON.stringify(data)}, projectRoot: ${JSON.stringify(root)}, devUrl: ${JSON.stringify(baseURL)}, isTrustedMainFrame: trusted });
      globalThis.miniFixture = { main, controller, publish: snapshot => ipcMain.emit('mini-monitor:publish', { sender: main.webContents, senderFrame: main.webContents.mainFrame }, snapshot) };
    }); app.on('window-all-closed', () => app.quit());`)
  let application: ElectronApplication | undefined
  try {
    application = await electron.launch({ args: [launcher], timeout: 60_000 })
    await application.firstWindow()
    await expect.poll(() => application!.evaluate(() => !!(globalThis as any).miniFixture)).toBe(true)
    const publish = (value: MiniMonitorSnapshot) => application!.evaluate((_electron, value) => (globalThis as any).miniFixture.publish(value), value)
    await publish(snapshot())
    const open = async () => {
      const window = application!.waitForEvent('window', { timeout: 20_000 })
      // BrowserWindow creation must run outside the Node inspector evaluation.
      await application!.evaluate(() => {
        const fixture = (globalThis as any).miniFixture
        fixture.openState = { status: 'opening' }
        setImmediate(() => {
          fixture.controller.open().then(
            () => { fixture.openState = { status: 'ready' } },
            (error: Error) => { fixture.openState = { status: 'failed', error: error.message } },
          )
        })
      })
      const mini = await window.catch(async error => {
        const state = await application!.evaluate(({ BrowserWindow }) => ({
          opening: (globalThis as any).miniFixture.openState,
          windows: BrowserWindow.getAllWindows().map(win => ({
            url: win.webContents.getURL(), visible: win.isVisible(), loading: win.webContents.isLoading(),
          })),
        }))
        throw Error(`${error.message}\nNative fixture: ${JSON.stringify(state)}`)
      })
      await expect(mini.locator('.mini-frame')).toBeVisible()
      return mini
    }
    const mini = await open()
    await mini.evaluate(({ key, ids }) => localStorage.setItem(key, JSON.stringify(ids)), { key: MINI_PREFERENCE_KEY, ids: visible })
    await mini.reload(); await expect(mini.locator('[data-mini-metric]')).toHaveCount(6)
    const bounds = () => application!.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows().find(win => win.webContents.getURL().includes('window=mini-monitor'))!.getBounds())
    const resize = (width: number, height: number) => application!.evaluate(({ BrowserWindow }, size) => {
      const win = BrowserWindow.getAllWindows().find(win => win.webContents.getURL().includes('window=mini-monitor'))!
      // setSize itself is programmatic. Supply the user-resize event before applying its native bounds.
      win.emit('will-resize', { preventDefault() {} } as Electron.Event, { ...win.getBounds(), ...size }, { edge: 'bottomRight' })
      win.setSize(size.width, size.height)
    }, { width, height })
    return { application, mini, publish, directory, data, bounds, resize, open,
      async close() { await application?.close() } }
  } catch (error) {
    fs.writeFileSync(path.join(directory, 'fixture-error.txt'), String(error))
    console.error(String(error))
    await application?.close().catch(() => {}); throw error
  }
}

const geometry = (page: Page) => page.locator('[data-mini-metric]').evaluateAll(nodes => nodes.map(node => {
  const card = node as HTMLElement, box = card.getBoundingClientRect()
  const readout = card.querySelector('.mini-readout')!.getBoundingClientRect()
  const unit = card.querySelector('.mini-unit')?.getBoundingClientRect()
  const text = [...card.querySelectorAll<HTMLElement>('.mini-readout strong, .mini-total, .mini-basis, .mini-unit')]
  return { id: card.dataset.miniMetric, height: box.height, readoutHeight: readout.height,
    unitY: unit ? unit.top - box.top : null,
    overflow: text.some(value => { const rect = value.getBoundingClientRect(); return rect.left < box.left || rect.right > box.right || value.scrollWidth > value.clientWidth + 1 }) }
}))

const clippedReadouts = (page: Page) => page.locator('.mini-value-line').evaluateAll(nodes => nodes.flatMap(node => {
  const line = node as HTMLElement
  return line.scrollWidth > line.clientWidth + 1 || line.scrollHeight > line.clientHeight + 1
    ? [{ id: line.closest<HTMLElement>('[data-mini-metric]')!.dataset.miniMetric,
      width: line.clientWidth, scrollWidth: line.scrollWidth, height: line.clientHeight, scrollHeight: line.scrollHeight }] : []
}))

test('小窗所有指标完整显示且同高，跨位和来源标签不推动单位或数字换行', async () => {
  test.setTimeout(90_000)
  const f = await fixture()
  try {
    await f.resize(380, 420)
    const before = await geometry(f.mini)
    expect(before.every(item => item.unitY !== null)).toBe(true)
    expect(new Set(before.map(item => item.height)).size).toBe(1)
    await f.publish(snapshot(true))
    await expect(f.mini.locator('[data-mini-metric="current"] strong')).toHaveText('12,345.67')
    expect(await geometry(f.mini)).toEqual(before)
    await expect(f.mini.locator('[data-mini-metric="context"] .mini-total')).toHaveText('/ 1,048,576')
    await expect(f.mini.locator('[data-mini-metric="average"] .mini-basis')).toHaveText('引擎平均')
    await f.mini.evaluate(({ key, ids }) => localStorage.setItem(key, JSON.stringify(ids)), { key: MINI_PREFERENCE_KEY, ids: miniMetrics.map(item => item.id) })
    await f.mini.reload(); await expect(f.mini.locator('[data-mini-metric]')).toHaveCount(20)
    for (const width of [220, 380, 700]) {
      await f.resize(width, 380)
      const values = await geometry(f.mini)
      expect(values.every(item => !item.overflow), JSON.stringify(values)).toBe(true)
      expect(new Set(values.map(item => item.height)).size).toBe(1)
      expect(await clippedReadouts(f.mini)).toEqual([])
    }
    await f.publish({ ...snapshot(true), fontScale: 1.3 })
    await f.resize(220, 380)
    await expect(f.mini.locator('.mini-frame')).toHaveCSS('--mini-scale', '1.3')
    expect((await geometry(f.mini)).every(item => !item.overflow)).toBe(true)
    expect(await clippedReadouts(f.mini)).toEqual([])
    expect(await f.mini.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
    await f.mini.screenshot({ path: path.join(f.directory, 'narrow.png') })
    await f.resize(700, 380)
    await f.mini.getByRole('button', { name: '适应内容', exact: true }).click()
    await f.mini.screenshot({ path: path.join(f.directory, 'all-metrics.png') })
  } finally { await f.close() }
})

test('原生适应内容保持宽度，手动高度不拉伸卡片，滚动不改列数且模式可恢复', async () => {
  test.setTimeout(90_000)
  const f = await fixture()
  try {
    await f.resize(380, 500)
    const heights = (await geometry(f.mini)).map(item => item.height)
    const fit = f.mini.getByRole('button', { name: '适应内容', exact: true })
    await expect(fit).toBeVisible()
    await expect(fit).toHaveText('')
    await expect(fit.locator('svg')).toHaveCount(1)
    await fit.click()
    await expect.poll(async () => (await f.bounds()).height).toBeLessThan(400)
    const fitted = await f.bounds()
    expect(fitted.width).toBe(380)
    const columns = () => f.mini.locator('.mini-grid').evaluate(node => getComputedStyle(node).gridTemplateColumns.split(' ').length)
    const originalColumns = await columns()
    await f.resize(380, 190)
    await expect.poll(() => f.mini.locator('.mini-scroll').evaluate(node => node.scrollHeight > node.clientHeight)).toBe(true)
    expect(await columns()).toBe(originalColumns)
    expect((await geometry(f.mini)).map(item => item.height)).toEqual(heights)
    await f.publish(snapshot(true))
    await expect(f.mini.locator('[data-mini-metric="current"] strong')).toHaveText('12,345.67')
    expect((await f.bounds()).height).toBe(190)
    await f.resize(700, 190)
    await expect.poll(columns).toBeGreaterThan(originalColumns)
    expect((await f.bounds()).height).toBe(190)
    await f.resize(380, 190)
    await expect.poll(columns).toBe(originalColumns)
    await fit.click()
    await expect.poll(async () => (await f.bounds()).height).toBe(fitted.height)
    // Fit mode follows column changes; changing only width does not turn it off.
    await f.resize(700, fitted.height)
    await expect.poll(async () => (await f.bounds()).height).toBeLessThan(fitted.height)
    const wide = await f.bounds()
    await f.mini.getByRole('button', { name: '选择显示项', exact: true }).click()
    await f.mini.getByRole('button', { name: '完成', exact: true }).click()
    await expect.poll(f.bounds).toEqual(wide)
    expect(await clippedReadouts(f.mini)).toEqual([])
    await expect(f.mini.evaluate(() => window.miniMonitor!.fitContent(Number.NaN))).rejects.toThrow('内容高度无效')
    await f.mini.screenshot({ path: path.join(f.directory, 'fit-content.png') })
    await Promise.all([
      f.mini.waitForEvent('close'),
      f.mini.getByRole('button', { name: '关闭小窗', exact: true }).click(),
    ])
    const saved = JSON.parse(fs.readFileSync(path.join(f.data, 'desktop/mini-monitor.json'), 'utf8'))
    expect(saved).toMatchObject({ fitContent: true, bounds: { width: wide.width, height: wide.height } })
    const reopened = await f.open()
    await expect.poll(f.bounds).toEqual(wide)
    await expect(reopened.getByRole('button', { name: '适应内容', exact: true })).toBeVisible()
    await f.resize(wide.width, 190)
    await expect.poll(async () => (await f.bounds()).height).toBe(190)
    await Promise.all([
      reopened.waitForEvent('close'),
      reopened.getByRole('button', { name: '关闭小窗', exact: true }).click(),
    ])
    expect(JSON.parse(fs.readFileSync(path.join(f.data, 'desktop/mini-monitor.json'), 'utf8')))
      .toMatchObject({ fitContent: false, bounds: { width: wide.width, height: 190 } })
    const manual = await f.open()
    await expect.poll(async () => (await f.bounds()).height).toBe(190)
    await expect(manual.getByRole('button', { name: '适应内容', exact: true })).toBeVisible()
  } finally { await f.close() }
})

test('紧凑布局保留拖排、键盘排序、显示项选择和空状态入口', async () => {
  const f = await fixture()
  try {
    await f.resize(700, 400)
    const ids = () => f.mini.locator('[data-mini-metric]').evaluateAll(nodes => nodes.map(node => (node as HTMLElement).dataset.miniMetric))
    const savedIds = () => f.mini.evaluate(key => JSON.parse(localStorage.getItem(key)!), MINI_PREFERENCE_KEY)
    await f.mini.locator('[data-mini-metric="average"]').press('Alt+ArrowLeft')
    const keyboardOrder = ['duration', 'average', 'current', 'context', 'prompt', 'generated']
    await expect.poll(ids).toEqual(keyboardOrder)
    await expect.poll(savedIds).toEqual(keyboardOrder)
    const from = await f.mini.locator('[data-mini-metric="generated"]').boundingBox()
    const target = await f.mini.locator('[data-mini-metric="duration"]').boundingBox()
    await f.mini.mouse.move(from!.x + 25, from!.y + 25); await f.mini.mouse.down()
    await f.mini.mouse.move(target!.x + 10, target!.y + 25)
    await expect(f.mini.locator('.mini-drag-preview')).toBeVisible()
    const dragOrder = ['generated', ...keyboardOrder.slice(0, -1)]
    await expect.poll(ids).toEqual(dragOrder)
    expect(await savedIds()).toEqual(keyboardOrder)
    await f.mini.mouse.up()
    await expect.poll(savedIds).toEqual(dragOrder)
    await f.mini.getByRole('button', { name: '选择显示项', exact: true }).click()
    await f.mini.getByLabel('上下文', { exact: true }).uncheck()
    await f.mini.getByLabel('CPU 使用率', { exact: true }).check()
    await f.mini.getByRole('button', { name: '完成', exact: true }).click()
    await expect(f.mini.locator('[data-mini-metric="context"]')).toHaveCount(0)
    await expect(f.mini.locator('[data-mini-metric="cpu"]')).toBeVisible()
    expect((await f.bounds()).height).toBe(400)
    await f.mini.getByRole('button', { name: '选择显示项', exact: true }).click()
    for (const label of ['输出 Token', '请求耗时', '平均速度', '当前速度', '输入 Token', 'CPU 使用率']) {
      await f.mini.getByLabel(label, { exact: true }).uncheck()
    }
    await f.mini.getByRole('button', { name: '完成', exact: true }).click()
    await expect(f.mini.locator('.mini-empty')).toBeVisible()
    await f.mini.getByRole('button', { name: '适应内容', exact: true }).click()
    await expect.poll(async () => (await f.bounds()).height).toBeLessThan(150)
    await f.mini.locator('.mini-empty').click()
    await f.mini.getByLabel('上下文', { exact: true }).check()
    await f.mini.getByRole('button', { name: '完成', exact: true }).click()
    await expect(f.mini.locator('[data-mini-metric]')).toHaveCount(1)
    await expect(f.mini.locator('[data-mini-metric="context"]')).toBeVisible()
    expect((await f.bounds()).width).toBe(700)
  } finally { await f.close() }
})
