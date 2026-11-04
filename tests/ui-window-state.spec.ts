import { test, expect, _electron as electron, type ElectronApplication } from '@playwright/test'
import * as fs from 'node:fs'
import * as path from 'node:path'

const root = path.resolve(__dirname, '..')

function createData(appearance?: unknown) {
  const data = fs.mkdtempSync(path.join(root, 'work', 'ui-window-state-'))
  if (appearance !== undefined) {
    fs.mkdirSync(path.join(data, 'desktop'), { recursive: true })
    fs.writeFileSync(path.join(data, 'desktop', 'appearance.json'), JSON.stringify(appearance))
  }
  return data
}

function savedAppearance(data: string) {
  return JSON.parse(fs.readFileSync(path.join(data, 'desktop', 'appearance.json'), 'utf8'))
}

async function launch(data: string) {
  const application = await electron.launch({
    ...(process.env.MODLOADER_TEST_EXECUTABLE ? { executablePath: process.env.MODLOADER_TEST_EXECUTABLE, args: [] } : { args: [root] }),
    env: { ...process.env, MODLOADER_DATA_DIR: data, MODLOADER_SENSOR_AUTO_START: '0' },
    timeout: 60_000,
  })
  try {
    const page = await application.firstWindow()
    await expect(page.locator('.app-shell')).toBeVisible({ timeout: 45_000 })
    await expect.poll(async () => (await nativeState(application)).visible).toBe(true)
    return application
  } catch (error) {
    await application.close()
    throw error
  }
}

function nativeState(application: ElectronApplication) {
  return application.evaluate(({ BrowserWindow, screen }) => {
    const win = BrowserWindow.getAllWindows()[0]
    return {
      bounds: win.getBounds(), normalBounds: win.getNormalBounds(),
      maximized: win.isMaximized(), minimized: win.isMinimized(), fullScreen: win.isFullScreen(),
      visible: win.isVisible(), zoomFactor: win.webContents.getZoomFactor(),
      workArea: screen.getDisplayMatching(win.getNormalBounds()).workArea,
    }
  })
}

async function setNormalBounds(application: ElectronApplication) {
  const target = await application.evaluate(({ BrowserWindow, screen }) => {
    const win = BrowserWindow.getAllWindows()[0]
    const area = screen.getDisplayMatching(win.getBounds()).workArea
    const width = Math.min(1180, area.width), height = Math.min(780, area.height)
    const bounds = {
      x: area.x + Math.min(48, area.width - width),
      y: area.y + Math.min(56, area.height - height), width, height,
    }
    win.setBounds(bounds)
    return bounds
  })
  await expect.poll(async () => (await nativeState(application)).bounds).toEqual(target)
  return target
}

test('普通窗口关闭到托盘、重新打开和关闭退出后保留位置、大小及缩放', async ({}, testInfo) => {
  test.setTimeout(120_000)
  // Older versions only saved zoomFactor in appearance.json.
  const data = createData({ zoomFactor: 1.2 })
  let application: ElectronApplication | undefined
  try {
    application = await launch(data)
    expect((await nativeState(application)).zoomFactor).toBeCloseTo(1.2)
    const bounds = await setNormalBounds(application)
    await application.evaluate(({ BrowserWindow }) => {
      const win = BrowserWindow.getAllWindows()[0]
      win.webContents.setZoomFactor(1.1)
      win.close()
    })
    await expect.poll(async () => (await nativeState(application!)).visible).toBe(false)
    expect(savedAppearance(data)).toMatchObject({ bounds, zoomFactor: 1.1, maximized: false, fullScreen: false })

    // Use the application's existing reopen entry point, not a replacement test handler.
    await application.evaluate(({ app }) => { app.emit('second-instance', {}, [], '') })
    await expect.poll(async () => (await nativeState(application!)).visible).toBe(true)
    expect(await nativeState(application)).toMatchObject({ bounds, zoomFactor: 1.1 })

    const page = await application.firstWindow()
    await page.evaluate(async () => {
      const connection = await window.modloader!.connection()
      const response = await fetch(connection.baseUrl + '/api/settings', {
        method: 'PATCH', headers: { Authorization: 'Bearer ' + connection.token, 'Content-Type': 'application/json' },
        body: JSON.stringify({ close_to_tray: false }),
      })
      if (!response.ok) throw new Error('无法关闭测试实例的关闭到托盘设置')
    })
    const closed = application.waitForEvent('close')
    await application.evaluate(({ BrowserWindow }) => { BrowserWindow.getAllWindows()[0].close() })
    await closed
    application = undefined
    expect(savedAppearance(data)).toMatchObject({ bounds, zoomFactor: 1.1, maximized: false })

    application = await launch(data)
    expect(await nativeState(application)).toMatchObject({ bounds, maximized: false, minimized: false, fullScreen: false, zoomFactor: 1.1 })
    await (await application.firstWindow()).screenshot({ path: testInfo.outputPath('restored-normal.png') })
  } finally {
    await application?.close()
  }
})

test('最大化、最小化退出和全屏跨重启恢复，并保留普通窗口尺寸', async ({}, testInfo) => {
  test.setTimeout(180_000)
  const data = createData()
  let application: ElectronApplication | undefined
  try {
    application = await launch(data)
    const bounds = await setNormalBounds(application)
    await application.evaluate(({ BrowserWindow }) => { BrowserWindow.getAllWindows()[0].maximize() })
    await expect.poll(async () => (await nativeState(application!)).maximized).toBe(true)
    await application.close()
    application = undefined
    expect(savedAppearance(data)).toMatchObject({ bounds, maximized: true, fullScreen: false })

    application = await launch(data)
    expect(await nativeState(application)).toMatchObject({ normalBounds: bounds, maximized: true, minimized: false })
    await (await application.firstWindow()).screenshot({ path: testInfo.outputPath('restored-maximized.png') })
    await application.evaluate(({ BrowserWindow }) => { BrowserWindow.getAllWindows()[0].unmaximize() })
    await expect.poll(async () => (await nativeState(application!)).bounds).toEqual(bounds)
    await application.evaluate(({ BrowserWindow }) => { BrowserWindow.getAllWindows()[0].maximize() })
    await expect.poll(async () => (await nativeState(application!)).maximized).toBe(true)
    await application.evaluate(({ BrowserWindow }) => { BrowserWindow.getAllWindows()[0].minimize() })
    await expect.poll(async () => (await nativeState(application!)).minimized).toBe(true)
    await application.close()
    application = undefined
    expect(savedAppearance(data)).toMatchObject({ bounds, maximized: true, fullScreen: false })

    application = await launch(data)
    expect(await nativeState(application)).toMatchObject({ normalBounds: bounds, maximized: true, minimized: false })
    await application.evaluate(({ BrowserWindow }) => { BrowserWindow.getAllWindows()[0].setFullScreen(true) })
    await expect.poll(async () => (await nativeState(application!)).fullScreen).toBe(true)
    await application.close()
    application = undefined
    expect(savedAppearance(data)).toMatchObject({ bounds, maximized: true, fullScreen: true })

    application = await launch(data)
    expect(await nativeState(application)).toMatchObject({ fullScreen: true, minimized: false })
    expect(savedAppearance(data)).toMatchObject({ bounds, maximized: true, fullScreen: true })
    await application.evaluate(({ BrowserWindow }) => { BrowserWindow.getAllWindows()[0].setFullScreen(false) })
    await expect.poll(async () => (await nativeState(application!)).fullScreen).toBe(false)
    await expect.poll(async () => (await nativeState(application!)).maximized).toBe(true)
    await application.evaluate(({ BrowserWindow }) => { BrowserWindow.getAllWindows()[0].unmaximize() })
    await expect.poll(async () => (await nativeState(application!)).bounds).toEqual(bounds)
  } finally {
    await application?.close()
  }
})

test('旧显示器位置及过大尺寸恢复到当前显示器工作区', async () => {
  test.setTimeout(60_000)
  const data = createData({ bounds: { x: 100000, y: 100000, width: 10000, height: 10000 }, zoomFactor: 1 })
  const application = await launch(data)
  try {
    const { bounds, workArea } = await nativeState(application)
    expect(bounds.width).toBeLessThanOrEqual(workArea.width)
    expect(bounds.height).toBeLessThanOrEqual(workArea.height)
    expect(bounds.x).toBeGreaterThanOrEqual(workArea.x)
    expect(bounds.y).toBeGreaterThanOrEqual(workArea.y)
    expect(bounds.x + bounds.width).toBeLessThanOrEqual(workArea.x + workArea.width)
    expect(bounds.y + bounds.height).toBeLessThanOrEqual(workArea.y + workArea.height)
    await expect.poll(() => savedAppearance(data).bounds).toEqual(bounds)
  } finally {
    await application.close()
  }
})

test('仍在屏幕内的自定义位置和超单屏宽度不被恢复逻辑裁切', async () => {
  test.setTimeout(90_000)
  const data = createData()
  let application: ElectronApplication | undefined
  try {
    application = await launch(data)
    const bounds = await application.evaluate(({ BrowserWindow, screen }) => {
      const win = BrowserWindow.getAllWindows()[0]
      const area = screen.getDisplayMatching(win.getBounds()).workArea
      win.setBounds({ x: area.x - 80, y: area.y + 48, width: area.width + 160, height: 780 })
      return win.getBounds()
    })
    await application.close()
    application = undefined
    expect(savedAppearance(data).bounds).toEqual(bounds)
    application = await launch(data)
    expect((await nativeState(application)).bounds).toEqual(bounds)
  } finally {
    await application?.close()
  }
})
