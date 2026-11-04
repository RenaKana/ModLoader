import { pinNavigation } from './navigation'
import { test, expect, _electron as electron } from '@playwright/test'
import * as fs from 'node:fs'
import * as path from 'node:path'

test('F11 切换原生全屏并还原窗口，输入框内同样生效', async () => {
  test.setTimeout(60_000)
  const root = path.resolve(__dirname, '..')
  const data = fs.mkdtempSync(path.join(root, 'work', 'ui-fullscreen-'))
  const application = await electron.launch({
    ...(process.env.MODLOADER_TEST_EXECUTABLE ? { executablePath: process.env.MODLOADER_TEST_EXECUTABLE, args: [] } : { args: [root] }),
    env: { ...process.env, MODLOADER_DATA_DIR: data }, timeout: 60_000,
  })
  const page = await application.firstWindow()
  const fullscreen = () => application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].isFullScreen())
  const pressF11 = () => application.evaluate(({ BrowserWindow }) => {
    const window = BrowserWindow.getAllWindows()[0]
    window.webContents.sendInputEvent({ type: 'keyDown', keyCode: 'F11' })
    window.webContents.sendInputEvent({ type: 'keyUp', keyCode: 'F11' })
  })
  try {
    await pinNavigation(page)
    await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 45_000 })
    await application.evaluate(({ BrowserWindow }) => { const window = BrowserWindow.getAllWindows()[0]; window.setSize(1200, 800); window.focus() })
    const initialBounds = await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].getBounds())
    await pressF11()
    await expect.poll(fullscreen).toBe(true)
    await pressF11()
    await expect.poll(fullscreen).toBe(false)
    await expect.poll(() => application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].getBounds())).toEqual(initialBounds)

    await page.getByRole('button', { name: '设置', exact: true }).click()
    const port = page.getByLabel(/^API 端口/)
    await port.focus()
    await pressF11()
    await expect.poll(fullscreen).toBe(true)
    await expect(port).toBeFocused()
    const image = await application.evaluate(async ({ BrowserWindow }) => (await BrowserWindow.getAllWindows()[0].webContents.capturePage()).toPNG().toString('base64'))
    fs.writeFileSync(path.join(data, 'fullscreen.png'), Buffer.from(image, 'base64'))
    await pressF11()
    await expect.poll(fullscreen).toBe(false)
    await expect(port).toBeFocused()
    console.log(JSON.stringify({ screenshot: path.join(data, 'fullscreen.png') }))
  } finally { await application.close() }
})
