import { pinNavigation } from './navigation'
import { test, expect, _electron as electron, type Locator } from '@playwright/test'
import * as path from 'node:path'
import * as fs from 'node:fs'

const pickerOnly = process.env.MODLOADER_UI_SCROLLBARS_PICKER_ONLY === '1'

test(pickerOnly ? 'base-select 自定义滚动条：可见滑块拖拽与悬浮' : '自定义滚动条：页面、模型列表拖拽、长弹窗和 base-select 下拉', async () => {
  test.setTimeout(90_000)
  const root = path.resolve(__dirname, '..')
  const directory = path.join(root, 'work', 'ui-scrollbars')
  const screenshots = path.join(directory, 'screenshots')
  fs.mkdirSync(screenshots, { recursive: true })
  const application = await electron.launch({ args: [root], env: { ...process.env, MODLOADER_DATA_DIR: path.join(directory, 'data') }, timeout: 60_000 })
  const page = await application.firstWindow()
  const errors: string[] = []
  const externalRequests: string[] = []
  const writes: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/*', async route => {
    const request = route.request()
    const url = new URL(request.url())
    if (['http:', 'https:'].includes(url.protocol) && !['127.0.0.1', 'localhost'].includes(url.hostname)) {
      externalRequests.push(url.origin)
      await route.abort()
    } else if (url.pathname.startsWith('/api/') && request.method() !== 'GET') {
      writes.push(`${request.method()} ${url.pathname}`)
      await route.abort()
    } else await route.continue()
  })
  async function nativeScreenshot(name: string) {
    await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))))
    const png = await application.evaluate(async ({ BrowserWindow }) => (await BrowserWindow.getAllWindows()[0].webContents.capturePage()).toPNG().toString('base64'))
    fs.writeFileSync(path.join(screenshots, name), Buffer.from(png, 'base64'))
  }
  async function checkStyle(target: Locator) {
    const pointer = await page.evaluate(() => ({ x: innerWidth / 2, y: 20 }))
    await page.mouse.move(pointer.x, pointer.y)
    const style = await target.evaluate(element => {
      const bar = getComputedStyle(element, '::-webkit-scrollbar')
      const thumb = getComputedStyle(element, '::-webkit-scrollbar-thumb')
      const track = getComputedStyle(element, '::-webkit-scrollbar-track')
      const arrows = getComputedStyle(element, '::-webkit-scrollbar-button')
      return { width: bar.width, thumbColor: thumb.backgroundColor, radius: thumb.borderRadius, trackColor: track.backgroundColor, arrows: arrows.display }
    })
    expect(style.width).toBe('10px')
    // Computed scrollbar pseudo-styles can inherit their host's hover state.
    expect(['rgb(189, 201, 220)', 'rgb(149, 169, 200)', 'rgb(116, 143, 188)']).toContain(style.thumbColor)
    expect(style.radius).toBe('10px')
    expect(style.trackColor).toBe('rgb(244, 246, 250)')
    expect(style.arrows).toBe('none')
    return style
  }

  try {
    await pinNavigation(page)
    await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
    await application.evaluate(({ BrowserWindow }) => { const window = BrowserWindow.getAllWindows()[0]; window.setSize(1440, 960); window.webContents.setZoomFactor(1) })
    await page.route('**/api/state', async route => {
      const response = await route.fetch()
      const state = await response.json()
      await route.fulfill({ response, json: { ...state,
        settings: { ...state.settings, font_scale: 1 },
        models: Array.from({ length: 80 }, (_, index) => ({ id: `scrollbar-model-${index}`, name: `滚动条 fixture ${index + 1}`, path: path.join(directory, `fixture-${index}.gguf`), format: 'gguf', size_bytes: 0, files: [], modalities: ['text'], metadata: {}, source: {}, managed: false, warnings: [] })),
        profiles: Array.from({ length: 12 }, (_, index) => ({ id: `scrollbar-profile-${index}`, model_id: 'scrollbar-model-0', name: `长页面配置 ${index + 1}`, backend: 'llama_cpp', runtime_id: 'scrollbar-runtime-0', alias: `scrollbar-fixture-${index + 1}`, parameters: index ? {} : Object.fromEntries(Array.from({ length: 32 }, (_, key) => [`scrollbar_fixture_${key}`, `长弹窗保留参数 ${key + 1}`])), generation: {}, extra_args: [], revision: 1 })),
        runtimes: Array.from({ length: 80 }, (_, index) => ({ id: `scrollbar-runtime-${index}`, name: `下拉滚动环境 fixture ${index + 1}`, version: 'UI', backend: 'llama_cpp', path: path.join(directory, `virtual-runtime-${index}`), managed: false, metadata: {} })),
        instances: [], downloads: [], installations: [],
      } })
    })
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await expect(page.getByRole('heading', { name: '滚动条 fixture 1', exact: true })).toBeVisible()
    await page.evaluate(() => document.fonts.ready)

    if (!pickerOnly) {
      await test.step('模型列表滑块可以悬浮和鼠标拖拽', async () => {
      const list = page.getByRole('region', { name: '模型列表', exact: true })
      await expect.poll(() => list.evaluate(element => element.scrollHeight > element.clientHeight)).toBe(true)
      await checkStyle(list)
      const geometry = await list.evaluate(element => {
        const box = element.getBoundingClientRect()
        const thumbHeight = Math.max(28, element.clientHeight ** 2 / element.scrollHeight)
        return { x: box.right - 5, y: box.top + thumbHeight / 2, dragDistance: Math.min(110, element.clientHeight / 2), scrollTop: element.scrollTop }
      })
      await page.mouse.move(geometry.x - 30, geometry.y + 50)
      await nativeScreenshot('model-list-normal.png')
      await page.mouse.move(geometry.x, geometry.y)
      // Nested scrollbar hover colors are verified in the native capture, not Chromium's computed pseudo-style.
      await nativeScreenshot('model-list-hover.png')
      await page.mouse.down()
      try { await page.mouse.move(geometry.x, geometry.y + geometry.dragDistance, { steps: 10 }) }
      finally { await page.mouse.up() }
      await expect.poll(() => list.evaluate(element => element.scrollTop)).toBeGreaterThan(geometry.scrollTop + 100)
      console.log(JSON.stringify({ surface: 'model-list', dragScrollTop: await list.evaluate(element => element.scrollTop) }))
      await list.evaluate(element => { element.scrollTop = 0 })
      })

      await test.step('页面滚动条使用同一外观并响应鼠标滚轮', async () => {
      await checkStyle(page.locator('html'))
      const detail = await page.locator('.model-detail').boundingBox()
      expect(detail).not.toBeNull()
      await page.mouse.move(detail!.x + detail!.width / 2, 550)
      await page.mouse.wheel(0, 500)
      await expect.poll(() => page.evaluate(() => scrollY)).toBeGreaterThan(0)
      await nativeScreenshot('page-scrollbar.png')
      await page.evaluate(() => window.scrollTo(0, 0))
      })
    }

    await page.locator('.profile-card').first().getByRole('button', { name: '编辑配置', exact: true }).click()
    const dialog = page.getByRole('dialog', { name: '编辑加载配置', exact: true })
    const body = dialog.locator('.modal-body')
    if (!pickerOnly) await test.step('长弹窗内容使用自定义滚动条', async () => {
      await expect(dialog).toBeVisible()
      await expect.poll(() => body.evaluate(element => element.scrollHeight > element.clientHeight)).toBe(true)
      await checkStyle(body)
      const box = await body.boundingBox()
      expect(box).not.toBeNull()
      await page.mouse.move(box!.x + box!.width / 2, box!.y + box!.height / 2)
      await page.mouse.wheel(0, 500)
      await expect.poll(() => body.evaluate(element => element.scrollTop)).toBeGreaterThan(0)
      await nativeScreenshot('modal-scrollbar.png')
      await body.evaluate(element => { element.scrollTop = 0 })
    })

    await test.step('base-select 下拉的实际 DOM 滚动条可以悬浮和拖拽', async () => {
      await expect(dialog.getByRole('combobox', { name: '运行后端', exact: true }).locator('option').first()).toBeHidden()
      const select = dialog.getByRole('combobox', { name: '运行环境', exact: true })
      await select.click()
      await expect.poll(() => select.evaluate(element => element.matches(':open'))).toBe(true)
      expect(await select.evaluate(element => getComputedStyle(element, '::picker(select)').appearance)).toBe('base-select')
      const group = select.locator('.select-options')
      await expect.poll(() => group.evaluate(element => element.scrollHeight > element.clientHeight)).toBe(true)
      await checkStyle(group)
      const geometry = await group.evaluate(element => {
        const box = element.getBoundingClientRect()
        const thumbHeight = Math.max(28, element.clientHeight ** 2 / element.scrollHeight)
        return { x: box.right - 5, y: box.top + thumbHeight / 2, scrollTop: element.scrollTop }
      })
      await page.mouse.move(geometry.x - 30, geometry.y + 50)
      await nativeScreenshot('picker-normal.png')
      await page.mouse.move(geometry.x, geometry.y)
      await nativeScreenshot('picker-hover.png')
      await page.mouse.down()
      try { await page.mouse.move(geometry.x, geometry.y + 90, { steps: 10 }) }
      finally { await page.mouse.up() }
      await expect.poll(() => group.evaluate(element => element.scrollTop)).toBeGreaterThan(geometry.scrollTop + 100)
      console.log(JSON.stringify({ surface: 'picker', dragScrollTop: await group.evaluate(element => element.scrollTop) }))
      await nativeScreenshot('picker-dragged.png')
      await group.evaluate(element => { element.scrollTop = 0 })
      const first = select.locator('option').first()
      const before = await first.evaluate(element => element.getBoundingClientRect().top)
      await select.locator('option').nth(3).hover()
      await page.mouse.wheel(0, 500)
      await expect.poll(() => first.evaluate(element => element.getBoundingClientRect().top)).toBeLessThan(before - 100)
      await nativeScreenshot('picker-scrollbar.png')
      if (!pickerOnly) {
        await page.keyboard.press('End')
        await expect.poll(() => select.locator('option').last().evaluate(element => { const box = element.getBoundingClientRect(); return box.top >= 0 && box.bottom <= innerHeight })).toBe(true)
      }
      await page.keyboard.press('Escape')
      await expect.poll(() => select.evaluate(element => element.matches(':open'))).toBe(false)
      await expect(select).toHaveValue('scrollbar-runtime-0')
    })
    await dialog.getByRole('button', { name: '取消', exact: true }).click()
    expect(externalRequests).toEqual([])
    expect(writes).toEqual([])
    expect(errors).toEqual([])
  } finally {
    await application.close()
  }
})
