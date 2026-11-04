import { test, expect, _electron as electron, type ElectronApplication, type Page, type Locator } from '@playwright/test'
import { createServer, type ServerResponse } from 'node:http'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { pinNavigation } from './navigation'
import { disableWebSearch } from './chat-controls'

test('聊天图片：用户附件、流式与历史多图、视图操作和原图保存', async () => {
  test.setTimeout(150_000)
  const root = path.resolve(__dirname, '..')
  const uiOnly = process.env.MODLOADER_TEST_IMAGE_UI_ONLY === '1'
  const saveStep = uiOnly ? test.step.skip : test.step
  const artifacts = process.env.MODLOADER_TEST_ARTIFACT_DIR || path.join(root, '.artifacts', 'codex', 'ui-chat-image-viewer')
  fs.mkdirSync(artifacts, { recursive: true })
  const directory = fs.mkdtempSync(path.join(artifacts, 'ui-chat-image-viewer-'))
  const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=', 'base64')
  const portrait = '<svg xmlns="http://www.w3.org/2000/svg" width="600" height="1400"><rect width="600" height="1400" fill="#c8d8e7"/><circle cx="300" cy="420" r="180" fill="#69829b"/><path d="M80 1200L300 700L520 1200Z" fill="#e9a980"/></svg>'
  const landscape = '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900"><rect width="1600" height="900" fill="#c9d8df"/><circle cx="1210" cy="260" r="125" fill="#f0d2a1"/><path d="M0 720L390 220L900 900H0" fill="#536e7a"/><path d="M450 900L1040 370L1600 900Z" fill="#8ba5ac"/><path d="M0 800Q500 650 980 800T1600 750V900H0" fill="#334d5d"/><text x="76" y="112" fill="#334d5d" font-family="system-ui" font-size="38" letter-spacing="8">MODLOADER / VISION</text></svg>'
  // Two original GIF frames, with different palette indices and a loop extension.
  const gif = Buffer.from('47494638396101000100800000ff00000000ff21ff0b4e45545343415045322e30030100000021f904000a0000002c000000000100010000020244010021f904000a0000002c00000000010001000002024c01003b', 'hex')
  let response: ServerResponse | undefined
  let failImage = false
  let imageRequests = 0
  const errors: string[] = []
  let application: ElectronApplication | undefined
  let page: Page | undefined
  const server = createServer(async (request, result) => {
    const pathname = new URL(request.url || '/', 'http://127.0.0.1').pathname
    if (pathname === '/landscape.svg') {
      imageRequests++
      result.setHeader('Cache-Control', 'no-store')
      result.setHeader('Content-Type', 'image/svg+xml')
      result.writeHead(failImage ? 404 : 200)
      result.end(failImage ? '' : landscape)
      return
    }
    if (pathname === '/animation.gif') { result.setHeader('Content-Type', 'image/gif'); result.end(gif); return }
    if (pathname === '/portrait.svg') { result.setHeader('Content-Type', 'image/svg+xml'); result.end(portrait); return }
    if (pathname === '/redirect-file.svg') { result.writeHead(302, { Location: 'file:///C:/Windows/win.ini' }); result.end(); return }
    if (pathname === '/redirect-scene.svg') { result.writeHead(302, { Location: '/landscape.svg' }); result.end(); return }
    if (pathname !== '/api/chat') { result.writeHead(404); result.end(); return }
    result.setHeader('Access-Control-Allow-Origin', '*')
    result.setHeader('Access-Control-Allow-Headers', '*')
    if (request.method === 'OPTIONS') { result.end(); return }
    for await (const _chunk of request) {}
    result.setHeader('Content-Type', 'text/event-stream')
    result.flushHeaders()
    response = result
  })
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const origin = `http://127.0.0.1:${(server.address() as { port: number }).port}`
  const event = (value: unknown) => response!.write(`data: ${JSON.stringify(value)}\n\n`)
  const delta = (content: string) => event({ choices: [{ delta: { content } }] })
  async function loaded(image: Locator) {
    await expect(image).toBeVisible()
    await expect.poll(() => image.evaluate(element => (element as HTMLImageElement).naturalWidth > 0)).toBe(true)
  }
  async function escape() {
    await application!.evaluate(({ BrowserWindow }) => {
      const contents = BrowserWindow.getAllWindows()[0].webContents
      contents.sendInputEvent({ type: 'keyDown', keyCode: 'Escape' })
      contents.sendInputEvent({ type: 'keyUp', keyCode: 'Escape' })
    })
    await expect(page!.getByRole('dialog', { name: '图片查看器', exact: true })).toHaveCount(0)
  }
  async function dialogState(output: string, canceled = false) {
    await application!.evaluate((_electron, { output, canceled }) => {
      Object.assign((globalThis as any).imageSaveTest, { output, canceled })
    }, { output, canceled })
  }
  async function readable(bytes: Buffer, mime: string, expected: number[]) {
    expect(await page!.evaluate(({ source, mime }) => new Promise<number[]>((resolve, reject) => {
      const image = new Image()
      image.onload = () => resolve([image.naturalWidth, image.naturalHeight])
      image.onerror = () => reject(new Error('保存的原图无法解码'))
      image.src = `data:${mime};base64,${source}`
    }), { source: bytes.toString('base64'), mime })).toEqual(expected)
  }
  try {
    const executablePath = process.env.MODLOADER_TEST_EXECUTABLE
    application = await electron.launch({ ...(executablePath ? { executablePath, args: [] } : { args: [root] }), env: { ...process.env, MODLOADER_DATA_DIR: directory, MODLOADER_SENSOR_AUTO_START: '0' }, timeout: 60_000 })
    page = await application.firstWindow()
    page.on('pageerror', error => errors.push(error.message))
    await pinNavigation(page)
    await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
    await page.route('**/api/state', async route => {
      const original = await route.fetch()
      const state = await original.json()
      await route.fulfill({ response: original, json: { ...state, instances: [{ id: 'image-viewer-fixture', model_id: 'image-model', profile_id: 'image-profile', alias: '图片预览模型', backend: 'llama_cpp', state: 'ready', capabilities: { input: ['text', 'image'] }, active_requests: 0 }] } })
    })
    await page.route('**/api/chat', route => route.continue({ url: `${origin}/api/chat` }))
    await page.route('**/api/instances/image-viewer-fixture/telemetry', route => route.fulfill({ json: { groups: [], unavailable: [], live_generation: null } }))
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '测试对话', exact: true }).click()
    await disableWebSearch(page)
    await application.evaluate(({ dialog, shell }) => {
      const state = { output: '', canceled: false, calls: 0, defaultPath: '', save: dialog.showSaveDialog, open: shell.openExternal, urls: [] as string[] }
      ;(globalThis as any).imageSaveTest = state
      dialog.showSaveDialog = async (_owner: unknown, options: any) => {
        state.calls++; state.defaultPath = options.defaultPath
        return { canceled: state.canceled, filePath: state.canceled ? undefined : state.output }
      }
      shell.openExternal = async url => { state.urls.push(url) }
    })
    const viewer = page.getByRole('dialog', { name: '图片查看器', exact: true })
    const displayed = viewer.locator('.image-viewer-image')
    const uploads = [{ name: '像素.png', mimeType: 'image/png', buffer: png }, { name: '竖图.svg', mimeType: 'image/svg+xml', buffer: Buffer.from(portrait) }]
    await test.step('草稿图片可查看、多图切换且保留移除操作', async () => {
      await page!.locator('input[type=file]').setInputFiles(uploads)
      const attachments = page!.locator('.attachment-list')
      await expect(attachments.locator('img')).toHaveCount(2)
      await attachments.getByRole('button', { name: '查看图片：像素.png', exact: true }).click()
      await loaded(displayed)
      await expect(viewer.getByLabel('图片位置', { exact: true })).toHaveText('1 / 2')
      await page!.keyboard.press('ArrowRight')
      await loaded(displayed)
      await expect(displayed).toHaveAttribute('alt', '竖图.svg')
      await escape()
      await expect(attachments.getByRole('button', { name: '查看图片：像素.png', exact: true })).toBeFocused()
      await attachments.getByRole('button', { name: '移除 竖图.svg', exact: true }).click()
      await expect(attachments.locator('img')).toHaveCount(1)
      await page!.locator('input[type=file]').setInputFiles([uploads[1]])
      await expect(attachments.locator('img')).toHaveCount(2)
    })
    await page.getByRole('textbox', { name: '对话消息', exact: true, includeHidden: true }).fill('请查看这两张图。')
    await page.getByRole('button', { name: '发送消息', exact: true }).click()
    await expect.poll(() => Boolean(response)).toBe(true)
    const user = page.locator('.chat-message.user').last()
    const assistant = page.locator('.chat-message.assistant').last()
    await saveStep('用户 data 图片经真实 IPC 保存原始字节，取消和写入失败反馈正确', async () => {
      const trigger = user.getByRole('button', { name: '查看图片：像素.png', exact: true })
      await trigger.focus()
      await page!.keyboard.press('Enter')
      await loaded(displayed)
      const output = path.join(directory, '保存-用户图片.png')
      await dialogState(output)
      await viewer.getByRole('button', { name: '保存图片', exact: true }).click()
      await expect(viewer.locator('.image-viewer-feedback[role=status]')).toHaveText('图片已保存')
      const saved = fs.readFileSync(output)
      expect(saved.equals(png)).toBe(true)
      await readable(saved, 'image/png', [1, 1])
      expect(await application!.evaluate(() => (globalThis as any).imageSaveTest.defaultPath)).toBe('像素.png')
      const canceled = path.join(directory, '不应保存.png')
      await dialogState(canceled, true)
      await viewer.getByRole('button', { name: '保存图片', exact: true }).click()
      await expect(viewer.getByRole('button', { name: '保存图片', exact: true })).toBeEnabled()
      await expect(viewer.locator('.image-viewer-feedback[role=status]')).toHaveCount(0)
      await expect(viewer.getByRole('alert')).toHaveCount(0)
      expect(fs.existsSync(canceled)).toBe(false)
      await dialogState(path.join(directory, '不存在的目录', '失败.png'))
      await viewer.getByRole('button', { name: '保存图片', exact: true }).click()
      await expect(viewer.getByRole('alert')).toContainText('ENOENT')
      await expect(viewer.getByRole('button', { name: '保存图片', exact: true })).toBeEnabled()
      await escape()
      await expect(trigger).toBeFocused()
    })
    delta(`![山景](${origin}/landscape.svg)\n\n![竖图](${origin}/portrait.svg)\n\n![动图](${origin}/animation.gif)\n\n![坏图](${origin}/missing.png)\n\n[普通外链](https://example.com/image-link)\n\n\`\`\`text\n![代码图片](${origin}/landscape.svg)\n\`\`\``)
    await test.step('AI HTTP 图片全屏、锚点缩放、拖动、旋转、适配和复位', async () => {
      const trigger = assistant.getByRole('button', { name: '查看图片：山景', exact: true })
      await loaded(trigger)
      await expect(assistant.locator('.message-image-error')).toContainText('图片加载失败')
      await expect(assistant.getByRole('button', { name: '查看图片：代码图片', exact: true })).toHaveCount(0)
      await trigger.click()
      await loaded(displayed)
      await expect(viewer).toBeFocused()
      await expect(viewer.getByRole('button', { name: '关闭图片查看器', exact: true })).toHaveCount(0)
      await expect(viewer.getByLabel('图片位置', { exact: true })).toHaveText('1 / 3')
      const fit = await displayed.boundingBox()
      const stage = await viewer.locator('.image-viewer-stage').boundingBox()
      expect(fit!.width).toBeLessThanOrEqual(stage!.width)
      expect(fit!.height).toBeLessThanOrEqual(stage!.height)
      const floats = await viewer.evaluate(element => ({ top: element.querySelector('.image-viewer-top')!.getBoundingClientRect().bottom, bottom: element.querySelector('.image-viewer-toolbar')!.getBoundingClientRect().top, backdrop: getComputedStyle(element, '::backdrop').backgroundColor, background: getComputedStyle(element).backgroundColor }))
      expect(fit!.y).toBeGreaterThan(floats.top)
      expect(fit!.y + fit!.height).toBeLessThan(floats.bottom)
      expect(floats.background).toBe('rgba(0, 0, 0, 0)')
      expect(floats.backdrop).toMatch(/^rgba\(/)
      await page!.screenshot({ path: path.join(directory, 'image-viewer-landscape.png') })
      await viewer.getByRole('button', { name: '原始比例', exact: true }).click()
      await expect(viewer.getByLabel('图片缩放比例', { exact: true })).toHaveText('100%')
      const before = (await displayed.boundingBox())!
      const point = { x: stage!.x + stage!.width / 2 + 90, y: stage!.y + stage!.height / 2 }
      await page!.mouse.move(point.x, point.y)
      await page!.mouse.wheel(0, -200)
      await expect(viewer.getByLabel('图片缩放比例', { exact: true })).not.toHaveText('100%')
      const after = (await displayed.boundingBox())!
      expect(after.width).toBeGreaterThan(before.width)
      expect(Math.abs((point.x - after.x) / after.width - (point.x - before.x) / before.width)).toBeLessThan(.003)
      await page!.mouse.move(point.x, point.y)
      await page!.mouse.down()
      await page!.mouse.move(point.x + 70, point.y + 35, { steps: 4 })
      await page!.mouse.up()
      const dragged = (await displayed.boundingBox())!
      expect(dragged.x - after.x).toBeGreaterThan(60)
      expect(dragged.y - after.y).toBeGreaterThan(25)
      await viewer.getByRole('button', { name: '旋转图片', exact: true }).click()
      const rotated = (await displayed.boundingBox())!
      expect(Math.abs(rotated.width - after.height)).toBeLessThan(2)
      expect(Math.abs(rotated.height - after.width)).toBeLessThan(2)
      await viewer.getByRole('button', { name: '适应窗口', exact: true }).click()
      const fittedRotation = (await displayed.boundingBox())!
      expect(fittedRotation.width).toBeLessThanOrEqual(stage!.width)
      expect(fittedRotation.height).toBeLessThanOrEqual(stage!.height)
      await page!.keyboard.press('0')
      await expect(displayed).toHaveCSS('transform', /matrix\([^,]+, 0, 0, [^,]+, [^,]+, [^)]+\)/)
      await expect(viewer.locator('.image-viewer-dimensions')).toHaveText('1600 × 900')
      await viewer.getByRole('button', { name: '放大图片', exact: true }).click()
      const retained = await displayed.getAttribute('style')
      const requests = imageRequests
      await page!.evaluate(() => { (window as any).viewerImageBeforeStream = document.querySelector('.image-viewer-image') })
      delta('\n\n流式追加的文字不会重置已打开的图片。')
      await expect(assistant.locator('.message-text')).toContainText('不会重置')
      expect(await page!.evaluate(() => (window as any).viewerImageBeforeStream === document.querySelector('.image-viewer-image'))).toBe(true)
      expect(await displayed.getAttribute('style')).toBe(retained)
      expect(imageRequests).toBe(requests)
    })
    await test.step('空白点击退出，图片与浮条不退出，轻微移动和跨图片边界拖动正确区分', async () => {
      const trigger = assistant.getByRole('button', { name: '查看图片：山景', exact: true })
      await page!.keyboard.press('0')
      await displayed.click()
      await viewer.locator('.image-viewer-heading').click()
      await viewer.getByLabel('图片位置', { exact: true }).click()
      await expect(viewer).toBeVisible()
      const area = (await viewer.locator('.image-viewer-stage').boundingBox())!
      let bounds = (await displayed.boundingBox())!
      const y = area.y + area.height / 2
      // A tiny image-to-background release must stay open, as must the reverse.
      for (const [from, to] of [[bounds.x + 1, bounds.x - 1], [bounds.x - 1, bounds.x + 1]]) {
        await page!.mouse.move(from, y)
        await page!.mouse.down()
        await page!.mouse.move(to, y)
        await page!.mouse.up()
        await expect(viewer).toBeVisible()
      }
      // Returning to the start does not turn a completed drag into a click.
      await page!.mouse.move(area.x + 3, y)
      await page!.mouse.down()
      await page!.mouse.move(area.x + 23, y, { steps: 3 })
      await page!.mouse.move(area.x + 3, y, { steps: 3 })
      await page!.mouse.up()
      await expect(viewer).toBeVisible()
      // Ordinary click jitter is still a click on the blank canvas.
      await page!.mouse.move(area.x + 3, y)
      await page!.mouse.down()
      await page!.mouse.move(area.x + 4, y + 1)
      await page!.mouse.up()
      await expect(viewer).toHaveCount(0)
      await expect(trigger).toBeFocused()
      await trigger.click()
      await loaded(displayed)
      await expect(viewer).toBeFocused()
      // The transparent top-right area is blank too, with no close button.
      const viewport = page!.viewportSize() || await page!.evaluate(() => ({ width: innerWidth, height: innerHeight }))
      await page!.mouse.click(viewport.width - 12, 24)
      await expect(viewer).toHaveCount(0)
      await expect(trigger).toBeFocused()
      await trigger.click()
      await loaded(displayed)
      await viewer.getByRole('button', { name: '原始比例', exact: true }).click()
      await viewer.getByRole('button', { name: '旋转图片', exact: true }).click()
      await viewer.getByRole('button', { name: '放大图片', exact: true }).click()
      bounds = (await displayed.boundingBox())!
      expect(bounds.x).toBeGreaterThan(area.x + 40)
      await page!.mouse.move(bounds.x + 10, y)
      await page!.mouse.down()
      await page!.mouse.move(bounds.x - 30, y + 60, { steps: 5 })
      await page!.mouse.up()
      await expect(viewer).toBeVisible()
      const moved = (await displayed.boundingBox())!
      expect(moved.y - bounds.y).toBeGreaterThan(50)
      // Blank dragging does not pan the image, nor dismiss it on release.
      await page!.mouse.move(area.x + 3, y)
      await page!.mouse.down()
      await page!.mouse.move(area.x + 3, y + 35, { steps: 3 })
      await page!.mouse.up()
      await expect(viewer).toBeVisible()
      expect((await displayed.boundingBox())!.y).toBeCloseTo(moved.y, 0)
      await viewer.locator('.image-viewer-stage').click({ position: { x: 3, y: 3 } })
      await expect(viewer).toHaveCount(0)
      await expect(trigger).toBeFocused()
      await trigger.click()
      await loaded(displayed)
    })
    await test.step('远程原图与动图直接保存，保留源字节和可解码尺寸', async () => {
      await saveStep('保存 HTTP 原始 SVG', async () => {
        const output = path.join(directory, '保存-远程原图.svg')
        await dialogState(output)
        await viewer.getByRole('button', { name: '保存图片', exact: true }).click()
        await expect(viewer.locator('.image-viewer-feedback[role=status]')).toHaveText('图片已保存')
        const bytes = fs.readFileSync(output)
        expect(bytes.toString('utf8')).toBe(landscape)
        await readable(bytes, 'image/svg+xml', [1600, 900])
      })
      await page!.keyboard.press('ArrowRight')
      await loaded(displayed)
      await expect(viewer.getByLabel('图片位置', { exact: true })).toHaveText('2 / 3')
      await expect(displayed).toHaveAttribute('alt', '竖图')
      await page!.keyboard.press('ArrowRight')
      await loaded(displayed)
      await expect(viewer.getByLabel('图片位置', { exact: true })).toHaveText('3 / 3')
      await saveStep('保存 HTTP 原始 GIF 双帧', async () => {
        const animation = path.join(directory, '保存-原始动图.gif')
        await dialogState(animation)
        await viewer.getByRole('button', { name: '保存图片', exact: true }).click()
        await expect(viewer.locator('.image-viewer-feedback[role=status]')).toHaveText('图片已保存')
        const animationBytes = fs.readFileSync(animation)
        expect(animationBytes.equals(gif)).toBe(true)
        await readable(animationBytes, 'image/gif', [1, 1])
        expect(animationBytes.toString('hex').match(/21f904/g)).toHaveLength(2)
      })
      await escape()
      await expect(assistant.getByRole('button', { name: '查看图片：山景', exact: true })).toBeFocused()
    })
    await saveStep('普通链接保留原操作，本地和非图片 data 协议不能经 IPC 读取', async () => {
      await assistant.getByRole('link', { name: '普通外链', exact: true }).click()
      await expect(viewer).toHaveCount(0)
      expect(await application!.evaluate(() => (globalThis as any).imageSaveTest.urls)).toEqual(['https://example.com/image-link'])
      for (const source of ['file:///C:/Windows/win.ini', 'data:text/html;base64,PHNjcmlwdD4=', 'javascript:alert(1)', 'blob:https://example.com/fake']) {
        const result = await page!.evaluate(async source => {
          try { await (window as any).modloader.saveImage(source, '不允许'); return 'unexpected success' }
          catch (error) { return String(error) }
        }, source)
        expect(result).toContain('只允许保存内嵌图片或 HTTP(S) 原图')
      }
      const calls = await application!.evaluate(() => (globalThis as any).imageSaveTest.calls)
      const unsafe = await page!.evaluate(async source => {
        try { await (window as any).modloader.saveImage(source, '重定向'); return 'unexpected success' }
        catch (error) { return String(error) }
      }, `${origin}/redirect-file.svg`)
      expect(unsafe).toContain('ERR_UNSAFE_REDIRECT')
      expect(await application!.evaluate(() => (globalThis as any).imageSaveTest.calls)).toBe(calls)
      const redirected = path.join(directory, '保存-HTTP重定向.svg')
      await dialogState(redirected)
      expect(await page!.evaluate(source => (window as any).modloader.saveImage(source, '重定向'), `${origin}/redirect-scene.svg`)).toEqual({ canceled: false })
      expect(fs.readFileSync(redirected, 'utf8')).toBe(landscape)
    })
    event({ choices: [{ delta: {}, finish_reason: 'stop' }] })
    response!.end('data: [DONE]\n\n')
    await expect(page.getByRole('button', { name: '停止生成', exact: true })).toHaveCount(0)
    await expect(page.locator('.chat-save-status')).toHaveText('已保存')
    await test.step('历史重载后仍可查看，坏图失败可结束，小窗口工具栏完整', async () => {
      await application!.evaluate(({ Menu }) => { Menu.getApplicationMenu()!.items.find(item => item.label === '视图')!.submenu!.items.find(item => item.label === '重新加载界面')!.click() })
      await expect(page!.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 30_000 })
      await page!.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '测试对话', exact: true }).click()
      const trigger = page!.locator('.chat-message.user').last().getByRole('button', { name: '查看图片：像素.png', exact: true })
      await expect(trigger).toBeVisible()
      await trigger.click()
      await loaded(displayed)
      await expect(viewer.getByLabel('图片位置', { exact: true })).toHaveText('1 / 2')
      await viewer.locator('.image-viewer-stage').click({ position: { x: 3, y: 3 } })
      await expect(viewer).toHaveCount(0)
      await expect(trigger).toBeFocused()
      const remote = page!.locator('.chat-message.assistant').last().getByRole('button', { name: '查看图片：山景', exact: true })
      await loaded(remote)
      failImage = true
      await remote.click()
      await expect(viewer.getByRole('alert')).toHaveText('图片加载失败')
      await expect(viewer.getByRole('button', { name: '保存图片', exact: true })).toBeDisabled()
      await escape()
      failImage = false
      await remote.click()
      await loaded(displayed)
      await application!.evaluate(({ BrowserWindow }) => { const window = BrowserWindow.getAllWindows()[0]; window.setSize(940, 640); window.webContents.setZoomFactor(1.2) })
      await expect(viewer.getByRole('button', { name: '关闭图片查看器', exact: true })).toHaveCount(0)
      await expect.poll(() => viewer.evaluate(element => element.scrollWidth - element.clientWidth)).toBeLessThanOrEqual(1)
      const bounds = await viewer.evaluate(element => ({ stage: element.querySelector('.image-viewer-stage')!.getBoundingClientRect().height, buttons: Array.from(element.querySelectorAll('button'), button => { const rect = button.getBoundingClientRect(); return rect.left >= 0 && rect.right <= innerWidth && rect.top >= 0 && rect.bottom <= innerHeight }) }))
      expect(bounds.stage).toBeGreaterThan(300)
      expect(bounds.buttons.every(Boolean)).toBe(true)
      const safeArea = await viewer.evaluate(element => {
        const image = element.querySelector('.image-viewer-image')!.getBoundingClientRect()
        const top = element.querySelector('.image-viewer-top')!.getBoundingClientRect()
        const tools = element.querySelector('.image-viewer-toolbar')!.getBoundingClientRect()
        return image.top > top.bottom && image.bottom < tools.top
      })
      expect(safeArea).toBe(true)
      const capture = await application!.evaluate(async ({ BrowserWindow }) => (await BrowserWindow.getAllWindows()[0].webContents.capturePage()).toPNG().toString('base64'))
      fs.writeFileSync(path.join(directory, 'image-viewer-small-window.png'), Buffer.from(capture, 'base64'))
      await escape()
      await expect(remote).toBeFocused()
      await page!.locator('.chat-message.user').last().getByRole('button', { name: '编辑消息', exact: true }).click()
      await expect(page!.getByRole('textbox', { name: '编辑消息内容', exact: true })).toHaveValue('请查看这两张图。')
      await page!.getByRole('button', { name: '取消编辑', exact: true }).click()
      await expect(page!.locator('.chat-message.user').last().locator('.message-images img')).toHaveCount(2)
    })
    expect(errors).toEqual([])
    console.log(`Image viewer ${uiOnly ? 'UI-only (original-save evidence reused)' : 'complete'} fixture and screenshots:`, directory)
  } catch (error) {
    if (page && !page.isClosed()) await page.screenshot({ path: path.join(directory, 'failure.png') }).catch(() => {})
    throw error
  } finally {
    response?.end()
    if (application) {
      await application.evaluate(({ dialog, shell }) => {
        const state = (globalThis as any).imageSaveTest
        if (state) { dialog.showSaveDialog = state.save; shell.openExternal = state.open }
      }).catch(() => {})
      await Promise.all(application.windows().map(window => window.unrouteAll({ behavior: 'wait' })))
      await application.close()
    }
    server.closeAllConnections()
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
})
