import { pinNavigation } from './navigation'
import { disableWebSearch } from './chat-controls'
import { test, expect, _electron as electron } from '@playwright/test'
import { createServer, type ServerResponse } from 'node:http'
import * as path from 'node:path'
import * as fs from 'node:fs'

test('长回复结束后保持位置，正文可以连续滚动到末尾', async () => {
  test.setTimeout(90_000)
  const root = path.resolve(__dirname, '..')
  const directory = fs.mkdtempSync(path.join(root, 'work', 'ui-chat-scroll-'))
  const executablePath = process.env.MODLOADER_TEST_EXECUTABLE
  let response: ServerResponse | undefined
  const server = createServer(async (request, result) => {
    result.setHeader('Access-Control-Allow-Origin', '*')
    result.setHeader('Access-Control-Allow-Headers', '*')
    if (request.method === 'OPTIONS') { result.end(); return }
    for await (const _chunk of request) {}
    result.setHeader('Content-Type', 'text/event-stream')
    result.flushHeaders()
    response = result
  })
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const address = server.address() as { port: number }
  const application = await electron.launch({ ...(executablePath ? { executablePath, args: [] } : { args: [root] }), env: { ...process.env, MODLOADER_DATA_DIR: directory }, timeout: 60_000 })
  const page = await application.firstWindow()
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  const input = page.getByRole('textbox', { name: '对话消息' })
  const messages = page.locator('.chat-messages')
  const assistant = page.locator('.chat-message.assistant').last()
  const event = (value: unknown) => response!.write(`data:${JSON.stringify(value)}\n\n`)
  const delta = (value: unknown) => event({ choices: [{ delta: value }] })
  const metrics = () => messages.evaluate(element => ({
    // Normalize zoom's floating-point scroll offset to Chromium's layout unit.
    top: Math.round(element.scrollTop * 64) / 64, height: element.scrollHeight, viewport: element.clientHeight,
    pageHeight: document.documentElement.scrollHeight, windowHeight: innerHeight,
  }))
  const screenshot = async (name: string) => {
    const png = await application.evaluate(async ({ BrowserWindow }) => (await BrowserWindow.getAllWindows()[0].webContents.capturePage()).toPNG().toString('base64'))
    fs.writeFileSync(path.join(directory, name), Buffer.from(png, 'base64'))
  }
  const finish = async () => {
    event({ choices: [{ delta: {}, finish_reason: 'stop' }] })
    event({ choices: [], usage: { prompt_tokens: 1024, completion_tokens: 2048 }, timings: { predicted_per_second: 160 } })
    response!.end('data: [DONE]\n\n')
    await expect(page.getByRole('button', { name: '停止生成', exact: true })).toHaveCount(0)
  }
  try {
    await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].webContents.setZoomFactor(1.2))
    await pinNavigation(page)
    await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
    await page.route('**/api/state', async route => {
      const original = await route.fetch()
      const state = await original.json()
      await route.fulfill({ response: original, json: { ...state, settings: { ...state.settings, font_scale: 1.3 }, instances: [{ id: 'scroll-fixture', profile_id: 'scroll-profile', model_id: 'scroll-model', alias: 'Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS', backend: 'strata', state: 'ready', capabilities: { input: ['text'] }, active_requests: 0 }] } })
    })
    await page.route('**/api/chat', route => route.continue({ url: `http://127.0.0.1:${address.port}/api/chat` }))
    await page.route('**/api/instances/scroll-fixture/telemetry', route => route.fulfill({ json: { groups: [], unavailable: [], live_generation: { tokens_per_second: 160, window_s: 2, available_for_chat: true } } }))
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '测试对话', exact: true }).click()
    await disableWebSearch(page)
    await page.getByRole('button', { name: '取消钉住顶部栏', exact: true }).click()
    await input.fill('输出一段足够长的分段回答')
    await page.getByRole('button', { name: '发送消息', exact: true }).click()
    await expect.poll(() => Boolean(response)).toBe(true)
    // Keep the pointer in the transcript while streaming, like normal reading.
    // Hovering or taking a screenshot after completion can refresh compositor
    // state before the very first wheel input and hide a stale scroll range.
    await messages.hover()
    const wheelPosition = await messages.evaluate(element => {
      const box = element.getBoundingClientRect()
      return { x: box.x + box.width / 2, y: box.y + box.height / 2 }
    })
    for (let index = 0; index < 30; index++) {
      delta({ reasoning_content: `思考第 ${index + 1} 步。\n\n` })
      await new Promise(resolve => setTimeout(resolve, 35))
    }
    await expect(assistant.getByRole('region', { name: '思考内容' })).toHaveCount(1)
    const longReply = Array.from({ length: 80 }, (_, index) => `### 回答第 ${index + 1} 段\n\n这是包含 **Markdown** 的详细正文，应该完整保留，并能够一直向下滚动阅读。\n\n`).join('')
    for (let offset = 0; offset < longReply.length; offset += 32) {
      delta({ content: longReply.slice(offset, offset + 32) })
      await new Promise(resolve => setTimeout(resolve, 35))
    }
    await finish()
    await application.evaluate(({ BrowserWindow }, position) => {
      const contents = BrowserWindow.getAllWindows()[0].webContents
      const zoom = contents.getZoomFactor()
      contents.sendInputEvent({ type: 'mouseWheel', x: Math.round(position.x * zoom), y: Math.round(position.y * zoom), deltaY: 180, wheelTicksY: 1, hasPreciseScrollingDeltas: false, canScroll: true })
    }, wheelPosition)
    await expect.poll(async () => { const value = await metrics(); return value.height - value.top - value.viewport }).toBeGreaterThan(100)
    const afterFirstWheel = await metrics()
    console.log('First wheel after completion:', afterFirstWheel)
    await screenshot('first-wheel.png')
    expect(afterFirstWheel.height - afterFirstWheel.top - afterFirstWheel.viewport).toBeLessThan(300)
    await application.evaluate(({ BrowserWindow }, position) => {
      const contents = BrowserWindow.getAllWindows()[0].webContents
      const zoom = contents.getZoomFactor()
      contents.sendInputEvent({ type: 'mouseWheel', x: Math.round(position.x * zoom), y: Math.round(position.y * zoom), deltaY: -10000, wheelTicksY: -50, hasPreciseScrollingDeltas: false, canScroll: true })
    }, wheelPosition)
    await expect.poll(async () => { const value = await metrics(); return value.height - value.top - value.viewport }).toBeLessThanOrEqual(1)
    await screenshot('finished.png')
    await expect.poll(async () => { const value = await metrics(); return value.height - value.top - value.viewport }).toBeLessThanOrEqual(1)
    await expect.poll(async () => { const value = await metrics(); return value.pageHeight - value.windowHeight }).toBeLessThanOrEqual(1)
    await messages.hover()
    await page.mouse.wheel(0, -20_000)
    await expect.poll(() => messages.evaluate(element => element.scrollTop)).toBe(0)
    await page.mouse.wheel(0, 400)
    await expect.poll(() => messages.evaluate(element => element.scrollTop)).toBeGreaterThan(300)
    await page.mouse.wheel(0, 20_000)
    await expect.poll(async () => { const value = await metrics(); return value.height - value.top - value.viewport }).toBeLessThanOrEqual(1)
    await expect(assistant.getByRole('heading', { name: '回答第 80 段', exact: true })).toBeInViewport()
    await test.step('输出结束后拖动原生滚动条可以到达全部内容的末尾', async () => {
      await messages.press('Home')
      await expect.poll(() => messages.evaluate(element => element.scrollTop)).toBe(0)
      const thumb = await messages.evaluate(element => {
        const box = element.getBoundingClientRect()
        const height = Math.max(28, element.clientHeight ** 2 / element.scrollHeight)
        return { x: box.right - 5, start: box.top + height / 2, end: box.bottom - 8 }
      })
      await page.mouse.move(thumb.x, thumb.start)
      await page.mouse.down()
      try { await page.mouse.move(thumb.x, thumb.end, { steps: 12 }) }
      finally { await page.mouse.up() }
      await expect.poll(async () => { const value = await metrics(); return value.height - value.top - value.viewport }).toBeLessThanOrEqual(1)
      await expect(assistant.getByRole('heading', { name: '回答第 80 段', exact: true })).toBeInViewport()
      await screenshot('finished-scrollbar-drag.png')
    })
    response = undefined
    await input.fill('输出一段很长的代码')
    await page.getByRole('button', { name: '发送消息', exact: true }).click()
    await expect.poll(() => Boolean(response)).toBe(true)
    delta({ content: '```python\n' })
    for (let part = 0; part < 10; part++) {
      delta({ content: Array.from({ length: 20 }, (_, index) => `print("第 ${part * 20 + index + 1} 行代码")\n`).join('') })
      await expect(assistant.locator('.message-text')).toContainText(`第 ${part * 20 + 20} 行代码`)
    }
    await expect.poll(async () => { const value = await metrics(); return value.height - value.top - value.viewport }).toBeLessThanOrEqual(1)
    await finish()
    await expect(assistant.locator('code.hljs')).toBeVisible()
    await expect.poll(async () => { const value = await metrics(); return value.height - value.top - value.viewport }).toBeLessThanOrEqual(1)
    await screenshot('code-finished.png')
    await messages.hover()
    await page.mouse.wheel(0, -30_000)
    await expect.poll(() => messages.evaluate(element => element.scrollTop)).toBe(0)
    await page.mouse.wheel(0, 500)
    await expect.poll(() => messages.evaluate(element => element.scrollTop)).toBeGreaterThan(300)
    await page.mouse.wheel(0, 30_000)
    await expect.poll(async () => { const value = await metrics(); return value.height - value.top - value.viewport }).toBeLessThanOrEqual(1)

    await test.step('上翻暂停后结束仍保留阅读位置，End 恢复跟随', async () => {
      response = undefined
      await input.fill('继续输出，检查手动阅读位置')
      await page.getByRole('button', { name: '发送消息', exact: true }).click()
      await expect.poll(() => Boolean(response)).toBe(true)
      delta({ content: Array.from({ length: 20 }, (_, index) => `手动阅读第 ${index + 1} 段。\n\n`).join('') })
      await expect(assistant.locator('.message-text')).toContainText('手动阅读第 20 段')
      await expect.poll(async () => { const value = await metrics(); return value.height - value.top - value.viewport }).toBeLessThanOrEqual(1)
      await messages.hover()
      await page.mouse.wheel(0, -500)
      await expect.poll(async () => { const value = await metrics(); return value.height - value.top - value.viewport }).toBeGreaterThan(300)
      const readingTop = await messages.evaluate(element => element.scrollTop)
      delta({ content: '\n\n上翻后继续收到的正文。\n\n'.repeat(10) })
      await expect(assistant.locator('.message-text')).toContainText('上翻后继续收到的正文。')
      await expect.poll(async () => Math.abs(await messages.evaluate(element => element.scrollTop) - readingTop)).toBeLessThanOrEqual(1)
      await finish()
      await expect.poll(async () => Math.abs(await messages.evaluate(element => element.scrollTop) - readingTop)).toBeLessThanOrEqual(1)
      await messages.press('End')
      await expect.poll(async () => { const value = await metrics(); return value.height - value.top - value.viewport }).toBeLessThanOrEqual(1)
      await screenshot('finished-manual-scroll.png')
    })
    await test.step('切换页面不丢失内容，回到对话后仍可上下滚动', async () => {
      const navigation = page.getByRole('navigation', { name: '主导航' })
      await navigation.getByRole('button', { name: '模型库', exact: true }).click()
      await navigation.getByRole('button', { name: '测试对话', exact: true }).click()
      await messages.hover()
      await page.mouse.wheel(0, -500)
      await expect.poll(async () => { const value = await metrics(); return value.height - value.top - value.viewport }).toBeGreaterThan(300)
      await messages.press('End')
      await expect.poll(async () => { const value = await metrics(); return value.height - value.top - value.viewport }).toBeLessThanOrEqual(1)
      await expect(assistant.locator('.message-text')).toContainText('手动阅读第 20 段')
    })
    expect(errors).toEqual([])
  } finally {
    response?.end()
    await application.close()
    server.closeAllConnections()
    await new Promise<void>(resolve => server.close(() => resolve()))
    console.log('Chat scroll screenshots:', directory)
  }
})
