import { pinNavigation } from './navigation'
import { test, expect, _electron as electron } from '@playwright/test'
import * as fs from 'node:fs'
import * as path from 'node:path'

test('数值框与弹窗保留焦点和滚动位置，切页返回可继续阅读', async () => {
  test.setTimeout(90_000)
  const root = path.resolve(__dirname, '..')
  const directory = fs.mkdtempSync(path.join(root, 'work', 'ui-modal-wheel-'))
  const application = await electron.launch({
    ...(process.env.MODLOADER_TEST_EXECUTABLE ? { executablePath: process.env.MODLOADER_TEST_EXECUTABLE, args: [] } : { args: [root] }),
    env: { ...process.env, MODLOADER_DATA_DIR: path.join(directory, 'data') }, timeout: 60_000,
  })
  const page = await application.firstWindow()
  const errors: string[] = []
  const mutations: string[] = []
  const geometry: Array<Record<string, unknown>> = []
  const estimates: number[] = []
  const pendingEstimates: Array<{ context: number; release: () => void }> = []
  let holdEditedEstimates = false
  const gib = 1024 ** 3
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/api/**', async route => {
    const request = route.request(), url = new URL(request.url())
    if (url.pathname === '/api/resources/estimate') {
      const context = Number(request.postDataJSON().parameters?.context_length || 2048)
      estimates.push(context)
      if (holdEditedEstimates && context !== 2048) await new Promise<void>(release => pendingEstimates.push({ context, release }))
      try {
        await route.fulfill({ json: { source: 'fixture', notes: [`上下文 ${context} Token 的界面预估，不执行模型加载。`, '权重与 KV 缓存均计入新增占用；实际用量由运行环境决定。'], resources: [
          { kind: 'ram', label: '系统内存', bytes: context * 1024 ** 2, quality: 'estimate', notes: [] },
          { kind: 'vram', device: 'CUDA0', label: 'GPU 显存', bytes: 1.5 * gib, quality: 'estimate', notes: [] },
          { kind: 'vram', device: 'CUDA1', label: 'GPU 显存', bytes: null, quality: 'dynamic', budget_fraction: .5, reserve_bytes: gib, notes: ['动态缓存预留仅作参考。'] },
        ] } })
      } catch { /* A later wheel value may abort an obsolete estimate request. */ }
      return
    }
    if (url.pathname === '/api/profiles/preflight') return route.fulfill({ json: { status: 'ready', message: '', candidates: [], errors: [], vision_enabled: false } })
    if (request.method() !== 'GET') { mutations.push(url.pathname); return route.abort() }
    return route.continue()
  })
  await page.route('**/api/state', async route => {
    const response = await route.fetch(), state = await response.json()
    return route.fulfill({ response, json: { ...state,
      models: [{ id: 'wheel-model', name: '滚轮交互样例', path: path.join(directory, 'fixture.gguf'), format: 'gguf', size_bytes: 0, files: [], modalities: ['text'], metadata: {}, source: {}, managed: false, warnings: [] }],
      profiles: Array.from({ length: 8 }, (_, i) => ({ id: `wheel-${i}`, name: `滚轮配置 ${i}`, model_id: 'wheel-model', runtime_id: 'wheel-runtime', backend: 'llama_cpp', alias: `wheel-${i}`, parameters: { context_length: 2048 }, generation: {}, extra_args: [], revision: 1 })),
      runtimes: [{ id: 'wheel-runtime', name: '界面样例', backend: 'llama_cpp', path: directory, version: 'fixture', metadata: {}, managed: false }],
      hardware: { memory_total: 32 * gib, memory_used: 8 * gib, memory_available: 24 * gib, gpus: [{ device: 'CUDA0', memory_total: 12 * gib, memory_used: 3 * gib }, { device: 'CUDA1', memory_total: 8 * gib, memory_used: 2 * gib }] },
      settings: { ...state.settings, font_scale: 1.3 }, instances: [],
    } })
  })
  const settle = () => page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))))
  async function nativeScreenshot(name: string) {
    await settle()
    const png = await application.evaluate(async ({ BrowserWindow }) => (await BrowserWindow.getAllWindows()[0].webContents.capturePage()).toPNG().toString('base64'))
    const output = path.join(directory, name)
    fs.writeFileSync(output, Buffer.from(png, 'base64'))
    console.log(`Modal wheel screenshot: ${output}`)
  }
  try {
    await pinNavigation(page)
    await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
    await application.evaluate(({ BrowserWindow }) => { const window = BrowserWindow.getAllWindows()[0]; window.setSize(1440, 960); window.webContents.setZoomFactor(1.2) })
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    const card = page.locator('.profile-card').filter({ has: page.getByRole('heading', { name: '滚轮配置 3', exact: true }) })
    await card.getByRole('button', { name: '编辑配置', exact: true }).click()
    const dialog = page.getByRole('dialog', { name: '编辑加载配置', exact: true })
    await expect(dialog.getByRole('textbox', { name: '配置名称', exact: true })).toBeFocused()
    const body = dialog.locator('.modal-body')
    const preview = dialog.locator('.resource-preview')
    const context = dialog.getByRole('spinbutton', { name: '上下文长度', exact: true })
    await expect(preview.locator('.resource-preview-row')).toHaveCount(3)
    await context.click()
    const pageTop = await page.evaluate(() => window.scrollY)
    const modalTop = await body.evaluate(element => element.scrollTop)
    await context.hover()
    const box = await context.boundingBox()
    if (!box) throw new Error('上下文长度输入框缺少可见边界。')
    const pointer = { x: box.x + box.width / 2, y: box.y + box.height / 2 }
    const snapshot = async (phase: string) => {
      const sample = await context.evaluate((element, point) => {
        const rect = element.getBoundingClientRect()
        const body = element.closest('.modal-body')!
        const modal = element.closest('.modal')!
        return { y: rect.y, height: rect.height, scrollTop: body.scrollTop, modalY: modal.getBoundingClientRect().y, modalHeight: modal.getBoundingClientRect().height, value: (element as HTMLInputElement).value, pointerHitsInput: document.elementFromPoint(point.x, point.y) === element }
      }, pointer)
      geometry.push({ phase, ...sample })
      return sample
    }
    const before = await snapshot('before-wheel')
    await nativeScreenshot('modal-wheel-before.png')
    holdEditedEstimates = true
    await page.mouse.wheel(0, -120)
    await expect(context).toHaveValue('2049')
    await expect.poll(() => pendingEstimates.some(item => item.context === 2049)).toBe(true)
    await settle()
    await expect(preview).toHaveAttribute('aria-busy', 'true')
    await expect(preview).toHaveAttribute('data-stale', 'true')
    await expect(preview.getByRole('status')).toContainText('更新中')
    await expect(preview.getByRole('status')).toContainText('上次估算')
    await expect(preview.locator('.resource-preview-row')).toHaveCount(3)
    const pending = await snapshot('estimate-pending')
    await nativeScreenshot('modal-wheel-pending.png')
    // Keep the pointer stationary across both the delayed response and successive edits.
    await page.mouse.wheel(0, -120)
    await settle()
    const during = await snapshot('second-wheel-during-pending')
    await expect.poll(() => pendingEstimates.some(item => item.context === Number(during.value))).toBe(true)
    holdEditedEstimates = false
    for (const estimate of pendingEstimates) estimate.release()
    await expect.poll(() => estimates.at(-1)).toBe(Number(during.value))
    await expect(preview).not.toHaveAttribute('aria-busy', 'true')
    await expect(preview).not.toHaveAttribute('data-stale', 'true')
    await settle()
    const completed = await snapshot('estimate-completed')
    await page.mouse.wheel(0, -120)
    await settle()
    const continued = await snapshot('third-wheel-after-response')
    console.log(JSON.stringify({ geometry, estimates }))
    expect(during.value, '预估等待时，鼠标不移动仍应调整同一输入框').toBe('2050')
    expect(continued.value, '预估返回后，鼠标不移动仍应调整同一输入框').toBe('2051')
    for (const sample of [pending, during, completed, continued]) {
      expect(Math.abs(sample.y - before.y), '预估刷新不应移动正在滚轮调整的输入框').toBeLessThanOrEqual(1)
      expect(sample.scrollTop).toBe(before.scrollTop)
      expect(sample.pointerHitsInput).toBe(true)
    }
    expect(await page.evaluate(() => window.scrollY)).toBe(pageTop)
    await context.press('ControlOrMeta+A')
    await context.pressSequentially('2048')
    await expect.poll(() => estimates.at(-1)).toBe(2048)
    await expect(preview).not.toHaveAttribute('aria-busy', 'true')

    // Losing focus restores scrolling over the field without changing its value.
    await dialog.getByRole('heading', { name: '编辑加载配置', exact: true }).click()
    await context.hover()
    await page.mouse.wheel(0, 240)
    await expect.poll(() => body.evaluate(element => element.scrollTop)).toBeGreaterThan(modalTop)
    await expect(context).toHaveValue('2048')

    // Neither overlay wheel nor a scroll at the end of the modal may reach the page.
    await page.mouse.move(4, 4)
    await page.mouse.wheel(0, 360)
    await settle()
    expect(await page.evaluate(() => window.scrollY)).toBe(pageTop)
    await body.evaluate(element => { element.scrollTop = element.scrollHeight })
    await body.hover({ position: { x: 12, y: 12 } })
    await page.mouse.wheel(0, 480)
    await settle()
    expect(await page.evaluate(() => window.scrollY)).toBe(pageTop)

    await dialog.getByRole('tab', { name: '生成默认值', exact: true }).click()
    const temperature = dialog.getByRole('spinbutton', { name: /^温度/ })
    await temperature.fill('0.7')
    await temperature.hover()
    const decimalTop = await body.evaluate(element => element.scrollTop)
    await page.mouse.wheel(0, -120)
    await expect(temperature).toHaveAttribute('step', '0.1')
    await expect(temperature).toHaveValue('0.8')
    await settle()
    expect(await body.evaluate(element => element.scrollTop)).toBe(decimalTop)
    await nativeScreenshot('modal-wheel.png')
    await dialog.getByRole('button', { name: '取消', exact: true }).click()
    await expect(dialog).toBeHidden()
    await expect(card.getByRole('button', { name: '编辑配置', exact: true })).toBeFocused()
    expect(await page.evaluate(() => window.scrollY)).toBe(pageTop)
    await page.mouse.move(500, 500)
    await page.mouse.wheel(0, 480)
    await expect.poll(() => page.evaluate(() => window.scrollY)).toBeGreaterThan(pageTop)

    const libraryScroll = await page.evaluate(() => window.scrollY)
    const nav = page.getByRole('navigation', { name: '主导航' })
    await nav.getByRole('button', { name: '运行监控', exact: true }).click()
    await expect.poll(() => page.evaluate(() => window.scrollY)).toBe(0)
    await nav.getByRole('button', { name: '运行环境', exact: true }).click()
    await expect.poll(() => page.evaluate(() => window.scrollY)).toBe(0)
    await page.mouse.move(500, 500)
    await page.mouse.wheel(0, 600)
    await expect.poll(() => page.evaluate(() => window.scrollY)).toBeGreaterThan(0)
    await settle()
    const runtimeScroll = await page.evaluate(() => window.scrollY)
    // Both the brand shortcut and the navigation tabs restore their own page position.
    await page.getByRole('button', { name: 'ModLoader 模型库', exact: true }).click()
    await expect.poll(() => page.evaluate(() => window.scrollY)).toBe(libraryScroll)
    for (let i = 0; i < 3; i++) {
      await nav.getByRole('button', { name: '运行环境', exact: true }).click()
      await expect.poll(() => page.evaluate(() => window.scrollY)).toBe(runtimeScroll)
      await nav.getByRole('button', { name: '模型库', exact: true }).click()
      await expect.poll(() => page.evaluate(() => window.scrollY)).toBe(libraryScroll)
    }

    // A second form and a non-form dialog use the same focus lifecycle.
    const importButton = page.getByRole('button', { name: '导入模型', exact: true })
    await importButton.click()
    const importDialog = page.getByRole('dialog', { name: '导入模型文件', exact: true })
    await expect(importDialog.getByRole('textbox', { name: '文件路径', exact: true })).toBeFocused()
    await page.keyboard.press('Escape')
    await expect(importDialog).toBeHidden()
    await expect(importButton).toBeFocused()
    const aboutButton = page.getByRole('button', { name: '关于 ModLoader', exact: true })
    await aboutButton.click()
    const aboutDialog = page.getByRole('dialog', { name: 'ModLoader', exact: true })
    await expect(aboutDialog).toBeFocused()
    await page.keyboard.press('Shift+Tab')
    await expect(aboutDialog.locator('.modal-footer').getByRole('button', { name: '关闭', exact: true })).toBeFocused()
    await page.keyboard.press('Tab')
    await expect(aboutDialog.locator('.modal-header').getByRole('button', { name: '关闭', exact: true })).toBeFocused()
    await expect(page.getByRole('tooltip')).toBeVisible()
    // Escape first dismisses the focused icon's tooltip, then the dialog.
    await page.keyboard.press('Escape')
    await expect(page.getByRole('tooltip')).toBeHidden()
    await expect(aboutDialog).toBeVisible()
    await page.keyboard.press('Escape')
    await expect(aboutDialog).toBeHidden()
    await expect(aboutButton).toBeFocused()

    await page.getByRole('button', { name: '测试对话', exact: true }).click()
    await expect.poll(() => page.evaluate(() => window.scrollY)).toBe(0)
    await page.getByRole('button', { name: '对话参数', exact: true }).click()
    const maxTokens = page.getByRole('spinbutton', { name: '最大输出 Token', exact: true })
    await maxTokens.fill('128')
    await maxTokens.hover()
    const chatPanel = page.locator('.chat-options-body')
    const chatScroll = await chatPanel.evaluate(element => element.scrollTop)
    await page.mouse.wheel(0, -120)
    await expect(maxTokens).toHaveValue('129')
    await settle()
    expect(await chatPanel.evaluate(element => element.scrollTop)).toBe(chatScroll)
    expect(errors).toEqual([])
    expect(mutations).toEqual([])
    console.log(JSON.stringify({ screenshots: directory }))
  } catch (error) {
    console.log(JSON.stringify({ geometry, estimates, errors }))
    if (!page.isClosed()) await nativeScreenshot('modal-wheel-failure.png')
    throw error
  } finally {
    for (const estimate of pendingEstimates) estimate.release()
    await application.close()
  }
})
