import { test, expect, _electron as electron, type ElectronApplication, type Page } from '@playwright/test'
import { createServer, type ServerResponse } from 'node:http'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { pinNavigation } from './navigation'
import { disableWebSearch } from './chat-controls'

test('窗口拖入图片和文档：读取、保存、发送、重试与拒绝状态', async () => {
  test.setTimeout(150_000)
  const root = path.resolve(__dirname, '..')
  const artifacts = process.env.MODLOADER_TEST_ARTIFACT_DIR || path.join(root, '.artifacts/codex/chat-file-drop-20261029')
  fs.mkdirSync(artifacts, { recursive: true })
  const directory = fs.mkdtempSync(path.join(artifacts, 'electron-'))
  const textPath = path.join(directory, '项目说明.md')
  const imagePath = path.join(directory, '像素.png')
  const badPath = path.join(directory, '不支持.bin')
  fs.writeFileSync(textPath, '# 项目说明\n请保留 42 个配置。\n```ts\nconst value = 42\n```\n')
  fs.writeFileSync(imagePath, Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=', 'base64'))
  fs.writeFileSync(badPath, Buffer.from([0, 255, 0, 3, 4, 0, 9]))
  let response: ServerResponse | undefined
  let body: any
  let vision = true
  let uploads = 0
  let application: ElectronApplication | undefined
  let page: Page | undefined
  const errors: string[] = []
  const server = createServer(async (request, result) => {
    result.setHeader('Access-Control-Allow-Origin', '*')
    result.setHeader('Access-Control-Allow-Headers', '*')
    if (request.method === 'OPTIONS') { result.end(); return }
    let raw = ''
    for await (const chunk of request) raw += chunk
    body = JSON.parse(raw)
    result.setHeader('Content-Type', 'text/event-stream'); result.flushHeaders(); response = result
  })
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  try {
    application = await electron.launch({ args: [root], env: { ...process.env, MODLOADER_DATA_DIR: directory, MODLOADER_SENSOR_AUTO_START: '0' }, timeout: 60_000 })
    page = await application.firstWindow()
    page.on('pageerror', error => errors.push(error.message))
    page.on('request', request => { if (request.url().includes('/api/chat/attachments/text?')) uploads++ })
    await pinNavigation(page)
    await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
    let fixtureState: Promise<any> | undefined
    await page.route('**/api/state', async route => {
      fixtureState ??= route.fetch().then(response => response.json())
      const state = await fixtureState
      await route.fulfill({ json: { ...state, instances: [{ id: 'drop-fixture', model_id: 'drop-model', profile_id: 'drop-profile', alias: '附件测试模型', backend: 'llama_cpp', state: 'ready', capabilities: { input: vision ? ['text', 'image'] : ['text'] }, active_requests: 0 }] } })
    })
    await page.route('**/api/instances/drop-fixture/telemetry', route => route.fulfill({ json: { groups: [], unavailable: [], live_generation: null } }))
    await page.route('**/api/chat', route => route.continue({ url: `http://127.0.0.1:${(server.address() as { port: number }).port}/api/chat` }))
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '测试对话', exact: true }).click()
    await disableWebSearch(page)
    const cdp = await page.context().newCDPSession(page)
    const initialUrl = page.url()
    const draft = page.locator('.chat-composer-area')
    const overlay = page.locator('.chat-file-drop')
    async function drag(files: string[], type: 'dragEnter' | 'dragOver' | 'drop' | 'dragCancel', selector = '.chat-messages') {
      const box = (await page!.locator(selector).boundingBox())!
      await cdp.send('Input.dispatchDragEvent', { type, x: box.x + box.width / 2, y: box.y + box.height / 2, data: { items: [], files, dragOperationsMask: 1 } })
    }
    async function drop(files: string[], selector?: string) {
      await drag(files, 'dragEnter', selector); await drag(files, 'dragOver', selector); await drag(files, 'drop', selector)
    }
    async function finish() {
      response!.end('data: {"choices":[{"delta":{"content":"已读取附件。"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
      await expect(page!.getByRole('button', { name: '停止生成', exact: true })).toHaveCount(0)
    }
    await test.step('真实文件拖放经过 Chromium，嵌套区域不丢失提示，取消后不上传', async () => {
      await drag([textPath], 'dragEnter')
      await drag([textPath], 'dragOver')
      await expect(overlay).toHaveText('松开以添加文件')
      await page!.screenshot({ path: path.join(directory, 'drop-overlay.png') })
      await drag([textPath], 'dragOver', '.composer-actions')
      await expect(overlay).toBeVisible()
      await page!.keyboard.press('Escape')
      await drag([textPath], 'dragCancel')
      await expect(overlay).toHaveCount(0)
      expect(uploads).toBe(0)
      await page!.getByRole('textbox', { name: '对话消息', exact: true }).dispatchEvent('dragenter', { dataTransfer: await page!.evaluateHandle(() => { const data = new DataTransfer(); data.setData('text/plain', '普通文本'); return data }) })
      await expect(overlay).toHaveCount(0)
    })
    await test.step('混合图片文档拖入窗口，正文预览与草稿重载', async () => {
      await drop([textPath, imagePath], '.app-header')
      await expect(draft.locator('.attachment img')).toHaveCount(1)
      await expect(draft.locator('.chat-document summary')).toContainText('项目说明.md')
      await draft.locator('.chat-document summary').click()
      await expect(draft.locator('.chat-document pre')).toHaveText(fs.readFileSync(textPath, 'utf8'))
      await expect(page!.locator('.chat-save-status')).toHaveText('已保存')
      await page!.reload()
      await page!.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '测试对话', exact: true }).click()
      await expect(draft.locator('.chat-document summary')).toContainText('项目说明.md')
      await expect(draft.locator('.attachment img')).toHaveCount(1)
      expect(page!.url()).toBe(initialUrl)
      await page!.screenshot({ path: path.join(directory, 'mixed-attachments.png') })
    })
    await test.step('发送请求包含正文和原图，生成中不接受拖入，重试保留文档', async () => {
      await page!.getByRole('button', { name: '发送消息', exact: true }).click()
      await expect.poll(() => Boolean(response)).toBe(true)
      const content = body.messages.at(-1).content
      expect(content[0].text).toContain('项目说明.md')
      expect(content[0].text).toContain('请保留 42 个配置。')
      expect(content[1].image_url.url).toMatch(/^data:image\/png;base64,/)
      const before = uploads
      await drop([textPath])
      expect(uploads).toBe(before)
      await expect(draft.locator('.chat-document')).toHaveCount(0)
      await finish()
      await expect(page!.locator('.chat-message.user .chat-document')).toHaveCount(1)
      response = undefined
      await page!.getByRole('button', { name: '重试回复', exact: true }).click()
      await expect.poll(() => Boolean(response)).toBe(true)
      expect(body.messages.at(-1).content[0].text).toContain('请保留 42 个配置。')
      await finish()
    })
    await test.step('文本模型可加文档；混合批次保留成功项，图片和二进制明确报错', async () => {
      await page!.getByRole('button', { name: '新建会话', exact: true }).click()
      vision = false
      await page!.getByRole('button', { name: '刷新状态', exact: true }).click()
      await drop([textPath, imagePath, badPath])
      await expect(draft.locator('.chat-document')).toHaveCount(1)
      await expect(draft.locator('.error-notice')).toContainText('当前模型未启用图片输入')
      await expect(draft.locator('.error-notice')).toContainText('不支持.bin')
      await expect(draft.locator('.attachment')).toHaveCount(0)
      await expect(page!.getByRole('button', { name: '发送消息', exact: true })).toBeEnabled()
      response = undefined
      await page!.getByRole('button', { name: '发送消息', exact: true }).click()
      await expect.poll(() => Boolean(response)).toBe(true)
      expect(body.messages.at(-1).content).toContain('const value = 42')
      await finish()
    })
    await test.step('文件选择入口、移除与非对话页隔离', async () => {
      await page!.locator('input[type=file]').setInputFiles(textPath)
      await expect(draft.locator('.chat-document')).toHaveCount(1)
      await draft.getByRole('button', { name: '移除 项目说明.md', exact: true }).click()
      await expect(draft.locator('.chat-document')).toHaveCount(0)
    })
    await test.step('PDF 和 DOCX 全文提取，文件夹拒绝', async () => {
      await drop([path.join(root, 'tests/fixtures/chat-documents/parameters.pdf'), path.join(root, 'tests/fixtures/chat-documents/parameters.docx')])
      await expect(draft.locator('.chat-document')).toHaveCount(2)
      const pdf = draft.locator('.chat-document').filter({ hasText: 'parameters.pdf' })
      const docx = draft.locator('.chat-document').filter({ hasText: 'parameters.docx' })
      await pdf.locator('summary').click()
      await docx.locator('summary').click()
      await expect(pdf.locator('pre')).toContainText('Final page: keep 42 items.')
      await expect(docx.locator('pre')).toContainText('末尾\t完整\n保留')
      await page!.screenshot({ path: path.join(directory, 'document-text-preview.png') })
      const folder = path.join(directory, 'empty-folder')
      fs.mkdirSync(folder)
      await drop([folder])
      await expect(draft.locator('.error-notice')).toContainText('不支持文件夹')
      await expect(draft.locator('.chat-document')).toHaveCount(2)
      await draft.getByRole('button', { name: '移除 parameters.pdf', exact: true }).click()
      await draft.getByRole('button', { name: '移除 parameters.docx', exact: true }).click()
      await expect(draft.locator('.chat-document')).toHaveCount(0)
    })
    await test.step('非对话页不误添加附件', async () => {
      await page!.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '模型库', exact: true }).click()
      const before = uploads
      const dataTransfer = await page!.evaluateHandle(() => { const data = new DataTransfer(); data.items.add(new File(['不要上传'], 'other.txt', { type: 'text/plain' })); return data })
      await page!.locator('body').dispatchEvent('dragenter', { dataTransfer })
      await page!.locator('body').dispatchEvent('drop', { dataTransfer })
      expect(uploads).toBe(before)
      await expect(overlay).toHaveCount(0)
      expect(page!.url()).toBe(initialUrl)
    })
    expect(errors).toEqual([])
  } finally {
    if (page && !page.isClosed()) await page.screenshot({ path: path.join(directory, 'final.png') }).catch(() => {})
    response?.end()
    if (page && !page.isClosed()) await page.unrouteAll({ behavior: 'wait' })
    await application?.close()
    server.closeAllConnections(); await new Promise<void>(resolve => server.close(() => resolve()))
  }
})
