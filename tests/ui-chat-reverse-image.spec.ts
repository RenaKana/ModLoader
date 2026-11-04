import { test, expect, _electron as electron, type Page } from '@playwright/test'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { pinNavigation } from './navigation'

async function management<T = any>(page: Page, route: string): Promise<T> {
  return page.evaluate(async route => {
    const { baseUrl, token } = await window.modloader!.connection()
    const response = await fetch(baseUrl + route, { headers: { Authorization: `Bearer ${token}` } })
    if (!response.ok) throw new Error(`${route}: ${response.status} ${await response.text()}`)
    return response.json()
  }, route)
}

test('以图搜源：模型按需直接搜索、验证图片ID并取消停止中的请求', async () => {
  test.setTimeout(90_000)
  const root = path.resolve(__dirname, '..')
  const artifacts = process.env.MODLOADER_TEST_ARTIFACT_DIR || path.join(root, '.artifacts/codex/reverse-image-search-direct-20261029')
  fs.mkdirSync(artifacts, { recursive: true })
  const directory = fs.mkdtempSync(path.join(artifacts, 'chat-ui-'))
  const app = await electron.launch({ args: [root], env: { ...process.env, MODLOADER_DATA_DIR: directory, MODLOADER_SENSOR_AUTO_START: '0' }, timeout: 60_000 })
  let page: Page | undefined
  let uploads = 0, modelCalls = 0
  let mode: 'success' | 'invalid' | 'stop' = 'success'
  let pendingSearch = false, releasePendingSearch: (() => void) | undefined
  let pendingSearchSettled = false, lateResponseAttempted = false
  const toolResults: any[] = [], apiFailures: string[] = [], errors: string[] = []
  const reverseRequests: any[] = []
  const modelRequests: any[] = []
  const match = { title: '示例作品', author: '作者甲', similarity: 92.3, thumbnail: '', urls: ['https://example.com/art/42'], index_name: 'Pixiv' }
  try {
    page = await app.firstWindow()
    page.on('pageerror', error => errors.push(error.message))
    page.on('requestfailed', request => {
      if (new URL(request.url()).pathname === '/api/web-reverse-image') apiFailures.push(request.failure()?.errorText || 'failed')
    })
    await pinNavigation(page)
    await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 45_000 })
    let fixtureState: Promise<any> | undefined
    await page.route('**/api/state', async route => {
      fixtureState ??= route.fetch().then(response => response.json())
      const state = await fixtureState
      await route.fulfill({ json: { ...state, instances: [{ id: 'reverse-model', model_id: 'm1', profile_id: 'p1', alias: '搜源测试模型', backend: 'llama_cpp', state: 'ready', capabilities: { input: ['text', 'image'], tools: true }, active_requests: 0 }], settings: { ...state.settings, saucenao_url: 'https://saucenao.example.test' } } })
    })
    await page.route('**/api/instances/reverse-model/telemetry', route => route.fulfill({ json: { groups: [], unavailable: [], live_generation: null } }))
    await page.route('**/api/web-reverse-image', async route => {
      uploads++
      const body = route.request().postDataJSON()
      reverseRequests.push(body)
      expect(body).toMatchObject({ filename: '待查图片.png', image: expect.stringMatching(/^data:image\/png;base64,/) })
      if (mode === 'stop') {
        pendingSearch = true
        await new Promise<void>(resolve => { releasePendingSearch = resolve })
        lateResponseAttempted = true
        try { await route.fulfill({ json: { provider: 'saucenao', elapsed_ms: 42, matches: [match] } }) }
        catch { /* The renderer should have cancelled this request already. */ }
        finally { pendingSearchSettled = true }
        return
      }
      await route.fulfill({ json: { provider: 'saucenao', elapsed_ms: 42, matches: [match] } })
    })
    await page.route('**/api/chat', async route => {
      modelCalls++
      const request = route.request().postDataJSON()
      modelRequests.push(request)
      const definition = request.tools.find((tool: any) => tool.function.name === 'reverse_image_search')
      expect(definition).toBeTruthy()
      expect(definition.function.parameters.properties.image_id).toEqual({ type: 'string' })
      const references = request.messages.filter((message: any) => message.role === 'user' && Array.isArray(message.content))
        .flatMap((message: any) => message.content).filter((part: any) => part.type === 'text' && part.text.startsWith('图片引用：'))
        .map((part: any) => JSON.parse(part.text.slice('图片引用：'.length)))
      const imageId = references.at(-1)?.image_id
      const results = request.messages.filter((message: any) => message.role === 'tool').map((message: any) => JSON.parse(message.content))
      if (results.length) toolResults.push(results.at(-1))
      const callAgain = Boolean(imageId) && !results.length
      const delta = callAgain ? { tool_calls: [{ index: 0, id: `reverse-${modelCalls}`, type: 'function', function: { name: 'reverse_image_search', arguments: JSON.stringify({ image_id: mode === 'invalid' ? 'not-an-attachment' : imageId }) } }] }
        : { content: !imageId ? '请提供图片。' : mode === 'success' ? '来源：[示例作品](https://example.com/art/42)，作者甲，相似度 92.3%。' : '未发送图片。' }
      await route.fulfill({ contentType: 'text/event-stream', body: `data: ${JSON.stringify({ choices: [{ delta, finish_reason: callAgain ? 'tool_calls' : 'stop' }] })}\n\ndata: [DONE]\n\n` })
    })
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await page.getByRole('navigation', { name: '主导航', exact: true }).getByRole('button', { name: '测试对话', exact: true }).click()
    const input = page.getByRole('textbox', { name: '对话消息', includeHidden: true })
    async function send() {
      await page!.locator('input[type=file]').setInputFiles({ name: '待查图片.png', mimeType: 'image/png', buffer: fs.readFileSync(path.join(root, 'tests/fixtures/chat-documents/color-sample.png')) })
      await expect(page!.locator('.attachment-list img')).toHaveCount(1)
      await input.fill('查一下这张图的作者和原始出处。')
      await page!.getByRole('button', { name: '发送消息', exact: true }).click()
    }
    const confirmation = page.getByRole('dialog', { name: '允许以图搜源？', exact: true })

    await test.step('无图时工具定义已固定，不调用搜源', async () => {
      await input.fill('稍后我会发图片。')
      await page!.getByRole('button', { name: '发送消息', exact: true }).click()
      await expect(page!.locator('.chat-message.assistant').last().locator('.message-text')).toHaveText('请提供图片。')
      expect(uploads).toBe(0)
      expect(modelCalls).toBe(1)
    })

    await test.step('模型调用直接上传，匹配回到模型并写入历史', async () => {
      await send()
      await expect(confirmation).toHaveCount(0)
      const assistant = page!.locator('.chat-message.assistant').last()
      await expect(assistant.locator('.message-text')).toContainText('来源：')
      await expect(assistant.locator('.message-text')).toContainText('作者甲')
      await expect(assistant.locator('.message-text')).toContainText('相似度 92.3%')
      expect(uploads).toBe(1)
      expect(reverseRequests).toHaveLength(1)
      expect(reverseRequests[0].image).toMatch(/^data:image\/png;base64,/)
      expect(reverseRequests[0].filename).toBe('待查图片.png')
      expect(modelCalls).toBe(3)
      expect(modelRequests[1].tools).toEqual(modelRequests[0].tools)
      expect(modelRequests[1].messages.slice(0, modelRequests[0].messages.length)).toEqual(modelRequests[0].messages)
      expect(toolResults.at(-1)).toMatchObject({ image_name: '待查图片.png', matches: [expect.objectContaining({ author: '作者甲', urls: ['https://example.com/art/42'] })] })
      await expect(page!.locator('.chat-save-status')).toHaveText('已保存')
      const listing = await management<any>(page!, '/api/conversations')
      const saved = await management<any>(page!, `/api/conversations/${listing.active_id}`)
      const message = saved.messages.at(-1)
      const savedTool = message.activity.filter((event: any) => event.kind === 'tool').at(-1)
      expect(savedTool.status).toBe('completed')
      expect(JSON.parse(savedTool.result)).toMatchObject({ image_name: '待查图片.png', matches: [expect.objectContaining({ author: '作者甲', urls: ['https://example.com/art/42'] })] })
      expect(message.toolResults).toContainEqual(expect.objectContaining({ content: expect.stringContaining('作者甲') }))
      await page!.screenshot({ path: path.join(directory, 'reverse-direct-result.png') })
    })

    await test.step('再发图片保持工具与旧消息前缀，新增图片 ID 仍可搜源', async () => {
      const previous = modelRequests[1]
      await send()
      await expect.poll(() => modelCalls).toBe(5)
      await expect(page!.locator('.chat-message.assistant').last().locator('.message-text')).toContainText('来源：')
      const current = modelRequests[3]
      expect(current.tools).toEqual(previous.tools)
      expect(current.messages.slice(0, previous.messages.length)).toEqual(previous.messages)
      const imageReferences = current.messages.filter((message: any) => message.role === 'user' && Array.isArray(message.content))
        .flatMap((message: any) => message.content).filter((part: any) => part.type === 'text' && part.text.startsWith('图片引用：'))
      expect(imageReferences).toHaveLength(2)
      expect(imageReferences[0].text).not.toBe(imageReferences[1].text)
      expect(uploads).toBe(2)
      expect(modelCalls).toBe(5)
    })

    await test.step('无效图片ID不会上传', async () => {
      mode = 'invalid'
      await page!.getByRole('button', { name: '新建会话', exact: true }).click()
      await send()
      await expect(confirmation).toHaveCount(0)
      await expect(page!.locator('.chat-message.assistant').last().locator('.message-text')).toHaveText('未发送图片。')
      expect(uploads).toBe(2)
      expect(reverseRequests).toHaveLength(2)
      expect(toolResults.at(-1)).toMatchObject({ error: { message: expect.stringContaining('未找到指定的对话图片') } })
    })

    await test.step('停止生成取消进行中的搜源请求，迟到匹配不成为成功', async () => {
      mode = 'stop'
      await page!.getByRole('button', { name: '新建会话', exact: true }).click()
      await send()
      await expect.poll(() => pendingSearch).toBe(true)
      const resultsBeforeStop = toolResults.length
      const assistant = page!.locator('.chat-message.assistant').last()
      await expect(page!.getByRole('button', { name: '停止', exact: true })).toBeVisible()
      await page!.getByRole('button', { name: '停止', exact: true }).click()
      await expect(assistant.locator('.message-label')).toContainText('已停止')
      await expect(assistant.locator('.chat-activity-tool').last()).toHaveAttribute('data-status', 'cancelled')
      await expect.poll(() => apiFailures.length).toBe(1)
      expect(modelCalls).toBe(8)
      expect(uploads).toBe(3)
      expect(toolResults).toHaveLength(resultsBeforeStop)
      releasePendingSearch?.()
      await expect.poll(() => pendingSearchSettled).toBe(true)
      expect(lateResponseAttempted).toBe(true)
      await expect(assistant).not.toContainText('示例作品')
      await expect(assistant.locator('.chat-activity-tool').last()).toHaveAttribute('data-status', 'cancelled')
    })
    expect(errors).toEqual([])
  } finally {
    releasePendingSearch?.()
    if (page && !page.isClosed()) { await page.screenshot({ path: path.join(directory, 'final.png') }); await page.unrouteAll({ behavior: 'wait' }) }
    await app.close()
  }
})
