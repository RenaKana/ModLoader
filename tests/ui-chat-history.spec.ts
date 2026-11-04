import { test, expect, _electron as electron, type ElectronApplication, type Page } from '@playwright/test'
import { createServer, type ServerResponse } from 'node:http'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { pinNavigation } from './navigation'
import { chatCapsule, chatModel, disableWebSearch, selectChatModel } from './chat-controls'

interface ChatRequest {
  body: { model: string; messages: Array<{ role: string; content: unknown; reasoning_content?: string }>; [key: string]: unknown }
  response: ServerResponse
  clientRequestId?: string
  clientCallId?: string
}
interface Conversation {
  id: string
  title: string
  selection: { profile_id: string; model_id: string; alias: string } | null
  draft: { text: string; images: Array<{ id: string; name: string; size: number; url: string }> }
  settings: { system: string; temperature?: number; maxTokens?: number; stream: boolean; toolsDraft: string }
  messages: Array<{ id: string; role: string; text: string; status: string; reasoning?: string; stopped?: boolean; error?: string; images: unknown[]; tools?: unknown[]; usage?: object; timings?: object; firstOutput?: number; elapsed?: number; finishReason?: string; createdAt: number; responseAt?: number }>
}
interface ConversationList { items: Array<{ id: string; title: string }>; active_id: string | null }

const root = path.resolve(__dirname, '..')
const textInstance = { id: 'history-text', profile_id: 'history-text-profile', model_id: 'history-text-model', profile_revision: 1, alias: '历史文本模型', backend: 'llama_cpp', runtime_id: 'history-runtime', state: 'ready', active_requests: 0, capabilities: { input: ['text'], tools: true } }
const imageInstance = { ...textInstance, id: 'history-image', profile_id: 'history-image-profile', model_id: 'history-image-model', alias: '历史图片模型', capabilities: { input: ['text', 'image'], tools: true } }

async function management<T>(page: Page, target: string, method = 'GET', body?: unknown): Promise<T> {
  return page.evaluate(async ({ target, method, body }) => {
    const connection = await (window as any).modloader.connection()
    const response = await fetch(`${connection.baseUrl}${target}`, { method, headers: { Authorization: `Bearer ${connection.token}`, 'Content-Type': 'application/json' }, ...(body === undefined ? {} : { body: JSON.stringify(body) }) })
    if (!response.ok) throw new Error(`${method} ${target}: ${response.status} ${await response.text()}`)
    return response.status === 204 ? null : response.json()
  }, { target, method, body })
}

async function history(page: Page) {
  const panel = page.getByRole('complementary', { name: '对话历史', exact: true })
  if (!await panel.isVisible()) await page.getByRole('button', { name: '对话历史', exact: true }).click()
  await expect(panel).toBeVisible()
  return panel
}

async function closeHistory(page: Page) {
  const panel = page.getByRole('complementary', { name: '对话历史', exact: true })
  if (await panel.isVisible()) await page.getByRole('button', { name: '对话历史', exact: true }).click()
}

function trackHistoryReorders(page: Page) {
  const requests: unknown[] = []
  page.on('request', request => {
    if (request.method() !== 'PATCH' || !request.url().includes('/api/conversations/')) return
    const body = request.postDataJSON()
    if (body && Object.hasOwn(body, 'move_before')) requests.push(body)
  })
  return requests
}

async function editComposer(page: Page) {
  const preview = page.getByRole('region', { name: '输入内容预览', exact: true })
  if (await preview.isVisible()) await preview.click({ position: { x: 8, y: 8 } })
  return page.getByRole('textbox', { name: '对话消息', exact: true })
}

async function saved(page: Page) {
  const empty = await page.locator('.chat-message').count() === 0 && !(await page.locator('.chat-composer > textarea').inputValue()).trim() && await page.locator('.attachment').count() === 0
  if (empty) await expect(page.locator('.chat-save-status')).toHaveCount(0)
  else await expect(page.getByText('已保存', { exact: true })).toBeVisible()
  await expect(page.getByText('未保存', { exact: true })).toHaveCount(0)
}

async function selectMenu(application: ElectronApplication, menu: string, item: string) {
  await application.evaluate(({ Menu }, { menu, item }) => {
    Menu.getApplicationMenu()!.items.find(entry => entry.label === menu)!.submenu!.items.find(entry => entry.label === item)!.click()
  }, { menu, item })
}

async function fixture(includeLibrary = false) {
  const directory = fs.mkdtempSync(path.join(process.env.MODLOADER_TEST_ARTIFACT_DIR || path.join(root, 'work'), 'ui-chat-history-'))
  const library = includeLibrary ? {
    models: [textInstance, imageInstance].map(instance => ({ id: instance.model_id, name: instance.alias, path: path.join(directory, `${instance.model_id}.gguf`), format: 'gguf', size_bytes: 0, files: [], modalities: instance.capabilities.input, metadata: {}, source: { kind: 'fixture' }, managed: false, warnings: [] })),
    profiles: [textInstance, imageInstance].map(instance => ({ id: instance.profile_id, model_id: instance.model_id, name: `${instance.alias}配置`, backend: instance.backend, runtime_id: instance.runtime_id, alias: instance.alias, parameters: {}, generation: {}, extra_args: [], revision: instance.profile_revision })),
  } : {}
  const requests: ChatRequest[] = []
  const errors: string[] = []
  let application: ElectronApplication | undefined
  let fontScale = 1
  let instances = [textInstance, imageInstance]
  const server = createServer(async (request, response) => {
    response.setHeader('Access-Control-Allow-Origin', '*')
    response.setHeader('Access-Control-Allow-Headers', '*')
    if (request.method === 'OPTIONS') { response.end(); return }
    let raw = ''
    for await (const chunk of request) raw += chunk
    const body = JSON.parse(raw)
    response.setHeader('Content-Type', body.stream ? 'text/event-stream' : 'application/json')
    response.flushHeaders()
    requests.push({ body, response, clientRequestId: request.headers['x-modloader-request-id'] as string | undefined, clientCallId: request.headers['x-modloader-call-id'] as string | undefined })
  })
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const address = server.address() as { port: number }
  return {
    directory, requests, errors,
    setFontScale(value: number) { fontScale = value },
    setInstances(value: Array<typeof textInstance>) { instances = value },
    async launch() {
      const executablePath = process.env.MODLOADER_TEST_EXECUTABLE
      application = await electron.launch({ ...(executablePath ? { executablePath, args: [] } : { args: [root] }), env: { ...process.env, MODLOADER_DATA_DIR: directory }, timeout: 60_000 })
      const page = await application.firstWindow()
      page.on('pageerror', error => errors.push(error.message))
      await pinNavigation(page)
      await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
      await page.route('**/api/state', async route => {
        const original = await route.fetch()
        const state = await original.json()
        await route.fulfill({ response: original, json: { ...state, ...library, settings: { ...state.settings, font_scale: fontScale }, instances } })
      })
      await page.route('**/api/chat', route => route.continue({ url: `http://127.0.0.1:${address.port}/api/chat` }))
      await page.route('**/api/instances/*/telemetry', route => route.fulfill({ json: { groups: [], unavailable: [], live_generation: { client_request_id: requests.at(-1)?.clientRequestId, tokens_per_second: 24, window_s: 2, available_for_chat: true } } }))
      await page.getByRole('button', { name: '刷新状态', exact: true }).click()
      await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '测试对话', exact: true }).click()
      await expect(page.getByRole('textbox', { name: '对话消息', exact: true, includeHidden: true })).toBeEnabled()
      await disableWebSearch(page)
      return { application, page }
    },
    async send(page: Page, text: string) {
      await disableWebSearch(page)
      const before = requests.length
      const preview = page.getByRole('region', { name: '输入内容预览', exact: true })
      if (await preview.isVisible()) await preview.click({ position: { x: 8, y: 8 } })
      await (await editComposer(page)).fill(text)
      await page.getByRole('button', { name: '发送消息', exact: true }).click()
      await expect.poll(() => requests.length).toBe(before + 1)
      return requests[before]
    },
    async quit() {
      const current = application!
      // Stop fixture-side route.fetch calls before the native path shuts down
      // the real core; terminal generation still flushes through its real API.
      await Promise.all(current.windows().map(page => page.unrouteAll({ behavior: 'wait' })))
      const closed = current.waitForEvent('close', { timeout: 30_000 })
      await selectMenu(current, '文件', '退出 ModLoader')
      await closed
      application = undefined
    },
    async cleanup() {
      requests.forEach(request => request.response.end())
      if (application) {
        await Promise.all(application.windows().map(page => page.unrouteAll({ behavior: 'wait' })))
        await application.close()
      }
      server.closeAllConnections()
      await new Promise<void>(resolve => server.close(() => resolve()))
      console.log('Chat history fixture and screenshots:', directory)
    },
  }
}

function event(request: ChatRequest, chunk: unknown) { request.response.write(`data: ${JSON.stringify(chunk)}\n\n`) }
function delta(request: ChatRequest, content: unknown) { event(request, { choices: [{ delta: content }] }) }
async function finish(page: Page, request: ChatRequest, text: string) {
  delta(request, { content: text })
  event(request, { choices: [{ delta: {}, finish_reason: 'stop' }] })
  event(request, { choices: [], usage: { prompt_tokens: 18, completion_tokens: 32 }, timings: { predicted_per_second: 24 } })
  request.response.end('data: [DONE]\n\n')
  await expect(page.getByRole('button', { name: '停止生成', exact: true })).toHaveCount(0)
  await saved(page)
}

test('内置回复保存实际后端，实例卸载后请求历史仍显示 strata', async () => {
  test.setTimeout(60_000)
  const context = await fixture()
  context.setInstances([{ ...textInstance, backend: 'strata' }])
  try {
    const { page } = await context.launch()
    await finish(page, await context.send(page, '保存本次实际使用的后端。'), '完成。')
    const id = (await management<ConversationList>(page, '/api/conversations')).active_id!
    const detail = await management<any>(page, `/api/conversations/${id}`)
    expect(detail.messages.at(-1).backend).toBe('strata')
    context.setInstances([])
    await page.reload(); await pinNavigation(page)
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '运行监控', exact: true }).click()
    await expect(page.locator('.monitor-ledger-group-meta')).toContainText(['strata · 1 次请求'])
    await page.screenshot({ path: path.join(context.directory, 'internal-backend-history.png') })
    expect(context.errors).toEqual([])
  } finally { await context.cleanup() }
})

test('清空兼容原有 selection 字段并重置标题，保留并行实例、参数和草稿', async () => {
  test.setTimeout(60_000)
  const context = await fixture()
  const parallel = { ...textInstance, id: 'history-text-2', alias: `${textInstance.alias}--2` }
  context.setInstances([textInstance, parallel])
  try {
    const { page } = await context.launch()
    const seedId = crypto.randomUUID(), title = '有 8 个任务和 8 台机器，每个任务需要分配给一台兼容的机器。'
    await management(page, '/api/conversations', 'POST', { id: seedId, title, selection: { profile_id: textInstance.profile_id, model_id: textInstance.model_id, alias: textInstance.alias }, draft: { text: '', images: [] }, settings: { system: '', stream: true, toolsDraft: '', webSearch: false } })
    await management(page, `/api/conversations/${seedId}`, 'PATCH', { activate: true, messages: [
      { id: 'u', role: 'user', text: title, images: [], createdAt: Date.now(), status: 'completed' },
      { id: 'a', role: 'assistant', text: '保留原始答案，直到确认清空。', images: [], createdAt: Date.now(), status: 'completed' },
    ] })
    // An already-running core accepts the original persisted selection contract.
    await page.route('**/api/conversations**', async route => {
      if (['POST', 'PATCH'].includes(route.request().method())) {
        const selection = route.request().postDataJSON()?.selection
        if (selection && Object.keys(selection).some(key => !['profile_id', 'model_id', 'alias'].includes(key))) {
          await route.fulfill({ status: 400, json: { detail: 'selection 字段不完整或包含不支持的字段' } }); return
        }
      }
      await route.continue()
    })
    await page.reload(); await pinNavigation(page)
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '测试对话', exact: true }).click()
    await selectChatModel(page, parallel.id)
    await page.getByRole('button', { name: '对话参数', exact: true }).click()
    await page.getByRole('textbox', { name: '系统提示词', exact: true }).fill('保留当前参数')
    await page.getByRole('button', { name: '收起对话参数', exact: true }).click()
    await (await editComposer(page)).fill('清空后继续编辑的草稿')
    await page.getByRole('button', { name: '清空当前对话', exact: true }).click()
    await page.getByRole('dialog').getByRole('button', { name: '清空对话', exact: true }).click()
    await expect(page.getByRole('dialog')).toBeHidden({ timeout: 5000 })
    await expect(page.locator('.chat-message')).toHaveCount(0)
    await expect(page.locator('.chat-session-title')).toHaveText('新会话')
    await saved(page)
    const id = (await management<ConversationList>(page, '/api/conversations')).active_id!
    const cleared = await management<Conversation>(page, `/api/conversations/${id}`)
    expect(cleared).toMatchObject({ title: '新会话', messages: [], draft: { text: '清空后继续编辑的草稿' }, settings: { system: '保留当前参数' }, selection: { profile_id: parallel.profile_id, alias: parallel.alias } })
    await page.reload(); await pinNavigation(page)
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '测试对话', exact: true }).click()
    await expect(page.locator('.chat-session-title')).toHaveText('新会话')
    await expect(chatCapsule(page)).toContainText(parallel.alias)
    const next = await context.send(page, '清空后的新标题')
    expect(next.body.instance_id).toBe(parallel.id)
    await finish(page, next, '新的回答。')
    await expect(page.locator('.chat-session-title')).toHaveText('清空后的新标题')
    expect((await management<Conversation>(page, `/api/conversations/${id}`)).title).toBe('清空后的新标题')
    await page.screenshot({ path: path.join(context.directory, 'cleared-and-reused.png') })
    expect(context.errors).toEqual([])
  } finally { await context.cleanup() }
})

test('历史标题悬停安全格式化、完整换行和视口内滚动，普通提示保持原文', async () => {
  test.setTimeout(120_000)
  const context = await fixture()
  context.setInstances([textInstance])
  try {
    const { page, application } = await context.launch()
    const original = String.raw`设矩阵 \(M\) 的尺寸为 \(14\times14\)，进行 ` + '**低秩分解** 和 _投影_，使用 `rank`，~~旧方法~~；' + '完整标题应当换行显示，不能只显示省略后的部分。'.repeat(5)
    await finish(page, await context.send(page, `${original}\n\n${'保留正文段落。\n\n'.repeat(35)}`), '保留原始标题。')
    const id = (await management<ConversationList>(page, '/api/conversations')).active_id!
    const panel = await history(page)
    let row = panel.getByRole('button', { name: original, exact: true })
    const tooltip = page.getByRole('tooltip')
    const host = page.locator('.app-tooltip')
    const inWindow = async () => {
      const bounds = (await tooltip.boundingBox())!
      const viewport = await page.evaluate(() => ({ width: innerWidth, height: innerHeight }))
      expect(bounds.x).toBeGreaterThanOrEqual(8)
      expect(bounds.y).toBeGreaterThanOrEqual(8)
      expect(bounds.x + bounds.width).toBeLessThanOrEqual(viewport.width - 7)
      expect(bounds.y + bounds.height).toBeLessThanOrEqual(viewport.height - 7)
    }
    await row.hover()
    await expect(tooltip).toBeVisible()
    await expect(tooltip).toHaveCSS('opacity', '1')
    await expect(host).toHaveAttribute('data-tooltip-format', 'chat-title')
    await expect(tooltip.locator('.katex')).toHaveCount(2)
    await expect(tooltip.locator('annotation')).toHaveText(['M', String.raw`14\times14`])
    await expect(tooltip.locator('strong')).toHaveText('低秩分解')
    await expect(tooltip.locator('em')).toHaveText('投影')
    await expect(tooltip.locator('code')).toHaveText('rank')
    await expect(tooltip.locator('del')).toHaveText('旧方法')
    await expect(tooltip.locator('.chat-title')).toHaveCSS('white-space', 'normal')
    expect(await tooltip.evaluate(element => element.getBoundingClientRect().height)).toBeGreaterThan(60)
    await expect(row).toHaveAttribute('aria-describedby', await host.getAttribute('id') as string)
    await inWindow()
    await page.screenshot({ path: path.join(context.directory, 'title-tooltip-formatted.png') })
    await page.keyboard.press('Escape')
    await expect(tooltip).toBeHidden()
    await expect(row).not.toHaveAttribute('aria-describedby')
    await page.mouse.move(900, 500)
    await panel.getByRole('button', { name: '删除当前会话', exact: true }).focus()
    await page.keyboard.press('Tab')
    await expect(row).toBeFocused()
    await expect(tooltip.locator('.katex')).toHaveCount(2)
    await page.keyboard.press('Escape')
    await expect(row).toBeFocused()
    await expect(row).not.toHaveAttribute('aria-describedby')
    await page.locator('.chat-session-title').hover()
    await expect(tooltip).toHaveText('双击重命名会话')
    await expect(tooltip.locator('.chat-title, .katex')).toHaveCount(0)
    await expect(host).not.toHaveAttribute('data-tooltip-format')
    await page.screenshot({ path: path.join(context.directory, 'title-tooltip-plain-operation.png') })

    const longFormula = String.raw`\(\underbrace{\mathtt{` + 'variable'.repeat(30) + String.raw`}}_{\text{完整长公式}}\)`
    const longTitle = String.raw`矩阵 \(M\) 的 \(14\times14\) 分块。` + '所有标题文字都应能完整阅读，长内容在窗口内滚动。'.repeat(55) + longFormula + '结束。'
    await page.locator('.chat-session-title').dblclick()
    await page.getByRole('textbox', { name: '会话名称', exact: true }).fill(longTitle)
    await page.getByRole('button', { name: '保存名称', exact: true }).click()
    await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(940, 620))
    context.setFontScale(1.3)
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    row = panel.getByRole('button', { name: longTitle, exact: true })
    await row.hover()
    await expect(tooltip.locator('.katex')).toHaveCount(3)
    await expect(tooltip).toHaveCSS('opacity', '1')
    await inWindow()
    expect(await tooltip.evaluate(element => element.scrollHeight - element.clientHeight)).toBeGreaterThan(100)
    await tooltip.hover(); await page.mouse.wheel(0, 10_000)
    await expect.poll(() => tooltip.evaluate(element => element.scrollTop)).toBeGreaterThan(100)
    await expect(tooltip).toBeVisible()
    expect(await tooltip.evaluate(element => element.scrollWidth - element.clientWidth)).toBeGreaterThan(100)
    await page.screenshot({ path: path.join(context.directory, 'title-tooltip-long-narrow.png') })
    await page.mouse.wheel(600, 0)
    await expect.poll(() => tooltip.evaluate(element => element.scrollLeft)).toBeGreaterThan(100)
    await expect(tooltip).toBeVisible()
    await page.locator('.chat-messages').evaluate(element => { element.scrollTop = 0 })
    await expect(tooltip).toBeHidden()
    await expect(row).not.toHaveAttribute('aria-describedby')

    const resourceRequests: string[] = []
    await page.route('**/title-tooltip-image-probe*', async route => { resourceRequests.push(route.request().url()); await route.abort() })
    await page.evaluate(() => { (window as any).titleTooltipAttack = 'clean' })
    const safeTitle = '**安全标题** [参考](https://example.com/) ![图示](https://example.invalid/title-tooltip-image-probe.png) <button onclick="top.titleTooltipAttack=\'event\'">按钮文字</button> ' + String.raw`\(14\times14\)`
    await page.locator('.chat-session-title').dblclick()
    await page.getByRole('textbox', { name: '会话名称', exact: true }).fill(safeTitle)
    await page.getByRole('button', { name: '保存名称', exact: true }).click()
    row = panel.getByRole('button', { name: safeTitle, exact: true })
    await row.hover()
    await expect(tooltip.locator('.katex')).toHaveCount(1)
    await expect(tooltip).toContainText('参考')
    await expect(tooltip).toContainText('图示')
    await expect(tooltip.locator('a, img, button, input, iframe, script, .message-code-block')).toHaveCount(0)
    expect(resourceRequests).toEqual([])
    expect(await page.evaluate(() => (window as any).titleTooltipAttack)).toBe('clean')
    expect((await management<Conversation>(page, `/api/conversations/${id}`)).title).toBe(safeTitle)
    expect(context.errors).toEqual([])
  } finally { await context.cleanup() }
})

test('会话标题支持行内公式和格式，保留原文重命名与历史操作', async () => {
  test.setTimeout(120_000)
  const context = await fixture()
  context.setInstances([textInstance])
  try {
    const { page, application } = await context.launch()
    const original = String.raw`矩阵 \(M\) 的 \(14\times14\) 分块：` + '**低秩**、_投影_、`rank`、~~旧式~~'
    await finish(page, await context.send(page, `${original}\n\n正文不会进入自动标题。`), '保留正文。')
    const id = (await management<ConversationList>(page, '/api/conversations')).active_id!
    expect((await management<Conversation>(page, `/api/conversations/${id}`)).title).toBe(original)
    const title = page.locator('.chat-session-title')
    const titleContent = title.locator('.chat-title')
    let panel = await history(page)
    let row = panel.getByRole('button', { name: original, exact: true })
    for (const content of [titleContent, row.locator('.chat-title')]) {
      await expect(content.locator('.katex')).toHaveCount(2)
      await expect(content.locator('annotation[encoding="application/x-tex"]')).toHaveText(['M', String.raw`14\times14`])
      await expect(content.locator('strong')).toHaveText('低秩')
      await expect(content.locator('em')).toHaveText('投影')
      await expect(content.locator('code')).toHaveText('rank')
      await expect(content.locator('del')).toHaveText('旧式')
      await expect(content.locator('p, pre, a, img, button, input, iframe, .katex-display')).toHaveCount(0)
      expect(await content.locator('strong').evaluate(element => getComputedStyle(element).display)).toBe('inline')
    }
    const screenshot = async (name: string) => {
      const bounds = (await page.locator('.chat-layout').boundingBox())!
      await page.screenshot({ path: path.join(context.directory, name), clip: { ...bounds, height: Math.min(bounds.height, 240) } })
    }
    await screenshot('chat-title-inline-formats.png')
    await title.locator('.katex').first().dblclick()
    const rename = page.getByRole('textbox', { name: '会话名称', exact: true })
    await expect(rename).toHaveValue(original)
    await rename.fill('取消的 **标题**')
    await page.getByRole('button', { name: '取消重命名', exact: true }).click()
    await expect(titleContent.locator('.katex')).toHaveCount(2)
    expect((await management<Conversation>(page, `/api/conversations/${id}`)).title).toBe(original)

    const resourceRequests: string[] = []
    await page.route('**/title-image-probe*', async route => { resourceRequests.push(route.request().url()); await route.abort() })
    await page.evaluate(() => { (window as any).titlePreviewAttack = 'clean' })
    const safeTitle = '**新标题** [**参考**](https://example.com/) ' + String.raw`\(14\times14\) $x^2$ ` + '![图示](https://example.invalid/title-image-probe.png) <button onclick="top.titlePreviewAttack=\'event\'">按钮文字</button>'
    await title.dblclick(); await rename.fill(safeTitle)
    await page.getByRole('button', { name: '保存名称', exact: true }).click()
    row = panel.getByRole('button', { name: safeTitle, exact: true })
    await expect(row).toBeVisible()
    await expect(title).toHaveAttribute('aria-label', safeTitle)
    for (const content of [titleContent, row.locator('.chat-title')]) {
      await expect(content).toContainText('参考')
      await expect(content).toContainText('图示')
      await expect(content.locator('a, img, button, input, iframe, script, .message-code-block')).toHaveCount(0)
    }
    expect(resourceRequests).toEqual([])
    expect(await page.evaluate(() => (window as any).titlePreviewAttack)).toBe('clean')
    await page.getByRole('button', { name: '新建会话', exact: true }).click()
    const secondTitle = String.raw`另一会话 \(N\)`
    await finish(page, await context.send(page, secondTitle), '另一条回答。')
    await row.locator('.chat-title strong').filter({ hasText: '参考' }).click()
    await expect(title).toHaveAttribute('aria-label', safeTitle)
    await expect(page.locator('.chat-message.assistant')).toContainText('保留正文')
    await panel.getByRole('button', { name: '置顶会话', exact: true }).click()
    await expect(row.locator('.chat-history-pin')).toBeVisible()
    expect((await management<Conversation>(page, `/api/conversations/${id}`)).title).toBe(safeTitle)
    await page.reload(); await pinNavigation(page)
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '测试对话', exact: true }).click()
    panel = await history(page)
    row = panel.getByRole('button', { name: safeTitle, exact: true })
    await expect(titleContent.locator('.katex')).toHaveCount(2)
    await expect(row.locator('.chat-title .katex')).toHaveCount(2)
    await title.dblclick(); await expect(rename).toHaveValue(safeTitle)
    const longTitle = String.raw`完整标题 \(M\)：` + '保留所有标题文字 '.repeat(30) + String.raw`末尾 $$\frac{a}{b}$$`
    await rename.fill(longTitle); await page.getByRole('button', { name: '保存名称', exact: true }).click()
    row = panel.getByRole('button', { name: longTitle, exact: true })
    expect((await management<Conversation>(page, `/api/conversations/${id}`)).title).toBe(longTitle)
    await expect(titleContent.locator('.katex')).toHaveCount(2)
    await expect(titleContent.locator('.katex-display')).toHaveCount(0)
    await expect(titleContent.locator('annotation').last()).toHaveText(String.raw`\frac{a}{b}`)
    await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(940, 760))
    context.setFontScale(1.3)
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await expect.poll(() => page.evaluate(() => document.documentElement.style.getPropertyValue('--font-scale'))).toBe('1.3')
    for (const container of [title, row.locator('.chat-history-title')]) {
      const geometry = await container.evaluate(element => {
        const style = getComputedStyle(element)
        return { height: element.getBoundingClientRect().height, line: parseFloat(style.lineHeight) + parseFloat(style.paddingTop) + parseFloat(style.paddingBottom), overflow: element.scrollWidth - element.clientWidth, ellipsis: style.textOverflow }
      })
      expect(geometry.height).toBeLessThanOrEqual(geometry.line + 1)
      expect(geometry.overflow).toBeGreaterThan(0)
      expect(geometry.ellipsis).toBe('ellipsis')
    }
    expect(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1)
    await screenshot('chat-title-narrow-ellipsis.png')
    await title.dblclick(); await expect(rename).toHaveValue(longTitle)
    await rename.press('Escape')
    expect(resourceRequests).toEqual([])
    expect(context.errors).toEqual([])
  } finally { await context.cleanup() }
})

test('输入框与历史编辑支持格式预览、复制原文和独立草稿', async () => {
  test.setTimeout(120_000)
  const context = await fixture()
  context.setInstances([imageInstance])
  try {
    let { page, application } = await context.launch()
    let input = page.getByRole('textbox', { name: '对话消息', exact: true, includeHidden: true })
    const png = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII='
    const source = `  ## 输入格式预览\n\n这是 **粗体** 和 $x^2$。\n\n第一行\n第二行\n\n\`\`\`python\ndef preview():\n    return "${'wide_content_'.repeat(100)}"\n\`\`\`\n\n|格式|内容|\n|---|---|\n|表格|保持原文|\n\n\`\`\`html\n<section><h3>输入安全预览</h3><script>top.inputPreviewAttack="script"</script><p onclick="top.inputPreviewAttack='event'">安全正文</p></section>\n\`\`\`\n\n${Array.from({ length: 25 }, (_, index) => `第 ${index + 1} 段：长草稿可以滚动预览。\n\n`).join('')}### 预览尾部\n `
    await expect(page.getByRole('button', { name: '预览输入格式', exact: true })).toHaveCount(0)
    await expect(page.getByRole('button', { name: '复制输入原文', exact: true })).toBeDisabled()
    const initialHeight = (await input.boundingBox())!.height
    await input.fill(source)
    await input.evaluate(element => (element as HTMLTextAreaElement).setSelectionRange(3, 8, 'backward'))
    await page.evaluate(() => { (window as any).inputPreviewAttack = 'clean' })
    await input.blur()
    let preview = page.getByRole('region', { name: '输入内容预览', exact: true })
    const showPreview = async () => { if (await input.isVisible()) { await input.focus(); await input.blur() }; await expect(preview).toBeVisible() }
    const editDraft = async () => { if (await preview.isVisible()) await preview.click({ position: { x: 8, y: 8 } }); await expect(input).toBeVisible() }
    await expect(input).toBeHidden()
    await expect(preview.getByRole('heading', { name: '输入格式预览', exact: true })).toBeVisible()
    await expect(preview.locator('.katex')).toHaveCount(1)
    await expect(preview.locator('code .hljs-keyword').first()).toBeVisible()
    expect(await preview.locator('p').filter({ hasText: '第一行' }).evaluate(element => {
      const first = document.createRange(), second = document.createRange()
      first.setStart(element.firstChild!, 0); first.setEnd(element.firstChild!, 3)
      second.setStart(element.firstChild!, 4); second.setEnd(element.firstChild!, 7)
      return second.getBoundingClientRect().top - first.getBoundingClientRect().top
    })).toBeGreaterThan(15)
    const code = preview.locator('.message-code-block[data-language="python"] pre')
    await code.hover(); await page.mouse.wheel(600, 0)
    await expect.poll(() => code.evaluate(element => element.scrollLeft)).toBeGreaterThan(100)
    await page.locator('.chat-composer-area').screenshot({ path: path.join(context.directory, 'composer-code-preview.png') })
    const html = preview.locator('.message-code-block[data-language="html"]')
    await expect(html.locator('iframe')).toHaveAttribute('sandbox', '')
    await expect(html.frameLocator('iframe').locator('script, [onclick]')).toHaveCount(0)
    expect(await page.evaluate(() => (window as any).inputPreviewAttack)).toBe('clean')
    await preview.hover(); await page.mouse.wheel(0, 20_000)
    await expect(preview.getByRole('heading', { name: '预览尾部', exact: true })).toBeInViewport()
    await expect(page.getByRole('button', { name: '发送消息', exact: true })).toBeInViewport()
    await page.getByRole('button', { name: '复制输入原文', exact: true }).click()
    expect(await application.evaluate(({ clipboard }) => clipboard.readText())).toBe(source)
    await expect(page.getByText('原文已复制', { exact: true })).toBeVisible()
    await editDraft()
    await expect(input).toBeFocused()
    await expect(input).toHaveValue(source)
    expect(await input.evaluate(element => { const input = element as HTMLTextAreaElement; return [input.selectionStart, input.selectionEnd, input.selectionDirection] })).toEqual([3, 8, 'backward'])
    expect(await input.evaluate(element => element.getBoundingClientRect().height)).toBeGreaterThan(100)
    await page.getByRole('button', { name: '关闭通知', exact: true }).click()
    await editDraft()
    await input.press('Control+Home')
    await page.locator('.chat-composer-area').screenshot({ path: path.join(context.directory, 'composer-original-editor.png') })
    await input.press('Control+End'); await input.press('Shift+Enter')
    const draft = `${source}\n`
    await expect(input).toHaveValue(draft)
    await input.dispatchEvent('keydown', { key: 'Enter', code: 'Enter', isComposing: true, bubbles: true, cancelable: true })
    expect(context.requests).toHaveLength(0)
    const pasteImage = async (target: ReturnType<Page['locator']>, name: string) => target.evaluate((element, { png, name }) => {
      const data = new DataTransfer()
      data.items.add(new File([Uint8Array.from(atob(png), character => character.charCodeAt(0))], name, { type: 'image/png' }))
      element.dispatchEvent(new ClipboardEvent('paste', { clipboardData: data, bubbles: true, cancelable: true }))
    }, { png, name })
    await pasteImage(input, 'raw-paste.png')
    await expect(page.locator('.attachment')).toHaveCount(1)
    await showPreview()
    await pasteImage(preview, 'preview-paste.png')
    await expect(page.locator('.attachment')).toHaveCount(2)
    await page.locator('.chat-session-title').dblclick()
    await page.getByRole('textbox', { name: '会话名称', exact: true }).fill('输入预览草稿')
    await page.getByRole('button', { name: '保存名称', exact: true }).click()
    const firstId = (await management<ConversationList>(page, '/api/conversations')).active_id!
    await page.getByRole('button', { name: '新建会话', exact: true }).click()
    await expect(input).toBeVisible(); await expect(input).toHaveValue('')
    await expect(preview).toHaveCount(0)
    await input.fill('另一个会话\n保持原文')
    await showPreview()
    await saved(page)
    const secondId = (await management<ConversationList>(page, '/api/conversations')).active_id!
    const panel = await history(page)
    await panel.getByRole('button', { name: '输入预览草稿', exact: true }).click()
    await expect(input).toBeHidden(); await expect(input).toHaveValue(draft)
    await expect(preview).toBeVisible()
    expect((await management<Conversation>(page, `/api/conversations/${secondId}`)).draft.text).toBe('另一个会话\n保持原文')
    await closeHistory(page)
    await showPreview()
    await preview.hover(); await page.mouse.wheel(0, -20_000)
    await expect.poll(() => preview.evaluate(element => element.scrollTop)).toBe(0)
    await page.locator('.chat-composer-area').screenshot({ path: path.join(context.directory, 'composer-format-preview.png') })
    await context.quit()
    ;({ page, application } = await context.launch())
    input = page.getByRole('textbox', { name: '对话消息', exact: true, includeHidden: true })
    preview = page.getByRole('region', { name: '输入内容预览', exact: true })
    await expect(input).toBeHidden(); await expect(input).toHaveValue(draft)
    await expect(preview).toBeVisible()
    await expect(page.locator('.attachment')).toHaveCount(2)
    expect((await management<Conversation>(page, `/api/conversations/${firstId}`)).draft.text).toBe(draft)
    await showPreview()
    const resize = page.getByRole('separator', { name: '调整输入框高度', exact: true })
    const edge = (await resize.boundingBox())!
    await page.mouse.move(edge.x + edge.width / 2, edge.y + edge.height / 2)
    await page.mouse.down(); await page.mouse.move(edge.x + edge.width / 2, edge.y - 140, { steps: 8 }); await page.mouse.up()
    const expandedHeight = (await preview.boundingBox())!.height
    expect(expandedHeight).toBeGreaterThan(250)
    await editDraft()
    expect((await input.boundingBox())!.height).toBeCloseTo(expandedHeight, 0)
    await input.press('End'); await input.press(' ')
    expect((await input.boundingBox())!.height).toBeCloseTo(expandedHeight, 0)
    await input.press('Backspace')
    const sendBefore = (await page.getByRole('button', { name: '发送消息', exact: true }).boundingBox())!
    await showPreview()
    const sendAfter = (await page.getByRole('button', { name: '发送消息', exact: true }).boundingBox())!
    expect(sendAfter.y).toBeCloseTo(sendBefore.y, 0)
    await page.locator('.chat-composer-area').screenshot({ path: path.join(context.directory, 'composer-resized-preview.png') })
    await editDraft()
    // Click straight from editing: blur renders the preview without moving the button or swallowing its click.
    await page.getByRole('button', { name: '发送消息', exact: true }).click()
    await expect.poll(async () => (await input.boundingBox())?.height).toBeCloseTo(initialHeight, 0)
    await expect.poll(() => context.requests.length).toBe(1)
    expect(context.requests[0].body.messages).toEqual([{ role: 'user', content: [{ type: 'text', text: draft.trim() }, { type: 'image_url', image_url: { url: `data:image/png;base64,${png}` } }, { type: 'image_url', image_url: { url: `data:image/png;base64,${png}` } }] }])
    await finish(page, context.requests[0], '收到输入原文。')
    await expect(input).toBeVisible(); await expect(input).toHaveValue('')
    const user = page.locator('.chat-message.user').first()
    await user.getByRole('button', { name: '复制消息', exact: true }).click()
    expect(await application.evaluate(({ clipboard }) => clipboard.readText())).toBe(draft.trim())
    await input.fill('后续草稿 **保持**')
    await showPreview()
    await user.getByRole('button', { name: '编辑消息', exact: true }).click()
    const editor = page.getByRole('textbox', { name: '编辑消息内容', exact: true })
    const editPreview = page.getByRole('region', { name: '编辑消息预览', exact: true })
    const revised = `${draft.trim()}\n\n新增 **编辑内容**。`
    await expect(editor).toBeVisible(); await expect(editPreview).toHaveCount(0)
    await editor.fill(revised)
    await editor.evaluate(element => (element as HTMLTextAreaElement).setSelectionRange(4, 9))
    await editor.focus(); await editor.blur()
    await expect(editor).toBeHidden(); await expect(editPreview).toBeVisible()
    await expect(page.getByText('重新发送将清除这条消息之后的对话。', { exact: true })).toHaveCount(0)
    await page.getByRole('button', { name: '复制编辑原文', exact: true }).click()
    expect(await application.evaluate(({ clipboard }) => clipboard.readText())).toBe(revised)
    await editPreview.click({ position: { x: 8, y: 8 } })
    await expect(editor).toBeFocused()
    expect(await editor.evaluate(element => { const input = element as HTMLTextAreaElement; return [input.selectionStart, input.selectionEnd] })).toEqual([4, 9])
    await editor.focus(); await editor.blur()
    await page.locator('.message-editor').screenshot({ path: path.join(context.directory, 'history-editor-format-preview.png') })
    await page.getByRole('button', { name: '取消编辑', exact: true }).click()
    expect((await management<Conversation>(page, `/api/conversations/${firstId}`)).messages[0].text).toBe(draft.trim())
    await expect(preview).toContainText('后续草稿')
    await user.getByRole('button', { name: '编辑消息', exact: true }).click()
    await expect(editor).toBeVisible(); await expect(editPreview).toHaveCount(0)
    await editor.fill(revised)
    await editor.focus(); await editor.blur()
    await page.getByRole('button', { name: '重新发送', exact: true }).click()
    await expect.poll(() => context.requests.length).toBe(2)
    expect((context.requests[1].body.messages[0].content as Array<{ text?: string }>)[0].text).toBe(revised)
    await finish(page, context.requests[1], '收到编辑原文。')
    const stored = await management<Conversation>(page, `/api/conversations/${firstId}`)
    expect(stored.messages[0].text).toBe(revised)
    expect(stored.draft.text).toBe('后续草稿 **保持**')
    await expect(preview).toContainText('后续草稿')
    await editDraft()
    await input.press('Enter')
    await expect.poll(() => context.requests.length).toBe(3)
    expect(context.requests[2].body.messages.at(-1)).toEqual({ role: 'user', content: '后续草稿 **保持**' })
    await finish(page, context.requests[2], '回到原文后仍可用 Enter 发送。')
    expect(context.errors).toEqual([])
  } finally { await context.cleanup() }
})

test('自动预览不打断输入法，拉伸可取消，草稿保存失败保留输入高度', async () => {
  const context = await fixture()
  try {
    const { page } = await context.launch()
    const input = page.getByRole('textbox', { name: '对话消息', exact: true, includeHidden: true })
    const preview = page.getByRole('region', { name: '输入内容预览', exact: true })
    const resize = page.getByRole('separator', { name: '调整输入框高度', exact: true })
    const originalHeight = (await input.boundingBox())!.height
    const source = '## 等待保存\n\n保留 **原文**。'
    let unavailable = true
    await page.route('**/api/conversations', async route => {
      if (route.request().method() === 'POST' && unavailable) await route.fulfill({ status: 503, json: { detail: 'fixture: 草稿暂时无法写入' } })
      else await route.continue()
    })
    await input.fill(source)
    await input.dispatchEvent('compositionstart', { data: '原' })
    await input.blur()
    await expect(input).toBeVisible(); await expect(preview).toHaveCount(0)
    await input.dispatchEvent('compositionend', { data: '原文' })
    await expect(preview).toBeVisible()
    await resize.focus()
    for (let index = 0; index < 6; index++) await resize.press('ArrowUp')
    const expanded = (await preview.boundingBox())!.height
    expect(expanded).toBeGreaterThan(180)
    const edge = (await resize.boundingBox())!
    await page.mouse.move(edge.x + edge.width / 2, edge.y + edge.height / 2)
    await page.mouse.down(); await page.mouse.move(edge.x + edge.width / 2, edge.y - 50, { steps: 5 })
    await resize.press('Escape'); await page.mouse.up()
    expect((await preview.boundingBox())!.height).toBeCloseTo(expanded, 0)
    await page.getByRole('button', { name: '发送消息', exact: true }).click()
    await expect(page.getByText('未保存', { exact: true })).toBeVisible()
    await expect(input).toHaveValue(source)
    expect(context.requests).toHaveLength(0)
    expect((await preview.boundingBox())!.height).toBeCloseTo(expanded, 0)
    unavailable = false
    await page.getByRole('button', { name: '发送消息', exact: true }).click()
    await expect.poll(() => context.requests.length).toBe(1)
    expect(context.requests[0].body.messages).toEqual([{ role: 'user', content: source }])
    await expect.poll(async () => (await input.boundingBox())?.height).toBeCloseTo(originalHeight, 0)
    await finish(page, context.requests[0], '已恢复。')
    expect(context.errors).toEqual([])
  } finally { await context.cleanup() }
})

test('用户消息富文本、复制原文、附件、长内容与重启后编辑保持一致', async () => {
  test.setTimeout(120_000)
  const context = await fixture()
  context.setInstances([imageInstance])
  try {
    let { page, application } = await context.launch()
    const png = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII='
    const imageUrl = `data:image/png;base64,${png}`
    const python = 'def greet(name):\n    return f"Hello, {name}"'
    const source = [
      '## 用户消息格式',
      '**粗体**、_斜体_、~~删除线~~、`inline_code` 和 [链接](https://example.com/)。',
      '保留第一行\n保留第二行',
      '- [x] 已检查\n- [ ] 待检查',
      '| 格式 | 结果 |\n| --- | --- |\n| Markdown | 支持 |\n| 表格 | 支持 |',
      '> 引用内容',
      String.raw`公式 $a^2+b^2=c^2$，以及 \(x+y\)。`,
      String.raw`$$\frac{1}{2}$$`,
      `\`\`\`python\n${python}\n\`\`\``,
      `![内嵌图片](${imageUrl})`,
    ].join('\n\n')
    await page.locator('input[type="file"]').setInputFiles({ name: 'message-pixel.png', mimeType: 'image/png', buffer: Buffer.from(png, 'base64') })
    const request = await context.send(page, source)
    expect(request.body.messages).toEqual([{ role: 'user', content: [{ type: 'text', text: source }, { type: 'image_url', image_url: { url: imageUrl } }] }])
    await finish(page, request, '已收到 **完整原文**。')
    const id = (await management<ConversationList>(page, '/api/conversations')).active_id!
    let user = page.locator('.chat-message.user').first()
    const rendered = user.locator('.message-markdown')
    await expect(user.locator('.message-label > strong')).toHaveText('你')
    await expect(rendered.getByRole('heading', { name: '用户消息格式', exact: true })).toBeVisible()
    await expect(rendered.locator('strong')).toHaveText('粗体')
    await expect(rendered.locator('em')).toHaveText('斜体')
    await expect(rendered.locator('del')).toHaveText('删除线')
    await expect(rendered.getByRole('link', { name: '链接', exact: true })).toHaveAttribute('href', 'https://example.com/')
    await expect(rendered.getByRole('checkbox')).toHaveCount(2)
    await expect(rendered.getByRole('checkbox').first()).toBeChecked()
    await expect(rendered.getByRole('checkbox').first()).toBeDisabled()
    await expect(rendered.getByRole('table')).toContainText('Markdown')
    await expect(rendered.locator('blockquote')).toContainText('引用内容')
    await expect(rendered.locator('.katex')).toHaveCount(3)
    await expect(rendered.locator('.katex-display')).toHaveCount(1)
    await expect(rendered.locator('code .hljs-keyword').first()).toBeVisible()
    await expect(rendered.locator('pre code')).toHaveText(`${python}\n`)
    expect(await rendered.locator('p').filter({ hasText: '保留第一行' }).evaluate(element => {
      const text = element.firstChild!
      const first = document.createRange(), second = document.createRange()
      first.setStart(text, 0); first.setEnd(text, 5)
      second.setStart(text, 6); second.setEnd(text, 11)
      return second.getBoundingClientRect().top - first.getBoundingClientRect().top
    })).toBeGreaterThan(15)
    for (const image of [user.locator('.message-images img'), rendered.getByRole('button', { name: '查看图片：内嵌图片', exact: true })]) {
      await expect(image).toBeVisible()
      await expect.poll(() => image.evaluate(element => (element as HTMLImageElement).naturalWidth)).toBe(1)
    }
    await page.locator('.chat-messages').hover(); await page.mouse.wheel(0, -20_000)
    await expect.poll(() => page.locator('.chat-messages').evaluate(element => element.scrollTop)).toBe(0)
    await page.screenshot({ path: path.join(context.directory, 'user-message-markdown.png') })
    await user.getByRole('button', { name: '复制消息', exact: true }).click()
    expect(await application.evaluate(({ clipboard }) => clipboard.readText())).toBe(source)
    await expect(page.getByText('消息已复制', { exact: true })).toBeVisible()
    await page.screenshot({ path: path.join(context.directory, 'user-message-copy.png') })

    await page.evaluate(() => { (window as any).userMessageAttack = 'clean' })
    const html = '<section><h2>用户 HTML 预览</h2><script>top.userMessageAttack="script"</script><p onclick="top.userMessageAttack=\'event\'">安全正文</p><a href="javascript:alert(1)">危险链接</a></section>'
    const diagramSource = `\`\`\`mermaid\nflowchart LR\n  A[用户提问] --> B[格式预览]\n\`\`\`\n\n\`\`\`dot\ndigraph { A -> B; }\n\`\`\`\n\n\`\`\`html\n${html}\n\`\`\`\n\n<article><h3>原始 HTML 块</h3><script>top.userMessageAttack="raw"</script></article>`
    await finish(page, await context.send(page, diagramSource), '图表与 HTML 已收到。')
    const previews = page.locator('.chat-message.user').nth(1)
    for (const language of ['mermaid', 'dot']) {
      const block = previews.locator(`.message-code-block[data-language="${language}"]`)
      await expect.poll(() => block.locator('img').evaluateAll(elements => elements.length === 1 && (elements[0] as HTMLImageElement).naturalWidth > 0)).toBe(true)
      await block.locator('.message-code-source > summary').click()
      await expect(block.locator('pre code')).toContainText(language === 'mermaid' ? 'A[用户提问]' : 'digraph')
      await block.locator('.message-code-source > summary').click()
    }
    const htmlBlock = previews.locator('.message-code-block[data-language="html"]').first()
    await expect(htmlBlock.locator('iframe')).toHaveAttribute('sandbox', '')
    await expect(htmlBlock.frameLocator('iframe').getByRole('heading', { name: '用户 HTML 预览' })).toBeVisible()
    await expect(htmlBlock.frameLocator('iframe').locator('script, [onclick], a[href]')).toHaveCount(0)
    await expect(previews.locator('.message-code-block[data-language="html"]').last().frameLocator('iframe').getByRole('heading', { name: '原始 HTML 块' })).toBeVisible()
    expect(await page.evaluate(() => (window as any).userMessageAttack)).toBe('clean')
    await htmlBlock.getByRole('button', { name: '展开预览', exact: true }).click()
    const viewer = page.getByRole('dialog', { name: 'HTML 预览', exact: true })
    await expect(viewer.locator('iframe')).toHaveAttribute('sandbox', '')
    await viewer.getByRole('button', { name: '源码', exact: true }).click()
    await expect(viewer.locator('pre code')).toHaveText(`${html}\n`)
    await page.screenshot({ path: path.join(context.directory, 'user-message-html-expanded.png') })
    await viewer.getByRole('button', { name: '关闭预览', exact: true }).click()
    await htmlBlock.scrollIntoViewIfNeeded()
    await page.screenshot({ path: path.join(context.directory, 'user-message-safe-preview.png') })

    const columns = Array.from({ length: 32 }, (_, index) => `列${index + 1}`)
    const table = `|${columns.join('|')}|\n|${columns.map(() => '---').join('|')}|\n|${columns.map(() => '值').join('|')}|`
    const longSource = `## 长消息\n\n\`\`\`text\n${'wide_content_'.repeat(180)}\n\`\`\`\n\n${table}\n\n${Array.from({ length: 35 }, (_, index) => `第 ${index + 1} 段：长消息应能连续滚动阅读。\n\n`).join('')}### 长消息结尾`
    await finish(page, await context.send(page, longSource), '长消息已收到。')
    const longUser = page.locator('.chat-message.user').last()
    await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(940, 760))
    const pre = longUser.locator('pre')
    await pre.hover()
    await page.mouse.wheel(600, 0)
    await expect.poll(() => pre.evaluate(element => element.scrollLeft)).toBeGreaterThan(100)
    const tableScroll = longUser.locator('.message-table-scroll')
    await tableScroll.hover()
    await page.mouse.wheel(600, 0)
    await expect.poll(() => tableScroll.evaluate(element => element.scrollLeft)).toBeGreaterThan(100)
    const transcript = page.locator('.chat-messages')
    expect(await transcript.evaluate(element => element.scrollWidth - element.clientWidth)).toBeLessThanOrEqual(1)
    expect(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1)
    await page.screenshot({ path: path.join(context.directory, 'user-message-long-narrow.png') })
    await transcript.hover(); await page.mouse.wheel(0, 20_000)
    await expect(longUser.getByRole('heading', { name: '长消息结尾', exact: true })).toBeInViewport()
    await expect(page.getByRole('textbox', { name: '对话消息', exact: true })).toBeInViewport()
    expect((await management<Conversation>(page, `/api/conversations/${id}`)).messages.filter(message => message.role === 'user').map(message => message.text)).toEqual([source, diagramSource, longSource])

    await context.quit()
    ;({ page, application } = await context.launch())
    user = page.locator('.chat-message.user').first()
    await user.getByRole('button', { name: '复制消息', exact: true }).click()
    expect(await application.evaluate(({ clipboard }) => clipboard.readText())).toBe(source)
    await user.getByRole('button', { name: '编辑消息', exact: true }).click()
    await expect(page.getByRole('textbox', { name: '编辑消息内容', exact: true })).toHaveValue(source)
    await page.getByRole('button', { name: '取消编辑', exact: true }).click()
    await expect(user.locator('.message-markdown')).toBeVisible()
    await user.getByRole('button', { name: '编辑消息', exact: true }).click()
    const revised = `${source}\n\n新增 **保留格式**。`
    await page.getByRole('textbox', { name: '编辑消息内容', exact: true }).fill(revised)
    await page.getByRole('button', { name: '重新发送', exact: true }).click()
    await expect.poll(() => context.requests.length).toBe(4)
    expect(context.requests[3].body.messages).toEqual([{ role: 'user', content: [{ type: 'text', text: revised }, { type: 'image_url', image_url: { url: imageUrl } }] }])
    await finish(page, context.requests[3], '修改后的格式仍保留。')
    await expect(page.locator('.chat-message')).toHaveCount(2)
    await expect(user.locator('.message-markdown strong').last()).toHaveText('保留格式')
    expect((await management<Conversation>(page, `/api/conversations/${id}`)).messages[0].text).toBe(revised)
    expect(context.errors).toEqual([])
  } finally { await context.cleanup() }
})

test('编辑替换已完成和生成中消息，重试历史回复截断后续且旧流不会混入', async () => {
  test.setTimeout(120_000)
  const context = await fixture()
  try {
    const { page } = await context.launch()
    const first = await context.send(page, '第一条原问题')
    delta(first, { reasoning_content: '第一条应保留的思考' })
    await finish(page, first, '第一条原回答')
    const second = await context.send(page, '第二条原问题')
    delta(second, { reasoning_content: '第二条应随重写清除的思考' })
    await finish(page, second, '第二条原回答')
    const id = (await management<ConversationList>(page, '/api/conversations')).active_id!
    const original = (await management<Conversation>(page, `/api/conversations/${id}`)).messages
    await page.locator('.chat-message.user').first().getByRole('button', { name: '编辑消息', exact: true }).click()
    await page.getByRole('textbox', { name: '编辑消息内容' }).fill('取消的改动')
    await page.getByRole('button', { name: '取消编辑' }).click()
    expect((await management<Conversation>(page, `/api/conversations/${id}`)).messages).toEqual(original)
    await page.locator('.chat-message.user').last().getByRole('button', { name: '编辑消息', exact: true }).click()
    await page.getByRole('textbox', { name: '编辑消息内容' }).fill('第二条修改后的问题')
    await expect(page.getByText('重新发送将清除这条消息之后的对话。')).toHaveCount(0)
    await page.locator('.message-editor').screenshot({ path: path.join(context.directory, 'history-editor-without-reminder.png') })
    await page.getByRole('button', { name: '重新发送', exact: true }).click()
    await expect.poll(() => context.requests.length).toBe(3)
    expect(context.requests[2].body.messages).toEqual([{ role: 'user', content: '第一条原问题' }, { role: 'assistant', content: '第一条原回答', reasoning_content: '第一条应保留的思考' }, { role: 'user', content: '第二条修改后的问题' }])
    await finish(page, context.requests[2], '修改后的回答')
    let stored = await management<Conversation>(page, `/api/conversations/${id}`)
    expect(stored.messages.map(message => message.text)).toEqual(['第一条原问题', '第一条原回答', '第二条修改后的问题', '修改后的回答'])
    expect(stored.messages[2].id).toBe(original[2].id)
    const active = await context.send(page, '生成中的第三条')
    delta(active, { content: '旧请求已到部分' })
    await expect(page.locator('.chat-message.assistant').last()).toContainText('旧请求已到部分')
    await page.locator('.chat-message.user').last().getByRole('button', { name: '编辑消息' }).click()
    await page.getByRole('textbox', { name: '编辑消息内容' }).fill('生成中改写后的第三条')
    await page.getByRole('button', { name: '重新发送', exact: true }).click()
    await expect.poll(() => context.requests.length).toBe(5)
    delta(active, { content: '旧请求晚到的污染内容' }); active.response.end('data: [DONE]\n\n')
    expect(context.requests[4].body.messages.at(-1)).toEqual({ role: 'user', content: '生成中改写后的第三条' })
    expect(context.requests[4].body.messages).toHaveLength(5)
    await finish(page, context.requests[4], '新请求的干净回复')
    stored = await management<Conversation>(page, `/api/conversations/${id}`)
    expect(JSON.stringify(stored.messages)).not.toContain('旧请求')
    await page.locator('.chat-message.assistant').first().getByRole('button', { name: '重试回复' }).click()
    await expect(page.getByRole('dialog')).toContainText('将清除这条回复及其后的消息')
    await page.getByRole('dialog').getByRole('button', { name: '取消', exact: true }).click()
    await expect(page.locator('.chat-message')).toHaveCount(6)
    await page.locator('.chat-message.assistant').first().getByRole('button', { name: '重试回复' }).click()
    await page.getByRole('dialog').getByRole('button', { name: '重新生成', exact: true }).click()
    await expect.poll(() => context.requests.length).toBe(6)
    expect(context.requests[5].body.messages).toEqual([{ role: 'user', content: '第一条原问题' }])
    await finish(page, context.requests[5], '第一条重试回答')
    await expect(page.locator('.chat-message')).toHaveCount(2)
    await expect(page.locator('.message-footer').last().locator('[data-metric="elapsed"]')).toHaveAccessibleName(/^本轮耗时/)
    await expect(page.locator('.chat-message.assistant .message-label')).not.toContainText('秒')
    await page.locator('.chat-message.assistant').last().getByRole('button', { name: '重试回复' }).click()
    await expect.poll(() => context.requests.length).toBe(7)
    await expect(page.getByRole('dialog')).toHaveCount(0)
    await finish(page, context.requests[6], '最新回复直接重试')
    expect((await management<Conversation>(page, `/api/conversations/${id}`)).messages).toHaveLength(2)
    expect(context.errors).toEqual([])
    await page.screenshot({ path: path.join(context.directory, 'edit-retry-metrics.png') })
  } finally { await context.cleanup() }
})

test('历史键盘与左滑多选、置顶、组内拖动和确认删除保持互斥并持久化', async () => {
  test.setTimeout(120_000)
  const context = await fixture()
  try {
    const { page } = await context.launch()
    const reorders = trackHistoryReorders(page)
    for (const title of ['历史甲', '历史乙', '历史丙']) {
      if (context.requests.length) await page.getByRole('button', { name: '新建会话' }).click()
      await finish(page, await context.send(page, title), `${title}回复`)
    }
    let panel = await history(page)
    const row = (name: string) => panel.getByRole('button', { name, exact: true })
    const order = () => panel.locator('.chat-history-item').evaluateAll(nodes => nodes.map(node => node.getAttribute('aria-label')))
    const swipe = async (name: string) => {
      await row(name).hover({ position: { x: 20, y: 20 } })
      const box = (await row(name).boundingBox())!
      await page.mouse.move(box.x + box.width - 20, box.y + 20); await page.mouse.down()
      await page.mouse.move(box.x + box.width - 85, box.y + 21, { steps: 6 }); await page.mouse.up()
    }
    const dragBefore = async (from: string, to: string, reduced = false) => {
      const original = await order(), writes = reorders.length
      const expected = original.filter(title => title !== from)
      expected.splice(expected.indexOf(to), 0, from)
      const source = (await row(from).locator('..').boundingBox())!, target = (await row(to).boundingBox())!
      const preview = page.locator('body > .chat-history-drag-preview')
      await row(from).hover({ position: { x: 70, y: 20 } }); await page.mouse.down()
      await expect(preview).toHaveCount(0)
      await expect(row(from).locator('..')).toHaveClass(/is-dragging/)
      await expect(preview).toContainText(from)
      await expect(preview).toHaveCSS('pointer-events', 'none')
      expect(await row(from).locator('..').boundingBox()).toEqual(source)
      const lifted = (await preview.boundingBox())!
      const frames = page.evaluate(async () => {
        const samples: Array<{ order: Array<string | null>; moving: Array<{ id: string | null; y: number }> }> = []
        for (let frame = 0; frame < 16; frame++) {
          await new Promise(requestAnimationFrame)
          const rows = [...document.querySelectorAll<HTMLElement>('.chat-history-row')]
          samples.push({ order: rows.map(node => node.querySelector('button')!.getAttribute('aria-label')), moving: rows.filter(node => !node.classList.contains('is-dragging') && node.getAnimations().some(animation => animation.id === 'reorder-position')).map(node => ({ id: node.dataset.historyId!, y: new DOMMatrixReadOnly(getComputedStyle(node).transform).m42 })) })
        }
        return samples
      })
      await page.mouse.move(target.x + 70, target.y + 3)
      await expect.poll(order).toEqual(expected)
      await expect(row(from).locator('..')).toHaveClass(/drop-before/)
      for (let frame = 0; frame < 3; frame++) {
        await page.evaluate(() => new Promise(requestAnimationFrame))
        await page.mouse.move(target.x + 70, target.y + 3)
        expect(await order()).toEqual(expected)
      }
      const sampled = await frames
      const moving = sampled.flatMap(frame => frame.moving)
      if (reduced) expect(moving).toEqual([])
      else {
        expect(moving.some(node => Math.abs(node.y) > .5)).toBe(true)
        expect(new Set(moving.map(node => `${node.id}:${Math.round(node.y)}`)).size).toBeGreaterThan(1)
      }
      const previewStart = sampled.findIndex(frame => JSON.stringify(frame.order) === JSON.stringify(expected))
      expect(previewStart).toBeGreaterThanOrEqual(0)
      expect(sampled.slice(previewStart).every(frame => JSON.stringify(frame.order) === JSON.stringify(expected))).toBe(true)
      expect(reorders).toHaveLength(writes)
      expect((await management<ConversationList>(page, '/api/conversations')).items.map(item => item.title)).toEqual(original)
      const floating = (await preview.boundingBox())!
      expect(floating.y).toBeLessThan(lifted.y)
      expect(floating.y + floating.height).toBeLessThan((await row(from).locator('..').boundingBox())!.y)
      await page.screenshot({ path: path.join(context.directory, `history-drag-preview${reduced ? '-reduced' : ''}.png`) })
      await page.mouse.up()
      await expect(preview).toHaveCount(0)
      await expect.poll(() => reorders.length).toBe(writes + 1)
      await expect.poll(async () => (await management<ConversationList>(page, '/api/conversations')).items.map(item => item.title)).toEqual(expected)
      await expect(row(from)).toBeEnabled()
    }
    await row('历史甲').click()
    expect(await order()).toEqual(['历史丙', '历史乙', '历史甲'])
    await row('历史丙').click({ modifiers: ['Control'] })
    await expect(page.locator('.chat-session-title')).toHaveText('历史甲')
    expect(await row('历史甲').evaluate(node => getComputedStyle(node).backgroundColor)).not.toBe(await row('历史丙').evaluate(node => getComputedStyle(node).backgroundColor))
    await row('历史甲').click({ modifiers: ['Shift'] })
    await expect(panel.getByRole('heading', { name: '已选择 3 条' })).toBeVisible()
    await row('历史乙').click({ modifiers: ['Control'] })
    await expect(panel.getByRole('heading', { name: '已选择 2 条' })).toBeVisible()
    await swipe('历史甲')
    await expect(panel.getByRole('heading', { name: '已选择 1 条' })).toBeVisible()
    await swipe('历史乙')
    await expect(panel.getByRole('heading', { name: '已选择 2 条' })).toBeVisible()
    await expect(panel.locator('input[type=checkbox], .chat-history-check, .chat-history-delete')).toHaveCount(0)
    await page.keyboard.press('Escape')
    await expect(panel.getByRole('heading', { name: '对话历史', exact: true })).toBeVisible()
    const unchanged = await order(), cancelTarget = (await row('历史丙').boundingBox())!
    for (const cancelDrag of [
      () => page.keyboard.press('Escape'),
      () => row('历史甲').dispatchEvent('pointercancel'),
      () => page.evaluate(() => window.dispatchEvent(new Event('blur'))),
    ]) {
      await row('历史甲').hover({ position: { x: 70, y: 20 } }); await page.mouse.down()
      await expect(page.locator('.chat-history-drag-preview')).toBeVisible()
      await page.mouse.move(cancelTarget.x + 70, cancelTarget.y + 3)
      await expect.poll(order).toEqual(['历史甲', '历史丙', '历史乙'])
      await cancelDrag()
      await expect(page.locator('.chat-history-drag-preview')).toHaveCount(0)
      await expect(panel.locator('.is-dragging, .drop-before, .drop-after')).toHaveCount(0)
      await page.mouse.up()
      expect(await order()).toEqual(unchanged)
      expect(reorders).toHaveLength(0)
      expect((await management<ConversationList>(page, '/api/conversations')).items.map(item => item.title)).toEqual(unchanged)
    }
    await swipe('历史甲')
    await row('历史丙').click({ modifiers: ['Control'] })
    await expect.poll(() => row('历史甲').evaluate(node => Math.abs(node.getBoundingClientRect().left - node.parentElement!.getBoundingClientRect().left))).toBeLessThan(1)
    await page.screenshot({ path: path.join(context.directory, 'history-multiselect.png') })
    await panel.getByRole('button', { name: '置顶会话', exact: true }).click()
    await expect.poll(order).toEqual(['历史丙', '历史甲', '历史乙'])
    await page.keyboard.press('Escape')
    await dragBefore('历史甲', '历史丙')
    await expect.poll(order).toEqual(['历史甲', '历史丙', '历史乙'])
    await expect(panel.getByRole('heading', { name: '对话历史', exact: true })).toBeVisible()
    await row('历史乙').click()
    await panel.getByRole('button', { name: '置顶会话', exact: true }).click()
    await expect.poll(order).toEqual(['历史乙', '历史甲', '历史丙'])
    await panel.getByRole('button', { name: '取消置顶会话', exact: true }).click()
    await expect.poll(order).toEqual(['历史甲', '历史丙', '历史乙'])
    await page.getByRole('button', { name: '新建会话', exact: true }).click()
    await expect(panel.getByRole('button', { name: '置顶会话', exact: true })).toBeDisabled()
    await expect(panel.getByRole('button', { name: '删除当前会话', exact: true })).toBeDisabled()
    await finish(page, await context.send(page, '历史丁'), '第四条')
    await expect.poll(order).toEqual(['历史甲', '历史丙', '历史丁', '历史乙'])
    await row('历史乙').click()
    await (await editComposer(page)).fill('仅草稿不置顶')
    await saved(page)
    expect(await order()).toEqual(['历史甲', '历史丙', '历史丁', '历史乙'])
    await page.getByRole('textbox', { name: '对话消息', exact: true }).press('Control+a')
    await expect(panel.getByRole('heading', { name: '对话历史', exact: true })).toBeVisible()
    await finish(page, await context.send(page, '普通再次发送'), '应到普通组首')
    await expect.poll(order).toEqual(['历史甲', '历史丙', '历史乙', '历史丁'])
    await page.emulateMedia({ reducedMotion: 'reduce' })
    await dragBefore('历史丁', '历史乙', true)
    await page.emulateMedia({ reducedMotion: 'no-preference' })
    await expect.poll(order).toEqual(['历史甲', '历史丙', '历史丁', '历史乙'])
    await page.reload(); await pinNavigation(page)
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '测试对话', exact: true }).click()
    panel = await history(page)
    expect(await order()).toEqual(['历史甲', '历史丙', '历史丁', '历史乙'])
    await expect(panel.locator('[data-pinned=true]')).toHaveCount(2)
    await row('历史甲').click({ modifiers: ['Control'] })
    await row('历史丙').click({ modifiers: ['Control'] })
    await panel.getByRole('button', { name: '删除所选会话', exact: true }).click()
    await page.getByRole('dialog').getByRole('button', { name: '取消', exact: true }).click()
    await expect(page.getByRole('dialog')).toBeHidden()
    expect(await page.locator('.chat-layout').evaluate(node => node.scrollLeft)).toBe(0)
    await expect(panel.getByRole('heading', { name: '已选择 2 条' })).toBeVisible()
    await panel.getByRole('button', { name: '删除所选会话', exact: true }).click()
    await page.getByRole('dialog').getByRole('button', { name: '删除会话', exact: true }).click()
    await expect.poll(order).toEqual(['历史丁', '历史乙'])
    await expect(page.locator('.chat-session-title')).toHaveText('历史乙')
    await row('历史乙').focus(); await page.keyboard.press('Control+a')
    await expect(panel.getByRole('heading', { name: '已选择 2 条' })).toBeVisible()
    await panel.getByRole('button', { name: '删除所选会话', exact: true }).click()
    await page.getByRole('dialog').getByRole('button', { name: '删除会话', exact: true }).click()
    await expect(panel.getByText('暂无会话', { exact: true })).toBeVisible()
    await expect(page.locator('.chat-message')).toHaveCount(0)
    expect((await management<ConversationList>(page, '/api/conversations')).items).toEqual([])
    expect(context.errors).toEqual([])
  } finally { await context.cleanup() }
})

test('唯一模型从停止到加载就绪自动跟随，重入页面和发送目标一致', async () => {
  test.setTimeout(90_000)
  const context = await fixture()
  try {
    const { page } = await context.launch()
    await finish(page, await context.send(page, '保留历史并跟随唯一实例'), '旧模型回答')
    const model = chatModel(page)
    const refresh = () => page.getByRole('button', { name: '刷新状态', exact: true }).click()
    context.setInstances([{ ...textInstance, state: 'stopped' }]); await refresh()
    await expect(model).toHaveJSProperty('value', '')
    await expect(page.getByRole('button', { name: '发送消息', exact: true })).toBeDisabled()
    context.setInstances([{ ...textInstance, state: 'stopped' }, { ...imageInstance, state: 'loading' }]); await refresh()
    await expect(model).toHaveJSProperty('value', imageInstance.id)
    await expect(page.locator('.chat-model-state .badge')).toBeVisible()
    await expect(page.locator('.chat-model-state .badge')).toHaveText('加载中')
    context.setInstances([{ ...textInstance, state: 'stopped' }, imageInstance]); await refresh()
    await expect(model).toHaveJSProperty('value', imageInstance.id)
    const nav = page.getByRole('navigation', { name: '主导航' })
    await nav.getByRole('button', { name: '运行监控', exact: true }).click()
    await nav.getByRole('button', { name: '测试对话', exact: true }).click()
    await expect(model).toHaveJSProperty('value', imageInstance.id)
    const next = await context.send(page, '新模型继续')
    expect(next.body.model).toBe(imageInstance.alias)
    expect(next.body.messages).toContainEqual({ role: 'assistant', content: '旧模型回答' })
    await finish(page, next, '新模型回答')
    context.setInstances([textInstance, imageInstance]); await refresh()
    await expect(model).toHaveJSProperty('value', imageInstance.id)
    await selectChatModel(page, textInstance.id); await refresh(); await expect(model).toHaveJSProperty('value', textInstance.id)
    context.setInstances([]); await refresh(); await expect(model).toHaveJSProperty('value', '')
    await expect(page.locator('.chat-message')).toHaveCount(4)
    expect(context.errors).toEqual([])
  } finally { await context.cleanup() }
})

test('宽窗口双侧向外展开、窄窗口让出正文，反复开关后参数仍保存并用于请求', async () => {
  test.setTimeout(100_000)
  const context = await fixture()
  try {
    const { page, application } = await context.launch()
    await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(2200, 1100))
    await expect.poll(() => page.evaluate(() => innerWidth)).toBeGreaterThan(2100)
    const center = page.locator('.chat-panel')
    const before = (await center.boundingBox())!
    expect(before.width).toBe(1404)
    await expect(page.getByRole('button', { name: '对话历史', exact: true })).toHaveAttribute('aria-expanded', 'false')
    await expect(page.getByRole('button', { name: '对话参数', exact: true })).toHaveAttribute('aria-expanded', 'false')
    await page.getByRole('button', { name: '对话历史', exact: true }).click()
    await page.getByRole('button', { name: '对话参数', exact: true }).click()
    await expect.poll(() => page.locator('.chat-frame').evaluate(element => element.getAnimations().some(animation => animation.playState === 'running'))).toBe(false)
    const both = (await center.boundingBox())!
    expect(Math.abs(both.x - before.x)).toBeLessThanOrEqual(1); expect(Math.abs(both.width - before.width)).toBeLessThanOrEqual(1)
    const left = (await page.locator('.history-slot').boundingBox())!, right = (await page.locator('.options-slot').boundingBox())!
    expect(left.x + left.width).toBeLessThanOrEqual(both.x + 1); expect(right.x).toBeGreaterThanOrEqual(both.x + both.width - 1)
    await page.screenshot({ path: path.join(context.directory, 'chat-wide-both-panels.png') })
    await page.getByRole('textbox', { name: '系统提示词', exact: true }).fill('参数保存测试')
    await page.getByLabel('温度', { exact: true }).fill('0.65')
    await page.getByLabel('最大输出 Token', { exact: true }).fill('128')
    await page.getByRole('textbox', { name: '工具定义', exact: true }).fill('[]')
    for (let i = 0; i < 12; i++) await page.getByRole('button', { name: '对话参数', exact: true }).click()
    await expect(page.getByRole('textbox', { name: '系统提示词', exact: true })).toHaveValue('参数保存测试')
    await saved(page)
    await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(940, 800))
    await expect.poll(() => page.evaluate(() => innerWidth)).toBeLessThan(950)
    const narrow = (await center.boundingBox())!
    expect(narrow.width).toBeLessThan(before.width)
    expect((await page.locator('.options-slot').boundingBox())!.x).toBeGreaterThanOrEqual(narrow.x + narrow.width - 1)
    expect(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1)
    await page.screenshot({ path: path.join(context.directory, 'chat-narrow-both-panels.png') })
    await closeHistory(page)
    await page.getByRole('button', { name: '收起对话参数', exact: true }).click()
    const request = await context.send(page, '反复开关后的正常对话')
    expect(request.body).toMatchObject({ temperature: 0.65, max_tokens: 128, stream: true, tools: [] })
    expect(request.body.messages[0]).toEqual({ role: 'system', content: '参数保存测试' })
    await finish(page, request, '参数仍正常应用。')
    expect(context.errors).toEqual([])
  } finally { await context.cleanup() }
})

test('窄窗口两侧栏与正文同步让位，反向开合和减少动态效果不跳位', async () => {
  test.setTimeout(90_000)
  const context = await fixture()
  try {
    const { page, application } = await context.launch()
    await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(1100, 900))
    const request = await context.send(page, '检查长正文中的侧栏开合')
    await finish(page, request, Array.from({ length: 40 }, (_, i) => `### 第 ${i + 1} 节\n\n${'保留正在阅读的内容，侧栏与正文同时让位。'.repeat(16)}`).join('\n\n'))
    async function trace(label: string) {
      return page.evaluate(async label => {
        const layout = document.querySelector('.chat-layout')!
        const center = layout.querySelector('.chat-panel')!
        const history = layout.querySelector('.history-slot')!, options = layout.querySelector('.options-slot')!
        const frames: Array<{ t: number; x: number; width: number; right: number; leftSeam: number; rightSeam: number; scaled: boolean }> = []
        const started = performance.now()
        function sample() {
          const bounds = center.getBoundingClientRect()
          frames.push({ t: performance.now() - started, x: bounds.x, width: bounds.width, right: bounds.right,
            leftSeam: history.getBoundingClientRect().right - bounds.x, rightSeam: options.getBoundingClientRect().left - bounds.right,
            scaled: new DOMMatrixReadOnly(getComputedStyle(center).transform).a !== 1 })
        }
        sample()
        ;(layout.querySelector(`.chat-toolbar button[aria-label="${label}"]`) as HTMLButtonElement).click()
        await new Promise<void>(resolve => {
          function next() { sample(); if (performance.now() - started < 330) requestAnimationFrame(next); else resolve() }
          requestAnimationFrame(next)
        })
        return frames
      }, label)
    }
    for (const label of ['对话历史', '对话参数']) {
      for (const opening of [true, false]) {
        const frames = await trace(label), first = frames[0], last = frames.at(-1)!
        expect(opening ? first.width - last.width : last.width - first.width).toBeGreaterThan(150)
        // The opposite edge stays fixed while the selected side gives/takes space.
        // This catches an immediate width commit followed by a separate translation.
        const edge = label === '对话历史' ? 'right' : 'x'
        for (const frame of frames) {
          expect(Math.abs(frame[edge] - first[edge])).toBeLessThan(1)
          expect(Math.abs(frame.leftSeam)).toBeLessThan(1)
          expect(Math.abs(frame.rightSeam)).toBeLessThan(1)
          expect(frame.scaled).toBe(false)
        }
        const range = Math.abs(last.width - first.width)
        expect(frames.some(frame => Math.abs(frame.width - first.width) > range * .1 && Math.abs(frame.width - last.width) > range * .1)).toBe(true)
        for (let i = 1; i < frames.length; i++) expect(opening ? frames[i].width - frames[i - 1].width : frames[i - 1].width - frames[i].width).toBeLessThan(1)
        expect(await page.locator('.chat-messages').evaluate(node => node.scrollHeight - node.clientHeight - node.scrollTop)).toBeLessThanOrEqual(2)
        console.log('Panel frames', label, opening, { count: frames.length, maxGap: Math.max(...frames.slice(1).map((frame, i) => frame.t - frames[i].t)) })
      }
    }
    // Reverse from a deterministic intermediate frame, rather than letting
    // Playwright wait for a moving toggle to settle before clicking it again.
    for (const label of ['对话历史', '对话参数']) {
      await page.getByRole('button', { name: label, exact: true }).click()
      const reversal = await page.evaluate(async label => {
        const layout = document.querySelector('.chat-layout')!
        const center = layout.querySelector('.chat-panel')!
        Array.from(layout.children).forEach(node => node.getAnimations().forEach(animation => { animation.pause(); animation.currentTime = 60 }))
        const before = center.getBoundingClientRect()
        const button = layout.querySelector(`.chat-toolbar button[aria-label="${label}"]`) as HTMLButtonElement
        button.click()
        await new Promise<void>(resolve => requestAnimationFrame(() => resolve()))
        const after = center.getBoundingClientRect()
        return { x: after.x - before.x, width: after.width - before.width, expanded: button.getAttribute('aria-expanded') }
      }, label)
      expect(reversal.expanded).toBe('false')
      expect(Math.abs(reversal.x)).toBeLessThan(1)
      expect(Math.abs(reversal.width)).toBeLessThan(1)
      await expect.poll(() => page.locator('.chat-layout').evaluate(layout => Array.from(layout.children).some(node => node.getAnimations().length))).toBe(false)
    }
    await page.emulateMedia({ reducedMotion: 'reduce' })
    await page.getByRole('button', { name: '对话历史', exact: true }).click()
    await page.getByRole('button', { name: '对话参数', exact: true }).click()
    expect(await page.locator('.chat-layout').evaluate(layout => Array.from(layout.children).some(node => node.getAnimations().length))).toBe(false)
    await expect(page.getByRole('textbox', { name: '系统提示词', exact: true })).toBeVisible()
    await page.screenshot({ path: path.join(context.directory, 'chat-narrow-synchronized-panels.png') })
    await page.keyboard.press('Escape')
    await page.emulateMedia({ reducedMotion: 'no-preference' })
    await page.getByRole('button', { name: '对话历史', exact: true }).click()
    await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(1280, 900))
    await expect.poll(() => page.locator('.chat-layout').evaluate(layout => Array.from(layout.children).some(node => node.getAnimations().length))).toBe(false)
    expect(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1)
    expect(context.errors).toEqual([])
  } finally { await context.cleanup() }
})

test('富文本旧消息在侧栏动画中保持占位，可见正文正常换行并保留阅读位置', async () => {
  test.setTimeout(120_000)
  const context = await fixture()
  try {
    const { page, application } = await context.launch()
    await page.emulateMedia({ reducedMotion: 'no-preference' })
    await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(1100, 900))
    const oldReply = Array.from({ length: 12 }, (_, i) => [
      `### 旧回复第 ${i + 1} 节`,
      '**富文本历史**中的正文应保留完整内容，侧栏开合不应反复排版屏幕外段落。'.repeat(12),
      String.raw`\[\sum_{k=1}^{n} k = \frac{n(n+1)}{2}\]`,
      '```text\n旧回复的原始文本块\n保留缩进与逐行内容\n```',
    ].join('\n\n')).join('\n\n')
    await finish(page, await context.send(page, '构造屏幕外的富文本旧回复'), oldReply)
    const old = page.locator('.chat-message.assistant').first()
    await expect(old.locator('.katex')).toHaveCount(12)
    await expect(old.locator('.message-code-block[data-language="text"]')).toHaveCount(12)
    const oldNode = await old.elementHandle()
    const visibleText = '末尾正文应随中心宽度连续换行，保持原有文字大小。'.repeat(14)
    const latestReply = Array.from({ length: 40 }, (_, i) => `### 后续第 ${i + 1} 节\n\n${'后续回答让旧富文本远离可见区域，并保留独立的阅读位置。'.repeat(6)}`).join('\n\n')
      + '\n\n```text\n' + Array.from({ length: 24 }, (_, i) => `后续文本块第 ${i + 1} 行`).join('\n') + '\n```\n\n' + visibleText
    await finish(page, await context.send(page, '继续到足够远的后续回复'), latestReply)
    await page.evaluate(() => document.fonts.ready.then(() => undefined))
    const area = page.getByRole('region', { name: '对话内容', exact: true })
    const bottomGap = () => area.evaluate(node => node.scrollHeight - node.clientHeight - node.scrollTop)
    await expect.poll(bottomGap).toBeLessThanOrEqual(2)
    const measure = () => page.evaluate(() => {
      // Only read the old message's outer box: descendants would force a
      // content-visibility subtree to lay out and invalidate this regression.
      const old = document.querySelector('.chat-message.assistant')!.getBoundingClientRect()
      const area = document.querySelector('.chat-messages')!.getBoundingClientRect()
      const paragraph = Array.from(document.querySelectorAll('.chat-message.assistant .message-markdown > p')).at(-1)!
      const visible = paragraph.getBoundingClientRect()
      const character = document.createRange()
      character.setStart(paragraph.firstChild!, 0); character.setEnd(paragraph.firstChild!, 1)
      const glyph = character.getBoundingClientRect()
      return { oldHeight: old.height, oldDistance: area.top - old.bottom, viewportHeight: area.height,
        width: visible.width, height: visible.height, glyphWidth: glyph.width, glyphHeight: glyph.height,
        visible: visible.top >= area.top && visible.bottom <= area.bottom,
        visibleContentVisibility: paragraph.closest<HTMLElement>('.chat-message')!.style.contentVisibility }
    })
    async function pauseToggle(label: string) {
      const count = await page.evaluate(async label => {
        const layout = document.querySelector('.chat-layout')!
        ;(layout.querySelector(`.chat-toolbar button[aria-label="${label}"]`) as HTMLButtonElement).click()
        await new Promise<void>(resolve => requestAnimationFrame(() => resolve()))
        const animations = Array.from(layout.children).flatMap(node => node.getAnimations())
        animations.forEach(animation => { animation.pause(); animation.currentTime = 60 })
        await new Promise<void>(resolve => requestAnimationFrame(() => resolve()))
        return animations.length
      }, label)
      expect(count).toBeGreaterThan(0)
      const frame = await measure()
      expect(frame.visibleContentVisibility).toBe('')
      return frame
    }
    async function settle(checkFinalAnchor = false) {
      const anchor = checkFinalAnchor ? await page.locator('.chat-layout').evaluate(async layout => {
        Array.from(layout.children).forEach(node => node.getAnimations().forEach(animation => {
          animation.pause()
          animation.currentTime = (animation.effect!.getComputedTiming().duration as number) - 1
        }))
        await new Promise<void>(resolve => requestAnimationFrame(() => resolve()))
        const area = layout.querySelector('.chat-messages')!, viewport = area.getBoundingClientRect()
        const visible = Array.from(area.querySelectorAll<HTMLElement>('.chat-message'), message => ({ message, rect: message.getBoundingClientRect() }))
          .find(({ rect }) => rect.bottom > viewport.top && rect.top < viewport.bottom)!
        return { id: visible.message.dataset.messageId!, top: visible.rect.top }
      }) : null
      await page.locator('.chat-layout').evaluate(layout => Array.from(layout.children).forEach(node => node.getAnimations().forEach(animation => animation.play())))
      await expect.poll(() => page.locator('.chat-layout').evaluate(layout => Array.from(layout.children).some(node => node.getAnimations().length))).toBe(false)
      await expect.poll(() => page.locator('.chat-message').evaluateAll(nodes => nodes.every(node => {
        const style = (node as HTMLElement).style
        return !style.contentVisibility && !style.getPropertyValue('contain-intrinsic-block-size')
      }))).toBe(true)
      if (anchor) expect(await page.locator(`.chat-message[data-message-id="${anchor.id}"]`).evaluate((node, top) => Math.abs(node.getBoundingClientRect().top - top), anchor.top)).toBeLessThanOrEqual(2)
    }
    for (const label of ['对话历史', '对话参数']) {
      const before = await measure()
      expect(before.oldDistance).toBeGreaterThan(before.viewportHeight * 3)
      const opening = await pauseToggle(label)
      expect(Math.abs(opening.oldHeight - before.oldHeight)).toBeLessThan(1)
      expect(opening.width).toBeLessThan(before.width - 70)
      expect(opening.height).toBeGreaterThan(before.height)
      expect(Math.abs(opening.glyphWidth - before.glyphWidth)).toBeLessThan(.1)
      expect(Math.abs(opening.glyphHeight - before.glyphHeight)).toBeLessThan(.1)
      expect(opening.visible).toBe(true)
      if (label === '对话历史') await settle()
      // The right panel reverses while paused; the left completes normally.
      // Both must retain the same offscreen height until their final animation.
      const closingFrom = await measure()
      const closing = await pauseToggle(label)
      expect(Math.abs(closing.oldHeight - closingFrom.oldHeight)).toBeLessThan(1)
      expect(closing.width).toBeGreaterThan(closingFrom.width)
      await settle()
      await expect.poll(bottomGap).toBeLessThanOrEqual(2)
    }
    await expect(page.locator('.chat-message')).toHaveCount(4)
    expect(await old.evaluate((node, original) => node === original, oldNode)).toBe(true)
    await expect(old.locator('.katex')).toHaveCount(12)
    await expect(old.locator('.message-code-block code')).toHaveText(Array(12).fill('旧回复的原始文本块\n保留缩进与逐行内容\n'))
    await expect(page.locator('.chat-message.assistant').last().locator('.message-markdown > p').last()).toHaveText(visibleText)

    await area.hover(); await page.mouse.wheel(0, -480)
    await expect.poll(bottomGap).toBeGreaterThan(300)
    let readingTop = -1, stableReadings = 0
    await expect.poll(async () => {
      const current = await area.evaluate(node => node.scrollTop)
      stableReadings = Math.abs(current - readingTop) < .5 ? stableReadings + 1 : 0
      readingTop = current
      return stableReadings
    }).toBeGreaterThanOrEqual(2)
    for (const label of ['对话历史', '对话参数']) {
      await pauseToggle(label); await settle(true)
      await pauseToggle(label); await settle(true)
      await expect.poll(() => area.evaluate((node, before) => Math.abs(node.scrollTop - before), readingTop)).toBeLessThanOrEqual(2)
    }
    expect(await bottomGap()).toBeGreaterThan(300)
    await expect(page.getByRole('button', { name: '返回最新', exact: true })).toBeVisible()
    expect(context.errors).toEqual([])
  } finally { await context.cleanup() }
})

test('Spark仅在终态显示明细，原生AVG与时长持久化且不冒充Decode', async () => {
  test.setTimeout(60_000)
  const context = await fixture()
  context.setInstances([{ ...textInstance, backend: 'sparkinfer' }])
  try {
    const { page } = await context.launch()
    await page.route('**/api/instances/*/telemetry', route => route.fulfill({ json: {
      instance_id: textInstance.id, backend: 'sparkinfer', state: 'ready', updated_at: Date.now() / 1000, unavailable: [],
      groups: [{ id: 'sparkinfer_prefix_cache', title: '前缀缓存', metrics: [
        { id: 'prefix_cache_host_bytes', label: '循环状态快照主机内存', value: 1048576, unit: 'bytes', kind: 'gauge' },
        { id: 'prefix_cache_host_kv_bytes', label: '主机 KV 层内存', value: 2097152, unit: 'bytes', kind: 'gauge' },
        { id: 'prefix_cache_tokens_reused_total', label: '累计复用输入 Token', value: 3000, unit: 'token', kind: 'counter' },
      ] }],
      live_generation: { request_key: context.requests.at(-1)?.clientCallId, client_call_id: context.requests.at(-1)?.clientCallId,
        client_request_id: context.requests.at(-1)?.clientRequestId, scope: 'request', source: '/metrics 请求调度', phase: 'output',
        backend_running: 1, backend_queued: 0, tokens_per_second: null, generated_tokens: null, available_for_chat: false, context_capacity: 8192 },
    } }))
    const call = await context.send(page, '检查 Spark 原生终态统计')
    delta(call, { content: '正在生成，尚未报告用量。' })
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '运行监控', exact: true }).click()
    const monitor = page.locator('.monitor-page')
    await expect(monitor.locator('.monitor-request-section')).toHaveCount(0)
    await expect(monitor.locator('.monitor-speed-panel')).toHaveCount(0)
    const usage = { prompt_tokens: 1000, completion_tokens: 200, total_tokens: 1200, prompt_tokens_details: { cached_tokens: 400 },
      ttft_ms: 500, generation_ms: 4000, decode_tps: 50 }
    event(call, { choices: [{ delta: {}, finish_reason: 'stop' }] })
    event(call, { choices: [], usage })
    event(call, { modloader_request: { request_key: call.clientCallId, client_call_id: call.clientCallId, status: 'completed', duration_s: 6,
      first_output_s: 1.2, usage, timings: {} } })
    call.response.end('data: [DONE]\n\n')
    const row = monitor.locator('.monitor-request-table tbody tr[data-request-index="1"]')
    await expect(row).toBeVisible()
    await expect(monitor.locator('.monitor-turn-summary')).toContainText('总输入1,000token')
    await expect(monitor.locator('.monitor-turn-summary')).toContainText('总输出200token')
    await expect(row.locator('[data-metric="engine"] .monitor-request-readout')).toHaveText('50.00AVG')
    await expect(row.locator('[data-metric="ttft"] .monitor-request-readout')).toHaveText('0.50')
    await expect(row.locator('[data-metric="ttft"]')).toHaveAttribute('data-tooltip', /引擎 · 网关 1.2 秒/)
    await expect(row.locator('[data-metric="elapsed"]')).toHaveAttribute('data-tooltip', /引擎生成 4 秒/)
    await expect(monitor.locator('.monitor-speed-panel')).toHaveCount(0)
    await monitor.getByRole('button', { name: '刷新监控', exact: true }).click()
    await expect(monitor.locator('.monitor-current-state')).toHaveText('已完成')
    await monitor.getByText('更多后端指标', { exact: true }).click()
    await expect(monitor.getByText('循环状态快照主机内存', { exact: true })).toBeVisible()
    await expect(monitor.getByText('主机 KV 层内存', { exact: true })).toBeVisible()
    await monitor.locator('.monitor-request-section').screenshot({ path: path.join(context.directory, 'spark-final-details.png') })
    await monitor.locator('.telemetry-group').filter({ hasText: '前缀缓存' }).screenshot({ path: path.join(context.directory, 'spark-prefix-cache.png') })
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '测试对话', exact: true }).click()
    await expect(page.locator('.message-turn-summary [data-metric="speed"]')).toContainText('AVG')
    await expect(page.locator('.message-turn-summary [data-metric="input"]')).toContainText('600')
    await saved(page)
    await page.reload(); await pinNavigation(page)
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '测试对话', exact: true }).click()
    await expect(page.locator('.message-turn-summary [data-metric="speed"]')).toContainText('AVG50.00')
    await page.getByRole('button', { name: '查看本轮 1 次模型请求统计', exact: true }).click()
    await expect(row.locator('[data-metric="engine"] .monitor-request-readout')).toHaveText('50.00AVG')
    expect(context.errors).toEqual([])
  } finally { await context.cleanup() }
})

test('连续usage无需解码计时即可更新实时曲线，优先于旧轮询', async () => {
  test.setTimeout(60_000)
  const context = await fixture()
  context.setInstances([{ ...textInstance, backend: 'vllm' }])
  try {
    const { application, page } = await context.launch()
    await page.route('**/api/instances/*/telemetry', route => route.fulfill({ json: {
      instance_id: textInstance.id, backend: 'vllm', state: 'ready', updated_at: Date.now() / 1000, groups: [], unavailable: [],
      live_generation: { request_key: context.requests.at(-1)?.clientCallId, client_call_id: context.requests.at(-1)?.clientCallId,
        client_request_id: context.requests.at(-1)?.clientRequestId, scope: 'request', source: '轮询夹具', phase: 'output',
        backend_running: 1, backend_queued: 0, tokens_per_second: 4, window_s: 3, available_for_chat: true, context_capacity: 8192 },
    } }))
    const call = await context.send(page, '检查连续 usage')
    const emit = (completion_tokens: number) => event(call, { choices: [{ delta: { content: '继续。' } }], usage: {
      prompt_tokens: 1000, completion_tokens, prompt_tokens_details: { cached_tokens: 400 },
    } })
    emit(10)
    await expect(page.locator('.message-turn-summary [data-metric="output"]')).toContainText('10')
    emit(30)
    await expect(page.locator('.message-turn-summary [data-metric="output"]')).toContainText('30')
    emit(80)
    await expect(page.locator('.message-turn-summary [data-metric="output"]')).toContainText('80')
    await expect(page.locator('.message-turn-summary [data-metric="speed"]')).toContainText('TPS')
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '运行监控', exact: true }).click()
    const monitor = page.locator('.monitor-page')
    await monitor.getByRole('button', { name: '实时', exact: true }).click()
    await monitor.getByRole('button', { name: '刷新监控', exact: true }).click()
    await expect(monitor.locator('.monitor-context-capacity')).toContainText('8,192')
    await expect(monitor.locator('.monitor-speed-current strong')).not.toHaveText('4.00')
    await expect(monitor.locator('.monitor-speed-note')).toContainText('接收采样')
    await expect(monitor.locator('.monitor-generation-chart .trend-line')).toHaveCount(1)
    const opened = application.waitForEvent('window')
    await monitor.getByRole('button', { name: '小窗监控', exact: true }).click()
    const mini = await opened
    await expect(mini.locator('[data-mini-metric="current"] strong')).toHaveText(await monitor.locator('.monitor-speed-current strong').innerText())
    emit(120)
    await expect(page.locator('.message-turn-summary [data-metric="output"]')).toContainText('120')
    emit(200)
    await expect(page.locator('.message-turn-summary [data-metric="output"]')).toContainText('200')
    await expect(monitor.locator('.monitor-speed-current strong')).not.toHaveText('4.00')
    await expect(mini.locator('[data-mini-metric="current"] strong')).toHaveText(await monitor.locator('.monitor-speed-current strong').innerText())
    await mini.screenshot({ path: path.join(context.directory, 'usage-mini-live.png') })
    await page.screenshot({ path: path.join(context.directory, 'usage-arrival-live.png') })
    event(call, { choices: [{ delta: {}, finish_reason: 'stop' }] })
    call.response.end('data: [DONE]\n\n')
    expect(context.errors).toEqual([])
  } finally { await context.cleanup() }
})

test('Strata终帧前实时计数贯通聊天与请求明细，结束后保留原生报告', async () => {
  test.setTimeout(60_000)
  const context = await fixture()
  context.setInstances([{ ...textInstance, backend: 'strata' }])
  try {
    const { page } = await context.launch()
    let observed = { prompt_tokens: 1000, generated_tokens: 200, average_tokens_per_second: 100, request_elapsed_s: 3,
      prefill_cached_tokens: null as number | null, prefill_processed_tokens: null as number | null }
    await page.route('**/api/instances/*/telemetry', route => {
      const call = context.requests.at(-1)
      return route.fulfill({ json: {
        instance_id: textInstance.id, backend: 'strata', state: 'ready', updated_at: Date.now() / 1000, groups: [], unavailable: [],
        live_generation: { ...observed, client_request_id: call?.clientRequestId, client_call_id: call?.clientCallId,
          request_key: call?.clientCallId, scope: 'request', source: 'Strata /metrics live', backend_running: call ? 1 : 0, backend_queued: 0,
          phase: 'thinking', available_for_chat: true, reason: null, tokens_per_second: 150, window_s: 3,
          average_basis: 'backend_decode', average_prefill_tokens_per_second: 800, prefill_average_basis: 'backend_prefill',
          context_capacity: 262144, context_tokens: observed.prompt_tokens + observed.generated_tokens, first_output_s: .8 },
      } })
    })
    const first = await context.send(page, '持续思考，观察 Strata 真实统计。')
    delta(first, { reasoning_content: '正在思考。' }) // This engine sends no usage/timings until the final frame.
    const footer = page.locator('.chat-message.assistant').last().locator('.message-turn-summary')
    await expect(footer.locator('[data-metric="output"]')).toHaveText('200')
    await expect(footer.locator('[data-metric="input"]')).toHaveText('—')
    await expect(footer.locator('[data-metric="cached"]')).toHaveText('—')
    await expect(footer.locator('[data-metric="speed"]')).toHaveText('DEC100.00tok/s')
    observed = { ...observed, prefill_cached_tokens: 600, prefill_processed_tokens: 400 }
    await expect(footer.locator('[data-metric="input"]')).toHaveText('400')
    await expect(footer.locator('[data-metric="cached"]')).toHaveText('600')
    // Begin just after a displayed update: the next real observation must not wait a second or three.
    observed = { ...observed, generated_tokens: 250, average_tokens_per_second: 125, request_elapsed_s: 3.5 }
    await expect(footer.locator('[data-metric="output"]')).toHaveText('250', { timeout: 900 })
    await expect(footer.locator('[data-metric="speed"]')).toHaveText('DEC125.00tok/s')
    await footer.screenshot({ path: path.join(context.directory, 'strata-live-footer.png') })
    await footer.getByRole('button', { name: '查看本轮 1 次模型请求统计' }).click()
    const monitor = page.locator('.monitor-page'), row = monitor.locator('.monitor-request-table tbody tr[data-request-index]').first()
    await expect(row.locator('[data-metric="input"] .monitor-request-readout strong')).toHaveText('1,000')
    await expect(row.locator('[data-metric="output"] .monitor-request-readout strong')).toHaveText('250')
    await expect(row.locator('[data-metric="cached"] .monitor-request-readout strong')).toHaveText('600')
    await expect(row.locator('[data-metric="elapsed"] .monitor-request-readout strong')).toHaveText('3.50')
    await expect(row.locator('[data-metric="ttft"] .monitor-request-readout strong')).toHaveText('0.80')
    await expect(row.locator('[data-metric="decode"] .monitor-request-readout strong')).toHaveText('125.00')
    await expect(row.locator('[data-metric="prefill"] .monitor-request-readout strong')).toHaveText('800.00')
    await expect(monitor.locator('.monitor-speed-current strong')).toHaveText('125.00')
    await monitor.getByRole('button', { name: '实时', exact: true }).click()
    await expect(monitor.locator('.monitor-speed-current strong')).toHaveText('150.00')
    await monitor.screenshot({ path: path.join(context.directory, 'strata-live-monitor.png') })
    event(first, { choices: [{ delta: { content: '完成。' }, finish_reason: 'stop' }],
      usage: { prompt_tokens: 1000, completion_tokens: 300, prompt_tokens_details: { cached_tokens: 600 } },
      timings: { predicted_n: 300, predicted_ms: 2400, prompt_n: 400, prompt_ms: 500, cache_n: 600 } })
    first.response.end('data: [DONE]\n\n')
    await expect(row.locator('[data-metric="output"] .monitor-request-readout strong')).toHaveText('300')
    const id = (await management<ConversationList>(page, '/api/conversations')).active_id!
    await expect.poll(async () => (await management<any>(page, `/api/conversations/${id}`)).messages.at(-1)?.turn?.status).toBe('completed')
    const stored = (await management<any>(page, `/api/conversations/${id}`)).messages.at(-1)
    expect(stored.requests[0].usage.completion_tokens).toBe(300)
    expect(stored.requests[0].live.generated_tokens).toBe(250)
    expect(stored.requests[0].elapsed).toBeGreaterThan(0)
    expect(stored.requests[0].elapsed).not.toBe(3.5) // The final duration is not frozen at the last telemetry sample.
    await page.reload(); await pinNavigation(page)
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '测试对话', exact: true }).click()
    await expect(footer.locator('[data-metric="output"]')).toHaveText('300')
    await expect(footer.locator('[data-metric="cached"]')).toHaveText('600')
    expect(context.errors).toEqual([])
  } finally { await context.cleanup() }
})

test('流式统计即时刷新与长请求列表跟随、暂停及恢复', async () => {
  test.setTimeout(90_000)
  const context = await fixture(true)
  try {
    const { page } = await context.launch()
    // The real stream supplies all speed observations; the polling endpoint cannot.
    await page.route('**/api/instances/*/telemetry', route => route.fulfill({ json: {
      instance_id: textInstance.id, backend: 'llama_cpp', state: 'ready', updated_at: Date.now() / 1000, groups: [], unavailable: [],
      live_generation: { request_key: null, tokens_per_second: null, available_for_chat: false, backend_running: 0, context_capacity: 4096 },
    } }))
    const id = crypto.randomUUID()
    await management(page, '/api/conversations', 'POST', { id, title: '连续工具计量',
      selection: { profile_id: textInstance.profile_id, model_id: textInstance.model_id, alias: textInstance.alias },
      draft: { text: '', images: [] }, settings: { system: '', stream: true, toolsDraft: '', webSearch: false, basicTools: true, webSearchRounds: 30, autoCompact: false } })
    await management(page, `/api/conversations/${id}`, 'PATCH', { activate: true })
    await page.reload(); await pinNavigation(page)
    const chat = () => page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '测试对话', exact: true }).click()
    await chat()
    const first = await context.send(page, '连续调用计算工具并报告结果。')
    const assistant = page.locator('.chat-message.assistant').last(), footer = assistant.locator('.message-turn-summary')
    const stream = (request: ChatRequest, tokens: number, milliseconds: number) => event(request, {
      choices: [{ delta: { reasoning_content: '核对。' } }],
      usage: { prompt_tokens: 300, completion_tokens: tokens, prompt_tokens_details: { cached_tokens: 200 } },
      timings: { predicted_n: tokens, predicted_ms: milliseconds },
    })
    stream(first, 1, 0)
    await expect(footer.locator('[data-metric="input"]')).toHaveText('100')
    await expect(footer.locator('[data-metric="cached"]')).toHaveText('200')
    await expect(footer.locator('[data-metric="output"]')).toHaveText('1')
    await expect(footer.locator('[data-metric="output"]')).toHaveAttribute('aria-label', '当前请求输出 1 Token')
    stream(first, 11, 100)
    await expect(page.locator('[data-metric="live-speed"]')).toHaveText('100.00tok/s')
    stream(first, 31, 200)
    await expect(page.locator('[data-metric="live-speed"]')).toHaveText('150.00tok/s')
    await footer.getByRole('button', { name: '查看本轮 1 次模型请求统计' }).click()
    const monitor = page.locator('.monitor-page'), chart = monitor.locator('.monitor-generation-chart')
    await monitor.getByRole('button', { name: '实时', exact: true }).click()
    await expect(monitor.locator('.monitor-speed-current strong')).toHaveText('150.00')
    stream(first, 51, 300)
    await expect(monitor.locator('.monitor-speed-current strong')).toHaveText('166.67')
    await expect(chart.locator('.trend-line')).toHaveCount(1)
    await monitor.getByRole('button', { name: '本轮', exact: true }).click()
    await expect(monitor.locator('.monitor-speed-current strong')).toHaveText('170.00')
    await monitor.locator('.monitor-speed-panel').screenshot({ path: path.join(context.directory, 'stream-rate.png') })
    const toolReply = async (request: ChatRequest, index: number) => {
      event(request, { choices: [{ delta: { tool_calls: [{ index: 0, id: `calc-${index}`, type: 'function', function: { name: 'calculate', arguments: '{"expression":"6*7"}' } }] }, finish_reason: 'tool_calls' }],
        usage: { prompt_tokens: 300 + index * 10, completion_tokens: 51, prompt_tokens_details: { cached_tokens: 200 } }, timings: { predicted_n: 51, predicted_ms: 300 } })
      request.response.end('data: [DONE]\n\n')
      await expect.poll(() => context.requests.length).toBe(index + 1)
    }
    await toolReply(first, 1)
    await chat()
    // A new request with no statistics is unknown, not the previous request's work.
    await expect(footer.locator('[data-metric="output"]')).toHaveText('—')
    await expect(footer.locator('[data-metric="output"]')).toHaveAttribute('aria-label', '当前请求输出 未知 Token')
    for (let index = 2; index <= 11; index++) await toolReply(context.requests[index - 1], index)
    await footer.getByRole('button', { name: '查看本轮 12 次模型请求统计' }).click()
    const scroller = monitor.getByRole('region', { name: '请求明细列表', exact: true })
    const rows = monitor.locator('.monitor-request-table tbody tr[data-request-index]')
    const gap = () => scroller.evaluate(node => node.scrollHeight - node.scrollTop - node.clientHeight)
    const position = () => scroller.evaluate(node => node.scrollTop)
    await expect(rows).toHaveCount(12)
    await scroller.scrollIntoViewIfNeeded()
    await expect.poll(gap).toBeLessThanOrEqual(2)
    expect(await position()).toBeGreaterThan(0)
    expect(await rows.last().locator('th').evaluate(node => getComputedStyle(node, '::before').content)).toBe('none')
    await scroller.hover(); await page.mouse.wheel(0, -240)
    await expect.poll(gap).toBeGreaterThan(80)
    const parked = await position()
    await toolReply(context.requests[11], 12)
    await expect(rows).toHaveCount(13)
    expect(await position()).toBeCloseTo(parked, 0)
    // Native keyboard scrolling back to the tail resumes following.
    await scroller.focus(); await page.keyboard.press('End')
    await expect.poll(gap).toBeLessThanOrEqual(2)
    await toolReply(context.requests[12], 13)
    await expect(rows).toHaveCount(14)
    await expect.poll(gap).toBeLessThanOrEqual(2)
    // A real text-scale/layout change grows existing rows without appending one.
    const oldHeight = await scroller.evaluate(node => node.scrollHeight)
    const oldScale = await page.evaluate(() => document.documentElement.style.getPropertyValue('--font-scale'))
    await page.evaluate(() => document.documentElement.style.setProperty('--font-scale', '1.3'))
    await expect.poll(() => scroller.evaluate(node => node.scrollHeight)).toBeGreaterThan(oldHeight)
    await expect.poll(gap).toBeLessThanOrEqual(2)
    await page.evaluate(value => document.documentElement.style.setProperty('--font-scale', value), oldScale)
    await expect.poll(gap).toBeLessThanOrEqual(2)
    await scroller.hover(); await page.mouse.wheel(0, -120)
    await expect(scroller).toHaveClass(/is-scrolling/)
    const inset = () => scroller.evaluate(node => parseFloat(getComputedStyle(node).getPropertyValue('--monitor-request-thumb-inset')))
    await expect.poll(inset).toBeCloseTo(2, 2)
    const movingWidth = await scroller.evaluate(node => node.clientWidth)
    await page.mouse.move(0, 0)
    await expect(scroller).not.toHaveClass(/is-scrolling/)
    await expect.poll(inset).toBeCloseTo(3.5, 2)
    expect(await scroller.evaluate(node => node.clientWidth)).toBe(movingWidth)
    await scroller.focus(); await page.keyboard.press('End')
    await expect.poll(gap).toBeLessThanOrEqual(2)
    // Explicit chart navigation wins over automatic tail following.
    await chart.getByRole('button', { name: /^请求 #1，/ }).click()
    await expect.poll(position).toBeLessThanOrEqual(2)
    await toolReply(context.requests[13], 14)
    await expect(rows).toHaveCount(15)
    expect(await position()).toBeLessThanOrEqual(2)
    const last = context.requests[14]
    event(last, { choices: [{ delta: { content: '全部完成。' }, finish_reason: 'stop' }], usage: { prompt_tokens: 1200, completion_tokens: 8, prompt_tokens_details: { cached_tokens: 1000 } }, timings: { predicted_n: 8, predicted_ms: 80 } })
    last.response.end('data: [DONE]\n\n')
    await expect.poll(async () => (await management<{ messages: Array<{ turn?: { status: string } }> }>(page, `/api/conversations/${id}`)).messages.at(-1)?.turn?.status).toBe('completed')
    expect(await position()).toBeLessThanOrEqual(2)
    await expect(monitor.locator('.monitor-request-table thead th')).toHaveText(['模型请求', 'Token', '时延 · s', '速度 · tok/s'])
    await expect(rows.first().locator('[data-group="tokens"] [data-metric]')).toHaveCount(3)
    await expect(rows.first().locator('[data-group="latency"] [data-metric]')).toHaveCount(2)
    // Compare actual text geometry: different digit counts share a right edge,
    // while the reserved space never separates the icon from its visible value.
    const geometry = await rows.evaluateAll(nodes => nodes.flatMap(row => Array.from(row.querySelectorAll<HTMLElement>('.monitor-request-value')).map(metric => {
      const readout = metric.querySelector<HTMLElement>('.monitor-request-readout')!, number = readout.querySelector('strong')!, icon = readout.querySelector('svg')
      const digits: number[] = [], text = number.firstChild
      if (text?.nodeType === Node.TEXT_NODE) for (let index = 0; index < text.textContent!.length; index++) if (/\d/.test(text.textContent![index])) {
        const range = document.createRange(); range.setStart(text, index); range.setEnd(text, index + 1); digits.push(range.getBoundingClientRect().width)
      }
      return { name: metric.dataset.metric!, right: number.getBoundingClientRect().right, gap: icon ? number.getBoundingClientRect().left - icon.getBoundingClientRect().right : null,
        digits, secondary: metric.classList.contains('is-secondary') }
    })))
    for (const name of ['input', 'output', 'cached', 'elapsed', 'ttft', 'decode', 'prefill']) {
      const positions = geometry.filter(item => item.name === name).map(item => item.right)
      expect(Math.max(...positions) - Math.min(...positions)).toBeLessThan(.6)
    }
    expect(geometry.every(item => item.gap === null || item.gap >= 4 && item.gap <= 6)).toBe(true)
    const mainDigits = geometry.filter(item => !item.secondary).flatMap(item => item.digits)
    expect(Math.max(...mainDigits) - Math.min(...mainDigits)).toBeLessThan(.1)
    await rows.first().locator('[data-metric="ttft"]').hover()
    await expect(page.getByRole('tooltip')).toContainText('首输出等待')
    await page.mouse.move(0, 0)
    await expect(page.getByRole('tooltip')).toBeHidden()
    await monitor.locator('.monitor-request-section').screenshot({ path: path.join(context.directory, 'request-list-scrolled.png') })
    await chat()
    await expect(footer.locator('[data-metric="input"]')).toHaveText('200')
    await expect(footer.locator('[data-metric="cached"]')).toHaveText('1,000')
    await expect(footer.locator('[data-metric="output"]')).toHaveText('8')
    await expect(footer.locator('[data-metric="output"]')).toHaveAttribute('aria-label', '末次请求输出 8 Token')
    await footer.screenshot({ path: path.join(context.directory, 'last-request-footer.png') })
    expect(context.errors).toEqual([])
  } finally { await context.cleanup() }
})

test('生成持续滚动时统计悬停提示保持，数字贴近图标', async () => {
  test.setTimeout(60_000)
  const context = await fixture()
  try {
    const { page } = await context.launch()
    const request = await context.send(page, '持续输出时查看统计提示')
    const assistant = page.locator('.chat-message.assistant').last()
    delta(request, { content: Array.from({ length: 35 }, (_, index) => `第 ${index} 段正文。`).join('\n\n') })
    await expect(assistant.locator('.message-text')).toContainText('第 34 段正文。')
    const scroller = page.locator('.chat-messages'), elapsed = assistant.locator('[data-metric="elapsed"]')
    await expect.poll(() => scroller.evaluate(node => node.scrollTop)).toBeGreaterThan(0)
    await elapsed.hover()
    const tooltip = page.getByRole('tooltip')
    await expect(tooltip).toContainText('本轮耗时')
    for (let index = 0; index < 4; index++) {
      const previousScroll = await scroller.evaluate(node => node.scrollTop)
      const text = `继续输出第 ${index} 段。`
      delta(request, { content: `\n\n${text}` })
      await expect(assistant.locator('.message-text')).toContainText(text)
      await expect.poll(() => scroller.evaluate(node => node.scrollTop)).toBeGreaterThan(previousScroll)
      await expect(tooltip).toBeVisible()
      await expect(tooltip).toContainText('本轮耗时')
    }
    await page.screenshot({ path: path.join(context.directory, 'streaming-tooltip.png') })
    await page.mouse.move(0, 0)
    await expect(tooltip).toBeHidden()
    await finish(page, request, '完成。')
    const gaps = await assistant.locator('.message-metric-value').evaluateAll(nodes => nodes.map(node => {
      const range = document.createRange(); range.selectNodeContents(node)
      return node.getBoundingClientRect().width - range.getBoundingClientRect().width
    }))
    expect(gaps.every(gap => gap < 1)).toBe(true)
    await assistant.locator('.message-turn-summary').screenshot({ path: path.join(context.directory, 'compact-metrics.png') })

    const hoverElapsed = async () => {
      await page.mouse.move(0, 0)
      await elapsed.hover()
      await expect(tooltip).toBeVisible()
      await expect(tooltip).toContainText('本轮耗时')
    }
    // Scrolling the trigger away still closes, as do hidden and inert sections.
    await hoverElapsed()
    await scroller.evaluate(node => { node.scrollTop -= 80 })
    await expect(tooltip).toBeHidden()
    await hoverElapsed()
    await elapsed.evaluate(node => { node.parentElement!.hidden = true })
    await expect(tooltip).toBeHidden()
    await elapsed.evaluate(node => { node.parentElement!.hidden = false })
    await hoverElapsed()
    await elapsed.evaluate(node => { node.parentElement!.inert = true })
    await expect(tooltip).toBeHidden()
    await elapsed.evaluate(node => { node.parentElement!.inert = false })

    // Escape stays dismissed while focused, but blur permits a fresh mouse visit.
    await page.mouse.move(0, 0)
    await elapsed.focus()
    await page.keyboard.press('Tab')
    await page.keyboard.press('Shift+Tab')
    await expect(tooltip).toBeVisible()
    await page.keyboard.press('Escape')
    await expect(tooltip).toBeHidden()
    await expect(elapsed).toBeFocused()
    await elapsed.evaluate(node => node.blur())
    await hoverElapsed()

    await page.mouse.move(0, 0)
    await page.evaluate(() => {
      const node = document.createElement('span')
      node.id = 'transient-tooltip-trigger'; node.dataset.tooltip = '临时目标'
      node.textContent = '临时目标'; node.style.cssText = 'position:fixed;left:120px;top:120px'
      document.body.append(node)
    })
    const transient = page.locator('#transient-tooltip-trigger')
    await transient.hover()
    await expect(tooltip).toHaveText('临时目标')
    await transient.evaluate(node => node.remove())
    await expect(tooltip).toBeHidden()
    expect(context.errors).toEqual([])
  } finally { await context.cleanup() }
})

test('生成中保存提示稳定，检查点按秒合并且用户头像不重复昵称', async () => {
  test.setTimeout(90_000)
  const context = await fixture()
  try {
    const { page } = await context.launch()
    const request = await context.send(page, '验证保存提示与用户头像')
    const status = page.locator('.chat-save-status')
    await expect(status).toHaveText('自动保存')
    const user = page.locator('.chat-message.user').last()
    await expect(user.locator('.message-label strong')).toHaveText('你')
    await expect(user.locator('.message-avatar')).toHaveText('')
    await expect(user.locator('.message-avatar svg')).toHaveCount(1)
    const writes: number[] = []
    page.on('request', request => {
      if (request.method() === 'PATCH' && request.url().includes('/api/conversations/')) writes.push(Date.now())
    })
    await status.evaluate(element => {
      const changes: string[] = []
      const observer = new MutationObserver(() => changes.push(element.textContent || ''))
      observer.observe(element, { childList: true, subtree: true, characterData: true })
      ;(window as any).saveStatusSample = { changes, observer }
    })
    const started = Date.now()
    for (let index = 0; index < 40; index++) {
      delta(request, index < 20 ? { reasoning_content: `思考${index}。` } : { content: `正文${index}。` })
      await new Promise(resolve => setTimeout(resolve, 80))
    }
    await expect(page.locator('.chat-message.assistant').last()).toContainText('正文39。')
    const changes = await page.evaluate(() => {
      const sample = (window as any).saveStatusSample
      sample.observer.disconnect()
      return sample.changes as string[]
    })
    expect(changes).toEqual([])
    expect(writes.length).toBeGreaterThanOrEqual(2)
    expect(writes.length).toBeLessThanOrEqual(Math.ceil((Date.now() - started) / 1000) + 1)
    for (let index = 1; index < writes.length; index++) expect(writes[index] - writes[index - 1]).toBeGreaterThanOrEqual(850)
    console.log('Chat checkpoint writes:', writes.length, 'during ms:', Date.now() - started)
    const id = (await management<ConversationList>(page, '/api/conversations')).active_id!
    expect((await management<Conversation>(page, `/api/conversations/${id}`)).messages[1].status).toBe('generating')
    await page.screenshot({ path: path.join(context.directory, 'stable-autosave-avatar.png') })
    let rejectCheckpoint = true
    await page.route('**/api/conversations/*', route => rejectCheckpoint && route.request().method() === 'PATCH'
      ? route.fulfill({ status: 503, json: { detail: 'fixture: 检查点保存失败' } }) : route.continue())
    delta(request, { content: '检查保存异常。' })
    await expect(status).toHaveText('未保存')
    rejectCheckpoint = false
    await page.getByRole('button', { name: '重试保存', exact: true }).click()
    await expect(status).toHaveText('自动保存')
    expect(context.requests).toHaveLength(1)
    await finish(page, request, '保存提示验收完成。')
    expect((await management<Conversation>(page, `/api/conversations/${id}`)).messages[1]).toMatchObject({ status: 'completed', text: expect.stringContaining('正文39。检查保存异常。保存提示验收完成。') })
    expect(context.errors).toEqual([])
  } finally { await context.cleanup() }
})

test('提问导航从当前位置平滑滚动，可被手动中断并尊重减少动态效果', async () => {
  test.setTimeout(90_000)
  const context = await fixture()
  try {
    const { page, application } = await context.launch()
    await page.emulateMedia({ reducedMotion: 'no-preference' })
    await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(1100, 900))
    const text = Array.from({ length: 30 }, (_, index) => `第 ${index + 1} 段：用于验证长距离滚动导航。\n\n`).join('')
    await finish(page, await context.send(page, '第一条平滑滚动提问'), text)
    await finish(page, await context.send(page, '第二条平滑滚动提问'), text)
    await finish(page, await context.send(page, '第三条平滑滚动提问'), text)
    const area = page.getByRole('region', { name: '对话内容', exact: true })
    const top = () => area.evaluate(element => element.scrollTop)
    const paused = await page.locator('.chat-layout').evaluate(async layout => {
      ;(layout.querySelector('.chat-toolbar button[aria-label="对话参数"]') as HTMLButtonElement).click()
      await new Promise<void>(resolve => requestAnimationFrame(() => resolve()))
      const animations = Array.from(layout.children).flatMap(node => node.getAnimations())
      animations.forEach(animation => { animation.pause(); animation.currentTime = 60 })
      await new Promise<void>(resolve => requestAnimationFrame(() => resolve()))
      return animations.length
    })
    expect(paused).toBeGreaterThan(0)
    const before = await top()
    await area.evaluate(element => {
      const sample = { values: [] as number[], listener: () => sample.values.push(element.scrollTop) }
      element.addEventListener('scroll', sample.listener)
      ;(window as any).promptScrollSample = sample
    })
    await page.getByRole('button', { name: '上一条提问', exact: true }).click()
    expect(await page.locator('.chat-layout').evaluate(layout => Array.from(layout.children).some(node => node.getAnimations().length))).toBe(false)
    expect(await page.locator('.chat-message').evaluateAll(nodes => nodes.every(node => !(node as HTMLElement).style.contentVisibility))).toBe(true)
    await expect.poll(top).toBeLessThan(before - 100)
    await expect.poll(() => page.locator('.chat-message.user').last().evaluate(element => Math.abs(element.getBoundingClientRect().top - element.closest('.chat-messages')!.getBoundingClientRect().top - 16))).toBeLessThan(2)
    const samples = await area.evaluate(element => {
      const sample = (window as any).promptScrollSample
      element.removeEventListener('scroll', sample.listener)
      return sample.values as number[]
    })
    const after = await top()
    expect(samples.some(value => value < before - 2 && value > after + 2)).toBe(true)
    expect(new Set(samples).size).toBeGreaterThan(2)
    const secondTarget = await page.locator('.chat-message.user').nth(1).evaluate(element => {
      const area = element.closest('.chat-messages')!
      return area.scrollTop + element.getBoundingClientRect().top - area.getBoundingClientRect().top - 16
    })
    await page.getByRole('button', { name: '上一条提问', exact: true }).click()
    await area.hover()
    await page.mouse.wheel(0, 160)
    await new Promise(resolve => setTimeout(resolve, 350))
    const interrupted = await top()
    expect(interrupted).toBeGreaterThan(secondTarget + 100)
    await new Promise(resolve => setTimeout(resolve, 200))
    expect(Math.abs(await top() - interrupted)).toBeLessThan(2)
    await page.emulateMedia({ reducedMotion: 'reduce' })
    await area.press('Home')
    await expect.poll(top).toBe(0)
    await page.getByRole('button', { name: '上一条提问', exact: true }).click()
    await page.getByRole('button', { name: '下一条提问', exact: true }).click()
    const next = await page.locator('.chat-message.user').nth(1).evaluate(element => Math.abs(element.getBoundingClientRect().top - element.closest('.chat-messages')!.getBoundingClientRect().top - 16))
    expect(next).toBeLessThan(2)
    expect(context.errors).toEqual([])
  } finally { await context.cleanup() }
})

test('多轮回传流式和非流式思考，重启、纯思考回复与模型切换保留原文', async () => {
  test.setTimeout(120_000)
  const context = await fixture()
  context.setInstances([{ ...textInstance, backend: 'ninfer' }])
  try {
    let { page } = await context.launch()
    const reasoning = '逐步核对。\n  保留缩进和末尾空白。  '
    const first = await context.send(page, '第一轮问题')
    expect(first.body).not.toHaveProperty('preserve_thinking')
    expect(first.body.messages).toEqual([{ role: 'user', content: '第一轮问题' }])
    delta(first, { reasoning_content: reasoning })
    await finish(page, first, '第一轮正文')
    const id = (await management<ConversationList>(page, '/api/conversations')).active_id!
    await page.getByRole('button', { name: '对话参数', exact: true }).click()
    await page.getByRole('checkbox', { name: '流式输出', exact: true }).uncheck()
    await page.getByRole('button', { name: '收起对话参数', exact: true }).click()
    const second = await context.send(page, '第二轮问题')
    const firstHistory = [{ role: 'user', content: '第一轮问题' }, { role: 'assistant', content: '第一轮正文', reasoning_content: reasoning }]
    expect(second.body.messages).toEqual([...firstHistory, { role: 'user', content: '第二轮问题' }])
    expect(second.body).toMatchObject({ stream: false, preserve_thinking: true })
    second.response.end(JSON.stringify({ choices: [{ message: { role: 'assistant', reasoning: '非流式思考', content: '第二轮正文' }, finish_reason: 'stop' }] }))
    await expect(page.getByRole('button', { name: '停止生成', exact: true })).toHaveCount(0)
    await saved(page)
    await context.quit()
    ;({ page } = await context.launch())
    const third = await context.send(page, '重启后第三轮问题')
    const secondHistory = [...firstHistory, { role: 'user', content: '第二轮问题' }, { role: 'assistant', content: '第二轮正文', reasoning_content: '非流式思考' }]
    expect(third.body.messages).toEqual([...secondHistory, { role: 'user', content: '重启后第三轮问题' }])
    expect(third.body.preserve_thinking).toBe(true)
    third.response.end(JSON.stringify({ choices: [{ message: { role: 'assistant', content: '<think>只返回思考，没有正文</think>' }, finish_reason: 'length' }] }))
    await expect(page.getByRole('button', { name: '停止生成', exact: true })).toHaveCount(0)
    await saved(page)
    expect((await management<Conversation>(page, `/api/conversations/${id}`)).messages.at(-1)).toMatchObject({ text: '', reasoning: '只返回思考，没有正文' })
    context.setInstances([imageInstance])
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await expect(chatModel(page)).toHaveJSProperty('value', imageInstance.id)
    const fourth = await context.send(page, '切换模型后继续')
    expect(fourth.body.model).toBe(imageInstance.alias)
    expect(fourth.body).not.toHaveProperty('preserve_thinking')
    expect(fourth.body.messages).toEqual([...secondHistory, { role: 'user', content: '重启后第三轮问题' }, { role: 'assistant', content: '', reasoning_content: '只返回思考，没有正文' }, { role: 'user', content: '切换模型后继续' }])
    fourth.response.end(JSON.stringify({ choices: [{ message: { role: 'assistant', content: '普通正文' }, finish_reason: 'stop' }] }))
    await expect(page.getByRole('button', { name: '停止生成', exact: true })).toHaveCount(0)
    await saved(page)
    await page.screenshot({ path: path.join(context.directory, 'reasoning-replay.png') })
    const clean = await context.send(page, '没有思考的回复不添加空字段')
    expect(clean.body.messages.at(-2)).toEqual({ role: 'assistant', content: '普通正文' })
    clean.response.end(JSON.stringify({ choices: [{ message: { role: 'assistant', content: '完成' }, finish_reason: 'stop' }] }))
    await expect(page.getByRole('button', { name: '停止生成', exact: true })).toHaveCount(0)
    await saved(page)
    expect(context.errors).toEqual([])
  } finally { await context.cleanup() }
})

test('真实会话存储在退出重启后恢复富文本、停止/失败结果、草稿和独立上下文', async () => {
  test.setTimeout(180_000)
  const context = await fixture()
  try {
    let { page } = await context.launch()
    const firstTitle = '持久化验收：富文本、思考和用量'
    const first = await context.send(page, firstTitle)
    const firstId = (await management<ConversationList>(page, '/api/conversations')).active_id!
    delta(first, { reasoning_content: '先核对持久化字段，再保留原始回复。' })
    delta(first, { tool_calls: [{ index: 1, id: 'history-tool-late', type: 'function', function: { name: 'inspect_history', arguments: '{"persist":' } }] })
    await expect.poll(async () => (await management<Conversation>(page, `/api/conversations/${firstId}`)).messages[1].tools).toEqual([{ id: 'history-tool-late', type: 'function', function: { name: 'inspect_history', arguments: '{"persist":' } }])
    delta(first, { tool_calls: [{ index: 1, function: { arguments: 'true}' } }, { index: 0, id: 'history-tool', type: 'function', function: { name: 'inspect_order', arguments: '{"first":true}' } }] })
    await finish(page, first, '## 重启后仍可阅读\n\n这是 **保存的富文本**。\n\n```python\nprint("history")\n```')
    let panel = await history(page)
    await expect(panel.getByRole('button', { name: firstTitle, exact: true })).toBeVisible()
    const initialDocument = await management<Conversation>(page, `/api/conversations/${firstId}`)
    expect(initialDocument.messages[1]).toMatchObject({ status: 'completed', reasoning: '先核对持久化字段，再保留原始回复。', usage: { prompt_tokens: 18, completion_tokens: 32 }, timings: { predicted_per_second: 24 }, finishReason: 'stop' })
    expect(initialDocument.messages[1].firstOutput).toBeGreaterThanOrEqual(0)
    expect(initialDocument.messages[1].elapsed).toBeGreaterThanOrEqual(0)
    expect(initialDocument.messages[1].createdAt).toBeGreaterThan(0)
    expect(initialDocument.messages[1].tools).toEqual([
      { id: 'history-tool', type: 'function', function: { name: 'inspect_order', arguments: '{"first":true}' } },
      { id: 'history-tool-late', type: 'function', function: { name: 'inspect_history', arguments: '{"persist":true}' } },
    ])

    const stopped = await context.send(page, '保留停止前内容')
    delta(stopped, { reasoning_content: '停止前的思考。', content: '停止前的正文。' })
    await expect(page.locator('.chat-message.assistant').last().locator('.message-text')).toHaveText('停止前的正文。')
    await expect(page.getByRole('textbox', { name: '对话消息', exact: true })).toBeDisabled()
    await expect(chatModel(page)).toBeDisabled()
    await expect(page.getByRole('button', { name: '新建会话', exact: true })).toBeEnabled()
    await expect(panel.getByRole('button', { name: firstTitle, exact: true })).toBeEnabled()
    await page.getByRole('button', { name: '停止生成', exact: true }).click()
    stopped.response.end()
    await expect(page.locator('.chat-message.assistant').last().locator('.message-label')).toContainText('已停止')
    await saved(page)
    const failed = await context.send(page, '保留失败前内容')
    delta(failed, { content: '提前断流前的正文。' })
    failed.response.end()
    await expect(page.locator('.chat-message.assistant').last().locator('.message-error')).toContainText('响应流提前结束')
    await saved(page)

    await page.getByRole('button', { name: '对话参数', exact: true }).click()
    await page.getByRole('textbox', { name: '系统提示词', exact: true }).fill('这是第一个会话的系统提示词。')
    await page.getByLabel('温度', { exact: true }).fill('0.4')
    await page.getByLabel('最大输出 Token', { exact: true }).fill('96')
    await page.getByRole('button', { name: '收起对话参数', exact: true }).click()
    await (await editComposer(page)).fill('第一个会话的未发送草稿')
    await saved(page)

    await page.getByRole('button', { name: '新建会话', exact: true }).click()
    await expect(page.locator('.chat-message')).toHaveCount(0)
    await expect(page.getByRole('textbox', { name: '对话消息', exact: true })).toHaveValue('')
    await selectChatModel(page, imageInstance.id)
    const second = await context.send(page, '第二个会话只保留自己的上下文')
    expect(second.body.model).toBe(imageInstance.alias)
    expect(JSON.stringify(second.body.messages)).not.toContain(firstTitle)
    await finish(page, second, '第二个会话的回答。')
    const secondList = await management<ConversationList>(page, '/api/conversations')
    const secondId = secondList.active_id!
    expect(secondId).not.toBe(firstId)
    await selectChatModel(page, textInstance.id)
    const switched = await context.send(page, '切换模型后继续这个会话')
    expect(switched.body.model).toBe(textInstance.alias)
    expect(switched.body.messages).toEqual([
      { role: 'user', content: '第二个会话只保留自己的上下文' },
      { role: 'assistant', content: '第二个会话的回答。' },
      { role: 'user', content: '切换模型后继续这个会话' },
    ])
    await finish(page, switched, '切换模型没有丢失当前会话上下文。')
    panel = await history(page)
    await panel.getByRole('button', { name: firstTitle, exact: true }).click()
    await expect(page.getByRole('textbox', { name: '对话消息', exact: true, includeHidden: true })).toHaveValue('第一个会话的未发送草稿')
    // Quit before the draft debounce elapses: the native exit handshake must flush it.
    await (await editComposer(page)).fill('关闭前最后编辑的草稿')
    await context.quit()

    ;({ page } = await context.launch())
    panel = await history(page)
    await expect(panel.getByRole('button', { name: firstTitle, exact: true })).toBeVisible()
    await expect(panel.getByRole('button', { name: '第二个会话只保留自己的上下文', exact: true })).toBeVisible()
    await expect(page.getByRole('textbox', { name: '对话消息', exact: true, includeHidden: true })).toHaveValue('关闭前最后编辑的草稿')
    await expect(chatModel(page)).toHaveJSProperty('value', textInstance.id)
    await expect(page.getByRole('heading', { name: '重启后仍可阅读', exact: true })).toBeVisible()
    await expect(page.locator('.chat-message.assistant').nth(0).locator('.message-reasoning')).toContainText('先核对持久化字段')
    await expect(page.locator('.chat-message.assistant').nth(1).locator('.message-label')).toContainText('已停止')
    await expect(page.locator('.chat-message.assistant').nth(1).locator('.message-text')).toHaveText('停止前的正文。')
    await expect(page.locator('.chat-message.assistant').nth(2).locator('.message-error')).toContainText('响应流提前结束')
    await page.getByRole('button', { name: '对话参数', exact: true }).click()
    await expect(page.getByRole('textbox', { name: '系统提示词', exact: true })).toHaveValue('这是第一个会话的系统提示词。')
    await expect(page.getByLabel('温度', { exact: true })).toHaveValue('0.4')
    await expect(page.getByLabel('最大输出 Token', { exact: true })).toHaveValue('96')
    await page.getByRole('button', { name: '收起对话参数', exact: true }).click()
    const restored = await management<Conversation>(page, `/api/conversations/${firstId}`)
    expect(restored.messages.map(message => message.status)).toEqual(['completed', 'completed', 'completed', 'stopped', 'completed', 'failed'])
    expect(restored.messages[1]).toEqual(initialDocument.messages[1])
    expect(restored.selection).toMatchObject({ profile_id: textInstance.profile_id, model_id: textInstance.model_id, alias: textInstance.alias })
    expect((await management<Conversation>(page, `/api/conversations/${secondId}`)).messages).toHaveLength(4)
    expect(context.errors).toEqual([])
  } finally { await context.cleanup() }
})

test('会话原位重命名、清空和删除确认；提问导航与窄窗口不改变阅读位置', async () => {
  test.setTimeout(120_000)
  const context = await fixture()
  try {
    const { page, application } = await context.launch()
    const firstTitle = '第一条提问：检查历史导航'
    const longReply = Array.from({ length: 28 }, (_, index) => `### 保存的段落 ${index + 1}\n\n正文包含 **强调内容**，应当留在当前会话并支持连续滚动阅读。\n\n`).join('')
    await finish(page, await context.send(page, firstTitle), longReply)
    await finish(page, await context.send(page, '第二条提问：跳转到这里'), longReply)
    await finish(page, await context.send(page, '第三条提问：保持末尾'), longReply + '\n\n最后一条回答。')
    const panel = await history(page)
    const messages = page.getByRole('region', { name: '对话内容', exact: true })
    await messages.press('Home')
    await expect.poll(() => messages.evaluate(element => element.scrollTop)).toBe(0)
    // The first prompt begins below the transcript's top padding. Snap to it
    // before testing relative navigation rather than assuming Home selected it.
    const previous = page.getByRole('button', { name: '上一条提问', exact: true })
    if (await previous.isEnabled()) await previous.click()
    await expect(page.locator('.chat-message.user').first()).toBeInViewport()
    await expect(page.getByRole('button', { name: '上一条提问', exact: true })).toBeDisabled()
    await page.getByRole('button', { name: '下一条提问', exact: true }).click()
    await expect(page.locator('.chat-message.user').nth(1)).toBeInViewport()
    await page.getByRole('button', { name: '下一条提问', exact: true }).click()
    await expect(page.locator('.chat-message.user').last()).toBeInViewport()
    await page.getByRole('button', { name: '上一条提问', exact: true }).click()
    await expect(page.locator('.chat-message.user').nth(1)).toBeInViewport()
    await expect.poll(() => page.locator('.chat-message.user').nth(1).evaluate(element => Math.abs(element.getBoundingClientRect().top - element.closest('.chat-messages')!.getBoundingClientRect().top - 16))).toBeLessThan(1)
    const readingTop = await messages.evaluate(element => element.scrollTop)
    const navigation = page.getByRole('navigation', { name: '主导航' })
    await navigation.getByRole('button', { name: '模型库', exact: true }).click()
    await navigation.getByRole('button', { name: '测试对话', exact: true }).click()
    await expect.poll(async () => Math.abs(await messages.evaluate(element => element.scrollTop) - readingTop)).toBeLessThanOrEqual(1)
    await page.getByRole('button', { name: '下一条提问', exact: true }).click()
    await expect(page.locator('.chat-message.user').last()).toBeInViewport()
    await page.getByRole('button', { name: '返回最新', exact: true }).click()
    await expect.poll(() => messages.evaluate(element => element.scrollHeight - element.scrollTop - element.clientHeight)).toBeLessThanOrEqual(1)
    await page.screenshot({ path: path.join(context.directory, 'history-normal.png') })
    await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(940, 760))
    await expect.poll(() => page.evaluate(() => innerWidth)).toBeLessThanOrEqual(940)
    await page.screenshot({ path: path.join(context.directory, 'history-940.png') })
    await closeHistory(page)
    context.setFontScale(1.3)
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await expect.poll(() => page.evaluate(() => document.documentElement.style.getPropertyValue('--font-scale'))).toBe('1.3')
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1)
    await expect(page.getByRole('textbox', { name: '对话消息', exact: true })).toBeInViewport()
    await expect(page.getByRole('button', { name: '删除会话', exact: true })).toBeInViewport()
    await page.screenshot({ path: path.join(context.directory, 'history-940-font-1.3.png') })

    await page.locator('.chat-session-title').dblclick()
    await expect(page.getByRole('dialog')).toHaveCount(0)
    await page.getByRole('textbox', { name: '会话名称', exact: true }).fill('重命名后的会话')
    await page.getByRole('button', { name: '保存名称', exact: true }).click()
    await history(page)
    await expect(panel.getByRole('button', { name: '重命名后的会话', exact: true })).toBeVisible()
    await closeHistory(page)
    const firstId = (await management<ConversationList>(page, '/api/conversations')).active_id!
    await page.getByRole('button', { name: '新建会话', exact: true }).click()
    await finish(page, await context.send(page, '准备清空和删除的会话'), '这条回复只属于待删除会话。')
    const secondId = (await management<ConversationList>(page, '/api/conversations')).active_id!
    await (await editComposer(page)).fill('清空时保留的草稿')
    await page.getByRole('button', { name: '清空当前对话', exact: true }).click()
    const dialog = page.getByRole('dialog')
    await dialog.getByRole('button', { name: '取消', exact: true }).click()
    await expect(page.locator('.chat-message')).toHaveCount(2)
    await page.getByRole('button', { name: '清空当前对话', exact: true }).click()
    await page.getByRole('dialog').getByRole('button', { name: '清空对话', exact: true }).click()
    await expect(page.locator('.chat-message')).toHaveCount(0)
    await saved(page)
    expect((await management<Conversation>(page, `/api/conversations/${secondId}`)).messages).toEqual([])
    await page.getByRole('button', { name: '删除会话', exact: true }).click()
    await page.getByRole('dialog').getByRole('button', { name: '取消', exact: true }).click()
    await history(page)
    await expect(panel.getByRole('button', { name: '新会话', exact: true })).toBeVisible()
    await closeHistory(page)
    await page.getByRole('button', { name: '删除会话', exact: true }).click()
    await page.getByRole('dialog').getByRole('button', { name: '删除会话', exact: true }).click()
    await history(page)
    await expect(panel.getByRole('button', { name: '新会话', exact: true })).toHaveCount(0)
    const remaining = await management<ConversationList>(page, '/api/conversations')
    expect(remaining.items.map(item => item.id)).toContain(firstId)
    expect(remaining.items.map(item => item.id)).not.toContain(secondId)
    await panel.getByRole('button', { name: '重命名后的会话', exact: true }).click()
    await expect(page.locator('.chat-message')).toHaveCount(6)
    expect(context.errors).toEqual([])
  } finally { await context.cleanup() }
})

test('原位名称编辑支持取消、空白拦截、失败重试，切会话和清空删除不会带入旧输入', async () => {
  test.setTimeout(90_000)
  const context = await fixture()
  try {
    const { page } = await context.launch()
    await finish(page, await context.send(page, '名称编辑原会话'), '保留的回答。')
    const firstId = (await management<ConversationList>(page, '/api/conversations')).active_id!
    const input = page.getByRole('textbox', { name: '会话名称', exact: true })
    const title = page.locator('.chat-session-title')
    await title.dblclick()
    await expect(page.getByRole('dialog')).toHaveCount(0)
    await expect(input).toBeFocused()
    await input.fill('   ')
    await expect(page.getByRole('button', { name: '保存名称', exact: true })).toBeDisabled()
    await input.press('Enter')
    await expect(input).toBeVisible()
    await input.press('Escape')
    await expect(title).toHaveText('名称编辑原会话')
    await title.dblclick()
    await input.fill('取消的名称')
    await page.getByRole('button', { name: '取消重命名', exact: true }).click()
    await expect(title).toHaveText('名称编辑原会话')
    let fail = true, attempts = 0
    await page.route('**/api/conversations/*', async route => {
      const body = route.request().method() === 'PATCH' ? route.request().postDataJSON() : null
      if (body?.title) {
        attempts++
        if (fail) { await route.fulfill({ status: 503, json: { detail: 'fixture: 名称写入失败' } }); return }
      }
      await route.continue()
    })
    await title.dblclick()
    await input.fill('  重试后保存的名称  ')
    await input.press('Enter')
    await expect(page.getByText('未保存', { exact: true })).toBeVisible()
    await expect(input).toBeEnabled()
    await expect(input).toHaveValue('  重试后保存的名称  ')
    expect(attempts).toBe(1)
    expect((await management<Conversation>(page, `/api/conversations/${firstId}`)).title).toBe('名称编辑原会话')
    await page.screenshot({ path: path.join(context.directory, 'inline-rename-retry.png') })
    fail = false
    await page.getByRole('button', { name: '保存名称', exact: true }).click()
    await expect(input).toHaveCount(0)
    await expect(title).toHaveText('重试后保存的名称')
    await saved(page)
    expect((await management<Conversation>(page, `/api/conversations/${firstId}`)).title).toBe('重试后保存的名称')
    fail = true
    await title.dblclick()
    await input.fill('失败后撤销的名称')
    await input.press('Enter')
    await expect(page.getByText('未保存', { exact: true })).toBeVisible()
    await expect(input).toBeEnabled()
    await input.press('Escape')
    await expect(title).toHaveText('重试后保存的名称')
    fail = false
    await page.getByRole('button', { name: '重试保存', exact: true }).click()
    await saved(page)
    expect((await management<Conversation>(page, `/api/conversations/${firstId}`)).title).toBe('重试后保存的名称')
    await title.dblclick()
    await input.fill('不能带入新会话')
    await page.getByRole('button', { name: '新建会话', exact: true }).click()
    await expect(input).toHaveCount(0)
    await expect(title).toHaveText('新会话')
    await finish(page, await context.send(page, '第二个名称编辑会话'), '第二个回答。')
    const secondId = (await management<ConversationList>(page, '/api/conversations')).active_id!
    await title.dblclick()
    await input.fill('不能带入另一个会话')
    const panel = await history(page)
    await panel.getByRole('button', { name: '重试后保存的名称', exact: true }).click()
    await expect(input).toHaveCount(0)
    await expect(title).toHaveText('重试后保存的名称')
    await closeHistory(page)
    await title.dblclick()
    await input.fill('清空时撤销的编辑')
    await page.getByRole('button', { name: '清空当前对话', exact: true }).click()
    await page.getByRole('dialog').getByRole('button', { name: '清空对话', exact: true }).click()
    await expect(input).toHaveCount(0)
    await expect(title).toHaveText('新会话')
    await title.dblclick()
    await input.fill('删除时撤销的编辑')
    await page.getByRole('button', { name: '删除会话', exact: true }).click()
    await page.getByRole('dialog').getByRole('button', { name: '删除会话', exact: true }).click()
    await expect(input).toHaveCount(0)
    await expect(title).toHaveText('第二个名称编辑会话')
    expect((await management<Conversation>(page, `/api/conversations/${secondId}`)).title).toBe('第二个名称编辑会话')
    expect(context.errors).toEqual([])
  } finally { await context.cleanup() }
})

test('首次保存失败时不发送模型请求，未保存状态可由用户重试恢复', async () => {
  test.setTimeout(90_000)
  const context = await fixture()
  try {
    const { page } = await context.launch()
    await finish(page, await context.send(page, '保存失败验收的会话'), '之前的回复正常保存。')
    const id = (await management<ConversationList>(page, '/api/conversations')).active_id!
    const prompt = '必须先保存，再发送到模型'
    let failed = false
    await page.route('**/api/conversations/*', async route => {
      const body = route.request().method() === 'PATCH' ? route.request().postDataJSON() : null
      if (!failed && route.request().method() === 'PATCH' && body?.messages?.some((message: { text: string }) => message.text === prompt)) {
        failed = true
        await route.fulfill({ status: 503, json: { detail: 'fixture: 会话暂时无法写入' } })
      } else await route.continue()
    })
    const before = context.requests.length
    await (await editComposer(page)).fill(prompt)
    await page.getByRole('button', { name: '发送消息', exact: true }).click()
    await expect.poll(() => failed).toBe(true)
    await expect(page.getByText('未保存', { exact: true })).toBeVisible()
    expect(context.requests).toHaveLength(before)
    expect((await management<Conversation>(page, `/api/conversations/${id}`)).messages.some(message => message.text === prompt)).toBe(false)
    await page.getByRole('button', { name: '重试保存', exact: true }).click()
    await saved(page)
    expect((await management<Conversation>(page, `/api/conversations/${id}`)).messages.some(message => message.text === prompt)).toBe(true)
    // Retrying storage must not silently submit a model request.
    expect(context.requests).toHaveLength(before)
    await page.getByRole('button', { name: '新建会话', exact: true }).click()
    let failInitialPatch = true
    await page.route('**/api/conversations/*', async route => {
      const body = route.request().method() === 'PATCH' ? route.request().postDataJSON() : null
      if (failInitialPatch && body?.draft?.text === '新建已成功，激活补写重试') {
        failInitialPatch = false
        await route.fulfill({ status: 503, json: { detail: 'fixture: 首次补写失败' } })
      } else await route.fallback()
    })
    await (await editComposer(page)).fill('新建已成功，激活补写重试')
    await expect(page.getByText('未保存', { exact: true })).toBeVisible()
    expect((await management<ConversationList>(page, '/api/conversations')).active_id).toBe(id)
    await page.getByRole('button', { name: '重试保存', exact: true }).click()
    await saved(page)
    const recoveredId = (await management<ConversationList>(page, '/api/conversations')).active_id!
    expect(recoveredId).not.toBe(id)
    expect((await management<Conversation>(page, `/api/conversations/${recoveredId}`)).draft.text).toBe('新建已成功，激活补写重试')
    expect(context.errors).toEqual([])
  } finally { await context.cleanup() }
})

test('图片读取期间 Enter 不发送，含历史图片的会话不能提交到纯文本模型', async () => {
  test.setTimeout(90_000)
  const context = await fixture()
  try {
    const { page } = await context.launch()
    await selectChatModel(page, imageInstance.id)
    const input = page.getByRole('textbox', { name: '对话消息', exact: true })
    await input.fill('请检查会话中的图片')
    await page.evaluate(() => {
      const original = FileReader.prototype.readAsDataURL
      FileReader.prototype.readAsDataURL = function (file) {
        ;(window as any).releaseHistoryImageRead = () => { FileReader.prototype.readAsDataURL = original; original.call(this, file) }
        ;(window as any).historyImageReadStarted = true
      }
    })
    await page.locator('input[type="file"]').setInputFiles({ name: 'history-pixel.png', mimeType: 'image/png', buffer: Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=', 'base64') })
    await expect.poll(() => page.evaluate(() => Boolean((window as any).historyImageReadStarted))).toBe(true)
    await expect(input).toBeDisabled()
    await input.dispatchEvent('keydown', { key: 'Enter', code: 'Enter', bubbles: true, cancelable: true })
    await expect(page.getByRole('button', { name: '发送消息', exact: true })).toBeDisabled()
    expect(context.requests).toHaveLength(0)
    await page.evaluate(() => (window as any).releaseHistoryImageRead())
    await expect(page.locator('.attachment-list').getByRole('button', { name: '查看图片：history-pixel.png', exact: true })).toBeVisible()
    await page.getByRole('button', { name: '发送消息', exact: true }).click()
    await expect.poll(() => context.requests.length).toBe(1)
    const request = context.requests[0]
    expect(request.body.messages[0]).toMatchObject({ role: 'user', content: [{ type: 'text', text: '请检查会话中的图片' }, { type: 'image_url', image_url: { url: expect.stringMatching(/^data:image\/png;base64,/) } }] })
    await finish(page, request, '图片随当前会话一起保存。')
    const id = (await management<ConversationList>(page, '/api/conversations')).active_id!
    expect((await management<Conversation>(page, `/api/conversations/${id}`)).messages[0].images).toHaveLength(1)
    await selectChatModel(page, textInstance.id)
    await input.fill('纯文本模型不应收到历史图片')
    await expect(page.getByRole('button', { name: '发送消息', exact: true })).toBeDisabled()
    await input.press('Enter')
    expect(context.requests).toHaveLength(1)
    await page.getByRole('button', { name: '新建会话', exact: true }).click()
    await selectChatModel(page, textInstance.id)
    const clean = await context.send(page, '新会话可以使用纯文本模型')
    expect(clean.body.messages).toEqual([{ role: 'user', content: '新会话可以使用纯文本模型' }])
    await finish(page, clean, '纯文本新会话正常。')
    expect(context.errors).toEqual([])
  } finally { await context.cleanup() }
})

test('生命周期：意外重载把已落盘生成检查点恢复为中断，并按配置匹配新实例', async () => {
  test.setTimeout(120_000)
  const context = await fixture()
  try {
    const { page } = await context.launch()
    const request = await context.send(page, '渲染器重载后只恢复，不自动续发')
    const partial = '意外重载前已经收到的部分正文。'
    delta(request, { reasoning_content: '检查点中的部分思考。', content: partial })
    const id = (await management<ConversationList>(page, '/api/conversations')).active_id!
    await expect.poll(async () => (await management<Conversation>(page, `/api/conversations/${id}`)).messages.at(-1)).toMatchObject({ status: 'generating', text: partial, reasoning: '检查点中的部分思考。' })
    const replacement = { ...textInstance, id: 'history-text-after-restart' }
    context.setInstances([replacement, imageInstance])

    // Bypass the orderly native reload command to simulate a lost renderer.
    await page.reload()
    await pinNavigation(page)
    await expect(page.getByText('服务在线', { exact: true })).toBeVisible()
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '测试对话', exact: true }).click()
    const assistant = page.locator('.chat-message.assistant').last()
    await expect(assistant.locator('.message-label')).toContainText('已中断')
    await expect(assistant.locator('.message-text')).toHaveText(partial)
    await expect(assistant.locator('.message-reasoning')).toContainText('检查点中的部分思考。')
    await expect(page.getByRole('button', { name: '停止生成', exact: true })).toHaveCount(0)
    const model = chatModel(page)
    await expect(model).toHaveJSProperty('value', replacement.id)
    await saved(page)
    expect((await management<Conversation>(page, `/api/conversations/${id}`)).messages.at(-1)).toMatchObject({ status: 'interrupted', text: partial })
    expect(context.requests).toHaveLength(1)

    // A sole ready model now follows automatically without deleting the transcript.
    context.setInstances([imageInstance])
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await expect(model).toHaveJSProperty('value', imageInstance.id)
    await expect(chatCapsule(page)).toBeVisible()
    await expect(assistant.locator('.message-text')).toHaveText(partial)
    const continued = await context.send(page, '使用唯一的新模型继续')
    expect(continued.body.model).toBe(imageInstance.alias)
    expect(continued.body.messages).toContainEqual({ role: 'assistant', content: partial, reasoning_content: '检查点中的部分思考。' })
    await finish(page, continued, '新模型继续保留的上下文。')
    expect((await management<Conversation>(page, `/api/conversations/${id}`)).selection).toMatchObject({ profile_id: imageInstance.profile_id })
    expect(context.requests).toHaveLength(2)
    expect(context.errors).toEqual([])
  } finally { await context.cleanup() }
})

test('生命周期：生成中正常菜单退出，重启后保留部分回复并标记已停止', async () => {
  test.setTimeout(120_000)
  const context = await fixture()
  try {
    let { page } = await context.launch()
    const request = await context.send(page, '生成过程中直接退出应用')
    const partial = '正常退出前收到的部分回复。'
    delta(request, { reasoning_content: '退出前尚未完成的思考。', content: partial })
    await expect(page.locator('.chat-message.assistant').last().locator('.message-text')).toHaveText(partial)
    const id = (await management<ConversationList>(page, '/api/conversations')).active_id!
    await context.quit()
    request.response.end()

    ;({ page } = await context.launch())
    const assistant = page.locator('.chat-message.assistant').last()
    await expect(assistant.locator('.message-label')).toContainText('已停止')
    await expect(assistant.locator('.message-text')).toHaveText(partial)
    await expect(assistant.locator('.message-reasoning')).toContainText('退出前尚未完成的思考。')
    await expect(page.getByRole('textbox', { name: '对话消息', exact: true })).toBeEnabled()
    const restored = await management<Conversation>(page, `/api/conversations/${id}`)
    expect(restored.messages.at(-1)).toMatchObject({ status: 'stopped', stopped: true, text: partial, reasoning: '退出前尚未完成的思考。' })
    expect(context.requests).toHaveLength(1)
    expect(context.errors).toEqual([])
  } finally { await context.cleanup() }
})

test('生命周期：保存持续失败时默认取消退出，恢复编辑后重试并由菜单重载保存最新草稿', async () => {
  test.setTimeout(120_000)
  const context = await fixture()
  let rejectSaves = true
  try {
    const { page, application } = await context.launch()
    await finish(page, await context.send(page, '退出取消与重新加载验收'), '这条已保存回复应保持完整。')
    const id = (await management<ConversationList>(page, '/api/conversations')).active_id!
    await application.evaluate(({ dialog }) => {
      ;(globalThis as any).historyCloseWarnings = []
      dialog.showMessageBox = (async (...args: any[]) => {
        ;(globalThis as any).historyCloseWarnings.push(args.at(-1))
        return { response: 0, checkboxChecked: false }
      }) as typeof dialog.showMessageBox
    })
    let deniedPatches = 0
    await page.route('**/api/conversations/*', async route => {
      if (rejectSaves && route.request().method() === 'PATCH') {
        deniedPatches++
        await route.fulfill({ status: 503, json: { detail: 'fixture: 保存持续失败' } })
      } else await route.continue()
    })
    const input = page.getByRole('textbox', { name: '对话消息', exact: true })
    await input.fill('保存失败时应当留在应用中的草稿')
    await expect(page.getByText('未保存', { exact: true })).toBeVisible()
    await selectMenu(application, '文件', '退出 ModLoader')
    await expect.poll(() => application.evaluate(() => (globalThis as any).historyCloseWarnings.length)).toBe(1)
    const warnings = await application.evaluate(() => (globalThis as any).historyCloseWarnings)
    expect(warnings[0]).toMatchObject({ type: 'warning', title: '退出 ModLoader', message: '更改尚未保存', buttons: ['取消退出', '仍然退出'], defaultId: 0, cancelId: 0 })
    expect(warnings[0].detail).toContain('可能丢失未保存的设置或聊天记录')
    expect(deniedPatches).toBeGreaterThanOrEqual(2)
    expect(page.isClosed()).toBe(false)
    expect(await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows().some(window => !window.isDestroyed() && window.isVisible()))).toBe(true)
    await expect(input).toBeEnabled()
    const afterCancel = '取消退出后继续编辑的最新草稿'
    await input.fill(afterCancel)
    rejectSaves = false
    await page.getByRole('button', { name: '重试保存', exact: true }).click()
    await saved(page)
    expect((await management<Conversation>(page, `/api/conversations/${id}`)).draft.text).toBe(afterCancel)

    const beforeReload = '菜单重新加载前最后编辑的草稿'
    await (await editComposer(page)).fill(beforeReload)
    const reloaded = page.waitForEvent('load')
    await selectMenu(application, '视图', '重新加载界面')
    await reloaded
    await pinNavigation(page)
    await expect(page.getByText('服务在线', { exact: true })).toBeVisible()
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '测试对话', exact: true }).click()
    await expect(await editComposer(page)).toBeEnabled()
    await expect(input).toHaveValue(beforeReload)
    expect((await management<Conversation>(page, `/api/conversations/${id}`)).draft.text).toBe(beforeReload)
    expect(await application.evaluate(() => (globalThis as any).historyCloseWarnings.length)).toBe(1)
    expect(context.requests).toHaveLength(1)
    expect(context.errors).toEqual([])
  } finally {
    rejectSaves = false
    await context.cleanup()
  }
})

test('模型库入口在生成中拒绝切换，结束后同一配置入口可选新模型继续原上下文', async () => {
  test.setTimeout(90_000)
  const context = await fixture(true)
  try {
    const { page } = await context.launch()
    const promptA = '模型A的上下文不应被忙时入口切走'
    const requestA = await context.send(page, promptA)
    expect(requestA.body.model).toBe(textInstance.alias)
    const partialA = '模型A仍在生成的部分回复。'
    delta(requestA, { content: partialA })
    await expect(page.locator('.chat-message.assistant').last().locator('.message-text')).toHaveText(partialA)
    const id = (await management<ConversationList>(page, '/api/conversations')).active_id!
    const navigation = page.getByRole('navigation', { name: '主导航' })
    const model = chatModel(page)
    const imageProfile = page.locator('.profile-card').filter({ has: page.getByRole('heading', { name: `${imageInstance.alias}配置`, exact: true }) })
    await navigation.getByRole('button', { name: '模型库', exact: true }).click()
    await page.getByRole('region', { name: '模型列表', exact: true }).getByRole('button', { name: new RegExp(`^${imageInstance.alias}`) }).click()
    await imageProfile.getByRole('button', { name: '测试对话', exact: true }).click()
    await expect(page.getByText('请先停止生成或等待当前操作完成，再切换模型。', { exact: true })).toBeVisible()
    await expect(chatCapsule(page)).toBeVisible()
    await expect(model).toHaveJSProperty('value', textInstance.id)
    await expect(page.getByRole('button', { name: '停止生成', exact: true })).toBeVisible()
    const busyDocument = await management<Conversation>(page, `/api/conversations/${id}`)
    expect(busyDocument.selection).toMatchObject({ profile_id: textInstance.profile_id, model_id: textInstance.model_id, alias: textInstance.alias })
    expect(busyDocument.messages.at(-1)).toMatchObject({ model: textInstance.alias, profileId: textInstance.profile_id, status: 'generating' })
    expect(context.requests).toHaveLength(1)
    const finalA = '模型A已完成当前回答。'
    await finish(page, requestA, finalA)

    await navigation.getByRole('button', { name: '模型库', exact: true }).click()
    await imageProfile.getByRole('button', { name: '测试对话', exact: true }).click()
    await expect(model).toHaveJSProperty('value', imageInstance.id)
    const promptB = '从模型B接着当前会话回答'
    const requestB = await context.send(page, promptB)
    expect(requestB.body.model).toBe(imageInstance.alias)
    expect(requestB.body.messages).toEqual([
      { role: 'user', content: promptA },
      { role: 'assistant', content: partialA + finalA },
      { role: 'user', content: promptB },
    ])
    expect((await management<ConversationList>(page, '/api/conversations')).active_id).toBe(id)
    await finish(page, requestB, '模型B使用同一个会话的上下文继续回答。')
    expect((await management<Conversation>(page, `/api/conversations/${id}`)).selection).toMatchObject({ profile_id: imageInstance.profile_id, model_id: imageInstance.model_id, alias: imageInstance.alias })
    expect(context.errors).toEqual([])
  } finally { await context.cleanup() }
})


test('导出当前会话 Markdown 保留原文，取消与写入失败不改会话，生成中导出快照', async () => {
  test.setTimeout(120_000)
  const context = await fixture()
  try {
    const { page, application } = await context.launch()
    const exportButton = page.getByRole('button', { name: '导出对话为 Markdown', exact: true })
    await expect(exportButton).toBeDisabled()
    const prompt = '检验 R_B：保留 **Markdown**、代码与公式'
    const answer = '原始公式：\n\n$$\\frac{1}{2} + \\frac{1}{3} = \\frac{5}{6}$$\n\n| 项 | 值 |\n| --- | --- |\n| $R_B$ | $\\frac{5}{6}$ |\n\n```python\nprint("中文")\n```'
    const request = await context.send(page, prompt)
    delta(request, { reasoning_content: '先通分，再验证 $R_B$。' })
    await finish(page, request, answer)
    const id = (await management<ConversationList>(page, '/api/conversations')).active_id!
    const original = await management<Conversation>(page, `/api/conversations/${id}`)
    const imageUrl = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a8e8AAAAASUVORK5CYII='
    original.messages[0].images = [{ id: 'export-image', name: '图[1].png', size: 68, url: imageUrl }]
    original.messages[1].tools = [{ function: { name: 'calculator', arguments: '{"text":"``` 保持原样"}' } }]
    await management(page, `/api/conversations/${id}`, 'PATCH', { title: '检验 R_B:样例/导出?', messages: original.messages })
    await page.reload()
    await pinNavigation(page)
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '测试对话', exact: true }).click()
    await expect(exportButton).toBeEnabled()
    const output = path.join(context.directory, '会话导出.md')
    await application.evaluate(({ dialog }, output) => {
      const state = { output, canceled: false, hold: false, count: 0, options: null as any, release: null as any }
      ;(globalThis as any).markdownExportTest = state
      dialog.showSaveDialog = async (_owner: any, options: any) => {
        state.options = options; state.count++
        if (state.hold) await new Promise(resolve => { state.release = resolve })
        return { canceled: state.canceled, filePath: state.canceled ? undefined : state.output }
      }
    }, output)
    const exportBox = await exportButton.boundingBox()
    const clearBox = await page.getByRole('button', { name: '清空当前对话', exact: true }).boundingBox()
    expect(exportBox!.x + exportBox!.width).toBeLessThan(clearBox!.x)
    await exportButton.click()
    await expect(page.getByText('已导出 Markdown', { exact: true })).toBeVisible()
    const markdown = fs.readFileSync(output, 'utf8')
    expect(markdown).toContain('# 检验 R\\_B:样例/导出?')
    expect(markdown).toContain('## 用户')
    expect(markdown).toContain('## 助手 · 历史文本模型')
    expect(markdown).toContain(prompt)
    expect(markdown).toContain(answer)
    expect(markdown).toContain('<summary>思考过程</summary>\n\n先通分，再验证 $R_B$。')
    expect(markdown).toContain(`![图\\[1\\].png](<${imageUrl}>)`)
    expect(markdown).toContain('````json\n{"text":"``` 保持原样"}\n````')
    expect(await application.evaluate(() => (globalThis as any).markdownExportTest.options.defaultPath)).toBe('检验 R_B_样例_导出_.md')
    const before = await management<Conversation>(page, `/api/conversations/${id}`)
    await application.evaluate(() => { (globalThis as any).markdownExportTest.canceled = true })
    await exportButton.click()
    await expect(exportButton).toBeEnabled()
    expect(fs.readFileSync(output, 'utf8')).toBe(markdown)
    expect(await management<Conversation>(page, `/api/conversations/${id}`)).toEqual(before)
    await application.evaluate(() => { const state = (globalThis as any).markdownExportTest; state.canceled = false; state.output += '/invalid.md' })
    await exportButton.click()
    await expect(page.getByText(/导出失败：/)).toBeVisible()
    await expect(exportButton).toBeEnabled()
    expect(await management<Conversation>(page, `/api/conversations/${id}`)).toEqual(before)
    await page.getByRole('button', { name: '新建会话', exact: true }).click()
    const live = await context.send(page, '导出生成中的内容')
    delta(live, { content: '导出瞬间的回复' })
    await expect(page.locator('.chat-message.assistant .message-text')).toContainText('导出瞬间的回复')
    const snapshot = path.join(context.directory, '生成中快照.md')
    await application.evaluate((_electron, output) => { const state = (globalThis as any).markdownExportTest; state.output = output; state.hold = true }, snapshot)
    await exportButton.click()
    await expect.poll(() => application.evaluate(() => Boolean((globalThis as any).markdownExportTest.release))).toBe(true)
    await expect(exportButton).toBeDisabled()
    delta(live, { content: '，选择保存位置时继续输出' })
    await expect(page.locator('.chat-message.assistant .message-text')).toContainText('选择保存位置时继续输出')
    await application.evaluate(() => { (globalThis as any).markdownExportTest.release() })
    await expect(exportButton).toBeEnabled()
    const liveMarkdown = fs.readFileSync(snapshot, 'utf8')
    expect(liveMarkdown).toContain('导出瞬间的回复')
    expect(liveMarkdown).toContain('生成中（导出时的内容）')
    expect(liveMarkdown).not.toContain('选择保存位置时继续输出')
    await finish(page, live, '，正常结束')
    await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(940, 760))
    await page.getByRole('button', { name: '对话历史', exact: true }).click()
    await page.getByRole('button', { name: '对话参数', exact: true }).click()
    await expect(page.getByRole('button', { name: '对话参数', exact: true })).toHaveAttribute('aria-expanded', 'true')
    await expect.poll(() => page.locator('.chat-session-bar').evaluate(node => node.scrollWidth <= node.clientWidth)).toBe(true)
    await expect(exportButton).toBeVisible()
    const actions = await page.locator('.chat-transcript-actions').boundingBox()
    const panel = await page.locator('.chat-panel').boundingBox()
    expect(actions!.x + actions!.width).toBeLessThanOrEqual(panel!.x + panel!.width)
    await page.screenshot({ path: path.join(context.directory, 'chat-export-narrow-both.png') })
    await closeHistory(page)
    await page.getByRole('button', { name: '收起对话参数', exact: true }).click()
    await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(1440, 960))
    await page.screenshot({ path: path.join(context.directory, 'chat-export-toolbar.png') })
    expect(context.errors).toEqual([])
  } finally { await context.cleanup() }
})

test('消息时间持久保存，等待输出时持续计时，紧凑统计与悬停说明保持口径', async () => {
  test.setTimeout(120_000)
  const context = await fixture()
  try {
    const { page, application } = await context.launch()
    let sample: Record<string, unknown> | null = null
    await page.route('**/api/instances/*/telemetry', route => route.fulfill({ json: { groups: [], unavailable: [], live_generation: sample } }))
    const beforeSend = Date.now()
    const first = await context.send(page, '查看时间和紧凑统计')
    const user = page.locator('.chat-message.user').first()
    const assistant = page.locator('.chat-message.assistant').first()
    const elapsed = assistant.locator('[data-metric="elapsed"]')
    await expect(elapsed).toHaveAccessibleName(/^当前用时/)
    await expect.poll(async () => parseFloat(await elapsed.innerText())).toBeGreaterThan(1)
    const userTime = await user.locator('time').getAttribute('datetime')
    expect(Date.parse(userTime!)).toBeGreaterThanOrEqual(beforeSend)
    const id = (await management<ConversationList>(page, '/api/conversations')).active_id!
    const pending = await management<Conversation>(page, `/api/conversations/${id}`)
    expect(pending.messages[1].elapsed).toBeUndefined()
    expect(pending.messages[1].responseAt).toBeUndefined()
    const ring = page.locator('.chat-context-usage')
    await expect(ring).toContainText('—')
    await expect(assistant.locator('[data-metric="output"]')).toHaveText('—')
    sample = { client_request_id: first.clientRequestId, context_tokens: 140, context_capacity: 8192, context_basis: '输入 + 已生成', generated_tokens: 22, available_for_chat: true, tokens_per_second: 24, window_s: 2 }
    await expect(assistant.locator('[data-metric="output"]')).toHaveText('22')
    await expect(ring).toHaveAttribute('data-tooltip', /已用 140 Token.*总容量 8,192 Token/s)
    await expect(assistant.locator('[data-metric="live-speed"]')).toHaveText('24.00tok/s')
    await expect(page.locator('.inline-tags [data-metric="live-speed"]')).toHaveText('24.00tok/s')
    await ring.hover()
    await expect(page.getByRole('tooltip')).toContainText('已用 140 Token')
    await expect.poll(() => page.getByRole('tooltip').evaluate(node => getComputedStyle(node).opacity)).toBe('1')
    sample = { ...sample, client_request_id: 'another-request', context_tokens: 777, generated_tokens: 999 }
    await expect(ring).toContainText('—')
    await expect(assistant.locator('[data-metric="output"]')).toHaveText('—')
    sample = { ...sample, client_request_id: first.clientRequestId, context_tokens: 140, generated_tokens: 22 }
    await expect(ring).toHaveAttribute('data-tooltip', /已用 140 Token/)
    const outputBefore = Date.now()
    delta(first, { reasoning_content: '先核对输入。' })
    await expect(assistant.locator('time')).toHaveAttribute('data-tooltip', /^回复时间 · /)
    const responseTime = await assistant.locator('time').getAttribute('datetime')
    expect(Date.parse(responseTime!)).toBeGreaterThanOrEqual(outputBefore)
    expect(Date.parse(responseTime!) - Date.parse(userTime!)).toBeGreaterThan(900)
    delta(first, { content: '按同一个计时起点继续。' })
    await expect(assistant.locator('.message-text')).toHaveText('按同一个计时起点继续。')
    await ring.hover()
    await expect(page.getByRole('tooltip')).toContainText('已用 140 Token')
    await expect.poll(() => page.getByRole('tooltip').evaluate(node => getComputedStyle(node).opacity)).toBe('1')
    await page.screenshot({ path: path.join(context.directory, 'chat-live-context-output.png') })
    await expect(assistant.locator('time')).toHaveAttribute('datetime', responseTime!)
    const runningSeconds = parseFloat(await elapsed.innerText())
    await finish(page, first, '这是最终回复。')
    await expect(elapsed).toHaveAccessibleName(/^总耗时/)
    await expect(ring).toHaveAttribute('data-tooltip', /已用 50 Token.*总容量 8,192 Token.*上次请求/s)
    const finalSeconds = (await elapsed.textContent())!
    expect(parseFloat(finalSeconds)).toBeGreaterThanOrEqual(runningSeconds)
    const stored = await management<Conversation>(page, `/api/conversations/${id}`)
    expect(stored.messages[1].responseAt).toBe(Date.parse(responseTime!))
    expect(stored.messages[0].createdAt).toBe(Date.parse(userTime!))
    expect(stored.messages[1].elapsed!.toFixed(1)).toBe(parseFloat(finalSeconds).toFixed(1))
    const inputMetric = assistant.locator('[data-metric="input"]')
    await expect(inputMetric).toHaveText('—')
    await expect(assistant.locator('[data-metric="cached"]')).toHaveText('—')
    await expect(inputMetric.locator('.lucide-arrow-up')).toHaveCount(1)
    await expect(assistant.locator('[data-metric="output"] .lucide-arrow-down')).toHaveCount(1)
    await expect(assistant.locator('[data-metric="speed"]')).toHaveText('DEC24.00tok/s')
    await expect(assistant.locator('[data-metric="first-output"]')).toContainText('TTFT')
    await inputMetric.hover()
    await expect(page.getByRole('tooltip')).toHaveText(/^(本轮)?未缓存输入 未知 Token$/)
    await expect(inputMetric).not.toHaveAttribute('title')
    await expect.poll(() => page.getByRole('tooltip').evaluate(node => getComputedStyle(node).opacity)).toBe('1')
    await page.screenshot({ path: path.join(context.directory, 'chat-timestamps-compact-tooltip.png') })
    await inputMetric.focus()
    await page.keyboard.press('Tab')
    await page.keyboard.press('Tab')
    await page.keyboard.press('Tab')
    await page.keyboard.press('Tab')
    await expect(assistant.locator('[data-metric="first-output"]')).toBeFocused()
    await expect(page.getByRole('tooltip')).toContainText('从发送请求到首段正文')
    await page.keyboard.press('Escape')
    await expect(page.getByRole('tooltip')).toBeHidden()
    const stopped = await context.send(page, '等待时停止也保留用时')
    const last = page.locator('.chat-message.assistant').last()
    await expect.poll(async () => parseFloat(await last.locator('[data-metric="elapsed"]').innerText())).toBeGreaterThan(.5)
    await page.getByRole('button', { name: '停止生成', exact: true }).click()
    await expect(page.getByRole('button', { name: '停止生成', exact: true })).toHaveCount(0)
    stopped.response.end()
    await saved(page)
    const stoppedElapsed = (await last.locator('[data-metric="elapsed"]').textContent())!
    await page.waitForTimeout(1100)
    await expect(last.locator('[data-metric="elapsed"]')).toHaveText(stoppedElapsed)
    await expect(elapsed).toHaveText(finalSeconds)
    await page.reload()
    await pinNavigation(page)
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '测试对话', exact: true }).click()
    await expect(user.locator('time')).toHaveAttribute('datetime', userTime!)
    await expect(assistant.locator('time')).toHaveAttribute('datetime', responseTime!)
    await expect(elapsed).toHaveText(finalSeconds)
    await user.getByRole('button', { name: '编辑消息', exact: true }).click()
    await page.getByRole('textbox', { name: '编辑消息内容' }).fill('替换后使用新的发送时间')
    await page.getByRole('button', { name: '重新发送', exact: true }).click()
    await expect.poll(() => context.requests.length).toBe(3)
    const editedTime = await user.locator('time').getAttribute('datetime')
    expect(Date.parse(editedTime!)).toBeGreaterThan(Date.parse(userTime!))
    await finish(page, context.requests[2], '编辑后的回复')
    await assistant.getByRole('button', { name: '重试回复', exact: true }).click()
    await expect.poll(() => context.requests.length).toBe(4)
    await expect(user.locator('time')).toHaveAttribute('datetime', editedTime!)
    await finish(page, context.requests[3], '重试后的回复')
    await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(940, 760))
    await page.getByRole('button', { name: '对话参数', exact: true }).click()
    await expect(page.getByRole('complementary', { name: '对话参数', exact: true })).toBeVisible()
    await history(page)
    await expect.poll(() => assistant.locator('.message-footer').evaluate(node => node.scrollWidth <= node.clientWidth)).toBe(true)
    await expect.poll(() => page.locator('.composer-actions').evaluate(node => node.scrollWidth <= node.clientWidth)).toBe(true)
    const composerBounds = await page.locator('.chat-composer').boundingBox()
    for (const target of [ring, page.getByRole('button', { name: '发送消息', exact: true })]) {
      const bounds = (await target.boundingBox())!
      expect(bounds.x).toBeGreaterThanOrEqual(composerBounds!.x)
      expect(bounds.x + bounds.width).toBeLessThanOrEqual(composerBounds!.x + composerBounds!.width)
    }
    await page.screenshot({ path: path.join(context.directory, 'chat-timestamps-narrow.png') })
    expect(context.errors).toEqual([])
  } finally { await context.cleanup() }
})

test('浮动顶部栏展开只遮挡自身宽度，左右聊天面板仍可见和点击', async () => {
  test.setTimeout(100_000)
  const context = await fixture()
  context.setFontScale(1.3)
  try {
    const { page, application } = await context.launch()
    await application.evaluate(({ BrowserWindow }) => { const window = BrowserWindow.getAllWindows()[0]; window.setSize(2200, 1100); window.webContents.setZoomFactor(1) })
    await expect.poll(() => page.evaluate(() => innerWidth)).toBeGreaterThan(2100)
    await finish(page, await context.send(page, '左侧应保持可见的历史会话'), '用于检查展开顶部栏时的面板边界。')
    const firstId = (await management<ConversationList>(page, '/api/conversations')).active_id!
    await page.getByRole('button', { name: '新建会话', exact: true }).click()
    await finish(page, await context.send(page, '当前会话'), '左右面板不应被栏体外的背景遮挡。')
    // Give the first fixture an actual later send, without changing the active one.
    const earlier = await management<Conversation>(page, `/api/conversations/${firstId}`)
    await management(page, `/api/conversations/${firstId}`, 'PATCH', { messages: [{ ...earlier.messages[0], createdAt: Date.now() }] })
    await page.reload()
    await pinNavigation(page)
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '测试对话', exact: true }).click()
    await history(page)
    await page.getByRole('button', { name: '对话参数', exact: true }).click()
    await page.getByRole('textbox', { name: '系统提示词', exact: true }).fill('右侧参数保持可用。')
    await page.getByRole('button', { name: '取消钉住顶部栏', exact: true }).click()
    await page.mouse.move(1100, 600)
    const header = page.locator('.app-header'), panel = page.locator('#header-panel')
    await expect(header).not.toHaveClass(/is-open/)
    async function expandWithKeyboardFocus() {
      await panel.hover()
      await expect(header).toHaveClass(/is-open/)
      await header.locator('.navigation button').first().focus()
      await page.keyboard.press('Tab')
      await expect.poll(() => header.locator('.navigation button').nth(1).evaluate(node => node.matches(':focus-visible'))).toBe(true)
      await expect.poll(() => panel.locator('.nav-label').first().evaluate(node => getComputedStyle(node).opacity)).toBe('1')
    }
    await expandWithKeyboardFocus()
    const leftItem = page.locator('.chat-history-item').filter({ hasText: '左侧应保持可见的历史会话' })
    const rightClose = page.getByRole('button', { name: '收起对话参数', exact: true })
    const geometry = await page.evaluate(() => {
      const header = document.querySelector('.app-header')!.getBoundingClientRect()
      const panel = document.querySelector('#header-panel')!.getBoundingClientRect()
      const left = [...document.querySelectorAll('.chat-history-item')].find(node => node.textContent!.includes('左侧应保持可见的历史会话'))!.getBoundingClientRect()
      const right = document.querySelector('.chat-options .panel-heading button')!.getBoundingClientRect()
      const y = Math.min(left.top + 10, panel.bottom - 10)
      const points = [{x:left.left+24,y}, {x:right.left+right.width/2,y:right.top+right.height/2}]
      return {header:{left:header.left,right:header.right,bottom:header.bottom},panel:{left:panel.left,right:panel.right,bottom:panel.bottom},leftTop:left.top,
        points:points.map(point=>({...point,hit:document.elementFromPoint(point.x,point.y)?.closest('.chat-history, .chat-options')?.className || document.elementFromPoint(point.x,point.y)?.className}))}
    })
    const shot = await application.evaluate(async ({ BrowserWindow }) => {
      const window = BrowserWindow.getAllWindows()[0]
      return (await window.webContents.capturePage({x:0,y:0,width:window.getContentBounds().width,height:360})).toPNG().toString('base64')
    })
    fs.writeFileSync(path.join(context.directory, 'floating-header-side-panels.png'),Buffer.from(shot,'base64'))
    fs.writeFileSync(path.join(context.directory, 'floating-header-geometry.json'),JSON.stringify(geometry,null,2))
    expect(geometry.points[0].x).toBeLessThan(geometry.panel.left)
    expect(geometry.points[1].x).toBeGreaterThan(geometry.panel.right)
    expect(geometry.leftTop).toBeLessThan(geometry.panel.bottom)
    expect(geometry.points[0].hit).toContain('chat-history')
    expect(geometry.points[1].hit).toContain('chat-options')
    // The top gap remains opaque within the actual header footprint.
    expect(await header.evaluate(node => getComputedStyle(node).backgroundColor)).not.toBe('rgba(0, 0, 0, 0)')
    expect(Math.abs(geometry.header.left-geometry.panel.left)).toBeLessThanOrEqual(1)
    expect(Math.abs(geometry.header.right-geometry.panel.right)).toBeLessThanOrEqual(1)
    await leftItem.click({position:{x:24,y:10}})
    await expect.poll(async () => (await management<ConversationList>(page, '/api/conversations')).active_id).toBe(firstId)
    await expandWithKeyboardFocus()
    await rightClose.click()
    await expect(page.getByRole('complementary', { name:'对话参数',exact:true })).toBeHidden()
    expect(context.errors).toEqual([])
  } finally { await context.cleanup() }
})


test('空会话不入历史，文字和图片草稿保留，清空草稿后移除空记录', async () => {
  test.setTimeout(90_000)
  const context = await fixture()
  try {
    const { page } = await context.launch()
    await selectChatModel(page, imageInstance.id)
    await page.getByRole('button', { name: '对话参数', exact: true }).click()
    await page.getByRole('textbox', { name: '系统提示词', exact: true }).fill('只有参数不创建历史')
    await page.getByRole('button', { name: '收起对话参数', exact: true }).click()
    await (await editComposer(page)).fill('   ')
    await saved(page)
    expect((await management<ConversationList>(page, '/api/conversations')).items).toEqual([])
    await page.getByRole('button', { name: '新建会话', exact: true }).click()
    await page.getByRole('button', { name: '新建会话', exact: true }).click()
    expect((await management<ConversationList>(page, '/api/conversations')).items).toEqual([])
    await (await editComposer(page)).fill('未发送的真实草稿')
    await saved(page)
    const draft = (await management<ConversationList>(page, '/api/conversations')).active_id!
    expect((await management<Conversation>(page, `/api/conversations/${draft}`)).draft.text).toBe('未发送的真实草稿')
    const panel = await history(page)
    await expect(panel.locator('.chat-history-item')).toContainText('草稿')
    await page.getByRole('button', { name: '新建会话', exact: true }).click()
    await panel.getByRole('button', { name: '新会话', exact: true }).click()
    await expect(page.getByRole('textbox', { name: '对话消息', exact: true, includeHidden: true })).toHaveValue('未发送的真实草稿')
    await (await editComposer(page)).fill('')
    await saved(page)
    expect((await management<ConversationList>(page, '/api/conversations')).items).toEqual([])
    await page.locator('input[type=file]').setInputFiles({ name: 'draft.png', mimeType: 'image/png', buffer: Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a4R8AAAAASUVORK5CYII=', 'base64') })
    await saved(page)
    await expect(page.locator('.attachment img')).toHaveCount(1)
    await saved(page)
    const imageDraft = (await management<ConversationList>(page, '/api/conversations')).active_id!
    expect((await management<Conversation>(page, `/api/conversations/${imageDraft}`)).draft.images).toHaveLength(1)
    await page.reload(); await pinNavigation(page)
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '测试对话', exact: true }).click()
    await expect(page.locator('.attachment img')).toHaveCount(1)
    await page.getByRole('button', { name: '移除 draft.png', exact: true }).click()
    await saved(page)
    expect((await management<ConversationList>(page, '/api/conversations')).items).toEqual([])
    expect(context.errors).toEqual([])
  } finally { await context.cleanup() }
})

test('后台生成按会话隔离，重命名和编辑不中断其他请求，清空删除安全停止', async () => {
  test.setTimeout(120_000)
  const context = await fixture()
  try {
    const { page } = await context.launch()
    const first = await context.send(page, '后台第一条')
    delta(first, { content: '第一段' })
    await expect(page.locator('.chat-message.assistant')).toContainText('第一段')
    const firstId = (await management<ConversationList>(page, '/api/conversations')).active_id!
    await expect.poll(async () => (await management<Conversation>(page, `/api/conversations/${firstId}`)).messages[1].text).toBe('第一段')
    const firstTime = (await management<any>(page, `/api/conversations/${firstId}`)).last_message_at
    await page.locator('.chat-session-title').dblclick()
    await page.getByRole('textbox', { name: '会话名称', exact: true }).fill('生成中重命名')
    await page.getByRole('button', { name: '保存名称', exact: true }).click()
    await expect(page.locator('.chat-session-title')).toHaveText('生成中重命名')
    expect((await management<any>(page, `/api/conversations/${firstId}`)).last_message_at).toBe(firstTime)
    await page.getByRole('button', { name: '新建会话', exact: true }).click()
    await selectChatModel(page, imageInstance.id)
    const second = await context.send(page, '后台第二条')
    const secondId = (await management<ConversationList>(page, '/api/conversations')).active_id!
    expect(second.body.messages).toEqual([{ role: 'user', content: '后台第二条' }])
    delta(first, { content: '后台追加' })
    delta(second, { content: '第二段' })
    await expect(page.locator('.chat-message.assistant')).toContainText('第二段')
    await expect(page.locator('.chat-messages')).not.toContainText('后台追加')
    await expect.poll(async () => (await management<Conversation>(page, `/api/conversations/${firstId}`)).messages[1].text).toBe('第一段后台追加')
    expect((await management<ConversationList>(page, '/api/conversations')).active_id).toBe(secondId)
    const panel = await history(page)
    await expect(panel.getByRole('button', { name: '生成中重命名', exact: true })).toContainText(/\d{4}\/\d{2}\/\d{2}.*\d{2}:\d{2}/)
    expect(await panel.locator('.chat-history-item').allTextContents()).toEqual([expect.stringContaining('后台第二条'), expect.stringContaining('生成中重命名')])
    await panel.getByRole('button', { name: '生成中重命名', exact: true }).click()
    expect(await panel.locator('.chat-history-item').allTextContents()).toEqual([expect.stringContaining('后台第二条'), expect.stringContaining('生成中重命名')])
    await expect(page.getByRole('button', { name: '停止生成', exact: true })).toBeVisible()
    await page.locator('.chat-message.user').getByRole('button', { name: '编辑消息', exact: true }).click()
    await page.getByRole('textbox', { name: '编辑消息内容' }).fill('第一条重写')
    await page.getByRole('button', { name: '重新发送', exact: true }).click()
    await expect.poll(() => context.requests.length).toBe(3)
    delta(first, { content: '旧流污染' }); first.response.end('data: [DONE]\n\n')
    expect(context.requests[2].body.model).toBe(textInstance.alias)
    expect(context.requests[2].body.messages).toEqual([{ role: 'user', content: '第一条重写' }])
    await expect(panel.locator('.chat-history-item').first()).toContainText('生成中重命名')
    await panel.getByRole('button', { name: '后台第二条', exact: true }).click()
    await expect(page.locator('.chat-message.assistant')).toContainText('第二段')
    await expect(page.getByRole('button', { name: '停止生成', exact: true })).toBeVisible()
    event(context.requests[2], { choices: [{ delta: { content: '重写后的干净回复' }, finish_reason: 'stop' }] })
    context.requests[2].response.end('data: [DONE]\n\n')
    await expect.poll(async () => (await management<Conversation>(page, `/api/conversations/${firstId}`)).messages[1].status).toBe('completed')
    expect((await management<ConversationList>(page, '/api/conversations')).active_id).toBe(secondId)
    await expect(page.locator('.chat-messages')).not.toContainText('重写后的干净回复')
    await page.getByRole('button', { name: '删除会话', exact: true }).click()
    await page.getByRole('dialog').getByRole('button', { name: '删除会话', exact: true }).click()
    delta(second, { content: '删除后污染' }); second.response.end('data: [DONE]\n\n')
    await expect(page.locator('.chat-session-title')).toHaveText('生成中重命名')
    expect((await management<ConversationList>(page, '/api/conversations')).items.map(item => item.id)).not.toContain(secondId)
    const fourth = await context.send(page, '清空运行中的请求')
    delta(fourth, { content: '清空前' })
    await page.getByRole('button', { name: '清空当前对话', exact: true }).click()
    await page.getByRole('dialog').getByRole('button', { name: '清空对话', exact: true }).click()
    delta(fourth, { content: '清空后污染' }); fourth.response.end('data: [DONE]\n\n')
    await expect(page.locator('.chat-message')).toHaveCount(0)
    await saved(page)
    expect((await management<ConversationList>(page, '/api/conversations')).items).toEqual([])
    await expect(page.locator('.chat-context-usage')).toContainText('—')
    expect(context.errors).toEqual([])
  } finally { await context.cleanup() }
})


test('历史长列表长按拖动在边缘持续滚动并保存跨屏顺序', async () => {
  test.setTimeout(90_000)
  const context = await fixture()
  try {
    const { page } = await context.launch()
    const reorders = trackHistoryReorders(page)
    for (let index = 1; index <= 30; index++) {
      const id = await page.evaluate(() => crypto.randomUUID())
      await management(page, '/api/conversations', 'POST', { id, title: `长列表 ${String(index).padStart(2, '0')}`, selection: null, draft: { text: '', images: [] }, settings: { system: '', stream: true, toolsDraft: '' } })
      await management(page, `/api/conversations/${id}`, 'PATCH', { messages: [{ id: 'question', role: 'user', text: '已发送', images: [], createdAt: index, status: 'completed' }] })
    }
    await page.reload(); await pinNavigation(page)
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '测试对话', exact: true }).click()
    const panel = await history(page), list = panel.locator('.chat-history-list')
    const row = panel.getByRole('button', { name: '长列表 30', exact: true })
    const source = (await row.boundingBox())!, bounds = (await list.boundingBox())!
    await row.hover({ position: { x: 60, y: 20 } }); await page.mouse.down()
    await expect(row.locator('..')).toHaveClass(/is-dragging/)
    await page.mouse.move(source.x + 60, bounds.y + bounds.height - 4, { steps: 8 })
    await expect.poll(() => list.evaluate(node => node.scrollHeight - node.clientHeight - node.scrollTop), { timeout: 15_000 }).toBeLessThanOrEqual(1)
    await expect(page.locator('.chat-history-drag-preview')).toContainText('长列表 30')
    await expect(list.locator('.chat-history-row').last()).toHaveClass(/is-dragging/)
    expect(reorders).toHaveLength(0)
    expect((await management<ConversationList>(page, '/api/conversations')).items[0].title).toBe('长列表 30')
    const stableOrders = await list.evaluate(async node => {
      const result: string[] = []
      for (let frame = 0; frame < 16; frame++) {
        await new Promise(requestAnimationFrame)
        result.push([...node.querySelectorAll<HTMLElement>('.chat-history-row')].map(row => row.dataset.historyId).join(','))
      }
      return result
    })
    expect(new Set(stableOrders).size).toBe(1)
    await page.screenshot({ path: path.join(context.directory, 'history-drag-scroll-preview.png') })
    await page.mouse.up()
    await expect(page.locator('.chat-history-drag-preview')).toHaveCount(0)
    await expect.poll(() => reorders.length).toBe(1)
    await expect.poll(async () => (await management<ConversationList>(page, '/api/conversations')).items.at(-1)?.title).toBe('长列表 30')
    await expect(panel.getByRole('heading', { name: '对话历史', exact: true })).toBeVisible()
    await page.screenshot({ path: path.join(context.directory, 'history-drag-scroll.png') })
    expect(context.errors).toEqual([])
  } finally { await context.cleanup() }
})
