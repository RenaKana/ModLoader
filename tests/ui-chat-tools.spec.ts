import { test, expect, _electron as electron, type Page } from '@playwright/test'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { pathToFileURL } from 'node:url'
import { pinNavigation } from './navigation'
import { disableWebSearch, openChatControls, chatModel } from './chat-controls'
import { emptyConversation, type Message } from '../src/chat-history'

const root = path.resolve(__dirname, '..'), artifacts = process.env.MODLOADER_TEST_ARTIFACTS || path.join(root, '.artifacts/codex/chat-tools-20261027')
async function management(page: Page, route: string, method = 'GET', body?: unknown) {
  return page.evaluate(async ({ route, method, body }) => {
    const { baseUrl, token } = await window.modloader!.connection()
    const response = await fetch(baseUrl + route, { method, headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) })
    if (!response.ok) throw new Error(await response.text())
    return response.status === 204 ? null : response.json()
  }, { route, method, body })
}

test('对话工具完整流程：快捷加载、计算、文件确认、自动与手动压缩及恢复', async () => {
  test.setTimeout(180_000)
  fs.mkdirSync(artifacts, { recursive: true })
  const directory = fs.mkdtempSync(path.join(artifacts, 'chat-ui-')), workspace = path.join(directory, 'workspace')
  fs.mkdirSync(workspace); fs.writeFileSync(path.join(workspace, 'note.txt'), '原始内容\n', 'utf8')
  const app = await electron.launch({ args: [root], env: { ...process.env, MODLOADER_DATA_DIR: path.join(directory, 'data'), MODLOADER_SENSOR_AUTO_START: '0', MODLOADER_DEV_URL: pathToFileURL(process.env.MODLOADER_TEST_RENDERER || path.join(root, 'dist/index.html')).href }, timeout: 60_000 })
  const errors: string[] = [], requests: any[] = []
  let scenario: 'calculate' | 'write' | 'reject' | 'auto' | 'summary-fail' | 'summary-hold' = 'calculate'
  let summaryCount = 0, loaded = false, unloaded = false
  try {
    const page = await app.firstWindow()
    page.setDefaultTimeout(15_000)
    page.on('pageerror', error => errors.push(error.message))
    await pinNavigation(page)
    await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
    const instance = { id: 'tool-model', alias: '工具测试模型', backend: 'llama_cpp', profile_id: 'p1', model_id: 'm1', runtime_id: 'r1', state: 'ready', active_requests: 0, capabilities: { input: ['text'], tools: true } }
    const second = { ...instance, id: 'quick-model', alias: '快捷模型-NInfer-Qwen3.8-27B-NVFP4-long-loading-state', profile_id: 'p2', state: 'starting' }
    await page.route('**/api/state', async route => {
      const response = await route.fetch(), state = await response.json()
      await route.fulfill({ response, json: { ...state, instances: [...(unloaded ? [] : [instance]), ...(loaded ? [second] : [])], models: [{ id: 'm1', name: '工具测试模型', architecture: 'qwen35', format: 'gguf', files: [], modalities: ['text'], source: {}, metadata: {}, warnings: [], managed: false, size_bytes: 0, path: '/fixture/model.gguf' }],
        profiles: ['p1', 'p2'].map(id => ({ id, model_id: 'm1', runtime_id: 'r1', backend: 'llama_cpp', alias: id === 'p1' ? instance.alias : second.alias, name: id === 'p1' ? '默认预设' : '快捷预设', parameters: { context_length: 4096 }, generation: {}, extra_args: [], revision: 1, chat_quick_load: id === 'p2' })),
        runtimes: [{ id: 'r1', backend: 'llama_cpp', name: '测试运行环境', version: 'fixture', path: directory, managed: false, metadata: { supported_flags: ['--jinja'] } }],
      } })
    })
    await page.route('**/api/instances/tool-model/stop', async route => { unloaded = true; await route.fulfill({ json: { ...instance, state: 'stopped' } }) })
    await page.route('**/api/profiles/p2/start', async route => { expect(unloaded).toBe(true); loaded = true; await route.fulfill({ json: second }) })
    await page.route('**/api/instances/*/telemetry', route => route.fulfill({ json: { groups: [], unavailable: [], live_generation: { context_capacity: 4096 } } }))
    await page.route('**/api/chat', async route => {
      const body = route.request().postDataJSON(); requests.push(body)
      if (String(body.messages.at(-1)?.content).includes('请把以上对话整理')) {
        summaryCount++
        if (scenario === 'summary-fail') { await route.fulfill({ status: 500, json: { detail: '摘要服务失败' } }); return }
        if (scenario === 'summary-hold') return
        await route.fulfill({ json: { choices: [{ message: { role: 'assistant', content: '摘要：保留精确数值 2048，已确定使用本地模型。' }, finish_reason: 'stop' }] } }); return
      }
      const last = body.messages.at(-1)
      let message: any = { role: 'assistant', content: '继续完成。' }
      if (scenario === 'calculate') message = last.role === 'tool' ? { role: 'assistant', content: `计算结果 ${JSON.parse(last.content).result}` } : { role: 'assistant', content: null, tool_calls: [{ id: 'calc', type: 'function', function: { name: 'calculate', arguments: '{"expression":"6*7"}' } }] }
      if (scenario === 'write' || scenario === 'reject') {
        if (last.role === 'tool' && last.tool_call_id === 'write') message = { role: 'assistant', content: JSON.parse(last.content).written ? '文件已写入。' : '本次写入已取消。' }
        else if (scenario === 'write' && last.role !== 'tool') message = { role: 'assistant', content: null, tool_calls: [{ id: 'list', type: 'function', function: { name: 'list_directory', arguments: '{"path":""}' } }] }
        else if (last.tool_call_id === 'list') message = { role: 'assistant', content: null, tool_calls: [{ id: 'read', type: 'function', function: { name: 'read_file', arguments: '{"path":"note.txt"}' } }] }
        else message = { role: 'assistant', content: null, tool_calls: [{ id: 'write', type: 'function', function: { name: 'write_file', arguments: JSON.stringify({ path: 'note.txt', content: scenario === 'write' ? '确认后的内容\n' : '不应落盘\n' }) } }] }
      }
      if (scenario === 'write') {
        const tool = message.tool_calls?.[0]
        const reason = tool?.id === 'list' ? '先看目录。' : tool?.id === 'read' ? '再读文件。' : tool?.id === 'write' ? '最后写入。' : '写入完成，整理结果。'
        const deltas: any[] = [{ reasoning_content: reason }]
        if (tool) {
          deltas.push({ content: `执行 ${tool.function.name}。` })
          const middle = Math.floor(tool.function.arguments.length / 2)
          deltas.push({ tool_calls: [{ index: 0, id: tool.id, type: 'function', function: { name: tool.function.name, arguments: tool.function.arguments.slice(0, middle) } }] })
          deltas.push({ tool_calls: [{ index: 0, function: { arguments: tool.function.arguments.slice(middle) } }] })
        } else deltas.push({ content: message.content })
        const chunks = deltas.map(delta => ({ choices: [{ delta }] }))
        chunks.push({ choices: [{ delta: {}, finish_reason: tool ? 'tool_calls' : 'stop' }], usage: { prompt_tokens: 120, completion_tokens: 15, total_tokens: 135 } } as any)
        await route.fulfill({ contentType: 'text/event-stream', body: chunks.map(chunk => `data: ${JSON.stringify(chunk)}\n\n`).join('') + 'data: [DONE]\n\n' }); return
      }
      const chunk = { choices: [{ message, finish_reason: message.tool_calls ? 'tool_calls' : 'stop' }], usage: { prompt_tokens: 120, completion_tokens: 15, total_tokens: 135 } }
      await route.fulfill({ json: chunk })
    })
    await app.evaluate(({ ipcMain }, directory) => { ipcMain.removeHandler('pick-path'); ipcMain.handle('pick-path', () => directory) }, workspace)
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    const chat = async () => { await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '测试对话', exact: true }).click() }
    await chat(); await disableWebSearch(page)
    await expect(page.getByRole('button', { name: '联网搜索', exact: true })).toHaveText('')
    await openChatControls(page); await chatModel(page).click()
    await page.getByRole('option', { name: /快捷预设/ }).click()
    await expect(page.getByRole('button', { name: /^对话模型与推理强度：快捷模型/ })).toBeVisible()
    const modelPanel = page.getByRole('dialog', { name: '模型与推理强度', exact: true })
    const pendingModel = modelPanel.locator('[data-model-id="quick-model"]')
    await expect(pendingModel).toBeVisible()
    await expect(modelPanel.locator('[data-profile-id="p2"]')).toContainText('加载中')
    await modelPanel.screenshot({ path: path.join(directory, 'quick-load-starting.png'), animations: 'disabled' })
    await expect(modelPanel.getByText('已加载', { exact: true })).toHaveCount(0, { timeout: 1500 })
    await expect(pendingModel.locator('.lucide-check')).toHaveCount(0)
    await expect(pendingModel.getByRole('img', { name: '加载中', exact: true })).toBeVisible()
    await expect(pendingModel).toHaveCSS('color', await modelPanel.locator('.chat-model-group-label').first().evaluate(node => getComputedStyle(node).color))
    await expect(pendingModel.getByRole('button', { name: /^卸载 / })).toBeEnabled()
    second.state = 'ready'
    await page.keyboard.press('Escape')
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await openChatControls(page); await chatModel(page).click()
    await expect(modelPanel.locator('.chat-model-group-label').filter({ hasText: /^已加载$/ })).toBeVisible()
    await expect(pendingModel.locator('.lucide-check')).toBeVisible()
    await expect(modelPanel.locator('[data-profile-id="p2"]')).toContainText('已加载')
    await modelPanel.screenshot({ path: path.join(directory, 'quick-load-ready.png'), animations: 'disabled' })
    await page.keyboard.press('Escape')
    const options = page.getByRole('button', { name: '对话参数', exact: true })
    await options.click()
    await expect(page.getByRole('combobox', { name: '联网模式', exact: true })).toHaveCount(0)
    await page.getByRole('button', { name: '选择本地工具目录', exact: true }).click()
    await expect(page.getByRole('textbox', { name: '本地工具目录' })).toHaveValue(workspace)
    await expect(page.getByRole('button', { name: '工具调用', exact: true })).toHaveAttribute('aria-pressed', 'true')
    await page.getByRole('button', { name: '收起对话参数', exact: true }).click()
    const input = page.getByRole('textbox', { name: '对话消息', exact: true, includeHidden: true })
    const send = async (value: string) => { const preview = page.getByRole('region', { name: '输入内容预览', exact: true }); if (await preview.isVisible()) await preview.click({ position: { x: 6, y: 6 } }); await input.fill(value); await page.getByRole('button', { name: '发送消息', exact: true }).click() }
    await send('6*7 是多少？')
    await expect(page.locator('.chat-message.assistant').last().locator('.message-text')).toHaveText('计算结果 42')
    const firstActivity = page.locator('.chat-message.assistant').last().locator('.message-activity')
    await firstActivity.locator('summary').click()
    await expect(firstActivity.getByRole('button', { name: /计算 6\*7/ })).toHaveAttribute('aria-expanded', 'false')
    await firstActivity.getByRole('button', { name: /计算 6\*7/ }).click()
    await expect(firstActivity.locator('pre').last()).toContainText('42')
    scenario = 'write'; await send('把结果保存到 note.txt')
    const approval = page.getByRole('dialog', { name: '允许修改文件？' })
    await expect(approval).toBeVisible()
    expect(fs.readFileSync(path.join(workspace, 'note.txt'), 'utf8')).toBe('原始内容\n')
    const timeline = page.locator('.chat-message.assistant').last().locator('.message-activity')
    await expect(timeline).toHaveAttribute('open', '')
    await expect(timeline.locator('.chat-activity-events > *')).toHaveCount(9)
    await expect(timeline.locator('.chat-activity-tool')).toHaveCount(3)
    await expect(timeline.locator('.chat-activity-status').last()).toHaveAttribute('data-status', 'running')
    expect(await timeline.locator('.chat-activity-events > *').evaluateAll(nodes => nodes.map(node => node.className))).toEqual(Array(3).fill(['chat-activity-reasoning', 'chat-activity-text', 'chat-activity-tool']).flat())
    await expect.poll(() => approval.evaluate(element => getComputedStyle(element).opacity)).toBe('1')
    await page.screenshot({ path: path.join(directory, 'file-approval.png') })
    await approval.getByRole('button', { name: '允许写入', exact: true }).click()
    await expect(page.locator('.chat-message.assistant').last().locator('.message-text')).toHaveText('文件已写入。')
    expect(fs.readFileSync(path.join(workspace, 'note.txt'), 'utf8')).toBe('确认后的内容\n')
    await timeline.locator('summary').click()
    await expect(timeline.locator('.chat-activity-status').last()).toHaveAttribute('data-status', 'completed')
    await expect(timeline.locator('.chat-activity-tool-toggle')).toHaveCount(3)
    await expect.poll(() => timeline.locator('.reasoning-collapse').evaluate(element => getComputedStyle(element).opacity)).toBe('1')
    await timeline.locator('.reasoning-scroll').evaluate(element => { element.scrollTop = 0 })
    await page.locator('.chat-message.assistant').last().screenshot({ path: path.join(directory, 'ordered-tools-compact.png') })
    const conversations = await management(page, '/api/conversations')
    const timelineId = conversations.active_id
    await expect.poll(async () => (await management(page, `/api/conversations/${timelineId}`)).messages.at(-1).activity?.length).toBe(10)
    const persisted = (await management(page, `/api/conversations/${timelineId}`)).messages.at(-1)
    expect(persisted.activity.filter((event: any) => event.kind === 'tool').map((event: any) => event.status)).toEqual(['completed', 'completed', 'completed'])
    expect(persisted.activity.filter((event: any) => event.kind === 'reasoning').map((event: any) => event.text)).toEqual(['先看目录。', '再读文件。', '最后写入。', '写入完成，整理结果。'])
    await page.reload(); await expect(page.getByText('服务在线', { exact: true })).toBeVisible(); await pinNavigation(page); await chat()
    await page.locator('.chat-message.assistant').last().locator('.message-activity > summary').click()
    await expect(page.locator('.chat-message.assistant').last().locator('.chat-activity-events > *')).toHaveCount(10)
    scenario = 'reject'; await send('再次写入')
    await expect(approval).toBeVisible(); await approval.getByRole('button', { name: '拒绝', exact: true }).click()
    await expect(page.locator('.chat-message.assistant').last().locator('.message-text')).toHaveText('本次写入已取消。')
    expect(fs.readFileSync(path.join(workspace, 'note.txt'), 'utf8')).toBe('确认后的内容\n')

    await page.locator('.chat-message.assistant').last().locator('.message-activity > summary').click()
    await expect(page.locator('.chat-message.assistant').last().locator('.chat-activity-status')).toHaveAttribute('data-status', 'cancelled')
    expect(requests.every(request => request.instance_id === second.id)).toBe(true)

    // Seed a long, real persisted conversation; the inference endpoint alone is mocked.
    const seed = emptyConversation({ profile_id: 'p2', model_id: 'm1', alias: second.alias })
    const base = Date.now() - 60_000
    const history: Message[] = Array.from({ length: 3 }, (_, i) => [
      { id: `u${i}`, role: 'user' as const, text: `第 ${i} 轮精确数值 2048`, images: [], createdAt: base + i * 2000, status: 'completed' as const },
      { id: `a${i}`, role: 'assistant' as const, text: `第 ${i} 轮答案`, images: [], createdAt: base + i * 2000 + 1000, status: 'completed' as const, model: second.alias, modelId: 'm1', profileId: 'p2', usage: { prompt_tokens: 3500, completion_tokens: 300, total_tokens: 3800 }, contextCapacity: 4096 },
    ]).flat()
    const { messages: _, message_count: __, ...metadata } = seed
    await management(page, '/api/conversations', 'POST', { id: metadata.id, title: metadata.title, selection: metadata.selection, draft: metadata.draft, settings: { ...metadata.settings, webSearch: false, basicTools: true, toolDirectory: workspace } })
    await management(page, `/api/conversations/${seed.id}`, 'PATCH', { messages: history, activate: true })
    await page.reload(); await pinNavigation(page); await chat()
    scenario = 'summary-fail'
    await send('继续工作')
    await expect(page.getByText('自动压缩失败：摘要服务失败', { exact: true })).toBeVisible()
    await expect(input).toHaveValue('继续工作')
    await expect(page.locator('.chat-message')).toHaveCount(6)
    expect((await management(page, `/api/conversations/${seed.id}`)).compaction).toBeFalsy()
    scenario = 'auto'; const beforeAuto = requests.length
    await send('继续工作')
    await expect(page.locator('.chat-message.assistant').last().locator('.message-text')).toHaveText('继续完成。')
    expect(summaryCount).toBe(2)
    expect(requests.slice(beforeAuto)).toHaveLength(2)
    const final = requests.at(-1).messages
    expect(final.find((message: any) => String(message.content).includes('<conversation_summary>'))?.role).toBe('user')
    expect(final.some((message: any) => message.content === history[0].text)).toBe(false)
    expect(final.some((message: any) => message.content === history[4].text)).toBe(true)
    await expect(page.locator('.chat-save-status').filter({ hasText: /^已保存$/ })).toBeVisible()
    let saved = await management(page, `/api/conversations/${seed.id}`)
    expect(saved.messages).toHaveLength(8); expect(saved.compaction.summary).toContain('2048')
    await page.reload(); await pinNavigation(page); await chat(); await options.click()
    await expect(page.getByLabel('自动压缩上下文', { exact: true })).toBeChecked()
    await page.locator('.chat-context-summary summary').click()
    await expect(page.locator('.chat-context-summary')).toContainText('2048')
    await page.screenshot({ path: path.join(directory, 'chat-toolbar-settings.png') })
    await page.locator('.chat-composer').screenshot({ path: path.join(directory, 'composer.png') })
    await page.getByRole('button', { name: '恢复完整上下文', exact: true }).click()
    await page.getByRole('button', { name: '收起对话参数', exact: true }).click()
    scenario = 'summary-fail'
    await page.getByRole('button', { name: '压缩上下文', exact: true }).click()
    await expect(page.getByText('压缩失败：摘要服务失败', { exact: true })).toBeVisible()
    saved = await management(page, `/api/conversations/${seed.id}`)
    expect(saved.compaction).toBeFalsy(); expect(saved.messages).toHaveLength(8)
    scenario = 'summary-hold'
    await page.getByRole('button', { name: '压缩上下文', exact: true }).click()
    await expect(page.getByRole('status').filter({ hasText: '正在压缩上下文' })).toBeVisible()
    await page.getByRole('button', { name: '停止', exact: true }).click()
    await expect(page.getByRole('button', { name: '发送消息', exact: true })).toBeVisible()
    scenario = 'auto'
    await page.getByRole('button', { name: '压缩上下文', exact: true }).click()
    await expect(page.getByRole('button', { name: '压缩上下文', exact: true })).toHaveAttribute('data-compacted', 'true')
    await expect(page.locator('.chat-message')).toHaveCount(8)
    expect(errors).toEqual([])
    console.log('Chat tool screenshots:', directory)
  } finally {
    await Promise.all(app.windows().map(page => page.unrouteAll({ behavior: 'ignoreErrors' })))
    await app.close()
  }
})
