import { test, expect, _electron as electron, type Page, type Route } from '@playwright/test'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { pathToFileURL } from 'node:url'
import { emptyConversation, type Message } from '../src/chat-history'
import { pinNavigation } from './navigation'

const root = path.resolve(__dirname, '..')
const artifacts = process.env.MODLOADER_TEST_ARTIFACTS || path.join(root, '.artifacts/codex/chat-turn-update-20261027/metrics')
async function management(page: Page, route: string, method = 'GET', body?: unknown): Promise<any> {
  return page.evaluate(async ({ route, method, body }) => {
    const { baseUrl, token } = await window.modloader!.connection()
    const response = await fetch(baseUrl + route, { method, headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body), signal: AbortSignal.timeout(10_000) })
    if (!response.ok) throw new Error(await response.text())
    return response.status === 204 ? null : response.json()
  }, { route, method, body })
}

test('整轮账本：自动压缩、短调用、工具确认、失败取消、历史监测与手动压缩独立', async () => {
  test.setTimeout(180_000)
  fs.mkdirSync(artifacts, { recursive: true })
  const directory = fs.mkdtempSync(path.join(artifacts, 'turn-ui-')), workspace = path.join(directory, 'workspace')
  fs.mkdirSync(workspace); fs.writeFileSync(path.join(workspace, 'result.txt'), '原始内容', 'utf8')
  const app = await electron.launch({ args: [root], env: { ...process.env, MODLOADER_DATA_DIR: path.join(directory, 'data'), MODLOADER_SENSOR_AUTO_START: '0', MODLOADER_DEV_URL: process.env.MODLOADER_TURN_RENDERER_URL || pathToFileURL(path.join(root, 'dist/index.html')).href }, timeout: 60_000 })
  const errors: string[] = [], calls: Array<{ id: string; turn: string; body: any }> = []
  let scenario = 'success', lastReport: any = null
  let contextCapacity: number | null = 4096
  let resumeInitialGeneration!: () => void
  const initialGeneration = new Promise<void>(resolve => { resumeInitialGeneration = resolve })
  const instance = { id: 'turn-model', alias: '轮次计量测试模型', backend: 'llama_cpp', profile_id: 'p1', model_id: 'm1', runtime_id: 'r1', state: 'ready', active_requests: 0, capabilities: { input: ['text'], tools: true } }
  try {
    const page = await app.firstWindow(); page.setDefaultTimeout(15_000)
    page.on('pageerror', error => errors.push(error.message))
    await page.route('**/api/state', async route => {
      const response = await route.fetch(), state = await response.json()
      await route.fulfill({ response, json: { ...state, instances: [instance], models: [{ id: 'm1', name: instance.alias, architecture: 'qwen35', format: 'gguf', files: [], modalities: ['text'], source: {}, metadata: {}, warnings: [], managed: false, size_bytes: 0, path: '/fixture/model.gguf' }], profiles: [{ id: 'p1', model_id: 'm1', runtime_id: 'r1', backend: 'llama_cpp', alias: instance.alias, name: '计量测试', parameters: { context_length: 4096 }, generation: {}, extra_args: [], revision: 1 }], runtimes: [{ id: 'r1', backend: 'llama_cpp', name: '测试运行环境', version: 'fixture', path: directory, managed: false, metadata: { supported_flags: ['--jinja'] } }] } })
    })
    await page.route('**/api/instances/turn-model/telemetry', route => route.fulfill({ json: { instance_id: instance.id, backend: instance.backend, state: 'ready', updated_at: Date.now() / 1000,
      // The monitor renders charts/context for measured requests, not an empty idle instance.
      groups: lastReport ? [] : [{ id: 'api_request', title: '先前请求', request_key: 'prior-request', request_status: 'completed', metrics: [
        { id: 'output_speed', label: 'Decode', value: 40, kind: 'gauge', rate_basis: 'decode' },
        { id: 'prompt_tokens', label: '输入', value: 100, kind: 'gauge' }, { id: 'completion_tokens', label: '输出', value: 20, kind: 'gauge' },
      ] }], unavailable: [], live_generation: { request_key: null, scope: 'request', source: 'fixture', backend_running: 0, backend_queued: 0, available_for_chat: false, tokens_per_second: null, window_s: null, reason: null, context_capacity: contextCapacity } } }))
    await page.route('**/api/instances/turn-model/request-metrics/*', route => lastReport && route.request().url().endsWith(lastReport.client_call_id) ? route.fulfill({ json: lastReport }) : route.fulfill({ status: 404, json: { detail: '记录不可用' } }))
    const respond = async (route: Route, message: any, usage: any, timings: any, streaming: boolean, failure = false) => {
      const headers = route.request().headers(), id = headers['x-modloader-call-id'], turn = headers['x-modloader-request-id']
      const report = { request_key: `backend-${id}`, client_call_id: id, client_request_id: turn, status: failure ? 'failed' : 'completed', started_at: Date.now() / 1000 - .1, ended_at: Date.now() / 1000, duration_s: .1, first_output_s: streaming ? .02 : null, usage, timings }
      lastReport = report
      const finish_reason = message.tool_calls ? 'tool_calls' : 'stop'
      if (!streaming) { await route.fulfill({ json: { choices: [{ message, finish_reason }], usage, timings, modloader_request: report } }); return }
      const delta = { ...message, ...(message.tool_calls ? { tool_calls: message.tool_calls.map((call: any, index: number) => ({ ...call, index })) } : {}) }
      const chunks = [{ choices: [{ delta }], usage: { completion_tokens: 2 } }, { choices: [{ delta: {}, finish_reason }], usage, timings }, { modloader_request: report, ...(failure ? { error: { message: '计量夹具失败' } } : {}) }]
      await route.fulfill({ contentType: 'text/event-stream', body: chunks.map(chunk => `data: ${JSON.stringify(chunk)}\n\n`).join('') + (failure ? '' : 'data: [DONE]\n\n') })
    }
    await page.route('**/api/chat', async route => {
      const body = route.request().postDataJSON(), headers = route.request().headers()
      const id = headers['x-modloader-call-id'], turn = headers['x-modloader-request-id']
      calls.push({ id, turn, body })
      const summary = String(body.messages.at(-1)?.content).includes('请把以上对话整理')
      if (summary) {
        if (scenario === 'auto-fail') { await route.fulfill({ status: 500, json: { detail: '摘要服务失败' } }); return }
        await respond(route, { role: 'assistant', content: '摘要：保留已确认数值 2048。' }, { prompt_tokens: 100, completion_tokens: 8, prompt_tokens_details: { cached_tokens: 20 }, completion_tokens_details: { reasoning_tokens: 2 } }, { predicted_n: 8, predicted_ms: 80, prompt_n: 80, prompt_ms: 20, cache_n: 20 }, false); return
      }
      if (scenario === 'cancel') {
        lastReport = { request_key: `backend-${id}`, client_call_id: id, client_request_id: turn, status: 'cancelled', duration_s: .2, first_output_s: .04, usage: { prompt_tokens: 90, completion_tokens: 7 }, timings: { predicted_n: 7, predicted_ms: 70 } }
        return
      }
      if (scenario === 'failure') { await respond(route, { content: '已保留部分输出' }, { prompt_tokens: 110, completion_tokens: 5 }, { predicted_n: 5, predicted_ms: 50 }, true, true); return }
      if (scenario === 'planning') {
        const planning = body.stream === false
        await respond(route, { content: planning ? '{"queries":["测试查询"],"image_queries":[]}' : '检索完成。' }, { prompt_tokens: planning ? 20 : 40, completion_tokens: planning ? 5 : 10 }, { predicted_n: planning ? 5 : 10, predicted_ms: planning ? 50 : 100 }, !planning); return
      }
      const tools = body.messages.at(-1)?.role !== 'tool'
      if (scenario === 'success' && tools) await initialGeneration
      const message = tools ? { role: 'assistant', content: null, tool_calls: [{ id: 'calc', type: 'function', function: { name: 'calculate', arguments: '{"expression":"6*7"}' } }, { id: 'write', type: 'function', function: { name: 'write_file', arguments: '{"path":"result.txt","content":"结果 42"}' } }] } : { role: 'assistant', content: '结果 42，已经保存。' }
      await respond(route, message, { prompt_tokens: tools ? 200 : 240, completion_tokens: tools ? 10 : 20, prompt_tokens_details: { cached_tokens: tools ? 50 : 40 }, completion_tokens_details: { reasoning_tokens: tools ? 3 : 4 } }, { predicted_n: tools ? 10 : 20, predicted_ms: tools ? 100 : 400, prompt_n: tools ? 150 : 200, prompt_ms: tools ? 30 : 40, cache_n: tools ? 50 : 40 }, true)
    })
    await page.route('**/api/web-search', route => route.fulfill({ json: { query: '测试查询', sources: [], elapsed_ms: 1 } }))
    await pinNavigation(page)
    await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
    const chat = async () => { await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '测试对话', exact: true }).click() }
    const navigateMonitor = async () => { await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '运行监控', exact: true }).click() }
    const longHistory: Message[] = Array.from({ length: 3 }, (_, index) => [
      { id: `u-${index}`, role: 'user' as const, text: `第 ${index} 轮上下文 2048`, images: [], createdAt: Date.now() - 60_000 + index * 2000, status: 'completed' as const },
      { id: `a-${index}`, role: 'assistant' as const, text: `第 ${index} 轮回答`, images: [], createdAt: Date.now() - 59_000 + index * 2000, status: 'completed' as const, model: instance.alias, profileId: 'p1', modelId: 'm1', usage: { prompt_tokens: 3500, completion_tokens: 300 }, contextCapacity: 4096 },
    ]).flat()
    const seed = async (name: string, history: Message[] = [], autoCompact = false) => {
      const conversation = emptyConversation({ profile_id: 'p1', model_id: 'm1', alias: instance.alias })
      await management(page, '/api/conversations', 'POST', { id: conversation.id, title: name, selection: conversation.selection, draft: conversation.draft, settings: { ...conversation.settings, stream: true, webSearch: name === 'planning', webSearchMode: name === 'planning' ? 'retrieve' : 'agent', basicTools: name !== 'planning', toolDirectory: workspace, autoCompact } })
      await management(page, `/api/conversations/${conversation.id}`, 'PATCH', { messages: history, activate: true })
      await page.reload(); await pinNavigation(page); await chat()
      return conversation.id
    }
    const send = async (text: string) => { await page.getByRole('textbox', { name: '对话消息', exact: true, includeHidden: true }).fill(text); await page.getByRole('button', { name: '发送消息', exact: true }).click() }
    const successId = await seed('success', longHistory, true)
    // Capacity is known by the monitor, then absent from later telemetry.
    await navigateMonitor()
    await expect(page.locator('.monitor-context-capacity')).toHaveText('容量 4,096 token')
    contextCapacity = null
    await chat()
    await send('计算 6*7 并写入结果。')
    const footer = page.locator('.chat-message.assistant').last().locator('.message-turn-summary')
    const elapsedMetric = footer.locator('[data-metric="elapsed"]'), tooltip = page.getByRole('tooltip')
    await elapsedMetric.hover()
    await expect(tooltip).toBeVisible()
    await expect(tooltip).toContainText('本轮耗时')
    const runningHint = await tooltip.innerText()
    await expect.poll(() => tooltip.innerText()).not.toBe(runningHint)
    await expect(tooltip).toHaveCSS('opacity', '1')
    await page.screenshot({ path: path.join(directory, 'running-metric-tooltip.png') })
    await footer.locator('[data-metric="input"]').hover()
    await expect(tooltip).toContainText('未缓存输入')
    await page.mouse.move(0, 0)
    await elapsedMetric.focus()
    await page.keyboard.press('Tab')
    await page.keyboard.press('Shift+Tab')
    await expect(tooltip).toContainText('本轮耗时')
    await page.keyboard.press('Escape')
    const escapedAt = await elapsedMetric.innerText()
    await expect.poll(() => elapsedMetric.innerText()).not.toBe(escapedAt)
    await expect(tooltip).toBeHidden()
    await page.mouse.move(0, 0)
    await footer.getByRole('button', { name: /^查看本轮 \d+ 次模型请求统计$/ }).click()
    await expect(page.locator('.monitor-context-capacity')).toHaveText('容量 4,096 token')
    await chat()
    resumeInitialGeneration()
    const approval = page.getByRole('dialog', { name: '允许修改文件？' })
    await expect(approval).toBeVisible()
    const inspect = footer.getByRole('button', { name: /^查看本轮 \d+ 次模型请求统计$/ })
    const initialInspectPosition = (await inspect.boundingBox())!
    const initialElapsed = parseFloat((await footer.locator('[data-metric="elapsed"]').innerText()).replace('s', ''))
    await expect.poll(async () => parseFloat((await footer.locator('[data-metric="elapsed"]').innerText()).replace('s', ''))).toBeGreaterThan(initialElapsed + .3)
    const laterInspectPosition = (await inspect.boundingBox())!
    expect(laterInspectPosition.x).toBeCloseTo(initialInspectPosition.x, 1)
    expect(laterInspectPosition.width).toBeCloseTo(initialInspectPosition.width, 1)
    await approval.getByRole('button', { name: '允许写入', exact: true }).click()
    await expect(page.locator('.chat-message.assistant').last().locator('.message-text')).toHaveText('结果 42，已经保存。')
    await expect.poll(async () => (await management(page, `/api/conversations/${successId}`)).messages.at(-1).turn?.status).toBe('completed')
    const success = (await management(page, `/api/conversations/${successId}`)).messages.at(-1)
    expect(success.requests.map((request: any) => request.kind)).toEqual(['compaction', 'generation', 'generation'])
    expect(success.requests.map((request: any) => request.index)).toEqual([1, 2, 3])
    expect(new Set(success.requests.map((request: any) => request.id)).size).toBe(3)
    expect(success.turn.input).toEqual({ value: 540, reported: 3 }); expect(success.turn.output.value).toBe(38)
    expect(success.turn.cached.value).toBe(110); expect(success.turn.reasoning.value).toBe(9)
    expect(success.contextCapacity).toBe(4096)
    expect(success.turn.decode.value).toBeCloseTo(38 / .58, 8)
    expect(success.usage).toMatchObject({ prompt_tokens: 240, completion_tokens: 20 })
    expect(success.timings.predicted_n).toBe(20)
    const events = success.activity.filter((event: any) => event.kind === 'tool')
    expect(events.map((event: any) => event.requestId)).toEqual([success.requests[1].id, success.requests[1].id])
    expect(events.every((event: any) => event.status === 'completed' && event.elapsed >= 0)).toBe(true)
    expect(success.turn.elapsed).toBeGreaterThan(events[1].elapsed)
    expect(fs.readFileSync(path.join(workspace, 'result.txt'), 'utf8')).toBe('结果 42')
    const inputMetric = footer.locator('[data-metric="input"]')
    await expect(inputMetric).toHaveText('200')
    await expect(footer.locator('[data-metric="cached"]')).toHaveText('40')
    await expect(footer.locator('[data-metric="output"]')).toHaveText('20')
    await expect(inputMetric).toHaveAttribute('aria-label', '末次请求未缓存输入 200 Token')
    const metricColor = await inputMetric.evaluate(node => getComputedStyle(node).color)
    await inputMetric.hover()
    await expect(inputMetric).toHaveCSS('color', metricColor)
    await expect(page.getByRole('tooltip')).toContainText('输入')
    await expect(page.getByRole('tooltip')).not.toContainText('次请求已报告')
    await inputMetric.click()
    await expect(page.locator('.chat-message.assistant').last()).toBeVisible()
    await expect(page.locator('.monitor-page')).toBeHidden()
    await expect(inspect).toHaveCount(1)
    await inspect.hover()
    await expect(inspect).not.toHaveCSS('color', metricColor)
    await expect(inputMetric).toHaveCSS('color', metricColor)
    await footer.screenshot({ path: path.join(directory, 'turn-footer.png'), animations: 'disabled' })
    await inspect.focus()
    await page.keyboard.press('Enter')
    const monitor = page.locator('.monitor-page'), chart = monitor.locator('.monitor-generation-chart')
    await expect(monitor.locator('.monitor-context-capacity')).toHaveText('容量 4,096 token')
    await expect(monitor.locator('.monitor-request-table tbody tr[data-request-index]')).toHaveCount(3)
    await expect(chart).toHaveAttribute('data-mode', 'turn')
    await expect(chart.locator('.monitor-request-point')).toHaveCount(3)
    await expect(monitor.locator('.monitor-speed-current strong')).toHaveText('65.52')
    const request2Row = monitor.locator('[data-request-index="2"]')
    const request2Button = monitor.getByRole('button', { name: '选择请求 #2', exact: true })
    const request2ChartPoint = chart.getByRole('button', { name: /^请求 #2，/ })
    const request2Expanded = monitor.locator('[data-request-index="2"] + .monitor-request-expanded')
    const expectRequest2Selected = async () => {
      await expect(request2Button).toHaveAttribute('aria-pressed', 'true')
      await expect(request2ChartPoint).toHaveAttribute('aria-pressed', 'true')
      await expect(monitor.locator('.monitor-request-heading h3')).toHaveText('请求 #2')
      await expect(monitor.locator('.monitor-request-live div').nth(2).locator('dd strong')).toHaveText('200')
      await expect(monitor.locator('.monitor-request-live div').nth(3).locator('dd strong')).toHaveText('10')
    }
    const expectRequest2Collapsed = async () => {
      await expect(request2Row).toHaveAttribute('data-selected', 'true')
      await expect(request2Row).toHaveAttribute('data-expanded', 'false')
      await expect(request2Expanded).toBeHidden()
      await expect(request2Button).toHaveAttribute('aria-expanded', 'false')
      await expect(request2ChartPoint).toHaveAttribute('aria-expanded', 'false')
      await expectRequest2Selected()
    }
    await monitor.getByRole('button', { name: '选择请求 #2', exact: true }).click()
    await expect(chart.getByRole('button', { name: /^请求 #2，/ })).toHaveAttribute('aria-pressed', 'true')
    await expect(monitor.locator('.monitor-request-detail')).toContainText('写入文件')
    await expect(request2Row).toHaveAttribute('data-selected', 'true')
    await expect(monitor.locator('[data-request-index="2"]')).toHaveAttribute('data-expanded', 'true')
    await expect(request2Button).toHaveAttribute('aria-expanded', 'true')
    await expect(request2ChartPoint).toHaveAttribute('aria-expanded', 'true')
    await expect(request2Expanded).toBeVisible()
    await expect(monitor.locator('[data-request-index="2"] + .monitor-request-expanded .monitor-request-detail')).toContainText('写入文件')
    await expect(monitor.locator('[data-request-index="2"] [data-metric="input"] .monitor-request-readout strong')).toHaveText('200')
    await expect(monitor.locator('[data-request-index="2"] [data-metric="output"] .monitor-request-readout strong')).toHaveText('10')
    await expect(monitor.locator('[data-request-index="2"] [data-metric="cached"] .monitor-request-readout strong')).toHaveText('50')
    const request2NameColor = await request2Row.locator('.monitor-request-name').evaluate(node => getComputedStyle(node).color)
    const request1NameColor = await monitor.locator('[data-request-index="1"] .monitor-request-name').evaluate(node => getComputedStyle(node).color)
    expect(request2NameColor).not.toBe(request1NameColor)
    const request1CellColor = await monitor.locator('[data-request-index="1"] td[data-group="tokens"]').evaluate(node => getComputedStyle(node).backgroundColor)
    await expect(request2Row.locator('td[data-group="tokens"]')).toHaveCSS('background-color', request1CellColor)
    await monitor.locator('[data-request-index="2"] [data-metric="output"]').hover()
    await expect(page.getByRole('tooltip')).toContainText('思考 3 Token')
    await page.mouse.move(0, 0)
    await expect(page.getByRole('tooltip')).toBeHidden()
    await monitor.locator('.monitor-request-section').screenshot({ path: path.join(directory, 'request-detail-panel.png'), animations: 'disabled' })
    const { width, height, fontScale } = await page.evaluate(() => ({ width: innerWidth, height: innerHeight, fontScale: document.documentElement.style.getPropertyValue('--font-scale') }))
    await page.setViewportSize({ width: 940, height: 900 })
    await page.evaluate(() => document.documentElement.style.setProperty('--font-scale', '1.3'))
    await monitor.locator('.monitor-request-section').screenshot({ path: path.join(directory, 'request-detail-narrow.png'), animations: 'disabled' })
    await page.evaluate(value => document.documentElement.style.setProperty('--font-scale', value), fontScale)
    await page.setViewportSize({ width, height })
    await request2Button.click()
    await expectRequest2Collapsed()
    await monitor.locator('.monitor-request-section').screenshot({ path: path.join(directory, 'request-detail-collapsed.png'), animations: 'disabled' })
    const request2TokensCell = request2Row.locator('td[data-group="tokens"]')
    await request2TokensCell.click()
    await expect(request2Row).toHaveAttribute('data-expanded', 'true')
    await expect(request2Button).toHaveAttribute('aria-expanded', 'true')
    await expect(request2ChartPoint).toHaveAttribute('aria-expanded', 'true')
    await expect(request2Expanded).toBeVisible()
    await expect(request2Expanded.locator('.monitor-request-detail')).toContainText('写入文件')
    await request2TokensCell.click()
    await expectRequest2Collapsed()
    await request2Button.focus()
    await page.keyboard.press('Space')
    await expect(request2Row).toHaveAttribute('data-expanded', 'true')
    await expect(request2Button).toHaveAttribute('aria-expanded', 'true')
    await expect(request2ChartPoint).toHaveAttribute('aria-expanded', 'true')
    await expect(request2Expanded).toBeVisible()
    await expect(request2Expanded.locator('.monitor-request-detail')).toContainText('写入文件')
    await request2Button.focus()
    await page.keyboard.press('Enter')
    await expectRequest2Collapsed()
    await request2ChartPoint.click()
    await expect(request2Row).toHaveAttribute('data-expanded', 'true')
    await expect(request2Button).toHaveAttribute('aria-expanded', 'true')
    await expect(request2ChartPoint).toHaveAttribute('aria-expanded', 'true')
    await expect(request2Expanded).toBeVisible()
    await expect(request2Expanded.locator('.monitor-request-detail')).toContainText('写入文件')
    await request2Button.click()
    await expectRequest2Collapsed()
    await request2ChartPoint.click()
    await expect(request2Row).toHaveAttribute('data-expanded', 'true')
    await expect(request2Button).toHaveAttribute('aria-expanded', 'true')
    await expect(request2ChartPoint).toHaveAttribute('aria-expanded', 'true')
    await expect(request2Expanded).toBeVisible()
    await expect(request2Expanded.locator('.monitor-request-detail')).toContainText('写入文件')
    await request2ChartPoint.click()
    await expectRequest2Collapsed()
    await request2ChartPoint.click()
    await expect(request2Row).toHaveAttribute('data-expanded', 'true')
    await expect(request2Button).toHaveAttribute('aria-expanded', 'true')
    await expect(request2ChartPoint).toHaveAttribute('aria-expanded', 'true')
    await expect(request2Expanded).toBeVisible()
    await request2ChartPoint.focus()
    await page.keyboard.press('Space')
    await expectRequest2Collapsed()
    await request2ChartPoint.focus()
    await page.keyboard.press('Enter')
    await expect(request2Row).toHaveAttribute('data-expanded', 'true')
    await expect(request2Button).toHaveAttribute('aria-expanded', 'true')
    await expect(request2ChartPoint).toHaveAttribute('aria-expanded', 'true')
    await expect(request2Expanded).toBeVisible()
    await expect(request2Expanded.locator('.monitor-request-detail')).toContainText('写入文件')
    await expectRequest2Selected()
    await chart.getByRole('button', { name: /^请求 #1，/ }).click()
    await expect(monitor.getByRole('button', { name: '选择请求 #1', exact: true })).toHaveAttribute('aria-pressed', 'true')
    await monitor.getByRole('button', { name: '实时', exact: true }).click()
    await expect(chart.locator('.monitor-generation-empty')).toHaveText('这次请求没有实时采样点')
    await expect(chart.locator('.trend-line, .trend-dot')).toHaveCount(0)
    await monitor.getByRole('button', { name: '本轮', exact: true }).click()
    await monitor.getByRole('button', { name: 'Prefill', exact: true }).click()
    await expect(monitor.locator('.monitor-speed-current strong')).toHaveText('4,777.78')
    await monitor.getByRole('button', { name: 'Decode', exact: true }).click()
    await monitor.locator('.monitor-generation-grid').screenshot({ path: path.join(directory, 'turn-monitor.png'), animations: 'disabled' })
    await page.reload(); await pinNavigation(page); await chat()
    await expect(page.locator('.chat-message.assistant').last().locator('[data-metric="input"]')).toHaveText('200')
    await expect(page.locator('.chat-message.assistant').last().locator('[data-metric="cached"]')).toHaveText('40')
    await expect(page.locator('.chat-message.assistant').last().locator('[data-metric="output"]')).toHaveText('20')
    await page.locator('.chat-message.assistant').last().getByRole('button', { name: '查看本轮 3 次模型请求统计' }).click()
    await expect(monitor.locator('.monitor-request-table tbody tr[data-request-index]')).toHaveCount(3)
    await expect(monitor.locator('.monitor-context-capacity')).toHaveText('容量 4,096 token')
    // Manual compression creates an independent operation, not a fourth request
    // in the completed answer's persisted totals.
    await chat(); scenario = 'manual'
    await page.getByRole('button', { name: '压缩上下文', exact: true }).click()
    await expect(page.getByRole('button', { name: '发送消息', exact: true })).toBeVisible()
    await navigateMonitor()
    await expect(monitor.locator('.monitor-selector-main')).not.toContainText('所选轮次')
    await expect(monitor.locator('.monitor-request-table tbody tr[data-request-index]')).toHaveCount(1)
    await expect(monitor.locator('.monitor-request-table tbody')).toContainText('手动压缩')
    expect((await management(page, `/api/conversations/${successId}`)).messages.at(-1).turn.requests).toBe(3)
    scenario = 'planning'; const planningId = await seed('planning')
    await send('查一下资料。')
    await expect(page.locator('.chat-message.assistant').last().locator('.message-text')).toHaveText('检索完成。')
    await expect.poll(async () => (await management(page, `/api/conversations/${planningId}`)).messages.at(-1).turn?.status).toBe('completed')
    const planning = (await management(page, `/api/conversations/${planningId}`)).messages.at(-1)
    expect(planning.requests.map((request: any) => request.kind)).toEqual(['planning', 'generation'])
    expect(planning.turn.input.value).toBe(60); expect(planning.turn.output.value).toBe(15)
    expect(planning.usage.prompt_tokens).toBe(40)
    scenario = 'failure'; const failedId = await seed('failure')
    await send('测试失败部分计量。')
    await expect(page.locator('.chat-message.assistant').last().locator('.message-error')).toHaveText('计量夹具失败')
    await expect.poll(async () => (await management(page, `/api/conversations/${failedId}`)).messages.at(-1).turn?.status).toBe('failed')
    const failed = (await management(page, `/api/conversations/${failedId}`)).messages.at(-1)
    expect(failed.requests[0]).toMatchObject({ status: 'failed', usage: { completion_tokens: 5 } }); expect(failed.turn.output.value).toBe(5)
    scenario = 'cancel'; const cancelId = await seed('cancel')
    const beforeCancel = calls.length
    await send('测试取消部分计量。')
    await expect.poll(() => calls.length).toBe(beforeCancel + 1)
    await page.getByRole('button', { name: '停止', exact: true }).click()
    await expect.poll(async () => (await management(page, `/api/conversations/${cancelId}`)).messages.at(-1).turn?.status).toBe('cancelled')
    const cancelled = (await management(page, `/api/conversations/${cancelId}`)).messages.at(-1)
    expect(cancelled.requests[0]).toMatchObject({ status: 'cancelled', usage: { prompt_tokens: 90, completion_tokens: 7 } })
    expect(cancelled.turn.output.value).toBe(7); expect(cancelled.status).toBe('stopped')
    scenario = 'auto-fail'; const autoFailedId = await seed('auto-fail', longHistory, true)
    await send('保留这个草稿。')
    await expect(page.getByText('自动压缩失败：摘要服务失败', { exact: true })).toBeVisible()
    await expect(page.getByRole('textbox', { name: '对话消息', exact: true, includeHidden: true })).toHaveValue('保留这个草稿。')
    expect((await management(page, `/api/conversations/${autoFailedId}`)).messages).toHaveLength(6)
    await navigateMonitor()
    await expect(monitor.locator('.monitor-request-table tbody tr[data-request-index]')).toHaveCount(1)
    await expect(monitor.locator('.monitor-request-table tbody')).toContainText('自动压缩')
    expect(calls.every(call => Boolean(call.id && call.turn))).toBe(true)
    // Persisted replies without a turn ledger keep the same split and stable units.
    const legacy: Message = { id: 'legacy-reply', role: 'assistant', text: '旧回复。', images: [], createdAt: Date.now(), status: 'completed', elapsed: 2,
      firstOutput: .5, usage: { prompt_tokens: 16, completion_tokens: 25, prompt_tokens_details: { cached_tokens: 4 } }, timings: { predicted_per_second: 12.5 } }
    await seed('legacy', [longHistory[0], legacy])
    const legacyFooter = page.locator('.chat-message.assistant').last().locator('.message-footer')
    await expect(legacyFooter.locator('[data-metric="input"]')).toHaveText('12')
    await expect(legacyFooter.locator('[data-metric="cached"]')).toHaveText('4')
    await expect(legacyFooter.locator('[data-metric="speed"]')).toHaveText('DEC12.50tok/s')
    await expect(legacyFooter.locator('.message-turn-inspect')).toHaveCount(0)
    await seed('legacy-unknown-cache', [longHistory[0], { ...legacy, usage: { prompt_tokens: 16, completion_tokens: 25 } }])
    await expect(legacyFooter.locator('[data-metric="input"]')).toHaveText('—')
    await expect(legacyFooter.locator('[data-metric="cached"]')).toHaveText('—')
    expect(errors).toEqual([])
    console.log('Turn metrics fixture:', directory)
  } finally { resumeInitialGeneration(); await Promise.all(app.windows().map(page => page.unrouteAll({ behavior: 'ignoreErrors' }))); await app.close() }
})
