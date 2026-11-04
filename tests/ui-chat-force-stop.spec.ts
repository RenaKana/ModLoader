import { test, expect, _electron as electron, type Page } from '@playwright/test'
import { createServer, type ServerResponse } from 'node:http'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { pathToFileURL } from 'node:url'
import { emptyConversation } from '../src/chat-history'
import { pinNavigation } from './navigation'

const root = path.resolve(__dirname, '..')
const artifacts = process.env.MODLOADER_TEST_ARTIFACTS || path.join(root, '.artifacts/codex/stream-stop-metrics-20261101/ui')
async function management(page: Page, route: string, method = 'GET', body?: unknown): Promise<any> {
  return page.evaluate(async ({ route, method, body }) => {
    const { baseUrl, token } = await window.modloader!.connection()
    const response = await fetch(baseUrl + route, { method, headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) })
    if (!response.ok) throw new Error(await response.text())
    return response.status === 204 ? null : response.json()
  }, { route, method, body })
}

for (const scenario of ['before-first', 'partial', 'receipt-recovery', 'non-stop-failure'] as const) {
  test(`强停终态：${scenario} 保留正文与请求状态、不继续工具链`, async () => {
    test.setTimeout(90_000)
    fs.mkdirSync(artifacts, { recursive: true })
    const directory = fs.mkdtempSync(path.join(artifacts, scenario + '-'))
    let response: ServerResponse | undefined
    let lastReport: any = null, requestCount = 0, recovered = 0
    const server = createServer(async (request, result) => {
      result.setHeader('Access-Control-Allow-Origin', '*')
      result.setHeader('Access-Control-Allow-Headers', '*')
      if (request.method === 'OPTIONS') { result.end(); return }
      let raw = ''
      for await (const chunk of request) raw += chunk
      JSON.parse(raw)
      requestCount++
      lastReport = { request_key: `fixture-${requestCount}`, client_call_id: request.headers['x-modloader-call-id'],
        client_request_id: request.headers['x-modloader-request-id'], status: scenario === 'non-stop-failure' ? 'failed' : 'cancelled',
        duration_s: .2, usage: { completion_tokens: scenario === 'before-first' ? 0 : 3 } }
      result.setHeader('Content-Type', 'text/event-stream')
      result.flushHeaders()
      response = result
    })
    await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
    const port = (server.address() as { port: number }).port
    const app = await electron.launch({ args: [root], env: { ...process.env, MODLOADER_SENSOR_AUTO_START: '0', MODLOADER_DATA_DIR: path.join(directory, 'data'),
      MODLOADER_DEV_URL: pathToFileURL(process.env.MODLOADER_TEST_RENDERER || path.join(root, 'dist/index.html')).href }, timeout: 60_000 })
    const errors: string[] = []
    try {
      const page = await app.firstWindow()
      page.setDefaultTimeout(12_000)
      page.on('pageerror', error => errors.push(error.message))
      await pinNavigation(page)
      await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
      const instance = { id: 'stop-fixture', alias: '强停验证模型', backend: 'llama_cpp', profile_id: 'p1', model_id: 'm1', runtime_id: 'r1', state: 'ready', active_requests: 0, capabilities: { input: ['text'], tools: true } }
      await page.route('**/api/state', async route => {
        const original = await route.fetch(), state = await original.json()
        await route.fulfill({ response: original, json: { ...state, instances: [instance],
          models: [{ id: 'm1', name: instance.alias, architecture: 'qwen35', format: 'gguf', files: [], modalities: ['text'], source: {}, metadata: {}, warnings: [], managed: false, size_bytes: 0, path: '/fixture/model.gguf' }],
          profiles: [{ id: 'p1', model_id: 'm1', runtime_id: 'r1', backend: 'llama_cpp', alias: instance.alias, name: '强停验证', parameters: {}, generation: {}, extra_args: [], revision: 1 }],
          runtimes: [{ id: 'r1', backend: 'llama_cpp', name: '无模型夹具', path: directory, version: 'fixture', metadata: {}, managed: false }] } })
      })
      await page.route('**/api/instances/stop-fixture/telemetry', route => route.fulfill({ json: { groups: [], unavailable: [], live_generation: null } }))
      await page.route('**/api/instances/stop-fixture/request-metrics/*', route => {
        recovered++
        return route.fulfill({ json: lastReport })
      })
      await page.route('**/api/chat', route => route.continue({ url: `http://127.0.0.1:${port}/api/chat` }))
      const conversation = emptyConversation({ profile_id: 'p1', model_id: 'm1', alias: instance.alias })
      await management(page, '/api/conversations', 'POST', { id: conversation.id, title: '强停终态验证', selection: conversation.selection,
        draft: conversation.draft, settings: { ...conversation.settings, stream: true, webSearch: false, basicTools: true } })
      await management(page, `/api/conversations/${conversation.id}`, 'PATCH', { activate: true })
      await page.reload(); await pinNavigation(page)
      await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '测试对话', exact: true }).click()
      await page.getByRole('textbox', { name: '对话消息', exact: true, includeHidden: true }).fill('保留部分回答，停止后不要继续计算。')
      await page.getByRole('button', { name: '发送消息', exact: true }).click()
      await expect.poll(() => Boolean(response)).toBe(true)
      const assistant = page.locator('.chat-message.assistant').last()
      if (scenario !== 'before-first') {
        response!.write(`data: ${JSON.stringify({ choices: [{ delta: { content: '已经生成的内容应保留。' } }] })}\n\n`)
        await expect(assistant.locator('.message-text')).toContainText('已经生成的内容应保留。')
        response!.write(`data: ${JSON.stringify({ choices: [{ delta: { tool_calls: [{ index: 0, id: 'must-not-run', type: 'function', function: { name: 'calculate', arguments: '{"expression":"6*7"}' } }] } }] })}\n\n`)
      }
      if (scenario !== 'receipt-recovery') response!.write(`data: ${JSON.stringify({ error: { message: '后端连接已结束' }, modloader_request: lastReport })}\n\n`)
      response!.end() // Forced stop must not manufacture [DONE].
      await expect(page.getByRole('button', { name: '停止生成', exact: true })).toHaveCount(0)
      const expected = scenario === 'non-stop-failure' ? 'failed' : 'cancelled'
      await expect.poll(async () => (await management(page, `/api/conversations/${conversation.id}`)).messages.at(-1).turn?.status).toBe(expected)
      const message = (await management(page, `/api/conversations/${conversation.id}`)).messages.at(-1)
      expect(message.requests).toHaveLength(1)
      expect(message.requests[0].status).toBe(expected)
      expect(requestCount).toBe(1)
      expect(message.activity.filter((event: any) => event.kind === 'tool' && event.status === 'completed')).toHaveLength(0)
      if (scenario === 'receipt-recovery') expect(recovered).toBe(1)
      if (scenario !== 'before-first') expect(message.text).toContain('已经生成的内容应保留。')
      if (expected === 'cancelled') {
        await expect(assistant.locator('.message-label')).toContainText('已停止')
        expect(message.error).toBeFalsy()
        await expect(page.getByText('响应流提前结束，已保留收到的内容。', { exact: true })).toHaveCount(0)
      } else await expect(assistant.locator('.message-label')).toContainText('生成失败')
      if (scenario === 'partial') {
        await assistant.locator('.message-activity summary').click()
        await expect(assistant.getByText('已经生成的内容应保留。', { exact: true })).toBeVisible()
      }
      await page.screenshot({ path: path.join(directory, 'chat-terminal.png'), animations: 'disabled' })
      fs.writeFileSync(path.join(directory, 'result.json'), JSON.stringify({ scenario, status: message.turn.status, requestStatus: message.requests[0].status, requestCount, recovered, errors }, null, 2))
      expect(errors).toEqual([])
    } finally {
      response?.end()
      await app.close()
      await new Promise<void>(resolve => { server.closeAllConnections(); server.close(() => resolve()) })
    }
  })
}
