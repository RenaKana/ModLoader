import { pinNavigation } from './navigation'
import { disableWebSearch } from './chat-controls'
import { test, expect, _electron as electron } from '@playwright/test'
import { createServer, type ServerResponse } from 'node:http'
import * as path from 'node:path'
import * as fs from 'node:fs'

test('对话显示真实用量与思考，兼容流式、普通回复、停止和不完整输出', async () => {
  test.setTimeout(120_000)
  const root = path.resolve(__dirname, '..')
  const directory = fs.mkdtempSync(path.join(process.env.MODLOADER_TEST_ARTIFACT_DIR || path.join(root, 'work'), 'ui-chat-'))
  let response: ServerResponse | undefined
  let body: any
  let clientRequestId: string | undefined
  let liveRate = 100
  let exclusive = true
  const sampleTimes: number[] = []
  const server = createServer(async (request, result) => {
    result.setHeader('Access-Control-Allow-Origin', '*')
    result.setHeader('Access-Control-Allow-Headers', '*')
    if (request.method === 'OPTIONS') { result.end(); return }
    let raw = ''
    for await (const chunk of request) raw += chunk
    body = JSON.parse(raw)
    result.setHeader('Content-Type', body.stream ? 'text/event-stream' : 'application/json')
    result.flushHeaders()
    response = result
  })
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const address = server.address() as { port: number }
  const application = await electron.launch({ args: [root], env: { ...process.env, MODLOADER_DATA_DIR: directory }, timeout: 60_000 })
  const page = await application.firstWindow()
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  const assistant = page.locator('.chat-message.assistant').last()
  const input = page.getByRole('textbox', { name: '对话消息', includeHidden: true })
  const send = async () => {
    const preview = page.getByRole('region', { name: '输入内容预览', exact: true })
    if (await preview.isVisible()) await preview.click({ position: { x: 8, y: 8 } })
    response = undefined
    await input.fill('验证这条回复')
    await page.getByRole('button', { name: '发送消息', exact: true }).click()
    await expect.poll(() => Boolean(response)).toBe(true)
  }
  const event = (value: unknown) => response!.write(`data:${JSON.stringify(value)}\r\n\r\n`)
  const delta = (value: unknown) => event({ choices: [{ delta: value }] })
  const finish = async (usage?: object, timings?: object, reason = 'stop') => {
    event({ choices: [{ delta: {}, finish_reason: reason }] })
    if (usage || timings) event({ choices: [], usage, timings })
    response!.end('data: [DONE]\n\n')
    await expect(page.getByRole('button', { name: '停止生成', exact: true })).toHaveCount(0)
  }
  try {
    await pinNavigation(page)
    await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
    await page.route('**/api/state', async route => {
      const original = await route.fetch()
      const state = await original.json()
      await route.fulfill({ response: original, json: { ...state, instances: [{ id: 'chat-fixture', profile_id: 'chat-profile', model_id: 'chat-model', alias: '本地推理测试', backend: 'llama_cpp', state: 'ready', capabilities: { input: ['text'] }, active_requests: 0 }] } })
    })
    await page.route('**/api/chat', route => {
      clientRequestId = route.request().headers()['x-modloader-request-id']
      return route.continue({ url: `http://127.0.0.1:${address.port}/api/chat` })
    })
    await page.route('**/api/instances/chat-fixture/telemetry', route => {
      sampleTimes.push(Date.now())
      return route.fulfill({ json: { instance_id: 'chat-fixture', backend: 'llama_cpp', state: 'ready', updated_at: Date.now() / 1000, groups: [], unavailable: [], live_generation: {
        client_request_id: clientRequestId, tokens_per_second: liveRate, window_s: 2, scope: 'model', source: 'fixture live counter', available_for_chat: exclusive, reason: exclusive ? null : '并发请求无法归属此对话', backend_running: exclusive ? 1 : 2, backend_queued: 0,
      } } })
    })
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '测试对话', exact: true }).click()
    await disableWebSearch(page)
    await expect(page.getByText('开始对话', { exact: true })).toBeVisible()
    await expect(page.getByRole('textbox', { name: '对话消息', exact: true })).toHaveAttribute('placeholder', '输入消息…')
    await expect(page.locator('.composer-actions')).not.toContainText('本机自动保存')

    await test.step('思考实时可见，末尾 usage-only 块提供用量与后端解码速度', async () => {
      await send()
      expect(body.stream_options).toEqual({ include_usage: true })
      delta({ reasoning_content: '先检查题目中的条件。' })
      await expect(assistant.locator('.message-reasoning')).toContainText('先检查题目中的条件。')
      await expect(assistant.locator('.message-label')).toContainText('思考中')
      await expect(assistant.locator('.message-live-speed')).toHaveText('100.00tok/s')
      await expect(page.locator('.chat-toolbar .message-live-speed')).toBeVisible()
      liveRate = 160
      await expect(assistant.locator('.message-live-speed')).toContainText('160.00tok/s')
      await expect(page.locator('.chat-toolbar .message-live-speed')).toContainText('160.00tok/s')
      // The app monitor and active conversation each sample once per second.
      // Observe a steady round of both consumers, excluding their immediate startup reads.
      const observed = sampleTimes.length
      await expect.poll(() => sampleTimes.length).toBeGreaterThanOrEqual(observed + 3)
      const round = sampleTimes.at(-1)! - sampleTimes.at(-3)!
      expect(round).toBeGreaterThanOrEqual(500)
      expect(round).toBeLessThan(1800)
      exclusive = false
      await expect(assistant.locator('.message-live-speed')).toHaveText('—tok/s')
      await expect(assistant.locator('.message-live-speed')).toHaveAttribute('data-tooltip', '并发请求无法归属此对话')
      exclusive = true
      delta({ reasoning: '然后计算。' })
      delta({ content: '答案是 42。' })
      await expect(assistant.locator('.message-reasoning > div')).toBeHidden()
      await finish({ prompt_tokens: 16, completion_tokens: 25 }, { predicted_per_second: 12.5 })
      await expect(assistant.locator('.message-text')).toHaveText('答案是 42。')
      await expect(assistant.locator('[data-metric="input"]')).toHaveAccessibleName(/^(本轮)?未缓存输入 未知 Token$/)
      await expect(assistant.locator('[data-metric="cached"]')).toHaveAccessibleName(/^(本轮)?缓存命中 未知 Token$/)
      await expect(assistant.locator('[data-metric="output"]')).toHaveAccessibleName('输出 25 Token')
      await expect(assistant.locator('.message-footer')).toContainText('DEC12.50tok/s')
      await expect(assistant.locator('.message-footer')).toContainText('TTFT')
      await expect(assistant.locator('.message-reasoning > div')).toBeHidden()
      await assistant.locator('.message-reasoning > summary').click()
      await expect(assistant.locator('.message-reasoning > div')).toBeVisible()
      await page.screenshot({ path: path.join(directory, 'chat-reasoning-metrics.png') })
    })
    await test.step('跨块 think 标签与正文分离，历史独立回传正文与完整思考', async () => {
      await send()
      expect(body.messages.find((message: any) => message.role === 'assistant').content).toBe('答案是 42。')
      expect(body.messages.find((message: any) => message.role === 'assistant').reasoning_content).toBe('先检查题目中的条件。然后计算。')
      delta({ content: '<thi' })
      await expect(assistant.locator('.message-text')).toHaveCount(0)
      delta({ content: 'nk>先思考</thi' })
      await expect(assistant.locator('.message-reasoning > div')).toHaveText('先思考')
      delta({ content: 'nk>这是最终回答。' })
      await finish({ completion_tokens: 10 })
      await expect(assistant.locator('.message-text')).toHaveText('这是最终回答。')
      await expect(assistant.locator('.message-footer')).toContainText('E2E')
    })
    await test.step('非流式 reasoning 字段，以及没有用量时不伪造速度', async () => {
      await page.getByRole('button', { name: '对话参数', exact: true }).click()
      await page.getByLabel('最大输出 Token', { exact: true }).fill('-3')
      await input.fill('不能发出无效参数')
      response = undefined
      await page.getByRole('button', { name: '发送消息', exact: true }).click()
      await expect(page.getByLabel('最大输出 Token', { exact: true })).toHaveAttribute('aria-invalid', 'true')
      expect(response).toBeUndefined()
      await expect(input).toHaveValue('不能发出无效参数')
      await page.getByLabel('最大输出 Token', { exact: true }).fill('64')
      await page.getByLabel('温度', { exact: true }).fill('0.7')
      await expect(page.locator('.chat-section .error-notice')).toHaveCount(0)
      await page.screenshot({ path: path.join(directory, 'chat-numeric-parameters.png') })
      await page.getByRole('checkbox', { name: /流式输出/ }).uncheck()
      await send()
      expect(body.stream_options).toBeUndefined()
      expect(body.max_tokens).toBe(64)
      expect(body.temperature).toBe(.7)
      response!.end(JSON.stringify({ choices: [{ message: { reasoning_content: '这是非流式思考。', content: '<think>这是非流式思考。</think>普通回答。' }, finish_reason: 'stop' }], usage: { prompt_tokens: 12, completion_tokens: 8 } }))
      await expect(assistant.locator('.message-reasoning > div')).toHaveText('这是非流式思考。')
      await expect(assistant.locator('.message-reasoning > div')).toBeHidden()
      await expect(assistant.locator('.message-footer')).toContainText('E2E')
      await expect(assistant.locator('.message-footer')).not.toContainText('TTFT')
      await send()
      response!.end(JSON.stringify({ choices: [{ message: { content: '无用量的回答。' }, finish_reason: 'stop' }] }))
      await expect(assistant.locator('[data-metric="speed"]')).toHaveAccessibleName('Token 速度不可用')
      await expect(assistant.locator('[data-metric="speed"]')).toHaveText('—tok/s')
      await page.getByRole('checkbox', { name: /流式输出/ }).check()
      await page.getByRole('button', { name: '收起对话参数', exact: true }).click()
    })
    await test.step('长思考区内部滚动、结束自动收起，正文按 Markdown 渲染', async () => {
      await send()
      const longReasoning = Array.from({ length: 40 }, (_, index) => `分析第 ${index + 1} 步：保留完整推理内容。`).join('\n\n')
      delta({ reasoning_content: longReasoning })
      const region = assistant.getByRole('region', { name: '思考内容', includeHidden: true })
      await expect(region).toBeVisible()
      await expect.poll(() => region.evaluate(element => element.scrollHeight > element.clientHeight)).toBe(true)
      await expect.poll(() => region.evaluate(element => element.clientHeight <= innerHeight * .35 + 1)).toBe(true)
      await expect.poll(() => region.evaluate(element => element.scrollHeight - element.scrollTop - element.clientHeight)).toBeLessThanOrEqual(1)
      for (let index = 0; index < 30; index++) {
        delta({ reasoning_content: `\n\n连续流第 ${index + 1} 段：继续分析。` })
        await new Promise(resolve => setTimeout(resolve, 10))
      }
      await expect(region).toContainText('连续流第 30 段')
      await expect.poll(() => region.evaluate(element => element.scrollHeight - element.scrollTop - element.clientHeight)).toBeLessThanOrEqual(1)
      await region.hover()
      await page.mouse.wheel(0, -await region.evaluate(element => element.scrollHeight))
      await expect.poll(() => region.evaluate(element => element.scrollTop)).toBe(0)
      delta({ reasoning_content: '\n\n用户向上查看时不跳回底部。' })
      await expect(region).toContainText('用户向上查看时不跳回底部。')
      await expect.poll(() => region.evaluate(element => element.scrollTop)).toBe(0)
      await region.press('End')
      await expect.poll(() => region.evaluate(element => element.scrollHeight - element.scrollTop - element.clientHeight)).toBeLessThanOrEqual(1)
      delta({ reasoning_content: '\n\n回到底部后恢复自动跟随。' })
      await expect(region).toContainText('回到底部后恢复自动跟随。')
      await expect.poll(() => region.evaluate(element => element.scrollHeight - element.scrollTop - element.clientHeight)).toBeLessThanOrEqual(1)
      await assistant.locator('.message-reasoning > summary').click()
      await expect(region).toBeHidden()
      delta({ reasoning_content: '\n\n收起期间继续接收思考。' })
      await expect(region).toContainText('收起期间继续接收思考。')
      await assistant.locator('.message-reasoning > summary').click()
      await expect(region).toBeVisible()
      await expect.poll(() => region.evaluate(element => element.scrollHeight - element.scrollTop - element.clientHeight)).toBeLessThanOrEqual(1)
      await region.press('Home')
      await expect.poll(() => region.evaluate(element => element.scrollTop)).toBe(0)
      await assistant.locator('.message-reasoning > summary').click()
      await expect(region).toBeHidden()
      delta({ reasoning_content: '\n\n暂停跟随的状态在收起期间保留。' })
      await expect(region).toContainText('暂停跟随的状态在收起期间保留。')
      await assistant.locator('.message-reasoning > summary').click()
      await expect(region).toBeVisible()
      await expect.poll(() => region.evaluate(element => element.scrollTop)).toBe(0)
      const markdown = '## 校验结果\n\n这是 **加粗** 和 *强调*。\n\n- 第一项\n- 第二项\n\n```python\nprint("hello")\n```\n\n| 名称 | 数值 |\n| --- | --- |\n| 速度 | 123 |\n\n[正常链接](https://example.com)\n\n[危险链接](javascript:alert(1))\n\n<img src="https://example.com/pixel" onerror="alert(1)">'
      delta({ content: markdown })
      await expect(region).toBeHidden()
      await finish({ completion_tokens: 100 })
      await expect(assistant.getByRole('heading', { name: '校验结果' })).toBeVisible()
      await expect(assistant.locator('strong').filter({ hasText: /^加粗$/ })).toBeVisible()
      await expect(assistant.locator('.message-text li')).toHaveCount(2)
      await expect(assistant.locator('.message-code-block[data-language="python"] pre code')).toHaveText('print("hello")\n')
      await expect(assistant.getByRole('table')).toBeVisible()
      await expect(assistant.getByRole('link', { name: '正常链接' })).toHaveAttribute('href', 'https://example.com/')
      await expect(assistant.locator('a[href^="javascript:"], img, script')).toHaveCount(0)
      await expect(assistant.locator('iframe')).toHaveAttribute('sandbox', '')
      await assistant.locator('.message-reasoning > summary').click()
      await expect(region).toBeVisible()
      await expect(region).toContainText('分析第 40 步')
      await assistant.scrollIntoViewIfNeeded()
      await page.screenshot({ path: path.join(directory, 'chat-markdown-reasoning-scroll.png') })
    })
    await test.step('思考中的延迟图片扩展高度后继续跟随底部', async () => {
      let releaseImage!: () => void
      const imageReady = new Promise<void>(resolve => { releaseImage = resolve })
      let imageRequested = false
      await page.route('http://reasoning-fixture.invalid/delayed.png', async route => {
        imageRequested = true
        await imageReady
        await route.fulfill({ contentType: 'image/svg+xml', body: '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="720"><rect width="400" height="720" fill="#e4eaf3"/></svg>' })
      })
      await send()
      const region = assistant.getByRole('region', { name: '思考内容' })
      delta({ reasoning_content: Array.from({ length: 24 }, (_, index) => `检查第 ${index + 1} 项。`).join('\n\n') + '\n\n![延迟示意图](http://reasoning-fixture.invalid/delayed.png)\n\n图片后的思考。' })
      await expect(region).toContainText('图片后的思考。')
      await expect.poll(() => imageRequested).toBe(true)
      await expect.poll(() => region.evaluate(element => element.scrollHeight - element.scrollTop - element.clientHeight)).toBeLessThanOrEqual(1)
      releaseImage()
      await expect.poll(() => region.locator('img').evaluate(image => image.naturalHeight)).toBe(720)
      await expect.poll(() => region.evaluate(element => element.scrollHeight - element.scrollTop - element.clientHeight)).toBeLessThanOrEqual(1)
      await page.screenshot({ path: path.join(directory, 'chat-reasoning-delayed-image.png') })
      delta({ reasoning_content: '\n\n图片加载后继续流入的思考。' })
      await expect(region).toContainText('图片加载后继续流入的思考。')
      await expect.poll(() => region.locator('img').evaluate(image => image.naturalHeight)).toBe(720)
      await expect.poll(() => region.evaluate(element => element.scrollHeight - element.scrollTop - element.clientHeight)).toBeLessThanOrEqual(1)
      await region.hover()
      await page.mouse.wheel(0, -100)
      await expect.poll(() => region.evaluate(element => element.scrollHeight - element.scrollTop - element.clientHeight)).toBeGreaterThan(24)
      delta({ reasoning_content: '\n\n只上翻一小段时也保留暂停跟随。' })
      await expect(region).toContainText('只上翻一小段时也保留暂停跟随。')
      await expect.poll(() => region.locator('img').evaluate(image => image.naturalHeight)).toBe(720)
      await page.screenshot({ path: path.join(directory, 'chat-reasoning-paused-image.png') })
      await expect.poll(() => region.evaluate(element => element.scrollHeight - element.scrollTop - element.clientHeight)).toBeGreaterThan(24)
      await region.press('End')
      await expect.poll(() => region.evaluate(element => element.scrollHeight - element.scrollTop - element.clientHeight)).toBeLessThanOrEqual(1)
      delta({ content: '延迟图片校验完成。' })
      await finish()
    })
    await test.step('长 Markdown 思考持续流入时，保留每秒速度刷新和连续显示', async () => {
      await send()
      const longReasoning = Array.from({ length: 1200 }, (_, index) => `分析第 ${index + 1} 步：这是持续增长的 **Markdown** 思考，检查多个条件并保留完整推理。\n\n`).join('')
      delta({ reasoning_content: longReasoning })
      const region = assistant.getByRole('region', { name: '思考内容' })
      await expect(region).toContainText('分析第 1200 步')
      await region.evaluate(element => {
        const sample = { updates: [] as number[], observer: new MutationObserver(() => sample.updates.push(performance.now())) }
        sample.observer.observe(element, { subtree: true, childList: true, characterData: true })
        ;(window as any).chatStreamSample = sample
      })
      const initialSamples = sampleTimes.length
      for (let index = 0; index < 220; index++) {
        delta({ reasoning_content: `继续检查${index}。` })
        await new Promise(resolve => setTimeout(resolve, 10))
      }
      await expect(region).toContainText('继续检查219。')
      const updates = await page.evaluate(() => {
        const sample = (window as any).chatStreamSample
        sample.observer.disconnect()
        delete (window as any).chatStreamSample
        return sample.updates as number[]
      })
      expect(updates.length).toBeGreaterThan(10)
      expect(Math.max(...updates.slice(1).map((now, index) => now - updates[index]))).toBeLessThan(180)
      expect(sampleTimes.length - initialSamples).toBeGreaterThanOrEqual(2)
      await expect(region).toBeVisible()
      await expect(assistant.locator('.message-live-speed')).toContainText('160.00tok/s')
      delta({ content: '### 持续流校验完成\n\n最终回答保留 **Markdown**。' })
      await finish({ completion_tokens: 13000 })
      await expect(region).toBeHidden()
      await expect(assistant.getByRole('heading', { name: '持续流校验完成' })).toBeVisible()
      await assistant.locator('.message-reasoning > summary').click()
      await expect(region).toContainText('继续检查219。')
      await expect(region).toContainText('分析第 1200 步')
    })
    await test.step('只产生思考达到上限、用户停止、断流均保留已收内容', async () => {
      await send()
      delta({ reasoning_content: '还在推理，输出预算已耗尽。' })
      await finish({ completion_tokens: 32 }, undefined, 'length')
      await expect(assistant.locator('.message-limit')).toContainText('模型尚未生成最终回答')
      await expect(assistant.locator('.message-error')).toHaveCount(0)
      await send()
      delta({ reasoning_content: '可停止的思考。' })
      await expect(assistant.locator('.message-reasoning')).toContainText('可停止的思考。')
      await page.getByRole('button', { name: '停止生成', exact: true }).click()
      response!.end()
      await expect(assistant.locator('.message-label')).toContainText('已停止')
      await expect(assistant.locator('[data-metric="speed"]')).toHaveCount(0)
      await send()
      delta({ content: '断流前收到的部分。' })
      response!.end()
      await expect(assistant.locator('.message-error')).toContainText('响应流提前结束')
      await expect(assistant.locator('.message-text')).toHaveText('断流前收到的部分。')
      await expect(assistant.locator('[data-metric="speed"]')).toHaveCount(0)
    })
    // Keep the original long-stream workload unchanged by adding tool DOM only afterwards.
    await test.step('未声明工具能力不阻止尝试，工具 JSON 与大数值原样发送', async () => {
      await page.getByRole('button', { name: '对话参数', exact: true }).click()
      const tools = page.getByRole('textbox', { name: /^工具定义/ })
      await expect(tools).toBeEnabled()
      await tools.fill('{}')
      await input.fill('保留工具定义输入校验')
      response = undefined
      await page.getByRole('button', { name: '发送消息', exact: true }).click()
      await expect(page.locator('.chat-section .error-notice')).toContainText('工具定义须为 JSON 数组')
      expect(response).toBeUndefined()
      const definitions = [{ type: 'function', function: { name: 'local_lookup', parameters: { type: 'object', properties: {} } } }]
      await tools.fill(JSON.stringify(definitions))
      await page.getByLabel('最大输出 Token', { exact: true }).fill('1048576')
      await page.getByLabel('温度', { exact: true }).fill('4')
      await page.screenshot({ path: path.join(directory, 'chat-tool-parameters.png') })
      await send()
      expect(body.tools).toEqual(definitions)
      expect(body.max_tokens).toBe(1048576)
      expect(body.temperature).toBe(4)
      delta({ tool_calls: [{ index: 0, id: 'lookup-request', type: 'function', function: { name: 'local_lookup', arguments: '{}' } }] })
      await finish(undefined, undefined, 'tool_calls')
      await expect(assistant.locator('.tool-call')).toContainText('工具调用请求 · local_lookup')
      await expect(assistant.locator('.tool-call pre')).toHaveText('{}')
      await expect(assistant.locator('.tool-call small')).toHaveCount(0)
      await page.screenshot({ path: path.join(directory, 'chat-tool-request.png') })
      await tools.fill('')
      await page.getByLabel('最大输出 Token', { exact: true }).fill('64')
      await page.getByLabel('温度', { exact: true }).fill('0.7')
      await page.getByRole('button', { name: '收起对话参数', exact: true }).click()
    })
    expect(errors).toEqual([])
  } finally {
    response?.end()
    await page.unrouteAll({ behavior: 'wait' })
    await application.close()
    server.closeAllConnections()
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
})
