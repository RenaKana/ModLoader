import { test, expect, chromium } from '@playwright/test'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { emptyConversation, type Message } from '../src/chat-history'
import { turnTotals, type ModelRequestRecord } from '../src/chat-turn-metrics'
import { pinNavigation } from './navigation'
import { historyPage, historyUsage } from './monitor-history-fixture'

test('返回对话自动跟随最新监控：Spark 无阶段、有曲线、实例移除且无请求记录', async () => {
  test.setTimeout(60_000)
  const root = path.resolve(__dirname, '..')
  const artifacts = process.env.MODLOADER_TEST_ARTIFACTS || path.join(root, '.artifacts/codex/monitor-implementation-20261031/legacy-ui')
  fs.mkdirSync(artifacts, { recursive: true })
  const directory = fs.mkdtempSync(path.join(artifacts, 'follow-latest-ui-'))
  const { preview } = await import('vite')
  const server = await preview({ root, configFile: false,
    build: { outDir: process.env.MODLOADER_TEST_DIST || path.join(root, 'dist') },
    preview: { host: '127.0.0.1', port: 0, strictPort: false } })
  const address = server.httpServer.address()
  if (!address || typeof address === 'string') throw new Error('Fixture server did not open a port')
  const browser = await chromium.launch({ channel: 'msedge', headless: true })
  const page = await browser.newPage({ viewport: { width: 1440, height: 1040 } })
  const errors: string[] = [], unexpected: string[] = []
  const instance = { id: 'spark-instance', model_id: 'spark-model', profile_id: 'spark-profile',
    alias: 'SparkInfer-Qwen3.8-27B-NVFP4', backend: 'sparkinfer', state: 'ready', active_requests: 0, capabilities: { input: ['text'] } }
  const startedAt = Date.now() - 300_000
  const request: ModelRequestRecord = { id: 'historical-request', index: 1, kind: 'generation', status: 'completed',
    startedAt, elapsed: 256, firstOutput: 232, labels: ['图片输入'],
    usage: { prompt_tokens: 19138, completion_tokens: 1434, generation_ms: 254300, ttft_ms: 230903, decode_tps: 5.639 } }
  const message: Message = { id: 'historical-answer', role: 'assistant', text: '这是一条历史回复。', images: [],
    createdAt: startedAt, status: 'completed', model: instance.alias, profileId: instance.profile_id, modelId: instance.model_id,
    instanceId: instance.id, contextCapacity: 256000, requests: [request], turn: turnTotals([request], startedAt, 256, 'completed') }
  const conversation = emptyConversation({ profile_id: instance.profile_id, model_id: instance.model_id, alias: instance.alias })
  conversation.messages = [message]
  conversation.message_count = 1
  let removed = false
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/api/**', route => {
    const url = new URL(route.request().url()), pathname = url.pathname
    if (pathname.endsWith('/generation-events') && route.request().method() === 'GET') return route.fulfill({ contentType: 'text/event-stream', body: ': fixture idle\n\n' })
    if (route.request().method() !== 'GET') {
      unexpected.push(`${route.request().method()} ${pathname}`)
      return route.abort()
    }
    if (pathname === '/api/state') return route.fulfill({ json: { models: [], profiles: [], runtimes: [],
      instances: removed ? [] : [instance], settings: {}, backends: [{ id: 'sparkinfer', label: 'SparkInfer', parameters: [] }],
      hardware: { cpu_info: { name: 'AMD Ryzen 9 9950X3D' }, cpu_percent: 10,
        gpus: [{ id: 'gpu', name: 'NVIDIA GeForce RTX 5090', utilization: 0, memory_total: 32 * 1024 ** 3 }] } } })
    if (pathname === '/api/chat-presets') return route.fulfill({ json: { items: [], default_id: null } })
    if (pathname === '/api/conversations') return route.fulfill({ json: { items: [conversation], active_id: conversation.id } })
    if (pathname === `/api/conversations/${conversation.id}`) return route.fulfill({ json: conversation })
    if (pathname === '/api/request-history/dates') return route.fulfill({ json: { current_run_id: 'current', today: new Date().toLocaleDateString('en-CA'), dates: [], issues: [] } })
    if (pathname === '/api/request-history') return route.fulfill({ json: historyPage([], url.searchParams) })
    if (pathname === '/api/request-history/usage') return route.fulfill({ json: historyUsage() })
    if (pathname === `/api/instances/${instance.id}/telemetry`) return route.fulfill({ json: {
      instance_id: instance.id, backend: instance.backend, state: 'ready', updated_at: Date.now() / 1000, unavailable: [],
      live_generation: { request_key: null, source: '/metrics', backend_running: 0, backend_queued: 0,
        available_for_chat: false, tokens_per_second: null, window_s: null, reason: null },
      groups: [{ id: 'api_request', title: '最近完成请求', request_key: 'latest-request', request_status: 'completed', metrics: [
        { id: 'prompt_tokens', label: '输入', value: 77, kind: 'gauge' },
        { id: 'completion_tokens', label: '输出', value: 31, kind: 'gauge' },
        { id: 'duration_s', label: '耗时', value: 2, kind: 'gauge' },
      ] }],
    } })
    unexpected.push(pathname)
    return route.fulfill({ status: 404, json: { detail: `Unexpected fixture request: ${pathname}` } })
  })
  try {
    const monitor = page.locator('.monitor-page')
    const selection = monitor.locator('.monitor-compact-generation')
    const parentInput = monitor.locator('.monitor-ledger-group td[data-column="input"]')
    const navigation = page.getByRole('navigation', { name: '主导航' })
    const returnToLatest = async () => {
      await navigation.getByRole('button', { name: '测试对话', exact: true }).click()
      await expect(page.locator('.chat-section')).toBeVisible()
      await navigation.getByRole('button', { name: '运行监控', exact: true }).click()
      await expect(monitor.locator('.monitor-compact-generation[data-scope="turn"]')).toHaveCount(0)
      await expect(monitor.getByRole('button', { name: '返回最新', exact: true })).toHaveCount(0)
    }
    const openHistory = async (count: number) => {
      await page.goto(`http://127.0.0.1:${address.port}`)
      await pinNavigation(page)
      await expect(page.getByText('服务在线', { exact: true })).toBeVisible()
      await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '测试对话', exact: true }).click()
      await page.getByRole('button', { name: `查看本轮 ${count} 次模型请求统计`, exact: true }).click()
      await expect(monitor).toBeVisible()
      if (count) await expect(selection).toHaveAttribute('data-scope', 'turn')
      else await expect(monitor.locator('.monitor-generation-idle')).toHaveText('尚无模型请求')
    }
    await test.step('SparkInfer 无阶段速度时也能从历史返回最新请求', async () => {
      await openHistory(1)
      await expect(monitor.locator('.monitor-generation-chart .trend-line, .monitor-generation-chart .trend-dot')).toHaveCount(0)
      await expect(monitor.locator('.monitor-speed-current strong')).toHaveText('5.64')
      await expect(monitor.locator('.monitor-speed-note')).toHaveText('引擎平均 · 含首 Token 等待')
      await expect(parentInput).toHaveText('19,138')
      await monitor.screenshot({ path: path.join(directory, 'spark-history.png') })
      await returnToLatest()
      await expect(selection).toHaveAttribute('data-scope', 'request')
      await expect(monitor.locator('.monitor-request-group')).toHaveAttribute('data-selected-request-key', 'latest-request')
      await expect(parentInput).toHaveText('77')
      await expect(monitor.locator('.monitor-ledger-group td[data-column="time"]')).toHaveText('—')
      await monitor.getByLabel('历史日期', { exact: true }).selectOption('yesterday')
      await expect(monitor.locator('.monitor-ledger-group')).toHaveCount(0)
      await monitor.getByLabel('历史日期', { exact: true }).selectOption('today')
      await expect(parentInput).toHaveText('77')
      await monitor.screenshot({ path: path.join(directory, 'spark-latest.png') })
      // Returning to automatic following does not disable inspecting the same historical reply again.
      await navigation.getByRole('button', { name: '测试对话', exact: true }).click()
      await page.getByRole('button', { name: '查看本轮 1 次模型请求统计', exact: true }).click()
      await expect(selection).toHaveAttribute('data-scope', 'turn')
      await expect(parentInput).toHaveText('19,138')
      await returnToLatest()
    })
    await test.step('有曲线的历史在窄屏返回对话后同样恢复最新', async () => {
      request.timings = { predicted_n: 1434, predicted_ms: 23400, prompt_n: 19138, prompt_ms: 230900 }
      message.turn = turnTotals([request], startedAt, 256, 'completed')
      await page.setViewportSize({ width: 760, height: 1000 })
      await openHistory(1)
      await expect(monitor.locator('.monitor-generation-chart')).toHaveCount(1)
      await expect(monitor.locator('.monitor-request-point')).toHaveCount(1)
      await expect(monitor.locator('.monitor-speed-current strong')).toHaveText('61.28')
      await monitor.screenshot({ path: path.join(directory, 'charted-history-narrow.png') })
      await returnToLatest()
      await expect(parentInput).toHaveText('77')
    })
    await test.step('实例已移除、没有请求明细时仍可退出所选轮次', async () => {
      removed = true
      message.requests = []
      message.turn = turnTotals([], startedAt, 0, 'incomplete')
      message.status = 'interrupted'
      await page.setViewportSize({ width: 1440, height: 1040 })
      await openHistory(0)
      await expect(monitor.locator('.monitor-generation-chart')).toHaveCount(0)
      await expect(monitor.locator('table.monitor-request-table')).toHaveCount(1)
      await returnToLatest()
      await expect(monitor.locator('.monitor-selector-main')).toContainText('暂无运行模型')
    })
    expect(errors).toEqual([])
    expect(unexpected).toEqual([])
    console.log(JSON.stringify({ screenshots: directory }))
  } finally {
    await browser.close()
    await new Promise<void>((resolve, reject) => server.httpServer.close(error => error ? reject(error) : resolve()))
  }
})

