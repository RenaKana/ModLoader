import { chromium, expect, test } from '@playwright/test'
import * as fs from 'node:fs'
import * as path from 'node:path'
import type { ExternalHistorySnapshot, ExternalRequestRecord, LiveGeneration } from '../src/types'
import { localDate } from '../src/monitor-history-dates'
import { pinNavigation } from './navigation'

test('首次外部请求立即画自身采样，升级保留窗口与选择，历史输入显示未缓存量', async () => {
  test.setTimeout(90_000)
  const root = path.resolve(__dirname, '..')
  const artifacts = process.env.MODLOADER_TEST_ARTIFACTS || path.join(root, '.artifacts/codex/20261030-first-request-metrics/frontend')
  fs.mkdirSync(artifacts, { recursive: true })
  const directory = fs.mkdtempSync(path.join(artifacts, 'ui-'))
  const { preview } = await import('vite')
  const server = await preview({ root, configFile: false, build: { outDir: process.env.MODLOADER_TEST_DIST || path.join(root, '.artifacts/codex/20261030-first-request-metrics/dist') },
    preview: { host: '127.0.0.1', port: 0, strictPort: false } })
  const address = server.httpServer.address()
  if (!address || typeof address === 'string') throw new Error('Fixture preview did not open')
  const browser = await chromium.launch({ channel: 'msedge', headless: true })
  const page = await browser.newPage({ viewport: { width: 1440, height: 1040 } })
  await page.clock.install()
  const now = Date.now() / 1000, today = localDate()
  const instance = { id: 'active', alias: 'Qwen · external', backend: 'llama_cpp', model_id: 'm', profile_id: 'p', state: 'ready', active_requests: 0, capabilities: { input: ['text'] } }
  const record = (key: string, input: number, cached: number, output: number, started: number): ExternalRequestRecord => ({ request_key: key, run_id: 'current', instance_id: instance.id,
    alias: instance.alias, backend: instance.backend, revision: 1, aggregation: 'request', turn_id: null, turn_state: null, labels: [], date: today,
    status: 'completed', started_at: started, ended_at: started + 1, duration_s: 1, first_output_s: .1,
    usage: { prompt_tokens: input, prompt_tokens_details: { cached_tokens: cached }, completion_tokens: output }, timings: { predicted_n: output, predicted_ms: 1000 } })
  const first = record('first', 48267, 45738, 13, now - 2)
  first.status = 'running'; first.ended_at = undefined
  const records = [first]
  let current: ExternalHistorySnapshot = { run_id: 'current', aggregation: 'request', turn_id: null, state: 'running', requests: [first] }
  let live: Partial<LiveGeneration> = { request_key: 'first', phase: 'output', backend_running: 1, tokens_per_second: 150 }, clock = now
  const errors: string[] = [], unexpected: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/api/**', async route => {
    const url = new URL(route.request().url()), query = url.searchParams
    if (route.request().method() !== 'GET') { unexpected.push(`${route.request().method()} ${url.pathname}`); return route.abort() }
    if (url.pathname === '/api/state') return route.fulfill({ json: { models: [], profiles: [], runtimes: [], downloads: [], installations: [], instances: [instance], settings: {},
      backends: [{ id: 'llama_cpp', label: 'llama.cpp', parameters: [] }], hardware: { cpu_info: { name: 'fixture' }, gpus: [] } } })
    if (url.pathname === '/api/conversations') return route.fulfill({ json: { items: [], active_id: null } })
    if (url.pathname === '/api/chat-presets') return route.fulfill({ json: { items: [], default_id: null } })
    if (url.pathname.endsWith('/telemetry')) return route.fulfill({ json: { instance_id: instance.id, backend: instance.backend, state: 'ready', updated_at: clock,
      groups: [], unavailable: [], external_history: current, live_generation: { request_key: null, phase: 'idle', scope: 'request', source: 'fixture', backend_running: 0, backend_queued: 0,
        tokens_per_second: null, window_s: null, available_for_chat: false, reason: null, ...live } } })
    if (url.pathname === '/api/request-history/dates') return route.fulfill({ json: { current_run_id: 'current', today, issues: [], dates: [
      { date: today, request_count: records.length, instances: [{ run_id: 'current', id: instance.id, alias: instance.alias, backend: instance.backend, started_at: now - 100 }] },
    ] } })
    if (url.pathname === '/api/request-history') {
      let items = [...records]
      if (query.has('request_key')) items = items.filter(item => item.request_key === query.get('request_key'))
      if (query.has('turn_id')) items = items.filter(item => item.turn_id === query.get('turn_id'))
      items.sort((a, b) => query.has('turn_id') ? a.started_at! - b.started_at! : b.started_at! - a.started_at!)
      return route.fulfill({ json: { run_id: query.get('run_id'), items, total: items.length, offset: 0, limit: 50, issues: [] } })
    }
    unexpected.push(url.pathname)
    return route.fulfill({ status: 404, json: { detail: `Unexpected fixture ${url.pathname}` } })
  })
  const screenshot = (name: string) => page.screenshot({ path: path.join(directory, name), fullPage: true })
  try {
    await page.goto(`http://127.0.0.1:${address.port}`); await pinNavigation(page)
    await expect(page.getByRole('status', { name: '服务在线', exact: true })).toBeVisible()
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '运行监控', exact: true }).click()
    const monitor = page.locator('.monitor-page'), chart = monitor.locator('.monitor-generation-chart')
    await expect(monitor.getByRole('button', { name: '调用历史', exact: true })).toHaveCount(0)
    await expect(monitor.getByLabel('历史日期', { exact: true })).toHaveValue('today')
    await expect(monitor.locator('table.monitor-request-table')).toHaveCount(1)
    const refresh = async () => { await page.clock.runFor(1700) }
    const firstRow = monitor.locator('[data-request-key="first"]')
    await expect(chart).toBeVisible()
    await expect(chart).toHaveAttribute('data-mode', 'realtime')
    await expect(monitor.locator('.monitor-speed-current')).toHaveText('150.00token/s')
    await expect(chart.locator('g[data-series="current"] .trend-dot')).toHaveCount(1)
    await expect(monitor.getByRole('group', { name: '图表范围', exact: true })).toHaveCount(0)
    await expect(monitor).not.toContainText('本轮')
    await expect(firstRow.locator('[data-metric="input"] .monitor-request-readout')).toHaveText('2,529')
    await expect(firstRow.locator('[data-metric="cached"] .monitor-request-readout')).toHaveText('45,738')
    await expect(firstRow.locator('[data-metric="input"]')).toHaveAttribute('data-tooltip', '输入 2,529 Token · 总输入 48,267 Token')
    await expect(monitor.locator('[data-resource="context"] .monitor-context-value')).toHaveText('48,280token')
    await screenshot('first-request-sampled.png')
    await monitor.getByRole('button', { name: '60 秒', exact: true }).click()
    await firstRow.getByRole('button').click()
    live.tokens_per_second = 170; clock++
    await refresh(); await expect(chart.locator('g[data-series="current"] .trend-line')).toHaveCount(1)
    const trace = await chart.locator('g[data-series="current"]').innerHTML()
    await chart.evaluate(node => node.setAttribute('data-fixture-continuity', 'first-chart'))
    Object.assign(first, { aggregation: 'turn', turn_id: 'T', turn_state: 'waiting', status: 'completed', ended_at: now + 1, duration_s: 3, revision: 2 })
    current = { run_id: 'current', aggregation: 'turn', turn_id: 'T', state: 'waiting', requests: [first] }; live = {}; clock++
    await refresh()
    await expect(chart).toHaveAttribute('data-mode', 'realtime')
    await expect(chart).toHaveAttribute('data-fixture-continuity', 'first-chart')
    await expect(monitor.getByRole('button', { name: '60 秒', exact: true })).toHaveAttribute('aria-pressed', 'true')
    const parent = monitor.locator('tr[data-turn-id="T"]')
    if (await parent.getAttribute('data-expanded') !== 'true') await parent.locator('.monitor-turn-expand').click()
    await expect(firstRow).toHaveAttribute('data-selected', 'true')
    expect(await chart.locator('g[data-series="current"]').innerHTML()).toBe(trace)
    const second = record('second', 52630, 51322, 15, now + 2), third = record('third', 55350, 55279, 17, now + 4)
    for (const item of [second, third]) Object.assign(item, { aggregation: 'turn', turn_id: 'T', turn_state: 'waiting' })
    records.push(second, third); current.requests = [...records]; clock = now + 6
    await refresh()
    for (const [key, input, cached] of [['first', '2,529', '45,738'], ['second', '1,308', '51,322'], ['third', '71', '55,279']]) {
      const row = monitor.locator(`[data-request-key="${key}"]`)
      await expect(row.locator('[data-metric="input"] .monitor-request-readout')).toHaveText(input)
      await expect(row.locator('[data-metric="cached"] .monitor-request-readout')).toHaveText(cached)
    }
    const inputSummary = monitor.locator('.monitor-turn-summary > div').filter({ has: page.getByText('输入', { exact: true }) })
    await expect(inputSummary.locator('dd')).toHaveText('3,908token')
    await expect(inputSummary).toHaveAttribute('data-tooltip', '总输入 156,247 Token')
    await expect(firstRow).toHaveAttribute('data-selected', 'true')
    await screenshot('three-requests-net-input.png')
    const plain = record('plain-history', 55350, 55279, 17, now + 6); records.push(plain)
    current = { run_id: 'current', aggregation: 'request', turn_id: null, state: 'completed', requests: [plain] }; clock = now + 7
    await refresh(); await expect(monitor.locator('[data-request-key="plain-history"]')).toBeVisible()
    const history = monitor.getByRole('region', { name: '请求监控', exact: true }), table = history.getByRole('region', { name: '请求记录', exact: true })
    await expect(table.locator('tr[data-request-key]')).toHaveCount(4)
    await expect(table.locator('.monitor-generation-chart')).toHaveCount(0)
    await table.locator('[data-request-key="plain-history"] button').click()
    const chosen = history.getByRole('region', { name: '监控生成统计', exact: true })
    await expect(chosen.locator('.monitor-generation-chart')).toBeVisible()
    await expect(chosen.locator('.monitor-speed-current')).toHaveText('—')
    await expect(chosen.locator('.monitor-speed-note')).not.toContainText('最后采样')
    await expect(chosen.locator('g[data-series="current"] .trend-line')).toHaveCount(0)
    await expect(chosen.locator('.monitor-request-live [data-tooltip="总输入 55,350 Token"] dd')).toHaveText('71token')
    await expect(chosen.getByRole('group', { name: '图表范围', exact: true })).toHaveCount(0)
    await expect(chosen).not.toContainText('本轮')
    expect(await history.ariaSnapshot()).not.toMatch(/请求\s*#\d+/)
    await screenshot('ordinary-history-request.png')
    // A long enough real request crosses all three windows. The final .25 s
    // makes every left boundary fall between samples, not exactly on one.
    const long = record('window-request', 100, 0, 5, now + 10)
    long.status = 'running'; long.ended_at = undefined; long.usage = {}; long.timings = {}
    records.push(long)
    current = { run_id: 'current', aggregation: 'request', turn_id: null, state: 'running', requests: [long] }
    const origin = now + 10
    live = { request_key: long.request_key, phase: 'output', backend_running: 1, tokens_per_second: 140 }
    clock = origin
    await monitor.getByRole('button', { name: '刷新到最新', exact: true }).click()
    await refresh()
    for (let index = 1; index <= 65; index++) {
      clock = origin + index; live.tokens_per_second = 140 + index % 7
      await refresh()
      await expect(chosen.locator('.monitor-speed-current')).toHaveText(`${live.tokens_per_second}.00token/s`)
    }
    clock = origin + 65.25; live.tokens_per_second = 151
    await refresh(); await expect(chosen.locator('.monitor-speed-current')).toHaveText('151.00token/s')
    for (const seconds of [10, 30, 60]) {
      await monitor.getByRole('button', { name: `${seconds} 秒`, exact: true }).click()
      const trace = await chart.locator('.trend-line').getAttribute('d')
      expect(trace).toMatch(/^M0\.00,/)
      expect(trace).toMatch(/L600\.00,[\d.]+$/)
      await page.clock.runFor(200)
      await chart.locator('.monitor-window-indicator').evaluate(async node => { await Promise.all(node.getAnimations().map(animation => animation.finished)) })
      await chosen.locator('.monitor-speed-panel').screenshot({ path: path.join(directory, `window-${seconds}s-clipped.png`) })
    }
    await monitor.getByRole('button', { name: '10 秒', exact: true }).click()
    await chart.locator('svg').hover({ position: { x: 1, y: 75 } })
    await expect(chart.locator('.monitor-generation-tooltip')).toContainText(new Date((origin + 56) * 1000).toLocaleTimeString('zh-CN', { hour12: false }))
    expect(errors).toEqual([]); expect(unexpected).toEqual([])
    console.log(JSON.stringify({ screenshots: directory }))
  } catch (issue) {
    await screenshot('failure.png')
    console.log(JSON.stringify({ errors, unexpected }))
    throw issue
  } finally {
    await browser.close()
    await new Promise<void>((resolve, reject) => server.httpServer.close(error => error ? reject(error) : resolve()))
  }
})

