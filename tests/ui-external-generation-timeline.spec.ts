import { expect, test, chromium } from '@playwright/test'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { pinNavigation } from './navigation'
import { cycleWindow, historyPage, historyUsage } from './monitor-history-fixture'
import { externalRequestIdentity } from '../src/monitor-generation'
import type { ExternalRequestRecord, GenerationAccounting, GenerationEvent, LiveGeneration, Telemetry } from '../src/types'

test('外部连续双轴：轮询间短请求、真实间隔、状态缺口、历史固定和窄图命中', async () => {
  test.setTimeout(120_000)
  const root = path.resolve(__dirname, '..'), artifacts = process.env.MODLOADER_TEST_ARTIFACTS || path.join(root, '.artifacts/codex/20261102-external-dual-axis')
  const screenshots = path.join(artifacts, 'screenshots')
  fs.mkdirSync(screenshots, { recursive: true })
  const { preview } = await import('vite')
  const server = await preview({ root, configFile: false, build: { outDir: process.env.MODLOADER_TEST_DIST || path.join(artifacts, 'candidate-dist') }, preview: { host: '127.0.0.1', port: 0 } })
  const address = server.httpServer.address()
  if (!address || typeof address === 'string') throw new Error('Isolated preview did not open')
  const browser = await chromium.launch({ channel: 'msedge', headless: true })
  const page = await browser.newPage({ viewport: { width: 1366, height: 900 } })
  const origin = `http://127.0.0.1:${address.port}`, run = 'dual-run'
  let base = Date.now()
  await page.clock.install({ time: new Date(base) })
  const instance = { id: 'dual-instance', model_id: 'fixture-model', profile_id: 'fixture-profile', alias: '外部连续曲线 fixture', backend: 'strata', state: 'ready', active_requests: 1, capabilities: {} }
  const errors: string[] = [], unexpected: string[] = [], reads = { telemetry: 0 }
  const live = (key: string | null, phase: string, value: number | null): LiveGeneration => ({ request_key: key, scope: 'request', source: 'isolated fixture', phase,
    backend_running: key ? 1 : 0, backend_queued: 0, tokens_per_second: phase === 'output' ? value : null, prefill_tokens_per_second: phase === 'prefill' ? value : null,
    window_s: .2, available_for_chat: true, reason: null, context_capacity: 200000, prompt_tokens: 20000, generated_tokens: phase === 'output' ? 40 : 0 })
  const record = (key: string, start: number, end?: number): ExternalRequestRecord => ({ run_id: run, instance_id: instance.id, alias: instance.alias, backend: instance.backend,
    request_key: key, revision: 1, aggregation: 'request', turn_id: null, turn_state: null, labels: [], status: end === undefined ? 'running' : 'completed', started_at: start / 1000,
    ...(end === undefined ? {} : { ended_at: end / 1000 }), duration_s: end === undefined ? 0 : (end - start) / 1000, usage: {}, timings: {} })
  const a = record('request-a', base - 2000), records = [a]
  let activeRecords = [a], current = live(a.request_key, 'prefill', 15000), sampledAt = base, failed = false
  const accounting = (request: ExternalRequestRecord): GenerationAccounting => ({ request: request as GenerationAccounting['request'], turn: {
    source: 'external', key: request.request_key, turn_id: null, client_request_id: null, request_keys: [request.request_key], prior: null } })
  const emit = async (at: number, request?: ExternalRequestRecord, value: LiveGeneration | null = null, finalized: ExternalRequestRecord[] = []) => {
    const event: GenerationEvent = { instance_id: instance.id, run_id: run, observed_at: at / 1000, live_generation: value,
      accounting: request ? accounting(request) : null, finalized: finalized.map(accounting) }
    await page.evaluate(event => (window as any).emitDualEvent(event), event)
    await page.clock.runFor(20)
  }
  await page.addInitScript(() => {
    const original = window.fetch.bind(window), streams: ReadableStreamDefaultController<Uint8Array>[] = []
    ;(window as any).emitDualEvent = (event: unknown) => streams.forEach(stream => stream.enqueue(new TextEncoder().encode(`event: generation\ndata: ${JSON.stringify(event)}\n\n`)))
    window.fetch = async (input, options) => {
      if (String(input).endsWith('/generation-events')) return new Response(new ReadableStream<Uint8Array>({ start(controller) {
        streams.push(controller)
        options?.signal?.addEventListener('abort', () => { const index = streams.indexOf(controller); if (index >= 0) streams.splice(index, 1); controller.error(new DOMException('Aborted', 'AbortError')) }, { once: true })
      } }), { headers: { 'Content-Type': 'text/event-stream' } })
      return original(input, options)
    }
  })
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/*', route => {
    const request = route.request(), url = new URL(request.url())
    if (url.origin !== origin) { unexpected.push(request.url()); return route.abort() }
    if (!url.pathname.startsWith('/api/')) return route.continue()
    if (request.method() !== 'GET') { unexpected.push(`${request.method()} ${url.pathname}`); return route.abort() }
    if (url.pathname === '/api/state') return route.fulfill({ json: { models: [], profiles: [], runtimes: [], instances: [instance], settings: {}, backends: [{ id: 'strata', label: 'Strata', parameters: [] }], hardware: {} } })
    if (url.pathname === '/api/chat-presets') return route.fulfill({ json: { items: [], default_id: null } })
    if (url.pathname === '/api/conversations') return route.fulfill({ json: { items: [], active_id: null } })
    if (url.pathname === `/api/instances/${instance.id}/telemetry`) {
      reads.telemetry++
      if (failed) return route.fulfill({ status: 500, json: { detail: 'isolated telemetry outage' } })
      const payload: Telemetry = { instance_id: instance.id, backend: instance.backend, state: 'ready', updated_at: sampledAt / 1000, groups: [], unavailable: [], live_generation: current,
        external_history: { run_id: run, aggregation: 'request', turn_id: null, state: activeRecords.some(record => record.status === 'running') ? 'running' : 'completed', requests: activeRecords } }
      return route.fulfill({ json: payload })
    }
    const today = new Date(base).toLocaleDateString('en-CA')
    if (url.pathname === '/api/request-history/dates') return route.fulfill({ json: { current_run_id: run, today, dates: [{ date: today, request_count: records.length,
      instances: [{ source: 'external', run_id: run, id: instance.id, alias: instance.alias, backend: instance.backend }] }], issues: [] } })
    if (url.pathname === '/api/request-history') return route.fulfill({ json: historyPage(url.searchParams.has('request_key') ? records.filter(record => record.request_key === url.searchParams.get('request_key')) : records, url.searchParams) })
    if (url.pathname === '/api/request-history/usage') return route.fulfill({ json: historyUsage() })
    unexpected.push(url.pathname)
    return route.fulfill({ status: 404, json: { detail: `Unexpected isolated request: ${url.pathname}` } })
  })
  try {
    await page.goto(origin)
    await pinNavigation(page)
    await expect(page.getByText('服务在线', { exact: true })).toBeVisible()
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '运行监控', exact: true }).click()
    const panel = page.locator('.monitor-external-generation'), chart = page.locator('.monitor-dual-chart')
    const speed = (phase: 'decode' | 'prefill') => panel.locator(`.monitor-dual-readouts [data-phase="${phase}"] strong`)
    await expect(chart).toBeVisible()
    await expect(speed('prefill')).toHaveText('15,000.00')
    await expect(speed('decode')).toHaveText('—')
    await page.clock.pauseAt(new Date(await page.evaluate(() => Date.now()) + 100))
    base = await page.evaluate(() => Date.now())
    await emit(base + 110, a, live(a.request_key, 'prefill', 18000))
    await emit(base + 190, a, live(a.request_key, 'output', 100))
    await emit(base + 290, a, live(a.request_key, 'output', 120))
    await expect(chart.locator('[data-series="prefill"] .trend-line')).toHaveCount(1)
    await expect(chart.locator('[data-series="decode"] .trend-line')).toHaveCount(1)
    const finalA = { ...record(a.request_key, a.started_at! * 1000, base + 320), usage: { prompt_tokens: 20000, completion_tokens: 40 }, timings: { predicted_n: 40, predicted_ms: 300 } }
    records[0] = finalA
    await emit(base + 340, undefined, null, [finalA])
    const short = { ...record('short-between-polls', base + 350, base + 410), usage: { prompt_tokens: 20000, completion_tokens: 4 },
      timings: { prompt_n: 100, prompt_ms: 20, predicted_n: 4, predicted_ms: 40, cache_n: 19900 } }
    const noRate = record('no-rate-between-polls', base + 370, base + 420)
    records.push(short, noRate)
    const before = reads.telemetry
    await emit(base + 430, undefined, null, [short, noRate])
    await expect(chart.locator('.monitor-phase-average[data-request-key="short-between-polls"]')).toHaveCount(2)
    await expect(chart.locator('.monitor-timeline-request[data-request-key="no-rate-between-polls"]')).toHaveCount(1)
    expect(reads.telemetry).toBe(before)
    const average = chart.locator('.monitor-timeline-hit[data-kind="average"][data-phase="prefill"]')
    await average.focus()
    await expect(chart.getByRole('status')).toContainText('Prefill 阶段平均 · 5,000.00 token/s · 耗时 0.02 s')
    const averageBox = await average.boundingBox()
    expect(averageBox!.width).toBeGreaterThanOrEqual(20)
    const tooltip = await chart.getByRole('status').boundingBox(), plotBox = await chart.locator('svg').boundingBox()
    expect(tooltip!.x).toBeGreaterThanOrEqual(plotBox!.x - 1)
    expect(tooltip!.x + tooltip!.width).toBeLessThanOrEqual(plotBox!.x + plotBox!.width + 1)
    await panel.screenshot({ path: path.join(screenshots, 'dual-axis-1366-tooltip.png') })
    await average.blur()
    activeRecords = [short]; current = live(null, 'idle', null); sampledAt = base + 500
    await page.clock.runFor(500)
    await expect(panel).toHaveAttribute('data-state', 'idle')
    await expect(speed('decode')).toHaveText('0.00')
    await expect(speed('prefill')).toHaveText('0.00')
    const endBefore = Number(await chart.getAttribute('data-window-end'))
    sampledAt = base + 2000
    await page.clock.runFor(1500)
    expect(Number(await chart.getAttribute('data-window-end'))).toBeGreaterThan(endBefore + 1000)
    await expect(chart.locator('.monitor-idle-zero')).not.toHaveCount(0)
    const b = record('request-b', base + 2200)
    activeRecords = [b]; current = live(b.request_key, 'output', 90); sampledAt = base + 2300; records.push(b)
    await page.clock.runFor(500)
    await expect(speed('decode')).toHaveText('90.00')
    await emit(base + 2400, b, live(b.request_key, 'output', 110))
    await expect(chart.locator('[data-series="decode"] > g')).toHaveCount(2)
    await expect(chart.locator('[data-series="prefill"] > g')).toHaveCount(1)
    current = { ...live(b.request_key, 'queued', null), backend_running: 0, backend_queued: 1 }; sampledAt = base + 3000
    await page.clock.runFor(500)
    await expect(panel).toHaveAttribute('data-state', 'queued')
    await expect(speed('decode')).toHaveText('—')
    current = live(b.request_key, 'prefill', null); sampledAt = base + 3500
    await page.clock.runFor(500)
    await expect(speed('prefill')).toHaveText('—')
    failed = true
    await page.clock.runFor(500)
    await expect(panel).toHaveAttribute('data-state', 'unknown')
    await expect(speed('decode')).toHaveText('—')
    failed = false; current = live(b.request_key, 'output', 95); sampledAt = base + 4500
    await page.clock.runFor(500)
    await expect(speed('decode')).toHaveText('95.00')
    // A historical request stays single-request while new events continue in the background.
    const historyRow = page.locator(`tr[data-group-key=${JSON.stringify(externalRequestIdentity(finalA))}]`)
    await historyRow.locator('.monitor-ledger-group-select').click()
    await expect(page.locator('.monitor-generation-chart')).toHaveAttribute('data-mode', 'realtime')
    await expect(page.locator('.monitor-generation-chart')).toHaveAttribute('data-request-key', a.request_key)
    await emit(base + 4600, b, live(b.request_key, 'output', 130))
    await expect(page.locator('.monitor-generation-chart')).toHaveAttribute('data-request-key', a.request_key)
    await page.getByRole('button', { name: '刷新到最新', exact: true }).click()
    await expect(chart).toBeVisible()
    await expect(speed('decode')).toHaveText('130.00')
    const queueBand = chart.locator('.monitor-state-band[data-state="queued"]').first()
    await queueBand.focus()
    await expect(chart.getByRole('status')).toHaveText('排队中 · 0.5 s')
    const queueBox = await queueBand.boundingBox()
    await page.mouse.move(queueBox!.x + queueBox!.width / 2, queueBox!.y)
    await expect(chart.getByRole('status')).toHaveText('排队中 · 0.5 s')
    await queueBand.blur()
    for (const seconds of [10, 60, 30] as const) { await cycleWindow(panel, seconds); await expect(chart.locator('.trend-axis')).toHaveText(`${seconds} 秒前${seconds / 2} 秒前现在`) }
    await panel.screenshot({ path: path.join(screenshots, 'dual-axis-1366-continuous.png') })
    await page.setViewportSize({ width: 760, height: 900 })
    await expect(chart.locator('.monitor-dual-ticks.is-prefill')).toBeVisible()
    await average.focus()
    await expect(chart.getByRole('status')).toContainText('阶段平均')
    const dimensions = await chart.evaluate(node => ({ width: node.getBoundingClientRect().width, scroll: node.scrollWidth,
      left: node.querySelector('.is-decode')!.getBoundingClientRect().right, plot: node.querySelector('svg')!.getBoundingClientRect().left,
      right: node.querySelector('.is-prefill')!.getBoundingClientRect().left, plotRight: node.querySelector('svg')!.getBoundingClientRect().right }))
    expect(dimensions.scroll).toBeLessThanOrEqual(dimensions.width + 1)
    expect(dimensions.left).toBeLessThan(dimensions.plot)
    expect(dimensions.right).toBeGreaterThan(dimensions.plotRight)
    await panel.screenshot({ path: path.join(screenshots, 'dual-axis-760-tooltip.png') })
    expect(errors).toEqual([])
    expect(unexpected).toEqual([])
    console.log(JSON.stringify({ screenshots, reads }))
  } finally {
    await browser.close()
    await new Promise<void>((resolve, reject) => server.httpServer.close(error => error ? reject(error) : resolve()))
  }
})
