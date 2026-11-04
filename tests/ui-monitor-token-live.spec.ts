import { expect, test, chromium, type Page } from '@playwright/test'
import { build } from 'esbuild'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { historyPage, historyStats, historyUsage } from './monitor-history-fixture'
import { externalRequestIdentity as fixtureGroupKey } from '../src/monitor-generation'
import type { ExternalRequestRecord, GenerationAccounting, Telemetry } from '../src/types'

const root = path.resolve(__dirname, '..')
const artifacts = process.env.MODLOADER_TEST_ARTIFACTS || path.join(root, '.artifacts/codex/20261102-external-dual-axis/tests/token-live')
const now = Date.parse('2026-10-31T12:00:00Z')
const instance = { id: 'token-instance', alias: 'API流式模型', backend: 'ninfer', state: 'ready', active_requests: 1 }
const external = (key: string, running: boolean, start: number): ExternalRequestRecord => ({ run_id: 'token-run', instance_id: instance.id, alias: instance.alias,
  backend: instance.backend, request_key: key, revision: 1, aggregation: 'turn', turn_id: `${key}-turn`, turn_state: running ? 'running' : 'completed', status: running ? 'running' : 'completed',
  labels: [], started_at: start / 1000, ...(running ? {} : { ended_at: (start + 5000) / 1000 }), duration_s: 5, usage: { prompt_tokens: 1000, completion_tokens: 100 },
  timings: { predicted_n: 100, predicted_ms: 2500, prompt_n: 500, prompt_ms: 100 } })
const current = external('current-api-call', true, now - 5000)
const history = external('historical-api-call', false, now - 60000)
const mainSpeed = (page: Page) => page.locator('.monitor-external-generation .monitor-dual-readouts [data-phase="decode"] strong, .monitor-compact-generation:not(.monitor-external-generation) .monitor-speed-current strong')
const prefillSpeed = (page: Page) => page.locator('.monitor-external-generation .monitor-dual-readouts [data-phase="prefill"] strong')
const nativeAccounting = (tokens: number, seconds: number, prompt = 1100, prefill = 750): GenerationAccounting => ({
  request: { ...current, streaming: true, counters: { input: prompt, output: tokens, cached: 100 },
    usage: { prompt_tokens: 1000, completion_tokens: 100, prompt_tokens_details: { cached_tokens: 100 }, ttft_ms: 250 },
    timings: { predicted_n: tokens, predicted_ms: seconds * 1000, predicted_per_second: tokens / seconds, prompt_n: prefill, prompt_ms: 125, prompt_per_second: prefill / .125 },
    measurements: { decode: { tokens, seconds, value: tokens / seconds, basis: 'backend_decode', reported: 1, coverage: 'complete' },
      prefill: { tokens: prefill, seconds: .125, value: prefill / .125, basis: 'backend_prefill', reported: 1, coverage: 'complete' } } },
  turn: { source: 'external', key: `turn:${current.turn_id}`, turn_id: current.turn_id, client_request_id: null,
    request_keys: ['earlier-api-call', current.request_key], prior: historyStats([{ ...external('earlier-api-call', false, now - 15000), turn_id: current.turn_id }]) },
})

async function fixture(withSibling = false, internal = false, noCurrent = false) {
  const sibling = { ...external('earlier-api-call', false, now - 15000), turn_id: current.turn_id }
  const activeRecords = withSibling ? [sibling, current] : [current]
  const telemetryRecords = [...activeRecords]
  fs.mkdirSync(artifacts, { recursive: true })
  const directory = fs.mkdtempSync(path.join(artifacts, 'token-live-'))
  const source = (file: string) => JSON.stringify(path.join(root, file).replaceAll('\\', '/'))
  fs.writeFileSync(path.join(directory, 'fixture.tsx'), `import React from'react';import{createRoot}from'react-dom/client';import{useMonitorState}from ${source('src/useMonitorState')};import{MonitorRequestHistory}from ${source('src/MonitorRequestHistory')};import{miniSnapshot,useMiniMonitorPublisher}from ${source('src/mini-monitor')};import{generationEventIdentity}from ${source('src/monitor-generation-events')};import ${source('src/styles.css')};import ${source('src/resource-cards.css')};
    const state={models:[],profiles:[],runtimes:[],instances:[${JSON.stringify(instance)}],downloads:[],installations:[],settings:{},backends:[],hardware:{}};
    function Fixture(){window.fixtureRenders=(window.fixtureRenders||0)+1;const monitor=useMonitorState(state,async()=>{});window.tokenMonitor=monitor;useMiniMonitorPublisher(miniSnapshot(state,monitor,true),monitor.generationEvents,generationEventIdentity(monitor.liveGeneration.requests.at(-1),monitor.generationEvents),monitor.liveGeneration.requests.at(-1));return <main className="monitor-page" style={{padding:24,'--font-scale':1}}><MonitorRequestHistory monitor={monitor} active/></main>};createRoot(document.getElementById('root')).render(<Fixture/>);`)
  await build({ entryPoints: [path.join(directory, 'fixture.tsx')], outfile: path.join(directory, 'fixture.js'), bundle: true, format: 'iife', jsx: 'automatic', loader: { '.woff2': 'file' }, define: { 'process.env.NODE_ENV': '"development"' } })
  const browser = await chromium.launch({ channel: 'msedge', headless: true })
  const page = await browser.newPage({ viewport: { width: 1366, height: 900 } })
  await page.clock.install({ time: new Date(now) })
  await page.addInitScript(() => {
    const original = window.fetch.bind(window), streams: ReadableStreamDefaultController<Uint8Array>[] = []
    ;(window as any).generationConnections = []
    ;(window as any).emitGeneration = (event: unknown) => {
      const bytes = new TextEncoder().encode(`event: generation\r\ndata: ${JSON.stringify({ accounting: null, finalized: [], ...(event as object) })}\r\n\r\n`)
      streams.forEach(stream => { stream.enqueue(bytes.slice(0, 17)); stream.enqueue(bytes.slice(17)) })
    }
    ;(window as any).endGeneration = () => { streams.splice(0).forEach(stream => stream.close()) }
    ;(window as any).miniSnapshots = []
    ;(window as any).modloader = { connection: async () => ({ baseUrl: 'http://token.fixture', token: 'fixture-token' }), publishMiniMonitor: (snapshot: unknown) => (window as any).miniSnapshots.push(snapshot) }
    window.fetch = async (input, options) => {
      if (String(input).endsWith('/generation-events')) {
        ;(window as any).generationConnections.push({ url: String(input), authorization: new Headers(options?.headers).get('Authorization') })
        return new Response(new ReadableStream<Uint8Array>({ start(controller) {
          streams.push(controller)
          options?.signal?.addEventListener('abort', () => { const index = streams.indexOf(controller); if (index >= 0) { streams.splice(index, 1); controller.error(new DOMException('Aborted', 'AbortError')) } }, { once: true })
        } }), { headers: { 'Content-Type': 'text/event-stream' } })
      }
      return original(input, options)
    }
  })
  const errors: string[] = [], reads = { telemetry: 0, list: 0, detail: 0, usage: 0 }
  page.on('pageerror', error => errors.push(error.message))
  let speed: number | null = noCurrent ? null : 11.11
  const telemetry = (): Telemetry => ({ instance_id: instance.id, backend: instance.backend, state: 'ready', updated_at: now / 1000 + reads.telemetry / 2,
    groups: [], unavailable: [], external_history: internal ? undefined : { run_id: current.run_id, aggregation: 'turn', turn_id: current.turn_id,
      state: telemetryRecords.some(record => record.status === 'running') ? 'running' : 'completed', requests: telemetryRecords },
    live_generation: telemetryRecords.every(record => record.status !== 'running') ? null : { request_key: internal ? 'previous-server-call' : current.request_key, ...(internal ? { client_request_id: 'previous-turn', client_call_id: 'previous-call' } : {}), phase: 'output', scope: 'request', source: 'SSE累计计数滚动增量', tokens_per_second: speed,
      average_tokens_per_second: 40, average_basis: 'backend_decode', window_s: 3, backend_running: 1, backend_queued: 0, available_for_chat: true, reason: null } })
  await page.route('http://token.fixture/**', async route => {
    const url = new URL(route.request().url())
    if (route.request().method() !== 'GET') { errors.push(`Unexpected write: ${route.request().method()} ${url.pathname}`); return route.abort() }
    if (url.pathname.endsWith('/telemetry')) { reads.telemetry++; return route.fulfill({ json: telemetry() }) }
    if (url.pathname === '/api/request-history/dates') return route.fulfill({ json: { current_run_id: current.run_id, today: '2026-10-31', usage_revision: 'unchanged', dates: [{ date: '2026-10-31', request_count: 2, instances: [{ source: 'external', run_id: current.run_id, id: instance.id, alias: instance.alias, backend: instance.backend }] }], issues: [] } })
    if (url.pathname === '/api/request-history/usage') { reads.usage++; return route.fulfill({ json: historyUsage() }) }
    if (url.pathname === '/api/request-history') {
      if (url.searchParams.has('turn_id')) reads.detail++; else reads.list++
      const records = url.searchParams.has('turn_id') ? [...activeRecords, history].filter(record => record.turn_id === url.searchParams.get('turn_id')) : [...activeRecords, history]
      return route.fulfill({ json: historyPage(records, url.searchParams) })
    }
    if (url.pathname === '/') return route.fulfill({ contentType: 'text/html', body: '<!doctype html><meta charset="utf-8"><link rel="stylesheet" href="fixture.css"><div id="root"></div><script src="fixture.js"></script>' })
    return route.fulfill({ path: path.join(directory, path.basename(url.pathname)) })
  })
  await page.goto('http://token.fixture/')
  await expect(mainSpeed(page)).toHaveText(noCurrent ? '—' : '11.11')
  await expect(page.locator('.monitor-request-table .monitor-ledger-group')).toHaveCount(internal || withSibling ? 3 : 2)
  await page.clock.pauseAt(new Date(now + 1000))
  await expect.poll(async () => await page.evaluate(() => (window as any).tokenMonitor.payload?.updated_at) === now / 1000 + reads.telemetry / 2).toBe(true)
  return { page, browser, directory, errors, reads, speed: (value: number) => { speed = value },
    finish: (request: GenerationAccounting['request'], includeTelemetry = true) => {
      const record: ExternalRequestRecord = { ...current, ...request, revision: 2, turn_state: 'completed', ended_at: now / 1000 + 1, first_output_s: .25 }
      activeRecords[activeRecords.length - 1] = record
      if (includeTelemetry) telemetryRecords[telemetryRecords.length - 1] = record
    } }
}

test('NInfer逐Token事件只更新双轴leaf，不增加旧单请求曲线点/用量读取或整树render', async () => {
  const f = await fixture()
  try {
    const baseline = await f.page.evaluate(() => ({ renders: (window as any).fixtureRenders, points: (window as any).tokenMonitor.generation.requests[0].phases.decode.points.length,
      observed: (window as any).tokenMonitor.payload.updated_at }))
    const reads = { ...f.reads }
    for (const [index, speed] of [22.22, 33.33, 44.44].entries()) {
      await f.page.evaluate(event => (window as any).emitGeneration(event), { run_id: current.run_id, instance_id: instance.id, observed_at: baseline.observed + (index + 1) / 10,
        live_generation: { request_key: current.request_key, phase: 'output', scope: 'request', tokens_per_second: speed, window_s: 3, backend_running: 1, backend_queued: 0 } })
      await f.page.clock.runFor(20)
      await expect(mainSpeed(f.page)).toHaveText(speed.toFixed(2))
      await expect(f.page.locator('.monitor-external-generation')).toHaveAttribute('data-state', 'decode')
      expect(await f.page.evaluate(() => (window as any).miniSnapshots.at(-1).metrics.current.value)).toBe(speed.toFixed(2))
    }
    for (const wrong of [{ run_id: 'wrong-run' }, { instance_id: 'wrong-instance' }, { live_generation: { request_key: 'wrong-call', phase: 'output', scope: 'request', tokens_per_second: 999, backend_running: 1 } }]) {
      await f.page.evaluate(event => (window as any).emitGeneration(event), { run_id: current.run_id, instance_id: instance.id, observed_at: baseline.observed + 1,
        live_generation: { request_key: current.request_key, phase: 'output', scope: 'request', tokens_per_second: 999, backend_running: 1 }, ...wrong })
      await f.page.clock.runFor(20)
      await expect(mainSpeed(f.page)).toHaveText('44.44')
      expect(await f.page.evaluate(() => (window as any).miniSnapshots.at(-1).metrics.current.value)).toBe('44.44')
    }
    expect(f.reads).toEqual(reads)
    expect(await f.page.evaluate(() => (window as any).fixtureRenders)).toBe(baseline.renders)
    expect(await f.page.evaluate(() => (window as any).tokenMonitor.generation.requests[0].phases.decode.points.length)).toBe(baseline.points)
    expect(await f.page.evaluate(() => (window as any).generationConnections)).toEqual([{ url: 'http://token.fixture/api/instances/token-instance/generation-events', authorization: 'Bearer fixture-token' }])
    await f.page.screenshot({ path: path.join(f.directory, 'ninfer-each-token.png') })
    await expect(f.page.locator('.monitor-phase-toggle')).toHaveCount(0)
    await expect(prefillSpeed(f.page)).toHaveText('—')
    await expect(f.page.locator('tr[data-request-key="current-api-call"] [aria-label^="Prefill "]')).toContainText('5,000.00')
    await expect(mainSpeed(f.page)).toHaveText('44.44')
    expect(f.errors).toEqual([])
  } finally { await f.browser.close() }
})

test('外部API实时显示逐个快采样变化，不被1.5s目录/低频用量及累计均值覆盖', async () => {
  const f = await fixture()
  try {
    const speed = mainSpeed(f.page), panel = f.page.locator('.monitor-external-generation')
    const slowReads = { list: f.reads.list, usage: f.reads.usage }, previousTelemetryReads = f.reads.telemetry
    await expect(panel).toHaveAttribute('data-state', 'decode')
    for (const value of [22.22, 33.33]) {
      f.speed(value)
      await f.page.clock.runFor(500)
      await expect(speed).toHaveText(value.toFixed(2))
      await expect(panel).toHaveAttribute('data-state', 'decode')
    }
    expect(f.reads.telemetry).toBe(previousTelemetryReads + 2)
    expect(f.reads.usage).toBe(slowReads.usage)
    expect(f.reads.list).toBeLessThanOrEqual(slowReads.list + 1)
    expect(await speed.innerText()).not.toBe('40.00')
    await f.page.screenshot({ path: path.join(f.directory, 'external-current-sample.png') })
    expect(f.errors).toEqual([])
  } finally { await f.browser.close() }
})

test('外部API历史pin不被快采样打断，手选当前请求行恢复实时跟随', async () => {
  const f = await fixture()
  try {
    const group = (record: ExternalRequestRecord) => f.page.locator(`tr[data-group-key=${JSON.stringify(fixtureGroupKey(record))}]`)
    await group(history).locator('.monitor-ledger-group-select').click()
    await expect(f.page.locator('.monitor-compact-generation')).toHaveAttribute('data-scope', 'request')
    await expect(mainSpeed(f.page)).toHaveText('40.00')
    f.speed(66.66); await f.page.clock.runFor(500)
    await expect(mainSpeed(f.page)).toHaveText('40.00')
    const observed = await f.page.evaluate(() => (window as any).tokenMonitor.payload.updated_at)
    await f.page.evaluate(event => (window as any).emitGeneration(event), { run_id: current.run_id, instance_id: instance.id, observed_at: observed + .1,
      live_generation: { request_key: current.request_key, phase: 'output', scope: 'request', tokens_per_second: 88.88, backend_running: 1 } })
    await f.page.clock.runFor(20)
    await expect(mainSpeed(f.page)).toHaveText('40.00')
    expect(await f.page.evaluate(() => (window as any).miniSnapshots.at(-1).metrics.current.value)).toBe('88.88')
    const activeRequest = f.page.locator('tr[data-request-key="current-api-call"] .monitor-ledger-group-select')
    await activeRequest.focus(); await f.page.keyboard.press('Enter')
    await expect(f.page.locator('.monitor-compact-generation')).toHaveAttribute('data-scope', 'request')
    await expect(mainSpeed(f.page)).toHaveText('88.88')
    f.speed(77.77); await f.page.clock.runFor(500)
    await expect(mainSpeed(f.page)).toHaveText('77.77')
    await expect(f.page.locator('.monitor-external-generation')).toHaveAttribute('data-state', 'decode')
    expect(f.errors).toEqual([])
  } finally { await f.browser.close() }
})

test('终态null和并发不保留快值，外部断连留缺口、mini沿用既有poll契约且不改采样次数', async () => {
  const f = await fixture()
  try {
    let observed = await f.page.evaluate(() => (window as any).tokenMonitor.payload.updated_at)
    const reads = { ...f.reads }, readout = mainSpeed(f.page)
    const emit = async (live: object | null, expected: string) => {
      await f.page.evaluate(event => (window as any).emitGeneration(event), { run_id: current.run_id, instance_id: instance.id, observed_at: observed += .1, live_generation: live })
      await f.page.clock.runFor(20)
      await expect(readout).toHaveText(expected)
      expect(await f.page.evaluate(() => (window as any).miniSnapshots.at(-1).metrics.current.value)).toBe(expected)
    }
    const live = { request_key: current.request_key, phase: 'output', scope: 'request', tokens_per_second: 100, backend_running: 1 }
    await emit(live, '100.00')
    await emit(null, '—')
    await emit({ ...live, tokens_per_second: 200 }, '200.00')
    await emit({ ...live, backend_running: 2 }, '—')
    await emit({ ...live, tokens_per_second: 300 }, '300.00')
    await f.page.evaluate(() => (window as any).endGeneration()); await f.page.clock.runFor(20)
    await expect(readout).toHaveText('—')
    expect(await f.page.evaluate(() => (window as any).miniSnapshots.at(-1).metrics.current.value)).toBe('11.11')
    expect(f.reads).toEqual(reads); expect(f.errors).toEqual([])
  } finally { await f.browser.close() }
})

test('外部活跃缺测不沿用旧poll速度；真实0和waiting有效，mini保留旧读取契约', async () => {
  const f = await fixture()
  try {
    let observed = await f.page.evaluate(() => (window as any).tokenMonitor.payload.updated_at)
    const reads = { ...f.reads }, readout = mainSpeed(f.page)
    const emit = async (phase: string, value: number | null, expected: string, miniExpected = expected) => {
      await f.page.evaluate(event => (window as any).emitGeneration(event), { run_id: current.run_id, instance_id: instance.id, observed_at: observed += .1,
        live_generation: { request_key: current.request_key, phase, scope: 'request', tokens_per_second: value, window_s: value == null ? null : 3, available_for_chat: value != null, backend_running: 1 } })
      await f.page.clock.runFor(20); await expect(readout).toHaveText(expected)
      expect(await f.page.evaluate(() => (window as any).miniSnapshots.at(-1).metrics.current.value)).toBe(miniExpected)
    }
    await emit('output', null, '—', '11.11')
    await emit('output', 0, '0.00')
    await emit('output', null, '—', '11.11')
    await emit('waiting', null, '—')
    expect(f.reads).toEqual(reads); expect(f.errors).toEqual([])
  } finally { await f.browser.close() }
})


test('paired快计量同步主图/当前请求行/mini，旧同轮prior不污染自身数值', async () => {
  const f = await fixture(true)
  try {
    const parent = f.page.locator(`tr[data-group-key=${JSON.stringify(fixtureGroupKey(current))}]`)
    await parent.locator('.monitor-ledger-group-select').click()
    await expect(mainSpeed(f.page)).toHaveText('11.11')
    const baseline = await f.page.evaluate(() => ({ renders: (window as any).fixtureRenders, observed: (window as any).tokenMonitor.payload.updated_at,
      points: (window as any).tokenMonitor.generation.requests.map((r: any) => r.phases.decode.points.length),
      structure: [...document.querySelectorAll('tr[data-group-key],tr[data-request-key]')].map(n => n.getAttribute('data-group-key') || n.getAttribute('data-request-key')) }))
    const reads = { ...f.reads }
    for (const [index, tokens] of [150, 180, 210].entries()) {
      const accounting = nativeAccounting(tokens, 3, 1100 + index * 100, 750 + index * 50)
      await f.page.evaluate(event => (window as any).emitGeneration(event), { run_id: current.run_id, instance_id: instance.id, observed_at: baseline.observed + .1 + index / 10,
        live_generation: { request_key: current.request_key, phase: 'output', scope: 'request', tokens_per_second: 90 + index, backend_running: 1 }, accounting })
      await f.page.clock.runFor(20)
      await expect(mainSpeed(f.page)).toHaveText((90 + index).toFixed(2))
      await expect(parent.locator('[data-column="speed"] > span')).toHaveText((tokens / 3).toFixed(2))
      const child = f.page.locator('tr[data-request-key="current-api-call"]')
      await expect(child.locator('[data-column="speed"] span').first()).toHaveText((tokens / 3).toFixed(2))
      await expect(child.locator('[aria-label^="输出 "]')).toContainText(String(tokens))
      await expect(f.page.locator('.monitor-request-live > div').first().locator('strong')).toHaveText((1100 + index * 100 + tokens).toLocaleString('en-US'))
      await expect(f.page.locator('.monitor-request-live > div').last().locator('strong')).toHaveText(String(tokens))
      const mini = await f.page.evaluate(() => (window as any).miniSnapshots.at(-1).metrics)
      expect(mini.current.value).toBe(String(90 + index) + '.00')
      expect(mini.average.value).toBe((tokens / 3).toFixed(2))
      expect(mini.generated.value).toBe(String(tokens))
      expect(mini.prompt.value).toBe((1100 + index * 100).toLocaleString('en-US'))
      expect(mini.context.value.split(' / ')[0]).toBe((1100 + index * 100 + tokens).toLocaleString('en-US'))
      expect(mini.first_output.value).toBe('0.25')
    }
    expect(f.reads).toEqual(reads)
    expect(await f.page.evaluate(() => (window as any).fixtureRenders)).toBe(baseline.renders)
    expect(await f.page.evaluate(() => (window as any).tokenMonitor.generation.requests.map((r: any) => r.phases.decode.points.length))).toEqual(baseline.points)
    expect(await f.page.evaluate(() => [...document.querySelectorAll('tr[data-group-key],tr[data-request-key]')].map(n => n.getAttribute('data-group-key') || n.getAttribute('data-request-key')))).toEqual(baseline.structure)
    await expect(prefillSpeed(f.page)).toHaveText('—')
    await expect(parent.locator('[aria-label^="Prefill "]')).toContainText('6,800.00')
    await f.page.screenshot({ path: path.join(f.directory, 'consistent-turn-and-mini.png') })
    fs.writeFileSync(path.join(f.directory, 'observations.json'), JSON.stringify({ baseline, reads, afterReads: f.reads, mini: await f.page.evaluate(() => (window as any).miniSnapshots.at(-1)) }, null, 2))
    expect(f.errors).toEqual([])
  } finally { await f.browser.close() }
})

test('默认双轴不以阶段均值冒充current；均值、TTFT与终态仍抵达逐请求和mini leaf', async () => {
  const f = await fixture(true)
  try {
    const parent = f.page.locator(`tr[data-group-key=${JSON.stringify(fixtureGroupKey(current))}]`)
    const observed = await f.page.evaluate(() => (window as any).tokenMonitor.payload.updated_at)
    const accounting = nativeAccounting(150, 3)
    const send = async (event: object, paint = true) => {
      await f.page.evaluate(event => (window as any).emitGeneration(event), { run_id: current.run_id, instance_id: instance.id, observed_at: observed + .1, ...event })
      if (paint) await f.page.clock.runFor(20)
    }
    await send({ accounting, live_generation: { request_key: current.request_key, phase: 'output', scope: 'request', tokens_per_second: 90, backend_running: 1 } })
    await expect(mainSpeed(f.page)).toHaveText('90.00')
    await expect(f.page.locator('.monitor-request-live > div').first().locator('strong')).toHaveText('1,250')
    await expect(f.page.locator('.monitor-request-live > div').last().locator('strong')).toHaveText('150')
    await expect(prefillSpeed(f.page)).toHaveText('—')
    await expect(parent.locator('[aria-label^="Prefill "]')).toContainText('6,000.00')
    const partial = structuredClone(accounting)
    partial.request.timings.predicted_per_second = 55
    partial.request.timings.prompt_per_second = 7000
    partial.request.usage.ttft_ms = 500
    partial.request.measurements.decode.coverage = 'partial'
    partial.request.measurements.prefill.coverage = 'partial'
    await send({ accounting: partial, live_generation: { request_key: current.request_key, phase: 'output', scope: 'request', tokens_per_second: 91, backend_running: 1 } })
    await expect(prefillSpeed(f.page)).toHaveText('—')
    await expect(parent.locator('[aria-label^="Prefill "]')).toContainText('7,000.00')
    expect(await f.page.evaluate(() => (window as any).miniSnapshots.at(-1).metrics.first_output.value)).toBe('0.25')
    expect(await f.page.evaluate(() => (window as any).miniSnapshots.at(-1).metrics.average.value)).toBe('55.00')
    await parent.locator('.monitor-ledger-group-select').click()
    await expect(mainSpeed(f.page)).toHaveText('91.00')
    const final = nativeAccounting(240, 8)
    final.request.status = 'completed'
    final.request.timings.predicted_per_second = 99 // sticky earlier mean is not the fresh final pair.
    await send({ accounting: partial, live_generation: { request_key: current.request_key, phase: 'output', scope: 'request', tokens_per_second: 92, backend_running: 1 } }, false)
    await send({ accounting: null, finalized: [final], live_generation: { request_key: 'next-unconfirmed-call', phase: 'waiting', scope: 'request', tokens_per_second: null, backend_running: 1 } })
    await expect(mainSpeed(f.page)).toHaveText('—')
    await expect(f.page.locator('tr[data-request-key="current-api-call"] [data-column="speed"] span').first()).toHaveText('30.00')
    await expect(parent.locator('[data-column="output"]')).toContainText('240')
    await expect(f.page.locator('.monitor-request-live > div').last().locator('strong')).toHaveText('240')
    const mini = await f.page.evaluate(() => (window as any).miniSnapshots.at(-1).metrics)
    expect(mini.current.value).toBe('—'); expect(mini.average.value).toBe('30.00'); expect(mini.generated.value).toBe('240'); expect(mini.context.value.split(' / ')[0]).toBe('1,340')
    await f.page.screenshot({ path: path.join(f.directory, 'finalized-before-next-frame.png') })
    expect(f.errors).toEqual([])
  } finally { await f.browser.close() }
})

test('internal实hook入口的新activity胜过旧poll身份，原生SSE均值/计数和mini不丢帧', async () => {
  const f = await fixture(false, true)
  try {
    const reads = { ...f.reads }, observed = await f.page.evaluate(() => (window as any).tokenMonitor.payload.updated_at)
    const at = observed * 1000 + 100
    await f.page.evaluate(at => {
      const monitor = (window as any).tokenMonitor
      monitor.reportChatActivity({ instanceId: 'token-instance', messageId: 'previous-turn', phase: 'completed', requests: [{ id: 'previous-call', requestKey: 'previous-server-call', index: 1, kind: 'generation', status: 'completed', startedAt: at - 10000, endedAt: at - 5000, elapsed: 5, labels: [], usage: { prompt_tokens: 1000, completion_tokens: 100 }, timings: { predicted_n: 100, predicted_ms: 2500 } }] })
    }, at)
    for (const [index, item] of [[30, 100, 200], [60, 200, 250], [110, 300, 333.33]].entries()) {
      const [tokens, ms, currentRate] = item
      await f.page.evaluate(({ tokens, ms, currentRate, index, at }) => {
        const record = { id: 'new-call', requestKey: 'new-server-call', index: 1, kind: 'generation', status: 'running', startedAt: at, labels: [], usage: { prompt_tokens: 1000, completion_tokens: tokens, ttft_ms: 200 }, timings: { predicted_n: tokens, predicted_ms: ms } }
        ;(window as any).tokenMonitor.reportChatActivity({ instanceId: 'token-instance', messageId: 'new-turn', phase: 'generating', requests: [record],
          stream: { requestId: record.id, sampledAt: at + index * 100, tokensPerSecond: currentRate, windowSeconds: ms / 1000 } })
      }, { tokens, ms, currentRate, index, at })
      await expect(mainSpeed(f.page)).toHaveText(currentRate.toFixed(2))
      await expect(f.page.locator('.monitor-request-group')).toHaveAttribute('data-selected-request-key', 'new-server-call')
      await expect(f.page.locator('.monitor-request-live > div').last().locator('strong')).toHaveText(String(tokens))
      const mini = await f.page.evaluate(() => (window as any).miniSnapshots.at(-1).metrics)
      expect(mini.average.value).toBe((tokens / (ms / 1000)).toFixed(2)); expect(mini.generated.value).toBe(String(tokens)); expect(mini.current.value).toBe(currentRate.toFixed(2)); expect(mini.first_output.value).toBe('0.2')
    }
    expect(f.reads).toEqual(reads)
    expect(f.errors).toEqual([])
    await f.page.screenshot({ path: path.join(f.directory, 'internal-new-activity.png') })
  } finally { await f.browser.close() }
})

test('无滚动快值时外部current为未知，Decode均值仍逐native配对更新逐请求和mini', async () => {
  const f = await fixture(true, false, true)
  try {
    const observed = await f.page.evaluate(() => (window as any).tokenMonitor.payload.updated_at)
    const baseline = await f.page.evaluate(() => (window as any).fixtureRenders), reads = { ...f.reads }
    for (const [index, tokens] of [150, 180, 210].entries()) {
      await f.page.evaluate(event => (window as any).emitGeneration(event), { run_id: current.run_id, instance_id: instance.id, observed_at: observed + index / 10 + .1,
        live_generation: { request_key: current.request_key, phase: 'output', scope: 'request', tokens_per_second: null, backend_running: 1 }, accounting: nativeAccounting(tokens, 3) })
      await f.page.clock.runFor(20)
      await expect(mainSpeed(f.page)).toHaveText('—')
      await expect(f.page.locator('tr[data-request-key="current-api-call"] [data-column="speed"] > span')).toHaveText((tokens / 3).toFixed(2))
      expect(await f.page.evaluate(() => (window as any).miniSnapshots.at(-1).metrics.average.value)).toBe((tokens / 3).toFixed(2))
    }
    expect(f.reads).toEqual(reads); expect(await f.page.evaluate(() => (window as any).fixtureRenders)).toBe(baseline)
    expect(f.errors).toEqual([])
  } finally { await f.browser.close() }
})

test('权威counter unknown不回退sticky；真实0保留，父轮已知部分与wrong identity隔离', async () => {
  const f = await fixture(true)
  try {
    const parent = f.page.locator(`tr[data-group-key=${JSON.stringify(fixtureGroupKey(current))}]`)
    const observed = await f.page.evaluate(() => (window as any).tokenMonitor.payload.updated_at)
    const unknown = nativeAccounting(150, 3)
    unknown.request.counters = { input: null, output: null, cached: null }
    const send = async (accounting: GenerationAccounting, rate: number | null, identity = current.request_key) => {
      await f.page.evaluate(event => (window as any).emitGeneration(event), { run_id: current.run_id, instance_id: instance.id, observed_at: observed + .1,
        live_generation: { request_key: identity, phase: 'output', scope: 'request', tokens_per_second: rate, backend_running: 1 }, accounting })
      await f.page.clock.runFor(20)
    }
    await send(unknown, null)
    await expect(f.page.locator('.monitor-request-live > div').first().locator('strong')).toHaveText('—')
    await expect(f.page.locator('.monitor-request-live > div').last().locator('strong')).toHaveText('—')
    const miniUnknown = await f.page.evaluate(() => (window as any).miniSnapshots.at(-1).metrics)
    expect(miniUnknown.generated.value).toBe('—'); expect(miniUnknown.prompt.value).toBe('—'); expect(miniUnknown.context.value.split(' / ')[0]).toBe('—')
    const child = f.page.locator('tr[data-request-key="current-api-call"]')
    await expect(child.locator('[data-column="output"]')).toContainText('—')
    await parent.locator('.monitor-ledger-group-select').click()
    await expect(f.page.locator('.monitor-request-live > div').last().locator('dt')).toHaveText('输出')
    const zero = nativeAccounting(0, 1, 0, 0)
    zero.request.counters = { input: 0, output: 0, cached: 0 }
    await send(zero, 0)
    await expect(child.locator('[data-column="output"]')).toContainText('0')
    await expect(f.page.locator('.monitor-request-live > div').last().locator('dt')).not.toContainText('部分')
    const miniZero = await f.page.evaluate(() => (window as any).miniSnapshots.at(-1).metrics)
    expect(miniZero.generated.value).toBe('0'); expect(miniZero.prompt.value).toBe('0'); expect(miniZero.current.value).toBe('0.00')
    await send(nativeAccounting(999, 1), 999, 'wrong-call')
    expect(await f.page.evaluate(() => (window as any).miniSnapshots.at(-1).metrics.generated.value)).toBe('0')
    expect(f.errors).toEqual([])
  } finally { await f.browser.close() }
})

test('只有真实引擎AVG的终态仍同步Mini既有fallback，不添加Decode实时能力', async () => {
  const f = await fixture(true)
  try {
    const observed = await f.page.evaluate(() => (window as any).tokenMonitor.payload.updated_at)
    const final = nativeAccounting(240, 8)
    final.request.status = 'completed'
    final.request.timings = {}
    final.request.usage = { prompt_tokens: 1100, completion_tokens: 240, generation_ms: 8000, decode_tps: 30, ttft_ms: 250 }
    const unknown = { tokens: null, seconds: null, value: null, basis: null, reported: 0 as const, coverage: 'unknown' as const }
    final.request.measurements = { decode: unknown, prefill: unknown }
    await f.page.evaluate(event => (window as any).emitGeneration(event), { run_id: current.run_id, instance_id: instance.id, observed_at: observed + .1, live_generation: null, finalized: [final] })
    await f.page.clock.runFor(20)
    await expect(mainSpeed(f.page)).toHaveText('—')
    await expect(f.page.locator('tr[data-request-key="current-api-call"] [data-column="speed"] > span')).toHaveText('30.00AVG')
    await expect(f.page.locator('tr[data-request-key="current-api-call"] [data-column="speed"] > span')).toHaveAttribute('aria-label', '引擎平均 30.00 token/s · 包含首 Token 等待')
    const mini = await f.page.evaluate(() => (window as any).miniSnapshots.at(-1).metrics)
    expect(mini.average.value).toBe('30.00'); expect(mini.average.basis).toBe('引擎平均'); expect(mini.current.value).toBe('—'); expect(mini.generated.value).toBe('240')
    expect(f.errors).toEqual([])
  } finally { await f.browser.close() }
})

test('目录终态尚被旧poll遮蔽时不提前释放快终值；实际慢照接管后清缓存且各leaf不回退', async () => {
  const f = await fixture(true)
  try {
    const parent = f.page.locator(`tr[data-group-key=${JSON.stringify(fixtureGroupKey(current))}]`)
    await parent.locator('.monitor-ledger-group-select').click()
    const observed = await f.page.evaluate(() => { (window as any).requestBeforeSlow = (window as any).tokenMonitor.generation.requests.at(-1); return (window as any).tokenMonitor.payload.updated_at })
    const final = nativeAccounting(240, 8); final.request.status = 'completed'
    await f.page.evaluate(event => (window as any).emitGeneration(event), { run_id: current.run_id, instance_id: instance.id, observed_at: observed + .1, live_generation: null, finalized: [final] })
    await f.page.clock.runFor(20)
    await expect(mainSpeed(f.page)).toHaveText('—')
    f.finish(final.request, false)
    await f.page.clock.runFor(1500)
    await expect(mainSpeed(f.page)).toHaveText('—')
    expect(await f.page.evaluate(() => (window as any).miniSnapshots.at(-1).metrics.generated.value)).toBe('240')
    f.finish(final.request)
    await f.page.clock.runFor(500)
    await expect.poll(async () => f.page.evaluate(() => (window as any).tokenMonitor.payload.external_history.requests.at(-1).status)).toBe('completed')
    await f.page.clock.runFor(20)
    await expect(mainSpeed(f.page)).toHaveText('—')
    await expect(parent.locator('[data-column="speed"] > span')).toHaveText('30.00')
    await expect(parent.locator('[data-column="output"]')).toContainText('240')
    expect(await f.page.evaluate(() => (window as any).tokenMonitor.generationEvents.accounting((window as any).requestBeforeSlow))).toBeNull()
    expect(await f.page.evaluate(() => (window as any).miniSnapshots.at(-1).metrics.generated.value)).toBe('240')
    expect(f.errors).toEqual([])
  } finally { await f.browser.close() }
})
