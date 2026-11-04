import { chromium, expect, test } from '@playwright/test'
import { build } from 'esbuild'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { MonitorRequestTable } from '../src/MonitorRequestTable'
import { externalGenerationView, externalRequestIdentity } from '../src/monitor-generation'
import { requestErrorSummary, type MonitorRequestTableEntry } from '../src/monitor-request-groups'
import { fixtureGroupKey, historyPage, historyStats, historyUsage } from './monitor-history-fixture'
import type { ExternalRequestRecord, Telemetry } from '../src/types'

const record = (key: string, more: Partial<ExternalRequestRecord> = {}): ExternalRequestRecord => ({
  run_id: 'recovery-run', instance_id: 'model', alias: 'NInfer · Qwen3.8', backend: 'ninfer', request_key: key, revision: 1,
  aggregation: 'turn', turn_id: 'user-turn', turn_state: 'recovered', status: 'completed', started_at: 100, ended_at: 102, duration_s: 2,
  labels: [], usage: { prompt_tokens: 100, completion_tokens: 20 }, timings: { predicted_n: 20, predicted_ms: 1000 },
  client_context: { source: 'opencode', session_id: 'session' }, ...more,
})
const failed = record('failed-call', { status: 'failed', upstream: { status: 400, code: 'context_length_exceeded', reason: '上下文超出限制' } })
const completed = record('completed-call', { started_at: 103, ended_at: 105 })
const view = (records = [failed, completed], state: 'recovered' | 'waiting' = 'recovered') => externalGenerationView({
  run_id: 'recovery-run', aggregation: 'turn', turn_id: 'user-turn', state, requests: records,
})
const entry = (): MonitorRequestTableEntry => ({ key: 'failed', turn: false, alias: failed.alias, backend: failed.backend, requests: view([failed]).requests,
  count: 1, status: 'failed', startedAt: 100000 })
const render = (value: MonitorRequestTableEntry) => renderToStaticMarkup(createElement(MonitorRequestTable, {
  entries: [value], onSelectGroup() {}, onSelectRequest() {}, onToggleGroup() {},
}))

test('旧已恢复轮次显示当前请求终态，失败尝试、客户端身份和upstream独立保留', () => {
  const recovered = view()
  expect(recovered.status).toBe('已完成')
  expect(recovered.working).toBe(false); expect(recovered.ended).toBe(true)
  expect(recovered.turn).toBeUndefined()
  expect(recovered.requests.map(request => request.status)).toEqual(['completed'])
  const failure = view([failed]).requests[0]
  expect(failure.external?.client_context).toEqual(failed.client_context)
  expect(failure.external?.upstream).toEqual(failed.upstream)
  expect(historyStats([failed, completed]).status_counts).toEqual({ running: 0, completed: 1, failed: 1, cancelled: 0, incomplete: 0 })
})

test('失败请求保留独立红点与简洁错误，不再保留已恢复父行或状态列', () => {
  const html = render(entry())
  expect(html).not.toContain('data-status="recovered"')
  expect(html).not.toContain('aria-label="已恢复"')
  expect(html).not.toContain('monitor-ledger-recovered')
  expect(html).toContain('上下文超出限制 · HTTP 400')
  expect(html).toContain('data-status="failed"')
  expect(html).not.toContain('context_length_exceeded')
  expect(html).not.toMatch(/<th[^>]*>状态<\/th>/)
  expect(html.match(/scope="col"/g)).toHaveLength(7)
})

test('错误reason优先、状态码不重复，code/HTTP可独立回退，旧无metadata不捏造', () => {
  const failure = view([failed]).requests[0]
  expect(requestErrorSummary(failure)).toBe('上下文超出限制 · HTTP 400')
  const withUpstream = (upstream: ExternalRequestRecord['upstream']) => ({ ...failure, external: { ...failed, upstream } })
  expect(requestErrorSummary(withUpstream({ status: 400, reason: 'HTTP 400: 上下文超出限制', code: 'ignored' }))).toBe('HTTP 400: 上下文超出限制')
  expect(requestErrorSummary(withUpstream({ status: 503, code: 'overloaded' }))).toBe('overloaded · HTTP 503')
  expect(requestErrorSummary(withUpstream({ status: 502 }))).toBe('HTTP 502')
  expect(requestErrorSummary(withUpstream(undefined))).toBe('')
  expect(requestErrorSummary(view([completed]).requests[0])).toBe('')
})

test('真实请求status优先于旧waiting和recovered轮次元数据', () => {
  const running = view([failed, { ...completed, status: 'running', ended_at: undefined }])
  expect(running.working).toBe(true); expect(running.status).toBe('等待响应'); expect(running.turn).toBeUndefined()
  const waiting = view([failed, completed], 'waiting')
  expect(waiting.waiting).toBe(false); expect(waiting.status).toBe('已完成')
  const runningHtml = render({ ...entry(), requests: running.requests })
  expect(runningHtml).toContain('data-status="running"'); expect(runningHtml).not.toContain('monitor-ledger-recovered')
  const waitingHtml = render({ ...entry(), requests: waiting.requests })
  expect(waitingHtml).toContain('data-status="completed"'); expect(waitingHtml).not.toContain('monitor-ledger-recovered')
})

test('失败流保留真实握手码，摘要只在HTTP错误码时附加状态，不把200染成HTTP失败', () => {
  const failure = view([failed]).requests[0]
  for (const status of [200, 204, 399]) {
    const request = { ...failure, external: { ...failed, upstream: { status, reason: '响应流中断' } } }
    expect(requestErrorSummary(request)).toBe('响应流中断')
    expect(request.external.upstream.status).toBe(status)
  }
  expect(requestErrorSummary({ ...failure, external: { ...failed, upstream: { status: 200, code: 'stream_error' } } })).toBe('stream_error')
  expect(requestErrorSummary({ ...failure, external: { ...failed, upstream: { status: 200 } } })).toBe('')
  expect(requestErrorSummary({ ...failure, external: { ...failed, upstream: { status: 400, reason: '响应流中断' } } })).toBe('响应流中断 · HTTP 400')
})

test('upstream数字code原样保留，code-only安全显示且reason仍优先', () => {
  const failure = view([failed]).requests[0]
  const codeOnly = { ...failure, external: { ...failed, upstream: { status: 400, code: 400 } } }
  expect(requestErrorSummary(codeOnly)).toBe('400')
  expect(codeOnly.external.upstream.code).toBe(400)
  expect(requestErrorSummary({ ...failure, external: { ...failed, upstream: { code: 503 } } })).toBe('503')
  expect(requestErrorSummary({ ...failure, external: { ...failed, upstream: { status: 400, code: 400, reason: '后端返回 HTTP 400' } } })).toBe('后端返回 HTTP 400')
})

test('真实监控入口拆开旧重试轮次，失败请求固定后不被新请求或旧waiting状态抢走', async () => {
  const root = path.resolve(__dirname, '..'), now = Date.parse('2026-10-31T12:00:00Z')
  const artifacts = process.env.MODLOADER_TEST_ARTIFACTS || path.join(root, '.artifacts/codex/monitor-recovery-20261031/tests')
  fs.mkdirSync(artifacts, { recursive: true })
  const directory = fs.mkdtempSync(path.join(artifacts, 'recovered-entry-'))
  const source = (file: string) => JSON.stringify(path.join(root, file).replaceAll('\\', '/'))
  const instance = { id: 'model', alias: 'NInfer · Qwen3.8', backend: 'ninfer', state: 'ready', active_requests: 1 }
  fs.writeFileSync(path.join(directory, 'fixture.tsx'), `import React from'react';import{createRoot}from'react-dom/client';import{useMonitorState}from ${source('src/useMonitorState')};import{MonitorRequestHistory}from ${source('src/MonitorRequestHistory')};import ${source('src/styles.css')};import ${source('src/resource-cards.css')};
    const state={models:[],profiles:[],runtimes:[],instances:[${JSON.stringify(instance)}],downloads:[],installations:[],settings:{},backends:[],hardware:{}};
    function Fixture(){const monitor=useMonitorState(state,async()=>{});return <main className="monitor-page" style={{padding:24,'--font-scale':1}}><MonitorRequestHistory monitor={monitor} active/></main>};createRoot(document.getElementById('root')).render(<Fixture/>);`)
  await build({ entryPoints: [path.join(directory, 'fixture.tsx')], outfile: path.join(directory, 'fixture.js'), bundle: true, format: 'iife', jsx: 'automatic', loader: { '.woff2': 'file' }, define: { 'process.env.NODE_ENV': '"development"' } })
  const browser = await chromium.launch({ channel: 'msedge', headless: true })
  try {
    const page = await browser.newPage({ viewport: { width: 1366, height: 900 } })
    await page.clock.install({ time: new Date(now) })
    await page.addInitScript(() => {
      ;(window as any).modloader = { connection: async () => ({ baseUrl: 'http://recovery.fixture', token: 'fixture-token' }) }
      const original = window.fetch.bind(window)
      window.fetch = async (input, options) => String(input).endsWith('/generation-events')
        ? new Response(new ReadableStream({ start(controller) { options?.signal?.addEventListener('abort', () => controller.close(), { once: true }) } }), { headers: { 'Content-Type': 'text/event-stream' } }) : original(input, options)
    })
    const errors: string[] = []
    page.on('pageerror', error => errors.push(error.message))
    let stage: 'recovered' | 'running' | 'waiting' = 'recovered', polls = 0
    const records = () => [failed, completed, ...(stage === 'recovered' ? [] : [record('new-call', { status: stage === 'running' ? 'running' : 'completed', started_at: 110, ended_at: stage === 'running' ? undefined : 112 })])]
      .map((item, index) => ({ ...item, started_at: now / 1000 - 60 + index * 3, ended_at: item.status === 'running' ? undefined : now / 1000 - 58 + index * 3, turn_state: stage }))
    await page.route('http://recovery.fixture/**', async route => {
      const url = new URL(route.request().url()), requests = records()
      if (url.pathname.endsWith('/telemetry')) {
        const telemetry: Telemetry = { instance_id: instance.id, backend: instance.backend, state: 'ready', updated_at: now / 1000 + ++polls / 2, groups: [], unavailable: [],
          external_history: { run_id: failed.run_id, aggregation: 'turn', turn_id: failed.turn_id, state: stage, requests },
          live_generation: stage === 'running' ? { request_key: 'new-call', phase: 'output', scope: 'request', tokens_per_second: 30, backend_running: 1, backend_queued: 0 } : null }
        return route.fulfill({ json: telemetry })
      }
      if (url.pathname === '/api/request-history/dates') return route.fulfill({ json: { current_run_id: failed.run_id, today: '2026-10-31', usage_revision: 'fixed', dates: [{ date: '2026-10-31', request_count: 2, instances: [{ source: 'external', run_id: failed.run_id, id: instance.id, alias: instance.alias, backend: instance.backend }] }], issues: [] } })
      if (url.pathname === '/api/request-history/usage') return route.fulfill({ json: historyUsage(historyStats([failed, completed])) })
      if (url.pathname === '/api/request-history') return route.fulfill({ json: historyPage(requests, url.searchParams) })
      if (url.pathname === '/') return route.fulfill({ contentType: 'text/html', body: '<!doctype html><meta charset="utf-8"><link rel="stylesheet" href="fixture.css"><div id="root"></div><script src="fixture.js"></script>' })
      return route.fulfill({ path: path.join(directory, path.basename(url.pathname)) })
    })
    await page.goto('http://recovery.fixture/')
    const success = page.locator('tr[data-request-key="completed-call"]')
    await expect(success.getByRole('img', { name: '已完成', exact: true })).toHaveAttribute('data-status', 'completed')
    await expect(page.locator('.monitor-ledger-recovered')).toHaveCount(0)
    const green = await success.locator('.monitor-ledger-dot').evaluate(node => getComputedStyle(node).backgroundColor)
    expect(green).not.toBe('rgba(0, 0, 0, 0)')
    await page.clock.pauseAt(new Date(now + 1000))
    const child = page.locator('tr[data-request-key="failed-call"]')
    await expect(child.locator('.monitor-ledger-request-error')).toHaveText('上下文超出限制 · HTTP 400')
    await expect(child.getByRole('img', { name: '失败', exact: true })).toHaveAttribute('data-status', 'failed')
    expect(await child.locator('.monitor-ledger-dot').evaluate(node => getComputedStyle(node).backgroundColor)).not.toBe(green)
    await child.locator('.monitor-ledger-group-select').click()
    await expect(page.locator('.monitor-request-group')).toHaveAttribute('data-selected-request-key', 'failed-call')
    await expect(page.locator('.monitor-compact-generation')).toHaveAttribute('data-scope', 'request')
    expect((await child.boundingBox())!.height).toBeGreaterThanOrEqual(80)
    await page.screenshot({ path: path.join(directory, 'recovered-and-failed.png') })
    stage = 'running'; await page.clock.runFor(500)
    await expect(page.locator('tr[data-request-key="new-call"] .monitor-ledger-dot')).toHaveAttribute('data-status', /running|waiting/)
    await expect(page.locator('.monitor-request-group')).toHaveAttribute('data-selected-request-key', 'failed-call')
    stage = 'waiting'; await page.clock.runFor(500)
    await expect(page.locator('tr[data-request-key="new-call"] .monitor-ledger-dot')).toHaveAttribute('data-status', 'completed')
    await expect(page.locator('.monitor-ledger-recovered')).toHaveCount(0)
    await expect(page.locator('.monitor-request-group')).toHaveAttribute('data-selected-request-key', 'failed-call')
    await expect(page.locator('thead th')).toHaveCount(7)
    expect(errors).toEqual([])
  } finally { await browser.close() }
})
