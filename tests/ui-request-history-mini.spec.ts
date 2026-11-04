import { test, expect, chromium } from '@playwright/test'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { pinNavigation } from './navigation'
import { fixtureGroupKey, historyPage, historyStats, historyUsage } from './monitor-history-fixture'
import { localDate } from '../src/monitor-history-dates'
import type { MiniMonitorSnapshot } from '../src/mini-monitor'
import type { ExternalRequestRecord } from '../src/types'

test('统一监控固定旧记录不切换小窗发布值、采样或停止对象，返回最新与无实例回看', async () => {
  test.setTimeout(60_000)
  const root = path.resolve(__dirname, '..')
  const output = process.env.MODLOADER_TEST_ARTIFACTS || path.join(root, '.artifacts/codex/20261031-unified-monitor/frontend')
  fs.mkdirSync(output, { recursive: true })
  const directory = fs.mkdtempSync(path.join(output, 'mini-bridge-'))
  const { preview } = await import('vite')
  const server = await preview({ root, configFile: false, build: { outDir: process.env.MODLOADER_TEST_DIST || path.join(root, '.artifacts/codex/20261031-unified-monitor/dist') },
    preview: { host: '127.0.0.1', port: 0, strictPort: false } })
  const address = server.httpServer.address()
  if (!address || typeof address === 'string') throw new Error('No preview')
  const browser = await chromium.launch({ channel: 'msedge', headless: true })
  const page = await browser.newPage({ viewport: { width: 1440, height: 1040 } })
  await page.clock.install()
  // Exercise the actual App publication boundary without opening a native window.
  await page.addInitScript(() => {
    const target = window as unknown as { modloader: unknown; fixtureMini?: unknown; fixtureMiniOpens: number }
    target.fixtureMiniOpens = 0
    target.modloader = { connection: async () => ({ baseUrl: '', token: '' }),
      publishMiniMonitor: (snapshot: unknown) => { target.fixtureMini = snapshot },
      openMiniMonitor: async () => { target.fixtureMiniOpens++ },
      onMiniMonitorAction: () => () => {}, onBeforeClose: () => () => {}, onCloseCancelled: () => () => {}, onWindowHidden: () => () => {} }
  })
  const now = Date.now() / 1000, today = localDate()
  const instance = { id: 'live', alias: 'live fixture', backend: 'sparkinfer', state: 'ready', active_requests: 0,
    model_id: 'fixture', profile_id: 'fixture', capabilities: { input: ['text'] } }
  const record = (key: string, input: number, elapsed: number): ExternalRequestRecord => ({ request_key: key, run_id: 'current', instance_id: 'live', alias: instance.alias,
    backend: 'sparkinfer', revision: 1, aggregation: 'request', turn_id: null, turn_state: null, labels: [], streaming: true, date: today,
    status: 'completed', started_at: now - elapsed - 1, ended_at: now - 1, duration_s: elapsed, first_output_s: .2,
    usage: { prompt_tokens: input, prompt_tokens_details: { cached_tokens: 0 }, completion_tokens: 24, ttft_ms: 80, generation_ms: 400, decode_tps: 60 }, timings: {}, prefill: {} })
  let latest = record('latest', 320, 2)
  const older = { ...record('older', 100, 30), run_id: 'prior', instance_id: 'removed', alias: 'archived fixture',
    usage: { prompt_tokens: 100, prompt_tokens_details: { cached_tokens: 20 }, completion_tokens: 24, ttft_ms: 900, generation_ms: 2000, decode_tps: 12 } }
  let telemetryReads = 0, removed = false
  const errors: string[] = [], unexpected: string[] = [], stops: string[] = [], logs: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/api/**', async route => {
    const url = new URL(route.request().url())
    if (url.pathname.endsWith('/generation-events') && route.request().method() === 'GET') return route.fulfill({ contentType: 'text/event-stream', body: ': fixture idle\n\n' })
    if (url.pathname.endsWith('/stop') && route.request().method() === 'POST') { stops.push(url.pathname); return route.fulfill({ json: { ok: true } }) }
    if (route.request().method() !== 'GET') { unexpected.push(`${route.request().method()} ${url.pathname}`); return route.abort() }
    if (url.pathname === '/api/state') return route.fulfill({ json: { models: [], profiles: [], runtimes: [], instances: removed ? [] : [instance], settings: {},
      backends: [{ id: 'sparkinfer', label: 'SparkInfer', parameters: [] }], hardware: { gpus: [] } } })
    if (url.pathname === '/api/conversations') return route.fulfill({ json: { items: [], active_id: null } })
    if (url.pathname === '/api/chat-presets') return route.fulfill({ json: { items: [], default_id: null } })
    if (url.pathname.endsWith('/logs')) { logs.push(url.pathname); return route.fulfill({ json: { lines: ['fixture log'] } }) }
    if (url.pathname.endsWith('/telemetry')) {
      telemetryReads++
      expect(url.pathname).toBe('/api/instances/live/telemetry')
      return route.fulfill({ json: { instance_id: 'live', backend: 'sparkinfer', state: 'ready', updated_at: Date.now() / 1000,
        groups: [], unavailable: [], live_generation: null, external_history: { run_id: 'current', aggregation: 'request', turn_id: null, state: 'completed', requests: [latest] } } })
    }
    if (url.pathname === '/api/request-history/dates') return route.fulfill({ json: { current_run_id: 'current', today, issues: [],
      dates: [{ date: today, request_count: 2, instances: [{ run_id: 'current', id: 'live', alias: instance.alias, backend: 'sparkinfer', started_at: now - 100 },
        { run_id: 'prior', id: 'removed', alias: older.alias, backend: 'sparkinfer', started_at: now - 90000 }] }] } })
    if (url.pathname === '/api/request-history/usage') return route.fulfill({ json: historyUsage(historyStats([latest, older])) })
    if (url.pathname === '/api/request-history') return route.fulfill({ json: historyPage(
      url.searchParams.has('request_key') ? [latest, older].filter(record => record.request_key === url.searchParams.get('request_key')) : [latest, older], url.searchParams) })
    unexpected.push(url.pathname)
    return route.fulfill({ status: 404, json: { detail: `Unexpected ${url.pathname}` } })
  })
  const mini = () => page.evaluate(() => (window as unknown as { fixtureMini: MiniMonitorSnapshot }).fixtureMini)
  try {
    await page.goto(`http://127.0.0.1:${address.port}`); await pinNavigation(page)
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '运行监控', exact: true }).click()
    const monitor = page.locator('.monitor-page'), chosen = monitor.getByRole('region', { name: '监控生成统计', exact: true })
    const table = monitor.getByRole('region', { name: '请求记录', exact: true })
    const refresh = async () => { await page.clock.runFor(1700) }
    const group = (record: ExternalRequestRecord) => table.locator(`tr[data-group-key=${JSON.stringify(fixtureGroupKey(record))}]`)
    const selectRequest = async (record: ExternalRequestRecord) => {
      const row = group(record)
      await row.locator('.monitor-ledger-group-select').click()
    }
    const expectArchivedTokens = async () => {
      await expect(chosen.locator('.monitor-request-live > div').filter({ has: page.locator('dt', { hasText: /^请求长度$/ }) }).locator('dd')).toHaveText('124')
      await expect(table.locator('[data-request-key="older"]').getByLabel('输入（含缓存） 100 Token', { exact: true })).toHaveText('100')
      await expect(table.locator('[data-request-key="older"]').getByLabel('缓存命中 20 Token', { exact: true })).toHaveText('2020.0%')
    }
    await expect(chosen.locator('.monitor-request-group')).toHaveAttribute('data-selected-request-key', 'latest')
    await expect.poll(async () => (await mini())?.metrics.average.value).toBe('60.00')
    await monitor.getByRole('button', { name: '小窗监控', exact: true }).click()
    expect(await page.evaluate(() => (window as unknown as { fixtureMiniOpens: number }).fixtureMiniOpens)).toBe(1)
    const beforeHistory = telemetryReads
    await selectRequest(older)
    await expectArchivedTokens()
    await expect(chosen.locator('.monitor-compact-identity')).toContainText('archived fixture')
    await page.clock.runFor(2200)
    await expect.poll(() => telemetryReads).toBeGreaterThan(beforeHistory + 1)
    expect((await mini()).instanceId).toBe('live')
    expect((await mini()).instances.map(item => item.id)).toEqual(['live'])
    expect((await mini()).metrics.average.value).toBe('60.00')
    expect((await mini()).metrics.first_output.value).toBe('0.08')
    await expect(monitor.locator('.monitor-selector')).toContainText('运行实例')
    await expect(monitor.locator('.monitor-selector')).toContainText('live fixture')
    await expect(monitor.locator('.monitor-selector')).not.toContainText('archived fixture')
    await monitor.getByRole('button', { name: '日志', exact: true }).click()
    await expect(page.getByRole('heading', { name: 'live fixture · 运行日志', exact: true })).toBeVisible()
    await expect(page.locator('.log-view')).toHaveText('fixture log')
    await page.getByRole('dialog', { name: 'live fixture · 运行日志', exact: true }).locator('.modal-footer').getByRole('button', { name: '关闭', exact: true }).click()
    await monitor.getByRole('button', { name: '停止', exact: true }).click()
    expect(stops).toEqual(['/api/instances/live/stop']); expect(logs).toContain('/api/instances/live/logs')
    latest = { ...record('new-latest', 640, 1), started_at: now + 1, usage: { ...latest.usage, prompt_tokens: 640, decode_tps: 61 } }
    await refresh()
    await expect(chosen.locator('.monitor-request-group')).toHaveAttribute('data-selected-request-key', 'older')
    await expect(group(latest)).toBeVisible()
    await expect(table.locator('[data-request-key="new-latest"]')).toBeVisible()
    await expect.poll(async () => (await mini()).metrics.average.value).toBe('61.00')
    await page.screenshot({ path: path.join(directory, 'unified-history-live-actions.png'), fullPage: true })
    await monitor.getByRole('button', { name: '刷新到最新', exact: true }).click()
    await expect(chosen.locator('.monitor-request-group')).toHaveAttribute('data-selected-request-key', 'new-latest')
    await expect(monitor.getByLabel('历史日期', { exact: true })).toHaveValue('today')
    await expect(monitor.getByLabel('历史实例', { exact: true })).toHaveValue('')
    removed = true; await refresh()
    await expect(monitor.locator('.monitor-selector')).toContainText('暂无运行模型')
    await expect(monitor.getByRole('button', { name: '停止', exact: true })).toHaveCount(0)
    await expect(chosen.locator('.monitor-request-group')).toHaveAttribute('data-selected-request-key', 'new-latest')
    await selectRequest(older)
    await expect(chosen.locator('.monitor-request-group')).toHaveAttribute('data-selected-request-key', 'older')
    await expectArchivedTokens()
    await page.setViewportSize({ width: 760, height: 1040 })
    await expect(monitor.locator('table.monitor-request-table')).toHaveCount(1)
    expect(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth)).toBe(false)
    await page.screenshot({ path: path.join(directory, 'unified-no-active-narrow.png'), fullPage: true })
    expect(errors).toEqual([]); expect(unexpected).toEqual([])
    console.log(JSON.stringify({ screenshots: directory }))
  } catch (issue) {
    await page.screenshot({ path: path.join(directory, 'failure.png'), fullPage: true })
    console.log(JSON.stringify({ errors, unexpected }))
    throw issue
  } finally {
    await browser.close()
    await new Promise<void>(resolve => server.httpServer.close(() => resolve()))
  }
})

