import { test, expect, chromium } from '@playwright/test'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { pinNavigation } from './navigation'
import { cyclePhase, cycleWindow, historyPage, historyUsage } from './monitor-history-fixture'

test('主监控：逐请求均值与固定实时窗口分开，阶段切换、缺测和短调用选择', async () => {
  test.setTimeout(120_000)
  const root = path.resolve(__dirname, '..'), artifacts = process.env.MODLOADER_TEST_ARTIFACTS || path.join(root, '.artifacts/codex/monitor-implementation-20261031/ui-check')
  fs.mkdirSync(artifacts, { recursive: true })
  const directory = fs.mkdtempSync(path.join(artifacts, 'phase-ui-'))
  const { preview } = await import('vite')
  const server = await preview({ root, configFile: false, build: { outDir: process.env.MODLOADER_TEST_DIST || path.join(root, 'dist') }, preview: { host: '127.0.0.1', port: 0 } })
  const address = server.httpServer.address()
  if (!address || typeof address === 'string') throw new Error('Fixture preview did not open')
  const browser = await chromium.launch({ channel: 'msedge', headless: true })
  const page = await browser.newPage({ viewport: { width: 1366, height: 900 } })
  const errors: string[] = [], unexpected: string[] = []
  let key = 'phase-a', client = 'turn-a', phase = 'prefill', sampleAt = Math.floor(Date.now() / 1000), current: number | null = 1800, mean = 1600
  let terminal: string | null = null, fontScale = 1, failed = false
  const instance = { id: 'phase-instance', model_id: 'fixture-model', profile_id: 'fixture-profile', alias: '阶段监控 fixture', backend: 'strata', state: 'ready', active_requests: 1, capabilities: {} }
  try {
    page.setDefaultTimeout(8000)
    page.on('pageerror', error => errors.push(error.message))
    await page.route('**/api/**', route => {
      const url = new URL(route.request().url())
      if (url.pathname.endsWith('/generation-events') && route.request().method() === 'GET') return route.fulfill({ contentType: 'text/event-stream', body: ': fixture idle\n\n' })
      if (route.request().method() !== 'GET') { unexpected.push(`${route.request().method()} ${url.pathname}`); return route.abort() }
      if (url.pathname === '/api/state') return route.fulfill({ json: { models: [], profiles: [], runtimes: [], downloads: [], installations: [], instances: [instance], settings: { font_scale: fontScale },
        backends: [{ id: 'strata', label: 'Strata', parameters: [] }], hardware: { cpu_info: { name: 'isolated fixture' }, gpus: [] } } })
      if (url.pathname === '/api/chat-presets') return route.fulfill({ json: { items: [], default_id: null } })
      if (url.pathname === '/api/conversations') return route.fulfill({ json: { items: [], active_id: null } })
      if (url.pathname === '/api/request-history/dates') return route.fulfill({ json: { current_run_id: 'phase-run', today: new Date().toLocaleDateString('en-CA'), dates: [], issues: [] } })
      if (url.pathname === '/api/request-history') return route.fulfill({ json: historyPage([], url.searchParams) })
      if (url.pathname === '/api/request-history/usage') return route.fulfill({ json: historyUsage() })
      unexpected.push(url.pathname)
      return route.fulfill({ status: 404, json: { detail: `Unexpected fixture: ${url.pathname}` } })
    })
    await page.route('**/api/instances/phase-instance/telemetry', route => {
      if (failed) return route.fulfill({ status: 500, json: { detail: '夹具采样失败' } })
      const prefill = phase === 'prefill', decode = ['thinking', 'output'].includes(phase)
      return route.fulfill({ json: { instance_id: instance.id, backend: 'strata', state: 'ready', updated_at: sampleAt, unavailable: [],
        live_generation: { scope: 'request', source: 'native fixture', request_key: terminal ? null : key, client_request_id: client, backend_running: terminal ? 0 : 1, backend_queued: 0,
          phase: terminal ? 'idle' : phase, available_for_chat: decode && current !== null, reason: null, tokens_per_second: decode ? current : null, average_tokens_per_second: decode ? mean : null, average_basis: decode ? 'backend_decode' : null, window_s: decode ? 3 : null,
          prefill_tokens_per_second: prefill ? current : null, average_prefill_tokens_per_second: prefill ? mean : null, prefill_average_basis: prefill ? 'backend_prefill' : null, prefill_window_s: prefill ? 3 : null,
          context_tokens: terminal ? null : 12100, context_capacity: 200000, context_basis: '输入 + 已生成', prompt_tokens: 12000, generated_tokens: decode ? 100 : 0 },
        groups: terminal ? [{ id: 'api_request', title: '最近请求', request_key: key, client_request_id: client, request_status: terminal, metrics: [
          { id: 'output_speed', label: 'Decode', value: mean, unit: 'token/s', kind: 'gauge', rate_basis: 'decode' },
          { id: 'prefill_speed', label: 'Prefill', value: 1600, unit: 'token/s', kind: 'gauge', rate_basis: 'prefill' },
          { id: 'prompt_tokens', label: '输入', value: 12000, kind: 'gauge' }, { id: 'completion_tokens', label: '输出', value: 100, kind: 'gauge' },
          { id: 'first_output_s', label: '首输出', value: .2, kind: 'gauge' }, { id: 'duration_s', label: '耗时', value: 2, kind: 'gauge' },
        ] }] : [] } })
    })
    await page.goto(`http://127.0.0.1:${address.port}`)
    await pinNavigation(page)
    await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '运行监控', exact: true }).click()
    const monitor = page.locator('.monitor-page'), chart = monitor.locator('.monitor-generation-chart'), speed = monitor.locator('.monitor-speed-current strong')
    const table = monitor.locator('.monitor-request-table'), windowSwitch = monitor.locator('button.monitor-window-cycle')
    const openGroup = async () => { const group = table.locator('tr[data-group-key]').first(); if (await group.getAttribute('data-expanded') !== 'true') await group.locator('td').last().click() }
    const pick = async (request: string) => { await openGroup(); await table.locator(`tr[data-request-key="${request}"]`).locator('td').last().click() }
    const pickTurn = () => table.locator('tr[data-group-key]').first().getByRole('button', { name: /^选择轮汇总/ }).click()
    const traceXs = async () => [...(await chart.locator('.trend-line').first().getAttribute('d') || '').matchAll(/[ML]([\d.]+),/g)].map(match => Number(match[1]))
    const refresh = async () => { sampleAt++; await monitor.getByRole('button', { name: '刷新到最新', exact: true }).click() }
    await pickTurn()
    await expect(chart).toHaveAttribute('data-mode', 'turn')
    await expect(chart).toHaveAttribute('data-phase', 'decode')
    await expect(speed).toHaveText('—')
    await openGroup()
    await expect(table.locator('tbody tr[data-request-index]')).toHaveCount(1)
    await cyclePhase(monitor, 'Prefill')
    await expect(speed).toHaveText('1,600.00')
    await expect(chart.locator('.monitor-request-point')).toHaveCount(1)
    await pick('phase-a')
    await expect(speed).toHaveText('1,800.00')
    await expect(windowSwitch).toHaveText('30s')
    await expect(chart.locator('.trend-axis')).toHaveText('30 秒前15 秒前现在')
    await expect(chart.locator('.trend-dot')).toHaveAttribute('cx', '600')
    await refresh()
    await expect(chart.locator('.trend-line')).toHaveCount(1)
    await expect(chart.locator('.trend-line')).toHaveCSS('stroke', 'rgb(24, 136, 120)')
    const prefillAxis = await chart.locator('.monitor-generation-tick').allTextContents()
    phase = 'thinking'; current = 145; mean = 132; await refresh()
    // User phase selection is stable; entering Decode does not replace Prefill.
    await expect(chart).toHaveAttribute('data-phase', 'prefill')
    await expect(speed).toHaveText('1,800.00')
    await cyclePhase(monitor, 'Decode')
    await expect(speed).toHaveText('145.00')
    expect(await chart.locator('.monitor-generation-tick').allTextContents()).not.toEqual(prefillAxis)
    phase = 'output'; current = 155; mean = 127.17; await refresh()
    await expect(speed).toHaveText('155.00')
    await expect(chart.locator('.trend-line')).toHaveCount(1)
    let xs = await traceXs()
    expect(xs).toHaveLength(2)
    expect(xs[1] - xs[0]).toBeCloseTo(600 / 30, 1)
    await refresh()
    await expect.poll(async () => (await traceXs()).length).toBe(3)
    xs = await traceXs()
    // One second always occupies the same width, even before the window is full.
    expect(xs[1] - xs[0]).toBeCloseTo(600 / 30, 1)
    expect(xs[2] - xs[1]).toBeCloseTo(600 / 30, 1)
    await monitor.locator('.monitor-compact-generation').screenshot({ path: path.join(directory, 'realtime-30s-short.png') })
    for (let index = 0; index < 62; index++) {
      current = 130 + index % 9 * 4
      await refresh()
      await expect(speed).toHaveText(`${current}.00`)
    }
    await expect.poll(async () => (await traceXs()).length).toBe(31)
    xs = await traceXs()
    expect(xs[1] - xs[0]).toBeCloseTo(600 / 30, 1)
    await cycleWindow(monitor, 10)
    await expect(chart.locator('.trend-axis')).toHaveText('10 秒前5 秒前现在')
    xs = await traceXs()
    expect(xs).toHaveLength(11)
    expect(xs[1] - xs[0]).toBeCloseTo(60, 1)
    // Changing the viewport never deletes older retained samples.
    await cycleWindow(monitor, 30)
    await windowSwitch.focus()
    await page.keyboard.press('Enter')
    await expect(windowSwitch).toHaveText('60s')
    xs = await traceXs()
    expect(xs).toHaveLength(61)
    expect(xs[1] - xs[0]).toBeCloseTo(600 / 60, 1)
    expect(xs[0]).toBe(0)
    expect(xs.at(-1)).toBe(600)
    await cyclePhase(monitor, 'Prefill')
    await expect(windowSwitch).toHaveText('60s')
    await expect(chart.locator('.trend-axis')).toHaveText('60 秒前30 秒前结束')
    await cyclePhase(monitor, 'Decode')
    await expect.poll(async () => (await traceXs()).length).toBe(61)
    await monitor.locator('.monitor-compact-generation').screenshot({ path: path.join(directory, 'realtime-60s-full.png') })
    const pathBeforeGap = await chart.locator('.trend-line').getAttribute('d')
    current = null; await refresh()
    await expect(speed).toHaveText('127.17')
    await expect(monitor.locator('.monitor-speed-note')).toHaveText('请求平均')
    expect(await chart.locator('.trend-line').getAttribute('d')).not.toBeNull()
    current = 0; await refresh()
    await expect(speed).toHaveText('0.00')
    await expect(chart.locator('.trend-dot')).toHaveCount(1)
    terminal = 'completed'; instance.active_requests = 0; mean = 125; await refresh()
    await expect(speed).toHaveText('0.00')
    await expect(chart.locator('.trend-axis')).toHaveText('60 秒前30 秒前结束')
    await pickTurn()
    await expect(windowSwitch).toBeHidden()
    await expect(speed).toHaveText('125.00')
    await openGroup()
    await expect(monitor.locator('.monitor-request-table tbody tr[data-request-index]')).toHaveCount(1)
    // The second call completed between polls. It has a mean point, but no
    // real-time samples; the first request still owns its actual sampled trace.
    key = 'phase-short'; mean = 300; await refresh()
    await openGroup()
    await expect(monitor.locator('.monitor-request-table tbody tr[data-request-index]')).toHaveCount(2)
    await pickTurn()
    await expect(chart.locator('.monitor-request-point')).toHaveCount(2)
    await pick('phase-a')
    await expect(chart).toHaveAttribute('data-request-key', 'phase-a')
    await expect(windowSwitch).toHaveText('60s')
    await expect(chart.locator('.trend-line')).toHaveCount(1)
    await expect(speed).toHaveText('0.00')
    await pick('phase-short')
    await expect(chart.locator('.monitor-generation-empty')).toHaveText('这次请求没有实时采样点')
    await expect(chart.locator('.trend-line, .trend-dot')).toHaveCount(0)
    await expect(speed).toHaveText('300.00')
    await expect(monitor.locator('.monitor-speed-note')).toHaveText('请求平均')
    await expect(chart.locator('.trend-axis')).toHaveText('60 秒前30 秒前结束')
    const tickFont = await chart.locator('.monitor-generation-tick').first().evaluate(node => parseFloat(getComputedStyle(node).fontSize))
    await pickTurn()
    fontScale = 1.3; await refresh()
    await expect.poll(() => chart.locator('.monitor-generation-tick').first().evaluate(node => parseFloat(getComputedStyle(node).fontSize))).toBeCloseTo(tickFont * 1.3, 1)
    await page.screenshot({ path: path.join(directory, 'request-means-and-history.png') })
    await pick('phase-short')
    await expect(windowSwitch).toHaveCSS('font-size', '16.9px') // 13px text control at 1.3 font scale.
    await monitor.locator('.monitor-compact-generation').screenshot({ path: path.join(directory, 'realtime-empty-large-font.png') })
    key = 'phase-new'; client = 'turn-new'; terminal = null; phase = 'waiting'; current = null; instance.active_requests = 1; await refresh()
    await openGroup()
    await expect(monitor.locator('.monitor-request-table tbody tr[data-request-index]')).toHaveCount(1)
    await expect(chart.locator('.trend-line,.trend-dot')).toHaveCount(0)
    failed = true; await refresh()
    await expect(monitor.getByText('夹具采样失败', { exact: true })).toBeVisible()
    await expect(monitor.getByRole('button', { name: '重试', exact: true })).toBeVisible()
    expect(errors).toEqual([])
    expect(unexpected).toEqual([])
    expect(pathBeforeGap).toBeTruthy()
    console.log('Request monitor fixture:', directory)
  } catch (issue) {
    await page.screenshot({ path: path.join(directory, 'failure.png'), fullPage: true })
    fs.writeFileSync(path.join(directory, 'failure-body.txt'), await page.locator('body').innerText())
    console.log(JSON.stringify({ errors, unexpected, directory }))
    throw issue
  } finally { await browser.close(); await new Promise<void>((resolve, reject) => server.httpServer.close(error => error ? reject(error) : resolve())) }
})


