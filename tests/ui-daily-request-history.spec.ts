import { chromium, expect, test } from '@playwright/test'
import * as fs from 'node:fs'
import * as path from 'node:path'
import type { ExternalHistorySnapshot, ExternalRequestRecord, LiveGeneration, RequestHistoryDates } from '../src/types'
import { historyInstanceIdentity, localDate, previousDate } from '../src/monitor-history-dates'
import { pinNavigation } from './navigation'
import { cyclePhase, cycleWindow, fixtureGroupKey, historyPage, historyParent, historyUsage } from './monitor-history-fixture'

test('单列表分组展开跨run/午夜/页，固定与升级稳定，唯一刷新解除选择且防旧响应复活', async () => {
  test.setTimeout(90_000)
  const root = path.resolve(__dirname, '..')
  const artifacts = process.env.MODLOADER_TEST_ARTIFACTS || path.join(root, '.artifacts/codex/monitor-implementation-20261031/ui-check')
  fs.mkdirSync(artifacts, { recursive: true })
  const directory = fs.mkdtempSync(path.join(artifacts, 'daily-ui-'))
  const { preview } = await import('vite')
  const server = await preview({ root, configFile: false, build: { outDir: process.env.MODLOADER_TEST_DIST || path.join(root, 'dist') },
    preview: { host: '127.0.0.1', port: 0, strictPort: false } })
  const address = server.httpServer.address()
  if (!address || typeof address === 'string') throw new Error('Fixture preview did not open')
  const browser = await chromium.launch({ channel: 'msedge', headless: true })
  const page = await browser.newPage({ viewport: { width: 1440, height: 1040 } })
  page.setDefaultTimeout(8000)
  await page.clock.install()
  const today = localDate(), yesterday = previousDate(today), midnight = new Date(`${today}T00:00:00`).getTime() / 1000, now = Date.now() / 1000
  const instance = { id: 'active', alias: '同名模型', backend: 'llama_cpp', model_id: 'm', profile_id: 'p', state: 'ready', active_requests: 0, capabilities: { input: ['text'] } }
  const record = (key: string, extra: Partial<ExternalRequestRecord> = {}): ExternalRequestRecord => ({ request_key: key, run_id: 'current', instance_id: instance.id,
    alias: instance.alias, backend: instance.backend, revision: 1, aggregation: 'request', turn_id: null, turn_state: null, labels: [], date: today,
    status: 'completed', started_at: now - 20, ended_at: (extra.started_at ?? now - 20) + 4, duration_s: 4, usage: { prompt_tokens: 100, completion_tokens: 20 }, timings: { predicted_n: 20, predicted_ms: 2000 }, ...extra })
  const a = record('A', { aggregation: 'turn', turn_id: 'main-T', turn_state: 'running', status: 'running', ended_at: undefined, usage: {}, timings: {} })
  const side = record('side', { started_at: now - 19 })
  const upgrade = record('upgrade', { started_at: now - 5, status: 'running', ended_at: undefined, usage: {}, timings: {} })
  const priorUpgrade = record('upgrade', { run_id: 'prior', started_at: now - 30, usage: { prompt_tokens: 8888, completion_tokens: 8 } })
  const before = record('midnight-before', { run_id: 'prior', instance_id: 'removed', alias: '跨午夜模型', aggregation: 'turn', turn_id: 'midnight-T', turn_state: 'completed',
    date: yesterday, started_at: midnight - 2, ended_at: midnight - 1, duration_s: 1, usage: { prompt_tokens: 101, completion_tokens: 11 }, timings: { predicted_n: 11, predicted_ms: 1000 } })
  const after = { ...before, request_key: 'midnight-after', date: today, started_at: midnight + 1, ended_at: midnight + 2,
    usage: { prompt_tokens: 202, completion_tokens: 22 }, timings: { predicted_n: 22, predicted_ms: 1000 } }
  const incomplete = record('partial', { run_id: 'prior', instance_id: 'removed', alias: '跨午夜模型', date: yesterday, started_at: midnight - 10,
    status: 'incomplete', record_incomplete: true, ended_at: undefined, usage: {}, timings: {} })
  const records = [a, side, upgrade, priorUpgrade, before, after, incomplete]
  let unit: ExternalHistorySnapshot = { run_id: 'current', aggregation: 'turn', turn_id: 'main-T', state: 'running', requests: [a] }
  let live: Partial<LiveGeneration> = { request_key: 'A', phase: 'output', backend_running: 1, tokens_per_second: 150 }, clock = now
  let allowInternal = false, internalCalls = 0
  let legacyKey: string | undefined
  let holdPrior = false, priorHeld = false, releasePrior: (() => void) | undefined
  const queries: URLSearchParams[] = [], errors: string[] = [], unexpected: string[] = []
  let usageRevision = 0
  const catalog = (): RequestHistoryDates => ({ current_run_id: 'current', usage_revision: String(usageRevision), today, issues: [], dates: [today, yesterday].map(date => {
    const items = records.filter(record => record.date === date)
    const instances = [...new Map(items.map(item => [historyInstanceIdentity({ run_id: item.run_id, id: item.instance_id }), {
      run_id: item.run_id, id: item.instance_id, alias: item.alias, backend: item.backend, started_at: item.run_id === 'current' ? now - 300 : now - 90000,
    }])).values()]
    return { date, request_count: items.length, instances }
  }) })
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/api/**', async route => {
    const url = new URL(route.request().url()), query = url.searchParams
    if (url.pathname.endsWith('/generation-events') && route.request().method() === 'GET') return route.fulfill({ contentType: 'text/event-stream', body: ': fixture idle\n\n' })
    if (allowInternal && url.pathname === '/api/chat' && route.request().method() === 'POST') {
      internalCalls++
      const headers = route.request().headers(), end = Date.now() / 1000
      return route.fulfill({ json: { choices: [{ message: { role: 'assistant', content: '内部完成。' }, finish_reason: 'stop' }],
        usage: { prompt_tokens: 2222, completion_tokens: 22 }, timings: { predicted_n: 22, predicted_ms: 1000 },
        modloader_request: { request_key: 'internal-fixture', client_call_id: headers['x-modloader-call-id'], client_request_id: headers['x-modloader-request-id'],
          status: 'completed', started_at: end - 1, ended_at: end, duration_s: 1, usage: { prompt_tokens: 2222, completion_tokens: 22 }, timings: { predicted_n: 22, predicted_ms: 1000 } } } })
    }
    if (allowInternal && url.pathname.startsWith('/api/conversations') && ['POST', 'PATCH'].includes(route.request().method())) {
      return route.fulfill({ json: { id: url.pathname.split('/').at(-1), created_at: new Date().toISOString(), updated_at: new Date().toISOString(), message_count: 2 } })
    }
    if (route.request().method() !== 'GET') { unexpected.push(`${route.request().method()} ${url.pathname}`); return route.abort() }
    if (url.pathname === '/api/state') return route.fulfill({ json: { models: [], profiles: [], runtimes: [], downloads: [], installations: [], instances: [instance], settings: {},
      backends: [{ id: 'llama_cpp', label: 'llama.cpp', parameters: [] }], hardware: { cpu_info: { name: 'fixture' }, gpus: [] } } })
    if (url.pathname === '/api/conversations') return route.fulfill({ json: { items: [], active_id: null } })
    if (url.pathname === '/api/chat-presets') return route.fulfill({ json: { items: [], default_id: null } })
    if (url.pathname.endsWith('/telemetry')) return route.fulfill({ json: { instance_id: instance.id, backend: instance.backend, state: 'ready', updated_at: clock,
      groups: legacyKey ? [{ id: 'api_request', label: 'API请求', metrics: [], request_key: legacyKey, request_status: 'running' }] : [], unavailable: [], external_history: unit, live_generation: { request_key: null, phase: 'idle', scope: 'request', source: 'fixture', backend_running: 0, backend_queued: 0,
        tokens_per_second: null, window_s: null, available_for_chat: false, reason: null, ...live } } })
    if (url.pathname === '/api/request-history/dates') return route.fulfill({ json: catalog() })
    if (url.pathname === '/api/request-history' || url.pathname === '/api/request-history/usage') {
      if (url.pathname === '/api/request-history') queries.push(new URLSearchParams(query))
      let items = [...records]
      if (query.has('run_id')) items = items.filter(item => item.run_id === query.get('run_id'))
      if (query.has('instance_id')) items = items.filter(item => item.instance_id === query.get('instance_id'))
      if (query.has('date_from')) items = items.filter(item => item.date! >= query.get('date_from')! && item.date! <= query.get('date_to')!)
      if (query.has('request_key')) items = items.filter(item => item.request_key === query.get('request_key'))
      if (query.has('turn_id')) items = items.filter(item => item.turn_id === query.get('turn_id'))
      items.sort((a, b) => query.has('turn_id') ? a.started_at! - b.started_at! : b.started_at! - a.started_at!)
      const offset = Number(query.get('offset') || 0), limit = Number(query.get('limit') || items.length)
      const result = historyPage(items, query, records)
      if (url.pathname === '/api/request-history/usage') return route.fulfill({ json: historyUsage(result.summary) })
      if (holdPrior && query.get('run_id') === 'prior' && query.get('request_key') === 'upgrade') { priorHeld = true; await new Promise<void>(resolve => { releasePrior = resolve }) }
      if (query.get('turn_id') === 'upgraded-T') await new Promise(resolve => setTimeout(resolve, 600))
      return route.fulfill({ json: result })
    }
    unexpected.push(url.pathname)
    return route.fulfill({ status: 404, json: { detail: `Unexpected fixture ${url.pathname}` } })
  })
  const screenshot = async (name: string) => { await page.evaluate(() => window.scrollTo(0, 0)); await page.screenshot({ path: path.join(directory, name), fullPage: true }) }
  try {
    await page.goto(`http://127.0.0.1:${address.port}`); await pinNavigation(page)
    await expect(page.getByRole('status', { name: '服务在线', exact: true })).toBeVisible()
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '运行监控', exact: true }).click()
    const monitor = page.locator('.monitor-page'), history = monitor.getByRole('region', { name: '请求监控', exact: true })
    const table = history.getByRole('region', { name: '请求记录', exact: true }), chosen = history.getByRole('region', { name: '监控生成统计', exact: true })
    const poll = async () => { await page.clock.runFor(1700) }
    const reset = () => monitor.getByRole('button', { name: '刷新到最新', exact: true }).click()
    const group = (turn: string) => table.locator(`tr.monitor-turn-group[data-turn-id="${turn}"]`)
    const pick = (key: string) => table.locator(`tr[data-request-key="${key}"] button`)
    const select = async (key: string, run = 'current') => {
      const record = records.find(record => record.request_key === key && record.run_id === run)!
      const row = table.locator(`tr[data-request-key="${key}"][data-request-run="${run}"]`)
      const parent = historyParent(table, record)
      await expect(parent).toBeVisible()
      if (await parent.getAttribute('data-expanded') !== 'true') await parent.locator('td').last().click()
      await expect(row).toBeVisible()
      await row.locator('td').last().click()
    }
    const window = chosen.locator('button.monitor-window-cycle')
    const picked = () => chosen.locator('.monitor-request-group')
    for (const label of ['调用历史', '查看本轮', '查看本次', '返回日期列表', '刷新记录', '返回最新', '跟随最新']) await expect(monitor.getByRole('button', { name: label, exact: true })).toHaveCount(0)
    await expect(monitor.locator('table.monitor-request-table')).toHaveCount(1)
    await expect(group('main-T')).toHaveAttribute('data-expanded', 'false')
    await select('A')
    await cycleWindow(chosen, 60)
    live.tokens_per_second = 160; clock++; await poll()
    await expect(chosen.locator('.monitor-speed-current')).toHaveText('160.00token/s')
    unit = { run_id: 'current', aggregation: 'request', turn_id: null, state: 'running', requests: [{ ...side, status: 'running' }] }
    legacyKey = 'A'; live = { request_key: side.request_key, phase: 'output', backend_running: 1, tokens_per_second: 900 }; clock++
    await poll(); await expect(picked()).toHaveAttribute('data-selected-request-key', 'side')
    unit = { run_id: 'current', aggregation: 'turn', turn_id: 'main-T', state: 'running', requests: [a] }
    live = { request_key: 'A', phase: 'output', backend_running: 1, tokens_per_second: 170 }; clock++
    await poll(); await expect(chosen.locator('.monitor-generation-chart')).toHaveAttribute('data-mode', 'realtime')
    await expect(window).toHaveText('60s')
    await expect(chosen.locator('.monitor-speed-current')).toHaveText('170.00token/s')
    await expect(chosen.locator('g[data-series="current"] .trend-line')).toHaveCount(1)
    await expect(chosen.locator('g[data-series="current"] .trend-dot')).toHaveCount(1)
    a.status = 'completed'; a.ended_at = now + 3; a.duration_s = 23; a.turn_state = 'waiting'; a.revision++; a.tool_names = ['mcp__web__search']
    const latest = record('latest', { started_at: now + 4 }); records.push(latest); usageRevision++
    unit = { run_id: 'current', aggregation: 'request', turn_id: null, state: 'completed', requests: [latest] }; live = {}; legacyKey = undefined; clock++
    await poll(); await expect(picked()).toHaveAttribute('data-selected-request-key', 'latest')
    await expect(history.getByLabel('历史日期', { exact: true })).toHaveValue('today')
    expect(queries.some(query => query.get('date_from') === today && query.get('date_to') === today && !query.has('run_id'))).toBe(true)
    const options = await history.getByLabel('历史实例').locator('option').allTextContents()
    expect(options.filter(label => label.startsWith('同名模型'))).toHaveLength(2)
    expect(new Set(options.filter(label => label.startsWith('同名模型'))).size).toBe(2)
    await expect(group('midnight-T').locator('[data-request-count]')).toHaveAttribute('aria-label', '请求数 2')
    await expect(group('midnight-T').locator('.monitor-ledger-matched')).toContainText('筛选内 1')
    await page.clock.runFor(5000)
    await expect(history.locator('[data-summary="count"]')).toHaveText('6')
    await expect(history.locator('[data-summary="input"]')).toHaveText('9,290')
    expect(queries.some(query => query.get('turn_id') === 'midnight-T')).toBe(false)
    await group('midnight-T').locator('.monitor-ledger-expand').click()
    await expect(group('midnight-T').locator('[data-request-count]')).toHaveAttribute('aria-label', '请求数 2')
    await expect(pick('midnight-before')).toBeVisible(); await expect(pick('midnight-after')).toBeVisible()
    await expect(picked()).toHaveAttribute('data-selected-request-key', 'latest')
    const full = queries.find(query => query.get('turn_id') === 'midnight-T')!
    expect(full.get('run_id')).toBe('prior'); expect(full.get('instance_id')).toBe('removed')
    expect(full.has('date_from')).toBe(false); expect(full.has('date_to')).toBe(false)
    await screenshot('single-list-cross-midnight.png')
    await group('midnight-T').locator('.monitor-ledger-expand').click()
    await expect(pick('midnight-before')).toHaveCount(0)
    await poll(); await expect(group('midnight-T')).toHaveAttribute('data-expanded', 'false')
    await history.getByLabel('历史日期', { exact: true }).selectOption('yesterday')
    await expect(group('midnight-T')).toBeVisible()
    await expect(history.locator('[data-summary="count"]')).toHaveText('2')
    await expect(history.locator('[data-summary="input"]')).toHaveText('101')
    await group('midnight-T').locator('.monitor-ledger-group-select').click()
    await expect(group('midnight-T').locator('[data-column="input"]')).toContainText('303')
    await expect(group('midnight-T')).toHaveAttribute('data-expanded', 'false')
    await group('midnight-T').locator('.monitor-ledger-expand').click()
    await expect(pick('midnight-after')).toBeVisible()
    await history.getByLabel('历史日期', { exact: true }).selectOption('date')
    await history.getByLabel('选择日期', { exact: true }).fill(today)
    await history.getByLabel('历史实例').selectOption(historyInstanceIdentity({ run_id: 'current', id: 'active' }))
    await expect(table.locator('[data-turn-run="prior"], [data-request-run="prior"]')).toHaveCount(0)
    await expect(picked()).toHaveAttribute('data-selected-instance', 'removed')
    await history.getByLabel('历史日期', { exact: true }).selectOption('range')
    await history.getByLabel('开始日期', { exact: true }).fill(yesterday)
    await expect(group('midnight-T')).toBeVisible()
    await history.getByLabel('结束日期', { exact: true }).fill(previousDate(yesterday))
    await expect(history.getByRole('alert')).toHaveText('结束日期不能早于开始日期')
    await history.getByLabel('结束日期', { exact: true }).fill(today)
    await expect(group('midnight-T')).toBeVisible()
    await screenshot('single-list-date-range.png')
    await history.getByLabel('历史日期', { exact: true }).selectOption('today')
    await select('A')
    await cycleWindow(chosen, 10)
    await expect(chosen.locator('g[data-series="current"] .trend-line')).toHaveCount(1)
    const curve = await chosen.locator('g[data-series="current"]').innerHTML()
    records.push(...Array.from({ length: 60 }, (_, index) => record(`new-${index}`, { started_at: now + 10 + index })))
    const newer = record('new-current', { started_at: now + 100 }); records.push(newer)
    unit = { run_id: 'current', aggregation: 'request', turn_id: null, state: 'completed', requests: [newer] }; clock++
    await poll(); await expect(pick('A')).toHaveCount(0)
    await expect(picked()).toHaveAttribute('data-selected-request-key', 'A')
    expect(await chosen.locator('g[data-series="current"]').innerHTML()).toBe(curve)
    await expect(window).toHaveText('10s')
    await screenshot('single-list-pinned-curve.png')
    await cyclePhase(chosen, 'Prefill')
    await history.getByRole('button', { name: '下一页', exact: true }).click()
    await select('upgrade')
    await chosen.locator('.monitor-compact-generation').evaluate(node => node.setAttribute('data-fixture-continuity', 'same-grid'))
    Object.assign(upgrade, { aggregation: 'turn', turn_id: 'upgraded-T', turn_state: 'waiting', status: 'completed', ended_at: now + 10, duration_s: 15,
      usage: { prompt_tokens: 111, completion_tokens: 11 }, timings: { predicted_n: 11, predicted_ms: 1000, prompt_n: 111, prompt_ms: 100 }, tool_names: ['functions.read_page'], revision: 2 })
    records.push(record('upgrade-continuation', { aggregation: 'turn', turn_id: 'upgraded-T', turn_state: 'waiting', started_at: now + 11 }))
    await poll(); await expect(chosen.locator('.monitor-generation-chart')).toHaveAttribute('data-group-key', fixtureGroupKey(upgrade))
    await expect(chosen.locator('.monitor-compact-generation')).toHaveAttribute('data-fixture-continuity', 'same-grid')
    await expect(chosen.locator('.monitor-generation-chart')).toHaveAttribute('data-mode', 'realtime')
    await expect(window).toHaveText('10s')
    await expect(chosen.locator('button.monitor-phase-toggle')).toHaveText('Prefill')
    await expect(group('upgraded-T').locator('[data-request-count]')).toHaveAttribute('aria-label', '请求数 2')
    await group('upgraded-T').locator('.monitor-ledger-expand').click()
    await expect(table.locator('[data-request-key="upgrade"][data-request-run="current"]')).toHaveAttribute('data-selected', 'true')
    await expect(table.locator('[data-request-key="upgrade"][data-request-run="current"] .monitor-ledger-request-select small')).toHaveText('functions.read_page')
    await expect(pick('upgrade-continuation')).toBeVisible()
    await group('upgraded-T').locator('.monitor-ledger-expand').click(); await poll()
    await expect(group('upgraded-T')).toHaveAttribute('data-expanded', 'false')
    expect(await table.ariaSnapshot()).not.toMatch(/请求\s*#\d+/)
    await screenshot('single-list-upgrade-preserved.png')
    holdPrior = true
    await select('upgrade', 'prior')
    await expect.poll(() => priorHeld).toBe(true)
    await history.getByLabel('历史日期', { exact: true }).selectOption('yesterday')
    await history.getByLabel('历史实例').selectOption(historyInstanceIdentity({ run_id: 'prior', id: 'removed' }))
    await reset(); releasePrior?.(); holdPrior = false
    await expect(picked()).toHaveAttribute('data-selected-request-key', 'new-current')
    await expect(history.getByLabel('历史日期', { exact: true })).toHaveValue('today')
    await expect(history.getByLabel('历史实例', { exact: true })).toHaveValue('')
    await expect(history.locator('.monitor-history-pagination > span')).toContainText('1–50 /')
    await poll(); await expect(picked()).toHaveAttribute('data-selected-request-key', 'new-current')
    await expect(monitor.locator('.monitor-selector')).toContainText('同名模型')
    await expect(monitor.locator('table.monitor-request-table')).toHaveCount(1)
    allowInternal = true
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '测试对话', exact: true }).click()
    await page.getByRole('textbox', { name: '对话消息', exact: true, includeHidden: true }).fill('完成内部I。')
    await page.getByRole('button', { name: '发送消息', exact: true }).click()
    await expect(page.locator('.chat-message.assistant .message-text')).toHaveText('内部完成。')
    await page.getByRole('button', { name: '查看本轮 1 次模型请求统计', exact: true }).click()
    const inspectedKey = await chosen.locator('.monitor-generation-chart').getAttribute('data-group-key')
    const inspectedParent = table.locator(`tr[data-group-key=${JSON.stringify(inspectedKey)}]`)
    if (await inspectedParent.getAttribute('data-expanded') !== 'true') await inspectedParent.locator('td').last().click()
    await expect(pick('internal-fixture')).toBeVisible()
    await expect(table.locator('[data-request-key="internal-fixture"] [data-column="input"] .monitor-ledger-value')).toHaveAttribute('aria-label', '输入（含缓存） 2,222 Token')
    await expect(chosen.locator('.monitor-generation-chart')).toHaveAttribute('data-mode', 'turn')
    await poll(); await expect(chosen.locator('.monitor-generation-chart')).toHaveAttribute('data-group-key', inspectedKey!)
    await expect(monitor.locator('table.monitor-request-table')).toHaveCount(1)
    // Returning from a hidden chat tab must realign headers over actual rows,
    // even when the first tbody child is now a spanning turn heading.
    await expect.poll(async () => table.locator('table').evaluate(node => {
      const heads = [...node.querySelectorAll('thead th')], cells = [...node.querySelector('tbody tr[data-group-key]')!.children]
      return heads.every((head, index) => Math.abs(head.getBoundingClientRect().x - cells[index].getBoundingClientRect().x) <= 1 && Math.abs(head.getBoundingClientRect().width - cells[index].getBoundingClientRect().width) <= 1)
    })).toBe(true)
    await screenshot('single-list-internal-inspection.png')
    expect(internalCalls).toBe(1); expect(errors).toEqual([]); expect(unexpected).toEqual([])
    console.log(JSON.stringify({ screenshots: directory }))
  } catch (issue) {
    fs.writeFileSync(path.join(directory, 'failure-body.txt'), await page.locator('body').innerText())
    console.log(JSON.stringify({ errors, unexpected, directory }))
    await page.screenshot({ path: path.join(directory, 'failure.png'), fullPage: true })
    throw issue
  } finally {
    releasePrior?.()
    await browser.close()
    await new Promise<void>((resolve, reject) => server.httpServer.close(error => error ? reject(error) : resolve()))
  }
})


