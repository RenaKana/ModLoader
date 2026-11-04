import { test, expect, chromium } from '@playwright/test'
import * as fs from 'node:fs'
import * as path from 'node:path'
import type { ExternalHistorySnapshot, ExternalRequestRecord, InternalRequestHistoryGroup, RequestHistoryGroup } from '../src/types'
import { emptyConversation } from '../src/chat-history'
import { turnTotals, type ModelRequestRecord } from '../src/chat-turn-metrics'
import { historyInstanceIdentity, localDate, previousDate } from '../src/monitor-history-dates'
import { pinNavigation } from './navigation'
import { historyPage, historyStats, historyUsage } from './monitor-history-fixture'
import { externalRequestIdentity as fixtureGroupKey } from '../src/monitor-generation'

test('旧后端外部请求监控：拆开归组档案、逐请求分页和固定选择，保留内置轮次', async () => {
  test.setTimeout(90_000)
  const root = path.resolve(__dirname, '..')
  const artifacts = process.env.MODLOADER_TEST_ARTIFACTS || path.join(root, '.artifacts/codex/monitor-implementation-20261031/ui-check')
  fs.mkdirSync(artifacts, { recursive: true })
  const directory = fs.mkdtempSync(path.join(artifacts, 'history-ui-'))
  const { preview } = await import('vite')
  const server = await preview({ root, configFile: false,
    build: { outDir: process.env.MODLOADER_TEST_DIST || path.join(root, 'dist') },
    preview: { host: '127.0.0.1', port: 0, strictPort: false } })
  const address = server.httpServer.address()
  if (!address || typeof address === 'string') throw new Error('Fixture server did not open a port')
  const browser = await chromium.launch({ channel: 'msedge', headless: true })
  const page = await browser.newPage({ viewport: { width: 1440, height: 1040 } })
  page.setDefaultTimeout(8000)
  await page.clock.install()
  const now = Date.now() / 1000
  const today = localDate(), yesterday = previousDate(today)
  const instance = { id: 'live-model', alias: 'Qwen fixture', model_id: 'fixture-model', profile_id: 'fixture-profile', backend: 'llama_cpp',
    state: 'ready', active_requests: 0, capabilities: { input: ['text'] } }
  const record = (key: string, extra: Partial<ExternalRequestRecord> = {}): ExternalRequestRecord => ({ request_key: key,
    run_id: 'current', instance_id: instance.id, alias: instance.alias, backend: instance.backend, revision: 1, aggregation: 'request',
    turn_id: null, turn_state: null, labels: [], date: today, status: 'completed', started_at: now - 20, ended_at: (extra.started_at ?? now - 20) + 4, duration_s: 4,
    first_output_s: .5, usage: { prompt_tokens: 100, completion_tokens: 20 }, timings: { predicted_n: 20, predicted_ms: 2000 }, ...extra })
  const ordinary = record('ordinary', { labels: ['tool_calls'] })
  const summary = record('summary-request', { alias: 'conversation summary', labels: ['tool_calls'], tool_names: ['compress_context'], first_output_s: .75,
    usage: { prompt_tokens: 8000, completion_tokens: 400, prompt_tokens_details: { cached_tokens: 2000 } },
    timings: { prompt_n: 6000, prompt_ms: 1500, predicted_n: 400, predicted_ms: 2000 } })
  const turn = [record('turn-one', { aggregation: 'turn', turn_id: 'tool-turn', turn_state: 'waiting', labels: ['tool_calls'],
    tool_names: ['mcp__web__search', 'functions.read_page', 'mcp__web__search'], started_at: now - 30, ended_at: now - 26 }),
    record('turn-two', { aggregation: 'turn', turn_id: 'tool-turn', turn_state: 'waiting', started_at: now - 25, ended_at: now - 21,
      usage: { prompt_tokens: 200, completion_tokens: 60 }, timings: { predicted_n: 60, predicted_ms: 3000 } })]
  const pinned = record('pinned', { status: 'running', started_at: now - 10, ended_at: undefined, duration_s: .1, first_output_s: null, usage: {}, timings: {} })
  const archive = record('archive-partial', { run_id: 'prior', instance_id: 'removed-model', alias: '已卸载模型', backend: 'sparkinfer',
    date: yesterday, status: 'incomplete', record_incomplete: true, started_at: now - 86400, ended_at: undefined, first_output_s: null, usage: {}, timings: {} })
  const old = Array.from({ length: 49 }, (_, index) => record(`old-${index}`, { started_at: now - 100 - index }))
  const latest = record('new-latest', { usage: { prompt_tokens: 777, completion_tokens: 33 } })
  const internal = emptyConversation({ profile_id: instance.profile_id, model_id: instance.model_id, alias: instance.alias })
  const internalRequests = [[], ['手动压缩']].map<ModelRequestRecord>((labels, index) => ({ id: `internal-compaction-${index}`, index: index + 1,
    kind: 'compaction', status: 'completed', startedAt: (now - 60 + index * 3) * 1000, endedAt: (now - 58 + index * 3) * 1000,
    elapsed: 2, labels, usage: { prompt_tokens: 100, completion_tokens: 20 }, timings: { predicted_n: 20, predicted_ms: 2000 } }))
  internal.title = '明确压缩回归'; internal.message_count = 1
  internal.messages = [{ id: 'internal-message', role: 'assistant', text: '已完成明确压缩。', images: [], createdAt: (now - 55) * 1000, status: 'completed',
    model: instance.alias, profileId: instance.profile_id, modelId: instance.model_id, instanceId: instance.id, requests: internalRequests,
    activity: [{ id: 'internal-tool', kind: 'tool', requestId: 'internal-compaction-1', status: 'completed', elapsed: .2,
      call: { id: 'clock', type: 'function', function: { name: 'current_time', arguments: '{"time_zone":"Asia/Shanghai"}' } }, result: '{"time":"fixture"}' }],
    turn: turnTotals(internalRequests, (now - 60) * 1000, 5, 'completed', (now - 55) * 1000) }]
  const internalCounters = internalRequests.map(request => record(request.id, { usage: request.usage!, timings: request.timings!, first_output_s: null }))
  const internalGroup: InternalRequestHistoryGroup = { ...historyStats(internalCounters), source: 'internal', conversation_id: internal.id,
    message_id: 'internal-message', instance_id: instance.id, alias: instance.alias, backend: instance.backend,
    started_at: now - 60, ended_at: now - 55, elapsed_s: 5, status: 'completed', requests: internalRequests }
  let current: ExternalHistorySnapshot = { run_id: 'current', aggregation: 'request', turn_id: null, state: 'completed', requests: [ordinary] }
  let removed = false, manyNew = false, readError = false
  let continuation: ExternalRequestRecord | null = null
  const errors: string[] = [], unexpected: string[] = []
  const selectedReads: string[] = []
  const selectedTurns: string[] = []
  const screenshot = async (name: string) => {
    await page.evaluate(() => window.scrollTo(0, 0))
    await page.screenshot({ path: path.join(directory, name), fullPage: true })
  }
  const currentItems = () => manyNew
    ? [...Array.from({ length: 60 }, (_, index) => record(`new-${index}`, { started_at: now + index })), ...(continuation ? [continuation] : []), pinned, ...turn, ordinary, summary, ...old]
    : [pinned, ...turn, ordinary, summary, ...old]
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/api/**', async route => {
    const url = new URL(route.request().url()), pathname = url.pathname
    if (pathname.endsWith('/generation-events') && route.request().method() === 'GET') return route.fulfill({ contentType: 'text/event-stream', body: ': fixture idle\n\n' })
    if (route.request().method() !== 'GET') { unexpected.push(`${route.request().method()} ${pathname}`); return route.abort() }
    if (pathname === '/api/state') return route.fulfill({ json: { models: [], profiles: [], runtimes: [], downloads: [], installations: [], instances: removed ? [] : [instance],
      settings: {}, backends: [{ id: 'llama_cpp', label: 'llama.cpp', parameters: [] }], hardware: { cpu_percent: 10, cpu_info: { name: 'CPU fixture' }, gpus: [] } } })
    if (pathname === '/api/chat-presets') return route.fulfill({ json: { items: [], default_id: null } })
    if (pathname === '/api/conversations') return route.fulfill({ json: { items: [{ id: internal.id, title: internal.title, created_at: internal.created_at,
      updated_at: internal.updated_at, message_count: internal.message_count }], active_id: internal.id } })
    if (pathname === `/api/conversations/${internal.id}`) return route.fulfill({ json: internal })
    if (pathname === `/api/instances/${instance.id}/telemetry`) return route.fulfill({ json: { instance_id: instance.id, backend: instance.backend,
      state: 'ready', updated_at: await page.evaluate(() => Date.now() / 1000), unavailable: [], groups: [], external_history: current,
      live_generation: { request_key: null, scope: 'request', source: 'fixture', backend_running: 0, backend_queued: 0,
        available_for_chat: false, tokens_per_second: null, window_s: null, phase: 'idle', reason: null } } })
    if (pathname === '/api/request-history/dates') return route.fulfill({ json: { current_run_id: 'current', today, issues: [], dates: [
      { date: today, request_count: currentItems().length + internalRequests.length, instances: [
        { source: 'external', run_id: 'current', id: instance.id, alias: instance.alias, backend: instance.backend, started_at: now - 300 },
        { source: 'internal', id: instance.id, alias: instance.alias, backend: instance.backend, started_at: internalGroup.started_at, conversation_id: internal.id },
      ] },
      { date: yesterday, request_count: 1, instances: [{ run_id: 'prior', id: archive.instance_id, alias: archive.alias, backend: archive.backend, started_at: now - 90000 }] },
    ] } })
    if (pathname === '/api/request-history' || pathname === '/api/request-history/usage') {
      // Explicitly emulate the running old gateway, which ignores this selector.
      url.searchParams.delete('external_group_by')
      if (readError) return route.fulfill({ status: 500, json: { detail: '读取记录失败：fixture-history.jsonl' } })
      let items = [...currentItems(), archive]
      if (url.searchParams.get('source') === 'internal') items = []
      if (url.searchParams.has('run_id')) items = items.filter(item => item.run_id === url.searchParams.get('run_id'))
      if (url.searchParams.has('date_from')) items = items.filter(item => item.date! >= url.searchParams.get('date_from')! && item.date! <= url.searchParams.get('date_to')!)
      if (url.searchParams.has('instance_id')) items = items.filter(item => item.instance_id === url.searchParams.get('instance_id'))
      if (url.searchParams.has('turn_id')) { selectedTurns.push(url.searchParams.get('turn_id')!); items = items.filter(item => item.turn_id === url.searchParams.get('turn_id')).sort((a, b) => a.started_at! - b.started_at!) }
      if (url.searchParams.has('request_key')) { selectedReads.push(url.searchParams.get('request_key')!); items = items.filter(item => item.request_key === url.searchParams.get('request_key')) }
      const offset = Number(url.searchParams.get('offset') || 0), limit = url.searchParams.has('limit') ? Number(url.searchParams.get('limit')) : items.length
      const includeInternal = (pathname === '/api/request-history/usage' || url.searchParams.get('group_by') === 'turn') && !url.searchParams.has('run_id') && !url.searchParams.has('turn_id') && !url.searchParams.has('request_key')
        && url.searchParams.get('source') !== 'external' && (!url.searchParams.has('instance_id') || url.searchParams.get('instance_id') === instance.id)
        && (!url.searchParams.has('date_from') || today >= url.searchParams.get('date_from')! && today <= url.searchParams.get('date_to')!)
      if (pathname === '/api/request-history/usage') return route.fulfill({ json: historyUsage(historyStats([...items, ...(includeInternal ? internalCounters : [])])) })
      if (!includeInternal) return route.fulfill({ json: historyPage(items, url.searchParams, [...currentItems(), archive]) })
      const completeQuery = new URLSearchParams(url.searchParams)
      completeQuery.set('offset', '0'); completeQuery.set('limit', String(items.length + 1))
      const external = historyPage(items, completeQuery, [...currentItems(), archive])
      const mixed: { item: ExternalRequestRecord; group: RequestHistoryGroup | InternalRequestHistoryGroup; internal: boolean }[] = external.items.map((item, index) => ({ item, group: external.groups![index], internal: false }))
      mixed.push({ item: internalCounters[0], group: internalGroup, internal: true })
      const selected = mixed.sort((a, b) => b.group.started_at - a.group.started_at).slice(offset, offset + limit)
      return route.fulfill({ json: { run_id: null, items: selected.filter(entry => !entry.internal).map(entry => entry.item),
        groups: selected.filter(entry => !entry.internal).map(entry => entry.group), internal_groups: selected.some(entry => entry.internal) ? [internalGroup] : [],
        group_total: mixed.length, total: external.total + internalRequests.length, summary: historyStats([...items, ...internalCounters]), offset, limit, issues: [] } })
    }
    unexpected.push(pathname)
    return route.fulfill({ status: 404, json: { detail: `Unexpected fixture request: ${pathname}` } })
  })
  try {
    await page.goto(`http://127.0.0.1:${address.port}`)
    await pinNavigation(page)
    await expect(page.getByRole('status', { name: '服务在线', exact: true })).toBeVisible()
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '运行监控', exact: true }).click()
    const monitor = page.locator('.monitor-page')
    const history = monitor.getByRole('region', { name: '请求监控', exact: true })
    const table = history.getByRole('region', { name: '请求记录', exact: true })
    const chosen = history.getByRole('region', { name: '监控生成统计', exact: true })
    const refresh = async () => { await page.clock.runFor(1700) }
    const child = (key: string) => table.locator(`tr[data-request-key="${key}"]`)
    const reveal = async (_record: ExternalRequestRecord) => {}
    const select = async (record: ExternalRequestRecord) => { await child(record.request_key).locator('td').last().click() }
    await test.step('普通外部请求直接一行显示，缺省指标不填零', async () => {
      await expect(child('ordinary')).toBeVisible()
      await expect(child('ordinary').locator('.monitor-ledger-expand')).toHaveCount(0)
      await reveal(ordinary)
      await expect(monitor.locator('.monitor-generation-chart')).toHaveCount(1)
      await expect(monitor.getByRole('group', { name: '图表范围', exact: true })).toHaveCount(0)
      await expect(monitor).not.toContainText('工具调用')
      const row = child('ordinary')
      await expect(row.locator('.monitor-ledger-group-meta')).toHaveCount(1)
      await expect(row.locator('[data-column="input"] .monitor-ledger-value')).toHaveText('100')
      await expect(row.locator('[data-column="input"] .monitor-ledger-value')).toHaveAttribute('aria-label', '输入（含缓存） 100 Token')
      await expect(row.getByLabel('Prefill 未知', { exact: true })).toHaveText('PF —')
      current = { ...current, state: 'running', requests: [pinned] }
      await refresh()
      await reveal(pinned)
      await expect(child('pinned').getByRole('img', { name: '等待响应', exact: true })).toBeVisible()
      await expect(child('pinned').locator('[data-column="output"] .monitor-ledger-value')).toHaveText('—')
    })
    await test.step('未知摘要仍为普通计量记录，工具名不作为压缩证据', async () => {
      current = { run_id: 'current', aggregation: 'request', turn_id: null, state: 'completed', requests: [summary] }
      await refresh()
      await reveal(summary)
      const row = child('summary-request')
      await expect(row.locator('.monitor-ledger-group-meta').nth(1)).toHaveText('compress_context')
      for (const [metric, value] of [['input', '8,000'], ['output', '400']]) {
        await expect(row.locator(`[data-column="${metric}"] .monitor-ledger-value`)).toHaveText(value)
      }
      await expect(row.locator('[data-column="cached"] .monitor-ledger-value')).toContainText('2,000')
      await expect(row.getByLabel('首字 0.75 秒', { exact: true })).toHaveText('首字 0.75')
      await expect(row.getByLabel('Decode 200.00 token/s', { exact: true })).toHaveText('200.00')
      await expect(row.getByLabel('Prefill 4,000.00 token/s', { exact: true })).toHaveText('PF 4,000.00')
      await expect(monitor.locator('.monitor-generation-chart')).toHaveCount(1)
      await expect(row).not.toContainText(/自动压缩|手动压缩|工具调用/)
      await screenshot('external-summary-metered.png')
    })
    await test.step('旧工具轮次拆为已完成请求，各自时钟冻结且不伪造工具执行', async () => {
      current = { run_id: 'current', aggregation: 'turn', turn_id: 'tool-turn', state: 'waiting', requests: turn }
      await refresh(); await select(turn[0])
      await expect(child('turn-one').getByRole('img', { name: '已完成', exact: true })).toBeVisible()
      await expect(child('turn-two').getByRole('img', { name: '已完成', exact: true })).toBeVisible()
      await expect(table.locator('.monitor-ledger-expand')).toHaveCount(1) // Internal turn only.
      await expect(chosen.locator('.monitor-compact-generation')).toHaveAttribute('data-scope', 'request')
      const duration = chosen.locator('.monitor-request-live > div').filter({ has: page.getByText('耗时', { exact: true }) }).locator('dd')
      await expect(duration).toHaveText('4s'); await page.clock.runFor(1200); await expect(duration).toHaveText('4s')
      await expect(monitor).not.toContainText('待续调')
      await expect(monitor).not.toContainText('工具执行')
      await expect(child('turn-one').locator('.monitor-ledger-group-meta').nth(1)).toHaveText('mcp__web__search · functions.read_page')
      await expect(child('turn-one').getByRole('button')).toHaveAccessibleName(/选择请求 · Qwen fixture · /)
      await expect(child('turn-two').getByRole('button')).toHaveAccessibleName(/选择请求 · Qwen fixture · /)
      await screenshot('external-completed-requests.png')
    })
    await test.step('内置明确压缩标签与实时轮次编号保留', async () => {
      await page.setViewportSize({ width: 1440, height: 1040 })
      await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '测试对话', exact: true }).click()
      await page.getByRole('button', { name: '查看本轮 2 次模型请求统计', exact: true }).click()
      const internalKey = JSON.stringify(['internal', internal.id, 'internal-message', instance.id])
      await expect(chosen.locator('.monitor-generation-chart')).toHaveAttribute('data-group-key', internalKey)
      const internalParent = table.locator(`tr[data-group-key=${JSON.stringify(internalKey)}]`)
      if (await internalParent.getAttribute('data-expanded') !== 'true') await internalParent.locator('td').last().click()
      await expect(child('internal-compaction-0').locator('.monitor-ledger-request-select small')).toHaveText('自动压缩')
      await expect(child('internal-compaction-1').locator('.monitor-ledger-request-select small')).toHaveText('手动压缩')
      await expect(table.locator('[data-request-key="internal-compaction-0"] button')).toBeVisible()
      await expect(table.locator('[data-request-key="internal-compaction-1"] button')).toBeVisible()
      await expect(internalParent.locator('[data-column="input"]')).toHaveText('200')
      await expect(internalParent.locator('[data-column="output"]')).toHaveText('40')
      await refresh()
      await expect(internalParent.locator('[data-request-count]')).toHaveAttribute('aria-label', '请求数 2')
      await expect(table.locator('[data-request-key="internal-compaction-0"]')).toBeVisible()
      await expect(monitor.locator('table.monitor-request-table')).toHaveCount(1)
      const details = table.locator('.monitor-request-detail'), point = chosen.locator('.monitor-request-point').last()
      await expect(details).toHaveCount(0)
      await point.click()
      await expect(details).toHaveAttribute('data-open', 'true')
      await expect(chosen.locator('.monitor-generation-chart')).toHaveAttribute('data-request-key', 'internal-compaction-1')
      await child('internal-compaction-0').locator('td').last().click()
      await expect(details).toHaveCount(0)
      await child('internal-compaction-1').locator('td').last().click()
      await expect(details).toHaveAttribute('data-open', 'true')
      await screenshot('internal-explicit-compaction.png')
    })
    await test.step('请求终态无假等待；刷新恢复网关最新快照', async () => {
      current = { run_id: 'current', aggregation: 'request', turn_id: null, state: 'completed', requests: [ordinary] }
      await page.reload()
      await pinNavigation(page)
      await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '运行监控', exact: true }).click()
      await reveal(ordinary)
      await expect(child('ordinary')).toBeVisible()
      await expect(chosen.locator('.monitor-compact-generation')).toHaveAttribute('data-state', 'idle')
      await expect(chosen.locator('.monitor-generation-chart')).toHaveAttribute('data-mode', 'external-realtime')
      await expect(monitor.locator('.monitor-generation-chart')).toHaveCount(1)
      await expect(child('ordinary').getByRole('img', { name: '已完成', exact: true })).toBeVisible()
    })
    await expect(monitor.getByRole('button', { name: '调用历史', exact: true })).toHaveCount(0)
    const expectNoHistoryNumbers = async () => {
      await expect(table).not.toContainText(/请求\s*#\d+/)
      expect(await table.ariaSnapshot()).not.toMatch(/请求\s*#\d+/)
    }
    await test.step('旧后端兼容页按外部请求与内置轮次计行，选择只查看该请求', async () => {
      await expect(history.getByLabel('历史日期', { exact: true })).toHaveValue('today')
      await expect(history.locator('.monitor-history-pagination > span')).toHaveText('1–50 / 55 条 · 56 次请求')
      await expect(history.locator('[data-summary="count"]')).toHaveText('56')
      await expectNoHistoryNumbers()
      const pinnedRow = child('pinned')
      await expect(pinnedRow.locator('time')).toHaveAttribute('datetime', new Date((pinned.started_at ?? 0) * 1000).toISOString())
      await expect(pinnedRow.getByRole('img', { name: '等待响应', exact: true })).toBeVisible()
      await history.getByRole('button', { name: '下一页', exact: true }).click()
      await expect(table.locator('tr[data-group-key]')).toHaveCount(5)
      await expect(history.locator('.monitor-history-pagination > span')).toHaveText('51–55 / 55 条 · 56 次请求')
      await expectNoHistoryNumbers()
      await history.getByRole('button', { name: '上一页', exact: true }).click()
      await select(turn[1])
      await expect(chosen.locator('.monitor-speed-current')).toHaveText('20.00token/s')
      await expect(chosen.locator('.monitor-request-group')).toHaveAttribute('data-selected-request-key', 'turn-two')
      await expect(chosen.locator('.monitor-generation-chart')).toHaveAttribute('data-mode', 'realtime')
      await expect(child('turn-two')).toHaveAttribute('data-selected', 'true')
      await select(turn[0])
      await expect(chosen.locator('.monitor-speed-current')).toHaveText('10.00token/s')
      await expect(child('turn-one')).toHaveAttribute('data-selected', 'true')
      expect(selectedTurns).toEqual([])
      await screenshot('history-request-selection.png')
    })
    await test.step('大量新调用与实例移除不抢走所选请求，独立按request_key取得终态', async () => {
      await select(pinned)
      await expect(chosen.locator('.monitor-request-group')).toHaveAttribute('data-selected-request-key', 'pinned')
      manyNew = true; removed = true
      pinned.status = 'completed'; pinned.ended_at = now + 1; pinned.duration_s = 11
      pinned.usage = { prompt_tokens: 99, completion_tokens: 17 }; pinned.revision++
      await refresh()
      await expect(child('new-59')).toBeVisible()
      await expect(table.locator('[data-request-key="pinned"]')).toHaveCount(0)
      await expectNoHistoryNumbers()
      await expect(chosen.locator('.monitor-compact-identity')).toContainText('已完成')
      await expect(chosen.locator('.monitor-request-live > div').filter({ has: page.getByText('输出', { exact: true }) }).locator('dd')).toHaveText('17')
      expect(selectedReads).toContain('pinned')
      await expect(history.getByLabel('历史日期', { exact: true })).toHaveValue('today')
      await screenshot('history-pinned.png')
    })
    await test.step('旧后端后来写入turn_id仍保持同一个请求选择，不再请求整组', async () => {
      pinned.aggregation = 'turn'; pinned.turn_id = 'detected-turn'; pinned.turn_state = 'waiting'; pinned.labels = ['tool_calls']; pinned.tool_names = ['functions.read_page']; pinned.revision++
      continuation = record('pinned-continuation', { aggregation: 'turn', turn_id: 'detected-turn', turn_state: 'waiting', started_at: now + 2, ended_at: now + 6 })
      await refresh()
      await expect(chosen.locator('.monitor-generation-chart')).toHaveAttribute('data-group-key', fixtureGroupKey(pinned))
      await expect(child('pinned')).toHaveCount(0)
      await history.getByRole('button', { name: '下一页', exact: true }).click()
      await expect(child('pinned')).toHaveAttribute('data-selected', 'true')
      await expect(child('pinned-continuation')).toHaveAttribute('data-selected', 'false')
      await expect(child('pinned').locator('.monitor-ledger-group-meta').nth(1)).toHaveText('functions.read_page')
      await expect(child('pinned').getByRole('img', { name: '已完成', exact: true })).toBeVisible()
      await expect(child('pinned-continuation').getByRole('img', { name: '已完成', exact: true })).toBeVisible()
      expect(selectedTurns).toEqual([])
      await expect(history.getByLabel('历史日期', { exact: true })).toHaveValue('today')
    })
    await test.step('旧运行和已移除实例使用快照；缺尾记录不假称请求失败', async () => {
      await history.getByLabel('历史日期', { exact: true }).selectOption('yesterday')
      await expect(history.getByLabel('历史实例').locator('option')).toContainText(['全部实例', '已卸载模型 · sparkinfer'])
      await history.getByLabel('历史实例').selectOption(historyInstanceIdentity({ run_id: 'prior', id: 'removed-model' }))
      await select(archive)
      await expect(chosen.locator('.monitor-compact-identity')).toContainText('记录未完整')
      await expect(child('archive-partial').getByRole('img', { name: '记录未完整', exact: true })).toBeVisible()
      await expect(chosen).not.toContainText('请求失败')
      await expect(chosen.locator('.monitor-generation-chart')).toHaveCount(1)
      await page.setViewportSize({ width: 760, height: 1040 })
      await screenshot('history-prior-narrow.png')
    })
    await test.step('具体读取失败可重试，返回最新恢复当前监控', async () => {
      readError = true
      await history.getByLabel('历史实例').selectOption('')
      await expect(history.getByText('读取记录失败：fixture-history.jsonl', { exact: true }).first()).toBeVisible()
      await expect(chosen.locator('.monitor-request-group')).toHaveAttribute('data-selected-request-key', 'archive-partial')
      readError = false; removed = false
      current = { run_id: 'current', aggregation: 'request', turn_id: null, state: 'completed', requests: [latest] }
      await history.locator(':scope > .error-notice').getByRole('button', { name: '重试', exact: true }).click()
      await expect(history.getByText('读取记录失败：fixture-history.jsonl', { exact: true })).toHaveCount(0)
      await expect(chosen.locator('.monitor-request-group')).toHaveAttribute('data-selected-request-key', 'archive-partial')
      await expect(history.getByLabel('历史日期', { exact: true })).toHaveValue('yesterday')
      await monitor.getByRole('button', { name: '刷新到最新', exact: true }).click()
      await expect(history).toHaveCount(1)
      await expect(history.getByLabel('历史日期', { exact: true })).toHaveValue('today')
      await expect(chosen.locator('.monitor-request-group')).toHaveAttribute('data-selected-request-key', 'new-latest')
      await expect(chosen.locator('.monitor-request-live > div').filter({ has: page.getByText('请求长度', { exact: true }) }).locator('dd')).toHaveText('810')
      await expect(monitor.locator('table.monitor-request-table')).toHaveCount(1)
    })
    expect(errors).toEqual([])
    expect(unexpected).toEqual([])
    console.log(JSON.stringify({ screenshots: directory }))
  } catch (issue) {
    fs.writeFileSync(path.join(directory, 'failure-body.txt'), await page.locator('body').innerText())
    console.log(JSON.stringify({ pageErrors: errors, unexpected, directory }))
    await page.screenshot({ path: path.join(directory, 'failure.png'), fullPage: true })
    throw issue
  } finally {
    await browser.close()
    await new Promise<void>((resolve, reject) => server.httpServer.close(error => error ? reject(error) : resolve()))
  }
})

