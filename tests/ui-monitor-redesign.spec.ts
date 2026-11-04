import { test, expect, chromium, type Locator, type Page, type TestInfo } from '@playwright/test'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { emptyConversation } from '../src/chat-history'
import { turnTotals, type ModelRequestRecord } from '../src/chat-turn-metrics'
import { localDate } from '../src/monitor-history-dates'
import { externalRequestIdentity } from '../src/monitor-generation'
import { externalTurnIdentity } from '../src/monitor-request-groups'
import type { ExternalRequestRecord, InternalRequestHistoryGroup, RequestHistoryGroup, RequestHistoryStats } from '../src/types'
import { pinNavigation } from './navigation'
import { historyUsage } from './monitor-history-fixture'

const root = path.resolve(__dirname, '..')
const artifacts = process.env.MODLOADER_TEST_ARTIFACTS || path.join(root, '.artifacts/codex/monitor-implementation-20261031/ui-check')
const groupKey = (record: ExternalRequestRecord) => record.turn_id ? externalTurnIdentity(record) : externalRequestIdentity(record)

// Fixture arithmetic is deliberately independent of the production aggregation helper.
function stats(records: ExternalRequestRecord[]): RequestHistoryStats {
  const input = records.reduce((sum, record) => sum + (record.usage.prompt_tokens ?? 0), 0)
  const output = records.reduce((sum, record) => sum + (record.usage.completion_tokens ?? 0), 0)
  const cached = records.reduce((sum, record) => sum + (record.usage.prompt_tokens_details?.cached_tokens ?? 0), 0)
  const count = records.length
  const status_counts = { running: 0, completed: 0, failed: 0, cancelled: 0, incomplete: 0 }
  records.forEach(record => status_counts[record.status]++)
  return { request_count: count, input: { value: input, reported: count }, output: { value: output, reported: count },
    total: { value: input + output, reported: count }, cached: { value: cached, reported: count },
    cache_pair: { input, cached, reported: count }, cache_ratio: input ? cached / input : null,
    decode: { value: count ? output / count : null, tokens: output, seconds: count, reported: count },
    prefill: { value: count ? input / count : null, tokens: input, seconds: count, reported: count },
    ttft: { value: count ? .4 : null, reported: count, engine_reported: 0, gateway_reported: count },
    status_counts, record_incomplete_count: 0 }
}

async function fixture(testInfo: TestInfo, options: { twoInstances?: boolean } = {}) {
  fs.mkdirSync(artifacts, { recursive: true })
  const directory = fs.mkdtempSync(path.join(artifacts, 'redesign-'))
  const { preview } = await import('vite')
  const server = await preview({ root, configFile: false,
    build: { outDir: process.env.MODLOADER_TEST_DIST || path.join(root, 'dist') },
    preview: { host: '127.0.0.1', port: 0, strictPort: false } })
  const address = server.httpServer.address()
  if (!address || typeof address === 'string') throw new Error('Fixture preview did not bind a local port')
  const browser = await chromium.launch({ channel: 'msedge', headless: true })
  const context = await browser.newContext({ viewport: { width: 1366, height: 768 }, colorScheme: 'light', reducedMotion: 'reduce' })
  const page = await context.newPage()
  page.setDefaultTimeout(8000)
  await context.tracing.start({ screenshots: true, snapshots: true })
  await page.clock.install()
  const now = Math.floor(Date.now() / 1000), today = localDate()
  const instance = { id: 'redesign-model', model_id: 'redesign-asset', profile_id: 'redesign-profile', alias: '当前轮',
    backend: 'llama_cpp', state: 'ready', active_requests: 1, capabilities: { input: ['text'] } }
  const otherInstance = { ...instance, id: 'redesign-other', profile_id: 'redesign-other-profile', alias: '另一实例当前轮' }
  const record = (key: string, prompt: number, completion: number, cached: number, extra: Partial<ExternalRequestRecord> = {}): ExternalRequestRecord => ({
    request_key: key, run_id: 'redesign-run', instance_id: instance.id, alias: instance.alias, backend: instance.backend,
    revision: 1, aggregation: 'request', turn_id: null, turn_state: null, labels: [], date: today, status: 'completed',
    started_at: now - 20, ended_at: (extra.started_at ?? now - 20) + 2, duration_s: 2, first_output_s: .4,
    usage: { prompt_tokens: prompt, completion_tokens: completion, prompt_tokens_details: { cached_tokens: cached } },
    timings: { prompt_n: prompt, prompt_ms: 1000, predicted_n: completion, predicted_ms: 1000 }, ...extra })
  let current = [record('current-1', 1000, 100, 800, { aggregation: 'turn', turn_id: 'current-turn', turn_state: 'running', started_at: now - 4 }),
    record('current-2', 2000, 200, 1500, { aggregation: 'turn', turn_id: 'current-turn', turn_state: 'running',
      status: 'running', started_at: now - 2, ended_at: undefined })]
  const archived = [record('old-1', 300, 30, 120, { alias: '固定历史轮', aggregation: 'turn', turn_id: 'old-turn', turn_state: 'completed', started_at: now - 30 }),
    record('old-2', 400, 40, 200, { alias: '固定历史轮', aggregation: 'turn', turn_id: 'old-turn', turn_state: 'completed', started_at: now - 28 })]
  const single = record('single', 500, 50, 250, { alias: '单次轮', started_at: now - 32 })
  const noSamples = record('no-samples', 600, 60, 300, { alias: '无采样历史', started_at: now - 35 })
  // 55 groups span more than one default 50-group page. The summary must include the second page.
  const filler = Array.from({ length: 51 }, (_, index) => record(`filler-${index}`, 100, 10, 50,
    { alias: `保留模型 ${String(index + 1).padStart(2, '0')}`, started_at: now - 40 - index * 3 }))
  let otherCurrent = [record('other-current-1', 1000, 100, 500, { instance_id: otherInstance.id, alias: otherInstance.alias,
    aggregation: 'turn', turn_id: 'other-turn', turn_state: 'running', started_at: now - 4 }),
  record('other-current-2', 2000, 200, 1000, { instance_id: otherInstance.id, alias: otherInstance.alias,
    aggregation: 'turn', turn_id: 'other-turn', turn_state: 'running', status: 'running', started_at: now - 2, ended_at: undefined })]
  const otherStale = record('other-stale', 600, 60, 300, { instance_id: otherInstance.id, alias: '另一实例旧记录',
    status: 'running', started_at: now - 60, ended_at: undefined })
  const allRecords = () => [...current, ...archived, single, noSamples, ...filler, ...(options.twoInstances ? [...otherCurrent, otherStale] : [])]
  const chatRequests: ModelRequestRecord[] = [0, 1].map(index => ({ id: `chat-explicit-${index}`, index: index + 1, kind: 'generation', status: 'completed',
    labels: [],
    startedAt: (now - 25 + index * 3) * 1000, endedAt: (now - 23 + index * 3) * 1000, elapsed: 2, firstOutput: .4,
    usage: { prompt_tokens: 3000 + index * 1000, completion_tokens: 300 + index * 100, prompt_tokens_details: { cached_tokens: 1000 } },
    timings: { prompt_n: 3000 + index * 1000, prompt_ms: 1000, predicted_n: 300 + index * 100, predicted_ms: 1000 } }))
  const conversation = emptyConversation({ profile_id: instance.profile_id, model_id: instance.model_id, alias: instance.alias })
  conversation.title = '明确指定统计轮'; conversation.message_count = 1
  conversation.messages = [{ id: 'explicit-chat-answer', role: 'assistant', text: '这条回复有明确的两次模型请求。', images: [], createdAt: (now - 25) * 1000,
    status: 'completed', model: '内置测试轮', profileId: instance.profile_id, modelId: instance.model_id, instanceId: instance.id,
    requests: chatRequests, turn: turnTotals(chatRequests, (now - 25) * 1000, 5, 'completed', (now - 20) * 1000) }]
  const internalCounters = chatRequests.map(request => record(request.id, request.usage!.prompt_tokens!, request.usage!.completion_tokens!, 1000))
  const internalGroup: InternalRequestHistoryGroup = { ...stats(internalCounters), source: 'internal', conversation_id: conversation.id,
    message_id: 'explicit-chat-answer', instance_id: instance.id, alias: '内置测试轮', backend: instance.backend,
    started_at: now - 25, ended_at: now - 20, elapsed_s: 5, status: 'completed', requests: chatRequests }
  const errors: string[] = [], unexpected: string[] = [], historyQueries: URLSearchParams[] = [], telemetryReads: string[] = []
  let sample = 0, next = 2, hardwareSample = 0
  page.on('pageerror', error => errors.push(error.message))
  await context.route('**/api/**', route => {
    const request = route.request(), url = new URL(request.url()), pathname = url.pathname
    if (pathname.endsWith('/generation-events') && request.method() === 'GET') return route.fulfill({ contentType: 'text/event-stream', body: ': fixture idle\n\n' })
    if (request.method() !== 'GET') { unexpected.push(`${request.method()} ${pathname}`); return route.abort() }
    if (pathname === '/api/state') return route.fulfill({ json: { models: [], profiles: [], runtimes: [], downloads: [], installations: [],
      instances: [instance, ...(options.twoInstances ? [otherInstance] : [])], settings: {}, backends: [{ id: instance.backend, label: 'llama.cpp', parameters: [] }],
      hardware: { sampled_at: now + hardwareSample++, cpu_info: { name: '隔离 CPU fixture' }, cpu_percent: 12, cpu_per_core: [10, 14], cpu_count: 2,
        cpu_power_w: 42, cpu_temperature_c: 55, cpu_temperature_status: 'ready',
        gpus: [{ id: 'redesign-gpu', name: '隔离 GPU fixture', utilization: 24, power: 90, temperature: 48, pcie_rx_bytes_s: 1e8, pcie_tx_bytes_s: 2e8 }] } } })
    if (pathname === '/api/chat-presets') return route.fulfill({ json: { items: [], default_id: null } })
    if (pathname === '/api/conversations') return route.fulfill({ json: { items: [conversation], active_id: conversation.id } })
    if (pathname === `/api/conversations/${conversation.id}`) return route.fulfill({ json: conversation })
    if (pathname === `/api/instances/${instance.id}/telemetry` || options.twoInstances && pathname === `/api/instances/${otherInstance.id}/telemetry`) {
      sample++
      const isOther = pathname.includes(`/${otherInstance.id}/`), selectedInstance = isOther ? otherInstance : instance, records = isOther ? otherCurrent : current
      telemetryReads.push(selectedInstance.id)
      const latest = records.at(-1)!
      return route.fulfill({ json: { instance_id: selectedInstance.id, backend: instance.backend, state: 'ready', updated_at: now + sample / 2,
        unavailable: [], groups: [], external_history: { run_id: 'redesign-run', aggregation: 'turn', turn_id: isOther ? 'other-turn' : 'current-turn', state: 'running', requests: records },
        live_generation: { request_key: latest.request_key, scope: 'request', source: 'isolated fixture', phase: 'output',
          backend_running: 1, backend_queued: 0, available_for_chat: true, tokens_per_second: 140 + sample,
          average_tokens_per_second: 120, average_basis: 'backend_decode', window_s: 3, first_output_s: .4,
          request_elapsed_s: 2 + sample / 2, prompt_tokens: latest.usage.prompt_tokens, generated_tokens: (latest.usage.completion_tokens ?? 0) + sample,
          prefill_tokens_per_second: null, average_prefill_tokens_per_second: 1000, prefill_average_basis: 'backend_prefill' } } })
    }
    if (pathname === '/api/request-history/dates') return route.fulfill({ json: { current_run_id: 'redesign-run', today, issues: [], dates: [
      { date: today, request_count: allRecords().length + chatRequests.length, instances: [
        { source: 'external', run_id: 'redesign-run', id: instance.id, alias: instance.alias, backend: instance.backend, started_at: now - 400 },
        ...(options.twoInstances ? [{ source: 'external', run_id: 'redesign-run', id: otherInstance.id, alias: otherInstance.alias, backend: otherInstance.backend, started_at: now - 300 }] : []),
        { source: 'internal', id: instance.id, conversation_id: conversation.id, alias: internalGroup.alias, backend: instance.backend, started_at: internalGroup.started_at },
      ] },
    ] } })
    if (pathname === '/api/request-history' || pathname === '/api/request-history/usage') {
      if (pathname === '/api/request-history') historyQueries.push(new URLSearchParams(url.searchParams))
      let items = url.searchParams.get('source') === 'internal' ? [] : allRecords()
      for (const key of ['run_id', 'instance_id', 'request_key', 'turn_id'] as const) if (url.searchParams.has(key)) items = items.filter(item => item[key] === url.searchParams.get(key))
      if (url.searchParams.has('date_from')) items = items.filter(item => item.date! >= url.searchParams.get('date_from')! && item.date! <= url.searchParams.get('date_to')!)
      const includeInternal = !url.searchParams.has('run_id') && !url.searchParams.has('request_key') && !url.searchParams.has('turn_id')
        && url.searchParams.get('source') !== 'external' && (!url.searchParams.has('instance_id') || url.searchParams.get('instance_id') === instance.id)
        && (!url.searchParams.has('date_from') || today >= url.searchParams.get('date_from')! && today <= url.searchParams.get('date_to')!)
      const summary = stats([...items, ...(includeInternal ? internalCounters : [])]), total = items.length + (includeInternal ? chatRequests.length : 0)
      if (pathname === '/api/request-history/usage') return route.fulfill({ json: historyUsage(summary) })
      const offset = Number(url.searchParams.get('offset') || 0), limit = Number(url.searchParams.get('limit') || 50)
      if (url.searchParams.get('group_by') === 'turn') {
        const groups = new Map<string, ExternalRequestRecord[]>()
        ;[...items].sort((a, b) => b.started_at! - a.started_at!).forEach(item => { const key = groupKey(item); groups.set(key, [...(groups.get(key) ?? []), item]) })
        const entries = [...groups.values()]
        const metadata = entries.map<RequestHistoryGroup>(records => {
          const representative = records[0], started = Math.min(...records.map(item => item.started_at!))
          const running = records.some(item => item.status === 'running'), ended = running ? null : Math.max(...records.map(item => item.ended_at ?? item.started_at! + item.duration_s))
          return { ...stats(records), run_id: representative.run_id, instance_id: representative.instance_id, turn_id: representative.turn_id,
            request_key: representative.request_key, aggregation: representative.aggregation, started_at: started, ended_at: ended,
            elapsed_s: (ended ?? now + sample / 2) - started, status: running ? 'running' : 'completed' }
        })
        const mixed = metadata.map((group, index) => ({ started: group.started_at, index }))
        if (includeInternal) mixed.push({ started: internalGroup.started_at, index: -1 })
        const selected = mixed.sort((a, b) => b.started - a.started).slice(offset, offset + limit)
        return route.fulfill({ json: { run_id: null, items: selected.filter(item => item.index >= 0).map(item => entries[item.index][0]),
          groups: selected.filter(item => item.index >= 0).map(item => metadata[item.index]), internal_groups: selected.some(item => item.index === -1) ? [internalGroup] : [],
          group_total: mixed.length, total, summary, offset, limit, issues: [] } })
      }
      items = [...items].sort((a, b) => url.searchParams.has('turn_id') ? a.started_at! - b.started_at! : b.started_at! - a.started_at!)
      return route.fulfill({ json: { run_id: null, items: items.slice(offset, offset + limit), total, summary, offset, limit, issues: [] } })
    }
    unexpected.push(pathname)
    return route.fulfill({ status: 404, json: { detail: `Unexpected isolated fixture request: ${pathname}` } })
  })
  const navigation = page.getByRole('navigation', { name: '主导航' }), monitor = page.locator('.monitor-page')
  const table = monitor.locator('table.monitor-request-table')
  const parent = (alias: string) => table.locator('tr[data-group-key]:not([data-request-key])').filter({ hasText: alias })
  const child = (key: string, instanceId = instance.id) => table.locator(`tr[data-request-key="${key}"][data-request-run="redesign-run"][data-request-instance="${instanceId}"]`)
  const chart = monitor.locator('.monitor-generation-chart')
  const advance = async () => {
    const previous = current.at(-1)!
    current = current.map(record => record.request_key === previous.request_key ? { ...record, status: 'completed', ended_at: now + next,
      duration_s: now + next - record.started_at!, revision: record.revision + 1 } : record)
    next++
    const latest = record(`current-${next}`, 2000 + next * 100, 200 + next * 10, 1500,
      { status: 'running', aggregation: 'turn', turn_id: 'current-turn', turn_state: 'running', started_at: now + next - 3, ended_at: undefined })
    current.push(latest)
    await page.clock.runFor(1700)
    await expect(child(latest.request_key).or(parent('当前轮'))).not.toHaveCount(0)
    return latest.request_key
  }
  const selectChild = async (alias: string, key: string, instanceId = instance.id) => {
    await expect(parent(alias)).toBeVisible()
    if (await parent(alias).getAttribute('data-expanded') !== 'true') await parent(alias).locator('td').last().click()
    await expect(child(key, instanceId)).toBeVisible()
    await child(key, instanceId).locator('td').last().click()
    await expect(chart).toHaveAttribute('data-request-key', key)
  }
  const advanceOther = async () => {
    const next = otherCurrent.length + 1, previous = otherCurrent.at(-1)!
    otherCurrent = otherCurrent.map(record => record.request_key === previous.request_key ? { ...record, status: 'completed',
      ended_at: now + next, duration_s: now + next - record.started_at!, revision: record.revision + 1 } : record)
    const latest = record(`other-current-${next}`, 2000, 200, 1000, { instance_id: otherInstance.id, alias: otherInstance.alias,
      aggregation: 'turn', turn_id: 'other-turn', turn_state: 'running', status: 'running', started_at: now + next - 3, ended_at: undefined })
    otherCurrent.push(latest)
    await page.clock.runFor(1700)
    return latest.request_key
  }
  const screenshot = async (name: string) => {
    await page.evaluate(() => { if (document.activeElement instanceof HTMLElement) document.activeElement.blur(); scrollTo(0, 0) })
    await page.mouse.move(1, 1)
    await page.screenshot({ path: path.join(directory, name), animations: 'disabled' })
  }
  await page.goto(`http://127.0.0.1:${address.port}`)
  await pinNavigation(page)
  await expect(page.getByRole('status', { name: '服务在线', exact: true })).toBeVisible()
  await navigation.getByRole('button', { name: '运行监控', exact: true }).click()
  return { page, monitor, table, chart, navigation, parent, child, selectChild, advance, screenshot, historyQueries,
    expectedSummary: () => stats([...allRecords(), ...internalCounters]), instance, otherInstance, otherStale, advanceOther, telemetryReads, archived, internalGroup, directory,
    async close() {
      await screenshot('final-state.png').catch(() => {})
      await context.tracing.stop({ path: path.join(directory, 'trace.zip') })
      await browser.close()
      await new Promise<void>((resolve, reject) => server.httpServer.close(error => error ? reject(error) : resolve()))
      console.log(JSON.stringify({ artifacts: directory, errors, unexpected }))
    },
    verifyNoLeaks() { expect(errors).toEqual([]); expect(unexpected).toEqual([]) },
  }
}

async function expectPlain(button: Locator) {
  await expect(button.locator('svg,i,img')).toHaveCount(0)
  const style = await button.evaluate(node => {
    const css = getComputedStyle(node)
    return { border: ['Top', 'Right', 'Bottom', 'Left'].map(side => parseFloat(css[`border${side}Width` as keyof CSSStyleDeclaration] as string)),
      background: css.backgroundColor, decoration: css.textDecorationLine, image: css.backgroundImage }
  })
  expect(style.border).toEqual([0, 0, 0, 0])
  expect(style.background).toBe('rgba(0, 0, 0, 0)')
  expect(style.decoration).toBe('none')
  expect(style.image).toBe('none')
}

async function anchor(button: Locator) {
  return button.evaluate(node => {
    const box = node.getBoundingClientRect()
    let x = box.x, y = box.y
    // Row clicks may scroll the page into view; compare layout, not the viewport.
    for (let parent = node.parentElement; parent; parent = parent.parentElement) { x += parent.scrollLeft; y += parent.scrollTop }
    return { x, y, width: box.width, height: box.height }
  })
}

test('监控重构：统一父子摘要、独立行/名称操作、全筛选统计、纯文字控件与真实空图', async ({}, testInfo) => {
  test.setTimeout(90_000)
  const f = await fixture(testInfo)
  try {
    await f.page.setViewportSize({ width: 1366, height: 900 })
    const phase = f.monitor.locator('button.monitor-phase-toggle')
    await expect(phase, 'Decode/Prefill is one cycling text button, not the old segmented pair').toHaveCount(1)
    await expect(phase).toHaveText('Decode')
    await expectPlain(phase)
    await expect(f.monitor.getByRole('group', { name: '图表范围', exact: true })).toHaveCount(0)
    await expect.poll(() => f.historyQueries.some(query => query.get('group_by') === 'turn')).toBe(true)
    const summary = f.monitor.getByRole('region', { name: '筛选范围汇总', exact: true })
    const expected = f.expectedSummary()
    for (const [key, value] of Object.entries({ count: expected.request_count, input: expected.input.value!, output: expected.output.value!, total: expected.total.value!, cached: expected.cached.value! })) {
      await expect(summary.locator(`[data-summary="${key}"]`)).toContainText(value.toLocaleString('en-US'))
    }
    await expect(summary.locator('[data-summary="hit"]')).toContainText(`${(expected.cache_ratio! * 100).toFixed(1)}%`)
    await expect(f.table.getByRole('columnheader', { name: '状态', exact: true })).toHaveCount(0)
    await expect(f.table.getByRole('columnheader', { name: '请求', exact: true })).toHaveCount(0)
    const headings = await f.table.locator('thead th').allTextContents()
    expect(headings[0]).toMatch(/模型/); expect(headings.at(-1)).toMatch(/时间/)
    await expect(f.parent('单次轮')).toHaveCount(1)
    await expect(f.parent('单次轮')).toContainText('1')
    await expect(f.table.locator('tr[data-request-key]')).toHaveCount(0)
    const layout = await f.page.evaluate(() => {
      const box = (selector: string) => { const node = document.querySelector(selector); if (!node) return null; const bounds = node.getBoundingClientRect(); return { top: bounds.top, height: bounds.height, bottom: bounds.bottom } }
      const fullyVisible = (selector: string) => [...document.querySelectorAll(selector)].filter(node => {
        const bounds = node.getBoundingClientRect()
        let top = 0, bottom = innerHeight
        for (let parent = node.parentElement; parent; parent = parent.parentElement) if (/(auto|scroll|hidden)/.test(getComputedStyle(parent).overflowY)) {
          const clip = parent.getBoundingClientRect(); top = Math.max(top, clip.top); bottom = Math.min(bottom, clip.bottom)
        }
        return bounds.height > 0 && bounds.top >= top && bounds.bottom <= bottom
      }).length
      return { header: box('.app-header'), pageHeader: box('.monitor-page .page-header'), selector: box('.monitor-selector'),
        filter: box('.monitor-history-controls'), summary: box('.monitor-history-summary'), selectionHeading: box('.monitor-selection-heading'),
        chart: box('.monitor-compact-generation'), table: box('.monitor-request-table'), ledger: box('.monitor-request-ledger'), tableHead: box('.monitor-request-table thead'),
        visibleParents: fullyVisible('.monitor-request-table tr[data-group-key]:not([data-request-key])'),
        visibleChildren: fullyVisible('.monitor-request-table tr[data-request-key]') }
    })
    fs.writeFileSync(path.join(f.directory, 'initial-layout.json'), JSON.stringify(layout, null, 2))
    console.log(JSON.stringify({ initialLayout: layout }))
    await f.screenshot('1366x900-initial.png')
    expect.soft(layout.visibleParents, '1366×900 keeps three full 80px summaries below the 190px chart').toBeGreaterThanOrEqual(3)
    expect(layout.summary!.top).toBeGreaterThan(layout.ledger!.bottom)
    const textSize = await f.parent('单次轮').locator('[data-column="input"]').evaluate(node => parseFloat(getComputedStyle(node).fontSize))
    expect(textSize, 'Primary values use the enlarged readable type scale').toBeGreaterThanOrEqual(14)
    const graphHeight = await f.chart.locator('svg').evaluate(node => node.getBoundingClientRect().height)
    expect(graphHeight).toBe(190)
    await f.page.setViewportSize({ width: 1024, height: 768 })
    await f.screenshot('1024x768-initial.png')
    await f.page.setViewportSize({ width: 1366, height: 768 })

    await test.step('父行非名称仅展开，名称仅选择；单次请求也有父子结构', async () => {
      const before = await f.chart.getAttribute('data-group-key')
      await f.parent('固定历史轮').locator('td').last().click()
      await expect(f.child('old-1')).toBeVisible(); await expect(f.child('old-2')).toBeVisible()
      await expect(f.chart).toHaveAttribute('data-group-key', before!)
      await f.parent('固定历史轮').getByRole('button', { name: /^选择轮汇总/ }).click()
      await expect(f.chart).toHaveAttribute('data-group-key', groupKey(f.archived[0]))
      await expect(f.chart).toHaveAttribute('data-mode', 'turn')
      await expect(f.child('old-1')).toBeVisible()
      await f.selectChild('固定历史轮', 'old-2')
      await f.parent('固定历史轮').locator('td').last().click()
      await expect(f.child('old-2')).toHaveCount(0)
      await expect(f.chart).toHaveAttribute('data-request-key', 'old-2')
      await f.parent('固定历史轮').getByRole('button', { name: /^选择轮汇总/ }).click()
      await expect(f.chart).toHaveAttribute('data-mode', 'turn')
      await expect(f.child('old-2')).toHaveCount(0)
      await f.parent('单次轮').locator('td').last().click()
      await expect(f.child('single')).toBeVisible()
      await f.parent('单次轮').getByRole('button', { name: /^选择轮汇总/ }).click()
      await expect(f.chart).toHaveAttribute('data-mode', 'turn')
      await expect(f.child('single')).toBeVisible()
    })

    await test.step('窗口紧邻阶段右侧；轮时隐藏留位，点击/Enter保留请求选择', async () => {
      const turnPhase = await anchor(phase)
      await f.selectChild('当前轮', 'current-2')
      const window = f.monitor.locator('button.monitor-window-cycle')
      await expect(window).toBeVisible(); await expectPlain(window)
      const requestPhase = await anchor(phase), windowBox = await anchor(window)
      for (const key of ['x', 'y', 'width', 'height'] as const) expect(Math.abs(turnPhase[key] - requestPhase[key]), `phase.${key} must not move on scope change`).toBeLessThanOrEqual(1)
      expect(requestPhase.x + requestPhase.width).toBeLessThanOrEqual(windowBox.x + 1)
      await expect(phase.locator('xpath=following-sibling::*[1]')).toHaveClass('monitor-window-slot')
      for (const seconds of [60, 10, 30]) { await window.click(); await expect(window).toHaveText(new RegExp(String(seconds))) }
      await phase.click(); await expect(f.chart).toHaveAttribute('data-phase', 'prefill')
      await expect(f.chart).toHaveAttribute('data-request-key', 'current-2')
      await phase.focus(); await phase.press('Enter'); await expect(f.chart).toHaveAttribute('data-phase', 'decode')
      await expect(f.chart).toHaveAttribute('data-request-key', 'current-2')
      await expectPlain(phase)
      await f.screenshot('1366x768-current-request.png')
      await f.page.setViewportSize({ width: 1024, height: 768 })
      await f.screenshot('1024x768-current-request.png')
      await f.page.setViewportSize({ width: 320, height: 768 })
      await f.page.evaluate(() => scrollTo(0, 0))
      for (const button of [phase, window]) {
        const box = await button.boundingBox()
        expect(box).not.toBeNull(); expect(box!.x).toBeGreaterThanOrEqual(0); expect(box!.x + box!.width).toBeLessThanOrEqual(320)
      }
      for (const selector of ['.monitor-compact-heading', '.monitor-compact-identity']) {
        const box = await f.monitor.locator(selector).boundingBox()
        expect(box).not.toBeNull(); expect(box!.x).toBeGreaterThanOrEqual(0); expect(box!.x + box!.width).toBeLessThanOrEqual(320)
      }
      await f.screenshot('320x768-controls.png')
      await f.page.setViewportSize({ width: 1366, height: 768 })
      await f.parent('当前轮').getByRole('button', { name: /^选择轮汇总/ }).click()
      await expect(window).toBeHidden()
      const restored = await anchor(phase)
      for (const key of ['x', 'y', 'width', 'height'] as const) expect(Math.abs(turnPhase[key] - restored[key])).toBeLessThanOrEqual(1)
    })

    await test.step('历史均值不伪装成真实采样', async () => {
      await f.selectChild('无采样历史', 'no-samples')
      await expect(f.chart).toHaveAttribute('data-mode', 'realtime')
      await expect(f.chart.locator('.trend-line,.trend-dot,.monitor-request-point')).toHaveCount(0)
      await expect(f.chart.locator('.monitor-generation-empty')).toHaveText('这次请求没有实时采样点')
      await expect(f.monitor.locator('.monitor-speed-note')).toHaveText('请求平均')
      await f.screenshot('1366x768-history-no-samples.png')
      await f.page.setViewportSize({ width: 1024, height: 768 })
      await f.screenshot('1024x768-history-no-samples.png')
    })
    await test.step('长列表内滚动保留硬件入口，GPU/CPU原趋势仍可展开', async () => {
      await f.page.setViewportSize({ width: 1366, height: 768 })
      const ledger = f.monitor.locator('.monitor-request-ledger')
      const clip = await ledger.evaluate(node => ({ height: node.clientHeight, scroll: node.scrollHeight }))
      expect(clip.scroll).toBeGreaterThan(clip.height)
      expect(clip.height).toBeGreaterThanOrEqual(340); expect(clip.height).toBeLessThanOrEqual(560)
      const hardware = f.monitor.getByRole('region', { name: '硬件读数', exact: true })
      await hardware.scrollIntoViewIfNeeded()
      await expect(hardware).toBeInViewport()
      for (const device of ['GPU', 'CPU']) {
        const control = hardware.getByRole('button', { name: `${device}整体趋势`, exact: true })
        await control.click()
        const detail = f.monitor.getByRole('region', { name: `${device}整体趋势详情`, exact: true })
        await expect(detail).toBeVisible()
        await expect(detail.getByRole('img', { name: new RegExp(`^${device}整体趋势最近 60 秒趋势`) })).toBeVisible()
        await detail.getByRole('button', { name: '收起', exact: true }).click()
        await expect(control).toBeFocused()
      }
      await hardware.scrollIntoViewIfNeeded()
      await f.page.screenshot({ path: path.join(f.directory, '1366x768-hardware-reachable.png'), animations: 'disabled' })
    })
    f.verifyNoLeaks()
  } finally { await f.close() }
})

test('监控选择生命周期：历史pin、当前请求恢复follow、普通返回/刷新、聊天明确跳转', async ({}, testInfo) => {
  test.setTimeout(90_000)
  const f = await fixture(testInfo)
  try {
    await f.selectChild('固定历史轮', 'old-2')
    const oldGroup = await f.chart.getAttribute('data-group-key')
    const third = await f.advance()
    await expect(f.chart).toHaveAttribute('data-request-key', 'old-2')
    await expect(f.chart).toHaveAttribute('data-group-key', oldGroup!)
    await f.selectChild('当前轮', third)
    const fourth = await f.advance()
    await expect(f.chart, 'Manually selecting the current running request resumes following the next request').toHaveAttribute('data-request-key', fourth)

    await f.selectChild('固定历史轮', 'old-2')
    await f.navigation.getByRole('button', { name: '设置', exact: true }).click()
    await f.navigation.getByRole('button', { name: '运行监控', exact: true }).click()
    await expect(f.chart).toHaveAttribute('data-group-key', groupKey({ ...f.archived[0], turn_id: 'current-turn' }))
    await expect(f.chart).not.toHaveAttribute('data-request-key', 'old-2')
    await f.selectChild('固定历史轮', 'old-2')
    await f.monitor.getByRole('button', { name: '刷新到最新', exact: true }).click()
    await expect(f.chart).not.toHaveAttribute('data-request-key', 'old-2')
    const fifth = await f.advance()
    await expect(f.chart).toHaveAttribute('data-group-key', groupKey({ ...f.archived[0], turn_id: 'current-turn' }))
    if (await f.chart.getAttribute('data-mode') === 'realtime') await expect(f.chart).toHaveAttribute('data-request-key', fifth)

    await f.navigation.getByRole('button', { name: '测试对话', exact: true }).click()
    await f.page.getByRole('button', { name: '查看本轮 2 次模型请求统计', exact: true }).click()
    await expect(f.monitor).toBeVisible()
    await expect(f.parent('内置测试轮')).toBeVisible()
    await expect(f.parent('内置测试轮')).toHaveAttribute('data-expanded', 'false')
    await expect(f.chart).toHaveAttribute('data-mode', 'turn')
    const explicitGroup = await f.chart.getAttribute('data-group-key')
    expect(explicitGroup).not.toBe(groupKey({ ...f.archived[0], turn_id: 'current-turn' }))
    await f.advance()
    await expect(f.chart, 'Explicit chat inspection must survive the monitor entry/follow effect and later gateway requests').toHaveAttribute('data-group-key', explicitGroup!)
    await f.screenshot('1366x768-explicit-chat-inspection.png')
    await f.navigation.getByRole('button', { name: '设置', exact: true }).click()
    await f.navigation.getByRole('button', { name: '运行监控', exact: true }).click()
    await expect(f.chart).toHaveAttribute('data-group-key', groupKey({ ...f.archived[0], turn_id: 'current-turn' }))
    await test.step('明确聊天检查期间手选真实当前请求恢复follow', async () => {
      await f.navigation.getByRole('button', { name: '测试对话', exact: true }).click()
      await f.page.getByRole('button', { name: '查看本轮 2 次模型请求统计', exact: true }).click()
      await expect(f.chart).toHaveAttribute('data-mode', 'turn')
      await expect(f.chart).toHaveAttribute('data-group-key', explicitGroup!)
      const current = await f.advance()
      await expect(f.chart).toHaveAttribute('data-group-key', explicitGroup!)
      await f.selectChild('当前轮', current)
      const following = await f.advance()
      await expect(f.chart, 'A real current request selected during explicit chat inspection must resume following its next request').toHaveAttribute('data-request-key', following)
    })
    f.verifyNoLeaks()
  } finally { await f.close() }
})

test('统一保留记录：外部与内置对话混合汇总、内置来源筛选及历史pin', async ({}, testInfo) => {
  test.setTimeout(60_000)
  const f = await fixture(testInfo)
  try {
    await expect(f.parent('内置测试轮')).toBeVisible()
    await expect(f.parent('固定历史轮')).toBeVisible()
    const summary = f.monitor.getByRole('region', { name: '筛选范围汇总', exact: true })
    await expect(summary.locator('[data-summary="count"]')).toHaveText('59')
    await expect(summary.locator('[data-summary="input"]')).toHaveText('16,900')
    await expect(summary.locator('[data-summary="output"]')).toHaveText('1,690')
    await expect(summary.locator('[data-summary="cached"]')).toHaveText('7,720')
    await f.parent('内置测试轮').getByRole('button', { name: /^选择轮汇总/ }).click()
    const internalKey = JSON.stringify(['internal', f.internalGroup.conversation_id, f.internalGroup.message_id, f.internalGroup.instance_id])
    await expect(f.chart).toHaveAttribute('data-group-key', internalKey)
    await f.parent('内置测试轮').locator('td').last().click()
    const request = f.table.locator('tr[data-request-key="chat-explicit-1"]')
    await expect(request).toBeVisible()
    await request.locator('td').last().click()
    await expect(f.chart).toHaveAttribute('data-request-key', 'chat-explicit-1')
    await f.advance()
    await expect(f.chart).toHaveAttribute('data-request-key', 'chat-explicit-1')
    await expect(f.chart).toHaveAttribute('data-group-key', internalKey)
    const instance = f.monitor.getByLabel('历史实例', { exact: true })
    const option = await instance.locator('option').filter({ hasText: '内置测试轮' }).getAttribute('value')
    expect(option).toBeTruthy()
    await instance.selectOption(option!)
    await expect.poll(() => f.historyQueries.some(query => query.get('source') === 'internal' && query.get('instance_id') === f.instance.id)).toBe(true)
    await expect(summary.locator('[data-summary="count"]')).toHaveText('2')
    await expect(summary.locator('[data-summary="input"]')).toHaveText('7,000')
    await expect(f.parent('固定历史轮')).toHaveCount(0)
    await expect(f.parent('内置测试轮')).toHaveCount(1)
    await f.screenshot('1366x768-internal-history-filter.png')
    f.verifyNoLeaks()
  } finally { await f.close() }
})

test('跨实例请求：真实进行中自动切采样并跟随，遗留running与实际telemetry不符时保持pin', async ({}, testInfo) => {
  test.setTimeout(60_000)
  const f = await fixture(testInfo, { twoInstances: true })
  try {
    const sampler = f.monitor.getByLabel('选择监控实例', { exact: true })
    await expect(sampler).toHaveValue(f.instance.id)
    await f.selectChild(f.otherStale.alias, f.otherStale.request_key, f.otherInstance.id)
    await expect(sampler, 'Selecting a running request on another instance switches the existing sampler automatically').toHaveValue(f.otherInstance.id)
    await f.page.clock.runFor(700)
    await expect(f.chart, 'A stale running ledger entry does not match the actual current request').toHaveAttribute('data-request-key', f.otherStale.request_key)
    const next = await f.advanceOther()
    await expect(f.chart, 'Unmatched stale running selection remains pinned after a newer real request').toHaveAttribute('data-request-key', f.otherStale.request_key)

    const before = f.telemetryReads.filter(id => id === f.instance.id).length
    await sampler.selectOption(f.instance.id)
    await expect.poll(() => f.telemetryReads.filter(id => id === f.instance.id).length).toBeGreaterThan(before)
    await f.selectChild(f.otherInstance.alias, next, f.otherInstance.id)
    await expect(sampler, 'No second manual instance-selection gesture is needed for the clicked running request').toHaveValue(f.otherInstance.id)
    await f.page.clock.runFor(700)
    await expect(f.chart).toHaveAttribute('data-request-key', next)
    const following = await f.advanceOther()
    await expect(f.chart, 'Confirmed current request resumes following subsequent requests on its own instance').toHaveAttribute('data-request-key', following)
    await expect(f.chart).toHaveAttribute('data-group-key', JSON.stringify(['external', 'redesign-run', f.otherInstance.id, 'other-turn']))
    await f.screenshot('1366x768-other-instance-follow.png')
    f.verifyNoLeaks()
  } finally { await f.close() }
})
