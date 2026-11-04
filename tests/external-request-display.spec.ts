import { expect, test } from '@playwright/test'
import { externalGenerationView, externalRequestIdentity } from '../src/monitor-generation'
import { historyRequestGroups } from '../src/monitor-request-groups'
import { projectGenerationTotals } from '../src/monitor-generation-readouts'
import { historyBackend } from '../src/monitor-history-dates'
import type { ExternalRequestRecord, GenerationAccounting, InternalRequestHistoryGroup, RequestHistoryDates, RequestHistoryPage } from '../src/types'
import { historyStats } from './monitor-history-fixture'
import { readMonitorHistoryPage } from '../src/monitor-history-page'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { MonitorRequestTable } from '../src/MonitorRequestTable'

const record = (key: string, extra: Partial<ExternalRequestRecord> = {}): ExternalRequestRecord => ({
  run_id: 'run', instance_id: 'instance', request_key: key, alias: 'Model', backend: 'strata', revision: 2,
  aggregation: 'turn', turn_id: 'legacy-turn', turn_state: 'waiting', labels: [], status: 'completed',
  started_at: 100, ended_at: 104, duration_s: 4, usage: { prompt_tokens: 100, completion_tokens: 10 }, timings: {}, ...extra,
})

test('旧后端轮次快照只显示当前HTTP请求，完成工具调用不再待续调', () => {
  const before = record('before', { status: 'failed' }), current = record('current', { started_at: 110 })
  const view = externalGenerationView({ run_id: 'run', aggregation: 'turn', turn_id: 'legacy-turn', state: 'waiting', requests: [before, current] })
  expect(view).toMatchObject({ aggregation: 'request', status: '已完成', waiting: false, active: false })
  expect(view.requests.map(request => request.requestKey)).toEqual(['current'])
  expect(view.turn).toBeUndefined()
  const entries = historyRequestGroups([before, current], [])
  expect(entries.map(entry => entry.key)).toEqual([externalRequestIdentity(before), externalRequestIdentity(current)])
  expect(entries.every(entry => !entry.turn && entry.records.length === 1)).toBe(true)
})

test('旧事件流的同轮prior不得计入当前外部请求', () => {
  const old = record('old'), current = record('current', { status: 'running', ended_at: undefined })
  const request = externalGenerationView({ run_id: 'run', aggregation: 'turn', turn_id: 'legacy-turn', state: 'running', requests: [current] }).requests[0]
  const accounting = { request: { ...current, started_at: 100, streaming: true }, turn: { source: 'external', key: 'turn:legacy-turn',
    turn_id: 'legacy-turn', client_request_id: null, request_keys: ['old', 'current'], prior: historyStats([old]) } } as GenerationAccounting
  expect(projectGenerationTotals([request], accounting)).toMatchObject({ request_count: 1, input: { value: 100 }, output: { value: 10 } })
})

test('外部行直接选择请求，不保留单请求父子摘要或展开入口', () => {
  const original = record('single', { tool_names: ['view_image'] })
  const request = externalGenerationView({ run_id: 'run', aggregation: 'turn', turn_id: 'legacy-turn', state: 'waiting', requests: [original] }).requests[0]
  const html = renderToStaticMarkup(createElement(MonitorRequestTable, {
    entries: [{ key: externalRequestIdentity(original), turn: false, requests: [request], representative: request, alias: 'Model', backend: 'strata', count: 1 }],
    onSelectGroup() {}, onSelectRequest() {}, onToggleGroup() {},
  }))
  expect(html).toContain('data-request-key="single"')
  expect(html).toContain('aria-label="已完成"')
  expect(html).toContain('strata · 外部')
  expect(html).toContain('view_image')
  expect(html).not.toContain('展开调用')
  expect(html).not.toContain('选择轮汇总')
  expect(html.match(/<tr\b/g)).toHaveLength(3)
})

test('旧内置backend仅按相同instance_id补充，不用模型名猜测', () => {
  const catalog = { dates: [{ date: '2026-11-02', instances: [{ id: 'same', alias: 'Model', backend: 'strata' }] }] } as RequestHistoryDates
  expect(historyBackend('same', '', [], catalog)).toBe('strata')
  expect(historyBackend('same', '', [{ id: 'same', backend: 'ninfer' }], catalog)).toBe('ninfer')
  expect(historyBackend('same', 'llama_cpp', [{ id: 'same', backend: 'ninfer' }], catalog)).toBe('llama_cpp')
  expect(historyBackend('other', '', [{ id: 'same', backend: 'ninfer' }], catalog)).toBe('')
  expect(historyBackend(null, '', [{ id: 'same', backend: 'ninfer' }], catalog)).toBe('')
})

test('旧接口兼容读取原始外部记录后按请求分页，内置仍为一行且不重复探测', async () => {
  const records = [record('one', { started_at: 100 }), record('two', { started_at: 120 }), record('three', { started_at: 130 })]
  const internal = { source: 'internal', conversation_id: 'conversation', message_id: 'message', instance_id: 'same', alias: 'Model',
    started_at: 110, request_count: 2, requests: [] } as unknown as InternalRequestHistoryGroup
  const queries: URLSearchParams[] = []
  const read = async (query: URLSearchParams): Promise<RequestHistoryPage> => {
    queries.push(new URLSearchParams(query))
    if (!query.has('group_by')) return { run_id: null, items: records, total: records.length, offset: 0, limit: 5, issues: [] }
    return { run_id: null, items: [records[2]], groups: [], internal_groups: [internal], group_total: 2,
      summary: historyStats(records), total: 5, offset: 0, limit: Number(query.get('limit')), issues: [] }
  }
  const mode = { current: undefined as boolean | undefined }
  const page = await readMonitorHistoryPage(new URLSearchParams({ group_by: 'turn', offset: '1', limit: '2' }), read, mode)
  expect(page.items.map(item => item.request_key)).toEqual(['two'])
  expect(page.internal_groups).toEqual([internal])
  expect(page).toMatchObject({ group_total: 4, total: 5, offset: 1, limit: 2 })
  expect(mode.current).toBe(false)
  queries.length = 0
  await readMonitorHistoryPage(new URLSearchParams({ group_by: 'turn', offset: '0', limit: '2' }), read, mode)
  expect(queries.every(query => !query.has('external_group_by'))).toBe(true)
})

test('新接口直接使用逐请求分页；网络或服务错误原样返回', async () => {
  const page = { external_group_by: 'request', groups: [], internal_groups: [], group_total: 0, summary: historyStats([]),
    run_id: null, items: [], total: 0, offset: 0, limit: 50, issues: [] } as RequestHistoryPage
  const queries: URLSearchParams[] = []
  const mode = { current: undefined as boolean | undefined }
  expect(await readMonitorHistoryPage(new URLSearchParams({ group_by: 'turn' }), async query => { queries.push(query); return page }, mode)).toBe(page)
  expect(queries).toHaveLength(1)
  expect(queries[0].get('external_group_by')).toBe('request')
  expect(mode.current).toBe(true)
  await expect(readMonitorHistoryPage(new URLSearchParams(), async () => { throw new Error('HTTP 500') }, mode)).rejects.toThrow('HTTP 500')
})

test('旧接口混合跨日内置轮次时，分页总请求数仍只计日期筛选内的请求', async () => {
  const external = record('today')
  const internal = { source: 'internal', started_at: 100, request_count: 3, matched_request_count: 1, requests: [] } as unknown as InternalRequestHistoryGroup
  const page: RequestHistoryPage = { run_id: null, items: [external], groups: [], internal_groups: [internal], group_total: 2,
    total: 2, summary: historyStats([external]), offset: 0, limit: 50, issues: [] }
  const result = await readMonitorHistoryPage(new URLSearchParams({ group_by: 'turn', limit: '50', date_from: '2026-11-02', date_to: '2026-11-02' }),
    async query => query.has('group_by') ? page : { ...page, items: [external], total: 1 }, { current: false })
  expect(result).toMatchObject({ total: 2, group_total: 2 })
  expect(result.internal_groups?.[0].request_count).toBe(3)
})
