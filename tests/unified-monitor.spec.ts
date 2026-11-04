import { expect, test } from '@playwright/test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { MonitorGeneration } from '../src/MonitorGeneration'
import { MonitorRequestHistory } from '../src/MonitorRequestHistory'
import { MonitorRequestTable } from '../src/MonitorRequestTable'
import { externalGenerationView, externalRequestIdentity } from '../src/monitor-generation'
import type { MonitorState } from '../src/useMonitorState'
import { historyRequestGroups } from '../src/monitor-request-groups'

const generation = externalGenerationView({ run_id: 'current', aggregation: 'request', turn_id: null, state: 'completed', requests: [
  { run_id: 'current', instance_id: 'live', request_key: 'first', revision: 1, alias: 'Live model', backend: 'llama_cpp', labels: [],
    aggregation: 'request', turn_id: null, turn_state: null, status: 'completed', started_at: 100, ended_at: 101, duration_s: 1,
    usage: { prompt_tokens: 100, prompt_tokens_details: { cached_tokens: 80 }, completion_tokens: 20 } },
] })

test('常驻统一监控：首条请求的图表与日期表同屏，只有一张请求表', () => {
  const monitor = { generation, selectedTurn: null, instanceId: 'live', historyMemory: () => undefined, followLatest: () => {} } as unknown as MonitorState
  const html = renderToStaticMarkup(createElement(MonitorRequestHistory, { monitor }))
  expect(html).toContain('class="monitor-generation-chart"')
  expect(html.match(/<table\b/g)).toHaveLength(1)
  expect(html).toContain('aria-label="历史日期"')
  expect(html).toContain('aria-label="历史实例"')
  expect(html).not.toContain('aria-label="所选调用"')
  expect(html).not.toContain('aria-label="图表范围"')
})

test('生成区单独渲染图表统计，不再夹带第二份请求明细表', () => {
  const html = renderToStaticMarkup(createElement(MonitorGeneration, { generation, scope: 'request', onSelectRequest: () => {} }))
  expect(html).toContain('class="monitor-generation-chart"')
  expect(html).toContain('aria-label="所选请求统计"')
  expect(html).not.toContain('<table')
  expect(html).toMatch(/<dt>请求长度<\/dt><dd><span[^>]*><strong>120<\/strong>/)
})

test('列表不再切换日期/本轮模式，四个旧操作入口全部移除', () => {
  const turn = externalGenerationView({ run_id: 'current', aggregation: 'turn', turn_id: 'T', state: 'completed', requests:
    generation.requests.map(request => ({ ...request.external!, aggregation: 'turn', turn_id: 'T', turn_state: 'completed' })) })
  for (const selectedTurn of [null, { instanceId: 'live', messageId: 'internal', model: 'Live model', phase: 'completed' }]) {
    const monitor = { generation: turn, selectedTurn, instanceId: 'live', historyMemory: () => undefined, followLatest: () => {} } as unknown as MonitorState
    const html = renderToStaticMarkup(createElement(MonitorRequestHistory, { monitor }))
    for (const label of ['查看本轮', '查看本次', '返回日期列表', '刷新记录', '返回最新']) expect(html).not.toContain(label)
    expect(html.match(/<table\b/g)).toHaveLength(1)
    expect(html).toContain('data-group-key=')
  }
})

test('日期页外部按真实run/instance/request独立显示，不再使用旧turn_id归组', () => {
  const base = generation.requests[0].external!
  const records = [
    { ...base, request_key: 'one', aggregation: 'turn' as const, turn_id: 'T' },
    { ...base, request_key: 'plain' },
    { ...base, request_key: 'two', aggregation: 'turn' as const, turn_id: 'T' },
    { ...base, request_key: 'prior', aggregation: 'turn' as const, turn_id: 'T', run_id: 'prior' },
    { ...base, request_key: 'other-instance', aggregation: 'turn' as const, turn_id: 'T', instance_id: 'other' },
  ]
  const entries = historyRequestGroups(records)
  expect(entries).toHaveLength(5)
  expect(entries[0].records.map(record => record.request_key)).toEqual(['one'])
  expect(entries[0].key).toBe(externalRequestIdentity(records[0]))
  expect(entries[1].turn).toBe(false)
  expect(entries[0]).not.toHaveProperty('count')
  expect(historyRequestGroups([...records].reverse()).find(entry => entry.key === entries[0].key)?.records.map(record => record.request_key)).toEqual(['one'])
})

test('折叠组未知总数不冒充本页条数，混合内置行仍显示原始总输入', () => {
  const external = generation.requests[0], internal = { ...external, id: 'internal', external: undefined }
  const html = renderToStaticMarkup(createElement(MonitorRequestTable, { onSelectGroup: () => {}, onToggleGroup: () => {}, onSelectRequest: () => {},
    entries: [
      { key: 'unknown', turn: true, requests: [], representative: external, alias: 'External', backend: 'llama_cpp', status: '已完成', expanded: false },
      { key: 'external', turn: false, requests: [external], alias: 'External', backend: 'llama_cpp' },
      { key: 'internal', turn: false, requests: [internal], alias: 'Internal', backend: 'llama_cpp' },
    ] }))
  expect(html).toContain('aria-label="请求数 未知"')
  const unknownRow = html.match(/<tr\b[^>]*data-group-key="unknown"[\s\S]*?<\/tr>/)?.[0] ?? ''
  expect(unknownRow).toContain('aria-label="请求数 未知"')
  expect(unknownRow).not.toContain('1 次请求')
  expect(html.match(/aria-label="输入（含缓存） 100 Token/g)).toHaveLength(2)
  expect(html.match(/<table\b/g)).toHaveLength(1)
})


test('无时间戳的当前后端指标仍在今天统一列表，不伪造1970年记录', () => {
  const backend = { ...generation, origin: 'backend' as const, requests: generation.requests.map(request => ({ ...request, external: undefined, startedAt: 0 })) }
  const monitor = { generation: backend, selectedTurn: null, instanceId: 'live', historyMemory: () => undefined, followLatest: () => {} } as unknown as MonitorState
  const html = renderToStaticMarkup(createElement(MonitorRequestHistory, { monitor }))
  const table = html.match(/<table\b[\s\S]*?<\/table>/)?.[0] ?? ''
  expect(table).toContain('data-group-key=')
  expect(table).toContain('aria-label="请求数 1"')
  expect(html).not.toContain('1970')
})
