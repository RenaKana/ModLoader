import { expect, test, chromium } from '@playwright/test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { build } from 'esbuild'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { pathToFileURL } from 'node:url'
import { MonitorRequestTable } from '../src/MonitorRequestTable'
import { externalGenerationView, type MonitorRequestView } from '../src/monitor-generation'
import type { MonitorRequestTableEntry } from '../src/monitor-request-groups'
import type { ExternalRequestRecord, RequestHistoryStats } from '../src/types'

const root = path.resolve(__dirname, '..')
const artifacts = process.env.MODLOADER_TEST_ARTIFACTS || path.join(root, '.artifacts/codex/monitor-tree-motion-20261031')
const record = (key: string, more: Partial<ExternalRequestRecord> = {}): ExternalRequestRecord => ({
  run_id: 'table-test', instance_id: 'model', request_key: key, revision: 1, alias: 'Solo model', backend: 'llama_cpp', labels: [],
  aggregation: 'request', turn_id: null, turn_state: null, status: 'completed', started_at: 100, ended_at: 102, duration_s: 2,
  usage: { prompt_tokens: 100, prompt_tokens_details: { cached_tokens: 80 }, completion_tokens: 20, ttft_ms: 800 },
  timings: { prompt_n: 20, prompt_ms: 100, predicted_n: 20, predicted_ms: 1000 }, ...more,
})
const modelRequests = (records: ExternalRequestRecord[]) => records.flatMap(record => externalGenerationView({ run_id: 'table-test', aggregation: 'request', turn_id: null, state: record.status, requests: [record] }).requests)
  .map((request, index) => ({ ...request, index: index + 1, external: undefined }))
const requests = modelRequests([record('one'), record('two', { started_at: 103, ended_at: 107, duration_s: 4,
  usage: { prompt_tokens: 200, completion_tokens: 40, ttft_ms: 1200 }, timings: { predicted_n: 40, predicted_ms: 4000 } })])
const solo: MonitorRequestTableEntry = { key: 'solo', turn: true, count: 1, requests: [requests[0]], alias: 'Solo model', backend: 'llama_cpp', startedAt: 100_000, status: 'completed' }
const turn: MonitorRequestTableEntry = { key: 'turn', turn: true, requests, alias: 'Round model', backend: 'strata', count: 2, startedAt: 100_000, status: 'completed', expanded: true }
const summary: RequestHistoryStats = {
  request_count: 8, input: { value: 12345, reported: 8 }, output: { value: 780, reported: 7 }, total: { value: 13125, reported: 7 },
  cached: { value: 11000, reported: 8 }, cache_pair: { input: 12345, cached: 11000, reported: 8 }, cache_ratio: 11000 / 12345,
  decode: { value: 90, tokens: 780, seconds: 780 / 90, reported: 7 }, prefill: { value: 2000, tokens: 1345, seconds: 1345 / 2000, reported: 8 },
  ttft: { value: .75, reported: 6, engine_reported: 4, gateway_reported: 2 },
  status_counts: { running: 0, completed: 7, cancelled: 0, failed: 1, incomplete: 0 }, record_incomplete_count: 0,
}
const engineRequests = modelRequests([
  record('spark-one', { usage: { completion_tokens: 100, generation_ms: 2000, decode_tps: 50 }, timings: {} }),
  record('spark-two', { usage: { completion_tokens: 100, generation_ms: 1000, decode_tps: 100 }, timings: {} }),
])
const engineTurn: MonitorRequestTableEntry = { ...turn, key: 'spark', alias: 'Spark model', backend: 'sparkinfer', requests: engineRequests }
const render = (entries: MonitorRequestTableEntry[], more = {}) => renderToStaticMarkup(createElement(MonitorRequestTable, {
  entries, onSelectGroup() {}, onSelectRequest() {}, onToggleGroup() {}, ...more,
}))

test('single-request internal turns keep their collapsed summary, with TTFT below duration and no status column', () => {
  const html = render([solo])
  expect(html).toContain('data-group-key="solo"')
  expect(html).not.toContain('data-request-key="one"')
  expect(html).toContain('aria-label="输入（含缓存） 100 Token"')
  expect(html).toContain('首字')
  expect(html).toContain('0.80')
  expect(html).toContain('aria-label="已完成"')
  expect(html).not.toMatch(/<th[^>]*>状态<\/th>/)
  expect(html.indexOf('模型 / 调用')).toBeLessThan(html.indexOf('输入'))
  expect(html.indexOf('速度 · tok/s')).toBeLessThan(html.indexOf('>时间</th>'))
  expect(html.match(/scope="col"/g)).toHaveLength(7)
  expect(html).not.toContain('lucide-chevron-right')
  expect(html).toContain('data-request-count="1"')
  expect(html).toContain('llama_cpp · ')
})

test('collapsed directory summaries show the full count and totals without using the representative request', () => {
  const html = render([{ ...turn, expanded: false, requests: [], representative: requests[0], summary, elapsed: 30, matchedRequestCount: 3 }])
  expect(html).toContain('aria-label="请求数 8"')
  expect(html).toContain('aria-label="输入（含缓存） 12,345 Token"')
  expect(html).toContain('aria-label="输出 780 Token · 部分 7/8"')
  expect(html).toContain('首字 0.75 秒 · 已知 6/8 次请求均值')
  expect(html).toContain('89.1%')
  expect(html).not.toContain('data-request-key="one"')
  expect(html).toContain('data-request-count="8"')
  expect(html).toContain('8 次请求')
  expect(html).toContain('筛选内 3')
  expect(html).not.toContain('data-column="count"')
})

test('only complete requests supply fallback summaries; partial pages cannot become full-round totals', () => {
  const complete = render([turn])
  expect(complete).toContain('aria-label="输入（含缓存） 300 Token"')
  expect(complete).toContain('aria-label="Decode 12.00 token/s"')
  expect(complete).toContain('首字 1.00 秒 · 已知 2/2 次请求均值')
  const partial = render([{ ...turn, requests: [requests[0]], count: 8, expanded: false }])
  expect(partial).toContain('aria-label="输入（含缓存） 未知"')
  expect(partial).not.toContain('aria-label="输入（含缓存） 100 Token"')
  const unknown = render([{ ...turn, requests: [requests[0]], count: undefined, expanded: false }])
  expect(unknown).toContain('aria-label="请求数 未知"')
})

test('expanded children keep request identity, selected tool details, and group errors readable', () => {
  const html = render([{ ...turn, error: '本轮记录读取失败' }], { selectedRequestId: requests[0].id, tools: [{
    id: 'tool-one', requestId: requests[0].id, kind: 'tool', status: 'failed',
    call: { id: 'call', type: 'function', function: { name: 'read_file', arguments: '{"path":"missing.txt"}' } },
    result: '{"error":"文件不存在"}',
  }] })
  expect(html).toContain('data-request-key="one"')
  expect(html).toContain('本轮记录读取失败')
  expect(html).toContain('文件不存在')
  expect(html).toContain('本次工具调用')
})

test('engine-only request and complete-round speeds retain AVG instead of being labeled Decode', () => {
  const html = render([engineTurn])
  expect(html).toContain('aria-label="引擎平均 66.67 token/s · 包含首 Token 等待"')
  expect(html).toContain('aria-label="引擎平均 50.00 token/s · 包含首 Token 等待"')
  expect(html).toContain('aria-label="引擎平均 100.00 token/s · 包含首 Token 等待"')
  expect(html.match(/>AVG<\/small>/g)).toHaveLength(3)
  expect(html).not.toContain('aria-label="Decode')
  const native = render([{ ...solo, requests: [{ ...engineRequests[0], decodeMean: 40, timings: { predicted_n: 100, predicted_ms: 2500 } }], expanded: true }])
  expect(native).toContain('aria-label="Decode 40.00 token/s"')
  expect(native).not.toContain('>AVG</small>')
})

test('summary parents derive missing engine means only from complete request ledgers', () => {
  const engineSummary = { ...summary, request_count: 2, decode: { value: null, tokens: 0, seconds: 0, reported: 0 } }
  const complete = render([{ ...engineTurn, summary: engineSummary, expanded: false }])
  expect(complete).toContain('aria-label="引擎平均 66.67 token/s · 包含首 Token 等待"')
  expect(complete).toContain('>AVG</small>')
  for (const entry of [
    { ...engineTurn, summary: engineSummary, requests: [engineRequests[0]], expanded: false },
    { ...engineTurn, summary: engineSummary, requests: [], representative: engineRequests[0], expanded: false },
    { ...engineTurn, requests: [engineRequests[0]], expanded: false },
  ]) {
    const incomplete = render([entry])
    expect(incomplete).not.toContain('>AVG</small>')
    expect(incomplete).toContain('aria-label="Decode 未知"')
  }
})

test('collapsed summary engine speeds display AVG before request details are loaded', () => {
  const entry: MonitorRequestTableEntry = { ...engineTurn, expanded: false, requests: [], representative: engineRequests[0],
    summary: { ...summary, decode: { value: null, tokens: 0, seconds: 0, reported: 0 }, engine: { value: 200 / 3, tokens: 200, seconds: 3, reported: 2 } } }
  const html = render([entry])
  expect(html).toContain('aria-label="引擎平均 66.67 token/s · 包含首 Token 等待"')
  expect(html).toContain('>AVG</small>')
  expect(html).not.toContain('aria-label="Decode')
  expect(html).not.toContain('data-request-key="spark-one"')
  const native = render([{ ...entry, summary: { ...entry.summary!, decode: { value: 40, tokens: 200, seconds: 5, reported: 2 } } }])
  expect(native).toContain('aria-label="Decode 40.00 token/s · 部分 2/8"')
  expect(native).not.toContain('>AVG</small>')
  const unknown = render([{ ...entry, summary: { ...entry.summary!, engine: { value: null, tokens: 0, seconds: 0, reported: 0 } } }])
  expect(unknown).toContain('aria-label="Decode 未知"')
  expect(unknown).not.toContain('>AVG</small>')
})

const nativeUnweighted: MonitorRequestView = { ...requests[0], usage: {}, timings: {}, prefill: undefined,
  live: { source: 'strata', average_tokens_per_second: 55, average_basis: 'backend_decode', average_prefill_tokens_per_second: 1600, prefill_average_basis: 'backend_prefill' },
  decodeMean: 55, prefillMean: 1600 }

test('a single native mean without token weights survives the collapsed parent summary', () => {
  for (const isTurn of [false, true]) {
    const html = render([{ ...solo, turn: isTurn, count: 1, requests: [nativeUnweighted], expanded: false }])
    expect(html).toContain('aria-label="Decode 55.00 token/s"')
    expect(html).toContain('aria-label="Prefill 1,600.00 token/s"')
  }
})

test('single-request summary gaps use native means, but multiple unweighted requests remain unknown', () => {
  const emptyRate = { value: null, tokens: 0, seconds: 0, reported: 0 }
  const nativeSummary = { ...summary, request_count: 1, decode: emptyRate, prefill: emptyRate }
  const entry = { ...solo, count: 1, requests: [nativeUnweighted], summary: nativeSummary, expanded: false }
  const html = render([entry])
  expect(html).toContain('aria-label="Decode 55.00 token/s"')
  expect(html).toContain('aria-label="Prefill 1,600.00 token/s"')
  const knownSummary = render([{ ...entry, summary: { ...nativeSummary, decode: { value: 60, tokens: 120, seconds: 2, reported: 1 }, prefill: { value: 1700, tokens: 1700, seconds: 1, reported: 1 } } }])
  expect(knownSummary).toContain('aria-label="Decode 60.00 token/s"')
  expect(knownSummary).toContain('aria-label="Prefill 1,700.00 token/s"')
  const multiple = [nativeUnweighted, { ...nativeUnweighted, id: 'native-second', index: 2 }]
  for (const value of [
    { ...entry, requests: [], representative: nativeUnweighted },
    { ...entry, count: 2, requests: multiple, summary: { ...nativeSummary, request_count: 2 } },
    { ...entry, count: 2, requests: multiple, summary: undefined },
  ]) {
    const unknown = render([value])
    expect(unknown).toContain('aria-label="Decode 未知"')
    expect(unknown).toContain('aria-label="Prefill 未知"')
  }
})

test('name selection is independent from expansion; other parent cells toggle and child cells select requests', async () => {
  test.setTimeout(60_000)
  fs.mkdirSync(artifacts, { recursive: true })
  const fixture = fs.mkdtempSync(path.join(artifacts, 'fixture-'))
  const source = (file: string) => JSON.stringify(path.join(root, file).replaceAll('\\', '/'))
  fs.writeFileSync(path.join(fixture, 'index.html'), '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><link rel="stylesheet" href="fixture.css"><div id="root"></div><script src="fixture.js"></script></html>')
  fs.writeFileSync(path.join(fixture, 'fixture.tsx'), `import React,{useState} from 'react';import{createRoot}from'react-dom/client';
    import{MonitorRequestTable}from ${source('src/MonitorRequestTable')};import ${source('src/styles.css')};import ${source('src/resource-cards.css')};import ${source('src/monitor-request-history.css')};
    const initial=${JSON.stringify([solo, { ...turn, expanded: false }, engineTurn])};
    function Fixture(){const[entries,setEntries]=useState(initial);const[group,setGroup]=useState('solo');const[request,setRequest]=useState('');
      return <main style={{padding:24,maxWidth:1200,margin:'auto'}}><output>{request?'request:'+request:'group:'+group}</output><MonitorRequestTable entries={entries} selectedGroupId={group} selectedRequestId={request}
        onSelectGroup={entry=>{setGroup(entry.key);setRequest('')}} onSelectRequest={(item,entry)=>{setGroup(entry.key);setRequest(item.id)}}
        onToggleGroup={entry=>setEntries(rows=>rows.map(row=>row.key===entry.key?{...row,expanded:!row.expanded}:row))}/></main>}
    createRoot(document.getElementById('root')).render(<Fixture/>);`)
  await build({ entryPoints: [path.join(fixture, 'fixture.tsx')], outfile: path.join(fixture, 'fixture.js'), bundle: true, format: 'iife', jsx: 'automatic',
    loader: { '.woff2': 'file' }, define: { 'process.env.NODE_ENV': '"development"' } })
  const browser = await chromium.launch({ channel: 'msedge' })
  try {
    const page = await browser.newPage({ viewport: { width: 1200, height: 800 } })
    const errors: string[] = []
    page.on('pageerror', error => errors.push(error.message))
    await page.goto(pathToFileURL(path.join(fixture, 'index.html')).href)
    const parent = page.locator('[data-group-key="solo"]')
    const name = parent.getByRole('button', { name: /选择轮汇总/ })
    await name.click()
    await expect(parent).toHaveAttribute('data-expanded', 'false')
    await expect(page.locator('output')).toHaveText('group:solo')
    await expect(page.locator('[data-group-key="spark"] td[data-column="speed"]')).toContainText('66.67AVG')
    await expect(page.locator('[data-group-key="spark"]').getByLabel('引擎平均 66.67 token/s · 包含首 Token 等待', { exact: true })).toBeVisible()
    await expect(page.locator('[data-request-key="spark-one"] td[data-column="speed"]')).toContainText('50.00AVG')
    await page.locator('[data-group-key="turn"]').getByRole('button', { name: /^选择轮汇总/ }).click()
    await parent.locator('[data-request-count]').click()
    await expect(parent).toHaveAttribute('data-expanded', 'true')
    await expect(page.locator('output')).toHaveText('group:turn')
    await parent.locator('td[data-column="input"]').click()
    await expect(parent).toHaveAttribute('data-expanded', 'false')
    await parent.locator('td[data-column="input"]').click()
    await expect(parent).toHaveAttribute('data-expanded', 'true')
    await expect(page.locator('output')).toHaveText('group:turn')
    await name.click()
    await expect(parent).toHaveAttribute('data-expanded', 'true')
    const child = page.locator('[data-request-key="one"]').first()
    await child.locator('td[data-column="elapsed"]').click()
    await expect(page.locator('output')).toHaveText(`request:${requests[0].id}`)
    await parent.getByRole('button', { name: /收起调用/ }).click()
    await expect(parent).toHaveAttribute('data-expanded', 'false')
    await expect(page.locator('output')).toHaveText(`request:${requests[0].id}`)
    await parent.getByRole('button', { name: /展开调用/ }).focus()
    await page.keyboard.press('Enter')
    await expect(parent).toHaveAttribute('data-expanded', 'true')
    await name.focus(); await page.keyboard.press('Enter')
    await expect(parent).toHaveAttribute('data-expanded', 'true')
    await expect(page.locator('output')).toHaveText('group:solo')
    expect(await parent.evaluate(node => node.getBoundingClientRect().height)).toBeGreaterThanOrEqual(78)
    expect(await parent.evaluate(node => node.getBoundingClientRect().height)).toBeLessThanOrEqual(80)
    expect(await page.locator('[data-group-key="turn"]').evaluate(node => node.getBoundingClientRect().height)).toBeLessThanOrEqual(80)
    expect(await parent.locator('th').evaluate(node => getComputedStyle(node).borderTopLeftRadius)).toBe('10px')
    expect(await page.locator('.monitor-ledger-spacer').first().evaluate(node => node.getBoundingClientRect().height)).toBe(8)
    const listHeight = await page.locator('.monitor-request-ledger').evaluate(node => parseFloat(getComputedStyle(node).maxHeight))
    expect(listHeight).toBeGreaterThanOrEqual(380)
    expect(listHeight).toBeLessThanOrEqual(600)
    await page.screenshot({ path: path.join(artifacts, 'request-table.png') })
    await page.setViewportSize({ width: 680, height: 800 })
    expect(await page.locator('.monitor-request-ledger').evaluate(node => node.scrollWidth > node.clientWidth)).toBe(true)
    await page.screenshot({ path: path.join(artifacts, 'request-table-narrow.png') })
    expect(errors).toEqual([])
  } finally { await browser.close() }
})

test('continuous tree rails follow real rows and tools; collapse reverses without blocking focus or reduced motion', async () => {
  test.setTimeout(60_000)
  fs.mkdirSync(artifacts, { recursive: true })
  const fixture = fs.mkdtempSync(path.join(artifacts, 'tree-fixture-'))
  const source = (file: string) => JSON.stringify(path.join(root, file).replaceAll('\\', '/'))
  const calls = requests.map((request, index) => ({ ...request, id: `motion-${index + 1}`, requestKey: `motion-${index + 1}`, labels: index ? ['工具结果'] : ['搜索网页'] }))
  const entry = { ...turn, key: 'motion', alias: 'Qwen3.8 Flash · IQ3_XXS', requests: calls, expanded: false, matchedRequestCount: 1, error: '已保留的读取错误' }
  const tools = [{ id: 'tool-motion', kind: 'tool', requestId: 'motion-1', status: 'failed', elapsed: 3.2,
    call: { id: 'read', type: 'function', function: { name: 'read_file', arguments: JSON.stringify({ path: 'D:\\documents\\a-long-file-path\\source-reference.txt' }) } },
    result: JSON.stringify({ error: '文件不存在。已保留本次工具调用的错误详情，便于核对输入与请求记录。' }) }]
  fs.writeFileSync(path.join(fixture, 'index.html'), '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><link rel="stylesheet" href="fixture.css"><div id="root"></div><script src="fixture.js"></script></html>')
  fs.writeFileSync(path.join(fixture, 'fixture.tsx'), `import React,{useState} from 'react';import{createRoot}from'react-dom/client';import{MonitorRequestTable}from ${source('src/MonitorRequestTable')};import ${source('src/styles.css')};import ${source('src/resource-cards.css')};import ${source('src/monitor-request-history.css')};
    function Fixture(){const[entries,setEntries]=useState(${JSON.stringify([entry, { ...engineTurn, key: 'next', expanded: false }])});const[group,setGroup]=useState('motion');const[request,setRequest]=useState('');
      window.bump=()=>setEntries(rows=>rows.map(row=>({...row,requests:row.requests.map(item=>({...item,elapsed:(item.elapsed||0)+.1}))})));
      return <main style={{padding:24,maxWidth:1300,margin:'auto'}}><MonitorRequestTable entries={entries} selectedGroupId={group} selectedRequestId={request} tools={${JSON.stringify(tools)}}
        onSelectGroup={entry=>{setGroup(entry.key);setRequest('')}} onSelectRequest={(item,entry)=>{setGroup(entry.key);setRequest(item.id)}} onToggleGroup={entry=>setEntries(rows=>rows.map(row=>row.key===entry.key?{...row,expanded:!row.expanded}:row))}/></main>}
    createRoot(document.getElementById('root')).render(<Fixture/>);`)
  await build({ entryPoints: [path.join(fixture, 'fixture.tsx')], outfile: path.join(fixture, 'fixture.js'), bundle: true, format: 'iife', jsx: 'automatic', loader: { '.woff2': 'file' }, define: { 'process.env.NODE_ENV': '"development"' } })
  const browser = await chromium.launch({ channel: 'msedge', headless: true })
  try {
    const page = await browser.newPage({ viewport: { width: 1366, height: 900 } })
    const errors: string[] = []
    page.on('pageerror', error => errors.push(error.message))
    await page.goto(pathToFileURL(path.join(fixture, 'index.html')).href)
    const parent = page.locator('tr[data-group-key="motion"]'), tree = page.locator('tbody[data-tree-key="motion"]')
    const toggle = parent.locator('.monitor-ledger-expand')
    await expect(parent.locator('svg')).toHaveCount(0)
    await expect(parent.locator('[data-request-count]')).toHaveText('2 次请求')
    const opening = await page.evaluate(async () => {
      const button = document.querySelector<HTMLButtonElement>('[data-group-key="motion"] .monitor-ledger-expand')!
      button.click(); const heights: number[] = []
      for (let index = 0; index < 32; index++) { await new Promise(requestAnimationFrame); heights.push(document.querySelector('[data-tree-key="motion"]')!.getBoundingClientRect().height) }
      return heights
    })
    expect(opening.some(value => value > opening[0] + 1 && value < opening.at(-1)! - 1)).toBe(true)
    await expect(tree.locator('[data-request-key]')).toHaveCount(2)
    await expect(tree.locator('tr[data-request-key="motion-1"] > *')).toHaveCount(7)
    expect(await tree.locator('.monitor-ledger-collapse-frame').first().evaluate(node => getComputedStyle(node).transitionDuration)).toBe('0.2s, 0.2s')
    expect(await page.locator('.monitor-ledger-selection').evaluate(node => getComputedStyle(node).transitionDuration)).toBe('0.16s, 0.16s, 0.16s')
    const selection = await page.evaluate(async () => {
      document.querySelector<HTMLButtonElement>('[data-request-key="motion-1"] button')!.click()
      const positions: number[] = []
      for (let index = 0; index < 32; index++) { await new Promise(requestAnimationFrame); positions.push(document.querySelector('.monitor-ledger-selection')!.getBoundingClientRect().top) }
      return positions
    })
    expect(selection.some(value => value > selection[0] + 1 && value < selection.at(-1)! - 1)).toBe(true)
    await expect(tree.getByText('文件不存在。已保留本次工具调用的错误详情，便于核对输入与请求记录。')).toBeVisible()
    await expect.poll(() => tree.locator('.monitor-ledger-tools').evaluate(node => node.getBoundingClientRect().height)).toBeGreaterThan(40)
    await expect.poll(() => tree.locator('.monitor-ledger-collapse-frame').evaluateAll(nodes => nodes.flatMap(node => node.getAnimations()).length)).toBe(0)
    const rail = await tree.evaluate(node => {
      const center = (element: Element) => { const r = element.getBoundingClientRect(); return { x: r.x+r.width/2, y: r.y+r.height/2 } }
      return { dots: [...node.querySelectorAll('.monitor-ledger-dot')].map(center), lines: [...node.querySelectorAll('.monitor-ledger-branch')].map(element => { const r=element.getBoundingClientRect();return {x:r.x+r.width/2,top:r.top,bottom:r.bottom} }) }
    })
    expect(rail.lines).toHaveLength(5)
    for (const dot of rail.dots) expect(dot.x).toBeCloseTo(rail.dots[0].x, 1)
    expect(rail.lines[0].top).toBeCloseTo(rail.dots[0].y, 0)
    for (let index = 1; index < rail.lines.length; index++) expect(Math.abs(rail.lines[index].top - rail.lines[index-1].bottom)).toBeLessThanOrEqual(1)
    expect(rail.lines.at(-1)!.bottom).toBeCloseTo(rail.dots.at(-1)!.y, 0)
    await expect(parent.getByRole('img', { name: '已完成', exact: true })).toBeVisible()
    await page.screenshot({ path: path.join(artifacts, 'tree-tools-connected.png') })
    await page.evaluate(() => (window as any).bump())
    expect(await tree.locator('.monitor-ledger-collapse-frame').evaluateAll(nodes => nodes.flatMap(node => node.getAnimations()).length)).toBe(0)
    const interruptedExit = await page.evaluate(async () => {
      const frame = () => new Promise(requestAnimationFrame)
      document.querySelector<HTMLButtonElement>('[data-request-key="motion-2"] button')!.click(); await frame()
      document.querySelector<HTMLButtonElement>('[data-request-key="motion-1"] button')!.click(); await frame(); await frame()
      document.querySelector<HTMLButtonElement>('[data-group-key="motion"] .monitor-ledger-expand')!.click(); await frame()
      const animations = document.querySelector('[data-request-key="motion-1"] .monitor-ledger-collapse-frame')!.getAnimations()
      return await Promise.all(animations.map(animation => animation.finished.then(() => 'finished', () => 'cancelled')))
    })
    expect(interruptedExit.length).toBeGreaterThan(0)
    expect(interruptedExit.every(result => result === 'finished')).toBe(true)
    await expect(tree.locator('[data-request-key]')).toHaveCount(0)
    await toggle.click()
    await expect.poll(() => tree.locator('.monitor-ledger-collapse-frame').evaluateAll(nodes => nodes.flatMap(node => node.getAnimations()).length)).toBe(0)
    const reversing = await page.evaluate(async () => {
      const button = document.querySelector<HTMLButtonElement>('[data-group-key="motion"] .monitor-ledger-expand')!, body = document.querySelector('[data-tree-key="motion"]')!
      const heights: number[] = []
      for (let round = 0; round < 4; round++) { button.click(); for (let index = 0; index < 3; index++) { await new Promise(requestAnimationFrame); heights.push(body.getBoundingClientRect().height) } }
      return heights
    })
    expect(new Set(reversing.map(value => Math.round(value)) ).size).toBeGreaterThan(4)
    await expect(parent).toHaveAttribute('data-expanded', 'true')
    await expect(toggle).toBeEnabled()
    await tree.locator('[data-request-key="motion-2"] button').focus()
    await toggle.evaluate((button: HTMLButtonElement) => button.click())
    await expect(toggle).toBeFocused()
    await expect(tree.getByRole('button', { name: /^选择请求/ })).toHaveCount(0)
    await expect(tree.locator('[data-request-key]')).toHaveCount(0)
    await page.keyboard.press('Tab')
    await expect(parent.getByRole('button', { name: /^选择轮汇总/ })).toBeFocused()
    await toggle.focus(); await page.keyboard.press('Enter')
    await expect(parent).toHaveAttribute('data-expanded', 'true')
    await expect(toggle).toBeFocused()
    await expect.poll(() => tree.locator('[data-request-key="motion-2"]').evaluate(node => node.getBoundingClientRect().height)).toBeGreaterThanOrEqual(72)
    await page.screenshot({ path: path.join(artifacts, 'tree-expanded.png') })
    await page.emulateMedia({ reducedMotion: 'reduce' })
    await toggle.click()
    await expect(tree.locator('[data-request-key]')).toHaveCount(0)
    await toggle.press('Enter')
    await expect(tree.locator('[data-request-key="motion-1"]')).toBeVisible()
    expect(await tree.locator('.monitor-ledger-collapse-frame').evaluateAll(nodes => nodes.flatMap(node => node.getAnimations()).length)).toBe(0)
    await page.screenshot({ path: path.join(artifacts, 'tree-reduced-motion.png') })
    fs.writeFileSync(path.join(artifacts, 'tree-animation-samples.json'), JSON.stringify({ opening, selection, reversing, interruptedExit, rail }, null, 2))
    expect(errors).toEqual([])
  } finally { await browser.close() }
})
