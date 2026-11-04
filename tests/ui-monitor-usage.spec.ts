import { expect, test, chromium } from '@playwright/test'
import { build } from 'esbuild'
import * as fs from 'node:fs'
import * as path from 'node:path'
import type { RequestHistoryStats, RequestHistoryUsage } from '../src/types'

const root = path.resolve(__dirname, '..')
const artifacts = process.env.MODLOADER_TEST_ARTIFACTS || path.join(root, '.artifacts/codex/monitor-spacious-dev-20261031/tests')
const now = Date.parse('2026-10-31T01:18:00+08:00') / 1000
const start = now - 18 * 60
function stats(total: number, count = 2): RequestHistoryStats {
  return { request_count: count, total: { value: total, reported: count }, input: { value: total * .8, reported: count }, output: { value: total * .2, reported: count },
    cached: { value: total * .4, reported: count }, cache_pair: { input: total * .8, cached: total * .4, reported: count }, cache_ratio: .5,
    decode: { value: null, reported: 0, tokens: 0, seconds: 0 }, prefill: { value: null, reported: 0, tokens: 0, seconds: 0 },
    ttft: { value: null, reported: 0, engine_reported: 0, gateway_reported: 0 }, status_counts: { running: 0, completed: count, failed: 0, cancelled: 0, incomplete: 0 }, record_incomplete_count: 0 }
}
function snapshot(total = 15000, revision = 'one'): RequestHistoryUsage {
  const buckets = [100, 200, 400, 250, 500].map((value, index) => ({ ...stats(value * total / 15000), start_at: start + index * 300, end_at: start + (index + 1) * 300 }))
  buckets[2].total = { value: null, reported: 0 }
  return { as_of: now, revision, summary: stats(total, 100), coverage: { legacy_message_count: 0, legacy_metrics_message_count: 0 }, issues: [],
    trend: { start_at: start, end_at: start + 1500, bucket_seconds: 300, next_refresh_at: now + 120, buckets } }
}
async function fixture() {
  fs.mkdirSync(artifacts, { recursive: true })
  const directory = fs.mkdtempSync(path.join(artifacts, 'usage-'))
  const source = (file: string) => JSON.stringify(path.join(root, file).replaceAll('\\', '/'))
  fs.writeFileSync(path.join(directory, 'fixture.tsx'), `import React,{useState}from'react';import{createRoot}from'react-dom/client';import{MonitorUsage}from ${source('src/MonitorUsage')};import ${source('src/styles.css')};
    function Fixture(){const[state,setState]=useState({query:'date_from=2026-10-31&date_to=2026-10-31',revision:'one',live:false});window.setUsageState=patch=>setState(old=>({...old,...patch}));return <main style={{padding:24,maxWidth:1100,margin:'auto'}}><button id="outside">外部控件</button><div className="monitor-usage-hardware-panel"><MonitorUsage {...state} active refreshVersion={0}/></div></main>};createRoot(document.getElementById('root')).render(<Fixture/>);`)
  await build({ entryPoints: [path.join(directory, 'fixture.tsx')], outfile: path.join(directory, 'fixture.js'), bundle: true, format: 'iife', jsx: 'automatic', loader: { '.woff2': 'file' }, define: { 'process.env.NODE_ENV': '"development"' } })
  const browser = await chromium.launch({ channel: 'msedge', headless: true })
  const page = await browser.newPage({ viewport: { width: 1180, height: 1000 } })
  await page.clock.install({ time: new Date(now * 1000) })
  const errors: string[] = [], reads: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  let response = snapshot(), slowRelease: (() => void) | undefined
  await page.route('http://usage.fixture/**', async route => {
    const url = new URL(route.request().url())
    if (url.pathname === '/api/request-history/usage') {
      reads.push(url.search)
      const captured = structuredClone(response)
      if (url.searchParams.get('instance_id') === 'slow') await new Promise<void>(resolve => { slowRelease = resolve })
      return route.fulfill({ json: captured })
    }
    if (url.pathname === '/') return route.fulfill({ contentType: 'text/html', body: '<!doctype html><meta charset="utf-8"><link rel="stylesheet" href="fixture.css"><div id="root"></div><script src="fixture.js"></script>' })
    const file = path.join(directory, path.basename(url.pathname))
    return route.fulfill({ path: file })
  })
  await page.goto('http://usage.fixture/')
  const usage = page.locator('.monitor-usage')
  await expect(usage.locator('[data-summary="total"]')).toHaveText('15,000')
  return { page, browser, directory, usage, reads, errors, respond: (value: RequestHistoryUsage) => { response = value }, release: () => slowRelease?.(),
    patch: async (value: object) => page.evaluate(value => (window as any).setUsageState(value), value) }
}

test('用量图：固定槽、最后槽/真实区间、无常驻点、图例和一次partial提示、键盘焦点', async () => {
  const f = await fixture()
  try {
    await f.usage.getByRole('button', { name: '展开 Token 用量趋势', exact: true }).click()
    await f.page.clock.runFor(250)
    const chart = f.usage.locator('.monitor-usage-chart'), svg = chart.locator('svg'), slots = chart.locator('button[data-usage-slot]')
    const expandedHeight = await chart.evaluate(node => node.getBoundingClientRect().height)
    await expect.poll(() => f.usage.locator('.monitor-usage-collapse').evaluate(node => node.getBoundingClientRect().height)).toBeGreaterThanOrEqual(expandedHeight - 1)
    await expect(slots).toHaveCount(5)
    await expect(svg.locator('circle')).toHaveCount(0)
    const geometry = await svg.evaluate(node => ({ width: node.viewBox.baseVal.width, left: 64, right: node.viewBox.baseVal.width - 48 }))
    const positions = await slots.evaluateAll(nodes => nodes.map(node => Number((node as HTMLElement).dataset.slotX)))
    expect(positions[0]).toBe(geometry.left); expect(positions.at(-1)).toBe(geometry.right)
    const gaps = positions.slice(1).map((value, index) => value - positions[index])
    gaps.forEach(gap => expect(gap).toBeCloseTo(gaps[0], 3))
    await slots.last().focus()
    await expect(chart.locator('[data-usage-guide]')).toHaveAttribute('d', `M${geometry.right} 22V190`)
    for (const marker of await svg.locator('circle').all()) await expect(marker).toHaveAttribute('cx', String(geometry.right))
    const tooltip = chart.getByRole('tooltip')
    await expect(tooltip).toContainText('01:20–01:25')
    await expect(tooltip.locator('[data-usage-tooltip-row]')).toHaveCount(2)
    await expect(chart.locator('[data-usage-range]')).toContainText('01:00—01:25')
    await f.page.screenshot({ path: path.join(f.directory, 'usage-tooltip-last-slot.png') })
    await chart.getByRole('button', { name: '输入（含缓存）', exact: true }).click()
    await slots.nth(2).focus()
    await expect(tooltip.locator('[data-usage-tooltip-row]')).toHaveCount(3)
    await expect(tooltip.getByText('部分数据', { exact: true })).toHaveCount(1)
    await expect(tooltip).not.toContainText('报告')
    await expect(svg.locator('[data-usage-marker="total"]')).toHaveCount(0)
    await f.page.keyboard.press('ArrowRight'); await expect(slots.nth(3)).toBeFocused()
    await f.page.keyboard.press('Escape'); await expect(f.usage.getByRole('button', { name: '收起 Token 用量趋势' })).toBeFocused()
    await f.page.screenshot({ path: path.join(f.directory, 'usage-open.png') })
    expect(f.errors).toEqual([])
  } finally { await f.browser.close() }
})

test('整个summary+chart快照仅完成revision/边界更新，hover/focus延后，筛选立即且旧响应不复活', async () => {
  const f = await fixture()
  try {
    await f.usage.getByRole('button', { name: '展开 Token 用量趋势' }).click()
    const chart = f.usage.locator('.monitor-usage-chart'), total = f.usage.locator('[data-summary="total"]')
    const previousLine = await chart.locator('[data-usage-series="total"]').getAttribute('d')
    await chart.hover(); f.respond(snapshot(20000, 'two')); await f.patch({ revision: 'two' })
    await expect.poll(() => f.reads.length).toBe(2)
    await expect(f.usage).toHaveAttribute('data-usage-pending', 'true'); await expect(total).toHaveText('15,000')
    await expect(chart.locator('[data-usage-series="total"]')).toHaveAttribute('d', previousLine!)
    await f.page.locator('#outside').hover(); await expect(total).toHaveText('20,000')
    await expect(f.usage).toHaveAttribute('data-usage-revision', 'two')
    await chart.locator('button[data-usage-slot]').last().focus(); f.respond(snapshot(30000, 'three')); await f.patch({ revision: 'three' })
    await expect(f.usage).toHaveAttribute('data-usage-pending', 'true'); await expect(total).toHaveText('20,000')
    await f.page.locator('#outside').focus(); await expect(total).toHaveText('30,000')
    for (let index = 0; index < 10; index++) await f.patch({ live: false })
    await f.patch({ live: true }); await f.page.clock.runFor(119000); expect(f.reads).toHaveLength(3)
    const next = snapshot(40000, 'three'); next.trend.next_refresh_at = now + 420; f.respond(next)
    await f.page.clock.runFor(1000); await expect(total).toHaveText('40,000'); expect(f.reads).toHaveLength(4)
    await f.patch({ query: 'date_from=2026-10-31&date_to=2026-10-31&instance_id=slow' })
    await expect.poll(() => f.reads.length).toBe(5)
    f.respond(snapshot(50000, 'three')); await f.patch({ query: 'date_from=2026-10-31&date_to=2026-10-31&instance_id=new' })
    await expect(total).toHaveText('50,000'); f.release()
    await expect(total).toHaveText('50,000')
    expect(f.reads).toHaveLength(6); expect(f.errors).toEqual([])
    await f.page.screenshot({ path: path.join(f.directory, 'usage-new-filter.png') })
  } finally { f.release(); await f.browser.close() }
})

test('单桶仍可键盘查看真实值但不发明静态线；空筛选保留真实0与空图', async () => {
  const f = await fixture()
  try {
    await f.usage.getByRole('button', { name: '展开 Token 用量趋势' }).click()
    const single = snapshot(900)
    single.trend.buckets = [single.trend.buckets[0]]
    single.trend.end_at = single.trend.buckets[0].end_at
    f.respond(single); await f.patch({ query: 'instance_id=single' })
    const chart = f.usage.locator('.monitor-usage-chart'), slot = chart.locator('button[data-usage-slot]')
    await expect(slot).toHaveCount(1)
    for (const line of await chart.locator('[data-usage-series], [data-usage-area]').all()) await expect(line).toHaveAttribute('d', '')
    await slot.focus()
    const center = await chart.locator('svg').evaluate(node => (64 + node.viewBox.baseVal.width - 48) / 2)
    await expect(slot).toHaveAttribute('data-slot-x', String(center))
    await expect(chart.locator('[data-usage-marker="total"]')).toHaveAttribute('cx', String(center))
    await expect(chart.getByRole('tooltip')).toContainText('6')
    await f.page.locator('#outside').focus()
    const empty = snapshot(0)
    empty.summary = stats(0, 0); empty.summary.cache_ratio = null
    empty.trend = { start_at: null, end_at: null, buckets: [], bucket_seconds: 60, next_refresh_at: now + 60 }
    f.respond(empty); await f.patch({ query: 'instance_id=empty' })
    await expect(f.usage.locator('[data-summary="total"]')).toHaveText('0')
    await expect(f.usage.locator('[data-summary="count"]')).toHaveText('0')
    await expect(f.usage.locator('[data-summary="hit"]')).toHaveText('—')
    await expect(chart.locator('button[data-usage-slot], circle')).toHaveCount(0)
    await expect(chart.getByText('没有符合筛选条件的记录', { exact: true })).toBeVisible()
    expect(f.errors).toEqual([])
  } finally { await f.browser.close() }
})

test('中间和末尾空桶命中曲线归零但读数仍未知；有请求缺测不画0，单空桶不发明线', async () => {
  const f = await fixture()
  try {
    const data = snapshot(15000, 'empty-buckets')
    for (const index of [2, 4]) data.trend.buckets[index] = { ...data.trend.buckets[index], ...stats(0, 0), cache_ratio: null, cache_pair: { input: null, cached: null, reported: 0 } }
    data.trend.buckets[1].cache_ratio = null
    data.trend.buckets[1].cache_pair = { input: null, cached: null, reported: 0 }
    f.respond(data); await f.patch({ query: 'instance_id=empty-buckets' })
    await expect(f.usage).toHaveAttribute('data-usage-revision', 'empty-buckets')
    await f.usage.getByRole('button', { name: '展开 Token 用量趋势' }).click(); await f.page.clock.runFor(250)
    const chart = f.usage.locator('.monitor-usage-chart'), slots = chart.locator('button[data-usage-slot]'), hit = chart.locator('[data-usage-series="hit"]')
    const expandedHeight = await chart.evaluate(node => node.getBoundingClientRect().height)
    await expect.poll(() => f.usage.locator('.monitor-usage-collapse').evaluate(node => node.getBoundingClientRect().height)).toBeGreaterThanOrEqual(expandedHeight - 1)
    const positions = await slots.evaluateAll(nodes => nodes.map(node => Number((node as HTMLElement).dataset.slotX)))
    await expect(slots).toHaveCount(5); await expect(chart.locator('circle')).toHaveCount(0)
    expect(await hit.getAttribute('d')).toContain(`${positions[2]},190`)
    expect(await hit.getAttribute('d')).toMatch(new RegExp(`${positions[4]},190$`))
    await slots.nth(1).focus()
    await expect(chart.locator('[data-usage-marker="hit"]')).toHaveCount(0)
    await expect(chart.locator('[data-usage-tooltip-row="hit"] strong')).toHaveText('—')
    await expect(chart.getByRole('tooltip')).toContainText('2 次请求')
    for (const index of [2, 4]) {
      await slots.nth(index).focus()
      await expect(chart.locator('[data-usage-marker="hit"]')).toHaveAttribute('cx', String(positions[index]))
      await expect(chart.locator('[data-usage-marker="hit"]')).toHaveAttribute('cy', '190')
      await expect(chart.locator('[data-usage-tooltip-row="hit"] strong')).toHaveText('—')
      await expect(chart.getByRole('tooltip')).toContainText('0 次请求')
      expect(data.trend.buckets[index].cache_ratio).toBeNull()
    }
    await f.page.screenshot({ path: path.join(f.directory, 'empty-middle-and-last.png') })
    await f.page.locator('#outside').focus()
    data.trend.buckets = [data.trend.buckets[4]]; data.trend.start_at = data.trend.buckets[0].start_at
    f.respond(data); await f.patch({ query: 'instance_id=single-empty-bucket' })
    await expect(slots).toHaveCount(1); await expect(hit).toHaveAttribute('d', '')
    await slots.first().focus(); await expect(chart.locator('[data-usage-marker="hit"]')).toHaveAttribute('cy', '190')
    await expect(chart.locator('[data-usage-tooltip-row="hit"] strong')).toHaveText('—')
    await expect(chart.getByRole('tooltip')).toContainText('0 次请求')
    await f.page.screenshot({ path: path.join(f.directory, 'single-empty-bucket.png') })
    expect(f.errors).toEqual([])
  } finally { await f.browser.close() }
})
