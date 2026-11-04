import { test, expect, chromium } from '@playwright/test'
import { createServer } from 'node:http'
import * as path from 'node:path'
import * as fs from 'node:fs'
import { pinNavigation } from './navigation'
import { historyPage, historyUsage } from './monitor-history-fixture'

test('监控：后台每秒采样不闪烁，手动刷新合并请求并保留读数', async () => {
  test.setTimeout(90_000)
  const root = path.resolve(__dirname, '..')
  const artifacts = process.env.MODLOADER_TEST_ARTIFACTS || path.join(root, '.artifacts/codex/monitor-implementation-20261031/legacy-ui')
  fs.mkdirSync(artifacts, { recursive: true })
  const directory = fs.mkdtempSync(path.join(artifacts, 'polling-ui-'))
  const screenshots = path.join(directory, 'screenshots')
  fs.mkdirSync(screenshots)
  const instances = ['monitor-a', 'monitor-b'].map((id, index) => ({
    id, alias: `监控 fixture ${index + 1}`, backend: 'llama_cpp', state: 'ready', model_id: `${id}-model`,
    profile_id: `${id}-profile`, profile_revision: 1, runtime_id: 'fixture-only', port: 0, started_at: '', active_requests: 0, capabilities: {},
  }))
  type Sample = { id: string; sequence: number; started: number; mode: string; finished?: number; aborted: boolean }
  const samples: Sample[] = []
  const activeReads = new Map<string, number>()
  let maxConcurrent = 0
  let nextMode: 'delay' | 'hold' | 'failure' = 'delay'
  let pending: { sample: Sample; release: (value?: number) => void } | undefined
  let stateReads = 0
  const errors: string[] = []
  const externalRequests: string[] = []
  const mutations: string[] = []
  const historyWrites: string[] = []
  const unexpectedPaths: string[] = []
  const server = createServer((request, response) => {
    response.setHeader('Access-Control-Allow-Origin', '*')
    response.setHeader('Access-Control-Allow-Headers', '*')
    if (request.method === 'OPTIONS') { response.end(); return }
    const id = /^\/api\/instances\/(monitor-[ab])\/telemetry$/.exec(request.url || '')?.[1]
    if (!id || request.method !== 'GET') { unexpectedPaths.push(`${request.method} ${request.url}`); response.writeHead(404).end(); return }
    const sample: Sample = { id, sequence: samples.length + 1, started: Date.now(), mode: nextMode, aborted: false }
    nextMode = 'delay'
    samples.push(sample)
    activeReads.set(id, (activeReads.get(id) || 0) + 1)
    maxConcurrent = Math.max(maxConcurrent, activeReads.get(id)!)
    response.once('close', () => {
      activeReads.set(id, activeReads.get(id)! - 1)
      sample.aborted = !response.writableEnded
    })
    const release = (value = (id === 'monitor-a' ? 1000 : 2000) + sample.sequence) => {
      if (response.destroyed) return
      sample.finished = Date.now()
      response.setHeader('Content-Type', 'application/json')
      if (sample.mode === 'failure') { response.writeHead(503).end(JSON.stringify({ detail: '监控 fixture 暂不可用' })); return }
      response.end(JSON.stringify({ instance_id: id, backend: 'llama_cpp', state: 'ready', updated_at: sample.finished / 1000,
        live_generation: { request_key: `${id}-request`, backend_running: 1, backend_queued: 0,
          tokens_per_second: 130 + sample.sequence / 100, average_tokens_per_second: 144.1,
          average_basis: 'backend_decode', context_tokens: 100000 + sample.sequence * 130,
          context_capacity: 262144, window_s: 3, phase: 'generating' },
        groups: [{ id: 'fixture', title: `${id} 运行采样`, metrics: [{ id: 'sample', label: '采样序号', value, kind: 'gauge' }] }], unavailable: [],
      }))
    }
    if (sample.mode === 'hold') pending = { sample, release }
    else setTimeout(release, 350)
  })
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const address = server.address() as { port: number }
  const { preview } = await import('vite')
  const previewServer = await preview({ root, configFile: false,
    build: { outDir: process.env.MODLOADER_TEST_DIST || path.join(root, 'dist') }, preview: { host: '127.0.0.1', port: 0 } })
  const previewAddress = previewServer.httpServer.address()
  if (!previewAddress || typeof previewAddress === 'string') throw new Error('Fixture preview did not open')
  const browser = await chromium.launch({ channel: 'msedge', headless: true })
  const page = await browser.newPage({ viewport: { width: 1440, height: 1040 } })
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/*', async route => {
    const request = route.request()
    const url = new URL(request.url())
    const historyWrite = ['POST', 'PATCH'].includes(request.method()) && /^\/api\/conversations(?:\/[^/]+)?$/.test(url.pathname)
    if (['http:', 'https:'].includes(url.protocol) && !['127.0.0.1', 'localhost'].includes(url.hostname)) {
      externalRequests.push(url.origin); await route.abort()
    } else if (url.pathname.startsWith('/api/') && !['GET', 'HEAD', 'OPTIONS'].includes(request.method()) && !historyWrite) {
      mutations.push(`${request.method()} ${url.pathname}`); await route.abort()
    } else {
      if (historyWrite) historyWrites.push(`${request.method()} ${url.pathname}`)
      await route.continue()
    }
  })
  await page.route('**/api/**', async route => {
    const request = route.request(), url = new URL(request.url())
    if (request.method() !== 'GET') {
      mutations.push(`${request.method()} ${url.pathname}`)
      return route.abort()
    }
    if (url.pathname === '/api/state') {
      stateReads++
      return route.fulfill({ json: { models: [], profiles: [], runtimes: [], instances, downloads: [], installations: [], settings: {},
        backends: [{ id: 'llama_cpp', label: 'llama.cpp', parameters: [] }], hardware: { cpu_info: { name: 'isolated fixture' }, gpus: [] } } })
    }
    if (url.pathname === '/api/chat-presets') return route.fulfill({ json: { items: [], default_id: null } })
    if (url.pathname === '/api/conversations') return route.fulfill({ json: { items: [], active_id: null } })
    if (url.pathname === '/api/request-history/dates') return route.fulfill({ json: { current_run_id: 'monitor-run', today: new Date().toLocaleDateString('en-CA'), dates: [], issues: [] } })
    if (url.pathname === '/api/request-history') return route.fulfill({ json: historyPage([], url.searchParams) })
    if (url.pathname === '/api/request-history/usage') return route.fulfill({ json: historyUsage() })
    unexpectedPaths.push(`${request.method()} ${url.pathname}`)
    return route.fulfill({ status: 404, json: { detail: `Unexpected fixture: ${url.pathname}` } })
  })
  await page.route('**/api/instances/*/telemetry', route => route.continue({ url: `http://127.0.0.1:${address.port}${new URL(route.request().url()).pathname}` }))
  const navigation = page.getByRole('navigation', { name: '主导航' })
  const refresh = page.getByRole('button', { name: '刷新到最新', exact: true })
  const metric = page.locator('.embedded-metric').filter({ hasText: '采样序号' }).locator('.monitor-metric-value strong')
  const spinner = page.locator('.monitor-runtime-heading .spin')
  async function holdNext() {
    pending = undefined; nextMode = 'hold'
    await expect.poll(() => Boolean(pending), { timeout: 4_000 }).toBe(true)
    return pending!
  }
  async function screenshot(name: string) {
    await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))))
    await page.screenshot({ path: path.join(screenshots, name) })
  }
  try {
    page.setDefaultTimeout(8000)
    await page.goto(`http://127.0.0.1:${previewAddress.port}`)
    await expect(page.locator('.app-shell')).toBeVisible({ timeout: 60_000 })
    await pinNavigation(page)
    await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await navigation.getByRole('button', { name: '运行监控', exact: true }).click()
    await page.locator('.monitor-backend-details > summary').click()
    await expect(metric).toHaveText(/^1,\d{3}$/)
    await expect(spinner).toHaveCount(0)
    await page.evaluate(() => document.fonts.ready)

    await test.step('连续三个自动更新保持按钮可用、透明度稳定及已有指标可见', async () => {
      const before = samples.length
      await refresh.evaluate(element => {
        const button = element as HTMLButtonElement
        const capture = { frames: 0, violations: [] as string[], readings: [] as string[], frame: 0 }
        const opacity = getComputedStyle(button).opacity
        const sample = () => {
          capture.frames++
          if (button.disabled) capture.violations.push('disabled')
          if (getComputedStyle(button).opacity !== opacity) capture.violations.push('opacity')
          if (document.querySelector('.monitor-runtime-heading .spin, .monitor-wait .spin')) capture.violations.push('spinner')
          const reading = document.querySelector('.embedded-metric .monitor-metric-value strong')?.textContent || ''
          if (!reading) capture.violations.push('cleared')
          if (capture.readings.at(-1) !== reading) capture.readings.push(reading)
          capture.frame = requestAnimationFrame(sample)
        }
        ;(window as any).monitorCapture = capture
        sample()
      })
      await expect.poll(() => samples.filter(sample => sample.sequence > before && sample.finished).length, { timeout: 6_000 }).toBeGreaterThanOrEqual(3)
      await expect.poll(async () => Number((await metric.textContent())?.replaceAll(',', '')) === 1000 + samples.filter(sample => sample.finished && !sample.aborted).at(-1)!.sequence).toBe(true)
      await expect.poll(() => page.evaluate(() => (window as any).monitorCapture.readings.length), { timeout: 2_000 }).toBeGreaterThanOrEqual(4)
      const capture = await page.evaluate(() => {
        const capture = (window as any).monitorCapture
        cancelAnimationFrame(capture.frame); delete (window as any).monitorCapture
        return capture as { frames: number; violations: string[]; readings: string[] }
      })
      const violationCounts = capture.violations.reduce<Record<string, number>>((counts, issue) => { counts[issue] = (counts[issue] || 0) + 1; return counts }, {})
      console.log(JSON.stringify({ automaticUpdates: capture.readings.length - 1, sampledFrames: capture.frames, violationCounts, screenshots }))
      expect(capture.frames).toBeGreaterThan(30)
      expect(violationCounts).toEqual({})
      expect(capture.readings.length).toBeGreaterThanOrEqual(4)
      const polls = samples.filter(sample => sample.sequence > before).slice(0, 3)
      for (let index = 1; index < polls.length; index++) {
        expect(polls[index].started - polls[index - 1].started).toBeGreaterThanOrEqual(700)
        expect(polls[index].started - polls[index - 1].started).toBeLessThan(1800)
      }
      await screenshot('monitor-stable-polling.png')
    })

    await test.step('在对话页停留后返回监控，当前阶段主图继续采样', async () => {
      const before = samples.length, states = stateReads
      await navigation.getByRole('button', { name: '测试对话', exact: true }).click()
      await expect(page.locator('.chat-section')).toBeVisible()
      await expect.poll(() => stateReads, { timeout: 7_000 }).toBeGreaterThanOrEqual(states + 4)
      const backgroundSamples = samples.slice(before).filter(sample => sample.finished && !sample.aborted)
      await navigation.getByRole('button', { name: '运行监控', exact: true }).click()
      await expect(metric).toHaveText(/^1,\d{3}$/)
      await expect.poll(() => samples.filter(sample => sample.sequence > before && sample.finished).length).toBeGreaterThan(0)
      await page.locator('.monitor-compact-generation').scrollIntoViewIfNeeded()
      await screenshot('monitor-after-chat.png')
      console.log(JSON.stringify({ backgroundSamples: backgroundSamples.length, screenshots }))
      expect(backgroundSamples.length).toBeGreaterThanOrEqual(3)
      await expect(page.locator('.monitor-generation-chart .trend-line')).toHaveCount(1)
      await expect(page.locator('[data-resource="average"] .trend-plot, [data-resource="context"] .trend-plot')).toHaveCount(0)
    })

    await test.step('在慢轮询期间手动刷新加入当前请求，期间保留最后读数', async () => {
      const held = await holdNext()
      const previous = await metric.textContent()
      const reads = samples.length
      await expect(refresh).toBeEnabled()
      await expect(refresh).toHaveCSS('opacity', '1')
      await expect(spinner).toHaveCount(0)
      await screenshot('monitor-background-in-flight.png')
      await refresh.click()
      await expect(refresh).toBeDisabled()
      await expect(metric).toHaveText(previous!)
      await page.waitForTimeout(1_200)
      expect(samples).toHaveLength(reads)
      expect(held.sample.aborted).toBe(false)
      await screenshot('monitor-manual-joined-poll.png')
      held.release()
      await expect(metric).toHaveText((1000 + held.sample.sequence).toLocaleString('zh-CN'))
      await expect(refresh).toBeEnabled()
      expect(maxConcurrent).toBe(1)
    })

    await test.step('自动采样失败保留最后读数，后续成功采样清除错误', async () => {
      nextMode = 'failure'
      await expect.poll(() => samples.at(-1)?.mode, { timeout: 4_000 }).toBe('failure')
      const previous = await metric.textContent()
      const failed = samples.at(-1)!
      await expect(page.getByRole('alert')).toContainText('监控 fixture 暂不可用')
      await expect(metric).toHaveText(previous!)
      await expect(refresh).toBeEnabled()
      await expect(spinner).toHaveCount(0)
      await screenshot('monitor-failed-last-reading.png')
      await expect.poll(() => samples.some(sample => sample.sequence > failed.sequence && sample.finished), { timeout: 4_000 }).toBe(true)
      await expect(page.getByRole('alert')).toHaveCount(0)
      await expect(metric).not.toHaveText(previous!)
    })

    await test.step('切换实例取消旧采样，隐藏页面仍完成当前实例的采样', async () => {
      const old = await holdNext()
      await refresh.click()
      await page.getByRole('combobox', { name: '选择监控实例', exact: true }).selectOption('monitor-b')
      await expect.poll(() => old.sample.aborted).toBe(true)
      await expect(page.getByRole('heading', { name: 'monitor-b 运行采样', exact: true })).toBeVisible()
      await expect(refresh).toBeEnabled()
      old.release(777777)
      await expect(metric).toHaveText(/^2,\d{3}$/)
      const hidden = await holdNext()
      await navigation.getByRole('button', { name: '设置', exact: true }).click()
      const afterLeaving = samples.length
      const states = stateReads
      await expect.poll(() => stateReads, { timeout: 4_000 }).toBeGreaterThanOrEqual(states + 2)
      expect(hidden.sample.aborted).toBe(false)
      // The pending request is shared rather than overlapped while the page is hidden.
      expect(samples).toHaveLength(afterLeaving)
      hidden.release()
      await expect.poll(() => samples.length, { timeout: 4_000 }).toBeGreaterThan(afterLeaving)
    })
    await test.step('唯一模型自动跟随，加载与停止清除旧运行指标，无实例时仍可刷新硬件', async () => {
      instances[1].state = 'stopped'
      await page.getByRole('button', { name: '刷新状态', exact: true }).click()
      await navigation.getByRole('button', { name: '运行监控', exact: true }).click()
      const selector = page.getByRole('combobox', { name: '选择监控实例', exact: true })
      const monitor = page.locator('.monitor-page')
      await expect(selector).toHaveCount(0)
      await expect(monitor.locator('.model-instance-name')).toContainText('监控 fixture 1')
      await expect(metric).toHaveText(/^1,\d{3}$/)
      instances[0].state = 'stopped'; instances[1].state = 'loading'
      await refresh.click()
      await expect(selector).toHaveCount(0)
      await expect(monitor.locator('.model-instance-name')).toContainText('监控 fixture 2')
      await expect(monitor.locator('.monitor-selector-main .badge')).toContainText('加载中')
      await expect(monitor.locator('.monitor-backend-details, .monitor-last-request')).toHaveCount(0)
      await expect(monitor.locator('.monitor-generation-idle')).toHaveText('尚无模型请求')
      await expect(monitor.locator('.monitor-compact-generation .trend-line')).toHaveCount(0)
      const beforeReady = samples.length
      const beforeState = stateReads
      await expect.poll(() => stateReads).toBeGreaterThanOrEqual(beforeState + 2)
      expect(samples).toHaveLength(beforeReady)
      instances[1].state = 'ready'
      await refresh.click()
      await expect(page.getByRole('heading', { name: 'monitor-b 运行采样', exact: true })).toBeVisible()
      await expect(metric).toHaveText(/^2,\d{3}$/)
      instances[1].state = 'stopped'
      await refresh.click()
      await expect(monitor.locator('.monitor-selector-main .badge')).toContainText('已停止')
      await expect(monitor.locator('.monitor-backend-details, .monitor-last-request')).toHaveCount(0)
      instances.splice(0)
      await refresh.click()
      await expect(selector).toHaveCount(0)
      await expect(monitor.locator('.monitor-selector-main')).toContainText('暂无运行模型')
      await expect(monitor.locator('.monitor-generation-chart, .monitor-backend-details')).toHaveCount(0)
      await expect(monitor.getByText('加载一个模型以开始监控', { exact: true })).toBeVisible()
      await expect(refresh).toBeEnabled()
    })
    expect(errors).toEqual([])
    expect(externalRequests).toEqual([])
    expect(mutations).toEqual([])
    expect(unexpectedPaths).toEqual([])
    console.log(JSON.stringify({ telemetryRequests: samples.length, maxConcurrentPerInstance: maxConcurrent, pageErrors: errors, historyWrites, screenshots }))
  } finally {
    pending?.release()
    await page.unrouteAll({ behavior: 'wait' })
    await browser.close()
    await new Promise<void>((resolve, reject) => previewServer.httpServer.close(error => error ? reject(error) : resolve()))
    server.closeAllConnections()
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
})



