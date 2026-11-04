import { test, expect, chromium, type Browser, type Locator } from '@playwright/test'
import * as fs from 'node:fs'
import * as path from 'node:path'
import type { PreviewServer } from 'vite'
import { pinNavigation } from './navigation'
import { historyPage, historyUsage } from './monitor-history-fixture'

const root = path.resolve(__dirname, '..')
const artifacts = process.env.MODLOADER_TEST_ARTIFACTS || path.join(root, '.artifacts/codex/number-layout-fix-20261101/main')
let server: PreviewServer, browser: Browser, baseURL: string

test.beforeAll(async () => {
  test.setTimeout(90_000)
  fs.mkdirSync(artifacts, { recursive: true })
  const { build, preview } = await import('vite')
  const outDir = process.env.MODLOADER_TEST_DIST || path.join(artifacts, 'fixture-dist')
  if (!process.env.MODLOADER_TEST_DIST) await build({ root, cacheDir: path.join(artifacts, 'vite-cache'), build: { outDir } })
  server = await preview({ root, configFile: false, build: { outDir }, preview: { host: '127.0.0.1', port: 0 } })
  const address = server.httpServer?.address()
  if (!address || typeof address === 'string') throw new Error('Number-layout fixture did not open')
  baseURL = `http://127.0.0.1:${address.port}`
  browser = await chromium.launch({ channel: 'msedge', headless: true })
})

test.afterAll(async () => { await browser?.close(); await new Promise<void>(resolve => server?.httpServer.close(() => resolve())) })

async function geometry(readout: Locator) {
  return readout.evaluate(node => {
    const value = node.querySelector('strong')!, unit = node.querySelector('small')
    const rect = (element: Element) => { const box = element.getBoundingClientRect(); return { x: box.x, y: box.y, right: box.right, bottom: box.bottom, width: box.width, height: box.height } }
    const range = document.createRange(); range.selectNodeContents(value)
    return { box: rect(node), value: rect(value), unit: unit ? rect(unit) : null,
      lines: range.getClientRects().length, text: value.textContent, textWidth: range.getBoundingClientRect().width }
  })
}

for (const { width, scale } of [{ width: 1366, scale: 1 }, { width: 1366, scale: 1.3 }, { width: 780, scale: 1.3 }, { width: 390, scale: 1.3 }]) {
  test(`主监控数值不跳行且完整可见 ${width}px ${scale}x`, async () => {
    test.setTimeout(90_000)
    const directory = path.join(artifacts, `${width}-${scale}`)
    fs.mkdirSync(directory, { recursive: true })
    const page = await browser.newPage({ viewport: { width, height: 1100 } })
    const errors: string[] = [], unexpected: string[] = []
    let speed = 99.99, sampledAt = Math.floor(Date.now() / 1000)
    const instance = { id: 'number-layout', model_id: 'fixture-model', profile_id: 'fixture-profile', alias: '数值布局 fixture', backend: 'strata', state: 'ready', active_requests: 1, capabilities: {} }
    try {
      page.setDefaultTimeout(8000)
      page.on('pageerror', error => errors.push(error.message))
      await page.route('**/api/**', route => {
        const url = new URL(route.request().url())
        if (url.pathname.endsWith('/generation-events') && route.request().method() === 'GET') return route.fulfill({ contentType: 'text/event-stream', body: ': fixture idle\n\n' })
        if (route.request().method() !== 'GET') { unexpected.push(`${route.request().method()} ${url.pathname}`); return route.abort() }
        if (url.pathname === '/api/state') return route.fulfill({ json: {
          models: [], profiles: [], runtimes: [], downloads: [], installations: [], instances: [instance], settings: { font_scale: scale },
          backends: [{ id: 'strata', label: 'Strata', parameters: [] }], hardware: {
            sampled_at: sampledAt, cpu_count: 32, cpu_info: { name: 'CPU fixture' }, cpu_percent: 99.99,
            cpu_power_w: 1000.25, cpu_temperature_c: 99.99, cpu_temperature_status: 'ready',
            gpus: [{ id: 'gpu-fixture', name: 'GPU fixture', utilization: 100, power: 10000.25, temperature: 100,
              pcie_rx_bytes_s: 1.25e9, pcie_tx_bytes_s: 1000.25e9 }], disk_read_bytes_per_second: 10000.25e6,
          },
        } })
        if (url.pathname === '/api/instances/number-layout/telemetry') return route.fulfill({ json: {
          instance_id: instance.id, backend: 'strata', state: 'ready', updated_at: sampledAt, unavailable: [], groups: [],
          live_generation: { scope: 'request', source: 'fixture', request_key: 'layout-request', client_request_id: 'layout-turn',
            backend_running: 1, backend_queued: 0, phase: 'output', available_for_chat: true,
            tokens_per_second: speed, average_tokens_per_second: speed, average_basis: 'backend_decode', window_s: 3,
            context_tokens: 1234567890, context_capacity: 2000000000, prompt_tokens: 1234500000, generated_tokens: 67890,
            first_output_s: 10000.25, request_elapsed_s: 100000.25 },
        } })
        if (url.pathname === '/api/chat-presets') return route.fulfill({ json: { items: [], default_id: null } })
        if (url.pathname === '/api/conversations') return route.fulfill({ json: { items: [], active_id: null } })
        if (url.pathname === '/api/request-history/dates') return route.fulfill({ json: { current_run_id: 'layout-run', today: new Date().toLocaleDateString('en-CA'), dates: [], issues: [] } })
        if (url.pathname === '/api/request-history') return route.fulfill({ json: historyPage([], url.searchParams) })
        if (url.pathname === '/api/request-history/usage') return route.fulfill({ json: historyUsage() })
        unexpected.push(url.pathname)
        return route.fulfill({ status: 404, json: { detail: `Unexpected fixture: ${url.pathname}` } })
      })
      await page.goto(baseURL, { timeout: 60_000 })
      await pinNavigation(page)
      await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
      await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '运行监控', exact: true }).click()
      const monitor = page.locator('.monitor-page'), panel = monitor.locator('.monitor-compact-generation')
      const group = monitor.locator('tr[data-group-key]').first()
      await expect(group).toBeVisible()
      if (await group.getAttribute('data-expanded') !== 'true') await group.locator('td').last().click()
      await monitor.locator('tr[data-request-key="layout-request"]').locator('td').last().click()
      await page.evaluate(() => document.fonts.ready)
      const readout = monitor.locator('.monitor-speed-current')
      await expect(readout.locator('strong')).toHaveText('99.99')
      const samples: Array<{ speed: number; geometry: Awaited<ReturnType<typeof geometry>> }> = []
      for (const value of [99.99, 100, 999.99, 1000, 10000, 1000000]) {
        speed = value; sampledAt++
        await monitor.getByRole('button', { name: '刷新到最新', exact: true }).click()
        await expect(readout.locator('strong')).toHaveText(value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }))
        await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => resolve())))
        samples.push({ speed, geometry: await geometry(readout) })
        if ([99.99, 100, 1000, 10000, 1000000].includes(value)) await panel.screenshot({ path: path.join(directory, `speed-${value}.png`) })
      }
      fs.writeFileSync(path.join(directory, 'geometry.json'), JSON.stringify(samples, null, 2))
      for (const sample of samples) {
        const { box, value, unit, lines, textWidth } = sample.geometry
        expect(lines, `speed ${sample.speed} must stay one numeric line`).toBe(1)
        expect(unit!.y, `speed ${sample.speed} always has its unit on the second line`).toBeGreaterThanOrEqual(value.bottom - 1)
        expect(textWidth).toBeLessThanOrEqual(box.width + 1)
        expect(value.right).toBeLessThanOrEqual(box.right + 1)
      }
      // Numeric boundaries must not move the unit or change the readout height.
      const normalSamples = samples.slice(0, 5).map(sample => sample.geometry)
      expect(Math.max(...normalSamples.map(sample => sample.box.height)) - Math.min(...normalSamples.map(sample => sample.box.height))).toBeLessThanOrEqual(1)
      const unitOffsets = normalSamples.map(sample => sample.unit!.y - sample.box.y)
      expect(Math.max(...unitOffsets) - Math.min(...unitOffsets)).toBeLessThanOrEqual(1)
      const chart = await monitor.locator('.monitor-generation-chart').boundingBox(), valueBox = samples.at(-1)!.geometry.value
      expect(chart).not.toBeNull()
      expect(valueBox.right <= chart!.x + 1 || valueBox.bottom <= chart!.y + 1).toBeTruthy()

      // Duration, first token and hardware values keep the number/unit pair atomic.
      const statistics = monitor.locator('.monitor-request-live')
      await expect(statistics.locator('dd strong')).toHaveText(['1,234,567,890', '100,000.25', '10,000.25', '67,890'])
      const atomicReadouts = [statistics.locator('dd').nth(1), statistics.locator('dd').nth(2),
        ...await monitor.locator('.monitor-hardware-metric-value').all()]
      for (const item of atomicReadouts) {
        const { value, unit, lines, textWidth, box } = await geometry(item)
        expect(lines).toBe(1)
        expect(textWidth).toBeLessThanOrEqual(box.width + 1)
        expect(value.right).toBeLessThanOrEqual(box.right + 1)
        if (unit) {
          expect(unit.y).toBeLessThan(value.bottom)
          expect(unit.bottom).toBeGreaterThan(value.y)
          expect(unit.right).toBeLessThanOrEqual(box.right + 1)
        }
      }
      for (const item of await statistics.locator('dd').all()) {
        const { textWidth, box, lines } = await geometry(item)
        expect(lines).toBe(1)
        expect(textWidth).toBeLessThanOrEqual(box.width + 1)
      }
      const hardware = monitor.locator('.monitor-hardware-summary')
      const hardwareBox = await hardware.boundingBox()
      expect(hardwareBox!.x).toBeGreaterThanOrEqual(0)
      expect(hardwareBox!.x + hardwareBox!.width).toBeLessThanOrEqual(width + 1)
      await hardware.screenshot({ path: path.join(directory, 'hardware.png') })
      expect(errors).toEqual([])
      expect(unexpected).toEqual([])
    } finally { await page.close() }
  })
}
