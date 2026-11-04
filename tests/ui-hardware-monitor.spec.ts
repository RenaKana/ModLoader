import { test, expect, _electron as electron } from '@playwright/test'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { appendHistory, numberText, seriesSegments } from '../src/resource-metrics'
import { pinNavigation } from './navigation'

test('历史采样保留零值、缺测断线、按时间窗口淘汰，不补点', () => {
  let points = appendHistory([], { time: 1000, values: { cpu: 0 } })
  points = appendHistory(points, { time: 2000, values: { cpu: null } })
  points = appendHistory(points, { time: 3000, values: { cpu: 20 } })
  points = appendHistory(points, { time: 7000, values: { cpu: 30 } })
  expect(seriesSegments(points, 'cpu').map(s => s.map(p => p.value))).toEqual([[0], [20], [30]])
  expect(appendHistory(points, { time: 7000, values: { cpu: 99 } })).toBe(points)
  expect(appendHistory(points, { time: 64000, values: { cpu: 0 } }).map(p => p.time)).toEqual([7000, 64000])
})

test('Token 速度固定两位小数，硬件数字最多保留两位真实小数', () => {
  expect([0, 7, 42.8, 1075.5].map(value => numberText(value, 2, true))).toEqual(['0.00', '7.00', '42.80', '1,075.50'])
  expect([42.8, 1075.5].map(value => numberText(value, 2))).toEqual(['42.8', '1,075.5'])
  expect(numberText(32)).toBe('32')
  expect(numberText(null, 2, true)).toBe('不可用')
  expect(numberText(32)).toBe('32')
  expect(numberText(1234567890123456, 0)).toBe('1,234,567,890,123,456')
})

test('硬件摘要：分组多指标合图、独立 PCIe 趋势、温度重试、GPU 选择及字体缩放', async () => {
  test.setTimeout(90_000)
  const root = path.resolve(__dirname, '..'), directory = fs.mkdtempSync(path.join(root, 'work', 'ui-hardware-monitor-'))
  const app = await electron.launch({ ...(process.env.MODLOADER_TEST_EXECUTABLE ? { executablePath: process.env.MODLOADER_TEST_EXECUTABLE, args: [] } : { args: [root] }), env: { ...process.env, MODLOADER_DATA_DIR: path.join(directory, 'data'), MODLOADER_SENSOR_AUTO_START: '0' } })
  const page = await app.firstWindow(), errors: string[] = [], mutations: string[] = [], externalRequests: string[] = []
  const historyWrites: string[] = [], historyFailures: number[] = [], temperatureEnables: string[] = []
  let releaseTemperature: (() => void) | undefined
  let sequence = 0, fontScale = 1, speed = 42.8, mean = 41
  let cpuPower: number | null = 75.125, gpuPower: number | null = 155.358
  let cpuTemperature: number | null = 63.25, gpuTemperature: number | null = 48
  let cpuTemperatureStatus = 'ready', gpuTemperatureSource: string | null = 'LibreHardwareMonitor · GPU Core'
  let gpuTemperatureReason: string | null = null, gpuTemperatureInitializationReason: string | null = null
  let cpuName: string | undefined = 'AMD Ryzen 9 9950X3D 16-Core Processor'
  let gpuName: string | undefined = 'NVIDIA RTX PRO 6000 Blackwell Workstation Edition'
  const instance = { id: 'test-live', model_id: 'fixture-model', profile_id: 'fixture-profile', alias: '监控测试模型', backend: 'strata', state: 'ready', active_requests: 1, capabilities: {} }
  const gib = 1024 ** 3
  page.on('pageerror', error => errors.push(error.message))
  page.on('response', response => { if (/\/api\/conversations(?:\/[^/]+)?$/.test(new URL(response.url()).pathname) && response.status() >= 400) historyFailures.push(response.status()) })
  await page.route('**/*', async route => {
    const request = route.request(), url = new URL(request.url())
    const historyWrite = ['POST', 'PATCH'].includes(request.method()) && /^\/api\/conversations(?:\/[^/]+)?$/.test(url.pathname)
    if (request.method() === 'POST' && url.pathname === '/api/hardware/temperatures/enable') {
      temperatureEnables.push(`${request.method()} ${url.pathname}`)
      await new Promise<void>(resolve => { releaseTemperature = resolve })
      cpuTemperatureStatus = 'ready'; cpuTemperature = 63.25
      gpuTemperatureInitializationReason = null; gpuTemperatureSource = 'LibreHardwareMonitor · GPU Core'; gpuTemperatureReason = null
      await route.fulfill({ json: { cpu_temperature_status: 'ready', cpu_temperature_c: cpuTemperature } })
    } else if (['http:', 'https:'].includes(url.protocol) && !['127.0.0.1', 'localhost'].includes(url.hostname)) {
      externalRequests.push(url.origin); await route.abort()
    } else if (url.pathname.startsWith('/api/') && !['GET', 'HEAD', 'OPTIONS'].includes(request.method()) && !historyWrite) {
      mutations.push(`${request.method()} ${url.pathname}`); await route.abort()
    } else {
      if (historyWrite) historyWrites.push(`${request.method()} ${url.pathname}`)
      await route.continue()
    }
  })
  await page.route('**/api/state', async route => {
    const response = await route.fetch(), state = await response.json()
    sequence++
    const gpu = { id: 'gpu-fixture', name: gpuName, memory_used: 0, memory_total: 32*gib, memory_free: 32*gib, utilization: 0, power: gpuPower,
      power_source: gpuPower === null ? null : 'NVML · nvmlDeviceGetPowerUsage', power_reason: gpuPower === null ? 'NVIDIA 驱动未提供功率读数' : null,
      temperature: gpuTemperature, temperature_source: gpuTemperatureSource, temperature_reason: gpuTemperatureReason, compute_capability: '12.0', cuda_cores: 21760,
      pcie_gen: 4, pcie_width: 16, pcie_gen_max: 5, pcie_width_max: 16, pcie_bandwidth_max_bytes_s: 63e9, memory_bus_bits: 512, memory_bandwidth_bytes_s: 1792e9, pcie_rx_bytes_s: .133e9, pcie_tx_bytes_s: 2e9 }
    const secondGpu = { ...gpu, id: 'gpu-other', name: 'GPU 2 fixture', utilization: 77, power: 221, temperature: 52, pcie_rx_bytes_s: .333e9 }
    await route.fulfill({ response, json: { ...state, models: [], profiles: [], runtimes: [], instances: [instance], downloads: [], installations: [], settings: { ...state.settings, font_scale: fontScale }, hardware: {
      sampled_at: Date.now()/1000, cpu_count: 32, cpu_percent: 11.6, cpu_per_core: Array.from({ length: 32 }, (_, index) => index),
      cpu_power_w: cpuPower, cpu_power_source: cpuPower === null ? null : 'LibreHardwareMonitor · CPU fixture · CPU Package', cpu_power_reason: cpuPower === null ? 'CPU 未提供封装功率传感器' : null,
      cpu_temperature_c: cpuTemperature, cpu_temperature_status: cpuTemperatureStatus, cpu_temperature_source: cpuTemperature === null ? null : 'LibreHardwareMonitor · CPU Package · CPU 传感器最高值',
      cpu_temperature_reason: cpuTemperature === null ? cpuTemperatureStatus === 'driver_missing' ? '未安装 PawnIO 驱动' : cpuTemperatureStatus === 'error' ? '独立 CPU 温度读取失败（fixture）' : '未读到 CPU 温度；请检查传感器工具及读取权限' : null,
      gpu_temperature_reason: gpuTemperatureInitializationReason, cpu_info: { name: cpuName, cores: 16, threads: 32, l3_bytes: 128*1024**2 }, memory_used: 38.4*gib, memory_total: 61.4*gib, memory_available: 23*gib,
      gpu, gpus: [gpu, secondGpu], disk_read_bytes_per_second: 0,
    } } })
  })
  await page.route('**/api/instances/test-live/telemetry', route => route.fulfill({ json: {
    instance_id: instance.id, backend: 'strata', state: 'ready', updated_at: Date.now()/1000,
    live_generation: { request_key: 'first', tokens_per_second: speed, average_tokens_per_second: mean, average_basis: 'sampled_decode', window_s: 3, phase: 'generating', backend_running: 1, backend_queued: 0,
      context_tokens: 8192, context_capacity: 262144, context_basis: '输入 + 已生成', prompt_tokens: 6144, generated_tokens: 2048, first_output_s: .46, request_elapsed_s: 50.46 },
    groups: [{ id: 'strata_context', title: '上下文', metrics: [{ id: 'max_context', label: '最大上下文', value: 262144, kind: 'gauge' }] },
      { id: 'api_request', title: '最近请求', request_status: 'completed', request_key: 'previous', metrics: [{ id: 'output_speed', label: '后端解码速度', value: 39.7, unit: 'token/s', kind: 'gauge', rate_basis: 'decode' }] }], unavailable: [],
  } }))
  const capture = async (name: string) => {
    await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))))
    const png = await app.evaluate(async ({ BrowserWindow }) => (await BrowserWindow.getAllWindows()[0].webContents.capturePage()).toPNG().toString('base64'))
    fs.writeFileSync(path.join(directory, name), Buffer.from(png, 'base64'))
  }
  try {
    await pinNavigation(page)
    await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
    const nav = page.getByRole('navigation', { name: '主导航' })
    await nav.getByRole('button', { name: '运行环境', exact: true }).click()
    const resources = page.locator('.hardware-resources')
    await expect(resources.locator('.resource-tile')).toHaveCount(4)
    await expect(resources.locator('[data-resource="cpu"]')).toContainText(cpuName!)
    await expect(resources.locator('[data-resource="memory"]')).toContainText('38.4/ 61.4GB')
    await expect(resources.locator('[data-resource="vram"]')).toContainText('0/ 32GB')
    await expect(resources.locator('.trend-plot, .resource-bar')).toHaveCount(0)
    await resources.locator('[data-resource="gpu"]').focus(); await page.keyboard.press('Enter')
    await expect(resources.getByRole('region', { name: 'GPU详情' })).toContainText('12.0')
    await resources.getByRole('button', { name: '收起', exact: true }).click()
    await expect(resources.locator('[data-resource="gpu"]')).toBeFocused()
    await nav.getByRole('button', { name: '运行监控', exact: true }).click()
    const monitor = page.locator('.monitor-page'), summary = monitor.getByRole('region', { name: '硬件读数' })
    const value = (id: string) => summary.locator(`[data-resource="${id}"] strong`)
    const refresh = monitor.getByRole('button', { name: '刷新到最新', exact: true })
    await expect(monitor.locator('[data-resource="current"] strong')).toHaveText('42.80')
    await expect(monitor.locator('[data-resource="average"] strong')).toHaveText('41.00')
    await expect(monitor.locator('.monitor-request-live')).toContainText('0.46s')
    await expect(summary.locator('.monitor-hardware-row')).toHaveCount(3)
    await expect(summary.locator('button')).toHaveCount(12)
    await expect(summary.locator('.trend-plot, svg[role="img"]')).toHaveCount(0)
    await expect(summary.locator('.monitor-hardware-device > strong')).toHaveText(['GPU', 'CPU', 'I/O'])
    await expect(value('gpu')).toHaveText('0')
    await expect(value('cpu_power')).toHaveText('75.13')
    await expect(value('gpu_power')).toHaveText('155.36')
    await expect(value('cpu_temperature')).toHaveText('63.25')
    await expect(value('pcie_rx')).toHaveText('0.133')
    await expect(value('pcie_tx')).toHaveText('2.000')
    await expect(monitor.locator('.monitor-last-request')).not.toBeVisible()
    await monitor.locator('.monitor-backend-details > summary').click()
    await expect(monitor.locator('.monitor-last-request')).toContainText('39.70')
    await monitor.locator('.monitor-backend-details > summary').click()
    await expect.poll(() => sequence).toBeGreaterThan(3)

    const groupMetrics = {
      gpu_group: [['gpu', 'GPU 使用率', '0', '%'], ['gpu_power', 'GPU 功率', '155.358', 'W'], ['gpu_temperature', 'GPU 温度', '48', '°C']],
      cpu_group: [['cpu', 'CPU 使用率', '11.6', '%'], ['cpu_power', 'CPU 功率', '75.125', 'W'], ['cpu_temperature', 'CPU 温度', '63.25', '°C']],
      io_group: [['disk', '磁盘读取', '0', 'MB/s'], ['pcie_rx', 'PCIe RX', '0.133', 'GB/s'], ['pcie_tx', 'PCIe TX', '2', 'GB/s']],
    } as const
    for (const [group, ids] of [
      ['gpu_group', ['gpu', 'gpu_power', 'gpu_temperature']],
      ['cpu_group', ['cpu', 'cpu_power', 'cpu_temperature']],
      ['io_group', ['disk', 'pcie_rx', 'pcie_tx']],
    ] as const) {
      const trigger = summary.locator(`[data-resource="${group}"]`)
      await trigger.focus(); await page.keyboard.press('Enter')
      await expect(trigger).toHaveAttribute('aria-expanded', 'true')
      const label = group === 'gpu_group' ? 'GPU' : group === 'cpu_group' ? 'CPU' : 'I/O'
      const detail = monitor.getByRole('region', { name: `${label}整体趋势详情`, exact: true })
      const chart = detail.locator('.hardware-trend-chart'), plots = detail.locator('.trend-plot'), svg = chart.locator('svg[role="img"]')
      await expect(detail).toBeVisible()
      await expect(chart).toHaveCount(1)
      await expect(plots).toHaveCount(1)
      await expect(svg).toHaveCount(1)
      await expect(svg).toHaveCSS('height', '230px')
      const series = chart.locator('g[data-series]')
      expect(await series.evaluateAll(elements => elements.map(element => (element as SVGGElement).dataset.series))).toEqual([...ids])
      const ariaLabel = await svg.getAttribute('aria-label')
      await expect(svg).toHaveAttribute('aria-label', /最近 60 秒趋势/)
      for (const [id, metricLabel, expected, unit] of groupMetrics[group]) {
        const legend = chart.locator(`.hardware-trend-legend-item[data-metric="${id}"]`)
        await expect(legend.locator('.hardware-trend-label')).toHaveText(metricLabel)
        await expect(legend.locator('strong')).toContainText(expected)
        const range = legend.locator('.hardware-trend-range')
        const rangeText = await range.innerText()
        expect(rangeText).toContain(unit)
        expect(rangeText).toMatch(/[\d,.]+[–—-][\d,.]+/)
        expect(ariaLabel).toContain(`${metricLabel}当前`)
        expect(ariaLabel).toContain(expected)
      }
      await expect(chart.locator('.hardware-trend-legend-item')).toHaveCount(3)
      await svg.hover({ position: { x: 10, y: 40 } })
      for (const [id, , expected] of groupMetrics[group]) await expect(chart.locator(`.hardware-trend-legend-item[data-metric="${id}"] strong`)).toContainText(expected)
      const lineStyles = await series.evaluateAll(groups => groups.map(group => {
        const line = group.querySelector('.trend-line')
        if (!line) return null
        const style = getComputedStyle(line)
        return `${style.stroke}|${style.strokeDasharray}`
      }))
      expect(lineStyles.every(Boolean)).toBe(true)
      expect(new Set(lineStyles).size).toBe(3)
      if (group === 'io_group') {
        await expect(chart.locator('.hardware-trend-legend-item[data-metric="pcie_rx"] strong')).toContainText('0.133 GB/s')
        await expect(chart.locator('.hardware-trend-legend-item[data-metric="pcie_tx"] strong')).toContainText('2 GB/s')
      }
      await detail.scrollIntoViewIfNeeded()
      await capture(`monitor-${group}.png`)
      await detail.getByRole('button', { name: '收起', exact: true }).click()
      await expect(trigger).toBeFocused()
      await page.keyboard.press('Enter'); await page.keyboard.press('Enter')
      await expect(detail).toHaveCount(0)
    }
    await summary.locator('[data-resource="gpu_group"]').click()
    await summary.locator('[data-resource="cpu_group"]').click()
    await expect(monitor.locator('.resource-detail')).toHaveCount(1)
    await expect(monitor.getByRole('region', { name: 'CPU整体趋势详情' })).toBeVisible()
    await expect(summary.locator('[data-resource="gpu_group"]')).toHaveAttribute('aria-expanded', 'false')
    await summary.locator('[data-resource="cpu"]').click()
    await expect(monitor.locator('.resource-detail .monitor-core')).toHaveCount(32)
    await monitor.getByRole('button', { name: '收起', exact: true }).click()

    for (const id of ['gpu', 'gpu_power', 'gpu_temperature', 'cpu', 'cpu_power', 'cpu_temperature', 'disk']) {
      await summary.locator(`[data-resource="${id}"]`).click()
      const detail = monitor.locator('.resource-detail')
      await expect(detail.locator('.trend-plot')).toHaveCount(id === 'cpu' ? 33 : 1)
      await expect(detail.locator(`[data-series="${id}"]`)).toHaveCount(1)
      if (id === 'cpu') await expect(detail.locator('.monitor-core')).toHaveCount(32)
      if (id === 'cpu_power') await expect(detail).toContainText('CPU Package')
      await detail.getByRole('button', { name: '收起', exact: true }).click()
      await expect(summary.locator(`[data-resource="${id}"]`)).toBeFocused()
    }
    for (const [id, label, reading] of [['pcie_rx', 'PCIe RX', '0.133 GB/s'], ['pcie_tx', 'PCIe TX', '2 GB/s']]) {
      await summary.locator(`[data-resource="${id}"]`).click()
      const pcieDetail = monitor.getByRole('region', { name: `${label}详情`, exact: true })
      await expect(pcieDetail.locator('.trend-plot svg')).toHaveCount(1)
      await expect(pcieDetail.locator('[data-series]')).toHaveCount(1)
      await expect(pcieDetail.locator('[data-series]')).toHaveAttribute('data-series', id)
      await expect(pcieDetail.locator('.trend-legend-item')).toHaveCount(0)
      await pcieDetail.locator('svg').hover({ position: { x: 10, y: 40 } })
      await expect(pcieDetail.locator('.trend-readout')).toContainText(reading)
      await expect(pcieDetail.locator('.trend-readout')).not.toContainText(id === 'pcie_rx' ? '2 GB/s' : '0.133 GB/s')
      await capture(`monitor-${id}.png`)
      await pcieDetail.getByRole('button', { name: '收起', exact: true }).click()
      await expect(summary.locator(`[data-resource="${id}"]`)).toBeFocused()
    }
    await summary.locator('[data-resource="gpu_group"]').click()
    await monitor.getByRole('combobox', { name: '监控 GPU', exact: true }).selectOption('gpu-other')
    await expect(monitor.locator('.resource-detail')).toHaveCount(0)
    await expect(value('gpu')).toHaveText('77'); await expect(value('pcie_rx')).toHaveText('0.333')
    await summary.locator('[data-resource="gpu_group"]').click()
    const otherGpuDetail = monitor.getByRole('region', { name: 'GPU整体趋势详情', exact: true })
    await expect(otherGpuDetail.locator('.resource-detail-heading')).toContainText('GPU 2 fixture')
    await expect(otherGpuDetail.locator('.hardware-trend-chart svg')).toHaveAttribute('aria-label', /GPU 使用率当前77 %/)
    await monitor.getByRole('combobox', { name: '监控 GPU', exact: true }).selectOption('gpu-fixture')
    await expect(monitor.locator('.resource-detail')).toHaveCount(0)
    await expect(value('gpu')).toHaveText('0')
    await summary.locator('[data-resource="gpu_group"]').click()
    const originalGpuDetail = monitor.getByRole('region', { name: 'GPU整体趋势详情', exact: true })
    await expect(originalGpuDetail.locator('.hardware-trend-legend-item[data-metric="gpu"] strong')).toContainText('0 %')
    await expect(originalGpuDetail.locator('.hardware-trend-chart svg')).toHaveAttribute('aria-label', /GPU 使用率当前0 %/)
    await originalGpuDetail.getByRole('button', { name: '收起', exact: true }).click()

    gpuTemperatureSource = null; gpuTemperatureReason = '未能唯一匹配 GPU 传感器，当前保留 NVIDIA GPU 温度。'
    await refresh.click(); await summary.locator('[data-resource="gpu_power"]').click()
    await expect(monitor.locator('.monitor-gpu-temperature-reason')).toHaveText(gpuTemperatureReason)
    await expect(monitor.getByRole('button', { name: '重试温度读取', exact: true })).toHaveCount(0)
    await monitor.getByRole('button', { name: '收起', exact: true }).click()
    cpuTemperature = null; cpuTemperatureStatus = 'starting'
    await refresh.click(); await expect(value('cpu_temperature')).toHaveText('不可用')
    await summary.locator('[data-resource="cpu_temperature"]').click()
    await expect(monitor.locator('.resource-detail').getByRole('status')).toContainText('正在连接或授权温度读取')
    await expect(monitor.getByRole('button', { name: '重试温度读取', exact: true })).toHaveCount(0)
    await monitor.getByRole('button', { name: '收起', exact: true }).click()
    cpuTemperatureStatus = 'error'; await refresh.click()
    await summary.locator('[data-resource="cpu_temperature"]').click()
    await expect(monitor.locator('.resource-detail').getByRole('alert')).toHaveText('独立 CPU 温度读取失败（fixture）')
    await monitor.getByRole('button', { name: '重试温度读取', exact: true }).click()
    await expect(monitor.locator('.resource-detail').getByRole('status')).toContainText('正在连接或授权温度读取')
    await expect.poll(() => Boolean(releaseTemperature)).toBe(true); releaseTemperature!(); releaseTemperature = undefined
    await expect(monitor.locator('.monitor-temperature-control')).toHaveCount(0)
    await expect(value('cpu_temperature')).toHaveText('63.25')
    await monitor.getByRole('button', { name: '收起', exact: true }).click()
    gpuTemperatureSource = null; gpuTemperatureInitializationReason = 'LHM GPU helper initialization failed'
    await refresh.click(); await summary.locator('[data-resource="gpu_temperature"]').click()
    await expect(monitor.locator('.resource-detail-heading')).toContainText(gpuTemperatureInitializationReason)
    await monitor.getByRole('button', { name: '重试温度读取', exact: true }).click()
    await expect(monitor.locator('.resource-detail').getByRole('status')).toContainText('正在连接温度读取')
    await expect.poll(() => Boolean(releaseTemperature)).toBe(true); releaseTemperature!(); releaseTemperature = undefined
    await expect(monitor.locator('.monitor-temperature-control')).toHaveCount(0)
    await monitor.getByRole('button', { name: '收起', exact: true }).click()
    expect(temperatureEnables).toEqual(['POST /api/hardware/temperatures/enable', 'POST /api/hardware/temperatures/enable'])

    cpuTemperatureStatus = 'driver_missing'; cpuTemperature = null; cpuPower = null; gpuPower = 0
    await refresh.click(); await expect(value('cpu_power')).toHaveText('不可用'); await expect(value('gpu_power')).toHaveText('0')
    await summary.locator('[data-resource="cpu_power"]').click()
    await expect(monitor.locator('.resource-detail')).toContainText('未安装 PawnIO 驱动')
    await expect(monitor.getByRole('button', { name: '官方 PawnIO 驱动', exact: true })).toBeVisible()
    await monitor.getByRole('button', { name: '收起', exact: true }).click()
    cpuName = undefined; gpuName = undefined; gpuTemperature = null
    await refresh.click()
    await expect(summary.locator('.monitor-hardware-device > span')).toHaveText(['GPU 型号不可用', 'CPU 型号不可用', '磁盘与 PCIe 传输'])
    await expect(value('gpu_temperature')).toHaveText('不可用')
    await expect(summary.locator('[data-resource="gpu_temperature"] small')).toHaveCount(0)
    await summary.locator('[data-resource="gpu_group"]').click()
    const unavailableGpuDetail = monitor.getByRole('region', { name: 'GPU整体趋势详情', exact: true })
    await expect(unavailableGpuDetail.locator('.hardware-trend-chart.trend-plot')).toHaveCount(1)
    await expect(unavailableGpuDetail.locator('.hardware-trend-chart svg')).toHaveAttribute('aria-label', /GPU 温度当前不可用/)
    await expect(unavailableGpuDetail.locator('.hardware-trend-legend-item[data-metric="gpu_temperature"] strong')).toHaveText('不可用')
    await monitor.getByRole('button', { name: '收起', exact: true }).click()
    cpuName = 'AMD Ryzen 9 9950X3D 16-Core Processor'; gpuName = 'NVIDIA RTX PRO 6000 Blackwell Workstation Edition'
    cpuTemperatureStatus = 'ready'; cpuTemperature = 63.25; gpuTemperature = 48; cpuPower = 75.125; gpuPower = 155.358
    fontScale = 1.3
    await refresh.click()
    await expect.poll(() => page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue('--font-scale'))).toBe('1.3')
    for (const size of [{ width: 1600, height: 1000, zoom: 1 }, { width: 1000, height: 1000, zoom: 1 }, { width: 940, height: 1100, zoom: 1.2 }]) {
      await app.evaluate(({ BrowserWindow }, size) => { const window = BrowserWindow.getAllWindows()[0]; window.setSize(size.width, size.height); window.webContents.setZoomFactor(size.zoom) }, size)
      await summary.scrollIntoViewIfNeeded()
      await expect.poll(() => summary.evaluate(element => element.scrollWidth <= element.clientWidth + 1)).toBe(true)
      const layout = await summary.locator('button').evaluateAll(elements => elements.map(element => { const rect = element.getBoundingClientRect(); return { left: rect.left, right: rect.right, overflow: element.scrollWidth > element.clientWidth + 1, width: innerWidth } }))
      expect(layout.every(metric => metric.left >= 0 && metric.right <= metric.width + 1 && !metric.overflow)).toBe(true)
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true)
      await capture(`monitor-summary-${size.width}.png`)
      await summary.locator('[data-resource="io_group"]').click()
      const groupDetail = monitor.getByRole('region', { name: 'I/O整体趋势详情' })
      await groupDetail.scrollIntoViewIfNeeded()
      await expect(groupDetail.locator('.hardware-trend-chart.trend-plot')).toHaveCount(1)
      await expect(groupDetail.locator('.hardware-trend-chart svg[role="img"]')).toHaveCount(1)
      await expect(groupDetail.locator('.hardware-trend-chart g[data-series]')).toHaveCount(3)
      expect(await groupDetail.evaluate(element => element.scrollWidth <= element.clientWidth + 1)).toBe(true)
      expect(await groupDetail.locator('.hardware-trend-chart').evaluate(element => element.scrollWidth <= element.clientWidth + 1)).toBe(true)
      await capture(`monitor-io-group-${size.width}.png`)
      await groupDetail.getByRole('button', { name: '收起', exact: true }).click()
    }
    await summary.locator('[data-resource="pcie_rx"]').click()
    const pcieDetail = monitor.getByRole('region', { name: 'PCIe RX详情', exact: true })
    await expect(pcieDetail.locator('.trend-plot svg')).toHaveCount(1)
    expect(await pcieDetail.evaluate(element => element.scrollWidth <= element.clientWidth + 1)).toBe(true)
    await capture('monitor-pcie-narrow.png')
    await pcieDetail.getByRole('button', { name: '收起', exact: true }).click()
    fontScale = 1; await refresh.click()
    await app.evaluate(({ BrowserWindow }) => { const window = BrowserWindow.getAllWindows()[0]; window.setSize(1440, 1100); window.webContents.setZoomFactor(1) })
    await monitor.locator('.monitor-generation-selection > .monitor-generation-grid').scrollIntoViewIfNeeded()
    const current = monitor.locator('.monitor-speed-current'), average = monitor.locator('.monitor-speed-mean')
    const alignment = async () => Promise.all([current, average].map(readout => readout.evaluate(element => { const number = element.querySelector('strong')!.getBoundingClientRect(), unit = element.querySelector('small')!.getBoundingClientRect(); return { gap: unit.left - number.right, expectedGap: parseFloat(getComputedStyle(element).gap) } })))
    const before = await alignment(); speed = 1075.5; mean = 7
    await expect(current.locator('strong')).toHaveText('1,075.50'); await expect(average.locator('strong')).toHaveText('7.00')
    expect([...before, ...await alignment()].every(readout => Math.abs(readout.gap - readout.expectedGap) <= 1)).toBe(true)
    await capture('monitor-summary-final.png')
    expect(errors).toEqual([]); expect(mutations).toEqual([]); expect(externalRequests).toEqual([]); expect(historyFailures).toEqual([])
    console.log(JSON.stringify({ directory, errors, mutations, externalRequests, historyWrites, historyFailures, temperatureEnables }))
  } finally {
    releaseTemperature?.()
    await page.unrouteAll({ behavior: 'wait' })
    await app.close()
  }
})


