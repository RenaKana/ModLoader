import { test, expect, _electron as electron } from '@playwright/test'
import * as path from 'node:path'
import * as fs from 'node:fs'
import { pinNavigation } from './navigation'
import { disableWebSearch } from './chat-controls'
import type { Job } from '../src/types'

type Point = { x: number; y: number }
type Sample = { time: number; open: boolean; icons: Point[]; panelTransform: string; panelOpacity: string; opacity: number; motions: { native: boolean; duration: number }[] }
type Motion = { frames: Sample[]; changes: Sample[] }
const distance = (a: Point, b: Point) => Math.hypot(a.x - b.x, a.y - b.y)

test('任务导航按模型下载与环境安装分别计数和跳转', async () => {
  test.setTimeout(60_000)
  const root = path.resolve(__dirname, '..')
  const directory = fs.mkdtempSync(path.join(root, 'work', 'ui-job-navigation-'))
  const application = await electron.launch({
    ...(process.env.MODLOADER_TEST_EXECUTABLE ? { executablePath: process.env.MODLOADER_TEST_EXECUTABLE, args: [] } : { args: [root] }),
    env: { ...process.env, MODLOADER_DATA_DIR: directory }, timeout: 60_000,
  })
  try {
    const page = await application.firstWindow()
    const errors: string[] = []
    page.on('pageerror', error => errors.push(error.message))
    let downloads: Job[] = []
    const installations: Job[] = [{ id: 'environment', backend: 'llama_cpp', state: 'running', message: '正在下载环境', progress: .25 }]
    await page.route('**/api/state', async route => {
      const response = await route.fetch(), state = await response.json()
      await route.fulfill({ response, json: { ...state, downloads, installations } })
    })
    await pinNavigation(page)
    await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 30_000 })
    const nav = page.getByRole('navigation', { name: '主导航' })
    const discover = nav.getByRole('button', { name: '发现模型', exact: true })
    const runtimes = nav.getByRole('button', { name: '运行环境', exact: true })
    const summary = page.locator('.summary-metrics')
    await discover.click()
    await expect(page.locator('.download-panel .heading-count')).toHaveText('0')
    await expect(discover.locator('.nav-count')).toHaveCount(0)
    await expect(runtimes.locator('.nav-count')).toHaveText('1')
    await expect(page.getByText('暂无下载任务', { exact: true })).toBeVisible()
    await summary.getByRole('button', { name: '1 项安装', exact: true }).click()
    await expect(runtimes).toHaveAttribute('aria-current', 'page')
    await expect(page.locator('.installation-panel').getByRole('tab', { name: '未完成 1', exact: true })).toBeVisible()

    downloads = [{ id: 'model', name: '模型下载', state: 'downloading', total_bytes: 1024, completed_bytes: 256 },
      { id: 'queued', name: '排队模型', state: 'queued' },
      { id: 'registering', name: '登记中模型', state: 'running', phase: 'registering', message: '登记模型', total_bytes: 1024, completed_bytes: 1024 },
      ...['completed', 'failed', 'cancelled'].map(state => ({ id: state, name: state, state }))]
    await expect(discover.locator('.nav-count')).toHaveText('3')
    await expect(runtimes.locator('.nav-count')).toHaveText('1')
    await summary.getByRole('button', { name: '3 项下载', exact: true }).click()
    await expect(discover).toHaveAttribute('aria-current', 'page')
    await expect(page.locator('.download-panel .heading-count')).toHaveText('3')
    await expect(page.locator('.download-list')).toContainText('模型下载')
    const downloading = page.locator('.download-list .job-row').filter({ hasText: '模型下载' })
    await expect(downloading.getByText('256 B / 1 KB', { exact: true })).toBeVisible()
    await expect(downloading.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '25')
    const registering = page.locator('.download-list .job-row').filter({ hasText: '登记中模型' })
    await expect(registering.getByText('登记中', { exact: true })).toBeVisible()
    await expect(registering.getByText('登记模型', { exact: true })).toBeVisible()
    await expect(registering.getByRole('progressbar')).toHaveCount(0)
    await summary.getByRole('button', { name: '1 项安装', exact: true }).click()
    await expect(runtimes).toHaveAttribute('aria-current', 'page')
    await page.evaluate(() => document.fonts.ready)
    const screenshot = await application.evaluate(async ({ BrowserWindow }) =>
      (await BrowserWindow.getAllWindows()[0].webContents.capturePage()).toPNG().toString('base64'))
    fs.writeFileSync(path.join(directory, 'mixed-tasks.png'), Buffer.from(screenshot, 'base64'))

    downloads[0].state = 'completed'
    downloads[1].state = 'cancelled'
    downloads[2].state = 'completed'
    installations[0].state = 'completed'
    await expect(discover.locator('.nav-count')).toHaveCount(0)
    await expect(runtimes.locator('.nav-count')).toHaveCount(0)
    await expect(summary.getByRole('button')).toHaveCount(0)
    await expect(page.locator('.installation-panel').getByRole('tab', { name: '未完成 0', exact: true })).toBeVisible()
    await discover.click()
    await expect(page.locator('.download-panel .heading-count')).toHaveText('0')
    expect(errors).toEqual([])
    console.log(JSON.stringify({ screenshots: directory }))
  } finally { await application.close() }
})

for (const kind of ['download', 'install']) {
  test(`下载任务控制 ${kind}：暂停、继续和停止`, async () => {
    test.setTimeout(60_000)
    const root = path.resolve(__dirname, '..')
    const directory = fs.mkdtempSync(path.join(root, 'work', `ui-job-${kind}-`))
    const application = await electron.launch({
      ...(process.env.MODLOADER_TEST_EXECUTABLE ? { executablePath: process.env.MODLOADER_TEST_EXECUTABLE, args: [] } : { args: [root] }),
      env: { ...process.env, MODLOADER_DATA_DIR: directory }, timeout: 60_000,
    })
    try {
      const page = await application.firstWindow()
      const errors: string[] = [], actions: string[] = []
      page.on('pageerror', error => errors.push(error.message))
      const job: Job = { id: 'controlled', kind: kind === 'download' ? 'download' : 'install', backend: kind === 'install' ? 'llama_cpp' : undefined, phase: 'download', total_files: kind === 'install' ? 1 : undefined, name: kind === 'install' ? 'llama.cpp 环境包' : '模型文件.gguf', state: 'running', can_pause: true, message: kind === 'install' ? '正在下载环境包' : undefined, total_bytes: 1024, completed_bytes: 256 }
      await page.route('**/api/**', async route => {
        const request = route.request(), url = new URL(request.url())
        if (url.pathname === '/api/state') {
          const response = await route.fetch(), state = await response.json()
          return route.fulfill({ response, json: { ...state, settings: { ...state.settings, font_scale: 1.3 }, downloads: kind === 'download' ? [job] : [], installations: kind === 'install' ? [job] : [] } })
        }
        if (url.pathname === '/api/installations/history') {
          const items = kind === 'install' && ['completed', 'complete', 'failed', 'error', 'cancelled', 'canceled', 'interrupted', 'stopped'].includes(job.state || '') ? [job] : []
          const currentPage = Number(url.searchParams.get('page') || 1)
          return route.fulfill({ json: { items, total: items.length, page: currentPage, page_size: 10 } })
        }
        if (request.method() !== 'GET') {
          actions.push(url.pathname)
          if (url.pathname === '/api/jobs/controlled/pause') job.state = 'pausing'
          else if (url.pathname === '/api/jobs/controlled/resume') job.state = 'running'
          else if (url.pathname === '/api/jobs/controlled/cancel') job.state = 'cancelling'
          else return route.abort()
          return route.fulfill({ json: job })
        }
        return route.continue()
      })
      await pinNavigation(page)
      await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 30_000 })
      if (kind === 'download') await application.evaluate(({ BrowserWindow }) => {
        const window = BrowserWindow.getAllWindows()[0]
        window.setSize(1050, 780)
        window.webContents.setZoomFactor(1.2)
      })
      const nav = page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: kind === 'install' ? '运行环境' : '发现模型', exact: true })
      await nav.click()
      const panel = page.locator(kind === 'install' ? '.installation-panel' : '.download-panel')
      const row = panel.locator('.job-row')
      await row.getByRole('button', { name: '暂停', exact: true }).click()
      await expect(row.getByText('暂停中', { exact: true })).toBeVisible()
      await expect(row.getByRole('button', { name: '暂停', exact: true })).toBeDisabled()
      await expect(row.getByRole('button', { name: '停止', exact: true })).toBeEnabled()
      job.state = 'paused'
      await expect(row.getByRole('button', { name: '继续', exact: true })).toBeVisible()
      await expect(row.getByText('已暂停', { exact: true })).toBeVisible()
      if (kind === 'download') await expect(panel.locator('.heading-count')).toHaveText('0')
      else await expect(panel.getByRole('tab', { name: '未完成 1', exact: true })).toBeVisible()
      await expect(nav.locator('.nav-count')).toHaveCount(0)
      expect(await row.locator('.progress-track').evaluate(element => {
        const fill = element.querySelector('span')!
        return fill.getBoundingClientRect().width / element.clientWidth
      })).toBeCloseTo(.25, 2)
      await expect(row.getByRole('button', { name: '停止', exact: true })).toBeEnabled()
      await page.evaluate(() => document.fonts.ready)
      expect(await row.evaluate(element => element.scrollWidth > element.clientWidth + 1)).toBe(false)
      const png = await application.evaluate(async ({ BrowserWindow }) => (await BrowserWindow.getAllWindows()[0].webContents.capturePage()).toPNG().toString('base64'))
      fs.writeFileSync(path.join(directory, 'paused.png'), Buffer.from(png, 'base64'))
      await row.getByRole('button', { name: '继续', exact: true }).click()
      await expect(row.getByRole('button', { name: '暂停', exact: true })).toBeEnabled()
      await expect(nav.locator('.nav-count')).toHaveText('1')
      if (kind === 'install') {
        job.can_pause = false
        job.phase = 'extract'
        job.message = '正在解压环境包'
        await expect(row.getByText('正在解压环境包', { exact: true })).toBeVisible()
        await expect(row.getByRole('button', { name: '暂停', exact: true })).toHaveCount(0)
        await expect(row.getByRole('progressbar')).toHaveCount(0)
      } else {
        await row.getByRole('button', { name: '暂停', exact: true }).click()
        await expect(row.getByText('暂停中', { exact: true })).toBeVisible()
        job.state = 'paused'
        await expect(row.getByRole('button', { name: '继续', exact: true })).toBeVisible()
      }
      await row.getByRole('button', { name: '停止', exact: true }).click()
      await expect(row.getByRole('button', { name: '停止中', exact: true })).toBeDisabled()
      job.state = 'cancelled'
      await expect(nav.locator('.nav-count')).toHaveCount(0)
      if (kind === 'download') await panel.getByRole('button', { name: '全部任务', exact: true }).click()
      await expect(row.getByText('已停止', { exact: true })).toBeVisible()
      await expect(row.getByRole('button', { name: kind === 'install' ? '重试安装' : '重试', exact: true })).toBeVisible()
      expect(actions).toEqual(['/api/jobs/controlled/pause', '/api/jobs/controlled/resume', ...(kind === 'download' ? ['/api/jobs/controlled/pause'] : []), '/api/jobs/controlled/cancel'])
      expect(errors).toEqual([])
      console.log(JSON.stringify({ screenshots: directory }))
    } finally { await application.close() }
  })
}

test('常驻六图标导航：悬停与键盘展开、连续位移、钉住留位及对话正文高度', async () => {
  test.setTimeout(120_000)
  const root = path.resolve(__dirname, '..')
  const directory = fs.mkdtempSync(path.join(root, 'work', 'ui-floating-header-'))
  const application = await electron.launch({
    ...(process.env.MODLOADER_TEST_EXECUTABLE ? { executablePath: process.env.MODLOADER_TEST_EXECUTABLE, args: [] } : { args: [root] }),
    env: { ...process.env, MODLOADER_DATA_DIR: path.join(directory, 'data') }, timeout: 60_000,
  })
  const page = await application.firstWindow()
  const errors: string[] = [], external: string[] = [], writes: string[] = []
  let fontScale = 1, stateReads = 0
  const labels = ['模型库', '发现模型', '运行环境', '运行监控', '测试对话', '设置']
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/*', async route => {
    const request = route.request(), url = new URL(request.url())
    if (['http:', 'https:'].includes(url.protocol) && !['127.0.0.1', 'localhost'].includes(url.hostname)) { external.push(url.origin); return route.abort() }
    if (url.pathname === '/api/chat') return route.fulfill({ contentType: 'text/event-stream', body: `data: ${JSON.stringify({ choices: [{ delta: { content: Array.from({ length: 35 }, (_, i) => `${i + 1}. **可见的模型输出**：这段内容用于检查阅读区域与内部滚动。`).join('\n\n') } }] })}\n\ndata: [DONE]\n\n` })
    if (url.pathname.endsWith('/telemetry')) return route.fulfill({ json: { groups: [], unavailable: [], updated_at: Date.now() / 1000 } })
    // Conversation writes use this test's isolated data directory.
    if (url.pathname.startsWith('/api/conversations')) return route.continue()
    if (url.pathname.startsWith('/api/') && request.method() !== 'GET') { writes.push(url.pathname); return route.abort() }
    return route.continue()
  })
  await page.route('**/api/state', async route => {
    stateReads++
    const response = await route.fetch(), state = await response.json()
    return route.fulfill({ response, json: { ...state, settings: { ...state.settings, font_scale: fontScale },
      models: [{ id: 'layout-model', name: '布局检查模型', path: 'fixture.gguf', format: 'gguf', size_bytes: 0, files: [], modalities: ['text'], metadata: {}, source: {}, managed: false, warnings: [] }],
      profiles: [{ id: 'layout-profile', model_id: 'layout-model', name: '布局配置', backend: 'llama_cpp', runtime_id: '', alias: 'layout-fixture', parameters: {}, generation: {}, extra_args: [], revision: 1 }],
      instances: [{ id: 'layout-instance', profile_id: 'layout-profile', model_id: 'layout-model', profile_revision: 1, alias: 'layout-fixture', backend: 'llama_cpp', runtime_id: '', state: 'ready', port: 0, active_requests: 0, started_at: '', capabilities: { input: ['text'] } }], runtimes: [], downloads: [], installations: [],
    } })
  })
  const header = page.locator('.app-header'), panel = page.locator('#header-panel')
  const wordmark = header.locator('.wordmark')
  const nav = page.getByRole('navigation', { name: '主导航', exact: true })
  const pin = page.getByRole('button', { name: '钉住顶部栏', exact: true })
  async function open(edge: 'left' | 'right' = 'left') {
    const box = await header.boundingBox()
    if (!box) throw new Error('顶部导航尚未显示')
    // The bar's own edges activate hover; outside side margins belong to the page.
    await page.mouse.move(edge === 'left' ? box.x + 2 : box.x + box.width - 2, box.y + 10)
    await expect(header).toHaveClass(/is-open/)
    await expect(panel).toBeVisible()
  }
  async function leave() {
    await page.mouse.move(5, 500)
    await page.locator('body').click({ position: { x: 5, y: 500 } })
    await expect(header).not.toHaveClass(/is-open/)
    await expect(panel).toBeVisible()
  }
  async function settle() {
    await page.evaluate(async () => {
      await Promise.all(Array.from(document.querySelectorAll('.navigation button, .nav-label')).flatMap(element => element.getAnimations()).map(animation => animation.finished.catch(() => undefined)))
      await new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve())))
    })
  }
  // Screen coordinates prove travel; observing the class change catches reversal rebasing.
  async function beginMotion() {
    await page.evaluate(() => {
      const header = document.querySelector('.app-header')!, panel = document.querySelector('#header-panel')!
      const read = (): Sample => ({
        time: performance.now(), open: header.classList.contains('is-open'),
        icons: Array.from(header.querySelectorAll('.nav-icon'), icon => { const rect = icon.getBoundingClientRect(); return { x: rect.x, y: rect.y } }),
        panelTransform: getComputedStyle(panel).transform, panelOpacity: getComputedStyle(panel).opacity,
        opacity: Number(getComputedStyle(header.querySelector('.nav-label')!).opacity),
        motions: Array.from(header.querySelectorAll('.navigation button')).flatMap(button => button.getAnimations().flatMap(animation => {
          const effect = animation.effect as KeyframeEffect
          return effect.getKeyframes().some(frame => frame.transform !== undefined)
            ? [{ duration: Number(effect.getTiming().duration), native: !(animation instanceof CSSAnimation) && !(animation instanceof CSSTransition) }] : []
        })),
      })
      const capture = { frames: [read()], changes: [] as Sample[], frame: 0, read, observer: new MutationObserver(() => capture.changes.push(read())) }
      const sample = () => { capture.frames.push(read()); capture.frame = requestAnimationFrame(sample) }
      capture.observer.observe(header, { attributes: true, attributeFilter: ['class'] })
      capture.frame = requestAnimationFrame(sample)
      ;(window as any).headerMotionCapture = capture
    })
  }
  async function endMotion(): Promise<Motion> {
    await settle()
    return page.evaluate(() => {
      const capture = (window as any).headerMotionCapture
      cancelAnimationFrame(capture.frame); capture.observer.disconnect()
      capture.frames.push(capture.read()); delete (window as any).headerMotionCapture
      return { frames: capture.frames, changes: capture.changes }
    })
  }
  function expectTravel(motion: Motion) {
    expect(motion.frames.length).toBeGreaterThan(3)
    expect(motion.frames[0].icons).toHaveLength(6)
    expect(motion.frames.some(frame => frame.motions.length === 6)).toBe(true)
    for (const sample of [...motion.frames, ...motion.changes]) {
      expect(sample.panelTransform).toBe('none'); expect(sample.panelOpacity).toBe('1')
      for (const animation of sample.motions) { expect(animation.native).toBe(true); expect(animation.duration).toBe(180) }
    }
    for (let index = 0; index < 6; index++) {
      const first = motion.frames[0].icons[index], last = motion.frames.at(-1)!.icons[index]
      expect(distance(first, last), `导航图标 ${index + 1} 的实际位移`).toBeGreaterThan(2)
      expect(motion.frames.some(frame => distance(frame.icons[index], first) > 1 && distance(frame.icons[index], last) > 1), `导航图标 ${index + 1} 的中间位置`).toBe(true)
    }
  }
  async function capture(name: string, settled = true) {
    if (settled) await settle()
    const png = await application.evaluate(async ({ BrowserWindow }) => (await BrowserWindow.getAllWindows()[0].webContents.capturePage()).toPNG().toString('base64'))
    fs.writeFileSync(path.join(directory, name), Buffer.from(png, 'base64'))
  }
  const frame = () => page.evaluate(() => {
    const messages = document.querySelector('.chat-messages')!.getBoundingClientRect()
    const composer = document.querySelector('.chat-composer-area')!.getBoundingClientRect()
    const chat = document.querySelector('.chat-panel')!.getBoundingClientRect()
    const header = document.querySelector('.app-header')!.getBoundingClientRect()
    return { chatTop: chat.top, messages: messages.height, composerBottom: composer.bottom, viewport: innerHeight, documentHeight: document.documentElement.scrollHeight, overflow: document.documentElement.scrollWidth > innerWidth + 1, space: document.querySelector('.topbar-space')!.getBoundingClientRect().height, headerHeight: header.height, headerWidth: header.width, viewportWidth: innerWidth }
  })
  try {
    await page.emulateMedia({ reducedMotion: 'no-preference' })
    await expect(header).toBeVisible({ timeout: 60_000 })
    await open()
    await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
    await page.evaluate(() => document.fonts.ready)
    for (const size of [{ w: 1920, h: 1080, zoom: 1, font: 1 }, { w: 1440, h: 960, zoom: 1.2, font: 1.3 }, { w: 1050, h: 780, zoom: 1.2, font: 1.3 }]) {
      fontScale = size.font
      await application.evaluate(({ BrowserWindow }, size) => { const window = BrowserWindow.getAllWindows()[0]; window.setSize(size.w, size.h); window.webContents.setZoomFactor(size.zoom) }, size)
      await open()
      const before = stateReads
      await page.getByRole('button', { name: '刷新状态', exact: true }).click()
      await expect.poll(() => stateReads).toBeGreaterThan(before)
      await expect.poll(() => page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue('--font-scale').trim())).toBe(String(size.font))
      await nav.getByRole('button', { name: '测试对话', exact: true }).click()
      await leave(); await settle()
      await expect(panel).toHaveCSS('transform', 'none')
      await expect(panel).toHaveCSS('opacity', '1')
      expect(await panel.evaluate(element => element.hasAttribute('inert') || element.getAttribute('aria-hidden') === 'true')).toBe(false)
      await expect(nav.getByRole('button')).toHaveCount(6)
      for (const label of labels) await expect(nav.getByRole('button', { name: label, exact: true })).toBeVisible()
      for (const label of await nav.locator('.nav-label').all()) await expect(label).toHaveCSS('opacity', '0')
      await expect(header.locator('.workspace-summary')).toBeHidden()
      await expect(page.getByRole('button', { name: '刷新状态', exact: true })).toBeHidden()
      await expect(page.getByRole('button', { name: '关于 ModLoader', exact: true })).toBeHidden()
      await expect(pin).toBeVisible()
      const collapsed = await frame()
      expect(collapsed.space).toBeGreaterThanOrEqual(50); expect(collapsed.space).toBeLessThanOrEqual(64)
      expect(collapsed.headerWidth).toBeCloseTo(Math.min(1456, collapsed.viewportWidth - 24), 0)
      expect(collapsed.chatTop).toBeGreaterThanOrEqual(collapsed.space)
      expect(collapsed.chatTop).toBeLessThan(collapsed.space + 24)
      expect(collapsed.composerBottom).toBeLessThanOrEqual(collapsed.viewport + 1)
      expect(collapsed.documentHeight).toBeLessThanOrEqual(collapsed.viewport + 1)
      expect(collapsed.messages / collapsed.viewport).toBeGreaterThan(.5)
      expect(collapsed.overflow).toBe(false)
      await capture(`header-collapsed-${size.w}.png`)

      await beginMotion(); await open()
      const expandedMotion = await endMotion()
      expectTravel(expandedMotion)
      expect(expandedMotion.frames.some(sample => sample.opacity > 0 && sample.opacity < 1)).toBe(true)
      for (const label of await nav.locator('.nav-label').all()) await expect(label).toHaveCSS('opacity', '1')
      await expect(header.locator('.workspace-summary')).toBeVisible()
      expect((await frame()).chatTop).toBe(collapsed.chatTop)
      expect((await frame()).messages).toBe(collapsed.messages)
      expect((await frame()).space).toBe(collapsed.space)
      await capture(`header-expanded-${size.w}.png`)
      await beginMotion(); await leave()
      const collapsedMotion = await endMotion()
      expectTravel(collapsedMotion)

      if (size.w === 1920) {
        await open()
        await page.evaluate(() => {
          for (const animation of document.querySelector('.app-header')!.getAnimations({ subtree: true })) {
            animation.pause()
            animation.currentTime = 90
          }
        })
        await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))))
        await capture('header-motion-midpoint-1920.png', false)
        await page.evaluate(() => { for (const animation of document.querySelector('.app-header')!.getAnimations({ subtree: true })) animation.finish() })
        await leave(); await settle()
      }
      // Reverse real pointer input while the opening WAAPI timeline is still running.
      await beginMotion(); await open()
      await page.evaluate(() => new Promise<void>((resolve, reject) => {
        const deadline = performance.now() + 1_000
        const wait = () => {
          const animation = document.querySelector('.navigation button')!.getAnimations().find(animation => (animation.effect as KeyframeEffect).getKeyframes().some(frame => frame.transform !== undefined))
          if (animation && Number(animation.currentTime) >= 80 && Number(animation.currentTime) < 120) { resolve(); return }
          if (performance.now() > deadline) { reject(new Error('未捕获运行中的导航位移动画')); return }
          requestAnimationFrame(wait)
        }
        wait()
      }))
      await page.mouse.move(5, 500)
      await expect(header).not.toHaveClass(/is-open/)
      const reversal = await endMotion(), reversed = reversal.changes.find(sample => !sample.open)
      expect(reversed).toBeDefined()
      const preceding = reversal.frames.filter(sample => sample.open && sample.time <= reversed!.time).at(-1)!
      for (let index = 0; index < 6; index++) {
        const compact = expandedMotion.frames[0].icons[index], expanded = expandedMotion.frames.at(-1)!.icons[index], point = reversed!.icons[index]
        expect(distance(point, compact)).toBeGreaterThan(1); expect(distance(point, expanded)).toBeGreaterThan(1)
        expect(distance(point, preceding.icons[index]), `导航图标 ${index + 1} 反向时不应跳到端点`).toBeLessThan(distance(compact, expanded) * .3 + 2)
        expect(distance(reversal.frames.at(-1)!.icons[index], compact)).toBeLessThan(1)
      }
      for (const sample of [...reversal.frames, ...reversal.changes]) { expect(sample.panelTransform).toBe('none'); expect(sample.panelOpacity).toBe('1') }
      await open('right')
      await pin.click()
      await page.mouse.move(5, 500)
      await expect(header).toHaveClass(/is-open/)
      await expect.poll(async () => (await frame()).space).toBeGreaterThan(collapsed.space + 20)
      const pinned = await frame()
      expect(pinned.space).toBeCloseTo(pinned.headerHeight, 0)
      expect(pinned.chatTop).toBeGreaterThanOrEqual(pinned.space)
      expect(pinned.messages).toBeLessThan(collapsed.messages)
      expect(pinned.composerBottom).toBeLessThanOrEqual(pinned.viewport + 1)
      await capture(`header-pinned-${size.w}.png`)
      await page.getByRole('button', { name: '取消钉住顶部栏', exact: true }).click()
      await leave()
      await expect.poll(async () => (await frame()).messages).toBe(collapsed.messages)
      console.log(JSON.stringify({ size, expandedFrames: expandedMotion.frames.length, collapsedFrames: collapsedMotion.frames.length, reversalFrames: reversal.frames.length, collapsedSpace: collapsed.space, pinnedSpace: pinned.space }))
    }

    const input = page.getByRole('textbox', { name: '对话消息', exact: true })
    await disableWebSearch(page)
    await input.fill('展示一段较长的 Markdown 内容')
    await page.getByRole('button', { name: '发送消息', exact: true }).click()
    await expect(page.locator('.chat-message.assistant .message-markdown')).toContainText('可见的模型输出')
    const body = page.locator('.chat-messages')
    await body.hover()
    await page.mouse.wheel(0, 720)
    expect(await page.evaluate(() => scrollY)).toBe(0)
    expect(await body.evaluate(element => element.scrollHeight > element.clientHeight)).toBe(true)
    await capture('chat-output-floating.png')
    await page.getByRole('button', { name: '对话参数', exact: true }).click()
    await expect(page.getByRole('spinbutton', { name: '最大输出 Token', exact: true })).toBeVisible()
    await page.getByRole('button', { name: '收起对话参数', exact: true }).click()

    await leave()
    // Chromium retains the previous sequential focus start after a body click.
    await wordmark.focus()
    await page.keyboard.press('Tab')
    await page.keyboard.press('Shift+Tab')
    await expect(wordmark).toBeFocused()
    await expect(header).toHaveClass(/is-open/)
    await page.keyboard.press('Tab')
    await expect(nav.getByRole('button', { name: '模型库', exact: true })).toBeFocused()
    await page.keyboard.press('Escape')
    await expect(header).not.toHaveClass(/is-open/)
    await expect(wordmark).toBeFocused()
    await expect(panel).toBeVisible()
    await leave(); await settle()
    await page.emulateMedia({ reducedMotion: 'reduce' })
    await beginMotion(); await open()
    const reduced = await endMotion()
    expect(reduced.frames.every(sample => sample.motions.length === 0)).toBe(true)
    for (const button of await nav.getByRole('button').all()) await expect(button).toHaveCSS('transform', 'none')
    for (let index = 0; index < labels.length; index++) {
      await leave()
      await nav.getByRole('button', { name: labels[index], exact: true }).click()
      await expect(nav.getByRole('button', { name: labels[index], exact: true })).toHaveAttribute('aria-current', 'page')
      await expect(page.locator('.workspace > section').nth(index)).toBeVisible()
    }
    await page.getByRole('button', { name: '关于 ModLoader', exact: true }).click()
    await expect(page.getByRole('dialog', { name: 'ModLoader', exact: true })).toBeVisible()
    await page.keyboard.press('Escape')
    await expect(page.getByRole('dialog', { name: 'ModLoader', exact: true })).toBeHidden()
    await open(); await pin.click()
    await page.keyboard.press('Escape')
    await expect(header).toHaveClass(/is-open/)
    await page.reload()
    await expect(page.getByRole('button', { name: '取消钉住顶部栏', exact: true })).toBeVisible()
    await expect(page.locator('.app-shell')).toHaveClass(/header-pinned/)
    expect(external).toEqual([])
    expect(writes).toEqual([])
    expect(errors).toEqual([])
    console.log(JSON.stringify({ screenshots: directory }))
  } finally { await page.unrouteAll({ behavior: 'wait' }); await application.close() }
})
