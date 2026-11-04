import { test, expect, _electron as electron, type ElectronApplication, type Locator, type Page } from '@playwright/test'
import { createServer, type ViteDevServer } from 'vite'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { pinNavigation } from './navigation'
import type { AppState, LoadProfile, ModelAsset } from '../src/types'

const root = path.resolve(__dirname, '..')
const artifactRoot = path.resolve(process.env.MODLOADER_TEST_ARTIFACT_DIR || path.join(root, '.artifacts', 'codex', 'library-reorder-20261029'))
let vite: ViteDevServer | undefined
let rendererUrl = ''
let runSequence = 0

test.beforeAll(async () => {
  fs.mkdirSync(artifactRoot, { recursive: true })
  if (process.env.MODLOADER_TEST_EXECUTABLE) return
  if (process.env.MODLOADER_DEV_URL) {
    rendererUrl = process.env.MODLOADER_DEV_URL
    return
  }
  vite = await createServer({
    root,
    cacheDir: path.join(artifactRoot, 'new-test-vite-cache'),
    optimizeDeps: { entries: ['index.html'] },
    server: {
      host: '127.0.0.1', port: 0, strictPort: false,
      watch: { ignored: ['**/data/**', '**/work/**', '**/release/**', '**/.artifacts/**', '**/.venv/**'] },
    },
  })
  await vite.listen()
  rendererUrl = vite.resolvedUrls?.local[0] || ''
  if (!rendererUrl) throw new Error('Vite 未返回本地 renderer 地址')
})

test.afterAll(async () => { await vite?.close() })

function newRunDirectory(label: string) {
  const directory = path.join(artifactRoot, `${label}-${process.pid}-${Date.now()}-${++runSequence}`)
  fs.mkdirSync(directory, { recursive: true })
  return directory
}

function fixtureState(): AppState {
  const models: ModelAsset[] = Array.from({ length: 18 }, (_, index) => {
    const suffix = String(index).padStart(2, '0')
    return {
      id: `model-${suffix}`, name: `排序样例模型 ${suffix}`, path: `/fixture/models/${suffix}`,
      format: index % 2 === 0 ? 'gguf' : 'safetensors', architecture: 'fixture', quantization: 'Q4_K_M',
      size_bytes: (index + 1) * 1024, files: [], modalities: ['text'], metadata: {},
      source: { kind: 'local' }, managed: false, warnings: [],
    }
  })
  const profile = (id: string, modelId: string, name: string, revision: number): LoadProfile => ({
    id, model_id: modelId, name, backend: 'llama_cpp', runtime_id: '', alias: id,
    parameters: {}, generation: {}, extra_args: [], revision, chat_quick_load: false,
  })
  const profiles = [
    profile('model-a-profile-1', 'model-00', 'A 配置 1', 11),
    profile('model-a-profile-2', 'model-00', 'A 配置 2', 12),
    profile('model-a-profile-3', 'model-00', 'A 配置 3', 13),
    profile('model-a-profile-4', 'model-00', 'A 配置 4', 14),
    profile('model-b-profile-1', 'model-01', 'B 配置 1', 21),
    profile('model-b-profile-2', 'model-01', 'B 配置 2', 22),
  ]
  return {
    models, profiles, runtimes: [], instances: [], downloads: [], installations: [], migrations: [],
    settings: { api_port: 12340, close_to_tray: false, font_scale: 1 }, backends: [], hardware: {},
  }
}

interface RunningFixture {
  application: ElectronApplication
  page: Page
  apiWrites: string[]
  externalRequests: string[]
  errors: string[]
}

async function launchFixture(state: AppState, dataDirectory: string): Promise<RunningFixture> {
  const application = await electron.launch({
    ...(process.env.MODLOADER_TEST_EXECUTABLE
      ? { executablePath: process.env.MODLOADER_TEST_EXECUTABLE, args: [] }
      : { args: [root] }),
    env: {
      ...process.env,
      MODLOADER_DATA_DIR: dataDirectory,
      MODLOADER_DEV_URL: rendererUrl,
      MODLOADER_SENSOR_AUTO_START: '0',
    },
    timeout: 60_000,
  })
  const page = await application.firstWindow()
  const apiWrites: string[] = [], externalRequests: string[] = [], errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/*', async route => {
    const request = route.request(), url = new URL(request.url())
    if (url.pathname === '/api/state' && request.method() === 'GET') return route.fulfill({ json: state })
    if (['http:', 'https:'].includes(url.protocol) && !['127.0.0.1', 'localhost'].includes(url.hostname)) {
      externalRequests.push(url.origin)
      return route.abort()
    }
    if (url.pathname.startsWith('/api/') && !['GET', 'HEAD', 'OPTIONS'].includes(request.method())) {
      apiWrites.push(`${request.method()} ${url.pathname}`)
      return route.abort()
    }
    return route.continue()
  })
  await pinNavigation(page)
  await page.getByRole('button', { name: '刷新状态', exact: true }).click()
  await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
  await expect(page.locator('.model-list-row[data-reorder-id]')).toHaveCount(state.models.length)
  await application.evaluate(({ BrowserWindow }) => {
    const window = BrowserWindow.getAllWindows()[0]
    window.setSize(1440, 900)
    window.show(); window.focus(); window.webContents.setZoomFactor(1)
  })
  await page.bringToFront()
  return { application, page, apiWrites, externalRequests, errors }
}

async function focusWindow(application: ElectronApplication, page: Page) {
  await application.evaluate(({ BrowserWindow }) => { const window = BrowserWindow.getAllWindows()[0]; window.show(); window.focus() })
  await page.bringToFront()
}

async function dragByLongPress(page: Page, application: ElectronApplication, source: Locator, target: Locator, position: 'before' | 'after', screenshot?: string, reducedMotion = false) {
  await source.scrollIntoViewIfNeeded()
  await target.scrollIntoViewIfNeeded()
  const sourceBox = await source.boundingBox(), targetBox = await target.boundingBox()
  if (!sourceBox || !targetBox) throw new Error('拖动源或目标当前不可见')
  await focusWindow(application, page)
  await page.mouse.move(sourceBox.x + sourceBox.width / 2, sourceBox.y + sourceBox.height / 2)
  await page.mouse.down()
  await page.waitForTimeout(540)
  await expect(page.locator('.library-drag-preview')).toBeVisible()
  const targetY = targetBox.y + targetBox.height * (position === 'before' ? 0.18 : 0.82)
  await page.mouse.move(targetBox.x + targetBox.width / 2, targetY, { steps: 8 })
  await expect(page.locator('.is-dragging')).toHaveCount(1)
  if (reducedMotion) {
    const animations = await page.locator('.profile-card[data-reorder-id]').evaluateAll(rows => rows.flatMap(row => row.getAnimations()).filter(animation => animation.id === 'reorder-position').length)
    expect(animations).toBe(0)
  }
  if (screenshot) {
    await page.evaluate(() => Promise.all(document.getAnimations().filter(animation => animation.id === 'reorder-position').map(animation => animation.finished.catch(() => {}))))
    await page.screenshot({ path: screenshot })
  }
  await page.mouse.up()
  await expect(page.locator('.library-drag-preview')).toHaveCount(0)
}

test('模型长按排序：短操作保留浏览，筛选只换可见槽位并跨重启恢复', async () => {
  test.setTimeout(120_000)
  const state = fixtureState(), directory = newRunDirectory('models')
  const dataDirectory = path.join(directory, 'data'), screenshots = directory
  let running: RunningFixture | undefined
  const sessions: RunningFixture[] = []
  const initial = state.models.map(model => model.id)
  const row = (page: Page, id: string) => page.locator(`.model-list-row[data-reorder-id="${id}"]`)
  const listOrder = (page: Page) => page.locator('.model-list-row[data-reorder-id]').evaluateAll(rows => rows.map(element => element.getAttribute('data-reorder-id')))

  try {
    running = await launchFixture(state, dataDirectory); sessions.push(running)
    const { application, page } = running
    const list = page.getByRole('region', { name: '模型列表', exact: true })
    const sort = page.getByRole('combobox', { name: '模型排序', exact: true })
    const filter = page.getByRole('combobox', { name: '筛选模型', exact: true })
    await expect(sort).toHaveValue('name-asc')
    await expect.poll(() => listOrder(page)).toEqual(initial)
    await expect(page.locator('.model-overview h2')).toHaveText('排序样例模型 00')

    await row(page, 'model-02').locator('.model-item').click()
    await expect(page.locator('.model-overview h2')).toHaveText('排序样例模型 02')
    await expect.poll(() => listOrder(page)).toEqual(initial)

    const earlySource = row(page, 'model-00').locator('.model-item')
    const earlyBox = await earlySource.boundingBox()
    if (!earlyBox) throw new Error('模型行不可见')
    await focusWindow(application, page)
    await page.mouse.move(earlyBox.x + earlyBox.width / 2, earlyBox.y + earlyBox.height / 2)
    await page.mouse.down()
    await page.waitForTimeout(120)
    await page.mouse.move(earlyBox.x + earlyBox.width / 2, earlyBox.y + earlyBox.height / 2 + 16, { steps: 2 })
    await page.mouse.up()
    await expect.poll(() => listOrder(page)).toEqual(initial)
    await expect(sort).toHaveValue('name-asc')
    await expect(page.locator('.model-overview h2')).toHaveText('排序样例模型 02')
    await expect.poll(() => page.evaluate(() => localStorage.getItem('modloader.models.order'))).toBeNull()

    // Edge auto-scroll is available during a hold; Escape cancels without saving.
    await list.evaluate(element => { element.scrollTop = 0 })
    const cancelSource = row(page, 'model-00').locator('.model-item')
    const cancelBox = await cancelSource.boundingBox(), listBox = await list.boundingBox()
    if (!cancelBox || !listBox) throw new Error('模型列表不可见')
    await focusWindow(application, page)
    await page.mouse.move(cancelBox.x + cancelBox.width / 2, cancelBox.y + cancelBox.height / 2)
    await page.mouse.down()
    await page.waitForTimeout(540)
    await expect(page.locator('.library-drag-preview')).toBeVisible()
    await page.mouse.move(listBox.x + listBox.width / 2, listBox.y + listBox.height - 5, { steps: 6 })
    await expect.poll(() => list.evaluate(element => element.scrollTop)).toBeGreaterThan(0)
    await page.keyboard.press('Escape')
    await expect(page.locator('.library-drag-preview')).toHaveCount(0)
    await page.mouse.up()
    await expect.poll(() => listOrder(page)).toEqual(initial)
    await expect(sort).toHaveValue('name-asc')
    await expect.poll(() => page.evaluate(() => localStorage.getItem('modloader.models.order'))).toBeNull()
    await list.evaluate(element => { element.scrollTop = 0 })

    // Drag from name order commits as custom and does not change the selected model.
    await dragByLongPress(page, application, row(page, 'model-00').locator('.model-item'), row(page, 'model-03'), 'before', path.join(screenshots, 'model-drag.png'))
    const afterFirstDrag = ['model-01', 'model-02', 'model-00', ...initial.slice(3)]
    await expect(sort).toHaveValue('custom')
    await expect.poll(() => listOrder(page)).toEqual(afterFirstDrag)
    await expect(page.locator('.model-overview h2')).toHaveText('排序样例模型 02')
    await expect(row(page, 'model-02').locator('.model-item')).toHaveAttribute('aria-pressed', 'true')

    await filter.selectOption('gguf')
    await expect.poll(() => listOrder(page)).toEqual(['model-02', 'model-00', ...initial.filter(id => Number(id.slice(-2)) >= 4 && Number(id.slice(-2)) % 2 === 0)])
    const fullOrderBeforeFilteredDrag = [...afterFirstDrag]
    await dragByLongPress(page, application, row(page, 'model-02'), row(page, 'model-06'), 'before')
    const filteredOrder = ['model-00', 'model-04', 'model-02', ...initial.filter(id => Number(id.slice(-2)) >= 6 && Number(id.slice(-2)) % 2 === 0)]
    await expect(sort).toHaveValue('custom')
    await expect.poll(() => listOrder(page)).toEqual(filteredOrder)
    await filter.selectOption('all')
    const expectedAll = ['model-01', 'model-00', 'model-04', 'model-03', 'model-02', 'model-05', ...initial.slice(6)]
    await expect.poll(() => listOrder(page)).toEqual(expectedAll)
    const visibleIds = new Set(['model-02', 'model-00', ...initial.filter(id => Number(id.slice(-2)) >= 4 && Number(id.slice(-2)) % 2 === 0)])
    const hiddenIds = fullOrderBeforeFilteredDrag.filter(id => !visibleIds.has(id))
    for (const id of hiddenIds) expect(expectedAll.indexOf(id)).toBe(fullOrderBeforeFilteredDrag.indexOf(id))
    await list.evaluate(element => { element.scrollTop = 0 })
    await page.screenshot({ path: path.join(screenshots, 'library-result.png') })
    await expect.poll(() => page.evaluate(() => JSON.parse(localStorage.getItem('modloader.models.order') || '[]'))).toEqual(expectedAll)

    await application.close(); running = undefined
    running = await launchFixture(state, dataDirectory); sessions.push(running)
    await expect(running.page.getByRole('combobox', { name: '模型排序', exact: true })).toHaveValue('custom')
    await expect.poll(() => listOrder(running.page)).toEqual(expectedAll)
    expect(sessions.flatMap(session => session.apiWrites)).toEqual([])
    expect(sessions.flatMap(session => session.externalRequests)).toEqual([])
    expect(sessions.flatMap(session => session.errors)).toEqual([])
  } finally {
    await running?.application.close()
  }
})

test('加载配置长按排序按模型隔离，按钮仍可编辑且重启后恢复', async () => {
  test.setTimeout(120_000)
  const state = fixtureState(), directory = newRunDirectory('profiles')
  const dataDirectory = path.join(directory, 'data'), screenshots = directory
  let running: RunningFixture | undefined
  const sessions: RunningFixture[] = []
  const profileOrder = (page: Page) => page.locator('.profile-card[data-reorder-id]').evaluateAll(rows => rows.map(element => element.getAttribute('data-reorder-id')))
  const profileCard = (page: Page, id: string) => page.locator(`.profile-card[data-reorder-id="${id}"]`)
  const modelButton = (page: Page, id: string) => page.locator(`.model-item[data-model-id="${id}"]`)

  try {
    running = await launchFixture(state, dataDirectory); sessions.push(running)
    let { application, page } = running
    await expect.poll(() => profileOrder(page)).toEqual(['model-a-profile-1', 'model-a-profile-2', 'model-a-profile-3', 'model-a-profile-4'])

    await profileCard(page, 'model-a-profile-1').getByRole('button', { name: '编辑配置', exact: true }).click()
    await expect(page.getByRole('dialog', { name: '编辑加载配置', exact: true })).toBeVisible()
    await page.getByRole('button', { name: '取消', exact: true }).click()
    await expect.poll(() => profileOrder(page)).toEqual(['model-a-profile-1', 'model-a-profile-2', 'model-a-profile-3', 'model-a-profile-4'])

    // The profile list scrolls the page; cancel a bottom-edge drag without saving.
    const cancelSource = profileCard(page, 'model-a-profile-1').getByRole('heading', { name: 'A 配置 1', exact: true })
    await cancelSource.scrollIntoViewIfNeeded()
    const cancelBox = await cancelSource.boundingBox()
    if (!cancelBox) throw new Error('配置标题不可见')
    const scrollBefore = await page.evaluate(() => window.scrollY)
    await focusWindow(application, page)
    await page.mouse.move(cancelBox.x + cancelBox.width / 2, cancelBox.y + cancelBox.height / 2)
    await page.mouse.down()
    await page.waitForTimeout(540)
    await expect(page.locator('.library-drag-preview')).toBeVisible()
    const viewport = await page.evaluate(() => ({ width: innerWidth, height: innerHeight }))
    await page.mouse.move(Math.min(viewport.width - 12, cancelBox.x + cancelBox.width / 2), viewport.height - 4, { steps: 6 })
    await expect.poll(() => page.evaluate(() => window.scrollY)).toBeGreaterThan(scrollBefore)
    await page.keyboard.press('Escape')
    await expect(page.locator('.library-drag-preview')).toHaveCount(0)
    await page.mouse.up()
    await expect.poll(() => profileOrder(page)).toEqual(['model-a-profile-1', 'model-a-profile-2', 'model-a-profile-3', 'model-a-profile-4'])
    await expect.poll(() => page.evaluate(() => localStorage.getItem('modloader.profiles.order'))).toBeNull()
    await page.evaluate(() => window.scrollTo(0, 0))
    await expect.poll(() => page.evaluate(() => window.scrollY)).toBe(0)

    await dragByLongPress(page, application,
      profileCard(page, 'model-a-profile-1').getByRole('heading', { name: 'A 配置 1', exact: true }),
      profileCard(page, 'model-a-profile-2'), 'after', path.join(screenshots, 'profile-drag.png'))
    await expect.poll(() => profileOrder(page)).toEqual(['model-a-profile-2', 'model-a-profile-1', 'model-a-profile-3', 'model-a-profile-4'])

    await modelButton(page, 'model-01').click()
    await expect(page.locator('.model-overview h2')).toHaveText('排序样例模型 01')
    await expect.poll(() => profileOrder(page)).toEqual(['model-b-profile-1', 'model-b-profile-2'])
    await page.emulateMedia({ reducedMotion: 'reduce' })
    await dragByLongPress(page, application,
      profileCard(page, 'model-b-profile-1').locator('.profile-config-summary'),
      profileCard(page, 'model-b-profile-2'), 'after', undefined, true)
    await expect.poll(() => profileOrder(page)).toEqual(['model-b-profile-2', 'model-b-profile-1'])

    await modelButton(page, 'model-00').click()
    await expect(page.locator('.model-overview h2')).toHaveText('排序样例模型 00')
    await expect.poll(() => profileOrder(page)).toEqual(['model-a-profile-2', 'model-a-profile-1', 'model-a-profile-3', 'model-a-profile-4'])
    await expect.poll(() => page.evaluate(() => JSON.parse(localStorage.getItem('modloader.profiles.order') || '[]'))).toEqual([
      'model-a-profile-2', 'model-a-profile-1', 'model-a-profile-3', 'model-a-profile-4', 'model-b-profile-2', 'model-b-profile-1',
    ])

    await application.close(); running = undefined
    running = await launchFixture(state, dataDirectory); sessions.push(running)
    ;({ application, page } = running)
    await expect.poll(() => profileOrder(page)).toEqual(['model-a-profile-2', 'model-a-profile-1', 'model-a-profile-3', 'model-a-profile-4'])
    await modelButton(page, 'model-01').click()
    await expect.poll(() => profileOrder(page)).toEqual(['model-b-profile-2', 'model-b-profile-1'])
    expect(sessions.flatMap(session => session.apiWrites)).toEqual([])
    expect(sessions.flatMap(session => session.externalRequests)).toEqual([])
    expect(sessions.flatMap(session => session.errors)).toEqual([])
  } finally {
    await running?.application.close()
  }
})
