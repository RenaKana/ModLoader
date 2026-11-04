import { test, expect, type Page } from '@playwright/test'
import { createServer, type ViteDevServer } from 'vite'
import * as fs from 'node:fs'
import * as path from 'node:path'
import type { AppState, LoadProfile, RuntimeInstance } from '../src/types'

const root = path.resolve(__dirname, '..')
const artifacts = path.join(root, '.artifacts', 'codex', 'chat-refinement-20261027', 'quick-load')
const favoriteArtifacts = path.join(root, '.artifacts', 'codex', 'chat-clear-fix-20261027', 'favorite')
let server: ViteDevServer, baseUrl: string
test.use({ channel: process.env.MODLOADER_TEST_BROWSER_CHANNEL })
test.beforeAll(async () => {
  fs.mkdirSync(artifacts, { recursive: true })
  fs.mkdirSync(favoriteArtifacts, { recursive: true })
  server = await createServer({ root, cacheDir: path.join(artifacts, 'vite-cache'), optimizeDeps: { entries: ['tests/fixtures/chat-quick-load.html'] },
    server: { host: '127.0.0.1', port: 0, strictPort: false, watch: { ignored: ['**/data/**', '**/work/**', '**/release/**', '**/.artifacts/**', '**/.venv/**'] } } })
  await server.listen()
  baseUrl = server.resolvedUrls!.local[0]
})
test.afterAll(async () => { await server?.close() })

function profile(id: string, enabled = true): LoadProfile {
  return { id, model_id: 'model', name: id === 'quick' ? '日常对话' : id, backend: 'llama_cpp', runtime_id: 'runtime', alias: id, parameters: {}, generation: {}, extra_args: [], revision: 1, chat_quick_load: enabled }
}
function instance(profileId: string, state = 'ready'): RuntimeInstance {
  return { id: `${profileId}-instance`, model_id: 'model', profile_id: profileId, profile_revision: 1, alias: profileId, backend: 'llama_cpp', runtime_id: 'runtime', state, port: 12345, started_at: new Date().toISOString(), active_requests: 0, capabilities: {} }
}
async function fixture(page: Page, options: { enabled?: boolean; loaded?: boolean; resultState?: string } = {}) {
  const state: AppState = { models: [{ id: 'model', name: 'Qwen3.8-27B', path: '/fixture/model.gguf', format: 'gguf', files: [], modalities: ['text'], size_bytes: 1, metadata: {}, source: {}, managed: false, warnings: [] }],
    profiles: [profile('quick', options.enabled ?? true), profile('other', false)], runtimes: [{ id: 'runtime', backend: 'llama_cpp', name: 'llama.cpp', path: '/fixture/runtime', version: 'fixture', managed: false, metadata: {} }],
    instances: options.loaded ? [instance('other')] : [], downloads: [], installations: [], settings: {}, hardware: {}, backends: [{ id: 'llama_cpp', label: 'llama.cpp', parameters: [] }] }
  const starts: string[] = [], patches: unknown[] = [], unexpected: string[] = [], errors: string[] = []
  let failPatch = false
  let failStop = false
  const stops: string[] = []
  let fail = false, hideInstances = false, release!: () => void
  let startGate = Promise.resolve()
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/api/**', async route => {
    const request = route.request(), pathname = new URL(request.url()).pathname
    if (pathname === '/api/state') return route.fulfill({ json: { ...state, instances: hideInstances ? [] : state.instances } })
    if (pathname === '/api/profiles/preflight') return route.fulfill({ json: { status: 'ready', parameters: [], candidates: [], errors: [], message: '' } })
    if (pathname === '/api/resources/estimate') return route.fulfill({ json: { resources: [], notes: [], source: 'quick-load-fixture' } })
    if (request.method() === 'PATCH' && pathname === '/api/profiles/quick') {
      const body = request.postDataJSON(); patches.push(body)
      if (failPatch) return route.fulfill({ status: 500, json: { detail: '保存失败夹具' } })
      state.profiles[0] = { ...state.profiles[0], ...body }
      return route.fulfill({ json: state.profiles[0] })
    }
    if (pathname === '/api/profiles/quick/start' && request.method() === 'POST') {
      expect(request.headers().accept).toBe('application/x-ndjson')
      starts.push(pathname)
      await startGate
      const progress = { type: 'progress', stage: 'startup', message: '检查启动条件' }
      if (fail) return route.fulfill({ contentType: 'application/x-ndjson', body: [progress, { type: 'error', status: 400, detail: '缺少 tokenizer 文件' }].map(event => JSON.stringify(event)).join('\n') + '\n' })
      const preparing = { ...instance('quick', options.resultState || 'preparing'), id: starts.length === 1 ? 'quick-instance' : `quick-instance-${starts.length}` }
      state.instances.push(preparing)
      return route.fulfill({ contentType: 'application/x-ndjson', body: [progress, { type: 'result', value: preparing }].map(event => JSON.stringify(event)).join('\n') + '\n' })
    }
    if (pathname.endsWith('/stop')) { stops.push(pathname); if (failStop) return route.fulfill({ status: 409, json: { detail: '模型仍有请求正在生成' } }); const target = state.instances.find(item => pathname.includes(item.id))!; target.state = 'stopped'; return route.fulfill({ json: target }) }
    unexpected.push(`${request.method()} ${pathname}`)
    return route.abort()
  })
  await page.goto(`${baseUrl}tests/fixtures/chat-quick-load.html`)
  return { state, starts, stops, patches, errors, unexpected, failPatch(value: boolean) { failPatch = value }, failStop(value: boolean) { failStop = value },
    hold(failure = false) { fail = failure; startGate = new Promise(resolve => { release = resolve }) },
    release() { release?.(); startGate = Promise.resolve() },
    hideInstances(value: boolean) { hideInstances = value },
    async refresh() { await page.getByRole('button', { name: '刷新夹具', exact: true }).click() },
    async ready() { state.instances = state.instances.map(item => item.profile_id === 'quick' ? { ...item, state: 'ready' } : item); await this.refresh() },
  }
}
const capsule = (page: Page) => page.getByRole('button', { name: /^对话模型与推理强度：/ })
const quick = (page: Page) => page.locator('[data-profile-id="quick"]')
async function list(page: Page) {
  await capsule(page).click()
  await expect(capsule(page)).toHaveAttribute('aria-expanded', 'true')
  const panel = page.getByRole('dialog', { name: '模型与推理强度', exact: true })
  if (await panel.getAttribute('data-view') !== 'models') await panel.getByRole('button', { name: '选择对话模型', exact: true }).click()
  await expect(page.getByRole('listbox', { name: '对话模型', exact: true })).toBeFocused()
  return panel
}

test('模型库图标添加和移除快捷加载，空载对话菜单直接打开', async ({ page }) => {
  const f = await fixture(page, { enabled: false })
  await expect(capsule(page)).toBeDisabled()
  await page.getByRole('button', { name: '模型库', exact: true }).click()
  await page.getByRole('button', { name: '添加到对话快捷加载', exact: true }).first().click()
  await expect(page.getByRole('button', { name: '从对话快捷加载移除', exact: true })).toHaveAttribute('aria-pressed', 'true')
  await page.locator('.profile-card').filter({ has: page.getByRole('heading', { name: '日常对话', exact: true }) }).screenshot({ path: path.join(artifacts, 'quick-load-library-toggle.png') })
  expect(f.patches).toEqual([{ chat_quick_load: true }])
  await page.getByRole('button', { name: '测试对话', exact: true }).click()
  await list(page)
  await expect(quick(page)).toContainText('llama.cpp')
  await page.keyboard.press('Escape')
  await expect(capsule(page)).toBeFocused()
  await page.getByRole('button', { name: '模型库', exact: true }).click()
  await page.getByRole('button', { name: '从对话快捷加载移除', exact: true }).click()
  await page.getByRole('button', { name: '测试对话', exact: true }).click()
  await expect(capsule(page)).toBeDisabled()
  expect(f.patches).toEqual([{ chat_quick_load: true }, { chat_quick_load: false }])
  expect(f.starts).toEqual([]); expect(f.errors).toEqual([]); expect(f.unexpected).toEqual([])
})

test('快捷加载展示检查、准备、就绪，重复操作只发起一次，错误可重试', async ({ page }) => {
  const f = await fixture(page)
  f.hold(true)
  const panel = await list(page)
  await page.keyboard.press('Enter')
  await expect(quick(page)).toContainText('检查启动条件')
  await expect(quick(page)).toHaveAttribute('aria-disabled', 'true')
  await page.keyboard.press('Enter')
  await expect.poll(() => f.starts.length).toBe(1)
  f.release()
  await expect(quick(page)).toContainText('缺少 tokenizer 文件')
  await expect(quick(page)).toContainText('重新加载')
  f.hold()
  await quick(page).click()
  await expect.poll(() => f.starts.length).toBe(2)
  f.release()
  await expect(quick(page)).toContainText('准备运行文件')
  await expect(quick(page)).toHaveAttribute('aria-disabled', 'true')
  await panel.screenshot({ path: path.join(artifacts, 'quick-load-preparing.png') })
  await page.keyboard.press('Enter')
  expect(f.starts).toHaveLength(2)
  await page.keyboard.press('Escape')
  await f.ready()
  await expect(page.getByRole('status', { name: '选中实例' })).toHaveText('quick-instance-2')
  await list(page)
  await expect(quick(page)).toContainText('已加载')
  await expect(page.locator('[data-model-id="quick-instance-2"]')).toBeVisible()
  expect(f.errors).toEqual([]); expect(f.unexpected).toEqual([])
})

test('并存已加载与快捷列表，保留键盘、焦点和推理控制，手动选择取消抢占', async ({ page }) => {
  const f = await fixture(page, { loaded: true })
  const panel = await list(page)
  await expect(panel.getByText('已加载', { exact: true })).toBeVisible()
  await expect(panel.getByText('快捷加载', { exact: true })).toBeVisible()
  await quick(page).click()
  await expect(quick(page)).toContainText('准备运行文件')
  await panel.screenshot({ path: path.join(artifacts, 'quick-load-mixed.png') })
  await page.getByRole('listbox').press('Home')
  await page.keyboard.press('Enter')
  await expect(panel).toHaveAttribute('data-view', 'reasoning')
  await expect(panel.getByRole('slider')).toBeFocused()
  await page.keyboard.press('End')
  await expect(page.getByRole('status', { name: '选中推理强度' })).toHaveText('xhigh')
  await page.keyboard.press('Escape')
  await expect(capsule(page)).toBeFocused()
  await f.ready()
  await expect(page.getByRole('status', { name: '选中实例' })).toHaveText('other-instance')
  expect(f.state.instances.find(item => item.profile_id === 'other')?.state).toBe('ready')
  expect(f.errors).toEqual([]); expect(f.unexpected).toEqual([])
})

test('切换会话不会在快捷加载完成后更改新会话模型', async ({ page }) => {
  const f = await fixture(page)
  f.hold()
  await list(page)
  await quick(page).click()
  await expect.poll(() => f.starts.length).toBe(1)
  await page.keyboard.press('Escape')
  await page.getByRole('button', { name: '切换夹具会话', exact: true }).click()
  f.release()
  await expect.poll(() => f.state.instances.length).toBe(1)
  await f.ready()
  await expect(page.getByRole('status', { name: '选中实例' })).toHaveText('')
  expect(f.errors).toEqual([]); expect(f.unexpected).toEqual([])
})

test('启动响应已就绪也等待实例进入状态列表再选中', async ({ page }) => {
  const f = await fixture(page, { resultState: 'ready' })
  await list(page)
  f.hideInstances(true)
  await quick(page).click()
  await expect.poll(() => f.starts.length).toBe(1)
  await expect(quick(page)).toContainText('已加载')
  await expect(page.getByRole('status', { name: '选中实例' })).toHaveText('')
  await page.keyboard.press('Escape')
  f.hideInstances(false)
  await f.refresh()
  await expect(page.getByRole('status', { name: '选中实例' })).toHaveText('quick-instance')
  expect(f.errors).toEqual([]); expect(f.unexpected).toEqual([])
})


test('切换先卸载选中实例，409阻止加载；另开实例保留并行且可分别卸载', async ({ page }) => {
  const f = await fixture(page, { loaded: true, resultState: 'ready' })
  await list(page)
  await page.locator('[data-model-id="other-instance"]').click()
  await page.getByRole('button', { name: '选择对话模型', exact: true }).click()
  f.failStop(true)
  await quick(page).click()
  await expect(quick(page)).toContainText('模型仍有请求正在生成')
  expect(f.starts).toHaveLength(0)
  f.failStop(false)
  await quick(page).click()
  await expect(page.getByRole('status', { name: '选中实例' })).toHaveText('quick-instance')
  expect(f.state.instances[0].state).toBe('stopped')
  await page.getByRole('button', { name: '另开实例 日常对话', exact: true }).press('Enter')
  await expect(page.getByRole('status', { name: '选中实例' })).toHaveText('quick-instance-2')
  expect(f.stops).toHaveLength(2)
  expect(f.state.instances.filter(item => item.state === 'ready')).toHaveLength(2)
  await page.locator('[data-model-id="quick-instance"]').getByRole('button', { name: '卸载 quick', exact: true }).press('Enter')
  await expect(page.locator('[data-model-id="quick-instance"]')).toHaveCount(0)
  expect(f.state.instances.find(item => item.id === 'quick-instance-2')?.state).toBe('ready')
  expect(f.errors).toEqual([]); expect(f.unexpected).toEqual([])
})

test('长预设名保持单行；关闭卸载策略保留当前实例', async ({ page }) => {
  const f = await fixture(page, { loaded: true, resultState: 'ready' })
  f.state.settings.chat_unload_before_load = false
  f.state.profiles[0].name = 'Qwen3.8-27B-NVFP4-very-long-model-preset-name-with-custom-context-and-parameters'
  await f.refresh()
  const panel = await list(page)
  const label = quick(page).locator('.chat-quick-load-label > span').first()
  await expect(label).toHaveCSS('white-space', 'nowrap')
  expect(await label.evaluate(node => node.scrollWidth > node.clientWidth)).toBe(true)
  await panel.screenshot({ path: path.join(artifacts, 'quick-load-long-name.png') })
  await page.locator('[data-model-id="other-instance"]').click()
  await page.getByRole('button', { name: '选择对话模型', exact: true }).click()
  await quick(page).click()
  await expect(page.getByRole('status', { name: '选中实例' })).toHaveText('quick-instance')
  expect(f.stops).toHaveLength(0)
  expect(f.state.instances[0].state).toBe('ready')
})


test('快捷收藏仅保存成功显示勾形确认，移除反向收起，失败不确认', async ({ page }) => {
  const f = await fixture(page, { enabled: false })
  await page.getByRole('button', { name: '模型库', exact: true }).click()
  const toggle = page.locator('.chat-quick-load-toggle').first()
  const staticBackground = () => toggle.evaluate(node => {
    const button = getComputedStyle(node), before = getComputedStyle(node, '::before')
    return { background: button.backgroundColor, border: `${button.borderStyle} ${button.borderWidth}`, beforeBackground: before.backgroundColor, beforeOpacity: before.opacity,
      confirmOpacity: getComputedStyle(node.querySelector('.quick-load-confirm')!).opacity, ringOpacity: getComputedStyle(node.querySelector('.quick-load-confirm-ring')!).opacity }
  })
  await page.mouse.move(0, 0)
  const unfavoritedBackground = await staticBackground()
  await toggle.screenshot({ path: path.join(favoriteArtifacts, 'quick-load-unfavorited.png') })
  f.failPatch(true)
  await toggle.click()
  await expect(page.getByRole('alert')).toContainText('保存失败夹具')
  await expect(toggle).not.toHaveAttribute('data-feedback')
  await expect(toggle).toHaveAttribute('aria-pressed', 'false')
  f.failPatch(false)
  await page.clock.install()
  await page.clock.pauseAt(new Date())
  await toggle.click()
  await expect(toggle).toHaveAttribute('data-feedback', 'added')
  await expect(toggle).toHaveAttribute('aria-pressed', 'true')
  await page.clock.runFor(200)
  await toggle.screenshot({ path: path.join(favoriteArtifacts, 'quick-load-confirm-added.png') })
  await page.clock.runFor(500)
  await expect(toggle).not.toHaveAttribute('data-feedback')
  await page.mouse.move(0, 0)
  await expect.poll(staticBackground).toEqual(unfavoritedBackground)
  await toggle.screenshot({ path: path.join(favoriteArtifacts, 'quick-load-favorited.png') })
  await toggle.click()
  await expect(toggle).toHaveAttribute('data-feedback', 'removed')
  await expect(toggle).toHaveAttribute('aria-pressed', 'false')
  await page.clock.runFor(200)
  await toggle.screenshot({ path: path.join(favoriteArtifacts, 'quick-load-confirm-removed.png') })
  await page.clock.runFor(500)
  await expect(toggle).not.toHaveAttribute('data-feedback')
  await page.mouse.move(0, 0)
  await expect.poll(staticBackground).toEqual(unfavoritedBackground)
  expect(f.errors).toEqual([])
})
