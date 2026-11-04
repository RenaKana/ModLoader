import { test, expect, chromium, type Browser, type Page } from '@playwright/test'
import * as fs from 'node:fs'
import * as path from 'node:path'
import type { PreviewServer } from 'vite'
import { pinNavigation } from './navigation'

const root = path.resolve(__dirname, '..')
const artifacts = process.env.MODLOADER_TEST_ARTIFACTS || path.join(root, '.artifacts/codex/managed-prefetch-20261101/ui')
let server: PreviewServer, browser: Browser, baseURL: string

test.beforeAll(async () => {
  test.setTimeout(120_000)
  fs.mkdirSync(artifacts, { recursive: true })
  const { build, preview } = await import('vite')
  const outDir = process.env.MODLOADER_TEST_DIST || path.join(artifacts, 'fixture-dist')
  if (!process.env.MODLOADER_TEST_DIST) await build({ root, cacheDir: path.join(artifacts, 'vite-cache'), build: { outDir } })
  server = await preview({ root, configFile: false, build: { outDir }, preview: { host: '127.0.0.1', port: 0 } })
  const address = server.httpServer?.address()
  if (!address || typeof address === 'string') throw new Error('Managed-prefetch fixture did not open')
  baseURL = `http://127.0.0.1:${address.port}`
  browser = await chromium.launch({ channel: process.env.MODLOADER_TEST_BROWSER_CHANNEL || 'msedge', headless: true })
})

test.afterAll(async () => { await browser?.close(); await new Promise<void>(resolve => server?.httpServer.close(() => resolve())) })

const fields = [
  { key: 'context_length', label: '上下文长度', type: 'number', default: 32768, options: [], advanced: false, description: '' },
  { key: 'configuration', label: 'Strata 配置文件', type: 'file', default: null, options: [], advanced: true, description: '' },
  { key: 'startup_expert_prefetch', label: '启动专家预取', type: 'boolean', default: null, options: [], advanced: true, description: '' },
]

function capability(available: boolean) {
  return { status: available ? 'ready' : 'missing', available, preparable: true, source: available ? 'managed' : null,
    message: available ? '启动预取已就绪' : '此环境尚未准备启动预取', engine_identity: { commit: available ? 'prepared-engine' : 'stock-engine' } }
}

async function fixture(page: Page, enabledPrefetch = false) {
  const state: any = {
    models: [{ id: 'model', name: 'Qwen3.8 MoE fixture', path: 'D:\\Fixture\\model.gguf', format: 'gguf', architecture: 'qwen4exp', files: [], size_bytes: 1, modalities: ['text'], metadata: {}, source: {}, managed: false, warnings: [] }],
    profiles: [{ id: 'legacy', model_id: 'model', name: '旧配置', backend: 'strata', runtime_id: 'old', alias: 'legacy', parameters: { context_length: 8192, ...(enabledPrefetch ? { startup_expert_prefetch: true } : {}) }, generation: {}, extra_args: [], revision: 1 }],
    runtimes: [
      { id: 'old', backend: 'strata', name: 'Strata 原环境', path: 'D:\\Fixture\\old', version: 'v0.2.4', managed: true, metadata: {}, startup_expert_prefetch: capability(true) },
      { id: 'missing', backend: 'strata', name: 'Strata 普通环境', path: 'D:\\Fixture\\missing', version: 'v0.2.5', managed: true, metadata: {}, startup_expert_prefetch: capability(false) },
      { id: 'ready', backend: 'strata', name: 'Strata 已优化环境', path: 'D:\\Fixture\\ready', version: 'v0.2.5', managed: true, metadata: {}, startup_expert_prefetch: capability(true) },
    ],
    backends: [{ id: 'strata', label: 'Strata', parameters: fields, installable: true, supported_platforms: ['windows'], import_kind: 'directory' }],
    instances: [], downloads: [], installations: [], migrations: [], settings: {}, hardware: {},
  }
  const writes: Array<{ pathname: string; query: string; body: any }> = [], unexpected: string[] = [], errors: string[] = []
  let prepareError = false, switchError = false, planNumber = 0, stateReads = 0
  let holdPreview: { started: () => void; wait: Promise<void> } | null = null
  const plans = new Map<string, any>()
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/api/**', async route => {
    const request = route.request(), url = new URL(request.url()), pathname = url.pathname
    const body = request.postDataJSON()
    if (request.method() !== 'GET') writes.push({ pathname, query: url.search, body })
    if (pathname === '/api/state') { stateReads++; return route.fulfill({ json: state }) }
    if (pathname === '/api/profiles/preflight') {
      const prefetch = state.runtimes.find((item: any) => item.id === body.runtime_id)?.startup_expert_prefetch
      const invalid = body.parameters.startup_expert_prefetch === true && !prefetch?.available
      return route.fulfill({ json: { status: invalid ? 'error' : 'ready', parameters: fields, candidates: [], errors: invalid ? ['此环境尚未准备启动预取'] : [], message: invalid ? '此环境尚未准备启动预取' : '', vision_enabled: false,
        startup_expert_prefetch: { ...prefetch, enabled: body.parameters.startup_expert_prefetch === true }, mtp: { enabled: false, status: 'unsupported', backend_supported: false, candidates: [] } } })
    }
    if (pathname === '/api/resources/estimate') return route.fulfill({ json: { resources: [], notes: [], source: 'managed-prefetch-fixture' } })
    if (pathname === '/api/profiles/legacy' && request.method() === 'PATCH') {
      Object.assign(state.profiles[0], body, { revision: state.profiles[0].revision + 1 })
      return route.fulfill({ json: state.profiles[0] })
    }
    if (pathname === '/api/runtimes/versions') return route.fulfill({ json: { versions: [{ version: 'v0.2.5', label: 'v0.2.5', status: 'available', variants: [{ id: 'cuda', label: 'CUDA' }] }], next_cursor: null } })
    if (pathname === '/api/runtimes/plan') {
      const plan = { id: `plan-${++planNumber}`, backend: body.backend, version: body.options.version, variant: body.options.variant, options: body.options,
        update_from: body.update_from, stage: { id: 'initial', title: 'Strata', files: [] } }
      plans.set(plan.id, plan); return route.fulfill({ json: plan })
    }
    if (pathname === '/api/runtimes/install') {
      const plan = plans.get(body.plan_id)
      const job = { id: `install-${plan.id}`, backend: 'strata', operation: plan.update_from ? 'update' : 'install', state: 'installing', phase: 'install', version: plan.version,
        update_from: plan.update_from, request: { backend: 'strata', options: plan.options }, name: 'Strata 安装 fixture', message: '安装隔离环境' }
      state.installations.push(job); return route.fulfill({ json: job })
    }
    const preparing = /^\/api\/runtimes\/([^/]+)\/prepare-startup-prefetch$/.exec(pathname)
    if (preparing) {
      if (prepareError) return route.fulfill({ status: 400, json: { detail: '缺少 MSVC 编译工具 fixture' } })
      const job = { id: `prepare-${state.installations.length + 1}`, backend: 'strata', operation: 'prepare_startup_prefetch', runtime_id: preparing[1], request: { backend: 'strata', runtime_id: preparing[1] },
        state: 'installing', phase: 'install', percent: null, name: '准备 Strata 启动预取', message: '编译启动预取组件 fixture' }
      state.installations.push(job); return route.fulfill({ json: job })
    }
    const switching = /^\/api\/runtimes\/([^/]+)\/(switch-preview|switch)$/.exec(pathname)
    if (switching) {
      const selected = state.profiles.filter((item: any) => body.profile_ids.includes(item.id))
      const affected = selected.filter((item: any) => item.parameters.startup_expert_prefetch === true).map((item: any) => ({ id: item.id, name: item.name }))
      const available = state.runtimes.find((item: any) => item.id === body.target_id).startup_expert_prefetch.available
      const status = !affected.length ? 'unchanged' : body.disable_startup_expert_prefetch ? 'disabled' : available ? 'preserved' : 'choice_required'
      const result = { profiles: selected.map((item: any) => ({ id: item.id, name: item.name })), warnings: [], startup_expert_prefetch: { status, affected_profiles: affected,
        message: status === 'choice_required' ? '目标环境缺少启动预取' : status === 'disabled' ? '将关闭启动预取' : status === 'preserved' ? '启动预取将保留' : '' } }
      if (switching[2] === 'switch') {
        if (switchError) return route.fulfill({ status: 409, json: { detail: '切换预览已过期 fixture，请重新检查' } })
        if (status === 'choice_required') return route.fulfill({ status: 409, json: { detail: '请明确选择是否关闭启动预取' } })
        for (const profile of selected) { profile.runtime_id = body.target_id; if (body.disable_startup_expert_prefetch) profile.parameters.startup_expert_prefetch = false; profile.revision++ }
      } else if (holdPreview) {
        const gate = holdPreview; holdPreview = null; gate.started(); await gate.wait
      }
      return route.fulfill({ json: result }).catch(() => {})
    }
    const removing = /^\/api\/runtimes\/([^/]+)$/.exec(pathname)
    if (removing && request.method() === 'DELETE') {
      const replacement = url.searchParams.get('replacement_id')
      for (const profile of state.profiles.filter((item: any) => item.runtime_id === removing[1])) {
        profile.runtime_id = replacement; if (url.searchParams.get('disable_startup_expert_prefetch') === 'true') profile.parameters.startup_expert_prefetch = false; profile.revision++
      }
      state.runtimes = state.runtimes.filter((item: any) => item.id !== removing[1]); return route.fulfill({ json: { ok: true } })
    }
    if (pathname === '/api/chat-presets') return route.fulfill({ json: { items: [], default_id: null } })
    if (pathname === '/api/conversations') return route.fulfill({ json: { items: [], active_id: null } })
    if (pathname === '/api/platforms/wsl') return route.fulfill({ json: { available: true, distributions: [] } })
    if (pathname === '/api/installations/history') {
      const items = state.installations.filter((item: any) => ['completed', 'failed'].includes(item.state))
      return route.fulfill({ json: { items, total: items.length, page: 1, page_size: 10 } })
    }
    unexpected.push(`${request.method()} ${pathname}`)
    return route.fulfill({ status: 404, json: { detail: 'Unexpected managed-prefetch fixture request' } })
  })
  await page.goto(baseURL)
  await pinNavigation(page)
  await expect(page.getByText('服务在线', { exact: true })).toBeVisible()
  return { state, writes, unexpected, errors,
    failPreparation(value: boolean) { prepareError = value }, failSwitch(value: boolean) { switchError = value },
    holdNextPreview() { let started!: () => void, release!: () => void; const began = new Promise<void>(resolve => { started = resolve }), wait = new Promise<void>(resolve => { release = resolve }); holdPreview = { started, wait }; return { started: began, release } },
    async refresh() { await page.getByRole('button', { name: '刷新状态', exact: true }).click() },
    async pollRefresh() { const before = stateReads; await expect.poll(() => stateReads).toBeGreaterThan(before) },
  }
}

test('启动预取位于文件与启动选项，旧配置不补默认值且显式关闭可保存', async () => {
  const page = await browser.newPage({ viewport: { width: 1366, height: 960 } })
  try {
    const f = await fixture(page)
    await page.getByRole('button', { name: '编辑配置', exact: true }).click()
    const dialog = page.getByRole('dialog', { name: '编辑加载配置', exact: true })
    const prefetch = dialog.getByRole('combobox', { name: '启动专家预取', exact: true })
    await expect(prefetch).toBeHidden()
    await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(dialog).toBeHidden()
    expect(f.writes.filter(write => write.pathname === '/api/profiles/legacy').at(-1)?.body.parameters).toEqual({ context_length: 8192 })
    await page.getByRole('button', { name: '关闭通知', exact: true }).click()
    await page.getByRole('button', { name: '编辑配置', exact: true }).click()
    await dialog.getByRole('button', { name: '文件与启动选项', exact: true }).click()
    await expect(prefetch).toHaveValue('')
    await expect(prefetch.locator('option').first()).toHaveText('引擎默认')
    await prefetch.selectOption('false')
    await expect(dialog.getByText('复用已有资源', { exact: true })).toBeVisible()
    await dialog.screenshot({ path: path.join(artifacts, 'profile-prefetch-off.png') })
    await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(dialog).toBeHidden()
    expect(f.writes.filter(write => write.pathname === '/api/profiles/legacy').at(-1)?.body.parameters).toEqual({ context_length: 8192, startup_expert_prefetch: false })
    await page.getByRole('button', { name: '编辑配置', exact: true }).click()
    await expect(prefetch).toHaveValue('false')
    expect(f.errors).toEqual([])
    expect(f.unexpected).toEqual([])
  } finally { await page.close() }
})

test('缺少预取产物时可准备，错误可见且复用现有任务进度和重试', async () => {
  const page = await browser.newPage({ viewport: { width: 1366, height: 960 } })
  try {
    const f = await fixture(page)
    f.state.profiles[0].runtime_id = 'missing'; await f.refresh()
    await page.getByRole('button', { name: '编辑配置', exact: true }).click()
    const dialog = page.getByRole('dialog', { name: '编辑加载配置', exact: true })
    await dialog.getByRole('button', { name: '文件与启动选项', exact: true }).click()
    await expect(dialog.getByText('此环境尚未准备启动预取', { exact: true })).toBeVisible()
    f.failPreparation(true)
    await dialog.getByRole('button', { name: '准备启动预取', exact: true }).click()
    await expect(dialog.getByRole('alert')).toHaveText('缺少 MSVC 编译工具 fixture')
    expect(f.state.profiles[0].parameters).not.toHaveProperty('startup_expert_prefetch')
    await dialog.getByRole('combobox', { name: '运行环境', exact: true }).selectOption('ready')
    await expect(dialog.getByRole('alert')).toHaveCount(0)
    await dialog.getByRole('combobox', { name: '运行环境', exact: true }).selectOption('missing')
    f.failPreparation(false)
    await dialog.getByRole('button', { name: '准备启动预取', exact: true }).click()
    await expect(dialog.getByRole('button', { name: '准备启动预取', exact: true })).toBeDisabled()
    await dialog.getByRole('button', { name: '取消', exact: true }).click()
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '运行环境', exact: true }).click()
    const pending = page.locator('[data-job-id="prepare-1"]')
    await expect(pending).toContainText('编译启动预取组件 fixture')
    await expect(pending.getByText('启动预取', { exact: true })).toBeVisible()
    await page.evaluate(() => window.scrollTo(0, 0))
    await page.screenshot({ path: path.join(artifacts, 'prefetch-job-progress.png'), fullPage: true })
    f.state.installations[0].state = 'failed'; f.state.installations[0].error = '编译中断 fixture'; await f.refresh()
    await page.getByRole('tab', { name: /历史记录/ }).click()
    await expect(pending.getByRole('alert')).toContainText('编译中断 fixture')
    await pending.getByRole('button', { name: '重试准备', exact: true }).click()
    await expect(page.locator('[data-job-id="prepare-2"]')).toContainText('编译启动预取组件 fixture')
    const runtime = page.locator('[data-runtime-id="missing"]')
    await expect(runtime.getByRole('button', { name: '准备启动预取', exact: true })).toBeDisabled()
    f.state.installations[1].state = 'completed'; f.state.runtimes.find((item: any) => item.id === 'missing').startup_expert_prefetch = capability(true); await f.refresh()
    await expect(runtime.getByText('启动预取已就绪', { exact: true })).toBeVisible()
    await expect(runtime.getByRole('button', { name: '准备启动预取', exact: true })).toHaveCount(0)
    await page.evaluate(() => window.scrollTo(0, 0))
    await page.locator('[data-backend-id="strata"]').screenshot({ path: path.join(artifacts, 'runtime-prefetch-ready.png') })
    expect(f.writes.filter(write => write.pathname.endsWith('/prepare-startup-prefetch'))).toHaveLength(3)
    expect(f.errors).toEqual([]); expect(f.unexpected).toEqual([])
  } finally { await page.close() }
})

test('一键安装可选择预取，更新已优化环境保留偏好并允许明确关闭', async () => {
  const page = await browser.newPage({ viewport: { width: 1366, height: 960 } })
  try {
    const f = await fixture(page)
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '运行环境', exact: true }).click()
    const backend = page.locator('[data-backend-id="strata"]')
    await backend.getByRole('button', { name: '安装新环境', exact: true }).click()
    let modal = page.getByRole('dialog', { name: '安装 Strata', exact: true })
    const choose = async () => { await modal.getByRole('combobox', { name: '官方版本', exact: true }).selectOption('v0.2.5'); await modal.getByRole('combobox', { name: '运行类型', exact: true }).selectOption('cuda') }
    await expect(modal.getByRole('checkbox', { name: '准备启动预取', exact: true })).not.toBeChecked()
    await modal.getByRole('checkbox', { name: '准备启动预取', exact: true }).check()
    await choose(); await modal.getByRole('button', { name: '开始安装', exact: true }).click(); await expect(modal).toBeHidden()
    expect(f.writes.find(write => write.pathname === '/api/runtimes/plan')?.body.options).toMatchObject({ startup_expert_prefetch: true })
    const runtime = page.locator('[data-runtime-id="old"]')
    await runtime.getByRole('button', { name: '更新环境', exact: true }).click()
    modal = page.getByRole('dialog', { name: '更新 Strata', exact: true })
    await expect(modal.getByRole('checkbox', { name: '准备启动预取', exact: true })).toBeChecked()
    await choose(); await modal.screenshot({ path: path.join(artifacts, 'update-prefetch-selected.png') })
    await modal.getByRole('button', { name: '开始更新', exact: true }).click(); await expect(modal).toBeHidden()
    expect(f.writes.filter(write => write.pathname === '/api/runtimes/plan').at(-1)?.body.options).toMatchObject({ startup_expert_prefetch: true })
    await runtime.getByRole('button', { name: '更新环境', exact: true }).click()
    await expect(modal.getByRole('checkbox', { name: '准备启动预取', exact: true })).toBeChecked()
    await modal.getByRole('checkbox', { name: '准备启动预取', exact: true }).uncheck()
    await choose(); await modal.getByRole('button', { name: '开始更新', exact: true }).click(); await expect(modal).toBeHidden()
    expect(f.writes.filter(write => write.pathname === '/api/runtimes/plan').at(-1)?.body.options).toMatchObject({ startup_expert_prefetch: false })
    expect(f.writes.filter(write => write.pathname === '/api/runtimes/install')).toHaveLength(3)
    expect(f.errors).toEqual([]); expect(f.unexpected).toEqual([])
  } finally { await page.close() }
})

test('切换预览显示预取保留且不改变预取参数', async () => {
  const page = await browser.newPage({ viewport: { width: 1366, height: 960 } })
  try {
    const f = await fixture(page, true)
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '运行环境', exact: true }).click()
    await page.locator('[data-runtime-id="old"]').getByRole('button', { name: '切换配置', exact: true }).click()
    const modal = page.getByRole('dialog', { name: '切换 Strata 原环境 的加载配置', exact: true })
    await modal.getByRole('combobox', { name: '目标运行环境', exact: true }).selectOption('ready')
    await modal.getByRole('button', { name: '检查切换', exact: true }).click()
    await expect(modal.getByText('启动预取将保留', { exact: true })).toBeVisible()
    await modal.screenshot({ path: path.join(artifacts, 'switch-prefetch-preserved.png') })
    await modal.getByRole('button', { name: '确认切换 1 个配置', exact: true }).click(); await expect(modal).toBeHidden()
    expect(f.state.profiles[0].runtime_id).toBe('ready'); expect(f.state.profiles[0].parameters.startup_expert_prefetch).toBe(true)
    expect(f.writes.find(write => write.pathname.endsWith('/switch'))?.body).toEqual({ target_id: 'ready', profile_ids: ['legacy'] })
    expect(f.state.runtimes.some((item: any) => item.id === 'old')).toBe(true)
    expect(f.errors).toEqual([]); expect(f.unexpected).toEqual([])
  } finally { await page.close() }
})

test('目标缺少预取时可保留原环境，或明确关闭预取并切换', async () => {
  const page = await browser.newPage({ viewport: { width: 940, height: 740 } })
  try {
    const f = await fixture(page, true)
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '运行环境', exact: true }).click()
    const open = async () => {
      await page.locator('[data-runtime-id="old"]').getByRole('button', { name: '切换配置', exact: true }).click()
      const modal = page.getByRole('dialog', { name: '切换 Strata 原环境 的加载配置', exact: true })
      await modal.getByRole('combobox', { name: '目标运行环境', exact: true }).selectOption('missing')
      await modal.getByRole('button', { name: '检查切换', exact: true }).click()
      return modal
    }
    let modal = await open()
    await expect(modal.getByText('目标环境缺少启动预取', { exact: true })).toBeVisible()
    await expect(modal.getByRole('button', { name: '确认切换 1 个配置', exact: true })).toHaveCount(0)
    await modal.getByRole('button', { name: '保留原环境', exact: true }).click(); await expect(modal).toBeHidden()
    expect(f.writes.filter(write => write.pathname.endsWith('/switch'))).toEqual([])
    expect(f.state.profiles[0].parameters.startup_expert_prefetch).toBe(true)
    modal = await open()
    await modal.screenshot({ path: path.join(artifacts, 'switch-prefetch-missing-940.png') })
    expect(await modal.evaluate(element => element.scrollWidth <= element.clientWidth + 2)).toBe(true)
    await modal.getByRole('button', { name: '关闭预取并切换', exact: true }).click(); await expect(modal).toBeHidden()
    expect(f.writes.filter(write => write.pathname.endsWith('/switch-preview')).at(-1)?.body).toEqual({ target_id: 'missing', profile_ids: ['legacy'], disable_startup_expert_prefetch: true })
    expect(f.writes.filter(write => write.pathname.endsWith('/switch')).at(-1)?.body).toEqual({ target_id: 'missing', profile_ids: ['legacy'], disable_startup_expert_prefetch: true })
    expect(f.state.profiles[0].runtime_id).toBe('missing'); expect(f.state.profiles[0].parameters.startup_expert_prefetch).toBe(false)
    expect(f.errors).toEqual([]); expect(f.unexpected).toEqual([])
  } finally { await page.close() }
})

test('预览响应过期、目标改变和后端拒绝后必须重新检查，不能沿用旧预览', async () => {
  const page = await browser.newPage({ viewport: { width: 1366, height: 960 } })
  try {
    const f = await fixture(page, true)
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '运行环境', exact: true }).click()
    await page.locator('[data-runtime-id="old"]').getByRole('button', { name: '切换配置', exact: true }).click()
    const modal = page.getByRole('dialog', { name: '切换 Strata 原环境 的加载配置', exact: true })
    const target = modal.getByRole('combobox', { name: '目标运行环境', exact: true })
    await target.selectOption('ready')
    const held = f.holdNextPreview()
    await modal.getByRole('button', { name: '检查切换', exact: true }).click(); await held.started
    f.state.profiles[0].revision++; f.state.profiles[0].name = '已更改配置'; await f.pollRefresh()
    await expect(modal.getByRole('checkbox', { name: /已更改配置/ })).toBeVisible(); held.release()
    await expect(modal.getByRole('button', { name: '检查切换', exact: true })).toBeEnabled()
    await expect(modal.getByText('将切换 1 个加载配置', { exact: true })).toHaveCount(0)
    await modal.getByRole('button', { name: '检查切换', exact: true }).click()
    await expect(modal.getByText('启动预取将保留', { exact: true })).toBeVisible()
    await target.selectOption('missing')
    await expect(modal.getByText('启动预取将保留', { exact: true })).toHaveCount(0)
    await expect(modal.getByRole('button', { name: '检查切换', exact: true })).toBeEnabled()
    await target.selectOption('ready'); await modal.getByRole('button', { name: '检查切换', exact: true }).click()
    f.failSwitch(true)
    await modal.getByRole('button', { name: '确认切换 1 个配置', exact: true }).click()
    await expect(modal.getByRole('alert')).toContainText('切换预览已过期 fixture，请重新检查')
    await expect(modal.getByRole('button', { name: '检查切换', exact: true })).toBeEnabled()
    expect(f.state.profiles[0].runtime_id).toBe('old')
    expect(f.errors).toEqual([]); expect(f.unexpected).toEqual([])
  } finally { await page.close() }
})

test('移除环境迁移到普通环境时，关闭预取必须明确选择', async () => {
  const page = await browser.newPage({ viewport: { width: 1366, height: 960 } })
  try {
    const f = await fixture(page, true)
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '运行环境', exact: true }).click()
    await page.locator('[data-runtime-id="old"]').getByRole('button', { name: '移除 / 卸载', exact: true }).click()
    const modal = page.getByRole('dialog', { name: '移除 Strata 原环境？', exact: true })
    await modal.getByRole('combobox', { name: '迁移到同一后端的其他环境', exact: true }).selectOption('missing')
    await expect(modal.getByText('目标环境缺少启动预取', { exact: true })).toBeVisible()
    await expect(modal.getByRole('button', { name: '迁移配置并移除登记', exact: true })).toBeDisabled()
    await modal.getByRole('checkbox', { name: '关闭预取并迁移', exact: true }).check()
    await modal.getByRole('button', { name: '迁移配置并移除登记', exact: true }).click(); await expect(modal).toBeHidden()
    const removal = f.writes.find(write => write.pathname === '/api/runtimes/old' && write.query.includes('replacement_id'))
    expect(new URLSearchParams(removal?.query).get('disable_startup_expert_prefetch')).toBe('true')
    expect(f.state.profiles[0].parameters.startup_expert_prefetch).toBe(false)
    expect(f.errors).toEqual([]); expect(f.unexpected).toEqual([])
  } finally { await page.close() }
})
