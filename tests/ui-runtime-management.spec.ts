import { test, expect, _electron as electron, type Route } from '@playwright/test'
import * as path from 'node:path'
import * as fs from 'node:fs'
import { pinNavigation } from './navigation'
import { runtimeDisplayName, runtimeLabel } from '../src/runtime-presentation'
import type { RuntimeInstallation } from '../src/types'

async function fixture() {
  const root = path.resolve(__dirname, '..')
  const artifacts = process.env.MODLOADER_TEST_ARTIFACT_DIR || path.join(root, '.artifacts', 'codex', 'ui-runtime-management')
  fs.mkdirSync(artifacts, { recursive: true })
  const directory = fs.mkdtempSync(path.join(artifacts, 'ui-runtime-management-'))
  fs.mkdirSync(path.join(directory, 'data', 'desktop', 'session'), { recursive: true })
  const application = await electron.launch({
    ...(process.env.MODLOADER_TEST_EXECUTABLE ? { executablePath: process.env.MODLOADER_TEST_EXECUTABLE, args: [] } : { args: [root] }),
    env: { ...process.env, MODLOADER_DATA_DIR: path.join(directory, 'data') }, timeout: 60_000,
  })
  const page = await application.firstWindow()
  const profiles = ['profile-a', 'profile-b', 'profile-external'].map((id, index) => ({
    id, model_id: 'fixture-model', name: ['配置甲', '配置乙', '外部配置'][index], backend: 'llama_cpp', runtime_id: index === 2 ? 'external' : 'old', alias: id,
    parameters: { context: 4096 + index, vision_enabled: index === 0 }, generation: { temperature: 0.7 }, extra_args: ['--fixture'], revision: 1,
  }))
  const engine = { id: 'engine', name: 'llama-engine.zip', purpose: 'llama.cpp 服务与运行库', size_bytes: 1024, cached: true, status: 'completed' }
  const dependency = { id: 'dependency', name: 'dependency.whl', purpose: '当前环境所需的 Python 依赖', size_bytes: null, cached: false, status: 'pending' }
  const initialStage = { id: 'initial', title: '初始运行文件', files: [engine] }
  const nextStage = { id: 'dependencies', title: 'Python 依赖', files: [dependency] }
  const state: any = {
    models: [], profiles, downloads: [], settings: { close_to_tray: false }, hardware: {},
    backends: [
      { id: 'llama_cpp', label: 'llama.cpp', parameters: [], description: 'GGUF 本地推理环境', subtitle: 'GGUF · CPU / CUDA', symbol: 'L', installable: true, import_kind: 'file' },
      { id: 'vllm', label: 'vLLM', parameters: [], description: '独立 WSL 环境', subtitle: 'WSL2 / CUDA', symbol: 'V', installable: true, import_kind: 'wsl' },
    ],
    runtimes: [
      { id: 'old', backend: 'llama_cpp', name: '旧环境', version: 'b1', path: 'C:\\Fixture\\old\\llama-server.exe', managed: true, metadata: { install_root: 'C:\\Fixture\\old' } },
      { id: 'new', backend: 'llama_cpp', name: '新环境', version: 'b2', path: 'C:\\Fixture\\new\\llama-server.exe', managed: true, metadata: { install_root: 'C:\\Fixture\\new' } },
      { id: 'external', backend: 'llama_cpp', name: '外部环境', version: '', path: 'C:\\External\\llama-server.exe', managed: false, metadata: {} },
      { id: 'wsl', backend: 'vllm', name: 'WSL 环境', version: '0.1', path: '/opt/modloader/fixture/bin/vllm', managed: true, metadata: { distribution: 'ModLoader', install_root: 'C:\\Fixture\\wsl' } },
    ],
    instances: [{ id: 'live-old', alias: '使用旧环境的实例', backend: 'llama_cpp', runtime_id: 'old', model_id: 'fixture-model', profile_id: 'profile-a', profile_revision: 1, state: 'ready', active_requests: 0, port: 0, started_at: '', capabilities: {} }],
    installations: [
      { id: 'paused', backend: 'llama_cpp', name: '暂停中的安装', state: 'paused', operation: 'install', can_pause: true, version: 'b2', downloaded_bytes: 512, total_bytes: null, completed_files: 1, total_files: 2, manifest: nextStage, stages: [initialStage, nextStage], created_at: '2026-10-16T01:00:00Z' },
      { id: 'awaiting', backend: 'llama_cpp', name: '待确认的安装', state: 'awaiting_confirmation', operation: 'install', can_pause: false, version: 'b2', downloaded_bytes: 1024, total_bytes: null, manifest: nextStage, stages: [initialStage, nextStage], created_at: null },
    ],
  }
  let history: any[] = [
    { id: 'updated', backend: 'llama_cpp', name: '已完成的独立更新', operation: 'update', state: 'completed', update_from: 'old', runtime_id: 'new', version: 'b2', created_at: '2026-10-16T01:00:00Z', finished_at: '2026-10-16T01:02:00Z', manifest: initialStage, stages: [initialStage] },
    { id: 'legacy', backend: 'llama_cpp', name: '旧版中断记录', state: 'interrupted', request: { backend: 'llama_cpp', options: { flavor: 'cpu' } }, created_at: null, finished_at: null },
    ...Array.from({ length: 10 }, (_, index) => ({ id: `history-${index}`, backend: 'llama_cpp', name: `历史安装 ${index}`, state: 'cancelled', version: 'b1', created_at: null, finished_at: null })),
  ]
  const writes: Array<{ method: string; path: string; query: string; body: any }> = []
  const errors: string[] = [], unexpected: string[] = [], external: string[] = []
  const historyReads: number[] = []
  let planNumber = 0
  let holdPlan: { release: () => void; started: Promise<void>; finished: Promise<void>; complete: () => void } | null = null
  let holdResolve: (() => void) | undefined
  const plans = new Map<string, any>()
  page.on('pageerror', error => errors.push(error.message))

  async function fulfill(route: Route, json: unknown, status = 200) { await route.fulfill({ status, json }) }
  await page.route('**/*', async route => {
    const req = route.request(), url = new URL(req.url()), pathname = url.pathname
    if (['http:', 'https:'].includes(url.protocol) && !['127.0.0.1', 'localhost'].includes(url.hostname)) { external.push(url.origin); await route.abort(); return }
    if (!pathname.startsWith('/api/')) { await route.continue(); return }
    // Chat selection autosaves even while this page is hidden. Use the isolated
    // real conversation store so app shutdown can flush without a native prompt.
    if (pathname === '/api/conversations' || pathname.startsWith('/api/conversations/')) { await route.continue(); return }
    const body = req.postDataJSON()
    if (req.method() !== 'GET') writes.push({ method: req.method(), path: pathname, query: url.search, body })
    if (pathname === '/api/state') { await fulfill(route, state); return }
    if (pathname === '/api/installations/history' && req.method() === 'GET') {
      const page = Number(url.searchParams.get('page') || 1)
      historyReads.push(page)
      await fulfill(route, { items: history.slice((page - 1) * 10, page * 10), total: history.length, page, page_size: 10 }); return
    }
    if (pathname === '/api/runtimes/versions') {
      await fulfill(route, { versions: url.searchParams.has('cursor') ? [{ version: 'b0', label: 'b0', status: 'available', variants: [{ id: 'cpu', label: 'CPU' }] }] : [
        { version: 'b3', label: 'b3', status: 'available', recommended: true, variants: [{ id: 'cpu', label: 'CPU' }, { id: 'cuda', label: 'CUDA' }] },
        { version: 'b2', label: 'b2', status: 'unverified', reason: '此版本尚未完成本机验证', variants: [{ id: 'cpu', label: 'CPU' }] },
        { version: 'missing', label: '缺失发行件', status: 'unavailable', reason: '官方发行件不适用于当前平台', variants: [] },
      ], next_cursor: url.searchParams.has('cursor') ? null : 'older' }); return
    }
    if (pathname === '/api/runtimes/plan') {
      const id = `plan-${++planNumber}`
      const plan = { id, backend: body.backend, version: body.options.version, variant: body.options.variant, options: body.options, stage: { id: `initial-${id}`, title: '安装文件预览', files: [{ ...engine, status: undefined }, { ...dependency, name: `${body.options.variant}-dependency.whl` }] } }
      plans.set(id, { ...plan, update_from: body.update_from })
      if (holdPlan) {
        const held = holdPlan
        holdResolve?.(); await new Promise<void>(resolve => { held.release = resolve })
        await fulfill(route, plan).catch(() => {}); held.complete(); return
      }
      await fulfill(route, plan).catch(() => {}); return
    }
    if (pathname === '/api/runtimes/install') {
      const plan = plans.get(body.plan_id)
      state.installations.push({ id: `new-install-${plan.id}`, name: '新建独立环境', backend: plan.backend, state: 'downloading', can_pause: true, operation: plan.update_from ? 'update' : 'install', update_from: plan.update_from, version: plan.version, request: { backend: plan.backend, options: plan.options }, manifest: plan.stage, stages: [plan.stage], file: engine.name, file_downloaded_bytes: 256, file_total_bytes: 1024, downloaded_bytes: 256, total_bytes: null, completed_files: 0, total_files: 2 })
      await fulfill(route, state.installations.at(-1)); return
    }
    const jobAction = /^\/api\/jobs\/([^/]+)\/(pause|resume|cancel|confirm)$/.exec(pathname)
    if (jobAction) {
      const job = state.installations.find((item: any) => item.id === jobAction[1])
      job.state = jobAction[2] === 'pause' ? 'paused' : jobAction[2] === 'cancel' ? 'cancelled' : 'downloading'
      if (jobAction[2] === 'cancel') { state.installations = state.installations.filter((item: any) => item.id !== job.id); history.unshift(job) }
      await fulfill(route, job); return
    }
    const switching = /^\/api\/runtimes\/([^/]+)\/(switch-preview|switch)$/.exec(pathname)
    if (switching) {
      const selected = state.profiles.filter((profile: any) => body.profile_ids.includes(profile.id))
      if (switching[2] === 'switch') for (const profile of selected) { profile.runtime_id = body.target_id; profile.revision++ }
      await fulfill(route, { profiles: selected.map((profile: any) => ({ id: profile.id, name: profile.name })), warnings: [], source: {}, target: {} }); return
    }
    const removing = /^\/api\/runtimes\/([^/]+)$/.exec(pathname)
    if (removing && req.method() === 'DELETE') {
      for (const profile of state.profiles) if (profile.runtime_id === removing[1]) { profile.runtime_id = url.searchParams.get('replacement_id'); profile.revision++ }
      state.runtimes = state.runtimes.filter((runtime: any) => runtime.id !== removing[1]); await fulfill(route, { ok: true }); return
    }
    if (pathname === '/api/instances/live-old/stop') { state.instances[0].state = 'stopped'; await fulfill(route, { ok: true }); return }
    if (req.method() === 'DELETE' && pathname === '/api/installations/history') { history = []; await fulfill(route, { ok: true }); return }
    if (req.method() === 'DELETE' && pathname.startsWith('/api/jobs/')) { history = history.filter(job => job.id !== pathname.split('/').at(-1)); await fulfill(route, { ok: true }); return }
    if (req.method() === 'GET') { await route.continue(); return }
    unexpected.push(`${req.method()} ${pathname}`); await fulfill(route, { detail: '未授权的 fixture 写入' }, 404)
  })
  await pinNavigation(page)
  await page.getByRole('button', { name: '刷新状态', exact: true }).click()
  await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
  // An empty chat hides this status only after history is ready, error-free,
  // and saved; wait for the automatic selection update to settle before closing.
  await expect(page.locator('.chat-capsule-model')).toHaveText(state.instances[0].alias)
  await expect(page.locator('.chat-save-status')).toHaveCount(0)
  await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(1440, 960))
  async function screenshot(name: string) {
    await page.evaluate(() => document.fonts.ready)
    const png = await application.evaluate(async ({ BrowserWindow }) => (await BrowserWindow.getAllWindows()[0].webContents.capturePage()).toPNG().toString('base64'))
    fs.writeFileSync(path.join(directory, name), Buffer.from(png, 'base64'))
  }
  return { application, page, directory, state, writes, errors, unexpected, external, historyReads, screenshot,
    holdNextPlan() { let started!: () => void, complete!: () => void; const value = { release: () => {}, started: new Promise<void>(resolve => { started = resolve }), finished: new Promise<void>(resolve => { complete = resolve }), complete: () => complete() }; holdResolve = started; holdPlan = value; return value },
    releasePlan() { const held = holdPlan; holdPlan = null; held?.release() },
  }
}

test('运行环境详情展开与收起保留原位入口和右侧操作', async () => {
  test.setTimeout(90_000)
  const f = await fixture()
  const { page, application, state } = f
  state.profiles.push(...Array.from({ length: 4 }, (_, index) => ({
    ...state.profiles[2], id: `external-profile-${index}`, name: `外部配置 ${index + 2}`,
  })))
  const externalRuntime = state.runtimes.find((runtime: any) => runtime.id === 'external')
  state.runtimes = [externalRuntime, ...state.runtimes.filter((runtime: any) => runtime.id !== 'external')]
  try {
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '运行环境', exact: true }).click()
    await page.evaluate(() => document.fonts.ready)
    const row = page.locator('[data-runtime-id="external"]')
    const details = row.locator('.runtime-usage-details')
    const summary = details.locator('summary')
    const usage = row.locator('.runtime-usage')
    const actions = row.locator('.runtime-row-actions')
    await expect(usage).toContainText('5 个加载配置')
    await row.scrollIntoViewIfNeeded()
    const closedSummary = await summary.boundingBox()
    const closedUsage = await usage.boundingBox()
    const closedActions = await actions.boundingBox()
    expect(closedSummary).not.toBeNull(); expect(closedUsage).not.toBeNull(); expect(closedActions).not.toBeNull()
    expect(Math.abs(closedSummary!.y - closedUsage!.y)).toBeLessThan(1)
    await row.screenshot({ path: path.join(f.directory, 'runtime-disclosure-closed.png'), animations: 'disabled' })

    await summary.click()
    await expect(details).toHaveAttribute('open', '')
    await expect(details.locator('dd').last()).toHaveText(externalRuntime.path)
    await expect(details.getByText('加载配置：外部配置、外部配置 2、外部配置 3、外部配置 4、外部配置 5', { exact: true })).toBeVisible()
    await row.screenshot({ path: path.join(f.directory, 'runtime-disclosure-open.png'), animations: 'disabled' })
    await f.screenshot('runtime-disclosure-page-open.png')
    const openedSummary = await summary.boundingBox()
    const openedActions = await actions.boundingBox()
    const content = await details.locator('dl').boundingBox()
    expect(openedSummary).not.toBeNull(); expect(openedActions).not.toBeNull(); expect(content).not.toBeNull()
    expect.soft(Math.abs(openedSummary!.x - closedSummary!.x), '详情入口横向位置').toBeLessThan(1)
    expect.soft(Math.abs(openedSummary!.y - closedSummary!.y), '详情入口纵向位置').toBeLessThan(1)
    expect.soft(Math.abs(openedActions!.y - closedActions!.y), '右侧操作纵向位置').toBeLessThan(1)
    expect(content!.y).toBeGreaterThanOrEqual(openedSummary!.y + openedSummary!.height)
    expect(content!.x).toBeGreaterThanOrEqual(closedUsage!.x)
    expect(content!.x + content!.width).toBeLessThanOrEqual(openedActions!.x)

    await summary.focus()
    await summary.press('Enter')
    await expect(details).not.toHaveAttribute('open', '')
    await expect(details.locator('dl')).toBeHidden()
    const closedAgain = await summary.boundingBox()
    expect(closedAgain).not.toBeNull()
    expect(Math.abs(closedAgain!.x - closedSummary!.x)).toBeLessThan(1)
    expect(Math.abs(closedAgain!.y - closedSummary!.y)).toBeLessThan(1)

    const managed = page.locator('[data-runtime-id="old"]')
    const managedSummary = managed.locator('.runtime-usage-details summary')
    const managedActions = managed.locator('.runtime-row-actions')
    const managedClosedSummary = await managedSummary.boundingBox()
    const managedClosedActions = await managedActions.boundingBox()
    await managedSummary.click()
    await expect(managed.locator('.runtime-usage-details')).toHaveAttribute('open', '')
    const managedOpenedSummary = await managedSummary.boundingBox()
    const managedOpenedActions = await managedActions.boundingBox()
    expect(Math.abs(managedOpenedSummary!.x - managedClosedSummary!.x)).toBeLessThan(1)
    expect(Math.abs(managedOpenedSummary!.y - managedClosedSummary!.y)).toBeLessThan(1)
    expect(Math.abs(managedOpenedActions!.y - managedClosedActions!.y)).toBeLessThan(1)
    await managedSummary.click()

    await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(1000, 960))
    await row.scrollIntoViewIfNeeded()
    const narrowClosed = await summary.boundingBox()
    await summary.click()
    await expect(details.locator('dl')).toBeVisible()
    const narrowOpened = await summary.boundingBox()
    expect(Math.abs(narrowOpened!.x - narrowClosed!.x)).toBeLessThan(1)
    expect(Math.abs(narrowOpened!.y - narrowClosed!.y)).toBeLessThan(1)
    expect(await row.evaluate(node => node.scrollWidth <= node.clientWidth)).toBe(true)
    await row.screenshot({ path: path.join(f.directory, 'runtime-disclosure-narrow-open.png'), animations: 'disabled' })
    expect(f.errors).toEqual([]); expect(f.unexpected).toEqual([]); expect(f.external).toEqual([])
    console.log(JSON.stringify({ screenshots: f.directory, runtimeDisclosure: 'stable entry and actions' }))
  } finally { await application.close() }
})

test('运行环境管理：一键更新、引用迁移、旧任务直接继续及分页历史', async () => {
  test.setTimeout(120_000)
  const f = await fixture()
  const { page, application, state, writes } = f
  const nav = page.getByRole('navigation', { name: '主导航' })
  try {
    expect(f.historyReads).toEqual([])
    expect(writes).toEqual([])
    await nav.getByRole('button', { name: '运行环境', exact: true }).click()
    const old = page.locator('[data-runtime-id="old"]')
    const external = page.locator('[data-runtime-id="external"]')
    await expect(external.locator('.runtime-row-title strong')).toContainText('版本未知')
    const externalActions = external.getByRole('button')
    await expect(externalActions).toHaveCount(3)
    await expect(externalActions.nth(0)).toHaveAccessibleName('刷新版本')
    await expect(externalActions.nth(1)).toHaveAccessibleName('打开路径')
    await expect(externalActions.nth(2)).toHaveAccessibleName('移除登记')
    await expect(old.getByRole('button', { name: '移除 / 卸载', exact: true })).toBeDisabled()
    await expect(old.getByText('2 个加载配置', { exact: true })).toBeVisible()

    await old.getByRole('button', { name: '切换配置', exact: true }).click()
    let modal = page.getByRole('dialog', { name: '切换 旧环境 的加载配置', exact: true })
    await modal.getByRole('combobox', { name: '目标运行环境', exact: true }).selectOption('new')
    await modal.getByRole('checkbox', { name: /配置乙/ }).uncheck()
    await modal.getByRole('button', { name: '检查切换', exact: true }).click()
    await expect(modal.getByText('将切换 1 个加载配置', { exact: true })).toBeVisible()
    await f.screenshot('switch-preview.png')
    await modal.getByRole('button', { name: '确认切换 1 个配置', exact: true }).click()
    await expect(modal).toBeHidden()
    expect(writes.find(write => write.path.endsWith('/switch-preview'))?.body).toEqual({ target_id: 'new', profile_ids: ['profile-a'] })
    expect(state.profiles[0].parameters).toEqual({ context: 4096, vision_enabled: true })
    expect(state.instances[0].runtime_id).toBe('old')

    await old.getByRole('button', { name: '更新环境', exact: true }).click()
    modal = page.getByRole('dialog', { name: '更新 llama.cpp', exact: true })
    await modal.getByRole('combobox', { name: '官方版本', exact: true }).selectOption('b2')
    await expect(modal.getByText('此版本尚未完成本机验证', { exact: true })).toBeVisible()
    await modal.getByRole('combobox', { name: '官方版本', exact: true }).selectOption('b3')
    await modal.getByRole('combobox', { name: '运行类型', exact: true }).selectOption('cpu')
    expect(writes.some(write => write.path === '/api/runtimes/install')).toBe(false)
    await expect(modal.getByRole('button', { name: '检查安装文件', exact: true })).toHaveCount(0)
    await modal.screenshot({ path: path.join(f.directory, 'one-click-update.png'), animations: 'disabled' })
    await modal.getByRole('button', { name: '开始更新', exact: true }).click()
    await expect(modal).toBeHidden()
    expect(writes.filter(write => ['/api/runtimes/plan', '/api/runtimes/install'].includes(write.path)).map(write => ({ path: write.path, body: write.body }))).toEqual([
      { path: '/api/runtimes/plan', body: { backend: 'llama_cpp', options: { version: 'b3', variant: 'cpu' }, update_from: 'old' } },
      { path: '/api/runtimes/install', body: { plan_id: 'plan-1' } },
    ])
    const newJob = page.locator('[data-job-id="new-install-plan-1"]')
    await expect(newJob.getByRole('progressbar', { name: '当前文件进度', exact: true })).toHaveAttribute('aria-valuenow', '25')
    await newJob.locator('summary').click()
    await expect(newJob.getByText('文件已知大小 1 KB · 1 个大小未知', { exact: true })).toBeVisible()
    await expect(newJob.getByText('需下载 1 个文件，大小未知', { exact: true })).toBeVisible()
    await expect(newJob.getByRole('cell', { name: '未知', exact: true })).toBeVisible()
    await expect(newJob.getByText('已缓存', { exact: true })).toBeVisible()
    await newJob.screenshot({ path: path.join(f.directory, 'installation-files.png') })
    await newJob.locator('summary').click()
    await expect(page.locator('[data-job-id="paused"]').getByText('已暂停', { exact: true })).toBeVisible()
    const awaiting = page.locator('[data-job-id="awaiting"]')
    await expect(awaiting.locator('.runtime-manifest')).toHaveCount(2)
    expect(writes.filter(write => write.path.endsWith('/confirm'))).toEqual([])
    await awaiting.getByRole('button', { name: '继续安装', exact: true }).click()
    await expect(awaiting.getByRole('button', { name: '继续安装', exact: true })).toHaveCount(0)
    await expect(page.getByRole('dialog')).toHaveCount(0)
    expect(writes.find(write => write.path === '/api/jobs/awaiting/confirm')?.body).toEqual({ stage_id: 'dependencies' })
    await expect(awaiting.locator('.runtime-manifest')).toHaveCount(2)

    await page.getByRole('tab', { name: /历史记录/ }).click()
    await expect(page.locator('.runtime-history-job')).toHaveCount(10)
    await page.getByRole('button', { name: '下一页', exact: true }).click()
    await expect(page.locator('.runtime-history-job')).toHaveCount(2)
    expect(f.historyReads).toContain(2)
    await page.getByRole('button', { name: '上一页', exact: true }).click()
    await page.locator('[data-job-id="legacy"]').getByRole('button', { name: '重试安装', exact: true }).click()
    modal = page.getByRole('dialog', { name: '安装 llama.cpp', exact: true })
    await expect(modal.getByRole('combobox', { name: '官方版本', exact: true })).toHaveValue('')
    await expect(modal.getByRole('button', { name: '开始安装', exact: true })).toBeDisabled()
    await modal.getByRole('combobox', { name: '官方版本', exact: true }).selectOption('b3')
    await modal.getByRole('combobox', { name: '运行类型', exact: true }).selectOption('cpu')
    await modal.getByRole('button', { name: '开始安装', exact: true }).click()
    await expect(modal).toBeHidden()
    expect(writes.filter(write => write.path === '/api/runtimes/plan').at(-1)?.body.retry_of).toBe('legacy')
    expect(writes.filter(write => write.path === '/api/runtimes/install').at(-1)?.body).toEqual({ plan_id: 'plan-2' })
    expect(writes.filter(write => /\/downloads\/.*\/retry$/.test(write.path))).toEqual([])
    await page.getByRole('tab', { name: /历史记录/ }).click()
    await page.locator('[data-job-id="updated"]').getByRole('button', { name: '切换配置', exact: true }).click()
    modal = page.getByRole('dialog', { name: '切换 旧环境 的加载配置', exact: true })
    await expect(modal.getByRole('combobox', { name: '目标运行环境', exact: true })).toHaveValue('new')
    await expect(modal.getByRole('checkbox', { name: /配置乙/ })).toBeChecked()
    await modal.getByRole('button', { name: '取消', exact: true }).click()
    await page.locator('[data-job-id="legacy"]').getByRole('button', { name: '删除记录', exact: true }).click()
    modal = page.getByRole('dialog', { name: '删除这条安装记录？', exact: true })
    await modal.getByRole('button', { name: '删除记录', exact: true }).click()
    await expect(modal).toBeHidden()
    await expect(page.locator('[data-job-id="legacy"]')).toHaveCount(0)
    await page.getByRole('button', { name: '清空历史', exact: true }).click()
    modal = page.getByRole('dialog', { name: '清空安装历史？', exact: true })
    await expect(modal.getByText('仅删除已结束记录，运行环境与下载文件保留。', { exact: true })).toBeVisible()
    await modal.getByRole('button', { name: '清空已结束记录', exact: true }).click()
    await expect(page.locator('.runtime-history-job')).toHaveCount(0)
    await page.getByRole('tab', { name: /未完成/ }).click()
    await expect(page.locator('[data-job-id="paused"]')).toBeVisible()

    const liveRow = page.locator('.instance-table-row').filter({ hasText: '使用旧环境的实例' })
    await liveRow.getByRole('button', { name: '停止', exact: true }).click()
    await expect(old.getByRole('button', { name: '移除 / 卸载', exact: true })).toBeEnabled()
    await old.getByRole('button', { name: '移除 / 卸载', exact: true }).click()
    modal = page.getByRole('dialog', { name: '移除 旧环境？', exact: true })
    await expect(modal.getByText('配置乙', { exact: true })).toBeVisible()
    await expect(modal.getByRole('button', { name: '迁移配置并移除登记', exact: true })).toBeDisabled()
    await expect(modal.getByRole('checkbox', { name: /同时卸载并删除环境文件/ })).not.toBeChecked()
    await modal.getByRole('checkbox', { name: /同时卸载并删除环境文件/ }).check()
    await modal.getByRole('combobox', { name: '迁移到同一后端的其他环境', exact: true }).selectOption('new')
    await f.screenshot('remove-migration.png')
    await modal.getByRole('button', { name: '迁移配置并卸载', exact: true }).click()
    await expect(old).toHaveCount(0)
    expect(writes.find(write => write.method === 'DELETE' && write.path === '/api/runtimes/old')?.query).toBe('?delete_files=true&replacement_id=new')
    expect(state.profiles[1].runtime_id).toBe('new')
    expect(state.profiles[1].parameters.context).toBe(4097)
    await external.getByRole('button', { name: '移除登记', exact: true }).click()
    modal = page.getByRole('dialog', { name: '移除登记 外部环境？', exact: true })
    await expect(modal.getByText('原始文件保留。', { exact: true })).toBeVisible()
    await modal.getByRole('combobox', { name: '迁移到同一后端的其他环境', exact: true }).selectOption('new')
    await modal.getByRole('button', { name: '迁移配置并移除登记', exact: true }).click()
    await expect(external).toHaveCount(0)
    expect(writes.find(write => write.method === 'DELETE' && write.path === '/api/runtimes/external')?.query).toBe('?replacement_id=new')
    await page.locator('[data-runtime-id="wsl"]').getByRole('button', { name: '移除 / 卸载', exact: true }).click()
    modal = page.getByRole('dialog', { name: '移除 WSL 环境？', exact: true })
    await expect(modal.getByText('仅删除此环境目录，保留共享的 ModLoader WSL 发行版及其他环境。', { exact: true })).toBeVisible()
    await modal.getByRole('button', { name: '取消', exact: true }).click()
    await page.evaluate(() => window.scrollTo(0, 0))
    await f.screenshot('runtime-management.png')
    await nav.getByRole('button', { name: '模型库', exact: true }).click()
    const reads = f.historyReads.length
    // Observe one full history polling period while the page is hidden.
    await page.waitForTimeout(3300)
    expect(f.historyReads).toHaveLength(reads)
    expect(f.errors).toEqual([]); expect(f.unexpected).toEqual([]); expect(f.external).toEqual([])
    console.log(JSON.stringify({ screenshots: f.directory, fixtureWrites: writes.length, historyPages: f.historyReads }))
  } finally { await application.close() }
})

test('依赖解析后自动准备文件并显示真实下载进度', async () => {
  const f = await fixture()
  const { page, state } = f
  const empty = { id: 'base', title: '基础文件', files: [] }
  const job: any = { id: 'resolving', backend: 'vllm', state: 'running', phase: 'resolve', downloaded_bytes: 0, total_bytes: null, total_files: 0, completed_files: 0, manifest: empty, stages: [empty], message: '正在读取依赖文件大小（63 / 199）' }
  state.installations.push(job)
  try {
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '运行环境', exact: true }).click()
    const row = page.locator('[data-job-id="resolving"]')
    await expect(row.getByText('正在解析依赖，下载大小待确定。', { exact: true })).toBeVisible()
    await expect(row.getByText('正在读取依赖文件大小（63 / 199）', { exact: true })).toBeVisible()
    await expect(row.getByRole('progressbar')).toHaveCount(0)
    await expect(row.getByText(/0 B/)).toHaveCount(0)
    await row.screenshot({ path: path.join(f.directory, 'dependency-resolving.png') })

    const mib = 1024 * 1024
    const dependencies = { id: 'deps', title: 'Python 依赖', files: [
      { id: 'vllm', name: 'vllm.whl', purpose: 'vLLM', size_bytes: 100 * mib, cached: false, status: 'pending' },
      { id: 'dependency', name: 'dependency.whl', purpose: '依赖', size_bytes: 50 * mib, cached: false, status: 'pending' },
    ] }
    Object.assign(job, { state: 'running', phase: 'prepare', message: '正在准备依赖文件', total_bytes: 150 * mib, total_files: 2, manifest: dependencies, stages: [empty, dependencies] })
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await expect(row.getByText('0 B / 150 MB', { exact: true })).toBeVisible()
    await expect(row.getByRole('button', { name: '继续安装', exact: true })).toHaveCount(0)
    await expect(page.getByRole('dialog')).toHaveCount(0)
    expect(f.writes.filter(write => write.path.endsWith('/confirm'))).toEqual([])
    await expect(row.locator('.runtime-file-table')).toBeHidden()
    await row.locator('summary').click()
    await expect(row.getByText('需下载 150 MB', { exact: true })).toBeVisible()
    await row.locator('summary').click()

    Object.assign(job, { state: 'running', phase: 'download', message: 'vllm.whl', downloaded_bytes: 20 * mib, file: 'vllm.whl', file_downloaded_bytes: 20 * mib, file_total_bytes: 100 * mib })
    dependencies.files[0].status = 'downloading'
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await expect(row.getByText('20 MB / 150 MB', { exact: true })).toBeVisible()
    Object.assign(job, { downloaded_bytes: 60 * mib, file_downloaded_bytes: 60 * mib })
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await expect(row.getByText('60 MB / 150 MB', { exact: true })).toBeVisible()
    await expect(row.getByRole('progressbar', { name: '总下载进度', exact: true })).toHaveAttribute('aria-valuenow', '40')
    await expect(row.getByRole('progressbar', { name: '当前文件进度', exact: true })).toHaveAttribute('aria-valuenow', '60')
    await row.screenshot({ path: path.join(f.directory, 'dependency-downloading.png') })

    Object.assign(dependencies.files[0], { downloaded_bytes: 60 * mib })
    Object.assign(dependencies.files[1], { status: 'downloading', downloaded_bytes: 10 * mib })
    Object.assign(job, { downloaded_bytes: 70 * mib, message: '正在并行下载 2 个文件' })
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await expect(row.getByRole('progressbar', { name: 'vllm.whl 下载进度', exact: true })).toHaveAttribute('aria-valuenow', '60')
    await expect(row.getByRole('progressbar', { name: 'dependency.whl 下载进度', exact: true })).toHaveAttribute('aria-valuenow', '20')
    await expect(row.getByText('70 MB / 150 MB', { exact: true })).toBeVisible()
    await expect(row.locator('.runtime-file-table')).toBeHidden()
    await row.screenshot({ path: path.join(f.directory, 'dependency-concurrent.png') })

    Object.assign(job, { phase: 'verify_cache', message: '正在复用依赖清单并校验本地缓存（1 / 2）' })
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await expect(row.getByText('正在校验下载缓存。', { exact: true })).toBeVisible()
    await expect(row.getByRole('progressbar')).toHaveCount(0)

    Object.assign(job, { phase: 'extract', message: '正在解压安装包', downloaded_bytes: 150 * mib, total_bytes: 150 * mib })
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await expect(row.getByText('解压中', { exact: true })).toBeVisible()
    await expect(row.getByText('正在解压安装包', { exact: true })).toBeVisible()
    await expect(row.getByRole('progressbar')).toHaveCount(0)
    await row.screenshot({ path: path.join(f.directory, 'dependency-extracting.png') })

    expect(f.errors).toEqual([])
    console.log(JSON.stringify({ screenshots: f.directory, dependencyProgress: 'passed' }))
  } finally { await f.application.close() }
})

test('一键安装忽略过期检查，重复点击、取消和关闭不会误启动', async () => {
  test.setTimeout(90_000)
  const f = await fixture()
  const { page, application } = f
  try {
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '运行环境', exact: true }).click()
    await page.locator('[data-backend-id="llama_cpp"]').getByRole('button', { name: '安装新环境', exact: true }).click()
    let modal = page.getByRole('dialog', { name: '安装 llama.cpp', exact: true })
    await modal.getByRole('combobox', { name: '官方版本', exact: true }).selectOption('b3')
    await modal.getByRole('combobox', { name: '运行类型', exact: true }).selectOption('cpu')
    const held = f.holdNextPlan()
    await modal.getByRole('button', { name: '开始安装', exact: true }).click()
    await held.started
    await expect(modal.getByRole('button', { name: '准备安装清单…', exact: true })).toBeDisabled()
    await modal.getByRole('combobox', { name: '运行类型', exact: true }).selectOption('cuda')
    f.releasePlan()
    await held.finished
    expect(f.writes.filter(write => write.path === '/api/runtimes/install')).toEqual([])
    const current = f.holdNextPlan()
    await modal.getByRole('button', { name: '开始安装', exact: true }).evaluate(element => { (element as HTMLButtonElement).click(); (element as HTMLButtonElement).click() })
    await current.started
    await expect(modal.getByRole('button', { name: '准备安装清单…', exact: true })).toBeDisabled()
    expect(f.writes.filter(write => write.path === '/api/runtimes/plan')).toHaveLength(2)
    expect(f.writes.filter(write => write.path === '/api/runtimes/plan').at(-1)?.body.options.variant).toBe('cuda')
    await application.evaluate(({ BrowserWindow }) => { const window = BrowserWindow.getAllWindows()[0]; window.setSize(1050, 780); window.webContents.setZoomFactor(1.25) })
    await page.evaluate(() => document.documentElement.style.setProperty('--font-scale', '1.3'))
    await expect.poll(() => modal.evaluate(element => element.scrollWidth <= element.clientWidth + 1)).toBe(true)
    await modal.getByRole('button', { name: '准备安装清单…', exact: true }).scrollIntoViewIfNeeded()
    await f.screenshot('one-click-install-checking-zoom.png')
    await modal.getByRole('button', { name: '取消', exact: true }).click()
    await expect(modal).toBeHidden()
    f.releasePlan(); await current.finished
    expect(f.writes.filter(write => write.path === '/api/runtimes/install')).toEqual([])

    for (const dismissal of ['关闭', 'Escape']) {
      await page.locator('[data-backend-id="llama_cpp"]').getByRole('button', { name: '安装新环境', exact: true }).click()
      modal = page.getByRole('dialog', { name: '安装 llama.cpp', exact: true })
      await modal.getByRole('combobox', { name: '官方版本', exact: true }).selectOption('b3')
      await modal.getByRole('combobox', { name: '运行类型', exact: true }).selectOption('cpu')
      const closing = f.holdNextPlan()
      await modal.getByRole('button', { name: '开始安装', exact: true }).click()
      await closing.started
      if (dismissal === 'Escape') await page.keyboard.press('Escape')
      else await modal.getByRole('button', { name: '关闭', exact: true }).click()
      await expect(modal).toBeHidden()
      f.releasePlan(); await closing.finished
      expect(f.writes.filter(write => write.path === '/api/runtimes/install')).toEqual([])
    }
    expect(f.errors).toEqual([]); expect(f.unexpected).toEqual([]); expect(f.external).toEqual([])
    console.log(JSON.stringify({ screenshots: f.directory, staleAndDismissedPlans: 'not installed', duplicateClicks: 'ignored' }))
  } finally { f.releasePlan(); await application.close() }
})

test('一键安装检查失败不启动，检查及安装错误都可重试', async () => {
  test.setTimeout(90_000)
  const f = await fixture()
  const { page } = f
  let planAttempts = 0
  const installAttempts: string[] = []
  await page.route('**/api/runtimes/plan', async route => {
    if (++planAttempts === 1) await route.fulfill({ status: 502, json: { detail: '官方版本信息暂不可用' } })
    else await route.fallback()
  })
  await page.route('**/api/runtimes/install', async route => {
    installAttempts.push(route.request().postDataJSON().plan_id)
    if (installAttempts.length === 1) await route.fulfill({ status: 409, json: { detail: '安装计划已过期，请重试' } })
    else await route.fallback()
  })
  try {
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '运行环境', exact: true }).click()
    await page.locator('[data-backend-id="llama_cpp"]').getByRole('button', { name: '安装新环境', exact: true }).click()
    const modal = page.getByRole('dialog', { name: '安装 llama.cpp', exact: true })
    await modal.getByRole('combobox', { name: '官方版本', exact: true }).selectOption('b3')
    await modal.getByRole('combobox', { name: '运行类型', exact: true }).selectOption('cpu')
    await modal.getByRole('button', { name: '开始安装', exact: true }).click()
    await expect(modal.getByRole('alert')).toContainText('官方版本信息暂不可用')
    expect(installAttempts).toEqual([])
    await expect(modal.getByRole('button', { name: '开始安装', exact: true })).toBeEnabled()
    await f.screenshot('one-click-install-plan-error.png')
    await modal.getByRole('button', { name: '重试', exact: true }).click()
    await expect(page.getByText('安装计划已过期，请重试', { exact: true })).toBeVisible()
    await expect(modal).toBeVisible()
    await expect(modal.getByRole('alert')).toHaveCount(0)
    await expect(modal.getByRole('button', { name: '开始安装', exact: true })).toBeEnabled()
    await modal.getByRole('button', { name: '开始安装', exact: true }).click()
    await expect(modal).toBeHidden()
    await expect(page.locator('[data-job-id="new-install-plan-2"]')).toBeVisible()
    expect(planAttempts).toBe(3)
    expect(installAttempts).toEqual(['plan-1', 'plan-2'])
    expect(f.errors).toEqual([]); expect(f.unexpected).toEqual([]); expect(f.external).toEqual([])
    console.log(JSON.stringify({ screenshots: f.directory, planFailure: 'not installed', retries: 'passed' }))
  } finally { await f.application.close() }
})

test('旧任务一键继续失败可重试，暂停和停止保留现有操作', async () => {
  test.setTimeout(90_000)
  const f = await fixture()
  const { page } = f
  let attempts = 0
  await page.route('**/api/jobs/awaiting/confirm', async route => {
    if (++attempts === 1) await route.fulfill({ status: 409, json: { detail: '安装任务暂不可继续' } })
    else await route.fallback()
  })
  try {
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '运行环境', exact: true }).click()
    const legacy = page.locator('[data-job-id="awaiting"]')
    await expect(legacy.getByRole('button', { name: '继续安装', exact: true })).toBeVisible()
    expect(attempts).toBe(0)
    await legacy.getByRole('button', { name: '继续安装', exact: true }).click()
    await expect(page.getByText('安装任务暂不可继续', { exact: true })).toBeVisible()
    await expect(page.getByRole('dialog')).toHaveCount(0)
    await expect(legacy.getByRole('button', { name: '继续安装', exact: true })).toBeEnabled()
    await legacy.screenshot({ path: path.join(f.directory, 'legacy-continue-install.png'), animations: 'disabled' })
    await legacy.getByRole('button', { name: '继续安装', exact: true }).click()
    await expect(legacy.getByRole('button', { name: '继续安装', exact: true })).toHaveCount(0)
    expect(attempts).toBe(2)
    expect(f.writes.find(write => write.path === '/api/jobs/awaiting/confirm')?.body).toEqual({ stage_id: 'dependencies' })
    await expect(legacy.locator('.runtime-manifest')).toHaveCount(2)

    const paused = page.locator('[data-job-id="paused"]')
    await paused.getByRole('button', { name: '继续', exact: true }).click()
    await paused.getByRole('button', { name: '暂停', exact: true }).click()
    await expect(paused.getByText('已暂停', { exact: true })).toBeVisible()
    await legacy.getByRole('button', { name: '停止', exact: true }).click()
    await expect(legacy).toHaveCount(0)
    expect(f.writes.filter(write => /\/api\/jobs\/(paused|awaiting)\//.test(write.path)).map(write => write.path)).toEqual([
      '/api/jobs/awaiting/confirm', '/api/jobs/paused/resume', '/api/jobs/paused/pause', '/api/jobs/awaiting/cancel',
    ])
    expect(f.errors).toEqual([]); expect(f.unexpected).toEqual([]); expect(f.external).toEqual([])
    console.log(JSON.stringify({ screenshots: f.directory, legacyContinue: 'one click with retry', controls: 'passed' }))
  } finally { await f.application.close() }
})

test('Strata 发行版本与引擎版本在环境卡、更新、切换及模型选择保持一致', async () => {
  test.setTimeout(90_000)
  const f = await fixture()
  const { page, state } = f
  const current: RuntimeInstallation = { id: 'strata-release', backend: 'strata', name: 'Strata · 0.1.46', version: '0.1.46', path: 'C:\\Fixture\\strata-release', managed: true, metadata: { requested_version: '0.2.1', provenance: { tag: 'v0.2.1' } } }
  const old: RuntimeInstallation = { ...current, id: 'strata-old', name: 'Strata · 0.1.44.1', version: '0.1.44.1', path: 'C:\\Fixture\\strata-old', metadata: { requested_version: '0.1.44.1' } }
  const imported: RuntimeInstallation = { ...current, id: 'strata-imported', name: 'Strata · 0.1.43', version: '0.1.43', path: 'C:\\Fixture\\strata-imported', managed: false, metadata: {} }
  const named: RuntimeInstallation = { ...current, id: 'strata-named', name: '实验环境', path: 'C:\\Fixture\\strata-named', metadata: { provenance: { tag: 'v0.2.1' } } }
  const label = 'Strata 0.2.1 · 引擎 0.1.46'
  const oldLabel = 'Strata 0.1.44.1 · 引擎 0.1.44.1'
  state.backends.push({ id: 'strata', label: 'Strata', parameters: [], description: 'Strata fixture', symbol: 'S', installable: true, import_kind: 'directory', python_option: true })
  state.runtimes.push(current, old, imported, named)
  state.models = [{ id: 'strata-model', name: 'Strata 版本显示 fixture', path: 'C:\\Fixture\\model.gguf', format: 'gguf', size_bytes: 1024, files: [], modalities: ['text'], metadata: {}, source: {}, managed: false, warnings: [] }]
  state.profiles = [{ id: 'strata-profile', model_id: 'strata-model', name: 'Strata 发行配置', backend: 'strata', runtime_id: current.id, alias: 'strata-release-model', parameters: {}, generation: {}, extra_args: [], revision: 1 }]
  state.instances = []
  await page.route('**/api/profiles/preflight', route => route.fulfill({ json: { status: 'ready', message: '', errors: [], candidates: [], parameters: [], vision_enabled: false, mtp: { enabled: false, status: 'unsupported', backend_supported: false, candidates: [] } } }))
  await page.route('**/api/resources/estimate', route => route.fulfill({ json: { resources: [], notes: [], source: 'runtime-version-fixture' } }))
  await page.route('**/api/runtimes/detect', route => route.fulfill({ json: { runtimes: [{ ...current, registered: true }], searched: [], errors: [], notes: [] } }))
  const nav = page.getByRole('navigation', { name: '主导航' })
  try {
    expect(runtimeLabel(current)).toBe(label)
    expect(runtimeLabel(old)).toBe(oldLabel)
    expect(runtimeLabel(imported)).toBe('Strata · 0.1.43')
    expect(runtimeDisplayName(named)).toBe(`实验环境 · ${label}`)
    expect(runtimeDisplayName(state.runtimes[0])).toBe('旧环境')
    expect(runtimeLabel({ ...current, backend: 'custom', name: '我的自定义引擎' })).toBe('我的自定义引擎')
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await nav.getByRole('button', { name: '运行环境', exact: true }).click()
    const row = page.locator('[data-runtime-id="strata-release"]')
    await expect(row.locator('.runtime-row-title strong')).toHaveText(label)
    await expect(page.locator('[data-runtime-id="strata-old"] .runtime-row-title strong')).toHaveText(oldLabel)
    await expect(page.locator('[data-runtime-id="strata-imported"] .runtime-row-title strong')).toHaveText('Strata · 0.1.43')
    await expect(page.locator('[data-runtime-id="strata-named"] .runtime-row-title strong')).toHaveText(`实验环境 · ${label}`)
    await row.screenshot({ path: path.join(f.directory, 'strata-release-card.png'), animations: 'disabled' })
    await row.getByRole('button', { name: '更新环境', exact: true }).click()
    let modal = page.getByRole('dialog', { name: '更新 Strata', exact: true })
    await expect(modal.locator('.runtime-source-note strong')).toHaveText(label)
    await modal.screenshot({ path: path.join(f.directory, 'strata-update-current.png'), animations: 'disabled' })
    await modal.getByRole('button', { name: '取消', exact: true }).click()

    await row.getByRole('button', { name: '切换配置', exact: true }).click()
    modal = page.getByRole('dialog', { name: `切换 ${label} 的加载配置`, exact: true })
    const targets = modal.getByRole('combobox', { name: '目标运行环境', exact: true })
    await expect(targets.locator('option[value="strata-old"]')).toHaveText(oldLabel)
    await expect(targets.locator('option[value="strata-named"]')).toHaveText(`实验环境 · ${label}`)
    await targets.selectOption(old.id)
    await modal.screenshot({ path: path.join(f.directory, 'strata-switch-selection.png'), animations: 'disabled' })
    await modal.getByRole('button', { name: '取消', exact: true }).click()

    await row.getByRole('button', { name: '移除 / 卸载', exact: true }).click()
    modal = page.getByRole('dialog', { name: `移除 ${label}？`, exact: true })
    await expect(modal.getByRole('combobox', { name: '迁移到同一后端的其他环境', exact: true }).locator('option[value="strata-old"]')).toHaveText(oldLabel)
    await modal.getByRole('button', { name: '取消', exact: true }).click()

    await page.getByRole('button', { name: '自动搜索环境', exact: true }).click()
    modal = page.getByRole('dialog', { name: '自动搜索运行环境', exact: true })
    await expect(modal.locator('.runtime-search-name strong')).toHaveText(label)
    await modal.getByRole('button', { name: '关闭', exact: true }).last().click()

    await nav.getByRole('button', { name: '模型库', exact: true }).click()
    const profile = page.locator('.profile-card').filter({ has: page.getByRole('heading', { name: 'Strata 发行配置', exact: true }) })
    await expect(profile.locator('.profile-subtitle')).toHaveText(label)
    await profile.getByRole('button', { name: '编辑配置', exact: true }).click()
    modal = page.getByRole('dialog', { name: '编辑加载配置', exact: true })
    const runtimes = modal.getByRole('combobox', { name: '运行环境', exact: true })
    await expect(runtimes).toHaveValue(current.id)
    await expect(runtimes.locator('option[value="strata-release"]')).toHaveText(label)
    await expect(runtimes.locator('option[value="strata-old"]')).toHaveText(oldLabel)
    await expect(runtimes.locator('option[value="strata-imported"]')).toHaveText('Strata · 0.1.43')
    await modal.locator('.runtime-details summary').click()
    await expect(modal.locator('.runtime-details dt').filter({ hasText: '引擎版本' })).toBeVisible()
    await modal.screenshot({ path: path.join(f.directory, 'strata-model-runtime-selection.png'), animations: 'disabled' })
    await modal.getByRole('button', { name: '取消', exact: true }).click()
    expect(current.version).toBe('0.1.46')
    expect(f.writes.filter(write => /\/api\/(jobs|runtimes)\//.test(write.path))).toEqual([])
    expect(f.errors).toEqual([]); expect(f.unexpected).toEqual([]); expect(f.external).toEqual([])
    console.log(JSON.stringify({ screenshots: f.directory, strataRelease: '0.2.1', engine: current.version, allSelections: 'consistent' }))
  } finally { await f.application.close() }
})
