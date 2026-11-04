import { test, expect, _electron as electron, type ElectronApplication, type Locator, type Page } from '@playwright/test'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { pinNavigation } from './navigation'

const root = path.resolve(__dirname, '..')
const artifacts = process.env.MODLOADER_TEST_ARTIFACTS || path.join(root, '.artifacts/codex/20261029-routing-dev/ui')
const routeKeys = ['web_search_provider', 'web_search_fallbacks', 'web_image_provider', 'web_image_fallbacks', 'web_fetch_provider', 'web_fetch_fallbacks'] as const
type Routes = Record<typeof routeKeys[number], string | string[]>
type Service = { id: string; label: string; capabilities: string[] }

async function management(page: Page, route: string, method = 'GET', body?: unknown) {
  return page.evaluate(async ({ route, method, body }) => {
    const { baseUrl, token } = await window.modloader!.connection()
    const response = await fetch(baseUrl + route, { method, headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) })
    if (!response.ok) throw new Error(`${route}: ${response.status} ${await response.text()}`)
    return response.json()
  }, { route, method, body })
}

async function settings(page: Page) { return (await management(page, '/api/state')).settings }
async function routes(page: Page): Promise<Routes> {
  const saved = await settings(page)
  return Object.fromEntries(routeKeys.map(field => [field, saved[field]])) as Routes
}
function card(page: Page) { return page.getByRole('region', { name: '联网服务', exact: true }) }
function backups(page: Page, label: string) { return card(page).getByRole('combobox', { name: new RegExp(`^${label}备用渠道 \\d+$`) }) }
function backup(page: Page, label: string, position: number) { return card(page).getByRole('combobox', { name: `${label}备用渠道 ${position}`, exact: true }) }
async function values(selects: Locator) { return selects.evaluateAll(nodes => nodes.map(node => (node as HTMLSelectElement).value)) }
async function options(select: Locator) { return select.locator('option').evaluateAll(nodes => nodes.map(node => (node as HTMLOptionElement).value)) }
async function add(page: Page, label: string) { await card(page).getByRole('button', { name: `添加${label}备用渠道`, exact: true }).click() }
async function navigate(page: Page, name: string) { await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name, exact: true }).click() }
async function menu(application: ElectronApplication, label: string, item: string) {
  await application.evaluate(({ Menu }, { label, item }) => {
    Menu.getApplicationMenu()!.items.find(entry => entry.label === label)!.submenu!.items.find(entry => entry.label === item)!.click()
  }, { label, item })
}
async function visible(application: ElectronApplication) { return application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].isVisible()) }
async function launch(data: string, errors: string[]) {
  const executablePath = process.env.MODLOADER_TEST_EXECUTABLE
  const application = await electron.launch({ executablePath, args: executablePath ? [] : [root], env: { ...process.env, MODLOADER_DATA_DIR: data, MODLOADER_DEV_URL: executablePath ? '' : process.env.MODLOADER_DEV_URL || '', MODLOADER_SENSOR_AUTO_START: '0' }, timeout: 60_000 })
  try {
    if (executablePath) expect(await application.evaluate(({ app }) => app.isPackaged)).toBe(true)
    const page = await application.firstWindow()
    page.on('pageerror', error => errors.push(error.message))
    await expect(page.getByRole('status', { name: '服务在线', exact: true })).toBeVisible({ timeout: 45_000 })
    await pinNavigation(page)
    return { application, page }
  } catch (error) { await application.close(); throw error }
}

test('路由草稿：多空行、离页提交、托盘隐藏和真正关闭后的同数据恢复', async ({}, testInfo) => {
  test.setTimeout(150_000)
  fs.mkdirSync(artifacts, { recursive: true })
  const directory = fs.mkdtempSync(path.join(artifacts, 'routing-lifecycle-'))
  const data = path.join(directory, 'data'), errors: string[] = []
  const patches: Partial<Routes>[] = []
  let application: ElectronApplication | undefined, page: Page | undefined
  let holdNext = false, releasePatch: (() => void) | undefined, beforeCloseCalls = 0
  try {
    ;({ application, page } = await launch(data, errors))
    const current = page
    const initial = await routes(current), services: Service[] = (await settings(current)).web_services
    await current.route('**/api/settings', async route => {
      if (route.request().method() !== 'PATCH') { await route.continue(); return }
      const body = route.request().postDataJSON()
      if (routeKeys.some(field => Object.hasOwn(body, field))) {
        patches.push(body)
        if (holdNext) { holdNext = false; await new Promise<void>(resolve => { releasePatch = resolve }) }
      }
      await route.continue()
    })
    await current.exposeFunction('recordRoutingBeforeClose', () => { beforeCloseCalls++ })
    await current.evaluate(() => {
      window.modloader!.onBeforeClose(async () => { await (window as any).recordRoutingBeforeClose() })
    })
    await navigate(current, '设置')
    await expect(card(current).locator('details')).toHaveCount(0)
    await expect(card(current).getByText('首选', { exact: true })).toHaveCount(0)
    await expect(card(current).locator('.web-fallback-position')).toHaveCount(0)
    await expect(card(current).getByRole('button', { name: '保存', exact: true })).toHaveCount(0)
    for (const [label, capability] of [['网页搜索', 'search'], ['关键词搜图', 'images'], ['网页读取', 'fetch']]) {
      await expect(card(current).getByRole('combobox', { name: label, exact: true })).toBeVisible()
      expect(await options(card(current).getByRole('combobox', { name: label, exact: true }))).toEqual(services.filter(service => service.capabilities.includes(capability)).map(service => service.id))
      await expect(card(current).getByRole('button', { name: `添加${label}备用渠道`, exact: true })).toBeEnabled()
    }
    const searchProviders = services.filter(service => service.capabilities.includes('search')).map(service => service.id)
    const order = searchProviders.filter(id => id !== initial.web_search_provider).reverse()
    expect(order.length).toBeGreaterThan(1)
    const rowCount = order.length * 2 + 2
    expect(rowCount).toBeGreaterThan(searchProviders.length)
    for (let index = 0; index < rowCount; index++) await add(current, '网页搜索')
    await expect(backups(current, '网页搜索')).toHaveCount(rowCount)
    expect(await values(backups(current, '网页搜索'))).toEqual(Array(rowCount).fill(''))
    for (let index = 0; index < order.length; index++) await backup(current, '网页搜索', index * 2 + 2).selectOption(order[index])
    const draft = Array.from({ length: rowCount }, (_, index) => index % 2 === 1 && index / 2 < order.length ? order[(index - 1) / 2] : '')
    expect(await values(backups(current, '网页搜索'))).toEqual(draft)
    await expect(backup(current, '网页搜索', 1).locator('option[value=""]')).toHaveText('选择渠道')
    expect(await options(backup(current, '网页搜索', 1))).toEqual([''])
    for (let index = 0; index < order.length; index++) expect(await options(backup(current, '网页搜索', index * 2 + 2))).toEqual(['', order[index]])
    await backup(current, '网页搜索', 2).selectOption('')
    await backup(current, '网页搜索', 2).selectOption(order[0])
    const firstName = services.find(service => service.id === order[0])!.label
    await card(current).getByRole('button', { name: `下移网页搜索备用渠道 ${firstName}`, exact: true }).click()
    await expect(backup(current, '网页搜索', 3)).toHaveValue(order[0])
    await card(current).getByRole('button', { name: `上移网页搜索备用渠道 ${firstName}`, exact: true }).click()
    expect(await values(backups(current, '网页搜索'))).toEqual(draft)
    await add(current, '网页搜索')
    await expect(backups(current, '网页搜索')).toHaveCount(rowCount + 1)
    await card(current).getByRole('button', { name: `删除网页搜索备用渠道 空渠道 ${rowCount + 1}`, exact: true }).click()
    await card(current).getByRole('tab', { name: 'Tavily', exact: true }).click()
    await card(current).getByRole('tab', { name: 'DeepSeek', exact: true }).click()
    expect(await values(backups(current, '网页搜索'))).toEqual(draft)
    expect(await routes(current)).toEqual(initial)
    expect(patches).toHaveLength(0)
    await current.screenshot({ path: path.join(directory, 'routing-many-empty-rows.png') })

    // Leaving starts one PATCH; re-entering during it cannot edit its snapshot.
    holdNext = true
    await navigate(current, '模型库')
    await expect.poll(() => Boolean(releasePatch)).toBe(true)
    await navigate(current, '设置')
    await expect(card(current).getByRole('group', { name: '高级路由', exact: true })).toHaveAttribute('disabled', '')
    await expect(card(current).getByRole('combobox', { name: '网页搜索', exact: true })).toBeDisabled()
    await expect(card(current).getByRole('button', { name: '添加网页搜索备用渠道', exact: true })).toBeDisabled()
    expect(patches).toHaveLength(1)
    expect(patches[0].web_search_fallbacks).toEqual(order)
    expect(await routes(current)).toEqual(initial)
    releasePatch!(); releasePatch = undefined
    const afterLeave = { ...initial, web_search_fallbacks: order }
    await expect.poll(() => routes(current)).toEqual(afterLeave)
    await expect(backups(current, '网页搜索')).toHaveCount(order.length)
    expect(await values(backups(current, '网页搜索'))).toEqual(order)

    // A real close-to-tray event saves routes without running close handlers.
    await card(current).getByRole('combobox', { name: '关键词搜图', exact: true }).selectOption('openai')
    await add(current, '关键词搜图'); await add(current, '关键词搜图')
    await backup(current, '关键词搜图', 2).selectOption('tavily')
    await card(current).getByRole('combobox', { name: '网页读取', exact: true }).selectOption('deepseek')
    await add(current, '网页读取'); await add(current, '网页读取')
    await backup(current, '网页读取', 1).selectOption('openai')
    expect(patches).toHaveLength(1)
    await application.evaluate(({ BrowserWindow }) => { BrowserWindow.getAllWindows()[0].close() })
    await expect.poll(() => visible(application!)).toBe(false)
    const afterHide = { ...afterLeave, web_image_provider: 'openai', web_image_fallbacks: ['tavily'], web_fetch_provider: 'deepseek', web_fetch_fallbacks: ['openai'] }
    await expect.poll(() => routes(current)).toEqual(afterHide)
    expect(patches).toHaveLength(2)
    expect(beforeCloseCalls).toBe(0)
    await application.evaluate(({ app }) => { app.emit('second-instance', {}, [], '') })
    await expect.poll(() => visible(application!)).toBe(true)
    await expect(backups(current, '关键词搜图')).toHaveCount(1)
    await expect(backup(current, '关键词搜图', 1)).toHaveValue('tavily')
    await expect(backups(current, '网页读取')).toHaveCount(1)

    // A real exit arriving during leave-save shares the same pending PATCH.
    await management(current, '/api/settings', 'PATCH', { close_to_tray: false })
    await card(current).getByRole('combobox', { name: '网页搜索', exact: true }).selectOption(order[0])
    await expect(backups(current, '网页搜索')).toHaveCount(order.length - 1)
    expect(await values(backups(current, '网页搜索'))).toEqual(order.slice(1))
    await add(current, '网页搜索')
    const finalRoutes = { ...afterHide, web_search_provider: order[0], web_search_fallbacks: order.slice(1) }
    holdNext = true
    await navigate(current, '模型库')
    await expect.poll(() => Boolean(releasePatch)).toBe(true)
    const closed = application.waitForEvent('close')
    await application.evaluate(({ BrowserWindow }) => { BrowserWindow.getAllWindows()[0].close() })
    await expect.poll(() => beforeCloseCalls).toBe(1)
    expect(patches).toHaveLength(3)
    expect(patches[2]).toMatchObject({ web_search_provider: order[0], web_search_fallbacks: order.slice(1) })
    releasePatch!(); releasePatch = undefined
    await closed
    application = undefined

    ;({ application, page } = await launch(data, errors))
    await navigate(page, '设置')
    expect(await routes(page)).toEqual(finalRoutes)
    await expect(card(page).getByRole('combobox', { name: '网页搜索', exact: true })).toHaveValue(order[0])
    expect(await values(backups(page, '网页搜索'))).toEqual(order.slice(1))
    expect(await values(backups(page, '关键词搜图'))).toEqual(['tavily'])
    expect(await values(backups(page, '网页读取'))).toEqual(['openai'])
    await card(page).getByRole('group', { name: '高级路由', exact: true }).screenshot({ path: path.join(directory, 'routing-restored-after-exit.png') })
    for (const name of ['routing-many-empty-rows.png', 'routing-restored-after-exit.png']) await testInfo.attach(name, { path: path.join(directory, name), contentType: 'image/png' })
    expect(errors).toEqual([])
  } finally {
    releasePatch?.()
    if (testInfo.status !== testInfo.expectedStatus && page && !page.isClosed()) await page.screenshot({ path: path.join(directory, 'failure.png') }).catch(() => undefined)
    await application?.close()
  }
})

test('路由保存失败保留空行和可重试草稿，退出取消与受控重载使用同一保存路径', async ({}, testInfo) => {
  test.setTimeout(120_000)
  fs.mkdirSync(artifacts, { recursive: true })
  const directory = fs.mkdtempSync(path.join(artifacts, 'routing-recovery-'))
  const errors: string[] = [], patches: Partial<Routes>[] = []
  const rejected = 'fixture: 路由保存被拒绝'
  let application: ElectronApplication | undefined, page: Page | undefined, rejectNext = true
  try {
    ;({ application, page } = await launch(path.join(directory, 'data'), errors))
    const current = page, initial = await routes(current)
    await current.route('**/api/settings', async route => {
      if (route.request().method() === 'PATCH' && routeKeys.some(field => Object.hasOwn(route.request().postDataJSON(), field))) {
        patches.push(route.request().postDataJSON())
        if (rejectNext) { rejectNext = false; await route.fulfill({ status: 503, json: { detail: rejected } }); return }
      }
      await route.continue()
    })
    await navigate(current, '设置')
    await card(current).getByRole('combobox', { name: '网页搜索', exact: true }).selectOption('openai')
    await add(current, '网页搜索'); await add(current, '网页搜索'); await add(current, '网页搜索')
    await backup(current, '网页搜索', 2).selectOption('tavily')
    await navigate(current, '模型库')
    await expect(current.getByText('路由保存失败，请返回设置重试。', { exact: true })).toBeVisible()
    await navigate(current, '设置')
    await expect(card(current).getByRole('combobox', { name: '网页搜索', exact: true })).toHaveValue('openai')
    expect(await values(backups(current, '网页搜索'))).toEqual(['', 'tavily', ''])
    expect(await routes(current)).toEqual(initial)
    expect(patches).toHaveLength(1)
    await current.locator('.error-notice').filter({ hasText: rejected }).getByRole('button', { name: '重试', exact: true }).click()
    const recovered = { ...initial, web_search_provider: 'openai', web_search_fallbacks: ['tavily'] }
    await expect.poll(() => routes(current)).toEqual(recovered)
    await expect(backups(current, '网页搜索')).toHaveCount(1)
    expect(patches).toHaveLength(2)

    await management(current, '/api/settings', 'PATCH', { close_to_tray: false })
    await application.evaluate(({ dialog }) => {
      ;(globalThis as any).routingCloseWarnings = []
      dialog.showMessageBox = (async (...args: any[]) => {
        ;(globalThis as any).routingCloseWarnings.push(args.at(-1))
        return { response: 0, checkboxChecked: false }
      }) as typeof dialog.showMessageBox
    })
    await card(current).getByRole('combobox', { name: '网页搜索', exact: true }).selectOption('deepseek')
    await add(current, '网页搜索')
    rejectNext = true
    await application.evaluate(({ BrowserWindow }) => { BrowserWindow.getAllWindows()[0].close() })
    await expect.poll(() => application!.evaluate(() => (globalThis as any).routingCloseWarnings.length)).toBe(1)
    const warnings = await application.evaluate(() => (globalThis as any).routingCloseWarnings)
    expect(warnings[0]).toMatchObject({ title: '退出 ModLoader', message: '更改尚未保存', buttons: ['取消退出', '仍然退出'], defaultId: 0, cancelId: 0 })
    expect(warnings[0].detail).toContain('可能丢失未保存的设置或聊天记录')
    expect(await visible(application)).toBe(true)
    await expect(card(current).getByRole('combobox', { name: '网页搜索', exact: true })).toBeEnabled()
    await expect(card(current).getByRole('combobox', { name: '网页搜索', exact: true })).toHaveValue('deepseek')
    expect(await values(backups(current, '网页搜索'))).toEqual(['tavily', ''])
    expect(await routes(current)).toEqual(recovered)
    expect(patches).toHaveLength(3)
    await current.locator('.error-notice').filter({ hasText: rejected }).getByRole('button', { name: '重试', exact: true }).click()
    await expect.poll(() => routes(current)).toEqual({ ...recovered, web_search_provider: 'deepseek' })
    await expect(backups(current, '网页搜索')).toHaveCount(1)
    expect(patches).toHaveLength(4)

    await card(current).getByRole('combobox', { name: '网页搜索', exact: true }).selectOption('tavily')
    await expect(backups(current, '网页搜索')).toHaveCount(0)
    await add(current, '网页搜索'); await add(current, '网页搜索')
    await backup(current, '网页搜索', 2).selectOption('openai')
    const reloaded = current.waitForEvent('load')
    await menu(application, '视图', '重新加载界面')
    await reloaded
    await expect(current.getByRole('status', { name: '服务在线', exact: true })).toBeVisible()
    await pinNavigation(current); await navigate(current, '设置')
    expect(await routes(current)).toEqual({ ...initial, web_search_provider: 'tavily', web_search_fallbacks: ['openai'] })
    await expect(card(current).getByRole('combobox', { name: '网页搜索', exact: true })).toHaveValue('tavily')
    expect(await values(backups(current, '网页搜索'))).toEqual(['openai'])
    expect(patches).toHaveLength(5)
    expect(await application.evaluate(() => (globalThis as any).routingCloseWarnings.length)).toBe(1)
    await current.screenshot({ path: path.join(directory, 'routing-recovered.png') })
    await testInfo.attach('routing-recovered', { path: path.join(directory, 'routing-recovered.png'), contentType: 'image/png' })
    expect(errors).toEqual([])
  } finally {
    rejectNext = false
    if (testInfo.status !== testInfo.expectedStatus && page && !page.isClosed()) await page.screenshot({ path: path.join(directory, 'failure.png') }).catch(() => undefined)
    await application?.close()
  }
})
