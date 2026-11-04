import { pinNavigation } from './navigation'
import { test, expect, _electron as electron } from '@playwright/test'
import * as path from 'node:path'
import * as fs from 'node:fs'
import type { AppState, ModelAsset } from '../src/types'

test('真实 Electron 批量移除：滑动与逐项多选、取消高亮、部分失败及仅重试失败登记', async () => {
  test.setTimeout(120_000)
  const root = path.resolve(__dirname, '..')
  const artifacts = process.env.MODLOADER_TEST_ARTIFACT_DIR || path.join(root, 'work')
  fs.mkdirSync(artifacts, { recursive: true })
  const directory = fs.mkdtempSync(path.join(artifacts, 'ui-bulk-remove-'))
  const screenshots = path.join(directory, 'screenshots')
  fs.mkdirSync(screenshots)
  const header = Buffer.alloc(24)
  header.write('GGUF'); header.writeUInt32LE(3, 4); header.writeBigUInt64LE(1n, 8)
  // These are valid inspection fixtures, without inference weights.
  const definitions = [
    { key: 'beta2', name: '批量样例 Beta2' },
    { key: 'alpha2', name: '批量样例 Alpha2' },
    { key: 'active', name: '批量样例 Alpha3 运行状态界面样例' },
    { key: 'beta1', name: '批量样例 Beta1' },
    { key: 'alpha1', name: '批量样例 Alpha1' },
  ]
  const files = definitions.map(definition => {
    const file = path.join(directory, `${definition.key}.gguf`)
    fs.writeFileSync(file, header)
    return file
  })
  // Strata import validates these paths statically. Nothing in this directory is executed.
  const runtimeDirectory = path.join(directory, 'static-runtime-fixture')
  fs.mkdirSync(path.join(runtimeDirectory, 'serve'), { recursive: true })
  fs.mkdirSync(path.join(runtimeDirectory, 'engine'))
  fs.writeFileSync(path.join(runtimeDirectory, 'serve', 'server.py'), '# Static registration fixture; never executed.\n')
  fs.writeFileSync(path.join(runtimeDirectory, 'engine', 'strata.exe'), 'Static registration fixture; never executed.')
  const application = await electron.launch({
    ...(process.env.MODLOADER_TEST_EXECUTABLE || process.env.MODLOADER_TEST_PACKAGED ? { executablePath: process.env.MODLOADER_TEST_EXECUTABLE || path.join(root, 'release', 'win-unpacked', 'ModLoader.exe'), args: [] } : { args: [root] }),
    env: { ...process.env, MODLOADER_DATA_DIR: path.join(directory, 'data') }, timeout: 60_000,
  })
  const page = await application.firstWindow()
  const errors: string[] = []
  const externalRequests: string[] = []
  const forbiddenRequests: string[] = []
  const deleteRequests: Array<{ id: string; query: string }> = []
  const models: Record<string, ModelAsset> = {}
  let setup = true
  let displayRunning = false
  let fontScale = 1
  let releaseConflict: (() => void) | undefined
  page.on('pageerror', error => errors.push(error.message))
  page.on('request', request => {
    const url = new URL(request.url())
    if (request.method() === 'DELETE' && /^\/api\/models\/[^/]+$/.test(url.pathname)) deleteRequests.push({ id: url.pathname.split('/').at(-1)!, query: url.search })
  })
  await page.route('**/*', async route => {
    const request = route.request()
    const url = new URL(request.url())
    if (['http:', 'https:'].includes(url.protocol) && !['127.0.0.1', 'localhost'].includes(url.hostname)) {
      externalRequests.push(url.origin)
      await route.abort()
    } else if (!url.pathname.startsWith('/api/') || ['GET', 'HEAD', 'OPTIONS'].includes(request.method())) {
      await route.continue()
    } else if (setup && request.method() === 'POST' && ['/api/models/import', '/api/runtimes/import', '/api/profiles'].includes(url.pathname)) {
      await route.continue()
    } else if (request.method() === 'DELETE' && /^\/api\/models\/[^/]+$/.test(url.pathname) && !url.search) {
      await route.continue()
    } else {
      forbiddenRequests.push(`${request.method()} ${url.pathname}${url.search}`)
      await route.abort()
    }
  })
  async function api(method: string, route: string, body?: unknown) {
    return page.evaluate(async ({ method, route, body }) => {
      const { baseUrl, token } = await window.modloader!.connection()
      const response = await fetch(baseUrl + route, {
        method, headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
        body: body === undefined ? undefined : JSON.stringify(body),
      })
      return { status: response.status, data: await response.json() }
    }, { method, route, body })
  }
  const list = page.getByRole('region', { name: '模型列表', exact: true })
  const search = page.getByRole('textbox', { name: '搜索本地模型', exact: true })
  const sort = page.getByRole('combobox', { name: '模型排序', exact: true })
  const all = page.getByRole('checkbox', { name: '全选当前结果', exact: true })
  const dialog = page.getByRole('dialog', { name: '批量移除模型登记', exact: true })
  const item = (key: string) => list.locator(`.model-item[data-model-id="${models[key].id}"]`)
  const check = (key: string) => list.getByRole('checkbox', { name: `勾选 ${models[key].name}`, exact: true })
  async function expectChecked(keys: string[]) {
    const expected = keys.map(key => `勾选 ${models[key].name}`).sort()
    await expect.poll(() => list.getByRole('checkbox', { checked: true }).evaluateAll(elements => elements.map(element => element.getAttribute('aria-label')).sort())).toEqual(expected)
    await expect(page.locator('.library-bottom')).toContainText(`已选 ${keys.length} 个`)
  }
  async function expectSelectionAppearance(keys: string[]) {
    const expected = keys.map(key => models[key].id).sort()
    await expect.poll(() => list.locator('.model-item.selected').evaluateAll(elements => elements.map(element => element.getAttribute('data-model-id')).sort())).toEqual(expected)
    await expect.poll(() => list.locator('.model-item[aria-pressed="true"]').evaluateAll(elements => elements.map(element => element.getAttribute('data-model-id')).sort())).toEqual(expected)
  }
  async function refresh() { await page.getByRole('button', { name: '刷新状态', exact: true }).click() }
  async function nativeScreenshot(name: string) {
    await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))))
    const png = await application.evaluate(async ({ BrowserWindow }) => (await BrowserWindow.getAllWindows()[0].webContents.capturePage()).toPNG().toString('base64'))
    fs.writeFileSync(path.join(screenshots, name), Buffer.from(png, 'base64'))
  }
  async function state() {
    const response = await api('GET', '/api/state')
    expect(response.status).toBe(200)
    return response.data as AppState
  }
  function expectFilesPreserved() {
    for (const file of files) expect(fs.readFileSync(file)).toEqual(header)
  }

  try {
    await pinNavigation(page)
    await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
    await application.evaluate(({ BrowserWindow }) => { const window = BrowserWindow.getAllWindows()[0]; window.setSize(1440, 960); window.webContents.setZoomFactor(1) })
    for (const [index, definition] of definitions.entries()) {
      const imported = await api('POST', '/api/models/import', { path: files[index], name: definition.name })
      expect(imported.status).toBe(200)
      models[definition.key] = imported.data
    }
    const runtime = await api('POST', '/api/runtimes/import', { backend: 'strata', path: runtimeDirectory, metadata: { python: path.join(root, '.venv', 'Scripts', 'python.exe'), version: '静态界面样例' } })
    expect(runtime.status).toBe(200)
    const profileName = '批量移除关联配置'
    const profile = await api('POST', '/api/profiles', { model_id: models.alpha1.id, name: profileName, backend: 'strata', runtime_id: runtime.data.id, alias: 'bulk-remove-associated-profile' })
    expect(profile.status).toBe(200)
    setup = false
    const initial = await state()
    expect(initial.models).toHaveLength(5)
    expect(initial.profiles).toHaveLength(1)
    expect(initial.instances).toHaveLength(0)
    // Only active status and font scale are renderer fixtures; model/profile data stays real.
    await page.route('**/api/state', async route => {
      const response = await route.fetch()
      const current = await response.json()
      const instances = displayRunning ? [...current.instances, { id: 'bulk-active-ui-fixture', model_id: models.active.id, alias: '运行状态界面样例（未启动模型）', backend: 'strata', profile_id: 'ui-only-profile', profile_revision: 1, runtime_id: runtime.data.id, state: 'ready', port: 0, started_at: '', active_requests: 0, capabilities: {} }] : current.instances
      await route.fulfill({ response, json: { ...current, instances, settings: { ...current.settings, font_scale: fontScale } } })
    })
    await refresh()
    await expect(list.locator('.model-item')).toHaveCount(5)
    await page.evaluate(() => document.fonts.ready)
    console.log(JSON.stringify({ electron: await application.evaluate(() => process.versions.electron), realImportedModels: 5, realProfiles: 1, screenshots }))

    await test.step('默认无勾选框，左滑进入逐项多选并跳过运行状态界面样例', async () => {
      await expect(list.getByRole('checkbox')).toHaveCount(0)
      await expect(all).toHaveCount(0)
      displayRunning = true
      await refresh()
      const box = await item('alpha1').boundingBox()
      expect(box).not.toBeNull()
      await page.mouse.move(box!.x + box!.width - 25, box!.y + box!.height / 2)
      await page.mouse.down()
      await page.mouse.move(box!.x + 25, box!.y + box!.height / 2, { steps: 8 })
      await page.mouse.up()
      await expect(check('alpha1')).toBeChecked()
      await expect(check('active')).toBeDisabled()
      await expectChecked(['alpha1'])
      await expectSelectionAppearance(['alpha1'])

      await item('beta1').click()
      await expectChecked(['alpha1', 'beta1'])
      await item('alpha1').click()
      await expectChecked(['beta1'])
      await expectSelectionAppearance(['beta1'])
      await expect(item('alpha1')).not.toHaveClass(/selected/)
      await expect(item('alpha1')).not.toHaveAttribute('aria-pressed', 'true')
      await nativeScreenshot('bulk-selection-cancelled-item.png')

      await item('alpha1').click()
      await expectChecked(['alpha1', 'beta1'])
      await expectSelectionAppearance(['alpha1', 'beta1'])
      await item('beta1').click({ modifiers: ['Control'] })
      await expectChecked(['alpha1'])
      await expectSelectionAppearance(['alpha1'])
      await item('beta2').click({ modifiers: ['Shift'] })
      await expectChecked(['alpha1', 'beta2'])
      await expectSelectionAppearance(['alpha1', 'beta2'])
      await item('beta1').click({ modifiers: ['Control', 'Shift'] })
      await expectChecked(['alpha1', 'beta1', 'beta2'])
      await expectSelectionAppearance(['alpha1', 'beta1', 'beta2'])

      await check('alpha2').check()
      await expectChecked(['alpha1', 'alpha2', 'beta1', 'beta2'])
      await expectSelectionAppearance(['alpha1', 'alpha2', 'beta1', 'beta2'])
      await check('beta1').uncheck()
      await expectChecked(['alpha1', 'alpha2', 'beta2'])
      await expectSelectionAppearance(['alpha1', 'alpha2', 'beta2'])

      await all.check()
      await expectChecked(['alpha1', 'alpha2', 'beta1', 'beta2'])
      await expectSelectionAppearance(['alpha1', 'alpha2', 'beta1', 'beta2'])
      await all.uncheck()
      await expectChecked([])
      await expectSelectionAppearance([])

      await all.check()
      await expectChecked(['alpha1', 'alpha2', 'beta1', 'beta2'])
      await expectSelectionAppearance(['alpha1', 'alpha2', 'beta1', 'beta2'])
      await search.fill('Alpha')
      await item('alpha2').click()
      await expectSelectionAppearance(['alpha1'])
      await sort.selectOption('name-desc')
      await item('alpha1').click({ modifiers: ['Shift'] })
      await expectSelectionAppearance([])
      await search.fill('')
      await expectChecked(['beta1', 'beta2'])
      await expectSelectionAppearance(['beta1', 'beta2'])
      await all.check()
      await expectChecked(['alpha1', 'alpha2', 'beta1', 'beta2'])
      await expectSelectionAppearance(['alpha1', 'alpha2', 'beta1', 'beta2'])
      await all.uncheck()
      await expectChecked([])
      await expectSelectionAppearance([])

      await page.getByRole('button', { name: '完成选择', exact: true }).click()
      await expect(list.getByRole('checkbox')).toHaveCount(0)
      await expect(all).toHaveCount(0)
      const enter = page.getByRole('button', { name: '多选', exact: true })
      await enter.focus()
      await page.keyboard.press('Space')
      await expectChecked([])
      await expectSelectionAppearance([])
      await page.getByRole('button', { name: '完成选择', exact: true }).click()
      await expect(list.getByRole('checkbox')).toHaveCount(0)
      displayRunning = false
      await refresh()
      await sort.selectOption('name-asc')
    })

    await test.step('已勾选项变为运行状态的 UI fixture 会阻止确认，并允许取消勾选', async () => {
      await page.getByRole('button', { name: '多选', exact: true }).click()
      await check('active').check()
      displayRunning = true
      await refresh()
      await expect(check('active')).toBeEnabled()
      await expect(check('active')).toBeChecked()
      await page.getByRole('button', { name: '移除登记', exact: true }).click()
      await expect(dialog.getByRole('button', { name: '确认移除', exact: true })).toBeDisabled()
      await expect(dialog).toContainText('请先停止')
      await dialog.getByRole('button', { name: '取消', exact: true }).click()
      await check('active').uncheck()
      await expectChecked([])
      await page.getByRole('button', { name: '完成选择', exact: true }).click()
      displayRunning = false
      await refresh()
      expect(deleteRequests).toEqual([])
    })

    await test.step('全选仅作用于当前结果，跨搜索及排序保留勾选，确认取消不写入', async () => {
      await page.getByRole('button', { name: '多选', exact: true }).click()
      await search.fill('Alpha1')
      await all.check()
      await search.fill('Alpha2')
      await all.check()
      await search.fill('Beta')
      await check('beta1').check()
      await sort.selectOption('name-desc')
      await expect(check('beta1')).toBeChecked()
      await expect(check('beta2')).not.toBeChecked()
      await search.fill('')
      await expectChecked(['alpha1', 'alpha2', 'beta1'])
      await page.getByRole('button', { name: '移除登记', exact: true }).click()
      await expect(dialog.locator('.bulk-remove-row')).toHaveCount(3)
      await expect(dialog.getByText('3 个模型', { exact: true })).toBeVisible()
      await expect(dialog.getByText('1 个加载配置', { exact: true })).toBeVisible()
      for (const key of ['alpha1', 'alpha2', 'beta1']) {
        const row = dialog.locator('.bulk-remove-row').filter({ hasText: models[key].name })
        await expect(row.getByText(models[key].name, { exact: true })).toBeVisible()
        await expect(row.getByText(models[key].path, { exact: true })).toBeVisible()
      }
      await expect(dialog.locator('.bulk-remove-row').filter({ hasText: models.alpha1.name })).toContainText(profileName)
      await nativeScreenshot('bulk-confirm-1440x960.png')
      await dialog.getByRole('button', { name: '取消', exact: true }).click()
      await expect(dialog).toBeHidden()
      await expectChecked(['alpha1', 'alpha2', 'beta1'])
      await expect(list.getByRole('checkbox')).toHaveCount(5)
      await expect(page.getByRole('button', { name: '完成选择', exact: true })).toBeVisible()
      const afterCancel = await state()
      expect(afterCancel.models.map(model => model.id).sort()).toEqual(initial.models.map(model => model.id).sort())
      expect(afterCancel.profiles).toEqual(initial.profiles)
      expect(deleteRequests).toEqual([])
      expectFilesPreserved()
    })

    await test.step('真实核心移除选中登记与关联配置，mock 409 后仅重试失败项，所有文件保留', async () => {
      fontScale = 1.3
      await application.evaluate(({ BrowserWindow }) => { const window = BrowserWindow.getAllWindows()[0]; window.setSize(1050, 780); window.webContents.setZoomFactor(1.25) })
      await refresh()
      await expect.poll(() => page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue('--font-scale').trim())).toBe('1.3')
      await page.getByRole('button', { name: '移除登记', exact: true }).click()
      await expect(dialog.getByRole('button', { name: '确认移除', exact: true })).toBeEnabled()
      expect(await dialog.evaluate(element => element.scrollWidth <= element.clientWidth + 1)).toBe(true)
      await nativeScreenshot('bulk-confirm-1050x780.png')
      let conflictPending = false
      let conflictOnce = true
      const delay = new Promise<void>(resolve => { releaseConflict = resolve })
      await page.route(`**/api/models/${models.alpha2.id}`, async route => {
        if (route.request().method() === 'DELETE' && conflictOnce) {
          conflictOnce = false; conflictPending = true
          await delay
          await route.fulfill({ status: 409, json: { detail: '界面测试 mock 409：模型在确认后变为运行状态' } })
        } else await route.fallback()
      })
      await dialog.getByRole('button', { name: '确认移除', exact: true }).click()
      await expect.poll(() => conflictPending, { timeout: 10_000 }).toBe(true)
      await expect(dialog.getByRole('button', { name: /^正在移除 / })).toBeDisabled()
      await expect(dialog.getByRole('button', { name: '取消', exact: true })).toBeDisabled()
      await expect(dialog.getByRole('button', { name: '关闭', exact: true })).toBeDisabled()
      await page.keyboard.press('Escape')
      await expect(dialog).toBeVisible()
      await page.locator('.modal-overlay').click({ position: { x: 3, y: 3 } })
      await expect(dialog).toBeVisible()
      releaseConflict!()
      await expect(dialog.getByText('已移除 2 个', { exact: true })).toBeVisible()
      await expect(dialog.getByText('失败 1 个', { exact: true })).toBeVisible()
      await expect(dialog.getByRole('alert')).toHaveText('界面测试 mock 409：模型在确认后变为运行状态')
      await expect(page.locator('.library-bottom')).toContainText('已选 1 个')
      await expect(list.getByRole('checkbox')).toHaveCount(3)
      const partial = await state()
      expect(partial.models.map(model => model.id).sort()).toEqual([models.alpha2.id, models.beta2.id, models.active.id].sort())
      expect(partial.profiles).toHaveLength(0)
      expectFilesPreserved()
      expect(deleteRequests.map(request => request.id).sort()).toEqual([models.alpha1.id, models.alpha2.id, models.beta1.id].sort())
      await nativeScreenshot('bulk-partial-failure.png')
      await dialog.getByRole('button', { name: '重试失败项', exact: true }).click()
      await expect(dialog.getByText('已移除 3 个', { exact: true })).toBeVisible()
      await expect(dialog.getByText('失败 1 个', { exact: true })).toBeHidden()
      await expect(dialog).toBeVisible()
      await expect(list.getByRole('checkbox')).toHaveCount(0)
      await expect(page.getByRole('button', { name: '多选', exact: true })).toBeVisible()
      expect(deleteRequests.map(request => request.id).sort()).toEqual([models.alpha1.id, models.alpha2.id, models.alpha2.id, models.beta1.id].sort())
      expect(deleteRequests.every(request => request.query === '')).toBe(true)
      await dialog.getByRole('button', { name: '完成', exact: true }).click()
      await expect(dialog).toBeHidden()
      const final = await state()
      expect(final.models.map(model => model.id).sort()).toEqual([models.beta2.id, models.active.id].sort())
      expect(final.profiles).toHaveLength(0)
      expect(final.runtimes.map(runtime => runtime.id)).toEqual([runtime.data.id])
      expect(final.instances).toHaveLength(0)
      expect(final.downloads).toHaveLength(0)
      expect(final.installations).toHaveLength(0)
      await expect(list.locator('.model-item')).toHaveCount(2)
      expectFilesPreserved()
    })

    await test.step('全部移除成功后退出多选，完成结果对话框保持打开', async () => {
      await page.getByRole('button', { name: '多选', exact: true }).click()
      await check('beta2').check()
      await expectChecked(['beta2'])
      await page.getByRole('button', { name: '移除登记', exact: true }).click()
      await dialog.getByRole('button', { name: '确认移除', exact: true }).click()
      await expect(dialog.getByText('已移除 1 个', { exact: true })).toBeVisible()
      await expect(dialog.getByRole('button', { name: '完成', exact: true })).toBeVisible()
      await expect(dialog).toBeVisible()
      await expect(list.getByRole('checkbox')).toHaveCount(0)
      await expect(list.locator('.model-item')).toHaveCount(1)
      const after = await state()
      expect(after.models.map(model => model.id)).toEqual([models.active.id])
      expect(after.profiles).toHaveLength(0)
      expect(deleteRequests.map(request => request.id).sort()).toEqual([models.alpha1.id, models.alpha2.id, models.alpha2.id, models.beta1.id, models.beta2.id].sort())
      expectFilesPreserved()
      await dialog.getByRole('button', { name: '完成', exact: true }).click()
      await expect(dialog).toBeHidden()
    })
    expect(externalRequests).toEqual([])
    expect(forbiddenRequests).toEqual([])
    expect(errors).toEqual([])
  } finally {
    releaseConflict?.()
    await application.close()
  }
})
