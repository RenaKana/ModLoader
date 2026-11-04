import { test, expect, _electron as electron } from '@playwright/test'
import * as path from 'node:path'
import * as fs from 'node:fs'
import { pinNavigation } from './navigation'

test('存储管理：运行中禁用迁移、默认保留、取消与共享配置切换', async () => {
  test.setTimeout(120_000)
  const root = path.resolve(__dirname, '..')
  const directory = fs.mkdtempSync(path.join(root, 'work', 'ui-model-storage-'))
  const application = await electron.launch({ args: [root], env: { ...process.env, MODLOADER_DATA_DIR: path.join(directory, 'data'), MODLOADER_SENSOR_AUTO_START: '0' }, timeout: 60_000 })
  const page = await application.firstWindow()
  const errors: string[] = [], mutations: string[] = [], submissions: any[] = [], previewRequests: any[] = []
  const gib = 1024 ** 3
  const source = { id: 'source', name: 'Qwen HF · 共享模型', format: 'safetensors', architecture: 'qwen3', quantization: 'NVFP4', path: 'D:\\Models\\Qwen', size_bytes: 20 * gib, files: [], modalities: ['text'], metadata: {}, source: { kind: 'hf' }, managed: true, warnings: [] }
  const native = { ...source, id: 'native', path: '/opt/modloader/models/Qwen-HF-NVFP4/Qwen', location: { platform: 'wsl2', distribution: 'ModLoader' }, source: { kind: 'storage_migration', original_source: source.source } }
  const profile = { id: 'wsl-profile', model_id: 'source', name: 'NInfer · 本地对话', backend: 'ninfer', runtime_id: 'ninfer', alias: 'qwen', parameters: {}, generation: {}, extra_args: [], revision: 1 }
  const windows = { ...profile, id: 'windows-profile', name: 'Windows 原配置', backend: 'transformers', runtime_id: 'windows', alias: 'qwen-windows' }
  const state: any = {
    models: [source], profiles: [profile, windows], migrations: [], downloads: [], installations: [], settings: { close_to_tray: false }, hardware: {},
    instances: [{ id: 'live', model_id: 'source', profile_id: profile.id, runtime_id: 'ninfer', profile_revision: 1, state: 'ready', alias: 'qwen', backend: 'ninfer', port: 12345, started_at: '', active_requests: 0, capabilities: {} }],
    runtimes: [{ id: 'ninfer', name: 'NInfer', backend: 'ninfer', platform: 'wsl2', version: 'pinned', path: '/opt/ninfer', metadata: { distribution: 'ModLoader' }, managed: true }, { id: 'windows', name: 'Transformers', backend: 'transformers', platform: 'windows', version: '5', path: 'D:\\Engine', metadata: {}, managed: true }],
    backends: [{ id: 'ninfer', label: 'NInfer', parameters: [], generation_parameters: [], supported_platforms: ['wsl2'] }, { id: 'transformers', label: 'Transformers', parameters: [], generation_parameters: [], supported_platforms: ['windows'] }],
  }
  let previews = 0
  let sharedSource = true
  page.on('pageerror', error => errors.push(error.message))
  try {
    await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(1440, 1000))
    await page.route('**/*', async route => {
      const req = route.request(), url = new URL(req.url()), pathname = url.pathname
      if (['http:', 'https:'].includes(url.protocol) && !['127.0.0.1', 'localhost'].includes(url.hostname)) { await route.abort(); return }
      if (!pathname.startsWith('/api/') || pathname.startsWith('/api/conversations')) { await route.continue(); return }
      const fulfill = (json: unknown, status = 200) => route.fulfill({ json, status })
      if (pathname === '/api/state') { await fulfill(state); return }
      if (pathname === '/api/profiles/preflight') { await fulfill({ status: 'ready', errors: [], candidates: [], parameters: [], generation_parameters: [], message: '' }); return }
      if (pathname === '/api/resources/estimate') { await fulfill({ resources: [], notes: [], source: 'fixture' }); return }
      if (pathname.endsWith('/storage-preview')) {
        previews++
        previewRequests.push(req.postDataJSON())
        await fulfill({ id: `plan-${previews}`, model_id: source.id, eligible: true, errors: [], warnings: [], source_path: source.path, target_path: native.path, target_platform: 'wsl2', distribution: 'ModLoader', total_bytes: 39.4 * gib, prepared_bytes: 19.4 * gib, prepared_count: 1, file_count: 17, profiles: [profile], preserved_profiles: sharedSource ? [{ id: windows.id, name: windows.name, reason: 'Windows 引擎继续使用原文件' }] : [], delete_source_allowed: !sharedSource, delete_source_reasons: sharedSource ? ['仍有 Windows 加载配置使用源模型'] : [] })
        return
      }
      if (pathname.endsWith('/storage-migrate')) {
        submissions.push(req.postDataJSON())
        const job = { id: `job-${submissions.length}`, kind: 'model_migration', name: 'Qwen HF · WSL 存储', state: 'running', phase: 'copying', request: { model_id: source.id }, downloaded_bytes: 4 * gib, total_bytes: 39.4 * gib, message: '正在复制模型文件', can_pause: false }
        state.migrations.push(job); await fulfill(job); return
      }
      if (/\/api\/jobs\/[^/]+\/cancel/.test(pathname)) {
        const job = state.migrations.find((item: any) => pathname.includes(item.id))
        Object.assign(job, { state: 'cancelled', message: '迁移已停止，原文件保留' }); await fulfill(job); return
      }
      if (pathname === '/api/installations/history') { await fulfill({ items: [], total: 0, page: 1, page_size: 10 }); return }
      if (req.method() !== 'GET') mutations.push(`${req.method()} ${pathname}`)
      await fulfill({ detail: 'Unexpected fixture request' }, 404)
    })
    await pinNavigation(page)
    const refresh = () => page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await refresh()
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '模型库', exact: true }).click()
    const manage = page.getByRole('region', { name: '模型存储', exact: true }).getByRole('button', { name: '存储管理', exact: true })
    const modal = page.getByRole('dialog', { name: '存储管理', exact: true })
    await expect(manage).toBeEnabled()
    await expect(page.getByRole('button', { name: '迁移到 WSL', exact: true })).toHaveCount(0)
    await manage.click()
    await expect(modal.getByText(source.path, { exact: true })).toBeVisible()
    await expect(modal.getByRole('button', { name: '迁移到 WSL', exact: true })).toBeDisabled()
    expect(previews).toBe(0)
    await modal.getByRole('button', { name: '关闭', exact: true }).last().click()
    state.instances = []
    await refresh()
    await manage.click()
    await modal.getByRole('button', { name: '迁移到 WSL', exact: true }).click()
    await expect(modal.getByText('1 组 · 19.4 GB', { exact: true })).toBeVisible()
    expect(previewRequests).toEqual([{ target_platform: 'wsl2' }])
    const removeSource = modal.getByRole('checkbox', { name: /完成后删除源文件/ })
    await expect(removeSource).not.toBeChecked()
    await expect(removeSource).toBeDisabled()
    await expect(modal.getByText('仍有 Windows 加载配置使用源模型', { exact: true })).toBeVisible()
    await modal.getByText('保留原路径 · 1 套配置', { exact: true }).click()
    await expect(modal.getByText('Windows 原配置', { exact: true })).toBeVisible()
    await page.screenshot({ path: path.join(directory, 'migration-preview.png') })
    expect(submissions).toEqual([])
    await modal.getByRole('button', { name: '复制到 WSL', exact: true }).click()
    expect(submissions).toEqual([{ plan_id: 'plan-1', delete_source: false }])
    await expect(modal.getByRole('status')).toContainText('4 GB / 39.4 GB')
    await modal.getByRole('button', { name: '关闭', exact: true }).last().click()
    await expect(modal).toBeHidden()
    await expect(page.locator('.profile-card').first().getByRole('button', { name: '加载模型', exact: true })).toBeDisabled()
    await page.screenshot({ path: path.join(directory, 'migration-progress.png') })
    await page.getByRole('region', { name: '模型存储', exact: true }).getByRole('button', { name: '停止', exact: true }).click()
    await manage.click()
    await expect(modal.getByText('迁移已停止，原文件保留', { exact: true })).toBeVisible()
    await modal.getByRole('button', { name: '重试', exact: true }).click()
    await expect(modal.getByRole('button', { name: '复制到 WSL', exact: true })).toBeEnabled()
    await expect(removeSource).not.toBeChecked()
    await modal.getByRole('button', { name: '复制到 WSL', exact: true }).click()
    expect(submissions.at(-1)).toEqual({ plan_id: 'plan-2', delete_source: false })
    state.models.push(native)
    state.profiles[0] = { ...profile, model_id: native.id, revision: 2 }
    Object.assign(state.migrations.at(-1), { state: 'completed', model_id: native.id, source_deleted: false, message: '已切换到 WSL 内部副本，原文件保留', warnings: [] })
    await expect(modal.getByRole('button', { name: '查看目标模型', exact: true })).toBeVisible()
    await expect(modal.getByText('已保留', { exact: true })).toBeVisible()
    await modal.getByRole('button', { name: '查看目标模型', exact: true }).click()
    await expect(page.locator('.model-overview .path-strip')).toContainText(native.path)
    await expect(page.locator('.model-detail')).toContainText('NInfer · 本地对话')
    await expect(page.locator('.model-detail')).not.toContainText('Windows 原配置')
    await expect(page.locator('.model-facts')).toContainText('Hugging Face')
    await expect(page.locator('.model-facts')).not.toContainText('storage_migration')
    await expect(page.locator('.model-overview')).not.toContainText(/原文件保留|源文件|已使用 WSL|引擎性能/)
    await expect(page.getByRole('button', { name: '整理目录', exact: true })).toHaveCount(0)
    expect(state.models.find((item: any) => item.id === source.id).path).toBe(source.path)
    expect(state.profiles.find((item: any) => item.id === windows.id).model_id).toBe(source.id)
    await page.screenshot({ path: path.join(directory, 'migration-completed.png') })
    await manage.click()
    await expect(modal.getByText('上次操作的源文件', { exact: true })).toBeVisible()
    await expect(modal.getByText('已保留', { exact: true })).toBeVisible()
    await modal.getByRole('button', { name: '移到 Windows', exact: true }).click()
    await expect(modal.getByRole('textbox', { name: 'Windows 目标文件夹', exact: true })).toBeVisible()
    await modal.getByRole('button', { name: '返回', exact: true }).click()
    await expect(modal.getByText('上次操作的源文件', { exact: true })).toBeVisible()
    await expect(modal.getByText('已保留', { exact: true })).toBeVisible()
    await expect(modal.getByText(native.path, { exact: true })).toBeVisible()
    await page.screenshot({ path: path.join(directory, 'storage-history-after-return.png'), animations: 'disabled' })
    await modal.getByRole('button', { name: '关闭', exact: true }).last().click()
    await page.getByRole('button', { name: '移除模型', exact: true }).click()
    const remove = page.getByRole('dialog', { name: '移除模型？', exact: true })
    await expect(remove.getByText('WSL 内部文件保留；这里只移除模型登记与加载配置。')).toBeVisible()
    await expect(remove.getByRole('checkbox')).toHaveCount(0)
    await remove.getByRole('button', { name: '取消', exact: true }).click()

    // Imported files outside the application are still the user's choice.
    Object.assign(source, { id: 'external-source', name: '自行导入的模型', path: 'D:\\ExternalModels\\Qwen', managed: false, source: { kind: 'local' } })
    sharedSource = false
    state.models = [source]; state.profiles = [{ ...profile, model_id: source.id }]; state.migrations = []
    await refresh()
    await manage.click()
    await modal.getByRole('button', { name: '迁移到 WSL', exact: true }).click()
    await expect(removeSource).toBeEnabled()
    await expect(removeSource).not.toBeChecked()
    await removeSource.check()
    await expect(modal.getByRole('button', { name: '移动到 WSL', exact: true })).toBeEnabled()
    await page.screenshot({ path: path.join(directory, 'external-source-delete-choice.png') })
    await modal.getByRole('button', { name: '移动到 WSL', exact: true }).click()
    expect(submissions.at(-1)).toEqual({ plan_id: 'plan-3', delete_source: true })
    await modal.getByRole('button', { name: '关闭', exact: true }).last().click()
    Object.assign(state.migrations.at(-1), { state: 'failed', error: '目标空间不足' })
    await refresh()
    const storage = page.getByRole('region', { name: '模型存储', exact: true })
    await expect(storage.getByText('迁移失败', { exact: true })).toBeVisible()
    await expect(storage).not.toContainText('目标空间不足')
    await page.screenshot({ path: path.join(directory, 'migration-failed-card.png'), animations: 'disabled' })
    await manage.click()
    await expect(modal.getByText('目标空间不足', { exact: true })).toBeVisible()
    await modal.getByRole('button', { name: '关闭', exact: true }).last().click()
    Object.assign(state.migrations.at(-1), { state: 'interrupted' })
    await refresh()
    await expect(storage.getByText('迁移已中断', { exact: true })).toBeVisible()
    expect(errors).toEqual([])
    expect(mutations).toEqual([])
    console.log(`MODEL_STORAGE_SCREENSHOTS=${directory}`)
  } finally { await application.close() }
})

test('原生 WSL 导入无迁移历史：选择 Windows 父目录、预览失效与移动完成', async () => {
  test.setTimeout(120_000)
  const root = path.resolve(__dirname, '..')
  const directory = fs.mkdtempSync(path.join(root, 'work', 'ui-model-storage-native-'))
  const application = await electron.launch({ args: [root], env: { ...process.env, MODLOADER_DATA_DIR: path.join(directory, 'data'), MODLOADER_SENSOR_AUTO_START: '0' }, timeout: 60_000 })
  const page = await application.firstWindow()
  const errors: string[] = [], unexpected: string[] = []
  const previews: Array<{ target_platform: string; destination: string }> = [], submissions: any[] = [], imports: any[] = []
  const gib = 1024 ** 3
  const filename = 'Qwen3-8B-Q4_K_M.gguf'
  const source = { id: 'wsl-import', name: 'Qwen3-8B-Q4_K_M', format: 'gguf', architecture: 'qwen3', quantization: 'Q4_K_M', path: `/home/User/models/${filename}`, location: { platform: 'wsl2', distribution: 'Ubuntu' }, size_bytes: 5 * gib, files: [`/home/User/models/${filename}`], modalities: ['text'], metadata: {}, source: { kind: 'local' }, managed: false, warnings: [] }
  const chosenParent = 'E:\\本地 模型'
  const changedParent = 'F:\\推理模型'
  const targetPath = (parent: string) => `${parent}\\Qwen3-8B-Q4_K_M\\${filename}`
  const unrelated = { ...source, id: 'unrelated', name: 'A 保留的模型', path: 'D:\\Existing\\model.gguf', location: { platform: 'windows' }, files: ['D:\\Existing\\model.gguf'] }
  const state: any = { models: [unrelated], profiles: [], migrations: [], instances: [], downloads: [], installations: [], runtimes: [], backends: [], settings: { close_to_tray: false, model_directory: 'D:\\Models' }, hardware: {} }
  page.on('pageerror', error => errors.push(error.message))
  try {
    await application.evaluate(({ BrowserWindow, dialog }, selected) => {
      BrowserWindow.getAllWindows()[0].setSize(1440, 1000)
      ;(globalThis as any).storageFolderDialogs = []
      dialog.showOpenDialog = (async (...args: any[]) => {
        ;(globalThis as any).storageFolderDialogs.push(args.at(-1))
        return { canceled: false, filePaths: [selected] }
      }) as typeof dialog.showOpenDialog
    }, chosenParent)
    await page.route('**/*', async route => {
      const req = route.request(), url = new URL(req.url()), pathname = url.pathname
      if (['http:', 'https:'].includes(url.protocol) && !['127.0.0.1', 'localhost'].includes(url.hostname)) { unexpected.push(url.origin); await route.abort(); return }
      if (!pathname.startsWith('/api/') || pathname.startsWith('/api/conversations')) { await route.continue(); return }
      const fulfill = (json: unknown, status = 200) => route.fulfill({ json, status })
      if (pathname === '/api/state') { await fulfill(state); return }
      if (pathname === '/api/platforms/wsl') { await fulfill({ available: true, distributions: [{ name: 'Ubuntu', state: 'Stopped', version: 2, managed: false }] }); return }
      if (pathname === '/api/platforms/wsl/browse') {
        const body = req.postDataJSON()
        expect(body).toEqual({ distribution: 'Ubuntu', path: '/home/User/models' })
        await fulfill({ distribution: 'Ubuntu', path: body.path, parent: '/home/User', entries: [{ name: filename, path: source.path, type: 'file', size_bytes: source.size_bytes }] }); return
      }
      if (pathname === '/api/models/import') { imports.push(req.postDataJSON()); state.models.push(source); await fulfill(source); return }
      if (pathname.endsWith('/storage-preview')) {
        const body = req.postDataJSON()
        previews.push(body)
        await fulfill({ id: `windows-plan-${previews.length}`, model_id: source.id, eligible: true, errors: [], warnings: [], source_path: source.path, target_path: targetPath(body.destination), target_platform: 'windows', distribution: 'Ubuntu', total_bytes: source.size_bytes, file_count: 1, profiles: [], preserved_profiles: [], delete_source_allowed: true, delete_source_reasons: [] }); return
      }
      if (pathname.endsWith('/storage-migrate')) {
        submissions.push(req.postDataJSON())
        const job = { id: 'to-windows', kind: 'model_migration', name: `${source.name} · Windows 存储`, state: 'running', phase: 'copying', request: { model_id: source.id }, downloaded_bytes: 2 * gib, total_bytes: source.size_bytes, message: '正在复制模型文件', can_pause: false }
        state.migrations.push(job); await fulfill(job); return
      }
      if (pathname === '/api/installations/history') { await fulfill({ items: [], total: 0, page: 1, page_size: 10 }); return }
      unexpected.push(`${req.method()} ${pathname}`)
      await fulfill({ detail: 'Unexpected fixture request' }, 404)
    })
    await pinNavigation(page)
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '模型库', exact: true }).click()
    await page.getByRole('button', { name: '导入模型', exact: true }).click()
    const importDialog = page.getByRole('dialog', { name: '导入模型文件', exact: true })
    await importDialog.getByRole('combobox', { name: '模型位置', exact: true }).selectOption('wsl2')
    await importDialog.getByRole('combobox', { name: 'WSL 发行版', exact: true }).selectOption('Ubuntu')
    await importDialog.getByRole('textbox', { name: '文件路径', exact: true }).fill(source.path)
    await importDialog.getByRole('button', { name: '浏览', exact: true }).click()
    const browser = page.getByRole('dialog', { name: '浏览 Linux 文件', exact: true })
    await browser.getByRole('button', { name: `${filename} 5 GB`, exact: true }).click()
    await expect(importDialog.getByRole('textbox', { name: '文件路径', exact: true })).toHaveValue(source.path)
    await importDialog.getByRole('button', { name: '导入', exact: true }).click()
    await expect(importDialog).toBeHidden()
    expect(imports).toEqual([{ path: source.path, location: source.location }])
    expect(state.migrations).toEqual([])
    expect(await application.evaluate(() => (globalThis as any).storageFolderDialogs)).toEqual([])
    await page.locator(`.model-item[data-model-id="${source.id}"]`).click()
    await expect(page.locator('.model-overview .path-strip')).toContainText(source.path)
    await expect(page.locator('.model-overview')).not.toContainText(/源文件|已使用 WSL|引擎性能/)
    await expect(page.getByRole('button', { name: '整理目录', exact: true })).toHaveCount(0)
    await screenshot('native-wsl-import.png')

    const manage = page.getByRole('region', { name: '模型存储', exact: true }).getByRole('button', { name: '存储管理', exact: true })
    const modal = page.getByRole('dialog', { name: '存储管理', exact: true })
    await manage.click()
    await expect(modal.getByText(source.path, { exact: true })).toBeVisible()
    await expect(modal.getByText('上次操作的源文件', { exact: true })).toHaveCount(0)
    await expect(modal.getByRole('button', { name: '移到 Windows', exact: true })).toBeEnabled()
    expect(previews).toEqual([])
    await screenshot('native-wsl-storage-details.png')
    await modal.getByRole('button', { name: '移到 Windows', exact: true }).click()
    const destination = modal.getByRole('textbox', { name: 'Windows 目标文件夹', exact: true })
    await expect(destination).toHaveValue('D:\\Models')
    await expect(modal.getByRole('button', { name: '移动到 Windows', exact: true })).toBeDisabled()
    await modal.getByRole('button', { name: '浏览', exact: true }).click()
    await expect(destination).toHaveValue(chosenParent)
    expect(await application.evaluate(() => (globalThis as any).storageFolderDialogs)).toEqual([{ properties: ['openDirectory'] }])
    expect(previews).toEqual([])
    await modal.getByRole('button', { name: '检查目标', exact: true }).click()
    await expect(modal.getByText(targetPath(chosenParent), { exact: true })).toBeVisible()
    const removeSource = modal.getByRole('checkbox', { name: /完成后删除源文件/ })
    await expect(removeSource).toBeChecked()
    await removeSource.uncheck()
    await expect(modal.getByRole('button', { name: '复制到 Windows', exact: true })).toBeEnabled()
    await removeSource.check()
    await expect(modal.getByRole('button', { name: '移动到 Windows', exact: true })).toBeEnabled()
    expect(previews).toEqual([{ target_platform: 'windows', destination: chosenParent }])

    // Editing the parent invalidates the checked plan until the user checks again.
    await destination.fill(changedParent)
    await expect(modal.getByText(targetPath(chosenParent), { exact: true })).toHaveCount(0)
    await expect(modal.getByRole('button', { name: '移动到 Windows', exact: true })).toBeDisabled()
    expect(previews).toHaveLength(1)
    expect(submissions).toEqual([])
    await modal.getByRole('button', { name: '检查目标', exact: true }).click()
    await expect(modal.getByText(targetPath(changedParent), { exact: true })).toBeVisible()
    await expect(removeSource).toBeChecked()
    expect(previews).toEqual([{ target_platform: 'windows', destination: chosenParent }, { target_platform: 'windows', destination: changedParent }])
    await screenshot('windows-storage-preview.png')
    await modal.getByRole('button', { name: '移动到 Windows', exact: true }).click()
    expect(submissions).toEqual([{ plan_id: 'windows-plan-2', delete_source: true }])
    await expect(modal.getByRole('status')).toContainText('2 GB / 5 GB')
    await screenshot('windows-storage-progress.png')

    const target = { ...source, id: 'windows-copy', path: targetPath(changedParent), location: { platform: 'windows' }, files: [targetPath(changedParent)], managed: true, source: { kind: 'storage_migration', original_source: source.source } }
    state.models = [unrelated, target]
    Object.assign(state.migrations[0], { state: 'completed', model_id: target.id, source_deleted: true, message: '模型存储位置已更新', warnings: [] })
    await expect(modal).toBeHidden()
    await expect(page.locator(`.model-item[data-model-id="${target.id}"]`)).toHaveAttribute('aria-pressed', 'true')
    await expect(page.locator(`.model-item[data-model-id="${unrelated.id}"]`)).toHaveAttribute('aria-pressed', 'false')
    await expect(page.locator('.model-overview .path-strip')).toContainText(target.path)
    await expect(page.locator('.model-facts')).toContainText('本地导入')
    await expect(page.locator('.model-overview')).not.toContainText(/已删除|源文件|已使用 WSL|引擎性能|storage_migration/)
    await expect(page.getByRole('button', { name: '整理目录', exact: true })).toHaveCount(0)
    await screenshot('windows-storage-completed.png')
    await manage.click()
    await expect(modal.getByText(target.path, { exact: true })).toBeVisible()
    await expect(modal.getByText('上次操作的源文件', { exact: true })).toBeVisible()
    await expect(modal.getByText('已删除', { exact: true })).toBeVisible()
    await screenshot('windows-storage-completed-details.png')
    expect(errors).toEqual([])
    expect(unexpected).toEqual([])
    console.log(`NATIVE_MODEL_STORAGE_SCREENSHOTS=${directory}`)
  } finally { await application.close() }

  async function screenshot(name: string) {
    await page.evaluate(async () => {
      await document.fonts.ready
      await Promise.all(document.getAnimations().filter(animation => animation.effect?.getComputedTiming().iterations !== Infinity).map(animation => animation.finished.catch(() => {})))
      await new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve())))
    })
    const png = await application.evaluate(async ({ BrowserWindow }) => (await BrowserWindow.getAllWindows()[0].webContents.capturePage()).toPNG().toString('base64'))
    fs.writeFileSync(path.join(directory, name), Buffer.from(png, 'base64'))
  }
})
