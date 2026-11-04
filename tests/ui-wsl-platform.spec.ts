import { test, expect, _electron as electron } from '@playwright/test'
import * as path from 'node:path'
import * as fs from 'node:fs'
import { pinNavigation } from './navigation'

test('WSL 平台 UI：显式检查、平台模板、Linux 浏览导入和同发行版配置', async () => {
  test.setTimeout(180_000)
  const root = path.resolve(__dirname, '..')
  const directory = fs.mkdtempSync(path.join(root, 'work', 'ui-wsl-platform-'))
  const application = await electron.launch({ args: [root], env: { ...process.env, MODLOADER_DATA_DIR: path.join(directory, 'data'), MODLOADER_SENSOR_AUTO_START: '0' }, timeout: 60_000 })
  const page = await application.firstWindow()
  const errors: string[] = [], unexpected: string[] = [], external: string[] = []
  const writes: Array<{ path: string; method: string; body: any }> = []
  const generation = [{ key: 'temperature', label: 'Temperature', type: 'number', default: null, options: [], description: '', advanced: false }]
  const template = { args: ['--model', '{model_path}', '--port', '{port}'], cwd: '', env: {}, health_path: '/v1/models', chat_path: '/v1/chat/completions', api_model_name: '{alias}', generation_parameters: ['temperature', 'max_tokens'], stream_usage: false }
  const runtime = (id: string, platform: string, distribution?: string) => ({ id, backend: 'custom', platform, name: id, path: platform === 'wsl2' ? '/opt/engine/bin/python' : 'D:\\Engine\\python.exe', version: 'custom', managed: false, metadata: { custom: structuredClone(template), preserved: 'retained', ...(distribution ? { distribution } : {}) } })
  const asset = (id: string, distribution: string, modelPath: string, format = 'gguf', name = '同路径模型') => ({ id, name, path: modelPath, location: { platform: 'wsl2', distribution }, format, architecture: null, quantization: null, size_bytes: 0, files: [modelPath], modalities: ['text'], metadata: { size_bytes_known: false }, source: { kind: 'local' }, managed: false, warnings: [] })
  const state: any = {
    models: [], profiles: [], instances: [{ id: 'wsl-live', model_id: 'fixture-model', profile_id: 'fixture-profile', profile_revision: 1, capabilities: {}, started_at: '2026-10-18T06:00:00Z', runtime_id: 'old-vllm', backend: 'vllm', alias: 'WSL 进程归属', platform: 'wsl2', distribution: 'ModLoader', state: 'ready', pid: 54321, guest_pid: 12345, port: 18080, active_requests: 0 }], downloads: [], installations: [], settings: { close_to_tray: false }, hardware: {}, generation_parameters: generation,
    backends: [
      { id: 'llama_cpp', label: 'llama.cpp', parameters: [], supported_platforms: ['windows'], import_kind: 'file', installable: true },
      { id: 'strata', label: 'Strata', parameters: [], supported_platforms: ['windows'], import_kind: 'directory', installable: true, python_option: true },
      { id: 'vllm', label: 'vLLM', parameters: [], generation_parameters: generation, supported_platforms: ['wsl2'], import_kind: 'wsl', installable: true },
      { id: 'sglang', label: 'SGLang', parameters: [], supported_platforms: ['wsl2'], import_kind: 'wsl', installable: true },
      { id: 'ninfer', label: 'nInfer', parameters: [], supported_platforms: ['wsl2'], import_kind: 'wsl', installable: true },
      { id: 'sparkinfer', label: 'SparkInfer', parameters: [], supported_platforms: ['wsl2'], import_kind: 'wsl', installable: true },
      { id: 'custom', label: '自定义引擎', parameters: [], generation_parameters: generation, supported_platforms: ['windows', 'wsl2'], import_kind: 'file', installable: false },
    ],
    runtimes: [runtime('Windows 自定义', 'windows'), runtime('Ubuntu 自定义', 'wsl2', 'Ubuntu'), runtime('Debian 自定义', 'wsl2', 'Debian'), runtime('缺失发行版模板', 'wsl2', 'ArchivedLinux'), { id: 'old-vllm', backend: 'vllm', name: '旧版 vLLM', version: '0.1', path: '/opt/venv/bin/python', managed: true, metadata: { distribution: 'ModLoader' } }],
  }
  const distributions = [
    { name: 'ModLoader', state: 'Stopped', version: 2, managed: true },
    { name: 'Ubuntu', state: 'Stopped', version: 2, managed: false },
    { name: 'Debian', state: 'Running', version: 2, managed: false },
    { name: 'Legacy', state: 'Stopped', version: 1, managed: false },
  ]
  let imported = 0
  page.on('pageerror', error => errors.push(error.message))
  try {
    await application.evaluate(({ BrowserWindow, ipcMain }) => {
      BrowserWindow.getAllWindows()[0].setSize(1440, 1000)
      ;(globalThis as any).wslOpenedPaths = []
      ipcMain.removeHandler('open-path')
      ipcMain.handle('open-path', (_event, value) => { (globalThis as any).wslOpenedPaths.push(value) })
    })
    await page.route('**/*', async route => {
      const req = route.request(), url = new URL(req.url()), pathname = url.pathname
      if (['http:', 'https:'].includes(url.protocol) && !['127.0.0.1', 'localhost'].includes(url.hostname)) { external.push(url.origin); await route.abort(); return }
      if (!pathname.startsWith('/api/')) { await route.continue(); return }
      if (pathname === '/api/conversations' || pathname.startsWith('/api/conversations/')) { await route.continue(); return }
      const body = req.postDataJSON()
      if (req.method() !== 'GET') writes.push({ path: pathname, method: req.method(), body })
      const fulfill = (json: unknown, status = 200) => route.fulfill({ status, json })
      if (pathname === '/api/state') { await fulfill(state); return }
      if (pathname === '/api/platforms/wsl') { await fulfill({ available: true, distributions }); return }
      if (pathname === '/api/platforms/wsl/check') { await fulfill({ distribution: body.distribution, version: 2, python: 'Python 3.12.3', gpu: null, ok: false, errors: ['GPU 驱动暂不可用'] }); return }
      if (pathname === '/api/platforms/wsl/browse') {
        if (body.path === '/denied') { await fulfill({ detail: '所选目录无读取权限' }, 403); return }
        const entries = body.path === '/models' ? [
          { name: '模型目录', path: '/models/模型目录', type: 'directory', size_bytes: null },
          { name: 'model.gguf', path: '/models/model.gguf', type: 'file', size_bytes: 1024 },
          { name: 'weights.bin', path: '/models/weights.bin', type: 'file', size_bytes: null },
          ...Array.from({ length: 130 }, (_, index) => ({ name: `目录 ${index + 1}`, path: `/models/目录 ${index + 1}`, type: 'directory', size_bytes: null })),
        ] : body.path.endsWith('/bin') ? [{ name: 'python', path: `${body.path}/python`, type: 'file', size_bytes: 0 }] : []
        await fulfill({ distribution: body.distribution, path: body.path, parent: body.path === '/' ? null : body.path.slice(0, body.path.lastIndexOf('/')) || '/', entries }); return
      }
      if (pathname === '/api/runtimes/import') {
        const item = { id: `added-${++imported}`, version: 'custom', managed: false, ...body }
        state.runtimes.push(item); await fulfill(item); return
      }
      if (/^\/api\/runtimes\/[^/]+$/.test(pathname) && req.method() === 'PATCH') {
        const item = state.runtimes.find((item: any) => item.id === decodeURIComponent(pathname.split('/').at(-1)!))
        Object.assign(item, body); await fulfill(item); return
      }
      if (pathname === '/api/models/import' || pathname === '/api/models/import-custom') {
        const item = asset(`model-${++imported}`, body.location.distribution, body.path, pathname.endsWith('custom') ? 'custom' : 'gguf', body.name || '同路径模型')
        state.models.push(item); await fulfill(item); return
      }
      if (pathname === '/api/models/import-folder') {
        const item = asset(`folder-${++imported}`, body.location.distribution, `${body.path}/model.gguf`, 'gguf', '目录模型')
        state.models.push(item); await fulfill({ root: body.path, imported: [item], existing: [], errors: [], skipped: [] }); return
      }
      if (pathname === '/api/profiles/preflight') { await fulfill({ status: 'ready', message: '', errors: [], candidates: [], parameters: [], generation_parameters: generation }); return }
      if (pathname === '/api/resources/estimate') { await fulfill({ resources: [], notes: [], source: 'UI fixture' }); return }
      if (pathname === '/api/profiles') { state.profiles.push({ id: 'profile-ui', revision: 1, ...body }); await fulfill(state.profiles.at(-1)); return }
      if (pathname === '/api/installations/history') { await fulfill({ items: [], total: 0, page: 1, page_size: 10 }); return }
      if (pathname === '/api/runtimes/versions') { await fulfill({ versions: [{ version: '1.0', label: '1.0', status: 'available', variants: [{ id: 'cuda', label: 'WSL2 CUDA' }] }], next_cursor: null }); return }
      if (req.method() === 'GET') { await route.continue(); return }
      unexpected.push(`${req.method()} ${pathname}`); await fulfill({ detail: '未授权 fixture 操作' }, 404)
    })
    await pinNavigation(page)
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
    const nav = page.getByRole('navigation', { name: '主导航' })
    await nav.getByRole('button', { name: '运行环境', exact: true }).click()
    const running = page.locator('.instance-table-row').filter({ hasText: 'WSL 进程归属' })
    await running.scrollIntoViewIfNeeded()
    await expect(running).toContainText('ModLoader · Linux PID 12345 · 宿主 PID 54321 · 端口 18080')
    await screenshot('wsl-instance-processes.png')
    const windows = page.locator('.runtime-platform[data-platform="windows"]')
    const wsl = page.locator('.runtime-platform[data-platform="wsl2"]')
    await expect(windows.locator('[data-runtime-id="Windows 自定义"]')).toBeVisible()
    await expect(windows.locator('[data-runtime-id="Ubuntu 自定义"]')).toHaveCount(0)
    await expect(wsl.locator('[data-runtime-id="Ubuntu 自定义"]')).toBeAttached()
    await expect(wsl.locator('[data-runtime-id="old-vllm"]')).toBeAttached()
    await expect(wsl.getByText('ArchivedLinux 当前未列出；相关运行环境登记仍保留。')).toBeVisible()
    await expect(wsl.locator('[data-distribution="Ubuntu"]')).toContainText('已停止')
    await expect(wsl.locator('[data-distribution="Ubuntu"]')).toContainText('用户发行版')
    expect(writes.filter(item => item.path.startsWith('/api/platforms/wsl/'))).toEqual([])
    await wsl.locator('[data-distribution="Ubuntu"]').getByRole('button', { name: '检查发行版', exact: true }).click()
    await expect(wsl.locator('[data-distribution="Ubuntu"]')).toContainText('GPU 驱动暂不可用')
    await expect(wsl.locator('[data-distribution="Legacy"]').getByRole('button', { name: '检查发行版', exact: true })).toBeDisabled()
    await wsl.locator('.section-heading').scrollIntoViewIfNeeded()
    await screenshot('wsl-platform-group.png')
    const ubuntuRow = wsl.locator('[data-runtime-id="Ubuntu 自定义"]')
    await expect(ubuntuRow.getByRole('button', { name: '更新环境', exact: true })).toHaveCount(0)
    await expect(ubuntuRow.getByRole('button', { name: '移除登记', exact: true })).toHaveCount(1)
    await ubuntuRow.getByRole('button', { name: '打开路径', exact: true }).click()
    expect(await application.evaluate(() => (globalThis as any).wslOpenedPaths)).toContain('\\\\wsl.localhost\\Ubuntu\\opt\\engine\\bin\\python')

    await expect(page.locator('.backend-card').getByRole('button', { name: '添加引擎', exact: true })).toHaveCount(0)
    await page.getByRole('button', { name: '导入运行环境', exact: true }).click()
    const importDialog = page.getByRole('dialog', { name: '导入已有运行环境', exact: true })
    await importDialog.getByRole('combobox', { name: '运行平台', exact: true }).selectOption('wsl2')
    await importDialog.getByRole('combobox', { name: '运行后端', exact: true }).selectOption('custom')
    let modal = page.getByRole('dialog', { name: '添加自定义引擎', exact: true })
    await expect(modal.getByRole('combobox', { name: '运行平台', exact: true })).toHaveValue('wsl2')
    await modal.getByRole('combobox', { name: 'WSL 发行版', exact: true }).selectOption('Ubuntu')
    await modal.getByRole('textbox', { name: '引擎名称', exact: true }).fill('Linux 自定义模板')
    await modal.getByRole('textbox', { name: '启动程序', exact: true }).fill('/opt/my engine/bin/python')
    await modal.getByRole('textbox', { name: '启动程序', exact: true }).locator('..').getByRole('button', { name: '浏览', exact: true }).click()
    let browser = page.getByRole('dialog', { name: '浏览 Linux 文件', exact: true })
    await expect(browser.getByRole('button', { name: 'python 0 B', exact: true })).toBeVisible()
    await page.keyboard.press('Escape')
    await expect(browser).toBeHidden()
    await expect(modal).toBeVisible()
    await modal.getByRole('textbox', { name: /^启动参数模板/ }).fill(template.args.join('\n'))
    await screenshot('wsl-custom-launch.png')
    await modal.getByRole('button', { name: '服务与环境选项', exact: true }).click()
    await modal.getByRole('textbox', { name: '工作目录', exact: true }).fill('/home/User/engine work')
    await modal.getByRole('textbox', { name: /^环境变量/ }).fill('CUDA_VISIBLE_DEVICES=0')
    await modal.getByRole('checkbox', { name: /^支持流式用量统计/ }).check()
    await screenshot('wsl-custom-options.png')
    await modal.getByRole('button', { name: '添加引擎', exact: true }).click()
    await expect(modal).toBeHidden()
    const added = state.runtimes.find((item: any) => item.name === 'Linux 自定义模板')
    expect(added).toMatchObject({ platform: 'wsl2', path: '/opt/my engine/bin/python', metadata: { distribution: 'Ubuntu', custom: { cwd: '/home/User/engine work', stream_usage: true, env: { CUDA_VISIBLE_DEVICES: '0' } } } })
    await ubuntuRow.getByRole('button', { name: '编辑模板', exact: true }).click()
    modal = page.getByRole('dialog', { name: '编辑自定义引擎', exact: true })
    await expect(modal.getByRole('combobox', { name: '运行平台', exact: true })).toHaveValue('wsl2')
    await expect(modal.getByRole('combobox', { name: 'WSL 发行版', exact: true })).toHaveValue('Ubuntu')
    await modal.getByRole('textbox', { name: '引擎名称', exact: true }).fill('Ubuntu 已编辑')
    await modal.getByRole('button', { name: '保存模板', exact: true }).click()
    await expect(modal).toBeHidden()
    expect(state.runtimes.find((item: any) => item.id === 'Ubuntu 自定义').metadata.preserved).toBe('retained')
    await page.getByRole('button', { name: '导入运行环境', exact: true }).click()
    await importDialog.getByRole('combobox', { name: '运行后端', exact: true }).selectOption('custom')
    modal = page.getByRole('dialog', { name: '添加自定义引擎', exact: true })
    await expect(modal.getByRole('combobox', { name: '运行平台', exact: true })).toHaveValue('windows')
    await expect(modal.getByRole('combobox', { name: 'WSL 发行版', exact: true })).toHaveCount(0)
    await modal.getByRole('textbox', { name: '引擎名称', exact: true }).fill('本机新增模板')
    await modal.getByRole('textbox', { name: '启动程序', exact: true }).fill('D:\\Engine\\python.exe')
    await modal.getByRole('textbox', { name: /^启动参数模板/ }).fill(template.args.join('\n'))
    await modal.getByRole('button', { name: '添加引擎', exact: true }).click()
    await expect(modal).toBeHidden()
    expect(state.runtimes.find((item: any) => item.name === '本机新增模板').platform).toBe('windows')
    await windows.locator('[data-backend-id="strata"]').getByRole('button', { name: '导入已有', exact: true }).click()
    modal = page.getByRole('dialog', { name: '导入已有运行环境', exact: true })
    await expect(modal.getByRole('combobox', { name: '运行平台', exact: true })).toHaveValue('windows')
    await expect(modal.getByRole('textbox', { name: 'Python 路径', exact: true })).toBeVisible()
    await expect(modal.getByRole('combobox', { name: 'WSL 发行版', exact: true })).toHaveCount(0)
    await modal.getByRole('button', { name: '取消', exact: true }).click()
    await wsl.locator('[data-backend-id="sglang"]').getByRole('button', { name: '安装环境', exact: true }).click()
    modal = page.getByRole('dialog', { name: '安装 SGLang', exact: true })
    await expect(modal).toContainText('共享的 ModLoader 发行版')
    await modal.getByRole('combobox', { name: '官方版本', exact: true }).selectOption('1.0')
    await modal.getByRole('combobox', { name: '运行类型', exact: true }).selectOption('cuda')
    await modal.getByRole('button', { name: '取消', exact: true }).click()

    await nav.getByRole('button', { name: '模型库', exact: true }).click()
    for (const distribution of ['Ubuntu', 'Debian']) {
      await page.getByRole('button', { name: '导入模型', exact: true }).click()
      modal = page.getByRole('dialog', { name: '导入模型文件', exact: true })
      await modal.getByRole('combobox', { name: '模型位置', exact: true }).selectOption('wsl2')
      await modal.getByRole('combobox', { name: 'WSL 发行版', exact: true }).selectOption(distribution)
      await modal.getByRole('textbox', { name: '文件路径', exact: true }).fill('/models/model.gguf')
      if (distribution === 'Ubuntu') {
        await modal.getByRole('button', { name: '浏览', exact: true }).click()
        browser = page.getByRole('dialog', { name: '浏览 Linux 文件', exact: true })
        await expect(browser.getByRole('button', { name: '目录 130 目录', exact: true })).toBeAttached()
        await browser.getByRole('textbox', { name: 'Linux 目录', exact: true }).fill('/denied')
        await browser.getByRole('button', { name: '前往', exact: true }).click()
        await expect(browser.getByRole('alert')).toContainText('无读取权限')
        await browser.getByRole('textbox', { name: 'Linux 目录', exact: true }).fill('/models')
        await browser.getByRole('button', { name: '前往', exact: true }).click()
        await expect(browser.getByRole('button', { name: 'model.gguf 1 KB', exact: true })).toBeVisible()
        await screenshot('wsl-linux-browser.png')
        await browser.getByRole('button', { name: 'model.gguf 1 KB', exact: true }).click()
      }
      await modal.getByRole('button', { name: '导入', exact: true }).click()
      await expect(modal).toBeHidden()
    }
    expect(state.models.filter((item: any) => item.path === '/models/model.gguf')).toHaveLength(2)
    await expect(page.locator('.model-item-meta').filter({ hasText: 'Ubuntu' })).toHaveCount(1)
    await expect(page.locator('.model-item-meta').filter({ hasText: 'Debian' })).toHaveCount(1)

    await page.getByRole('button', { name: '导入模型', exact: true }).click()
    modal = page.getByRole('dialog', { name: '导入模型文件', exact: true })
    await modal.getByRole('combobox', { name: '模型位置', exact: true }).selectOption('wsl2')
    await modal.getByRole('combobox', { name: 'WSL 发行版', exact: true }).selectOption('Ubuntu')
    await modal.getByRole('checkbox', { name: '按自定义模型路径登记', exact: true }).check()
    await modal.getByRole('combobox', { name: '路径类型', exact: true }).selectOption('directory')
    await modal.getByRole('textbox', { name: '模型路径', exact: true }).fill('/home/User/模型目录')
    await modal.getByRole('textbox', { name: /^显示名称/ }).fill('Linux 自定义模型')
    await modal.getByRole('button', { name: '导入', exact: true }).click()
    await expect(modal).toBeHidden()
    const customModel = state.models.find((item: any) => item.format === 'custom')
    await page.locator(`.model-item[data-model-id="${customModel.id}"]`).click()
    await expect(page.locator('.model-facts')).toContainText('未统计')
    await page.getByRole('button', { name: '打开模型目录', exact: true }).click()
    expect(await application.evaluate(() => (globalThis as any).wslOpenedPaths)).toContain('\\\\wsl.localhost\\Ubuntu\\home\\User\\模型目录')
    await page.getByRole('button', { name: '新建配置', exact: true }).first().click()
    modal = page.getByRole('dialog', { name: '新建加载配置', exact: true })
    await expect(modal.getByRole('combobox', { name: '运行后端', exact: true })).toHaveValue('custom')
    const choices = await modal.getByRole('combobox', { name: '运行环境', exact: true }).locator('option').evaluateAll(items => items.map(item => (item as HTMLOptionElement).value).filter(Boolean))
    expect(choices).toEqual(['Ubuntu 自定义', added.id])
    await expect(modal).toContainText('仅列出同一发行版')
    await screenshot('wsl-model-profile.png')
    await modal.getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(modal).toBeHidden()
    expect(state.profiles[0].runtime_id).toBe('Ubuntu 自定义')

    await page.getByRole('button', { name: '导入文件夹', exact: true }).click()
    modal = page.getByRole('dialog', { name: '导入模型文件夹', exact: true })
    await modal.getByRole('combobox', { name: '模型位置', exact: true }).selectOption('wsl2')
    await modal.getByRole('combobox', { name: 'WSL 发行版', exact: true }).selectOption('Debian')
    await modal.getByRole('textbox', { name: '模型文件夹路径', exact: true }).fill('/models/模型目录')
    await modal.getByRole('button', { name: '浏览', exact: true }).click()
    browser = page.getByRole('dialog', { name: '浏览 Linux 目录', exact: true })
    await browser.getByRole('button', { name: '选择此目录', exact: true }).click()
    await modal.getByRole('button', { name: '扫描并导入', exact: true }).click()
    await expect(modal).toContainText('新增 1 个')
    await screenshot('wsl-folder-import.png')
    await modal.getByRole('button', { name: '完成', exact: true }).click()
    expect(writes.find(item => item.path === '/api/models/import-folder')?.body).toEqual({ path: '/models/模型目录', location: { platform: 'wsl2', distribution: 'Debian' } })
    await page.getByRole('button', { name: '导入启动配置', exact: true }).click()
    modal = page.getByRole('dialog', { name: '导入启动配置', exact: true })
    await expect(modal.getByRole('combobox', { name: '模型位置', exact: true })).toHaveCount(0)
    await modal.getByRole('button', { name: '取消', exact: true }).click()
    expect(writes.filter(item => /\/start$|\/install$|\/download/.test(item.path))).toEqual([])
    expect(unexpected).toEqual([])
    expect(external).toEqual([])
    expect(errors).toEqual([])
    console.log(`WSL UI evidence: ${directory}`)
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
