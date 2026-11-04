import { test as base, expect, _electron as electron, type ElectronApplication, type Locator, type Page } from '@playwright/test'
import { spawn, spawnSync, type ChildProcess } from 'node:child_process'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { pinNavigation } from './navigation'

const root = path.resolve(__dirname, '..')
const artifactRoot = process.env.MODLOADER_TEST_ARTIFACT_DIR || path.join(root, '.artifacts', 'codex', 'startup-parameters-implementation-20261101')
type Profile = { id: string; name: string; backend: string; parameters: Record<string, unknown>; generation: Record<string, unknown>; extra_args: string[]; [key: string]: unknown }
type Fixture = { model_id: string; profile_id: string; runtime_id: string; model_path: string; runtime_path: string; data_dir: string; strata_model_id: string; strata_profile_id: string; strata_executable: string }
type Scenario = { application: ElectronApplication; page: Page; directory: string; fixture: Fixture; patches: Profile[]; forbidden: string[]; api: (target: string, method?: string, body?: unknown) => Promise<any>; card: (name?: string) => Locator; dialog: () => Locator }

let devServer: ChildProcess | undefined
let devUrl = process.env.MODLOADER_DEV_URL || ''

base.beforeAll(async () => {
  if (devUrl) return
  fs.mkdirSync(artifactRoot, { recursive: true })
  const viteDirectory = fs.mkdtempSync(path.join(artifactRoot, 'ui-vite-'))
  const source = `import('vite').then(async ({createServer})=>{const server=await createServer({configFile:false,root:${JSON.stringify(root)},base:'./',cacheDir:${JSON.stringify(path.join(viteDirectory, 'cache'))},esbuild:{jsx:'automatic'},optimizeDeps:{entries:['index.html']},server:{host:'127.0.0.1',port:0,strictPort:true,hmr:false,watch:{ignored:['**/.artifacts/**','**/work/**','**/data/**','**/release/**']}}});await server.listen();console.log('STARTUP_TEST_DEV_URL='+server.resolvedUrls.local[0]);process.stdin.resume();process.stdin.on('end',async()=>{await server.close();process.exit(0)});})`
  devServer = spawn(process.execPath, ['--input-type=module', '--eval', source], { cwd: root, windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'] })
  devUrl = await new Promise<string>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('启动参数测试 Vite 启动超时')), 30_000)
    devServer!.stdout!.on('data', chunk => {
      fs.appendFileSync(path.join(viteDirectory, 'vite.log'), chunk)
      const found = String(chunk).match(/STARTUP_TEST_DEV_URL=(http:\/\/127\.0\.0\.1:\d+\/)/)
      if (found) { clearTimeout(timer); resolve(found[1]) }
    })
    devServer!.stderr!.on('data', chunk => fs.appendFileSync(path.join(viteDirectory, 'vite.log'), chunk))
    devServer!.once('error', error => { clearTimeout(timer); reject(error) })
    devServer!.once('exit', code => { clearTimeout(timer); reject(new Error(`启动参数测试 Vite 提前退出 (${code})`)) })
  })
})

base.afterAll(async () => {
  if (!devServer) return
  const server = devServer
  await new Promise<void>(resolve => {
    if (server.exitCode !== null || server.signalCode !== null) { resolve(); return }
    const timer = setTimeout(() => { server.kill(); resolve() }, 10_000)
    server.once('exit', () => { clearTimeout(timer); resolve() })
    server.stdin!.end()
  })
})

const test = base.extend<{ scenario: Scenario }>({
  scenario: async ({}, use, testInfo) => {
    fs.mkdirSync(artifactRoot, { recursive: true })
    const directory = fs.mkdtempSync(path.join(artifactRoot, 'ui-scenario-'))
    const seeded = spawnSync(path.join(root, '.venv', 'Scripts', 'python.exe'), [path.join(root, 'tests', 'fixtures', 'startup-seed.py'), directory], {
      cwd: root, encoding: 'utf8', windowsHide: true,
      env: { ...process.env, PYTHONPATH: path.join(root, 'core'), PYTHONUTF8: '1', PYTHONDONTWRITEBYTECODE: '1', PYTHONPYCACHEPREFIX: path.join(directory, 'pycache') },
    })
    if (seeded.status !== 0) throw new Error(`启动参数 fixture 登记失败：${seeded.stderr}`)
    const fixture = JSON.parse(seeded.stdout) as Fixture
    const application = await electron.launch({ args: [root], env: { ...process.env, MODLOADER_DATA_DIR: fixture.data_dir, MODLOADER_DEV_URL: devUrl, MODLOADER_SENSOR_AUTO_START: '0', PYTHONPYCACHEPREFIX: path.join(directory, 'pycache') }, timeout: 60_000 })
    const patches: Profile[] = [], forbidden: string[] = [], pageErrors: string[] = []
    application.process().stderr?.on('data', chunk => fs.appendFileSync(path.join(directory, 'electron-stderr.log'), chunk))
    await application.context().route('**/*', async route => {
      const request = route.request(), url = new URL(request.url())
      if (['http:', 'https:'].includes(url.protocol) && !['127.0.0.1', 'localhost'].includes(url.hostname)) {
        forbidden.push(`${request.method()} ${url.origin}`); return route.abort()
      }
      if (url.pathname.startsWith('/api/') && !['GET', 'HEAD', 'OPTIONS'].includes(request.method())) {
        const safe = request.method() === 'POST' && ['/api/profiles/preflight', '/api/profiles/launch-parameters', '/api/resources/estimate'].includes(url.pathname)
          || request.method() === 'PATCH' && /^\/api\/profiles\/startup-[\w-]+$/.test(url.pathname)
        if (!safe) { forbidden.push(`${request.method()} ${url.pathname}`); return route.abort() }
        if (request.method() === 'PATCH') patches.push(request.postDataJSON())
      }
      await route.continue()
    })
    const page = await application.firstWindow({ timeout: 60_000 })
    page.on('pageerror', error => pageErrors.push(error.message))
    page.on('console', message => {
      if (['error', 'warning'].includes(message.type())) fs.appendFileSync(path.join(directory, 'renderer.log'), `${message.type()}: ${message.text()}\n`)
    })
    const api = async (target: string, method = 'GET', body?: unknown) => page.evaluate(async ({ target, method, body }) => {
      const { baseUrl, token } = await window.modloader!.connection()
      const response = await fetch(`${baseUrl}${target}`, { method, headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' }, ...(body === undefined ? {} : { body: JSON.stringify(body) }) })
      const value = await response.json()
      if (!response.ok) throw new Error(`${target}: ${response.status} ${JSON.stringify(value)}`)
      return value
    }, { target, method, body })
    const card = (name = 'llama 启动参数') => page.locator('.profile-card').filter({ has: page.getByRole('heading', { name, exact: true }) })
    const dialog = () => page.getByRole('dialog', { name: '编辑加载配置', exact: true })
    try {
      await pinNavigation(page)
      await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
      await page.setViewportSize({ width: 1360, height: 960 })
      await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '模型库', exact: true }).click()
      await page.locator(`.model-item[data-model-id="${fixture.model_id}"]`).click()
      await expect(card()).toBeVisible()
      await use({ application, page, directory, fixture, patches, forbidden, api, card, dialog })
      expect(pageErrors).toEqual([])
      expect(forbidden).toEqual([])
    } finally {
      try {
        if (!page.isClosed()) {
          fs.writeFileSync(path.join(directory, 'renderer-state.json'), JSON.stringify({ url: page.url(), pageErrors, editor: await page.locator('.profile-editor').evaluateAll(nodes => nodes.map(node => node.outerHTML)).catch(() => []) }, null, 2))
          if (testInfo.status !== testInfo.expectedStatus) await page.screenshot({ path: path.join(directory, 'failure.png') }).catch(() => {})
          const state = await api('/api/state')
          const evidence = { test: testInfo.title, instances: state.instances, downloads: state.downloads, installations: state.installations, forbidden, patches: patches.length }
          fs.writeFileSync(path.join(directory, 'safety-evidence.json'), JSON.stringify(evidence, null, 2))
          expect(state.instances).toEqual([])
          expect(state.downloads).toEqual([])
          expect(state.installations).toEqual([])
        }
      } finally {
        await application.close()
        console.log(`Startup parameters UI evidence: ${directory}`)
      }
    }
  },
})

test('配置卡启动参数直接进入编辑加载配置的启动参数页', async ({ scenario: { page, card, dialog, directory } }) => {
  await expect(card().getByRole('button', { name: '命令预览', exact: true })).toHaveCount(0)
  await card().getByRole('button', { name: '启动参数', exact: true }).click()
  await expect(page.getByRole('dialog')).toHaveCount(1)
  await expect(dialog()).toBeVisible()
  await expect(dialog().getByRole('tab', { name: '启动参数', exact: true })).toHaveAttribute('aria-selected', 'true')
  await expect(dialog().getByRole('textbox', { name: '启动参数', exact: true })).toBeVisible()
  await expect(dialog().getByRole('tab', { name: '额外启动参数', exact: true })).toHaveCount(0)
  const text = dialog().getByRole('textbox', { name: '启动参数', exact: true })
  await expect(text).toHaveValue(/--ctx-size 32768/)
  const lines = (await text.inputValue()).split('\n')
  expect(lines.filter(line => /^--/.test(line)).every(line => /^--[\w-]+(?: .*)?$/.test(line))).toBe(true)
  await page.screenshot({ path: path.join(directory, 'startup-parameters-entry.png') })
})

async function pasteParameters(textbox: Locator, value: string) {
  // Deliver paste + input without touching the user's system clipboard.
  await textbox.dispatchEvent('paste')
  await textbox.fill(value)
}

async function savedProfile(scenario: Scenario, id = scenario.fixture.profile_id): Promise<Profile> {
  return (await scenario.api('/api/state')).profiles.find((profile: Profile) => profile.id === id)
}

test('完整命令回填、后续表单修改为准，保存重开不重复映射参数', async ({ scenario }) => {
  const { page, application, card, dialog, fixture, directory } = scenario
  await card().getByRole('button', { name: '启动参数', exact: true }).click()
  const text = dialog().getByRole('textbox', { name: '启动参数', exact: true })
  await expect(text).toHaveValue(/--ctx-size/)
  const futureValue = 'C:\\fixture output 😀\\file'
  const command = `"${fixture.runtime_path}" --model "${fixture.model_path}" --host 127.0.0.1 --port 18888 --alias startup-llama --ctx-size 4096 -c 8192 -ngl auto --cache-type-k q4_0 --cache-type-v q8_0 --temp 0.6 --top-p 0.9 --seed -1 --future-option "${futureValue}"`
  await pasteParameters(text, command)
  await dialog().getByRole('tab', { name: '加载参数', exact: true }).click()
  await expect(dialog().getByLabel('上下文长度', { exact: true })).toHaveValue('8192')
  await expect(dialog().getByLabel('K 缓存类型', { exact: true })).toHaveValue('q4_0')
  await expect(dialog().getByLabel('V 缓存类型', { exact: true })).toHaveValue('q8_0')
  await dialog().getByLabel('上下文长度', { exact: true }).fill('16384')
  await dialog().getByRole('tab', { name: '生成默认值', exact: true }).click()
  await expect(dialog().getByLabel('温度', { exact: true })).toHaveValue('0.6')
  await expect(dialog().getByLabel('Top P', { exact: true })).toHaveValue('0.9')
  await dialog().getByLabel('温度', { exact: true }).fill('0')
  await dialog().getByRole('tab', { name: '启动参数', exact: true }).click()
  await expect(text).toHaveValue(/--ctx-size 16384/)
  await expect(text).toHaveValue(/--temp 0(?:\n|$)/)
  expect((await text.inputValue()).match(/--ctx-size\b/g)).toHaveLength(1)

  // Exercise the real native copy IPC while keeping the operating-system clipboard untouched.
  await application.evaluate(({ clipboard }) => {
    ;(globalThis as any).startupCopied = []
    clipboard.writeText = (value: string) => { (globalThis as any).startupCopied.push(value) }
  })
  await dialog().getByRole('button', { name: '复制参数', exact: true }).click()
  await expect(dialog().getByRole('button', { name: '已复制', exact: true })).toBeVisible()
  const copied = await application.evaluate(() => (globalThis as any).startupCopied)
  expect(copied).toEqual([await text.inputValue()])
  await dialog().getByRole('button', { name: '保存配置', exact: true }).click()
  await expect(dialog()).toBeHidden()
  const saved = await savedProfile(scenario)
  expect(saved.parameters).toMatchObject({ context_length: 16384, gpu_layers: 'auto', cache_type_k: 'q4_0', cache_type_v: 'q8_0' })
  expect(saved.generation).toMatchObject({ temperature: 0, top_p: 0.9, seed: -1 })
  expect(saved.extra_args).toEqual(['--future-option', futureValue])
  expect(saved.parameter_editor_version).toBe(1)
  await card().getByRole('button', { name: '启动参数', exact: true }).click()
  await expect(text).toHaveValue(/--ctx-size 16384/)
  const preview = await scenario.api('/api/profiles/launch-parameters', 'POST', { ...saved, preview: true })
  expect(preview.errors).toEqual([])
  expect(preview.argv.filter((arg: string) => arg === '--ctx-size')).toHaveLength(1)
  expect(preview.argv[preview.argv.indexOf('--ctx-size') + 1]).toBe('16384')
  expect(preview.argv.filter((arg: string) => arg === '--temp')).toHaveLength(1)
  expect(preview.argv[preview.argv.indexOf('--temp') + 1]).toBe('0')
  await page.screenshot({ path: path.join(directory, 'startup-roundtrip.png') })
})

test('错误标注覆盖 flag 和 value，hover 具体原因，修改只即时撤销受影响标注且禁止保存', async ({ scenario }) => {
  const { page, card, dialog, patches, directory } = scenario
  await card().getByRole('button', { name: '启动参数', exact: true }).click()
  const text = dialog().getByRole('textbox', { name: '启动参数', exact: true })
  await expect(text).toHaveValue(/--ctx-size/)
  const invalid = '--future-option "😀"\n--ctx-size 8192\n--cache-type-k f1\n--parallel 0\n--cache-type-v f1'
  await pasteParameters(text, invalid)
  const marks = dialog().locator('mark[data-tooltip]')
  await expect(marks).toHaveText(['--cache-type-k f1', '--parallel 0', '--cache-type-v f1'])
  await expect(text).toHaveAttribute('aria-invalid', 'true')
  const red = await marks.first().evaluate(node => {
    const rgb = getComputedStyle(node).color.match(/\d+/g)!.map(Number)
    return rgb[0] > rgb[1] * 1.3 && rgb[0] > rgb[2] * 1.3
  })
  expect(red).toBe(true)
  await marks.first().hover()
  const tooltip = page.getByRole('tooltip')
  await expect(tooltip).toBeVisible()
  await expect(tooltip).toContainText('K 缓存类型')
  await expect(tooltip).toContainText('q8_0')
  await page.screenshot({ path: path.join(directory, 'startup-error-tooltip.png') })
  await dialog().getByRole('button', { name: '保存配置', exact: true }).click()
  await expect(dialog()).toBeVisible()
  expect(patches).toHaveLength(0)
  expect((await savedProfile(scenario)).parameters).toEqual({})

  await text.fill(invalid.replace('--cache-type-k f1', '--cache-type-k q8_0'))
  await expect(marks).toHaveText(['--parallel 0', '--cache-type-v f1'], { timeout: 300 })
  await expect(dialog().locator('.startup-parameter-count')).toHaveText('2 处错误')
  await text.fill('--ctx-size 8192\n--cache-type-k q8_0\n--parallel 1\n--cache-type-v q8_0')
  await dialog().getByRole('button', { name: '保存配置', exact: true }).click()
  await expect(dialog()).toBeHidden()
  expect(patches).toHaveLength(1)
  expect((await savedProfile(scenario)).parameters).toMatchObject({ context_length: 8192, parallel: 1, cache_type_k: 'q8_0', cache_type_v: 'q8_0' })
})

test('Strata 未校验参数逐行精简提示，预览缺失不阻止合法配置持久保存', async ({ scenario }) => {
  const { page, dialog, fixture, directory, patches } = scenario
  await page.locator(`.model-item[data-model-id="${fixture.strata_model_id}"]`).click()
  const card = scenario.card('Strata 启动参数')
  await card.getByRole('button', { name: '启动参数', exact: true }).click()
  const text = dialog().getByRole('textbox', { name: '启动参数', exact: true })
  await expect(text).toHaveValue(/--max-context 8192/)
  await pasteParameters(text, '--max-context 12288\n--unsupported-startup-flag 7')
  const unsupported = dialog().locator('mark[data-tooltip]')
  await expect(unsupported).toHaveText(['--unsupported-startup-flag 7'])
  await expect(unsupported).toHaveAttribute('data-tooltip', /(?:未列出|不支持)/)
  await expect(text).toHaveAttribute('aria-invalid', 'true')
  await dialog().getByRole('button', { name: '保存配置', exact: true }).click()
  await expect(dialog()).toBeVisible()
  expect(patches).toHaveLength(0)
  const unchanged = await savedProfile(scenario, fixture.strata_profile_id)
  expect(unchanged.parameters.context_length).toBe(8192)
  expect(unchanged.extra_args).toEqual([])
  await page.screenshot({ path: path.join(directory, 'startup-known-unsupported.png') })

  // Missing runtime files affect preview readiness, not the validity of saved parameters.
  // Keep the fake engine file intact in this scenario's directory and restore it afterward.
  const heldEngine = path.join(directory, 'strata-engine-preview-unavailable.fixture')
  fs.renameSync(fixture.strata_executable, heldEngine)
  try {
    await pasteParameters(text, '--max-context 12288\n--batch-size 123456789\n--dws 4\n--dws 5')
    const status = dialog().locator('.startup-parameter-count')
    await expect(status).toHaveText(/^\d+ 项参数 · 2 项未校验$/)
    await expect(status).toHaveAttribute('data-tooltip', '--batch-size：缺少校验规则\n--dws：缺少校验规则')
    await expect(unsupported).toHaveCount(0)
    await expect(dialog().locator('.startup-preview-error')).toContainText('engine/strata.exe')
    await status.hover()
    const tooltip = page.getByRole('tooltip')
    await expect(tooltip).toBeVisible()
    await expect(tooltip).toHaveText('--batch-size：缺少校验规则\n--dws：缺少校验规则')
    const lines = await tooltip.evaluate(node => {
      const range = document.createRange()
      range.selectNodeContents(node)
      return [...range.getClientRects()].filter(rect => rect.width > 0).map(rect => ({ left: rect.left, top: rect.top }))
    })
    expect(lines).toHaveLength(2)
    expect(Math.abs(lines[0].left - lines[1].left)).toBeLessThan(0.5)
    expect(lines[1].top).toBeGreaterThan(lines[0].top)
    await page.screenshot({ path: path.join(directory, 'startup-unchecked-tooltip.png') })
    await dialog().getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(dialog()).toBeHidden()
    expect(patches).toHaveLength(1)
    const saved = await savedProfile(scenario, fixture.strata_profile_id)
    expect(saved.parameters.context_length).toBe(12288)
    expect(saved.extra_args).toEqual(['--batch-size', '123456789', '--dws', '4', '--dws', '5'])
    const preview = await scenario.api('/api/profiles/launch-parameters', 'POST', { ...saved, preview: true })
    expect(preview.errors).toEqual([])
    expect(preview.preview_error).toContain('engine/strata.exe')
    fs.writeFileSync(path.join(directory, 'preview-unavailable-evidence.json'), JSON.stringify({ errors: preview.errors, preview_error: preview.preview_error, extra_args: saved.extra_args }, null, 2))
    await card.getByRole('button', { name: '启动参数', exact: true }).click()
    await expect(text).toHaveValue(/--dws 4/)
  } finally {
    fs.renameSync(heldEngine, fixture.strata_executable)
  }
})

test('Strata 提供有效配置而非已释放临时路径，编辑后参数进入实际 adapter 配置', async ({ scenario }) => {
  const { page, dialog, fixture, directory } = scenario
  await page.locator(`.model-item[data-model-id="${fixture.strata_model_id}"]`).click()
  await scenario.card('Strata 启动参数').getByRole('button', { name: '启动参数', exact: true }).click()
  const text = dialog().getByRole('textbox', { name: '启动参数', exact: true })
  await expect(text).toHaveValue(/--native/)
  await expect(text).toHaveValue(/--pack/)
  expect(await text.inputValue()).not.toMatch(/--config\s/)
  await dialog().locator('.startup-effective-config > summary').click()
  const config = dialog().locator('.startup-effective-config > pre')
  await expect(config).toContainText('"exe"')
  await expect(config).toContainText('--native')
  await expect(config).toContainText('--pack')
  await page.screenshot({ path: path.join(directory, 'startup-strata-effective-config.png') })
  await pasteParameters(text, '--max-context 16384\n--kv k8v4\n--spec 4')
  await dialog().getByRole('tab', { name: '加载参数', exact: true }).click()
  await expect(dialog().getByLabel('上下文长度', { exact: true })).toHaveValue('16384')
  await expect(dialog().getByLabel('KV 缓存', { exact: true })).toHaveValue('k8v4')
  await dialog().getByRole('button', { name: '保存配置', exact: true }).click()
  await expect(dialog()).toBeHidden()
  const saved = await savedProfile(scenario, fixture.strata_profile_id)
  expect(saved.extra_args).toEqual([])
  const preview = await scenario.api('/api/profiles/launch-parameters', 'POST', { ...saved, preview: true })
  expect(preview.errors).toEqual([])
  expect(preview.preparation.status).toBe('ready')
  expect(preview.config.exe).toBe(fixture.strata_executable)
  for (const [flag, expected] of [['--max-context', '16384'], ['--kv', 'k8v4'], ['--spec', '4']]) {
    expect(preview.config.args.filter((arg: string) => arg === flag)).toHaveLength(1)
    expect(preview.config.args[preview.config.args.indexOf(flag) + 1]).toBe(expected)
  }
  expect(fs.existsSync(preview.config.args[preview.config.args.indexOf('--native') + 1])).toBe(true)
  expect(fs.existsSync(preview.config.args[preview.config.args.indexOf('--pack') + 1])).toBe(true)
  fs.writeFileSync(path.join(directory, 'strata-effective-preview.json'), JSON.stringify({ config: preview.config, text: preview.text }, null, 2))
})

test('只查看启动参数不将空白默认落盘，GPU 特殊值保存重开保持原值', async ({ scenario }) => {
  const { card, dialog } = scenario
  await card().getByRole('button', { name: '启动参数', exact: true }).click()
  const text = dialog().getByRole('textbox', { name: '启动参数', exact: true })
  await expect(text).toHaveValue(/--ctx-size 32768/)
  await dialog().getByRole('tab', { name: '加载参数', exact: true }).click()
  await expect(dialog().getByLabel('上下文长度', { exact: true })).toHaveValue('')
  await expect(dialog().getByLabel('GPU 层数', { exact: true })).toHaveValue('')
  await dialog().getByRole('tab', { name: '生成默认值', exact: true }).click()
  await expect(dialog().getByLabel('温度', { exact: true })).toHaveValue('')
  await dialog().getByRole('button', { name: '保存配置', exact: true }).click()
  await expect(dialog()).toBeHidden()
  const empty = await savedProfile(scenario)
  expect(empty.parameters).toEqual({})
  expect(empty.generation).toEqual({})
  expect(empty.extra_args).toEqual([])
  for (const value of ['auto', 'all', -1, -2, 0]) {
    await card().getByRole('button', { name: '启动参数', exact: true }).click()
    await expect(text).toHaveValue(/--ctx-size/)
    await text.fill((await text.inputValue()).replace(/--n-gpu-layers [^\r\n]+/, `--n-gpu-layers ${value}`))
    // Save before the 650 ms typing debounce: persistence must await this final edit.
    await dialog().getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(dialog()).toBeHidden()
    expect((await savedProfile(scenario)).parameters.gpu_layers).toBe(value)
    await card().getByRole('button', { name: '启动参数', exact: true }).click()
    await expect(text).toHaveValue(new RegExp(`--n-gpu-layers ${value}(?:\\n|$)`))
    expect((await text.inputValue()).match(/--n-gpu-layers\b/g)).toHaveLength(1)
    await dialog().getByRole('button', { name: '取消', exact: true }).click()
    await expect(dialog()).toBeHidden()
  }
})

test('合法自动值和别名贯通真实解析、表单、保存与启动预览', async ({ scenario }) => {
  const { page, card, dialog, directory } = scenario
  await card().getByRole('button', { name: '启动参数', exact: true }).click()
  const text = dialog().getByRole('textbox', { name: '启动参数', exact: true })
  await expect(text).toHaveValue(/--ctx-size/)
  await pasteParameters(text, '--ubatch-size 0 -ngl -2 --flash-attn enabled --threads -2 --n-predict 0 --top-p 2')
  await dialog().getByRole('tab', { name: '加载参数', exact: true }).click()
  await expect(dialog().getByLabel('GPU 层数', { exact: true })).toHaveValue('-2')
  await expect(dialog().getByLabel('CPU 线程', { exact: true })).toHaveValue('-2')
  await dialog().getByRole('tab', { name: '生成默认值', exact: true }).click()
  await expect(dialog().getByLabel('最大输出 Token', { exact: true })).toHaveValue('0')
  await expect(dialog().getByLabel('Top P', { exact: true })).toHaveValue('2')
  await dialog().getByRole('tab', { name: '启动参数', exact: true }).click()
  await expect(text).toHaveValue(/--flash-attn on/)
  await expect(dialog().locator('.startup-parameter-error')).toHaveCount(0)
  await page.screenshot({ path: path.join(directory, 'legal-automatic-values.png') })
  await dialog().getByRole('button', { name: '保存配置', exact: true }).click()
  await expect(dialog()).toBeHidden()
  const saved = await savedProfile(scenario)
  expect(saved.parameters).toMatchObject({ ubatch_size: 0, gpu_layers: -2, threads: -2, flash_attention: 'on' })
  expect(saved.generation).toMatchObject({ max_tokens: 0, top_p: 2 })
  expect(saved.extra_args).toEqual([])
  const preview = await scenario.api('/api/profiles/launch-parameters', 'POST', { ...saved, preview: true })
  expect(preview.errors).toEqual([])
  expect(preview.text).toContain('--ubatch-size 0')
  expect(preview.text).toContain('--n-gpu-layers -2')
})

test('手动删除映射参数行持久关闭该项，片段粘贴保留未提及设置', async ({ scenario }) => {
  const { card, dialog } = scenario
  await card().getByRole('button', { name: '启动参数', exact: true }).click()
  const text = dialog().getByRole('textbox', { name: '启动参数', exact: true })
  await expect(text).toHaveValue(/--ctx-size/)
  await pasteParameters(text, '--ctx-size 8192\n--n-gpu-layers auto\n--cache-type-k q4_0')
  await dialog().getByRole('tab', { name: '加载参数', exact: true }).click()
  await expect(dialog().getByLabel('上下文长度', { exact: true })).toHaveValue('8192')
  await dialog().getByRole('tab', { name: '启动参数', exact: true }).click()
  await expect(text).toHaveValue(/--ctx-size 8192/)
  const withoutContext = (await text.inputValue()).split('\n').filter(line => !/^--ctx-size\b/.test(line)).join('\n')
  await text.fill(withoutContext)
  await dialog().getByRole('button', { name: '保存配置', exact: true }).click()
  await expect(dialog()).toBeHidden()
  const deleted = await savedProfile(scenario)
  expect(deleted.parameters).toMatchObject({ context_length: null, gpu_layers: 'auto', cache_type_k: 'q4_0' })
  const preview = await scenario.api('/api/profiles/launch-parameters', 'POST', { ...deleted, preview: true })
  expect(preview.errors).toEqual([])
  expect(preview.argv).not.toContain('--ctx-size')

  await card().getByRole('button', { name: '启动参数', exact: true }).click()
  await expect(text).toHaveValue(/--n-gpu-layers auto/)
  expect(await text.inputValue()).not.toMatch(/--ctx-size\b/)
  await pasteParameters(text, '--threads 4')
  await dialog().getByRole('button', { name: '保存配置', exact: true }).click()
  await expect(dialog()).toBeHidden()
  const pasted = await savedProfile(scenario)
  expect(pasted.parameters).toMatchObject({ context_length: null, gpu_layers: 'auto', cache_type_k: 'q4_0', threads: 4 })
  await card().getByRole('button', { name: '启动参数', exact: true }).click()
  await expect(text).toHaveValue(/--threads 4/)
  await expect(text).toHaveValue(/--cache-type-k q4_0/)
  expect(await text.inputValue()).not.toMatch(/--ctx-size\b/)
})
