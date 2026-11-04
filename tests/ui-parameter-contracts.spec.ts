import { test, expect, type Page } from '@playwright/test'
import { createServer, type ViteDevServer } from 'vite'
import * as fs from 'node:fs'
import * as path from 'node:path'
import type { AppState, LoadProfile, ParameterField } from '../src/types'

const root = path.resolve(__dirname, '..')
const artifacts = path.join(root, '.artifacts', 'codex', 'parameter-validation-fixes-20261102', 'ui')
let server: ViteDevServer, baseUrl: string
test.use({ channel: process.env.MODLOADER_TEST_BROWSER_CHANNEL, viewport: { width: 1360, height: 960 } })
test.beforeAll(async () => {
  fs.mkdirSync(artifacts, { recursive: true })
  server = await createServer({ configFile: false, root, base: './', esbuild: { jsx: 'automatic' },
    cacheDir: path.join(artifacts, 'vite-cache'), optimizeDeps: { entries: ['tests/fixtures/chat-quick-load.html'] },
    server: { host: '127.0.0.1', port: 0, strictPort: true, hmr: false,
      watch: { ignored: ['**/data/**', '**/work/**', '**/release/**', '**/.artifacts/**', '**/.venv/**'] } } })
  await server.listen()
  baseUrl = server.resolvedUrls!.local[0]
})
test.afterAll(async () => { await server?.close() })

const numeric = (key: string, label: string, bounds: Partial<ParameterField> = {}): ParameterField => ({
  key, label, type: 'number', default: null, options: [], description: '', advanced: false, integer: true, step: 1, ...bounds,
})

async function fixture(page: Page, options: { backend?: 'llama_cpp' | 'ninfer'; parameters?: Record<string, unknown>; maximum?: number | null } = {}) {
  const backend = options.backend || 'llama_cpp'
  const gpu = numeric('gpu_layers', 'GPU 层数', { minimum: 0, maximum: options.maximum === undefined ? 49 : options.maximum, special_values: [-1, -2, 'auto', 'all'] })
  const draft = numeric('draft_tokens', 'MTP 草稿 Token', { minimum: 1, maximum: 5 })
  const fields: ParameterField[] = backend === 'llama_cpp' ? [gpu, numeric('threads', 'CPU 线程', { minimum: -2147483648, maximum: 2147483647 })] : [draft]
  const profile: LoadProfile = { id: 'parameter-contract', model_id: 'parameter-model', name: '参数合同配置', backend, runtime_id: 'parameter-runtime', alias: 'parameter-contract',
    parameters: options.parameters || {}, generation: {}, extra_args: [], revision: 1 }
  const state: AppState = { models: [{ id: profile.model_id, name: '参数合同模型', path: '/fixture/model.gguf', format: backend === 'ninfer' ? 'ninfer' : 'gguf', size_bytes: 1,
    files: [], modalities: ['text'], metadata: {}, source: {}, managed: false, warnings: [] }], profiles: [profile],
    runtimes: [{ id: profile.runtime_id, backend, name: backend, path: '/fixture/runtime', version: 'fixture', managed: false, metadata: {} }],
    instances: [], downloads: [], installations: [], hardware: {}, settings: {}, backends: [{ id: backend, label: backend, parameters: fields }] }
  const writes: LoadProfile[] = [], unexpected: string[] = [], errors: string[] = []
  const startupBackfills: unknown[] = []
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/api/**', async route => {
    const request = route.request(), pathname = new URL(request.url()).pathname
    if (pathname === '/api/state') return route.fulfill({ json: state })
    if (pathname === '/api/resources/estimate') return route.fulfill({ json: { resources: [], notes: [], source: 'parameter-contract-fixture' } })
    if (pathname === '/api/profiles/preflight') {
      const body = request.postDataJSON(), mode = body.parameters.acceleration_mode || 'none'
      return route.fulfill({ json: { status: 'ready', message: '', errors: [], candidates: [], vision_enabled: false, parameters: fields, generation_parameters: [],
        mtp: { enabled: false, backend_supported: false, status: 'unsupported', message: '', candidates: [], path: '', needs_path: false, resident_experts: false },
        acceleration: { mode, enabled: mode !== 'none', status: mode === 'none' ? 'disabled' : 'available', fields: backend === 'ninfer' ? [draft] : [], message: '' }, preparation_fields: [] } })
    }
    if (pathname === '/api/profiles/launch-parameters') {
      const body = request.postDataJSON(), parameters = { ...body.parameters }
      // The parser is exercised by backend tests. Here its accepted result is the UI boundary.
      if (body.text !== undefined) {
        const values: Record<string, number | string> = { '-ngl -2': -2, '-ngl auto': 'auto', '-ngl all': 'all' }
        if (!(body.text in values)) throw new Error(`Unexpected startup fixture input: ${body.text}`)
        parameters.gpu_layers = values[body.text]
        startupBackfills.push(parameters.gpu_layers)
      }
      return route.fulfill({ json: { parameters, generation: body.generation, extra_args: body.extra_args, parameter_editor_version: 1,
        text: `--n-gpu-layers ${parameters.gpu_layers ?? -1}`, errors: [], unchecked: [] } })
    }
    if (request.method() === 'PATCH' && pathname === `/api/profiles/${profile.id}`) {
      const saved = { ...request.postDataJSON(), id: profile.id, revision: state.profiles[0].revision + 1 } as LoadProfile
      writes.push(saved); state.profiles[0] = saved
      return route.fulfill({ json: saved })
    }
    unexpected.push(`${request.method()} ${pathname}`)
    return route.abort()
  })
  await page.goto(`${baseUrl}tests/fixtures/chat-quick-load.html`)
  await page.getByRole('button', { name: '模型库', exact: true }).click()
  const card = page.locator('.profile-card').filter({ has: page.getByRole('heading', { name: profile.name, exact: true }) })
  const dialog = page.getByRole('dialog', { name: '编辑加载配置', exact: true })
  await expect(card).toBeVisible()
  return { state, writes, startupBackfills, async edit() {
    await card.getByRole('button', { name: '编辑配置', exact: true }).click()
    await expect(dialog.getByLabel(backend === 'ninfer' ? '推测加速' : 'GPU 层数', { exact: true })).toBeVisible()
    return dialog
  }, dialog, async verify(name: string) {
    expect(unexpected).toEqual([]); expect(errors).toEqual([])
    expect(state.instances).toEqual([]); expect(state.downloads).toEqual([])
    fs.writeFileSync(path.join(artifacts, `${name}.json`), JSON.stringify({ writes, startupBackfills, unexpected, errors, instances: state.instances, downloads: state.downloads }, null, 2))
  } }
}

test('GPU 数字特殊值在手动编辑后仍有效，已知模型层数上限仍生效', async ({ page }) => {
  const f = await fixture(page)
  const dialog = await f.edit(), gpu = dialog.getByLabel('GPU 层数', { exact: true })
  await expect(gpu).toHaveAttribute('min', '-2')
  await expect(gpu).toHaveAttribute('max', '49')
  for (const value of [-1, -2]) {
    await gpu.fill(String(value))
    await expect(gpu).toHaveAttribute('aria-invalid', 'false')
    expect(await gpu.evaluate(node => (node as HTMLInputElement).validity.rangeUnderflow)).toBe(false)
  }
  await page.screenshot({ path: path.join(artifacts, 'gpu-special-minus-two.png'), animations: 'disabled' })
  await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
  await expect(dialog).toBeHidden()
  expect(f.writes.at(-1)!.parameters).toEqual({ gpu_layers: -2 })
  await f.edit()
  for (const value of [-3, 50, 2.5]) {
    await gpu.fill(String(value))
    await expect(gpu).toHaveAttribute('aria-invalid', 'true')
    await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(dialog).toBeVisible()
    expect(f.writes).toHaveLength(1)
  }
  await gpu.fill('49'); await gpu.press('ArrowUp')
  await expect(gpu).toHaveValue('49')
  await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
  await expect(dialog).toBeHidden()
  expect(f.writes.at(-1)!.parameters).toEqual({ gpu_layers: 49 })
  await f.verify('gpu-numeric-sentinels')
})

test('先编辑 GPU 再粘贴启动参数，-2、auto、all 回填后均可保存并重开', async ({ page }) => {
  const f = await fixture(page, { parameters: { gpu_layers: 8 } })
  for (const value of [-2, 'auto', 'all']) {
    const dialog = await f.edit(), gpu = dialog.getByLabel('GPU 层数', { exact: true })
    await gpu.fill('7')
    await dialog.getByRole('tab', { name: '启动参数', exact: true }).click()
    const text = dialog.getByRole('textbox', { name: '启动参数', exact: true })
    await expect(text).toHaveValue('--n-gpu-layers 7')
    await text.dispatchEvent('paste'); await text.fill(`-ngl ${value}`)
    await dialog.getByRole('tab', { name: '加载参数', exact: true }).click()
    await expect(gpu).toHaveAttribute('aria-invalid', 'false')
    await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(dialog).toBeHidden()
    expect(f.writes.at(-1)!.parameters.gpu_layers).toBe(value)
    await f.edit()
    await dialog.getByRole('tab', { name: '启动参数', exact: true }).click()
    await expect(text).toHaveValue(`--n-gpu-layers ${value}`)
    await dialog.getByRole('button', { name: '取消', exact: true }).click()
  }
  expect(f.startupBackfills).toEqual([-2, 'auto', 'all'])
  await f.verify('gpu-startup-sentinels')
})

test('未触碰的旧版 999 全部层数值保存时保持原值', async ({ page }) => {
  const f = await fixture(page, { parameters: { gpu_layers: 999 } })
  const dialog = await f.edit()
  await expect(dialog.getByLabel('GPU 层数', { exact: true })).toHaveValue('49')
  await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
  await expect(dialog).toBeHidden()
  expect(f.writes.at(-1)!.parameters).toEqual({ gpu_layers: 999 })
  await f.verify('gpu-legacy-all')
})

test('未知模型层数不引入上限，CPU 线程接受有效负数但拒绝失真整数', async ({ page }) => {
  const f = await fixture(page, { maximum: null })
  const dialog = await f.edit(), gpu = dialog.getByLabel('GPU 层数', { exact: true }), threads = dialog.getByLabel('CPU 线程', { exact: true })
  await expect(gpu).not.toHaveAttribute('max')
  await gpu.fill('1000'); await threads.fill('-7')
  await expect(threads).toHaveAttribute('aria-invalid', 'false')
  await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
  await expect(dialog).toBeHidden()
  expect(f.writes.at(-1)!.parameters).toEqual({ gpu_layers: 1000, threads: -7 })
  await f.edit()
  await threads.fill('9007199254740992')
  await expect(threads).toHaveAttribute('aria-invalid', 'true')
  await expect(dialog.getByRole('alert')).toHaveText('CPU 线程必须是可精确表示的整数。')
  await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
  await expect(dialog).toBeVisible()
  expect(f.writes).toHaveLength(1)
  await f.verify('numeric-domain-and-precision')
})

test('关闭推测时保留隐藏的 draft_tokens=0，启用后才校验有效范围', async ({ page }) => {
  const f = await fixture(page, { backend: 'ninfer', parameters: { acceleration_mode: 'none', draft_tokens: 0 } })
  const dialog = await f.edit()
  await expect(dialog.getByLabel('MTP 草稿 Token', { exact: true })).toHaveCount(0)
  await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
  await expect(dialog).toBeHidden()
  expect(f.writes.at(-1)!.parameters).toEqual({ acceleration_mode: 'none', draft_tokens: 0 })
  await f.edit()
  await dialog.getByLabel('推测加速', { exact: true }).selectOption('mtp')
  const draft = dialog.getByLabel('MTP 草稿 Token', { exact: true })
  await expect(draft).toHaveValue('0')
  await expect(draft).toHaveAttribute('aria-invalid', 'true')
  await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
  await expect(dialog).toBeVisible()
  expect(f.writes).toHaveLength(1)
  await draft.fill('3')
  await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
  await expect(dialog).toBeHidden()
  expect(f.writes.at(-1)!.parameters).toEqual({ acceleration_mode: 'mtp', draft_tokens: 3 })
  await f.verify('inactive-draft-preservation')
})

test('配置卡上下文摘要按后端显示自动值，保留其他后端及正数语义', async ({ page }) => {
  const f = await fixture(page)
  const cases = [
    { backend: 'vllm', context: -1, expected: '自动', screenshot: 'profile-context-vllm-auto.png' },
    { backend: 'sparkinfer', context: -2147483648, expected: '模型默认', screenshot: 'profile-context-spark-default.png' },
    { backend: 'sparkinfer', context: 0, expected: '模型默认' },
    { backend: 'llama_cpp', context: -1, expected: '-1' },
    { backend: 'ninfer', context: -1, expected: '-1' },
    { backend: 'vllm', context: 32768, expected: '32,768' },
  ]
  const original = f.state.profiles[0], runtime = f.state.runtimes[0]
  f.state.backends = [...new Set(cases.map(item => item.backend))].map(id => ({ id, label: id, parameters: [] }))
  f.state.runtimes = f.state.backends.map(({ id }) => ({ ...runtime, id, backend: id, name: id }))
  f.state.profiles = cases.map(({ backend, context }, index) => ({ ...original, id: `context-${index}`, backend, runtime_id: backend,
    name: `${backend} ${context}`, alias: `context-${index}`, parameters: { context_length: context } }))
  await page.getByRole('button', { name: '刷新夹具', exact: true }).click()
  const summaries = []
  for (const item of cases) {
    const card = page.locator('.profile-card').filter({ has: page.getByRole('heading', { name: `${item.backend} ${item.context}`, exact: true }) })
    const summary = card.locator('.profile-config-summary')
    await expect(summary.getByText(`上下文 ${item.expected}`, { exact: true })).toBeVisible()
    summaries.push({ ...item, summary: await summary.innerText() })
    if (item.screenshot) await card.screenshot({ path: path.join(artifacts, item.screenshot), animations: 'disabled' })
  }
  expect(f.writes).toEqual([])
  fs.writeFileSync(path.join(artifacts, 'context-summary.json'), JSON.stringify(summaries, null, 2))
  await f.verify('context-summary-safety')
})
