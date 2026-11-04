import { test, expect, _electron as electron, type Page } from '@playwright/test'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { generationDefaultText, parameterDefaultText } from '../src/parameter-defaults'
import { runtimeGenerationFields } from '../src/custom-engine'
import { chatReasoningCapability, reasoningDefaultLabel, reasoningRequestParameters } from '../src/chat-reasoning'
import type { Backend, LoadProfile, ModelAsset, ParameterField, RuntimeInstallation, RuntimeInstance } from '../src/types'
import { pinNavigation } from './navigation'
import { disableWebSearch, selectChatModel } from './chat-controls'
import ninferState from './fixtures/ninfer-chat-state.json'

const root = path.resolve(__dirname, '..')
const artifactRoot = path.join(root, '.artifacts', 'codex', 'ninfer-stop-defaults-20261025', 'defaults')
const field = (key: string, value: unknown = null): ParameterField => ({ key, label: key, type: 'number', default: value, options: [], description: '', advanced: false })

test('提示保留真实默认的 0、false 和选择标签；不从未知引擎猜数值', () => {
  expect(parameterDefaultText(field('count', 0))).toBe('默认 0')
  expect(parameterDefaultText({ ...field('enabled', false), type: 'boolean' })).toBe('默认 关闭')
  expect(parameterDefaultText({ ...field('mode', 'none'), type: 'select', options: [{ value: 'none', label: '关闭' }] })).toBe('默认 关闭')
  expect(parameterDefaultText(field('unknown'))).toBe('引擎默认')
  const profile = { backend: 'ninfer', generation: { temperature: 0 }, extra_args: ['--temperature', '0.9', '--default-max-tokens', '32768'] }
  expect(generationDefaultText(field('temperature'), profile)).toBe('配置 0')
  expect(generationDefaultText(field('max_tokens', 8192), profile)).toBe('启动配置 32768')
  expect(generationDefaultText(field('temperature'), { ...profile, generation: {}, extra_args: ['--greedy'] })).toBe('启动配置 0')
  expect(generationDefaultText(field('temperature'), { ...profile, generation: {}, extra_args: [] })).toBe('模型生成配置')
  const runtime = { backend: 'ninfer', generation_parameters: [field('max_tokens', 8192)] } as RuntimeInstallation
  expect(runtimeGenerationFields([field('max_tokens')], runtime)[0].default).toBe(8192)
  expect(runtimeGenerationFields([field('max_tokens')], { ...runtime, generation_parameters: undefined })[0].default).toBeNull()
})

test('推理默认仅显示明确配置或模板来源，不将灰色提示变成请求参数', () => {
  const { instance, model, runtime, profile } = ninferState as unknown as { instance: RuntimeInstance; model: ModelAsset; runtime: RuntimeInstallation; profile: LoadProfile }
  const capability = chatReasoningCapability(instance, model, runtime, profile)
  expect(reasoningDefaultLabel(capability, profile)).toBe('模板默认')
  expect(reasoningDefaultLabel(undefined, profile)).toBe('引擎默认')
  expect(reasoningDefaultLabel(capability, { ...profile, extra_args: ['--no-thinking'] })).toBe('启动配置 关闭')
  expect(reasoningDefaultLabel(capability, { ...profile, generation: { reasoning_effort: 'medium' }, extra_args: ['--no-thinking'] })).toBe('配置 中')
  expect(reasoningRequestParameters(capability, undefined, 'ninfer')).toEqual({})
})

async function management(page: Page, target: string) {
  return page.evaluate(async target => {
    const connection = await (window as any).modloader.connection()
    const response = await fetch(`${connection.baseUrl}${target}`, { headers: { Authorization: `Bearer ${connection.token}` } })
    if (!response.ok) throw new Error(`${target}: ${response.status}`)
    return response.json()
  }, target)
}

async function checkDefaultHints(scope: 'chat' | 'models') {
  test.setTimeout(150_000)
  fs.mkdirSync(artifactRoot, { recursive: true })
  const directory = fs.mkdtempSync(path.join(artifactRoot, 'ui-'))
  const app = await electron.launch({ args: [root], env: { ...process.env, MODLOADER_DEV_URL: '', MODLOADER_DATA_DIR: path.join(directory, 'data'), MODLOADER_SENSOR_AUTO_START: '0' }, timeout: 60_000 })
  app.process().stderr?.on('data', chunk => fs.appendFileSync(path.join(directory, 'electron-stderr.log'), chunk))
  let page: Page
  try { page = await app.firstWindow({ timeout: 60_000 }) }
  catch (error) {
    console.log('Default hints startup windows:', await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows().map(window => ({ id: window.id, url: window.webContents.getURL() }))))
    await app.close(); throw error
  }
  const errors: string[] = [], forbidden: string[] = [], requests: Record<string, any>[] = [], saved: LoadProfile[] = []
  let backends: Backend[] = []
  const knownRuntime = { ...ninferState.runtime, generation_parameters: [] as ParameterField[] } as RuntimeInstallation
  const unknownRuntime = { ...knownRuntime, id: 'ninfer-unknown-runtime', version: 'unknown', metadata: {} }
  const transformerRuntime = { id: 'transformer-runtime', backend: 'transformers', name: 'Transformers fixture', path: directory, version: 'fixture', managed: false, metadata: {} } as RuntimeInstallation
  const llamaRuntime = { ...transformerRuntime, id: 'llama-runtime', backend: 'llama_cpp', name: 'llama.cpp fixture' }
  const runtimes = [knownRuntime, unknownRuntime, transformerRuntime, llamaRuntime]
  const nativeModel = ninferState.model as ModelAsset
  const hfModel = { ...nativeModel, id: 'hf-model', name: 'HF 默认提示 fixture', path: path.join(directory, 'hf-model'), location: { platform: 'windows' }, format: 'safetensors' } as ModelAsset
  const ggufModel = { ...hfModel, id: 'gguf-model', name: 'GPU 默认提示 fixture', format: 'gguf', metadata: { 'general.architecture': 'qwen35', 'qwen35.block_count': 65 } } as ModelAsset
  let profiles: LoadProfile[] = [
    { ...ninferState.profile, generation: { max_tokens: 255000, temperature: 0 } },
    { ...ninferState.profile, id: 'ninfer-empty', name: 'NInfer 空配置', alias: 'NInfer 引擎默认' },
    { ...ninferState.profile, id: 'ninfer-unknown', name: 'NInfer 未知版本', alias: 'NInfer 未知版本', runtime_id: unknownRuntime.id },
    { ...ninferState.profile, id: 'transformer-empty', model_id: hfModel.id, backend: 'transformers', runtime_id: transformerRuntime.id, name: '布尔空配置', alias: 'transformer-empty' },
    { ...ninferState.profile, id: 'gpu-empty', model_id: ggufModel.id, backend: 'llama_cpp', runtime_id: llamaRuntime.id, name: 'GPU 空配置', alias: 'gpu-empty' },
  ] as LoadProfile[]
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/*', async route => {
    const url = new URL(route.request().url())
    if (['http:', 'https:'].includes(url.protocol) && !['127.0.0.1', 'localhost'].includes(url.hostname)) { forbidden.push(url.origin); await route.abort() }
    else if (route.request().method() !== 'GET' && /\/api\/(?:instances|runtimes|sources|downloads|config|models)/.test(url.pathname)) { forbidden.push(url.pathname); await route.abort() }
    else await route.continue()
  })
  await page.route('**/api/state', async route => {
    const response = await route.fetch(), state = await response.json()
    backends = state.backends
    const generation = backends.find(backend => backend.id === 'ninfer')!.generation_parameters!
    knownRuntime.generation_parameters = generation.map(field => ({ ...field, default: field.key === 'max_tokens' ? 8192 : field.default }))
    unknownRuntime.generation_parameters = generation
    await route.fulfill({ response, json: { ...state, models: [nativeModel, hfModel, ggufModel], profiles, runtimes,
      instances: profiles.filter(profile => profile.backend === 'ninfer').map(profile => ({ ...ninferState.instance, id: profile.id, alias: profile.alias, profile_id: profile.id, runtime_id: profile.runtime_id })), downloads: [], installations: [],
    } })
  })
  await page.route('**/api/profiles/**', async route => {
    const request = route.request(), pathname = new URL(request.url()).pathname
    if (pathname.endsWith('/preflight')) {
      const profile = request.postDataJSON() as LoadProfile, backend = backends.find(backend => backend.id === profile.backend)!, runtime = runtimes.find(runtime => runtime.id === profile.runtime_id)
      return route.fulfill({ json: { status: 'ready', message: '', candidates: [], errors: [], vision_enabled: false, parameters: backend.parameters.map(field => field.key === 'gpu_layers' ? { ...field, maximum: 66 } : field), generation_parameters: runtime?.generation_parameters || backend.generation_parameters,
        mtp: { enabled: false, status: 'unsupported', backend_supported: false, candidates: [] }, preparation_fields: [], acceleration: { mode: 'none', enabled: false, status: 'available', fields: [] },
      } })
    }
    if (request.method() !== 'PATCH') { forbidden.push(pathname); return route.abort() }
    const id = pathname.split('/')[3], value = { ...request.postDataJSON(), id } as LoadProfile
    saved.push(value); profiles = profiles.map(profile => profile.id === id ? value : profile)
    await route.fulfill({ json: value })
  })
  await page.route('**/api/resources/estimate', route => route.fulfill({ json: { resources: [], notes: [], source: 'default-hints-fixture' } }))
  await page.route('**/api/instances/*/telemetry', route => route.fulfill({ json: { groups: [], unavailable: [], live_generation: null } }))
  await page.route('**/api/chat', async route => {
    requests.push(route.request().postDataJSON())
    await route.fulfill({ contentType: 'text/event-stream', body: `data: ${JSON.stringify({ choices: [{ delta: { content: '默认提示检查完成。' }, finish_reason: 'stop' }] })}\n\ndata: [DONE]\n\n` })
  })
  const screenshot = async (name: string) => {
    await page.mouse.move(20, 108)
    await page.evaluate(() => document.fonts.ready)
    await expect.poll(() => page.evaluate(() => document.getAnimations().some(animation => animation.playState === 'running' && Number.isFinite(Number(animation.effect?.getComputedTiming().iterations))))).toBe(false)
    await page.screenshot({ path: path.join(directory, name) })
  }
  const send = async (text: string) => {
    await disableWebSearch(page)
    const preview = page.getByRole('region', { name: '输入内容预览', exact: true })
    if (await preview.isVisible()) await preview.click({ position: { x: 6, y: 6 } })
    await page.getByRole('textbox', { name: '对话消息', exact: true }).fill(text)
    const count = requests.length
    await page.getByRole('button', { name: '发送消息', exact: true }).click()
    await expect.poll(() => requests.length).toBe(count + 1)
    await expect(page.getByRole('button', { name: '停止生成', exact: true })).toHaveCount(0)
    await expect(page.getByText('已保存', { exact: true })).toBeVisible()
    return requests.at(-1)!
  }
  const card = (name: string) => page.locator('.profile-card').filter({ has: page.getByRole('heading', { name, exact: true }) })
  const dialog = () => page.getByRole('dialog', { name: '编辑加载配置', exact: true })
  try {
    await pinNavigation(page)
    await page.setViewportSize({ width: 1280, height: 900 })
    await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    if (scope === 'chat') {
      await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '测试对话', exact: true }).click()
      await page.getByRole('button', { name: '对话参数', exact: true }).click()
      const options = page.getByRole('complementary', { name: '对话参数', exact: true })
      const temperature = options.getByRole('spinbutton', { name: '温度', exact: true }), maxTokens = options.getByRole('spinbutton', { name: '最大输出 Token', exact: true })
      await expect(maxTokens).toHaveValue('')
      await expect(maxTokens).toHaveAttribute('placeholder', '配置 255000')
      await expect(temperature).toHaveValue('')
      await expect(temperature).toHaveAttribute('placeholder', '配置 0')
      expect(await maxTokens.evaluate(node => getComputedStyle(node, '::placeholder').color)).toBe('rgb(118, 129, 149)')
      await expect(page.locator('.chat-capsule-effort')).toHaveText('模板默认')
      await expect(page.locator('.chat-capsule-effort')).toHaveClass(/is-default/)
      await screenshot('chat-profile-defaults.png')
      const omitted = await send('保持配置默认')
      for (const key of ['temperature', 'max_tokens', 'reasoning_effort']) expect(omitted).not.toHaveProperty(key)
      await temperature.fill('0'); await maxTokens.fill('0')
      expect(await send('显式零值')).toMatchObject({ temperature: 0, max_tokens: 0 })
      profiles[0] = { ...profiles[0], generation: { temperature: 0.8, max_tokens: 4096, reasoning_effort: 'medium' } }
      await page.getByRole('button', { name: '刷新状态', exact: true }).click()
      await expect(maxTokens).toHaveAttribute('placeholder', '配置 4096')
      await expect(maxTokens).toHaveValue('0')
      await expect(page.locator('.chat-capsule-effort')).toHaveText('配置 中')
      await temperature.fill(''); await maxTokens.fill('')
      const cleared = await send('清空恢复继承')
      for (const key of ['temperature', 'max_tokens', 'reasoning_effort']) expect(cleared).not.toHaveProperty(key)
      const catalog = await management(page, '/api/conversations'), conversation = await management(page, `/api/conversations/${catalog.active_id}`)
      expect(conversation.settings.temperature == null).toBe(true)
      expect(conversation.settings.maxTokens == null).toBe(true)
      expect(conversation.settings.reasoningEffort == null).toBe(true)
      await selectChatModel(page, 'ninfer-empty')
      await expect(maxTokens).toHaveValue('')
      await expect(maxTokens).toHaveAttribute('placeholder', '默认 8192')
      await expect(temperature).toHaveAttribute('placeholder', '模型生成配置')
      await screenshot('chat-engine-defaults.png')
      await selectChatModel(page, 'ninfer-unknown')
      await expect(maxTokens).toHaveAttribute('placeholder', '引擎默认')
      expect(await send('未知版本保持省略')).not.toHaveProperty('max_tokens')
    } else {
      await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '模型库', exact: true }).click()
      await page.locator(`.model-item[data-model-id="${nativeModel.id}"]`).click()
      await card('NInfer 空配置').getByRole('button', { name: '编辑配置', exact: true }).click()
      await expect(dialog().getByRole('spinbutton', { name: '最大并行请求', exact: true })).toHaveAttribute('placeholder', '默认 1')
      const kv = dialog().getByLabel('KV 数据类型', { exact: true })
      await expect(kv).toHaveValue('')
      await expect(kv.locator('option').first()).toHaveText('默认 bf16')
      await expect(kv).toHaveCSS('color', 'rgb(118, 129, 149)')
      await screenshot('model-load-defaults.png')
      await kv.selectOption('int8'); await expect(kv).not.toHaveClass(/parameter-default/)
      await kv.selectOption('')
      await dialog().getByRole('tab', { name: '生成默认值', exact: true }).click()
      const generationMax = dialog().getByRole('spinbutton', { name: '最大输出 Token', exact: true })
      await expect(generationMax).toHaveValue('')
      await expect(generationMax).toHaveAttribute('placeholder', '默认 8192')
      await screenshot('model-generation-defaults.png')
      await generationMax.fill('0')
      await dialog().getByRole('button', { name: '保存配置', exact: true }).click()
      await expect(dialog()).toBeHidden()
      expect(saved.at(-1)!.parameters).toEqual({})
      expect(saved.at(-1)!.generation).toEqual({ max_tokens: 0 })
      await card('NInfer 空配置').getByRole('button', { name: '编辑配置', exact: true }).click()
      await dialog().getByRole('tab', { name: '生成默认值', exact: true }).click()
      await generationMax.fill('')
      await dialog().getByRole('button', { name: '保存配置', exact: true }).click()
      await expect(dialog()).toBeHidden()
      expect(saved.at(-1)!.generation).toEqual({})

      await page.locator(`.model-item[data-model-id="${hfModel.id}"]`).click()
      await card('布尔空配置').getByRole('button', { name: '编辑配置', exact: true }).click()
      const compile = dialog().getByLabel('torch.compile', { exact: true })
      await expect(compile).toHaveValue('')
      await expect(compile.locator('option').first()).toHaveText('默认 关闭')
      await expect(compile).toHaveCSS('color', 'rgb(118, 129, 149)')
      await screenshot('model-boolean-defaults.png')
      await compile.selectOption('false')
      await expect(compile).not.toHaveClass(/parameter-default/)
      await dialog().getByRole('button', { name: '保存配置', exact: true }).click()
      await expect(dialog()).toBeHidden()
      expect(saved.at(-1)!.parameters).toEqual({ compile: false })
      await card('布尔空配置').getByRole('button', { name: '编辑配置', exact: true }).click()
      await expect(compile).toHaveValue('false')
      await compile.selectOption('')
      await dialog().getByRole('button', { name: '保存配置', exact: true }).click()
      await expect(dialog()).toBeHidden()
      expect(saved.at(-1)!.parameters).toEqual({})
      await page.locator(`.model-item[data-model-id="${ggufModel.id}"]`).click()
      await card('GPU 空配置').getByRole('button', { name: '编辑配置', exact: true }).click()
      await dialog().locator('details.profile-gpu > summary').click()
      const gpu = dialog().getByRole('spinbutton', { name: 'GPU 层数', exact: true })
      await expect(gpu).toHaveValue('')
      await expect(gpu).toHaveAttribute('placeholder', '默认 66')
      await screenshot('model-gpu-defaults.png')
      await dialog().getByRole('button', { name: '保存配置', exact: true }).click()
      await expect(dialog()).toBeHidden()
      expect(saved.at(-1)!.parameters).toEqual({})
      await card('GPU 空配置').getByRole('button', { name: '编辑配置', exact: true }).click()
      await dialog().locator('details.profile-gpu > summary').click()
      await gpu.fill('0')
      await dialog().getByRole('button', { name: '保存配置', exact: true }).click()
      await expect(dialog()).toBeHidden()
      expect(saved.at(-1)!.parameters).toEqual({ gpu_layers: 0 })
      await card('GPU 空配置').getByRole('button', { name: '编辑配置', exact: true }).click()
      await gpu.fill('')
      await dialog().getByRole('button', { name: '保存配置', exact: true }).click()
      await expect(dialog()).toBeHidden()
      expect(saved.at(-1)!.parameters).toEqual({})
      for (const legacy of [-1, 'all', 999]) {
        profiles = profiles.map(profile => profile.id === 'gpu-empty' ? { ...profile, parameters: { gpu_layers: legacy } } : profile)
        await page.getByRole('button', { name: '刷新状态', exact: true }).click()
        await card('GPU 空配置').getByRole('button', { name: '编辑配置', exact: true }).click()
        await expect(gpu).toHaveValue('66')
        await dialog().getByRole('button', { name: '保存配置', exact: true }).click()
        await expect(dialog()).toBeHidden()
        expect(saved.at(-1)!.parameters).toEqual({ gpu_layers: legacy })
      }
    }
    expect(forbidden).toEqual([])
    expect(errors).toEqual([])
    fs.writeFileSync(path.join(directory, 'requests-and-saves.json'), JSON.stringify({ requests, saved }, null, 2))
  } finally {
    await page.unrouteAll({ behavior: 'wait' })
    await app.close()
    console.log('Default hints fixture and screenshots:', directory)
  }
}

test('聊天继承与刷新、已知和未知引擎默认不改空值请求', async () => checkDefaultHints('chat'))
test('模型配置灰色默认、空值保存、显式 0 和 false、GPU 默认与 legacy 值', async () => checkDefaultHints('models'))
