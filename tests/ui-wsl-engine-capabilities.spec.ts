import { test, expect, _electron as electron } from '@playwright/test'
import * as fs from 'node:fs'
import * as path from 'node:path'
import type { ParameterField } from '../src/types'
import { pinNavigation } from './navigation'

test('正式 WSL 引擎配置：准备状态、原生加速模式及 Strata 语义保留', async () => {
  test.setTimeout(150_000)
  const root = path.resolve(__dirname, '..')
  const artifactRoot = path.join(root, '.artifacts', 'codex', '20261026-release-searxng', 'implementation', 'ninfer-vision')
  fs.mkdirSync(artifactRoot, { recursive: true })
  const directory = fs.mkdtempSync(path.join(artifactRoot, 'ui-wsl-capabilities-'))
  const application = await electron.launch({ args: [root], env: { ...process.env, MODLOADER_DATA_DIR: path.join(directory, 'data'), MODLOADER_SENSOR_AUTO_START: '0' }, timeout: 60_000 })
  const page = await application.firstWindow()
  const errors: string[] = [], unexpected: string[] = []
  const writes: Array<{ path: string; body: any }> = []
  const field = (key: string, label: string, type: ParameterField['type'], attributes: Partial<ParameterField> = {}): ParameterField => ({ key, label, type, default: null, options: [], description: '', advanced: false, ...attributes })
  const sglangFields = [
    field('speculative_num_steps', 'MTP 草稿步数', 'number', { default: 3, minimum: 1, integer: true, step: 1 }),
    field('speculative_eagle_topk', '每步草稿候选', 'number', { default: 1, minimum: 1, integer: true, step: 1 }),
    field('speculative_num_draft_tokens', '验证窗口 Token', 'number', { default: 4, minimum: 1, integer: true, step: 1, description: '候选数为 1 时，验证窗口必须等于草稿步数 + 1。' }),
  ]
  const sglangParameters = [field('context_length', '最大上下文', 'number', { minimum: 1, integer: true }), field('acceleration_mode', '推测解码', 'select', { default: 'none', options: ['none', 'mtp'] }), ...sglangFields]
  const recipes = [
    ['qwen3_6_27b', 'Qwen3.6 27B'], ['qwen3_6_27b_nvfp4', 'Qwen3.6 27B · NVFP4'],
    ['qwen3_8_27b', 'Qwen3.8 27B'], ['qwen3_8_27b_nvfp4', 'Qwen3.8 27B · NVFP4'],
    ['qwen3_6_35b_a3b', 'Qwen3.6 35B A3B'], ['qwen3_8_flash_next', 'Qwen3.8 Flash Next'],
  ].map(([id, label]) => ({ id, label }))
  const preparationFields = [
    field('conversion_recipe', '转换方案', 'select', { options: recipes.map(item => item.id) }),
    field('conversion_device', '转换设备', 'select', { default: 'cpu', options: ['cpu', 'cuda'] }),
    field('prepare_vision', '准备视觉权重', 'boolean', { default: false }),
    field('prepare_mtp', '准备 MTP 权重', 'boolean', { default: false }),
    field('prepare_proposal', '准备 Proposal 权重', 'boolean', { default: false }),
    field('chat_template', '本地聊天模板', 'file'),
    field('generation_config', '本地生成配置（generation_config.json）', 'file'),
    field('tokenizer_directory', '配套分词器目录', 'file', { description: '同时使用 tokenizer.json 和 tokenizer_config.json 两件套。' }),
  ]
  const sourceField = field('conversion_source', '转换源文件或目录', 'file')
  const ninferFields = [field('draft_tokens', '草稿 Token 数', 'number', { default: 3, minimum: 1, maximum: 5, integer: true, step: 1 })]
  const ninferParameters = [field('context_length', '最大上下文', 'number', { minimum: 1, integer: true }), field('vision_enabled', '图像输入', 'boolean'), ...preparationFields, sourceField, field('acceleration_mode', '推测解码', 'select', { default: 'none', options: ['none', 'mtp'] }), ...ninferFields]
  const tokenizerField = field('tokenizer', 'Tokenizer 路径', 'file')
  const sparkFields = [field('draft_model', '草稿模型', 'file', { description: '选择与当前加速模式匹配的本地草稿目录。' }), field('draft_proposals', '草稿提案数', 'number', { minimum: 0, maximum: 15, integer: true, description: '0 为原生自适应。' })]
  const sparkParameters = [field('context_length', '最大上下文', 'number', { minimum: 1, integer: true }), tokenizerField, field('acceleration_mode', '推测解码', 'select', { default: 'none', options: ['none', 'dflash2', 'dspark'] }), ...sparkFields]
  const model = (id: string, name: string, format = 'hf') => ({ id, name, format, path: `/models/${id}`, location: { platform: 'wsl2', distribution: 'ModLoader' }, size_bytes: 0, files: [], modalities: ['text'], metadata: { size_bytes_known: false }, source: {}, managed: false, warnings: [] })
  const state: any = {
    models: [model('sglang-model', '01 SGLang 模型'), model('ninfer-source', '02 NInfer 原始模型'), { ...model('ninfer-artifact', '03 NInfer 已有产物', 'ninfer'), path: '/models/existing.ninfer', modalities: ['text', 'image'] }, model('spark-model', '04 SparkInfer 模型'), { ...model('ninfer-text', '05 NInfer 文本产物', 'ninfer'), path: '/models/text.ninfer' }, { ...model('strata-model', '09 Strata 模型', 'gguf'), path: 'D:\\Models\\strata.gguf', location: { platform: 'windows' } }],
    profiles: [
      { id: 'sglang-profile', name: 'SGLang 配置', model_id: 'sglang-model', runtime_id: 'sglang-runtime', backend: 'sglang', alias: 'sglang-fixture', parameters: { acceleration_mode: 'none', speculative_num_steps: 5, speculative_eagle_topk: 1, speculative_num_draft_tokens: 6 }, generation: { temperature: 0.4 }, extra_args: [], revision: 1 },
      { id: 'ninfer-source-profile', name: 'NInfer 原始配置', model_id: 'ninfer-source', runtime_id: 'ninfer-runtime', backend: 'ninfer', alias: 'ninfer-source', parameters: {}, generation: {}, extra_args: [], revision: 1 },
      { id: 'ninfer-artifact-profile', name: 'NInfer 产物配置', model_id: 'ninfer-artifact', runtime_id: 'ninfer-runtime', backend: 'ninfer', alias: 'ninfer-artifact', parameters: { conversion_recipe: 'legacy-recipe', conversion_device: 'legacy-device', chat_template: '/sources/old-template.jinja', generation_config: '/sources/old-generation-config.json', tokenizer_json: '/sources/old-tokenizer.json', tokenizer_directory: '/sources/old-tokenizer' }, generation: {}, extra_args: [], revision: 1 },
      { id: 'ninfer-text-profile', name: 'NInfer 文本配置', model_id: 'ninfer-text', runtime_id: 'ninfer-runtime', backend: 'ninfer', alias: 'ninfer-text', parameters: {}, generation: {}, extra_args: [], revision: 1 },
      { id: 'spark-profile', name: 'SparkInfer 配置', model_id: 'spark-model', runtime_id: 'spark-runtime', backend: 'sparkinfer', alias: 'spark-fixture', parameters: {}, generation: {}, extra_args: [], revision: 1 },
      { id: 'strata-profile', name: 'Strata 配置', model_id: 'strata-model', runtime_id: 'strata-runtime', backend: 'strata', alias: 'strata-fixture', parameters: {}, generation: {}, extra_args: [], revision: 1 },
    ],
    runtimes: [
      { id: 'sglang-runtime', backend: 'sglang', platform: 'wsl2', name: 'SGLang 环境', path: '/opt/sglang/bin/python', version: 'fixture', managed: false, metadata: { distribution: 'ModLoader' } },
      { id: 'ninfer-runtime', backend: 'ninfer', platform: 'wsl2', name: 'NInfer 环境', path: '/opt/ninfer/bin/python', version: 'fixture', managed: false, metadata: { distribution: 'ModLoader' } },
      { id: 'spark-runtime', backend: 'sparkinfer', platform: 'wsl2', name: 'SparkInfer 环境', path: '/opt/sparkinfer/bin/python', version: 'fixture', managed: false, metadata: { distribution: 'ModLoader' } },
      { id: 'strata-runtime', backend: 'strata', platform: 'windows', name: 'Strata 环境', path: 'D:\\Strata', version: 'fixture', managed: false, metadata: {} },
    ],
    backends: [
      { id: 'sglang', label: 'SGLang', supported_platforms: ['wsl2'], import_kind: 'wsl', parameters: sglangParameters, generation_parameters: [] },
      { id: 'ninfer', label: 'NInfer', supported_platforms: ['wsl2'], import_kind: 'wsl', parameters: ninferParameters, generation_parameters: [field('temperature', 'Temperature', 'number'), field('top_k', 'Top K', 'number'), field('seed', '随机种子', 'number')] },
      { id: 'sparkinfer', label: 'SparkInfer', supported_platforms: ['wsl2'], import_kind: 'wsl', parameters: sparkParameters, generation_parameters: [] },
      { id: 'strata', label: 'Strata', supported_platforms: ['windows'], import_kind: 'directory', parameters: [], generation_parameters: [] },
    ], instances: [], downloads: [], installations: [], settings: { close_to_tray: false }, hardware: {},
  }
  page.on('pageerror', error => errors.push(error.message))
  page.on('console', message => { if (message.type() === 'error') console.error('Renderer:', message.text()) })
  try {
    await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(1420, 960))
    await page.route('**/api/**', async route => {
      const req = route.request(), pathname = new URL(req.url()).pathname, body = req.postDataJSON()
      if (pathname === '/api/conversations' || pathname.startsWith('/api/conversations/')) { await route.continue(); return }
      if (pathname === '/api/state') { await route.fulfill({ json: state }); return }
      if (pathname === '/api/resources/estimate') { await route.fulfill({ json: { resources: [], notes: [], source: 'UI fixture' } }); return }
      if (pathname === '/api/platforms/wsl/browse') {
        await route.fulfill({ json: { distribution: body.distribution, path: body.path, parent: '/', entries: body.path === '/drafts/dspark' ? [] : [{ name: 'dspark', path: '/drafts/dspark', type: 'directory', size_bytes: null }] } }); return
      }
      if (pathname === '/api/profiles/preflight') {
        if (body.backend === 'strata') {
          await route.fulfill({ json: { status: 'ready', source: { kind: 'profile', label: '已有 Strata 资源' }, pack: 'D:\\Strata\\pack', tokenizer: 'D:\\Strata\\tokenizer', candidates: [], errors: [], message: '', parameters: [], mtp: { enabled: false, status: 'builtin', message: 'Strata 原有 MTP', path: '', path_kind: 'directory', needs_path: false, candidates: [], resident_experts: false, backend_supported: true } } }); return
        }
        const mode = body.parameters.acceleration_mode || 'none'
        if (body.backend === 'ninfer') {
          const ready = ['ninfer-artifact', 'ninfer-text'].includes(body.model_id), recipe = body.parameters.conversion_recipe
          const available = ready ? body.model_id === 'ninfer-artifact' : body.parameters.prepare_vision === true
          const enabled = body.parameters.vision_enabled ?? available, missingVision = enabled && !available
          const fields = ready ? [] : [...preparationFields, ...(/nvfp4|flash_next/.test(recipe || '') ? [sourceField] : [])]
          const message = missingVision ? '该 NInfer artifact 未包含视觉组件；请从原始权重重新准备并开启“准备视觉组件”。' : ready ? '使用已有 NInfer 产物' : recipe ? '所选转换方案将在加载时准备；预检不会运行转换。' : '请选择与源模型匹配的转换方案。'
          await route.fulfill({ json: { status: missingVision ? 'error' : ready ? 'ready' : recipe ? 'needs_preparation' : 'choice_required', kind: 'ninfer_artifact', artifact: ready ? `/models/${available ? 'existing' : 'text'}.ninfer` : null, recipes, preparation_fields: fields, errors: missingVision ? [message] : [], candidates: [], parameters: ninferParameters, vision_enabled: enabled, message,
            acceleration: { mode, enabled: mode !== 'none', status: mode === 'none' ? 'disabled' : 'unverified', kind: 'mtp', label: '原生 MTP', requires_draft: false, fields: ninferFields, message: '使用配套的原生 MTP 权重。' } } }); return
        }
        if (body.backend === 'sparkinfer') {
          const draft = body.parameters.draft_model || '', missing = mode !== 'none' && !draft, mismatch = mode === 'dspark' && draft && !draft.includes('dspark')
          const message = missing ? '请选择与所选加速模式匹配的本地草稿模型。' : mismatch ? '草稿配置与 DSpark 不匹配。' : '使用本地模型，不需要转换。'
          await route.fulfill({ json: { status: missing || mismatch ? 'error' : 'ready', kind: 'sparkinfer_local', artifact: '/models/spark-model', preparation_fields: [tokenizerField], errors: missing || mismatch ? [message] : [], candidates: [], parameters: sparkParameters, message,
            acceleration: { mode, enabled: mode !== 'none', status: mode === 'none' ? 'disabled' : missing || mismatch ? 'unsupported' : 'unverified', kind: mode, label: mode === 'dspark' ? 'DSpark' : 'DFlash2', requires_draft: true, fields: sparkFields, message: missing || mismatch ? message : '使用本地草稿配置；不会下载缺失文件。' } } }); return
        }
        await route.fulfill({ json: { status: 'ready', source: null, candidates: [], errors: [], message: '', parameters: sglangParameters, acceleration: { mode, enabled: mode !== 'none', status: mode === 'none' ? 'disabled' : 'unverified', kind: 'mtp', label: '原生 MTP / NEXTN', requires_draft: false, fields: sglangFields, message: 'NEXTN 使用模型内置 MTP 权重；不会下载独立草稿模型。' } } }); return
      }
      if (/^\/api\/profiles\/[^/]+$/.test(pathname) && req.method() === 'PATCH') {
        writes.push({ path: pathname, body: structuredClone(body) })
        const profile = state.profiles.find((item: any) => item.id === pathname.split('/').at(-1))
        Object.assign(profile, body, { revision: profile.revision + 1 }); await route.fulfill({ json: profile }); return
      }
      if (req.method() === 'GET') { await route.continue(); return }
      unexpected.push(`${req.method()} ${pathname}`); await route.fulfill({ status: 404, json: { detail: '未授权 fixture 操作' } })
    })
    await pinNavigation(page)
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
    await page.getByRole('button', { name: /01 SGLang 模型/ }).click()
    await page.getByRole('button', { name: '编辑配置', exact: true }).click()
    let dialog = page.getByRole('dialog', { name: '编辑加载配置', exact: true })
    await expect(dialog.getByRole('combobox', { name: '推测加速', exact: true })).toHaveValue('none')
    await expect(dialog.getByRole('checkbox', { name: 'MTP 推测加速', exact: true })).toHaveCount(0)
    await dialog.getByRole('combobox', { name: '推测加速', exact: true }).selectOption('mtp')
    await expect(dialog.getByRole('spinbutton', { name: 'MTP 草稿步数', exact: true })).toHaveCount(1)
    await expect(dialog.getByRole('spinbutton', { name: 'MTP 草稿步数', exact: true })).toHaveValue('5')
    await expect(dialog.locator('.load-parameters').getByRole('spinbutton', { name: 'MTP 草稿步数', exact: true })).toHaveCount(0)
    await expect(dialog.getByRole('combobox', { name: '推测解码', exact: true })).toHaveCount(0)
    const steps = dialog.getByRole('spinbutton', { name: 'MTP 草稿步数', exact: true })
    await steps.fill('0')
    await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(steps).toHaveAttribute('aria-invalid', 'true')
    expect(writes).toEqual([])
    await steps.fill('64')
    await expect(steps).not.toHaveAttribute('max')
    await dialog.getByRole('spinbutton', { name: '验证窗口 Token', exact: true }).fill('65')
    await screenshot('sglang-native-mtp.png')
    await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(dialog).toBeHidden()
    expect(writes.at(-1)?.body.parameters).toMatchObject({ acceleration_mode: 'mtp', speculative_num_steps: 64, speculative_num_draft_tokens: 65 })
    await page.getByRole('button', { name: '编辑配置', exact: true }).click()
    dialog = page.getByRole('dialog', { name: '编辑加载配置', exact: true })
    await dialog.getByRole('spinbutton', { name: 'MTP 草稿步数', exact: true }).fill('0')
    await dialog.getByRole('combobox', { name: '推测加速', exact: true }).selectOption('none')
    await expect(dialog.getByRole('spinbutton', { name: 'MTP 草稿步数', exact: true })).toHaveCount(0)
    await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(dialog).toBeHidden()
    expect(writes.at(-1)?.body.parameters).toMatchObject({ acceleration_mode: 'none', speculative_num_steps: 0, speculative_num_draft_tokens: 65 })

    await page.getByRole('button', { name: /02 NInfer 原始模型/ }).click()
    await page.getByRole('button', { name: '编辑配置', exact: true }).click()
    dialog = page.getByRole('dialog', { name: '编辑加载配置', exact: true })
    const recipeSelect = dialog.getByRole('combobox', { name: '转换方案', exact: true })
    await expect(recipeSelect).toHaveValue('')
    await expect(dialog.getByText('需要选择', { exact: true })).toBeVisible()
    await expect(dialog.getByRole('checkbox', { name: '图像输入', exact: true })).not.toBeChecked()
    expect(await recipeSelect.locator('option').evaluateAll(items => items.map(item => (item as HTMLOptionElement).value).filter(Boolean))).toEqual(recipes.map(item => item.id))
    await expect(dialog.locator('.load-parameters').getByRole('textbox', { name: '转换源文件或目录', exact: true })).toHaveCount(0)
    await recipeSelect.selectOption('qwen3_6_27b_nvfp4')
    await expect(dialog.getByRole('textbox', { name: '转换源文件或目录', exact: true })).toBeVisible()
    await dialog.getByRole('textbox', { name: '转换源文件或目录', exact: true }).fill('/sources/nvfp4-base')
    await recipeSelect.selectOption('qwen3_8_27b')
    await expect(dialog.getByRole('textbox', { name: '转换源文件或目录', exact: true })).toHaveCount(0)
    await expect(dialog.getByText('加载时准备', { exact: true })).toBeVisible()
    await dialog.getByRole('combobox', { name: '准备视觉权重', exact: true }).selectOption('true')
    await expect(dialog.getByRole('checkbox', { name: '图像输入', exact: true })).toBeChecked()
    await expect(dialog.getByRole('combobox', { name: '视觉投影器', exact: true })).toHaveCount(0)
    await dialog.getByRole('checkbox', { name: '图像输入', exact: true }).uncheck()
    const generationConfig = dialog.getByRole('textbox', { name: '本地生成配置（generation_config.json）', exact: true })
    await generationConfig.fill('/resources/official generation_config.json')
    await generationConfig.evaluate(element => element.scrollIntoView({ block: 'center' }))
    await screenshot('ninfer-generation-resource.png')
    const tokenizerDirectory = dialog.getByRole('textbox', { name: '配套分词器目录', exact: true })
    await tokenizerDirectory.fill('/resources/same-base official tokenizer')
    await tokenizerDirectory.locator('..').getByRole('button', { name: '浏览', exact: true }).click()
    const tokenizerBrowser = page.getByRole('dialog', { name: '浏览 Linux 目录', exact: true })
    await tokenizerBrowser.getByRole('button', { name: '选择此目录', exact: true }).click()
    await expect(tokenizerDirectory).toHaveValue('/resources/same-base official tokenizer')
    await tokenizerDirectory.evaluate(element => element.scrollIntoView({ block: 'center' }))
    await screenshot('ninfer-tokenizer-resource.png')
    await dialog.getByRole('combobox', { name: '转换设备', exact: true }).selectOption('cuda')
    await dialog.getByRole('combobox', { name: '准备 MTP 权重', exact: true }).selectOption('true')
    await dialog.getByRole('combobox', { name: '推测加速', exact: true }).selectOption('mtp')
    const draftTokens = dialog.getByRole('spinbutton', { name: '草稿 Token 数', exact: true })
    await expect(draftTokens).toHaveAttribute('max', '5')
    const beforeInvalid = writes.length
    await draftTokens.fill('6')
    await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(draftTokens).toHaveAttribute('aria-invalid', 'true')
    expect(writes.length).toBe(beforeInvalid)
    await draftTokens.fill('5')
    await screenshot('ninfer-preparation-mtp.png')
    await dialog.getByRole('tab', { name: '生成默认值', exact: true }).click()
    await expect(dialog.getByRole('spinbutton', { name: '重复惩罚', exact: true })).toHaveCount(0)
    await expect(dialog.getByRole('spinbutton', { name: 'Top K', exact: true })).toBeVisible()
    await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(dialog).toBeHidden()
    expect(writes.at(-1)?.body.parameters).toMatchObject({ conversion_recipe: 'qwen3_8_27b', conversion_source: '/sources/nvfp4-base', conversion_device: 'cuda', prepare_vision: true, vision_enabled: false, prepare_mtp: true, acceleration_mode: 'mtp', draft_tokens: 5, generation_config: '/resources/official generation_config.json', tokenizer_directory: '/resources/same-base official tokenizer' })

    await page.getByRole('button', { name: /03 NInfer 已有产物/ }).click()
    await page.getByRole('button', { name: '编辑配置', exact: true }).click()
    dialog = page.getByRole('dialog', { name: '编辑加载配置', exact: true })
    await expect(dialog.locator('.engine-artifact')).toContainText('/models/existing.ninfer')
    await expect(dialog.getByRole('combobox', { name: '转换方案', exact: true })).toHaveCount(0)
    await expect(dialog.getByRole('combobox', { name: '转换设备', exact: true })).toHaveCount(0)
    await expect(dialog.getByRole('textbox', { name: '本地聊天模板', exact: true })).toHaveCount(0)
    await expect(dialog.getByRole('textbox', { name: '本地生成配置（generation_config.json）', exact: true })).toHaveCount(0)
    await expect(dialog.getByRole('button', { name: '移除参数 generation_config', exact: true })).toHaveCount(0)
    await expect(dialog.getByRole('textbox', { name: '本地分词器（tokenizer.json）', exact: true })).toHaveCount(0)
    await expect(dialog.getByRole('button', { name: '移除参数 tokenizer_json', exact: true })).toHaveCount(0)
    await expect(dialog.getByRole('textbox', { name: '配套分词器目录', exact: true })).toHaveCount(0)
    await expect(dialog.getByRole('button', { name: '移除参数 tokenizer_directory', exact: true })).toHaveCount(0)
    await expect(dialog.getByText('已就绪', { exact: true })).toBeVisible()
    await expect(dialog.getByRole('checkbox', { name: '图像输入', exact: true })).toBeChecked()
    await expect(dialog.getByRole('combobox', { name: '图像输入', exact: true })).toHaveCount(0)
    await expect(dialog.getByRole('combobox', { name: '视觉投影器', exact: true })).toHaveCount(0)
    await screenshot('ninfer-artifact-vision.png')
    await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(dialog).toBeHidden()
    expect(writes.at(-1)?.body.parameters).toMatchObject({ conversion_recipe: 'legacy-recipe', conversion_device: 'legacy-device', chat_template: '/sources/old-template.jinja', generation_config: '/sources/old-generation-config.json', tokenizer_json: '/sources/old-tokenizer.json', tokenizer_directory: '/sources/old-tokenizer' })
    await page.getByRole('button', { name: '新建配置', exact: true }).first().click()
    dialog = page.getByRole('dialog', { name: '新建加载配置', exact: true })
    await expect(dialog.getByRole('combobox', { name: '运行后端', exact: true })).toHaveValue('ninfer')
    await dialog.getByRole('button', { name: '取消', exact: true }).click()

    await page.getByRole('button', { name: /05 NInfer 文本产物/ }).click()
    await page.getByRole('button', { name: '编辑配置', exact: true }).click()
    dialog = page.getByRole('dialog', { name: '编辑加载配置', exact: true })
    await expect(dialog.getByRole('checkbox', { name: '图像输入', exact: true })).not.toBeChecked()
    await dialog.getByRole('checkbox', { name: '图像输入', exact: true }).check()
    await expect(dialog.getByText('该 NInfer artifact 未包含视觉组件；请从原始权重重新准备并开启“准备视觉组件”。', { exact: true })).toBeVisible()
    await dialog.getByRole('checkbox', { name: '图像输入', exact: true }).uncheck()
    await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(dialog).toBeHidden()
    expect(writes.at(-1)?.body.parameters).toEqual({ vision_enabled: false })

    await page.getByRole('button', { name: /04 SparkInfer 模型/ }).click()
    await page.getByRole('button', { name: '编辑配置', exact: true }).click()
    dialog = page.getByRole('dialog', { name: '编辑加载配置', exact: true })
    await dialog.getByRole('combobox', { name: '推测加速', exact: true }).selectOption('dflash2')
    await expect(dialog.getByText('请选择与所选加速模式匹配的本地草稿模型。', { exact: true })).toBeVisible()
    await expect(dialog).not.toContainText('MTP')
    const draftPath = () => dialog.getByRole('textbox', { name: '草稿模型', exact: true })
    await draftPath().fill('/drafts/dflash/model.json')
    await draftPath().locator('..').getByRole('button', { name: '浏览', exact: true }).click()
    let browser = page.getByRole('dialog', { name: '浏览 Linux 文件或目录', exact: true })
    await browser.getByRole('button', { name: '选择此目录', exact: true }).click()
    await expect(draftPath()).toHaveValue('/drafts/dflash')
    await expect(dialog.getByText('已就绪', { exact: true })).toBeVisible()
    const proposals = dialog.getByRole('spinbutton', { name: '草稿提案数', exact: true })
    await expect(proposals).toHaveAttribute('min', '0')
    await expect(proposals).toHaveAttribute('max', '15')
    await proposals.fill('1')
    await screenshot('sparkinfer-dflash2.png')
    await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(dialog).toBeHidden()
    expect(writes.at(-1)?.body.parameters).toMatchObject({ acceleration_mode: 'dflash2', draft_model: '/drafts/dflash', draft_proposals: 1 })
    await page.getByRole('button', { name: '编辑配置', exact: true }).click()
    dialog = page.getByRole('dialog', { name: '编辑加载配置', exact: true })
    await dialog.getByRole('combobox', { name: '推测加速', exact: true }).selectOption('dspark')
    await expect(dialog.getByText('草稿配置与 DSpark 不匹配。', { exact: true })).toBeVisible()
    await draftPath().locator('..').getByRole('button', { name: '浏览', exact: true }).click()
    browser = page.getByRole('dialog', { name: '浏览 Linux 文件或目录', exact: true })
    await browser.getByRole('button', { name: 'dspark 目录', exact: true }).click()
    await browser.getByRole('button', { name: '选择此目录', exact: true }).click()
    await expect(draftPath()).toHaveValue('/drafts/dspark')
    await expect(dialog.getByText('已就绪', { exact: true })).toBeVisible()
    await expect(dialog).not.toContainText('MTP')
    await screenshot('sparkinfer-dspark.png')
    await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(dialog).toBeHidden()
    expect(writes.at(-1)?.body.parameters).toMatchObject({ acceleration_mode: 'dspark', draft_model: '/drafts/dspark', draft_proposals: 1 })

    await page.getByRole('button', { name: /09 Strata 模型/ }).click()
    await page.getByRole('button', { name: '编辑配置', exact: true }).click()
    dialog = page.getByRole('dialog', { name: '编辑加载配置', exact: true })
    await expect(dialog.getByRole('checkbox', { name: 'MTP 推测加速', exact: true })).toBeVisible()
    await expect(dialog.getByRole('combobox', { name: '推测加速', exact: true })).toHaveCount(0)
    await expect(dialog.getByText('复用已有资源', { exact: true })).toBeVisible()
    await screenshot('strata-existing-controls.png')
    await dialog.getByRole('button', { name: '取消', exact: true }).click()
    expect(unexpected).toEqual([])
    expect(errors).toEqual([])
    console.log(`WSL engine capability UI evidence: ${directory}`)
  } catch (error) {
    console.error('WSL capability failure:', { url: page.url(), errors })
    await page.screenshot({ path: path.join(directory, 'failure.png') }).catch(() => {})
    throw error
  } finally { await application.close() }

  async function screenshot(name: string) {
    const status = page.locator('.profile-editor .engine-acceleration-mode [role="status"]')
    if (await status.count()) await expect(status).not.toHaveText('检查中…')
    const lastField = page.locator('.profile-editor .engine-acceleration-fields > div').last()
    if (await lastField.count()) await lastField.evaluate(element => element.scrollIntoView({ block: 'center' }))
    await page.evaluate(async () => {
      await document.fonts.ready
      await Promise.all(document.getAnimations().filter(animation => animation.effect?.getComputedTiming().iterations !== Infinity).map(animation => animation.finished.catch(() => {})))
      await new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve())))
    })
    const png = await application.evaluate(async ({ BrowserWindow }) => (await BrowserWindow.getAllWindows()[0].webContents.capturePage()).toPNG().toString('base64'))
    fs.writeFileSync(path.join(directory, name), Buffer.from(png, 'base64'))
  }
})
