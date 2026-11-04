import { test, expect, _electron as electron, type ElectronApplication, type Locator, type Page } from '@playwright/test'
import { createServer, type ServerResponse } from 'node:http'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { pinNavigation } from './navigation'
import { chatCapsule, chatModel, chatModelList, disableWebSearch, openChatControls, selectChatModel, setChatReasoning } from './chat-controls'
import ninferState from './fixtures/ninfer-chat-state.json'

const root = path.resolve(__dirname, '..')
// The artifact's identifying fields match the local record; the ready instance
// and paths are isolated. The HF entry reproduces assets.py's original model_type.
const ninferSource = {
  model: { ...ninferState.model, id: 'ninfer-hf-model', name: 'unsloth--Qwen3.8-27B-NVFP4', path: '/fixture/Qwen3.8-27B-HF', format: 'safetensors', architecture: 'qwen3_5', metadata: { model_type: 'qwen3_5', architectures: ['Qwen3_5ForConditionalGeneration'] } },
  profile: { ...ninferState.profile, id: 'ninfer-hf-profile', model_id: 'ninfer-hf-model', alias: 'NInfer-Qwen3.8-27B-NVFP4 HF', parameters: { conversion_recipe: 'qwen3_8_27b_nvfp4' } },
  instance: { ...ninferState.instance, id: 'ninfer-hf-ready', model_id: 'ninfer-hf-model', profile_id: 'ninfer-hf-profile', alias: 'NInfer-Qwen3.8-27B-NVFP4 HF' },
}
const instances = [
  { id: 'qwen-strata', alias: 'Qwen3.8 Flash-Next GSQ-RCO abliterated IQ3_XXS · 本地长模型别名', backend: 'strata', architecture: 'qwen4exp' },
  { id: 'qwen-llama', alias: 'Qwen3.8 27B NVFP4 MTP HIGH', backend: 'llama_cpp', architecture: 'qwen35' },
  { id: 'deepseek', alias: 'DeepSeek V4.1 Flash', backend: 'strata', architecture: 'deepseek41' },
  { id: 'unknown', alias: '未知模板模型', backend: 'llama_cpp', architecture: 'qwen35' },
].map(item => ({ ...item, profile_id: `${item.id}-profile`, model_id: `${item.id}-model`, runtime_id: `${item.backend}-runtime`, profile_revision: 1, state: 'ready', active_requests: 0, capabilities: { input: ['text'], tools: true } }))
interface Request { body: Record<string, any>; response: ServerResponse }

async function management<T = any>(page: Page, target: string, method = 'GET', body?: unknown): Promise<T> {
  return page.evaluate(async ({ target, method, body }) => {
    const connection = await (window as any).modloader.connection()
    const response = await fetch(`${connection.baseUrl}${target}`, { method, headers: { Authorization: `Bearer ${connection.token}`, 'Content-Type': 'application/json' }, ...(body === undefined ? {} : { body: JSON.stringify(body) }) })
    if (!response.ok) throw new Error(`${method} ${target}: ${response.status} ${await response.text()}`)
    return response.json()
  }, { target, method, body })
}

async function fixture(ninfer = false, recording = false) {
  const directory = fs.mkdtempSync(path.join(root, 'work', recording ? 'chat-reasoning-motion-ui-' : ninfer ? 'chat-reasoning-ninfer-ui-' : 'chat-reasoning-ui-'))
  const requests: Request[] = [], errors: string[] = []
  let application: ElectronApplication | undefined
  const server = createServer(async (request, response) => {
    response.setHeader('Access-Control-Allow-Origin', '*'); response.setHeader('Access-Control-Allow-Headers', '*')
    if (request.method === 'OPTIONS') { response.end(); return }
    let raw = ''
    for await (const chunk of request) raw += chunk
    const body = JSON.parse(raw)
    response.setHeader('Content-Type', body.stream ? 'text/event-stream' : 'application/json')
    response.flushHeaders(); requests.push({ body, response })
  })
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const address = server.address() as { port: number }
  return {
    directory, requests, errors,
    async launch() {
      application = await electron.launch({ args: [root], env: { ...process.env, MODLOADER_DATA_DIR: directory, MODLOADER_SENSOR_AUTO_START: '0' }, timeout: 60_000 })
      const page = await application.firstWindow()
      console.log('Reasoning renderer:', page.url())
      page.on('pageerror', error => errors.push(error.message))
      await pinNavigation(page)
      await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
      await page.route('**/api/state', async route => {
        const original = await route.fetch(), state = await original.json()
        await route.fulfill({ response: original, json: { ...state, instances: [...(ninfer ? [ninferState.instance, ninferSource.instance] : []), ...instances],
          models: [...(ninfer ? [ninferState.model, ninferSource.model] : []), ...instances.map(item => ({ id: item.model_id, name: item.alias, architecture: item.architecture, metadata: {}, format: 'gguf', files: [], modalities: ['text'], source: {}, warnings: [], managed: false, size_bytes: 0, path: path.join(directory, `${item.id}.gguf`) }))],
          profiles: [...(ninfer ? [ninferState.profile, ninferSource.profile] : []), ...instances.map(item => ({ id: item.profile_id, model_id: item.model_id, backend: item.backend, runtime_id: item.runtime_id, alias: item.alias, name: item.alias, parameters: {}, generation: {}, extra_args: [], revision: 1 }))],
          runtimes: [...(ninfer ? [ninferState.runtime] : []), ...['strata', 'llama_cpp'].map(backend => ({ id: `${backend}-runtime`, backend, name: backend, version: 'fixture', path: directory, managed: false, metadata: { supported_flags: ['--jinja', '--reasoning-effort'] } }))],
        } })
      })
      await page.route('**/api/chat', route => route.continue({ url: `http://127.0.0.1:${address.port}/api/chat` }))
      await page.route('**/api/instances/*/telemetry', route => route.fulfill({ json: { groups: [], unavailable: [], live_generation: null } }))
      await page.getByRole('button', { name: '刷新状态', exact: true }).click()
      await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '测试对话', exact: true }).click()
      await expect(page.getByRole('textbox', { name: '对话消息', exact: true, includeHidden: true })).toBeEnabled()
      await disableWebSearch(page)
      return { page, application }
    },
    async send(page: Page, text: string) {
      await disableWebSearch(page)
      const preview = page.getByRole('region', { name: '输入内容预览', exact: true })
      if (await preview.isVisible()) await preview.click({ position: { x: 6, y: 6 } })
      await page.getByRole('textbox', { name: '对话消息', exact: true }).fill(text)
      const before = requests.length
      await page.getByRole('button', { name: '发送消息', exact: true }).click()
      await expect.poll(() => requests.length).toBe(before + 1)
      return requests[before]
    },
    async quit() {
      const app = application!
      await Promise.all(app.windows().map(page => page.unrouteAll({ behavior: 'wait' })))
      const closed = app.waitForEvent('close', { timeout: 30_000 })
      await app.evaluate(({ Menu }) => Menu.getApplicationMenu()!.items.find(item => item.label === '文件')!.submenu!.items.find(item => item.label === '退出 ModLoader')!.click())
      await closed; application = undefined
    },
    async cleanup() {
      if (errors.length) console.log('Reasoning renderer errors:', errors)
      requests.forEach(request => request.response.end())
      if (application) { await Promise.all(application.windows().map(page => page.unrouteAll({ behavior: 'wait' }))); await application.close() }
      server.closeAllConnections(); await new Promise<void>(resolve => server.close(() => resolve()))
      console.log('Reasoning fixture and screenshots:', directory)
    },
  }
}

async function finish(page: Page, request: Request) {
  const chunk = { choices: [{ message: { role: 'assistant', content: '已验证强度参数。' }, finish_reason: 'stop' }], usage: { prompt_tokens: 10, completion_tokens: 12, total_tokens: 22 } }
  request.response.end(request.body.stream ? `data: ${JSON.stringify(chunk)}\n\ndata: [DONE]\n\n` : JSON.stringify(chunk))
  await expect(page.getByRole('button', { name: '停止生成', exact: true })).toHaveCount(0)
  await expect(page.getByText('已保存', { exact: true })).toBeVisible()
}

async function openReasoning(page: Page) {
  return openChatControls(page)
}

async function settleReasoning(panel: Locator) {
  await expect.poll(() => panel.evaluate(node => node.getAnimations({ subtree: true }).some(animation => animation.playState === 'running'))).toBe(false)
}

test('同壳强度与模型视图：拖动预览、提交、反向和减少动态效果', async () => {
  test.setTimeout(120_000)
  const context = await fixture(false, true)
  try {
    const { page } = await context.launch()
    await page.setViewportSize({ width: 1280, height: 800 })
    await page.emulateMedia({ reducedMotion: 'no-preference' })
    await expect(page.locator('.chat-message')).toHaveCount(0)
    await page.waitForTimeout(350)
    const demoStart = Date.now()
    const cues: Array<{ name: string; timestamp: number }> = []
    const mark = (name: string) => cues.push({ name, timestamp: Date.now() / 1000 })
    const capture = await page.context().newCDPSession(page)
    const frames: Array<{ file: string; timestamp: number }> = []
    const frameDirectory = path.join(context.directory, 'frames')
    fs.mkdirSync(frameDirectory)
    capture.on('Page.screencastFrame', frame => {
      const file = `frame-${String(frames.length).padStart(5, '0')}.jpg`
      fs.writeFileSync(path.join(frameDirectory, file), Buffer.from(frame.data, 'base64'))
      frames.push({ file, timestamp: Math.max(demoStart / 1000, frame.metadata.timestamp!) })
      void capture.send('Page.screencastFrameAck', { sessionId: frame.sessionId }).catch(() => {})
    })
    await capture.send('Page.startScreencast', { format: 'jpeg', quality: 85, maxWidth: 1280, maxHeight: 800, everyNthFrame: 1 })
    mark('open')
    await chatCapsule(page).click()
    const popup = page.getByRole('dialog', { name: '模型与推理强度', exact: true })
    await expect(popup).toHaveCSS('opacity', '1')
    const slider = popup.getByRole('slider', { name: '推理强度档位', exact: true })
    await expect(slider).toBeFocused()
    await page.waitForTimeout(400)
    await page.screenshot({ path: path.join(context.directory, 'core-slider-1280.png') })
    const box = (await slider.boundingBox())!
    const stored = async () => { const catalog = await management(page, '/api/conversations'); return catalog.active_id ? (await management(page, `/api/conversations/${catalog.active_id}`)).settings.reasoningEffort : undefined }
    const before = await stored()
    mark('drag')
    await page.mouse.move(box.x + 14, box.y + box.height / 2)
    await page.mouse.down()
    for (let step = 1; step <= 8; step++) {
      await page.mouse.move(box.x + 14 + (box.width - 28) * step / 8, box.y + box.height / 2)
      await page.waitForTimeout(80)
    }
    await expect(popup.locator('.chat-reasoning-heading')).toHaveText('高')
    expect(await stored()).toBe(before)
    mark('release')
    await page.mouse.up()
    await expect(popup.locator('.chat-reasoning-heading')).toHaveText('高')
    await page.waitForTimeout(400)
    await page.screenshot({ path: path.join(context.directory, 'core-high-1280.png') })
    mark('models')
    await chatModel(page).click()
    await expect(chatModelList(page)).toBeFocused()
    await page.waitForTimeout(450)
    await page.screenshot({ path: path.join(context.directory, 'core-models-1280.png') })
    mark('select')
    await page.keyboard.press('ArrowDown'); await page.keyboard.press('Enter')
    await expect(chatModel(page)).toHaveJSProperty('value', 'qwen-llama')
    await expect(slider).toBeFocused()
    await page.waitForTimeout(450)
    mark('reverse-list')
    await chatModel(page).click()
    await page.waitForTimeout(70)
    mark('reverse-back')
    await page.keyboard.press('ArrowLeft')
    await page.waitForTimeout(70)
    mark('reverse-list-again')
    await chatModel(page).click()
    await page.waitForTimeout(70)
    await page.keyboard.press('Home'); await page.keyboard.press('Enter')
    await expect(chatModel(page)).toHaveJSProperty('value', 'qwen-strata')
    await page.waitForTimeout(450)
    await slider.focus(); await page.keyboard.press('End')
    await expect(popup.locator('.chat-reasoning-heading')).toHaveText('高')
    await page.mouse.move(box.x + box.width - 14, box.y + box.height / 2); await page.mouse.down()
    await page.mouse.move(box.x + 14, box.y + box.height / 2, { steps: 8 })
    await expect(popup.locator('.chat-reasoning-heading')).toHaveText('模板默认')
    mark('cancel-drag')
    await page.keyboard.press('Escape'); await page.mouse.up()
    await expect(popup).toBeHidden()
    await page.waitForTimeout(300)
    await chatCapsule(page).click(); await page.waitForTimeout(90)
    await page.keyboard.press('Escape'); await page.waitForTimeout(70)
    await chatCapsule(page).click(); await page.waitForTimeout(450)
    await expect(popup.locator('.chat-reasoning-heading')).toHaveText('高')
    mark('escape-list')
    await chatModel(page).click()
    await page.waitForTimeout(70)
    mark('escape-list-close')
    await page.keyboard.press('Escape')
    await expect(popup).toBeHidden()
    await expect(chatCapsule(page)).toBeFocused()
    await page.waitForTimeout(400)
    const demoEnd = Date.now()
    await capture.send('Page.stopScreencast')
    expect(frames.length).toBeGreaterThan(20)
    fs.writeFileSync(path.join(frameDirectory, 'frames.json'), JSON.stringify({ demoStart, demoEnd, cues, frames }, null, 2))
    fs.writeFileSync(path.join(frameDirectory, 'frames.ffconcat'), 'ffconcat version 1.0\n' + frames.map((frame, index) => `file '${frame.file}'\noption framerate 1000\nduration ${Math.max(.001, (frames[index + 1]?.timestamp ?? demoEnd / 1000) - frame.timestamp)}\n`).join('') + `file '${frames.at(-1)!.file}'\noption framerate 1000\n`)
    await capture.detach()
    await page.getByRole('textbox', { name: '对话消息', exact: true }).fill('只验证设置持久化，不发送')
    await expect.poll(stored).toBe('xhigh')
    await page.emulateMedia({ reducedMotion: 'reduce' })
    await chatCapsule(page).click(); await expect(popup).toHaveCSS('opacity', '1')
    expect(await popup.evaluate(node => getComputedStyle(node).transform)).toBe('none')
    await chatModel(page).click()
    await expect(popup.locator('.chat-model-view')).toHaveCSS('transform', 'none')
    expect(await popup.evaluate(node => node.getAnimations().filter(animation => (animation as CSSTransition).transitionProperty === 'height').length)).toBe(0)
    await page.keyboard.press('ArrowLeft')
    await slider.focus(); await page.keyboard.press('Home'); await page.keyboard.press('ArrowRight')
    await expect(popup.locator('.chat-reasoning-heading')).toHaveText('关闭')
    expect(await popup.locator('.chat-reasoning-thumb').evaluate(node => getComputedStyle(node).transitionProperty)).toBe('none')
    await page.keyboard.press('Escape')
    await page.getByRole('button', { name: '对话历史', exact: true }).click()
    await page.getByRole('button', { name: '对话参数', exact: true }).click()
    await page.setViewportSize({ width: 940, height: 700 })
    await page.waitForTimeout(300)
    await openReasoning(page)
    await page.waitForTimeout(400)
    const capsule = (await chatCapsule(page).boundingBox())!, panel = (await popup.boundingBox())!
    expect(Math.abs(panel.x + panel.width / 2 - capsule.x - capsule.width / 2)).toBeLessThan(1)
    await page.screenshot({ path: path.join(context.directory, 'core-narrow-940.png') })
    await selectChatModel(page, 'deepseek', true)
    await slider.focus(); await page.keyboard.press('Home')
    await expect(popup.locator('.chat-reasoning-heading')).toHaveText('1')
    await popup.getByRole('button', { name: '默认', exact: true }).click()
    const numericBox = (await slider.boundingBox())!
    await page.mouse.click(numericBox.x + 14, numericBox.y + numericBox.height / 2)
    await expect(popup.locator('.chat-reasoning-heading')).toHaveText('1')
    await popup.getByRole('button', { name: '关闭', exact: true }).click()
    await slider.focus(); await page.keyboard.press('End')
    await expect(popup.locator('.chat-reasoning-heading')).toHaveText('100')
    await expect.poll(stored).toBe(100)
    expect(context.requests).toHaveLength(0)
    expect(context.errors).toEqual([])
  } finally { await context.cleanup() }
})

test('NInfer 实际工件元数据恢复滑杆，默认、关闭与各档位进入真实请求', async () => {
  test.setTimeout(120_000)
  const context = await fixture(true)
  try {
    const { page } = await context.launch()
    await page.setViewportSize({ width: 1280, height: 800 })
    await expect(chatCapsule(page)).toContainText('NInfer-Qwen3.8-27B-NVFP4')
    await expect(page.locator('.chat-toolbar select')).toHaveCount(0)
    let popup = await openReasoning(page)
    const slider = popup.getByRole('slider', { name: '推理强度档位', exact: true })
    await expect(slider).toBeVisible()
    await expect(slider).toHaveAttribute('max', '4')
    await expect(popup.locator('.chat-reasoning-levels')).toHaveCount(0)
    await page.keyboard.press('Escape')
    let request = await context.send(page, 'NInfer 保持默认')
    expect(request.body).not.toHaveProperty('reasoning_effort')
    expect(request.body).not.toHaveProperty('chat_template_kwargs')
    await finish(page, request)
    for (const [label, effort] of [['低', 'low'], ['中', 'medium'], ['高', 'xhigh'], ['关闭', 'none']] as const) {
      popup = await openReasoning(page)
      await setChatReasoning(page, label)
      await expect(page.locator('.chat-capsule-effort')).toHaveText(label)
      if (label === '高' || label === '关闭') {
        await settleReasoning(popup)
        await expect(popup).toHaveCSS('opacity', '1')
        const capsule = (await chatCapsule(page).boundingBox())!, panel = (await popup.boundingBox())!
        expect(Math.abs(panel.x + panel.width / 2 - capsule.x - capsule.width / 2)).toBeLessThan(1)
        expect(await popup.locator('.lucide-zap, .lucide-rotate-ccw').count()).toBe(0)
        await page.screenshot({ path: path.join(context.directory, label === '高' ? 'ninfer-high-1280.png' : 'ninfer-off-1280.png') })
      }
      await page.keyboard.press('Escape')
      request = await context.send(page, `NInfer ${label}`)
      expect(request.body).toMatchObject({ model: ninferState.profile.alias, reasoning_effort: effort })
      expect(request.body).not.toHaveProperty('thinking_budget')
      expect(request.body).not.toHaveProperty('reasoning_budget_tokens')
      await expect(chatCapsule(page)).toBeDisabled()
      await finish(page, request)
    }
    await page.getByRole('button', { name: '对话参数', exact: true }).click()
    await page.getByRole('checkbox', { name: '流式输出', exact: true }).uncheck()
    popup = await openReasoning(page)
    await slider.focus(); await page.keyboard.press('End')
    await page.keyboard.press('Escape')
    request = await context.send(page, 'NInfer 非流式高')
    expect(request.body).toMatchObject({ stream: false, reasoning_effort: 'xhigh' })
    await finish(page, request)
    await selectChatModel(page, 'unknown', true)
    await expect(popup.getByRole('slider')).toHaveCount(0)
    await page.keyboard.press('Escape')
    request = await context.send(page, '切模型不发送 NInfer 强度')
    expect(request.body).not.toHaveProperty('reasoning_effort')
    await finish(page, request)
    await selectChatModel(page, ninferSource.instance.id, true)
    await expect(slider).toBeVisible()
    await setChatReasoning(page, '中')
    await page.keyboard.press('Escape')
    request = await context.send(page, 'HF 源转换后保持同一模板档位')
    expect(request.body).toMatchObject({ model: ninferSource.profile.alias, reasoning_effort: 'medium' })
    await finish(page, request)
    await selectChatModel(page, ninferState.instance.id, true)
    await expect(slider).toBeVisible()
    await expect(page.locator('.chat-capsule-effort')).toHaveText('模板默认')
    await page.keyboard.press('Escape')
    await page.getByRole('button', { name: '对话历史', exact: true }).click()
    await page.setViewportSize({ width: 940, height: 700 })
    await page.waitForTimeout(300)
    popup = await openReasoning(page)
    await expect(slider).toBeVisible()
    await expect(popup).toHaveCSS('opacity', '1')
    await page.screenshot({ path: path.join(context.directory, 'ninfer-both-panels-940.png') })
    const capsule = (await chatCapsule(page).boundingBox())!, panel = (await popup.boundingBox())!
    expect(Math.abs(panel.x + panel.width / 2 - capsule.x - capsule.width / 2)).toBeLessThan(1)
    await page.keyboard.press('Escape')
    fs.writeFileSync(path.join(context.directory, 'ninfer-requests.json'), JSON.stringify(context.requests.map(request => request.body).filter(body => body.model.startsWith(ninferState.profile.alias)), null, 2))
    expect(context.errors).toEqual([])
  } finally { await context.cleanup() }
})

test('强度真实传参、重试、切模型默认、预设与跨重启会话快照', async () => {
  test.setTimeout(180_000)
  const context = await fixture()
  try {
    let { page } = await context.launch()
    const model = () => chatModel(page)
    await expect(page.locator('.chat-toolbar select')).toHaveCount(0)
    await expect(chatCapsule(page)).toBeVisible()
    let popup = await openReasoning(page)
    await setChatReasoning(page, '高')
    await page.keyboard.press('Escape')
    await expect(popup).toBeHidden()
    await expect(chatCapsule(page)).toBeFocused()
    let request = await context.send(page, 'Strata 强度')
    expect(request.body.reasoning_effort).toBe('xhigh')
    expect(request.body).not.toHaveProperty('reasoning_budget_tokens')
    await expect(model()).toBeDisabled()
    await expect(chatCapsule(page)).toBeDisabled()
    await finish(page, request)
    const firstId = (await management(page, '/api/conversations')).active_id
    expect((await management(page, `/api/conversations/${firstId}`)).settings.reasoningEffort).toBe('xhigh')
    await page.getByRole('button', { name: '重试回复', exact: true }).click()
    await expect.poll(() => context.requests.length).toBe(2)
    request = context.requests.at(-1)!
    expect(request.body.reasoning_effort).toBe('xhigh')
    await finish(page, request)

    await selectChatModel(page, 'qwen-llama')
    await expect(page.locator('.chat-capsule-effort')).toHaveText('模板默认')
    request = await context.send(page, '使用后端默认')
    expect(request.body).not.toHaveProperty('reasoning_effort')
    expect(request.body).not.toHaveProperty('chat_template_kwargs')
    await finish(page, request)
    popup = await openReasoning(page)
    await setChatReasoning(page, '关闭')
    await page.locator('.chat-session-title').click()
    request = await context.send(page, '关闭思考')
    expect(request.body.reasoning_effort).toBe('none')
    expect(request.body.chat_template_kwargs).toEqual({ enable_thinking: false })
    await finish(page, request)

    await selectChatModel(page, 'deepseek', true)
    popup = await openReasoning(page)
    await popup.getByRole('spinbutton', { name: '自定义推理强度', exact: true }).fill('17')
    await expect(page.locator('.chat-capsule-effort')).toHaveText('17')
    await page.keyboard.press('Escape')
    await page.getByRole('button', { name: '对话参数', exact: true }).click()
    await page.getByRole('checkbox', { name: '流式输出', exact: true }).uncheck()
    await page.getByRole('button', { name: '保存对话预设', exact: true }).click()
    await page.getByRole('textbox', { name: '预设名称', exact: true }).fill('DeepSeek 17')
    await page.getByRole('checkbox', { name: '用于新会话', exact: true }).check()
    await page.getByRole('dialog', { name: '保存对话预设', exact: true }).getByRole('button', { name: '保存预设', exact: true }).click()
    await expect(page.getByRole('dialog', { name: '保存对话预设', exact: true })).toHaveCount(0)
    const preset = (await management(page, '/api/chat-presets')).items[0]
    expect(preset.reasoningEffort).toBe(17)
    request = await context.send(page, '数值非流式')
    expect(request.body).toMatchObject({ stream: false, reasoning_effort: 17 })
    await finish(page, request)
    const snapshot = await management(page, `/api/conversations/${firstId}`)
    expect(snapshot.settings.reasoningEffort).toBe(17)
    await page.getByRole('button', { name: '新建会话', exact: true }).click()
    await expect(page.locator('.chat-capsule-effort')).toHaveText('17')
    await expect(page.getByRole('checkbox', { name: '流式输出', exact: true })).toBeChecked()
    await selectChatModel(page, 'unknown', true)
    await expect(page.locator('.chat-capsule-effort')).toHaveText('引擎默认')
    await expect(popup.getByRole('slider')).toHaveCount(0)
    await page.keyboard.press('Escape')
    await page.getByRole('combobox', { name: '对话预设', exact: true }).selectOption('')
    await page.getByRole('combobox', { name: '对话预设', exact: true }).selectOption(preset.id)
    request = await context.send(page, '未知模板仍能发送')
    expect(request.body).not.toHaveProperty('reasoning_effort')
    await finish(page, request)
    await page.getByRole('button', { name: '对话历史', exact: true }).click()
    await page.locator(`[data-history-id="${firstId}"] .chat-history-item`).click()
    await expect(page.locator('.chat-capsule-effort')).toHaveText('17')
    await context.quit()
    ;({ page } = await context.launch())
    await expect(page.locator('.chat-capsule-effort')).toHaveText('17')
    expect((await management(page, `/api/conversations/${firstId}`)).settings).toEqual(snapshot.settings)
    expect(context.errors).toEqual([])
  } finally { await context.cleanup() }
})

test('胶囊、浮层与长模型名在窄窗口和侧栏展开时可操作', async () => {
  test.setTimeout(120_000)
  const context = await fixture()
  try {
    const { page } = await context.launch()
    await page.emulateMedia({ reducedMotion: 'no-preference' })
    await page.setViewportSize({ width: 1280, height: 800 })
    const centered = async () => {
      const group = (await page.locator('.composer-context-controls').boundingBox())!, composer = (await page.locator('.chat-composer').boundingBox())!
      expect(group.x).toBeLessThan(composer.x + composer.width / 2)
      const capsule = (await chatCapsule(page).boundingBox())!, label = (await page.locator('.chat-capsule-model').boundingBox())!
      expect(Math.abs(label.y + label.height / 2 - capsule.y - capsule.height / 2)).toBeLessThan(1)
    }
    await centered()
    await page.screenshot({ path: path.join(context.directory, 'capsule-left-1280.png') })
    await chatCapsule(page).focus(); await page.keyboard.press('Enter')
    let popup = await openReasoning(page)
    const slider = popup.getByRole('slider', { name: '推理强度档位', exact: true })
    await expect(slider).toBeFocused()
    await page.keyboard.press('End')
    await expect(page.locator('.chat-capsule-effort')).toHaveText('高')
    await expect(popup).toHaveCSS('opacity', '1')
    expect(await popup.locator('.lucide-zap, .lucide-rotate-ccw').count()).toBe(0)
    expect(await page.locator('.chat-model-controls > button').count()).toBe(1)
    await settleReasoning(popup)
    const capsule = (await chatCapsule(page).boundingBox())!, panel = (await popup.boundingBox())!
    expect(Math.abs(panel.x + panel.width / 2 - capsule.x - capsule.width / 2)).toBeLessThan(1)
    await page.screenshot({ path: path.join(context.directory, 'reasoning-qwen-1280.png') })
    await setChatReasoning(page, '默认')
    await expect(page.locator('.chat-capsule-effort')).toHaveText('模板默认')
    await page.locator('.chat-session-title').click()
    await expect(popup).toBeHidden()
    await page.locator('.chat-capsule-effort').click()
    popup = await openReasoning(page)
    const model = chatModel(page)
    await model.click()
    await expect(popup.locator('.chat-model-view')).toHaveCSS('opacity', '1')
    await expect(model).toHaveJSProperty('value', 'qwen-strata')
    await page.screenshot({ path: path.join(context.directory, 'model-list-1280.png') })
    await page.keyboard.press('Escape')
    await expect(popup).toBeHidden()
    await expect(chatCapsule(page)).toBeFocused()
    await expect(chatModelList(page)).toHaveCount(0)
    await openReasoning(page)
    await model.click()
    await chatModelList(page).locator('[data-model-id="deepseek"]').click()
    await expect(popup).toBeVisible()
    await expect(model).toHaveJSProperty('value', 'deepseek')
    await popup.getByRole('spinbutton', { name: '自定义推理强度', exact: true }).fill('101')
    await expect(popup.getByRole('alert')).toHaveText('请输入 1–100 的整数。')
    await popup.getByRole('spinbutton', { name: '自定义推理强度', exact: true }).fill('100')
    await expect(page.locator('.chat-capsule-effort')).toHaveText('100')
    await expect(popup.getByRole('alert')).toHaveCount(0)
    await expect(popup).toHaveCSS('opacity', '1')
    await settleReasoning(popup)
    await page.screenshot({ path: path.join(context.directory, 'reasoning-deepseek-1280.png') })
    await page.keyboard.press('Escape')
    await selectChatModel(page, 'qwen-strata')
    await page.getByRole('button', { name: '对话历史', exact: true }).click()
    await page.getByRole('button', { name: '对话参数', exact: true }).click()
    await page.setViewportSize({ width: 940, height: 700 })
    await page.getByRole('textbox', { name: '对话消息', exact: true }).fill('窄窗口下发送')
    await page.waitForTimeout(300)
    popup = await openReasoning(page)
    await expect(popup).toHaveCSS('opacity', '1')
    await centered()
    await page.screenshot({ path: path.join(context.directory, 'reasoning-both-panels-940.png') })
    const composer = (await page.locator('.chat-composer').boundingBox())!, send = (await page.getByRole('button', { name: '发送消息', exact: true }).boundingBox())!, control = (await page.locator('.chat-model-controls').boundingBox())!, pop = (await popup.boundingBox())!
    expect(send.x + send.width).toBeLessThanOrEqual(composer.x + composer.width)
    expect(control.x + control.width).toBeLessThanOrEqual(send.x)
    expect(control.width).toBeGreaterThan(100)
    expect(pop.x).toBeGreaterThanOrEqual(0)
    expect(pop.x + pop.width).toBeLessThanOrEqual(940)
    expect(pop.y).toBeGreaterThanOrEqual(0)
    await page.keyboard.press('Escape')
    await expect(popup).toBeHidden()
    await expect(page.getByRole('complementary', { name: '对话历史', exact: true })).toBeVisible()
    await expect(page.getByRole('complementary', { name: '对话参数', exact: true })).toBeVisible()
    const request = await context.send(page, '窄窗口下发送')
    await finish(page, request)
    await page.getByRole('textbox', { name: '对话消息', exact: true, includeHidden: true }).focus()
    await page.keyboard.press('Escape')
    await expect(page.getByRole('complementary', { name: '对话历史', exact: true })).toBeHidden()
    await expect(page.getByRole('complementary', { name: '对话参数', exact: true })).toBeHidden()
    expect(context.errors).toEqual([])
  } finally { await context.cleanup() }
})
