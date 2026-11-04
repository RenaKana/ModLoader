import { test, expect, _electron as electron, type ElectronApplication, type Page } from '@playwright/test'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { pinNavigation } from './navigation'
import { selectChatModel } from './chat-controls'

interface Preset { id: string; name: string; system: string; toolsDraft: string; toolConcurrency?: number; created_at: string; updated_at: string }
interface Catalog { items: Preset[]; default_id: string | null }
interface Settings { system: string; toolsDraft: string; stream: boolean; temperature?: number; maxTokens?: number; presetId?: string; webSearch?: boolean; webSearchMode?: 'agent' | 'retrieve'; webSearchRounds?: number; toolConcurrency?: number }
interface Conversation { id: string; settings: Settings; draft: { text: string }; selection: { model_id: string } | null }
interface ConversationList { items: Array<{ id: string }>; active_id: string | null }

const root = path.resolve(__dirname, '..')
const instances = ['first', 'second'].map(id => ({ id: `preset-${id}`, profile_id: `profile-${id}`, model_id: `model-${id}`, alias: `预设模型 ${id}`, backend: 'llama_cpp', runtime_id: 'preset-runtime', state: 'ready', active_requests: 0, capabilities: { input: ['text'], tools: true } }))
const system = '跨模型复用的系统提示词。\n保留第二行。'
const tools = '[\n  {"type": "function", "function": {"name": "lookup", "description": "保留源文本"}}\n]\n'

async function management<T>(page: Page, target: string, method = 'GET', body?: unknown): Promise<T> {
  return page.evaluate(async ({ target, method, body }) => {
    const connection = await (window as any).modloader.connection()
    const response = await fetch(`${connection.baseUrl}${target}`, { method, headers: { Authorization: `Bearer ${connection.token}`, 'Content-Type': 'application/json' }, ...(body === undefined ? {} : { body: JSON.stringify(body) }) })
    if (!response.ok) throw new Error(`${method} ${target}: ${response.status} ${await response.text()}`)
    return response.status === 204 ? null : response.json()
  }, { target, method, body })
}

async function chat(page: Page) {
  await pinNavigation(page)
  await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '测试对话', exact: true }).click()
}

async function options(page: Page) {
  const panel = page.getByRole('complementary', { name: '对话参数', exact: true })
  if (!await panel.isVisible()) await page.getByRole('button', { name: '对话参数', exact: true }).click()
  await expect(panel).toBeVisible()
  return panel
}

async function fillDraft(page: Page, text: string) {
  const preview = page.getByRole('region', { name: '输入内容预览', exact: true })
  if (await preview.isVisible()) await preview.click()
  await page.getByRole('textbox', { name: '对话消息', exact: true }).fill(text)
}

async function savedConversation(page: Page) {
  await expect(page.getByText('已保存', { exact: true })).toBeVisible()
  await expect.poll(async () => (await management<ConversationList>(page, '/api/conversations')).active_id).not.toBeNull()
  const id = (await management<ConversationList>(page, '/api/conversations')).active_id!
  return management<Conversation>(page, `/api/conversations/${id}`)
}

function fixture() {
  const directory = fs.mkdtempSync(path.join(process.env.MODLOADER_TEST_ARTIFACT_DIR || path.join(root, 'work'), 'ui-chat-presets-'))
  const errors: string[] = []
  let application: ElectronApplication | undefined
  return {
    directory, errors,
    async launch() {
      const executablePath = process.env.MODLOADER_TEST_EXECUTABLE
      application = await electron.launch({ ...(executablePath ? { executablePath, args: [] } : { args: [root] }), env: { ...process.env, MODLOADER_DATA_DIR: directory, MODLOADER_SENSOR_AUTO_START: '0' }, timeout: 60_000 })
      const page = await application.firstWindow()
      page.on('pageerror', error => errors.push(error.message))
      await pinNavigation(page)
      await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
      await page.route('**/api/state', async route => {
        const original = await route.fetch()
        const state = await original.json()
        await route.fulfill({ response: original, json: { ...state, instances } })
      })
      await page.getByRole('button', { name: '刷新状态', exact: true }).click()
      await chat(page)
      await expect(page.getByRole('textbox', { name: '对话消息', exact: true, includeHidden: true })).toBeEnabled()
      return { application, page }
    },
    async quit() {
      const current = application!
      await Promise.all(current.windows().map(page => page.unrouteAll({ behavior: 'wait' })))
      const closed = current.waitForEvent('close', { timeout: 30_000 })
      await current.evaluate(({ Menu }) => Menu.getApplicationMenu()!.items.find(item => item.label === '文件')!.submenu!.items.find(item => item.label === '退出 ModLoader')!.click())
      await closed
      application = undefined
    },
    async cleanup() {
      if (application) {
        await Promise.all(application.windows().map(page => page.unrouteAll({ behavior: 'wait' })))
        await application.close()
      }
      console.log('Chat preset fixture and screenshots:', directory)
    },
  }
}

test('保存预设弹窗覆盖视口，三按钮不被参数栏裁切且关闭后恢复焦点', async () => {
  test.setTimeout(100_000)
  const context = fixture()
  try {
    const { page } = await context.launch()
    await page.emulateMedia({ reducedMotion: 'no-preference' })
    const id = await page.evaluate(() => crypto.randomUUID())
    await management(page, '/api/chat-presets', 'POST', { id, name: '数学大师', system, toolsDraft: tools, make_default: true })
    await page.reload(); await chat(page)
    const trigger = page.getByRole('button', { name: '保存对话预设', exact: true })
    const dialog = page.getByRole('dialog', { name: '保存对话预设', exact: true })
    for (const [index, viewport] of [{ width: 1280, height: 800 }, { width: 900, height: 650 }, { width: 360, height: 720 }].entries()) {
      await page.setViewportSize(viewport)
      await page.evaluate(scale => document.documentElement.style.setProperty('--font-scale', String(scale)), index ? 1.3 : 1)
      await options(page)
      await trigger.click()
      await expect(dialog.getByRole('textbox', { name: '预设名称', exact: true })).toBeFocused()
      await expect(dialog.getByRole('button', { name: '另存为', exact: true })).toBeVisible()
      await expect(dialog).toHaveCSS('opacity', '1')
      await page.screenshot({ path: path.join(context.directory, `preset-dialog-${viewport.width}.png`) })
      const overlay = await page.locator('.modal-overlay').boundingBox()
      expect(overlay!.x).toBeCloseTo(0)
      expect(overlay!.y).toBeCloseTo(0)
      expect(overlay!.width).toBeCloseTo(viewport.width)
      expect(overlay!.height).toBeCloseTo(viewport.height)
      const bounds = await dialog.boundingBox()
      expect(bounds!.x + bounds!.width / 2).toBeCloseTo(viewport.width / 2)
      for (const button of await dialog.locator('.modal-footer button').all()) {
        const box = (await button.boundingBox())!
        expect(box.x).toBeGreaterThanOrEqual(bounds!.x)
        expect(box.x + box.width).toBeLessThanOrEqual(bounds!.x + bounds!.width)
        expect(box.y + box.height).toBeLessThanOrEqual(bounds!.y + bounds!.height)
        expect(await button.evaluate(node => {
          const rect = node.getBoundingClientRect()
          return node.contains(document.elementFromPoint(rect.x + rect.width / 2, rect.y + rect.height / 2))
        })).toBe(true)
      }
      await dialog.getByRole('button', { name: '更新预设', exact: true }).focus()
      await page.keyboard.press('Tab')
      await expect(dialog.getByRole('button', { name: '关闭', exact: true })).toBeFocused()
      if (index === 0) await dialog.getByRole('button', { name: '取消', exact: true }).click()
      else if (index === 1) {
        // Escape on an icon first dismisses its tooltip; test modal dismissal from the field.
        await dialog.getByRole('textbox', { name: '预设名称', exact: true }).click()
        await page.keyboard.press('Escape')
      }
      else await dialog.getByRole('button', { name: '关闭', exact: true }).click()
      await expect(dialog).toHaveCount(0)
      await expect(trigger).toBeFocused()
    }
    expect(context.errors).toEqual([])
  } finally { await context.cleanup() }
})

test('命名预设与默认跨重启保存，编辑、模型切换和更新保留各会话快照', async () => {
  test.setTimeout(150_000)
  const context = fixture()
  try {
    let { page } = await context.launch()
    await options(page)
    const concurrency = page.getByRole('spinbutton', { name: '工具并行数', exact: true })
    await expect(concurrency).toHaveValue('3')
    expect(await concurrency.getAttribute('max')).toBeNull()
    await concurrency.fill('5')
    await page.getByRole('textbox', { name: '系统提示词', exact: true }).fill(system)
    await page.getByRole('textbox', { name: '工具定义', exact: true }).fill('{}')
    await page.getByRole('button', { name: '保存对话预设', exact: true }).click()
    await page.getByRole('textbox', { name: '预设名称', exact: true }).fill('通用工具')
    await page.getByRole('checkbox', { name: '用于新会话', exact: true }).check()
    await page.getByRole('dialog').getByRole('button', { name: '保存预设', exact: true }).click()
    await expect(page.getByRole('dialog').getByRole('alert')).toContainText('工具定义须为 JSON 数组')
    expect((await management<Catalog>(page, '/api/chat-presets')).items).toEqual([])
    await page.getByRole('dialog').getByRole('button', { name: '取消', exact: true }).click()
    await page.getByRole('textbox', { name: '工具定义', exact: true }).fill(tools)
    await page.getByRole('spinbutton', { name: '温度', exact: true }).fill('1.1')
    await page.getByRole('spinbutton', { name: '最大输出 Token', exact: true }).fill('777')
    await page.getByRole('checkbox', { name: '流式输出', exact: true }).uncheck()
    await page.getByRole('button', { name: '保存对话预设', exact: true }).click()
    await page.getByRole('textbox', { name: '预设名称', exact: true }).fill('通用工具')
    await page.getByRole('checkbox', { name: '用于新会话', exact: true }).check()
    await page.getByRole('dialog').screenshot({ path: path.join(context.directory, 'preset-first-save.png') })
    await page.getByRole('dialog').getByRole('button', { name: '保存预设', exact: true }).click()
    await expect(page.getByRole('dialog')).toHaveCount(0)
    let catalog = await management<Catalog>(page, '/api/chat-presets')
    const id = catalog.items[0].id
    expect(catalog.default_id).toBe(id)
    expect(catalog.items[0]).toMatchObject({ name: '通用工具', system, toolsDraft: tools, toolConcurrency: 5 })
    expect(Object.keys(catalog.items[0]).sort()).toEqual(['created_at', 'id', 'name', 'system', 'toolConcurrency', 'toolsDraft', 'updated_at'])
    await fillDraft(page, '保留旧会话草稿')
    const first = await savedConversation(page)
    expect(first.settings).toEqual({ system, toolsDraft: tools, stream: false, temperature: 1.1, maxTokens: 777, presetId: id, webSearch: true, webSearchMode: 'agent', webSearchRounds: 25, toolConcurrency: 5 })

    await page.getByRole('button', { name: '新建会话', exact: true }).click()
    await expect(page.getByRole('textbox', { name: '系统提示词', exact: true })).toHaveValue(system)
    await expect(page.getByRole('textbox', { name: '工具定义', exact: true })).toHaveValue(tools)
    await expect(page.getByRole('spinbutton', { name: '温度', exact: true })).toHaveValue('')
    await expect(page.getByRole('spinbutton', { name: '最大输出 Token', exact: true })).toHaveValue('')
    await expect(page.getByRole('checkbox', { name: '流式输出', exact: true })).toBeChecked()
    await expect(concurrency).toHaveValue('5')
    await concurrency.fill('2')
    await page.getByRole('textbox', { name: '系统提示词', exact: true }).fill('新版系统提示')
    expect((await management<Catalog>(page, '/api/chat-presets')).items[0].system).toBe(system)
    await selectChatModel(page, instances[1].id)
    await expect(page.getByRole('textbox', { name: '系统提示词', exact: true })).toHaveValue('新版系统提示')
    await expect(page.getByRole('textbox', { name: '工具定义', exact: true })).toHaveValue(tools)
    await fillDraft(page, '保存新版会话')
    await savedConversation(page)
    await page.getByRole('button', { name: '保存对话预设', exact: true }).click()
    await page.getByRole('textbox', { name: '预设名称', exact: true }).fill('通用工具 v2')
    await page.getByRole('dialog').getByRole('button', { name: '更新预设', exact: true }).click()
    await expect(page.getByRole('dialog')).toHaveCount(0)
    catalog = await management<Catalog>(page, '/api/chat-presets')
    expect(catalog.items[0]).toMatchObject({ id, name: '通用工具 v2', system: '新版系统提示', toolConcurrency: 2 })
    expect((await management<Conversation>(page, `/api/conversations/${first.id}`)).settings).toEqual(first.settings)

    await page.getByRole('button', { name: '对话历史', exact: true }).click()
    await page.locator(`[data-history-id="${first.id}"] .chat-history-item`).click()
    await expect(concurrency).toHaveValue('5')
    await expect(page.getByRole('textbox', { name: '系统提示词', exact: true })).toHaveValue(system)
    await expect(page.getByRole('textbox', { name: '工具定义', exact: true })).toHaveValue(tools)
    await selectChatModel(page, instances[1].id)
    await expect(page.getByRole('textbox', { name: '系统提示词', exact: true })).toHaveValue(system)
    await page.getByRole('combobox', { name: '对话预设', exact: true }).selectOption('')
    await page.getByRole('combobox', { name: '对话预设', exact: true }).selectOption(id)
    await expect(concurrency).toHaveValue('2')
    await expect(page.getByRole('textbox', { name: '系统提示词', exact: true })).toHaveValue('新版系统提示')
    await expect(page.getByRole('spinbutton', { name: '温度', exact: true })).toHaveValue('1.1')
    await expect(page.getByRole('spinbutton', { name: '最大输出 Token', exact: true })).toHaveValue('777')
    await expect(page.getByRole('checkbox', { name: '流式输出', exact: true })).not.toBeChecked()
    await page.getByRole('textbox', { name: '系统提示词', exact: true }).fill('只属于旧会话')
    await page.screenshot({ path: path.join(context.directory, 'preset-picker-snapshot.png') })
    // Quit immediately after editing: the native close path must flush the snapshot.
    await context.quit()
    ;({ page } = await context.launch())
    await options(page)
    await expect(page.getByRole('textbox', { name: '系统提示词', exact: true })).toHaveValue('只属于旧会话')
    await expect(page.getByRole('spinbutton', { name: '工具并行数', exact: true })).toHaveValue('2')
    expect((await management<Catalog>(page, '/api/chat-presets')).default_id).toBe(id)
    await page.getByRole('button', { name: '新建会话', exact: true }).click()
    await expect(page.getByRole('textbox', { name: '系统提示词', exact: true })).toHaveValue('新版系统提示')
    await expect(page.getByRole('spinbutton', { name: '工具并行数', exact: true })).toHaveValue('2')
    await page.getByRole('button', { name: '保存对话预设', exact: true }).click()
    await expect(page.getByRole('checkbox', { name: '用于新会话', exact: true })).toBeChecked()
    await page.getByRole('checkbox', { name: '用于新会话', exact: true }).uncheck()
    await page.getByRole('textbox', { name: '预设名称', exact: true }).fill('取消后不应保存')
    await page.getByRole('dialog').getByRole('button', { name: '取消', exact: true }).click()
    catalog = await management<Catalog>(page, '/api/chat-presets')
    expect(catalog.default_id).toBe(id)
    expect(catalog.items).toHaveLength(1)
    expect(catalog.items[0].name).toBe('通用工具 v2')
    await page.getByRole('button', { name: '保存对话预设', exact: true }).click()
    await expect(page.getByRole('checkbox', { name: '用于新会话', exact: true })).toBeChecked()
    await page.getByRole('textbox', { name: '预设名称', exact: true }).fill('独立副本')
    await page.getByRole('dialog').getByRole('button', { name: '另存为', exact: true }).click()
    await expect(page.getByRole('dialog')).toHaveCount(0)
    catalog = await management<Catalog>(page, '/api/chat-presets')
    expect(catalog.items).toHaveLength(2)
    expect(catalog.items[0]).toMatchObject({ id, name: '通用工具 v2' })
    const copy = catalog.items[1]
    expect(catalog.default_id).toBe(copy.id)
    await page.getByRole('combobox', { name: '对话预设', exact: true }).selectOption(id)
    await page.getByRole('button', { name: '保存对话预设', exact: true }).click()
    await expect(page.getByRole('checkbox', { name: '用于新会话', exact: true })).not.toBeChecked()
    await page.getByRole('dialog').getByRole('button', { name: '更新预设', exact: true }).click()
    await expect(page.getByRole('dialog')).toHaveCount(0)
    expect((await management<Catalog>(page, '/api/chat-presets')).default_id).toBe(copy.id)
    expect(context.errors).toEqual([])
  } finally { await context.cleanup() }
})

test('默认目录加载失败可重试，启动和删除最后会话使用默认快照', async () => {
  test.setTimeout(100_000)
  const context = fixture()
  try {
    const { page } = await context.launch()
    const id = await page.evaluate(() => crypto.randomUUID())
    await management(page, '/api/chat-presets', 'POST', { id, name: '启动默认', system, toolsDraft: tools, make_default: true })
    let fail = true
    await page.route('**/api/chat-presets', route => fail && route.request().method() === 'GET'
      ? route.fulfill({ status: 503, json: { detail: '预设目录暂时不可读' } }) : route.continue())
    await page.reload(); await chat(page)
    await expect(page.getByRole('alert')).toContainText('读取对话预设失败')
    await expect(page.getByRole('textbox', { name: '对话消息', exact: true })).toBeDisabled()
    await expect(page.getByRole('button', { name: '新建会话', exact: true })).toBeDisabled()
    fail = false
    await page.getByRole('button', { name: '重新读取预设', exact: true }).click()
    await expect(page.getByRole('textbox', { name: '对话消息', exact: true })).toBeEnabled()
    await options(page)
    await expect(page.getByRole('textbox', { name: '系统提示词', exact: true })).toHaveValue(system)
    await expect(page.getByRole('textbox', { name: '工具定义', exact: true })).toHaveValue(tools)
    await page.getByRole('textbox', { name: '系统提示词', exact: true }).fill('被删除的会话快照')
    await fillDraft(page, '删除最后一条草稿')
    await savedConversation(page)
    await page.getByRole('button', { name: '删除会话', exact: true }).click()
    await page.getByRole('dialog').getByRole('button', { name: '删除会话', exact: true }).click()
    await expect(page.getByRole('dialog')).toHaveCount(0)
    await expect(page.getByRole('textbox', { name: '系统提示词', exact: true })).toHaveValue(system)
    await expect(page.getByRole('textbox', { name: '工具定义', exact: true })).toHaveValue(tools)
    expect((await management<ConversationList>(page, '/api/conversations')).items).toEqual([])
    await page.getByRole('button', { name: '保存对话预设', exact: true }).click()
    await expect(page.getByRole('checkbox', { name: '用于新会话', exact: true })).toBeChecked()
    await page.getByRole('checkbox', { name: '用于新会话', exact: true }).uncheck()
    expect((await management<Catalog>(page, '/api/chat-presets')).default_id).toBe(id)
    await page.getByRole('dialog').getByRole('button', { name: '更新预设', exact: true }).click()
    await expect(page.getByRole('dialog')).toHaveCount(0)
    expect((await management<Catalog>(page, '/api/chat-presets')).default_id).toBeNull()
    await expect(page.getByRole('textbox', { name: '系统提示词', exact: true })).toHaveValue(system)
    await page.getByRole('button', { name: '新建会话', exact: true }).click()
    await expect(page.getByRole('textbox', { name: '系统提示词', exact: true })).toHaveValue('')
    expect(context.errors).toEqual([])
  } finally { await context.cleanup() }
})

test('预设目录失败不阻断旧会话，重试保留当前编辑且不重读历史', async () => {
  test.setTimeout(100_000)
  const context = fixture()
  try {
    const { page } = await context.launch()
    const ids = await page.evaluate(() => [crypto.randomUUID(), crypto.randomUUID(), crypto.randomUUID()])
    await management(page, '/api/chat-presets', 'POST', { id: ids[0], name: '恢复后默认', system, toolsDraft: tools, make_default: true })
    for (const [index, id] of ids.slice(1).entries()) {
      await management(page, '/api/conversations', 'POST', {
        id, title: `旧会话 ${index + 1}`, selection: { profile_id: instances[index].profile_id, model_id: instances[index].model_id, alias: instances[index].alias },
        draft: { text: `旧草稿 ${index + 1}`, images: [] }, settings: { system: `旧系统 ${index + 1}`, toolsDraft: '[]', stream: false },
      })
    }
    await management(page, `/api/conversations/${ids[1]}`, 'PATCH', { activate: true })
    let fail = true
    await page.route('**/api/chat-presets', route => fail && route.request().method() === 'GET'
      ? route.fulfill({ status: 503, json: { detail: '预设目录暂时不可读' } }) : route.continue())
    await page.reload(); await chat(page)
    await expect(page.getByRole('alert')).toContainText('读取对话预设失败')
    await expect(page.getByRole('textbox', { name: '对话消息', exact: true, includeHidden: true })).toBeEnabled()
    await expect(page.getByRole('button', { name: '新建会话', exact: true })).toBeDisabled()
    await options(page)
    await expect(page.getByRole('textbox', { name: '系统提示词', exact: true })).toHaveValue('旧系统 1')
    await expect(page.getByRole('combobox', { name: '对话预设', exact: true })).toBeDisabled()
    await expect(page.getByRole('button', { name: '保存对话预设', exact: true })).toBeDisabled()
    await fillDraft(page, '目录失败时照常保存草稿')
    await page.getByRole('textbox', { name: '系统提示词', exact: true }).fill('目录失败时照常保存系统词')
    const saved = await savedConversation(page)
    expect(saved.id).toBe(ids[1])
    expect(saved.draft.text).toBe('目录失败时照常保存草稿')
    expect(saved.settings.system).toBe('目录失败时照常保存系统词')
    await page.getByRole('button', { name: '对话历史', exact: true }).click()
    await page.locator(`[data-history-id="${ids[2]}"] .chat-history-item`).click()
    await expect(page.getByRole('textbox', { name: '系统提示词', exact: true })).toHaveValue('旧系统 2')
    await selectChatModel(page, instances[0].id)
    await expect(page.getByRole('textbox', { name: '系统提示词', exact: true })).toHaveValue('旧系统 2')
    await fillDraft(page, '预设重试前的新草稿')
    await page.getByRole('textbox', { name: '系统提示词', exact: true }).fill('预设重试前的新系统词')
    let historyReads = 0
    page.on('request', request => { if (request.method() === 'GET' && request.url().includes('/api/conversations')) historyReads++ })
    fail = false
    await page.getByRole('button', { name: '重新读取预设', exact: true }).click()
    await expect(page.getByRole('combobox', { name: '对话预设', exact: true })).toBeEnabled()
    await expect(page.getByRole('button', { name: '新建会话', exact: true })).toBeEnabled()
    await expect(page.getByRole('textbox', { name: '系统提示词', exact: true })).toHaveValue('预设重试前的新系统词')
    await expect(page.getByRole('textbox', { name: '对话消息', exact: true, includeHidden: true })).toHaveValue('预设重试前的新草稿')
    expect(historyReads).toBe(0)
    await page.getByRole('button', { name: '新建会话', exact: true }).click()
    await expect(page.getByRole('textbox', { name: '系统提示词', exact: true })).toHaveValue(system)
    expect(context.errors).toEqual([])
  } finally { await context.cleanup() }
})
