import { test, expect, _electron as electron, type Page } from '@playwright/test'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { pathToFileURL } from 'node:url'
import { pinNavigation } from './navigation'
import { disableWebSearch } from './chat-controls'

const root = path.resolve(__dirname, '..')
const artifacts = path.join(root, '.artifacts/codex/chat-turn-update-20261027/resources-flow')
async function management(page: Page, route: string, method = 'GET', body?: unknown) {
  return page.evaluate(async ({ route, method, body }) => {
    const { baseUrl, token } = await window.modloader!.connection()
    const response = await fetch(baseUrl + route, { method, headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) })
    if (!response.ok) throw new Error(await response.text())
    return response.status === 204 ? null : response.json()
  }, { route, method, body })
}

test('本地附件从真实工具到回复、历史、授权与整轮计量', async () => {
  test.setTimeout(120_000)
  fs.mkdirSync(artifacts, { recursive: true })
  const directory = fs.mkdtempSync(path.join(artifacts, 'app-')), workspace = path.join(directory, 'workspace')
  fs.mkdirSync(workspace); fs.writeFileSync(path.join(workspace, '报告.md'), '# 本地资料\n保留在授权目录。\n', 'utf8')
  const temp = path.join(directory, 'temp'); fs.mkdirSync(temp)
  const app = await electron.launch({ args: [root], env: { ...process.env, TEMP: temp, TMP: temp, MODLOADER_DATA_DIR: path.join(directory, 'data'), MODLOADER_SENSOR_AUTO_START: '0', MODLOADER_DEV_URL: pathToFileURL(path.join(root, 'dist/index.html')).href }, timeout: 60_000 })
  try {
    const page = await app.firstWindow(), errors: string[] = [], requests: any[] = []
    page.setDefaultTimeout(15_000)
    page.on('pageerror', error => errors.push(error.message))
    await pinNavigation(page)
    await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
    const picture = await page.evaluate(() => {
      const canvas = document.createElement('canvas'); canvas.width = 640; canvas.height = 260
      const context = canvas.getContext('2d')!; context.fillStyle = '#ecf2ff'; context.fillRect(0, 0, 640, 260)
      context.fillStyle = '#4867df'; context.fillRect(30, 30, 110, 200)
      context.fillStyle = '#b9c8f0'; context.fillRect(164, 30, 444, 83); context.fillRect(164, 137, 210, 93); context.fillRect(398, 137, 210, 93)
      return canvas.toDataURL('image/png').split(',')[1]
    })
    fs.writeFileSync(path.join(workspace, '示意.png'), Buffer.from(picture, 'base64'))
    const instance = { id: 'resource-model', alias: '附件测试模型', backend: 'llama_cpp', profile_id: 'p1', model_id: 'm1', runtime_id: 'r1', state: 'ready', active_requests: 0, capabilities: { input: ['text'], tools: true } }
    await page.route('**/api/state', async route => {
      const response = await route.fetch(), state = await response.json()
      await route.fulfill({ response, json: { ...state, instances: [instance], models: [{ id: 'm1', name: instance.alias, architecture: 'qwen35', format: 'gguf', files: [], modalities: ['text'], source: {}, metadata: {}, warnings: [], managed: false, size_bytes: 0, path: '/fixture/model.gguf' }],
        profiles: [{ id: 'p1', model_id: 'm1', runtime_id: 'r1', backend: 'llama_cpp', alias: instance.alias, name: '默认预设', parameters: { context_length: 4096 }, generation: {}, extra_args: [], revision: 1 }],
        runtimes: [{ id: 'r1', backend: 'llama_cpp', name: '测试运行环境', version: 'fixture', path: directory, managed: false, metadata: { supported_flags: ['--jinja'] } }],
      } })
    })
    await page.route('**/api/instances/*/telemetry', route => route.fulfill({ json: { groups: [], unavailable: [], live_generation: { context_capacity: 4096 } } }))
    await page.route('**/api/chat', async route => {
      const body = route.request().postDataJSON(); requests.push(body)
      expect(body.tools.some((tool: any) => tool.function.name === 'attach_file')).toBe(true)
      const results = body.messages.filter((item: any) => item.role === 'tool').map((item: any) => JSON.parse(item.content))
      const message = results.length ? { role: 'assistant', content: `已找到资料。\n\n[报告](modloader-resource:${results[0].resource.id})\n\n![示意](modloader-resource:${results[1].resource.id})` }
        : { role: 'assistant', content: null, reasoning_content: '引用目录中的文档与图片。', tool_calls: ['报告.md', '示意.png'].map((name, index) => ({ id: `attach-${index}`, type: 'function', function: { name: 'attach_file', arguments: JSON.stringify({ path: name }) } })) }
      await route.fulfill({ json: { choices: [{ message, finish_reason: results.length ? 'stop' : 'tool_calls' }], usage: { prompt_tokens: results.length ? 140 : 100, completion_tokens: results.length ? 10 : 20 }, timings: { predicted_n: results.length ? 10 : 20, predicted_ms: results.length ? 100 : 200 } } })
    })
    await app.evaluate(({ ipcMain }, workspace) => { ipcMain.removeHandler('pick-path'); ipcMain.handle('pick-path', () => workspace) }, workspace)
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    const chat = () => page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '测试对话', exact: true }).click()
    await chat(); await disableWebSearch(page)
    await page.getByRole('button', { name: '对话参数', exact: true }).click()
    await page.getByRole('button', { name: '选择本地工具目录', exact: true }).click()
    await expect(page.getByRole('textbox', { name: '本地工具目录' })).toHaveValue(workspace)
    await page.getByRole('button', { name: '收起对话参数', exact: true }).click()
    await page.getByRole('textbox', { name: '对话消息', exact: true, includeHidden: true }).fill('展示本地报告和示意图。')
    await page.getByRole('button', { name: '发送消息', exact: true }).click()
    const response = page.locator('.chat-message.assistant').last()
    await expect(response.getByRole('button', { name: '查看本轮 2 次模型请求统计', exact: true })).toBeVisible()
    await expect(response.locator('.chat-local-resource')).toHaveCount(2)
    await expect(response.locator('.chat-resource-image img')).toHaveJSProperty('naturalWidth', 640)
    expect(requests).toHaveLength(2)
    expect(JSON.stringify(requests[1])).not.toContain('data:image')
    const active = (await management(page, '/api/conversations')).active_id
    await expect.poll(async () => (await management(page, `/api/conversations/${active}`)).messages.at(-1).resources?.length).toBe(2)
    const saved = (await management(page, `/api/conversations/${active}`)).messages.at(-1)
    expect(saved.resources.map((value: any) => value.kind)).toEqual(['file', 'image'])
    expect(saved.turn.input.value).toBe(240); expect(saved.turn.output.value).toBe(30)
    expect(saved.turn.decode.value).toBeCloseTo(100); expect(saved.usage.prompt_tokens).toBe(140)
    expect(saved.requests[0].labels).toEqual(expect.arrayContaining(['引用文件', '引用图片']))
    expect(saved.requests[1].labels).toEqual(['工具结果'])
    await response.getByRole('button', { name: '附件详情：报告.md', exact: true }).click({ position: { x: 200, y: 25 } })
    await expect(page.getByRole('dialog', { name: '附件详情', exact: true })).toContainText(workspace)
    await page.getByRole('button', { name: '关闭', exact: true }).click()
    await response.screenshot({ path: path.join(directory, 'chat-local-resources.png') })
    await response.getByRole('button', { name: '查看本轮 2 次模型请求统计', exact: true }).click()
    await expect(page.getByRole('button', { name: '选择请求 #1', exact: true })).toBeVisible()
    await expect(page.getByRole('button', { name: '选择请求 #1', exact: true }).getByText('引用图片', { exact: true })).toBeVisible()
    await page.screenshot({ path: path.join(directory, 'resource-request-ledger.png') })
    await page.reload(); await expect(page.getByText('服务在线', { exact: true })).toBeVisible(); await pinNavigation(page); await chat()
    await expect(page.locator('.chat-resource-image img')).toHaveJSProperty('naturalWidth', 640)
    await page.getByRole('button', { name: '对话参数', exact: true }).click()
    await page.getByRole('button', { name: '撤销本地目录授权', exact: true }).click()
    await expect(page.locator('.chat-resource-image img')).toHaveCount(0)
    await expect.poll(async () => (await management(page, `/api/conversations/${active}`)).settings.toolDirectory).toBe('')
    const denied = await page.evaluate(async id => {
      const { baseUrl, token } = await window.modloader!.connection()
      return (await fetch(`${baseUrl}/api/chat-tools/resources/${id}/content`, { headers: { Authorization: `Bearer ${token}` } })).status
    }, saved.resources[1].id)
    expect(denied).toBe(403)
    await expect(page.evaluate(id => window.modloader!.localResourceAction(id, 'open'), saved.resources[0].id)).rejects.toThrow('授权已撤回')
    expect(fs.readFileSync(path.join(workspace, '报告.md'), 'utf8')).toContain('保留在授权目录。')
    expect(errors).toEqual([])
    console.log('Local resource integration screenshots:', directory)
  } finally {
    await Promise.all(app.windows().map(page => page.unrouteAll({ behavior: 'ignoreErrors' })))
    await app.close()
  }
})
