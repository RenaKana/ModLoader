import { test, expect, _electron as electron } from '@playwright/test'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { pinNavigation } from './navigation'

test('模型加载等待：按配置显示、阶段更新与监控手选不被就绪实例抢回', async () => {
  test.setTimeout(90_000)
  const root = path.resolve(__dirname, '..')
  const directory = fs.mkdtempSync(path.join(root, 'work', 'ui-loading-feedback-'))
  const app = await electron.launch({
    ...(process.env.MODLOADER_TEST_EXECUTABLE ? { executablePath: process.env.MODLOADER_TEST_EXECUTABLE, args: [] } : { args: [root] }),
    env: { ...process.env, MODLOADER_DATA_DIR: path.join(directory, 'data'), MODLOADER_SENSOR_AUTO_START: '0' }, timeout: 60_000,
  })
  const page = await app.firstWindow(), errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  let releaseStart: (() => void) | undefined
  let stateReads = 0, startingReads = 0
  const startGate = new Promise<void>(resolve => { releaseStart = resolve })
  try {
    await pinNavigation(page)
    await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
    const baseline = await page.evaluate(async () => {
      const { baseUrl, token } = await window.modloader!.connection()
      return (await fetch(baseUrl + '/api/state', { headers: { Authorization: 'Bearer ' + token } })).json()
    })
    const model = { id: 'load-model', name: '等待反馈模型', path: 'D:\\Fixture\\model.gguf', format: 'gguf', architecture: 'llama', quantization: 'Q4_K_M', files: [], size_bytes: 4096, modalities: ['text'], metadata: {}, source: {}, managed: false, warnings: [] }
    const runtime = { id: 'load-runtime', backend: 'llama_cpp', name: 'Fixture', path: 'D:\\Fixture\\runtime', managed: false, metadata: {} }
    const profiles = ['正在操作配置', '另一套配置'].map((name, index) => ({ id: 'load-profile-' + index, model_id: model.id, name, backend: 'llama_cpp', runtime_id: runtime.id, alias: 'load-' + index, parameters: {}, generation: {}, extra_args: [], revision: 1 }))
    const ready = { id: 'other-ready', model_id: 'other-model', profile_id: 'other-profile', profile_revision: 1, alias: '已有就绪实例', backend: 'llama_cpp', runtime_id: runtime.id, port: 0, started_at: new Date().toISOString(), active_requests: 0, capabilities: {}, state: 'ready', phase: 'ready', message: '模型已就绪' }
    const loading = { ...ready, id: 'loading-instance', model_id: model.id, profile_id: profiles[0].id, alias: '正在加载实例', started_at: new Date(Date.now() - 5000).toISOString(), state: 'preparing', phase: 'preparing', message: '正在准备运行文件 2/8', error: null as string | null }
    let instances = [ready]
    await page.route('**/api/**', async route => {
      const request = route.request(), pathname = new URL(request.url()).pathname
      if (pathname === '/api/state') { stateReads++; return route.fulfill({ json: { ...baseline, models: [model], profiles, runtimes: [runtime], instances } }) }
      if (pathname === '/api/profiles/preflight') return route.fulfill({ json: { status: 'ready', candidates: [], errors: [], message: '' } })
      if (pathname === '/api/resources/estimate') return route.fulfill({ json: { resources: [], notes: [], source: 'fixture' } })
      if (pathname === '/api/profiles/load-profile-0/start') {
        await startGate
        instances = [ready, loading]
        return route.fulfill({ json: loading })
      }
      if (pathname.endsWith('/telemetry')) {
        const id = pathname.split('/')[3]
        if (id === loading.id) startingReads++
        return route.fulfill({ json: { instance_id: id, backend: 'llama_cpp', state: 'ready', updated_at: Date.now() / 1000, groups: [], unavailable: [], live_generation: { backend_running: 0, backend_queued: 0 } } })
      }
      return route.continue()
    })
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await page.getByRole('button', { name: /等待反馈模型/ }).click()
    const first = page.locator('.profile-card').filter({ hasText: '正在操作配置' })
    const second = page.locator('.profile-card').filter({ hasText: '另一套配置' })
    await first.getByRole('button', { name: '加载模型', exact: true }).click()
    await expect(first.getByRole('button', { name: '检查启动条件', exact: true })).toBeDisabled()
    await expect(first.locator('.model-loading-status')).toContainText('检查启动条件')
    await expect(second.getByRole('button', { name: '加载模型', exact: true })).toBeEnabled()
    await expect(second.locator('button .spin')).toHaveCount(0)
    await expect(page.getByRole('button', { name: '启动参数', exact: true }).locator('.spin')).toHaveCount(0)
    releaseStart!()
    await expect(first.locator('.model-loading-status')).toContainText('正在准备运行文件 2/8')
    await expect(first.locator('.model-loading-elapsed')).toContainText('已用')
    await expect(first.getByRole('button', { name: '取消准备', exact: true })).toBeEnabled()
    await expect(first.locator('.model-loading-status')).not.toContainText('%')
    await page.screenshot({ path: path.join(directory, 'model-preparing.png') })

    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '运行监控', exact: true }).click()
    const monitor = page.locator('.monitor-page')
    const select = monitor.getByRole('combobox', { name: '选择监控实例', exact: true })
    await expect(select).toHaveValue(ready.id)
    await select.selectOption(loading.id)
    const beforePolling = stateReads
    await expect.poll(() => stateReads).toBeGreaterThanOrEqual(beforePolling + 2)
    await expect(select).toHaveValue(loading.id)
    await expect(monitor.locator('.model-loading-status')).toContainText('2/8')
    loading.state = 'starting'; loading.phase = 'loading'; loading.message = 'load tensors 3/12'
    await monitor.getByRole('button', { name: '刷新监控', exact: true }).click()
    await expect(monitor.locator('.model-loading-status')).toContainText('load tensors 3/12')
    await expect(monitor.locator('.model-loading-heading')).toContainText('加载模型')
    await expect(select).toHaveValue(loading.id)
    expect(startingReads).toBe(0)
    await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(940, 820))
    await page.screenshot({ path: path.join(directory, 'monitor-loading-narrow.png') })

    loading.state = 'ready'; loading.phase = 'ready'; loading.message = '模型已就绪'
    await monitor.getByRole('button', { name: '刷新监控', exact: true }).click()
    await expect(monitor.locator('.model-loading-status')).toHaveCount(0)
    await expect(select).toHaveValue(loading.id)
    await expect.poll(() => startingReads).toBeGreaterThan(0)
    loading.state = 'stopping'; loading.phase = 'stopping'; loading.message = '正在停止所属后端进程'
    await monitor.getByRole('button', { name: '刷新监控', exact: true }).click()
    await expect(monitor.locator('.model-loading-status')).toContainText(loading.message)
    await expect(monitor.locator('.model-loading-heading')).toContainText('正在停止')
    await expect(monitor.locator('.model-loading-elapsed')).toHaveCount(0)
    loading.state = 'failed'; loading.phase = 'failed'; loading.error = 'fixture 模型加载失败'; loading.message = loading.error
    await monitor.getByRole('button', { name: '刷新监控', exact: true }).click()
    await expect(select).toHaveCount(0)
    await expect(monitor.locator('.model-instance-name')).toContainText(ready.alias)
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '模型库', exact: true }).click()
    await expect(first.getByRole('alert')).toContainText('加载失败')
    await expect(first.locator('.model-loading-status')).toHaveCount(0)
    await expect(first.getByRole('button', { name: '重新加载', exact: true })).toBeEnabled()
    expect(errors).toEqual([])
    console.log(JSON.stringify({ directory, stateReads, errors }))
  } finally {
    releaseStart?.()
    await page.unrouteAll({ behavior: 'wait' })
    await app.close()
  }
})
