import { test, expect, _electron as electron, type Locator } from '@playwright/test'
import * as path from 'node:path'
import * as fs from 'node:fs'
import { pinNavigation } from './navigation'

async function samplePreparationToggle(paths: Locator, toggles = [0, 55, 110, 165]) {
  return paths.evaluate(async (element, toggles) => {
    const details = element as HTMLDetailsElement
    await new Promise<void>((resolve, reject) => {
      const deadline = performance.now() + 1000
      const settled = () => {
        const style = getComputedStyle(details, '::details-content')
        if (Number(style.opacity) === (details.open ? 1 : 0) && style.contentVisibility === (details.open ? 'visible' : 'hidden')) resolve()
        else if (performance.now() >= deadline) reject(new Error('Preparation disclosure did not settle'))
        else requestAnimationFrame(settled)
      }
      settled()
    })
    const heading = details.querySelector('summary')!
    const title = heading.querySelector('strong')!, status = heading.querySelector('span')!
    const section = details.closest('.profile-preparation')!, body = details.closest('.modal-body')!
    const nextField = section.nextElementSibling!.querySelector('.field')!
    const geometry = () => ({ height: details.getBoundingClientRect().height, sectionHeight: section.getBoundingClientRect().height,
      nextFieldTop: nextField.getBoundingClientRect().top, nextFieldContentTop: nextField.getBoundingClientRect().top - body.getBoundingClientRect().top + body.scrollTop,
      scrollTop: body.scrollTop })
    const baseline = { title: title.getBoundingClientRect().height, status: status.getBoundingClientRect().height,
      headingHeight: heading.getBoundingClientRect().height, ...geometry() }
    const frames: Array<ReturnType<typeof geometry> & { time: number; segment: number; open: boolean; title: number; status: number; headingWidth: number; opacity: string; visibility: string }> = []
    const switches: Array<{ time: number; open: boolean; beforeHeight: number; afterHeight: number }> = []
    // Sample the native toggle every frame, including reversals before its fade completes.
    let next = 0
    return new Promise<{ baseline: typeof baseline; frames: typeof frames; switches: typeof switches }>(resolve => {
      const start = performance.now()
      const sample = () => {
        const time = performance.now() - start
        if (next < toggles.length && time >= toggles[next]) {
          const beforeHeight = details.getBoundingClientRect().height
          details.querySelector('summary')!.click()
          switches.push({ time, open: details.open, beforeHeight, afterHeight: details.getBoundingClientRect().height })
          next++
        }
        const content = getComputedStyle(details, '::details-content')
        const frame = { time, segment: next - 1, open: details.open, title: title.getBoundingClientRect().height,
          status: status.getBoundingClientRect().height, headingWidth: heading.getBoundingClientRect().width,
          opacity: content.opacity, visibility: content.contentVisibility, ...geometry() }
        frames.push(frame)
        if (frame.title > baseline.title + 1 || frame.status > baseline.status + 1) {
          resolve({ baseline, frames, switches })
        } else if (next === toggles.length && time >= toggles.at(-1)! + 200) resolve({ baseline, frames, switches })
        else requestAnimationFrame(sample)
      }
      requestAnimationFrame(sample)
    })
  }, toggles)
}

function expectSmoothPreparationMotion(sample: Awaited<ReturnType<typeof samplePreparationToggle>>) {
  const collapsed = sample.baseline.headingHeight
  const expanded = Math.max(sample.baseline.height, ...sample.frames.map(frame => frame.height))
  expect(expanded).toBeGreaterThan(collapsed + 1)
  expect(sample.frames.filter(frame => frame.height > collapsed + 1 && frame.height < expanded - 1).length).toBeGreaterThan(2)
  for (const frame of sample.frames) {
    expect(Math.abs((frame.height - collapsed) / (expanded - collapsed) - Number(frame.opacity))).toBeLessThan(.03)
    // All list margins must move inside the animated box, not jump at its end.
    expect(Math.abs(frame.sectionHeight - frame.height - (sample.baseline.sectionHeight - sample.baseline.height))).toBeLessThan(1)
    expect(Math.abs(frame.nextFieldContentTop - sample.baseline.nextFieldContentTop - (frame.height - sample.baseline.height))).toBeLessThan(1)
  }
  for (const [segment, change] of sample.switches.entries()) {
    expect(Math.abs(change.afterHeight - change.beforeHeight)).toBeLessThan(1)
    const frames = sample.frames.filter(frame => frame.segment === segment)
    for (let index = 1; index < frames.length; index++) {
      const movement = frames[index].height - frames[index - 1].height
      expect(change.open ? movement >= -1 : movement <= 1).toBe(true)
    }
    expect(change.open ? frames.at(-1)!.height > frames[0].height : frames.at(-1)!.height < frames[0].height).toBe(true)
  }
}

test('原生配置界面：资源来源、图像输入、首次准备与取消', async () => {
  test.setTimeout(90_000)
  const root = path.resolve(__dirname, '..')
  const directory = fs.mkdtempSync(path.join(root, 'work', 'ui-preparation-'))
  const application = await electron.launch({
    ...(process.env.MODLOADER_TEST_EXECUTABLE ? { executablePath: process.env.MODLOADER_TEST_EXECUTABLE, args: [] } : { args: [root] }),
    env: { ...process.env, MODLOADER_DATA_DIR: directory }, timeout: 60_000,
  })
  try {
    const page = await application.firstWindow()
    await page.emulateMedia({ reducedMotion: 'no-preference' })
    const errors: string[] = []
    page.on('pageerror', error => errors.push(error.message))
    await pinNavigation(page)
    await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
    const baseline = await page.evaluate(async () => {
      const connection = await window.modloader!.connection()
      return (await fetch(connection.baseUrl + '/api/state', { headers: { Authorization: 'Bearer ' + connection.token } })).json()
    })
    const runtime = { id: 'fixture-runtime', backend: 'strata', name: 'Strata 界面样例', path: 'D:\\Fixture\\Strata', version: 'fixture', managed: false, metadata: {} }
    const models = ['已有资源模型', '待准备模型'].map((name, index) => ({ id: 'fixture-' + index, name, path: `D:\\Fixture\\${index}\\IQ3_XXS\\model.gguf`, format: 'gguf', architecture: 'qwen4exp', quantization: 'IQ3_XXS', files: [], size_bytes: 76_000_000_000, modalities: ['text'], metadata: {}, source: {}, managed: false, warnings: [] }))
    const profiles = models.map((model, index) => ({ id: 'profile-' + index, model_id: model.id, name: '独立加载配置', backend: 'strata', runtime_id: runtime.id, alias: 'ui-fixture-' + index, parameters: { kv_cache: 'q4_0' } as Record<string, unknown>, generation: {}, extra_args: [], revision: 1 }))
    const pack = 'D:\\Fixture\\Models\\Qwen3.8-235B-A22B-Instruct-2507-IQ3_XXS\\Strata\\prepared-resources\\original-imported-configuration'
    let preflightMode: 'ready' | 'choice' | 'error' | 'request_error' | 'empty' = 'ready'
    let instances: Array<Record<string, unknown>> = []
    let saved: Record<string, unknown> | null = null
    let starts = 0, stops = 0
    const state = () => ({ ...baseline, models, profiles, runtimes: [runtime], instances })
    await page.route('**/api/**', async route => {
      const request = route.request(), url = new URL(request.url())
      if (url.pathname === '/api/state') return route.fulfill({ json: state() })
      if (url.pathname === '/api/resources/estimate') return route.fulfill({ json: { resources: [], notes: [], source: 'UI fixture' } })
      if (url.pathname === '/api/profiles/preflight') {
        const body = request.postDataJSON(), existing = body.model_id === 'fixture-0'
        if (existing && preflightMode === 'request_error') return route.fulfill({ status: 500, json: { detail: '准备文件检查暂时失败' } })
        const status = !existing ? 'needs_preparation' : preflightMode === 'choice' && !body.parameters.strata_resource_id ? 'choice_required' : preflightMode === 'error' ? 'error' : 'ready'
        const ready = status === 'ready', hasPaths = ready && preflightMode !== 'empty'
        return route.fulfill({ json: { status, source: ready ? { kind: 'profile', label: '原有导入配置', profile_id: 'original' } : null, pack: hasPaths ? pack : null, tokenizer: hasPaths ? pack + '\\tokenizer' : null, message: status === 'error' ? '指定的准备文件目录不可用' : ready ? '复用本模型已有的 Strata 资源' : '首次加载需要在本地准备 Strata 资源', errors: status === 'error' ? ['缺少 tokenizer.json'] : [], candidates: existing ? [{ id: 'original', label: '原有导入配置', pack, tokenizer: pack + '\\tokenizer' }] : [], mmproj_candidates: ['D:\\Fixture\\mmproj-Qwen-BF16.gguf'], vision_enabled: Boolean(body.parameters.vision_enabled), mmproj: body.parameters.mmproj || null } })
      }
      if (url.pathname === '/api/profiles/profile-0' && request.method() === 'PATCH') {
        saved = request.postDataJSON()
        Object.assign(profiles[0], saved, { revision: 2 })
        return route.fulfill({ json: profiles[0] })
      }
      if (url.pathname === '/api/profiles/profile-1/start') {
        starts++
        instances = [{ id: 'preparing-fixture', model_id: 'fixture-1', profile_id: 'profile-1', profile_revision: 1, alias: 'ui-fixture-1', backend: 'strata', runtime_id: runtime.id, state: 'preparing', phase: 'preparing', message: '正在导出分词器，完成后继续加载', port: 0, pid: null, started_at: '', active_requests: 0, capabilities: {} }]
        return route.fulfill({ json: instances[0] })
      }
      if (url.pathname === '/api/instances/preparing-fixture/stop') {
        stops++; instances[0].state = 'stopped'
        return route.fulfill({ json: instances[0] })
      }
      if (request.method() !== 'GET') throw new Error('Unexpected mutation ' + request.method() + ' ' + url.pathname)
      return route.continue()
    })
    await pinNavigation(page)
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await page.getByRole('button', { name: /已有资源模型/ }).click()
    await expect(page.locator('.profile-card')).toContainText('ui-fixture-0')
    await page.getByRole('button', { name: '编辑配置', exact: true }).click()
    const dialog = page.getByRole('dialog', { name: '编辑加载配置', exact: true })
    await expect(dialog.getByText('复用已有资源', { exact: true })).toBeVisible()
    const paths = dialog.locator('.preparation-paths')
    const title = paths.locator('summary').getByText('模型准备文件', { exact: true })
    await expect(dialog.getByText('查看路径与来源', { exact: true })).toHaveCount(0)
    await expect(paths.locator('summary')).toHaveText('模型准备文件复用已有资源')
    await expect(paths.locator('dl')).toBeHidden()
    await page.screenshot({ path: path.join(directory, 'preparation-collapsed-1440.png') })
    await paths.locator('summary').focus()
    await paths.locator('summary').press('Enter')
    await expect(paths).toHaveAttribute('open')
    await paths.locator('summary').press('Space')
    await expect(paths).not.toHaveAttribute('open')
    await title.click()
    await expect(paths).toContainText(pack)
    await expect(paths).toContainText('原有导入配置')
    for (const [width, height] of [[1440, 960], [940, 640]]) {
      await application.evaluate(({ BrowserWindow }, size) => BrowserWindow.getAllWindows()[0].setSize(size[0], size[1]), [width, height])
      const samples: Awaited<ReturnType<typeof samplePreparationToggle>>[] = []
      for (const toggles of [[0], [0], [0, 55, 110, 165]]) {
        const sample = await samplePreparationToggle(paths, toggles)
        samples.push(sample)
        if (!sample.frames.at(-1)!.open) {
          await dialog.locator('.profile-preparation').scrollIntoViewIfNeeded()
          await page.screenshot({ path: path.join(directory, `preparation-collapsed-${width}.png`) })
        }
        if (sample.frames.some(frame => frame.title > sample.baseline.title + 1 || frame.status > sample.baseline.status + 1)) break
      }
      fs.writeFileSync(path.join(directory, `preparation-toggle-${width}.json`), JSON.stringify(samples, null, 2))
      await dialog.locator('.profile-preparation').scrollIntoViewIfNeeded()
      await page.screenshot({ path: path.join(directory, `preparation-expanded-${width}.png`) })
      for (const sample of samples) {
        expect(sample.frames.filter(frame => frame.title > sample.baseline.title + 1 || frame.status > sample.baseline.status + 1)).toEqual([])
        expect(sample.frames.length).toBeGreaterThan(4)
        expectSmoothPreparationMotion(sample)
      }
      expect(samples[0].frames.at(-1)).toMatchObject({ open: false, visibility: 'hidden' })
      expect(samples[0].frames.some(frame => Number(frame.opacity) > 0 && Number(frame.opacity) < 1)).toBe(true)
      await expect(paths).toHaveAttribute('open')
    }
    await page.emulateMedia({ reducedMotion: 'reduce' })
    const reduced = await samplePreparationToggle(paths, [0])
    expect(reduced.frames.filter(frame => frame.title > reduced.baseline.title + 1 || frame.status > reduced.baseline.status + 1)).toEqual([])
    expect(reduced.frames.every(frame => frame.visibility === 'hidden')).toBe(true)
    expect(reduced.frames.every(frame => Math.abs(frame.height - reduced.baseline.headingHeight) < 1)).toBe(true)
    await paths.locator('summary').click()
    await page.emulateMedia({ reducedMotion: 'no-preference' })
    await dialog.getByRole('checkbox', { name: '图像输入', exact: true }).check()
    await expect(dialog.getByRole('combobox', { name: '视觉投影器', exact: true })).toHaveValue('D:\\Fixture\\mmproj-Qwen-BF16.gguf')
    await dialog.getByRole('button', { name: '文件与启动选项', exact: true }).click()
    await expect(dialog.getByLabel('本地 Strata tokenizer 目录', { exact: true })).toHaveValue('')
    await expect(dialog.getByLabel('已准备的 Strata pack 目录', { exact: true })).toHaveValue('')
    await expect(dialog.getByText('复用已有资源', { exact: true })).toBeVisible()
    for (const [width, height] of [[1440, 960], [940, 640]]) {
      await application.evaluate(({ BrowserWindow }, size) => BrowserWindow.getAllWindows()[0].setSize(size[0], size[1]), [width, height])
      await expect(dialog.getByRole('button', { name: '保存配置', exact: true })).toBeVisible()
      expect(await dialog.evaluate(element => element.scrollWidth <= element.clientWidth + 2)).toBe(true)
      await page.screenshot({ path: path.join(directory, `configuration-${width}.png`) })
    }
    await dialog.getByRole('button', { name: '保存配置', exact: true }).click()
    await expect(dialog).toBeHidden()
    expect(saved).not.toBeNull()
    expect((saved as unknown as {parameters:Record<string, unknown>}).parameters).toMatchObject({ kv_cache: 'q4_0', vision_enabled: true, mmproj: 'D:\\Fixture\\mmproj-Qwen-BF16.gguf' })
    expect((saved as unknown as {parameters:Record<string, unknown>}).parameters).not.toHaveProperty('pack')
    const preparation = dialog.locator('.profile-preparation')
    const selection = preparation.getByRole('combobox', { name: '使用已有准备文件', exact: true })
    preflightMode = 'choice'
    await page.getByRole('button', { name: '编辑配置', exact: true }).click()
    await expect(preparation.locator('.preparation-heading')).toContainText('选择已有资源')
    await expect(paths).toHaveCount(0)
    await expect(selection).toBeVisible()
    await selection.selectOption('original')
    await expect(title).toBeVisible()
    await expect(paths).not.toHaveAttribute('open')
    await expect(selection).toHaveValue('original')
    await expect(selection).toBeVisible()
    await preparation.scrollIntoViewIfNeeded()
    await page.screenshot({ path: path.join(directory, 'preparation-selected-collapsed.png') })
    await title.click()
    await expect(paths.locator('dl')).toBeVisible()
    await expect(selection).toBeVisible()
    await preparation.scrollIntoViewIfNeeded()
    await page.screenshot({ path: path.join(directory, 'preparation-selected-expanded.png') })
    await title.click()
    await expect(selection).toBeVisible()
    await expect(selection).toHaveValue('original')
    await dialog.getByRole('button', { name: '取消', exact: true }).click()

    preflightMode = 'error'
    await page.getByRole('button', { name: '编辑配置', exact: true }).click()
    await expect(preparation.locator('.preparation-heading')).toContainText('需要处理')
    await expect(paths).toHaveCount(0)
    await expect(preparation.getByRole('alert')).toHaveText('准备文件不可用')
    await preparation.getByText('诊断详情', { exact: true }).click()
    await expect(preparation.locator('.diagnostic-notice pre')).toContainText('缺少 tokenizer.json')
    await preparation.scrollIntoViewIfNeeded()
    await page.screenshot({ path: path.join(directory, 'preparation-error.png') })
    await dialog.getByRole('button', { name: '取消', exact: true }).click()

    preflightMode = 'request_error'
    await page.getByRole('button', { name: '编辑配置', exact: true }).click()
    await expect(preparation.locator('.preparation-heading')).toContainText('检查失败')
    await expect(paths).toHaveCount(0)
    await expect(preparation.getByRole('button', { name: '重试', exact: true })).toBeVisible()
    preflightMode = 'ready'
    await preparation.getByRole('button', { name: '重试', exact: true }).click()
    await expect(title).toBeVisible()
    await expect(paths).not.toHaveAttribute('open')
    await dialog.getByRole('button', { name: '取消', exact: true }).click()

    preflightMode = 'empty'
    await page.getByRole('button', { name: '编辑配置', exact: true }).click()
    await expect(preparation.locator('.preparation-heading')).toContainText('复用已有资源')
    await expect(title).toBeVisible()
    await title.click()
    await expect(paths.locator('dl')).toBeVisible()
    await expect(paths.locator('dl')).toContainText('原有导入配置')
    await dialog.getByRole('button', { name: '取消', exact: true }).click()
    preflightMode = 'ready'
    await page.getByRole('button', { name: /待准备模型/ }).click()
    await expect(page.locator('.preparation-summary')).toContainText('首次加载将生成本地运行文件')
    await page.getByRole('button', { name: '编辑配置', exact: true }).click()
    await expect(preparation.locator('.preparation-heading')).toContainText('首次加载时生成')
    await expect(paths).toHaveCount(0)
    await dialog.getByRole('button', { name: '取消', exact: true }).click()
    await page.getByRole('button', { name: '加载模型', exact: true }).click()
    await expect(page.locator('.profile-card').getByText('准备中', { exact: true })).toBeVisible()
    await expect(page.locator('.profile-card .profile-preparing')).toContainText('正在导出分词器')
    await page.screenshot({ path: path.join(directory, 'preparing-940.png') })
    await page.getByRole('button', { name: '取消准备', exact: true }).click()
    await expect(page.getByRole('button', { name: '加载模型', exact: true })).toBeVisible()
    expect(starts).toBe(1); expect(stops).toBe(1); expect(errors).toEqual([])
    console.log(JSON.stringify({ screenshots: directory, native: true, backend: 'fixture' }))
  } finally { await application.close() }
})
