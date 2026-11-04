import { test, expect, _electron as electron } from '@playwright/test'
import * as path from 'node:path'
import * as fs from 'node:fs'
import { pinNavigation } from './navigation'

for (const source of ['hf', 'modelscope']) {
  test(`模型源契约 ${source}：分页、文件选择与下载计划`, async () => {
    test.setTimeout(60_000)
    const root = path.resolve(__dirname, '..')
    const directory = fs.mkdtempSync(path.join(root, 'work', 'ui-discovery-contract-'))
    const application = await electron.launch({
      ...(process.env.MODLOADER_TEST_EXECUTABLE ? { executablePath: process.env.MODLOADER_TEST_EXECUTABLE, args: [] } : { args: [root] }),
      env: { ...process.env, MODLOADER_DATA_DIR: directory }, timeout: 60_000,
    })
    try {
      const page = await application.firstWindow()
      const errors: string[] = []
      const mutations: string[] = []
      let readmeAttempts = 0
      let missingReadme = false
      page.on('pageerror', error => errors.push(error.message))
      // Search results deliberately use the backend's id field, not repo_id.
      const items = ['First', 'Second', 'Third'].map(name => ({ id: `fixture/${name}`, name, description: 'text-generation', url: `https://example.com/${name}`, downloads: 1234, gated: name === 'Second' ? 'manual' : false }))
      const files = [
        { path: 'model-Q4-00001-of-00002.gguf', size: 1024, role: 'shard', group_id: 'q4' },
        { path: 'model-Q4-00002-of-00002.gguf', size: 2048, role: 'shard', group_id: 'q4' },
        { path: 'model-Q8.gguf', size: null, role: 'model', group_id: 'q8' },
        { path: 'mmproj-f16.gguf', size: 128, role: 'vision', group_id: 'vision-f16' },
        { path: 'mmproj-Q8.gguf', size: 64, role: 'vision', group_id: 'vision-q8' },
        { path: 'mtp-head.safetensors', size: 256, role: 'other' },
        { path: 'README.md', size: 64, role: 'other' },
        { path: 'images/preview.png', size: 32, role: 'other' },
      ]
      const groups = [
        { id: 'q4', name: 'model-Q4.gguf', format: 'gguf', kind: 'model', files: files.slice(0, 2).map(file => file.path), size: 3072, quantization: 'Q4' },
        { id: 'q8', name: 'model-Q8.gguf', format: 'gguf', kind: 'model', files: [files[2].path], size: null, quantization: 'Q8' },
        { id: 'vision-f16', name: 'mmproj-f16.gguf', format: 'gguf', kind: 'vision', files: [files[3].path], size: 128 },
        { id: 'vision-q8', name: 'mmproj-Q8.gguf', format: 'gguf', kind: 'vision', files: [files[4].path], size: 64 },
      ]
      const planRequests: Array<{ files: string[]; include_vision: boolean; revision: string }> = []
      const downloadRequests: Array<{ files: string[]; include_vision: boolean; revision: string; checked_files: unknown[] }> = []
      let lastPlanFiles: unknown[] = []
      await page.route('**/api/**', async route => {
        const request = route.request()
        const url = new URL(request.url())
        if (url.pathname === `/api/sources/${source}/search`) {
          expect(url.searchParams.get('q')).toBe('fixture')
          const more = url.searchParams.has('cursor')
          if (more) expect(url.searchParams.get('cursor')).toBe('next-page')
          await route.fulfill({ json: { items: more ? items.slice(1) : items.slice(0, 2), next_cursor: more ? null : 'next-page' } })
        } else if (url.pathname === '/api/sources/resolve') {
          expect(request.postDataJSON()).toEqual({ source, repo_id: 'fixture/Second', revision: 'main' })
          await route.fulfill({ json: { source, repo_id: 'fixture/Second', revision: 'resolved-revision', name: 'Second', description: '用于中文对话的轻量模型。', url: `https://${source === 'hf' ? 'huggingface.co' : 'modelscope.cn/models'}/fixture/Second`, metadata: {}, files, groups } })
        } else if (url.pathname === '/api/sources/readme') {
          expect(request.postDataJSON()).toEqual({ source, repo_id: 'fixture/Second', revision: 'resolved-revision' })
          if (++readmeAttempts === 1) return route.fulfill({ status: 503, json: { detail: '网络暂不可用' } })
          await route.fulfill({ json: { path: missingReadme ? null : 'README.md', revision: 'resolved-revision', url: '', content: missingReadme ? '' : '---\nlicense: apache-2.0\n---\n# 模型能力\n\n支持中文对话与代码理解。\n\n| 参数 | 说明 |\n| --- | --- |\n| 许可 | Apache 2.0 |\n\n[使用示例](examples/usage.md)\n\n[危险链接](javascript:alert(1))\n\n![架构](images/architecture.png)' } })
        } else if (url.pathname === '/api/downloads/plan') {
          const body = request.postDataJSON()
          expect(body.source).toBe(source)
          expect(body.repo_id).toBe('fixture/Second')
          expect(body.revision).toBe('resolved-revision')
          expect(typeof body.include_vision).toBe('boolean')
          planRequests.push(body)
          const selected = new Set<string>(body.files)
          const models = groups.filter(group => group.kind === 'model' && group.files.some(file => selected.has(file)))
          models.forEach(group => group.files.forEach(file => selected.add(file)))
          const vision = groups.filter(group => group.kind === 'vision' && group.files.some(file => selected.has(file)))
          const needsVision = body.include_vision && !vision.length
          const planned = files.filter(file => selected.has(file.path)).map(file => ({ ...file, reason: body.files.includes(file.path) ? '手动选择' : '同一模型的必要分片' }))
          lastPlanFiles = planned
          await route.fulfill({ json: { files: planned, automatic_files: planned.filter(file => !body.files.includes(file.path)), optional_files: files.filter(file => file.role === 'vision' && !selected.has(file.path)), model_groups: models, vision_groups: vision, total_bytes: planned.reduce((sum, file) => sum + (file.size || 0), 0), unknown_size_files: planned.filter(file => file.size === null).map(file => file.path), revision: 'resolved-revision', requires_vision_selection: needsVision, warnings: needsVision ? ['有多个视觉组件，请先选择与模型匹配的一项，再检查下载。'] : [] } })
        } else if (url.pathname === '/api/downloads' && request.method() === 'POST') {
          const body = request.postDataJSON()
          downloadRequests.push(body)
          expect(body.files).toEqual([files[0].path, files[2].path])
          expect(body.include_vision).toBe(false)
          expect(body.revision).toBe('resolved-revision')
          expect(body.checked_files).toEqual(lastPlanFiles)
          await route.fulfill({ json: { id: 'fixture-download-job' } })
        } else if (request.method() !== 'GET') {
          mutations.push(`${request.method()} ${url.pathname}`)
          await route.abort()
        } else await route.continue()
      })
      await pinNavigation(page)
      await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
      await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '发现模型', exact: true }).click()
      if (source === 'modelscope') await page.getByRole('button', { name: /ModelScope$/ }).click()
      await page.getByRole('textbox', { name: '搜索在线模型' }).fill('fixture')
      await page.getByRole('button', { name: '搜索', exact: true }).click()
      await expect(page.locator('.repository-card')).toHaveCount(2)
      await page.getByRole('button', { name: '加载更多', exact: true }).click()
      await expect(page.locator('.repository-card')).toHaveCount(3)
      const card = page.locator('.repository-card').filter({ has: page.getByRole('heading', { name: 'Second', exact: true }) })
      await expect(card).toContainText('fixture/Second')
      await expect(card).toContainText('需授权访问')
      await expect(card).toContainText('1,234')
      await card.getByRole('button', { name: '模型介绍', exact: true }).click()
      const intro = page.getByRole('dialog', { name: '模型介绍', exact: true })
      const document = intro.frameLocator('iframe.readme-document')
      await expect(intro).toContainText('用于中文对话的轻量模型。')
      await expect(intro).toContainText('网络暂不可用')
      await intro.getByRole('button', { name: /重试/ }).click()
      await expect(document.getByRole('heading', { name: '模型能力', exact: true })).toBeVisible()
      await expect(document.getByRole('table')).toContainText('Apache 2.0')
      await expect(document.locator('body')).not.toContainText('license: apache-2.0')
      await expect(document.getByRole('link', { name: '危险链接', exact: true })).toHaveCount(0)
      const exampleUrl = source === 'hf' ? 'https://huggingface.co/fixture/Second/blob/resolved-revision/examples/usage.md' : 'https://modelscope.cn/models/fixture/Second/files?revision=resolved-revision&fileName=examples%2Fusage.md'
      await expect(document.getByRole('link', { name: '使用示例', exact: true })).toHaveAttribute('href', exampleUrl)
      await page.screenshot({ path: path.join(directory, `${source}-introduction.png`) })
      await intro.getByRole('button', { name: '选择文件', exact: true }).click()
      const dialog = page.getByRole('dialog', { name: '选择下载文件', exact: true })
      await expect(dialog).toBeVisible()
      await expect(dialog.locator('.download-model-options .download-model-choice')).toHaveCount(2)
      await expect(dialog.locator('.download-all-files')).not.toHaveAttribute('open', '')
      await expect(dialog.getByRole('checkbox', { name: /同时下载视觉组件/ })).not.toBeChecked()
      await dialog.getByRole('checkbox', { name: 'model-Q4.gguf', exact: true }).check()
      await expect(dialog.locator('.download-model-choice').filter({ hasText: 'model-Q4.gguf' })).toContainText('2 个分片')
      await dialog.locator('.download-all-files > summary').click()
      await expect(dialog.getByRole('checkbox', { name: '文件 model-Q4-00002-of-00002.gguf', exact: true })).toBeChecked()
      await expect(dialog.getByRole('checkbox', { name: '文件 model-Q4-00002-of-00002.gguf', exact: true })).toBeDisabled()
      await expect(dialog.getByRole('checkbox', { name: '文件 mtp-head.safetensors', exact: true })).not.toBeChecked()
      await expect(dialog.getByRole('checkbox', { name: '文件 README.md', exact: true })).not.toBeChecked()
      await dialog.locator('.download-all-files > summary').click()
      missingReadme = true
      await dialog.getByRole('button', { name: '模型介绍', exact: true }).click()
      await expect(intro).toContainText('仓库未提供 README')
      await intro.getByRole('button', { name: '下载文件', exact: true }).click()
      await expect(dialog.getByRole('checkbox', { name: 'model-Q4.gguf', exact: true })).toBeChecked()
      await dialog.getByRole('button', { name: '检查下载', exact: true }).click()
      await expect(dialog.getByText('下载计划已生成', { exact: true })).toBeVisible()
      await expect(dialog.getByRole('button', { name: '开始下载', exact: true })).toBeEnabled()
      expect(planRequests.at(-1)).toEqual({ source, repo_id: 'fixture/Second', revision: 'resolved-revision', files: [files[0].path], include_vision: false })
      await dialog.getByRole('checkbox', { name: /同时下载视觉组件/ }).check()
      await expect(dialog.getByRole('button', { name: '检查下载', exact: true })).toBeVisible()
      await dialog.getByRole('button', { name: '检查下载', exact: true }).click()
      await expect(dialog.getByText('请选择视觉组件', { exact: true })).toBeVisible()
      await expect(dialog.getByRole('button', { name: '开始下载', exact: true })).toHaveCount(0)
      await dialog.getByRole('checkbox', { name: 'mmproj-Q8.gguf', exact: true }).check()
      await dialog.getByRole('button', { name: '检查下载', exact: true }).click()
      await expect(dialog.getByRole('button', { name: '开始下载', exact: true })).toBeEnabled()
      await dialog.getByRole('checkbox', { name: /同时下载视觉组件/ }).uncheck()
      await dialog.getByRole('checkbox', { name: 'model-Q8.gguf', exact: true }).check()
      await expect(dialog.locator('.download-selection-counts')).toContainText('2 个')
      await expect(dialog.locator('.footer-note')).toContainText('大小未知')
      await dialog.getByRole('button', { name: '检查下载', exact: true }).click()
      await expect(dialog.getByRole('button', { name: '开始下载', exact: true })).toBeEnabled()
      expect(planRequests.at(-1)?.include_vision).toBe(false)
      expect(planRequests.at(-1)?.files).toEqual([files[0].path, files[2].path])
      const version = dialog.getByRole('textbox', { name: '仓库版本', exact: true })
      await version.fill('another-revision')
      await expect(dialog.getByRole('button', { name: '检查下载', exact: true })).toBeVisible()
      await page.screenshot({ path: path.join(directory, `${source}-selection.png`) })
      await version.fill('resolved-revision')
      await dialog.getByRole('button', { name: '检查下载', exact: true }).click()
      await dialog.getByRole('button', { name: '开始下载', exact: true }).click()
      await expect(dialog).toHaveCount(0)
      expect(downloadRequests).toHaveLength(1)
      await page.getByRole('button', { name: '直接链接下载', exact: true }).click()
      const direct = page.getByRole('dialog', { name: '直接链接下载', exact: true })
      await expect(direct.getByRole('textbox', { name: '下载链接', exact: true })).toBeVisible()
      await expect(direct.locator('.selection-guide')).toHaveCount(0)
      await page.screenshot({ path: path.join(directory, `${source}-direct-download.png`), animations: 'disabled' })
      await direct.getByRole('button', { name: '取消', exact: true }).click()
      expect(errors).toEqual([])
      expect(mutations).toEqual([])
    } finally { await application.close() }
  })
}

test('HF 配置必需的视觉组件：默认关闭仍能选择，并保留已确认的必需项', async () => {
  test.setTimeout(60_000)
  const root = path.resolve(__dirname, '..')
  const directory = fs.mkdtempSync(path.join(root, 'work', 'ui-discovery-required-vision-'))
  const application = await electron.launch({
    ...(process.env.MODLOADER_TEST_EXECUTABLE ? { executablePath: process.env.MODLOADER_TEST_EXECUTABLE, args: [] } : { args: [root] }),
    env: { ...process.env, MODLOADER_DATA_DIR: directory }, timeout: 60_000,
  })
  try {
    const page = await application.firstWindow()
    const files = [
      { path: 'model.safetensors', size: 1024, role: 'model', group_id: 'main' },
      { path: 'config.json', size: 16, role: 'required', group_id: 'main' },
      { path: 'vision_tower/a.gguf', size: 128, role: 'vision', group_id: 'tower-a' },
      { path: 'vision_tower/b.gguf', size: 256, role: 'vision', group_id: 'tower-b' },
      { path: 'chat_template.json', size: 32, role: 'other', group_id: undefined },
      { path: 'params.json', size: 32, role: 'other', group_id: undefined },
    ]
    const groups = [
      { id: 'main', name: 'HF model', format: 'hf', kind: 'model', files: files.slice(0, 2).map(file => file.path), size: 1040 },
      { id: 'tower-a', name: 'Vision A', format: 'gguf', kind: 'vision', files: [files[2].path], size: 128 },
      { id: 'tower-b', name: 'Vision B', format: 'gguf', kind: 'vision', files: [files[3].path], size: 256 },
    ]
    const planRequests: Array<{ files: string[]; include_vision: boolean }> = []
    await page.route('**/api/**', async route => {
      const url = new URL(route.request().url())
      if (url.pathname === '/api/sources/resolve') {
        await route.fulfill({ json: { source: 'hf', repo_id: 'fixture/Required', revision: 'fixed', name: 'Required', files, groups, metadata: {} } })
      } else if (url.pathname === '/api/downloads/plan') {
        const body = route.request().postDataJSON()
        planRequests.push(body)
        const required = groups.slice(1).filter(group => group.files.some(file => body.files.includes(file)))
        const paths = new Set([...body.files, ...groups[0].files, ...required.flatMap(group => group.files)])
        const planned = files.filter(file => paths.has(file.path)).map(file => ({ ...file, reason: file.role === 'vision' ? 'HF配置要求的视觉组件与配置' : '模型配置与 tokenizer 等必要文件' }))
        await route.fulfill({ json: { revision: 'fixed', files: planned, groups, model_groups: [groups[0]], vision_groups: required, required_vision_groups: required, automatic_files: planned.filter(file => !body.files.includes(file.path)), optional_files: files.filter(file => file.role === 'vision' && !paths.has(file.path)), requires_vision_selection: required.length !== 1, warnings: required.length !== 1 ? ['HF配置要求视觉组件，请明确选择匹配的一项。'] : [] } })
      } else if (route.request().method() !== 'GET') {
        throw new Error(`Unexpected mutation ${url.pathname}`)
      } else await route.continue()
    })
    await pinNavigation(page)
    await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '发现模型', exact: true }).click()
    await page.getByRole('button', { name: '打开模型仓库', exact: true }).first().click()
    const open = page.getByRole('dialog', { name: '打开 Hugging Face 仓库', exact: true })
    await open.getByRole('textbox', { name: '仓库 ID', exact: true }).fill('fixture/Required')
    await open.getByRole('button', { name: '读取文件清单', exact: true }).click()
    const dialog = page.getByRole('dialog', { name: '选择下载文件', exact: true })
    await dialog.getByRole('checkbox', { name: 'HF model', exact: true }).check()
    await dialog.locator('.download-all-files > summary').click()
    await expect(dialog.getByRole('checkbox', { name: '文件 config.json', exact: true })).toBeChecked()
    await expect(dialog.getByRole('checkbox', { name: '文件 config.json', exact: true })).toBeDisabled()
    const sidecars = ['chat_template.json', 'params.json']
    for (const name of sidecars) {
      const checkbox = dialog.getByRole('checkbox', { name: `文件 ${name}`, exact: true })
      await expect(checkbox).not.toBeChecked()
      await expect(checkbox).toBeEnabled()
      await checkbox.check()
    }
    const optional = dialog.getByRole('checkbox', { name: /同时下载视觉组件/ })
    await expect(optional).not.toBeChecked()
    await dialog.getByRole('button', { name: '检查下载', exact: true }).click()
    await expect(dialog.getByRole('checkbox', { name: 'Vision A', exact: true })).toBeVisible()
    expect(planRequests.at(-1)?.files).toEqual(['model.safetensors', ...sidecars])
    for (const name of sidecars) await dialog.getByRole('checkbox', { name: `文件 ${name}`, exact: true }).uncheck()
    await dialog.locator('.download-all-files > summary').click()
    await expect(dialog.getByRole('button', { name: '开始下载', exact: true })).toHaveCount(0)
    await dialog.getByRole('checkbox', { name: 'Vision A', exact: true }).check()
    await expect(dialog.getByRole('checkbox', { name: 'Vision A', exact: true })).toBeChecked()
    await expect(dialog.getByRole('checkbox', { name: 'Vision B', exact: true })).toBeVisible()
    await expect(optional).not.toBeChecked()
    await dialog.getByRole('button', { name: '检查下载', exact: true }).click()
    await expect(dialog.getByRole('button', { name: '开始下载', exact: true })).toBeEnabled()
    await optional.check()
    await optional.uncheck()
    await dialog.getByRole('button', { name: '检查下载', exact: true }).click()
    expect(planRequests.at(-1)?.files).toEqual(['model.safetensors', 'vision_tower/a.gguf'])
    expect(planRequests.at(-1)?.include_vision).toBe(false)
    await expect(dialog.getByRole('button', { name: '开始下载', exact: true })).toBeEnabled()
    await dialog.getByRole('checkbox', { name: 'HF model', exact: true }).uncheck()
    await expect(dialog.getByRole('checkbox', { name: 'Vision A', exact: true })).toHaveCount(0)
    await dialog.getByRole('button', { name: '取消', exact: true }).click()
  } finally { await application.close() }
})

for (const source of ['hf', 'modelscope']) {
  test(`真实模型源 ${source}：搜索后选择文件`, async () => {
    test.skip(!process.env.MODLOADER_TEST_LIVE_SOURCES, '仅在需要真实平台验证时运行')
    test.setTimeout(150_000)
    const root = path.resolve(__dirname, '..')
    const directory = fs.mkdtempSync(path.join(root, 'work', 'ui-discovery-live-'))
    const application = await electron.launch({
      ...(process.env.MODLOADER_TEST_EXECUTABLE ? { executablePath: process.env.MODLOADER_TEST_EXECUTABLE, args: [] } : { args: [root] }),
      env: { ...process.env, MODLOADER_DATA_DIR: directory }, timeout: 60_000,
    })
    try {
      const page = await application.firstWindow()
      const errors: string[] = []
      page.on('pageerror', error => errors.push(error.message))
      // Browsing and planning are allowed; this check never starts a model download.
      await page.route('**/api/downloads', route => route.abort())
      await pinNavigation(page)
      await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
      await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '发现模型', exact: true }).click()
      if (source === 'modelscope') await page.getByRole('button', { name: /ModelScope$/ }).click()
      await page.getByRole('textbox', { name: '搜索在线模型' }).fill('Qwen2.5-0.5B-Instruct')
      const searchResponse = page.waitForResponse(response => new URL(response.url()).pathname === `/api/sources/${source}/search`, { timeout: 90_000 })
      await page.getByRole('button', { name: '搜索', exact: true }).click()
      const response = await searchResponse
      const payload = await response.json()
      expect(response.ok(), JSON.stringify({ status: response.status(), payload })).toBe(true)
      expect(payload.items.length).toBeGreaterThan(0)
      console.log(JSON.stringify({ source, firstResult: payload.items[0] }))
      await page.locator('.repository-card').first().getByRole('button', { name: '模型介绍', exact: true }).click()
      await expect(page.locator('.toast.error')).toHaveCount(0)
      const intro = page.getByRole('dialog', { name: '模型介绍', exact: true })
      await expect(intro.locator('.readme-heading')).toContainText('README', { timeout: 90_000 })
      await expect(intro.frameLocator('iframe.readme-document').locator('body')).toContainText('Qwen')
      await page.screenshot({ path: path.join(directory, `${source}-introduction.png`) })
      await intro.getByRole('button', { name: '选择文件', exact: true }).click()
      const dialog = page.getByRole('dialog', { name: '选择下载文件', exact: true })
      await expect(dialog).toBeVisible({ timeout: 90_000 })
      await expect(dialog).toContainText(payload.items[0].id)
      expect(await dialog.locator('.download-model-choice').count()).toBeGreaterThan(0)
      await page.screenshot({ path: path.join(directory, `${source}-files.png`) })
      console.log(JSON.stringify({ source, groupCount: await dialog.locator('.download-model-choice').count(), screenshot: path.join(directory, `${source}-files.png`) }))
      await dialog.getByRole('button', { name: '取消', exact: true }).click()
      expect(errors).toEqual([])
    } finally { await application.close() }
  })
}
