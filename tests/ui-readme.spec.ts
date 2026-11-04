import { test, expect, _electron as electron } from '@playwright/test'
import * as path from 'node:path'
import * as fs from 'node:fs'
import { pinNavigation } from './navigation'

const root = path.resolve(__dirname, '..')
const fixture = path.join(__dirname, 'fixtures', 'model-readme')
const { repo_id, revision } = JSON.parse(fs.readFileSync(path.join(fixture, 'metadata.json'), 'utf8'))

for (const live of [false, true]) {
  test(`${live ? '真实仓库' : '原始失败样本'}：SC117 连续 HTML 文档与中英文切换`, async () => {
    test.skip(live && !process.env.MODLOADER_TEST_LIVE_SOURCES, '真实平台检查按需运行')
    test.setTimeout(150_000)
    const directory = fs.mkdtempSync(path.join(root, 'work', 'ui-readme-'))
    const application = await electron.launch({
      ...(process.env.MODLOADER_TEST_EXECUTABLE ? { executablePath: process.env.MODLOADER_TEST_EXECUTABLE, args: [] } : { args: [root] }),
      env: { ...process.env, MODLOADER_DATA_DIR: directory }, timeout: 60_000,
    })
    let releaseImages: (() => void) | undefined
    try {
      const page = await application.firstWindow()
      const capture = async (name: string) => {
        await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))))
        const png = await application.evaluate(async ({ BrowserWindow }) => (await BrowserWindow.getAllWindows()[0].webContents.capturePage()).toPNG().toString('base64'))
        fs.writeFileSync(path.join(directory, name), Buffer.from(png, 'base64'))
      }
      const errors: string[] = [], requests: Array<{ file_path?: string; revision: string }> = []
      page.on('pageerror', error => errors.push(error.message))
      await application.evaluate(({ ipcMain }) => {
        ;(globalThis as any).readmeExternalLinks = []
        ipcMain.removeHandler('open-external')
        ipcMain.handle('open-external', (_, url) => { (globalThis as any).readmeExternalLinks.push(url) })
      })
      await page.route('**/api/downloads', route => route.abort())
      // Reproduce the real slow-image failure without making text/layout wait
      // for the banner. Release these requests during cleanup.
      const slowImages = new Promise<void>(resolve => { releaseImages = resolve })
      if (!live) await page.route('**/resolve/**', async route => {
        if (route.request().resourceType() !== 'image') return route.continue()
        await slowImages
        await route.abort().catch(() => undefined)
      })
      if (!live) await page.route('**/api/sources/resolve', route => route.fulfill({ json: {
        source: 'hf', repo_id, revision, name: repo_id.split('/').at(-1), description: '', url: `https://huggingface.co/${repo_id}`,
        files: [{ path: 'README.md', size: 49825 }, { path: 'README.zh-CN.md', size: 47601 }], metadata: { requested_revision: 'main' },
      } }))
      await page.route('**/api/sources/readme', async route => {
        const body = route.request().postDataJSON()
        requests.push(body)
        if (live) return route.continue()
        const file = body.file_path || 'README.md'
        expect(['README.md', 'README.zh-CN.md']).toContain(file)
        return route.fulfill({ json: { content: fs.readFileSync(path.join(fixture, file), 'utf8'), path: file, revision, url: `https://huggingface.co/${repo_id}` } })
      })
      await pinNavigation(page)
      await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
      await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '发现模型', exact: true }).click()
      await page.locator('.page-actions').getByRole('button', { name: '打开模型仓库', exact: true }).click()
      const open = page.getByRole('dialog', { name: '打开 Hugging Face 仓库', exact: true })
      await open.getByRole('textbox', { name: '仓库 ID', exact: true }).fill(repo_id)
      await open.getByRole('button', { name: '读取文件清单', exact: true }).click()
      await page.getByRole('dialog', { name: '选择下载文件', exact: true }).getByRole('button', { name: '模型介绍', exact: true }).click({ timeout: 90_000 })
      const intro = page.getByRole('dialog', { name: '模型介绍', exact: true })
      const document = intro.frameLocator('.readme-document')
      await expect(document.locator('body')).toContainText('The refusal direction removed', { timeout: 90_000 })
      await expect(intro.locator('iframe')).toHaveCount(1)
      await expect(intro.locator('.message-code-block')).toHaveCount(0)
      await expect(document.locator('iframe')).toHaveCount(0)
      await expect.poll(() => document.locator('body').evaluate(body => Math.abs(body.scrollHeight - body.ownerDocument.documentElement.clientHeight))).toBeLessThanOrEqual(2)
      expect(await intro.locator('.modal-body').evaluate(element => element.scrollHeight > element.clientHeight * 2)).toBe(true)
      await capture('english.png')
      await document.getByRole('link', { name: /简体中文/ }).first().click()
      await expect(intro.locator('.readme-heading')).toHaveText('README.zh-CN.md', { timeout: 90_000 })
      await expect(document.locator('body')).toContainText('重建后四档都只比官方多')
      await capture('chinese.png')
      await application.evaluate(({ BrowserWindow }) => { const window = BrowserWindow.getAllWindows()[0]; window.setSize(1050, 780); window.webContents.setZoomFactor(1.2) })
      await expect.poll(() => page.evaluate(() => innerWidth)).toBeLessThan(900)
      await expect.poll(() => intro.evaluate(element => element.getBoundingClientRect().right - innerWidth)).toBeLessThanOrEqual(1)
      await expect.poll(() => document.locator('body').evaluate(body => Math.abs(body.scrollHeight - body.ownerDocument.documentElement.clientHeight))).toBeLessThanOrEqual(2)
      await capture('chinese-narrow.png')
      await document.getByRole('link', { name: /English/ }).first().click()
      await expect(intro.locator('.readme-heading')).toHaveText('README.md', { timeout: 90_000 })
      await expect(document.locator('body')).toContainText('The refusal direction removed')
      expect(await application.evaluate(() => (globalThis as any).readmeExternalLinks)).toEqual([])
      expect(requests.map(request => request.file_path)).toEqual([undefined, 'README.zh-CN.md', 'README.md'])
      expect(new Set(requests.map(request => request.revision)).size).toBe(1)
      await document.getByRole('link', { name: /ISTA-DASLab GSQ-RCO quantizations/ }).first().click()
      expect(await application.evaluate(() => (globalThis as any).readmeExternalLinks)).toEqual(['https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF'])
      await intro.getByRole('button', { name: '选择文件', exact: true }).click()
      await expect(page.getByRole('dialog', { name: '选择下载文件', exact: true }).locator('.download-file')).not.toHaveCount(0)
      expect(errors).toEqual([])
      console.log(JSON.stringify({ live, screenshots: directory, readmeRequests: requests.map(({ file_path, revision }) => ({ file_path, revision })) }))
    } finally { releaseImages?.(); await application.close() }
  })
}

test('README 文内锚点、折叠内容和脚本隔离', async () => {
  const directory = fs.mkdtempSync(path.join(root, 'work', 'ui-readme-safety-'))
  const application = await electron.launch({ args: [root], env: { ...process.env, MODLOADER_DATA_DIR: directory } })
  try {
    const page = await application.firstWindow()
    const alerts: string[] = []
    page.on('dialog', async dialog => { alerts.push(dialog.message()); await dialog.dismiss() })
    await page.route('**/api/sources/resolve', route => route.fulfill({ json: { source: 'hf', repo_id: 'fixture/README', revision: 'main', name: 'README', url: '', description: '', metadata: {}, files: [] } }))
    const content = '# 文档标题\n\n[跳到中文](#中文说明)\n\n<div style="min-height:900px;color:rgb(12,34,56)">连续 HTML 内容</div>\n\n## 中文说明\n\n<details><summary>展开用法</summary><div style="height:300px">这里是使用方式</div></details>\n\n<script>parent.alert("source script ran")</script><img src="invalid:" onerror="parent.alert(\'event ran\')"><iframe src="https://example.com"></iframe><a href="javascript:parent.alert(1)">危险链接</a>\n\n```html\n<div>这段示例应保留代码</div>\n```'
    await page.route('**/api/sources/readme', route => route.fulfill({ json: { content, path: 'README.md', revision: 'main', url: '' } }))
    await pinNavigation(page)
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '发现模型', exact: true }).click()
    await page.locator('.page-actions').getByRole('button', { name: '打开模型仓库', exact: true }).click()
    const open = page.getByRole('dialog', { name: '打开 Hugging Face 仓库', exact: true })
    await open.getByRole('textbox', { name: '仓库 ID', exact: true }).fill('fixture/README')
    await open.getByRole('button', { name: '读取文件清单', exact: true }).click()
    await page.getByRole('dialog', { name: '选择下载文件', exact: true }).getByRole('button', { name: '模型介绍', exact: true }).click()
    const intro = page.getByRole('dialog', { name: '模型介绍', exact: true })
    const document = intro.frameLocator('.readme-document')
    await expect(document.getByRole('heading', { name: '文档标题', exact: true })).toBeVisible()
    await expect(document.locator('script,iframe,[onerror]')).toHaveCount(0)
    await expect(document.getByRole('link', { name: '危险链接', exact: true })).toHaveCount(0)
    expect(await document.locator('body').evaluate(() => typeof (window as any).modloader)).toBe('undefined')
    await expect(document.locator('pre code')).toHaveText('<div>这段示例应保留代码</div>\n')
    await document.getByRole('link', { name: '跳到中文', exact: true }).click()
    expect(await intro.locator('.modal-body').evaluate(element => element.scrollTop)).toBeGreaterThan(500)
    await document.getByText('展开用法', { exact: true }).click()
    await expect(document.getByText('这里是使用方式', { exact: true })).toBeVisible()
    await expect.poll(() => document.locator('body').evaluate(body => Math.abs(body.scrollHeight - body.ownerDocument.documentElement.clientHeight))).toBeLessThanOrEqual(2)
    expect(alerts).toEqual([])
    await page.keyboard.press('Escape')
    await expect(intro).toHaveCount(0)
  } finally { await application.close() }
})
