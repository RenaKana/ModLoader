import { pinNavigation } from './navigation'
import { test, expect, _electron as electron, type Locator } from '@playwright/test'
import * as path from 'node:path'
import * as fs from 'node:fs'

test('设置密钥：反复显示与隐藏，凭据草稿仅在离开或 Enter 时保存', async () => {
  test.setTimeout(90_000)
  const root = path.resolve(__dirname, '..')
  const artifacts = path.join(root, '.artifacts/codex/20261026-web-services')
  fs.mkdirSync(artifacts, { recursive: true })
  const directory = fs.mkdtempSync(path.join(artifacts, 'key-visibility-'))
  const screenshots = path.join(directory, 'screenshots')
  fs.mkdirSync(screenshots, { recursive: true })
  const application = await electron.launch({
    ...(process.env.MODLOADER_TEST_EXECUTABLE ? { executablePath: process.env.MODLOADER_TEST_EXECUTABLE, args: [] } : { args: [root] }),
    env: { ...process.env, MODLOADER_DATA_DIR: path.join(directory, 'data') }, timeout: 60_000,
  })
  const page = await application.firstWindow()
  const errors: string[] = []
  const externalRequests: string[] = []
  const unexpectedMutations: string[] = []
  const updates: Record<string, unknown>[] = []
  const configured: Record<string, boolean> = { hf: false, modelscope: false }
  let fixtureKey = 'modloader-ui-fake-api-key'
  let keyReads = 0
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/*', async route => {
    const request = route.request()
    const url = new URL(request.url())
    if (['http:', 'https:'].includes(url.protocol) && !['127.0.0.1', 'localhost'].includes(url.hostname)) {
      externalRequests.push(url.origin)
      await route.abort()
    } else if (url.pathname.startsWith('/api/') && !['GET', 'HEAD', 'OPTIONS'].includes(request.method())) {
      unexpectedMutations.push(`${request.method()} ${url.pathname}`)
      await route.abort()
    } else await route.continue()
  })
  await page.route('**/api/state', async route => {
    const response = await route.fetch()
    const state = await response.json()
    await route.fulfill({ response, json: { ...state, settings: { ...state.settings,
      font_scale: 1, hf_configured: configured.hf, hf_token_set: false,
      modelscope_configured: configured.modelscope, modelscope_token_set: false,
    } } })
  })
  // This endpoint is always fulfilled locally; no stored credential is fetched.
  await page.route('**/api/settings/api-key', async route => {
    keyReads++
    await route.fulfill({ json: { key: fixtureKey } })
  })
  await page.route('**/api/settings', async route => {
    if (route.request().method() !== 'PATCH') { await route.abort(); return }
    const update = route.request().postDataJSON() as Record<string, unknown>
    updates.push(update)
    if (typeof update.api_key === 'string') fixtureKey = update.api_key
    if (update.secrets && typeof update.secrets === 'object') {
      for (const [source, value] of Object.entries(update.secrets)) configured[source] = Boolean(value)
    }
    await route.fulfill({ json: {} })
  })
  async function clickIcon(button: Locator) {
    // Native SVG hit targets also need to toggle once per physical click.
    await button.locator('svg').click()
  }
  async function nativeScreenshot(name: string) {
    await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))))
    const png = await application.evaluate(async ({ BrowserWindow }) => (await BrowserWindow.getAllWindows()[0].webContents.capturePage()).toPNG().toString('base64'))
    fs.writeFileSync(path.join(screenshots, name), Buffer.from(png, 'base64'))
  }

  try {
    await pinNavigation(page)
    await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
    await application.evaluate(({ BrowserWindow }) => { const window = BrowserWindow.getAllWindows()[0]; window.setSize(1440, 960); window.webContents.setZoomFactor(1) })
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '设置', exact: true }).click()
    await page.evaluate(() => document.fonts.ready)
    const localKey = page.locator('.api-key-control > code')

    await test.step('本地 API Key 反复显示后可再次隐藏，并且查看按钮只读取一次', async () => {
      await expect(localKey).toHaveText('••••••••••••••••••••••••')
      expect(keyReads).toBe(0)
      for (let cycle = 0; cycle < 3; cycle++) {
        await clickIcon(page.getByRole('button', { name: '查看 API Key', exact: true }))
        await expect(localKey).toHaveText(fixtureKey)
        await clickIcon(page.getByRole('button', { name: '隐藏 API Key', exact: true }))
        await expect(localKey).toHaveText('••••••••••••••••••••••••')
      }
      expect(keyReads).toBe(1)
      expect(updates).toEqual([])
      await nativeScreenshot('local-key-hidden.png')
    })

    await test.step('两个来源的凭据显示按钮保留焦点与草稿，不触发保存', async () => {
      for (const [source, label] of [['hf', 'Hugging Face Token'], ['modelscope', 'ModelScope Token']] as const) {
        const input = page.getByLabel(label, { exact: true })
        const draft = `fake-${source}-token`
        await input.fill(draft)
        const before = updates.length
        for (let cycle = 0; cycle < 2; cycle++) {
          await clickIcon(page.getByRole('button', { name: `显示 ${label}`, exact: true }))
          await expect(input).toHaveAttribute('type', 'text')
          await clickIcon(page.getByRole('button', { name: `隐藏 ${label}`, exact: true }))
          await expect(input).toHaveAttribute('type', 'password')
          await expect(input).toHaveValue(draft)
          await expect(input).toBeFocused()
        }
        expect(updates).toHaveLength(before)
        if (source === 'hf') await input.press('Enter')
        else await page.getByRole('heading', { name: '模型源凭据', exact: true }).click()
        await expect.poll(() => updates.length).toBe(before + 1)
        expect(updates[before]).toEqual({ secrets: { [source]: draft } })
        await expect(input).toHaveValue('')
      }
    })

    await test.step('更换密钥草稿可反复显示与隐藏，取消保留旧密钥，确认后重新按需读取', async () => {
      await page.getByRole('button', { name: '更换', exact: true }).click()
      const dialog = page.getByRole('dialog', { name: '更换本地 API Key', exact: true })
      const input = dialog.getByPlaceholder('输入新密钥', { exact: true })
      const draft = 'modloader-ui-fake-new-key'
      await input.fill(draft)
      const before = updates.length
      for (let cycle = 0; cycle < 2; cycle++) {
        await clickIcon(dialog.getByRole('button', { name: '显示新密钥', exact: true }))
        await expect(input).toHaveAttribute('type', 'text')
        await clickIcon(dialog.getByRole('button', { name: '隐藏新密钥', exact: true }))
        await expect(input).toHaveAttribute('type', 'password')
        await expect(input).toHaveValue(draft)
      }
      expect(updates).toHaveLength(before)
      await dialog.getByRole('button', { name: '取消', exact: true }).click()
      await expect(dialog).toBeHidden()
      expect(updates).toHaveLength(before)
      await clickIcon(page.getByRole('button', { name: '查看 API Key', exact: true }))
      await expect(localKey).toHaveText(fixtureKey)
      await page.getByRole('button', { name: '更换', exact: true }).click()
      await input.fill(draft)
      await dialog.getByRole('button', { name: '更换密钥', exact: true }).click()
      await expect(dialog).toBeHidden()
      expect(updates[before]).toEqual({ api_key: draft })
      await expect(localKey).toHaveText('••••••••••••••••••••••••')
      await clickIcon(page.getByRole('button', { name: '查看 API Key', exact: true }))
      await expect(localKey).toHaveText(draft)
      expect(keyReads).toBe(2)
      await clickIcon(page.getByRole('button', { name: '隐藏 API Key', exact: true }))
      await expect(localKey).toHaveText('••••••••••••••••••••••••')
      await nativeScreenshot('new-key-hidden.png')
    })

    expect(externalRequests).toEqual([])
    expect(unexpectedMutations).toEqual([])
    expect(errors).toEqual([])
  } finally {
    await application.close()
  }
})
