import { test, expect, chromium } from '@playwright/test'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { pinNavigation } from './navigation'

test('调用记录设置：保留时间、目录与开关持久化，写入和清理分别报错重试', async () => {
  const root = path.resolve(__dirname, '..')
  const artifacts = process.env.MODLOADER_TEST_ARTIFACTS || path.join(root, '.artifacts/codex/20261030-external-api-history/root')
  fs.mkdirSync(artifacts, { recursive: true })
  const { preview } = await import('vite')
  const server = await preview({ root, configFile: false,
    build: { outDir: process.env.MODLOADER_TEST_DIST || path.join(root, 'dist') },
    preview: { host: '127.0.0.1', port: 0, strictPort: false } })
  const address = server.httpServer.address()
  if (!address || typeof address === 'string') throw new Error('Preview did not start')
  const browser = await chromium.launch({ channel: 'msedge', headless: true })
  const page = await browser.newPage({ viewport: { width: 1380, height: 1000 } })
  const settings = { api_port: 12340, api_history_enabled: false, api_history_directory: 'D:\\fixture\\history', api_history_retention_days: 0,
    api_history_status: { state: 'disabled', error: null as string | null, cleanup_error: null as string | null, directory: 'D:\\fixture\\history' } }
  const patches: object[] = [], errors: string[] = []
  let retries = 0
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/api/**', async route => {
    const pathname = new URL(route.request().url()).pathname
    if (pathname === '/api/settings' && route.request().method() === 'PATCH') {
      const update = route.request().postDataJSON()
      patches.push(update)
      Object.assign(settings, update)
      settings.api_history_status = { state: update.api_history_enabled === true ? 'error' : settings.api_history_enabled ? 'saved' : 'disabled',
        error: update.api_history_enabled === true ? '无法写入调用记录：磁盘空间不足' : null,
        cleanup_error: update.api_history_retention_days === 7 ? '过期记录清理失败：访问被拒绝' : null,
        directory: settings.api_history_directory }
      return route.fulfill({ json: settings })
    }
    if (pathname === '/api/request-history/retry') {
      retries++
      settings.api_history_status = { state: 'saved', error: null, cleanup_error: null, directory: settings.api_history_directory }
      return route.fulfill({ json: settings.api_history_status })
    }
    if (pathname === '/api/state') return route.fulfill({ json: { models: [], profiles: [], runtimes: [], instances: [],
      downloads: [], installations: [], hardware: {}, backends: [], settings } })
    if (pathname === '/api/conversations') return route.fulfill({ json: { items: [], active_id: null } })
    if (pathname === '/api/chat-presets') return route.fulfill({ json: { items: [], active_id: null } })
    return route.fulfill({ json: {} })
  })
  try {
    await page.goto(`http://127.0.0.1:${address.port}`)
    await pinNavigation(page)
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '设置', exact: true }).click()
    const toggle = page.getByRole('switch', { name: '保存调用记录', exact: true })
    const directory = page.getByLabel('记录目录', { exact: true })
    const retention = page.getByRole('combobox', { name: '调用记录保留时间', exact: true })
    await expect(toggle).toHaveAttribute('aria-checked', 'false')
    await expect(directory).toHaveValue('D:\\fixture\\history')
    await expect(retention).toHaveValue('0')
    await expect(retention.locator('option')).toHaveText(['7 天', '30 天', '永久'])
    await directory.fill('D:\\fixture\\changed-history')
    await directory.press('Enter')
    await expect.poll(() => settings.api_history_directory).toBe('D:\\fixture\\changed-history')
    await expect(toggle).toBeEnabled()
    await toggle.click()
    await expect(toggle).toHaveAttribute('aria-checked', 'true')
    await expect(page.getByRole('status', { name: '调用记录保存状态' })).toContainText('调用记录保存失败')
    await expect(page.getByText('无法写入调用记录：磁盘空间不足', { exact: true })).toBeVisible()
    await page.getByRole('button', { name: '重试写入', exact: true }).click()
    await expect(page.getByRole('status', { name: '调用记录保存状态' })).toContainText('调用记录已保存')
    expect(retries).toBe(1)
    await retention.selectOption('7')
    await expect.poll(() => settings.api_history_retention_days).toBe(7)
    await expect(page.getByRole('status', { name: '调用记录保存状态' })).toContainText('调用记录已保存')
    await expect(page.getByText('过期记录清理失败：访问被拒绝', { exact: true })).toBeVisible()
    await page.getByRole('button', { name: '重试清理', exact: true }).click()
    await expect(page.getByText('过期记录清理失败：访问被拒绝', { exact: true })).toHaveCount(0)
    expect(retries).toBe(2)
    await retention.selectOption('30')
    await expect.poll(() => settings.api_history_retention_days).toBe(30)
    await expect(retention).toBeEnabled()
    await page.screenshot({ path: path.join(artifacts, 'settings-history.png'), fullPage: true })
    await page.reload()
    await pinNavigation(page)
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '设置', exact: true }).click()
    await expect(toggle).toHaveAttribute('aria-checked', 'true')
    await expect(directory).toHaveValue('D:\\fixture\\changed-history')
    await expect(retention).toHaveValue('30')
    await retention.selectOption('0')
    await expect.poll(() => settings.api_history_retention_days).toBe(0)
    await expect(toggle).toBeEnabled()
    await toggle.click()
    await expect(page.getByRole('status', { name: '调用记录保存状态' })).toContainText('仅保留本次运行记录')
    expect(patches).toEqual([{ api_history_directory: 'D:\\fixture\\changed-history' },
      { api_history_enabled: true }, { api_history_retention_days: 7 }, { api_history_retention_days: 30 },
      { api_history_retention_days: 0 }, { api_history_enabled: false }])
    expect(errors).toEqual([])
  } finally {
    await browser.close()
    await new Promise<void>(resolve => server.httpServer.close(() => resolve()))
  }
})
