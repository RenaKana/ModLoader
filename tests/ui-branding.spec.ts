import { expect, test, _electron as electron } from '@playwright/test'
import * as fs from 'node:fs'
import * as path from 'node:path'

const root = path.resolve(__dirname, '..')
const artifacts = process.env.MODLOADER_TEST_ARTIFACT_DIR || path.join(root, '.artifacts/codex/ui-branding')

test('主界面、关于与 favicon 加载统一图标，品牌按钮保留模型库入口', async () => {
  test.setTimeout(90_000)
  fs.mkdirSync(artifacts, { recursive: true })
  const directory = fs.mkdtempSync(path.join(artifacts, 'run-'))
  const temp = path.join(directory, 'temp')
  fs.mkdirSync(temp)
  const application = await electron.launch({
    ...(process.env.MODLOADER_TEST_EXECUTABLE ? { executablePath: process.env.MODLOADER_TEST_EXECUTABLE, args: [] } : { args: [root] }),
    env: { ...process.env, MODLOADER_DATA_DIR: path.join(directory, 'data'), MODLOADER_SENSOR_AUTO_START: '0', TEMP: temp, TMP: temp },
    timeout: 60_000,
  })
  try {
    const page = await application.firstWindow()
    const errors: string[] = []
    page.on('pageerror', error => errors.push(error.message))
    await expect(page.locator('.app-shell')).toBeVisible({ timeout: 45_000 })
    const brand = page.getByRole('button', { name: 'ModLoader 模型库', exact: true })
    const headerIcon = brand.locator('img')
    await expect(headerIcon).toBeVisible()
    await expect.poll(() => headerIcon.evaluate((image: HTMLImageElement) => [image.complete, image.naturalWidth, image.naturalHeight])).toEqual([true, 256, 256])
    await expect(headerIcon).toHaveCSS('width', '28px')
    await expect(headerIcon).toHaveCSS('height', '28px')
    await page.mouse.move(600, 500)
    await expect(page.locator('.app-header')).not.toHaveClass(/is-open/)
    await page.screenshot({ path: path.join(directory, 'header-compact.png'), animations: 'disabled' })

    await brand.hover()
    const nav = page.getByRole('navigation', { name: '主导航' })
    await nav.getByRole('button', { name: '设置', exact: true }).click()
    await expect(nav.getByRole('button', { name: '设置', exact: true })).toHaveAttribute('aria-current', 'page')
    await brand.click()
    await expect(nav.getByRole('button', { name: '模型库', exact: true })).toHaveAttribute('aria-current', 'page')
    await expect(page.locator('.workspace > section').first()).toBeVisible()

    const favicon = await page.evaluate(async () => {
      const link = document.querySelector<HTMLLinkElement>('link[rel="icon"]')
      if (!link) throw Error('缺少 favicon')
      const image = new Image()
      image.src = link.href
      await image.decode()
      return { width: image.naturalWidth, height: image.naturalHeight }
    })
    expect(favicon).toEqual({ width: 256, height: 256 })

    await brand.hover()
    await page.getByRole('button', { name: '关于 ModLoader', exact: true }).click()
    const about = page.getByRole('dialog', { name: 'ModLoader', exact: true })
    await expect(about).toBeVisible()
    const aboutIcon = about.locator('.about-brand img')
    await expect(aboutIcon).toBeVisible()
    await expect(aboutIcon).toHaveAttribute('src', await headerIcon.getAttribute('src') || '')
    await expect.poll(() => aboutIcon.evaluate((image: HTMLImageElement) => [image.complete, image.naturalWidth, image.naturalHeight])).toEqual([true, 256, 256])
    await expect(aboutIcon).toHaveCSS('width', '70px')
    await expect(aboutIcon).toHaveCSS('height', '70px')
    await page.screenshot({ path: path.join(directory, 'about.png'), animations: 'disabled' })
    await page.keyboard.press('Escape')
    await expect(about).toBeHidden()
    expect(errors).toEqual([])
    console.log(JSON.stringify({ screenshots: directory }))
  } finally {
    await application.close()
  }
})
