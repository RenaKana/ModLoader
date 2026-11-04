import { pinNavigation } from './navigation'
import { test, expect, _electron as electron } from '@playwright/test'
import * as path from 'node:path'
import * as fs from 'node:fs'

test('文件夹选择后递归导入、多次扫描去重并展示单项失败', async () => {
  test.setTimeout(90_000)
  const root = path.resolve(__dirname, '..')
  const task = fs.mkdtempSync(path.join(root, 'work', 'ui-folder-import-'))
  const folder = path.join(task, 'models')
  const deep = path.join(folder, '厂商', '系列', '版本', '量化', '第五层')
  fs.mkdirSync(deep, { recursive: true })
  const header = Buffer.alloc(24)
  header.write('GGUF'); header.writeUInt32LE(3, 4); header.writeBigUInt64LE(1n, 8)
  fs.writeFileSync(path.join(deep, 'deep-model.gguf'), header)
  fs.writeFileSync(path.join(deep, 'broken.gguf'), 'invalid')
  const app = await electron.launch({
    ...(process.env.MODLOADER_TEST_PACKAGED ? { executablePath: path.join(root, 'release', 'win-unpacked', 'ModLoader.exe'), args: [] } : { args: [root] }),
    env: { ...process.env, MODLOADER_DATA_DIR: path.join(task, 'data') }, timeout: 60_000,
  })
  const page = await app.firstWindow()
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  try {
    await pinNavigation(page)
    await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
    // Supply the native chooser's result; all scanning and persistence use the real core.
    await app.evaluate(({ dialog }, selected) => { dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [selected] }) }, folder)
    await page.getByRole('button', { name: '导入文件夹', exact: true }).click()
    const modal = page.getByRole('dialog', { name: '导入模型文件夹' })
    await modal.getByRole('button', { name: '浏览', exact: true }).click()
    await expect(modal.getByRole('textbox')).toHaveValue(folder)
    await modal.getByRole('button', { name: '扫描并导入', exact: true }).click()
    await expect(modal.getByText('新增 1 个', { exact: true })).toBeVisible()
    await expect(modal.getByText('未导入 1 项', { exact: true })).toBeVisible()
    await expect(modal.getByText(/broken\.gguf$/)).toBeVisible()
    await modal.getByText('已导入的模型（1）', { exact: true }).click()
    await expect(modal.getByText('deep-model', { exact: true })).toBeVisible()
    await modal.getByRole('button', { name: '重新扫描', exact: true }).click()
    await expect(modal.getByText('新增 0 个', { exact: true })).toBeVisible()
    await expect(modal.getByText('已存在 1 个', { exact: true })).toBeVisible()
    await app.evaluate(({ BrowserWindow }) => { const win=BrowserWindow.getAllWindows()[0]; win.setSize(1050, 780); win.webContents.setZoomFactor(1.25) })
    const viewport = await modal.evaluate(element => ({ width:element.scrollWidth,client:element.clientWidth }))
    expect(viewport.width).toBeLessThanOrEqual(viewport.client+1)
    const screenshot=await app.evaluate(async ({ BrowserWindow }) => (await BrowserWindow.getAllWindows()[0].webContents.capturePage()).toPNG().toString('base64'))
    fs.mkdirSync(path.join(root,'work','screenshots'),{recursive:true})
    fs.writeFileSync(path.join(root,'work','screenshots','folder-import-result.png'),Buffer.from(screenshot,'base64'))
    await modal.getByRole('button', { name: '完成', exact: true }).click()
    await expect(page.locator('.model-item')).toHaveCount(1)
    const state=await page.evaluate(async () => { const {baseUrl,token}=await window.modloader!.connection();return (await fetch(baseUrl+'/api/state',{headers:{Authorization:'Bearer '+token}})).json() })
    expect(state.models[0].path).toBe(path.join(deep,'deep-model.gguf'))
    expect(state.models[0].managed).toBe(false)
    expect(state.instances).toHaveLength(0)
    expect(state.downloads).toHaveLength(0)
    expect(errors).toEqual([])
  } finally { await app.close() }
})
