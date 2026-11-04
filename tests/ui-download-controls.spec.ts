import { test, expect, _electron as electron } from '@playwright/test'
import * as path from 'node:path'
import * as fs from 'node:fs'
import * as http from 'node:http'
import { pinNavigation } from './navigation'

test('本地 HTTP 下载完整流程：暂停保留字节、继续完成、暂停后停止', async () => {
  test.setTimeout(90_000)
  const root = path.resolve(__dirname, '..')
  const directory = fs.mkdtempSync(path.join(root, 'work', 'ui-download-control-live-'))
  const data = Buffer.alloc(16 * 1024 * 1024, 0x42)
  data.write('GGUF')
  data.writeUInt32LE(3, 4)
  data.writeBigUInt64LE(1n, 8)
  data.writeBigUInt64LE(0n, 16)
  let transfers = 0
  const server = http.createServer((request, response) => {
    response.writeHead(200, { 'Content-Length': data.length, 'Content-Type': 'application/octet-stream', ETag: '"ui-download-fixture"' })
    if (request.method === 'HEAD') return response.end()
    transfers++
    let offset = 0
    const timer = setInterval(() => {
      if (response.writableNeedDrain) return
      const next = Math.min(offset + 64 * 1024, data.length)
      response.write(data.subarray(offset, next))
      offset = next
      if (offset === data.length) { clearInterval(timer); response.end() }
    }, 30)
    response.on('close', () => clearInterval(timer))
  })
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const address = server.address() as { port: number }
  const application = await electron.launch({
    ...(process.env.MODLOADER_TEST_EXECUTABLE ? { executablePath: process.env.MODLOADER_TEST_EXECUTABLE, args: [] } : { args: [root] }),
    env: { ...process.env, MODLOADER_DATA_DIR: directory }, timeout: 60_000,
  })
  try {
    const page = await application.firstWindow()
    const errors: string[] = []
    page.on('pageerror', error => errors.push(error.message))
    await pinNavigation(page)
    await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 30_000 })
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '发现模型', exact: true }).click()
    const panel = page.locator('.download-panel')
    await panel.getByRole('button', { name: '全部任务', exact: true }).click()
    async function createDownload(filename: string) {
      await page.getByRole('button', { name: '直接链接下载', exact: true }).click()
      const dialog = page.getByRole('dialog', { name: '直接链接下载', exact: true })
      await dialog.getByRole('textbox', { name: '下载链接' }).fill(`http://127.0.0.1:${address.port}/model.gguf`)
      await dialog.getByRole('textbox', { name: /^保存文件名/ }).fill(filename)
      await dialog.getByRole('button', { name: '检查下载', exact: true }).click()
      await dialog.getByRole('button', { name: '开始下载', exact: true }).click()
      await expect(dialog).toBeHidden()
    }
    const jobs = () => page.evaluate(async () => {
      const { baseUrl, token } = await window.modloader!.connection()
      const state = await (await fetch(`${baseUrl}/api/state`, { headers: { Authorization: `Bearer ${token}` } })).json()
      return state.downloads as Array<{ id: string; state: string; completed_bytes: number; request: { destination: string } }>
    })
    await createDownload('completed.gguf')
    const row = panel.locator('.job-row').first()
    await row.getByRole('button', { name: '暂停', exact: true }).click()
    await expect(row.getByText('已暂停', { exact: true })).toBeVisible({ timeout: 15_000 })
    const paused = (await jobs())[0]
    const output = path.join(paused.request.destination, 'completed.gguf')
    const part = `${output}.modloader-part`
    const pausedBytes = fs.statSync(part).size
    expect(pausedBytes).toBeGreaterThan(0)
    expect(pausedBytes).toBeLessThan(data.length)
    // Observe two normal UI refresh periods while the real stream is paused.
    await page.waitForTimeout(2200)
    expect(fs.statSync(part).size).toBe(pausedBytes)
    expect((await jobs())[0].state).toBe('paused')
    await row.getByRole('button', { name: '继续', exact: true }).click()
    await expect(row.getByText('已完成', { exact: true })).toBeVisible({ timeout: 30_000 })
    expect(fs.readFileSync(output).equals(data)).toBe(true)
    expect((await jobs())[0].id).toBe(paused.id)
    expect(transfers).toBe(1)

    await createDownload('stopped.gguf')
    const pending = panel.locator('.job-row').filter({ has: page.getByRole('button', { name: '暂停', exact: true }) })
    await pending.getByRole('button', { name: '暂停', exact: true }).click()
    const stopped = panel.locator('.job-row').filter({ hasText: '已暂停' })
    await expect(stopped).toHaveCount(1)
    const second = (await jobs()).find(job => job.id !== paused.id)!
    const stoppedPart = path.join(second.request.destination, 'stopped.gguf.modloader-part')
    await stopped.getByRole('button', { name: '停止', exact: true }).click()
    await expect(panel.getByText('已停止', { exact: true })).toBeVisible()
    const stoppedBytes = fs.statSync(stoppedPart).size
    await page.waitForTimeout(1200)
    expect(fs.statSync(stoppedPart).size).toBe(stoppedBytes)
    expect(fs.existsSync(path.join(second.request.destination, 'stopped.gguf'))).toBe(false)
    expect((await jobs()).find(job => job.id === second.id)?.state).toBe('cancelled')
    expect(errors).toEqual([])
    console.log(JSON.stringify({ dataDirectory: directory, transfers, pausedBytes, stoppedBytes }))
  } finally {
    await application.close()
    server.closeAllConnections()
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
})
