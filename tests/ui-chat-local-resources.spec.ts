import { test, expect, _electron as electron } from '@playwright/test'
import { build } from 'esbuild'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { pathToFileURL } from 'node:url'
import type { LocalResource } from '../src/chat-local-resources'

const root = path.resolve(__dirname, '..')
const artifacts = path.join(root, '.artifacts/codex/chat-turn-update-20261027/local-resources')
const directory = 'D:\\isolated-attachments'
const file: LocalResource = { id: 'f'.repeat(43), directory, path: '报告.txt', name: '报告.txt', size: 1024, mime: 'text/plain', kind: 'file', can_open: true }
const script: LocalResource = { ...file, id: 's'.repeat(43), path: 'script.ps1', name: 'script.ps1', can_open: false }
const image: LocalResource = { ...file, id: 'i'.repeat(43), path: '图示.png', name: '图示.png', size: 68, mime: 'image/png', kind: 'image' }
const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=', 'base64')
let url: string, main: string

test.beforeAll(async () => {
  fs.mkdirSync(artifacts, { recursive: true })
  const fixture = fs.mkdtempSync(path.join(artifacts, 'ui-fixture-'))
  main = path.join(fixture, 'main.cjs')
  fs.writeFileSync(main, `const {app,BrowserWindow}=require('electron');app.setPath('userData',${JSON.stringify(path.join(fixture, 'desktop'))});app.whenReady().then(()=>{const window=new BrowserWindow({show:false,width:1100,height:820});window.loadURL('about:blank')});app.on('window-all-closed',()=>app.quit());`)
  fs.writeFileSync(path.join(fixture, 'index.html'), '<!doctype html><html lang="zh-CN"><meta charset="UTF-8"><link rel="stylesheet" href="./fixture.css"><div id="root"></div><script type="module" src="./fixture.js"></script></html>')
  fs.writeFileSync(path.join(fixture, 'fixture.tsx'), `import React, {useState} from 'react';import {createRoot} from 'react-dom/client';import {ChatLocalResources} from ${JSON.stringify(path.join(root, 'src/ChatLocalResources').replaceAll('\\', '/'))};import ${JSON.stringify(path.join(root, 'src/styles.css').replaceAll('\\', '/'))};
    const resources=${JSON.stringify([file, script, image])};function Fixture(){const [directory,setDirectory]=useState(${JSON.stringify(directory)}); window.revoke=()=>setDirectory('');return <main style={{maxWidth:730,margin:'40px auto',padding:24,background:'white'}}><h2>本地附件</h2><ChatLocalResources resources={resources} directory={directory}/></main>}createRoot(document.getElementById('root')).render(<Fixture/>);`)
  await build({ entryPoints: [path.join(fixture, 'fixture.tsx')], outfile: path.join(fixture, 'fixture.js'), bundle: true, format: 'esm', jsx: 'automatic', loader: { '.woff2': 'file' }, define: { 'process.env.NODE_ENV': '"development"' } })
  url = pathToFileURL(path.join(fixture, 'index.html')).href
})

test('whole card opens details; reveal stays independent; failure is inline; revoked images close', async () => {
  test.setTimeout(60_000)
  const temp = path.join(artifacts, 'electron-temp')
  fs.mkdirSync(temp, { recursive: true })
  const app = await electron.launch({ args: [main], env: { ...process.env, TEMP: temp, TMP: temp }, timeout: 30_000 })
  try {
  const page = await app.firstWindow()
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  await page.setViewportSize({ width: 1100, height: 820 })
  await page.addInitScript(() => {
    const state = window as any
    state.actions = []; state.failAction = false
    state.modloader = { connection: async () => ({ baseUrl: 'http://127.0.0.1:12345', token: '' }), localResourceAction: async (id: string, action: string) => {
      state.actions.push({ id, action }); if (state.failAction) throw new Error('系统未能打开此附件'); return { handed_off: true }
    } }
  })
  await page.route('**/api/chat-tools/resources/**', route => {
    const id = route.request().url().split('/resources/')[1].split('/')[0]
    if (route.request().url().endsWith('/content')) return route.fulfill({ contentType: 'image/png', body: png })
    return route.fulfill({ json: [file, script, image].find(item => item.id === id) })
  })
  await page.goto(url)
  const fileCard = page.locator('.chat-local-resource').filter({ has: page.getByRole('button', { name: '附件详情：报告.txt', exact: true }) })
  await fileCard.getByRole('button', { name: '附件详情：报告.txt', exact: true }).click({ position: { x: 400, y: 45 } })
  const detail = page.getByRole('dialog', { name: '附件详情', exact: true })
  await expect(detail).toBeVisible()
  await expect(detail).toContainText(directory)
  await detail.getByRole('button', { name: '用默认应用打开', exact: true }).click()
  await expect(detail.getByRole('button', { name: '已交给默认应用', exact: true })).toBeVisible()
  await page.screenshot({ path: path.join(artifacts, 'local-attachment-detail.png') })
  await expect(page.locator('.toast')).toHaveCount(0)
  await detail.getByRole('button', { name: '关闭', exact: true }).click()
  await fileCard.getByRole('button', { name: '在文件夹中定位', exact: true }).click()
  await expect(detail).toHaveCount(0)
  await expect(fileCard.getByRole('button', { name: '已请求文件夹定位', exact: true })).toBeVisible()
  expect(await page.evaluate(() => (window as any).actions)).toEqual([{ id: file.id, action: 'open' }, { id: file.id, action: 'reveal' }])
  await page.getByRole('button', { name: '附件详情：script.ps1', exact: true }).click()
  await expect(detail.getByRole('button', { name: '用默认应用打开', exact: true })).toBeDisabled()
  await detail.getByRole('button', { name: '关闭', exact: true }).click()
  await page.evaluate(() => { (window as any).failAction = true })
  await fileCard.getByRole('button', { name: '附件详情：报告.txt', exact: true }).click()
  await detail.getByRole('button', { name: '用默认应用打开', exact: true }).click()
  await expect(detail.getByRole('alert')).toContainText('系统未能打开此附件')
  await expect(detail.getByRole('button', { name: '用默认应用打开', exact: true })).toBeEnabled()
  await detail.getByRole('button', { name: '关闭', exact: true }).click()
  await page.screenshot({ path: path.join(artifacts, 'local-attachments.png') })
  await page.getByRole('button', { name: '查看图片：图示.png', exact: true }).click()
  await expect(page.locator('dialog[open]')).toBeVisible()
  await page.evaluate(() => (window as any).revoke())
  await expect(page.locator('dialog[open]')).toHaveCount(0)
  await expect(page.locator('.chat-resource-image img')).toHaveCount(0)
  await expect(fileCard).toContainText('附件所属目录已不再授权')
  expect(errors).toEqual([])
  } finally { await app.close() }
})
