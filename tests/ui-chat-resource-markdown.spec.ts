import { test, expect, _electron as electron } from '@playwright/test'
import { build } from 'esbuild'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { pathToFileURL } from 'node:url'
import type { LocalResource } from '../src/chat-local-resources'

const root = path.resolve(__dirname, '..')
const artifacts = path.join(root, '.artifacts/codex/chat-turn-update-20261027/resource-markdown')
const directory = 'D:\\isolated-attachments'
const file: LocalResource = { id: 'f'.repeat(43), directory, path: '设计说明.md', name: '设计说明.md', size: 1258, mime: 'text/markdown', kind: 'file', can_open: true }
const picture: LocalResource = { ...file, id: 'i'.repeat(43), path: '交互示意.png', name: '交互示意.png', mime: 'image/png', kind: 'image' }
let url: string, main: string

test.beforeAll(async () => {
  fs.mkdirSync(artifacts, { recursive: true })
  const fixture = fs.mkdtempSync(path.join(artifacts, 'fixture-'))
  main = path.join(fixture, 'main.cjs')
  fs.writeFileSync(main, `const {app,BrowserWindow}=require('electron');app.setPath('userData',${JSON.stringify(path.join(fixture, 'desktop'))});app.whenReady().then(()=>{const window=new BrowserWindow({show:false,width:1100,height:900});window.loadURL('about:blank')});app.on('window-all-closed',()=>app.quit());`)
  fs.writeFileSync(path.join(fixture, 'index.html'), '<!doctype html><html lang="zh-CN"><meta charset="UTF-8"><link rel="stylesheet" href="./fixture.css"><div id="root"></div><script type="module" src="./fixture.js"></script></html>')
  const markdown = `已找到文件与图片。\n\n[设计说明](modloader-resource:${file.id})\n\n![交互示意](modloader-resource:${picture.id})\n\n\`\`\`md\n[只是代码](modloader-resource:${file.id})\n\`\`\`\n\n[未知引用](modloader-resource:${'x'.repeat(43)})`
  fs.writeFileSync(path.join(fixture, 'fixture.tsx'), `import React,{useState} from 'react';import {createRoot} from 'react-dom/client';import {MarkdownContent} from ${JSON.stringify(path.join(root, 'src/MessageContent').replaceAll('\\', '/'))};import {ImageGallery} from ${JSON.stringify(path.join(root, 'src/ImageViewer').replaceAll('\\', '/'))};import ${JSON.stringify(path.join(root, 'src/styles.css').replaceAll('\\', '/'))};
    const resources=${JSON.stringify([file, picture])};function Fixture(){const [directory,setDirectory]=useState(${JSON.stringify(directory)});const [text,setText]=useState(${JSON.stringify(markdown)});window.revoke=()=>setDirectory('');window.fallback=()=>setText('附件已整理。');return <main style={{maxWidth:760,margin:'32px auto',padding:28,background:'white',borderRadius:18}}><ImageGallery className="message-body"><MarkdownContent text={text} resources={resources} directory={directory}/></ImageGallery></main>}createRoot(document.getElementById('root')).render(<Fixture/>);`)
  await build({ entryPoints: [path.join(fixture, 'fixture.tsx')], outfile: path.join(fixture, 'fixture.js'), bundle: true, format: 'esm', jsx: 'automatic', loader: { '.woff2': 'file', '.woff': 'file', '.ttf': 'file' }, define: { 'process.env.NODE_ENV': '"development"' } })
  url = pathToFileURL(path.join(fixture, 'index.html')).href
})

test('controlled Markdown references own the image viewer, save blobs and close on revoke', async () => {
  test.setTimeout(60_000)
  const temp = path.join(artifacts, 'electron-temp')
  fs.mkdirSync(temp, { recursive: true })
  const app = await electron.launch({ args: [main], env: { ...process.env, TEMP: temp, TMP: temp } })
  try {
    const page = await app.firstWindow()
    const errors: string[] = []
    page.on('pageerror', error => errors.push(error.message))
    const pngData = await page.evaluate(() => {
      const canvas = document.createElement('canvas'); canvas.width = 640; canvas.height = 300
      const context = canvas.getContext('2d')!
      context.fillStyle = '#ecf2ff'; context.fillRect(0, 0, 640, 300)
      context.fillStyle = '#4364dc'; context.fillRect(48, 48, 112, 204)
      context.fillStyle = '#9eafe5'; context.fillRect(180, 48, 412, 84)
      context.fillStyle = '#ccd6f4'; context.fillRect(180, 152, 192, 100); context.fillRect(392, 152, 200, 100)
      return canvas.toDataURL('image/png').split(',')[1]
    })
    await page.addInitScript(() => {
      const state = window as any
      state.savedImages = []
      state.modloader = { connection: async () => ({ baseUrl: 'http://127.0.0.1:12345', token: '' }), localResourceAction: async () => ({ handed_off: true }), saveImage: async (source: string, name: string) => { state.savedImages.push({ source, name }); return { canceled: false } } }
    })
    const requested: string[] = []
    await page.route('**/api/chat-tools/resources/**', route => {
      const id = route.request().url().split('/resources/')[1].split('/')[0]
      requested.push(id)
      if (route.request().url().endsWith('/content')) return route.fulfill({ contentType: 'image/png', body: Buffer.from(pngData, 'base64') })
      return route.fulfill({ json: [file, picture].find(item => item.id === id) })
    })
    await page.goto(url)
    await expect(page.locator('.chat-local-resource')).toHaveCount(2)
    await expect(page.locator('pre')).toContainText('[只是代码](modloader-resource:')
    expect(requested).not.toContain('x'.repeat(43))
    await expect(page.getByText('未知引用', { exact: true })).toBeVisible()
    await expect(page.getByRole('link', { name: '未知引用' })).toHaveCount(0)
    await page.getByRole('button', { name: '附件详情：设计说明.md', exact: true }).click({ position: { x: 450, y: 25 } })
    await expect(page.getByRole('dialog', { name: '附件详情', exact: true })).toBeVisible()
    await page.getByRole('button', { name: '关闭', exact: true }).click()
    await expect(page.locator('.chat-resource-image img')).toHaveJSProperty('naturalWidth', 640)
    await page.screenshot({ path: path.join(artifacts, 'markdown-local-attachments.png') })
    await page.locator('.chat-resource-image img').click()
    const viewer = page.getByRole('dialog', { name: '图片查看器', exact: true })
    await expect(viewer).toBeVisible()
    await expect(viewer.getByRole('button', { name: '保存图片', exact: true })).toBeEnabled()
    await viewer.getByRole('button', { name: '保存图片', exact: true }).click()
    await expect.poll(() => page.evaluate(() => (window as any).savedImages.length)).toBe(1)
    const saved = await page.evaluate(() => (window as any).savedImages[0])
    expect(saved.source).toBe(`data:image/png;base64,${pngData}`)
    expect(saved.name).toBe(picture.name)
    await page.evaluate(() => (window as any).revoke())
    await expect(viewer).toHaveCount(0)
    await expect(page.locator('.chat-resource-image img')).toHaveCount(0)
    await expect(page.getByRole('button', { name: '在文件夹中定位', exact: true }).first()).toBeDisabled()
    await page.evaluate(() => (window as any).fallback())
    await expect(page.locator('.chat-local-resource')).toHaveCount(2)
    expect(errors).toEqual([])
  } finally { await app.close() }
})
