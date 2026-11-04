import { expect, test, chromium } from '@playwright/test'
import { createServer } from 'vite'
import * as fs from 'node:fs'
import * as path from 'node:path'
import type { Message, ToolCall } from '../src/chat-history'
import type { ChatActivityEvent } from '../src/chat-activity'

const call = (name: string, args: unknown, id = 'reused-call'): ToolCall => ({ id, type: 'function', function: { name, arguments: JSON.stringify(args) } })
const tool = (id: string, name: string, args: unknown, status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled', result?: unknown): ChatActivityEvent => ({ id, kind: 'tool', call: call(name, args), status, ...(result === undefined ? {} : { result: JSON.stringify(result) }) })

test('工具按实际位置内联呈现，详情默认折叠，状态与折叠动效保留阅读位置和键盘操作', async () => {
  test.setTimeout(90_000)
  const root = path.resolve(__dirname, '..')
  const artifacts = path.join(root, '.artifacts/codex/chat-refinement-20261027/tool-timeline')
  fs.mkdirSync(artifacts, { recursive: true })
  const directory = fs.mkdtempSync(path.join(artifacts, 'renderer-'))
  fs.writeFileSync(path.join(directory, 'index.html'), `<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>工具时间线</title>
    <main class="chat-section" style="padding:32px;background:white;max-width:820px;margin:auto"><div id="activity"></div><div class="message-text"><p>最终答案保留在思考过程之外。</p></div></main>
    <script type="module">
      import React from 'react';import{createRoot}from'react-dom/client';import{flushSync}from'react-dom';
      import{ChatActivity}from'/src/ChatActivity.tsx';import{ReasoningContent}from'/src/MessageContent.tsx';import'/src/styles.css';import'/src/chat-layout.css';
      const root=createRoot(document.getElementById('activity'));
      window.renderActivityFixture=props=>flushSync(()=>root.render(React.createElement(ChatActivity,{...props,key:props.message.id})));
      window.renderLegacyReasoningFixture=props=>flushSync(()=>root.render(React.createElement(ReasoningContent,props)));
    </script></html>`)
  const server = await createServer({ root, cacheDir: path.join(directory, 'vite-cache'),
    optimizeDeps: { noDiscovery: true, include: ['react', 'react/jsx-runtime', 'react/jsx-dev-runtime', 'react-dom/client', 'react-dom', 'react-markdown', 'remark-gfm', 'remark-math', 'rehype-katex', 'highlight.js', 'dompurify', 'lucide-react'] },
    server: { host: '127.0.0.1', port: 0, watch: { ignored: ['**/.artifacts/**', '**/work/**', '**/release/**', '**/data/**'] } },
  })
  await server.listen()
  const browser = await chromium.launch({ channel: 'msedge' })
  try {
    const page = await browser.newPage({ viewport: { width: 1000, height: 820 } })
    const errors: string[] = []
    page.on('pageerror', error => errors.push(error.message))
    const address = server.httpServer!.address() as { port: number }
    await page.goto(`http://127.0.0.1:${address.port}/${path.relative(root, directory).replaceAll('\\', '/')}/index.html`, { waitUntil: 'commit' })
    await page.waitForFunction(() => typeof (window as any).renderActivityFixture === 'function')
    const base: Message = { id: 'fixture', role: 'assistant', text: '最终答案保留在思考过程之外。', images: [], createdAt: 1, status: 'generating', thinking: true }
    const events: ChatActivityEvent[] = [
      { id: 'reason-1', kind: 'reasoning', text: '先查询官方文档，再核对正文。' },
      tool('search-1', 'search_web', { query: 'Qwen3.5 官方文档', extra: '默认隐藏的完整参数' }, 'completed', { sources: [{ title: '官方文档', url: 'https://example.com' }] }),
      { id: 'reason-2', kind: 'reasoning', text: '搜索命中之后，读取相关页面。' },
      tool('read-1', 'read_page', { url: 'https://example.com/docs', start_line: 81, line_count: 80 }, 'running'),
    ]
    let fixtureId = base.id
    const render = (activity: ChatActivityEvent[], generating = true, detail: 'compact' | 'full' = 'compact', more: Partial<Message> = {}) => page.evaluate(props => (window as any).renderActivityFixture(props), { message: { ...base, id: fixtureId, ...more, activity }, generating, detail })
    const outer = page.locator('.message-activity')
    const region = page.getByRole('region', { name: '思考内容', includeHidden: true })
    const summary = outer.locator(':scope > summary')

    await test.step('同一思考容器按事件顺序显示，参数与原始结果默认不占页面', async () => {
      await render(events)
      await expect(outer).toHaveAttribute('open', '')
      expect(await page.locator('.chat-activity-events > [data-activity-id]').evaluateAll(elements => elements.map(element => element.getAttribute('data-activity-id')))).toEqual(['reason-1', 'search-1', 'reason-2', 'read-1'])
      const search = page.getByRole('button', { name: /搜索网页 Qwen3.5/ })
      await expect(search).toContainText('1 条结果')
      await expect(search).toHaveAttribute('aria-expanded', 'false')
      await expect(page.getByLabel('搜索网页参数', { exact: true })).toBeHidden()
      await expect(page.locator('[data-activity-id="read-1"] .chat-activity-status')).toHaveAttribute('data-status', 'running')
      await expect.poll(() => outer.locator('.reasoning-collapse').evaluate(element => getComputedStyle(element).opacity)).toBe('1')
      await page.locator('main').screenshot({ path: path.join(directory, 'compact-running.png') })
      await search.focus(); await search.press('Enter')
      await expect(page.getByLabel('搜索网页参数', { exact: true })).toBeVisible()
      await expect(page.getByLabel('搜索网页参数', { exact: true })).toContainText('默认隐藏的完整参数')
      await search.press('Enter')
      await expect(page.getByLabel('搜索网页参数', { exact: true })).toBeHidden()
    })

    await test.step('相同事件原位从运行转完成，full 可独立收起；拒绝与错误就地反馈', async () => {
      await page.evaluate(() => { (window as any).savedRead = document.querySelector('[data-activity-id="read-1"]') })
      events[3] = tool('read-1', 'read_page', { url: 'https://example.com/docs', start_line: 81, line_count: 80 }, 'completed', { start_line: 81, end_line: 160, total_lines: 310, content: '正文片段\n'.repeat(300) })
      events.push({ id: 'text-1', kind: 'text', text: '接下来检查本地目录。' }, tool('list-1', 'list_directory', {}, 'completed', { entries: Array.from({ length: 200 }, (_, index) => ({ name: `file-${index}.txt` })), total_entries: 350 }), tool('write-1', 'write_file', { path: 'note.txt', content: '写入内容' }, 'completed', { written: false, error: '用户拒绝了这次写入。不要重复请求同一变更。' }), tool('fail-1', 'read_file', { path: 'missing.txt' }, 'failed', { error: { message: '文件不存在。' } }), tool('custom-1', 'custom_tool', { query: '自定义请求' }, 'pending'))
      await render(events, true, 'full', { thinking: false })
      expect(await page.evaluate(() => (window as any).savedRead === document.querySelector('[data-activity-id="read-1"]'))).toBe(true)
      await expect(page.locator('[data-activity-id="read-1"] .chat-activity-status')).toHaveAttribute('data-status', 'completed')
      await expect(page.getByRole('button', { name: /读取网页/ })).toContainText('第 81–160 行')
      const list = page.getByRole('button', { name: /列出目录/ })
      await expect(list).toContainText('200 项 · 共 350 项')
      await expect(list).toHaveAttribute('aria-expanded', 'true')
      const fullList = page.getByLabel('列出目录结果', { exact: true })
      await expect(fullList).toContainText('file-199.txt')
      expect(await fullList.evaluate(element => element.clientHeight)).toBeLessThanOrEqual(220)
      expect(await fullList.evaluate(element => element.scrollHeight)).toBeGreaterThan(220)
      await expect(page.locator('[data-activity-id="write-1"]')).toHaveAttribute('data-status', 'cancelled')
      await expect(page.getByRole('button', { name: /写入文件/ })).toContainText('已拒绝')
      await expect(page.locator('[data-activity-id="fail-1"] .chat-activity-error')).toHaveText('文件不存在。')
      await expect(page.getByRole('button', { name: /custom_tool/ })).toContainText('待处理')
      await list.click()
      await expect(list).toHaveAttribute('aria-expanded', 'false')
      await render(events, true, 'full', { thinking: false })
      await expect(list).toHaveAttribute('aria-expanded', 'false')
      await render(events, true, 'compact', { thinking: false })
      await region.focus(); await region.press('End')
      await region.press('Home')
      await expect.poll(() => region.evaluate(element => element.scrollTop)).toBe(0)
      await expect.poll(() => page.locator('.chat-activity-tool-collapse').evaluateAll(elements => elements.every(element => element.getBoundingClientRect().height === 0))).toBe(true)
      await page.locator('main').screenshot({ path: path.join(directory, 'compact-states.png') })
      await region.press('End')
      await page.locator('main').screenshot({ path: path.join(directory, 'compact-cancelled-and-error.png') })
    })

    await test.step('用户收起不会被工具空档重开，完成后自动折叠有中间高度', async () => {
      await summary.click()
      await expect(region).toBeHidden()
      await render(events, true, 'compact', { thinking: true })
      await expect(region).toBeHidden()
      await summary.click()
      await expect(region).toBeVisible()
      await expect.poll(() => outer.locator('.reasoning-collapse').evaluate(element => getComputedStyle(element).opacity)).toBe('1')
      const before = await outer.locator('.reasoning-collapse').evaluate(element => element.getBoundingClientRect().height)
      await page.evaluate(() => {
        const heights: number[] = []
        ;(window as any).collapseHeights = heights
        function frame() { const height = document.querySelector('.reasoning-collapse')!.getBoundingClientRect().height; heights.push(height); if (height > 0) requestAnimationFrame(frame) }
        requestAnimationFrame(frame)
      })
      await render(events, false, 'compact', { status: 'completed', thinking: false })
      await expect(region).toBeHidden()
      const heights: number[] = await page.evaluate(() => (window as any).collapseHeights)
      expect(heights.some(height => height > 0 && height < before)).toBe(true)
      await page.locator('main').screenshot({ path: path.join(directory, 'completed-collapsed.png') })
      await summary.focus(); await summary.press('Enter')
      await expect(region).toBeVisible()
    })

    await test.step('新时间线跟随最新活动，用户向上阅读后不被工具更新拉回底部', async () => {
      fixtureId = 'long-message'
      const long: ChatActivityEvent[] = Array.from({ length: 32 }, (_, index) => ({ id: `long-${index}`, kind: 'reasoning', text: `核对项目 ${index + 1}：保留当前阅读位置。` }))
      long.push(tool('long-tool', 'calculate', { expression: '6*7' }, 'running'))
      await render(long, true, 'compact', { thinking: true })
      await expect.poll(() => region.evaluate(element => element.scrollHeight - element.scrollTop - element.clientHeight)).toBeLessThanOrEqual(1)
      await region.hover()
      await page.mouse.wheel(0, -await region.evaluate(element => element.scrollHeight))
      await expect.poll(() => region.evaluate(element => element.scrollTop)).toBe(0)
      long[long.length - 1] = tool('long-tool', 'calculate', { expression: '6*7' }, 'completed', { result: 42 })
      long.push({ id: 'long-continuation', kind: 'reasoning', text: '工具完成后继续核对。' })
      await render(long, true, 'compact', { thinking: false })
      expect(await region.evaluate(element => element.scrollTop)).toBe(0)
      await region.focus(); await region.press('End')
      await expect.poll(() => region.evaluate(element => element.scrollHeight - element.scrollTop - element.clientHeight)).toBeLessThanOrEqual(1)
      long.push({ id: 'long-follow', kind: 'reasoning', text: '回到底部后继续跟随。' })
      await render(long, true, 'compact', { thinking: true })
      await expect.poll(() => region.evaluate(element => element.scrollHeight - element.scrollTop - element.clientHeight)).toBeLessThanOrEqual(1)
    })

    await test.step('只含工具的记录可展开，reduced motion 无旋转且仍能读详情', async () => {
      fixtureId = 'tool-only-message'
      await page.emulateMedia({ reducedMotion: 'reduce' })
      const single = [tool('only', 'calculate', { expression: '6*7' }, 'running')]
      await render(single, true, 'compact', { thinking: false })
      await expect(page.getByRole('button', { name: /计算 6\*7/ })).toBeVisible()
      expect(await page.locator('[data-activity-id="only"] .chat-activity-status-running').evaluate(element => getComputedStyle(element).animationName)).toBe('none')
      await page.getByRole('button', { name: /计算 6\*7/ }).click()
      await expect(page.getByLabel('计算参数', { exact: true })).toBeVisible()
      expect(await region.evaluate(element => element.scrollTop)).toBe(0)
      await render([tool('only', 'calculate', { expression: '6*7' }, 'completed', { result: 42 })], true, 'compact', { thinking: false })
      await expect(page.getByRole('button', { name: /计算 6\*7/ })).toContainText('= 42')
      await page.locator('main').screenshot({ path: path.join(directory, 'reduced-motion-tool-only.png') })
    })
    await test.step('原 ReasoningContent 无插槽调用保持 active 与手动折叠约定', async () => {
      const legacy = (text: string, thinking: boolean, active?: boolean) => page.evaluate(props => (window as any).renderLegacyReasoningFixture(props), { text, thinking, active })
      const details = page.locator('.message-reasoning')
      await legacy('旧接口思考内容。', true, true)
      await expect(region).toBeVisible()
      await details.locator('summary').click()
      await legacy('工具空档的新内容。', false, true)
      await expect(region).toBeHidden()
      await details.locator('summary').click()
      await expect(region).toBeVisible()
      await legacy('完成。', false, false)
      await expect(region).toBeHidden()
      await legacy('未传 active 的旧调用。', true)
      await expect(region).toBeVisible()
      await legacy('未传 active 的旧调用。', false)
      await expect(region).toBeHidden()
    })
    expect(errors).toEqual([])
    console.log('Chat activity renderer artifacts:', directory)
  } finally {
    await browser.close()
    await server.close()
  }
})
