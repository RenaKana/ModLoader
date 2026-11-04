import { test, expect, chromium } from '@playwright/test'
import { createServer } from 'vite'
import * as fs from 'node:fs'
import * as path from 'node:path'

test('代码与纯文本块复制原始内容，预览源码和流式片段支持复制及失败重试', async () => {
  test.setTimeout(90_000)
  const root = path.resolve(__dirname, '..')
  const artifacts = process.env.MODLOADER_TEST_ARTIFACTS || path.join(root, '.artifacts/codex/chat-copy-no-citations-20261029/copy-ui')
  fs.mkdirSync(artifacts, { recursive: true })
  const directory = fs.mkdtempSync(path.join(artifacts, 'run-'))
  fs.writeFileSync(path.join(directory, 'index.html'), `<!doctype html><meta charset="utf-8">
    <main class="chat-section" style="padding:24px;background:white;max-width:820px;margin:auto"><div class="message-text" id="markdown"></div></main>
    <script type="module">
      import React from 'react'; import {createRoot} from 'react-dom/client'; import {flushSync} from 'react-dom';
      import {MarkdownContent} from '/src/MessageContent.tsx'; import '/src/styles.css'; import '/src/chat-layout.css';
      const root=createRoot(document.getElementById('markdown'));
      window.copiedBlocks=[]; window.failCopy=false;
      // Keep the user's system clipboard intact; exercise the production copy helper's desktop bridge.
      window.modloader={copy:async text=>{if(window.failCopy)throw new Error('剪贴板暂不可用');window.copiedBlocks.push(text)}};
      window.renderCopyFixture=props=>flushSync(()=>root.render(React.createElement(MarkdownContent,props)));
    </script>`)
  const server = await createServer({
    root,
    cacheDir: path.join(directory, 'vite-cache'),
    optimizeDeps: { noDiscovery: true, include: ['react', 'react/jsx-runtime', 'react/jsx-dev-runtime', 'react-dom/client', 'react-dom', 'react-markdown', 'remark-gfm', 'remark-math', 'rehype-katex', 'highlight.js', 'dompurify', 'lucide-react', 'mermaid', '@viz-js/viz'] },
    server: { port: 0, host: '127.0.0.1', watch: { ignored: ['**/.artifacts/**', '**/work/**', '**/release/**', '**/data/**'] } },
  })
  await server.listen()
  const browser = await chromium.launch({ channel: 'msedge' })
  try {
    const page = await browser.newPage({ viewport: { width: 1000, height: 900 } })
    const errors: string[] = []
    page.on('pageerror', error => errors.push(error.message))
    const address = server.httpServer!.address() as { port: number }
    await page.goto(`http://127.0.0.1:${address.port}/${path.relative(root, directory).replaceAll('\\', '/')}/index.html`)
    await page.waitForFunction(() => typeof (window as any).renderCopyFixture === 'function')
    const render = (text: string, streaming = false) => page.evaluate(props => (window as any).renderCopyFixture(props), { text, streaming })
    const copied = () => page.evaluate(() => (window as any).copiedBlocks.at(-1) as string)
    const fence = (language: string, source: string) => `\`\`\`${language}\n${source}\`\`\``

    await test.step('有语言、text、无语言和缩进代码块逐块复制，中文、空行与缩进完整保留', async () => {
      const samples = [
        'def 显示():\n\tprint("你好 <世界> & Markdown")\n\n    return "完成"\n',
        '  中文第一行  \n\n\t第二行：`原样` <标签>\n',
        '没有语言标签。\n    保留四个空格。\n',
        '缩进式代码\n    中文保留\n',
      ]
      await render([
        '行内代码 `保持行内`。',
        fence('python', samples[0]), fence('text', samples[1]), fence('', samples[2]),
        '    缩进式代码\n        中文保留\n',
      ].join('\n\n'))
      const blocks = page.locator('.message-code-block')
      await expect(blocks).toHaveCount(4)
      await expect(page.getByRole('button', { name: '复制内容', exact: true })).toHaveCount(4)
      await expect(blocks.first().locator('.hljs-keyword').first()).toBeVisible()
      for (let index = 0; index < samples.length; index++) {
        const block = blocks.nth(index)
        expect(await block.locator('pre code').textContent()).toBe(samples[index])
        const button = block.getByRole('button', { name: '复制内容', exact: true })
        if (index === 0) { await button.focus(); await button.press('Enter') }
        else await button.click()
        await expect.poll(copied).toBe(samples[index])
        await expect(button.getByRole('status')).toHaveText('已复制')
      }
      await page.locator('main').screenshot({ path: path.join(directory, 'code-and-text-copy.png') })
    })

    await test.step('流式片段复制当前内容，新增片段后清除旧成功状态，失败后可重试', async () => {
      const partial = '第一行：中文\n  保留缩进\n'
      await render(`\`\`\`text\n${partial}`, true)
      const block = page.locator('.message-code-block')
      const button = block.getByRole('button', { name: '复制内容', exact: true })
      await button.click()
      await expect.poll(copied).toBe(partial)
      await expect(button.getByRole('status')).toHaveText('已复制')
      const complete = `${partial}\n新增片段：完成\n`
      await render(fence('text', complete), true)
      await expect(button).toHaveText('复制')
      await page.evaluate(() => { (window as any).failCopy = true })
      await button.click()
      await expect(button.getByRole('alert')).toHaveText('复制失败')
      await expect(button).toHaveAttribute('title', '复制失败：剪贴板暂不可用')
      await expect(button).toBeEnabled()
      expect(await copied()).toBe(partial)
      await page.locator('main').screenshot({ path: path.join(directory, 'copy-failed.png') })
      await page.evaluate(() => { (window as any).failCopy = false })
      await button.focus(); await button.press('Space')
      await expect.poll(copied).toBe(complete)
      await expect(button.getByRole('status')).toHaveText('已复制')
    })

    await test.step('HTML、SVG、Mermaid、DOT 与 Graphviz 预览复用标题栏复制，展开 HTML 也可复制源码', async () => {
      const previews = [
        ['html', '<section>\n  <h2>源码 <em>保留中文</em></h2>\n</section>\n'],
        ['svg', '<svg xmlns="http://www.w3.org/2000/svg" width="240" height="80"><text x="10" y="40">中文图形</text></svg>\n'],
        ['mermaid', 'flowchart LR\n  A[开始] --> B[完成]\n'],
        ['dot', 'digraph { A -> B [label="保留源码"]; }\n'],
        ['graphviz', 'graph { A -- B; B -- C; }\n'],
      ]
      for (const [language, source] of previews) {
        await render(fence(language, source))
        const block = page.locator('.message-code-block')
        const button = block.getByRole('button', { name: '复制源码', exact: true })
        await expect(button).toHaveCount(1)
        if (language === 'html' || language === 'svg') await expect(block.locator('iframe')).toBeVisible()
        else await expect(block.locator('.message-diagram-scroll img')).toBeVisible()
        await button.click()
        await expect.poll(copied).toBe(source)
        await expect(button.getByRole('status')).toHaveText('已复制')
        await block.locator('.message-code-source summary').click()
        expect(await block.locator('pre code').textContent()).toBe(source)
        if (language === 'html') {
          await block.getByRole('button', { name: '展开预览', exact: true }).click()
          const viewer = page.getByRole('dialog', { name: 'HTML 预览', exact: true })
          await viewer.getByRole('button', { name: '源码', exact: true }).click()
          await viewer.getByRole('button', { name: '复制源码', exact: true }).click()
          await expect.poll(copied).toBe(source)
          await expect(viewer.getByRole('button', { name: '复制源码', exact: true }).getByRole('status')).toHaveText('已复制')
          await page.screenshot({ path: path.join(directory, 'expanded-html-source-copy.png') })
          await viewer.getByRole('button', { name: '关闭预览', exact: true }).click()
        }
        if (language === 'mermaid') await page.locator('main').screenshot({ path: path.join(directory, 'diagram-source-copy.png') })
      }
    })
    expect(errors).toEqual([])
    console.log('Message copy UI artifacts:', directory)
  } finally {
    await browser.close()
    await server.close()
  }
})
