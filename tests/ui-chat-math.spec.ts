import { test, expect, chromium, type Locator } from '@playwright/test'
import { createServer } from 'vite'
import * as fs from 'node:fs'
import * as path from 'node:path'

// Extracted unchanged from the saved reply shown in the R_B overlap report.
const realArray = fs.readFileSync(path.join(__dirname, 'fixtures/chat-math-array.md'), 'utf8')

async function arrayGeometry(array: Locator) {
  return array.evaluate(element => {
    const context = document.createElement('canvas').getContext('2d')!
    const bounds = (row: Element) => {
      const walker = document.createTreeWalker(row, NodeFilter.SHOW_TEXT)
      const boxes: DOMRect[] = []
      while (walker.nextNode()) {
        if (!walker.currentNode.textContent?.replace(/[\s\u200b]/g, '')) continue
        const range = document.createRange()
        range.selectNodeContents(walker.currentNode)
        const style = getComputedStyle(walker.currentNode.parentElement!)
        context.font = `${style.fontStyle} ${style.fontWeight} ${style.fontSize} ${style.fontFamily}`
        const metrics = context.measureText(walker.currentNode.textContent!)
        for (const box of range.getClientRects()) {
          const baseline = box.top + metrics.fontBoundingBoxAscent
          boxes.push(new DOMRect(box.left, baseline - metrics.actualBoundingBoxAscent, box.width,
            metrics.actualBoundingBoxAscent + metrics.actualBoundingBoxDescent))
        }
      }
      return { top: Math.min(...boxes.map(box => box.top)), bottom: Math.max(...boxes.map(box => box.bottom)) }
    }
    const columns = [...element.querySelectorAll(':scope > [class^="col-align"]')].map(column =>
      [...column.querySelectorAll(':scope > .vlist-t > .vlist-r:first-child > .vlist > span')].map(bounds))
    const gaps = columns.flatMap(rows => rows.slice(1).map((row, index) => row.top - rows[index].bottom))
    const display = element.closest('.katex-display')!.getBoundingClientRect()
    return {
      rowCounts: columns.map(rows => rows.length), minimumGap: Math.min(...gaps),
      topClearance: Math.min(...columns.flat().map(row => row.top)) - display.top,
      bottomClearance: display.bottom - Math.max(...columns.flat().map(row => row.bottom)),
    }
  })
}

test('真实 dfrac array 不重叠，局部行距不改普通公式、显式排版和流式源码', async () => {
  test.setTimeout(120_000)
  const root = path.resolve(__dirname, '..')
  const directory = fs.mkdtempSync(path.join(root, 'work', 'ui-chat-math-'))
  fs.writeFileSync(path.join(directory, 'index.html'), `<!doctype html><meta charset="utf-8">
    <main class="chat-section" style="padding:24px;background:white"><div class="message-text" id="root"></div></main>
    <script type="module">
      import React from 'react'; import {createRoot} from 'react-dom/client'; import {flushSync} from 'react-dom';
      import {MarkdownContent} from '/src/MessageContent.tsx'; import '/src/styles.css'; import '/src/chat-layout.css';
      const root=createRoot(document.getElementById('root'));
      window.renderMathFixture=(text,streaming=false)=>flushSync(()=>root.render(React.createElement(MarkdownContent,{text,streaming})));
    </script>`)
  const server = await createServer({ root, server: { port: 0, host: '127.0.0.1' } })
  await server.listen()
  const browser = await chromium.launch({ channel: 'msedge' })
  try {
    const page = await browser.newPage({ viewport: { width: 1100, height: 1000 } })
    const errors: string[] = []
    page.on('pageerror', error => errors.push(error.message))
    const address = server.httpServer!.address() as { port: number }
    await page.goto(`http://127.0.0.1:${address.port}/${path.relative(root, directory).replaceAll('\\', '/')}/index.html`)
    await page.waitForFunction(() => typeof (window as any).renderMathFixture === 'function')
    const render = async (text: string, streaming = false) => {
      await page.evaluate(({ text, streaming }) => (window as any).renderMathFixture(text, streaming), { text, streaming })
      await page.evaluate(() => document.fonts.ready)
    }

    await render(realArray)
    await expect(page.locator('.katex-error')).toHaveCount(0)
    const geometry = await arrayGeometry(page.locator('.mtable'))
    const output = path.join(directory, 'real-array.png')
    await page.locator('main').screenshot({ path: output })
    console.log('Real array geometry:', geometry, 'Screenshot:', output)
    expect(geometry.rowCounts).toEqual([13, 13])
    expect(geometry.minimumGap).toBeGreaterThan(1)
    expect(geometry.topClearance).toBeGreaterThanOrEqual(0)
    expect(geometry.bottomClearance).toBeGreaterThanOrEqual(0)
    await expect(page.locator('.vertical-separator')).toHaveCount(1)

    const plain = String.raw`\begin{array}{c}1\\2\end{array}`
    const compact = String.raw`\begin{array}{c}\dfrac12\\\dfrac34\end{array}`
    await render(`\\[${compact} + ${plain}\\]`)
    const afterArray = await page.locator('.mtable').last().boundingBox()
    await render(`\\[${plain}\\]`)
    expect((await page.locator('.mtable').boundingBox())!.height).toBeCloseTo(afterArray!.height, 1)
    const nested = String.raw`\begin{array}{c|c}\begin{array}{c}\dfrac12\\\dfrac34\end{array}&\dfrac56\\1&2\end{array}`
    await render(`\\[${nested}\\]`)
    await expect(page.locator('.katex-error')).toHaveCount(0)
    await expect(page.locator('.mtable')).toHaveCount(2)

    for (const authored of [String.raw`\def\arraystretch{1.4}${compact}`, compact.replace('\\\\', '\\\\[3pt]'), plain, String.raw`\frac12 + \dfrac34`]) {
      await render(`\\[${authored}\\]`)
      await expect(page.locator('annotation')).toHaveText(authored)
    }
    await render(`字面量：\`${compact}\`\n\n\`\`\`latex\n${compact}`, true)
    await expect(page.locator('.katex')).toHaveCount(0)
    await expect(page.locator('pre code')).toHaveText(`${compact}\n`)
    await render(`\`\`\`latex\n${compact}\n\`\`\``)
    await expect(page.locator('.mtable')).toHaveCount(1)
    expect((await arrayGeometry(page.locator('.mtable'))).minimumGap).toBeGreaterThan(1)
    expect(errors).toEqual([])
  } finally {
    await browser.close()
    await server.close()
  }
})
