import { test, expect, chromium } from '@playwright/test'
import { createServer } from 'vite'
import * as fs from 'node:fs'
import * as path from 'node:path'
import type { WebSearchSource } from '../src/web-search'

test('引用按原始 Markdown 显示，工具空档保留思考展开与跟随状态', async () => {
  test.setTimeout(90_000)
  const root = path.resolve(__dirname, '..')
  const artifacts = process.env.MODLOADER_TEST_ARTIFACTS || path.join(root, '.artifacts/codex/raw-markdown-citations-20261029/citation-ui')
  fs.mkdirSync(artifacts, { recursive: true })
  const directory = fs.mkdtempSync(path.join(artifacts, 'run-'))
  fs.writeFileSync(path.join(directory, 'index.html'), `<!doctype html><meta charset="utf-8">
    <main class="chat-section" style="padding:32px;background:white;max-width:820px;margin:auto">
      <div class="message-text" id="markdown"></div><div id="reasoning" style="margin-top:28px"></div>
    </main>
    <script type="module">
      import React from 'react'; import {createRoot} from 'react-dom/client'; import {flushSync} from 'react-dom';
      import {MarkdownContent,ReasoningContent} from '/src/MessageContent.tsx'; import '/src/styles.css'; import '/src/chat-layout.css';
      const markdown=createRoot(document.getElementById('markdown')), reasoning=createRoot(document.getElementById('reasoning'));
      window.openedSources=[];
      window.modloader={openExternal:async url=>window.openedSources.push(url)};
      window.renderMarkdownFixture=props=>flushSync(()=>markdown.render(React.createElement(MarkdownContent,props)));
      window.renderReasoningFixture=props=>flushSync(()=>reasoning.render(React.createElement(ReasoningContent,props)));
    </script>`)
  const server = await createServer({
    root,
    cacheDir: path.join(directory, 'vite-cache'),
    optimizeDeps: { noDiscovery: true, include: ['react', 'react/jsx-runtime', 'react/jsx-dev-runtime', 'react-dom/client', 'react-dom', 'react-markdown', 'remark-gfm', 'remark-math', 'rehype-katex', 'highlight.js', 'dompurify', 'lucide-react'] },
    server: { port: 0, host: '127.0.0.1', watch: { ignored: ['**/.artifacts/**', '**/work/**', '**/release/**', '**/data/**'] } },
  })
  await server.listen()
  const browser = await chromium.launch({ channel: 'msedge' })
  try {
    const page = await browser.newPage({ viewport: { width: 1000, height: 820 } })
    const errors: string[] = []
    page.on('pageerror', error => errors.push(error.message))
    page.on('console', message => { if (message.type() === 'error') console.log('Citation fixture browser error:', message.text()) })
    const pending = new Set<string>()
    page.on('request', request => pending.add(request.url()))
    page.on('requestfinished', request => pending.delete(request.url()))
    page.on('requestfailed', request => pending.delete(request.url()))
    const address = server.httpServer!.address() as { port: number }
    await page.goto(`http://127.0.0.1:${address.port}/${path.relative(root, directory).replaceAll('\\', '/')}/index.html`, { waitUntil: 'commit' })
    try { await page.waitForFunction(() => typeof (window as any).renderMarkdownFixture === 'function', undefined, { timeout: 30_000 }) }
    catch (error) { console.log('Citation fixture pending:', [...pending], 'Errors:', errors); throw error }
    const render = (text: string, streaming = false) => page.evaluate(props => (window as any).renderMarkdownFixture(props), { text, streaming })
    const reason = (text: string, thinking: boolean, active?: boolean) => page.evaluate(props => (window as any).renderReasoningFixture(props), { text, thinking, active })
    const sources: WebSearchSource[] = [
      { id: 16, title: 'pathlib — 面向对象的文件系统路径', url: 'https://docs.python.org/3.14/library/pathlib.html', snippet: 'pathlib 官方文档' },
      { id: 19, title: 'Python 官方网站', url: 'https://www.python.org/', snippet: 'Python 项目' },
    ]
    const image = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRk0AAAAASUVORK5CYII='
    const text = [
      `正文中的三种引用：[来源16](${sources[0].url})、[16](${sources[0].url})、[来源 19](https://www.python.org:443)。`,
      `[普通文档](${sources[0].url}) · [来源16](${sources[1].url}) · [20](${sources[0].url}) · [来源16](https://example.com/imposter)`,
      '`[来源16](' + sources[0].url + ')`',
      '```text\n[来源16](' + sources[0].url + ')\n```',
      `![来源16](${image})`,
      `[![来源16](${image})](${sources[0].url})`,
      '[来源16](javascript:alert(1))',
      '```html\n<h3>保留预览实例</h3>\n```',
    ].join('\n\n')

    await test.step('真实表格保留模型的链接标签，匹配与未匹配的引用都是普通链接', async () => {
      const table = [
        '| 例子 | 为什么我归到“现在大致代际” |',
        '|---|---|',
        '| **GPT-5.6** | 资料里说它进入 ChatGPT 的 Plus/Pro/Max 等档位，并且有 Sol/Terra/Luna 分工，像是产品化模型。[来源15](https://www.aiposthub.com/openai-gpt-5-6-sol-terra-luna-models-analysis/) |',
        '| **Claude Opus 5 / Sonnet 5** | 多份资料把 Opus 5、Sonnet 5 当成 Anthropic 当前模型线。[来源4](https://www.elcamy.com/blog/aimodel) |',
        '| **Gemini 3.x** | 资料里把 Gemini 3.1 Pro、3.5/3.6 Flash 等作为常用主力或产品选项。[来源4](https://www.elcamy.com/blog/aimodel) |',
        '| **Grok 4.3** | 有专门对比 GPT-5.5 的发布资料，说明它是 xAI 当时推理旗舰。[来源32](https://help.apiyi.com/zh-hant/grok-4-3-vs-gpt-5-5-api-comparison-2026-zh-hant.html) |',
        '| **DeepSeek V4** | 多篇资料都把 V4 Pro / V4 Flash 当成当前 DeepSeek 主力。[来源41](https://myclaw.ai/zh-TW/blog/deepseek-v4-vs-qwen-3-6) [来源47](https://tokenhub.com/ja/compare/kimi-k3-vs-deepseek-v4-pro) |',
      ].join('\n')
      const earlierSources: WebSearchSource[] = [
        { id: 4, title: 'AIモデルの選び方【2026年8月版】GPT・Claude・Gemini徹底比較', url: 'https://www.elcamy.com/blog/aimodel', snippet: '' },
        { id: 15, title: 'OpenAI GPT-5.6 正式亮相：Sol、Terra、Luna...', url: 'https://www.aiposthub.com/openai-gpt-5-6-sol-terra-luna-models-analysis/', snippet: '' },
        { id: 32, title: 'Grok 4.3 vs GPT-5.5 API... - Apiyi.com Blog', url: 'https://help.apiyi.com/zh-hant/grok-4-3-vs-gpt-5-5-api-comparison-2026-zh-hant.html', snippet: '' },
        { id: 41, title: 'DeepSeek V4 與 Qwen 3.6：你該選哪一個模型？ | MyClaw.ai', url: 'https://myclaw.ai/zh-TW/blog/deepseek-v4-vs-qwen-3-6', snippet: '' },
        { id: 47, title: 'Kimi K3 vs DeepSeek V4 Pro モデル比較 | TokenHub', url: 'https://tokenhub.com/ja/compare/kimi-k3-vs-deepseek-v4-pro', snippet: '' },
      ]
      await render(table)
      const rendered = page.getByRole('table')
      await expect(rendered.getByRole('row')).toHaveCount(6)
      await expect(rendered.getByRole('cell')).toHaveCount(10)
      await expect(rendered.locator('sup')).toHaveCount(0)
      await expect(rendered.locator('td a')).toHaveText(['来源15', '来源4', '来源4', '来源32', '来源41', '来源47'])
      for (const link of await rendered.getByRole('link').all()) {
        const appearance = await link.evaluate(element => ({
          font: parseFloat(getComputedStyle(element).fontSize),
          bodyFont: parseFloat(getComputedStyle(element.closest('td')!).fontSize),
          verticalAlign: getComputedStyle(element).verticalAlign,
          decoration: getComputedStyle(element).textDecorationLine,
        }))
        expect(appearance.font).toBe(appearance.bodyFont)
        expect(appearance.verticalAlign).toBe('baseline')
        expect(appearance.decoration).toBe('underline')
      }
      const first = rendered.getByRole('link').first()
      await expect(first).toHaveAttribute('href', earlierSources[1].url)
      await expect(first).toHaveAccessibleName('来源15')
      await page.locator('#markdown').screenshot({ path: path.join(directory, 'real-table-raw-markdown.png') })

      await render(`${table}\n| 其他链接 | [来源15](https://example.com/fabricated) [来源999](${earlierSources[1].url}) [普通文档](${earlierSources[1].url}) |`)
      await expect(page.locator('#markdown sup')).toHaveCount(0)
      const unmatched = page.getByRole('row').last()
      await expect(unmatched.getByRole('link')).toHaveText(['来源15', '来源999', '普通文档'])
      await expect(unmatched.getByRole('link').first()).toHaveAttribute('href', 'https://example.com/fabricated')
    })

    await test.step('真实回复保留段尾硬换行与多个引用原来的独立行', async () => {
      const realSources: WebSearchSource[] = [
        { id: 27, title: 'Design Guidelines | OpenAI', url: 'https://openai.com/brand', snippet: '' },
        { id: 110, title: "OpenAI Dots: Colorful AI Agents Rival Meta's Mascot", url: 'https://fact-hub.com/breaking-news/openai-dots-ai-agent-mascots-meta-rivalry/', snippet: '' },
        { id: 111, title: 'AI agent mascots and trust: Fairplay calls for a probe | NeoTeo', url: 'https://www.neoteo.com/en/fairplay-asks-regulators-to-investigate-whether-meta-and-openai-use-ai-mascots-to-attract-children', snippet: '' },
        { id: 144, title: 'DeepSeek里长出了个“鲸鱼娘” | 人人都是产品经理', url: 'https://www.woshipm.com/ai/6451688.html', snippet: '' },
      ]
      // Exact paragraphs from the reply in the reported screenshot, including two-space hard breaks.
      const realReply = [
        'OpenAI 主视觉长期更偏抽象 logo/品牌符号，而不是像 DeepSeek “大肥鱼/鲸鱼娘”那样有一个强拟人形象。它品牌指南里核心也是 logo 符号系统。  \n[来源27](https://openai.com/brand)',
        'OpenAI 在 2026 DevDay 上推出的 always-on agent **Dots**，就带了一组官方产品向的“友好小角色”。它们颜色各异，用户还可以给它们起名字、做个性化。启动材料里出现的例子包括 **Iggy、Felipe、Todd、Alfred、Jojo**，造型有戴眼镜、戴耳机、戴贝雷帽、打领结等。  \n[来源110](https://fact-hub.com/breaking-news/openai-dots-ai-agent-mascots-meta-rivalry/)  \n[来源111](https://www.neoteo.com/en/fairplay-asks-regulators-to-investigate-whether-meta-and-openai-use-ai-mascots-to-attract-children)',
        'OpenAI 的 Codex 还加了可选的动画宠物功能，可以用 `/pet` 召唤。它会以悬浮窗口形式出现，不只是卖萌，还能根据任务状态给可视化反馈，比如运行中、完成、需要注意等。  \n[来源144](https://www.woshipm.com/ai/6451688.html)',
      ].join('\n\n')
      await render(realReply)
      await page.locator('#markdown').screenshot({ path: path.join(directory, 'real-citation-paragraphs.png') })
      const paragraphs = page.locator('#markdown .message-markdown > p')
      await expect(paragraphs).toHaveCount(3)
      await expect(paragraphs.locator('br')).toHaveCount(4)
      await expect(paragraphs.getByRole('link')).toHaveText(['来源27', '来源110', '来源111', '来源144'])
      for (let index = 0; index < 3; index++) {
        const geometry = await paragraphs.nth(index).evaluate(paragraph => {
          const firstBreak = paragraph.querySelector('br')!
          const range = document.createRange()
          range.selectNodeContents(firstBreak.previousSibling!)
          const prose = [...range.getClientRects()].at(-1)!
          const reference = paragraph.querySelector('a')!.getBoundingClientRect()
          return { proseBottom: prose.bottom, citationTop: reference.top }
        })
        expect(geometry.citationTop).toBeGreaterThanOrEqual(geometry.proseBottom)
      }
      const positions = await paragraphs.nth(1).getByRole('link').evaluateAll(elements => elements.map(element => ({ top: element.getBoundingClientRect().top, bottom: element.getBoundingClientRect().bottom })))
      expect(positions[1].top).toBeGreaterThanOrEqual(positions[0].bottom)

      await render(`软换行引用。\n[来源27](${realSources[0].url})\n[来源110](${realSources[1].url})`)
      await expect(page.locator('#markdown p')).toHaveText('软换行引用。 来源27 来源110')
      await expect(page.locator('#markdown br')).toHaveCount(0)
      const controls = [
        '普通段落一。\n\n普通段落二。',
        '普通硬换行一。  \n普通硬换行二。',
        `未知编号。  \n[来源999](${realSources[0].url})`,
        `来源 URL 不匹配。  \n[来源27](${realSources[1].url})`,
        `普通链接。  \n[品牌文档](${realSources[0].url})`,
        `硬换行后还有正文。  \n[来源27](${realSources[0].url}) 后续正文。`,
        `- 列表第一行。  \n  保留第二行。  \n  [来源27](${realSources[0].url})\n- 列表第二项。`,
        '`[来源27](' + realSources[0].url + ')`',
        '```text\n正文。  \n[来源27](' + realSources[0].url + ')\n```',
      ].join('\n\n')
      await render(controls)
      await expect(page.locator('#markdown p').filter({ hasText: /^普通段落[一二]。$/ })).toHaveCount(2)
      await expect(page.locator('#markdown br')).toHaveCount(7)
      await expect(page.locator('#markdown li')).toHaveCount(2)
      await expect(page.locator('#markdown li').first().locator('br')).toHaveCount(2)
      await expect(page.locator('#markdown li').first().getByRole('link')).toHaveText('来源27')
      await expect(page.getByRole('link', { name: '品牌文档', exact: true })).toHaveAttribute('href', realSources[0].url)
      await expect(page.locator('#markdown p > code')).toHaveText(`[来源27](${realSources[0].url})`)
      await expect(page.locator('#markdown pre code')).toHaveText(`正文。  \n[来源27](${realSources[0].url})\n`)
    })

    await test.step('保留原始标签、外链打开、URL 过滤及代码图片与 HTML 预览', async () => {
      await render(text, true)
      await expect(page.locator('#markdown sup')).toHaveCount(0)
      const links = page.locator('#markdown p').first().getByRole('link')
      await expect(links).toHaveText(['来源16', '16', '来源 19'])
      const first = links.first()
      await expect(first).toHaveAccessibleName('来源16')
      await expect(first).toHaveAttribute('href', sources[0].url)
      expect(await first.evaluate(element => getComputedStyle(element).textDecorationLine)).toBe('underline')
      await first.focus()
      await expect(first).toBeFocused()
      await first.press('Enter')
      expect(await page.evaluate(() => (window as any).openedSources)).toEqual([sources[0].url])
      await expect(page.getByRole('link', { name: '普通文档', exact: true })).toHaveAttribute('href', sources[0].url)
      await expect(page.locator('#markdown a').filter({ hasText: /^来源16$/ })).toHaveCount(3)
      await expect(page.locator('#markdown p > code')).toHaveText(`[来源16](${sources[0].url})`)
      await expect(page.locator('#markdown .message-code-block[data-language="text"] code')).toHaveText(`[来源16](${sources[0].url})\n`)
      await expect(page.locator('#markdown a[href^="javascript:"]')).toHaveCount(0)
      await expect(page.locator('#markdown .message-image img')).toHaveCount(2)
      await expect(page.locator('#markdown .message-code-block[data-language="html"] iframe')).toBeVisible()
      await page.evaluate(() => {
        ;(window as any).savedImage = document.querySelector('#markdown .message-image img')
        ;(window as any).savedPreview = document.querySelector('#markdown iframe')
      })
      await render(`${text}\n\n新的流式片段。`, true)
      expect(await page.evaluate(() => ({
        image: (window as any).savedImage === document.querySelector('#markdown .message-image img'),
        preview: (window as any).savedPreview === document.querySelector('#markdown iframe'),
      }))).toEqual({ image: true, preview: true })
      await page.locator('main').screenshot({ path: path.join(directory, 'raw-markdown-links-and-previews.png') })
      await render(`[来源16](${sources[0].url})`)
      await expect(page.locator('#markdown sup')).toHaveCount(0)
      await expect(page.getByRole('link', { name: '来源16', exact: true })).toHaveAttribute('href', sources[0].url)
    })

    await test.step('整轮 active 控制自动收起，thinking 的工具间隙不覆盖用户手动展开与收起', async () => {
      let text = '第一轮：先查询官方文档。'
      await reason(text, true, true)
      const details = page.locator('.message-reasoning')
      const region = page.getByRole('region', { name: '思考内容', includeHidden: true })
      const summary = details.locator('summary')
      await expect(region).toBeVisible()
      await reason(text, false, true)
      await expect(region).toBeVisible()
      await expect(summary).toHaveText('思考过程')
      await summary.click()
      await expect(region).toBeHidden()
      text += '\n\n第二轮：继续读取网页。'
      await reason(text, true, true)
      await expect(region).toBeHidden()
      await summary.click()
      await expect(region).toBeVisible()
      await reason(text, false, true)
      await expect(region).toBeVisible()
      await expect(region).toContainText('第一轮')
      await expect(region).toContainText('第二轮')

      text += Array.from({ length: 45 }, (_, index) => `\n\n分析 ${index + 1}：核对来源与上下文。`).join('')
      await reason(text, true, true)
      await expect.poll(() => region.evaluate(element => element.scrollHeight - element.scrollTop - element.clientHeight)).toBeLessThanOrEqual(1)
      await region.hover()
      await page.mouse.wheel(0, -await region.evaluate(element => element.scrollHeight))
      await expect.poll(() => region.evaluate(element => element.scrollTop)).toBe(0)
      text += '\n\n工具空档保留阅读位置。'
      await reason(text, false, true)
      expect(await region.evaluate(element => element.scrollTop)).toBe(0)
      await region.press('End')
      await expect.poll(() => region.evaluate(element => element.scrollHeight - element.scrollTop - element.clientHeight)).toBeLessThanOrEqual(1)
      text += '\n\n工具空档仍跟随新增思考。'
      await reason(text, false, true)
      await expect.poll(() => region.evaluate(element => element.scrollHeight - element.scrollTop - element.clientHeight)).toBeLessThanOrEqual(1)
      await page.locator('#reasoning').screenshot({ path: path.join(directory, 'reasoning-tool-gap.png') })
      await reason(text, false, false)
      await expect(region).toBeHidden()
      await summary.click()
      await expect(region).toBeVisible()
      await reason(text, false, false)
      await expect(region).toBeVisible()
      await reason('兼容未传 active 的原有调用。', true)
      await expect(region).toBeVisible()
      await reason('兼容未传 active 的原有调用。', false)
      await expect(region).toBeHidden()
    })
    expect(errors).toEqual([])
    console.log('Citation UI artifacts:', directory)
  } finally {
    await browser.close()
    await server.close()
  }
})
