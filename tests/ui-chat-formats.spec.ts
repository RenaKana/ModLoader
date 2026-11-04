import { pinNavigation } from './navigation'
import { disableWebSearch } from './chat-controls'
import { test, expect, _electron as electron, type ElectronApplication, type Locator, type Page } from '@playwright/test'
import { createServer, type ServerResponse } from 'node:http'
import * as path from 'node:path'
import * as fs from 'node:fs'
import { fileURLToPath } from 'node:url'

test('聊天格式支持公式、高亮、静态图表与安全预览，并保留流式源码', async () => {
  test.setTimeout(120_000)
  const root = path.resolve(__dirname, '..')
  const directory = fs.mkdtempSync(path.join(process.env.MODLOADER_TEST_ARTIFACT_DIR || path.join(root, 'work'), 'ui-chat-formats-'))
  const executablePath = process.env.MODLOADER_TEST_EXECUTABLE
  let response: ServerResponse | undefined
  let application: ElectronApplication | undefined
  let activePage: Page | undefined
  const errors: string[] = []
  const securityRequests: string[] = []
  const fixtureSvg = '<svg xmlns="http://www.w3.org/2000/svg" width="160" height="90" viewBox="0 0 160 90"><rect width="160" height="90" rx="12" fill="#ddf2e5"/><circle cx="44" cy="45" r="23" fill="#4f8b64"/><path d="M85 30h48M85 45h38M85 60h43" stroke="#28563b" stroke-width="5" stroke-linecap="round"/></svg>'
  const server = createServer(async (request, result) => {
    result.setHeader('Access-Control-Allow-Origin', '*')
    result.setHeader('Access-Control-Allow-Headers', '*')
    if (request.method === 'OPTIONS') { result.end(); return }
    const pathname = new URL(request.url || '/', 'http://127.0.0.1').pathname
    if (pathname.startsWith('/security-')) securityRequests.push(pathname)
    if (pathname === '/fixture.svg') {
      result.setHeader('Content-Type', 'image/svg+xml')
      result.end(fixtureSvg)
      return
    }
    if (pathname !== '/api/chat') { result.writeHead(404); result.end(); return }
    for await (const _chunk of request) {}
    result.setHeader('Content-Type', 'text/event-stream')
    result.flushHeaders()
    response = result
  })
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const address = server.address() as { port: number }
  const fixtureUrl = `http://127.0.0.1:${address.port}`

  try {
    application = await electron.launch({ ...(executablePath ? { executablePath, args: [] } : { args: [root] }), env: { ...process.env, MODLOADER_DATA_DIR: directory }, timeout: 60_000 })
    await application.evaluate(({ app }, directory) => app.setPath('temp', directory), directory)
    const runtime = await application.evaluate(({ app }) => ({ packaged: app.isPackaged, appPath: app.getAppPath(), executable: app.getPath('exe'), electron: process.versions.electron }))
    console.log('Chat formats runtime:', runtime)
    if (executablePath) expect(runtime.packaged).toBe(true)
    const page = await application.firstWindow()
    activePage = page
    const dialogs: string[] = []
    page.on('pageerror', error => errors.push(error.message))
    page.on('dialog', async dialog => { dialogs.push(dialog.message()); await dialog.dismiss() })
    const assistant = page.locator('.chat-message.assistant').last()
    const input = page.getByRole('textbox', { name: '对话消息' })
    const send = async () => {
      await disableWebSearch(page)
      response = undefined
      await input.fill('验证这条格式回复')
      await page.getByRole('button', { name: '发送消息', exact: true }).click()
      await expect.poll(() => Boolean(response)).toBe(true)
    }
    const event = (value: unknown) => response!.write(`data:${JSON.stringify(value)}\r\n\r\n`)
    const delta = (value: unknown) => event({ choices: [{ delta: value }] })
    const finish = async () => {
      event({ choices: [{ delta: {}, finish_reason: 'stop' }] })
      response!.end('data: [DONE]\n\n')
      await expect(page.getByRole('button', { name: '停止生成', exact: true })).toHaveCount(0)
    }
    const reply = async (content: string, reasoning?: string) => {
      await send()
      delta({ content, ...(reasoning ? { reasoning_content: reasoning } : {}) })
      await finish()
    }
    const showSource = async (block: Locator) => {
      const source = block.locator('.message-code-source')
      if (!await source.evaluate(element => (element as HTMLDetailsElement).open)) await source.locator('summary').click()
      return source.locator('pre code')
    }
    const loadedImage = async (image: Locator) => {
      await expect(image).toBeVisible()
      await expect.poll(() => image.evaluate(element => (element as HTMLImageElement).complete && (element as HTMLImageElement).naturalWidth > 0)).toBe(true)
    }
    const screenshot = async (name: string, region?: Locator) => {
      await (region || assistant).scrollIntoViewIfNeeded()
      const output = path.join(directory, `${name}.png`)
      if (region) await region.screenshot({ path: output })
      else await page.screenshot({ path: output })
      console.log(`Chat formats screenshot: ${output}`)
    }

    await pinNavigation(page)
    await expect(page.getByText('服务在线', { exact: true })).toBeVisible({ timeout: 60_000 })
    await page.route('**/api/state', async route => {
      const original = await route.fetch()
      const state = await original.json()
      await route.fulfill({ response: original, json: { ...state, instances: [{ id: 'format-fixture', profile_id: 'format-profile', model_id: 'format-model', alias: '格式渲染测试', backend: 'llama_cpp', state: 'ready', capabilities: { input: ['text'] }, active_requests: 0 }] } })
    })
    await page.route('**/api/chat', route => route.continue({ url: `${fixtureUrl}/api/chat` }))
    await page.route('**/api/instances/format-fixture/telemetry', route => route.fulfill({ json: { updated_at: Date.now() / 1000, groups: [], unavailable: [], live_generation: null } }))
    await page.getByRole('button', { name: '刷新状态', exact: true }).click()
    await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '测试对话', exact: true }).click()
    await expect(page.getByText('开始对话', { exact: true })).toBeVisible()

    const setextFormulas = [String.raw`\[
2\left(\frac78-p_3\right)+3p_3
=
\frac74-p_2? 
\]`, String.raw`\[
2\left(\frac78-p_3\right)+3p_3
=
\frac{14}{8}-2p_3+3p_3
=
\frac74+p_3.
\]`]
    await test.step('真实失败公式内的独占等号不被解析为标题，代码与普通标题不受影响', async () => {
      await reply(['这里注意符号：更直接地算：', setextFormulas[0], '重新写清楚：', setextFormulas[1], '正常标题\n---', `\`\`\`text\n${setextFormulas[1]}\n\`\`\``].join('\n\n'))
      const rendered = assistant.locator('.message-text')
      await expect(rendered.locator('.katex-display')).toHaveCount(2)
      await expect(rendered.locator('.katex-error, h1')).toHaveCount(0)
      await expect(rendered.locator('h2')).toHaveText('正常标题')
      expect((await rendered.locator('annotation[encoding="application/x-tex"]').allTextContents()).map(value => value.trim())).toEqual(setextFormulas.map(value => value.slice(2, -2).trim()))
      await expect(rendered.locator('.message-code-block pre code')).toHaveText(`${setextFormulas[1]}\n`)
      await expect(rendered.locator('pre .katex')).toHaveCount(0)
      await screenshot('block-math-equals', rendered)
    })

    await test.step('真实中文公式、两种数学分隔符、高亮、金额及代码字面量', async () => {
      const failedFormula = String.raw`\[ (\text{圆形苹果} \ge 1 \text{ 且 五角星桃子} \ge 1) \]`
      const multilineFormula = String.raw`\[
\begin{aligned}
x + y &= 3 \\
x - y &= 1
\end{aligned}
\]`
      const python = String.raw`def total(n):
    return f"\[ {n} \]"  # $literal$`
      await reply([
        '## 格式校验',
        String.raw`内联 $a^2 + b^2 = c^2$，另一种内联 \( x + y \)。`,
        String.raw`$$\frac{1}{2}$$`,
        failedFormula,
        multilineFormula,
        '价格是 $5 和 $10。',
        '字面代码：`' + String.raw`\( x + y \) $literal$` + '`。',
        `\`\`\`python\n${python}\n\`\`\``,
      ].join('\n\n'))
      const rendered = assistant.locator('.message-text')
      await expect(rendered.locator('.katex')).toHaveCount(5)
      await expect(rendered.locator('.katex-display')).toHaveCount(3)
      await expect(rendered.locator('annotation[encoding="application/x-tex"]').filter({ hasText: String.raw`\text{圆形苹果}` })).toHaveCount(1)
      await expect(rendered.getByText('价格是 $5 和 $10。', { exact: true })).toBeVisible()
      await expect(rendered.locator('p > code')).toHaveText(String.raw`\( x + y \) $literal$`)
      await expect(rendered.locator('code .katex')).toHaveCount(0)
      const code = rendered.locator('.message-code-block[data-language="python"] pre code')
      await expect(code).toHaveText(`${python}\n`)
      await expect(code.locator('.hljs-keyword').first()).toBeVisible()
      const mathFontLoaded = await page.evaluate(async () => {
        await document.fonts.ready
        return [...document.fonts].some(font => /KaTeX_Main/.test(font.family) && font.status === 'loaded')
      })
      expect(mathFontLoaded).toBe(true)
      await screenshot('math-and-highlight')
    })

    await test.step('Mermaid、DOT 和 Graphviz 以惰性图片预览并保留源码', async () => {
      const mermaid = 'flowchart LR\n  A[检查] --> B{通过?}\n  B -->|是| C[完成]'
      const dot = 'digraph { rankdir=LR; A [label="Start"]; A -> B [label="next"]; }'
      const graphviz = 'graph { A -- B; B -- C; }'
      await reply(`\`\`\`mermaid\n${mermaid}\n\`\`\`\n\n\`\`\`dot\n${dot}\n\`\`\`\n\n\`\`\`graphviz\n${graphviz}\n\`\`\``)
      for (const [language, source] of [['mermaid', mermaid], ['dot', dot], ['graphviz', graphviz]]) {
        const block = assistant.locator(`.message-code-block[data-language="${language}"]`)
        const image = block.locator('img')
        await loadedImage(image)
        await expect(image).toHaveAttribute('src', /^(?:data:image\/svg\+xml|blob:)/)
        await expect(block.locator('iframe, script')).toHaveCount(0)
        await expect(await showSource(block)).toHaveText(`${source}\n`)
        await block.locator('.message-code-source > summary').click()
        if (language === 'mermaid') {
          const visibleLabels = await image.evaluate(async element => {
            const source = await fetch((element as HTMLImageElement).src).then(result => result.text())
            const svg = new DOMParser().parseFromString(source, 'image/svg+xml')
            return [...svg.querySelectorAll('text')].map(text => text.textContent?.trim()).join(' ')
          })
          for (const label of ['检查', '通过?', '完成', '是']) expect(visibleLabels).toContain(label)
          await screenshot('mermaid-full-preview', block)
        }
      }
      await screenshot('static-diagrams')
    })

    await test.step('HTML、SVG 和独立 HTML 静态预览隔离危险内容，HTTP 图片与失效回退', async () => {
      await page.evaluate(() => { (window as any).formatPreviewAttack = 'clean' })
      const hostileHtml = `<!doctype html><html><head><meta http-equiv="refresh" content="0;url=${fixtureUrl}/security-meta"></head><body><h2>静态安全预览</h2><script>top.formatPreviewAttack='script';fetch('${fixtureUrl}/security-script')</script><img src="${fixtureUrl}/missing-preview.png" onerror="top.formatPreviewAttack='onerror';fetch('${fixtureUrl}/security-onerror')"><form action="${fixtureUrl}/security-form" method="post"><input name="value" value="fixture"><button>提交</button></form><a href="javascript:top.formatPreviewAttack='link'">危险链接</a><a href="${fixtureUrl}/security-link" target="_top">外部链接</a><iframe src="${fixtureUrl}/security-frame"></iframe></body></html>`
      const hostileSvg = `<svg xmlns="http://www.w3.org/2000/svg" width="220" height="80" onload="top.formatPreviewAttack='svg'"><script>fetch('${fixtureUrl}/security-svg')</script><rect width="220" height="80" fill="#dcecf6"/><text x="16" y="45" fill="#244c6b">静态 SVG</text><a href="javascript:alert(1)"><text x="16" y="65">危险链接</text></a></svg>`
      await reply(`\`\`\`html\n${hostileHtml}\n\`\`\`\n\n\`\`\`svg\n${hostileSvg}\n\`\`\`\n\n![在线图片](${fixtureUrl}/fixture.svg)\n\n![损坏图片](${fixtureUrl}/missing.png)`)
      for (const language of ['html', 'svg']) {
        const block = assistant.locator(`.message-code-block[data-language="${language}"]`)
        const iframe = block.locator('iframe')
        await expect(iframe).toBeVisible()
        await expect(iframe).toHaveAttribute('sandbox', '')
        const preview = block.frameLocator('iframe')
        await expect(preview.locator('body')).toBeVisible()
        await expect(preview.locator('script, iframe, form, meta[http-equiv="refresh" i], a[href], [onload], [onerror]')).toHaveCount(0)
        const source = await showSource(block)
        await expect(source).toContainText(language === 'html' ? 'security-script' : 'security-svg')
        await block.locator('.message-code-source > summary').click()
        if (language === 'html') await screenshot('html-full-preview', block)
      }
      await loadedImage(assistant.getByRole('button', { name: '查看图片：在线图片', exact: true }))
      await expect(assistant.locator('.message-image-error')).toContainText('图片加载失败')
      await expect(assistant.locator('.message-text > script, .message-text [onerror], .message-text a[href^="javascript:"]')).toHaveCount(0)
      await expect(input).toBeVisible()
      expect(await page.evaluate(() => (window as any).formatPreviewAttack)).toBe('clean')
      await screenshot('safe-html-svg-and-images')

      await reply(`<section><h2>独立 HTML 静态预览</h2><script>top.formatPreviewAttack='raw';fetch('${fixtureUrl}/security-raw')</script><p onclick="alert(1)">保留可见文字</p></section>`)
      const rawFrame = assistant.locator('.message-text iframe')
      await expect(rawFrame).toBeVisible()
      await expect(rawFrame).toHaveAttribute('sandbox', '')
      await expect(assistant.frameLocator('iframe').getByRole('heading', { name: '独立 HTML 静态预览' })).toBeVisible()
      await expect(assistant.frameLocator('iframe').locator('script, [onclick]')).toHaveCount(0)
      expect(await page.evaluate(() => (window as any).formatPreviewAttack)).toBe('clean')
      expect(securityRequests).toEqual([])
      expect(dialogs).toEqual([])
    })

    await test.step('HTML 展开不受聊天容器裁切，源码和外部文件完整保留', async () => {
      const source = `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>ModLoader HTML 预览测试</title><style>body{margin:0;background:#eef3fa;font-family:system-ui}main{max-width:1200px;margin:auto;padding:24px}svg{display:block;width:100%;height:auto}button{padding:8px 16px}#wheel{transform-origin:300px 420px;animation:spin 4s linear infinite}@keyframes spin{to{transform:rotate(360deg)}}</style></head><body><main><h1>完整画布，不止一个小框</h1><p>HTML / SVG 预览与外部交互测试</p><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 640"><rect width="1200" height="640" rx="24" fill="#dbe8f4"/><path d="M60 540H1140" stroke="#9bacbe" stroke-width="3"/><g id="wheel" stroke="#486384" stroke-width="12" fill="none"><circle cx="300" cy="420" r="100"/><path d="M200 420H400M300 320V520"/></g><circle cx="870" cy="420" r="100" stroke="#486384" stroke-width="12" fill="none"/><path d="M300 420L490 240L680 420H300M490 240H780L870 420M680 420L780 240" fill="none" stroke="#ec794b" stroke-width="18" stroke-linejoin="round"/><text x="60" y="610" fill="#486384" font-size="24">完整 SVG 画布底部</text></svg><button id="counter" onclick="this.textContent='已点击 '+(++window.clicks)+' 次'">点击测试</button><script>window.clicks=0;document.body.dataset.scriptRan='yes'</script></main></body></html>`
      await reply(`\`\`\`html\n${source}\n\`\`\``)
      const original = `${source}\n`
      const block = assistant.locator('.message-code-block[data-language="html"]')
      const expand = block.getByRole('button', { name: '展开预览', exact: true })
      await expect(expand).toBeVisible()
      const inlineHeight = await block.locator('iframe').evaluate(element => element.getBoundingClientRect().height)
      expect(inlineHeight).toBeGreaterThan(280)
      await expect(block.frameLocator('iframe').locator('script, [onclick]')).toHaveCount(0)
      await screenshot('html-inline-controls', block)

      await expand.click()
      const viewer = page.getByRole('dialog', { name: 'HTML 预览', exact: true })
      await expect(viewer).toBeVisible()
      const geometry = await viewer.evaluate(element => {
        const box = element.getBoundingClientRect()
        return { width: box.width / innerWidth, height: box.height / innerHeight, frameHeight: element.querySelector('iframe')!.getBoundingClientRect().height }
      })
      expect(geometry.width).toBeGreaterThan(.9)
      expect(geometry.height).toBeGreaterThan(.9)
      expect(geometry.frameHeight).toBeGreaterThan(inlineHeight * 1.5)
      await expect(viewer.locator('iframe')).toHaveAttribute('sandbox', '')
      await expect(viewer.frameLocator('iframe').locator('script, [onclick]')).toHaveCount(0)
      expect(await viewer.frameLocator('iframe').locator('body').evaluate(() => typeof (window as any).modloader)).toBe('undefined')
      await page.screenshot({ path: path.join(directory, 'html-expanded-preview.png') })
      await viewer.getByRole('button', { name: '源码', exact: true }).click()
      await expect(viewer.locator('pre code')).toHaveText(original)
      await expect(viewer.locator('iframe')).toBeHidden()
      await viewer.getByRole('button', { name: '预览', exact: true }).click()
      await expect(viewer.locator('iframe')).toBeVisible()

      // Keep the production IPC, file writing and URL conversion. Stub only OS dialogs / launching.
      const savedPath = path.join(directory, '保存的网页.html')
      await application!.evaluate(({ dialog, shell }, output) => {
        const state = { output, canceled: false, failOpen: false, urls: [] as string[], open: shell.openExternal, save: dialog.showSaveDialog }
        ;(globalThis as any).htmlExportTest = state
        shell.openExternal = async url => { if (state.failOpen) throw new Error('测试：浏览器启动失败'); state.urls.push(url) }
        dialog.showSaveDialog = async () => ({ canceled: state.canceled, filePath: state.canceled ? undefined : state.output })
      }, savedPath)
      await viewer.getByRole('button', { name: '保存', exact: true }).click()
      await expect(viewer.getByRole('status')).toHaveText('已保存 HTML')
      expect(fs.readFileSync(savedPath, 'utf8')).toBe(`\ufeff${original}`)
      await viewer.getByRole('button', { name: '浏览器打开', exact: true }).click()
      await expect(viewer.getByRole('status')).toHaveText('已在浏览器中打开')
      const url = await application!.evaluate(() => (globalThis as any).htmlExportTest.urls[0] as string)
      expect(url.startsWith('file:///')).toBe(true)
      const exportedPath = fileURLToPath(url)
      expect(path.basename(exportedPath)).toBe('preview.html')
      expect(fs.readFileSync(exportedPath, 'utf8')).toBe(`\ufeff${original}`)
      console.log('HTML external preview fixture:', exportedPath)

      await application!.evaluate(() => { (globalThis as any).htmlExportTest.failOpen = true })
      await viewer.getByRole('button', { name: '浏览器打开', exact: true }).click()
      await expect(viewer.getByRole('alert')).toContainText('浏览器启动失败')
      await expect(viewer.getByRole('button', { name: '浏览器打开', exact: true })).toBeEnabled()
      await application!.evaluate(() => { (globalThis as any).htmlExportTest.canceled = true })
      await viewer.getByRole('button', { name: '保存', exact: true }).click()
      await expect(viewer.getByRole('button', { name: '保存', exact: true })).toBeEnabled()
      await expect(viewer.getByRole('status')).toHaveCount(0)
      await expect(viewer.getByRole('alert')).toHaveCount(0)
      await application!.evaluate(({ dialog, shell }) => {
        const state = (globalThis as any).htmlExportTest
        shell.openExternal = state.open
        dialog.showSaveDialog = state.save
        delete (globalThis as any).htmlExportTest
      })
      if (process.env.MODLOADER_TEST_OPEN_EXTERNAL === '1') {
        await viewer.getByRole('button', { name: '浏览器打开', exact: true }).click()
        await expect(viewer.getByRole('status')).toHaveText('已在浏览器中打开')
        console.log('Opened the harmless HTML interaction fixture in the system browser.')
      }

      await application!.evaluate(({ BrowserWindow }) => { const window = BrowserWindow.getAllWindows()[0]; window.setSize(940, 760); window.webContents.setZoomFactor(1.2) })
      await expect(viewer.getByRole('button', { name: '关闭预览', exact: true })).toBeVisible()
      const overflow = await viewer.evaluate(element => element.scrollWidth - element.clientWidth)
      expect(overflow).toBeLessThanOrEqual(1)
      // Electron zoom needs native capture/input; CDP screenshots crop the zoomed viewport,
      // and CDP Escape does not invoke Chromium's native dialog close watcher.
      const smallView = await application!.evaluate(async ({ BrowserWindow }) => (await BrowserWindow.getAllWindows()[0].webContents.capturePage()).toPNG().toString('base64'))
      fs.writeFileSync(path.join(directory, 'html-expanded-small-window.png'), Buffer.from(smallView, 'base64'))
      console.log('Small HTML viewer:', await viewer.evaluate(element => ({ viewport: [innerWidth, innerHeight], bounds: element.getBoundingClientRect().toJSON(), close: element.querySelector('.html-preview-close')!.getBoundingClientRect().toJSON() })))
      await viewer.frameLocator('iframe').getByRole('heading').click()
      await application!.evaluate(({ BrowserWindow }) => {
        const contents = BrowserWindow.getAllWindows()[0].webContents
        contents.sendInputEvent({ type: 'keyDown', keyCode: 'Escape' })
        contents.sendInputEvent({ type: 'keyUp', keyCode: 'Escape' })
      })
      await expect(viewer).toHaveCount(0)
      await expect(expand).toBeFocused()
      await expect(input).toBeVisible()
      await application!.evaluate(({ BrowserWindow }) => { const window = BrowserWindow.getAllWindows()[0]; window.setSize(1440, 960); window.webContents.setZoomFactor(1) })
      await expand.click()
      await viewer.getByRole('button', { name: '关闭预览', exact: true }).click()
      await expect(viewer).toHaveCount(0)
    })

    await test.step('流式不完整图表等待，闭合后渲染，最终错误保留源码，思考复用格式', async () => {
      await send()
      const partialFormula = String.raw`\[
2\left(\frac78-p_3\right)+3p_3
=`
      delta({ content: partialFormula })
      await expect(assistant.locator('.message-text')).toHaveText(partialFormula)
      await expect(assistant.locator('.message-text .katex, .message-text h1, .message-text h2')).toHaveCount(0)
      delta({ content: String.raw`
\frac74+p_3.
\]` })
      await expect(assistant.locator('.message-text .katex-display')).toHaveCount(1)
      await expect(assistant.locator('.message-text h1, .message-text h2, .message-text .katex-error')).toHaveCount(0)
      await finish()

      await reply(partialFormula)
      await expect(assistant.locator('.message-text')).toHaveText(partialFormula)
      await expect(assistant.locator('.message-text .katex, .message-text h1, .message-text h2')).toHaveCount(0)

      await send()
      delta({ content: String.raw`\[
\begin{aligned}
\text{圆形苹果} &\ge 1 \\` })
      await expect(assistant.locator('.message-text')).toContainText('圆形苹果')
      await expect(page.getByRole('button', { name: '停止生成', exact: true })).toBeVisible()
      expect(errors).toEqual([])
      delta({ content: String.raw`
\text{五角星桃子} &\ge 1
\end{aligned}
\]` })
      await finish()
      await expect(assistant.locator('.message-text .katex-display')).toHaveCount(1)

      await send()
      delta({ content: '```mermaid\nflowchart LR\n  Start[' })
      const pending = assistant.locator('.message-code-block[data-language="mermaid"]')
      await expect(pending.locator('pre code')).toContainText('Start[')
      await expect(pending.getByRole('status')).toBeVisible()
      await expect(pending.locator('img, .message-preview-error')).toHaveCount(0)
      delta({ content: '开始] --> End[完成]\n```' })
      await loadedImage(pending.locator('img'))
      await finish()

      const invalid = 'this is not a valid mermaid diagram'
      await reply(`\`\`\`mermaid\n${invalid}\n\`\`\``)
      const block = assistant.locator('.message-code-block[data-language="mermaid"]')
      await expect(block.locator('.message-preview-error')).toBeVisible()
      await expect(await showSource(block)).toHaveText(`${invalid}\n`)
      await screenshot('invalid-diagram-source')

      await reply('思考内容也支持公式和代码。', `推理公式 $r^2$。\n\n${setextFormulas[1]}\n\n\`\`\`json\n{"step": 1, "ready": true}\n\`\`\``)
      await assistant.locator('.message-reasoning > summary').click()
      const reasoning = assistant.getByRole('region', { name: '思考内容' })
      await expect(reasoning.locator('.katex')).toHaveCount(2)
      await expect(reasoning.locator('.katex-display')).toHaveCount(1)
      await expect(reasoning.locator('h1, h2, .katex-error')).toHaveCount(0)
      await expect(reasoning.locator('code .hljs-attr').first()).toBeVisible()
      await screenshot('reasoning-shared-renderer')
    })
    expect(errors).toEqual([])
  } catch (error) {
    console.log('Chat formats page errors:', errors)
    if (activePage && !activePage.isClosed()) {
      const output = path.join(directory, 'failure.png')
      await activePage.screenshot({ path: output })
      console.log(`Chat formats failure screenshot: ${output}`)
    }
    throw error
  } finally {
    response?.end()
    await application?.close()
    server.closeAllConnections()
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
})
