import { test, expect, chromium } from '@playwright/test'
import { createServer } from 'vite'
import * as fs from 'node:fs'
import * as path from 'node:path'

test('统一悬停说明：延迟、键盘、快速切换、可悬停、窗口避让与清理', async () => {
  test.setTimeout(60_000)
  const root = path.resolve(__dirname, '..')
  const directory = fs.mkdtempSync(path.join(root, 'work', 'ui-tooltip-'))
  fs.writeFileSync(path.join(directory, 'index.html'), `<!doctype html><meta charset="utf-8">
    <style>
      .fixture { padding:100px 60px; }
      .fixture .panel { max-width:560px; padding:24px; }
      .fixture h2 { margin:0 0 20px; font-size:20px; }
      .fixture-stats { display:flex; align-items:center; gap:18px; color:var(--muted); }
      .fixture-stats > button { background:transparent; border:0; padding:4px; font-size:13px; }
      #scroll { width:180px; height:50px; overflow:auto; margin-top:24px; }
    </style>
    <main class="fixture"><section class="panel"><h2>测试对话</h2><div class="fixture-stats">
      <button id="first" data-tooltip="当前生成速度\n当前请求 · 最近 1 s" aria-label="当前生成速度" aria-describedby="existing">42.80 tok/s</button>
      <button id="second" data-tooltip="本次请求平均生成速度" aria-label="本次请求平均生成速度">41.00 tok/s</button>
      <span id="icon-root"></span>
    </div><div id="existing" hidden>已有说明</div>
    <div id="scroll"><button id="scroll-trigger" data-tooltip="滚动时关闭说明">滚动区域</button><div style="height:200px"></div></div>
    </section></main>
    <button id="corner" data-tooltip="输入 Token：8,192\n输出 Token：1,024\n这是一条靠右上角的多行说明，完整显示而不越过窗口边缘。" style="position:fixed;right:4px;top:4px">右上</button>
    <button id="bottom" data-tooltip="左下角说明" style="position:fixed;left:4px;bottom:4px">左下</button>
    <div id="tooltip-root"></div>
    <script type="module">
      import React from 'react'; import {createRoot} from 'react-dom/client'; import {RefreshCw} from 'lucide-react';
      import Tooltip from '/src/Tooltip.tsx'; import {IconButton} from '/src/components.tsx'; import '/src/styles.css';
      window.fixtureClicks=0;
      window.renderIcon=(disabled=false)=>createRootIcon.render(React.createElement(IconButton,{icon:RefreshCw,label:'刷新状态',disabled,onClick:()=>window.fixtureClicks++}));
      const createRootIcon=createRoot(document.getElementById('icon-root')); window.renderIcon();
      createRoot(document.getElementById('tooltip-root')).render(React.createElement(React.StrictMode,null,React.createElement(Tooltip)));
    </script>`)
  const reused = process.env.MODLOADER_TEST_BASE_URL
  const server = reused ? null : await createServer({
    root,
    optimizeDeps: { entries: [path.join(directory, 'index.html')] },
    server: { port: 0, host: '127.0.0.1', watch: { ignored: ['**/work/**'] } },
  })
  if (server) await server.listen()
  const origin = reused || `http://127.0.0.1:${(server!.httpServer!.address() as { port: number }).port}`
  const browser = await chromium.launch({ channel: 'msedge' })
  try {
    const page = await browser.newPage({ viewport: { width: 1000, height: 700 } })
    const errors: string[] = []
    page.on('pageerror', error => errors.push(error.message))
    await page.goto(`${origin}/${path.relative(root, directory).replaceAll('\\', '/')}/index.html`)
    const first = page.locator('#first'), second = page.locator('#second'), tooltip = page.getByRole('tooltip')
    const host = page.locator('.app-tooltip')
    const icon = page.getByRole('button', { name: '刷新状态', exact: true })
    await expect(icon).toBeVisible()
    await page.evaluate(() => document.fonts.ready)
    const leave = async () => {
      await page.mouse.move(500, 500)
      await page.evaluate(() => (document.activeElement as HTMLElement)?.blur())
      await expect(tooltip).toBeHidden()
    }
    const inWindow = async () => {
      const box = (await tooltip.boundingBox())!
      expect(box.x).toBeGreaterThanOrEqual(8)
      expect(box.y).toBeGreaterThanOrEqual(8)
      expect(box.x + box.width).toBeLessThanOrEqual(992)
      expect(box.y + box.height).toBeLessThanOrEqual(692)
    }

    // A short fly-by never mounts a visible tooltip or an accessible description.
    await first.hover()
    expect(await host.evaluate(element => element.matches(':popover-open'))).toBe(false)
    await page.mouse.move(500, 500)
    await page.waitForTimeout(340)
    await expect(tooltip).toBeHidden()
    await expect(first).toHaveAttribute('aria-describedby', 'existing')

    const entry = host.evaluate(element => new Promise<{ opacity: number; transform: string; duration: number[] }[]>((resolve, reject) => {
      const samples: { opacity: number; transform: string; duration: number[] }[] = []
      const deadline = performance.now() + 2000
      let start = 0
      const sample = () => {
        const now = performance.now()
        if (element.matches(':popover-open')) {
          start ||= now
          const style = getComputedStyle(element)
          samples.push({ opacity: Number(style.opacity), transform: style.transform, duration: element.getAnimations().map(animation => Number(animation.effect!.getTiming().duration)) })
          if (now - start >= 180) { resolve(samples); return }
        }
        if (now > deadline) { reject(new Error('未捕获悬停说明入场')); return }
        requestAnimationFrame(sample)
      }
      sample()
    }))
    await first.hover()
    await expect(tooltip).toHaveText('当前生成速度\n当前请求 · 最近 1 s')
    const entryFrames = await entry
    expect(entryFrames.some(frame => frame.opacity > 0 && frame.opacity < 1)).toBe(true)
    expect(entryFrames.some(frame => frame.transform !== 'matrix(1, 0, 0, 1, 0, 0)' && frame.transform !== 'none')).toBe(true)
    expect(entryFrames.some(frame => frame.duration.includes(160))).toBe(true)
    await expect(tooltip).toHaveCSS('opacity', '1')
    await expect(first).toHaveAttribute('aria-describedby', new RegExp(`existing .+`))
    await expect(tooltip).toHaveCSS('background-color', 'rgb(255, 255, 255)')
    await expect(tooltip).toHaveCSS('border-radius', '9px')
    await expect(tooltip).toHaveCSS('font-size', '12px')
    await page.screenshot({ path: path.join(directory, 'tooltip-multiline.png') })

    // Move through the gap into the tooltip; it remains readable and selectable.
    const floating = (await tooltip.boundingBox())!
    await page.mouse.move(floating.x + floating.width / 2, floating.y + floating.height / 2, { steps: 3 })
    await page.waitForTimeout(180)
    await expect(tooltip).toBeVisible()
    await second.hover()
    await expect(tooltip).toHaveText('本次请求平均生成速度')
    await expect(host).toHaveAttribute('data-instant', '')
    await expect(first).toHaveAttribute('aria-describedby', 'existing')
    await page.keyboard.press('Escape')
    await expect(tooltip).toBeHidden()
    await second.hover()
    await expect(tooltip).toBeHidden()
    await leave()

    // Keyboard input opens immediately and Escape leaves focus on its trigger.
    await first.focus()
    await page.keyboard.press('Tab')
    await expect(second).toBeFocused()
    await expect(tooltip).toHaveText('本次请求平均生成速度')
    await page.keyboard.press('Escape')
    await expect(second).toBeFocused()
    await expect(tooltip).toBeHidden()
    await leave()

    await page.locator('#corner').hover()
    await expect(tooltip).toContainText('输入 Token：8,192')
    await expect(tooltip).toHaveCSS('opacity', '1')
    await inWindow()
    await page.screenshot({ path: path.join(directory, 'tooltip-edge.png') })
    await page.locator('#bottom').hover()
    await expect(tooltip).toHaveText('左下角说明')
    await inWindow()
    await leave()

    await icon.hover()
    await expect(tooltip).toHaveText('刷新状态')
    await expect(icon).not.toHaveAttribute('title')
    await icon.click()
    expect(await page.evaluate(() => (window as any).fixtureClicks)).toBe(1)
    await expect(tooltip).toBeHidden()
    await page.evaluate(() => (window as any).renderIcon(true))
    await leave()
    await icon.hover()
    await expect(tooltip).toHaveText('刷新状态')
    await expect(icon).toBeDisabled()
    await leave()

    // Ordinary tooltips retain literal paths, Markdown and HTML-looking text.
    const literal = String.raw`C:\models\Qwen\weights.gguf · \(M\) **原文** <b>保持文字</b>`
    await first.evaluate((element, text) => { (element as HTMLElement).dataset.tooltip = text }, literal)
    await first.hover()
    await expect(tooltip).toHaveText(literal)
    await expect(host).not.toHaveAttribute('data-tooltip-format')
    await expect(tooltip.locator('.chat-title, .katex, b')).toHaveCount(0)
    await page.screenshot({ path: path.join(directory, 'tooltip-plain-literal.png') })
    await leave()

    // Text refreshes in place; trigger removal, hidden sections and scroll clean up.
    await first.hover()
    await expect(tooltip).toBeVisible()
    await first.evaluate(element => { (element as HTMLElement).dataset.tooltip = '当前请求 · 最近 2 s' })
    await expect(tooltip).toHaveText('当前请求 · 最近 2 s')
    await first.evaluate(element => element.remove())
    await expect(tooltip).toBeHidden()
    await second.hover()
    await expect(tooltip).toBeVisible()
    await second.evaluate(element => { element.parentElement!.hidden = true })
    await expect(tooltip).toBeHidden()
    await second.evaluate(element => { element.parentElement!.hidden = false })
    await page.locator('#scroll-trigger').hover()
    await expect(tooltip).toBeVisible()
    await page.locator('#scroll').evaluate(element => { element.scrollTop = 50 })
    await expect(tooltip).toBeHidden()

    await page.emulateMedia({ reducedMotion: 'reduce' })
    await page.evaluate(() => document.documentElement.style.setProperty('--font-scale', '1.5'))
    await page.locator('#corner').hover()
    await expect(tooltip).toContainText('输入 Token')
    await expect(tooltip).toHaveCSS('transform', 'none')
    await expect(tooltip).toHaveCSS('font-size', '18px')
    await inWindow()
    await page.screenshot({ path: path.join(directory, 'tooltip-scaled-reduced-motion.png') })
    expect(errors).toEqual([])
    console.log('Tooltip screenshots:', directory)
  } finally {
    await browser.close()
    await server?.close()
  }
})
