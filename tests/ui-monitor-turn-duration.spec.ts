import { expect, test, chromium } from '@playwright/test'
import { build } from 'esbuild'
import * as fs from 'node:fs'
import * as path from 'node:path'

const root = path.resolve(__dirname, '..')
const artifacts = process.env.MODLOADER_TEST_ARTIFACTS || path.join(root, '.artifacts/codex/monitor-token-consistency-20261031/frontend/turn-duration-switch')

test('切换轮次首个DOM提交不能借用上一轮耗时；250ms局部钟表及终态冻结不变', async () => {
  fs.mkdirSync(artifacts, { recursive: true })
  const directory = fs.mkdtempSync(path.join(artifacts, 'turn-duration-'))
  const source = (file: string) => JSON.stringify(path.join(root, file).replaceAll('\\', '/'))
  fs.writeFileSync(path.join(directory, 'fixture.tsx'), `
    import React,{useLayoutEffect,useState}from'react';import{createRoot}from'react-dom/client';
    import{MonitorGeneration}from ${source('src/MonitorGeneration')};import{externalGenerationView}from ${source('src/monitor-generation')};
    import ${source('src/styles.css')};import ${source('src/resource-cards.css')};
    window.commits=[];window.timers=[];window.clearedTimers=[];
    const start=window.setInterval.bind(window),stop=window.clearInterval.bind(window);
    window.setInterval=(callback,ms,...args)=>{const id=start(callback,ms,...args);window.timers.push({id,ms});return id};
    window.clearInterval=id=>{window.clearedTimers.push(id);stop(id)};
    function Fixture(){
      const[current,setCurrent]=useState(false),duration=current?21.43:29.49,key=current?'current-turn':'previous-turn';
      const status=current?'running':'completed',at=Date.now()/1000;
      const record={run_id:'run',instance_id:'instance',request_key:key,alias:'NInfer',backend:'ninfer',revision:1,
        aggregation:'turn',turn_id:key,turn_state:status,status,labels:[],started_at:at-duration,...(current?{}:{ended_at:at}),
        duration_s:duration,usage:{prompt_tokens:1000,completion_tokens:100},timings:{predicted_n:100,predicted_ms:2500}};
      const generation=externalGenerationView({run_id:'run',aggregation:'turn',turn_id:key,state:status,requests:[record]});
      useLayoutEffect(()=>{window.commits.push({key,elapsed:document.querySelector('.monitor-request-live>div:nth-child(2) strong')?.textContent})});
      return <main style={{padding:24}}><button onClick={()=>setCurrent(false)}>上一轮</button><button onClick={()=>setCurrent(true)}>本轮</button>
        <MonitorGeneration generation={generation} scope="turn" modelLabel="NInfer" onSelectRequest={()=>{}}/></main>
    };createRoot(document.getElementById('root')).render(<Fixture/>);`)
  await build({ entryPoints: [path.join(directory, 'fixture.tsx')], outfile: path.join(directory, 'fixture.js'), bundle: true, format: 'iife', jsx: 'automatic', loader: { '.woff2': 'file' }, define: { 'process.env.NODE_ENV': '"development"' } })
  const browser = await chromium.launch({ channel: 'msedge', headless: true })
  try {
    const page = await browser.newPage({ viewport: { width: 1366, height: 460 } })
    const errors: string[] = []
    page.on('pageerror', error => errors.push(error.message))
    const now = Date.parse('2026-11-01T00:26:40+08:00')
    await page.clock.install({ time: new Date(now) })
    await page.route('http://duration.fixture/**', route => {
      if (new URL(route.request().url()).pathname === '/') return route.fulfill({ contentType: 'text/html', body: '<!doctype html><meta charset="utf-8"><link rel="stylesheet" href="fixture.css"><div id="root"></div><script src="fixture.js"></script>' })
      return route.fulfill({ path: path.join(directory, path.basename(new URL(route.request().url()).pathname)) })
    })
    await page.goto('http://duration.fixture/')
    const duration = page.locator('.monitor-request-live>div:nth-child(2) strong')
    await expect(duration).toHaveText('29.49')
    await page.clock.pauseAt(new Date(now + 1000))
    await page.getByRole('button', { name: '本轮', exact: true }).click()
    const commits = await page.evaluate(() => (window as any).commits)
    fs.writeFileSync(path.join(directory, 'first-commit.json'), JSON.stringify(commits, null, 2))
    expect(commits.at(-1)).toEqual({ key: 'current-turn', elapsed: '21.43' })
    await expect(duration).toHaveText('21.43')
    await page.screenshot({ path: path.join(directory, 'turn-duration-switch.png') })
    expect(await page.evaluate(() => (window as any).timers.map((timer: any) => timer.ms))).toEqual([250])
    await page.clock.runFor(249); await expect(duration).toHaveText('21.43')
    await page.clock.runFor(1); await expect(duration).toHaveText('21.68')
    expect(await page.evaluate(() => (window as any).commits.length)).toBe(2)
    await page.getByRole('button', { name: '上一轮', exact: true }).click()
    await expect(duration).toHaveText('29.49')
    await page.clock.runFor(500); await expect(duration).toHaveText('29.49')
    expect(await page.evaluate(() => (window as any).clearedTimers)).toEqual(await page.evaluate(() => (window as any).timers.map((timer: any) => timer.id)))
    expect(errors).toEqual([])
  } finally { await browser.close() }
})
