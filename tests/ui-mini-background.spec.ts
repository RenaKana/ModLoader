import { expect, test, chromium } from '@playwright/test'
import { build } from 'esbuild'
import * as fs from 'node:fs'
import * as path from 'node:path'
import type { GenerationAccounting, GenerationEvent } from '../src/types'

const root = path.resolve(__dirname, '..')
const artifacts = process.env.MODLOADER_TEST_ARTIFACTS || path.join(root, '.artifacts/codex/monitor-mini-background-20261101/ui')
const record = { run_id: 'run', instance_id: 'instance', request_key: 'call', alias: 'NInfer', backend: 'ninfer', revision: 1,
  aggregation: 'request', turn_id: null, turn_state: 'running', status: 'running', labels: [], started_at: 10, duration_s: 3,
  usage: { prompt_tokens: 1000, completion_tokens: 100 }, timings: { predicted_n: 100, predicted_ms: 2500 } }
function event(tokens: number, observed = 21, final = false): GenerationEvent {
  const accounting: GenerationAccounting = { request: { ...record, status: final ? 'completed' : 'running', streaming: true,
    counters: { input: 1100, output: tokens, cached: 100 },
    measurements: { decode: { tokens, seconds: 3, value: tokens / 3, reported: 1, coverage: 'complete', basis: 'backend_decode' },
      prefill: { tokens: null, seconds: null, value: null, reported: 0, coverage: 'unknown', basis: null } } },
    turn: { source: 'external', key: 'request:call', turn_id: null, client_request_id: null, request_keys: ['call'], prior: null } }
  return { run_id: 'run', instance_id: 'instance', observed_at: observed, accounting: final ? null : accounting, finalized: final ? [accounting] : [],
    live_generation: final ? null : { request_key: 'call', phase: 'output', scope: 'request', source: 'native', tokens_per_second: tokens / 2,
      backend_running: 1, backend_queued: 0, available_for_chat: true, reason: null, window_s: 3 } }
}
async function fixture(initial = true, delayed = false) {
  fs.mkdirSync(artifacts, { recursive: true }); const directory = fs.mkdtempSync(path.join(artifacts, 'background-'))
  const source = (file: string) => JSON.stringify(path.join(root, file).replaceAll('\\', '/'))
  fs.writeFileSync(path.join(directory, 'fixture.tsx'), `import React,{useState}from'react';import{createRoot}from'react-dom/client';
    import{createGenerationEventStore,readGenerationEvents,generationEventIdentity}from ${source('src/monitor-generation-events')};
    import{miniSnapshot,useMiniMonitorPublisher}from ${source('src/mini-monitor')};import{externalGenerationView,rememberGeneration}from ${source('src/monitor-generation')};
    import{MonitorGeneration}from ${source('src/MonitorGeneration')};import ${source('src/styles.css')};
    const frames=[];window.frameNotifications=0;window.miniSnapshots=[];window.visibilityGets=0;
    const store=createGenerationEventStore(callback=>frames.push(callback));window.store=store;
    window.flushFrames=()=>frames.splice(0).forEach(callback=>callback(0));window.frameCount=()=>frames.length;
    const listeners=new Set();let visible=${initial};window.setVisible=value=>{visible=value;listeners.forEach(fn=>fn(value))};
    window.modloader={connection:async()=>({baseUrl:'http://background.fixture',token:'fixture'}),
      getMonitorVisibility:()=>{window.visibilityGets++;return ${delayed ? 'new Promise(resolve=>window.resolveInitial=resolve)' : 'Promise.resolve(visible)'}},
      onMonitorVisibility:fn=>{listeners.add(fn);return()=>listeners.delete(fn)},publishMiniMonitor:snapshot=>window.miniSnapshots.push({at:performance.now(),...snapshot})};
    let controller;const stream=new ReadableStream({start(value){controller=value}});window.emit=event=>controller.enqueue(new TextEncoder().encode('event: generation\\ndata: '+JSON.stringify(event)+'\\n\\n'));
    window.disconnect=()=>{controller.close();store.clear()};const fetch=window.fetch.bind(window);window.fetch=(input,options)=>String(input).endsWith('/generation-events')?Promise.resolve(new Response(stream,{headers:{'Content-Type':'text/event-stream'}})):fetch(input,options);
    let slow=${JSON.stringify(record)};const instance={id:'instance',alias:'NInfer',backend:'ninfer',state:'ready',active_requests:1};
    const payload=()=>({instance_id:'instance',backend:'ninfer',state:'ready',updated_at:20,groups:[],unavailable:[],
      external_history:{run_id:'run',aggregation:'request',turn_id:null,state:slow.status,requests:[slow]},
      live_generation:slow.status==='running'?{request_key:'call',scope:'request',phase:'output',source:'poll',tokens_per_second:11.11,average_tokens_per_second:40,average_basis:'backend_decode',backend_running:1,backend_queued:0,available_for_chat:true,reason:null,window_s:3}:null});
    store.reset('instance');store.confirm(payload());window.flushFrames();store.subscribe(()=>window.frameNotifications++);
    readGenerationEvents('instance',new AbortController().signal,event=>store.receive(event)).catch(error=>{throw error});
    function Fixture(){window.rootRenders=(window.rootRenders||0)+1;const[version,setVersion]=useState(0);
      window.slow=record=>{if(record)slow={...slow,...record};store.confirm(payload());setVersion(value=>value+1)};
      const data=payload(),memory=rememberGeneration(undefined,data),generation=externalGenerationView(data.external_history,memory,data),
      state={instances:[instance],settings:{},hardware:{cpu_percent:21}},monitor={instance,instances:[instance],instanceId:'instance',ready:true,payload:data,sampleError:'',memory,generation,gpus:[]};
      useMiniMonitorPublisher(miniSnapshot(state,monitor,true),store,generationEventIdentity(generation.requests.at(-1),store),generation.requests.at(-1));
      return <main style={{padding:24}}><MonitorGeneration generation={generation} scope="request" onSelectRequest={()=>{}} generationEvents={store}/></main>
    };createRoot(document.getElementById('root')).render(<Fixture/>);`)
  await build({ entryPoints: [path.join(directory, 'fixture.tsx')], outfile: path.join(directory, 'fixture.js'), bundle: true, format: 'iife', jsx: 'automatic', loader: { '.woff2': 'file' }, define: { 'process.env.NODE_ENV': '"development"' } })
  const browser = await chromium.launch({ channel: 'msedge', headless: true }), page = await browser.newPage()
  const errors: string[] = []; page.on('pageerror', error => errors.push(error.message))
  const now = Date.parse('2026-11-01T01:00:00Z'); await page.clock.install({ time: new Date(now) })
  await page.route('http://background.fixture/**', route => new URL(route.request().url()).pathname === '/'
    ? route.fulfill({ contentType: 'text/html', body: '<link rel="stylesheet" href="fixture.css"><div id="root"></div><script src="fixture.js"></script>' })
    : route.fulfill({ path: path.join(directory, path.basename(new URL(route.request().url()).pathname)) }))
  await page.goto('http://background.fixture/'); await expect(page.locator('.monitor-speed-current strong')).toHaveText('11.11'); await page.clock.pauseAt(new Date(now + 1000))
  const send = async (value: GenerationEvent) => { await page.evaluate(value => (window as any).emit(value), value); await page.clock.runFor(20) }
  const latest = () => page.evaluate(() => (window as any).miniSnapshots.at(-1))
  return { browser, page, errors, directory, send, latest }
}

test('RAF冻结且Mini仍可见时连续20ms计量即时发布；主窗只在恢复时合并一帧', async () => {
  const f = await fixture()
  try {
    const before = await f.page.evaluate(() => ({ renders: (window as any).rootRenders, notices: (window as any).frameNotifications }))
    for (const [i, tokens] of [150, 180, 210].entries()) {
      await f.send(event(tokens, 21 + i)); const mini = await f.latest()
      expect(mini.metrics.current.value).toBe((tokens / 2).toFixed(2)); expect(mini.metrics.average.value).toBe((tokens / 3).toFixed(2))
      expect(mini.metrics.generated.value).toBe(String(tokens)); expect(mini.metrics.context.value).toBe(`${(1100 + tokens).toLocaleString('en-US')} / —`)
    }
    await expect(f.page.locator('.monitor-speed-current strong')).toHaveText('11.11')
    expect(await f.page.evaluate(() => ({ renders: (window as any).rootRenders, notices: (window as any).frameNotifications }))).toEqual(before)
    expect(await f.page.evaluate(() => (window as any).frameCount())).toBe(1)
    await f.page.evaluate(() => (window as any).flushFrames()); await expect(f.page.locator('.monitor-speed-current strong')).toHaveText('105.00')
    expect(await f.page.evaluate(() => (window as any).frameNotifications)).toBe(before.notices + 1)
    const snapshots = await f.page.evaluate(() => (window as any).miniSnapshots)
    const ticks = snapshots.filter((snapshot: any) => ['150', '180', '210'].includes(snapshot.metrics.generated.value)).map((snapshot: any) => snapshot.at)
    const intervalsMs = ticks.slice(1).map((at: number, index: number) => at - ticks[index])
    expect(ticks).toHaveLength(3); expect(intervalsMs).toEqual([20, 20])
    fs.writeFileSync(path.join(f.directory, 'cadence.json'), JSON.stringify({ intervalsMs, snapshots }, null, 2))
    expect(f.errors).toEqual([])
  } finally { await f.browser.close() }
})

test('初始两窗后台不逐事件发；slow照仍发，恢复立即最新且迟到初值不能覆盖可见事件', async () => {
  const f = await fixture(false, true)
  try {
    const count = await f.page.evaluate(() => (window as any).miniSnapshots.length)
    await f.send(event(150)); await f.send(event(180, 22))
    expect(await f.page.evaluate(() => (window as any).miniSnapshots.length)).toBe(count)
    await f.page.evaluate(() => (window as any).slow()); await expect.poll(async () => (await f.latest()).metrics.generated.value).toBe('180')
    await f.page.evaluate(() => (window as any).setVisible(true)); expect((await f.latest()).metrics.average.value).toBe('60.00')
    await f.page.evaluate(() => (window as any).resolveInitial(false)); await f.send(event(210, 23)); expect((await f.latest()).metrics.generated.value).toBe('210')
    await f.page.evaluate(() => (window as any).setVisible(false)); const hiddenCount = await f.page.evaluate(() => (window as any).miniSnapshots.length)
    await f.send(event(240, 24)); expect(await f.page.evaluate(() => (window as any).miniSnapshots.length)).toBe(hiddenCount)
    await f.page.evaluate(() => (window as any).setVisible(true)); expect((await f.latest()).metrics.generated.value).toBe('240')
    await f.page.evaluate(() => { for (const value of [false, true, false, true, true]) (window as any).setVisible(value) })
    const resumedCount = await f.page.evaluate(() => (window as any).miniSnapshots.length)
    await f.send(event(270, 25)); expect((await f.latest()).metrics.generated.value).toBe('270')
    expect(await f.page.evaluate(() => (window as any).miniSnapshots.length)).toBe(resumedCount + 1)
    expect(await f.page.evaluate(() => (window as any).visibilityGets)).toBe(1); expect(f.errors).toEqual([])
  } finally { await f.browser.close() }
})

test('无RAF终态与下一身份同帧不丢；断连/慢照接管/reset和错身份不复活旧快值', async () => {
  const f = await fixture()
  try {
    await f.send(event(150)); expect((await f.latest()).metrics.generated.value).toBe('150')
    for (const wrong of [{ run_id: 'wrong' }, { instance_id: 'wrong' }, { live_generation: { ...event(999).live_generation!, request_key: 'wrong' } }]) {
      await f.send({ ...event(999, 22), ...wrong }); expect((await f.latest()).metrics.generated.value).toBe('150')
    }
    await f.page.evaluate(() => (window as any).store.reset('another'))
    expect((await f.latest()).metrics).toMatchObject({ current: { value: '11.11' }, generated: { value: '100' } })
    await f.send(event(999, 23)); expect((await f.latest()).metrics.generated.value).toBe('100')
    await f.page.evaluate(() => { (window as any).store.reset('instance'); (window as any).slow() })
    await f.send(event(180, 21)); expect((await f.latest()).metrics.generated.value).toBe('180')
    const final = event(240, 22, true)
    await f.send({ ...final, live_generation: { ...event(0).live_generation!, request_key: 'next', phase: 'waiting', tokens_per_second: null } })
    expect((await f.latest()).metrics).toMatchObject({ current: { value: '—' }, average: { value: '80.00' }, generated: { value: '240' }, context: { value: '1,340 / —' } })
    await f.page.evaluate(() => (window as any).disconnect()); expect((await f.latest()).metrics.generated.value).toBe('240')
    const beforeSlow = await f.page.evaluate(() => (window as any).miniSnapshots.length)
    await f.page.evaluate(request => (window as any).slow(request), final.finalized[0].request)
    expect((await f.latest()).metrics).toMatchObject({ average: { value: '80.00' }, generated: { value: '240' } })
    expect(await f.page.evaluate(start => (window as any).miniSnapshots.slice(start).every((snapshot: any) => snapshot.metrics.generated.value === '240'), beforeSlow)).toBe(true)
    await f.page.evaluate(() => (window as any).store.reset('another'))
    expect((await f.latest()).metrics.generated.value).toBe('240') // authoritative terminal baseline, not another instance's fast data
    expect(f.errors).toEqual([])
  } finally { await f.browser.close() }
})
