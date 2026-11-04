import { test, expect, _electron as electron, type ElectronApplication, type Page } from '@playwright/test'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { miniSnapshot, moveMiniMetric, readMiniSelection, MINI_PREFERENCE_KEY } from '../src/mini-monitor'
import { rememberGeneration } from '../src/monitor-generation'
import type { AppState, Telemetry } from '../src/types'
import type { MonitorState } from '../src/useMonitorState'
import { pinNavigation } from './navigation'

test('小窗保留请求结果但不借给新请求，不把采样失败显示为零速', () => {
  const complete: Telemetry = { instance_id:'a', backend:'llama_cpp', state:'ready', updated_at:1, unavailable:[],
    groups:[{id:'api_request',title:'最近请求',request_status:'completed',request_key:'old',metrics:[
      {id:'first_output_s',label:'首输出',value:.68,kind:'gauge'}, {id:'duration_s',label:'耗时',value:5,kind:'gauge'},
      {id:'prompt_tokens',label:'输入',value:128,kind:'gauge'}, {id:'completion_tokens',label:'输出',value:32,kind:'gauge'},
      {id:'output_speed',label:'速度',value:40.31,kind:'gauge',rate_basis:'decode'}]}],
    live_generation:{scope:'request',source:'fixture',backend_running:0,backend_queued:0,tokens_per_second:null,window_s:null,available_for_chat:false,reason:null} }
  const instance = { id:'a',alias:'fixture',state:'ready',active_requests:0 }
  const state = {settings:{},hardware:{},instances:[instance]} as AppState
  const monitor = {instance,instanceId:'a',instances:[instance],ready:true,payload:complete,sampleError:'',memory:rememberGeneration(undefined,complete),gpus:[]} as unknown as MonitorState
  let snapshot = miniSnapshot(state,monitor,true)
  expect(snapshot.metrics.first_output).toMatchObject({value:'0.68',basis:'上次'})
  expect(snapshot.metrics.current.value).toBe('0.00')
  expect(snapshot.metrics.average).toMatchObject({value:'40.31',basis:'上次'})
  monitor.payload = {...complete,live_generation:{...complete.live_generation!,request_key:'new',backend_running:1}}
  instance.active_requests = 1
  snapshot = miniSnapshot(state,monitor,true)
  expect(snapshot.metrics.first_output.value).toBe('—')
  expect(snapshot.metrics.prompt.value).toBe('—')
  expect(snapshot.metrics.average.value).toBe('—')
  monitor.sampleError = 'failed'
  expect(miniSnapshot(state,monitor,true).metrics.current.value).toBe('—')
  expect(miniSnapshot(state,monitor,false).stateLabel).toBe('服务离线')
})

test('显示项允许全部隐藏，排序不重复或丢失已选项', () => {
  expect(readMiniSelection([])).toEqual([])
  expect(readMiniSelection(['current','unknown','current','average'])).toEqual(['current','average'])
  expect(readMiniSelection(null)).toEqual(['first_output','current','average'])
  expect(moveMiniMetric(['first_output','current','average'],'average','first_output')).toEqual(['average','first_output','current'])
  expect(moveMiniMetric(['current','average'],'current',null)).toEqual(['average','current'])
})

test('独立小窗共享采样、自由排列缩放、隐藏后更新及跨重启恢复', async () => {
  test.setTimeout(120_000)
  const root = path.resolve(__dirname,'..'), directory = fs.mkdtempSync(path.join(root,'work','ui-mini-monitor-'))
  const data = path.join(directory,'data')
  let mode: 'generating' | 'idle' | 'new' | 'failed' = 'generating', scale = 1, reads = 0
  let application: ElectronApplication, main: Page, mini: Page
  const errors: string[] = []
  const instances = ['a','b'].map(id=>({id,alias:id==='a'?'Qwen3 · 30B':'第二个模型',backend:'llama_cpp',state:'ready',
    model_id:id,profile_id:id,profile_revision:1,runtime_id:'fixture',port:0,started_at:'',active_requests:1,capabilities:{}}))
  const hardware = { cpu_percent:21, memory_used:32*1024**3,memory_total:64*1024**3,
    gpus:[{id:'gpu-a',name:'NVIDIA RTX 5090',utilization:97,memory_used:24.6*1024**3,memory_total:32*1024**3,power:386},
      {id:'gpu-b',name:'第二张 GPU',utilization:11,memory_used:4*1024**3,memory_total:24*1024**3,power:75}] }
  async function launch() {
    application = await electron.launch({args:[root],env:{...process.env,MODLOADER_DATA_DIR:data,MODLOADER_SENSOR_AUTO_START:'0'},timeout:60_000})
    main = await application.firstWindow()
    main.on('pageerror',issue=>errors.push(issue.message))
    await main.route('**/api/state', async route=>{
      const response=await route.fetch(), state=await response.json()
      await route.fulfill({response,json:{...state,instances:instances.map(item=>({...item,active_requests:mode==='idle'?0:1})),
        hardware:{...hardware,sampled_at:Date.now()/1000},settings:{...state.settings,font_scale:scale,close_to_tray:true}}})
    })
    await main.route('**/api/instances/*/telemetry', async route=>{
      reads++
      const id=new URL(route.request().url()).pathname.split('/')[3]
      if(mode==='failed'){ await route.fulfill({status:503,json:{detail:'测试采样失败'}}); return }
      const live={scope:'request',source:'fixture',request_key:mode==='idle'?null:mode==='new'?'next':'current',
        backend_running:mode==='idle'?0:1,backend_queued:0,phase:mode==='new'?'reading':'generating',
        tokens_per_second:mode==='generating'?(id==='a'?42.86:21.43):null,average_tokens_per_second:mode==='generating'?40.31:null,
        average_basis:'backend_decode',first_output_s:mode==='generating'?.68:null,request_elapsed_s:reads,
        prompt_tokens:128,generated_tokens:64,context_tokens:192,context_capacity:32768,window_s:3}
      await route.fulfill({json:{instance_id:id,backend:'llama_cpp',state:'ready',updated_at:Date.now()/1000,unavailable:[],live_generation:live,
        groups:mode==='idle'?[{id:'api_request',title:'完成',request_status:'completed',request_key:'current',metrics:[
          {id:'first_output_s',label:'首输出',value:.68,kind:'gauge'},
          {id:'output_speed',label:'速度',value:40.31,rate_basis:'decode',kind:'gauge'}]}]:[]}})
    })
    await expect(main.locator('.app-shell')).toBeVisible()
    await pinNavigation(main)
    await main.getByRole('navigation').getByRole('button',{name:'运行监控',exact:true}).click()
    await expect(main.locator('[data-resource="current"] strong')).toHaveText('42.86')
  }
  async function openMini() {
    const opened=application.waitForEvent('window')
    await main.getByRole('button',{name:'小窗监控',exact:true}).click()
    mini=await opened
    mini.on('pageerror',issue=>errors.push(issue.message))
    await expect(mini.locator('.mini-frame')).toBeVisible()
  }
  const nativeState = () => application.evaluate(({BrowserWindow})=>{
    const win=BrowserWindow.getAllWindows().find(item=>item.webContents.getURL().includes('window=mini-monitor'))!
    return {bounds:win.getBounds(),pinned:win.isAlwaysOnTop(),opacity:win.getOpacity(),resizable:win.isResizable(),parent:win.getParentWindow()?.id??null}
  })
  const resize = (width:number,height:number) => application.evaluate(({BrowserWindow},{width,height})=>{
    BrowserWindow.getAllWindows().find(item=>item.webContents.getURL().includes('window=mini-monitor'))!.setSize(width,height)
  },{width,height})
  const ids = () => mini.locator('[data-mini-metric]').evaluateAll(nodes=>nodes.map(node=>(node as HTMLElement).dataset.miniMetric))
  const savedIds = () => mini.evaluate(key=>JSON.parse(localStorage.getItem(key)!),MINI_PREFERENCE_KEY)
  const columns = () => mini.locator('[data-mini-metric]').evaluateAll(nodes=>new Set(nodes.map(node=>Math.round(node.getBoundingClientRect().left))).size)
  const capture = async (name:string) => {
    const png=await application.evaluate(async ({BrowserWindow})=>(await BrowserWindow.getAllWindows().find(item=>item.webContents.getURL().includes('window=mini-monitor'))!.webContents.capturePage()).toPNG().toString('base64'))
    fs.writeFileSync(path.join(directory,name),Buffer.from(png,'base64'))
  }
  try {
    await launch(); await openMini()
    await expect(mini.locator('[data-mini-metric="current"] strong')).toHaveText('42.86')
    expect(await nativeState()).toMatchObject({pinned:true,resizable:true,parent:null})
    expect(await mini.evaluate(()=>({management:typeof window.modloader,bridge:typeof window.miniMonitor}))).toEqual({management:'undefined',bridge:'object'})
    expect(await mini.evaluate(()=>performance.getEntriesByType('resource').filter(item=>item.name.includes('/api/')).length)).toBe(0)
    await main.evaluate(()=>window.modloader!.openMiniMonitor())
    expect(application.windows().length).toBe(2)
    await capture('mini-default.png')

    await mini.getByLabel('选择监控实例').selectOption('b')
    await expect(main.getByLabel('选择监控实例')).toHaveValue('b')
    await expect(mini.locator('[data-mini-metric="current"] strong')).toHaveText('21.43')
    await main.getByLabel('选择监控实例').selectOption('a')
    await expect(mini.getByLabel('选择监控实例')).toHaveValue('a')
    await mini.bringToFront()
    const from=await mini.locator('[data-mini-metric="average"]').boundingBox(), target=await mini.locator('[data-mini-metric="first_output"]').boundingBox()
    type Frame = { order: string[]; x: number; animated: boolean }
    await mini.evaluate(()=>{
      const state=window as typeof window & { miniReorderTrace: Promise<Frame[]> }
      state.miniReorderTrace=new Promise(resolve=>{
        const frames:Frame[]=[], started=performance.now()
        const sample=()=>{
          const nodes=[...document.querySelectorAll<HTMLElement>('[data-mini-metric]')]
          frames.push({order:nodes.map(node=>node.dataset.miniMetric!),x:nodes.find(node=>node.dataset.miniMetric==='first_output')!.getBoundingClientRect().left,
            animated:nodes.some(node=>node.getAnimations().some(animation=>animation.id==='reorder-position'))})
          if(performance.now()-started<450) requestAnimationFrame(sample); else resolve(frames)
        }
        requestAnimationFrame(sample)
      })
    })
    await mini.mouse.move(from!.x+30,from!.y+30); await mini.mouse.down()
    await mini.mouse.move(target!.x+10,target!.y+30)
    await expect(mini.locator('.mini-drag-preview')).toBeVisible()
    await expect(mini.locator('[data-mini-metric="average"]')).toHaveClass(/is-dragging/)
    await expect.poll(ids).toEqual(['average','first_output','current'])
    expect(await savedIds()).toEqual(['first_output','current','average'])
    const frames=await mini.evaluate(()=>(window as typeof window & {miniReorderTrace:Promise<Frame[]>}).miniReorderTrace)
    expect(frames.some(frame=>frame.animated)).toBe(true)
    expect(Math.max(...frames.map(frame=>frame.x))-Math.min(...frames.map(frame=>frame.x))).toBeGreaterThan(10)
    expect(frames.filter(frame=>frame.order[0]==='average').every(frame=>frame.order.join(',')==='average,first_output,current')).toBe(true)
    expect(await mini.locator('.mini-drag-card').evaluate(node=>{const box=node.getBoundingClientRect();return box.left>=0&&box.right<=innerWidth&&box.top>=0&&box.bottom<=innerHeight})).toBe(true)
    await capture('mini-drag.png')
    await mini.mouse.up()
    await expect.poll(ids).toEqual(['average','first_output','current'])
    expect(await savedIds()).toEqual(['average','first_output','current'])
    const cancel=await mini.locator('[data-mini-metric="average"]').boundingBox()
    const end=await mini.locator('[data-mini-metric="current"]').boundingBox()
    await mini.mouse.move(cancel!.x+25,cancel!.y+25); await mini.mouse.down(); await mini.mouse.move(end!.x+end!.width-10,end!.y+28)
    await expect.poll(ids).toEqual(['first_output','current','average'])
    expect(await savedIds()).toEqual(['average','first_output','current'])
    await mini.keyboard.press('Escape'); await mini.mouse.up()
    await expect(mini.locator('.mini-drag-preview')).toHaveCount(0)
    expect(await ids()).toEqual(['average','first_output','current'])
    await mini.emulateMedia({reducedMotion:'reduce'})
    const reducedSource=await mini.locator('[data-mini-metric="current"]').boundingBox()
    await mini.mouse.move(reducedSource!.x+25,reducedSource!.y+25); await mini.mouse.down(); await mini.mouse.move(target!.x+10,target!.y+30)
    await expect.poll(ids).toEqual(['current','average','first_output'])
    expect(await mini.locator('[data-mini-metric]').evaluateAll(nodes=>nodes.flatMap(node=>node.getAnimations()).filter(animation=>animation.id==='reorder-position').length)).toBe(0)
    await mini.keyboard.press('Escape'); await mini.mouse.up()
    expect(await ids()).toEqual(['average','first_output','current'])
    expect(await savedIds()).toEqual(['average','first_output','current'])
    await mini.emulateMedia({reducedMotion:'no-preference'})

    await mini.getByRole('button',{name:'选择显示项',exact:true}).click()
    const opacitySlider = mini.getByRole('slider',{name:'窗口不透明度'})
    await opacitySlider.focus(); await opacitySlider.press('End')
    for(let i=0;i<25;i++) await opacitySlider.press('ArrowLeft')
    expect((await nativeState()).opacity).toBe(1)
    for(const label of ['GPU 使用率','显存','GPU 功率']) await mini.getByLabel(label,{exact:true}).check()
    await mini.getByLabel('监控 GPU').selectOption('gpu-b')
    await expect(main.getByLabel('监控 GPU')).toHaveValue('gpu-b')
    await capture('mini-settings.png')
    await mini.getByRole('button',{name:'完成',exact:true}).click()
    await expect.poll(async()=>(await nativeState()).opacity).toBeCloseTo(.75,2)
    await expect(mini.locator('[data-mini-metric="gpu_power"] strong')).toHaveText('75')
    expect((await nativeState()).bounds.height).toBe(140)
    await resize(440,220); await expect.poll(columns).toBe(3)
    await mini.mouse.move(280,20)
    await capture('mini-expanded.png')
    const crossRow=await mini.locator('[data-mini-metric="gpu_power"]').boundingBox()
    await mini.mouse.move(crossRow!.x+25,crossRow!.y+25); await mini.mouse.down(); await mini.mouse.move(target!.x+10,target!.y+30)
    await expect.poll(ids).toEqual(['gpu_power','average','first_output','current','gpu','vram'])
    await capture('mini-cross-row-preview.png')
    await mini.keyboard.press('Escape'); await mini.mouse.up()
    await resize(300,220); await expect.poll(columns).toBe(2)
    await resize(220,220); await expect.poll(columns).toBe(1)
    expect(await ids()).toEqual(['average','first_output','current','gpu','vram','gpu_power'])
    expect(await mini.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true)
    const scrollingSource=await mini.locator('[data-mini-metric="average"]').boundingBox(), scrollArea=await mini.locator('.mini-scroll').boundingBox()
    await mini.mouse.move(scrollingSource!.x+25,scrollingSource!.y+25); await mini.mouse.down()
    await mini.mouse.move(scrollArea!.x+50,scrollArea!.y+scrollArea!.height-8)
    await expect.poll(()=>mini.locator('.mini-scroll').evaluate(node=>node.scrollTop)).toBeGreaterThan(100)
    await expect.poll(async()=>(await ids()).at(-1)).toBe('average')
    expect(await savedIds()).toEqual(['average','first_output','current','gpu','vram','gpu_power'])
    await mini.locator('.mini-grid').dispatchEvent('pointercancel')
    await mini.mouse.up()
    expect(await ids()).toEqual(['average','first_output','current','gpu','vram','gpu_power'])
    await mini.locator('.mini-scroll').evaluate(node=>{node.scrollTop=0})
    await mini.evaluate(()=>{(document.activeElement as HTMLElement)?.blur()})
    await mini.mouse.move(120,16)
    await expect.poll(()=>mini.locator('.mini-scroll').evaluate(node=>getComputedStyle(node).scrollbarColor)).toContain('rgba(0, 0, 0, 0) rgba(0, 0, 0, 0)')
    await mini.mouse.move(180,120); await mini.mouse.wheel(0,180)
    await expect(mini.locator('.mini-scroll')).toHaveClass(/is-scrolling/)
    await mini.mouse.move(120,16)
    await expect(mini.locator('.mini-scroll')).not.toHaveClass(/is-scrolling/)
    await resize(920,140); await expect.poll(columns).toBe(6)
    await resize(440,220)
    scale=1.3
    await expect.poll(()=>mini.locator('.mini-frame').evaluate(node=>getComputedStyle(node).getPropertyValue('--mini-scale').trim())).toBe('1.3')
    await expect.poll(columns).toBe(2)
    await capture('mini-text-scale.png')
    scale=1
    await expect.poll(columns).toBe(3)

    await application.evaluate(({BrowserWindow})=>{
      const main=BrowserWindow.getAllWindows().find(item=>!item.webContents.getURL().includes('window=mini-monitor'))!
      main.minimize()
    })
    const before=reads
    await expect.poll(()=>reads).toBeGreaterThan(before+1)
    await application.evaluate(({BrowserWindow})=>BrowserWindow.getAllWindows().find(item=>!item.webContents.getURL().includes('window=mini-monitor'))!.hide())
    mode='idle'
    await expect(mini.locator('[data-mini-metric="current"] strong')).toHaveText('0.00')
    await expect(mini.locator('[data-mini-metric="first_output"] .mini-basis')).toHaveText('上次')
    mode='new'
    await expect(mini.locator('[data-mini-metric="first_output"] strong')).toHaveText('—')
    mode='failed'
    await expect(mini.getByRole('status')).toHaveText('采样失败')
    await expect(mini.locator('[data-mini-metric="current"] strong')).toHaveText('—')
    mode='generating'
    await expect(mini.locator('[data-mini-metric="current"] strong')).toHaveText('42.86')
    await mini.getByRole('button',{name:'取消置顶',exact:true}).click()
    expect((await nativeState()).pinned).toBe(false)
    await mini.getByRole('button',{name:'关闭小窗',exact:true}).click()
    await expect.poll(()=>application.windows().length).toBe(1)
    const afterClose=reads
    await expect.poll(()=>reads).toBeGreaterThan(afterClose)
    await application.close()

    await launch(); await openMini()
    expect(await ids()).toEqual(['average','first_output','current','gpu','vram','gpu_power'])
    expect(await nativeState()).toMatchObject({pinned:false,bounds:{width:440,height:220}})
    expect((await nativeState()).opacity).toBeCloseTo(.75,2)
    await mini.getByRole('button',{name:'选择显示项',exact:true}).click()
    await mini.getByRole('slider',{name:'窗口不透明度'}).press('Home')
    await mini.getByRole('button',{name:'完成',exact:true}).click()
    await expect.poll(async()=>(await nativeState()).opacity).toBe(0)
    await main.evaluate(()=>window.modloader!.openMiniMonitor())
    await expect(mini.getByRole('slider',{name:'窗口不透明度'})).toBeVisible()
    expect((await nativeState()).opacity).toBe(1)
    await expect(mini.getByRole('slider',{name:'窗口不透明度'})).toHaveValue('0')
    await mini.getByRole('slider',{name:'窗口不透明度'}).press('End')
    await mini.getByRole('button',{name:'完成',exact:true}).click()
    await mini.getByRole('button',{name:'选择显示项',exact:true}).click()
    await expect(mini.getByRole('checkbox').first()).toBeVisible()
    for(const input of await mini.getByRole('checkbox').all()) if(await input.isChecked()) await input.uncheck()
    await mini.getByRole('button',{name:'完成',exact:true}).click()
    await expect(mini.locator('[data-mini-metric]')).toHaveCount(0)
    await expect(mini.getByRole('button',{name:'选择显示项',exact:true}).last()).toBeVisible()
    expect(errors).toEqual([])
    console.log(JSON.stringify({screenshots:directory,telemetryReads:reads,pageErrors:errors}))
  } finally { await application!?.close().catch(()=>{}) }
})
