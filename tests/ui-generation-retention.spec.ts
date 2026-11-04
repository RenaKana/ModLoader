import { test, expect } from '@playwright/test'
import type { RuntimeInstance, Telemetry } from '../src/types'
import { backendContext, generationView, monitorInstanceId, rememberGeneration } from '../src/monitor-generation'
import { adaptiveBounds, appendHistory, seriesSegments } from '../src/resource-metrics'

test('唯一可用实例自动跟随，多实例保留手选；加载、停止与移除不沿用旧实例', () => {
  const instance = (id: string, state: string) => ({ id, state } as RuntimeInstance)
  expect(monitorInstanceId([instance('old','stopped'),instance('new','ready')], 'old')).toBe('new')
  expect(monitorInstanceId([instance('old','stopped'),instance('new','loading')], 'old')).toBe('new')
  expect(monitorInstanceId([instance('a','ready'),instance('b','ready')], 'b')).toBe('b')
  expect(monitorInstanceId([instance('a','running'),instance('b','loading')], 'b')).toBe('a')
  expect(monitorInstanceId([instance('a','loading'),instance('b','starting')], 'b')).toBe('b')
  expect(monitorInstanceId([instance('a','stopped')], 'a')).toBe('a')
  expect(monitorInstanceId([], 'a')).toBe('')
  const loading = [instance('a', 'ready'), instance('b', 'preparing')]
  expect(monitorInstanceId(loading, 'b')).toBe('a')
  expect(monitorInstanceId(loading, 'b', 'b')).toBe('b')
  expect(monitorInstanceId([loading[0], instance('b', 'starting')], 'b', 'b')).toBe('b')
  expect(monitorInstanceId([loading[0], instance('b', 'stopped')], 'b', 'b')).toBe('a')
  expect(monitorInstanceId([loading[0]], 'b', 'b')).toBe('a')
})

test('功率与平均速度轴按近期有效数据留白，恒定小波动不夸大或频繁跳轴', () => {
  const constant = adaptiveBounds([null,100,100.01],20,1000)
  expect(constant.minimum).toBeGreaterThan(0)
  expect(constant.minimum).toBeLessThan(100)
  expect(constant.maximum).toBeGreaterThan(100.01)
  expect(constant.maximum-constant.minimum).toBeGreaterThanOrEqual(20)
  expect(adaptiveBounds([99.99,100.03],20,2000,constant)).toBe(constant)
  const spike = adaptiveBounds([100,500],20,3000,constant)
  expect(spike.maximum).toBeGreaterThan(500)
  expect(adaptiveBounds([100,100.01],20,4000,spike)).toBe(spike)
  const recovered = adaptiveBounds([100,100.01],20,14000,spike)
  expect(recovered.maximum-recovered.minimum).toBeLessThan(spike.maximum-spike.minimum)
  expect(adaptiveBounds([0,0,null],20,1000)).toMatchObject({minimum:0})
  expect(adaptiveBounds([0,0,null],20,1000).maximum).toBeGreaterThanOrEqual(20)
  expect(adaptiveBounds([40,40.01],5,1000).maximum-adaptiveBounds([40,40.01],5,1000).minimum).toBeGreaterThanOrEqual(8)
  expect(adaptiveBounds([null],20,2000)).toMatchObject({minimum:0,maximum:20})
})

function payload(live: Record<string, unknown> = {}, groups: Telemetry['groups'] = []): Telemetry {
  return { instance_id: 'one', backend: 'strata', state: 'ready', updated_at: 100, unavailable: [], groups,
    live_generation: { request_key: null, scope: 'model', source: 'fixture', backend_running: 0, backend_queued: 0,
      tokens_per_second: null, window_s: null, available_for_chat: false, reason: null, ...live } }
}
const capacity: Telemetry['groups'][number] = {id:'strata_context',title:'上下文',metrics:[{id:'max_context',label:'容量',value:4096,kind:'gauge'}]}
function completed(basis: 'decode' | 'end_to_end' = 'decode', mean = 40): Telemetry['groups'][number] {
  return { id:'api_request',title:'最近请求',request_key:'first',request_status:'completed',metrics:[
    {id:'output_speed',label:'速度',value:mean,kind:'gauge',rate_basis:basis},
    {id:'prompt_tokens',label:'输入',value:128,kind:'gauge'}, {id:'completion_tokens',label:'输出',value:32,kind:'gauge'}] }
}

test('生成结束后保留最终解码均值与请求长度，容量独立于 live 字段', () => {
  const running = payload({request_key:'first',backend_running:1,average_tokens_per_second:41,context_tokens:144,context_capacity:4096})
  const before = rememberGeneration(undefined,running)
  const idle = payload({},[capacity,completed()])
  const view = generationView(idle,rememberGeneration(before,idle),true,0)
  expect(view.current).toBe(0)
  expect(view.mean).toBe(40) // Final response, not last observed 41.
  expect(view.meanNote).toContain('上次完成请求')
  expect(view.used).toBe(160)
  expect(view.contextBasis).toBe('上次完成请求长度')
  expect(view.capacity).toBe(4096)
  expect(view.chart).toEqual({current:0,average:null,context:null})
})

test('端到端速度不冒充解码均值；只有末次采样时标明来源', () => {
  const idle=payload({},[capacity,completed('end_to_end',9)])
  expect(generationView(idle,rememberGeneration(undefined,idle),true,0).mean).toBeNull()
  const sampled=rememberGeneration(undefined,payload({request_key:'first',backend_running:1,average_tokens_per_second:41}))
  const retained=rememberGeneration(sampled,idle)
  const view=generationView(idle,retained,true,0)
  expect(view.mean).toBe(41)
  expect(view.meanNote).toContain('末次采样')
  const next=payload({request_key:'second',backend_running:1})
  const nextView=generationView(next,rememberGeneration(retained,next),true,1)
  expect(nextView.mean).toBeNull()
  expect(nextView.used).toBeNull()
  expect(nextView.comparison).toBe('上次采样 41.00 token/s')
})

test('采样区间均值与原生解码均值标注各自口径', () => {
  const sampled=payload({request_key:'first',backend_running:1,average_tokens_per_second:23.5,average_basis:'sampled_decode'})
  const sampledView=generationView(sampled,rememberGeneration(undefined,sampled),true,1)
  expect(sampledView.mean).toBe(23.5)
  expect(sampledView.meanNote).toBe('当前请求 · 已采样区间均值')
  expect(generationView(payload(), rememberGeneration(undefined,sampled),true,0).meanNote).toBe('上次采样 · 已采样区间均值')
  const native=payload({request_key:'first',backend_running:1,average_tokens_per_second:24,average_basis:'backend_decode'})
  expect(generationView(native,rememberGeneration(undefined,native),true,1).meanNote).toBe('当前请求 · 解码均值')
})

test('NInfer 精确 SSE 实时计数可显示，空闲后容量保留且历史长度不冒充 KV', () => {
  const running = { ...payload({scope:'request',request_key:'first',backend_running:1,tokens_per_second:180,
    window_s:.5,average_tokens_per_second:192,average_basis:'backend_decode',prompt_tokens:28,generated_tokens:80,
    context_tokens:108,context_capacity:128000,context_basis:'输入 + 已生成'}), backend:'ninfer' }
  const memory = rememberGeneration(undefined,running)
  const liveView = generationView(running,memory,true,1)
  expect(liveView.current).toBe(180)
  expect(liveView.currentNote).toBe('当前请求 · 近 0.5 秒滚动')
  expect(liveView.mean).toBe(192)
  expect(liveView.used).toBe(108)
  expect(liveView.capacity).toBe(128000)
  expect(liveView.contextBasis).toBe('当前请求 · 输入 + 已生成')
  expect(liveView.chart).toEqual({current:180,average:192,context:108})
  const idle = {...payload({context_capacity:128000},[completed('decode',191)]),backend:'ninfer'}
  const retained = generationView(idle,rememberGeneration(memory,idle),true,0)
  expect(retained.current).toBe(0)
  expect(retained.capacity).toBe(128000)
  expect(retained.meanNote).toContain('上次完成请求')
  expect(retained.contextBasis).toBe('上次完成请求长度')
  expect(retained.chart.context).toBeNull()
})

test('启动配置上下文上限保留来源，不冒充后端实际 KV 容量', () => {
  const running = {...payload({scope:'request',request_key:'first',backend_running:1,context_tokens:108,
    context_capacity:2048,context_capacity_basis:'启动配置上限'}),backend:'vllm'}
  const memory = rememberGeneration(undefined,running)
  expect(generationView(running,memory,true,1)).toMatchObject({capacity:2048,capacityBasis:'启动配置上限',used:108})
  const unavailable = generationView(null,memory,false,0)
  expect(unavailable.capacity).toBe(2048)
  expect(unavailable.capacityBasis).toBe('启动配置上限')
  expect(unavailable.chart.context).toBeNull()
})

test('空闲、排队、失败、未就绪不能混为零速；实际零值保留', () => {
  const idle=payload({},[completed('decode',0)])
  const memory=rememberGeneration(undefined,idle)
  expect(generationView(idle,memory,true,0).mean).toBe(0)
  expect(generationView(idle,memory,true,0,true).current).toBeNull()
  expect(generationView(idle,memory,false,0).current).toBeNull()
  expect(generationView(payload({backend_queued:1}),memory,true,0).current).toBeNull()
  expect(generationView(payload({backend_paused:1}),memory,true,0).current).toBeNull()
  expect(generationView(payload({backend_paused:null}),memory,true,0).current).toBeNull()
  expect(generationView(idle,memory,true,1).current).toBeNull()
  expect(generationView(payload(),undefined,true,0).mean).toBeNull() // Different instance.
})

test('空闲槽位计数持续可读，清零为 0；多槽位不任取第一项', () => {
  const slot=(id:number,used:number,total:number|null): Telemetry['groups'][number] => ({id:`llama_slot_${id}`,title:'槽位',metrics:[
    {id:'is_processing',label:'正在处理',value:'否',kind:'text'},
    {id:'n_prompt_tokens',label:'已持有',value:used,kind:'gauge'},{id:'n_ctx',label:'容量',value:total,kind:'gauge'}]})
  const idle={...payload({},[slot(0,0,8192)]),backend:'llama_cpp'}
  const memory=rememberGeneration(undefined,idle)
  const view=generationView(idle,memory,true,0)
  expect(view.used).toBe(0);expect(view.capacity).toBe(8192)
  expect(view.contextBasis).toBe('槽位 0 当前已用')
  expect(view.chart.context).toBe(0)
  expect(backendContext({...idle,groups:[slot(0,900,8192),slot(1,500,null)]})).toMatchObject({used:null,capacity:null})
})

test('请求结束与下一次请求之间曲线保留但不连接，也不复制历史均值', () => {
  let history=appendHistory([],{time:1000,segment:'first',values:{average:40}})
  history=appendHistory(history,{time:2000,segment:'first',values:{average:41}})
  history=appendHistory(history,{time:3000,segment:'idle',values:{average:null}})
  history=appendHistory(history,{time:4000,segment:'second',values:{average:70}})
  history=appendHistory(history,{time:5000,segment:'second',values:{average:71}})
  expect(seriesSegments(history,'average').map(s=>s.map(p=>p.value))).toEqual([[40,41],[70,71]])
})
