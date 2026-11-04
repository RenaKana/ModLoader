import type { AppState } from './types'
import type { MonitorState } from './useMonitorState'
import { generationView, type MonitorRequestView } from './monitor-generation'
import { projectGenerationRequest } from './monitor-generation-readouts'
import { requestEngineRate, requestFirstOutput, requestRate, requestTokens } from './chat-turn-metrics'
import { finite, gpuKey, gpuLabel, numberText, rateText } from './resource-metrics'
import { useEffect, useRef } from 'react'
import type { GenerationEventIdentity, GenerationEventStore } from './monitor-generation-events'

export const miniMetrics = [
  { id: 'first_output', label: '首字延迟', group: 'generation' },
  { id: 'current', label: '当前速度', group: 'generation' },
  { id: 'average', label: '平均速度', group: 'generation' },
  { id: 'prompt', label: '输入 Token', group: 'generation' },
  { id: 'generated', label: '输出 Token', group: 'generation' },
  { id: 'context', label: '上下文', group: 'generation' },
  { id: 'duration', label: '请求耗时', group: 'generation' },
  { id: 'running', label: '运行请求', group: 'generation' },
  { id: 'queued', label: '排队请求', group: 'generation' },
  { id: 'cpu', label: 'CPU 使用率', group: 'hardware' },
  { id: 'gpu', label: 'GPU 使用率', group: 'hardware' },
  { id: 'memory', label: '系统内存', group: 'hardware' },
  { id: 'vram', label: '显存', group: 'hardware' },
  { id: 'cpu_temperature', label: 'CPU 温度', group: 'hardware' },
  { id: 'gpu_temperature', label: 'GPU 温度', group: 'hardware' },
  { id: 'cpu_power', label: 'CPU 功率', group: 'hardware' },
  { id: 'gpu_power', label: 'GPU 功率', group: 'hardware' },
  { id: 'disk', label: '磁盘读取', group: 'hardware' },
  { id: 'pcie_rx', label: 'PCIe RX', group: 'hardware' },
  { id: 'pcie_tx', label: 'PCIe TX', group: 'hardware' },
] as const
export type MiniMetricId = typeof miniMetrics[number]['id']
export type MiniReadout = { value: string; unit: string; basis?: string }
export type MiniMonitorSnapshot = {
  instanceId: string; instances: Array<{ id: string; label: string; state: string }>
  gpuId: string; gpus: Array<{ id: string; label: string }>
  stateLabel: string; tone: 'active' | 'idle' | 'error' | 'waiting'; fontScale: number
  metrics: Record<MiniMetricId, MiniReadout>
}
export type MiniMonitorAction = { type: 'instance' | 'gpu'; id: string }
export interface MiniMonitorBridge {
  ready(): Promise<{ snapshot: MiniMonitorSnapshot | null; alwaysOnTop: boolean; opacity: number; editing: boolean }>
  onSnapshot(handler: (snapshot: MiniMonitorSnapshot | null) => void): () => void
  onAlwaysOnTop(handler: (value: boolean) => void): () => void
  onOpacity(handler: (value: number) => void): () => void
  onEditing(handler: (value: boolean) => void): () => void
  selectInstance(id: string): void
  selectGpu(id: string): void
  setAlwaysOnTop(value: boolean): Promise<boolean>
  setOpacity(value: number): Promise<number>
  setEditing(value: boolean): Promise<void>
  fitContent(height: number): Promise<void>
  reportContentHeight(height: number): void
  close(): void
}

// Native visibility gates only the imperative fast channel. Normal snapshots
// keep their existing cadence even when both windows are hidden.
export function useMiniMonitorPublisher(snapshot: MiniMonitorSnapshot, events: GenerationEventStore | undefined, identity: GenerationEventIdentity | undefined, request?: MonitorRequestView) {
  const latest = useRef({ snapshot, identity, request }), publishNormal = useRef<(() => void) | undefined>(undefined)
  latest.current = { snapshot, identity, request }
  useEffect(() => {
    const publish = (fast = false) => {
      const { snapshot, identity, request } = latest.current
      const value = events?.readLatest(request, identity, 'decode'), rate = value?.rate, accounting = value?.accounting
      // A next request may arrive before React consumes its normal snapshot.
      // Leave the already-published final result intact rather than replaying an old poll.
      if (fast && value?.event && rate === undefined && !accounting
        && value.event.live_generation?.request_key !== identity?.requestKey) return
      const metrics = { ...snapshot.metrics }
      if (rate !== undefined) metrics.current = { ...metrics.current, value: rate === null ? '—' : numberText(rate, 2, true) }
      if (request && accounting) {
        const native = projectGenerationRequest(request, accounting), tokens = requestTokens(native), mean = requestRate(native, 'decode').value
        const engine = mean === undefined ? requestEngineRate(native).value : undefined
        const ttft = requestFirstOutput(native), basis = native.status !== 'running' ? '上次' : undefined
        metrics.average = numeric(mean ?? engine, 'token/s', 2, true, engine !== undefined ? '引擎平均' : basis)
        metrics.prompt = numeric(tokens.input, 'token', 0, false, basis)
        metrics.generated = numeric(tokens.output, 'token', 0, false, basis)
        const used = tokens.input !== undefined && tokens.output !== undefined ? numberText(tokens.input + tokens.output, 0) : '—'
        metrics.context = { ...metrics.context, value: `${used} / ${snapshot.metrics.context.value.split(' / ')[1]}`, basis }
        if (native.status !== 'running') metrics.current = { ...metrics.current, value: '—' }
        if (ttft.value !== undefined) metrics.first_output = numeric(ttft.value, 's', 2, false, ttft.basis === '引擎' ? '引擎' : basis)
      }
      window.modloader?.publishMiniMonitor?.({ ...snapshot, metrics })
    }
    publishNormal.current = publish
    let active = false, disposed = false, observedVisibility = false, unsubscribeFast: (() => void) | undefined
    const setVisible = (visible: boolean) => {
      if (disposed || active === visible) return
      active = visible; unsubscribeFast?.(); unsubscribeFast = undefined
      if (visible) { unsubscribeFast = events?.subscribeImmediate(() => publish(true)); publish(true) }
    }
    const unsubscribeVisibility = window.modloader?.onMonitorVisibility?.(visible => { observedVisibility = true; setVisible(visible) })
    if (window.modloader?.getMonitorVisibility) {
      void window.modloader.getMonitorVisibility().then(visible => { if (!observedVisibility) setVisible(visible) })
        .catch(error => console.error('读取监控窗口可见性失败：', error))
    } else setVisible(true)
    return () => { disposed = true; unsubscribeFast?.(); unsubscribeVisibility?.(); publishNormal.current = undefined }
  }, [events])
  useEffect(() => { publishNormal.current?.() }, [snapshot, events, identity?.instanceId, identity?.runId, identity?.requestKey, request])
}
export const defaultMiniMetrics: MiniMetricId[] = ['first_output', 'current', 'average']
export const MINI_PREFERENCE_KEY = 'modloader.monitor.mini.metrics'
export function readMiniSelection(value: unknown): MiniMetricId[] {
  if (!Array.isArray(value)) return [...defaultMiniMetrics]
  return [...new Set(value.filter((id): id is MiniMetricId => miniMetrics.some(metric => metric.id === id)))]
}
export function moveMiniMetric(ids: MiniMetricId[], id: MiniMetricId, before: MiniMetricId | null) {
  if (id === before || !ids.includes(id)) return ids
  const next = ids.filter(item => item !== id)
  const index = before === null ? next.length : next.indexOf(before)
  if (index < 0) return ids
  next.splice(index, 0, id)
  return next
}
const numeric = (value: unknown, unit: string, digits = 2, fixed = false, basis?: string): MiniReadout => ({
  value: finite(value) === null ? '—' : numberText(value, digits, fixed), unit, basis,
})
function memoryPair(used: unknown, total: unknown): MiniReadout {
  const divisor = 1024 ** 3
  return { value: `${finite(used) === null ? '—' : numberText(Number(used) / divisor, 1)} / ${finite(total) === null ? '—' : numberText(Number(total) / divisor, 1)}`, unit: 'GB' }
}
function compactBasis(note: string) {
  if (note.includes('上次') || note.includes('最近请求')) return note.includes('采样') ? '上次采样' : '上次'
  if (note.includes('已采样区间')) return '采样均值'
  if (note.includes('槽位')) return '槽位'
  return undefined
}

// Only presentation data crosses to the mini renderer; never AppState/settings/connection.
export function miniSnapshot(state: AppState, monitor: MonitorState, connected: boolean): MiniMonitorSnapshot {
  const { instance, instanceId, ready, payload, sampleError, memory, gpu, gpus } = monitor
  const failed = !connected || !!sampleError
  const view = generationView(payload, memory, ready, instance?.active_requests ?? 0, failed)
  const live = ready && !failed ? payload?.live_generation : null
  const generation = ready && !failed && !monitor.generation?.historical ? monitor.generation : undefined
  const request = generation?.requests.at(-1)
  const tokens = request ? requestTokens(request) : undefined
  const used = request ? tokens?.input !== undefined && tokens.output !== undefined ? tokens.input + tokens.output : null : view.used
  const nativeRate = request ? requestRate(request, 'decode').value : undefined
  const engineRate = request ? requestEngineRate(request).value : undefined
  const currentRequest = request ? request.status === 'running' : !!live?.request_key
  // A new/ambiguous request must not inherit the previous request's timing or counts.
  const last = !currentRequest && (instance?.active_requests ?? 0) === 0 && !live?.backend_running && !live?.backend_queued ? memory?.lastRequest : undefined
  const requestBasis = !currentRequest && (request || last) ? '上次' : undefined
  const hardware = connected && !state.hardware.error ? state.hardware : {}
  const device = connected && !state.hardware.gpu_error ? gpu : undefined
  const rate = (value: unknown, divisor: number, unit: string): MiniReadout => ({ value: finite(value) === null ? '—' : rateText(Number(value) / divisor), unit })
  const selectedInstances = monitor.instances.some(item => item.id === instanceId) || !instance ? monitor.instances : [...monitor.instances, instance]
  const stateLabel = !connected ? '服务离线' : !instance ? '未选择实例' : !ready ? ['stopped', 'failed'].includes(instance.state) ? '实例已停止' : '等待就绪'
    : failed ? '采样失败' : view.idle ? '空闲' : live?.phase === 'thinking' ? '思考中' : live?.phase === 'output' ? '输出中'
      : live?.phase === 'generating' ? '生成中' : live?.phase === 'prefill' ? 'Prefill' : live?.phase === 'queued' ? '排队中' : '等待采样'
  return { instanceId, instances: selectedInstances.map(item => ({ id: item.id, label: item.alias, state: item.state })),
    gpuId: gpuKey(gpu), gpus: gpus.map(g => ({ id: gpuKey(g), label: gpuLabel(g) })),
    stateLabel, tone: !connected || failed ? 'error' : view.idle ? 'idle' : ready && live?.phase ? 'active' : 'waiting',
    fontScale: Number(state.settings.font_scale) || 1,
    metrics: {
      first_output: numeric(request ? requestFirstOutput(request).value : currentRequest ? live?.first_output_s : last?.firstOutput, 's', 2, false,
        request && requestFirstOutput(request).basis === '引擎' ? '引擎' : requestBasis),
      current: numeric(request?.status === 'running' ? request.phases.decode.current : view.current, 'token/s', 2, true),
      average: numeric(request ? nativeRate ?? engineRate ?? request.phases.decode.mean : view.mean, 'token/s', 2, true,
        nativeRate === undefined && engineRate !== undefined ? '引擎平均' : request ? compactBasis(request.phases.decode.meanNote) ?? requestBasis : compactBasis(view.meanNote)),
      prompt: numeric(request ? tokens?.input : currentRequest ? live?.prompt_tokens : last?.prompt, 'token', 0, false, requestBasis),
      generated: numeric(request ? tokens?.output : currentRequest ? live?.generated_tokens : last?.generated, 'token', 0, false, requestBasis),
      context: { value: `${finite(used) === null ? '—' : numberText(used, 0)} / ${finite(view.capacity) === null ? '—' : numberText(view.capacity, 0)}`, unit: 'token', basis: request ? requestBasis : compactBasis(view.contextBasis) },
      duration: numeric(request ? request.elapsed : currentRequest ? live?.request_elapsed_s : last?.duration, 's', 2, false, requestBasis),
      running: numeric(live?.backend_running, '个', 0), queued: numeric(live?.backend_queued, '个', 0),
      cpu: numeric(hardware.cpu_percent, '%'), gpu: numeric(device?.utilization, '%'),
      memory: memoryPair(hardware.memory_used, hardware.memory_total ?? hardware.ram_total_bytes),
      vram: memoryPair(device?.memory_used, device?.memory_total),
      cpu_temperature: numeric(hardware.cpu_temperature_c, '°C'), gpu_temperature: numeric(device?.temperature, '°C'),
      cpu_power: numeric(hardware.cpu_power_w, 'W'), gpu_power: numeric(device?.power, 'W'),
      disk: rate(hardware.disk_read_bytes_per_second, 1e6, 'MB/s'),
      pcie_rx: rate(device?.pcie_rx_bytes_s, 1e9, 'GB/s'), pcie_tx: rate(device?.pcie_tx_bytes_s, 1e9, 'GB/s'),
    } }
}
