import type { ChatGenerationActivity, ExternalHistorySnapshot, ExternalRequestRecord, LiveGeneration, RuntimeInstance, Telemetry, TelemetryGroup } from './types'
import { isActive, isReady } from './api'
import { HISTORY_MS, finite, numberText, seriesSegments, type HistoryPoint } from './resource-metrics'
import { requestRate, requestTokens, turnTotals, type ModelRequestRecord, type StreamDecodeSample, type TurnTotals } from './chat-turn-metrics'
import { reportedNumber, type Usage } from './chat-response'
import { retainChartWindow } from './monitor-chart-window'

type Snapshot = { key: string; mean: number | null; meanBasis: LiveGeneration['average_basis']; used: number | null; meanFinal: boolean; usedFinal: boolean }
export type GenerationMemory = { capacity: number | null; capacityBasis?: string; sample: Snapshot | null; completed: Snapshot | null
  lastRequest?: { key: string; firstOutput: number | null; duration: number | null; prompt: number | null; generated: number | null }
  session?: GenerationSession; sessions?: GenerationSession[]; streamDecode?: StreamDecodeHistory[]; externalHistory?: ExternalHistorySnapshot; gatewayRunId?: string }
type StreamDecodeHistory = StreamDecodeSample & { messageId: string; ended: boolean; points: GenerationPoint[]; boundaryPoint?: GenerationPoint; startedAt: number }
export type GenerationPhase = 'prefill' | 'decode'
export type RequestReadout = { key: string; firstOutput: number | null; duration: number | null; prompt: number | null; generated: number | null; usage?: Usage }
export type GenerationPoint = HistoryPoint & { wallTime: number }
type PhaseSamples = { points: GenerationPoint[]; boundaryPoint?: GenerationPoint; startedAt: number | null; clockOffset: number | null; last: number | null; mean: number | null; meanBasis: string | null; window: number | null }
type GenerationSession = {
  key: string; phase: GenerationPhase | null; status: string; active: boolean
  observedAt?: number
  clientRequestId?: string; clientCallId?: string; lastPhase?: GenerationPhase | null
  rates: Record<GenerationPhase, PhaseSamples>; request: RequestReadout; used: number | null
  priorTerminalKey?: string
  sampleSegment?: number
}
export type MonitorRequestView = ModelRequestRecord & { phases: Record<GenerationPhase, GenerationPhaseView>; decodeMean: number | null; prefillMean: number | null; liveMetrics: boolean; external?: ExternalRequestRecord }
export type GenerationPhaseView = {
  phase: GenerationPhase; current: number | null; mean: number | null; currentNote: string; meanNote: string; ended: boolean
  points: GenerationPoint[]; boundaryPoint?: GenerationPoint; historyStartedAt: number | null; historyKey: string
}
export type MonitorGenerationView = {
  key: string | null; phase: GenerationPhase | null; current: number | null; mean: number | null
  currentNote: string; meanNote: string; status: string; active: boolean; working: boolean; ended: boolean
  sessionKey: string; phases: Record<GenerationPhase, GenerationPhaseView>
  points: GenerationPoint[]; historyStartedAt: number | null; historyKey: string; used: number | null; capacity: number | null; contextBasis: string
  request: RequestReadout | null
  requests: MonitorRequestView[]; turn?: TurnTotals; toolCount?: number; historical: boolean
  tools: NonNullable<ChatGenerationActivity['toolActivity']>
  aggregation: 'turn' | 'request'; origin: 'internal' | 'external' | 'backend'; waiting?: boolean
}
// Real-time series belong to one model request. Tool waits are never compressed
// into a fabricated continuous decode trace.
export function generationSegments(points: GenerationPoint[]) {
  return seriesSegments(points, 'current')
}

const metric = (group: TelemetryGroup | undefined, id: string) => group?.metrics.find(item => item.id === id)
const value = (group: TelemetryGroup | undefined, id: string) => finite(metric(group, id)?.value)
const capacityValue = (value: unknown) => { const number = finite(value); return number !== null && number > 0 ? number : null }

export function monitorInstanceId(instances: RuntimeInstance[], current: string, explicit = '') {
  if (instances.some(instance => instance.id === explicit && isActive(instance.state))) return explicit
  const available = instances.filter(instance => isReady(instance.state))
  if (available.length === 1) return available[0].id
  const active = instances.filter(instance => isActive(instance.state))
  if (!available.length && active.length === 1) return active[0].id
  if (instances.some(instance => instance.id === current)) return current
  return available[0]?.id ?? active[0]?.id ?? ''
}

// These are live slot counters, not the configured KV resident capacity or
// conversation_cache.last_prompt (which is only a historical request field).
export function backendContext(payload: Telemetry | null) {
  const groups = payload?.groups ?? []
  if (payload?.backend === 'strata') {
    const slots = groups.filter(g => g.id.startsWith('strata_slot_'))
    const slot = slots.length === 1 ? slots[0] : undefined
    return { capacity: value(groups.find(g => g.id === 'strata_context'), 'max_context'),
      used: payload.live_generation?.backend_running === 0 ? value(slot, 'held_tokens') : null,
      basis: slot ? `槽位 ${slot.id.slice('strata_slot_'.length)} 当前保留` : '' }
  }
  const slots = groups.filter(g => g.id.startsWith('llama_slot_'))
  const busy = slots.filter(g => metric(g, 'is_processing')?.value === '是')
  const slot = slots.length === 1 ? slots[0] : busy.length === 1 ? busy[0] : undefined
  const capacities = [...new Set(slots.map(g => value(g, 'n_ctx')).filter(n => n !== null))]
  return { capacity: slot ? value(slot, 'n_ctx') : capacities.length === 1 && slots.every(g => value(g, 'n_ctx') === capacities[0]) ? capacities[0] : null,
    used: value(slot, 'n_prompt_tokens'), basis: slot ? `槽位 ${slot.id.slice('llama_slot_'.length)} 当前已用` : '' }
}

// Capacity belongs to the loaded instance even when no request can be matched.
// A backend's automatic/zero setting is not a reported token capacity.
export function reportedContextCapacity(payload: Telemetry | null) {
  return capacityValue(payload?.live_generation?.context_capacity) ?? capacityValue(backendContext(payload).capacity)
}

/** A turn id alone is not enough: tool loops contain several distinct calls. */
export function matchingRequestLive(live: LiveGeneration | null | undefined, request: ModelRequestRecord | undefined) {
  if (!live || !request) return null
  return (live.client_call_id ? live.client_call_id === request.id : !!request.requestKey && live.request_key === request.requestKey) ? live : null
}

/** Only an unambiguous, still-running call can consume live backend counters. */
export function mergeRequestLive(request: ModelRequestRecord, sample: LiveGeneration, external = false): ModelRequestRecord {
  const live = external ? request.requestKey && sample.request_key === request.requestKey ? sample : null : matchingRequestLive(sample, request)
  if (!live || live.scope !== 'request' || request.status !== 'running') return request
  const { source, prompt_tokens, generated_tokens, average_tokens_per_second, average_basis,
    average_prefill_tokens_per_second, prefill_average_basis, prefill_processed_tokens, prefill_cached_tokens } = live
  return { ...request, ...(live.request_key ? { requestKey: live.request_key } : {}),
    ...(reportedNumber(live.request_elapsed_s) === undefined ? {} : { elapsed: live.request_elapsed_s! }),
    ...(reportedNumber(live.first_output_s) === undefined ? {} : { firstOutput: live.first_output_s!, firstOutputBasis: 'gateway' }),
    live: { source, prompt_tokens, generated_tokens, average_tokens_per_second, average_basis,
      average_prefill_tokens_per_second, prefill_average_basis, prefill_processed_tokens, prefill_cached_tokens } }
}

export function chatRequestLive(live: LiveGeneration | null | undefined, request: ModelRequestRecord | undefined, stream?: StreamDecodeSample) {
  const matched = matchingRequestLive(live, request)
  if (!request) return matched
  const tokens = requestTokens(request), sampled = stream?.requestId === request.id && stream.tokensPerSecond !== null ? stream : undefined
  if (!sampled && tokens.input === undefined && tokens.output === undefined) return matched
  const prompt = tokens.input ?? matched?.prompt_tokens ?? null, output = tokens.output ?? matched?.generated_tokens ?? null
  return { ...matched, client_call_id: request.id, request_key: request.requestKey ?? request.id,
    tokens_per_second: sampled?.tokensPerSecond ?? matched?.tokens_per_second ?? null,
    window_s: sampled?.windowSeconds ?? matched?.window_s ?? null,
    available_for_chat: !!sampled || matched?.available_for_chat === true,
    scope: 'request' as const, source: sampled ? sampled.clock === 'arrival' ? 'SSE 累计 Token / 接收时间增量' : 'SSE 累计 Token / 后端解码计时增量' : matched?.source ?? 'SSE usage/timings',
    backend_running: matched?.backend_running ?? null, backend_queued: matched?.backend_queued ?? null,
    reason: sampled ? null : matched?.reason ?? null, prompt_tokens: prompt, generated_tokens: output,
    context_tokens: prompt !== null && output !== null ? prompt + output : matched?.context_tokens ?? null,
    context_basis: prompt !== null && output !== null ? '输入 + 已生成' : matched?.context_basis }
}

/** Retain event-driven request curves separately from raw usage and telemetry. */
export function rememberChatStream(previous: GenerationMemory | undefined, activity: ChatGenerationActivity): GenerationMemory {
  const memory = previous ?? { capacity: null, sample: null, completed: null }
  const histories = (memory.streamDecode ?? []).map(item => {
    const record = item.messageId === activity.messageId ? activity.requests?.find(request => request.id === item.requestId) : undefined
    return record ? { ...item, ended: record.status !== 'running' } : item
  })
  const sample = activity.stream, request = activity.requests?.find(item => item.id === sample?.requestId)
  if (sample && request) {
    const index = histories.findIndex(item => item.messageId === activity.messageId && item.requestId === sample.requestId)
    const old = histories[index], time = sample.sampledAt
    const retained = old?.sampledAt === time && old.tokensPerSecond === sample.tokensPerSecond ? { points: old.points, boundaryPoint: old.boundaryPoint }
      : retainChartWindow([...(old?.boundaryPoint ? [old.boundaryPoint] : []), ...(old?.points ?? []).filter(point => point.time < time),
        { time, wallTime: time, values: { current: sample.tokensPerSecond }, segment: `${request.id}:decode` }], time - HISTORY_MS)
    const history = { ...sample, messageId: activity.messageId, ended: request.status !== 'running', ...retained, startedAt: old?.startedAt ?? time }
    if (index < 0) histories.push(history)
    else histories[index] = history
  }
  return { ...memory, streamDecode: histories }
}

export function rememberGeneration(previous: GenerationMemory | undefined, payload: Telemetry): GenerationMemory {
  const gatewayRunId = payload.external_history?.run_id ?? previous?.gatewayRunId
  if (previous?.gatewayRunId && gatewayRunId !== previous.gatewayRunId) previous = undefined
  const live = payload.live_generation, context = backendContext(payload)
  const sessions = [...(previous?.sessions ?? (previous?.session ? [previous.session] : []))]
  const liveKey = live?.request_key && live.phase !== 'idle' ? live.request_key : undefined
  const key = liveKey ?? payload.groups.find(group => group.id === 'api_request')?.request_key
  // An explicit live identity resumes its cache; only stale terminal fallback is suppressed.
  const retained = key && (liveKey || key !== previous?.session?.priorTerminalKey) ? sessions.find(item => item.key === key) : undefined
  const session = rememberSession(retained ?? previous?.session, payload, !!retained && previous?.session?.key !== retained.key)
  const externalHistory = payload.external_history ?? previous?.externalHistory
  if (session) {
    const index = sessions.findIndex(item => item.key === session.key)
    if (index < 0) sessions.push(session)
    else sessions[index] = session
  }
  const memory: GenerationMemory = { capacity: reportedContextCapacity(payload) ?? previous?.capacity ?? null,
    capacityBasis: capacityValue(live?.context_capacity) !== null ? live?.context_capacity_basis || '容量' : capacityValue(context.capacity) !== null ? '容量' : previous?.capacityBasis,
    sample: previous?.sample ?? null, completed: previous?.completed ?? null, lastRequest: previous?.lastRequest,
    session, sessions, streamDecode: previous?.streamDecode, externalHistory, gatewayRunId }
  if (live?.request_key) {
    const old = memory.sample?.key === live.request_key ? memory.sample : null
    memory.sample = { key: live.request_key, mean: finite(live.average_tokens_per_second) ?? old?.mean ?? null,
      meanBasis: finite(live.average_tokens_per_second) !== null ? live.average_basis ?? 'backend_decode' : old?.meanBasis,
      used: finite(live.context_tokens) ?? old?.used ?? null, meanFinal: false, usedFinal: false }
  }
  const recent = payload.groups.find(g => g.id === 'api_request')
  if (recent?.request_status === 'completed' && recent.request_key) {
    memory.lastRequest = { key: recent.request_key, firstOutput: value(recent, 'first_output_s'),
      duration: value(recent, 'duration_s'), prompt: value(recent, 'prompt_tokens'), generated: value(recent, 'completion_tokens') }
    const speed = metric(recent, 'output_speed')
    const mean = speed?.rate_basis === 'decode' ? finite(speed.value) : null
    const input = value(recent, 'prompt_tokens'), output = value(recent, 'completion_tokens')
    const used = input !== null && output !== null ? input + output : null
    const sampled = memory.sample?.key === recent.request_key ? memory.sample : null
    const old = memory.completed?.key === recent.request_key ? memory.completed : null
    memory.completed = { key: recent.request_key, mean: mean ?? sampled?.mean ?? old?.mean ?? null,
      meanBasis: mean !== null ? 'backend_decode' : sampled?.meanBasis ?? old?.meanBasis,
      used: used ?? sampled?.used ?? old?.used ?? null,
      meanFinal: mean !== null || !!old?.meanFinal, usedFinal: used !== null || !!old?.usedFinal }
  }
  return memory
}

const emptyPhase = (): PhaseSamples => ({ points: [], startedAt: null, clockOffset: null, last: null, mean: null, meanBasis: null, window: null })
const emptySession = (key: string): GenerationSession => ({ key, phase: null, status: 'waiting', active: true,
  rates: { prefill: emptyPhase(), decode: emptyPhase() },
  request: { key, firstOutput: null, duration: null, prompt: null, generated: null }, used: null })
const terminalStatus = (status?: string) => !!status && ['completed', 'cancelled', 'failed', 'incomplete', 'stopped'].includes(status)
const sampleTime = (payload: Telemetry) => typeof payload.updated_at === 'number' ? payload.updated_at * 1000 : Date.parse(payload.updated_at)

function sessionFor(previous: GenerationSession | undefined, key: string, clientRequestId?: string | null): GenerationSession {
  if (previous?.key === key) return { ...previous, clientRequestId: clientRequestId || previous.clientRequestId }
  return { ...emptySession(key), clientRequestId: clientRequestId || undefined }
}

function reportedPhase(live: LiveGeneration): GenerationPhase | null {
  if (['generating', 'decode', 'thinking', 'output'].includes(live.phase || '') || (finite(live.generated_tokens) ?? 0) > 0 || finite(live.tokens_per_second) !== null) return 'decode'
  return live.phase === 'prefill' ? 'prefill' : null
}

function rememberSession(previous: GenerationSession | undefined, payload: Telemetry, resumed = false): GenerationSession | undefined {
  const live = payload.live_generation
  const recent = payload.groups.find(group => group.id === 'api_request')
  if (live?.request_key && live.phase !== 'idle') {
    const old = { ...sessionFor(previous, live.request_key, live.client_request_id), clientCallId: live.client_call_id || previous?.key === live.request_key && previous.clientCallId || undefined }
    if (resumed) old.sampleSegment = (old.sampleSegment ?? 0) + 1
    old.observedAt ??= sampleTime(payload)
    if (previous?.key !== live.request_key || resumed) old.priorTerminalKey = recent?.request_key ?? undefined
    const reported = reportedPhase(live)
    // Late Prefill observations cannot reset a request that has entered Decode.
    const phase = old.phase === 'decode' ? 'decode' : reported ?? old.phase
    const status = phase === 'decode' ? ['thinking', 'output'].includes(live.phase || '') ? live.phase! : old.status === 'output' ? 'output' : 'generating'
      : phase === 'prefill' ? 'prefill' : live.phase === 'queued' ? 'queued' : 'waiting'
    const request = { key: old.key, firstOutput: finite(live.first_output_s) ?? old.request.firstOutput,
      duration: finite(live.request_elapsed_s) ?? old.request.duration, prompt: finite(live.prompt_tokens) ?? old.request.prompt,
      generated: finite(live.generated_tokens) ?? old.request.generated }
    const rates = { ...old.rates }
    if (phase) {
      const current = finite(phase === 'prefill' ? live.prefill_tokens_per_second : live.tokens_per_second)
      const mean = finite(phase === 'prefill' ? live.average_prefill_tokens_per_second : live.average_tokens_per_second)
      const previous = old.rates[phase], wallTime = sampleTime(payload)
      const segment = `${old.key}:${phase}:${old.sampleSegment ?? 0}`
      const last = previous.points.at(-1)
      const time = wallTime
      const duplicate = last?.wallTime === wallTime && last.segment === segment
      const retained = Number.isFinite(time) && !duplicate
        ? retainChartWindow([...(previous.boundaryPoint ? [previous.boundaryPoint] : []), ...previous.points, { time, wallTime, values: { current }, segment }], time - HISTORY_MS)
        : { points: previous.points, boundaryPoint: previous.boundaryPoint }
      rates[phase] = { ...retained, clockOffset: null, startedAt: previous.startedAt ?? (Number.isFinite(time) ? time : null),
        last: current ?? previous.last, mean: mean ?? previous.mean,
        meanBasis: mean !== null ? (phase === 'prefill' ? live.prefill_average_basis : live.average_basis) ?? null : previous.meanBasis,
        window: finite(phase === 'prefill' ? live.prefill_window_s : live.window_s) ?? previous.window }
    }
    return { ...old, phase, status, active: true, rates, request, used: finite(live.context_tokens) ?? old.used }
  }
  if (recent?.request_key && recent.request_key !== previous?.priorTerminalKey && terminalStatus(recent.request_status)) {
    const old = { ...sessionFor(previous, recent.request_key, recent.client_request_id), clientCallId: recent.client_call_id || previous?.key === recent.request_key && previous.clientCallId || undefined }
    old.observedAt ??= sampleTime(payload)
    const prompt = value(recent, 'prompt_tokens') ?? old.request.prompt
    const generated = value(recent, 'completion_tokens') ?? old.request.generated
    const prefillSpeed = metric(recent, 'prefill_speed')
    const prefillMean = prefillSpeed?.rate_basis === 'prefill' ? finite(prefillSpeed.value) : null
    const phase = old.phase ?? (generated !== null && generated > 0 ? 'decode' : prefillMean !== null ? 'prefill' : null)
    const speed = metric(recent, 'output_speed')
    const mean = speed?.rate_basis === 'decode' ? finite(speed.value) : null
    return { ...old, phase, status: recent.request_status!, active: false,
      rates: { prefill: prefillMean !== null ? { ...old.rates.prefill, mean: prefillMean, meanBasis: 'backend_prefill' } : old.rates.prefill,
        decode: mean !== null ? { ...old.rates.decode, mean, meanBasis: 'backend_decode' } : old.rates.decode },
      request: { key: old.key, firstOutput: value(recent, 'first_output_s') ?? old.request.firstOutput,
        duration: value(recent, 'duration_s') ?? old.request.duration, prompt, generated,
        usage: { ttft_ms: value(recent, 'backend_ttft_ms') ?? undefined, generation_ms: value(recent, 'backend_generation_ms') ?? undefined,
          decode_tps: value(recent, 'backend_average_tps') ?? undefined } },
      used: prompt !== null && generated !== null ? prompt + generated : old.used }
  }
  if (previous?.active && live?.backend_running === 0 && live.backend_queued === 0 && (live.backend_paused === undefined || live.backend_paused === 0)) {
    // Direct backend requests may have no gateway terminal record.
    return { ...previous, active: false, status: 'finished' }
  }
  return previous
}

function requestViews(records: ModelRequestRecord[], memory: GenerationMemory | undefined, live: LiveGeneration | null | undefined,
  active: boolean, messageId?: string, session = memory?.session): MonitorRequestView[] {
  return records.map((record): MonitorRequestView => {
    const sampled = memory?.sessions?.find(item => item.key === (record.requestKey || record.id)) ?? memory?.sessions?.find(item => item.clientCallId === record.id)
      ?? (session?.key === (record.requestKey || record.id) || session?.clientCallId === record.id ? session : undefined)
    const stream = memory?.streamDecode?.find(item => item.requestId === record.id && item.messageId === messageId)
    const liveRequest = active && sampled?.key === live?.request_key && record.status === 'running'
    const view = (phase: GenerationPhase): GenerationPhaseView => {
      const rate = sampled?.rates[phase], finalMean = requestRate(record, phase).value
      const selectedActive = liveRequest && sampled?.phase === phase
      const phaseEnded = record.status !== 'running' || sampled?.phase != null && sampled.phase !== phase
      if (phase === 'decode' && stream?.points.some(point => point.values.current !== null)) {
        return { phase, current: stream.tokensPerSecond ?? (phaseEnded ? rate?.last ?? null : selectedActive ? finite(live?.tokens_per_second) : null), mean: finalMean ?? rate?.mean ?? null,
          currentNote: record.status === 'running' && stream.windowSeconds !== null ? `近 ${numberText(stream.windowSeconds)} 秒${stream.clock === 'arrival' ? '接收采样' : '解码计时'}滚动`
            : selectedActive && rate?.window != null ? `近 ${numberText(rate.window)} 秒滚动` : '',
          meanNote: finalMean === undefined && rate?.meanBasis?.startsWith('sampled_') ? '已采样区间均值' : '', ended: record.status !== 'running',
          points: stream.points, boundaryPoint: stream.boundaryPoint, historyStartedAt: stream.startedAt, historyKey: `${record.id}:decode` }
      }
      return { phase, current: phaseEnded ? rate?.last ?? null : selectedActive
        ? finite(phase === 'decode' ? live?.tokens_per_second : live?.prefill_tokens_per_second) : null, mean: finalMean ?? rate?.mean ?? null,
        currentNote: selectedActive && rate?.window != null ? `近 ${numberText(rate.window)} 秒滚动` : '',
        meanNote: finalMean === undefined && rate?.meanBasis?.startsWith('sampled_') ? '已采样区间均值' : '', ended: !selectedActive,
        points: rate?.points ?? [], boundaryPoint: rate?.boundaryPoint, historyStartedAt: rate?.startedAt ?? null, historyKey: `${record.id}:${phase}` }
    }
    const phases = { decode: view('decode'), prefill: view('prefill') }
    const liveMetrics = [record.live?.generated_tokens, record.live?.prefill_processed_tokens, record.live?.average_tokens_per_second,
      record.live?.average_prefill_tokens_per_second, sampled?.request.generated, phases.decode.current, phases.prefill.current,
      record.timings?.predicted_n, record.timings?.prompt_n].some(value => reportedNumber(value) !== undefined)
      || !!stream?.points.some(point => point.values.current !== null)
    return { ...record, phases, liveMetrics,
      decodeMean: requestRate(record, 'decode').value ?? null, prefillMean: requestRate(record, 'prefill').value ?? null }
  })
}

/** A page and a pinned lookup can finish out of order; the newest request revision wins. */
export function updatedHistorySelection(selected: ExternalRequestRecord | null, records: readonly ExternalRequestRecord[]): ExternalRequestRecord | null {
  if (!selected) return null
  const record = records.find(record => record.run_id === selected.run_id && record.instance_id === selected.instance_id && record.request_key === selected.request_key)
  return record && record.revision > selected.revision ? record : selected
}

export const externalRequestIdentity = (record: Pick<ExternalRequestRecord, 'run_id' | 'instance_id' | 'request_key'>) => JSON.stringify([record.run_id, record.instance_id, record.request_key])

export function externalModelRequest(record: ExternalRequestRecord, index: number): ModelRequestRecord {
  return { id: externalRequestIdentity(record), requestKey: record.request_key, index, kind: 'generation', status: record.status,
    startedAt: (record.started_at ?? 0) * 1000, ...(record.ended_at == null ? {} : { endedAt: record.ended_at * 1000 }),
    elapsed: reportedNumber(record.duration_s), firstOutput: reportedNumber(record.first_output_s), firstOutputBasis: 'gateway',
    labels: [...new Set(record.tool_names ?? [])], usage: record.usage, timings: record.timings, prefill: record.prefill,
    counters: record.counters, measurements: record.measurements,
    ...(record.live ? { live: record.live } : {}) }
}

/** Old gateways may return a whole inferred turn; the monitor still selects one HTTP request. */
export function externalGenerationView(snapshot: ExternalHistorySnapshot, memory?: GenerationMemory,
  payload?: Telemetry | null, historical = false): MonitorGenerationView {
  const sorted = [...snapshot.requests].sort((a, b) => (a.started_at ?? 0) - (b.started_at ?? 0))
  const currentRecord = sorted.find(record => record.status === 'running' && record.request_key === payload?.live_generation?.request_key)
    ?? sorted.find(record => record.status === 'running') ?? sorted.at(-1)
  const records = currentRecord ? [currentRecord] : []
  const state = currentRecord?.status ?? 'completed'
  const running = state === 'running'
  const live = payload?.live_generation
  const models = records.map((record, index) => {
    const request = externalModelRequest(record, index + 1)
    return live && request.status === 'running' && live.request_key === record.request_key ? mergeRequestLive(request, live, true) : request
  })
  const requests = requestViews(models, memory, live, running).map(request => ({ ...request, external: records.find(record => externalRequestIdentity(record) === request.id) }))
  const last = requests.at(-1)
  const selected = requests.find(request => request.status === 'running') ?? last
  const sampled = memory?.sessions?.find(session => session.key === selected?.requestKey)
  const phase = sampled?.phase ?? sampled?.lastPhase ?? (selected && selected.phases.decode.mean !== null ? 'decode' : selected && selected.phases.prefill.mean !== null ? 'prefill' : null)
  const empty = (phase: GenerationPhase): GenerationPhaseView => ({ phase, current: null, mean: null, currentNote: '', meanNote: '',
    ended: !running, points: [], historyStartedAt: null, historyKey: `${snapshot.run_id}:${phase}` })
  const phases = selected?.phases ?? { decode: empty('decode'), prefill: empty('prefill') }
  const current = phase ? phases[phase] : undefined
  const statusNames = { running: sampled?.status === 'prefill' ? '上下文填充' : sampled?.status === 'thinking' ? '正在思考' : sampled?.status === 'output' ? '正在输出' : sampled?.phase === 'decode' ? '正在生成' : '等待响应',
    completed: '已完成', failed: '请求失败', cancelled: '已中止', incomplete: '记录未完整' }
  const tokens = last ? requestTokens(last) : undefined
  const used = tokens?.input !== undefined && tokens.output !== undefined ? tokens.input + tokens.output : null
  const sessionKey = `${snapshot.run_id}:${currentRecord?.instance_id ?? ''}:${currentRecord?.request_key ?? ''}`
  return { key: selected?.requestKey ?? null, sessionKey, aggregation: 'request', origin: 'external', waiting: false,
    status: records.some(record => record.record_incomplete) ? '记录未完整' : statusNames[state],
    active: running, working: running, ended: !running, phase, phases, current: current?.current ?? null, mean: current?.mean ?? null,
    currentNote: current?.currentNote ?? '', meanNote: current?.meanNote ?? '', points: current?.points ?? [], historyStartedAt: current?.historyStartedAt ?? null,
    historyKey: current?.historyKey ?? sessionKey, used, capacity: reportedContextCapacity(payload ?? null) ?? memory?.capacity ?? null,
    contextBasis: '最后请求长度', request: selected ? { key: selected.id, firstOutput: selected.firstOutput ?? null, duration: selected.elapsed ?? null,
      prompt: tokens?.input ?? null, generated: tokens?.output ?? null } : null, requests, historical, tools: [] }
}

const activityStart = (activity: ChatGenerationActivity) => Math.max(activity.turn?.startedAt ?? 0, ...(activity.requests ?? []).map(request => request.startedAt))
const activityEventTime = (activity: ChatGenerationActivity) => Math.max(activity.turn?.endedAt ?? activity.turn?.startedAt ?? 0,
  ...(activity.requests ?? []).map(request => request.endedAt ?? request.startedAt))

// The main monitor follows phases and freezes its request result. Existing mini
// and chat callers retain their Decode-only current/idle contract above.
export function monitorGenerationView(payload: Telemetry | null, memory: GenerationMemory | undefined,
  instance: RuntimeInstance | undefined, failed = false, chatActivities: ChatGenerationActivity[] = [], selectedTurn?: ChatGenerationActivity | null): MonitorGenerationView {
  const ready = !!instance && isReady(instance.state), fresh = ready && !failed
  const live = fresh ? payload?.live_generation : null
  const base = generationView(payload, memory, ready, instance?.active_requests ?? 0, failed)
  const busy = (instance?.active_requests ?? 0) > 0 || (live?.backend_running ?? 0) > 0 || (live?.backend_queued ?? 0) > 0 || (live?.backend_paused ?? 0) > 0
  const activities = chatActivities.filter(activity => activity.instanceId === instance?.id)
  const running = activities.filter(activity => !terminalStatus(activity.phase))
  const latestActivity = [...activities].sort((a, b) => activityEventTime(a) - activityEventTime(b)).at(-1)
  const external = payload?.external_history ?? memory?.externalHistory
  const externalStart = Math.max(...(external?.requests ?? []).map(request => (request.started_at ?? 0) * 1000))
  const externalEventTime = Math.max(...(external?.requests ?? []).map(request => (request.ended_at ?? request.started_at ?? 0) * 1000))
  const liveExternal = !!live?.request_key && external?.requests.some(request => request.request_key === live.request_key)
  const liveInternal = !!live?.request_key && activities.some(activity => activity.requests?.some(request => matchingRequestLive(live, request))
    || !activity.requests?.length && activity.messageId === live.client_request_id)
  const externalCurrent = liveExternal || !liveInternal && (external?.state === 'running'
    ? !running.length || externalStart >= Math.max(...running.map(activityStart))
    : !running.length && externalEventTime >= (latestActivity ? activityEventTime(latestActivity) : -Infinity))
  if (!selectedTurn && external?.requests.length && externalCurrent
    && (ready || !instance || ['stopped', 'failed'].includes(instance.state))) {
    const view = externalGenerationView(external, memory, payload)
    if (failed) return { ...view, status: '采样失败', active: false, working: false, current: null, mean: null }
    if (instance && !ready) return { ...view, status: instance.state === 'failed' ? '实例异常' : '已停止', active: false, working: false, ended: true }
    return view
  }
  // Match by assistant message, not by the currently open conversation. Another
  // model/parallel request must never inherit this reply's tool status.
  const retainedActivity = activities.find(item => item.messageId === memory?.session?.clientRequestId)
  const matchingActivity = live?.request_key ? activities.find(item => item.messageId === live.client_request_id) : undefined
  const newerActivity = live?.request_key && running.length === 1 && (!matchingActivity || terminalStatus(matchingActivity.phase))
    && Math.max(activityStart(running[0]), running[0].stream?.sampledAt ?? 0) > (payload ? sampleTime(payload) : -Infinity) ? running[0] : undefined
  const activity = selectedTurn ?? newerActivity ?? (live?.request_key ? matchingActivity
    : running.length === 1 ? running[0] : running.length > 1 ? undefined : activities.find(item => item.messageId === memory?.session?.clientRequestId)
      ?? (!memory?.session || retainedActivity || (memory.session.observedAt ?? 0) <= (latestActivity ? activityEventTime(latestActivity) : 0) ? latestActivity : undefined))
  const toolActive = activity?.phase === 'searching' || activity?.phase === 'reading' || activity?.phase === 'tools'
  const activityEnded = terminalStatus(activity?.phase)
  const ambiguous = fresh && busy && !live?.request_key && !(activity && live?.backend_running === 0)
  const starting = !!instance && !ready && isActive(instance.state) && instance.state !== 'stopping'
  const session = ambiguous || starting || !instance || (activity && memory?.session?.clientRequestId !== activity.messageId) ? undefined : memory?.session
  const instanceStopped = !!instance && ['stopped', 'failed'].includes(instance.state)
  const active = fresh && !!live?.request_key && session?.key === live.request_key && !toolActive && !activityEnded && (!selectedTurn || selectedTurn.messageId === live.client_request_id)
  const working = !instanceStopped && (activity ? !activityEnded : fresh && (active || running.length > 0))
  const ended = activityEnded || !!session && (toolActive || !session.active || instanceStopped || (base.idle && !active))
  const phase = session?.phase ?? session?.lastPhase ?? null
  const sessionKey = `${instance?.id ?? ''}:${activity?.messageId ?? memory?.session?.clientRequestId ?? memory?.session?.key ?? ''}`
  function phaseView(selected: GenerationPhase): GenerationPhaseView {
    const rates = session?.rates[selected]
    const selectedActive = active && selected === phase
    const selectedEnded = ended || selected !== phase
    const current = failed ? null : selectedEnded ? rates?.last ?? null : selectedActive
      ? finite(selected === 'prefill' ? live?.prefill_tokens_per_second : live?.tokens_per_second) : null
    const mean = failed ? null : selectedEnded ? rates?.mean ?? null : selectedActive
      ? finite(selected === 'prefill' ? live?.average_prefill_tokens_per_second : live?.average_tokens_per_second) : null
    const window = selectedActive ? finite(selected === 'prefill' ? live?.prefill_window_s : live?.window_s) : null
    return { phase: selected, current, mean, ended: selectedEnded,
      currentNote: !selectedEnded && window !== null ? `近 ${numberText(window)} 秒滚动` : '',
      meanNote: rates?.meanBasis?.startsWith('sampled_') ? '已采样区间均值' : '',
      points: rates?.points ?? [], boundaryPoint: rates?.boundaryPoint, historyStartedAt: rates?.startedAt ?? null, historyKey: `${sessionKey}:${selected}` }
  }
  const phases = { prefill: phaseView('prefill'), decode: phaseView('decode') }
  const selected = phase ? phases[phase] : undefined
  const statusNames: Record<string, string> = { planning: '规划检索', prefill: '上下文填充', generating: '正在生成', thinking: '正在思考', output: '正在输出', waiting: '等待响应', queued: '排队中',
    completed: '已完成', stopped: '已停止', cancelled: '已中止', failed: '请求失败', incomplete: '已中断', finished: '已结束' }
  const status = selectedTurn && activityEnded ? statusNames[selectedTurn.phase] : failed ? '采样失败' : !instance ? '未选择实例' : instanceStopped ? instance.state === 'failed' ? '实例异常' : '已停止'
    : !ready ? instance.state === 'stopping' ? '正在停止' : '等待实例就绪' : ambiguous ? (live?.backend_running ?? 0) > 1 ? '多请求运行' : (live?.backend_queued ?? 0) > 0 ? '排队中' : '等待响应'
      : toolActive ? '调用工具' : activity?.phase === 'compacting' ? '压缩上下文' : activityEnded || activity?.phase === 'planning' ? statusNames[activity!.phase]
        : running.length > 1 && !live?.request_key ? '多请求运行' : activity && !active ? '等待响应'
          : session && (active || ended) ? statusNames[session.status] ?? '等待响应' : base.idle ? '空闲' : '等待响应'
  const context = fresh ? backendContext(payload) : { used: null, basis: '' }
  const historicalTokens = selectedTurn?.requests?.length ? requestTokens(selectedTurn.requests.at(-1)!) : undefined
  const used = selectedTurn ? historicalTokens?.input !== undefined && historicalTokens.output !== undefined ? historicalTokens.input + historicalTokens.output : null
    : ended ? context.used ?? session?.used ?? base.used : base.used
  const contextBasis = selectedTurn ? '所选轮次 · 最后请求长度' : ended && context.used === null && session?.used != null ? '上次请求长度' : base.contextBasis
  // Legacy /api/chat telemetry carries the assistant message id even before the
  // renderer's activity ledger arrives. External units use external_history above.
  const records = activity?.requests ?? (session ? (session.clientRequestId ? (memory?.sessions ?? [session]).filter(item => item.clientRequestId === session.clientRequestId) : [session]).map((item, index): ModelRequestRecord => ({
    id: item.key, requestKey: item.key, index: index + 1, kind: 'generation', labels: [], status: item.active ? 'running' : terminalStatus(item.status) ? item.status as ModelRequestRecord['status'] : 'completed',
    startedAt: item.rates.prefill.startedAt ?? item.rates.decode.startedAt ?? 0, elapsed: item.request.duration ?? undefined, firstOutput: item.request.firstOutput ?? undefined,
    usage: { ...item.request.usage, prompt_tokens: item.request.prompt ?? undefined, completion_tokens: item.request.generated ?? undefined },
    timings: { predicted_per_second: item.rates.decode.meanBasis === 'backend_decode' ? item.rates.decode.mean ?? undefined : undefined,
      prompt_per_second: item.rates.prefill.meanBasis === 'backend_prefill' ? item.rates.prefill.mean ?? undefined : undefined },
  })) : [])
  const requests = requestViews(records, memory, live, active, activity?.messageId, session)
  const streamed = ready && !toolActive && !activityEnded ? requests.find(request => request.status === 'running'
    && memory?.streamDecode?.some(item => item.requestId === request.id && item.messageId === activity?.messageId && item.tokensPerSecond !== null)) : undefined
  const displayed = streamed?.phases.decode ?? selected
  const displayedPhases = streamed ? { ...phases, decode: streamed.phases.decode } : phases
  const displayedPhase = streamed ? 'decode' : phase
  return { key: streamed?.requestKey ?? streamed?.id ?? session?.key ?? null, sessionKey, phases: displayedPhases, phase: displayedPhase, current: displayed?.current ?? null, mean: displayed?.mean ?? null,
    currentNote: displayed?.currentNote ?? '', meanNote: displayed?.meanNote ?? '',
    status: streamed && !active ? activity?.phase === 'planning' ? '规划检索' : activity?.phase === 'compacting' ? '压缩上下文' : '正在生成' : status,
    active: active || !!streamed, working, ended: streamed ? false : ended, points: displayed?.points ?? [], historyStartedAt: displayed?.historyStartedAt ?? null, historyKey: `${sessionKey}:${displayedPhase ?? ''}`,
    used, capacity: selectedTurn ? capacityValue(selectedTurn.contextCapacity) ?? (selectedTurn.instanceId && selectedTurn.instanceId === instance?.id ? base.capacity : null) : base.capacity,
    contextBasis, request: session && (active || ended) ? session.request : null,
    requests, turn: activity?.turn, toolCount: activity?.toolCount, tools: activity?.toolActivity ?? [], historical: !!selectedTurn,
    aggregation: activity || session?.clientRequestId ? 'turn' : 'request', origin: activity || session?.clientRequestId ? 'internal' : 'backend' }
}

export function generationView(payload: Telemetry | null, memory: GenerationMemory | undefined,
  ready: boolean, activeRequests: number, failed = false) {
  const fresh = ready && !failed, live = fresh ? payload?.live_generation : null
  const context = fresh ? backendContext(payload) : { capacity: null, used: null, basis: '' }
  const idle = fresh && activeRequests === 0 && live?.backend_running === 0 && live.backend_queued === 0
    && (live.backend_paused === undefined || live.backend_paused === 0)
  const current = idle ? 0 : finite(live?.tokens_per_second)
  const nativeLast = payload?.groups.find(g => g.id === 'strata_last')
  const nativeMean = value(nativeLast, 'decode_tok_s')
  const completed = memory?.completed
  const historicalMean = completed?.mean ?? nativeMean ?? memory?.sample?.mean ?? null
  const historicalMeanNote = completed?.mean != null ? completed.meanFinal ? '上次完成请求 · 解码均值' : '上次完成请求 · 末次采样均值'
    : nativeMean !== null ? '后端最近请求 · 解码均值' : memory?.sample?.meanBasis === 'sampled_decode' ? '上次采样 · 已采样区间均值' : '上次采样 · 解码均值'
  const liveMean = finite(live?.average_tokens_per_second)
  const usingLive = !!live?.request_key
  const mean = usingLive ? liveMean : historicalMean
  const meanNote = usingLive ? live.average_basis === 'sampled_decode' ? '当前请求 · 已采样区间均值' : '当前请求 · 解码均值' : mean === null ? '尚无解码均值' : historicalMeanNote
  const comparison = usingLive && completed?.mean != null && completed.key !== live?.request_key
    ? `${completed.meanFinal ? '上次完成' : '上次采样'} ${numberText(completed.mean, 2, true)} token/s` : undefined
  const capacity = capacityValue(live?.context_capacity) ?? capacityValue(context.capacity) ?? capacityValue(memory?.capacity)
  const capacityBasis = capacityValue(live?.context_capacity) !== null ? live?.context_capacity_basis || '容量' : capacityValue(context.capacity) !== null ? '容量' : memory?.capacityBasis || '容量'
  const liveUsed = finite(live?.context_tokens)
  const nativeInput = value(nativeLast, 'prompt_tokens'), nativeOutput = value(nativeLast, 'output_tokens')
  const nativeUsed = nativeInput !== null && nativeOutput !== null ? nativeInput + nativeOutput : null
  // Historical request length remains visible, but is never relabeled as live KV occupancy.
  const historicalUsed = completed?.used ?? nativeUsed ?? memory?.sample?.used ?? null
  const used = liveUsed ?? context.used ?? (usingLive ? null : historicalUsed)
  const contextBasis = liveUsed !== null ? `当前请求 · ${live?.context_basis || 'Token 用量'}` : context.used !== null ? context.basis
    : usingLive ? '当前请求' : completed?.used != null ? completed.usedFinal ? '上次完成请求长度' : '上次完成请求 · 末次采样'
      : nativeUsed !== null ? '后端最近请求长度' : historicalUsed !== null ? '上次请求 · 末次采样' : '当前用量未知'
  return { idle, current, mean, meanNote, comparison, used, capacity, capacityBasis, contextBasis,
    currentNote: idle ? '当前空闲 · 无生成' : live?.window_s != null ? `当前请求 · 近 ${numberText(live.window_s)} 秒滚动` : failed ? '采样失败' : '等待有效采样',
    // Only current observations enter the time series. Retained results are not flat live samples.
    chart: { current, average: liveMean, context: liveUsed ?? context.used },
    segment: live?.request_key ?? (idle ? 'idle' : 'unknown') }
}
