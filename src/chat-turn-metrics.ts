import type { ToolCall } from './chat-history'
import type { LiveGeneration } from './types'
import { reportedNumber, type Timings, type Usage } from './chat-response'

export type RequestStatus = 'running' | 'completed' | 'cancelled' | 'failed' | 'incomplete'
export interface PhaseMeasurement {
  tokens: number | null
  seconds: number | null
  value: number | null
  basis: 'backend_decode' | 'backend_prefill' | null
  reported: 0 | 1
  coverage: 'complete' | 'partial' | 'unknown'
}
export interface RequestCounters { input: number | null; output: number | null; cached: number | null }
export interface PromptProgress { total?: number; processed?: number; cache?: number }
export interface ModelRequestRecord {
  id: string
  requestKey?: string
  index: number
  kind: 'generation' | 'planning' | 'compaction'
  status: RequestStatus
  startedAt: number
  endedAt?: number
  elapsed?: number
  firstOutput?: number
  firstOutputBasis?: 'gateway' | 'client'
  labels: string[]
  usage?: Usage
  timings?: Timings
  measurements?: { decode: PhaseMeasurement; prefill: PhaseMeasurement }
  counters?: RequestCounters
  // Request-bound backend observations stay separate from the final API usage/timings.
  live?: Pick<LiveGeneration, 'source' | 'prompt_tokens' | 'generated_tokens' | 'average_tokens_per_second' | 'average_basis'
    | 'average_prefill_tokens_per_second' | 'prefill_average_basis' | 'prefill_processed_tokens' | 'prefill_cached_tokens' | 'prefill_total_tokens'>
  prefill?: { average_prefill_tokens_per_second?: number; prefill_average_basis?: 'backend_prefill' | 'sampled_prefill'; prefill_processed_tokens?: number; prefill_cached_tokens?: number; prefill_total_tokens?: number }
}
export interface TokenTotal { value: number | null; reported: number }
export interface RateTotal { value: number | null; tokens: number; seconds: number; reported: number; partial_reported?: number }
export interface TurnTotals {
  startedAt: number
  endedAt?: number
  elapsed: number
  status: RequestStatus
  requests: number
  input: TokenTotal
  output: TokenTotal
  cached: TokenTotal
  reasoning: TokenTotal
  decode: RateTotal
  prefill: RateTotal
  e2e: number | null
}
export interface BackendRequestReport {
  request_key: string
  client_request_id?: string | null
  client_call_id?: string | null
  status: RequestStatus
  started_at?: number
  ended_at?: number
  duration_s: number
  first_output_s?: number | null
  usage: Usage
  timings: Timings
  prefill?: ModelRequestRecord['prefill']
  measurements?: ModelRequestRecord['measurements']
  counters?: RequestCounters
}

export interface StreamDecodeSample {
  requestId: string
  sampledAt: number
  tokensPerSecond: number | null
  windowSeconds: number | null
  clock?: 'arrival'
}
type DecodeCounter = { tokens: number; milliseconds: number }
export interface StreamDecodeState {
  requestId: string
  counter: 'timings' | 'usage' | 'arrival'
  counters: DecodeCounter[]
  sample: StreamDecodeSample
}
const DECODE_WINDOW_MS = 3000

/** Prefer paired engine timings; otherwise time cumulative usage arrivals, never text chunks. */
export function sampleStreamDecode(previous: StreamDecodeState | undefined, requestId: string,
  chunk: { usage?: Usage; timings?: Timings }, sampledAt: number): StreamDecodeState | undefined {
  const engineMilliseconds = reportedNumber(chunk.timings?.predicted_ms)
  const timingTokens = reportedNumber(chunk.timings?.predicted_n)
  const tokens = engineMilliseconds === undefined ? reportedNumber(chunk.usage?.completion_tokens) : timingTokens ?? reportedNumber(chunk.usage?.completion_tokens)
  if (tokens === undefined) return previous?.requestId === requestId ? previous : undefined
  const milliseconds = engineMilliseconds ?? sampledAt
  const counter = engineMilliseconds === undefined ? 'arrival' : timingTokens === undefined ? 'usage' : 'timings'
  const old = previous?.requestId === requestId && previous.counter === counter ? previous : undefined
  const last = old?.counters.at(-1)
  const reset = !last || tokens < last.tokens || milliseconds < last.milliseconds
  const sample: StreamDecodeSample = { requestId, sampledAt, tokensPerSecond: null, windowSeconds: null, ...(counter === 'arrival' ? { clock: 'arrival' as const } : {}) }
  if (reset) return { requestId, counter, counters: [{ tokens, milliseconds }], sample }
  if (tokens === last.tokens && milliseconds === last.milliseconds) return old
  // A repeated decode clock cannot time the additional tokens. Keep the latest
  // count as the next baseline, but do not manufacture a new speed observation.
  if (milliseconds === last.milliseconds) return { ...old!, counters: [...old!.counters.slice(0, -1), { tokens, milliseconds }] }
  const cutoff = milliseconds - DECODE_WINDOW_MS
  const counters = [...old!.counters, { tokens, milliseconds }].filter(point => point.milliseconds >= cutoff)
  // A first/final-only timing report is not a rolling real-time observation.
  if (counters.length < 2) return { requestId, counter, counters, sample }
  const first = counters[0], seconds = (milliseconds - first.milliseconds) / 1000
  return { requestId, counter, counters, sample: { ...sample, tokensPerSecond: (tokens - first.tokens) / seconds, windowSeconds: seconds } }
}

/** Native timing supplements the original gateway/client clocks rather than overwriting them. */
export function requestFirstOutput(request?: Pick<ModelRequestRecord, 'usage' | 'firstOutput' | 'firstOutputBasis'>) {
  const native = reportedNumber(request?.usage?.ttft_ms)
  return native !== undefined ? { value: native / 1000, basis: '引擎' }
    : { value: request?.firstOutput, basis: request?.firstOutputBasis === 'gateway' ? '网关' : request?.firstOutputBasis === 'client' ? '客户端' : '' }
}

export function requestEngineRate(request: ModelRequestRecord) {
  const tokens = requestTokens(request).output
  const milliseconds = reportedNumber(request.usage?.generation_ms)
  const reported = reportedNumber(request.usage?.decode_tps)
  const seconds = milliseconds === undefined ? tokens !== undefined && reported !== undefined && reported > 0 ? tokens / reported : undefined : milliseconds / 1000
  const value = request.status === 'completed' ? reported ?? (tokens !== undefined && seconds !== undefined && seconds > 0 ? tokens / seconds : undefined) : undefined
  return { value, tokens, seconds }
}

export function engineRateTotal(requests: readonly ModelRequestRecord[]): RateTotal {
  const values = requests.map(requestEngineRate).filter(rate => rate.value !== undefined && rate.tokens !== undefined && rate.seconds !== undefined)
  const tokens = values.reduce((sum, rate) => sum + rate.tokens!, 0), seconds = values.reduce((sum, rate) => sum + rate.seconds!, 0)
  return { value: seconds > 0 ? tokens / seconds : null, tokens, seconds, reported: values.length }
}

/** Footer counters describe the active call, or the last call during tool waits / after completion. */
export function footerRequest(requests: readonly ModelRequestRecord[], generating: boolean): ModelRequestRecord | undefined {
  if (generating) {
    for (let index = requests.length - 1; index >= 0; index--) if (requests[index].status === 'running') return requests[index]
  }
  return requests.at(-1)
}

export function requestTokens(request: Pick<ModelRequestRecord, 'usage' | 'timings' | 'prefill' | 'live' | 'counters'>) {
  if (request.counters) {
    const input = reportedNumber(request.counters.input), output = reportedNumber(request.counters.output), cached = reportedNumber(request.counters.cached)
    return { input, output, cached, uncached: input !== undefined && cached !== undefined ? cached <= input ? input - cached : undefined
      : reportedNumber(request.timings?.prompt_n) ?? reportedNumber(request.prefill?.prefill_processed_tokens), reasoning: reportedNumber(request.usage?.completion_tokens_details?.reasoning_tokens) }
  }
  const timing = request.timings
  const processed = reportedNumber(timing?.prompt_n), cached = reportedNumber(request.usage?.prompt_tokens_details?.cached_tokens) ?? reportedNumber(timing?.cache_n) ?? reportedNumber(request.prefill?.prefill_cached_tokens) ?? reportedNumber(request.live?.prefill_cached_tokens)
  const input = reportedNumber(request.usage?.prompt_tokens) ?? reportedNumber(request.prefill?.prefill_total_tokens) ?? reportedNumber(request.live?.prefill_total_tokens)
    ?? (processed !== undefined && cached !== undefined ? processed + cached : undefined) ?? reportedNumber(request.live?.prompt_tokens)
  return {
    input,
    uncached: input !== undefined && cached !== undefined ? cached <= input ? input - cached : undefined
      : processed ?? reportedNumber(request.prefill?.prefill_processed_tokens) ?? reportedNumber(request.live?.prefill_processed_tokens),
    output: reportedNumber(request.usage?.completion_tokens) ?? reportedNumber(timing?.predicted_n) ?? reportedNumber(request.live?.generated_tokens), cached,
    reasoning: reportedNumber(request.usage?.completion_tokens_details?.reasoning_tokens),
  }
}

export function uncachedInputTotal(requests: readonly ModelRequestRecord[]): TokenTotal {
  const values = requests.map(request => requestTokens(request).uncached).filter((value): value is number => value !== undefined)
  return { value: values.length ? values.reduce((sum, value) => sum + value, 0) : null, reported: values.length }
}

function nativePhaseRate(timings: Timings | undefined, phase: 'decode' | 'prefill') {
  const perSecond = reportedNumber(phase === 'decode' ? timings?.predicted_per_second : timings?.prompt_per_second)
  const perToken = reportedNumber(phase === 'decode' ? timings?.predicted_per_token_ms : timings?.prompt_per_token_ms)
  return perSecond ?? (perToken !== undefined && perToken > 0 ? 1000 / perToken : undefined)
}

export function requestRate(request: ModelRequestRecord, phase: 'decode' | 'prefill') {
  const timings = request.timings
  const tokens = reportedNumber(phase === 'decode' ? timings?.predicted_n : timings?.prompt_n) ??
    (phase === 'decode' ? reportedNumber(request.usage?.completion_tokens) : reportedNumber(request.prefill?.prefill_processed_tokens))
  const milliseconds = reportedNumber(phase === 'decode' ? timings?.predicted_ms : timings?.prompt_ms)
  const reportedRate = nativePhaseRate(timings, phase) ??
    (phase === 'prefill' && request.prefill?.prefill_average_basis !== 'sampled_prefill' ? reportedNumber(request.prefill?.average_prefill_tokens_per_second) : undefined)
  const measurement = request.measurements?.[phase]
  if (measurement) return { value: measurement.coverage === 'complete' ? measurement.value ?? reportedRate : reportedRate ?? measurement.value ?? undefined,
    tokens: measurement.reported ? measurement.tokens ?? undefined : undefined,
    seconds: measurement.reported ? measurement.seconds ?? undefined : undefined, partial: measurement.coverage === 'partial' }
  const seconds = milliseconds !== undefined && milliseconds > 0 ? milliseconds / 1000 : tokens === 0 && milliseconds === 0 ? 0
    : tokens !== undefined && reportedRate !== undefined && reportedRate > 0 ? tokens / reportedRate : undefined
  const value = reportedRate ?? (tokens !== undefined && seconds !== undefined && seconds > 0 ? tokens / seconds : undefined)
  if (value !== undefined) return { value, tokens, seconds }
  const live = request.live
  const liveRate = phase === 'decode'
    ? live?.average_basis === 'backend_decode' ? reportedNumber(live.average_tokens_per_second) : undefined
    : live?.prefill_average_basis === 'backend_prefill' ? reportedNumber(live.average_prefill_tokens_per_second) : undefined
  const liveTokens = reportedNumber(phase === 'decode' ? live?.generated_tokens : live?.prefill_processed_tokens)
  // Weight a backend mean with its own observed counter; never invent native timing fields.
  return liveRate === undefined ? { value, tokens, seconds }
    : { value: liveRate, tokens: liveTokens, seconds: liveTokens !== undefined && liveRate > 0 ? liveTokens / liveRate : undefined }
}

/** Each request is a ledger entry; SSE usage is cumulative within that entry. */
export function turnTotals(requests: readonly ModelRequestRecord[], startedAt: number, elapsed: number, status: RequestStatus = 'running', endedAt?: number): TurnTotals {
  const total = (field: keyof ReturnType<typeof requestTokens>): TokenTotal => {
    const values = requests.map(request => requestTokens(request)[field]).filter((value): value is number => value !== undefined)
    return { value: values.length ? values.reduce((sum, value) => sum + value, 0) : null, reported: values.length }
  }
  const rate = (phase: 'decode' | 'prefill'): RateTotal => {
    const values = requests.map(request => requestRate(request, phase)).filter(value => value.tokens !== undefined && value.seconds !== undefined)
    const tokens = values.reduce((sum, value) => sum + value.tokens!, 0), seconds = values.reduce((sum, value) => sum + value.seconds!, 0)
    const partial = values.filter(value => value.partial).length
    return { value: seconds > 0 ? tokens / seconds : null, tokens, seconds, reported: values.length, ...(partial ? { partial_reported: partial } : {}) }
  }
  const input = total('input'), output = total('output')
  return { startedAt, ...(endedAt === undefined ? {} : { endedAt }), elapsed, status, requests: requests.length,
    input, output, cached: total('cached'), reasoning: total('reasoning'), decode: rate('decode'), prefill: rate('prefill'),
    e2e: output.value !== null && output.reported === requests.length && elapsed > 0 ? output.value / elapsed : null }
}

/** Capture native cumulative pairs before merging sticky SSE fields. */
export function mergeRequestMetrics(request: ModelRequestRecord, chunk: Pick<BackendRequestReport, 'usage' | 'timings' | 'prefill' | 'measurements' | 'counters'> & { prompt_progress?: PromptProgress }, multipleChoices = false): ModelRequestRecord {
  if (multipleChoices) chunk = { ...chunk, timings: chunk.timings?.cache_n === undefined ? {} : { cache_n: chunk.timings.cache_n },
    prefill: { prefill_cached_tokens: chunk.prefill?.prefill_cached_tokens, prefill_total_tokens: chunk.prefill?.prefill_total_tokens } }
  const old = requestTokens(request)
  const decoded = request.timings?.predicted_n !== undefined || request.timings?.predicted_ms !== undefined || (old.output ?? 0) > 0
  const progress = decoded ? undefined : chunk.prompt_progress
  const total = reportedNumber(progress?.total)
  const cached = reportedNumber(chunk.timings?.cache_n) ?? reportedNumber(chunk.usage?.prompt_tokens_details?.cached_tokens) ?? reportedNumber(progress?.cache) ?? old.cached
  const processed = reportedNumber(progress?.processed)
  const work = !multipleChoices && processed !== undefined && cached !== undefined && processed >= cached ? processed - cached : undefined
  const prefill = { ...chunk.prefill, ...(total === undefined ? {} : { prefill_total_tokens: total }),
    ...(work === undefined ? {} : { prefill_processed_tokens: work }), ...(progress?.cache === undefined ? {} : { prefill_cached_tokens: cached }) }
  const completeInput = reportedNumber(chunk.usage?.prompt_tokens) ?? reportedNumber(prefill.prefill_total_tokens)
  const previousComplete = old.input !== undefined && (reportedNumber(request.usage?.prompt_tokens) !== undefined || reportedNumber(request.prefill?.prefill_total_tokens) !== undefined)
  const observedWork = reportedNumber(chunk.timings?.prompt_n) ?? work ?? reportedNumber(request.timings?.prompt_n) ?? reportedNumber(request.prefill?.prefill_processed_tokens)
  const counters = chunk.counters ?? { input: completeInput ?? (previousComplete ? old.input : observedWork !== undefined && cached !== undefined ? observedWork + cached : old.input) ?? null,
    output: reportedNumber(chunk.timings?.predicted_n) ?? reportedNumber(chunk.usage?.completion_tokens) ?? old.output ?? null, cached: cached ?? null }
  const measure = (phase: 'decode' | 'prefill'): PhaseMeasurement => {
    if (multipleChoices) return { tokens: null, seconds: null, value: null, basis: null, reported: 0, coverage: 'unknown' }
    const old = request.measurements?.[phase], timing = chunk.timings
    const tokens = reportedNumber(phase === 'decode' ? timing?.predicted_n : timing?.prompt_n) ??
      (phase === 'decode' ? reportedNumber(chunk.usage?.completion_tokens) : reportedNumber(prefill.prefill_processed_tokens))
    const ms = reportedNumber(phase === 'decode' ? timing?.predicted_ms : timing?.prompt_ms)
    const rate = nativePhaseRate(timing, phase) ??
      (phase === 'prefill' && prefill.prefill_average_basis === 'backend_prefill' ? reportedNumber(prefill.average_prefill_tokens_per_second) : undefined)
    const seconds = ms !== undefined && (ms > 0 || tokens === 0) ? ms / 1000 : tokens !== undefined && rate !== undefined && rate > 0 ? tokens / rate : undefined
    if (tokens !== undefined && seconds !== undefined) return { tokens, seconds, value: seconds > 0 ? tokens / seconds : null,
      basis: phase === 'decode' ? 'backend_decode' : 'backend_prefill', reported: 1, coverage: 'complete' }
    return old?.reported ? { ...old, ...(tokens !== undefined || ms !== undefined || rate !== undefined ? { coverage: 'partial' as const } : {}) }
      : { tokens: null, seconds: null, value: null, basis: null, reported: 0, coverage: 'unknown' }
  }
  const usage = { ...request.usage, ...chunk.usage }
  if (reportedNumber(request.usage?.ttft_ms) !== undefined) usage.ttft_ms = request.usage!.ttft_ms
  const measurements = chunk.measurements ?? { decode: measure('decode'), prefill: measure('prefill') }
  const timings = { ...(multipleChoices ? {} : request.timings), ...chunk.timings }
  const mergedPrefill = { ...(multipleChoices ? {} : request.prefill), ...prefill }
  const decodeObserved = [chunk.timings?.predicted_n, chunk.timings?.predicted_ms, chunk.usage?.completion_tokens].some(value => reportedNumber(value) !== undefined)
  const prefillObserved = [chunk.timings?.prompt_n, chunk.timings?.prompt_ms, prefill.prefill_processed_tokens].some(value => reportedNumber(value) !== undefined)
  if (decodeObserved || nativePhaseRate(chunk.timings, 'decode') !== undefined) {
    if (reportedNumber(chunk.timings?.predicted_per_second) === undefined) delete timings.predicted_per_second
    if (reportedNumber(chunk.timings?.predicted_per_token_ms) === undefined) delete timings.predicted_per_token_ms
  }
  if (prefillObserved || nativePhaseRate(chunk.timings, 'prefill') !== undefined || reportedNumber(prefill.average_prefill_tokens_per_second) !== undefined) {
    if (reportedNumber(chunk.timings?.prompt_per_second) === undefined) delete timings.prompt_per_second
    if (reportedNumber(chunk.timings?.prompt_per_token_ms) === undefined) delete timings.prompt_per_token_ms
    if (reportedNumber(prefill.average_prefill_tokens_per_second) === undefined) { delete mergedPrefill.average_prefill_tokens_per_second; delete mergedPrefill.prefill_average_basis }
  }
  return { ...request, usage, counters, timings, prefill: mergedPrefill, measurements }
}

export function mergeRequestReport(request: ModelRequestRecord, report: BackendRequestReport): ModelRequestRecord {
  const full = report.counters && report.measurements
  const base = full ? { ...request, usage: reportedNumber(request.usage?.ttft_ms) === undefined ? undefined : { ttft_ms: request.usage!.ttft_ms },
    timings: undefined, prefill: undefined, live: undefined, measurements: undefined, counters: report.counters } : request
  return { ...mergeRequestMetrics(base, report), requestKey: report.request_key, status: report.status,
    ...(report.ended_at === undefined ? {} : { endedAt: report.ended_at * 1000 }), elapsed: report.duration_s,
    ...(reportedNumber(report.first_output_s) === undefined ? {} : { firstOutput: report.first_output_s!, firstOutputBasis: 'gateway' }),
  }
}

export function toolMetricLabel(name: string, result?: unknown) {
  if (name === 'attach_file') {
    const object = result !== null && typeof result === 'object' ? result as Record<string, unknown> : {}
    const resource = object.resource !== null && typeof object.resource === 'object' ? object.resource as Record<string, unknown> : {}
    return resource.kind === 'image' ? '引用图片' : resource.kind ? '引用文件' : '引用本地资源'
  }
  return ({ search_web: '搜索网页', search_images: '关键词搜图', reverse_image_search: '以图搜源', read_page: '读取网页', list_directory: '列出目录', read_file: '读取文件', write_file: '写入文件', current_time: '获取时间', calculate: '计算' } as Record<string, string>)[name] || name
}

/** Tags come from structured input / calls, never from a model's thoughts. */
export function modelRequestLabels(messages: readonly Record<string, unknown>[], calls: readonly ToolCall[] = []): string[] {
  const labels = calls.map(call => toolMetricLabel(call.function.name))
  if (messages.some(message => Array.isArray(message.content) && message.content.some(part => part && typeof part === 'object' && part.type === 'image_url'))) labels.push('图片输入')
  if (messages.some(message => message.role === 'tool')) labels.push('工具结果')
  return [...new Set(labels.filter(Boolean))]
}
