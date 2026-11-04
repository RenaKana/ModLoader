import { useSyncExternalStore } from 'react'
import { apiFetch, ApiError } from './api'
import type { GenerationAccounting, GenerationEvent, Telemetry } from './types'
import type { GenerationPhase, MonitorRequestView } from './monitor-generation'
import { reportedNumber } from './chat-response'

export type GenerationEventIdentity = { runId: string; instanceId: string; requestKey: string }
export function generationEventIdentity(request?: MonitorRequestView, store?: GenerationEventStore): GenerationEventIdentity | undefined {
  const record = request?.external
  return record && request.status === 'running' ? { runId: record.run_id, instanceId: record.instance_id, requestKey: record.request_key } : store?.identity(request)
}
const noSubscribe = () => () => {}
const noSnapshot = () => null

// This store is a readout channel, not telemetry or chart history. One render
// frame publishes the latest real event to React. Imperative consumers can read
// pending values independently of a hidden renderer's animation frames.
export function createGenerationEventStore(frame: (callback: FrameRequestCallback) => number = callback => requestAnimationFrame(callback)) {
  let instanceId = '', runId: string | undefined, requestKey: string | null | undefined, pollAt = Infinity
  let snapshot: GenerationEvent | null = null, pending: GenerationEvent | null = null, queued = false, observed = -Infinity
  let records = new Map<string, GenerationAccounting>(), published = new Map<string, GenerationAccounting>()
  let clientCallId: string | null | undefined
  const listeners = new Set<() => void>()
  const immediateListeners = new Set<() => void>()
  function notify(immediate = false) {
    if (immediate) immediateListeners.forEach(listener => listener())
    if (queued) return
    queued = true
    frame(() => { queued = false; snapshot = pending; published = new Map(records); listeners.forEach(listener => listener()) })
  }
  function upsert(value: GenerationAccounting) {
    const previous = records.get(value.request.request_key)?.request, ttft = reportedNumber(previous?.usage.ttft_ms)
    records.set(value.request.request_key, !previous ? value : { ...value, request: { ...value.request,
      first_output_s: previous.first_output_s ?? value.request.first_output_s,
      usage: ttft === undefined ? value.request.usage : { ...value.request.usage, ttft_ms: ttft } } })
  }
  const store = {
    subscribe(listener: () => void) { listeners.add(listener); return () => { listeners.delete(listener) } },
    subscribeImmediate(listener: () => void) { immediateListeners.add(listener); return () => { immediateListeners.delete(listener) } },
    getSnapshot: () => snapshot,
    reset(id: string) { instanceId = id; runId = undefined; requestKey = undefined; pollAt = Infinity; observed = -Infinity; pending = null; records = new Map(); notify(true) },
    clear() { pending = null; records = new Map([...records].filter(([, value]) => value.request.status !== 'running')); observed = -Infinity; notify(true) },
    confirm(payload: Telemetry) {
      if (payload.instance_id !== instanceId) return
      const nextRun = payload.external_history?.run_id
      if (nextRun && (runId && nextRun !== runId || pending && pending.run_id !== nextRun)) { pending = null; records = new Map(); observed = -Infinity; notify() }
      runId = nextRun ?? runId ?? (pending?.live_generation?.request_key === payload.live_generation?.request_key ? pending?.run_id : undefined)
      requestKey = payload.live_generation?.request_key
      clientCallId = payload.live_generation?.client_call_id
      pollAt = typeof payload.updated_at === 'number' ? payload.updated_at : Date.parse(payload.updated_at) / 1000
      store.confirmFinalized(payload.external_history?.requests ?? [])
    },
    confirmFinalized(values: readonly { request_key: string; status: string; instance_id?: string; run_id?: string }[], sourceInstance = instanceId) {
      if (sourceInstance !== instanceId) return
      const consumed = new Set(values.filter(value => value.status !== 'running' && (!value.instance_id || value.instance_id === instanceId)
        && (!value.run_id || value.run_id === runId)).map(value => value.request_key))
      let changed = false
      consumed.forEach(key => { changed = records.delete(key) || changed })
      if (changed) { if (pending) pending = { ...pending, finalized: (pending.finalized ?? []).filter(value => !consumed.has(value.request.request_key)) }; notify() }
    },
    receive(event: GenerationEvent) {
      if (!event || event.instance_id !== instanceId || typeof event.run_id !== 'string' || !Number.isFinite(event.observed_at)
        || runId && event.run_id !== runId || event.observed_at < observed) return
      if (requestKey && event.live_generation?.request_key && event.live_generation.request_key !== requestKey && !event.finalized?.length) return
      if (!runId && requestKey && event.live_generation?.request_key === requestKey) runId = event.run_id
      for (const final of event.finalized ?? []) upsert(final)
      if (event.accounting && event.live_generation && (!requestKey || event.live_generation.request_key === requestKey) && event.live_generation.scope === 'request'
        && event.accounting.request.request_key === event.live_generation.request_key && (event.live_generation.backend_running ?? 0) <= 1) upsert(event.accounting)
      else for (const [key, value] of records) if (value.request.status === 'running') records.delete(key)
      const finals = new Map((queued ? pending?.finalized ?? [] : []).map(value => [value.request.request_key, value]))
      for (const value of event.finalized ?? []) finals.set(value.request.request_key, value)
      observed = event.observed_at; pending = { ...event, finalized: [...finals.values()] }; notify(true)
    },
    accounting: selectAccounting,
    readLatest(request: MonitorRequestView | undefined, identity: GenerationEventIdentity | undefined, phase: GenerationPhase) {
      return { event: pending, accounting: selectAccounting(request, pending, records), rate: selectRate(pending, identity, phase, records) }
    },
    identity(request: MonitorRequestView | undefined): GenerationEventIdentity | undefined {
      return request?.status === 'running' && runId && requestKey && (request.requestKey === requestKey || request.id === clientCallId)
        ? { runId, instanceId, requestKey } : undefined
    },
    turn(requests: readonly MonitorRequestView[]) {
      let value: GenerationAccounting | null = null
      for (const request of requests) value = store.accounting(request) ?? value
      return value
    },
    rate: selectRate,
  }
  function selectAccounting(request: MonitorRequestView | undefined, current = snapshot, values = published): GenerationAccounting | null {
      if (!request || request.status !== 'running' || !runId || current && current.run_id !== runId) return null
      if (request.external && (request.external.instance_id !== instanceId || request.external.run_id !== runId)) return null
      const value = values.get(request.requestKey ?? request.id) ?? [...values.values()].find(value => value.turn.source === 'internal' && value.request.client_call_id === request.id)
      if (!value || value.request.status === 'running' && (!current || current.observed_at < pollAt || value.request.request_key !== requestKey)) return null
      return value
  }
  function selectRate(event: GenerationEvent | null, identity: GenerationEventIdentity | undefined, phase: GenerationPhase, values = published): number | null | undefined {
      if (!identity || identity.instanceId !== instanceId || identity.runId !== runId || identity.requestKey !== requestKey) return undefined
      const final = values.get(identity.requestKey)
      if (final && final.request.status !== 'running') return null
      if (!event || event.run_id !== identity.runId || event.instance_id !== identity.instanceId || event.observed_at < pollAt) return undefined
      const live = event.live_generation
      if (!live) return null
      if (live.request_key !== identity.requestKey) return undefined
      if (live.scope !== 'request' || (live.backend_running ?? 0) > 1) return null
      const decoding = ['generating', 'decode', 'thinking', 'output'].includes(live.phase ?? '')
      if (phase === 'prefill' && decoding || phase === 'decode' && live.phase === 'prefill') return undefined
      if (!decoding && live.phase !== 'prefill') return null
      const value = phase === 'decode' ? live.tokens_per_second : live.prefill_tokens_per_second
      return typeof value === 'number' && Number.isFinite(value) ? value : undefined
  }
  return store
}
export type GenerationEventStore = ReturnType<typeof createGenerationEventStore>
export function useGenerationAccounting(store: GenerationEventStore | undefined, requests: readonly MonitorRequestView[]) {
  return useSyncExternalStore(store?.subscribe ?? noSubscribe, () => store?.turn(requests) ?? null, noSnapshot)
}
export function useGenerationEventRate(store: GenerationEventStore | undefined, identity: GenerationEventIdentity | undefined, phase: GenerationPhase) {
  const event = useSyncExternalStore(store?.subscribe ?? noSubscribe, store?.getSnapshot ?? noSnapshot, noSnapshot)
  return store?.rate(event, identity, phase)
}

export async function readGenerationEvents(instanceId: string, signal: AbortSignal, receive: (event: GenerationEvent) => void) {
  const response = await apiFetch(`/api/instances/${encodeURIComponent(instanceId)}/generation-events`, { signal, headers: { Accept: 'text/event-stream' } })
  if (!response.headers.get('content-type')?.includes('text/event-stream') || !response.body) throw new Error('生成事件流不可用')
  const reader = response.body.getReader(), decoder = new TextDecoder()
  let buffer = ''
  const consume = (final = false) => {
    const events = buffer.replace(/\r\n/g, '\n').split('\n\n')
    buffer = final ? '' : events.pop() ?? ''
    for (const event of events) {
      const lines = event.split('\n')
      if (!lines.some(line => line.trim() === 'event: generation')) continue
      const data = lines.filter(line => line.startsWith('data:')).map(line => line.slice(5).trimStart()).join('\n')
      if (data) receive(JSON.parse(data) as GenerationEvent)
    }
  }
  try {
    while (!signal.aborted) {
      const result = await reader.read()
      buffer += decoder.decode(result.value, { stream: !result.done })
      consume(result.done)
      if (result.done) break
    }
  } finally { await reader.cancel().catch(() => {}); reader.releaseLock() }
}

// Polling remains available during a dropped stream. Bound automatic recovery;
// changing instance or manually refreshing also opens a new connection.
export async function followGenerationEvents(instanceId: string, signal: AbortSignal, store: GenerationEventStore) {
  for (let attempt = 0; attempt < 3 && !signal.aborted; attempt++) {
    let unavailable = false
    try { await readGenerationEvents(instanceId, signal, event => { if (!signal.aborted) store.receive(event) }) }
    catch (error) { unavailable = error instanceof ApiError && error.status > 0 && error.status < 500 }
    finally { if (!signal.aborted) store.clear() }
    if (unavailable || signal.aborted || attempt === 2) return
    await new Promise<void>(resolve => {
      const finish = () => { clearTimeout(timer); signal.removeEventListener('abort', finish); resolve() }
      const timer = setTimeout(finish, 1000)
      signal.addEventListener('abort', finish, { once: true })
    })
  }
}
