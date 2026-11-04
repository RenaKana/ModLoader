import type { ExternalRequestRecord, GenerationAccounting, GenerationEvent, LiveGeneration, Telemetry } from './types'
import type { GenerationPhase, GenerationPoint } from './monitor-generation'
import { finite, HISTORY_MS } from './resource-metrics'
import { retainChartWindow, windowSegments } from './monitor-chart-window'
import { REFRESH_INTERVAL_MS } from './api'

export type ExternalTimelineState = 'idle' | 'queued' | 'waiting' | 'prefill' | 'decode' | 'concurrent' | 'unknown'
type Observation = { time: number; state: ExternalTimelineState; requestKey?: string; value: number | null; source: 'poll' | 'events' }
type TimelineRequest = { record: ExternalRequestRecord; observedAt: number; finalizedAt?: number
  samples: Record<GenerationPhase, GenerationPoint[]>; sampled: Record<GenerationPhase, boolean> }
export type ExternalTimelineSnapshot = { instanceId: string; runId: string; requests: TimelineRequest[]; observations: Observation[]; current: Observation }
const empty = (instanceId: string): ExternalTimelineSnapshot => ({ instanceId, runId: '', requests: [], observations: [], current: { time: 0, state: 'unknown', value: null, source: 'poll' } })
const terminal = (record: ExternalRequestRecord) => record.status !== 'running'
const positiveTime = (value: unknown) => finite(value) !== null && (value as number) >= 0 ? value as number : undefined
const phaseOf = (live: LiveGeneration | null): GenerationPhase | undefined => live?.phase === 'prefill' ? 'prefill'
  : ['generating', 'decode', 'thinking', 'output'].includes(live?.phase ?? '') ? 'decode' : undefined
const validRate = (value: unknown) => finite(value) !== null && (value as number) >= 0 ? value as number : null
// The normal poll is one second. Three missed polls also break the existing
// 2.5-second chart continuity; a stalled channel cannot keep a live readout.
const FRESH_MS = REFRESH_INTERVAL_MS * 3

/** Independent chart history: the accounting store may release a final after a poll consumes it. */
export function createExternalTimelineStore(frame: (callback: FrameRequestCallback) => number = callback => requestAnimationFrame(callback)) {
  const instances = new Map<string, ExternalTimelineSnapshot>()
  const listeners = new Set<() => void>()
  let current = empty(''), queued = false
  function publish(next: ExternalTimelineSnapshot) {
    current = next; instances.set(next.instanceId, next)
    if (queued) return
    queued = true
    frame(() => { queued = false; listeners.forEach(listener => listener()) })
  }
  function ingest(records: ExternalRequestRecord[], time: number, live: LiveGeneration | null, source: Observation['source']) {
    const requests = new Map(current.requests.map(request => [request.record.request_key, request]))
    for (const record of records) {
      if (record.run_id !== current.runId || record.instance_id !== current.instanceId) continue
      const old = requests.get(record.request_key)
      if (old && terminal(old.record) && !terminal(record)) continue
      if (old && old.observedAt > time && !terminal(record)) continue
      requests.set(record.request_key, { ...old, record, observedAt: Math.max(time, old?.observedAt ?? 0),
        ...(terminal(record) ? { finalizedAt: old?.finalizedAt ?? time } : {}),
        samples: old?.samples ?? { decode: [], prefill: [] }, sampled: old?.sampled ?? { decode: false, prefill: false } })
    }
    const known = live?.request_key ? requests.get(live.request_key) : undefined
    const unambiguous = live?.scope === 'request' && (live.backend_running ?? 0) <= 1
    let state: ExternalTimelineState = 'unknown', value: number | null = null
    const phase = phaseOf(live)
    if (source === 'poll' && live?.backend_running === 0 && live.backend_queued === 0 && (live.backend_paused ?? 0) === 0) state = 'idle'
    else if ((live?.backend_running ?? 0) > 1) state = 'concurrent'
    else if ((live?.backend_queued ?? 0) > 0 && live?.backend_running === 0 || live?.phase === 'queued') state = 'queued'
    else if (known && !terminal(known.record) && unambiguous) {
      state = phase ?? 'waiting'
      value = phase ? validRate(phase === 'decode' ? live!.tokens_per_second : live!.prefill_tokens_per_second) : null
    }
    const observation: Observation = { time, state, value, source, ...(known ? { requestKey: known.record.request_key } : {}) }
    const ordered = time >= current.current.time
    if (ordered && known && !terminal(known.record) && unambiguous && phase) {
      const old = known.samples[phase], segment = `${known.record.request_key}:${phase}`
      const point: GenerationPoint = { time, wallTime: time, values: { current: value }, segment }
      const same = old.at(-1)?.time === time
      const samples = same ? [...old.slice(0, -1), point] : [...old, point]
      requests.set(known.record.request_key, { ...known, samples: { ...known.samples, [phase]: samples },
        sampled: { ...known.sampled, [phase]: known.sampled[phase] || value !== null } })
    }
    // A phase/wait boundary is a real gap, including a pause and resume of the same request.
    if (ordered && current.current.requestKey && (current.current.requestKey !== observation.requestKey || current.current.state !== observation.state)) {
      const old = requests.get(current.current.requestKey), oldPhase = current.current.state
      if (old && (oldPhase === 'decode' || oldPhase === 'prefill')) {
        const samples = old.samples[oldPhase]
        if (samples.length && samples.at(-1)!.time < time) requests.set(old.record.request_key, { ...old,
          samples: { ...old.samples, [oldPhase]: [...samples, { time, wallTime: time, values: { current: null }, segment: `${old.record.request_key}:${oldPhase}` }] } })
      }
    }
    const latest = ordered ? observation : current.current, cutoff = latest.time - HISTORY_MS
    const retained = [...requests.values()].filter(request => !terminal(request.record)
      || (positiveTime(request.record.ended_at) !== undefined ? request.record.ended_at! * 1000 : request.finalizedAt ?? request.observedAt) >= cutoff)
      .map(request => ({ ...request, samples: Object.fromEntries((['decode', 'prefill'] as const).map(phase => {
        const retained = retainChartWindow(request.samples[phase], cutoff)
        return [phase, [...(retained.boundaryPoint ? [retained.boundaryPoint] : []), ...retained.points]]
      })) as TimelineRequest['samples'] }))
    let observations = current.observations
    if (ordered) {
      const previous = observations.at(-1)
      // State bands need transitions, not thousands of identical event frames.
      if (!previous || previous.state !== state || previous.requestKey !== observation.requestKey) observations = [...observations, observation]
      const retained = retainChartWindow(observations.map(item => ({ ...item, values: {} })), cutoff)
      observations = [...(retained.boundaryPoint ? [retained.boundaryPoint] : []), ...retained.points]
    }
    publish({ ...current, requests: retained, observations, current: latest })
  }
  const fromAccounting = (value: GenerationAccounting, event: GenerationEvent): ExternalRequestRecord => ({ ...value.request,
    run_id: event.run_id, instance_id: event.instance_id, revision: 0, alias: '', backend: '', labels: [], aggregation: 'request', turn_id: null, turn_state: null })
  return {
    subscribe(listener: () => void) { listeners.add(listener); return () => { listeners.delete(listener) } },
    getSnapshot: () => current,
    select(instanceId: string) { if (instanceId !== current.instanceId) publish(instances.get(instanceId) ?? empty(instanceId)) },
    confirm(payload: Telemetry) {
      if (payload.instance_id !== current.instanceId) return
      const runId = payload.external_history?.run_id
      if (runId && current.runId !== runId) current = empty(payload.instance_id)
      current = { ...current, runId: runId ?? current.runId }
      const time = typeof payload.updated_at === 'number' ? payload.updated_at * 1000 : Date.parse(payload.updated_at)
      if (!Number.isFinite(time)) return
      ingest(payload.external_history?.requests ?? [], time, payload.live_generation ?? null, 'poll')
    },
    receive(event: GenerationEvent) {
      if (event.instance_id !== current.instanceId || !event.run_id || current.runId && event.run_id !== current.runId || !Number.isFinite(event.observed_at)) return
      if (event.live_generation?.request_key && !event.accounting && !event.finalized?.length && (event.live_generation.backend_running ?? 0) <= 1
        && !current.requests.some(request => request.record.request_key === event.live_generation!.request_key)) return
      if (!current.runId) current = { ...current, runId: event.run_id }
      const values = [...(event.accounting ? [event.accounting] : []), ...(event.finalized ?? [])].filter(value => value.turn.source === 'external')
      ingest(values.map(value => fromAccounting(value, event)), event.observed_at * 1000, event.live_generation, 'events')
    },
    unavailable(time: number, source: Observation['source']) {
      if (source === 'events' && current.current.source !== 'events') return
      ingest([], Math.max(time, current.current.time), null, source)
    },
  }
}
export type ExternalTimelineStore = ReturnType<typeof createExternalTimelineStore>

function phaseAverage(record: ExternalRequestRecord, phase: GenerationPhase) {
  const measured = record.measurements?.[phase]
  if (measured?.coverage === 'partial') return null
  const timings = record.timings, milliseconds = phase === 'decode' ? timings.predicted_ms : timings.prompt_ms
  // A reported zero clock is not an interval, even if an adjacent rate field exists.
  if (milliseconds !== undefined && (!(milliseconds > 0) || !Number.isFinite(milliseconds))) return null
  const tokens = measured?.coverage === 'complete' ? validRate(measured.tokens) : validRate(phase === 'decode'
    ? timings.predicted_n ?? record.usage.completion_tokens : timings.prompt_n ?? record.prefill?.prefill_processed_tokens)
  const seconds = measured?.coverage === 'complete' ? finite(measured.seconds) : milliseconds === undefined ? null : milliseconds / 1000
  return tokens !== null && seconds !== null && seconds > 0 ? { value: tokens / seconds, tokens, seconds } : null
}

const axis = (values: number[]) => {
  let peak = 1
  for (const value of values) peak = Math.max(peak, value)
  const target = peak * 1.1 / 3, magnitude = 10 ** Math.floor(Math.log10(target))
  const step = [1, 2, 5, 10].find(value => value * magnitude >= target)! * magnitude
  return { maximum: step * 3, ticks: [0, step, step * 2, step * 3] }
}

/** Geometry only: no averages, zeros or phase boundaries are inserted into sampled series. */
export function projectExternalTimeline(snapshot: ExternalTimelineSnapshot, end: number, windowSeconds: number, width = 600) {
  const start = end - windowSeconds * 1000, span = end - start
  const series: Record<GenerationPhase, { requestKey: string; points: { time: number; value: number }[]; marker: boolean }[]> = { decode: [], prefill: [] }
  const averages: { requestKey: string; phase: GenerationPhase; time: number; value: number; tokens: number; seconds: number }[] = []
  const requests: { requestKey: string; time: number; status: ExternalRequestRecord['status']; startedAt?: number }[] = []
  for (const request of snapshot.requests) {
    const record = request.record, endpoint = terminal(record) ? positiveTime(record.ended_at) === undefined ? request.finalizedAt : record.ended_at! * 1000
      : positiveTime(record.started_at) === undefined ? undefined : record.started_at! * 1000
    if (endpoint !== undefined && endpoint >= start && endpoint <= end) requests.push({ requestKey: record.request_key, time: endpoint, status: record.status,
      ...(positiveTime(record.started_at) === undefined ? {} : { startedAt: record.started_at! * 1000 }) })
    for (const phase of ['decode', 'prefill'] as const) {
      for (const points of windowSegments(request.samples[phase], start, end)) series[phase].push({ requestKey: record.request_key, points,
        marker: points.length === 1 || (points.at(-1)!.time - points[0].time) / span * width < 8 })
      if (!request.sampled[phase] && terminal(record) && endpoint !== undefined && endpoint >= start && endpoint <= end) {
        const average = phaseAverage(record, phase)
        if (average) averages.push({ requestKey: record.request_key, phase, time: endpoint, ...average })
      }
    }
  }
  const axes = { decode: axis([...series.decode.flatMap(segment => segment.points.map(point => point.value)), ...averages.filter(marker => marker.phase === 'decode').map(marker => marker.value)]),
    prefill: axis([...series.prefill.flatMap(segment => segment.points.map(point => point.value)), ...averages.filter(marker => marker.phase === 'prefill').map(marker => marker.value)]) }
  const stale = end - snapshot.current.time > FRESH_MS
  const finalizedCurrent = snapshot.current.requestKey && snapshot.requests.some(request => request.record.request_key === snapshot.current.requestKey && terminal(request.record))
  const state = stale || finalizedCurrent && snapshot.current.state !== 'idle' ? 'unknown' : snapshot.current.state
  const readouts = { decode: state === 'idle' ? 0 : state === 'decode' ? snapshot.current.value : null,
    prefill: state === 'idle' ? 0 : state === 'prefill' ? snapshot.current.value : null }
  const states = snapshot.observations.map((observation, index) => ({ state: observation.state, since: observation.time, start: Math.max(start, observation.time),
    end: Math.min(end, snapshot.observations[index + 1]?.time ?? Math.min(end, snapshot.current.time + FRESH_MS)) }))
    .filter(state => state.start < state.end)
  return { start, end, series, averages, requests, axes, state, readouts, states }
}
