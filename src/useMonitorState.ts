import { useCallback, useEffect, useRef, useState } from 'react'
import type { AppState, ChatGenerationActivity, Telemetry } from './types'
import { apiFetch, errorMessage, GENERATION_REFRESH_INTERVAL_MS, isActive, isReady, REFRESH_INTERVAL_MS } from './api'
import { generationView, monitorGenerationView, monitorInstanceId, rememberChatStream, rememberGeneration, type GenerationMemory } from './monitor-generation'
import { gpuKey, gpuList } from './resource-metrics'
import { createGenerationEventStore, followGenerationEvents } from './monitor-generation-events'
import { createExternalTimelineStore } from './external-generation-timeline'

// One sampler belongs to the main App, regardless of which window/page is visible.
export function useMonitorState(state: AppState, refresh: () => Promise<void>) {
  const generationMemory = useRef(new Map<string, GenerationMemory>())
  const generationEventsRef = useRef<ReturnType<typeof createGenerationEventStore> | null>(null)
  if (!generationEventsRef.current) generationEventsRef.current = createGenerationEventStore()
  const generationEvents = generationEventsRef.current
  const externalTimelineRef = useRef<ReturnType<typeof createExternalTimelineStore> | null>(null)
  if (!externalTimelineRef.current) externalTimelineRef.current = createExternalTimelineStore()
  const timeline = externalTimelineRef.current
  const [eventVersion, setEventVersion] = useState(0)
  const [chatActivities, setChatActivities] = useState<ChatGenerationActivity[]>([])
  const [inspectionVersion, setInspectionVersion] = useState(0)
  const [selectedTurn, setSelectedTurn] = useState<ChatGenerationActivity | null>(null)
  const reportChatActivity = useCallback((activity: ChatGenerationActivity) => {
    generationEvents.confirmFinalized((activity.requests ?? []).filter(request => request.requestKey).map(request => ({ request_key: request.requestKey!, status: request.status })), activity.instanceId)
    generationMemory.current.set(activity.instanceId, rememberChatStream(generationMemory.current.get(activity.instanceId), activity))
    setChatActivities(previous => {
      const known = previous.find(item => item.messageId === activity.messageId && item.instanceId === activity.instanceId)
      return [...previous.filter(item => item.messageId !== activity.messageId
        && (item.instanceId !== activity.instanceId || !['completed', 'cancelled', 'failed'].includes(item.phase))),
      { ...activity, contextCapacity: activity.contextCapacity ?? known?.contextCapacity }]
    })
    setSelectedTurn(previous => previous?.messageId === activity.messageId && previous.instanceId === activity.instanceId
      ? { ...activity, contextCapacity: activity.contextCapacity ?? previous.contextCapacity } : previous)
  }, [])
  const [chosenInstance, setChosenInstance] = useState('')
  const explicitInstance = useRef('')
  const setInstanceId = useCallback((id: string) => { explicitInstance.current = id; setChosenInstance(id); setSelectedTurn(null) }, [])
  const showChatTurn = useCallback((activity: ChatGenerationActivity, navigationInstanceId = activity.instanceId) => {
    setSelectedTurn(activity)
    setInspectionVersion(version => version + 1)
    if (navigationInstanceId) { explicitInstance.current = navigationInstanceId; setChosenInstance(navigationInstanceId) }
  }, [])
  const followLatest = useCallback(() => setSelectedTurn(null), [])
  const [device, setDevice] = useState('')
  const [telemetry, setTelemetry] = useState<Telemetry | null>(null)
  const [error, setError] = useState('')
  const [initialLoading, setInitialLoading] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const readTelemetry = useRef<(() => Promise<void>) | null>(null)
  const manualRequest = useRef<symbol | null>(null)
  const knownContextCapacity = useCallback((id: string) => generationMemory.current.get(id)?.capacity ?? null, [])
  const historyMemory = useCallback((runId: string, id: string) => {
    const memory = generationMemory.current.get(id)
    return memory?.gatewayRunId === runId ? memory : undefined
  }, [])
  const instances = state.instances.filter(item => isActive(item.state))
  const instanceId = monitorInstanceId(state.instances, chosenInstance, explicitInstance.current)
  const instance = state.instances.find(item => item.id === instanceId)
  const ready = !!instance && isReady(instance.state)
  const samplingInterval = (instance?.active_requests ?? 0) > 0 || chatActivities.some(activity => activity.instanceId === instanceId && activity.requests?.some(request => request.status === 'running'))
    ? GENERATION_REFRESH_INTERVAL_MS : REFRESH_INTERVAL_MS
  const telemetryScope = `${instanceId}:${instance?.state ?? 'none'}`
  const [sampleScope, setSampleScope] = useState('')
  useEffect(() => {
    generationEvents.reset(instanceId)
    timeline.select(instanceId)
    if (!instanceId || !ready) return
    const controller = new AbortController()
    void followGenerationEvents(instanceId, controller.signal, { ...generationEvents,
      receive(event) { timeline.receive(event); generationEvents.receive(event) },
      clear() { generationEvents.clear(); timeline.unavailable(Date.now(), 'events') },
    })
    return () => { controller.abort(); generationEvents.clear() }
  }, [instanceId, ready, telemetryScope, eventVersion, generationEvents])
  useEffect(() => { if (chosenInstance !== instanceId) setChosenInstance(instanceId) }, [chosenInstance, instanceId])
  useEffect(() => {
    manualRequest.current = null
    setRefreshing(false); setInitialLoading(false)
    setTelemetry(null); setError(''); setSampleScope(telemetryScope)
    if (!instanceId || !ready) return
    let cancelled = false
    let inFlight: Promise<void> | null = null
    const controller = new AbortController()
    setInitialLoading(true)
    async function read() {
      if (inFlight) return inFlight
      inFlight = (async () => {
        try {
          const response = await apiFetch(`/api/instances/${instanceId}/telemetry`, { signal: controller.signal })
          const payload = await response.json() as Telemetry
          if (!cancelled) {
            if (payload.instance_id !== instanceId) throw new Error('监控响应与当前实例不匹配')
            if (isReady(payload.state)) generationMemory.current.set(instanceId, rememberGeneration(generationMemory.current.get(instanceId), payload))
            timeline.confirm(payload)
            generationEvents.confirm(payload)
            setTelemetry(payload); setError('')
          }
        } catch (issue) { if (!cancelled && !controller.signal.aborted) { timeline.unavailable(Date.now(), 'poll'); setError(errorMessage(issue)) } }
      })()
      try { await inFlight } finally { inFlight = null }
    }
    readTelemetry.current = read
    void read().finally(() => { if (!cancelled) setInitialLoading(false) })
    return () => {
      cancelled = true; controller.abort()
      if (readTelemetry.current === read) readTelemetry.current = null
    }
  }, [instanceId, ready, telemetryScope])
  useEffect(() => {
    const timer = window.setInterval(() => void readTelemetry.current?.(), samplingInterval)
    return () => window.clearInterval(timer)
  }, [samplingInterval])

  async function refreshMonitor() {
    if (manualRequest.current) return
    const request = Symbol()
    manualRequest.current = request
    setRefreshing(true)
    setEventVersion(value => value + 1)
    try { await Promise.all([readTelemetry.current?.(), refresh()]) }
    finally { if (manualRequest.current === request) { manualRequest.current = null; setRefreshing(false) } }
  }
  const payload = ready && sampleScope === telemetryScope && telemetry?.instance_id === instanceId && isReady(telemetry.state) ? telemetry : null
  const sampleError = sampleScope === telemetryScope ? error : ''
  const sampleSettled = sampleScope === telemetryScope && (telemetry?.instance_id === instanceId || !!error)
  const memory = instance ? generationMemory.current.get(instanceId) : undefined
  const view = generationView(payload, memory, ready, instance?.active_requests ?? 0, !!sampleError)
  const liveGeneration = monitorGenerationView(payload, memory, instance, !!sampleError, chatActivities)
  const generation = selectedTurn ? monitorGenerationView(payload, memory, instance, !!sampleError, chatActivities, selectedTurn) : liveGeneration
  const gpus = gpuList(state.hardware), gpu = gpus.find(g => gpuKey(g) === device) ?? gpus[0]
  return { instances, instance, instanceId, setInstanceId, device, setDevice, gpus, gpu,
    reportChatActivity, showChatTurn, followLatest, selectedTurn, inspectionVersion, knownContextCapacity, historyMemory, payload, live: payload?.live_generation, sampleError, sampleSettled, memory, view, generation, liveGeneration, ready, telemetryScope,
    generationEvents, externalTimeline: timeline, refreshing, loading: initialLoading || refreshing, refreshMonitor }
}

export type MonitorState = ReturnType<typeof useMonitorState>

