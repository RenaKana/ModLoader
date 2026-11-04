import { useEffect, useImperativeHandle, useRef, useState, type ReactNode, type Ref } from 'react'
import { ChevronLeft, ChevronRight, Loader2 } from 'lucide-react'
import { apiFetch, errorMessage, isReady } from './api'
import { Button, DiagnosticNotice, ErrorNotice, Select } from './components'
import { externalGenerationView, externalRequestIdentity, monitorGenerationView, updatedHistorySelection, type MonitorGenerationView, type MonitorRequestView } from './monitor-generation'
import { historyBackend, historyDateRange, historyInstanceIdentity, historyInstances, historyInstanceMatches, localDate, type HistoryDateMode, type HistoryInstance } from './monitor-history-dates'
import { historyRequestGroups, type MonitorRequestTableEntry } from './monitor-request-groups'
import { readMonitorHistoryPage } from './monitor-history-page'
import { turnTotals } from './chat-turn-metrics'
import { currentHistoryInRange, historyDirectoryIsLive, canFollowHistoryRequest, confirmHistoryFollow, type PendingHistoryFollow } from './monitor-history-policy'
import { MonitorGeneration } from './MonitorGeneration'
import { MonitorRequestTable } from './MonitorRequestTable'
import { MonitorUsage } from './MonitorUsage'
import type { ExternalHistorySnapshot, ExternalRequestRecord, InternalRequestHistoryGroup, RequestHistoryPage, RequestHistoryDates, RequestHistoryGroup } from './types'
import type { MonitorState } from './useMonitorState'
import './monitor-request-history.css'

const PAGE_SIZE = 50

export { requireGroupedHistoryPage } from './monitor-history-page'
const externalKey = externalRequestIdentity
const internalKey = (group: InternalRequestHistoryGroup) => JSON.stringify(['internal', group.conversation_id, group.message_id, group.instance_id])
const snapshot = (requests: ExternalRequestRecord[]): ExternalHistorySnapshot => {
  const last = [...requests].sort((a, b) => (a.started_at ?? 0) - (b.started_at ?? 0)).at(-1)
  return { run_id: last?.run_id ?? '', aggregation: 'request', turn_id: null, state: last?.status ?? 'completed', requests }
}
type Source = { key: string; view: MonitorGenerationView; entry: MonitorRequestTableEntry; external?: ExternalRequestRecord; internal?: InternalRequestHistoryGroup }
type Selection = { kind: 'follow'; instance?: HistoryInstance } | { kind: 'pinned'; source: Source; scope: 'turn' | 'request'; requestId?: string; awaitLiveInstance?: PendingHistoryFollow } | { kind: 'inspection' }
export type MonitorRequestHistoryHandle = { refresh: () => void }

// Historical pins are presentation-only; choosing a running request may hand the existing sampler to its instance.
export function MonitorRequestHistory({ monitor, active = true, ref, hardware, hardwareControls }: { monitor: MonitorState; active?: boolean; ref?: Ref<MonitorRequestHistoryHandle>; hardware?: ReactNode; hardwareControls?: ReactNode }) {
  const currentLive = useRef(monitor.liveGeneration)
  currentLive.current = monitor.liveGeneration
  function releaseConsumedRecords(records: ExternalRequestRecord[]) {
    const shadowed = new Set(currentLive.current.requests.filter(request => request.status === 'running' && request.external)
      .map(request => externalRequestIdentity(request.external!)))
    monitor.generationEvents?.confirmFinalized(records.filter(record => !shadowed.has(externalRequestIdentity(record))))
  }
  const [catalog, setCatalog] = useState<RequestHistoryDates | null>(null)
  const [mode, setMode] = useState<HistoryDateMode>('today')
  const [from, setFrom] = useState(localDate), [to, setTo] = useState(localDate)
  const [instance, setInstance] = useState<HistoryInstance | null>(null)
  const [offset, setOffset] = useState(0), [page, setPage] = useState<RequestHistoryPage | null>(null)
  const [selection, setSelection] = useState<Selection>({ kind: 'follow' })
  const [expanded, setExpanded] = useState(new Map<string, boolean>())
  const [groups, setGroups] = useState(new Map<string, RequestHistoryPage>())
  const [groupErrors, setGroupErrors] = useState(new Map<string, string>())
  const [groupLoading, setGroupLoading] = useState(new Set<string>())
  const [catalogError, setCatalogError] = useState(''), [pageError, setPageError] = useState('')
  const [loading, setLoading] = useState(true), [readVersion, setReadVersion] = useState(0)
  const epoch = useRef(0), wasActive = useRef(false), consumedInspection = useRef(0)
  const requestMode = useRef<boolean | undefined>(undefined)
  const retry = () => setReadVersion(value => value + 1)
  const today = catalog?.today ?? localDate()
  const range = historyDateRange(mode, today, from, to)
  const options = historyInstances(catalog, range.from, range.to)
  const instanceValue = instance ? historyInstanceIdentity(instance) : ''
  const livePage = historyDirectoryIsLive(instance, catalog?.current_run_id, range.from, range.to, today)
  const usageFilter = new URLSearchParams({ date_from: range.from, date_to: range.to })
  if (instance) {
    usageFilter.set('source', instance.source === 'internal' ? 'internal' : 'external')
    if (instance.id) usageFilter.set('instance_id', instance.id)
    if (instance.source === 'internal') { if (!instance.id && instance.conversation_id) usageFilter.set('conversation_id', instance.conversation_id) }
    else if (instance.run_id) usageFilter.set('run_id', instance.run_id)
  }

  function follow(target?: HistoryInstance) {
    setSelection({ kind: 'follow', instance: target }); monitor.followLatest()
    setMode('today'); setFrom(today); setTo(today); setInstance(target ?? null); setOffset(0)
  }
  function reset() {
    epoch.current++; follow()
    setPage(null); setExpanded(new Map()); setGroups(new Map()); setGroupErrors(new Map()); setGroupLoading(new Set()); retry()
  }
  useImperativeHandle(ref, () => ({ refresh: reset }))
  useEffect(() => {
    const entering = active && !wasActive.current
    wasActive.current = active
    if (!active) return
    if (monitor.inspectionVersion !== consumedInspection.current && monitor.selectedTurn) {
      consumedInspection.current = monitor.inspectionVersion
      setSelection({ kind: 'inspection' }); setInstance(null); setOffset(0)
      const start = monitor.selectedTurn.turn?.startedAt ?? monitor.selectedTurn.requests?.[0]?.startedAt
      if (start) { const date = localDate(new Date(start)); setMode('date'); setFrom(date); setTo(date) }
    } else if (entering) follow()
  }, [active, monitor.inspectionVersion])

  useEffect(() => {
    if (!active) return
    const controller = new AbortController(), version = epoch.current
    let reading = false
    async function read() {
      if (reading) return
      reading = true
      try {
        const response = await apiFetch('/api/request-history/dates', { signal: controller.signal })
        const result = await response.json() as RequestHistoryDates
        if (!controller.signal.aborted && version === epoch.current) { setCatalog(result); setCatalogError('') }
      } catch (issue) { if (!controller.signal.aborted && version === epoch.current) { setCatalogError(errorMessage(issue)); setLoading(false) } }
      finally { reading = false }
    }
    void read()
    const timer = window.setInterval(() => void read(), 5000)
    return () => { controller.abort(); window.clearInterval(timer) }
  }, [active, readVersion])
  useEffect(() => {
    if (!active) return
    setPage(null); setPageError(''); setLoading(!range.error)
    if (!catalog || range.error) return
    const controller = new AbortController(), version = epoch.current
    let reading = false
    async function read() {
      if (reading) return
      reading = true
      const query = new URLSearchParams({ group_by: 'turn', date_from: range.from, date_to: range.to, offset: String(offset), limit: String(PAGE_SIZE) })
      if (instance) {
        query.set('source', instance.source === 'internal' ? 'internal' : 'external')
        if (instance.id) query.set('instance_id', instance.id)
        if (instance.source === 'internal') { if (!instance.id && instance.conversation_id) query.set('conversation_id', instance.conversation_id) }
        else if (instance.run_id) query.set('run_id', instance.run_id)
      }
      try {
        const result = await readMonitorHistoryPage(query, async current => {
          const response = await apiFetch(`/api/request-history?${current}`, { signal: controller.signal })
          return await response.json() as RequestHistoryPage
        }, requestMode)
        if (!controller.signal.aborted && version === epoch.current) { setPage(result); setPageError(''); releaseConsumedRecords(result.items) }
      } catch (issue) { if (!controller.signal.aborted && version === epoch.current) setPageError(errorMessage(issue)) }
      finally { reading = false; if (!controller.signal.aborted && version === epoch.current) setLoading(false) }
    }
    void read()
    const timer = livePage ? window.setInterval(() => void read(), 1500) : null
    return () => { controller.abort(); if (timer !== null) window.clearInterval(timer) }
  }, [active, !!catalog, range.from, range.to, range.error, instanceValue, offset, livePage, readVersion])

  function externalView(records: ExternalRequestRecord[]) {
    const record = records[0], memory = record ? monitor.historyMemory(record.run_id, record.instance_id) : undefined
    return externalGenerationView(snapshot(records), memory, record && memory && record.instance_id === monitor.instanceId ? monitor.payload : null, true)
  }
  function internalView(group: InternalRequestHistoryGroup) {
    const sameInstance = group.instance_id === monitor.instanceId
    const inspected = monitor.selectedTurn?.messageId === group.message_id && monitor.selectedTurn.instanceId === (group.instance_id ?? '') ? monitor.selectedTurn : undefined
    const retained = selection.kind === 'pinned' && selection.source.key === internalKey(group) ? selection.source.view : undefined
    const status = group.status === 'recovered' ? 'completed' : group.status
    const activity = { instanceId: group.instance_id ?? '', messageId: group.message_id, model: group.alias,
      phase: (status === 'running' || status === 'waiting' ? 'generating' : status === 'incomplete' ? 'cancelled' : status) as 'generating' | 'completed' | 'cancelled' | 'failed',
      requests: group.requests, contextCapacity: group.context_capacity, toolActivity: inspected?.toolActivity ?? retained?.tools, toolCount: inspected?.toolCount ?? retained?.toolCount,
      turn: turnTotals(group.requests, group.started_at * 1000, group.elapsed_s, status === 'waiting' ? 'running' : status, group.ended_at == null ? undefined : group.ended_at * 1000) }
    const view = monitorGenerationView(sameInstance ? monitor.payload : null, sameInstance ? monitor.memory : undefined, sameInstance ? monitor.instance : undefined, false, [], activity)
    const live = monitor.generation
    // A persisted request is replaced by its matching live record, never added twice.
    if (live.origin === 'internal' && live.sessionKey.endsWith(`:${group.message_id}`) && sameInstance && live.requests.some(request => group.requests.some(stored => stored.id === request.id))) {
      const requests = [...new Map([...view.requests, ...live.requests].map(request => [request.id, request])).values()].sort((a, b) => a.startedAt - b.startedAt)
      return { ...live, tools: live.tools.length ? live.tools : view.tools, toolCount: live.toolCount ?? view.toolCount, requests, turn: turnTotals(requests, group.started_at * 1000, live.turn?.elapsed ?? group.elapsed_s, live.turn?.status ?? activity.turn.status) }
    }
    return view
  }
  const live = monitor.generation
  const liveExternal = live.requests.flatMap(request => request.external ? [request.external] : [])
  const liveExternalKey = liveExternal[0] ? externalKey(liveExternal[0]) : undefined
  const sources = new Map<string, Source>()
  const isExpanded = (key: string) => expanded.get(key) ?? false
  function externalSource(record: ExternalRequestRecord, summary?: RequestHistoryGroup): Source {
    const key = externalKey(record)
    const cached = groups.get(key)
    const current = liveExternalKey === key ? live : undefined
    const view = current ?? externalView([updatedHistorySelection(record, cached?.items ?? []) ?? record])
    return { key, view, external: record, entry: { key, turn: false, requests: view.requests, representative: view.requests[0],
      alias: record.alias, backend: record.backend, startedAt: (record.started_at ?? 0) * 1000,
      status: view.status, count: 1, summary: current ? undefined : summary, elapsed: view.requests[0]?.elapsed,
      loading: groupLoading.has(key), error: groupErrors.get(key) } }
  }
  for (const entry of historyRequestGroups(page?.items ?? [], page?.groups)) sources.set(entry.key, externalSource(entry.records[0], entry.summary))
  for (const group of page?.internal_groups ?? []) {
    const key = internalKey(group), view = internalView(group)
    const overlaid = view.requests !== group.requests && live.origin === 'internal' && live.sessionKey.endsWith(`:${group.message_id}`) && group.instance_id === monitor.instanceId && live.requests.some(request => group.requests.some(stored => stored.id === request.id))
    sources.set(key, { key, view, internal: group, entry: { key, turn: true, requests: view.requests, representative: view.requests.at(-1), alias: group.alias, backend: historyBackend(group.instance_id, group.backend, monitor.instances ?? [], catalog),
      startedAt: group.started_at * 1000, status: !view.working && group.status === 'recovered' ? 'recovered' : view.status, count: view.requests.length, summary: overlaid ? undefined : group, matchedRequestCount: group.matched_request_count,
      elapsed: overlaid ? view.turn?.elapsed : group.elapsed_s, expanded: isExpanded(key) } })
  }
  const inRange = (view: MonitorGenerationView) => view.requests.some(request => currentHistoryInRange(request.startedAt, range.from, range.to, today))
  let liveSource = liveExternalKey ? sources.get(liveExternalKey) ?? externalSource(liveExternal[0]) : [...sources.values()].find(source => source.internal && live.sessionKey.endsWith(`:${source.internal.message_id}`) && source.internal.instance_id === monitor.instanceId)
  if (!liveSource && live.requests.length) {
    const key = JSON.stringify(['internal-live', live.sessionKey])
    liveSource = { key, view: live, entry: { key, turn: live.aggregation === 'turn', requests: live.requests, representative: live.requests.at(-1),
      alias: monitor.selectedTurn?.model || monitor.instance?.alias || '模型调用', backend: monitor.instance?.backend ?? '', startedAt: live.requests[0]?.startedAt,
      status: live.status, count: live.requests.length, elapsed: live.turn?.elapsed, expanded: isExpanded(key) } }
  }
  const liveFilterIdentity: HistoryInstance | undefined = liveSource?.external
    ? { source: 'external', run_id: liveSource.external.run_id, id: liveSource.external.instance_id, alias: liveSource.entry.alias, backend: liveSource.entry.backend }
    : live.origin === 'internal' ? { source: 'internal', id: monitor.selectedTurn?.instanceId || monitor.instanceId || null,
      conversation_id: liveSource?.internal?.conversation_id, alias: liveSource?.entry.alias ?? '', backend: liveSource?.entry.backend ?? '' } : undefined
  const liveMatchesFilter = historyInstanceMatches(instance, liveFilterIdentity)
  if (liveSource && !sources.has(liveSource.key) && offset === 0 && inRange(live) && liveMatchesFilter) sources.set(liveSource.key, liveSource)

  let selectedSource: Source | undefined
  if (selection.kind === 'inspection' && monitor.selectedTurn) selectedSource = liveSource
  else if (selection.kind === 'pinned') {
    const pinned = selection.source
    const refreshed = pinned.external ? updatedHistorySelection(pinned.external, [...(page?.items ?? []), ...[...groups.values()].flatMap(group => group.items), ...liveExternal]) : null
    selectedSource = sources.get(refreshed ? externalKey(refreshed) : pinned.key)
      ?? (refreshed ? externalSource(refreshed, pinned.entry.summary as RequestHistoryGroup) : pinned.internal ? { ...pinned, view: internalView(pinned.internal) } : pinned)
  } else if (selection.kind === 'follow') {
    selectedSource = selection.instance ? [...sources.values()].filter(source => source.external?.instance_id === selection.instance?.id && source.external?.run_id === selection.instance?.run_id).sort((a, b) => (b.entry.startedAt ?? 0) - (a.entry.startedAt ?? 0))[0] : liveSource
    if (!selectedSource && !live.requests.length) selectedSource = [...sources.values()].sort((a, b) => (b.entry.startedAt ?? 0) - (a.entry.startedAt ?? 0))[0]
  }
  const view = selectedSource?.view ?? (selection.kind === 'follow' && selection.instance ? externalView([]) : live)
  const scope = selection.kind === 'pinned' ? selection.scope : selection.kind === 'inspection' ? 'turn' : 'request'
  const pickedId = scope === 'request' ? selection.kind === 'pinned' ? selection.requestId : view.requests.find(request => request.status === 'running')?.id ?? view.requests.at(-1)?.id : undefined
  const selectedGroupId = selectedSource?.key

  useEffect(() => {
    if (selection.kind !== 'pinned' || !selection.awaitLiveInstance) return
    const pending = selection.awaitLiveInstance
    const decision = active ? confirmHistoryFollow(pending, { instanceId: monitor.instanceId, ready: monitor.ready,
      payload: monitor.payload, error: monitor.sampleError, settled: monitor.sampleSettled, live: monitor.liveGeneration, currentRun: catalog?.current_run_id }) : 'pin'
    if (decision === 'wait') return
    if (decision === 'follow') follow()
    else setSelection(current => current.kind === 'pinned' && current.awaitLiveInstance === pending ? { ...current, awaitLiveInstance: undefined } : current)
  }, [selection, active, monitor.instanceId, monitor.ready, monitor.payload, monitor.sampleError, monitor.sampleSettled, monitor.liveGeneration, catalog?.current_run_id])

  const needed = new Map<string, ExternalRequestRecord>()
  if (selectedSource?.external && selection.kind === 'pinned') needed.set(selectedSource.key, selectedSource.external)
  const neededKey = JSON.stringify([...needed].sort(([a], [b]) => a.localeCompare(b)).map(([key, record]) => [key, record.run_id, record.instance_id, record.request_key]))
  useEffect(() => {
    if (!active || !needed.size) return
    const controller = new AbortController(), version = epoch.current, reading = new Set<string>()
    async function read(key: string, record: ExternalRequestRecord) {
      if (reading.has(key)) return
      reading.add(key); setGroupLoading(previous => new Set(previous).add(key))
      const query = new URLSearchParams({ run_id: record.run_id, instance_id: record.instance_id })
      query.set('request_key', record.request_key)
      try {
        const response = await apiFetch(`/api/request-history?${query}`, { signal: controller.signal })
        const result = await response.json() as RequestHistoryPage
        if (!controller.signal.aborted && version === epoch.current) {
          setGroups(previous => new Map(previous).set(key, result)); setGroupErrors(previous => { const next = new Map(previous); next.delete(key); return next })
          releaseConsumedRecords(result.items)
        }
      } catch (issue) { if (!controller.signal.aborted && version === epoch.current) setGroupErrors(previous => new Map(previous).set(key, errorMessage(issue))) }
      finally { reading.delete(key); if (!controller.signal.aborted && version === epoch.current) setGroupLoading(previous => { const next = new Set(previous); next.delete(key); return next }) }
    }
    needed.forEach((record, key) => void read(key, record))
    const timer = window.setInterval(() => needed.forEach((record, key) => { if (record.run_id === catalog?.current_run_id) void read(key, record) }), 1500)
    return () => { controller.abort(); window.clearInterval(timer) }
  }, [active, neededKey, catalog?.current_run_id, readVersion])

  function selectGroup(entry: MonitorRequestTableEntry) {
    const source = sources.get(entry.key)
    if (!source) return
    setSelection({ kind: 'pinned', source, scope: 'turn' }); monitor.followLatest()
  }
  function selectRequest(request: MonitorRequestView, entry?: MonitorRequestTableEntry) {
    const source = entry ? sources.get(entry.key) : selectedSource
    if (!source) return
    if (canFollowHistoryRequest(request, source.view, monitor.liveGeneration, catalog?.current_run_id)) {
      const record = request.external
      follow(record && record.instance_id !== monitor.instanceId ? { source: 'external', run_id: record.run_id, id: record.instance_id, alias: record.alias, backend: record.backend } : undefined)
    } else {
      const targetId = request.external?.instance_id ?? source.internal?.instance_id ?? monitor.selectedTurn?.instanceId
      const target = targetId && targetId !== monitor.instanceId ? monitor.instances.find(instance => instance.id === targetId && isReady(instance.state)) : undefined
      const pending = target && request.status === 'running' && (!request.external || request.external.run_id === catalog?.current_run_id)
        ? { instanceId: target.id, request, messageId: source.internal?.message_id ?? monitor.selectedTurn?.messageId } : undefined
      setSelection({ kind: 'pinned', source, scope: 'request', requestId: request.id, awaitLiveInstance: pending })
      monitor.followLatest()
      if (pending) monitor.setInstanceId(pending.instanceId)
    }
  }
  const entries = [...sources.values()].map(source => source.entry).sort((a, b) => (b.startedAt ?? 0) - (a.startedAt ?? 0))
  const issues = [...new Set([...(catalog?.issues ?? []), ...(page?.issues ?? []), ...[...groups.values()].flatMap(group => group.issues ?? [])])]
  const legacy = page?.coverage?.legacy_message_count ?? catalog?.coverage?.legacy_message_count ?? 0
  const totalGroups = page?.group_total ?? 0
  return <section className="monitor-call-history" aria-label="请求监控">
    <div className="monitor-history-controls">
      <Select aria-label="历史日期" value={mode} onChange={event => { setMode(event.target.value as HistoryDateMode); if (mode !== 'range') { setFrom(range.from); setTo(range.to) }; setInstance(null); setOffset(0) }}>
        <option value="today">今天</option><option value="yesterday">昨天</option><option value="date">指定日期</option><option value="range">日期范围</option>
      </Select>
      {mode === 'date' && <input type="date" aria-label="选择日期" value={from} onChange={event => { setFrom(event.target.value); setInstance(null); setOffset(0) }} />}
      {mode === 'range' && <div className="monitor-history-date-range"><input type="date" aria-label="开始日期" value={from} onChange={event => { setFrom(event.target.value); setInstance(null); setOffset(0) }} /><span>至</span><input type="date" aria-label="结束日期" value={to} onChange={event => { setTo(event.target.value); setInstance(null); setOffset(0) }} /></div>}
      <Select aria-label="历史实例" value={instanceValue} onChange={event => { setInstance(options.find(item => item.value === event.target.value) ?? null); setOffset(0) }}>
        <option value="">全部实例</option>
        {instance && !options.some(item => item.value === instanceValue) && <option value={instanceValue}>{instance.alias} · {instance.source === 'internal' ? '内置' : '外部'}</option>}
        {options.map(item => <option key={item.value} value={item.value}>{item.label}</option>)}
      </Select>
    </div>
    {range.error && <div className="field-error" role="alert">{range.error}</div>}
    {catalogError && <ErrorNotice message={catalogError} retry={retry} />}{pageError && <ErrorNotice message={pageError} retry={retry} />}
    {issues.length > 0 && <DiagnosticNotice summary="历史读取异常" detail={issues.join('\n')} />}
    {legacy > 0 && <span className="monitor-history-coverage">{legacy} 条旧消息缺少逐请求记录</span>}
    <section className="monitor-generation-selection" aria-label="监控生成统计">
      {selectedGroupId && groupErrors.get(selectedGroupId) && <ErrorNotice message={groupErrors.get(selectedGroupId)!} retry={retry} />}
      <MonitorGeneration generation={view} scope={scope} selectedRequestId={pickedId} selectedGroupId={selectedGroupId} modelLabel={selectedSource?.entry.alias} onSelectRequest={selectRequest} generationEvents={monitor.generationEvents}
        externalTimeline={selection.kind === 'follow' && monitor.externalTimeline && view.origin === 'external' && monitor.liveGeneration.origin === 'external'
          && monitor.externalTimeline?.getSnapshot().instanceId === monitor.instanceId
          && view.requests.some(request => request.external?.instance_id === monitor.instanceId && request.external?.run_id === monitor.externalTimeline?.getSnapshot().runId)
          ? monitor.externalTimeline : undefined} />
    </section>
    <section className="monitor-unified-requests" aria-label="请求记录">
      {loading && !page && <div className="monitor-history-loading" role="status"><Loader2 size={14} className="spin" />正在读取请求记录…</div>}
      <MonitorRequestTable entries={entries} selectedGroupId={selectedGroupId} selectedRequestId={pickedId} tools={view.tools} generationEvents={monitor.generationEvents}
        onSelectGroup={selectGroup} onSelectRequest={selectRequest} onToggleGroup={entry => setExpanded(previous => new Map(previous).set(entry.key, !entry.expanded))} onRetryGroup={retry} />
      {totalGroups > 0 && <div className="monitor-history-pagination"><span>{offset + 1}–{Math.min(offset + PAGE_SIZE, totalGroups)} / {totalGroups} 条 · {page?.total} 次请求</span><Button small variant="ghost" icon={ChevronLeft} disabled={offset === 0 || loading} onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}>上一页</Button><Button small variant="ghost" icon={ChevronRight} disabled={offset + PAGE_SIZE >= totalGroups || loading} onClick={() => setOffset(offset + PAGE_SIZE)}>下一页</Button></div>}
    </section>
    <section className="monitor-usage-hardware" aria-label="用量与硬件">
      <div className="section-heading monitor-hardware-heading"><h3>用量与硬件</h3><div>{hardwareControls}</div></div>
      <div className="monitor-usage-hardware-panel"><MonitorUsage query={range.error ? '' : usageFilter.toString()} active={active && !!catalog && !range.error} revision={catalog?.usage_revision}
        live={livePage && (page?.summary?.status_counts.running ?? 0) > 0} refreshVersion={readVersion} />{hardware}</div>
    </section>
  </section>
}

