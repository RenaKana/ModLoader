import { Fragment, useLayoutEffect, useRef, useState, type ReactNode, type TdHTMLAttributes } from 'react'
import { LoaderCircle } from 'lucide-react'
import { ErrorNotice } from './components'
import { engineRateTotal, requestEngineRate, requestFirstOutput, requestTokens, turnTotals, type TokenTotal } from './chat-turn-metrics'
import type { MonitorGenerationView, MonitorRequestView } from './monitor-generation'
import { requestErrorSummary, type MonitorRequestTableEntry } from './monitor-request-groups'
import { RequestDetails } from './MonitorGeneration'
import { numberText } from './resource-metrics'
import { useGenerationAccounting, type GenerationEventStore } from './monitor-generation-events'
import { projectGenerationRequest, projectGenerationTotals } from './monitor-generation-readouts'
import './monitor-request-table.css'

export type MonitorRequestTableProps = {
  entries: MonitorRequestTableEntry[]
  selectedGroupId?: string
  selectedRequestId?: string
  onSelectGroup: (entry: MonitorRequestTableEntry) => void
  onSelectRequest: (request: MonitorRequestView, entry: MonitorRequestTableEntry) => void
  onToggleGroup: (entry: MonitorRequestTableEntry) => void
  onRetryGroup?: (entry: MonitorRequestTableEntry) => void
  tools?: MonitorGenerationView['tools']
  generationEvents?: GenerationEventStore
}

const statusNames: Record<string, string> = {
  running: '进行中', waiting: '等待响应', completed: '已完成', recovered: '已恢复', cancelled: '已中止', failed: '失败', incomplete: '已中断', pending: '待处理',
}
const statusKeys: Record<string, string> = {
  进行中: 'running', 正在生成: 'running', 正在输出: 'running', 正在思考: 'running', 等待响应: 'waiting', 调用工具: 'waiting', 待续调: 'waiting',
  已完成: 'completed', 已恢复: 'recovered', 已结束: 'completed', 已中止: 'cancelled', 已停止: 'cancelled', 失败: 'failed', 请求失败: 'failed', 已中断: 'incomplete',
}
const timeText = (startedAt?: number) => startedAt ? new Date(startedAt).toLocaleTimeString('zh-CN', { hour12: false }) : '—'
const identity = (entry: MonitorRequestTableEntry) => `${entry.alias || '模型调用'} · ${entry.startedAt ? new Date(entry.startedAt).toLocaleString('zh-CN', { hour12: false }) : '开始时间未知'}`
const numeric = (value?: number | null) => value != null && Number.isFinite(value) ? value : undefined
type TableValues = {
  count?: number; input?: TokenTotal; output?: TokenTotal; cached?: TokenTotal; cacheRatio?: number | null
  elapsed?: number; decode?: number | null; engine?: number | null; prefill?: number | null; ttft?: number | null; ttftReported?: number
  decodePartial?: string; prefillPartial?: string
}
const rateCoverage = (rate: { value: number | null; reported: number; partial_reported?: number }, count: number) => rate.value !== null && (rate.reported < count || rate.partial_reported)
  ? ` · 部分 ${rate.reported}/${count}` : ''

function requestValues(request: MonitorRequestView): TableValues {
  const tokens = requestTokens(request)
  const total = (value?: number): TokenTotal => ({ value: value ?? null, reported: value === undefined ? 0 : 1 })
  return { input: total(tokens.input), output: total(tokens.output), cached: total(tokens.cached), elapsed: request.elapsed,
    cacheRatio: tokens.input !== undefined && tokens.input > 0 && tokens.cached !== undefined && tokens.cached <= tokens.input ? tokens.cached / tokens.input : null,
    decode: request.decodeMean, engine: request.decodeMean === null ? requestEngineRate(request).value : undefined,
    prefill: request.prefillMean, ttft: requestFirstOutput(request).value }
}

function groupValues(entry: MonitorRequestTableEntry): TableValues {
  if (entry.summary) {
    const summary = entry.summary
    const complete = entry.requests.length > 0 && entry.requests.length === summary.request_count
    const single = complete && summary.request_count === 1 ? entry.requests[0] : undefined
    const decode = summary.decode.value ?? single?.decodeMean, prefill = summary.prefill.value ?? single?.prefillMean
    return { count: summary.request_count, input: summary.input, output: summary.output, cached: summary.cached, cacheRatio: summary.cache_ratio,
      elapsed: entry.elapsed, decode, decodePartial: rateCoverage(summary.decode, summary.request_count), prefillPartial: rateCoverage(summary.prefill, summary.request_count),
      engine: decode == null ? summary.engine?.value ?? (complete ? engineRateTotal(entry.requests).value : undefined) : undefined,
      prefill, ttft: summary.ttft.value, ttftReported: summary.ttft.reported }
  }
  const count = entry.count ?? (!entry.turn ? entry.requests.length : undefined)
  // Directory representatives and a history page are not the complete request ledger.
  if (!entry.requests.length || count !== entry.requests.length) return { count, elapsed: entry.elapsed }
  const starts = entry.requests.map(request => request.startedAt)
  const ends = entry.requests.map(request => request.endedAt ?? (numeric(request.elapsed) !== undefined && request.startedAt ? request.startedAt + request.elapsed! * 1000 : undefined))
  const elapsed = entry.elapsed ?? (count === 1 ? entry.requests[0].elapsed
    : starts.every(Boolean) && ends.every(end => end !== undefined) ? (Math.max(...ends as number[]) - Math.min(...starts)) / 1000 : undefined)
  const totals = turnTotals(entry.requests, entry.startedAt ?? Math.min(...starts), elapsed ?? 0)
  const single = count === 1 ? entry.requests[0] : undefined
  const decode = totals.decode.value ?? single?.decodeMean, prefill = totals.prefill.value ?? single?.prefillMean
  const firstOutputs = entry.requests.map(request => numeric(requestFirstOutput(request).value)).filter((value): value is number => value !== undefined)
  const pairs = entry.requests.map(requestTokens).filter(tokens => tokens.input !== undefined && tokens.cached !== undefined && tokens.cached <= tokens.input)
  const pairedInput = pairs.reduce((sum, pair) => sum + pair.input!, 0), pairedCache = pairs.reduce((sum, pair) => sum + pair.cached!, 0)
  return { count, input: totals.input, output: totals.output, cached: totals.cached, elapsed, decode, prefill,
    decodePartial: rateCoverage(totals.decode, count), prefillPartial: rateCoverage(totals.prefill, count),
    engine: decode == null ? engineRateTotal(entry.requests).value : undefined,
    cacheRatio: pairedInput > 0 ? pairedCache / pairedInput : null,
    ttft: firstOutputs.length ? firstOutputs.reduce((sum, value) => sum + value, 0) / firstOutputs.length : null, ttftReported: firstOutputs.length }
}

function StatusDot({ status, incomplete = false }: { status?: string; incomplete?: boolean }) {
  const key = incomplete ? 'incomplete' : statusKeys[status ?? ''] ?? status ?? 'unknown'
  const label = incomplete ? '记录未完整' : statusNames[key] ?? status ?? '状态未知'
  return <span className="monitor-ledger-dot" data-status={key} role="img" aria-label={label} title={label} />
}

function TokenValue({ label, total, count, secondary }: { label: string; total?: TokenTotal; count?: number; secondary?: string }) {
  const value = numeric(total?.value), partial = value !== undefined && count !== undefined && (total?.reported ?? 0) < count ? `部分 ${total?.reported ?? 0}/${count}` : ''
  const description = `${label} ${value === undefined ? '未知' : `${numberText(value, 0)} Token`}${partial ? ` · ${partial}` : ''}`
  const subvalue = secondary ? `${secondary}${partial ? ` · ${total?.reported ?? 0}/${count}` : ''}` : partial
  return <span className="monitor-ledger-value" aria-label={description} title={description}>
    {value === undefined ? '—' : numberText(value, 0)}{subvalue && <small className="monitor-ledger-subvalue">{subvalue}</small>}
  </span>
}

function CollapseFrame({ children }: { children: ReactNode }) {
  return <div className="monitor-ledger-collapse-frame"><div className="monitor-ledger-collapse-clip">{children}</div></div>
}

function MetricCell({ collapse, children, ...props }: TdHTMLAttributes<HTMLTableCellElement> & { collapse?: boolean }) {
  return <td {...props}>{collapse ? <CollapseFrame><div className="monitor-ledger-cell-body">{children}</div></CollapseFrame> : children}</td>
}

const Branch = () => <span className="monitor-ledger-branch" aria-hidden="true" />
const isCollapsing = (node: ParentNode) => [...node.querySelectorAll('.monitor-ledger-collapse-frame')]
  .some(frame => frame.getAnimations().some(animation => animation.playState === 'running' || animation.pending))

function MetricCells({ values, startedAt, parent = false }: { values: TableValues; startedAt?: number; parent?: boolean }) {
  const { count, elapsed, decode, engine, prefill, ttft, ttftReported } = values
  const engineOnly = numeric(decode) === undefined && numeric(engine) !== undefined, speed = engineOnly ? engine : decode
  const speedDescription = `${engineOnly ? '引擎平均' : 'Decode'} ${numeric(speed) === undefined ? '未知' : `${numberText(speed, 2, true)} token/s`}${engineOnly ? ' · 包含首 Token 等待' : values.decodePartial ?? ''}`
  const firstOutput = `首字 ${numeric(ttft) === undefined ? '未知' : `${numberText(ttft, 2, true)} 秒`}${parent && ttftReported ? ` · 已知 ${ttftReported}/${count ?? '—'} 次请求均值` : ''}`
  return <>
    <MetricCell collapse={!parent} data-column="input"><TokenValue label="输入（含缓存）" total={values.input} count={parent ? count : undefined} /></MetricCell>
    <MetricCell collapse={!parent} data-column="output"><TokenValue label="输出" total={values.output} count={parent ? count : undefined} /></MetricCell>
    <MetricCell collapse={!parent} data-column="cached"><TokenValue label="缓存命中" total={values.cached} count={parent ? count : undefined}
      secondary={numeric(values.cacheRatio) === undefined ? '—' : `${numberText(values.cacheRatio! * 100, 1, true)}%`} /></MetricCell>
    <MetricCell collapse={!parent} data-column="elapsed"><span aria-label={`耗时 ${numeric(elapsed) === undefined ? '未知' : `${numberText(elapsed, 2, true)} 秒`}`}>{numeric(elapsed) === undefined ? '—' : numberText(elapsed, 2, true)}</span>
      <small className="monitor-ledger-subvalue" aria-label={firstOutput} title={firstOutput}>首字 {numeric(ttft) === undefined ? '—' : numberText(ttft, 2, true)}</small></MetricCell>
    <MetricCell collapse={!parent} data-column="speed"><span aria-label={speedDescription} title={speedDescription}>{numeric(speed) === undefined ? '—' : numberText(speed, 2, true)}{engineOnly && <small className="monitor-ledger-rate-basis">AVG</small>}</span>
      <small className="monitor-ledger-subvalue" aria-label={`Prefill ${numeric(prefill) === undefined ? '未知' : `${numberText(prefill, 2, true)} token/s`}${values.prefillPartial ?? ''}`}>PF {numeric(prefill) === undefined ? '—' : numberText(prefill, 2, true)}</small></MetricCell>
    <MetricCell collapse={!parent} className="monitor-ledger-time" data-column="time"><time dateTime={startedAt ? new Date(startedAt).toISOString() : undefined} title={startedAt ? new Date(startedAt).toLocaleString('zh-CN', { hour12: false }) : undefined}>{timeText(startedAt)}</time></MetricCell>
  </>
}

function RequestMetricCells({ request, store }: { request: MonitorRequestView; store?: GenerationEventStore }) {
  const accounting = useGenerationAccounting(store, [request])
  return <MetricCells values={requestValues(projectGenerationRequest(request, accounting))} startedAt={request.startedAt} />
}
function GroupMetricCells({ entry, store }: { entry: MonitorRequestTableEntry; store?: GenerationEventStore }) {
  const accounting = useGenerationAccounting(store, entry.requests)
  const summary = accounting ? projectGenerationTotals(entry.requests, accounting) : undefined
  return <MetricCells values={groupValues(summary ? { ...entry, summary } : entry)} startedAt={entry.startedAt} parent />
}
function RequestCount({ entry, store }: { entry: MonitorRequestTableEntry; store?: GenerationEventStore }) {
  const accounting = useGenerationAccounting(store, entry.requests)
  const count = accounting ? projectGenerationTotals(entry.requests, accounting).request_count : groupValues(entry).count
  return <span data-request-count={count ?? ''} aria-label={`请求数 ${count ?? '未知'}`}>{count === undefined ? '请求数 —' : `${count} 次请求`}</span>
}

function ExternalRequestRow({ entry, selectedRequestId, onSelectRequest, generationEvents }: Omit<MonitorRequestTableProps, 'entries'> & { entry: MonitorRequestTableEntry }) {
  const original = entry.requests[0], accounting = useGenerationAccounting(generationEvents, entry.requests)
  const request = projectGenerationRequest(original, accounting), record = request.external!
  const picked = request.id === selectedRequestId, description = identity(entry)
  const error = requestErrorSummary(request), detail = [...request.labels, error].filter(Boolean).join(' · ')
  return <tbody className="monitor-ledger-tree" data-tree-key={entry.key}>
    <tr className="monitor-ledger-spacer" aria-hidden="true"><td colSpan={7} /></tr>
    <tr className="monitor-ledger-group monitor-external-request" data-group-key={entry.key} data-request-key={record.request_key}
      data-request-run={record.run_id} data-request-instance={record.instance_id} data-selected={picked} onClick={() => onSelectRequest(request, entry)}>
      <th scope="row"><div className="monitor-ledger-group-heading">
        <span className="monitor-ledger-request-status"><StatusDot status={request.status === 'running' && !request.liveMetrics ? 'waiting' : request.status} incomplete={record.record_incomplete} /></span>
        <div className="monitor-ledger-group-identity">
          <button type="button" className="monitor-ledger-group-select" aria-label={`选择请求 · ${description}`} aria-pressed={picked}
            onClick={event => { event.stopPropagation(); onSelectRequest(request, entry) }}><strong title={entry.alias}>{entry.alias || '模型调用'}</strong></button>
          <small className="monitor-ledger-group-meta">{entry.backend || '—'} · 外部</small>
          {detail && <small className={`monitor-ledger-group-meta${error ? ' monitor-ledger-request-error' : ''}`} title={detail}>{detail}</small>}
        </div>
      </div></th>
      <MetricCells values={requestValues(request)} startedAt={request.startedAt} parent />
    </tr>
  </tbody>
}

function RequestGroup({ entry, selectedGroupId, selectedRequestId, onSelectGroup, onSelectRequest, onToggleGroup, onRetryGroup, tools = [], generationEvents }: Omit<MonitorRequestTableProps, 'entries'> & { entry: MonitorRequestTableEntry }) {
  const expanded = !!entry.expanded, picked = entry.key === selectedGroupId && !selectedRequestId
  const [present, setPresent] = useState(expanded)
  const body = useRef<HTMLTableSectionElement>(null)
  const mounted = expanded || present, connected = mounted && entry.requests.length > 0
  const description = identity(entry), record = entry.representative?.external ?? entry.requests[0]?.external, values = groupValues(entry)
  const groupState = statusKeys[entry.status ?? ''] ?? entry.status
  const status = groupState === 'recovered' && (entry.requests.some(request => request.status === 'running') || (entry.summary?.status_counts.running ?? 0) > 0) ? 'running' : groupState
  useLayoutEffect(() => {
    if (expanded) { setPresent(true); return }
    const node = body.current
    if (!node?.querySelector('tr[data-collapsible]')) return
    if (document.activeElement && node.contains(document.activeElement) && document.activeElement.closest('tr[data-collapsible]'))
      node.querySelector<HTMLButtonElement>('.monitor-ledger-expand')?.focus({ preventScroll: true })
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) { setPresent(false); return }
    // An open/close within one frame has no CSS transition to finish.
    const frame = requestAnimationFrame(() => {
      if (!isCollapsing(node)) setPresent(false)
    })
    return () => cancelAnimationFrame(frame)
  }, [expanded])
  return <tbody ref={body} className="monitor-ledger-tree" data-tree-key={entry.key} data-open={expanded} onTransitionEnd={event => {
    if (!expanded && (event.target as HTMLElement).classList.contains('monitor-ledger-collapse-frame') && !isCollapsing(event.currentTarget)) setPresent(false)
  }}>
        <tr className="monitor-ledger-spacer" aria-hidden="true"><td colSpan={7} /></tr>
        <tr className="monitor-turn-group monitor-ledger-group" data-group-key={entry.key} data-turn-key={entry.key} data-turn-id={record?.turn_id ?? entry.key} data-turn-run={record?.run_id} data-turn-instance={record?.instance_id} data-selected={picked} data-expanded={expanded} onClick={() => onToggleGroup(entry)}>
          <th scope="row">{connected && <Branch />}<div className="monitor-ledger-group-heading">
            <button type="button" className="monitor-ledger-expand" aria-label={`${expanded ? '收起' : '展开'}调用 · ${description}`} aria-expanded={expanded} onClick={event => { event.stopPropagation(); onToggleGroup(entry) }}><StatusDot status={status} /></button>
            <div className="monitor-ledger-group-identity">
              <button type="button" className="monitor-ledger-group-select" aria-label={`选择轮汇总 · ${description}`} aria-pressed={picked} onClick={event => { event.stopPropagation(); onSelectGroup(entry) }}><strong title={entry.alias}>{entry.alias || '模型调用'}</strong></button>
              <small className="monitor-ledger-group-meta">{entry.backend || '—'} · <RequestCount entry={entry} store={generationEvents} />{status === 'recovered' && <span className="monitor-ledger-recovered"> · 已恢复</span>}{values.count != null && entry.matchedRequestCount != null && entry.matchedRequestCount < values.count && <span className="monitor-ledger-matched" title={`本轮 ${values.count} 次请求，当前日期范围内 ${entry.matchedRequestCount} 次`}> · 筛选内 {entry.matchedRequestCount}</span>}</small>
            </div>
          </div></th>
          <GroupMetricCells entry={entry} store={generationEvents} />
        </tr>
        {mounted && <>
          {entry.error && <tr className="monitor-ledger-notice" data-collapsible aria-hidden={!expanded} inert={!expanded}><td colSpan={7}>{connected && <Branch />}<CollapseFrame><div className="monitor-ledger-notice-body"><ErrorNotice message={entry.error} retry={onRetryGroup ? () => onRetryGroup(entry) : undefined} /></div></CollapseFrame></td></tr>}
          {entry.loading && !entry.requests.length && <tr className="monitor-ledger-notice" data-collapsible aria-hidden={!expanded} inert={!expanded}><td colSpan={7}><CollapseFrame><div className="monitor-ledger-notice-body"><span role="status"><LoaderCircle size={14} className="spin" aria-hidden="true" />正在读取请求…</span></div></CollapseFrame></td></tr>}
          {entry.requests.map((request, index) => {
            const selected = request.id === selectedRequestId
            const events = selected ? tools.filter(event => event.requestId === request.id) : []
            const labels = [...new Set([...(request.kind === 'planning' ? ['检索规划'] : request.kind === 'compaction' && !request.labels.includes('手动压缩') ? ['自动压缩'] : []), ...request.labels])]
            const error = requestErrorSummary(request)
            const detail = [...labels, error].filter(Boolean).join(' · ')
            const last = index === entry.requests.length - 1
            return <Fragment key={request.id}><tr className="monitor-group-request monitor-ledger-child" data-collapsible aria-hidden={!expanded} inert={!expanded} data-parent-group={entry.key} data-last={last} data-request-index={request.index} data-request-key={request.requestKey || request.id} data-request-run={request.external?.run_id} data-request-instance={request.external?.instance_id} data-selected={selected} onClick={() => onSelectRequest(request, entry)}>
              <th scope="row"><Branch />
                <StatusDot status={request.status === 'running' && !request.liveMetrics ? 'waiting' : request.status} incomplete={request.external?.record_incomplete} />
                <CollapseFrame><button type="button" className="monitor-ledger-request-select" aria-label={`选择请求 ${request.index} · ${description} · ${timeText(request.startedAt)}`} aria-pressed={selected}>
                <span><strong>请求 {request.index}</strong>{detail && <small className={error ? 'monitor-ledger-request-error' : undefined} title={detail}>{detail}</small>}</span>
              </button></CollapseFrame></th>
              <RequestMetricCells request={request} store={generationEvents} />
            </tr>{events.length > 0 && <tr className="monitor-ledger-tools" data-collapsible aria-hidden={!expanded} inert={!expanded}><td colSpan={7}>{!last && <Branch />}<CollapseFrame><div className="monitor-ledger-notice-body"><RequestDetails events={events} label={`请求 ${request.index} 明细`} expanded /></div></CollapseFrame></td></tr>}</Fragment>
          })}
        </>}
  </tbody>
}

export function MonitorRequestTable({ entries, ...props }: MonitorRequestTableProps) {
  const content = useRef<HTMLDivElement>(null), marker = useRef<HTMLSpanElement>(null)
  useLayoutEffect(() => {
    const node = content.current, indicator = marker.current
    if (!node || !indicator) return
    const move = () => {
      const row = node.querySelector<HTMLElement>('tr[data-selected="true"]:not([aria-hidden="true"])')
      if (!row) { indicator.style.opacity = '0'; return }
      const rect = row.getBoundingClientRect(), origin = node.getBoundingClientRect()
      indicator.style.transform = `translateY(${rect.top - origin.top}px)`
      indicator.style.height = `${rect.height}px`; indicator.style.opacity = '1'
    }
    move()
    const frame = requestAnimationFrame(() => { indicator.dataset.ready = 'true' })
    const observer = new ResizeObserver(move)
    observer.observe(node)
    node.querySelectorAll('tbody').forEach(body => observer.observe(body))
    return () => { observer.disconnect(); cancelAnimationFrame(frame) }
  }, [props.selectedGroupId, props.selectedRequestId, entries.map(entry => entry.key).join('\n')])
  return <div className="monitor-request-ledger"><div className="monitor-ledger-content" ref={content}>
    <span className="monitor-ledger-selection" ref={marker} aria-hidden="true" />
    <table className="monitor-request-table monitor-request-ledger-table" aria-label="请求记录">
      <colgroup><col className="monitor-ledger-name-column" /><col /><col /><col /><col /><col /><col className="monitor-ledger-time-column" /></colgroup>
      <thead><tr><th scope="col">模型 / 调用</th><th scope="col" title="输入（含缓存） · Token">输入</th><th scope="col" title="Token">输出</th><th scope="col" title="Token / 命中率">缓存命中</th><th scope="col">耗时 · s</th><th scope="col">速度 · tok/s</th><th scope="col">时间</th></tr></thead>
      {entries.map(entry => !entry.turn && entry.requests[0]?.external
        ? <ExternalRequestRow key={entry.key} entry={entry} {...props} />
        : <RequestGroup key={entry.key} entry={entry} {...props} />)}
      {!entries.length && <tbody><tr className="monitor-ledger-empty"><td colSpan={7}>没有符合条件的请求记录</td></tr></tbody>}
    </table>
  </div></div>
}
