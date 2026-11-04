import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties } from 'react'
import { Ban, Check, Clock3, LoaderCircle, X } from 'lucide-react'
import { describeToolActivity } from './chat-activity'
import { engineRateTotal, requestEngineRate, requestFirstOutput, requestTokens, turnTotals, type TurnTotals } from './chat-turn-metrics'
import { finite, numberText } from './resource-metrics'
import { generationSegments, type GenerationPhase, type GenerationPoint, type MonitorRequestView } from './monitor-generation'
import type { MonitorState } from './useMonitorState'
import { windowSegments } from './monitor-chart-window'
import { generationEventIdentity, useGenerationAccounting, useGenerationEventRate, type GenerationEventStore } from './monitor-generation-events'
import { projectGenerationRequest, projectGenerationTotals } from './monitor-generation-readouts'
import { NumericReadout } from './ResourceCards'
import { ExternalGenerationTimeline } from './ExternalGenerationTimeline'
import type { ExternalTimelineStore } from './external-generation-timeline'
import './monitor-generation-compact.css'

const statusNames: Record<string, string> = { running: '进行中', completed: '已完成', cancelled: '已中止', failed: '失败', incomplete: '已中断', pending: '待处理' }
const chartWindows = [10, 30, 60] as const
const requestStatus = (request: MonitorRequestView) => request.external?.record_incomplete ? '记录未完整' : request.status === 'running' && !request.liveMetrics ? '等待响应' : statusNames[request.status]
const requestTime = (request?: MonitorRequestView) => request?.startedAt ? new Date(request.startedAt).toLocaleTimeString('zh-CN', { hour12: false }) : '—'
const requestIdentity = (request: MonitorRequestView, alias?: string) => `${alias || request.external?.alias || '模型请求'} · ${request.startedAt ? new Date(request.startedAt).toLocaleString('zh-CN', { hour12: false }) : '开始时间未知'} · ${requestStatus(request)}`
type ChartWindow = typeof chartWindows[number]
const readout = (value: number | null | undefined, unit = '', speed = false) => <NumericReadout value={value == null ? '—' : numberText(value, speed ? 2 : unit === 'token' ? 0 : 2, speed)} unit={value != null ? unit : undefined} stacked={speed} />

export function MonitorGeneration({ generation, scope, selectedRequestId, selectedGroupId, modelLabel, onSelectRequest, generationEvents, externalTimeline }: {
  generation: MonitorState['generation']; scope: 'turn' | 'request'; selectedRequestId?: string; selectedGroupId?: string; modelLabel?: string
  onSelectRequest: (request: MonitorRequestView) => void
  generationEvents?: GenerationEventStore
  externalTimeline?: ExternalTimelineStore
}) {
  const [phase, setPhase] = useState<GenerationPhase>('decode')
  const [windowSeconds, setWindowSeconds] = useState<ChartWindow>(30)
  const requests = generation.requests
  const selected = requests.find(request => request.id === selectedRequestId) ?? requests.at(-1)
  const mode = scope === 'turn' ? 'turn' : 'realtime'
  const turn = generation.turn ?? (requests.length ? turnTotals(requests, requests[0].startedAt,
    selected?.elapsed ?? 0, selected?.status ?? 'completed', selected?.endedAt) : undefined)
  if (externalTimeline && scope === 'request') return <ExternalGenerationTimeline store={externalTimeline} windowSeconds={windowSeconds} modelLabel={modelLabel}
    nextWindow={chartWindows[(chartWindows.indexOf(windowSeconds) + 1) % chartWindows.length]} onCycleWindow={() => setWindowSeconds(chartWindows[(chartWindows.indexOf(windowSeconds) + 1) % chartWindows.length])}
    aside={<GenerationRequestReadout generation={generation} selected={selected} turn={turn} scope={scope} store={generationEvents} />} />
  if (!requests.length) return <div className="monitor-generation-idle">尚无模型请求</div>
  return <section className="panel monitor-compact-generation" aria-label="生成监测" data-scope={scope}>
    <div className="monitor-compact-heading">
      <div className="monitor-compact-switches">
        <button type="button" className="monitor-phase-toggle" aria-label={`${phase === 'decode' ? 'Decode' : 'Prefill'}，切换到 ${phase === 'decode' ? 'Prefill' : 'Decode'}`}
          data-tooltip={`切换到 ${phase === 'decode' ? 'Prefill' : 'Decode'}`} onClick={() => setPhase(phase === 'decode' ? 'prefill' : 'decode')}>{phase === 'decode' ? 'Decode' : 'Prefill'}</button>
        <span className="monitor-window-slot"><button type="button" className="monitor-window-cycle" hidden={scope === 'turn'}
          aria-label={`曲线窗口 ${windowSeconds} 秒，切换到 ${chartWindows[(chartWindows.indexOf(windowSeconds) + 1) % chartWindows.length]} 秒`}
          data-tooltip="切换曲线时间窗口" onClick={() => setWindowSeconds(chartWindows[(chartWindows.indexOf(windowSeconds) + 1) % chartWindows.length])}>{windowSeconds}s</button></span>
      </div>
      <span className="monitor-compact-identity" aria-live="polite">{modelLabel && <span className="monitor-compact-model" title={modelLabel}>{modelLabel} · </span>}{scope === 'turn' ? `本轮 · ${requests.length} 次请求` : `请求 #${selected?.index ?? '—'} · ${selected ? requestStatus(selected) : '—'}`}</span>
    </div>
    <div className="monitor-compact-body">
      <div className="monitor-compact-main"><div className="monitor-rate-readout">
        <GenerationSpeedReadout requests={requests} selected={selected} turn={turn} scope={scope} phase={phase} store={generationEvents} />
        <time className="monitor-speed-time">{requestTime(scope === 'turn' ? requests[0] : selected)}</time>
      </div><GenerationChart key={`${generation.sessionKey}:${mode}:${phase}:${scope === 'request' ? selected?.id : ''}`} mode={mode} phase={phase} requests={requests}
        selected={scope === 'request' ? selected : undefined} onSelect={onSelectRequest} windowSeconds={windowSeconds} numbered
        groupKey={selectedGroupId} /></div>
      <GenerationRequestReadout generation={generation} selected={selected} turn={turn} scope={scope} store={generationEvents} />
    </div>
  </section>
}

function GenerationSpeedReadout({ requests, selected: original, turn, scope, phase, store }: { requests: MonitorRequestView[]; selected?: MonitorRequestView; turn?: TurnTotals; scope: 'turn' | 'request'; phase: GenerationPhase; store?: GenerationEventStore }) {
  const accounting = useGenerationAccounting(store, scope === 'request' ? original ? [original] : [] : requests)
  const selected = original ? projectGenerationRequest(original, accounting) : undefined
  const totals = accounting ? projectGenerationTotals(requests, accounting) : undefined
  const count = totals?.request_count ?? requests.length
  const realtime = selected?.phases[phase]
  const average = scope === 'turn' && count > 1 ? (totals?.[phase] ?? turn?.[phase])?.value : selected?.[phase === 'decode' ? 'decodeMean' : 'prefillMean']
  const sampled = scope === 'request' && realtime?.current != null && !(accounting && selected?.status !== 'running')
  const engine = phase === 'decode' && average == null && !sampled
    ? scope === 'turn' ? engineRateTotal(requests).value : selected ? requestEngineRate(selected).value : null : null
  const speed = sampled ? realtime.current : average ?? engine
  const rateTotal = totals?.[phase] ?? turn?.[phase]
  const partial = scope === 'turn' && count > 1 && rateTotal?.value != null && (rateTotal.reported < count || !!rateTotal.partial_reported)
  const label = sampled ? realtime.ended ? '最后采样' : '实时速度'
    : engine != null ? '引擎平均 · 含首 Token 等待'
      : scope === 'turn' ? count === 1 ? '本轮平均' : `本轮加权平均${partial ? ` · 部分 ${rateTotal!.reported}/${count}` : ''}`
        : average != null ? '请求平均' : '暂无速度数据'
  const eventRate = useGenerationEventRate(store, scope === 'request' ? generationEventIdentity(original, store) : undefined, phase)
  const fastRate = accounting && selected?.status !== 'running' ? undefined : eventRate
  return <><div className="monitor-speed-current">{readout(fastRate === undefined ? speed : fastRate, 'token/s', true)}</div>
    <div className="monitor-speed-note">{fastRate === undefined ? label : fastRate === null ? '暂无速度数据' : '实时速度'}</div></>
}

function GenerationRequestReadout({ generation, selected: original, turn, scope, store }: { generation: MonitorState['generation']; selected?: MonitorRequestView; turn?: TurnTotals; scope: 'turn' | 'request'; store?: GenerationEventStore }) {
  const accounting = useGenerationAccounting(store, scope === 'request' ? original ? [original] : [] : generation.requests)
  const selected = original ? projectGenerationRequest(original, accounting) : undefined
  const tokens = selected ? requestTokens(selected) : undefined
  const total = accounting ? projectGenerationTotals(generation.requests, accounting) : undefined
  const output = total?.output ?? turn?.output
  const outputPartial = scope === 'turn' && output?.value != null && output.reported < (total?.request_count ?? turn?.requests ?? generation.requests.length)
  const used = accounting || scope === 'request' ? tokens?.input !== undefined && tokens.output !== undefined ? tokens.input + tokens.output : null : generation.used
  const capacity = generation.capacity
  return <aside className="monitor-compact-aside" aria-label="上下文与请求统计">
        <section className="monitor-request-group" aria-label={scope === 'turn' ? '所选轮次统计' : '所选请求统计'} data-selected-request-key={scope === 'request' ? selected?.requestKey || selected?.id : ''}
          data-selected-run={selected?.external?.run_id} data-selected-instance={selected?.external?.instance_id}>
          <dl className="monitor-request-live">
            <div><dt>{scope === 'turn' ? '最后请求长度' : '请求长度'}</dt><dd>{readout(used)}</dd></div>
            <div><dt>{scope === 'turn' ? '本轮耗时' : '耗时'}</dt><dd>{scope === 'turn' && turn ? <TurnDuration key={generation.sessionKey} turn={turn} frozen={generation.waiting} /> : readout(selected?.elapsed, 's')}</dd></div>
            <div data-tooltip={requestFirstOutput(selected).basis}><dt>{scope === 'turn' ? '最后首字' : '首字'}</dt><dd>{readout(requestFirstOutput(selected).value, 's')}</dd></div>
            <div><dt>{scope === 'turn' ? `本轮输出${outputPartial ? ' · 部分' : ''}` : '输出'}</dt><dd>{readout(scope === 'turn' ? output?.value : tokens?.output)}</dd></div>
          </dl>
        </section>
        {capacity != null && <div className="monitor-context-compact" data-resource="context" data-tooltip={used != null ? `剩余 ${numberText(capacity-used, 0)} token` : undefined}>
          容量 {numberText(capacity, 0)}{used != null ? ` · ${numberText(used / capacity * 100, 1)}%` : ''}
        </div>}
      </aside>
}

export function RequestDetails({ events, label, expanded }: { events: MonitorState['generation']['tools']; label: string; expanded: boolean }) {
  const body = useRef<HTMLDivElement>(null)
  const [height, setHeight] = useState(0)
  useLayoutEffect(() => {
    const node = body.current
    if (!node) return
    const measure = () => setHeight(node.getBoundingClientRect().height)
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(node)
    return () => observer.disconnect()
  }, [])
  return <div className="monitor-request-detail" data-open={expanded} aria-label={label} aria-hidden={!expanded} inert={!expanded} style={{ '--request-detail-height': `${height}px` } as CSSProperties}>
    <div className="monitor-request-detail-body" ref={body}><ul className="monitor-request-tools" aria-label="本次工具调用">{events.map(event => {
      const description = describeToolActivity(event)
      const StatusIcon = description.status === 'completed' ? Check : description.status === 'running' ? LoaderCircle : description.status === 'failed' ? X : description.status === 'cancelled' ? Ban : Clock3
      return <li key={event.id} className="monitor-tool-readout">
        <span className="monitor-tool-status" data-status={description.status} role="img" aria-label={statusNames[description.status]} title={statusNames[description.status]}><StatusIcon size={14} className={description.status === 'running' ? 'spin' : undefined} aria-hidden="true" /></span>
        <div className="monitor-tool-copy"><strong>{description.action}</strong>{description.outcome && <span>{description.outcome}</span>}{description.target && <span className="monitor-tool-target" title={description.target}>{description.target}</span>}{description.error && <span className="monitor-tool-error">{description.error}</span>}</div>
        <span className="monitor-tool-duration">{readout(event.elapsed, 's')}</span>
      </li>
    })}</ul></div>
  </div>
}

function GenerationChart({ mode, phase, requests, selected, detailsExpanded, onSelect, windowSeconds, numbered, groupKey }: { mode: 'turn' | 'realtime'; phase: GenerationPhase; requests: MonitorRequestView[]; selected?: MonitorRequestView; detailsExpanded?: boolean; onSelect: (request: MonitorRequestView) => void; windowSeconds: ChartWindow; numbered: boolean; groupKey?: string }) {
  const [hoverPoint, setHoverPoint] = useState<GenerationPoint | null>(null)
  const realtime = selected?.phases[phase]
  const history: GenerationPoint[] = mode === 'turn' ? requests.map(request => ({ time: request.index, wallTime: request.startedAt, values: { current: request[phase === 'decode' ? 'decodeMean' : 'prefillMean'] }, segment: 'request-means' })) : realtime?.points ?? []
  const end = history.at(-1)?.time ?? 0
  const span = mode === 'turn' ? Math.max(1, requests.length - 1) : windowSeconds * 1000
  const start = mode === 'turn' ? 1 : end - span
  // Keep the full retained history; only the viewport changes. Early requests
  // leave empty space rather than stretching, and old samples never pile up at x=0.
  const points = mode === 'turn' ? history : history.filter(point => point.time >= start)
  const hover = hoverPoint && points.some(point => point.time === hoverPoint.time && point.wallTime === hoverPoint.wallTime) ? hoverPoint : null
  const hoveredRequest = mode === 'turn' && hover ? requests.find(request => request.index === hover.time) : undefined
  const segments = mode === 'turn' ? generationSegments(points)
    : windowSegments([...(realtime?.boundaryPoint ? [realtime.boundaryPoint] : []), ...history], start, end)
  let peak = 1
  for (const segment of segments) for (const point of segment) peak = Math.max(peak, point.value)
  const magnitude = 10 ** Math.floor(Math.log10(peak * 1.1 / 3))
  const step = [1, 2, 5, 10].find(value => value * magnitude >= peak * 1.1 / 3)! * magnitude
  const maximum = step * 3
  const x = (time: number) => mode === 'turn' && requests.length === 1 ? 300
    : mode === 'turn' ? 8 + Math.max(0, Math.min(1, (time - start) / span)) * 584 : Math.max(0, Math.min(1, (time - start) / span)) * 600
  const y = (value: number) => 150 - Math.max(0, Math.min(1, value / maximum)) * 144
  const trace = (segment: Array<{ time: number; value: number }>) => segment.map((point, index) => `${index ? 'L' : 'M'}${x(point.time).toFixed(2)},${y(point.value).toFixed(2)}`).join(' ')
  const label = phase === 'decode' ? 'Decode' : 'Prefill'
  const positionLabel = (index: number) => numbered ? `#${index + 1}` : requestTime(requests[index])
  const selectedPoint = mode === 'turn' && selected ? points.find(point => point.time === selected.index) : null
  return <div className="monitor-generation-chart" data-phase={phase} data-mode={mode} data-request-key={mode === 'realtime' ? selected?.requestKey || selected?.id : ''} data-group-key={groupKey}
    style={{ '--generation-axis-width': `max(44px, calc(${numberText(maximum, 1).length}ch + 9px))` } as CSSProperties}>
    <div className="monitor-generation-graph">
      <div className="monitor-generation-ticks" aria-hidden="true">{[0, 1, 2, 3].map(index => <span className="monitor-generation-tick" key={index} style={{ top: `${y(step * index) / 156 * 100}%` }}>{numberText(step * index, 1)}</span>)}</div>
      <svg viewBox="0 0 600 156" preserveAspectRatio="none" role={mode === 'turn' ? 'group' : 'img'} aria-label={mode === 'turn' ? `本轮各次模型请求平均 ${label} 速度` : `${numbered ? `请求 #${selected?.index ?? '—'}` : selected ? requestIdentity(selected) : '所选请求'} ${label} ${realtime?.ended ? '速度采样记录' : '实时速度'}`}
        onPointerMove={event => { const rect = event.currentTarget.getBoundingClientRect(), inset = mode === 'turn' ? 8 : 0, width = mode === 'turn' ? 584 : 600; const time = start + Math.max(0, Math.min(1, (event.clientX - rect.left - inset / 600 * rect.width) / (rect.width * width / 600))) * span; setHoverPoint(points.reduce<GenerationPoint | null>((best, point) => !best || Math.abs(point.time - time) <= Math.abs(best.time - time) ? point : best, null)) }} onPointerLeave={() => setHoverPoint(null)}>
        {[0, 1, 2, 3].map(index => <path key={index} className="trend-grid" d={`M0 ${y(step * index)}H600`} />)}
        <g data-series="current">{segments.map((segment, index) => <g key={index}>{segment.length > 1 ? <><path className="monitor-generation-fill" d={`${trace(segment)} L${x(segment.at(-1)!.time)},150 L${x(segment[0].time)},150Z`} /><path className="trend-line" d={trace(segment)} /></> : mode === 'realtime' ? <circle className="trend-dot" cx={x(segment[0].time)} cy={y(segment[0].value)} r="2" /> : null}</g>)}</g>
        {(selectedPoint || hover) && <path className="monitor-generation-guide" d={`M${x((hover || selectedPoint)!.time)} 6V150`} />}
        {mode === 'turn' && points.map((point, index) => <g key={point.time} className="monitor-request-point" role="button" tabIndex={0} aria-label={`${numbered ? `请求 #${requests[index].index}` : requestIdentity(requests[index])}，平均 ${label} ${point.values.current === null ? '未知' : `${numberText(point.values.current, 2, true)} token/s`}`} aria-pressed={requests[index].id === selected?.id} aria-expanded={requests[index].id === selected?.id ? detailsExpanded : undefined}
          onClick={() => onSelect(requests[index])} onFocus={() => setHoverPoint(point)} onBlur={() => setHoverPoint(null)} onKeyDown={event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onSelect(requests[index]) } }}>
          <circle className="monitor-request-point-hit" cx={x(point.time)} cy={finite(point.values.current) === null ? 150 : y(point.values.current!)} r="13" />
          <circle className={`monitor-generation-marker${finite(point.values.current) === null ? ' is-unknown' : ''}`} cx={x(point.time)} cy={finite(point.values.current) === null ? 150 : y(point.values.current!)} r={requests[index].id === selected?.id ? 4.5 : 3} />
        </g>)}
        {mode === 'realtime' && hover && finite(hover.values.current) !== null && <circle className="monitor-generation-marker" cx={x(hover.time)} cy={y(hover.values.current!)} r="3.5" />}
      </svg>
      {!segments.length && <div className="monitor-generation-empty">{mode === 'turn' ? requests.length ? '后端未报告平均速度' : '尚无模型请求' : '这次请求没有实时采样点'}</div>}
      {hover && <div className="monitor-generation-tooltip">{mode === 'turn' ? `${numbered ? `请求 #${hover.time}` : hoveredRequest ? requestIdentity(hoveredRequest) : '所选请求'} · ` : `${new Date(hover.wallTime).toLocaleTimeString('zh-CN', { hour12: false })} · `}{finite(hover.values.current) === null ? '未知' : `${numberText(hover.values.current, 2, true)} token/s`}</div>}
    </div>
    <div className="trend-axis"><span>{mode === 'turn' ? requests.length ? positionLabel(0) : '请求' : `${windowSeconds} 秒前`}</span><span>{mode === 'turn' ? requests.length > 2 ? positionLabel(Math.ceil(requests.length / 2) - 1) : '' : `${windowSeconds / 2} 秒前`}</span><span>{mode === 'turn' ? requests.length > 1 ? positionLabel(requests.length - 1) : '' : realtime?.ended ? '结束' : '现在'}</span></div>
  </div>
}

function TurnDuration({ turn, frozen = false }: { turn: TurnTotals; frozen?: boolean }) {
  const [elapsed, setElapsed] = useState(turn.elapsed)
  useEffect(() => {
    if (turn.status !== 'running' || frozen) { setElapsed(turn.elapsed); return }
    const origin = performance.now(), initial = turn.elapsed
    const timer = window.setInterval(() => setElapsed(initial + (performance.now() - origin) / 1000), 250)
    setElapsed(initial)
    return () => window.clearInterval(timer)
  }, [turn.startedAt, turn.elapsed, turn.status, frozen])
  return readout(turn.status === 'running' && !frozen ? elapsed : turn.elapsed, 's')
}
