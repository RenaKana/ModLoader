import { useEffect, useId, useLayoutEffect, useRef, useState, type CSSProperties } from 'react'
import { ChevronDown } from 'lucide-react'
import { apiFetch, errorMessage } from './api'
import { DiagnosticNotice, ErrorNotice } from './components'
import { numberText } from './resource-metrics'
import { nearestUsageSlot, nextUsageBoundary, usageHitRange, usagePartial, usagePaths, usagePlotValue, usageSeries, usageSlots, usageValue, type UsageSeries } from './monitor-usage'
import type { RequestHistoryUsage } from './types'
import './monitor-usage.css'

function useUsageSnapshot(query: string, active: boolean, revision: string | undefined, live: boolean, refreshVersion: number) {
  const [shown, setShown] = useState<{ query: string; data: RequestHistoryUsage } | null>(null)
  const [latest, setLatest] = useState<{ query: string; data: RequestHistoryUsage } | null>(null)
  const [error, setError] = useState(''), [pending, setPending] = useState(false)
  const current = useRef(shown), held = useRef<typeof shown>(null), inspecting = useRef(false)
  const read = useRef<(() => void) | null>(null), seenRevision = useRef(revision)
  const inspect = (value: boolean) => {
    inspecting.current = value
    if (!value && held.current) { current.current = held.current; setShown(held.current); held.current = null; setPending(false) }
  }
  useEffect(() => {
    if (!active || !query) return
    const controller = new AbortController()
    let reading = false, again = false
    seenRevision.current = revision
    held.current = null; setPending(false); setError('')
    if (current.current?.query !== query) { current.current = null; setShown(null); setLatest(null) }
    async function load() {
      if (reading) { again = true; return }
      reading = true
      try {
        const response = await apiFetch(`/api/request-history/usage?${query}`, { signal: controller.signal })
        const data = await response.json() as RequestHistoryUsage
        if (controller.signal.aborted) return
        const candidate = { query, data }
        setLatest(candidate); setError('')
        if (inspecting.current && current.current?.query === query) { held.current = candidate; setPending(true) }
        else { current.current = candidate; held.current = null; setShown(candidate); setPending(false) }
      } catch (issue) { if (!controller.signal.aborted) setError(errorMessage(issue)) }
      finally { reading = false; if (again && !controller.signal.aborted) { again = false; void load() } }
    }
    const reload = () => { void load() }
    read.current = reload; reload()
    return () => { controller.abort(); if (read.current === reload) read.current = null }
  }, [active, query, refreshVersion])
  useEffect(() => {
    const previous = seenRevision.current
    seenRevision.current = revision
    if (active && previous !== undefined && revision !== undefined && previous !== revision) read.current?.()
  }, [active, revision])
  const cadence = latest?.query === query ? latest.data : null
  useEffect(() => {
    if (!active || !cadence || !(live || cadence.summary.status_counts.running > 0)) return
    const next = nextUsageBoundary(cadence.trend.next_refresh_at, cadence.trend.bucket_seconds, Date.now() / 1000)
    const timer = window.setTimeout(() => read.current?.(), Math.max(0, next * 1000 - Date.now()))
    return () => window.clearTimeout(timer)
  }, [active, live, cadence])
  return { data: shown?.query === query ? shown.data : null, error, pending, inspect, retry: () => read.current?.() }
}

const timeText = (seconds: number) => new Date(seconds * 1000).toLocaleTimeString('zh-CN', { hour12: false, hour: '2-digit', minute: '2-digit' })
const dateText = (seconds: number) => new Date(seconds * 1000).toLocaleDateString('zh-CN', { month: '2-digit', day: '2-digit' })
const valueText = (value: number | null, key: UsageSeries) => value == null ? '—' : `${numberText(value, key === 'hit' ? 1 : 0, key === 'hit')}${key === 'hit' ? '%' : ''}`

export function MonitorUsage({ query, active = true, revision, live = false, refreshVersion = 0 }: {
  query: string; active?: boolean; revision?: string; live?: boolean; refreshVersion?: number
}) {
  const { data, error, pending, inspect, retry } = useUsageSnapshot(query, active, revision, live, refreshVersion)
  const [expanded, setExpanded] = useState(false)
  const [visible, setVisible] = useState<Record<UsageSeries, boolean>>({ total: true, hit: true, input: false, output: false, cached: false })
  const id = useId(), toggle = useRef<HTMLButtonElement>(null), pointer = useRef(false), chart = useRef<HTMLDivElement>(null)
  const summary = data?.summary
  const partial = summary && summary.request_count > 0 && [summary.total, summary.input, summary.output, summary.cached, summary.cache_pair].some(item => item.reported < summary.request_count)
  const token = (key: 'total' | 'input' | 'output' | 'cached', label: string) => <div><div className="monitor-summary-label">{label}</div><strong data-summary={key}>{summary?.[key].value == null ? '—' : numberText(summary[key].value, 0)}</strong></div>
  const updateInspection = () => inspect(pointer.current || !!chart.current?.contains(document.activeElement))
  return <section className="monitor-usage" data-usage-revision={data?.revision} data-usage-pending={pending}>
    <div className="monitor-usage-overview">
      <button type="button" className="monitor-usage-toggle" ref={toggle} aria-label={`${expanded ? '收起' : '展开'} Token 用量趋势`} aria-expanded={expanded} aria-controls={id} onClick={() => setExpanded(value => !value)} />
      <header className="monitor-usage-heading"><h3>Token 用量</h3><span>{partial && <small>部分数据</small>}{data && `统计截至 ${timeText(data.as_of)}`}</span><ChevronDown size={16} aria-hidden="true" /></header>
      <section className="monitor-history-summary" aria-label="筛选范围汇总">
        <div className="monitor-usage-primary">{token('total', '总 Token')}<div><div className="monitor-summary-label">请求数</div><strong data-summary="count">{summary ? numberText(summary.request_count, 0) : '—'}</strong></div><div><div className="monitor-summary-label">缓存命中率</div><strong data-summary="hit">{valueText(summary?.cache_ratio == null ? null : summary.cache_ratio * 100, 'hit')}</strong></div></div>
        <div className="monitor-usage-secondary">{token('input', '输入（含缓存）')}{token('output', '输出')}{token('cached', '缓存命中量')}</div>
      </section>
    </div>
    {error && <ErrorNotice message={error} retry={retry} />}
    {!!data?.issues.length && <DiagnosticNotice summary="用量读取异常" detail={data.issues.join('\n')} />}
    <div className="monitor-usage-collapse" data-open={expanded} id={id} aria-hidden={!expanded} inert={!expanded}>
      <div className="monitor-usage-clip"><div className="monitor-usage-chart" ref={chart}
        onPointerEnter={() => { pointer.current = true; updateInspection() }} onPointerLeave={() => { pointer.current = false; updateInspection() }}
        onFocusCapture={updateInspection} onBlurCapture={() => queueMicrotask(updateInspection)}>
        <UsageChart key={query} data={data} visible={visible} onToggle={key => setVisible(previous => ({ ...previous, [key]: !previous[key] }))} onEscape={() => { toggle.current?.focus({ preventScroll: true }); pointer.current = false; inspect(false) }} />
      </div></div>
    </div>
  </section>
}

function UsageChart({ data, visible, onToggle, onEscape }: { data: RequestHistoryUsage | null; visible: Record<UsageSeries, boolean>; onToggle: (key: UsageSeries) => void; onEscape: () => void }) {
  const plot = useRef<HTMLDivElement>(null), tooltip = useRef<HTMLDivElement>(null), hits = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(600), [hover, setHover] = useState<number | null>(null), [tipWidth, setTipWidth] = useState(220)
  const gradient = useId(), tooltipId = useId()
  useLayoutEffect(() => {
    const node = plot.current
    if (!node) return
    const resize = () => { const measured = node.getBoundingClientRect().width; if (measured) setWidth(measured) }
    resize(); const observer = new ResizeObserver(resize); observer.observe(node)
    return () => observer.disconnect()
  }, [])
  useEffect(() => { setHover(null) }, [data])
  useLayoutEffect(() => { if (tooltip.current && hover !== null) setTipWidth(tooltip.current.getBoundingClientRect().width) }, [hover, visible])
  const buckets = data?.trend.buckets ?? [], left = 64, right = width - 48, top = 22, bottom = 190
  const slots = usageSlots(buckets.length, left, right), bucket = hover == null ? undefined : buckets[hover]
  const enabled = usageSeries.filter(series => visible[series.key])
  const peak = Math.max(0, ...buckets.flatMap(item => enabled.filter(series => series.key !== 'hit').map(series => usageValue(item, series.key) ?? 0)))
  const magnitude = 10 ** Math.floor(Math.log10(Math.max(1, peak * 1.08) / 3))
  const step = [1, 2, 5, 10].find(value => value * magnitude >= Math.max(1, peak * 1.08) / 3)! * magnitude, maximum = step * 3
  const y = (key: UsageSeries, value: number) => bottom - value / (key === 'hit' ? 100 : maximum) * (bottom - top)
  const traces = Object.fromEntries(usageSeries.map(series => [series.key, usagePaths(buckets.map((item, index) => ({ x: slots[index], value: usagePlotValue(item, series.key) })), value => y(series.key, value), bottom)])) as Record<UsageSeries, { line: string; area: string }>
  const interval = (start: number, end: number) => `${dateText(start)} ${timeText(start)}–${dateText(start) === dateText(end) ? '' : `${dateText(end)} `}${timeText(end)}`
  const formatAxis = (value: number) => new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 }).format(value)
  const start = data?.trend.start_at, end = data?.trend.end_at
  const crossDate = start != null && end != null && dateText(start) !== dateText(end)
  const stride = Math.max(1, Math.ceil(buckets.length / Math.max(2, Math.floor((right - left) / 80))))
  const tickIndices = [...new Set(buckets.flatMap((_, index) => index % stride === 0 || index === buckets.length - 1 ? [index] : []))]
  const slot = hover == null ? left : slots[hover]
  const tipLeft = Math.max(8, Math.min(width - tipWidth - 8, slot + 14 + tipWidth > width - 8 ? slot - tipWidth - 14 : slot + 14))
  return <>
    <header className="monitor-usage-chart-heading"><h4>用量趋势</h4><span data-usage-range>{start != null && end != null ? `${dateText(start)} ${timeText(start)}—${crossDate ? `${dateText(end)} ` : ''}${timeText(end)} · 每 ${data!.trend.bucket_seconds / 60} 分钟` : data ? '无记录' : '正在读取用量…'}</span></header>
    <div className="monitor-usage-plot" ref={plot} onPointerMove={event => { if (buckets.length) { const rect = event.currentTarget.getBoundingClientRect(); setHover(nearestUsageSlot(event.clientX - rect.left, buckets.length, left, right)) } }}
      onPointerLeave={() => { if (!hits.current?.contains(document.activeElement)) setHover(null) }}>
      <svg viewBox={`0 0 ${width} 252`} role="img" aria-label="按请求开始时间汇总的 Token 与缓存命中率，左轴 Token 数，右轴百分比" data-usage-token-max={maximum} data-usage-hit-max="100">
        <defs><linearGradient id={gradient} x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="var(--blue)" stopOpacity=".18" /><stop offset=".65" stopColor="var(--blue)" stopOpacity=".05" /><stop offset="1" stopColor="var(--blue)" stopOpacity="0" /></linearGradient></defs>
        {[0, 1, 2, 3].map(index => <g key={index}><path className="trend-grid" d={`M${left} ${y('total', step * index)}H${right}`} /><text className="monitor-usage-axis" x={left - 8} y={y('total', step * index) + 4} textAnchor="end">{formatAxis(step * index)}</text></g>)}
        {[0, 50, 100].map(value => <text key={value} className="monitor-usage-axis" x={right + 8} y={y('hit', value) + 4}>{value}%</text>)}
        {visible.total && <path data-usage-area="total" d={traces.total.area} fill={`url(#${gradient})`} />}
        {enabled.map(series => <path key={series.key} className="monitor-usage-line" data-usage-series={series.key} d={traces[series.key].line} />)}
        {bucket && <><path className="monitor-usage-guide" data-usage-guide d={`M${slot} ${top}V${bottom}`} />{enabled.map(series => { const value = usagePlotValue(bucket, series.key); return value == null ? null : <circle className="monitor-usage-marker" key={series.key} data-usage-marker={series.key} cx={slot} cy={y(series.key, value)} r="3.5" /> })}</>}
        {tickIndices.map(index => <text key={index} className="monitor-usage-axis" data-usage-time-label x={slots[index]} y="216" textAnchor={buckets.length === 1 ? 'middle' : index === 0 ? 'start' : index === buckets.length - 1 ? 'end' : 'middle'}>{timeText(buckets[index].start_at)}{crossDate && <tspan x={slots[index]} dy="14">{dateText(buckets[index].start_at)}</tspan>}</text>)}
        <text className="monitor-usage-axis" x={left} y="12">Token</text><text className="monitor-usage-axis" x={right} y="12" textAnchor="end">命中率 %</text><text className="monitor-usage-axis" x={(left + right) / 2} y="249" textAnchor="middle">时间</text>
      </svg>
      <div className="monitor-usage-hits" ref={hits} onBlur={() => queueMicrotask(() => { if (!hits.current?.contains(document.activeElement)) setHover(null) })}>{buckets.map((item, index) => {
        const [hitLeft, hitRight] = usageHitRange(slots, index, left, right)
        return <button type="button" key={item.start_at} data-usage-slot={item.start_at} data-slot-x={slots[index]} aria-describedby={hover === index ? tooltipId : undefined}
          aria-label={`${interval(item.start_at, item.end_at)}，${item.request_count} 次请求${usagePartial(item) ? '，部分数据' : ''}，${enabled.map(series => `${series.label} ${valueText(usageValue(item, series.key), series.key)}`).join('，')}`}
          style={{ left: hitLeft, width: hitRight - hitLeft, top, height: bottom - top }} onFocus={() => setHover(index)} onClick={() => setHover(index)}
          onKeyDown={event => { if (['ArrowLeft', 'ArrowRight', 'Home', 'End', 'Escape'].includes(event.key)) { event.preventDefault(); if (event.key === 'Escape') { setHover(null); onEscape(); return }; const next = event.key === 'Home' ? 0 : event.key === 'End' ? buckets.length - 1 : Math.max(0, Math.min(buckets.length - 1, index + (event.key === 'ArrowRight' ? 1 : -1))); hits.current?.querySelectorAll<HTMLButtonElement>('button')[next]?.focus() } }} />
      })}</div>
      <div className="monitor-usage-tooltip" role="tooltip" ref={tooltip} id={tooltipId} hidden={!bucket} style={{ left: tipLeft, top: 34 } as CSSProperties}>{bucket && <>
        <header><span>{interval(bucket.start_at, bucket.end_at)}</span>{usagePartial(bucket) && <span className="monitor-usage-partial">部分数据</span>}</header><div className="monitor-usage-tooltip-count">{bucket.request_count} 次请求</div>
        <div className="monitor-usage-tooltip-rows">{enabled.map(series => <div key={series.key} data-usage-tooltip-row={series.key}><span><i aria-hidden="true" />{series.label}</span><strong>{valueText(usageValue(bucket, series.key), series.key)}</strong></div>)}</div>
      </>}</div>
      {!buckets.length && <div className="monitor-usage-empty">{data ? '没有符合筛选条件的记录' : '正在读取用量…'}</div>}
    </div>
    <div className="monitor-usage-legend" aria-label="用量趋势曲线">{usageSeries.map(series => <button type="button" key={series.key} data-usage-legend={series.key} aria-pressed={visible[series.key]} onClick={() => onToggle(series.key)}><i aria-hidden="true" /><span>{series.label}</span></button>)}</div>
  </>
}
