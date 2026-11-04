import { useEffect, useLayoutEffect, useRef, useState, useSyncExternalStore, type CSSProperties, type ReactNode } from 'react'
import { NumericReadout } from './ResourceCards'
import type { GenerationPhase } from './monitor-generation'
import { numberText } from './resource-metrics'
import { projectExternalTimeline, type ExternalTimelineStore, type ExternalTimelineState } from './external-generation-timeline'
import './external-generation-timeline.css'

const phases = ['decode', 'prefill'] as const
const phaseName = { decode: 'Decode', prefill: 'Prefill' }
const states: Record<ExternalTimelineState, string> = { idle: '空闲', queued: '排队中', waiting: '等待处理', prefill: 'Prefill', decode: 'Decode', concurrent: '并发请求', unknown: '速度缺测' }
const statuses = { running: '进行中', completed: '已完成', cancelled: '已中止', failed: '失败', incomplete: '已中断' }
const timeLabel = (time: number) => new Date(time).toLocaleTimeString('zh-CN', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit', fractionalSecondDigits: 3 })
type Hover = { id: string; requestKey: string; time: number; value?: number; phase?: GenerationPhase; kind: 'sample' | 'average' | 'request' | 'state'; seconds?: number; status?: keyof typeof statuses; state?: ExternalTimelineState }

export function ExternalGenerationTimeline({ store, windowSeconds, onCycleWindow, nextWindow, modelLabel, aside }: {
  store: ExternalTimelineStore; windowSeconds: number; nextWindow: number; onCycleWindow: () => void; modelLabel?: string; aside: ReactNode
}) {
  const snapshot = useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot)
  const [now, setNow] = useState(Date.now)
  const [width, setWidth] = useState(600)
  const [hover, setHover] = useState<Hover | null>(null)
  const graph = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 250)
    return () => window.clearInterval(timer)
  }, [])
  useLayoutEffect(() => {
    const element = graph.current
    if (!element) return
    const measure = () => setWidth(Math.max(1, element.getBoundingClientRect().width))
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(element)
    return () => observer.disconnect()
  }, [])
  const plot = projectExternalTimeline(snapshot, Math.max(now, snapshot.current.time), windowSeconds, width)
  const x = (time: number) => (time - plot.start) / (plot.end - plot.start) * width
  const y = (phase: GenerationPhase, value: number) => 150 - Math.min(1, value / plot.axes[phase].maximum) * 144
  const trace = (phase: GenerationPhase, points: { time: number; value: number }[]) => points.map((point, index) => `${index ? 'L' : 'M'}${x(point.time).toFixed(2)},${y(phase, point.value).toFixed(2)}`).join(' ')
  const hits: Hover[] = snapshot.requests.flatMap(request => phases.flatMap(phase => request.samples[phase]
    .filter(point => point.time >= plot.start && point.time <= plot.end && point.values.current !== null)
    .map(point => ({ id: `sample:${request.record.request_key}:${phase}:${point.time}`, requestKey: request.record.request_key, phase, time: point.time, value: point.values.current!, kind: 'sample' as const }))))
  hits.push(...plot.averages.map(marker => ({ ...marker, id: `average:${marker.requestKey}:${marker.phase}`, kind: 'average' as const })),
    ...plot.requests.map(marker => ({ ...marker, id: `request:${marker.requestKey}`, kind: 'request' as const })))
  const visibleSampleMarkers = new Set(phases.flatMap(phase => plot.series[phase].filter(segment => segment.marker)
    .map(segment => `sample:${segment.requestKey}:${phase}:${segment.points.at(-1)!.time}`)))
  const focusHits = hits.filter(hit => hit.kind !== 'sample' || visibleSampleMarkers.has(hit.id))
  const stateHits = plot.states.map(state => ({ ...state, id: `state:${state.since}:${state.state}`, requestKey: '', time: (state.start + state.end) / 2,
    seconds: (state.end - state.start) / 1000, kind: 'state' as const }))
  const currentHover = hover && (hover.kind === 'state' ? stateHits : hits).find(point => point.id === hover.id)
  const hitY = (hit: Hover) => hit.phase ? y(hit.phase, hit.value!) : 174
  const describe = (hit: Hover) => hit.kind === 'state' ? `${states[hit.state!]} · ${numberText(hit.seconds, 2)} s` : `${hit.phase ? `${phaseName[hit.phase]} ${hit.kind === 'average' ? '阶段平均' : '实时采样'} · ${numberText(hit.value, 2, true)} token/s${hit.kind === 'average' ? ` · 耗时 ${numberText(hit.seconds, 3)} s` : ''}`
    : `请求${statuses[hit.status!]}`} · ${timeLabel(hit.time)} · ${hit.requestKey}`
  return <section className="panel monitor-compact-generation monitor-external-generation" aria-label="生成监测" data-scope="request" data-state={plot.state}>
    <div className="monitor-compact-heading">
      <div className="monitor-compact-switches"><span className="monitor-external-title">外部实时</span>
        <button type="button" className="monitor-window-cycle" aria-label={`曲线窗口 ${windowSeconds} 秒，切换到 ${nextWindow} 秒`} data-tooltip="切换曲线时间窗口" onClick={onCycleWindow}>{windowSeconds}s</button>
      </div>
      <span className="monitor-compact-identity">{modelLabel && <span className="monitor-compact-model" title={modelLabel}>{modelLabel} · </span>}{states[plot.state]}</span>
    </div>
    <div className="monitor-compact-body">
      <div className="monitor-compact-main">
        <dl className="monitor-dual-readouts">{phases.map(phase => <div key={phase} data-phase={phase} aria-label={`${phaseName[phase]} 实时速度`}>
          <dt>{phaseName[phase]} <span>{phase === 'decode' ? '左轴' : '右轴'}</span></dt>
          <dd className="monitor-speed-current"><NumericReadout value={plot.readouts[phase] === null ? '—' : numberText(plot.readouts[phase], 2, true)} unit={plot.readouts[phase] === null ? undefined : 'token/s'} stacked /></dd>
        </div>)}</dl>
        <div className="monitor-generation-chart monitor-dual-chart" data-mode="external-realtime" data-run={snapshot.runId} data-instance={snapshot.instanceId}
          data-window-end={plot.end} style={{ '--decode-axis-width': `max(40px, calc(${numberText(plot.axes.decode.maximum, 1).length}ch + 9px))`,
            '--prefill-axis-width': `max(40px, calc(${numberText(plot.axes.prefill.maximum, 1).length}ch + 9px))` } as CSSProperties}>
          <div className="monitor-generation-graph" ref={graph}>
            {phases.map(phase => <div key={phase} className={`monitor-dual-ticks is-${phase}`} aria-label={`${phaseName[phase]} ${phase === 'decode' ? '左轴' : '右轴'}，token/s`}>
              {plot.axes[phase].ticks.map(value => <span key={value} className="monitor-generation-tick" style={{ top: `${y(phase, value) / 190 * 100}%` }}>{numberText(value, 1)}</span>)}
            </div>)}
            <svg viewBox={`0 0 ${width} 190`} preserveAspectRatio="none" role="group" aria-label="外部请求 Prefill 与 Decode 双轴实时速度"
              onPointerMove={event => {
                const rect = event.currentTarget.getBoundingClientRect(), px = event.clientX - rect.left, py = (event.clientY - rect.top) / rect.height * 190
                if (py >= 157 && py <= 170) {
                  const time = plot.start + px / width * (plot.end - plot.start)
                  setHover(stateHits.find(state => state.start <= time && state.end >= time) ?? null)
                  return
                }
                let nearest: Hover | null = null, distance = 18
                for (const hit of hits) { const candidate = Math.hypot(x(hit.time) - px, hitY(hit) - py); if (candidate <= distance) { nearest = hit; distance = candidate } }
                setHover(nearest)
              }} onPointerLeave={() => setHover(null)}>
              {[0, 1, 2, 3].map(index => <path key={index} className="trend-grid" d={`M0 ${150 - index * 48}H${width}`} />)}
              <g>{plot.states.map((state, index) => <g key={index} data-timeline-state={state.state}>
                {state.state === 'idle' && <path className="monitor-idle-zero" aria-hidden="true" d={`M${x(state.start)} 150H${x(state.end)}`} />}
                <path className="monitor-state-band" data-state={state.state} d={`M${x(state.start)} 164H${x(state.end)}`} tabIndex={0} role="img"
                  aria-label={describe(stateHits[index])} onFocus={() => setHover(stateHits[index])} onBlur={() => setHover(null)} />
              </g>)}</g>
              {phases.map(phase => <g key={phase} data-series={phase}>{plot.series[phase].map((segment, index) => <g key={`${segment.requestKey}:${index}`} data-request-key={segment.requestKey}>
                {segment.points.length > 1 && <path className="trend-line" d={trace(phase, segment.points)} />}
                {segment.marker && <circle className="trend-dot" cx={x(segment.points.at(-1)!.time)} cy={y(phase, segment.points.at(-1)!.value)} r="3.5" />}
              </g>)}</g>)}
              {plot.averages.map(marker => <circle key={`${marker.requestKey}:${marker.phase}`} className="monitor-phase-average" data-phase={marker.phase} data-request-key={marker.requestKey}
                data-time={marker.time} cx={x(marker.time)} cy={y(marker.phase, marker.value)} r="4.5" />)}
              {plot.requests.map(marker => <g key={marker.requestKey} className="monitor-timeline-request" data-request-key={marker.requestKey} data-status={marker.status} transform={`translate(${x(marker.time)} 174)`}>
                {marker.status === 'failed' ? <path d="M-3 -3L3 3M-3 3L3 -3" /> : marker.status === 'cancelled' || marker.status === 'incomplete' ? <path d="M0 -4L4 0L0 4L-4 0Z" />
                  : marker.status === 'running' ? <path d="M0 -4V4" /> : <circle r="2.5" />}
              </g>)}
              {currentHover && currentHover.kind !== 'state' && <path className="monitor-generation-guide" d={`M${x(currentHover.time)} 6V178`} />}
              {focusHits.map(hit => <circle key={hit.id} className="monitor-timeline-hit" data-kind={hit.kind} data-phase={hit.phase} data-request-key={hit.requestKey}
                cx={x(hit.time)} cy={hitY(hit)} r="11" tabIndex={0} role="img" aria-label={describe(hit)} onFocus={() => setHover(hit)} onBlur={() => setHover(null)} />)}
            </svg>
            {currentHover && <div className="monitor-generation-tooltip" role="status">{describe(currentHover)}</div>}
          </div>
          <div className="trend-axis"><span>{windowSeconds} 秒前</span><span>{windowSeconds / 2} 秒前</span><span>现在</span></div>
        </div>
      </div>
      {aside}
    </div>
  </section>
}
