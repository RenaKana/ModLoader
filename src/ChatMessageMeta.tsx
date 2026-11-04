import { useEffect, useState, type ReactNode } from 'react'
import { ArrowDown, ArrowUp, BarChart3, Clock3, Database } from 'lucide-react'
import type { Message } from './chat-history'
import type { LiveGeneration } from './types'
import { reportedNumber, responseSpeed } from './chat-response'
import { messageTime } from './chat-time'
import { engineRateTotal, footerRequest, requestTokens } from './chat-turn-metrics'

const numberFormat = new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 2 })
const number = (value: number) => numberFormat.format(value)
const rateFormat = new Intl.NumberFormat('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
const rate = (value: number) => rateFormat.format(value)

export function MessageTime({ message }: { message: Message }) {
  const time = messageTime(message)
  if (!time) return null
  return <time className="message-time" dateTime={time.iso} tabIndex={0} data-tooltip={`${time.label} · ${time.full}`} aria-label={`${time.label} ${time.full}`}>{time.short}</time>
}

function Metric({ name, label, detail = label, children, className = '' }: { name: string; label: string; detail?: string; children: ReactNode; className?: string }) {
  return <span className={`message-metric ${className}`} data-metric={name} tabIndex={0} aria-label={label} data-tooltip={detail}>{children}</span>
}

function MetricValue({ value, kind = 'tokens' }: { value: number | null | undefined; kind?: 'tokens' | 'rate' | 'elapsed' | 'ttft' | 'requests' }) {
  const text = value === undefined || value === null ? '—' : kind === 'rate' ? rate(value) : kind === 'elapsed' ? value.toFixed(1) : kind === 'ttft' ? value.toFixed(2) : number(value)
  return <span className={`message-metric-value message-metric-value-${kind}`}>{text}</span>
}

function ElapsedMetric({ generating, started, elapsed, turn = false }: { generating: boolean; started?: number; elapsed?: number; turn?: boolean }) {
  const [current, setCurrent] = useState(0)
  useEffect(() => {
    if (!generating || started === undefined) return
    const update = () => setCurrent(Math.max(0, performance.now() - started) / 1000)
    update()
    // Only this readout ticks: no transcript re-render or history save per tick.
    const timer = window.setInterval(update, 250)
    return () => window.clearInterval(timer)
  }, [generating, started])
  const value = generating ? started === undefined ? undefined : current : elapsed
  if (value === undefined) return null
  const label = turn ? '本轮耗时' : '请求耗时'
  return <Metric name="elapsed" label={`${label} ${value.toFixed(1)} 秒`}><Clock3 size={13} aria-hidden="true" /><MetricValue value={value} kind="elapsed" /><small>s</small></Metric>
}

export function LiveSpeed({ live, error }: { live?: LiveGeneration | null; error: string }) {
  const value = live?.available_for_chat ? reportedNumber(live.tokens_per_second) : undefined
  const detail = error || live?.reason || (value === undefined ? live === undefined ? '当前速度 · 等待后端采样' : '当前速度不可用'
    : `当前 Token 速度 · ${live?.window_s ? `近 ${number(live.window_s)} 秒滚动平均` : '滚动平均'}${live?.source ? `\n${live.source}` : ''}`)
  return <Metric name="live-speed" className="message-live-speed" label={value === undefined ? '当前 Token 速度不可用' : `当前 Token 速度 ${rate(value)} token/s`} detail={detail}>
    <MetricValue value={value} kind="rate" /><small>tok/s</small>
  </Metric>
}

export function MessageMetrics({ message, generating, started, live, liveError, onInspect }: { message: Message; generating: boolean; started?: number; live?: LiveGeneration | null; liveError: string; onInspect?: () => void }) {
  if (message.turn) {
    const turn = message.turn
    const request = footerRequest(message.requests ?? [], generating)
    const tokens = request ? requestTokens(request) : undefined
    const engineAverage = turn.decode.value === null ? engineRateTotal(message.requests ?? []) : null
    const displayedRate = engineAverage?.value !== null && engineAverage?.value !== undefined ? engineAverage : turn.decode
    const sampledRate = displayedRate.value === null && generating && request?.status === 'running' && live?.available_for_chat ? reportedNumber(live.tokens_per_second) : undefined
    const rateValue = sampledRate ?? displayedRate.value
    const rateLabel = sampledRate !== undefined ? '实时采样' : displayedRate === engineAverage ? '引擎平均' : 'Decode'
    const rateCode = sampledRate !== undefined ? 'TPS' : displayedRate === engineAverage ? 'AVG' : 'DEC'
    const scope = request?.status === 'running' && generating ? '当前请求' : '末次请求'
    const partial = (reported: number, value: number | null, partialReported = 0) => value !== null && (reported < turn.requests || partialReported > 0) ? <small className="message-metric-partial">部分</small> : null
    const label = (name: string, value: number | undefined) => `${scope}${name} ${value === undefined ? '未知' : number(value)} Token`
    const requests = <><BarChart3 size={13} aria-hidden="true" /><MetricValue value={turn.requests} kind="requests" /><small>次</small></>
    return <span className="message-turn-summary">
      <ElapsedMetric generating={generating} started={started} elapsed={turn.elapsed} turn />
      <Metric name="input" label={label('未缓存输入', tokens?.uncached)}><ArrowUp size={13} aria-hidden="true" /><MetricValue value={tokens?.uncached} /></Metric>
      <Metric name="cached" label={label('缓存命中', tokens?.cached)}><Database size={13} aria-hidden="true" /><MetricValue value={tokens?.cached} /></Metric>
      <Metric name="output" label={label('输出', tokens?.output)}><ArrowDown size={13} aria-hidden="true" /><MetricValue value={tokens?.output} /></Metric>
      <Metric name="speed" label={`${sampledRate !== undefined ? '当前请求' : '本轮'} ${rateLabel} ${rateValue === null ? '未知' : rate(rateValue)} token/s${sampledRate === undefined && displayedRate.value !== null && (displayedRate.reported < turn.requests || displayedRate.partial_reported) ? '（部分）' : ''}${rateCode === 'AVG' ? ' · 包含首 Token 等待' : ''}`}><span className="metric-abbr">{rateCode}</span><MetricValue value={rateValue} kind="rate" /><small>tok/s</small>{sampledRate === undefined && partial(displayedRate.reported, displayedRate.value, displayedRate.partial_reported)}</Metric>
      {onInspect ? <button type="button" className="message-metric message-turn-inspect" aria-label={`查看本轮 ${turn.requests} 次模型请求统计`} onClick={onInspect}>{requests}</button>
        : <span className="message-metric" aria-label={`本轮 ${turn.requests} 次模型请求`}>{requests}</span>}
    </span>
  }
  const input = requestTokens(message)
  const output = (generating ? reportedNumber(live?.generated_tokens) : undefined)
    ?? reportedNumber(message.usage?.completion_tokens) ?? reportedNumber(message.timings?.predicted_n)
  const speed = !message.error && !message.stopped && message.status !== 'interrupted' ? responseSpeed(message.usage, message.timings, message.elapsed) : undefined
  return <>
    <ElapsedMetric generating={generating} started={started} elapsed={message.elapsed} />
    {(input.input !== undefined || input.cached !== undefined || input.uncached !== undefined) && <>
      <Metric name="input" label={`未缓存输入 ${input.uncached === undefined ? '未知' : number(input.uncached)} Token`}><ArrowUp size={13} aria-hidden="true" /><MetricValue value={input.uncached} /></Metric>
      <Metric name="cached" label={`缓存命中 ${input.cached === undefined ? '未知' : number(input.cached)} Token`}><Database size={13} aria-hidden="true" /><MetricValue value={input.cached} /></Metric>
    </>}
    {(output !== undefined || generating) && <Metric name="output" label={output === undefined ? '本次输出 Token 等待采样' : `输出 ${number(output)} Token`} detail={generating ? `本次已输出 Token · ${output === undefined ? '等待后端采样' : '后端实时计数'}` : '输出 Token'}><ArrowDown size={13} aria-hidden="true" /><MetricValue value={output} /></Metric>}
    {generating ? <LiveSpeed live={live} error={liveError} /> : speed ? <Metric name="speed" label={`${speed.label} ${rate(speed.value)} token/s`} detail={`${speed.label} · ${speed.detail}`}>
      <span className="metric-abbr">{speed.label === '端到端平均' ? 'E2E' : 'DEC'}</span><MetricValue value={speed.value} kind="rate" /><small>tok/s</small>
    </Metric> : !message.error && !message.stopped && message.status !== 'interrupted' && <Metric name="speed" label="Token 速度不可用" detail="后端未返回本次请求的 Token 用量或生成耗时"><MetricValue value={undefined} kind="rate" /><small>tok/s</small></Metric>}
    {message.firstOutput !== undefined && <Metric name="first-output" label={`首输出等待 ${message.firstOutput.toFixed(2)} 秒`} detail="TTFT · 从发送请求到首段正文、思考或工具输出"><span className="metric-abbr">TTFT</span><MetricValue value={message.firstOutput} kind="ttft" /><small>s</small></Metric>}
  </>
}
