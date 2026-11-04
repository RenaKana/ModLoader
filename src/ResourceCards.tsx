import { useId, useRef, useState, type ReactNode, type ElementType } from 'react'
import { Cpu, CircuitBoard, MemoryStick, HardDrive } from 'lucide-react'
import { Button, DiagnosticNotice, Select, formatBytes } from './components'
import type { Hardware } from './types'
import { adaptiveBounds, finite, gpuKey, gpuLabel, gpuList, numberText, pcieLink, seriesSegments, HISTORY_MS, type ChartBounds, type HistoryPoint } from './resource-metrics'
import './resource-cards.css'

export function NumericReadout({ value, unit, stacked = false }: { value: ReactNode; unit?: string; stacked?: boolean }) {
  return <span className={`monitor-number${stacked ? ' is-stacked' : ''}`}><strong>{value}</strong>{unit && <small>{unit}</small>}</span>
}

export function Facts({ items }: { items: Array<[string, ReactNode]> }) {
  return <dl className="resource-facts">{items.map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{value ?? '不可用'}</dd></div>)}</dl>
}

export function ResourceDetail({ title, note, onClose, children }: { title: string; note?: ReactNode; onClose: () => void; children: ReactNode }) {
  return <section className="panel resource-detail" aria-label={`${title}详情`}><div className="resource-detail-heading"><div><h3>{title}</h3>{note && <span>{note}</span>}</div><Button small variant="ghost" onClick={onClose}>收起</Button></div>{children}</section>
}

export function ResourceTile({ id, label, icon: Icon, value, unit, note, extra, selected, onClick, children }: {
  id: string; label: string; icon: ElementType; value: ReactNode; unit?: string; note?: ReactNode; extra?: ReactNode;
  selected: boolean; onClick: () => void; children?: ReactNode
}) {
  return <button type="button" className={`panel resource-tile ${selected ? 'selected' : ''}`} data-resource={id} aria-label={`${label}详情`} aria-expanded={selected} onClick={onClick}>
    <span className="resource-tile-label"><Icon size={16} />{label}</span><span className={`resource-tile-value ${unit ? '' : 'text-value'}`}>{value}{unit && <small>{unit}</small>}</span>
    {note && <span className="resource-tile-note">{note}</span>}{extra && <span className="resource-tile-meta">{extra}</span>}{children}
  </button>
}

export function TrendChart({ points, metric, label, unit, maximum, secondary, adaptive = false, expanded = false }: {
  points: HistoryPoint[]; metric: string; label: string; unit: string; maximum?: number | null;
  secondary?: { metric: string; label: string }; adaptive?: boolean; expanded?: boolean
}) {
  const [hover, setHover] = useState<HistoryPoint | null>(null)
  const bounds = useRef<ChartBounds | undefined>(undefined)
  const series = [{ metric, label }, ...(secondary ? [secondary] : [])]
  const segments = series.map(item => seriesSegments(points, item.metric)), end = points.at(-1)?.time ?? 0
  const peak = Math.max(1, ...points.flatMap(p => series.map(item => finite(p.values[item.metric]) ?? 0)))
  const minimumSpan = unit === 'W' ? 20 : 5
  if (adaptive) bounds.current = adaptiveBounds(points.flatMap(point => series.map(item => finite(point.values[item.metric]))), minimumSpan, end, points.length > 1 ? bounds.current : undefined)
  const min = adaptive ? bounds.current!.minimum : 0
  const max = adaptive ? bounds.current!.maximum : Math.max(maximum ?? 0, maximum ? peak : Math.ceil(peak * 1.1))
  const x = (time: number) => Math.max(0, Math.min(600, (time - end + HISTORY_MS) / HISTORY_MS * 600))
  const y = (value: number) => 98 - Math.max(0, Math.min(1, (value - min) / (max - min))) * 96
  const format = (value: unknown, digits = 2) => numberText(value, unit === 'token/s' ? 2 : digits, unit === 'token/s')
  const readout = (point: HistoryPoint | undefined) => series.map(item => `${secondary ? `${item.label} ` : ''}${format(point?.values[item.metric], 3)} ${finite(point?.values[item.metric]) === null ? '' : unit}`).join(' · ')
  return <div className={`trend-plot ${expanded ? 'expanded' : ''}`}>
    {expanded && <div className={`trend-caption${secondary ? ' dual' : ''}`}>
      {secondary ? <span className="trend-legend">{series.map((item, index) => <span key={item.metric} className={`trend-legend-item${index ? ' secondary' : ''}`}>{item.label}</span>)}</span> : <span>{label}</span>}
      <span>{numberText(min)}–{numberText(max)} {unit}</span>
    </div>}
    <svg viewBox="0 0 600 100" preserveAspectRatio="none" role="img" aria-label={`${label}最近 60 秒趋势，当前${readout(points.at(-1))}`}
      onPointerMove={expanded ? event => { const rect = event.currentTarget.getBoundingClientRect(); const time = end - HISTORY_MS + (event.clientX - rect.left) / rect.width * HISTORY_MS; setHover(points.reduce<HistoryPoint | null>((best, p) => !best || Math.abs(p.time - time) < Math.abs(best.time - time) ? p : best, null)) } : undefined}
      onPointerLeave={() => setHover(null)}>
      <path className="trend-grid" d={expanded ? 'M0 2H600 M0 26H600 M0 50H600 M0 74H600 M0 98H600 M150 0V100 M300 0V100 M450 0V100' : 'M0 98H600'} />
      {segments.map((parts, seriesIndex) => <g key={series[seriesIndex].metric} data-series={series[seriesIndex].metric} className={seriesIndex ? 'trend-secondary' : undefined}>{parts.map((segment, index) => segment.length === 1 ? <circle key={index} cx={x(segment[0].time)} cy={y(segment[0].value)} r="1.6" className="trend-dot" /> : <path key={index} className="trend-line" d={segment.map((p, i) => `${i ? 'L' : 'M'}${x(p.time).toFixed(2)},${y(p.value).toFixed(2)}`).join(' ')} />)}</g>)}
    </svg>
    {expanded && <><div className="trend-axis"><span>60 秒前</span><span>30 秒前</span><span>现在</span></div><div className="trend-readout">{hover ? `${new Date(hover.time).toLocaleTimeString('zh-CN', { hour12: false })} · ${readout(hover)}` : !segments.some(parts => parts.length) ? '等待有效采样' : '\u00a0'}</div></>}
  </div>
}

export function HardwareTrendChart({ points, metrics, label }: {
  points: HistoryPoint[]; label: string;
  metrics: Array<{ id: string; label: string; unit: string; max: number | null; adaptive?: boolean }>
}) {
  const [hoverTime, setHoverTime] = useState<number | null>(null)
  const bounds = useRef<Record<string, ChartBounds>>({})
  const end = points.at(-1)?.time ?? 0
  // Keep like units comparable (PCIe RX/TX), without mixing %, W or °C.
  for (const unit of new Set(metrics.map(metric => metric.unit))) {
    const matching = metrics.filter(metric => metric.unit === unit)
    const values = points.flatMap(point => matching.map(metric => finite(point.values[metric.id])))
    const maximum = Math.max(0, ...matching.map(metric => metric.max ?? 0))
    const peak = Math.max(1, ...values.map(value => value ?? 0))
    bounds.current[unit] = matching.some(metric => metric.adaptive)
      ? adaptiveBounds(values, unit === 'W' ? 20 : 5, end, points.length > 1 ? bounds.current[unit] : undefined)
      : { minimum: 0, maximum: Math.max(maximum, maximum ? peak : Math.ceil(peak * 1.1)), updatedAt: end }
  }
  const series = metrics.map(metric => ({ ...metric, bounds: bounds.current[metric.unit], segments: seriesSegments(points, metric.id) }))
  const hovered = hoverTime === null ? null : points.reduce<HistoryPoint | null>((best, point) => !best || Math.abs(point.time - hoverTime) < Math.abs(best.time - hoverTime) ? point : best, null)
  const reading = hovered ?? points.at(-1)
  const valueText = (point: HistoryPoint | undefined, metric: typeof metrics[number]) => finite(point?.values[metric.id]) === null ? '不可用' : `${numberText(point?.values[metric.id], 3)} ${metric.unit}`
  const x = (time: number) => Math.max(0, Math.min(600, (time - end + HISTORY_MS) / HISTORY_MS * 600))
  const y = (value: number, range: ChartBounds) => 98 - Math.max(0, Math.min(1, (value - range.minimum) / (range.maximum - range.minimum))) * 96
  return <div className="trend-plot expanded hardware-trend-chart">
    <div className="hardware-trend-legend" role="list" aria-label="曲线与刻度">{series.map((metric, index) => <div key={metric.id} className="hardware-trend-legend-item" data-metric={metric.id} data-series-index={index} role="listitem">
      <span className="hardware-trend-label">{metric.label}</span>
      <strong>{valueText(reading, metric)}</strong>
      <span className="hardware-trend-range">{numberText(metric.bounds.minimum)}–{numberText(metric.bounds.maximum)} {metric.unit}</span>
    </div>)}</div>
    <svg viewBox="0 0 600 100" preserveAspectRatio="none" role="img" aria-label={`${label}最近 60 秒趋势，${metrics.map(metric => `${metric.label}当前${valueText(points.at(-1), metric)}`).join('，')}`}
      onPointerMove={event => { const rect = event.currentTarget.getBoundingClientRect(); setHoverTime(end - HISTORY_MS + (event.clientX - rect.left) / rect.width * HISTORY_MS) }}
      onPointerLeave={() => setHoverTime(null)}>
      <path className="trend-grid" d="M0 2H600 M0 26H600 M0 50H600 M0 74H600 M0 98H600 M150 0V100 M300 0V100 M450 0V100" />
      {series.map((metric, seriesIndex) => <g key={metric.id} data-series={metric.id} data-series-index={seriesIndex}>
        {metric.segments.map((segment, index) => segment.length === 1
          ? <circle key={index} cx={x(segment[0].time)} cy={y(segment[0].value, metric.bounds)} r="1.6" className="trend-dot" />
          : <path key={index} className="trend-line" d={segment.map((point, i) => `${i ? 'L' : 'M'}${x(point.time).toFixed(2)},${y(point.value, metric.bounds).toFixed(2)}`).join(' ')} />)}
      </g>)}
      {hovered && <><path className="hardware-trend-guide" d={`M${x(hovered.time)} 0V100`} />{series.map((metric, index) => finite(hovered.values[metric.id]) !== null && <circle key={metric.id} data-series-index={index} className="hardware-trend-marker" cx={x(hovered.time)} cy={y(hovered.values[metric.id]!, metric.bounds)} r="2.2" />)}</>}
    </svg>
    <div className="trend-axis"><span>60 秒前</span><span>30 秒前</span><span>现在</span></div>
    <div className="trend-readout"><span>{hovered ? new Date(hovered.time).toLocaleTimeString('zh-CN', { hour12: false }) : series.some(metric => metric.segments.length) ? '当前' : '等待有效采样'}</span><span>按单位独立刻度</span></div>
  </div>
}

function capacity(used: unknown, total: unknown) {
  const u = finite(used), t = finite(total)
  return <><strong>{u === null ? '未知' : numberText(u / 1024 ** 3, 1)}</strong><span className="capacity-total">/ {t === null ? '未知' : numberText(t / 1024 ** 3, 1)}</span><small>GB</small></>
}

export default function HardwareResources({ hardware }: { hardware: Hardware }) {
  const [selected, setSelected] = useState<string | null>(null), [device, setDevice] = useState('')
  const rootId = useId(), gpus = gpuList(hardware), gpu = gpus.find(g => gpuKey(g) === device) ?? gpus[0]
  const cpu = hardware.cpu_info, modules = hardware.memory_modules ?? []
  const memoryType = [...new Set(modules.map(m => m.type).filter(Boolean))].join(' / ')
  const speed = [...new Set(modules.map(m => m.speed_mt_s).filter(Boolean))].join(' / ')
  const total = hardware.memory_total ?? hardware.ram_total_bytes
  const details: Record<string, { label: string; note?: string; facts: Array<[string, ReactNode]> }> = {
    cpu: { label: 'CPU', note: cpu?.name, facts: [['核心 / 线程', `${numberText(cpu?.cores)} / ${numberText(cpu?.threads ?? hardware.cpu_count)}`], ['系统报告频率', cpu?.max_clock_mhz ? `${numberText(cpu.max_clock_mhz)} MHz` : null], ['L2 缓存', formatBytes(cpu?.l2_bytes)], ['L3 缓存', formatBytes(cpu?.l3_bytes)]] },
    memory: { label: '系统内存', facts: [['已用 / 总量', capacity(hardware.memory_used, total)], ['可用', formatBytes(hardware.memory_available)], ['内存类型', memoryType || null], ['配置速率', speed ? `${speed} MT/s` : null], ...modules.map((m, i): [string, ReactNode] => [m.slot || `内存条 ${i + 1}`, formatBytes(m.capacity)])] },
    gpu: { label: 'GPU', note: typeof gpu?.name === 'string' ? gpu.name : undefined, facts: [['CUDA 计算能力', String(gpu?.compute_capability ?? '不可用')], ['CUDA 核心', numberText(gpu?.cuda_cores, 0)], ['PCIe 链路上限', pcieLink(gpu, true)], ['当前 PCIe 链路', pcieLink(gpu)], ['显存位宽', finite(gpu?.memory_bus_bits) === null ? null : `${gpu?.memory_bus_bits} bit`]] },
    vram: { label: '显存', facts: [['已用 / 总量', capacity(gpu?.memory_used, gpu?.memory_total)], ['剩余', formatBytes(finite(gpu?.memory_free))], ['显存位宽', finite(gpu?.memory_bus_bits) === null ? null : `${gpu?.memory_bus_bits} bit`], ['最大显存时钟', finite(gpu?.memory_max_clock_mhz) === null ? null : `${numberText(gpu?.memory_max_clock_mhz)} MHz`], ['显存理论带宽', finite(gpu?.memory_bandwidth_bytes_s) === null ? null : `${numberText(Number(gpu?.memory_bandwidth_bytes_s) / 1e9, 1)} GB/s`]] },
  }
  const tiles = [
    { id: 'cpu', label: 'CPU', icon: Cpu, value: cpu?.name || (typeof hardware.cpu === 'string' ? hardware.cpu : '未知处理器'), note: `${numberText(cpu?.cores)} 核 · ${numberText(cpu?.threads ?? hardware.cpu_count)} 线程`, extra: cpu?.l3_bytes ? `L3 ${formatBytes(cpu.l3_bytes)}` : null },
    { id: 'memory', label: '系统内存', icon: MemoryStick, value: capacity(hardware.memory_used, total), note: '已用 / 总量', extra: [memoryType, speed && `${speed} MT/s`].filter(Boolean).join(' · ') },
    { id: 'gpu', label: 'GPU', icon: CircuitBoard, value: typeof gpu?.name === 'string' ? gpu.name : '未取得 GPU 信息', note: gpu?.compute_capability ? `计算能力 ${gpu.compute_capability}` : '计算能力不可用', extra: finite(gpu?.cuda_cores) === null ? null : `${numberText(gpu?.cuda_cores, 0)} 个 CUDA 核心` },
    { id: 'vram', label: '显存', icon: HardDrive, value: capacity(gpu?.memory_used, gpu?.memory_total), note: '已用 / 总量', extra: finite(gpu?.memory_bus_bits) === null ? null : `${gpu?.memory_bus_bits} bit` },
  ]
  const close = () => { const id = selected; setSelected(null); document.getElementById(rootId)?.querySelector<HTMLButtonElement>(`[data-resource="${id}"]`)?.focus() }
  return <section id={rootId} className="hardware-resources"><div className="section-heading"><h3>本机资源</h3>{gpus.length > 1 && <Select aria-label="资源 GPU" value={gpuKey(gpu)} onChange={event => { setDevice(event.target.value); setSelected(null) }}>{gpus.map(g => <option key={gpuKey(g)} value={gpuKey(g)}>{gpuLabel(g)}</option>)}</Select>}</div>
    <div className="hardware-card-grid">{tiles.map(tile => <ResourceTile key={tile.id} {...tile} selected={selected === tile.id} onClick={() => setSelected(selected === tile.id ? null : tile.id)} />)}</div>
    <div className="hardware-spec-grid">{[['PCIe 链路上限', pcieLink(gpu, true)], ['PCIe 理论带宽 · 单向', finite(gpu?.pcie_bandwidth_max_bytes_s) === null ? '不可用' : `${numberText(Number(gpu?.pcie_bandwidth_max_bytes_s) / 1e9, 1)} GB/s`], ['显存位宽', finite(gpu?.memory_bus_bits) === null ? '不可用' : `${gpu?.memory_bus_bits} bit`], ['显存理论带宽', finite(gpu?.memory_bandwidth_bytes_s) === null ? '不可用' : `${numberText(Number(gpu?.memory_bandwidth_bytes_s) / 1e9, 1)} GB/s`]].map(([label, value]) => <div className="panel hardware-spec" key={label}><span>{label}</span><strong>{value}</strong></div>)}</div>
    {selected && <ResourceDetail title={details[selected].label} note={details[selected].note} onClose={close}><Facts items={details[selected].facts} /></ResourceDetail>}
    {hardware.specs_error != null && <DiagnosticNotice summary="部分硬件规格不可用" detail={hardware.specs_error} />}
  </section>
}
