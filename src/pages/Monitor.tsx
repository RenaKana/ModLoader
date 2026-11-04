import { useEffect, useRef, useState } from 'react'
import { ArrowRight, BarChart3, Cpu, ExternalLink, Gauge, Loader2, RefreshCw, Square, Terminal, PanelsTopLeft } from 'lucide-react'
import type { PageProps, TelemetryMetric } from '../types'
import { apiFetch, errorMessage, isActive, openExternal } from '../api'
import { Badge, Button, DiagnosticNotice, Empty, ErrorNotice, IconButton, PageHeader, StateBadge, formatBytes } from '../components'
import { Select } from '../components'
import { ResourceDetail, TrendChart, HardwareTrendChart, Facts, NumericReadout } from '../ResourceCards'
import { MonitorRequestHistory, type MonitorRequestHistoryHandle } from '../MonitorRequestHistory'
import { finite, gpuKey, gpuLabel, numberText, pcieLink, rateText, useHistory, type HistoryPoint } from '../resource-metrics'
import type { MonitorState } from '../useMonitorState'
import ModelLoadingStatus from '../ModelLoadingStatus'

type MonitorMetric = { id: string; label: string; value: number | null; unit: string; note: string; max: number | null; adaptive?: boolean }

export default function Monitor({ monitor, active = true, ...props }: PageProps & { monitor: MonitorState; active?: boolean }) {
  const { instances, instance, instanceId, setInstanceId, setDevice, gpus, gpu, payload, live,
    sampleError, telemetryScope, refreshing, loading, refreshMonitor } = monitor
  const root = useRef<HTMLDivElement>(null)
  const requests = useRef<MonitorRequestHistoryHandle>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [moreOpen, setMoreOpen] = useState(false)
  const backendLabel = props.state.backends.find(backend => backend.id === instance?.backend)?.label || instance?.backend || ''
  const modelOptions = instances.map(item => ({ value: item.id, label: `${item.alias} · ${props.state.backends.find(backend => backend.id === item.backend)?.label || item.backend}` }))
  if (instance && !isActive(instance.state)) modelOptions.push({ value: instance.id, label: `${instance.alias} · 已停止` })
  const modelLabel = instance ? `${instance.alias} · ${backendLabel}` : ''
  useEffect(() => { setSelected(null) }, [telemetryScope, monitor.device])
  const hardware = props.state.hardware
  const cpuTemperatureStatus = hardware.cpu_temperature_status || 'disabled'
  const gpuTemperatureInitializationReason = temperatureSourceLabel(hardware.gpu_temperature_reason)
  const [enablingCpuTemperature, setEnablingCpuTemperature] = useState(false)
  const [temperatureReaderError, setTemperatureReaderError] = useState('')
  async function retryTemperatureReader() {
    if (enablingCpuTemperature || (cpuTemperatureStatus === 'ready' && !gpuTemperatureInitializationReason && finite(hardware.cpu_power_w) !== null)) return
    setEnablingCpuTemperature(true)
    setTemperatureReaderError('')
    try {
      await apiFetch('/api/hardware/temperatures/enable', { method: 'POST' })
    } catch (issue) {
      setTemperatureReaderError(errorMessage(issue))
    } finally {
      await props.refresh()
      setEnablingCpuTemperature(false)
    }
  }
  const scale = (value: unknown, divisor: number) => finite(value) === null ? null : Number(value) / divisor
  const cpuName = hardware.cpu_info?.name?.trim() || 'CPU 型号不可用'
  const gpuName = typeof gpu?.name === 'string' && gpu.name.trim() ? gpu.name.trim() : 'GPU 型号不可用'
  const cpuTemperatureSource = temperatureSourceLabel(hardware.cpu_temperature_source)
  const cpuTemperatureReason = temperatureSourceLabel(hardware.cpu_temperature_reason)
  const cpuTemperatureNote = cpuTemperatureStatus === 'starting' ? '正在连接或授权温度读取' : cpuTemperatureSource || cpuTemperatureReason || 'CPU 温度传感器未提供数据'
  const gpuTemperatureSource = temperatureSourceLabel(gpu?.temperature_source)
  const gpuTemperatureReason = temperatureSourceLabel(gpu?.temperature_reason) || gpuTemperatureInitializationReason
  const cpuPowerSource = temperatureSourceLabel(hardware.cpu_power_source)
  const cpuPowerReason = temperatureSourceLabel(hardware.cpu_power_reason)
  const gpuPowerSource = temperatureSourceLabel(gpu?.power_source)
  const gpuPowerReason = temperatureSourceLabel(gpu?.power_reason)
  const systemMetrics: MonitorMetric[] = [
    { id: 'cpu', label: 'CPU 使用率', value: finite(hardware.cpu_percent), unit: '%', note: `${numberText(hardware.cpu_count)} 个逻辑处理器`, max: 100 },
    { id: 'gpu', label: 'GPU 使用率', value: finite(gpu?.utilization), unit: '%', note: gpuName, max: 100 },
    { id: 'cpu_power', label: 'CPU 功率', value: finite(hardware.cpu_power_w), unit: 'W', note: 'CPU 封装', max: null, adaptive: true },
    { id: 'gpu_power', label: 'GPU 功率', value: finite(gpu?.power), unit: 'W', note: 'GPU 设备', max: null, adaptive: true },
    { id: 'cpu_temperature', label: 'CPU 温度', value: finite(hardware.cpu_temperature_c), unit: '°C', note: cpuTemperatureSource || cpuTemperatureNote, max: null, adaptive: true },
    { id: 'gpu_temperature', label: 'GPU 温度', value: finite(gpu?.temperature), unit: '°C', note: gpuTemperatureSource || gpuTemperatureReason || gpuName, max: null, adaptive: true },
    { id: 'disk', label: '磁盘读取', value: scale(hardware.disk_read_bytes_per_second, 1e6), unit: 'MB/s', note: '系统整体读取', max: null },
  ]
  const pcieMetrics: MonitorMetric[] = [
    { id: 'pcie_rx', label: 'PCIe RX', value: scale(gpu?.pcie_rx_bytes_s, 1e9), unit: 'GB/s', note: '流入 GPU', max: null },
    { id: 'pcie_tx', label: 'PCIe TX', value: scale(gpu?.pcie_tx_bytes_s, 1e9), unit: 'GB/s', note: '流出 GPU', max: null },
  ]
  const hardwareTime = (hardware.sampled_at ?? 0) * 1000
  const coreValues = Object.fromEntries((hardware.cpu_per_core ?? []).map((v, i) => [`core${i}`, finite(v)]))
  const hardwareHistory = useHistory(gpuKey(gpu), hardwareTime ? { time: hardwareTime, values: { ...Object.fromEntries([...systemMetrics, ...pcieMetrics].map(m => [m.id, m.value])), ...coreValues } } : null)
  const updatedAt = payload?.updated_at ? new Date(typeof payload.updated_at === 'number' ? payload.updated_at * 1000 : payload.updated_at) : null
  const updated = updatedAt && !Number.isNaN(updatedAt.getTime()) ? updatedAt.toLocaleTimeString('zh-CN', { hour12: false }) : null
  const recent = payload?.groups.find(g => g.id === 'api_request')
  const backendGroups = payload?.groups.filter(g => !['api_request', 'model_generation'].includes(g.id)) ?? []
  const hardwareMetrics = [...systemMetrics, ...pcieMetrics]
  const hardwareGroups = [
    { id: 'gpu_group', label: 'GPU', note: gpuName, metrics: ['gpu', 'gpu_power', 'gpu_temperature'] },
    { id: 'cpu_group', label: 'CPU', note: cpuName, metrics: ['cpu', 'cpu_power', 'cpu_temperature'] },
    { id: 'io_group', label: 'I/O', note: '磁盘与 PCIe 传输', metrics: ['disk', 'pcie_rx', 'pcie_tx'] },
  ]
  const deviceButton = (id: string) => {
    const group = hardwareGroups.find(item => item.id === id)!
    return <button type="button" className="monitor-hardware-device" data-resource={id} aria-label={`${group.label}整体趋势`} aria-expanded={selected === id} onClick={() => setSelected(selected === id ? null : id)}><strong>{group.label}</strong><span>{group.note}</span></button>
  }
  const metricButton = (id: string, label: string) => {
    const metric = hardwareMetrics.find(item => item.id === id)!
    return <button type="button" key={id} className="monitor-hardware-metric" data-resource={id} aria-label={`${metric.label}详情`} aria-expanded={selected === id} onClick={() => setSelected(selected === id ? null : id)}><span className="monitor-hardware-metric-label">{label}</span><span className="monitor-hardware-metric-value"><NumericReadout value={metric.unit.endsWith('B/s') ? rateText(metric.value) : numberText(metric.value)} unit={metric.value !== null ? metric.unit : undefined} /></span></button>
  }
  const chart = (metric: typeof systemMetrics[number], points: HistoryPoint[]) => <TrendChart key={metric.id} points={points} metric={metric.id} label={metric.label} unit={metric.unit} maximum={metric.max} adaptive={metric.adaptive} expanded />
  const closeDetail = () => { const previous = selected; setSelected(null); root.current?.querySelector<HTMLButtonElement>(`[data-resource="${previous}"]`)?.focus() }
  function detail(points: HistoryPoint[]) {
    const group = hardwareGroups.find(item => item.id === selected)
    if (group) return <ResourceDetail key={`${gpuKey(gpu)}:${group.id}`} title={`${group.label}整体趋势`} note={group.note} onClose={closeDetail}>
      <HardwareTrendChart points={points} metrics={group.metrics.map(id => hardwareMetrics.find(metric => metric.id === id)!)} label={`${group.label}整体趋势`} />
    </ResourceDetail>
    const metric = hardwareMetrics.find(m => m.id === selected)
    if (!metric) return null
    const cpuSensor = metric.id === 'cpu_power' || metric.id === 'cpu_temperature'
    const gpuSensor = metric.id === 'gpu_power' || metric.id === 'gpu_temperature'
    return <ResourceDetail key={`${gpuKey(gpu)}:${metric.id}`} title={metric.label} note={metric.note} onClose={closeDetail}>
      {chart(metric, points)}
      {(metric.id === 'pcie_rx' || metric.id === 'pcie_tx') && <Facts items={[["当前链路", pcieLink(gpu)], ['链路上限', pcieLink(gpu, true)]]} />}
      {(metric.id === 'cpu_power' || metric.id === 'gpu_power') && <div className="monitor-sensor-facts"><Facts items={cpuSensor
        ? [['采集来源', cpuPowerSource || cpuPowerReason || 'CPU 未提供封装功率'], ['CPU 温度', `${numberText(hardware.cpu_temperature_c)} °C`], ['温度来源', cpuTemperatureSource || cpuTemperatureNote]]
        : [['采集来源', gpuPowerSource || gpuPowerReason || 'GPU 未提供功率读数'], ['GPU 温度', `${numberText(gpu?.temperature)} °C`], ['温度来源', gpuTemperatureSource || gpuTemperatureReason || gpuName]]} /></div>}
      {metric.id === 'gpu_power' && gpuTemperatureReason && <p className="monitor-gpu-temperature-reason">{gpuTemperatureReason}</p>}
      {gpuSensor && gpuTemperatureInitializationReason && <div className="monitor-temperature-control">
        {cpuTemperatureStatus === 'starting' || enablingCpuTemperature
          ? <div className="monitor-temperature-pending" role="status"><Loader2 size={13} className="spin" /><span>正在连接温度读取</span></div>
          : <div className="monitor-temperature-actions"><Button small icon={RefreshCw} busy={enablingCpuTemperature} onClick={() => void retryTemperatureReader()}>重试温度读取</Button></div>}
        {temperatureReaderError && <p className="monitor-temperature-error" role="alert">{temperatureReaderError}</p>}
      </div>}
      {cpuSensor && (cpuTemperatureStatus !== 'ready' || (metric.id === 'cpu_power' && finite(hardware.cpu_power_w) === null)) && <div className="monitor-temperature-control">
        {enablingCpuTemperature || cpuTemperatureStatus === 'starting'
          ? <div className="monitor-temperature-pending" role="status"><Loader2 size={13} className="spin" /><span>{finite(hardware.cpu_power_w) === null ? '正在连接或授权功率读取' : '正在连接或授权温度读取'}</span><small>仅授权独立读取进程，本次运行有效；取消后不会自动重试。</small></div>
          : <div className="monitor-temperature-actions"><Button small icon={RefreshCw} busy={enablingCpuTemperature} onClick={() => void retryTemperatureReader()}>{finite(hardware.cpu_power_w) === null ? '重试功率读取' : '重试温度读取'}</Button>
            {cpuTemperatureStatus === 'driver_missing' && <Button small variant="ghost" icon={ExternalLink} onClick={() => void openExternal('https://github.com/namazso/PawnIO.Setup/releases/tag/2.2.0').catch(issue => setTemperatureReaderError(errorMessage(issue)))}>官方 PawnIO 驱动</Button>}
            <span>仅授权独立读取进程，本次运行有效；取消后不会自动重试。</span></div>}
        {(cpuTemperatureStatus === 'error' || temperatureReaderError) && <p className="monitor-temperature-error" role="alert">{temperatureReaderError || cpuTemperatureReason || 'CPU 温度读取失败'}</p>}
      </div>}
      {metric.id === 'cpu' && <><div className="section-heading"><h3>逻辑处理器</h3><span>{hardware.cpu_per_core?.length ?? 0} 个</span></div><div className="monitor-cores">{(hardware.cpu_per_core ?? []).map((v, i) => <div className="monitor-core" key={i}><div><span>CPU {i}</span><span>{numberText(v, 1)}%</span></div><TrendChart points={points} metric={`core${i}`} label={`CPU ${i}`} unit="%" maximum={100} /></div>)}</div></>}
    </ResourceDetail>
  }

  return <div className="monitor-page" ref={root}>
    <PageHeader eyebrow="RUNTIME MONITOR" title="运行监控">{window.modloader?.openMiniMonitor && <Button icon={PanelsTopLeft} onClick={() => void window.modloader!.openMiniMonitor().catch(issue => props.notify(errorMessage(issue), 'error'))}>小窗监控</Button>}<IconButton icon={RefreshCw} label="刷新到最新" disabled={refreshing} onClick={() => { requests.current?.refresh(); void refreshMonitor() }} /></PageHeader>
    <div className="monitor-selector" aria-label="实时运行实例"><div className="monitor-selector-main">
      <span className="monitor-instance-label">运行实例</span>
      {modelOptions.length > 1 ? <div className="model-instance-control"><Select aria-label="选择监控实例" value={instanceId} onChange={event => setInstanceId(event.target.value)}>{modelOptions.map(item => <option key={item.value} value={item.value}>{item.label}</option>)}</Select></div>
        : instance ? <span className="model-instance-control model-instance-name" data-tooltip={modelLabel}>{modelLabel}</span>
          : <><span className="model-instance-empty">暂无运行模型</span><Button variant="ghost" onClick={() => props.navigate('models')}>前往模型库</Button></>}
      {instance && <StateBadge state={instance.state} />}
    </div><div className="monitor-selector-end">{instance && <Button variant="ghost" icon={Terminal} onClick={() => props.openLogs(instance)}>日志</Button>}{instance && isActive(instance.state) && <Button variant="ghost" className="stop-instance" icon={Square} disabled={instance.state === 'stopping'} onClick={() => void props.stopInstance(instance)}>停止</Button>}</div></div>
    <ModelLoadingStatus instance={instance} />
    {instance?.state === 'failed' && instance.error && <DiagnosticNotice summary="加载失败" detail={instance.error} />}
    <MonitorRequestHistory monitor={monitor} active={active} ref={requests} hardwareControls={<><span>系统整体</span>{gpus.length > 1 && <Select aria-label="监控 GPU" value={gpuKey(gpu)} onChange={event => { setDevice(event.target.value); setSelected(null) }}>{gpus.map(g => <option key={gpuKey(g)} value={gpuKey(g)}>{gpuLabel(g)}</option>)}</Select>}<span>{pcieLink(gpu)}</span></>} hardware={<>
    <section className="panel monitor-hardware-summary" aria-label="硬件读数">
      <div className="monitor-hardware-row">{deviceButton('gpu_group')}{metricButton('gpu', '使用率')}{metricButton('gpu_power', '功率')}{metricButton('gpu_temperature', '温度')}</div>
      <div className="monitor-hardware-row">{deviceButton('cpu_group')}{metricButton('cpu', '使用率')}{metricButton('cpu_power', '功率')}{metricButton('cpu_temperature', '温度')}</div>
      <div className="monitor-hardware-row monitor-hardware-io">{deviceButton('io_group')}{metricButton('disk', '磁盘读取')}{metricButton('pcie_rx', 'PCIe RX')}{metricButton('pcie_tx', 'PCIe TX')}</div>
    </section>{detail(hardwareHistory)}
    {hardware.gpu_error != null && <DiagnosticNotice summary="GPU 采样失败" detail={hardware.gpu_error} />}
    {hardware.error != null && <DiagnosticNotice summary="硬件采样失败" detail={hardware.error} />}
    </>} />
    <div className="monitor-runtime-heading"><div><h3>{instance ? `${backendLabel} 运行指标` : '后端运行指标'}</h3></div><span>{loading && <Loader2 size={12} className="spin" />}{updated ? `最近采样 ${updated}` : instance ? '等待后端采样' : '尚未选择实例'}</span></div>
    {sampleError && <ErrorNotice message={sampleError} retry={() => void refreshMonitor()} />}
    {!instance ? <div className="panel"><Empty icon={BarChart3} title="加载一个模型以开始监控"><Button small onClick={() => props.navigate('models')}>前往模型库 <ArrowRight size={13} /></Button></Empty></div>
      : backendGroups.length || recent ? <details className="monitor-backend-details" open={moreOpen} onToggle={event => setMoreOpen(event.currentTarget.open)}><summary>更多后端指标</summary><div className="telemetry-groups">{recent && <div className="panel telemetry-group request-group monitor-last-request"><div className="panel-heading"><h3>最近完成请求</h3></div><RequestMetrics metrics={recent.metrics} /></div>}{backendGroups.map(group => <div className="panel telemetry-group" key={group.id}><div className="panel-heading"><h3><Gauge size={16} />{group.title}</h3></div><div className="telemetry-metric-grid">{group.metrics.map(metric => <MetricCard embedded key={metric.id} metric={metric} />)}</div></div>)}</div></details>
        : <div className="panel"><div className="inline-empty monitor-wait">{loading ? <Loader2 size={18} className="spin" /> : <Cpu size={18} />}<span>{loading ? '正在读取后端监控…' : '后端尚未返回可用指标'}</span></div></div>}
    {(!!payload?.unavailable?.length || live?.reason) && <details className="monitor-unavailable"><summary>采样诊断</summary>{live?.reason && <p>{live.reason}</p>}{payload?.unavailable.map(reason => <p key={reason}>{reason}</p>)}</details>}
  </div>
}

function MetricCard({ metric, embedded = false }: { metric: TelemetryMetric; embedded?: boolean }) {
  const unavailable = metric.value === null || metric.value === undefined || (typeof metric.value === 'string' && ['unknown', 'unavailable', 'n/a'].includes(metric.value.toLowerCase()))
  const value = unavailable ? '不可用' : metric.kind === 'timestamp' && typeof metric.value === 'number' ? new Date(metric.value * 1000).toLocaleString('zh-CN', { hour12: false }) : typeof metric.value === 'number' && metric.unit === 'bytes' ? formatBytes(metric.value) : typeof metric.value === 'number' ? numberText(metric.value, 2, metric.unit === 'token/s') : metric.value
  const parts = typeof value === 'string' && /^(.*) (B|KB|MB|GB|TB|PB)$/.exec(value)
  return <div className={`${embedded ? 'embedded-metric' : 'panel monitor-metric'} ${metric.kind === 'text' || metric.kind === 'timestamp' ? 'text-metric' : 'numeric-metric'}`}>
    <div className="monitor-metric-label"><span>{metric.label}</span>{metric.kind === 'counter' && <Badge>累计</Badge>}</div><div className={`monitor-metric-value ${unavailable ? 'unavailable' : ''}`}><strong>{parts ? parts[1] : value}</strong>{!unavailable && (parts || metric.unit) && <small>{[parts ? parts[2] : '', parts && metric.unit === 'bytes' ? '' : metric.unit].filter(Boolean).join(' ')}</small>}</div>
  </div>
}

function temperatureSourceLabel(source: unknown) {
  return typeof source === 'string' ? source.replace(/^LibreHardwareMonitor\s*·\s*/, '').trim() : ''
}

function RequestMetrics({ metrics }: { metrics: TelemetryMetric[] }) {
  const priority = ['first_output_s', 'output_speed', 'duration_s']
  const primary = priority.flatMap(id => metrics.filter(metric => metric.id === id))
  const secondary = metrics.filter(metric => !priority.includes(metric.id))
  return <><div className="request-primary" style={{ gridTemplateColumns: `repeat(${primary.length || 1}, minmax(0, 1fr))` }}>{primary.map(metric => <MetricCard embedded key={metric.id} metric={metric} />)}</div>
    <div className="request-secondary">{secondary.map(metric => <MetricCard embedded key={metric.id} metric={metric} />)}</div></>
}
