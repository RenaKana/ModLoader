import { useEffect, useState } from 'react'
import { requestProgress, errorMessage } from './api'
import OperationStatus, { type OperationProgress } from './OperationStatus'
import { DiagnosticNotice, formatBytes } from './components'
import type { Hardware, LoadProfile } from './types'
import { projectStrataMemory, type StrataResidentPlan } from './strata-memory-preview'
import './resource-preview.css'
import { gpuLabel, gpuList } from './resource-metrics'

export interface ResourceEstimate {
  kind: 'ram' | 'vram'
  device?: string
  logical_device?: string
  gpu_id?: string
  pci_bus_id?: string
  identity_status?: 'resolved' | 'unknown'
  aggregate?: boolean
  label: string
  bytes: number | null
  quality: 'estimate' | 'partial' | 'dynamic' | 'unknown'
  reserve_bytes?: number
  fixed_bytes?: number
  budget_bytes?: number
  budget_component?: string
  budget_fraction?: number
  budget_basis?: 'total' | 'available'
  logical?: boolean
  resident_plan?: StrataResidentPlan
  components?: { label: string; bytes: number | null; reference?: boolean }[]
  notes: string[]
}
export interface ResourceForecast { resources: ResourceEstimate[]; notes: string[]; source: string; diagnostic?: string }
type Draft = Pick<LoadProfile, 'model_id' | 'runtime_id' | 'backend' | 'parameters' | 'extra_args'>
const numeric = (value: unknown) => typeof value === 'number' && Number.isFinite(value) ? value : null
const percent = (value: number) => Math.max(0, Math.min(100, value * 100))
const currentResources: ResourceEstimate[] = [
  { kind: 'vram', label: 'GPU 显存', bytes: null, quality: 'unknown', notes: [] },
  { kind: 'ram', label: '系统内存', bytes: null, quality: 'unknown', notes: [] },
]

function isUnused(resource: ResourceEstimate) {
  const logical = resource.logical ?? (resource.quality === 'partial' && resource.kind === 'ram')
  // Logical/mapped size is a reference, not a pending physical RAM estimate.
  if (logical) return true
  if (resource.bytes !== 0 || resource.quality === 'unknown' || resource.quality === 'dynamic') return false
  // A zero lower bound is not zero usage.
  if (resource.quality === 'partial') return false
  return !resource.components?.some(part => !part.reference && part.bytes !== 0)
}

function resourceCapacity(resource: ResourceEstimate, hardware: Hardware): { total: number | null; used: number | null; gpu?: Record<string, unknown> } {
  if (hardware.error) return { total: null, used: null }
  if (resource.kind === 'ram') {
    const total = numeric(hardware.memory_total ?? hardware.ram_total_bytes)
    const available = numeric(hardware.memory_available ?? hardware.ram_available_bytes)
    return { total, used: numeric(hardware.memory_used) ?? (total !== null && available !== null ? total - available : null) }
  }
  if (resource.aggregate || resource.identity_status === 'unknown') return { total: null, used: null }
  const devices = gpuList(hardware)
  const pci = (value: unknown) => typeof value === 'string' ? value.toLowerCase().replace(/^0+(?=[0-9a-f]+:)/, '') : ''
  const matches = resource.gpu_id
    ? devices.filter(item => String(item.gpu_id ?? item.id).toLowerCase() === resource.gpu_id!.toLowerCase())
    : resource.pci_bus_id ? devices.filter(item => pci(item.pci_bus_id) === pci(resource.pci_bus_id))
    : resource.identity_status ? [] : devices.filter(item => item.device === resource.device && !!resource.device)
  // Legacy single-device payloads predate explicit identities. Never infer a
  // second card, or substitute another card when an explicit identity is missing.
  const gpu = matches.length === 1 ? matches[0]
    : !resource.identity_status && !resource.gpu_id && !resource.pci_bus_id && devices.length === 1
      && (!resource.device || resource.device === 'CUDA0') ? devices[0] : undefined
  return { total: numeric(gpu?.memory_total), used: numeric(gpu?.memory_used), gpu }
}

export default function ResourcePreview({ draft, hardware, active = true, alreadyLoaded = false, compact = false }: {
  draft: Draft; hardware: Hardware; active?: boolean; alreadyLoaded?: boolean; compact?: boolean
}) {
  const key = JSON.stringify(draft)
  const scope = JSON.stringify([draft.model_id, draft.runtime_id, draft.backend])
  const [result, setResult] = useState<{ key: string; scope: string; value?: ResourceForecast; error?: string } | null>(null)
  const [retry, setRetry] = useState(0)
  const [operation, setOperation] = useState<{ key: string; startedAt: number; progress?: OperationProgress }>({ key, startedAt: Date.now() })
  useEffect(() => {
    if (!active || !draft.runtime_id) return
    const abort = new AbortController()
    setOperation({ key, startedAt: Date.now() })
    const timer = window.setTimeout(async () => {
      try {
        const value = await requestProgress<ResourceForecast>('POST', '/api/resources/estimate', JSON.parse(key), {
          signal: abort.signal, onProgress: progress => {
            if (!abort.signal.aborted) setOperation(previous => ({ ...previous, progress }))
          },
        })
        if (!abort.signal.aborted) setResult({ key, scope, value })
      } catch (error) {
        if (!abort.signal.aborted) setResult(previous => ({ key, scope, error: errorMessage(error), value: previous?.scope === scope ? previous.value : undefined }))
      }
    }, draft.backend === 'strata' ? 150 : 450)
    return () => { window.clearTimeout(timer); abort.abort() }
  }, [key, scope, active, retry, draft.runtime_id])
  // Keep the last estimate mounted while editing this model/runtime. Collapsing
  // it into a loading line moves the numeric fields away from a stationary mouse.
  const available = result?.scope === scope ? result : null
  const current = available?.key === key ? available : null
  const value = available?.value
  const stale = Boolean(value && (!current || current.error))
  function retryEstimate() {
    setResult(previous => previous ? { ...previous, key: '', error: undefined } : null)
    setRetry(previous => previous + 1)
  }
  const notes = [...new Set([...(value?.notes || []), ...(value?.resources.flatMap(item => item.notes) || [])])]
  if (!draft.runtime_id) return null
  const unavailable = value && !value.resources.length
  const resources = value?.resources.length ? value.resources : currentResources
  const forecastStatus = value ? 'complete' : current?.error ? 'error' : current ? 'complete' : 'pending'
  let ramLimited = false
  let ramShortfall = 0
  const projected = resources.map(resource => {
    const plan = resource.resident_plan
    if (!plan) return resource
    const gpu = resources.find(item => item.kind === 'vram' && (item.device ?? 'CUDA0') === plan.gpu_device)
    const gpuCapacity = resourceCapacity(gpu ?? { ...resource, kind: 'vram', device: plan.gpu_device }, hardware)
    const ramCapacity = resourceCapacity(resource, hardware)
    const estimate = projectStrataMemory(plan, {
      ram_available: ramCapacity.total !== null && ramCapacity.used !== null ? ramCapacity.total - ramCapacity.used : null,
      commit_available: hardware.error ? null : numeric(hardware.memory_commit_available),
      gpu_available: gpuCapacity.total !== null && gpuCapacity.used !== null ? gpuCapacity.total - gpuCapacity.used : null,
      gpu_fixed: numeric(gpu?.fixed_bytes ?? gpu?.bytes), gpu_reserve: numeric(gpu?.reserve_bytes),
    })
    ramLimited ||= estimate.limited
    ramShortfall = Math.max(ramShortfall, estimate.shortfall ?? 0)
    return { ...resource, bytes: estimate.bytes, quality: estimate.quality, logical: false }
  })
  // A dynamic budget is not an allocation estimate; it needs no list of omitted allocations.
  const excluded = [notes.some(note => /未计入 MTP/.test(note)) && 'MTP', notes.some(note => /未计入视觉/.test(note)) && '图像输入'].filter(Boolean)
  const optionWarnings = notes.filter(note => note.startsWith('未计入高级选项'))
  const flashDisabled = notes.some(note => /Flash Attention.*已自动关闭/.test(note))
  const visibleResources = projected.filter(resource => !isUnused(resource))
  const resourceOrder = (resource: ResourceEstimate) => resource.kind === 'ram' ? 2 : resource.aggregate ? 1 : 0
  if (current && !stale && !current.error && !visibleResources.length && !unavailable && !excluded.length
    && !optionWarnings.length && !flashDisabled && !value?.diagnostic && !ramLimited && !ramShortfall) return null
  return <section className={`resource-preview${compact ? ' compact' : ''}`} aria-label="加载资源预估" aria-busy={!current} data-stale={stale || undefined}>
    <div className="resource-preview-heading"><strong>加载预估</strong><span className="resource-preview-status">
      {stale ? current?.error ? '上次估算'
        : <><OperationStatus progress={operation.key === key ? operation.progress : undefined} startedAt={operation.startedAt} fallback="更新估算" /><span>· 上次估算</span></>
        : !current ? <OperationStatus progress={operation.key === key ? operation.progress : undefined} startedAt={operation.startedAt} fallback="估算中" /> : ''}
    </span></div>
    {alreadyLoaded && <p className="resource-preview-caveat">按再加载一份计算</p>}
    {current?.error && <DiagnosticNotice summary={value ? '估算更新失败，显示上次结果' : '资源估算失败'} retry={retryEstimate} />}
    {!!visibleResources.length && <div className="resource-preview-grid">{[...visibleResources].sort((a, b) => resourceOrder(a) - resourceOrder(b)).map((resource, index) => <ResourceRow key={`${resource.kind}-${resource.gpu_id || resource.device || index}`} resource={resource} hardware={hardware} showDevice={visibleResources.filter(item => item.kind === 'vram').length > 1} status={forecastStatus} />)}</div>}
    {ramLimited && <p className="resource-preview-caveat">RAM 驻留未覆盖全部专家</p>}
    {ramShortfall > 0 && <p className="resource-preview-caveat">驻留预算不足 {formatBytes(ramShortfall)}</p>}
    {unavailable && (value.diagnostic ? <DiagnosticNotice summary={value.notes[0] || '当前配置无法估算'} /> : <p className="resource-preview-caveat">{value.notes[0] || '当前配置无法估算'}</p>)}
    {!unavailable && !!excluded.length && <p className="resource-preview-caveat">未计入：{excluded.join('、')}额外占用</p>}
    {!unavailable && optionWarnings.map(note => <p className="resource-preview-caveat" key={note}>{note}</p>)}
    {!unavailable && flashDisabled && <p className="resource-preview-caveat">Flash Attention 未通过设备检查，估算按关闭计算</p>}
    {!unavailable && value?.diagnostic && <DiagnosticNotice summary="部分估算数据不可用" />}
  </section>
}

function ResourceRow({ resource, hardware, showDevice, status }: { resource: ResourceEstimate; hardware: Hardware; showDevice: boolean; status: 'pending' | 'complete' | 'error' }) {
  const references = resource.components?.filter(part => part.bytes !== null) ?? []
  if (resource.aggregate) return <div className="resource-preview-row aggregate" data-resource={resource.kind}>
    <div className="resource-preview-label">多 GPU 合计参考</div>
    {references.length ? <dl className="resource-preview-references">{references.map((part, index) => <div key={index}><dt>{part.label}</dt><dd>{formatBytes(part.bytes!)}</dd></div>)}</dl> : <p className="resource-preview-caveat">分项待估算</p>}
  </div>
  const { total, used, gpu } = resourceCapacity(resource, hardware)
  const measured = total !== null && total > 0 && used !== null
  const addition = numeric(resource.bytes)
  // A component target does not establish total RAM or post-load free RAM.
  const componentTarget = resource.budget_component
  const dynamic = resource.quality === 'dynamic' && !componentTarget
  const partial = resource.quality === 'partial'
  // llama.cpp Host buffers include mapped weights. Never add them to physical RAM usage.
  const logical = resource.logical ?? (partial && resource.kind === 'ram')
  const fixed = dynamic ? numeric(resource.fixed_bytes) : !logical && resource.quality !== 'unknown' ? addition : null
  let budget = dynamic ? numeric(resource.budget_bytes) : null
  if (dynamic && budget === null && measured) {
    if (resource.budget_fraction !== undefined) budget = (resource.budget_basis === 'available' ? Math.max(0, total! - used!) : total!) * resource.budget_fraction
    else if (resource.reserve_bytes !== undefined) budget = Math.max(0, total! - used! - resource.reserve_bytes)
  }
  // A reserve target cannot hide fixed allocations that already exceed it.
  const showBudget = budget !== null && (fixed === null || fixed <= budget)
  const amount = showBudget ? budget : fixed
  const remaining = measured && amount !== null ? total! - used! - amount : null
  const device = resource.logical_device ?? resource.device
  const title = `${resource.label === resource.device || resource.label === 'GPU 显存预算' ? 'GPU 显存' : resource.label === '内存缓冲参考' ? '系统内存' : resource.label}${device && (showDevice || device !== 'CUDA0') ? ` · ${device}` : ''}`
  const exceeded = remaining !== null && remaining < 0
  const lowerBound = (partial || dynamic) && fixed !== null
  const estimateLabel = showBudget ? '分配预算' : lowerBound ? '至少占用' : '预计占用'
  const unknownValue = status === 'pending' ? '待估算' : status === 'error' ? '估算失败' : '无法估算'
  const estimateValue = amount !== null ? formatBytes(amount) : unknownValue
  const remainderLabel = showBudget ? exceeded ? '预算超出' : '预算后剩余'
    : exceeded ? lowerBound ? '至少超出' : '预计超出' : lowerBound && remaining !== null ? '最多剩余' : '加载后剩余'
  const remainderValue = remaining !== null ? formatBytes(Math.abs(remaining)) : measured ? unknownValue : '容量未读取'
  const usedWidth = measured ? percent(used! / total!) : 0
  const fixedWidth = measured && fixed !== null ? Math.min(100 - usedWidth, percent(fixed / total!)) : 0
  const budgetWidth = measured && showBudget ? Math.min(100 - usedWidth - fixedWidth, percent((budget! - (fixed ?? 0)) / total!)) : 0
  const fixedLabel = dynamic ? `已估固定占用 ${formatBytes(fixed ?? 0)}` : `${estimateLabel} ${estimateValue}`
  return <div className={`resource-preview-row${exceeded ? ' exceeded' : ''}`} data-resource={resource.kind} data-device={device} data-gpu-id={resource.gpu_id} data-quality={resource.quality}>
    <div className="resource-preview-label">{title}</div>
    {showDevice && gpu?.name != null && <div className="resource-preview-device">{gpuLabel(gpu)}</div>}
    <div className="resource-forecast-bar" role={measured ? 'meter' : undefined}
      aria-label={`${title}${dynamic ? '分配预览' : fixed !== null ? '加载预估' : '当前占用'}`}
      aria-valuemin={measured ? 0 : undefined} aria-valuemax={measured ? 100 : undefined}
      aria-valuenow={measured ? percent((used! + (fixed ?? 0)) / total!) : undefined}
      aria-valuetext={measured ? `当前已用 ${formatBytes(used!)}，${estimateLabel} ${estimateValue}，${remainderLabel} ${remainderValue}${dynamic && fixed !== null ? `，${fixedLabel}` : ''}` : undefined}
      aria-hidden={!measured || undefined}>
      <span className="forecast-used" style={{ width: `${usedWidth}%` }} title={measured ? `当前已用 ${formatBytes(used!)}` : undefined} />
      {fixed !== null && <span className="forecast-addition" style={{ width: `${fixedWidth}%` }} title={fixedLabel} />}
      {showBudget && <span className="forecast-addition budget" style={{ width: `${budgetWidth}%` }} title={`分配预算 ${estimateValue}`} />}
    </div>
    <dl className="resource-preview-numbers">
      <div><dt>{estimateLabel}</dt><dd>{estimateValue}</dd></div>
      <div><dt>{remainderLabel}</dt><dd>{remainderValue}</dd></div>
    </dl>
    {componentTarget && <dl className="resource-preview-references"><div><dt>{componentTarget}目标</dt><dd>{numeric(resource.budget_bytes) !== null ? formatBytes(resource.budget_bytes) : unknownValue}</dd></div></dl>}
  </div>
}
