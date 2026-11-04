import { useEffect, useState, type ReactNode } from 'react'
import { ArrowRight, FolderInput } from 'lucide-react'
import type { Job, ModelAsset, PageProps, RuntimePlatform } from './types'
import { apiFetch, errorMessage, isActive, isJobActive, request } from './api'
import { Button, ErrorNotice, Field, JobRow, KeyValue, Modal, formatBytes } from './components'
import { locationLabel, runtimeDistribution, runtimePlatform } from './platforms'
import { PlatformPathInput } from './WslPlatform'
import { useAction } from './hooks'
import './model-storage.css'

type StoragePreview = {
  id: string
  eligible: boolean
  errors: string[]
  warnings: string[]
  source_path: string
  target_path: string
  distribution: string
  target_platform: RuntimePlatform
  file_count: number
  total_bytes: number
  prepared_bytes?: number
  prepared_count?: number
  profiles: Array<{ id: string; name: string }>
  preserved_profiles: Array<{ id: string; name: string; reason: string }>
  delete_source_allowed: boolean
  delete_source_reasons: string[]
}

export default function ModelStorage({ model, onSelect, children, ...props }: PageProps & { model: ModelAsset; onSelect: (id: string) => void; children: ReactNode }) {
  const [open, setOpen] = useState(false)
  const job = [...(props.state.migrations || [])].reverse().find(item => item.request?.model_id === model.id || item.model_id === model.id)
  const native = model.location?.platform === 'wsl2'
  const available = ['gguf', 'safetensors', 'hf', 'ninfer'].includes(model.format.toLowerCase())
    && (native || props.state.runtimes.some(runtime => runtimePlatform(runtime, props.state.backends) === 'wsl2' && runtimeDistribution(runtime) === 'ModLoader'))
  const action = useAction(props.notify, props.refresh)
  if (!native && !available && !job) return children
  const active = !!job && isJobActive(job)
  return <section className="model-storage" aria-label="模型存储">
    <div className="model-storage-line">{children}<div className="model-storage-action">{job && ['failed', 'error', 'interrupted'].includes(job.state || '') && <span className="error-text">{job.state === 'interrupted' ? '迁移已中断' : '迁移失败'}</span>}<Button small icon={FolderInput} onClick={() => setOpen(true)}>存储管理</Button></div></div>
    {active && <JobRow job={job} busy={action.busy} onCancel={() => void action.run(() => request('POST', `/api/jobs/${job.id}/cancel`))} />}
    {open && <StorageDialog model={model} initialJob={job} available={available} onClose={() => setOpen(false)} onSelect={onSelect} {...props} />}
  </section>
}

function StorageDialog({ model, initialJob, available, onClose, onSelect, ...props }: PageProps & { model: ModelAsset; initialJob?: Job; available: boolean; onClose: () => void; onSelect: (id: string) => void }) {
  const [started, setStarted] = useState<Job | null>(initialJob || null)
  const [targetPlatform, setTargetPlatform] = useState<RuntimePlatform | null>(null)
  const [destination, setDestination] = useState(props.state.settings.model_directory || '')
  const [preview, setPreview] = useState<StoragePreview | null>(null)
  const [error, setError] = useState('')
  const [checking, setChecking] = useState(false)
  const [attempt, setAttempt] = useState(0)
  const [deleteSource, setDeleteSource] = useState(false)
  const action = useAction(props.notify, props.refresh)
  const job = started && (props.state.migrations?.find(item => item.id === started.id) || started)
  const active = !!job && isJobActive(job)
  const running = props.state.instances.some(instance => instance.model_id === model.id && isActive(instance.state))
  const native = model.location?.platform === 'wsl2'
  const target = job?.state === 'completed' && typeof job.model_id === 'string' && job.model_id !== model.id && props.state.models.some(item => item.id === job.model_id) ? job.model_id : undefined
  useEffect(() => {
    if (!targetPlatform || targetPlatform === 'windows' && !attempt) return
    const abort = new AbortController()
    setChecking(true); setError(''); setPreview(null)
    void apiFetch(`/api/models/${model.id}/storage-preview`, {
      method: 'POST', signal: abort.signal,
      body: JSON.stringify({ target_platform: targetPlatform, ...(targetPlatform === 'windows' ? { destination } : {}) }),
    }).then(response => response.json()).then((value: StoragePreview) => {
      if (!abort.signal.aborted) { setPreview(value); if (!value.delete_source_allowed) setDeleteSource(false) }
    }).catch(reason => { if (!abort.signal.aborted) setError(errorMessage(reason)) })
      .finally(() => { if (!abort.signal.aborted) setChecking(false) })
    return () => abort.abort()
  }, [model.id, targetPlatform, attempt, destination])
  function choose(platform: RuntimePlatform) {
    setTargetPlatform(platform); setAttempt(0); setPreview(null); setError(''); setDeleteSource(platform === 'windows')
  }
  function back() { setTargetPlatform(null); setPreview(null); setError(''); setChecking(false); setAttempt(0) }
  function changeDestination(value: string) { setDestination(value); setPreview(null); setError(''); setAttempt(0); setChecking(false) }
  async function start() {
    if (!preview) return
    const result = await action.run(() => request<Job>('POST', `/api/models/${model.id}/storage-migrate`, { plan_id: preview.id, delete_source: deleteSource }))
    if (result.ok) { setStarted(result.data); setTargetPlatform(null); setPreview(null) }
  }
  const canStart = !checking && preview?.eligible && !error && !running
  return <Modal title="存储管理" description={model.name} onClose={onClose} dismissible={!action.busy} footer={<>
    {targetPlatform && <Button disabled={action.busy} onClick={back}>返回</Button>}
    <Button disabled={action.busy} onClick={onClose}>关闭</Button>
    {targetPlatform && <Button variant="primary" icon={FolderInput} busy={action.busy} disabled={!canStart} onClick={() => void start()}>{deleteSource ? '移动' : '复制'}到 {targetPlatform === 'windows' ? 'Windows' : 'WSL'}</Button>}
  </>}>
    {targetPlatform ? <div className="migration-preview">
      {targetPlatform === 'windows' && <Field label="Windows 目标文件夹"><div className="storage-destination"><PlatformPathInput label="Windows 目标文件夹" kind="directory" value={destination} onChange={changeDestination} disabled={action.busy} placeholder="D:\Models" /><Button disabled={!destination.trim() || checking || action.busy} onClick={() => setAttempt(value => value + 1)}>检查目标</Button></div></Field>}
      {checking && <p role="status" className="muted">正在检查文件、关联配置和磁盘空间…</p>}
      {error && <ErrorNotice message={error} retry={() => setAttempt(value => value + 1)} />}
      {preview && <>
        {preview.errors.map(message => <ErrorNotice key={message} message={message} />)}
        <KeyValue label="当前位置" value={preview.source_path} mono />
        <KeyValue label={`目标 · ${targetPlatform === 'windows' ? 'Windows' : preview.distribution}`} value={preview.target_path} mono />
        <KeyValue label="文件大小" value={`${formatBytes(preview.total_bytes)} · ${preview.file_count} 个文件`} />
        {!!preview.prepared_count && <KeyValue label="其中运行文件" value={`${preview.prepared_count} 组 · ${formatBytes(preview.prepared_bytes)}`} />}
        {!!preview.profiles.length && <div className="migration-profiles"><h4>更新路径 · {preview.profiles.length} 套配置</h4><ul>{preview.profiles.map(profile => <li key={profile.id}>{profile.name}</li>)}</ul></div>}
        {!!preview.preserved_profiles.length && <details className="migration-preserved"><summary>保留原路径 · {preview.preserved_profiles.length} 套配置</summary><ul>{preview.preserved_profiles.map(profile => <li key={profile.id}><strong>{profile.name}</strong><span>{profile.reason}</span></li>)}</ul></details>}
        <label className="check-option danger-option"><input type="checkbox" checked={deleteSource} disabled={!preview.delete_source_allowed || !preview.eligible || action.busy} onChange={event => setDeleteSource(event.target.checked)} /><div><strong>完成后删除源文件</strong>{!preview.delete_source_allowed && <span>{preview.delete_source_reasons.join('；')}</span>}</div></label>
        {preview.warnings.length > 0 && <div className="warning-note">{preview.warnings.join('；')}</div>}
      </>}
    </div> : <div className="storage-details">
      <KeyValue label={locationLabel(model.location)} value={model.path} mono />
      {job?.state === 'completed' && typeof job.source_deleted === 'boolean' && <KeyValue label="上次操作的源文件" value={job.source_deleted ? '已删除' : '已保留'} />}
      {job && job.state !== 'completed' && <div className="migration-job"><JobRow job={job} busy={action.busy} onCancel={() => void action.run(() => request('POST', `/api/jobs/${job.id}/cancel`))} onRetry={() => choose(native ? 'windows' : 'wsl2')} />{active && <p role="status">{formatBytes(job.downloaded_bytes || 0)} / {formatBytes(job.total_bytes)}</p>}</div>}
      {job && Array.isArray(job.warnings) && job.warnings.length > 0 && <div className="warning-note">{job.warnings.join('；')}</div>}
      {!active && <div className="storage-options">
        {target && <Button icon={ArrowRight} onClick={() => { onSelect(target); onClose() }}>查看目标模型</Button>}
        {available && <Button icon={FolderInput} disabled={running} onClick={() => choose(native ? 'windows' : 'wsl2')}>{native ? '移到 Windows' : '迁移到 WSL'}</Button>}
      </div>}
      {running && !active && <p className="field-help">请先停止该模型，再移动文件。</p>}
    </div>}
  </Modal>
}
