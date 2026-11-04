import { useState } from 'react'
import { Trash2 } from 'lucide-react'
import type { ModelAsset, PageProps } from './types'
import { errorMessage, isActive, request } from './api'
import { Badge, Button, ErrorNotice, Modal } from './components'

export default function BulkRemoveModels({ models, onClose, onComplete, ...props }: PageProps & { models: ModelAsset[]; onClose: () => void; onComplete: () => void }) {
  const [busy, setBusy] = useState(false)
  const [removed, setRemoved] = useState<Set<string>>(new Set())
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [progress, setProgress] = useState('')
  const [submitted, setSubmitted] = useState(false)
  const [refreshError, setRefreshError] = useState('')
  const remaining = models.filter(model => !removed.has(model.id))
  const active = new Set(props.state.instances.filter(instance => isActive(instance.state)).map(instance => instance.model_id))
  const blocked = remaining.filter(model => active.has(model.id))
  const profileCount = props.state.profiles.filter(profile => remaining.some(model => model.id === profile.model_id)).length
  async function submit() {
    setBusy(true); setErrors({}); setRefreshError('')
    const completed = new Set(removed)
    const failures: Record<string, string> = {}
    try {
      for (const [index, model] of remaining.entries()) {
        setProgress(`${index + 1} / ${remaining.length}`)
        try {
          await request('DELETE', `/api/models/${model.id}`)
          completed.add(model.id)
        } catch (reason) { failures[model.id] = errorMessage(reason) }
      }
      setRemoved(completed); setErrors(failures); setSubmitted(true)
      try { await props.refresh() } catch (reason) { setRefreshError(`登记已处理，但刷新失败：${errorMessage(reason)}`) }
      if (Object.keys(failures).length === 0) onComplete()
    } finally { setBusy(false) }
  }
  return <Modal title="批量移除模型登记" description="移除所选模型及其加载配置，磁盘中的模型文件会保留。" onClose={onClose} dismissible={!busy}
    footer={<><Button disabled={busy} onClick={onClose}>{submitted ? '完成' : '取消'}</Button>{remaining.length > 0 && <Button variant="danger" icon={Trash2} busy={busy} disabled={blocked.length > 0} onClick={() => void submit()}>{busy ? `正在移除 ${progress}` : submitted ? '重试失败项' : '确认移除'}</Button>}</>}>
    <div className="bulk-remove-summary" aria-live="polite">{submitted ? <><Badge tone="green">已移除 {removed.size} 个</Badge>{Object.keys(errors).length > 0 && <Badge tone="red">失败 {Object.keys(errors).length} 个</Badge>}</> : <><Badge>{models.length} 个模型</Badge><Badge>{profileCount} 个加载配置</Badge></>}</div>
    {blocked.length > 0 && <div className="warning-note">有 {blocked.length} 个模型正在运行，请先停止，或返回取消勾选这些模型。</div>}
    {refreshError && <ErrorNotice message={refreshError} retry={() => void props.refresh().then(() => setRefreshError('')).catch(reason => setRefreshError(errorMessage(reason)))} />}
    <div className="bulk-remove-list">{models.map(model => {
      const profiles = props.state.profiles.filter(profile => profile.model_id === model.id)
      return <div className="bulk-remove-row" key={model.id}><div><strong>{model.name}</strong>{removed.has(model.id) ? <Badge tone="green">已移除</Badge> : active.has(model.id) ? <Badge tone="amber">需先停止</Badge> : null}</div><span className="mono">{model.path}</span>{!removed.has(model.id) && profiles.length > 0 && <span>同时移除配置：{profiles.map(profile => profile.name).join('、')}</span>}{errors[model.id] && <span className="error-text" role="alert">{errors[model.id]}</span>}</div>
    })}</div>
  </Modal>
}
