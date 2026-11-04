import { useImperativeHandle, useRef, useState, type Ref } from 'react'
import { Eye, EyeOff } from 'lucide-react'
import type { PageProps } from './types'
import { errorMessage, request } from './api'
import { Badge, Button, Modal } from './components'

type SaveState = 'idle' | 'saving' | 'saved' | 'error'
export type CredentialFieldHandle = { commit: () => Promise<boolean> }

export default function CredentialField({ source, label, configured, kind = 'Token', clearDescription = '私有或授权仓库将无法继续使用此凭据访问。', ref, disabled = false, ...props }: PageProps & { source: string; label: string; configured: boolean; kind?: 'Token' | 'API Key'; clearDescription?: string; ref?: Ref<CredentialFieldHandle>; disabled?: boolean }) {
  const [draft, setDraft] = useState('')
  const draftRef = useRef(draft)
  draftRef.current = draft
  const [visible, setVisible] = useState(false)
  const [status, setStatus] = useState<SaveState>('idle')
  const [error, setError] = useState('')
  const [clearOpen, setClearOpen] = useState(false)
  const saving = useRef<Promise<boolean> | null>(null)
  function save(value = draftRef.current, clear = false): Promise<boolean> {
    if (saving.current) return saving.current
    if (!value.trim() && !clear) return Promise.resolve(true)
    setStatus('saving'); setError('')
    const pending = (async () => {
      try {
        await request('PATCH', '/api/settings', { secrets: { [source]: value.trim() } })
        if (draftRef.current === value) { draftRef.current = ''; setDraft('') }
        setStatus('saved'); setClearOpen(false); await props.refresh()
        return true
      } catch (issue) { setStatus('error'); setError(errorMessage(issue)); return false }
    })()
    saving.current = pending
    void pending.finally(() => { if (saving.current === pending) saving.current = null })
    return pending
  }
  useImperativeHandle(ref, () => ({
    async commit() {
      if (saving.current && !await saving.current) return false
      return save()
    },
  }))
  return <div className="credential-field"><div className="credential-label"><strong>{label}</strong><Badge tone={configured ? 'green' : 'neutral'}>{configured ? '已配置' : '未配置'}</Badge></div><div className="credential-input" onBlur={event => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) void save() }}><input aria-label={label} disabled={disabled} type={visible ? 'text' : 'password'} autoComplete="off" value={draft} placeholder={configured ? `输入新 ${kind} 以替换` : kind === 'Token' ? '输入 Token（可选）' : '输入 API Key'} onChange={event => { setDraft(event.target.value); setStatus('idle') }} onKeyDown={event => { if (event.key === 'Enter') { event.preventDefault(); void save() } }} /><button data-visibility-toggle type="button" className="icon-button" aria-label={visible ? `隐藏 ${label}` : `显示 ${label}`} title={visible ? `隐藏 ${kind}` : `显示 ${kind}`}  onMouseDown={event => event.preventDefault()} onClick={() => setVisible(!visible)}>{visible ? <EyeOff size={16} /> : <Eye size={16} />}</button></div><div className="credential-footer"><span className={status === 'error' ? 'error-text' : ''}>{status === 'saving' ? '正在保存…' : status === 'saved' ? '已保存' : status === 'error' ? error : '离开输入框或按 Enter 保存'}</span>{status === 'error' && <Button small variant="ghost" onClick={() => void save()}>重试</Button>}{configured && <button disabled={disabled} onClick={() => setClearOpen(true)}>清除</button>}</div>{clearOpen && <Modal title={`清除 ${label}？`} onClose={() => setClearOpen(false)} footer={<><Button onClick={() => setClearOpen(false)}>取消</Button><Button variant="danger" busy={status === 'saving'} onClick={() => void save('', true)}>清除凭据</Button></>}><p className="muted">{clearDescription}</p></Modal>}</div>
}

